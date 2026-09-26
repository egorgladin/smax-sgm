"""Warm-start and compiled-kernel follow-up for the VFD Farrow experiment.

This script supplements ``run_experiment.py`` with:

* a Numba-compiled SMax-SGM loop with exact uniform sampling without
  replacement;
* a charged comparison of zero and WLS initialization;
* uniform, suffix, and warm-start-safe output rules;
* an exact restricted-SOCP exchange baseline solved by Clarabel.

The finite-max objective is always evaluated on the training grid when an
iterate is selected.  The independent validation grid is used only after the
algorithm has finished.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import clarabel
import matplotlib.pyplot as plt
import numba
import numpy as np
import scipy
from numba import njit
from scipy import sparse

import run_experiment as base


@dataclass
class CompiledTrajectory:
    sample_queries: np.ndarray
    elapsed_seconds: np.ndarray
    uniform_iterates: list[np.ndarray]
    suffix_iterates: list[np.ndarray]
    compilation_seconds: float


@dataclass
class SOCPExchangeResult:
    iterate: np.ndarray
    best_iterates: list[np.ndarray]
    upper_bound: float
    lower_bound: float
    value_queries: int
    elapsed_seconds: float
    active_constraints: int
    master_solves: int
    status: str
    history: list[dict[str, float | int | str]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frequency-count", type=int, default=1000)
    parser.add_argument("--delay-count", type=int, default=200)
    parser.add_argument("--validation-frequency-count", type=int, default=2001)
    parser.add_argument("--validation-delay-count", type=int, default=401)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--value-budget", type=int, default=4_000_000)
    parser.add_argument("--subset-size", type=int, default=1024)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--eta-scale", type=float, default=8.0)
    parser.add_argument("--history-points", type=int, default=16)
    parser.add_argument("--kernel-benchmark-budget", type=int, default=500_000)
    parser.add_argument(
        "--kernel-benchmark-subsets",
        type=int,
        nargs="+",
        default=[64, 256, 1024],
    )
    parser.add_argument("--socp-initial-top", type=int, default=128)
    parser.add_argument(
        "--socp-stratified-frequency-count",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--socp-stratified-delay-count",
        type=int,
        default=9,
    )
    parser.add_argument("--socp-batch-size", type=int, default=64)
    parser.add_argument("--socp-max-iterations", type=int, default=30)
    parser.add_argument("--socp-absolute-tolerance", type=float, default=1e-8)
    parser.add_argument("--socp-relative-tolerance", type=float, default=2e-4)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("results_followup"),
    )
    return parser.parse_args()


@njit(nogil=True)
def _splitmix64(state: np.uint64) -> tuple[np.uint64, np.uint64]:
    state = state + np.uint64(0x9E3779B97F4A7C15)
    value = state
    value = (value ^ (value >> np.uint64(30))) * np.uint64(
        0xBF58476D1CE4E5B9
    )
    value = (value ^ (value >> np.uint64(27))) * np.uint64(
        0x94D049BB133111EB
    )
    value = value ^ (value >> np.uint64(31))
    return state, value


@njit(nogil=True)
def _draw_without_replacement(
    state: np.uint64,
    markers: np.ndarray,
    epoch: int,
    sample: np.ndarray,
) -> np.uint64:
    """Draw an ordered uniform subset using rejection and epoch markers."""
    population = markers.size
    bound = np.uint64(population)
    # Reject the short prefix that would otherwise create modulo bias when
    # 2**64 is not divisible by the population size.
    threshold = (np.uint64(0) - bound) % bound
    accepted = 0
    while accepted < sample.size:
        state, random_bits = _splitmix64(state)
        if random_bits < threshold:
            continue
        candidate = int(random_bits % bound)
        if markers[candidate] != epoch:
            markers[candidate] = epoch
            sample[accepted] = candidate
            accepted += 1
    return state


@njit(nogil=True)
def _active_sampled_component(
    y: np.ndarray,
    sample: np.ndarray,
    delay_count: int,
    even_frequency: np.ndarray,
    odd_frequency: np.ndarray,
    even_powers: np.ndarray,
    odd_powers: np.ndarray,
    target_r: np.ndarray,
    target_q: np.ndarray,
) -> tuple[int, int, float, float]:
    best_norm_squared = -1.0
    best_frequency = 0
    best_delay = 0
    best_residual_r = 0.0
    best_residual_q = 0.0

    for sample_position in range(sample.size):
        flat_index = sample[sample_position]
        frequency_index = flat_index // delay_count
        delay_index = flat_index - frequency_index * delay_count

        residual_r = -target_r[frequency_index, delay_index]
        for power_index in range(3):
            power = even_powers[delay_index, power_index]
            coefficient_offset = 31 * power_index
            for harmonic_index in range(31):
                residual_r += (
                    power
                    * even_frequency[frequency_index, harmonic_index]
                    * y[coefficient_offset + harmonic_index]
                )

        residual_q = -target_q[frequency_index, delay_index]
        for power_index in range(2):
            power = odd_powers[delay_index, power_index]
            coefficient_offset = 93 + 30 * power_index
            for harmonic_index in range(30):
                residual_q += (
                    power
                    * odd_frequency[frequency_index, harmonic_index]
                    * y[coefficient_offset + harmonic_index]
                )

        norm_squared = residual_r * residual_r + residual_q * residual_q
        if norm_squared > best_norm_squared:
            best_norm_squared = norm_squared
            best_frequency = frequency_index
            best_delay = delay_index
            best_residual_r = residual_r
            best_residual_q = residual_q

    return (
        best_frequency,
        best_delay,
        best_residual_r,
        best_residual_q,
    )


@njit(nogil=True)
def _compiled_smax_segment(
    y: np.ndarray,
    uniform_average: np.ndarray,
    suffix_average: np.ndarray,
    markers: np.ndarray,
    sample: np.ndarray,
    state: np.uint64,
    start_iteration: int,
    end_iteration: int,
    suffix_burn_in: int,
    eta0: float,
    stepsize_power: float,
    stepsize_offset: float,
    coefficient_bound: float,
    even_frequency: np.ndarray,
    odd_frequency: np.ndarray,
    even_powers: np.ndarray,
    odd_powers: np.ndarray,
    target_r: np.ndarray,
    target_q: np.ndarray,
) -> np.uint64:
    """Advance the complete sequential SMax loop in Numba nopython mode."""
    delay_count = even_powers.shape[0]
    tiny = np.finfo(np.float64).tiny

    for iteration in range(start_iteration + 1, end_iteration + 1):
        step = eta0 / math.pow(iteration + stepsize_offset, stepsize_power)
        inverse_uniform_count = 1.0 / iteration
        for coordinate in range(y.size):
            uniform_average[coordinate] += (
                y[coordinate] - uniform_average[coordinate]
            ) * inverse_uniform_count

        if iteration > suffix_burn_in:
            inverse_suffix_count = 1.0 / (iteration - suffix_burn_in)
            for coordinate in range(y.size):
                suffix_average[coordinate] += (
                    y[coordinate] - suffix_average[coordinate]
                ) * inverse_suffix_count

        state = _draw_without_replacement(
            state,
            markers,
            iteration,
            sample,
        )
        (
            frequency_index,
            delay_index,
            residual_r,
            residual_q,
        ) = _active_sampled_component(
            y,
            sample,
            delay_count,
            even_frequency,
            odd_frequency,
            even_powers,
            odd_powers,
            target_r,
            target_q,
        )
        residual_norm = math.sqrt(
            residual_r * residual_r + residual_q * residual_q
        )
        if residual_norm <= tiny:
            continue

        real_scale = step * residual_r / residual_norm
        quadrature_scale = step * residual_q / residual_norm

        for power_index in range(3):
            power = even_powers[delay_index, power_index]
            coefficient_offset = 31 * power_index
            for harmonic_index in range(31):
                coordinate = coefficient_offset + harmonic_index
                updated = y[coordinate] - (
                    real_scale
                    * power
                    * even_frequency[frequency_index, harmonic_index]
                )
                if updated > coefficient_bound:
                    updated = coefficient_bound
                elif updated < -coefficient_bound:
                    updated = -coefficient_bound
                y[coordinate] = updated

        for power_index in range(2):
            power = odd_powers[delay_index, power_index]
            coefficient_offset = 93 + 30 * power_index
            for harmonic_index in range(30):
                coordinate = coefficient_offset + harmonic_index
                updated = y[coordinate] - (
                    quadrature_scale
                    * power
                    * odd_frequency[frequency_index, harmonic_index]
                )
                if updated > coefficient_bound:
                    updated = coefficient_bound
                elif updated < -coefficient_bound:
                    updated = -coefficient_bound
                y[coordinate] = updated

    return state


def warm_compiled_kernel(
    grid: base.VFDGrid,
    subset_size: int,
) -> float:
    """Compile the exact signature used below without charging a method."""
    y = np.zeros(base.DIMENSION, dtype=np.float64)
    markers = np.full(grid.size, -1, dtype=np.int64)
    sample = np.empty(subset_size, dtype=np.int64)
    start = time.perf_counter()
    _compiled_smax_segment(
        y,
        y.copy(),
        y.copy(),
        markers,
        sample,
        np.uint64(1),
        0,
        0,
        0,
        1.0,
        0.5,
        0.0,
        1.0,
        grid.even_frequency,
        grid.odd_frequency,
        grid.even_powers,
        grid.odd_powers,
        grid.target_r,
        grid.target_q,
    )
    return time.perf_counter() - start


def compiled_smax_trajectory(
    grid: base.VFDGrid,
    initial_y: np.ndarray,
    coefficient_bound: float,
    sample_budget: int,
    eta0: float,
    subset_size: int,
    seed: int,
    suffix_burn_in: int,
    checkpoint_iterations: np.ndarray,
    stepsize_power: float = 0.5,
    stepsize_offset: float = 0.0,
) -> CompiledTrajectory:
    if not 1 <= subset_size < grid.size:
        raise ValueError("Compiled SMax requires 1 <= subset_size < grid size.")
    iterations = sample_budget // subset_size
    checkpoints = np.unique(
        np.asarray(checkpoint_iterations, dtype=np.int64)
    )
    checkpoints = checkpoints[(checkpoints >= 1) & (checkpoints <= iterations)]
    if checkpoints.size == 0 or checkpoints[-1] != iterations:
        checkpoints = np.append(checkpoints, iterations)

    start = time.perf_counter()
    y = np.asarray(initial_y, dtype=np.float64).copy()
    uniform_average = y.copy()
    suffix_average = y.copy()
    markers = np.full(grid.size, -1, dtype=np.int64)
    sample = np.empty(subset_size, dtype=np.int64)
    state = np.uint64(seed + 1)
    uniform_iterates: list[np.ndarray] = []
    suffix_iterates: list[np.ndarray] = []
    elapsed: list[float] = []

    previous = 0
    for checkpoint in checkpoints:
        state = np.uint64(
            _compiled_smax_segment(
                y,
                uniform_average,
                suffix_average,
                markers,
                sample,
                state,
                previous,
                int(checkpoint),
                suffix_burn_in,
                eta0,
                stepsize_power,
                stepsize_offset,
                coefficient_bound,
                grid.even_frequency,
                grid.odd_frequency,
                grid.even_powers,
                grid.odd_powers,
                grid.target_r,
                grid.target_q,
            )
        )
        uniform_iterates.append(uniform_average.copy())
        if checkpoint > suffix_burn_in:
            suffix_iterates.append(suffix_average.copy())
        else:
            suffix_iterates.append(initial_y.copy())
        elapsed.append(time.perf_counter() - start)
        previous = int(checkpoint)

    return CompiledTrajectory(
        sample_queries=checkpoints * subset_size,
        elapsed_seconds=np.asarray(elapsed, dtype=np.float64),
        uniform_iterates=uniform_iterates,
        suffix_iterates=suffix_iterates,
        compilation_seconds=0.0,
    )


def kernel_self_checks(grid: base.VFDGrid, y: np.ndarray) -> dict[str, object]:
    """Check sampled responses, sampling uniqueness, and nopython compilation."""
    test_indices = np.array(
        [0, grid.delay.size + 3, grid.size // 2, grid.size - 1],
        dtype=np.int64,
    )
    (
        frequency_index,
        delay_index,
        residual_r,
        residual_q,
    ) = _active_sampled_component(
        y,
        test_indices,
        grid.delay.size,
        grid.even_frequency,
        grid.odd_frequency,
        grid.even_powers,
        grid.odd_powers,
        grid.target_r,
        grid.target_q,
    )
    vector_r, vector_q, vector_frequency, vector_delay = base.sampled_errors(
        y,
        grid,
        test_indices,
    )
    active = int(np.argmax(np.hypot(vector_r, vector_q)))
    response_error = max(
        abs(residual_r - float(vector_r[active])),
        abs(residual_q - float(vector_q[active])),
    )
    active_match = bool(
        frequency_index == int(vector_frequency[active])
        and delay_index == int(vector_delay[active])
    )

    markers = np.full(101, -1, dtype=np.int64)
    sample = np.empty(37, dtype=np.int64)
    state = _draw_without_replacement(
        np.uint64(123),
        markers,
        1,
        sample,
    )
    sampling_unique = bool(np.unique(sample).size == sample.size)
    sampling_in_range = bool(np.all((sample >= 0) & (sample < markers.size)))
    return {
        "sampled_response_max_absolute_error": float(response_error),
        "sampled_active_index_match": active_match,
        "sampler_unique": sampling_unique,
        "sampler_in_range": sampling_in_range,
        "sampler_state_after_test": int(state),
        "segment_nopython_signatures": [
            str(signature)
            for signature in _compiled_smax_segment.nopython_signatures
        ],
    }


def _component_features(
    grid: base.VFDGrid,
    flat_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    delay_count = grid.delay.size
    frequency_indices = flat_indices // delay_count
    delay_indices = flat_indices % delay_count
    even_features = (
        grid.even_powers[delay_indices, :, None]
        * grid.even_frequency[frequency_indices, None, :]
    ).reshape(flat_indices.size, -1)
    odd_features = (
        grid.odd_powers[delay_indices, :, None]
        * grid.odd_frequency[frequency_indices, None, :]
    ).reshape(flat_indices.size, -1)
    return (
        even_features,
        odd_features,
        grid.target_r[frequency_indices, delay_indices],
        grid.target_q[frequency_indices, delay_indices],
    )


def solve_restricted_socp(
    grid: base.VFDGrid,
    active_indices: np.ndarray,
    coefficient_bound: float,
) -> tuple[np.ndarray, float, float, object]:
    """Solve the exact norm epigraph on the active component set."""
    active_indices = np.asarray(active_indices, dtype=np.int64)
    even_features, odd_features, target_r, target_q = _component_features(
        grid,
        active_indices,
    )
    variable_count = base.DIMENSION + 1
    active_count = active_indices.size

    objective = np.zeros(variable_count, dtype=np.float64)
    objective[-1] = 1.0
    quadratic = sparse.csc_matrix((variable_count, variable_count))

    box_matrix = sparse.hstack(
        (
            sparse.vstack(
                (
                    sparse.eye(base.DIMENSION, format="csc"),
                    -sparse.eye(base.DIMENSION, format="csc"),
                ),
                format="csc",
            ),
            sparse.csc_matrix((2 * base.DIMENSION, 1)),
        ),
        format="csc",
    )
    box_rhs = np.full(
        2 * base.DIMENSION,
        coefficient_bound,
        dtype=np.float64,
    )

    cone_dense = np.zeros(
        (3 * active_count, variable_count),
        dtype=np.float64,
    )
    cone_rhs = np.zeros(3 * active_count, dtype=np.float64)
    cone_dense[0::3, -1] = -1.0
    cone_dense[1::3, :93] = even_features
    cone_dense[2::3, 93:153] = odd_features
    cone_rhs[1::3] = target_r
    cone_rhs[2::3] = target_q

    constraint_matrix = sparse.vstack(
        (box_matrix, sparse.csc_matrix(cone_dense)),
        format="csc",
    )
    constraint_rhs = np.concatenate((box_rhs, cone_rhs))
    cones: list[object] = [
        clarabel.NonnegativeConeT(2 * base.DIMENSION),
        *[clarabel.SecondOrderConeT(3) for _ in range(active_count)],
    ]
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.max_iter = 200
    settings.tol_gap_abs = 1e-9
    settings.tol_gap_rel = 1e-8
    settings.tol_feas = 1e-9

    solution = clarabel.DefaultSolver(
        quadratic,
        objective,
        constraint_matrix,
        constraint_rhs,
        cones,
        settings,
    ).solve()
    solver_status = str(solution.status)
    if solver_status not in {"Solved", "AlmostSolved"}:
        raise RuntimeError(f"Clarabel restricted master failed: {solution.status}")
    y = np.asarray(solution.x[:-1], dtype=np.float64)
    t = float(solution.x[-1])
    primal_dual_gap = abs(float(solution.obj_val) - float(solution.obj_val_dual))
    almost_solved_acceptable = (
        float(solution.r_prim) <= 2e-7
        and float(solution.r_dual) <= 2e-7
        and primal_dual_gap <= 2e-7
    )
    if solver_status == "AlmostSolved" and not almost_solved_acceptable:
        dual_value = float("-inf")
    else:
        dual_value = float(solution.obj_val_dual)
    box_violation = float(np.max(np.abs(y)) - coefficient_bound)
    if box_violation > 2e-7:
        raise RuntimeError(
            "Restricted SOCP returned a box-infeasible design: "
            f"violation={box_violation:.3e}"
        )
    active_r, active_q, _, _ = base.sampled_errors(
        y,
        grid,
        active_indices,
    )
    active_violation = float(np.max(np.hypot(active_r, active_q)) - t)
    if active_violation > 2e-7:
        raise RuntimeError(
            "Restricted SOCP returned an infeasible active design: "
            f"violation={active_violation:.3e}"
        )
    return y, t, dual_value, solution


def _stratified_indices(
    frequency_count: int,
    delay_count: int,
    selected_frequency_count: int,
    selected_delay_count: int,
) -> np.ndarray:
    frequency_indices = np.linspace(
        0,
        frequency_count - 1,
        selected_frequency_count,
        dtype=np.int64,
    )
    delay_indices = np.linspace(
        0,
        delay_count - 1,
        selected_delay_count,
        dtype=np.int64,
    )
    return np.unique(
        (
            frequency_indices[:, None] * delay_count
            + delay_indices[None, :]
        ).ravel()
    )


def restricted_socp_exchange(
    grid: base.VFDGrid,
    initial_y: np.ndarray,
    coefficient_bound: float,
    initial_top: int,
    stratified_frequency_count: int,
    stratified_delay_count: int,
    batch_size: int,
    max_iterations: int,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> SOCPExchangeResult:
    """Exchange exact active SOC constraints and form a numerical bracket."""
    start = time.perf_counter()
    initial_errors = base.error_array(initial_y, grid)
    value_queries = grid.size
    best_y = initial_y.copy()
    best_upper = float(np.max(initial_errors))
    lower_bound = 0.0

    active = np.zeros(grid.size, dtype=bool)
    top = base.top_flat_indices(initial_errors, initial_top)
    stratified = _stratified_indices(
        grid.omega.size,
        grid.delay.size,
        stratified_frequency_count,
        stratified_delay_count,
    )
    active[np.unique(np.concatenate((top, stratified)))] = True
    history: list[dict[str, float | int | str]] = []
    best_iterates: list[np.ndarray] = []
    status = "maximum iterations reached"

    for master_index in range(1, max_iterations + 1):
        active_indices = np.flatnonzero(active)
        candidate, master_t, dual_value, solution = solve_restricted_socp(
            grid,
            active_indices,
            coefficient_bound,
        )
        lower_bound = max(lower_bound, dual_value)
        errors = base.error_array(candidate, grid)
        value_queries += grid.size
        candidate_upper = float(np.max(errors))
        if candidate_upper < best_upper:
            best_upper = candidate_upper
            best_y = candidate.copy()
        best_iterates.append(best_y.copy())

        gap = max(0.0, best_upper - lower_bound)
        history.append(
            {
                "master_solve": master_index,
                "active_constraints": int(active_indices.size),
                "master_primal_value": master_t,
                "master_dual_value": dual_value,
                "numerical_dual_lower_bound": lower_bound,
                "candidate_full_grid_max": candidate_upper,
                "best_full_grid_max": best_upper,
                "numerical_gap": gap,
                "value_queries": value_queries,
                "elapsed_seconds": time.perf_counter() - start,
                "clarabel_iterations": int(solution.iterations),
                "clarabel_status": str(solution.status),
            }
        )
        tolerance = absolute_tolerance + relative_tolerance * best_upper
        if gap <= tolerance:
            status = (
                f"numerically converged in {master_index} masters; "
                f"lower={lower_bound:.8e}; upper={best_upper:.8e}; "
                f"gap={gap:.3e}"
            )
            break

        violations = errors.ravel() - master_t
        violations[active] = -np.inf
        violated_count = int(np.count_nonzero(violations > 1e-10))
        if violated_count == 0:
            status = (
                "no inactive violated SOC remained, but the numerical "
                f"certificate gap was {gap:.3e}"
            )
            break
        selected = base.top_flat_indices(
            violations.reshape(errors.shape),
            min(batch_size, violated_count),
        )
        selected = selected[violations[selected] > 1e-10]
        if selected.size == 0:
            status = "separation failed to return a violated inactive component"
            break
        active[selected] = True

    elapsed = time.perf_counter() - start
    return SOCPExchangeResult(
        iterate=best_y,
        best_iterates=best_iterates,
        upper_bound=best_upper,
        lower_bound=lower_bound,
        value_queries=value_queries,
        elapsed_seconds=elapsed,
        active_constraints=int(np.count_nonzero(active)),
        master_solves=len(history),
        status=status,
        history=history,
    )


def _trajectory_checkpoints(
    iterations: int,
    safe_iteration: int,
    points: int,
) -> np.ndarray:
    checkpoints = base.checkpoint_iterations(iterations, points)
    checkpoints = np.unique(
        np.concatenate(
            (
                checkpoints,
                np.array([safe_iteration, iterations], dtype=np.int64),
            )
        )
    )
    return checkpoints[(checkpoints >= 1) & (checkpoints <= iterations)]


def _evaluate_iterates(
    iterates: list[np.ndarray],
    grid: base.VFDGrid,
) -> np.ndarray:
    return np.asarray(
        [base.max_error(iterate, grid) for iterate in iterates],
        dtype=np.float64,
    )


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_initialization(
    path: Path,
    rows: list[dict[str, object]],
    x_column: str,
    x_label: str,
    wls_value: float,
) -> None:
    fig, axis = plt.subplots(figsize=(7.0, 4.4))
    styles = {
        "Zero + uniform average": ("#CC79A7", "--"),
        "WLS + uniform average": ("#0072B2", "-"),
        "WLS + suffix average": ("#009E73", "-."),
    }
    for method, (color, line_style) in styles.items():
        method_rows = [row for row in rows if row["method"] == method]
        seeds = sorted({int(row["seed"]) for row in method_rows})
        if not seeds:
            continue
        series = [
            sorted(
                (
                    row
                    for row in method_rows
                    if int(row["seed"]) == seed
                ),
                key=lambda row: int(row["sample_queries"]),
            )
            for seed in seeds
        ]
        common_length = min(len(seed_rows) for seed_rows in series)
        x_matrix = np.asarray(
            [
                [float(row[x_column]) for row in seed_rows[:common_length]]
                for seed_rows in series
            ]
        )
        value_matrix = np.asarray(
            [
                [
                    float(row["validation_max_error"])
                    for row in seed_rows[:common_length]
                ]
                for seed_rows in series
            ]
        )
        x_values = np.median(x_matrix, axis=0)
        median = np.median(value_matrix, axis=0)
        lower = np.quantile(value_matrix, 0.25, axis=0)
        upper = np.quantile(value_matrix, 0.75, axis=0)
        axis.plot(
            x_values,
            median,
            color=color,
            linestyle=line_style,
            linewidth=2.0,
            label=method,
        )
        axis.fill_between(x_values, lower, upper, color=color, alpha=0.16)

    safe_rows = [row for row in rows if row["method"] == "WLS-safe selection"]
    if safe_rows:
        safe_x = float(np.median([float(row[x_column]) for row in safe_rows]))
        safe_y = float(
            np.median(
                [float(row["validation_max_error"]) for row in safe_rows]
            )
        )
        axis.scatter(
            [safe_x],
            [safe_y],
            color="#D55E00",
            marker="*",
            s=150,
            zorder=5,
            label="WLS-safe final output",
        )

    axis.axhline(
        wls_value,
        color="#E69F00",
        linewidth=1.6,
        linestyle=":",
        label="WLS error level",
    )
    axis.set_xlabel(x_label)
    axis.set_ylabel("independent-grid maximum error")
    axis.set_yscale("log")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_socp_certificate(
    path: Path,
    history: list[dict[str, float | int | str]],
) -> None:
    queries = np.asarray(
        [int(row["value_queries"]) for row in history],
        dtype=np.int64,
    )
    lower = np.asarray(
        [float(row["numerical_dual_lower_bound"]) for row in history],
        dtype=np.float64,
    )
    upper = np.asarray(
        [float(row["best_full_grid_max"]) for row in history],
        dtype=np.float64,
    )
    fig, axis = plt.subplots(figsize=(6.8, 4.1))
    axis.plot(
        queries,
        upper,
        color="#D55E00",
        linewidth=2.0,
        marker="o",
        markersize=3.5,
        label="best feasible upper bound",
    )
    axis.plot(
        queries,
        lower,
        color="#0072B2",
        linewidth=2.0,
        marker="s",
        markersize=3.0,
        label="restricted-master dual lower bound",
    )
    axis.fill_between(queries, lower, upper, color="#999999", alpha=0.18)
    axis.set_xlabel("component values used in full-grid separation")
    axis.set_ylabel("training-grid maximum error")
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _aggregate_rows(
    final_rows: list[dict[str, object]],
    wls_validation_error: float,
) -> list[dict[str, object]]:
    methods = list(dict.fromkeys(str(row["method"]) for row in final_rows))
    aggregate: list[dict[str, object]] = []
    for method in methods:
        method_rows = [row for row in final_rows if row["method"] == method]
        validation = np.asarray(
            [float(row["validation_max_error"]) for row in method_rows]
        )
        training = np.asarray(
            [float(row["training_max_error"]) for row in method_rows]
        )
        runtime = np.asarray(
            [float(row["runtime_seconds"]) for row in method_rows]
        )
        aggregate.append(
            {
                "method": method,
                "runs": len(method_rows),
                "validation_q25": float(np.quantile(validation, 0.25)),
                "validation_median": float(np.median(validation)),
                "validation_q75": float(np.quantile(validation, 0.75)),
                "training_median": float(np.median(training)),
                "runtime_median_seconds": float(np.median(runtime)),
                "total_value_queries_median": int(
                    np.median(
                        [
                            int(row["total_value_queries"])
                            for row in method_rows
                        ]
                    )
                ),
                "runs_improving_on_wls_validation": int(
                    np.count_nonzero(validation < wls_validation_error)
                ),
            }
        )
    return aggregate


def _kernel_benchmark(
    grid: base.VFDGrid,
    wls_y: np.ndarray,
    coefficient_bound: float,
    eta0: float,
    budget: int,
    subset_sizes: list[int],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for subset_size in subset_sizes:
        if subset_size >= grid.size:
            continue
        iterations = budget // subset_size
        checkpoints = np.asarray([iterations], dtype=np.int64)

        python_run = base.projected_subgradient(
            grid,
            wls_y,
            coefficient_bound,
            budget,
            eta0,
            2,
            subset_size,
            123,
            "Python SMax",
        )
        compiled = compiled_smax_trajectory(
            grid,
            wls_y,
            coefficient_bound,
            budget,
            eta0,
            subset_size,
            123,
            math.ceil(grid.size / subset_size),
            checkpoints,
        )
        compiled_runtime = float(compiled.elapsed_seconds[-1])
        rows.append(
            {
                "subset_size": subset_size,
                "sample_queries": iterations * subset_size,
                "python_runtime_seconds": python_run.runtime_seconds,
                "numba_runtime_seconds": compiled_runtime,
                "warm_speedup": python_run.runtime_seconds / compiled_runtime,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    training_grid = base.make_grid(
        args.frequency_count,
        args.delay_count,
        args.alpha,
    )
    validation_grid = base.make_grid(
        args.validation_frequency_count,
        args.validation_delay_count,
        args.alpha,
    )
    if args.subset_size >= training_grid.size:
        raise ValueError("The follow-up requires subset_size < training grid size.")

    print("Computing the shared WLS design...", flush=True)
    wls_start = time.perf_counter()
    wls_y = base.wls_solution(training_grid)
    wls_training_error = base.max_error(wls_y, training_grid)
    wls_runtime = time.perf_counter() - wls_start
    wls_validation_error = base.max_error(wls_y, validation_grid)
    coefficient_bound = max(1.0, 2.0 * float(np.max(np.abs(wls_y))))
    feature_norm = base.max_feature_norm(training_grid)
    eta0 = args.eta_scale * wls_training_error / feature_norm
    print(
        f"WLS train={wls_training_error:.8e}, validation={wls_validation_error:.8e}, "
        f"time={wls_runtime:.4f}s, eta0={eta0:.4e}",
        flush=True,
    )

    print("JIT-compiling the full sequential SMax loop...", flush=True)
    compilation_seconds = warm_compiled_kernel(
        training_grid,
        max(args.kernel_benchmark_subsets + [args.subset_size]),
    )
    print(f"Numba compilation time={compilation_seconds:.3f}s", flush=True)
    checks = kernel_self_checks(training_grid, wls_y)
    if (
        not checks["sampled_active_index_match"]
        or not checks["sampler_unique"]
        or not checks["sampler_in_range"]
        or float(checks["sampled_response_max_absolute_error"]) > 1e-10
    ):
        raise RuntimeError(f"Compiled-kernel self-check failed: {checks}")

    print("Benchmarking Python and compiled SMax kernels...", flush=True)
    benchmark_rows = _kernel_benchmark(
        training_grid,
        wls_y,
        coefficient_bound,
        eta0,
        args.kernel_benchmark_budget,
        args.kernel_benchmark_subsets,
    )
    _write_rows(args.output_dir / "kernel_benchmark.csv", benchmark_rows)
    for row in benchmark_rows:
        print(
            f"  m={row['subset_size']}: "
            f"{float(row['warm_speedup']):.2f}x warm speedup",
            flush=True,
        )

    total_iterations_zero = args.value_budget // args.subset_size
    total_iterations_wls = (
        args.value_budget - training_grid.size
    ) // args.subset_size
    safe_iterations = (
        args.value_budget - 3 * training_grid.size
    ) // args.subset_size
    if safe_iterations <= 0:
        raise ValueError(
            "value_budget must cover WLS and two safe-selection grid scans."
        )
    suffix_burn_in = math.ceil(training_grid.size / args.subset_size)
    zero_checkpoints = _trajectory_checkpoints(
        total_iterations_zero,
        total_iterations_zero,
        args.history_points,
    )
    wls_checkpoints = _trajectory_checkpoints(
        total_iterations_wls,
        safe_iterations,
        args.history_points,
    )
    safe_checkpoint_position = int(
        np.flatnonzero(wls_checkpoints == safe_iterations)[0]
    )

    history_rows: list[dict[str, object]] = []
    final_rows: list[dict[str, object]] = []
    safe_choices: dict[str, int] = {"WLS": 0, "uniform": 0, "suffix": 0}
    coefficients: dict[str, np.ndarray] = {"wls": wls_y}

    for seed in range(args.seeds):
        print(
            f"Compiled initialization ablation, seed {seed + 1}/{args.seeds}...",
            flush=True,
        )
        zero_trajectory = compiled_smax_trajectory(
            training_grid,
            np.zeros_like(wls_y),
            coefficient_bound,
            args.value_budget,
            eta0,
            args.subset_size,
            seed,
            suffix_burn_in,
            zero_checkpoints,
        )
        wls_trajectory = compiled_smax_trajectory(
            training_grid,
            wls_y,
            coefficient_bound,
            args.value_budget - training_grid.size,
            eta0,
            args.subset_size,
            seed,
            suffix_burn_in,
            wls_checkpoints,
        )

        variants = [
            (
                "Zero + uniform average",
                zero_trajectory,
                zero_trajectory.uniform_iterates,
                0,
                0.0,
            ),
            (
                "WLS + uniform average",
                wls_trajectory,
                wls_trajectory.uniform_iterates,
                training_grid.size,
                wls_runtime,
            ),
            (
                "WLS + suffix average",
                wls_trajectory,
                wls_trajectory.suffix_iterates,
                training_grid.size,
                wls_runtime,
            ),
        ]
        for (
            method,
            trajectory,
            iterates,
            query_offset,
            time_offset,
        ) in variants:
            validation_values = _evaluate_iterates(iterates, validation_grid)
            for position, validation_value in enumerate(validation_values):
                history_rows.append(
                    {
                        "method": method,
                        "seed": seed,
                        "sample_queries": int(
                            trajectory.sample_queries[position]
                        ),
                        "total_value_queries": int(
                            query_offset + trajectory.sample_queries[position]
                        ),
                        "runtime_seconds": float(
                            time_offset + trajectory.elapsed_seconds[position]
                        ),
                        "validation_max_error": float(validation_value),
                    }
                )
            final_rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "subset_size": args.subset_size,
                    "sample_queries": int(trajectory.sample_queries[-1]),
                    "total_value_queries": int(
                        query_offset + trajectory.sample_queries[-1]
                    ),
                    "runtime_seconds": float(
                        time_offset + trajectory.elapsed_seconds[-1]
                    ),
                    "training_max_error": base.max_error(
                        iterates[-1],
                        training_grid,
                    ),
                    "validation_max_error": float(validation_values[-1]),
                    "selection": "none",
                }
            )

        safe_uniform = wls_trajectory.uniform_iterates[safe_checkpoint_position]
        safe_suffix = wls_trajectory.suffix_iterates[safe_checkpoint_position]
        selection_start = time.perf_counter()
        uniform_training = base.max_error(safe_uniform, training_grid)
        suffix_training = base.max_error(safe_suffix, training_grid)
        selection_runtime = time.perf_counter() - selection_start
        candidate_names = ["WLS", "uniform", "suffix"]
        candidate_iterates = [wls_y, safe_uniform, safe_suffix]
        candidate_training = [
            wls_training_error,
            uniform_training,
            suffix_training,
        ]
        choice_position = int(np.argmin(candidate_training))
        safe_choice = candidate_names[choice_position]
        safe_choices[safe_choice] += 1
        safe_y = candidate_iterates[choice_position]
        safe_validation = base.max_error(safe_y, validation_grid)
        safe_sample_queries = safe_iterations * args.subset_size
        safe_total_queries = (
            training_grid.size
            + safe_sample_queries
            + 2 * training_grid.size
        )
        safe_runtime = (
            wls_runtime
            + float(wls_trajectory.elapsed_seconds[safe_checkpoint_position])
            + selection_runtime
        )
        safe_row = {
            "method": "WLS-safe selection",
            "seed": seed,
            "subset_size": args.subset_size,
            "sample_queries": safe_sample_queries,
            "total_value_queries": safe_total_queries,
            "runtime_seconds": safe_runtime,
            "training_max_error": candidate_training[choice_position],
            "validation_max_error": safe_validation,
            "selection": safe_choice,
        }
        final_rows.append(safe_row)
        coefficients[f"safe_seed{seed}"] = safe_y
        history_rows.append(
            {
                "method": "WLS-safe selection",
                "seed": seed,
                "sample_queries": safe_sample_queries,
                "total_value_queries": safe_total_queries,
                "runtime_seconds": safe_runtime,
                "validation_max_error": safe_validation,
            }
        )

    print("Running exact restricted-SOCP exchange...", flush=True)
    socp = restricted_socp_exchange(
        training_grid,
        wls_y,
        coefficient_bound,
        args.socp_initial_top,
        args.socp_stratified_frequency_count,
        args.socp_stratified_delay_count,
        args.socp_batch_size,
        args.socp_max_iterations,
        args.socp_absolute_tolerance,
        args.socp_relative_tolerance,
    )
    socp_validation = base.max_error(socp.iterate, validation_grid)
    coefficients["restricted_socp_exchange"] = socp.iterate
    final_rows.append(
        {
            "method": "Restricted-SOCP exchange",
            "seed": "",
            "subset_size": "",
            "sample_queries": 0,
            "total_value_queries": socp.value_queries,
            "runtime_seconds": wls_runtime + socp.elapsed_seconds,
            "training_max_error": socp.upper_bound,
            "validation_max_error": socp_validation,
            "selection": socp.status,
        }
    )
    final_rows.append(
        {
            "method": "WLS",
            "seed": "",
            "subset_size": "",
            "sample_queries": 0,
            "total_value_queries": training_grid.size,
            "runtime_seconds": wls_runtime,
            "training_max_error": wls_training_error,
            "validation_max_error": wls_validation_error,
            "selection": "closed-form tensor least squares",
        }
    )

    _write_rows(args.output_dir / "initialization_history.csv", history_rows)
    _write_rows(args.output_dir / "final_summary.csv", final_rows)
    _write_rows(
        args.output_dir / "aggregate_summary.csv",
        _aggregate_rows(final_rows, wls_validation_error),
    )
    _write_rows(args.output_dir / "socp_history.csv", socp.history)
    np.savez_compressed(args.output_dir / "coefficients.npz", **coefficients)
    _plot_initialization(
        args.output_dir / "initialization_vs_value_queries.png",
        history_rows,
        "total_value_queries",
        "total component-value evaluations",
        wls_validation_error,
    )
    _plot_socp_certificate(
        args.output_dir / "socp_certificate.png",
        socp.history,
    )
    _plot_initialization(
        args.output_dir / "initialization_vs_runtime.png",
        history_rows,
        "runtime_seconds",
        "algorithm runtime (seconds)",
        wls_validation_error,
    )

    config = {
        "source_setup": {
            "reference": "Zhao and Tay (2023), Example 1",
            "doi": "10.1007/s00034-022-02169-2",
            "filter_order_N": 60,
            "farrow_degree_M": 4,
            "independent_real_coefficients": base.DIMENSION,
        },
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "training_components": training_grid.size,
        "validation_components": validation_grid.size,
        "wls_training_max_error": wls_training_error,
        "wls_validation_max_error": wls_validation_error,
        "wls_runtime_seconds": wls_runtime,
        "coefficient_bound": coefficient_bound,
        "eta0": eta0,
        "eta_rule": (
            "fixed eta0/sqrt(t); eta_scale=8 frozen from the earlier "
            "one-seed development pilot; no new pilot is run"
        ),
        "historical_stepsize_pilot": {
            "candidate_count": 4,
            "sampled_values_per_candidate": 500000,
            "training_grid_scoring_values_per_candidate": training_grid.size,
            "total_component_values": 4 * (500000 + training_grid.size),
            "charged_to_each_followup_run": False,
        },
        "suffix_burn_in_iterations": suffix_burn_in,
        "safe_selection": {
            "rule": (
                "minimum training maximum among WLS, uniform average, "
                "and suffix average"
            ),
            "candidate_full_grid_scans_charged": 2,
            "selection_counts": safe_choices,
        },
        "compiled_kernel": {
            "numba_compilation_seconds_excluded_from_method_timings": (
                compilation_seconds
            ),
            "checks": checks,
            "benchmark": benchmark_rows,
        },
        "restricted_socp_exchange": {
            "training_upper_bound": socp.upper_bound,
            "numerical_dual_lower_bound": socp.lower_bound,
            "numerical_gap": socp.upper_bound - socp.lower_bound,
            "validation_max_error": socp_validation,
            "max_abs_scaled_coefficient": float(
                np.max(np.abs(socp.iterate))
            ),
            "active_constraints": socp.active_constraints,
            "master_solves": socp.master_solves,
            "status": socp.status,
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "numba": numba.__version__,
            "clarabel": clarabel.__version__,
            "matplotlib": plt.matplotlib.__version__,
        },
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
    }
    with (args.output_dir / "run_config.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(config, handle, indent=2)

    print(
        f"SOCP exchange: train={socp.upper_bound:.8e}, "
        f"validation={socp_validation:.8e}, "
        f"gap={socp.upper_bound - socp.lower_bound:.3e}, "
        f"time={socp.elapsed_seconds:.2f}s",
        flush=True,
    )
    print(f"Saved follow-up outputs to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
