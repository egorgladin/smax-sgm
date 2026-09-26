"""Initial variable-fractional-delay FIR finite-max benchmark.

The model follows Example 1 of Zhao and Tay (2023): an order-60
algebraic-polynomial Farrow filter with polynomial degree four.  The response
is optimized over a frequency-by-delay grid.  Symmetry leaves 153 independent
real coefficients, and each component is the Euclidean norm of a two-
dimensional affine residual.

The implementation never materializes a d-by-153 feature matrix.  Full-grid
methods use the tensor structure, while SMax generates only sampled rows.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.optimize import linprog, minimize
from scipy.special import logsumexp


EVEN_SHAPE = (3, 31)
ODD_SHAPE = (2, 30)
DIMENSION = int(np.prod(EVEN_SHAPE) + np.prod(ODD_SHAPE))


@dataclass(frozen=True)
class VFDGrid:
    omega: np.ndarray
    delay: np.ndarray
    even_frequency: np.ndarray
    odd_frequency: np.ndarray
    even_powers: np.ndarray
    odd_powers: np.ndarray
    target_r: np.ndarray
    target_q: np.ndarray

    @property
    def size(self) -> int:
        return int(self.omega.size * self.delay.size)


@dataclass
class Run:
    method: str
    seed: int | None
    subset_size: int | None
    value_queries: np.ndarray
    component_gradient_rows: np.ndarray
    elapsed_seconds: np.ndarray
    iterates: list[np.ndarray]
    runtime_seconds: float
    status: str = "ok"
    lower_bound: float | None = None


class EvaluationBudgetReached(RuntimeError):
    """Internal signal used to stop LSE at a full-grid evaluation cap."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frequency-count", type=int, default=250)
    parser.add_argument("--delay-count", type=int, default=50)
    parser.add_argument("--validation-frequency-count", type=int, default=1001)
    parser.add_argument("--validation-delay-count", type=int, default=201)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--value-budget", type=int, default=1_000_000)
    parser.add_argument("--pilot-budget", type=int, default=250_000)
    parser.add_argument(
        "--eta-scales", type=float, nargs="+", default=[1.0, 4.0, 8.0, 16.0]
    )
    parser.add_argument("--subset-sizes", type=int, nargs="+", default=[64, 256, 1024])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--history-points", type=int, default=50)
    parser.add_argument("--exchange-batch-size", type=int, default=128)
    parser.add_argument("--exchange-initial-cuts", type=int, default=256)
    parser.add_argument("--exchange-max-iterations", type=int, default=10)
    parser.add_argument("--exchange-absolute-tolerance", type=float, default=1e-8)
    parser.add_argument("--exchange-relative-tolerance", type=float, default=2e-4)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("results"),
    )
    return parser.parse_args()


def make_grid(
    frequency_count: int,
    delay_count: int,
    alpha: float,
) -> VFDGrid:
    if frequency_count < 2 or delay_count < 2:
        raise ValueError("Both tensor-grid dimensions must be at least two.")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one.")

    omega = np.linspace(0.0, alpha * np.pi, frequency_count, dtype=np.float64)
    delay = np.linspace(0.0, 0.5, delay_count, dtype=np.float64)

    harmonics_even = np.arange(31, dtype=np.float64)
    even_frequency = np.cos(omega[:, None] * harmonics_even[None, :])
    even_frequency[:, 1:] *= 2.0

    harmonics_odd = np.arange(1, 31, dtype=np.float64)
    odd_frequency = 2.0 * np.sin(omega[:, None] * harmonics_odd[None, :])

    # u=2p is a diagonal reparameterization of the same Farrow polynomial
    # space.  It avoids tiny p^4 coordinates on p in [0, 0.5].
    normalized_delay = 2.0 * delay
    even_powers = normalized_delay[:, None] ** np.array([0, 2, 4])
    odd_powers = normalized_delay[:, None] ** np.array([1, 3])

    phase = omega[:, None] * delay[None, :]
    target_r = np.cos(phase)
    # We represent the complex response as H=R-iQ.  Hence Q targets
    # +sin(omega*p), even though Im(exp(-i*omega*p)) is negative.
    target_q = np.sin(phase)
    return VFDGrid(
        omega=omega,
        delay=delay,
        even_frequency=even_frequency,
        odd_frequency=odd_frequency,
        even_powers=even_powers,
        odd_powers=odd_powers,
        target_r=target_r,
        target_q=target_q,
    )


def pack(even: np.ndarray, odd: np.ndarray) -> np.ndarray:
    return np.concatenate((even.ravel(), odd.ravel()))


def unpack(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    split = int(np.prod(EVEN_SHAPE))
    return y[:split].reshape(EVEN_SHAPE), y[split:].reshape(ODD_SHAPE)


def residual_arrays(
    y: np.ndarray,
    grid: VFDGrid,
) -> tuple[np.ndarray, np.ndarray]:
    """Return residual coordinates for H=R-iQ and D=cos-i sin."""
    even, odd = unpack(y)
    response_r = (grid.even_frequency @ even.T) @ grid.even_powers.T
    response_q = (grid.odd_frequency @ odd.T) @ grid.odd_powers.T
    return response_r - grid.target_r, response_q - grid.target_q


def error_array(y: np.ndarray, grid: VFDGrid) -> np.ndarray:
    residual_r, residual_q = residual_arrays(y, grid)
    return np.hypot(residual_r, residual_q)


def max_error(y: np.ndarray, grid: VFDGrid) -> float:
    return float(np.max(error_array(y, grid)))


def max_feature_norm(grid: VFDGrid) -> float:
    even_frequency_norm = np.sum(grid.even_frequency**2, axis=1)
    odd_frequency_norm = np.sum(grid.odd_frequency**2, axis=1)
    even_power_norm = np.sum(grid.even_powers**2, axis=1)
    odd_power_norm = np.sum(grid.odd_powers**2, axis=1)
    squared = (
        even_frequency_norm[:, None] * even_power_norm[None, :]
        + odd_frequency_norm[:, None] * odd_power_norm[None, :]
    )
    return float(np.sqrt(np.max(squared)))


def wls_solution(grid: VFDGrid) -> np.ndarray:
    """Uniform tensor-grid least-squares solution in reduced coordinates."""
    even = (
        np.linalg.pinv(grid.even_powers)
        @ grid.target_r.T
        @ np.linalg.pinv(grid.even_frequency.T)
    )
    odd = (
        np.linalg.pinv(grid.odd_powers)
        @ grid.target_q.T
        @ np.linalg.pinv(grid.odd_frequency.T)
    )
    return pack(even, odd)


def checkpoint_iterations(iterations: int, points: int) -> np.ndarray:
    if iterations <= 0:
        return np.empty(0, dtype=np.int64)
    raw = np.geomspace(1, iterations, min(points, iterations))
    return np.unique(np.rint(raw).astype(np.int64))


def sampled_errors(
    y: np.ndarray,
    grid: VFDGrid,
    flat_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    delay_count = grid.delay.size
    frequency_indices = flat_indices // delay_count
    delay_indices = flat_indices % delay_count
    even, odd = unpack(y)
    response_r = np.sum(
        (grid.even_frequency[frequency_indices] @ even.T)
        * grid.even_powers[delay_indices],
        axis=1,
    )
    response_q = np.sum(
        (grid.odd_frequency[frequency_indices] @ odd.T)
        * grid.odd_powers[delay_indices],
        axis=1,
    )
    residual_r = response_r - grid.target_r[frequency_indices, delay_indices]
    residual_q = response_q - grid.target_q[frequency_indices, delay_indices]
    return residual_r, residual_q, frequency_indices, delay_indices


def active_subgradient(
    residual_r: float,
    residual_q: float,
    frequency_index: int,
    delay_index: int,
    grid: VFDGrid,
) -> np.ndarray:
    norm = math.hypot(residual_r, residual_q)
    if norm <= np.finfo(np.float64).tiny:
        return np.zeros(DIMENSION, dtype=np.float64)
    even_gradient = (
        (residual_r / norm)
        * grid.even_powers[delay_index, :, None]
        * grid.even_frequency[frequency_index, None, :]
    )
    odd_gradient = (
        (residual_q / norm)
        * grid.odd_powers[delay_index, :, None]
        * grid.odd_frequency[frequency_index, None, :]
    )
    return pack(even_gradient, odd_gradient)


def projected_subgradient(
    grid: VFDGrid,
    initial_y: np.ndarray,
    coefficient_bound: float,
    value_budget: int,
    eta0: float,
    history_points: int,
    subset_size: int,
    seed: int | None,
    label: str,
) -> Run:
    """Averaged projected SGM; subset_size=d gives the full-grid control."""
    if not 1 <= subset_size <= grid.size:
        raise ValueError("subset_size must be between one and the grid size.")
    iterations = value_budget // subset_size
    checkpoints = checkpoint_iterations(iterations, history_points)
    checkpoint_set = set(checkpoints.tolist())
    rng = np.random.default_rng(seed)
    y = initial_y.copy()
    y_average = np.zeros_like(y)
    iterates: list[np.ndarray] = []
    queries: list[int] = []
    gradient_rows: list[int] = []
    elapsed: list[float] = []

    start = time.perf_counter()
    for iteration in range(1, iterations + 1):
        # Average the point at which this iteration's subgradient is evaluated.
        y_average += (y - y_average) / iteration

        if subset_size == grid.size:
            residual_r, residual_q = residual_arrays(y, grid)
            errors = np.hypot(residual_r, residual_q)
            flat_active = int(np.argmax(errors))
            frequency_index, delay_index = np.unravel_index(
                flat_active, errors.shape
            )
            active_r = float(residual_r[frequency_index, delay_index])
            active_q = float(residual_q[frequency_index, delay_index])
        else:
            indices = rng.choice(
                grid.size, size=subset_size, replace=False, shuffle=False
            )
            residual_r, residual_q, frequency_indices, delay_indices = sampled_errors(
                y, grid, indices
            )
            local_active = int(np.argmax(np.hypot(residual_r, residual_q)))
            frequency_index = int(frequency_indices[local_active])
            delay_index = int(delay_indices[local_active])
            active_r = float(residual_r[local_active])
            active_q = float(residual_q[local_active])

        gradient = active_subgradient(
            active_r, active_q, frequency_index, delay_index, grid
        )
        step = eta0 / math.sqrt(iteration)
        y = np.clip(y - step * gradient, -coefficient_bound, coefficient_bound)

        if iteration in checkpoint_set:
            iterates.append(y_average.copy())
            queries.append(iteration * subset_size)
            gradient_rows.append(iteration)
            elapsed.append(time.perf_counter() - start)

    runtime = time.perf_counter() - start
    return Run(
        method=label,
        seed=seed,
        subset_size=subset_size,
        value_queries=np.asarray(queries, dtype=np.int64),
        component_gradient_rows=np.asarray(gradient_rows, dtype=np.int64),
        elapsed_seconds=np.asarray(elapsed, dtype=np.float64),
        iterates=iterates,
        runtime_seconds=runtime,
    )


def lse_lbfgs(
    grid: VFDGrid,
    initial_y: np.ndarray,
    coefficient_bound: float,
    reference_scale: float,
    evaluation_limit: int | None = None,
) -> Run:
    """Tensorized log-sum-exp continuation with bounded L-BFGS."""
    if evaluation_limit is not None and evaluation_limit < 1:
        raise ValueError("The LSE evaluation limit must be positive.")
    multipliers = (1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4)
    mus = [max(reference_scale * multiplier, 1e-12) for multiplier in multipliers]
    norm_smoothing = max(reference_scale * 1e-8, 1e-14)
    y = initial_y.copy()
    evaluation_count = 0
    iterates: list[np.ndarray] = []
    queries: list[int] = []
    gradient_rows: list[int] = []
    elapsed: list[float] = []
    messages: list[str] = []

    start = time.perf_counter()
    for mu in mus:

        def objective_and_gradient(
            candidate: np.ndarray,
        ) -> tuple[float, np.ndarray]:
            nonlocal evaluation_count
            if (
                evaluation_limit is not None
                and evaluation_count >= evaluation_limit
            ):
                raise EvaluationBudgetReached
            residual_r, residual_q = residual_arrays(candidate, grid)
            smooth_norm = np.sqrt(
                residual_r**2 + residual_q**2 + norm_smoothing**2
            )
            shifted_norm = smooth_norm - norm_smoothing
            normalizer = logsumexp(shifted_norm / mu)
            weights = np.exp(shifted_norm / mu - normalizer)
            weighted_r = weights * residual_r / smooth_norm
            weighted_q = weights * residual_q / smooth_norm
            even_gradient = (
                grid.even_powers.T
                @ weighted_r.T
                @ grid.even_frequency
            )
            odd_gradient = (
                grid.odd_powers.T
                @ weighted_q.T
                @ grid.odd_frequency
            )
            evaluation_count += 1
            return float(mu * normalizer), pack(even_gradient, odd_gradient)

        def callback(candidate: np.ndarray) -> None:
            iterates.append(candidate.copy())
            queries.append(evaluation_count * grid.size)
            gradient_rows.append(evaluation_count * grid.size)
            elapsed.append(time.perf_counter() - start)

        try:
            result = minimize(
                objective_and_gradient,
                y,
                method="L-BFGS-B",
                jac=True,
                bounds=[(-coefficient_bound, coefficient_bound)] * DIMENSION,
                callback=callback,
                options={
                    "maxiter": 35,
                    "ftol": 1e-15,
                    "gtol": 1e-11,
                    "maxls": 30,
                    "maxcor": 10,
                },
            )
        except EvaluationBudgetReached:
            if iterates:
                y = iterates[-1].copy()
            charged_queries = evaluation_count * grid.size
            if not queries or queries[-1] != charged_queries:
                iterates.append(y.copy())
                queries.append(charged_queries)
                gradient_rows.append(charged_queries)
                elapsed.append(time.perf_counter() - start)
            messages.append(
                f"stopped at {evaluation_count} joint full-grid evaluations"
            )
            break
        y = np.asarray(result.x, dtype=np.float64)
        messages.append(f"mu={mu:.3e}: {result.message}")
        if not iterates or not np.array_equal(iterates[-1], y):
            iterates.append(y.copy())
            queries.append(evaluation_count * grid.size)
            gradient_rows.append(evaluation_count * grid.size)
            elapsed.append(time.perf_counter() - start)

    runtime = time.perf_counter() - start
    return Run(
        method="LSE--L-BFGS",
        seed=None,
        subset_size=None,
        value_queries=np.asarray(queries, dtype=np.int64),
        component_gradient_rows=np.asarray(gradient_rows, dtype=np.int64),
        elapsed_seconds=np.asarray(elapsed, dtype=np.float64),
        iterates=iterates,
        runtime_seconds=runtime,
        status="; ".join(messages),
    )


def top_flat_indices(values: np.ndarray, count: int) -> np.ndarray:
    flat = values.ravel()
    count = min(count, flat.size)
    if count == flat.size:
        return np.argsort(flat)[::-1]
    selected = np.argpartition(flat, -count)[-count:]
    return selected[np.argsort(flat[selected])[::-1]]


def cut_rows(
    y: np.ndarray,
    grid: VFDGrid,
    flat_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    residual_r, residual_q, frequency_indices, delay_indices = sampled_errors(
        y, grid, flat_indices
    )
    norms = np.hypot(residual_r, residual_q)
    tangent_r = np.divide(
        residual_r,
        norms,
        out=np.ones_like(residual_r),
        where=norms > np.finfo(np.float64).tiny,
    )
    tangent_q = np.divide(
        residual_q,
        norms,
        out=np.zeros_like(residual_q),
        where=norms > np.finfo(np.float64).tiny,
    )

    # The tangent is the maximally violated SOC supporting direction.  Adding
    # its negative and the signed perpendicular pair prevents the next LP
    # solution from merely flipping or rotating the same residual.
    directions_r = np.stack(
        (tangent_r, -tangent_r, -tangent_q, tangent_q), axis=1
    ).ravel()
    directions_q = np.stack(
        (tangent_q, -tangent_q, tangent_r, -tangent_r), axis=1
    ).ravel()
    frequency_indices = np.repeat(frequency_indices, 4)
    delay_indices = np.repeat(delay_indices, 4)

    even_features = (
        grid.even_powers[delay_indices, :, None]
        * grid.even_frequency[frequency_indices, None, :]
    ).reshape(directions_r.size, -1)
    odd_features = (
        grid.odd_powers[delay_indices, :, None]
        * grid.odd_frequency[frequency_indices, None, :]
    ).reshape(directions_r.size, -1)
    rows = np.column_stack(
        (
            directions_r[:, None] * even_features,
            directions_q[:, None] * odd_features,
            -np.ones(directions_r.size, dtype=np.float64),
        )
    )
    rhs = (
        directions_r * grid.target_r[frequency_indices, delay_indices]
        + directions_q * grid.target_q[frequency_indices, delay_indices]
    )
    return rows, rhs


def conic_cutting_plane(
    grid: VFDGrid,
    initial_y: np.ndarray,
    coefficient_bound: float,
    initial_cuts: int,
    batch_size: int,
    max_iterations: int,
    absolute_tolerance: float,
    relative_tolerance: float,
    label: str,
) -> Run:
    """Outer-approximation exchange for the finite-grid SOCP.

    A complex-norm epigraph constraint is the intersection of its supporting
    halfspaces over all unit directions.  Each separation scan adds tangent
    halfspaces at the largest currently violated residuals.  The master LP
    objective is a lower bound and the scanned maximum is an upper bound.
    """
    start = time.perf_counter()
    initial_errors = error_array(initial_y, grid)
    value_queries = grid.size
    best_y = initial_y.copy()
    best_upper_bound = float(np.max(initial_errors))
    top_count = max(1, initial_cuts // 2)
    uniform_count = max(1, initial_cuts - top_count)
    initial_indices = np.unique(
        np.concatenate(
            (
                top_flat_indices(initial_errors, top_count),
                np.linspace(0, grid.size - 1, uniform_count, dtype=np.int64),
            )
        )
    )
    rows, rhs = cut_rows(initial_y, grid, initial_indices)

    objective = np.zeros(DIMENSION + 1, dtype=np.float64)
    objective[-1] = 1.0
    bounds = [(-coefficient_bound, coefficient_bound)] * DIMENSION + [(0.0, None)]
    iterates: list[np.ndarray] = []
    queries: list[int] = []
    gradient_rows: list[int] = []
    elapsed: list[float] = []
    lower_bound = 0.0
    status = "maximum iterations reached"

    for iteration in range(1, max_iterations + 1):
        result = linprog(
            objective,
            A_ub=rows,
            b_ub=rhs,
            bounds=bounds,
            method="highs",
            options={
                "dual_feasibility_tolerance": 1e-9,
                "primal_feasibility_tolerance": 1e-9,
            },
        )
        if not result.success:
            status = f"HiGHS failed: {result.message}"
            break

        candidate = np.asarray(result.x[:-1], dtype=np.float64)
        lower_bound = float(result.x[-1])
        residual_r, residual_q = residual_arrays(candidate, grid)
        errors = np.hypot(residual_r, residual_q)
        upper_bound = float(np.max(errors))
        if upper_bound < best_upper_bound:
            best_upper_bound = upper_bound
            best_y = candidate.copy()
        value_queries += grid.size
        iterates.append(candidate.copy())
        queries.append(value_queries)
        gradient_rows.append(0)
        elapsed.append(time.perf_counter() - start)

        gap = max(0.0, upper_bound - lower_bound)
        tolerance = absolute_tolerance + relative_tolerance * upper_bound
        if gap <= tolerance:
            status = (
                f"converged in {iteration} master solves; "
                f"lower={lower_bound:.8e}; upper={upper_bound:.8e}; gap={gap:.3e}"
            )
            break

        selected = top_flat_indices(errors, batch_size)
        new_rows, new_rhs = cut_rows(candidate, grid, selected)
        rows = np.vstack((rows, new_rows))
        rhs = np.concatenate((rhs, new_rhs))

    runtime = time.perf_counter() - start
    if status == "maximum iterations reached":
        status = (
            f"stopped after {max_iterations} master solves; "
            f"lower={lower_bound:.8e}; best upper={best_upper_bound:.8e}; "
            f"gap={max(0.0, best_upper_bound - lower_bound):.3e}"
        )
    if not iterates:
        iterates = [initial_y.copy()]
        queries = [value_queries]
        gradient_rows = [0]
        elapsed = [runtime]
    elif not np.array_equal(iterates[-1], best_y):
        iterates.append(best_y)
        queries.append(value_queries)
        gradient_rows.append(0)
        elapsed.append(runtime)
    return Run(
        method=label,
        seed=None,
        subset_size=None,
        value_queries=np.asarray(queries, dtype=np.int64),
        component_gradient_rows=np.asarray(gradient_rows, dtype=np.int64),
        elapsed_seconds=np.asarray(elapsed, dtype=np.float64),
        iterates=iterates,
        runtime_seconds=runtime,
        status=status,
        lower_bound=lower_bound,
    )


def singleton_run(
    method: str,
    y: np.ndarray,
    value_queries: int,
    runtime_seconds: float,
    status: str,
) -> Run:
    return Run(
        method=method,
        seed=None,
        subset_size=None,
        value_queries=np.asarray([value_queries], dtype=np.int64),
        component_gradient_rows=np.asarray([0], dtype=np.int64),
        elapsed_seconds=np.asarray([runtime_seconds], dtype=np.float64),
        iterates=[y.copy()],
        runtime_seconds=runtime_seconds,
        status=status,
    )


def validation_values(run: Run, validation_grid: VFDGrid) -> np.ndarray:
    return np.asarray(
        [max_error(y, validation_grid) for y in run.iterates],
        dtype=np.float64,
    )


def write_history(
    path: Path,
    evaluated_runs: list[tuple[Run, np.ndarray]],
    reference_value: float,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "method",
                "seed",
                "subset_size",
                "value_queries",
                "component_gradient_rows",
                "elapsed_seconds",
                "validation_max_error",
                "validation_excess_error",
            ]
        )
        for run, values in evaluated_runs:
            for query, gradients, elapsed, value in zip(
                run.value_queries,
                run.component_gradient_rows,
                run.elapsed_seconds,
                values,
                strict=True,
            ):
                writer.writerow(
                    [
                        run.method,
                        "" if run.seed is None else run.seed,
                        "" if run.subset_size is None else run.subset_size,
                        int(query),
                        int(gradients),
                        f"{elapsed:.9f}",
                        f"{value:.16e}",
                        f"{max(0.0, value - reference_value):.16e}",
                    ]
                )


def write_summary(
    path: Path,
    evaluated_runs: list[tuple[Run, np.ndarray]],
    training_grid: VFDGrid,
    reference_value: float,
    wls_preprocessing_seconds: float,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "method",
                "seed",
                "subset_size",
                "training_max_error",
                "validation_max_error",
                "validation_error_db",
                "validation_excess_error",
                "value_queries",
                "component_gradient_rows",
                "optimization_runtime_seconds",
                "runtime_including_common_wls_seconds",
                "max_abs_scaled_coefficient",
                "lower_bound",
                "status",
            ]
        )
        for run, values in evaluated_runs:
            final = run.iterates[-1]
            value = float(values[-1])
            preprocessing = 0.0 if run.method == "WLS" else wls_preprocessing_seconds
            writer.writerow(
                [
                    run.method,
                    "" if run.seed is None else run.seed,
                    "" if run.subset_size is None else run.subset_size,
                    f"{max_error(final, training_grid):.16e}",
                    f"{value:.16e}",
                    f"{20.0 * math.log10(max(value, np.finfo(float).tiny)):.9f}",
                    f"{max(0.0, value - reference_value):.16e}",
                    int(run.value_queries[-1]),
                    int(run.component_gradient_rows[-1]),
                    f"{run.runtime_seconds:.9f}",
                    f"{run.runtime_seconds + preprocessing:.9f}",
                    f"{np.max(np.abs(final)):.16e}",
                    "" if run.lower_bound is None else f"{run.lower_bound:.16e}",
                    run.status,
                ]
            )


def grouped_smax(
    evaluated_runs: Iterable[tuple[Run, np.ndarray]],
) -> dict[int, list[tuple[Run, np.ndarray]]]:
    result: dict[int, list[tuple[Run, np.ndarray]]] = {}
    for run, values in evaluated_runs:
        if run.method == "SMax-SGM" and run.subset_size is not None:
            result.setdefault(run.subset_size, []).append((run, values))
    return result


def plot_convergence(
    path: Path,
    evaluated_runs: list[tuple[Run, np.ndarray]],
    reference_value: float,
    x_kind: str,
    frequency_count: int,
    delay_count: int,
) -> None:
    if x_kind not in {"queries", "runtime"}:
        raise ValueError("x_kind must be 'queries' or 'runtime'.")
    fig, axis = plt.subplots(figsize=(7.3, 4.7), constrained_layout=True)
    colors = {64: "#0072B2", 256: "#009E73", 1024: "#D55E00"}

    smax_runs = grouped_smax(evaluated_runs)
    for subset_size, group in sorted(smax_runs.items()):
        stacked_values = np.vstack([values for _, values in group])
        if x_kind == "queries":
            x_values = group[0][0].value_queries
        else:
            x_values = np.median(
                np.vstack([run.elapsed_seconds for run, _ in group]), axis=0
            )
        median = np.median(stacked_values, axis=0)
        lower = np.quantile(stacked_values, 0.25, axis=0)
        upper = np.quantile(stacked_values, 0.75, axis=0)
        color = colors.get(subset_size)
        axis.plot(
            x_values,
            median,
            linewidth=2.0,
            color=color,
            label=rf"SMax-SGM ($m={subset_size}$)",
        )
        axis.fill_between(
            x_values,
            lower,
            upper,
            color=color,
            alpha=0.16,
            linewidth=0,
        )

    styles = {
        "Full-grid SGM": ("#CC79A7", "--"),
        "LSE--L-BFGS": ("#56B4E9", "-."),
        "Training conic exchange": ("#000000", ":"),
        "WLS": ("#E69F00", "None"),
    }
    for run, values in evaluated_runs:
        if run.method in {"SMax-SGM", "Fine-grid LSE reference"}:
            continue
        x_values = (
            run.value_queries if x_kind == "queries" else run.elapsed_seconds
        )
        color, linestyle = styles.get(run.method, (None, "-"))
        if run.method == "WLS":
            axis.scatter(
                x_values,
                values,
                color=color,
                marker="o",
                s=35,
                label=run.method,
                zorder=4,
            )
        else:
            axis.plot(
                x_values,
                values,
                color=color,
                linestyle=linestyle,
                linewidth=1.8,
                label=run.method,
            )

    axis.axhline(
        reference_value,
        color="#777777",
        linewidth=1.2,
        linestyle="--",
        label="Best observed",
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel(
        "Component-value queries" if x_kind == "queries" else "Optimization time (s)"
    )
    axis.set_ylabel("Validation maximum complex error")
    axis.set_title(
        rf"Farrow VFD minimax: $N=60$, $M=4$, "
        rf"$I\times L={frequency_count}\times{delay_count}$"
    )
    axis.grid(True, which="both", alpha=0.22)
    axis.legend(fontsize=8.2, ncol=1)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def median_final_rows(
    evaluated_runs: Iterable[tuple[Run, np.ndarray]],
) -> list[dict[str, float | str]]:
    grouped: dict[str, list[tuple[Run, np.ndarray]]] = {}
    for run, values in evaluated_runs:
        key = run.method
        if run.subset_size is not None:
            key += f" (m={run.subset_size})"
        grouped.setdefault(key, []).append((run, values))
    rows: list[dict[str, float | str]] = []
    for method, group in grouped.items():
        rows.append(
            {
                "method": method,
                "validation_error": float(
                    np.median([values[-1] for _, values in group])
                ),
                "runtime_seconds": float(
                    np.median([run.runtime_seconds for run, _ in group])
                ),
                "value_queries": float(
                    np.median([run.value_queries[-1] for run, _ in group])
                ),
            }
        )
    return rows


def run_gradient_checks(grid: VFDGrid, y: np.ndarray) -> dict[str, float]:
    rng = np.random.default_rng(917)
    flat_index = int(rng.integers(grid.size))
    residual_r, residual_q, frequency_indices, delay_indices = sampled_errors(
        y, grid, np.asarray([flat_index], dtype=np.int64)
    )
    gradient = active_subgradient(
        float(residual_r[0]),
        float(residual_q[0]),
        int(frequency_indices[0]),
        int(delay_indices[0]),
        grid,
    )
    direction = rng.normal(size=DIMENSION)
    direction /= np.linalg.norm(direction)
    step = 1e-7
    plus = sampled_errors(
        y + step * direction, grid, np.asarray([flat_index], dtype=np.int64)
    )
    minus = sampled_errors(
        y - step * direction, grid, np.asarray([flat_index], dtype=np.int64)
    )
    finite_difference = (
        math.hypot(float(plus[0][0]), float(plus[1][0]))
        - math.hypot(float(minus[0][0]), float(minus[1][0]))
    ) / (2.0 * step)
    analytic = float(gradient @ direction)
    return {
        "component_directional_derivative_absolute_error": abs(
            finite_difference - analytic
        ),
        "component_directional_derivative_finite_difference": finite_difference,
        "component_directional_derivative_analytic": analytic,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    training_grid = make_grid(args.frequency_count, args.delay_count, args.alpha)
    validation_grid = make_grid(
        args.validation_frequency_count,
        args.validation_delay_count,
        args.alpha,
    )

    print("Computing the common tensor least-squares warm start...", flush=True)
    wls_start = time.perf_counter()
    wls_y = wls_solution(training_grid)
    wls_runtime = time.perf_counter() - wls_start
    wls_training_error = max_error(wls_y, training_grid)
    coefficient_bound = max(1.0, 2.0 * float(np.max(np.abs(wls_y))))
    feature_norm = max_feature_norm(training_grid)
    print(
        f"WLS training max={wls_training_error:.8e}; "
        f"bound={coefficient_bound:.3e}; feature bound={feature_norm:.3e}",
        flush=True,
    )
    wls_run = singleton_run(
        "WLS",
        wls_y,
        training_grid.size,
        wls_runtime,
        "uniform tensor least-squares solution",
    )

    print("Selecting the common SGM stepsize in a declared pilot...", flush=True)
    eta_candidates = [
        scale * wls_training_error / feature_norm for scale in args.eta_scales
    ]
    pilot_records: list[dict[str, float]] = []
    pilot_subset = min(256, training_grid.size)
    for scale, candidate in zip(args.eta_scales, eta_candidates, strict=True):
        pilot = projected_subgradient(
            training_grid,
            wls_y,
            coefficient_bound,
            args.pilot_budget,
            candidate,
            min(20, args.history_points),
            pilot_subset,
            seed=10_000,
            label="pilot",
        )
        pilot_error = max_error(pilot.iterates[-1], training_grid)
        pilot_records.append(
            {
                "eta_scale": float(scale),
                "eta0": float(candidate),
                "training_max_error": pilot_error,
            }
        )
        print(
            f"  scale={scale:g}; eta0={candidate:.3e}; "
            f"pilot max={pilot_error:.8e}",
            flush=True,
        )
    selected_pilot = min(pilot_records, key=lambda row: row["training_max_error"])
    eta0 = float(selected_pilot["eta0"])
    print(f"Selected eta0={eta0:.3e}", flush=True)

    print("Computing a fine-grid empirical LSE reference...", flush=True)
    reference_initial = wls_solution(validation_grid)
    reference_initial_error = max_error(reference_initial, validation_grid)
    reference_run = lse_lbfgs(
        validation_grid,
        reference_initial,
        coefficient_bound,
        reference_initial_error,
    )
    reference_run.method = "Fine-grid LSE reference"
    reference_y = reference_run.iterates[-1]
    reference_value = max_error(reference_y, validation_grid)
    print(
        f"Reference validation max={reference_value:.8e}; "
        f"time={reference_run.runtime_seconds:.3f}s",
        flush=True,
    )

    print("Running training-grid conic exchange...", flush=True)
    exchange_run = conic_cutting_plane(
        training_grid,
        wls_y,
        coefficient_bound,
        args.exchange_initial_cuts,
        args.exchange_batch_size,
        args.exchange_max_iterations,
        args.exchange_absolute_tolerance,
        args.exchange_relative_tolerance,
        label="Training conic exchange",
    )

    print("Running full-grid SGM...", flush=True)
    full_sgm = projected_subgradient(
        training_grid,
        wls_y,
        coefficient_bound,
        args.value_budget,
        eta0,
        args.history_points,
        training_grid.size,
        seed=0,
        label="Full-grid SGM",
    )

    print("Running tensorized log-sum-exp continuation...", flush=True)
    smooth_run = lse_lbfgs(
        training_grid,
        wls_y,
        coefficient_bound,
        wls_training_error,
    )

    stochastic_runs: list[Run] = []
    for subset_size in args.subset_sizes:
        if subset_size > training_grid.size:
            continue
        for seed in range(args.seeds):
            print(f"Running SMax-SGM m={subset_size}, seed={seed}...", flush=True)
            stochastic_runs.append(
                projected_subgradient(
                    training_grid,
                    wls_y,
                    coefficient_bound,
                    args.value_budget,
                    eta0,
                    args.history_points,
                    subset_size,
                    seed,
                    "SMax-SGM",
                )
            )

    all_runs = [
        reference_run,
        wls_run,
        exchange_run,
        full_sgm,
        smooth_run,
        *stochastic_runs,
    ]
    print("Evaluating stored iterates on the independent validation grid...", flush=True)
    evaluated_runs = [
        (run, validation_values(run, validation_grid)) for run in all_runs
    ]

    # If another method improves on the exchange reference, use the best observed
    # validation value for nonnegative plotting while retaining the exchange gap.
    best_observed = min(float(np.min(values)) for _, values in evaluated_runs)
    plotting_reference = min(reference_value, best_observed)

    write_history(
        args.output_dir / "history.csv",
        evaluated_runs,
        plotting_reference,
    )
    write_summary(
        args.output_dir / "summary.csv",
        evaluated_runs,
        training_grid,
        plotting_reference,
        wls_runtime,
    )
    plot_convergence(
        args.output_dir / "convergence_vs_value_queries.png",
        evaluated_runs,
        plotting_reference,
        "queries",
        args.frequency_count,
        args.delay_count,
    )
    plot_convergence(
        args.output_dir / "convergence_vs_runtime.png",
        evaluated_runs,
        plotting_reference,
        "runtime",
        args.frequency_count,
        args.delay_count,
    )

    coefficients: dict[str, np.ndarray] = {
        "wls": wls_y,
        "fine_grid_lse_reference": reference_y,
        "training_exchange": exchange_run.iterates[-1],
        "full_grid_sgm": full_sgm.iterates[-1],
        "lse_lbfgs": smooth_run.iterates[-1],
    }
    for run in stochastic_runs:
        coefficients[f"smax_m{run.subset_size}_seed{run.seed}"] = run.iterates[-1]
    np.savez(args.output_dir / "coefficients.npz", **coefficients)

    try:
        recorded_output_dir = str(
            args.output_dir.resolve().relative_to(Path.cwd().resolve())
        )
    except ValueError:
        recorded_output_dir = str(args.output_dir.resolve())
    recorded_arguments = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    recorded_arguments["output_dir"] = recorded_output_dir
    config = {
        "source_setup": {
            "reference": "Zhao and Tay (2023), Example 1",
            "doi": "10.1007/s00034-022-02169-2",
            "filter_order_N": 60,
            "farrow_degree_M": 4,
            "independent_real_coefficients": DIMENSION,
            "frequency_interval": f"[0, {args.alpha} pi]",
            "delay_interval": "[0, 0.5]",
            "scaled_delay_coordinate": "u=2p; same polynomial space",
        },
        "arguments": recorded_arguments,
        "training_components": training_grid.size,
        "validation_components": validation_grid.size,
        "coefficient_bound": coefficient_bound,
        "wls_training_max_error": wls_training_error,
        "wls_preprocessing_seconds": wls_runtime,
        "max_component_feature_norm": feature_norm,
        "pilot_step_size_selection": {
            "subset_size": pilot_subset,
            "seed": 10_000,
            "value_budget": args.pilot_budget,
            "candidates": pilot_records,
            "selected_eta0": eta0,
            "selection_rule": "smallest final training-grid maximum error",
        },
        "reference": {
            "kind": (
                "empirical fine-grid LSE continuation; not an optimality "
                "certificate"
            ),
            "validation_max_error": reference_value,
            "best_observed_validation_error": best_observed,
            "plotting_reference": plotting_reference,
            "status": reference_run.status,
        },
        "implementation_work": {
            "full_feature_matrix_materialized": False,
            "naive_real_feature_storage_bytes": int(
                training_grid.size * DIMENSION * np.dtype(np.float64).itemsize
            ),
            "naive_complex_feature_storage_bytes": int(
                training_grid.size * DIMENSION * np.dtype(np.complex128).itemsize
            ),
            "two_coordinate_error_grid_bytes": int(
                2 * training_grid.size * np.dtype(np.float64).itemsize
            ),
            "full_methods": (
                "tensor contractions over frequency and delay; every joint "
                "objective-gradient evaluation still processes the full error grid"
            ),
            "smax": "lazy sampled component rows",
            "lse_joint_evaluations": int(
                smooth_run.value_queries[-1] // training_grid.size
            ),
            "lse_forward_adjoint_full_grid_passes": int(
                2 * smooth_run.value_queries[-1] // training_grid.size
            ),
            "lbfgs_maxcor": 10,
            "lbfgs_dense_hessian": False,
        },
        "checks": run_gradient_checks(training_grid, wls_y),
        "software": {
            "python": sys.version,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "matplotlib": plt.matplotlib.__version__,
        },
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
    }
    with (args.output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    print("\nMedian final results", flush=True)
    for row in median_final_rows(evaluated_runs):
        print(
            f"{row['method']:<42} "
            f"E_val={row['validation_error']:.8e}  "
            f"queries={row['value_queries']:.0f}  "
            f"time={row['runtime_seconds']:.3f}s",
            flush=True,
        )
    print(f"\nOutputs saved to {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
