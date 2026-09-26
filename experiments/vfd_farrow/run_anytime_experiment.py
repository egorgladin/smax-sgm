"""Unified anytime VFD comparison under a predeclared 250-pass budget.

SMax-SGM keeps the configuration selected using only the training objective at
one million component values.  Full-grid SGM is tuned separately for its final
training objective at the full 250d budget, avoiding a five-iteration tuning
horizon.  LSE--L-BFGS runs the prefix of its fixed continuation within the same
budget, and the stored Restricted-SOCP exchange trajectory supplies a certified
lower bound.  The primary metric is relative training suboptimality with
respect to that lower bound; fine-grid validation remains offline and uncharged.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SMAX = "SMax-SGM"
SGM = "Full-grid SGM"
LSE = "LSE--L-BFGS"
SOCP = "Restricted-SOCP exchange"
METHODS = (SMAX, SGM, LSE, SOCP)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frequency-count", type=int, default=1000)
    parser.add_argument("--delay-count", type=int, default=200)
    parser.add_argument("--validation-frequency-count", type=int, default=2001)
    parser.add_argument("--validation-delay-count", type=int, default=401)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--optimizer-value-budget", type=int, default=50_000_000)
    parser.add_argument("--tuning-budget", type=int, default=1_000_000)
    parser.add_argument("--relative-gap-target", type=float, default=0.05)
    parser.add_argument("--history-points", type=int, default=36)
    parser.add_argument("--confirmatory-seeds", type=int, default=20)
    parser.add_argument("--confirmatory-seed-start", type=int, default=200)
    parser.add_argument("--lse-timing-repeats", type=int, default=3)
    parser.add_argument("--sgm-timing-repeats", type=int, default=3)
    parser.add_argument(
        "--early-results-dir",
        type=Path,
        default=script_dir / "results_early_budget_final",
    )
    parser.add_argument(
        "--socp-results-dir",
        type=Path,
        default=script_dir / "results_virtual_age_comparison",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "results_anytime_final",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Recreate the figure and summaries from stored history.",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty table {path}.")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def relative_gap(value: float, lower_bound: float) -> float:
    return max(0.0, (value - lower_bound) / lower_bound)


def schedule_from_endpoint(
    warm_value: float,
    component_lipschitz_bound: float,
    first_step_scale: float,
    endpoint_ratio: float,
    iterations: int,
) -> dict[str, float]:
    """Parameterize eta_t=eta_0/sqrt(t+t_0) by its first and last values."""
    first_stepsize = (
        first_step_scale
        * warm_value
        / (component_lipschitz_bound * component_lipschitz_bound)
    )
    ratio_squared = endpoint_ratio * endpoint_ratio
    virtual_time = ratio_squared * (iterations - 1) / (1.0 - ratio_squared)
    if virtual_time < 1.0:
        raise ValueError("Requested endpoint ratio would require negative t_0.")
    return {
        "first_step_scale": first_step_scale,
        "endpoint_ratio": endpoint_ratio,
        "first_stepsize": first_stepsize,
        "eta0": first_stepsize * math.sqrt(virtual_time),
        "stepsize_offset": virtual_time - 1.0,
    }


def early_smax_iterations(path: Path, subset_size: int) -> np.ndarray:
    rows = read_rows(path)
    seeds = sorted({row["seed"] for row in rows if row["method"] == SMAX})
    if not seeds:
        return np.empty(0, dtype=np.int64)
    queries = {
        int(row["optimizer_value_queries"])
        for row in rows
        if row["method"] == SMAX and row["seed"] == seeds[0]
    }
    return np.asarray(
        sorted(query // subset_size for query in queries if query > 0),
        dtype=np.int64,
    )


def combined_checkpoints(
    iterations: int,
    count: int,
    required: np.ndarray,
) -> np.ndarray:
    points = long_run.checkpoints(iterations, count)
    required = np.asarray(required, dtype=np.int64)
    required = required[(required >= 1) & (required <= iterations)]
    return np.unique(np.concatenate((points, required, [iterations])))


def run_smax(
    args: argparse.Namespace,
    training_grid: base.VFDGrid,
    validation_grid: base.VFDGrid,
    wls_y: np.ndarray,
    coefficient_bound: float,
    lower_bound: float,
    selected: dict[str, object],
) -> tuple[list[dict[str, object]], dict[str, np.ndarray], float, dict[str, object]]:
    subset_size = int(selected["subset_size"])
    iterations = args.optimizer_value_budget // subset_size
    required = early_smax_iterations(
        args.early_results_dir / "history.csv",
        subset_size,
    )
    required = np.unique(
        np.append(required, args.tuning_budget // subset_size)
    )
    checkpoints = combined_checkpoints(
        iterations,
        args.history_points,
        required,
    )
    compilation_seconds = followup.warm_compiled_kernel(
        training_grid,
        subset_size,
    )
    checks = followup.kernel_self_checks(training_grid, wls_y)
    if (
        not checks["sampled_active_index_match"]
        or not checks["sampler_unique"]
        or not checks["sampler_in_range"]
        or float(checks["sampled_response_max_absolute_error"]) > 1e-10
    ):
        raise RuntimeError(f"Compiled SMax kernel checks failed: {checks}")

    warm_train = base.max_error(wls_y, training_grid)
    warm_validation = base.max_error(wls_y, validation_grid)
    rows: list[dict[str, object]] = []
    coefficients: dict[str, np.ndarray] = {}
    seeds = range(
        args.confirmatory_seed_start,
        args.confirmatory_seed_start + args.confirmatory_seeds,
    )
    for position, seed in enumerate(seeds, start=1):
        print(
            f"SMax confirmation {position}/{args.confirmatory_seeds}, seed={seed}",
            flush=True,
        )
        trajectory = followup.compiled_smax_trajectory(
            training_grid,
            wls_y,
            coefficient_bound,
            args.optimizer_value_budget,
            float(selected["eta0"]),
            subset_size,
            seed,
            math.ceil(training_grid.size / subset_size),
            checkpoints,
            stepsize_power=0.5,
            stepsize_offset=float(selected["stepsize_offset"]),
        )
        rows.append(
            {
                "method": SMAX,
                "seed": seed,
                "optimizer_value_queries": 0,
                "optimizer_runtime_seconds": 0.0,
                "training_max_error": warm_train,
                "validation_max_error": warm_validation,
                "relative_training_gap": relative_gap(warm_train, lower_bound),
            }
        )
        for query, elapsed, iterate in zip(
            trajectory.sample_queries,
            trajectory.elapsed_seconds,
            trajectory.uniform_iterates,
        ):
            train = base.max_error(iterate, training_grid)
            rows.append(
                {
                    "method": SMAX,
                    "seed": seed,
                    "optimizer_value_queries": int(query),
                    "optimizer_runtime_seconds": float(elapsed),
                    "training_max_error": train,
                    "validation_max_error": base.max_error(
                        iterate,
                        validation_grid,
                    ),
                    "relative_training_gap": relative_gap(train, lower_bound),
                }
            )
        coefficients[f"smax_seed_{seed}"] = trajectory.uniform_iterates[-1]
    metadata = {
        "subset_size": subset_size,
        "iterations": iterations,
        "eta0": float(selected["eta0"]),
        "stepsize_offset": float(selected["stepsize_offset"]),
        "first_stepsize": float(selected["first_stepsize"]),
        "checkpoints": checkpoints.tolist(),
        "kernel_checks": checks,
    }
    return rows, coefficients, compilation_seconds, metadata


def full_grid_sgm(
    grid: base.VFDGrid,
    initial_y: np.ndarray,
    coefficient_bound: float,
    optimizer_value_budget: int,
    eta0: float,
    stepsize_offset: float,
    checkpoints: np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, np.ndarray]:
    checkpoint_set = set(int(value) for value in checkpoints)
    y = initial_y.copy()
    average = y.copy()
    iterates: list[np.ndarray] = []
    elapsed: list[float] = []
    start = time.perf_counter()
    iterations = optimizer_value_budget // grid.size
    for iteration in range(1, iterations + 1):
        average += (y - average) / iteration
        residual_r, residual_q = base.residual_arrays(y, grid)
        errors = np.hypot(residual_r, residual_q)
        frequency_index, delay_index = np.unravel_index(
            int(np.argmax(errors)),
            errors.shape,
        )
        gradient = base.active_subgradient(
            float(residual_r[frequency_index, delay_index]),
            float(residual_q[frequency_index, delay_index]),
            int(frequency_index),
            int(delay_index),
            grid,
        )
        step = eta0 / math.sqrt(iteration + stepsize_offset)
        y = np.clip(
            y - step * gradient,
            -coefficient_bound,
            coefficient_bound,
        )
        if iteration in checkpoint_set:
            iterates.append(average.copy())
            elapsed.append(time.perf_counter() - start)
    return (
        y,
        iterates,
        checkpoints * grid.size,
        np.asarray(elapsed, dtype=np.float64),
    )


def tune_full_grid_sgm(
    args: argparse.Namespace,
    training_grid: base.VFDGrid,
    wls_y: np.ndarray,
    coefficient_bound: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Tune SGM conventionally by its final training objective at 250d.

    The search never queries the validation grid.  Selecting the final
    objective rather than time-to-5% avoids tailoring the baseline to the
    presentation threshold while giving it a meaningful 250-iteration horizon.
    """
    iterations = args.optimizer_value_budget // training_grid.size
    if iterations < 2:
        raise ValueError("Full-budget SGM tuning requires at least two iterations.")
    warm_value = base.max_error(wls_y, training_grid)
    component_lipschitz_bound = base.max_feature_norm(training_grid)
    first_step_scales = [0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0]
    endpoint_ratios = [0.075, 0.1, 0.2, 0.35, 0.5, 0.75, 0.95]
    rows: list[dict[str, object]] = []
    total = len(first_step_scales) * len(endpoint_ratios)
    candidate_index = 0
    for first_step_scale in first_step_scales:
        for endpoint_ratio in endpoint_ratios:
            candidate_index += 1
            try:
                schedule = schedule_from_endpoint(
                    warm_value,
                    component_lipschitz_bound,
                    first_step_scale,
                    endpoint_ratio,
                    iterations,
                )
            except ValueError:
                continue
            _, iterates, queries, elapsed = full_grid_sgm(
                training_grid,
                wls_y,
                coefficient_bound,
                args.optimizer_value_budget,
                schedule["eta0"],
                schedule["stepsize_offset"],
                np.asarray([iterations], dtype=np.int64),
            )
            final_training = base.max_error(iterates[-1], training_grid)
            rows.append(
                {
                    **schedule,
                    "iterations": iterations,
                    "optimizer_value_queries": int(queries[-1]),
                    "training_max_error": final_training,
                    "optimizer_runtime_seconds": float(elapsed[-1]),
                }
            )
            print(
                f"SGM tuning {candidate_index}/{total}: "
                f"scale={first_step_scale:g}, rho={endpoint_ratio:g}, "
                f"train={final_training:.8g}",
                flush=True,
            )
    winner = dict(
        min(
            rows,
            key=lambda row: (
                float(row["training_max_error"]),
                float(row["first_step_scale"]),
                float(row["endpoint_ratio"]),
            ),
        )
    )
    winner["selection_rule"] = "minimum final training-grid maximum at 250d"
    winner["validation_used"] = False
    return rows, winner


def run_sgm(
    args: argparse.Namespace,
    training_grid: base.VFDGrid,
    validation_grid: base.VFDGrid,
    wls_y: np.ndarray,
    coefficient_bound: float,
    lower_bound: float,
    selected: dict[str, object],
) -> tuple[list[dict[str, object]], np.ndarray, dict[str, object]]:
    iterations = args.optimizer_value_budget // training_grid.size
    required = np.asarray(
        [args.tuning_budget // training_grid.size],
        dtype=np.int64,
    )
    checkpoints = combined_checkpoints(
        iterations,
        args.history_points,
        required,
    )
    runs = [
        full_grid_sgm(
            training_grid,
            wls_y,
            coefficient_bound,
            args.optimizer_value_budget,
            float(selected["eta0"]),
            float(selected["stepsize_offset"]),
            checkpoints,
        )
        for _ in range(args.sgm_timing_repeats)
    ]
    _, iterates, queries, _ = runs[0]
    runtimes = np.median(
        np.asarray([run[3] for run in runs], dtype=np.float64),
        axis=0,
    )
    warm_train = base.max_error(wls_y, training_grid)
    rows: list[dict[str, object]] = [
        {
            "method": SGM,
            "seed": "",
            "optimizer_value_queries": 0,
            "optimizer_runtime_seconds": 0.0,
            "training_max_error": warm_train,
            "validation_max_error": base.max_error(wls_y, validation_grid),
            "relative_training_gap": relative_gap(warm_train, lower_bound),
        }
    ]
    for query, elapsed, iterate in zip(queries, runtimes, iterates):
        train = base.max_error(iterate, training_grid)
        rows.append(
            {
                "method": SGM,
                "seed": "",
                "optimizer_value_queries": int(query),
                "optimizer_runtime_seconds": float(elapsed),
                "training_max_error": train,
                "validation_max_error": base.max_error(
                    iterate,
                    validation_grid,
                ),
                "relative_training_gap": relative_gap(train, lower_bound),
            }
        )
    metadata = {
        "iterations": iterations,
        "eta0": float(selected["eta0"]),
        "stepsize_offset": float(selected["stepsize_offset"]),
        "first_stepsize": float(selected["first_stepsize"]),
        "timing_repeats_seconds": [float(run[3][-1]) for run in runs],
        "checkpoints": checkpoints.tolist(),
    }
    return rows, iterates[-1], metadata


def run_lse(
    args: argparse.Namespace,
    training_grid: base.VFDGrid,
    validation_grid: base.VFDGrid,
    wls_y: np.ndarray,
    coefficient_bound: float,
    lower_bound: float,
) -> tuple[list[dict[str, object]], np.ndarray, dict[str, object]]:
    warm_train = base.max_error(wls_y, training_grid)
    runs: list[base.Run] = []
    for repeat in range(args.lse_timing_repeats):
        run = base.lse_lbfgs(
            training_grid,
            wls_y,
            coefficient_bound,
            warm_train,
            evaluation_limit=(
                args.optimizer_value_budget // training_grid.size
            ),
        )
        runs.append(run)
        retained = np.flatnonzero(
            run.value_queries <= args.optimizer_value_budget
        )
        if retained.size == 0:
            raise RuntimeError("LSE produced no iterate within the common budget.")
        print(
            f"LSE timing repeat {repeat + 1}/{args.lse_timing_repeats}: "
            f"{run.value_queries[retained[-1]]:,} retained values, "
            f"{run.elapsed_seconds[retained[-1]]:.3f}s",
            flush=True,
        )
    run = runs[0]
    retained = np.flatnonzero(run.value_queries <= args.optimizer_value_budget)
    for item in runs[1:]:
        if not np.array_equal(item.value_queries, run.value_queries):
            raise RuntimeError("LSE timing repeats have inconsistent trajectories.")
    elapsed = np.median(
        np.asarray([item.elapsed_seconds[retained] for item in runs]),
        axis=0,
    )
    median_total = float(elapsed[-1])
    rows: list[dict[str, object]] = [
        {
            "method": LSE,
            "seed": "",
            "optimizer_value_queries": 0,
            "optimizer_runtime_seconds": 0.0,
            "training_max_error": warm_train,
            "validation_max_error": base.max_error(wls_y, validation_grid),
            "relative_training_gap": relative_gap(warm_train, lower_bound),
        }
    ]
    retained_queries = run.value_queries[retained]
    retained_iterates = [run.iterates[int(index)] for index in retained]
    for query, seconds, iterate in zip(
        retained_queries,
        elapsed,
        retained_iterates,
    ):
        train = base.max_error(iterate, training_grid)
        rows.append(
            {
                "method": LSE,
                "seed": "",
                "optimizer_value_queries": int(query),
                "optimizer_runtime_seconds": float(seconds),
                "training_max_error": train,
                "validation_max_error": base.max_error(
                    iterate,
                    validation_grid,
                ),
                "relative_training_gap": relative_gap(train, lower_bound),
            }
        )
    metadata = {
        "component_value_queries": int(retained_queries[-1]),
        "joint_full_grid_evaluations": int(
            retained_queries[-1] // training_grid.size
        ),
        "fixed_temperature_multipliers": [0.1, 0.03, 0.01, 0.003, 0.001, 0.0003],
        "timing_repeats_seconds": [
            float(item.elapsed_seconds[retained[-1]]) for item in runs
        ],
        "timing_median_seconds": median_total,
        "common_component_value_cap": args.optimizer_value_budget,
        "trajectory_rule": "last accepted L-BFGS-B iterate within the cap",
        "status": run.status,
    }
    return rows, retained_iterates[-1], metadata


def stored_socp(
    args: argparse.Namespace,
    training_grid: base.VFDGrid,
    validation_grid: base.VFDGrid,
    wls_y: np.ndarray,
    lower_bound: float,
) -> list[dict[str, object]]:
    warm_train = base.max_error(wls_y, training_grid)
    rows: list[dict[str, object]] = [
        {
            "method": SOCP,
            "seed": "",
            "optimizer_value_queries": 0,
            "optimizer_runtime_seconds": 0.0,
            "training_max_error": warm_train,
            "validation_max_error": base.max_error(wls_y, validation_grid),
            "relative_training_gap": relative_gap(warm_train, lower_bound),
        }
    ]
    for source in read_rows(args.socp_results_dir / "socp_history.csv"):
        train = float(source["best_full_grid_max"])
        rows.append(
            {
                "method": SOCP,
                "seed": "",
                "optimizer_value_queries": int(source["value_queries"]),
                "optimizer_runtime_seconds": float(source["elapsed_seconds"]),
                "training_max_error": train,
                "validation_max_error": float(source["validation_max_error"]),
                "relative_training_gap": relative_gap(train, lower_bound),
            }
        )
    return rows


def aggregate_smax(
    rows: list[dict[str, str]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    seeds = sorted({int(row["seed"]) for row in rows})
    runs = [
        sorted(
            [row for row in rows if int(row["seed"]) == seed],
            key=lambda row: int(row["optimizer_value_queries"]),
        )
        for seed in seeds
    ]
    queries = np.asarray(
        [[int(row["optimizer_value_queries"]) for row in run] for run in runs]
    )
    if not np.all(queries == queries[0]):
        raise ValueError("SMax confirmatory queries are inconsistent.")
    runtime = np.asarray(
        [[float(row["optimizer_runtime_seconds"]) for row in run] for run in runs]
    )
    gaps = np.asarray(
        [[float(row["relative_training_gap"]) for row in run] for run in runs]
    )
    validation = np.asarray(
        [[float(row["validation_max_error"]) for row in run] for run in runs]
    )
    return queries[0], np.median(runtime, axis=0), gaps, validation, np.asarray(seeds)


def first_crossing(
    queries: np.ndarray,
    runtime: np.ndarray,
    gaps: np.ndarray,
    target: float,
) -> tuple[int | str, float | str, float | str]:
    positions = np.flatnonzero(gaps <= target)
    if positions.size == 0:
        return "", "", ""
    position = int(positions[0])
    return int(queries[position]), float(runtime[position]), float(gaps[position])


def write_summaries(
    output_dir: Path,
    relative_gap_target: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows = read_rows(output_dir / "history.csv")
    smax_rows = [row for row in rows if row["method"] == SMAX]
    queries, runtime, gaps, validation, seeds = aggregate_smax(smax_rows)
    median_gaps = np.median(gaps, axis=0)
    median_validation = np.median(validation, axis=0)
    crossing = first_crossing(
        queries,
        runtime,
        median_gaps,
        relative_gap_target,
    )
    target_rows: list[dict[str, object]] = [
        {
            "method": SMAX,
            "aggregation": f"median over {seeds.size} seeds",
            "target_relative_gap": relative_gap_target,
            "optimizer_value_queries_to_target": crossing[0],
            "optimizer_runtime_seconds_to_target": crossing[1],
            "relative_gap_at_crossing": crossing[2],
        }
    ]
    final_rows: list[dict[str, object]] = [
        {
            "method": SMAX,
            "runs": seeds.size,
            "optimizer_value_queries": int(queries[-1]),
            "optimizer_runtime_seconds": float(runtime[-1]),
            "training_max_error": float(
                np.median(
                    [
                        float(row["training_max_error"])
                        for row in smax_rows
                        if int(row["optimizer_value_queries"]) == queries[-1]
                    ]
                )
            ),
            "validation_max_error": float(median_validation[-1]),
            "relative_training_gap": float(median_gaps[-1]),
            "validation_q25": float(np.quantile(validation[:, -1], 0.25)),
            "validation_q75": float(np.quantile(validation[:, -1], 0.75)),
        }
    ]
    for method in (SGM, LSE, SOCP):
        method_rows = sorted(
            [row for row in rows if row["method"] == method],
            key=lambda row: int(row["optimizer_value_queries"]),
        )
        method_queries = np.asarray(
            [int(row["optimizer_value_queries"]) for row in method_rows]
        )
        method_runtime = np.asarray(
            [float(row["optimizer_runtime_seconds"]) for row in method_rows]
        )
        method_gaps = np.asarray(
            [float(row["relative_training_gap"]) for row in method_rows]
        )
        crossing = first_crossing(
            method_queries,
            method_runtime,
            method_gaps,
            relative_gap_target,
        )
        target_rows.append(
            {
                "method": method,
                "aggregation": "deterministic",
                "target_relative_gap": relative_gap_target,
                "optimizer_value_queries_to_target": crossing[0],
                "optimizer_runtime_seconds_to_target": crossing[1],
                "relative_gap_at_crossing": crossing[2],
            }
        )
        endpoint = method_rows[-1]
        final_rows.append(
            {
                "method": method,
                "runs": 1,
                "optimizer_value_queries": int(
                    endpoint["optimizer_value_queries"]
                ),
                "optimizer_runtime_seconds": float(
                    endpoint["optimizer_runtime_seconds"]
                ),
                "training_max_error": float(endpoint["training_max_error"]),
                "validation_max_error": float(
                    endpoint["validation_max_error"]
                ),
                "relative_training_gap": float(
                    endpoint["relative_training_gap"]
                ),
                "validation_q25": "",
                "validation_q75": "",
            }
        )
    write_rows(output_dir / "time_to_target.csv", target_rows)
    write_rows(output_dir / "final_summary.csv", final_rows)
    return target_rows, final_rows


def plot_results(
    output_dir: Path,
    relative_gap_target: float,
    tuning_budget: int,
) -> None:
    lower_y_limit = 2e-5
    rows = read_rows(output_dir / "history.csv")
    smax_rows = [row for row in rows if row["method"] == SMAX]
    queries, runtime, gaps, _, _ = aggregate_smax(smax_rows)
    deterministic = {
        method: sorted(
            [row for row in rows if row["method"] == method],
            key=lambda row: int(row["optimizer_value_queries"]),
        )
        for method in (SGM, LSE, SOCP)
    }
    colors = {
        SMAX: "#0072B2",
        SGM: "#D55E00",
        LSE: "#009E73",
        SOCP: "#555555",
    }
    display_names = {
        SMAX: "SMax\u2013SGM, $m=8192$",
        SGM: "Full-grid SGM",
        LSE: "LSE\u2013L-BFGS",
        SOCP: "Restricted SOCP",
    }

    def endpoint_label(value: float) -> str:
        mantissa, exponent = f"{value:.1e}".split("e")
        return rf"${mantissa}\times 10^{{{int(exponent)}}}$"

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.35), constrained_layout=True)
    for axis, x_field, smax_x, x_label in (
        (
            axes[0],
            "optimizer_value_queries",
            queries,
            "Optimizer component-value queries",
        ),
        (
            axes[1],
            "optimizer_runtime_seconds",
            runtime,
            "Optimizer runtime (seconds)",
        ),
    ):
        median = np.maximum(np.median(gaps, axis=0), 1e-9)
        q25 = np.maximum(np.quantile(gaps, 0.25, axis=0), 1e-9)
        q75 = np.maximum(np.quantile(gaps, 0.75, axis=0), 1e-9)
        positive = smax_x > 0.0
        axis.plot(
            smax_x[positive],
            median[positive],
            color=colors[SMAX],
            linewidth=2.2,
            label=display_names[SMAX],
        )
        axis.fill_between(
            smax_x[positive],
            q25[positive],
            q75[positive],
            color=colors[SMAX],
            alpha=0.17,
            linewidth=0,
        )
        for method in (SGM, LSE, SOCP):
            method_rows = deterministic[method]
            x = np.asarray([float(row[x_field]) for row in method_rows])
            raw_y = np.asarray(
                [float(row["relative_training_gap"]) for row in method_rows]
            )
            y = np.maximum(
                raw_y,
                lower_y_limit if method == SOCP else 1e-9,
            )
            positive = x > 0.0
            axis.plot(
                x[positive],
                y[positive],
                color=colors[method],
                linewidth=1.8,
                marker="o" if method == SGM else "s",
                markersize=3.0,
                markevery=max(1, len(x) // 18),
                drawstyle="steps-post" if method == SOCP else "default",
                label=display_names[method],
            )
            if method == SOCP and raw_y[-1] < lower_y_limit:
                text_offset = (-10, 23) if axis is axes[1] else (10, 23)
                axis.annotate(
                    endpoint_label(raw_y[-1]),
                    xy=(x[-1], lower_y_limit),
                    xytext=text_offset,
                    textcoords="offset points",
                    ha="right" if axis is axes[1] else "left",
                    va="bottom",
                    arrowprops={
                        "arrowstyle": "->",
                        "color": colors[method],
                        "linewidth": 0.9,
                    },
                )
        axis.axhline(
            relative_gap_target,
            color="#777777",
            linestyle="--",
            linewidth=1.1,
            label="5% target" if axis is axes[0] else None,
        )
        axis.set_yscale("log")
        axis.set_xscale("log")
        axis.set_ylim(bottom=lower_y_limit)
        axis.set_xlabel(x_label)
        axis.grid(True, which="major", color="#808080", alpha=0.35, linewidth=0.7)
        axis.grid(True, which="minor", color="#A0A0A0", alpha=0.14, linewidth=0.45)
    axes[0].axvline(
        tuning_budget,
        color="#777777",
        linestyle=":",
        linewidth=1.0,
    )
    axes[0].set_ylabel("Relative gap to numerical SOCP reference")
    axes[0].legend(fontsize=7.7, loc="best")
    fig.savefig(output_dir / "anytime_relative_gap_comparison.png", dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        write_summaries(args.output_dir, args.relative_gap_target)
        plot_results(
            args.output_dir,
            args.relative_gap_target,
            args.tuning_budget,
        )
        print(f"Recreated stored outputs in {args.output_dir}", flush=True)
        return

    global base, followup, long_run
    import run_experiment as base
    import run_followup_experiment as followup
    import run_long_smax_experiment as long_run

    if args.confirmatory_seeds < 1:
        raise ValueError("At least one confirmatory seed is required.")
    if args.optimizer_value_budget < args.tuning_budget:
        raise ValueError("The final budget cannot be smaller than the tuning budget.")

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
    wls_start = time.perf_counter()
    wls_y = base.wls_solution(training_grid)
    wls_runtime = time.perf_counter() - wls_start
    coefficient_bound = max(1.0, 2.0 * float(np.max(np.abs(wls_y))))
    early_config = json.loads(
        (args.early_results_dir / "run_config.json").read_text(encoding="utf-8")
    )
    selected_smax = json.loads(
        (args.early_results_dir / "selected_smax.json").read_text(
            encoding="utf-8"
        )
    )
    socp_config = json.loads(
        (args.socp_results_dir / "run_config.json").read_text(encoding="utf-8")
    )
    lower_bound = float(
        socp_config["restricted_socp"]["training_lower_bound"]
    )

    sgm_tuning_rows, selected_sgm = tune_full_grid_sgm(
        args,
        training_grid,
        wls_y,
        coefficient_bound,
    )
    write_rows(
        args.output_dir / "sgm_full_budget_tuning.csv",
        sgm_tuning_rows,
    )
    (args.output_dir / "selected_sgm_full_budget.json").write_text(
        json.dumps(selected_sgm, indent=2),
        encoding="utf-8",
    )
    print(f"Frozen full-budget SGM configuration: {selected_sgm}", flush=True)

    smax_rows, smax_coefficients, compilation_seconds, smax_metadata = run_smax(
        args,
        training_grid,
        validation_grid,
        wls_y,
        coefficient_bound,
        lower_bound,
        selected_smax,
    )
    sgm_rows, sgm_y, sgm_metadata = run_sgm(
        args,
        training_grid,
        validation_grid,
        wls_y,
        coefficient_bound,
        lower_bound,
        selected_sgm,
    )
    lse_rows, lse_y, lse_metadata = run_lse(
        args,
        training_grid,
        validation_grid,
        wls_y,
        coefficient_bound,
        lower_bound,
    )
    socp_rows = stored_socp(
        args,
        training_grid,
        validation_grid,
        wls_y,
        lower_bound,
    )
    history: list[dict[str, object]] = []
    history.extend(smax_rows)
    history.extend(sgm_rows)
    history.extend(lse_rows)
    history.extend(socp_rows)
    write_rows(args.output_dir / "history.csv", history)

    coefficients: dict[str, np.ndarray] = {
        "wls": wls_y,
        "full_grid_sgm": sgm_y,
        "lse_lbfgs": lse_y,
    }
    coefficients.update(smax_coefficients)
    np.savez_compressed(args.output_dir / "coefficients.npz", **coefficients)
    target_rows, final_rows = write_summaries(
        args.output_dir,
        args.relative_gap_target,
    )
    plot_results(
        args.output_dir,
        args.relative_gap_target,
        args.tuning_budget,
    )

    config = {
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "problem": {
            "dimension": base.DIMENSION,
            "training_components": training_grid.size,
            "validation_components": validation_grid.size,
            "coefficient_bound": coefficient_bound,
            "wls_runtime_seconds_excluded": wls_runtime,
            "wls_training_max_error": base.max_error(wls_y, training_grid),
            "wls_validation_max_error": base.max_error(wls_y, validation_grid),
        },
        "reference": {
            "source": str(args.socp_results_dir / "run_config.json"),
            "certified_training_lower_bound": lower_bound,
            "relative_gap_definition": "(training maximum - lower bound) / lower bound",
            "target_relative_gap": args.relative_gap_target,
        },
        "protocol": {
            "single_anytime_experiment": True,
            "smax_tuned_at_optimizer_values": args.tuning_budget,
            "early_tuning_source": str(args.early_results_dir),
            "early_tuning_validation_used": early_config["protocol"][
                "validation_used_for_tuning"
            ],
            "smax_long_horizon_retuning": False,
            "full_grid_sgm_tuned_at_optimizer_values": (
                args.optimizer_value_budget
            ),
            "full_grid_sgm_tuning_selection_rule": selected_sgm[
                "selection_rule"
            ],
            "full_grid_sgm_tuning_validation_used": False,
            "confirmatory_seeds": list(
                range(
                    args.confirmatory_seed_start,
                    args.confirmatory_seed_start + args.confirmatory_seeds,
                )
            ),
            "validation": (
                "independent dense grid; evaluation only, with no retraining "
                "or output selection"
            ),
            "query_axis": "optimizer component-value queries; WLS excluded",
            "runtime_axis": "optimizer runtime; WLS and validation excluded",
            "reported_outputs": {
                "SMax-SGM": "uniform average of pre-update iterates per seed",
                "Full-grid SGM": "uniform average of pre-update iterates",
                "LSE--L-BFGS": (
                    "last accepted L-BFGS-B iterate within the common cap"
                ),
                "Restricted-SOCP exchange": "best feasible incumbent",
            },
        },
        "smax": {
            **smax_metadata,
            "compiled_setup_seconds_excluded": compilation_seconds,
        },
        "full_grid_sgm": {
            **sgm_metadata,
            "tuning": selected_sgm,
            "tuning_table": str(
                args.output_dir / "sgm_full_budget_tuning.csv"
            ),
        },
        "lse": lse_metadata,
        "restricted_socp": {
            "trajectory_source": str(
                args.socp_results_dir / "socp_history.csv"
            ),
            "stored_runtime_reused": True,
            "training_upper_bound": socp_config["restricted_socp"][
                "training_upper_bound"
            ],
        },
        "time_to_target": target_rows,
        "final_summary": final_rows,
        "plot_reproduction": {
            "requires_optimizer_reruns": False,
            "command": (
                "python experiments/vfd_farrow/run_anytime_experiment.py "
                "--plot-only"
            ),
        },
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )
    print(f"Saved unified anytime experiment to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
