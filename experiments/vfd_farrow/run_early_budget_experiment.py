"""Train-only tuning and confirmation in the early-query VFD regime.

The target production budget is fixed before tuning and shared by all
methods.  SMax is tuned only by its final training-grid maximum on development
seeds.  The chosen configuration is then frozen and evaluated on disjoint
confirmatory seeds and an independent validation grid.

The stepsize is parameterized by quantities that remain comparable when the
subset size changes:

    eta_t = eta_0 / sqrt(t + t_0),
    eta_1 = c h(y_WLS) / G^2,
    eta_T / eta_1 = rho.

Here T=floor(B/m), G is the uniform component-subgradient bound, and the
clock shift t_0 and eta_0 are determined by (c, rho, T).  This is still the
standard shifted inverse-square-root schedule.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import run_experiment as base
import run_followup_experiment as followup
import run_long_smax_experiment as long_run
import run_virtual_age_comparison as comparison


SMax_METHOD = "SMax-SGM"
SGM_METHOD = "Full-grid SGM"
LSE_METHOD = "LSE--L-BFGS"
SOCP_METHOD = "Restricted-SOCP exchange"


@dataclass(frozen=True)
class Schedule:
    first_stepsize: float
    endpoint_ratio: float
    eta0: float
    offset: float


class EvaluationBudgetReached(RuntimeError):
    """Internal signal used to stop LSE before an extra full-grid pass."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frequency-count", type=int, default=1000)
    parser.add_argument("--delay-count", type=int, default=200)
    parser.add_argument("--validation-frequency-count", type=int, default=2001)
    parser.add_argument("--validation-delay-count", type=int, default=401)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--target-budget", type=int, default=5_000_000)
    parser.add_argument(
        "--subset-sizes",
        type=int,
        nargs="+",
        default=[256, 512, 1024, 2048, 4096, 8192, 16384],
    )
    parser.add_argument(
        "--first-step-scales",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 2.0, 4.0, 6.0, 8.0],
    )
    parser.add_argument(
        "--endpoint-ratios",
        type=float,
        nargs="+",
        default=[0.15, 0.25, 0.5, 0.75, 0.9, 0.97],
    )
    parser.add_argument("--development-seeds", type=int, default=3)
    parser.add_argument("--development-seed-start", type=int, default=0)
    parser.add_argument("--confirmatory-seeds", type=int, default=20)
    parser.add_argument("--confirmatory-seed-start", type=int, default=200)
    parser.add_argument("--history-points", type=int, default=24)
    parser.add_argument("--sgm-timing-repeats", type=int, default=3)
    parser.add_argument(
        "--socp-history",
        type=Path,
        default=Path(__file__).with_name("results_virtual_age_comparison")
        / "socp_history.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("results_early_budget"),
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Recreate the figure and target table from stored history.",
    )
    parser.add_argument(
        "--reuse-smax-tuning",
        action="store_true",
        help="Reuse selected_smax.json and retain the stored tuning tables.",
    )
    return parser.parse_args()


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty table {path}.")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def schedule_from_endpoint(
    warm_value: float,
    component_lipschitz_bound: float,
    first_step_scale: float,
    endpoint_ratio: float,
    iterations: int,
) -> Schedule:
    if iterations < 2:
        raise ValueError("At least two iterations are required.")
    if first_step_scale <= 0.0:
        raise ValueError("The first-step scale must be positive.")
    if not 0.0 < endpoint_ratio < 1.0:
        raise ValueError("The endpoint ratio must lie strictly between 0 and 1.")
    first_stepsize = (
        first_step_scale
        * warm_value
        / (component_lipschitz_bound * component_lipschitz_bound)
    )
    ratio_squared = endpoint_ratio * endpoint_ratio
    virtual_time = ratio_squared * (iterations - 1) / (1.0 - ratio_squared)
    if virtual_time < 1.0:
        raise ValueError("Requested endpoint ratio would require negative t_0.")
    offset = virtual_time - 1.0
    eta0 = first_stepsize * math.sqrt(virtual_time)
    realized_ratio = math.sqrt(virtual_time / (iterations + offset))
    if not math.isclose(realized_ratio, endpoint_ratio, rel_tol=1e-12):
        raise AssertionError("Stepsize endpoint parameterization is inconsistent.")
    return Schedule(first_stepsize, endpoint_ratio, eta0, offset)


def tune_smax(
    args: argparse.Namespace,
    training_grid: base.VFDGrid,
    wls_y: np.ndarray,
    coefficient_bound: float,
    warm_value: float,
    component_lipschitz_bound: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    development_seeds = list(
        range(
            args.development_seed_start,
            args.development_seed_start + args.development_seeds,
        )
    )
    candidates: list[tuple[int, float, float]] = [
        (m, c, rho)
        for m in args.subset_sizes
        for c in args.first_step_scales
        for rho in args.endpoint_ratios
    ]
    np.random.default_rng(20260802).shuffle(candidates)
    run_rows: list[dict[str, object]] = []
    total = len(candidates)
    for candidate_index, (subset_size, first_scale, endpoint_ratio) in enumerate(
        candidates,
        start=1,
    ):
        iterations = args.target_budget // subset_size
        try:
            schedule = schedule_from_endpoint(
                warm_value,
                component_lipschitz_bound,
                first_scale,
                endpoint_ratio,
                iterations,
            )
        except ValueError as error:
            if "negative t_0" not in str(error):
                raise
            continue
        if candidate_index == 1 or candidate_index % 10 == 0:
            print(
                f"SMax tuning configuration {candidate_index}/{total}: "
                f"m={subset_size}, c={first_scale:g}, rho={endpoint_ratio:g}",
                flush=True,
            )
        for seed in development_seeds:
            trajectory = followup.compiled_smax_trajectory(
                training_grid,
                wls_y,
                coefficient_bound,
                args.target_budget,
                schedule.eta0,
                subset_size,
                seed,
                math.ceil(training_grid.size / subset_size),
                np.asarray([iterations], dtype=np.int64),
                stepsize_power=0.5,
                stepsize_offset=schedule.offset,
            )
            final_y = trajectory.uniform_iterates[-1]
            run_rows.append(
                {
                    "subset_size": subset_size,
                    "first_step_scale": first_scale,
                    "endpoint_ratio": endpoint_ratio,
                    "seed": seed,
                    "iterations": iterations,
                    "optimizer_value_queries": int(
                        trajectory.sample_queries[-1]
                    ),
                    "first_stepsize": schedule.first_stepsize,
                    "eta0": schedule.eta0,
                    "stepsize_offset": schedule.offset,
                    "training_max_error": base.max_error(
                        final_y,
                        training_grid,
                    ),
                    "optimizer_runtime_seconds": float(
                        trajectory.elapsed_seconds[-1]
                    ),
                }
            )

    summary_rows: list[dict[str, object]] = []
    keys = sorted(
        {
            (
                int(row["subset_size"]),
                float(row["first_step_scale"]),
                float(row["endpoint_ratio"]),
            )
            for row in run_rows
        }
    )
    for subset_size, first_scale, endpoint_ratio in keys:
        selected = [
            row
            for row in run_rows
            if row["subset_size"] == subset_size
            and row["first_step_scale"] == first_scale
            and row["endpoint_ratio"] == endpoint_ratio
        ]
        values = np.asarray(
            [float(row["training_max_error"]) for row in selected]
        )
        runtimes = np.asarray(
            [float(row["optimizer_runtime_seconds"]) for row in selected]
        )
        first = selected[0]
        summary_rows.append(
            {
                "subset_size": subset_size,
                "first_step_scale": first_scale,
                "endpoint_ratio": endpoint_ratio,
                "development_seeds": len(selected),
                "iterations": first["iterations"],
                "optimizer_value_queries": first["optimizer_value_queries"],
                "first_stepsize": first["first_stepsize"],
                "eta0": first["eta0"],
                "stepsize_offset": first["stepsize_offset"],
                "training_median": float(np.median(values)),
                "training_q75": float(np.quantile(values, 0.75)),
                "training_mean": float(np.mean(values)),
                "runtime_median_seconds": float(np.median(runtimes)),
            }
        )
    ranked = sorted(
        summary_rows,
        key=lambda row: (
            float(row["training_median"]),
            float(row["training_q75"]),
            float(row["training_mean"]),
            int(row["subset_size"]),
        ),
    )
    winner = dict(ranked[0])
    winner["selection_rule"] = (
        "lexicographic minimum of training median, q75, mean, then subset size"
    )
    return run_rows, summary_rows, winner


def tune_full_grid_sgm(
    args: argparse.Namespace,
    training_grid: base.VFDGrid,
    wls_y: np.ndarray,
    coefficient_bound: float,
    warm_value: float,
    component_lipschitz_bound: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    iterations = args.target_budget // training_grid.size
    rows: list[dict[str, object]] = []
    first_step_scales = [0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0]
    endpoint_ratios = [0.5, 0.6, 0.75, 0.9, 0.97]
    for first_scale in first_step_scales:
        for endpoint_ratio in endpoint_ratios:
            try:
                schedule = schedule_from_endpoint(
                    warm_value,
                    component_lipschitz_bound,
                    first_scale,
                    endpoint_ratio,
                    iterations,
                )
            except ValueError as error:
                if "negative t_0" not in str(error):
                    raise
                continue
            run = comparison.full_grid_virtual_age_sgm(
                training_grid,
                wls_y,
                coefficient_bound,
                args.target_budget,
                schedule.eta0,
                schedule.offset,
                history_points=2,
            )
            final_y = run.iterates[-1]
            rows.append(
                {
                    "first_step_scale": first_scale,
                    "endpoint_ratio": endpoint_ratio,
                    "iterations": iterations,
                    "optimizer_value_queries": int(
                        run.optimizer_value_queries[-1]
                    ),
                    "first_stepsize": schedule.first_stepsize,
                    "eta0": schedule.eta0,
                    "stepsize_offset": schedule.offset,
                    "training_max_error": base.max_error(
                        final_y,
                        training_grid,
                    ),
                    "optimizer_runtime_seconds": float(
                        run.optimizer_elapsed_seconds[-1]
                    ),
                }
            )
    winner = dict(
        min(
            rows,
            key=lambda row: (
                float(row["training_max_error"]),
                float(row["first_step_scale"]),
            ),
        )
    )
    winner["selection_rule"] = "minimum final training-grid maximum"
    return rows, winner


def confirm_smax(
    args: argparse.Namespace,
    training_grid: base.VFDGrid,
    validation_grid: base.VFDGrid,
    wls_y: np.ndarray,
    coefficient_bound: float,
    winner: dict[str, object],
) -> tuple[
    list[dict[str, object]],
    dict[int, np.ndarray],
    dict[int, np.ndarray],
]:
    subset_size = int(winner["subset_size"])
    iterations = args.target_budget // subset_size
    run_checkpoints = long_run.checkpoints(iterations, args.history_points)
    seeds = range(
        args.confirmatory_seed_start,
        args.confirmatory_seed_start + args.confirmatory_seeds,
    )
    rows: list[dict[str, object]] = []
    final_coefficients: dict[int, np.ndarray] = {}
    checkpoint_coefficients: dict[int, np.ndarray] = {}
    warm_train = base.max_error(wls_y, training_grid)
    warm_validation = base.max_error(wls_y, validation_grid)
    for position, seed in enumerate(seeds, start=1):
        print(
            f"Confirmatory SMax {position}/{args.confirmatory_seeds}, seed={seed}",
            flush=True,
        )
        trajectory = followup.compiled_smax_trajectory(
            training_grid,
            wls_y,
            coefficient_bound,
            args.target_budget,
            float(winner["eta0"]),
            subset_size,
            seed,
            math.ceil(training_grid.size / subset_size),
            run_checkpoints,
            stepsize_power=0.5,
            stepsize_offset=float(winner["stepsize_offset"]),
        )
        rows.append(
            {
                "method": SMax_METHOD,
                "seed": seed,
                "optimizer_value_queries": 0,
                "optimizer_runtime_seconds": 0.0,
                "training_max_error": warm_train,
                "validation_max_error": warm_validation,
            }
        )
        for query, elapsed, iterate in zip(
            trajectory.sample_queries,
            trajectory.elapsed_seconds,
            trajectory.uniform_iterates,
        ):
            rows.append(
                {
                    "method": SMax_METHOD,
                    "seed": seed,
                    "optimizer_value_queries": int(query),
                    "optimizer_runtime_seconds": float(elapsed),
                    "training_max_error": base.max_error(
                        iterate,
                        training_grid,
                    ),
                    "validation_max_error": base.max_error(
                        iterate,
                        validation_grid,
                    ),
                }
            )
        final_coefficients[seed] = trajectory.uniform_iterates[-1]
        checkpoint_coefficients[seed] = np.stack(
            trajectory.uniform_iterates,
            axis=0,
        )
    return rows, final_coefficients, checkpoint_coefficients


def confirm_sgm(
    args: argparse.Namespace,
    training_grid: base.VFDGrid,
    validation_grid: base.VFDGrid,
    wls_y: np.ndarray,
    coefficient_bound: float,
    winner: dict[str, object],
) -> tuple[list[dict[str, object]], np.ndarray]:
    runs: list[comparison.SGMRun] = []
    for _ in range(args.sgm_timing_repeats):
        runs.append(
            comparison.full_grid_virtual_age_sgm(
                training_grid,
                wls_y,
                coefficient_bound,
                args.target_budget,
                float(winner["eta0"]),
                float(winner["stepsize_offset"]),
                args.history_points,
            )
        )
    runtime_matrix = np.asarray(
        [run.optimizer_elapsed_seconds for run in runs],
        dtype=np.float64,
    )
    median_runtime = np.median(runtime_matrix, axis=0)
    run = runs[0]
    rows: list[dict[str, object]] = [
        {
            "method": SGM_METHOD,
            "seed": "",
            "optimizer_value_queries": 0,
            "optimizer_runtime_seconds": 0.0,
            "training_max_error": base.max_error(wls_y, training_grid),
            "validation_max_error": base.max_error(wls_y, validation_grid),
        }
    ]
    for query, elapsed, iterate in zip(
        run.optimizer_value_queries,
        median_runtime,
        run.iterates,
    ):
        rows.append(
            {
                "method": SGM_METHOD,
                "seed": "",
                "optimizer_value_queries": int(query),
                "optimizer_runtime_seconds": float(elapsed),
                "training_max_error": base.max_error(iterate, training_grid),
                "validation_max_error": base.max_error(
                    iterate,
                    validation_grid,
                ),
            }
        )
    return rows, run.iterates[-1]


def budgeted_lse(
    args: argparse.Namespace,
    training_grid: base.VFDGrid,
    validation_grid: base.VFDGrid,
    wls_y: np.ndarray,
    coefficient_bound: float,
    warm_value: float,
) -> tuple[list[dict[str, object]], np.ndarray, dict[str, object]]:
    """Run the first fixed-temperature LSE stage under a hard pass budget.

    In addition to accepted L-BFGS-B iterates, retain the trial point with the
    smallest exact training maximum among already charged joint evaluations.
    This gives the short-horizon baseline useful output even if its line
    search has not accepted the first iteration.
    """
    from scipy.optimize import minimize
    from scipy.special import logsumexp

    evaluation_limit = args.target_budget // training_grid.size
    mu = max(warm_value * 1e-1, 1e-12)
    norm_smoothing = max(warm_value * 1e-8, 1e-14)
    evaluation_count = 0
    accepted_y = wls_y.copy()
    best_y = wls_y.copy()
    best_training = warm_value
    accepted_queries: list[int] = []
    accepted_elapsed: list[float] = []
    accepted_points: list[np.ndarray] = []
    start = time.perf_counter()

    def objective_and_gradient(candidate: np.ndarray) -> tuple[float, np.ndarray]:
        nonlocal evaluation_count, best_y, best_training
        if evaluation_count >= evaluation_limit:
            raise EvaluationBudgetReached
        residual_r, residual_q = base.residual_arrays(candidate, training_grid)
        candidate_training = float(np.max(np.hypot(residual_r, residual_q)))
        if candidate_training < best_training:
            best_training = candidate_training
            best_y = np.asarray(candidate, dtype=np.float64).copy()
        smooth_norm = np.sqrt(
            residual_r**2 + residual_q**2 + norm_smoothing**2
        )
        shifted_norm = smooth_norm - norm_smoothing
        normalizer = logsumexp(shifted_norm / mu)
        weights = np.exp(shifted_norm / mu - normalizer)
        weighted_r = weights * residual_r / smooth_norm
        weighted_q = weights * residual_q / smooth_norm
        even_gradient = (
            training_grid.even_powers.T
            @ weighted_r.T
            @ training_grid.even_frequency
        )
        odd_gradient = (
            training_grid.odd_powers.T
            @ weighted_q.T
            @ training_grid.odd_frequency
        )
        evaluation_count += 1
        return float(mu * normalizer), base.pack(even_gradient, odd_gradient)

    def callback(candidate: np.ndarray) -> None:
        nonlocal accepted_y
        accepted_y = np.asarray(candidate, dtype=np.float64).copy()
        accepted_queries.append(evaluation_count * training_grid.size)
        accepted_elapsed.append(time.perf_counter() - start)
        accepted_points.append(accepted_y.copy())

    status = "budget reached"
    try:
        result = minimize(
            objective_and_gradient,
            wls_y,
            method="L-BFGS-B",
            jac=True,
            bounds=[(-coefficient_bound, coefficient_bound)] * base.DIMENSION,
            callback=callback,
            options={
                "maxiter": 35,
                "ftol": 1e-15,
                "gtol": 1e-11,
                "maxls": 30,
                "maxcor": 10,
            },
        )
        accepted_y = np.asarray(result.x, dtype=np.float64).copy()
        status = str(result.message)
    except EvaluationBudgetReached:
        pass
    runtime = time.perf_counter() - start
    rows: list[dict[str, object]] = [
        {
            "method": LSE_METHOD,
            "seed": "",
            "optimizer_value_queries": 0,
            "optimizer_runtime_seconds": 0.0,
            "training_max_error": base.max_error(wls_y, training_grid),
            "validation_max_error": base.max_error(wls_y, validation_grid),
        }
    ]
    for query, elapsed, point in zip(
        accepted_queries,
        accepted_elapsed,
        accepted_points,
    ):
        rows.append(
            {
                "method": LSE_METHOD,
                "seed": "",
                "optimizer_value_queries": query,
                "optimizer_runtime_seconds": elapsed,
                "training_max_error": base.max_error(
                    point,
                    training_grid,
                ),
                "validation_max_error": base.max_error(
                    point,
                    validation_grid,
                ),
            }
        )
    best_row = {
        "method": LSE_METHOD,
        "seed": "",
        "optimizer_value_queries": evaluation_count * training_grid.size,
        "optimizer_runtime_seconds": runtime,
        "training_max_error": base.max_error(best_y, training_grid),
        "validation_max_error": base.max_error(
            best_y,
            validation_grid,
        ),
    }
    if (
        len(rows) > 1
        and int(rows[-1]["optimizer_value_queries"])
        == int(best_row["optimizer_value_queries"])
    ):
        rows[-1] = best_row
    else:
        rows.append(best_row)
    metadata = {
        "temperature": mu,
        "temperature_multiplier": 0.1,
        "joint_evaluation_limit": evaluation_limit,
        "joint_evaluations_used": evaluation_count,
        "accepted_iterates": len(accepted_queries),
        "status": status,
        "output_rule": (
            "smallest exact training maximum among charged trial points"
        ),
        "best_observed_training_max_error": best_training,
    }
    return rows, best_y, metadata


def socp_rows(
    args: argparse.Namespace,
    training_grid: base.VFDGrid,
    validation_grid: base.VFDGrid,
    wls_y: np.ndarray,
) -> list[dict[str, object]]:
    warm_train = base.max_error(wls_y, training_grid)
    warm_validation = base.max_error(wls_y, validation_grid)
    rows: list[dict[str, object]] = [
        {
            "method": SOCP_METHOD,
            "seed": "",
            "optimizer_value_queries": 0,
            "optimizer_runtime_seconds": 0.0,
            "training_max_error": warm_train,
            "validation_max_error": warm_validation,
        }
    ]
    for source in read_rows(args.socp_history):
        queries = int(source["value_queries"])
        if queries > args.target_budget:
            break
        rows.append(
            {
                "method": SOCP_METHOD,
                "seed": "",
                "optimizer_value_queries": queries,
                "optimizer_runtime_seconds": float(source["elapsed_seconds"]),
                "training_max_error": float(source["best_full_grid_max"]),
                "validation_max_error": float(source["validation_max_error"]),
            }
        )
    if int(rows[-1]["optimizer_value_queries"]) < args.target_budget:
        last = dict(rows[-1])
        last["optimizer_value_queries"] = args.target_budget
        rows.append(last)
    return rows


def aggregate_smax(
    rows: list[dict[str, str]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    seeds = sorted({int(row["seed"]) for row in rows if row["seed"]})
    runs = [
        sorted(
            [row for row in rows if row["seed"] and int(row["seed"]) == seed],
            key=lambda row: int(row["optimizer_value_queries"]),
        )
        for seed in seeds
    ]
    queries = np.asarray(
        [[int(row["optimizer_value_queries"]) for row in run] for run in runs]
    )
    runtimes = np.asarray(
        [[float(row["optimizer_runtime_seconds"]) for row in run] for run in runs]
    )
    training = np.asarray(
        [[float(row["training_max_error"]) for row in run] for run in runs]
    )
    validation = np.asarray(
        [[float(row["validation_max_error"]) for row in run] for run in runs]
    )
    if not np.all(queries == queries[0]):
        raise ValueError("Confirmatory SMax runs have inconsistent checkpoints.")
    return (
        queries[0],
        np.median(runtimes, axis=0),
        training,
        validation,
        np.asarray(seeds, dtype=np.int64),
    )


def make_target_summary(
    output_dir: Path,
    target_budget: int,
) -> list[dict[str, object]]:
    rows = read_rows(output_dir / "history.csv")
    summary: list[dict[str, object]] = []
    for method in (SMax_METHOD, SGM_METHOD, LSE_METHOD, SOCP_METHOD):
        method_rows = [row for row in rows if row["method"] == method]
        if method == SMax_METHOD:
            seeds = sorted({int(row["seed"]) for row in method_rows})
            endpoints = []
            for seed in seeds:
                eligible = [
                    row
                    for row in method_rows
                    if int(row["seed"]) == seed
                    and int(row["optimizer_value_queries"]) <= target_budget
                ]
                endpoints.append(
                    max(eligible, key=lambda row: int(row["optimizer_value_queries"]))
                )
            validation = np.asarray(
                [float(row["validation_max_error"]) for row in endpoints]
            )
            training = np.asarray(
                [float(row["training_max_error"]) for row in endpoints]
            )
            runtimes = np.asarray(
                [float(row["optimizer_runtime_seconds"]) for row in endpoints]
            )
            summary.append(
                {
                    "method": method,
                    "runs": len(endpoints),
                    "optimizer_value_queries": int(
                        endpoints[0]["optimizer_value_queries"]
                    ),
                    "optimizer_runtime_seconds": float(np.median(runtimes)),
                    "training_max_error": float(np.median(training)),
                    "validation_max_error": float(np.median(validation)),
                    "validation_q25": float(np.quantile(validation, 0.25)),
                    "validation_q75": float(np.quantile(validation, 0.75)),
                }
            )
        else:
            endpoint = max(
                [
                    row
                    for row in method_rows
                    if int(row["optimizer_value_queries"]) <= target_budget
                ],
                key=lambda row: int(row["optimizer_value_queries"]),
            )
            summary.append(
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
                    "validation_q25": "",
                    "validation_q75": "",
                }
            )
    write_rows(output_dir / "target_budget_summary.csv", summary)
    return summary


def plot_results(output_dir: Path, target_budget: int) -> None:
    rows = read_rows(output_dir / "history.csv")
    smax_rows = [row for row in rows if row["method"] == SMax_METHOD]
    queries, runtimes, _, validation, _ = aggregate_smax(smax_rows)
    selected_smax = json.loads(
        (output_dir / "selected_smax.json").read_text(encoding="utf-8")
    )
    smax_label = f"SMax-SGM, $m={int(selected_smax['subset_size'])}$"
    deterministic: dict[str, list[dict[str, str]]] = {}
    for method in (SGM_METHOD, LSE_METHOD, SOCP_METHOD):
        deterministic[method] = sorted(
            [row for row in rows if row["method"] == method],
            key=lambda row: int(row["optimizer_value_queries"]),
        )
    colors = {
        SMax_METHOD: "#0072B2",
        SGM_METHOD: "#D55E00",
        LSE_METHOD: "#009E73",
        SOCP_METHOD: "#555555",
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.35), constrained_layout=True)
    for axis, x_field, x_values, x_label in (
        (
            axes[0],
            "optimizer_value_queries",
            queries,
            "Optimizer component-value queries",
        ),
        (
            axes[1],
            "optimizer_runtime_seconds",
            runtimes,
            "Optimizer runtime (seconds)",
        ),
    ):
        median = np.median(validation, axis=0)
        axis.plot(
            x_values,
            median,
            color=colors[SMax_METHOD],
            linewidth=2.2,
            label=smax_label,
        )
        axis.fill_between(
            x_values,
            np.quantile(validation, 0.25, axis=0),
            np.quantile(validation, 0.75, axis=0),
            color=colors[SMax_METHOD],
            alpha=0.17,
            linewidth=0,
        )
        for method in (SGM_METHOD, LSE_METHOD, SOCP_METHOD):
            method_rows = deterministic[method]
            x = np.asarray([float(row[x_field]) for row in method_rows])
            y = np.asarray(
                [float(row["validation_max_error"]) for row in method_rows]
            )
            axis.plot(
                x,
                y,
                color=colors[method],
                linewidth=1.8,
                marker="o" if method == SGM_METHOD else "s",
                markersize=3.2,
                drawstyle="steps-post" if method == SOCP_METHOD else "default",
                label=method,
            )
        if x_field.endswith("seconds"):
            axis.set_xscale("symlog", linthresh=0.01)
            axis.set_xticks([0.0, 0.01, 0.1, 1.0])
            axis.set_xticklabels(["0", "$10^{-2}$", "$10^{-1}$", "$10^0$"])
        else:
            axis.set_xscale("symlog", linthresh=10_000)
            axis.set_xticks([0.0, 10_000, 100_000, 1_000_000])
            axis.set_xticklabels(["0", "$10^4$", "$10^5$", "$10^6$"])
        axis.set_yscale("log")
        axis.set_xlabel(x_label)
        axis.grid(True, which="both", axis="both", alpha=0.25)
        axis.set_xlim(left=0)
    axes[0].axvline(target_budget, color="#777777", linestyle=":", linewidth=1.0)
    axes[0].set_ylabel("Validation maximum complex error")
    axes[0].legend(fontsize=7.7, loc="best")
    fig.savefig(output_dir / "early_budget_cost_time_comparison.png", dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        plot_results(args.output_dir, args.target_budget)
        make_target_summary(args.output_dir, args.target_budget)
        print(f"Recreated stored outputs in {args.output_dir}", flush=True)
        return
    if args.development_seeds < 1 or args.confirmatory_seeds < 1:
        raise ValueError("Positive development and confirmatory seed counts required.")
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
    if args.target_budget < training_grid.size:
        raise ValueError("Target budget must allow at least one full-grid scan.")
    wls_start = time.perf_counter()
    wls_y = base.wls_solution(training_grid)
    wls_runtime = time.perf_counter() - wls_start
    warm_training = base.max_error(wls_y, training_grid)
    warm_validation = base.max_error(wls_y, validation_grid)
    coefficient_bound = max(1.0, 2.0 * float(np.max(np.abs(wls_y))))
    component_lipschitz_bound = base.max_feature_norm(training_grid)

    print("Compiling SMax kernel...", flush=True)
    compilation_seconds = followup.warm_compiled_kernel(
        training_grid,
        max(args.subset_sizes),
    )
    checks = followup.kernel_self_checks(training_grid, wls_y)
    if not checks["sampled_active_index_match"] or not checks["sampler_unique"]:
        raise RuntimeError(f"Compiled-kernel checks failed: {checks}")

    if args.reuse_smax_tuning:
        smax_winner = json.loads(
            (args.output_dir / "selected_smax.json").read_text(
                encoding="utf-8"
            )
        )
    else:
        tuning_rows, tuning_summary, smax_winner = tune_smax(
            args,
            training_grid,
            wls_y,
            coefficient_bound,
            warm_training,
            component_lipschitz_bound,
        )
        write_rows(args.output_dir / "smax_tuning_runs.csv", tuning_rows)
        write_rows(args.output_dir / "smax_tuning_summary.csv", tuning_summary)
        (args.output_dir / "selected_smax.json").write_text(
            json.dumps(smax_winner, indent=2),
            encoding="utf-8",
        )
    print(f"Frozen SMax configuration: {smax_winner}", flush=True)

    sgm_tuning, sgm_winner = tune_full_grid_sgm(
        args,
        training_grid,
        wls_y,
        coefficient_bound,
        warm_training,
        component_lipschitz_bound,
    )
    write_rows(args.output_dir / "sgm_tuning.csv", sgm_tuning)
    (args.output_dir / "selected_sgm.json").write_text(
        json.dumps(sgm_winner, indent=2),
        encoding="utf-8",
    )
    print(f"Frozen full-grid SGM configuration: {sgm_winner}", flush=True)

    smax_history, smax_coefficients, smax_checkpoint_coefficients = confirm_smax(
        args,
        training_grid,
        validation_grid,
        wls_y,
        coefficient_bound,
        smax_winner,
    )
    sgm_history, sgm_y = confirm_sgm(
        args,
        training_grid,
        validation_grid,
        wls_y,
        coefficient_bound,
        sgm_winner,
    )
    lse_history, lse_y, lse_metadata = budgeted_lse(
        args,
        training_grid,
        validation_grid,
        wls_y,
        coefficient_bound,
        warm_training,
    )
    restricted_socp_history = socp_rows(
        args,
        training_grid,
        validation_grid,
        wls_y,
    )
    history: list[dict[str, object]] = []
    history.extend(smax_history)
    history.extend(sgm_history)
    history.extend(lse_history)
    history.extend(restricted_socp_history)
    write_rows(args.output_dir / "history.csv", history)

    arrays: dict[str, np.ndarray] = {
        "wls": wls_y,
        "full_grid_sgm": sgm_y,
        "lse_lbfgs": lse_y,
    }
    arrays.update(
        {f"smax_seed_{seed}": value for seed, value in smax_coefficients.items()}
    )
    arrays.update(
        {
            f"smax_seed_{seed}_checkpoints": value
            for seed, value in smax_checkpoint_coefficients.items()
        }
    )
    socp_coefficients = args.socp_history.parent / "coefficients.npz"
    if socp_coefficients.exists():
        with np.load(socp_coefficients, allow_pickle=False) as source:
            if "restricted_socp" in source:
                arrays["restricted_socp"] = np.asarray(
                    source["restricted_socp"],
                    dtype=np.float64,
                )
    np.savez_compressed(args.output_dir / "coefficients.npz", **arrays)
    summary = make_target_summary(args.output_dir, args.target_budget)
    plot_results(args.output_dir, args.target_budget)

    config = {
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "protocol": {
            "target_budget_chosen_before_tuning": True,
            "target_budget_reason": (
                "predeclared common cap equal to "
                f"{args.target_budget / training_grid.size:g} "
                "training-grid scans"
            ),
            "tuning_criterion": "final training-grid maximum only",
            "validation_used_for_tuning": False,
            "development_seeds": list(
                range(
                    args.development_seed_start,
                    args.development_seed_start + args.development_seeds,
                )
            ),
            "confirmatory_seeds": list(
                range(
                    args.confirmatory_seed_start,
                    args.confirmatory_seed_start + args.confirmatory_seeds,
                )
            ),
            "production_axes_exclude_common_wls": True,
            "development_search_not_charged_to_production_runs": True,
            "smax_tuning_reused_in_this_invocation": args.reuse_smax_tuning,
            "tuning_stages": (
                [
                    {
                        "name": "broad search",
                        "runs": "broad_smax_tuning_runs.csv",
                        "summary": "broad_smax_tuning_summary.csv",
                        "refinement_trigger": (
                            "training-only optimum lay on the smallest "
                            "endpoint-ratio boundary"
                        ),
                    },
                    {
                        "name": "local refinement",
                        "runs": "smax_tuning_runs.csv",
                        "summary": "smax_tuning_summary.csv",
                    },
                ]
                if (
                    args.output_dir / "broad_smax_tuning_summary.csv"
                ).exists()
                else [
                    {
                        "name": "search",
                        "runs": "smax_tuning_runs.csv",
                        "summary": "smax_tuning_summary.csv",
                    }
                ]
            ),
        },
        "problem": {
            "training_components": training_grid.size,
            "validation_components": validation_grid.size,
            "dimension": base.DIMENSION,
            "wls_training_max_error": warm_training,
            "wls_validation_max_error": warm_validation,
            "wls_runtime_seconds": wls_runtime,
            "coefficient_bound": coefficient_bound,
            "component_lipschitz_bound": component_lipschitz_bound,
        },
        "compiled_kernel": {
            "setup_seconds_excluded": compilation_seconds,
            "checks": checks,
        },
        "selected_smax": smax_winner,
        "selected_sgm": sgm_winner,
        "lse": lse_metadata,
        "restricted_socp_source": str(args.socp_history),
        "target_summary": summary,
        "plot_reproduction": {
            "requires_optimizer_reruns": False,
            "command": (
                "python experiments/vfd_farrow/run_early_budget_experiment.py "
                f"--plot-only --target-budget {args.target_budget} "
                f"--output-dir {args.output_dir}"
            ),
        },
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )
    print(f"Saved early-budget experiment to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
