"""Cost-aware comparison of virtual-age SMax, SGM, and restricted SOCP.

The confirmed virtual-age SMax trajectories are reused from
``results_long_smax``.  Full-grid SGM is rerun with the same WLS start,
stepsize coefficient, virtual-age offset, and optimizer component-value
budget.  Restricted-SOCP exchange is rerun so that every incumbent can be
evaluated on the independent validation grid.  Validation evaluations are
offline and are not charged to either query or runtime axes.
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
import plot_cost_time_comparison as stored_plot


@dataclass
class SGMRun:
    optimizer_value_queries: np.ndarray
    optimizer_elapsed_seconds: np.ndarray
    iterates: list[np.ndarray]
    final_point: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frequency-count", type=int, default=1000)
    parser.add_argument("--delay-count", type=int, default=200)
    parser.add_argument("--validation-frequency-count", type=int, default=2001)
    parser.add_argument("--validation-delay-count", type=int, default=401)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--subset-size", type=int, default=4096)
    parser.add_argument("--theta", type=float, default=4.0)
    parser.add_argument("--eta-scale", type=float, default=8.0)
    parser.add_argument("--history-points", type=int, default=18)
    parser.add_argument("--sgm-timing-repeats", type=int, default=3)
    parser.add_argument(
        "--smax-results-dir",
        type=Path,
        default=Path(__file__).with_name("results_long_smax"),
    )
    parser.add_argument(
        "--socp-results-dir",
        type=Path,
        default=Path(__file__).with_name("results_followup"),
    )
    parser.add_argument(
        "--lse-results-dir",
        type=Path,
        default=Path(__file__).with_name("results_full_grid"),
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Recreate plots from stored histories without rerunning methods.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("results_virtual_age_comparison"),
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty table to {path}.")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def full_grid_virtual_age_sgm(
    grid: base.VFDGrid,
    initial_y: np.ndarray,
    coefficient_bound: float,
    optimizer_value_budget: int,
    eta0: float,
    stepsize_offset: float,
    history_points: int,
) -> SGMRun:
    """Run averaged full-grid SGM with eta_t=eta0/sqrt(t+t0)."""
    iterations = optimizer_value_budget // grid.size
    checkpoints = long_run.checkpoints(iterations, history_points)
    checkpoint_set = set(checkpoints.tolist())
    y = initial_y.copy()
    uniform_average = y.copy()
    iterates: list[np.ndarray] = []
    elapsed: list[float] = []

    start = time.perf_counter()
    for iteration in range(1, iterations + 1):
        uniform_average += (y - uniform_average) / iteration
        residual_r, residual_q = base.residual_arrays(y, grid)
        errors = np.hypot(residual_r, residual_q)
        flat_active = int(np.argmax(errors))
        frequency_index, delay_index = np.unravel_index(
            flat_active,
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
            iterates.append(uniform_average.copy())
            elapsed.append(time.perf_counter() - start)

    return SGMRun(
        optimizer_value_queries=checkpoints * grid.size,
        optimizer_elapsed_seconds=np.asarray(elapsed, dtype=np.float64),
        iterates=iterates,
        final_point=y,
    )


def median_smax_history(
    source_rows: list[dict[str, str]],
    subset_size: int,
    theta: float,
) -> tuple[list[dict[str, object]], np.ndarray, np.ndarray, np.ndarray]:
    curve = f"Virtual age theta={theta:g}, m={subset_size}"
    selected = [row for row in source_rows if row["curve"] == curve]
    if not selected:
        raise ValueError(f"No confirmed SMax rows found for {curve}.")
    seeds = sorted({int(row["seed"]) for row in selected})
    runs = [
        sorted(
            [row for row in selected if int(row["seed"]) == seed],
            key=lambda row: int(row["total_value_queries"]),
        )
        for seed in seeds
    ]
    query_matrix = np.asarray(
        [[int(row["total_value_queries"]) for row in run] for run in runs],
        dtype=np.int64,
    )
    runtime_matrix = np.asarray(
        [
            [float(row["runtime_including_wls_seconds"]) for row in run]
            for run in runs
        ],
        dtype=np.float64,
    )
    value_matrix = np.asarray(
        [
            [float(row["validation_max_error"]) for row in run]
            for run in runs
        ],
        dtype=np.float64,
    )
    if not np.all(query_matrix == query_matrix[0]):
        raise ValueError("Confirmed SMax runs do not share query checkpoints.")

    rows: list[dict[str, object]] = []
    for run in runs:
        for row in run:
            rows.append(
                {
                    "method": f"SMax, m={subset_size}",
                    "seed": int(row["seed"]),
                    "total_value_queries": int(row["total_value_queries"]),
                    "runtime_including_wls_seconds": float(
                        row["runtime_including_wls_seconds"]
                    ),
                    "validation_max_error": float(
                        row["validation_max_error"]
                    ),
                }
            )
    return (
        rows,
        query_matrix[0],
        np.median(runtime_matrix, axis=0),
        value_matrix,
    )


def plot_comparison(
    path: Path,
    smax_queries: np.ndarray,
    smax_runtime: np.ndarray,
    smax_values: np.ndarray,
    sgm_queries: np.ndarray,
    sgm_runtime: np.ndarray,
    sgm_values: np.ndarray,
    socp_queries: np.ndarray,
    socp_runtime: np.ndarray,
    socp_values: np.ndarray,
    x_kind: str,
) -> None:
    if x_kind == "queries":
        smax_x = smax_queries
        sgm_x = sgm_queries
        socp_x = socp_queries
        x_label = "Total component-value queries"
    elif x_kind == "runtime":
        smax_x = smax_runtime
        sgm_x = sgm_runtime
        socp_x = socp_runtime
        x_label = "Runtime including WLS (seconds)"
    else:
        raise ValueError(f"Unknown x_kind={x_kind}.")

    fig, axis = plt.subplots(figsize=(7.1, 4.5), constrained_layout=True)
    smax_median = np.median(smax_values, axis=0)
    axis.plot(
        smax_x,
        smax_median,
        color="#0072B2",
        linewidth=2.1,
        label="SMax, virtual age, $m=4096$",
    )
    axis.fill_between(
        smax_x,
        np.quantile(smax_values, 0.25, axis=0),
        np.quantile(smax_values, 0.75, axis=0),
        color="#0072B2",
        alpha=0.16,
        linewidth=0,
    )
    axis.plot(
        sgm_x,
        sgm_values,
        color="#D55E00",
        linewidth=2.0,
        marker="o",
        markersize=3.2,
        label="Full-grid SGM, virtual age",
    )
    axis.plot(
        socp_x,
        socp_values,
        color="#555555",
        linewidth=1.8,
        marker="s",
        markersize=3.0,
        drawstyle="steps-post",
        label="Restricted-SOCP exchange",
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel(x_label)
    axis.set_ylabel("Validation maximum complex error")
    axis.grid(True, which="both", alpha=0.22)
    axis.legend(fontsize=8.4)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_combined(
    path: Path,
    plot_data: tuple[np.ndarray, ...],
) -> None:
    (
        smax_queries,
        smax_runtime,
        smax_values,
        sgm_queries,
        sgm_runtime,
        sgm_values,
        socp_queries,
        socp_runtime,
        socp_values,
    ) = plot_data
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.8, 4.15),
        constrained_layout=True,
        sharey=True,
    )
    for axis, x_kind in zip(axes, ("queries", "runtime")):
        if x_kind == "queries":
            x_values = (smax_queries, sgm_queries, socp_queries)
            x_label = "Total component-value queries"
        else:
            x_values = (smax_runtime, sgm_runtime, socp_runtime)
            x_label = "Runtime including WLS (seconds)"
        smax_median = np.median(smax_values, axis=0)
        axis.plot(
            x_values[0],
            smax_median,
            color="#0072B2",
            linewidth=2.1,
            label="SMax, virtual age, $m=4096$",
        )
        axis.fill_between(
            x_values[0],
            np.quantile(smax_values, 0.25, axis=0),
            np.quantile(smax_values, 0.75, axis=0),
            color="#0072B2",
            alpha=0.16,
            linewidth=0,
        )
        axis.plot(
            x_values[1],
            sgm_values,
            color="#D55E00",
            linewidth=2.0,
            marker="o",
            markersize=3.0,
            label="Full-grid SGM, virtual age",
        )
        axis.plot(
            x_values[2],
            socp_values,
            color="#555555",
            linewidth=1.8,
            marker="s",
            markersize=2.8,
            drawstyle="steps-post",
            label="Restricted-SOCP exchange",
        )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel(x_label)
        axis.grid(True, which="both", alpha=0.22)
    axes[0].set_ylabel("Validation maximum complex error")
    axes[0].legend(fontsize=7.9)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        stored_plot.update_results(
            args.output_dir,
            args.lse_results_dir,
            args.smax_results_dir,
            args.subset_size,
        )
        print(f"Recreated stored plots in {args.output_dir}", flush=True)
        return
    if args.sgm_timing_repeats < 1:
        raise ValueError("sgm-timing-repeats must be positive.")

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
    wls_training = base.max_error(wls_y, training_grid)
    wls_runtime = time.perf_counter() - wls_start
    coefficient_bound = max(1.0, 2.0 * float(np.max(np.abs(wls_y))))
    component_lipschitz_bound = base.max_feature_norm(training_grid)
    eta0 = args.eta_scale * wls_training / component_lipschitz_bound
    stepsize_offset, first_stepsize = long_run.virtual_age_offset(
        eta0,
        component_lipschitz_bound,
        wls_training,
        args.theta,
    )

    source_config = json.loads(
        (args.smax_results_dir / "run_config.json").read_text(encoding="utf-8")
    )
    socp_source_config = json.loads(
        (args.socp_results_dir / "run_config.json").read_text(encoding="utf-8")
    )
    socp_arguments = socp_source_config["arguments"]
    optimizer_value_budget = int(source_config["optimizer_value_budget"])
    source_rows = read_rows(args.smax_results_dir / "history.csv")
    (
        smax_rows,
        smax_queries,
        smax_runtime,
        smax_values,
    ) = median_smax_history(source_rows, args.subset_size, args.theta)

    print("Running full-grid SGM timing repeats...", flush=True)
    sgm_runs: list[SGMRun] = []
    for repeat in range(args.sgm_timing_repeats):
        run = full_grid_virtual_age_sgm(
            training_grid,
            wls_y,
            coefficient_bound,
            optimizer_value_budget,
            eta0,
            stepsize_offset,
            args.history_points,
        )
        sgm_runs.append(run)
        print(
            f"  repeat {repeat + 1}: "
            f"{run.optimizer_elapsed_seconds[-1]:.3f}s",
            flush=True,
        )
    sgm_values = long_run.evaluate_iterates(
        sgm_runs[0].iterates,
        validation_grid,
    )
    sgm_runtime = np.median(
        np.asarray(
            [run.optimizer_elapsed_seconds for run in sgm_runs],
            dtype=np.float64,
        ),
        axis=0,
    ) + wls_runtime
    sgm_queries = sgm_runs[0].optimizer_value_queries + training_grid.size

    print("Running Restricted-SOCP exchange with incumbent retention...", flush=True)
    socp = followup.restricted_socp_exchange(
        training_grid,
        wls_y,
        coefficient_bound,
        initial_top=int(socp_arguments["socp_initial_top"]),
        stratified_frequency_count=int(
            socp_arguments["socp_stratified_frequency_count"]
        ),
        stratified_delay_count=int(
            socp_arguments["socp_stratified_delay_count"]
        ),
        batch_size=int(socp_arguments["socp_batch_size"]),
        max_iterations=int(socp_arguments["socp_max_iterations"]),
        absolute_tolerance=float(
            socp_arguments["socp_absolute_tolerance"]
        ),
        relative_tolerance=float(
            socp_arguments["socp_relative_tolerance"]
        ),
    )
    if len(socp.best_iterates) != len(socp.history):
        raise RuntimeError("SOCP incumbent/history length mismatch.")
    socp_values = long_run.evaluate_iterates(
        socp.best_iterates,
        validation_grid,
    )
    socp_queries = np.asarray(
        [int(row["value_queries"]) for row in socp.history],
        dtype=np.int64,
    ) + training_grid.size
    socp_runtime = np.asarray(
        [float(row["elapsed_seconds"]) for row in socp.history],
        dtype=np.float64,
    ) + wls_runtime

    history_rows: list[dict[str, object]] = list(smax_rows)
    for position, value in enumerate(sgm_values):
        history_rows.append(
            {
                "method": "Full-grid SGM",
                "seed": "",
                "total_value_queries": int(sgm_queries[position]),
                "runtime_including_wls_seconds": float(sgm_runtime[position]),
                "validation_max_error": float(value),
            }
        )
    for position, value in enumerate(socp_values):
        history_rows.append(
            {
                "method": "Restricted-SOCP exchange",
                "seed": "",
                "total_value_queries": int(socp_queries[position]),
                "runtime_including_wls_seconds": float(socp_runtime[position]),
                "validation_max_error": float(value),
            }
        )

    final_rows: list[dict[str, object]] = [
        {
            "method": f"SMax, m={args.subset_size}",
            "runs": smax_values.shape[0],
            "validation_final": float(np.median(smax_values[:, -1])),
            "total_value_queries": int(smax_queries[-1]),
            "runtime_including_wls_seconds": float(smax_runtime[-1]),
        },
        {
            "method": "Full-grid SGM",
            "runs": 1,
            "validation_final": float(sgm_values[-1]),
            "total_value_queries": int(sgm_queries[-1]),
            "runtime_including_wls_seconds": float(sgm_runtime[-1]),
        },
        {
            "method": "Restricted-SOCP exchange",
            "runs": 1,
            "validation_final": float(socp_values[-1]),
            "total_value_queries": int(socp_queries[-1]),
            "runtime_including_wls_seconds": float(socp_runtime[-1]),
        },
    ]
    write_rows(args.output_dir / "history.csv", history_rows)
    write_rows(args.output_dir / "final_summary.csv", final_rows)

    socp_rows: list[dict[str, object]] = []
    for position, row in enumerate(socp.history):
        enriched = dict(row)
        enriched["total_value_queries"] = int(socp_queries[position])
        enriched["runtime_including_wls_seconds"] = float(
            socp_runtime[position]
        )
        enriched["validation_max_error"] = float(socp_values[position])
        socp_rows.append(enriched)
    write_rows(args.output_dir / "socp_history.csv", socp_rows)
    np.savez_compressed(
        args.output_dir / "coefficients.npz",
        wls=wls_y,
        full_grid_sgm_average=sgm_runs[0].iterates[-1],
        full_grid_sgm_last=sgm_runs[0].final_point,
        restricted_socp=socp.iterate,
    )

    config = {
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "training_components": training_grid.size,
        "validation_components": validation_grid.size,
        "optimizer_value_budget": optimizer_value_budget,
        "common_wls": {
            "runtime_seconds": wls_runtime,
            "total_value_queries": training_grid.size,
        },
        "stepsize": {
            "eta0": eta0,
            "theta": args.theta,
            "stepsize_offset": stepsize_offset,
            "first_stepsize": first_stepsize,
            "rule": "eta_t=eta0/sqrt(t+t0)",
        },
        "smax": {
            "source": str(args.smax_results_dir),
            "confirmatory_seeds": sorted(
                {int(row["seed"]) for row in smax_rows}
            ),
        },
        "sgm": {
            "full_grid": True,
            "iterations": int(
                sgm_runs[0].optimizer_value_queries[-1] // training_grid.size
            ),
            "timing_repeats_seconds": [
                float(run.optimizer_elapsed_seconds[-1]) for run in sgm_runs
            ],
        },
        "restricted_socp": {
            "source": str(args.socp_results_dir),
            "arguments": {
                key: socp_arguments[key]
                for key in (
                    "socp_initial_top",
                    "socp_stratified_frequency_count",
                    "socp_stratified_delay_count",
                    "socp_batch_size",
                    "socp_max_iterations",
                    "socp_absolute_tolerance",
                    "socp_relative_tolerance",
                )
            },
            "status": socp.status,
            "master_solves": socp.master_solves,
            "active_constraints": socp.active_constraints,
            "training_lower_bound": socp.lower_bound,
            "training_upper_bound": socp.upper_bound,
        },
        "accounting": {
            "query_axis": (
                "total component-value queries including the common WLS "
                "training-grid scan"
            ),
            "runtime_axis": "measured runtime including WLS",
            "validation": "offline and uncharged",
            "socp": (
                "includes initial active-set scan, every full-grid separation "
                "scan, and every restricted Clarabel master solve"
            ),
        },
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )
    stored_plot.update_results(
        args.output_dir,
        args.lse_results_dir,
        args.smax_results_dir,
        args.subset_size,
        refresh_lse=True,
    )
    print(f"Saved outputs to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
