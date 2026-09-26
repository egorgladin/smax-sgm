"""Matched-budget and WLS-aware SMax experiment for the VFD benchmark.

The experiment answers three questions left by the initial run:

* what happens when SMax receives the same component-value budget as the
  tensorized LSE--L-BFGS baseline;
* whether subset sizes larger than 1024 improve the fixed-query tradeoff;
* whether a virtual-age stepsize can preserve the useful WLS warm start
  without relying only on a safe final-output rule.

All SMax trajectories use the Numba kernel in ``run_followup_experiment.py``.
Validation is independent and is never used for stepsize or iterate selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numba
import numpy as np
import scipy

import run_experiment as base
import run_followup_experiment as followup


LSE_MULTIPLIERS = (1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4)


@dataclass
class SMaxRecord:
    family: str
    average: str
    subset_size: int
    seed: int
    theta: float | None
    eta0: float
    stepsize_offset: float
    optimizer_value_queries: np.ndarray
    component_gradient_rows: np.ndarray
    optimizer_elapsed_seconds: np.ndarray
    iterates: list[np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frequency-count", type=int, default=1000)
    parser.add_argument("--delay-count", type=int, default=200)
    parser.add_argument("--validation-frequency-count", type=int, default=2001)
    parser.add_argument("--validation-delay-count", type=int, default=401)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument(
        "--subset-sizes",
        type=int,
        nargs="+",
        default=[1024, 4096, 8192],
    )
    parser.add_argument("--virtual-subset-size", type=int, default=4096)
    parser.add_argument(
        "--virtual-thetas",
        type=float,
        nargs="+",
        default=[4.0],
        help=(
            "Global-L cap multipliers. The default 4 was selected on three "
            "development seeds and is frozen for the confirmatory seeds."
        ),
    )
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=100)
    parser.add_argument("--history-points", type=int, default=18)
    parser.add_argument("--eta-scale", type=float, default=8.0)
    parser.add_argument("--lse-timing-repeats", type=int, default=3)
    parser.add_argument(
        "--optimizer-value-budget",
        type=int,
        default=0,
        help=(
            "SMax optimizer-value budget. Zero matches the actual LSE--L-BFGS "
            "component-value count."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("results_long_smax"),
    )
    return parser.parse_args()


def checkpoints(iterations: int, count: int) -> np.ndarray:
    if iterations <= 0:
        raise ValueError("The optimizer budget must permit at least one step.")
    raw = np.geomspace(1, iterations, num=max(2, count))
    result = np.unique(np.rint(raw).astype(np.int64))
    if result[-1] != iterations:
        result = np.append(result, iterations)
    return result


def virtual_age_offset(
    eta0: float,
    component_lipschitz_bound: float,
    warm_value: float,
    theta: float,
) -> tuple[float, float]:
    """Return t0 and eta_1 for eta_t=eta0/sqrt(t+t0).

    The clock is shifted until the worst-case linearized residual change of
    the first step is at most ``theta * warm_value``:

        eta_1 * L^2 <= theta * warm_value.
    """
    if theta <= 0.0:
        raise ValueError("virtual-age theta must be positive.")
    target_first_step = (
        theta
        * warm_value
        / (component_lipschitz_bound * component_lipschitz_bound)
    )
    offset = max(0.0, (eta0 / target_first_step) ** 2 - 1.0)
    first_step = eta0 / math.sqrt(1.0 + offset)
    return offset, first_step


def evaluate_iterates(
    iterates: list[np.ndarray],
    grid: base.VFDGrid,
) -> np.ndarray:
    return np.asarray(
        [base.max_error(iterate, grid) for iterate in iterates],
        dtype=np.float64,
    )


def max_value_and_active_gradient_norm(
    y: np.ndarray,
    grid: base.VFDGrid,
) -> tuple[float, float]:
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
    return (
        float(errors[frequency_index, delay_index]),
        float(np.linalg.norm(gradient)),
    )


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty table to {path}.")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def label_for(family: str, subset_size: int, theta: float | None) -> str:
    if family == "Original":
        return f"Original schedule, m={subset_size}"
    if theta is None:
        raise ValueError("Virtual-age records require theta.")
    return f"Virtual age theta={theta:g}, m={subset_size}"


def group_histories(
    history_rows: list[dict[str, object]],
) -> dict[str, list[list[dict[str, object]]]]:
    per_run: dict[tuple[str, int], list[dict[str, object]]] = {}
    for row in history_rows:
        key = (str(row["curve"]), int(row["seed"]))
        per_run.setdefault(key, []).append(row)
    grouped: dict[str, list[list[dict[str, object]]]] = {}
    for (curve, _), rows in per_run.items():
        rows.sort(key=lambda row: int(row["optimizer_value_queries"]))
        grouped.setdefault(curve, []).append(rows)
    return grouped


def plot_query_panels(
    path: Path,
    history_rows: list[dict[str, object]],
    lse_queries: np.ndarray,
    lse_validation: np.ndarray,
    wls_value: float,
    socp_value: float,
    subset_sizes: list[int],
    virtual_subset_size: int,
    virtual_thetas: list[float],
) -> None:
    grouped = group_histories(history_rows)
    colors = {
        1024: "#0072B2",
        4096: "#D55E00",
        8192: "#009E73",
        16384: "#CC79A7",
    }
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.2, 4.5),
        sharey=True,
        constrained_layout=True,
    )

    for subset_size in subset_sizes:
        curve = label_for("Original", subset_size, None)
        if curve not in grouped:
            continue
        runs = grouped[curve]
        x_values = np.asarray(
            [int(row["optimizer_value_queries"]) for row in runs[0]],
            dtype=np.int64,
        )
        values = np.asarray(
            [
                [float(row["validation_max_error"]) for row in run]
                for run in runs
            ],
            dtype=np.float64,
        )
        color = colors.get(subset_size)
        axes[0].plot(
            x_values,
            np.median(values, axis=0),
            color=color,
            linewidth=2.0,
            label=rf"$m={subset_size}$",
        )
        axes[0].fill_between(
            x_values,
            np.quantile(values, 0.25, axis=0),
            np.quantile(values, 0.75, axis=0),
            color=color,
            alpha=0.15,
            linewidth=0,
        )

    original_curve = label_for("Original", virtual_subset_size, None)
    if original_curve in grouped:
        runs = grouped[original_curve]
        x_values = np.asarray(
            [int(row["optimizer_value_queries"]) for row in runs[0]],
            dtype=np.int64,
        )
        values = np.asarray(
            [
                [float(row["validation_max_error"]) for row in run]
                for run in runs
            ],
            dtype=np.float64,
        )
        axes[1].plot(
            x_values,
            np.median(values, axis=0),
            color="#555555",
            linestyle="--",
            linewidth=1.8,
            label="Original schedule",
        )

    theta_colors = plt.cm.viridis(
        np.linspace(0.15, 0.85, len(virtual_thetas))
    )
    for theta, color in zip(virtual_thetas, theta_colors, strict=True):
        curve = label_for("Virtual age", virtual_subset_size, theta)
        if curve not in grouped:
            continue
        runs = grouped[curve]
        x_values = np.asarray(
            [int(row["optimizer_value_queries"]) for row in runs[0]],
            dtype=np.int64,
        )
        values = np.asarray(
            [
                [float(row["validation_max_error"]) for row in run]
                for run in runs
            ],
            dtype=np.float64,
        )
        axes[1].plot(
            x_values,
            np.median(values, axis=0),
            color=color,
            linewidth=2.0,
            label=rf"Virtual age, $\theta={theta:g}$",
        )
        axes[1].fill_between(
            x_values,
            np.quantile(values, 0.25, axis=0),
            np.quantile(values, 0.75, axis=0),
            color=color,
            alpha=0.15,
            linewidth=0,
        )

    for axis in axes:
        axis.plot(
            lse_queries,
            lse_validation,
            color="#000000",
            linestyle="-.",
            linewidth=1.6,
            label="LSE--L-BFGS",
        )
        axis.axhline(
            wls_value,
            color="#E69F00",
            linestyle=":",
            linewidth=1.2,
            label="WLS",
        )
        axis.axhline(
            socp_value,
            color="#777777",
            linestyle="--",
            linewidth=1.2,
            label="Restricted SOCP",
        )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.grid(True, which="both", alpha=0.2)
        axis.set_xlabel("Optimizer component-value queries")
        axis.legend(fontsize=8.0)

    axes[0].set_title("Long-run subset-size comparison")
    axes[1].set_title(r"WLS-aware virtual-age schedule")
    axes[0].set_ylabel("Validation maximum complex error")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_runtime(
    path: Path,
    history_rows: list[dict[str, object]],
    lse_elapsed: np.ndarray,
    lse_validation: np.ndarray,
    socp_value: float,
    primary_subset_size: int,
    primary_theta: float,
) -> None:
    grouped = group_histories(history_rows)
    selected = [
        (
            label_for("Original", primary_subset_size, None),
            "Original schedule",
            "#555555",
            "--",
        ),
        (
            label_for("Virtual age", primary_subset_size, primary_theta),
            rf"Virtual age, $\theta={primary_theta:g}$",
            "#D55E00",
            "-",
        ),
    ]
    fig, axis = plt.subplots(figsize=(7.1, 4.5), constrained_layout=True)
    for curve, label, color, linestyle in selected:
        if curve not in grouped:
            continue
        runs = grouped[curve]
        x_values = np.median(
            np.asarray(
                [
                    [float(row["optimizer_runtime_seconds"]) for row in run]
                    for run in runs
                ],
                dtype=np.float64,
            ),
            axis=0,
        )
        values = np.asarray(
            [
                [float(row["validation_max_error"]) for row in run]
                for run in runs
            ],
            dtype=np.float64,
        )
        axis.plot(
            x_values,
            np.median(values, axis=0),
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=2.0,
        )
        axis.fill_between(
            x_values,
            np.quantile(values, 0.25, axis=0),
            np.quantile(values, 0.75, axis=0),
            color=color,
            alpha=0.15,
            linewidth=0,
        )
    axis.plot(
        lse_elapsed,
        lse_validation,
        color="#000000",
        linestyle="-.",
        linewidth=1.7,
        label="LSE--L-BFGS",
    )
    axis.axhline(
        socp_value,
        color="#777777",
        linestyle="--",
        linewidth=1.2,
        label="Restricted SOCP value",
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Optimizer runtime (seconds)")
    axis.set_ylabel("Validation maximum complex error")
    axis.grid(True, which="both", alpha=0.2)
    axis.legend(fontsize=8.2)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def aggregate_rows(
    final_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in final_rows:
        grouped.setdefault(str(row["method"]), []).append(row)
    result: list[dict[str, object]] = []
    for method, rows in grouped.items():
        validation = np.asarray(
            [float(row["validation_max_error"]) for row in rows]
        )
        training = np.asarray(
            [float(row["training_max_error"]) for row in rows]
        )
        runtime = np.asarray(
            [float(row["runtime_including_wls_seconds"]) for row in rows]
        )
        result.append(
            {
                "method": method,
                "runs": len(rows),
                "validation_q25": float(np.quantile(validation, 0.25)),
                "validation_median": float(np.median(validation)),
                "validation_q75": float(np.quantile(validation, 0.75)),
                "training_median": float(np.median(training)),
                "runtime_including_wls_median_seconds": float(
                    np.median(runtime)
                ),
                "optimizer_value_queries_median": int(
                    np.median(
                        [
                            int(row["optimizer_value_queries"])
                            for row in rows
                        ]
                    )
                ),
                "total_value_queries_median": int(
                    np.median(
                        [int(row["total_value_queries"]) for row in rows]
                    )
                ),
                "component_gradient_rows_median": int(
                    np.median(
                        [
                            int(row["component_gradient_rows"])
                            for row in rows
                        ]
                    )
                ),
            }
        )
    return result


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if 64 in args.subset_sizes:
        raise ValueError("m=64 was intentionally dropped from this experiment.")
    if args.virtual_subset_size not in args.subset_sizes:
        raise ValueError("virtual-subset-size must also appear in subset-sizes.")
    if args.seeds < 1 or args.history_points < 2:
        raise ValueError("seeds >= 1 and history-points >= 2 are required.")

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
    (
        wls_training,
        wls_active_subgradient_norm,
    ) = max_value_and_active_gradient_norm(wls_y, training_grid)
    wls_runtime = time.perf_counter() - wls_start
    wls_validation = base.max_error(wls_y, validation_grid)
    coefficient_bound = max(1.0, 2.0 * float(np.max(np.abs(wls_y))))
    component_lipschitz_bound = base.max_feature_norm(training_grid)
    eta0 = (
        args.eta_scale
        * wls_training
        / component_lipschitz_bound
    )
    if args.virtual_thetas is None:
        args.virtual_thetas = [
            (
                component_lipschitz_bound
                * component_lipschitz_bound
                / (
                    wls_active_subgradient_norm
                    * wls_active_subgradient_norm
                )
            )
        ]
        theta_source = (
            "derived from eta_1*||g_active(WLS)||^2=h(WLS); no theta pilot"
        )
    else:
        theta_source = "frozen development-sensitivity values"

    print("Running LSE--L-BFGS baseline and timing repeats...", flush=True)
    lse_runs: list[base.Run] = []
    for repeat in range(args.lse_timing_repeats):
        run = base.lse_lbfgs(
            training_grid,
            wls_y,
            coefficient_bound,
            wls_training,
        )
        lse_runs.append(run)
        print(
            f"  repeat {repeat + 1}: {run.value_queries[-1]:,} values, "
            f"{run.runtime_seconds:.3f}s",
            flush=True,
        )
    lse_run = lse_runs[0]
    lse_validation = evaluate_iterates(lse_run.iterates, validation_grid)
    lse_training_final = base.max_error(
        lse_run.iterates[-1],
        training_grid,
    )
    lse_runtime_median = float(
        np.median([run.runtime_seconds for run in lse_runs])
    )
    lse_curve_elapsed = (
        lse_run.elapsed_seconds
        * lse_runtime_median
        / lse_run.runtime_seconds
    )
    optimizer_value_budget = (
        int(lse_run.value_queries[-1])
        if args.optimizer_value_budget == 0
        else args.optimizer_value_budget
    )

    print("Compiling the largest SMax kernel signature...", flush=True)
    jit_seconds = followup.warm_compiled_kernel(
        training_grid,
        max(args.subset_sizes),
    )
    kernel_checks = followup.kernel_self_checks(training_grid, wls_y)
    if (
        not kernel_checks["sampled_active_index_match"]
        or not kernel_checks["sampler_unique"]
        or not kernel_checks["sampler_in_range"]
        or float(kernel_checks["sampled_response_max_absolute_error"]) > 1e-10
    ):
        raise RuntimeError(f"Compiled-kernel check failed: {kernel_checks}")

    virtual_parameters: dict[float, tuple[float, float]] = {}
    for theta in args.virtual_thetas:
        virtual_parameters[theta] = virtual_age_offset(
            eta0,
            component_lipschitz_bound,
            wls_training,
            theta,
        )
        offset, first_step = virtual_parameters[theta]
        print(
            f"theta={theta:g}: virtual age={offset:.1f}, "
            f"eta_1={first_step:.3e}",
            flush=True,
        )

    jobs: list[tuple[str, int, int, float | None]] = []
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    for subset_size in args.subset_sizes:
        for seed in seeds:
            jobs.append(("Original", subset_size, seed, None))
    for theta in args.virtual_thetas:
        for seed in seeds:
            jobs.append(
                ("Virtual age", args.virtual_subset_size, seed, theta)
            )
    np.random.default_rng(20260729).shuffle(jobs)

    stored: list[tuple[SMaxRecord, list[np.ndarray]]] = []
    coefficients: dict[str, np.ndarray] = {
        "wls": wls_y,
        "lse_lbfgs": lse_run.iterates[-1],
    }
    for job_position, (family, subset_size, seed, theta) in enumerate(
        jobs,
        start=1,
    ):
        iterations = optimizer_value_budget // subset_size
        run_checkpoints = checkpoints(iterations, args.history_points)
        if family == "Original":
            stepsize_offset = 0.0
        else:
            if theta is None:
                raise AssertionError("Missing theta for virtual-age job.")
            stepsize_offset = virtual_parameters[theta][0]
        print(
            f"[{job_position}/{len(jobs)}] {family}, m={subset_size}, "
            f"seed={seed}"
            + ("" if theta is None else f", theta={theta:g}"),
            flush=True,
        )
        trajectory = followup.compiled_smax_trajectory(
            training_grid,
            wls_y,
            coefficient_bound,
            optimizer_value_budget,
            eta0,
            subset_size,
            seed,
            math.ceil(training_grid.size / subset_size),
            run_checkpoints,
            stepsize_power=0.5,
            stepsize_offset=stepsize_offset,
        )
        if family == "Original":
            average = "uniform"
            iterates = trajectory.uniform_iterates
        else:
            average = "uniform"
            iterates = trajectory.uniform_iterates
        record = SMaxRecord(
            family=family,
            average=average,
            subset_size=subset_size,
            seed=seed,
            theta=theta,
            eta0=eta0,
            stepsize_offset=stepsize_offset,
            optimizer_value_queries=trajectory.sample_queries,
            component_gradient_rows=(
                trajectory.sample_queries // subset_size
            ),
            optimizer_elapsed_seconds=trajectory.elapsed_seconds,
            iterates=iterates,
        )
        stored.append((record, iterates))
        theta_key = "" if theta is None else f"_theta{theta:g}"
        coefficients[
            f"{family.lower().replace(' ', '_')}_m{subset_size}"
            f"_seed{seed}{theta_key}"
        ] = iterates[-1]

    print("Evaluating stored iterates on the independent grid...", flush=True)
    history_rows: list[dict[str, object]] = []
    final_rows: list[dict[str, object]] = []
    for record, iterates in stored:
        values = evaluate_iterates(iterates, validation_grid)
        curve = label_for(
            record.family,
            record.subset_size,
            record.theta,
        )
        for position, validation_value in enumerate(values):
            history_rows.append(
                {
                    "curve": curve,
                    "family": record.family,
                    "average": record.average,
                    "subset_size": record.subset_size,
                    "seed": record.seed,
                    "theta": "" if record.theta is None else record.theta,
                    "optimizer_value_queries": int(
                        record.optimizer_value_queries[position]
                    ),
                    "total_value_queries": int(
                        training_grid.size
                        + record.optimizer_value_queries[position]
                    ),
                    "component_gradient_rows": int(
                        record.component_gradient_rows[position]
                    ),
                    "optimizer_runtime_seconds": float(
                        record.optimizer_elapsed_seconds[position]
                    ),
                    "runtime_including_wls_seconds": float(
                        wls_runtime
                        + record.optimizer_elapsed_seconds[position]
                    ),
                    "validation_max_error": float(validation_value),
                }
            )
        final = iterates[-1]
        final_rows.append(
            {
                "method": curve,
                "family": record.family,
                "average": record.average,
                "subset_size": record.subset_size,
                "seed": record.seed,
                "theta": "" if record.theta is None else record.theta,
                "eta0": record.eta0,
                "stepsize_offset": record.stepsize_offset,
                "first_stepsize": record.eta0
                / math.sqrt(1.0 + record.stepsize_offset),
                "training_max_error": base.max_error(final, training_grid),
                "validation_max_error": float(values[-1]),
                "optimizer_value_queries": int(
                    record.optimizer_value_queries[-1]
                ),
                "total_value_queries": int(
                    training_grid.size
                    + record.optimizer_value_queries[-1]
                ),
                "component_gradient_rows": int(
                    record.component_gradient_rows[-1]
                ),
                "optimizer_runtime_seconds": float(
                    record.optimizer_elapsed_seconds[-1]
                ),
                "runtime_including_wls_seconds": float(
                    wls_runtime + record.optimizer_elapsed_seconds[-1]
                ),
            }
        )

    socp_config_path = Path(__file__).with_name(
        "results_followup"
    ) / "run_config.json"
    with socp_config_path.open(encoding="utf-8") as handle:
        followup_config = json.load(handle)
    socp_config = followup_config["restricted_socp_exchange"]
    socp_training = float(socp_config["training_upper_bound"])
    socp_validation = float(socp_config["validation_max_error"])
    socp_summary_path = Path(__file__).with_name(
        "results_followup"
    ) / "final_summary.csv"
    with socp_summary_path.open(newline="", encoding="utf-8") as handle:
        socp_rows = list(csv.DictReader(handle))
    socp_row = next(
        row
        for row in socp_rows
        if row["method"] == "Restricted-SOCP exchange"
    )
    socp_optimizer_queries = int(socp_row["total_value_queries"])
    socp_runtime_including_wls = float(socp_row["runtime_seconds"])
    followup_wls_runtime = float(followup_config["wls_runtime_seconds"])
    socp_optimizer_runtime = (
        socp_runtime_including_wls - followup_wls_runtime
    )

    deterministic_rows = [
        {
            "method": "WLS",
            "family": "Deterministic",
            "average": "",
            "subset_size": "",
            "seed": "",
            "theta": "",
            "eta0": "",
            "stepsize_offset": "",
            "first_stepsize": "",
            "training_max_error": wls_training,
            "validation_max_error": wls_validation,
            "optimizer_value_queries": 0,
            "total_value_queries": training_grid.size,
            "component_gradient_rows": 0,
            "optimizer_runtime_seconds": 0.0,
            "runtime_including_wls_seconds": wls_runtime,
        },
        {
            "method": "LSE--L-BFGS",
            "family": "Deterministic",
            "average": "",
            "subset_size": "",
            "seed": "",
            "theta": "",
            "eta0": "",
            "stepsize_offset": "",
            "first_stepsize": "",
            "training_max_error": lse_training_final,
            "validation_max_error": float(lse_validation[-1]),
            "optimizer_value_queries": int(lse_run.value_queries[-1]),
            "total_value_queries": int(
                training_grid.size + lse_run.value_queries[-1]
            ),
            "component_gradient_rows": int(lse_run.value_queries[-1]),
            "optimizer_runtime_seconds": lse_runtime_median,
            "runtime_including_wls_seconds": (
                wls_runtime + lse_runtime_median
            ),
        },
        {
            "method": "Restricted-SOCP exchange",
            "family": "Deterministic",
            "average": "",
            "subset_size": "",
            "seed": "",
            "theta": "",
            "eta0": "",
            "stepsize_offset": "",
            "first_stepsize": "",
            "training_max_error": socp_training,
            "validation_max_error": socp_validation,
            "optimizer_value_queries": socp_optimizer_queries,
            "total_value_queries": (
                training_grid.size + socp_optimizer_queries
            ),
            "component_gradient_rows": 0,
            "optimizer_runtime_seconds": socp_optimizer_runtime,
            "runtime_including_wls_seconds": socp_runtime_including_wls,
        },
    ]
    final_rows.extend(deterministic_rows)

    write_rows(args.output_dir / "history.csv", history_rows)
    write_rows(args.output_dir / "final_summary.csv", final_rows)
    aggregate = aggregate_rows(final_rows)
    write_rows(args.output_dir / "aggregate_summary.csv", aggregate)
    np.savez_compressed(args.output_dir / "coefficients.npz", **coefficients)

    primary_theta = args.virtual_thetas[0]
    plot_query_panels(
        args.output_dir / "convergence_vs_value_queries.png",
        history_rows,
        lse_run.value_queries,
        lse_validation,
        wls_validation,
        socp_validation,
        args.subset_sizes,
        args.virtual_subset_size,
        args.virtual_thetas,
    )
    plot_runtime(
        args.output_dir / "convergence_vs_runtime.png",
        history_rows,
        lse_curve_elapsed,
        lse_validation,
        socp_validation,
        args.virtual_subset_size,
        primary_theta,
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
        "optimizer_value_budget": optimizer_value_budget,
        "common_wls": {
            "training_max_error": wls_training,
            "validation_max_error": wls_validation,
            "runtime_seconds": wls_runtime,
            "value_queries_charged_in_total_column": training_grid.size,
        },
        "original_stepsize": {
            "eta0": eta0,
            "rule": "eta_t=eta0/sqrt(t)",
            "eta_scale": args.eta_scale,
        },
        "virtual_age_stepsize": {
            "rule": "eta_t=eta0/sqrt(t+t0), uniform iterate average",
            "component_lipschitz_bound": component_lipschitz_bound,
            "condition": (
                "eta_1 * L^2 <= theta * h(WLS), with equality whenever "
                "the derived offset is positive"
            ),
            "wls_active_subgradient_norm": wls_active_subgradient_norm,
            "default_active_calibration": (
                "eta_1 * ||g_active(WLS)||^2 = h(WLS)"
            ),
            "theta_source": theta_source,
            "parameters": {
                str(theta): {
                    "theta": theta,
                    "stepsize_offset": virtual_parameters[theta][0],
                    "first_stepsize": virtual_parameters[theta][1],
                }
                for theta in args.virtual_thetas
            },
            "selection_data": (
                "none; theta is derived from WLS training quantities"
                if theta_source.startswith("derived")
                else (
                    "theta selected using validation medians on development "
                    "seeds 0--2 and frozen before confirmatory seeds; "
                    "validation within confirmatory runs is offline"
                )
            ),
            "development_sensitivity": {
                "seeds": [0, 1, 2],
                "subset_sizes": [4096, 8192],
                "theta_values": [0.25, 0.5, 1.0, 2.0, 4.0],
                "selected_configuration": {
                    "subset_size": 4096,
                    "theta": 4.0,
                },
                "confirmatory_seeds": [
                    args.seed_start,
                    args.seed_start + args.seeds - 1,
                ],
                "charged_to_each_confirmatory_run": False,
            },
        },
        "lse": {
            "joint_objective_gradient_evaluations": int(
                lse_run.value_queries[-1] // training_grid.size
            ),
            "component_value_queries": int(lse_run.value_queries[-1]),
            "component_gradient_rows": int(lse_run.value_queries[-1]),
            "temperature_name": "log-sum-exp smoothing parameter mu",
            "mu_multipliers_of_wls_max": list(LSE_MULTIPLIERS),
            "mu_values": [
                wls_training * multiplier
                for multiplier in LSE_MULTIPLIERS
            ],
            "final_uniform_approximation_allowance_mu_log_d": (
                wls_training
                * LSE_MULTIPLIERS[-1]
                * math.log(training_grid.size)
            ),
            "temperature_tuned": False,
            "timing_repeats_seconds": [
                run.runtime_seconds for run in lse_runs
            ],
            "timing_median_seconds": lse_runtime_median,
        },
        "compiled_kernel": {
            "jit_seconds_excluded": jit_seconds,
            "timing_includes_per_run_array_and_sampler_allocation": True,
            "checks": kernel_checks,
        },
        "reporting": {
            "primary_axis": "optimizer component-value queries",
            "secondary_axis": "optimizer wall time",
            "component_gradient_rows_reported_separately": True,
            "validation_charged": False,
            "historical_stepsize_pilot_charged_per_run": False,
            "historical_stepsize_pilot_component_values": 2_800_000,
            "historical_stepsize_pilot_note": (
                "four 500,000-value SMax candidates plus four 200,000-value "
                "training-grid scoring scans; paid once in development"
            ),
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "numba": numba.__version__,
            "matplotlib": plt.matplotlib.__version__,
        },
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "thread_environment": {
                name: os.environ.get(name)
                for name in (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
        },
    }
    with (args.output_dir / "run_config.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(config, handle, indent=2)

    print("Final aggregate rows:", flush=True)
    for row in aggregate:
        print(
            f"  {row['method']}: "
            f"validation={float(row['validation_median']):.8e}, "
            f"time={float(row['runtime_including_wls_median_seconds']):.3f}s",
            flush=True,
        )
    print(f"Saved results to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
