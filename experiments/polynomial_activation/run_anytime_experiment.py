"""Polynomial-activation experiment under a common production budget.

SMax, full-grid SGM, and LSE are tuned by the endpoint training objective.
SMax's tuning horizon can be shortened independently, while each method's
production run receives the same component-value cap. Restricted LP exchange
may stop early when its certificate is complete.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import scipy

import run_experiment as base


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=script_dir / "features" / "cifar100_resnet20_q14",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "results" / "anytime_final",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--train-samples", type=int, default=None)
    parser.add_argument("--test-samples", type=int, default=None)
    parser.add_argument("--sample-chunk", type=int, default=1024)
    parser.add_argument("--evaluation-coefficient-block", type=int, default=16)
    parser.add_argument("--tuning-coefficient-block", type=int, default=32)
    parser.add_argument("--functional-box-kappa", type=float, default=4.0)
    parser.add_argument("--functional-grid-points", type=int, default=65_537)
    parser.add_argument("--smax-value-budget", type=int, default=125_000_000)
    parser.add_argument("--other-value-budget", type=int, default=125_000_000)
    parser.add_argument("--target-relative-gap", type=float, default=0.05)
    parser.add_argument(
        "--smax-subset-candidates",
        type=int,
        nargs="+",
        default=[1024, 2048, 4096, 8192, 16384],
    )
    parser.add_argument(
        "--smax-candidate",
        action="append",
        default=None,
        metavar="M,SCALE,RATIO",
        help=(
            "Explicit SMax candidate. Repeat to replace the Cartesian product "
            "of the subset, first-step-scale, and endpoint-ratio grids."
        ),
    )
    parser.add_argument(
        "--smax-first-step-scales",
        type=float,
        nargs="+",
        default=[0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
    )
    parser.add_argument(
        "--smax-endpoint-ratios",
        type=float,
        nargs="+",
        default=[0.075, 0.1, 0.2, 0.35, 0.5, 0.75, 0.95],
    )
    parser.add_argument(
        "--smax-tuning-seeds", type=int, nargs="+", default=[0, 1, 2]
    )
    parser.add_argument(
        "--confirmatory-seeds",
        type=int,
        nargs="+",
        default=list(range(100, 120)),
    )
    parser.add_argument(
        "--sgm-first-step-scales",
        type=float,
        nargs="+",
        default=[0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0],
    )
    parser.add_argument(
        "--sgm-endpoint-ratios",
        type=float,
        nargs="+",
        default=[0.2, 0.35, 0.5, 0.75, 0.95],
    )
    parser.add_argument(
        "--lse-candidate",
        action="append",
        default=None,
        help=(
            "Comma-separated temperature multipliers. Repeat to replace the "
            "default training-only LSE grid."
        ),
    )
    parser.add_argument(
        "--lse-maxiter-candidates", type=int, nargs="+", default=[2, 4, 8]
    )
    parser.add_argument("--history-points", type=int, default=36)
    parser.add_argument("--timing-repeats", type=int, default=2)
    parser.add_argument("--lp-timing-repeats", type=int, default=1)
    parser.add_argument("--exchange-initial-rows", type=int, default=96)
    parser.add_argument("--exchange-add-rows", type=int, default=48)
    parser.add_argument("--exchange-max-iterations", type=int, default=30)
    parser.add_argument("--exchange-absolute-tolerance", type=float, default=1e-7)
    parser.add_argument("--exchange-relative-tolerance", type=float, default=1e-6)
    parser.add_argument("--tail-sample-size", type=int, default=200_000)
    parser.add_argument("--tail-seed", type=int, default=918273)
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Recreate the figure and summary tables from stored history.",
    )
    parser.add_argument(
        "--reuse-tuning-dir",
        type=Path,
        default=None,
        help=(
            "Reuse frozen training-only tuning tables, selected JSON files, "
            "and tuning arrays from an earlier compatible result directory."
        ),
    )
    parser.add_argument(
        "--reuse-baseline-tuning-dir",
        type=Path,
        default=None,
        help=(
            "Tune SMax in this run but reuse frozen full-grid SGM and LSE "
            "training-only tuning from a compatible result directory."
        ),
    )
    return parser.parse_args()


def finalize_args(args: argparse.Namespace) -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    if args.lse_candidate is None:
        args.lse_candidates = [
            [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4],
            [3e-2, 1e-2, 3e-3, 1e-3],
            [1e-2, 3e-3, 1e-3, 3e-4],
            [3e-3, 1e-3, 3e-4, 1e-4],
            [1e-2, 1e-3, 1e-4],
        ]
    else:
        args.lse_candidates = [
            [float(value) for value in item.split(",")]
            for item in args.lse_candidate
        ]
    if args.smax_candidate is None:
        args.smax_candidates = None
    else:
        parsed_candidates: list[tuple[int, float, float]] = []
        for item in args.smax_candidate:
            fields = item.split(",")
            if len(fields) != 3:
                raise ValueError(
                    "Each --smax-candidate must be M,SCALE,RATIO."
                )
            parsed_candidates.append(
                (int(fields[0]), float(fields[1]), float(fields[2]))
            )
        args.smax_candidates = parsed_candidates
    if args.smoke:
        args.train_samples = 256 if args.train_samples is None else args.train_samples
        args.test_samples = 128 if args.test_samples is None else args.test_samples
        args.sample_chunk = min(args.sample_chunk, 64)
        d_smoke = args.train_samples * 100
        args.smax_value_budget = min(args.smax_value_budget, 20_000)
        args.other_value_budget = min(args.other_value_budget, 3 * d_smoke)
        args.smax_subset_candidates = [32, 64, 128]
        args.smax_first_step_scales = [0.5, 2.0]
        args.smax_endpoint_ratios = [0.5, 0.9]
        args.smax_tuning_seeds = [0]
        args.confirmatory_seeds = [100, 101]
        args.sgm_first_step_scales = [0.0625, 0.25]
        args.sgm_endpoint_ratios = [0.75, 0.95]
        args.lse_candidates = [[1e-2], [1e-3]]
        args.lse_maxiter_candidates = [2]
        args.history_points = min(args.history_points, 8)
        args.timing_repeats = 1
        args.lp_timing_repeats = 1
        args.exchange_initial_rows = min(args.exchange_initial_rows, 24)
        args.exchange_add_rows = min(args.exchange_add_rows, 8)
        args.exchange_max_iterations = min(args.exchange_max_iterations, 4)
        args.tail_sample_size = min(args.tail_sample_size, 5_000)
        if args.output_dir == script_dir / "results" / "anytime_final":
            args.output_dir = script_dir / "results" / "anytime_smoke"
    args.feature_dir = args.feature_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.reuse_tuning_dir is not None:
        args.reuse_tuning_dir = args.reuse_tuning_dir.resolve()
    if args.reuse_baseline_tuning_dir is not None:
        args.reuse_baseline_tuning_dir = args.reuse_baseline_tuning_dir.resolve()
    if args.reuse_tuning_dir is not None and args.reuse_baseline_tuning_dir is not None:
        raise ValueError(
            "Use at most one of --reuse-tuning-dir and "
            "--reuse-baseline-tuning-dir."
        )
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    positive_integers = (
        "sample_chunk",
        "evaluation_coefficient_block",
        "tuning_coefficient_block",
        "functional_grid_points",
        "smax_value_budget",
        "other_value_budget",
        "history_points",
        "timing_repeats",
        "lp_timing_repeats",
        "exchange_initial_rows",
        "exchange_add_rows",
        "exchange_max_iterations",
        "tail_sample_size",
    )
    for name in positive_integers:
        if int(getattr(args, name)) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.smax_value_budget > args.other_value_budget:
        raise ValueError("The SMax tuning budget cannot exceed the production cap.")
    if not 0.0 < args.target_relative_gap < 1.0:
        raise ValueError("--target-relative-gap must lie in (0,1).")
    if args.functional_box_kappa <= 0.0:
        raise ValueError("--functional-box-kappa must be positive.")
    if len(set(args.smax_tuning_seeds)) != len(args.smax_tuning_seeds):
        raise ValueError("SMax tuning seeds must be distinct.")
    if len(set(args.confirmatory_seeds)) != len(args.confirmatory_seeds):
        raise ValueError("Confirmatory seeds must be distinct.")
    if set(args.smax_tuning_seeds) & set(args.confirmatory_seeds):
        raise ValueError("Tuning and confirmatory SMax seeds must be disjoint.")
    positive_lists = (
        args.smax_subset_candidates,
        args.smax_first_step_scales,
        args.smax_endpoint_ratios,
        args.sgm_first_step_scales,
        args.sgm_endpoint_ratios,
        args.lse_maxiter_candidates,
    )
    if any(any(float(value) <= 0.0 for value in values) for values in positive_lists):
        raise ValueError("All tuning-grid values must be positive.")
    if any(ratio >= 1.0 for ratio in args.smax_endpoint_ratios):
        raise ValueError("SMax endpoint ratios must be smaller than one.")
    if args.smax_candidates is not None:
        if not args.smax_candidates:
            raise ValueError("At least one explicit SMax candidate is required.")
        if any(
            subset <= 0 or scale <= 0.0 or not 0.0 < ratio < 1.0
            for subset, scale, ratio in args.smax_candidates
        ):
            raise ValueError(
                "Explicit SMax candidates require M>0, SCALE>0, and 0<RATIO<1."
            )
        if len(set(args.smax_candidates)) != len(args.smax_candidates):
            raise ValueError("Explicit SMax candidates must be distinct.")
    if any(ratio >= 1.0 for ratio in args.sgm_endpoint_ratios):
        raise ValueError("SGM endpoint ratios must be smaller than one.")
    if any(any(value <= 0.0 for value in candidate) for candidate in args.lse_candidates):
        raise ValueError("LSE temperature multipliers must be positive.")


def schedule_from_endpoint(
    warm_value: float,
    component_lipschitz_bound: float,
    first_step_scale: float,
    endpoint_ratio: float,
    iterations: int,
) -> dict[str, float]:
    """Parameterize eta_t=eta0/sqrt(t+t0) by eta_1 and eta_T/eta_1."""

    if iterations < 2:
        raise ValueError("A shifted schedule requires at least two iterations.")
    first_stepsize = (
        first_step_scale
        * warm_value
        / max(component_lipschitz_bound**2, np.finfo(float).tiny)
    )
    ratio_squared = endpoint_ratio**2
    virtual_time = ratio_squared * (iterations - 1) / (1.0 - ratio_squared)
    if virtual_time < 1.0:
        raise ValueError("Endpoint ratio would require a negative shift.")
    return {
        "first_step_scale": float(first_step_scale),
        "endpoint_ratio": float(endpoint_ratio),
        "first_stepsize": float(first_stepsize),
        "eta0": float(first_stepsize * math.sqrt(virtual_time)),
        "stepsize_offset": float(virtual_time - 1.0),
    }


def run_smax_shifted(
    split: base.SplitData,
    initial: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    value_budget: int,
    subset_size: int,
    schedule: dict[str, float],
    history_points: int,
    seed: int,
    method: str = base.SMAX_METHOD,
    required_iterations: np.ndarray | None = None,
) -> base.Trajectory:
    iterations = value_budget // subset_size
    if iterations < 2:
        raise ValueError("SMax budget must permit at least two iterations.")
    checkpoints = base.checkpoint_iterations(iterations, history_points)
    if required_iterations is not None:
        required = np.asarray(required_iterations, dtype=np.int64)
        required = required[(required >= 1) & (required <= iterations)]
        checkpoints = np.unique(np.concatenate((checkpoints, required)))
    checkpoint_set = set(int(value) for value in checkpoints)
    rng = np.random.default_rng(seed)
    y = np.asarray(initial, dtype=np.float64).copy()
    average = np.zeros_like(y)
    coefficients: list[np.ndarray] = []
    queries: list[int] = []
    times: list[float] = []
    subgradients: list[int] = []
    start_time = time.perf_counter()
    for iteration in range(1, iterations + 1):
        indices = rng.choice(
            split.n_components,
            size=subset_size,
            replace=False,
            shuffle=False,
        )
        phi, target = split.gather_centered_rows(indices)
        residual = phi @ y - target
        active = int(np.argmax(np.abs(residual)))
        active_value = float(residual[active])
        if active_value != 0.0:
            stepsize = schedule["eta0"] / math.sqrt(
                iteration + schedule["stepsize_offset"]
            )
            y -= stepsize * math.copysign(1.0, active_value) * phi[active]
            np.clip(y, lower_bounds, upper_bounds, out=y)
        average += (y - average) / iteration
        if iteration in checkpoint_set:
            coefficients.append(average.copy())
            queries.append(iteration * subset_size)
            times.append(time.perf_counter() - start_time)
            subgradients.append(iteration)
    return base.Trajectory(
        method=method,
        seed=seed,
        subset_size=subset_size if method == base.SMAX_METHOD else None,
        coefficients=np.vstack(coefficients),
        optimizer_value_queries=np.asarray(queries, dtype=np.int64),
        optimizer_runtime_seconds=np.asarray(times, dtype=np.float64),
        subgradient_queries=np.asarray(subgradients, dtype=np.int64),
        component_gradient_rows=np.asarray(subgradients, dtype=np.int64),
        raw_state={
            "checkpoint_iterations": checkpoints,
            "eta0": np.asarray([schedule["eta0"]]),
            "stepsize_offset": np.asarray([schedule["stepsize_offset"]]),
            "first_stepsize": np.asarray([schedule["first_stepsize"]]),
            "endpoint_ratio": np.asarray([schedule["endpoint_ratio"]]),
        },
    )


def run_full_sgm_shifted(
    split: base.SplitData,
    initial: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    value_budget: int,
    schedule: dict[str, float],
    history_points: int,
    sample_chunk: int,
) -> base.Trajectory:
    iterations = value_budget // split.n_components
    if iterations < 2:
        raise ValueError("Full-grid SGM budget must permit at least two scans.")
    checkpoints = base.checkpoint_iterations(iterations, history_points)
    checkpoint_set = set(int(value) for value in checkpoints)
    y = np.asarray(initial, dtype=np.float64).copy()
    average = np.zeros_like(y)
    coefficients: list[np.ndarray] = []
    queries: list[int] = []
    times: list[float] = []
    start_time = time.perf_counter()
    for iteration in range(1, iterations + 1):
        best_absolute = -math.inf
        best_residual = 0.0
        best_phi: np.ndarray | None = None
        for _, _, phi, target in split.iter_centered_chunks(sample_chunk):
            residual = phi @ y - target
            local = int(np.argmax(np.abs(residual)))
            local_absolute = float(abs(residual[local]))
            if local_absolute > best_absolute:
                best_absolute = local_absolute
                best_residual = float(residual[local])
                best_phi = phi[local].copy()
        if best_phi is None:
            raise RuntimeError("Full-grid SGM scan found no component.")
        if best_residual != 0.0:
            stepsize = schedule["eta0"] / math.sqrt(
                iteration + schedule["stepsize_offset"]
            )
            y -= stepsize * math.copysign(1.0, best_residual) * best_phi
            np.clip(y, lower_bounds, upper_bounds, out=y)
        average += (y - average) / iteration
        if iteration in checkpoint_set:
            coefficients.append(average.copy())
            queries.append(iteration * split.n_components)
            times.append(time.perf_counter() - start_time)
    return base.Trajectory(
        method=base.FULL_SGM_METHOD,
        coefficients=np.vstack(coefficients),
        optimizer_value_queries=np.asarray(queries, dtype=np.int64),
        optimizer_runtime_seconds=np.asarray(times, dtype=np.float64),
        subgradient_queries=checkpoints.copy(),
        component_gradient_rows=checkpoints.copy(),
        raw_state={
            "checkpoint_iterations": checkpoints,
            "eta0": np.asarray([schedule["eta0"]]),
            "stepsize_offset": np.asarray([schedule["stepsize_offset"]]),
            "first_stepsize": np.asarray([schedule["first_stepsize"]]),
            "endpoint_ratio": np.asarray([schedule["endpoint_ratio"]]),
        },
    )


def evaluate_train_max_batch(
    split: base.SplitData,
    coefficients: np.ndarray,
    sample_chunk: int,
    coefficient_block: int,
) -> np.ndarray:
    values = np.asarray(coefficients, dtype=np.float64)
    maxima = np.full(len(values), -math.inf, dtype=np.float64)
    for block_start in range(0, len(values), coefficient_block):
        block_stop = min(block_start + coefficient_block, len(values))
        block = values[block_start:block_stop]
        block_max = np.full(len(block), -math.inf, dtype=np.float64)
        for _, _, phi, target in split.iter_centered_chunks(sample_chunk):
            residual = phi @ block.T - target[:, None]
            block_max = np.maximum(block_max, np.max(np.abs(residual), axis=0))
        maxima[block_start:block_stop] = block_max
    return maxima


def tune_smax(
    args: argparse.Namespace,
    split: base.SplitData,
    initial: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    warm_value: float,
    max_row_norm: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, np.ndarray]]:
    run_rows: list[dict[str, Any]] = []
    final_coefficients: list[np.ndarray] = []
    candidate_id = 0
    candidates = (
        args.smax_candidates
        if args.smax_candidates is not None
        else [
            (subset_size, scale, ratio)
            for subset_size in args.smax_subset_candidates
            for scale in args.smax_first_step_scales
            for ratio in args.smax_endpoint_ratios
        ]
    )
    for subset_size, scale, ratio in candidates:
        if subset_size > split.n_components or subset_size > args.smax_value_budget:
            continue
        iterations = args.smax_value_budget // subset_size
        try:
            schedule = schedule_from_endpoint(
                warm_value, max_row_norm, scale, ratio, iterations
            )
        except ValueError:
            continue
        for seed in args.smax_tuning_seeds:
            trajectory = run_smax_shifted(
                split,
                initial,
                lower_bounds,
                upper_bounds,
                args.smax_value_budget,
                subset_size,
                schedule,
                1,
                seed,
                method="train-only SMax tuning",
            )
            final_coefficients.append(trajectory.coefficients[-1])
            run_rows.append(
                {
                    "candidate_id": candidate_id,
                    "seed": seed,
                    "subset_size": subset_size,
                    "iterations": iterations,
                    **schedule,
                    "optimizer_value_queries": int(
                        trajectory.optimizer_value_queries[-1]
                    ),
                    "optimizer_runtime_seconds": float(
                        trajectory.optimizer_runtime_seconds[-1]
                    ),
                }
            )
        candidate_id += 1
    if not run_rows:
        raise RuntimeError("The SMax tuning grid produced no valid schedule.")
    coefficient_array = np.vstack(final_coefficients)
    scores = evaluate_train_max_batch(
        split,
        coefficient_array,
        args.sample_chunk,
        args.tuning_coefficient_block,
    )
    for row, score in zip(run_rows, scores):
        row["final_train_max"] = float(score)
        row["validation_value_queries_uncharged"] = split.n_components
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        groups[int(row["candidate_id"])].append(row)
    summary_rows: list[dict[str, Any]] = []
    for identifier, members in groups.items():
        values = np.asarray([float(row["final_train_max"]) for row in members])
        first = members[0]
        summary_rows.append(
            {
                "candidate_id": identifier,
                "subset_size": first["subset_size"],
                "iterations": first["iterations"],
                "first_step_scale": first["first_step_scale"],
                "endpoint_ratio": first["endpoint_ratio"],
                "first_stepsize": first["first_stepsize"],
                "eta0": first["eta0"],
                "stepsize_offset": first["stepsize_offset"],
                "replicate_count": len(members),
                "median_final_train_max": float(np.median(values)),
                "q75_final_train_max": float(np.quantile(values, 0.75)),
                "mean_final_train_max": float(np.mean(values)),
            }
        )
    winner = min(
        summary_rows,
        key=lambda row: (
            float(row["median_final_train_max"]),
            float(row["q75_final_train_max"]),
            float(row["mean_final_train_max"]),
            int(row["subset_size"]),
        ),
    )
    selected = dict(winner)
    selected.update(
        {
            "selection_rule": (
                "minimum median endpoint training maximum, then upper "
                "quartile and mean, at the fixed small SMax budget"
            ),
            "tuning_seeds": list(args.smax_tuning_seeds),
            "test_used": False,
        }
    )
    arrays = {
        "smax_tuning_final_coefficients": coefficient_array,
        "smax_tuning_final_train_max": scores,
    }
    return run_rows, summary_rows, selected, arrays


def tune_full_sgm(
    args: argparse.Namespace,
    split: base.SplitData,
    initial: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    warm_value: float,
    max_row_norm: float,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, np.ndarray]]:
    iterations = args.other_value_budget // split.n_components
    rows: list[dict[str, Any]] = []
    coefficients: list[np.ndarray] = []
    for scale in args.sgm_first_step_scales:
        for ratio in args.sgm_endpoint_ratios:
            try:
                schedule = schedule_from_endpoint(
                    warm_value, max_row_norm, scale, ratio, iterations
                )
            except ValueError:
                continue
            trajectory = run_full_sgm_shifted(
                split,
                initial,
                lower_bounds,
                upper_bounds,
                args.other_value_budget,
                schedule,
                1,
                args.sample_chunk,
            )
            score, _, _, _ = base.scan_max_and_norm(
                split, trajectory.coefficients[-1], args.sample_chunk
            )
            coefficients.append(trajectory.coefficients[-1])
            rows.append(
                {
                    "candidate_id": len(rows),
                    "iterations": iterations,
                    **schedule,
                    "optimizer_value_queries": int(
                        trajectory.optimizer_value_queries[-1]
                    ),
                    "optimizer_runtime_seconds": float(
                        trajectory.optimizer_runtime_seconds[-1]
                    ),
                    "final_train_max": float(score),
                    "validation_value_queries_uncharged": split.n_components,
                }
            )
    if not rows:
        raise RuntimeError("The full-grid SGM tuning grid produced no schedule.")
    selected = dict(
        min(
            rows,
            key=lambda row: (
                float(row["final_train_max"]),
                float(row["first_step_scale"]),
                float(row["endpoint_ratio"]),
            ),
        )
    )
    selected.update(
        {
            "selection_rule": (
                "minimum endpoint training maximum at the full other-method budget"
            ),
            "test_used": False,
        }
    )
    arrays = {"full_sgm_tuning_final_coefficients": np.vstack(coefficients)}
    return rows, selected, arrays


def tune_lse(
    args: argparse.Namespace,
    split: base.SplitData,
    initial: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    warm_value: float,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, np.ndarray]]:
    rows: list[dict[str, Any]] = []
    coefficients: list[np.ndarray] = []
    for multipliers in args.lse_candidates:
        for maxiter in args.lse_maxiter_candidates:
            trajectory = base.run_lse_lbfgsb(
                split,
                initial,
                lower_bounds,
                upper_bounds,
                warm_value,
                multipliers,
                maxiter,
                args.sample_chunk,
                args.other_value_budget,
            )
            score, _, _, _ = base.scan_max_and_norm(
                split, trajectory.coefficients[-1], args.sample_chunk
            )
            coefficients.append(trajectory.coefficients[-1])
            rows.append(
                {
                    "candidate_id": len(rows),
                    "multipliers": json.dumps(multipliers),
                    "maxiter_per_stage": maxiter,
                    "optimizer_value_queries": int(
                        trajectory.optimizer_value_queries[-1]
                    ),
                    "optimizer_runtime_seconds": float(
                        trajectory.optimizer_runtime_seconds[-1]
                    ),
                    "final_train_max": float(score),
                    "status": trajectory.status,
                    "validation_value_queries_uncharged": split.n_components,
                }
            )
    selected = dict(
        min(
            rows,
            key=lambda row: (
                float(row["final_train_max"]),
                int(row["maxiter_per_stage"]),
                str(row["multipliers"]),
            ),
        )
    )
    selected["multipliers"] = json.loads(str(selected["multipliers"]))
    selected.update(
        {
            "selection_rule": (
                "minimum endpoint training maximum at the full other-method budget"
            ),
            "test_used": False,
        }
    )
    arrays = {"lse_tuning_final_coefficients": np.vstack(coefficients)}
    return rows, selected, arrays


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_frozen_tuning(
    directory: Path,
) -> tuple[
    list[dict[str, str]],
    list[dict[str, str]],
    dict[str, Any],
    list[dict[str, str]],
    dict[str, Any],
    list[dict[str, str]],
    dict[str, Any],
    dict[str, np.ndarray],
]:
    required = (
        "smax_tuning_runs.csv",
        "smax_tuning_summary.csv",
        "selected_smax.json",
        "sgm_full_budget_tuning.csv",
        "selected_sgm_full_budget.json",
        "lse_full_budget_tuning.csv",
        "selected_lse_full_budget.json",
        "method_state.npz",
    )
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Frozen tuning directory {directory} is missing {missing}."
        )
    selected_smax = json.loads(
        (directory / "selected_smax.json").read_text(encoding="utf-8")
    )
    selected_sgm = json.loads(
        (directory / "selected_sgm_full_budget.json").read_text(encoding="utf-8")
    )
    selected_lse = json.loads(
        (directory / "selected_lse_full_budget.json").read_text(encoding="utf-8")
    )
    with np.load(directory / "method_state.npz", allow_pickle=False) as state:
        tuning_arrays = {
            name: np.asarray(state[name])
            for name in state.files
            if name.startswith(
                ("smax_tuning_", "full_sgm_tuning_", "lse_tuning_")
            )
        }
    return (
        read_csv(directory / "smax_tuning_runs.csv"),
        read_csv(directory / "smax_tuning_summary.csv"),
        selected_smax,
        read_csv(directory / "sgm_full_budget_tuning.csv"),
        selected_sgm,
        read_csv(directory / "lse_full_budget_tuning.csv"),
        selected_lse,
        tuning_arrays,
    )


def add_relative_gaps(path: Path, lower_bound: float) -> None:
    rows = read_csv(path)
    for row in rows:
        train_value = float(row["train_max_centered_logit_distortion"])
        row["certified_training_lower_bound"] = lower_bound
        row["relative_training_gap"] = max(
            0.0, (train_value - lower_bound) / lower_bound
        )
    base._write_csv(path, rows)


def first_target_crossing(
    rows: list[dict[str, str]], target: float
) -> tuple[int | None, float | None, float | None]:
    ordered = sorted(rows, key=lambda row: int(float(row["optimizer_value_queries"])))
    for row in ordered:
        gap = float(row["relative_training_gap"])
        if gap <= target:
            return (
                int(float(row["optimizer_value_queries"])),
                float(row["optimizer_runtime_seconds"]),
                gap,
            )
    return None, None, None


def write_public_summaries(output_dir: Path, target: float) -> None:
    history = read_csv(output_dir / "history.csv")
    summary = read_csv(output_dir / "summary.csv")
    methods = [
        base.SMAX_METHOD,
        base.FULL_SGM_METHOD,
        base.LSE_METHOD,
        base.LP_METHOD,
    ]
    final_rows: list[dict[str, Any]] = []
    crossing_rows: list[dict[str, Any]] = []
    for method in methods:
        method_history = [row for row in history if row["method"] == method]
        if method == base.SMAX_METHOD:
            by_checkpoint: dict[int, list[dict[str, str]]] = defaultdict(list)
            for row in method_history:
                by_checkpoint[int(row["checkpoint"])].append(row)
            aggregate_history: list[dict[str, str]] = []
            for checkpoint in sorted(by_checkpoint):
                members = by_checkpoint[checkpoint]
                aggregate_history.append(
                    {
                        "optimizer_value_queries": str(
                            int(np.median([float(row["optimizer_value_queries"]) for row in members]))
                        ),
                        "optimizer_runtime_seconds": str(
                            float(np.median([float(row["optimizer_runtime_seconds"]) for row in members]))
                        ),
                        "relative_training_gap": str(
                            float(np.median([float(row["relative_training_gap"]) for row in members]))
                        ),
                    }
                )
            crossing = first_target_crossing(aggregate_history, target)
            individuals = [
                row
                for row in summary
                if row["method"] == method and row.get("row_type") == "individual_final"
            ]
            endpoint: dict[str, Any] = {
                "method": method,
                "subset_size": individuals[0]["subset_size"],
                "replicate_count": len(individuals),
            }
            for name in (
                "optimizer_value_queries",
                "optimizer_runtime_seconds",
                "train_max_centered_logit_distortion",
                "relative_training_gap",
                "test_max_centered_logit_distortion",
                "test_ground_truth_accuracy",
                "test_original_prediction_agreement",
            ):
                endpoint[name] = float(np.median([float(row[name]) for row in individuals]))
                endpoint[f"{name}_q25"] = float(
                    np.quantile([float(row[name]) for row in individuals], 0.25)
                )
                endpoint[f"{name}_q75"] = float(
                    np.quantile([float(row[name]) for row in individuals], 0.75)
                )
        else:
            crossing = first_target_crossing(method_history, target)
            endpoint_source = [
                row
                for row in summary
                if row["method"] == method and row.get("row_type") == "individual_final"
            ][0]
            endpoint = {
                "method": method,
                "subset_size": endpoint_source.get("subset_size", ""),
                "replicate_count": 1,
            }
            for name in (
                "optimizer_value_queries",
                "optimizer_runtime_seconds",
                "train_max_centered_logit_distortion",
                "relative_training_gap",
                "test_max_centered_logit_distortion",
                "test_ground_truth_accuracy",
                "test_original_prediction_agreement",
            ):
                endpoint[name] = float(endpoint_source[name])
        endpoint["target_relative_gap"] = target
        endpoint["queries_to_target"] = "" if crossing[0] is None else crossing[0]
        endpoint["seconds_to_target"] = "" if crossing[1] is None else crossing[1]
        endpoint["gap_at_target_checkpoint"] = "" if crossing[2] is None else crossing[2]
        final_rows.append(endpoint)
        crossing_rows.append(
            {
                "method": method,
                "target_relative_gap": target,
                "queries_to_target": "" if crossing[0] is None else crossing[0],
                "seconds_to_target": "" if crossing[1] is None else crossing[1],
                "gap_at_target_checkpoint": "" if crossing[2] is None else crossing[2],
            }
        )
    base._write_csv(output_dir / "final_summary.csv", final_rows)
    base._write_csv(output_dir / "time_to_target.csv", crossing_rows)


def main() -> None:
    args = finalize_args(parse_args())
    if args.plot_only:
        add_configured_outputs(args.output_dir, args.target_relative_gap)
        return

    print(f"Loading cached features from {args.feature_dir}", flush=True)
    problem = base.load_cached_problem(
        args.feature_dir, args.train_samples, args.test_samples
    )
    setup_start = time.perf_counter()
    train_mean_seconds = problem.train.prepare_random_access_means(args.sample_chunk)
    test_mean_seconds = problem.test.prepare_random_access_means(args.sample_chunk)
    scalar_start = time.perf_counter()
    y_fun, e_fun, functional_diagnostics = base.scalar_near_minimax_relu(
        problem.degree, args.functional_grid_points
    )
    functional_seconds = time.perf_counter() - scalar_start
    halfwidth = args.functional_box_kappa * e_fun / (problem.degree + 1)
    lower_bounds = y_fun - halfwidth
    upper_bounds = y_fun + halfwidth
    base.warm_up_kernels(
        problem.train, y_fun, lower_bounds, upper_bounds, args.sample_chunk
    )

    wls_values: list[np.ndarray] = []
    wls_seconds: list[float] = []
    for _ in range(args.timing_repeats):
        value, seconds = base.streaming_wls(
            problem.train, args.sample_chunk, lower_bounds, upper_bounds
        )
        wls_values.append(value)
        wls_seconds.append(seconds)
    wls = wls_values[0]
    if any(
        not np.allclose(wls, value, rtol=1e-8, atol=1e-10)
        for value in wls_values[1:]
    ):
        raise RuntimeError("Repeated WLS solves were not reproducible.")
    cheb_start = time.perf_counter()
    cheb = base.chebyshev_node_relu_coefficients(problem.degree)
    cheb_seconds = time.perf_counter() - cheb_start
    warm_value, max_row_norm, _, _ = base.scan_max_and_norm(
        problem.train, wls, args.sample_chunk
    )

    if args.reuse_tuning_dir is None:
        print("Tuning SMax at its small fixed budget...", flush=True)
        smax_runs, smax_summary, selected_smax, tuning_arrays = tune_smax(
            args,
            problem.train,
            wls,
            lower_bounds,
            upper_bounds,
            warm_value,
            max_row_norm,
        )
        print(f"Selected SMax: {selected_smax}", flush=True)
        if args.reuse_baseline_tuning_dir is None:
            print("Tuning full-grid SGM at the full budget...", flush=True)
            sgm_tuning, selected_sgm, sgm_arrays = tune_full_sgm(
                args,
                problem.train,
                wls,
                lower_bounds,
                upper_bounds,
                warm_value,
                max_row_norm,
            )
            tuning_arrays.update(sgm_arrays)
            print(f"Selected SGM: {selected_sgm}", flush=True)
            print("Tuning LSE continuation at the full budget...", flush=True)
            lse_tuning, selected_lse, lse_arrays = tune_lse(
                args,
                problem.train,
                wls,
                lower_bounds,
                upper_bounds,
                warm_value,
            )
            tuning_arrays.update(lse_arrays)
            print(f"Selected LSE: {selected_lse}", flush=True)
        else:
            print(
                "Reusing frozen training-only baseline tuning from "
                f"{args.reuse_baseline_tuning_dir}",
                flush=True,
            )
            (
                _,
                _,
                _,
                sgm_tuning,
                selected_sgm,
                lse_tuning,
                selected_lse,
                reused_arrays,
            ) = load_frozen_tuning(args.reuse_baseline_tuning_dir)
            tuning_arrays.update(
                {
                    name: values
                    for name, values in reused_arrays.items()
                    if name.startswith(("full_sgm_tuning_", "lse_tuning_"))
                }
            )
    else:
        print(
            f"Reusing frozen training-only tuning from {args.reuse_tuning_dir}",
            flush=True,
        )
        (
            smax_runs,
            smax_summary,
            selected_smax,
            sgm_tuning,
            selected_sgm,
            lse_tuning,
            selected_lse,
            tuning_arrays,
        ) = load_frozen_tuning(args.reuse_tuning_dir)
    setup_elapsed_seconds = time.perf_counter() - setup_start

    trajectories: list[base.Trajectory] = [
        base.Trajectory(
            method=base.WLS_METHOD,
            coefficients=wls,
            optimizer_value_queries=np.zeros(1, dtype=np.int64),
            optimizer_runtime_seconds=np.zeros(1),
            subgradient_queries=np.zeros(1, dtype=np.int64),
            component_gradient_rows=np.zeros(1, dtype=np.int64),
            timing_repeats_seconds=list(wls_seconds),
        ),
        base.Trajectory(
            method=base.FUNCTIONAL_METHOD,
            coefficients=y_fun,
            optimizer_value_queries=np.zeros(1, dtype=np.int64),
            optimizer_runtime_seconds=np.zeros(1),
            subgradient_queries=np.zeros(1, dtype=np.int64),
            component_gradient_rows=np.zeros(1, dtype=np.int64),
        ),
        base.Trajectory(
            method=base.CHEB_METHOD,
            coefficients=cheb,
            optimizer_value_queries=np.zeros(1, dtype=np.int64),
            optimizer_runtime_seconds=np.zeros(1),
            subgradient_queries=np.zeros(1, dtype=np.int64),
            component_gradient_rows=np.zeros(1, dtype=np.int64),
        ),
    ]

    smax_schedule = {
        name: float(selected_smax[name])
        for name in (
            "first_step_scale",
            "endpoint_ratio",
            "first_stepsize",
            "eta0",
            "stepsize_offset",
        )
    }
    selected_subset = int(selected_smax["subset_size"])
    tuning_horizon_iterations = (
        args.smax_value_budget // selected_subset
    )
    early_checkpoints = base.checkpoint_iterations(
        tuning_horizon_iterations, args.history_points
    )
    print(
        "Running full-budget confirmatory SMax trajectories with the frozen "
        "tuned schedule...",
        flush=True,
    )
    for seed in args.confirmatory_seeds:
        trajectories.append(
            run_smax_shifted(
                problem.train,
                wls,
                lower_bounds,
                upper_bounds,
                args.other_value_budget,
                selected_subset,
                smax_schedule,
                args.history_points,
                seed,
                required_iterations=early_checkpoints,
            )
        )

    sgm_schedule = {
        name: float(selected_sgm[name])
        for name in (
            "first_step_scale",
            "endpoint_ratio",
            "first_stepsize",
            "eta0",
            "stepsize_offset",
        )
    }

    def full_once() -> base.Trajectory:
        return run_full_sgm_shifted(
            problem.train,
            wls,
            lower_bounds,
            upper_bounds,
            args.other_value_budget,
            sgm_schedule,
            args.history_points,
            args.sample_chunk,
        )

    full = full_once()
    full_times = [float(full.optimizer_runtime_seconds[-1])]
    for _ in range(1, args.timing_repeats):
        repeated = full_once()
        full_times.append(float(repeated.optimizer_runtime_seconds[-1]))
    base.rescale_trajectory_timing(full, full_times)
    trajectories.append(full)

    def lse_once() -> base.Trajectory:
        return base.run_lse_lbfgsb(
            problem.train,
            wls,
            lower_bounds,
            upper_bounds,
            warm_value,
            selected_lse["multipliers"],
            int(selected_lse["maxiter_per_stage"]),
            args.sample_chunk,
            args.other_value_budget,
        )

    lse = lse_once()
    lse_times = [float(lse.optimizer_runtime_seconds[-1])]
    for _ in range(1, args.timing_repeats):
        repeated = lse_once()
        lse_times.append(float(repeated.optimizer_runtime_seconds[-1]))
    base.rescale_trajectory_timing(lse, lse_times)
    trajectories.append(lse)

    def lp_once() -> base.Trajectory:
        return base.run_restricted_lp_exchange(
            problem.train,
            wls,
            lower_bounds,
            upper_bounds,
            args.other_value_budget,
            args.exchange_initial_rows,
            args.exchange_add_rows,
            args.exchange_max_iterations,
            args.exchange_absolute_tolerance,
            args.exchange_relative_tolerance,
            args.sample_chunk,
        )

    lp = lp_once()
    lp_times = [float(lp.optimizer_runtime_seconds[-1])]
    for _ in range(1, args.lp_timing_repeats):
        repeated = lp_once()
        lp_times.append(float(repeated.optimizer_runtime_seconds[-1]))
    base.rescale_trajectory_timing(lp, lp_times)
    trajectories.append(lp)

    slices: list[slice] = []
    blocks: list[np.ndarray] = []
    offset = 0
    for trajectory in trajectories:
        stop = offset + len(trajectory.coefficients)
        slices.append(slice(offset, stop))
        blocks.append(trajectory.coefficients)
        offset = stop
    all_coefficients = np.vstack(blocks)
    print(f"Offline evaluation of {len(all_coefficients)} stored iterates...", flush=True)
    evaluation = base.evaluate_coefficients(
        problem.train,
        problem.test,
        all_coefficients,
        args.sample_chunk,
        args.evaluation_coefficient_block,
    )
    scalar_diagnostics = base.scalar_polynomial_diagnostics(
        all_coefficients,
        y_fun,
        lower_bounds,
        upper_bounds,
        problem.scale_B,
    )
    smax_trajectories = [
        trajectory for trajectory in trajectories if trajectory.method == base.SMAX_METHOD
    ]
    diagnostic_coefficients = {
        base.WLS_METHOD: wls,
        (
            f"{base.SMAX_METHOD} middle "
            f"(m={selected_subset}, seed={smax_trajectories[0].seed})"
        ): smax_trajectories[0].coefficients[len(smax_trajectories[0].coefficients) // 2],
        f"{base.LP_METHOD} incumbent": lp.coefficients[-1],
    }
    tail_state = base.residual_tail_statistics(
        problem.train,
        diagnostic_coefficients,
        args.tail_sample_size,
        args.tail_seed,
    )
    exact_tail_state = base.exact_near_max_cardinality(
        problem.train, diagnostic_coefficients, args.sample_chunk
    )
    setup_state = {
        "train_random_access_mean_seconds": train_mean_seconds,
        "test_random_access_mean_seconds": test_mean_seconds,
        "scalar_functional_minimax_seconds": functional_seconds,
        "total_before_production_optimization_seconds": setup_elapsed_seconds,
        "includes_all_training_only_tuning": True,
    }
    args.value_budget = args.other_value_budget
    args.subset_sizes = [selected_subset]
    args.seeds = list(args.confirmatory_seeds)
    args.lse_multipliers = list(selected_lse["multipliers"])
    args.lse_maxiter_per_stage = int(selected_lse["maxiter_per_stage"])
    base.save_results(
        args,
        problem,
        trajectories,
        all_coefficients,
        slices,
        evaluation,
        scalar_diagnostics,
        y_fun,
        e_fun,
        functional_diagnostics,
        lower_bounds,
        upper_bounds,
        wls_seconds,
        {
            base.FUNCTIONAL_METHOD: functional_seconds,
            base.CHEB_METHOD: cheb_seconds,
        },
        float(selected_smax["eta0"]),
        selected_smax,
        tuning_arrays,
        float(selected_sgm["eta0"]),
        selected_sgm,
        setup_state,
        tail_state,
        exact_tail_state,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base._write_csv(args.output_dir / "smax_tuning_runs.csv", smax_runs)
    base._write_csv(args.output_dir / "smax_tuning_summary.csv", smax_summary)
    base._write_csv(args.output_dir / "sgm_full_budget_tuning.csv", sgm_tuning)
    base._write_csv(args.output_dir / "lse_full_budget_tuning.csv", lse_tuning)
    (args.output_dir / "selected_smax.json").write_text(
        json.dumps(base._jsonable(selected_smax), indent=2), encoding="utf-8"
    )
    (args.output_dir / "selected_sgm_full_budget.json").write_text(
        json.dumps(base._jsonable(selected_sgm), indent=2), encoding="utf-8"
    )
    (args.output_dir / "selected_lse_full_budget.json").write_text(
        json.dumps(base._jsonable(selected_lse), indent=2), encoding="utf-8"
    )
    lower_bound = float(lp.lower_bounds[-1])
    add_relative_gaps(args.output_dir / "history.csv", lower_bound)
    add_relative_gaps(args.output_dir / "summary.csv", lower_bound)
    write_public_summaries(args.output_dir, args.target_relative_gap)
    config_path = args.output_dir / "run_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["budget_protocol"] = {
        "smax_tuning_component_value_budget": args.smax_value_budget,
        "smax_production_component_value_budget": args.other_value_budget,
        "other_method_component_value_cap": args.other_value_budget,
        "smax_tuning_budget_in_full_data_equivalents": (
            args.smax_value_budget / problem.train.n_components
        ),
        "smax_production_budget_in_full_data_equivalents": (
            args.other_value_budget / problem.train.n_components
        ),
        "other_budget_in_full_data_equivalents": (
            args.other_value_budget / problem.train.n_components
        ),
        "certified_training_lower_bound": lower_bound,
        "relative_gap_definition": "max(0, (training maximum - lower bound) / lower bound)",
        "target_relative_gap": args.target_relative_gap,
        "smax_tuning_horizon_shorter_than_production": (
            args.smax_value_budget < args.other_value_budget
        ),
        "full_grid_sgm_tuned_at_full_budget": True,
        "lse_tuned_at_full_budget": True,
        "all_tuning_uses_training_only": True,
        "confirmatory_smax_seeds": args.confirmatory_seeds,
        "tuning_reused_from": (
            None
            if args.reuse_tuning_dir is None
            else str(args.reuse_tuning_dir)
        ),
    }
    config["selected_smax_shifted_schedule"] = selected_smax
    config["selected_full_sgm_shifted_schedule"] = selected_sgm
    config["selected_lse_continuation"] = selected_lse
    config["files"].update(
        {
            "plot": "anytime_relative_training_gap.png",
            "final_summary": "final_summary.csv",
            "time_to_target": "time_to_target.csv",
            "smax_tuning_runs": "smax_tuning_runs.csv",
            "smax_tuning_summary": "smax_tuning_summary.csv",
            "sgm_tuning": "sgm_full_budget_tuning.csv",
            "lse_tuning": "lse_full_budget_tuning.csv",
        }
    )
    config["software"].update(
        {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        }
    )
    config_path.write_text(
        json.dumps(base._jsonable(config), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    add_configured_outputs(args.output_dir, args.target_relative_gap)
    print(f"Saved budget comparison to {args.output_dir}", flush=True)


def add_configured_outputs(output_dir: Path, target: float) -> None:
    history_path = output_dir / "history.csv"
    config_path = output_dir / "run_config.json"
    if not history_path.is_file() or not config_path.is_file():
        raise FileNotFoundError(f"Stored history/configuration not found in {output_dir}.")
    write_public_summaries(output_dir, target)
    from plot_anytime_results import make_plot

    plot_path = make_plot(output_dir)
    print(f"Plot: {plot_path}", flush=True)


if __name__ == "__main__":
    main()
