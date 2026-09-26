"""Summarize the common-budget protocol and compute pointwise effective k.

The optimizer histories and coefficients are treated as immutable inputs.  All
extra full-grid scans in this script are offline diagnostics and are excluded
from optimizer query and runtime accounting.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

import run_experiment as base


SMAX = "SMax-SGM"
SGM = "Full-grid SGM"
LSE = "LSE--L-BFGS"
SOCP = "Restricted-SOCP exchange"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--target-relative-gap", type=float, default=0.05)
    parser.add_argument("--curve-points", type=int, default=80)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty table {path}.")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def relative_gap(value: float, lower_bound: float) -> float:
    return max(0.0, (value - lower_bound) / lower_bound)


def numerical_lower_bound(config: dict[str, Any]) -> float:
    source = Path(config["restricted_socp_source"])
    rows = read_rows(source)
    values = [
        float(row["numerical_dual_lower_bound"])
        for row in rows
        if row.get("numerical_dual_lower_bound")
    ]
    if not values:
        raise ValueError(f"No numerical lower bound found in {source}.")
    return max(values)


def effective_k_statistics(
    values: np.ndarray,
    threshold: float,
    curve_points: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ordered = np.sort(np.asarray(values, dtype=np.float64).ravel())[::-1]
    prefix = np.cumsum(ordered, dtype=np.float64)
    counts = np.arange(1, len(ordered) + 1, dtype=np.float64)
    gaps = ordered[0] - prefix / counts
    np.maximum.accumulate(gaps, out=gaps)
    effective_k = int(np.searchsorted(gaps, threshold, side="right"))
    effective_k = min(max(effective_k, 1), len(ordered))

    requested = np.unique(
        np.rint(np.geomspace(1, len(ordered), max(2, curve_points))).astype(int)
    )
    boundary = [effective_k]
    if effective_k < len(ordered):
        boundary.append(effective_k + 1)
    requested = np.unique(np.concatenate((requested, np.asarray(boundary))))
    curve = [
        {
            "k": int(k),
            "top_k_mean": float(prefix[k - 1] / k),
            "pointwise_top_k_gap": float(gaps[k - 1]),
            "passes_epsilon_over_4": bool(gaps[k - 1] <= threshold),
        }
        for k in requested
    ]
    summary = {
        "component_count": int(len(ordered)),
        "component_max": float(ordered[0]),
        "component_min": float(ordered[-1]),
        "component_width": float(ordered[0] - ordered[-1]),
        "effective_k": effective_k,
        "effective_k_fraction": float(effective_k / len(ordered)),
        "N_over_effective_k": float(len(ordered) / effective_k),
        "top_k_mean_at_effective_k": float(prefix[effective_k - 1] / effective_k),
        "gap_at_effective_k": float(gaps[effective_k - 1]),
        "gap_at_next_k": (
            "" if effective_k == len(ordered) else float(gaps[effective_k])
        ),
    }
    return summary, curve


def quantile_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["point_role"])].append(row)
    result: list[dict[str, Any]] = []
    fields = (
        "optimizer_value_queries",
        "training_max_error",
        "relative_training_gap",
        "effective_k",
        "effective_k_fraction",
        "N_over_effective_k",
        "gap_at_effective_k",
    )
    for role, members in grouped.items():
        output: dict[str, Any] = {"point_role": role, "replicate_count": len(members)}
        for field in fields:
            values = np.asarray([float(row[field]) for row in members])
            output[f"{field}_median"] = float(np.median(values))
            output[f"{field}_q25"] = float(np.quantile(values, 0.25))
            output[f"{field}_q75"] = float(np.quantile(values, 0.75))
        result.append(output)
    return result


def crossing_tables(
    history: list[dict[str, str]],
    lower_bound: float,
    target: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    detail: list[dict[str, Any]] = []
    methods = (SMAX, SGM, LSE, SOCP)
    for method in methods:
        method_rows = [row for row in history if row["method"] == method]
        if method == SMAX:
            groups = [
                (
                    str(seed),
                    [row for row in method_rows if int(row["seed"]) == seed],
                )
                for seed in sorted({int(row["seed"]) for row in method_rows})
            ]
        else:
            groups = [("", method_rows)]
        for seed, rows in groups:
            rows.sort(key=lambda row: int(row["optimizer_value_queries"]))
            crossing = next(
                (
                    row
                    for row in rows
                    if relative_gap(
                        float(row["training_max_error"]), lower_bound
                    )
                    <= target
                ),
                None,
            )
            detail.append(
                {
                    "method": method,
                    "seed": seed,
                    "reached_target": crossing is not None,
                    "optimizer_value_queries": (
                        "" if crossing is None else int(crossing["optimizer_value_queries"])
                    ),
                    "optimizer_runtime_seconds": (
                        ""
                        if crossing is None
                        else float(crossing["optimizer_runtime_seconds"])
                    ),
                    "training_max_error": (
                        "" if crossing is None else float(crossing["training_max_error"])
                    ),
                    "relative_training_gap": (
                        ""
                        if crossing is None
                        else relative_gap(
                            float(crossing["training_max_error"]), lower_bound
                        )
                    ),
                }
            )

    summary: list[dict[str, Any]] = []
    for method in methods:
        members = [row for row in detail if row["method"] == method]
        reached = [row for row in members if row["reached_target"]]
        row: dict[str, Any] = {
            "method": method,
            "runs": len(members),
            "runs_reaching_target": len(reached),
            "target_relative_gap": target,
        }
        for field in ("optimizer_value_queries", "optimizer_runtime_seconds"):
            if reached:
                values = np.asarray([float(item[field]) for item in reached])
                row[f"{field}_median"] = float(np.median(values))
                row[f"{field}_q25"] = float(np.quantile(values, 0.25))
                row[f"{field}_q75"] = float(np.quantile(values, 0.75))
            else:
                row[f"{field}_median"] = ""
                row[f"{field}_q25"] = ""
                row[f"{field}_q75"] = ""
        summary.append(row)
    return detail, summary


def plot_relative_gap(
    results_dir: Path,
    history: list[dict[str, str]],
    lower_bound: float,
    target: float,
) -> None:
    smax_rows = [row for row in history if row["method"] == SMAX]
    seeds = sorted({int(row["seed"]) for row in smax_rows})
    runs = [
        sorted(
            [row for row in smax_rows if int(row["seed"]) == seed],
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
    gaps = np.asarray(
        [
            [relative_gap(float(row["training_max_error"]), lower_bound) for row in run]
            for run in runs
        ]
    )
    if not np.all(queries == queries[0]):
        raise ValueError("SMax confirmatory checkpoints are inconsistent.")

    colors = {
        SMAX: "#0072B2",
        SGM: "#D55E00",
        LSE: "#009E73",
        SOCP: "#555555",
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.35), constrained_layout=True)
    for axis, x_field, smax_x, x_label in (
        (
            axes[0],
            "optimizer_value_queries",
            queries[0],
            "Optimizer component-value queries",
        ),
        (
            axes[1],
            "optimizer_runtime_seconds",
            np.median(runtimes, axis=0),
            "Optimizer runtime (seconds)",
        ),
    ):
        axis.plot(
            smax_x,
            np.median(gaps, axis=0),
            color=colors[SMAX],
            linewidth=2.2,
            label=r"SMax--SGM, $m=16384$",
        )
        axis.fill_between(
            smax_x,
            np.quantile(gaps, 0.25, axis=0),
            np.quantile(gaps, 0.75, axis=0),
            color=colors[SMAX],
            alpha=0.17,
            linewidth=0,
        )
        for method in (SGM, LSE, SOCP):
            rows = sorted(
                [row for row in history if row["method"] == method],
                key=lambda row: int(row["optimizer_value_queries"]),
            )
            x = np.asarray([float(row[x_field]) for row in rows])
            y = np.asarray(
                [
                    relative_gap(float(row["training_max_error"]), lower_bound)
                    for row in rows
                ]
            )
            axis.plot(
                x,
                y,
                color=colors[method],
                linewidth=1.8,
                marker="o" if method == SGM else "s",
                markersize=3.2,
                drawstyle="steps-post" if method == SOCP else "default",
                label=method,
            )
        axis.axhline(target, color="#777777", linestyle="--", linewidth=1.0)
        if x_field.endswith("seconds"):
            axis.set_xscale("symlog", linthresh=0.01)
        else:
            axis.set_xscale("symlog", linthresh=10_000)
        axis.set_yscale("log")
        axis.set_xlabel(x_label)
        axis.grid(True, which="both", axis="both", alpha=0.25)
        axis.set_xlim(left=0)
    axes[0].set_ylabel("Relative training gap to numerical reference")
    axes[0].legend(fontsize=7.5, loc="best")
    fig.savefig(results_dir / "symmetric_relative_training_gap.png", dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not 0.0 < args.target_relative_gap < 1.0:
        raise ValueError("The target relative gap must lie in (0, 1).")
    if args.curve_points < 2:
        raise ValueError("At least two curve points are required.")

    results_dir = args.results_dir.resolve()
    config = json.loads((results_dir / "run_config.json").read_text(encoding="utf-8"))
    history = read_rows(results_dir / "history.csv")
    lower_bound = numerical_lower_bound(config)
    epsilon_absolute = args.target_relative_gap * lower_bound
    epsilon_over_4 = epsilon_absolute / 4.0

    crossing_detail, crossing_summary = crossing_tables(
        history, lower_bound, args.target_relative_gap
    )
    write_rows(results_dir / "target_crossing_runs.csv", crossing_detail)
    write_rows(results_dir / "time_to_target_summary.csv", crossing_summary)

    problem = config["problem"]
    training_grid = base.make_grid(
        int(config["arguments"]["frequency_count"]),
        int(config["arguments"]["delay_count"]),
        float(config["arguments"]["alpha"]),
    )
    if training_grid.size != int(problem["training_components"]):
        raise ValueError("Training grid reconstructed with a different size.")

    key_points: list[tuple[str, dict[str, str], str, int | None]] = []
    first_smax = next(row for row in history if row["method"] == SMAX)
    key_points.append(("wls_start", first_smax, "wls", None))
    smax_rows = [row for row in history if row["method"] == SMAX]
    for seed in sorted({int(row["seed"]) for row in smax_rows}):
        rows = sorted(
            [row for row in smax_rows if int(row["seed"]) == seed],
            key=lambda row: int(row["optimizer_value_queries"]),
        )
        positive = [row for row in rows if int(row["optimizer_value_queries"]) > 0]
        crossing = next(
            (
                (index, row)
                for index, row in enumerate(positive)
                if relative_gap(float(row["training_max_error"]), lower_bound)
                <= args.target_relative_gap
            ),
            None,
        )
        array_name = f"smax_seed_{seed}_checkpoints"
        if crossing is not None:
            key_points.append(
                ("smax_first_target_crossing", crossing[1], array_name, crossing[0])
            )
        key_points.append(("smax_final", positive[-1], array_name, len(positive) - 1))

    for method, role, array_name in (
        (SGM, "full_grid_sgm_final", "full_grid_sgm"),
        (LSE, "lse_final", "lse_lbfgs"),
    ):
        rows = sorted(
            [row for row in history if row["method"] == method],
            key=lambda row: int(row["optimizer_value_queries"]),
        )
        key_points.append((role, rows[-1], array_name, None))
    socp_source = sorted(
        read_rows(Path(config["restricted_socp_source"])),
        key=lambda row: int(row["value_queries"]),
    )[-1]
    key_points.append(
        (
            "restricted_socp_reference",
            {
                "method": SOCP,
                "seed": "",
                "optimizer_value_queries": socp_source["value_queries"],
                "optimizer_runtime_seconds": socp_source["elapsed_seconds"],
            },
            "restricted_socp",
            None,
        )
    )

    detail_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    with np.load(results_dir / "coefficients.npz", allow_pickle=False) as arrays:
        for position, (role, row, array_name, index) in enumerate(key_points, start=1):
            print(
                f"[{position}/{len(key_points)}] {role}, seed={row.get('seed', '')}",
                flush=True,
            )
            coefficient_array = arrays[array_name]
            coefficient = (
                coefficient_array
                if index is None
                else coefficient_array[int(index)]
            )
            values = base.error_array(
                np.asarray(coefficient, dtype=np.float64), training_grid
            )
            statistics, curve = effective_k_statistics(
                values, epsilon_over_4, args.curve_points
            )
            train = float(np.max(values))
            identity = {
                "point_role": role,
                "method": row["method"],
                "seed": row.get("seed", ""),
                "optimizer_value_queries": int(row["optimizer_value_queries"]),
                "optimizer_runtime_seconds": float(
                    row["optimizer_runtime_seconds"]
                ),
                "training_max_error": train,
                "relative_training_gap": relative_gap(train, lower_bound),
                "numerical_lower_reference": lower_bound,
                "target_relative_gap": args.target_relative_gap,
                "epsilon_absolute": epsilon_absolute,
                "epsilon_over_4": epsilon_over_4,
                "coefficient_key": (
                    array_name if index is None else f"{array_name}[{index}]"
                ),
            }
            detail_rows.append({**identity, **statistics})
            curve_rows.extend({**identity, **item} for item in curve)

    write_rows(results_dir / "effective_k_key_points.csv", detail_rows)
    write_rows(results_dir / "effective_k_curve.csv", curve_rows)
    write_rows(results_dir / "effective_k_summary.csv", quantile_summary(detail_rows))
    plot_relative_gap(
        results_dir,
        history,
        lower_bound,
        args.target_relative_gap,
    )
    metadata = {
        "definition": "max k such that F(x)-AT_k(x) <= epsilon_absolute/4",
        "region_caveat": (
            "Pointwise diagnostic at stored key points; it is not a supremum "
            "over the full near-optimal sublevel set."
        ),
        "offline_accounting": (
            "Diagnostic full-grid scans are excluded from optimizer queries and runtime."
        ),
        "numerical_lower_reference": lower_bound,
        "target_relative_gap": args.target_relative_gap,
        "epsilon_absolute": epsilon_absolute,
        "epsilon_over_4": epsilon_over_4,
        "files": {
            "target_crossing_runs": "target_crossing_runs.csv",
            "time_to_target_summary": "time_to_target_summary.csv",
            "effective_k_key_points": "effective_k_key_points.csv",
            "effective_k_curve": "effective_k_curve.csv",
            "effective_k_summary": "effective_k_summary.csv",
        },
    }
    (results_dir / "protocol_diagnostics.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Saved protocol diagnostics to {results_dir}", flush=True)


if __name__ == "__main__":
    main()
