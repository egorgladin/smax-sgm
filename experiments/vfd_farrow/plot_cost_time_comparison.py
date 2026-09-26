"""Recreate the VFD cost/time plots entirely from stored checkpoint values."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SMAX_METHOD = "SMax, m=4096"
SGM_METHOD = "Full-grid SGM"
LSE_METHOD = "LSE--L-BFGS"
SOCP_METHOD = "Restricted-SOCP exchange"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    script_dir = Path(__file__).parent
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=script_dir / "results_virtual_age_comparison",
    )
    parser.add_argument(
        "--lse-results-dir",
        type=Path,
        default=script_dir / "results_full_grid",
    )
    parser.add_argument(
        "--matched-results-dir",
        type=Path,
        default=script_dir / "results_long_smax",
    )
    parser.add_argument("--subset-size", type=int, default=4096)
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


def normalized_method(method: str, subset_size: int) -> str:
    aliases = {
        f"SMax, virtual age, m={subset_size}": f"SMax, m={subset_size}",
        "Full-grid SGM, virtual age": SGM_METHOD,
    }
    return aliases.get(method, method)


def lse_rows(
    lse_results_dir: Path,
    matched_results_dir: Path,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    source = [
        row
        for row in read_rows(lse_results_dir / "history.csv")
        if row["method"] == LSE_METHOD
    ]
    if not source:
        raise ValueError("The stored full-grid history has no LSE trajectory.")
    matched_config = json.loads(
        (matched_results_dir / "run_config.json").read_text(encoding="utf-8")
    )
    optimizer_runtime = float(
        matched_config["lse"]["timing_median_seconds"]
    )
    wls_runtime = float(matched_config["common_wls"]["runtime_seconds"])
    wls_queries = int(
        matched_config["common_wls"][
            "value_queries_charged_in_total_column"
        ]
    )
    source_final_runtime = max(float(row["elapsed_seconds"]) for row in source)
    runtime_scale = optimizer_runtime / source_final_runtime
    rows: list[dict[str, object]] = []
    for row in source:
        rows.append(
            {
                "method": LSE_METHOD,
                "seed": "",
                "optimizer_value_queries": int(row["value_queries"]),
                "total_value_queries": (
                    wls_queries + int(row["value_queries"])
                ),
                "optimizer_runtime_seconds": (
                    runtime_scale * float(row["elapsed_seconds"])
                ),
                "runtime_including_wls_seconds": (
                    wls_runtime
                    + runtime_scale * float(row["elapsed_seconds"])
                ),
                "validation_max_error": float(row["validation_max_error"]),
            }
        )
    metadata: dict[str, object] = {
        "source": str(lse_results_dir / "history.csv"),
        "stored_checkpoints": len(rows),
        "optimizer_runtime_seconds": optimizer_runtime,
        "wls_runtime_seconds": wls_runtime,
        "wls_value_queries": wls_queries,
        "runtime_rescaling": (
            "stored trajectory elapsed times scaled to the matched-run median "
            "optimizer time; common matched-run WLS time then added"
        ),
        "final_total_value_queries": int(rows[-1]["total_value_queries"]),
        "final_runtime_including_wls_seconds": float(
            rows[-1]["runtime_including_wls_seconds"]
        ),
        "final_validation_max_error": float(
            rows[-1]["validation_max_error"]
        ),
    }
    return rows, metadata


def augment_history(
    results_dir: Path,
    lse_results_dir: Path,
    matched_results_dir: Path,
    subset_size: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    current = read_rows(results_dir / "history.csv")
    config = json.loads(
        (results_dir / "run_config.json").read_text(encoding="utf-8")
    )
    matched_config = json.loads(
        (matched_results_dir / "run_config.json").read_text(encoding="utf-8")
    )
    common_wls_queries = int(
        config["common_wls"]["total_value_queries"]
    )
    common_wls_runtime = float(config["common_wls"]["runtime_seconds"])
    matched_wls_runtime = float(
        matched_config["common_wls"]["runtime_seconds"]
    )
    rows: list[dict[str, object]] = []
    for row in current:
        method = normalized_method(row["method"], subset_size)
        if method == LSE_METHOD:
            continue
        wls_runtime = (
            matched_wls_runtime
            if method == f"SMax, m={subset_size}"
            else common_wls_runtime
        )
        rows.append(
            {
                "method": method,
                "seed": row["seed"],
                "optimizer_value_queries": int(
                    row.get("optimizer_value_queries", "")
                    or (
                        int(row["total_value_queries"])
                        - common_wls_queries
                    )
                ),
                "total_value_queries": int(row["total_value_queries"]),
                "optimizer_runtime_seconds": float(
                    row.get("optimizer_runtime_seconds", "")
                    or (
                        float(row["runtime_including_wls_seconds"])
                        - wls_runtime
                    )
                ),
                "runtime_including_wls_seconds": float(
                    row["runtime_including_wls_seconds"]
                ),
                "validation_max_error": float(
                    row["validation_max_error"]
                ),
            }
        )
    stored_lse, metadata = lse_rows(
        lse_results_dir,
        matched_results_dir,
    )
    rows.extend(stored_lse)
    write_rows(results_dir / "history.csv", rows)
    return rows, metadata


def stochastic_series(
    rows: list[dict[str, object]],
    method: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = [row for row in rows if row["method"] == method]
    seeds = sorted({int(row["seed"]) for row in selected})
    if not seeds:
        raise ValueError(f"No stochastic rows found for {method}.")
    runs = [
        sorted(
            [row for row in selected if int(row["seed"]) == seed],
            key=lambda row: int(row["optimizer_value_queries"]),
        )
        for seed in seeds
    ]
    queries = np.asarray(
        [
            [int(row["optimizer_value_queries"]) for row in run]
            for run in runs
        ],
        dtype=np.int64,
    )
    runtimes = np.asarray(
        [
            [float(row["optimizer_runtime_seconds"]) for row in run]
            for run in runs
        ],
        dtype=np.float64,
    )
    values = np.asarray(
        [
            [float(row["validation_max_error"]) for row in run]
            for run in runs
        ],
        dtype=np.float64,
    )
    if not np.all(queries == queries[0]):
        raise ValueError(f"{method} runs do not share query checkpoints.")
    return queries[0], np.median(runtimes, axis=0), values


def deterministic_series(
    rows: list[dict[str, object]],
    method: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = sorted(
        [row for row in rows if row["method"] == method],
        key=lambda row: int(row["optimizer_value_queries"]),
    )
    if not selected:
        raise ValueError(f"No stored trajectory found for {method}.")
    return (
        np.asarray(
            [int(row["optimizer_value_queries"]) for row in selected],
            dtype=np.int64,
        ),
        np.asarray(
            [
                float(row["optimizer_runtime_seconds"])
                for row in selected
            ],
            dtype=np.float64,
        ),
        np.asarray(
            [float(row["validation_max_error"]) for row in selected],
            dtype=np.float64,
        ),
    )


def configure_axis(axis: plt.Axes, x_label: str) -> None:
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel(x_label)
    axis.set_axisbelow(True)
    axis.grid(
        True,
        which="major",
        axis="x",
        color="#A8A8A8",
        alpha=0.42,
        linewidth=0.7,
    )
    axis.grid(
        True,
        which="minor",
        axis="x",
        color="#C8C8C8",
        alpha=0.24,
        linewidth=0.5,
    )
    axis.grid(
        True,
        which="both",
        axis="y",
        color="#989898",
        alpha=0.55,
        linewidth=0.7,
    )


def draw_curves(
    axis: plt.Axes,
    series: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    x_index: int,
) -> None:
    smax = series[SMAX_METHOD]
    smax_x = smax[x_index]
    axis.plot(
        smax_x,
        np.median(smax[2], axis=0),
        color="#0072B2",
        linewidth=2.1,
        label=r"SMax, $m=4096$",
    )
    axis.fill_between(
        smax_x,
        np.quantile(smax[2], 0.25, axis=0),
        np.quantile(smax[2], 0.75, axis=0),
        color="#0072B2",
        alpha=0.16,
        linewidth=0,
    )

    sgm = series[SGM_METHOD]
    axis.plot(
        sgm[x_index],
        sgm[2],
        color="#D55E00",
        linewidth=2.0,
        marker="o",
        markersize=3.0,
        label="Full-grid SGM",
    )

    lse = series[LSE_METHOD]
    axis.plot(
        lse[x_index],
        lse[2],
        color="#000000",
        linestyle="-.",
        linewidth=1.8,
        label="LSE--L-BFGS",
    )

    socp = series[SOCP_METHOD]
    axis.plot(
        socp[x_index],
        socp[2],
        color="#666666",
        linewidth=1.8,
        marker="s",
        markersize=2.8,
        drawstyle="steps-post",
        label="Restricted-SOCP exchange",
    )


def plot_single(
    path: Path,
    series: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    x_index: int,
) -> None:
    x_label = (
        "Optimizer component-value queries"
        if x_index == 0
        else "Optimizer runtime (seconds)"
    )
    fig, axis = plt.subplots(figsize=(7.1, 4.5), constrained_layout=True)
    draw_curves(axis, series, x_index)
    configure_axis(axis, x_label)
    axis.set_ylabel("Validation maximum complex error")
    axis.legend(fontsize=8.2)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_combined(
    path: Path,
    series: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.8, 4.15),
        constrained_layout=True,
        sharey=True,
    )
    for axis, x_index, x_label in zip(
        axes,
        (0, 1),
        (
            "Optimizer component-value queries",
            "Optimizer runtime (seconds)",
        ),
    ):
        draw_curves(axis, series, x_index)
        configure_axis(axis, x_label)
    axes[0].set_ylabel("Validation maximum complex error")
    axes[0].legend(fontsize=7.6)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def final_summary(
    rows: list[dict[str, object]],
    series: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    for method in (SMAX_METHOD, SGM_METHOD, LSE_METHOD, SOCP_METHOD):
        queries, runtimes, values = series[method]
        if method == SMAX_METHOD:
            runs = len(
                {
                    int(row["seed"])
                    for row in rows
                    if row["method"] == method
                }
            )
            final_value = float(np.median(values[:, -1]))
        else:
            runs = 1
            final_value = float(values[-1])
        summary.append(
            {
                "method": method,
                "runs": runs,
                "validation_final": final_value,
                "optimizer_value_queries": int(queries[-1]),
                "total_value_queries": int(
                    np.median(
                        [
                            int(row["total_value_queries"])
                            for row in rows
                            if row["method"] == method
                            and int(row["optimizer_value_queries"])
                            == int(queries[-1])
                        ]
                    )
                ),
                "optimizer_runtime_seconds": float(runtimes[-1]),
                "runtime_including_wls_seconds": float(
                    np.median(
                        [
                            float(row["runtime_including_wls_seconds"])
                            for row in rows
                            if row["method"] == method
                            and int(row["optimizer_value_queries"])
                            == int(queries[-1])
                        ]
                    )
                ),
            }
        )
    return summary


def update_results(
    results_dir: Path,
    lse_results_dir: Path,
    matched_results_dir: Path,
    subset_size: int = 4096,
    refresh_lse: bool = False,
) -> None:
    current = read_rows(results_dir / "history.csv")
    has_lse = any(row["method"] == LSE_METHOD for row in current)
    if refresh_lse or not has_lse:
        rows, lse_metadata = augment_history(
            results_dir,
            lse_results_dir,
            matched_results_dir,
            subset_size,
        )
    else:
        config = json.loads(
            (results_dir / "run_config.json").read_text(encoding="utf-8")
        )
        common_wls_queries = int(
            config["common_wls"]["total_value_queries"]
        )
        common_wls_runtime = float(config["common_wls"]["runtime_seconds"])
        matched_wls_runtime = float(
            config["lse"]["wls_runtime_seconds"]
        )
        rows = [
            {
                "method": (
                    method := normalized_method(row["method"], subset_size)
                ),
                "seed": row["seed"],
                "optimizer_value_queries": int(
                    row.get("optimizer_value_queries", "")
                    or (
                        int(row["total_value_queries"])
                        - common_wls_queries
                    )
                ),
                "total_value_queries": int(row["total_value_queries"]),
                "optimizer_runtime_seconds": float(
                    row.get("optimizer_runtime_seconds", "")
                    or (
                        float(row["runtime_including_wls_seconds"])
                        - (
                            matched_wls_runtime
                            if method in {
                                f"SMax, m={subset_size}",
                                LSE_METHOD,
                            }
                            else common_wls_runtime
                        )
                    )
                ),
                "runtime_including_wls_seconds": float(
                    row["runtime_including_wls_seconds"]
                ),
                "validation_max_error": float(
                    row["validation_max_error"]
                ),
            }
            for row in current
        ]
        lse_metadata = config["lse"]
        write_rows(results_dir / "history.csv", rows)
    series = {
        SMAX_METHOD: stochastic_series(rows, f"SMax, m={subset_size}"),
        SGM_METHOD: deterministic_series(rows, SGM_METHOD),
        LSE_METHOD: deterministic_series(rows, LSE_METHOD),
        SOCP_METHOD: deterministic_series(rows, SOCP_METHOD),
    }
    plot_single(
        results_dir / "convergence_vs_total_value_queries.png",
        series,
        x_index=0,
    )
    plot_single(
        results_dir / "convergence_vs_total_runtime.png",
        series,
        x_index=1,
    )
    plot_combined(
        results_dir / "virtual_age_cost_time_comparison.png",
        series,
    )
    write_rows(
        results_dir / "final_summary.csv",
        final_summary(rows, series),
    )

    config_path = results_dir / "run_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["lse"] = lse_metadata
    config["accounting"]["query_axis"] = (
        "optimizer component-value queries; common WLS scan excluded"
    )
    config["accounting"]["runtime_axis"] = (
        "measured optimizer runtime; WLS time excluded"
    )
    config["accounting"]["stored_coordinates"] = (
        "history.csv stores both optimizer-only and WLS-inclusive coordinates"
    )
    config["plot_reproduction"] = {
        "requires_method_reruns": False,
        "stored_history": str(results_dir / "history.csv"),
        "script": str(Path(__file__)),
        "command": (
            "python experiments/vfd_farrow/plot_cost_time_comparison.py"
        ),
        "plotted_methods": [
            f"SMax, m={subset_size}",
            SGM_METHOD,
            LSE_METHOD,
            SOCP_METHOD,
        ],
    }
    config_path.write_text(
        json.dumps(config, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    update_results(
        args.results_dir,
        args.lse_results_dir,
        args.matched_results_dir,
        args.subset_size,
    )
    print(f"Recreated plots from stored histories in {args.results_dir}")


if __name__ == "__main__":
    main()
