"""Recreate the polynomial-activation cost/time figure from stored CSV data.

This module is intentionally read-only with respect to experimental artifacts:
it reads ``history.csv`` and ``run_config.json`` and writes only the requested
PNG.  No method is rerun and no stored metric is recomputed.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SMAX_METHOD = "SMax-SGM"
FULL_SGM_METHOD = "Full-grid SGM"
LSE_METHOD = "LSE--L-BFGS"
LP_METHOD = "Restricted LP exchange"
WLS_METHOD = "WLS"
FUNCTIONAL_METHOD = "Functional minimax ReLU"
CHEB_METHOD = "Chebyshev-node ReLU"
Y_COLUMN = "test_max_centered_logit_distortion"


def read_history(results_dir: Path) -> list[dict[str, str]]:
    history_path = results_dir / "history.csv"
    config_path = results_dir / "run_config.json"
    if not history_path.is_file() or not config_path.is_file():
        raise FileNotFoundError(
            f"{results_dir} must contain history.csv and run_config.json."
        )
    # Parsing the config is a schema/integrity check.  It is never modified.
    json.loads(config_path.read_text(encoding="utf-8"))
    with history_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{history_path} is empty.")
    return rows


def _numeric(row: dict[str, str], name: str) -> float:
    return float(row[name])


def stochastic_series(
    rows: list[dict[str, str]], subset_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    selected = [
        row
        for row in rows
        if row["method"] == SMAX_METHOD
        and int(row["subset_size"]) == subset_size
    ]
    seeds = sorted({int(row["seed"]) for row in selected})
    runs = [
        sorted(
            [row for row in selected if int(row["seed"]) == seed],
            key=lambda row: int(row["checkpoint"]),
        )
        for seed in seeds
    ]
    if not runs or any(len(run) != len(runs[0]) for run in runs):
        raise ValueError(
            f"SMax subset {subset_size} has inconsistent stored checkpoints."
        )
    queries = np.asarray(
        [
            [_numeric(row, "optimizer_value_queries") for row in run]
            for run in runs
        ]
    )
    runtimes = np.asarray(
        [
            [_numeric(row, "optimizer_runtime_seconds") for row in run]
            for run in runs
        ]
    )
    values = np.asarray(
        [[_numeric(row, Y_COLUMN) for row in run] for run in runs]
    )
    if not np.all(queries == queries[0]):
        raise ValueError(
            f"SMax subset {subset_size} seeds do not share query checkpoints."
        )
    return (
        queries[0],
        np.median(runtimes, axis=0),
        np.median(values, axis=0),
        np.quantile(values, 0.25, axis=0),
        np.quantile(values, 0.75, axis=0),
    )


def deterministic_series(
    rows: list[dict[str, str]], method: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = sorted(
        [row for row in rows if row["method"] == method],
        key=lambda row: int(row["checkpoint"]),
    )
    if not selected:
        raise ValueError(f"No stored trajectory for {method}.")
    return (
        np.asarray(
            [_numeric(row, "optimizer_value_queries") for row in selected]
        ),
        np.asarray(
            [_numeric(row, "optimizer_runtime_seconds") for row in selected]
        ),
        np.asarray([_numeric(row, Y_COLUMN) for row in selected]),
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
    rows: list[dict[str, str]],
    x_name: str,
) -> None:
    subset_sizes = sorted(
        {
            int(row["subset_size"])
            for row in rows
            if row["method"] == SMAX_METHOD
        }
    )
    blues = ("#0072B2", "#56B4E9", "#004C73")
    for index, subset_size in enumerate(subset_sizes):
        queries, runtimes, median, q25, q75 = stochastic_series(
            rows, subset_size
        )
        x = (
            queries
            if x_name == "optimizer_value_queries"
            else runtimes
        )
        color = blues[index % len(blues)]
        axis.plot(
            x,
            median,
            color=color,
            linewidth=2.1,
            label=rf"SMax, $m={subset_size}$",
        )
        axis.fill_between(
            x, q25, q75, color=color, alpha=0.16, linewidth=0
        )

    styles: dict[str, dict[str, Any]] = {
        FULL_SGM_METHOD: {
            "color": "#D55E00",
            "marker": "o",
            "label": "Full-grid SGM",
        },
        LSE_METHOD: {
            "color": "#000000",
            "linestyle": "-.",
            "label": "LSE–L-BFGS-B",
        },
        LP_METHOD: {
            "color": "#666666",
            "marker": "s",
            "drawstyle": "steps-post",
            "label": "Restricted LP exchange",
        },
    }
    for method, style in styles.items():
        queries, runtimes, values = deterministic_series(rows, method)
        x = (
            queries
            if x_name == "optimizer_value_queries"
            else runtimes
        )
        axis.plot(
            x,
            values,
            linewidth=1.8,
            markersize=3.0,
            **style,
        )

    # All iterative curves start from the same bounded WLS solution.  Its cost
    # is intentionally excluded, as in the VFD comparison, so place the stored
    # WLS value just to the left of the first positive optimizer checkpoint.
    positive_x: list[float] = []
    for row in rows:
        if row["method"] in (
            SMAX_METHOD,
            FULL_SGM_METHOD,
            LSE_METHOD,
            LP_METHOD,
        ):
            value = _numeric(row, x_name)
            if value > 0.0:
                positive_x.append(value)
    wls_rows = [row for row in rows if row["method"] == WLS_METHOD]
    if positive_x and wls_rows:
        left_edge = min(positive_x) * 0.65
        axis.scatter(
            [left_edge],
            [_numeric(wls_rows[-1], Y_COLUMN)],
            marker="D",
            s=42,
            color="#CC79A7",
            edgecolors="white",
            linewidths=0.5,
            zorder=5,
            label="Bounded WLS initialization",
        )


def make_plot(results_dir: Path, output: Path | None = None) -> Path:
    results_dir = Path(results_dir).resolve()
    rows = read_history(results_dir)
    output_path = (
        Path(output).resolve()
        if output is not None
        else results_dir / "polynomial_activation_cost_time_comparison.png"
    )
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.8, 4.15),
        constrained_layout=True,
        sharey=True,
    )
    for axis, x_name, x_label in zip(
        axes,
        ("optimizer_value_queries", "optimizer_runtime_seconds"),
        (
            "Optimizer component-value queries",
            "Optimizer runtime (seconds)",
        ),
    ):
        draw_curves(axis, rows, x_name)
        configure_axis(axis, x_label)
    axes[0].set_ylabel("Test maximum centered-logit distortion")
    axes[0].legend(fontsize=7.0, ncol=1)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "final",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    print(make_plot(args.results_dir, args.output))


if __name__ == "__main__":
    main()
