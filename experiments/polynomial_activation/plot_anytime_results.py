"""Recreate the budget-asymmetric training-gap figure from stored CSV/JSON."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SMAX = "SMax-SGM"
METHOD_STYLES = {
    SMAX: {"color": "#0072B2", "linestyle": "-", "linewidth": 2.4},
    "Full-grid SGM": {
        "color": "#D55E00",
        "linestyle": "-",
        "linewidth": 1.8,
        "marker": "o",
        "markersize": 3.0,
    },
    "LSE--L-BFGS": {
        "color": "#009E73",
        "linestyle": "-",
        "linewidth": 1.8,
        "marker": "s",
        "markersize": 3.0,
    },
    "Restricted LP exchange": {
        "color": "#666666",
        "linestyle": "-",
        "linewidth": 1.8,
    },
}
DISPLAY_NAMES = {
    SMAX: "SMax\u2013SGM, $m=1024$",
    "Full-grid SGM": "Full-grid SGM",
    "LSE--L-BFGS": "LSE\u2013L-BFGS-B",
    "Restricted LP exchange": "Restricted LP",
}


def endpoint_label(value: float) -> str:
    mantissa, exponent = f"{value:.1e}".split("e")
    return rf"${mantissa}\times 10^{{{int(exponent)}}}$"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path} is empty.")
    return rows


def aggregate_smax(
    rows: list[dict[str, str]], x_name: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["method"] == SMAX:
            grouped[int(row["checkpoint"])].append(row)
    x_values: list[float] = []
    medians: list[float] = []
    q25_values: list[float] = []
    q75_values: list[float] = []
    for checkpoint in sorted(grouped):
        members = grouped[checkpoint]
        x_values.append(float(np.median([float(row[x_name]) for row in members])))
        gaps = np.asarray([float(row["relative_training_gap"]) for row in members])
        medians.append(float(np.median(gaps)))
        q25_values.append(float(np.quantile(gaps, 0.25)))
        q75_values.append(float(np.quantile(gaps, 0.75)))
    return (
        np.asarray(x_values),
        np.asarray(medians),
        np.asarray(q25_values),
        np.asarray(q75_values),
    )


def make_plot(results_dir: Path, output: Path | None = None) -> Path:
    results_dir = results_dir.resolve()
    rows = read_rows(results_dir / "history.csv")
    config = json.loads((results_dir / "run_config.json").read_text(encoding="utf-8"))
    protocol = config.get("budget_protocol", config.get("budget_asymmetric_protocol"))
    if protocol is None:
        raise KeyError("Missing budget protocol in run_config.json")
    target = float(protocol["target_relative_gap"])
    lower_y_limit = 5e-3
    if output is None:
        output = results_dir / "anytime_relative_training_gap.png"
    output = output.resolve()

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.0, 4.35),
        sharey=True,
        constrained_layout=True,
    )
    x_columns = (
        "optimizer_value_queries",
        "optimizer_runtime_seconds",
    )
    x_labels = (
        "Optimizer component-value queries",
        "Optimizer runtime (seconds)",
    )
    for axis, x_name, x_label in zip(axes, x_columns, x_labels):
        x, median, q25, q75 = aggregate_smax(rows, x_name)
        positive = (x > 0.0) & (median > 0.0)
        style = METHOD_STYLES[SMAX]
        axis.fill_between(
            x[positive],
            q25[positive],
            q75[positive],
            color=style["color"],
            alpha=0.16,
            linewidth=0.0,
        )
        axis.plot(
            x[positive],
            median[positive],
            label=DISPLAY_NAMES[SMAX],
            **style,
        )
        for method in ("Full-grid SGM", "LSE--L-BFGS", "Restricted LP exchange"):
            method_rows = sorted(
                (row for row in rows if row["method"] == method),
                key=lambda row: float(row[x_name]),
            )
            method_x = np.asarray([float(row[x_name]) for row in method_rows])
            method_y = np.asarray(
                [float(row["relative_training_gap"]) for row in method_rows]
            )
            keep = (method_x > 0.0) & (method_y > 0.0)
            method_x = method_x[keep]
            method_y = method_y[keep]
            if method == "Restricted LP exchange":
                clipped_y = np.maximum(method_y, lower_y_limit)
                axis.plot(
                    method_x,
                    clipped_y,
                    label=DISPLAY_NAMES[method],
                    drawstyle="steps-post",
                    **METHOD_STYLES[method],
                )
                if len(method_x) > 1:
                    axis.scatter(
                        method_x[:-1],
                        clipped_y[:-1],
                        marker="s",
                        s=18,
                        color=METHOD_STYLES[method]["color"],
                        zorder=4,
                    )
                axis.annotate(
                    endpoint_label(method_y[-1]),
                    xy=(method_x[-1], lower_y_limit),
                    xytext=(10, 23),
                    textcoords="offset points",
                    ha="left",
                    va="bottom",
                    arrowprops={
                        "arrowstyle": "->",
                        "color": METHOD_STYLES[method]["color"],
                        "linewidth": 0.9,
                    },
                )
            else:
                axis.plot(
                    method_x,
                    method_y,
                    label=DISPLAY_NAMES[method],
                    **METHOD_STYLES[method],
                )
        axis.axhline(
            target,
            color="#555555",
            linestyle="--",
            linewidth=1.1,
            label=f"{100 * target:g}% target" if axis is axes[0] else None,
        )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_ylim(bottom=lower_y_limit)
        axis.set_yticks([1e-2, 2e-2, 5e-2, 1e-1, 2e-1, 5e-1])
        axis.set_yticklabels(
            [
                r"$10^{-2}$",
                r"$2\times10^{-2}$",
                r"$5\times10^{-2}$",
                r"$10^{-1}$",
                r"$2\times10^{-1}$",
                r"$5\times10^{-1}$",
            ]
        )
        axis.set_xlabel(x_label)
        axis.grid(True, which="major", color="#808080", alpha=0.35, linewidth=0.7)
        axis.grid(True, which="minor", color="#A0A0A0", alpha=0.14, linewidth=0.45)
    axes[0].set_ylabel("Relative gap to numerical LP reference")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, loc="best", framealpha=0.9, fontsize=8.2)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=script_dir / "results" / "anytime_final",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    print(make_plot(args.results_dir, args.output))


if __name__ == "__main__":
    main()
