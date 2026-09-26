"""Compute the empirical effective top-k size at key stored checkpoints.

The script is offline: it reads stored coefficients, performs exact full
training-set scans, and does not alter optimizer query or runtime accounting.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

import run_experiment as base


COEFFICIENT_KEY = re.compile(r"^(?P<array>.+)\[(?P<index>\d+)\]$")


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=script_dir / "features" / "cifar100_resnet20_q14",
    )
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--sample-chunk", type=int, default=4096)
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


def coefficient_for_row(
    row: dict[str, str], arrays: np.lib.npyio.NpzFile
) -> np.ndarray:
    match = COEFFICIENT_KEY.fullmatch(row["coefficient_key"])
    if match is None:
        raise ValueError(f"Invalid coefficient key {row['coefficient_key']!r}.")
    return np.asarray(
        arrays[match.group("array")][int(match.group("index"))],
        dtype=np.float64,
    )


def select_key_rows(
    history: list[dict[str, str]], target: float
) -> list[tuple[str, dict[str, str]]]:
    selected: list[tuple[str, dict[str, str]]] = []
    wls = [row for row in history if row["method"] == base.WLS_METHOD]
    if len(wls) != 1:
        raise ValueError(f"Expected one WLS row, found {len(wls)}.")
    selected.append(("wls_start", wls[0]))

    smax = [row for row in history if row["method"] == base.SMAX_METHOD]
    by_seed: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in smax:
        by_seed[int(row["seed"])].append(row)
    for seed, rows in sorted(by_seed.items()):
        rows.sort(key=lambda row: int(float(row["optimizer_value_queries"])))
        crossing = next(
            (
                row
                for row in rows
                if float(row["relative_training_gap"]) <= target
            ),
            None,
        )
        if crossing is not None:
            selected.append(("smax_first_target_crossing", crossing))
        selected.append(("smax_final", rows[-1]))

    reference = [
        row for row in history if row["method"] == base.LP_METHOD
    ]
    if not reference:
        raise ValueError("No restricted-LP reference trajectory found.")
    reference.sort(key=lambda row: int(float(row["optimizer_value_queries"])))
    selected.append(("restricted_lp_reference", reference[-1]))
    return selected


def component_values(
    split: base.SplitData, coefficient: np.ndarray, sample_chunk: int
) -> np.ndarray:
    values = np.empty(split.n_components, dtype=np.float64)
    for start, stop, phi, target in split.iter_centered_chunks(sample_chunk):
        values[start:stop] = np.abs(phi @ coefficient - target)
    return values


def effective_k_statistics(
    values: np.ndarray,
    threshold: float,
    curve_points: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ordered = np.sort(np.asarray(values, dtype=np.float64))[::-1]
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
            ""
            if effective_k == len(ordered)
            else float(gaps[effective_k])
        ),
    }
    return summary, curve


def quantile_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["point_role"])].append(row)
    result: list[dict[str, Any]] = []
    numeric_fields = (
        "optimizer_value_queries",
        "optimizer_runtime_seconds",
        "training_objective",
        "relative_training_gap",
        "effective_k",
        "effective_k_fraction",
        "N_over_effective_k",
        "gap_at_effective_k",
    )
    for role, members in grouped.items():
        output: dict[str, Any] = {"point_role": role, "replicate_count": len(members)}
        for field in numeric_fields:
            values = np.asarray([float(row[field]) for row in members])
            output[f"{field}_median"] = float(np.median(values))
            output[f"{field}_q25"] = float(np.quantile(values, 0.25))
            output[f"{field}_q75"] = float(np.quantile(values, 0.75))
        result.append(output)
    return result


def main() -> None:
    args = parse_args()
    if args.sample_chunk < 1 or args.curve_points < 2:
        raise ValueError("Positive sample chunk and at least two curve points required.")
    if not 0.0 < args.target_relative_gap < 1.0:
        raise ValueError("The target relative gap must lie in (0,1).")

    results_dir = args.results_dir.resolve()
    history = read_rows(results_dir / "history.csv")
    lower_bounds = {
        float(row["certified_training_lower_bound"])
        for row in history
        if row.get("certified_training_lower_bound")
    }
    if len(lower_bounds) != 1:
        raise ValueError(f"Expected one numerical lower reference, got {lower_bounds}.")
    lower_bound = lower_bounds.pop()
    epsilon_absolute = args.target_relative_gap * lower_bound
    threshold = epsilon_absolute / 4.0

    problem = base.load_cached_problem(args.feature_dir.resolve(), None, None)
    key_rows = select_key_rows(history, args.target_relative_gap)
    detail_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    with np.load(results_dir / "coefficients.npz", allow_pickle=False) as arrays:
        for position, (role, row) in enumerate(key_rows, start=1):
            print(
                f"[{position}/{len(key_rows)}] {role}, seed={row.get('seed', '')}",
                flush=True,
            )
            coefficient = coefficient_for_row(row, arrays)
            values = component_values(problem.train, coefficient, args.sample_chunk)
            summary, curve = effective_k_statistics(
                values, threshold, args.curve_points
            )
            identity = {
                "point_role": role,
                "method": row["method"],
                "seed": row.get("seed", ""),
                "optimizer_value_queries": int(float(row["optimizer_value_queries"])),
                "optimizer_runtime_seconds": float(
                    row["optimizer_runtime_seconds"]
                ),
                "training_objective": float(
                    row["train_max_centered_logit_distortion"]
                ),
                "relative_training_gap": float(row["relative_training_gap"]),
                "numerical_lower_reference": lower_bound,
                "target_relative_gap": args.target_relative_gap,
                "epsilon_absolute": epsilon_absolute,
                "epsilon_over_4": threshold,
                "coefficient_key": row["coefficient_key"],
            }
            detail_rows.append({**identity, **summary})
            curve_rows.extend({**identity, **item} for item in curve)

    write_rows(results_dir / "effective_k_key_points.csv", detail_rows)
    write_rows(results_dir / "effective_k_curve.csv", curve_rows)
    write_rows(
        results_dir / "effective_k_summary.csv", quantile_summary(detail_rows)
    )
    metadata = {
        "definition": "max k such that F(x)-AT_k(x) <= epsilon_absolute/4",
        "region_caveat": (
            "Pointwise diagnostic at stored key points; it is not a supremum "
            "over the full near-optimal sublevel set."
        ),
        "numerical_lower_reference": lower_bound,
        "target_relative_gap": args.target_relative_gap,
        "epsilon_absolute": epsilon_absolute,
        "epsilon_over_4": threshold,
        "key_point_count": len(detail_rows),
        "files": {
            "key_points": "effective_k_key_points.csv",
            "curve": "effective_k_curve.csv",
            "summary": "effective_k_summary.csv",
        },
    }
    (results_dir / "effective_k_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Saved effective-k diagnostics to {results_dir}", flush=True)


if __name__ == "__main__":
    main()
