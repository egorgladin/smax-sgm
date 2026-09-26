"""Finite-max optimization for a polynomial activation in private inference.

The cached instance is produced by ``extract_cifar100_features.py``.  For each
image and class it stores a Chebyshev feature row and the original final-ReLU
logit contribution.  This script works in the normalized coefficient
parameterization

    p_y(z) = B * sum_{r=0}^q y_r T_r(z / B)

and minimizes the maximum *centered-logit* distortion.  Centering each image
across classes removes the logit-shift direction, which cannot affect softmax
probabilities or classifications.

All optimization trajectories and all metrics needed to recreate the figure
are stored.  Test-set evaluation is offline and is never used to select an
iterate, a stepsize, or a stopping time.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

import numpy as np
import scipy
from scipy.optimize import linprog, minimize


SMAX_METHOD = "SMax-SGM"
FULL_SGM_METHOD = "Full-grid SGM"
LSE_METHOD = "LSE--L-BFGS"
LP_METHOD = "Restricted LP exchange"
WLS_METHOD = "WLS"
CHEB_METHOD = "Chebyshev-node ReLU"
FUNCTIONAL_METHOD = "Functional minimax ReLU"
STATIC_METHODS = {WLS_METHOD, CHEB_METHOD, FUNCTIONAL_METHOD}


@dataclass
class SplitData:
    """Memory-mapped split with centered-row access."""

    name: str
    phi: np.ndarray
    b: np.ndarray
    labels: np.ndarray
    original_predictions: np.ndarray
    original_logits: np.ndarray
    fc_bias: np.ndarray
    scale_B: float
    normalized_coordinate: bool = True
    phi_includes_scale_B_and_centering: bool = True
    _phi_mean: np.ndarray | None = field(default=None, init=False, repr=False)
    _b_mean: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.phi.ndim != 3:
            raise ValueError(
                f"{self.name}_phi must have shape [image,class,order], "
                f"got {self.phi.shape}."
            )
        if self.b.shape != self.phi.shape[:2]:
            raise ValueError(
                f"{self.name}_b shape {self.b.shape} is incompatible with "
                f"phi shape {self.phi.shape}."
            )
        if self.labels.shape != (self.phi.shape[0],):
            raise ValueError(
                f"{self.name}_labels has shape {self.labels.shape}; expected "
                f"{(self.phi.shape[0],)}."
            )
        if self.original_predictions.shape != (self.phi.shape[0],):
            raise ValueError(
                f"{self.name}_original_predictions has shape "
                f"{self.original_predictions.shape}; expected "
                f"{(self.phi.shape[0],)}."
            )
        if self.original_logits.shape != self.b.shape:
            raise ValueError(
                f"{self.name}_original_logits has shape "
                f"{self.original_logits.shape}; expected {self.b.shape}."
            )
        if self.fc_bias.shape != (self.phi.shape[1],):
            raise ValueError(
                f"fc_bias has shape {self.fc_bias.shape}; expected "
                f"{(self.phi.shape[1],)}."
            )
        if not math.isfinite(self.scale_B) or self.scale_B <= 0.0:
            raise ValueError(f"Invalid activation scale B={self.scale_B}.")

    @property
    def n_samples(self) -> int:
        return int(self.phi.shape[0])

    @property
    def n_classes(self) -> int:
        return int(self.phi.shape[1])

    @property
    def dimension(self) -> int:
        return int(self.phi.shape[2])

    @property
    def n_components(self) -> int:
        return self.n_samples * self.n_classes

    @property
    def degree(self) -> int:
        return self.dimension - 1

    @property
    def design_multiplier(self) -> float:
        # Extractor schema v1 stores the affine design for the normalized
        # parameterization directly: phi already includes B and is centered.
        # The fallback keeps compatibility with an older/raw T_r design.
        return 1.0 if self.phi_includes_scale_B_and_centering else self.scale_B

    @property
    def centered_bias(self) -> np.ndarray:
        bias = np.asarray(self.fc_bias, dtype=np.float64)
        return bias - float(np.mean(bias))

    def truncate(self, samples: int | None) -> "SplitData":
        if samples is None or samples >= self.n_samples:
            return self
        if samples < 1:
            raise ValueError("A split limit must be positive.")
        return SplitData(
            name=self.name,
            phi=self.phi[:samples],
            b=self.b[:samples],
            labels=self.labels[:samples],
            original_predictions=self.original_predictions[:samples],
            original_logits=self.original_logits[:samples],
            fc_bias=self.fc_bias,
            scale_B=self.scale_B,
            normalized_coordinate=self.normalized_coordinate,
            phi_includes_scale_B_and_centering=(
                self.phi_includes_scale_B_and_centering
            ),
        )

    def prepare_random_access_means(self, sample_chunk: int) -> float:
        """Cache per-image class means for exact random row gathering."""

        if self._phi_mean is not None and self._b_mean is not None:
            return 0.0
        start_time = time.perf_counter()
        self._phi_mean = np.empty(
            (self.n_samples, self.dimension), dtype=np.float32
        )
        self._b_mean = np.empty(self.n_samples, dtype=np.float32)
        for start in range(0, self.n_samples, sample_chunk):
            stop = min(start + sample_chunk, self.n_samples)
            raw_phi = np.asarray(self.phi[start:stop], dtype=np.float32)
            raw_b = np.asarray(self.b[start:stop], dtype=np.float32)
            self._phi_mean[start:stop] = np.mean(
                raw_phi, axis=1, dtype=np.float32
            )
            self._b_mean[start:stop] = np.mean(
                raw_b, axis=1, dtype=np.float32
            )
        return time.perf_counter() - start_time

    def iter_centered_chunks(
        self, sample_chunk: int
    ) -> Iterator[tuple[int, int, np.ndarray, np.ndarray]]:
        """Yield contiguous centered component rows in float64."""

        multiplier = self.design_multiplier
        for sample_start in range(0, self.n_samples, sample_chunk):
            sample_stop = min(sample_start + sample_chunk, self.n_samples)
            raw_phi = np.asarray(
                self.phi[sample_start:sample_stop], dtype=np.float64
            )
            raw_b = np.asarray(self.b[sample_start:sample_stop], dtype=np.float64)
            centered_phi = raw_phi - np.mean(raw_phi, axis=1, keepdims=True)
            centered_phi *= multiplier
            centered_b = raw_b - np.mean(raw_b, axis=1, keepdims=True)
            row_start = sample_start * self.n_classes
            row_stop = sample_stop * self.n_classes
            yield (
                row_start,
                row_stop,
                centered_phi.reshape(-1, self.dimension),
                centered_b.reshape(-1),
            )

    def gather_centered_rows(
        self, indices: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Gather arbitrary flattened rows without replacement or approximation."""

        if self._phi_mean is None or self._b_mean is None:
            raise RuntimeError(
                "prepare_random_access_means must be called before row gathering."
            )
        flat = np.asarray(indices, dtype=np.int64)
        if flat.ndim != 1:
            raise ValueError("indices must be one-dimensional.")
        if len(flat) and (int(flat.min()) < 0 or int(flat.max()) >= self.n_components):
            raise IndexError("A component index lies outside the split.")
        samples = flat // self.n_classes
        classes = flat % self.n_classes
        raw_phi = np.asarray(self.phi[samples, classes, :], dtype=np.float64)
        raw_b = np.asarray(self.b[samples, classes], dtype=np.float64)
        centered_phi = (
            raw_phi
            - np.asarray(self._phi_mean[samples], dtype=np.float64)
        ) * self.design_multiplier
        centered_b = raw_b - np.asarray(self._b_mean[samples], dtype=np.float64)
        return centered_phi, centered_b


@dataclass
class CachedProblem:
    feature_dir: Path
    metadata: dict[str, Any]
    train: SplitData
    test: SplitData

    @property
    def dimension(self) -> int:
        return self.train.dimension

    @property
    def degree(self) -> int:
        return self.dimension - 1

    @property
    def scale_B(self) -> float:
        return self.train.scale_B


@dataclass
class Trajectory:
    method: str
    coefficients: np.ndarray
    optimizer_value_queries: np.ndarray
    optimizer_runtime_seconds: np.ndarray
    subgradient_queries: np.ndarray
    component_gradient_rows: np.ndarray
    seed: int | None = None
    subset_size: int | None = None
    lower_bounds: np.ndarray | None = None
    status: str = "ok"
    timing_repeats_seconds: list[float] = field(default_factory=list)
    raw_state: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.coefficients = np.asarray(self.coefficients, dtype=np.float64)
        if self.coefficients.ndim == 1:
            self.coefficients = self.coefficients[None, :]
        count = self.coefficients.shape[0]
        for name in (
            "optimizer_value_queries",
            "optimizer_runtime_seconds",
            "subgradient_queries",
            "component_gradient_rows",
        ):
            value = np.asarray(getattr(self, name))
            if value.shape != (count,):
                raise ValueError(
                    f"{self.method}: {name} has shape {value.shape}, "
                    f"expected {(count,)}."
                )
            setattr(self, name, value)
        if self.lower_bounds is None:
            self.lower_bounds = np.full(count, np.nan, dtype=np.float64)
        else:
            self.lower_bounds = np.asarray(self.lower_bounds, dtype=np.float64)
            if self.lower_bounds.shape != (count,):
                raise ValueError(
                    f"{self.method}: lower_bounds has shape "
                    f"{self.lower_bounds.shape}, expected {(count,)}."
                )


@dataclass
class Evaluation:
    train_max: np.ndarray
    test_max: np.ndarray
    test_samplewise_p999: np.ndarray
    test_agreement: np.ndarray
    test_certified_agreement: np.ndarray
    test_accuracy: np.ndarray


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=None,
        help=(
            "Extractor output directory. By default, choose the newest "
            "q=14 cache under features/."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "results" / "final",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--train-samples", type=int, default=None)
    parser.add_argument("--test-samples", type=int, default=None)
    parser.add_argument("--sample-chunk", type=int, default=512)
    parser.add_argument("--evaluation-coefficient-block", type=int, default=16)
    parser.add_argument("--value-budget", type=int, default=25_000_000)
    parser.add_argument(
        "--subset-sizes", type=int, nargs="+", default=[1024, 4096]
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=list(range(100, 110))
    )
    parser.add_argument("--history-points", type=int, default=24)
    parser.add_argument(
        "--functional-box-kappa",
        "--kappa",
        dest="functional_box_kappa",
        type=float,
        default=4.0,
        help=(
            "Half-width multiplier for the functionally anchored box: "
            "|y_r-y_fun,r| <= kappa*e_fun/(q+1)."
        ),
    )
    parser.add_argument(
        "--functional-grid-points",
        type=int,
        default=65_537,
        help="Odd dense-grid size used to construct and certify y_fun.",
    )
    parser.add_argument(
        "--eta-theta-candidates",
        type=float,
        nargs="+",
        default=[0.25, 1.0, 4.0, 16.0],
    )
    parser.add_argument(
        "--eta0",
        type=float,
        default=0.0,
        help="Positive value skips the train-only SMax stepsize pilot.",
    )
    parser.add_argument(
        "--full-sgm-eta0",
        type=float,
        default=0.0,
        help=(
            "Positive value skips the separate training-only full-grid SGM "
            "stepsize pilot."
        ),
    )
    parser.add_argument("--pilot-seed", type=int, default=0)
    parser.add_argument("--pilot-subset-size", type=int, default=4096)
    parser.add_argument("--pilot-value-budget", type=int, default=1_000_000)
    parser.add_argument(
        "--full-sgm-pilot-value-budget",
        type=int,
        default=25_000_000,
    )
    parser.add_argument(
        "--lse-multipliers",
        type=float,
        nargs="+",
        default=[1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4],
    )
    parser.add_argument("--lse-maxiter-per-stage", type=int, default=20)
    parser.add_argument("--exchange-initial-rows", type=int, default=96)
    parser.add_argument("--exchange-add-rows", type=int, default=48)
    parser.add_argument("--exchange-max-iterations", type=int, default=30)
    parser.add_argument("--exchange-absolute-tolerance", type=float, default=1e-7)
    parser.add_argument("--exchange-relative-tolerance", type=float, default=1e-6)
    parser.add_argument("--timing-repeats", type=int, default=2)
    parser.add_argument("--lp-timing-repeats", type=int, default=1)
    parser.add_argument("--tail-sample-size", type=int, default=200_000)
    parser.add_argument("--tail-seed", type=int, default=918273)
    return parser.parse_args()


def finalize_args(args: argparse.Namespace) -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    if args.smoke:
        if args.train_samples is None:
            args.train_samples = 256
        if args.test_samples is None:
            args.test_samples = 128
        args.value_budget = min(args.value_budget, 150_000)
        args.subset_sizes = [size for size in (16, 64) if size <= 25_600]
        args.seeds = [0, 1]
        args.history_points = min(args.history_points, 8)
        args.pilot_subset_size = 32
        args.pilot_value_budget = min(args.pilot_value_budget, 2_000)
        args.full_sgm_pilot_value_budget = min(
            args.full_sgm_pilot_value_budget, 150_000
        )
        args.lse_multipliers = [1e-1, 1e-2]
        args.lse_maxiter_per_stage = min(args.lse_maxiter_per_stage, 3)
        args.exchange_initial_rows = min(args.exchange_initial_rows, 24)
        args.exchange_add_rows = min(args.exchange_add_rows, 8)
        args.exchange_max_iterations = min(args.exchange_max_iterations, 4)
        args.timing_repeats = 1
        args.lp_timing_repeats = 1
        args.tail_sample_size = min(args.tail_sample_size, 5_000)
        if args.output_dir == script_dir / "results" / "final":
            args.output_dir = script_dir / "results_smoke"
    if args.feature_dir is None:
        args.feature_dir = find_default_feature_dir(script_dir / "features", args.smoke)
    args.feature_dir = args.feature_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    positive_integer_names = (
        "sample_chunk",
        "evaluation_coefficient_block",
        "value_budget",
        "history_points",
        "pilot_subset_size",
        "pilot_value_budget",
        "full_sgm_pilot_value_budget",
        "lse_maxiter_per_stage",
        "exchange_initial_rows",
        "exchange_add_rows",
        "exchange_max_iterations",
        "timing_repeats",
        "lp_timing_repeats",
        "tail_sample_size",
        "functional_grid_points",
    )
    for name in positive_integer_names:
        if int(getattr(args, name)) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if any(size < 1 for size in args.subset_sizes):
        raise ValueError("--subset-sizes must contain positive integers.")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds must be distinct.")
    if args.functional_box_kappa <= 0.0:
        raise ValueError("--functional-box-kappa must be positive.")
    if args.functional_grid_points < 1025:
        raise ValueError("--functional-grid-points must be at least 1025.")
    if args.eta0 < 0.0:
        raise ValueError("--eta0 cannot be negative.")
    if args.full_sgm_eta0 < 0.0:
        raise ValueError("--full-sgm-eta0 cannot be negative.")
    if any(theta <= 0.0 for theta in args.eta_theta_candidates):
        raise ValueError("Stepsize pilot multipliers must be positive.")
    if any(mu <= 0.0 for mu in args.lse_multipliers):
        raise ValueError("LSE multipliers must be positive.")
    if args.exchange_absolute_tolerance < 0.0:
        raise ValueError("Exchange absolute tolerance cannot be negative.")
    if args.exchange_relative_tolerance < 0.0:
        raise ValueError("Exchange relative tolerance cannot be negative.")


def find_default_feature_dir(root: Path, smoke: bool) -> Path:
    if not root.is_dir():
        expected = root / ("cifar100_resnet20_q14_smoke" if smoke else "cifar100_resnet20_q14")
        raise FileNotFoundError(
            f"No feature cache exists under {root}. Run "
            "extract_cifar100_features.py first, or pass --feature-dir. "
            f"Expected a directory such as {expected}."
        )
    candidates: list[tuple[int, float, Path]] = []
    for metadata_path in root.glob("*/metadata.json"):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            degree = int(metadata.get("problem", {}).get("degree_q", -1))
            train_rows = int(
                metadata.get("extraction", {})
                .get("train", {})
                .get("examples", 0)
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
        if degree != 14:
            continue
        smoke_name = "smoke" in metadata_path.parent.name.lower()
        preference = 2 if smoke_name == smoke else 1
        candidates.append(
            (
                preference * 10**12 + train_rows,
                metadata_path.stat().st_mtime,
                metadata_path.parent,
            )
        )
    if not candidates:
        raise FileNotFoundError(
            f"No q=14 feature cache with metadata.json was found under {root}."
        )
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def _load_array(feature_dir: Path, names: Sequence[str]) -> np.ndarray:
    for name in names:
        path = feature_dir / name
        if path.is_file():
            return np.load(path, mmap_mode="r", allow_pickle=False)
    raise FileNotFoundError(
        f"None of {list(names)} exists in feature directory {feature_dir}."
    )


def _reshape_schema(
    phi: np.ndarray,
    b: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    split: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Small abstraction for a possible flattened extractor schema."""

    if phi.ndim == 3 and b.ndim == 2:
        return phi, b
    n_samples = int(labels.shape[0])
    if phi.ndim == 2 and phi.shape[0] == n_samples * num_classes:
        phi = phi.reshape(n_samples, num_classes, phi.shape[1])
    if b.ndim == 1 and b.shape[0] == n_samples * num_classes:
        b = b.reshape(n_samples, num_classes)
    if phi.ndim != 3 or b.ndim != 2:
        raise ValueError(
            f"Unsupported {split} cache schema: phi={phi.shape}, b={b.shape}."
        )
    return phi, b


def load_cached_problem(
    feature_dir: Path,
    train_samples: int | None,
    test_samples: int | None,
) -> CachedProblem:
    metadata_path = feature_dir / "metadata.json"
    if not metadata_path.is_file():
        alternate = feature_dir / "manifest.json"
        if alternate.is_file():
            metadata_path = alternate
        else:
            raise FileNotFoundError(
                f"{feature_dir} has neither metadata.json nor manifest.json."
            )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    problem_meta = metadata.get("problem", {})
    degree = int(problem_meta.get("degree_q", metadata.get("degree", -1)))
    scale_B = float(
        problem_meta.get(
            "scale_B",
            metadata.get("scale_selection", {}).get("B", math.nan),
        )
    )
    if degree < 0 or not math.isfinite(scale_B):
        raise ValueError("Feature metadata lacks problem.degree_q or scale_B.")
    design_description = " ".join(
        str(problem_meta.get(key, ""))
        for key in ("basis", "normalized_activation", "design_equation")
    ).lower()
    phi_includes_scale = bool(
        problem_meta.get(
            "phi_includes_scale_B_and_centering",
            (
                "b * chebyshev" in design_description
                and "phi" in design_description
                and "y_r" in design_description
            ),
        )
    )

    fc_bias = _load_array(feature_dir, ("fc_bias.npy", "classifier_bias.npy"))
    num_classes = int(fc_bias.shape[0])

    def load_split(name: str) -> SplitData:
        phi = _load_array(feature_dir, (f"{name}_phi.npy", f"phi_{name}.npy"))
        b = _load_array(feature_dir, (f"{name}_b.npy", f"b_{name}.npy"))
        labels = _load_array(
            feature_dir, (f"{name}_labels.npy", f"labels_{name}.npy")
        )
        try:
            predictions = _load_array(
                feature_dir,
                (
                    f"{name}_original_predictions.npy",
                    f"{name}_predictions.npy",
                    f"predictions_{name}.npy",
                ),
            )
        except FileNotFoundError:
            predictions = np.asarray(
                np.argmax(
                    np.asarray(b, dtype=np.float64)
                    + np.asarray(fc_bias, dtype=np.float64)[None, :],
                    axis=1,
                ),
                dtype=np.int64,
            )
        phi, b = _reshape_schema(phi, b, labels, num_classes, name)
        try:
            original_logits = _load_array(
                feature_dir,
                (
                    f"{name}_original_logits.npy",
                    f"original_logits_{name}.npy",
                ),
            )
        except FileNotFoundError:
            # b is class-centered, which is enough to reconstruct all margins
            # and predictions after adding the FC bias.
            original_logits = np.asarray(
                b, dtype=np.float64
            ) + np.asarray(fc_bias, dtype=np.float64)[None, :]
        if phi.shape[2] != degree + 1:
            raise ValueError(
                f"{name} phi has {phi.shape[2]} orders but metadata says q={degree}."
            )
        return SplitData(
            name=name,
            phi=phi,
            b=b,
            labels=labels,
            original_predictions=predictions,
            original_logits=original_logits,
            fc_bias=fc_bias,
            scale_B=scale_B,
            normalized_coordinate=True,
            phi_includes_scale_B_and_centering=phi_includes_scale,
        )

    train = load_split("train").truncate(train_samples)
    test = load_split("test").truncate(test_samples)
    if train.dimension != test.dimension or train.n_classes != test.n_classes:
        raise ValueError("Training and test cache shapes are inconsistent.")
    return CachedProblem(feature_dir, metadata, train, test)


def checkpoint_iterations(iterations: int, points: int) -> np.ndarray:
    if iterations < 1:
        return np.empty(0, dtype=np.int64)
    raw = np.geomspace(1, iterations, min(iterations, points))
    checkpoints = np.unique(np.rint(raw).astype(np.int64))
    if checkpoints[-1] != iterations:
        checkpoints = np.append(checkpoints, iterations)
    return checkpoints


def chebyshev_node_relu_coefficients(degree: int) -> np.ndarray:
    """Interpolate normalized ReLU on degree+1 Chebyshev roots."""

    return np.polynomial.chebyshev.chebinterpolate(
        lambda values: np.maximum(values, 0.0),
        degree,
    ).astype(np.float64, copy=False)


def scalar_near_minimax_relu(
    degree: int,
    dense_grid_points: int,
    max_exchange_iterations: int = 30,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    """Compute a small exchange-LP near-minimax polynomial for ReLU on [-1,1].

    ``e_fun`` is a continuous-domain certificate, rather than just the sampled
    error.  If ``h`` is the dense-grid spacing, the error is Lipschitz with
    constant at most ``1 + sum_r |y_r| r^2``.  Consequently the grid maximum
    plus ``h/2`` times this bound dominates the error between grid points.
    """

    if dense_grid_points % 2 == 0:
        dense_grid_points += 1
    dense_x = np.linspace(-1.0, 1.0, dense_grid_points, dtype=np.float64)
    dense_target = np.maximum(dense_x, 0.0)
    dense_vander = np.polynomial.chebyshev.chebvander(dense_x, degree)

    # A modest working set makes the construction cheap, while exact dense
    # separation prevents the usual failure of an under-resolved scalar grid.
    initial_count = max(8 * (degree + 1), 129)
    active = np.unique(
        np.rint(np.linspace(0, dense_grid_points - 1, initial_count)).astype(
            np.int64
        )
    )
    lower_bound = math.nan
    status = "iteration_limit"
    iterations = 0
    for iterations in range(1, max_exchange_iterations + 1):
        vander = dense_vander[active]
        target = dense_target[active]
        constraints = np.vstack(
            (
                np.column_stack((vander, -np.ones(len(active)))),
                np.column_stack((-vander, -np.ones(len(active)))),
            )
        )
        rhs = np.concatenate((target, -target))
        objective = np.zeros(degree + 2, dtype=np.float64)
        objective[-1] = 1.0
        result = linprog(
            objective,
            A_ub=constraints,
            b_ub=rhs,
            bounds=[(None, None)] * (degree + 1) + [(0.0, None)],
            method="highs",
        )
        if not result.success:
            raise RuntimeError(
                "Scalar ReLU exchange LP failed: "
                f"{result.status} {result.message}"
            )
        y_fun = np.asarray(result.x[:-1], dtype=np.float64)
        lower_bound = float(result.x[-1])
        dense_error_vector = dense_vander @ y_fun - dense_target
        dense_absolute = np.abs(dense_error_vector)
        dense_error = float(np.max(dense_absolute))
        violation = dense_error - lower_bound
        if violation <= max(1e-13, 1e-10 * max(1.0, dense_error)):
            status = "separated"
            break
        add_count = min(4 * (degree + 2), dense_grid_points)
        candidates = np.argpartition(
            dense_absolute, dense_grid_points - add_count
        )[-add_count:]
        new_active = np.union1d(active, candidates)
        if len(new_active) == len(active):
            status = "stalled"
            break
        active = new_active

    derivative_lipschitz_bound = 1.0 + float(
        np.sum(
            np.abs(y_fun)
            * np.square(np.arange(degree + 1, dtype=np.float64))
        )
    )
    grid_spacing = 2.0 / (dense_grid_points - 1)
    certificate_padding = 0.5 * grid_spacing * derivative_lipschitz_bound
    e_fun = dense_error + certificate_padding
    diagnostics: dict[str, Any] = {
        "construction": "dense-separation epigraph LP in Chebyshev basis",
        "status": status,
        "exchange_iterations": iterations,
        "active_grid_rows": int(len(active)),
        "dense_grid_points": int(dense_grid_points),
        "discrete_lp_lower_bound": lower_bound,
        "dense_grid_sup_error": dense_error,
        "derivative_lipschitz_bound": derivative_lipschitz_bound,
        "grid_spacing": grid_spacing,
        "certificate_padding": certificate_padding,
        "continuous_sup_error_certificate_e_fun": e_fun,
    }
    return y_fun, e_fun, diagnostics


def streaming_wls(
    split: SplitData,
    sample_chunk: int,
    lower_bounds: np.ndarray | None = None,
    upper_bounds: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    gram = np.zeros((split.dimension, split.dimension), dtype=np.float64)
    rhs = np.zeros(split.dimension, dtype=np.float64)
    start_time = time.perf_counter()
    for _, _, phi, target in split.iter_centered_chunks(sample_chunk):
        gram += phi.T @ phi
        rhs += phi.T @ target
    normalization = float(max(1, split.n_components))
    gram /= normalization
    rhs /= normalization
    try:
        unconstrained = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        unconstrained = np.linalg.lstsq(gram, rhs, rcond=None)[0]
    y = np.asarray(unconstrained, dtype=np.float64)
    if (lower_bounds is None) != (upper_bounds is None):
        raise ValueError("Both lower_bounds and upper_bounds must be supplied.")
    if lower_bounds is not None and upper_bounds is not None:
        start = np.clip(y, lower_bounds, upper_bounds)

        def objective(value: np.ndarray) -> tuple[float, np.ndarray]:
            gradient = gram @ value - rhs
            function = 0.5 * float(value @ (gram @ value)) - float(rhs @ value)
            return function, gradient

        result = minimize(
            objective,
            start,
            jac=True,
            method="L-BFGS-B",
            bounds=list(zip(lower_bounds, upper_bounds)),
            options={
                "maxiter": 5000,
                "maxls": 100,
                "ftol": 1e-15,
                "gtol": 1e-12,
            },
        )
        if not result.success and result.status not in (1, 2):
            raise RuntimeError(
                f"Bounded WLS solve failed: {result.status} {result.message}"
            )
        y = np.asarray(result.x, dtype=np.float64)
    return np.asarray(y, dtype=np.float64), time.perf_counter() - start_time


def scan_max_and_norm(
    split: SplitData,
    y: np.ndarray,
    sample_chunk: int,
    top_k: int = 0,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """One full scan returning max residual, max row norm, and top rows."""

    maximum = -math.inf
    max_norm = 0.0
    kept_indices = np.empty(0, dtype=np.int64)
    kept_values = np.empty(0, dtype=np.float64)
    keep = max(0, int(top_k))
    for row_start, _, phi, target in split.iter_centered_chunks(sample_chunk):
        absolute = np.abs(phi @ y - target)
        if len(absolute):
            maximum = max(maximum, float(np.max(absolute)))
            max_norm = max(
                max_norm,
                float(np.sqrt(np.max(np.einsum("ij,ij->i", phi, phi)))),
            )
        if keep:
            local_keep = min(keep, len(absolute))
            local = np.argpartition(absolute, len(absolute) - local_keep)[
                -local_keep:
            ]
            candidate_indices = np.concatenate(
                (kept_indices, row_start + local.astype(np.int64))
            )
            candidate_values = np.concatenate((kept_values, absolute[local]))
            if len(candidate_values) > keep:
                selected = np.argpartition(
                    candidate_values, len(candidate_values) - keep
                )[-keep:]
                kept_indices = candidate_indices[selected]
                kept_values = candidate_values[selected]
            else:
                kept_indices = candidate_indices
                kept_values = candidate_values
    if not math.isfinite(maximum):
        raise RuntimeError(f"{split.name} split produced no component rows.")
    if keep and len(kept_values):
        order = np.argsort(kept_values)[::-1]
        kept_indices = kept_indices[order]
        kept_values = kept_values[order]
    return maximum, max_norm, kept_indices, kept_values


def run_smax(
    split: SplitData,
    initial: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    value_budget: int,
    subset_size: int,
    eta0: float,
    history_points: int,
    seed: int,
    method: str = SMAX_METHOD,
) -> Trajectory:
    if subset_size > split.n_components:
        raise ValueError(
            f"subset size {subset_size} exceeds d={split.n_components}."
        )
    iterations = value_budget // subset_size
    if iterations < 1:
        raise ValueError(
            f"value budget {value_budget} is smaller than subset size {subset_size}."
        )
    checkpoints = checkpoint_iterations(iterations, history_points)
    checkpoint_position = 0
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
            y -= (
                eta0
                / math.sqrt(iteration)
                * math.copysign(1.0, active_value)
                * phi[active]
            )
            np.clip(y, lower_bounds, upper_bounds, out=y)
        average += (y - average) / iteration
        if (
            checkpoint_position < len(checkpoints)
            and iteration == int(checkpoints[checkpoint_position])
        ):
            coefficients.append(average.copy())
            queries.append(iteration * subset_size)
            times.append(time.perf_counter() - start_time)
            subgradients.append(iteration)
            checkpoint_position += 1
    return Trajectory(
        method=method,
        seed=seed,
        subset_size=subset_size if method == SMAX_METHOD else None,
        coefficients=np.vstack(coefficients),
        optimizer_value_queries=np.asarray(queries, dtype=np.int64),
        optimizer_runtime_seconds=np.asarray(times, dtype=np.float64),
        subgradient_queries=np.asarray(subgradients, dtype=np.int64),
        component_gradient_rows=np.asarray(subgradients, dtype=np.int64),
        raw_state={"checkpoint_iterations": checkpoints},
    )


def run_full_grid_sgm(
    split: SplitData,
    initial: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    value_budget: int,
    eta0: float,
    history_points: int,
    sample_chunk: int,
) -> Trajectory:
    iterations = value_budget // split.n_components
    if iterations < 1:
        raise ValueError(
            f"value budget {value_budget} is smaller than d={split.n_components}."
        )
    checkpoints = checkpoint_iterations(iterations, history_points)
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
            y -= (
                eta0
                / math.sqrt(iteration)
                * math.copysign(1.0, best_residual)
                * best_phi
            )
            np.clip(y, lower_bounds, upper_bounds, out=y)
        average += (y - average) / iteration
        if iteration in checkpoint_set:
            coefficients.append(average.copy())
            queries.append(iteration * split.n_components)
            times.append(time.perf_counter() - start_time)
    return Trajectory(
        method=FULL_SGM_METHOD,
        coefficients=np.vstack(coefficients),
        optimizer_value_queries=np.asarray(queries, dtype=np.int64),
        optimizer_runtime_seconds=np.asarray(times, dtype=np.float64),
        subgradient_queries=checkpoints.copy(),
        component_gradient_rows=checkpoints.copy(),
        raw_state={"checkpoint_iterations": checkpoints},
    )


def run_lse_lbfgsb(
    split: SplitData,
    initial: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    reference_scale: float,
    multipliers: Sequence[float],
    maxiter_per_stage: int,
    sample_chunk: int,
    value_budget: int,
) -> Trajectory:
    """Continuation on a streamed log-sum-exp smoothing of max |residual|."""

    y = np.asarray(initial, dtype=np.float64).copy()
    coefficients: list[np.ndarray] = []
    queries: list[int] = []
    times: list[float] = []
    gradient_rows: list[int] = []
    stage_values: list[int] = []
    mu_values: list[float] = []
    value_evaluations = 0
    start_time = time.perf_counter()
    statuses: list[str] = []
    maximum_evaluations = value_budget // split.n_components
    if maximum_evaluations < 1:
        raise ValueError(
            "LSE value budget must permit at least one full-grid evaluation."
        )
    budget_stopped = False

    class EvaluationBudgetExhausted(RuntimeError):
        pass

    for stage, multiplier in enumerate(multipliers):
        mu = max(
            np.finfo(np.float64).tiny,
            float(reference_scale) * float(multiplier),
        )

        def objective(value: np.ndarray) -> tuple[float, np.ndarray]:
            nonlocal value_evaluations
            if value_evaluations >= maximum_evaluations:
                raise EvaluationBudgetExhausted
            shift = -math.inf
            normalizer = 0.0
            numerator = np.zeros(split.dimension, dtype=np.float64)
            for _, _, phi, target in split.iter_centered_chunks(sample_chunk):
                scaled = (phi @ value - target) / mu
                chunk_shift = float(np.max(np.abs(scaled)))
                if chunk_shift > shift:
                    rescale = 0.0 if not math.isfinite(shift) else math.exp(
                        shift - chunk_shift
                    )
                    normalizer *= rescale
                    numerator *= rescale
                    shift = chunk_shift
                positive = np.exp(scaled - shift)
                negative = np.exp(-scaled - shift)
                normalizer += float(np.sum(positive) + np.sum(negative))
                numerator += phi.T @ (positive - negative)
            value_evaluations += 1
            function = mu * (
                shift
                + math.log(normalizer)
                - math.log(2.0 * split.n_components)
            )
            gradient = numerator / normalizer
            return float(function), gradient

        def callback(value: np.ndarray) -> None:
            nonlocal y
            y = np.asarray(value, dtype=np.float64).copy()
            coefficients.append(np.asarray(value, dtype=np.float64).copy())
            queries.append(value_evaluations * split.n_components)
            times.append(time.perf_counter() - start_time)
            gradient_rows.append(value_evaluations * split.n_components)
            stage_values.append(stage)
            mu_values.append(mu)

        try:
            result = minimize(
                objective,
                y,
                jac=True,
                method="L-BFGS-B",
                bounds=list(zip(lower_bounds, upper_bounds)),
                callback=callback,
                options={
                    "maxiter": maxiter_per_stage,
                    "maxls": 30,
                    "ftol": 1e-12,
                    "gtol": 1e-8,
                },
            )
        except EvaluationBudgetExhausted:
            statuses.append(
                f"budget stop after {value_evaluations} full evaluations"
            )
            budget_stopped = True
            charged_queries = value_evaluations * split.n_components
            if not queries or queries[-1] != charged_queries:
                coefficients.append(y.copy())
                queries.append(charged_queries)
                times.append(time.perf_counter() - start_time)
                gradient_rows.append(charged_queries)
                stage_values.append(stage)
                mu_values.append(mu)
            break
        y = np.asarray(result.x, dtype=np.float64)
        statuses.append(f"stage {stage}: {result.status} {result.message}")
        if not coefficients or not np.array_equal(coefficients[-1], y):
            coefficients.append(y.copy())
            queries.append(value_evaluations * split.n_components)
            times.append(time.perf_counter() - start_time)
            gradient_rows.append(value_evaluations * split.n_components)
            stage_values.append(stage)
            mu_values.append(mu)
        if value_evaluations >= maximum_evaluations:
            statuses.append(
                f"budget stop after {value_evaluations} full evaluations"
            )
            budget_stopped = True
            break

    if not coefficients:
        coefficients.append(np.asarray(initial, dtype=np.float64).copy())
        queries.append(value_evaluations * split.n_components)
        times.append(time.perf_counter() - start_time)
        gradient_rows.append(value_evaluations * split.n_components)
        stage_values.append(0)
        mu_values.append(
            max(
                np.finfo(np.float64).tiny,
                float(reference_scale) * float(multipliers[0]),
            )
        )

    return Trajectory(
        method=LSE_METHOD,
        coefficients=np.vstack(coefficients),
        optimizer_value_queries=np.asarray(queries, dtype=np.int64),
        optimizer_runtime_seconds=np.asarray(times, dtype=np.float64),
        subgradient_queries=np.zeros(len(coefficients), dtype=np.int64),
        component_gradient_rows=np.asarray(gradient_rows, dtype=np.int64),
        status="; ".join(statuses),
        raw_state={
            "stage": np.asarray(stage_values, dtype=np.int64),
            "mu": np.asarray(mu_values, dtype=np.float64),
            "full_value_and_gradient_evaluations": np.asarray(
                [value_evaluations], dtype=np.int64
            ),
            "value_budget": np.asarray([value_budget], dtype=np.int64),
            "budget_stopped": np.asarray(
                [int(budget_stopped)], dtype=np.int8
            ),
        },
    )


def separate_top_rows(
    split: SplitData,
    y: np.ndarray,
    sample_chunk: int,
    top_k: int,
    excluded_mask: np.ndarray | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Stream exact separation and retain the largest non-active residuals."""

    maximum = -math.inf
    kept_indices = np.empty(0, dtype=np.int64)
    kept_values = np.empty(0, dtype=np.float64)
    for row_start, _, phi, target in split.iter_centered_chunks(sample_chunk):
        absolute = np.abs(phi @ y - target)
        if len(absolute):
            maximum = max(maximum, float(np.max(absolute)))
        if top_k <= 0:
            continue
        local_indices = np.arange(
            row_start, row_start + len(absolute), dtype=np.int64
        )
        if excluded_mask is not None:
            permitted = ~excluded_mask[row_start : row_start + len(absolute)]
            local_indices = local_indices[permitted]
            absolute = absolute[permitted]
        if not len(absolute):
            continue
        local_keep = min(top_k, len(absolute))
        selected = np.argpartition(absolute, len(absolute) - local_keep)[
            -local_keep:
        ]
        candidate_indices = np.concatenate(
            (kept_indices, local_indices[selected])
        )
        candidate_values = np.concatenate((kept_values, absolute[selected]))
        if len(candidate_values) > top_k:
            selected = np.argpartition(
                candidate_values, len(candidate_values) - top_k
            )[-top_k:]
            kept_indices = candidate_indices[selected]
            kept_values = candidate_values[selected]
        else:
            kept_indices = candidate_indices
            kept_values = candidate_values
    if not math.isfinite(maximum):
        raise RuntimeError("Restricted LP separation found no component.")
    if len(kept_values):
        order = np.argsort(kept_values)[::-1]
        kept_indices = kept_indices[order]
        kept_values = kept_values[order]
    return maximum, kept_indices, kept_values


def run_restricted_lp_exchange(
    split: SplitData,
    initial: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    value_budget: int,
    initial_rows: int,
    add_rows: int,
    max_iterations: int,
    absolute_tolerance: float,
    relative_tolerance: float,
    sample_chunk: int,
) -> Trajectory:
    """Epigraph LP with exact streamed separation over the full finite max."""

    maximum_scans = max(1, value_budget // split.n_components)
    maximum_scans = min(maximum_scans, max_iterations + 1)
    start_time = time.perf_counter()
    initial_upper, active_indices, _ = separate_top_rows(
        split,
        initial,
        sample_chunk,
        min(initial_rows, split.n_components),
    )
    active_mask = np.zeros(split.n_components, dtype=bool)
    active_mask[active_indices] = True
    best_y = np.asarray(initial, dtype=np.float64).copy()
    best_upper = initial_upper
    coefficients = [best_y.copy()]
    queries = [split.n_components]
    times = [time.perf_counter() - start_time]
    lower_history = [math.nan]
    active_sizes = [int(np.sum(active_mask))]
    incumbent_upper = [best_upper]
    master_seconds = [0.0]
    statuses: list[str] = []

    for _ in range(1, maximum_scans):
        ordered_active = np.flatnonzero(active_mask)
        phi, target = split.gather_centered_rows(ordered_active)
        constraints = np.vstack(
            (
                np.column_stack((phi, -np.ones(len(phi)))),
                np.column_stack((-phi, -np.ones(len(phi)))),
            )
        )
        rhs = np.concatenate((target, -target))
        objective = np.zeros(split.dimension + 1, dtype=np.float64)
        objective[-1] = 1.0
        master_start = time.perf_counter()
        result = linprog(
            objective,
            A_ub=constraints,
            b_ub=rhs,
            bounds=list(zip(lower_bounds, upper_bounds)) + [(0.0, None)],
            method="highs",
        )
        current_master_seconds = time.perf_counter() - master_start
        if not result.success:
            statuses.append(f"{result.status} {result.message}")
            break
        candidate = np.asarray(result.x[:-1], dtype=np.float64)
        master_lower = float(result.x[-1])
        upper, new_indices, new_values = separate_top_rows(
            split,
            candidate,
            sample_chunk,
            min(
                add_rows,
                max(0, split.n_components - int(np.sum(active_mask))),
            ),
            active_mask,
        )
        if upper < best_upper:
            best_upper = upper
            best_y = candidate.copy()
        coefficients.append(best_y.copy())
        queries.append(len(coefficients) * split.n_components)
        times.append(time.perf_counter() - start_time)
        lower_history.append(master_lower)
        incumbent_upper.append(best_upper)
        master_seconds.append(current_master_seconds)
        gap = best_upper - master_lower
        tolerance = absolute_tolerance + relative_tolerance * max(
            1.0, abs(best_upper)
        )
        if gap <= tolerance:
            statuses.append("certified gap tolerance")
            active_sizes.append(int(np.sum(active_mask)))
            break
        violating = [
            int(index)
            for index, value in zip(new_indices, new_values)
            if float(value) > master_lower + absolute_tolerance
        ]
        if not violating:
            statuses.append("no new violating row")
            active_sizes.append(int(np.sum(active_mask)))
            break
        active_mask[np.asarray(violating, dtype=np.int64)] = True
        active_sizes.append(int(np.sum(active_mask)))
    else:
        statuses.append("scan budget reached")

    return Trajectory(
        method=LP_METHOD,
        coefficients=np.vstack(coefficients),
        optimizer_value_queries=np.asarray(queries, dtype=np.int64),
        optimizer_runtime_seconds=np.asarray(times, dtype=np.float64),
        subgradient_queries=np.zeros(len(coefficients), dtype=np.int64),
        component_gradient_rows=np.zeros(len(coefficients), dtype=np.int64),
        lower_bounds=np.asarray(lower_history, dtype=np.float64),
        status="; ".join(statuses),
        raw_state={
            "final_active_indices": np.flatnonzero(active_mask),
            "active_set_sizes": np.asarray(active_sizes, dtype=np.int64),
            "incumbent_upper_bounds": np.asarray(
                incumbent_upper, dtype=np.float64
            ),
            "master_solve_seconds": np.asarray(
                master_seconds, dtype=np.float64
            ),
        },
    )


def rescale_trajectory_timing(
    trajectory: Trajectory, final_seconds: Sequence[float]
) -> None:
    """Scale checkpoint timings to a median repeated final runtime."""

    repeats = [float(value) for value in final_seconds]
    trajectory.timing_repeats_seconds = repeats
    if not repeats or not len(trajectory.optimizer_runtime_seconds):
        return
    primary_final = float(trajectory.optimizer_runtime_seconds[-1])
    median_final = float(np.median(repeats))
    if primary_final > 0.0:
        trajectory.optimizer_runtime_seconds *= median_final / primary_final


def warm_up_kernels(
    split: SplitData,
    initial: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    sample_chunk: int,
) -> None:
    """Touch mmap pages and initialize BLAS/HiGHS outside timed regions."""

    iterator = split.iter_centered_chunks(min(sample_chunk, split.n_samples))
    _, _, phi, target = next(iterator)
    small = min(256, len(phi))
    _ = phi[:small] @ initial
    _ = phi[:small].T @ target[:small]
    if split._phi_mean is not None:
        indices = np.arange(min(32, split.n_components), dtype=np.int64)
        split.gather_centered_rows(indices)
    objective = np.zeros(split.dimension + 1, dtype=np.float64)
    objective[-1] = 1.0
    tiny_phi = phi[: min(2 * split.dimension, len(phi))]
    tiny_target = target[: len(tiny_phi)]
    constraints = np.vstack(
        (
            np.column_stack((tiny_phi, -np.ones(len(tiny_phi)))),
            np.column_stack((-tiny_phi, -np.ones(len(tiny_phi)))),
        )
    )
    linprog(
        objective,
        A_ub=constraints,
        b_ub=np.concatenate((tiny_target, -tiny_target)),
        bounds=list(zip(lower_bounds, upper_bounds)) + [(0.0, None)],
        method="highs",
    )


def run_stepsize_pilot(
    split: SplitData,
    initial: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    args: argparse.Namespace,
) -> tuple[float, dict[str, Any], dict[str, np.ndarray]]:
    """Choose eta0 using the training split only; pilot work is uncharged."""

    wls_max, max_row_norm, _, _ = scan_max_and_norm(
        split, initial, args.sample_chunk
    )
    if args.eta0 > 0.0:
        return (
            float(args.eta0),
            {
                "selection": "user supplied",
                "selected_eta0": float(args.eta0),
                "bounded_wls_train_max": wls_max,
                "max_design_row_norm": max_row_norm,
            },
            {},
        )
    base = wls_max / max(max_row_norm * max_row_norm, np.finfo(float).tiny)
    subset_size = min(args.pilot_subset_size, split.n_components)
    budget = max(subset_size, args.pilot_value_budget)
    candidates: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    for candidate_index, theta in enumerate(args.eta_theta_candidates):
        eta = float(theta) * base
        trajectory = run_smax(
            split,
            initial,
            lower_bounds,
            upper_bounds,
            budget,
            subset_size,
            eta,
            min(args.history_points, 12),
            args.pilot_seed,
            method="train-only stepsize pilot",
        )
        score, _, _, _ = scan_max_and_norm(
            split, trajectory.coefficients[-1], args.sample_chunk
        )
        candidates.append(
            {
                "theta": float(theta),
                "eta0": eta,
                "final_train_max": score,
                "sampled_value_queries": int(
                    trajectory.optimizer_value_queries[-1]
                ),
                "validation_value_queries_unaccounted": split.n_components,
            }
        )
        arrays[f"pilot_{candidate_index:02d}_coefficients"] = (
            trajectory.coefficients
        )
        arrays[f"pilot_{candidate_index:02d}_optimizer_queries"] = (
            trajectory.optimizer_value_queries
        )
    selected_index = min(
        range(len(candidates)),
        key=lambda index: (
            candidates[index]["final_train_max"],
            candidates[index]["eta0"],
        ),
    )
    selected = float(candidates[selected_index]["eta0"])
    state = {
        "selection": "minimum exact training max after fixed-budget pilot",
        "selected_index": selected_index,
        "selected_eta0": selected,
        "base_eta0_h_over_G_squared": base,
        "bounded_wls_train_max": wls_max,
        "max_design_row_norm": max_row_norm,
        "pilot_seed": int(args.pilot_seed),
        "pilot_subset_size": int(subset_size),
        "pilot_value_budget": int(budget),
        "candidates": candidates,
        "accounting": (
            "All pilot optimization and exact training validation scans are "
            "configuration work and excluded from production query/time axes."
        ),
    }
    return selected, state, arrays


def run_full_sgm_stepsize_pilot(
    split: SplitData,
    initial: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    args: argparse.Namespace,
) -> tuple[float, dict[str, Any], dict[str, np.ndarray]]:
    """Tune full-grid SGM separately, using only exact training scans."""

    wls_max, max_row_norm, _, _ = scan_max_and_norm(
        split, initial, args.sample_chunk
    )
    if args.full_sgm_eta0 > 0.0:
        return (
            float(args.full_sgm_eta0),
            {
                "selection": "user supplied",
                "selected_eta0": float(args.full_sgm_eta0),
                "bounded_wls_train_max": wls_max,
                "max_design_row_norm": max_row_norm,
            },
            {},
        )
    base = wls_max / max(max_row_norm * max_row_norm, np.finfo(float).tiny)
    budget = max(
        split.n_components, int(args.full_sgm_pilot_value_budget)
    )
    candidates: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    for candidate_index, theta in enumerate(args.eta_theta_candidates):
        eta = float(theta) * base
        trajectory = run_full_grid_sgm(
            split,
            initial,
            lower_bounds,
            upper_bounds,
            budget,
            eta,
            min(args.history_points, 12),
            args.sample_chunk,
        )
        score, _, _, _ = scan_max_and_norm(
            split, trajectory.coefficients[-1], args.sample_chunk
        )
        candidates.append(
            {
                "theta": float(theta),
                "eta0": eta,
                "final_train_max": score,
                "full_grid_iterations": int(
                    trajectory.subgradient_queries[-1]
                ),
                "optimizer_value_queries": int(
                    trajectory.optimizer_value_queries[-1]
                ),
                "validation_value_queries_unaccounted": split.n_components,
            }
        )
        arrays[
            f"full_sgm_pilot_{candidate_index:02d}_coefficients"
        ] = trajectory.coefficients
        arrays[
            f"full_sgm_pilot_{candidate_index:02d}_optimizer_queries"
        ] = trajectory.optimizer_value_queries
    selected_index = min(
        range(len(candidates)),
        key=lambda index: (
            candidates[index]["final_train_max"],
            candidates[index]["eta0"],
        ),
    )
    selected = float(candidates[selected_index]["eta0"])
    return (
        selected,
        {
            "selection": (
                "minimum exact training max after a separate fixed-budget "
                "full-grid pilot"
            ),
            "selected_index": selected_index,
            "selected_eta0": selected,
            "base_eta0_h_over_G_squared": base,
            "bounded_wls_train_max": wls_max,
            "max_design_row_norm": max_row_norm,
            "pilot_value_budget": budget,
            "candidates": candidates,
            "accounting": (
                "This full-grid pilot and its exact training validation scans "
                "are excluded from production query/time axes."
            ),
        },
        arrays,
    )


def evaluate_coefficients(
    train: SplitData,
    test: SplitData,
    coefficients: np.ndarray,
    sample_chunk: int,
    coefficient_block: int,
) -> Evaluation:
    """Evaluate every stored iterate in batched, uncharged offline passes."""

    values = np.asarray(coefficients, dtype=np.float64)
    count = values.shape[0]
    train_max = np.full(count, -math.inf, dtype=np.float64)
    test_max = np.full(count, -math.inf, dtype=np.float64)
    agreement_count = np.zeros(count, dtype=np.int64)
    certified_count = np.zeros(count, dtype=np.int64)
    accuracy_count = np.zeros(count, dtype=np.int64)
    samplewise_test_error = np.empty(
        (test.n_samples, count), dtype=np.float64
    )

    for _, _, phi, target in train.iter_centered_chunks(sample_chunk):
        for start in range(0, count, coefficient_block):
            stop = min(start + coefficient_block, count)
            residual = phi @ values[start:stop].T - target[:, None]
            train_max[start:stop] = np.maximum(
                train_max[start:stop], np.max(np.abs(residual), axis=0)
            )

    centered_bias = test.centered_bias
    for row_start, row_stop, phi, target in test.iter_centered_chunks(
        sample_chunk
    ):
        sample_start = row_start // test.n_classes
        sample_stop = row_stop // test.n_classes
        originals = np.asarray(
            test.original_predictions[sample_start:sample_stop],
            dtype=np.int64,
        )
        labels = np.asarray(
            test.labels[sample_start:sample_stop], dtype=np.int64
        )
        original_logits = np.asarray(
            test.original_logits[sample_start:sample_stop], dtype=np.float64
        )
        largest_two = np.partition(original_logits, -2, axis=1)[:, -2:]
        original_margin = largest_two[:, 1] - largest_two[:, 0]
        sample_count = sample_stop - sample_start
        for start in range(0, count, coefficient_block):
            stop = min(start + coefficient_block, count)
            block_count = stop - start
            contribution = phi @ values[start:stop].T
            residual = contribution - target[:, None]
            test_max[start:stop] = np.maximum(
                test_max[start:stop], np.max(np.abs(residual), axis=0)
            )
            component_delta = residual.reshape(
                sample_count, test.n_classes, block_count
            )
            per_sample_error = np.max(np.abs(component_delta), axis=1)
            samplewise_test_error[
                sample_start:sample_stop, start:stop
            ] = per_sample_error
            logits = original_logits[:, :, None] + component_delta
            predictions = np.argmax(logits, axis=1)
            agreement_count[start:stop] += np.sum(
                predictions == originals[:, None], axis=0
            )
            accuracy_count[start:stop] += np.sum(
                predictions == labels[:, None], axis=0
            )
            certified_count[start:stop] += np.sum(
                original_margin[:, None] > 2.0 * per_sample_error,
                axis=0,
            )
    return Evaluation(
        train_max=train_max,
        test_max=test_max,
        test_samplewise_p999=np.quantile(
            samplewise_test_error, 0.999, axis=0
        ),
        test_agreement=agreement_count / test.n_samples,
        test_certified_agreement=certified_count / test.n_samples,
        test_accuracy=accuracy_count / test.n_samples,
    )


def scalar_polynomial_diagnostics(
    coefficients: np.ndarray,
    y_fun: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    scale_B: float,
    grid_points: int = 16_385,
) -> dict[str, np.ndarray]:
    """Diagnostics for privacy-inference usefulness of every polynomial."""

    values = np.asarray(coefficients, dtype=np.float64)
    degree = values.shape[1] - 1
    grid = np.linspace(-1.0, 1.0, grid_points, dtype=np.float64)
    vander = np.polynomial.chebyshev.chebvander(grid, degree)
    target = np.maximum(grid, 0.0)
    half_width = (upper_bounds - lower_bounds) / 2.0
    sup_error = np.empty(len(values), dtype=np.float64)
    maximum_derivative = np.empty(len(values), dtype=np.float64)
    minimum_derivative = np.empty(len(values), dtype=np.float64)
    derivative_total_variation = np.empty(len(values), dtype=np.float64)
    error_extrema = np.empty(len(values), dtype=np.int64)
    error_sign_alternations = np.empty(len(values), dtype=np.int64)
    max_bound_activity = np.empty(len(values), dtype=np.float64)
    active_bound_count = np.empty(len(values), dtype=np.int64)

    for index, y in enumerate(values):
        error = vander @ y - target
        absolute = np.abs(error)
        sup_error[index] = float(np.max(absolute))
        derivative_coefficients = np.polynomial.chebyshev.chebder(y)
        derivative = np.polynomial.chebyshev.chebval(
            grid, derivative_coefficients
        )
        maximum_derivative[index] = float(np.max(derivative))
        minimum_derivative[index] = float(np.min(derivative))
        derivative_total_variation[index] = float(
            np.sum(np.abs(np.diff(derivative)))
        )
        differences = np.diff(error)
        extrema_indices = np.flatnonzero(
            differences[:-1] * differences[1:] <= 0.0
        ) + 1
        if len(extrema_indices):
            significant = extrema_indices[
                absolute[extrema_indices]
                >= 0.25 * max(sup_error[index], np.finfo(float).tiny)
            ]
            signs = np.sign(error[significant])
            signs = signs[signs != 0.0]
        else:
            significant = np.empty(0, dtype=np.int64)
            signs = np.empty(0, dtype=np.float64)
        error_extrema[index] = len(significant)
        error_sign_alternations[index] = (
            int(np.sum(signs[1:] != signs[:-1])) if len(signs) > 1 else 0
        )
        activity = np.abs(y - y_fun) / half_width
        max_bound_activity[index] = float(np.max(activity))
        active_bound_count[index] = int(np.sum(activity >= 1.0 - 1e-7))

    return {
        "scalar_dense_grid_sup_error": sup_error,
        "scalar_physical_sup_error": scale_B * sup_error,
        "scalar_maximum_derivative": maximum_derivative,
        "scalar_minimum_derivative": minimum_derivative,
        "scalar_monotonicity_violation": np.maximum(
            0.0, -minimum_derivative
        ),
        "scalar_derivative_total_variation": derivative_total_variation,
        "scalar_significant_error_extrema": error_extrema,
        "scalar_error_sign_alternations": error_sign_alternations,
        "max_bound_activity": max_bound_activity,
        "active_bound_count": active_bound_count,
    }


def residual_tail_statistics(
    split: SplitData,
    final_coefficients: dict[str, np.ndarray],
    sample_size: int,
    seed: int,
) -> dict[str, Any]:
    """Persist deterministic residual-tail summaries for final iterates."""

    rng = np.random.default_rng(seed)
    count = min(sample_size, split.n_components)
    indices = np.sort(
        rng.choice(split.n_components, size=count, replace=False)
    )
    phi, target = split.gather_centered_rows(indices)
    probabilities = np.asarray([0.5, 0.9, 0.95, 0.99, 0.999, 1.0])
    methods: dict[str, Any] = {}
    for name, y in final_coefficients.items():
        absolute = np.abs(phi @ y - target)
        methods[name] = {
            "quantile_probabilities": probabilities.tolist(),
            "absolute_residual_quantiles": np.quantile(
                absolute, probabilities
            ).tolist(),
            "mean_absolute_residual": float(np.mean(absolute)),
            "root_mean_square_residual": float(
                np.sqrt(np.mean(np.square(absolute)))
            ),
        }
    return {
        "split": split.name,
        "seed": int(seed),
        "sample_size": int(count),
        "sampled_flat_component_indices": indices.tolist(),
        "methods": methods,
    }


def exact_near_max_cardinality(
    split: SplitData,
    selected_coefficients: dict[str, np.ndarray],
    sample_chunk: int,
) -> dict[str, Any]:
    """Exact K counts within 1% and 0.1% of each full-grid maximum."""

    names = list(selected_coefficients)
    values = np.vstack([selected_coefficients[name] for name in names])
    maxima = np.full(len(names), -math.inf, dtype=np.float64)
    for _, _, phi, target in split.iter_centered_chunks(sample_chunk):
        absolute = np.abs(phi @ values.T - target[:, None])
        maxima = np.maximum(maxima, np.max(absolute, axis=0))
    counts_1pct = np.zeros(len(names), dtype=np.int64)
    counts_01pct = np.zeros(len(names), dtype=np.int64)
    for _, _, phi, target in split.iter_centered_chunks(sample_chunk):
        absolute = np.abs(phi @ values.T - target[:, None])
        counts_1pct += np.sum(
            absolute >= 0.99 * maxima[None, :], axis=0
        )
        counts_01pct += np.sum(
            absolute >= 0.999 * maxima[None, :], axis=0
        )
    return {
        name: {
            "exact_max": float(maxima[index]),
            "K_within_1_percent": int(counts_1pct[index]),
            "fraction_within_1_percent": float(
                counts_1pct[index] / split.n_components
            ),
            "K_within_0_1_percent": int(counts_01pct[index]),
            "fraction_within_0_1_percent": float(
                counts_01pct[index] / split.n_components
            ),
        }
        for index, name in enumerate(names)
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty table {path}.")
    fieldnames: list[str] = []
    for row in rows:
        for name in row:
            if name not in fieldnames:
                fieldnames.append(name)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_results(
    args: argparse.Namespace,
    problem: CachedProblem,
    trajectories: list[Trajectory],
    all_coefficients: np.ndarray,
    slices: list[slice],
    evaluation: Evaluation,
    scalar_diagnostics: dict[str, np.ndarray],
    y_fun: np.ndarray,
    e_fun: float,
    functional_diagnostics: dict[str, Any],
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    wls_seconds: Sequence[float],
    static_runtime: dict[str, float],
    eta0: float,
    pilot_state: dict[str, Any],
    pilot_arrays: dict[str, np.ndarray],
    full_sgm_eta0: float,
    full_sgm_pilot_state: dict[str, Any],
    setup_state: dict[str, Any],
    tail_state: dict[str, Any],
    exact_tail_state: dict[str, Any],
) -> None:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    wls_runtime = float(np.median(wls_seconds))
    history_rows: list[dict[str, Any]] = []
    coefficient_payload: dict[str, np.ndarray] = {
        "y_fun": y_fun,
        "e_fun": np.asarray([e_fun], dtype=np.float64),
        "kappa": np.asarray(
            [args.functional_box_kappa], dtype=np.float64
        ),
        "functional_box_lower": lower_bounds,
        "functional_box_upper": upper_bounds,
    }
    method_state_payload: dict[str, np.ndarray] = dict(pilot_arrays)

    for trajectory_index, (trajectory, coefficient_slice) in enumerate(
        zip(trajectories, slices)
    ):
        key = f"trajectory_{trajectory_index:03d}_coefficients"
        coefficient_payload[key] = trajectory.coefficients
        for state_name, state_value in trajectory.raw_state.items():
            method_state_payload[
                f"trajectory_{trajectory_index:03d}_{state_name}"
            ] = np.asarray(state_value)
        for checkpoint in range(len(trajectory.coefficients)):
            metric_index = coefficient_slice.start + checkpoint
            if trajectory.method == WLS_METHOD:
                initialization_queries = problem.train.n_components
                initialization_runtime = wls_runtime
                total_queries = initialization_queries
                total_runtime = initialization_runtime
            elif trajectory.method in (CHEB_METHOD, FUNCTIONAL_METHOD):
                initialization_queries = 0
                initialization_runtime = 0.0
                total_queries = 0
                total_runtime = static_runtime[trajectory.method]
            else:
                initialization_queries = problem.train.n_components
                initialization_runtime = wls_runtime
                total_queries = initialization_queries + int(
                    trajectory.optimizer_value_queries[checkpoint]
                )
                total_runtime = initialization_runtime + float(
                    trajectory.optimizer_runtime_seconds[checkpoint]
                )
            lower = float(trajectory.lower_bounds[checkpoint])
            row: dict[str, Any] = {
                "trajectory_id": trajectory_index,
                "method": trajectory.method,
                "seed": "" if trajectory.seed is None else trajectory.seed,
                "subset_size": (
                    "" if trajectory.subset_size is None else trajectory.subset_size
                ),
                "checkpoint": checkpoint,
                "coefficient_key": f"{key}[{checkpoint}]",
                "status": trajectory.status,
                "value_queries": total_queries,
                "runtime_seconds": total_runtime,
                "initialization_value_queries": initialization_queries,
                "initialization_runtime_seconds": initialization_runtime,
                "optimizer_value_queries": int(
                    trajectory.optimizer_value_queries[checkpoint]
                ),
                "optimizer_runtime_seconds": float(
                    trajectory.optimizer_runtime_seconds[checkpoint]
                ),
                "subgradient_queries": int(
                    trajectory.subgradient_queries[checkpoint]
                ),
                "component_gradient_rows": int(
                    trajectory.component_gradient_rows[checkpoint]
                ),
                "train_max_centered_logit_distortion": float(
                    evaluation.train_max[metric_index]
                ),
                "test_max_centered_logit_distortion": float(
                    evaluation.test_max[metric_index]
                ),
                "test_samplewise_max_distortion_p999": float(
                    evaluation.test_samplewise_p999[metric_index]
                ),
                "test_original_prediction_agreement": float(
                    evaluation.test_agreement[metric_index]
                ),
                "test_certified_agreement_margin_gt_2E": float(
                    evaluation.test_certified_agreement[metric_index]
                ),
                "test_ground_truth_accuracy": float(
                    evaluation.test_accuracy[metric_index]
                ),
                "lp_master_lower_bound": (
                    "" if not math.isfinite(lower) else lower
                ),
                "lp_validated_optimality_gap": (
                    ""
                    if not math.isfinite(lower)
                    else float(evaluation.train_max[metric_index] - lower)
                ),
            }
            for diagnostic_name, diagnostic_values in scalar_diagnostics.items():
                row[diagnostic_name] = _jsonable(
                    diagnostic_values[metric_index]
                )
            for state_name in ("stage", "mu", "active_set_sizes"):
                state_values = trajectory.raw_state.get(state_name)
                if state_values is not None and len(state_values) == len(
                    trajectory.coefficients
                ):
                    row[state_name] = _jsonable(state_values[checkpoint])
            history_rows.append(row)

    np.savez_compressed(output_dir / "coefficients.npz", **coefficient_payload)
    np.savez_compressed(output_dir / "method_state.npz", **method_state_payload)
    _write_csv(output_dir / "history.csv", history_rows)

    final_rows = []
    for trajectory_index in range(len(trajectories)):
        matching = [
            row
            for row in history_rows
            if row["trajectory_id"] == trajectory_index
        ]
        final = dict(matching[-1])
        final["row_type"] = "individual_final"
        final_rows.append(final)
    smax_finals = [
        row for row in final_rows if row["method"] == SMAX_METHOD
    ]
    for subset_size in sorted(
        {int(row["subset_size"]) for row in smax_finals}
    ):
        members = [
            row
            for row in smax_finals
            if int(row["subset_size"]) == subset_size
        ]
        aggregate: dict[str, Any] = {
            "row_type": "aggregate_final",
            "method": SMAX_METHOD,
            "seed": "aggregate",
            "subset_size": subset_size,
            "replicate_count": len(members),
            "status": "median with interquartile range over seeds",
        }
        numeric_names = (
            "value_queries",
            "runtime_seconds",
            "train_max_centered_logit_distortion",
            "test_max_centered_logit_distortion",
            "test_samplewise_max_distortion_p999",
            "test_original_prediction_agreement",
            "test_certified_agreement_margin_gt_2E",
            "test_ground_truth_accuracy",
            "scalar_dense_grid_sup_error",
            "scalar_minimum_derivative",
            "max_bound_activity",
            "active_bound_count",
        )
        for name in numeric_names:
            data = np.asarray([float(row[name]) for row in members])
            aggregate[name] = float(np.median(data))
            aggregate[f"{name}_q25"] = float(np.quantile(data, 0.25))
            aggregate[f"{name}_q75"] = float(np.quantile(data, 0.75))
        final_rows.append(aggregate)
    _write_csv(output_dir / "summary.csv", final_rows)

    original_test_accuracy = float(
        np.mean(
            np.asarray(problem.test.original_predictions)
            == np.asarray(problem.test.labels)
        )
    )
    configuration = {
        "command": [sys.executable, *sys.argv],
        "arguments": vars(args),
        "feature_metadata": problem.metadata,
        "feature_schema_expected": {
            "phi": (
                "[N,C,q+1], already B-scaled and class-centered; phi @ y is "
                "the centered replacement-activation logit contribution"
            ),
            "b": "[N,C], original final-ReLU contribution, class-centered",
            "labels": "[N]",
            "original_predictions": "[N]",
            "original_logits": (
                "[N,C], used for top-one margins and prediction updates"
            ),
            "fc_bias": "[C]",
        },
        "problem": {
            "degree_q": problem.degree,
            "dimension": problem.dimension,
            "scale_B": problem.scale_B,
            "train_samples": problem.train.n_samples,
            "test_samples": problem.test.n_samples,
            "classes": problem.train.n_classes,
            "finite_max_components_d": problem.train.n_components,
            "objective": (
                "max_{image,class} |phi[image,class] @ y - b[image,class]|"
            ),
            "class_centered_logits": True,
            "original_test_accuracy": original_test_accuracy,
        },
        "functional_box": {
            "parameterization": "p_y(z)=B sum_r y_r T_r(z/B)",
            "kappa": args.functional_box_kappa,
            "halfwidth_formula": "kappa * e_fun / (q+1)",
            "halfwidth": float(
                args.functional_box_kappa * e_fun / (problem.degree + 1)
            ),
            "y_fun": y_fun,
            "e_fun_continuous_certificate": e_fun,
            "lower_bounds": lower_bounds,
            "upper_bounds": upper_bounds,
            "guarantee": (
                "|p_y(z)-ReLU(z)| <= (1+kappa) B e_fun on [-B,B]"
            ),
            "construction_diagnostics": functional_diagnostics,
        },
        "smax_stepsize_pilot": pilot_state,
        "selected_smax_eta0": eta0,
        "full_sgm_stepsize_pilot": full_sgm_pilot_state,
        "selected_full_sgm_eta0": full_sgm_eta0,
        "timing": {
            "wls_repeats_seconds": list(wls_seconds),
            "wls_median_seconds": wls_runtime,
            "trajectory_repeats_seconds": {
                str(index): trajectory.timing_repeats_seconds
                for index, trajectory in enumerate(trajectories)
                if trajectory.timing_repeats_seconds
            },
            "warmups_excluded": True,
            "setup": setup_state,
        },
        "query_accounting": {
            "charged": (
                "WLS initialization reads d component rows; SMax charges the "
                "sampled subset per iteration; a full separation, full-grid "
                "subgradient, or LSE joint value/gradient evaluation charges d."
            ),
            "uncharged": (
                "Feature extraction, mmap mean preparation, warmups, scalar "
                "functional-box construction, train-only stepsize pilot, all "
                "offline train/test metrics, and diagnostic separation passes."
            ),
            "runtime_axis": (
                "The canonical plot uses optimizer-only time, excluding the "
                "common bounded-WLS initialization; SMax seeds serve as its "
                "timing repetitions. Total time is also retained in history."
            ),
            "canonical_plot_axes": (
                "optimizer_value_queries and optimizer_runtime_seconds; the "
                "common bounded-WLS start is shown as a left-edge reference"
            ),
        },
        "tail_diagnostics_sampled_quantiles": tail_state,
        "tail_diagnostics_exact_train_near_max_cardinality": exact_tail_state,
        "files": {
            "history": "history.csv",
            "summary": "summary.csv",
            "coefficients": "coefficients.npz",
            "method_state": "method_state.npz",
            "plot": "polynomial_activation_cost_time_comparison.png",
        },
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(_jsonable(configuration), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    args = finalize_args(parse_args())
    print(f"Loading cached features from {args.feature_dir}", flush=True)
    problem = load_cached_problem(
        args.feature_dir, args.train_samples, args.test_samples
    )
    if problem.degree != 14:
        print(
            f"Warning: primary paper configuration uses q=14; cache has "
            f"q={problem.degree}.",
            flush=True,
        )

    setup_start = time.perf_counter()
    train_mean_seconds = problem.train.prepare_random_access_means(
        args.sample_chunk
    )
    test_mean_seconds = problem.test.prepare_random_access_means(
        args.sample_chunk
    )
    scalar_start = time.perf_counter()
    y_fun, e_fun, functional_diagnostics = scalar_near_minimax_relu(
        problem.degree, args.functional_grid_points
    )
    functional_seconds = time.perf_counter() - scalar_start
    halfwidth = (
        args.functional_box_kappa * e_fun / (problem.degree + 1)
    )
    lower_bounds = y_fun - halfwidth
    upper_bounds = y_fun + halfwidth
    if not np.allclose(
        upper_bounds - y_fun,
        args.functional_box_kappa * e_fun / (problem.degree + 1),
        rtol=0.0,
        atol=10 * np.finfo(float).eps,
    ):
        raise AssertionError("Functional box half-width was formed incorrectly.")
    warm_up_kernels(
        problem.train, y_fun, lower_bounds, upper_bounds, args.sample_chunk
    )

    wls_values: list[np.ndarray] = []
    wls_seconds: list[float] = []
    for _ in range(args.timing_repeats):
        value, seconds = streaming_wls(
            problem.train,
            args.sample_chunk,
            lower_bounds,
            upper_bounds,
        )
        wls_values.append(value)
        wls_seconds.append(seconds)
    wls = wls_values[0]
    if any(
        not np.allclose(wls, value, rtol=1e-8, atol=1e-10)
        for value in wls_values[1:]
    ):
        raise RuntimeError("Repeated bounded WLS solves were not reproducible.")

    cheb_start = time.perf_counter()
    cheb = chebyshev_node_relu_coefficients(problem.degree)
    cheb_seconds = time.perf_counter() - cheb_start
    eta0, pilot_state, pilot_arrays = run_stepsize_pilot(
        problem.train, wls, lower_bounds, upper_bounds, args
    )
    full_sgm_eta0, full_sgm_pilot_state, full_sgm_pilot_arrays = (
        run_full_sgm_stepsize_pilot(
            problem.train, wls, lower_bounds, upper_bounds, args
        )
    )
    pilot_arrays.update(full_sgm_pilot_arrays)
    setup_elapsed_seconds = time.perf_counter() - setup_start

    trajectories: list[Trajectory] = [
        Trajectory(
            method=WLS_METHOD,
            coefficients=wls,
            optimizer_value_queries=np.zeros(1, dtype=np.int64),
            optimizer_runtime_seconds=np.zeros(1, dtype=np.float64),
            subgradient_queries=np.zeros(1, dtype=np.int64),
            component_gradient_rows=np.zeros(1, dtype=np.int64),
            timing_repeats_seconds=list(wls_seconds),
        ),
        Trajectory(
            method=FUNCTIONAL_METHOD,
            coefficients=y_fun,
            optimizer_value_queries=np.zeros(1, dtype=np.int64),
            optimizer_runtime_seconds=np.zeros(1, dtype=np.float64),
            subgradient_queries=np.zeros(1, dtype=np.int64),
            component_gradient_rows=np.zeros(1, dtype=np.int64),
        ),
        Trajectory(
            method=CHEB_METHOD,
            coefficients=cheb,
            optimizer_value_queries=np.zeros(1, dtype=np.int64),
            optimizer_runtime_seconds=np.zeros(1, dtype=np.float64),
            subgradient_queries=np.zeros(1, dtype=np.int64),
            component_gradient_rows=np.zeros(1, dtype=np.int64),
        ),
    ]

    subset_sizes = [
        size
        for size in args.subset_sizes
        if size <= problem.train.n_components and size <= args.value_budget
    ]
    if not subset_sizes:
        subset_sizes = [
            min(problem.train.n_components, args.value_budget)
        ]
    print(
        f"Running SMax with d={problem.train.n_components:,}, "
        f"subsets={subset_sizes}, eta0={eta0:.4g}; "
        f"full-grid eta0={full_sgm_eta0:.4g}",
        flush=True,
    )
    for subset_size in subset_sizes:
        for seed in args.seeds:
            trajectories.append(
                run_smax(
                    problem.train,
                    wls,
                    lower_bounds,
                    upper_bounds,
                    args.value_budget,
                    subset_size,
                    eta0,
                    args.history_points,
                    seed,
                )
            )

    def full_once() -> Trajectory:
        return run_full_grid_sgm(
            problem.train,
            wls,
            lower_bounds,
            upper_bounds,
            args.value_budget,
            full_sgm_eta0,
            args.history_points,
            args.sample_chunk,
        )

    full = full_once()
    full_times = [float(full.optimizer_runtime_seconds[-1])]
    for _ in range(1, args.timing_repeats):
        repeated = full_once()
        full_times.append(float(repeated.optimizer_runtime_seconds[-1]))
    rescale_trajectory_timing(full, full_times)
    trajectories.append(full)

    wls_reference_max, _, _, _ = scan_max_and_norm(
        problem.train, wls, args.sample_chunk
    )

    def lse_once() -> Trajectory:
        return run_lse_lbfgsb(
            problem.train,
            wls,
            lower_bounds,
            upper_bounds,
            wls_reference_max,
            args.lse_multipliers,
            args.lse_maxiter_per_stage,
            args.sample_chunk,
            args.value_budget,
        )

    lse = lse_once()
    lse_times = [float(lse.optimizer_runtime_seconds[-1])]
    for _ in range(1, args.timing_repeats):
        repeated = lse_once()
        lse_times.append(float(repeated.optimizer_runtime_seconds[-1]))
    rescale_trajectory_timing(lse, lse_times)
    trajectories.append(lse)

    def lp_once() -> Trajectory:
        return run_restricted_lp_exchange(
            problem.train,
            wls,
            lower_bounds,
            upper_bounds,
            args.value_budget,
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
    rescale_trajectory_timing(lp, lp_times)
    trajectories.append(lp)

    slices: list[slice] = []
    coefficient_blocks: list[np.ndarray] = []
    offset = 0
    for trajectory in trajectories:
        stop = offset + len(trajectory.coefficients)
        slices.append(slice(offset, stop))
        coefficient_blocks.append(trajectory.coefficients)
        offset = stop
    all_coefficients = np.vstack(coefficient_blocks)
    print(
        f"Offline evaluation of {len(all_coefficients)} stored iterates...",
        flush=True,
    )
    evaluation = evaluate_coefficients(
        problem.train,
        problem.test,
        all_coefficients,
        args.sample_chunk,
        args.evaluation_coefficient_block,
    )
    scalar_diagnostics = scalar_polynomial_diagnostics(
        all_coefficients,
        y_fun,
        lower_bounds,
        upper_bounds,
        problem.scale_B,
    )

    smax_trajectories = [
        trajectory
        for trajectory in trajectories
        if trajectory.method == SMAX_METHOD
    ]
    first_smax = next(
        (
            trajectory
            for trajectory in smax_trajectories
            if trajectory.subset_size == 4096
        ),
        smax_trajectories[0],
    )
    middle_smax = first_smax.coefficients[
        len(first_smax.coefficients) // 2
    ]
    diagnostic_coefficients = {
        WLS_METHOD: wls,
        (
            f"{SMAX_METHOD} middle "
            f"(s={first_smax.subset_size}, seed={first_smax.seed})"
        ): middle_smax,
        f"{LP_METHOD} incumbent": lp.coefficients[-1],
    }
    tail_state = residual_tail_statistics(
        problem.train,
        diagnostic_coefficients,
        args.tail_sample_size,
        args.tail_seed,
    )
    exact_tail_state = exact_near_max_cardinality(
        problem.train, diagnostic_coefficients, args.sample_chunk
    )
    setup_state = {
        "train_random_access_mean_seconds": train_mean_seconds,
        "test_random_access_mean_seconds": test_mean_seconds,
        "scalar_functional_minimax_seconds": functional_seconds,
        "total_before_production_optimization_seconds": setup_elapsed_seconds,
    }
    save_results(
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
            FUNCTIONAL_METHOD: functional_seconds,
            CHEB_METHOD: cheb_seconds,
        },
        eta0,
        pilot_state,
        pilot_arrays,
        full_sgm_eta0,
        full_sgm_pilot_state,
        setup_state,
        tail_state,
        exact_tail_state,
    )
    from plot_results import make_plot

    plot_path = make_plot(args.output_dir)
    print(f"Saved reproducible results to {args.output_dir}", flush=True)
    print(f"Plot: {plot_path}", flush=True)


if __name__ == "__main__":
    main()
