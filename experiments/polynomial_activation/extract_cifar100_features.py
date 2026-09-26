"""Extract a finite-max design for the final ReLU of CIFAR-100 ResNet-20.

The pretrained model is loaded from ``chenyaofo/pytorch-cifar-models`` with
``torch.hub``.  For every CIFAR-100 image, the script manually exposes the
preactivation ``z`` of the ReLU in the final residual block.  If

    p_y(z) = B sum_{r=0}^q y_r T_r(z / B),

then the final-ReLU contribution to class ``k`` is affine in ``y``:

    contribution[n, k] = sum_r phi[n, k, r] y_r.

The arrays ``{split}_phi.npy`` and ``{split}_b.npy`` contain the design rows
and the original ReLU contributions, respectively.  Both are centered across
the 100 classes separately for each image, which removes a softmax-invariant
common logit shift.  They are standard ``.npy`` files written through NumPy
memory maps, so downstream optimizers can reopen them with
``np.load(path, mmap_mode="r")`` without loading them into RAM.

The extraction is intentionally two-pass.  A first pass over the training
split chooses ``B`` from the maximum absolute preactivation and a declared
margin.  A second pass writes the train and test designs and verifies the
manual forward, reconstructed logits, and affine Chebyshev contraction.

The default data backend streams the local Hugging Face Parquet mirror when
``data/hf/train.parquet`` and ``data/hf/test.parquet`` are present.  Otherwise
it falls back to ``torchvision.datasets.CIFAR100``.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import io
import json
import math
import os
import platform
import random
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Protocol

import numpy as np

# Required by torch's deterministic-algorithm guard for CUDA >= 10.2.  This
# must be established before the first cuBLAS handle is created.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn as nn
import torchvision
from PIL import Image
from numpy.lib.format import open_memmap
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


HUB_REPOSITORY = "chenyaofo/pytorch-cifar-models"
HUB_REF = "master"
MODEL_NAME = "cifar100_resnet20"
MODEL_WEIGHTS_URL = (
    "https://github.com/chenyaofo/pytorch-cifar-models/releases/download/"
    "resnet/cifar100_resnet20-23dac2f1.pt"
)
MODEL_CONFIG_URL = (
    "https://github.com/chenyaofo/image-classification-codebase/blob/"
    "master/conf/cifar100.conf"
)

# Exact values in chenyaofo/image-classification-codebase/conf/cifar100.conf,
# the training/evaluation configuration associated with the Hub checkpoint.
CIFAR100_MEAN = (0.5070, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2761)
CIFAR_IMAGE_SIZE = (32, 32)
NUM_CLASSES = 100

OUTPUT_FLOAT_DTYPE = np.dtype(np.float32)
OUTPUT_INT_DTYPE = np.dtype(np.int64)
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ImageBatch:
    """One deterministic, sequential dataset batch."""

    indices: np.ndarray
    images: torch.Tensor
    labels: torch.Tensor


@dataclass(frozen=True)
class ManualForward:
    """Outputs of the manually exposed final residual block."""

    preactivation: torch.Tensor
    activated: torch.Tensor
    pooled: torch.Tensor
    logits: torch.Tensor


@dataclass
class ActivationStats:
    """Streaming statistics for the final-block preactivation."""

    minimum: float = math.inf
    maximum: float = -math.inf
    max_abs: float = 0.0
    elements: int = 0
    out_of_range: int = 0

    def update(self, values: torch.Tensor, bound: float | None = None) -> None:
        detached = values.detach()
        self.minimum = min(self.minimum, float(detached.amin().item()))
        self.maximum = max(self.maximum, float(detached.amax().item()))
        self.max_abs = max(
            self.max_abs,
            float(detached.abs().amax().item()),
        )
        self.elements += detached.numel()
        if bound is not None:
            self.out_of_range += int(
                torch.count_nonzero(detached.abs() > bound).item()
            )

    def as_dict(self, bound: float | None = None) -> dict[str, Any]:
        result: dict[str, Any] = {
            "minimum": self.minimum,
            "maximum": self.maximum,
            "max_abs": self.max_abs,
            "elements": self.elements,
        }
        if bound is not None:
            result.update(
                {
                    "bound_B": bound,
                    "normalized_max_abs": self.max_abs / bound,
                    "out_of_range_count": self.out_of_range,
                    "out_of_range_fraction": (
                        self.out_of_range / self.elements
                        if self.elements
                        else 0.0
                    ),
                }
            )
        return result


@dataclass
class CheckStats:
    """Maximum errors accumulated over all verified batches."""

    manual_vs_model_logits: float = 0.0
    contribution_plus_bias_vs_model_logits: float = 0.0
    affine_design_vs_explicit_polynomial: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "manual_vs_model_logits_max_abs": self.manual_vs_model_logits,
            "contribution_plus_bias_vs_model_logits_max_abs": (
                self.contribution_plus_bias_vs_model_logits
            ),
            "affine_design_vs_explicit_polynomial_max_abs": (
                self.affine_design_vs_explicit_polynomial
            ),
        }


class SplitSource(Protocol):
    """Minimal interface shared by Parquet and torchvision backends."""

    split: str
    total_count: int
    count: int

    def iter_batches(self, batch_size: int) -> Iterator[ImageBatch]:
        """Yield deterministic batches in original dataset order."""

    def metadata(self) -> dict[str, Any]:
        """Return JSON-serializable source metadata."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    script_dir = Path(__file__).resolve().parent
    parser.add_argument(
        "--degree",
        "-q",
        type=int,
        default=14,
        help="Maximum Chebyshev degree q (default: 14).",
    )
    parser.add_argument(
        "--scale-margin",
        type=float,
        default=1.10,
        help="Set B to this factor times the training max |z| (default: 1.10).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Inference and Parquet decoding batch size (default: 128).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, or an explicit CUDA device such as cuda:0.",
    )
    parser.add_argument(
        "--data-source",
        choices=("auto", "parquet", "torchvision"),
        default="auto",
        help="Prefer local Parquet in auto mode, then torchvision fallback.",
    )
    parser.add_argument(
        "--parquet-root",
        type=Path,
        default=script_dir / "data" / "hf",
        help="Directory containing train.parquet and test.parquet.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=script_dir / "data",
        help="torchvision CIFAR-100 download/extraction directory.",
    )
    parser.add_argument(
        "--torch-cache",
        type=Path,
        default=script_dir / ".torch_cache",
        help="Experiment-local root for the Torch Hub cache.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. The default is "
            "features/cifar100_resnet20_q{degree}."
        ),
    )
    parser.add_argument(
        "--train-limit",
        type=int,
        default=None,
        help="Use only the first N training examples.",
    )
    parser.add_argument(
        "--test-limit",
        type=int,
        default=None,
        help="Use only the first N test examples.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Default unspecified limits to 256 train and 128 test examples.",
    )
    parser.add_argument(
        "--download",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow torchvision to download/extract CIFAR-100 (default: true).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers for the torchvision backend (default: 0).",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=0,
        help="Set torch CPU threads when positive; 0 keeps the environment default.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--hub-force-reload",
        action="store_true",
        help="Force a fresh Torch Hub source download.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite this script's known output files if they exist.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="Print progress every N batches; 0 disables intermediate logs.",
    )
    parser.add_argument(
        "--check-atol",
        type=float,
        default=2e-5,
        help="Absolute tolerance for reconstructed logits.",
    )
    parser.add_argument(
        "--check-rtol",
        type=float,
        default=1e-5,
        help="Relative tolerance for reconstructed logits.",
    )
    parser.add_argument(
        "--design-check-atol",
        type=float,
        default=2e-4,
        help="Absolute tolerance for the affine-design identity.",
    )
    parser.add_argument(
        "--skip-data-sha256",
        action="store_true",
        help="Skip hashing the source Parquet/archive files.",
    )
    parser.add_argument(
        "--store-original-logits",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Store raw pretrained-model logits in addition to labels and "
            "predictions (default: true)."
        ),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.degree < 0:
        raise ValueError("--degree must be nonnegative.")
    if args.scale_margin < 1.0:
        raise ValueError("--scale-margin must be at least 1.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be nonnegative.")
    if args.cpu_threads < 0:
        raise ValueError("--cpu-threads must be nonnegative.")
    if args.progress_every < 0:
        raise ValueError("--progress-every must be nonnegative.")
    for name in ("train_limit", "test_limit"):
        value = getattr(args, name)
        if value is not None and value < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    for name in ("check_atol", "check_rtol", "design_check_atol"):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be nonnegative.")


def finalize_args(args: argparse.Namespace) -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    if args.smoke:
        if args.train_limit is None:
            args.train_limit = 256
        if args.test_limit is None:
            args.test_limit = 128
    if args.output_dir is None:
        suffix = "_smoke" if args.smoke else ""
        args.output_dir = (
            script_dir
            / "features"
            / f"cifar100_resnet20_q{args.degree}{suffix}"
        )
    args.output_dir = args.output_dir.resolve()
    args.parquet_root = args.parquet_root.resolve()
    args.data_root = args.data_root.resolve()
    args.torch_cache = args.torch_cache.resolve()
    return args


def configure_determinism(seed: int, cpu_threads: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False
    if cpu_threads > 0:
        torch.set_num_threads(cpu_threads)


def select_device(requested: str) -> torch.device:
    normalized = requested.lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested ({requested}) but is unavailable.")
    return device


def preprocessing_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )


def limited_count(total: int, limit: int | None) -> int:
    return total if limit is None else min(total, limit)


class ParquetSplitSource:
    """Streaming CIFAR split backed by a Hugging Face Parquet file."""

    def __init__(
        self,
        path: Path,
        split: str,
        transform: transforms.Compose,
        limit: int | None,
        include_sha256: bool,
    ) -> None:
        try:
            import pyarrow.parquet as pq
        except ImportError as error:
            raise RuntimeError(
                "The Parquet backend requires pyarrow. Install it in the "
                "requested environment or select --data-source torchvision."
            ) from error

        self.path = path
        self.split = split
        self.transform = transform
        self._parquet = pq.ParquetFile(path)
        self.total_count = int(self._parquet.metadata.num_rows)
        self.count = limited_count(self.total_count, limit)
        self._include_sha256 = include_sha256
        fields = set(self._parquet.schema_arrow.names)
        required = {"img", "fine_label"}
        if not required.issubset(fields):
            raise ValueError(
                f"{path} lacks required columns {sorted(required)}; "
                f"found {sorted(fields)}."
            )

    def iter_batches(self, batch_size: int) -> Iterator[ImageBatch]:
        emitted = 0
        for record_batch in self._parquet.iter_batches(
            batch_size=batch_size,
            columns=["img", "fine_label"],
            use_threads=False,
        ):
            if emitted >= self.count:
                break
            take = min(record_batch.num_rows, self.count - emitted)
            image_struct = record_batch.column(0).slice(0, take)
            byte_values = image_struct.field("bytes")
            path_values = image_struct.field("path")
            labels = np.asarray(
                record_batch.column(1).slice(0, take).to_numpy(),
                dtype=np.int64,
            ).copy()
            tensors: list[torch.Tensor] = []
            for position in range(take):
                raw = byte_values[position].as_py()
                image_path = path_values[position].as_py()
                if raw is not None:
                    image_context = Image.open(io.BytesIO(raw))
                elif image_path is not None:
                    candidate = Path(image_path)
                    if not candidate.is_absolute():
                        candidate = self.path.parent / candidate
                    image_context = Image.open(candidate)
                else:
                    raise ValueError(
                        f"{self.path}: image row {emitted + position} has "
                        "neither embedded bytes nor a path."
                    )
                with image_context as image:
                    rgb = image.convert("RGB")
                    if rgb.size != CIFAR_IMAGE_SIZE:
                        raise ValueError(
                            f"Unexpected image size {rgb.size} at "
                            f"{self.split}[{emitted + position}]."
                        )
                    tensors.append(self.transform(rgb))
            indices = np.arange(emitted, emitted + take, dtype=np.int64)
            emitted += take
            yield ImageBatch(
                indices=indices,
                images=torch.stack(tensors, dim=0),
                labels=torch.from_numpy(labels),
            )
        if emitted != self.count:
            raise RuntimeError(
                f"Expected {self.count} rows from {self.path}, got {emitted}."
            )

    def metadata(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "backend": "huggingface_parquet",
            "split": self.split,
            "path": str(self.path),
            "total_rows": self.total_count,
            "selected_rows": self.count,
            "row_groups": int(self._parquet.metadata.num_row_groups),
            "schema": str(self._parquet.schema_arrow),
            "streaming": (
                "pyarrow ParquetFile.iter_batches in physical row order; "
                "images decoded only within the current batch"
            ),
            "file_bytes": self.path.stat().st_size,
        }
        if self._include_sha256:
            result["sha256"] = sha256_file(self.path)
        return result


def seed_data_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class TorchvisionSplitSource:
    """Sequential CIFAR-100 split using torchvision's official archive."""

    def __init__(
        self,
        root: Path,
        split: str,
        transform: transforms.Compose,
        limit: int | None,
        download: bool,
        num_workers: int,
        seed: int,
        include_sha256: bool,
    ) -> None:
        self.root = root
        self.split = split
        self.num_workers = num_workers
        self.seed = seed
        self.dataset = datasets.CIFAR100(
            root=str(root),
            train=split == "train",
            transform=transform,
            download=download,
        )
        self.total_count = len(self.dataset)
        self.count = limited_count(self.total_count, limit)
        self._include_sha256 = include_sha256

    def iter_batches(self, batch_size: int) -> Iterator[ImageBatch]:
        subset = Subset(self.dataset, range(self.count))
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            persistent_workers=self.num_workers > 0,
            worker_init_fn=seed_data_worker,
            generator=generator,
        )
        emitted = 0
        for images, labels in loader:
            take = int(labels.shape[0])
            indices = np.arange(emitted, emitted + take, dtype=np.int64)
            emitted += take
            yield ImageBatch(
                indices=indices,
                images=images,
                labels=labels.to(dtype=torch.int64),
            )
        if emitted != self.count:
            raise RuntimeError(
                f"Expected {self.count} torchvision rows, got {emitted}."
            )

    def metadata(self) -> dict[str, Any]:
        archive_path = self.root / self.dataset.filename
        result: dict[str, Any] = {
            "backend": "torchvision.datasets.CIFAR100",
            "split": self.split,
            "root": str(self.root),
            "total_rows": self.total_count,
            "selected_rows": self.count,
            "url": self.dataset.url,
            "filename": self.dataset.filename,
            "archive_md5": self.dataset.tgz_md5,
            "archive_path": str(archive_path),
            "num_workers": self.num_workers,
            "shuffle": False,
            "augmentation": False,
        }
        if archive_path.exists():
            result["archive_bytes"] = archive_path.stat().st_size
            if self._include_sha256:
                result["archive_sha256"] = sha256_file(archive_path)
        return result


def choose_data_backend(args: argparse.Namespace) -> str:
    train_path = args.parquet_root / "train.parquet"
    test_path = args.parquet_root / "test.parquet"
    has_both_parquet = train_path.is_file() and test_path.is_file()
    if args.data_source == "auto":
        return "parquet" if has_both_parquet else "torchvision"
    if args.data_source == "parquet" and not has_both_parquet:
        raise FileNotFoundError(
            "Parquet mode requires both files: "
            f"{train_path} and {test_path}."
        )
    return args.data_source


def build_sources(
    args: argparse.Namespace,
) -> tuple[SplitSource, SplitSource, str]:
    transform = preprocessing_transform()
    backend = choose_data_backend(args)
    include_sha256 = not args.skip_data_sha256
    if backend == "parquet":
        train_source: SplitSource = ParquetSplitSource(
            args.parquet_root / "train.parquet",
            "train",
            transform,
            args.train_limit,
            include_sha256,
        )
        test_source: SplitSource = ParquetSplitSource(
            args.parquet_root / "test.parquet",
            "test",
            transform,
            args.test_limit,
            include_sha256,
        )
    else:
        train_source = TorchvisionSplitSource(
            args.data_root,
            "train",
            transform,
            args.train_limit,
            args.download,
            args.num_workers,
            args.seed,
            include_sha256,
        )
        test_source = TorchvisionSplitSource(
            args.data_root,
            "test",
            transform,
            args.test_limit,
            args.download,
            args.num_workers,
            args.seed,
            include_sha256,
        )
    return train_source, test_source, backend


def load_model(args: argparse.Namespace, device: torch.device) -> nn.Module:
    hub_dir = args.torch_cache / "hub"
    hub_dir.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(hub_dir))
    model = torch.hub.load(
        f"{HUB_REPOSITORY}:{HUB_REF}",
        MODEL_NAME,
        pretrained=True,
        force_reload=args.hub_force_reload,
        trust_repo=True,
    )
    if not isinstance(model, nn.Module):
        raise TypeError(f"Torch Hub returned {type(model)!r}, not nn.Module.")
    model.eval()
    model.requires_grad_(False)
    model.to(device)
    validate_model_architecture(model)
    return model


def validate_model_architecture(model: nn.Module) -> None:
    required = (
        "conv1",
        "bn1",
        "relu",
        "layer1",
        "layer2",
        "layer3",
        "avgpool",
        "fc",
    )
    missing = [name for name in required if not hasattr(model, name)]
    if missing:
        raise ValueError(f"Model lacks required modules: {missing}.")
    layer3 = list(model.layer3.children())
    if not layer3:
        raise ValueError("model.layer3 has no residual blocks.")
    final_block = layer3[-1]
    block_required = ("conv1", "bn1", "relu", "conv2", "bn2", "downsample")
    block_missing = [
        name for name in block_required if not hasattr(final_block, name)
    ]
    if block_missing:
        raise ValueError(
            f"Final residual block lacks required modules: {block_missing}."
        )
    if not isinstance(model.fc, nn.Linear):
        raise TypeError("The model's final classifier must be nn.Linear.")
    if model.fc.in_features != 64 or model.fc.out_features != NUM_CLASSES:
        raise ValueError(
            "Expected a 64-to-100 classifier for CIFAR-100 ResNet-20, got "
            f"{model.fc.in_features}-to-{model.fc.out_features}."
        )
    if model.fc.bias is None:
        raise ValueError("The final classifier is expected to have a bias.")


def manual_final_block_forward(
    model: nn.Module,
    images: torch.Tensor,
) -> ManualForward:
    """Run ResNet-20 while preserving the final residual preactivation."""

    x = model.conv1(images)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.layer1(x)
    x = model.layer2(x)

    layer3_blocks = list(model.layer3.children())
    for block in layer3_blocks[:-1]:
        x = block(x)

    block = layer3_blocks[-1]
    identity = x
    out = block.conv1(x)
    out = block.bn1(out)
    out = block.relu(out)
    out = block.conv2(out)
    out = block.bn2(out)
    if block.downsample is not None:
        identity = block.downsample(x)
    preactivation = out + identity

    # Do not call the block's in-place ReLU here: z must remain available.
    activated = torch.relu(preactivation)
    pooled = model.avgpool(activated)
    pooled = torch.flatten(pooled, 1)
    logits = model.fc(pooled)
    return ManualForward(
        preactivation=preactivation,
        activated=activated,
        pooled=pooled,
        logits=logits,
    )


def scan_training_scale(
    source: SplitSource,
    model: nn.Module,
    device: torch.device,
    batch_size: int,
    progress_every: int,
) -> tuple[ActivationStats, float, tuple[int, ...]]:
    stats = ActivationStats()
    activation_shape: tuple[int, ...] | None = None
    start = time.perf_counter()
    with torch.inference_mode():
        for batch_number, batch in enumerate(
            source.iter_batches(batch_size),
            start=1,
        ):
            images = batch.images.to(device, non_blocking=False)
            manual = manual_final_block_forward(model, images)
            stats.update(manual.preactivation)
            current_shape = tuple(int(x) for x in manual.preactivation.shape[1:])
            if activation_shape is None:
                activation_shape = current_shape
            elif current_shape != activation_shape:
                raise RuntimeError(
                    "Final preactivation shape changed from "
                    f"{activation_shape} to {current_shape}."
                )
            if progress_every and batch_number % progress_every == 0:
                processed = min(batch_number * batch_size, source.count)
                print(
                    f"  scale pass: {processed}/{source.count}; "
                    f"max |z|={stats.max_abs:.7g}",
                    flush=True,
                )
    synchronize_if_needed(device)
    elapsed = time.perf_counter() - start
    if activation_shape is None or stats.elements == 0:
        raise RuntimeError("The training scale pass produced no activations.")
    return stats, elapsed, activation_shape


def deterministic_check_coefficients(
    degree: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    orders = torch.arange(degree + 1, device=device, dtype=dtype)
    signs = torch.where(
        (orders.to(torch.int64) % 2) == 0,
        torch.ones_like(orders),
        -torch.ones_like(orders),
    )
    return signs / ((degree + 1) * (orders + 1))


def chebyshev_affine_design(
    preactivation: torch.Tensor,
    bound: float,
    degree: int,
    fc_weight: torch.Tensor,
    check_coefficients: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return phi[n, class, r] and an explicit check-polynomial output."""

    scaled = preactivation / bound
    batch_size = int(scaled.shape[0])
    num_classes = int(fc_weight.shape[0])
    design = torch.empty(
        (batch_size, num_classes, degree + 1),
        dtype=scaled.dtype,
        device=scaled.device,
    )

    previous = torch.ones_like(scaled)
    pooled = previous.mean(dim=(2, 3))
    design[:, :, 0] = pooled @ fc_weight.t()
    explicit_polynomial = check_coefficients[0] * previous

    if degree >= 1:
        current = scaled
        pooled = current.mean(dim=(2, 3))
        design[:, :, 1] = pooled @ fc_weight.t()
        explicit_polynomial = (
            explicit_polynomial + check_coefficients[1] * current
        )
        for order in range(2, degree + 1):
            following = 2.0 * scaled * current - previous
            pooled = following.mean(dim=(2, 3))
            design[:, :, order] = pooled @ fc_weight.t()
            explicit_polynomial = (
                explicit_polynomial
                + check_coefficients[order] * following
            )
            previous, current = current, following

    explicit_contribution = (
        explicit_polynomial.mean(dim=(2, 3)) @ fc_weight.t()
    )
    # The normalized activation is p(z)=B*sum_r y_r*T_r(z/B), so y is
    # dimensionless. Centering over classes removes a sample-specific logit
    # shift without changing softmax probabilities or predictions.
    design = bound * design
    design = design - design.mean(dim=1, keepdim=True)
    explicit_contribution = bound * explicit_contribution
    explicit_contribution = (
        explicit_contribution
        - explicit_contribution.mean(dim=1, keepdim=True)
    )
    return design, explicit_contribution


def assert_close_and_measure(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    atol: float,
    rtol: float,
) -> float:
    if actual.shape != expected.shape:
        raise AssertionError(
            f"{name}: shape mismatch {actual.shape} != {expected.shape}."
        )
    absolute = (actual - expected).abs()
    max_absolute = float(absolute.amax().item()) if absolute.numel() else 0.0
    allowed = atol + rtol * expected.abs()
    violation = absolute > allowed
    if bool(torch.any(violation).item()):
        count = int(torch.count_nonzero(violation).item())
        worst_flat = int(torch.argmax(absolute).item())
        raise AssertionError(
            f"{name} failed for {count}/{absolute.numel()} entries; "
            f"max abs error={max_absolute:.8e} at flat index {worst_flat}, "
            f"atol={atol:.3e}, rtol={rtol:.3e}."
        )
    return max_absolute


def split_output_paths(
    output_dir: Path,
    split: str,
    store_original_logits: bool = True,
) -> dict[str, Path]:
    paths = {
        "phi": output_dir / f"{split}_phi.npy",
        "b": output_dir / f"{split}_b.npy",
        "labels": output_dir / f"{split}_labels.npy",
        "predictions": output_dir / f"{split}_original_predictions.npy",
        "indices": output_dir / f"{split}_indices.npy",
    }
    if store_original_logits:
        paths["logits"] = output_dir / f"{split}_original_logits.npy"
    return paths


def create_split_memmaps(
    paths: dict[str, Path],
    count: int,
    degree: int,
) -> dict[str, np.memmap]:
    arrays = {
        "phi": open_memmap(
            paths["phi"],
            mode="w+",
            dtype=OUTPUT_FLOAT_DTYPE,
            shape=(count, NUM_CLASSES, degree + 1),
        ),
        "b": open_memmap(
            paths["b"],
            mode="w+",
            dtype=OUTPUT_FLOAT_DTYPE,
            shape=(count, NUM_CLASSES),
        ),
        "labels": open_memmap(
            paths["labels"],
            mode="w+",
            dtype=OUTPUT_INT_DTYPE,
            shape=(count,),
        ),
        "predictions": open_memmap(
            paths["predictions"],
            mode="w+",
            dtype=OUTPUT_INT_DTYPE,
            shape=(count,),
        ),
        "indices": open_memmap(
            paths["indices"],
            mode="w+",
            dtype=OUTPUT_INT_DTYPE,
            shape=(count,),
        ),
    }
    if "logits" in paths:
        arrays["logits"] = open_memmap(
            paths["logits"],
            mode="w+",
            dtype=OUTPUT_FLOAT_DTYPE,
            shape=(count, NUM_CLASSES),
        )
    return arrays


def flush_memmaps(memmap_arrays: dict[str, np.memmap]) -> None:
    for array in memmap_arrays.values():
        array.flush()


def extract_split(
    source: SplitSource,
    model: nn.Module,
    device: torch.device,
    output_dir: Path,
    bound: float,
    degree: int,
    batch_size: int,
    progress_every: int,
    check_atol: float,
    check_rtol: float,
    design_check_atol: float,
    store_original_logits: bool,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    paths = split_output_paths(
        output_dir,
        source.split,
        store_original_logits,
    )
    arrays = create_split_memmaps(paths, source.count, degree)
    stats = ActivationStats()
    checks = CheckStats()
    correct_top1 = 0
    correct_top5 = 0
    processed = 0
    fc_weight = model.fc.weight.detach()
    fc_bias = model.fc.bias.detach()
    check_coefficients = deterministic_check_coefficients(
        degree,
        device,
        fc_weight.dtype,
    )
    start = time.perf_counter()

    try:
        with torch.inference_mode():
            for batch_number, batch in enumerate(
                source.iter_batches(batch_size),
                start=1,
            ):
                expected_indices = np.arange(
                    processed,
                    processed + len(batch.indices),
                    dtype=np.int64,
                )
                if not np.array_equal(batch.indices, expected_indices):
                    raise RuntimeError(
                        f"{source.split} source order is not sequential at "
                        f"output row {processed}."
                    )
                images = batch.images.to(device, non_blocking=False)
                labels = batch.labels.to(
                    device,
                    dtype=torch.int64,
                    non_blocking=False,
                )
                if bool(torch.any((labels < 0) | (labels >= NUM_CLASSES)).item()):
                    raise ValueError(
                        f"{source.split} contains labels outside [0, 99]."
                    )

                original_logits = model(images)
                manual = manual_final_block_forward(model, images)
                manual_error = assert_close_and_measure(
                    f"{source.split}: manual forward vs model",
                    manual.logits,
                    original_logits,
                    check_atol,
                    check_rtol,
                )
                checks.manual_vs_model_logits = max(
                    checks.manual_vs_model_logits,
                    manual_error,
                )

                raw_contribution = manual.pooled @ fc_weight.t()
                reconstructed_logits = raw_contribution + fc_bias
                reconstruction_error = assert_close_and_measure(
                    f"{source.split}: contribution + bias vs model",
                    reconstructed_logits,
                    original_logits,
                    check_atol,
                    check_rtol,
                )
                checks.contribution_plus_bias_vs_model_logits = max(
                    checks.contribution_plus_bias_vs_model_logits,
                    reconstruction_error,
                )

                design, explicit_check = chebyshev_affine_design(
                    manual.preactivation,
                    bound,
                    degree,
                    fc_weight,
                    check_coefficients,
                )
                if not bool(torch.all(torch.isfinite(design)).item()):
                    raise FloatingPointError(
                        f"Nonfinite Chebyshev design in {source.split} "
                        f"batch {batch_number}."
                    )
                affine_check = torch.einsum(
                    "nkr,r->nk",
                    design,
                    check_coefficients,
                )
                design_error = assert_close_and_measure(
                    f"{source.split}: affine design identity",
                    affine_check,
                    explicit_check,
                    design_check_atol,
                    check_rtol,
                )
                checks.affine_design_vs_explicit_polynomial = max(
                    checks.affine_design_vs_explicit_polynomial,
                    design_error,
                )

                contribution = (
                    raw_contribution
                    - raw_contribution.mean(dim=1, keepdim=True)
                )
                centered_predictions = torch.argmax(
                    contribution + fc_bias,
                    dim=1,
                )
                stats.update(manual.preactivation, bound)
                predictions = torch.argmax(original_logits, dim=1)
                if not bool(torch.all(centered_predictions == predictions).item()):
                    raise AssertionError(
                        f"{source.split}: class-centering changed a model "
                        "prediction, which should be impossible."
                    )
                top5 = torch.topk(
                    original_logits,
                    k=5,
                    dim=1,
                    largest=True,
                    sorted=True,
                ).indices
                correct_top1 += int(
                    torch.count_nonzero(predictions == labels).item()
                )
                correct_top5 += int(
                    torch.count_nonzero(
                        torch.any(top5 == labels[:, None], dim=1)
                    ).item()
                )

                batch_count = int(labels.shape[0])
                rows = slice(processed, processed + batch_count)
                arrays["phi"][rows] = design.cpu().numpy()
                arrays["b"][rows] = contribution.cpu().numpy()
                arrays["labels"][rows] = labels.cpu().numpy()
                if "logits" in arrays:
                    arrays["logits"][rows] = original_logits.cpu().numpy()
                arrays["predictions"][rows] = predictions.cpu().numpy()
                arrays["indices"][rows] = batch.indices
                processed += batch_count

                if progress_every and batch_number % progress_every == 0:
                    print(
                        f"  {source.split}: {processed}/{source.count}; "
                        f"top-1={correct_top1 / processed:.4f}; "
                        f"max |z|/B={stats.max_abs / bound:.5f}",
                        flush=True,
                    )
    finally:
        flush_memmaps(arrays)

    synchronize_if_needed(device)
    elapsed = time.perf_counter() - start
    if processed != source.count:
        raise RuntimeError(
            f"Wrote {processed}/{source.count} rows for {source.split}."
        )
    if source.split == "train" and stats.out_of_range:
        raise AssertionError(
            f"Training pass found {stats.out_of_range} values outside B, "
            "despite B being selected from the same deterministic split."
        )

    file_metadata: dict[str, dict[str, Any]] = {}
    for semantic, path in paths.items():
        loaded = np.load(path, mmap_mode="r", allow_pickle=False)
        file_metadata[path.name] = {
            "semantic": semantic,
            "shape": list(loaded.shape),
            "dtype": str(loaded.dtype),
            "bytes_on_disk": path.stat().st_size,
        }

    summary = {
        "split": source.split,
        "examples": processed,
        "runtime_seconds": elapsed,
        "top1_accuracy": correct_top1 / processed,
        "top5_accuracy": correct_top5 / processed,
        "activation_stats": stats.as_dict(bound),
        "checks": checks.as_dict(),
    }
    return summary, file_metadata


def synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def sha256_file(path: Path, chunk_bytes: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def state_dict_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def python_tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def find_hub_source_root(model: nn.Module) -> tuple[Path, Path]:
    source_file = Path(inspect.getfile(type(model))).resolve()
    for candidate in (source_file.parent, *source_file.parents):
        if (candidate / "hubconf.py").is_file():
            return source_file, candidate
    return source_file, source_file.parent


def checkpoint_metadata(torch_cache: Path) -> list[dict[str, Any]]:
    checkpoint_dir = torch_cache / "hub" / "checkpoints"
    if not checkpoint_dir.is_dir():
        return []
    candidates = sorted(checkpoint_dir.glob(f"{MODEL_NAME}-*.pt"))
    return [
        {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in candidates
    ]


def output_file_names() -> set[str]:
    names = {"metadata.json", "fc_weight.npy", "fc_bias.npy"}
    for split in ("train", "test"):
        names.update(path.name for path in split_output_paths(Path(), split).values())
    return names


def prepare_output_directory(output_dir: Path, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [
        output_dir / name
        for name in output_file_names()
        if (output_dir / name).exists()
    ]
    if existing and not overwrite:
        rendered = "\n  ".join(str(path) for path in existing)
        raise FileExistsError(
            "Refusing to overwrite existing extraction outputs. Use "
            f"--overwrite to replace these known files:\n  {rendered}"
        )


def estimated_output_bytes(
    train_count: int,
    test_count: int,
    degree: int,
    store_original_logits: bool,
) -> int:
    per_example = (
        NUM_CLASSES * (degree + 1) * OUTPUT_FLOAT_DTYPE.itemsize
        + NUM_CLASSES * OUTPUT_FLOAT_DTYPE.itemsize
        + 3 * OUTPUT_INT_DTYPE.itemsize
    )
    if store_original_logits:
        per_example += NUM_CLASSES * OUTPUT_FLOAT_DTYPE.itemsize
    fc_bytes = (
        NUM_CLASSES * 64 + NUM_CLASSES
    ) * OUTPUT_FLOAT_DTYPE.itemsize
    return (train_count + test_count) * per_example + fc_bytes


def check_disk_capacity(output_dir: Path, required_bytes: int) -> dict[str, int]:
    usage = shutil.disk_usage(output_dir)
    if required_bytes > usage.free:
        raise OSError(
            f"Estimated outputs need {required_bytes / 2**30:.2f} GiB, "
            f"but only {usage.free / 2**30:.2f} GiB is free at {output_dir}."
        )
    return {
        "estimated_output_bytes": required_bytes,
        "free_bytes_before_extraction": usage.free,
    }


def jsonable_arguments(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def software_metadata(device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "pillow": Image.__version__,
        "device": str(device),
        "torch_num_threads": torch.get_num_threads(),
        "cuda_available": torch.cuda.is_available(),
        "cudnn": (
            str(torch.backends.cudnn.version())
            if torch.backends.cudnn.is_available()
            else None
        ),
        "deterministic_algorithms": (
            torch.are_deterministic_algorithms_enabled()
        ),
        "tf32_matmul": (
            torch.backends.cuda.matmul.allow_tf32
            if hasattr(torch.backends.cuda.matmul, "allow_tf32")
            else None
        ),
    }
    if device.type == "cuda":
        result["cuda_runtime"] = torch.version.cuda
        result["cuda_device_name"] = torch.cuda.get_device_name(device)
        result["cuda_device_capability"] = list(
            torch.cuda.get_device_capability(device)
        )
    try:
        import pyarrow

        result["pyarrow"] = pyarrow.__version__
    except ImportError:
        result["pyarrow"] = None
    return result


def write_metadata_atomic(path: Path, metadata: dict[str, Any]) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    args = finalize_args(parse_args())
    validate_args(args)
    configure_determinism(args.seed, args.cpu_threads)
    device = select_device(args.device)

    print(f"Using device: {device}", flush=True)
    train_source, test_source, backend = build_sources(args)
    print(
        f"Data backend: {backend}; train={train_source.count}; "
        f"test={test_source.count}",
        flush=True,
    )
    prepare_output_directory(args.output_dir, args.overwrite)
    disk_metadata = check_disk_capacity(
        args.output_dir,
        estimated_output_bytes(
            train_source.count,
            test_source.count,
            args.degree,
            args.store_original_logits,
        ),
    )

    print("Loading pretrained CIFAR-100 ResNet-20 from Torch Hub...", flush=True)
    model = load_model(args, device)
    source_file, hub_source_root = find_hub_source_root(model)

    print("First pass: selecting B from training preactivations...", flush=True)
    first_pass_stats, first_pass_seconds, activation_shape = scan_training_scale(
        train_source,
        model,
        device,
        args.batch_size,
        args.progress_every,
    )
    bound = args.scale_margin * first_pass_stats.max_abs
    if not math.isfinite(bound) or bound <= 0:
        raise FloatingPointError(f"Invalid scale B={bound}.")
    print(
        f"Selected B={bound:.9g} = {args.scale_margin:g} * "
        f"{first_pass_stats.max_abs:.9g}",
        flush=True,
    )

    print("Second pass: extracting training design...", flush=True)
    train_summary, train_files = extract_split(
        train_source,
        model,
        device,
        args.output_dir,
        bound,
        args.degree,
        args.batch_size,
        args.progress_every,
        args.check_atol,
        args.check_rtol,
        args.design_check_atol,
        args.store_original_logits,
    )
    print("Second pass: extracting test design...", flush=True)
    test_summary, test_files = extract_split(
        test_source,
        model,
        device,
        args.output_dir,
        bound,
        args.degree,
        args.batch_size,
        args.progress_every,
        args.check_atol,
        args.check_rtol,
        args.design_check_atol,
        args.store_original_logits,
    )

    fc_weight_path = args.output_dir / "fc_weight.npy"
    fc_bias_path = args.output_dir / "fc_bias.npy"
    np.save(
        fc_weight_path,
        model.fc.weight.detach().cpu().numpy().astype(np.float32, copy=False),
        allow_pickle=False,
    )
    np.save(
        fc_bias_path,
        model.fc.bias.detach().cpu().numpy().astype(np.float32, copy=False),
        allow_pickle=False,
    )
    model_files = {
        fc_weight_path.name: {
            "semantic": "final classifier weights W[class, channel]",
            "shape": list(model.fc.weight.shape),
            "dtype": "float32",
            "bytes_on_disk": fc_weight_path.stat().st_size,
        },
        fc_bias_path.name: {
            "semantic": "final classifier bias c[class]",
            "shape": list(model.fc.bias.shape),
            "dtype": "float32",
            "bytes_on_disk": fc_bias_path.stat().st_size,
        },
    }

    check_coefficients = (
        deterministic_check_coefficients(
            args.degree,
            torch.device("cpu"),
            torch.float32,
        )
        .numpy()
        .tolist()
    )
    metadata: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "arguments": jsonable_arguments(args),
        "script": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "problem": {
            "activation": "ReLU in the final residual block",
            "preactivation_symbol": "z",
            "basis": "B * Chebyshev T_r(z/B), r=0,...,q",
            "normalized_activation": (
                "p_y(z) = B * sum_{r=0}^q y_r T_r(z/B); coefficients y "
                "are dimensionless"
            ),
            "degree_q": args.degree,
            "scale_B": bound,
            "scale_margin": args.scale_margin,
            "design_equation": (
                "Replacing ReLU(z) by B*sum_r y_r T_r(z/B), applying GAP "
                "and the FC weights, then centering over classes gives "
                "sum_r phi[n,class,r] y_r."
            ),
            "target_b": (
                "Original final-ReLU logit contribution W@GAP(ReLU(z)), "
                "centered over the 100 classes separately for each image."
            ),
            "full_logit_reconstruction": (
                "softmax(original_logits) = softmax(b + fc_bias); equality "
                "holds up to the removed per-image common logit shift. Raw "
                "original logits are optionally stored."
            ),
            "class_centering": (
                "phi[n,:,r] and b[n,:] each have zero class mean; this is "
                "softmax- and argmax-invariant."
            ),
            "component_indexing": (
                "One affine residual per selected image and class; phi has "
                "axis order [image, class, Chebyshev order]."
            ),
            "training_scalar_components": train_source.count * NUM_CLASSES,
            "full_training_scalar_components": 50_000 * NUM_CLASSES,
            "full_training_component_description": (
                "50,000 images * 100 centered class-logit residuals = "
                "5,000,000 scalar finite-max components"
            ),
        },
        "preprocessing": {
            "dataset": "CIFAR-100",
            "image_mode": "RGB",
            "image_size": list(CIFAR_IMAGE_SIZE),
            "to_tensor_scale": "uint8 divided by 255",
            "normalize_mean": list(CIFAR100_MEAN),
            "normalize_std": list(CIFAR100_STD),
            "augmentation": False,
            "shuffle": False,
            "source_config": MODEL_CONFIG_URL,
        },
        "data": {
            "selected_backend": backend,
            "train": train_source.metadata(),
            "test": test_source.metadata(),
        },
        "model": {
            "hub_repository": HUB_REPOSITORY,
            "hub_ref": HUB_REF,
            "hub_entrypoint": MODEL_NAME,
            "pretrained": True,
            "weights_url": MODEL_WEIGHTS_URL,
            "model_class": (
                f"{type(model).__module__}.{type(model).__qualname__}"
            ),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "state_dict_sha256": state_dict_sha256(model),
            "source_file": str(source_file),
            "source_file_sha256": sha256_file(source_file),
            "hub_source_root": str(hub_source_root),
            "hub_python_tree_sha256": python_tree_sha256(hub_source_root),
            "checkpoint_files": checkpoint_metadata(args.torch_cache),
            "final_block": "layer3[-1]",
            "final_preactivation_shape_per_image": list(activation_shape),
            "classifier_in_features": model.fc.in_features,
            "classifier_out_features": model.fc.out_features,
        },
        "scale_selection": {
            "split": "train",
            "pass": 1,
            "runtime_seconds": first_pass_seconds,
            "training_preactivation": first_pass_stats.as_dict(),
            "rule": "B = scale_margin * training max_abs",
            "B": bound,
        },
        "extraction": {
            "pass": 2,
            "output_float_dtype": str(OUTPUT_FLOAT_DTYPE),
            "npy_memory_mappable": True,
            "store_original_logits": args.store_original_logits,
            "train": train_summary,
            "test": test_summary,
            "check_coefficients": check_coefficients,
            "check_tolerances": {
                "logit_atol": args.check_atol,
                "logit_rtol": args.check_rtol,
                "affine_design_atol": args.design_check_atol,
            },
        },
        "files": {
            **train_files,
            **test_files,
            **model_files,
        },
        "disk": disk_metadata,
        "software_and_machine": software_metadata(device),
    }
    write_metadata_atomic(args.output_dir / "metadata.json", metadata)

    print(
        "Extraction complete.\n"
        f"  output: {args.output_dir}\n"
        f"  B: {bound:.9g}\n"
        f"  train top-1: {train_summary['top1_accuracy']:.4%}\n"
        f"  test top-1: {test_summary['top1_accuracy']:.4%}\n"
        f"  test values outside [-B,B]: "
        f"{test_summary['activation_stats']['out_of_range_count']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
