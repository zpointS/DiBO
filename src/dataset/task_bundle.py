from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import numpy as np

Array: TypeAlias = np.ndarray

SUPPORTED_TASKS = (
    "TFBind8-Exact-v0",
    "TFBind10-Exact-v0",
    "AntMorphology-Exact-v0",
    "DKittyMorphology-Exact-v0",
)
PERCENTILE_CUTOFF = {
    "TFBind8-Exact-v0": 50.0,
    "AntMorphology-Exact-v0": 40.0,
    "DKittyMorphology-Exact-v0": 40.0,
}


@dataclass(frozen=True)
class TaskBundle:
    """Preprocessed designs and scores sorted by oracle score."""

    task_name: str
    X_raw_sorted: Array
    y_task_sorted: Array
    y_oracle_sorted: Array
    X_normalized_sorted: Array
    y_task_normalized_sorted: Array
    y_oracle_normalized_sorted: Array


def load_task_bundle(npz_path: str | Path) -> TaskBundle:
    path = Path(npz_path)
    with np.load(path, allow_pickle=False) as data:
        return TaskBundle(
            task_name=str(data["task_name"].tolist()),
            X_raw_sorted=data["X_raw_sorted"],
            y_task_sorted=data["y_task_sorted"],
            y_oracle_sorted=data["y_oracle_sorted"],
            X_normalized_sorted=data["X_normalized_sorted"],
            y_task_normalized_sorted=data["y_task_normalized_sorted"],
            y_oracle_normalized_sorted=data["y_oracle_normalized_sorted"],
        )


def save_task_bundle(bundle: TaskBundle, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{bundle.task_name}.npz"
    np.savez_compressed(
        path,
        task_name=np.array(bundle.task_name),
        X_raw_sorted=bundle.X_raw_sorted,
        y_task_sorted=bundle.y_task_sorted,
        y_oracle_sorted=bundle.y_oracle_sorted,
        X_normalized_sorted=bundle.X_normalized_sorted,
        y_task_normalized_sorted=bundle.y_task_normalized_sorted,
        y_oracle_normalized_sorted=bundle.y_oracle_normalized_sorted,
    )
    return path


def _load_xy(path: Path) -> tuple[Array, Array]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        return np.asarray(data["x"]), np.asarray(data["y"]).ravel()


def _visible_subset(x: Array, y: Array, max_percentile: float) -> tuple[Array, Array]:
    cutoff = np.percentile(y, max_percentile)
    visible = y <= cutoff
    return x[visible], y[visible]


def _to_logits(x: Array, soft_interpolation: float = 0.6) -> Array:
    if not np.issubdtype(x.dtype, np.integer):
        raise TypeError("Discrete designs must use an integer dtype.")
    one_hot = np.eye(4, dtype=np.float32)[x]
    uniform = np.full_like(one_hot, 0.25)
    probabilities = soft_interpolation * one_hot + (1.0 - soft_interpolation) * uniform
    log_probabilities = np.log(probabilities)
    return (log_probabilities[:, :, 1:] - log_probabilities[:, :, :1]).astype(np.float32)


def _normalize_x(x: Array) -> Array:
    x = np.asarray(x, dtype=np.float32)
    mean = np.mean(x, axis=0, keepdims=True)
    std = np.std(x, axis=0, keepdims=True)
    std = np.where(std == 0.0, 1.0, std)
    return ((x - mean) / std).astype(np.float32)


def _normalize_y(y: Array) -> Array:
    y = np.asarray(y, dtype=np.float32)
    std = float(np.std(y))
    if std == 0.0:
        std = 1.0
    return ((y - float(np.mean(y))) / std).astype(np.float32)


def _row_indices(rows: Array, reference: Array) -> Array:
    if rows.shape == reference.shape and np.array_equal(rows, reference):
        return np.arange(len(rows), dtype=np.int64)
    index = {np.ascontiguousarray(row).tobytes(): i for i, row in enumerate(reference)}
    try:
        return np.asarray(
            [index[np.ascontiguousarray(row).tobytes()] for row in rows],
            dtype=np.int64,
        )
    except KeyError as exc:
        raise ValueError("Relabeled designs do not match the visible raw dataset.") from exc


def build_task_bundle(
    task_name: str,
    *,
    raw_dir: str | Path = "data/raw",
    relabeled_dir: str | Path = "data/relabeled",
) -> TaskBundle:
    """Rebuild a bundle from the released raw artifacts."""

    if task_name not in SUPPORTED_TASKS:
        raise ValueError(f"Unsupported task: {task_name}")
    if task_name == "TFBind10-Exact-v0":
        raise ValueError(
            "The canonical TFBind10 bundle cannot be reconstructed exactly from the "
            "released corrected lookup because its historical tie ordering and "
            "normalization source were not retained. Use the included bundle."
        )

    raw_path = Path(raw_dir) / f"{task_name}.npz"
    x_full, y_full = _load_xy(raw_path)
    x_visible, y_visible = _visible_subset(
        x_full,
        y_full,
        PERCENTILE_CUTOFF[task_name],
    )

    if task_name == "TFBind8-Exact-v0":
        x_raw = x_visible
        y_task = y_visible.astype(np.float32)
        y_oracle = y_task.copy()
        x_normalized = _normalize_x(_to_logits(x_visible))
        y_task_normalized = _normalize_y(y_task)
    else:
        x_raw, y_oracle = _load_xy(Path(relabeled_dir) / f"{task_name}.npz")
        source_indices = _row_indices(x_raw, x_visible)
        y_task = y_visible[source_indices].astype(np.float32)
        x_normalized_visible = _normalize_x(x_visible)
        x_normalized = x_normalized_visible[source_indices]
        y_task_normalized = np.array([], dtype=np.float64)

    order = np.argsort(y_oracle)
    return TaskBundle(
        task_name=task_name,
        X_raw_sorted=x_raw[order],
        y_task_sorted=y_task[order],
        y_oracle_sorted=y_oracle[order],
        X_normalized_sorted=x_normalized[order],
        y_task_normalized_sorted=(
            y_task_normalized[order] if y_task_normalized.size else y_task_normalized
        ),
        y_oracle_normalized_sorted=np.array([], dtype=np.float64),
    )


def validate_bundle(path: str | Path, expected_task: str | None = None) -> TaskBundle:
    bundle = load_task_bundle(path)
    if expected_task is not None and bundle.task_name != expected_task:
        raise ValueError(f"Bundle task is {bundle.task_name!r}, expected {expected_task!r}.")
    if bundle.task_name not in SUPPORTED_TASKS:
        raise ValueError(f"Unsupported task: {bundle.task_name}")

    n = len(bundle.y_oracle_sorted)
    if n == 0:
        raise ValueError("Bundle is empty.")
    if bundle.X_raw_sorted.shape[0] != n or bundle.X_normalized_sorted.shape[0] != n:
        raise ValueError("Bundle arrays have inconsistent first dimensions.")
    if np.any(np.diff(bundle.y_oracle_sorted) < 0):
        raise ValueError("y_oracle_sorted is not non-decreasing.")
    return bundle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build or validate DiBO task bundles.")
    parser.add_argument("--stage", choices=("build", "validate"), default="validate")
    parser.add_argument(
        "--tasks", nargs="+", choices=SUPPORTED_TASKS, default=list(SUPPORTED_TASKS)
    )
    parser.add_argument("--raw_dir", default="data/raw")
    parser.add_argument("--relabeled_dir", default="data/relabeled")
    parser.add_argument("--bundle_dir", default="data/task_bundles")
    parser.add_argument("--output_dir", default="outputs/task_bundles")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.stage == "build":
        for task_name in args.tasks:
            bundle = build_task_bundle(
                task_name,
                raw_dir=args.raw_dir,
                relabeled_dir=args.relabeled_dir,
            )
            path = save_task_bundle(bundle, args.output_dir)
            print(f"[Bundle] wrote {path}")
    else:
        for task_name in args.tasks:
            path = Path(args.bundle_dir) / f"{task_name}.npz"
            bundle = validate_bundle(path, expected_task=task_name)
            print(f"[Bundle] valid {path} ({len(bundle.y_oracle_sorted)} rows)")


if __name__ == "__main__":
    main()
