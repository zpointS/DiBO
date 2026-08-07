from __future__ import annotations

import argparse
from collections import defaultdict, deque
import json
from pathlib import Path
from typing import Mapping

import numpy as np

ROW_ARRAYS = ("X_raw_sorted", "y_task_sorted", "y_oracle_sorted")
NORMALIZED_ARRAYS = (
    "X_normalized_sorted",
    "y_task_normalized_sorted",
    "y_oracle_normalized_sorted",
)


def compare_arrays(
    name: str,
    ref: np.ndarray,
    new: np.ndarray,
    *,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> dict | None:
    if ref.shape != new.shape:
        return {
            "array": name,
            "kind": "shape",
            "reference": list(ref.shape),
            "candidate": list(new.shape),
        }
    if ref.dtype != new.dtype:
        return {
            "array": name,
            "kind": "dtype",
            "reference": str(ref.dtype),
            "candidate": str(new.dtype),
        }
    if np.array_equal(ref, new):
        return None
    if (
        np.issubdtype(ref.dtype, np.number)
        and np.isfinite(ref).all()
        and np.isfinite(new).all()
        and np.allclose(ref, new, atol=atol, rtol=rtol)
    ):
        return None

    diff = {
        "array": name,
        "kind": "value",
        "shape": list(ref.shape),
        "dtype": str(ref.dtype),
    }
    if ref.size and np.issubdtype(ref.dtype, np.number):
        delta = ref.astype(np.float64) - new.astype(np.float64)
        diff["max_abs"] = float(np.nanmax(np.abs(delta)))
        mismatch = np.argwhere(ref != new)
        if mismatch.size:
            idx = tuple(int(i) for i in mismatch[0])
            diff["first_index"] = list(idx)
            diff["reference_value"] = float(ref[idx])
            diff["candidate_value"] = float(new[idx])
    return diff


def exact_diffs(
    reference: Mapping[str, np.ndarray],
    candidate: Mapping[str, np.ndarray],
) -> list[dict]:
    diffs: list[dict] = []
    reference_keys = set(reference.keys())
    candidate_keys = set(candidate.keys())
    if reference_keys != candidate_keys:
        diffs.append(
            {
                "kind": "keys",
                "reference_only": sorted(reference_keys - candidate_keys),
                "candidate_only": sorted(candidate_keys - reference_keys),
            }
        )
    for key in sorted(reference_keys & candidate_keys):
        diff = compare_arrays(key, reference[key], candidate[key])
        if diff is not None:
            diffs.append(diff)
    return diffs


def _row_key(x: np.ndarray, y_task: np.generic, y_oracle: np.generic) -> tuple[bytes, bytes, bytes]:
    return (
        np.ascontiguousarray(x).tobytes(),
        np.ascontiguousarray(y_task).tobytes(),
        np.ascontiguousarray(y_oracle).tobytes(),
    )


def semantic_diffs(
    reference: Mapping[str, np.ndarray],
    candidate: Mapping[str, np.ndarray],
    *,
    atol: float = 5e-4,
    rtol: float = 0.0,
) -> list[dict]:
    """Compare bundle contents while ignoring score-tie row ordering."""

    diffs: list[dict] = []
    expected = {"task_name", *ROW_ARRAYS, *NORMALIZED_ARRAYS}
    reference_keys = set(reference.keys())
    candidate_keys = set(candidate.keys())
    if reference_keys != expected or candidate_keys != expected:
        return [
            {
                "kind": "keys",
                "reference_only": sorted(expected - reference_keys),
                "reference_extra": sorted(reference_keys - expected),
                "candidate_only": sorted(expected - candidate_keys),
                "candidate_extra": sorted(candidate_keys - expected),
            }
        ]

    task_diff = compare_arrays(
        "task_name",
        reference["task_name"],
        candidate["task_name"],
    )
    if task_diff is not None:
        diffs.append(task_diff)

    for name in ROW_ARRAYS:
        ref = reference[name]
        new = candidate[name]
        if ref.shape != new.shape or ref.dtype != new.dtype:
            diff = compare_arrays(name, ref, new)
            if diff is not None:
                diffs.append(diff)
    if diffs:
        return diffs

    ref_x = reference["X_raw_sorted"]
    new_x = candidate["X_raw_sorted"]
    ref_y_task = reference["y_task_sorted"].reshape(-1)
    new_y_task = candidate["y_task_sorted"].reshape(-1)
    ref_y_oracle = reference["y_oracle_sorted"].reshape(-1)
    new_y_oracle = candidate["y_oracle_sorted"].reshape(-1)
    n_rows = len(ref_x)
    if not all(
        len(values) == n_rows
        for values in (new_x, ref_y_task, new_y_task, ref_y_oracle, new_y_oracle)
    ):
        return [{"kind": "row_count", "reference": n_rows, "candidate": len(new_x)}]

    candidate_rows: dict[tuple[bytes, bytes, bytes], deque[int]] = defaultdict(deque)
    for index in range(n_rows):
        candidate_rows[_row_key(new_x[index], new_y_task[index], new_y_oracle[index])].append(index)

    alignment: list[int] = []
    for index in range(n_rows):
        key = _row_key(ref_x[index], ref_y_task[index], ref_y_oracle[index])
        if not candidate_rows[key]:
            return [{"kind": "row_multiset", "reference_index": index}]
        alignment.append(candidate_rows[key].popleft())
    if any(indices for indices in candidate_rows.values()):
        return [{"kind": "row_multiset", "candidate_has_unmatched_rows": True}]

    candidate_order = np.asarray(alignment, dtype=np.int64)
    for name in NORMALIZED_ARRAYS:
        ref = reference[name]
        new = candidate[name]
        if ref.shape != new.shape or ref.dtype != new.dtype:
            diff = compare_arrays(name, ref, new)
        elif ref.size == 0:
            diff = compare_arrays(name, ref, new)
        elif ref.shape[0] != n_rows:
            diff = {
                "array": name,
                "kind": "row_count",
                "reference": int(ref.shape[0]),
                "candidate": int(new.shape[0]),
            }
        else:
            diff = compare_arrays(
                name,
                ref,
                new[candidate_order],
                atol=atol,
                rtol=rtol,
            )
        if diff is not None:
            diffs.append(diff)
    return diffs


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two DiBO task bundles.")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--mode", choices=("semantic", "exact"), default="semantic")
    parser.add_argument("--atol", type=float, default=5e-4)
    parser.add_argument("--rtol", type=float, default=0.0)
    parser.add_argument("--diff_json", default=None)
    args = parser.parse_args()

    reference_path = Path(args.reference)
    candidate_path = Path(args.candidate)
    with (
        np.load(reference_path, allow_pickle=False) as reference,
        np.load(candidate_path, allow_pickle=False) as candidate,
    ):
        if args.mode == "exact":
            diffs = exact_diffs(reference, candidate)
        else:
            diffs = semantic_diffs(
                reference,
                candidate,
                atol=float(args.atol),
                rtol=float(args.rtol),
            )

    print(f"reference={reference_path}")
    print(f"candidate={candidate_path}")
    print(f"mode={args.mode}")
    print(f"diffs={len(diffs)}")
    print(f"match={str(len(diffs) == 0).lower()}")

    if args.diff_json:
        out = Path(args.diff_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(diffs, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"wrote_diff_json={out}")

    if diffs:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
