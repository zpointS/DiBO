# Offline RL reward statistics.
#
# Outputs include:
#   1) reward_raw = resp_label - max(prompt_labels)
#   2) adv_r_over_std = reward_raw / std(reward_raw)
#   3) adv_centered_over_std = (reward_raw - mean(reward_raw)) / std(reward_raw)
#
# Training can clip directly by percentiles from these advantage values.

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from src.dataset.task_bundle import TaskBundle
from src.dataset.task_pool_dataset import PoolConfig, TaskPoolDataset


Array = np.ndarray


def load_task_bundle_npz(path: str) -> TaskBundle:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(str(p))
    with np.load(str(p), allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    if "task_name" not in d:
        d["task_name"] = np.array(p.stem)
    if isinstance(d["task_name"], np.ndarray):
        d["task_name"] = str(d["task_name"].tolist())
    return TaskBundle(**d)


DEFAULT_QS: List[float] = [
    # Denser lower tail.
    0.1,
    0.5,
    1,
    2,
    3,
    5,
    10,
    20,
    # Median.
    50,
    # Upper tail.
    80,
    90,
    95,
    97,
    98,
    99,
    99.5,
    99.9,
]


def _percentiles(arr: Array, qs: List[float]) -> Dict[str, float]:
    arr = np.asarray(arr, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return {}
    vals = np.percentile(arr, qs).tolist()
    out: Dict[str, float] = {}
    for q, v in zip(qs, vals):
        if float(q).is_integer():
            key = f"p{int(q)}"
        else:
            key = f"p{str(q).replace('.', '_')}"
        out[key] = float(v)
    return out


def _summary(arr: Array) -> Dict[str, float]:
    arr = np.asarray(arr, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return {}
    return {
        "count": float(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _stats_with_qs(arr: Array, qs: List[float]) -> Dict[str, Any]:
    s = _summary(arr)
    s["percentiles"] = _percentiles(arr, qs)
    return s


def _winsorize(arr: Array, lo_q: float, hi_q: float) -> Array:
    arr = np.asarray(arr, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return arr
    lo, hi = np.percentile(arr, [lo_q, hi_q])
    return np.clip(arr, lo, hi)


def _clip_diag(arr: Array, lo_q: float, hi_q: float) -> Dict[str, float]:
    """
    Compute diagnostics for percentile clipping.
    """
    arr = np.asarray(arr, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return {
            "lo_q": lo_q,
            "hi_q": hi_q,
            "lo": 0.0,
            "hi": 0.0,
            "frac_clipped": 0.0,
            "mean_abs_delta": 0.0,
            "max_abs_delta": 0.0,
        }

    lo, hi = np.percentile(arr, [lo_q, hi_q])
    clipped = np.clip(arr, lo, hi)
    delta = np.abs(clipped - arr)
    frac = float(np.mean((arr < lo) | (arr > hi)))
    return {
        "lo_q": float(lo_q),
        "hi_q": float(hi_q),
        "lo": float(lo),
        "hi": float(hi),
        "frac_clipped": frac,
        "mean_abs_delta": float(delta.mean()),
        "max_abs_delta": float(delta.max()),
    }


def build_task_dataset_reward_only(
    task_name: str,
    bundle_path: str,
    *,
    split: str,
    n_pool: int,
    n_few_shot: int,
    k_pool: int,
    ratio: float,
    seed: int,
    num_template: Tuple[int, int],
    use_oracle: bool = True,
    ablation_use_random_neighbors: str = "d1-d2",
    ablation_use_high_or_low_pool: str = "evenly",
    lower_ratio: float = 0.7,
    upper_ratio: float = 0.95,
) -> TaskPoolDataset:
    """
    Build a TaskPoolDataset used only for reward sampling.

    This path never tokenizes prompts, so dummy tokenizer/templates are enough.
    """
    bundle = load_task_bundle_npz(bundle_path)
    if str(bundle.task_name) != str(task_name):
        raise ValueError(f"bundle.task_name={bundle.task_name} != task_name={task_name}")

    class _DummyTok:
        def convert_tokens_to_ids(self, x):
            return 0

        def __call__(self, *args, **kwargs):
            raise RuntimeError("Dummy tokenizer should not be called in reward-only stats script.")

    dummy_tok = _DummyTok()
    dummy_templates = ["{pairs}"] * sum(num_template)

    cfg = PoolConfig(
        split=split,
        mode="rl",
        use_oracle=use_oracle,
        n_pool=n_pool,
        n_few_shot=n_few_shot,
        K_pool=k_pool,
        ratio=ratio,
        seed=seed,
        num_template=num_template,
        ablation_use_random_neighbors=ablation_use_random_neighbors,
        ablation_use_high_or_low_pool=ablation_use_high_or_low_pool,
        lower_ratio=lower_ratio,
        upper_ratio=upper_ratio,
    )
    return TaskPoolDataset(
        tokenizer=dummy_tok, templates=dummy_templates, task_bundle=bundle, cfg=cfg
    )


def sample_rl_reward_only(ds: TaskPoolDataset, rng: np.random.Generator) -> Dict[str, Any]:
    """
    Reproduce TaskPoolDataset's RL reward sampling while computing only rewards.

    Current training uses the difference after pool-level 0-1 normalization:
      reward_norm = y_resp_norm - y_maxprompt_norm
    """
    if ds.mode != "rl":
        raise ValueError("Dataset mode must be 'rl' for RL reward sampling.")

    candidates = ds.rl_valid_anchors
    if len(candidates) == 0:
        raise RuntimeError(f"[{ds.task_name}] Empty rl_valid_anchors.")

    use_replacement = bool(ds.n_pool <= 20)

    j = int(rng.choice(candidates))
    lower_list = ds.rl_lower[j]
    higher_list = ds.rl_higher[j]

    y = ds.pool_labels.astype(np.float32)

    # Match training: use the candidate-pool min/max for 0-1 normalization.
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    denom = (y_max - y_min) if (y_max > y_min) else 1.0

    want_positive = bool(rng.integers(0, 2))

    if want_positive:
        K_low = min(ds.K_lower_rl, len(lower_list))
        cand = lower_list[:K_low]
        if len(cand) < ds.n_few_shot and not use_replacement:
            raise RuntimeError(
                f"[{ds.task_name}] lower_list too small: {len(cand)} < n_few_shot={ds.n_few_shot}"
            )
        chosen = rng.choice(cand, size=ds.n_few_shot, replace=use_replacement)
        prompt_indices = list(map(int, chosen))
    else:
        K_high = min(ds.K_higher_rl, len(higher_list))
        high_cand = higher_list[:K_high]
        if len(high_cand) < 1:
            raise RuntimeError(f"[{ds.task_name}] higher_list empty for anchor {j}.")

        high_idx = int(rng.choice(high_cand))
        y_h = float(y[high_idx])

        rest_pool: List[int] = list(lower_list)
        for idx2 in higher_list:
            idx2 = int(idx2)
            if idx2 == high_idx:
                continue
            if float(y[idx2]) < y_h:
                rest_pool.append(idx2)

        max_rest = min(len(rest_pool), ds.K_pool)
        rest_cand = rest_pool[:max_rest]

        need = ds.n_few_shot - 1
        if len(rest_cand) < need and not use_replacement:
            raise RuntimeError(
                f"[{ds.task_name}] rest_pool too small: {len(rest_cand)} < need={need}"
            )

        rest = rng.choice(rest_cand, size=need, replace=use_replacement)
        prompt_indices = [high_idx] + list(map(int, rest))
        rng.shuffle(prompt_indices)

    prompt_labels = [float(y[i]) for i in prompt_indices]
    resp_label = float(y[j])
    max_prompt_label = float(max(prompt_labels))

    resp_norm = (resp_label - y_min) / denom
    maxp_norm = (max_prompt_label - y_min) / denom
    reward_norm = resp_norm - maxp_norm

    anchor_rank01 = float(j) / float(max(1, ds.n_pool - 1))

    return {
        "reward_raw": float(reward_norm),
        "resp_label": float(resp_norm),
        "max_prompt_label": float(maxp_norm),
        "want_positive": float(want_positive),
        "anchor_j": float(j),
        "anchor_rank01": float(anchor_rank01),
        "lower_len": float(len(lower_list)),
        "higher_len": float(len(higher_list)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle_dir", type=str, default="data/task_bundles")
    ap.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=[
            "TFBind8-Exact-v0",
            "TFBind10-Exact-v0",
            "AntMorphology-Exact-v0",
            "DKittyMorphology-Exact-v0",
        ],
    )

    ap.add_argument(
        "--few_shot",
        type=int,
        nargs="+",
        default=[7] * 4,
    )
    ap.add_argument("--n_pool", type=int, default=500)
    ap.add_argument("--k_pool", type=int, default=50)
    ap.add_argument("--ratio", type=float, default=0.8)
    ap.add_argument("--lower_ratio", type=float, default=0.7)
    ap.add_argument("--upper_ratio", type=float, default=0.95)

    ap.add_argument("--split", type=str, default="train", choices=["train", "valid"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n_samples_per_task", type=int, default=100000)
    ap.add_argument("--num_train_templates", type=int, default=8)
    ap.add_argument("--num_valid_templates", type=int, default=2)

    ap.add_argument("--qs", nargs="*", type=float, default=DEFAULT_QS)

    ap.add_argument("--clip_lo_q", type=float, default=1.0)
    ap.add_argument("--clip_hi_q", type=float, default=99.0)

    ap.add_argument("--out_json", type=str, default="data/reward_stats/reward_stats.json")

    args = ap.parse_args()

    if len(args.few_shot) != len(args.tasks):
        raise ValueError(
            f"--few_shot must align with --tasks. Got {len(args.few_shot)} vs {len(args.tasks)}"
        )

    qs = [float(x) for x in args.qs]
    clip_lo_q = float(args.clip_lo_q)
    clip_hi_q = float(args.clip_hi_q)

    bundle_dir = Path(args.bundle_dir).expanduser().resolve()
    num_template = (args.num_train_templates, args.num_valid_templates)

    out: Dict[str, Any] = {
        "meta": {
            "split": args.split,
            "mode": "rl",
            "n_pool": args.n_pool,
            "K_pool": args.k_pool,
            "ratio": args.ratio,
            "lower_ratio": args.lower_ratio,
            "upper_ratio": args.upper_ratio,
            "n_samples_per_task": args.n_samples_per_task,
            "num_template": list(num_template),
            "percentiles_qs": qs,
            "adv_clip_lo_q": clip_lo_q,
            "adv_clip_hi_q": clip_hi_q,
        },
        "tasks": {},
    }

    for tname, n_few in zip(args.tasks, args.few_shot):
        bundle_path = bundle_dir / f"{tname}.npz"
        if not bundle_path.exists():
            raise FileNotFoundError(f"Bundle not found: {bundle_path}")

        ds = build_task_dataset_reward_only(
            task_name=tname,
            bundle_path=str(bundle_path),
            split=args.split,
            n_pool=args.n_pool,
            n_few_shot=int(n_few),
            k_pool=args.k_pool,
            ratio=args.ratio,
            seed=args.seed,
            num_template=num_template,
            lower_ratio=args.lower_ratio,
            upper_ratio=args.upper_ratio,
        )

        rewards: List[float] = []
        resp: List[float] = []
        maxp: List[float] = []
        want_pos: List[float] = []
        anchor_rank01: List[float] = []
        lower_len: List[float] = []
        higher_len: List[float] = []

        for i in tqdm(range(int(args.n_samples_per_task)), desc=f"{tname} reward-only stats"):
            # Use a deterministic per-sample seed so offline stats are reproducible.
            rng = np.random.default_rng(int(args.seed) + 1_000_042 * int(i))
            rec = sample_rl_reward_only(ds, rng)
            rewards.append(rec["reward_raw"])
            resp.append(rec["resp_label"])
            maxp.append(rec["max_prompt_label"])
            want_pos.append(rec["want_positive"])
            anchor_rank01.append(rec["anchor_rank01"])
            lower_len.append(rec["lower_len"])
            higher_len.append(rec["higher_len"])

        rewards_arr = np.asarray(rewards, dtype=np.float64)
        resp_arr = np.asarray(resp, dtype=np.float64)
        maxp_arr = np.asarray(maxp, dtype=np.float64)
        want_arr = np.asarray(want_pos, dtype=np.float64)
        rank_arr = np.asarray(anchor_rank01, dtype=np.float64)
        lower_arr = np.asarray(lower_len, dtype=np.float64)
        higher_arr = np.asarray(higher_len, dtype=np.float64)

        eps = 1e-8
        r_mean = float(rewards_arr.mean())
        r_std = float(rewards_arr.std(ddof=0))
        adv_r_over_std = rewards_arr / (r_std + eps)
        adv_centered_over_std = (rewards_arr - r_mean) / (r_std + eps)

        adv_r_over_std_clip = _winsorize(adv_r_over_std, clip_lo_q, clip_hi_q)
        adv_centered_over_std_clip = _winsorize(adv_centered_over_std, clip_lo_q, clip_hi_q)

        task_rec: Dict[str, Any] = {
            "task_name": tname,
            "n_few_shot": int(n_few),
            "reward_raw": _stats_with_qs(rewards_arr, qs),
            "resp_label": _stats_with_qs(resp_arr, qs),
            "max_prompt_label": _stats_with_qs(maxp_arr, qs),
            "want_positive_rate": float(want_arr.mean()),
            "anchor_rank01": _stats_with_qs(rank_arr, qs),
            "lower_len": _stats_with_qs(lower_arr, qs),
            "higher_len": _stats_with_qs(higher_arr, qs),
            "adv_variants": {
                "adv_r_over_std": {
                    "global_reward_mean": r_mean,
                    "global_reward_std": r_std,
                    "stats": _stats_with_qs(adv_r_over_std, qs),
                    "winsorize_diag": _clip_diag(adv_r_over_std, clip_lo_q, clip_hi_q),
                    "stats_winsorized": _stats_with_qs(adv_r_over_std_clip, qs),
                },
                "adv_centered_over_std": {
                    "global_reward_mean": r_mean,
                    "global_reward_std": r_std,
                    "stats": _stats_with_qs(adv_centered_over_std, qs),
                    "winsorize_diag": _clip_diag(adv_centered_over_std, clip_lo_q, clip_hi_q),
                    "stats_winsorized": _stats_with_qs(adv_centered_over_std_clip, qs),
                },
            },
        }

        out["tasks"][tname] = task_rec

        s = task_rec["reward_raw"]
        p = s["percentiles"]
        print(
            f"[{tname}] reward_raw: mean={s['mean']:.6f} std={s['std']:.6f} "
            f"p1={p.get('p1', float('nan')):.6f} p50={p.get('p50', float('nan')):.6f} "
            f"p99={p.get('p99', float('nan')):.6f} max={s['max']:.6f}"
        )
        diag1 = task_rec["adv_variants"]["adv_r_over_std"]["winsorize_diag"]
        diag2 = task_rec["adv_variants"]["adv_centered_over_std"]["winsorize_diag"]
        print(
            f"[{tname}] adv(r/std) winsorize {clip_lo_q}-{clip_hi_q}: "
            f"frac_clipped={diag1['frac_clipped']:.6f} mean|Δ|={diag1['mean_abs_delta']:.6f}"
        )
        print(
            f"[{tname}] adv((r-mean)/std) winsorize {clip_lo_q}-{clip_hi_q}: "
            f"frac_clipped={diag2['frac_clipped']:.6f} mean|Δ|={diag2['mean_abs_delta']:.6f}"
        )

    out_path = Path(args.out_json).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(out, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
