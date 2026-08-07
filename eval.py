from __future__ import annotations

import argparse
from functools import lru_cache
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm

import src.compat.compat_patches  # noqa: F401
from src.dataset.task_pool_dataset import PoolConfig, TaskPoolDataset
from src.dataset.utils import load_bundle_for_task, load_templates_for_task
from src.model.dllm import DEFAULT_MODEL_ID, load_dibo_pretrained, load_model_and_tokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEFAULT_MASK_TOKEN_ID = 126336

DNA_BASE2INT = {"A": 0, "C": 1, "G": 2, "T": 3}
SUPPORTED_TASKS = (
    "TFBind8-Exact-v0",
    "TFBind10-Exact-v0",
    "AntMorphology-Exact-v0",
    "DKittyMorphology-Exact-v0",
)
MAX_EVAL_SEED = 2**32 - 1


@dataclass
class PromptSpec:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    mask_positions: torch.Tensor
    prompt_text: str
    task_name: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_eval_seed(value: str) -> int:
    seed = int(value)
    if not 0 <= seed <= MAX_EVAL_SEED:
        raise argparse.ArgumentTypeError(
            f"evaluation seeds must be between 0 and {MAX_EVAL_SEED}"
        )
    return seed


def resolve_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    return (Path(__file__).resolve().parent / p).resolve()


def load_y_range(task_name: str, path: str | Path) -> Tuple[float, float]:
    p = resolve_path(path)
    if not p.exists():
        raise FileNotFoundError(f"Normalization range file not found: {p}")
    ranges = json.loads(p.read_text(encoding="utf-8"))
    if task_name not in ranges:
        raise KeyError(f"No normalization range for task: {task_name}")
    y_min, y_max = ranges[task_name]
    return float(y_min), float(y_max)


def dna_str_to_int_array(seq: str) -> np.ndarray:
    return np.array([DNA_BASE2INT[c] for c in seq], dtype=np.int64)


def serialize_design(design: Union[str, np.ndarray]) -> Union[str, List[float]]:
    if isinstance(design, str):
        return design
    return np.asarray(design, dtype=np.float32).astype(float).tolist()


def extract_design_text(ds: TaskPoolDataset, text: str) -> Optional[str]:
    start = text.rfind(ds._design_start_token)
    if start < 0:
        return None
    start += len(ds._design_start_token)
    return text[start : len(text) - len(ds._design_end_token)].strip()


def extract_dna_design(ds: TaskPoolDataset, text: str, expected_len: int) -> Optional[str]:
    inner = extract_design_text(ds, text)
    if inner is None:
        return None
    seq = "".join(c for c in inner if c in DNA_BASE2INT)
    return seq if len(seq) == expected_len else None


def extract_float_design(ds: TaskPoolDataset, text: str, expected_dim: int) -> Optional[np.ndarray]:
    inner = extract_design_text(ds, text)
    if inner is None:
        return None
    cleaned = inner.replace(",", " ").replace("\n", " ").replace("\t", " ")
    tokens = [token for token in cleaned[1:-1].split(" ") if token]
    values: List[float] = []
    for token in tokens:
        try:
            values.append(float(token))
        except Exception:
            continue
    if len(values) != expected_dim:
        return None
    design = np.asarray(values, dtype=np.float32)
    return design if np.isfinite(design).all() else None


def build_dataset_prompt_spec(ds: TaskPoolDataset, tokenizer: Any) -> PromptSpec:
    item = ds._sample_one()
    input_ids = item["input_ids"].clone()
    attention_mask = item["attention_mask"].clone()
    start_idx = int(item["response_start_idx"].item())
    end_idx = int(item["response_end_idx"].item())

    mask_l = start_idx + 1
    mask_r = end_idx - 1
    if mask_l > mask_r:
        raise ValueError(f"Invalid response span: start={start_idx}, end={end_idx}")

    mask_token_id = tokenizer.mask_token_id or DEFAULT_MASK_TOKEN_ID
    mask_positions = torch.arange(mask_l, mask_r + 1, dtype=torch.long)
    input_ids[mask_positions] = mask_token_id

    return PromptSpec(
        input_ids=input_ids,
        attention_mask=attention_mask,
        mask_positions=mask_positions,
        prompt_text=str(item.get("prompt_text", "")),
        task_name=str(item.get("task_name", ds.task_name)),
    )


def build_random_prompt_spec(
    args, ds: TaskPoolDataset, tokenizer: Any, rng: np.random.Generator
) -> PromptSpec:
    prompt_indices = rng.choice(ds.n_pool, size=ds.n_few_shot, replace=False)
    response_index = int(rng.choice(ds.n_pool))
    template_id = int(rng.choice(ds._template_ids))

    prompt_designs = [ds.pool_designs_raw[int(i)] for i in prompt_indices]
    prompt_labels = [float(ds.pool_labels[int(i)]) for i in prompt_indices]
    response_design = ds.pool_designs_raw[response_index]

    pairs_text = "\n".join(
        ds._render_pair_block(d, y) for d, y in zip(prompt_designs, prompt_labels)
    )
    prompt_text = ds.templates[template_id].format(pairs=pairs_text)
    response_inner = ds._to_list_token_render(response_design)
    full_text = (
        prompt_text + f"Response:\n{ds._design_start_token}{response_inner}{ds._design_end_token}"
    )

    enc = tokenizer(full_text, return_tensors="pt")
    input_ids = enc["input_ids"].squeeze(0)
    attention_mask = enc["attention_mask"].squeeze(0)
    ids = input_ids.tolist()
    start_idx = max(i for i, token in enumerate(ids) if token == ds._design_start_id)
    if ds.special_token_type == "natural":
        start_idx += 2
    end_idx = int(attention_mask.sum().item()) - 1

    if args.special_token_type == "natural":
        mask_l = start_idx + 1
        mask_r = end_idx - 1
    else:
        mask_l = start_idx
        mask_r = end_idx
    if mask_l > mask_r:
        raise ValueError(f"Invalid random-prompt span: start={start_idx}, end={end_idx}")

    mask_positions = torch.arange(mask_l, mask_r + 1, dtype=torch.long)
    input_ids[mask_positions] = DEFAULT_MASK_TOKEN_ID
    return PromptSpec(
        input_ids=input_ids,
        attention_mask=attention_mask,
        mask_positions=mask_positions,
        prompt_text=prompt_text,
        task_name=ds.task_name,
    )


@torch.no_grad()
def greedy_fill(model: Any, tokenizer: Any, spec: PromptSpec, device: str) -> str:
    input_ids = spec.input_ids.unsqueeze(0).to(device)
    attention_mask = spec.attention_mask.unsqueeze(0).to(device)
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits[0, spec.mask_positions, :]
    pred = logits.argmax(dim=-1)
    filled = spec.input_ids.clone()
    filled[spec.mask_positions] = pred.cpu()
    return tokenizer.decode(filled.tolist(), skip_special_tokens=False)


def build_eval_dataset(args, task_name: str, tokenizer: Any, seed: int) -> TaskPoolDataset:
    train_templates, valid_templates = load_templates_for_task(
        task_name, prompts_dir=resolve_path(args.prompts_dir)
    )
    bundle = load_bundle_for_task(task_name, bundle_dir=resolve_path(args.bundle_dir))
    cfg = PoolConfig(
        split="valid",
        mode=str(args.dataset_mode),
        use_oracle=True,
        special_token_type=str(args.special_token_type),
        ablation_use_random_neighbors=str(args.ablation_use_random_neighbors),
        ablation_use_high_or_low_pool=str(args.ablation_use_high_or_low_pool),
        n_pool=int(args.n_pool),
        n_few_shot=int(args.n_few_shot),
        K_pool=int(args.k_pool),
        ratio=float(args.ratio),
        seed=int(seed),
        num_template=tuple(args.num_template),
    )
    return TaskPoolDataset(
        tokenizer=tokenizer,
        templates=list(train_templates) + list(valid_templates),
        task_bundle=bundle,
        cfg=cfg,
    )


def generate_candidates(
    args, model: Any, tokenizer: Any, task_name: str, ds: TaskPoolDataset, seed: int
) -> List[Union[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    device = str(next(model.parameters()).device)
    seen = set()
    designs: List[Union[str, np.ndarray]] = []
    attempts = 0
    invalid = 0
    expected_dim = int(ds.pool_designs_raw.shape[1])

    pbar = tqdm(total=int(args.num_candidates), desc=f"sampling[{task_name}]", leave=False)
    while len(designs) < int(args.num_candidates) and attempts < int(args.max_attempts):
        attempts += 1
        if args.prompt_source == "dataset":
            spec = build_dataset_prompt_spec(ds, tokenizer)
        else:
            spec = build_random_prompt_spec(args, ds, tokenizer, rng)

        filled_text = greedy_fill(model, tokenizer, spec, device)
        if task_name.startswith("TFBind"):
            design = extract_dna_design(ds, filled_text, expected_len=expected_dim)
        else:
            design = extract_float_design(ds, filled_text, expected_dim=expected_dim)

        if design is None:
            invalid += 1
            continue

        key = design if isinstance(design, str) else np.asarray(design, dtype=np.float32).tobytes()
        if key in seen:
            continue
        seen.add(key)
        designs.append(design)
        pbar.update(1)

    pbar.close()
    print(
        f"[Eval] {task_name}: generated {len(designs)} candidates in {attempts} attempts; invalid outputs={invalid}."
    )
    if len(designs) != int(args.num_candidates):
        raise RuntimeError(
            f"{task_name} produced {len(designs)} unique valid candidates, "
            f"but the evaluation requires {args.num_candidates} after "
            f"{attempts} attempts. "
            "Increase --max_attempts or inspect the checkpoint outputs."
        )
    return designs


def _base4_keys(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.int64)
    if x.ndim != 2 or np.any((x < 0) | (x > 3)):
        raise ValueError("TFBind10 designs must have shape [n, 10] with values in [0, 3].")
    powers = 4 ** np.arange(x.shape[1] - 1, -1, -1, dtype=np.int64)
    return x @ powers


@lru_cache(maxsize=2)
def _load_tf10_lookup(path: str) -> Tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        x = np.asarray(data["x"], dtype=np.int64)
        y = np.asarray(data["y"]).ravel()
    keys = _base4_keys(x)
    order = np.argsort(keys)
    keys = keys[order]
    y = y[order]
    if len(keys) != 4**10 or np.any(np.diff(keys) == 0):
        raise ValueError(
            "TFBind10 lookup must contain each length-10 base-4 sequence exactly once."
        )
    return keys, y


def tf10_predict(x_query: np.ndarray, lookup_path: str | Path) -> np.ndarray:
    path = str(resolve_path(lookup_path))
    keys, y = _load_tf10_lookup(path)
    query_keys = _base4_keys(x_query)
    indices = np.searchsorted(keys, query_keys)
    valid = (indices < len(keys)) & (keys[np.minimum(indices, len(keys) - 1)] == query_keys)
    if not np.all(valid):
        raise KeyError("One or more TFBind10 designs are absent from the lookup.")
    return np.asarray(y[indices], dtype=np.float32)


def load_design_bench_task(task_name: str):
    # DeepChem 2.8's SmilesTokenizer assigns to `self.vocab`, but newer
    # Transformers exposes `vocab` as a read-only property. Design-Bench's
    # top-level registry constructs ChEMBL feature extractors at import time,
    # so patch the setter before importing design_bench.
    try:
        from deepchem.feat.smiles_tokenizer import SmilesTokenizer

        vocab_attr = getattr(SmilesTokenizer, "vocab", None)
        if isinstance(vocab_attr, property) and vocab_attr.fset is None:

            def _get_vocab(self):
                return self.__dict__.get("_vocab")

            def _set_vocab(self, value):
                self.__dict__["_vocab"] = value

            SmilesTokenizer.vocab = property(_get_vocab, _set_vocab)
    except Exception as exc:
        print(f"[Eval] DeepChem SmilesTokenizer compat patch skipped: {exc}")

    import design_bench

    return design_bench.make(task_name)


def evaluate_designs(
    args,
    task_name: str,
    designs: Sequence[Union[str, np.ndarray]],
    *,
    return_details: bool = False,
) -> Union[Dict[str, float], Tuple[Dict[str, float], List[float], Optional[List[float]]]]:
    if len(designs) == 0:
        keys = [
            "n",
            "raw_max",
            "raw_mean",
            "raw_median",
            "raw_top5_avg",
            "raw_top10_avg",
            "raw_top20_avg",
        ]
        if not args.no_normalize:
            keys += [
                "norm_max",
                "norm_mean",
                "norm_median",
                "norm_top5_avg",
                "norm_top10_avg",
                "norm_top20_avg",
            ]
        stats = {key: float("nan") for key in keys} | {"n": 0.0}
        if return_details:
            return stats, [], None
        return stats

    if task_name.startswith("TFBind"):
        x_query = np.stack([dna_str_to_int_array(str(s)) for s in designs], axis=0)
    else:
        x_query = np.stack([np.asarray(d, dtype=np.float32) for d in designs], axis=0)

    if task_name == "TFBind10-Exact-v0":
        y_raw = tf10_predict(x_query, args.tf10_lookup)
    else:
        task_obj = load_design_bench_task(task_name)
        y_raw = task_obj.predict(x_query).astype(np.float32).ravel()

    stats: Dict[str, float] = {
        "n": float(len(y_raw)),
        "raw_max": float(y_raw.max()),
        "raw_mean": float(y_raw.mean()),
        "raw_median": float(np.median(y_raw)),
    }
    order = np.argsort(y_raw)
    for topk in (5, 10, 20):
        idx = order[-min(topk, len(y_raw)) :]
        stats[f"raw_top{topk}_avg"] = float(np.mean(y_raw[idx]))

    y_norm = None
    if not args.no_normalize:
        y_min, y_max = load_y_range(task_name, args.normalization_ranges)
        y_norm = (y_raw - y_min) / (y_max - y_min)
        stats.update(
            {
                "norm_max": float(y_norm.max()),
                "norm_mean": float(y_norm.mean()),
                "norm_median": float(np.median(y_norm)),
            }
        )
        for topk in (5, 10, 20):
            idx = order[-min(topk, len(y_raw)) :]
            stats[f"norm_top{topk}_avg"] = float(np.mean(y_norm[idx]))

    best_idx = int(np.argmax(y_raw))
    print(f"[Eval] {task_name}: best_raw={float(y_raw[best_idx]):.6f}, design={designs[best_idx]}")
    if return_details:
        raw_scores = np.asarray(y_raw, dtype=np.float32).astype(float).tolist()
        norm_scores = (
            None if y_norm is None else np.asarray(y_norm, dtype=np.float32).astype(float).tolist()
        )
        return stats, raw_scores, norm_scores
    return stats


def checkpoint_paths(args) -> List[Optional[Path]]:
    if args.checkpoint_path:
        return [resolve_path(args.checkpoint_path)]
    if args.checkpoint_dir:
        ckpt_dir = resolve_path(args.checkpoint_dir)
        return [
            ckpt_dir / str(args.checkpoint_name_template).format(step=step) for step in args.steps
        ]
    return [None]


def resolve_model_source(args) -> str:
    """Select one evaluation loading path without changing legacy checkpoint use."""
    if args.checkpoint_path and args.checkpoint_dir:
        raise ValueError("Specify either --checkpoint_path or --checkpoint_dir, not both.")
    has_original_checkpoint = bool(args.checkpoint_path or args.checkpoint_dir)
    if has_original_checkpoint:
        return "original_checkpoint"
    if str(args.model_name_or_path) != DEFAULT_MODEL_ID:
        return "pretrained_export"
    if args.model_revision is not None:
        raise ValueError(
            "--model_revision is for --model_name_or_path pointing to a packaged DiBO export."
        )
    return "base"


def load_checkpoint_if_needed(model: Any, path: Optional[Path], device: str) -> str:
    if path is None:
        print("[Eval] Using base model weights.")
        return "base"
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    print(f"[Eval] Loading checkpoint: {path}")
    ckpt = torch.load(str(path), map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    del ckpt
    return str(path)


def effective_eval_seed(seed: int, transform: str) -> int:
    if transform == "square":
        return int(seed) * int(seed)
    return int(seed)


def infer_step_from_checkpoint_label(label: str) -> Optional[int]:
    if label == "base":
        return None
    name = Path(label).name
    if not name.startswith("optim_step="):
        return None
    step_text = name[len("optim_step=") :].split("_", 1)[0].split(".", 1)[0]
    try:
        return int(step_text)
    except ValueError:
        return None


def summarize_normalized_max(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[float]] = {}
    for row in rows:
        value = row.get("norm_max")
        if value is None or not np.isfinite(value):
            continue
        key = (str(row["checkpoint"]), str(row["task"]))
        grouped.setdefault(key, []).append(float(value))

    summaries: List[Dict[str, Any]] = []
    for (checkpoint, task), values in grouped.items():
        scores = np.asarray(values, dtype=np.float64)
        summaries.append(
            {
                "checkpoint": checkpoint,
                "task": task,
                "n_seeds": int(len(scores)),
                "norm_max_mean": float(scores.mean()),
                "norm_max_standard_deviation": float(scores.std(ddof=0)),
            }
        )
    return summaries


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate candidate designs generated by a DiBO checkpoint."
    )
    parser.add_argument("--tasks", nargs="+", choices=SUPPORTED_TASKS, default=["TFBind8-Exact-v0"])
    parser.add_argument(
        "--seeds",
        type=parse_eval_seed,
        nargs="+",
        required=True,
        metavar="SEED",
        help="One or more caller-selected evaluation seeds.",
    )
    parser.add_argument("--num_candidates", type=int, default=128)
    parser.add_argument("--max_attempts", type=int, default=1000)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument(
        "--checkpoint_name_template", type=str, default="optim_step={step}_final.pt"
    )
    parser.add_argument("--steps", type=int, nargs="+", default=[128])
    parser.add_argument("--model_name_or_path", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument(
        "--model_revision",
        type=str,
        default=None,
        help=(
            "revision for a standard DiBO Transformers export passed through "
            "--model_name_or_path"
        ),
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--bundle_dir", type=str, default="data/task_bundles")
    parser.add_argument("--prompts_dir", type=str, default="data/prompts")
    parser.add_argument(
        "--normalization_ranges", type=str, default="data/normalization_ranges.json"
    )
    parser.add_argument("--tf10_lookup", type=str, default="data/raw/TFBind10-Exact-v0.npz")
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--save_details", action="store_true")
    parser.add_argument("--no_normalize", action="store_true")

    parser.add_argument("--prompt_source", choices=["dataset", "random"], default="random")
    parser.add_argument(
        "--seed_transform",
        choices=["square", "identity"],
        default="square",
        help=(
            "Transform applied to each caller-selected seed before prompt and "
            "candidate sampling. The effective seed is recorded in the output."
        ),
    )
    parser.add_argument("--dataset_mode", choices=["da", "sft", "rl"], default="da")
    parser.add_argument("--special_token_type", choices=["special", "natural"], default="special")
    parser.add_argument(
        "--ablation_use_random_neighbors", choices=["d1-d2", "random"], default="d1-d2"
    )
    parser.add_argument(
        "--ablation_use_high_or_low_pool",
        choices=["evenly", "high", "low", "random"],
        default="evenly",
    )
    parser.add_argument("--n_pool", type=int, default=500)
    parser.add_argument("--n_few_shot", type=int, default=7)
    parser.add_argument("--k_pool", type=int, default=50)
    parser.add_argument("--ratio", type=float, default=0.8)
    parser.add_argument("--num_template", type=int, nargs=2, default=[8, 2])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print("[Eval] Arguments:", args)

    source = resolve_model_source(args)
    if source == "pretrained_export":
        model, tokenizer = load_dibo_pretrained(
            str(args.model_name_or_path),
            revision=args.model_revision,
            device=str(args.device),
        )
        checkpoint_labels = [
            f"{args.model_name_or_path}@{args.model_revision or 'main'}"
        ]
    else:
        model, tokenizer = load_model_and_tokenizer(
            model_name_or_path=str(args.model_name_or_path),
            device=str(args.device),
        )
        checkpoint_labels = None
    model.eval()

    rows: List[Dict[str, Any]] = []
    for index, ckpt_path in enumerate(checkpoint_paths(args)):
        if source == "pretrained_export":
            ckpt_label = checkpoint_labels[index]
            print(f"[Eval] Using standard Transformers export: {ckpt_label}")
        else:
            ckpt_label = load_checkpoint_if_needed(model, ckpt_path, str(args.device))
        ckpt_step = infer_step_from_checkpoint_label(ckpt_label)
        for task_name in args.tasks:
            for seed in args.seeds:
                set_seed(seed)
                eval_seed = effective_eval_seed(int(seed), str(args.seed_transform))
                ds = build_eval_dataset(args, task_name, tokenizer, seed=eval_seed)
                designs = generate_candidates(args, model, tokenizer, task_name, ds, seed=eval_seed)
                eval_result = evaluate_designs(
                    args, task_name, designs, return_details=bool(args.save_details)
                )
                if args.save_details:
                    stats, raw_scores, norm_scores = eval_result
                else:
                    stats = eval_result
                    raw_scores = None
                    norm_scores = None
                row = {
                    "checkpoint": ckpt_label,
                    "task": task_name,
                    "seed": int(seed),
                    "effective_seed": int(eval_seed),
                    **stats,
                }
                if ckpt_step is not None:
                    row["step"] = int(ckpt_step)
                if args.save_details:
                    row["designs"] = [serialize_design(d) for d in designs]
                    row["raw_scores"] = raw_scores
                    row["norm_scores"] = norm_scores
                rows.append(row)
                compact_row = {
                    k: v
                    for k, v in row.items()
                    if k not in {"designs", "raw_scores", "norm_scores"}
                }
                print("[Eval] Result:", compact_row)

    print(
        "\ncheckpoint\ttask\tseed\tn\traw_max\traw_top5_avg\traw_top10_avg\traw_top20_avg\tnorm_max"
    )
    for row in rows:
        print(
            f"{row['checkpoint']}\t{row['task']}\t{row['seed']}\t{int(row['n'])}\t"
            f"{row['raw_max']:.4f}\t{row['raw_top5_avg']:.4f}\t"
            f"{row['raw_top10_avg']:.4f}\t{row['raw_top20_avg']:.4f}\t"
            f"{row.get('norm_max', float('nan')):.4f}"
        )

    summaries = summarize_normalized_max(rows)
    if summaries:
        print("\ncheckpoint\ttask\tn_seeds\tnorm_max_mean\t" "norm_max_population_std")
        for summary in summaries:
            print(
                f"{summary['checkpoint']}\t{summary['task']}\t"
                f"{summary['n_seeds']}\t{summary['norm_max_mean']:.6f}\t"
                f"{summary['norm_max_standard_deviation']:.6f}"
            )

    if args.output_json:
        out_path = resolve_path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[Eval] Wrote results: {out_path}")


if __name__ == "__main__":
    main()
