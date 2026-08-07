from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf


CLI_OVERRIDE_SPECS = (
    ("seed", "train_cfg", "basic.seed", int),
    ("ckpt_path", "train_cfg", "ckpt_path", str),
    ("mode", "train_cfg", "train.mode", str),
    ("extra_name", "train_cfg", "extra_name", str),
    ("special_token_type", "train_cfg", "train.special_token_type", str),
    ("ablation_use_random_neighbors", "train_cfg", "train.ablation_use_random_neighbors", str),
    ("ablation_use_high_or_low_pool", "train_cfg", "train.ablation_use_high_or_low_pool", str),
    ("max_opt_steps", "train_cfg", "train.max_opt_steps", int),
    ("log_every_steps", "train_cfg", "train.log_every_steps", int),
    ("eval_every_steps", "train_cfg", "train.eval_every_steps", int),
    ("save_every_steps", "train_cfg", "train.save_every_steps", int),
    ("batch_size", "train_cfg", "train.batch_size", int),
    ("grad_accum_steps", "train_cfg", "train.grad_accum_steps", int),
    ("lr", "train_cfg", "train.lr", float),
    ("warmup_steps", "train_cfg", "train.warmup_steps", int),
    ("warmup_ratio", "train_cfg", "train.warmup_ratio", float),
    ("lr_schedule", "train_cfg", "train.lr_schedule", str),
    ("max_grad_norm", "train_cfg", "train.max_grad_norm", float),
    ("optimizer_name", "train_cfg", "train.optimizer_name", str),
    ("gradient_checkpointing", "train_cfg", "train.gradient_checkpointing", bool),
    ("wandb_mode", "train_cfg", "wandb.wandb_mode", str),
    ("project", "train_cfg", "wandb.project", str),
    ("reward_center", "train_cfg", "train.reward_center", bool),
    ("use_wandb", "train_cfg", "wandb.use_wandb", bool),
    ("n_samples_valid", "task_cfg", "n_samples_valid", int),
    ("use_oracle", "task_cfg", "use_oracle", bool),
    ("n_pool", "task_cfg", "n_pool", int),
    ("n_few_shot", "task_cfg", "n_few_shot", int),
    ("k_pool", "task_cfg", "k_pool", int),
    ("ratio", "task_cfg", "ratio", float),
    ("num_template", "task_cfg", "num_template", list),
)


MIX_OVERRIDE_SPECS = (
    ("mix_mode", "mix.mode", str),
    ("mix_weights", "mix.weights", None),
    ("mix_fixed_order", "mix.fixed_order", None),
    ("mix_rr_shuffle_tasks", "mix.rr_shuffle_tasks", bool),
)


def load_cfg(path: str, verbose: bool = False) -> DictConfig:
    cfg = OmegaConf.load(path)
    if verbose:
        print("All Arguments:")
        print(cfg)
    return cfg


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def load_reward_stats_json(path: str, task_name: str) -> Dict[str, Any]:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(str(p))
    obj = json.loads(p.read_text(encoding="utf-8"))
    tasks = obj.get("tasks", obj)
    task_obj = tasks.get(task_name, None)
    if task_obj is None:
        raise ValueError(f"Task name '{task_name}' not found in reward stats JSON at {str(p)}")

    return task_obj


def infinite_loader(dataloader):
    while True:
        for batch in dataloader:
            yield batch


def _parse_named_weights(items: List[str]) -> Dict[str, float]:
    weights: Dict[str, float] = {}
    for item in items:
        if "=" not in str(item):
            raise ValueError(f"mix weight must use task_name=prob format, got: {item}")
        name, value = str(item).split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"empty task name in mix weight: {item}")
        weights[name] = float(value)
    return weights


def _set_cfg_value(cfg: DictConfig, dotted_path: str, value: Any) -> None:
    target = cfg
    parts = dotted_path.split(".")
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value


def _has_any_arg(args, names: List[str]) -> bool:
    return any(getattr(args, name, None) is not None for name in names)


def apply_overrides(train_cfg, task_cfg, args) -> None:
    """
    Apply CLI arguments to loaded OmegaConf configs.

    Only explicitly provided user arguments override config-file values.
    """

    roots = {"train_cfg": train_cfg, "task_cfg": task_cfg}
    for arg_name, root_name, cfg_path, caster in CLI_OVERRIDE_SPECS:
        raw = getattr(args, arg_name, None)
        if raw is None:
            continue
        _set_cfg_value(roots[root_name], cfg_path, caster(raw))

    if getattr(args, "use", None) is not None:
        chosen = [str(x) for x in args.use]
        chosen_set = set(chosen)
        unknown = chosen_set - set(task_cfg.use.keys())
        if unknown:
            raise ValueError(f"Unsupported task names: {sorted(unknown)}")

        for t in list(task_cfg.use.keys()):
            task_cfg.use[t] = t in chosen_set

    mix_arg_names = [spec[0] for spec in MIX_OVERRIDE_SPECS]
    if _has_any_arg(args, mix_arg_names) and "mix" not in task_cfg:
        task_cfg.mix = {}

    for arg_name, cfg_path, caster in MIX_OVERRIDE_SPECS:
        raw = getattr(args, arg_name, None)
        if raw is None:
            continue
        if arg_name == "mix_weights":
            value = _parse_named_weights(raw)
        elif arg_name == "mix_fixed_order":
            value = [str(x) for x in raw]
        else:
            value = caster(raw)
        _set_cfg_value(task_cfg, cfg_path, value)
