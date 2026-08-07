from __future__ import annotations

import datetime as _dt
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from omegaconf import OmegaConf

from src.dataset.mixed_dataset import build_mixed_dataset
from src.model.dllm import load_model_and_tokenizer
from src.trainer_dllm import DllmTrainer
from src.utils import apply_overrides, load_cfg, seed_everything


STAGE_LRS = {
    "da": 2e-5,
    "sft": 2e-5,
    "rl": 1e-6,
}

DISCRETE_TASKS = {"TFBind8-Exact-v0", "TFBind10-Exact-v0"}
CONTINUOUS_TASKS = {"AntMorphology-Exact-v0", "DKittyMorphology-Exact-v0"}

PAPER_STAGE_LRS = {
    "discrete": {"da": 2e-5, "sft": 2e-5, "rl": 1e-6},
    "continuous": {"da": 1e-5, "sft": 2e-5, "rl": 1e-6},
}

PAPER_STAGE_STEPS = {
    "discrete": {"da": 1024, "sft": 1024, "rl": 128},
    "continuous": {"da": 2048, "sft": 1024, "rl": 128},
}

PAPER_GRAD_ACCUM_STEPS = {
    "discrete": 16,
    "continuous": 8,
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(path: str | Path, *, root: Path) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    return (root / p).resolve()


def selected_tasks(task_cfg) -> List[str]:
    return [str(name) for name, enabled in task_cfg.use.items() if bool(enabled)]


def task_short_name(task_cfg, task_name: str) -> str:
    rec = task_cfg.get(task_name)
    return str(rec.short_name) if rec is not None and "short_name" in rec else task_name


def infer_paper_task_type(tasks: Sequence[str]) -> Optional[str]:
    task_set = set(tasks)
    if task_set and task_set <= DISCRETE_TASKS:
        return "discrete"
    if task_set and task_set <= CONTINUOUS_TASKS:
        return "continuous"
    return None


def clone_cfg(cfg):
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))


def _stage_lr(args, train_cfg, stage: str, paper_task_type: Optional[str]) -> float:
    if getattr(args, "lr", None) is not None:
        return float(train_cfg.train.lr)
    val = getattr(args, f"{stage}_lr", None)
    if val is not None:
        return float(val)
    if paper_task_type in PAPER_STAGE_LRS:
        return float(PAPER_STAGE_LRS[paper_task_type][stage])
    return float(STAGE_LRS[stage])


def _stage_steps(args, train_cfg, stage: str, paper_task_type: Optional[str]) -> int:
    val = getattr(args, f"{stage}_steps", None)
    if val is not None:
        return int(val)
    if getattr(args, "max_opt_steps", None) is not None:
        return int(train_cfg.train.max_opt_steps)
    if paper_task_type in PAPER_STAGE_STEPS:
        return int(PAPER_STAGE_STEPS[paper_task_type][stage])
    return int(train_cfg.train.max_opt_steps)


def _should_save_final(policy: str, stage_index: int, n_stages: int) -> bool:
    if policy == "none":
        return False
    if policy == "last":
        return stage_index == n_stages - 1
    if policy in {"each_stage", "periodic"}:
        return True
    raise ValueError(f"Unknown checkpoint_policy: {policy}")


def _periodic_save_every(policy: str, save_every_steps: int) -> int:
    return int(save_every_steps) if policy == "periodic" else 0


def _load_checkpoint(model, checkpoint_path: str | Path, *, device: str) -> None:
    p = Path(checkpoint_path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"resume checkpoint not found: {p}")
    print(f"[CKPT] Loading initial checkpoint: {p}")
    ckpt = torch.load(str(p), map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    del ckpt


def _init_wandb(train_cfg, task_cfg, stage_dir: Path):
    if not bool(train_cfg.wandb.use_wandb):
        return None

    wandb_dir = resolve_path(train_cfg.wandb.wandb_dir, root=repo_root())
    wandb_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("WANDB_MODE", str(train_cfg.wandb.wandb_mode))
    os.environ.setdefault("WANDB_DIR", str(wandb_dir))
    import wandb

    train_cfg.wandb.name = stage_dir.name
    stage_name = str(train_cfg.train.mode)
    return wandb.init(
        project=str(train_cfg.wandb.project),
        name=str(train_cfg.wandb.name),
        dir=str(wandb_dir),
        mode=str(train_cfg.wandb.wandb_mode),
        group=stage_dir.parent.name,
        job_type=stage_name,
        tags=[stage_name],
        config={
            "train": OmegaConf.to_container(train_cfg, resolve=True),
            "task": OmegaConf.to_container(task_cfg, resolve=True),
        },
    )


def _save_effective_configs(stage_dir: Path, train_cfg, task_cfg) -> None:
    cfg_dir = stage_dir / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(train_cfg, cfg_dir / "train.yaml")
    OmegaConf.save(task_cfg, cfg_dir / "task.yaml")


def prepare_configs(args):
    root = repo_root()
    train_cfg = load_cfg(args.train_cfg_path, verbose=False)
    task_cfg = load_cfg(args.task_cfg_path, verbose=False)
    apply_overrides(train_cfg, task_cfg, args)

    if getattr(args, "device", None):
        train_cfg.basic.device = str(args.device)
    if getattr(args, "model_name_or_path", None):
        train_cfg.model.name_or_path = str(args.model_name_or_path)
    if getattr(args, "output_dir", None):
        train_cfg.path.output_dir = str(args.output_dir)
    elif os.environ.get("DIBO_OUTPUT_DIR"):
        train_cfg.path.output_dir = os.environ["DIBO_OUTPUT_DIR"]
    if getattr(args, "bundle_dir", None):
        train_cfg.path.bundle_dir = str(args.bundle_dir)
        task_cfg.path.bundle_dir = str(args.bundle_dir)
    if getattr(args, "prompts_dir", None):
        train_cfg.path.prompts_dir = str(args.prompts_dir)
        task_cfg.path.prompts_dir = str(args.prompts_dir)
    if getattr(args, "reward_stats_json", None):
        train_cfg.path.reward_stats_json = str(args.reward_stats_json)
    if getattr(args, "wandb_dir", None):
        train_cfg.wandb.wandb_dir = str(args.wandb_dir)
    if getattr(args, "use_wandb", None) is not None:
        train_cfg.wandb.use_wandb = bool(args.use_wandb)

    train_cfg.path.bundle_dir = str(resolve_path(train_cfg.path.bundle_dir, root=root))
    train_cfg.path.prompts_dir = str(resolve_path(train_cfg.path.prompts_dir, root=root))
    train_cfg.path.output_dir = str(resolve_path(train_cfg.path.output_dir, root=root))
    train_cfg.path.reward_stats_json = str(
        resolve_path(train_cfg.path.reward_stats_json, root=root)
    )
    train_cfg.wandb.wandb_dir = str(resolve_path(train_cfg.wandb.wandb_dir, root=root))
    task_cfg.path.bundle_dir = train_cfg.path.bundle_dir
    task_cfg.path.prompts_dir = train_cfg.path.prompts_dir

    tasks = selected_tasks(task_cfg)
    if not tasks:
        raise ValueError("No tasks selected. Pass --use <task-name>.")
    return train_cfg, task_cfg


def run_pipeline(args) -> Dict[str, Any]:
    train_cfg, task_cfg = prepare_configs(args)
    stages: List[str] = [str(s) for s in args.stages]
    for stage in stages:
        if stage not in STAGE_LRS:
            raise ValueError(f"Unsupported stage: {stage}")

    if train_cfg.basic.seed is None:
        raise ValueError(
            "A training seed is required. Set basic.seed in the training "
            "configuration or pass --seed."
        )
    seed_everything(int(train_cfg.basic.seed))

    root = repo_root()
    output_dir = Path(train_cfg.path.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = selected_tasks(task_cfg)
    paper_task_type = infer_paper_task_type(tasks)
    if paper_task_type is not None and getattr(args, "grad_accum_steps", None) is None:
        train_cfg.train.grad_accum_steps = int(PAPER_GRAD_ACCUM_STEPS[paper_task_type])

    short_tasks = "-".join(task_short_name(task_cfg, t) for t in tasks)
    ts = _dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    extra = str(train_cfg.extra_name or "run")
    pipeline_root = output_dir / f"{extra}_{short_tasks}_{'-'.join(stages)}_{ts}"
    pipeline_root.mkdir(parents=True, exist_ok=True)

    print(f"[Pipeline] root: {pipeline_root}")
    print(f"[Pipeline] tasks: {tasks}")
    print(f"[Pipeline] stages: {stages}")
    print(f"[Pipeline] checkpoint_policy: {args.checkpoint_policy}")
    if paper_task_type is not None:
        print(f"[Pipeline] paper_task_type: {paper_task_type}")

    model, tokenizer = load_model_and_tokenizer(
        model_name_or_path=str(train_cfg.model.name_or_path),
        device=str(train_cfg.basic.device),
    )
    if args.resume_from_checkpoint:
        _load_checkpoint(model, args.resume_from_checkpoint, device=str(train_cfg.basic.device))

    manifest: Dict[str, Any] = {
        "repo": str(root),
        "pipeline_root": str(pipeline_root),
        "tasks": tasks,
        "paper_task_type": paper_task_type,
        "stages": [],
        "checkpoint_policy": str(args.checkpoint_policy),
        "created_at": ts,
    }

    for stage_index, stage in enumerate(stages):
        stage_start_epoch = time.time()
        stage_started_at = _dt.datetime.now().isoformat(timespec="seconds")
        stage_train_cfg = clone_cfg(train_cfg)
        stage_task_cfg = clone_cfg(task_cfg)
        stage_train_cfg.train.mode = stage
        stage_train_cfg.train.lr = _stage_lr(args, stage_train_cfg, stage, paper_task_type)
        stage_train_cfg.train.max_opt_steps = _stage_steps(
            args, stage_train_cfg, stage, paper_task_type
        )
        stage_train_cfg.train.save_every_steps = _periodic_save_every(
            str(args.checkpoint_policy),
            int(stage_train_cfg.train.save_every_steps),
        )

        stage_dir = pipeline_root / f"{stage_index + 1:02d}_{stage}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        _save_effective_configs(stage_dir, stage_train_cfg, stage_task_cfg)

        print(f"\n[Stage {stage_index + 1}/{len(stages)}] {stage}")
        print(
            f"[Stage] lr={stage_train_cfg.train.lr} max_opt_steps={stage_train_cfg.train.max_opt_steps}"
        )
        print(f"[Stage] output_dir={stage_dir}")

        train_ds = build_mixed_dataset(
            stage_train_cfg, stage_task_cfg, split="train", tokenizer=tokenizer
        )
        val_ds = None
        if (
            int(stage_train_cfg.train.eval_every_steps) > 0
            and int(stage_task_cfg.n_samples_valid) > 0
        ):
            val_ds = build_mixed_dataset(
                stage_train_cfg, stage_task_cfg, split="valid", tokenizer=tokenizer
            )

        wandb_run = _init_wandb(stage_train_cfg, stage_task_cfg, stage_dir)
        trainer = DllmTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_ds,
            val_dataset=val_ds,
            mode=stage,
            device=str(stage_train_cfg.basic.device),
            batch_size=int(stage_train_cfg.train.batch_size),
            grad_accum_steps=int(stage_train_cfg.train.grad_accum_steps),
            lr=float(stage_train_cfg.train.lr),
            warmup_steps=stage_train_cfg.train.get("warmup_steps", None),
            warmup_ratio=stage_train_cfg.train.get("warmup_ratio", None),
            lr_schedule=str(stage_train_cfg.train.get("lr_schedule", "warmup_constant")),
            max_grad_norm=float(stage_train_cfg.train.max_grad_norm),
            max_opt_steps=int(stage_train_cfg.train.max_opt_steps),
            log_every_steps=int(stage_train_cfg.train.log_every_steps),
            eval_every_steps=int(stage_train_cfg.train.eval_every_steps),
            n_samples_valid=int(stage_task_cfg.n_samples_valid),
            save_every_steps=int(stage_train_cfg.train.save_every_steps),
            num_workers=int(stage_train_cfg.train.num_workers),
            optimizer_name=str(stage_train_cfg.train.optimizer_name),
            use_bf16=bool(stage_train_cfg.basic.use_bf16),
            use_kv_cache=bool(stage_train_cfg.basic.use_kv_cache),
            reward_stats_json=str(stage_train_cfg.path.reward_stats_json),
            reward_center=bool(stage_train_cfg.train.reward_center),
            gradient_checkpointing=bool(stage_train_cfg.train.get("gradient_checkpointing", False)),
            output_dir=str(stage_dir),
        )

        train_result = trainer.train(mode=stage, wandb_run=wandb_run)
        final_checkpoint = None
        if _should_save_final(str(args.checkpoint_policy), stage_index, len(stages)):
            final_checkpoint = trainer.save_checkpoint(
                int(train_result["optim_step"]), suffix="final"
            )
            print(f"[CKPT] Saved final checkpoint: {final_checkpoint}")

        if wandb_run is not None:
            wandb_run.finish()

        stage_end_epoch = time.time()
        stage_finished_at = _dt.datetime.now().isoformat(timespec="seconds")

        stage_record = {
            "stage": stage,
            "stage_dir": str(stage_dir),
            "lr": float(stage_train_cfg.train.lr),
            "max_opt_steps": int(stage_train_cfg.train.max_opt_steps),
            "warmup_steps": stage_train_cfg.train.get("warmup_steps", None),
            "lr_schedule": str(stage_train_cfg.train.get("lr_schedule", "warmup_constant")),
            "optim_step": int(train_result["optim_step"]),
            "periodic_checkpoint": train_result.get("last_checkpoint_path"),
            "final_checkpoint": str(final_checkpoint) if final_checkpoint else None,
            "started_at": stage_started_at,
            "finished_at": stage_finished_at,
            "wall_seconds": float(stage_end_epoch - stage_start_epoch),
        }
        manifest["stages"].append(stage_record)
        (pipeline_root / "pipeline_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    print(f"\n[Pipeline] complete: {pipeline_root}")
    return manifest
