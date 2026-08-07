from __future__ import annotations

import argparse

import src.compat.compat_patches  # noqa: F401

SUPPORTED_TASKS = (
    "TFBind8-Exact-v0",
    "TFBind10-Exact-v0",
    "AntMorphology-Exact-v0",
    "DKittyMorphology-Exact-v0",
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run the DiBO DA/SFT/RL training pipeline.")

    p.add_argument("--train_cfg_path", type=str, default="./configs/train.yaml")
    p.add_argument("--task_cfg_path", type=str, default="./configs/task.yaml")
    p.add_argument("--stages", nargs="+", default=["da", "sft", "rl"], choices=["da", "sft", "rl"])
    p.add_argument(
        "--checkpoint_policy", choices=["none", "last", "each_stage", "periodic"], default="last"
    )
    p.add_argument("--resume_from_checkpoint", type=str, default=None)

    p.add_argument("--model_name_or_path", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--bundle_dir", type=str, default=None)
    p.add_argument("--prompts_dir", type=str, default=None)
    p.add_argument("--reward_stats_json", type=str, default=None)
    p.add_argument("--wandb_dir", type=str, default=None)

    p.add_argument("--extra_name", type=str, default=None)
    p.add_argument("--special_token_type", type=str, default=None, choices=["special", "natural"])
    p.add_argument(
        "--ablation_use_random_neighbors", type=str, default=None, choices=["d1-d2", "random"]
    )
    p.add_argument(
        "--ablation_use_high_or_low_pool",
        type=str,
        default=None,
        choices=["evenly", "high", "low", "random"],
    )

    p.add_argument("--max_opt_steps", type=int, default=None)
    p.add_argument("--da_steps", type=int, default=None)
    p.add_argument("--sft_steps", type=int, default=None)
    p.add_argument("--rl_steps", type=int, default=None)
    p.add_argument("--log_every_steps", type=int, default=None)
    p.add_argument("--eval_every_steps", type=int, default=None)
    p.add_argument("--save_every_steps", type=int, default=None)

    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--grad_accum_steps", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--da_lr", type=float, default=None)
    p.add_argument("--sft_lr", type=float, default=None)
    p.add_argument("--rl_lr", type=float, default=None)
    p.add_argument("--warmup_steps", type=int, default=None)
    p.add_argument("--warmup_ratio", type=float, default=None)
    p.add_argument(
        "--lr_schedule", type=str, default=None, choices=["warmup_constant", "linear_decay"]
    )
    p.add_argument("--max_grad_norm", type=float, default=None)
    p.add_argument("--optimizer_name", type=str, default=None)
    p.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=None)

    p.add_argument("--project", type=str, default=None)
    p.add_argument("--wandb_mode", type=str, default=None)
    p.add_argument("--use_wandb", action=argparse.BooleanOptionalAction, default=None)

    p.add_argument(
        "--use",
        nargs="+",
        choices=SUPPORTED_TASKS,
        default=None,
        help="Task names to train on.",
    )
    p.add_argument("--n_samples_valid", type=int, default=None)
    p.add_argument("--use_oracle", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--n_pool", type=int, default=None)
    p.add_argument("--n_few_shot", type=int, default=None)
    p.add_argument("--k_pool", type=int, default=None)
    p.add_argument("--ratio", type=float, default=None)
    p.add_argument(
        "--mix_mode",
        type=str,
        default=None,
        choices=["round_robin", "random_uniform", "weighted", "fixed_order"],
    )
    p.add_argument(
        "--mix_weights", nargs="*", default=None, help="Task weights as task_name=prob entries."
    )
    p.add_argument("--mix_fixed_order", nargs="+", default=None)
    p.add_argument("--mix_rr_shuffle_tasks", action=argparse.BooleanOptionalAction, default=None)

    return p


def main() -> None:
    args = build_parser().parse_args()
    from src.pipeline import run_pipeline

    run_pipeline(args)


if __name__ == "__main__":
    main()
