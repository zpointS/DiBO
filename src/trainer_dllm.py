from __future__ import annotations

import inspect
import math, random
import os
from tqdm import tqdm
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.utils.rnn import pad_sequence

from src.utils import _to_device, load_reward_stats_json, infinite_loader


MASK_TOKEN_ID = 126336  # LLaDA mask token id.


def masked_importance_weighted_ce_sum(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    masked_indices: torch.Tensor,
    p_mask: torch.Tensor,
    *,
    chunk_size: int = 512,
) -> torch.Tensor:
    """Return sum(CE(masked token) / p_mask) without materializing all masked logits."""
    positions = masked_indices.nonzero(as_tuple=False)
    if positions.numel() == 0:
        return logits.sum() * 0.0

    input_ids = input_ids.to(logits.device)
    p_mask = p_mask.to(logits.device)
    chunk_size = max(1, int(chunk_size))
    total = torch.zeros((), dtype=torch.float32, device=logits.device)
    for pos in positions.split(chunk_size, dim=0):
        batch_idx = pos[:, 0]
        token_idx = pos[:, 1]
        chunk_logits = logits[batch_idx, token_idx, :].float()
        chunk_targets = input_ids[batch_idx, token_idx]
        chunk_p = p_mask[batch_idx, token_idx].float().clamp_min(1e-6)
        total = (
            total + (F.cross_entropy(chunk_logits, chunk_targets, reduction="none") / chunk_p).sum()
        )
    return total


@torch.no_grad()
def masked_entropy_mean(
    logits: torch.Tensor,
    masked_indices: torch.Tensor,
    *,
    chunk_size: int = 512,
) -> torch.Tensor:
    """Return mean entropy on masked tokens only, chunked and detached for metrics."""
    positions = masked_indices.nonzero(as_tuple=False)
    if positions.numel() == 0:
        return torch.zeros((), dtype=torch.float32, device=logits.device)

    chunk_size = max(1, int(chunk_size))
    total = torch.zeros((), dtype=torch.float32, device=logits.device)
    count = 0
    for pos in positions.split(chunk_size, dim=0):
        batch_idx = pos[:, 0]
        token_idx = pos[:, 1]
        chunk_logits = logits[batch_idx, token_idx, :].float()
        logprobs = torch.log_softmax(chunk_logits, dim=-1)
        probs = logprobs.exp()
        total = total + (-(probs * logprobs).sum(dim=-1)).sum()
        count += int(pos.size(0))
    return total / max(1, count)


def masked_reward_weighted_probability_mean(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    masked_indices: torch.Tensor,
    reward: torch.Tensor,
    *,
    chunk_size: int = 512,
) -> torch.Tensor:
    """Return token-level mean(reward * P(target token)) without full softmax."""
    positions = masked_indices.nonzero(as_tuple=False)
    if positions.numel() == 0:
        return logits.sum() * 0.0

    input_ids = input_ids.to(logits.device)
    reward = reward.to(logits.device).view(-1)
    if reward.numel() != logits.size(0):
        raise ValueError(
            "RL reward batch dimension does not match logits: "
            f"reward={tuple(reward.shape)} logits={tuple(logits.shape)}"
        )
    chunk_size = max(1, int(chunk_size))
    total = torch.zeros((), dtype=torch.float32, device=logits.device)
    count = 0
    for pos in positions.split(chunk_size, dim=0):
        batch_idx = pos[:, 0]
        token_idx = pos[:, 1]
        chunk_logits = logits[batch_idx, token_idx, :].float()
        chunk_targets = input_ids[batch_idx, token_idx]
        target_logits = chunk_logits.gather(dim=-1, index=chunk_targets.unsqueeze(-1)).squeeze(-1)
        p_correct = (target_logits - torch.logsumexp(chunk_logits, dim=-1)).exp()
        total = total + (reward[batch_idx] * p_correct).sum()
        count += int(pos.size(0))
    return total / max(1, count)


def collate_task_items(items: List[Dict[str, Any]], pad_token_id: int) -> Dict[str, Any]:
    """
    Collate TaskPoolDataset/MixedDataset items into one batch.

    - input_ids and attention_mask are padded to the longest sequence.
    - response_token_ids stays as a list because sequence lengths may differ.
    - Scalar and vector tensor fields are stacked.
    - String metadata remains as Python lists.
    """
    input_ids = [x["input_ids"] for x in items]
    attention_mask = [x["attention_mask"] for x in items]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

    batch: Dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }

    if "response_token_ids" in items[0]:
        batch["response_token_ids"] = [x["response_token_ids"] for x in items]

    for k in [
        "response_start_idx",
        "response_end_idx",
        "anchor_index",
        "prompt_indices",
        "prompt_labels",
        "response_label",
        "reward_raw",
        "template_id",
        "task_id",
    ]:
        if k in items[0]:
            v0 = items[0][k]
            if torch.is_tensor(v0):
                batch[k] = torch.stack([x[k] for x in items], dim=0)
            else:
                batch[k] = [x[k] for x in items]

    for k in ["task_name"]:
        if k in items[0]:
            batch[k] = [x[k] for x in items]

    return batch


class DllmTrainer:
    def __init__(
        self,
        model,
        tokenizer,
        train_dataset,
        val_dataset=None,
        *,
        mode: str = "da",
        device: str = "cuda",
        batch_size: int = 1,
        grad_accum_steps: int = 8,
        lr: float = 1e-5,
        warmup_steps: Optional[int] = 100,
        warmup_ratio: Optional[float] = None,
        lr_schedule: str = "warmup_constant",
        max_grad_norm: float = 2.0,
        max_opt_steps: int = 2048,
        log_every_steps: int = 1,
        eval_every_steps: int = 200,
        n_samples_valid: int = 256,
        save_every_steps: int = 128,
        num_workers: int = 0,
        optimizer_name: str = "pagedadamw8bit",
        use_bf16: bool = True,
        use_kv_cache: bool = False,
        reward_stats_json: Optional[str] = None,
        reward_center: bool = False,  # Whether RL reward normalization subtracts the mean.
        gradient_checkpointing: bool = False,
        output_dir: str = "outputs/checkpoints",
    ):

        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.mode = str(mode)
        self.task_name = train_dataset.task_names[0]

        self.use_bf16 = bool(use_bf16)
        if self.use_bf16:
            print("Using bfloat16 mixed precision.")
            self.model.to(dtype=torch.bfloat16)

        self.use_kv_cache = bool(use_kv_cache)
        if not self.use_kv_cache and hasattr(self.model, "config"):
            print("Disable model caching.")
            self.model.config.use_cache = False

        self.gradient_checkpointing = bool(gradient_checkpointing)
        if self.gradient_checkpointing:
            enabled_checkpointing = False
            backbone = getattr(self.model, "model", None)
            if backbone is not None and hasattr(backbone, "set_activation_checkpointing"):
                module = inspect.getmodule(backbone)
                strategy_cls = (
                    getattr(module, "ActivationCheckpointingStrategy", None)
                    if module is not None
                    else None
                )
                strategy_name = os.environ.get(
                    "LLADA_ACTIVATION_CHECKPOINTING_STRATEGY", "whole_layer"
                ).strip()
                combo_strategy = strategy_name in {
                    "whole_layer_fine_grained",
                    "whole_layer+fine_grained",
                    "whole_layer_and_fine_grained",
                }
                base_strategy_name = "whole_layer" if combo_strategy else strategy_name
                strategy = (
                    getattr(strategy_cls, base_strategy_name, None)
                    if strategy_cls is not None
                    else None
                )
                if strategy is not None:
                    print(f"Enable LLaDA activation checkpointing: {strategy_name}.")
                    backbone.set_activation_checkpointing(strategy)
                    if combo_strategy:
                        fine_strategy = getattr(strategy_cls, "fine_grained", None)
                        if fine_strategy is None:
                            raise ValueError(
                                "LLaDA fine_grained activation checkpointing strategy is unavailable"
                            )
                        n_blocks = 0
                        for submodule in backbone.modules():
                            if any(cls.__name__ == "LLaDABlock" for cls in type(submodule).__mro__):
                                submodule.set_activation_checkpointing(fine_strategy)
                                n_blocks += 1
                        print(
                            f"Enable LLaDA fine-grained checkpointing inside {n_blocks} transformer blocks."
                        )
                    enabled_checkpointing = True
                elif strategy_cls is not None:
                    raise ValueError(
                        "Unknown LLaDA activation checkpointing strategy: " f"{strategy_name!r}"
                    )

            if not enabled_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
                print("Enable HF gradient checkpointing.")
                try:
                    self.model.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs={"use_reentrant": False}
                    )
                except TypeError:
                    self.model.gradient_checkpointing_enable()
                enabled_checkpointing = True

            if not enabled_checkpointing:
                print(
                    "[Warning] gradient_checkpointing requested but no supported checkpointing hook was found."
                )

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.batch_size = int(batch_size)
        self.grad_accum_steps = int(max(1, grad_accum_steps))
        self.masked_ce_chunk_size = int(os.environ.get("DIBO_MASKED_CE_CHUNK_SIZE", "512"))
        if self.masked_ce_chunk_size < 1:
            raise ValueError("DIBO_MASKED_CE_CHUNK_SIZE must be >= 1")
        print(f"[Memory] masked CE chunk size: {self.masked_ce_chunk_size}")

        self.lr = float(lr)
        self.warmup_ratio = None if warmup_ratio is None else float(warmup_ratio)
        self.lr_schedule = str(lr_schedule)
        self.max_grad_norm = float(max_grad_norm)

        self.max_opt_steps = int(max_opt_steps)
        self.max_steps = self.max_opt_steps * self.grad_accum_steps
        self.log_every_steps = int(log_every_steps)
        self.eval_every_steps = int(eval_every_steps)
        self.n_samples_valid = int(n_samples_valid)
        self.save_every_steps = int(save_every_steps)

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.num_workers = int(num_workers)

        pad_id = self.tokenizer.pad_token_id

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=lambda items: collate_task_items(items, pad_token_id=pad_id),
        )

        self.val_loader = None
        if self.val_dataset is not None:
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                collate_fn=lambda items: collate_task_items(items, pad_token_id=pad_id),
            )

        self.optimizer_name = str(optimizer_name).lower()
        decay, no_decay = [], []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim == 1 or n.endswith(".bias") or "norm" in n.lower():
                no_decay.append(p)
            else:
                decay.append(p)

        bnb_optimizer_names = {
            "adamw8bit",
            "pagedadamw8bit",
            "pagedadamw32bit",
            "pagedadamw",
        }
        if self.optimizer_name in bnb_optimizer_names:
            try:
                import bitsandbytes as bnb
            except ImportError as exc:
                raise ImportError(
                    "bitsandbytes is required for optimizer_name="
                    f"{self.optimizer_name!r}. Use --optimizer_name adamw if "
                    "bitsandbytes is not installed."
                ) from exc

        if self.optimizer_name == "adamw8bit":
            self.optimizer = bnb.optim.AdamW8bit(
                [
                    {"params": decay, "weight_decay": 0.01},
                    {"params": no_decay, "weight_decay": 0.0},
                ],
                lr=self.lr,
                betas=(0.9, 0.999),
                eps=1e-8,
            )
        elif self.optimizer_name == "pagedadamw8bit":
            self.optimizer = bnb.optim.PagedAdamW8bit(
                [
                    {"params": decay, "weight_decay": 0.01},
                    {"params": no_decay, "weight_decay": 0.0},
                ],
                lr=self.lr,
            )
        elif self.optimizer_name in {"pagedadamw32bit", "pagedadamw"}:
            self.optimizer = bnb.optim.PagedAdamW32bit(
                [
                    {"params": decay, "weight_decay": 0.01},
                    {"params": no_decay, "weight_decay": 0.0},
                ],
                lr=self.lr,
            )
        elif self.optimizer_name == "adamw":
            self.optimizer = AdamW(
                [
                    {"params": decay, "weight_decay": 0.01},
                    {"params": no_decay, "weight_decay": 0.0},
                ],
                lr=self.lr,
                betas=(0.9, 0.999),
                eps=1e-8,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")

        def warmup_constant_lr(optimizer, warmup_steps: int):
            def lr_lambda(step: int):
                if step < warmup_steps:
                    return float(step + 1) / float(warmup_steps)
                return 1.0

            return LambdaLR(optimizer, lr_lambda)

        if warmup_steps is None:
            ratio = 0.0 if self.warmup_ratio is None else self.warmup_ratio
            warmup_steps = int(round(ratio * self.max_opt_steps))
        warmup_steps = max(0, int(warmup_steps))
        self.warmup_steps = warmup_steps

        if self.lr_schedule == "warmup_constant":
            print(f"[Scheduler] mode={self.mode} constant LR, warmup_steps={warmup_steps}")
            self.scheduler = warmup_constant_lr(
                self.optimizer,
                warmup_steps=warmup_steps,
            )
        elif self.lr_schedule == "linear_decay":
            print(f"[Scheduler] mode={self.mode} linear decay, warmup_steps={warmup_steps}")
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=self.max_opt_steps,
            )
        else:
            raise ValueError(f"Unknown lr_schedule: {self.lr_schedule}")

        self.reward_center = bool(reward_center)
        self.reward_stats_by_task: Optional[Dict[str, Dict[str, Any]]] = None
        self._reward_mean_by_id: Optional[torch.Tensor] = None
        self._reward_std_by_id: Optional[torch.Tensor] = None

        if reward_stats_json is not None:
            means, stds = [], []
            self.reward_stats_by_task = {}
            for tname in self.train_dataset.task_names:
                rec = load_reward_stats_json(reward_stats_json, str(tname))
                stats = rec["adv_variants"]["adv_r_over_std"]
                means.append(float(stats["global_reward_mean"]))
                stds.append(max(float(stats["global_reward_std"]), 1e-6))
                self.reward_stats_by_task[str(tname)] = rec
                print(f"[Reward norm] task={tname} mean={means[-1]:.6f} std={stds[-1]:.6f}")
            self._reward_mean_by_id = torch.tensor(means, dtype=torch.float32, device=self.device)
            self._reward_std_by_id = torch.tensor(stds, dtype=torch.float32, device=self.device)

        self.last_checkpoint_path: Optional[Path] = None

    @staticmethod
    def _grad_total_norm(parameters) -> float:
        total_sq = 0.0
        for param in parameters:
            if param.grad is None:
                continue
            grad_norm = param.grad.detach().norm(2)
            total_sq += float(grad_norm.detach().cpu()) ** 2
        return float(total_sq**0.5)

    def save_checkpoint(self, optim_step: int, *, suffix: Optional[str] = None) -> Path:
        name = f"optim_step={int(optim_step)}"
        if suffix:
            name += f"_{suffix}"
        ckpt = self.output_dir / f"{name}.pt"
        torch.save({"model": self.model.state_dict()}, ckpt)
        self.last_checkpoint_path = ckpt
        return ckpt

    def _forward_corrupt(self, input_ids: torch.Tensor, eps: float = 1e-3):
        """
        LLaDA-style forward corruption.

        DA and RL first mask tokens with diffusion-style random corruption, then
        narrow the supervision span according to the stage.
        """
        b, l = input_ids.shape
        t = torch.rand(b, device=self.device)
        p_mask = (1 - eps) * t
        p_mask = p_mask[:, None].repeat(1, l)
        masked_indices = torch.rand((b, l), device=self.device) < p_mask
        noisy_batch = torch.where(
            masked_indices, torch.tensor(MASK_TOKEN_ID, device=self.device), input_ids
        )
        return noisy_batch, masked_indices, p_mask

    def _mask_response_span(
        self,
        noisy_input: torch.Tensor,
        masked_indices: torch.Tensor,
        batch: Dict[str, Any],
        mode: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Restrict the masked region according to the training stage.

        - da: keep the original random mask.
        - sft: keep only random masks inside the response span.
        - rl: mask the full response span.
        """
        if "response_start_idx" not in batch or "response_end_idx" not in batch:
            print(
                f"[Warning] (mode={mode}) batch does not contain response span indices; using original masking."
            )
            return noisy_input, masked_indices

        token_pos = torch.arange(noisy_input.size(1), device=self.device).expand_as(noisy_input)
        start = batch["response_start_idx"].unsqueeze(1)
        end = batch["response_end_idx"].unsqueeze(1)

        outside_mask = (token_pos <= start) | (token_pos >= end)
        inside_mask = ~outside_mask

        if mode == "da":
            return noisy_input, masked_indices

        if mode == "sft":
            noisy_input[outside_mask] = batch["input_ids"][outside_mask]
            masked_indices[outside_mask] = False
            return noisy_input, masked_indices

        if mode == "rl":
            noisy_input[outside_mask] = batch["input_ids"][outside_mask]
            noisy_input[inside_mask] = MASK_TOKEN_ID
            masked_indices[outside_mask] = False
            masked_indices[inside_mask] = True
            return noisy_input, masked_indices

        return noisy_input, masked_indices

    def _normalize_reward(self, reward_raw: torch.Tensor, batch: Dict[str, Any]) -> torch.Tensor:
        """
        Normalize rewards by task_id.

        Returns r/std by default; if reward_center=True, returns (r-mean)/std.
        """
        if self._reward_std_by_id is None:
            print("[Warning] No reward stats loaded; skipping reward normalization.")
            return reward_raw

        task_id = batch.get("task_id")
        if task_id is None:
            task_id = torch.zeros_like(reward_raw, dtype=torch.long, device=self.device)
        elif torch.is_tensor(task_id):
            task_id = task_id.to(self.device).long().view(-1)
        else:
            task_id = torch.tensor(task_id, dtype=torch.long, device=self.device).view(-1)

        std = self._reward_std_by_id.gather(0, task_id).clamp_min(1e-6)
        if self.reward_center:
            assert self._reward_mean_by_id is not None
            mean = self._reward_mean_by_id.gather(0, task_id)
            return (reward_raw - mean) / std
        return reward_raw / std

    def _compute_loss(
        self,
        logits: torch.Tensor,
        batch: Dict[str, Any],
        masked_indices: torch.Tensor,
        p_mask: torch.Tensor,
        *,
        mode: str,
        reward_raw: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the stage-specific loss.

        - da/sft: diffusion token loss on masked tokens.
        - rl: token loss on the response span, weighted by normalized reward.
        """
        metrics: Dict[str, float] = {}
        input_ids = batch["input_ids"]

        num_masked = masked_indices.sum().item()
        if num_masked == 0:
            print(f"[{mode}] no masked tokens in this batch")

        if mode in {"da", "sft"}:
            input_ids = batch["input_ids"]
            loss_numerator = masked_importance_weighted_ce_sum(
                logits,
                input_ids,
                masked_indices,
                p_mask,
                chunk_size=self.masked_ce_chunk_size,
            )
            entropy_masked = masked_entropy_mean(
                logits, masked_indices, chunk_size=self.masked_ce_chunk_size
            )

            if mode == "da":
                denom = float(input_ids.numel())
            elif mode == "sft":
                denom = masked_indices.sum().clamp_min(1).float()

            loss = loss_numerator / denom
            metrics["loss"] = float(loss.detach().cpu())
            metrics["entropy_masked"] = float(entropy_masked.detach().cpu())
            return loss, metrics

        if mode == "rl":
            assert reward_raw is not None, "RL requires reward_raw in batch"

            reward = self._normalize_reward(reward_raw, batch)
            reward_detached = reward.detach()

            input_ids = batch["input_ids"]
            mask = masked_indices

            entropy_masked = masked_entropy_mean(logits, mask, chunk_size=self.masked_ce_chunk_size)
            reward_weighted_prob = masked_reward_weighted_probability_mean(
                logits,
                input_ids,
                mask,
                reward_detached,
                chunk_size=self.masked_ce_chunk_size,
            )
            loss = -reward_weighted_prob

            metrics = {
                "loss": float(loss.detach().cpu()),
                "reward_norm": float(reward_detached.mean().detach().cpu()),
                "entropy_masked": float(entropy_masked.detach().cpu()),
            }
            return loss, metrics

        raise ValueError(f"Unknown mode: {mode}")

    def train(self, *, mode: str = "da", wandb_run=None) -> Dict[str, Any]:
        mode = str(mode)
        self.model.train()

        global_step = 0
        micro_step = 0
        optim_step = 0

        val_loss = float("nan")
        train_optim_step_mean = float("nan")
        step_losses = []
        metrics = []

        self.optimizer.zero_grad(set_to_none=True)

        train_iter = infinite_loader(self.train_loader)
        val_iter = infinite_loader(self.val_loader) if self.val_loader is not None else None
        pbar = tqdm(total=self.max_opt_steps, desc=f"train mode ({mode})", dynamic_ncols=True)

        while global_step < self.max_steps:
            batch = next(train_iter)
            global_step += 1
            micro_step += 1

            batch = _to_device(batch, self.device)

            noisy_input, masked_indices, p_mask = self._forward_corrupt(batch["input_ids"])

            attn = batch.get("attention_mask", None)
            if attn is not None:
                attn_bool = attn.bool()
                masked_indices = masked_indices & attn_bool
                noisy_input[~attn_bool] = batch["input_ids"][~attn_bool]

            noisy_input, masked_indices = self._mask_response_span(
                noisy_input, masked_indices, batch, mode=mode
            )

            reward_raw = batch.get("reward_raw", None)
            if reward_raw is not None and torch.is_tensor(reward_raw):
                reward_raw = reward_raw.to(self.device).float().view(-1)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.use_bf16):
                if not self.use_bf16:
                    print("Warning: not using bfloat16!")

                out = self.model(
                    input_ids=noisy_input,
                    attention_mask=batch.get("attention_mask", None),
                    use_cache=False,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
                logits = out.logits
                loss, m = self._compute_loss(
                    logits,
                    batch,
                    masked_indices,
                    p_mask,
                    mode=mode,
                    reward_raw=reward_raw,
                )
                metrics.append(m)

            step_losses.append(float(loss.detach().cpu()))
            loss_scaled = loss / float(self.grad_accum_steps)
            loss_scaled.backward()

            del out, logits, loss, loss_scaled
            del noisy_input, masked_indices, p_mask
            if reward_raw is not None:
                del reward_raw

            if (micro_step) % self.grad_accum_steps == 0:
                lr_used = float(self.optimizer.param_groups[0]["lr"])
                grad_norm_pre_clip = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                grad_norm_pre_clip = float(grad_norm_pre_clip.detach().cpu())
                grad_norm_post_clip = self._grad_total_norm(self.model.parameters())
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                optim_step += 1

                if self.save_every_steps > 0 and optim_step % self.save_every_steps == 0:
                    print(f"Saving checkpoint for optim_step {optim_step}...")
                    self.save_checkpoint(optim_step)

                if self.log_every_steps > 0 and (optim_step % self.log_every_steps) == 0:
                    train_optim_step_mean = float(sum(step_losses) / max(1, len(step_losses)))
                    if wandb_run is not None:
                        wandb_run.log(
                            {
                                "train/train_loss": train_optim_step_mean,
                                "train/lr": lr_used,
                                "train/optim_step": optim_step,
                                "train/global_step": global_step,
                                "train/grad_norm_pre_clip": grad_norm_pre_clip,
                                "train/grad_norm_post_clip": grad_norm_post_clip,
                                "train/entropy_masked": (
                                    np.mean([m.get("entropy_masked", 0.0) for m in metrics])
                                    if metrics
                                    else 0.0
                                ),
                                "train/reward_norm": (
                                    np.mean([m.get("reward_norm", 0.0) for m in metrics])
                                    if metrics
                                    else 0.0
                                ),
                            },
                            step=optim_step,
                        )
                    step_losses = []
                    metrics = []

                if self.eval_every_steps > 0 and optim_step % self.eval_every_steps == 0:
                    if val_iter is not None:
                        val_loss = self.evaluate(
                            val_iter=val_iter, mode=mode, wandb_run=wandb_run, step=optim_step
                        )

                pbar.set_postfix(
                    train_loss=f"{train_optim_step_mean:.10f}",
                    val_loss=f"{val_loss:.10f}",
                    lr=f"{lr_used:.2e}",
                    optim_step=str(optim_step),
                )
                pbar.update(1)

        pbar.close()
        return {
            "optim_step": optim_step,
            "global_step": global_step,
            "last_checkpoint_path": (
                str(self.last_checkpoint_path) if self.last_checkpoint_path else None
            ),
        }

    @torch.no_grad()
    def evaluate(self, val_iter, *, mode: str = "da", wandb_run=None, step: int = 0) -> None:
        self.model.eval()
        losses = []
        metrics = []
        global_step = 0

        while global_step < self.n_samples_valid:
            global_step += 1

            batch = next(val_iter)
            batch = _to_device(batch, self.device)

            noisy_input, masked_indices, p_mask = self._forward_corrupt(batch["input_ids"])

            attn = batch.get("attention_mask", None)
            if attn is not None:
                attn_bool = attn.bool()
                masked_indices = masked_indices & attn_bool
                noisy_input[~attn_bool] = batch["input_ids"][~attn_bool]

            noisy_input, masked_indices = self._mask_response_span(
                noisy_input, masked_indices, batch, mode=mode
            )

            reward_raw = batch.get("reward_raw", None)
            if reward_raw is not None and torch.is_tensor(reward_raw):
                reward_raw = reward_raw.to(self.device).float().view(-1)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.use_bf16):
                if not self.use_bf16:
                    print("Warning: not using bfloat16!")

                out = self.model(
                    input_ids=noisy_input,
                    attention_mask=batch.get("attention_mask", None),
                    use_cache=False,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
                logits = out.logits
                loss, m = self._compute_loss(
                    logits,
                    batch,
                    masked_indices,
                    p_mask,
                    mode=mode,
                    reward_raw=reward_raw,
                )
                metrics.append(m)

                del out, logits, noisy_input, masked_indices, p_mask
                del reward_raw

            losses.append(float(loss.cpu()))

        mean_loss = float(np.mean(losses)) if losses else 0.0
        if wandb_run is not None:
            wandb_run.log(
                {
                    "val/val_loss": mean_loss,
                    "val/optim_step": step,
                    "val/entropy_masked": (
                        float(np.mean([m.get("entropy_masked", 0.0) for m in metrics]))
                        if metrics
                        else 0.0
                    ),
                    "val/reward_norm": (
                        float(np.mean([m.get("reward_norm", 0.0) for m in metrics]))
                        if metrics
                        else 0.0
                    ),
                },
                step=step,
            )

        self.model.train()
        return mean_loss
