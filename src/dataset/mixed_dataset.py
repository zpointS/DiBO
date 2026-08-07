from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from src.dataset.task_pool_dataset import TaskPoolDataset, PoolConfig
from src.dataset.utils import load_templates_for_task, load_bundle_for_task

MixMode = Literal["round_robin", "random_uniform", "weighted", "fixed_order"]


@dataclass(frozen=True)
class MixConfig:
    """
    Multi-task sampling configuration.

    These datasets are IterableDatasets. The training loop stops by optimizer
    steps, so no explicit epoch length is needed.
    """

    mode: MixMode = "random_uniform"

    weights: Optional[Dict[str, float]] = (
        None  # Task-name to probability map for weighted sampling.
    )

    fixed_order: Optional[List[str]] = None  # Repeated task order for fixed-order sampling.

    # Whether round-robin mode reshuffles tasks each cycle.
    rr_shuffle_tasks: bool = False


class MixedDataset(IterableDataset):
    """
    Mix multiple TaskPoolDataset instances.

    Output keeps the TaskPoolDataset item structure and adds task_id/task_name.
    """

    def __init__(
        self,
        task_datasets: Sequence[TaskPoolDataset],
        *,
        cfg: MixConfig,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        if len(task_datasets) == 0:
            raise ValueError("task_datasets must be non-empty")
        if seed is None:
            raise ValueError("MixedDataset requires an explicit seed.")

        self.task_datasets: List[TaskPoolDataset] = list(task_datasets)
        self.cfg = cfg
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)

        self.task_names = [ds.task_name for ds in self.task_datasets]
        self.name2tid = {name: i for i, name in enumerate(self.task_names)}
        self._step = 0
        self._rr_order = list(range(len(self.task_datasets)))

        if self.cfg.mode == "weighted":
            self._validate_weights()

        if self.cfg.mode == "fixed_order":
            self._validate_fixed_order()

    def __iter__(self):
        worker = get_worker_info()
        worker_offset = 0 if worker is None else int(worker.id)
        self._rng = np.random.default_rng(self.seed + worker_offset)
        self._step = 0
        self._rr_order = list(range(len(self.task_datasets)))
        if self.cfg.mode == "round_robin" and self.cfg.rr_shuffle_tasks:
            self._rng.shuffle(self._rr_order)

        task_iters = [iter(ds) for ds in self.task_datasets]

        while True:
            task_id = self._choose_task_id()

            item = next(task_iters[task_id])

            item["task_id"] = torch.tensor(task_id, dtype=torch.long)
            item["task_name"] = self.task_datasets[task_id].task_name
            self._step += 1
            yield item

    def _choose_task_id(self) -> int:
        mode = self.cfg.mode
        if mode == "round_robin":
            return self._choose_round_robin()
        if mode == "random_uniform":
            return self._choose_random_uniform()
        if mode == "weighted":
            return self._choose_weighted()
        if mode == "fixed_order":
            return self._choose_fixed_order()
        raise ValueError(f"Unknown mix mode: {mode}")

    def _choose_round_robin(self) -> int:
        if self._step > 0 and self._step % len(self._rr_order) == 0 and self.cfg.rr_shuffle_tasks:
            self._rng.shuffle(self._rr_order)
        return int(self._rr_order[self._step % len(self._rr_order)])

    def _choose_random_uniform(self) -> int:
        T = len(self.task_datasets)
        return int(self._rng.integers(0, T))

    def _choose_weighted(self) -> int:
        weights = self._weights_vector()
        tids = np.arange(len(self.task_datasets))
        return int(self._rng.choice(tids, p=weights))

    def _choose_fixed_order(self) -> int:
        assert self.cfg.fixed_order is not None
        name = self.cfg.fixed_order[self._step % len(self.cfg.fixed_order)]
        return int(self.name2tid[name])

    def _validate_weights(self) -> None:
        if not self.cfg.weights:
            raise ValueError("mode='weighted' requires cfg.weights")

        total = 0.0
        for name, w in self.cfg.weights.items():
            if name not in self.name2tid:
                raise ValueError(f"weights contains unknown task_name: {name}")
            if float(w) < 0:
                raise ValueError(f"weights must be non-negative, got {name}={w}")
            total += float(w)

        if not np.isfinite(total) or abs(total - 1.0) > 1e-6:
            raise ValueError(f"weights must sum to 1.0, got sum={total}")

    def _weights_vector(self) -> np.ndarray:
        assert self.cfg.weights is not None
        vec = np.zeros(len(self.task_datasets), dtype=np.float64)
        for name, w in self.cfg.weights.items():
            vec[self.name2tid[name]] = float(w)
        vec = vec / vec.sum()
        return vec

    def _validate_fixed_order(self) -> None:
        if not self.cfg.fixed_order:
            raise ValueError("mode='fixed_order' requires cfg.fixed_order")
        for name in self.cfg.fixed_order:
            if name not in self.name2tid:
                raise ValueError(f"fixed_order contains unknown task_name: {name}")


def build_mixed_dataset(
    train_cfg,
    task_cfg,
    split: str,
    tokenizer,
) -> MixedDataset:

    assert split in {"train", "valid"}, f"split must be train/valid, got {split}"

    task_datasets: List[TaskPoolDataset] = []

    for i, (task_name, use_flag) in enumerate(task_cfg.use.items()):
        if not use_flag:
            continue
        else:
            idx = task_cfg.task_names.index(task_name)
            i += idx + 1  # Keep distinct seeds even when some tasks are skipped.
            print(f"[Dataset] {split}: {task_name}")

        train_templates, valid_templates = load_templates_for_task(
            task_name, prompts_dir=task_cfg.path.prompts_dir
        )
        templates = train_templates + valid_templates

        bundle = load_bundle_for_task(task_name, bundle_dir=task_cfg.path.bundle_dir)

        seed_biased = (
            int(train_cfg.basic.seed + i * 42)
            if split == "valid"
            else int(train_cfg.basic.seed + i * 43)
        )

        per_task_cfg = PoolConfig(
            split=str(split),
            mode=str(train_cfg.train.mode),
            use_oracle=bool(task_cfg.use_oracle),
            special_token_type=str(train_cfg.train.special_token_type),
            ablation_use_random_neighbors=str(train_cfg.train.ablation_use_random_neighbors),
            ablation_use_high_or_low_pool=str(train_cfg.train.ablation_use_high_or_low_pool),
            n_pool=int(task_cfg.n_pool),
            n_few_shot=int(task_cfg.n_few_shot),
            K_pool=int(task_cfg.k_pool),
            ratio=float(task_cfg.ratio),
            seed=seed_biased,
            num_template=task_cfg.num_template,
        )

        ds = TaskPoolDataset(
            tokenizer=tokenizer,
            templates=templates,
            task_bundle=bundle,
            cfg=per_task_cfg,
        )
        task_datasets.append(ds)

    mix_section = task_cfg.get("mix", {})
    weights = mix_section.get("weights", None)
    fixed_order = mix_section.get("fixed_order", None)
    if weights is not None:
        weights = {str(name): float(weight) for name, weight in weights.items()}
    if fixed_order is not None:
        fixed_order = [str(name) for name in fixed_order]

    mix_cfg = MixConfig(
        mode=str(mix_section.get("mode", "random_uniform")),
        weights=weights,
        fixed_order=fixed_order,
        rr_shuffle_tasks=bool(mix_section.get("rr_shuffle_tasks", False)),
    )

    return MixedDataset(task_datasets=task_datasets, cfg=mix_cfg, seed=train_cfg.basic.seed)
