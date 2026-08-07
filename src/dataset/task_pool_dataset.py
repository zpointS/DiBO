from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import IterableDataset
from transformers import PreTrainedTokenizer

from src.dataset.task_bundle import TaskBundle

DESIGN_START = "|design-start|"
DESIGN_END = "|design-end|"
LABEL_START = "|label-start|"
LABEL_END = "|label-end|"
SUPPORTED_TASKS = {
    "TFBind8-Exact-v0",
    "TFBind10-Exact-v0",
    "AntMorphology-Exact-v0",
    "DKittyMorphology-Exact-v0",
}


@dataclass(frozen=True)
class PoolConfig:
    """
    Single-task dataset configuration.

    Conventions:
    - TaskBundle is already sorted by oracle(raw) in ascending order.
    - n_pool selects evenly spaced points from that sorted order by default.
    """

    split: str = "train"  # Dataset split.
    mode: str = "da"  # Training stage.
    use_oracle: bool = True  # Whether to use oracle labels or raw task labels.
    special_token_type: str = "special"  # Delimiter style.
    ablation_use_random_neighbors: str = (
        "d1-d2"  # Whether to replace local neighbors with random neighbors.
    )
    ablation_use_high_or_low_pool: str = "evenly"  # Pool sampling ablation.

    n_pool: int = 500
    n_few_shot: int = 7
    K_pool: int = 50  # Number of neighbors considered per anchor.
    ratio: float = 0.8  # Fraction of the pool assigned to d1.

    seed: Optional[int] = None

    num_template: Tuple[int, int] = (8, 2)

    lower_ratio: float = 0.7
    upper_ratio: float = 0.95


class TaskPoolDataset(IterableDataset):
    """
    Single-task IterableDataset backed by a TaskBundle.

    - Prompt rendering uses bundle.X_raw_sorted.
    - Kernel similarity uses bundle.X_normalized_sorted.
    - Training and reward use bundle.y_oracle_sorted.
    - The bundle itself is already sorted by oracle(raw) in ascending order.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        templates: Sequence[str],
        *,
        task_bundle: TaskBundle,
        cfg: PoolConfig,
    ) -> None:
        super().__init__()
        assert cfg.split in {"train", "valid"}
        assert cfg.mode in {"da", "sft", "rl"}
        assert 0.0 < cfg.ratio < 1.0
        assert 0.0 < cfg.lower_ratio < cfg.upper_ratio <= 1.0
        if cfg.seed is None:
            raise ValueError("TaskPoolDataset requires an explicit seed.")

        self.cfg = cfg
        self.seed = int(cfg.seed)

        self.tokenizer = tokenizer
        self.templates = list(templates)
        self.bundle = task_bundle
        self.task_name = task_bundle.task_name
        if self.task_name not in SUPPORTED_TASKS:
            raise ValueError(f"Unsupported task: {self.task_name}")

        self.split = cfg.split
        self.mode = cfg.mode
        self.ablation_use_random_neighbors = cfg.ablation_use_random_neighbors
        self.ablation_use_high_or_low_pool = cfg.ablation_use_high_or_low_pool

        self.n_pool = int(cfg.n_pool)
        self.n_few_shot = int(cfg.n_few_shot)
        self.K_pool = int(cfg.K_pool)
        self.ratio = float(cfg.ratio)

        self.lower_ratio = float(cfg.lower_ratio)
        self.upper_ratio = float(cfg.upper_ratio)

        self.K_lower_rl = self.K_pool // 2
        self.K_higher_rl = self.K_pool - self.K_lower_rl

        self._rng = np.random.default_rng(self.seed)

        self.use_oracle = bool(cfg.use_oracle)
        self.special_token_type = str(cfg.special_token_type)

        n_train, n_valid = cfg.num_template
        if self.split == "train":
            self._template_ids = list(range(0, n_train))
        else:
            self._template_ids = list(range(n_train, n_train + n_valid))

        X_raw_sorted = self.bundle.X_raw_sorted
        X_normalized_sorted = self.bundle.X_normalized_sorted

        if self.use_oracle:
            y_sorted = self.bundle.y_oracle_sorted.astype(np.float32).ravel()
        else:
            y_sorted = self.bundle.y_task_sorted.astype(np.float32).ravel()

        N_full = int(y_sorted.shape[0])
        if self.n_pool > N_full:
            raise ValueError(
                f"[{self.task_name}] n_pool={self.n_pool} > N_full={N_full}. "
                f"Either reduce n_pool or build a bigger bundle."
            )

        if self.ablation_use_high_or_low_pool == "evenly":
            pool_idx = self._evenly_spaced_indices(N_full, self.n_pool)
        elif self.ablation_use_high_or_low_pool == "random":
            self._pool_rng = np.random.default_rng(self.seed)
            pool_idx = self._pool_rng.choice(N_full, size=self.n_pool, replace=False)
            pool_idx = np.sort(pool_idx).astype(np.int64)
        elif self.ablation_use_high_or_low_pool == "high":
            pool_idx = np.arange(N_full - self.n_pool, N_full, dtype=np.int64)
        elif self.ablation_use_high_or_low_pool == "low":
            pool_idx = np.arange(0, self.n_pool, dtype=np.int64)
        else:
            raise ValueError(
                "ablation_use_high_or_low_pool must be one of: evenly, random, high, low"
            )

        self.pool_global_idx = pool_idx.astype(np.int64)

        self.pool_designs_raw = X_raw_sorted[self.pool_global_idx]
        self.pool_designs_normalized = X_normalized_sorted[self.pool_global_idx]
        self.pool_labels = y_sorted[self.pool_global_idx]

        self.y_min = float(np.min(self.pool_labels))
        self.y_max = float(np.max(self.pool_labels))

        assert self.pool_designs_raw.shape[0] == self.n_pool
        assert self.pool_designs_normalized.shape[0] == self.n_pool
        assert self.pool_labels.shape[0] == self.n_pool

        n_d2 = max(1, min(self.n_pool - 1, int(round(self.n_pool * (1.0 - self.ratio)))))

        band_low = int(self.lower_ratio * self.n_pool)  # Inclusive left bound.
        band_high = int(self.upper_ratio * self.n_pool)  # Exclusive right bound.
        band_low = max(0, min(band_low, self.n_pool - 1))
        band_high = max(band_low + 1, min(band_high, self.n_pool))

        candidate = list(range(band_low, band_high))
        if n_d2 >= len(candidate):
            d2_idx = candidate[:]
        else:
            gap = len(candidate) / float(n_d2)
            d2_idx = [candidate[int(round(i * gap))] for i in range(n_d2)]
        d2_idx = sorted(set(d2_idx))

        d1_idx = [i for i in range(self.n_pool) if i not in d2_idx]
        d1_idx = sorted(set(d1_idx))

        self.d1_idx = d1_idx
        self.d2_idx = d2_idx
        self._d1_set = set(self.d1_idx)
        self._d2_set = set(self.d2_idx)

        X_normalized = self.pool_designs_normalized.astype(np.float32)
        if X_normalized.ndim > 2:
            X_normalized = X_normalized.reshape(self.n_pool, -1)
        self.X_for_kernel = X_normalized

        self.sigma = self._estimate_sigma(self.X_for_kernel)

        if self.ablation_use_random_neighbors == "random":
            self._build_random_neighbors_and_valid_anchors_keep_d1_d2()
            self._build_random_rl_neighbors_keep_d1_d2()
        elif self.ablation_use_random_neighbors == "d1-d2":
            self._build_neighbors_and_valid_anchors()
            self._build_rl_neighbors()
        else:
            raise ValueError("ablation_use_random_neighbors must be 'd1-d2' or 'random'")

        if self.special_token_type == "special":
            self._design_start_token = DESIGN_START
            self._design_end_token = DESIGN_END
            self._label_start_token = LABEL_START
            self._label_end_token = LABEL_END

            self._design_start_id = self.tokenizer.convert_tokens_to_ids(self._design_start_token)
            self._design_end_id = self.tokenizer.convert_tokens_to_ids(self._design_end_token)
            self._label_start_id = self.tokenizer.convert_tokens_to_ids(self._label_start_token)
            self._label_end_id = self.tokenizer.convert_tokens_to_ids(self._label_end_token)

        elif self.special_token_type == "natural":
            self._design_start_token = (
                "Design:  "  # Two spaces after the colon keep tokenization stable.
            )
            self._design_end_token = " "  # Natural-language design terminator.
            self._label_start_token = "Label: "
            self._label_end_token = ""

            self._design_start_id = self.tokenizer.convert_tokens_to_ids("Design")
            self._design_end_id = self.tokenizer.convert_tokens_to_ids(self._design_end_token)
            self._label_start_id = self.tokenizer.convert_tokens_to_ids("Label")
            self._label_end_id = self.tokenizer.convert_tokens_to_ids(self._label_end_token)
        else:
            raise ValueError("special_token_type must be 'special' or 'natural'")

    def __iter__(self):
        while True:
            yield self._sample_one()

    def _sample_one(self) -> Dict[str, Any]:
        if self.n_pool <= 20:  # Tiny-pool smoke tests and ablations may sample with replacement.
            use_replacement = True
        else:
            use_replacement = False

        if self.mode in {"da", "sft"}:
            j = self._sample_anchor_index()
            neigh_info = self.neighbors[j]
            assert len(neigh_info) >= self.n_few_shot or self.n_pool <= 20, (
                f"[{self.task_name}] Anchor {j} has only {len(neigh_info)} neighbors, "
                f"but n_few_shot={self.n_few_shot}."
            )

            chosen = self._rng.choice(neigh_info, size=self.n_few_shot, replace=use_replacement)
            prompt_indices = [int(c["idx"]) for c in chosen]

            prompt_designs = [self.pool_designs_raw[i] for i in prompt_indices]
            prompt_labels = [float(self.pool_labels[i]) for i in prompt_indices]

            resp_design = self.pool_designs_raw[j]
            resp_label = float(self.pool_labels[j])

            max_prompt_label = max(prompt_labels)
            assert resp_label > max_prompt_label, (
                f"[{self.task_name}] Response label must be > max prompt label, "
                f"got resp={resp_label:.4f}, max_prompt={max_prompt_label:.4f}"
            )
            normalized_resp_label = (resp_label - self.y_min) / (self.y_max - self.y_min)
            normalized_max_prompt = (max_prompt_label - self.y_min) / (self.y_max - self.y_min)
            reward_raw = normalized_resp_label - normalized_max_prompt

        else:
            j = self._sample_anchor_index()

            lower_list = self.rl_lower[j]
            higher_list = self.rl_higher[j]
            y = self.pool_labels.astype(np.float32)

            assert len(lower_list) >= self.n_few_shot or self.n_pool <= 20, (
                f"[{self.task_name}][RL] Anchor {j} has only {len(lower_list)} lower neighbors, "
                f"but n_few_shot={self.n_few_shot}."
            )
            assert (
                len(higher_list) >= 1 or self.n_pool <= 20
            ), f"[{self.task_name}][RL] Anchor {j} has no higher neighbors."

            want_positive = bool(self._rng.integers(0, 2))

            if want_positive:
                K_low = min(self.K_lower_rl, len(lower_list))
                original_candidates = lower_list[:K_low]

                chosen = self._rng.choice(
                    original_candidates, size=self.n_few_shot, replace=use_replacement
                )
                prompt_indices = list(map(int, chosen))
            else:
                K_high = min(self.K_higher_rl, len(higher_list))
                high_candidates = higher_list[:K_high]
                high_idx = int(self._rng.choice(high_candidates))
                y_h = float(y[high_idx])

                rest_pool: List[int] = list(lower_list)
                for idx2 in higher_list:
                    if idx2 == high_idx:
                        continue
                    if float(y[idx2]) < y_h:
                        rest_pool.append(int(idx2))

                assert len(rest_pool) >= self.n_few_shot - 1 or self.n_pool <= 20, (
                    f"[{self.task_name}][RL] Anchor {j} has only {len(rest_pool)} rest candidates "
                    f"below y_h, need {self.n_few_shot - 1}."
                )

                max_rest = min(len(rest_pool), self.K_pool)
                rest_candidates = rest_pool[:max_rest]
                rest = self._rng.choice(
                    rest_candidates, size=self.n_few_shot - 1, replace=use_replacement
                )

                prompt_indices = [high_idx] + list(map(int, rest))
                self._rng.shuffle(prompt_indices)

            prompt_designs = [self.pool_designs_raw[i] for i in prompt_indices]
            prompt_labels = [float(self.pool_labels[i]) for i in prompt_indices]

            resp_design = self.pool_designs_raw[j]
            resp_label = float(self.pool_labels[j])

            max_prompt_label = max(prompt_labels)

            normalized_resp_label = (resp_label - self.y_min) / (self.y_max - self.y_min)
            normalized_max_prompt = (max_prompt_label - self.y_min) / (self.y_max - self.y_min)
            reward_raw = normalized_resp_label - normalized_max_prompt

        pairs_text = "\n".join(
            self._render_pair_block(d, yy) for d, yy in zip(prompt_designs, prompt_labels)
        )

        template_id = int(self._rng.choice(self._template_ids))
        template = self.templates[template_id]
        prompt_text = template.format(pairs=pairs_text)

        response_text = f"Response:\n{self._design_start_token}{self._to_list_token_render(resp_design)}{self._design_end_token}"
        full_text = prompt_text + response_text

        enc = self.tokenizer(full_text, return_tensors="pt")
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        ids = input_ids.tolist()

        start_idx = max(i for i, tok in enumerate(ids) if tok == self._design_start_id)
        end_idx = (
            int(attention_mask.sum().item()) - 1
        )  # Shared end position for both delimiter modes.
        if self.special_token_type == "natural":
            assert ids[start_idx + 1] == 25 and (
                ids[start_idx + 2] == 9812 or ids[start_idx + 2] == 831 or ids[start_idx + 2] == 220
            ), f"tokenization mismatch for 'Design: ', got {ids[start_idx:start_idx+5]}"
            start_idx += 2
        elif self.special_token_type == "special":
            start_idx -= 1  # Special-token mode includes delimiter tokens in the training span.
            end_idx += 1

        resp_enc = self.tokenizer(self._to_list_token_render(resp_design), return_tensors="pt")
        response_token_ids = resp_enc["input_ids"].squeeze(0)

        item: Dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "response_token_ids": response_token_ids,
            "response_start_idx": torch.tensor(start_idx, dtype=torch.long),
            "response_end_idx": torch.tensor(end_idx, dtype=torch.long),
            "prompt_indices": torch.tensor(prompt_indices, dtype=torch.long),
            "anchor_index": torch.tensor(j, dtype=torch.long),
            "prompt_labels": torch.tensor(prompt_labels, dtype=torch.float32),
            "response_label": torch.tensor(resp_label, dtype=torch.float32),
            "reward_raw": torch.tensor(reward_raw, dtype=torch.float32),
            "template_id": torch.tensor(template_id, dtype=torch.long),
            "task_name": self.task_name,
            "pool_global_idx": torch.tensor(self.pool_global_idx, dtype=torch.long),
            "template_str": template,
            "pairs_text": pairs_text,
            "prompt_text": prompt_text,
            "response_text": response_text,
            "full_text": full_text,
        }
        return item

    def _build_neighbors_and_valid_anchors(self) -> None:
        N = self.n_pool
        X = self.X_for_kernel
        y = self.pool_labels.astype(np.float32)

        neighbors: List[List[Dict[str, Any]]] = []
        for j in range(N):
            x_j = X[j]
            y_j = float(y[j])

            candidates = [i for i in self.d1_idx if (i < j and y[i] < y_j)]
            if not candidates:
                neighbors.append([])
                continue

            sims: List[Tuple[int, float]] = []
            for i in candidates:
                x_i = X[i]
                dist2 = float(np.sum((x_i - x_j) ** 2))
                sim = math.exp(-dist2 / (2.0 * self.sigma * self.sigma)) if self.sigma > 0 else 1.0
                sims.append((i, sim))

            sims.sort(key=lambda t: t[1], reverse=True)

            top = sims[: min(self.K_pool, len(sims))]

            neigh_info: List[Dict[str, Any]] = []
            for i, sim in top:
                neigh_info.append({"idx": int(i), "sim": float(sim), "y": float(y[i])})
            neighbors.append(neigh_info)

        self.neighbors = neighbors

        self.valid_anchor_d1 = [j for j in self.d1_idx if len(self.neighbors[j]) >= self.n_few_shot]
        self.valid_anchor_d2 = [j for j in self.d2_idx if len(self.neighbors[j]) >= self.n_few_shot]
        self.valid_anchor_all = self.valid_anchor_d1 + self.valid_anchor_d2

        if self.n_pool <= 20:  # Tiny-pool smoke tests and ablations relax anchor filtering.
            self.valid_anchor_d1 = self.d1_idx[1:]
            self.valid_anchor_d2 = self.d2_idx[:]
            self.valid_anchor_all = self.valid_anchor_d1 + self.valid_anchor_d2

    def _build_rl_neighbors(self) -> None:
        X = self.X_for_kernel
        y = self.pool_labels.astype(np.float32)

        self.rl_lower: Dict[int, List[int]] = {}
        self.rl_higher: Dict[int, List[int]] = {}
        self.rl_valid_anchors: List[int] = []

        for j in self.d2_idx:
            x_j = X[j]
            y_j = float(y[j])

            sims: List[Tuple[int, float]] = []
            for i in self.d1_idx:
                x_i = X[i]
                dist2 = float(np.sum((x_i - x_j) ** 2))
                sim = math.exp(-dist2 / (2.0 * self.sigma * self.sigma)) if self.sigma > 0 else 1.0
                sims.append((i, sim))

            sims.sort(key=lambda t: t[1], reverse=True)

            lower_list: List[int] = []
            higher_list: List[int] = []

            for idx, sim in sims:
                y_i = float(y[idx])
                if y_i < y_j and len(lower_list) < self.K_lower_rl:
                    lower_list.append(int(idx))
                elif y_i > y_j and len(higher_list) < self.K_higher_rl:
                    higher_list.append(int(idx))

                if len(lower_list) >= self.K_lower_rl and len(higher_list) >= self.K_higher_rl:
                    break

            self.rl_lower[j] = lower_list
            self.rl_higher[j] = higher_list

            if len(lower_list) >= self.n_few_shot and len(higher_list) >= 1:
                self.rl_valid_anchors.append(int(j))

            if self.n_pool <= 20:  # Tiny-pool smoke tests and ablations relax anchor filtering.
                self.rl_valid_anchors.append(int(j))

    def _build_random_neighbors_and_valid_anchors_keep_d1_d2(self) -> None:
        N = self.n_pool
        X = self.X_for_kernel
        y = self.pool_labels.astype(np.float32)

        neighbors: List[List[Dict[str, Any]]] = []
        for j in range(N):
            x_j = X[j]
            y_j = float(y[j])

            candidates = [i for i in self.d1_idx if (i < j and y[i] < y_j)]
            if not candidates:
                neighbors.append([])
                continue

            rng_neigh = np.random.default_rng(self.seed + 12345)
            rng_neigh.shuffle(candidates)
            candidates = candidates[: self.K_pool]

            neigh_info: List[Dict[str, Any]] = []
            for i in candidates:
                neigh_info.append({"idx": int(i), "sim": None, "y": float(y[i])})
            neighbors.append(neigh_info)

        self.neighbors = neighbors

        self.valid_anchor_d1 = [j for j in self.d1_idx if len(self.neighbors[j]) >= self.n_few_shot]
        self.valid_anchor_d2 = [j for j in self.d2_idx if len(self.neighbors[j]) >= self.n_few_shot]
        self.valid_anchor_all = self.valid_anchor_d1 + self.valid_anchor_d2

    def _build_random_rl_neighbors_keep_d1_d2(self) -> None:
        X = self.X_for_kernel
        y = self.pool_labels.astype(np.float32)

        self.rl_lower: Dict[int, List[int]] = {}
        self.rl_higher: Dict[int, List[int]] = {}
        self.rl_valid_anchors: List[int] = []

        for j in self.d2_idx:
            x_j = X[j]
            y_j = float(y[j])

            candidates = [i for i in self.d1_idx]
            rng_neigh = np.random.default_rng(self.seed + 12345)
            rng_neigh.shuffle(candidates)

            lower_list: List[int] = []
            higher_list: List[int] = []

            for idx in candidates:
                y_i = float(y[idx])
                if y_i < y_j and len(lower_list) < self.K_lower_rl:
                    lower_list.append(int(idx))
                elif y_i > y_j and len(higher_list) < self.K_higher_rl:
                    higher_list.append(int(idx))

                if len(lower_list) >= self.K_lower_rl and len(higher_list) >= self.K_higher_rl:
                    break

            self.rl_lower[j] = lower_list
            self.rl_higher[j] = higher_list

            if len(lower_list) >= self.n_few_shot and len(higher_list) >= 1:
                self.rl_valid_anchors.append(int(j))

    def _sample_anchor_index(self) -> int:
        if self.mode == "sft":
            candidates = self.valid_anchor_d2
        elif self.mode == "da":
            candidates = self.valid_anchor_all
        elif self.mode == "rl":
            candidates = self.rl_valid_anchors
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        assert len(candidates) > 0, f"[{self.task_name}] No valid anchors for mode={self.mode}."
        return int(self._rng.choice(candidates))

    @staticmethod
    def _evenly_spaced_indices(n_total: int, n_select: int) -> np.ndarray:
        if n_select <= 1:
            return np.array([0], dtype=int)
        idx = np.linspace(0, n_total - 1, num=n_select)
        idx = np.round(idx).astype(int)
        idx = np.clip(idx, 0, n_total - 1)
        return idx

    @staticmethod
    def _estimate_sigma(X: np.ndarray) -> float:
        N = X.shape[0]
        dists: List[float] = []
        for i in range(N):
            for j in range(i + 1, N):
                dists.append(float(np.linalg.norm(X[i] - X[j])))

        if not dists:
            return 1.0

        arr = np.array(dists, dtype=np.float32)
        sigma = float(np.median(arr))
        if sigma <= 0.0:
            sigma = float(arr.mean() + 1e-6)
        if sigma <= 0.0:
            sigma = 1.0
        return sigma

    @staticmethod
    def _estimate_lambda(X, y, n_pairs=5000) -> float:
        rng = np.random.default_rng()
        N = X.shape[0]
        if N < 2:
            return 1.0
        dx2_list = []
        dy2_list = []
        for _ in range(n_pairs):
            a = int(rng.integers(0, N))
            b = int(rng.integers(0, N - 1))
            y_a_norm = (y[a] - np.min(y)) / (np.max(y) - np.min(y) + 1e-12)
            y_b_norm = (y[b] - np.min(y)) / (np.max(y) - np.min(y) + 1e-12)
            if b >= a:
                b += 1
            d = X[a] - X[b]
            dx2_list.append(float(np.sum(d * d)))
            dy = float(y_a_norm - y_b_norm)
            dy2_list.append(dy * dy)
        mdx2 = float(np.median(dx2_list))
        mdy2 = float(np.median(dy2_list))
        if mdy2 < 1e-12:
            return 0.0  # Disable the y-distance term when y does not vary.
        return mdx2 / mdy2

    @staticmethod
    def _format_signed_float(v: float, *, decimals: int = 3, min_int_digits: int = 3) -> str:
        """
        Format a continuous value as sign + zero-padded integer part + fixed decimals.

        Integer parts wider than the default are never truncated.
        """
        sign = "+" if float(v) >= 0 else "-"
        s = f"{abs(float(v)):.{decimals}f}"
        int_part, frac_part = s.split(".")
        int_part = int_part.zfill(min_int_digits)
        return f"{sign}{int_part}.{frac_part}"

    def _to_list_token_render(self, seq: Sequence[Any]) -> str:
        """
        Render a design according to task type.

        - TFBind*: 0/1/2/3 -> A/C/G/T.
        - Ant/DKitty: signed continuous values with fixed decimals.
        """
        name = str(self.task_name)

        if name.startswith("TFBind"):
            mapping = {"0": "A", "1": "C", "2": "G", "3": "T"}
            seq_str = [mapping[str(int(v))] for v in seq]
            inner = ", ".join(f"'{c}'" for c in seq_str)
            return f"[{inner}]"

        if name == "AntMorphology-Exact-v0":
            vals = [float(v) for v in seq]
            inner = ", ".join(
                self._format_signed_float(v, decimals=3, min_int_digits=3) for v in vals
            )
            return f"[ {inner} ]"

        if name == "DKittyMorphology-Exact-v0":
            vals = [float(v) for v in seq]
            inner = ", ".join(
                self._format_signed_float(v, decimals=3, min_int_digits=3) for v in vals
            )
            return f"[ {inner} ]"

        raise ValueError(f"Unsupported task: {name}")

    def _render_pair_block(self, design: Sequence[Any], label: float) -> str:
        label_str = self._format_signed_float(label, decimals=3, min_int_digits=3)
        design_block = f"{self._design_start_token}{self._to_list_token_render(design)}{self._design_end_token}"
        label_block = f"{self._label_start_token}{label_str}{self._label_end_token}"
        return f"{design_block} {label_block}"
