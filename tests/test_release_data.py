from __future__ import annotations

from contextlib import redirect_stderr
from io import StringIO
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from eval import (
    MAX_EVAL_SEED,
    SUPPORTED_TASKS,
    build_parser,
    extract_float_design,
    generate_candidates,
    summarize_normalized_max,
    tf10_predict,
)
from scripts.compare_task_bundles import semantic_diffs
from src.dataset.task_bundle import build_task_bundle, save_task_bundle, validate_bundle


ROOT = Path(__file__).resolve().parents[1]


class ReleaseDataTest(unittest.TestCase):
    def test_all_canonical_bundles_validate(self) -> None:
        for task_name in SUPPORTED_TASKS:
            path = ROOT / "data/task_bundles" / f"{task_name}.npz"
            bundle = validate_bundle(path, expected_task=task_name)
            self.assertGreater(len(bundle.y_oracle_sorted), 0)

    def test_rebuildable_tasks_match_semantically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for task_name in (
                "TFBind8-Exact-v0",
                "AntMorphology-Exact-v0",
                "DKittyMorphology-Exact-v0",
            ):
                rebuilt = build_task_bundle(
                    task_name,
                    raw_dir=ROOT / "data/raw",
                    relabeled_dir=ROOT / "data/relabeled",
                )
                candidate_path = save_task_bundle(rebuilt, directory)
                reference_path = ROOT / "data/task_bundles" / f"{task_name}.npz"
                with (
                    np.load(reference_path, allow_pickle=False) as reference,
                    np.load(candidate_path, allow_pickle=False) as candidate,
                ):
                    self.assertEqual(
                        semantic_diffs(reference, candidate),
                        [],
                        task_name,
                    )

    def test_tfbind10_lookup(self) -> None:
        lookup = ROOT / "data/raw/TFBind10-Exact-v0.npz"
        with np.load(lookup, allow_pickle=False) as data:
            indices = np.array([0, len(data["x"]) // 2, len(data["x"]) - 1])
            query = data["x"][indices]
            expected = data["y"][indices].astype(np.float32)
        np.testing.assert_array_equal(tf10_predict(query, lookup), expected)

    def test_eval_requires_explicit_seeds(self) -> None:
        parser = build_parser()
        seed_action = next(
            action for action in parser._actions if action.dest == "seeds"
        )
        self.assertTrue(seed_action.required)
        self.assertIsNone(seed_action.default)
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            parser.parse_args([])

    def test_eval_accepts_caller_selected_seeds(self) -> None:
        args = build_parser().parse_args(["--seeds", str(MAX_EVAL_SEED)])
        self.assertEqual(args.seeds, [MAX_EVAL_SEED])
        self.assertEqual(args.max_attempts, 1000)

    def test_eval_rejects_out_of_range_seeds(self) -> None:
        for value in (str(-MAX_EVAL_SEED), str(MAX_EVAL_SEED + 1)):
            with (
                self.subTest(value=value),
                redirect_stderr(StringIO()),
                self.assertRaises(SystemExit),
            ):
                build_parser().parse_args(["--seeds", value])

    def test_eval_rejects_partial_candidate_sets(self) -> None:
        args = SimpleNamespace(num_candidates=1, max_attempts=0)
        dataset = SimpleNamespace(pool_designs_raw=np.zeros((1, 8)))
        with self.assertRaisesRegex(RuntimeError, "requires 1"):
            generate_candidates(
                args,
                torch.nn.Linear(1, 1),
                tokenizer=None,
                task_name="TFBind8-Exact-v0",
                ds=dataset,
                seed=MAX_EVAL_SEED,
            )

    def test_eval_rejects_nonfinite_continuous_designs(self) -> None:
        dataset = SimpleNamespace(
            _design_start_token="|design-start|",
            _design_end_token="|design-end|",
        )
        text = "Response: |design-start|[nan inf]|design-end|"
        self.assertIsNone(extract_float_design(dataset, text, expected_dim=2))

    def test_eval_summary_uses_population_standard_deviation(self) -> None:
        rows = [
            {
                "checkpoint": "checkpoint.pt",
                "task": "TFBind8-Exact-v0",
                "norm_max": value,
            }
            for value in (0.8, 0.9, 1.0)
        ]
        summary = summarize_normalized_max(rows)
        self.assertEqual(summary[0]["n_seeds"], 3)
        self.assertAlmostEqual(summary[0]["norm_max_mean"], 0.9)
        self.assertAlmostEqual(
            summary[0]["norm_max_standard_deviation"],
            float(np.std([0.8, 0.9, 1.0], ddof=0)),
        )

    def test_reward_statistics_cover_four_tasks(self) -> None:
        reward = json.loads(
            (ROOT / "data/reward_stats/reward_stats.json").read_text(encoding="utf-8")
        )
        self.assertEqual(set(reward["tasks"]), set(SUPPORTED_TASKS))
        self.assertEqual(reward["meta"]["n_samples_per_task"], 100000)
        self.assertEqual(reward["meta"]["lower_ratio"], 0.7)
        self.assertEqual(reward["meta"]["upper_ratio"], 0.95)


if __name__ == "__main__":
    unittest.main()
