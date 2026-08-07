from __future__ import annotations

from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from eval import build_parser, resolve_model_source
from src.model.dllm import (
    DIBO_DELIMITER_TOKENS,
    load_dibo_checkpoint_from_hub,
    load_dibo_pretrained,
    validate_dibo_model_and_tokenizer,
)
from src.release.hf_artifacts import (
    REMOTE_WRAPPER_PATH,
    configure_dibo_export,
    finalize_dibo_export_config,
    load_release_spec,
    render_model_card,
    release_task_spec,
)
from scripts.validate_hf_export import auto_map_target_matches


class FakeTokenizer:
    def __init__(self) -> None:
        self.tokens = {token: 126349 + index for index, token in enumerate(DIBO_DELIMITER_TOKENS)}
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"

    def __len__(self) -> int:
        return 126353

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.tokens[token]

    def encode(self, token: str, add_special_tokens: bool = False):
        del add_special_tokens
        return [self.tokens[token]]


class FakeModel:
    def __init__(self, input_rows: int = 126353, output_rows: int = 126464) -> None:
        self.input = SimpleNamespace(weight=torch.empty((input_rows, 2)))
        self.output = SimpleNamespace(weight=torch.empty((output_rows, 2)))
        self.config = SimpleNamespace(vocab_size=input_rows, embedding_size=126464)

    def get_input_embeddings(self):
        return self.input

    def get_output_embeddings(self):
        return self.output

    def to(self, device: str):
        del device
        return self


class HFReleaseTest(unittest.TestCase):
    def test_public_release_spec_is_complete(self) -> None:
        spec = load_release_spec()
        self.assertEqual(spec["auto_class"], "AutoModel")
        self.assertEqual(tuple(spec["delimiter_tokens"]), DIBO_DELIMITER_TOKENS)
        self.assertEqual(len(spec["tasks"]), 4)
        for task in spec["tasks"]:
            task_spec = release_task_spec(task)
            self.assertTrue(task_spec["repository_id"].startswith("zpointsun/DiBO-"))
            self.assertTrue(task_spec["original_checkpoint_filename"].endswith(".pt"))

    def test_export_configuration_records_asymmetric_vocabularies(self) -> None:
        model = FakeModel()
        tokenizer = FakeTokenizer()
        result = configure_dibo_export(model, tokenizer)
        self.assertEqual(result["input_embedding_size"], 126353)
        self.assertEqual(result["output_embedding_size"], 126464)
        self.assertEqual(model.config.vocab_size, 126353)
        self.assertEqual(model.config.embedding_size, 126353)
        self.assertEqual(model.config.dibo_output_embedding_size, 126464)
        self.assertEqual(
            model.config.auto_map["AutoModel"],
            "modeling_dibo_llada.DiBOLLaDAModelLM",
        )

    def test_model_validation_accepts_the_canonical_asymmetric_projection(self) -> None:
        ids = validate_dibo_model_and_tokenizer(FakeModel(), FakeTokenizer())
        self.assertEqual(ids, (126349, 126350, 126351, 126352))

    def test_wrapper_is_a_tracked_export_template(self) -> None:
        source = REMOTE_WRAPPER_PATH.read_text(encoding="utf-8")
        self.assertIn("class DiBOLLaDAModelLM", source)
        self.assertIn("dibo_output_embedding_size", source)

    def test_finalized_export_config_uses_wrapper_without_local_path(self) -> None:
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(
                json.dumps(
                    {
                        "_name_or_path": "/private/path",
                        "architectures": ["LLaDAModelLM"],
                        "auto_map": {"AutoModel": "modeling_llada.LLaDAModelLM"},
                        "dibo_input_embedding_size": 126353,
                        "dibo_output_embedding_size": 126464,
                    }
                ),
                encoding="utf-8",
            )
            finalize_dibo_export_config(directory)
            config = json.loads(path.read_text(encoding="utf-8"))
        self.assertNotIn("_name_or_path", config)
        self.assertEqual(config["architectures"], ["DiBOLLaDAModelLM"])
        self.assertEqual(
            config["auto_map"]["AutoModel"],
            "modeling_dibo_llada.DiBOLLaDAModelLM",
        )

    def test_eval_source_selection_is_unambiguous(self) -> None:
        parser = build_parser()
        original = parser.parse_args(["--seeds", "3", "--checkpoint_path", "checkpoint.pt"])
        self.assertEqual(resolve_model_source(original), "original_checkpoint")
        standard = parser.parse_args(
            ["--seeds", "3", "--model_name_or_path", "zpointsun/DiBO-TFBind8", "--model_revision", "v1.1.0"]
        )
        self.assertEqual(resolve_model_source(standard), "pretrained_export")
        ambiguous = parser.parse_args(
            ["--seeds", "3", "--checkpoint_path", "one.pt", "--checkpoint_dir", "checkpoints"]
        )
        with self.assertRaisesRegex(ValueError, "either --checkpoint_path or --checkpoint_dir"):
            resolve_model_source(ambiguous)
        invalid_revision = parser.parse_args(["--seeds", "3", "--model_revision", "v1.1.0"])
        with self.assertRaisesRegex(ValueError, "--model_revision"):
            resolve_model_source(invalid_revision)

    def test_pretrained_loader_uses_native_auto_classes_without_checkpoint_download(self) -> None:
        tokenizer = FakeTokenizer()
        model = FakeModel()
        with (
            patch("src.model.dllm.AutoTokenizer.from_pretrained", return_value=tokenizer) as load_tokenizer,
            patch("src.model.dllm.AutoModel.from_pretrained", return_value=model) as load_model,
        ):
            loaded_model, loaded_tokenizer = load_dibo_pretrained(
                "zpointsun/DiBO-TFBind8", revision="v1.1.0", device="cpu"
            )
        self.assertIs(loaded_model, model)
        self.assertIs(loaded_tokenizer, tokenizer)
        self.assertEqual(load_tokenizer.call_args.kwargs["revision"], "v1.1.0")
        self.assertTrue(load_model.call_args.kwargs["use_safetensors"])
        self.assertNotIn("add_special_tokens", load_tokenizer.call_args.kwargs)

    def test_hub_checkpoint_loader_reuses_canonical_path(self) -> None:
        with (
            patch("huggingface_hub.hf_hub_download", return_value="/tmp/checkpoint.pt") as download,
            patch("src.model.dllm.load_dibo_checkpoint", return_value=("model", "tokenizer")) as load_checkpoint,
        ):
            result = load_dibo_checkpoint_from_hub(
                "zpointsun/DiBO-TFBind8", "dibo_tfbind8_final.pt", revision="v1.1.0", device="cpu"
            )
        self.assertEqual(result, ("model", "tokenizer"))
        self.assertEqual(download.call_args.kwargs["filename"], "dibo_tfbind8_final.pt")
        self.assertEqual(load_checkpoint.call_args.kwargs["device"], "cpu")

    def test_rendered_card_describes_both_artifact_formats(self) -> None:
        card = render_model_card(
            {
                "artifact_name": "DiBO-TFBind8",
                "task": "TFBind8-Exact-v0",
                "base_model": "GSAI-ML/LLaDA-8B-Instruct",
                "base_model_revision": "revision",
                "paper": {"url": "https://huggingface.co/papers/2603.17919"},
                "github_repository": "https://github.com/zpointS/DiBO",
                "delimiter_tokens": list(DIBO_DELIMITER_TOKENS),
                "original_checkpoint": {
                    "path": "dibo_tfbind8_final.pt",
                    "byte_size": 1,
                    "sha256": "abc",
                    "state_dict_key": "model",
                },
                "transformers_export": {
                    "model_class": "DiBOLLaDAModelLM",
                    "auto_class": "AutoModel",
                    "tokenizer_class": "PreTrainedTokenizerFast",
                    "input_embedding_size": 126353,
                    "output_embedding_size": 126464,
                    "total_weight_bytes": 1,
                    "weight_files": [{"filename": "model.safetensors", "byte_size": 1, "sha256": "def"}],
                },
                "validation": {"weight_equality": True},
            },
            release_tag="v1.1.0",
        )
        self.assertIn("Original PyTorch checkpoint", card)
        self.assertIn("Transformers/safetensors export", card)
        self.assertIn("AutoModel.from_pretrained", card)
        self.assertIn("dibo_tfbind8_final.pt", card)
        self.assertIn("four DiBO delimiter tokens", card)
        self.assertIn("pipeline_tag: other", card)
        self.assertIn("  - llm", card)
        self.assertIn("hf_hub_download", card)
        self.assertIn("[DiBO repository](https://github.com/zpointS/DiBO)", card)
        self.assertNotIn("blob/main/docs/installation.md", card)
        self.assertNotIn("## Artifact details", card)

    def test_remote_auto_map_prefix_is_accepted(self) -> None:
        target = "modeling_dibo_llada.DiBOLLaDAModelLM"
        self.assertTrue(auto_map_target_matches(target, target))
        self.assertTrue(
            auto_map_target_matches("zpointsun/DiBO-TFBind8--" + target, target)
        )
        self.assertFalse(auto_map_target_matches("modeling_llada.LLaDAModelLM", target))


if __name__ == "__main__":
    unittest.main()
