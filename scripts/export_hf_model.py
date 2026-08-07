#!/usr/bin/env python3
"""Export a canonical DiBO checkpoint as a self-contained HF safetensors model."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model.dllm import (  # noqa: E402
    DEFAULT_MODEL_ID,
    LLADA_MODEL_REVISION,
    load_dibo_checkpoint,
)
from src.release.hf_artifacts import (  # noqa: E402
    GITHUB_REPOSITORY,
    PAPER_ID,
    PAPER_URL,
    configure_dibo_export,
    copy_export_remote_code,
    finalize_dibo_export_config,
    git_commit,
    model_parameter_count,
    parameter_dtype_counts,
    release_task_spec,
    sha256_file,
    standard_export_weight_metadata,
    write_sha256sums,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export one strict DiBO .pt checkpoint as a Transformers safetensors model."
    )
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--repo-id", type=str, default=None)
    parser.add_argument("--base-model", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--base-revision", type=str, default=LLADA_MODEL_REVISION)
    parser.add_argument("--max-shard-size", type=str, default="5GB")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def stage_original_checkpoint(source: Path, output: Path, filename: str) -> Path:
    staged = output / filename
    if staged.exists():
        if sha256_file(staged) != sha256_file(source):
            raise FileExistsError(
                f"Existing staged checkpoint differs from source: {staged}"
            )
        return staged
    try:
        os.link(source, staged)
    except OSError:
        shutil.copy2(source, staged)
    return staged


def clear_existing_standard_export(output: Path, force: bool) -> None:
    existing = list(output.glob("model*.safetensors"))
    existing.extend(
        output / name
        for name in (
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "added_tokens.json",
            "configuration_llada.py",
            "modeling_llada.py",
            "modeling_dibo_llada.py",
            "__init__.py",
            "UPSTREAM_NOTICE.md",
        )
        if (output / name).exists()
    )
    if existing and not force:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(
            f"Standard export files already exist in {output}: {names}. Pass --force to replace them."
        )
    for path in existing:
        path.unlink()


def main() -> None:
    args = parse_args()
    source = args.source_checkpoint.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Source checkpoint not found: {source}")
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    task = release_task_spec(args.task)
    clear_existing_standard_export(output, bool(args.force))

    staged_original = stage_original_checkpoint(
        source, output, str(task["original_checkpoint_filename"])
    )
    source_sha256 = sha256_file(source)
    print(f"[Export] Strictly loading {source.name}")
    model, tokenizer = load_dibo_checkpoint(
        source,
        device=str(args.device),
        base_model_id=str(args.base_model),
        base_model_revision=str(args.base_revision),
    )
    model.eval()
    dimensions = configure_dibo_export(model, tokenizer)
    print(
        "[Export] Writing BF16 safetensors "
        f"({dimensions['input_embedding_size']} input / "
        f"{dimensions['output_embedding_size']} output vocabulary rows)"
    )
    model.save_pretrained(
        output,
        safe_serialization=True,
        max_shard_size=str(args.max_shard_size),
    )
    finalize_dibo_export_config(output)
    tokenizer.save_pretrained(output)
    copied_code = copy_export_remote_code(model, output)

    weights = standard_export_weight_metadata(output)
    if not weights:
        raise RuntimeError("save_pretrained did not produce any safetensors weights.")
    manifest = {
        "artifact_name": task["artifact_name"],
        "task": task["task"],
        "paper": {"identifier": PAPER_ID, "url": PAPER_URL},
        "github_repository": GITHUB_REPOSITORY,
        "github_commit": git_commit(),
        "base_model": str(args.base_model),
        "base_model_revision": str(args.base_revision),
        "training_stages": ["DA", "SFT", "RL"],
        "delimiter_tokens": list(task["delimiter_tokens"]),
        "delimiter_token_ids": dimensions["delimiter_token_ids"],
        "original_checkpoint": {
            "path": staged_original.name,
            "format": "PyTorch checkpoint dictionary",
            "state_dict_key": "model",
            "byte_size": int(staged_original.stat().st_size),
            "sha256": source_sha256,
            "canonical_paper_artifact": True,
        },
        "transformers_export": {
            "format": "Transformers safetensors",
            "derived_from_original_checkpoint_sha256": source_sha256,
            "model_class": "DiBOLLaDAModelLM",
            "config_class": "LLaDAConfig",
            "auto_class": "AutoModel",
            "tokenizer_class": type(tokenizer).__name__,
            "input_embedding_size": dimensions["input_embedding_size"],
            "output_embedding_size": dimensions["output_embedding_size"],
            "parameter_count": model_parameter_count(model),
            "dtype_counts": parameter_dtype_counts(model),
            "weight_files": weights,
            "total_weight_bytes": int(sum(item["byte_size"] for item in weights)),
            "remote_code_files": copied_code,
            "export_tool": "scripts/export_hf_model.py",
            "export_git_commit": git_commit(),
        },
    }
    (output / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_sha256sums(output, output.iterdir())
    print(f"[Export] Complete: {output}")


if __name__ == "__main__":
    main()
