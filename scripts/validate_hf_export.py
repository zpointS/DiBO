#!/usr/bin/env python3
"""Validate a DiBO safetensors export against its canonical .pt checkpoint."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval import (  # noqa: E402
    build_dataset_prompt_spec,
    build_eval_dataset,
    evaluate_designs,
    extract_dna_design,
    extract_float_design,
    generate_candidates,
    greedy_fill,
    serialize_design,
    set_seed,
)
from src.model.dllm import (  # noqa: E402
    DEFAULT_MODEL_ID,
    DIBO_DELIMITER_TOKENS,
    LLADA_MODEL_REVISION,
    dibo_delimiter_token_ids,
    load_dibo_checkpoint,
    load_dibo_pretrained,
)
from src.release.hf_artifacts import (  # noqa: E402
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
        description="Compare one canonical DiBO checkpoint with a Transformers export."
    )
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--smoke-seed", type=int, required=True)
    parser.add_argument("--run-task-smoke", action="store_true")
    parser.add_argument("--smoke-max-attempts", type=int, default=1000)
    parser.add_argument("--remote-repo-id", type=str, default=None)
    parser.add_argument("--remote-revision", type=str, default=None)
    parser.add_argument("--remote-cache-dir", type=Path, default=None)
    return parser.parse_args()


def tensor_digest(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().view(torch.uint8).cpu()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def state_metadata(model: torch.nn.Module) -> Dict[str, Dict[str, Any]]:
    return {
        name: {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "sha256": tensor_digest(tensor),
        }
        for name, tensor in model.state_dict().items()
    }


def tokenizer_metadata(tokenizer: Any) -> Dict[str, Any]:
    vocabulary = tokenizer.get_vocab()
    encoded = tokenizer.encode(
        "Response:\n|design-start|ACGT|design-end|", add_special_tokens=False
    )
    return {
        "class": type(tokenizer).__name__,
        "length": int(len(tokenizer)),
        "vocabulary_sha256": hashlib.sha256(
            json.dumps(vocabulary, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "delimiter_token_ids": list(dibo_delimiter_token_ids(tokenizer)),
        "additional_special_tokens": list(tokenizer.additional_special_tokens),
        "representative_encoding": list(encoded),
        "representative_decoding": tokenizer.decode(encoded, skip_special_tokens=False),
    }


def eval_args(task: str, seed: int, max_attempts: int) -> SimpleNamespace:
    return SimpleNamespace(
        tasks=[task],
        seeds=[seed],
        num_candidates=1,
        max_attempts=max_attempts,
        checkpoint_path=None,
        checkpoint_dir=None,
        checkpoint_name_template="optim_step={step}_final.pt",
        steps=[128],
        model_name_or_path=DEFAULT_MODEL_ID,
        model_revision=None,
        device="cpu",
        bundle_dir="data/task_bundles",
        prompts_dir="data/prompts",
        normalization_ranges="data/normalization_ranges.json",
        tf10_lookup="data/raw/TFBind10-Exact-v0.npz",
        output_json=None,
        save_details=False,
        no_normalize=False,
        prompt_source="dataset",
        seed_transform="identity",
        dataset_mode="da",
        special_token_type="special",
        ablation_use_random_neighbors="d1-d2",
        ablation_use_high_or_low_pool="evenly",
        n_pool=500,
        n_few_shot=7,
        k_pool=50,
        ratio=0.8,
        num_template=[8, 2],
    )


@torch.no_grad()
def probe_forward(model: torch.nn.Module, tokenizer: Any, device: str) -> Dict[str, Any]:
    encoded = tokenizer(
        "Response:\n|design-start|ACGT|design-end|", return_tensors="pt"
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.detach().cpu()
    return {
        "shape": list(logits.shape),
        "dtype": str(logits.dtype).removeprefix("torch."),
        "sha256": tensor_digest(logits),
        "argmax": logits.argmax(dim=-1).tolist(),
        "logits": logits,
    }


def candidate_snapshot(
    model: torch.nn.Module, tokenizer: Any, task: str, seed: int, device: str
) -> Dict[str, Any]:
    args = eval_args(task, seed, max_attempts=1)
    set_seed(seed)
    dataset = build_eval_dataset(args, task, tokenizer, seed=seed)
    spec = build_dataset_prompt_spec(dataset, tokenizer)
    text = greedy_fill(model, tokenizer, spec, device)
    expected_dim = int(dataset.pool_designs_raw.shape[1])
    if task.startswith("TFBind"):
        design = extract_dna_design(dataset, text, expected_len=expected_dim)
    else:
        design = extract_float_design(dataset, text, expected_dim=expected_dim)
    return {
        "decoded_response_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "parsed_design": serialize_design(design) if design is not None else None,
        "valid": design is not None,
    }


def task_smoke(
    model: torch.nn.Module,
    tokenizer: Any,
    task: str,
    seed: int,
    max_attempts: int,
) -> Dict[str, Any]:
    args = eval_args(task, seed, max_attempts=max_attempts)
    set_seed(seed)
    dataset = build_eval_dataset(args, task, tokenizer, seed=seed)
    designs = generate_candidates(args, model, tokenizer, task, dataset, seed=seed)
    statistics = evaluate_designs(args, task, designs)
    return {
        "candidate_count": len(designs),
        "valid_candidates": len(designs),
        "statistics": statistics,
    }


def compare_state(
    expected: Dict[str, torch.Tensor], model: torch.nn.Module
) -> Tuple[bool, bool, bool, bool]:
    actual = model.state_dict()
    key_match = set(expected) == set(actual)
    if not key_match:
        return False, False, False, False
    shape_match = all(expected[name].shape == actual[name].shape for name in expected)
    dtype_match = all(expected[name].dtype == actual[name].dtype for name in expected)
    value_match = all(torch.equal(expected[name].cpu(), actual[name].cpu()) for name in expected)
    return key_match, shape_match, dtype_match, value_match


def auto_map_target_matches(actual: Any, expected: str) -> bool:
    """Accept Transformers' runtime ``<repo>--<module>`` remote-code prefix."""
    return isinstance(actual, str) and (
        actual == expected or actual.endswith(f"--{expected}")
    )


def load_remote_checkpoint(
    repo_id: str, filename: str, revision: str, cache_dir: Path
) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            cache_dir=str(cache_dir),
        )
    )


def validate_pair(
    *,
    source_checkpoint: Path,
    export_loader: Callable[[], Tuple[torch.nn.Module, Any]],
    task: str,
    device: str,
    seed: int,
    run_task_smoke: bool,
    smoke_max_attempts: int,
) -> Dict[str, Any]:
    source_model, source_tokenizer = load_dibo_checkpoint(source_checkpoint, device=device)
    source_model.eval()
    source_tokenizer_info = tokenizer_metadata(source_tokenizer)
    source_model_class = type(source_model).__name__
    source_config_class = type(source_model.config).__name__
    source_dtype_counts = parameter_dtype_counts(source_model)
    source_input_shape = list(source_model.get_input_embeddings().weight.shape)
    source_output_shape = list(source_model.get_output_embeddings().weight.shape)
    source_forward = probe_forward(source_model, source_tokenizer, device)
    source_candidate = candidate_snapshot(source_model, source_tokenizer, task, seed, device)
    source_smoke = (
        task_smoke(source_model, source_tokenizer, task, seed, smoke_max_attempts)
        if run_task_smoke
        else None
    )
    source_state = torch.load(str(source_checkpoint), map_location="cpu")["model"]
    del source_model, source_tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    export_model, export_tokenizer = export_loader()
    export_model.eval()
    export_tokenizer_info = tokenizer_metadata(export_tokenizer)
    key_match, shape_match, dtype_match, value_match = compare_state(source_state, export_model)
    export_forward = probe_forward(export_model, export_tokenizer, device)
    export_candidate = candidate_snapshot(export_model, export_tokenizer, task, seed, device)
    export_smoke = (
        task_smoke(export_model, export_tokenizer, task, seed, smoke_max_attempts)
        if run_task_smoke
        else None
    )
    forward_parity = (
        source_forward["shape"] == export_forward["shape"]
        and source_forward["dtype"] == export_forward["dtype"]
        and source_forward["sha256"] == export_forward["sha256"]
        and source_forward["argmax"] == export_forward["argmax"]
    )
    candidate_parity = source_candidate == export_candidate
    task_parity = source_smoke == export_smoke
    config_match = (
        int(getattr(export_model.config, "vocab_size")) == source_input_shape[0]
        and int(getattr(export_model.config, "dibo_input_embedding_size"))
        == source_input_shape[0]
        and int(getattr(export_model.config, "dibo_output_embedding_size"))
        == source_output_shape[0]
        and auto_map_target_matches(
            export_model.config.auto_map.get("AutoModel"),
            "modeling_dibo_llada.DiBOLLaDAModelLM",
        )
    )
    result = {
        "strict_source_load": True,
        "tokenizer_match": source_tokenizer_info == export_tokenizer_info,
        "state_dict_key_match": key_match,
        "tensor_shape_match": shape_match,
        "tensor_dtype_match": dtype_match,
        "tensor_value_match": value_match,
        "config_match": config_match,
        "forward_parity": forward_parity,
        "candidate_generation_parity": candidate_parity,
        "task_smoke_result": {
            "ran": run_task_smoke,
            "source": source_smoke,
            "export": export_smoke,
            "parity": task_parity if run_task_smoke else None,
        },
        "source_tokenizer": source_tokenizer_info,
        "export_tokenizer": export_tokenizer_info,
        "source_model_class": source_model_class,
        "source_config_class": source_config_class,
        "source_dtype_counts": source_dtype_counts,
        "source_input_embedding_shape": source_input_shape,
        "source_output_vocab_shape": source_output_shape,
        "source_forward": {k: v for k, v in source_forward.items() if k != "logits"},
        "export_forward": {k: v for k, v in export_forward.items() if k != "logits"},
        "source_candidate": source_candidate,
        "export_candidate": export_candidate,
        "export_model_class": type(export_model).__name__,
        "export_config_class": type(export_model.config).__name__,
        "export_config_auto_map": dict(getattr(export_model.config, "auto_map", {})),
        "final_vocab_size": int(len(export_tokenizer)),
        "input_embedding_shape": list(export_model.get_input_embeddings().weight.shape),
        "output_vocab_shape": list(export_model.get_output_embeddings().weight.shape),
        "parameter_count": model_parameter_count(export_model),
        "exported_dtype_counts": parameter_dtype_counts(export_model),
    }
    del source_state, export_model, export_tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def update_manifest(export_dir: Path, report_path: Path, report: Dict[str, Any]) -> None:
    manifest_path = export_dir / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # The manifest identifies the code revision that produced the final report.
    manifest["github_commit"] = git_commit()
    validation = report["local"]
    remote = report.get("remote", {})
    manifest["validation"] = {
        "report_file": report_path.name,
        "strict_load": validation["strict_source_load"],
        "tokenizer_equality": validation["tokenizer_match"],
        "config_equality": validation["config_match"],
        "weight_equality": validation["tensor_value_match"],
        "forward_parity": validation["forward_parity"],
        "candidate_parity": validation["candidate_generation_parity"],
        "task_smoke": validation["task_smoke_result"]["parity"],
        "remote_pt_download": remote.get("pt_hash_match"),
        "remote_from_pretrained": remote.get("from_pretrained"),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    source = args.source_checkpoint.expanduser().resolve()
    export_dir = args.export_dir.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Source checkpoint not found: {source}")
    if not (export_dir / "config.json").is_file():
        raise FileNotFoundError(f"Transformers export not found: {export_dir}")
    task = release_task_spec(args.task)
    local = validate_pair(
        source_checkpoint=source,
        export_loader=lambda: load_dibo_pretrained(
            str(export_dir), device=str(args.device), local_files_only=True
        ),
        task=args.task,
        device=str(args.device),
        seed=int(args.smoke_seed),
        run_task_smoke=bool(args.run_task_smoke),
        smoke_max_attempts=int(args.smoke_max_attempts),
    )
    report: Dict[str, Any] = {
        "source_checkpoint": source.name,
        "source_checkpoint_sha256": sha256_file(source),
        "source_checkpoint_size": int(source.stat().st_size),
        "exported_model_directory": export_dir.name,
        "task": args.task,
        "base_model": DEFAULT_MODEL_ID,
        "base_revision": LLADA_MODEL_REVISION,
        "git_commit": git_commit(),
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "huggingface_hub_version": __import__("huggingface_hub").__version__,
        "safetensors_version": __import__("safetensors").__version__,
        "delimiter_tokens": list(DIBO_DELIMITER_TOKENS),
        "local": local,
        "transformers_export_weights": standard_export_weight_metadata(export_dir),
    }
    if args.remote_repo_id:
        if not args.remote_revision or args.remote_cache_dir is None:
            raise ValueError(
                "--remote-repo-id requires --remote-revision and --remote-cache-dir."
            )
        remote_cache = args.remote_cache_dir.expanduser().resolve()
        remote_cache.mkdir(parents=True, exist_ok=True)
        remote_checkpoint = load_remote_checkpoint(
            args.remote_repo_id,
            str(task["original_checkpoint_filename"]),
            args.remote_revision,
            remote_cache / "original_checkpoint",
        )
        remote = validate_pair(
            source_checkpoint=remote_checkpoint,
            export_loader=lambda: load_dibo_pretrained(
                args.remote_repo_id,
                revision=args.remote_revision,
                device=str(args.device),
                cache_dir=str(remote_cache / "standard_export"),
            ),
            task=args.task,
            device=str(args.device),
            seed=int(args.smoke_seed),
            run_task_smoke=bool(args.run_task_smoke),
            smoke_max_attempts=int(args.smoke_max_attempts),
        )
        remote_files = [
            path.relative_to(remote_cache / "standard_export").as_posix()
            for path in (remote_cache / "standard_export").rglob("*")
            if path.is_file()
        ]
        report["remote"] = {
            "repository_id": args.remote_repo_id,
            "revision": args.remote_revision,
            "downloaded_checkpoint": remote_checkpoint.name,
            "pt_hash_match": sha256_file(remote_checkpoint) == sha256_file(source),
            "from_pretrained": all(
                (
                    remote["strict_source_load"],
                    remote["tokenizer_match"],
                    remote["config_match"],
                    remote["tensor_value_match"],
                    remote["forward_parity"],
                    remote["candidate_generation_parity"],
                    not args.run_task_smoke or remote["task_smoke_result"]["parity"],
                )
            ),
            "standard_cache_contains_original_checkpoint": any(
                path.endswith(str(task["original_checkpoint_filename"])) for path in remote_files
            ),
            "standard_cache_contains_base_weight_snapshot": any(
                "models--GSAI-ML--LLaDA-8B-Instruct" in path for path in remote_files
            ),
            "validation": remote,
        }
    checks = [
        local["strict_source_load"],
        local["tokenizer_match"],
        local["state_dict_key_match"],
        local["tensor_shape_match"],
        local["tensor_dtype_match"],
        local["tensor_value_match"],
        local["config_match"],
        local["forward_parity"],
        local["candidate_generation_parity"],
        not args.run_task_smoke or local["task_smoke_result"]["parity"],
    ]
    if args.remote_repo_id:
        remote = report["remote"]
        checks.extend(
            (
                remote["pt_hash_match"],
                remote["from_pretrained"],
                not remote["standard_cache_contains_original_checkpoint"],
                not remote["standard_cache_contains_base_weight_snapshot"],
            )
        )
    report["final_status"] = "passed" if all(checks) else "failed"
    output = args.output.expanduser().resolve() if args.output else export_dir / "export_validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    update_manifest(export_dir, output, report)
    write_sha256sums(export_dir, export_dir.iterdir())
    print(f"[Validation] {report['final_status']}: {output}")
    if report["final_status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
