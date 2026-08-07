import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Optional, Tuple

import torch
from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

DEFAULT_MODEL_ID = "GSAI-ML/LLaDA-8B-Instruct"
LLADA_MODEL_REVISION = "08b83a6feb34df1a6011b80c3c00c7563e963b07"
DIBO_DELIMITER_TOKENS = (
    "|design-start|",
    "|design-end|",
    "|label-start|",
    "|label-end|",
)
_OFFLINE_ENV_VARS = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
_TRUE_ENV_VALUES = {"1", "on", "true", "yes"}
_MATERIALIZATION_VERSION = 3


def _default_materialized_root() -> Path:
    cache_home = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return cache_home.expanduser() / "dibo" / "model_overlays"


def _offline_mode_enabled() -> bool:
    return any(
        os.environ.get(name, "").strip().lower() in _TRUE_ENV_VALUES for name in _OFFLINE_ENV_VARS
    )


def _local_snapshot_path(model_name_or_path: str, revision: Optional[str] = None) -> Path:
    candidate = Path(model_name_or_path).expanduser()
    if candidate.exists():
        return candidate.resolve()

    from huggingface_hub import snapshot_download

    download_args = {
        "repo_id": model_name_or_path,
        "local_files_only": _offline_mode_enabled(),
    }
    if revision is not None:
        download_args["revision"] = revision
    elif model_name_or_path == DEFAULT_MODEL_ID:
        download_args["revision"] = LLADA_MODEL_REVISION
    return Path(snapshot_download(**download_args)).resolve()


def _needs_materialized_remote_code(path: Path) -> bool:
    return any(p.is_symlink() for p in path.glob("*.py"))


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.suffix == ".py":
        shutil.copy2(src, dst)
        return
    try:
        os.link(src, dst)
    except OSError:
        if src.stat().st_size <= 16 * 1024 * 1024:
            shutil.copy2(src, dst)
            return
        try:
            dst.symlink_to(src)
        except OSError:
            shutil.copy2(src, dst)


def _remote_code_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    for source in sorted(path.glob("*.py")):
        digest.update(source.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(source.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _materialize_remote_code_snapshot(path: Path) -> Path:
    """
    Create a local overlay whose files keep their snapshot filenames.

    Resolving cached snapshot symlinks can move LLaDA's Python files into the
    cache `blobs/` directory, where relative remote-code imports are unavailable.
    """

    source = path.resolve()
    source_digest = hashlib.sha256(str(source).encode("utf-8")).hexdigest()[:12]
    code_digest = _remote_code_fingerprint(source)
    dest = _default_materialized_root() / (f"{source.name}-{source_digest}-{code_digest[:12]}")
    marker = dest / ".dibo_materialized_snapshot.json"
    if marker.exists():
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
        if (
            payload.get("version") == _MATERIALIZATION_VERSION
            and payload.get("source") == str(source)
            and payload.get("remote_code_sha256") == code_digest
        ):
            return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{dest.name}.tmp-", dir=str(dest.parent)))
    try:
        for entry in source.iterdir():
            target = tmp / entry.name
            if entry.is_dir():
                shutil.copytree(entry, target, symlinks=False)
            elif entry.is_file() or entry.is_symlink():
                _link_or_copy(entry.resolve(), target)
        (tmp / ".dibo_materialized_snapshot.json").write_text(
            json.dumps(
                {
                    "version": _MATERIALIZATION_VERSION,
                    "source": str(source),
                    "remote_code_sha256": code_digest,
                    "reason": "transformers_remote_code_symlink_resolution",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        if dest.exists():
            shutil.rmtree(dest)
        os.replace(tmp, dest)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)
    return dest


def resolve_model_name_or_path_for_remote_code(
    model_name_or_path: str, revision: Optional[str] = None
) -> str:
    path = _local_snapshot_path(str(model_name_or_path), revision=revision)
    if _needs_materialized_remote_code(path):
        materialized = _materialize_remote_code_snapshot(path)
        print(f"[Model] Materialized remote-code snapshot: {materialized}")
        return str(materialized)
    return str(path)


def _ensure_generation_config_defaults(model: PreTrainedModel) -> None:
    """Backfill config fields expected by older LLaDA remote-code forwards."""
    for module in (model, getattr(model, "model", None), getattr(model, "transformer", None)):
        cfg = getattr(module, "config", None)
        if cfg is not None and not hasattr(cfg, "use_cache"):
            setattr(cfg, "use_cache", False)


def install_dibo_delimiter_tokens(tokenizer: PreTrainedTokenizer) -> int:
    """Install the four delimiter tokens used by the DiBO training pipeline."""
    return int(
        tokenizer.add_special_tokens(
            {"additional_special_tokens": list(DIBO_DELIMITER_TOKENS)}
        )
    )


def dibo_delimiter_token_ids(tokenizer: PreTrainedTokenizer) -> Tuple[int, ...]:
    """Return and validate the one-token encodings used by DiBO prompts."""
    ids = []
    for token in DIBO_DELIMITER_TOKENS:
        token_id = int(tokenizer.convert_tokens_to_ids(token))
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if encoded != [token_id]:
            raise ValueError(
                f"DiBO delimiter {token!r} is not represented as one tokenizer token: {encoded}"
            )
        ids.append(token_id)
    return tuple(ids)


def validate_dibo_model_and_tokenizer(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizer
) -> Tuple[int, ...]:
    """Check the tokenizer and vocabulary dimensions required by a DiBO model."""
    ids = dibo_delimiter_token_ids(tokenizer)
    vocab_size = len(tokenizer)
    input_embeddings = model.get_input_embeddings()
    if input_embeddings is None or input_embeddings.weight.shape[0] != vocab_size:
        actual = None if input_embeddings is None else int(input_embeddings.weight.shape[0])
        raise ValueError(
            "DiBO tokenizer and input embedding vocabulary sizes differ: "
            f"tokenizer={vocab_size}, input_embeddings={actual}"
        )

    output_embeddings = model.get_output_embeddings()
    if output_embeddings is not None and output_embeddings.weight.shape[0] < vocab_size:
        raise ValueError(
            "DiBO output vocabulary is smaller than the tokenizer vocabulary: "
            f"tokenizer={vocab_size}, output_embeddings={int(output_embeddings.weight.shape[0])}"
        )

    configured_size = getattr(model.config, "vocab_size", None)
    if configured_size is not None and int(configured_size) != vocab_size:
        raise ValueError(
            "DiBO tokenizer and config vocabulary sizes differ: "
            f"tokenizer={vocab_size}, config={int(configured_size)}"
        )
    packaged_output_size = getattr(model.config, "dibo_output_embedding_size", None)
    if (
        packaged_output_size is not None
        and output_embeddings is not None
        and int(packaged_output_size) != int(output_embeddings.weight.shape[0])
    ):
        raise ValueError(
            "DiBO packaged output vocabulary does not match the output projection: "
            f"config={int(packaged_output_size)}, "
            f"output_embeddings={int(output_embeddings.weight.shape[0])}"
        )
    return ids


def load_model_and_tokenizer(
    model_name_or_path: str = DEFAULT_MODEL_ID,
    device: str = "cuda",
    add_special_tokens: bool = True,
    revision: Optional[str] = None,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load the diffusion language model and add DiBO delimiter tokens.
    """

    print(f"[Loading] {model_name_or_path}")
    model_name_or_path = resolve_model_name_or_path_for_remote_code(
        model_name_or_path, revision=revision
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        use_fast=False,
    )

    # Some tokenizers do not define a pad token; fall back to eos.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("[Tokenizer] pad_token set to eos_token")

    if add_special_tokens:
        num_added = install_dibo_delimiter_tokens(tokenizer)
        if num_added > 0:
            print(f"[Tokenizer] Added {num_added} special tokens")

    print("[Model] Loading LLaDA with AutoModel")
    model = AutoModel.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    _ensure_generation_config_defaults(model)

    # Resize embeddings after adding special tokens.
    if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:

        old = model.get_input_embeddings().weight.shape[0]
        new = len(tokenizer)

        model.resize_token_embeddings(new)

        print(f"[Model] Resized embeddings: {old} -> {new}")

    print("[Model] Moving model to device...")
    model.to(device)
    print("[Model] Model moved.")

    if add_special_tokens:
        validate_dibo_model_and_tokenizer(model, tokenizer)

    return model, tokenizer


def _strict_load_dibo_state_dict(
    model: PreTrainedModel, checkpoint_path: str | Path, device: str
) -> None:
    path = Path(checkpoint_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"DiBO checkpoint not found: {path}")
    checkpoint: Any = torch.load(str(path), map_location=device)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(
            "Expected a canonical DiBO checkpoint dictionary with a 'model' state-dict key."
        )
    model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint


def load_dibo_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str = "cuda",
    base_model_id: str = DEFAULT_MODEL_ID,
    base_model_revision: Optional[str] = None,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load the canonical paper-faithful DiBO ``.pt`` checkpoint path."""
    model, tokenizer = load_model_and_tokenizer(
        model_name_or_path=base_model_id,
        device=device,
        add_special_tokens=True,
        revision=base_model_revision,
    )
    _strict_load_dibo_state_dict(model, checkpoint_path, device)
    validate_dibo_model_and_tokenizer(model, tokenizer)
    return model, tokenizer


def load_dibo_checkpoint_from_hub(
    repo_id: str,
    filename: str,
    *,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    local_files_only: Optional[bool] = None,
    device: str = "cuda",
    base_model_id: str = DEFAULT_MODEL_ID,
    base_model_revision: Optional[str] = None,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Download an original checkpoint and load it through the canonical DiBO path."""
    from huggingface_hub import hf_hub_download

    kwargs: dict[str, Any] = {"repo_id": repo_id, "filename": filename}
    if revision is not None:
        kwargs["revision"] = revision
    if cache_dir is not None:
        kwargs["cache_dir"] = cache_dir
    if local_files_only is not None:
        kwargs["local_files_only"] = local_files_only
    checkpoint_path = hf_hub_download(**kwargs)
    return load_dibo_checkpoint(
        checkpoint_path,
        device=device,
        base_model_id=base_model_id,
        base_model_revision=base_model_revision,
    )


def load_dibo_pretrained(
    repo_id_or_path: str,
    *,
    revision: Optional[str] = None,
    device: str = "cuda",
    local_files_only: bool = False,
    cache_dir: Optional[str] = None,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load a packaged DiBO Transformers export without using the base checkpoint path."""
    tokenizer = AutoTokenizer.from_pretrained(
        repo_id_or_path,
        revision=revision,
        trust_remote_code=True,
        use_fast=False,
        local_files_only=local_files_only,
        cache_dir=cache_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(
        repo_id_or_path,
        revision=revision,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        local_files_only=local_files_only,
        cache_dir=cache_dir,
    )
    _ensure_generation_config_defaults(model)
    model.to(device)
    validate_dibo_model_and_tokenizer(model, tokenizer)
    return model, tokenizer
