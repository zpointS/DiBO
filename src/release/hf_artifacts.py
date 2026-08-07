from __future__ import annotations

import hashlib
import inspect
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.model.dllm import DIBO_DELIMITER_TOKENS, dibo_delimiter_token_ids


ROOT = Path(__file__).resolve().parents[2]
RELEASE_SPEC_PATH = ROOT / "configs" / "hf_release.json"
REMOTE_WRAPPER_PATH = ROOT / "src" / "model" / "hf_export_remote_code" / "modeling_dibo_llada.py"
PAPER_ID = "arXiv:2603.17919"
PAPER_URL = "https://huggingface.co/papers/2603.17919"
GITHUB_REPOSITORY = "https://github.com/zpointS/DiBO"


def load_release_spec(path: Path = RELEASE_SPEC_PATH) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def release_task_spec(task: str) -> Dict[str, Any]:
    spec = load_release_spec()
    try:
        task_spec = dict(spec["tasks"][task])
    except KeyError as error:
        choices = ", ".join(sorted(spec.get("tasks", {})))
        raise ValueError(f"Unknown release task {task!r}. Expected one of: {choices}") from error
    task_spec["task"] = task
    task_spec["base_model"] = spec["base_model"]
    task_spec["base_model_revision"] = spec["base_model_revision"]
    task_spec["delimiter_tokens"] = list(spec["delimiter_tokens"])
    task_spec["auto_class"] = spec["auto_class"]
    return task_spec


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def parameter_dtype_counts(model: PreTrainedModel) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    for parameter in model.parameters():
        counts[str(parameter.dtype).removeprefix("torch.")] += int(parameter.numel())
    return dict(sorted(counts.items()))


def model_parameter_count(model: PreTrainedModel) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def standard_export_weight_files(directory: str | Path) -> list[Path]:
    return sorted(Path(directory).glob("model*.safetensors"))


def standard_export_weight_metadata(directory: str | Path) -> list[Dict[str, Any]]:
    return [
        {
            "filename": path.name,
            "byte_size": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
        for path in standard_export_weight_files(directory)
    ]


def copy_export_remote_code(model: PreTrainedModel, output_dir: str | Path) -> list[str]:
    """Vendor the pinned LLaDA code and DiBO shape wrapper needed by AutoModel."""
    output = Path(output_dir)
    source_dir = Path(inspect.getfile(type(model))).resolve().parent
    copied = []
    for name in ("configuration_llada.py", "modeling_llada.py"):
        source = source_dir / name
        if not source.is_file():
            raise FileNotFoundError(f"Pinned LLaDA source file not found: {source}")
        shutil.copy2(source, output / name)
        copied.append(name)
    shutil.copy2(REMOTE_WRAPPER_PATH, output / REMOTE_WRAPPER_PATH.name)
    (output / "__init__.py").write_text(
        "# Required for relative imports in DiBO's remote model code.\n",
        encoding="utf-8",
    )
    (output / "UPSTREAM_NOTICE.md").write_text(
        "# Upstream LLaDA code notice\n\n"
        "`configuration_llada.py` and `modeling_llada.py` are copied from "
        "[`GSAI-ML/LLaDA-8B-Instruct`](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct) "
        "revision `08b83a6feb34df1a6011b80c3c00c7563e963b07`. "
        "The base model card declares the MIT license. The copied files remain "
        "subject to the upstream terms. `modeling_dibo_llada.py` is DiBO release code.\n",
        encoding="utf-8",
    )
    copied.extend(["modeling_dibo_llada.py", "__init__.py", "UPSTREAM_NOTICE.md"])
    return copied


def configure_dibo_export(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizer
) -> Dict[str, Any]:
    """Record the asymmetric vocabulary dimensions used by released checkpoints."""
    input_size = int(model.get_input_embeddings().weight.shape[0])
    output_embeddings = model.get_output_embeddings()
    if output_embeddings is None:
        raise ValueError("The DiBO LLaDA export requires an output projection.")
    output_size = int(output_embeddings.weight.shape[0])
    if input_size != len(tokenizer):
        raise ValueError(
            f"Tokenizer length {len(tokenizer)} does not match input embeddings {input_size}."
        )
    if output_size < input_size:
        raise ValueError(
            f"Output projection {output_size} is smaller than input vocabulary {input_size}."
        )

    model.config.vocab_size = input_size
    model.config.embedding_size = input_size
    model.config.dibo_input_embedding_size = input_size
    model.config.dibo_output_embedding_size = output_size
    model.config.architectures = ["DiBOLLaDAModelLM"]
    model.config.auto_map = {
        "AutoConfig": "configuration_llada.LLaDAConfig",
        "AutoModel": "modeling_dibo_llada.DiBOLLaDAModelLM",
        "AutoModelForCausalLM": "modeling_dibo_llada.DiBOLLaDAModelLM",
    }
    return {
        "input_embedding_size": input_size,
        "output_embedding_size": output_size,
        "delimiter_token_ids": list(dibo_delimiter_token_ids(tokenizer)),
    }


def finalize_dibo_export_config(directory: str | Path) -> None:
    """Restore DiBO's wrapper mapping after Transformers saves the base class config."""
    path = Path(directory) / "config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    input_size = int(config["dibo_input_embedding_size"])
    output_size = int(config["dibo_output_embedding_size"])
    config.pop("_name_or_path", None)
    config["vocab_size"] = input_size
    config["embedding_size"] = input_size
    config["architectures"] = ["DiBOLLaDAModelLM"]
    config["auto_map"] = {
        "AutoConfig": "configuration_llada.LLaDAConfig",
        "AutoModel": "modeling_dibo_llada.DiBOLLaDAModelLM",
        "AutoModelForCausalLM": "modeling_dibo_llada.DiBOLLaDAModelLM",
    }
    config["dibo_input_embedding_size"] = input_size
    config["dibo_output_embedding_size"] = output_size
    path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_sha256sums(directory: str | Path, paths: Iterable[Path]) -> None:
    output = Path(directory)
    rows = []
    for path in sorted({Path(path) for path in paths}, key=lambda item: item.name):
        if path.is_file() and path.name != "SHA256SUMS":
            rows.append(f"{sha256_file(path)}  {path.name}")
    (output / "SHA256SUMS").write_text("\n".join(rows) + "\n", encoding="utf-8")


def render_model_card(manifest: Dict[str, Any], *, release_tag: str) -> str:
    """Render the public dual-format model card from measured artifact metadata."""
    task = str(manifest["task"])
    artifact_name = str(manifest["artifact_name"])
    repo_id = str(release_task_spec(task)["repository_id"])
    original = manifest["original_checkpoint"]
    related = [
        item["repository_id"]
        for item_task, item in load_release_spec()["tasks"].items()
        if item_task != task
    ]
    related_markdown = "\n".join(
        f"- [{repo.split('/', 1)[1]}](https://huggingface.co/{repo})" for repo in related
    )
    return f'''---
library_name: transformers
pipeline_tag: reinforcement-learning
base_model: {manifest["base_model"]}
base_model_relation: finetune
license: mit
tags:
  - llada
  - llm
  - diffusion-language-model
  - reinforcement-learning
  - black-box-optimization
  - offline-black-box-optimization
  - design-bench
  - dibo
---

# {artifact_name}

Final task-specific DiBO model for `{task}`, released with
[Training Diffusion Language Models for Black-Box Optimization](https://arxiv.org/abs/2603.17919)
(ICML 2026 Spotlight). See also the
[Hugging Face paper page]({manifest["paper"]["url"]}) and the
[DiBO code repository]({manifest["github_repository"]}).

This model completed domain adaptation (DA), supervised fine-tuning (SFT),
and reinforcement learning (RL).

## Available model formats

This repository provides the same final task-specific DiBO model in two formats.

1. **Original PyTorch checkpoint.** `{original["path"]}` is the canonical
   paper-faithful checkpoint produced by the DiBO training pipeline. It stores
   the state dictionary under the `{original["state_dict_key"]}` key and is
   loaded through the DiBO codebase on top of the pinned LLaDA base revision.
2. **Transformers/safetensors export.** The root-level config, tokenizer,
   custom modeling code, and sharded safetensors files are a validated
   convenience export derived deterministically from the original checkpoint.
   They load directly with `AutoModel.from_pretrained(...)`.

The safetensors model was not trained separately. The LLaDA base weights are
not duplicated in this repository.

## A. Load the standard Transformers export

```python
from transformers import AutoModel, AutoTokenizer

repo_id = "{repo_id}"
tokenizer = AutoTokenizer.from_pretrained(
    repo_id,
    revision="{release_tag}",
    trust_remote_code=True,
)
model = AutoModel.from_pretrained(
    repo_id,
    revision="{release_tag}",
    trust_remote_code=True,
    use_safetensors=True,
    torch_dtype="auto",
)
model.eval()
```

The packaged tokenizer already includes the four DiBO delimiter tokens. Do not
add them or resize embeddings again after loading this export.

The tokenizer configuration retains LLaDA's `chat_template` metadata, but DiBO
does not call `apply_chat_template` during training or evaluation. DiBO directly
tokenizes its rendered unified prompt-response corpus with the delimiter tokens
above; do not insert chat headers when reproducing the released evaluation path.

## B. Download and load the original checkpoint

The original artifact uses the released DiBO loader, which initializes the
pinned LLaDA base, adds the four delimiter tokens, resizes the input embedding,
and strictly loads `checkpoint["{original["state_dict_key"]}"]`.

```bash
hf download {repo_id} {original["path"]} \\
  --revision {release_tag} --local-dir checkpoints/{artifact_name.lower()}
```

```python
import torch
from huggingface_hub import hf_hub_download
from src.model.dllm import DEFAULT_MODEL_ID, LLADA_MODEL_REVISION, load_model_and_tokenizer

assert DEFAULT_MODEL_ID == "{manifest["base_model"]}"
assert LLADA_MODEL_REVISION == "{manifest["base_model_revision"]}"
checkpoint_path = hf_hub_download(
    "{repo_id}",
    filename="{original["path"]}",
    revision="{release_tag}",
)
model, tokenizer = load_model_and_tokenizer(DEFAULT_MODEL_ID, device="cuda")
checkpoint = torch.load(checkpoint_path, map_location="cuda")
model.load_state_dict(checkpoint["{original["state_dict_key"]}"], strict=True)
model.eval()
```

## C. Evaluate either format

From a checkout of the released DiBO code and its oracle environment:

```bash
# Standard Transformers export
python eval.py --tasks {task} \\
  --model_name_or_path {repo_id} --model_revision {release_tag} \\
  --seeds <SEEDS> --max_attempts 1000

# Canonical local .pt checkpoint
python eval.py --tasks {task} \\
  --checkpoint_path checkpoints/{artifact_name.lower()}/{original["path"]} \\
  --seeds <SEEDS> --max_attempts 1000
```

Both choices share the same downstream DiBO evaluation path. Direct oracle
evaluation requires the Design-Bench data cache and task dependencies described
in the [DiBO repository]({manifest["github_repository"]}).
For the exact Design-Bench snapshot used in the DiBO experiments, see
[DiBO-DesignBench-Snapshot](https://huggingface.co/datasets/zpointsun/DiBO-DesignBench-Snapshot).

## Limitations

Practical inference requires a CUDA-capable PyTorch environment. These
task-specific models are designed for DiBO's masked-response generation and
evaluation workflow; this release does not claim generic text-generation
pipeline support. Loading a released final model is for evaluation or use and
does not reproduce the DA/SFT/RL training process.

## Other DiBO task models

{related_markdown}

## Citation

If you find DiBO helpful, please cite:

```bibtex
@article{{sun2026training,
  title={{Training diffusion language models for black-box optimization}},
  author={{Sun, Zipeng and Chen, Can and Yuan, Ye and Wu, Haolun and Gu, Jiayao and Pal, Christopher and Liu, Xue}},
  journal={{arXiv preprint arXiv:2603.17919}},
  year={{2026}}
}}
```
'''
