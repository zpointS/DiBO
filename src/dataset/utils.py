from __future__ import annotations

from typing import Any, List, Tuple

from pathlib import Path
import importlib.util

from src.dataset.task_bundle import TaskBundle, load_task_bundle


def _import_module_from_path(py_path: Path):
    spec = importlib.util.spec_from_file_location(py_path.stem, str(py_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import module from: {py_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _pick_templates_from_module(mod: Any) -> Tuple[List[str], List[str]]:
    """
    Find TRAIN_/VALID_ prompt template lists in a prompts module.
    """
    train_key = None
    valid_key = None

    for name in dir(mod):
        if name.startswith("TRAIN_PROMPT_TEMPLATES"):
            train_key = name
        if name.startswith("VALID_PROMPT_TEMPLATES"):
            valid_key = name

    if train_key is None or valid_key is None:
        found = [n for n in dir(mod) if "PROMPT_TEMPLATES" in n]
        raise RuntimeError(
            f"Cannot find TRAIN_/VALID_ templates in module={mod.__file__}. "
            f"Found attrs: {found}"
        )

    train = getattr(mod, train_key)
    valid = getattr(mod, valid_key)

    if not isinstance(train, (list, tuple)) or not isinstance(valid, (list, tuple)):
        raise RuntimeError(
            f"Templates must be list/tuple. Got train={type(train)}, valid={type(valid)}"
        )

    return list(train), list(valid)


def load_templates_for_task(task_name: str, prompts_dir: str | Path) -> Tuple[List[str], List[str]]:
    pdir = Path(prompts_dir).expanduser().resolve()
    if not pdir.exists():
        raise FileNotFoundError(f"prompts_dir not found: {pdir}")

    py_path = pdir / f"{task_name}_prompts.py"
    if not py_path.exists():
        raise FileNotFoundError(f"prompts file not found: {py_path}")

    mod = _import_module_from_path(py_path)
    return _pick_templates_from_module(mod)


def load_bundle_for_task(task_name: str, bundle_dir: str | Path) -> TaskBundle:
    bdir = Path(bundle_dir).expanduser().resolve()
    if not bdir.exists():
        raise FileNotFoundError(f"bundle_dir not found: {bdir}")

    npz_path = bdir / f"{task_name}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"bundle npz not found: {npz_path}")

    return load_task_bundle(npz_path)
