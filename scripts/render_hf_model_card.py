#!/usr/bin/env python3
"""Render a model card from the measured DiBO Hugging Face artifact manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.release.hf_artifacts import render_model_card  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--release-tag", type=str, required=True)
    args = parser.parse_args()

    artifact_dir = args.artifact_dir.expanduser().resolve()
    manifest_path = artifact_dir / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    (artifact_dir / "README.md").write_text(
        render_model_card(manifest, release_tag=args.release_tag), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
