from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.model.dllm import (
    DEFAULT_MODEL_ID,
    LLADA_MODEL_REVISION,
    _local_snapshot_path,
    _materialize_remote_code_snapshot,
)


class ModelLoaderTest(unittest.TestCase):
    def test_default_model_uses_pinned_revision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(
                os.environ,
                {"HF_HUB_OFFLINE": "0", "TRANSFORMERS_OFFLINE": "0"},
            ):
                with patch(
                    "huggingface_hub.snapshot_download",
                    return_value=directory,
                ) as download:
                    resolved = _local_snapshot_path(DEFAULT_MODEL_ID)

        self.assertEqual(resolved, Path(directory).resolve())
        download.assert_called_once_with(
            repo_id=DEFAULT_MODEL_ID,
            revision=LLADA_MODEL_REVISION,
            local_files_only=False,
        )

    def test_custom_model_does_not_reuse_llada_revision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(
                os.environ,
                {"HF_HUB_OFFLINE": "0", "TRANSFORMERS_OFFLINE": "0"},
            ):
                with patch(
                    "huggingface_hub.snapshot_download",
                    return_value=directory,
                ) as download:
                    _local_snapshot_path("organization/custom-model")

        download.assert_called_once_with(
            repo_id="organization/custom-model",
            local_files_only=False,
        )

    def test_remote_python_files_are_copied_out_of_hf_blobs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "snapshot"
            snapshot.mkdir()
            blob = root / "blob-without-extension"
            blob.write_text("original = True\n", encoding="utf-8")
            (snapshot / "modeling_example.py").symlink_to(blob)

            with patch(
                "src.model.dllm._default_materialized_root",
                return_value=root / "overlays",
            ):
                materialized = _materialize_remote_code_snapshot(snapshot)

            copied = materialized / "modeling_example.py"
            self.assertEqual(copied.read_bytes(), blob.read_bytes())
            self.assertNotEqual(copied.stat().st_ino, blob.stat().st_ino)


if __name__ == "__main__":
    unittest.main()
