from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import stat
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile, ZipInfo

from scripts.prepare_design_bench_data import (
    ARCHIVE_PREFIX,
    REQUIRED_ARCHIVE_MEMBERS,
    build_parser,
    extract_archive,
)


class PrepareDesignBenchDataTest(unittest.TestCase):
    def write_archive(
        self,
        path: Path,
        *,
        missing: frozenset[str] = frozenset(),
        extras: tuple[str, ...] = (),
    ) -> None:
        with ZipFile(path, "w") as archive:
            archive.writestr(f"{ARCHIVE_PREFIX}/", b"")
            for name in sorted(REQUIRED_ARCHIVE_MEMBERS - missing):
                archive.writestr(name, name.encode("utf-8"))
            for name in extras:
                archive.writestr(name, b"extra")

    def test_archive_argument(self) -> None:
        args = build_parser().parse_args(
            ["--archive", "cache.zip", "--target", "cache"]
        )
        self.assertEqual(args.archive, Path("cache.zip"))
        self.assertEqual(args.target, Path("cache"))

    def test_archive_argument_is_required(self) -> None:
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            build_parser().parse_args([])

    def test_extracts_required_and_additional_cache_files(self) -> None:
        extra = f"{ARCHIVE_PREFIX}/metadata/cache.json"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / "cache.zip"
            target = root / "target"
            self.write_archive(archive_path, extras=(extra,))

            with redirect_stdout(StringIO()):
                count = extract_archive(archive_path, target, force=False)
                repeated_count = extract_archive(
                    archive_path, target, force=False
                )

            self.assertEqual(count, len(REQUIRED_ARCHIVE_MEMBERS) + 1)
            self.assertEqual(repeated_count, count)
            self.assertEqual(
                (target / "metadata/cache.json").read_bytes(), b"extra"
            )

    def test_rejects_archive_missing_required_files(self) -> None:
        missing = frozenset({next(iter(REQUIRED_ARCHIVE_MEMBERS))})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / "cache.zip"
            self.write_archive(archive_path, missing=missing)

            with self.assertRaisesRegex(ValueError, "missing required members"):
                extract_archive(archive_path, root / "target", force=False)

    def test_rejects_members_outside_cache_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / "cache.zip"
            self.write_archive(archive_path, extras=("outside/data.npy",))

            with self.assertRaisesRegex(ValueError, "must be rooted"):
                extract_archive(archive_path, root / "target", force=False)

    def test_rejects_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / "cache.zip"
            self.write_archive(
                archive_path,
                extras=(f"{ARCHIVE_PREFIX}/../escape.npy",),
            )

            with self.assertRaisesRegex(ValueError, "Unsafe archive member"):
                extract_archive(archive_path, root / "target", force=False)

    def test_rejects_archive_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / "cache.zip"
            self.write_archive(archive_path)
            info = ZipInfo(f"{ARCHIVE_PREFIX}/metadata/link")
            info.create_system = 3
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            with ZipFile(archive_path, "a") as archive:
                archive.writestr(info, b"target")

            with self.assertRaisesRegex(ValueError, "symlinks"):
                extract_archive(archive_path, root / "target", force=False)

    def test_rejects_archive_file_prefix_collision(self) -> None:
        extras = (
            f"{ARCHIVE_PREFIX}/metadata",
            f"{ARCHIVE_PREFIX}/metadata/cache.json",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / "cache.zip"
            self.write_archive(archive_path, extras=extras)

            with self.assertRaisesRegex(ValueError, "parent of another member"):
                extract_archive(archive_path, root / "target", force=False)

    def test_rejects_normalized_member_collision(self) -> None:
        extras = (
            f"{ARCHIVE_PREFIX}/metadata/cache.json",
            f"{ARCHIVE_PREFIX}/metadata/./cache.json",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / "cache.zip"
            self.write_archive(archive_path, extras=extras)

            with self.assertRaisesRegex(ValueError, "colliding member paths"):
                extract_archive(archive_path, root / "target", force=False)

    def test_rejects_existing_file_in_required_parent_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / "cache.zip"
            target = root / "target"
            target.mkdir()
            (target / "tf_bind_10-pho4").write_bytes(b"conflict")
            self.write_archive(archive_path)

            with self.assertRaisesRegex(
                FileExistsError, "Required parent is not a directory"
            ):
                extract_archive(archive_path, target, force=False)
            self.assertFalse((target / "ant_morphology").exists())

    def test_force_controls_existing_file_replacement(self) -> None:
        member = "ant_morphology/ant_morphology-x-0.npy"
        expected = f"{ARCHIVE_PREFIX}/{member}".encode("utf-8")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / "cache.zip"
            target = root / "target"
            self.write_archive(archive_path)

            with redirect_stdout(StringIO()):
                extract_archive(archive_path, target, force=False)
            destination = target / member
            destination.write_bytes(b"different")

            with self.assertRaisesRegex(
                FileExistsError, "Refusing to replace"
            ):
                extract_archive(archive_path, target, force=False)
            self.assertEqual(destination.read_bytes(), b"different")

            with redirect_stdout(StringIO()):
                extract_archive(archive_path, target, force=True)
            self.assertEqual(destination.read_bytes(), expected)


if __name__ == "__main__":
    unittest.main()
