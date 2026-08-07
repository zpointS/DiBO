from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import stat
import tempfile
from pathlib import Path, PurePosixPath
from zipfile import ZipFile, ZipInfo


ARCHIVE_PREFIX = "design_bench_data"


def required_archive_members() -> frozenset[str]:
    prefix = ARCHIVE_PREFIX
    members = {
        f"{prefix}/ant_morphology/ant_morphology-x-0.npy",
        f"{prefix}/ant_morphology/ant_morphology-y-0.npy",
        f"{prefix}/ant_morphology/ant_oracle.pkl",
        f"{prefix}/dkitty_morphology/dkitty_morphology-x-0.npy",
        f"{prefix}/dkitty_morphology/dkitty_morphology-y-0.npy",
        f"{prefix}/dkitty_morphology/dkitty_oracle.pkl",
        f"{prefix}/smiles_vocab.txt",
        f"{prefix}/tf_bind_8-SIX6_REF_R1/tf_bind_8-x-0.npy",
        f"{prefix}/tf_bind_8-SIX6_REF_R1/tf_bind_8-y-0.npy",
    }
    for index in range(84):
        members.add(
            f"{prefix}/tf_bind_10-pho4/tf_bind_10-x-{index}.npy"
        )
        members.add(
            f"{prefix}/tf_bind_10-pho4/tf_bind_10-y-{index}.npy"
        )
    return frozenset(members)


REQUIRED_ARCHIVE_MEMBERS = required_archive_members()


def default_target() -> Path:
    spec = importlib.util.find_spec("design_bench")
    if spec is None or spec.origin is None:
        raise ModuleNotFoundError(
            "design-bench is not installed; install requirements-oracle.txt first."
        )
    return Path(spec.origin).resolve().parent.parent / ARCHIVE_PREFIX


def member_target(target: Path, info: ZipInfo) -> Path:
    relative = PurePosixPath(info.filename)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or "\\" in info.filename
    ):
        raise ValueError(f"Unsafe archive member: {info.filename}")
    if not relative.parts or relative.parts[0] != ARCHIVE_PREFIX:
        raise ValueError(
            f"Archive members must be rooted at {ARCHIVE_PREFIX}/: "
            f"{info.filename}"
        )
    if stat.S_ISLNK(info.external_attr >> 16):
        raise ValueError(f"Archive symlinks are not supported: {info.filename}")

    parts = relative.parts[1:]
    if not parts:
        if info.is_dir():
            return target
        raise ValueError(f"Archive root must be a directory: {info.filename}")
    destination = target.joinpath(*parts)
    if not destination.resolve().is_relative_to(target.resolve()):
        raise ValueError(f"Archive member escapes target: {info.filename}")
    return destination


def files_match(path: Path, archive: ZipFile, info: ZipInfo) -> bool:
    if not path.is_file() or path.stat().st_size != info.file_size:
        return False
    with path.open("rb") as existing, archive.open(info) as incoming:
        while True:
            left = existing.read(1024 * 1024)
            right = incoming.read(1024 * 1024)
            if left != right:
                return False
            if not left:
                return True


def validated_members(
    archive: ZipFile, target: Path
) -> list[tuple[ZipInfo, Path]]:
    infos = archive.infolist()
    names = [info.filename for info in infos]
    if len(names) != len(set(names)):
        raise ValueError("The Design-Bench archive contains duplicate members.")
    entries = [(info, member_target(target, info)) for info in infos]
    destinations = [destination for _, destination in entries]
    if len(destinations) != len(set(destinations)):
        raise ValueError("The Design-Bench archive has colliding member paths.")

    file_entries = [
        (info, destination)
        for info, destination in entries
        if not info.is_dir()
    ]
    file_infos = [info for info, _ in file_entries]
    file_names = {info.filename for info in file_infos}
    missing = sorted(REQUIRED_ARCHIVE_MEMBERS - file_names)
    if missing:
        raise ValueError(
            f"Design-Bench archive is missing required members: {missing}"
        )

    file_destinations = {destination for _, destination in file_entries}
    for _, destination in entries:
        if destination == target:
            continue
        for parent in destination.parents:
            if parent == target:
                break
            if parent in file_destinations:
                raise ValueError(
                    "Archive file member is the parent of another member: "
                    f"{parent}"
                )

    bad_member = archive.testzip()
    if bad_member is not None:
        raise ValueError(f"Corrupt Design-Bench archive member: {bad_member}")
    return file_entries


def validate_target_paths(
    target: Path,
    members: list[tuple[ZipInfo, Path]],
    archive: ZipFile,
    force: bool,
) -> None:
    if target.exists() and not target.is_dir():
        raise FileExistsError(f"Data target is not a directory: {target}")

    for info, destination in members:
        for parent in destination.parents:
            if parent == target:
                break
            if parent.is_symlink() or (
                parent.exists() and not parent.is_dir()
            ):
                raise FileExistsError(
                    f"Required parent is not a directory: {parent}"
                )

        if destination.is_symlink():
            raise FileExistsError(
                f"Refusing to replace a symlink: {destination}"
            )
        if destination.exists() and not destination.is_file():
            raise FileExistsError(
                f"Data destination is not a file: {destination}"
            )
        if (
            destination.exists()
            and not force
            and not files_match(destination, archive, info)
        ):
            raise FileExistsError(
                f"Refusing to replace different data: {destination}. "
                "Pass --force to overwrite it."
            )


def write_member(
    destination: Path, archive: ZipFile, info: ZipInfo
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with archive.open(info) as source, tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            delete=False,
        ) as output:
            temporary = Path(output.name)
            shutil.copyfileobj(source, output)
        assert temporary is not None
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def extract_archive(archive_path: Path, target: Path, force: bool) -> int:
    installed = 0
    target = target.expanduser().resolve()
    with ZipFile(archive_path) as archive:
        members = validated_members(archive, target)
        validate_target_paths(target, members, archive, force)

        for info, destination in members:
            if destination.exists() and not force:
                print(f"ok     {destination}")
            else:
                write_member(destination, archive, info)
                print(f"wrote  {destination}")
            installed += 1
    return installed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Install a local Design-Bench data-cache ZIP archive."
    )
    parser.add_argument(
        "--archive",
        type=Path,
        required=True,
        help="path to a local Design-Bench data-cache ZIP",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=None,
        help="staging directory; defaults to the location used by Design-Bench",
    )
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    archive_path = args.archive.expanduser().resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(f"Design-Bench archive not found: {archive_path}")
    target = args.target.expanduser().resolve() if args.target else default_target()
    count = extract_archive(archive_path, target, args.force)
    print(f"Design-Bench data directory ready: {target} ({count} files)")


if __name__ == "__main__":
    main()
