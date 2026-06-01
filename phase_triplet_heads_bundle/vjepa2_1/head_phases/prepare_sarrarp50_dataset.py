#!/usr/bin/env python3
"""
Prepare the real SAR_RARP50 dataset for the local V-JEPA action probe.

This script handles the parts before model training:

1. Download the SAR_RARP50 Synapse dataset, or use an already-downloaded folder.
2. Extract zip/tar archives if the dataset is still packed.
3. Create the expected local layout:

       <data-root>/train1/video_*/action_discrete.txt
       <data-root>/train2/video_*/action_discrete.txt
       <data-root>/test/video_*

4. Optionally run `build_sarrarp50_action_dataset.py` to make clip MP4s and
   train/val probe CSVs.

The SAR_RARP50 evaluation GitHub repository is only the toolkit; it does not
contain the actual train/test video data. The public Synapse project for the
dataset is syn31997652.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_ROOT = Path("/path/to/surg_vid/sarrarp50/globus/SAR_RARP50_raw")
DEFAULT_DATA_ROOT = Path("/path/to/surg_vid/sarrarp50/globus/SAR_RARP50")
DEFAULT_SYNAPSE_ID = "syn31997652"
SPLIT_NAMES = ("train1", "train2", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synapse-id", default=DEFAULT_SYNAPSE_ID)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help="Use an already-downloaded SAR_RARP50 folder/archive tree instead of downloading from Synapse.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Do not contact Synapse; only normalize/extract existing --raw-root or --source-root.",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy data into --data-root instead of symlinking split/video directories.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove existing split links/directories inside --data-root before relinking/copying.",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip archive extraction and only look for existing train1/train2/test folders.",
    )
    parser.add_argument(
        "--skip-build-clips",
        action="store_true",
        help="Only prepare the SAR_RARP50 data layout; do not build V-JEPA clip CSVs.",
    )
    parser.add_argument(
        "--builder-extra-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra args passed to build_sarrarp50_action_dataset.py after --.",
    )
    return parser.parse_args()


def require_synapse_modules():
    try:
        import synapseclient  # type: ignore
        import synapseutils  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing Synapse Python client. Install it in the repo venv first:\n"
            "  python3 -m pip install synapseclient\n"
            "Then login/accept terms if needed:\n"
            "  python3 -m synapseclient login\n"
        ) from exc
    return synapseclient, synapseutils


def download_from_synapse(synapse_id: str, raw_root: Path) -> None:
    from synapseclient.core.exceptions import SynapseNoCredentialsError

    synapseclient, synapseutils = require_synapse_modules()
    raw_root.mkdir(parents=True, exist_ok=True)
    print(f"Logging in to Synapse and downloading {synapse_id} to {raw_root}")
    try:
        syn = synapseclient.login()
    except SynapseNoCredentialsError as exc:
        raise SystemExit(
            "Synapse credentials are not configured on this machine.\n"
            "Create a Synapse personal access token, then run one of:\n\n"
            "  export SYNAPSE_AUTH_TOKEN='<token>'\n\n"
            "or create ~/.synapseConfig with:\n\n"
            "  [authentication]\n"
            "  authtoken = <token>\n\n"
            "Then rerun this script. If Synapse asks for SAR_RARP50 data-access "
            "terms, accept them on the Synapse web page for syn31997652 first."
        ) from exc
    synapseutils.syncFromSynapse(
        syn,
        synapse_id,
        path=str(raw_root),
        ifcollision="keep.local",
        followLink=True,
    )


def safe_extract_zip(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = destination / member.filename
            if not target.resolve().is_relative_to(destination.resolve()):
                raise ValueError(f"Refusing unsafe zip member {member.filename} in {archive_path}")
        archive.extractall(destination)


def safe_extract_tar(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path) as archive:
        for member in archive.getmembers():
            target = destination / member.name
            if not target.resolve().is_relative_to(destination.resolve()):
                raise ValueError(f"Refusing unsafe tar member {member.name} in {archive_path}")
        archive.extractall(destination)


def extract_archives(search_root: Path, extract_root: Path) -> list[Path]:
    extracted_roots: list[Path] = []
    archives = sorted(
        path
        for path in search_root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".zip", ".tar", ".tgz", ".gz"}
        and not path.name.endswith(".nii.gz")
    )
    if not archives:
        return extracted_roots

    for archive_path in archives:
        relative = archive_path.relative_to(search_root)
        archive_extract_root = extract_root / relative.with_suffix("")
        marker = archive_extract_root / ".extract_complete"
        if marker.exists():
            extracted_roots.append(archive_extract_root)
            continue

        print(f"Extracting {archive_path} -> {archive_extract_root}")
        if archive_path.suffix.lower() == ".zip":
            safe_extract_zip(archive_path, archive_extract_root)
        elif archive_path.suffix.lower() in {".tar", ".tgz", ".gz"}:
            safe_extract_tar(archive_path, archive_extract_root)
        marker.touch()
        extracted_roots.append(archive_extract_root)
    return extracted_roots


def extract_archives_recursive(search_roots: list[Path], extract_root: Path) -> list[Path]:
    extracted_roots: list[Path] = []
    queue: list[tuple[Path, Path]] = [(root, extract_root) for root in search_roots if root.exists()]
    scanned: set[Path] = set()

    while queue:
        root, root_extract_root = queue.pop(0)
        resolved = root.resolve()
        if resolved in scanned:
            continue
        scanned.add(resolved)

        new_roots = extract_archives(root, root_extract_root)
        for new_root in new_roots:
            if new_root not in extracted_roots:
                extracted_roots.append(new_root)
            # Nested SAR_RARP50 video archives should unpack next to themselves:
            # _extracted/24932529/video_01.zip -> _extracted/24932529/video_01/
            queue.append((new_root, new_root))
    return extracted_roots


def has_downloaded_payload(search_roots: list[Path]) -> bool:
    payload_suffixes = {
        ".avi",
        ".mp4",
        ".mov",
        ".mkv",
        ".png",
        ".jpg",
        ".jpeg",
        ".zip",
        ".tar",
        ".tgz",
        ".gz",
        ".txt",
    }
    for root in search_roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.name == "SYNAPSE_METADATA_MANIFEST.tsv":
                continue
            if path.suffix.lower() in payload_suffixes:
                return True
    return False


def contains_action_labels(split_dir: Path) -> bool:
    return any(split_dir.glob("video_*/action_discrete.txt"))


def is_video_dir(path: Path) -> bool:
    return path.is_dir() and path.name.startswith("video_") and (
        (path / "action_discrete.txt").exists() or (path / "video_left.avi").exists()
    )


def infer_split_from_video_dir(video_dir: Path) -> str:
    parent_name = video_dir.parent.name.lower()
    if parent_name in SPLIT_NAMES:
        return parent_name

    match = re.search(r"video_(\d+)", video_dir.name)
    if match:
        video_number = int(match.group(1))
        # The downloaded Synapse archives observed locally are split as
        # videos 01-40 (+ *_1/*_2 fragments) for train and videos 41-50 for test.
        if video_number >= 41:
            return "test"
    return "train1"


def find_split_video_dirs(search_roots: list[Path]) -> dict[str, list[Path]]:
    found: dict[str, list[Path]] = {split: [] for split in SPLIT_NAMES}
    seen: set[Path] = set()

    def add(split_name: str, video_dir: Path) -> None:
        resolved = video_dir.resolve()
        if resolved in seen:
            return
        seen.add(resolved)
        found[split_name].append(video_dir)

    for root in search_roots:
        if not root.exists():
            continue

        for split_name in SPLIT_NAMES:
            for split_dir in [root / split_name, *root.rglob(split_name)]:
                if not split_dir.is_dir():
                    continue
                for video_dir in sorted(path for path in split_dir.glob("video_*") if is_video_dir(path)):
                    add(split_name, video_dir)

        for video_dir in sorted(path for path in [root, *root.rglob("video_*")] if is_video_dir(path)):
            add(infer_split_from_video_dir(video_dir), video_dir)

    for split_name in SPLIT_NAMES:
        found[split_name].sort(key=lambda path: path.name)
    return found


def find_split_dirs(search_roots: list[Path]) -> dict[str, list[Path]]:
    found: dict[str, list[Path]] = {split: [] for split in SPLIT_NAMES}
    for root in search_roots:
        if not root.exists():
            continue
        for split_name in SPLIT_NAMES:
            for candidate in [root / split_name, *root.rglob(split_name)]:
                if candidate.is_dir() and (
                    split_name == "test" or contains_action_labels(candidate)
                ):
                    if candidate not in found[split_name]:
                        found[split_name].append(candidate)
    return found


def replace_path(path: Path, force: bool) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if not force:
        raise FileExistsError(f"{path} already exists; pass --force to replace it")
    if path.is_symlink() or path.is_file():
        path.unlink()
    else:
        shutil.rmtree(path)


def link_or_copy(src: Path, dst: Path, copy: bool, force: bool) -> None:
    replace_path(dst, force=force)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copytree(src, dst)
    else:
        os.symlink(src.resolve(), dst, target_is_directory=True)


def merge_split_video_dirs(
    split_name: str,
    video_dirs: list[Path],
    data_root: Path,
    copy: bool,
    force: bool,
) -> None:
    if not video_dirs:
        return
    split_dst = data_root / split_name
    replace_path(split_dst, force=force)
    split_dst.mkdir(parents=True, exist_ok=True)
    for video_dir in video_dirs:
        dst = split_dst / video_dir.name
        link_or_copy(video_dir, dst, copy=copy, force=force)


def normalize_dataset(source_roots: list[Path], data_root: Path, copy: bool, force: bool) -> dict[str, list[Path]]:
    if not has_downloaded_payload(source_roots):
        raise SystemExit(
            "No SAR_RARP50 payload files were downloaded. Synapse may have allowed READ "
            "metadata access but denied DOWNLOAD access.\n"
            "Open https://www.synapse.org/Synapse:syn31997652 with the same Synapse account, "
            "accept/request the SAR_RARP50 data access/download terms, then rerun this script.\n"
            "If you download the archives manually through the browser, rerun with:\n"
            "  head_phases/prepare_sarrarp50_dataset.py "
            "--source-root /path/to/downloaded/SAR_RARP50_or_archives --skip-download --force"
        )

    split_video_dirs = find_split_video_dirs(source_roots)
    train_video_dirs = split_video_dirs["train1"] + split_video_dirs["train2"]
    if not train_video_dirs:
        searched = "\n  ".join(str(root) for root in source_roots)
        raise SystemExit(
            "Could not find SAR_RARP50 training action-label folders.\n"
            "Expected e.g. train1/video_*/action_discrete.txt, or loose "
            "video_01/video_02 archives extracted from the training zip.\n"
            f"Searched:\n  {searched}"
        )

    data_root.mkdir(parents=True, exist_ok=True)
    for split_name in SPLIT_NAMES:
        merge_split_video_dirs(
            split_name=split_name,
            video_dirs=split_video_dirs[split_name],
            data_root=data_root,
            copy=copy,
            force=force,
        )

    print(f"Prepared SAR_RARP50 layout under {data_root}")
    for split_name in SPLIT_NAMES:
        split_path = data_root / split_name
        if split_path.exists():
            num_videos = len([path for path in split_path.glob("video_*") if path.is_dir()])
            num_labels = len(list(split_path.glob("video_*/action_discrete.txt")))
            print(f"  {split_name}: {num_videos} video dirs, {num_labels} action files")
    return split_video_dirs


def build_probe_clips(data_root: Path, split_video_dirs: dict[str, list[Path]], extra_args: list[str]) -> None:
    builder = REPO_ROOT / "head_phases" / "build_sarrarp50_action_dataset.py"
    train_splits = [split_name for split_name in ("train1", "train2") if split_video_dirs[split_name]]
    test_splits = [split_name for split_name in ("test",) if split_video_dirs[split_name]]
    command = [
        sys.executable,
        str(builder),
        "--data-root",
        str(data_root),
        "--train-splits",
        *train_splits,
        "--sample-stride",
        "10",
        "--unpack-missing-rgb",
    ]
    if test_splits:
        command.extend(["--test-splits", *test_splits])
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]
    command.extend(extra_args)
    print("Running clip/CSV builder:")
    print("  " + " ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    source_roots: list[Path] = []

    if args.source_root is not None:
        source_roots.append(args.source_root)
    else:
        source_roots.append(args.raw_root)

    if not args.skip_download and args.source_root is None:
        download_from_synapse(args.synapse_id, args.raw_root)

    if not args.skip_extract:
        extract_root = args.raw_root / "_extracted"
        roots_to_extract = [root for root in source_roots if root.exists()]
        source_roots.extend(extract_archives_recursive(roots_to_extract, extract_root))
        source_roots.append(extract_root)

    split_video_dirs = normalize_dataset(
        source_roots=source_roots,
        data_root=args.data_root,
        copy=args.copy,
        force=args.force,
    )

    if not args.skip_build_clips:
        build_probe_clips(args.data_root, split_video_dirs, args.builder_extra_args)


if __name__ == "__main__":
    main()
