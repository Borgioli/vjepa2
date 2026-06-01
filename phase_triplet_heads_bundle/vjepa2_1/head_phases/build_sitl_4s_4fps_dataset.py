#!/usr/bin/env python3
"""
Build a 4-second, 4fps SITL phase dataset from the existing frame-chunk clips.

The current `globus/sitl_phases` dataset stores 4-frame clips at 30fps with a
label per tiny clip. This script groups those clips into non-overlapping
4-second windows, downsamples to 4fps, and assigns each output clip the
dominant label inside its source window.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


DEFAULT_INPUT_ROOT = Path("/path/to/phase_triplet_heads_bundle/globus/sitl_phases")
DEFAULT_OUTPUT_ROOT = Path("/path/to/phase_triplet_heads_bundle/globus/sitl_phases_4s_4fps")
DEFAULT_OUTPUT_FPS = 4.0
DEFAULT_OUTPUT_DURATION_SECONDS = 4.0


@dataclass(frozen=True)
class SourceClipSpec:
    fps: float
    frames_per_clip: int
    duration_seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-fps", type=float, default=DEFAULT_OUTPUT_FPS)
    parser.add_argument(
        "--output-duration-seconds",
        type=float,
        default=DEFAULT_OUTPUT_DURATION_SECONDS,
    )
    parser.add_argument(
        "--annotations-csv",
        type=Path,
        default=None,
        help="Defaults to <input-root>/annotations.csv",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=max(1, min(4, os.cpu_count() or 1)),
        help="Parallel ffmpeg jobs to run across videos.",
    )
    parser.add_argument(
        "--ffmpeg-bin",
        default="ffmpeg",
        help="Path to the ffmpeg executable.",
    )
    parser.add_argument(
        "--ffprobe-bin",
        default="ffprobe",
        help="Path to the ffprobe executable.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove the output root before rebuilding it.",
    )
    parser.add_argument(
        "--limit-videos",
        type=int,
        default=None,
        help="Optional debug knob to only process the first N videos.",
    )
    parser.add_argument(
        "--video-names",
        nargs="+",
        default=None,
        help="Optional explicit list of video directory names to process, e.g. video10 video35.",
    )
    return parser.parse_args()


def parse_fraction(text: str) -> float:
    numerator, denominator = text.split("/", 1)
    return float(numerator) / float(denominator)


def probe_source_clip(sample_clip: Path, ffprobe_bin: str) -> SourceClipSpec:
    command = [
        ffprobe_bin,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate,nb_frames",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1",
        str(sample_clip),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    parsed: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        parsed[key.strip()] = value.strip()

    fps = parse_fraction(parsed["avg_frame_rate"])
    frames_per_clip = int(parsed["nb_frames"])
    container_duration_seconds = float(parsed["duration"])
    duration_seconds = frames_per_clip / fps
    if not math.isclose(
        duration_seconds,
        container_duration_seconds,
        rel_tol=1e-3,
        abs_tol=1e-3,
    ):
        raise ValueError(
            "Source clip stream duration and container duration disagree too much: "
            f"{duration_seconds} vs {container_duration_seconds} for {sample_clip}"
        )
    return SourceClipSpec(
        fps=fps,
        frames_per_clip=frames_per_clip,
        duration_seconds=duration_seconds,
    )


def parse_clip_index(path: Path) -> int:
    stem = path.stem
    prefix = "clip_"
    if not stem.startswith(prefix):
        raise ValueError(f"Unexpected clip filename: {path.name}")
    return int(stem[len(prefix) :])


def load_annotations(input_root: Path, annotations_csv: Path) -> dict[str, list[tuple[Path, str]]]:
    samples_by_video: dict[str, list[tuple[Path, str]]] = defaultdict(list)
    with annotations_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        expected_columns = {"path_to_the_clip", "label"}
        if set(reader.fieldnames or ()) != expected_columns:
            raise ValueError(
                f"Unexpected annotation columns: {reader.fieldnames}; expected {sorted(expected_columns)}"
            )

        for row in reader:
            relative_path = Path(row["path_to_the_clip"])
            label = row["label"]
            if relative_path.parts[:1] != ("clips",):
                raise ValueError(f"Unexpected clip path inside annotations.csv: {relative_path}")
            absolute_path = input_root / relative_path
            video_name = relative_path.parts[1]
            samples_by_video[video_name].append((absolute_path, label))

    for video_name, samples in samples_by_video.items():
        samples.sort(key=lambda item: parse_clip_index(item[0]))
        expected_indices = list(range(len(samples)))
        actual_indices = [parse_clip_index(path) for path, _ in samples]
        if actual_indices != expected_indices:
            raise ValueError(
                f"Expected sequential clip indices for {video_name}, got a mismatch around "
                f"{actual_indices[:10]}..."
            )
    return dict(sorted(samples_by_video.items()))


def choose_window_label(labels: list[str]) -> str:
    counts = Counter(labels)
    top_count = max(counts.values())
    candidates = {label for label, count in counts.items() if count == top_count}
    if len(candidates) == 1:
        return next(iter(candidates))

    center_label = labels[len(labels) // 2]
    if center_label in candidates:
        return center_label

    for label in labels:
        if label in candidates:
            return label
    raise AssertionError("Unreachable tie-break path for labels")


def build_concat_manifest(paths: list[Path], manifest_path: Path) -> None:
    with manifest_path.open("w", encoding="utf-8") as handle:
        for path in paths:
            handle.write(f"file '{path.as_posix()}'\n")


def generate_video(
    video_name: str,
    samples: list[tuple[Path, str]],
    output_root: Path,
    ffmpeg_bin: str,
    source_spec: SourceClipSpec,
    output_fps: float,
    output_duration_seconds: float,
) -> tuple[list[tuple[str, str]], int, int]:
    clips_per_output = round(output_duration_seconds / source_spec.duration_seconds)
    if not math.isclose(
        clips_per_output * source_spec.duration_seconds,
        output_duration_seconds,
        rel_tol=1e-6,
        abs_tol=1e-6,
    ):
        raise ValueError(
            "Output duration is not an integer multiple of the source clip duration: "
            f"{output_duration_seconds} vs {source_spec.duration_seconds}"
        )

    output_frames_per_clip = round(output_duration_seconds * output_fps)
    if output_frames_per_clip <= 0:
        raise ValueError("Output clip must contain at least one frame")

    full_groups = len(samples) // clips_per_output
    discarded_source_clips = len(samples) - (full_groups * clips_per_output)
    if full_groups == 0:
        return [], len(samples), discarded_source_clips

    video_output_dir = output_root / "clips" / video_name
    video_output_dir.mkdir(parents=True, exist_ok=True)
    labels = [label for _, label in samples[: full_groups * clips_per_output]]

    rows: list[tuple[str, str]] = []
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
        manifest_path = Path(handle.name)
    try:
        for group_idx in range(full_groups):
            start = group_idx * clips_per_output
            end = start + clips_per_output
            output_relative = f"clips/{video_name}/clip_{group_idx:06d}.mp4"
            output_path = output_root / output_relative
            source_paths = [path for path, _ in samples[start:end]]

            build_concat_manifest(source_paths, manifest_path)
            command = [
                ffmpeg_bin,
                "-y",
                "-loglevel",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(manifest_path),
                "-an",
                "-vf",
                f"fps={output_fps}",
                "-frames:v",
                str(output_frames_per_clip),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(output_path),
            ]
            subprocess.run(command, check=True)
            if not output_path.exists():
                raise FileNotFoundError(f"ffmpeg did not create expected output clip: {output_path}")
            rows.append((output_relative, choose_window_label(labels[start:end])))
    finally:
        manifest_path.unlink(missing_ok=True)

    return rows, full_groups, discarded_source_clips


def write_annotations(rows: list[tuple[str, str]], output_csv: Path) -> None:
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path_to_the_clip", "label"])
        writer.writerows(rows)


def write_summary(
    summary_path: Path,
    *,
    input_root: Path,
    annotations_csv: Path,
    output_root: Path,
    source_spec: SourceClipSpec,
    output_fps: float,
    output_duration_seconds: float,
    clips_per_output: int,
    total_videos: int,
    total_clips: int,
    discarded_source_clips: int,
) -> None:
    lines = [
        f"input_root={input_root}",
        f"annotations_csv={annotations_csv}",
        f"output_root={output_root}",
        f"source_clip_fps={source_spec.fps}",
        f"source_clip_frames={source_spec.frames_per_clip}",
        f"source_clip_duration_seconds={source_spec.duration_seconds}",
        f"output_fps={output_fps}",
        f"output_duration_seconds={output_duration_seconds}",
        f"source_clips_per_output={clips_per_output}",
        f"discarded_source_clips={discarded_source_clips}",
        f"videos_generated={total_videos}",
        f"clips={total_clips}",
    ]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    annotations_csv = args.annotations_csv or (args.input_root / "annotations.csv")

    if not args.input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {args.input_root}")
    if not annotations_csv.exists():
        raise FileNotFoundError(f"Annotations CSV does not exist: {annotations_csv}")
    if shutil.which(args.ffmpeg_bin) is None:
        raise FileNotFoundError(f"Could not find ffmpeg executable: {args.ffmpeg_bin}")
    if shutil.which(args.ffprobe_bin) is None:
        raise FileNotFoundError(f"Could not find ffprobe executable: {args.ffprobe_bin}")

    if args.output_root.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output root already exists: {args.output_root}. Pass --overwrite to rebuild it."
            )
        shutil.rmtree(args.output_root)

    samples_by_video = load_annotations(args.input_root, annotations_csv)
    if args.video_names is not None:
        requested = set(args.video_names)
        missing = sorted(requested.difference(samples_by_video))
        if missing:
            raise ValueError(f"Requested video names not found in annotations: {missing}")
        samples_by_video = {
            video_name: samples
            for video_name, samples in samples_by_video.items()
            if video_name in requested
        }
    if args.limit_videos is not None:
        limited_items = list(samples_by_video.items())[: args.limit_videos]
        samples_by_video = dict(limited_items)

    first_video = next(iter(samples_by_video.values()))
    first_clip = first_video[0][0]
    source_spec = probe_source_clip(first_clip, ffprobe_bin=args.ffprobe_bin)
    clips_per_output = round(args.output_duration_seconds / source_spec.duration_seconds)

    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "clips").mkdir(exist_ok=True)

    rows_by_video: dict[str, list[tuple[str, str]]] = {}
    total_output_clips = 0
    total_discarded_source_clips = 0

    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as executor:
        future_map = {
            executor.submit(
                generate_video,
                video_name,
                samples,
                args.output_root,
                args.ffmpeg_bin,
                source_spec,
                args.output_fps,
                args.output_duration_seconds,
            ): video_name
            for video_name, samples in samples_by_video.items()
        }
        for future in as_completed(future_map):
            video_name = future_map[future]
            rows, output_count, discarded_source_clips = future.result()
            rows_by_video[video_name] = rows
            total_output_clips += output_count
            total_discarded_source_clips += discarded_source_clips

    ordered_rows: list[tuple[str, str]] = []
    for video_name in sorted(rows_by_video):
        ordered_rows.extend(rows_by_video[video_name])

    write_annotations(ordered_rows, args.output_root / "annotations.csv")
    write_summary(
        args.output_root / "generation_summary.txt",
        input_root=args.input_root,
        annotations_csv=annotations_csv,
        output_root=args.output_root,
        source_spec=source_spec,
        output_fps=args.output_fps,
        output_duration_seconds=args.output_duration_seconds,
        clips_per_output=clips_per_output,
        total_videos=len(rows_by_video),
        total_clips=total_output_clips,
        discarded_source_clips=total_discarded_source_clips,
    )

    print(f"Created {total_output_clips} clips across {len(rows_by_video)} videos at {args.output_root}")


if __name__ == "__main__":
    main()
