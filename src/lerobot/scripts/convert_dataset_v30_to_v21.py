# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Convert a LeRobot dataset from codebase version v3.0 back to v2.1.

VENDORED into lerobot-new by anredlich.
Origin : NVIDIA Isaac-GR00T, scripts/lerobot_conversion/convert_v3_to_v2.py
         (via github.com/Pequi-Mecanico-Research/lerobot-v3-v2-converter)
         commit 23ace64f17aa5015259b8609d371eb61a357c776
License: Apache-2.0 (see LICENSE and NOTICE in the repo root).

Local modifications for lerobot 0.6.2 (marked with `# anr:` below):
  1. load_info/load_tasks/write_info moved to lerobot.datasets.io_utils in 0.6.x.
  2. load_info() now returns a typed DatasetInfo (not a dict) and write_info()
     expects one; we read/write meta/info.json directly as JSON instead.
  3. Added convert_dataset_path(): convert a local dataset DIRECTORY (not a
     Hub repo_id), non-destructive by default — used by lerobot-record's
     --dataset.output_version=v2.1 flag and by the standalone CLI.
  4. Preserve meta/modality.json (GR00T) if present.

Usage examples
--------------

- Convert a lerobot v3.0 dataset that already exists locally (The v3.0 path will be overwritten by the v2.1 path and a folder with the suffix _v30 will be created containing the original v3.0 dataset)

- This needs lerobot version atleast after commit f55c6e89f.

- Tested on lerobot version 0.4.0 (commit: f25ac02)
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import logging
import math
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Iterable

from huggingface_hub import snapshot_download
import jsonlines
import numpy as np
import pyarrow.parquet as pq
import tqdm


if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[3]))
from lerobot.datasets.utils import (  # anr: these still live in datasets.utils (0.6.2)
    DEFAULT_CHUNK_SIZE,
    DEFAULT_DATA_PATH,
    DEFAULT_VIDEO_PATH,
    EPISODES_DIR,
    LEGACY_EPISODES_PATH,
    LEGACY_EPISODES_STATS_PATH,
    LEGACY_TASKS_PATH,
    serialize_dict,
    unflatten_dict,
)
from lerobot.datasets.io_utils import load_tasks  # anr: moved here in 0.6.x
from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.utils.utils import init_logging


V21 = "v2.1"
V30 = "v3.0"

LEGACY_DATA_PATH_TEMPLATE = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
LEGACY_VIDEO_PATH_TEMPLATE = (
    "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
)

MIN_VIDEO_DURATION = 1e-6

INFO_JSON_REL = "meta/info.json"  # anr


def _read_info(root: Path) -> dict:
    """anr: read meta/info.json as a plain dict (0.6.2 load_info() returns a
    typed DatasetInfo, which breaks the dict mutation convert_info() needs)."""
    import json

    with open(Path(root) / INFO_JSON_REL) as f:
        return json.load(f)


def _write_info(info: dict, new_root: Path) -> None:
    """anr: write meta/info.json directly (0.6.2 write_info() expects a DatasetInfo)."""
    import json

    out = Path(new_root) / INFO_JSON_REL
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(info, f, indent=4)


def _to_serializable(value: Any) -> Any:
    """Convert numpy/pyarrow values into standard Python types for JSON dumps."""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_to_serializable(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_serializable(val) for key, val in value.items()}
    return value


def validate_local_dataset_version(local_path: Path) -> None:
    info = _read_info(local_path)  # anr
    dataset_version = info.get("codebase_version", "unknown")
    if dataset_version != V30:
        raise ValueError(
            f"Local dataset has codebase version '{dataset_version}', expected '{V30}'. "
            f"This script converts datasets from v3.0 back to v2.1."
        )


def load_episode_records(root: Path) -> list[dict[str, Any]]:
    """Load the consolidated metadata rows stored in ``meta/episodes``."""

    episodes_dir = root / EPISODES_DIR
    pq_paths = sorted(episodes_dir.glob("chunk-*/file-*.parquet"))
    if not pq_paths:
        raise FileNotFoundError(f"No episode parquet files found in {episodes_dir}.")

    records: list[dict[str, Any]] = []
    for pq_path in pq_paths:
        table = pq.read_table(pq_path)
        records.extend(table.to_pylist())

    records.sort(key=lambda rec: int(rec["episode_index"]))
    return records


def convert_tasks(root: Path, new_root: Path) -> None:
    logging.info("Converting tasks parquet to legacy JSONL")
    tasks = load_tasks(root)
    tasks = tasks.sort_values("task_index")

    out_path = new_root / LEGACY_TASKS_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with jsonlines.open(out_path, mode="w") as writer:
        for task, row in tasks.iterrows():
            writer.write(
                {
                    "task_index": int(row["task_index"]),
                    "task": _to_serializable(task),
                }
            )


def convert_info(
    root: Path,
    new_root: Path,
    episode_records: list[dict[str, Any]],
    video_keys: list[str],
    trossen_subversion: str | None = None,  # anr
    rename_motors: list[str] | None = None,  # anr
) -> None:
    info = _read_info(root)  # anr
    logging.info("Converting info.json metadata to v2.1 schema")

    total_episodes = info.get("total_episodes") or len(episode_records)
    chunks_size = info.get("chunks_size", DEFAULT_CHUNK_SIZE)

    info["codebase_version"] = V21

    # Restore legacy layout templates.
    info["data_path"] = LEGACY_DATA_PATH_TEMPLATE
    if info.get("video_path") is not None and len(video_keys) > 0:
        info["video_path"] = LEGACY_VIDEO_PATH_TEMPLATE
    else:
        info["video_path"] = None

    # Remove v3-specific sizing hints which do not exist in v2.1.
    info.pop("data_files_size_in_mb", None)
    info.pop("video_files_size_in_mb", None)

    # Restore per-feature metadata: camera entries already contain their own fps.
    for key, ft in info["features"].items():
        if ft.get("dtype") != "video":
            ft.pop("fps", None)

    info["total_chunks"] = math.ceil(total_episodes / chunks_size) if total_episodes > 0 else 0
    info["total_videos"] = total_episodes * len(video_keys)

    # anr: rename the state/action motor names (metadata only; v3.0 keeps its native names).
    if rename_motors is not None:
        for key in ("action", "observation.state"):
            ft = info.get("features", {}).get(key)
            if ft is None:
                continue
            old_names = ft.get("names")
            if old_names is not None and len(old_names) != len(rename_motors):
                raise ValueError(
                    f"rename_motors has {len(rename_motors)} names but feature '{key}' "
                    f"has {len(old_names)} ({old_names}); counts must match."
                )
            ft["names"] = list(rename_motors)
        logging.info("Renamed state/action motors to: %s", rename_motors)

    # anr: stamp trossen_subversion right after codebase_version (matches the old upgrade script),
    # so the Interbotix/LeRobot loader and openpi/GR00T accept the dataset.
    if trossen_subversion:
        reordered = {}
        for k, v in info.items():
            reordered[k] = v
            if k == "codebase_version":
                reordered["trossen_subversion"] = trossen_subversion
        info = reordered
        logging.info("Stamped trossen_subversion: %s", trossen_subversion)

    _write_info(info, new_root)  # anr


def _group_episodes_by_data_file(
    episode_records: Iterable[dict[str, Any]],
) -> dict[tuple[int, int], list[dict[str, Any]]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for record in episode_records:
        key = (
            int(record["data/chunk_index"]),
            int(record["data/file_index"]),
        )
        grouped[key].append(record)
    return grouped


def convert_data(
    root: Path,
    new_root: Path,
    episode_records: list[dict[str, Any]],
    chunks_size: int,
    skip_existing: bool = True,  # anr: incremental resume support
) -> None:
    logging.info("Converting consolidated parquet files back to per-episode files")
    grouped = _group_episodes_by_data_file(episode_records)

    for (chunk_idx, file_idx), records in tqdm.tqdm(grouped.items(), desc="convert data files"):
        records = sorted(records, key=lambda rec: int(rec["dataset_from_index"]))

        # anr: if every episode in this source file is already converted, don't even read it.
        def _dest(rec):
            ep = int(rec["episode_index"])
            return new_root / LEGACY_DATA_PATH_TEMPLATE.format(
                episode_chunk=ep // chunks_size, episode_index=ep
            )

        if skip_existing and all(_dest(rec).exists() for rec in records):
            continue

        source_path = root / DEFAULT_DATA_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
        if not source_path.exists():
            raise FileNotFoundError(f"Expected source parquet file not found: {source_path}")

        table = pq.read_table(source_path)
        file_offset = int(records[0]["dataset_from_index"])

        for record in records:
            episode_index = int(record["episode_index"])

            dest_chunk = episode_index // chunks_size
            dest_path = new_root / LEGACY_DATA_PATH_TEMPLATE.format(
                episode_chunk=dest_chunk,
                episode_index=episode_index,
            )
            if skip_existing and dest_path.exists():  # anr: already converted (resume)
                continue

            start = int(record["dataset_from_index"]) - file_offset
            stop = int(record["dataset_to_index"]) - file_offset
            length = stop - start

            if length <= 0:
                raise ValueError(
                    "Invalid episode length computed during data conversion: "
                    f"episode_index={episode_index}, length={length}"
                )

            episode_table = table.slice(start, length)

            # anr: drop the embedded HuggingFace-`datasets` schema metadata. lerobot 0.6.2
            # writes array columns with the newer "List" feature type, which older `datasets`
            # (the old-fork / openpi / GR00T envs that read v2.1) can't parse. Removing the
            # "huggingface" key makes those envs infer `Sequence` from the Arrow schema instead.
            _md = episode_table.schema.metadata or {}
            _md = {k: v for k, v in _md.items() if k != b"huggingface"}
            episode_table = episode_table.replace_schema_metadata(_md or None)

            dest_path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(episode_table, dest_path)


def _group_episodes_by_video_file(
    episode_records: Iterable[dict[str, Any]],
    video_key: str,
) -> dict[tuple[int, int], list[dict[str, Any]]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    chunk_column = f"videos/{video_key}/chunk_index"
    file_column = f"videos/{video_key}/file_index"

    for record in episode_records:
        if chunk_column not in record or file_column not in record:
            continue
        chunk_idx = record.get(chunk_column)
        file_idx = record.get(file_column)
        if chunk_idx is None or file_idx is None:
            continue
        grouped[(int(chunk_idx), int(file_idx))].append(record)
    return grouped


def _validate_video_paths(src: Path, dst: Path) -> None:
    """Validate source and destination paths to prevent security issues."""

    # Convert to Path objects if they aren't already
    src = Path(src)
    dst = Path(dst)

    # Resolve paths to handle symlinks and normalize them
    try:
        src_resolved = src.resolve()
        dst_resolved = dst.resolve()
    except OSError as exc:
        raise ValueError(f"Invalid path provided: {exc}") from exc

    # Check that source file exists and is a regular file
    if not src_resolved.exists():
        raise FileNotFoundError(f"Source video file does not exist: {src_resolved}")

    if not src_resolved.is_file():
        raise ValueError(f"Source path is not a regular file: {src_resolved}")

    # Validate file extensions for video files
    valid_video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
    if src_resolved.suffix.lower() not in valid_video_extensions:
        raise ValueError(f"Source file does not have a valid video extension: {src_resolved}")

    if dst_resolved.suffix.lower() not in valid_video_extensions:
        raise ValueError(f"Destination file does not have a valid video extension: {dst_resolved}")

    # Check for path traversal attempts in the original paths
    src_str = str(src)
    dst_str = str(dst)

    # Ensure paths don't contain null bytes or other control characters
    for path_str, name in [(src_str, "source"), (dst_str, "destination")]:
        if "\0" in path_str:
            raise ValueError(f"Path contains null bytes: {name} path")
        if any(ord(c) < 32 and c not in ["\t", "\n", "\r"] for c in path_str):
            raise ValueError(f"Path contains invalid control characters: {name} path")

    # Additional check: ensure resolved paths don't point to system directories
    system_dirs = {"/etc", "/sys", "/proc", "/dev", "/boot", "/root"}
    for resolved_path, name in [(src_resolved, "source"), (dst_resolved, "destination")]:
        path_str = str(resolved_path)
        for sys_dir in system_dirs:
            if path_str.startswith(sys_dir + "/") or path_str == sys_dir:
                raise ValueError(f"Path points to system directory: {name} path {resolved_path}")

    # Ensure the destination directory can be created safely
    try:
        dst_parent = dst_resolved.parent
        if not dst_parent.exists():
            # Check if we can create the parent directory structure
            dst_parent.resolve()
    except OSError as exc:
        raise ValueError(f"Cannot create destination directory: {exc}") from exc


def _extract_video_segment(
    src: Path,
    dst: Path,
    start: float,
    end: float,
) -> None:
    # Validate paths to prevent security issues
    _validate_video_paths(src, dst)

    # Validate numeric parameters to prevent injection
    if not (0 <= start <= 86400):  # 24 hours max
        raise ValueError(f"Invalid start time: {start}")
    if not (0 <= end <= 86400):  # 24 hours max
        raise ValueError(f"Invalid end time: {end}")
    if start >= end:
        raise ValueError(f"Start time {start} must be less than end time {end}")

    duration = max(end - start, MIN_VIDEO_DURATION)

    # Validate duration is reasonable
    if duration > 3600:  # 1 hour max
        raise ValueError(f"Video segment duration too long: {duration} seconds")

    dst.parent.mkdir(parents=True, exist_ok=True)

    # Build command with validated parameters
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start:.6f}",
        "-i",
        str(src),
        "-t",
        f"{duration:.6f}",
        "-c",
        "copy",
        "-avoid_negative_ts",
        "1",
        "-y",
        str(dst),
    ]

    try:
        # Use more secure subprocess call with explicit timeout
        subprocess.run(
            cmd,
            check=True,
            timeout=300,
            capture_output=True,
            text=True,  # 5 minute timeout
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"ffmpeg timed out while processing video '{src}' -> '{dst}'") from exc
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffmpeg executable not found; it is required for video conversion"
        ) from exc
    except subprocess.CalledProcessError as exc:
        error_msg = f"ffmpeg failed while splitting video '{src}' into '{dst}'"
        if exc.stderr:
            error_msg += f". Error: {exc.stderr.strip()}"
        raise RuntimeError(error_msg) from exc


def convert_videos(
    root: Path,
    new_root: Path,
    episode_records: list[dict[str, Any]],
    video_keys: list[str],
    chunks_size: int,
    skip_existing: bool = True,  # anr: incremental resume support
) -> None:
    if len(video_keys) == 0:
        logging.info("No video features detected; skipping video conversion")
        return

    logging.info("Converting concatenated MP4 files back to per-episode videos")

    for video_key in video_keys:
        grouped = _group_episodes_by_video_file(episode_records, video_key)
        if len(grouped) == 0:
            logging.info("No video metadata found for key '%s'; skipping", video_key)
            continue

        for (chunk_idx, file_idx), records in tqdm.tqdm(
            grouped.items(), desc=f"convert videos ({video_key})"
        ):
            src_path = root / DEFAULT_VIDEO_PATH.format(
                video_key=video_key,
                chunk_index=chunk_idx,
                file_index=file_idx,
            )
            if not src_path.exists():
                raise FileNotFoundError(f"Expected MP4 file not found: {src_path}")

            records = sorted(
                records, key=lambda rec: float(rec[f"videos/{video_key}/from_timestamp"])
            )

            for record in records:
                episode_index = int(record["episode_index"])
                start = float(record[f"videos/{video_key}/from_timestamp"])
                end = float(record[f"videos/{video_key}/to_timestamp"])

                dest_chunk = episode_index // chunks_size
                dest_path = new_root / LEGACY_VIDEO_PATH_TEMPLATE.format(
                    episode_chunk=dest_chunk,
                    video_key=video_key,
                    episode_index=episode_index,
                )
                if skip_existing and dest_path.exists():  # anr: already converted (resume)
                    continue

                _extract_video_segment(src_path, dest_path, start=start, end=end)


def convert_episodes_metadata(new_root: Path, episode_records: list[dict[str, Any]]) -> None:
    logging.info("Reconstructing legacy episodes and episodes_stats JSONL files")

    episodes_path = new_root / LEGACY_EPISODES_PATH
    stats_path = new_root / LEGACY_EPISODES_STATS_PATH
    episodes_path.parent.mkdir(parents=True, exist_ok=True)

    with (
        jsonlines.open(episodes_path, mode="w") as episodes_writer,
        jsonlines.open(stats_path, mode="w") as stats_writer,
    ):
        for record in sorted(episode_records, key=lambda rec: int(rec["episode_index"])):
            legacy_episode = {
                key: value
                for key, value in record.items()
                if not key.startswith("data/")
                and not key.startswith("videos/")
                and not key.startswith("stats/")
                and not key.startswith("meta/")
                and key not in {"dataset_from_index", "dataset_to_index"}
            }

            # Ensure legacy episodes include a length; compute from dataset indices if missing
            if "length" not in legacy_episode:
                if "dataset_from_index" in record and "dataset_to_index" in record:
                    legacy_episode["length"] = int(record["dataset_to_index"]) - int(
                        record["dataset_from_index"]
                    )

            serializable_episode = {
                key: _to_serializable(value) for key, value in legacy_episode.items()
            }
            episodes_writer.write(serializable_episode)

            stats_flat = {key: record[key] for key in record if key.startswith("stats/")}
            stats_nested = unflatten_dict(stats_flat).get("stats", {})
            stats_serialized = serialize_dict(stats_nested)
            stats_writer.write(
                {
                    "episode_index": int(record["episode_index"]),
                    "stats": stats_serialized,
                }
            )


def copy_global_stats(root: Path, new_root: Path) -> None:
    source_stats = root / "meta" / "stats.json"
    if source_stats.exists():
        target_stats = new_root / "meta" / "stats.json"
        target_stats.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_stats, target_stats)


def copy_ancillary_directories(root: Path, new_root: Path) -> None:
    for subdir in ["images"]:
        source = root / subdir
        if source.exists():
            shutil.copytree(source, new_root / subdir, dirs_exist_ok=True)
    # anr: preserve GR00T modality.json (and any other meta/*.json we don't regenerate)
    for extra in ["meta/modality.json"]:
        source = root / extra
        if source.exists():
            dest = new_root / extra
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)


def convert_dataset(
    repo_id: str,
    root: str | Path | None = None,
    force_conversion: bool = False,
) -> None:
    root = HF_LEROBOT_HOME / repo_id if root is None else Path(root) / repo_id

    if root.exists() and force_conversion:
        logging.info("--force-conversion enabled: removing existing snapshot at %s", root)
        shutil.rmtree(root)

    if root.exists():
        validate_local_dataset_version(root)
        logging.info("Using existing local dataset at %s", root)
    else:
        logging.info("Downloading dataset snapshot from the Hub")
        snapshot_download(repo_id, repo_type="dataset", local_dir=root)

    episode_records = load_episode_records(root)
    info = _read_info(root)  # anr
    video_keys = [key for key, ft in info["features"].items() if ft.get("dtype") == "video"]
    chunks_size = info.get("chunks_size", DEFAULT_CHUNK_SIZE)

    backup_root = root.parent / f"{root.name}_{V30}"
    new_root = root.parent / f"{root.name}_{V21}"

    if backup_root.is_dir():
        shutil.rmtree(backup_root)
    if new_root.is_dir():
        shutil.rmtree(new_root)

    new_root.mkdir(parents=True, exist_ok=True)

    convert_info(root, new_root, episode_records, video_keys)  # anr: Hub path keeps native names
    copy_global_stats(root, new_root)
    convert_tasks(root, new_root)
    convert_data(root, new_root, episode_records, chunks_size)
    convert_videos(root, new_root, episode_records, video_keys, chunks_size)
    convert_episodes_metadata(new_root, episode_records)
    copy_ancillary_directories(root, new_root)

    shutil.move(str(root), str(backup_root))
    shutil.move(str(new_root), str(root))


def convert_dataset_path(
    dataset_dir: str | Path,
    output_dir: str | Path | None = None,
    delete_source: bool = False,
    incremental: bool = True,
    trossen_subversion: str | None = None,  # anr
    rename_motors: list[str] | None = None,  # anr
) -> Path:
    """anr: convert a local v3.0 dataset DIRECTORY to v2.1 (non-destructive by default).

    dataset_dir : path to the v3.0 dataset root (contains meta/, data/, videos/).
    output_dir  : where to write the v2.1 dataset. If None, writes to a sibling
                  '<name>_v21' and leaves the v3.0 dataset intact unless delete_source.
    Returns the path to the resulting v2.1 dataset directory.

    Used by lerobot-record's --dataset.output_version=v2.1 flag and by the CLI.
    Unlike convert_dataset() this takes a plain path, not a Hub repo_id, and never
    does a Hub snapshot_download or an in-place destructive move.
    """
    root = Path(dataset_dir)
    validate_local_dataset_version(root)

    info = _read_info(root)
    video_keys = [key for key, ft in info["features"].items() if ft.get("dtype") == "video"]
    chunks_size = info.get("chunks_size", DEFAULT_CHUNK_SIZE)
    episode_records = load_episode_records(root)

    new_root = Path(output_dir) if output_dir is not None else root.parent / f"{root.name}_v21"
    if new_root.resolve() == root.resolve():
        raise ValueError("output_dir must differ from dataset_dir")
    if new_root.is_dir() and not incremental:  # anr: keep existing output for incremental resume
        shutil.rmtree(new_root)
    new_root.mkdir(parents=True, exist_ok=True)

    # anr: metadata is always regenerated in full (cheap); data/videos skip already-converted
    # episodes so a --resume only processes the newly added ones.
    convert_info(
        root,
        new_root,
        episode_records,
        video_keys,
        trossen_subversion=trossen_subversion,  # anr
        rename_motors=rename_motors,  # anr
    )
    copy_global_stats(root, new_root)
    convert_tasks(root, new_root)
    convert_data(root, new_root, episode_records, chunks_size, skip_existing=incremental)
    convert_videos(root, new_root, episode_records, video_keys, chunks_size, skip_existing=incremental)
    convert_episodes_metadata(new_root, episode_records)
    copy_ancillary_directories(root, new_root)

    if delete_source:
        shutil.rmtree(root)
    logging.info("v2.1 dataset written to %s", new_root)
    return new_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-id",
        type=str,
        required=False,  # anr: optional when --dataset-dir is used
        help="Repository identifier on Hugging Face (e.g. `lerobot/pusht`).",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Local directory under which the dataset should be stored.",
    )
    parser.add_argument(
        "--force-conversion",
        action="store_true",
        help="Ignore any existing local snapshot and re-download it from the Hub.",
    )
    # anr: local-directory mode (no Hub)
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="anr: convert a local v3.0 dataset DIRECTORY (contains meta/, data/, videos/). "
        "Mutually exclusive with --repo-id.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="anr: with --dataset-dir, where to write the v2.1 dataset "
        "(default: sibling '<name>_v21').",
    )
    parser.add_argument(
        "--delete-source",
        action="store_true",
        help="anr: with --dataset-dir, delete the v3.0 source after a successful conversion.",
    )
    parser.add_argument(
        "--trossen-subversion",
        type=str,
        default="v1.0",
        help="anr: with --dataset-dir, stamp this trossen_subversion into the v2.1 info.json "
        "(default v1.0; pass '' to skip).",
    )
    parser.add_argument(
        "--joint-names",
        type=str,
        nargs="*",
        default=None,
        help="anr: with --dataset-dir, rename the state/action motor names in the v2.1 "
        "info.json (space-separated; count must match the feature dim).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    init_logging()
    args = parse_args()
    if args.dataset_dir is not None:  # anr: local-directory mode
        convert_dataset_path(
            args.dataset_dir,
            output_dir=args.output_dir,
            delete_source=args.delete_source,
            trossen_subversion=(args.trossen_subversion or None),
            rename_motors=args.joint_names,
        )
    else:
        if not args.repo_id:
            raise SystemExit("Provide either --repo-id or --dataset-dir.")
        convert_dataset(
            repo_id=args.repo_id, root=args.root, force_conversion=args.force_conversion
        )
