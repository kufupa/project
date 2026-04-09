#!/usr/bin/env python3
"""Extract one episode's PNG bytes from parquet and render an mp4.

This script is designed for datasets where `observation.image` is stored as:
  {"bytes": <png-bytes>, "path": <optional external path>}
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from PIL import Image


@dataclass(frozen=True)
class EpisodeFrame:
    frame_index: int
    png_bytes: bytes
    source_parquet: str


def _iter_parquet_files(dataset_root: Path) -> list[Path]:
    data_root = dataset_root / "data"
    search_root = data_root if data_root.exists() else dataset_root
    files = sorted(search_root.rglob("file-*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No parquet files found under {search_root}. Expected file-*.parquet."
        )
    return files


def _resolve_image_bytes(cell: Any, parquet_path: Path) -> bytes:
    if isinstance(cell, dict):
        payload = cell.get("bytes")
        if isinstance(payload, bytes):
            return payload
        if isinstance(payload, bytearray):
            return bytes(payload)

        img_path = cell.get("path")
        if isinstance(img_path, str) and img_path.strip():
            candidate = Path(img_path)
            if not candidate.is_absolute():
                candidate = (parquet_path.parent / candidate).resolve()
            if candidate.is_file():
                return candidate.read_bytes()
            raise FileNotFoundError(
                f"Image path from parquet row does not exist: {candidate}"
            )
        raise ValueError(
            "observation.image dict missing both usable `bytes` and `path` fields."
        )

    raise TypeError(
        f"Unsupported observation.image payload type: {type(cell).__name__}"
    )


def collect_episode_png_bytes(dataset_root: Path | str, episode_index: int) -> list[EpisodeFrame]:
    root = Path(dataset_root).expanduser().resolve()
    frames: list[EpisodeFrame] = []

    for parquet_file in _iter_parquet_files(root):
        table = pq.read_table(
            parquet_file,
            columns=["episode_index", "frame_index", "observation.image"],
        )
        for row in table.to_pylist():
            ep = int(row.get("episode_index", -1))
            if ep != int(episode_index):
                continue
            frame_index = int(row.get("frame_index", 0))
            png_bytes = _resolve_image_bytes(row.get("observation.image"), parquet_file)
            frames.append(
                EpisodeFrame(
                    frame_index=frame_index,
                    png_bytes=png_bytes,
                    source_parquet=str(parquet_file),
                )
            )

    if not frames:
        raise ValueError(
            f"No frames found for episode_index={episode_index} under {root}"
        )
    frames.sort(key=lambda item: item.frame_index)
    return frames


def _png_bytes_to_rgb_array(png_bytes: bytes) -> np.ndarray:
    arr = np.asarray(Image.open(io.BytesIO(png_bytes)).convert("RGB"), dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Expected RGB image, got shape={arr.shape}")
    return arr


def _to_uint8_rgb(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr)
    if out.ndim == 2:
        out = np.repeat(out[:, :, None], 3, axis=2)
    if out.ndim == 3 and out.shape[-1] == 4:
        out = out[:, :, :3]
    if out.dtype != np.uint8:
        out = np.clip(out, 0, 255).astype(np.uint8)
    if out.ndim != 3 or out.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB array, got shape={out.shape}")
    return out


def write_episode_video(
    frames: list[EpisodeFrame],
    output_path: Path | str,
    fps: int = 30,
) -> None:
    if not frames:
        raise ValueError("Cannot write video from empty frame list.")

    out = Path(output_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    decoded = [_png_bytes_to_rgb_array(frame.png_bytes) for frame in frames]
    first_shape = decoded[0].shape
    for idx, arr in enumerate(decoded):
        if arr.shape != first_shape:
            raise ValueError(
                f"Inconsistent frame shape at index={idx}: {arr.shape} vs {first_shape}"
            )

    # Prefer OpenCV in this environment; fallback to imageio if unavailable.
    try:
        import cv2  # type: ignore

        height, width = first_shape[0], first_shape[1]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out), fourcc, float(fps), (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"OpenCV VideoWriter failed to open output: {out}")
        for arr in decoded:
            writer.write(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
        writer.release()
    except Exception:
        import imageio.v2 as imageio  # type: ignore

        with imageio.get_writer(str(out), fps=fps, codec="libx264") as writer:
            for arr in decoded:
                writer.append_data(arr)

    if not out.is_file() or out.stat().st_size <= 0:
        raise RuntimeError(f"Video output was not created successfully: {out}")


def _frame_array_hash(arr: np.ndarray) -> str:
    safe = _to_uint8_rgb(arr)
    h = hashlib.sha256()
    h.update(str(safe.shape).encode("utf-8"))
    h.update(safe.tobytes())
    return h.hexdigest()


def resolve_source_episode_from_first_frame(
    first_png_bytes: bytes,
    source_episodes_root: Path | str | None,
) -> str | None:
    if not source_episodes_root:
        return None
    root = Path(source_episodes_root).expanduser().resolve()
    if not root.is_dir():
        return None

    import torch

    target_hash = _frame_array_hash(_png_bytes_to_rgb_array(first_png_bytes))
    for episode_path in sorted(root.glob("episode_*.pt")):
        try:
            payload = torch.load(episode_path, map_location="cpu", weights_only=False)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        images = payload.get("images")
        if not isinstance(images, list) or not images:
            continue
        try:
            source_hash = _frame_array_hash(np.asarray(images[0]))
        except Exception:
            continue
        if source_hash == target_hash:
            return str(episode_path)
    return None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="LeRobot dataset split root (e.g. .../train or .../val).",
    )
    parser.add_argument(
        "--episode-index",
        type=int,
        required=True,
        help="episode_index to render.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output mp4 path.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Output video FPS (default: 30).",
    )
    parser.add_argument(
        "--source-episodes-root",
        default="",
        help="Optional root containing source episode_*.pt files for provenance lookup.",
    )
    parser.add_argument(
        "--report-json",
        default="",
        help="Optional output JSON report path. Defaults to <output>.json",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    report = (
        Path(args.report_json).expanduser().resolve()
        if args.report_json
        else output.with_suffix(output.suffix + ".json")
    )

    frames = collect_episode_png_bytes(dataset_root, episode_index=args.episode_index)
    write_episode_video(frames, output_path=output, fps=args.fps)
    source_episode = resolve_source_episode_from_first_frame(
        first_png_bytes=frames[0].png_bytes,
        source_episodes_root=args.source_episodes_root or None,
    )

    payload = {
        "dataset_root": str(dataset_root),
        "episode_index": int(args.episode_index),
        "frame_count": len(frames),
        "first_frame_source_parquet": frames[0].source_parquet,
        "output_video": str(output),
        "source_episode_match": source_episode,
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
