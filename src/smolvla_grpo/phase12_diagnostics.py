"""Decoded WM prediction diagnostics for Phase12 smoke runs."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np


_AGENT_DEBUG_LOG_PATH = Path("/vol/bitbucket/aa6622/.logs/debug-588128.log")
_AGENT_DEBUG_LOG_COUNT = 0


def _agent_debug_log(*, hypothesis_id: str, location: str, message: str, data: dict[str, Any]) -> None:
    global _AGENT_DEBUG_LOG_COUNT
    if os.environ.get("AGENT_DEBUG_PHASE12_WM_ACTIONS", "").strip().lower() not in {"1", "true", "yes"}:
        return
    if _AGENT_DEBUG_LOG_COUNT >= 20:
        return
    _AGENT_DEBUG_LOG_COUNT += 1
    try:
        payload = {
            "sessionId": "588128",
            "id": f"phase12_decode_artifacts_{os.getpid()}_{_AGENT_DEBUG_LOG_COUNT}",
            "timestamp": int(time.time() * 1000),
            "runId": os.environ.get("AGENT_DEBUG_RUN_ID", "phase12-bounded-wm-issue"),
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
        }
        _AGENT_DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _AGENT_DEBUG_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, sort_keys=True) + "\n")
    except Exception:
        return


@dataclass(frozen=True)
class Phase12DecodeArtifactResult:
    paths: dict[str, Path]
    metadata: dict[str, Any]


def expected_wm_decode_steps(*, chunk_len: int, env_steps_per_wm_step: int) -> int:
    chunk = int(chunk_len)
    factor = max(1, int(env_steps_per_wm_step))
    if chunk < 1:
        raise ValueError("chunk_len must be >= 1")
    return int(np.ceil(float(chunk) / float(factor)))


def _to_rgb_uint8(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim != 3 or arr.shape[-1] not in (3, 4):
        raise ValueError(f"frame must be HxWx3/4, got {arr.shape}")
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating) and float(np.max(arr)) <= 1.5:
            arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _resize_like(frame: np.ndarray, target: np.ndarray) -> np.ndarray:
    f = _to_rgb_uint8(frame)
    t = _to_rgb_uint8(target)
    if f.shape[:2] == t.shape[:2]:
        return f
    from PIL import Image

    return np.asarray(Image.fromarray(f).resize((t.shape[1], t.shape[0])))


def _write_strip(path: Path, columns: list[np.ndarray]) -> None:
    if not columns:
        raise ValueError("cannot write empty strip")
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    first = _to_rgb_uint8(columns[0])
    strip_columns = [first] + [_resize_like(col, first) for col in columns[1:]]
    strip = np.concatenate(strip_columns, axis=1)
    imageio.imwrite(path, strip)


def _write_real_vs_pred(path: Path, real_frames: list[np.ndarray], pred_frames: list[np.ndarray]) -> None:
    if not real_frames or not pred_frames:
        raise ValueError("real and predicted frames are required")
    pairs: list[np.ndarray] = []
    for real, pred in zip(real_frames, pred_frames, strict=False):
        real_rgb = _to_rgb_uint8(real)
        pred_rgb = _resize_like(pred, real_rgb)
        pairs.append(np.concatenate([real_rgb, pred_rgb], axis=0))
    _write_strip(path, pairs)


def build_decode_artifacts(
    *,
    decode_fn: Callable[[], list[np.ndarray]],
    output_dir: Path,
    real_frames: list[np.ndarray],
    strict_decode: bool,
    segment_index: int = 0,
    selected_candidate_index: int = 0,
    env_steps_per_wm_step: int = 5,
    carried_steps: int | None = None,
) -> Phase12DecodeArtifactResult:
    output_dir = Path(output_dir)
    paths: dict[str, Path] = {}
    metadata: dict[str, Any] = {
        "decode_enabled": True,
        "decode_candidates": "selected",
        "decode_status": "unknown",
        "decode_failure_reason": None,
        "decoded_frame_count": 0,
        "env_steps_per_wm_step": int(env_steps_per_wm_step),
    }
    try:
        pred_frames = list(decode_fn())
        if not pred_frames:
            raise RuntimeError("decode returned no predicted frames")
        segment_dir = output_dir / f"segment_{int(segment_index):04d}"
        selected_path = segment_dir / "wm_decode_selected_strip.png"
        init_frame = real_frames[0] if real_frames else pred_frames[0]
        _write_strip(selected_path, [_to_rgb_uint8(init_frame)] + [_to_rgb_uint8(f) for f in pred_frames])
        paths["wm_decode_selected_strip_path"] = selected_path

        if real_frames:
            factor = max(1, int(env_steps_per_wm_step))
            carry = max(0, int(carried_steps if carried_steps is not None else len(real_frames) - 1))
            aligned_real: list[np.ndarray] = []
            aligned_pred: list[np.ndarray] = []
            aligned_indices: list[int] = []
            for k, pred in enumerate(pred_frames):
                ridx = min((k + 1) * factor, carry)
                if ridx < len(real_frames):
                    aligned_real.append(real_frames[ridx])
                    aligned_pred.append(pred)
                    aligned_indices.append(int(ridx))
            if aligned_real and aligned_pred:
                real_vs_pred_path = segment_dir / "wm_real_vs_pred_selected_strip.png"
                _write_real_vs_pred(real_vs_pred_path, aligned_real, aligned_pred)
                paths["wm_real_vs_pred_selected_strip_path"] = real_vs_pred_path
            # region agent log
            _agent_debug_log(
                hypothesis_id="H4",
                location="src/smolvla_grpo/phase12_diagnostics.py:build_decode_artifacts",
                message="phase12 aligned real frames with decoded WM prediction frames",
                data={
                    "real_frame_count": int(len(real_frames)),
                    "pred_frame_count": int(len(pred_frames)),
                    "env_steps_per_wm_step": int(factor),
                    "carried_steps": int(carry),
                    "aligned_real_indices": aligned_indices,
                    "aligned_pair_count": int(len(aligned_indices)),
                    "selected_candidate_index": int(selected_candidate_index),
                    "segment_index": int(segment_index),
                },
            )
            # endregion

        metadata["decode_status"] = "ok"
        metadata["decoded_frame_count"] = int(len(pred_frames))
    except Exception as exc:
        if strict_decode:
            raise
        metadata["decode_status"] = "failed"
        metadata["decode_failure_reason"] = str(exc)

    for key, path in paths.items():
        metadata[key] = str(path)
    return Phase12DecodeArtifactResult(paths=paths, metadata=metadata)


def write_phase12_episode_video(
    *,
    video_path: Path,
    frames: list[np.ndarray],
    rewards: list[float],
    successes: list[bool],
    fps: int,
    overlay_mode: str = "cumulative_reward",
) -> Path:
    """Write a Phase12 episode MP4 using the existing evaluator video path."""

    if not frames:
        raise RuntimeError("No frames captured for Phase12 episode video.")
    from smolvla_pipeline.evaluator import _write_episode_video

    video_path = Path(video_path)
    _write_episode_video(
        video_path=video_path,
        frames=[_to_rgb_uint8(frame) for frame in frames],
        rewards=[float(x) for x in rewards],
        successes=[bool(x) for x in successes],
        overlay_mode=str(overlay_mode),
        fps=int(fps),
    )
    return video_path

