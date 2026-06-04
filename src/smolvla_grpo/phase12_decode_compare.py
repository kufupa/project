"""Utilities for comparing Phase12 JEPA decodes from shared actions."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class ActionVariants:
    env_actions: np.ndarray
    raw_wm_actions: np.ndarray
    bounded_wm_actions: np.ndarray
    metadata: dict[str, Any]


@dataclass(frozen=True)
class Phase12LatentTrace:
    visual_latents: list[Any]
    proprio_latents: list[Any]
    structured_latents: list[dict[str, Any]]
    frames: list[Any] | None
    wm_factor: int


def build_action_variants(
    *,
    raw_actions: np.ndarray,
    clipped_actions: np.ndarray | None,
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> ActionVariants:
    raw = np.asarray(raw_actions, dtype=np.float32)
    if raw.ndim != 2:
        raise ValueError(f"raw_actions must be 2-D, got {raw.shape}")
    low = np.asarray(action_low, dtype=np.float32).reshape(1, -1)
    high = np.asarray(action_high, dtype=np.float32).reshape(1, -1)
    if clipped_actions is None:
        bounded = np.clip(raw, low, high).astype(np.float32, copy=False)
    else:
        bounded = np.asarray(clipped_actions, dtype=np.float32)
    if bounded.shape != raw.shape:
        raise ValueError(f"bounded actions shape {bounded.shape} != raw shape {raw.shape}")
    delta = raw - bounded
    changed = raw != bounded
    return ActionVariants(
        env_actions=raw,
        raw_wm_actions=raw,
        bounded_wm_actions=bounded,
        metadata={
            "env_action_source": "raw_postprocessed",
            "raw_wm_action_source": "raw_postprocessed",
            "bounded_wm_action_source": "clipped",
            "raw_action_max_abs": float(np.max(np.abs(raw))) if raw.size else 0.0,
            "bounded_action_max_abs": float(np.max(np.abs(bounded))) if bounded.size else 0.0,
            "clip_delta_max_abs": float(np.max(np.abs(delta))) if delta.size else 0.0,
            "clip_fraction": float(np.mean(changed)) if raw.size else 0.0,
            "clip_any": bool(np.any(changed)),
        },
    )


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


def aligned_real_indices(
    *,
    pred_count: int,
    env_steps_per_wm_step: int,
    carried_steps: int,
    real_frame_count: int,
) -> list[int]:
    factor = max(1, int(env_steps_per_wm_step))
    carry = max(0, int(carried_steps))
    indices: list[int] = []
    for k in range(int(pred_count)):
        ridx = min((k + 1) * factor, carry)
        if ridx >= int(real_frame_count):
            break
        indices.append(int(ridx))
    if not indices:
        raise ValueError("no aligned decode frames")
    return indices


def align_decode_rows(
    *,
    real_frames: list[np.ndarray],
    raw_pred_frames: list[np.ndarray],
    bounded_pred_frames: list[np.ndarray],
    env_steps_per_wm_step: int,
    carried_steps: int,
) -> tuple[list[list[np.ndarray]], list[int]]:
    count = min(len(raw_pred_frames), len(bounded_pred_frames))
    rows: list[list[np.ndarray]] = [[], [], []]
    indices = aligned_real_indices(
        pred_count=count,
        env_steps_per_wm_step=env_steps_per_wm_step,
        carried_steps=carried_steps,
        real_frame_count=len(real_frames),
    )
    for k, ridx in enumerate(indices):
        real_rgb = _to_rgb_uint8(real_frames[ridx])
        rows[0].append(real_rgb)
        rows[1].append(_resize_like(raw_pred_frames[k], real_rgb))
        rows[2].append(_resize_like(bounded_pred_frames[k], real_rgb))
    return rows, indices


def write_three_row_decode_strip(
    path: Path,
    *,
    real_frames: list[np.ndarray],
    raw_pred_frames: list[np.ndarray],
    bounded_pred_frames: list[np.ndarray],
    env_steps_per_wm_step: int,
    carried_steps: int,
) -> Path:
    import imageio.v2 as imageio

    rows, _indices = align_decode_rows(
        real_frames=real_frames,
        raw_pred_frames=raw_pred_frames,
        bounded_pred_frames=bounded_pred_frames,
        env_steps_per_wm_step=env_steps_per_wm_step,
        carried_steps=carried_steps,
    )
    row_images = [np.concatenate(row, axis=1) for row in rows]
    strip = np.concatenate(row_images, axis=0)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, strip)
    return path


def _as_l2_labels(values: Sequence[float] | None, prefix: str, count: int) -> list[str]:
    if values is None:
        return [""] * int(count)
    vals = [float(v) for v in values]
    return [f"{prefix} L2={vals[i]:.1f}" if i < len(vals) else "" for i in range(int(count))]


def write_three_row_decode_strip_with_l2(
    path: Path,
    *,
    real_frames: list[np.ndarray],
    raw_pred_frames: list[np.ndarray],
    bounded_pred_frames: list[np.ndarray],
    env_steps_per_wm_step: int,
    carried_steps: int,
    raw_combined_l2: Sequence[float] | None = None,
    bounded_combined_l2: Sequence[float] | None = None,
) -> Path:
    from PIL import Image, ImageDraw

    rows, _indices = align_decode_rows(
        real_frames=real_frames,
        raw_pred_frames=raw_pred_frames,
        bounded_pred_frames=bounded_pred_frames,
        env_steps_per_wm_step=env_steps_per_wm_step,
        carried_steps=carried_steps,
    )
    count = len(rows[0])
    band_h = 18
    raw_labels = _as_l2_labels(raw_combined_l2, "raw", count)
    bounded_labels = _as_l2_labels(bounded_combined_l2, "clip", count)

    rendered_rows: list[np.ndarray] = []
    for row_idx, row in enumerate(rows):
        cells: list[np.ndarray] = []
        for col_idx, frame in enumerate(row):
            rgb = _to_rgb_uint8(frame)
            canvas = Image.new("RGB", (rgb.shape[1], rgb.shape[0] + band_h), color=(255, 255, 255))
            canvas.paste(Image.fromarray(rgb), (0, 0))
            label = ""
            if row_idx == 1:
                label = raw_labels[col_idx]
            elif row_idx == 2:
                label = bounded_labels[col_idx]
            if label:
                ImageDraw.Draw(canvas).text((3, rgb.shape[0] + 2), label, fill=(0, 0, 0))
            cells.append(np.asarray(canvas))
        rendered_rows.append(np.concatenate(cells, axis=1))
    strip = np.concatenate(rendered_rows, axis=0)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    import imageio.v2 as imageio

    imageio.imwrite(path, strip)
    return path


def write_actions_npz(
    path: Path,
    *,
    raw_actions: np.ndarray,
    bounded_actions: np.ndarray,
    env_actions: np.ndarray,
    sampled_raw_actions: np.ndarray | None = None,
    sampled_bounded_actions: np.ndarray | None = None,
) -> Path:
    raw = np.asarray(raw_actions, dtype=np.float32)
    bounded = np.asarray(bounded_actions, dtype=np.float32)
    env = np.asarray(env_actions, dtype=np.float32)
    payload: dict[str, np.ndarray] = {
        "raw_actions": raw,
        "bounded_actions": bounded,
        "env_actions": env,
        "clip_delta": raw - bounded,
    }
    if sampled_raw_actions is not None:
        payload["sampled_raw_actions"] = np.asarray(sampled_raw_actions, dtype=np.float32)
    if sampled_bounded_actions is not None:
        payload["sampled_bounded_actions"] = np.asarray(sampled_bounded_actions, dtype=np.float32)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    return path


def _structured_field(value: Any, key: str) -> Any:
    try:
        if key in value:
            return value[key]
    except Exception:
        pass
    if hasattr(value, "get"):
        try:
            return value.get(key)
        except Exception:
            pass
    try:
        return value[key]
    except Exception:
        return None


def _split_latent_for_mode(value: Any, *, mode: str) -> dict[str, Any]:
    if str(mode) == "visual_only_ablation":
        visual = _structured_field(value, "visual")
        return {"visual": visual if visual is not None else value}
    visual = _structured_field(value, "visual")
    proprio = _structured_field(value, "proprio")
    if visual is None:
        visual = value
    return {"visual": visual, "proprio": proprio}


def _latent_context_for_mode(latent: Mapping[str, Any], *, mode: str) -> Any:
    if str(mode) == "visual_only_ablation":
        return latent["visual"]
    try:
        from tensordict import TensorDict

        return TensorDict({"visual": latent["visual"], "proprio": latent["proprio"]}, batch_size=[])
    except Exception:
        return {"visual": latent["visual"], "proprio": latent["proprio"]}


def _build_wm_action_tensor(wm_bundle: Any, actions: Any) -> tuple[Any, int]:
    import torch
    from segment_grpo_loop import (
        _infer_env_action_dim,
        _infer_model_action_dim,
        _normalize_env_actions_for_wm,
        _pack_env_actions_for_wm,
        _wm_action_block_factor,
    )

    actions_np = np.asarray(actions, dtype=np.float32)
    env_dim = _infer_env_action_dim(wm_bundle, actions_np)
    model_action_dim = _infer_model_action_dim(wm_bundle.model)
    wm_dim = int(model_action_dim) if model_action_dim else int(wm_bundle.planner_action_dim)
    factor = _wm_action_block_factor(env_dim, wm_dim)
    normalized = _normalize_env_actions_for_wm(
        wm_bundle.preprocessor,
        actions_np[:, :env_dim],
        env_dim,
        wm_bundle.device,
    )
    packed = _pack_env_actions_for_wm(normalized, factor, env_dim, wm_dim)
    return torch.from_numpy(packed).to(wm_bundle.device).float().unsqueeze(1), int(factor)


def unroll_phase12_latent_trace(
    wm_bundle: Any,
    *,
    image: Any,
    proprio: Any,
    actions: Any,
    mode: str,
    decode_frames: bool = False,
) -> Phase12LatentTrace:
    import torch
    from segment_grpo_loop import (
        DecodeTrace,
        _decode_latent_trace_to_frames,
        _next_latent_state_after_unroll,
        _to_wm_proprio,
        _to_wm_visual,
    )

    obs = {
        "visual": _to_wm_visual(image, wm_bundle.device),
        "proprio": _to_wm_proprio(proprio, int(wm_bundle.proprio_dim), wm_bundle.device),
    }
    action_t, factor = _build_wm_action_tensor(wm_bundle, actions)
    visual_latents: list[Any] = []
    proprio_latents: list[Any] = []
    structured_latents: list[dict[str, Any]] = []
    with torch.no_grad():
        encoded = _split_latent_for_mode(wm_bundle.model.encode(obs), mode=mode)
        latent = _latent_context_for_mode(encoded, mode=mode)
        for t in range(int(action_t.shape[0])):
            unroll_out = wm_bundle.model.unroll(latent, act_suffix=action_t[t : t + 1], debug=False)
            structured = _split_latent_for_mode(_next_latent_state_after_unroll(unroll_out), mode=mode)
            structured_latents.append(structured)
            visual_latents.append(structured["visual"])
            if str(mode) == "visual_proprio":
                proprio_latents.append(structured.get("proprio"))
            latent = _latent_context_for_mode(structured, mode=mode)

    frames = None
    if decode_frames:
        frames, failure = _decode_latent_trace_to_frames(
            wm_bundle,
            DecodeTrace(visual_latents=visual_latents, proprio_latents=proprio_latents),
        )
        if failure is not None:
            raise RuntimeError(failure)
    return Phase12LatentTrace(
        visual_latents=visual_latents,
        proprio_latents=proprio_latents,
        structured_latents=structured_latents,
        frames=frames,
        wm_factor=factor,
    )


def encode_real_latents_for_indices(
    wm_bundle: Any,
    *,
    frames: Sequence[Any],
    proprios: Sequence[Any],
    indices: Sequence[int],
    mode: str,
) -> list[dict[str, Any]]:
    import torch
    from segment_grpo_loop import _to_wm_proprio, _to_wm_visual

    out: list[dict[str, Any]] = []
    with torch.no_grad():
        for idx in indices:
            obs = {
                "visual": _to_wm_visual(frames[int(idx)], wm_bundle.device),
                "proprio": _to_wm_proprio(proprios[int(idx)], int(wm_bundle.proprio_dim), wm_bundle.device),
            }
            out.append(_split_latent_for_mode(wm_bundle.model.encode(obs), mode=mode))
    return out


def latent_l2_distance(
    pred: Mapping[str, Any],
    real: Mapping[str, Any],
    *,
    mode: str,
    proprio_alpha: float = 1.0,
) -> dict[str, float]:
    import torch

    visual_l2 = float(
        torch.linalg.vector_norm(
            pred["visual"].detach().float().reshape(-1) - real["visual"].detach().float().reshape(-1)
        ).cpu()
    )
    if str(mode) == "visual_only_ablation":
        return {"visual_l2": visual_l2, "combined_l2": visual_l2}
    proprio_l2 = float(
        torch.linalg.vector_norm(
            pred["proprio"].detach().float().reshape(-1) - real["proprio"].detach().float().reshape(-1)
        ).cpu()
    )
    combined = float((visual_l2**2 + (float(proprio_alpha) * proprio_l2) ** 2) ** 0.5)
    return {"visual_l2": visual_l2, "proprio_l2": proprio_l2, "combined_l2": combined}


def compute_raw_bounded_l2_metrics(
    *,
    raw_pred_latents: Sequence[Mapping[str, Any]],
    bounded_pred_latents: Sequence[Mapping[str, Any]],
    real_latents: Sequence[Mapping[str, Any]],
    mode: str,
    proprio_alpha: float = 1.0,
) -> dict[str, Any]:
    count = min(len(raw_pred_latents), len(bounded_pred_latents), len(real_latents))
    raw_metrics = [
        latent_l2_distance(raw_pred_latents[i], real_latents[i], mode=mode, proprio_alpha=proprio_alpha)
        for i in range(count)
    ]
    bounded_metrics = [
        latent_l2_distance(bounded_pred_latents[i], real_latents[i], mode=mode, proprio_alpha=proprio_alpha)
        for i in range(count)
    ]
    winners: list[str] = []
    for raw_m, bounded_m in zip(raw_metrics, bounded_metrics, strict=True):
        raw_v = float(raw_m["combined_l2"])
        bounded_v = float(bounded_m["combined_l2"])
        if abs(raw_v - bounded_v) <= 1e-8:
            winners.append("tie")
        elif raw_v < bounded_v:
            winners.append("raw")
        else:
            winners.append("bounded")
    return {
        "raw": {
            "visual_l2": [float(x["visual_l2"]) for x in raw_metrics],
            "proprio_l2": [float(x.get("proprio_l2", 0.0)) for x in raw_metrics],
            "combined_l2": [float(x["combined_l2"]) for x in raw_metrics],
        },
        "bounded": {
            "visual_l2": [float(x["visual_l2"]) for x in bounded_metrics],
            "proprio_l2": [float(x.get("proprio_l2", 0.0)) for x in bounded_metrics],
            "combined_l2": [float(x["combined_l2"]) for x in bounded_metrics],
        },
        "winner_by_column": winners,
        "winner_counts": {
            "raw": int(sum(1 for x in winners if x == "raw")),
            "bounded": int(sum(1 for x in winners if x == "bounded")),
            "tie": int(sum(1 for x in winners if x == "tie")),
        },
    }


def decode_phase12_prediction_frames(
    wm_bundle: Any,
    *,
    image: Any,
    proprio: Any,
    actions: Any,
    mode: str,
) -> list[Any]:
    trace = unroll_phase12_latent_trace(
        wm_bundle,
        image=image,
        proprio=proprio,
        actions=actions,
        mode=mode,
        decode_frames=True,
    )
    try:
        actions_np = np.asarray(actions, dtype=np.float32)
        try:
            if os.environ.get("AGENT_DEBUG_PHASE12_WM_ACTIONS", "").strip().lower() in {"1", "true", "yes"}:
                payload = {
                    "sessionId": "588128",
                    "id": f"phase12_decode_{os.getpid()}_{int(time.time() * 1000)}",
                    "timestamp": int(time.time() * 1000),
                    "runId": os.environ.get("AGENT_DEBUG_RUN_ID", "phase12-decode-compare"),
                    "hypothesisId": "H2",
                    "location": "src/smolvla_grpo/phase12_decode_compare.py:decode_phase12_prediction_frames",
                    "message": "phase12 decode final latent trace built",
                    "data": {
                        "actions_shape": list(actions_np.shape),
                        "factor": int(trace.wm_factor),
                        "visual_latent_count": int(len(trace.visual_latents)),
                        "proprio_latent_count": int(len(trace.proprio_latents)),
                    },
                }
                with open("/vol/bitbucket/aa6622/.logs/debug-588128.log", "a", encoding="utf-8") as f:
                    f.write(json.dumps(payload, sort_keys=True) + "\n")
        except Exception:
            pass
    except Exception:
        pass
    return list(trace.frames or [])


def build_summary(
    *,
    output_dir: Path,
    args: Any,
    episodes: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "comparison_type": "raw_vs_bounded_same_actions",
        "output_dir": str(Path(output_dir)),
        "task": str(args.task),
        "num_episodes": int(args.num_episodes),
        "chunk_len": int(args.chunk_len),
        "max_steps": int(args.max_steps),
        "train_seed_base": int(args.train_seed_base),
        "env_dispatched_source": "raw_postprocessed",
        "raw_wm_action_source": "raw_postprocessed",
        "bounded_wm_action_source": "clipped",
        "episodes": episodes,
    }
