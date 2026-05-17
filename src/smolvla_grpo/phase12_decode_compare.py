"""Utilities for comparing Phase12 JEPA decodes from shared actions."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ActionVariants:
    env_actions: np.ndarray
    raw_wm_actions: np.ndarray
    bounded_wm_actions: np.ndarray
    metadata: dict[str, Any]


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


def align_decode_rows(
    *,
    real_frames: list[np.ndarray],
    raw_pred_frames: list[np.ndarray],
    bounded_pred_frames: list[np.ndarray],
    env_steps_per_wm_step: int,
    carried_steps: int,
) -> tuple[list[list[np.ndarray]], list[int]]:
    factor = max(1, int(env_steps_per_wm_step))
    carry = max(0, int(carried_steps))
    count = min(len(raw_pred_frames), len(bounded_pred_frames))
    rows: list[list[np.ndarray]] = [[], [], []]
    indices: list[int] = []
    for k in range(count):
        ridx = min((k + 1) * factor, carry)
        if ridx >= len(real_frames):
            break
        real_rgb = _to_rgb_uint8(real_frames[ridx])
        rows[0].append(real_rgb)
        rows[1].append(_resize_like(raw_pred_frames[k], real_rgb))
        rows[2].append(_resize_like(bounded_pred_frames[k], real_rgb))
        indices.append(int(ridx))
    if not rows[0]:
        raise ValueError("no aligned decode frames")
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


def decode_phase12_prediction_frames(
    wm_bundle: Any,
    *,
    image: Any,
    proprio: Any,
    actions: Any,
    mode: str,
) -> list[Any]:
    import torch
    from segment_grpo_loop import (
        DecodeTrace,
        _decode_latent_trace_to_frames,
        _infer_env_action_dim,
        _infer_model_action_dim,
        _next_latent_state_after_unroll,
        _normalize_env_actions_for_wm,
        _pack_env_actions_for_wm,
        _to_wm_proprio,
        _to_wm_visual,
        _wm_action_block_factor,
    )

    obs = {
        "visual": _to_wm_visual(image, wm_bundle.device),
        "proprio": _to_wm_proprio(proprio, int(wm_bundle.proprio_dim), wm_bundle.device),
    }
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
    action_t = torch.from_numpy(packed).to(wm_bundle.device).float().unsqueeze(1)
    with torch.no_grad():
        latent = wm_bundle.model.encode(obs)
        if isinstance(latent, dict) and mode == "visual_proprio":
            try:
                from tensordict import TensorDict

                latent = TensorDict(
                    {"visual": latent["visual"], "proprio": latent["proprio"]},
                    batch_size=[],
                )
            except Exception:
                pass
        elif isinstance(latent, dict) and mode == "visual_only_ablation":
            latent = latent["visual"]
        visual_latents: list[Any] = []
        proprio_latents: list[Any] = []
        decode_step_shapes: list[dict[str, Any]] = []
        for t in range(int(action_t.shape[0])):
            unroll_out = wm_bundle.model.unroll(latent, act_suffix=action_t[t : t + 1], debug=False)
            latent = _next_latent_state_after_unroll(unroll_out)
            visual = _structured_field(latent, "visual")
            proprio_latent = _structured_field(latent, "proprio")
            if visual is not None:
                visual_latents.append(visual)
                if mode == "visual_proprio":
                    proprio_latents.append(proprio_latent)
            else:
                visual_latents.append(latent)
            if isinstance(latent, dict) and mode == "visual_proprio":
                try:
                    from tensordict import TensorDict

                    latent = TensorDict(
                        {"visual": latent["visual"], "proprio": latent["proprio"]},
                        batch_size=[],
                    )
                except Exception:
                    pass
            elif isinstance(latent, dict) and mode == "visual_only_ablation":
                latent = latent["visual"]
            if len(decode_step_shapes) < 8:
                decode_step_shapes.append(
                    {
                        "step": int(t),
                        "act_suffix_shape": [int(x) for x in action_t[t : t + 1].shape],
                        "visual_shape": [int(x) for x in visual.shape] if hasattr(visual, "shape") else None,
                        "proprio_shape": [int(x) for x in proprio_latent.shape] if hasattr(proprio_latent, "shape") else None,
                    }
                )
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
                        "normalized_shape": list(normalized.shape),
                        "packed_shape": list(packed.shape),
                        "action_t_shape": [int(x) for x in action_t.shape],
                        "env_dim": int(env_dim),
                        "wm_dim": int(wm_dim),
                        "factor": int(factor),
                        "visual_latent_count": int(len(visual_latents)),
                        "proprio_latent_count": int(len(proprio_latents)),
                        "decode_step_shapes": decode_step_shapes,
                    },
                }
                with open("/vol/bitbucket/aa6622/.logs/debug-588128.log", "a", encoding="utf-8") as f:
                    f.write(json.dumps(payload, sort_keys=True) + "\n")
        except Exception:
            pass
    frames, failure = _decode_latent_trace_to_frames(
        wm_bundle,
        DecodeTrace(visual_latents=visual_latents, proprio_latents=proprio_latents),
    )
    if failure is not None:
        raise RuntimeError(failure)
    return frames


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
