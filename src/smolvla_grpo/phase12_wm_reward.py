"""Structured JEPA-WM reward backend for Phase12 chunk scoring."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from segment_grpo_loop import (
    _infer_env_action_dim,
    _infer_model_action_dim,
    _next_latent_state_after_unroll,
    _normalize_env_actions_for_wm,
    _pack_env_actions_for_wm,
    _to_wm_proprio,
    _to_wm_visual,
    _wm_action_block_factor,
)
from smolvla_grpo.phase12_objective import Phase12Score, score_progress, split_structured_latent

_AGENT_DEBUG_LOG_PATH = Path("/vol/bitbucket/aa6622/.logs/debug-588128.log")
_AGENT_DEBUG_LOG_COUNT = 0


def _agent_debug_log(*, hypothesis_id: str, location: str, message: str, data: dict[str, Any]) -> None:
    global _AGENT_DEBUG_LOG_COUNT
    if os.environ.get("AGENT_DEBUG_PHASE12_WM_ACTIONS", "").strip().lower() not in {"1", "true", "yes"}:
        return
    if _AGENT_DEBUG_LOG_COUNT >= 40:
        return
    _AGENT_DEBUG_LOG_COUNT += 1
    try:
        payload = {
            "sessionId": "588128",
            "id": f"phase12_wm_reward_{os.getpid()}_{_AGENT_DEBUG_LOG_COUNT}",
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


def _image_debug(image: Any) -> dict[str, Any]:
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]
    arr = np.ascontiguousarray(arr)
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "mean": float(np.mean(arr)) if arr.size else 0.0,
        "sha16": hashlib.sha256(arr.tobytes()).hexdigest()[:16],
        "hflip_sha16": hashlib.sha256(np.ascontiguousarray(np.flip(arr, 1)).tobytes()).hexdigest()[:16]
        if arr.ndim == 3
        else None,
        "vflip_sha16": hashlib.sha256(np.ascontiguousarray(np.flip(arr, 0)).tobytes()).hexdigest()[:16]
        if arr.ndim == 3
        else None,
    }


def _array_stats(name: str, value: Any) -> dict[str, Any]:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    return {
        f"{name}_shape": list(np.asarray(value).shape),
        f"{name}_min": float(np.min(arr)) if arr.size else 0.0,
        f"{name}_max": float(np.max(arr)) if arr.size else 0.0,
        f"{name}_mean": float(np.mean(arr)) if arr.size else 0.0,
        f"{name}_std": float(np.std(arr)) if arr.size else 0.0,
    }


def _encode_structured(
    wm_bundle: Any,
    image: np.ndarray,
    proprio: np.ndarray,
    *,
    mode: str,
) -> dict[str, torch.Tensor]:
    obs = {
        "visual": _to_wm_visual(image, wm_bundle.device),
        "proprio": _to_wm_proprio(proprio, int(wm_bundle.proprio_dim), wm_bundle.device),
    }
    # region agent log
    _agent_debug_log(
        hypothesis_id="H3,H4",
        location="src/smolvla_grpo/phase12_wm_reward.py:_encode_structured",
        message="Phase12 image/proprio entering JEPA-WM encode",
        data={
            "mode": str(mode),
            "image": _image_debug(image),
            **_array_stats("proprio", proprio),
            "wm_visual_tensor_shape": [int(x) for x in obs["visual"].shape],
            "wm_visual_min": float(obs["visual"].min().detach().cpu().item()),
            "wm_visual_max": float(obs["visual"].max().detach().cpu().item()),
            "wm_proprio_tensor_shape": [int(x) for x in obs["proprio"].shape],
        },
    )
    # endregion
    with torch.no_grad():
        encoded = wm_bundle.model.encode(obs)
    return split_structured_latent(encoded, mode=mode)


def _final_structured_after_unroll(
    wm_bundle: Any,
    start_latent: Mapping[str, torch.Tensor],
    actions: np.ndarray,
    *,
    mode: str,
) -> dict[str, torch.Tensor]:
    env_dim = _infer_env_action_dim(wm_bundle, actions)
    model_action_dim = _infer_model_action_dim(wm_bundle.model)
    wm_dim = int(model_action_dim) if model_action_dim else int(wm_bundle.planner_action_dim)
    factor = _wm_action_block_factor(env_dim, wm_dim)
    normalized = _normalize_env_actions_for_wm(
        wm_bundle.preprocessor,
        actions[:, :env_dim],
        env_dim,
        wm_bundle.device,
    )
    packed = _pack_env_actions_for_wm(normalized, factor, env_dim, wm_dim)
    action_t = torch.from_numpy(packed).to(wm_bundle.device).float().unsqueeze(1)
    # region agent log
    _agent_debug_log(
        hypothesis_id="H1,H2",
        location="src/smolvla_grpo/phase12_wm_reward.py:_final_structured_after_unroll",
        message="Phase12 actions entering JEPA-WM unroll",
        data={
            "actions_shape": list(actions.shape),
            "env_dim": int(env_dim),
            "wm_dim": int(wm_dim),
            "factor": int(factor),
            "normalized_shape": list(normalized.shape),
            "packed_shape": list(packed.shape),
            "action_t_shape": [int(x) for x in action_t.shape],
            **_array_stats("raw_action", actions),
            **_array_stats("normalized_action", normalized),
            "first_raw_rows": actions[: min(5, actions.shape[0]), : min(4, actions.shape[1])].tolist()
            if actions.size
            else [],
            "first_packed_row": packed[0, : min(20, packed.shape[1])].tolist() if packed.size else [],
        },
    )
    # endregion
    latent: Any = _unroll_context(start_latent, mode=mode)
    with torch.no_grad():
        for t in range(int(action_t.shape[0])):
            latent = wm_bundle.model.unroll(latent, act_suffix=action_t[t : t + 1], debug=False)
            latent = _unroll_context(split_structured_latent(_next_latent_state_after_unroll(latent), mode=mode), mode=mode)
    return split_structured_latent(latent, mode=mode)


def _repeat_latent_batch(latent: Mapping[str, torch.Tensor], batch_size: int, *, mode: str) -> dict[str, torch.Tensor]:
    repeated: dict[str, torch.Tensor] = {}
    for key, value in latent.items():
        tensor = torch.as_tensor(value)
        if tensor.dim() < 2:
            raise ValueError(f"latent field {key!r} must have at least 2 dims, got {tuple(tensor.shape)}")
        if int(tensor.shape[0]) != 1:
            raise ValueError(f"latent field {key!r} must have batch dim 1 before repeat, got {tuple(tensor.shape)}")
        if int(tensor.shape[1]) != 1:
            raise ValueError(f"latent field {key!r} must have tau dim 1 before repeat, got {tuple(tensor.shape)}")
        repeated[key] = tensor.expand(int(batch_size), *tensor.shape[1:]).clone()
    if str(mode) == "visual_only_ablation":
        return {"visual": repeated["visual"]}
    return repeated


def _pack_actions_for_wm_batch(
    wm_bundle: Any,
    action_chunks: Sequence[np.ndarray],
) -> torch.Tensor:
    packed_rows: list[np.ndarray] = []
    expected_shape: tuple[int, int] | None = None
    for actions in action_chunks:
        arr = np.asarray(actions, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"chunk_actions must be 2D, got {arr.shape}")
        env_dim = _infer_env_action_dim(wm_bundle, arr)
        model_action_dim = _infer_model_action_dim(wm_bundle.model)
        wm_dim = int(model_action_dim) if model_action_dim else int(wm_bundle.planner_action_dim)
        factor = _wm_action_block_factor(env_dim, wm_dim)
        normalized = _normalize_env_actions_for_wm(
            wm_bundle.preprocessor,
            arr[:, :env_dim],
            env_dim,
            wm_bundle.device,
        )
        packed = _pack_env_actions_for_wm(normalized, factor, env_dim, wm_dim)
        shape = (int(packed.shape[0]), int(packed.shape[1]))
        if expected_shape is None:
            expected_shape = shape
        elif shape != expected_shape:
            raise ValueError(f"all packed action chunks must share shape {expected_shape}, got {shape}")
        packed_rows.append(packed)
    if not packed_rows:
        raise ValueError("action_chunks must be non-empty")
    stacked = np.stack(packed_rows, axis=1)  # T_wm, B, wm_dim
    return torch.from_numpy(stacked).to(wm_bundle.device).float()


def _final_structured_after_batched_unroll(
    wm_bundle: Any,
    start_latent: Mapping[str, torch.Tensor],
    action_chunks: Sequence[np.ndarray],
    *,
    mode: str,
) -> dict[str, torch.Tensor]:
    action_t = _pack_actions_for_wm_batch(wm_bundle, action_chunks)
    batch_size = int(action_t.shape[1])
    repeated = _repeat_latent_batch(start_latent, batch_size, mode=mode)
    latent: Any = _unroll_context(repeated, mode=mode)
    for key, value in repeated.items():
        if tuple(value.shape[:2]) != (batch_size, 1):
            raise RuntimeError(f"batched latent {key} has wrong pre-unroll shape {tuple(value.shape)}")
    with torch.no_grad():
        out = wm_bundle.model.unroll(latent, act_suffix=action_t, debug=False)
    final = split_structured_latent(_next_latent_state_after_unroll(out), mode=mode)
    out_final: dict[str, torch.Tensor] = {}
    for key, value in final.items():
        tensor = torch.as_tensor(value).detach().float()
        if tensor.dim() < 2:
            raise RuntimeError(f"batched final latent {key} has too few dims: {tuple(tensor.shape)}")
        # JEPA returns time-major [T, B, ...]; after _next helper this is [1, B, ...].
        if int(tensor.shape[0]) == 1 and int(tensor.shape[1]) == batch_size:
            tensor = tensor.permute(1, 0, *range(2, tensor.dim())).contiguous()
        if int(tensor.shape[0]) != batch_size or int(tensor.shape[1]) != 1:
            raise RuntimeError(f"batched final latent {key} must be [B,1,...], got {tuple(tensor.shape)}")
        out_final[key] = tensor
    return out_final


def _unroll_context(latent: Mapping[str, torch.Tensor], *, mode: str) -> Any:
    if str(mode) == "visual_only_ablation":
        return latent["visual"]
    try:
        from tensordict import TensorDict

        first = next(iter(latent.values()))
        batch_size = list(torch.as_tensor(first).shape[:2]) if torch.as_tensor(first).dim() >= 2 else []
        return TensorDict(
            {"visual": latent["visual"], "proprio": latent["proprio"]},
            batch_size=batch_size,
        )
    except Exception:
        return {"visual": latent["visual"], "proprio": latent["proprio"]}


def score_phase12_chunk_with_wm(
    *,
    wm_bundle: Any,
    image: np.ndarray,
    proprio: np.ndarray,
    chunk_actions: np.ndarray,
    goal: Mapping[str, torch.Tensor],
    candidate_index: int,
    proprio_alpha: float,
    mode: str,
    debug_npz_path: str | None = None,
) -> Phase12Score:
    actions = np.asarray(chunk_actions, dtype=np.float32)
    if actions.ndim != 2:
        raise ValueError(f"chunk_actions must be 2D, got {actions.shape}")
    start = _encode_structured(wm_bundle, image, proprio, mode=mode)
    final = _final_structured_after_unroll(wm_bundle, start, actions, mode=mode)
    return score_progress(
        candidate_index=int(candidate_index),
        start=start,
        final=final,
        goal=goal,
        proprio_alpha=float(proprio_alpha),
        mode=mode,
        debug_npz_path=debug_npz_path,
    )


def score_phase12_chunks_with_wm(
    *,
    wm_bundle: Any,
    image: np.ndarray,
    proprio: np.ndarray,
    chunk_actions: Sequence[np.ndarray],
    candidate_indices: Sequence[int],
    goal: Mapping[str, torch.Tensor],
    proprio_alpha: float,
    mode: str,
    batch_size: int,
    telemetry: dict[str, Any] | None = None,
) -> list[Phase12Score]:
    chunks = [np.asarray(chunk, dtype=np.float32).copy() for chunk in chunk_actions]
    indices = [int(idx) for idx in candidate_indices]
    if len(chunks) != len(indices):
        raise ValueError("chunk_actions and candidate_indices must have same length")
    if not chunks:
        return []
    bs = max(int(batch_size), 1)
    tel = telemetry if telemetry is not None else {}
    prev_seconds = float(tel.get("wm_score_seconds", 0.0))
    prev_candidate_count = int(tel.get("wm_score_candidate_count", 0))
    prev_batch_count = int(tel.get("wm_score_batch_count", 0))
    prev_batch_sizes = list(tel.get("wm_score_batch_sizes", []))
    prev_peak = int(tel.get("wm_score_cuda_peak_allocated_bytes", 0))
    t0 = time.perf_counter()
    tel["wm_score_mode"] = "batched"
    tel["wm_score_candidate_count"] = prev_candidate_count + int(len(chunks))
    tel["wm_score_batch_size"] = int(bs)
    local_batch_sizes: list[int] = []
    start = _encode_structured(wm_bundle, image, proprio, mode=mode)
    scores: list[Phase12Score] = []
    for batch_start in range(0, len(chunks), bs):
        batch_chunks = chunks[batch_start : batch_start + bs]
        batch_indices = indices[batch_start : batch_start + bs]
        local_batch_sizes.append(int(len(batch_chunks)))
        final_batch = _final_structured_after_batched_unroll(
            wm_bundle,
            start,
            batch_chunks,
            mode=mode,
        )
        for row, candidate_index in enumerate(batch_indices):
            final = {key: value[row : row + 1] for key, value in final_batch.items()}
            scores.append(
                score_progress(
                    candidate_index=int(candidate_index),
                    start=start,
                    final=final,
                    goal=goal,
                    proprio_alpha=float(proprio_alpha),
                    mode=mode,
                )
            )
    tel["wm_score_batch_sizes"] = [*prev_batch_sizes, *local_batch_sizes]
    tel["wm_score_batch_count"] = prev_batch_count + int(len(local_batch_sizes))
    tel["wm_score_seconds"] = prev_seconds + float(time.perf_counter() - t0)
    if torch.cuda.is_available():
        tel["wm_score_cuda_allocated_after_bytes"] = int(torch.cuda.memory_allocated())
        tel["wm_score_cuda_peak_allocated_bytes"] = max(prev_peak, int(torch.cuda.max_memory_allocated()))
    else:
        tel["wm_score_cuda_allocated_after_bytes"] = 0
        tel["wm_score_cuda_peak_allocated_bytes"] = prev_peak
    return scores
