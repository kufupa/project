"""Structured JEPA-WM reward backend for Phase12 chunk scoring."""

from __future__ import annotations

from typing import Any, Mapping

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
    latent: Any = _unroll_context(start_latent, mode=mode)
    with torch.no_grad():
        for t in range(int(action_t.shape[0])):
            latent = wm_bundle.model.unroll(latent, act_suffix=action_t[t : t + 1], debug=False)
            latent = _unroll_context(split_structured_latent(_next_latent_state_after_unroll(latent), mode=mode), mode=mode)
    return split_structured_latent(latent, mode=mode)


def _unroll_context(latent: Mapping[str, torch.Tensor], *, mode: str) -> Any:
    if str(mode) == "visual_only_ablation":
        return latent["visual"]
    try:
        from tensordict import TensorDict

        return TensorDict(
            {"visual": latent["visual"], "proprio": latent["proprio"]},
            batch_size=[],
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
