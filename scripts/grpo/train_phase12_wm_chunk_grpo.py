#!/usr/bin/env python3
"""Phase12: WM-scored receding-horizon chunk GRPO for SmolVLA."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from smolvla_grpo.checkpointing import save_grpo_checkpoint
from smolvla_grpo.phase12_logging import (
    assert_smoke_manifest_contract,
    utc_now_iso,
    write_jsonl_row,
    write_manifest,
)
from smolvla_grpo.phase12_diagnostics import build_decode_artifacts, write_phase12_episode_video
from smolvla_grpo.phase12_pixels import policy_rgb_from_obs, wm_rgb_from_policy_rgb_corner2


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("rollout_validation", "wm_grpo_train"), default="wm_grpo_train")
    p.add_argument("--checkpoint", type=str, default="jadechoghari/smolvla_metaworld")
    p.add_argument("--jepa-ckpt", type=str, default="jepa_wm_metaworld.pth.tar")
    p.add_argument("--jepa-repo", type=str, default="")
    p.add_argument("--output-dir", type=Path, default=Path("artifacts/phase12_wm_chunk_grpo/dry_run"))
    p.add_argument("--task", type=str, default="push-v3")
    p.add_argument("--env-backend", choices=("official_lerobot_guarded", "custom_oracle_aligned"), default="official_lerobot_guarded")
    p.add_argument("--action-profile", choices=("official_jepa_mirror", "bounded_executed"), default="official_jepa_mirror")
    p.add_argument("--chunk-len", type=int, default=25)
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--num-episodes", type=int, default=1)
    p.add_argument("--num-updates", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=120)
    p.add_argument("--goal-latent-mode", choices=("visual_proprio", "visual_only_ablation"), default="visual_proprio")
    p.add_argument("--proprio-alpha", type=float, default=0.1)
    p.add_argument("--reward-key", choices=("wm_latent_progress", "latent_return"), default="wm_latent_progress")
    p.add_argument("--ratio-mode", choices=("chunk", "per_step_ablation"), default="chunk")
    p.add_argument("--logprob-backward-mode", choices=("stack", "microbatch"), default="stack")
    p.add_argument("--old-policy-inference-mode", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--action-transform", choices=("no_tanh", "tanh_norm_ablation"), default="no_tanh")
    p.add_argument("--reset-mismatch", choices=("fail", "skip", "warn"), default="fail")
    p.add_argument("--decode-candidates", choices=("selected", "all", "none"), default="selected")
    p.add_argument("--save-wm-decodes", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--strict-wm-scoring", action="store_true")
    p.add_argument("--strict-decode", action="store_true")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--update-epochs", type=int, default=1)
    p.add_argument("--init-log-std", type=float, default=-2.0)
    p.add_argument("--euler-step-noise-std", type=float, default=0.2)
    p.add_argument("--save-every", type=int, default=5)
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--train-seed-base", type=int, default=2000)
    p.add_argument("--allow-wm-fallback", action="store_true", default=False)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def build_manifest(args: argparse.Namespace) -> dict:
    is_train = str(args.mode) == "wm_grpo_train"
    return {
        "created_at": utc_now_iso(),
        "mode": str(args.mode),
        "method_label": (
            "wm_scored_receding_horizon_chunk_grpo"
            if is_train
            else "wm_scored_receding_horizon_rollout_validation"
        ),
        "uses_cem": False,
        "optimizer_updates": int(args.num_updates) if is_train else 0,
        "checkpoint": str(args.checkpoint),
        "jepa_ckpt": str(args.jepa_ckpt),
        "jepa_repo": str(args.jepa_repo),
        "task": str(args.task),
        "env_backend": str(args.env_backend),
        "action_profile": str(args.action_profile),
        "chunk_len": int(args.chunk_len),
        "group_size": int(args.group_size),
        "num_episodes": int(args.num_episodes),
        "num_updates": int(args.num_updates),
        "max_steps": int(args.max_steps),
        "objective_type": "L2",
        "goal_latent_mode": str(args.goal_latent_mode),
        "proprio_alpha": float(args.proprio_alpha),
        "reward_key": str(args.reward_key),
        "ratio_mode": str(args.ratio_mode),
        "logprob_backward_mode": str(args.logprob_backward_mode),
        "old_policy_inference_mode": bool(args.old_policy_inference_mode),
        "action_transform": str(args.action_transform),
        "reset_mismatch": str(args.reset_mismatch),
        "decode_candidates": str(args.decode_candidates),
        "save_wm_decodes": bool(args.save_wm_decodes),
        "train_seed_base": int(args.train_seed_base),
        "lr": float(args.lr),
        "clip_eps": float(args.clip_eps),
        "save_every": int(args.save_every),
        "phase12_policy_frame_contract": "lerobot_corner2_vhflip",
        "phase12_wm_frame_contract": "jepa_corner2_vflip",
        "phase12_goal_frame_contract": "jepa_corner2_vflip",
        "phase12_decode_real_frame_source": "wm_frames",
    }


def _validate_real_mode(args: argparse.Namespace) -> str | None:
    if not str(args.jepa_repo).strip():
        return "Missing --jepa-repo for real Phase12 WM scoring."
    if not str(args.jepa_ckpt).strip():
        return "Missing --jepa-ckpt for real Phase12 WM scoring."
    if int(args.num_episodes) < 1:
        return "--num-episodes must be >= 1."
    if int(args.group_size) < 1:
        return "--group-size must be >= 1."
    if int(args.chunk_len) < 1:
        return "--chunk-len must be >= 1."
    if args.mode == "wm_grpo_train" and int(args.num_episodes) != int(args.num_updates):
        return "wm_grpo_train requires --num-episodes == --num-updates."
    if args.mode == "wm_grpo_train" and bool(args.allow_wm_fallback):
        return "wm_grpo_train refuses --allow-wm-fallback."
    return None


def _episode_smoke_manifest(episode: Any) -> dict[str, Any]:
    meta = dict(getattr(episode, "metadata", {}) or {})
    out = {
        "rollout_validation_video": str(meta.get("rollout_validation_video", meta.get("selected_action_rollout_video", ""))),
        "selected_action_rollout_video": str(meta.get("selected_action_rollout_video", "")),
        "oracle_baseline_video": str(meta.get("oracle_baseline_video", "")),
        "oracle_baseline_video_status": str(meta.get("oracle_baseline_video_status", "")),
        "wm_decode_status": str(meta.get("wm_decode_status", "")),
        "wm_decode_selected_strip_path": str(meta.get("wm_decode_selected_strip_path", "")),
        "wm_real_vs_pred_selected_strip_path": str(meta.get("wm_real_vs_pred_selected_strip_path", "")),
        "success_any": bool(getattr(episode, "success_any", meta.get("success_any", False))),
        "success_last": bool(getattr(episode, "success_last", meta.get("success_last", False))),
    }
    assert_smoke_manifest_contract(out)
    return out


def run_phase12_episode(**kwargs: Any) -> Any:
    """Run one real Phase12 smoke episode using existing WM chunk rollout plumbing."""

    import torch
    from segment_grpo_loop import load_smolvla_bundle, load_wm_bundle, rollout_with_chunks

    args = kwargs["args"]
    output_dir = Path(kwargs["output_dir"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wm_bundle = load_wm_bundle(args.jepa_repo, args.jepa_ckpt, device, required=True)
    smolvla_bundle = load_smolvla_bundle(args.checkpoint, device=device, n_action_steps=int(args.chunk_len))
    episode_log, _adapter = rollout_with_chunks(
        smolvla_bundle,
        wm_bundle,
        task=args.task,
        episode_index=int(kwargs.get("episode_index", 0)),
        chunk_len=int(args.chunk_len),
        num_candidates=int(args.group_size),
        max_steps=int(args.max_steps),
        carry_mode="sim",
        comparison_root=output_dir / "rollouts",
        seed=int(kwargs.get("seed", 0)),
        train_steps=0,
        dry_run=False,
        strict_wm_scoring=bool(args.strict_wm_scoring),
        strict_decode=bool(args.strict_decode),
        wm_rollout_mode="iterative",
        wm_scoring_latent="concat" if args.goal_latent_mode == "visual_proprio" else "visual",
        smolvla_n_action_steps=int(args.chunk_len),
        comparison_strip_overlay=True,
    )
    # Adapt Phase8 episode log to the trainer contract without pretending it is
    # already optimizing SmolVLA; this is the smoke rollout/artifact bridge.
    scores = [
        float(getattr(candidate, "score", 0.0))
        for segment in getattr(episode_log, "segments", [])
        for candidate in getattr(segment, "candidates", [])
    ]
    progress = float(max(scores)) if scores else 0.0
    meta = dict(getattr(episode_log, "metadata", {}) or {})
    meta.setdefault("wm_latent_progress", progress)
    meta.setdefault("latent_return", progress)
    class _EpisodeAdapter:
        total_env_reward = float(sum(float(getattr(seg, "env_reward_sum", 0.0)) for seg in getattr(episode_log, "segments", [])))
        success_any = bool(any(bool(getattr(seg, "success_any", False)) for seg in getattr(episode_log, "segments", [])))
        success_last = bool(bool(getattr(episode_log, "segments", [])) and bool(getattr(episode_log.segments[-1], "success_last", False)))
        metadata = meta

    return _EpisodeAdapter()


def load_phase12_train_resources(args: argparse.Namespace) -> tuple[Any, Any, int]:
    import torch
    from segment_grpo_loop import load_smolvla_bundle, load_wm_bundle

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wm_bundle = load_wm_bundle(args.jepa_repo, args.jepa_ckpt, device, required=True)
    smolvla_bundle = load_smolvla_bundle(args.checkpoint, device=device, n_action_steps=int(args.chunk_len))
    return smolvla_bundle, wm_bundle, 4


def build_train_wrapper(args: argparse.Namespace, bundle: Any, action_dim: int) -> tuple[Any, list[Any]]:
    from smolvla_grpo.policy_wrapper import MetaWorldSmolVLAGRPOPolicy, freeze_all_but_grpo_trainables
    from smolvla_pipeline.evaluator import _resolve_camera_name, _resolve_flip_corner2, _resolve_task_text

    wrapper = MetaWorldSmolVLAGRPOPolicy(
        bundle,
        task=args.task,
        task_text=_resolve_task_text(args.task),
        camera_name=_resolve_camera_name(),
        flip_corner2=_resolve_flip_corner2(),
        action_dim=int(action_dim),
        action_transform=args.action_transform,
    )
    wrapper.assert_grpo_api()
    wrapper.set_log_std(args.init_log_std)
    wrapper.set_euler_step_noise_std(args.euler_step_noise_std)
    trainable = freeze_all_but_grpo_trainables(bundle.policy)
    if not trainable:
        trainable = [p for p in bundle.policy.parameters() if getattr(p, "requires_grad", False)]
    if not trainable:
        raise RuntimeError("No trainable GRPO parameters after freeze_all_but_grpo_trainables.")
    return wrapper, trainable


def build_old_wrapper(args: argparse.Namespace, bundle: Any, old_policy: Any, action_dim: int) -> Any:
    from smolvla_grpo.policy_wrapper import MetaWorldSmolVLAGRPOPolicy
    from smolvla_pipeline.evaluator import _resolve_camera_name, _resolve_flip_corner2, _resolve_task_text

    old_bundle = SimpleNamespace(
        policy=old_policy,
        preprocessor=bundle.preprocessor,
        postprocessor=bundle.postprocessor,
        device=bundle.device,
        obs_image_key=getattr(bundle, "obs_image_key", "observation.image"),
        obs_state_key=getattr(bundle, "obs_state_key", "observation.state"),
        obs_env_state_key=getattr(bundle, "obs_env_state_key", "observation.environment_state"),
    )
    return MetaWorldSmolVLAGRPOPolicy(
        old_bundle,
        task=args.task,
        task_text=_resolve_task_text(args.task),
        camera_name=_resolve_camera_name(),
        flip_corner2=_resolve_flip_corner2(),
        action_dim=int(action_dim),
        action_transform=args.action_transform,
        policy_module=old_policy,
    )


def _sample_old_action_chunk(
    old_wrapper: Any,
    proc: Any,
    *,
    chunk_len: int,
    rng: Any,
    use_inference_mode: bool,
) -> Any:
    import torch

    if bool(use_inference_mode):
        with torch.inference_mode():
            return old_wrapper.sample_action_chunk_from_proc(proc, chunk_len=int(chunk_len), rng=rng)
    return old_wrapper.sample_action_chunk_from_proc(proc, chunk_len=int(chunk_len), rng=rng)


def phase12_episode_training_metadata(episode: Any, reward_key: str) -> dict[str, Any]:
    import numpy as np

    candidates = [candidate for segment in episode.segments for candidate in segment.candidates]
    scores = [score for segment in episode.segments for score in segment.scores]
    segment_candidate_rewards = [
        [float(_field(score, reward_key)) for score in segment.scores]
        for segment in episode.segments
    ]
    return {
        "segment_candidate_rewards": segment_candidate_rewards,
        "candidate_rewards": [float(_field(score, reward_key)) for score in scores],
        "old_logprob_sums": [float(candidate.old_logprob_sum) for candidate in candidates],
        "proc_root_snapshots": [candidate.proc_root_snapshot for candidate in candidates],
        "unsquashed_chunks": [candidate.unsquashed_chunk for candidate in candidates],
        "segment_candidate_counts": [len(segment.candidates) for segment in episode.segments],
        "wm_status_counts": _count_values(_field(score, "wm_status", "unknown") for score in scores),
        "action_clip_fraction": float(np.mean([candidate.action_metadata.get("clip_fraction", 0.0) for candidate in candidates])) if candidates else 0.0,
        "action_clip_any_fraction": float(np.mean([candidate.action_metadata.get("clip_any", False) for candidate in candidates])) if candidates else 0.0,
        "raw_action_max_abs": max(
            (float(candidate.action_metadata.get("raw_action_max_abs", 0.0)) for candidate in candidates),
            default=0.0,
        ),
        "clipped_action_max_abs": max(
            (float(candidate.action_metadata.get("clipped_action_max_abs", 0.0)) for candidate in candidates),
            default=0.0,
        ),
        "clip_delta_max_abs": max(
            (float(candidate.action_metadata.get("clip_delta_max_abs", 0.0)) for candidate in candidates),
            default=0.0,
        ),
    }


def _field(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


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


def _phase12_sample_to_candidate_dict(sample: Any, *, candidate_index: int) -> dict[str, Any]:
    import numpy as np

    raw_actions = getattr(sample, "raw_postprocessed_action_np", None)
    if raw_actions is None:
        raw_actions = getattr(sample, "exec_action_np")
    clipped_actions = getattr(sample, "exec_action_np", raw_actions)
    return {
        "candidate_index": int(candidate_index),
        "proc_root_snapshot": None,
        "unsquashed_chunk": sample.unsquashed_chunk.detach().cpu(),
        "old_logprob_steps": sample.log_prob_steps.detach().cpu().numpy(),
        "old_logprob_sum": float(sample.log_prob_sum.detach().cpu().item()),
        "exec_actions_raw_postprocessed": np.asarray(raw_actions, dtype=np.float32),
        "exec_actions_clipped": np.asarray(clipped_actions, dtype=np.float32),
        "action_metadata": {
            "sample_clip_fraction_mean": float(np.mean(sample.action_clip_fraction)),
            "sample_clip_any_fraction": float(np.mean(sample.action_clip_any)),
            "unique_action_rows": int(sample.unique_action_rows),
        },
    }


def _with_episode_metadata(episode: Any, metadata: dict[str, Any]) -> Any:
    try:
        return replace(episode, metadata=dict(metadata))
    except Exception:
        episode.metadata = dict(metadata)
        return episode


def _merge_phase12_decode_metadata(meta: dict[str, Any], decode_metadata: dict[str, Any]) -> None:
    meta.update(decode_metadata)
    if "decode_status" in decode_metadata:
        meta["wm_decode_status"] = decode_metadata["decode_status"]


def _build_phase12_selected_decode_artifacts(
    *,
    args: Any,
    episode: Any,
    episode_dir: Path,
    rollout_env: Any,
    score_inputs: dict[tuple[int, int], dict[str, Any]],
    wm_bundle: Any,
    action_dim: int,
    meta: dict[str, Any],
) -> None:
    if not bool(getattr(args, "save_wm_decodes", False)) or not getattr(episode, "segments", []):
        meta.setdefault("wm_decode_status", "disabled")
        return
    first_segment = episode.segments[0]
    key = (0, int(first_segment.selected_candidate_index))
    decode_input = score_inputs.get(key)
    if decode_input is None:
        if bool(getattr(args, "strict_decode", False)):
            raise RuntimeError("strict Phase12 decode requested but selected decode input was not recorded")
        meta.setdefault("wm_decode_status", "missing_input")
        return
    real_frames = list(getattr(rollout_env, "wm_frames", rollout_env.frames))
    decode_result = build_decode_artifacts(
        decode_fn=lambda: _decode_phase12_prediction_frames(
            wm_bundle,
            image=decode_input["image"],
            proprio=decode_input["proprio"],
            actions=decode_input["actions"],
            mode=args.goal_latent_mode,
        ),
        output_dir=episode_dir,
        real_frames=real_frames,
        strict_decode=bool(args.strict_decode),
        segment_index=0,
        selected_candidate_index=int(first_segment.selected_candidate_index),
        env_steps_per_wm_step=max(1, int(wm_bundle.planner_action_dim) // max(1, int(action_dim))),
        carried_steps=min(int(args.chunk_len), max(0, len(real_frames) - 1)),
    )
    _merge_phase12_decode_metadata(meta, decode_result.metadata)


def _count_values(values) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        key = str(value)
        out[key] = out.get(key, 0) + 1
    return out


def _phase12_agent_debug_log(*, run_id: str, hypothesis_id: str, location: str, message: str, data: dict[str, Any]) -> None:
    try:
        import os
        import time

        if os.environ.get("AGENT_DEBUG_PHASE12_WM_ACTIONS", "").strip().lower() not in {"1", "true", "yes"}:
            return
        payload = {
            "sessionId": "588128",
            "id": f"phase12_train_{os.getpid()}_{int(time.time() * 1000)}",
            "timestamp": int(time.time() * 1000),
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
        }
        with open("/vol/bitbucket/aa6622/.logs/debug-588128.log", "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, sort_keys=True) + "\n")
    except Exception:
        pass


def _phase12_frame_debug(frame: Any) -> dict[str, Any]:
    import hashlib
    import numpy as np

    arr = np.asarray(frame)
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "mean": float(arr.mean()) if arr.size else 0.0,
        "sha16": hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16],
    }


def _phase12_vector_debug(vec: Any, *, max_items: int = 16) -> dict[str, Any]:
    import numpy as np

    arr = np.asarray(vec, dtype=np.float64).reshape(-1)
    return {
        "shape": list(arr.shape),
        "values": [float(x) for x in arr[: int(max_items)]],
        "max_abs": float(np.max(np.abs(arr))) if arr.size else 0.0,
    }


def _save_phase12_reset_debug_artifacts(
    *,
    output_dir: Path,
    oracle_frame: Any,
    reset_frame: Any,
    oracle_raw_obs: Any,
    reset_raw_obs: Any,
    reset_metrics: dict[str, float],
    reset_seed: int,
) -> dict[str, Any]:
    import json
    import numpy as np
    import imageio.v2 as imageio

    debug_dir = Path(output_dir) / "debug_reset_frame_compare" / f"seed_{int(reset_seed)}_gate"
    debug_dir.mkdir(parents=True, exist_ok=True)
    oracle_arr = np.asarray(oracle_frame, dtype=np.uint8)
    reset_arr = np.asarray(reset_frame, dtype=np.uint8)
    diff = np.abs(oracle_arr.astype(np.int16) - reset_arr.astype(np.int16)).astype(np.uint8)
    diff_x50 = np.clip(diff.astype(np.int16) * 50, 0, 255).astype(np.uint8)

    oracle_path = debug_dir / "oracle_frame0_raw.png"
    reset_path = debug_dir / "reset_frame0_raw.png"
    diff_path = debug_dir / "oracle_vs_reset_absdiff_x50.png"
    report_path = debug_dir / "reset_parity_vectors.json"
    imageio.imwrite(oracle_path, oracle_arr)
    imageio.imwrite(reset_path, reset_arr)
    imageio.imwrite(diff_path, diff_x50)

    oracle_vec = np.asarray(oracle_raw_obs, dtype=np.float64).reshape(-1)
    reset_vec = np.asarray(reset_raw_obs, dtype=np.float64).reshape(-1)
    raw_diff = np.abs(oracle_vec - reset_vec) if oracle_vec.shape == reset_vec.shape else np.asarray([], dtype=np.float64)
    report = {
        "reset_seed": int(reset_seed),
        "reset_metrics": reset_metrics,
        "oracle_raw_obs": _phase12_vector_debug(oracle_vec, max_items=80),
        "reset_raw_obs": _phase12_vector_debug(reset_vec, max_items=80),
        "raw_obs_shape_match": list(oracle_vec.shape) == list(reset_vec.shape),
        "raw_obs_max_abs_diff": float(np.max(raw_diff)) if raw_diff.size else None,
        "raw_obs_mean_abs_diff": float(np.mean(raw_diff)) if raw_diff.size else None,
        "paths": {
            "oracle_frame0_raw": str(oracle_path),
            "reset_frame0_raw": str(reset_path),
            "oracle_vs_reset_absdiff_x50": str(diff_path),
            "vector_report": str(report_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _phase12_reset_gate_decision(
    *,
    reset_metrics: dict[str, float],
    reset_debug_report: dict[str, Any],
    raw_obs_max_threshold: float = 1e-9,
    proprio_max_threshold: float = 1e-5,
    image_mean_threshold: float = 1.0,
) -> dict[str, Any]:
    """Strict on simulator state, tolerant of sparse MuJoCo RGB render jitter."""

    raw_max = reset_debug_report.get("raw_obs_max_abs_diff")
    raw_mismatch = raw_max is None or float(raw_max) > float(raw_obs_max_threshold)
    proprio_mismatch = float(reset_metrics.get("proprio_max_abs_diff", 0.0)) > float(proprio_max_threshold)
    gross_image_mismatch = float(reset_metrics.get("image_mean_abs_diff", 0.0)) > float(image_mean_threshold)
    old_strict_image_gate_would_fail = (
        float(reset_metrics.get("image_mean_abs_diff", 0.0)) > 0.01
        or float(reset_metrics.get("image_max_abs_diff", 0.0)) > 2.0
    )
    return {
        "fail": bool(raw_mismatch or proprio_mismatch or gross_image_mismatch),
        "raw_mismatch": bool(raw_mismatch),
        "proprio_mismatch": bool(proprio_mismatch),
        "gross_image_mismatch": bool(gross_image_mismatch),
        "old_strict_image_gate_would_fail": bool(old_strict_image_gate_would_fail),
        "raw_obs_max_threshold": float(raw_obs_max_threshold),
        "proprio_max_threshold": float(proprio_max_threshold),
        "image_mean_threshold": float(image_mean_threshold),
    }


def collect_phase12_training_episode(**kwargs: Any) -> Any:
    """Collect one real LeRobot-backed Phase12 WM-GRPO episode."""

    import numpy as np
    import torch
    from smolvla_grpo.phase11_rollout import detach_proc_snapshot
    from smolvla_grpo.lerobot_metaworld_adapter import (
        OfficialLeRobotMetaWorldGRPORollout,
        resolve_lerobot_horizon,
    )
    from smolvla_grpo.phase12_goals import (
        Phase12Goal,
        build_subgoal_schedule,
        compute_reset_parity,
        frame_index_to_filename,
        select_required_oracle_frame_indices,
        should_fail_reset_parity,
    )
    from smolvla_grpo.phase12_rollout import collect_phase12_episode
    from smolvla_grpo.phase12_wm_reward import _encode_structured, score_phase12_chunk_with_wm

    args = kwargs["args"]
    bundle = kwargs["bundle"]
    wm_bundle = kwargs["wm_bundle"]
    old_wrapper = kwargs["old_wrapper"]
    action_dim = int(kwargs["action_dim"])
    update_index = int(kwargs["update_index"])
    reset_seed = int(kwargs["reset_seed"])
    output_dir = Path(kwargs["output_dir"])
    episode_dir = output_dir / "rollouts" / f"update_{update_index:04d}_episode_{update_index:04d}"
    oracle_dir = output_dir / "oracle" / f"seed_{reset_seed}"

    env_h = OfficialLeRobotMetaWorldGRPORollout(task=args.task, n_envs=1, enable_expert_oracle=True)
    try:
        max_steps = resolve_lerobot_horizon(env_h, int(args.max_steps))
        oracle = _rollout_phase12_oracle(
            env_h=env_h,
            seed=reset_seed,
            max_steps=max_steps,
            output_dir=oracle_dir,
            fps=6,
        )
        schedule = build_subgoal_schedule(
            max_frame_1based=len(oracle["frames"]),
            chunk_len=int(args.chunk_len),
            success_frame_1based=oracle["success_frame_1based"],
        )
        _write_selected_frames_png(
            oracle_dir / "frames",
            oracle["frames"],
            select_required_oracle_frame_indices(
                max_frame_1based=len(oracle["frames"]),
                schedule=schedule,
            ),
        )
        goals: list[Phase12Goal] = []
        for subgoal_index, frame_idx in enumerate(schedule.primary_frames_1based):
            frame_idx = min(int(frame_idx), len(oracle["frames"]))
            companion = frame_idx + 1 if frame_idx + 1 <= len(oracle["frames"]) else None
            frame_path = oracle_dir / "frames" / frame_index_to_filename(frame_idx)
            companion_path = (
                oracle_dir / "frames" / frame_index_to_filename(companion)
                if companion is not None
                else None
            )
            encoded = _encode_structured(
                wm_bundle,
                oracle["wm_frames"][frame_idx - 1],
                oracle["proprios"][frame_idx - 1],
                mode=args.goal_latent_mode,
            )
            goals.append(
                Phase12Goal(
                    subgoal_index=subgoal_index,
                    frame_index_1based=frame_idx,
                    frame_path=frame_path,
                    companion_frame_index_1based=companion,
                    companion_frame_path=companion_path,
                    proprio=oracle["proprios"][frame_idx - 1],
                    goal_visual=encoded["visual"],
                    goal_proprio=encoded.get("proprio"),
                    source="lerobot_expert_oracle",
                )
            )

        reset_obs = env_h.reset(reset_seed)
        reset_frame = policy_rgb_from_obs(reset_obs)
        reset_proprio = np.asarray(env_h.last_agent_pos(), dtype=np.float32)
        reset_raw_obs = np.asarray(env_h.last_raw_obs(), dtype=np.float64)
        reset_metrics = compute_reset_parity(
            oracle["frames"][0],
            reset_frame,
            oracle["proprios"][0],
            reset_proprio,
        )
        reset_debug_report = _save_phase12_reset_debug_artifacts(
            output_dir=output_dir,
            oracle_frame=oracle["frames"][0],
            reset_frame=reset_frame,
            oracle_raw_obs=oracle["raw_obs"][0],
            reset_raw_obs=reset_raw_obs,
            reset_metrics=reset_metrics,
            reset_seed=reset_seed,
        )
        reset_gate = _phase12_reset_gate_decision(
            reset_metrics=reset_metrics,
            reset_debug_report=reset_debug_report,
        )
        # region agent log
        _phase12_agent_debug_log(
            run_id=__import__("os").environ.get("AGENT_DEBUG_RUN_ID", "phase12-decode-post-fix-bounded"),
            hypothesis_id="H4,H5,H6,H7",
            location="scripts/grpo/train_phase12_wm_chunk_grpo.py:collect_phase12_training_episode",
            message="phase12 reset parity full-state comparison before selected rollout",
            data={
                "reset_seed": int(reset_seed),
                "reset_metrics": reset_metrics,
                "oracle_frame0": _phase12_frame_debug(oracle["frames"][0]),
                "reset_frame": _phase12_frame_debug(reset_frame),
                "oracle_proprio0": _phase12_vector_debug(oracle["proprios"][0]),
                "reset_proprio": _phase12_vector_debug(reset_proprio),
                "oracle_raw_obs0": _phase12_vector_debug(oracle["raw_obs"][0], max_items=80),
                "reset_raw_obs": _phase12_vector_debug(reset_raw_obs, max_items=80),
                "raw_obs_shape_match": reset_debug_report["raw_obs_shape_match"],
                "raw_obs_max_abs_diff": reset_debug_report["raw_obs_max_abs_diff"],
                "raw_obs_mean_abs_diff": reset_debug_report["raw_obs_mean_abs_diff"],
                "reset_gate": reset_gate,
                "debug_artifact_paths": reset_debug_report["paths"],
            },
        )
        # endregion
        if args.reset_mismatch == "fail" and reset_gate["fail"]:
            raise RuntimeError(
                f"Phase12 reset mismatch for seed {reset_seed}: "
                f"{reset_metrics} gate={reset_gate} artifacts={reset_debug_report['paths']}"
            )

        rollout_env = _Phase12SelectedRolloutEnv(
            env_h=env_h,
            bundle=bundle,
            seed=reset_seed,
            initial_obs=reset_obs,
            initial_frame=reset_frame,
            initial_proprio=reset_proprio,
        )
        score_inputs: dict[tuple[int, int], dict[str, Any]] = {}

        def sampler(root_observation, *, root_id, num_candidates, segment_index, goal):
            del root_id, goal
            proc = root_observation["proc"]
            for candidate_index in range(int(num_candidates)):
                gen = torch.Generator(device=old_wrapper.bundle.device)
                gen.manual_seed(reset_seed * 1000003 + int(segment_index) * 7919 + int(candidate_index))
                sample = _sample_old_action_chunk(
                    old_wrapper,
                    proc,
                    chunk_len=int(args.chunk_len),
                    rng=gen,
                    use_inference_mode=bool(args.old_policy_inference_mode),
                )
                candidate = _phase12_sample_to_candidate_dict(sample, candidate_index=int(candidate_index))
                candidate["proc_root_snapshot"] = detach_proc_snapshot(proc)
                yield candidate

        def score_fn(root_observation, candidate, goal, *, root_id, segment_index):
            del root_id
            goal_latent = {"visual": goal.goal_visual}
            if args.goal_latent_mode == "visual_proprio":
                goal_latent["proprio"] = goal.goal_proprio
            # region agent log
            _phase12_agent_debug_log(
                run_id=__import__("os").environ.get("AGENT_DEBUG_RUN_ID", "phase12-bounded-wm-issue"),
                hypothesis_id="H1,H2,H3",
                location="scripts/grpo/train_phase12_wm_chunk_grpo.py:collect_phase12_training_episode.score_fn",
                message="Phase12 candidate selected for WM scoring input audit",
                data={
                    "segment_index": int(segment_index),
                    "candidate_index": int(candidate.candidate_index),
                    "goal_frame_index_1based": int(goal.frame_index_1based),
                    "goal_frame_path": str(goal.frame_path),
                    "companion_frame_index_1based": None
                    if goal.companion_frame_index_1based is None
                    else int(goal.companion_frame_index_1based),
                    "action_metadata": dict(candidate.action_metadata),
                },
            )
            # endregion
            score_inputs[(int(segment_index), int(candidate.candidate_index))] = {
                "image": root_observation["image"],
                "proprio": root_observation["proprio"],
                "actions": candidate.exec_actions_for_wm,
            }
            return score_phase12_chunk_with_wm(
                wm_bundle=wm_bundle,
                image=root_observation["image"],
                proprio=root_observation["proprio"],
                chunk_actions=candidate.exec_actions_for_wm,
                goal=goal_latent,
                candidate_index=int(candidate.candidate_index),
                proprio_alpha=float(args.proprio_alpha),
                mode=args.goal_latent_mode,
            )

        episode = collect_phase12_episode(
            env=rollout_env,
            policy_sampler=sampler,
            score_fn=score_fn,
            goals=goals,
            num_candidates=int(args.group_size),
            update_index=update_index,
            episode_index=update_index,
            reward_key=args.reward_key,
            action_profile=args.action_profile,
            action_low=np.full((action_dim,), -1.0, dtype=np.float32),
            action_high=np.full((action_dim,), 1.0, dtype=np.float32),
            preprocessor=wm_bundle.preprocessor,
            env_action_dim=action_dim,
            wm_action_dim=int(wm_bundle.planner_action_dim),
            metadata={"reset_metrics": reset_metrics},
        )
        selected_video = write_phase12_episode_video(
            video_path=episode_dir / "selected_action_rollout.mp4",
            frames=rollout_env.frames,
            rewards=rollout_env.rewards,
            successes=rollout_env.successes,
            fps=6,
            overlay_mode="cumulative_reward",
        )
        meta = dict(episode.metadata)
        meta.update(phase12_episode_training_metadata(episode, args.reward_key))
        meta.update(
            {
                "frames": rollout_env.frames,
                "env_rewards": rollout_env.rewards,
                "successes": rollout_env.successes,
                "rollout_validation_video": str(selected_video),
                "selected_action_rollout_video": str(selected_video),
                "oracle_baseline_video": str(oracle["video_path"]),
                "oracle_baseline_video_status": "ok",
                "oracle_manifest_path": str(oracle["manifest_path"]),
            }
        )
        _build_phase12_selected_decode_artifacts(
            args=args,
            episode=episode,
            episode_dir=episode_dir,
            rollout_env=rollout_env,
            score_inputs=score_inputs,
            wm_bundle=wm_bundle,
            action_dim=action_dim,
            meta=meta,
        )
        meta.setdefault("wm_decode_status", "disabled")
        return _with_episode_metadata(episode, meta)
    finally:
        env_h.close()


def _write_selected_frames_png(frames_dir: Path, frames: list[Any], frame_indices_1based: list[int]) -> None:
    import imageio.v2 as imageio
    import numpy as np
    from smolvla_grpo.phase12_goals import frame_index_to_filename

    frames_dir.mkdir(parents=True, exist_ok=True)
    for frame_index in frame_indices_1based:
        idx = int(frame_index)
        if idx < 1 or idx > len(frames):
            raise ValueError(f"frame index {idx} out of range for {len(frames)} frames")
        imageio.imwrite(frames_dir / frame_index_to_filename(idx), np.asarray(frames[idx - 1], dtype=np.uint8))


def _rollout_phase12_oracle(
    *,
    env_h: Any,
    seed: int,
    max_steps: int,
    output_dir: Path,
    fps: int,
) -> dict[str, Any]:
    import json
    import numpy as np

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    obs = env_h.reset(int(seed))
    policy_frame = policy_rgb_from_obs(obs)
    frames: list[np.ndarray] = [policy_frame]
    wm_frames: list[np.ndarray] = [wm_rgb_from_policy_rgb_corner2(policy_frame)]
    proprios: list[np.ndarray] = [np.asarray(env_h.last_agent_pos(), dtype=np.float32)]
    raw_obs: list[np.ndarray] = [np.asarray(env_h.last_raw_obs(), dtype=np.float64)]
    actions: list[list[float]] = []
    rewards: list[float] = []
    successes: list[bool] = []
    success_frame_1based: int | None = None
    for step_idx in range(int(max_steps)):
        action = np.clip(env_h.expert_action(), -1.0, 1.0).reshape(1, -1).astype(np.float32)
        step = env_h.step(action)
        actions.append(action.reshape(-1).astype(float).tolist())
        rewards.append(float(step.reward))
        successes.append(bool(step.success))
        policy_frame = policy_rgb_from_obs(step.observation)
        frames.append(policy_frame)
        wm_frames.append(wm_rgb_from_policy_rgb_corner2(policy_frame))
        proprios.append(np.asarray(env_h.last_agent_pos(), dtype=np.float32))
        raw_obs.append(np.asarray(env_h.last_raw_obs(), dtype=np.float64))
        if bool(step.success) and success_frame_1based is None:
            success_frame_1based = int(step_idx + 2)
        if bool(step.success or step.terminated or step.truncated):
            break
    video_path = write_phase12_episode_video(
        video_path=output_dir / "oracle_baseline.mp4",
        frames=frames,
        rewards=rewards,
        successes=successes,
        fps=int(fps),
        overlay_mode="cumulative_reward",
    )
    manifest = {
        "seed": int(seed),
        "max_steps": int(max_steps),
        "frame_count": len(frames),
        "wm_frame_count": len(wm_frames),
        "action_count": len(actions),
        "success_any": any(successes),
        "success_frame_1based": success_frame_1based,
        "video_path": str(video_path),
        "policy_frame_contract": "lerobot_corner2_vhflip",
        "wm_frame_contract": "jepa_corner2_vflip",
    }
    manifest_path = output_dir / "oracle_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "frames": frames,
        "wm_frames": wm_frames,
        "proprios": proprios,
        "raw_obs": raw_obs,
        "actions": actions,
        "rewards": rewards,
        "successes": successes,
        "success_frame_1based": success_frame_1based,
        "video_path": video_path,
        "manifest_path": manifest_path,
    }


class _Phase12SelectedRolloutEnv:
    def __init__(
        self,
        *,
        env_h: Any,
        bundle: Any,
        seed: int,
        initial_obs: dict[str, Any],
        initial_frame: Any,
        initial_proprio: Any,
    ) -> None:
        self.env_h = env_h
        self.bundle = bundle
        self.seed = int(seed)
        self._obs = initial_obs
        self._frame = np.asarray(initial_frame, dtype=np.uint8)
        self._wm_frame = wm_rgb_from_policy_rgb_corner2(self._frame)
        self._proprio = initial_proprio
        self.frames: list[Any] = [self._frame]
        self.wm_frames: list[Any] = [self._wm_frame]
        self.rewards: list[float] = []
        self.successes: list[bool] = []
        self.action_space = getattr(env_h.inner, "single_action_space", None)

    def reset(self) -> dict[str, Any]:
        return self._root()

    def _root(self) -> dict[str, Any]:
        return {
            "id": f"seed{self.seed}_step{len(self.rewards)}",
            "seed": self.seed,
            "obs": self._obs,
            "image": self._wm_frame,
            "policy_image": self._frame,
            "proprio": self._proprio,
            "proc": self.env_h.build_proc(self._obs, bundle=self.bundle),
        }

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        import numpy as np

        action_batch = np.asarray(action, dtype=np.float32).reshape(1, -1)
        step = self.env_h.step(action_batch)
        self._obs = step.observation
        self._frame = policy_rgb_from_obs(step.observation)
        self._wm_frame = wm_rgb_from_policy_rgb_corner2(self._frame)
        self._proprio = np.asarray(self.env_h.last_agent_pos(), dtype=np.float32)
        self.frames.append(self._frame)
        self.wm_frames.append(self._wm_frame)
        self.rewards.append(float(step.reward))
        self.successes.append(bool(step.success))
        info = dict(step.info)
        info["success"] = bool(step.success)
        return self._root(), float(step.reward), bool(step.terminated), bool(step.truncated), info


def _decode_phase12_prediction_frames(
    wm_bundle: Any,
    *,
    image: Any,
    proprio: Any,
    actions: Any,
    mode: str,
) -> list[Any]:
    import numpy as np
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
        # region agent log
        try:
            import json as _json
            import os as _os
            import time as _time

            if _os.environ.get("AGENT_DEBUG_PHASE12_WM_ACTIONS", "").strip().lower() in {"1", "true", "yes"}:
                payload = {
                    "sessionId": "588128",
                    "id": f"phase12_decode_{_os.getpid()}_{int(_time.time() * 1000)}",
                    "timestamp": int(_time.time() * 1000),
                    "runId": _os.environ.get("AGENT_DEBUG_RUN_ID", "phase12-decode-post-fix"),
                    "hypothesisId": "H2",
                    "location": "scripts/grpo/train_phase12_wm_chunk_grpo.py:_decode_phase12_prediction_frames",
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
                with open("/vol/bitbucket/aa6622/.logs/debug-588128.log", "a", encoding="utf-8") as _f:
                    _f.write(_json.dumps(payload, sort_keys=True) + "\n")
        except Exception:
            pass
        # endregion
    frames, failure = _decode_latent_trace_to_frames(
        wm_bundle,
        DecodeTrace(visual_latents=visual_latents, proprio_latents=proprio_latents),
    )
    if failure is not None:
        raise RuntimeError(failure)
    return frames


def _ensure_checkpointable_policy_state(policy: Any) -> dict:
    state = policy.state_dict()
    return dict(state) if state else {"_empty_policy_state": True}


def _save_latest_and_numbered(
    *,
    ckpt_dir: Path,
    policy_state: dict,
    optimizer_state: dict,
    update_index: int,
    args: argparse.Namespace,
    extra: dict[str, Any],
) -> Path:
    ckpt_path = ckpt_dir / "latest.pt"
    save_grpo_checkpoint(
        ckpt_path,
        policy_state=policy_state,
        optimizer_state=optimizer_state,
        update_index=update_index,
        args=vars(args),
        extra=extra,
    )
    if (int(update_index) + 1) % int(args.save_every) == 0 or int(args.num_updates) == 1:
        save_grpo_checkpoint(
            ckpt_dir / f"update_{int(update_index) + 1:04d}.pt",
            policy_state=policy_state,
            optimizer_state=optimizer_state,
            update_index=update_index,
            args=vars(args),
            extra=extra,
        )
    return ckpt_path


def _chunk_grpo_row_loss(
    new_lp: Any,
    old_lp: Any,
    advantage: Any,
    *,
    clip_eps: float,
    normalizer: int,
) -> tuple[Any, Any]:
    import torch

    ratio = torch.exp(new_lp.float() - old_lp.float())
    clipped_ratio = torch.clamp(ratio, 1.0 - float(clip_eps), 1.0 + float(clip_eps))
    loss = -torch.minimum(ratio * advantage.float(), clipped_ratio * advantage.float()) / float(normalizer)
    return loss, ratio


def _ratio_stats_from_tensors(old_lp: Any, new_lp: Any, *, clip_eps: float) -> dict[str, float]:
    import torch

    old = torch.as_tensor(old_lp).float()
    new = torch.as_tensor(new_lp).float()
    ratio = torch.exp(new - old)
    low = 1.0 - float(clip_eps)
    high = 1.0 + float(clip_eps)
    clip_fraction = ((ratio < low) | (ratio > high)).float().mean()
    return {
        "ratio_mean": float(ratio.mean().item()),
        "ratio_max": float(ratio.max().item()),
        "ratio_min": float(ratio.min().item()),
        "ratio_clip_fraction": float(clip_fraction.item()),
        "approx_kl": float((old.detach() - new.detach()).float().mean().item()),
    }


def _backward_chunk_grpo_loss_microbatched(
    *,
    train_wrapper: Any,
    proc_snapshots: list[Any],
    unsquashed_chunks: list[Any],
    old_lp: Any,
    advantages: Any,
    clip_eps: float,
) -> tuple[float, dict[str, float], Any]:
    import torch

    row_count = len(unsquashed_chunks)
    if row_count == 0:
        raise RuntimeError("microbatch GRPO loss needs at least one chunk")
    old_lp = torch.as_tensor(old_lp).float()
    advantages = torch.as_tensor(advantages).float()
    new_lp_values: list[torch.Tensor] = []
    loss_values: list[torch.Tensor] = []
    for row_idx, (proc, chunk) in enumerate(zip(proc_snapshots, unsquashed_chunks, strict=True)):
        new_lp_row = train_wrapper.get_action_probs_for_chunk_from_proc(proc, chunk).sum()
        loss_row, _ratio = _chunk_grpo_row_loss(
            new_lp_row,
            old_lp[row_idx].to(new_lp_row.device),
            advantages[row_idx].to(new_lp_row.device),
            clip_eps=float(clip_eps),
            normalizer=row_count,
        )
        loss_values.append(loss_row.detach().cpu())
        new_lp_values.append(new_lp_row.detach().cpu())
        loss_row.backward()
    new_lp = torch.stack(new_lp_values).reshape(old_lp.shape)
    stats = _ratio_stats_from_tensors(old_lp.detach().cpu(), new_lp, clip_eps=float(clip_eps))
    loss = float(torch.stack(loss_values).sum().item())
    return loss, stats, new_lp


def run_wm_grpo_train(args: argparse.Namespace, out: Path) -> int:
    import torch
    from torch import nn

    from smolvla_grpo.checkpointing import load_grpo_checkpoint
    from smolvla_grpo.grpo_math import compute_group_advantages
    from smolvla_grpo.phase12_rollout import chunk_grpo_loss

    bundle, wm_bundle, action_dim = load_phase12_train_resources(args)
    train_wrapper, trainable = build_train_wrapper(args, bundle, action_dim)
    optimizer = torch.optim.AdamW(trainable, lr=float(args.lr), betas=(0.9, 0.95))
    start_update = 0
    if args.resume is not None:
        ckpt = load_grpo_checkpoint(args.resume, map_location=bundle.device)
        bundle.policy.load_state_dict(ckpt["policy_state_dict"], strict=False)
        if ckpt.get("optimizer_state_dict"):
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_update = int(ckpt.get("update_index", -1)) + 1

    old_policy = copy.deepcopy(bundle.policy).eval().to(bundle.device)
    for param in old_policy.parameters():
        param.requires_grad_(False)
    old_wrapper = build_old_wrapper(args, bundle, old_policy, action_dim)
    ckpt_dir = out / "checkpoints"
    first_episode: Any | None = None
    end_update = start_update + int(args.num_updates)
    for update_index in range(start_update, end_update):
        update_t0 = time.perf_counter()
        reset_seed = int(args.train_seed_base) + int(update_index)
        episode = collect_phase12_training_episode(
            args=args,
            bundle=bundle,
            wm_bundle=wm_bundle,
            old_policy=old_policy,
            old_wrapper=old_wrapper,
            action_dim=action_dim,
            update_index=update_index,
            reset_seed=reset_seed,
            output_dir=out,
        )
        if hasattr(episode, "segments"):
            meta = dict(getattr(episode, "metadata", {}) or {})
            meta.update(phase12_episode_training_metadata(episode, args.reward_key))
            episode = _with_episode_metadata(episode, meta)
        if first_episode is None:
            first_episode = episode
        meta = dict(getattr(episode, "metadata", {}) or {})
        segment_rows = meta.get("segment_candidate_rewards") or [meta.get("candidate_rewards", [])]
        segment_rewards = [
            torch.tensor(row, dtype=torch.float32, device=bundle.device)
            for row in segment_rows
            if len(row) > 0
        ]
        segment_advantages = [compute_group_advantages(row) for row in segment_rewards]
        if not segment_advantages:
            raise RuntimeError("Phase12 episode produced no candidate rewards")
        if int(args.num_updates) == 1 and all(torch.allclose(a, torch.zeros_like(a)) for a in segment_advantages):
            raise RuntimeError("one-update WM-GRPO smoke produced zero advantages for every segment")
        rewards = torch.cat(segment_rewards, dim=0)
        advantages = torch.cat(segment_advantages, dim=0)
        old_lp = torch.tensor(meta["old_logprob_sums"], dtype=torch.float32, device=bundle.device)
        if torch.allclose(advantages, torch.zeros_like(advantages)):
            ckpt_path = _save_latest_and_numbered(
                ckpt_dir=ckpt_dir,
                policy_state=_ensure_checkpointable_policy_state(bundle.policy),
                optimizer_state=optimizer.state_dict(),
                update_index=update_index,
                args=args,
                extra={"skipped": True, "reason": "zero_advantages"},
            )
            row = {
                "created_at": utc_now_iso(),
                "event": "update_complete",
                "mode": "wm_grpo_train",
                "update_index": int(update_index),
                "reset_seed": int(reset_seed),
                "skipped": True,
                "reason": "zero_advantages",
                "optimizer_step": False,
                "checkpoint_path": str(ckpt_path),
            }
            write_jsonl_row(out / "progress.jsonl", row)
            continue
        optimizer.zero_grad(set_to_none=True)
        if args.logprob_backward_mode == "microbatch":
            loss_value, ratio_stats, new_lp_for_log = _backward_chunk_grpo_loss_microbatched(
                train_wrapper=train_wrapper,
                proc_snapshots=meta["proc_root_snapshots"],
                unsquashed_chunks=meta["unsquashed_chunks"],
                old_lp=old_lp,
                advantages=advantages,
                clip_eps=float(args.clip_eps),
            )
        else:
            new_lp_rows = [
                train_wrapper.get_action_probs_for_chunk_from_proc(proc, chunk).sum()
                for proc, chunk in zip(meta["proc_root_snapshots"], meta["unsquashed_chunks"], strict=True)
            ]
            new_lp = torch.stack(new_lp_rows)
            loss, ratio_stats = chunk_grpo_loss(old_lp, new_lp, advantages, clip_eps=float(args.clip_eps))
            loss_value = float(loss.detach().cpu())
            new_lp_for_log = new_lp.detach().cpu()
            loss.backward()
        nn.utils.clip_grad_norm_(trainable, float(args.grad_clip))
        optimizer.step()
        old_policy.load_state_dict(bundle.policy.state_dict())
        old_policy.eval()
        old_wrapper._policy = old_policy
        extra = {"loss": float(loss_value), **ratio_stats}
        ckpt_path = _save_latest_and_numbered(
            ckpt_dir=ckpt_dir,
            policy_state=_ensure_checkpointable_policy_state(bundle.policy),
            optimizer_state=optimizer.state_dict(),
            update_index=update_index,
            args=args,
            extra=extra,
        )
        write_jsonl_row(
            out / "progress.jsonl",
            {
                "created_at": utc_now_iso(),
                "event": "update_complete",
                "mode": "wm_grpo_train",
                "update_index": int(update_index),
                "reset_seed": int(reset_seed),
                "optimizer_step": True,
                "loss": float(loss_value),
                "advantages": advantages.detach().cpu().tolist(),
                "returns": rewards.detach().cpu().tolist(),
                "new_logprob_sums": new_lp_for_log.reshape(-1).tolist(),
                "segment_candidate_rewards": [row.detach().cpu().tolist() for row in segment_rewards],
                "segment_advantages": [row.detach().cpu().tolist() for row in segment_advantages],
                "checkpoint_path": str(ckpt_path),
                "update_seconds": float(time.perf_counter() - update_t0),
                "action_clip_fraction": float(meta.get("action_clip_fraction", 0.0)),
                "action_clip_any_fraction": float(meta.get("action_clip_any_fraction", 0.0)),
                "raw_action_max_abs": float(meta.get("raw_action_max_abs", 0.0)),
                "clipped_action_max_abs": float(meta.get("clipped_action_max_abs", 0.0)),
                "clip_delta_max_abs": float(meta.get("clip_delta_max_abs", 0.0)),
                **ratio_stats,
            },
        )
    if first_episode is not None:
        smoke = _episode_smoke_manifest(first_episode)
        write_manifest(out / "smoke_manifest.json", smoke)
    print("PHASE12_WM_CHUNK_GRPO_TRAIN_DONE", f"updates={int(args.num_updates)}", f"out={out}", flush=True)
    return 0


def run_rollout_validation(args: argparse.Namespace, out: Path) -> int:
    episode = run_phase12_episode(
        args=args,
        output_dir=out,
        episode_index=0,
        action_profile=args.action_profile,
        seed=int(args.train_seed_base),
    )
    smoke = _episode_smoke_manifest(episode)
    write_manifest(out / "smoke_manifest.json", smoke)
    print("PHASE12_WM_CHUNK_ROLLOUT_VALIDATION_DONE", f"out={out}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(args)
    write_manifest(out / "train_manifest.json", manifest)
    write_jsonl_row(
        out / "progress.jsonl",
        {
            "created_at": utc_now_iso(),
            "event": "dry_run" if args.dry_run else "run_start",
            "mode": args.mode,
            "action_profile": args.action_profile,
            "chunk_len": int(args.chunk_len),
            "goal_latent_mode": args.goal_latent_mode,
            "proprio_alpha": float(args.proprio_alpha),
        },
    )
    if args.dry_run:
        print("PHASE12_WM_CHUNK_DRY_RUN_OK", flush=True)
        return 0
    config_error = _validate_real_mode(args)
    if config_error is not None:
        write_jsonl_row(
            out / "progress.jsonl",
            {
                "created_at": utc_now_iso(),
                "event": "configuration_error",
                "reason": config_error,
            },
        )
        print(f"PHASE12_CONFIGURATION_ERROR {config_error}", flush=True)
        return 2
    if args.mode == "wm_grpo_train":
        return run_wm_grpo_train(args, out)
    return run_rollout_validation(args, out)


if __name__ == "__main__":
    raise SystemExit(main())

