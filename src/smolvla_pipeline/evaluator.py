from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
import time
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Protocol, Sequence

from smolvla_obs_state import flatten_obs_state as _flatten_obs_state

from metaworld_determinism import gymnasium_reset_strict, seed_metaworld_process

from smolvla_pipeline.hf_hub_local_resolve import (
    resolve_hf_hub_repo_to_local_snapshot,
    should_resolve_hf_hub_to_local,
    should_strict_require_local_hf,
)

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover - environment dependent import
    plt = None


_FALLBACK_PNG_BYTES = bytes(
    [
        0x89,
        0x50,
        0x4E,
        0x47,
        0x0D,
        0x0A,
        0x1A,
        0x0A,
        0x00,
        0x00,
        0x00,
        0x0D,
        0x49,
        0x48,
        0x44,
        0x52,
        0x00,
        0x00,
        0x00,
        0x01,
        0x00,
        0x00,
        0x00,
        0x01,
        0x08,
        0x06,
        0x00,
        0x00,
        0x00,
        0x1F,
        0x15,
        0xC4,
        0x89,
        0x00,
        0x00,
        0x00,
        0x0D,
        0x49,
        0x44,
        0x41,
        0x54,
        0x78,
        0x9C,
        0x63,
        0x60,
        0x60,
        0x60,
        0xF8,
        0x0F,
        0x00,
        0x01,
        0x04,
        0x01,
        0x00,
        0x5F,
        0x09,
        0x96,
        0x89,
        0x00,
        0x00,
        0x00,
        0x00,
        0x49,
        0x45,
        0x4E,
        0x44,
        0xAE,
        0x42,
        0x60,
        0x82,
    ]
)


@dataclass
class EpisodeRollout:
    actions: list[list[float]]
    rewards: list[float]
    successes: list[bool]
    frames: list[Any]
    terminated: bool
    truncated: bool


class EvalBackend(Protocol):
    def rollout_episode(self, *, episode_index: int, reset_seed: int) -> EpisodeRollout:
        ...

    def close(self) -> None:
        ...


BackendFactory = Callable[..., EvalBackend]


def _smolvla_eval_log(message: str) -> None:
    print(message, flush=True)


# #region agent log
def _agent_debug_ndjson_enabled() -> bool:
    raw = os.environ.get("SMOLVLA_AGENT_DEBUG_NDJSON", "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _agent_debug_ndjson(
    *,
    hypothesis_id: str,
    location: str,
    message: str,
    data: dict[str, Any] | None = None,
) -> None:
    """Append one NDJSON line for debug-mode analysis (Slurm-safe, no secrets)."""
    if not _agent_debug_ndjson_enabled():
        return
    try:
        row: dict[str, Any] = {
            "sessionId": "d2f934",
            "timestamp": int(time.time() * 1000),
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data or {},
        }
        log_path = Path("/vol/bitbucket/aa6622/.cursor/debug-d2f934.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(row) + "\n")
    except Exception:
        pass


# #endregion


def _progress_jsonl_enabled() -> bool:
    raw = os.environ.get("SMOLVLA_EVAL_PROGRESS_JSONL", "true").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _as_bool(raw: str | bool) -> bool:
    if isinstance(raw, bool):
        return raw
    value = str(raw).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {raw!r}")


def _is_remote_checkpoint_id(checkpoint: str) -> bool:
    parts = checkpoint.split("/")
    if len(parts) != 2:
        return False
    if not parts[0] or not parts[1]:
        return False
    if parts[0].startswith(".") or parts[0].startswith("~"):
        return False
    return True


def _maybe_resolve_hf_repo_id(repo_id: str, *, label: str) -> str:
    if not should_resolve_hf_hub_to_local():
        return repo_id
    strict = should_strict_require_local_hf()
    resolved = resolve_hf_hub_repo_to_local_snapshot(repo_id, strict=strict)
    if resolved != repo_id:
        _smolvla_eval_log(
            f"smolvla_eval: hf_hub_local_snapshot label={label} repo_id={repo_id!r} path={resolved!r}"
        )
    return resolved


def _is_local_checkpoint_like(checkpoint: str) -> bool:
    if checkpoint in {".", "..", "~"}:
        return True
    if checkpoint.startswith(("/", "./", "../", "~")):
        return True
    if checkpoint.endswith("/"):
        return True
    if "\\" in checkpoint:
        return True
    if _is_remote_checkpoint_id(checkpoint):
        return False
    return True


def _validate_checkpoint(checkpoint: str) -> str:
    if not _is_local_checkpoint_like(checkpoint):
        return checkpoint

    checkpoint_path = Path(checkpoint).expanduser()
    if not checkpoint_path.exists():
        raise ValueError(
            f"Checkpoint path does not exist: {checkpoint!r}. "
            "Provide an existing local path or a remote model id like 'owner/name'."
        )
    return str(checkpoint_path.resolve())


def _validate_overlay_mode(overlay_mode: str) -> str:
    if overlay_mode not in {"cumulative_reward", "reward", "reward_delta"}:
        raise ValueError(
            f"Invalid overlay_mode {overlay_mode!r}. "
            "Expected one of: 'cumulative_reward', 'reward', 'reward_delta'."
        )
    return overlay_mode


def _safe_success(info: dict[str, Any]) -> bool:
    for key in ("success", "is_success"):
        value = info.get(key)
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
    return False


def _resolve_max_steps() -> int:
    raw = os.environ.get("SMOLVLA_MAX_STEPS", "120").strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"Invalid SMOLVLA_MAX_STEPS={raw!r}. Expected a positive integer."
        ) from exc
    if value < 1:
        raise ValueError("SMOLVLA_MAX_STEPS must be >= 1")
    return value


def _validate_max_steps(max_steps: int) -> int:
    if int(max_steps) < 1:
        raise ValueError("max_steps must be >= 1")
    return int(max_steps)


def _resolve_camera_name() -> str:
    camera_name = os.environ.get("SMOLVLA_METAWORLD_CAMERA_NAME", "corner2").strip()
    if not camera_name:
        raise ValueError("SMOLVLA_METAWORLD_CAMERA_NAME must be non-empty.")
    return camera_name


def _resolve_flip_corner2() -> bool:
    return _as_bool(os.environ.get("SMOLVLA_FLIP_CORNER2", "true"))


def _resolve_save_frames() -> bool:
    return _as_bool(os.environ.get("SMOLVLA_SAVE_FRAMES", "false"))


def _resolve_save_action_trace() -> bool:
    return _as_bool(os.environ.get("SMOLVLA_SAVE_ACTIONS", "false"))


def _resolve_optional_int_env(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    value = raw.strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer when set; got {raw!r}.") from exc


def _resolve_task_text(task: str, *, override: str | None = None) -> str:
    if override is not None:
        cleaned = str(override).strip()
        if cleaned:
            return cleaned
    task_clean = str(task).strip()
    try:
        from lerobot.envs.metaworld import TASK_DESCRIPTIONS  # type: ignore

        text = TASK_DESCRIPTIONS.get(task_clean, "")
        if isinstance(text, str) and text.strip():
            return text.strip()
    except Exception:
        pass
    fallback = {
        "push-v3": "Push the puck to a goal",
    }
    return fallback.get(task_clean, f"meta-world task: {task_clean}")


def _maybe_flip_corner2_frame(frame: Any, *, camera_name: str, flip_corner2: bool) -> Any:
    import numpy as np

    if not (flip_corner2 and camera_name == "corner2"):
        return frame
    return np.ascontiguousarray(np.flip(np.asarray(frame), (0, 1)))


def _coerce_exec_action(action: Any, *, action_dim: int, np_module: Any) -> Any:
    if hasattr(action, "detach"):
        action_np = action.detach().float().cpu().numpy().reshape(-1)
    else:
        action_np = np_module.asarray(action, dtype=np_module.float32).reshape(-1)
    if action_np.size != int(action_dim):
        raise RuntimeError(
            f"Policy action dim mismatch: expected {action_dim}, got {action_np.size}. "
            "Refusing silent pad/truncate."
        )
    return np_module.clip(action_np, -1.0, 1.0)


def _write_reward_curve_png(path: Path, y_values: Sequence[float], overlay_mode: str) -> None:
    if plt is None:
        path.write_bytes(_FALLBACK_PNG_BYTES)
        return

    if overlay_mode == "cumulative_reward":
        y_label = "cumulative_reward"
        title = "Time vs cumulative reward"
    elif overlay_mode == "reward":
        y_label = "reward"
        title = "Time vs reward"
    else:
        y_label = "reward_delta"
        title = "Time vs reward delta"
    x_values = list(range(len(y_values)))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x_values, list(y_values), color="#1f77b4", linewidth=2.0)
    ax.set_xlabel("step")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _render_rgb_frame(env: Any) -> Any | None:
    import numpy as np

    try:
        frame = env.render()
    except Exception:
        return None
    if frame is None:
        return None
    frame_np = np.asarray(frame)
    if frame_np.ndim != 3:
        return None
    if frame_np.shape[-1] == 4:
        frame_np = frame_np[..., :3]
    if frame_np.shape[0] == 3 and frame_np.shape[-1] != 3:
        frame_np = np.transpose(frame_np, (1, 2, 0))
    if frame_np.dtype != np.uint8:
        if np.issubdtype(frame_np.dtype, np.floating) and float(np.max(frame_np)) <= 1.5:
            frame_np = (np.clip(frame_np, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            frame_np = np.clip(frame_np, 0, 255).astype(np.uint8)
    if frame_np.shape[-1] != 3:
        return None
    return np.ascontiguousarray(frame_np)


def _overlay_frame(frame: Any, text: str) -> Any:
    import numpy as np
    from PIL import Image, ImageDraw

    frame_np = np.asarray(frame)
    if frame_np.dtype != np.uint8:
        frame_np = np.clip(frame_np, 0, 255).astype(np.uint8)
    image = Image.fromarray(frame_np)
    draw = ImageDraw.Draw(image)
    draw.rectangle((8, 8, image.width - 8, 44), fill=(0, 0, 0))
    draw.text((14, 16), text, fill=(255, 255, 255))
    return np.asarray(image, dtype=np.uint8)


def _overlay_metric_value(
    *, reward: float, cumulative_reward: float, reward_delta: float, overlay_mode: str
) -> float:
    if overlay_mode == "cumulative_reward":
        return float(cumulative_reward)
    if overlay_mode == "reward":
        return float(reward)
    if overlay_mode == "reward_delta":
        return float(reward_delta)
    raise ValueError(f"Unsupported overlay_mode for metric selection: {overlay_mode!r}")


def _build_overlay_text(
    *,
    step: int,
    reward: float,
    cumulative_reward: float,
    reward_delta: float,
    success: bool,
    overlay_mode: str,
) -> str:
    metric_value = _overlay_metric_value(
        reward=reward,
        cumulative_reward=cumulative_reward,
        reward_delta=reward_delta,
        overlay_mode=overlay_mode,
    )
    return (
        f"step={step} reward={reward:.4f} "
        f"cumulative_reward={cumulative_reward:.4f} "
        f"reward_delta={reward_delta:.4f} "
        f"mode={overlay_mode} metric={metric_value:.4f} "
        f"success={int(success)}"
    )


def _write_episode_video(
    *,
    video_path: Path,
    frames: list[Any],
    rewards: list[float],
    successes: list[bool],
    overlay_mode: str,
    fps: int,
) -> None:
    import imageio.v2 as imageio

    if not frames:
        raise RuntimeError("No rendered frames were captured for this episode.")
    if fps < 1:
        raise ValueError("fps must be >= 1")

    video_path.parent.mkdir(parents=True, exist_ok=True)
    cumulative_by_step: list[float] = []
    reward_delta_by_step: list[float] = []
    running = 0.0
    prev_reward: float | None = None
    for reward in rewards:
        reward_value = float(reward)
        running += float(reward)
        cumulative_by_step.append(float(running))
        if prev_reward is None:
            reward_delta_by_step.append(0.0)
        else:
            reward_delta_by_step.append(float(reward_value - prev_reward))
        prev_reward = reward_value

    with imageio.get_writer(video_path, fps=fps) as writer:
        for idx, frame in enumerate(frames):
            if idx == 0 or not rewards:
                step = 0
                reward = 0.0
                cumulative = 0.0
                reward_delta = 0.0
                success = False
            else:
                step_idx = min(idx - 1, len(rewards) - 1)
                step = step_idx + 1
                reward = float(rewards[step_idx])
                cumulative = float(cumulative_by_step[step_idx])
                reward_delta = float(reward_delta_by_step[step_idx])
                success = bool(successes[step_idx]) if step_idx < len(successes) else False
            overlay_text = _build_overlay_text(
                step=step,
                reward=reward,
                cumulative_reward=cumulative,
                reward_delta=reward_delta,
                success=success,
                overlay_mode=overlay_mode,
            )
            writer.append_data(_overlay_frame(frame, overlay_text))

    if not video_path.exists() or video_path.stat().st_size <= 0:
        raise RuntimeError(f"Video write failed for {video_path}")


def _write_episode_frames_png(*, frames_dir: Path, frames: Sequence[Any]) -> None:
    """Write raw RGB frames as PNGs (oracle-style ``frame_000000.png``)."""
    import numpy as np

    try:
        import imageio.v2 as imageio
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "save_frames requires imageio. Install imageio in the evaluator environment."
        ) from exc

    if not frames:
        raise RuntimeError("No frames to write for save_frames.")
    frames_dir.mkdir(parents=True, exist_ok=True)
    for idx, frame in enumerate(frames):
        frame_np = np.asarray(frame)
        if frame_np.ndim != 3 or frame_np.shape[-1] not in (3, 4):
            raise RuntimeError(f"Frame {idx} has invalid shape {frame_np.shape}; expected HxWx3/4.")
        if frame_np.shape[-1] == 4:
            frame_np = frame_np[..., :3]
        if frame_np.dtype != np.uint8:
            if np.issubdtype(frame_np.dtype, np.floating) and float(np.max(frame_np)) <= 1.5:
                frame_np = (np.clip(frame_np, 0.0, 1.0) * 255.0).astype(np.uint8)
            else:
                frame_np = np.clip(frame_np, 0, 255).astype(np.uint8)
        path = frames_dir / f"frame_{idx:06d}.png"
        imageio.imwrite(path, np.ascontiguousarray(frame_np))


def write_episode_artifacts(
    *,
    episode_dir: Path,
    actions: Sequence[Sequence[float]],
    rewards: Sequence[float],
    successes: Sequence[bool],
    frames_dir: Path | None = None,
    overlay_mode: str = "cumulative_reward",
    save_actions: bool = True,
) -> dict[str, str]:
    if not (len(actions) == len(rewards) == len(successes)):
        raise ValueError("actions, rewards, and successes must have matching lengths.")
    overlay_mode = _validate_overlay_mode(overlay_mode)

    episode_dir.mkdir(parents=True, exist_ok=True)
    if frames_dir is not None:
        frames_dir.mkdir(parents=True, exist_ok=True)

    action_rows: list[dict[str, Any]] = []
    cumulative_reward = 0.0
    prev_reward: float | None = None
    for step, (action, reward, success) in enumerate(zip(actions, rewards, successes)):
        reward_value = float(reward)
        cumulative_reward += reward_value
        reward_delta = 0.0 if prev_reward is None else float(reward_value - prev_reward)
        prev_reward = reward_value
        action_rows.append(
            {
                "step": int(step),
                "action": [float(component) for component in action],
                "reward": reward_value,
                "cumulative_reward": float(cumulative_reward),
                "reward_delta": float(reward_delta),
                "success": bool(success),
            }
        )

    actions_path = episode_dir / "actions.jsonl"
    if save_actions:
        with actions_path.open("w", encoding="utf-8") as action_fp:
            for row in action_rows:
                action_fp.write(json.dumps(row) + "\n")

    csv_path = episode_dir / "reward_curve.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as csv_fp:
        writer = csv.DictWriter(
            csv_fp, fieldnames=["step", "reward", "cumulative_reward", "reward_delta"]
        )
        writer.writeheader()
        for row in action_rows:
            writer.writerow(
                {
                    "step": row["step"],
                    "reward": row["reward"],
                    "cumulative_reward": row["cumulative_reward"],
                    "reward_delta": row["reward_delta"],
                }
            )

    png_path = episode_dir / "reward_curve.png"
    if overlay_mode == "cumulative_reward":
        plot_values = [float(row["cumulative_reward"]) for row in action_rows]
    elif overlay_mode == "reward":
        plot_values = [float(row["reward"]) for row in action_rows]
    else:
        plot_values = [float(row["reward_delta"]) for row in action_rows]

    _write_reward_curve_png(
        png_path,
        plot_values,
        overlay_mode,
    )
    out: dict[str, str] = {
        "reward_curve_csv": str(csv_path),
        "reward_curve_png": str(png_path),
    }
    if save_actions:
        out["actions"] = str(actions_path)
    return out


def _patch_external_datasets() -> None:
    import importlib.util
    import site
    import sys

    site_paths = list(site.getsitepackages())
    user_site = site.getusersitepackages()
    if user_site:
        site_paths.append(user_site)
    for item in site_paths:
        path = Path(item) / "datasets" / "__init__.py"
        if path.exists():
            spec = importlib.util.spec_from_file_location("datasets", str(path))
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                module.__file__ = str(path)
                sys.modules["datasets"] = module
                return


@dataclass
class _SmolVLABundle:
    policy: Any
    preprocessor: Any
    postprocessor: Any
    device: Any
    obs_image_key: str
    obs_state_key: str
    obs_env_state_key: str


def _resolve_policy_device(torch_module: Any) -> Any:
    policy_dev_raw = os.environ.get("SMOLVLA_POLICY_DEVICE", "").strip().lower()
    if policy_dev_raw in ("", "default", "auto"):
        if torch_module.cuda.is_available():
            return torch_module.device("cuda")
        return torch_module.device("cpu")
    if policy_dev_raw == "cuda" and not torch_module.cuda.is_available():
        return torch_module.device("cpu")
    return torch_module.device(policy_dev_raw)


def _load_smolvla_bundle(checkpoint: str) -> _SmolVLABundle:
    t0 = time.perf_counter()
    _smolvla_eval_log(f"smolvla_eval: load_bundle_begin checkpoint={checkpoint!r}")
    _agent_debug_ndjson(
        hypothesis_id="H1_load_phases",
        location="evaluator.py:_load_smolvla_bundle",
        message="bundle_load_begin",
        data={
            "checkpoint": checkpoint,
            "HF_HOME": os.environ.get("HF_HOME", ""),
            "elapsed_s": round(time.perf_counter() - t0, 4),
        },
    )
    ckpt_load = _maybe_resolve_hf_repo_id(checkpoint, label="policy_checkpoint")
    _patch_external_datasets()
    _smolvla_eval_log(
        f"smolvla_eval: load_bundle_patch_datasets_done elapsed_s={time.perf_counter() - t0:.2f}"
    )
    import inspect

    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.processor import PolicyProcessorPipeline
    from lerobot.processor.converters import (
        batch_to_transition,
        policy_action_to_transition,
        transition_to_batch,
        transition_to_policy_action,
    )
    from lerobot.utils.constants import OBS_ENV_STATE, OBS_IMAGE, OBS_STATE

    _smolvla_eval_log(
        f"smolvla_eval: load_bundle_imports_done elapsed_s={time.perf_counter() - t0:.2f}"
    )

    device = _resolve_policy_device(torch)
    vlm_raw = os.environ.get(
        "SMOLVLA_VLM_MODEL_NAME", "HuggingFaceTB/SmolVLM2-500M-Instruct"
    ).strip()
    load_vlm = os.environ.get("SMOLVLA_LOAD_VLM_WEIGHTS", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )
    vlm_resolved = (
        _maybe_resolve_hf_repo_id(vlm_raw, label="vlm_backbone")
        if load_vlm and vlm_raw
        else vlm_raw
    )
    model_kwargs: dict[str, Any] = {
        "device": str(device),
        "n_action_steps": 1,
        "expert_width_multiplier": 0.5,
        "self_attn_every_n_layers": 0,
        "load_vlm_weights": load_vlm,
        "vlm_model_name": vlm_resolved,
    }

    sig = inspect.signature(SmolVLAPolicy.from_pretrained)
    params = sig.parameters
    pretrained_keys = (
        "pretrained_name_or_path",
        "pretrained_model_name_or_path",
        "pretrained_path",
    )
    pretrained_key = next((k for k in pretrained_keys if k in params), None)
    accepts_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    supported_kwargs = dict(model_kwargs) if accepts_var_kwargs else {
        k: v for k, v in model_kwargs.items() if k in params
    }
    if "config" in params:
        try:
            policy_cfg = PreTrainedConfig.from_pretrained(pretrained_name_or_path=ckpt_load)
            for key, value in model_kwargs.items():
                if hasattr(policy_cfg, key):
                    setattr(policy_cfg, key, value)
            supported_kwargs["config"] = policy_cfg
        except Exception:
            pass
    _smolvla_eval_log(
        f"smolvla_eval: load_bundle_config_done checkpoint={ckpt_load!r} "
        f"elapsed_s={time.perf_counter() - t0:.2f}"
    )

    overrides = {"device_processor": {"device": str(device)}}
    _smolvla_eval_log(
        f"smolvla_eval: load_bundle_preprocessor_begin checkpoint={ckpt_load!r} "
        f"elapsed_s={time.perf_counter() - t0:.2f}"
    )
    preprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=ckpt_load,
        config_filename="policy_preprocessor.json",
        overrides=overrides,
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    )
    _agent_debug_ndjson(
        hypothesis_id="H1_load_phases",
        location="evaluator.py:_load_smolvla_bundle",
        message="preprocessor_from_pretrained_done",
        data={"elapsed_s": round(time.perf_counter() - t0, 4)},
    )
    _smolvla_eval_log(
        f"smolvla_eval: load_bundle_preprocessor_done elapsed_s={time.perf_counter() - t0:.2f}"
    )
    _smolvla_eval_log(
        f"smolvla_eval: load_bundle_postprocessor_begin checkpoint={ckpt_load!r} "
        f"elapsed_s={time.perf_counter() - t0:.2f}"
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=ckpt_load,
        config_filename="policy_postprocessor.json",
        overrides=overrides,
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    _agent_debug_ndjson(
        hypothesis_id="H1_load_phases",
        location="evaluator.py:_load_smolvla_bundle",
        message="postprocessor_from_pretrained_done",
        data={"elapsed_s": round(time.perf_counter() - t0, 4)},
    )
    _smolvla_eval_log(
        f"smolvla_eval: load_bundle_postprocessor_done elapsed_s={time.perf_counter() - t0:.2f}"
    )
    _smolvla_eval_log(
        f"smolvla_eval: load_bundle_policy_begin elapsed_s={time.perf_counter() - t0:.2f}"
    )
    _agent_debug_ndjson(
        hypothesis_id="H2_policy_weights",
        location="evaluator.py:_load_smolvla_bundle",
        message="before_SmolVLAPolicy_from_pretrained",
        data={
            "elapsed_s": round(time.perf_counter() - t0, 4),
            "load_vlm_weights": model_kwargs.get("load_vlm_weights"),
        },
    )
    if pretrained_key:
        supported_kwargs[pretrained_key] = ckpt_load
        policy = SmolVLAPolicy.from_pretrained(**supported_kwargs)
    else:
        policy = SmolVLAPolicy.from_pretrained(ckpt_load, **supported_kwargs)
    _agent_debug_ndjson(
        hypothesis_id="H2_policy_weights",
        location="evaluator.py:_load_smolvla_bundle",
        message="after_SmolVLAPolicy_from_pretrained",
        data={"elapsed_s": round(time.perf_counter() - t0, 4)},
    )
    policy.eval()
    _smolvla_eval_log(
        f"smolvla_eval: load_bundle_policy_done elapsed_s={time.perf_counter() - t0:.2f}"
    )
    _smolvla_eval_log(
        f"smolvla_eval: load_bundle_done elapsed_s={time.perf_counter() - t0:.2f}"
    )
    _agent_debug_ndjson(
        hypothesis_id="H1_load_phases",
        location="evaluator.py:_load_smolvla_bundle",
        message="bundle_load_done",
        data={"elapsed_s": round(time.perf_counter() - t0, 4)},
    )

    return _SmolVLABundle(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        device=device,
        obs_image_key=OBS_IMAGE,
        obs_state_key=OBS_STATE,
        obs_env_state_key=OBS_ENV_STATE,
    )


def _smolvla_state_dims(policy: Any) -> tuple[int, int]:
    agent_dim, env_dim = 4, 39
    feats = getattr(getattr(policy, "config", None), "input_features", None) or {}
    for name, ft in feats.items():
        shape = getattr(ft, "shape", None) or ()
        dim0 = int(shape[0]) if shape else 0
        if dim0 <= 0:
            continue
        if "environment_state" in name:
            env_dim = dim0
        elif name.endswith(".state") and "environment" not in name:
            agent_dim = dim0
    return agent_dim, env_dim


def _vectors_for_smolvla(flat: Any, agent_dim: int, env_dim: int) -> tuple[Any, Any]:
    import numpy as np

    flat_np = np.asarray(flat, dtype=np.float32).reshape(-1)
    env_vec = np.zeros(env_dim, dtype=np.float32)
    env_vec[: min(env_dim, flat_np.size)] = flat_np[: min(env_dim, flat_np.size)]
    agent_vec = np.zeros(agent_dim, dtype=np.float32)
    agent_vec[: min(agent_dim, env_vec.size)] = env_vec[: min(agent_dim, env_vec.size)]
    return agent_vec, env_vec


def _collect_policy_rgb(env: Any, obs: Any) -> Any:
    import numpy as np

    if isinstance(obs, dict):
        candidate_keys = ("image", "pixels", "rgb", "observation")
        for key in candidate_keys:
            if key not in obs:
                continue
            arr = np.asarray(obs[key])
            if arr.ndim == 3 and arr.shape[-1] in (3, 4):
                arr = arr[..., :3]
                if arr.dtype != np.uint8:
                    arr = np.clip(arr, 0, 255).astype(np.uint8)
                return arr
            if arr.ndim == 3 and arr.shape[0] in (3, 4):
                arr = np.transpose(arr[:3, ...], (1, 2, 0))
                if arr.dtype != np.uint8:
                    arr = np.clip(arr, 0, 255).astype(np.uint8)
                return arr
    frame = _render_rgb_frame(env)
    if frame is None:
        raise RuntimeError("Failed to collect RGB frame for SmolVLA policy input.")
    return frame


class _LeRobotMetaWorldBackend:
    def __init__(
        self,
        *,
        task: str,
        checkpoint: str,
        seed: int,
        max_steps: int,
        task_text: str | None = None,
        collect_frames: bool = True,
        bundle: _SmolVLABundle | None = None,
    ):
        import metaworld
        import numpy as np

        self._np = np
        self._task = task
        self._max_steps = max_steps
        self._camera_name = _resolve_camera_name()
        self._flip_corner2 = _resolve_flip_corner2()
        self._task_text = _resolve_task_text(task, override=task_text)
        self._bundle = bundle if bundle is not None else _load_smolvla_bundle(checkpoint)
        try:
            self._bundle.policy.reset()
        except Exception:
            pass
        self._agent_dim, self._env_dim = _smolvla_state_dims(self._bundle.policy)
        _smolvla_eval_log(f"smolvla_eval: backend_metaworld_mt1 task={task!r}")
        self._mt1 = metaworld.MT1(task)
        if task not in self._mt1.train_classes:
            available = ", ".join(sorted(self._mt1.train_classes.keys()))
            raise ValueError(f"Task {task!r} is not available in Meta-World MT1. Available: {available}")
        env_cls = self._mt1.train_classes[task]
        try:
            self._env = env_cls(render_mode="rgb_array", camera_name=self._camera_name)
        except Exception:
            self._env = env_cls()
        try:
            if hasattr(self._env, "render_mode"):
                self._env.render_mode = "rgb_array"
        except Exception:
            pass
        if self._camera_name == "corner2":
            try:
                self._env.model.cam_pos[2] = [0.75, 0.075, 0.7]
            except Exception:
                pass
        self._tasks = list(getattr(self._mt1, "train_tasks", []) or [])
        self._action_dim = int(np.prod(self._env.action_space.shape))
        _smolvla_eval_log(
            f"smolvla_eval: backend_env_ready task={task!r} action_dim={self._action_dim}"
        )
        self._target_episode_index_override = _resolve_optional_int_env(
            "SMOLVLA_TARGET_EPISODE_INDEX"
        )
        self._collect_frames = bool(collect_frames)

    def _reset(self, reset_seed: int) -> tuple[Any, dict[str, Any]]:
        reset_out = gymnasium_reset_strict(self._env, int(reset_seed))
        if isinstance(reset_out, tuple) and len(reset_out) >= 2:
            return reset_out[0], reset_out[1] if isinstance(reset_out[1], dict) else {}
        return reset_out, {}

    def _step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        step_out = self._env.step(action)
        if isinstance(step_out, tuple) and len(step_out) == 5:
            obs, reward, terminated, truncated, info = step_out
            return obs, float(reward), bool(terminated), bool(truncated), info if isinstance(info, dict) else {}
        if isinstance(step_out, tuple) and len(step_out) == 4:
            obs, reward, done, info = step_out
            return obs, float(reward), bool(done), False, info if isinstance(info, dict) else {}
        raise RuntimeError("Unexpected env.step() return format from Meta-World.")

    def _select_action(self, obs: Any) -> Any:
        import torch

        flat = _flatten_obs_state(obs)
        if flat.size == 0:
            raise RuntimeError("Received empty observation vector from environment.")
        agent_vec, env_vec = _vectors_for_smolvla(flat, self._agent_dim, self._env_dim)
        rgb = _collect_policy_rgb(self._env, obs)
        rgb = _maybe_flip_corner2_frame(
            rgb,
            camera_name=self._camera_name,
            flip_corner2=self._flip_corner2,
        )
        timg = torch.from_numpy(rgb).unsqueeze(0).permute(0, 3, 1, 2).contiguous().float() / 255.0
        timg = timg.to(self._bundle.device)
        st = torch.from_numpy(agent_vec).unsqueeze(0).to(self._bundle.device)
        es = torch.from_numpy(env_vec).unsqueeze(0).to(self._bundle.device)
        batch = {
            self._bundle.obs_image_key: timg,
            self._bundle.obs_state_key: st,
            self._bundle.obs_env_state_key: es,
            "task": self._task_text,
        }
        proc = self._bundle.preprocessor(batch)
        with torch.inference_mode():
            action = self._bundle.policy.select_action(proc)
        action = self._bundle.postprocessor(action)
        exec_action = _coerce_exec_action(action, action_dim=self._action_dim, np_module=self._np)
        return exec_action

    def _render_frame(self) -> Any | None:
        frame = _render_rgb_frame(self._env)
        if frame is None:
            return None
        return _maybe_flip_corner2_frame(
            frame,
            camera_name=self._camera_name,
            flip_corner2=self._flip_corner2,
        )

    def rollout_episode(self, *, episode_index: int, reset_seed: int) -> EpisodeRollout:
        seed_metaworld_process(int(reset_seed))
        if self._tasks:
            task_episode_index = (
                int(self._target_episode_index_override)
                if self._target_episode_index_override is not None
                else int(episode_index)
            )
            self._env.set_task(self._tasks[task_episode_index % len(self._tasks)])
        obs, _info = self._reset(reset_seed)
        try:
            self._bundle.policy.reset()
        except Exception:
            pass

        actions: list[list[float]] = []
        rewards: list[float] = []
        successes: list[bool] = []
        frames: list[Any] = []
        if getattr(self, "_collect_frames", True):
            first_frame = self._render_frame()
            if first_frame is not None:
                frames.append(first_frame)

        episode_terminated = False
        episode_truncated = False
        for _ in range(self._max_steps):
            action = self._select_action(obs)
            obs, reward, terminated, truncated, info = self._step(action)
            actions.append(action.astype(self._np.float32).reshape(-1).tolist())
            rewards.append(float(reward))
            successes.append(_safe_success(info))
            if getattr(self, "_collect_frames", True):
                frame = self._render_frame()
                if frame is not None:
                    frames.append(frame)
            if terminated or truncated:
                episode_terminated = bool(terminated)
                episode_truncated = bool(truncated)
                break

        return EpisodeRollout(
            actions=actions,
            rewards=rewards,
            successes=successes,
            frames=frames,
            terminated=episode_terminated,
            truncated=episode_truncated,
        )

    def close(self) -> None:
        try:
            self._env.close()
        except Exception:
            pass


def _create_lerobot_metaworld_backend(
    *,
    task: str,
    checkpoint: str,
    seed: int,
    max_steps: int,
    task_text: str | None = None,
    collect_frames: bool = True,
) -> EvalBackend:
    try:
        return _LeRobotMetaWorldBackend(
            task=task,
            checkpoint=checkpoint,
            seed=seed,
            max_steps=max_steps,
            task_text=task_text,
            collect_frames=collect_frames,
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to initialize real LeRobot + Meta-World backend. "
            "Ensure metaworld, lerobot, torch, imageio, and pillow are installed in the active env."
        ) from exc


def run_smolvla_eval(
    *,
    task: str,
    episodes: int,
    seed: int,
    checkpoint: str,
    output_dir: Path,
    video: str | bool,
    fps: int,
    overlay_mode: str,
    max_steps: int | None = None,
    save_frames: bool | None = None,
    save_actions: bool | None = None,
    task_text: str | None = None,
    backend_factory: BackendFactory | None = None,
) -> dict[str, Any]:
    if episodes < 1:
        raise ValueError("episodes must be >= 1")
    if fps < 1:
        raise ValueError("fps must be >= 1")
    video_enabled = _as_bool(video)
    overlay_mode = _validate_overlay_mode(overlay_mode)
    checkpoint_resolved = _validate_checkpoint(checkpoint)
    resolved_max_steps = _resolve_max_steps() if max_steps is None else _validate_max_steps(max_steps)
    resolved_save_frames = (
        bool(save_frames) if save_frames is not None else _resolve_save_frames()
    )
    resolved_save_actions = (
        bool(save_actions) if save_actions is not None else _resolve_save_action_trace()
    )
    collect_episode_frames = bool(video_enabled or resolved_save_frames)
    resolved_task_text = _resolve_task_text(task, override=task_text)

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    progress_path = output_dir / "progress.jsonl"
    _smolvla_eval_log(
        "smolvla_eval: start "
        f"task={task!r} episodes={episodes} checkpoint={checkpoint!r} "
        f"checkpoint_resolved={checkpoint_resolved!r} max_steps={resolved_max_steps} "
        f"save_frames={resolved_save_frames} save_actions={resolved_save_actions} "
        f"video_enabled={video_enabled} task_text={resolved_task_text!r} "
        f"HF_HOME={os.environ.get('HF_HOME', '')!r} "
        f"HUGGINGFACE_HUB_CACHE={os.environ.get('HUGGINGFACE_HUB_CACHE', '')!r} "
        f"output_dir={output_dir}"
    )

    factory = backend_factory or _create_lerobot_metaworld_backend
    t_backend = time.perf_counter()
    backend = factory(
        task=task,
        checkpoint=checkpoint_resolved,
        seed=seed,
        max_steps=resolved_max_steps,
        task_text=task_text,
        collect_frames=collect_episode_frames,
    )
    _smolvla_eval_log(
        f"smolvla_eval: backend_ready elapsed_s={time.perf_counter() - t_backend:.2f}"
    )

    episode_rows: list[dict[str, Any]] = []
    sum_rewards: list[float] = []
    max_rewards: list[float] = []
    success_rows: list[bool] = []
    video_paths: list[str] = []

    try:
        fixed_reset_seed_override = _resolve_optional_int_env("SMOLVLA_FIXED_RESET_SEED")
        for episode_index in range(episodes):
            reset_seed = (
                int(fixed_reset_seed_override)
                if fixed_reset_seed_override is not None
                else int(seed + episode_index)
            )
            episode_dir = output_dir / "episodes" / f"episode_{episode_index:04d}"
            _smolvla_eval_log(
                f"smolvla_eval: episode_begin {episode_index + 1}/{episodes} "
                f"episode_index={episode_index} reset_seed={reset_seed}"
            )
            t_roll = time.perf_counter()
            rollout = backend.rollout_episode(episode_index=episode_index, reset_seed=reset_seed)
            roll_s = time.perf_counter() - t_roll
            _smolvla_eval_log(
                f"smolvla_eval: episode_rollout_done {episode_index + 1}/{episodes} "
                f"steps={len(rollout.rewards)} elapsed_s={roll_s:.2f}"
            )
            if not (
                len(rollout.actions) == len(rollout.rewards) == len(rollout.successes)
            ):
                raise RuntimeError(
                    "Backend returned mismatched rollout lengths for actions/rewards/successes."
                )

            t_artifacts = time.perf_counter()
            artifact_paths = write_episode_artifacts(
                episode_dir=episode_dir,
                actions=rollout.actions,
                rewards=rollout.rewards,
                successes=rollout.successes,
                overlay_mode=overlay_mode,
                save_actions=resolved_save_actions,
            )

            frames_dir: Path | None = None
            if resolved_save_frames:
                frames_dir = output_dir / "frames" / f"episode_{episode_index:04d}"
                _write_episode_frames_png(frames_dir=frames_dir, frames=rollout.frames)

            video_path = output_dir / "videos" / f"{task}_0" / f"eval_episode_{episode_index:04d}.mp4"
            if video_enabled:
                _write_episode_video(
                    video_path=video_path,
                    frames=rollout.frames,
                    rewards=rollout.rewards,
                    successes=rollout.successes,
                    overlay_mode=overlay_mode,
                    fps=fps,
                )
                video_paths.append(str(video_path))
            art_s = time.perf_counter() - t_artifacts

            sum_reward = float(sum(rollout.rewards))
            max_reward = float(max(rollout.rewards) if rollout.rewards else 0.0)
            success_last = bool(rollout.successes[-1]) if rollout.successes else False
            success_any = any(bool(v) for v in rollout.successes)
            first_success_step = next(
                (int(step_idx) for step_idx, step_success in enumerate(rollout.successes) if bool(step_success)),
                None,
            )
            episode_success = bool(success_any)
            sum_rewards.append(sum_reward)
            max_rewards.append(max_reward)
            success_rows.append(episode_success)

            paths_obj: dict[str, str] = {
                "reward_curve_csv": str(
                    Path(artifact_paths["reward_curve_csv"]).relative_to(output_dir)
                ),
                "reward_curve_png": str(
                    Path(artifact_paths["reward_curve_png"]).relative_to(output_dir)
                ),
            }
            if resolved_save_actions and "actions" in artifact_paths:
                paths_obj["actions"] = str(
                    Path(artifact_paths["actions"]).relative_to(output_dir)
                )
            if video_enabled:
                paths_obj["video"] = str(video_path.relative_to(output_dir))
            if frames_dir is not None:
                paths_obj["frames_dir"] = str(frames_dir.relative_to(output_dir))

            episode_meta = {
                "episode_index": int(episode_index),
                "reset_seed": int(reset_seed),
                "n_steps": int(len(rollout.rewards)),
                "n_frames": int(len(rollout.frames)),
                "sum_reward": sum_reward,
                "max_reward": max_reward,
                "success": episode_success,
                "success_any": bool(success_any),
                "success_last": bool(success_last),
                "first_success_step": first_success_step,
                "terminated": bool(rollout.terminated),
                "truncated": bool(rollout.truncated),
                "paths": paths_obj,
                "reward_curve_mode": overlay_mode,
            }
            (episode_dir / "episode_meta.json").write_text(
                json.dumps(episode_meta, indent=2), encoding="utf-8"
            )
            episode_rows.append(episode_meta)
            _smolvla_eval_log(
                f"smolvla_eval: episode_artifacts_done {episode_index + 1}/{episodes} "
                f"success={episode_success} sum_reward={sum_reward:.4f} "
                f"elapsed_artifacts_s={art_s:.2f}"
            )
            if _progress_jsonl_enabled():
                progress_row = {
                    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "episode_index": int(episode_index),
                    "episodes_total": int(episodes),
                    "reset_seed": int(reset_seed),
                    "n_steps": int(len(rollout.rewards)),
                    "success": bool(episode_success),
                    "success_any": bool(success_any),
                    "success_last": bool(success_last),
                    "first_success_step": first_success_step,
                    "sum_reward": float(sum_reward),
                    "max_reward": float(max_reward),
                    "elapsed_rollout_s": round(roll_s, 4),
                    "elapsed_artifacts_s": round(art_s, 4),
                }
                if video_enabled:
                    progress_row["video"] = str(video_path.relative_to(output_dir))
                with progress_path.open("a", encoding="utf-8") as fp:
                    fp.write(json.dumps(progress_row) + "\n")
    finally:
        backend.close()

    success_percent = 100.0 * float(sum(1 for value in success_rows if value)) / float(
        len(success_rows)
    )
    eval_info = {
        "per_group": {
            task: {
                "avg_sum_reward": float(mean(sum_rewards)),
                "avg_max_reward": float(mean(max_rewards)),
                "pc_success": float(success_percent),
                "n_episodes": int(len(sum_rewards)),
                "video_paths": list(video_paths),
            }
        },
        "overall": {
            "avg_sum_reward": float(mean(sum_rewards)),
            "avg_max_reward": float(mean(max_rewards)),
            "pc_success": float(success_percent),
            "n_episodes": int(len(sum_rewards)),
            "video_paths": list(video_paths),
        },
    }
    (output_dir / "eval_info.json").write_text(json.dumps(eval_info, indent=2), encoding="utf-8")

    run_manifest = {
        "schema_version": "smolvla_run_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "started_at_utc": started_at,
        "task": task,
        "seed": int(seed),
        "episodes_requested": int(episodes),
        "max_steps": int(resolved_max_steps),
        "fps": int(fps),
        "video_enabled": bool(video_enabled),
        "checkpoint": str(checkpoint),
        "checkpoint_resolved": checkpoint_resolved,
        "overlay_mode": str(overlay_mode),
        "runtime_backend": "lerobot_metaworld",
        "camera_name": _resolve_camera_name(),
        "flip_corner2": _resolve_flip_corner2(),
        "task_text": resolved_task_text,
        "save_frames": bool(resolved_save_frames),
        "save_actions": bool(resolved_save_actions),
        "episodes": episode_rows,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2), encoding="utf-8"
    )
    _smolvla_eval_log(
        f"smolvla_eval: run_manifest_written episodes={len(episode_rows)} "
        f"output_dir={output_dir}"
    )
    return {"output_dir": str(output_dir), "eval_info": eval_info, "run_manifest": run_manifest}
