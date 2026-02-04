#!/usr/bin/env python3
"""Paired push-v3 rollouts: executed trajectories + JEPA-WM CEM-planned latents.

Writes per-episode trajectory shards and export_manifest.json for bridge_builder.
See docs/CEM_PAIRED_PUSHV3_SCHEMA.md for the contract.

- **Executed actions (default CEM-primary):** With ``--execution-policy=cem_primary`` (default),
  exporter executes WM/CEM first action when available, then SmolVLA action if policy is loaded,
  else heuristic fallback. ``--execution-policy=smolvla_primary`` is available as an explicit
  ablation mode.
- **Latent / CEM arm:** Whenever the WM loads, still runs CEM on the latent unroll and
  records per-step metadata (independent of which controller produced ``a_exec``).
- **CUDA:** WM+CEM runs whenever the hub model loads; ``device`` follows ``--device`` /
  availability (CPU ok for **smoke** / dev; Slurm phase07 should still use a GPU job).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import resource
import shutil
import site
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


SCHEMA_VERSION = "cem_paired_push_v3_v0"
EXPORT_MODE = "cem_paired_push_v3"
_EXECUTION_POLICIES = ("cem_primary", "smolvla_primary")


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    return text not in ("", "0", "false", "no", "off")


def _compute_export_quality_metrics(episodes: list[dict[str, Any]]) -> dict[str, float]:
    metrics_acc = ExportQualityAccumulator()
    for episode in episodes:
        metrics_acc.update(episode)
    return metrics_acc.to_metrics()


@dataclass
class ExportQualityAccumulator:
    total_episodes: int = 0
    episodes_with_images: int = 0
    heuristic_episodes: int = 0
    total_steps: int = 0
    wm_error_steps: int = 0
    policy_error_steps: int = 0

    def update(self, episode: dict[str, Any]) -> None:
        self.total_episodes += 1
        images = episode.get("images")
        if isinstance(images, list) and len(images) > 0:
            self.episodes_with_images += 1
        per_step = (
            ((episode.get("cem_plan") or {}).get("per_step") if isinstance(episode.get("cem_plan"), dict) else [])
            or []
        )
        has_heuristic = False
        for row in per_step:
            if not isinstance(row, dict):
                continue
            self.total_steps += 1
            policy_source = str(row.get("policy_source", "")).strip()
            if policy_source in {"heuristic_fallback", "heuristic"}:
                has_heuristic = True
            planner_metadata = row.get("planner_metadata") if isinstance(row.get("planner_metadata"), dict) else {}
            if isinstance(planner_metadata, dict):
                if planner_metadata.get("wm_step_error"):
                    self.wm_error_steps += 1
                if planner_metadata.get("policy_exec_error"):
                    self.policy_error_steps += 1
        if not has_heuristic:
            policy_label = str((episode.get("meta") or {}).get("policy", "")).strip() if isinstance(episode.get("meta"), dict) else ""
            has_heuristic = policy_label in {"heuristic_fallback", "heuristic"}
        if has_heuristic:
            self.heuristic_episodes += 1

    def to_metrics(self) -> dict[str, float]:
        wm_step_error_rate = float(self.wm_error_steps / self.total_steps) if self.total_steps > 0 else 0.0
        policy_exec_error_rate = float(self.policy_error_steps / self.total_steps) if self.total_steps > 0 else 0.0
        heuristic_ratio = (
            float(self.heuristic_episodes / self.total_episodes) if self.total_episodes > 0 else 0.0
        )
        return {
            "total_episodes": float(self.total_episodes),
            "episodes_with_images": float(self.episodes_with_images),
            "total_steps": float(self.total_steps),
            "wm_step_error_rate": wm_step_error_rate,
            "policy_exec_error_rate": policy_exec_error_rate,
            "heuristic_fallback_episode_ratio": heuristic_ratio,
        }


def _infer_episode_latent_pred_dim(episode: dict[str, Any]) -> int | None:
    plan = episode.get("cem_plan") if isinstance(episode, dict) else None
    per_step = plan.get("per_step") if isinstance(plan, dict) else []
    for row in per_step:
        if not isinstance(row, dict):
            continue
        dim = row.get("latent_pred_dim")
        try:
            dim_i = int(dim)
            if dim_i > 0:
                return dim_i
        except Exception:
            pass
        latent = row.get("latent_pred")
        if torch.is_tensor(latent):
            try:
                latent_dim = int(latent.detach().reshape(-1).numel())
                if latent_dim > 0:
                    return latent_dim
            except Exception:
                pass
        elif hasattr(latent, "shape"):
            latent_arr = np.asarray(latent).reshape(-1)
            if int(latent_arr.size) > 0:
                return int(latent_arr.size)
        elif isinstance(latent, list) and len(latent) > 0:
            return int(len(latent))
    return None


def _cleanup_episode_shards(path: Path) -> None:
    target = Path(path)
    if not target.exists():
        return
    if target.is_dir():
        shutil.rmtree(target, ignore_errors=True)
        return
    try:
        target.unlink()
    except Exception:
        pass


def _promote_episode_shards(staging_dir: Path, final_dir: Path) -> None:
    src = Path(staging_dir)
    dst = Path(final_dir)
    if dst.exists():
        _cleanup_episode_shards(dst)
    os.replace(src, dst)


def _enforce_export_quality_gates(
    metrics: dict[str, float],
    *,
    max_wm_error_rate: float,
    max_policy_error_rate: float,
    require_images: bool,
    max_heuristic_ratio: float,
) -> None:
    wm_error_rate = float(metrics.get("wm_step_error_rate", 0.0))
    policy_error_rate = float(metrics.get("policy_exec_error_rate", 0.0))
    episodes_with_images = int(metrics.get("episodes_with_images", 0))
    total_episodes = int(metrics.get("total_episodes", 0))
    heuristic_ratio = float(metrics.get("heuristic_fallback_episode_ratio", 0.0))
    if wm_error_rate > max_wm_error_rate:
        raise RuntimeError(f"wm_step_error_rate above threshold: {wm_error_rate:.4f} > {max_wm_error_rate:.4f}")
    if policy_error_rate > max_policy_error_rate:
        raise RuntimeError(
            f"policy_exec_error_rate above threshold: {policy_error_rate:.4f} > {max_policy_error_rate:.4f}"
        )
    if require_images and episodes_with_images < total_episodes:
        raise RuntimeError(
            f"episodes_with_images below required coverage: {episodes_with_images}/{total_episodes}"
        )
    if heuristic_ratio > max_heuristic_ratio:
        raise RuntimeError(
            "heuristic_fallback_episode_ratio above threshold: "
            f"{heuristic_ratio:.4f} > {max_heuristic_ratio:.4f}"
        )


def _current_rss_gb() -> float:
    """Best-effort current process RSS in GiB."""
    try:
        statm = Path("/proc/self/statm")
        if statm.exists():
            parts = statm.read_text(encoding="utf-8").strip().split()
            if len(parts) >= 2:
                rss_pages = int(parts[1])
                page_size = int(os.sysconf("SC_PAGE_SIZE"))
                return float(rss_pages * page_size) / float(1024**3)
    except Exception:
        pass
    try:
        rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if sys.platform != "darwin":
            rss *= 1024.0
        return rss / float(1024**3)
    except Exception:
        return 0.0


def _enforce_rss_limit(max_rss_gb: float, context: str) -> None:
    limit = float(max_rss_gb)
    if limit <= 0.0:
        return
    rss_gb = _current_rss_gb()
    if rss_gb > limit:
        raise RuntimeError(f"rss_gb above threshold at {context}: {rss_gb:.3f} > {limit:.3f}")


def _to_rgb_list(arr: Any) -> list[list[list[float]]]:
    if arr is None:
        return []
    x = np.asarray(arr)
    if x.dtype != np.float32 and x.dtype != np.float64:
        x = x.astype(np.float32) / 255.0
    return x.tolist()


def _flatten_obs_state(obs: Any) -> list[float]:
    if isinstance(obs, dict):
        for k in ("observation.state", "state", "agent_pos", "observation"):
            if k in obs:
                v = obs[k]
                return np.asarray(v, dtype=np.float32).reshape(-1).tolist()
        for v in obs.values():
            if hasattr(v, "shape"):
                return np.asarray(v, dtype=np.float32).reshape(-1).tolist()
    return np.asarray(obs, dtype=np.float32).reshape(-1).tolist()


def _find_image(obs: Any) -> Any:
    if not isinstance(obs, dict):
        return None
    for k in ("image", "top", "pixels/top", "observation.image", "rgb"):
        if k in obs:
            return obs[k]
    for v in obs.values():
        if hasattr(v, "shape") and len(getattr(v, "shape", ())) == 3:
            return v
    return None


def _as_contiguous_rgb_uint8(arr: Any) -> np.ndarray:
    x = np.asarray(arr)
    if x.ndim != 3 or x.shape[-1] not in (3, 4):
        raise RuntimeError("bad image shape")
    if x.shape[-1] == 4:
        x = x[..., :3]
    if x.dtype != np.uint8:
        if np.issubdtype(x.dtype, np.floating) and float(np.max(x)) <= 1.5:
            x = (np.clip(x, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            x = np.clip(x, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(x)


def _encode_image_payload(image: Any) -> np.ndarray:
    """Return compact image payload as contiguous HWC uint8."""
    return _as_contiguous_rgb_uint8(image)


def _encode_latent_payload(latent_vec: Any, full_latents_export: bool) -> Any:
    """Return compact latent payload as tensor/ndarray (never Python list)."""
    if torch.is_tensor(latent_vec):
        payload = latent_vec.detach().float().cpu().reshape(-1)
        return payload if full_latents_export else payload[:256]
    payload = np.asarray(latent_vec, dtype=np.float32).reshape(-1)
    return payload if full_latents_export else payload[:256]


class EpisodeShardWriter:
    """Write episode shards incrementally to bound exporter memory usage."""

    def __init__(self, out_dir: Path, episodes_per_shard: int = 1) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.episodes_per_shard = int(episodes_per_shard)
        if self.episodes_per_shard <= 0:
            raise ValueError("episodes_per_shard must be > 0")
        self._pending: list[dict[str, Any]] = []
        self._episode_index = 0
        self._written_files: list[Path] = []

    def _write_episode_file(self, episode: dict[str, Any]) -> Path:
        episode_path = self.out_dir / f"episode_{self._episode_index:06d}.pt"
        torch.save(episode, episode_path)
        self._episode_index += 1
        self._written_files.append(episode_path)
        return episode_path

    def _flush_pending(self) -> list[Path]:
        if not self._pending:
            return []
        pending = list(self._pending)
        self._pending.clear()
        return [self._write_episode_file(episode) for episode in pending]

    def write_episode(self, episode: dict[str, Any]) -> Path | None:
        self._pending.append(episode)
        if len(self._pending) >= self.episodes_per_shard:
            written = self._flush_pending()
            if written:
                return written[-1]
        return None

    def finalize(self) -> list[Path]:
        self._flush_pending()
        return list(self._written_files)


def _collect_step_image(obs: Any, env: Any) -> np.ndarray:
    img = _find_image(obs)
    if img is None:
        img = env.render()
    return _as_contiguous_rgb_uint8(img)


def _clip_action_to_env(action: np.ndarray, env_action_dim: int) -> np.ndarray:
    return np.clip(
        np.asarray(action, dtype=np.float32).reshape(-1)[:env_action_dim],
        -1.0,
        1.0,
    ).astype(np.float32)


def _select_executed_action(
    *,
    obs: Any,
    env: Any,
    action_wm_cem_first: np.ndarray | None,
    action_smolvla_raw: np.ndarray | None,
    env_action_dim: int,
    wm_available: bool,
    execution_policy: str = "cem_primary",
) -> dict[str, Any]:
    policy = str(execution_policy).strip().lower()
    if policy not in _EXECUTION_POLICIES:
        policy = "cem_primary"
    if policy == "smolvla_primary" and action_smolvla_raw is not None:
        a = _clip_action_to_env(action_smolvla_raw, env_action_dim)
        return {"action_executed": a.tolist(), "policy_source": "smolvla"}
    if action_wm_cem_first is not None:
        a = _clip_action_to_env(action_wm_cem_first, env_action_dim)
        return {"action_executed": a.tolist(), "policy_source": "cem_mpc_wm"}
    if action_smolvla_raw is not None:
        a = _clip_action_to_env(action_smolvla_raw, env_action_dim)
        return {"action_executed": a.tolist(), "policy_source": "smolvla"}
    a = _clip_action_to_env(heuristic_push_action(obs, env), env_action_dim)
    source = "heuristic_fallback" if wm_available else "heuristic"
    return {"action_executed": a.tolist(), "policy_source": source}


def _patch_external_datasets() -> None:
    """Avoid local ``lerobot.datasets`` shadowing HuggingFace ``datasets`` when importing policies."""
    for item in site.getsitepackages() + [site.getusersitepackages() or ""]:
        if not item:
            continue
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
class SmolVLAExecBundle:
    policy: Any
    preprocessor: Any
    postprocessor: Any
    device: torch.device


def _try_load_smolvla_exec(checkpoint: str, device: torch.device) -> SmolVLAExecBundle | None:
    ckpt = (checkpoint or "").strip()
    if not ckpt:
        return None
    try:
        _patch_external_datasets()
        import inspect  # noqa: PLC0415

        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # noqa: PLC0415
        from lerobot.configs.policies import PreTrainedConfig  # noqa: PLC0415
        from lerobot.processor import PolicyProcessorPipeline  # noqa: PLC0415
        from lerobot.processor.converters import (  # noqa: PLC0415
            batch_to_transition,
            policy_action_to_transition,
            transition_to_batch,
            transition_to_policy_action,
        )

        policy_dev_raw = os.environ.get("SMOLVLA_JEPA_EXPORT_POLICY_DEVICE", "").strip().lower()
        if policy_dev_raw in ("", "default"):
            dev = device
        elif policy_dev_raw == "auto":
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif policy_dev_raw == "cuda" and not torch.cuda.is_available():
            dev = torch.device("cpu")
        else:
            dev = torch.device(policy_dev_raw)
        dev_str = str(dev)
        load_vlm_raw = os.environ.get(
            "SMOLVLA_JEPA_EXPORT_POLICY_LOAD_VLM_WEIGHTS", "1"
        )
        load_vlm_weights = load_vlm_raw.strip().lower() not in ("0", "false", "no")
        print(
            f"[cem_paired_export] policy load_vlm raw='{load_vlm_raw}' parsed={load_vlm_weights}"
        )
        print(
            f"[cem_paired_export] policy device raw='{policy_dev_raw or '<default>'}' resolved='{dev_str}'"
        )
        model_kwargs: dict[str, Any] = {
            "device": dev_str,
            "n_action_steps": 1,
            "expert_width_multiplier": 0.5,
            "self_attn_every_n_layers": 0,
            "load_vlm_weights": load_vlm_weights,
            "vlm_model_name": "HuggingFaceTB/SmolVLM2-500M-Instruct",
        }
        sig = inspect.signature(SmolVLAPolicy.from_pretrained)
        params = sig.parameters
        pretrained_keys = (
            "pretrained_name_or_path",
            "pretrained_model_name_or_path",
            "pretrained_path",
        )
        pretrained_key = next((k for k in pretrained_keys if k in params), None)
        accepts_var_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
        if accepts_var_kwargs:
            supported_kwargs = dict(model_kwargs)
        else:
            supported_kwargs = {k: v for k, v in model_kwargs.items() if k in params}
        config_override_applied = False
        if "config" in params:
            try:
                policy_cfg = PreTrainedConfig.from_pretrained(pretrained_name_or_path=ckpt)
                for key, value in model_kwargs.items():
                    if hasattr(policy_cfg, key):
                        setattr(policy_cfg, key, value)
                supported_kwargs["config"] = policy_cfg
                config_override_applied = True
                print(
                    "[cem_paired_export] policy config override:"
                    f" load_vlm_weights={getattr(policy_cfg, 'load_vlm_weights', None)}"
                    f" device={getattr(policy_cfg, 'device', None)}"
                )
            except Exception as exc:
                print(f"[cem_paired_export] policy config override skipped: {exc}")
        print(f"[cem_paired_export] policy config override applied={config_override_applied}")
        overrides = {"device_processor": {"device": dev_str}}
        preprocessor = PolicyProcessorPipeline.from_pretrained(
            pretrained_model_name_or_path=ckpt,
            config_filename="policy_preprocessor.json",
            overrides=overrides,
            to_transition=batch_to_transition,
            to_output=transition_to_batch,
        )
        postprocessor = PolicyProcessorPipeline.from_pretrained(
            pretrained_model_name_or_path=ckpt,
            config_filename="policy_postprocessor.json",
            overrides=overrides,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        )
        if pretrained_key:
            supported_kwargs[pretrained_key] = ckpt
            policy = SmolVLAPolicy.from_pretrained(**supported_kwargs)
        else:
            policy = SmolVLAPolicy.from_pretrained(ckpt, **supported_kwargs)
        print(
            "[cem_paired_export] loaded policy config:"
            f" load_vlm_weights={getattr(getattr(policy, 'config', None), 'load_vlm_weights', None)}"
        )
        policy.eval()
        return SmolVLAExecBundle(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            device=dev,
        )
    except Exception:
        return None


def _smolvla_state_dims(policy: Any) -> tuple[int, int]:
    """Return (agent_state_dim, environment_state_dim) from policy input features with MetaWorld-ish defaults."""
    agent_dim, env_dim = 4, 39
    feats = getattr(getattr(policy, "config", None), "input_features", None) or {}
    for name, ft in feats.items():
        sh = getattr(ft, "shape", None) or ()
        dim0 = int(sh[0]) if sh else 0
        if not dim0:
            continue
        if "environment_state" in name:
            env_dim = dim0
        elif name.endswith(".state") and "environment" not in name:
            agent_dim = dim0
    return agent_dim, env_dim


def _vectors_for_smolvla(flat: np.ndarray, agent_dim: int, env_dim: int) -> tuple[np.ndarray, np.ndarray]:
    flat = np.asarray(flat, dtype=np.float32).reshape(-1)
    env_vec = np.zeros(env_dim, dtype=np.float32)
    env_vec[: min(env_dim, flat.size)] = flat[: min(env_dim, flat.size)]
    agent_vec = np.zeros(agent_dim, dtype=np.float32)
    agent_vec[:] = env_vec[:agent_dim]
    return agent_vec, env_vec


def _policy_rgb_hwc(env: Any, obs: Any) -> np.ndarray:
    return _collect_step_image(obs, env)


def _smolvla_exec_action(
    bundle: SmolVLAExecBundle,
    obs: Any,
    env: Any,
    task_text: str,
) -> np.ndarray:
    from lerobot.utils.constants import (  # noqa: PLC0415
        OBS_ENV_STATE,
        OBS_IMAGE,
        OBS_STATE,
    )

    agent_dim, env_dim = _smolvla_state_dims(bundle.policy)
    if isinstance(obs, np.ndarray):
        flat = np.asarray(obs, dtype=np.float32).reshape(-1)
    else:
        flat = np.asarray(_flatten_obs_state(obs), dtype=np.float32).reshape(-1)
    if flat.size == 0:
        raise RuntimeError("empty state vector")
    _agent_vec, env_vec = _vectors_for_smolvla(flat, agent_dim, env_dim)
    rgb = _policy_rgb_hwc(env, obs)
    timg = torch.from_numpy(rgb).unsqueeze(0).permute(0, 3, 1, 2).contiguous().float() / 255.0
    timg = timg.to(bundle.device)
    st = torch.from_numpy(_agent_vec).unsqueeze(0).to(bundle.device)
    es = torch.from_numpy(env_vec).unsqueeze(0).to(bundle.device)
    batch = {
        OBS_IMAGE: timg,
        OBS_STATE: st,
        OBS_ENV_STATE: es,
        "task": task_text,
    }
    proc = bundle.preprocessor(batch)
    with torch.inference_mode():
        act = bundle.policy.select_action(proc)
    act = bundle.postprocessor(act)
    out = act.detach().float().cpu().numpy().reshape(-1)
    return out


def heuristic_push_action(obs: Any, env: Any) -> np.ndarray:
    """Non-random push-v3 prior: move in the direction object -> goal in proprio slice."""
    adim = int(np.prod(env.action_space.shape))
    if isinstance(obs, dict):
        flat = _flatten_obs_state(obs)
        o = np.asarray(flat, dtype=np.float32).reshape(-1)
    else:
        o = np.asarray(obs, dtype=np.float32).reshape(-1)
    if o.size < 12:
        return np.zeros(adim, dtype=np.float32)
    # Typical MT1 Sawyer layouts: gripper(4) object(3) ... goal(3) at end
    obj = o[4:7] if o.size > 10 else o[:3]
    goal = o[-3:] if o.size >= 15 else obj
    delta = goal - obj
    n = float(np.linalg.norm(delta) + 1e-6)
    planar = (delta / n) * 2.0
    vec = np.zeros(adim, dtype=np.float32)
    m = min(adim, int(planar.shape[0]))
    vec[:m] = np.clip(planar[:m], -1.0, 1.0)
    return vec


def _render_to_wm_visual(env: Any, obs: Any, device: torch.device) -> torch.Tensor | None:
    """Return visual tensor (1,1,3,256,256) float, or None.

    Prefer pixels already in ``obs`` (avoids an extra ``env.render()`` when available).
    """
    try:
        arr = _collect_step_image(obs, env)
    except Exception:
        return None
    t = torch.from_numpy(arr).float() / 255.0
    t = t.permute(2, 0, 1).unsqueeze(0)  # 1,3,H,W
    t = torch.nn.functional.interpolate(t, size=(256, 256), mode="bilinear", align_corners=False)
    return t.unsqueeze(0).to(device)  # 1,1,3,256,256


def _build_proprio(flat_state: list[float], proprio_dim: int, device: torch.device) -> torch.Tensor:
    v = np.asarray(flat_state, dtype=np.float32).reshape(-1)
    if v.size >= proprio_dim:
        p = v[:proprio_dim].copy()
    else:
        p = np.zeros(proprio_dim, dtype=np.float32)
        p[: v.size] = v
    t = torch.from_numpy(p).float().view(1, 1, -1).to(device)
    return t


def _resolve_ckpt(ckpt_hint: str) -> str:
    if not ckpt_hint:
        return "jepa_wm_metaworld.pth.tar"
    maybe = Path(ckpt_hint)
    if maybe.is_file():
        return str(maybe.resolve())
    hf_home = Path.home() / ".cache" / "huggingface" / "hub"
    if hf_home.exists():
        matches = sorted(hf_home.rglob(ckpt_hint))
        if matches:
            return str(matches[0].resolve())
    return ckpt_hint


def _try_load_wm(repo: Path, ckpt: str, device: torch.device) -> tuple[Any, Any] | None:
    try:
        import os

        if not os.environ.get("JEPAWM_LOGS"):
            os.environ["JEPAWM_LOGS"] = str((Path.home() / ".cache" / "jepa_wm").resolve())
        os.environ["JEPAWM_CKPT"] = _resolve_ckpt(ckpt)
        if repo.is_dir():
            model, preprocessor = torch.hub.load(
                str(repo), "jepa_wm_metaworld", source="local", pretrained=True, device=str(device)
            )
        else:
            model, preprocessor = torch.hub.load(
                str(repo), "jepa_wm_metaworld", source="github", pretrained=True, device=str(device)
            )
        model.eval()
        return model, preprocessor
    except Exception:
        return None


def _infer_action_dims(model: Any, preprocessor: Any) -> list[int]:
    """Match ``jepa_smoke_check._infer_action_dims``: WM may accept 20-D actions while env is 4-D."""
    dims: list[int] = []

    def _add(value: object) -> None:
        try:
            dim = int(value)
        except Exception:
            return
        if dim > 0 and dim not in dims:
            dims.append(dim)

    _add(getattr(getattr(preprocessor, "action_mean", None), "numel", lambda: 0)() or 0)
    model_module = getattr(model, "model", None)
    if model_module is not None:
        _add(getattr(model_module, "action_dim", 0) or 0)
        _add(getattr(getattr(model_module, "action_encoder", None), "in_features", 0) or 0)
        predictor = getattr(model_module, "predictor", None)
        if predictor is not None:
            _add(getattr(getattr(predictor, "action_encoder", None), "in_features", 0) or 0)
    if not dims:
        dims = [4]
    return dims


def _score_unroll(z_pred: Any) -> float:
    """Higher is better (CEM maximizes)."""
    try:
        if isinstance(z_pred, dict):
            lat = z_pred.get("latent")
            if lat is None:
                lat = next(iter(z_pred.values()))
        else:
            lat = z_pred
        if not torch.is_tensor(lat):
            return 0.0
        return float(-lat.pow(2).mean().item())
    except Exception:
        return 0.0


def cem_first_action(
    model: Any,
    z: torch.Tensor,
    action_dim: int,
    horizon: int,
    pop_size: int,
    cem_iters: int,
    device: torch.device,
    rng: np.random.Generator,
    full_latents_export: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return first action of best CEM sequence and debug metadata."""
    best_seq = torch.zeros(horizon, action_dim, device=device, dtype=torch.float32)
    best_score = -1e18
    best_z_pred: Any = None
    successful_unrolls = 0
    for _ in range(cem_iters):
        for _p in range(pop_size):
            noise = torch.randn(horizon, action_dim, device=device, dtype=torch.float32) * 0.35
            seq = torch.clamp(best_seq + noise, -1.0, 1.0)
            try:
                act_suffix = seq.unsqueeze(1)
                z_pred = model.unroll(z, act_suffix=act_suffix, debug=False)
                successful_unrolls += 1
                sc = _score_unroll(z_pred)
                if sc > best_score:
                    best_score = sc
                    best_seq = seq.detach().clone()
                    best_z_pred = z_pred
            except Exception:
                continue
    if successful_unrolls <= 0 or best_z_pred is None:
        raise RuntimeError("cem_unroll_failed_all_candidates")
    meta = {
        "cem_iterations": int(cem_iters * pop_size),
        "cem_cost": float(-best_score),
        "cem_seed": int(rng.integers(0, 2**31 - 1)),
        "cem_horizon": horizon,
        "cem_population": pop_size,
    }
    a0 = best_seq[0].detach().cpu().numpy().reshape(-1)
    latent_summary: Any = []
    latent_pred_dim = 0
    if best_z_pred is not None:
        try:
            if isinstance(best_z_pred, dict):
                lat = best_z_pred.get("latent", list(best_z_pred.values())[0])
            else:
                lat = best_z_pred
            if torch.is_tensor(lat):
                latent_pred_dim = int(lat.detach().reshape(-1).numel())
            else:
                latent_pred_dim = int(np.asarray(lat).reshape(-1).size)
            if latent_pred_dim > 0:
                latent_summary = _encode_latent_payload(lat, full_latents_export=full_latents_export)
        except Exception:
            pass
    return a0, {"meta": meta, "latent_pred": latent_summary, "latent_pred_dim": latent_pred_dim}


def rollout_episode(
    env: Any,
    max_steps: int,
    pair_key: str,
    wm_bundle: tuple[Any, Any, int, int, torch.device] | None,
    smolvla_bundle: SmolVLAExecBundle | None,
    task_text: str,
    cem_horizon: int,
    cem_pop: int,
    cem_iters: int,
    execution_policy: str,
    store_cem_plan_seq: bool,
    store_smolvla_action: bool,
    full_latents_export: bool,
    rng: np.random.Generator,
    max_rss_gb: float = 0.0,
    rss_log_interval_steps: int = 0,
    episode_index: int | None = None,
) -> dict[str, Any]:
    seed = int(rng.integers(0, 2**31 - 1))
    try:
        out = env.reset(seed=seed)
    except TypeError:
        out = env.reset()
    if isinstance(out, tuple):
        obs = out[0]
        info = out[1] if len(out) > 1 and isinstance(out[1], dict) else {}
    else:
        obs, info = out, {}

    if smolvla_bundle is not None:
        smolvla_bundle.policy.reset()
        smolvla_bundle.preprocessor.reset()
        smolvla_bundle.postprocessor.reset()

    images: list[Any] = []
    states: list[list[float]] = []
    actions: list[list[float]] = []
    cem_steps: list[dict[str, Any]] = []

    model = preproc = proprio_dim = None
    device = torch.device("cpu")
    env_action_dim = int(np.prod(env.action_space.shape))
    planner_action_dim = env_action_dim
    if wm_bundle is not None:
        model, preproc, proprio_dim, planner_action_dim, device = wm_bundle

    policy_used = "heuristic"

    for step_idx in range(max_steps):
        rss_context = f"ep={episode_index if episode_index is not None else '?'} step={step_idx}"
        if int(rss_log_interval_steps) > 0 and (step_idx % int(rss_log_interval_steps) == 0):
            print(f"[cem_paired_export][rss] {rss_context} rss_gb={_current_rss_gb():.3f}")
        _enforce_rss_limit(max_rss_gb=max_rss_gb, context=f"{rss_context} pre_step")

        try:
            images.append(_encode_image_payload(_collect_step_image(obs, env)))
        except Exception:
            pass
        st = _flatten_obs_state(obs)
        states.append(st)

        step_record: dict[str, Any] = {
            "step_index": step_idx,
            "cem_iterations": 0,
            "cem_cost": 0.0,
            "cem_seed": seed,
            "latent_pred": [],
            "planner_metadata": {
                "planner_action_dim": int(planner_action_dim),
                "env_action_dim": int(env_action_dim),
            },
        }

        a_cem: np.ndarray | None = None
        if model is not None and preproc is not None:
            try:
                vis = _render_to_wm_visual(env, obs, device)
                if vis is None:
                    raise RuntimeError("no render")
                pr = _build_proprio(st, proprio_dim, device)
                obs_wm = {"visual": vis, "proprio": pr}
                with torch.no_grad():
                    z = model.encode(obs_wm)
                z = z.to(device)
                a_cem, cem_dbg = cem_first_action(
                    model,
                    z,
                    planner_action_dim,
                    cem_horizon,
                    cem_pop,
                    cem_iters,
                    device,
                    rng,
                    full_latents_export=full_latents_export,
                )
                meta = dict(step_record.get("planner_metadata") or {})
                meta.update(
                    {
                        "horizon": cem_horizon,
                        "population": cem_pop,
                        "latent_pred_dim": int(cem_dbg.get("latent_pred_dim", 0)),
                    }
                )
                step_record.update(
                    {
                        "cem_iterations": cem_dbg["meta"]["cem_iterations"],
                        "cem_cost": cem_dbg["meta"]["cem_cost"],
                        "cem_seed": cem_dbg["meta"]["cem_seed"],
                        "latent_pred": _encode_latent_payload(
                            cem_dbg.get("latent_pred", []),
                            full_latents_export=full_latents_export,
                        ),
                        "latent_pred_dim": int(cem_dbg.get("latent_pred_dim", 0)),
                        "planner_metadata": meta,
                    }
                )
            except Exception as exc:
                meta = dict(step_record.get("planner_metadata") or {})
                meta["wm_step_error"] = str(exc)[:200]
                step_record["planner_metadata"] = meta
        elif wm_bundle is None:
            step_record["planner_metadata"] = {"wm_skipped": True}

        a_smolvla: np.ndarray | None = None
        if smolvla_bundle is not None:
            try:
                raw_act = _smolvla_exec_action(smolvla_bundle, obs, env, task_text)
                a_smolvla = np.asarray(raw_act, dtype=np.float32).reshape(-1)
                if store_smolvla_action:
                    step_record["action_smolvla_raw"] = a_smolvla.tolist()
            except Exception as exc:
                meta = dict(step_record.get("planner_metadata") or {})
                meta["policy_exec_error"] = str(exc)[:200]
                step_record["planner_metadata"] = meta

        action_pick = _select_executed_action(
            obs=obs,
            env=env,
            action_wm_cem_first=a_cem,
            action_smolvla_raw=a_smolvla,
            env_action_dim=env_action_dim,
            wm_available=model is not None,
            execution_policy=execution_policy,
        )
        a_exec = np.asarray(action_pick["action_executed"], dtype=np.float32).reshape(-1)
        policy_used = str(action_pick["policy_source"])
        step_record["policy_source"] = policy_used

        a_list = np.asarray(a_exec, dtype=np.float32).reshape(-1).tolist()
        actions.append(a_list)
        cem_steps.append(step_record)

        step_out = env.step(a_exec)
        if len(step_out) == 5:
            obs, reward, terminated, truncated, info = step_out
            done = bool(terminated or truncated)
        else:
            obs, reward, done, info = step_out

        _enforce_rss_limit(max_rss_gb=max_rss_gb, context=f"{rss_context} post_step")
        if done:
            break

    success = bool(info.get("success", False)) if isinstance(info, dict) else False
    return {
        "images": images,
        "state": states,
        "actions": actions,
        "action_chunk": actions,
        "language": "push the puck to the goal",
        "done": True,
        "success": success,
        "pair_key": pair_key,
        "cem_plan": {"per_step": cem_steps if store_cem_plan_seq else []},
        "meta": {
            "schema_version": SCHEMA_VERSION,
            "pair_key": pair_key,
            "export_mode": EXPORT_MODE,
            "task": "push-v3",
            "steps": len(actions),
            "policy": policy_used,
            "pairing": "executed_latent_aligned",
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="push-v3")
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument(
        "--episodes-per-shard",
        type=int,
        default=int(os.environ.get("SMOLVLA_JEPA_EXPORT_EPISODES_PER_SHARD", "1")),
        help="Number of episodes buffered before shard flush (must be > 0).",
    )
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--jepa-repo", type=Path, default=None, help="Path to local jepa-wms repo (with hubconf)")
    ap.add_argument("--jepa-ckpt", type=str, default="")
    ap.add_argument("--cem-horizon", type=int, default=4)
    ap.add_argument("--cem-pop", type=int, default=8)
    ap.add_argument("--cem-iters", type=int, default=2)
    ap.add_argument(
        "--execution-policy",
        choices=list(_EXECUTION_POLICIES),
        default=os.environ.get("SMOLVLA_JEPA_EXPORT_EXECUTION_POLICY", "cem_primary"),
        help=(
            "Executed-action arbitration. Default 'cem_primary' enforces WM/CEM-first, "
            "with SmolVLA then heuristic fallback; 'smolvla_primary' is optional ablation."
        ),
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--max-wm-error-rate",
        type=float,
        default=float(os.environ.get("SMOLVLA_JEPA_EXPORT_MAX_WM_ERROR_RATE", "0.05")),
    )
    ap.add_argument(
        "--max-policy-error-rate",
        type=float,
        default=float(os.environ.get("SMOLVLA_JEPA_EXPORT_MAX_POLICY_ERROR_RATE", "0.05")),
    )
    ap.add_argument("--max-heuristic-fallback-episode-ratio", type=float, default=0.10)
    ap.add_argument(
        "--require-images",
        type=int,
        default=int(os.environ.get("SMOLVLA_JEPA_EXPORT_REQUIRE_IMAGES", "1")),
        help="Require every episode to contain at least one image frame (1=yes,0=no).",
    )
    ap.add_argument(
        "--store-cem-plan-seq",
        type=int,
        default=int(os.environ.get("SMOLVLA_JEPA_EXPORT_STORE_CEM_PLAN_SEQ", "1")),
        help="Store per-step CEM planner records in trajectories (1=yes,0=no).",
    )
    ap.add_argument(
        "--store-smolvla-action",
        type=int,
        default=int(os.environ.get("SMOLVLA_JEPA_EXPORT_STORE_SMOLVLA_ACTION", "1")),
        help="Store per-step raw SmolVLA action vectors when available (1=yes,0=no).",
    )
    ap.add_argument(
        "--full-latents-export",
        type=int,
        default=int(os.environ.get("SMOLVLA_JEPA_EXPORT_FULL_LATENTS", "1")),
        help="Store full latent vectors in per-step telemetry (1=yes,0=truncate to summary).",
    )
    ap.add_argument(
        "--policy-checkpoint",
        default=os.environ.get("SMOLVLA_INIT_CHECKPOINT", ""),
        help="SmolVLA HF id or local dir; empty disables. Default: $SMOLVLA_INIT_CHECKPOINT.",
    )
    ap.add_argument(
        "--max-rss-gb",
        type=float,
        default=float(os.environ.get("SMOLVLA_JEPA_EXPORT_MAX_RSS_GB", "0")),
        help="Abort rollout when process RSS exceeds this GiB limit (0 disables).",
    )
    ap.add_argument(
        "--rss-log-interval-steps",
        type=int,
        default=int(os.environ.get("SMOLVLA_JEPA_EXPORT_RSS_LOG_INTERVAL_STEPS", "25")),
        help="Log RSS every N rollout steps (<=0 disables).",
    )
    args = ap.parse_args()
    if int(args.episodes_per_shard) <= 0:
        ap.error("--episodes-per-shard must be > 0")
    store_cem_plan_seq = _as_bool(args.store_cem_plan_seq)
    store_smolvla_action = _as_bool(args.store_smolvla_action)
    full_latents_export = _as_bool(args.full_latents_export)

    dev_name = args.device
    if dev_name == "auto":
        dev_name = "cuda" if torch.cuda.is_available() else "cpu"
    if dev_name == "cuda" and not torch.cuda.is_available():
        dev_name = "cpu"
    dev = torch.device(dev_name)

    policy_ckpt = (args.policy_checkpoint or "").strip()
    smolvla_bundle = _try_load_smolvla_exec(policy_ckpt, dev) if policy_ckpt else None
    task_text = "push the puck to the goal"

    wm_bundle: tuple[Any, Any, int, int, torch.device] | None = None
    repo = args.jepa_repo
    skip_wm = os.environ.get("SMOLVLA_JEPA_EXPORT_SKIP_WM", "").strip().lower() in ("1", "true", "yes")
    if skip_wm:
        print("[cem_paired_export] SMOLVLA_JEPA_EXPORT_SKIP_WM=1: skipping WM hub load (heuristic / env-only rollouts)")
    elif repo is not None and repo.is_dir():
        loaded = _try_load_wm(repo, args.jepa_ckpt or "jepa_wm_metaworld.pth.tar", dev)
        if loaded is not None:
            model, preprocessor = loaded
            proprio_dim = int(getattr(preprocessor, "proprio_mean").numel())
            action_dims = _infer_action_dims(model, preprocessor)
            planner_action_dim = max(action_dims)
            print(
                f"[cem_paired_export] wm action_dim candidates={action_dims}, "
                f"planner_action_dim={planner_action_dim} (env uses 4-D MT1 actions)"
            )
            wm_bundle = (model, preprocessor, proprio_dim, planner_action_dim, dev)

    import metaworld  # noqa: PLC0415

    ml1 = metaworld.ML1(args.task, seed=int(args.seed))
    env_cls = ml1.train_classes[args.task]
    env = env_cls()
    try:
        if hasattr(env, "render_mode"):
            env.render_mode = "rgb_array"
    except Exception:
        pass
    tasks = getattr(ml1, "train_tasks", None)
    if tasks:
        env.set_task(tasks[0])

    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    episodes_dir = args.out / "episodes"
    staging_episodes_dir = args.out / f".episodes_staging_{uuid.uuid4().hex}"
    episode_writer = EpisodeShardWriter(staging_episodes_dir, episodes_per_shard=int(args.episodes_per_shard))
    metrics_acc = ExportQualityAccumulator()
    latent_pred_dim: int | None = None
    written_episode_files: list[Path] = []
    quality_metrics: dict[str, float] = {}
    try:
        try:
            for ep in range(args.episodes):
                pk = str(uuid.uuid4())
                episode = rollout_episode(
                    env,
                    args.max_steps,
                    pk,
                    wm_bundle,
                    smolvla_bundle,
                    task_text,
                    args.cem_horizon,
                    args.cem_pop,
                    args.cem_iters,
                    args.execution_policy,
                    store_cem_plan_seq,
                    store_smolvla_action,
                    full_latents_export,
                    rng,
                    max_rss_gb=float(args.max_rss_gb),
                    rss_log_interval_steps=int(args.rss_log_interval_steps),
                    episode_index=ep,
                )
                episode_writer.write_episode(episode)
                metrics_acc.update(episode)
                if latent_pred_dim is None:
                    latent_pred_dim = _infer_episode_latent_pred_dim(episode)
                del episode
            written_episode_files = episode_writer.finalize()
            quality_metrics = metrics_acc.to_metrics()
        except RuntimeError as exc:
            _cleanup_episode_shards(staging_episodes_dir)
            print(f"[cem_paired_export] episode generation failed: {exc}")
            return 1
        except Exception as exc:
            _cleanup_episode_shards(staging_episodes_dir)
            print(f"[cem_paired_export] unexpected episode generation failure: {exc}")
            return 1

        try:
            _enforce_export_quality_gates(
                quality_metrics,
                max_wm_error_rate=float(args.max_wm_error_rate),
                max_policy_error_rate=float(args.max_policy_error_rate),
                require_images=_as_bool(args.require_images),
                max_heuristic_ratio=float(args.max_heuristic_fallback_episode_ratio),
            )
        except RuntimeError as exc:
            _cleanup_episode_shards(staging_episodes_dir)
            print(f"[cem_paired_export] quality gate failed: {exc}")
            return 1

        try:
            _promote_episode_shards(staging_episodes_dir, episodes_dir)
        except Exception as exc:
            _cleanup_episode_shards(staging_episodes_dir)
            print(f"[cem_paired_export] failed to promote episode shards: {exc}")
            return 1
    finally:
        try:
            env.close()
        except Exception:
            pass

    # Compute shard metadata from final promoted output paths.
    shard_paths = sorted(path for path in episodes_dir.glob("episode_*.pt") if path.is_file())
    shard_files = [str(path.relative_to(args.out)) for path in shard_paths]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "export_mode": EXPORT_MODE,
        "trajectories_file": "episodes",
        "trajectories_format": "pt_per_episode",
        "trajectories_glob": "episodes/episode_*.pt",
        "episodes_per_shard": int(args.episodes_per_shard),
        "shard_count": len(shard_files),
        "shard_files": shard_files,
        "complete_episodes": len(shard_files),
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "task_id": args.task,
        "jepa_ckpt": args.jepa_ckpt or _resolve_ckpt("jepa_wm_metaworld.pth.tar"),
        "pairing": "executed_latent_aligned",
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "cem_horizon": args.cem_horizon,
        "cem_pop": args.cem_pop,
        "cem_iters": args.cem_iters,
        "wm_loaded": wm_bundle is not None,
        "wm_planner_action_dim": (wm_bundle[3] if wm_bundle is not None else None),
        "wm_skipped_export": bool(skip_wm),
        "policy_checkpoint": policy_ckpt or None,
        "policy_loaded": smolvla_bundle is not None,
        "execution_policy": args.execution_policy,
        "store_cem_plan_seq": store_cem_plan_seq,
        "store_smolvla_action": store_smolvla_action,
        "full_latents_exported": full_latents_export,
        "latent_pred_dim": latent_pred_dim,
        "policy_load_vlm_weights": os.environ.get("SMOLVLA_JEPA_EXPORT_POLICY_LOAD_VLM_WEIGHTS", "1"),
        "bridge_hint": "Point SMOLVLA_JEPA_SOURCE at this directory for phase08.",
        "quality_metrics": quality_metrics,
        "quality_thresholds": {
            "max_wm_error_rate": float(args.max_wm_error_rate),
            "max_policy_error_rate": float(args.max_policy_error_rate),
            "max_heuristic_fallback_episode_ratio": float(args.max_heuristic_fallback_episode_ratio),
            "require_images": _as_bool(args.require_images),
        },
        "storage_options": {
            "store_cem_plan_seq": store_cem_plan_seq,
            "store_smolvla_action": store_smolvla_action,
            "full_latents_exported": full_latents_export,
        },
    }
    (args.out / "export_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[cem_paired_export] wrote {len(written_episode_files)} episodes -> {episodes_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
