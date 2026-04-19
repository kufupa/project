from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os
import logging
from typing import Any, Literal

import importlib.util
import json
from typing import Sequence

import numpy as np

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None
    nn = None

logger = logging.getLogger(__name__)


def _require_torch(message: str) -> None:
    if torch is None:
        raise RuntimeError(
            f"{message} Install PyTorch to use this path. The segment loop uses torch for model encoding/scoring and adapter updates."
        )



CarryMode = Literal["sim", "replay"]


@dataclass
class SegmentState:
    """State at the start of a rollout segment."""

    step_index: int
    image: np.ndarray
    proprio: np.ndarray
    done: bool = False
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChunkCandidate:
    """Candidate chunk sampled for one segment."""

    index: int
    actions: np.ndarray
    score: float
    latent_distance: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "score": float(self.score),
            "latent_distance": None if self.latent_distance is None else float(self.latent_distance),
            "meta": dict(self.meta),
            "actions": np.asarray(self.actions, dtype=np.float32).tolist(),
        }


@dataclass
class ScoreTrace:
    """Per-step score vectors used for chunk scoring."""

    step_vectors: list[np.ndarray]
    final_vector: np.ndarray
    """Scoring latent at encode (before first WM unroll step), for Δ vs pre-rollout distance."""
    initial_vector: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    selected_candidate_index: int | None = None


@dataclass
class DecodeTrace:
    """Structured latent trace preserved for JEPA visual decoding."""

    visual_latents: list[np.ndarray]
    proprio_latents: list[np.ndarray] = field(default_factory=list)
    selected_candidate_index: int | None = None
    env_steps_per_wm_step: int = 1


@dataclass
class SegmentLog:
    """Per-segment rollout metadata."""

    segment_index: int
    start_step: int
    selected_index: int
    selected_score: float
    latent_distance: float | None
    carried_steps: int
    carry_mode: CarryMode
    done: bool = False
    candidates: list[ChunkCandidate] = field(default_factory=list)
    executed_actions: list[list[float]] = field(default_factory=list)
    comparison_strip_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_index": self.segment_index,
            "start_step": self.start_step,
            "selected_index": self.selected_index,
            "selected_score": float(self.selected_score),
            "latent_distance": None if self.latent_distance is None else float(self.latent_distance),
            "carried_steps": self.carried_steps,
            "carry_mode": self.carry_mode,
            "done": self.done,
            "candidates": [c.to_dict() for c in self.candidates],
            "executed_actions": [list(map(float, a)) for a in self.executed_actions],
            "comparison_strip_path": self.comparison_strip_path,
            "metadata": dict(self.metadata),
        }


@dataclass
class EpisodeLog:
    """Episode-level rollout log for JSON artifacts."""

    episode_index: int
    task: str
    carry_mode: CarryMode
    chunk_len: int
    num_candidates: int
    max_steps: int
    actions: list[list[float]] = field(default_factory=list)
    latent_scores: list[float] = field(default_factory=list)
    selected_scores: list[float] = field(default_factory=list)
    selected_indices: list[int] = field(default_factory=list)
    goal_frame_index: int | None = None
    goal_source: str | None = None
    start_frame_similarity: float | None = None
    reset_frame_warning: bool = False
    selected_candidate_indices: list[int] = field(default_factory=list)
    candidate_distances: list[float] = field(default_factory=list)
    comparison_strip_path: str | None = None
    comparison_video_path: str | None = None
    segments: list[SegmentLog] = field(default_factory=list)
    done: bool = False
    steps: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_index": self.episode_index,
            "task": self.task,
            "carry_mode": self.carry_mode,
            "chunk_len": self.chunk_len,
            "num_candidates": self.num_candidates,
            "max_steps": self.max_steps,
            "steps": self.steps,
            "done": self.done,
            "actions": [list(map(float, a)) for a in self.actions],
            "latent_scores": [float(x) for x in self.latent_scores],
            "selected_scores": [float(x) for x in self.selected_scores],
            "selected_indices": [int(i) for i in self.selected_indices],
            "selected_candidate_indices": [int(i) for i in self.selected_candidate_indices],
            "goal_frame_index": self.goal_frame_index,
            "goal_source": self.goal_source,
            "start_frame_similarity": self.start_frame_similarity,
            "reset_frame_warning": bool(self.reset_frame_warning),
            "candidate_distances": [float(x) for x in self.candidate_distances],
            "comparison_strip_path": self.comparison_strip_path,
            "comparison_video_path": self.comparison_video_path,
            "segments": [s.to_dict() for s in self.segments],
            "metadata": dict(self.metadata),
        }


@dataclass
class WMBundle:
    model: Any
    preprocessor: Any
    proprio_dim: int
    planner_action_dim: int
    device: torch.device


_HELPER_MODULE: Any | None = None
_log = logging.getLogger(__name__)


def _load_jepa_helper_module() -> Any:
    """Load copied helper module containing existing SmolVLA/JEPA-WM helpers."""
    global _HELPER_MODULE
    if _HELPER_MODULE is not None:
        return _HELPER_MODULE

    module_path = Path(__file__).resolve().parents[1] / "vendor" / "pi05" / "jepa_cem_paired_pushv3_export.py"
    if not module_path.exists():
        raise RuntimeError(
            f"Missing helper module at {module_path}. "
            "Expected copied file vendor/pi05/jepa_cem_paired_pushv3_export.py."
        )

    module_name = "jepa_cem_paired_pushv3_export"
    try:
        loader = importlib.machinery.SourceFileLoader(module_name, str(module_path))
        module = loader.load_module(module_name)
    except Exception as exc:
        # Fallback to explicit spec-based import if runtime complains.
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to load helper spec from {module_path}") from exc
        module = importlib.util.module_from_spec(spec)
        import sys

        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    _HELPER_MODULE = module
    return module


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if torch is None:
        return "cpu"
    requested = (str(device) if device is not None else "cpu").lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def _to_rgb_uint8(image: Any) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[-1] not in (3, 4):
        raise RuntimeError(f"Observation image must be HWC with 3/4 channels, got shape {arr.shape}")
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr_f = np.asarray(arr)
        if np.issubdtype(arr_f.dtype, np.floating):
            if float(np.max(arr_f)) <= 1.5:
                arr = (np.clip(arr_f, 0.0, 1.0) * 255.0).astype(np.uint8)
            else:
                arr = np.clip(arr_f, 0.0, 255.0).astype(np.uint8)
        else:
            arr = np.clip(arr_f, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _to_wm_visual(image: Any, device: torch.device) -> torch.Tensor:
    _require_torch("WM visual conversion requires torch.")
    rgb = _to_rgb_uint8(image)
    if not rgb.flags.writeable:
        rgb = rgb.copy()
    # JEPA hub EncPredWM.encode divides by 255 once; feed float RGB in [0, 255], not [0, 1].
    tensor = torch.from_numpy(rgb).float()
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # 1,3,H,W
    tensor = torch.nn.functional.interpolate(
        tensor, size=(256, 256), mode="bilinear", align_corners=False
    )  # 1,3,256,256
    return tensor.unsqueeze(0).to(device)  # 1,1,3,256,256


def _derive_policy_rgb_for_smolvla(
    wm_or_env_rgb: Any,
    *,
    jepa_parity_sim: bool,
    policy_hflip_corner2: bool,
) -> np.ndarray:
    """Map jepa-parity V-only RGB to evaluator-style corner2 pixels (adds H-flip).

    best_video uses ``np.flip(frame, (0, 1))`` on raw corner2; jepa stream is
    V-flip only, so H-flip on ``wm_image`` matches the policy training view.
    """
    rgb = _to_rgb_uint8(wm_or_env_rgb)
    if jepa_parity_sim and policy_hflip_corner2:
        return np.ascontiguousarray(np.flip(rgb, axis=1))
    return rgb


def _prepare_goal_image_for_wm(image: Any, *, flip_horizontal: bool) -> np.ndarray:
    """Oracle PNGs are V+H vs raw; jepa-wms live RGB is V-only → H-flip goal for WM encode when enabled."""
    rgb = _to_rgb_uint8(image)
    if not flip_horizontal:
        return rgb
    return np.ascontiguousarray(np.flip(rgb, axis=1))


def _write_wm_goal_encode_debug(rgb: np.ndarray, path: Path | str) -> None:
    """Persist uint8 HWC exactly as passed into WM encode preprocessing (after goal H-flip when on)."""
    try:
        import imageio.v2 as imageio
    except Exception as exc:
        logger.warning("wm_goal_for_encode.png skipped (imageio unavailable): %s", exc)
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(str(out), np.asarray(rgb, dtype=np.uint8))


def _to_wm_proprio(proprio: Any, proprio_dim: int, device: torch.device) -> torch.Tensor:
    _require_torch("WM proprio conversion requires torch.")
    vec = np.asarray(proprio, dtype=np.float32).reshape(-1)
    if vec.size >= proprio_dim:
        body = vec[:proprio_dim].copy()
    else:
        body = np.zeros(int(proprio_dim), dtype=np.float32)
        body[: int(vec.size)] = vec
    return torch.from_numpy(body).float().view(1, 1, -1).to(device)


def _pad_or_truncate(vec: np.ndarray, dim: int) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    if arr.size >= dim:
        return arr[:dim].copy()
    out = np.zeros(int(dim), dtype=np.float32)
    out[: arr.size] = arr
    return out


def _ensure_action_matrix(actions: np.ndarray, action_dim: int, length: int) -> np.ndarray:
    arr = np.asarray(actions, dtype=np.float32)
    if arr.ndim != 2:
        raise RuntimeError(f"Action chunk must be 2D, got {arr.ndim}D.")
    if arr.shape[1] != action_dim:
        arr = np.stack([_pad_or_truncate(row, action_dim) for row in arr], axis=0)
    if arr.shape[0] > length:
        arr = arr[:length]
    elif arr.shape[0] < length:
        pad_rows = np.repeat(arr[-1:], length - arr.shape[0], axis=0) if arr.shape[0] > 0 else np.zeros(
            (length, action_dim), dtype=np.float32
        )
        arr = np.concatenate([arr, pad_rows], axis=0)
    return arr


def _infer_env_action_dim(wm_bundle: WMBundle, chunk_actions: np.ndarray) -> int:
    """Infer executed env action width for WM scoring (Metaworld 4D vs packed WM rows)."""
    mean = getattr(wm_bundle.preprocessor, "action_mean", None)
    if mean is not None and hasattr(mean, "numel"):
        try:
            n = int(mean.numel())
            if n > 0:
                return n
        except Exception:
            pass
    wm_dim = _infer_model_action_dim(wm_bundle.model) or int(wm_bundle.planner_action_dim)
    w = int(chunk_actions.shape[1])
    if wm_dim > 0 and w == wm_dim:
        return w
    if wm_dim > 0 and w < wm_dim and wm_dim % w == 0:
        return w
    if wm_dim >= 4 and wm_dim % 4 == 0:
        return 4
    return max(1, w)


def _wm_action_block_factor(env_action_dim: int, wm_action_dim: int) -> int:
    if env_action_dim <= 0 or wm_action_dim <= 0:
        return 1
    if wm_action_dim % env_action_dim != 0:
        return 1
    f = wm_action_dim // env_action_dim
    return f if f > 0 else 1


def _normalize_env_actions_for_wm(
    preprocessor: Any,
    actions_2d: np.ndarray,
    env_dim: int,
    device: Any,
) -> np.ndarray:
    arr = _ensure_action_matrix(np.asarray(actions_2d, dtype=np.float32), env_dim, actions_2d.shape[0])
    norm_fn = getattr(preprocessor, "normalize_actions", None)
    if not callable(norm_fn):
        return arr.astype(np.float32, copy=False)
    _require_torch("WM action normalization requires torch.")
    batch = torch.from_numpy(arr).unsqueeze(0).to(device=device, dtype=torch.float32)
    mean = getattr(preprocessor, "action_mean", None)
    std = getattr(preprocessor, "action_std", None)
    if mean is not None:
        mean_d = torch.as_tensor(mean, dtype=torch.float32, device=device).reshape(1, 1, -1)
        if std is not None:
            std_d = torch.as_tensor(std, dtype=torch.float32, device=device).reshape(1, 1, -1)
        else:
            std_d = torch.ones_like(mean_d)
        out = (batch - mean_d) / std_d
        return out.squeeze(0).detach().cpu().numpy().astype(np.float32)
    out = norm_fn(batch)
    if torch.is_tensor(out):
        return out.squeeze(0).detach().cpu().numpy().astype(np.float32)
    return np.asarray(out, dtype=np.float32).reshape(arr.shape[0], env_dim)


def _pack_env_actions_for_wm(
    actions_norm_2d: np.ndarray,
    factor: int,
    env_dim: int,
    wm_dim: int,
) -> np.ndarray:
    """Pad trailing env steps with **zeros in normalized space** so ``T`` is a multiple of ``factor``."""
    arr = np.asarray(actions_norm_2d, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != env_dim:
        raise RuntimeError(f"Expected normalized actions (T, {env_dim}), got {arr.shape}.")
    if factor <= 1:
        if arr.shape[1] != wm_dim:
            raise RuntimeError(f"WM action dim {wm_dim} != env dim {env_dim} with factor 1.")
        return arr.astype(np.float32, copy=False)
    if factor * env_dim != wm_dim:
        raise RuntimeError(f"WM dim {wm_dim} must equal factor*{env_dim}={factor * env_dim}.")
    t = int(arr.shape[0])
    n_pad = (factor - (t % factor)) % factor
    if n_pad:
        # Pad in normalized space with zeros (= preprocessor mean action when std is finite).
        pad = np.zeros((n_pad, int(env_dim)), dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=0)
    n_blk = int(arr.shape[0]) // factor
    packed = arr.reshape(n_blk, wm_dim)
    return packed.astype(np.float32, copy=False)


def _frame_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Return mean absolute difference normalized to [0,1], with resize fallback."""
    arr_a = np.asarray(a, dtype=np.float32)
    arr_b = np.asarray(b, dtype=np.float32)
    if arr_a.shape != arr_b.shape:
        try:
            from PIL import Image

            arr_b = np.array(Image.fromarray(np.asarray(arr_b, dtype=np.uint8)).resize((arr_a.shape[1], arr_a.shape[0])))
        except Exception as exc:
            raise ValueError(f"Unable to compare frame shapes {arr_a.shape} and {arr_b.shape}: {exc}") from exc
    if arr_a.ndim != 3 or arr_b.ndim != 3:
        raise ValueError(f"Frame similarity expects HWC images, got {arr_a.shape} and {arr_b.shape}")
    if arr_a.dtype == np.uint8 or arr_a.max() > 1.0:
        arr_a = arr_a / 255.0
    if arr_b.dtype == np.uint8 or arr_b.max() > 1.0:
        arr_b = arr_b / 255.0
    return float(np.mean(np.abs(arr_a - arr_b)))


def _to_channel_last(image_like: Any) -> np.ndarray:
    frame = np.asarray(image_like)
    if frame.ndim != 3:
        raise ValueError(f"Decoded frame expected HWC or CHW image, got shape {frame.shape}")
    if frame.shape[0] == 3 and frame.shape[-1] != 3:
        frame = np.transpose(frame, (1, 2, 0))
    return _to_rgb_uint8(frame)


def _to_tensor(
    latents: Any,
    label: str,
    device: torch.device,
) -> tuple[torch.Tensor | None, str | None]:
    if torch is None:
        return None, "Torch is not available for latent tensor conversion."
    if latents is None:
        return None, f"{label} latent trace is empty."
    if isinstance(latents, np.ndarray):
        if latents.size == 0:
            return None, f"{label} latent trace is empty."
    elif isinstance(latents, (list, tuple)):
        if len(latents) == 0:
            return None, f"{label} latent trace is empty."
    else:
        try:
            if len(latents) == 0:  # type: ignore[arg-type]
                return None, f"{label} latent trace is empty."
        except TypeError as exc:
            return None, f"Failed to inspect {label} latent trace length: {exc}"
    try:
        latent_np = np.asarray(latents, dtype=np.float32)
    except Exception as exc:
        return None, f"Failed to convert {label} latents to numpy: {exc}"
    if latent_np.size == 0:
        return None, f"{label} latent trace is empty."
    if latent_np.ndim == 0:
        return None, f"{label} latent tensor is a scalar; expected a trailing feature dimension for decode."
    try:
        lat = torch.as_tensor(latent_np, dtype=torch.float32, device=device)
    except Exception as exc:
        return None, f"Failed to convert {label} latents to torch tensor: {exc}"
    try:
        normalized = lat
        while normalized.ndim > 6:
            squeeze_idx = None
            for idx in range(1, normalized.ndim - 1):
                if normalized.shape[idx] == 1:
                    squeeze_idx = idx
                    break
            if squeeze_idx is None:
                return (
                    None,
                    f"{label} latent tensor shape {tuple(normalized.shape)} has unsupported rank >6 for decode-unroll; "
                    "expected to expand/pad to [T,B,V,H,W,D] after removing singleton extras.",
                )
            normalized = normalized.squeeze(squeeze_idx)
        while normalized.ndim < 6:
            normalized = normalized.unsqueeze(-2)
        if normalized.ndim != 6:
            return (
                None,
                f"{label} latent tensor shape {tuple(normalized.shape)} cannot be normalized to [T,B,V,H,W,D].",
            )
    except Exception as exc:
        return None, f"Failed to normalize {label} latent tensor shape: {exc}"
    return normalized, None


def _adapt_trace_for_decode_unroll(
    latent_trace: DecodeTrace,
    device: torch.device,
) -> dict[str, dict[str, torch.Tensor | None | str | None]]:
    visual_tensor, visual_failure = _to_tensor(latent_trace.visual_latents, "visual", device)
    proprio_tensor, proprio_failure = _to_tensor(latent_trace.proprio_latents, "proprio", device)
    return {
        "visual": {"tensor": visual_tensor, "failure": visual_failure},
        "proprio": {"tensor": proprio_tensor, "failure": proprio_failure},
    }


def _fallback_scoring_distance(
    chunk_actions: np.ndarray,
    reason: str,
) -> tuple[float, str]:
    chunk_flat = np.asarray(chunk_actions, dtype=np.float32).reshape(-1)
    if chunk_flat.size == 0:
        return 1.0e6, f"{reason}: fallback score used for empty action chunk."

    non_finite_count = int(np.size(chunk_flat) - np.sum(np.isfinite(chunk_flat)))
    normalized_chunk = np.nan_to_num(chunk_flat, nan=0.0, posinf=0.0, neginf=0.0)
    base_distance = float(np.linalg.norm(normalized_chunk))
    if not np.isfinite(base_distance):
        base_distance = 0.0

    fallback_distance = float(base_distance + 1.0e6)
    if non_finite_count:
        return (
            fallback_distance,
            f"{reason}: score fallback used. Replaced {non_finite_count} non-finite actions with 0.0, then used norm distance.",
        )
    return (
        fallback_distance,
        f"{reason}: score fallback used. Used action-norm distance as finite fallback.",
    )


def _decode_selected_trace(
    model_bundle: WMBundle,
    latent_trace: DecodeTrace,
) -> tuple[list[np.ndarray], str | None]:
    adapted = _adapt_trace_for_decode_unroll(latent_trace, model_bundle.device)
    visual_entry = adapted["visual"]
    proprio_entry = adapted["proprio"]
    visual_latent_t = visual_entry["tensor"]
    proprio_latent_t = proprio_entry["tensor"]
    visual_failure = visual_entry["failure"]
    proprio_failure = proprio_entry["failure"]

    def _decode_modal(
        modality: str,
        latent_obj: Any,
        preparation_failure: str | None,
    ) -> tuple[list[np.ndarray] | None, str | None]:
        if latent_obj is None:
            if preparation_failure is None:
                return None, f"{modality} latent trace is unavailable."
            return None, preparation_failure

        decoded = None
        decode_unroll = getattr(model_bundle.model, "decode_unroll", None)
        if decode_unroll is None:
            decode_fn = getattr(model_bundle.model, "decode", None)
            if decode_fn is None:
                return None, "Model has no decode_unroll or decode method."
            try:
                decoded = decode_fn(latent_obj, debug=False)
            except TypeError:
                try:
                    decoded = decode_fn(latent_obj)
                except Exception as exc:
                    return None, f"{modality} decode fallback failed: {exc}"
            except Exception as exc:
                return None, f"{modality} decode fallback failed: {exc}"
            if decoded is None:
                return None, f"{modality} decode fallback returned no output."
        else:
            if modality == "proprio" and torch is not None and torch.is_tensor(latent_obj):
                return None, "proprio decode_unroll: no image_head path for proprio-only in JEPA EncPredWM"

            decode_input = latent_obj if isinstance(latent_obj, dict) else {modality: latent_obj}
            if torch is not None and torch.is_tensor(latent_obj) and modality == "visual":
                try:
                    decoded = decode_unroll(latent_obj, batch=True)
                except TypeError:
                    decoded = decode_unroll(latent_obj)
                except Exception as exc:
                    return None, f"{modality} decode_unroll failed: {exc}"
                if decoded is None:
                    return None, f"{modality} decode_unroll returned no output."
            else:
                try:
                    decoded = decode_unroll(decode_input, batch=True)
                except TypeError:
                    try:
                        decoded = decode_unroll(decode_input)
                    except Exception as exc:
                        try:
                            decoded = decode_unroll(latent_obj)
                        except Exception as fallback_exc:
                            return (
                                None,
                                f"{modality} decode_unroll calls failed: {exc}; fallback tensor input failed: {fallback_exc}",
                            )
                except Exception as exc:
                    return None, f"{modality} decode_unroll failed: {exc}"
                if decoded is None:
                    return None, f"{modality} decode_unroll returned no output."

        if isinstance(decoded, dict):
            if "recon" in decoded:
                decoded = decoded["recon"]
            elif "decoded" in decoded:
                decoded = decoded["decoded"]
            elif len(decoded):
                decoded = next(iter(decoded.values()))
            else:
                return None, f"{modality} decode_unroll output was an empty dict."

        try:
            decoded_np = np.asarray(decoded)
        except Exception as exc:
            return None, f"Failed to convert decoded {modality} output to array: {exc}"
        if decoded_np.size == 0:
            return None, f"{modality} decode_unroll returned empty output."
        if decoded_np.ndim == 5:
            decoded_np = decoded_np[0]
        if decoded_np.ndim == 4:
            try:
                return [_to_channel_last(frame) for frame in decoded_np], None
            except Exception as exc:
                return None, f"Failed to convert decoded {modality} frames to images: {exc}"
        if decoded_np.ndim == 3:
            try:
                return [_to_channel_last(decoded_np)], None
            except Exception as exc:
                return None, f"Failed to convert decoded {modality} frame to image: {exc}"
        return None, f"Unexpected decoded frame rank {decoded_np.ndim} for {modality} trace."

    fused_frames = None
    fused_failure = None
    if visual_latent_t is not None and proprio_latent_t is not None:
        fused_frames, fused_failure = _decode_modal(
            "visual+proprio",
            {"visual": visual_latent_t, "proprio": proprio_latent_t},
            None,
        )
        if fused_frames is not None:
            return fused_frames, None

    visual_frames = None
    if visual_latent_t is not None:
        visual_frames, visual_failure = _decode_modal("visual", visual_latent_t, visual_failure)
        if visual_frames is not None:
            return visual_frames, None

    proprio_frames = None
    if proprio_latent_t is not None:
        proprio_frames, proprio_failure = _decode_modal("proprio", proprio_latent_t, proprio_failure)
        if proprio_frames is not None:
            return proprio_frames, None

    failure_parts: list[str] = []
    if fused_failure is not None:
        failure_parts.append(fused_failure)
    if visual_failure is not None:
        failure_parts.append(visual_failure)
    if proprio_failure is not None:
        failure_parts.append(proprio_failure)
    if failure_parts:
        return [], "; ".join(failure_parts)
    return [], "Decode failed for both modalities."


def _decode_latent_trace_to_frames(
    model_bundle: WMBundle,
    latent_trace: DecodeTrace | list[np.ndarray],
) -> tuple[list[np.ndarray], str | None]:
    if isinstance(latent_trace, DecodeTrace):
        return _decode_selected_trace(model_bundle, latent_trace)
    if not latent_trace:
        return [], "Decode trace is empty."
    return _decode_selected_trace(model_bundle, DecodeTrace(visual_latents=list(latent_trace)))


def _select_comparison_frames(
    real_frames: list[np.ndarray],
    pred_frames: list[np.ndarray],
    carried_steps: int | None = None,
    *,
    env_steps_per_wm_step: int | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray], list[int]]:
    """Align predicted frames against real rollout with explicit t0 context policy.

    Returns filtered ``(real, pred, pred_indices)`` where ``pred_indices[i]`` indexes the
    original ``pred_frames`` list for strip column ``i``.
    """
    if not real_frames or not pred_frames:
        return [], [], []
    factor = int(env_steps_per_wm_step or 1)
    if factor > 1:
        if carried_steps is not None:
            cs = int(carried_steps)
        else:
            cs = max(0, len(real_frames) - 1)
        if cs <= 0:
            return [], [], []
        out_real: list[np.ndarray] = []
        out_pred: list[np.ndarray] = []
        pred_indices: list[int] = []
        for k in range(len(pred_frames)):
            ridx = min((k + 1) * factor, cs)
            if ridx < len(real_frames):
                out_real.append(real_frames[ridx])
                out_pred.append(pred_frames[k])
                pred_indices.append(k)
        return out_real, out_pred, pred_indices
    # Always pair predicted rollout steps against real state transitions.
    # real_frames includes the segment start frame + one frame per carried step.
    limit = max(0, len(real_frames) - 1)
    limit = min(limit, len(pred_frames))
    if carried_steps is not None:
        step_limit = int(carried_steps)
        if step_limit <= 0:
            return [], [], []
        limit = min(limit, step_limit)
    if limit <= 0:
        return [], [], []
    pred_indices = list(range(limit))
    return list(real_frames[:limit]), list(pred_frames[:limit]), pred_indices


def _comparison_ridx_for_column(
    pair_index: int,
    *,
    factor: int,
    carried_steps: int,
) -> int:
    """Match `_select_comparison_frames` real frame index for column `pair_index`."""
    f = max(1, int(factor))
    cs = max(0, int(carried_steps))
    if f > 1:
        return min((int(pair_index) + 1) * f, cs)
    return int(pair_index)


def _comparison_strip_overlay_lines(
    *,
    column_idx: int,
    total_columns: int,
    factor: int,
    carried_steps: int,
    overlay_env_step_start: int,
    overlay_selected_candidate_index: int | None,
    wm_step_index: int,
    d_full: list[float] | None,
    delta_full: list[float] | None,
    overlay_segment_index: int | None = None,
) -> list[str]:
    """Text lines for the WM decode overlay box (one comparison-strip column)."""
    ridx = _comparison_ridx_for_column(column_idx, factor=factor, carried_steps=carried_steps)
    env_t = int(overlay_env_step_start) + int(ridx)
    cand = int(overlay_selected_candidate_index) if overlay_selected_candidate_index is not None else -1
    lines: list[str] = []
    k = int(wm_step_index)
    if d_full is not None and delta_full is not None and 0 <= k < len(d_full):
        d_k = d_full[k]
        d_str = f"{d_k:.4f}" if np.isfinite(d_k) else "n/a"
        dd = delta_full[k]
        dd_str = f"{dd:+.4f}" if np.isfinite(dd) else "n/a"
        lines.append(f"d {d_str}  delta {dd_str}")
    else:
        lines.append("d n/a  Δ n/a")
    if overlay_segment_index is not None:
        lines.append(f"seg {int(overlay_segment_index):04d}")
    lines.extend(
        [
            f"real_i{ridx:02d} env_t{env_t:04d}",
            f"wm_dec {column_idx + 1}/{total_columns} f{max(1, factor)}",
            f"cand{cand:03d}",
        ]
    )
    return lines


def _l2_goal_distance_np(vec: np.ndarray, goal: np.ndarray) -> float:
    """Euclidean distance to goal in shared prefix, matching WM scoring slice."""
    v = np.asarray(vec, dtype=np.float64).reshape(-1)
    g = np.asarray(goal, dtype=np.float64).reshape(-1)
    if v.size == 0 or g.size == 0:
        return float("nan")
    n = min(int(v.size), int(g.size))
    d = float(np.linalg.norm(v[:n] - g[:n]))
    return d if np.isfinite(d) else float("nan")


def _latent_overlay_distance_tables(
    initial_vector: np.ndarray,
    step_vectors: list[np.ndarray],
    goal_np: np.ndarray,
) -> tuple[list[float], list[float]]:
    """Per-WM-step L2 to goal and Δ vs previous (first step vs pre-unroll initial)."""
    d_full: list[float] = []
    delta_full: list[float] = []
    d_prev = _l2_goal_distance_np(initial_vector, goal_np)
    for sv in step_vectors:
        d_k = _l2_goal_distance_np(sv, goal_np)
        d_full.append(d_k)
        delta_full.append(d_k - d_prev)
        d_prev = d_k
    return d_full, delta_full


def _all_candidates_wm_goal_l2_rows(
    goal_latent_np: np.ndarray,
    candidate_score_traces: dict[int, ScoreTrace | None],
    *,
    num_candidates: int,
    title: str = "goal_L2_wm_int",
) -> tuple[list[str], list[dict[str, Any]], int]:
    """Footer + JSON: per candidate, integer-rounded L2(goal) per WM unroll only (no Δ, no blk).

    Returns ``(lines, json_rows, wm_step_count)``. ``lines[0]`` is title; one line per candidate index
    in ``0 .. num_candidates-1``. Missing / failed WM trace → ``n/a`` / JSON ``null`` padded to ``wm_step_count``.
    """
    g = np.asarray(goal_latent_np, dtype=np.float32).reshape(-1)
    if g.size == 0 or int(num_candidates) <= 0:
        return [], [], 0
    json_rows: list[dict[str, Any]] = []
    raw_lists: list[list[float | None] | None] = []
    for idx in range(int(num_candidates)):
        st = candidate_score_traces.get(idx)
        d_list: list[float | None] | None = None
        if st is not None and st.step_vectors:
            d_full, _dd = _latent_overlay_distance_tables(
                np.asarray(st.initial_vector, dtype=np.float32).reshape(-1),
                list(st.step_vectors),
                g,
            )
            d_list = [float(x) if np.isfinite(x) else None for x in d_full]
        raw_lists.append(d_list)
    max_k = max((len(d) for d in raw_lists if d), default=0)
    lines: list[str] = [title]
    for idx, d_list in enumerate(raw_lists):
        ints_out: list[int | None] = []
        png_parts: list[str] = []
        if max_k <= 0:
            json_rows.append({"candidate_index": int(idx), "d_goal_l2_wm_int": []})
            lines.append(f"cand{int(idx):02d}  n/a")
            continue
        for k in range(max_k):
            v = d_list[k] if d_list is not None and k < len(d_list) else None
            if v is None or not np.isfinite(v):
                ints_out.append(None)
                png_parts.append(f"{'n/a':>6s}")
            else:
                iv = int(round(float(v)))
                ints_out.append(iv)
                png_parts.append(f"{iv:6d}")
        json_rows.append({"candidate_index": int(idx), "d_goal_l2_wm_int": ints_out})
        lines.append(f"cand{int(idx):02d}  " + " ".join(png_parts))
    return lines, json_rows, int(max_k)


def _wm_megastep_action_range_footer_line(
    *,
    carried_steps: int,
    wm_stride: int,
    wm_step_count: int,
    title: str = "range:",
    cell_width: int = 6,
) -> str:
    """Footer row showing half-open action ranges per WM megastep (label + cells)."""
    c = int(carried_steps)
    step_count = int(wm_step_count)
    if c <= 0 or step_count <= 0:
        return title
    f = max(1, int(wm_stride))
    ranges: list[str] = []
    for k in range(step_count):
        lo = k * f
        hi = min((k + 1) * f, c)
        ranges.append(f"{lo}:{hi}".rjust(cell_width))
    return title + "  " + " ".join(ranges)


def _append_wm_megastep_footer(
    strip_rgb: np.ndarray,
    lines: list[str],
    *,
    min_text_lines: int | None = None,
) -> np.ndarray:
    """Monospace footer band under full-width comparison strip (equalize stitch height via padding)."""
    from PIL import Image, ImageDraw, ImageFont

    base = np.asarray(strip_rgb, dtype=np.uint8)
    if base.ndim == 2:
        raise ValueError("strip_rgb must be color HWC")
    if base.shape[2] == 4:
        base = base[:, :, :3]
    elif base.shape[2] != 3:
        raise ValueError("strip_rgb must be 3 or 4 channels")
    w = int(base.shape[1])
    draw_lines = list(lines)
    if min_text_lines is not None and int(min_text_lines) > len(draw_lines):
        draw_lines.extend([""] * (int(min_text_lines) - len(draw_lines)))
    if not draw_lines:
        return base

    font = getattr(_overlay_decode_panel_metadata, "_font_cache", None)
    if font is None:
        font = None
        for fp in (
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
        ):
            try:
                font = ImageFont.truetype(fp, 13)
                break
            except OSError:
                continue
        if font is None:
            font = ImageFont.load_default()
        setattr(_overlay_decode_panel_metadata, "_font_cache", font)

    line_spacing = 2
    try:
        line_bbox = font.getbbox("Mg")
        line_h = max(14, int(line_bbox[3] - line_bbox[1]))
    except Exception:
        line_h = 14
    pad = 4
    text_h = len(draw_lines) * line_h + max(0, len(draw_lines) - 1) * line_spacing
    footer_h = max(1, text_h + 2 * pad)
    footer = Image.new("RGB", (max(1, w), footer_h), (0, 0, 0))
    draw = ImageDraw.Draw(footer)
    ty = pad
    max_text_x = max(0, w - 1)
    for line in draw_lines:
        if ty > footer_h - pad:
            break
        draw.text((4, min(ty, footer_h - 1)), str(line), fill=(255, 255, 255), font=font)
        ty += line_h + line_spacing
    foot_np = np.asarray(footer, dtype=np.uint8)
    return np.concatenate([base, foot_np], axis=0)


def _overlay_decode_panel_metadata(pred_rgb: np.ndarray, lines: list[str]) -> np.ndarray:
    """Append metadata footer below decoded panel for readable per-column overlays."""
    from PIL import Image, ImageDraw, ImageFont

    pred = _to_rgb_uint8(pred_rgb)
    _, pred_w = pred.shape[0], pred.shape[1]

    font = getattr(_overlay_decode_panel_metadata, "_font_cache", None)
    if font is None:
        font = None
        for fp in (
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
        ):
            try:
                font = ImageFont.truetype(fp, 13)
                break
            except OSError:
                continue
        if font is None:
            font = ImageFont.load_default()
        setattr(_overlay_decode_panel_metadata, "_font_cache", font)

    line_spacing = 2
    line_bbox = font.getbbox("Mg")
    line_h = (line_bbox[3] - line_bbox[1]) if line_bbox is not None else 14
    if line_h <= 0:
        line_h = 14
    line_count = len(lines)
    text_h = line_count * line_h + max(0, line_count - 1) * line_spacing
    tx0 = 4
    pad = 4
    footer_h = max(1, text_h + 2 * pad)
    footer = Image.new("RGB", (pred_w, footer_h), (0, 0, 0))
    draw = ImageDraw.Draw(footer)
    ty = pad
    max_text_x = max(0, pred_w - 1)
    for line in lines:
        if ty > footer_h - pad:
            break
        draw.text((min(tx0, max_text_x), ty), line, fill=(255, 255, 255), font=font)
        ty += line_h + line_spacing
        # Guard against pathological long lines with tiny widths; keep row readable and bounded.
        if ty >= footer_h - pad:
            break
    footer_np = np.asarray(footer, dtype=np.uint8)
    return np.concatenate([pred, footer_np], axis=0)


def _build_real_vs_pred_strip(
    real_frames: list[np.ndarray],
    pred_frames: list[np.ndarray],
    *,
    carried_steps: int | None = None,
    env_steps_per_wm_step: int | None = None,
    overlay_decode_meta: bool = False,
    overlay_env_step_start: int | None = None,
    overlay_selected_candidate_index: int | None = None,
    overlay_wm_env_steps_per_wm_step: int = 1,
    overlay_goal_latent_np: np.ndarray | None = None,
    overlay_score_trace: ScoreTrace | None = None,
    overlay_segment_index: int | None = None,
) -> np.ndarray:
    if not real_frames or not pred_frames:
        raise ValueError("Real and predicted frames are required for strip rendering.")
    real_frames, pred_frames, pred_indices = _select_comparison_frames(
        real_frames,
        pred_frames,
        carried_steps=carried_steps,
        env_steps_per_wm_step=env_steps_per_wm_step,
    )
    if not real_frames or not pred_frames:
        raise ValueError("Real and predicted frames are required for strip rendering.")
    total = len(real_frames)
    factor = int(env_steps_per_wm_step or 1)
    cs = 0 if carried_steps is None else int(carried_steps)
    d_full: list[float] | None = None
    delta_full: list[float] | None = None
    g_np: np.ndarray | None = None
    if (
        overlay_decode_meta
        and overlay_goal_latent_np is not None
        and overlay_score_trace is not None
        and overlay_score_trace.step_vectors
    ):
        g_np = np.asarray(overlay_goal_latent_np, dtype=np.float32).reshape(-1)
        d_full, delta_full = _latent_overlay_distance_tables(
            overlay_score_trace.initial_vector,
            overlay_score_trace.step_vectors,
            g_np,
        )
    pairs: list[np.ndarray] = []
    for idx in range(total):
        real_rgb = _to_rgb_uint8(real_frames[idx])
        pred_rgb = _to_rgb_uint8(pred_frames[idx])
        if pred_rgb.shape != real_rgb.shape:
            from PIL import Image

            pred_rgb = np.array(Image.fromarray(pred_rgb).resize((real_rgb.shape[1], real_rgb.shape[0])))
        if overlay_decode_meta and overlay_env_step_start is not None:
            k = int(pred_indices[idx]) if idx < len(pred_indices) else -1
            lines = _comparison_strip_overlay_lines(
                column_idx=idx,
                total_columns=total,
                factor=factor,
                carried_steps=cs,
                overlay_env_step_start=int(overlay_env_step_start),
                overlay_selected_candidate_index=overlay_selected_candidate_index,
                wm_step_index=k,
                d_full=d_full,
                delta_full=delta_full,
                overlay_segment_index=overlay_segment_index,
            )
            pred_rgb = _overlay_decode_panel_metadata(pred_rgb, lines)
        pairs.append(np.concatenate([real_rgb, pred_rgb], axis=0))
    return np.concatenate(pairs, axis=1)


def _comparison_strip_basename(
    *,
    segment_index: int,
    env_step_start: int,
    carried_steps: int,
    selected_candidate_index: int,
    wm_env_steps_per_wm_step: int = 1,
) -> str:
    start = int(env_step_start)
    carry = max(0, int(carried_steps))
    end = start + carry
    wm = int(wm_env_steps_per_wm_step) if wm_env_steps_per_wm_step is not None else 1
    wm = max(1, wm)
    wm_part = f"wmf{wm:02d}_" if wm > 1 else ""
    return (
        f"{wm_part}comparison_strip_steps_{start:04d}_to_{end:04d}_"
        f"seg{segment_index:04d}_cand{selected_candidate_index:03d}.png"
    )


def _write_comparison_segment_strip(
    out_dir: Path,
    episode_index: int,
    segment_index: int,
    real_frames: list[np.ndarray],
    pred_frames: list[np.ndarray],
    *,
    env_step_start: int,
    selected_candidate_index: int,
    carried_steps: int | None = None,
    env_steps_per_wm_step: int | None = None,
    wm_env_steps_per_wm_step: int = 1,
    overlay_decode_meta: bool = False,
    overlay_goal_latent_np: np.ndarray | None = None,
    overlay_score_trace: ScoreTrace | None = None,
    wm_megastep_footer_lines: list[str] | None = None,
    wm_megastep_footer_min_lines: int | None = None,
) -> tuple[Path | None, str | None]:
    if carried_steps is not None and int(carried_steps) <= 0:
        return None, None
    if not real_frames or not pred_frames:
        return None, None
    out_dir = Path(out_dir)
    try:
        strip = _build_real_vs_pred_strip(
            real_frames,
            pred_frames,
            carried_steps=carried_steps,
            env_steps_per_wm_step=env_steps_per_wm_step,
            overlay_decode_meta=bool(overlay_decode_meta),
            overlay_env_step_start=int(env_step_start),
            overlay_selected_candidate_index=int(selected_candidate_index),
            overlay_wm_env_steps_per_wm_step=int(wm_env_steps_per_wm_step),
            overlay_goal_latent_np=overlay_goal_latent_np,
            overlay_score_trace=overlay_score_trace,
            overlay_segment_index=int(segment_index) if overlay_decode_meta else None,
        )
    except Exception as exc:
        reason = f"Failed to build comparison strip for episode {episode_index} segment {segment_index}: {exc}"
        print(f"[segment_grpo] {reason}")
        return None, reason

    try:
        import imageio.v2 as imageio
    except Exception as exc:
        reason = f"Failed to import imageio for comparison strip for episode {episode_index} segment {segment_index}: {exc}"
        print(f"[segment_grpo] {reason}")
        return None, reason

    if wm_megastep_footer_lines:
        try:
            strip = _append_wm_megastep_footer(
                strip,
                wm_megastep_footer_lines,
                min_text_lines=wm_megastep_footer_min_lines,
            )
        except Exception as exc:
            reason = f"Failed to append WM megastep footer for episode {episode_index} segment {segment_index}: {exc}"
            print(f"[segment_grpo] {reason}")
            return None, reason

    carry = 0 if carried_steps is None else int(carried_steps)
    basename = _comparison_strip_basename(
        segment_index=int(segment_index),
        env_step_start=int(env_step_start),
        carried_steps=carry,
        selected_candidate_index=int(selected_candidate_index),
        wm_env_steps_per_wm_step=int(wm_env_steps_per_wm_step),
    )
    episode_dir = Path(out_dir) / f"episode_{episode_index:04d}"
    path = episode_dir / basename
    try:
        episode_dir.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(path, strip)
        return path, None
    except Exception as exc:
        reason = f"Failed to write comparison strip to {path}: {exc}"
        print(f"[segment_grpo] {reason}")
        return None, reason


def _stitch_comparison_strip(
    segments: list[Path],
    output_path: Path,
    *,
    gutter_pixels: int = 0,
    gutter_rgb: tuple[int, int, int] = (48, 48, 48),
) -> tuple[Path | None, str | None]:
    if not segments:
        return None, None

    try:
        import imageio.v2 as imageio
    except Exception as exc:
        reason = f"Failed to import imageio for stitching comparison strip: {exc}"
        print(f"[segment_grpo] {reason}")
        return None, reason

    strips: list[np.ndarray] = []
    try:
        for seg in segments:
            strips.append(imageio.imread(seg))
    except Exception as exc:
        reason = f"Failed to read comparison strip segment for stitching: {exc}"
        print(f"[segment_grpo] {reason}")
        return None, reason
    if not strips:
        return None, "No comparison strip segments could be loaded."
    try:
        gw = max(0, int(gutter_pixels))
        normalized: list[np.ndarray] = []
        target_h: int | None = None
        for arr in strips:
            if arr.ndim == 2:
                return None, "Comparison strip segment must be RGB/RGBA (got grayscale)."
            if arr.ndim != 3:
                return None, "Comparison strip segment must be a 3-D array (H, W, C)."
            h, _, c = arr.shape[0], arr.shape[1], arr.shape[2]
            if target_h is None:
                target_h = int(h)
            elif int(h) != target_h:
                return None, f"Comparison strip height mismatch for stitch: expected {target_h}, got {h}."
            if c == 4:
                arr = arr[:, :, :3]
            elif c != 3:
                return None, f"Comparison strip segment must have 3 or 4 channels; got {c}."
            normalized.append(np.asarray(arr, dtype=np.uint8))

        parts: list[np.ndarray] = []
        for i, arr in enumerate(normalized):
            parts.append(arr)
            if gw > 0 and i < len(normalized) - 1:
                fill = np.array(gutter_rgb, dtype=np.uint8).reshape(1, 1, 3)
                gutter = np.tile(fill, (target_h, gw, 1))
                parts.append(gutter)

        stitched = np.concatenate(parts, axis=1)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(output_path, stitched)
        return output_path, None
    except Exception as exc:
        reason = f"Failed to write stitched comparison strip to {output_path}: {exc}"
        print(f"[segment_grpo] {reason}")
        return None, reason


def _extract_image_and_proprio(obs: Any, env: Any) -> tuple[np.ndarray, np.ndarray]:
    from smolvla_obs_state import flatten_obs_state

    module = _load_jepa_helper_module()
    img = module._find_image(obs)
    if img is None and env is not None and hasattr(env, "render"):
        img = env.render()
    if img is None:
        raise RuntimeError("Unable to extract an image from observation; ensure env.render() is available.")
    proprio = flatten_obs_state(obs)
    return _to_rgb_uint8(img), np.asarray(proprio, dtype=np.float32).reshape(-1)


def load_smolvla_bundle(
    checkpoint: str | Path | None,
    device: str | torch.device | None,
    *,
    n_action_steps: int = 1,
) -> Any:
    """
    Load SmolVLA execution bundle through shared helper.

    Returns the same bundle type as vendor helper _try_load_smolvla_exec.
    """
    _require_torch("SmolVLA execution requires torch.")
    ckpt = (str(checkpoint) if checkpoint is not None else "").strip()
    if not ckpt:
        raise RuntimeError(
            "Missing SmolVLA checkpoint. Set --checkpoint (or --dry-run to use synthetic chunks)."
        )
    helper = _load_jepa_helper_module()
    try:
        bundle = helper._try_load_smolvla_exec(
            ckpt, _resolve_device(device), int(max(1, int(n_action_steps)))
        )
    except Exception as exc:  # pragma: no cover - robust runtime path
        raise RuntimeError(
            f"Failed to initialize SmolVLA from '{ckpt}'. "
            "Check the checkpoint string/path and that lerobot + dependencies are available."
        ) from exc
    if bundle is None:
        raise RuntimeError(
            f"SmolVLA helper returned None for checkpoint '{ckpt}'. "
            "Check SMOLVLA policy checkpoint and install dependencies."
        )
    try:
        bundle.policy.reset()
        bundle.preprocessor.reset()
        bundle.postprocessor.reset()
    except Exception as exc:
        print(f"[segment_grpo] failed to reset SmolVLA bundle helpers: {exc}")
    return bundle


def load_wm_bundle(
    jepa_repo: str | Path | None,
    checkpoint: str,
    device: str | torch.device | None,
    *,
    required: bool = False,
) -> WMBundle | None:
    """
    Load JEPA-WM bundle with helper functions and graceful fallback.
    """
    _require_torch("JEPA-WM loading requires torch.")
    repo_path = (str(jepa_repo) if jepa_repo is not None else "").strip()
    helper = _load_jepa_helper_module()
    if not repo_path:
        if required:
            raise RuntimeError(
                "Missing --jepa-repo. Provide a local/Hub repo path for JEPA-WM, or add --dry-run "
                "to continue with synthetic latent scoring."
            )
        return None

    ckpt = (str(checkpoint).strip() if checkpoint is not None else "jepa_wm_metaworld.pth.tar")
    dev = _resolve_device(device)
    try:
        loaded = helper._try_load_wm(Path(repo_path), ckpt, dev)
    except Exception as exc:  # pragma: no cover - robust runtime path
        if required:
            raise RuntimeError(
                f"JEPA-WM load failed from '{repo_path}' with checkpoint '{ckpt}'. "
                "Verify repo path exists and torch.hub can load the entry point."
            ) from exc
        return None

    if loaded is None:
        if required:
            raise RuntimeError(
                f"JEPA-WM helper returned None from '{repo_path}'. "
                "This usually means torch.hub could not resolve the checkpoint."
            )
        return None

    model, preprocessor = loaded
    try:
        action_dims = helper._infer_action_dims(model, preprocessor)
    except Exception as exc:  # pragma: no cover
        if required:
            raise RuntimeError("Failed to infer JEPA-WM action dimensions.") from exc
        action_dims = [4]
    planner_action_dim = int(max(action_dims)) if action_dims else 4
    model_action_dim = _infer_model_action_dim(model)
    if model_action_dim > 0:
        planner_action_dim = max(planner_action_dim, model_action_dim)

    proprio_mean = getattr(preprocessor, "proprio_mean", None)
    proprio_dim = int(np.asarray(proprio_mean).reshape(-1).size) if proprio_mean is not None else 0
    if proprio_dim <= 0:
        proprio_dim = 4

    return WMBundle(
        model=model,
        preprocessor=preprocessor,
        proprio_dim=proprio_dim,
        planner_action_dim=planner_action_dim,
        device=dev,
    )


def _extract_latent(z_pred: Any) -> torch.Tensor:
    _require_torch("Latent extraction requires torch.")
    candidate: Any = z_pred
    if isinstance(z_pred, dict):
        if "latent" in z_pred:
            candidate = z_pred["latent"]
        elif len(z_pred) > 0:
            candidate = next(iter(z_pred.values()))
        else:
            raise RuntimeError("Empty latent dict.")
    elif hasattr(z_pred, "keys"):
        try:
            if "latent" in z_pred:  # type: ignore[index]
                candidate = z_pred["latent"]  # type: ignore[index]
            else:
                preferred_keys = ("visual", "proprio", "state", "prediction", "pred", "z", "latent")
                found = False
                for key in preferred_keys:
                    try:
                        value = z_pred[key]  # type: ignore[index]
                    except Exception:
                        continue
                    if torch.is_tensor(value):
                        candidate = value
                        found = True
                        break
                if not found:
                    try:
                        values = list(z_pred.values())  # type: ignore[call-arg]
                    except Exception as exc:
                        raise RuntimeError("Could not iterate latent dictionary.") from exc
                    for value in values:
                        if torch.is_tensor(value):
                            candidate = value
                            found = True
                            break
                    if not found:
                        raise RuntimeError("No tensor-valued latent found in dictionary output.")
        except TypeError:
            # If z_pred behaves like a mapping but cannot be indexed as expected,
            # fall back to numpy/torch conversion below.
            candidate = z_pred
    if candidate is None:
        raise RuntimeError("Failed to resolve latent candidate from model output.")
    if torch.is_tensor(candidate):
        candidate_t = candidate
    else:
        candidate_t = torch.as_tensor(np.asarray(candidate), dtype=torch.float32)
    if candidate_t.numel() <= 0:
        raise RuntimeError("Empty latent tensor extracted from model output.")
    return candidate_t


def _as_tensor_dict_if_available(state: Any) -> Any:
    """Best-effort conversion of dict-like WM states to tensordict TensorDict."""
    try:
        from tensordict import TensorDict
    except Exception:
        return state
    if isinstance(state, TensorDict):
        return state
    if not hasattr(state, "keys"):
        return state
    try:
        has_visual = "visual" in state  # type: ignore[index]
        has_proprio = "proprio" in state  # type: ignore[index]
    except Exception:
        return state
    if not (has_visual and has_proprio):
        return state
    try:
        visual = state["visual"]  # type: ignore[index]
        proprio = state["proprio"]  # type: ignore[index]
    except Exception:
        return state
    if not (torch is not None and torch.is_tensor(visual) and torch.is_tensor(proprio)):
        return state
    try:
        return TensorDict({"visual": visual, "proprio": proprio}, device=visual.device)
    except Exception:
        return state


def _extract_latent_with_fallback(z_pred: Any) -> torch.Tensor:
    """Prefer `_extract_latent` and fall back to permissive tensor extraction for atypical outputs."""
    try:
        return _extract_latent(z_pred)
    except RuntimeError as primary_exc:
        if hasattr(z_pred, "keys"):
            try:
                if "latent" in z_pred:  # type: ignore[index]
                    candidate = z_pred["latent"]  # type: ignore[index]
                    candidate_t = candidate if torch.is_tensor(candidate) else torch.as_tensor(np.asarray(candidate), dtype=torch.float32)
                    if candidate_t.numel() > 0:
                        return candidate_t
                preferred_keys = ("visual", "proprio", "state", "prediction", "pred", "z")
                for key in preferred_keys:
                    try:
                        candidate = z_pred[key]  # type: ignore[index]
                    except Exception:
                        continue
                    if torch.is_tensor(candidate) and candidate.numel() > 0:
                        return candidate
                try:
                    values = list(z_pred.values())  # type: ignore[call-arg]
                except Exception:
                    values = []
                for value in values:
                    if torch.is_tensor(value) and value.numel() > 0:
                        return value
            except Exception:
                pass
        if isinstance(z_pred, (list, tuple)):
            for value in z_pred:
                if torch.is_tensor(value) and value.numel() > 0:
                    return value
        raise primary_exc


def _extract_scoring_latent(z_pred: Any, mode: str = "visual") -> torch.Tensor:
    _require_torch("WM scoring requires torch.")
    normalized_mode = str(mode).strip().lower()
    if normalized_mode not in ("visual", "proprio", "concat"):
        raise ValueError(f"Unsupported WM scoring mode {mode!r}; expected visual, proprio, or concat.")

    if torch is not None and torch.is_tensor(z_pred):
        candidate = z_pred
    elif hasattr(z_pred, "keys"):
        if normalized_mode == "visual":
            if "visual" not in z_pred:
                raise KeyError("visual")
            candidate = z_pred["visual"]  # type: ignore[index]
        elif normalized_mode == "proprio":
            if "proprio" not in z_pred:
                raise KeyError("proprio")
            candidate = z_pred["proprio"]  # type: ignore[index]
        else:
            if "visual" not in z_pred:
                raise KeyError("visual")
            if "proprio" not in z_pred:
                raise KeyError("proprio")
            visual = z_pred["visual"]  # type: ignore[index]
            proprio = z_pred["proprio"]  # type: ignore[index]
            if not torch.is_tensor(visual):
                visual = torch.as_tensor(np.asarray(visual), dtype=torch.float32)
            if not torch.is_tensor(proprio):
                proprio = torch.as_tensor(np.asarray(proprio), dtype=torch.float32)
            candidate = torch.cat([visual, proprio], dim=-1)
            if candidate.ndim < 1:
                raise RuntimeError("Concatenated scoring latent is empty.")
            return candidate
    else:
        candidate = torch.as_tensor(np.asarray(z_pred), dtype=torch.float32)

    if not torch.is_tensor(candidate):
        candidate = torch.as_tensor(np.asarray(candidate), dtype=torch.float32)
    if int(candidate.numel()) <= 0:
        raise RuntimeError("No tensor-valued latent available for scoring extraction.")
    return candidate


def _next_latent_state_after_unroll(out: Any) -> Any:
    """Reduce multi-step unroll output to latent state suitable for another one-step unroll."""
    tensordict_type = None
    try:
        from tensordict import TensorDict
        tensordict_type = TensorDict
    except Exception:
        tensordict_type = None

    if tensordict_type is not None and isinstance(out, tensordict_type):
        items = list(out.items())
        nxt: dict[str, Any] = {}
        for key, val in items:
            if torch is not None and torch.is_tensor(val) and val.dim() >= 1:
                nxt[str(key)] = val[-1:].detach() if int(val.shape[0]) > 1 else val.detach()
            else:
                nxt[str(key)] = val
        try:
            device = getattr(out, "device", None)
            return tensordict_type(nxt, device=device)
        except Exception:
            return nxt
    if isinstance(out, dict):
        nxt: dict[str, Any] = {}
        for key, val in out.items():
            if torch is not None and torch.is_tensor(val) and val.dim() >= 1:
                nxt[key] = val[-1:].detach() if int(val.shape[0]) > 1 else val.detach()
            else:
                nxt[key] = val
        return nxt
    if hasattr(out, "keys"):
        nxt: dict[str, Any] = {}
        for key in list(out.keys()):
            try:
                val = out[key]  # type: ignore[index]
            except Exception:
                continue
            if torch is not None and torch.is_tensor(val) and val.dim() >= 1:
                nxt[str(key)] = val[-1:].detach() if int(val.shape[0]) > 1 else val.detach()
            else:
                nxt[str(key)] = val
        if nxt:
            return nxt
    if torch is not None and torch.is_tensor(out):
        if out.dim() >= 1 and int(out.shape[0]) > 1:
            return out[-1:].detach()
        return out.detach()
    return out


def _latent_vector_from_unroll_step(unroll_out: Any, scoring_mode: str = "visual") -> torch.Tensor:
    """Single-step latent used for distance scoring and trace (flattened 1-D)."""
    _require_torch("WM scoring requires torch.")
    lat = _extract_scoring_latent(unroll_out, mode=scoring_mode)
    if lat.ndim >= 3:
        final = lat[-1]
    elif lat.ndim == 2 and int(lat.shape[0]) > 1:
        final = lat[-1]
    else:
        final = lat
    return final.reshape(-1)


def _infer_model_action_dim(model: Any) -> int:
    model_root = getattr(model, "model", model)
    candidates: list[int] = []

    def _add(value: Any) -> None:
        try:
            dim = int(value)
        except (TypeError, ValueError):
            return
        if dim > 0:
            candidates.append(dim)

    _add(getattr(model_root, "action_dim", 0))
    action_encoder = getattr(model_root, "action_encoder", None)
    _add(getattr(action_encoder, "in_features", 0))
    predictor = getattr(model_root, "predictor", None)
    _add(getattr(getattr(predictor, "action_encoder", None), "in_features", 0))
    if not candidates:
        predictor = model_root
        _add(getattr(predictor, "action_dim", 0))
    if not candidates:
        return 0
    return max(int(x) for x in candidates)


def score_chunk_by_goal_latent(
    wm_bundle: WMBundle,
    image: np.ndarray,
    proprio: np.ndarray,
    chunk_actions: np.ndarray,
    goal_latent: torch.Tensor,
    *,
    chunk_len: int | None = None,
    return_latent_trace: bool = False,
    wm_rollout_mode: str = "iterative",
    wm_scoring_latent: str = "visual",
) -> float | tuple[float, ScoreTrace, DecodeTrace]:
    """
    Score a candidate chunk by JEPA-WM latent distance to goal latent.

    ``wm_rollout_mode``:
    - ``iterative``: one ``model.unroll`` per WM timestep (packed env actions when ``wm_dim > env_dim``).
    - ``batched``: single ``unroll`` over the full WM action suffix (legacy).

    Returns euclidean distance (smaller is better).
    When ``return_latent_trace`` is True, also returns score and decode traces.
    """
    if not isinstance(chunk_actions, np.ndarray):
        chunk_actions = np.asarray(chunk_actions, dtype=np.float32)
    if chunk_actions.ndim != 2:
        raise RuntimeError(f"chunk_actions must be 2-D, got shape {chunk_actions.shape}")

    env_dim = _infer_env_action_dim(wm_bundle, chunk_actions)
    model_action_dim = _infer_model_action_dim(wm_bundle.model)
    wm_dim = int(model_action_dim) if model_action_dim else int(wm_bundle.planner_action_dim)
    if wm_dim <= 0:
        wm_dim = int(wm_bundle.planner_action_dim)
    factor = _wm_action_block_factor(env_dim, wm_dim)
    length = int(chunk_len or chunk_actions.shape[0])
    n_take = min(int(chunk_actions.shape[1]), env_dim)
    chunk_env = _ensure_action_matrix(
        np.asarray(chunk_actions[:, :n_take], dtype=np.float32),
        env_dim,
        length,
    )
    actions_norm_np = _normalize_env_actions_for_wm(
        wm_bundle.preprocessor, chunk_env, env_dim, wm_bundle.device
    )
    packed_np = _pack_env_actions_for_wm(actions_norm_np, factor, env_dim, wm_dim)
    _require_torch("score_chunk_by_goal_latent requires torch.")
    actions_t = torch.from_numpy(packed_np).to(wm_bundle.device).to(dtype=torch.float32).unsqueeze(1)
    mode = str(wm_rollout_mode or "iterative").strip().lower()
    if mode not in ("iterative", "batched"):
        raise ValueError(f"wm_rollout_mode must be 'iterative' or 'batched', got {wm_rollout_mode!r}")
    def _append_trace_steps(target: list[np.ndarray], raw: Any) -> None:
        if raw is None:
            return
        if torch is not None and torch.is_tensor(raw):
            raw_np = raw.detach().cpu().numpy()
        else:
            raw_np = np.asarray(raw)
        if raw_np.ndim == 0:
            return
        target.extend([np.asarray(item, dtype=np.float32) for item in raw_np])

    def _append_last_timestep_as_numpy_list(target: list[np.ndarray], raw: Any) -> None:
        if raw is None:
            return
        if torch is not None and torch.is_tensor(raw):
            raw_t = raw.detach().cpu()
            if raw_t.ndim == 0 or raw_t.shape[0] == 0:
                return
            target.append(np.asarray(raw_t[-1], dtype=np.float32))
            return
        raw_np = np.asarray(raw)
        if raw_np.ndim == 0 or raw_np.shape[0] == 0:
            return
        target.append(np.asarray(raw_np[-1], dtype=np.float32))

    initial_vector_np = np.zeros(0, dtype=np.float32)
    with torch.no_grad():
        debug_unroll = os.environ.get("DEBUG_WM_UNROLL_OUTPUT", "").strip().lower() in {"1", "true", "yes"}
        visual = _to_wm_visual(image, wm_bundle.device)
        proprio_t = _to_wm_proprio(proprio, wm_bundle.proprio_dim, wm_bundle.device)
        latent_state = _as_tensor_dict_if_available(
            wm_bundle.model.encode({"visual": visual, "proprio": proprio_t})
        )
        try:
            iv0 = _latent_vector_from_unroll_step(latent_state, wm_scoring_latent)
            initial_vector_np = np.asarray(iv0.detach().cpu(), dtype=np.float32).reshape(-1)
        except Exception:
            pass
        seq_actions = actions_t.to(wm_bundle.device).to(dtype=torch.float32)
        if seq_actions.ndim != 3:
            raise RuntimeError(f"Expected action sequence tensor shape (T, B, A), got {seq_actions.shape}")

        score_trace_steps: list[np.ndarray] = []
        decode_trace_steps: list[np.ndarray] = []
        decode_visual_steps: list[np.ndarray] = []
        decode_proprio_steps: list[np.ndarray] = []
        if mode == "batched":
            unroll_output = wm_bundle.model.unroll(latent_state, act_suffix=seq_actions, debug=False)
            if debug_unroll:
                out_shape = None
                try:
                    out_shape = tuple(int(x) for x in unroll_output.shape)
                except Exception:
                    pass
                print(f"[wm-debug] batched unroll_output type={type(unroll_output)} shape={out_shape}", flush=True)
                if hasattr(unroll_output, "keys"):
                    try:
                        keys = list(unroll_output.keys())  # type: ignore[call-arg]
                    except Exception:
                        keys = None
                    if keys is not None:
                        print(f"[wm-debug]   keys={keys}", flush=True)
                        for key in keys:
                            try:
                                value = unroll_output[key]  # type: ignore[index]
                            except Exception:
                                continue
                            if torch is not None and torch.is_tensor(value):
                                print(f"[wm-debug]     key={key} shape={tuple(int(x) for x in value.shape)}", flush=True)
            if hasattr(unroll_output, "keys"):
                try:
                    _append_trace_steps(decode_visual_steps, unroll_output.get("visual"))
                    _append_trace_steps(decode_proprio_steps, unroll_output.get("proprio"))
                except Exception:
                    _log.warning("decode visual/proprio trace append failed for batched unroll output.")
                    try:
                        _append_trace_steps(decode_visual_steps, unroll_output["visual"])  # type: ignore[index]
                        _append_trace_steps(decode_proprio_steps, unroll_output["proprio"])  # type: ignore[index]
                    except Exception:
                        _log.warning("decode visual/proprio trace append fallback failed for batched unroll output.")
            scoring_trace = _extract_scoring_latent(unroll_output, mode=wm_scoring_latent)
            rollout_trace = _extract_latent(scoring_trace)
            if rollout_trace.ndim < 3:
                raise RuntimeError(
                    f"Expected rollout latent trace with time dimension, got shape {tuple(rollout_trace.shape)}"
                )
            for step in rollout_trace:
                step_latent = step.detach().cpu().numpy().astype(np.float32, copy=False)
                score_vec = _latent_vector_from_unroll_step(step, scoring_mode=wm_scoring_latent)
                score_trace_steps.append(np.asarray(score_vec.detach().cpu(), dtype=np.float32))
                decode_trace_steps.append(step_latent)
            pred = torch.as_tensor(score_trace_steps[-1], dtype=torch.float32, device=wm_bundle.device)
        else:
            z_t: Any = _next_latent_state_after_unroll(latent_state)
            for t in range(int(seq_actions.shape[0])):
                one = seq_actions[t : t + 1]
                unroll_out = wm_bundle.model.unroll(z_t, act_suffix=one, debug=False)
                if debug_unroll:
                    print(f"[wm-debug] iterative t={t} output type={type(unroll_out)}", flush=True)
                    if torch is not None and torch.is_tensor(unroll_out):
                        print(f"[wm-debug]   output tensor shape={tuple(int(x) for x in unroll_out.shape)}", flush=True)
                    if hasattr(unroll_out, "keys"):
                        try:
                            keys = list(unroll_out.keys())  # type: ignore[call-arg]
                        except Exception:
                            keys = None
                        if keys is not None:
                            print(f"[wm-debug]   output keys={keys}", flush=True)
                            for key in keys:
                                try:
                                    value = unroll_out[key]  # type: ignore[index]
                                except Exception:
                                    continue
                                if torch is not None and torch.is_tensor(value):
                                    print(f"[wm-debug]     key={key} shape={tuple(int(x) for x in value.shape)}", flush=True)
                                else:
                                    print(f"[wm-debug]     key={key} type={type(value)}", flush=True)
                                if isinstance(value, dict):
                                    try:
                                        inner_keys = list(value.keys())  # type: ignore[call-arg]
                                    except Exception:
                                        inner_keys = None
                                    if inner_keys is not None:
                                        print(f"[wm-debug]       nested keys={inner_keys}", flush=True)
                step_latent = _extract_latent_with_fallback(unroll_out)
                step_vec = _latent_vector_from_unroll_step(unroll_out, scoring_mode=wm_scoring_latent)
                score_trace_steps.append(np.asarray(step_vec.detach().cpu(), dtype=np.float32))
                decode_trace_steps.append(np.asarray(step_latent.detach().cpu(), dtype=np.float32))
                if hasattr(unroll_out, "keys"):
                    try:
                        _append_last_timestep_as_numpy_list(decode_visual_steps, unroll_out.get("visual"))  # type: ignore[attr-defined]
                        _append_last_timestep_as_numpy_list(decode_proprio_steps, unroll_out.get("proprio"))  # type: ignore[attr-defined]
                    except Exception:
                        try:
                            _append_last_timestep_as_numpy_list(
                                decode_visual_steps, unroll_out["visual"]
                            )  # type: ignore[index]
                            _append_last_timestep_as_numpy_list(
                                decode_proprio_steps, unroll_out["proprio"]
                            )  # type: ignore[index]
                        except Exception:
                            _log.warning(
                                "decode visual/proprio trace append failed for iterative unroll output at step t=%s.",
                                t,
                            )
                z_t = _as_tensor_dict_if_available(_next_latent_state_after_unroll(unroll_out))
            pred = torch.as_tensor(score_trace_steps[-1], dtype=torch.float32, device=wm_bundle.device)

        goal = goal_latent.reshape(-1).to(wm_bundle.device)
        if pred.numel() == 0 or goal.numel() == 0:
            raise RuntimeError("Invalid latent shape for scoring.")
        n = min(pred.numel(), goal.numel())
        distance = torch.linalg.vector_norm(pred[:n] - goal[:n], ord=2).item()
    if return_latent_trace:
        score_trace = ScoreTrace(
            step_vectors=score_trace_steps,
            final_vector=score_trace_steps[-1] if score_trace_steps else np.asarray([], dtype=np.float32),
            initial_vector=initial_vector_np,
        )
        scoring_mode = str(wm_scoring_latent or "visual").strip().lower()
        visual_for_decode = list(decode_visual_steps) if decode_visual_steps else []
        if not visual_for_decode and scoring_mode == "visual":
            visual_for_decode = list(decode_trace_steps)
        decode_trace = DecodeTrace(
            visual_latents=visual_for_decode,
            proprio_latents=decode_proprio_steps,
            env_steps_per_wm_step=int(factor),
        )
        return float(distance), score_trace, decode_trace
    return float(distance)


def _count_unique_action_rows(chunk: np.ndarray) -> int:
    """Stable-ish count of distinct action rows (float-safe)."""
    arr = np.asarray(chunk, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return 0
    rounded = np.round(arr, decimals=6)
    return int(len({tuple(row.tolist()) for row in rounded}))


def _sample_smolvla_chunk(
    smolvla_bundle: Any,
    image: np.ndarray,
    proprio: np.ndarray,
    chunk_len: int,
    env_action_dim: int,
    task_text: str,
    rng: np.random.Generator,
    *,
    noise_std: float = 0.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    helper = _load_jepa_helper_module()
    # Proprio only: dict obs with both image+state is flattened with sorted keys ("image" first),
    # so _smolvla_exec_action* would fill state tensors from pixels. RGB comes from render_proxy.
    obs = np.asarray(proprio, dtype=np.float32).reshape(-1)

    class _RenderProxy:
        def __init__(self, frame: np.ndarray):
            self._frame = frame

        def render(self, *args: Any, **kwargs: Any) -> np.ndarray:
            return self._frame

    std = float(noise_std)
    meta: dict[str, Any] = {"chunk_generation_mode": "unknown"}

    def _rows_from_base(base_vec: np.ndarray) -> np.ndarray:
        base_vec = np.asarray(base_vec, dtype=np.float32).reshape(-1)
        rows_out: list[np.ndarray] = []
        for _ in range(chunk_len):
            padded = _pad_or_truncate(base_vec, int(env_action_dim))
            if std > 0.0:
                noise = rng.normal(0.0, std, size=int(env_action_dim)).astype(np.float32)
                action = np.clip(padded + noise, -1.0, 1.0)
            else:
                action = np.clip(padded, -1.0, 1.0)
            rows_out.append(action)
        return np.stack(rows_out, axis=0).astype(np.float32)

    render_proxy = _RenderProxy(_to_rgb_uint8(image))
    try:
        seq = helper._smolvla_exec_action_chunk(
            smolvla_bundle, obs, render_proxy, task_text
        )
        seq = np.asarray(seq, dtype=np.float32)
        if seq.ndim != 2 or seq.shape[0] < 1:
            raise RuntimeError(f"invalid chunk shape from SmolVLA: {seq.shape}")
        meta["chunk_generation_mode"] = "sequence_head"
        meta["policy_chunk_rows_raw"] = int(seq.shape[0])
        t_raw = int(seq.shape[0])
        rows_out: list[np.ndarray] = []
        for i in range(chunk_len):
            row = seq[min(i, t_raw - 1)]
            padded = _pad_or_truncate(row, int(env_action_dim))
            if std > 0.0:
                noise = rng.normal(0.0, std, size=int(env_action_dim)).astype(np.float32)
                action = np.clip(padded + noise, -1.0, 1.0)
            else:
                action = np.clip(padded, -1.0, 1.0)
            rows_out.append(action)
        chunk = np.stack(rows_out, axis=0).astype(np.float32)
    except Exception as chunk_exc:
        meta["chunk_generation_error"] = f"{type(chunk_exc).__name__}: {chunk_exc}"
        try:
            base = helper._smolvla_exec_action(
                smolvla_bundle, obs, render_proxy, task_text
            )
            base = np.asarray(base, dtype=np.float32).reshape(-1)
        except Exception as exc:
            raise RuntimeError(
                f"SmolVLA inference failed for chunk sampling: {exc}"
            ) from exc
        meta["chunk_generation_mode"] = "single_step_fallback"
        meta["policy_chunk_rows_raw"] = 1
        chunk = _rows_from_base(base)

    meta["policy_n_action_steps_returned"] = int(chunk.shape[0])
    meta["unique_action_rows"] = _count_unique_action_rows(chunk)
    return chunk, meta


def _synthetic_chunk(plan_dim: int, chunk_len: int, candidate_idx: int, rng: np.random.Generator) -> np.ndarray:
    base = rng.normal(0.0, 0.15, size=plan_dim).astype(np.float32)
    base = np.tanh(base)
    out = []
    for step_idx in range(chunk_len):
        phase = (candidate_idx + 1) * 0.13
        sine = np.sin(0.2 * (step_idx + 1) + phase).astype(np.float32)
        if out:
            pattern = np.array(np.tanh(sine), dtype=np.float32) * 0.25
            if pattern.size != 1:
                pattern = np.array([pattern], dtype=np.float32)
        else:
            pattern = np.array([sine], dtype=np.float32)
        repeat = int(np.ceil(plan_dim / pattern.size))
        noise = np.tile(pattern, repeat)[:plan_dim]
        act = np.clip(base + 0.2 * noise, -1.0, 1.0).astype(np.float32)
        out.append(act)
    return np.stack(out, axis=0).astype(np.float32)


def _build_synthetic_replay(
    length: int,
    proprio_dim: int,
    seed: int,
    image_shape: tuple[int, int, int] = (256, 256, 3),
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    steps = int(max(1, length))
    images = rng.integers(0, 256, size=(steps + 1, *image_shape), dtype=np.uint8)
    proprio = rng.normal(0.0, 1.0, size=(steps + 1, int(proprio_dim))).astype(np.float32)
    return images, proprio


def _load_replay_root(replay_root: str | Path | None, *, dry_run: bool, seed: int, fallback_len: int) -> tuple[np.ndarray, np.ndarray]:
    if replay_root is None:
        if not dry_run:
            raise RuntimeError("carry_mode='replay' requires --replay-root unless --dry-run.")
        return _build_synthetic_replay(fallback_len, 16, seed)

    root = Path(replay_root)
    if root.is_dir():
        candidates = sorted(root.glob("*.json")) + sorted(root.glob("*.npz")) + sorted(root.glob("*.npy")) + sorted(root.glob("*.pt"))
        if not candidates:
            if not dry_run:
                raise RuntimeError(f"No replay file found in {root}")
            return _build_synthetic_replay(fallback_len, 16, seed)
        root = candidates[0]

    if not root.exists():
        if dry_run:
            return _build_synthetic_replay(fallback_len, 16, seed)
        raise RuntimeError(f"Replay file '{root}' does not exist.")

    if root.suffix == ".json":
        with open(root, "r", encoding="utf-8") as f:
            payload = json.load(f)
        images = payload.get("images")
        proprio = payload.get("proprio")
        if images is None or proprio is None:
            raise RuntimeError(f"Replay JSON must contain 'images' and 'proprio': {root}")
        images_a = np.asarray(images, dtype=np.uint8)
        proprio_a = np.asarray(proprio, dtype=np.float32)
    elif root.suffix == ".npz":
        obj = np.load(root, allow_pickle=True)
        images_a = np.asarray(obj["images"], dtype=np.uint8) if "images" in obj.files else None
        proprio_a = np.asarray(obj["proprio"], dtype=np.float32) if "proprio" in obj.files else None
        if images_a is None or proprio_a is None:
            raise RuntimeError(f".npz replay needs keys images/proprio: {root}")
    elif root.suffix == ".npy":
        payload = np.load(root, allow_pickle=True).item()
        if not isinstance(payload, dict):
            raise RuntimeError(f".npy replay should contain dict(images, proprio): {root}")
        images_a = np.asarray(payload.get("images"), dtype=np.uint8)
        proprio_a = np.asarray(payload.get("proprio"), dtype=np.float32)
    else:
        payload = torch.load(root, map_location="cpu")
        if isinstance(payload, dict):
            raw_images = payload.get("images")
            raw_proprio = payload.get("proprio")
            if raw_images is None or raw_proprio is None:
                raise RuntimeError(f"Replay .pt payload must include images/proprio keys: {root}")
            images_a = np.asarray(raw_images, dtype=np.uint8)
            proprio_a = np.asarray(raw_proprio, dtype=np.float32)
        else:
            raise RuntimeError(f"Replay .pt payload unsupported shape for {root}")
        if images_a.size == 0 or proprio_a.size == 0:
            raise RuntimeError(f"Replay payload missing expected keys at {root}")

    if images_a.ndim != 4 or proprio_a.ndim != 2:
        raise RuntimeError(f"Replay arrays must be [T,H,W,C] and [T,D], got {images_a.shape}/{proprio_a.shape}")
    return images_a, proprio_a


def _load_goal_latent(
    source: str | None,
    wm_bundle: WMBundle | None,
    fallback_image: np.ndarray | None,
    fallback_proprio: np.ndarray | None,
    *,
    goal_frame: np.ndarray | None = None,
    wm_scoring_latent: str = "visual",
    wm_goal_flip_horizontal: bool = True,
    wm_goal_debug_path: Path | str | None = None,
) -> tuple[Any | None, str | None]:
    """
    Resolve a goal latent from a source file with fallback to current state latent.

    Returns (latent_or_none, path_str_if_debug_png_written).
    """
    debug_written: str | None = None
    source = (source or "").strip()
    if source and Path(source).exists():
        p = Path(source)
        payload_obj: Any = None
        if p.suffix == ".json":
            with open(p, "r", encoding="utf-8") as f:
                payload_obj = json.load(f)
        elif p.suffix == ".npy":
            payload_obj = np.load(p, allow_pickle=True)
        elif p.suffix == ".npz":
            payload_obj = np.load(p, allow_pickle=True)
        else:
            payload_obj = torch.load(p, map_location="cpu")

        if isinstance(payload_obj, dict) and "latent" in payload_obj:
            return (
                torch.tensor(np.asarray(payload_obj["latent"], dtype=np.float32).reshape(-1), dtype=torch.float32),
                None,
            )

        if isinstance(payload_obj, dict) and "image" in payload_obj and "proprio" in payload_obj and wm_bundle is not None:
            img = _prepare_goal_image_for_wm(payload_obj["image"], flip_horizontal=wm_goal_flip_horizontal)
            proprio = np.asarray(payload_obj["proprio"], dtype=np.float32)
            if wm_goal_debug_path is not None:
                _write_wm_goal_encode_debug(img, wm_goal_debug_path)
                debug_written = str(Path(wm_goal_debug_path).resolve())
            return _encode_state_to_latent(wm_bundle, img, proprio, wm_scoring_latent=wm_scoring_latent), debug_written

    if wm_bundle is not None and goal_frame is not None:
        proprio = (
            np.asarray(fallback_proprio, dtype=np.float32).reshape(-1)
            if fallback_proprio is not None
            else np.zeros(int(wm_bundle.proprio_dim), dtype=np.float32)
        )
        prepared = _prepare_goal_image_for_wm(goal_frame, flip_horizontal=wm_goal_flip_horizontal)
        if wm_goal_debug_path is not None:
            _write_wm_goal_encode_debug(prepared, wm_goal_debug_path)
            debug_written = str(Path(wm_goal_debug_path).resolve())
        return _encode_state_to_latent(wm_bundle, prepared, proprio, wm_scoring_latent=wm_scoring_latent), debug_written

    if wm_bundle is not None and fallback_image is not None and fallback_proprio is not None:
        return _encode_state_to_latent(wm_bundle, fallback_image, fallback_proprio, wm_scoring_latent=wm_scoring_latent), None
    return None, None


def _encode_state_to_latent(
    bundle: WMBundle, image: np.ndarray, proprio: np.ndarray, wm_scoring_latent: str = "visual"
) -> torch.Tensor:
    _require_torch("WM encoding requires torch.")
    with torch.no_grad():
        obs = {"visual": _to_wm_visual(image, bundle.device), "proprio": _to_wm_proprio(proprio, bundle.proprio_dim, bundle.device)}
        z = bundle.model.encode(obs)
    return _extract_scoring_latent(z, mode=wm_scoring_latent).reshape(-1)


def _reset_env(env: Any, seed: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    try:
        out = env.reset(seed=seed)
    except TypeError:
        out = env.reset()
    if isinstance(out, tuple):
        obs = out[0]
        info = out[1] if len(out) > 1 and isinstance(out[1], dict) else {}
    else:
        obs = out
        info = {}
    image, proprio = _extract_image_and_proprio(obs, env)
    return image, proprio, info


def _step_env(env: Any, action: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any], bool]:
    out = env.step(np.asarray(action, dtype=np.float32))
    if len(out) == 5:
        obs, _reward, terminated, truncated, info = out
        done = bool(terminated or truncated)
    else:
        obs, _reward, done, info = out
        done = bool(done)
    image, proprio = _extract_image_and_proprio(obs, env)
    if not isinstance(info, dict):
        info = {}
    return image, proprio, info, done


def _take_action_for_env(action: np.ndarray, env_action_dim: int) -> np.ndarray:
    return _pad_or_truncate(np.asarray(action, dtype=np.float32), env_action_dim)


def update_grpo_step(
    candidates: Sequence[ChunkCandidate],
    *,
    train_steps: int,
    seed: int,
    device: Any,
    adapter: Any = None,
) -> Any:
    _require_torch("GRPO update requires torch.")
    """
    Optional lightweight adapter update from candidate chunk scores.
    Uses deterministic weighted MSE on -score targets.
    """
    if train_steps <= 0:
        return adapter

    if not candidates:
        return adapter

    valid: list[ChunkCandidate] = [c for c in candidates if np.isfinite(c.score)]
    if len(valid) < 2:
        return adapter

    x_np = np.stack([np.asarray(c.actions, dtype=np.float32).reshape(-1) for c in valid], axis=0)
    # convert score where larger is better into a [0,1] target
    y_np = np.array([c.score for c in valid], dtype=np.float32)
    y_np -= float(np.min(y_np))
    max_y = float(np.max(y_np))
    if max_y > 0.0:
        y_np = y_np / max_y
    else:
        y_np = np.zeros_like(y_np)

    if adapter is None:
        torch.manual_seed(int(seed))
        np.random.seed(int(seed) % (2**32 - 1))
        input_dim = int(x_np.shape[1])
        adapter = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        ).to(device=device)

    x = torch.as_tensor(x_np, device=device, dtype=torch.float32)
    y = torch.as_tensor(y_np, device=device, dtype=torch.float32)
    weights = torch.softmax(y.detach(), dim=0)
    opt = torch.optim.Adam(adapter.parameters(), lr=1.0e-3)
    for _ in range(int(train_steps)):
        pred = adapter(x).view(-1)
        loss = torch.mean(weights * (pred - y.detach()) ** 2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return adapter


def rollout_with_chunks(
    smolvla_bundle: Any,
    wm_bundle: WMBundle | None,
    *,
    task: str,
    episode_index: int,
    chunk_len: int,
    num_candidates: int,
    max_steps: int,
    carry_mode: CarryMode,
    replay_root: str | Path | None = None,
    goal_latent_source: str | None = None,
    goal_frame: np.ndarray | None = None,
    start_frame: np.ndarray | None = None,
    goal_frame_index: int | None = None,
    goal_source: str | None = None,
    comparison_root: Path | str | None = None,
    reset_frame_warning_threshold: float = 0.08,
    seed: int = 0,
    train_steps: int = 0,
    dry_run: bool = False,
    strict_wm_scoring: bool = False,
    strict_decode: bool = False,
    adapter: nn.Module | None = None,
    wm_rollout_mode: str = "iterative",
    wm_scoring_latent: str = "visual",
    wm_goal_flip_horizontal: bool = True,
    wm_sim_camera_parity: bool = True,
    wm_sim_img_size: int = 224,
    smolvla_policy_hflip_corner2: bool = True,
    smolvla_noise_std: float = 0.0,
    smolvla_n_action_steps: int = 1,
    comparison_strip_overlay: bool = False,
    comparison_strip_stitch_gutter_pixels: int = 0,
    oracle_action_sequence: np.ndarray | None = None,
    oracle_action_source: str | None = None,
) -> tuple[EpisodeLog, nn.Module | None]:
    """
    Run one segment-level GRPO rollout.

    - sample K action chunks of length A from current (image, proprio),
    - score each chunk by JEPA-WM latent distance-to-goal,
    - execute selected chunk prefix of A actions,
    - carry simulation forward either by env stepping (sim) or replay index jumps (replay).
    """
    if chunk_len <= 0:
        raise ValueError("chunk_len must be > 0")
    if num_candidates <= 0:
        raise ValueError("num_candidates must be > 0")
    if max_steps <= 0:
        raise ValueError("max_steps must be > 0")
    carry_mode = str(carry_mode).strip().lower()
    if carry_mode not in ("sim", "replay"):
        raise ValueError("carry_mode must be 'sim' or 'replay'")

    if oracle_action_sequence is not None:
        if int(num_candidates) != 1:
            raise ValueError("oracle_action_sequence requires num_candidates == 1")
        _ora = np.asarray(oracle_action_sequence, dtype=np.float32)
        if _ora.ndim != 2 or _ora.shape[0] < 1:
            raise ValueError(f"oracle_action_sequence must be 2-D (T, A) with T>=1; got shape {_ora.shape}")

    rng = np.random.default_rng(seed)
    from smolvla_pipeline.evaluator import _resolve_task_text  # noqa: PLC0415

    task_text = _resolve_task_text(task)
    jepa_parity_sim = False
    _render_jepa_rgb = None

    if carry_mode == "replay":
        images, proprio_seq = _load_replay_root(replay_root, dry_run=dry_run, seed=seed, fallback_len=max_steps + chunk_len)
        env = None
        env_action_dim = wm_bundle.planner_action_dim if wm_bundle is not None else 4
        replay_idx = 0
        current_image = images[0]
        current_proprio = proprio_seq[0]
        env_done = False
    else:
        env = None
        env_done = False
        env_action_dim = 4
        proprio_dim = 16
        if not dry_run:
            try:
                import metaworld

                os.environ.setdefault("MUJOCO_GL", "egl")
                if wm_sim_camera_parity:
                    from metaworld_jepa_render import build_jepa_metaworld_env, render_jepa_rgb

                    env, train_tasks = build_jepa_metaworld_env(task, img_size=int(wm_sim_img_size), seed=seed)
                    if train_tasks:
                        env.set_task(train_tasks[int(episode_index) % len(train_tasks)])
                    _, current_proprio, _ = _reset_env(env, seed=seed)
                    _render_jepa_rgb = render_jepa_rgb
                    current_image = _render_jepa_rgb(env)
                    jepa_parity_sim = True
                else:
                    mt1 = metaworld.MT1(task)
                    env_cls = mt1.train_classes[task]
                    try:
                        env = env_cls(render_mode="rgb_array")
                    except TypeError:
                        env = env_cls()
                        try:
                            if hasattr(env, "render_mode"):
                                env.render_mode = "rgb_array"
                        except Exception as exc:
                            print(f"[segment_grpo] failed to set env render_mode on simulation env: {exc}")
                    tasks = getattr(mt1, "train_tasks", None)
                    if tasks:
                        env.set_task(tasks[int(episode_index) % len(tasks)])
                    current_image, current_proprio, _ = _reset_env(env, seed=seed)
                env_action_dim = int(np.prod(env.action_space.shape))
            except Exception as exc:
                if not dry_run:
                    raise RuntimeError(
                        "carry_mode='sim' requires metaworld. Use --dry-run to skip env dependencies."
                    ) from exc

        if dry_run:
            images, proprio_seq = _build_synthetic_replay(max_steps + chunk_len, proprio_dim, seed)
            env = None
            env_action_dim = wm_bundle.planner_action_dim if wm_bundle is not None else 4
            replay_idx = 0
            current_image = images[0]
            current_proprio = proprio_seq[0]
            env_done = False

    comparison_root_path = Path(comparison_root) if comparison_root is not None else None
    episode_strip_parts: list[Path] = []
    start_frame_similarity = None
    reset_frame_warning = False
    if start_frame is not None:
        try:
            # Oracle start_frame is V+H vs raw; jepa-parity live RGB is V-only on raw → H-flip oracle for compare.
            sf = np.asarray(start_frame)
            if wm_goal_flip_horizontal and jepa_parity_sim:
                sf = np.flip(sf, axis=1)
            start_frame_similarity = _frame_similarity(current_image, sf)
            if start_frame_similarity > float(reset_frame_warning_threshold):
                reset_frame_warning = True
                print(
                    f"[segment_grpo] reset frame mismatch for episode {episode_index}: "
                    f"distance={start_frame_similarity:.6f}, threshold={reset_frame_warning_threshold:.3f}"
                )
        except Exception as exc:
            print(f"[segment_grpo] reset frame compare failed for episode {episode_index}: {exc}")

    if smolvla_bundle is None:
        if not dry_run and oracle_action_sequence is None:
            raise RuntimeError("SmolVLA bundle is required when not in dry-run mode.")
        smolvla_bundle = None

    wm_goal_debug_path: Path | None = None
    if comparison_root is not None and wm_bundle is not None:
        wm_goal_debug_path = Path(comparison_root) / f"episode_{episode_index:04d}" / "wm_goal_for_encode.png"

    wm_goal_encode_dbg_path: str | None = None
    if wm_bundle is not None:
        goal_latent, wm_goal_encode_dbg_path = _load_goal_latent(
            goal_latent_source,
            wm_bundle,
            fallback_image=current_image,
            fallback_proprio=current_proprio,
            goal_frame=goal_frame,
            wm_scoring_latent=wm_scoring_latent,
            wm_goal_flip_horizontal=wm_goal_flip_horizontal,
            wm_goal_debug_path=wm_goal_debug_path,
        )
    else:
        goal_latent = None
    if wm_bundle is not None and goal_latent is None and not dry_run:
        goal_latent = _encode_state_to_latent(wm_bundle, current_image, current_proprio, wm_scoring_latent=wm_scoring_latent)

    planner_action_dim = wm_bundle.planner_action_dim if wm_bundle is not None else int(env_action_dim)

    episode_log = EpisodeLog(
        episode_index=episode_index,
        task=task,
        carry_mode=carry_mode,
        chunk_len=chunk_len,
        num_candidates=num_candidates,
        max_steps=max_steps,
        goal_frame_index=(int(goal_frame_index) if goal_frame_index is not None else None),
        goal_source=goal_source,
        start_frame_similarity=start_frame_similarity,
        reset_frame_warning=bool(reset_frame_warning),
        metadata={
            "smolvla_task_text": task_text,
            "dry_run": dry_run,
            "wm_loaded": wm_bundle is not None,
            "goal_latent_loaded": goal_latent is not None,
            "goal_frame_index": int(goal_frame_index) if goal_frame_index is not None else None,
            "goal_source": goal_source,
            "carry_mode": carry_mode,
            "train_steps": int(train_steps),
            "wm_rollout_mode": str(wm_rollout_mode),
            "wm_goal_flip_horizontal": bool(wm_goal_flip_horizontal),
            "wm_sim_camera_parity": bool(wm_sim_camera_parity),
            "wm_sim_img_size": int(wm_sim_img_size),
            "smolvla_policy_hflip_corner2": bool(smolvla_policy_hflip_corner2),
            "smolvla_noise_std": float(smolvla_noise_std),
            "smolvla_n_action_steps_requested": int(smolvla_n_action_steps),
            "policy_chunk_api_enabled": True,
            "comparison_strip_overlay": bool(comparison_strip_overlay),
            "comparison_strip_stitch_gutter_pixels": int(comparison_strip_stitch_gutter_pixels),
            "wm_goal_for_encode_path": wm_goal_encode_dbg_path,
            "strict_wm_scoring": bool(strict_wm_scoring),
            "strict_decode": bool(strict_decode),
            "wm_scoring_latent": str(wm_scoring_latent),
            "wm_scoring_statuses": [],
            "decode_statuses": [],
            "scoring_failure_reasons": [],
            "decode_failure_reasons": [],
            "oracle_action_mode": oracle_action_sequence is not None,
            "oracle_action_n_steps": (
                int(np.asarray(oracle_action_sequence).shape[0]) if oracle_action_sequence is not None else None
            ),
            "oracle_action_source": oracle_action_source,
        },
    )

    current_policy_image = _derive_policy_rgb_for_smolvla(
        current_image,
        jepa_parity_sim=jepa_parity_sim,
        policy_hflip_corner2=smolvla_policy_hflip_corner2,
    )

    current_step = 0
    while current_step < max_steps and not env_done:
        segment_idx = len(episode_log.segments)
        segment_start = current_step
        effective_len = min(chunk_len, max_steps - current_step)

        segment_candidates: list[ChunkCandidate] = []
        candidate_traces: dict[int, DecodeTrace | None] = {}
        candidate_score_traces: dict[int, ScoreTrace | None] = {}
        segment_real_frames: list[np.ndarray] = [np.asarray(current_image, copy=True)]
        for candidate_idx in range(num_candidates):
            chunk_meta: dict[str, Any] = {}
            if oracle_action_sequence is not None:
                ora = np.asarray(oracle_action_sequence, dtype=np.float32)
                slice_end = int(current_step) + int(effective_len)
                if slice_end > int(ora.shape[0]):
                    raise RuntimeError(
                        f"oracle_action_sequence exhausted at env step {current_step}: "
                        f"need rows [:{slice_end}], have T={ora.shape[0]}"
                    )
                oracle_slice = np.asarray(ora[int(current_step) : slice_end], dtype=np.float32)
                if int(oracle_slice.shape[1]) != int(env_action_dim):
                    raise ValueError(
                        f"oracle_action_sequence width {oracle_slice.shape[1]} != env_action_dim {env_action_dim}"
                    )
                chunk = oracle_slice
                chunk_meta = {
                    "chunk_source": "oracle",
                    "oracle_start_step": int(current_step),
                }
                if oracle_action_source:
                    chunk_meta["oracle_action_source"] = str(oracle_action_source)
            elif smolvla_bundle is not None and not dry_run:
                chunk, chunk_meta = _sample_smolvla_chunk(
                    smolvla_bundle,
                    current_policy_image,
                    current_proprio,
                    effective_len,
                    int(env_action_dim),
                    task_text,
                    rng,
                    noise_std=float(smolvla_noise_std),
                )
            else:
                chunk = _synthetic_chunk(int(env_action_dim), effective_len, candidate_idx, rng)

            # Score by latent distance when possible; fallback to deterministic synthetic score otherwise.
            latent_trace: list[np.ndarray] | None = None
            scoring_failure_reason: str | None = None
            wm_scoring_status = "fallback"
            wm_stride_meta = 1
            score_trace: ScoreTrace | None = None
            if wm_bundle is not None and goal_latent is not None:
                try:
                    distance, score_trace, decode_trace = score_chunk_by_goal_latent(
                        wm_bundle,
                        current_image,
                        current_proprio,
                        chunk,
                        goal_latent,
                        chunk_len=effective_len,
                        return_latent_trace=True,
                        wm_rollout_mode=wm_rollout_mode,
                        wm_scoring_latent=wm_scoring_latent,
                    )
                    score_trace.selected_candidate_index = int(candidate_idx)
                    decode_trace.selected_candidate_index = int(candidate_idx)
                    score = -distance
                    wm_scoring_status = "ok"
                    wm_stride_meta = int(getattr(decode_trace, "env_steps_per_wm_step", 1) or 1)
                except Exception as exc:
                    if strict_wm_scoring:
                        raise
                    distance, scoring_failure_reason = _fallback_scoring_distance(
                        chunk,
                        f"score_chunk_by_goal_latent failed ({type(exc).__name__}: {exc})",
                    )
                    score = -distance
                    decode_trace = None
                    score_trace = None
            else:
                wm_scoring_status = "fallback"
                scoring_failure_reason = (
                    "WM scoring skipped because bundle/goal latent was unavailable; synthetic fallback score used."
                )
                chunk_flat = np.asarray(chunk, dtype=np.float32).reshape(-1)
                distance = float(np.linalg.norm(chunk_flat))
                score = -distance
                decode_trace = None
                score_trace = None

            candidate_traces[int(candidate_idx)] = decode_trace
            candidate_score_traces[int(candidate_idx)] = score_trace

            segment_candidates.append(
                ChunkCandidate(
                    index=candidate_idx,
                    actions=chunk.astype(np.float32),
                    score=float(score),
                    latent_distance=float(distance),
                    meta={
                        **chunk_meta,
                        "planner_action_dim": int(planner_action_dim),
                        "env_action_dim": int(env_action_dim),
                        "effective_chunk_len": int(effective_len),
                        "wm_scoring_status": wm_scoring_status,
                        "wm_env_steps_per_wm_step": int(wm_stride_meta),
                        "decode_status": "skipped",
                        "scoring_failure_reason": str(scoring_failure_reason) if scoring_failure_reason is not None else None,
                        "latent_trace_len": 0
                        if decode_trace is None
                        else int(
                            len(decode_trace.visual_latents)
                            if decode_trace.visual_latents
                            else len(decode_trace.proprio_latents)
                        ),
                        **({"failure_reason": str(scoring_failure_reason)} if scoring_failure_reason is not None else {}),
                    },
                )
            )

        nonfinite_penalty = -1.0e12
        best = max(
            segment_candidates,
            key=lambda c: (
                float(c.score) if np.isfinite(c.score) else nonfinite_penalty,
                -int(c.index),
            ),
        )
        selected_idx = int(best.index)
        selected_distance = best.latent_distance if best.latent_distance is not None else None
        best_actions = _ensure_action_matrix(best.actions, int(env_action_dim), effective_len)
        selected_trace = candidate_traces.get(selected_idx)
        selected_score_trace = candidate_score_traces.get(selected_idx)
        selected_meta = segment_candidates[selected_idx].meta if 0 <= selected_idx < len(segment_candidates) else None
        pred_frames: list[np.ndarray] = []

        executed_actions: list[list[float]] = []
        segment_done = False
        carried_steps = 0

        if carry_mode == "sim" and env is not None:
            for i in range(effective_len):
                action_env = _take_action_for_env(best_actions[i], int(env_action_dim))
                _obs_img, current_proprio, _info, step_done = _step_env(env, action_env)
                if jepa_parity_sim and _render_jepa_rgb is not None:
                    current_image = _render_jepa_rgb(env)
                else:
                    current_image = _obs_img
                current_policy_image = _derive_policy_rgb_for_smolvla(
                    current_image,
                    jepa_parity_sim=jepa_parity_sim,
                    policy_hflip_corner2=smolvla_policy_hflip_corner2,
                )
                carried_steps += 1
                current_step += 1
                executed_actions.append(action_env.tolist())
                segment_real_frames.append(np.asarray(current_image, copy=True))
                if step_done:
                    segment_done = True
                    env_done = True
                    break
        elif carry_mode == "sim":
            # dry-run without env: replay synthetic stream.
            target_idx = min(replay_idx + effective_len, len(proprio_seq) - 1)
            if replay_idx < target_idx:
                actual = target_idx - replay_idx
                step_start = int(replay_idx)
                replay_idx = target_idx
                current_image = images[replay_idx]
                current_proprio = proprio_seq[replay_idx]
                current_policy_image = _derive_policy_rgb_for_smolvla(
                    current_image,
                    jepa_parity_sim=jepa_parity_sim,
                    policy_hflip_corner2=smolvla_policy_hflip_corner2,
                )
                carried_steps = int(actual)
                current_step += actual
                segment_real_frames.extend(np.asarray(images[i], copy=True) for i in range(step_start + 1, replay_idx + 1))
                executed_actions.extend(
                    [_take_action_for_env(best_actions[i], int(env_action_dim)).tolist() for i in range(actual)]
                )
                if replay_idx >= len(proprio_seq) - 1:
                    env_done = True
            else:
                segment_done = True
                env_done = True
        else:
            target_idx = min(replay_idx + effective_len, len(proprio_seq) - 1)
            actual = target_idx - replay_idx
            carried_steps = int(actual)
            if actual <= 0:
                segment_done = True
                env_done = True
            else:
                step_start = int(replay_idx)
                replay_idx = target_idx
                current_image = images[replay_idx]
                current_proprio = proprio_seq[replay_idx]
                current_policy_image = _derive_policy_rgb_for_smolvla(
                    current_image,
                    jepa_parity_sim=jepa_parity_sim,
                    policy_hflip_corner2=smolvla_policy_hflip_corner2,
                )
                current_step += actual
                segment_real_frames.extend(np.asarray(images[i], copy=True) for i in range(step_start + 1, replay_idx + 1))
                executed_actions.extend(
                    [_take_action_for_env(best_actions[i], int(env_action_dim)).tolist() for i in range(actual)]
                )
                if replay_idx >= len(proprio_seq) - 1:
                    segment_done = True
                    env_done = True

        if wm_bundle is not None and selected_trace is not None:
            pred_frames, decode_failure_reason = _decode_latent_trace_to_frames(wm_bundle, selected_trace)
            if decode_failure_reason is not None and 0 <= selected_idx < len(segment_candidates):
                if strict_decode:
                    raise RuntimeError(
                        f"Decode failed for selected candidate {selected_idx}: {decode_failure_reason}"
                    )
                segment_candidates[int(selected_idx)].meta["decode_status"] = "failed"
                segment_candidates[int(selected_idx)].meta["decode_failure_reason"] = decode_failure_reason
                segment_candidates[int(selected_idx)].meta["failure_reason"] = decode_failure_reason
                candidate_traces[int(selected_idx)] = None
                pred_frames = []
            elif decode_failure_reason is None:
                segment_candidates[int(selected_idx)].meta["decode_status"] = "ok"
        elif wm_bundle is not None and selected_trace is None and selected_meta is not None:
            decode_failure_reason = "No latent trace available for selected candidate."
            if strict_decode:
                raise RuntimeError(
                    f"Decode failed for selected candidate {selected_idx}: {decode_failure_reason}"
                )
            selected_failure_reason = selected_meta.get("failure_reason")
            selected_meta["decode_status"] = "failed"
            selected_meta["decode_failure_reason"] = "No latent trace available for selected candidate."
            if isinstance(selected_failure_reason, str):
                selected_meta["failure_reason"] = selected_failure_reason
            else:
                selected_meta["failure_reason"] = "No latent trace available for selected candidate."

        selected_trace_len = 0
        if pred_frames:
            selected_trace_len = len(pred_frames)
        elif selected_trace is not None:
            selected_trace_len = (
                len(selected_trace.visual_latents) if selected_trace.visual_latents else len(selected_trace.proprio_latents)
            )
        if selected_meta is not None:
            selected_meta["latent_trace_len"] = int(selected_trace_len)

        if selected_meta is not None and "decode_status" not in selected_meta:
            selected_meta["decode_status"] = "skipped"
        if selected_meta is not None and "selected" not in selected_meta:
            selected_meta["selected"] = True

        selected_meta = segment_candidates[selected_idx].meta if 0 <= selected_idx < len(segment_candidates) else None
        selected_wm_scoring_status = "fallback"
        selected_decode_status = "skipped"
        if selected_meta is not None:
            selected_wm_scoring_status = str(selected_meta.get("wm_scoring_status", "fallback"))
            selected_decode_status = str(selected_meta.get("decode_status", "skipped"))
            if selected_meta.get("scoring_failure_reason"):
                episode_log.metadata["scoring_failure_reasons"].append(str(selected_meta.get("scoring_failure_reason")))
            if selected_meta.get("decode_failure_reason"):
                episode_log.metadata["decode_failure_reasons"].append(str(selected_meta.get("decode_failure_reason")))
            episode_log.metadata["wm_scoring_statuses"].append(selected_wm_scoring_status)
            episode_log.metadata["decode_statuses"].append(selected_decode_status)

        segment_metadata: dict[str, Any] = {
            "wm_scoring_status": selected_wm_scoring_status,
            "decode_status": selected_decode_status,
            "scoring_failure_reason": selected_meta.get("scoring_failure_reason") if selected_meta is not None else None,
            "decode_failure_reason": selected_meta.get("decode_failure_reason") if selected_meta is not None else None,
            "selected": True,
        }

        wm_stride = 1
        if selected_trace is not None:
            wm_stride = int(getattr(selected_trace, "env_steps_per_wm_step", 1) or 1)

        wm_footer_lines: list[str] | None = None
        wm_footer_min_lines: int | None = None
        candidate_wm_goal_l2_meta: list[dict[str, Any]] | None = None
        footer_goal_np: np.ndarray | None = None
        if goal_latent is not None:
            try:
                footer_goal_np = np.asarray(
                    goal_latent.detach().cpu().numpy().reshape(-1), dtype=np.float32
                )
            except Exception:
                footer_goal_np = None
        if footer_goal_np is not None and int(num_candidates) > 0:
            wm_lines, wm_meta, wm_step_count = _all_candidates_wm_goal_l2_rows(
                footer_goal_np,
                candidate_score_traces,
                num_candidates=int(num_candidates),
            )
            if wm_lines:
                # goal_L2 block first, then action-range row (read top-to-bottom).
                wm_footer_lines = list(wm_lines) + [
                    _wm_megastep_action_range_footer_line(
                        carried_steps=int(carried_steps),
                        wm_stride=wm_stride,
                        wm_step_count=wm_step_count,
                    )
                ]
                candidate_wm_goal_l2_meta = wm_meta
                wm_footer_min_lines = 2 + int(num_candidates)
                if wm_step_count > 0:
                    segment_metadata["wm_step_count"] = int(wm_step_count)

        segment_comparison_path = None
        if comparison_root_path is not None and pred_frames and carried_steps > 0:
            overlay_goal_np = None
            if comparison_strip_overlay and goal_latent is not None:
                try:
                    overlay_goal_np = np.asarray(
                        goal_latent.detach().cpu().numpy().reshape(-1), dtype=np.float32
                    )
                except Exception:
                    overlay_goal_np = None
            segment_path, segment_strip_failure_reason = _write_comparison_segment_strip(
                comparison_root_path,
                episode_index,
                segment_idx,
                segment_real_frames,
                pred_frames,
                env_step_start=int(segment_start),
                selected_candidate_index=int(selected_idx),
                carried_steps=carried_steps,
                env_steps_per_wm_step=wm_stride if wm_stride > 1 else None,
                wm_env_steps_per_wm_step=wm_stride,
                overlay_decode_meta=bool(comparison_strip_overlay),
                overlay_goal_latent_np=overlay_goal_np,
                overlay_score_trace=selected_score_trace,
                wm_megastep_footer_lines=wm_footer_lines,
                wm_megastep_footer_min_lines=wm_footer_min_lines,
            )
            if segment_path is not None:
                segment_comparison_path = segment_path
                episode_strip_parts.append(segment_path)
            elif segment_strip_failure_reason is not None and selected_meta is not None:
                selected_meta["comparison_strip_status"] = "failed"
                selected_meta["comparison_strip_failure_reason"] = segment_strip_failure_reason

        segment_metadata["comparison_env_step_start"] = int(segment_start)
        segment_metadata["comparison_env_step_end"] = int(segment_start + carried_steps)
        segment_metadata["comparison_wm_env_steps_per_wm_step"] = int(wm_stride)
        if segment_comparison_path is not None:
            segment_metadata["comparison_strip_filename"] = segment_comparison_path.name
        if candidate_wm_goal_l2_meta:
            segment_metadata["candidate_wm_goal_l2_int"] = candidate_wm_goal_l2_meta

        episode_log.actions.extend(executed_actions)
        episode_log.latent_scores.append(0.0 if selected_distance is None else float(selected_distance))
        episode_log.selected_scores.append(float(best.score))
        episode_log.selected_indices.append(selected_idx)
        episode_log.selected_candidate_indices.append(selected_idx)
        episode_log.candidate_distances.append(0.0 if selected_distance is None else float(selected_distance))

        episode_log.segments.append(
            SegmentLog(
                segment_index=segment_idx,
                start_step=segment_start,
                selected_index=selected_idx,
                selected_score=float(best.score),
                latent_distance=selected_distance,
                carried_steps=int(carried_steps),
                carry_mode=carry_mode,
                done=segment_done or env_done,
                candidates=segment_candidates,
                executed_actions=executed_actions,
                comparison_strip_path=str(segment_comparison_path) if segment_comparison_path is not None else None,
                metadata=segment_metadata,
            )
        )

        episode_log.steps = int(current_step)
        episode_log.done = bool(env_done) or current_step >= max_steps

        if train_steps > 0:
            adapter = update_grpo_step(
                segment_candidates,
                train_steps=train_steps,
                seed=seed + segment_idx,
                device=wm_bundle.device if wm_bundle is not None else _resolve_device("cpu"),
                adapter=adapter,
            )

    if comparison_root_path is not None and episode_strip_parts:
        stitched_path, stitched_failure_reason = _stitch_comparison_strip(
            episode_strip_parts,
            comparison_root_path / f"episode_{episode_index:04d}_comparison_strip.png",
            gutter_pixels=int(comparison_strip_stitch_gutter_pixels),
        )
        if stitched_path is not None:
            episode_log.comparison_strip_path = str(stitched_path)
        elif stitched_failure_reason is not None:
            print(f"[segment_grpo] failed to stitch episode comparison strip: {stitched_failure_reason}")

    if env is not None:
        try:
            env.close()
        except Exception as exc:
            print(f"[segment_grpo] failed to close simulation env: {exc}")
    return episode_log, adapter


State = SegmentState
Candidate = ChunkCandidate
Episode = EpisodeLog
