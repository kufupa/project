from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os
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
    tensor = torch.from_numpy(rgb).float() / 255.0
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # 1,3,H,W
    tensor = torch.nn.functional.interpolate(
        tensor, size=(256, 256), mode="bilinear", align_corners=False
    )  # 1,3,256,256
    return tensor.unsqueeze(0).to(device)  # 1,1,3,256,256


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


def _decode_latent_trace_to_frames(model_bundle: WMBundle, latent_trace: list[np.ndarray]) -> list[np.ndarray]:
    if not latent_trace:
        return []
    decode_fn = getattr(model_bundle.model, "decode_unroll", None)
    if decode_fn is None:
        decode_fn = getattr(model_bundle.model, "decode", None)
    if decode_fn is None:
        return []

    try:
        lat = np.asarray(latent_trace, dtype=np.float32)
        lat_t = torch.as_tensor(lat, dtype=torch.float32, device=model_bundle.device)
        if lat_t.ndim == 2:
            lat_t = lat_t.unsqueeze(0)

        decoded = None
        try:
            decoded = decode_fn(lat_t, debug=False)
        except TypeError:
            decoded = decode_fn(lat_t)

        if isinstance(decoded, dict):
            if "recon" in decoded:
                decoded = decoded["recon"]
            elif "decoded" in decoded:
                decoded = decoded["decoded"]
            else:
                decoded = next(iter(decoded.values()))
        decoded_np = np.asarray(decoded)
        if decoded_np.size == 0:
            return []

        if decoded_np.ndim == 5:
            decoded_np = decoded_np[0]
        if decoded_np.ndim == 4:
            return [_to_channel_last(frame) for frame in decoded_np]
        if decoded_np.ndim == 3:
            return [_to_channel_last(decoded_np)]
    except Exception:
        return []
    return []


def _build_real_vs_pred_strip(real_frames: list[np.ndarray], pred_frames: list[np.ndarray]) -> np.ndarray:
    if not real_frames or not pred_frames:
        raise ValueError("Real and predicted frames are required for strip rendering.")
    total = min(len(real_frames), len(pred_frames))
    pairs: list[np.ndarray] = []
    for idx in range(total):
        real_rgb = _to_rgb_uint8(real_frames[idx])
        pred_rgb = _to_rgb_uint8(pred_frames[idx])
        if pred_rgb.shape != real_rgb.shape:
            from PIL import Image

            pred_rgb = np.array(Image.fromarray(pred_rgb).resize((real_rgb.shape[1], real_rgb.shape[0])))
        pairs.append(np.concatenate([real_rgb, pred_rgb], axis=0))
    return np.concatenate(pairs, axis=1)


def _write_comparison_segment_strip(
    out_dir: Path,
    episode_index: int,
    segment_index: int,
    real_frames: list[np.ndarray],
    pred_frames: list[np.ndarray],
) -> Path | None:
    if not real_frames or not pred_frames:
        return None
    out_dir = Path(out_dir)
    try:
        strip = _build_real_vs_pred_strip(real_frames, pred_frames)
    except Exception:
        return None
    import imageio.v2 as imageio

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"episode_{episode_index:04d}_segment_{segment_index:04d}.png"
    imageio.imwrite(path, strip)
    return path


def _stitch_comparison_strip(segments: list[Path], output_path: Path) -> Path | None:
    if not segments:
        return None
    import imageio.v2 as imageio

    strips: list[np.ndarray] = []
    for seg in segments:
        strips.append(imageio.imread(seg))
    if not strips:
        return None
    stitched = np.concatenate(strips, axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(output_path, stitched)
    return output_path


def _extract_image_and_proprio(obs: Any, env: Any) -> tuple[np.ndarray, np.ndarray]:
    module = _load_jepa_helper_module()
    img = module._find_image(obs)
    if img is None and env is not None and hasattr(env, "render"):
        img = env.render()
    if img is None:
        raise RuntimeError("Unable to extract an image from observation; ensure env.render() is available.")
    proprio = module._flatten_obs_state(obs)
    return _to_rgb_uint8(img), np.asarray(proprio, dtype=np.float32).reshape(-1)


def load_smolvla_bundle(checkpoint: str | Path | None, device: str | torch.device | None) -> Any:
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
        bundle = helper._try_load_smolvla_exec(ckpt, _resolve_device(device))
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
    except Exception:
        pass
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
                preferred_keys = ("proprio", "visual", "state", "prediction", "pred", "z", "latent")
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


def _next_latent_state_after_unroll(out: Any) -> Any:
    """Reduce multi-step unroll output to latent state suitable for another one-step unroll."""
    if isinstance(out, dict):
        nxt: dict[str, Any] = {}
        for key, val in out.items():
            if torch is not None and torch.is_tensor(val) and val.dim() >= 1:
                nxt[key] = val[-1:].detach() if int(val.shape[0]) > 1 else val.detach()
            else:
                nxt[key] = val
        return nxt
    if torch is not None and torch.is_tensor(out):
        if out.dim() >= 1 and int(out.shape[0]) > 1:
            return out[-1:].detach()
        return out.detach()
    return out


def _latent_vector_from_unroll_step(unroll_out: Any) -> torch.Tensor:
    """Single-step latent used for distance scoring and trace (flattened 1-D)."""
    _require_torch("WM scoring requires torch.")
    lat = _extract_latent(unroll_out)
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
) -> float | tuple[float, list[np.ndarray]]:
    """
    Score a candidate chunk by JEPA-WM latent distance to goal latent.

    ``wm_rollout_mode``:
    - ``iterative``: one ``model.unroll`` call per action in the chunk (state carried forward).
    - ``batched``: single ``unroll`` over the full action suffix (legacy).

    Returns euclidean distance (smaller is better), and optionally latent trace.
    """
    if not isinstance(chunk_actions, np.ndarray):
        chunk_actions = np.asarray(chunk_actions, dtype=np.float32)
    if chunk_actions.ndim != 2:
        raise RuntimeError(f"chunk_actions must be 2-D, got shape {chunk_actions.shape}")

    model_action_dim = _infer_model_action_dim(wm_bundle.model)
    target_action_dim = wm_bundle.planner_action_dim
    if model_action_dim and model_action_dim != target_action_dim:
        target_action_dim = model_action_dim

    actions = _ensure_action_matrix(
        chunk_actions,
        target_action_dim,
        int(chunk_len or chunk_actions.shape[0]),
    )
    actions_t = torch.from_numpy(actions).to(wm_bundle.device).to(dtype=torch.float32)
    mode = str(wm_rollout_mode or "iterative").strip().lower()
    if mode not in ("iterative", "batched"):
        raise ValueError(f"wm_rollout_mode must be 'iterative' or 'batched', got {wm_rollout_mode!r}")

    with torch.no_grad():
        visual = _to_wm_visual(image, wm_bundle.device)
        proprio_t = _to_wm_proprio(proprio, wm_bundle.proprio_dim, wm_bundle.device)
        latent_state = wm_bundle.model.encode({"visual": visual, "proprio": proprio_t})
        seq_actions = actions_t.to(wm_bundle.device).to(dtype=torch.float32).unsqueeze(1)
        if seq_actions.ndim != 3:
            raise RuntimeError(f"Expected action sequence tensor shape (T, B, A), got {seq_actions.shape}")

        latent_trace: list[torch.Tensor] = []
        if mode == "batched":
            unroll_output = wm_bundle.model.unroll(latent_state, act_suffix=seq_actions, debug=False)
            rollout_trace = _extract_latent(unroll_output)
            if rollout_trace.ndim < 3:
                raise RuntimeError(
                    f"Expected rollout latent trace with time dimension, got shape {tuple(rollout_trace.shape)}"
                )
            latent_trace.extend(rollout_trace)
            pred = rollout_trace[-1].reshape(-1)
        else:
            z_t: Any = latent_state
            for t in range(int(seq_actions.shape[0])):
                one = seq_actions[t : t + 1]
                unroll_out = wm_bundle.model.unroll(z_t, act_suffix=one, debug=False)
                step_vec = _latent_vector_from_unroll_step(unroll_out)
                latent_trace.append(step_vec)
                z_t = _next_latent_state_after_unroll(unroll_out)
            pred = latent_trace[-1].reshape(-1)

        goal = goal_latent.reshape(-1).to(wm_bundle.device)
        if pred.numel() == 0 or goal.numel() == 0:
            raise RuntimeError("Invalid latent shape for scoring.")
        n = min(pred.numel(), goal.numel())
        distance = torch.linalg.vector_norm(pred[:n] - goal[:n], ord=2).item()
    if return_latent_trace:
        decoded_trace = [torch.as_tensor(v).reshape(-1).detach().cpu().numpy() for v in latent_trace]
        return float(distance), decoded_trace
    return float(distance)


def _sample_smolvla_chunk(
    smolvla_bundle: Any,
    image: np.ndarray,
    proprio: np.ndarray,
    chunk_len: int,
    planner_action_dim: int,
    task_text: str,
    rng: np.random.Generator,
) -> np.ndarray:
    helper = _load_jepa_helper_module()
    obs = {"image": _to_rgb_uint8(image), "state": np.asarray(proprio, dtype=np.float32).reshape(-1)}

    class _RenderProxy:
        def __init__(self, frame: np.ndarray):
            self._frame = frame

        def render(self, *args: Any, **kwargs: Any) -> np.ndarray:
            return self._frame

    try:
        base = helper._smolvla_exec_action(smolvla_bundle, obs, _RenderProxy(_to_rgb_uint8(image)), task_text)
        base = np.asarray(base, dtype=np.float32).reshape(-1)
    except Exception as exc:
        raise RuntimeError(f"SmolVLA inference failed for chunk sampling: {exc}") from exc

    chunk: list[np.ndarray] = []
    for _ in range(chunk_len):
        noise = rng.normal(0.0, 0.10, size=planner_action_dim).astype(np.float32)
        action = np.clip(_pad_or_truncate(base, planner_action_dim) + noise, -1.0, 1.0)
        chunk.append(action)
    return np.stack(chunk, axis=0).astype(np.float32)


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
) -> torch.Tensor | None:
    """
    Resolve a goal latent from a source file with fallback to current state latent.
    """
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
            return torch.tensor(np.asarray(payload_obj["latent"], dtype=np.float32).reshape(-1), dtype=torch.float32)

        if isinstance(payload_obj, dict) and "image" in payload_obj and "proprio" in payload_obj and wm_bundle is not None:
            img = np.asarray(payload_obj["image"])
            proprio = np.asarray(payload_obj["proprio"], dtype=np.float32)
            return _encode_state_to_latent(wm_bundle, img, proprio)

    if wm_bundle is not None and goal_frame is not None:
        proprio = (
            np.asarray(fallback_proprio, dtype=np.float32).reshape(-1)
            if fallback_proprio is not None
            else np.zeros(int(wm_bundle.proprio_dim), dtype=np.float32)
        )
        return _encode_state_to_latent(wm_bundle, goal_frame, proprio)

    if wm_bundle is not None and fallback_image is not None and fallback_proprio is not None:
        return _encode_state_to_latent(wm_bundle, fallback_image, fallback_proprio)
    return None


def _encode_state_to_latent(bundle: WMBundle, image: np.ndarray, proprio: np.ndarray) -> torch.Tensor:
    _require_torch("WM encoding requires torch.")
    with torch.no_grad():
        obs = {"visual": _to_wm_visual(image, bundle.device), "proprio": _to_wm_proprio(proprio, bundle.proprio_dim, bundle.device)}
        z = bundle.model.encode(obs)
    return torch.as_tensor(_extract_latent(z)).reshape(-1)


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
    adapter: nn.Module | None = None,
    wm_rollout_mode: str = "iterative",
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

    rng = np.random.default_rng(seed)
    task_text = f"{task}"

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

                mt1 = metaworld.MT1(task)
                env_cls = mt1.train_classes[task]
                os.environ.setdefault("MUJOCO_GL", "egl")
                try:
                    env = env_cls(render_mode="rgb_array")
                except TypeError:
                    env = env_cls()
                    try:
                        if hasattr(env, "render_mode"):
                            env.render_mode = "rgb_array"
                    except Exception:
                        pass
                tasks = getattr(mt1, "train_tasks", None)
                if tasks:
                    env.set_task(tasks[0])
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
            start_frame_similarity = _frame_similarity(current_image, start_frame)
            if start_frame_similarity > float(reset_frame_warning_threshold):
                reset_frame_warning = True
                print(
                    f"[segment_grpo] reset frame mismatch for episode {episode_index}: "
                    f"distance={start_frame_similarity:.6f}, threshold={reset_frame_warning_threshold:.3f}"
                )
        except Exception as exc:
            print(f"[segment_grpo] reset frame compare failed for episode {episode_index}: {exc}")

    if smolvla_bundle is None:
        if not dry_run:
            raise RuntimeError("SmolVLA bundle is required when not in dry-run mode.")
        smolvla_bundle = None

    if wm_bundle is not None:
        goal_latent = _load_goal_latent(
            goal_latent_source,
            wm_bundle,
            fallback_image=current_image,
            fallback_proprio=current_proprio,
            goal_frame=goal_frame,
        )
    else:
        goal_latent = None
    if wm_bundle is not None and goal_latent is None and not dry_run:
        goal_latent = _encode_state_to_latent(wm_bundle, current_image, current_proprio)

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
            "dry_run": dry_run,
            "wm_loaded": wm_bundle is not None,
            "goal_latent_loaded": goal_latent is not None,
            "goal_frame_index": int(goal_frame_index) if goal_frame_index is not None else None,
            "goal_source": goal_source,
            "carry_mode": carry_mode,
            "train_steps": int(train_steps),
            "wm_rollout_mode": str(wm_rollout_mode),
        },
    )

    current_step = 0
    while current_step < max_steps and not env_done:
        segment_idx = len(episode_log.segments)
        segment_start = current_step
        effective_len = min(chunk_len, max_steps - current_step)

        segment_candidates: list[ChunkCandidate] = []
        candidate_traces: dict[int, list[np.ndarray] | None] = {}
        segment_real_frames: list[np.ndarray] = [np.asarray(current_image, copy=True)]
        for candidate_idx in range(num_candidates):
            if smolvla_bundle is not None and not dry_run:
                chunk = _sample_smolvla_chunk(
                    smolvla_bundle,
                    current_image,
                    current_proprio,
                    effective_len,
                    planner_action_dim,
                    task_text,
                    rng,
                )
            else:
                chunk = _synthetic_chunk(planner_action_dim, effective_len, candidate_idx, rng)

            # Score by latent distance when possible; fallback to deterministic synthetic score otherwise.
            latent_trace: list[np.ndarray] | None = None
            if wm_bundle is not None and goal_latent is not None:
                try:
                    distance, latent_trace = score_chunk_by_goal_latent(
                        wm_bundle,
                        current_image,
                        current_proprio,
                        chunk,
                        goal_latent,
                        chunk_len=effective_len,
                        return_latent_trace=True,
                        wm_rollout_mode=wm_rollout_mode,
                    )
                    score = -distance
                except Exception:
                    chunk_flat = np.asarray(chunk, dtype=np.float32).reshape(-1)
                    distance = float(np.linalg.norm(chunk_flat))
                    score = -distance
                    latent_trace = None
            else:
                chunk_flat = np.asarray(chunk, dtype=np.float32).reshape(-1)
                distance = float(np.linalg.norm(chunk_flat))
                score = -distance
                latent_trace = None

            candidate_traces[int(candidate_idx)] = latent_trace

            segment_candidates.append(
                ChunkCandidate(
                    index=candidate_idx,
                    actions=chunk.astype(np.float32),
                    score=float(score),
                    latent_distance=float(distance),
                    meta={
                        "planner_action_dim": int(planner_action_dim),
                        "effective_chunk_len": int(effective_len),
                        "latent_trace_len": 0 if latent_trace is None else int(len(latent_trace)),
                    },
                )
            )

        best = max(segment_candidates, key=lambda c: (float(c.score), -int(c.index)))
        selected_idx = int(best.index)
        selected_distance = best.latent_distance if best.latent_distance is not None else None
        best_actions = _ensure_action_matrix(best.actions, planner_action_dim, effective_len)
        selected_trace = candidate_traces.get(selected_idx)
        pred_frames: list[np.ndarray] = []

        executed_actions: list[list[float]] = []
        segment_done = False
        carried_steps = 0

        if carry_mode == "sim" and env is not None:
            for i in range(effective_len):
                action_env = _take_action_for_env(best_actions[i], int(env_action_dim))
                current_image, current_proprio, _info, step_done = _step_env(env, action_env)
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
                current_step += actual
                segment_real_frames.extend(np.asarray(images[i], copy=True) for i in range(step_start + 1, replay_idx + 1))
                executed_actions.extend(
                    [_take_action_for_env(best_actions[i], int(env_action_dim)).tolist() for i in range(actual)]
                )
                if replay_idx >= len(proprio_seq) - 1:
                    segment_done = True
                    env_done = True

        if wm_bundle is not None and selected_trace is not None:
            pred_frames = _decode_latent_trace_to_frames(wm_bundle, selected_trace)
        if comparison_root_path is not None and selected_trace is not None:
            segment_path = _write_comparison_segment_strip(
                comparison_root_path, episode_index, segment_idx, segment_real_frames, pred_frames
            )
            if segment_path is not None:
                episode_strip_parts.append(segment_path)

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
        episode_log.comparison_strip_path = str(
            _stitch_comparison_strip(
                episode_strip_parts,
                comparison_root_path / f"episode_{episode_index:04d}_comparison_strip.png",
            )
            or comparison_root_path / f"episode_{episode_index:04d}_comparison_strip.png"
        )

    if env is not None:
        try:
            env.close()
        except Exception:
            pass
    return episode_log, adapter


State = SegmentState
Candidate = ChunkCandidate
Episode = EpisodeLog
