#!/usr/bin/env python3
"""JEPA-WM smoke check for Meta-World rollout compatibility."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback

def _safe_readable_file(path: str) -> str:
    p = Path(path)
    try:
        return str(p.resolve())
    except Exception:
        return str(p)


def _resolve_checkpoint(ckpt_hint: str) -> str:
    if not ckpt_hint:
        return "jepa_wm_metaworld.pth.tar"
    maybe = Path(ckpt_hint)
    if maybe.is_file():
        return _safe_readable_file(str(maybe))

    hf_home = Path.home() / ".cache" / "huggingface"
    hub_cache = hf_home / "hub"
    if not hub_cache.exists():
        return ckpt_hint
    matches = sorted(hub_cache.rglob(ckpt_hint))
    if matches:
        return _safe_readable_file(matches[0])
    return ckpt_hint


def _infer_action_dims(model, preprocessor) -> list[int]:
    candidates: list[int] = []

    def _add(value: object) -> None:
        try:
            dim = int(value)
        except Exception:
            return
        if dim > 0 and dim not in candidates:
            candidates.append(dim)

    _add(getattr(preprocessor, "action_mean").numel())

    model_module = getattr(model, "model", None)
    if model_module is not None:
        _add(getattr(model_module, "action_dim", None))

        predictor = getattr(model_module, "predictor", None)
        for obj_name in ("action_encoder",):
            encoder = getattr(model_module, obj_name, None)
            if encoder is not None:
                _add(getattr(encoder, "in_features", None))
            if predictor is not None:
                enc = getattr(predictor, obj_name, None)
                if enc is not None:
                    _add(getattr(enc, "in_features", None))

    if not candidates:
        candidates = [4]
    return candidates


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--repo", default="facebook/jepa-wms", help="torch.hub repo path")
  parser.add_argument("--ckpt", default="jepa_wm_metaworld.pth.tar", help="Unused default checkpoint name")
  parser.add_argument("--task", default="push-v3", help="Task label")
  parser.add_argument("--pretrained", action="store_true", default=False, help="Use remote pretrained checkpoint from torch.hub model registry")
  parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device for smoke check")
  parser.add_argument("--smoke-steps", type=int, default=6, dest="smoke_steps")
  args = parser.parse_args()

  out = {
    "repo": args.repo,
    "task": args.task,
    "smoke_steps": args.smoke_steps,
    "status": "unknown",
    "errors": [],
  }

  try:
    import torch

    import os
    import site
    import sys

    repo_hint = args.ckpt
    if not os.environ.get("JEPAWM_LOGS"):
      os.environ["JEPAWM_LOGS"] = str((Path.home() / ".cache" / "jepa_wm").resolve())
    if not os.environ.get("JEPAWM_CKPT") or "${" in os.environ.get("JEPAWM_CKPT", ""):
      os.environ["JEPAWM_CKPT"] = _resolve_checkpoint(repo_hint)

    device = "cpu" if not torch.cuda.is_available() else "cuda"
    if args.device != "auto":
      device = args.device

    print(f"[jepa-smoke] torch {torch.__version__}")

    # Prefer local repo load to avoid internet-only source.
    repo_dir = Path(args.repo)
    if repo_dir.is_dir():
      model, preprocessor = torch.hub.load(
        str(repo_dir),
        "jepa_wm_metaworld",
        source="local",
        pretrained=args.pretrained,
        device=device,
      )
    else:
      model, preprocessor = torch.hub.load(
        args.repo, "jepa_wm_metaworld", source="github", pretrained=args.pretrained, device=device
      )

    model.eval()
    print("[jepa-smoke] model loaded")

    action_dim_candidates = _infer_action_dims(model, preprocessor)
    if len(action_dim_candidates) > 1:
      print(f"[jepa-smoke] trying action_dim candidates: {action_dim_candidates}")
    action_dim = int(action_dim_candidates[0])
    proprio_dim = int(getattr(preprocessor, "proprio_mean").numel())

    b = 1
    # A single-context frame keeps temporal tokenization aligned across JEPA-WM
    # variants and avoids context-length dependent token-count mismatches in smoke
    # rollout tests.
    context_len = 1
    model.to(device)

    obs = {
      "visual": torch.randint(
        low=0,
        high=256,
        size=(b, context_len, 3, 256, 256),
        dtype=torch.float32,
      ),
      "proprio": torch.zeros((b, context_len, proprio_dim), dtype=torch.float32),
    }
    z = model.encode(obs)

    z = z.to(device)
    unroll_exc: Exception | None = None
    frame_count = None
    frames = None
    with torch.no_grad():
      for candidate_action_dim in action_dim_candidates:
        try:
          act_suffix = torch.randn(args.smoke_steps, b, int(candidate_action_dim), device=device)
          act_suffix = act_suffix.to(device)
          z_pred = model.unroll(z, act_suffix=act_suffix, debug=False)
          frame_count = int(z_pred["visual"].shape[0]) if hasattr(z_pred, "keys") else int(z_pred.shape[0])
          frames = None
          if hasattr(model, "decode_unroll"):
            decoded = model.decode_unroll(z_pred, batch=True)
            frames = int(decoded.shape[1]) if decoded is not None else 0
          action_dim = int(candidate_action_dim)
          unroll_exc = None
          break
        except Exception as exc:
          unroll_exc = exc
          out["errors"].append(f"action_dim={candidate_action_dim}: {exc}")
      if unroll_exc is not None:
        raise unroll_exc

    out.update(
      {
        "status": "pass",
        "action_dim": action_dim,
        "proprio_dim": proprio_dim,
        "context_len": context_len,
        "predicted_frames": frame_count,
        "decoded_frames": frames,
        "device": device,
      }
    )
    print(json.dumps(out, indent=2))
    return 0
  except Exception as exc:
    out["status"] = "fail"
    out["errors"].append(str(exc))
    print(json.dumps(out, indent=2))
    traceback.print_exc()
    return 1


if __name__ == "__main__":
  raise SystemExit(main())
