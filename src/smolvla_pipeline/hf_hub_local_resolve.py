"""Resolve Hugging Face hub ``owner/name`` ids to local snapshot dirs (stdlib only)."""

from __future__ import annotations

import os
from pathlib import Path


def _truthy_env(name: str) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def is_remote_hub_repo_id(repo_id: str) -> bool:
    parts = str(repo_id).strip().split("/")
    if len(parts) != 2:
        return False
    if not parts[0] or not parts[1]:
        return False
    if parts[0].startswith(".") or parts[0].startswith("~"):
        return False
    return True


def hf_hub_cache_root() -> Path:
    raw = os.environ.get("HUGGINGFACE_HUB_CACHE", "").strip()
    if raw:
        return Path(raw)
    hf = os.environ.get("HF_HOME", "").strip()
    if hf:
        return Path(hf) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def hf_hub_model_dir_name(repo_id: str) -> str:
    rid = str(repo_id).strip().strip("/")
    parts = rid.split("/")
    if len(parts) != 2:
        return ""
    return f"models--{parts[0]}--{parts[1]}"


def should_resolve_hf_hub_to_local() -> bool:
    if _truthy_env("SMOLVLA_LOCAL_FILES_ONLY"):
        return True
    if os.environ.get("HF_HUB_OFFLINE", "").strip() == "1":
        return True
    if _truthy_env("MT50_PHASE07_USE_LOCAL_HF_SNAPSHOT"):
        return True
    if _truthy_env("SMOLVLA_PREFER_LOCAL_HF_SNAPSHOT"):
        return True
    return False


def should_strict_require_local_hf() -> bool:
    if _truthy_env("SMOLVLA_LOCAL_FILES_ONLY"):
        return True
    if os.environ.get("HF_HUB_OFFLINE", "").strip() == "1":
        return True
    if _truthy_env("MT50_PHASE07_USE_LOCAL_HF_SNAPSHOT"):
        return True
    return False


def resolve_hf_hub_repo_to_local_snapshot(
    repo_id: str,
    *,
    hub_cache: Path | None = None,
    strict: bool | None = None,
) -> str:
    rid = str(repo_id).strip()
    if not rid or not is_remote_hub_repo_id(rid):
        return rid
    use_strict = should_strict_require_local_hf() if strict is None else bool(strict)
    hub = hub_cache if hub_cache is not None else hf_hub_cache_root()
    name = hf_hub_model_dir_name(rid)
    if not name:
        if use_strict:
            raise RuntimeError(f"invalid Hugging Face repo id for hub resolve: {rid!r}")
        return rid
    snapshots_dir = hub / name / "snapshots"
    if not snapshots_dir.is_dir():
        if use_strict:
            raise RuntimeError(
                f"no local Hugging Face hub snapshot for {rid!r} under {snapshots_dir}. "
                "Warm the cache first or unset SMOLVLA_LOCAL_FILES_ONLY / HF_HUB_OFFLINE / "
                "MT50_PHASE07_USE_LOCAL_HF_SNAPSHOT."
            )
        return rid
    candidates = [p for p in snapshots_dir.iterdir() if p.is_dir()]
    if not candidates:
        if use_strict:
            raise RuntimeError(
                f"no local Hugging Face hub snapshot for {rid!r} under {snapshots_dir} (empty)."
            )
        return rid
    picked = max(candidates, key=lambda p: p.stat().st_mtime)
    return str(picked.resolve())
