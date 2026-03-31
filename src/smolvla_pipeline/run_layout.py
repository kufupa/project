from __future__ import annotations

import os
import re
import secrets
from datetime import datetime, timezone
from pathlib import Path


def _slug_component(value: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", value).lower().strip("_")


def _slug_task(task: str) -> str:
    return _slug_component(task)


def effective_run_name_prefix_slug() -> str:
    """Slug from RUN_NAME_PREFIX or ORACLE_RUN_PREFIX when set (empty otherwise)."""
    raw = (os.environ.get("RUN_NAME_PREFIX") or os.environ.get("ORACLE_RUN_PREFIX") or "").strip()
    if not raw:
        return ""
    return _slug_component(raw)


def slug_run_directory_prefix(label: str) -> str:
    """Sanitize a user-provided run directory prefix (e.g. ``mt10``)."""
    return _slug_component(label.strip())


def _resolved_run_name_prefix_slug(run_name_prefix: str | None) -> str:
    if run_name_prefix is not None:
        stripped = run_name_prefix.strip()
        if not stripped:
            return ""
        return _slug_component(stripped)
    return effective_run_name_prefix_slug()


def build_run_dir_name(
    *,
    timestamp_utc: str,
    episodes: int,
    task: str,
    seed: int,
    variant: str,
    nonce: str,
    run_name_prefix: str | None = None,
) -> str:
    task_slug = _slug_task(task)
    variant_slug = _slug_component(variant)
    base = f"run_{timestamp_utc}_ep{episodes}_v{variant_slug}_t{task_slug}_s{seed}_r{nonce}"
    pslug = _resolved_run_name_prefix_slug(run_name_prefix)
    if pslug:
        return f"{pslug}_{base}"
    return base


def ensure_unique_run_dir(
    output_root: Path,
    *,
    episodes: int,
    task: str,
    seed: int,
    variant: str,
    run_name_prefix: str | None = None,
) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp_utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    for _ in range(20):
        nonce = f"{secrets.randbelow(1_000_000):06d}"
        run_dir_name = build_run_dir_name(
            timestamp_utc=timestamp_utc,
            episodes=episodes,
            task=task,
            seed=seed,
            variant=variant,
            nonce=nonce,
            run_name_prefix=run_name_prefix,
        )
        run_dir = output_root / run_dir_name
        try:
            run_dir.mkdir()
            return run_dir
        except FileExistsError:
            continue

    raise RuntimeError("Unable to create unique run directory after retries.")
