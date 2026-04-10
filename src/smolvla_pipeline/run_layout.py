from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re
import secrets


def _slug_task(task: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", task).lower().strip("_")


def build_run_dir_name(
    *,
    timestamp_utc: str,
    episodes: int,
    task: str,
    seed: int,
    variant: str,
    nonce: str,
) -> str:
    slug_task = _slug_task(task)
    return f"run_{timestamp_utc}_ep{episodes}_v{variant}_t{slug_task}_s{seed}_r{nonce}"


def ensure_unique_run_dir(
    output_root: Path,
    *,
    episodes: int,
    task: str,
    seed: int,
    variant: str,
) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp_utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    for _ in range(20):
        nonce = f"{secrets.randbelow(1_000_000):06d}"
        run_name = build_run_dir_name(
            timestamp_utc=timestamp_utc,
            episodes=episodes,
            task=task,
            seed=seed,
            variant=variant,
            nonce=nonce,
        )
        run_dir = output_root / run_name
        try:
            run_dir.mkdir()
        except FileExistsError:
            continue
        return run_dir

    raise RuntimeError("Failed to create a unique run directory after 20 attempts.")
