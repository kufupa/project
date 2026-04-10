from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re
import secrets


def _slug_component(value: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", value).lower().strip("_")


def _slug_task(task: str) -> str:
    return _slug_component(task)


def build_run_dir_name(
    *,
    timestamp_utc: str,
    episodes: int,
    task: str,
    seed: int,
    variant: str,
    nonce: str,
) -> str:
    task_slug = _slug_task(task)
    variant_slug = _slug_component(variant)
    return f"run_{timestamp_utc}_ep{episodes}_v{variant_slug}_t{task_slug}_s{seed}_r{nonce}"


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
        run_dir_name = build_run_dir_name(
            timestamp_utc=timestamp_utc,
            episodes=episodes,
            task=task,
            seed=seed,
            variant=variant,
            nonce=nonce,
        )
        run_dir = output_root / run_dir_name
        try:
            run_dir.mkdir()
            return run_dir
        except FileExistsError:
            continue

    raise RuntimeError("Unable to create unique run directory after retries.")
