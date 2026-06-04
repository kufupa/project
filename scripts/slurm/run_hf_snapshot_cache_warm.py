#!/usr/bin/env python3
"""Warm the Hugging Face snapshot cache for a repo on a Slurm compute node."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path


def _count_snapshot_files(snapshot_dir: Path) -> tuple[int, int]:
    files = 0
    bytes_total = 0
    for path in snapshot_dir.rglob("*"):
        if not path.is_file():
            continue
        files += 1
        try:
            bytes_total += path.stat().st_size
        except OSError:
            pass
    return files, bytes_total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id",
        default="jadechoghari/smolvla_metaworld",
        help="Hugging Face repo id to download into the local cache.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional revision, branch, tag, or commit SHA.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=int(os.environ.get("HF_CACHE_WARM_MAX_WORKERS", "4")),
        help="Parallel download workers for huggingface_hub.snapshot_download.",
    )
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    hub_cache = (
        os.environ.get("HF_HUB_CACHE")
        or os.environ.get("HUGGINGFACE_HUB_CACHE")
        or None
    )
    t0 = time.perf_counter()
    print(
        "hf_snapshot_download_begin "
        f"repo_id={args.repo_id!r} revision={args.revision!r} "
        f"HF_HOME={os.environ.get('HF_HOME', '')!r} hub_cache={hub_cache!r}",
        flush=True,
    )
    snapshot_path = snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        cache_dir=hub_cache,
        max_workers=max(1, args.max_workers),
    )
    files, bytes_total = _count_snapshot_files(Path(snapshot_path))
    print(
        "hf_snapshot_download_ok "
        f"repo_id={args.repo_id!r} snapshot_path={snapshot_path!r} "
        f"files={files} bytes={bytes_total} elapsed_s={time.perf_counter() - t0:.2f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
