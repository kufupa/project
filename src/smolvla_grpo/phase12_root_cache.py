"""Root observation cache helpers for Phase12 pure-WM training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Phase12RootEntry:
    frame_index_1based: int
    policy_image: np.ndarray
    wm_image: np.ndarray
    proprio: np.ndarray
    proc: Any


def build_oracle_root_entry(
    *,
    env_h: Any,
    bundle: Any,
    policy_frame: np.ndarray,
    wm_frame: np.ndarray,
    raw_obs: np.ndarray,
    proprio: np.ndarray,
    frame_index_1based: int,
) -> Phase12RootEntry:
    policy_arr = np.asarray(policy_frame, dtype=np.uint8).copy()
    wm_arr = np.asarray(wm_frame, dtype=np.uint8).copy()
    prop = np.asarray(proprio, dtype=np.float32).reshape(-1).copy()
    raw = np.asarray(raw_obs, dtype=np.float64).reshape(-1)
    if raw.size < 4:
        raise ValueError("raw_obs must contain at least 4 agent position values")
    obs = {
        "pixels": policy_arr[None, ...],
        "agent_pos": raw[:4][None, ...],
    }
    proc = env_h.build_proc(obs, bundle=bundle)
    return Phase12RootEntry(
        frame_index_1based=int(frame_index_1based),
        policy_image=policy_arr,
        wm_image=wm_arr,
        proprio=prop,
        proc=proc,
    )
