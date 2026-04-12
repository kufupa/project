from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def test_render_jepa_rgb_shape_push_v3() -> None:
    pytest.importorskip("metaworld")
    from metaworld_jepa_render import build_jepa_metaworld_env, render_jepa_rgb

    size = 224
    env, train_tasks = build_jepa_metaworld_env("push-v3", img_size=size, seed=0)
    if train_tasks:
        env.set_task(train_tasks[0])
    env.reset(seed=0)
    img = render_jepa_rgb(env)
    assert img.ndim == 3
    assert img.shape[2] == 3
    assert img.shape[0] == img.shape[1] == size
    assert img.dtype == np.uint8
    try:
        env.close()
    except Exception:
        pass
