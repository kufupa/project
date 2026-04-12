from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from segment_grpo_loop import _prepare_goal_image_for_wm  # noqa: E402


def test_prepare_goal_image_for_wm_horizontal_swap() -> None:
    # 2x2x3: columns swap when axis=1 flip
    img = np.arange(2 * 2 * 3, dtype=np.uint8).reshape(2, 2, 3)
    out = _prepare_goal_image_for_wm(img, flip_horizontal=True)
    assert out.shape == (2, 2, 3)
    np.testing.assert_array_equal(out[0, 0], img[0, 1])
    np.testing.assert_array_equal(out[0, 1], img[0, 0])
    same = _prepare_goal_image_for_wm(img, flip_horizontal=False)
    np.testing.assert_array_equal(same, img)
