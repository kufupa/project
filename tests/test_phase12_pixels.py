from __future__ import annotations

import numpy as np

from smolvla_grpo.phase12_pixels import (
    policy_rgb_from_obs,
    policy_rgb_from_raw_corner2,
    to_rgb_uint8,
    wm_rgb_from_policy_rgb_corner2,
    wm_rgb_from_raw_corner2,
)


def _raw_frame() -> np.ndarray:
    return np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)


def test_policy_rgb_from_raw_corner2_is_vhflip() -> None:
    raw = _raw_frame()

    out = policy_rgb_from_raw_corner2(raw)

    np.testing.assert_array_equal(out, np.flip(raw, (0, 1)))
    assert out.flags.c_contiguous


def test_wm_rgb_from_raw_corner2_is_vflip_only() -> None:
    raw = _raw_frame()

    out = wm_rgb_from_raw_corner2(raw)

    np.testing.assert_array_equal(out, np.flip(raw, 0))
    assert out.flags.c_contiguous


def test_wm_rgb_from_policy_rgb_corner2_removes_horizontal_flip_only() -> None:
    raw = _raw_frame()
    policy = policy_rgb_from_raw_corner2(raw)

    out = wm_rgb_from_policy_rgb_corner2(policy)

    np.testing.assert_array_equal(out, wm_rgb_from_raw_corner2(raw))
    assert out.flags.c_contiguous


def test_to_rgb_uint8_drops_alpha_and_scales_unit_float() -> None:
    rgba = np.zeros((2, 2, 4), dtype=np.float32)
    rgba[..., 1] = 1.0
    rgba[..., 3] = 0.25

    out = to_rgb_uint8(rgba)

    assert out.dtype == np.uint8
    assert out.shape == (2, 2, 3)
    assert int(out[..., 1].max()) == 255


def test_policy_rgb_from_obs_extracts_single_env_pixels() -> None:
    raw = _raw_frame()
    obs = {"pixels": raw[None]}

    out = policy_rgb_from_obs(obs)

    np.testing.assert_array_equal(out, raw)
    assert out.flags.c_contiguous


def test_policy_rgb_from_obs_accepts_unbatched_pixels() -> None:
    raw = _raw_frame()
    obs = {"pixels": raw}

    out = policy_rgb_from_obs(obs)

    np.testing.assert_array_equal(out, raw)
    assert out.flags.c_contiguous
