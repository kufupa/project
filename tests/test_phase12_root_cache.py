from __future__ import annotations

import numpy as np
import torch

from smolvla_grpo.phase12_root_cache import build_oracle_root_entry


class _Bundle:
    pass


class _Env:
    def build_proc(self, obs, *, bundle):
        assert isinstance(bundle, _Bundle)
        assert set(obs) == {"pixels", "agent_pos"}
        assert obs["pixels"].shape == (1, 4, 4, 3)
        assert obs["agent_pos"].shape == (1, 4)
        assert obs["pixels"].dtype == np.uint8
        return {
            "pixels": torch.as_tensor(obs["pixels"]),
            "agent_pos": torch.as_tensor(obs["agent_pos"]),
        }


def test_build_oracle_root_entry_contains_policy_and_wm_views() -> None:
    policy_frame = np.full((4, 4, 3), 7, dtype=np.uint8)
    wm_frame = np.full((4, 4, 3), 9, dtype=np.uint8)
    raw_obs = np.array([1.0, 2.0, 3.0, 4.0, 99.0], dtype=np.float64)
    proprio = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

    entry = build_oracle_root_entry(
        env_h=_Env(),
        bundle=_Bundle(),
        policy_frame=policy_frame,
        wm_frame=wm_frame,
        raw_obs=raw_obs,
        proprio=proprio,
        frame_index_1based=5,
    )

    assert entry.frame_index_1based == 5
    assert entry.policy_image.shape == (4, 4, 3)
    assert entry.wm_image.shape == (4, 4, 3)
    assert entry.proprio.tolist() == [1.0, 2.0, 3.0, 4.0]
    assert tuple(entry.proc["agent_pos"].shape) == (1, 4)
    assert entry.proc["agent_pos"].dtype == torch.float64

