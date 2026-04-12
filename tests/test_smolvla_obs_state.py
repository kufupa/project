from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from smolvla_obs_state import flatten_obs_state  # noqa: E402


def test_flatten_obs_state_sorted_dict_concat() -> None:
    obs = {"z": np.array([1.0], dtype=np.float32), "a": np.array([2.0, 3.0], dtype=np.float32)}
    out = flatten_obs_state(obs)
    np.testing.assert_allclose(out, np.array([2.0, 3.0, 1.0], dtype=np.float32))


def test_flatten_obs_state_matches_evaluator_contract_import() -> None:
    """Evaluator must use the same module (import-time check)."""
    from smolvla_pipeline import evaluator as ev

    obs = {"m": np.zeros(2, dtype=np.float32), "n": np.ones(1, dtype=np.float32)}
    np.testing.assert_allclose(flatten_obs_state(obs), ev._flatten_obs_state(obs))
