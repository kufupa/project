from __future__ import annotations

import numpy as np
import pytest


def test_seed_metaworld_process_numpy_repeatable() -> None:
    pytest.importorskip("torch")
    from metaworld_determinism import seed_metaworld_process

    seed_metaworld_process(12345)
    a = float(np.random.random())
    seed_metaworld_process(12345)
    b = float(np.random.random())
    assert a == b


def test_gymnasium_reset_strict_raises_on_unexpected_keyword_env() -> None:
    from metaworld_determinism import gymnasium_reset_strict

    class LegacyEnv:
        def reset(self):
            return ({"obs": 1}, {})

    with pytest.raises(RuntimeError, match=r"env\.reset\(seed="):
        gymnasium_reset_strict(LegacyEnv(), seed=0)


def test_gymnasium_reset_strict_tuple_obs_info() -> None:
    from metaworld_determinism import gymnasium_reset_strict

    class OkEnv:
        def reset(self, *, seed=None):
            return ({"x": 1}, {"ok": True})

    out = gymnasium_reset_strict(OkEnv(), 7)
    assert isinstance(out, tuple)
    assert out[0] == {"x": 1}
    assert out[1] == {"ok": True}


def test_metaworld_strict_ctor_requested(monkeypatch: pytest.MonkeyPatch) -> None:
    from metaworld_determinism import metaworld_strict_ctor_requested

    monkeypatch.delenv("METAWORLD_STRICT_CTOR", raising=False)
    assert metaworld_strict_ctor_requested() is False
    monkeypatch.setenv("METAWORLD_STRICT_CTOR", "1")
    assert metaworld_strict_ctor_requested() is True


def test_seed_metaworld_process_torch_error_not_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")

    from metaworld_determinism import seed_metaworld_process

    def _boom(_s: int) -> None:
        raise RuntimeError("torch_seed_failed")

    monkeypatch.setattr(torch, "manual_seed", _boom)
    with pytest.raises(RuntimeError, match="torch_seed_failed"):
        seed_metaworld_process(99)
