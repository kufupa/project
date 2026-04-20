from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


@pytest.mark.skipif(
    importlib.util.find_spec("metaworld") is None,
    reason="metaworld not installed",
)
def test_mt10_vec_env_reset_step() -> None:
    pytest.importorskip("gymnasium")
    script = Path(__file__).resolve().parents[2] / "mt10" / "verify_env.py"
    spec = importlib.util.spec_from_file_location("mt10_verify_env", script)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main() == 0
