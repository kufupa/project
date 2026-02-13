from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


def _load_run_segment_grpo_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "run_segment_grpo.py"
    spec = importlib.util.spec_from_file_location("run_segment_grpo_script", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prepare_output_json_auto_nests_direct_artifacts_json(tmp_path: Path, monkeypatch):
    module = _load_run_segment_grpo_module()
    artifacts_root = tmp_path / "artifacts"
    artifacts_root.mkdir(parents=True, exist_ok=True)
    output_root = artifacts_root / "phase08_segment_grpo_baseline"

    fixed_run_dir = output_root / "run_fixed"

    def _fake_unique_run_dir(*_args, **_kwargs):
        fixed_run_dir.mkdir(parents=True, exist_ok=True)
        return fixed_run_dir

    monkeypatch.setattr(module, "ensure_unique_run_dir", _fake_unique_run_dir)

    args = SimpleNamespace(
        output_json=artifacts_root / "segment_grpo_default2.json",
        output_root=output_root,
        artifacts_root=artifacts_root,
        episodes=1,
        task="push-v3",
        seed=1000,
        flat_output=False,
    )

    output_json, run_dir = module._prepare_output_json(args)
    assert run_dir == fixed_run_dir
    assert output_json == fixed_run_dir / "segment_grpo_default2.json"


def test_prepare_output_json_respects_flat_output_flag(tmp_path: Path):
    module = _load_run_segment_grpo_module()
    artifacts_root = tmp_path / "artifacts"
    target = artifacts_root / "segment_grpo_default2.json"

    args = SimpleNamespace(
        output_json=target,
        output_root=artifacts_root / "phase08_segment_grpo_baseline",
        artifacts_root=artifacts_root,
        episodes=1,
        task="push-v3",
        seed=1000,
        flat_output=True,
    )

    output_json, run_dir = module._prepare_output_json(args)
    assert run_dir is None
    assert output_json == target


def test_prepare_output_json_keeps_pre_nested_run_path(tmp_path: Path):
    module = _load_run_segment_grpo_module()
    artifacts_root = tmp_path / "artifacts"
    nested = artifacts_root / "phase08_segment_grpo_baseline" / "run_20260411T170000Z_ep1_vsegment_grpo_tpush_v3_s1000_r123456"
    nested.mkdir(parents=True, exist_ok=True)
    target = nested / "segment_grpo_default2.json"

    args = SimpleNamespace(
        output_json=target,
        output_root=artifacts_root / "phase08_segment_grpo_baseline",
        artifacts_root=artifacts_root,
        episodes=1,
        task="push-v3",
        seed=1000,
        flat_output=False,
    )

    output_json, run_dir = module._prepare_output_json(args)
    assert run_dir is None
    assert output_json == target
