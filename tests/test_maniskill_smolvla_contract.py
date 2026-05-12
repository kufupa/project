from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_episode(path: Path, *, grip_sequence: list[float], image_value: int = 0) -> None:
    n = len(grip_sequence)
    actions = np.zeros((n, 7), dtype=np.float32)
    states = np.zeros((n, 7), dtype=np.float32)
    actions[:, 6] = np.asarray(grip_sequence, dtype=np.float32)
    states[:, 6] = np.asarray(grip_sequence, dtype=np.float32)
    images = [np.full((4, 5, 3), image_value, dtype=np.uint8) for _ in range(n)]
    info = [{"success": i == n - 1} for i in range(n)]
    np.savez_compressed(
        path,
        {
            "image": images,
            "instruction": "put carrot on plate",
            "state": states,
            "action": actions,
            "info": info,
        },
    )


def test_audit_detects_duplicate_decoded_signatures(tmp_path: Path) -> None:
    audit = _load_module(
        PROJECT / "scripts" / "maniskill_smolvla" / "audit_npz_contract.py",
        "audit_npz_contract",
    )
    _write_episode(tmp_path / "success_proc_0_numid_0_epsid_0.npz", grip_sequence=[1, -1], image_value=7)
    _write_episode(tmp_path / "success_proc_1_numid_0_epsid_0.npz", grip_sequence=[1, -1], image_value=7)

    summary = audit.audit_root(tmp_path, min_episodes=2, sample_limit=0)

    assert summary["episodes"] == 2
    assert summary["duplicate_decoded_signature_count"] == 1
    assert summary["state_gripper_equal_action_gripper_count"] == 2


def test_audit_reports_temporal_filter_damage(tmp_path: Path) -> None:
    audit = _load_module(
        PROJECT / "scripts" / "maniskill_smolvla" / "audit_npz_contract.py",
        "audit_npz_contract",
    )
    _write_episode(tmp_path / "success_proc_0_numid_0_epsid_0.npz", grip_sequence=[1, 1, 1, -1], image_value=3)

    summary = audit.audit_root(tmp_path, min_episodes=1, sample_limit=0)

    assert summary["episodes"] == 1
    assert summary["filter_small_actions_skip_fraction_max"] > 0.0
    assert summary["success_true_count_min"] == 1


def test_previous_action_gripper_makes_state_causal() -> None:
    converter = _load_module(
        PROJECT / "scripts" / "maniskill_smolvla" / "convert_npz_to_lerobot.py",
        "convert_npz_to_lerobot",
    )
    states = np.zeros((4, 7), dtype=np.float32)
    states[:, 6] = np.asarray([1, -1, -1, 1], dtype=np.float32)
    actions = np.zeros((4, 7), dtype=np.float32)
    actions[:, 6] = np.asarray([1, -1, -1, 1], dtype=np.float32)

    fixed = converter.apply_state_gripper_mode(
        states,
        actions,
        mode="previous-action",
        initial_gripper=1.0,
    )

    assert fixed[:, 6].tolist() == [1.0, 1.0, -1.0, -1.0]
