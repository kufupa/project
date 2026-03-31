from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
SCRIPT_ROOT = ROOT / "scripts"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from run_phase9_oracle_vs_wm import main as phase9_main  # noqa: E402


def test_mt10_task_array_has_ten_ids() -> None:
    script = ROOT / "scripts" / "mt10" / "mt10_tasks.sh"
    r = subprocess.run(
        [
            "bash",
            "-c",
            f"source '{script}' && echo ${{#MT10_TASK_IDS[@]}}",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert r.stdout.strip() == "10"


@pytest.mark.parametrize(
    "rel",
    [
        "mt10_tasks.sh",
        "run_phase6_mt10.sh",
        "run_phase8_mt10.sh",
        "run_phase9_mt10.sh",
        "run_preflight_smoke.sh",
    ],
)
def test_mt10_shell_scripts_parse(rel: str) -> None:
    path = ROOT / "scripts" / "mt10" / rel
    subprocess.run(["bash", "-n", str(path)], check=True)


def test_run_phase6_mt10_dry_run_writes_index(tmp_path: Path) -> None:
    idx = tmp_path / "mt10_phase6_index.json"
    env = os.environ.copy()
    env["MT10_SKIP_POLICY_GATE"] = "1"
    env["MT10_PHASE6_INDEX_JSON"] = str(idx)
    r = subprocess.run(
        ["bash", str(ROOT / "scripts" / "mt10" / "run_phase6_mt10.sh"), "--dry-run"],
        cwd=str(ROOT),
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "dry-run" in r.stdout.lower() or "dry-run" in r.stdout
    data = json.loads(idx.read_text(encoding="utf-8"))
    assert data["phase"] == 6
    assert len(data["tasks"]) == 10
    assert all(v == "(dry-run-placeholder)" for v in data["tasks"].values())


def test_run_phase8_mt10_dry_run_requires_phase6_index(tmp_path: Path) -> None:
    idx = tmp_path / "mt10_phase6_index.json"
    all_tasks = [
        "button-press-topdown-v3",
        "door-open-v3",
        "drawer-close-v3",
        "drawer-open-v3",
        "peg-insert-side-v3",
        "pick-place-v3",
        "push-v3",
        "reach-v3",
        "window-close-v3",
        "window-open-v3",
    ]
    tasks = {t: f"/tmp/mt10_dummy_{i}" for i, t in enumerate(all_tasks)}
    idx.write_text(json.dumps({"phase": 6, "tasks": tasks}, indent=2), encoding="utf-8")
    env = os.environ.copy()
    env["MT10_PHASE6_INDEX_JSON"] = str(idx)
    r = subprocess.run(
        ["bash", str(ROOT / "scripts" / "mt10" / "run_phase8_mt10.sh"), "--dry-run"],
        cwd=str(ROOT),
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "dry-run" in r.stdout.lower()


def test_run_phase9_oracle_vs_wm_requires_oracle_run_dir() -> None:
    env = {k: v for k, v in os.environ.items() if k != "ORACLE_RUN_DIR"}
    r = subprocess.run(
        ["bash", str(ROOT / "scripts" / "segment_grpo" / "run_phase9_oracle_vs_wm.sh")],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    assert r.returncode != 0
    assert "ORACLE_RUN_DIR" in r.stderr


def test_phase9_manifest_task_mismatch_exits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    oracle_root = tmp_path / "oracle"
    oracle_root.mkdir()
    (oracle_root / "run_manifest.json").write_text(
        json.dumps(
            {
                "seed": 1000,
                "task": "push-v3",
                "episodes": [{"episode_index": 0, "reset_seed": 1000}],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_phase9_oracle_vs_wm.py",
            "--oracle-run-root",
            str(oracle_root),
            "--artifacts-root",
            str(tmp_path),
            "--output-root",
            str(tmp_path / "phase09"),
            "--task",
            "door-open-v3",
            "--episodes",
            "1",
            "--goal-frame-index",
            "50",
            "--max-steps",
            "50",
            "--chunk-len",
            "50",
            "--dry-run",
        ],
    )
    with pytest.raises(SystemExit) as exc:
        phase9_main()
    assert "mismatch" in str(exc.value).lower()
