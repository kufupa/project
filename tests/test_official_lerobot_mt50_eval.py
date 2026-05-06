"""Tests for official LeRobot MT50 Phase071 wrapper and eval_info summarizer."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]


def _load_summarize():
    module_path = _REPO / "scripts" / "mt50" / "summarize_official_lerobot_eval.py"
    spec = importlib.util.spec_from_file_location("summarize_official_lerobot_eval", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_official_wrapper_dry_run_prints_lerobot_eval_command() -> None:
    script = _REPO / "scripts" / "mt50" / "run_official_lerobot_mt50_eval.sh"
    env = os.environ.copy()
    env["MT50_PHASE071_DRY_RUN"] = "true"
    env.pop("MT50_PHASE071_TASK", None)
    env.pop("MT50_PHASE071_OUTPUT_ROOT", None)
    proc = subprocess.run(
        ["bash", str(script)],
        cwd=str(_REPO),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    out = proc.stdout + proc.stderr
    assert "lerobot.scripts.lerobot_eval" in out
    assert "--env.type=metaworld" in out
    assert "--env.task=assembly-v3" in out
    assert "--eval.n_episodes=1" in out
    assert "--eval.batch_size=1" in out
    assert "--seed=1000" in out
    assert "MT50_Phase071_official_lerobot_1task_1ep" in out


def test_summarize_multitask_shape_matches_phase071_schema() -> None:
    mod = _load_summarize()
    eval_info = {
        "per_task": [
            {
                "task_group": "assembly-v3",
                "task_id": 0,
                "metrics": {
                    "sum_rewards": [12.0],
                    "max_rewards": [3.0],
                    "successes": [False],
                    "video_paths": ["videos/assembly-v3_0/eval_episode_0.mp4"],
                },
            }
        ],
        "overall": {
            "pc_success": 0.0,
            "n_episodes": 1,
            "video_paths": ["videos/assembly-v3_0/eval_episode_0.mp4"],
        },
    }
    run_root = Path("/tmp/does_not_matter")
    index = mod.build_official_index(eval_info, run_root=run_root, seed=1000)
    assert index["phase"] == "MT50_Phase071"
    assert index["source"] == "official_lerobot_eval"
    assert index["task_count"] == 1
    assert index["seed"] == 1000
    assert index["tasks"][0]["task_group"] == "assembly-v3"
    assert index["tasks"][0]["pc_success"] == 0.0
    assert index["tasks"][0]["video_paths"] == ["videos/assembly-v3_0/eval_episode_0.mp4"]


def test_summarize_warns_on_missing_video_file(tmp_path: Path) -> None:
    mod = _load_summarize()
    eval_info = {
        "per_task": [
            {
                "task_group": "assembly-v3",
                "task_id": 0,
                "metrics": {
                    "successes": [True],
                    "video_paths": ["videos/assembly-v3_0/eval_episode_0.mp4"],
                },
            }
        ],
        "overall": {"pc_success": 100.0, "n_episodes": 1},
    }
    index = mod.build_official_index(eval_info, run_root=tmp_path, seed=None)
    assert "warnings" in index
    assert any("missing_video" in w for w in index["warnings"])


def test_phase071_slurm_and_wrapper_bash_syntax() -> None:
    for rel in (
        "scripts/mt50/run_official_lerobot_mt50_eval.sh",
        "scripts/mt50/submit_mt50_phase071_official_1task_1ep.slurm",
        "scripts/mt50/submit_mt50_phase071_official_full_1ep.slurm",
        "scripts/mt50/submit_mt50_phase072_10ep_shard0.slurm",
        "scripts/mt50/submit_mt50_phase072_10ep_shard1.slurm",
        "scripts/mt50/submit_mt50_phase072_10ep_shard2.slurm",
    ):
        subprocess.run(["bash", "-n", str(_REPO / rel)], check=True, cwd=str(_REPO))


def test_merge_shard_eval_infos(tmp_path: Path) -> None:
    mod_path = _REPO / "scripts" / "mt50" / "merge_official_lerobot_eval_shards.py"
    spec = importlib.util.spec_from_file_location("merge_official", mod_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    parent = tmp_path / "run"
    for i, name in enumerate(["shard0", "shard1"]):
        d = parent / name
        d.mkdir(parents=True)
        (d / "eval_info.json").write_text(
            json.dumps(
                {
                    "per_task": [
                        {
                            "task_group": f"t{i}",
                            "task_id": 0,
                            "metrics": {
                                "sum_rewards": [1.0],
                                "max_rewards": [1.0],
                                "successes": [True],
                                "video_paths": [],
                            },
                        }
                    ],
                    "per_group": {f"t{i}": {"n_episodes": 1, "pc_success": 100.0}},
                    "overall": {"eval_s": 10.0, "n_episodes": 1},
                }
            ),
            encoding="utf-8",
        )
    merged = mod.merge_shard_eval_infos(parent, ["shard0", "shard1"], expected_tasks=2)
    assert len(merged["per_task"]) == 2
    assert merged["overall"]["n_episodes"] == 2
    assert "t0" in merged["per_group"] and "t1" in merged["per_group"]


def test_summarize_cli_writes_json(tmp_path: Path) -> None:
    mod_path = _REPO / "scripts" / "mt50" / "summarize_official_lerobot_eval.py"
    eval_path = tmp_path / "eval_info.json"
    eval_path.write_text(
        json.dumps(
            {
                "per_task": [
                    {
                        "task_group": "easy",
                        "task_id": 0,
                        "metrics": {"successes": [True], "video_paths": []},
                    }
                ],
                "overall": {"pc_success": 100.0, "n_episodes": 1},
            }
        ),
        encoding="utf-8",
    )
    out_path = tmp_path / "index.json"
    subprocess.run(
        [
            os.environ.get("PYTHON", "python3"),
            str(mod_path),
            "--eval-info",
            str(eval_path),
            "--output",
            str(out_path),
            "--run-root",
            str(tmp_path),
            "--seed",
            "42",
        ],
        check=True,
    )
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["seed"] == 42
    assert data["tasks"][0]["task_group"] == "easy"
