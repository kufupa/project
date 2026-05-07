"""Tests for official LeRobot MT50 Phase071 wrapper and eval_info summarizer."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]


def _load_summarize():
    module_path = _REPO / "scripts" / "mt50" / "summarize_official_lerobot_eval.py"
    spec = importlib.util.spec_from_file_location("summarize_official_lerobot_eval", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_latex_tables():
    module_path = _REPO / "scripts" / "mt50" / "render_official_lerobot_latex_tables.py"
    spec = importlib.util.spec_from_file_location("render_official_lerobot_latex_tables", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_configurable_rendering():
    module_path = _REPO / "scripts" / "mt50" / "lerobot_eval_configurable_rendering.py"
    spec = importlib.util.spec_from_file_location("lerobot_eval_configurable_rendering", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
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


def test_official_wrapper_dry_run_uses_no_video_adapter() -> None:
    script = _REPO / "scripts" / "mt50" / "run_official_lerobot_mt50_eval.sh"
    env = os.environ.copy()
    env["MT50_PHASE071_DRY_RUN"] = "true"
    env["MT50_PHASE071_TASK"] = "push-v3"
    env["MT50_PHASE071_EPISODES"] = "10"
    env["MT50_PHASE071_OUTPUT_ROOT"] = str(
        _REPO / "artifacts" / "MT50_Phase072_official_lerobot_push_10ep"
    )
    env["MT50_LEROBOT_MAX_EPISODES_RENDERED"] = "0"
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
    assert "lerobot_eval_configurable_rendering.py" in out
    assert "--env.task=push-v3" in out
    assert "--eval.n_episodes=10" in out
    assert "MT50_Phase072_official_lerobot_push_10ep" in out
    assert "max_episodes_rendered=0" in out


def test_configurable_rendering_forces_zero_videos() -> None:
    mod = _load_configurable_rendering()
    calls = []

    def fake_eval_policy_all(*args, **kwargs):
        calls.append(kwargs)
        return {"overall": {"n_episodes": 1}}

    wrapped = mod.with_configurable_rendering(fake_eval_policy_all, max_episodes_rendered=0)
    result = wrapped(max_episodes_rendered=10, videos_dir=Path("/tmp/videos"))

    assert result == {"overall": {"n_episodes": 1}}
    assert calls == [{"max_episodes_rendered": 0, "videos_dir": None}]


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
        "scripts/mt50/submit_mt50_phase072_official_push_10ep_no_video.slurm",
        "scripts/mt50/submit_mt50_phase072_10ep_all_shards.sh",
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


def test_render_latex_tables_are_deterministic(tmp_path: Path) -> None:
    mod = _load_latex_tables()
    run_root = tmp_path / "phase072"
    shard = run_root / "shard0"
    shard.mkdir(parents=True)
    (shard / "eval_info.json").write_text(
        json.dumps(
            {
                "per_task": [
                    {
                        "task_group": "assembly-v3",
                        "task_id": 0,
                        "metrics": {"successes": [True, False, True]},
                    },
                    {
                        "task_group": "button-press-v3",
                        "task_id": 0,
                        "metrics": {"successes": [True, True, True]},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    diff_path = tmp_path / "difficulty.json"
    diff_path.write_text(
        json.dumps(
            {
                "tasks": [
                    {"task": "assembly-v3", "difficulty": "hard"},
                    {"task": "button-press-v3", "difficulty": "easy"},
                ]
            }
        ),
        encoding="utf-8",
    )

    rows = mod.collect_rows(run_root, ["shard0"], diff_path)
    assert [row.task for row in rows] == ["button-press-v3", "assembly-v3"]

    latex = mod.render_latex(rows, caption_prefix="MT50 Phase072 official")
    assert "button-press-v3 & easy & 3/3 & 100.0\\%" in latex
    assert "assembly-v3 & hard & 2/3 & 66.7\\%" in latex
    assert "easy & 3/3 & 100.0\\% & 1" in latex
    assert "hard & 2/3 & 66.7\\% & 1" in latex
