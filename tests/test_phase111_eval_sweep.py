from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_module():
    repo = Path(__file__).resolve().parents[1]
    path = repo / "scripts" / "grpo" / "eval_phase111_grpo_sweep.py"
    spec = importlib.util.spec_from_file_location("eval_phase111_grpo_sweep", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_run_sweep_collects_rows_and_topk(tmp_path, monkeypatch) -> None:
    mod = _load_module()
    run_dir = tmp_path / "run"
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "update_0005.pt").write_text("x", encoding="utf-8")
    (ckpt_dir / "update_0010.pt").write_text("x", encoding="utf-8")

    def fake_run(cmd, check):  # noqa: ANN001
        out_dir = Path(cmd[cmd.index("--output-dir") + 1])
        ckpt = Path(cmd[cmd.index("--grpo-checkpoint") + 1]).name
        episodes = int(cmd[cmd.index("--episodes") + 1])
        update = int(ckpt.replace("update_", "").replace(".pt", ""))
        out_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "pc_success": float(update),
            "avg_sum_reward": float(update) + 0.5,
            "avg_max_reward": float(update) + 1.0,
            "episodes": episodes,
        }
        (out_dir / "eval_summary.json").write_text(json.dumps(summary), encoding="utf-8")
        return None

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    result = mod.run_sweep(
        base_checkpoint="dummy",
        run_dir=run_dir,
        task="push-v3",
        episodes=10,
        eval_seed_start=1000,
        top_k=1,
        top_k_episodes=50,
    )
    assert len(result["rows"]) == 2
    assert result["rows"][0]["update"] == 5
    assert result["rows"][1]["update"] == 10
    assert "topk" in result
    assert len(result["topk"]["rows"]) == 1
    assert result["topk"]["rows"][0]["update"] == 10
    out_path = run_dir / "eval_sweep" / "eval_sweep_summary.json"
    assert out_path.exists()
