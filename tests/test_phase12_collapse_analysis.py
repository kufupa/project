from __future__ import annotations

import json
from pathlib import Path

from scripts.grpo.analyze_phase12_wm_collapse import analyze_run


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_analyzer_flags_wm_reward_real_eval_divergence(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    eval_dir = run_dir / "eval100_u0005_0020_stride5_nenv25_async"
    eval_dir.mkdir(parents=True)
    _write_jsonl(
        run_dir / "progress.jsonl",
        [
            {"event": "update_complete", "update_index": 9, "segment_candidate_rewards": [[0.1, 0.2]], "action_clip_fraction": 0.1},
            {"event": "update_complete", "update_index": 14, "segment_candidate_rewards": [[0.3, 0.4]], "action_clip_fraction": 0.5},
        ],
    )
    (eval_dir / "eval_sweep_summary.json").write_text(
        json.dumps({"episodes": 100, "rows": [{"update": 10, "pc_success": 32.0}, {"update": 15, "pc_success": 17.0}]}),
        encoding="utf-8",
    )

    report = analyze_run(run_dir)

    assert report["best_update"] == 10
    assert report["wm_reward_increased_while_eval_dropped"] is True
    assert report["clip_fraction_grew_after_best"] is True

