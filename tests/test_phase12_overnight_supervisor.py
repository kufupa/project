from __future__ import annotations

from pathlib import Path

from scripts.grpo import supervise_phase12_wm_grpo_overnight as sup


def test_latest_checkpoint_picks_highest_update(tmp_path: Path) -> None:
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    (ckpt_dir / "update_0005.pt").write_bytes(b"5")
    (ckpt_dir / "update_0020.pt").write_bytes(b"20")
    (ckpt_dir / "latest.pt").write_bytes(b"latest")

    path, update = sup.latest_checkpoint(tmp_path)

    assert path == ckpt_dir / "update_0020.pt"
    assert update == 20


def test_classify_failure_marks_partial_checkpoint_as_walltime(monkeypatch, tmp_path: Path) -> None:
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    (ckpt_dir / "update_0010.pt").write_bytes(b"10")
    monkeypatch.setattr(sup, "read_recent_logs", lambda: "")

    cause, symptom = sup.classify_failure(tmp_path)

    assert cause == "pbs_walltime_timeout"
    assert "update_0010.pt" in symptom


def test_classify_failure_detects_cuda_oom(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(sup, "read_recent_logs", lambda: "RuntimeError: CUDA out of memory")

    cause, symptom = sup.classify_failure(tmp_path)

    assert cause == "oom_or_cuda_memory"
    assert "CUDA OOM" in symptom


def test_auto_resume_uses_checkpoint_filename_as_next_start(monkeypatch, tmp_path: Path) -> None:
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    ckpt = ckpt_dir / "update_0010.pt"
    ckpt.write_bytes(b"10")
    submitted: dict[str, object] = {}

    monkeypatch.setattr(sup, "qstat_visible", lambda _job_id: False)
    monkeypatch.setattr(sup, "read_recent_logs", lambda: "")

    def fake_submit(run_dir, *, resume=None, start_update=None):
        submitted["run_dir"] = run_dir
        submitted["resume"] = resume
        submitted["start_update"] = start_update
        return "12345.pbs"

    monkeypatch.setattr(sup, "submit_strict", fake_submit)

    state = sup.handle_once(tmp_path, auto_resume=True)

    assert state.phase == "resubmitted"
    assert submitted["resume"] == ckpt
    assert submitted["start_update"] == 10


def test_final_checkpoint_without_eval_blocks_as_eval_failure(monkeypatch, tmp_path: Path) -> None:
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    (ckpt_dir / "update_0020.pt").write_bytes(b"20")
    monkeypatch.setattr(sup, "qstat_visible", lambda _job_id: False)

    state = sup.handle_once(tmp_path, auto_resume=True)

    assert state.phase == "blocked"
    assert state.failures[-1]["root_cause"] == "eval_cli_failure"
