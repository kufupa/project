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
    monkeypatch.setattr(sup, "read_recent_logs", lambda: "PBS: job killed after walltime limit")

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
    monkeypatch.setattr(sup, "read_recent_logs", lambda: "resources_used.walltime exceeded")

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

    state = sup.handle_once(tmp_path, auto_resume=False)

    assert state.phase == "blocked"
    assert state.failures[-1]["root_cause"] == "eval_cli_failure"


def test_partial_checkpoint_without_walltime_evidence_does_not_auto_resume(monkeypatch, tmp_path: Path) -> None:
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    (ckpt_dir / "update_0010.pt").write_bytes(b"10")
    monkeypatch.setattr(sup, "qstat_visible", lambda _job_id: False)
    monkeypatch.setattr(sup, "read_recent_logs", lambda: "RuntimeError: deterministic crash")

    state = sup.handle_once(tmp_path, auto_resume=True)

    assert state.phase == "blocked"
    assert state.failures[-1]["root_cause"] == "partial_checkpoint_no_walltime_evidence"


def test_latest_checkpoint_uses_latest_meta_when_newer(tmp_path: Path) -> None:
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    (ckpt_dir / "update_0005.pt").write_bytes(b"5")
    latest = ckpt_dir / "latest.pt"
    latest.write_bytes(b"latest")
    (ckpt_dir / "latest.pt.meta.json").write_text('{"update_index": 7}', encoding="utf-8")

    path, start_update = sup.latest_checkpoint(tmp_path)

    assert path == latest
    assert start_update == 8


def test_eval_missing_auto_submits_eval_once(monkeypatch, tmp_path: Path) -> None:
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    (ckpt_dir / "update_0020.pt").write_bytes(b"20")
    monkeypatch.setattr(sup, "qstat_visible", lambda _job_id: False)
    monkeypatch.setattr(sup, "submit_eval100", lambda _run_dir: "456.pbs")

    state = sup.handle_once(tmp_path, auto_resume=True)

    assert state.phase == "resubmitted"
    assert state.job_ids[-1] == "456.pbs"
    assert state.resume_attempts
