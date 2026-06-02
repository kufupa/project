from __future__ import annotations

from pathlib import Path

from scripts.grpo import supervise_phase12_wm_grpo_overnight as sup
from scripts.grpo import supervise_phase12_pure_wm_overnight as pure_sup


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
    monkeypatch.setattr(sup, "read_recent_logs", lambda _run_dir, _job_ids=None: "PBS: job killed after walltime limit")

    cause, symptom = sup.classify_failure(tmp_path)

    assert cause == "pbs_walltime_timeout"
    assert "update_0010.pt" in symptom


def test_classify_failure_detects_cuda_oom(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(sup, "read_recent_logs", lambda _run_dir, _job_ids=None: "RuntimeError: CUDA out of memory")

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
    monkeypatch.setattr(sup, "read_recent_logs", lambda _run_dir, _job_ids=None: "resources_used.walltime exceeded")

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
    monkeypatch.setattr(sup, "read_recent_logs", lambda _run_dir, _job_ids=None: "RuntimeError: deterministic crash")

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


def test_latest_checkpoint_falls_back_to_payload_when_meta_missing(tmp_path: Path) -> None:
    import torch

    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    latest = ckpt_dir / "latest.pt"
    torch.save({"update_index": 3}, latest)

    path, start_update = sup.latest_checkpoint(tmp_path)

    assert path == latest
    assert start_update == 4


def test_read_recent_logs_scopes_to_run_dir_and_job_id(monkeypatch, tmp_path: Path) -> None:
    log_dir = tmp_path / "logs" / "pbs" / "grpo"
    log_dir.mkdir(parents=True)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (log_dir / "phase12_other.log").write_text("walltime other run", encoding="utf-8")
    (log_dir / "phase12_train_chunk_10u.12345.log").write_text("job-specific walltime", encoding="utf-8")
    (log_dir / "phase12_run.log").write_text(f"run={run_dir} CUDA out of memory", encoding="utf-8")
    monkeypatch.setattr(sup, "PROJECT_ROOT", tmp_path)

    text = sup.read_recent_logs(run_dir, ["12345.pbs-7"])

    assert "job-specific walltime" in text
    assert "CUDA out of memory" in text
    assert "other run" not in text


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


def test_pure_wm_supervisor_classifies_eval_only_recovery(tmp_path: Path) -> None:
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    (ckpt_dir / "update_0020.pt").write_bytes(b"20")

    diagnosis = pure_sup.diagnose_run(tmp_path)

    assert diagnosis["phase"] == "needs_eval_resume"
    assert diagnosis["known_safe"] is True


def test_pure_wm_qsub_resume_passes_all_hparams(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, object] = {}
    resume = tmp_path / "checkpoints" / "update_0010.pt"
    resume.parent.mkdir()
    resume.write_bytes(b"10")

    class Result:
        stdout = "12345.pbs\n"

    def fake_run(cmd, *, cwd, check, text, capture_output):
        seen["cmd"] = cmd
        seen["cwd"] = cwd
        seen["check"] = check
        seen["text"] = text
        seen["capture_output"] = capture_output
        return Result()

    monkeypatch.setattr(pure_sup.subprocess, "run", fake_run)

    job_id = pure_sup.qsub_train_eval(
        tmp_path,
        root_mode="oracle_teacher_forced",
        loss_mode="group_sqrt_segments",
        action_l2=0.003,
        lr="1e-5",
        clip_eps="0.2",
        init_log_std="-2.0",
        euler_noise="0.2",
        resume=resume,
        start_update=10,
        expected_update=20,
    )

    varlist = seen["cmd"][2]
    assert job_id == "12345.pbs"
    assert "PHASE12_WM_ONLY_ROOT_MODE=oracle_teacher_forced" in varlist
    assert "PHASE12_LOSS_NORMALIZER_MODE=group_sqrt_segments" in varlist
    assert "PHASE12_WM_ACTION_L2_PENALTY=0.003" in varlist
    assert f"PHASE12_RESUME={resume}" in varlist
    assert "PHASE12_START_UPDATE=10" in varlist
    assert "PHASE12_NUM_UPDATES=10" in varlist
