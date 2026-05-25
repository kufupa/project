from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
PIRL = PROJECT / "scripts" / "pirl"
PBS_FILES = (
    "pirl_maniskill_2gpu_smoke.pbs",
    "pirl_maniskill_2gpu_profile.pbs",
    "pirl_maniskill_2gpu_baseline.pbs",
    "pirl_maniskill_2gpu_baseline_video.pbs",
    "pirl_maniskill_2gpu_chunk.pbs",
)
PREFLIGHT_PBS_FILES = ("pirl_maniskill_py312_tests.pbs",)
ALL_FILES = PBS_FILES + (
    "pirl_maniskill_common.sh",
    "pirl_maniskill_profile_sweep.sh",
    "pirl_maniskill_queue_next.sh",
    "pirl_python312_test.sh",
    "pirl_runtime_setup.sh",
) + PREFLIGHT_PBS_FILES
RESOURCE = "#PBS -l select=1:ncpus=64:mem=384gb:ngpus=2:gpu_type=RTX6000"
RTX6000_256GB_RESOURCE = "#PBS -l select=1:ncpus=64:mem=256gb:ngpus=2:gpu_type=RTX6000"


def _read(name: str) -> str:
    return (PIRL / name).read_text(encoding="utf-8")


def _project_read(relative_path: str) -> str:
    return (PROJECT / relative_path).read_text(encoding="utf-8")


def _walltime_to_minutes(line: str) -> int:
    value = line.split("walltime=", 1)[1].strip()
    hours, minutes, seconds = (int(part) for part in value.split(":"))
    return hours * 60 + minutes + (1 if seconds else 0)


def test_all_pbs_scripts_use_cx3_pbs_not_slurm() -> None:
    for name in PBS_FILES:
        text = _read(name)
        assert "#PBS" in text
        assert "#SBATCH" not in text
        assert "#PBS -q v1_gpu72" in text
        if name in {"pirl_maniskill_2gpu_smoke.pbs", "pirl_maniskill_2gpu_profile.pbs"}:
            assert RTX6000_256GB_RESOURCE in text
        else:
            assert RESOURCE in text
    for name in PREFLIGHT_PBS_FILES:
        text = _read(name)
        assert "#PBS" in text
        assert "#SBATCH" not in text


def test_no_python311_and_expected_modules() -> None:
    combined = "\n".join(_read(name) for name in ALL_FILES)
    assert "Python/3.11" not in combined
    assert "module load tools/prod" in combined
    assert "module load Python/3.12.3-GCCcore-13.3.0" in combined
    assert "module load Mesa/24.1.3-GCCcore-13.3.0" in combined


def test_python312_guardrails_are_enforced() -> None:
    common = _read("pirl_maniskill_common.sh")
    harness = _read("pirl_python312_test.sh")
    runtime_setup = _read("pirl_runtime_setup.sh")
    preflight = _read("pirl_maniskill_py312_tests.pbs")

    assert "pirl_assert_python312()" in common
    assert "version[:2] != (3, 12)" in common
    assert '"${PIRL_PYTHON:-python3}"' in common
    assert "pirl_assert_python312" in harness
    assert "python3 -m venv" in harness
    assert "pip install pytest pyyaml" in harness
    assert "pytest tests/test_pirl_maniskill_pbs_static.py" in harness
    assert "pytest tests/unit_tests/test_pirl_maniskill_config.py" in harness
    assert "pirl-rlinf-py312" in runtime_setup
    assert "--ignore-requires-python" in runtime_setup
    assert "git+https://github.com/RLinf/openpi" in runtime_setup
    assert "tqdm-loggable" in runtime_setup
    assert "jax==0.5.3" in runtime_setup
    assert "MS_ASSET_DIR" in common
    assert "MANISKILL_ASSET_DIR" in common
    assert "RLinf/maniskill_assets" in runtime_setup
    assert "bridge_v2_real2sim" in runtime_setup
    assert "widowx250s" in runtime_setup
    assert "pirl_python312_test.sh" in preflight


def test_python312_guard_reaches_every_pirl_pbs_entrypoint() -> None:
    for name in PBS_FILES:
        text = _read(name)
        if "pirl_maniskill_profile_sweep.sh" in text:
            continue
        assert "pirl_setup_modules" in text, name
        assert "pirl_prepare_runtime" in text, name

    assert "pirl_setup_modules" in _read("pirl_maniskill_profile_sweep.sh")
    assert "pirl_prepare_runtime" in _read("pirl_maniskill_profile_sweep.sh")
    assert "pirl_assert_python312" in _read("pirl_maniskill_common.sh")


def test_python312_venv_is_ignored() -> None:
    gitignore = _project_read(".gitignore")
    assert ".venv/" in gitignore
    assert ".venvs/" in gitignore


def test_gpu_snapshot_and_jupyter_exclusion_are_present() -> None:
    combined = "\n".join(_read(name) for name in ALL_FILES)
    assert "$HOME/.agents/skills/checking-pbs-gpu-availability/scripts/pbs_gpu_snapshot.py" in combined
    assert "-q v1_gpu72" in combined
    assert "v1_jupytergpu" in combined


def test_walltimes_are_within_task3_contract() -> None:
    for name in PBS_FILES:
        text = _read(name)
        walltime_lines = [line for line in text.splitlines() if line.startswith("#PBS -l walltime=")]
        assert walltime_lines, name
        for line in walltime_lines:
            assert _walltime_to_minutes(line) <= 150, (name, line)

    assert "#PBS -l walltime=02:30:00" in _read("pirl_maniskill_2gpu_chunk.pbs")
    assert "#PBS -l walltime=02:30:00" in _read("pirl_maniskill_2gpu_profile.pbs")
    assert "#PBS -l walltime=02:30:00" in _read("pirl_maniskill_2gpu_baseline.pbs")


def test_common_uses_native_rlinf_entrypoint_and_paths() -> None:
    common = _read("pirl_maniskill_common.sh")
    assert "examples/embodiment/train_embodied_agent.py" in common
    assert "--config-name \"${RLINF_CONFIG}\"" in common
    assert "PROJECT_ROOT:-/rds/general/user/aa6622/home/project" in common
    assert 'RLINF_ROOT:-${PROJECT_ROOT}/RLinf' in common
    assert "maniskill_ppo_openpi_pi05_rtx6000_flow_sde" in common
    assert "hf_models/RLinf-Pi05-ManiSkill-25Main-SFT" in common
    assert "snapshot_download" in common
    assert "RLinf/RLinf-Pi05-ManiSkill-25Main-SFT" in common
    assert "PIRL_ARTIFACT_ROOT}/runs/${job_tag}" in common
    assert "/tmp/pirl_${USER}_${job_tag}/ray_tmp" in common
    assert "RAY_TMPDIR" in common
    assert 'PIRL_ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts/pirl_maniskill' in common
    assert 'PIRL_RUNTIME_VENV:-${PROJECT_ROOT}/.venvs/pirl-rlinf-py312' in common
    assert 'RAY_TMPDIR:-/tmp/pirl_${USER}_${job_tag}/ray_tmp' in common
    assert 'EMBODIED_PATH:-${RLINF_ROOT}/examples/embodiment' in common
    assert 'PYTHONPATH="${RLINF_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"' in common


def test_profile_script_runs_sweep_and_planned_shapes() -> None:
    profile_pbs = _read("pirl_maniskill_2gpu_profile.pbs")
    sweep = _read("pirl_maniskill_profile_sweep.sh")

    assert "pirl_maniskill_profile_sweep.sh" in profile_pbs
    assert "PIRL_PROFILE_ENVS:-16 32 48 64 80 96 128 160 192 224 256 320" in sweep
    for env_count in ("16", "32", "48", "64", "80", "96", "128", "160", "192", "224", "256", "320"):
        assert env_count in sweep

    for env_count, rollout_epoch, max_steps, global_batch in (
        ("64", "5", "80", "5120"),
        ("80", "4", "80", "5120"),
        ("128", "5", "40", "5120"),
        ("160", "2", "80", "5120"),
        ("320", "1", "80", "5120"),
    ):
        assert f"{env_count})" in sweep
        assert f'rollout_epoch="${{rollout_epoch:-{rollout_epoch}}}"' in sweep
        assert f'max_steps="${{max_steps:-{max_steps}}}"' in sweep
        assert f'global_batch="${{global_batch:-{global_batch}}}"' in sweep
    assert "small smoke fallback" in sweep
    assert "planned 5120 shape" in sweep


def test_no_custom_trainers_are_called() -> None:
    combined = "\n".join(_read(name) for name in ALL_FILES)
    assert "examples/embodiment/train_embodied_agent.py" in combined
    for forbidden in (
        "run_training_chunk.py",
        "profile_maniskill_envs.py",
        "run_zero_shot_baseline.py",
    ):
        assert forbidden not in combined


def test_common_exposes_required_functions_and_safe_ray_cleanup() -> None:
    common = _read("pirl_maniskill_common.sh")
    for fn_name in (
        "pirl_setup_modules",
        "pirl_assert_python312",
        "pirl_resolve_paths",
        "pirl_prepare_runtime",
        "pirl_gpu_snapshot",
        "pirl_cleanup_ray",
        "pirl_run_rlinf",
        "pirl_latest_checkpoint",
    ):
        assert f"{fn_name}()" in common
    assert "ray stop --force || true" in common
    assert "pkill -9 -f ray" not in common
    for name in ALL_FILES:
        assert "pkill -9 -f ray" not in _read(name)


def test_smoke_overrides_are_tiny() -> None:
    smoke = _read("pirl_maniskill_2gpu_smoke.pbs")
    assert "runner.max_epochs=1" in smoke
    assert "runner.val_check_interval=-1" in smoke
    assert "runner.save_interval=-1" in smoke
    assert "env.train.total_num_envs=2" in smoke
    assert "env.eval.total_num_envs=2" in smoke
    assert "env.train.max_steps_per_rollout_epoch=5" in smoke
    assert "env.eval.max_steps_per_rollout_epoch=5" in smoke
    assert "algorithm.rollout_epoch=1" in smoke
    assert "actor.micro_batch_size=1" in smoke
    assert "actor.global_batch_size=2" in smoke
    assert "env.eval.video_cfg.save_video=False" in smoke


def test_baselines_video_and_numeric_contracts() -> None:
    numeric = _read("pirl_maniskill_2gpu_baseline.pbs")
    video = _read("pirl_maniskill_2gpu_baseline_video.pbs")

    assert "runner.only_eval=True" in numeric
    assert "env.eval.video_cfg.save_video=False" in numeric
    assert "env.eval.use_fixed_reset_state_ids=True" in numeric
    assert "env.eval.ignore_terminations=True" in numeric
    assert "PIRL_BASELINE_ENVS:-64" in numeric

    assert "runner.only_eval=True" in video
    assert "env.eval.video_cfg.save_video=True" in video
    assert "baseline_success" in video
    assert "PIRL_VIDEO_ENVS:-1" in video
    assert "PIRL_VIDEO_ATTEMPTS:-3" in video
    assert "SUCCESS_VIDEO_VERIFIED" in video
    assert "attribution ambiguous" in video
    assert "no verified success video found" in video
    assert "copied_count=0" in video
    assert '"${copied_count}" -gt 0' in video
    assert "success metric observed but no mp4 found; not marking verified" in video


def test_chunk_and_queue_contracts() -> None:
    chunk = _read("pirl_maniskill_2gpu_chunk.pbs")
    queue = _read("pirl_maniskill_queue_next.sh")

    assert "PIRL_TRAIN_ENVS:-64" in chunk
    assert "PIRL_ROLLOUT_EPOCH:-5" in chunk
    assert "PIRL_MAX_STEPS:-80" in chunk
    assert "PIRL_MICRO_BATCH:-2" in chunk
    assert "PIRL_GLOBAL_BATCH:-5120" in chunk
    assert "PIRL_RESUME_DIR" in chunk
    assert "PIRL_MAX_EPOCHS" in chunk
    assert "PIRL_SAVE_INTERVAL" in chunk
    assert "PIRL_VAL_CHECK_INTERVAL" in chunk
    assert "pirl_maniskill_queue_next.sh" in chunk
    assert "PIRL_CHUNK_TIMEOUT:-2h20m" in chunk
    assert "timeout --preserve-status" in chunk
    assert "partial chunk" in chunk

    assert 'PIRL_AUTO_QUEUE:-0}" != "1"' in queue
    assert "PIRL_MAX_CHUNKS:-4" in queue
    assert "PIRL_CHUNK_INDEX" in queue
    assert "PIRL_GATE_STATUS" in queue
    assert "/HALT" in queue
    assert '"${PIRL_GATE_STATUS:-}" != "pass"' in queue
    assert "/GATE_PASS" in queue
    assert "PIRL_GATE_STATUS=pass,PIRL_CHUNK_INDEX" in queue
    assert "pirl_latest_checkpoint" in queue
    assert "qsub -q v1_gpu72" in queue
    assert "PIRL_RESUME_DIR=" in queue


def test_profile_sweep_records_failures_and_continues() -> None:
    sweep = _read("pirl_maniskill_profile_sweep.sh")
    assert "profile_status.tsv" in sweep
    assert "trial failed/OOM-like; continuing" in sweep
    assert "continue" in sweep
