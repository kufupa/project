from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
MSM = PROJECT / "scripts" / "maniskill_smolvla"


def _read(name: str) -> str:
    return (MSM / name).read_text(encoding="utf-8")


def test_pbs_scripts_use_pbs_and_expected_queues() -> None:
    pbs_files = [
        "00_build_envs.pbs",
        "01_data_probe.pbs",
        "02_data_full.pbs",
        "03a_audit_full.pbs",
        "03_convert_full.pbs",
        "04_sft_smoke.pbs",
        "05_sft_train.pbs",
        "06_benchmark.pbs",
        "07_autonomous_supervisor.pbs",
    ]
    for name in pbs_files:
        text = _read(name)
        assert "#PBS" in text
        assert "#SBATCH" not in text
        assert "scripts/maniskill_smolvla/common.sh" in text
        assert "artifacts/smolvla_maniskill" in text

    assert "#PBS -q v1_gpu72" in _read("01_data_probe.pbs")
    data_full = _read("02_data_full.pbs")
    assert "#PBS -q v1_large24a" in data_full
    assert "ngpus" not in data_full
    assert "#PBS -q v1_gpu72" in _read("04_sft_smoke.pbs")
    assert "#PBS -q v1_gpu72" in _read("05_sft_train.pbs")
    assert "#PBS -q v1_gpu72" in _read("06_benchmark.pbs")
    assert "#PBS -q v1_small24" in _read("07_autonomous_supervisor.pbs")


def test_common_sets_ld_library_path_for_batch_venv() -> None:
    common = _read("common.sh")
    assert "EBROOTPython" in common
    assert 'LD_LIBRARY_PATH="${EBROOTPython}/lib' in common


def test_main_sft_uses_10k_steps_and_1k_checkpoints() -> None:
    train = _read("05_sft_train.pbs")
    assert 'MSM_SFT_STEPS="${MSM_SFT_STEPS:-10000}"' in train
    assert 'MSM_SFT_SAVE_FREQ="${MSM_SFT_SAVE_FREQ:-1000}"' in train
    assert "#PBS -l walltime=16:00:00" in train
    assert "MSM_CHECKPOINT_ROOT" in _read("common.sh")


def test_common_keeps_large_artifacts_in_ephemeral() -> None:
    common = _read("common.sh")
    assert "/rds/general/user/aa6622/ephemeral/eggroll/smolvla_maniskill" in common
    assert "MSM_RAW_ROOT" in common
    assert "MSM_LEROBOT_ROOT" in common
    assert "MSM_CHECKPOINT_ROOT" in common
    assert "HF_LEROBOT_HOME" in common
    assert "RAY_TMPDIR" in common
    assert "MS_SKIP_ASSET_DOWNLOAD_PROMPT" in common


def test_collector_patch_records_state() -> None:
    collector_path = PROJECT / "RL4VLA/ManiSkill/mani_skill/examples/data_collector/vla_data_collector.py"
    if not collector_path.exists():
        return
    collector = collector_path.read_text(encoding="utf-8")
    assert '"state": []' in collector
    assert "def update_state" in collector
    assert "ee_pose_at_robot_base" in collector
    assert "matrix_to_euler_angles" in collector
    assert "self.update_state(action)" in collector


def test_converter_uses_lerobot_7d_contract() -> None:
    converter = _read("convert_npz_to_lerobot.py")
    assert "LeRobotDataset.create" in converter
    assert '"observation.images.front"' in converter
    assert '"observation.state"' in converter
    assert '"action"' in converter
    assert '"shape": (7,)' in converter
    assert '"task": instruction' in converter
    assert "dataset.finalize()" in converter
    assert "--append-completion-frames" in converter
    assert "appended_completion_frames" in converter
    assert "--state-gripper-mode" in converter
    assert 'default="previous-action"' in converter
    assert "default=0" in converter
    assert "--append-completion-frames 0" in _read("03_convert_full.pbs")
    assert "--state-gripper-mode previous-action" in _read("03_convert_full.pbs")
    assert "--dedupe-decoded-signatures" in _read("03_convert_full.pbs")
    assert "--no-filter-small-actions" in _read("03_convert_full.pbs")


def test_sft_uses_explicit_one_camera_7d_policy_contract() -> None:
    args_path = MSM / "smolvla_policy_args.sh"
    assert args_path.exists()
    args = args_path.read_text(encoding="utf-8")
    assert "observation.images.front" in args
    assert '"shape":[3,480,640]' in args.replace(" ", "")
    assert '"shape":[7]' in args.replace(" ", "")
    assert "--policy.n_action_steps=1" in args
    assert "--policy.chunk_size=50" in args
    assert "--policy.max_state_dim=32" in args
    assert "--policy.max_action_dim=32" in args
    combined = _read("04_sft_smoke.pbs") + "\n" + _read("05_sft_train.pbs")
    assert "--policy.path=lerobot/smolvla_base" in combined
    assert 'source "${MSM_SCRIPT_ROOT}/smolvla_policy_args.sh"' in combined
    assert '"${MSM_SMOLVLA_POLICY_ARGS[@]}"' in combined
    assert "check_smolvla_policy_contract.py" in _read("04_sft_smoke.pbs")
    assert "--save_freq=\"${MSM_SFT_SAVE_FREQ}\"" in combined


def test_benchmark_uses_maniskill_rollout() -> None:
    benchmark = _read("06_benchmark.pbs")
    evaluator = _read("eval_maniskill_smolvla.py")
    assert "eval_maniskill_smolvla.py" in benchmark
    assert "gym.make" in evaluator
    assert "SmolVLAPolicy.from_pretrained" in evaluator
    assert "make_pre_post_processors" in evaluator
    assert "def select_fresh_first_action" in evaluator
    assert "policy.predict_action_chunk" in evaluator
    assert "policy.select_action" not in evaluator
    assert "--sustained-success-steps" in evaluator
    assert "consecutive_successes" in evaluator
    assert 'MSM_EVAL_EPISODES="${MSM_EVAL_EPISODES:-25}"' in benchmark
    assert "success_rate" in evaluator


def test_autopilot_supports_host_retry_and_exit_status() -> None:
    autopilot = _read("autopilot.sh")
    assert "MSM_PBS_HOST" in autopilot
    assert "vnode=${host}" in autopilot
    assert "vnode=${host}:gpu_type=RTX6000" in autopilot
    assert 'tolower($1) ~ /exit_status' in autopilot
    assert 'qsub -l "select=${select_spec}"' in autopilot
    assert "03a_audit_full.pbs" in autopilot
    assert "pbs_gpu_snapshot.py" in autopilot
    assert "-q v1_gpu72" in autopilot
    assert "Failure classification hints" in autopilot
    assert "smolvla_maniskill_handoff.md" in autopilot
    assert "MSM_STAGE_MAX_RETRIES" in autopilot
    assert "exceeded max_retries" in autopilot


def test_audit_full_runs_on_cpu_pbs() -> None:
    audit = _read("03a_audit_full.pbs")
    assert "#PBS -q v1_small24" in audit
    assert "ngpus" not in audit
    assert "audit_npz_contract.py" in audit
    assert "MSM_AUDIT_FULL_DONE" in audit


def test_autonomous_supervisor_runs_under_cpu_pbs() -> None:
    supervisor = _read("07_autonomous_supervisor.pbs")
    assert "#PBS -q v1_small24" in supervisor
    assert "ngpus" not in supervisor
    assert "autopilot.sh" in supervisor
    assert "MSM_AUTOPILOT_POLL_SECONDS" in supervisor


def test_env_rebuilds_toppra_without_native_avx512() -> None:
    build_envs = _read("build_envs.sh")
    assert "MSM_REBUILD_TOPPRA_PORTABLE" in build_envs
    assert "-march=x86-64 -mtune=generic" in build_envs
    assert "--no-build-isolation --no-binary=toppra" in build_envs
    assert "toppra.constraint.linear_joint_velocity" in build_envs
    assert "MSM_PIN_DATA_NUMPY" in build_envs
    assert "numpy<2.0.0" in build_envs
    assert "mplib 0.1.1 segfaults" in build_envs


def test_gitignore_covers_large_artifacts() -> None:
    gitignore = (PROJECT / ".gitignore").read_text(encoding="utf-8")
    for pattern in ["artifacts/", "*.npz", "*.safetensors", "*.pt", "*.mp4"]:
        assert pattern in gitignore


def test_data_full_seed_is_configurable_and_manifested() -> None:
    data_full = _read("02_data_full.pbs")
    assert 'MSM_FULL_SEED="${MSM_FULL_SEED:-100}"' in data_full
    assert '--seed "${MSM_FULL_SEED}"' in data_full
    assert '"seed=${MSM_FULL_SEED}"' in data_full


def test_data_full_resume_uses_record_split_and_proc_offset() -> None:
    resume = _read("02_data_full_resume.pbs")
    assert 'MSM_RECORD_SPLIT="${MSM_RECORD_SPLIT:-16400}"' in resume
    assert "--record-split" in resume
    assert "--proc-id-offset" in resume
    assert 'MSM_PROC_ID_OFFSET="${MSM_PROC_ID_OFFSET:-32}"' in resume
    assert "MSM_RESUME_NUM_TRAJ" in resume
    assert "${MSM_RECORD_SPLIT}/data" in resume


def test_data_full_uses_large24a_record_dir_and_render_cpu() -> None:
    data_full = _read("02_data_full.pbs")
    convert = _read("03_convert_full.pbs")
    assert "ncpus=32" in data_full
    assert 'MSM_FULL_NUM_PROCS="${MSM_FULL_NUM_PROCS:-32}"' in data_full
    assert 'MSM_FULL_RECORD_DIR="${MSM_FULL_RECORD_DIR:-${MSM_RAW_ROOT}/full_cpu124_v1}"' in data_full
    assert "--sim_backend cpu" in data_full
    assert "--render_backend cpu" in data_full
    assert "full_cpu124_v1" in convert
    collect = (
        PROJECT
        / "RL4VLA/ManiSkill/mani_skill/examples/motionplanning/widowx/collect_simpler.py"
    ).read_text(encoding="utf-8")
    assert "record_split" in collect
    assert "proc_id_offset" in collect
    assert "record_split_name" in collect


def test_cleanup_requires_verified_replacement_before_delete() -> None:
    cleanup = _read("cleanup_stale_artifacts.sh")
    assert "cleanup blocked" in cleanup
    assert "MSM_EPHEMERAL_ROOT" in cleanup
    assert "audit_full.json" in cleanup
    assert "convert_full.json" in cleanup
    assert "sft_smoke.env" in cleanup
    assert "sft_smoke_policy_contract.json" in cleanup
    assert "cleanup confidence >=95%" in cleanup
    assert "rm -rf --" in cleanup
