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
        "03_convert_full.pbs",
        "04_sft_smoke.pbs",
        "05_sft_train.pbs",
        "06_benchmark.pbs",
    ]
    for name in pbs_files:
        text = _read(name)
        assert "#PBS" in text
        assert "#SBATCH" not in text
        assert "scripts/maniskill_smolvla/common.sh" in text
        assert "artifacts/smolvla_maniskill" in text

    assert "#PBS -q v1_gpu72" in _read("01_data_probe.pbs")
    assert "#PBS -q v1_gpu72" in _read("02_data_full.pbs")
    assert "#PBS -q v1_gpu72" in _read("04_sft_smoke.pbs")
    assert "#PBS -q v1_gpu72" in _read("05_sft_train.pbs")
    assert "#PBS -q v1_gpu72" in _read("06_benchmark.pbs")


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
    assert "default=15" in converter
    assert "--stop-after-success-count 15" in _read("03_convert_full.pbs")


def test_sft_keeps_pretrained_padding_dims() -> None:
    combined = _read("04_sft_smoke.pbs") + "\n" + _read("05_sft_train.pbs")
    assert "--policy.path=lerobot/smolvla_base" in combined
    assert "--policy.max_state_dim=32" in combined
    assert "--policy.max_action_dim=32" in combined
    assert "--save_freq=\"${MSM_SFT_SAVE_FREQ}\"" in combined


def test_benchmark_uses_maniskill_rollout() -> None:
    benchmark = _read("06_benchmark.pbs")
    evaluator = _read("eval_maniskill_smolvla.py")
    assert "eval_maniskill_smolvla.py" in benchmark
    assert "gym.make" in evaluator
    assert "SmolVLAPolicy.from_pretrained" in evaluator
    assert "make_pre_post_processors" in evaluator
    assert "policy.select_action" in evaluator
    assert "success_rate" in evaluator


def test_autopilot_supports_host_retry_and_exit_status() -> None:
    autopilot = _read("autopilot.sh")
    assert "MSM_PBS_HOST" in autopilot
    assert "vnode=${host}" in autopilot
    assert "vnode=${host}:gpu_type=RTX6000" in autopilot
    assert 'tolower($1) ~ /exit_status' in autopilot
    assert 'qsub -l "select=${select_spec}"' in autopilot


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
