from __future__ import annotations

from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]


def _read(name: str) -> str:
    return (PROJECT / "scripts" / "grpo" / name).read_text(encoding="utf-8")


def test_raw_vs_bounded_decode_pbs_contract() -> None:
    text = _read("submit_phase12_raw_vs_bounded_decode_6ep.pbs")

    assert "#SBATCH" not in text
    assert "PBS_O_WORKDIR" in text
    assert "#PBS -l select=1:ncpus=8:mem=48gb:ngpus=1:gpu_type=RTX6000" in text
    assert "module load tools/prod" in text
    assert "module load Python/3.12.3-GCCcore-13.3.0" in text
    assert "module load Mesa/24.1.3-GCCcore-13.3.0" in text
    assert 'slurm_resolve_project_root "scripts/grpo/compare_phase12_raw_vs_bounded_decode.py"' in text
    assert "slurm_export_pythonpath" in text
    assert 'export MUJOCO_GL="${MUJOCO_GL:-osmesa}"' in text
    assert 'export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"' in text
    assert 'export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"' in text
    assert 'export JEPA_WM_DISABLE_IMAGE_HEAD="${JEPA_WM_DISABLE_IMAGE_HEAD:-0}"' in text
    assert 'NUM_EPISODES="${PHASE12_NUM_EPISODES:-6}"' in text
    assert 'CHUNK_LEN="${PHASE12_CHUNK_LEN:-50}"' in text
    assert "--num-episodes \"${NUM_EPISODES}\"" in text
    assert "--chunk-len \"${CHUNK_LEN}\"" in text
    assert "compare_summary.json" in text
    assert "real_raw_bounded_decode_strip.png" in text
    assert "actions_raw_bounded_env.npz" in text
    assert "PHASE12_RAW_VS_BOUNDED_DECODE_DONE" in text


def test_phase56_oracle_action_audit_pbs_contract() -> None:
    text = _read("submit_phase56_oracle_action_audit.pbs")

    assert "#SBATCH" not in text
    assert "PBS_O_WORKDIR" in text
    assert "#PBS -l select=1:ncpus=8:mem=16gb:ngpus=1:gpu_type=RTX6000" in text
    assert "#PBS -l walltime=00:05:00" in text
    assert "module load tools/prod" in text
    assert "module load Python/3.12.3-GCCcore-13.3.0" in text
    assert "module load Mesa/24.1.3-GCCcore-13.3.0" in text
    assert 'slurm_resolve_project_root "scripts/grpo/phase56_oracle_action_audit.py"' in text
    assert "slurm_export_pythonpath" in text
    assert 'export MUJOCO_GL="${MUJOCO_GL:-osmesa}"' in text
    assert 'export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"' in text
    assert 'TASK="${PHASE56_TASK:-push-v3}"' in text
    assert 'SEED="${PHASE56_SEED:-2000}"' in text
    assert 'MAX_STEPS="${PHASE56_MAX_STEPS:-120}"' in text
    assert "--task \"${TASK}\"" in text
    assert "--seed \"${SEED}\"" in text
    assert "--max-steps \"${MAX_STEPS}\"" in text
    assert "oracle_action_range_summary.json" in text
    assert "oracle_actions_raw.npy" in text
    assert "oracle_action_steps.csv" in text
    assert "PHASE56_ORACLE_ACTION_AUDIT_DONE" in text


def test_phase57_mt50_decode_pbs_contract() -> None:
    text = _read("submit_phase57_mt50_decode_shard.pbs")
    all5 = _read("submit_phase57_mt50_decode_all5.sh")

    assert "#SBATCH" not in text
    assert "PBS_O_WORKDIR" in text
    assert "#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000" in text
    assert "#PBS -l walltime=16:00:00" in text
    assert "module load tools/prod" in text
    assert "module load Python/3.12.3-GCCcore-13.3.0" in text
    assert "module load Mesa/24.1.3-GCCcore-13.3.0" in text
    assert 'slurm_resolve_project_root "scripts/grpo/phase57_mt50_raw_vs_bounded_decode.py"' in text
    assert 'export JEPA_WM_DISABLE_IMAGE_HEAD="${JEPA_WM_DISABLE_IMAGE_HEAD:-0}"' in text
    assert 'export SMOLVLA_METAWORLD_RESET_MODE="${SMOLVLA_METAWORLD_RESET_MODE:-random_seeded}"' in text
    assert 'SHARD_COUNT="${PHASE57_SHARD_COUNT:-5}"' in text
    assert 'EPISODES="${PHASE57_EPISODES:-25}"' in text
    assert 'N_ENVS="${PHASE57_N_ENVS:-3}"' in text
    assert 'CHUNK_LEN="${PHASE57_CHUNK_LEN:-50}"' in text
    assert 'MAX_STEPS="${PHASE57_MAX_STEPS:-180}"' in text
    assert "--n-envs \"${N_ENVS}\"" in text
    assert "--episodes \"${EPISODES}\"" in text
    assert "--chunk-len \"${CHUNK_LEN}\"" in text
    assert "--max-steps \"${MAX_STEPS}\"" in text
    assert "MT50_Phase57_raw_vs_bounded_decode_25ep_s1000_max180_5x1gpu" in text
    assert "PHASE57_MT50_DECODE_SHARD_DONE" in text
    assert "for shard in 0 1 2 3 4" in all5


def test_phase57_raw_boundary_goal_bugfix_pbs_contract() -> None:
    text = _read("submit_phase57_raw_boundary_check_goal_bugfix.pbs")

    assert "#SBATCH" not in text
    assert "PBS_O_WORKDIR" in text
    assert "#PBS -l select=1:ncpus=32:mem=128gb:ngpus=1:gpu_type=RTX6000" in text
    assert "#PBS -l walltime=04:00:00" in text
    assert 'slurm_resolve_project_root "scripts/grpo/phase57_mt50_raw_vs_bounded_decode.py"' in text
    assert 'export SMOLVLA_METAWORLD_RESET_MODE="${SMOLVLA_METAWORLD_RESET_MODE:-random_seeded}"' in text
    assert "phase57-raw-boundary-check-goal-bugfix" in text
    assert "RAW_BOUNDARY_TASKS_DEFAULT=" in text
    assert text.count("-v3") == 18
    assert "coffee-pull-v3" in text
    assert "stick-push-v3" in text
    assert 'EPISODES="${PHASE57_RAW_BOUNDARY_EPISODES:-25}"' in text
    assert 'N_ENVS="${PHASE57_RAW_BOUNDARY_N_ENVS:-25}"' in text
    assert 'CHUNK_LEN="${PHASE57_RAW_BOUNDARY_CHUNK_LEN:-50}"' in text
    assert 'MAX_STEPS="${PHASE57_RAW_BOUNDARY_MAX_STEPS:-180}"' in text
    assert 'GPU_TELEMETRY_DIR="${OUT}/gpu_telemetry"' in text
    assert 'GPU_TELEMETRY_INTERVAL="${PHASE57_RAW_BOUNDARY_GPU_TELEMETRY_INTERVAL:-5}"' in text
    assert "nvidia_smi_samples.csv" in text
    assert "nvidia_smi_pmon.txt" in text
    assert "nvidia_smi_start.txt" in text
    assert "nvidia_smi_end.txt" in text
    assert "pbs_resource_snapshot.txt" in text
    assert "summarize_phase58_gpu_telemetry.py" in text
    assert "gpu_telemetry_summary.json" in text
    assert "command -v nvidia-smi" in text
    assert "command -v dcgmi" in text
    assert "stop_gpu_telemetry" in text
    assert "trap stop_gpu_telemetry EXIT" in text
    assert "exit \"${status}\"" in text
    assert "--n-envs \"${N_ENVS}\"" in text
    assert "--episodes \"${EPISODES}\"" in text
    assert "--chunk-len \"${CHUNK_LEN}\"" in text
    assert "--max-steps \"${MAX_STEPS}\"" in text
    assert "phase57_raw_boundary_check_goal_bugfix_18tasks_25ep_s1000_max180_nenv25_1xgpu" in text
    assert "PHASE57_RAW_BOUNDARY_CHECK_GOAL_BUGFIX_DONE" in text


def test_smolvla_pushv3_chunk_sweep_pbs_contract() -> None:
    text = _read("submit_smolvla_baseline_pushv3_chunk_sweep.pbs")
    all5 = _read("submit_smolvla_baseline_pushv3_chunk_sweep_all5.sh")

    assert "#SBATCH" not in text
    assert "PBS_O_WORKDIR" in text
    assert "#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000" in text
    assert "#PBS -l walltime=02:00:00" in text
    assert 'export SMOLVLA_METAWORLD_RESET_MODE="${SMOLVLA_METAWORLD_RESET_MODE:-random_seeded}"' in text
    assert 'TASK="${PHASE58_TASK:-push-v3}"' in text
    assert 'EPISODES="${PHASE58_EPISODES:-25}"' in text
    assert 'N_ENVS="${PHASE58_N_ENVS:-3}"' in text
    assert 'MAX_STEPS="${PHASE58_MAX_STEPS:-180}"' in text
    assert 'CHUNK_LEN="${PHASE58_CHUNK_LEN:-2}"' in text
    assert "--chunk-len \"${CHUNK_LEN}\"" in text
    assert 'GPU_TELEMETRY_DIR="${OUT}/gpu_telemetry"' in text
    assert 'GPU_TELEMETRY_INTERVAL="${PHASE58_GPU_TELEMETRY_INTERVAL:-5}"' in text
    assert "nvidia_smi_samples.csv" in text
    assert "nvidia_smi_pmon.txt" in text
    assert "nvidia_smi_start.txt" in text
    assert "nvidia_smi_end.txt" in text
    assert "pbs_resource_snapshot.txt" in text
    assert "summarize_phase58_gpu_telemetry.py" in text
    assert "gpu_telemetry_summary.json" in text
    assert 'test -f "${OUT}/timing_summary.json"' in text
    assert "timing_summary.json" in text
    assert "command -v nvidia-smi" in text
    assert "command -v dcgmi" in text
    assert "stop_gpu_telemetry" in text
    assert "trap stop_gpu_telemetry EXIT" in text
    assert "|| true" in text
    assert "phase58_smolvla_baseline_pushv3_chunk_sweep_25ep_s1000_max180_5x1gpu" in text
    assert "SMOLVLA_PUSHV3_CHUNK_SWEEP_DONE" in text
    assert "chunks=(2 5 10 15 20)" in all5
    assert "qsub" in all5
    assert "PHASE58_CHUNK_LEN=${chunk}" in all5


def test_smolvla_true_parallel_pushv3_chunk5_pbs_contract() -> None:
    text = _read("submit_smolvla_true_parallel_pushv3_chunk5.pbs")

    assert "#SBATCH" not in text
    assert "PBS_O_WORKDIR" in text
    assert "#PBS -l select=1:ncpus=16:mem=64gb:ngpus=1:gpu_type=RTX6000" in text
    assert "#PBS -l walltime=02:00:00" in text
    assert 'EPISODES="${PHASE61_EPISODES:-8}"' in text
    assert 'N_ENVS="${PHASE61_N_ENVS:-8}"' in text
    assert 'SEED_START="${PHASE61_EVAL_SEED_START:-1000}"' in text
    assert 'CHUNK_LEN="${PHASE61_CHUNK_LEN:-5}"' in text
    assert 'ENV_VECTOR_MODE="${PHASE61_ENV_VECTOR_MODE:-async}"' in text
    assert "--output-dir \"${OUT}\"" in text
    assert "--env-vector-mode \"${ENV_VECTOR_MODE}\"" in text
    assert "OMP_NUM_THREADS" in text
    assert "SMOLVLA_GRPO_ASYNC_MP_CONTEXT" in text
    assert "gpu_telemetry_summary.json" in text
    assert "timing_summary.json" in text
