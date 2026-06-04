"""Lightweight checks on Phase11 Slurm / shell entrypoints."""

from __future__ import annotations

import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]


def test_submit_phase11_grpo_uses_explicit_python_and_rollout_smoke() -> None:
    text = (_REPO_ROOT / "scripts" / "grpo" / "submit_phase11_grpo.slurm").read_text(encoding="utf-8")
    assert "GRPO_PYTHON=" in text
    assert 'exec "${GRPO_PYTHON}"' in text
    assert "rollout-smoke" in text
    assert "smoke_phase11_rollout.py" in text
    assert "_PHASE11_PROJECT_FALLBACK" in text
    assert "scripts/slurm/common_env.sh" in text


def test_submit_phase11_chain_uses_chdir() -> None:
    text = (_REPO_ROOT / "scripts" / "grpo" / "submit_phase11_chain.sh").read_text(encoding="utf-8")
    assert "--chdir=" in text
    assert "PROJECT_CHDIR" in text


def test_slurm_and_chain_scripts_bash_syntax() -> None:
    grpo = _REPO_ROOT / "scripts" / "grpo"
    for name in (
        "submit_phase11_grpo.slurm",
        "submit_phase111_on_grpo_lerobot_smoke.slurm",
        "submit_phase111_single_task_grpo.slurm",
        "submit_phase111_vector_rollout_smoke.slurm",
        "submit_flow_sde_chunk_grpo_smoke_a30.slurm",
        "submit_flow_sde_chunk_grpo_train16_a30.slurm",
        "submit_flow_sde_chunk_grpo_train10_dense_a30.slurm",
        "submit_flow_sde_chunk_grpo_train10_sparse_a30.slurm",
        "submit_flow_sde_chunk_grpo_train10_success_first_dense_a30.slurm",
        "submit_flow_sde_chunk_grpo_resume30_a30.slurm",
        "submit_flow_sde_chunk_grpo_eval25_a30.slurm",
        "submit_flow_sde_chunk_grpo_eval25_ablation_a30.slurm",
        "submit_flow_sde_chunk_grpo_eval100_ablation_a30.slurm",
        "submit_flow_sde_chunk_grpo_eval25_eval100_chain_a30.slurm",
        "submit_flow_sde_chunk_grpo_moonshot30_dense_chain_a30.slurm",
        "submit_flow_sde_chunk_grpo_moonshot30_sparse_chain_a30.slurm",
        "submit_phase111_eval_sweep.slurm",
        "submit_phase11_chain.sh",
        "submit_api_gate_smoke.slurm",
        "submit_phase11_eval_sweep_seed1020.slurm",
        "submit_phase11_eval_sweep_seed1000_ge50.slurm",
    ):
        path = grpo / name
        subprocess.run(["bash", "-n", str(path)], check=True, cwd=str(_REPO_ROOT))
    common = _REPO_ROOT / "scripts" / "slurm" / "common_env.sh"
    subprocess.run(["bash", "-n", str(common)], check=True, cwd=str(_REPO_ROOT))


def test_submit_phase111_on_grpo_lerobot_smoke_is_official_backend() -> None:
    text = (_REPO_ROOT / "scripts" / "grpo" / "submit_phase111_on_grpo_lerobot_smoke.slurm").read_text(
        encoding="utf-8"
    )
    assert "#SBATCH --export=NIL" in text
    assert "phase111_on_grpo_lerobot_smoke" in text
    assert "--env-backend official_lerobot" in text
    assert 'MAX_STEPS="${GRPO_PHASE111_MAX_STEPS:-0}"' in text
    assert '--max-steps "${MAX_STEPS}"' in text
    assert "PHASE111_GRPO_LEROBOT_ARTIFACTS_OK" in text
    assert "PHASE111_GRPO_LEROBOT_SMOKE_OK" in text


def test_submit_phase111_vector_rollout_smoke_trains_sync_and_async() -> None:
    path = _REPO_ROOT / "scripts" / "grpo" / "submit_phase111_vector_rollout_smoke.slurm"
    text = path.read_text(encoding="utf-8")
    assert "#SBATCH --export=NIL" in text
    assert "scripts/slurm/common_env.sh" in text
    assert "--rollout-execution vector_sync" in text
    assert "--rollout-execution vector_async" in text
    assert "--async-start-method forkserver" in text
    assert "PHASE111_VECTOR_ROLLOUT_SMOKE_OK" in text
    assert 'm_async.get("async_start_method") == "forkserver"' in text
    subprocess.run(["bash", "-n", str(path)], check=True, cwd=str(_REPO_ROOT))


def test_submit_flow_sde_chunk_grpo_smoke_uses_chunk_flow_mode() -> None:
    path = _REPO_ROOT / "scripts" / "grpo" / "submit_flow_sde_chunk_grpo_smoke_a30.slurm"
    text = path.read_text(encoding="utf-8")
    assert "#SBATCH --export=NIL" in text
    assert "scripts/slurm/common_env.sh" in text
    assert "--rollout-unit chunk" in text
    assert "--rollout-chunk-len 5" in text
    assert "--rollout-execution serial" in text
    assert "--logprob-mode flow_sde" in text
    assert "--flow-sde-noise-level 0.5" in text
    assert "--flow-sde-trace-step -1" in text
    assert "--fail-on-parity-violation" in text
    assert 'test -f "${OUT}/checkpoints_eval/update_0001.pt"' in text
    assert "FLOW_SDE_CHUNK_GRPO_SMOKE_OK" in text
    subprocess.run(["bash", "-n", str(path)], check=True, cwd=str(_REPO_ROOT))


def test_submit_flow_sde_chunk_grpo_train16_has_gate_contract() -> None:
    path = _REPO_ROOT / "scripts" / "grpo" / "submit_flow_sde_chunk_grpo_train16_a30.slurm"
    text = path.read_text(encoding="utf-8")
    assert "#SBATCH --export=NIL" in text
    assert "scripts/slurm/common_env.sh" in text
    assert "--rollout-unit chunk" in text
    assert "--rollout-chunk-len 5" in text
    assert "--rollout-execution serial" in text
    assert "--num-updates 16" in text
    assert 'test -f "${OUT}/checkpoints/update_0016.pt"' in text
    assert 'test -f "${OUT}/checkpoints_eval/update_0016.pt"' in text
    assert "FLOW_SDE_CHUNK_GRPO_TRAIN16_OK" in text
    subprocess.run(["bash", "-n", str(path)], check=True, cwd=str(_REPO_ROOT))


def test_flow_sde_chunk_train10_ablation_scripts_set_reward_modes() -> None:
    scripts = {
        "submit_flow_sde_chunk_grpo_train10_dense_a30.slurm": "dense_return",
        "submit_flow_sde_chunk_grpo_train10_sparse_a30.slurm": "sparse_success_delta",
        "submit_flow_sde_chunk_grpo_train10_success_first_dense_a30.slurm": "success_first_dense",
    }
    for name, reward_mode in scripts.items():
        path = _REPO_ROOT / "scripts" / "grpo" / name
        text = path.read_text(encoding="utf-8")
        assert "#SBATCH --gres=gpu:1" in text
        assert "#SBATCH --cpus-per-task=10" in text
        assert "#SBATCH --mem=48G" in text
        assert "--rollout-unit chunk" in text
        assert "--rollout-chunk-len 5" in text
        assert "--rollout-execution vector_async" in text
        assert "--async-start-method forkserver" in text
        assert "--num-updates 10" in text
        assert f"--reward-mode {reward_mode}" in text
        assert "--fail-on-parity-violation" in text
        assert 'if not r.get("skipped")' in text
        assert "FLOW_SDE_CHUNK_GRPO_TRAIN10_OK" in text
        subprocess.run(["bash", "-n", str(path)], check=True, cwd=str(_REPO_ROOT))


def test_flow_sde_chunk_resume30_script_resumes_and_saves_every5() -> None:
    path = _REPO_ROOT / "scripts" / "grpo" / "submit_flow_sde_chunk_grpo_resume30_a30.slurm"
    text = path.read_text(encoding="utf-8")
    assert "#SBATCH --gres=gpu:1" in text
    assert "#SBATCH --cpus-per-task=10" in text
    assert "#SBATCH --mem=48G" in text
    assert 'RESUME="${1:?resume checkpoint}"' in text
    assert 'OUT="${2:?output dir}"' in text
    assert 'REWARD_MODE="${3:?reward mode}"' in text
    assert 'RUN_LABEL="${4:?run label}"' in text
    assert "--resume" in text
    assert "--num-updates 30" in text
    assert "--save-every 5" in text
    assert "--rollout-execution vector_async" in text
    assert "--async-start-method forkserver" in text
    assert "--rollout-unit chunk" in text
    assert "--reward-mode" in text
    assert 'test -f "${OUT}/checkpoints/update_0040.pt"' in text
    assert "FLOW_SDE_CHUNK_GRPO_RESUME30_OK" in text
    subprocess.run(["bash", "-n", str(path)], check=True, cwd=str(_REPO_ROOT))


def test_flow_sde_chunk_eval25_ablation_script_uses_bounded_resources() -> None:
    path = _REPO_ROOT / "scripts" / "grpo" / "submit_flow_sde_chunk_grpo_eval25_ablation_a30.slurm"
    text = path.read_text(encoding="utf-8")
    assert "#SBATCH --gres=gpu:1" in text
    assert "#SBATCH --cpus-per-task=10" in text
    assert "#SBATCH --mem=60G" in text
    assert "--num-episodes 25" in text
    assert "--chunk-len 5" in text
    assert '--checkpoint-dir "${CKPT_DIR}"' in text
    assert "FLOW_SDE_CHUNK_GRPO_EVAL25_ABLATION_OK" in text
    subprocess.run(["bash", "-n", str(path)], check=True, cwd=str(_REPO_ROOT))


def test_flow_sde_chunk_eval100_ablation_script_uses_bounded_resources() -> None:
    path = _REPO_ROOT / "scripts" / "grpo" / "submit_flow_sde_chunk_grpo_eval100_ablation_a30.slurm"
    text = path.read_text(encoding="utf-8")
    assert "#SBATCH --gres=gpu:1" in text
    assert "#SBATCH --cpus-per-task=10" in text
    assert "#SBATCH --mem=60G" in text
    assert "--num-episodes 100" in text
    assert "--num-envs 25" in text
    assert 'ONLY_UPDATES="${4:-}"' in text
    assert "--only-updates" in text
    assert "--chunk-len 5" in text
    assert '--checkpoint-dir "${CKPT_DIR}"' in text
    assert "eval100_summary.json" in text
    assert "FLOW_SDE_CHUNK_GRPO_EVAL100_ABLATION_OK" in text
    subprocess.run(["bash", "-n", str(path)], check=True, cwd=str(_REPO_ROOT))


def test_flow_sde_chunk_eval25_eval100_chain_script_runs_both_sweeps() -> None:
    path = _REPO_ROOT / "scripts" / "grpo" / "submit_flow_sde_chunk_grpo_eval25_eval100_chain_a30.slurm"
    text = path.read_text(encoding="utf-8")
    assert "#SBATCH --gres=gpu:1" in text
    assert "#SBATCH --cpus-per-task=10" in text
    assert "#SBATCH --mem=60G" in text
    assert 'CKPT_DIR="${1:?checkpoint dir}"' in text
    assert 'OUT25_DIR="${2:?25ep output dir}"' in text
    assert 'OUT100_DIR="${4:?100ep output dir}"' in text
    assert "--num-episodes 25" in text
    assert "--num-episodes 100" in text
    assert "--only-updates" in text
    assert "eval25_summary.json" in text
    assert "eval100_summary.json" in text
    assert "FLOW_SDE_CHUNK_GRPO_EVAL25_EVAL100_CHAIN_OK" in text
    subprocess.run(["bash", "-n", str(path)], check=True, cwd=str(_REPO_ROOT))


def test_flow_sde_chunk_moonshot_scripts_encode_final_step_contract() -> None:
    scripts = {
        "submit_flow_sde_chunk_grpo_moonshot30_dense_chain_a30.slurm": {
            "walltime": "#SBATCH --time=05:15:00",
            "group_size": "--group-size 8",
            "reward_mode": "--reward-mode dense_return",
            "noise_level": "--flow-sde-noise-level 0.5",
            "euler": "--euler-step-noise-std 0.0",
            "lr": "--lr 1e-5",
        },
        "submit_flow_sde_chunk_grpo_moonshot30_sparse_chain_a30.slurm": {
            "walltime": "#SBATCH --time=06:00:00",
            "group_size": "--group-size 16",
            "reward_mode": "--reward-mode sparse_success_delta",
            "noise_level": "--flow-sde-noise-level 1.0",
            "euler": "--euler-step-noise-std 0.0",
            "lr": "--lr 7.5e-6",
        },
    }
    for name, spec in scripts.items():
        path = _REPO_ROOT / "scripts" / "grpo" / name
        text = path.read_text(encoding="utf-8")
        assert "#SBATCH --gres=gpu:1" in text
        assert "#SBATCH --cpus-per-task=16" in text
        assert "#SBATCH --mem=64G" in text
        assert spec["walltime"] in text
        assert spec["group_size"] in text
        assert spec["reward_mode"] in text
        assert spec["noise_level"] in text
        assert spec["euler"] in text
        assert spec["lr"] in text
        if "dense" in name:
            assert "--allow-euler-noise" not in text
        assert "--num-updates 30" in text
        assert "--save-every 5" in text
        assert "--rollout-execution vector_async" in text
        assert "--async-start-method forkserver" in text
        assert "--flow-sde-trace-step 9" in text
        assert "--num-episodes 25" in text
        assert "--num-episodes 100" in text
        assert '--only-updates "10,20,30"' in text
        assert '"${GRPO_PYTHON_BIN}" -u "${PROJECT_ROOT}/scripts/grpo/train_phase11_env_on_policy_grpo.py"' in text
        assert '--checkpoint "${BASE_CKPT}"' in text
        assert "--policy-path" not in text
        assert "--train-seed-base 2000" in text
        train_text = text.split('CKPT_DIR="${TRAIN_OUT}/checkpoints"', maxsplit=1)[0]
        for unsupported_flag in (
            "--task-description",
            "--num-envs",
            "--horizon",
            "--seed",
            "--device",
            "--dtype",
            "--save-grpo-checkpoint",
            "--save-rlinf-eval-checkpoint",
        ):
            assert unsupported_flag not in train_text
        assert "FLOW_SDE_MOONSHOT_" in text
        subprocess.run(["bash", "-n", str(path)], check=True, cwd=str(_REPO_ROOT))


def test_submit_flow_sde_chunk_grpo_eval25_leaves_resolution_to_rlinf_eval() -> None:
    path = _REPO_ROOT / "scripts" / "grpo" / "submit_flow_sde_chunk_grpo_eval25_a30.slurm"
    text = path.read_text(encoding="utf-8")
    assert "#SBATCH --export=NIL" in text
    assert "scripts/slurm/common_env.sh" in text
    assert "--num-episodes 25" in text
    assert "--chunk-len 5" in text
    assert '--checkpoint-dir "${CKPT_DIR}"' in text
    assert "eval_smolvla_metaworld_ckpt_sweep.py" in text
    assert "FLOW_SDE_CHUNK_GRPO_EVAL25_OK" in text
    subprocess.run(["bash", "-n", str(path)], check=True, cwd=str(_REPO_ROOT))


def test_submit_phase111_single_task_grpo_is_official_backend() -> None:
    text = (_REPO_ROOT / "scripts" / "grpo" / "submit_phase111_single_task_grpo.slurm").read_text(
        encoding="utf-8"
    )
    assert "#SBATCH --export=NIL" in text
    assert "#SBATCH --time=48:00:00" in text
    assert "scripts/slurm/common_env.sh" in text
    assert "--env-backend official_lerobot" in text
    assert 'TASK="${GRPO_PHASE111_TASK:-push-v3}"' in text
    assert 'NUM_UPDATES="${GRPO_PHASE111_NUM_UPDATES:-100}"' in text
    assert 'GROUP_SIZE="${GRPO_PHASE111_GROUP_SIZE:-4}"' in text
    assert 'SAVE_EVERY="${GRPO_PHASE111_SAVE_EVERY:-5}"' in text
    assert 'MAX_STEPS="${GRPO_PHASE111_MAX_STEPS:-120}"' in text
    assert "120-step cap to keep update cost comparable" in text
    assert '_PHASE111_PROJECT_FALLBACK="/vol/bitbucket/aa6622/project"' in text
    assert "PHASE111_SINGLE_TASK_GRPO_OK" in text
    assert "GRPO_PHASE111_ROLLOUT_EXECUTION" in text
    assert '$10 rollout_execution' in text
    assert '$11 async_start_method' in text
    assert '$12 action_transform' in text
    assert '$13 run_label' in text
    assert 'if [[ $# -ge 10' in text
    assert 'if [[ $# -ge 11' in text
    assert 'if [[ $# -ge 12' in text
    assert 'if [[ $# -ge 13' in text
    assert "GRPO_PHASE111_ACTION_TRANSFORM" in text
    assert "GRPO_PHASE111_RUN_LABEL" in text
    assert 'manifest["rollout_execution"]' in text
    assert 'manifest["action_transform"]' in text
    assert 'manifest["run_label"]' in text
    assert 'manifest["async_start_method"] == "forkserver"' in text
    assert 'len(rows) == end_update' in text
    assert 'f"update_{end_update:04d}.pt"' in text
    assert '"rollout_seconds" in row' in text
    assert '"optimize_seconds" in row' in text
    assert '"update_seconds" in row' in text
    assert '"num_env_steps" in row' in text
    assert '"action_clip_fraction" in row' in text
    assert '"action_clip_any_fraction" in row' in text


def test_phase111_eval_sweep_runs_eval_from_repo_root() -> None:
    text = (_REPO_ROOT / "scripts" / "grpo" / "eval_phase111_grpo_sweep.py").read_text(encoding="utf-8")
    assert "_REPO = Path(__file__).resolve().parents[2]" in text
    assert 'subprocess.run(cmd, check=True, cwd=str(_REPO))' in text
    assert "--sweep-name" in text
    assert "--execution-mode" in text
    assert "inprocess_vector" in text
    assert "vector_async" in text
    assert 'sweep_dir = run_dir / sweep_name' in text


def test_submit_phase111_eval_sweep_uses_common_env_and_sweep_name() -> None:
    path = _REPO_ROOT / "scripts" / "grpo" / "submit_phase111_eval_sweep.slurm"
    text = path.read_text(encoding="utf-8")
    assert "#SBATCH --export=NIL" in text
    assert "scripts/slurm/common_env.sh" in text
    assert 'RUN_DIR="${1:-}"' in text
    assert 'BASE_CKPT="${2:-}"' in text
    assert 'SWEEP_NAME="${8:-eval_sweep_current}"' in text
    assert 'EXECUTION_MODE="${PHASE111_EVAL_EXECUTION_MODE:-inprocess_vector}"' in text
    assert 'ROLLOUT_EXECUTION="${PHASE111_EVAL_ROLLOUT_EXECUTION:-vector_async}"' in text
    assert "--sweep-name" in text
    assert "--execution-mode" in text
    assert "--rollout-execution" in text
    assert "PHASE111_EVAL_SWEEP_OK" in text
    subprocess.run(["bash", "-n", str(path)], check=True, cwd=str(_REPO_ROOT))


def test_phase11_pop128_smoke_pbs_has_rollout_policy_batch_size() -> None:
    path = _REPO_ROOT / "scripts" / "grpo" / "phase11_pop128_rolloutpbs32_smoke_u1.pbs"
    text = path.read_text(encoding="utf-8")
    assert "--rollout-policy-batch-size 16" in text
    assert "--group-size 128" in text
    assert "--logprob-batch-size 16" in text
    assert "PHASE11_POP128_ROLLOUTPBS32_SMOKE_OK" in text
    subprocess.run(["bash", "-n", str(path)], check=True, cwd=str(_REPO_ROOT))


def test_phase11_pop128_train_scripts_have_microbatch_cap() -> None:
    for name in (
        "phase11_P128A_lr2e6_clip005_train_0001_0050.pbs",
        "phase11_P128B_lr5e6_clip01_train_0001_0050.pbs",
        "phase11_P128C_lr5e6_clip01_lownoise_train_0001_0050.pbs",
    ):
        path = _REPO_ROOT / "scripts" / "grpo" / name
        text = path.read_text(encoding="utf-8")
        assert "--group-size 128" in text
        assert "--rollout-policy-batch-size 16" in text
        assert "--logprob-batch-size 16" in text
        subprocess.run(["bash", "-n", str(path)], check=True, cwd=str(_REPO_ROOT))


def test_phase11_batched_logprob_smoke_pbs_defaults_to_batched_bs16() -> None:
    path = _REPO_ROOT / "scripts" / "grpo" / "phase11_batched_logprob_smoke_u2.pbs"
    text = path.read_text(encoding="utf-8")
    assert "#PBS -l walltime=00:30:00" in text
    assert "#PBS -l select=1:ncpus=48:mem=64gb:ngpus=1:gpu_type=RTX6000" in text
    assert 'RUN_DIR="artifacts/phase11_pushv3_batched_logprob_smoke_u2"' in text
    assert 'SMOLVLA_METAWORLD_RESET_MODE="${SMOLVLA_METAWORLD_RESET_MODE:-random_seeded}"' in text
    assert "--num-updates 2" in text
    assert "--group-size 32" in text
    assert "--logprob-recompute-mode batched" in text
    assert "--logprob-batch-size 16" in text
    assert "PHASE11_BATCHED_LOGPROB_SMOKE_OK" in text
    subprocess.run(["bash", "-n", str(path)], check=True, cwd=str(_REPO_ROOT))


def test_phase11_seedbatch_smoke_pbs_requests_resources() -> None:
    path = _REPO_ROOT / "scripts" / "grpo" / "phase11_seedbatch_smoke_u2.pbs"
    text = path.read_text(encoding="utf-8")
    assert "#PBS -l walltime=00:30:00" in text
    assert "#PBS -l select=1:ncpus=48:mem=64gb:ngpus=1:gpu_type=RTX6000" in text
    assert 'RUN_DIR="artifacts/phase11_pushv3_seedbatch_smoke_u2"' in text
    assert "--batch-size 2" in text
    assert "--group-size 8" in text
    assert "--num-updates 2" in text
    assert "--logprob-batch-size 16" in text
    assert "--rollout-policy-batch-size 16" in text
    assert "phase11_cpu_mem_telemetry.sh" in text
    assert "run_phase11_with_cpu_mem_telemetry" in text
    assert "PHASE11_SEEDBATCH_SMOKE_OK" in text
    subprocess.run(["bash", "-n", str(path)], check=True, cwd=str(_REPO_ROOT))


def test_phase11_seedbatch_prod_pbs_is_not_pop128_env_peak() -> None:
    path = _REPO_ROOT / "scripts" / "grpo" / "phase11_seedbatch_b4_g32_train_0000_0050.pbs"
    text = path.read_text(encoding="utf-8")
    assert "#PBS -l select=1:ncpus=48:mem=128gb:ngpus=1:gpu_type=RTX6000" in text
    assert "#PBS -l walltime=48:00:00" in text
    assert "--batch-size 4" in text
    assert "--group-size 32" in text
    assert "--num-updates 50" in text
    assert "--save-every 2" in text
    assert "--rollout-policy-batch-size 32" in text
    assert "--logprob-batch-size 16" in text
    assert "PHASE11_SEEDBATCH_B4_G32_TRAIN_DONE" in text
    subprocess.run(["bash", "-n", str(path)], check=True, cwd=str(_REPO_ROOT))


def test_phase11_train_pbs_scripts_track_cpu_memory() -> None:
    helper = (_REPO_ROOT / "scripts" / "grpo" / "phase11_cpu_mem_telemetry.sh").read_text(
        encoding="utf-8"
    )
    assert 'scripts/grpo/sample_process_tree_memory.py' in helper
    assert 'scripts/grpo/summarize_process_tree_memory.py' in helper
    assert 'process_tree_memory.csv' in helper
    assert 'process_tree_memory_summary.json' in helper
    subprocess.run(
        ["bash", "-n", str(_REPO_ROOT / "scripts" / "grpo" / "phase11_cpu_mem_telemetry.sh")],
        check=True,
        cwd=str(_REPO_ROOT),
    )

    names = (
        "phase11_pop128_rolloutpbs32_smoke_u1.pbs",
        "phase11_P128A_lr2e6_clip005_train_0001_0050.pbs",
        "phase11_P128B_lr5e6_clip01_train_0001_0050.pbs",
        "phase11_P128C_lr5e6_clip01_lownoise_train_0001_0050.pbs",
        "phase11_R1_g32_lr2e6_clip005_train_0001_0050.pbs",
        "phase11_R2_g32_lr5e6_clip01_lownoise_train_0001_0050.pbs",
        "phase11_R3_g64_lr5e6_clip01_train_0001_0050.pbs",
        "phase11_A_g32_lr5e6_clip01_train_0000_0030.pbs",
        "phase11_A_g32_lr5e6_clip01_resume_train_0014_0030.pbs",
        "phase11_g16_lr5e6_clip02_train_0000_0010.pbs",
        "phase11_g16_lr5e6_clip02_train_0010_0020_resume.pbs",
        "phase11_batched_logprob_smoke_u2.pbs",
        "phase11_seedbatch_smoke_u2.pbs",
        "phase11_seedbatch_b4_g32_train_0000_0050.pbs",
    )
    for name in names:
        path = _REPO_ROOT / "scripts" / "grpo" / name
        text = path.read_text(encoding="utf-8")
        assert 'CPU_MEM_TELEMETRY_INTERVAL="${CPU_MEM_TELEMETRY_INTERVAL:-5}"' in text
        assert 'CPU_MEM_TELEMETRY_DIR="${RUN_DIR}/cpu_mem_telemetry/train"' in text
        assert "source scripts/grpo/phase11_cpu_mem_telemetry.sh" in text
        assert "run_phase11_with_cpu_mem_telemetry" in text
        subprocess.run(["bash", "-n", str(path)], check=True, cwd=str(_REPO_ROOT))
