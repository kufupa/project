from __future__ import annotations

from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (PROJECT / path).read_text(encoding="utf-8")


def test_phase12_train_chunk_pbs_contract() -> None:
    text = _read("scripts/grpo/phase12_train_chunk_10u.pbs")
    assert "#PBS -l select=1:ncpus=48:mem=96gb:ngpus=1:gpu_type=RTX6000" in text
    assert "#PBS -l walltime=04:00:00" in text
    assert "#PBS -q v1_gpu72" in text
    assert "--phase12-train-mode \"${TRAIN_MODE}\"" in text
    assert "--start-update \"${START_UPDATE}\"" in text
    assert "--batch-size \"${BATCH_SIZE}\"" in text
    assert "--group-size \"${GROUP_SIZE}\"" in text
    assert "--chunk-len \"${CHUNK_LEN}\"" in text
    assert "--max-steps \"${MAX_STEPS}\"" in text
    assert "--disable-videos" in text
    assert "--no-save-wm-decodes" in text
    assert "--wm-score-mode \"${WM_SCORE_MODE}\"" in text
    assert "PHASE12_TRAIN_CHUNK_DONE" in text
    assert "proc_mem_after_optimize_tree_rss_kb" in text


def test_phase12_eval_last5_pbs_contract() -> None:
    text = _read("scripts/grpo/phase12_eval_last5_25ep.pbs")
    assert "#PBS -l select=1:ncpus=32:mem=32gb:ngpus=1:gpu_type=RTX6000" in text
    assert "#PBS -l walltime=00:30:00" in text
    assert "--execution-mode inprocess_vector" in text
    assert "--n-envs \"${N_ENVS}\"" in text
    assert "--rollout-execution vector_async" in text
    assert "--chunk-len \"${CHUNK_LEN}\"" in text
    assert "--max-steps \"${MAX_STEPS}\"" in text
    assert "--episodes \"${EPISODES}\"" in text
    assert "PHASE12_EVAL_LAST5_DONE" in text


def test_phase12_two_exp_chain_contract() -> None:
    text = _read("scripts/grpo/submit_phase12_two_exp_70u_chain.sh")
    assert "pbs_gpu_snapshot.py" in text
    assert "official_g8_lr1e5" in text
    assert "bounded_g8_lr1e5" in text
    assert "maxperf_b4_g16_lr5e6_clip01_lownoise" in text
    assert "PHASE12_ENABLE_MAXPERF" in text
    assert "initial_dep" in text
    assert "depend=afterok:${prev_train}" in text
    assert "PHASE12_NUM_UPDATES=10" in text
    assert "PHASE12_CHAIN_BOOTSTRAP_SUBMITTED" in text


def test_phase12_g8_u20_strict_pbs_contract() -> None:
    text = _read("scripts/grpo/phase12_g8_u20_wm_train_eval100_stride5.pbs")
    assert "#PBS -q v1_gpu72" in text
    assert "PHASE12_GROUP_SIZE=\"${PHASE12_GROUP_SIZE:-8}\"" in text
    assert "PHASE12_NUM_UPDATES=\"${PHASE12_NUM_UPDATES:-20}\"" in text
    assert "PHASE12_CHUNK_LEN=\"${PHASE12_CHUNK_LEN:-5}\"" in text
    assert "PHASE12_ACTION_PROFILE=\"${PHASE12_ACTION_PROFILE:-bounded_executed}\"" in text
    assert "PHASE12_EVAL_EPISODES=\"${PHASE12_EVAL_EPISODES:-100}\"" in text
    assert "PHASE12_SWEEP_NAME=\"${PHASE12_SWEEP_NAME:-eval100_u0005_0020_stride5_nenv25_async}\"" in text


def test_phase12_supervisor_loop_pbs_contract() -> None:
    text = _read("scripts/grpo/phase12_wm_grpo_supervisor_loop.pbs")
    assert "#PBS -l select=1:ncpus=1:mem=4gb" in text
    assert "--auto-resume" in text
    assert "--loop" in text
    assert "--poll-seconds \"${POLL_SECONDS}\"" in text
