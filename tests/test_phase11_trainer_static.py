from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from scripts.grpo.train_phase11_env_on_policy_grpo import compute_live_chunk_logprob_parity


TRAINER = Path(__file__).resolve().parents[1] / "scripts" / "grpo" / "train_phase11_env_on_policy_grpo.py"


def test_trainer_exposes_rollout_unit_and_rollout_chunk_len() -> None:
    text = TRAINER.read_text(encoding="utf-8")
    assert "--rollout-unit" in text
    assert 'choices=("step", "chunk")' in text
    assert "--rollout-chunk-len" in text


def test_flow_sde_step_mode_is_guarded() -> None:
    text = TRAINER.read_text(encoding="utf-8")
    assert "flow_sde requires --rollout-unit chunk" in text


def test_chunk_mode_is_flow_sde_only_for_first_implementation() -> None:
    text = TRAINER.read_text(encoding="utf-8")
    assert "chunk rollout currently requires --logprob-mode flow_sde" in text


def test_chunk_mode_loads_bundle_with_rollout_chunk_len() -> None:
    text = TRAINER.read_text(encoding="utf-8")
    assert 'n_action_steps=(int(args.rollout_chunk_len) if args.rollout_unit == "chunk" else 1)' in text


def test_chunk_mode_uses_chunk_rollout_collector() -> None:
    text = TRAINER.read_text(encoding="utf-8")
    assert "collect_chunk_rollout_group" in text


def test_chunk_parity_replays_one_chunk_at_a_time() -> None:
    class _Wrapper:
        def get_flow_sde_log_probs_for_chunk_from_proc_list(self, procs, traces, *, chunk_len: int):
            del traces, chunk_len
            if len(procs) != 1:
                return torch.tensor([[100.0, 100.0]]), None, None
            return torch.tensor([[1.0, 2.0]]), None, None

    chunk_a = SimpleNamespace(
        proc_snapshot={"x": torch.zeros(1, 1)},
        flow_sde_trace={"A_next": torch.zeros(1, 2, 4)},
        valid_action_mask=torch.tensor([True, True]),
        log_probs=torch.tensor([1.0, 2.0]),
    )
    chunk_b = SimpleNamespace(
        proc_snapshot={"x": torch.zeros(1, 1)},
        flow_sde_trace={"A_next": torch.zeros(1, 2, 4)},
        valid_action_mask=torch.tensor([True, False]),
        log_probs=torch.tensor([1.0, 2.0]),
    )
    stats, payload = compute_live_chunk_logprob_parity(
        train_wrapper=_Wrapper(),
        rollouts=[SimpleNamespace(chunks=[chunk_a, chunk_b])],
        chunk_len=2,
        tolerance=0.02,
    )

    assert stats.within_tolerance
    assert payload["max_abs_per_action_logprob"] == 0.0


def test_trainer_writes_rlinf_eval_checkpoints_next_to_resume_checkpoints() -> None:
    text = TRAINER.read_text(encoding="utf-8")
    assert "save_rlinf_eval_checkpoint" in text
    assert "validate_rlinf_eval_checkpoint" in text
    assert 'eval_ckpt_dir = out / "checkpoints_eval"' in text
    assert "eval_ckpt_dir.mkdir(parents=True, exist_ok=True)" in text
    assert "eval_ckpt_dir / name" in text
    assert "expected_update=update_index + 1" in text
    assert "source_checkpoint=full_path" in text


def test_trainer_exposes_reward_mode_for_chunk_ablation() -> None:
    text = TRAINER.read_text(encoding="utf-8")
    assert "--reward-mode" in text
    assert "dense_return" in text
    assert "sparse_success_delta" in text
    assert "success_first_dense" in text
    assert "episode_return_for_mode" in text
    assert '"reward_mode": args.reward_mode' in text


def test_trainer_keeps_full_resume_checkpoints_unchanged() -> None:
    text = TRAINER.read_text(encoding="utf-8")
    assert "save_grpo_checkpoint(" in text
    assert "policy_state=bundle.policy.state_dict()" in text
    assert "optimizer_state=optimizer.state_dict()" in text
