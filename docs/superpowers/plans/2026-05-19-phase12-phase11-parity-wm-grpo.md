# Phase12 Phase11-Parity WM-GRPO Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Phase12 into a Phase11-parity WM-GRPO trainer for `push-v3`: same operational skeleton as Phase11 seed-batch GRPO, but reward/candidate ranking comes from JEPA-WM and runs can be scheduled immediately after smokes.

**Architecture:** Keep Phase11 as the real MetaWorld control. Continue Phase12 as the WM owner, add explicit train modes, then port Phase11's recent seed-batch, loss-normalizer, batching, telemetry, PBS, and eval-sweep contracts. The first WM-only train mode removes selected MetaWorld stepping from the update while still using official reset/proc/oracle-goal plumbing; a serialized root-bank mode comes after that contract is stable.

**Tech Stack:** Python 3.12, PyTorch, patched LeRobot SmolVLA GRPO APIs, MetaWorld official LeRobot backend, JEPA-WM, PBS, pytest.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `scripts/grpo/train_phase12_wm_chunk_grpo.py` | Phase12 trainer CLI, manifest/progress contract, selected-env and WM-only train branches, seed-batch orchestration, loss/backward path |
| `src/smolvla_grpo/phase12_rollout.py` | Small rollout dataclasses and action-profile application for candidate groups |
| `src/smolvla_grpo/phase12_wm_only_rollout.py` | New WM-only candidate collector: no selected `env.step`, one root per seed/update, WM scores all candidates |
| `src/smolvla_grpo/phase12_wm_reward.py` | JEPA-WM root/goal encoding, candidate scoring, cache and batch/fallback scoring helpers |
| `src/smolvla_grpo/grpo_math.py` | Existing `compute_seed_batch_advantages`; reused for Phase12 seed-major groups |
| `src/smolvla_grpo/process_memory.py` | Existing process-tree memory helpers; reused by Phase12 telemetry |
| `scripts/grpo/eval_phase12_checkpoint_sweep.py` | Phase12 checkpoint eval sweep; add Phase11-style top-k 100ep confirmation |
| `scripts/grpo/phase12_seedbatch_smoke_u2.pbs` | New PBS smoke for Phase12 seed-batch WM-GRPO |
| `scripts/grpo/phase12_seedbatch_b4_g16_train_0000_0050.pbs` | New conservative production PBS train script |
| `scripts/grpo/phase12_eval_sweep_topk.pbs` | New PBS 25ep sweep plus top-k 100ep confirmation wrapper |
| `scripts/grpo/submit_phase12_pbs_chain.sh` | Login-node helper that submits smokes, five train jobs, and afterok evals |
| `tests/test_phase12_wm_only_rollout.py` | New tests for WM-only no-selected-step contract |
| `tests/test_phase12_training_loop.py` | Trainer aggregation, seed-batch, logprob, checkpoint/progress tests |
| `tests/test_phase12_wm_reward.py` | WM encode cache and batch/fallback score tests |
| `tests/test_phase12_pbs_static.py` | PBS script contract tests |
| `tests/test_phase12_eval_sweep.py` | Eval top-k ranking and summary tests |

Implementation order follows dependency flow: trainer mode labels -> loss denominator -> WM-only collector -> seed-batch -> batched logprob -> WM cache/batching -> telemetry -> PBS/eval scheduling.

---

## Task 1: Phase12 Train Mode Contract

**Files:**
- Modify: `scripts/grpo/train_phase12_wm_chunk_grpo.py`
- Test: `tests/test_phase12_training_loop.py`

- [ ] **Step 1: Write failing tests for train-mode manifest and validation**

Append this test to `tests/test_phase12_training_loop.py`:

```python
def test_phase12_manifest_records_train_mode(tmp_path) -> None:
    args = trainer.parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--jepa-repo",
            "/tmp/jepa",
            "--jepa-ckpt",
            "wm.pt",
            "--phase12-train-mode",
            "wm_only",
        ]
    )

    manifest = trainer.build_manifest(args)

    assert manifest["phase12_train_mode"] == "wm_only"
    assert manifest["real_env_selected_rollout"] is False
    assert manifest["rollout_execution"] == "wm_only_single_root"


def test_phase12_rejects_group_size_one_for_grpo() -> None:
    args = trainer.parse_args(
        [
            "--jepa-repo",
            "/tmp/jepa",
            "--jepa-ckpt",
            "wm.pt",
            "--group-size",
            "1",
        ]
    )

    assert trainer._validate_real_mode(args) == "--group-size must be >= 2 for GRPO advantage normalization."
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
PYTHONPATH="${PWD}/src:${PYTHONPATH:-}" /rds/general/user/aa6622/home/.envs/lerobot_mw_py312/bin/python -m pytest tests/test_phase12_training_loop.py -q -k "train_mode or group_size_one"
```

Expected: FAIL because `--phase12-train-mode` is not defined and group-size validation allows `1`.

- [ ] **Step 3: Add CLI and manifest fields**

In `scripts/grpo/train_phase12_wm_chunk_grpo.py`, add this argument after `--mode`:

```python
    p.add_argument(
        "--phase12-train-mode",
        choices=("selected_env", "wm_only"),
        default="selected_env",
        help="selected_env keeps current Phase12 winner env.step path; wm_only scores candidates with JEPA-WM and does not step selected chunks.",
    )
```

Change `_validate_real_mode` group-size check to:

```python
    if int(args.group_size) < 2:
        return "--group-size must be >= 2 for GRPO advantage normalization."
```

Change `build_manifest` by replacing the fixed rollout fields with:

```python
        "phase12_train_mode": str(args.phase12_train_mode),
        "env_vector_mode": "serial",
        "rollout_execution": (
            "serial_selected_rollout"
            if str(args.phase12_train_mode) == "selected_env"
            else "wm_only_single_root"
        ),
        "real_env_selected_rollout": str(args.phase12_train_mode) == "selected_env",
        "true_parallel_metaworld": False,
        "true_parallel_metaworld_note": (
            "Phase12 selected_env remains serial because oracle/reset parity/WM scoring are per-episode coupled."
            if str(args.phase12_train_mode) == "selected_env"
            else "Phase12 wm_only uses official reset/proc/goal plumbing but no selected env.step during the GRPO update."
        ),
```

- [ ] **Step 4: Run tests and verify pass**

Run:

```bash
PYTHONPATH="${PWD}/src:${PYTHONPATH:-}" /rds/general/user/aa6622/home/.envs/lerobot_mw_py312/bin/python -m pytest tests/test_phase12_training_loop.py -q -k "train_mode or group_size_one"
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/grpo/train_phase12_wm_chunk_grpo.py tests/test_phase12_training_loop.py
git commit -m "feat(phase12): add explicit train mode contract"
```

---

## Task 2: Explicit Phase12 GRPO Group Normalizer

**Files:**
- Modify: `scripts/grpo/train_phase12_wm_chunk_grpo.py`
- Test: `tests/test_phase12_training_loop.py`

- [ ] **Step 1: Write failing test for explicit `grpo_group_size`**

Append this test to `tests/test_phase12_training_loop.py`:

```python
def test_phase12_microbatch_loss_uses_explicit_group_size() -> None:
    class Wrapper:
        def __init__(self) -> None:
            self.scale = torch.nn.Parameter(torch.tensor(0.0))

        def get_action_probs_for_chunk_from_proc(self, proc, chunk):
            del proc
            return self.scale + torch.as_tensor(chunk).float().reshape(-1) * 0.0

    chunks = [torch.zeros((1, 4), dtype=torch.float32) for _ in range(4)]
    old_lp = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    advantages = torch.tensor([2.0, 1.0, 3.0, 0.0], dtype=torch.float32)

    good = Wrapper()
    trainer._backward_chunk_grpo_loss_microbatched(
        train_wrapper=good,
        proc_snapshots=[{} for _ in chunks],
        unsquashed_chunks=chunks,
        old_lp=old_lp,
        advantages=advantages,
        clip_eps=0.2,
        grpo_group_size=2,
    )

    wrong = Wrapper()
    trainer._backward_chunk_grpo_loss_microbatched(
        train_wrapper=wrong,
        proc_snapshots=[{} for _ in chunks],
        unsquashed_chunks=chunks,
        old_lp=old_lp,
        advantages=advantages,
        clip_eps=0.2,
        grpo_group_size=None,
    )

    assert good.scale.grad is not None
    assert wrong.scale.grad is not None
    assert not torch.allclose(good.scale.grad, wrong.scale.grad)
```

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
PYTHONPATH="${PWD}/src:${PYTHONPATH:-}" /rds/general/user/aa6622/home/.envs/lerobot_mw_py312/bin/python -m pytest tests/test_phase12_training_loop.py::test_phase12_microbatch_loss_uses_explicit_group_size -q
```

Expected: FAIL because `_backward_chunk_grpo_loss_microbatched` does not accept `grpo_group_size`.

- [ ] **Step 3: Update microbatch loss signature and denominator**

Replace `_backward_chunk_grpo_loss_microbatched` signature with:

```python
def _backward_chunk_grpo_loss_microbatched(
    *,
    train_wrapper: Any,
    proc_snapshots: list[Any],
    unsquashed_chunks: list[Any],
    old_lp: Any,
    advantages: Any,
    clip_eps: float,
    grpo_group_size: int | None = None,
) -> tuple[float, dict[str, float], Any]:
```

Add this after `row_count` validation:

```python
    G = int(grpo_group_size if grpo_group_size is not None else row_count)
    if G < 1:
        raise ValueError("grpo_group_size must be >= 1")
    if row_count % G != 0:
        raise ValueError(f"row count {row_count} must be a multiple of grpo_group_size={G}")
```

Change loss normalizer from `row_count` to `G`:

```python
            normalizer=G,
```

At call site in `run_wm_grpo_train`, pass:

```python
                grpo_group_size=int(args.group_size),
```

- [ ] **Step 4: Add stack-mode explicit normalizer**

Add helper near `_chunk_grpo_row_loss`:

```python
def _chunk_grpo_loss_with_group_normalizer(
    *,
    old_lp: Any,
    new_lp: Any,
    advantages: Any,
    clip_eps: float,
    grpo_group_size: int,
) -> tuple[Any, dict[str, float]]:
    import torch

    old = torch.as_tensor(old_lp).float()
    new = torch.as_tensor(new_lp).float()
    adv = torch.as_tensor(advantages).float()
    G = int(grpo_group_size)
    if G < 1:
        raise ValueError("grpo_group_size must be >= 1")
    if int(old.numel()) % G != 0:
        raise ValueError(f"row count {old.numel()} must be a multiple of grpo_group_size={G}")
    ratio = torch.exp(new - old)
    clipped_ratio = torch.clamp(ratio, 1.0 - float(clip_eps), 1.0 + float(clip_eps))
    row_loss = -torch.minimum(ratio * adv, clipped_ratio * adv) / float(G)
    loss = row_loss.sum()
    return loss, _ratio_stats_from_tensors(old.detach(), new.detach(), clip_eps=float(clip_eps))
```

Replace stack-mode call to `chunk_grpo_loss(...)` with:

```python
            loss, ratio_stats = _chunk_grpo_loss_with_group_normalizer(
                old_lp=old_lp,
                new_lp=new_lp,
                advantages=advantages,
                clip_eps=float(args.clip_eps),
                grpo_group_size=int(args.group_size),
            )
```

- [ ] **Step 5: Run tests**

Run:

```bash
PYTHONPATH="${PWD}/src:${PYTHONPATH:-}" /rds/general/user/aa6622/home/.envs/lerobot_mw_py312/bin/python -m pytest tests/test_phase12_training_loop.py tests/test_phase12_rollout.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/grpo/train_phase12_wm_chunk_grpo.py tests/test_phase12_training_loop.py
git commit -m "fix(phase12): use explicit GRPO group size"
```

---

## Task 3: WM-Only Single-Root Collector

**Files:**
- Create: `src/smolvla_grpo/phase12_wm_only_rollout.py`
- Modify: `scripts/grpo/train_phase12_wm_chunk_grpo.py`
- Test: `tests/test_phase12_wm_only_rollout.py`

- [ ] **Step 1: Create failing tests for no selected env step**

Create `tests/test_phase12_wm_only_rollout.py`:

```python
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from smolvla_grpo.phase12_wm_only_rollout import collect_phase12_wm_only_episode


def _score(candidate_index: int, progress: float) -> dict:
    return {
        "candidate_index": int(candidate_index),
        "wm_latent_progress": float(progress),
        "latent_return": float(progress),
        "final_combined_distance": float(10.0 - progress),
    }


def test_wm_only_collector_scores_candidates_without_env_step() -> None:
    step_calls = 0

    class RootSource:
        def reset(self, seed: int):
            assert seed == 123
            return {
                "id": "root-123",
                "image": np.zeros((8, 8, 3), dtype=np.uint8),
                "proprio": np.zeros(4, dtype=np.float32),
                "proc": {"x": torch.zeros(1, 1)},
            }

        def step(self, _action):
            nonlocal step_calls
            step_calls += 1
            raise AssertionError("wm_only collector must not step env")

    def sampler(root, *, num_candidates: int, segment_index: int):
        assert root["id"] == "root-123"
        assert segment_index == 0
        for i in range(num_candidates):
            yield {
                "candidate_index": i,
                "proc_root_snapshot": root["proc"],
                "unsquashed_chunk": torch.full((2, 4), float(i), dtype=torch.float32),
                "old_logprob_steps": np.array([-0.1, -0.2], dtype=np.float32),
                "exec_actions_raw_postprocessed": np.full((2, 4), float(i), dtype=np.float32),
            }

    def score_fn(root, candidate, goal, *, segment_index: int):
        assert root["id"] == "root-123"
        assert goal.frame_index_1based == 25
        return _score(candidate.candidate_index, progress=float(candidate.candidate_index))

    result = collect_phase12_wm_only_episode(
        root_source=RootSource(),
        reset_seed=123,
        policy_sampler=sampler,
        score_fn=score_fn,
        goals=[SimpleNamespace(frame_index_1based=25)],
        group_size=3,
        reward_key="wm_latent_progress",
    )

    assert step_calls == 0
    assert result.success_any is False
    assert result.metadata["candidate_rewards"] == [0.0, 1.0, 2.0]
    assert result.metadata["selected_candidate_indices"] == [2]
    assert len(result.metadata["old_logprob_sums"]) == 3
    assert len(result.metadata["unsquashed_chunks"]) == 3
```

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
PYTHONPATH="${PWD}/src:${PYTHONPATH:-}" /rds/general/user/aa6622/home/.envs/lerobot_mw_py312/bin/python -m pytest tests/test_phase12_wm_only_rollout.py -q
```

Expected: FAIL because `phase12_wm_only_rollout.py` does not exist.

- [ ] **Step 3: Add WM-only collector**

Create `src/smolvla_grpo/phase12_wm_only_rollout.py`:

```python
"""WM-only Phase12 rollout helpers: score chunks without selected env stepping."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import replace
from typing import Any

import numpy as np

from smolvla_grpo.phase12_rollout import (
    Phase12EpisodeResult,
    Phase12SegmentRecord,
    _candidate_from_sample,
    _field,
    select_best_candidate,
)


def _score_value(score: Any, key: str) -> float:
    if isinstance(score, dict):
        return float(score[key])
    return float(getattr(score, key))


def collect_phase12_wm_only_episode(
    *,
    root_source: Any,
    reset_seed: int,
    policy_sampler: Callable[..., Iterable[Any]],
    score_fn: Callable[..., Any],
    goals: Sequence[Any],
    group_size: int,
    reward_key: str,
    action_profile: str = "official_jepa_mirror",
    action_low: float | np.ndarray | None = None,
    action_high: float | np.ndarray | None = None,
    preprocessor: Any | None = None,
    env_action_dim: int | None = None,
    wm_action_dim: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> Phase12EpisodeResult:
    root = root_source.reset(int(reset_seed))
    segments: list[Phase12SegmentRecord] = []
    selected_candidate_indices: list[int] = []
    segment_candidate_rewards: list[list[float]] = []
    old_logprob_sums: list[float] = []
    proc_root_snapshots: list[Any] = []
    unsquashed_chunks: list[Any] = []

    for segment_index, goal in enumerate(goals):
        samples = list(
            policy_sampler(
                root,
                num_candidates=int(group_size),
                segment_index=int(segment_index),
            )
        )
        candidates = [
            _candidate_from_sample(
                sample,
                default_index=i,
                root_snapshot=root.get("proc", root),
                action_profile=action_profile,
                action_low=action_low if action_low is not None else np.full((int(env_action_dim or 4),), -1.0, dtype=np.float32),
                action_high=action_high if action_high is not None else np.full((int(env_action_dim or 4),), 1.0, dtype=np.float32),
                preprocessor=preprocessor,
                env_action_dim=env_action_dim,
                wm_action_dim=wm_action_dim,
            )
            for i, sample in enumerate(samples)
        ]
        if len(candidates) != int(group_size):
            raise ValueError(f"policy_sampler returned {len(candidates)} candidates, expected {group_size}")
        scores = [
            score_fn(root, candidate, goal, segment_index=int(segment_index))
            for candidate in candidates
        ]
        selected = select_best_candidate(scores, reward_key=reward_key)
        selected_candidate_indices.append(int(selected))
        segment_candidate_rewards.append([_score_value(score, reward_key) for score in scores])
        old_logprob_sums.extend(float(candidate.old_logprob_sum) for candidate in candidates)
        proc_root_snapshots.extend(candidate.proc_root_snapshot for candidate in candidates)
        unsquashed_chunks.extend(candidate.unsquashed_chunk for candidate in candidates)
        segments.append(
            Phase12SegmentRecord(
                update_index=0,
                episode_index=0,
                segment_index=int(segment_index),
                goal_frame_index_1based=int(getattr(goal, "frame_index_1based", segment_index + 1)),
                selected_candidate_index=int(selected),
                scores=list(scores),
                candidates=list(candidates),
                success_any=False,
                success_last=False,
                env_reward_sum=0.0,
                decode_metadata={"wm_only": True},
            )
        )

    flat_rewards = [reward for row in segment_candidate_rewards for reward in row]
    meta = dict(metadata or {})
    meta.update(
        {
            "phase12_train_mode": "wm_only",
            "candidate_rewards": flat_rewards,
            "segment_candidate_rewards": segment_candidate_rewards,
            "selected_candidate_indices": selected_candidate_indices,
            "old_logprob_sums": old_logprob_sums,
            "proc_root_snapshots": proc_root_snapshots,
            "unsquashed_chunks": unsquashed_chunks,
            "success_any": False,
            "success_last": False,
        }
    )
    return Phase12EpisodeResult(
        segments=segments,
        total_env_reward=0.0,
        success_any=False,
        success_last=False,
        metadata=meta,
    )
```

- [ ] **Step 4: Run collector tests**

Run:

```bash
PYTHONPATH="${PWD}/src:${PYTHONPATH:-}" /rds/general/user/aa6622/home/.envs/lerobot_mw_py312/bin/python -m pytest tests/test_phase12_wm_only_rollout.py -q
```

Expected: PASS.

- [ ] **Step 5: Wire trainer branch**

In `run_wm_grpo_train`, replace direct call to `collect_phase12_training_episode` with:

```python
        if args.phase12_train_mode == "wm_only":
            episode = collect_phase12_wm_only_training_episode(
                args=args,
                bundle=bundle,
                wm_bundle=wm_bundle,
                old_wrapper=old_wrapper,
                action_dim=action_dim,
                update_index=update_index,
                reset_seed=reset_seed,
                output_dir=out,
            )
        else:
            episode = collect_phase12_training_episode(
                args=args,
                bundle=bundle,
                wm_bundle=wm_bundle,
                old_policy=old_policy,
                old_wrapper=old_wrapper,
                action_dim=action_dim,
                update_index=update_index,
                reset_seed=reset_seed,
                output_dir=out,
            )
```

Add import inside `run_wm_grpo_train`:

```python
    from smolvla_grpo.phase12_wm_only_rollout import collect_phase12_wm_only_episode
```

Add a small adapter function above `run_wm_grpo_train`:

```python
def collect_phase12_wm_only_training_episode(**kwargs: Any) -> Any:
    """Collect one Phase12 update from a reset root without selected env stepping."""
    args = kwargs["args"]
    # Reuse the selected-env implementation in Task 3 only for root/proc/goal construction.
    # The selected chunk is not stepped by collect_phase12_wm_only_episode.
    selected = collect_phase12_training_episode(**kwargs)
    meta = dict(getattr(selected, "metadata", {}) or {})
    meta["phase12_train_mode"] = "wm_only"
    meta["wm_only_bootstrap_source"] = "selected_env_goal_and_root_plumbing"
    return _with_episode_metadata(selected, meta)
```

This adapter still uses current root/goal plumbing. Task 4 replaces the selected-env stepping inside this branch with the new collector. This two-step change keeps tests and script imports small.

- [ ] **Step 6: Commit**

```bash
git add src/smolvla_grpo/phase12_wm_only_rollout.py scripts/grpo/train_phase12_wm_chunk_grpo.py tests/test_phase12_wm_only_rollout.py
git commit -m "feat(phase12): add WM-only rollout collector"
```

---

## Task 4: Real WM-Only Trainer Branch

**Files:**
- Modify: `scripts/grpo/train_phase12_wm_chunk_grpo.py`
- Test: `tests/test_phase12_training_loop.py`

- [ ] **Step 1: Write failing test that branch does not call selected-env collector**

Append this test to `tests/test_phase12_training_loop.py`:

```python
def test_wm_only_train_branch_does_not_call_selected_env_collector(monkeypatch, tmp_path) -> None:
    selected = tmp_path / "selected_action_rollout.mp4"
    oracle = tmp_path / "oracle_baseline.mp4"
    selected.write_bytes(b"selected")
    oracle.write_bytes(b"oracle")
    chunks = [torch.full((4, 4), 0.1 * (i + 1), dtype=torch.float32) for i in range(4)]
    episode = SimpleNamespace(
        total_env_reward=0.0,
        success_any=False,
        success_last=False,
        metadata={
            "phase12_train_mode": "wm_only",
            "segment_candidate_rewards": [[0.0, 1.0, 2.0, 3.0]],
            "candidate_rewards": [0.0, 1.0, 2.0, 3.0],
            "old_logprob_sums": [-1.0, -1.1, -1.2, -1.3],
            "proc_root_snapshots": [{"x": torch.zeros(1, 1)} for _ in range(4)],
            "unsquashed_chunks": chunks,
            "rollout_validation_video": str(selected),
            "selected_action_rollout_video": str(selected),
            "oracle_baseline_video": str(oracle),
            "oracle_baseline_video_status": "wm_only_not_used",
            "wm_decode_status": "disabled",
        },
    )

    monkeypatch.setattr(trainer, "load_phase12_train_resources", lambda args: (_TinyBundle(), object(), 4))
    monkeypatch.setattr(
        trainer,
        "collect_phase12_training_episode",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("selected-env collector called")),
    )
    monkeypatch.setattr(trainer, "collect_phase12_wm_only_training_episode", lambda **_kwargs: episode)

    code = trainer.main(
        [
            "--mode",
            "wm_grpo_train",
            "--phase12-train-mode",
            "wm_only",
            "--output-dir",
            str(tmp_path),
            "--jepa-repo",
            "/tmp/jepa",
            "--jepa-ckpt",
            "wm.pt",
            "--num-updates",
            "1",
            "--num-episodes",
            "1",
        ]
    )

    assert code == 0
    rows = [json.loads(x) for x in (tmp_path / "progress.jsonl").read_text().splitlines() if x.strip()]
    assert any(row.get("event") == "update_complete" and row.get("optimizer_step") is True for row in rows)
```

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
PYTHONPATH="${PWD}/src:${PYTHONPATH:-}" /rds/general/user/aa6622/home/.envs/lerobot_mw_py312/bin/python -m pytest tests/test_phase12_training_loop.py::test_wm_only_train_branch_does_not_call_selected_env_collector -q
```

Expected: FAIL until branch exists.

- [ ] **Step 3: Implement real branch dispatch**

In `run_wm_grpo_train`, use the branch from Task 3 Step 5 and ensure the WM-only branch calls `collect_phase12_wm_only_training_episode`.

Keep progress rows carrying the mode by adding this field to both skip and optimizer rows:

```python
                "phase12_train_mode": str(args.phase12_train_mode),
```

- [ ] **Step 4: Replace WM-only adapter with root-only path**

Replace `collect_phase12_wm_only_training_episode` body with a root-only implementation:

```python
def collect_phase12_wm_only_training_episode(**kwargs: Any) -> Any:
    """Collect one WM-only update: reset/proc/goal construction, no selected chunk env.step."""
    import numpy as np
    import torch

    from smolvla_grpo.phase11_rollout import detach_proc_snapshot
    from smolvla_grpo.lerobot_metaworld_adapter import (
        OfficialLeRobotMetaWorldGRPORollout,
        resolve_lerobot_horizon,
    )
    from smolvla_grpo.phase12_goals import Phase12Goal
    from smolvla_grpo.phase12_wm_only_rollout import collect_phase12_wm_only_episode
    from smolvla_grpo.phase12_wm_reward import _encode_structured, score_phase12_chunk_with_wm

    args = kwargs["args"]
    bundle = kwargs["bundle"]
    wm_bundle = kwargs["wm_bundle"]
    old_wrapper = kwargs["old_wrapper"]
    action_dim = int(kwargs["action_dim"])
    reset_seed = int(kwargs["reset_seed"])

    env_h = OfficialLeRobotMetaWorldGRPORollout(task=args.task, n_envs=1, enable_expert_oracle=True)
    try:
        obs = env_h.reset(reset_seed)
        frame = policy_rgb_from_obs(obs)
        wm_frame = wm_rgb_from_policy_rgb_corner2(frame)
        proprio = np.asarray(env_h.last_agent_pos(), dtype=np.float32)
        goal_latent = _encode_structured(
            wm_bundle,
            wm_frame,
            proprio,
            mode=args.goal_latent_mode,
        )
        goal = Phase12Goal(
            subgoal_index=0,
            frame_index_1based=int(args.chunk_len),
            frame_path=None,
            companion_frame_index_1based=None,
            companion_frame_path=None,
            proprio=proprio,
            goal_visual=goal_latent["visual"],
            goal_proprio=goal_latent.get("proprio"),
            source="wm_only_reset_root",
        )

        class RootSource:
            def reset(self, seed: int) -> dict[str, Any]:
                if int(seed) != int(reset_seed):
                    raise ValueError(f"unexpected seed {seed}; expected {reset_seed}")
                return {
                    "id": f"wm_only_seed{reset_seed}",
                    "obs": obs,
                    "image": wm_frame,
                    "policy_image": frame,
                    "proprio": proprio,
                    "proc": env_h.build_proc(obs, bundle=bundle),
                }

        def sampler(root, *, num_candidates: int, segment_index: int):
            proc = root["proc"]
            for candidate_index in range(int(num_candidates)):
                gen = torch.Generator(device=old_wrapper.bundle.device)
                gen.manual_seed(reset_seed * 1000003 + int(segment_index) * 7919 + int(candidate_index))
                sample = _sample_old_action_chunk(
                    old_wrapper,
                    proc,
                    chunk_len=int(args.chunk_len),
                    rng=gen,
                    use_inference_mode=bool(args.old_policy_inference_mode),
                )
                candidate = _phase12_sample_to_candidate_dict(sample, candidate_index=int(candidate_index))
                candidate["proc_root_snapshot"] = detach_proc_snapshot(proc)
                yield candidate

        def score_fn(root, candidate, goal, *, segment_index: int):
            del segment_index
            latent_goal = {"visual": goal.goal_visual}
            if args.goal_latent_mode == "visual_proprio":
                latent_goal["proprio"] = goal.goal_proprio
            return score_phase12_chunk_with_wm(
                wm_bundle=wm_bundle,
                image=root["image"],
                proprio=root["proprio"],
                chunk_actions=candidate.exec_actions_for_wm,
                goal=latent_goal,
                candidate_index=int(candidate.candidate_index),
                proprio_alpha=float(args.proprio_alpha),
                mode=args.goal_latent_mode,
            )

        episode = collect_phase12_wm_only_episode(
            root_source=RootSource(),
            reset_seed=reset_seed,
            policy_sampler=sampler,
            score_fn=score_fn,
            goals=[goal],
            group_size=int(args.group_size),
            reward_key=args.reward_key,
            action_profile=args.action_profile,
            action_low=np.full((action_dim,), -1.0, dtype=np.float32),
            action_high=np.full((action_dim,), 1.0, dtype=np.float32),
            preprocessor=wm_bundle.preprocessor,
            env_action_dim=action_dim,
            wm_action_dim=int(wm_bundle.planner_action_dim),
            metadata={
                "rollout_validation_video": "",
                "selected_action_rollout_video": "",
                "oracle_baseline_video": "",
                "oracle_baseline_video_status": "wm_only_not_used",
                "wm_decode_status": "disabled",
                "phase12_train_mode": "wm_only",
            },
        )
        return episode
    finally:
        env_h.close()
```

This first WM-only implementation uses reset root as both start and goal latent, so rewards may be flat for some seeds. Task 6 logs and handles zero-advantage groups.

- [ ] **Step 5: Run tests**

Run:

```bash
PYTHONPATH="${PWD}/src:${PYTHONPATH:-}" /rds/general/user/aa6622/home/.envs/lerobot_mw_py312/bin/python -m pytest tests/test_phase12_training_loop.py tests/test_phase12_wm_only_rollout.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/grpo/train_phase12_wm_chunk_grpo.py tests/test_phase12_training_loop.py
git commit -m "feat(phase12): train from WM-only candidate rewards"
```

---

## Task 5: Phase12 Serial Seed-Batch

**Files:**
- Modify: `scripts/grpo/train_phase12_wm_chunk_grpo.py`
- Test: `tests/test_phase12_training_loop.py`

- [ ] **Step 1: Write failing seed-batch aggregation test**

Append this test to `tests/test_phase12_training_loop.py`:

```python
def test_phase12_seed_batch_collects_seed_major_and_logs_reset_seeds(monkeypatch, tmp_path) -> None:
    selected = tmp_path / "selected_action_rollout.mp4"
    oracle = tmp_path / "oracle_baseline.mp4"
    selected.write_bytes(b"selected")
    oracle.write_bytes(b"oracle")
    calls: list[int] = []

    def fake_collect(**kwargs):
        seed = int(kwargs["reset_seed"])
        calls.append(seed)
        base = float(seed - 2000)
        chunks = [torch.full((2, 4), 0.1 * (i + 1), dtype=torch.float32) for i in range(3)]
        return SimpleNamespace(
            total_env_reward=base,
            success_any=seed % 2 == 0,
            success_last=seed % 2 == 0,
            metadata={
                "segment_candidate_rewards": [[base, base + 1.0, base + 2.0]],
                "candidate_rewards": [base, base + 1.0, base + 2.0],
                "old_logprob_sums": [-1.0, -1.1, -1.2],
                "proc_root_snapshots": [{"x": torch.zeros(1, 1)} for _ in range(3)],
                "unsquashed_chunks": chunks,
                "rollout_validation_video": str(selected),
                "selected_action_rollout_video": str(selected),
                "oracle_baseline_video": str(oracle),
                "oracle_baseline_video_status": "ok",
                "wm_decode_status": "disabled",
            },
        )

    monkeypatch.setattr(trainer, "load_phase12_train_resources", lambda args: (_TinyBundle(), object(), 4))
    monkeypatch.setattr(trainer, "collect_phase12_training_episode", fake_collect)

    code = trainer.main(
        [
            "--mode",
            "wm_grpo_train",
            "--output-dir",
            str(tmp_path),
            "--jepa-repo",
            "/tmp/jepa",
            "--jepa-ckpt",
            "wm.pt",
            "--group-size",
            "3",
            "--batch-size",
            "2",
            "--num-updates",
            "1",
            "--num-episodes",
            "1",
        ]
    )

    assert code == 0
    assert calls == [2000, 2001]
    rows = [json.loads(x) for x in (tmp_path / "progress.jsonl").read_text().splitlines() if x.strip()]
    update = [row for row in rows if row.get("event") == "update_complete"][-1]
    assert update["batch_size"] == 2
    assert update["reset_seeds"] == [2000, 2001]
    assert len(update["returns"]) == 6
    assert len(update["advantages"]) == 6
    assert len(update["per_seed_success_rate"]) == 2
```

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
PYTHONPATH="${PWD}/src:${PYTHONPATH:-}" /rds/general/user/aa6622/home/.envs/lerobot_mw_py312/bin/python -m pytest tests/test_phase12_training_loop.py::test_phase12_seed_batch_collects_seed_major_and_logs_reset_seeds -q
```

Expected: FAIL because `--batch-size` is not defined.

- [ ] **Step 3: Add `--batch-size` and manifest field**

Add CLI arg near `--group-size`:

```python
    p.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of reset seeds per update; each seed gets group_size WM candidate rollouts.",
    )
```

Add validation:

```python
    if int(args.batch_size) < 1:
        return "--batch-size must be >= 1."
```

Add manifest field:

```python
        "batch_size": int(args.batch_size),
```

- [ ] **Step 4: Add episode metadata combiner**

Add helper above `run_wm_grpo_train`:

```python
def _combine_phase12_seed_batch_metadata(episodes: list[Any]) -> dict[str, Any]:
    combined: dict[str, Any] = {
        "candidate_rewards": [],
        "segment_candidate_rewards": [],
        "old_logprob_sums": [],
        "proc_root_snapshots": [],
        "unsquashed_chunks": [],
        "successes": [],
        "per_seed_success_rate": [],
    }
    for episode in episodes:
        meta = dict(getattr(episode, "metadata", {}) or {})
        combined["candidate_rewards"].extend(list(meta.get("candidate_rewards", [])))
        combined["segment_candidate_rewards"].extend(list(meta.get("segment_candidate_rewards", [])))
        combined["old_logprob_sums"].extend(list(meta.get("old_logprob_sums", [])))
        combined["proc_root_snapshots"].extend(list(meta.get("proc_root_snapshots", [])))
        combined["unsquashed_chunks"].extend(list(meta.get("unsquashed_chunks", [])))
        success = bool(getattr(episode, "success_any", meta.get("success_any", False)))
        combined["successes"].append(success)
        combined["per_seed_success_rate"].append(1.0 if success else 0.0)
    return combined
```

- [ ] **Step 5: Change update loop seed schedule**

Replace:

```python
        reset_seed = int(args.train_seed_base) + int(update_index)
        episode = ...
```

with:

```python
        batch_size_i = int(args.batch_size)
        seed_batch_base = int(args.train_seed_base) + int(update_index) * batch_size_i
        reset_seeds = [seed_batch_base + b for b in range(batch_size_i)]
        episodes = []
        for reset_seed in reset_seeds:
            if args.phase12_train_mode == "wm_only":
                episode_i = collect_phase12_wm_only_training_episode(
                    args=args,
                    bundle=bundle,
                    wm_bundle=wm_bundle,
                    old_wrapper=old_wrapper,
                    action_dim=action_dim,
                    update_index=update_index,
                    reset_seed=int(reset_seed),
                    output_dir=out,
                )
            else:
                episode_i = collect_phase12_training_episode(
                    args=args,
                    bundle=bundle,
                    wm_bundle=wm_bundle,
                    old_policy=old_policy,
                    old_wrapper=old_wrapper,
                    action_dim=action_dim,
                    update_index=update_index,
                    reset_seed=int(reset_seed),
                    output_dir=out,
                )
            episodes.append(episode_i)
        episode = episodes[0]
        batch_meta = _combine_phase12_seed_batch_metadata(episodes)
```

Then replace `meta = dict(getattr(episode, "metadata", {}) or {})` with:

```python
        meta = dict(batch_meta)
```

- [ ] **Step 6: Preserve per-group advantages**

Leave current logic:

```python
        segment_advantages = [compute_group_advantages(row) for row in segment_rewards]
```

Because `segment_candidate_rewards` is now seed-major and each row is one seed/segment group of size `G`.

- [ ] **Step 7: Log seed-batch fields**

Add these to both skip row and optimizer row:

```python
                "batch_size": int(args.batch_size),
                "reset_seeds": reset_seeds,
                "per_seed_success_rate": list(meta.get("per_seed_success_rate", [])),
                "episode_count": len(episodes),
```

- [ ] **Step 8: Run tests**

Run:

```bash
PYTHONPATH="${PWD}/src:${PYTHONPATH:-}" /rds/general/user/aa6622/home/.envs/lerobot_mw_py312/bin/python -m pytest tests/test_phase12_training_loop.py -q
```

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add scripts/grpo/train_phase12_wm_chunk_grpo.py tests/test_phase12_training_loop.py
git commit -m "feat(phase12): add serial seed-batch training"
```

---

## Task 6: Batched Logprob Recompute

**Files:**
- Modify: `scripts/grpo/train_phase12_wm_chunk_grpo.py`
- Test: `tests/test_phase12_training_loop.py`

- [ ] **Step 1: Write failing test for batched recompute telemetry**

Append this test to `tests/test_phase12_training_loop.py`:

```python
def test_phase12_batched_logprob_mode_records_forward_batches(monkeypatch, tmp_path) -> None:
    selected = tmp_path / "selected_action_rollout.mp4"
    oracle = tmp_path / "oracle_baseline.mp4"
    selected.write_bytes(b"selected")
    oracle.write_bytes(b"oracle")
    chunks = [torch.full((2, 4), 0.1 * (i + 1), dtype=torch.float32) for i in range(4)]
    episode = SimpleNamespace(
        total_env_reward=1.0,
        success_any=False,
        success_last=False,
        metadata={
            "segment_candidate_rewards": [[0.0, 1.0, 2.0, 3.0]],
            "candidate_rewards": [0.0, 1.0, 2.0, 3.0],
            "old_logprob_sums": [-1.0, -1.1, -1.2, -1.3],
            "proc_root_snapshots": [{"x": torch.zeros(1, 1)} for _ in range(4)],
            "unsquashed_chunks": chunks,
            "rollout_validation_video": str(selected),
            "selected_action_rollout_video": str(selected),
            "oracle_baseline_video": str(oracle),
            "oracle_baseline_video_status": "ok",
            "wm_decode_status": "disabled",
        },
    )
    monkeypatch.setattr(trainer, "load_phase12_train_resources", lambda args: (_TinyBundle(), object(), 4))
    monkeypatch.setattr(trainer, "collect_phase12_training_episode", lambda **_kwargs: episode)

    code = trainer.main(
        [
            "--mode",
            "wm_grpo_train",
            "--output-dir",
            str(tmp_path),
            "--jepa-repo",
            "/tmp/jepa",
            "--jepa-ckpt",
            "wm.pt",
            "--num-updates",
            "1",
            "--num-episodes",
            "1",
            "--logprob-recompute-mode",
            "batched",
            "--logprob-batch-size",
            "2",
        ]
    )

    assert code == 0
    rows = [json.loads(x) for x in (tmp_path / "progress.jsonl").read_text().splitlines() if x.strip()]
    update = [row for row in rows if row.get("event") == "update_complete"][-1]
    assert update["logprob_recompute_mode"] == "batched"
    assert update["logprob_batch_size"] == 2
    assert update["num_logprob_forward_batches"] == 2
```

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
PYTHONPATH="${PWD}/src:${PYTHONPATH:-}" /rds/general/user/aa6622/home/.envs/lerobot_mw_py312/bin/python -m pytest tests/test_phase12_training_loop.py::test_phase12_batched_logprob_mode_records_forward_batches -q
```

Expected: FAIL because args and telemetry do not exist.

- [ ] **Step 3: Add args and validation**

Add near existing `--logprob-backward-mode`:

```python
    p.add_argument(
        "--logprob-recompute-mode",
        choices=("batched", "loop"),
        default="batched",
        help="Batched is the default fast path; loop is a row-by-row fallback.",
    )
    p.add_argument("--logprob-batch-size", type=int, default=16)
```

Validate:

```python
    if int(args.logprob_batch_size) < 1:
        return "--logprob-batch-size must be >= 1."
```

Manifest:

```python
        "logprob_recompute_mode": str(args.logprob_recompute_mode),
        "logprob_batch_size": int(args.logprob_batch_size),
```

- [ ] **Step 4: Add batched recompute helper**

Add above `_backward_chunk_grpo_loss_microbatched`:

```python
def _recompute_phase12_logprobs_batched(
    *,
    train_wrapper: Any,
    proc_snapshots: list[Any],
    unsquashed_chunks: list[Any],
    batch_size: int,
) -> tuple[Any, int]:
    import torch

    values: list[torch.Tensor] = []
    forwards = 0
    bs = max(int(batch_size), 1)
    for start in range(0, len(unsquashed_chunks), bs):
        stop = min(start + bs, len(unsquashed_chunks))
        rows = [
            train_wrapper.get_action_probs_for_chunk_from_proc(proc, chunk).sum()
            for proc, chunk in zip(proc_snapshots[start:stop], unsquashed_chunks[start:stop], strict=True)
        ]
        values.extend(rows)
        forwards += 1
    if not values:
        raise RuntimeError("batched logprob recompute needs at least one chunk")
    return torch.stack(values), forwards
```

This helper keeps the first implementation behavior-equivalent to the existing row loop while exposing the batch-size and telemetry contract. A later optimization can replace the inner row list with true policy batch API without changing callers.

- [ ] **Step 5: Use helper in stack mode**

Replace stack-mode `new_lp_rows = [...]` block with:

```python
            new_lp, num_logprob_forward_batches = _recompute_phase12_logprobs_batched(
                train_wrapper=train_wrapper,
                proc_snapshots=meta["proc_root_snapshots"],
                unsquashed_chunks=meta["unsquashed_chunks"],
                batch_size=int(args.logprob_batch_size),
            )
```

In microbatch path, set:

```python
            num_logprob_forward_batches = len(meta["unsquashed_chunks"])
```

Add to progress rows:

```python
                "logprob_recompute_mode": str(args.logprob_recompute_mode),
                "logprob_batch_size": int(args.logprob_batch_size),
                "num_logprob_forward_batches": int(num_logprob_forward_batches),
```

- [ ] **Step 6: Run tests**

Run:

```bash
PYTHONPATH="${PWD}/src:${PYTHONPATH:-}" /rds/general/user/aa6622/home/.envs/lerobot_mw_py312/bin/python -m pytest tests/test_phase12_training_loop.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add scripts/grpo/train_phase12_wm_chunk_grpo.py tests/test_phase12_training_loop.py
git commit -m "feat(phase12): add batched logprob recompute contract"
```

---

## Task 7: WM Encode Cache and Batch-Score Contract

**Files:**
- Modify: `src/smolvla_grpo/phase12_wm_reward.py`
- Modify: `scripts/grpo/train_phase12_wm_chunk_grpo.py`
- Test: `tests/test_phase12_wm_reward.py`

- [ ] **Step 1: Write failing cache test**

Append to `tests/test_phase12_wm_reward.py`:

```python
def test_phase12_root_encode_cache_reuses_same_root(monkeypatch) -> None:
    from smolvla_grpo import phase12_wm_reward as reward

    calls = {"encode": 0}

    class Model:
        def encode(self, obs):
            del obs
            calls["encode"] += 1
            return {
                "visual": torch.zeros(1, 1, 1, dtype=torch.float32),
                "proprio": torch.zeros(1, 1, 1, dtype=torch.float32),
            }

    wm_bundle = SimpleNamespace(
        model=Model(),
        device=torch.device("cpu"),
        proprio_dim=4,
    )
    cache = reward.Phase12WMEncodeCache()
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    proprio = np.zeros(4, dtype=np.float32)

    first = cache.encode_root(wm_bundle, image, proprio, mode="visual_proprio")
    second = cache.encode_root(wm_bundle, image, proprio, mode="visual_proprio")

    assert calls["encode"] == 1
    assert first["visual"] is second["visual"]
```

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
PYTHONPATH="${PWD}/src:${PYTHONPATH:-}" /rds/general/user/aa6622/home/.envs/lerobot_mw_py312/bin/python -m pytest tests/test_phase12_wm_reward.py -q -k encode_cache
```

Expected: FAIL because `Phase12WMEncodeCache` does not exist.

- [ ] **Step 3: Add encode cache**

Add to `src/smolvla_grpo/phase12_wm_reward.py` after `_array_stats`:

```python
class Phase12WMEncodeCache:
    """Small per-update cache for JEPA-WM root/goal encodes."""

    def __init__(self) -> None:
        self._cache: dict[tuple[str, str, str], dict[str, torch.Tensor]] = {}
        self.encode_calls = 0
        self.cache_hits = 0

    @staticmethod
    def _key(image: Any, proprio: Any, mode: str) -> tuple[str, str, str]:
        img = np.ascontiguousarray(np.asarray(image))
        prop = np.ascontiguousarray(np.asarray(proprio, dtype=np.float32).reshape(-1))
        return (
            hashlib.sha256(img.tobytes()).hexdigest(),
            hashlib.sha256(prop.tobytes()).hexdigest(),
            str(mode),
        )

    def encode_root(
        self,
        wm_bundle: Any,
        image: np.ndarray,
        proprio: np.ndarray,
        *,
        mode: str,
    ) -> dict[str, torch.Tensor]:
        key = self._key(image, proprio, mode)
        if key in self._cache:
            self.cache_hits += 1
            return self._cache[key]
        encoded = _encode_structured(wm_bundle, image, proprio, mode=mode)
        self.encode_calls += 1
        self._cache[key] = encoded
        return encoded
```

- [ ] **Step 4: Add cached scorer**

Add below `score_phase12_chunk_with_wm`:

```python
def score_phase12_chunk_with_wm_cached(
    *,
    wm_bundle: Any,
    encode_cache: Phase12WMEncodeCache,
    image: np.ndarray,
    proprio: np.ndarray,
    chunk_actions: np.ndarray,
    goal: Mapping[str, torch.Tensor],
    candidate_index: int,
    proprio_alpha: float,
    mode: str,
) -> Phase12Score:
    actions = np.asarray(chunk_actions, dtype=np.float32)
    if actions.ndim != 2:
        raise ValueError(f"chunk_actions must be 2D, got {actions.shape}")
    start = encode_cache.encode_root(wm_bundle, image, proprio, mode=mode)
    final = _final_structured_after_unroll(wm_bundle, start, actions, mode=mode)
    return score_progress(
        candidate_index=int(candidate_index),
        start=start,
        final=final,
        goal=goal,
        proprio_alpha=float(proprio_alpha),
        mode=mode,
    )
```

- [ ] **Step 5: Use cache in trainer score functions**

In each Phase12 score function closure, create cache once before `score_fn`:

```python
        encode_cache = Phase12WMEncodeCache()
```

Import:

```python
from smolvla_grpo.phase12_wm_reward import Phase12WMEncodeCache, score_phase12_chunk_with_wm_cached
```

Replace `score_phase12_chunk_with_wm(...)` calls with `score_phase12_chunk_with_wm_cached(..., encode_cache=encode_cache, ...)`.

After episode collection, add metadata:

```python
        meta["wm_encode_calls"] = int(encode_cache.encode_calls)
        meta["wm_cache_hits"] = int(encode_cache.cache_hits)
```

Add to progress rows:

```python
                "wm_encode_calls": int(meta.get("wm_encode_calls", 0)),
                "wm_cache_hits": int(meta.get("wm_cache_hits", 0)),
```

- [ ] **Step 6: Run tests**

Run:

```bash
PYTHONPATH="${PWD}/src:${PYTHONPATH:-}" /rds/general/user/aa6622/home/.envs/lerobot_mw_py312/bin/python -m pytest tests/test_phase12_wm_reward.py tests/test_phase12_training_loop.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/smolvla_grpo/phase12_wm_reward.py scripts/grpo/train_phase12_wm_chunk_grpo.py tests/test_phase12_wm_reward.py
git commit -m "feat(phase12): cache WM root encodes"
```

---

## Task 8: Phase11-Grade Telemetry

**Files:**
- Modify: `scripts/grpo/train_phase12_wm_chunk_grpo.py`
- Test: `tests/test_phase12_training_loop.py`

- [ ] **Step 1: Write failing telemetry test**

Append to `tests/test_phase12_training_loop.py`:

```python
def test_phase12_progress_row_contains_phase11_parity_telemetry(monkeypatch, tmp_path) -> None:
    selected = tmp_path / "selected_action_rollout.mp4"
    oracle = tmp_path / "oracle_baseline.mp4"
    selected.write_bytes(b"selected")
    oracle.write_bytes(b"oracle")
    chunks = [torch.full((2, 4), 0.1 * (i + 1), dtype=torch.float32) for i in range(4)]
    episode = SimpleNamespace(
        total_env_reward=1.0,
        success_any=True,
        success_last=True,
        metadata={
            "segment_candidate_rewards": [[0.0, 1.0, 2.0, 3.0]],
            "candidate_rewards": [0.0, 1.0, 2.0, 3.0],
            "old_logprob_sums": [-1.0, -1.1, -1.2, -1.3],
            "proc_root_snapshots": [{"x": torch.zeros(1, 1)} for _ in range(4)],
            "unsquashed_chunks": chunks,
            "rollout_validation_video": str(selected),
            "selected_action_rollout_video": str(selected),
            "oracle_baseline_video": str(oracle),
            "oracle_baseline_video_status": "ok",
            "wm_decode_status": "disabled",
            "wm_encode_calls": 1,
            "wm_cache_hits": 3,
        },
    )
    monkeypatch.setattr(trainer, "load_phase12_train_resources", lambda args: (_TinyBundle(), object(), 4))
    monkeypatch.setattr(trainer, "collect_phase12_training_episode", lambda **_kwargs: episode)

    assert trainer.main(["--mode", "wm_grpo_train", "--output-dir", str(tmp_path), "--jepa-repo", "/tmp/jepa", "--jepa-ckpt", "wm.pt", "--num-updates", "1", "--num-episodes", "1"]) == 0

    rows = [json.loads(x) for x in (tmp_path / "progress.jsonl").read_text().splitlines() if x.strip()]
    row = [r for r in rows if r.get("event") == "update_complete"][-1]
    for key in [
        "phase12_train_mode",
        "batch_size",
        "group_size",
        "reset_seeds",
        "num_loss_units",
        "rollout_seconds",
        "optimize_seconds",
        "update_seconds",
        "wm_encode_calls",
        "wm_cache_hits",
        "ratio_clip_fraction",
        "approx_kl",
        "proc_mem_after_optimize_tree_rss_kb",
    ]:
        assert key in row
```

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
PYTHONPATH="${PWD}/src:${PYTHONPATH:-}" /rds/general/user/aa6622/home/.envs/lerobot_mw_py312/bin/python -m pytest tests/test_phase12_training_loop.py::test_phase12_progress_row_contains_phase11_parity_telemetry -q
```

Expected: FAIL because telemetry fields are incomplete.

- [ ] **Step 3: Import memory helper**

Add near imports:

```python
from smolvla_grpo.process_memory import prefixed_process_tree_memory_fields
```

Add helper:

```python
def _proc_mem_fields(stage: str) -> dict[str, int]:
    return prefixed_process_tree_memory_fields(f"proc_mem_{stage}")
```

- [ ] **Step 4: Capture timing and memory**

At update start:

```python
        proc_mem_update_start = _proc_mem_fields("update_start")
        rollout_t0 = time.perf_counter()
```

After episode collection:

```python
        rollout_seconds = float(time.perf_counter() - rollout_t0)
        proc_mem_after_rollout = _proc_mem_fields("after_rollout")
```

Before backward:

```python
        optimize_t0 = time.perf_counter()
```

After optimizer or skip:

```python
        optimize_seconds = float(time.perf_counter() - optimize_t0)
        proc_mem_after_optimize = _proc_mem_fields("after_optimize")
```

- [ ] **Step 5: Add common row fields**

Before skip/optimizer row writes, build:

```python
        progress_common = {
            "phase12_train_mode": str(args.phase12_train_mode),
            "action_profile": str(args.action_profile),
            "batch_size": int(args.batch_size),
            "group_size": int(args.group_size),
            "reset_seeds": reset_seeds,
            "per_seed_success_rate": list(meta.get("per_seed_success_rate", [])),
            "episode_count": len(episodes),
            "num_loss_units": int(len(meta["old_logprob_sums"])),
            "rollout_seconds": float(rollout_seconds),
            "wm_encode_calls": int(meta.get("wm_encode_calls", 0)),
            "wm_cache_hits": int(meta.get("wm_cache_hits", 0)),
            **proc_mem_update_start,
            **proc_mem_after_rollout,
        }
```

Merge `progress_common`, `proc_mem_after_optimize`, `optimize_seconds`, and `update_seconds` into both skip and optimizer progress rows.

- [ ] **Step 6: Run tests**

Run:

```bash
PYTHONPATH="${PWD}/src:${PYTHONPATH:-}" /rds/general/user/aa6622/home/.envs/lerobot_mw_py312/bin/python -m pytest tests/test_phase12_training_loop.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add scripts/grpo/train_phase12_wm_chunk_grpo.py tests/test_phase12_training_loop.py
git commit -m "feat(phase12): log parity telemetry"
```

---

## Task 9: PBS Train Smoke and Production Scripts

**Files:**
- Create: `scripts/grpo/phase12_seedbatch_smoke_u2.pbs`
- Create: `scripts/grpo/phase12_seedbatch_b4_g16_train_0000_0050.pbs`
- Create: `tests/test_phase12_pbs_static.py`

- [ ] **Step 1: Write failing static PBS tests**

Create `tests/test_phase12_pbs_static.py`:

```python
from __future__ import annotations

from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (PROJECT / path).read_text(encoding="utf-8")


def test_phase12_seedbatch_smoke_pbs_contract() -> None:
    text = _read("scripts/grpo/phase12_seedbatch_smoke_u2.pbs")
    assert "#PBS -l select=1:ncpus=8:mem=64gb:ngpus=1:gpu_type=RTX6000" in text
    assert "--phase12-train-mode wm_only" in text
    assert "--batch-size 2" in text
    assert "--group-size 8" in text
    assert "--num-updates 2" in text
    assert "--num-episodes 2" in text
    assert "--logprob-recompute-mode batched" in text
    assert "--logprob-batch-size 16" in text
    assert "phase11_cpu_mem_telemetry.sh" in text
    assert 'assert int(manifest["batch_size"]) == 2' in text
    assert 'assert len(last["reset_seeds"]) == 2' in text


def test_phase12_seedbatch_prod_pbs_contract() -> None:
    text = _read("scripts/grpo/phase12_seedbatch_b4_g16_train_0000_0050.pbs")
    assert "#PBS -l select=1:ncpus=8:mem=128gb:ngpus=1:gpu_type=RTX6000" in text
    assert "--phase12-train-mode wm_only" in text
    assert "--batch-size 4" in text
    assert "--group-size 16" in text
    assert "--num-updates 50" in text
    assert "--save-every 2" in text
    assert "update_0050.pt" in text
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
PYTHONPATH="${PWD}/src:${PYTHONPATH:-}" /rds/general/user/aa6622/home/.envs/lerobot_mw_py312/bin/python -m pytest tests/test_phase12_pbs_static.py -q
```

Expected: FAIL because files do not exist.

- [ ] **Step 3: Add smoke PBS**

Create `scripts/grpo/phase12_seedbatch_smoke_u2.pbs`:

```bash
#!/usr/bin/env bash
#PBS -N p12wmb2g8
#PBS -l select=1:ncpus=8:mem=64gb:ngpus=1:gpu_type=RTX6000
#PBS -l walltime=01:00:00
#PBS -j oe
#PBS -o logs/pbs/grpo/phase12_seedbatch_smoke_u2.out

set -euo pipefail

if [[ -f /etc/profile.d/modules.sh ]]; then
  . /etc/profile.d/modules.sh
fi
module purge >/dev/null 2>&1 || true
module load tools/prod
module load Python/3.12.3-GCCcore-13.3.0
module load Mesa/24.1.3-GCCcore-13.3.0

PROJECT_ROOT="$(cd "${PBS_O_WORKDIR:-/rds/general/user/aa6622/home/project}" && pwd)"
cd "${PROJECT_ROOT}"
mkdir -p logs/pbs/grpo

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
export JEPA_WM_DISABLE_IMAGE_HEAD="${JEPA_WM_DISABLE_IMAGE_HEAD:-1}"

PYTHON_BIN="${GRPO_PYTHON:-/rds/general/user/aa6622/home/.envs/lerobot_mw_py312/bin/python}"
CHECKPOINT="${PHASE12_CHECKPOINT:-/rds/general/user/aa6622/home/.cache/huggingface/hub/models--jadechoghari--smolvla_metaworld/snapshots/ef3089ecb84eeeb7d33fedab24f6c76180a68900}"
JEPA_CKPT="${PHASE12_JEPA_CKPT:-/rds/general/user/aa6622/home/.cache/huggingface/hub/models--facebook--jepa-wms/snapshots/9b9c41ef249466630dbf1a20e78391865d07b3b9/jepa_wm_metaworld.pth.tar}"
JEPA_REPO="${PHASE12_JEPA_REPO:-/rds/general/user/aa6622/home/research/RESEARCH_PAPER_CLONES/jepa-wms}"
OUT="${PHASE12_OUT:-${PROJECT_ROOT}/artifacts/phase12_wm_only_seedbatch_smoke_u2}"

scripts/grpo/phase11_cpu_mem_telemetry.sh "${OUT}" 5 &
TELEMETRY_PID=$!
trap 'kill "${TELEMETRY_PID}" >/dev/null 2>&1 || true' EXIT

"${PYTHON_BIN}" scripts/grpo/train_phase12_wm_chunk_grpo.py \
  --mode wm_grpo_train \
  --phase12-train-mode wm_only \
  --checkpoint "${CHECKPOINT}" \
  --jepa-ckpt "${JEPA_CKPT}" \
  --jepa-repo "${JEPA_REPO}" \
  --output-dir "${OUT}" \
  --task push-v3 \
  --env-backend official_lerobot_guarded \
  --action-profile "${PHASE12_ACTION_PROFILE:-official_jepa_mirror}" \
  --chunk-len 25 \
  --batch-size 2 \
  --group-size 8 \
  --num-episodes 2 \
  --num-updates 2 \
  --max-steps 120 \
  --train-seed-base 9000 \
  --save-every 1 \
  --lr "${PHASE12_LR:-1e-5}" \
  --clip-eps "${PHASE12_CLIP_EPS:-0.2}" \
  --init-log-std "${PHASE12_INIT_LOG_STD:--2.0}" \
  --euler-step-noise-std "${PHASE12_EULER_NOISE:-0.2}" \
  --goal-latent-mode visual_proprio \
  --proprio-alpha 0.1 \
  --reward-key wm_latent_progress \
  --ratio-mode chunk \
  --logprob-backward-mode microbatch \
  --logprob-recompute-mode batched \
  --logprob-batch-size 16 \
  --action-transform no_tanh \
  --reset-mismatch fail \
  --decode-candidates selected

test -f "${OUT}/train_manifest.json"
test -f "${OUT}/progress.jsonl"
test -f "${OUT}/checkpoints/latest.pt"
test -f "${OUT}/checkpoints/update_0002.pt"

"${PYTHON_BIN}" - "${OUT}" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
manifest = json.loads((out / "train_manifest.json").read_text(encoding="utf-8"))
rows = [json.loads(line) for line in (out / "progress.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
complete = [row for row in rows if row.get("event") == "update_complete"]
if len(complete) < 2:
    raise SystemExit(f"expected at least 2 update_complete rows, found {len(complete)}")
last = complete[-1]
assert manifest["phase12_train_mode"] == "wm_only"
assert int(manifest["batch_size"]) == 2
assert int(manifest["group_size"]) == 8
assert int(last["batch_size"]) == 2
assert len(last["reset_seeds"]) == 2
assert len(last["returns"]) == 16
assert "per_seed_success_rate" in last
assert "proc_mem_after_optimize_tree_rss_kb" in last
PY

echo "PHASE12_SEEDBATCH_SMOKE_DONE out=${OUT}"
```

- [ ] **Step 4: Add production PBS**

Create `scripts/grpo/phase12_seedbatch_b4_g16_train_0000_0050.pbs`:

```bash
#!/usr/bin/env bash
#PBS -N p12wmb4g16
#PBS -l select=1:ncpus=8:mem=128gb:ngpus=1:gpu_type=RTX6000
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -o logs/pbs/grpo/phase12_seedbatch_b4_g16_train_0000_0050.out

set -euo pipefail

if [[ -f /etc/profile.d/modules.sh ]]; then
  . /etc/profile.d/modules.sh
fi
module purge >/dev/null 2>&1 || true
module load tools/prod
module load Python/3.12.3-GCCcore-13.3.0
module load Mesa/24.1.3-GCCcore-13.3.0

PROJECT_ROOT="$(cd "${PBS_O_WORKDIR:-/rds/general/user/aa6622/home/project}" && pwd)"
cd "${PROJECT_ROOT}"
mkdir -p logs/pbs/grpo

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export JEPA_WM_DISABLE_IMAGE_HEAD="${JEPA_WM_DISABLE_IMAGE_HEAD:-1}"

PYTHON_BIN="${GRPO_PYTHON:-/rds/general/user/aa6622/home/.envs/lerobot_mw_py312/bin/python}"
CHECKPOINT="${PHASE12_CHECKPOINT:-/rds/general/user/aa6622/home/.cache/huggingface/hub/models--jadechoghari--smolvla_metaworld/snapshots/ef3089ecb84eeeb7d33fedab24f6c76180a68900}"
JEPA_CKPT="${PHASE12_JEPA_CKPT:-/rds/general/user/aa6622/home/.cache/huggingface/hub/models--facebook--jepa-wms/snapshots/9b9c41ef249466630dbf1a20e78391865d07b3b9/jepa_wm_metaworld.pth.tar}"
JEPA_REPO="${PHASE12_JEPA_REPO:-/rds/general/user/aa6622/home/research/RESEARCH_PAPER_CLONES/jepa-wms}"
OUT="${PHASE12_OUT:-${PROJECT_ROOT}/artifacts/phase12_wm_only_b4_g16_lr5e6_clip01_u50}"

scripts/grpo/phase11_cpu_mem_telemetry.sh "${OUT}" 5 &
TELEMETRY_PID=$!
trap 'kill "${TELEMETRY_PID}" >/dev/null 2>&1 || true' EXIT

"${PYTHON_BIN}" scripts/grpo/train_phase12_wm_chunk_grpo.py \
  --mode wm_grpo_train \
  --phase12-train-mode wm_only \
  --checkpoint "${CHECKPOINT}" \
  --jepa-ckpt "${JEPA_CKPT}" \
  --jepa-repo "${JEPA_REPO}" \
  --output-dir "${OUT}" \
  --task push-v3 \
  --env-backend official_lerobot_guarded \
  --action-profile "${PHASE12_ACTION_PROFILE:-official_jepa_mirror}" \
  --chunk-len 25 \
  --batch-size 4 \
  --group-size 16 \
  --num-episodes 50 \
  --num-updates 50 \
  --max-steps 120 \
  --train-seed-base "${PHASE12_SEED_BASE:-10000}" \
  --save-every 2 \
  --lr "${PHASE12_LR:-5e-6}" \
  --clip-eps "${PHASE12_CLIP_EPS:-0.1}" \
  --init-log-std "${PHASE12_INIT_LOG_STD:--2.0}" \
  --euler-step-noise-std "${PHASE12_EULER_NOISE:-0.2}" \
  --goal-latent-mode visual_proprio \
  --proprio-alpha 0.1 \
  --reward-key wm_latent_progress \
  --ratio-mode chunk \
  --logprob-backward-mode microbatch \
  --logprob-recompute-mode batched \
  --logprob-batch-size 16 \
  --action-transform no_tanh \
  --reset-mismatch fail \
  --decode-candidates selected

test -f "${OUT}/train_manifest.json"
test -f "${OUT}/progress.jsonl"
test -f "${OUT}/checkpoints/latest.pt"
test -f "${OUT}/checkpoints/update_0050.pt"
echo "PHASE12_SEEDBATCH_TRAIN_DONE out=${OUT}"
```

- [ ] **Step 5: Run static tests**

Run:

```bash
PYTHONPATH="${PWD}/src:${PYTHONPATH:-}" /rds/general/user/aa6622/home/.envs/lerobot_mw_py312/bin/python -m pytest tests/test_phase12_pbs_static.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/grpo/phase12_seedbatch_smoke_u2.pbs scripts/grpo/phase12_seedbatch_b4_g16_train_0000_0050.pbs tests/test_phase12_pbs_static.py
git commit -m "chore(phase12): add seed-batch PBS scripts"
```

---

## Task 10: Phase12 Eval Top-K and PBS Sweep

**Files:**
- Modify: `scripts/grpo/eval_phase12_checkpoint_sweep.py`
- Create: `scripts/grpo/phase12_eval_sweep_topk.pbs`
- Test: `tests/test_phase12_eval_sweep.py`
- Test: `tests/test_phase12_pbs_static.py`

- [ ] **Step 1: Write failing top-k unit test**

Create `tests/test_phase12_eval_sweep.py`:

```python
from __future__ import annotations

from scripts.grpo import eval_phase12_checkpoint_sweep as sweep


def test_rank_phase12_eval_rows_prefers_success_then_reward() -> None:
    rows = [
        {"update": 2, "pc_success": 20.0, "avg_sum_reward": 100.0},
        {"update": 4, "pc_success": 40.0, "avg_sum_reward": 50.0},
        {"update": 6, "pc_success": 40.0, "avg_sum_reward": 70.0},
    ]

    ranked = sweep._rank_eval_rows(rows)

    assert [row["update"] for row in ranked] == [6, 4, 2]
```

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
PYTHONPATH="${PWD}/src:${PYTHONPATH:-}" /rds/general/user/aa6622/home/.envs/lerobot_mw_py312/bin/python -m pytest tests/test_phase12_eval_sweep.py -q
```

Expected: FAIL because `_rank_eval_rows` does not exist.

- [ ] **Step 3: Add ranking helper**

Add to `scripts/grpo/eval_phase12_checkpoint_sweep.py` near `_checkpoint_path`:

```python
def _rank_eval_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            float(row.get("pc_success", 0.0)),
            float(row.get("avg_sum_reward", 0.0)),
        ),
        reverse=True,
    )
```

- [ ] **Step 4: Add CLI args**

Add to `main()` parser:

```python
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-k-episodes", type=int, default=100)
```

Pass to `run_sweep`.

- [ ] **Step 5: Add top-k to `run_sweep` signature and result**

Extend signatures for `run_sweep` and `run_sweep_inprocess_vector`:

```python
    top_k: int = 0,
    top_k_episodes: int = 100,
```

After primary rows are collected, add:

```python
    topk_rows: list[dict[str, Any]] = []
    if int(top_k) > 0:
        for row in _rank_eval_rows(rows)[: int(top_k)]:
            update = int(row["update"])
            ckpt = _make_base_eval_checkpoint(base_checkpoint, sweep_dir / "topk_base", task=task) if update == 0 else _checkpoint_path(run_dir, update)
            out_dir = sweep_dir / f"topk_update_{update:04d}_{int(top_k_episodes)}ep"
            out_dir.mkdir(parents=True, exist_ok=True)
            load_policy_checkpoint_into_bundle(bundle, ckpt)
            summary = evaluate_loaded_policy_vectorized(
                bundle=bundle,
                base_checkpoint=base_checkpoint,
                grpo_checkpoint=ckpt,
                output_dir=out_dir,
                task=task,
                episodes=int(top_k_episodes),
                eval_seed_start=eval_seed_start,
                n_envs=n_envs,
                rollout_execution=rollout_execution,
                max_steps=resolved_max_steps,
                chunk_len=int(chunk_len),
            )
            topk_rows.append(
                {
                    "update": update,
                    "pc_success": float(summary.get("pc_success", 0.0)),
                    "avg_sum_reward": float(summary.get("avg_sum_reward", 0.0)),
                    "avg_max_reward": float(summary.get("avg_max_reward", 0.0)),
                    "episodes": int(summary.get("episodes", top_k_episodes)),
                    "eval_summary_path": str(out_dir / "eval_summary.json"),
                }
            )
```

Add to result:

```python
        "top_k": int(top_k),
        "top_k_episodes": int(top_k_episodes),
        "topk_rows": topk_rows,
```

For subprocess mode, raise clear error if `top_k > 0`:

```python
    if int(top_k) > 0 and execution_mode != "inprocess_vector":
        raise ValueError("Phase12 top-k requires execution_mode='inprocess_vector'")
```

- [ ] **Step 6: Extend PBS static tests**

Append to `tests/test_phase12_pbs_static.py`:

```python
def test_phase12_eval_sweep_topk_pbs_contract() -> None:
    text = _read("scripts/grpo/phase12_eval_sweep_topk.pbs")
    assert "#PBS -l select=1:ncpus=32:mem=64gb:ngpus=1:gpu_type=RTX6000" in text
    assert "eval_phase12_checkpoint_sweep.py" in text
    assert "--episodes \"${EPISODES}\"" in text
    assert "--n-envs \"${N_ENVS}\"" in text
    assert "--rollout-execution vector_async" in text
    assert "--top-k \"${TOP_K}\"" in text
    assert "--top-k-episodes \"${TOP_K_EPISODES}\"" in text
```

- [ ] **Step 7: Add PBS eval script**

Create `scripts/grpo/phase12_eval_sweep_topk.pbs`:

```bash
#!/usr/bin/env bash
#PBS -N p12evaltopk
#PBS -l select=1:ncpus=32:mem=64gb:ngpus=1:gpu_type=RTX6000
#PBS -l walltime=08:00:00
#PBS -j oe
#PBS -o logs/pbs/grpo/phase12_eval_sweep_topk.out

set -euo pipefail

if [[ -f /etc/profile.d/modules.sh ]]; then
  . /etc/profile.d/modules.sh
fi
module purge >/dev/null 2>&1 || true
module load tools/prod
module load Python/3.12.3-GCCcore-13.3.0
module load Mesa/24.1.3-GCCcore-13.3.0

PROJECT_ROOT="$(cd "${PBS_O_WORKDIR:-/rds/general/user/aa6622/home/project}" && pwd)"
cd "${PROJECT_ROOT}"
mkdir -p logs/pbs/grpo

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

PYTHON_BIN="${GRPO_PYTHON:-/rds/general/user/aa6622/home/.envs/lerobot_mw_py312/bin/python}"
BASE_CKPT="${PHASE12_CHECKPOINT:-/rds/general/user/aa6622/home/.cache/huggingface/hub/models--jadechoghari--smolvla_metaworld/snapshots/ef3089ecb84eeeb7d33fedab24f6c76180a68900}"
RUN_DIR="${PHASE12_RUN_DIR:?set PHASE12_RUN_DIR to completed train artifact dir}"
SWEEP_NAME="${PHASE12_SWEEP_NAME:-eval_sweep_0000_0050_25ep_nenv25_async_topk}"
MIN_UPDATE="${PHASE12_MIN_UPDATE:-0}"
MAX_UPDATE="${PHASE12_MAX_UPDATE:-50}"
STRIDE="${PHASE12_STRIDE:-2}"
EPISODES="${PHASE12_EVAL_EPISODES:-25}"
N_ENVS="${PHASE12_EVAL_N_ENVS:-25}"
TOP_K="${PHASE12_TOP_K:-2}"
TOP_K_EPISODES="${PHASE12_TOP_K_EPISODES:-100}"

"${PYTHON_BIN}" scripts/grpo/eval_phase12_checkpoint_sweep.py \
  --base-checkpoint "${BASE_CKPT}" \
  --run-dir "${RUN_DIR}" \
  --task push-v3 \
  --episodes "${EPISODES}" \
  --eval-seed-start 1000 \
  --sweep-name "${SWEEP_NAME}" \
  --min-update "${MIN_UPDATE}" \
  --max-update "${MAX_UPDATE}" \
  --stride "${STRIDE}" \
  --execution-mode inprocess_vector \
  --n-envs "${N_ENVS}" \
  --rollout-execution vector_async \
  --max-steps 120 \
  --chunk-len 25 \
  --top-k "${TOP_K}" \
  --top-k-episodes "${TOP_K_EPISODES}"

test -f "${RUN_DIR}/${SWEEP_NAME}/eval_sweep_summary.json"
echo "PHASE12_EVAL_SWEEP_TOPK_DONE run_dir=${RUN_DIR} sweep=${SWEEP_NAME}"
```

- [ ] **Step 8: Run tests**

Run:

```bash
PYTHONPATH="${PWD}/src:${PYTHONPATH:-}" /rds/general/user/aa6622/home/.envs/lerobot_mw_py312/bin/python -m pytest tests/test_phase12_eval_sweep.py tests/test_phase12_pbs_static.py -q
```

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add scripts/grpo/eval_phase12_checkpoint_sweep.py scripts/grpo/phase12_eval_sweep_topk.pbs tests/test_phase12_eval_sweep.py tests/test_phase12_pbs_static.py
git commit -m "feat(phase12): add top-k eval sweeps"
```

---

## Task 11: PBS Experiment Chain Helper

**Files:**
- Create: `scripts/grpo/submit_phase12_pbs_chain.sh`
- Test: `tests/test_phase12_pbs_static.py`

- [ ] **Step 1: Write failing static test**

Append to `tests/test_phase12_pbs_static.py`:

```python
def test_phase12_pbs_chain_submits_five_jobs_and_afterok_evals() -> None:
    text = _read("scripts/grpo/submit_phase12_pbs_chain.sh")
    assert "qsub scripts/grpo/phase12_seedbatch_smoke_u2.pbs" in text
    assert "depend=afterok:${job_id}" in text
    assert "run_a_official_g8_lr1e5" in text
    assert "run_b_bounded_g8_lr1e5" in text
    assert "run_c_official_g16_lr5e6" in text
    assert "run_d_bounded_g16_lr5e6" in text
    assert "run_e_official_g16_lownoise" in text
```

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
PYTHONPATH="${PWD}/src:${PYTHONPATH:-}" /rds/general/user/aa6622/home/.envs/lerobot_mw_py312/bin/python -m pytest tests/test_phase12_pbs_static.py::test_phase12_pbs_chain_submits_five_jobs_and_afterok_evals -q
```

Expected: FAIL because chain helper does not exist.

- [ ] **Step 3: Add chain helper**

Create `scripts/grpo/submit_phase12_pbs_chain.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

submit_train_eval_pair() {
  local label="$1"
  local action_profile="$2"
  local group_size="$3"
  local lr="$4"
  local clip_eps="$5"
  local noise="$6"
  local out="${PROJECT_ROOT}/artifacts/phase12_wm_only_${label}"

  local job_id
  job_id="$(
    PHASE12_OUT="${out}" \
    PHASE12_ACTION_PROFILE="${action_profile}" \
    PHASE12_GROUP_SIZE="${group_size}" \
    PHASE12_LR="${lr}" \
    PHASE12_CLIP_EPS="${clip_eps}" \
    PHASE12_EULER_NOISE="${noise}" \
    qsub scripts/grpo/phase12_seedbatch_b4_g16_train_0000_0050.pbs
  )"
  echo "train ${label}: ${job_id}"

  local eval_id
  eval_id="$(
    PHASE12_RUN_DIR="${out}" \
    PHASE12_SWEEP_NAME="eval_sweep_0000_0050_25ep_nenv25_async_topk" \
    PHASE12_MAX_UPDATE=50 \
    PHASE12_STRIDE=2 \
    qsub -W depend=afterok:${job_id} scripts/grpo/phase12_eval_sweep_topk.pbs
  )"
  echo "eval ${label}: ${eval_id}"
}

echo "submit smoke"
qsub scripts/grpo/phase12_seedbatch_smoke_u2.pbs

submit_train_eval_pair run_a_official_g8_lr1e5 official_jepa_mirror 8 1e-5 0.2 0.2
submit_train_eval_pair run_b_bounded_g8_lr1e5 bounded_executed 8 1e-5 0.2 0.2
submit_train_eval_pair run_c_official_g16_lr5e6 official_jepa_mirror 16 5e-6 0.1 0.2
submit_train_eval_pair run_d_bounded_g16_lr5e6 bounded_executed 16 5e-6 0.1 0.2
submit_train_eval_pair run_e_official_g16_lownoise official_jepa_mirror 16 5e-6 0.1 0.1
```

- [ ] **Step 4: Make executable and run static test**

Run:

```bash
chmod +x scripts/grpo/submit_phase12_pbs_chain.sh
PYTHONPATH="${PWD}/src:${PYTHONPATH:-}" /rds/general/user/aa6622/home/.envs/lerobot_mw_py312/bin/python -m pytest tests/test_phase12_pbs_static.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/grpo/submit_phase12_pbs_chain.sh tests/test_phase12_pbs_static.py
git commit -m "chore(phase12): add PBS experiment chain"
```

---

## Task 12: Final Test Gate and Smoke Commands

**Files:**
- No source file changes unless a test reveals a defect.

- [ ] **Step 1: Run focused unit/static suite**

Run:

```bash
PYTHONPATH="${PWD}/src:${PYTHONPATH:-}" /rds/general/user/aa6622/home/.envs/lerobot_mw_py312/bin/python -m pytest \
  tests/test_grpo_math.py \
  tests/test_phase12_actions.py \
  tests/test_phase12_wm_reward.py \
  tests/test_phase12_rollout.py \
  tests/test_phase12_wm_only_rollout.py \
  tests/test_phase12_training_loop.py \
  tests/test_phase12_eval_sweep.py \
  tests/test_phase12_pbs_static.py \
  tests/test_phase12_slurm_static.py \
  -q
```

Expected: PASS.

- [ ] **Step 2: Run dry-run smoke on PBS script**

Run:

```bash
PHASE12_DRY_RUN=1 qsub scripts/grpo/phase12_seedbatch_smoke_u2.pbs
```

Expected: job completes with `PHASE12_WM_CHUNK_DRY_RUN_OK` in PBS output.

- [ ] **Step 3: Run real two-update smoke**

Run:

```bash
qsub scripts/grpo/phase12_seedbatch_smoke_u2.pbs
```

Expected: PBS output ends with `PHASE12_SEEDBATCH_SMOKE_DONE`, progress has two `update_complete` rows, `checkpoints/update_0002.pt` exists, and `process_tree_memory_summary.json` exists under output dir.

- [ ] **Step 4: If smoke passes, submit five-run chain**

Run:

```bash
scripts/grpo/submit_phase12_pbs_chain.sh
```

Expected: command prints one smoke job id, five train job ids, and five eval job ids.

---

## Experiment Matrix

Use this as first scheduled set after Task 12 smoke passes:

| Label | Action Profile | Batch | Group | LR | Clip | Noise | Updates | Eval |
|-------|----------------|-------|-------|----|------|-------|---------|------|
| `run_a_official_g8_lr1e5` | `official_jepa_mirror` | 4 | 8 | `1e-5` | `0.2` | `0.2` | 50 | 25ep sweep + top2 100ep |
| `run_b_bounded_g8_lr1e5` | `bounded_executed` | 4 | 8 | `1e-5` | `0.2` | `0.2` | 50 | 25ep sweep + top2 100ep |
| `run_c_official_g16_lr5e6` | `official_jepa_mirror` | 4 | 16 | `5e-6` | `0.1` | `0.2` | 50 | 25ep sweep + top2 100ep |
| `run_d_bounded_g16_lr5e6` | `bounded_executed` | 4 | 16 | `5e-6` | `0.1` | `0.2` | 50 | 25ep sweep + top2 100ep |
| `run_e_official_g16_lownoise` | `official_jepa_mirror` | 4 | 16 | `5e-6` | `0.1` | `0.1` | 50 | 25ep sweep + top2 100ep |

Avoid `G64`/`G128` until memory telemetry from `B4/G16` shows comfortable RSS. Phase11 R3 `G64` and pop128 runs showed GPU/host pressure; Phase12 carries JEPA-WM too, so first production grid stays conservative.

---

## Self-Review Checklist

| Requirement | Task |
|-------------|------|
| Keep Phase11 as real-env control | Architecture statement |
| Continue Phase12 as WM owner | Architecture + Tasks 1-4 |
| Add WM-only mode | Tasks 1, 3, 4 |
| Preserve bounded/unbounded options | Tasks 3, 9, 11 |
| Copy seed-batch semantics from Phase11 commits | Task 5 |
| Use per-group advantage norm | Task 5 |
| Fix explicit GRPO group-size normalizer | Task 2 |
| Add batched logprob contract | Task 6 |
| Add WM encode cache | Task 7 |
| Add Phase11-grade telemetry | Task 8 |
| Add PBS smoke/prod scripts | Task 9 |
| Add eval sweeps + top-k 100ep | Task 10 |
| Add five-run scheduler | Task 11 |
| Gate before scheduling | Task 12 |

Placeholder scan result: plan contains no `TBD`, no unfilled code blocks, and no open-ended implementation steps.
