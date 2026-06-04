# Phase12 One-GPU Vector Eval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Speed up Phase12 checkpoint evaluation on one A16 GPU by keeping one SmolVLA model resident and evaluating multiple MetaWorld episodes through vectorized CPU env rows with batched GPU inference.

**Architecture:** Add a resident in-process evaluator to `eval_phase12_checkpoint_sweep.py`. It loads the base SmolVLA bundle once, swaps checkpoint policy weights per update, runs a manual pool of `N` official LeRobot MetaWorld envs with explicit per-env seeds, batches only active observations through the GPU policy, and writes the same eval artifacts as the current serial subprocess path.

**Tech Stack:** Python, PyTorch, LeRobot SmolVLA, Gymnasium vector envs, MetaWorld, Slurm `sbatch --export=NIL`, pytest.

---

## Critical Analysis Of Prior Plan

### Bugs And Bad Assumptions

- The prior plan trusted `OfficialLeRobotMetaWorldGRPORollout.reset(reset_seed)` for vector eval, but current implementation duplicates the same seed across all rows: `[reset_seed] * n_envs`. That would make a 4-env eval run 4 copies of the same episode seed instead of seeds `1000,1001,1002,1003`.

- The prior plan said to track an `active` mask while still stepping a vector env. Gymnasium vector envs have autoreset behavior (`NEXT_STEP`, `SAME_STEP`, or `DISABLED`) that is awkward for variable-length eval episodes. If we keep sending actions for inactive rows, completed envs can reset or assert depending on mode. Masked metrics do not prevent hidden env state and policy input drift.

- The prior plan proposed using `bundle.policy.select_action(proc)` directly. SmolVLA has an internal action queue. Batched `select_action` can work, but only if `n_action_steps=1` and `policy.reset()` happens at every checkpoint and every episode wave. Otherwise stale queued actions can cross checkpoint or episode boundaries.

- The prior plan loaded GRPO checkpoints with `strict=False` but did not validate missing/unexpected keys. In a resident model, loose loading can leave stale parameters from the previous checkpoint if a key is absent.

- A later draft reused `MetaWorldSmolVLAGRPOPolicy.sample_action_batch_from_proc()`. That is wrong for eval: it samples from the GRPO distribution path and can change behavior versus current deterministic serial eval. The vector path must mirror current eval with `bundle.policy.select_action(proc)` followed by `bundle.postprocessor(action)`.

- The prior plan did not preserve serial artifact semantics tightly enough. Downstream expects `eval_summary.json`, `eval_info.json`, `eval_episodes.jsonl`, and `episodes/episode_XXXX/*` with stable ordering and episode counts.

### Better Decision

Use a dedicated resident manual-env-pool eval harness, not four GPU processes and not a thin patch over the serial loop.

Core invariant:

```text
one checkpoint loaded -> one policy queue reset -> env pool with explicit seeds -> batch only active observations -> artifacts sorted by episode_index
```

Default first smoke should be `n_envs=4`, per user preference. If it fails or looks unstable, fall back to `n_envs=2`; if it passes but performance is unclear, optionally run a short `n_envs=2` comparison after the `n_envs=4` smoke.

## File Structure

- Modify `src/smolvla_grpo/lerobot_metaworld_adapter.py`
  - Add explicit per-row reset support for vector eval.
  - Keep current scalar `reset(int)` behavior compatible.

- Create `src/smolvla_grpo/phase12_vector_eval.py`
  - Own the resident one-GPU eval implementation.
  - Keep eval-specific logic out of the large sweep script.
  - Use the same deterministic policy path as current eval: `bundle.policy.select_action(proc)` followed by `bundle.postprocessor(action)`.
  - Avoid Gym vector autoreset during eval by using a manual pool of per-episode env handles and batching active observations only.

- Modify `scripts/grpo/eval_phase12_checkpoint_sweep.py`
  - Add CLI flags and dispatch to either current subprocess path or new resident vector path.
  - Keep current subprocess mode as default fallback.

- Modify `scripts/grpo/submit_phase12_eval_sweep.slurm`
  - Add one-GPU vector eval knobs while preserving `--export=NIL` and `common_env.sh`.

- Add `tests/test_phase12_vector_eval.py`
  - Unit tests for wave scheduling, seed mapping, output schema, and checkpoint load validation.

- Modify `tests/test_grpo_lerobot_adapter.py`
  - Cover vector reset with per-row seeds.

- Modify `tests/test_phase12_slurm_static.py`
  - Cover new Slurm flags and success markers.

## Task 1: Add Per-Row Vector Reset

**Files:**
- Modify: `src/smolvla_grpo/lerobot_metaworld_adapter.py`
- Modify: `tests/test_grpo_lerobot_adapter.py`

- [ ] **Step 1: Add failing test for per-row reset seeds**

Append to `tests/test_grpo_lerobot_adapter.py`:

```python
def test_official_adapter_reset_many_uses_per_row_seeds(monkeypatch):
    fake_vec = _install_fake_official_lerobot(monkeypatch, n_envs=3)
    from smolvla_grpo.lerobot_metaworld_adapter import OfficialLeRobotMetaWorldGRPORollout

    rollout = OfficialLeRobotMetaWorldGRPORollout(task="assembly-v3", n_envs=3)
    try:
        obs = rollout.reset_many([1000, 1001, 1002])
        assert fake_vec.reset_seeds == [1000, 1001, 1002]
        assert obs["pixels"].shape[0] == 3
    finally:
        rollout.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cd /vol/bitbucket/aa6622/project
PYTHONPATH=src /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python -m pytest \
  tests/test_grpo_lerobot_adapter.py::test_official_adapter_reset_many_uses_per_row_seeds -v
```

Expected: `AttributeError: 'OfficialLeRobotMetaWorldGRPORollout' object has no attribute 'reset_many'`.

- [ ] **Step 3: Implement `reset_many()`**

Add this method beside `reset()` in `OfficialLeRobotMetaWorldGRPORollout`:

```python
    def reset_many(self, reset_seeds: Sequence[int]) -> dict[str, Any]:
        seeds = [int(seed) for seed in reset_seeds]
        if len(seeds) != self.n_envs:
            raise ValueError(f"reset_many expected {self.n_envs} seeds; got {len(seeds)}")
        obs, _info = self.vec_env.reset(seed=seeds)
        return obs
```

Keep existing `reset()` unchanged for scalar callers:

```python
    def reset(self, reset_seed: int) -> dict[str, Any]:
        seeds = [int(reset_seed)] * self.n_envs
        obs, _info = self.vec_env.reset(seed=seeds)
        return obs
```

- [ ] **Step 4: Run adapter tests**

Run:

```bash
cd /vol/bitbucket/aa6622/project
PYTHONPATH=src /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python -m pytest \
  tests/test_grpo_lerobot_adapter.py::test_official_adapter_reset_many_uses_per_row_seeds \
  tests/test_grpo_lerobot_adapter.py::test_official_adapter_uses_make_env_and_vector_contract \
  tests/test_grpo_lerobot_adapter.py::test_official_adapter_step_batch_matches_n_envs \
  -v
```

Expected: all pass.

## Task 2: Create Resident Vector Eval Module

**Files:**
- Create: `src/smolvla_grpo/phase12_vector_eval.py`
- Add: `tests/test_phase12_vector_eval.py`

- [ ] **Step 1: Add tests for wave seed planning**

Create `tests/test_phase12_vector_eval.py` with:

```python
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


def test_episode_waves_use_distinct_eval_seeds() -> None:
    from smolvla_grpo.phase12_vector_eval import build_episode_waves

    waves = build_episode_waves(episodes=10, eval_seed_start=1000, n_envs=4)

    assert waves == [
        [(0, 1000), (1, 1001), (2, 1002), (3, 1003)],
        [(4, 1004), (5, 1005), (6, 1006), (7, 1007)],
        [(8, 1008), (9, 1009)],
    ]


def test_episode_waves_reject_bad_n_envs() -> None:
    from smolvla_grpo.phase12_vector_eval import build_episode_waves

    with pytest.raises(ValueError, match="n_envs must be >= 1"):
        build_episode_waves(episodes=10, eval_seed_start=1000, n_envs=0)
```

- [ ] **Step 2: Add minimal implementation**

Create `src/smolvla_grpo/phase12_vector_eval.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import torch

from smolvla_grpo.checkpointing import load_grpo_checkpoint
from smolvla_grpo.lerobot_metaworld_adapter import (
    OfficialLeRobotMetaWorldGRPORollout,
    resolve_lerobot_horizon,
)
from smolvla_grpo.phase11_rollout import load_bundle_for_grpo
from smolvla_pipeline.evaluator import write_episode_artifacts


@dataclass(frozen=True)
class EpisodeResult:
    episode_index: int
    reset_seed: int
    actions: list[list[float]]
    rewards: list[float]
    successes: list[bool]
    terminated: bool
    truncated: bool


def build_episode_waves(*, episodes: int, eval_seed_start: int, n_envs: int) -> list[list[tuple[int, int]]]:
    if int(n_envs) < 1:
        raise ValueError("n_envs must be >= 1")
    if int(episodes) < 1:
        raise ValueError("episodes must be >= 1")
    pairs = [(ep, int(eval_seed_start) + ep) for ep in range(int(episodes))]
    return [pairs[i : i + int(n_envs)] for i in range(0, len(pairs), int(n_envs))]
```

- [ ] **Step 3: Run seed tests**

Run:

```bash
cd /vol/bitbucket/aa6622/project
PYTHONPATH=src /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python -m pytest \
  tests/test_phase12_vector_eval.py::test_episode_waves_use_distinct_eval_seeds \
  tests/test_phase12_vector_eval.py::test_episode_waves_reject_bad_n_envs \
  -v
```

Expected: both pass.

## Task 3: Add Strict Checkpoint Loading

**Files:**
- Modify: `src/smolvla_grpo/phase12_vector_eval.py`
- Modify: `tests/test_phase12_vector_eval.py`

- [ ] **Step 1: Add missing-key validation tests**

Append to `tests/test_phase12_vector_eval.py`:

```python
class TinyPolicy:
    def __init__(self):
        self.loaded = None

    def state_dict(self):
        return {
            "linear.weight": object(),
            "linear.bias": object(),
        }

    def load_state_dict(self, state, strict=False):
        self.loaded = dict(state)
        missing = [k for k in self.state_dict() if k not in state]
        unexpected = [k for k in state if k not in self.state_dict()]
        return missing, unexpected


def test_validate_checkpoint_state_rejects_missing_non_whitelisted_key() -> None:
    from smolvla_grpo.phase12_vector_eval import validate_checkpoint_state

    policy = TinyPolicy()
    with pytest.raises(RuntimeError, match="missing checkpoint keys"):
        validate_checkpoint_state(policy, {"linear.weight": object()})


def test_validate_checkpoint_state_allows_log_std_missing_only() -> None:
    from smolvla_grpo.phase12_vector_eval import validate_checkpoint_state

    class PolicyWithLogStd(TinyPolicy):
        def state_dict(self):
            return {
                "linear.weight": object(),
                "model.log_std": object(),
            }

    validate_checkpoint_state(PolicyWithLogStd(), {"linear.weight": object()})
```

- [ ] **Step 2: Implement validation helper**

Add to `src/smolvla_grpo/phase12_vector_eval.py`:

```python
_ALLOWED_MISSING_KEYS = {"model.log_std"}


def _normalise_incompatible_keys(result: Any) -> tuple[list[str], list[str]]:
    if isinstance(result, tuple) and len(result) == 2:
        return list(result[0]), list(result[1])
    missing = list(getattr(result, "missing_keys", []))
    unexpected = list(getattr(result, "unexpected_keys", []))
    return missing, unexpected


def validate_checkpoint_state(policy: Any, state: dict[str, Any]) -> None:
    expected = set(policy.state_dict().keys())
    supplied = set(state.keys())
    missing = sorted(expected - supplied)
    unexpected = sorted(supplied - expected)
    bad_missing = [key for key in missing if key not in _ALLOWED_MISSING_KEYS]
    if bad_missing:
        raise RuntimeError(f"missing checkpoint keys: {bad_missing[:20]}")
    if unexpected:
        raise RuntimeError(f"unexpected checkpoint keys: {unexpected[:20]}")


def load_policy_checkpoint_into_bundle(bundle: Any, checkpoint_path: Path) -> dict[str, Any]:
    payload = load_grpo_checkpoint(checkpoint_path.expanduser().resolve(), map_location="cpu")
    state = payload["policy_state_dict"]
    validate_checkpoint_state(bundle.policy, state)
    result = bundle.policy.load_state_dict(state, strict=False)
    missing, unexpected = _normalise_incompatible_keys(result)
    bad_missing = [key for key in missing if key not in _ALLOWED_MISSING_KEYS]
    if bad_missing or unexpected:
        raise RuntimeError(f"checkpoint load mismatch missing={bad_missing[:20]} unexpected={unexpected[:20]}")
    bundle.policy.eval()
    reset = getattr(bundle.policy, "reset", None)
    if callable(reset):
        reset()
    return payload
```

- [ ] **Step 3: Run checkpoint validation tests**

Run:

```bash
cd /vol/bitbucket/aa6622/project
PYTHONPATH=src /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python -m pytest \
  tests/test_phase12_vector_eval.py::test_validate_checkpoint_state_rejects_missing_non_whitelisted_key \
  tests/test_phase12_vector_eval.py::test_validate_checkpoint_state_allows_log_std_missing_only \
  -v
```

Expected: both pass.

## Task 4: Implement Manual-Pool Batched Episode Evaluation

**Files:**
- Modify: `src/smolvla_grpo/phase12_vector_eval.py`
- Add: `tests/test_phase12_vector_eval.py`

- [ ] **Step 1: Add output summary test**

Append to `tests/test_phase12_vector_eval.py`:

```python
def test_write_eval_artifacts_preserves_schema(tmp_path, monkeypatch) -> None:
    from smolvla_grpo.phase12_vector_eval import EpisodeResult, write_eval_artifacts

    calls = []

    def fake_write_episode_artifacts(*, episode_dir, actions, rewards, successes, overlay_mode):
        calls.append((episode_dir.name, list(actions), list(rewards), list(successes), overlay_mode))
        episode_dir.mkdir(parents=True, exist_ok=True)
        (episode_dir / "actions.json").write_text("[]", encoding="utf-8")

    monkeypatch.setattr("smolvla_grpo.phase12_vector_eval.write_episode_artifacts", fake_write_episode_artifacts)

    results = [
        EpisodeResult(
            episode_index=1,
            reset_seed=1001,
            actions=[[0.0, 0.0, 0.0, 0.0]],
            rewards=[2.0],
            successes=[False],
            terminated=False,
            truncated=True,
        ),
        EpisodeResult(
            episode_index=0,
            reset_seed=1000,
            actions=[[1.0, 0.0, 0.0, 0.0]],
            rewards=[1.0],
            successes=[True],
            terminated=True,
            truncated=False,
        ),
    ]

    summary = write_eval_artifacts(
        base_checkpoint="base",
        grpo_checkpoint=Path("update_0010.pt"),
        output_dir=tmp_path,
        task="push-v3",
        episodes=2,
        eval_seed_start=1000,
        results=results,
    )

    assert summary["episodes"] == 2
    assert summary["pc_success"] == 50.0
    assert summary["avg_sum_reward"] == 1.5
    rows = [json.loads(line) for line in (tmp_path / "eval_episodes.jsonl").read_text().splitlines()]
    assert [row["episode_index"] for row in rows] == [0, 1]
    assert [row["reset_seed"] for row in rows] == [1000, 1001]
    assert (tmp_path / "eval_summary.json").exists()
    assert (tmp_path / "eval_info.json").exists()
    assert calls[0][0] == "episode_0000"
```

- [ ] **Step 2: Implement artifact writer**

Add to `src/smolvla_grpo/phase12_vector_eval.py`:

```python
def write_eval_artifacts(
    *,
    base_checkpoint: str,
    grpo_checkpoint: Path | None,
    output_dir: Path,
    task: str,
    episodes: int,
    eval_seed_start: int,
    results: list[EpisodeResult],
) -> dict[str, Any]:
    ordered = sorted(results, key=lambda item: item.episode_index)
    if len(ordered) != int(episodes):
        raise RuntimeError(f"expected {episodes} episode results; got {len(ordered)}")
    output_dir.mkdir(parents=True, exist_ok=True)

    sum_rewards = [float(sum(item.rewards)) for item in ordered]
    max_rewards = [float(max(item.rewards)) if item.rewards else 0.0 for item in ordered]
    success_flags = [any(bool(v) for v in item.successes) for item in ordered]
    pc_success = 100.0 * sum(1 for value in success_flags if value) / max(len(success_flags), 1)

    rows = []
    for item, sr, mr, ok in zip(ordered, sum_rewards, max_rewards, success_flags, strict=True):
        ep_dir = output_dir / "episodes" / f"episode_{item.episode_index:04d}"
        write_episode_artifacts(
            episode_dir=ep_dir,
            actions=item.actions,
            rewards=item.rewards,
            successes=item.successes,
            overlay_mode="cumulative_reward",
        )
        rows.append(
            {
                "episode_index": int(item.episode_index),
                "reset_seed": int(item.reset_seed),
                "sum_reward": float(sr),
                "max_reward": float(mr),
                "success": bool(ok),
                "n_steps": len(item.rewards),
                "env_backend": "official_lerobot",
            }
        )

    summary = {
        "grpo_checkpoint": str(grpo_checkpoint) if grpo_checkpoint is not None else "base_checkpoint",
        "base_checkpoint": base_checkpoint,
        "task": task,
        "env_backend": "official_lerobot",
        "eval_seed_start": int(eval_seed_start),
        "episodes": int(episodes),
        "avg_sum_reward": float(mean(sum_rewards)) if sum_rewards else 0.0,
        "avg_max_reward": float(mean(max_rewards)) if max_rewards else 0.0,
        "pc_success": float(pc_success),
    }
    (output_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "eval_episodes.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )
    eval_info = {
        "per_task": [
            {
                "task_group": task,
                "task_id": 0,
                "metrics": {
                    "sum_rewards": sum_rewards,
                    "max_rewards": max_rewards,
                    "successes": success_flags,
                    "video_paths": [],
                },
            }
        ],
        "per_group": {
            task: {
                "avg_sum_reward": summary["avg_sum_reward"],
                "avg_max_reward": summary["avg_max_reward"],
                "pc_success": summary["pc_success"],
                "n_episodes": len(sum_rewards),
                "video_paths": [],
            }
        },
        "overall": {
            "avg_sum_reward": summary["avg_sum_reward"],
            "avg_max_reward": summary["avg_max_reward"],
            "pc_success": summary["pc_success"],
            "n_episodes": len(sum_rewards),
            "video_paths": [],
        },
    }
    (output_dir / "eval_info.json").write_text(json.dumps(eval_info, indent=2), encoding="utf-8")
    return summary
```

- [ ] **Step 3: Run artifact test**

Run:

```bash
cd /vol/bitbucket/aa6622/project
PYTHONPATH=src /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python -m pytest \
  tests/test_phase12_vector_eval.py::test_write_eval_artifacts_preserves_schema -v
```

Expected: pass.

- [ ] **Step 4: Add batched eval action coercion**

Add to `src/smolvla_grpo/phase12_vector_eval.py`:

```python
def coerce_exec_action_batch(action: Any, *, action_dim: int, n_envs: int) -> np.ndarray:
    if hasattr(action, "detach"):
        action_np = action.detach().float().cpu().numpy()
    else:
        action_np = np.asarray(action, dtype=np.float32)
    action_np = np.asarray(action_np, dtype=np.float32)
    expected_size = int(n_envs) * int(action_dim)
    if action_np.size != expected_size:
        raise RuntimeError(
            f"Policy action dim mismatch: expected batch ({n_envs}, {action_dim}) "
            f"with {expected_size} values, got shape {tuple(action_np.shape)} and size {action_np.size}. "
            "Refusing silent pad/truncate."
        )
    return np.clip(action_np.reshape(int(n_envs), int(action_dim)), -1.0, 1.0).astype(np.float32, copy=False)
```

Do not use `MetaWorldSmolVLAGRPOPolicy.sample_action_batch_from_proc()` in eval. That helper samples from policy distribution for GRPO training and can change eval behavior relative to the current serial `policy.select_action()` path.

- [ ] **Step 5: Add action coercion test**

Append to `tests/test_phase12_vector_eval.py`:

```python
def test_coerce_exec_action_batch_preserves_rows() -> None:
    from smolvla_grpo.phase12_vector_eval import coerce_exec_action_batch

    action = np.array([[2.0, 0.5, -0.5, -2.0], [0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
    out = coerce_exec_action_batch(action, action_dim=4, n_envs=2)

    assert out.shape == (2, 4)
    np.testing.assert_allclose(out[0], np.array([1.0, 0.5, -0.5, -1.0], dtype=np.float32))
    np.testing.assert_allclose(out[1], np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32))
```

- [ ] **Step 6: Implement vector evaluator loop**

Add to `src/smolvla_grpo/phase12_vector_eval.py`:

```python
def _reset_policy(policy: Any) -> None:
    reset = getattr(policy, "reset", None)
    if callable(reset):
        reset()


def evaluate_loaded_policy_vectorized(
    *,
    bundle: Any,
    base_checkpoint: str,
    grpo_checkpoint: Path | None,
    output_dir: Path,
    task: str,
    episodes: int,
    eval_seed_start: int,
    n_envs: int,
    rollout_execution: str,
    max_steps: int,
) -> dict[str, Any]:
    if rollout_execution != "vector_sync":
        raise ValueError("manual-pool eval currently supports rollout_execution='vector_sync' only")
    action_dim = _resolve_action_dim(task)
    waves = build_episode_waves(episodes=episodes, eval_seed_start=eval_seed_start, n_envs=n_envs)
    all_results: list[EpisodeResult] = []

    for wave in waves:
        wave_n = len(wave)
        envs = [OfficialLeRobotMetaWorldGRPORollout(task=task, n_envs=1) for _ in range(wave_n)]
        try:
            resolved_steps = resolve_lerobot_horizon(envs[0], max_steps)
            obs_by_row = [env.reset(seed) for env, (_ep, seed) in zip(envs, wave, strict=True)]
            _reset_policy(bundle.policy)
            active = np.ones((wave_n,), dtype=np.bool_)
            actions: list[list[list[float]]] = [[] for _ in range(wave_n)]
            rewards: list[list[float]] = [[] for _ in range(wave_n)]
            successes: list[list[bool]] = [[] for _ in range(wave_n)]
            terminated = [False for _ in range(wave_n)]
            truncated = [False for _ in range(wave_n)]

            for _step in range(int(resolved_steps)):
                if not bool(np.any(active)):
                    break
                active_rows = [idx for idx in range(wave_n) if bool(active[idx])]
                proc_rows = [envs[idx].build_proc(obs_by_row[idx], bundle=bundle) for idx in active_rows]
                proc = concatenate_proc_rows(proc_rows)
                with torch.inference_mode():
                    action = bundle.policy.select_action(proc)
                    post = bundle.postprocessor(action)
                exec_action_np = coerce_exec_action_batch(post, action_dim=action_dim, n_envs=len(active_rows))

                for batch_row, row in enumerate(active_rows):
                    step = envs[row].step(exec_action_np[batch_row : batch_row + 1])
                    obs_by_row[row] = step.observation
                    actions[row].append(exec_action_np[batch_row].reshape(-1).tolist())
                    rewards[row].append(float(step.reward))
                    successes[row].append(bool(step.success))
                    if step.success or step.terminated or step.truncated:
                        active[row] = False
                        terminated[row] = bool(step.terminated)
                        truncated[row] = bool(step.truncated)

            for row, (episode_index, reset_seed) in enumerate(wave):
                all_results.append(
                    EpisodeResult(
                        episode_index=int(episode_index),
                        reset_seed=int(reset_seed),
                        actions=actions[row],
                        rewards=rewards[row],
                        successes=successes[row],
                        terminated=terminated[row],
                        truncated=truncated[row] or (len(rewards[row]) >= int(resolved_steps) and not any(successes[row])),
                    )
                )
        finally:
            for env in envs:
                env.close()

    return write_eval_artifacts(
        base_checkpoint=base_checkpoint,
        grpo_checkpoint=grpo_checkpoint,
        output_dir=output_dir,
        task=task,
        episodes=episodes,
        eval_seed_start=eval_seed_start,
        results=all_results,
    )
```

Then add `concatenate_proc_rows()` and `_resolve_action_dim()`:

```python
def concatenate_proc_rows(proc_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not proc_rows:
        raise ValueError("proc_rows must be non-empty")
    out: dict[str, Any] = {}
    keys = proc_rows[0].keys()
    for key in keys:
        vals = [row[key] for row in proc_rows]
        first = vals[0]
        if torch.is_tensor(first):
            out[key] = torch.cat(vals, dim=0)
        elif isinstance(first, np.ndarray):
            out[key] = np.concatenate(vals, axis=0)
        elif isinstance(first, list):
            merged: list[Any] = []
            for value in vals:
                merged.extend(value)
            out[key] = merged
        else:
            out[key] = vals
    return out
```

```python
def _resolve_action_dim(task: str) -> int:
    probe = OfficialLeRobotMetaWorldGRPORollout(task=task, n_envs=1)
    try:
        return int(probe.action_dim)
    finally:
        probe.close()
```

Note: this deliberately avoids Gym vector env autoreset. Env stepping is serial inside the process, but GPU inference is batched over active rows and model loading is resident. If CPU stepping remains bottleneck after correctness passes, add a `ThreadPoolExecutor` around per-row `env.step()` as a separate optimization.

## Task 5: Wire Resident Vector Sweep

**Files:**
- Modify: `scripts/grpo/eval_phase12_checkpoint_sweep.py`
- Modify: `tests/test_phase12_vector_eval.py`

- [ ] **Step 1: Add sweep dispatch tests**

Append to `tests/test_phase12_vector_eval.py`:

```python
import importlib.util


def _load_phase12_sweep_module():
    repo = Path(__file__).resolve().parents[1]
    path = repo / "scripts" / "grpo" / "eval_phase12_checkpoint_sweep.py"
    spec = importlib.util.spec_from_file_location("eval_phase12_checkpoint_sweep", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_phase12_vector_sweep_uses_resident_eval(tmp_path, monkeypatch) -> None:
    mod = _load_phase12_sweep_module()
    run_dir = tmp_path / "run"
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "checkpoints" / "update_0010.pt").write_bytes(b"fake")
    calls = []

    def fake_run_vector_sweep(**kwargs):
        calls.append(kwargs)
        sweep_dir = kwargs["run_dir"] / kwargs["sweep_name"]
        update_dir = sweep_dir / "update_0010"
        update_dir.mkdir(parents=True)
        (update_dir / "eval_summary.json").write_text(
            json.dumps({"pc_success": 0.0, "avg_sum_reward": 1.0, "avg_max_reward": 2.0, "episodes": 4}),
            encoding="utf-8",
        )
        result = {
            "task": kwargs["task"],
            "episodes": kwargs["episodes"],
            "eval_seed_start": kwargs["eval_seed_start"],
            "sweep_name": kwargs["sweep_name"],
            "min_update": kwargs["min_update"],
            "max_update": kwargs["max_update"],
            "stride": kwargs["stride"],
            "rows": [
                {
                    "update": 10,
                    "checkpoint": str(run_dir / "checkpoints" / "update_0010.pt"),
                    "pc_success": 0.0,
                    "avg_sum_reward": 1.0,
                    "avg_max_reward": 2.0,
                    "episodes": 4,
                    "eval_summary_path": str(update_dir / "eval_summary.json"),
                }
            ],
        }
        (sweep_dir / "eval_sweep_summary.json").write_text(json.dumps(result), encoding="utf-8")
        return result

    monkeypatch.setattr(mod, "run_sweep_inprocess_vector", fake_run_vector_sweep)
    result = mod.run_sweep(
        base_checkpoint="base",
        run_dir=run_dir,
        task="push-v3",
        episodes=4,
        eval_seed_start=1000,
        sweep_name="vec",
        min_update=10,
        max_update=10,
        stride=10,
        execution_mode="inprocess_vector",
        n_envs=2,
        rollout_execution="vector_sync",
        max_steps=5,
    )

    assert len(result["rows"]) == 1
    assert calls[0]["n_envs"] == 2
    assert calls[0]["rollout_execution"] == "vector_sync"
```

- [ ] **Step 2: Extend `run_sweep()` signature**

In `scripts/grpo/eval_phase12_checkpoint_sweep.py`, change `run_sweep()` to accept:

```python
    execution_mode: str = "subprocess",
    n_envs: int = 1,
    rollout_execution: str = "serial",
    max_steps: int | None = None,
```

At the top of `run_sweep()`, after checkpoint existence validation:

```python
    if execution_mode == "inprocess_vector":
        return run_sweep_inprocess_vector(
            base_checkpoint=base_checkpoint,
            run_dir=run_dir,
            task=task,
            episodes=episodes,
            eval_seed_start=eval_seed_start,
            sweep_name=sweep_name,
            min_update=min_update,
            max_update=max_update,
            stride=stride,
            n_envs=n_envs,
            rollout_execution=rollout_execution,
            max_steps=max_steps,
        )
    if execution_mode != "subprocess":
        raise ValueError("execution_mode must be 'subprocess' or 'inprocess_vector'")
```

- [ ] **Step 3: Add in-process vector sweep function**

In `scripts/grpo/eval_phase12_checkpoint_sweep.py`, import:

```python
from smolvla_grpo.phase12_vector_eval import (
    evaluate_loaded_policy_vectorized,
    load_policy_checkpoint_into_bundle,
)
```

Add function:

```python
def run_sweep_inprocess_vector(
    *,
    base_checkpoint: str,
    run_dir: Path,
    task: str,
    episodes: int,
    eval_seed_start: int,
    sweep_name: str,
    min_update: int,
    max_update: int,
    stride: int,
    n_envs: int,
    rollout_execution: str,
    max_steps: int | None,
) -> dict[str, Any]:
    from smolvla_grpo.phase11_rollout import load_bundle_for_grpo
    from smolvla_pipeline.evaluator import _resolve_max_steps

    resolved_max_steps = _resolve_max_steps() if max_steps is None else int(max_steps)
    bundle, _action_dim = load_bundle_for_grpo(
        base_checkpoint,
        task=task,
        env_backend="official_lerobot",
        n_action_steps=1,
    )
    sweep_dir = run_dir / sweep_name
    sweep_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    updates = list(range(int(min_update), int(max_update) + 1, int(stride)))

    for update in updates:
        out_dir = sweep_dir / f"update_{update:04d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        ckpt = _make_base_eval_checkpoint(base_checkpoint, out_dir, task=task) if update == 0 else _checkpoint_path(run_dir, update)
        load_policy_checkpoint_into_bundle(bundle, ckpt)
        summary = evaluate_loaded_policy_vectorized(
            bundle=bundle,
            base_checkpoint=base_checkpoint,
            grpo_checkpoint=ckpt,
            output_dir=out_dir,
            task=task,
            episodes=episodes,
            eval_seed_start=eval_seed_start,
            n_envs=n_envs,
            rollout_execution=rollout_execution,
            max_steps=resolved_max_steps,
        )
        rows.append(
            {
                "update": int(update),
                "checkpoint": "base_checkpoint" if update == 0 else str(ckpt),
                "pc_success": float(summary.get("pc_success", 0.0)),
                "avg_sum_reward": float(summary.get("avg_sum_reward", 0.0)),
                "avg_max_reward": float(summary.get("avg_max_reward", 0.0)),
                "episodes": int(summary.get("episodes", episodes)),
                "eval_summary_path": str(out_dir / "eval_summary.json"),
            }
        )

    result = {
        "task": task,
        "episodes": int(episodes),
        "eval_seed_start": int(eval_seed_start),
        "sweep_name": sweep_name,
        "min_update": int(min_update),
        "max_update": int(max_update),
        "stride": int(stride),
        "execution_mode": "inprocess_vector",
        "n_envs": int(n_envs),
        "rollout_execution": rollout_execution,
        "rows": rows,
    }
    out_path = sweep_dir / "eval_sweep_summary.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["output_path"] = str(out_path)
    return result
```

- [ ] **Step 4: Add CLI args**

In `main()` parser:

```python
    parser.add_argument("--execution-mode", choices=("subprocess", "inprocess_vector"), default="subprocess")
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--rollout-execution", choices=("serial", "vector_sync"), default="serial")
    parser.add_argument("--max-steps", type=int, default=None)
```

Pass these into `run_sweep()`.

- [ ] **Step 5: Run sweep dispatch test**

Run:

```bash
cd /vol/bitbucket/aa6622/project
PYTHONPATH=src /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python -m pytest \
  tests/test_phase12_vector_eval.py::test_phase12_vector_sweep_uses_resident_eval -v
```

Expected: pass.

## Task 6: Update Slurm Wrapper

**Files:**
- Modify: `scripts/grpo/submit_phase12_eval_sweep.slurm`
- Modify: `tests/test_phase12_slurm_static.py`

- [ ] **Step 1: Add Slurm static test**

Append to `tests/test_phase12_slurm_static.py`:

```python
def test_eval_sweep_slurm_supports_one_gpu_vector_mode() -> None:
    text = _read("submit_phase12_eval_sweep.slurm")

    assert "#SBATCH --gres=gpu:1" in text
    assert "#SBATCH --export=NIL" in text
    assert 'slurm_resolve_project_root "scripts/grpo/eval_phase12_checkpoint_sweep.py"' in text
    assert 'slurm_export_hf_torch_cache "phase12-eval-sweep"' in text
    assert 'EXECUTION_MODE="${PHASE12_EVAL_EXECUTION_MODE:-subprocess}"' in text
    assert 'N_ENVS="${PHASE12_EVAL_N_ENVS:-1}"' in text
    assert 'ROLLOUT_EXECUTION="${PHASE12_EVAL_ROLLOUT_EXECUTION:-serial}"' in text
    assert '--execution-mode "${EXECUTION_MODE}"' in text
    assert '--n-envs "${N_ENVS}"' in text
    assert '--rollout-execution "${ROLLOUT_EXECUTION}"' in text
    assert "PHASE12_EVAL_SWEEP_DONE" in text
    assert "sbatch" not in _body_without_sbatch_header(text)
```

- [ ] **Step 2: Modify Slurm script**

In `scripts/grpo/submit_phase12_eval_sweep.slurm`, add:

```bash
EXECUTION_MODE="${PHASE12_EVAL_EXECUTION_MODE:-subprocess}"
N_ENVS="${PHASE12_EVAL_N_ENVS:-1}"
ROLLOUT_EXECUTION="${PHASE12_EVAL_ROLLOUT_EXECUTION:-serial}"
MAX_STEPS="${PHASE12_EVAL_MAX_STEPS:-}"
```

Add echo lines:

```bash
echo "[phase12-eval-sweep] execution_mode=${EXECUTION_MODE}"
echo "[phase12-eval-sweep] n_envs=${N_ENVS}"
echo "[phase12-eval-sweep] rollout_execution=${ROLLOUT_EXECUTION}"
echo "[phase12-eval-sweep] max_steps=${MAX_STEPS:-<default>}"
```

Build args array before Python call:

```bash
EXTRA_ARGS=(
  --execution-mode "${EXECUTION_MODE}"
  --n-envs "${N_ENVS}"
  --rollout-execution "${ROLLOUT_EXECUTION}"
)
if [[ -n "${MAX_STEPS}" ]]; then
  EXTRA_ARGS+=(--max-steps "${MAX_STEPS}")
fi
```

Append `"${EXTRA_ARGS[@]}"` to the Python call.

- [ ] **Step 3: Run Slurm static and syntax tests**

Run:

```bash
cd /vol/bitbucket/aa6622/project
PYTHONPATH=src /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python -m pytest \
  tests/test_phase12_slurm_static.py::test_eval_sweep_slurm_supports_one_gpu_vector_mode -v
bash -n scripts/grpo/submit_phase12_eval_sweep.slurm
```

Expected: pytest pass and `bash -n` exit 0.

## Task 7: Run Local Test Suite Slice

**Files:**
- No edits.

- [ ] **Step 1: Run focused unit/static tests**

Run:

```bash
cd /vol/bitbucket/aa6622/project
PYTHONPATH=src /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python -m pytest \
  tests/test_phase12_vector_eval.py \
  tests/test_grpo_lerobot_adapter.py::test_official_adapter_reset_many_uses_per_row_seeds \
  tests/test_grpo_lerobot_adapter.py::test_official_adapter_step_batch_matches_n_envs \
  tests/test_phase12_slurm_static.py::test_eval_sweep_slurm_supports_one_gpu_vector_mode \
  -v
```

Expected: all pass.

- [ ] **Step 2: Run regression tests for existing eval/sweep behavior**

Run:

```bash
cd /vol/bitbucket/aa6622/project
PYTHONPATH=src /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python -m pytest \
  tests/test_phase111_eval_sweep.py \
  tests/test_grpo_lerobot_adapter.py::test_grpo_eval_can_write_official_eval_info \
  tests/test_grpo_lerobot_adapter.py::test_official_eval_path_matches_rollout_semantics_static \
  -v
```

Expected: all pass.

## Task 8: One-GPU Smoke Gates

**Files:**
- No edits unless smoke reveals bug.

- [ ] **Step 1: Slurm test-only for vector mode**

Run from login node, not inside a Slurm job:

```bash
cd /vol/bitbucket/aa6622/project
sbatch --test-only --chdir=/vol/bitbucket/aa6622/project --export=NIL \
  scripts/grpo/submit_phase12_eval_sweep.slurm \
  /vol/bitbucket/aa6622/project/artifacts/phase12_wm_chunk_grpo_train/push-v3/wm_view_fix_bounded_u100_seed2000 \
  bounded_eval_vec_smoke_u100_vec4 \
  100 100 10 4 1000
```

Expected: Slurm accepts script and prints job test info.

- [ ] **Step 2: Submit 4-episode `n_envs=4` smoke first**

Run:

```bash
cd /vol/bitbucket/aa6622/project
jid=$(sbatch --parsable --chdir=/vol/bitbucket/aa6622/project \
  --export=NIL,PHASE12_EVAL_EXECUTION_MODE=inprocess_vector,PHASE12_EVAL_N_ENVS=4,PHASE12_EVAL_ROLLOUT_EXECUTION=vector_sync \
  scripts/grpo/submit_phase12_eval_sweep.slurm \
  /vol/bitbucket/aa6622/project/artifacts/phase12_wm_chunk_grpo_train/push-v3/wm_view_fix_bounded_u100_seed2000 \
  bounded_eval_vec_smoke_u100_vec4 \
  100 100 10 4 1000)
echo "$jid"
```

Expected:

- stdout has `phase12_eval_sweep_ok rows=1`.
- stdout has `PHASE12_EVAL_SWEEP_DONE`.
- `bounded_eval_vec_smoke_u100_vec4/update_0100/eval_episodes.jsonl` has exactly 4 rows.
- row seeds are `1000`, `1001`, `1002`, `1003`.
- `nvidia-smi` shows one Python process, not multiple model processes.
- VRAM below A16 limit.
- no OOM, no action shape mismatch.

- [ ] **Step 3: Optional 2-episode `n_envs=2` comparison/fallback smoke**

Run only if `n_envs=4` fails, looks unstable, or needs a speed comparison:

```bash
cd /vol/bitbucket/aa6622/project
jid=$(sbatch --parsable --chdir=/vol/bitbucket/aa6622/project \
  --export=NIL,PHASE12_EVAL_EXECUTION_MODE=inprocess_vector,PHASE12_EVAL_N_ENVS=2,PHASE12_EVAL_ROLLOUT_EXECUTION=vector_sync \
  scripts/grpo/submit_phase12_eval_sweep.slurm \
  /vol/bitbucket/aa6622/project/artifacts/phase12_wm_chunk_grpo_train/push-v3/wm_view_fix_bounded_u100_seed2000 \
  bounded_eval_vec_smoke_u100_vec2 \
  100 100 10 2 1000)
echo "$jid"
```

Expected:

- 2 eval rows.
- seeds `1000` and `1001`.
- lower VRAM than `n_envs=4`.
- use only as fallback or benchmark comparison.

## Task 9: Production Run

**Files:**
- No edits.

- [ ] **Step 1: Choose production `n_envs`**

Use:

- `n_envs=4` if smoke VRAM is below A16 limit and artifacts are correct.
- `n_envs=2` only if `n_envs=4` is unstable, OOM-prone, or clearly slower after optional comparison.

- [ ] **Step 2: Submit update `100..300` sweep**

For `n_envs=4`:

```bash
cd /vol/bitbucket/aa6622/project
jid=$(sbatch --parsable --chdir=/vol/bitbucket/aa6622/project \
  --export=NIL,PHASE12_EVAL_EXECUTION_MODE=inprocess_vector,PHASE12_EVAL_N_ENVS=4,PHASE12_EVAL_ROLLOUT_EXECUTION=vector_sync \
  scripts/grpo/submit_phase12_eval_sweep.slurm \
  /vol/bitbucket/aa6622/project/artifacts/phase12_wm_chunk_grpo_train/push-v3/wm_view_fix_bounded_u100_seed2000 \
  bounded_eval_every10_u100_u300_onegpu_vec4 \
  100 300 10 10 1000)
echo "$jid"
```

For `n_envs=2`, change `PHASE12_EVAL_N_ENVS=2` and sweep name to `bounded_eval_every10_u100_u300_onegpu_vec2`.

- [ ] **Step 3: Monitor**

Run:

```bash
squeue -j "$jid" -o "%.18i %.9P %.30j %.8T %.10M %.6D %R"
```

Use overlap resource check only after job starts:

```bash
srun --jobid="$jid" --overlap bash -lc 'nvidia-smi; ps -o pid,pcpu,pmem,rss,vsz,cmd -u "$USER" | head -20; free -h'
```

Expected:

- one Python PID owns GPU.
- VRAM remains below A16 limit.
- CPU uses multiple cores during vector stepping.
- output dirs appear for `update_0100`, `update_0110`, etc.

- [ ] **Step 4: Verify final output**

Run:

```bash
cd /vol/bitbucket/aa6622/project
/vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python - <<'PY'
import json
from pathlib import Path

run_dir = Path("/vol/bitbucket/aa6622/project/artifacts/phase12_wm_chunk_grpo_train/push-v3/wm_view_fix_bounded_u100_seed2000")
sweep = run_dir / "bounded_eval_every10_u100_u300_onegpu_vec4"
summary = json.loads((sweep / "eval_sweep_summary.json").read_text())
updates = [row["update"] for row in summary["rows"]]
assert updates == list(range(100, 301, 10)), updates
assert all(int(row["episodes"]) == 10 for row in summary["rows"])
for update in updates:
    ep_path = sweep / f"update_{update:04d}" / "eval_episodes.jsonl"
    rows = [json.loads(line) for line in ep_path.read_text().splitlines()]
    assert len(rows) == 10, (update, len(rows))
    assert [row["reset_seed"] for row in rows] == list(range(1000, 1010)), update
print("phase12_onegpu_vector_eval_verified", len(updates), str(sweep))
PY
```

Expected: prints `phase12_onegpu_vector_eval_verified 21 ...`.

## Fallback Plan

- If strict checkpoint validation rejects real checkpoints because only `model.log_std` is missing, keep whitelist as-is.
- If additional harmless missing keys appear, stop and inspect before expanding whitelist.
- If manual env-pool batching causes action-shape or processor issues, reduce to `n_envs=1` resident mode. This still saves repeated model loads.
- If `vector_sync` is correct but CPU stepping is still the bottleneck, add a later `ThreadPoolExecutor` env-step experiment as a separate optimization.
- If resident mode fails late, switch back to current `subprocess` mode with the same Slurm script by setting `PHASE12_EVAL_EXECUTION_MODE=subprocess`.

## Self-Review

- Spec coverage: one-GPU utilization, seed correctness, checkpoint loading, artifact parity, Slurm safety, smoke gates, and production run are covered.
- Placeholder scan: no placeholders remain; each task has exact files, code, and commands.
- Type consistency: names are consistent across plan: `reset_many`, `build_episode_waves`, `EpisodeResult`, `validate_checkpoint_state`, `load_policy_checkpoint_into_bundle`, `evaluate_loaded_policy_vectorized`, and `run_sweep_inprocess_vector`.

