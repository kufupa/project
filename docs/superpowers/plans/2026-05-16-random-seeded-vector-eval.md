# Random Seeded Vector Eval Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make MetaWorld reset layouts deterministic by seed everywhere and make Phase12/Phase50 vector eval safe when some environments finish before others.

**Architecture:** `random_seeded` becomes the default reset protocol in the adapter and every Phase50 PBS entry point exports it explicitly. Phase12 vector eval stops using SmolVLA's queued `select_action()` path for active-row batches and instead calls a queue-free helper that samples a fresh action chunk for the current active batch, then executes only the first action.

**Tech Stack:** Python 3.12, PyTorch, LeRobot SmolVLA, MetaWorld, Gymnasium vector envs, PBS, pytest.

---

## Current State

`random_seeded` is not done yet.

- `src/smolvla_grpo/lerobot_metaworld_adapter.py` currently defaults `SMOLVLA_METAWORLD_RESET_MODE` to `fixed` in both `DeferredLeRobotMetaworldEnv` and `OfficialLeRobotMetaWorldGRPORollout`.
- `scripts/eggroll/submit_phase50_eggroll_eval_sweep.pbs` explicitly exports `SMOLVLA_METAWORLD_RESET_MODE="${SMOLVLA_METAWORLD_RESET_MODE:-fixed}"`.
- `scripts/eggroll/submit_phase50_eggroll_smoke.pbs`, `scripts/eggroll/submit_phase50_eggroll_calibrate.pbs`, and `scripts/eggroll/submit_phase50_eggroll_train_from_calibration.pbs` do not export reset mode, so they inherit the adapter default.
- `src/smolvla_grpo/phase12_vector_eval.py` calls `bundle.policy.select_action(proc)` with a changing active batch size. SmolVLA queues chunked actions keyed only by time, not by surviving env row, so early success can leave a queued `(old_batch, action_dim)` action tensor that no longer matches `len(active_rows)`.

## File Structure

- Modify: `src/smolvla_grpo/lerobot_metaworld_adapter.py`
  - Responsibility: MetaWorld reset mode selection and private `_freeze_rand_vec` / `seeded_rand_vec` bridge.
- Modify: `src/smolvla_grpo/phase12_vector_eval.py`
  - Responsibility: in-process checkpoint eval with manual episode waves.
- Modify: `tests/test_grpo_lerobot_adapter.py`
  - Responsibility: adapter reset-mode unit coverage without importing real MuJoCo.
- Modify: `tests/test_phase12_vector_eval.py`
  - Responsibility: vector eval batching and early-finish regression tests.
- Modify: `scripts/eggroll/submit_phase50_eggroll_smoke.pbs`
  - Responsibility: GPU smoke entry point reset protocol.
- Modify: `scripts/eggroll/submit_phase50_eggroll_calibrate.pbs`
  - Responsibility: calibration entry point reset protocol.
- Modify: `scripts/eggroll/submit_phase50_eggroll_train_from_calibration.pbs`
  - Responsibility: overnight train entry point reset protocol.
- Modify: `scripts/eggroll/submit_phase50_eggroll_eval_sweep.pbs`
  - Responsibility: checkpoint eval entry point reset protocol.
- Modify: `tests/test_eggroll_pbs_static.py`
  - Responsibility: PBS static contract tests.

---

### Task 1: Adapter Default Becomes `random_seeded`

**Files:**
- Modify: `src/smolvla_grpo/lerobot_metaworld_adapter.py:87-92`
- Modify: `src/smolvla_grpo/lerobot_metaworld_adapter.py:428-433`
- Test: `tests/test_grpo_lerobot_adapter.py`

- [ ] **Step 1: Write failing adapter default tests**

Replace `test_deferred_metaworld_env_defaults_to_fixed_reset` with:

```python
def test_deferred_metaworld_env_defaults_to_random_seeded_reset(monkeypatch):
    _install_fake_deferred_deps(monkeypatch)
    from smolvla_grpo.lerobot_metaworld_adapter import DeferredLeRobotMetaworldEnv

    env = DeferredLeRobotMetaworldEnv(task="push-v3")
    try:
        env.reset(seed=2000)
        inner = env._env  # noqa: SLF001
        assert inner._freeze_rand_vec is False
        assert inner.seed_calls[-1] == 2000
        assert inner.seeded_rand_vec is True
    finally:
        env.close()
```

Replace `test_official_adapter_defaults_to_fixed_reset_for_vector_env` with:

```python
def test_official_adapter_defaults_to_random_seeded_reset_for_vector_env(monkeypatch):
    _install_fake_deferred_deps(monkeypatch)
    from smolvla_grpo.lerobot_metaworld_adapter import OfficialLeRobotMetaWorldGRPORollout

    rollout = OfficialLeRobotMetaWorldGRPORollout(task="push-v3", n_envs=3)
    try:
        obs = rollout.reset_many([2000, 2001, 2002])
        assert obs["pixels"].shape[0] == 3
        assert rollout.vec_env.num_envs == 3
        assert [env._env._freeze_rand_vec for env in rollout.vec_env.envs] == [False, False, False]  # noqa: SLF001
        assert [env._env.seed_calls[-1] for env in rollout.vec_env.envs] == [2000, 2001, 2002]  # noqa: SLF001
        assert [env._env.seeded_rand_vec for env in rollout.vec_env.envs] == [True, True, True]  # noqa: SLF001
    finally:
        rollout.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
PYTHONPATH=src /rds/general/user/aa6622/home/.envs/lerobot_mw_py312/bin/python -m pytest \
  tests/test_grpo_lerobot_adapter.py::test_deferred_metaworld_env_defaults_to_random_seeded_reset \
  tests/test_grpo_lerobot_adapter.py::test_official_adapter_defaults_to_random_seeded_reset_for_vector_env -q
```

Expected: FAIL because default is still `fixed`.

- [ ] **Step 3: Change adapter defaults**

In `src/smolvla_grpo/lerobot_metaworld_adapter.py`, change both default expressions from:

```python
reset_randomization_mode or os.environ.get("SMOLVLA_METAWORLD_RESET_MODE", "fixed")
```

to:

```python
reset_randomization_mode or os.environ.get("SMOLVLA_METAWORLD_RESET_MODE", "random_seeded")
```

There are two occurrences: one in `DeferredLeRobotMetaworldEnv.__init__`, one in `OfficialLeRobotMetaWorldGRPORollout.__init__`.

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
PYTHONPATH=src /rds/general/user/aa6622/home/.envs/lerobot_mw_py312/bin/python -m pytest \
  tests/test_grpo_lerobot_adapter.py::test_deferred_metaworld_env_defaults_to_random_seeded_reset \
  tests/test_grpo_lerobot_adapter.py::test_official_adapter_defaults_to_random_seeded_reset_for_vector_env -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/smolvla_grpo/lerobot_metaworld_adapter.py tests/test_grpo_lerobot_adapter.py
git commit -m "fix: default metaworld resets to seeded random"
```

---

### Task 2: PBS Scripts Export `random_seeded`

**Files:**
- Modify: `scripts/eggroll/submit_phase50_eggroll_smoke.pbs`
- Modify: `scripts/eggroll/submit_phase50_eggroll_calibrate.pbs`
- Modify: `scripts/eggroll/submit_phase50_eggroll_train_from_calibration.pbs`
- Modify: `scripts/eggroll/submit_phase50_eggroll_eval_sweep.pbs`
- Test: `tests/test_eggroll_pbs_static.py`

- [ ] **Step 1: Write failing PBS static tests**

In `tests/test_eggroll_pbs_static.py`, replace:

```python
assert 'SMOLVLA_METAWORLD_RESET_MODE="${SMOLVLA_METAWORLD_RESET_MODE:-fixed}"' in eval_text
```

with:

```python
for text in (calib, train, eval_text, _read("submit_phase50_eggroll_smoke.pbs")):
    assert 'SMOLVLA_METAWORLD_RESET_MODE="${SMOLVLA_METAWORLD_RESET_MODE:-random_seeded}"' in text
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
PYTHONPATH=src /rds/general/user/aa6622/home/.envs/lerobot_mw_py312/bin/python -m pytest \
  tests/test_eggroll_pbs_static.py::test_train_and_eval_contracts -q
```

Expected: FAIL because three scripts lack the export and eval still defaults to `fixed`.

- [ ] **Step 3: Add explicit reset mode export to all Phase50 PBS scripts**

In each of these files, immediately after `export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"`, add:

```bash
export SMOLVLA_METAWORLD_RESET_MODE="${SMOLVLA_METAWORLD_RESET_MODE:-random_seeded}"
```

Files:

```text
scripts/eggroll/submit_phase50_eggroll_smoke.pbs
scripts/eggroll/submit_phase50_eggroll_calibrate.pbs
scripts/eggroll/submit_phase50_eggroll_train_from_calibration.pbs
scripts/eggroll/submit_phase50_eggroll_eval_sweep.pbs
```

In `scripts/eggroll/submit_phase50_eggroll_eval_sweep.pbs`, replace the existing `fixed` export with the same `random_seeded` export.

- [ ] **Step 4: Run PBS static tests**

Run:

```bash
PYTHONPATH=src /rds/general/user/aa6622/home/.envs/lerobot_mw_py312/bin/python -m pytest tests/test_eggroll_pbs_static.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/eggroll/submit_phase50_eggroll_*.pbs tests/test_eggroll_pbs_static.py
git commit -m "fix: use seeded resets in eggroll pbs"
```

---

### Task 3: Queue-Free Action Helper For Vector Eval

**Files:**
- Modify: `src/smolvla_grpo/phase12_vector_eval.py:205-221`
- Modify: `src/smolvla_grpo/phase12_vector_eval.py:260-269`
- Test: `tests/test_phase12_vector_eval.py`

- [ ] **Step 1: Write failing queue regression test**

Append this test to `tests/test_phase12_vector_eval.py`:

```python
def test_vector_eval_uses_queue_free_action_when_active_rows_shrink(tmp_path, monkeypatch):
    import sys
    import types
    from collections import deque
    from types import SimpleNamespace

    import torch

    from smolvla_grpo.phase12_vector_eval import evaluate_loaded_policy_vectorized

    class FakePolicy:
        def __init__(self):
            self.reset_calls = 0
            self.select_action_calls = 0
            self.queue_free_calls = []
            self._queues = {"action": deque(maxlen=50)}

        def reset(self):
            self.reset_calls += 1
            self._queues["action"].clear()

        def select_action(self, proc):
            self.select_action_calls += 1
            batch = int(proc["observation.state"].shape[0])
            if not self._queues["action"]:
                for _ in range(50):
                    self._queues["action"].append(torch.zeros(batch, 4))
            return self._queues["action"].popleft()

    class FakeBundle:
        base_checkpoint = "base"
        grpo_checkpoint = None
        device = torch.device("cpu")

        def __init__(self):
            self.policy = FakePolicy()

        def postprocessor(self, action):
            return action

    class FakeEnv:
        action_dim = 4

        def __init__(self, *, task, n_envs=1):
            del task
            assert n_envs == 1
            self.seed = None
            self.steps = 0

        def reset(self, seed):
            self.seed = int(seed)
            self.steps = 0
            return {"state": np.array([float(seed), 0.0, 0.0, 0.0], dtype=np.float32)}

        def build_proc(self, obs, *, bundle):
            del bundle
            return {
                "observation.state": torch.tensor([[obs["state"][0], 0.0, 0.0, 0.0]], dtype=torch.float32),
                "task": [f"seed-{self.seed}"],
            }

        def step(self, action):
            self.steps += 1
            success = self.seed == 1001 and self.steps == 1
            return SimpleNamespace(
                observation={"state": np.array([float(self.seed), float(self.steps), 0.0, 0.0], dtype=np.float32)},
                reward=1.0,
                success=success,
                terminated=False,
                truncated=False,
            )

        def close(self):
            return None

    fake_adapter = types.ModuleType("smolvla_grpo.lerobot_metaworld_adapter")
    fake_adapter.OfficialLeRobotMetaWorldGRPORollout = FakeEnv
    fake_adapter.resolve_lerobot_horizon = lambda env, max_steps: int(max_steps)
    monkeypatch.setitem(sys.modules, "smolvla_grpo.lerobot_metaworld_adapter", fake_adapter)
    monkeypatch.setattr("smolvla_grpo.phase12_vector_eval._resolve_action_dim", lambda task: 4)

    def fake_queue_free(policy, proc):
        batch = int(proc["observation.state"].shape[0])
        policy.queue_free_calls.append(batch)
        return torch.zeros(batch, 4)

    monkeypatch.setattr("smolvla_grpo.phase12_vector_eval.select_eval_action_queue_free", fake_queue_free)

    bundle = FakeBundle()
    summary = evaluate_loaded_policy_vectorized(
        bundle=bundle,
        base_checkpoint="base",
        grpo_checkpoint=None,
        output_dir=tmp_path,
        task="push-v3",
        episodes=3,
        eval_seed_start=1000,
        n_envs=3,
        rollout_execution="vector_sync",
        max_steps=3,
    )

    assert summary["episodes"] == 3
    assert bundle.policy.select_action_calls == 0
    assert bundle.policy.queue_free_calls == [3, 2, 2]
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
PYTHONPATH=src /rds/general/user/aa6622/home/.envs/lerobot_mw_py312/bin/python -m pytest \
  tests/test_phase12_vector_eval.py::test_vector_eval_uses_queue_free_action_when_active_rows_shrink -q
```

Expected: FAIL with missing `select_eval_action_queue_free` or `select_action_calls == 1`.

- [ ] **Step 3: Add queue-free eval helper**

In `src/smolvla_grpo/phase12_vector_eval.py`, add this helper after `_reset_policy`:

```python
def select_eval_action_queue_free(policy: Any, proc: dict[str, Any]) -> torch.Tensor:
    """Return first eval action without using SmolVLA's cross-step action queue."""

    if all(hasattr(policy, name) for name in ("_prepare_batch", "prepare_images", "prepare_state")) and hasattr(
        getattr(policy, "model", None), "sample_actions"
    ):
        batch = policy._prepare_batch(proc)
        images, img_masks = policy.prepare_images(batch)
        state = policy.prepare_state(batch)
        lang_tokens = batch["observation.language.tokens"]
        lang_masks = batch["observation.language.attention_mask"]
        actions = policy.model.sample_actions(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            noise=None,
        )
        if not torch.is_tensor(actions):
            raise RuntimeError("SmolVLA sample_actions must return a tensor during vector eval")
        action_dim = int(policy.config.action_feature.shape[0])
        return actions[:, 0, :action_dim]

    return policy.select_action(proc)
```

- [ ] **Step 4: Use helper in vector eval loop**

In `evaluate_loaded_policy_vectorized`, replace:

```python
with torch.inference_mode():
    action = bundle.policy.select_action(proc)
    post = bundle.postprocessor(action)
```

with:

```python
with torch.inference_mode():
    action = select_eval_action_queue_free(bundle.policy, proc)
    post = bundle.postprocessor(action)
```

- [ ] **Step 5: Run queue regression test**

Run:

```bash
PYTHONPATH=src /rds/general/user/aa6622/home/.envs/lerobot_mw_py312/bin/python -m pytest \
  tests/test_phase12_vector_eval.py::test_vector_eval_uses_queue_free_action_when_active_rows_shrink -q
```

Expected: PASS.

- [ ] **Step 6: Run full vector eval tests**

Run:

```bash
PYTHONPATH=src /rds/general/user/aa6622/home/.envs/lerobot_mw_py312/bin/python -m pytest tests/test_phase12_vector_eval.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/smolvla_grpo/phase12_vector_eval.py tests/test_phase12_vector_eval.py
git commit -m "fix: avoid queued actions in vector eval"
```

---

### Task 4: Reset Mode Manifest Recording

**Files:**
- Modify: `src/smolvla_grpo/eggroll_trainer.py`
- Modify: `src/smolvla_grpo/phase12_vector_eval.py`
- Test: `tests/test_phase12_vector_eval.py`

- [ ] **Step 1: Add eval artifact test for reset mode**

In `tests/test_phase12_vector_eval.py`, inside `test_write_eval_artifacts_preserves_schema`, after:

```python
assert (tmp_path / "eval_info.json").exists()
```

add:

```python
info = json.loads((tmp_path / "eval_info.json").read_text(encoding="utf-8"))
assert "reset_randomization_mode" in info
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
PYTHONPATH=src /rds/general/user/aa6622/home/.envs/lerobot_mw_py312/bin/python -m pytest \
  tests/test_phase12_vector_eval.py::test_write_eval_artifacts_preserves_schema -q
```

Expected: FAIL because `eval_info.json` does not record reset mode yet.

- [ ] **Step 3: Record reset mode in eval artifacts**

In `src/smolvla_grpo/phase12_vector_eval.py`, import `os` if missing:

```python
import os
```

In `write_eval_artifacts`, add this key to `info`:

```python
"reset_randomization_mode": os.environ.get("SMOLVLA_METAWORLD_RESET_MODE", "random_seeded"),
```

- [ ] **Step 4: Record reset mode in EGGROLL train manifest**

In `src/smolvla_grpo/eggroll_trainer.py`, import `os` if missing:

```python
import os
```

When building the manifest config dict, add:

```python
"reset_randomization_mode": os.environ.get("SMOLVLA_METAWORLD_RESET_MODE", "random_seeded"),
```

- [ ] **Step 5: Run focused tests**

Run:

```bash
PYTHONPATH=src /rds/general/user/aa6622/home/.envs/lerobot_mw_py312/bin/python -m pytest \
  tests/test_phase12_vector_eval.py::test_write_eval_artifacts_preserves_schema \
  tests/test_phase12_vector_eval.py::test_vector_eval_uses_queue_free_action_when_active_rows_shrink -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/smolvla_grpo/eggroll_trainer.py src/smolvla_grpo/phase12_vector_eval.py tests/test_phase12_vector_eval.py
git commit -m "chore: record metaworld reset protocol"
```

---

### Task 5: End-To-End Verification

**Files:**
- Verify: all modified files

- [ ] **Step 1: Run static/unit suite for touched areas**

Run:

```bash
PYTHONPATH=src /rds/general/user/aa6622/home/.envs/lerobot_mw_py312/bin/python -m pytest \
  tests/test_grpo_lerobot_adapter.py \
  tests/test_phase12_vector_eval.py \
  tests/test_eggroll_pbs_static.py \
  tests/test_eggroll_rollout_static.py -q
```

Expected: PASS.

- [ ] **Step 2: Run live no-GPU reset determinism probe**

Run:

```bash
if [[ -f /etc/profile.d/modules.sh ]]; then . /etc/profile.d/modules.sh; fi
module load tools/prod >/dev/null 2>&1 || true
module load Mesa/24.1.3-GCCcore-13.3.0 >/dev/null 2>&1 || true
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa PYTHONPATH=src
SMOLVLA_METAWORLD_RESET_MODE=random_seeded /rds/general/user/aa6622/home/.envs/lerobot_mw_py312/bin/python - <<'PY'
import numpy as np
from smolvla_grpo.lerobot_metaworld_adapter import OfficialLeRobotMetaWorldGRPORollout

def vec_for(seed):
    env = OfficialLeRobotMetaWorldGRPORollout(task="push-v3", n_envs=1)
    try:
        env.reset(seed)
        return np.asarray(env.vec_env.envs[0]._env._last_rand_vec, dtype=float)
    finally:
        env.close()

a = vec_for(1000)
b = vec_for(1000)
c = vec_for(1001)
print("same_seed_same_layout", bool(np.allclose(a, b)))
print("different_seed_different_layout", bool(not np.allclose(a, c)))
PY
```

Expected:

```text
same_seed_same_layout True
different_seed_different_layout True
```

- [ ] **Step 3: Check lints in edited files**

Run Cursor lints for:

```text
src/smolvla_grpo/lerobot_metaworld_adapter.py
src/smolvla_grpo/phase12_vector_eval.py
src/smolvla_grpo/eggroll_trainer.py
tests/test_grpo_lerobot_adapter.py
tests/test_phase12_vector_eval.py
tests/test_eggroll_pbs_static.py
```

Expected: no new lint errors.

- [ ] **Step 4: Commit verification-only metadata if any test fixtures changed**

If no files changed in Task 5, do not commit.

If files changed, run:

```bash
git add <changed-files>
git commit -m "test: verify seeded vector eval protocol"
```

---

## Self-Review

Spec coverage:

- Fix `n_envs=3` early-success queued-action bug: Task 3.
- Set `random_seeded` as default everywhere: Tasks 1 and 2.
- Preserve deterministic seeded layouts across PBS jobs: Tasks 1, 2, and Task 5 live probe.
- Make future results auditable: Task 4 records reset mode in train/eval manifests.

Placeholder scan:

- No `TBD`.
- No `TODO`.
- No “similar to Task N”.
- All changed behavior has explicit code blocks and commands.

Type consistency:

- `select_eval_action_queue_free(policy: Any, proc: dict[str, Any]) -> torch.Tensor` matches usage in `evaluate_loaded_policy_vectorized`.
- `reset_randomization_mode` string values remain one of `fixed`, `random_seeded`, `random_unseeded`, `lerobot_default`.
- PBS export uses same env var name consumed by adapter: `SMOLVLA_METAWORLD_RESET_MODE`.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-16-random-seeded-vector-eval.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
