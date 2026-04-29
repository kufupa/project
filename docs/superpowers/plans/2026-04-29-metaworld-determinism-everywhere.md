# MetaWorld Determinism Everywhere Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every known Python MetaWorld reset path in `/vol/bitbucket/aa6622/project` follow the deterministic contract: seed global RNGs, set MT task when applicable, then call strict `reset(seed=...)`.

**Architecture:** Keep `src/metaworld_determinism.py` as the single source of truth. Do not add a `metaworld_episode_reset()` helper that seeds inside reset, because that hides and can break the required `seed -> set_task -> reset` ordering. Fix only paths that still call `env.reset(seed=...)` without first seeding globals; leave oracle and segment GRPO behavior intact except for verification because they already match the order.

**Tech Stack:** Python 3.10+, pytest, NumPy, PyTorch, MetaWorld, Gymnasium.

---

## Critical Review Of Previous Plan

- Previous plan’s biggest bug: `metaworld_episode_reset(env, seed)` would seed after `set_task` in existing callers. That violates current module contract and risks changing oracle/SmolVLA initial states if `set_task` touches RNG.
- Previous plan duplicated SmolVLA action clipping in `_step`, but `_coerce_exec_action()` already clips. Best fix: no extra clip there unless replacing existing clip with a shared helper in a separate refactor.
- Previous plan over-touched oracle and segment GRPO. Oracle already does `seed -> set_task -> gymnasium_reset_strict`; segment GRPO seeds before env/task setup and uses strict reset. Treat them as reference paths.
- Previous grep gate was too broad. It would flag valid tests and `gymnasium_reset_strict` itself. Gate must target unsafe raw resets outside approved helper/test locations.
- Previous vendor/smoke plan needed clearer import strategy. `mt10/verify_env.py` currently says “No project src imports”; changing that must update the docstring and fail loudly if project helper cannot import.

## File Map

- Modify: `src/smolvla_pipeline/evaluator.py`
  Production bug. Add explicit `seed_metaworld_process(reset_seed)` before task selection, then use `gymnasium_reset_strict` inside `_reset`.
- Modify: `tests/test_smolvla_eval_artifacts.py`
  Add fake-backend tests proving SmolVLA order and strict reset delegation without importing real MetaWorld.
- Modify: `tests/test_metaworld_jepa_render.py`
  Seed globals before `set_task`, then use strict reset.
- Modify: `mt10/verify_env.py`
  Smoke script. Import project helper, update docstring, seed before MT1 task/reset and MT10 vector reset.
- Modify: `vendor/pi05/jepa_cem_paired_pushv3_export.py`
  Vendor-ish exporter still in repo. Add project helper import, seed before initial task selection, and seed before each strict reset. Remove unseeded `env.reset()` fallback.
- No change intended: `scripts/oracle/run_metaworld_oracle_eval.py`
  Already does `seed -> set_task -> gymnasium_reset_strict`.
- No change intended: `src/segment_grpo_loop.py`
  Already seeds before live env/task setup and `_reset_env()` uses `gymnasium_reset_strict`.

---

### Task 1: Fix SmolVLA MetaWorld Reset Order

**Files:**
- Modify: `src/smolvla_pipeline/evaluator.py`
- Test: `tests/test_smolvla_eval_artifacts.py`

- [ ] **Step 1: Add failing test for seed/task/reset ordering**

Append to `tests/test_smolvla_eval_artifacts.py`:

```python
def test_lerobot_backend_seeds_before_set_task_then_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, object]] = []

    class FakeEnv:
        def set_task(self, task: object) -> None:
            calls.append(("set_task", task))

    backend = evaluator._LeRobotMetaWorldBackend.__new__(evaluator._LeRobotMetaWorldBackend)
    backend._env = FakeEnv()
    backend._tasks = ["task0"]
    backend._target_episode_index_override = None
    backend._max_steps = 0
    backend._np = np

    def fake_seed(seed: int) -> None:
        calls.append(("seed", int(seed)))

    def fake_reset(self: object, reset_seed: int) -> tuple[dict[str, bool], dict[str, bool]]:
        calls.append(("reset", int(reset_seed)))
        return ({"ok": True}, {"info": True})

    monkeypatch.setattr(evaluator, "seed_metaworld_process", fake_seed, raising=False)
    monkeypatch.setattr(evaluator._LeRobotMetaWorldBackend, "_reset", fake_reset)
    monkeypatch.setattr(evaluator._LeRobotMetaWorldBackend, "_render_frame", lambda self: None)

    rollout = backend.rollout_episode(episode_index=0, reset_seed=7)

    assert calls == [("seed", 7), ("set_task", "task0"), ("reset", 7)]
    assert rollout.actions == []
    assert rollout.rewards == []
```

- [ ] **Step 2: Add failing test for strict reset delegation**

Append to `tests/test_smolvla_eval_artifacts.py`:

```python
def test_lerobot_backend_reset_delegates_to_gymnasium_reset_strict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = object()
    calls: list[tuple[int, int]] = []

    backend = evaluator._LeRobotMetaWorldBackend.__new__(evaluator._LeRobotMetaWorldBackend)
    backend._env = env

    def fake_strict(seen_env: object, seed: int) -> tuple[dict[str, int], dict[str, bool]]:
        calls.append((id(seen_env), int(seed)))
        return ({"obs": 1}, {"ok": True})

    monkeypatch.setattr(evaluator, "gymnasium_reset_strict", fake_strict, raising=False)

    obs, info = backend._reset(11)

    assert calls == [(id(env), 11)]
    assert obs == {"obs": 1}
    assert info == {"ok": True}
```

- [ ] **Step 3: Run tests to verify failure**

Run:

```bash
cd /vol/bitbucket/aa6622/project
PYTHONPATH=src pytest \
  tests/test_smolvla_eval_artifacts.py::test_lerobot_backend_seeds_before_set_task_then_reset \
  tests/test_smolvla_eval_artifacts.py::test_lerobot_backend_reset_delegates_to_gymnasium_reset_strict \
  -v
```

Expected: FAIL. First test fails because evaluator does not call `seed_metaworld_process`. Second fails because `_reset()` calls raw `self._env.reset(seed=...)`, not `gymnasium_reset_strict`.

- [ ] **Step 4: Import deterministic helpers**

In `src/smolvla_pipeline/evaluator.py`, after existing local imports:

```python
from metaworld_determinism import gymnasium_reset_strict, seed_metaworld_process
```

- [ ] **Step 5: Change `_reset()` to strict reset only**

Replace `_reset()` with:

```python
def _reset(self, reset_seed: int) -> tuple[Any, dict[str, Any]]:
    reset_out = gymnasium_reset_strict(self._env, int(reset_seed))
    if isinstance(reset_out, tuple) and len(reset_out) >= 2:
        return reset_out[0], reset_out[1] if isinstance(reset_out[1], dict) else {}
    return reset_out, {}
```

- [ ] **Step 6: Seed before task selection in `rollout_episode()`**

At start of `rollout_episode()` before `if self._tasks:`:

```python
seed_metaworld_process(int(reset_seed))
```

Resulting order:

```python
def rollout_episode(self, *, episode_index: int, reset_seed: int) -> EpisodeRollout:
    seed_metaworld_process(int(reset_seed))
    if self._tasks:
        task_episode_index = (
            int(self._target_episode_index_override)
            if self._target_episode_index_override is not None
            else int(episode_index)
        )
        self._env.set_task(self._tasks[task_episode_index % len(self._tasks)])
    obs, _info = self._reset(reset_seed)
```

- [ ] **Step 7: Re-run focused tests**

Run:

```bash
cd /vol/bitbucket/aa6622/project
PYTHONPATH=src pytest \
  tests/test_smolvla_eval_artifacts.py::test_lerobot_backend_seeds_before_set_task_then_reset \
  tests/test_smolvla_eval_artifacts.py::test_lerobot_backend_reset_delegates_to_gymnasium_reset_strict \
  -v
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/smolvla_pipeline/evaluator.py tests/test_smolvla_eval_artifacts.py
git commit -m "fix(smolvla): seed metaworld before task reset"
```

---

### Task 2: Fix JEPA Render Test Reset

**Files:**
- Modify: `tests/test_metaworld_jepa_render.py`

- [ ] **Step 1: Replace raw reset with seeded strict reset**

In `tests/test_metaworld_jepa_render.py`, add inside the test after importing `build_jepa_metaworld_env`:

```python
from metaworld_determinism import gymnasium_reset_strict, seed_metaworld_process
```

Replace:

```python
if train_tasks:
    env.set_task(train_tasks[0])
env.reset(seed=0)
```

with:

```python
seed_metaworld_process(0)
if train_tasks:
    env.set_task(train_tasks[0])
gymnasium_reset_strict(env, 0)
```

- [ ] **Step 2: Run test**

Run:

```bash
cd /vol/bitbucket/aa6622/project
PYTHONPATH=src pytest tests/test_metaworld_jepa_render.py -v
```

Expected: PASS or SKIP if `metaworld` unavailable.

- [ ] **Step 3: Commit**

```bash
git add tests/test_metaworld_jepa_render.py
git commit -m "test(metaworld): seed jepa render reset"
```

---

### Task 3: Fix MT10 Smoke Script Reset

**Files:**
- Modify: `mt10/verify_env.py`

- [ ] **Step 1: Update docstring**

Replace:

```python
"""Smoke-test Meta-World MT1 (push-v3) + Gymnasium vec MT10. No project src imports."""
```

with:

```python
"""Smoke-test Meta-World MT1 (push-v3) + Gymnasium vec MT10 with project determinism helpers."""
```

- [ ] **Step 2: Add `Path` import and project src path**

After imports:

```python
from pathlib import Path
```

Then:

```python
PROJECT_SRC = Path(__file__).resolve().parents[1] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from metaworld_determinism import gymnasium_reset_strict, seed_metaworld_process
```

- [ ] **Step 3: Seed and strict reset MT1 smoke**

Replace:

```python
if train_tasks:
    env.set_task(train_tasks[0])
env.reset(seed=0)
```

with:

```python
seed_metaworld_process(0)
if train_tasks:
    env.set_task(train_tasks[0])
gymnasium_reset_strict(env, 0)
```

- [ ] **Step 4: Seed and strict reset MT10 vector smoke**

Replace:

```python
envs.reset(seed=0)
```

with:

```python
seed_metaworld_process(0)
gymnasium_reset_strict(envs, 0)
```

- [ ] **Step 5: Run smoke only if environment has MetaWorld**

Run:

```bash
cd /vol/bitbucket/aa6622/project
PYTHONPATH=src python mt10/verify_env.py
```

Expected if deps installed: prints `mt10_ok`. If deps missing: document exact missing dependency in final handoff.

- [ ] **Step 6: Commit**

```bash
git add mt10/verify_env.py
git commit -m "fix(mt10): seed smoke resets"
```

---

### Task 4: Fix Vendor Export Reset

**Files:**
- Modify: `vendor/pi05/jepa_cem_paired_pushv3_export.py`

- [ ] **Step 1: Add project helper import path near top**

After constants:

```python
PROJECT_SRC = Path(__file__).resolve().parents[2] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from metaworld_determinism import gymnasium_reset_strict, seed_metaworld_process
```

- [ ] **Step 2: Seed before initial task selection**

Before:

```python
tasks = getattr(mt1, "train_tasks", None)
if tasks:
    env.set_task(tasks[0])
```

insert:

```python
seed_metaworld_process(int(args.seed))
```

This pins any RNG touched by one-time task setup.

- [ ] **Step 3: Replace rollout raw reset and unseeded fallback**

Replace:

```python
seed = int(rng.integers(0, 2**31 - 1))
try:
    out = env.reset(seed=seed)
except TypeError:
    out = env.reset()
```

with:

```python
seed = int(rng.integers(0, 2**31 - 1))
seed_metaworld_process(seed)
out = gymnasium_reset_strict(env, seed)
```

- [ ] **Step 4: Run lightweight syntax check**

Run:

```bash
cd /vol/bitbucket/aa6622/project
PYTHONPATH=src python -m py_compile vendor/pi05/jepa_cem_paired_pushv3_export.py
```

Expected: no output, exit 0.

- [ ] **Step 5: Commit**

```bash
git add vendor/pi05/jepa_cem_paired_pushv3_export.py
git commit -m "fix(vendor): seed metaworld export resets"
```

---

### Task 5: Verify Already-Fixed Oracle And GRPO Paths

**Files:**
- Read-only verification: `scripts/oracle/run_metaworld_oracle_eval.py`
- Read-only verification: `src/segment_grpo_loop.py`

- [ ] **Step 1: Confirm oracle order remains correct**

Run:

```bash
cd /vol/bitbucket/aa6622/project
rg "_seed_all\\(reset_seed\\)|env\\.set_task|gymnasium_reset_strict" scripts/oracle/run_metaworld_oracle_eval.py -n -C 2
```

Expected ordering in episode loop: `_seed_all(reset_seed)` before `env.set_task(...)`, then `gymnasium_reset_strict(env, reset_seed)`.

- [ ] **Step 2: Confirm GRPO order remains correct**

Run:

```bash
cd /vol/bitbucket/aa6622/project
rg "seed_metaworld_process|env\\.set_task|_reset_env\\(|gymnasium_reset_strict" src/segment_grpo_loop.py -n -C 2
```

Expected: live sim path seeds before task setup, and `_reset_env()` calls `gymnasium_reset_strict`.

No commit for this task unless code was changed.

---

### Task 6: Repo-Wide Gates

**Files:**
- No direct source change unless grep finds missed unsafe reset.

- [ ] **Step 1: Raw reset audit**

Run:

```bash
cd /vol/bitbucket/aa6622/project
rg "env\\.reset\\(\\)" --glob "*.py"
```

Expected: no results.

- [ ] **Step 2: Direct seeded reset audit**

Run:

```bash
cd /vol/bitbucket/aa6622/project
rg "\\.reset\\(seed=" --glob "*.py"
```

Expected allowed results only:

```text
src/metaworld_determinism.py
```

Possible acceptable exceptions:

```text
tests/... fake env definitions with def reset(self, *, seed=None)
comments/docstrings explaining reset(seed=...)
```

If `src/`, `scripts/`, `mt10/`, or `vendor/` contain direct runtime `env.reset(seed=...)`, fix them before completion.

- [ ] **Step 3: Focused pytest suite**

Run:

```bash
cd /vol/bitbucket/aa6622/project
PYTHONPATH=src pytest \
  tests/test_metaworld_determinism.py \
  tests/test_smolvla_eval_artifacts.py \
  tests/test_metaworld_jepa_render.py \
  tests/test_segment_grpo_loop.py::test_reset_env_strict_requires_reset_seed_kwarg \
  tests/test_segment_grpo_loop.py::test_reset_env_delegates_to_gymnasium_reset_strict \
  -v
```

Expected: PASS, except `test_metaworld_jepa_render.py` may SKIP if MetaWorld unavailable.

- [ ] **Step 4: Optional production smoke**

If LeRobot/MetaWorld GPU env is available, run one SmolVLA parity episode:

```bash
cd /vol/bitbucket/aa6622/project
bash scripts/smolvla/run_pushv3_smolvla_parity_benchmark.sh
```

Expected: run writes valid `eval_info.json`, `run_manifest.json`, and video artifacts. If dependencies are unavailable, record that smoke was not run.

---

## Self-Review

- Spec coverage: every `rg`-visible runtime reset site is covered. SmolVLA production path gets actual bug fix; oracle/GRPO stay as working references; smoke/test/vendor paths are made strict.
- No hidden ordering abstraction: plan preserves visible `seed -> set_task -> reset` order.
- No redundant SmolVLA action clip: existing `_coerce_exec_action()` remains single normal-path clip.
- Verification gate narrowed: raw `env.reset()` forbidden; direct runtime `reset(seed=...)` forbidden outside helper.

Plan complete and saved to `docs/superpowers/plans/2026-04-29-metaworld-determinism-everywhere.md`.

Two execution options:

1. **Subagent-Driven (recommended)** - dispatch one fresh subagent per task, review between tasks.
2. **Inline Execution** - execute tasks in this session with checkpoints.

Which approach?
