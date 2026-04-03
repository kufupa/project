# MT10 Phase 6/8/9 vs push-v3 oracle consistency audit

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align MetaWorld oracle baselines (push-v3 pipeline, phase6 MT10), segment-GRPO sim (phase8), and oracle-vs-WM (phase9) on the same RNG + action-step contracts so episode seeds and logged actions match oracle eval semantics.

**Architecture:** Single source of truth for “what oracle eval does” lives in `scripts/oracle/run_metaworld_oracle_eval.py` (`_seed_all` each episode, `reset_seed = base_seed + episode_index`, clip `[-1,1]` before `env.step`). Phase8/9 call `segment_grpo_loop.rollout_with_chunks` with `reset_seed` from campaign/manifest; phase6 shells call `pushv3_oracle_data_pipeline.sh`, which invokes that oracle script. MT10 bash wrappers wire env vars and parse log lines into index JSON.

**Tech stack:** Bash (Slurm-friendly wrappers), Python 3.10+, MetaWorld, `segment_grpo_loop`, pytest.

---

## File map (ownership)

| Path | Role |
|------|------|
| `project/scripts/oracle/run_metaworld_oracle_eval.py` | Oracle rollouts: `_seed_all`, per-episode `reset_seed`, clipped `env.step`. |
| `project/scripts/oracle/pushv3_oracle_data_pipeline.sh` | Full push-v3 flow; logs `Baseline output directory:` for phase6 awk. |
| `project/scripts/oracle/run_oracle_baseline_eval.sh` | Direct baseline only; logs `Baseline eval output directory:` (preflight). |
| `project/scripts/mt10/run_phase6_mt10.sh` | Per-task `PUSHV3_*` → pipeline; parses `Baseline output directory:`. |
| `project/scripts/mt10/run_phase8_mt10.sh` | Reads phase6 index; runs `run_all60_frame50_k3.py`; parses `[campaign] run_dir=`. |
| `project/scripts/mt10/run_phase9_mt10.sh` | Reads phase6 index; runs `run_phase9_oracle_vs_wm.py`; parses `[phase9] run directory:`. |
| `project/scripts/mt10/run_preflight_smoke.sh` | Sequential 1-ep smoke; default report JSON path (now job-suffixed under Slurm). |
| `project/src/segment_grpo_loop.py` | WM/sim rollout: must mirror oracle seed + clip for parity. |
| `project/scripts/segment_grpo/run_all60_frame50_k3.py` | `reset_seed = seed_base + target_episode` (matches oracle). |
| `project/scripts/run_phase9_oracle_vs_wm.py` | Uses manifest `reset_seed` rows + `rollout_with_chunks`. |
| `project/tests/test_oracle_eval_script.py` | Oracle `_clip_action` contract. |
| `project/tests/test_segment_grpo_loop.py` | Segment loop unit tests including clip contract. |

---

## Consistency matrix (verified)

| Concern | Oracle eval | Phase6 (pushv3) | Preflight (baseline only) | Phase8/9 (`rollout_with_chunks`) |
|---------|-------------|-----------------|----------------------------|----------------------------------|
| Per-episode seed | `reset_seed = --seed + episode_index` | Same via `run_metaworld_oracle_eval.py` | 1 ep, seed 1000 | `run_all60`: `seed_base + ep`; phase9 reads `reset_seed` from oracle manifest |
| Global RNG before env | `_seed_all(reset_seed)` each episode | Same | Same | **Now** `_seed_metaworld_globals(seed)` once before MetaWorld import when `carry_mode=sim` and not `dry_run` |
| Action to `env.step` | Clipped `[-1,1]` | Same | Same | **Now** `_clip_action_for_metaworld_box` inside `_step_env`; `executed_actions` logs stepped vector |
| Log line for output dir | N/A (inner) | `Baseline output directory:` | `Baseline eval output directory:` | Campaign / phase9 print their own markers |

---

## Self-review (spec coverage)

1. **Spec coverage:** RNG parity, action clip parity, bash log parsing, preflight JSON collision — each has a task below.
2. **Placeholder scan:** None.
3. **Type consistency:** `_step_env` returns 5-tuple `(image, proprio, info, done, stepped)`; only caller updated in `segment_grpo_loop.py`.

---

### Task 1: Oracle eval contract (reference + tests)

**Files:**

- Reference: `project/scripts/oracle/run_metaworld_oracle_eval.py` (`_seed_all`, `_clip_action`, main loop)
- Test: `project/tests/test_oracle_eval_script.py`

- [ ] **Step 1: Confirm `_clip_action` return order**

Open `run_metaworld_oracle_eval.py` and verify `_clip_action` returns `(clipped_list, raw_list, oob_bool)` and `env.step` uses clipped float array.

- [ ] **Step 2: Run oracle script tests**

Run:

```bash
cd /vol/bitbucket/aa6622/project && python3 -m pytest tests/test_oracle_eval_script.py -q --tb=short
```

Expected: all tests passed (e.g. `9 passed` in a minimal env).

- [ ] **Step 3: Commit (if git repo)**

```bash
git add project/scripts/oracle/run_metaworld_oracle_eval.py project/tests/test_oracle_eval_script.py
git commit -m "fix(oracle): document seed+clip contract in tests"
```

Skip if workspace is not a git checkout.

---

### Task 2: Segment-GRPO sim parity with oracle (implemented)

**Files:**

- Modify: `project/src/segment_grpo_loop.py:32-44` (new helpers), `2119-2131` (`_step_env`), `2276-2278` (seed before MetaWorld), `2565-2580` (unpack stepped, log clipped actions)
- Modify: `project/tests/test_segment_grpo_loop.py` (import + `test_clip_action_for_metaworld_box_matches_oracle_eval_contract`)

**Target code (must be present after this task):**

```32:44:project/src/segment_grpo_loop.py
def _seed_metaworld_globals(seed: int) -> None:
    """Match scripts/oracle/run_metaworld_oracle_eval.py _seed_all: env.reset(seed=) alone is not enough."""
    import random

    random.seed(int(seed))
    np.random.seed(int(seed))
    if torch is not None:
        torch.manual_seed(int(seed))


def _clip_action_for_metaworld_box(action: np.ndarray) -> np.ndarray:
    """MetaWorld scripted policies use [-1, 1]; oracle eval clips before env.step for reproducibility."""
    return np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
```

```2119:2131:project/src/segment_grpo_loop.py
def _step_env(env: Any, action: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any], bool, np.ndarray]:
    stepped = _clip_action_for_metaworld_box(action)
    out = env.step(stepped)
    if len(out) == 5:
        obs, _reward, terminated, truncated, info = out
        done = bool(terminated or truncated)
    else:
        obs, _reward, done, info = out
        done = bool(done)
    image, proprio = _extract_image_and_proprio(obs, env)
    if not isinstance(info, dict):
        info = {}
    return image, proprio, info, done, stepped
```

```2276:2278:project/src/segment_grpo_loop.py
        if not dry_run:
            _seed_metaworld_globals(int(seed))
            try:
```

```2565:2580:project/src/segment_grpo_loop.py
        if carry_mode == "sim" and env is not None:
            for i in range(effective_len):
                action_env = _take_action_for_env(best_actions[i], int(env_action_dim))
                _obs_img, current_proprio, _info, step_done, stepped = _step_env(env, action_env)
                if jepa_parity_sim and _render_jepa_rgb is not None:
                    current_image = _render_jepa_rgb(env)
                else:
                    current_image = _obs_img
                current_policy_image = _derive_policy_rgb_for_smolvla(
                    current_image,
                    jepa_parity_sim=jepa_parity_sim,
                    policy_hflip_corner2=smolvla_policy_hflip_corner2,
                )
                carried_steps += 1
                current_step += 1
                executed_actions.append(stepped.tolist())
```

**New test (full file excerpt for copy-paste):**

```python
def test_clip_action_for_metaworld_box_matches_oracle_eval_contract() -> None:
    a = np.array([2.0, -1.5, 0.25], dtype=np.float32)
    clipped = _clip_action_for_metaworld_box(a)
    assert clipped.dtype == np.float32
    assert np.allclose(clipped, np.array([1.0, -1.0, 0.25], dtype=np.float32))
```

- [ ] **Step 1: Run clip test**

```bash
cd /vol/bitbucket/aa6622/project && python3 -m pytest tests/test_segment_grpo_loop.py::test_clip_action_for_metaworld_box_matches_oracle_eval_contract -v
```

Expected: `PASSED`.

- [ ] **Step 2: Run full `test_segment_grpo_loop.py`**

```bash
cd /vol/bitbucket/aa6622/project && python3 -m pytest tests/test_segment_grpo_loop.py -q --tb=line
```

Expected: no failures (skips OK).

- [ ] **Step 3: Commit**

```bash
git add project/src/segment_grpo_loop.py project/tests/test_segment_grpo_loop.py
git commit -m "fix(segment-grpo): seed globals + clip actions like oracle eval"
```

---

### Task 3: Preflight report path collision under Slurm (implemented)

**Files:**

- Modify: `project/scripts/mt10/run_preflight_smoke.sh:18-19`

**Required line:**

```bash
REPORT_JSON="${MT10_PREFLIGHT_REPORT_JSON:-${PROJECT_ROOT}/artifacts/mt10_runs/mt10_preflight_report${SLURM_JOB_ID:+_job${SLURM_JOB_ID}}.json}"
```

- [ ] **Step 1: Shellcheck (optional)**

```bash
shellcheck /vol/bitbucket/aa6622/project/scripts/mt10/run_preflight_smoke.sh
```

Expected: no errors (warnings acceptable per project policy).

- [ ] **Step 2: Commit**

```bash
git add project/scripts/mt10/run_preflight_smoke.sh
git commit -m "fix(mt10): default preflight report path unique per Slurm job"
```

---

### Task 4: Phase6/8/9 bash vs pushv3 (read-only verification)

**Files:** `run_phase6_mt10.sh`, `run_phase8_mt10.sh`, `run_phase9_mt10.sh`, `pushv3_oracle_data_pipeline.sh`, `run_preflight_smoke.sh`

- [ ] **Step 1: Confirm phase6 awk matches pushv3 log**

```bash
grep -n "Baseline output directory" /vol/bitbucket/aa6622/project/scripts/oracle/pushv3_oracle_data_pipeline.sh
grep -n "Baseline output directory" /vol/bitbucket/aa6622/project/scripts/mt10/run_phase6_mt10.sh
```

Expected: both contain the substring `Baseline output directory:`.

- [ ] **Step 2: Confirm preflight awk matches baseline-only script**

```bash
grep -n "Baseline eval output directory" /vol/bitbucket/aa6622/project/scripts/mt10/run_preflight_smoke.sh
grep -n "Baseline eval output directory" /vol/bitbucket/aa6622/project/scripts/oracle/run_oracle_baseline_eval.sh
```

Expected: both reference the same string the baseline script prints.

- [ ] **Step 3: Confirm default counts align with push-v3 MT10 intent**

| Variable | Phase6 default | Preflight |
|----------|----------------|-----------|
| Episodes | `MT10_PHASE6_EPISODES:-60` | `ORACLE_BASELINE_EPISODES=1` |
| Seed | `MT10_PHASE6_SEED:-1000` | `ORACLE_BASELINE_SEED=1000` |
| Episode length | `MT10_PHASE6_EPISODE_LENGTH:-120` | `120` |

No code change unless product requirements change.

---

### Task 5 (optional YAGNI): Deduplicate `_seed_all` / `_seed_metaworld_globals`

**Files:** Could add `project/src/metaworld_seed.py` and import from oracle script + segment loop — only do this if a third caller appears; until then duplicate 8 lines is acceptable per YAGNI.

- [ ] **Step 1: Close as cancelled** unless duplication causes drift again.

---

## Execution handoff

Plan complete and saved to `project/docs/superpowers/plans/2026-04-21-mt10-pushv3-phase-consistency-audit.md`. Two execution options:

**1. Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks, fast iteration. **REQUIRED SUB-SKILL:** superpowers:subagent-driven-development.

**2. Inline Execution** — run checkbox steps in this session using executing-plans with checkpoints. **REQUIRED SUB-SKILL:** superpowers:executing-plans.

**Which approach?**

**Note:** `/dispatching-parallel-agents` not used here — RNG/action/bash parsing issues share one contract; parallel agents would duplicate context and risk conflicting edits.

**Worktree:** writing-plans skill prefers a dedicated git worktree for implementation; this audit was applied in-place because the workspace path was not a git root from the agent shell.
