# Phase12 Phase11-Parity WM-GRPO Implementation Plan (v2 — post-review)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Supersedes:** `docs/superpowers/plans/2026-05-19-phase12-phase11-parity-wm-grpo.md` for sequencing and correctness. Reuse v1 step code where marked **UNCHANGED**; do not copy v1 Task 3 Step 5 adapter or v1 Task 4 WM-only body.

**Goal:** Schedule overnight WM-GRPO on `push-v3` with Phase11 operational contracts (seed-batch, telemetry, PBS, eval top-k) while keeping the real training signal: **oracle subgoals + WM candidate ranking**, not MetaWorld reward on the selected chunk.

**Architecture:** Three train semantics, not two:
- `selected_env` — current hybrid (oracle + WM score + winner `env.step`).
- `wm_only` — oracle + WM score + **no** winner `env.step` (this is the overnight default).
- `vector_selected_env` — deferred to `docs/superpowers/plans/2026-05-18-phase12-async-wm-grpo.md` after `wm_only` smokes pass.

WM-only **still runs expert oracle rollout** in MetaWorld to build subgoal latents; it only removes selected-chunk execution during the GRPO update.

**Tech Stack:** Python 3.12, PyTorch, LeRobot SmolVLA GRPO, MetaWorld official backend, JEPA-WM, PBS, pytest.

---

## Review findings (why v2 exists)

| Severity | Issue in v1 | v2 fix |
|----------|-------------|--------|
| **P0** | Task 4 `wm_only` uses reset frame as goal → flat WM rewards | Task 2: factor shared `build_phase12_goals_from_oracle`; `wm_only` reuses it |
| **P0** | Task 3 interim adapter calls `collect_phase12_training_episode` (selected env steps) | Drop adapter; wire real `wm_only` in one task |
| **P1** | PBS chain sets `PHASE12_GROUP_SIZE` but prod PBS hardcodes `--group-size 16` | Parameterize PBS; matrix matches CLI |
| **P1** | Task 12 dry-run expects `PHASE12_DRY_RUN` on new smoke PBS | Add dry-run branch to smoke PBS |
| **P1** | Task 6 “batched logprob” is still per-row loop | Task 7: real `get_action_probs_for_chunk_batch_from_proc_list` |
| **P1** | `goal_frame_index_1based` wrong in `phase12_rollout.py` | Task 1 preflight fix |
| **P2** | Eval subprocess uses step eval (`chunk_len` 1) | Task 10: default vector + `--chunk-len 25` |
| **P2** | `rollout_validation` uses Phase8 path | Task 1: document or gate; not overnight blocker |
| **P2** | Overlap with async plan | v2 = serial parity first; async plan stage 2 after smoke |
| **Note** | Task 5 uses per-row `compute_group_advantages` not `compute_seed_batch_advantages` | **Correct** for multi-segment Phase12; document in manifest |

---

## File structure (delta from v1)

| File | Change |
|------|--------|
| `src/smolvla_grpo/phase12_oracle_goals.py` | **Create** — shared oracle → goals builder |
| `src/smolvla_grpo/phase12_wm_only_rollout.py` | **Create** — UNCHANGED collector from v1 Task 3 |
| `src/smolvla_grpo/phase12_rollout.py` | Fix `goal_frame_index_1based` |
| `scripts/grpo/train_phase12_wm_chunk_grpo.py` | Modes, seed-batch, loss, logprob, telemetry |
| `scripts/grpo/phase12_seedbatch_*.pbs` | Parameterized `GROUP_SIZE`, dry-run |
| `scripts/grpo/submit_phase12_pbs_chain.sh` | Pass env vars PBS actually reads |

---

## Task 0: Preflight correctness (before new features)

**Files:**
- Modify: `src/smolvla_grpo/phase12_rollout.py`
- Modify: `scripts/grpo/train_phase12_wm_chunk_grpo.py`
- Test: `tests/test_phase12_rollout.py`

- [ ] **Step 1: Failing test for goal frame metadata**

```python
def test_segment_record_uses_goal_frame_not_segment_index():
    # Build minimal episode with goal.frame_index_1based=25, segment_index=0
    # Assert segment.goal_frame_index_1based == 25
```

- [ ] **Step 2: Fix collector**

In `collect_phase12_episode`, replace:

```python
goal_frame_index_1based=int(segment_index + 1),
```

with:

```python
goal_frame_index_1based=int(getattr(goal, "frame_index_1based", segment_index + 1)),
```

- [ ] **Step 3: Probe action_dim in `load_phase12_train_resources`**

Replace hardcoded `4` with bundle/env probe (mirror Phase11 `load_bundle_for_grpo`).

- [ ] **Step 4: Manifest note for `num_episodes`**

Add manifest field:

```python
"episodes_per_update_semantics": "one_update_may_include_batch_size_reset_seeds",
```

Keep validation `num_episodes == num_updates` for now (matches Phase11 “updates” counter); document that `batch_size` multiplies seeds per update, not episode count.

- [ ] **Step 5: Run tests + commit**

```bash
PYTHONPATH="${PWD}/src:${PYTHONPATH:-}" /rds/general/user/aa6622/home/.envs/lerobot_mw_py312/bin/python -m pytest tests/test_phase12_rollout.py -q
git add src/smolvla_grpo/phase12_rollout.py scripts/grpo/train_phase12_wm_chunk_grpo.py tests/test_phase12_rollout.py
git commit -m "fix(phase12): goal frame metadata and action_dim probe"
```

---

## Task 1: Shared oracle → goals builder

**Files:**
- Create: `src/smolvla_grpo/phase12_oracle_goals.py`
- Modify: `scripts/grpo/train_phase12_wm_chunk_grpo.py`
- Test: `tests/test_phase12_oracle_goals.py`

- [ ] **Step 1: Extract from `collect_phase12_training_episode`**

```python
def build_phase12_oracle_context(
    *,
    env_h,
    wm_bundle,
    args,
    reset_seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    """Returns oracle dict, schedule, goals list, oracle_dir, max_steps."""
```

Move `_rollout_phase12_oracle`, `build_subgoal_schedule`, goal encoding loop into this module. `collect_phase12_training_episode` becomes a thin caller.

- [ ] **Step 2: Test — goals length matches schedule**

- [ ] **Step 3: Commit**

```bash
git commit -m "refactor(phase12): extract oracle goal builder"
```

---

## Task 2: Train modes + real `wm_only` (no v1 adapter)

**Files:**
- Modify: `scripts/grpo/train_phase12_wm_chunk_grpo.py`
- Create: `src/smolvla_grpo/phase12_wm_only_rollout.py` (v1 Task 3 collector — **UNCHANGED**)
- Test: `tests/test_phase12_wm_only_rollout.py`, `tests/test_phase12_training_loop.py`

- [ ] **Step 1: CLI `--phase12-train-mode {selected_env,wm_only}`** — v1 Task 1 **UNCHANGED**

- [ ] **Step 2: Implement `collect_phase12_wm_only_training_episode`**

```python
def collect_phase12_wm_only_training_episode(**kwargs) -> Any:
    ctx = build_phase12_oracle_context(...)
    root_after_parity = reset_and_parity_gate(ctx["env_h"], seed=reset_seed, ...)
    root = {
        "image": wm_frame_from_obs(root_after_parity),
        "proprio": proprio,
        "proc": env_h.build_proc(obs, bundle=bundle),
    }
    return collect_phase12_wm_only_episode(
        root_source=RootSource(root),
        goals=ctx["goals"],  # oracle subgoals — NOT reset-as-goal
        ...
    )
```

**Do not** call `collect_phase12_training_episode` or `collect_phase12_episode` (selected path).

- [ ] **Step 3: Tests** — v1 `test_wm_only_collector_scores_candidates_without_env_step` + trainer branch test (v1 Task 4) **UNCHANGED**

- [ ] **Step 4: Commit**

```bash
git commit -m "feat(phase12): wm_only uses oracle goals without selected env step"
```

---

## Task 3: Explicit GRPO group normalizer

v1 **Task 2 UNCHANGED**, with one doc addition in commit message:

- Microbatch/stack normalizer `G = group_size` is per **candidate group** (per segment), not per update.
- Stack mode must use `_chunk_grpo_loss_with_group_normalizer` (sum of row losses / G), not `.mean()` over all rows.

---

## Task 4: Serial seed-batch

v1 **Task 5 UNCHANGED** except manifest annotation:

```python
"advantage_mode": "per_segment_group",  # not phase11 flat seed-batch; multi-segment
```

Test assertion remains: `len(last["returns"]) == batch_size * group_size * num_segments` — for smoke with 1 segment: `2 * 8 = 16` ✓

---

## Task 5: Real batched logprob recompute

**Files:**
- Modify: `scripts/grpo/train_phase12_wm_chunk_grpo.py`
- Modify: `src/smolvla_grpo/policy_wrapper.py` (only if batch API missing chunk path)
- Test: `tests/test_phase12_training_loop.py`

- [ ] **Step 1: Add CLI** — v1 Task 6 args **UNCHANGED**

- [ ] **Step 2: Implement with policy batch API**

```python
def _recompute_phase12_logprobs_batched(...):
    # Group rows by (proc signature, chunk shape) like Phase11
    # Call train_wrapper.get_action_probs_for_chunk_batch_from_proc_list
    # Count forwards = number of batch API calls, not len(chunks)
```

- [ ] **Step 3: Wire `logprob_recompute_mode=loop` fallback** to current row loop

- [ ] **Step 4: Commit**

```bash
git commit -m "feat(phase12): true batched chunk logprob recompute"
```

---

## Task 6: WM encode cache

v1 **Task 7 UNCHANGED**

---

## Task 7: Phase11-grade telemetry

v1 **Task 8 UNCHANGED** + add fields Phase11 already logs:

```python
"rollout_policy_batch_size": None,  # train serial; null
"logprob_recompute_mode": str(args.logprob_recompute_mode),
```

---

## Task 8: PBS scripts (parameterized)

**Files:**
- Create: `scripts/grpo/phase12_seedbatch_smoke_u2.pbs`
- Create: `scripts/grpo/phase12_seedbatch_b4_g16_train_0000_0050.pbs`

Critical PBS template lines:

```bash
GROUP_SIZE="${PHASE12_GROUP_SIZE:-16}"
BATCH_SIZE="${PHASE12_BATCH_SIZE:-4}"
...
  --group-size "${GROUP_SIZE}" \
  --batch-size "${BATCH_SIZE}" \
...
if [[ "${PHASE12_DRY_RUN:-0}" == "1" ]]; then
  EXTRA_FLAGS+=(--dry-run)
fi
```

Dry-run success string: `PHASE12_WM_CHUNK_DRY_RUN_OK` (trainer already prints on `--dry-run`).

Manifest assert uses `"${GROUP_SIZE}"` not hardcoded 16.

---

## Task 9: Eval top-k

v1 **Task 10 UNCHANGED** plus:

- Default PBS eval: `--chunk-len 25` always
- Subprocess mode: print warning in manifest if `chunk_len != train chunk_len`
- Rank key matches Phase111: `(pc_success, avg_sum_reward)` — v1 correct

---

## Task 10: PBS chain + experiment matrix

**Files:**
- Create: `scripts/grpo/submit_phase12_pbs_chain.sh`

Matrix (aligned with parameterized PBS):

| Label | action_profile | BATCH | GROUP | LR | clip | noise |
|-------|----------------|-------|-------|-----|------|-------|
| run_a | official_jepa_mirror | 4 | 8 | 1e-5 | 0.2 | 0.2 |
| run_b | bounded_executed | 4 | 8 | 1e-5 | 0.2 | 0.2 |
| run_c | official_jepa_mirror | 4 | 16 | 5e-6 | 0.1 | 0.2 |
| run_d | bounded_executed | 4 | 16 | 5e-6 | 0.1 | 0.2 |
| run_e | official_jepa_mirror | 4 | 16 | 5e-6 | 0.1 | 0.1 |

Chain must export `PHASE12_GROUP_SIZE` **and** PBS must use it (Task 8).

---

## Task 11: Gate + smokes

v1 **Task 12** with fixes:

```bash
# Step 2 dry-run
PHASE12_DRY_RUN=1 qsub scripts/grpo/phase12_seedbatch_smoke_u2.pbs

# Step 3 real smoke — wm_only + b2 g8 u2
qsub scripts/grpo/phase12_seedbatch_smoke_u2.pbs

# Optional selected_env control (one job)
PHASE12_TRAIN_MODE=selected_env PHASE12_GROUP_SIZE=4 qsub ...
```

---

## Deferred (explicit — not v2 scope)

| Item | Plan |
|------|------|
| Vector train `rollout-execution vector_async` | `2026-05-18-phase12-async-wm-grpo.md` after wm_only smoke |
| `rollout_policy_batch_size` on train | Async plan stage 2 + benchmark |
| WM candidate batch scoring | After encode cache stable |
| `update_epochs` > 1 | Only if Phase11 ablations show benefit |
| `rollout_validation` Phase12 parity | Separate small task |
| Second-wave 5 jobs `afterok` | Manual `qsub -W depend=afterok:...` once wave-1 walltime known |

---

## Self-review (v2)

| Requirement | Task |
|-------------|------|
| Oracle subgoals for wm_only | 1–2 |
| No selected env.step in wm_only | 2 |
| Seed-batch | 4 |
| Explicit G normalizer | 3 |
| True batched logprob | 5 |
| WM cache | 6 |
| Telemetry | 7 |
| Parameterized PBS + dry-run | 8 |
| Eval top-k + chunk_len 25 | 9 |
| Five-run chain | 10 |
| Preflight bugs | 0 |

Placeholder scan: none.
