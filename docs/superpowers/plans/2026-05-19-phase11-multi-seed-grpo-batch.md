# Phase11 Multi-Seed GRPO Batch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable `--batch-size B` on Phase11 so each update runs `B` independent reset seeds × `group_size G` rollouts, with GRPO advantages normalized **per seed** (not across all `B×G` trajectories), matching pop128 sample count at ~1/B peak CPU RAM.

**Architecture:** v1 uses **serial seed batches**: loop `B` calls to existing `collect_rollout_group(reset_seed=…, group_size=G)` (peak `G` vector envs, not `B×G`). Trainer concatenates trajectories, computes advantages via new `compute_seed_batch_advantages`, and passes explicit `grpo_group_size=G` into loss so normalizers stay `n_units × G`. Do **not** merge all seeds into one `compute_group_advantages` call — that would change the algorithm.

**Tech Stack:** Python 3.12, PyTorch, Phase11 trainer (`train_phase11_env_on_policy_grpo.py`), rollout (`phase11_rollout.py`, `official_lerobot_vector_rollout.py`), `grpo_math.py`, PBS, pytest.

---

## Why this (not bigger `group_size`)

| Knob | What it does today | Effect |
|------|-------------------|--------|
| `group_size` | `G` parallel envs, **same** `reset_seed`, different traj RNG | Better rank estimate for **one** MetaWorld initial state |
| `batch_size` (new) | `B` **different** `reset_seed`s per update, each with its own GRPO group of `G` | Same total rollouts `B×G`, more seed diversity, lower peak RAM than `group_size=B×G` |

Literature aligns: GRPO/verl normalizes rewards **within a group tied to one prompt/context** ([verl GRPO docs](https://verl.readthedocs.io/en/latest/algo/grpo.html)); TGRPO uses multiple parallel envs per task instruction ([TGRPO paper](https://arxiv.org/html/2506.08440v1)). Our `reset_seed` ≈ prompt; `group_size` ≈ `rollout.n`.

**Out of scope for v1:** single vector env with `n_envs=B×G` and `reset_many` (faster walltime, same RAM as pop128 — defer to v2).

---

## File map

| File | Responsibility |
|------|----------------|
| `src/smolvla_grpo/grpo_math.py` | `compute_seed_batch_advantages` |
| `src/smolvla_grpo/phase11_rollout.py` | `collect_rollout_seed_batch` orchestrator |
| `scripts/grpo/train_phase11_env_on_policy_grpo.py` | Lift `batch_size==1` guard; seed list; loss `grpo_group_size` |
| `tests/test_grpo_math.py` | Advantage grouping tests (create if missing, else extend) |
| `tests/test_phase11_true_action_chunking.py` | Rollout orchestrator + loss `grpo_group_size` tests |
| `scripts/grpo/phase11_seedbatch_smoke_u2.pbs` | 30m smoke: `batch_size=2`, `group_size=8`, 2 updates |
| `scripts/grpo/phase11_seedbatch_b4_g32_train_0000_0050.pbs` | Production: `batch_size=4`, `group_size=32` (128 total rollouts/update) |

---

## Algorithm contract (do not break)

1. **Rollout:** For update `u`, seeds `s_b = train_seed_base + u * batch_size + b` for `b in 0..B-1`. Collect `G` trajectories per `s_b`.
2. **Advantages:** For each seed group, `A = (R - mean(R)) / (std(R) + eps)` over exactly `G` returns. Concatenate to length `B×G` in seed order.
3. **Loss normalizer:** Each chunk row from trajectory `i` in group `b` uses `normalizer = n_units_i * G` where `G = group_size` (**not** `len(all_rollouts)`).
4. **Skip:** If **all** advantages ≈ 0, skip update (same as today). If one seed group has zero std, that group's advantages are 0; other groups still train.
5. **Metadata:** Set `traj.metadata["seed_batch_index"] = b` and keep `reset_seed` per trajectory.

---

### Task 1: Seed-batch advantage helper

**Files:**
- Modify: `src/smolvla_grpo/grpo_math.py`
- Create or modify: `tests/test_grpo_math.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_grpo_math.py
import torch
from smolvla_grpo.grpo_math import compute_group_advantages, compute_seed_batch_advantages


def test_compute_seed_batch_advantages_matches_per_group_manual():
    # batch_size=2, group_size=3
    returns = torch.tensor([10.0, 12.0, 8.0, 1.0, 1.0, 1.0], dtype=torch.float32)
    got = compute_seed_batch_advantages(returns, group_size=3)
    g0 = compute_group_advantages(returns[:3])
    g1 = compute_group_advantages(returns[3:])
    expected = torch.cat([g0, g1], dim=0)
    torch.testing.assert_close(got, expected)


def test_compute_seed_batch_advantages_rejects_bad_shape():
    import pytest

    with pytest.raises(ValueError, match="multiple of group_size"):
        compute_seed_batch_advantages(torch.tensor([1.0, 2.0]), group_size=3)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cd /rds/general/user/aa6622/home/project
PYTHONPATH="${PWD}/src:${PYTHONPATH:-}" /rds/general/user/aa6622/home/.envs/lerobot_mw_py312/bin/python -m pytest tests/test_grpo_math.py -v
```

Expected: FAIL with `ImportError` or `cannot import name 'compute_seed_batch_advantages'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/smolvla_grpo/grpo_math.py`:

```python
def compute_seed_batch_advantages(
    returns: torch.Tensor,
    *,
    group_size: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Normalize returns within each contiguous seed group of size group_size."""
    flat = returns.reshape(-1).float()
    g = int(group_size)
    if g < 1:
        raise ValueError("group_size must be >= 1")
    if flat.numel() == 0:
        return flat
    if flat.numel() % g != 0:
        raise ValueError(
            f"returns length {flat.numel()} must be a multiple of group_size={g}"
        )
    chunks = flat.reshape(-1, g)
    adv = torch.stack(
        [compute_group_advantages(chunk, eps=eps) for chunk in chunks],
        dim=0,
    )
    return adv.reshape(-1)
```

- [ ] **Step 4: Run test to verify it passes**

Run same pytest command. Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /rds/general/user/aa6622/home/project
git add src/smolvla_grpo/grpo_math.py tests/test_grpo_math.py
git commit -m "$(cat <<'EOF'
feat(grpo): add per-seed batch advantage helper

Phase11 multi-seed batching needs advantages normalized within each
reset_seed group, not across the full B×G rollout tensor.
EOF
)"
```

---

### Task 2: Explicit `grpo_group_size` in Phase11 group loss

**Files:**
- Modify: `scripts/grpo/train_phase11_env_on_policy_grpo.py`
- Modify: `tests/test_phase11_true_action_chunking.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_phase11_true_action_chunking.py`:

```python
def test_phase11_group_loss_uses_explicit_grpo_group_size_not_rollout_count():
    train = _load_phase11_train_module()
    # 4 rollouts but GRPO group_size=2 → normalizer n_units*2, not *4
    rollouts = [
        SimpleNamespace(action_chunks=[_action_chunk("a0", steps=5)]),
        SimpleNamespace(action_chunks=[_action_chunk("a1", steps=5)]),
        SimpleNamespace(action_chunks=[_action_chunk("b0", steps=5)]),
        SimpleNamespace(action_chunks=[_action_chunk("b1", steps=5)]),
    ]
    wrapper = FakeChunkTrainWrapper(init=-0.1)
    train._backward_phase11_group_loss(
        train_wrapper=wrapper,
        rollouts=rollouts,
        advantages=torch.tensor([1.0, -1.0, 1.0, -1.0], dtype=torch.float32),
        device=torch.device("cpu"),
        optimizer_chunk_size=1,
        clip_eps=0.2,
        logprob_recompute_mode="batched",
        logprob_batch_size=16,
        grpo_group_size=2,
        telemetry=None,
    )
    # Same as two independent groups of size 2 with adv [1,-1] each:
    expected = -torch.exp(torch.tensor(-0.1)) * (
        torch.tensor(1.0) / 2 + torch.tensor(-1.0) / 2
        + torch.tensor(1.0) / 2 + torch.tensor(-1.0) / 2
    )
    torch.testing.assert_close(wrapper.scale.grad, expected)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cd /rds/general/user/aa6622/home/project
PYTHONPATH="${PWD}/src:${PYTHONPATH:-}" /rds/general/user/aa6622/home/.envs/lerobot_mw_py312/bin/python -m pytest \
  tests/test_phase11_true_action_chunking.py::test_phase11_group_loss_uses_explicit_grpo_group_size_not_rollout_count -v
```

Expected: FAIL (`TypeError: unexpected keyword argument 'grpo_group_size'` or wrong gradient)

- [ ] **Step 3: Write minimal implementation**

In `_backward_phase11_group_loss` signature add `grpo_group_size: int | None = None`, and replace:

```python
    G = len(rollouts)
```

with:

```python
    G = int(grpo_group_size if grpo_group_size is not None else len(rollouts))
    if G < 1:
        raise ValueError("grpo_group_size must be >= 1")
    if len(rollouts) % G != 0:
        raise ValueError(
            f"rollout count {len(rollouts)} must be a multiple of grpo_group_size={G}"
        )
```

Update the single call site in `main()`:

```python
            _backward_phase11_group_loss(
                ...
                grpo_group_size=int(args.group_size),
            )
```

- [ ] **Step 4: Run tests**

```bash
cd /rds/general/user/aa6622/home/project
PYTHONPATH="${PWD}/src:${PYTHONPATH:-}" /rds/general/user/aa6622/home/.envs/lerobot_mw_py312/bin/python -m pytest \
  tests/test_phase11_true_action_chunking.py::test_phase11_group_loss_matches_closed_form_per_trajectory_normalizer \
  tests/test_phase11_true_action_chunking.py::test_phase11_group_loss_uses_explicit_grpo_group_size_not_rollout_count -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/grpo/train_phase11_env_on_policy_grpo.py tests/test_phase11_true_action_chunking.py
git commit -m "$(cat <<'EOF'
fix(grpo): phase11 loss uses explicit group_size

Multi-seed batching concatenates B×G rollouts; loss must normalize
per trajectory with group_size G, not len(rollouts).
EOF
)"
```

---

### Task 3: Rollout orchestrator `collect_rollout_seed_batch`

**Files:**
- Modify: `src/smolvla_grpo/phase11_rollout.py`
- Modify: `tests/test_phase11_true_action_chunking.py`

- [ ] **Step 1: Write the failing test**

```python
def test_collect_rollout_seed_batch_calls_group_per_seed(monkeypatch):
    calls = []

    def fake_collect_rollout_group(**kwargs):
        calls.append(int(kwargs["reset_seed"]))
        idx = len(calls) - 1
        return [
            RolloutTrajectory(reset_seed=int(kwargs["reset_seed"]), rollout_index=r)
            for r in range(int(kwargs["group_size"]))
        ]

    monkeypatch.setattr(phase11_rollout, "collect_rollout_group", fake_collect_rollout_group)
    out = phase11_rollout.collect_rollout_seed_batch(
        bundle=FakeBundle(),
        policy_old=FakeBundle().policy,
        task="push-v3",
        task_text="push",
        reset_seeds=[2000, 2001],
        episode_index=7,
        max_steps=10,
        group_size=3,
        action_dim=4,
        device=torch.device("cpu"),
    )
    assert calls == [2000, 2001]
    assert len(out) == 6
    assert [tr.reset_seed for tr in out] == [2000, 2000, 2000, 2001, 2001, 2001]
    assert [tr.metadata["seed_batch_index"] for tr in out] == [0, 0, 0, 1, 1, 1]
```

(Add `RolloutTrajectory` import from `smolvla_grpo.phase11_rollout`.)

- [ ] **Step 2: Run test — expect FAIL**

```bash
PYTHONPATH="${PWD}/src:${PYTHONPATH:-}" /rds/general/user/aa6622/home/.envs/lerobot_mw_py312/bin/python -m pytest \
  tests/test_phase11_true_action_chunking.py::test_collect_rollout_seed_batch_calls_group_per_seed -v
```

- [ ] **Step 3: Implement in `phase11_rollout.py`**

```python
def collect_rollout_seed_batch(
    *,
    bundle: _SmolVLABundle,
    policy_old: torch.nn.Module,
    task: str,
    task_text: str,
    reset_seeds: Sequence[int],
    episode_index: int,
    max_steps: int,
    group_size: int,
    action_dim: int,
    device: torch.device,
    env_backend: str = "custom",
    rollout_execution: str = "serial",
    async_start_method: str = "forkserver",
    action_transform: str = "no_tanh",
    action_chunk_size: int = 1,
    rollout_policy_batch_size: int = 32,
) -> list[RolloutTrajectory]:
    seeds = [int(s) for s in reset_seeds]
    if not seeds:
        raise ValueError("reset_seeds must be non-empty")
    if int(group_size) < 1:
        raise ValueError("group_size must be >= 1")
    merged: list[RolloutTrajectory] = []
    for batch_index, reset_seed in enumerate(seeds):
        group = collect_rollout_group(
            bundle=bundle,
            policy_old=policy_old,
            task=task,
            task_text=task_text,
            reset_seed=reset_seed,
            episode_index=episode_index,
            max_steps=max_steps,
            group_size=group_size,
            action_dim=action_dim,
            device=device,
            env_backend=env_backend,
            rollout_execution=rollout_execution,
            async_start_method=async_start_method,
            action_transform=action_transform,
            action_chunk_size=action_chunk_size,
            rollout_policy_batch_size=rollout_policy_batch_size,
        )
        for tr in group:
            tr.metadata["seed_batch_index"] = int(batch_index)
            tr.metadata["seed_batch_size"] = len(seeds)
        merged.extend(group)
    return merged
```

Add `from collections.abc import Sequence` if missing.

- [ ] **Step 4: Run test — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add src/smolvla_grpo/phase11_rollout.py tests/test_phase11_true_action_chunking.py
git commit -m "$(cat <<'EOF'
feat(grpo): collect phase11 rollouts per seed batch

Serial seed batches reuse collect_rollout_group so peak env count stays
at group_size instead of batch_size×group_size.
EOF
)"
```

---

### Task 4: Trainer wiring (`--batch-size`, seeds, advantages, logging)

**Files:**
- Modify: `scripts/grpo/train_phase11_env_on_policy_grpo.py`
- Modify: `tests/test_phase11_true_action_chunking.py` (static test for imports/strings)

- [ ] **Step 1: Write failing static test**

```python
def test_phase11_train_script_supports_seed_batch_size_static():
    text = (_REPO / "scripts/grpo/train_phase11_env_on_policy_grpo.py").read_text(encoding="utf-8")
    assert "collect_rollout_seed_batch" in text
    assert "compute_seed_batch_advantages" in text
    assert "reset_seeds" in text
    assert "Only batch_size=1 supported" not in text
```

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Trainer changes**

1. Import:

```python
from smolvla_grpo.grpo_math import compute_group_advantages, compute_seed_batch_advantages
from smolvla_grpo.phase11_rollout import collect_rollout_group, collect_rollout_seed_batch, load_bundle_for_grpo
```

2. Replace guard:

```python
    if int(args.batch_size) < 1:
        raise SystemExit("--batch-size must be >= 1")
```

3. Update `--batch-size` help text: `Number of reset seeds per update (each with group_size rollouts).`

4. In manifest add `"batch_size": int(args.batch_size)`.

5. Replace rollout block:

```python
        seed_batch_base = int(args.train_seed_base) + int(update) * int(args.batch_size)
        reset_seeds = [seed_batch_base + b for b in range(int(args.batch_size))]
        reset_seed = int(reset_seeds[0])  # legacy field: first seed of batch

        if int(args.batch_size) == 1:
            rollouts = collect_rollout_group(
                ...
                reset_seed=reset_seeds[0],
                ...
            )
        else:
            rollouts = collect_rollout_seed_batch(
                ...
                reset_seeds=reset_seeds,
                ...
            )
```

6. Replace advantages:

```python
        if int(args.batch_size) == 1:
            advantages = compute_group_advantages(returns)
        else:
            advantages = compute_seed_batch_advantages(
                returns, group_size=int(args.group_size)
            )
```

7. Progress row fields:

```python
                "reset_seeds": reset_seeds,
                "batch_size": int(args.batch_size),
```

8. Print line: `seeds={reset_seeds}` instead of only `seed=`.

- [ ] **Step 4: Run static + existing phase11 tests**

```bash
cd /rds/general/user/aa6622/home/project
PYTHONPATH="${PWD}/src:${PYTHONPATH:-}" /rds/general/user/aa6622/home/.envs/lerobot_mw_py312/bin/python -m pytest \
  tests/test_phase11_true_action_chunking.py \
  tests/test_grpo_math.py -q
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/grpo/train_phase11_env_on_policy_grpo.py tests/test_phase11_true_action_chunking.py
git commit -m "$(cat <<'EOF'
feat(grpo): phase11 multi-seed batch training

Each update can use batch_size reset seeds with independent GRPO
advantage groups while keeping group_size parallel rollouts per seed.
EOF
)"
```

---

### Task 5: PBS smoke + production scripts

**Files:**
- Create: `scripts/grpo/phase11_seedbatch_smoke_u2.pbs`
- Create: `scripts/grpo/phase11_seedbatch_b4_g32_train_0000_0050.pbs`
- Modify: `tests/test_phase11_slurm_scripts.py`

- [ ] **Step 1: Write failing PBS static test**

```python
def test_phase11_seedbatch_smoke_pbs_requests_resources() -> None:
    text = (_REPO / "scripts/grpo/phase11_seedbatch_smoke_u2.pbs").read_text(encoding="utf-8")
    assert "--batch-size 2" in text
    assert "--group-size 8" in text
    assert "phase11_cpu_mem_telemetry.sh" in text
    assert "run_phase11_with_cpu_mem_telemetry" in text
```

- [ ] **Step 2: Create smoke PBS** (pattern from `phase11_batched_logprob_smoke_u2.pbs`)

Key train args:

```bash
  --batch-size 2 \
  --group-size 8 \
  --num-updates 2 \
  --save-every 1 \
  --rollout-execution vector_async \
  --rollout-policy-batch-size 16 \
  --logprob-batch-size 16 \
  --action-chunk-size 5 \
  --train-seed-base 9000 \
```

Resources: `select=1:ncpus=48:mem=64gb:ngpus=1:gpu_type=RTX6000`, walltime `00:30:00`, new output dir `artifacts/phase11_pushv3_seedbatch_smoke_u2`.

Post-run assert in PBS footer:

```python
assert int(manifest["batch_size"]) == 2
assert "reset_seeds" in last
assert len(last["reset_seeds"]) == 2
```

- [ ] **Step 3: Create production PBS** `batch_size=4`, `group_size=32` (128 rollouts/update, 32 parallel envs peak)

Match hyperparams from best G32 runs: `lr=5e-6`, `clip_eps=0.1`, `action_chunk_size=5`, `mem=128gb` (same as G32; **not** 128 envs at once).

Walltime: start `24:00:00` (4× rollout per update vs G32); tune after smoke.

- [ ] **Step 4: Run static test — PASS**

- [ ] **Step 5: Commit**

```bash
git add scripts/grpo/phase11_seedbatch_*.pbs tests/test_phase11_slurm_scripts.py
git commit -m "$(cat <<'EOF'
chore(grpo): add phase11 seed-batch PBS scripts

Smoke batch2×G8; production batch4×G32 for 128 rollouts/update at G32 RAM.
EOF
)"
```

---

### Task 6: End-to-end smoke on cluster

- [ ] **Step 1: Submit smoke**

```bash
cd /rds/general/user/aa6622/home/project
qsub scripts/grpo/phase11_seedbatch_smoke_u2.pbs
```

- [ ] **Step 2: Gate checks after job completes**

```bash
OUT=artifacts/phase11_pushv3_seedbatch_smoke_u2
python - <<'PY'
import json
from pathlib import Path
rows = [json.loads(x) for x in Path("artifacts/phase11_pushv3_seedbatch_smoke_u2/progress.jsonl").read_text().splitlines() if x.strip()]
assert rows[-1]["batch_size"] == 2
assert len(rows[-1]["reset_seeds"]) == 2
assert len(rows[-1]["returns"]) == 16  # 2*8
assert rows[-1].get("skipped") is not True
print("ok", rows[-1]["success_rate"], rows[-1]["update_seconds"])
PY
```

Expected: `ok` printed; `update_seconds` finite; no OOM in PBS log (`Exit_status=137`).

- [ ] **Step 3: Compare RSS** — `process_tree_memory_summary.json` peak RSS should be closer to G8/G32 job than pop128 job.

- [ ] **Step 4: If smoke passes, queue production**

```bash
qsub scripts/grpo/phase11_seedbatch_b4_g32_train_0000_0050.pbs
```

---

## v2 (optional follow-up plan, not this PR)

| Idea | Benefit | Cost |
|------|---------|------|
| `collect_official_lerobot_vector_rollout_seed_batch` with `reset_many` + `n_envs=G` | Faster than B serial vector inits | Still B env creations unless env pooled |
| Reuse one `OfficialLeRobotMetaWorldGRPORollout` across seeds | Less forkserver churn | Careful `policy.reset()` between seeds |
| `n_envs=B×G` + grouped advantages | One rollout pass | **128GB+ RAM** — same pop128 problem |

---

## Self-review

| Requirement | Task |
|-------------|------|
| Per-seed GRPO normalization | Task 1 + Task 4 |
| Loss normalizer `n_units×G` with `B×G` rollouts | Task 2 |
| Peak RAM `G` not `B×G` | Task 3 serial orchestrator |
| Lift `batch_size==1` guard | Task 4 |
| Telemetry / CPU mem | Task 5 PBS wiring |
| Tests | Tasks 1–5 |
| Smoke + prod PBS | Tasks 5–6 |

No placeholders. Types consistent: `grpo_group_size` == CLI `group_size`; `batch_size` == `len(reset_seeds)`.

---

## Execution handoff

**Plan saved to** `docs/superpowers/plans/2026-05-19-phase11-multi-seed-grpo-batch.md`.

**Recommended execution: Inline Execution.**

Reason:
- GRPO math is delicate here: per-seed advantage groups and `n_units × group_size` normalizers must stay aligned across trainer, rollout, and tests.
- Scope is small enough for one context: `grpo_math.py`, `phase11_rollout.py`, `train_phase11_env_on_policy_grpo.py`, tests, and PBS scripts.
- Git hygiene is easier inline because current branch/worktree may contain unrelated dirty files.
- Subagents were useful for research; implementation should keep one reviewer responsible for invariants.

Execution style:
- Use small commits:
  1. Advantage helper.
  2. Explicit loss `grpo_group_size`.
  3. Rollout seed-batch orchestrator.
  4. Trainer wiring.
  5. PBS smoke/prod scripts.
- Run focused tests after each logical chunk.
- Queue smoke only after code passes.

Fallback:
- Use a readonly subagent only if implementation hits unclear existing behavior or test failures need independent investigation.
