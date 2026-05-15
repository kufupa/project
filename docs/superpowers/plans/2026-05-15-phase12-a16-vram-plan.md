# Phase12 A16 VRAM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Phase12 bounded WM-GRPO resume to update 300 on A16 by reducing peak CUDA graph memory, then run a 10-episode eval sweep for checkpoints 0, 10, 20, ..., 300.

**Architecture:** Keep math same, change memory schedule. Old-policy candidate sampling runs inference-only. Train-policy logprob recompute uses per-candidate microbatch backward, so each SmolVLA graph is freed before next candidate forward.

**Tech Stack:** PyTorch autograd, SmolVLA/LeRobot, Phase12 WM-GRPO trainer, Slurm `--export=NIL`, repo `common_env.sh`.

---

## Evidence

- Failed A16 job died in SmolVLA attention softmax during train-policy logprob recompute.
- Failing update had 76 frames -> 4 segments. With group size 4, trainer built 16 train-policy forward graphs, then did one `loss.backward()`.
- Previous shorter updates were 3 segments -> 12 graphs. A16 had no headroom with train policy + old policy + WM resident.
- PyTorch docs confirm separate forward/backward frees graph after each backward; gradients accumulate by default. `torch.inference_mode()` disables autograd recording for eval-only work and reduces overhead.

## File Structure

- Modify `scripts/grpo/train_phase12_wm_chunk_grpo.py`
  - Add CLI controls: `--logprob-backward-mode {stack,microbatch}`, `--old-policy-inference-mode`.
  - Add `_sample_old_action_chunk()` wrapper for old-policy sampling.
  - Add `_backward_chunk_grpo_loss_microbatched()` for per-row backward.
  - Keep stack path for parity/debug.
  - Remove ad-hoc mem-audit instrumentation from production path.
- Modify `tests/test_phase12_training_loop.py`
  - Add helper-level tests proving microbatch calls backward before next forward.
  - Add helper-level test proving old-policy sampling runs with grad disabled.
- Modify `scripts/grpo/submit_phase12_wm_chunk_grpo_train.slurm`
  - Default A16-safe env knobs: microbatch backward, old-policy inference, save every 10.
  - Keep positional resume/out behavior.
- Create `scripts/grpo/eval_phase12_checkpoint_sweep.py`
  - Evaluate update 0 via base checkpoint.
  - Evaluate update >0 via GRPO checkpoint.
  - Only eval requested stride 10 checkpoints.
- Create `scripts/grpo/submit_phase12_eval_sweep.slurm`
  - Sequential 10-episode eval sweep.
  - Designed for `sbatch --dependency=afterok:<train_job_id>` from login node.

---

### Task 1: Test Microbatch Backward

**Files:**
- Modify: `tests/test_phase12_training_loop.py`
- Modify: `scripts/grpo/train_phase12_wm_chunk_grpo.py`

- [ ] **Step 1: Write failing test**

Add test:

```python
def test_phase12_microbatch_backward_frees_each_logprob_graph_before_next_forward() -> None:
    class TrackGraph(torch.autograd.Function):
        active = 0
        max_active = 0

        @staticmethod
        def forward(ctx, weight, value):
            TrackGraph.active += 1
            TrackGraph.max_active = max(TrackGraph.max_active, TrackGraph.active)
            return weight * 0.0 + weight.new_tensor(float(value))

        @staticmethod
        def backward(ctx, grad_output):
            TrackGraph.active -= 1
            return torch.zeros_like(grad_output), None

    class Wrapper:
        def __init__(self, weight):
            self.weight = weight
            self.calls = 0

        def get_action_probs_for_chunk_from_proc(self, proc, chunk):
            del proc, chunk
            self.calls += 1
            return TrackGraph.apply(self.weight, -0.1 * self.calls)

    weight = torch.nn.Parameter(torch.tensor(1.0))
    wrapper = Wrapper(weight)
    old_lp = torch.zeros(4)
    advantages = torch.tensor([-1.0, -0.5, 0.5, 1.0])
    procs = [{"x": torch.zeros(1)} for _ in range(4)]
    chunks = [torch.zeros(1, 4) for _ in range(4)]

    loss, stats, new_lp = trainer._backward_chunk_grpo_loss_microbatched(
        train_wrapper=wrapper,
        proc_snapshots=procs,
        unsquashed_chunks=chunks,
        old_lp=old_lp,
        advantages=advantages,
        clip_eps=0.2,
    )

    assert wrapper.calls == 4
    assert TrackGraph.max_active == 1
    assert TrackGraph.active == 0
    assert isinstance(loss, float)
    assert len(new_lp) == 4
    assert "ratio_mean" in stats
```

- [ ] **Step 2: Verify RED**

Run: `PYTHONPATH=src:. pytest tests/test_phase12_training_loop.py::test_phase12_microbatch_backward_frees_each_logprob_graph_before_next_forward -q`

Expected: FAIL because `_backward_chunk_grpo_loss_microbatched` missing.

- [ ] **Step 3: Implement helper**

Add helper in trainer:

```python
def _chunk_grpo_row_loss(new_lp, old_lp, advantage, *, clip_eps: float, normalizer: int):
    ratio = torch.exp(new_lp.float() - old_lp.float())
    clipped = torch.clamp(ratio, 1.0 - float(clip_eps), 1.0 + float(clip_eps))
    loss = -torch.minimum(ratio * advantage.float(), clipped * advantage.float()) / float(normalizer)
    return loss, ratio


def _ratio_stats_from_tensors(old_lp, new_lp, *, clip_eps: float) -> dict[str, float]:
    ratio = torch.exp(new_lp.float() - old_lp.float())
    low = 1.0 - float(clip_eps)
    high = 1.0 + float(clip_eps)
    clip_fraction = ((ratio < low) | (ratio > high)).float().mean()
    return {
        "ratio_mean": float(ratio.mean().item()),
        "ratio_max": float(ratio.max().item()),
        "ratio_min": float(ratio.min().item()),
        "ratio_clip_fraction": float(clip_fraction.item()),
        "approx_kl": float((old_lp.float() - new_lp.float()).mean().item()),
    }
```

Then implement microbatch helper using one forward, one backward, detach stats, repeat.

- [ ] **Step 4: Verify GREEN**

Run same test. Expected: PASS.

---

### Task 2: Test Old-Policy Inference

**Files:**
- Modify: `tests/test_phase12_training_loop.py`
- Modify: `scripts/grpo/train_phase12_wm_chunk_grpo.py`

- [ ] **Step 1: Write failing test**

Add test:

```python
def test_phase12_old_policy_sampling_uses_inference_mode_when_enabled() -> None:
    seen = {}

    class Wrapper:
        bundle = SimpleNamespace(device=torch.device("cpu"))

        def sample_action_chunk_from_proc(self, proc, *, chunk_len, rng):
            del proc, chunk_len, rng
            seen["grad_enabled"] = torch.is_grad_enabled()
            seen["inference_mode"] = torch.is_inference_mode_enabled()
            return SimpleNamespace(
                unsquashed_chunk=torch.zeros(2, 4),
                log_prob_steps=torch.zeros(2),
                log_prob_sum=torch.tensor(0.0),
                exec_action_np=np.zeros((2, 4), dtype=np.float32),
                action_clip_fraction=np.zeros(2),
                action_clip_any=np.zeros(2, dtype=bool),
                unique_action_rows=1,
            )

    sample = trainer._sample_old_action_chunk(
        Wrapper(),
        {"x": torch.zeros(1)},
        chunk_len=2,
        rng=torch.Generator(device="cpu"),
        use_inference_mode=True,
    )

    assert sample.unique_action_rows == 1
    assert seen == {"grad_enabled": False, "inference_mode": True}
```

- [ ] **Step 2: Verify RED**

Run: `PYTHONPATH=src:. pytest tests/test_phase12_training_loop.py::test_phase12_old_policy_sampling_uses_inference_mode_when_enabled -q`

Expected: FAIL because helper missing.

- [ ] **Step 3: Implement helper and use in sampler**

In `collect_phase12_training_episode.sampler`, replace direct old wrapper call with:

```python
sample = _sample_old_action_chunk(
    old_wrapper,
    proc,
    chunk_len=int(args.chunk_len),
    rng=gen,
    use_inference_mode=bool(args.old_policy_inference_mode),
)
```

- [ ] **Step 4: Verify GREEN**

Run old-policy inference test. Expected: PASS.

---

### Task 3: Wire CLI and Train Loop

**Files:**
- Modify: `scripts/grpo/train_phase12_wm_chunk_grpo.py`
- Modify: `tests/test_phase12_trainer_static.py`

- [ ] **Step 1: Add static/default tests**

Assert parser exposes:

```python
assert args.logprob_backward_mode == "stack"
assert args.old_policy_inference_mode is True
```

Assert manifest records:

```python
assert manifest["logprob_backward_mode"] == "stack"
assert manifest["old_policy_inference_mode"] is True
```

- [ ] **Step 2: Verify RED**

Run: `PYTHONPATH=src:. pytest tests/test_phase12_trainer_static.py::test_phase12_cli_defaults tests/test_phase12_trainer_static.py::test_manifest_records_phase12_contract -q`

Expected: FAIL.

- [ ] **Step 3: Implement CLI and loop**

Add args:

```python
p.add_argument("--logprob-backward-mode", choices=("stack", "microbatch"), default="stack")
p.add_argument("--old-policy-inference-mode", action=argparse.BooleanOptionalAction, default=True)
```

In train loop:

- Compute rewards/advantages.
- If all advantages zero, save skip checkpoint before train forward.
- If mode `microbatch`, call microbatch helper after `optimizer.zero_grad(set_to_none=True)`.
- If mode `stack`, keep existing stack + single backward behavior.
- Clip grad + optimizer step same in both modes.
- Sync old policy after optimizer step.

- [ ] **Step 4: Verify tests**

Run focused tests. Expected: PASS.

---

### Task 4: A16 Resume Slurm

**Files:**
- Modify: `scripts/grpo/submit_phase12_wm_chunk_grpo_train.slurm`

- [ ] **Step 1: Add script controls**

Add:

```bash
BACKWARD_MODE="${PHASE12_BACKWARD_MODE:-microbatch}"
OLD_POLICY_INFERENCE="${PHASE12_OLD_POLICY_INFERENCE:-1}"
```

Pass:

```bash
--logprob-backward-mode "${BACKWARD_MODE}"
```

And:

```bash
if [[ "${OLD_POLICY_INFERENCE}" == "1" ]]; then
  EXTRA+=(--old-policy-inference-mode)
else
  EXTRA+=(--no-old-policy-inference-mode)
fi
```

- [ ] **Step 2: Validate syntax**

Run: `bash -n scripts/grpo/submit_phase12_wm_chunk_grpo_train.slurm`

Expected: no output, exit 0.

---

### Task 5: Phase12 Eval Sweep

**Files:**
- Create: `scripts/grpo/eval_phase12_checkpoint_sweep.py`
- Create: `scripts/grpo/submit_phase12_eval_sweep.slurm`

- [ ] **Step 1: Implement sweep script**

Behavior:

- Args: base checkpoint, run dir, task, episodes, eval seed, min update, max update, stride, sweep name.
- Update 0 evals base checkpoint with no GRPO load.
- Updates >0 require `run_dir/checkpoints/update_%04d.pt`.
- Only evaluate multiples of stride.
- Output `eval_sweep_summary.json`.

- [ ] **Step 2: Implement Slurm wrapper**

Use `common_env.sh`, offline HF defaults, `MUJOCO_GL=egl`, then run sweep script.

- [ ] **Step 3: Syntax checks**

Run:

```bash
python -m py_compile scripts/grpo/eval_phase12_checkpoint_sweep.py
bash -n scripts/grpo/submit_phase12_eval_sweep.slurm
```

Expected: PASS.

---

### Task 6: Verify and Submit

**Files:**
- All touched files.

- [ ] **Step 1: Run focused CPU tests**

Run:

```bash
PYTHONPATH=src:. pytest \
  tests/test_phase12_training_loop.py \
  tests/test_phase12_trainer_static.py \
  tests/test_phase12_pixels.py \
  tests/test_grpo_lerobot_adapter.py \
  -q
```

Expected: PASS.

- [ ] **Step 2: Run Slurm test-only**

Run:

```bash
sbatch --test-only --chdir=/vol/bitbucket/aa6622/project --export=NIL scripts/grpo/submit_phase12_wm_chunk_grpo_train.slurm
sbatch --test-only --chdir=/vol/bitbucket/aa6622/project --export=NIL scripts/grpo/submit_phase12_eval_sweep.slurm
```

Expected: both accepted.

- [ ] **Step 3: Cancel stale A40 if A16 path ready**

Run only after tests pass:

```bash
scancel 241217
```

- [ ] **Step 4: Submit A16 resume to 300**

Run from login node:

```bash
jid=$(sbatch --parsable \
  --chdir=/vol/bitbucket/aa6622/project \
  --export=NIL \
  --partition=a16 \
  scripts/grpo/submit_phase12_wm_chunk_grpo_train.slurm \
  bounded_executed \
  280 \
  /vol/bitbucket/aa6622/project/artifacts/phase12_wm_chunk_grpo_train/push-v3/wm_view_fix_bounded_u100_seed2000 \
  /vol/bitbucket/aa6622/project/artifacts/phase12_wm_chunk_grpo_train/push-v3/wm_view_fix_bounded_u100_seed2000/checkpoints/latest.pt)
echo "${jid}"
```

Rationale: latest checkpoint is update 20, so 280 more updates reaches update 300.

- [ ] **Step 5: Chain eval sweep after train**

Run:

```bash
eval_jid=$(sbatch --parsable \
  --chdir=/vol/bitbucket/aa6622/project \
  --export=NIL \
  --dependency=afterok:${jid} \
  --partition=a16 \
  scripts/grpo/submit_phase12_eval_sweep.slurm \
  /vol/bitbucket/aa6622/project/artifacts/phase12_wm_chunk_grpo_train/push-v3/wm_view_fix_bounded_u100_seed2000 \
  bounded_eval_every10_u0_u300 \
  0 \
  300 \
  10 \
  10)
echo "${eval_jid}"
```

- [ ] **Step 6: Verify queue**

Run:

```bash
squeue -j "${jid},${eval_jid}" -o "%.18i %.9P %.40j %.2t %.10M %.20S %R"
```

Expected: train pending/running on `a16`; eval pending with `Dependency`.

---

## Self-Review

- Spec coverage: VRAM reduction, A16 resume, update 300 target, afterok eval sweep every 10th checkpoint including 0 covered.
- Placeholder scan: no TODO/TBD placeholders.
- Type consistency: helper names used in tests match implementation tasks.
