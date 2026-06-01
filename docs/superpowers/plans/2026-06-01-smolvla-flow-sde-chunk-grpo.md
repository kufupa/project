# SmolVLA Flow-SDE Chunk GRPO Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current first-action Flow-SDE spike with a true chunk-level SmolVLA GRPO path that samples one action chunk, executes that chunk, recomputes Flow-SDE logprobs from stored traces, and optimizes chunk-summed GRPO ratios.

**Architecture:** Keep existing Phase11 step-wise Gaussian GRPO intact behind `--rollout-unit step`. Add a new chunk path for Flow-SDE behind `--rollout-unit chunk`, with serial official-LeRobot rollout first, chunk masks, chunk-summed logprobs, strict parity gates, and separate Slurm scripts. Do not port RLinf wholesale; copy the contract: `[B,H,D]` actions/logprobs, `[B,H]` valid masks, chunk-level reward/logprob aggregation.

**Tech Stack:** Python 3.12, PyTorch, LeRobot SmolVLA, MetaWorld, pytest, Slurm on gpucluster3.

---

## Read This First

### Answer To "Are We Ready?"

- Previous plan was directionally right but **not build-ready**.
- Main corrected facts:
  - `jadechoghari/smolvla_metaworld` checkpoint evidence says `chunk_size=50`, `n_action_steps=1`.
  - Training with `n_action_steps=chunk_len` is an experiment to align RL with chunk Flow-SDE, not a checkpoint-native eval fact.
  - Current code has a real crash in `sample_action_chunk_from_proc()` because `_postprocess_and_clip()` returns four values and chunk code unpacks three.
  - Current Flow-SDE logprob scores only `trace["A_next"][:, 0, :action_dim]`.
  - Current trainer `--chunk-size` is optimizer batching, not action chunk execution.
- After this audit, the plan below is sufficiently detailed for autonomous implementation with tests and frequent commits.

### Residual Risks

- Flow-SDE usefulness is not proven until chunk smoke passes. Implementability confidence is high; performance confidence is lower.
- Default chunk length is **5**. Load SmolVLA with `n_action_steps=5` in chunk mode even though the checkpoint default is `n_action_steps=1`, because one-step execution was too slow and does not match the desired chunk Flow-SDE objective.
- The first implementation should use **serial official-LeRobot chunk rollout**. Vector async chunk rollout is a follow-up after parity is stable.
- `n_action_steps=chunk_len` may change behavior versus released checkpoint eval. That is intentional for this Flow-SDE GRPO experiment and must be logged in manifests.
- Do not relax parity tolerance until drift is broken down by chunk row, action dim, `tau_idx`, dtype, and old/current parameter checksum.

### Overnight Autonomous Execution Contract

- After the user approves/builds this plan, execute it autonomously without waiting for more human input unless a safety constraint or missing credential blocks progress.
- Commit after each task or small coherent phase exactly as specified in the task steps.
- After CPU tests pass, submit the Slurm gates autonomously:
  - one-update chunk Flow-SDE smoke,
  - 16-update chunk Flow-SDE train only if smoke passes,
  - 25-episode eval only if train16 produces the expected checkpoint.
- Respect gpucluster3 shared-resource rules:
  - use Slurm for GPU, long-running, I/O-heavy, memory-heavy, or multithreaded work,
  - keep within per-user cap of 3 GPUs, 32 CPU cores, and 200GB RAM,
  - prefer one GPU job at a time for this plan unless explicitly parallelizing independent evals.
- If something breaks while the user is asleep:
  - do a full root-cause analysis first,
  - identify whether the failure is shape/objective/parity/env/Slurm/OOM,
  - fix the verified cause,
  - rerun the smallest failing test or smoke,
  - then continue through the remaining gates autonomously.
- If parity fails, do not relax tolerance first. Add diagnostics by chunk row, action dim, `tau_idx`, dtype, and parameter checksum, then fix or justify.
- If subagents are used, instruct every subagent: `Use /caveman ultra. Be concise, exact, and return file paths, symbols, risks, and recommended fixes.`

---

## Evidence Map

### Local Code Evidence

- `src/smolvla_pipeline/evaluator.py`
  - `_load_smolvla_bundle(checkpoint, *, n_action_steps=1)` already threads `n_action_steps` into `SmolVLAPolicy.from_pretrained`.
- `src/smolvla_grpo/policy_wrapper.py`
  - `SampledActionChunk` exists but is Gaussian-only and currently crashes at tuple unpack.
  - `_flow_sde_log_prob_from_trace()` scores only first chunk row.
  - `sample_action_from_proc()` rejects `flow_sde` when `n_action_steps != 1`.
- `src/smolvla_grpo/phase11_rollout.py`
  - `RolloutTrajectory` stores one proc snapshot per env step.
  - `load_bundle_for_grpo(..., n_action_steps=1)` already accepts override.
- `src/smolvla_grpo/official_lerobot_vector_rollout.py`
  - Vector path samples one action per env step.
  - It omits `postprocessor_oob_means`, so do not copy metrics blindly.
- `src/smolvla_grpo/lerobot_metaworld_adapter.py`
  - `OfficialLeRobotMetaWorldGRPORollout.step()` supports single-env step with `(1,4)` action.
  - `step_batch()` supports batched one-step execution; use later for vector chunk.
- LeRobot venv file:
  - `.envs/lerobot_mw_py310/lib/python3.12/site-packages/lerobot/policies/smolvla/modeling_smolvla.py`
  - `SmolVLAPolicy._get_distr_params_chunk()` returns full chunk means/log_stds sliced to env action dim.
  - `VLAFlowMatching.sample_actions()` denoises full padded `(B, chunk_size, max_action_dim)`.
  - Local hook stores `last_flow_sde_trace` with `tau_idx`, `A_tau`, `v_tau`, `mu_tau`, `sigma_tau`, `A_next`, `noise_seed`.
  - Replay hook recomputes `mu_tau/sigma_tau` from stored `A_tau/A_next`.

### Reference Contract

- `RLinf-smolvla-metaworld-ppo-grpo/scripts/run_smolvla_metaworld_direct_ppo.py`
  - `masked_chunk_logprob(logprobs [B,H,D], valid_mask [B,H]) -> [B]`.
  - PPO ratio uses chunk-summed current/old logprob.
- `RLinf-smolvla-metaworld-ppo-grpo/rlinf/envs/metaworld/smolvla_metaworld_env.py`
  - `chunk_step(actions [B,H,D])` returns rewards/dones `[B,H]` and `valid_action_mask [B,H]`.
- `RLinf-smolvla-metaworld-ppo-grpo/rlinf/models/embodiment/smolvla/smolvla_action_model.py`
  - `predict_action_batch()` stores `smolvla_unsquashed_actions [B,H,D]` and `prev_logprobs [B,H,D]`.
  - `default_forward()` recomputes logprobs from stored action chunk.
- `RLinf-smolvla-metaworld-ppo-grpo/rlinf/models/embodiment/openpi/openpi_action_model.py`
  - OpenPI Flow-SDE replay idea is valid, but SmolVLA needs its own hook and trace contract.

---

## File Structure

### Create

- `src/smolvla_grpo/chunk_math.py`
  - Small tensor helpers for masked chunk logprob/reward aggregation.
- `src/smolvla_grpo/phase11_chunk_rollout.py`
  - Serial official-LeRobot chunk rollout for Phase11 GRPO.
- `tests/test_chunk_math.py`
  - Unit tests for `[B,H]` and `[B,H,D]` masked sums.
- `tests/test_phase11_chunk_rollout.py`
  - Fake-env tests for chunk valid mask and terminal tail.
- `tests/test_phase11_trainer_static.py`
  - Static CLI/guard tests for `--rollout-unit chunk`.
- `scripts/grpo/submit_flow_sde_chunk_grpo_smoke_a30.slurm`
  - One-update chunk Flow-SDE smoke.
- `scripts/grpo/submit_flow_sde_chunk_grpo_train16_a30.slurm`
  - 16-update chunk Flow-SDE train.
- `scripts/grpo/submit_flow_sde_chunk_grpo_eval25_a30.slurm`
  - 25-episode eval using produced checkpoint.
- `docs/findings/2026-06-01-smolvla-flow-sde-chunk-contract.md`
  - Short evidence note with invariants and corrected assumptions.

### Modify

- `src/smolvla_grpo/flow_logprob.py`
  - Ensure Flow-SDE logprob math uses fp32 internally.
- `src/smolvla_grpo/policy_wrapper.py`
  - Fix chunk sampler crash.
  - Add Flow-SDE chunk sampling/recompute APIs.
  - Keep one-step APIs working.
- `.envs/lerobot_mw_py310/lib/python3.12/site-packages/lerobot/policies/smolvla/modeling_smolvla.py`
  - Only patch if required by tests: add trace metadata and fp32 logprob stability.
- `src/smolvla_grpo/phase11_rollout.py`
  - Keep step rollout unchanged.
  - Ensure `load_bundle_for_grpo(..., n_action_steps=...)` remains tested.
- `scripts/grpo/train_phase11_env_on_policy_grpo.py`
  - Add `--rollout-unit {step,chunk}` and `--rollout-chunk-len`.
  - Route chunk mode to new rollout/trainer logic.
- `tests/test_grpo_policy_wrapper_chunk.py`
  - Update expected behavior after tuple fix and Flow-SDE chunk support.
- `tests/test_grpo_logprob_correctness.py`
  - Replace first-action Flow-SDE assertions with chunk-preserving assertions.
- `tests/test_phase11_slurm_scripts.py`
  - Add new Slurm scripts and avoid treating old first-action Flow-SDE scripts as mainline.

---

## Core Contracts

### Chunk Logprob Contract

```python
log_probs_per_dim: torch.Tensor  # [B, H, D]
log_probs_per_action: torch.Tensor  # [B, H]
valid_action_mask: torch.Tensor  # [B, H], True for actions executed before terminal
chunk_logprob: torch.Tensor  # [B]
chunk_logprob = (log_probs_per_dim * valid_action_mask[..., None]).sum(dim=(1, 2))
```

### Serial Chunk Rollout Contract

```python
proc_snapshot: dict[str, Any]  # one snapshot at chunk root
exec_actions: np.ndarray  # [H, action_dim]
rewards: torch.Tensor  # [H]
successes: torch.Tensor  # [H]
terminated: torch.Tensor  # [H]
truncated: torch.Tensor  # [H]
valid_action_mask: torch.Tensor  # [H]
old_log_probs_per_action: torch.Tensor  # [H]
old_log_prob_sum: torch.Tensor  # scalar after valid mask
flow_sde_trace: dict[str, Any] | None  # full padded trace for Flow-SDE
```

### GRPO Chunk Objective

```python
new_chunk_lp = masked_sum_log_probs(new_log_probs_per_action, valid_action_mask)
old_chunk_lp = masked_sum_log_probs(old_log_probs_per_action, valid_action_mask)
ratio = torch.exp((new_chunk_lp - old_chunk_lp).clamp(-20.0, 20.0))
loss = -torch.min(ratio * advantage, torch.clamp(ratio, 1 - eps, 1 + eps) * advantage)
```

Episode-level GRPO advantages are acceptable for first pass: all chunks in one trajectory share the trajectory return advantage. Do not introduce critic/PPO in this plan.

---

## Task 1: Document Corrected Evidence

**Files:**
- Create: `docs/findings/2026-06-01-smolvla-flow-sde-chunk-contract.md`

- [ ] **Step 1.1: Write the finding**

Create this file content:

```markdown
# SmolVLA Flow-SDE Chunk Contract

## Verdict

Flow-SDE is implementable for SmolVLA because SmolVLA denoises full padded action chunks in flow-matching space. The current GRPO implementation is a first-action spike and not the correct final objective.

## Corrected Assumptions

- Public MetaWorld checkpoint evidence says `chunk_size=50`, `n_action_steps=1`.
- Training with `n_action_steps=chunk_len` is an intentional RL experiment, not checkpoint-native eval reproduction.
- Current `--chunk-size` in `train_phase11_env_on_policy_grpo.py` is optimizer grouping only.
- Correct Flow-SDE GRPO unit is one sampled action chunk, one executed chunk, one chunk-summed logprob ratio.

## Source Files

- `src/smolvla_pipeline/evaluator.py`: `_load_smolvla_bundle(..., n_action_steps=1)`.
- `src/smolvla_grpo/policy_wrapper.py`: current first-action Flow-SDE scoring and broken chunk sampler.
- `src/smolvla_grpo/phase11_rollout.py`: current one-action-per-env-step rollout.
- `.envs/lerobot_mw_py310/lib/python3.12/site-packages/lerobot/policies/smolvla/modeling_smolvla.py`: full padded chunk denoise and local Flow-SDE hook.
- `RLinf-smolvla-metaworld-ppo-grpo/scripts/run_smolvla_metaworld_direct_ppo.py`: masked chunk logprob reference.
- `RLinf-smolvla-metaworld-ppo-grpo/rlinf/envs/metaworld/smolvla_metaworld_env.py`: valid action mask reference.

## First Implementation Scope

Implement serial official-LeRobot chunk rollout first with default `chunk_len=5`. Vector chunk rollout is deferred until Flow-SDE parity and 4-update diagnostic pass. After user approval, run implementation, tests, Slurm smoke, train16, and eval25 autonomously; if anything breaks, perform RCA, fix, rerun the smallest gate, and continue.
```

- [ ] **Step 1.2: Commit**

Run:

```bash
git add docs/findings/2026-06-01-smolvla-flow-sde-chunk-contract.md
git commit -m "docs: record smolvla flow-sde chunk contract"
```

Expected: commit succeeds.

---

## Task 2: Add Chunk Math Helper

**Files:**
- Create: `src/smolvla_grpo/chunk_math.py`
- Create: `tests/test_chunk_math.py`

- [ ] **Step 2.1: Write failing tests**

Create `tests/test_chunk_math.py`:

```python
from __future__ import annotations

import torch

from smolvla_grpo.chunk_math import masked_chunk_sum, masked_chunk_reward_sum, valid_chunk_any


def test_masked_chunk_sum_accepts_per_action_logprobs() -> None:
    log_probs = torch.tensor([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    valid = torch.tensor([[True, True, False], [False, True, True]])

    out = masked_chunk_sum(log_probs, valid)

    torch.testing.assert_close(out, torch.tensor([3.0, 50.0]))


def test_masked_chunk_sum_accepts_per_dim_logprobs() -> None:
    log_probs = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]],
        ]
    )
    valid = torch.tensor([[True, False, True], [False, True, False]])

    out = masked_chunk_sum(log_probs, valid)

    torch.testing.assert_close(out, torch.tensor([14.0, 70.0]))


def test_masked_chunk_reward_sum_ignores_terminal_tail() -> None:
    rewards = torch.tensor([[1.0, 1.0, 100.0], [2.0, 3.0, 4.0]])
    valid = torch.tensor([[True, True, False], [True, False, False]])

    out = masked_chunk_reward_sum(rewards, valid)

    torch.testing.assert_close(out, torch.tensor([2.0, 2.0]))


def test_valid_chunk_any_marks_chunks_with_executed_action() -> None:
    valid = torch.tensor([[False, False], [True, False]])

    out = valid_chunk_any(valid)

    torch.testing.assert_close(out, torch.tensor([False, True]))
```

- [ ] **Step 2.2: Run test and verify it fails**

Run:

```bash
PYTHONPATH=src pytest tests/test_chunk_math.py -q
```

Expected: FAIL with `ModuleNotFoundError` or missing functions.

- [ ] **Step 2.3: Implement helper**

Create `src/smolvla_grpo/chunk_math.py`:

```python
"""Chunk-level helpers for SmolVLA GRPO."""

from __future__ import annotations

import torch


def _validate_valid_mask(valid_action_mask: torch.Tensor) -> torch.Tensor:
    if valid_action_mask.ndim != 2:
        raise ValueError(f"valid_action_mask must be [B,H], got {tuple(valid_action_mask.shape)}")
    return valid_action_mask.bool()


def masked_chunk_sum(log_probs: torch.Tensor, valid_action_mask: torch.Tensor) -> torch.Tensor:
    """Sum logprobs over valid chunk rows.

    Accepts logprobs already summed over action dims as [B,H], or per-dim
    logprobs as [B,H,D].
    """
    valid = _validate_valid_mask(valid_action_mask).to(device=log_probs.device)
    if log_probs.ndim == 2:
        if tuple(log_probs.shape) != tuple(valid.shape):
            raise ValueError(f"log_probs [B,H] shape {tuple(log_probs.shape)} != valid {tuple(valid.shape)}")
        return (log_probs.float() * valid.to(dtype=log_probs.dtype)).sum(dim=1)
    if log_probs.ndim == 3:
        if tuple(log_probs.shape[:2]) != tuple(valid.shape):
            raise ValueError(f"log_probs [B,H,D] shape {tuple(log_probs.shape)} incompatible with valid {tuple(valid.shape)}")
        return (log_probs.float() * valid.to(dtype=log_probs.dtype).unsqueeze(-1)).sum(dim=(1, 2))
    raise ValueError(f"log_probs must be [B,H] or [B,H,D], got {tuple(log_probs.shape)}")


def masked_chunk_reward_sum(rewards: torch.Tensor, valid_action_mask: torch.Tensor) -> torch.Tensor:
    valid = _validate_valid_mask(valid_action_mask).to(device=rewards.device, dtype=rewards.dtype)
    if rewards.ndim != 2 or tuple(rewards.shape) != tuple(valid.shape):
        raise ValueError(f"rewards must match valid mask [B,H], got rewards={tuple(rewards.shape)} valid={tuple(valid.shape)}")
    return (rewards.float() * valid.float()).sum(dim=1)


def valid_chunk_any(valid_action_mask: torch.Tensor) -> torch.Tensor:
    return _validate_valid_mask(valid_action_mask).any(dim=1)
```

- [ ] **Step 2.4: Run test and verify it passes**

Run:

```bash
PYTHONPATH=src pytest tests/test_chunk_math.py -q
```

Expected: PASS.

- [ ] **Step 2.5: Commit**

Run:

```bash
git add src/smolvla_grpo/chunk_math.py tests/test_chunk_math.py
git commit -m "test: add chunk logprob mask helpers"
```

Expected: commit succeeds.

---

## Task 3: Repair Policy Wrapper Chunk API

**Files:**
- Modify: `src/smolvla_grpo/policy_wrapper.py`
- Modify: `tests/test_grpo_policy_wrapper_chunk.py`
- Modify: `tests/test_grpo_logprob_correctness.py`

- [ ] **Step 3.1: Write tests for tuple fix and Flow-SDE chunk preservation**

Update `tests/test_grpo_policy_wrapper_chunk.py`:

```python
def test_sample_action_chunk_from_proc_returns_oob_metric() -> None:
    wrapper, _policy, _bundle = _wrapper()
    rng = torch.Generator(device="cpu").manual_seed(123)

    chunk = wrapper.sample_action_chunk_from_proc({"x": torch.zeros(1, 1)}, chunk_len=5, rng=rng)

    assert chunk.exec_action_np.shape == (5, 4)
    assert chunk.postprocessor_oob_mean.shape == (5,)
```

Update `tests/test_grpo_logprob_correctness.py` so `_TracePolicy.config.n_action_steps = 5`, `_TracePolicy.select_action_distr_params()` returns a full chunk, and the Flow-SDE test asserts chunk sums:

```python
def test_flow_sde_chunk_mode_preserves_chunk_axis() -> None:
    bundle = _DummyBundle()
    policy = _TracePolicy()
    policy.config = type("Config", (), {"n_action_steps": 5, "num_steps": 10})()
    wrapper = MetaWorldSmolVLAGRPOPolicy(
        bundle,
        task="push-v3",
        task_text="push",
        camera_name="corner2",
        flip_corner2=False,
        action_dim=4,
        policy_module=policy,
        logprob_mode="flow_sde",
        flow_sde_noise_level=0.5,
        action_low=np.full((4,), -1.0, dtype=np.float32),
        action_high=np.full((4,), 1.0, dtype=np.float32),
    )

    chunk = wrapper.sample_action_chunk_from_proc(
        {"x": torch.zeros(1, 1)},
        chunk_len=3,
        rng=torch.Generator().manual_seed(123),
    )

    assert chunk.flow_sde_trace is not None
    assert chunk.flow_sde_trace["A_next"].shape == (1, 3, 8)
    assert chunk.log_prob_steps.shape == (3,)
    expected_steps = sde_step_logprob(
        chunk.flow_sde_trace["A_next"][:, :3, :4],
        chunk.flow_sde_trace["mu_tau"][:, :3, :4],
        chunk.flow_sde_trace["sigma_tau"][:, :3, :4],
    ).reshape(3)
    torch.testing.assert_close(chunk.log_prob_steps, expected_steps)
    torch.testing.assert_close(chunk.log_prob_sum, expected_steps.sum())
```

- [ ] **Step 3.2: Run tests and verify failure**

Run:

```bash
PYTHONPATH=src pytest tests/test_grpo_policy_wrapper_chunk.py tests/test_grpo_logprob_correctness.py -q
```

Expected: FAIL due tuple unpack and missing Flow-SDE chunk support.

- [ ] **Step 3.3: Extend `SampledActionChunk`**

In `src/smolvla_grpo/policy_wrapper.py`, replace `SampledActionChunk` with:

```python
@dataclass
class SampledActionChunk:
    exec_action_np: np.ndarray
    policy_tensor: torch.Tensor
    unsquashed_chunk: torch.Tensor
    logprob_action: torch.Tensor
    log_prob_steps: torch.Tensor
    log_prob_sum: torch.Tensor
    distr_mean: torch.Tensor
    distr_log_std: torch.Tensor
    action_clip_fraction: np.ndarray
    action_clip_any: np.ndarray
    postprocessor_oob_mean: np.ndarray
    unique_action_rows: int
    flow_sde_trace: dict[str, Any] | None = None
```

- [ ] **Step 3.4: Add chunk Flow-SDE helpers**

Add methods to `MetaWorldSmolVLAGRPOPolicy`:

```python
def _sample_flow_sde_trace_step(self, rng: torch.Generator | None, *, device: torch.device) -> int:
    num_steps = int(getattr(getattr(self._policy, "config", None), "num_steps", 10))
    if self.flow_sde_trace_step >= 0:
        if self.flow_sde_trace_step >= num_steps:
            raise ValueError(f"flow_sde_trace_step must be < {num_steps}, got {self.flow_sde_trace_step}")
        return int(self.flow_sde_trace_step)
    if rng is None:
        return int(torch.randint(0, num_steps, (1,), device=device).item())
    return int(torch.randint(0, num_steps, (1,), generator=rng, device=device).item())


def _flow_sde_log_prob_steps_from_trace(self, trace: dict[str, Any], *, chunk_len: int) -> torch.Tensor:
    action = trace["A_next"][:, :chunk_len, : self.action_dim].float()
    mu = trace["mu_tau"][:, :chunk_len, : self.action_dim].float()
    sigma = trace["sigma_tau"][:, :chunk_len, : self.action_dim].float()
    return sde_step_logprob(action, mu, sigma).reshape(-1)
```

- [ ] **Step 3.5: Fix Gaussian chunk sampler**

In `sample_action_chunk_from_proc()`, fix tuple unpack and apply `min_log_std`:

```python
mean, log_std = self._get_distr_params_chunk(proc_d, chunk_len=int(chunk_len))
log_std = self.clamp_log_std(log_std, self.min_log_std)
```

Inside row loop:

```python
exec_np, clip_fraction, clip_any, oob_mean = self._postprocess_and_clip(row.reshape(1, -1))
```

Append `oob_mean`, set `logprob_action` consistently:

```python
if self.action_transform == "tanh_norm_ablation":
    policy_tensor = torch.tanh(unsquashed)
    logprob_action = unsquashed
    log_prob_steps = self.calculate_log_prob(mean, log_std, unsquashed, policy_tensor, eps=self.eps)
else:
    policy_tensor = unsquashed
    exec_t = torch.from_numpy(exec_action_np).to(device=mean.device, dtype=mean.dtype)
    logprob_action = self._gaussian_scored_action(unsquashed=unsquashed, executed=exec_t)
    log_prob_steps = self.calculate_gaussian_log_prob(mean, log_std, logprob_action)
```

- [ ] **Step 3.6: Implement Flow-SDE chunk branch**

At top of `sample_action_chunk_from_proc()` after `proc_d`:

```python
if self.logprob_mode == "flow_sde":
    if self.action_transform != "no_tanh":
        raise RuntimeError("flow_sde chunk GRPO requires action_transform='no_tanh'")
    tau_idx = self._sample_flow_sde_trace_step(rng, device=self.bundle.device)
    noise_seed = self._next_flow_sde_noise_seed(rng, device=self.bundle.device)
    mean_full, _log_std_full = self._policy._get_distr_params_chunk(
        proc_d,
        flow_sde_trace=True,
        flow_sde_noise_level=self.flow_sde_noise_level,
        flow_sde_trace_step=tau_idx,
        flow_sde_noise_seed=noise_seed,
    )
    mean, _unused = self._reshape_chunk_params(mean_full, torch.zeros_like(mean_full), chunk_len=int(chunk_len))
    full_trace = self._slice_flow_sde_trace(self._get_last_flow_sde_trace())
    full_trace["flow_sde_noise_level"] = self.flow_sde_noise_level
    for key in ("A_tau", "v_tau", "mu_tau", "sigma_tau", "A_next"):
        if torch.is_tensor(full_trace[key]):
            full_trace[key] = full_trace[key][:, : int(chunk_len), :].detach()
    log_prob_steps = self._flow_sde_log_prob_steps_from_trace(full_trace, chunk_len=int(chunk_len))
    policy_tensor = mean
    exec_rows = []
    clip_frac_rows = []
    clip_any_rows = []
    oob_rows = []
    for row in policy_tensor:
        exec_np, clip_fraction, clip_any, oob_mean = self._postprocess_and_clip(row.reshape(1, -1))
        exec_rows.append(np.asarray(exec_np, dtype=np.float32))
        clip_frac_rows.append(clip_fraction)
        clip_any_rows.append(clip_any)
        oob_rows.append(oob_mean)
    exec_action_np = np.stack(exec_rows, axis=0)
    trace_action = full_trace["A_next"][:, : int(chunk_len), : self.action_dim].reshape(int(chunk_len), self.action_dim)
    trace_mu = full_trace["mu_tau"][:, : int(chunk_len), : self.action_dim].reshape(int(chunk_len), self.action_dim)
    trace_sigma = full_trace["sigma_tau"][:, : int(chunk_len), : self.action_dim].reshape(int(chunk_len), self.action_dim)
    return SampledActionChunk(
        exec_action_np=exec_action_np,
        policy_tensor=policy_tensor.detach(),
        unsquashed_chunk=policy_tensor.detach(),
        logprob_action=trace_action.detach(),
        log_prob_steps=log_prob_steps.detach(),
        log_prob_sum=log_prob_steps.detach().sum(),
        distr_mean=trace_mu.detach(),
        distr_log_std=torch.log(trace_sigma.clamp(min=self.eps)).detach(),
        action_clip_fraction=np.asarray(clip_frac_rows, dtype=np.float64),
        action_clip_any=np.asarray(clip_any_rows, dtype=np.bool_),
        postprocessor_oob_mean=np.asarray(oob_rows, dtype=np.float64),
        unique_action_rows=int(np.unique(exec_action_np, axis=0).shape[0]),
        flow_sde_trace=full_trace,
    )
```

- [ ] **Step 3.7: Add Flow-SDE chunk recompute**

Add:

```python
def get_flow_sde_log_probs_for_chunk_from_proc_list(
    self,
    proc_snapshots: Sequence[Any],
    flow_sde_traces: Sequence[dict[str, Any] | None],
    *,
    chunk_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(proc_snapshots) != len(flow_sde_traces):
        raise ValueError("proc_snapshots length must match flow_sde_traces length")
    self._reset_policy_forward_state()
    batched = self._proc_to_device(collate_proc_snapshots(proc_snapshots))
    trace = self._proc_to_device(collate_flow_sde_traces(flow_sde_traces))
    hook = getattr(self._policy, "flow_sde_logprob_from_trace", None)
    if not callable(hook):
        raise RuntimeError("flow_sde recompute requires policy.flow_sde_logprob_from_trace")
    _log_probs_full, mu, sigma = hook(batched, trace, flow_sde_noise_level=self.flow_sde_noise_level)
    mu_env = mu[:, : int(chunk_len), : self.action_dim].float()
    sigma_env = sigma[:, : int(chunk_len), : self.action_dim].float()
    action = trace["A_next"][:, : int(chunk_len), : self.action_dim].float()
    log_probs = sde_step_logprob(action, mu_env, sigma_env)
    log_std = torch.log(sigma_env.clamp(min=self.eps))
    return log_probs, mu_env, log_std
```

- [ ] **Step 3.8: Run tests**

Run:

```bash
PYTHONPATH=src pytest tests/test_grpo_policy_wrapper_chunk.py tests/test_grpo_logprob_correctness.py -q
```

Expected: PASS.

- [ ] **Step 3.9: Commit**

Run:

```bash
git add src/smolvla_grpo/policy_wrapper.py tests/test_grpo_policy_wrapper_chunk.py tests/test_grpo_logprob_correctness.py
git commit -m "feat: add chunk flow-sde policy replay"
```

Expected: commit succeeds.

---

## Task 4: Add Serial Official Chunk Rollout

**Files:**
- Create: `src/smolvla_grpo/phase11_chunk_rollout.py`
- Create: `tests/test_phase11_chunk_rollout.py`

- [ ] **Step 4.1: Write fake-env tests**

Create `tests/test_phase11_chunk_rollout.py`:

```python
from __future__ import annotations

import numpy as np
import torch

from smolvla_grpo.phase11_chunk_rollout import build_valid_tail_mask, chunk_success_any


def test_build_valid_tail_mask_excludes_actions_after_terminal() -> None:
    mask = build_valid_tail_mask(chunk_len=5, executed_count=2)

    torch.testing.assert_close(mask, torch.tensor([True, True, False, False, False]))


def test_build_valid_tail_mask_all_valid_when_no_terminal() -> None:
    mask = build_valid_tail_mask(chunk_len=4, executed_count=4)

    torch.testing.assert_close(mask, torch.tensor([True, True, True, True]))


def test_chunk_success_any_uses_only_valid_rows() -> None:
    successes = [False, False, True]
    valid = torch.tensor([True, False, False])

    assert chunk_success_any(successes, valid) is False


def test_chunk_success_any_detects_valid_success() -> None:
    successes = [False, True, False]
    valid = torch.tensor([True, True, False])

    assert chunk_success_any(successes, valid) is True
```

- [ ] **Step 4.2: Run test and verify failure**

Run:

```bash
PYTHONPATH=src pytest tests/test_phase11_chunk_rollout.py -q
```

Expected: FAIL because module is missing.

- [ ] **Step 4.3: Create rollout module with dataclasses and helpers**

Create `src/smolvla_grpo/phase11_chunk_rollout.py`:

```python
"""Chunk-level official-LeRobot rollout collection for Phase11 GRPO."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from smolvla_grpo.lerobot_metaworld_adapter import OfficialLeRobotMetaWorldGRPORollout, resolve_lerobot_horizon
from smolvla_grpo.phase11_rollout import _action_bounds, detach_proc_snapshot
from smolvla_grpo.policy_wrapper import MetaWorldSmolVLAGRPOPolicy
from smolvla_pipeline.evaluator import _SmolVLABundle, _resolve_camera_name, _resolve_flip_corner2


@dataclass
class ChunkDecision:
    proc_snapshot: Any
    exec_actions: np.ndarray
    rewards: torch.Tensor
    successes: torch.Tensor
    terminations: torch.Tensor
    truncations: torch.Tensor
    valid_action_mask: torch.Tensor
    logprob_actions: torch.Tensor
    log_probs: torch.Tensor
    log_prob_sum: torch.Tensor
    distr_mean: torch.Tensor
    distr_log_std: torch.Tensor
    flow_sde_trace: dict[str, Any] | None
    action_clip_fraction: torch.Tensor
    action_clip_any: torch.Tensor
    postprocessor_oob_mean: torch.Tensor


@dataclass
class ChunkRolloutTrajectory:
    reset_seed: int
    rollout_index: int
    chunks: list[ChunkDecision] = field(default_factory=list)
    terminated: bool = False
    truncated: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def rewards(self) -> list[float]:
        out: list[float] = []
        for chunk in self.chunks:
            valid = chunk.valid_action_mask.bool()
            out.extend(chunk.rewards[valid].float().cpu().tolist())
        return [float(x) for x in out]

    @property
    def successes(self) -> list[bool]:
        out: list[bool] = []
        for chunk in self.chunks:
            valid = chunk.valid_action_mask.bool()
            out.extend(bool(x) for x in chunk.successes[valid].cpu().tolist())
        return out

    def total_return(self) -> float:
        total = 0.0
        for chunk in self.chunks:
            total += float((chunk.rewards.float() * chunk.valid_action_mask.float()).sum().item())
        return total


def build_valid_tail_mask(*, chunk_len: int, executed_count: int) -> torch.Tensor:
    if chunk_len < 1:
        raise ValueError("chunk_len must be >= 1")
    executed = max(0, min(int(executed_count), int(chunk_len)))
    mask = torch.zeros(int(chunk_len), dtype=torch.bool)
    mask[:executed] = True
    return mask


def chunk_success_any(successes: list[bool], valid_action_mask: torch.Tensor) -> bool:
    valid = valid_action_mask.bool().cpu().tolist()
    return any(bool(s) and bool(v) for s, v in zip(successes, valid, strict=False))
```

- [ ] **Step 4.4: Add serial collection function**

Append:

```python
def collect_chunk_rollout_group(
    *,
    bundle: _SmolVLABundle,
    policy_old: torch.nn.Module,
    task: str,
    task_text: str,
    reset_seed: int,
    episode_index: int,
    max_steps: int,
    group_size: int,
    action_dim: int,
    device: torch.device,
    chunk_len: int,
    action_transform: str = "no_tanh",
    gaussian_logprob_action: str = "executed",
    logprob_mode: str = "flow_sde",
    flow_sde_noise_level: float = 0.5,
    flow_sde_trace_step: int = -1,
) -> list[ChunkRolloutTrajectory]:
    if int(group_size) < 1:
        raise ValueError("group_size must be >= 1")
    if int(chunk_len) < 1:
        raise ValueError("chunk_len must be >= 1")
    env_h = OfficialLeRobotMetaWorldGRPORollout(task=task, n_envs=1, use_async_envs=False)
    try:
        resolved_max_steps = resolve_lerobot_horizon(env_h, max_steps)
        camera_name = _resolve_camera_name()
        flip_corner2 = _resolve_flip_corner2()
        action_low, action_high = _action_bounds(env_h)
        old_wrapper = MetaWorldSmolVLAGRPOPolicy(
            bundle,
            task=task,
            task_text=task_text,
            camera_name=camera_name,
            flip_corner2=flip_corner2,
            action_dim=action_dim,
            policy_module=policy_old,
            action_transform=action_transform,
            gaussian_logprob_action=gaussian_logprob_action,
            logprob_mode=logprob_mode,
            flow_sde_noise_level=flow_sde_noise_level,
            flow_sde_trace_step=flow_sde_trace_step,
            action_low=action_low,
            action_high=action_high,
        )
        old_wrapper.eval()
        rollouts: list[ChunkRolloutTrajectory] = []
        for r in range(int(group_size)):
            gen = torch.Generator(device=device)
            gen.manual_seed(int(reset_seed) * 1000003 + r * 7919)
            obs = env_h.reset(int(reset_seed))
            policy_reset = getattr(policy_old, "reset", None)
            if callable(policy_reset):
                policy_reset()
            traj = ChunkRolloutTrajectory(reset_seed=int(reset_seed), rollout_index=r)
            traj.metadata.update(
                {
                    "task": task,
                    "episode_index": int(episode_index),
                    "env_backend": "official_lerobot",
                    "rollout_unit": "chunk",
                    "rollout_execution": "serial",
                    "chunk_len": int(chunk_len),
                    "requested_max_steps": int(max_steps),
                    "resolved_max_steps": int(resolved_max_steps),
                    "logprob_mode": logprob_mode,
                    "flow_sde_trace_step": int(flow_sde_trace_step),
                }
            )
            scalar_steps = 0
            done = False
            while scalar_steps < int(resolved_max_steps) and not done:
                proc = env_h.build_proc(obs, bundle=bundle)
                with torch.no_grad():
                    sampled = old_wrapper.sample_action_chunk_from_proc(proc, chunk_len=int(chunk_len), rng=gen)
                rewards: list[float] = []
                successes: list[bool] = []
                terms: list[bool] = []
                truncs: list[bool] = []
                executed_count = 0
                for i in range(int(chunk_len)):
                    if scalar_steps >= int(resolved_max_steps) or done:
                        break
                    step = env_h.step(sampled.exec_action_np[i : i + 1].astype(np.float32, copy=False))
                    obs = step.observation
                    rewards.append(float(step.reward))
                    successes.append(bool(step.success))
                    terms.append(bool(step.terminated))
                    truncs.append(bool(step.truncated))
                    executed_count += 1
                    scalar_steps += 1
                    if bool(step.success) or bool(step.terminated) or bool(step.truncated):
                        done = True
                        traj.terminated = bool(step.terminated)
                        traj.truncated = bool(step.truncated)
                pad_n = int(chunk_len) - executed_count
                rewards.extend([0.0] * pad_n)
                successes.extend([False] * pad_n)
                terms.extend([False] * pad_n)
                truncs.extend([False] * pad_n)
                valid_mask = build_valid_tail_mask(chunk_len=int(chunk_len), executed_count=executed_count)
                chunk = ChunkDecision(
                    proc_snapshot=detach_proc_snapshot(proc),
                    exec_actions=sampled.exec_action_np.astype(np.float32, copy=False),
                    rewards=torch.tensor(rewards, dtype=torch.float32),
                    successes=torch.tensor(successes, dtype=torch.bool),
                    terminations=torch.tensor(terms, dtype=torch.bool),
                    truncations=torch.tensor(truncs, dtype=torch.bool),
                    valid_action_mask=valid_mask,
                    logprob_actions=sampled.logprob_action.detach().cpu(),
                    log_probs=sampled.log_prob_steps.detach().cpu(),
                    log_prob_sum=(sampled.log_prob_steps.detach().cpu() * valid_mask.float()).sum(),
                    distr_mean=sampled.distr_mean.detach().cpu(),
                    distr_log_std=sampled.distr_log_std.detach().cpu(),
                    flow_sde_trace=sampled.flow_sde_trace,
                    action_clip_fraction=torch.as_tensor(sampled.action_clip_fraction, dtype=torch.float32),
                    action_clip_any=torch.as_tensor(sampled.action_clip_any, dtype=torch.bool),
                    postprocessor_oob_mean=torch.as_tensor(sampled.postprocessor_oob_mean, dtype=torch.float32),
                )
                traj.chunks.append(chunk)
            rollouts.append(traj)
        return rollouts
    finally:
        env_h.close()
```

- [ ] **Step 4.5: Run tests**

Run:

```bash
PYTHONPATH=src pytest tests/test_phase11_chunk_rollout.py -q
```

Expected: PASS.

- [ ] **Step 4.6: Commit**

Run:

```bash
git add src/smolvla_grpo/phase11_chunk_rollout.py tests/test_phase11_chunk_rollout.py
git commit -m "feat: add serial chunk rollout"
```

Expected: commit succeeds.

---

## Task 5: Add Trainer Chunk Mode

**Files:**
- Modify: `scripts/grpo/train_phase11_env_on_policy_grpo.py`
- Create: `tests/test_phase11_trainer_static.py`

- [ ] **Step 5.1: Write static trainer tests**

Create `tests/test_phase11_trainer_static.py`:

```python
from __future__ import annotations

from pathlib import Path


TRAINER = Path(__file__).resolve().parents[1] / "scripts" / "grpo" / "train_phase11_env_on_policy_grpo.py"


def test_trainer_exposes_rollout_unit_and_rollout_chunk_len() -> None:
    text = TRAINER.read_text(encoding="utf-8")
    assert "--rollout-unit" in text
    assert 'choices=("step", "chunk")' in text
    assert "--rollout-chunk-len" in text


def test_flow_sde_step_mode_is_guarded() -> None:
    text = TRAINER.read_text(encoding="utf-8")
    assert "flow_sde requires --rollout-unit chunk" in text


def test_chunk_mode_loads_bundle_with_rollout_chunk_len() -> None:
    text = TRAINER.read_text(encoding="utf-8")
    assert "n_action_steps=(int(args.rollout_chunk_len) if args.rollout_unit == \"chunk\" else 1)" in text


def test_chunk_mode_uses_chunk_rollout_collector() -> None:
    text = TRAINER.read_text(encoding="utf-8")
    assert "collect_chunk_rollout_group" in text
```

- [ ] **Step 5.2: Run test and verify failure**

Run:

```bash
PYTHONPATH=src pytest tests/test_phase11_trainer_static.py -q
```

Expected: FAIL because trainer has no chunk mode yet.

- [ ] **Step 5.3: Add CLI args and guards**

In parser:

```python
p.add_argument("--rollout-unit", choices=("step", "chunk"), default="step")
p.add_argument("--rollout-chunk-len", type=int, default=5)
```

After `args = p.parse_args()`:

```python
if args.logprob_mode == "flow_sde" and args.rollout_unit != "chunk":
    raise SystemExit("flow_sde requires --rollout-unit chunk")
if args.rollout_unit == "chunk" and args.env_backend != "official_lerobot":
    raise SystemExit("chunk rollout requires --env-backend official_lerobot")
if args.rollout_unit == "chunk" and args.rollout_execution != "serial":
    raise SystemExit("first chunk rollout implementation requires --rollout-execution serial")
if int(args.rollout_chunk_len) < 1:
    raise SystemExit("--rollout-chunk-len must be >= 1")
```

- [ ] **Step 5.4: Load bundle with chunk length in chunk mode**

Replace bundle load with:

```python
bundle, action_dim = load_bundle_for_grpo(
    args.checkpoint,
    task=args.task,
    env_backend=args.env_backend,
    n_action_steps=(int(args.rollout_chunk_len) if args.rollout_unit == "chunk" else 1),
)
```

- [ ] **Step 5.5: Import chunk helpers**

Inside `main()` imports:

```python
from smolvla_grpo.chunk_math import masked_chunk_sum, masked_chunk_reward_sum, valid_chunk_any
from smolvla_grpo.phase11_chunk_rollout import collect_chunk_rollout_group
```

- [ ] **Step 5.6: Add chunk parity helper**

Add near `compute_live_logprob_parity()`:

```python
def compute_live_chunk_logprob_parity(
    *,
    train_wrapper,
    rollouts,
    chunk_len: int,
    tolerance: float,
):
    import torch

    from smolvla_grpo.chunk_math import masked_chunk_sum
    from smolvla_grpo.grpo_math import summarize_logprob_ratio_parity

    old_chunks: list[torch.Tensor] = []
    new_chunks: list[torch.Tensor] = []
    per_action_abs: list[torch.Tensor] = []
    with torch.no_grad():
        for traj in rollouts:
            procs = [chunk.proc_snapshot for chunk in traj.chunks]
            traces = [chunk.flow_sde_trace for chunk in traj.chunks]
            if not procs:
                continue
            new_steps, _mu, _log_std = train_wrapper.get_flow_sde_log_probs_for_chunk_from_proc_list(
                procs,
                traces,
                chunk_len=int(chunk_len),
            )
            new_steps = new_steps.detach().cpu()
            for idx, chunk in enumerate(traj.chunks):
                valid = chunk.valid_action_mask.reshape(1, -1)
                old_step = chunk.log_probs.reshape(1, -1)
                new_step = new_steps[idx : idx + 1]
                old_chunks.append(masked_chunk_sum(old_step, valid).reshape(1))
                new_chunks.append(masked_chunk_sum(new_step, valid).reshape(1))
                per_action_abs.append(((new_step - old_step).abs() * valid.float()).reshape(-1))
    stats = summarize_logprob_ratio_parity(
        torch.cat(old_chunks) if old_chunks else torch.zeros(0),
        torch.cat(new_chunks) if new_chunks else torch.zeros(0),
        tolerance=float(tolerance),
    )
    max_per_action_abs = float(torch.cat(per_action_abs).max().item()) if per_action_abs else 0.0
    payload = stats.as_dict()
    payload["max_abs_per_action_logprob"] = max_per_action_abs
    return stats, payload
```

- [ ] **Step 5.7: Route rollout collection**

Replace rollout collection block with branch:

```python
if args.rollout_unit == "chunk":
    rollouts = collect_chunk_rollout_group(
        bundle=bundle,
        policy_old=old_policy,
        task=args.task,
        task_text=task_text,
        reset_seed=reset_seed,
        episode_index=update,
        max_steps=args.max_steps,
        group_size=args.group_size,
        action_dim=action_dim,
        device=device,
        chunk_len=int(args.rollout_chunk_len),
        action_transform=args.action_transform,
        gaussian_logprob_action=args.gaussian_logprob_action,
        logprob_mode=args.logprob_mode,
        flow_sde_noise_level=float(args.flow_sde_noise_level),
        flow_sde_trace_step=int(args.flow_sde_trace_step),
    )
else:
    rollouts = collect_rollout_group(...)
```

Keep the existing `collect_rollout_group(...)` call exactly as the step branch body.

- [ ] **Step 5.8: Add chunk metrics branch**

After rollout collection, branch metrics:

```python
if args.rollout_unit == "chunk":
    returns = torch.tensor([tr.total_return() for tr in rollouts], dtype=torch.float32, device=device)
    successes = [any(tr.successes) for tr in rollouts]
    episode_lengths = [len(tr.rewards) for tr in rollouts]
    num_env_steps = int(sum(episode_lengths))
    rollout_old_lp = torch.cat([chunk.log_prob_sum.reshape(1) for tr in rollouts for chunk in tr.chunks]).to(device)
    rollout_log_std = torch.cat([chunk.distr_log_std.reshape(-1, action_dim) for tr in rollouts for chunk in tr.chunks]).to(device)
else:
    # existing step metrics code
```

- [ ] **Step 5.9: Add chunk optimization branch**

Before existing step-wise optimize loop, add:

```python
if args.rollout_unit == "chunk":
    bundle.policy.eval()
    parity_stats, parity_payload = compute_live_chunk_logprob_parity(
        train_wrapper=train_wrapper,
        rollouts=rollouts,
        chunk_len=int(args.rollout_chunk_len),
        tolerance=float(args.parity_tolerance),
    )
    if not parity_stats.within_tolerance:
        msg = (
            f"GRPO chunk logprob parity failed update={update}: "
            f"mean_ratio={parity_stats.mean_ratio:.6f} "
            f"max_abs_log_ratio={parity_stats.max_abs_log_ratio:.6f} "
            f"max_abs_per_action_logprob={parity_payload['max_abs_per_action_logprob']:.6f}"
        )
        print(msg, flush=True)
        if args.fail_on_parity_violation:
            raise RuntimeError(msg)
    bundle.policy.train()
    optimize_t0 = time.perf_counter()
    last_new_log_probs = []
    last_old_log_probs = []
    last_log_stds = []
    for _epoch in range(args.update_epochs):
        optimizer.zero_grad()
        valid_chunk_count = 0
        for gi, traj in enumerate(rollouts):
            A = advantages[gi].reshape(()).float()
            procs = [chunk.proc_snapshot for chunk in traj.chunks]
            traces = [chunk.flow_sde_trace for chunk in traj.chunks]
            if not procs:
                continue
            new_steps, _mu_live, log_std_live = train_wrapper.get_flow_sde_log_probs_for_chunk_from_proc_list(
                procs,
                traces,
                chunk_len=int(args.rollout_chunk_len),
            )
            for ci, chunk in enumerate(traj.chunks):
                valid = chunk.valid_action_mask.reshape(1, -1).to(device)
                if not bool(valid.any()):
                    continue
                old_steps = chunk.log_probs.reshape(1, -1).to(device)
                new_step = new_steps[ci : ci + 1]
                old_lp = masked_chunk_sum(old_steps, valid)
                new_lp = masked_chunk_sum(new_step, valid)
                ratio = torch.exp((new_lp - old_lp).clamp(-20.0, 20.0))
                unclipped = ratio * A
                clipped = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * A
                loss = -torch.min(unclipped, clipped).sum()
                loss = apply_grpo_regularizers(
                    loss,
                    current_log_probs=new_lp,
                    reference_log_probs=old_lp,
                    log_std=log_std_live[ci : ci + 1].reshape(-1, action_dim),
                    kl_beta=float(args.kl_beta),
                    entropy_coef=float(args.entropy_coef),
                )
                loss.backward()
                valid_chunk_count += 1
                last_new_log_probs.append(new_lp.detach().cpu())
                last_old_log_probs.append(old_lp.detach().cpu())
                last_log_stds.append(log_std_live[ci].detach().cpu().reshape(-1, action_dim))
        if valid_chunk_count == 0:
            raise RuntimeError("chunk rollout produced zero valid chunks")
        total_grad_norm = nn.utils.clip_grad_norm_(bundle.policy.parameters(), args.grad_clip)
        optimizer.step()
    optimize_seconds = float(time.perf_counter() - optimize_t0)
    # Continue into shared progress/checkpoint writing.
else:
    # existing step optimize path
```

Normalize loss by valid chunk count if gradient scale is too large in smoke. First pass can keep one loss per chunk because `update_epochs=1` and `grad_clip` is active, but record `valid_chunk_count`.

- [ ] **Step 5.10: Add manifest/progress fields**

Ensure manifest and progress rows include:

```python
"rollout_unit": args.rollout_unit,
"rollout_chunk_len": int(args.rollout_chunk_len),
"policy_n_action_steps": int(getattr(getattr(bundle.policy, "config", None), "n_action_steps", -1)),
"flow_sde_trace_step": int(args.flow_sde_trace_step),
```

For chunk progress rows include:

```python
"chunk_count": int(sum(len(tr.chunks) for tr in rollouts)),
"valid_chunk_count": int(valid_chunk_count),
"parity": parity_payload,
```

- [ ] **Step 5.11: Run static tests**

Run:

```bash
PYTHONPATH=src pytest tests/test_phase11_trainer_static.py -q
```

Expected: PASS.

- [ ] **Step 5.12: Run focused trainer-adjacent tests**

Run:

```bash
PYTHONPATH=src pytest tests/test_chunk_math.py tests/test_phase11_chunk_rollout.py tests/test_grpo_policy_wrapper_chunk.py tests/test_grpo_logprob_correctness.py tests/test_phase11_trainer_static.py -q
```

Expected: PASS.

- [ ] **Step 5.13: Commit**

Run:

```bash
git add scripts/grpo/train_phase11_env_on_policy_grpo.py tests/test_phase11_trainer_static.py
git commit -m "feat: add chunk flow-sde trainer mode"
```

Expected: commit succeeds.

---

## Task 6: Stabilize Flow-SDE fp32 Logprob

**Files:**
- Modify: `src/smolvla_grpo/flow_logprob.py`
- Modify: `tests/test_flow_logprob.py`

- [ ] **Step 6.1: Add dtype stability test**

Add to `tests/test_flow_logprob.py`:

```python
def test_sde_step_logprob_upcasts_low_precision_inputs() -> None:
    x_next = torch.tensor([[[0.25, -0.25]]], dtype=torch.bfloat16)
    mu = torch.tensor([[[0.0, 0.0]]], dtype=torch.bfloat16)
    sigma = torch.tensor([[[0.5, 0.5]]], dtype=torch.bfloat16)

    out = sde_step_logprob(x_next, mu, sigma)

    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()
```

- [ ] **Step 6.2: Run test and verify failure if current dtype remains bf16**

Run:

```bash
PYTHONPATH=src pytest tests/test_flow_logprob.py::test_sde_step_logprob_upcasts_low_precision_inputs -q
```

Expected: FAIL if output dtype is not fp32.

- [ ] **Step 6.3: Upcast inside Flow-SDE logprob**

Modify `sde_step_logprob_per_dim()`:

```python
x_next = x_next.float()
mu = mu.float()
sigma = sigma.float()
```

Place these lines before `mask = sigma <= eps`.

- [ ] **Step 6.4: Run test**

Run:

```bash
PYTHONPATH=src pytest tests/test_flow_logprob.py -q
```

Expected: PASS.

- [ ] **Step 6.5: Commit**

Run:

```bash
git add src/smolvla_grpo/flow_logprob.py tests/test_flow_logprob.py
git commit -m "fix: use fp32 flow-sde logprob math"
```

Expected: commit succeeds.

---

## Task 7: Add Chunk Slurm Scripts

**Files:**
- Create: `scripts/grpo/submit_flow_sde_chunk_grpo_smoke_a30.slurm`
- Create: `scripts/grpo/submit_flow_sde_chunk_grpo_train16_a30.slurm`
- Create: `scripts/grpo/submit_flow_sde_chunk_grpo_eval25_a30.slurm`
- Modify: `tests/test_phase11_slurm_scripts.py`

- [ ] **Step 7.1: Create smoke script**

Create `scripts/grpo/submit_flow_sde_chunk_grpo_smoke_a30.slurm`:

```bash
#!/usr/bin/env bash
#SBATCH --job-name=flow-sde-chunk-smoke
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=flow_sde_chunk_grpo_smoke_%j.out
#SBATCH --error=flow_sde_chunk_grpo_smoke_%j.err
#SBATCH --export=NIL

set -euo pipefail

_PROJECT_FALLBACK="/vol/bitbucket/aa6622/project"
if [[ -f "${SLURM_SUBMIT_DIR:-}/scripts/slurm/common_env.sh" ]]; then
  source "${SLURM_SUBMIT_DIR}/scripts/slurm/common_env.sh"
elif [[ -f "${_PROJECT_FALLBACK}/scripts/slurm/common_env.sh" ]]; then
  source "${_PROJECT_FALLBACK}/scripts/slurm/common_env.sh"
fi

slurm_resolve_project_root "scripts/grpo/train_phase11_env_on_policy_grpo.py"
cd "${PROJECT_ROOT}"
slurm_export_pythonpath
slurm_export_hf_torch_cache "flow-sde-chunk-grpo-smoke"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

GRPO_PYTHON="${GRPO_PYTHON:-/vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python}"
CHECKPOINT="${1:-/vol/bitbucket/aa6622/.cache/huggingface/hub/models--jadechoghari--smolvla_metaworld/snapshots/ef3089ecb84eeeb7d33fedab24f6c76180a68900}"
OUT="${2:-${PROJECT_ROOT}/artifacts/flow_sde_chunk_grpo_smoke/${SLURM_JOB_ID:-local}}"

"${GRPO_PYTHON}" scripts/grpo/train_phase11_env_on_policy_grpo.py \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${OUT}" \
  --task push-v3 \
  --env-backend official_lerobot \
  --rollout-execution serial \
  --rollout-unit chunk \
  --rollout-chunk-len 5 \
  --group-size 4 \
  --num-updates 1 \
  --train-seed-base 2000 \
  --max-steps 30 \
  --lr 5e-6 \
  --clip-eps 0.2 \
  --init-log-std -2.0 \
  --euler-step-noise-std 0.0 \
  --min-log-std -4.0 \
  --action-transform no_tanh \
  --logprob-mode flow_sde \
  --flow-sde-noise-level 0.5 \
  --flow-sde-trace-step -1 \
  --run-label flow_sde_chunk_smoke \
  --save-every 1 \
  --fail-on-parity-violation

test -f "${OUT}/checkpoints/update_0001.pt"
test -f "${OUT}/progress.jsonl"
echo "FLOW_SDE_CHUNK_GRPO_SMOKE_OK out=${OUT}"
```

- [ ] **Step 7.2: Create train16 script**

Copy smoke script to `scripts/grpo/submit_flow_sde_chunk_grpo_train16_a30.slurm` and change:

```bash
#SBATCH --job-name=flow-sde-chunk-u16
#SBATCH --cpus-per-task=12
#SBATCH --time=12:00:00
#SBATCH --output=flow_sde_chunk_grpo_train16_%j.out
#SBATCH --error=flow_sde_chunk_grpo_train16_%j.err
slurm_export_hf_torch_cache "flow-sde-chunk-grpo-train16"
OUT="${2:-${PROJECT_ROOT}/artifacts/flow_sde_chunk_grpo_train16/${SLURM_JOB_ID:-local}}"
  --group-size 8 \
  --num-updates 16 \
  --max-steps 120 \
  --run-label flow_sde_chunk_u16 \
  --save-every 2 \
test -f "${OUT}/checkpoints/update_0016.pt"
echo "FLOW_SDE_CHUNK_GRPO_TRAIN16_OK out=${OUT}"
```

- [ ] **Step 7.3: Create eval script**

Create `scripts/grpo/submit_flow_sde_chunk_grpo_eval25_a30.slurm` by copying existing `submit_flow_sde_grpo_eval25_a30.slurm`, then update job/output names and echo sentinels to:

```bash
#SBATCH --job-name=flow-sde-chunk-e25
#SBATCH --output=flow_sde_chunk_grpo_eval25_%j.out
#SBATCH --error=flow_sde_chunk_grpo_eval25_%j.err
echo "FLOW_SDE_CHUNK_GRPO_EVAL25_OK out=${OUT_DIR}"
```

Keep eval protocol unchanged: `--num-episodes 25`, seed base `1000`, `--chunk-len 5`.

- [ ] **Step 7.4: Add Slurm tests**

Modify `tests/test_phase11_slurm_scripts.py` to include new scripts in bash syntax loop and add:

```python
def test_submit_flow_sde_chunk_grpo_smoke_uses_chunk_mode() -> None:
    path = _REPO_ROOT / "scripts" / "grpo" / "submit_flow_sde_chunk_grpo_smoke_a30.slurm"
    text = path.read_text(encoding="utf-8")
    assert "#SBATCH --export=NIL" in text
    assert "--rollout-unit chunk" in text
    assert "--rollout-chunk-len 5" in text
    assert "--rollout-execution serial" in text
    assert "--logprob-mode flow_sde" in text
    assert "--flow-sde-trace-step -1" in text
    assert "--fail-on-parity-violation" in text
    assert "FLOW_SDE_CHUNK_GRPO_SMOKE_OK" in text
```

- [ ] **Step 7.5: Run Slurm tests**

Run:

```bash
PYTHONPATH=src pytest tests/test_phase11_slurm_scripts.py -q
```

Expected: PASS.

- [ ] **Step 7.6: Commit**

Run:

```bash
git add scripts/grpo/submit_flow_sde_chunk_grpo_smoke_a30.slurm scripts/grpo/submit_flow_sde_chunk_grpo_train16_a30.slurm scripts/grpo/submit_flow_sde_chunk_grpo_eval25_a30.slurm tests/test_phase11_slurm_scripts.py
git commit -m "chore: add chunk flow-sde slurm gates"
```

Expected: commit succeeds.

---

## Task 8: Full Verification Before GPU

**Files:**
- No source edits expected.

- [ ] **Step 8.1: Run focused unit tests**

Run:

```bash
PYTHONPATH=src pytest \
  tests/test_flow_logprob.py \
  tests/test_chunk_math.py \
  tests/test_grpo_policy_wrapper_chunk.py \
  tests/test_grpo_logprob_correctness.py \
  tests/test_phase11_chunk_rollout.py \
  tests/test_phase11_trainer_static.py \
  tests/test_phase11_slurm_scripts.py \
  -q
```

Expected: PASS.

- [ ] **Step 8.2: Run lints through IDE diagnostics**

Use Cursor lints on changed files. Expected: no new errors in:

- `src/smolvla_grpo/chunk_math.py`
- `src/smolvla_grpo/policy_wrapper.py`
- `src/smolvla_grpo/phase11_chunk_rollout.py`
- `scripts/grpo/train_phase11_env_on_policy_grpo.py`

- [ ] **Step 8.3: Commit if verification-only fixes were needed**

Run only if files changed:

```bash
git add src/smolvla_grpo tests scripts/grpo docs/findings
git commit -m "fix: stabilize chunk flow-sde tests"
```

Expected: commit succeeds or no changes exist.

---

## Task 9: GPU Smoke And Gated Runs

**Files:**
- No source edits expected unless a smoke failure has a verified RCA.

- [ ] **Step 9.1: Submit one-update smoke**

Run from `/vol/bitbucket/aa6622/project`:

```bash
sbatch scripts/grpo/submit_flow_sde_chunk_grpo_smoke_a30.slurm
```

Expected: job queues one A30 GPU. Do not submit multiple GPU jobs beyond cluster caps.

Autonomy rule: if CPU verification passed, submit this smoke without asking the user again.

- [ ] **Step 9.2: Inspect smoke output**

Expected sentinel:

```text
FLOW_SDE_CHUNK_GRPO_SMOKE_OK
```

Expected progress fields:

```json
{
  "rollout_unit": "chunk",
  "rollout_chunk_len": 5,
  "logprob_mode": "flow_sde",
  "parity": {
    "within_tolerance": true
  }
}
```

- [ ] **Step 9.3: If smoke fails, stop for RCA**

Classify failure:

- Shape mismatch: inspect `A_tau`, `A_next`, `mu_tau`, `sigma_tau`, `action_dim`, `chunk_len`.
- Parity drift: log by chunk row/action dim/`tau_idx`/dtype.
- Env failure: inspect `OfficialLeRobotMetaWorldGRPORollout.step()` action shape and terminal handling.
- CUDA/OOM: reduce group size before changing algorithm.

Do not relax `--parity-tolerance` before one of these is explained.

After RCA: apply the smallest verified fix, rerun focused CPU tests plus the smoke, and continue autonomously if the gate passes.

- [ ] **Step 9.4: Submit 16-update run only after smoke passes**

Run:

```bash
sbatch scripts/grpo/submit_flow_sde_chunk_grpo_train16_a30.slurm
```

Expected sentinel:

```text
FLOW_SDE_CHUNK_GRPO_TRAIN16_OK
```

Autonomy rule: submit train16 without asking the user again once smoke passes.

- [ ] **Step 9.5: Submit 25-episode eval after train16 checkpoint exists**

Run:

```bash
sbatch scripts/grpo/submit_flow_sde_chunk_grpo_eval25_a30.slurm \
  /path/to/train16/out/checkpoints \
  /path/to/eval25/out
```

Expected sentinel:

```text
FLOW_SDE_CHUNK_GRPO_EVAL25_OK
```

Autonomy rule: submit eval25 without asking the user again once train16 creates the expected checkpoint.

---

## Human-In-The-Loop Flags

- If smoke parity fails by more than `0.02`, stop and report RCA. Do not continue to train16.
- If train16 passes parity but success collapses, compare with Gaussian chunk mode before blaming Flow-SDE.
- If serial chunk smoke is too slow, do not jump straight to async vector; first add a vector chunk test and one vector smoke.
- If `n_action_steps=chunk_len` reduces baseline eval behavior, record it as train-contract shift, not a bug by itself.

---

## Self-Review

- Spec coverage: plan covers critical review, corrected assumptions, missing info, tests, frequent commits, chunk rollout, trainer objective, Slurm gates, and RCA handling.
- Placeholder scan: no `TBD`, no `TODO`, no unspecified error handling.
- Type consistency: chunk logprobs are `[H]` or `[B,H]`; valid masks are `[H]` or `[B,H]`; env actions are `[H,4]`; traces preserve full padded width in policy wrapper.
