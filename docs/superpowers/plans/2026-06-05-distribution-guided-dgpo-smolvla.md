# Distribution-Guided DGPO for SmolVLA + MetaWorld Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the *correct* DGPO — **Distribution-Guided Policy Optimization** (arXiv 2605.03327), a token-level credit-reassignment upgrade to GRPO — to SmolVLA on MetaWorld push-v3, first on the proven 41% Phase-11 flow-sde-GRPO trainer (Phase A), then ported into RLinf (Phase B). Target: **beat the 41% Flow-SDE GRPO baseline**.

**Architecture:** DGPO keeps the GRPO trajectory advantage `A_i` and *redistributes* it across the episode's action-chunks with a softmax weight `w_{i,t}` built from a bounded per-chunk **Hellinger deviation** `d_{i,t}=H²(π_θ‖π_ref)` (optionally entropy-gated). The per-chunk advantage `Â_{i,t}=A_i·w_{i,t}` then flows into the *existing* PPO-clipped loss. No new objective, no critic, no KL term. It is a multiplicative rescale of the advantage right before the policy loss.

**Tech Stack:** Python 3.10, PyTorch, the project `smolvla_grpo` package + `train_phase11_env_on_policy_grpo.py` trainer (Phase A); RLinf FSDP embodied actor (Phase B). SLURM `a30` partition. pytest for unit tests.

---

## Context the engineer MUST read first (zero-context safe)

### The two-DGPO mix-up (do not repeat it)
There are two unrelated papers abbreviated "DGPO". **We are implementing arXiv 2605.03327 "Distribution-Guided Policy Optimization"** (LLM-reasoning GRPO upgrade). We are **NOT** implementing arXiv 2510.08425 "Direct Group Preference Optimization" (diffusion image preference). The prior RLinf `nft_loss_form=dgpo` branch implemented the *wrong* one and also had a broken grouping bug; it scored 19% vs the 41% baseline. **Ignore / do not extend the NFT `dgpo` branch.** Paper PDF: `/vol/bitbucket/aa6622/project/docs/papers/sita dgpo 2605.03327v2.pdf`. HTML: https://arxiv.org/html/2605.03327v2.

### What the paper actually specifies (exact math)
For prompt `i`, group of `G` rollouts, sequence of `T_i` steps `t`:
1. **GRPO advantage** (unchanged): `A_i = (r_i − mean(r)) / (std(r) + ε)`.
2. **Hellinger deviation** per step: `d_{i,t} = H²(π_θ(·|s_t) ‖ π_ref(·|s_t)) ∈ [0,1]` (bounded → no gradient explosion).
3. **Normalized entropy gate**: `H̃_{i,t} = H(π_θ(·|s_t)) / H_max ∈ [0,1]`.
4. **Gated score**: `s_{i,t} = d_{i,t} · H̃_{i,t}^κ`  (κ default 1.0; **we default κ=0**, see caveat below).
5. **Redistribution weight** (unit-mean softmax): `w_{i,t} = T_i · exp(s_{i,t}/τ) / Σ_j exp(s_{i,j}/τ)`, so `(1/T_i)Σ_t w_{i,t} = 1`. τ default 0.5 (range 0.5–1.0).
6. **Token advantage**: `Â_{i,t} = A_i · w_{i,t}`.
7. **Loss**: standard PPO-clipped objective with `Â_{i,t}`, **no KL penalty**.

### VLA adaptation decisions (the parts that are NOT 1-to-1)
- **Continuous, not categorical.** "Token" = one **action chunk** (push-v3: chunk_len=5, ~24 chunks/episode). π is a **diagonal Gaussian** over the action chunk (mean `μ`, std `σ=exp(log_std)`). Hellinger uses the **closed form** below, not a vocab sum.
- **Hellinger (diagonal Gaussian), general unequal-variance form:**
  `BC = Π_k sqrt( 2·σ1_k·σ2_k / (σ1_k² + σ2_k²) ) · exp( −(μ1_k−μ2_k)² / (4·(σ1_k² + σ2_k²)) )`,  `d = 1 − BC ∈ [0,1]`. Product over the chunk's flattened dims `k`.
- **Entropy gate caveat (FLAG):** SmolVLA's `log_std` is a **state-independent** parameter (`policy.model.log_std`, see `_log_std_telemetry` at `scripts/grpo/train_phase11_env_on_policy_grpo.py:212`). So differential entropy is ~constant across chunks → the entropy gate degenerates to a temperature rescale. **Default κ=0 (deviation-only).** A real gate needs a heteroscedastic (state-dependent) std head — out of scope, noted as future work.
- **Reference policy π_ref:** a **frozen copy of the base SFT policy** (the checkpoint we start from). Deviation starts ≈0 (DGPO≈GRPO early) and grows over training as π_θ moves — a safe, monotone credit signal. An ablation uses π_old (rollout policy).

### Resources / paths (verified to exist)
- **Primary trainer (41% baseline):** `/vol/bitbucket/aa6622/project/scripts/grpo/train_phase11_env_on_policy_grpo.py`
  - **The whole trainer is one `if __name__ == "__main__":` block** (no `def main()`). Heavy deps (`MetaWorldSmolVLAGRPOPolicy`, `load_bundle_for_grpo`, `compute_group_advantages`, …) are imported *inside* that block at `:503-531`. Argparse parser var is `p`, parsed at `args = p.parse_args()` (`:668`). `device` at `:701`. Update loop `for update in range(start_u, end_u):` at `:847`. Rollouts (`--rollout-unit chunk`) collected via `collect_chunk_rollout_group(...)` into `rollouts` at `:857`.
  - **THE SEAM WE EDIT (verified): the chunk path uses a custom inline loss loop, NOT `_backward_phase11_group_loss`.** The chunk branch runs `:918-1219` and ends with `continue` (`:1219`); `_backward_phase11_group_loss` (`:339`/`:1381`) is the **non-chunk** path — *do not edit it for Phase A*. The inline loop is `:1108-1148`:
    - `advantages = compute_group_advantages(returns)` at `:975` → **one scalar advantage per trajectory** (`returns` shape `[G]`).
    - epoch loop `for _epoch in range(args.update_epochs):` (`:1108`); `for gi, traj in enumerate(rollouts):` (`:1113`); `A = advantages[gi].reshape(()).float()` (`:1114`); `for chunk in traj.chunks:` (`:1115`); **skips invalid chunks** `if not bool(valid.any()): continue` (`:1117-1118`); advantage applied at `unclipped = ratio * A` / `clipped = ...* A` (`:1131-1132`); `chunk_loss.backward()` (`:1142`). Normalizer is `valid_chunk_count` (`:960`).
- **Policy package:** `/vol/bitbucket/aa6622/project/src/smolvla_grpo/`
  - **Per-chunk Gaussian is ALREADY stored on every rollout chunk:** `ChunkDecision` (`phase11_chunk_rollout.py:12-29`) has `.distr_mean` and `.distr_log_std`, each shape **`[chunk_len, action_dim]`** (assigned `policy_wrapper.py:844-845,854-855`; copied `phase11_chunk_rollout.py:224-225`). These are the **behavior/old-policy** Gaussian (`collect_chunk_rollout_group(policy_old=old_policy,...)` at `:857-876`); since the trainer is on-policy and `old_policy` is synced to the current policy at the end of every update (`:1192-1194`), `distr_mean ≈ current π_θ at update start`. Each `ChunkDecision` also has `.proc_snapshot`, `.flow_sde_trace`, `.valid_action_mask`, `.log_probs`, `.log_prob_sum`.
  - Reference-policy per-chunk `(μ, log_std)` recompute: `policy_wrapper.py:get_flow_sde_log_probs_for_chunk_from_proc_list` (`:1104`, returns `(log_probs[b,chunk], mu_env[b,chunk,adim], log_std[b,chunk,adim])`).
  - GRPO advantage: `grpo_math.py:compute_group_advantages` (`:104`). Per-chunk KL/entropy regularizers: `apply_grpo_regularizers` (used at `:1134`) — DGPO does not touch these.
  - **Tests need `src` + repo root on the path.** Run pytest with `PYTHONPATH=/vol/bitbucket/aa6622/project:/vol/bitbucket/aa6622/project/src` (this is what `slurm_export_pythonpath` exports; there is no editable install). `tests/conftest.py` only sets MuJoCo EGL, it does NOT add paths.
- **Proven flow-sde GRPO slurm to clone:** `/vol/bitbucket/aa6622/project/scripts/grpo/submit_flow_sde_chunk_grpo_moonshot30_sparse_chain_a30.slurm` (trainer flags: `--logprob-mode flow_sde --rollout-unit chunk --rollout-chunk-len 5 --group-size 16 --reward-mode sparse_success_delta --max-steps 120 --flow-sde-noise-level 1.0`).
- **Eval sweeper:** `RLinf-smolvla-metaworld-ppo-grpo/scripts/eval_smolvla_metaworld_ckpt_sweep.py` (used by the slurm above, eval25 + eval100, seeds 1000+).
- **Train venv:** `/vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python` (`GRPO_PYTHON_BIN`). **Eval venv:** the RLinf `PYTHON_BIN` from `rlinf_smolvla_common.sh`.
- **Base model:** `/vol/bitbucket/aa6622/.cache/huggingface/hub/models--jadechoghari--smolvla_metaworld/snapshots/ef3089ecb84eeeb7d33fedab24f6c76180a68900`.
- **RLinf port targets (Phase B):** loss `RLinf-.../rlinf/algorithms/losses.py:167` (`compute_ppo_actor_loss`); embodied loss call `rlinf/workers/actor/fsdp_actor_worker.py:1451-1465`; advantage broadcast `rlinf/algorithms/utils.py:134-167`; dist params `rlinf/models/embodiment/smolvla/smolvla_action_model.py:341-378`.

### Execution rules (READ — this is how the plan is run)

**Mode: INLINE execution on `main` / `cluster3`.**
- Execute every task **inline in the main session** (we are NOT using the subagent-driven-development orchestration mode, and NOT using `git worktree`). Stay on the current branch of each repo. **No new branches, no worktrees.**
- Because there is no branch isolation, **commit after every task (and every green test)** with `git commit -s` (DCO sign-off required by RLinf; harmless for the project repo). Small, frequent commits are the safety net — if a step breaks, the previous commit is the rollback point.
- **Selective subagents (inline-first):** Before each task, decide inline vs subagent. Default inline for ≤2-file edits with known locations. Spawn subagents for: code search/locate (`Explore`/`cavecrew-investigator`), SLURM log RCA on failure (`general-purpose`/`cavecrew-investigator`), risky diff review before commit (`cavecrew-reviewer`). Every subagent prompt MUST include `/caveman ultra`.
- TDD: write the failing test → see it fail → implement → see it pass → commit. Do not skip the "see it fail" step.
- Phase A is the **primary, graded path** — finish AND validate (beat baseline) before starting Phase B.

**Subagent usage policy (selective, not per-task).**
We still use subagents — but selectively, deciding *per task* whether one helps, rather than spawning one for every task. **Before starting each task, decide and note: "inline" or "subagent + which type".** Default to inline for the small focused edits in this plan. Spawn a subagent when the task is genuinely parallelizable or context-heavy:
- **Code search / locate** (e.g. "find the exact checkpoint-loader method name in `policy_wrapper.py`", "find the update-loop counter variable"): use `Explore` or `cavecrew-investigator`.
- **Log triage / RCA on a failed SLURM job** (sifting big `.err`/`.out`): use a `general-purpose` or `cavecrew-investigator` subagent to extract the root cause, so the main session context stays lean.
- **Code review of a finished task's diff** before committing a risky change (A4 loss wiring, B3/B4 worker changes): use `cavecrew-reviewer` or `requesting-code-review`.
- Keep inline: the pure-math module (A1), CLI/config edits (A5/B5), and anything under ~2 files where you already know the location.
- **Every subagent you spawn MUST be told to use `/caveman ultra`** (put it in the subagent prompt). This keeps their returned output compressed so the long overnight session doesn't blow context.

**Autonomous overnight execution contract (the user is ASLEEP — do not wait for input).**
This plan must run end-to-end without a human. Concretely:
1. **Implement → test → commit** each task without pausing for approval.
2. **Queue jobs onto the GPU yourself.** Phase A is not "done" at code — you must `sbatch` the smoke, confirm it is green, then `sbatch` the train+eval arms (Task A6/A7) onto the `a30` partition, exactly like every prior SmolVLA/MetaWorld run.
3. **Poll and monitor** queued jobs (`squeue -u $USER`, tail `.out`/`.err`) on a cadence until they finish. Use a background monitor loop where appropriate (see Task A6 Step 3 / the existing `scripts/grpo/queue_flow_sde_moonshots_when_slots.sh` pattern) so you keep working while jobs run.
4. **On ANY break (smoke fail, train crash, OOM, NaN, eval error): do a FULL root-cause diagnosis** from the actual `.err`/`.out` logs (not a guess), apply the fix, **resubmit**, and **continue autonomously**. Record every failure + RCA + fix + new job id in `docs/dgpo_overnight_log.md`. Known failure modes already logged there (Ray collision, `QOSMaxMemoryPerUser` at 100G → use ≤50G, A30 OOM → offload/`expandable_segments`) — check it first.
5. **Compute budget:** 3× A30, ≤32 CPU and ≤200 GB RAM total. **One job at a time** until DGPO is confirmed green, then up to **2 parallel** hparam arms (E1 + one of E2/E3). Never exceed the budget; if `QOSMaxMemoryPerUser` trips, drop to `--mem=50G` and resubmit.
6. **Do not stop** until the Phase A verification gate (A7) is met or you hit a genuinely unrecoverable blocker — in which case log it clearly in `docs/dgpo_overnight_log.md` for the user to read on waking, and keep any still-recoverable arms running.

### Default hyperparameters (locked)
`τ=0.5`, `κ=0.0` (deviation-only), `ref=frozen_sft`, `clip_eps=0.2`, `group_size=16`, `lr=7.5e-6`, `num_updates=30`, `reward_mode=sparse_success_delta`, `flow_sde_noise_level=1.0`, `chunk_len=5`, `max_steps=120`. Eval: 25ep then 100ep at updates 10/20/30, seeds 1000+, `max_episode_steps=150`.

---

## File Structure

**Phase A (project repo `/vol/bitbucket/aa6622/project`):**
- Create: `src/smolvla_grpo/dgpo.py` — pure math: `gaussian_hellinger_sq`, `dgpo_redistribution_weights`. No torch-cuda, no policy deps. One responsibility: the DGPO weight math.
- Create: `tests/test_dgpo_math.py` — unit tests for the pure math.
- Modify: `scripts/grpo/train_phase11_env_on_policy_grpo.py` — add frozen-ref construction (inline, after `:743`), module-level `_compute_dgpo_chunk_weights` (~`:230`), thread per-valid-chunk weights into the **chunk-path inline loss loop (`:1108-1148`)**, CLI flags (`:533-667`), and weight telemetry. (Do NOT edit `_backward_phase11_group_loss` — it is the non-chunk path.)
- Create: `tests/test_dgpo_chunk_weights.py` — test the trainer helper with a tiny fake wrapper/rollout.
- Create: `scripts/grpo/submit_dgpo_chunk_grpo_train_eval_a30.slurm` — clone of the proven flow-sde slurm + DGPO flags.
- Modify: `docs/dgpo_overnight_log.md` — append DGPO (distribution-guided) run results.

**Phase B (RLinf repo `/vol/bitbucket/aa6622/RLinf-smolvla-metaworld-ppo-grpo`):**
- Create: `rlinf/algorithms/dgpo.py` — same math, RLinf-tensorized (`[bsz, num_chunk]`).
- Create: `tests/unit_tests/test_dgpo_redistribution.py`.
- Modify: `rlinf/models/embodiment/smolvla/smolvla_action_model.py` — return per-chunk `action_mean` from the training forward.
- Modify: `rlinf/workers/actor/fsdp_actor_worker.py` — frozen-ref mean precompute + advantage rescale before `policy_loss`.
- Create: `examples/embodiment/config/metaworld_pushv3_dgpo_smolvla.yaml`.
- Create: `scripts/slurm/smolvla_rlinf_dgpo_{smoke,train,eval100}_a30.slurm`.

---

# PHASE A — Distribution-Guided DGPO on the Phase-11 trainer (PRIMARY)

### Task A1: Pure DGPO math module (`src/smolvla_grpo/dgpo.py`)

**Files:**
- Create: `src/smolvla_grpo/dgpo.py`
- Test: `tests/test_dgpo_math.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_dgpo_math.py
import math
import torch
from smolvla_grpo.dgpo import gaussian_hellinger_sq, dgpo_redistribution_weights


def test_hellinger_zero_when_identical():
    mu = torch.randn(3, 5, 4)
    log_std = torch.zeros(3, 5, 4)
    d = gaussian_hellinger_sq(mu, mu, log_std, log_std)
    assert d.shape == (3, 5)
    assert torch.allclose(d, torch.zeros(3, 5), atol=1e-6)


def test_hellinger_bounded_and_monotonic():
    log_std = torch.zeros(1, 1, 1)
    mu_a = torch.zeros(1, 1, 1)
    d_small = gaussian_hellinger_sq(mu_a, mu_a + 0.5, log_std, log_std)
    d_big = gaussian_hellinger_sq(mu_a, mu_a + 5.0, log_std, log_std)
    assert 0.0 <= d_small.item() < d_big.item() <= 1.0


def test_weights_unit_mean_and_uniform_when_no_deviation():
    # all deviations equal -> softmax uniform -> all weights == 1.0
    dev = torch.zeros(2, 6)
    mask = torch.ones(2, 6, dtype=torch.bool)
    w = dgpo_redistribution_weights(dev, mask, tau=0.5, kappa=0.0)
    assert torch.allclose(w, torch.ones(2, 6), atol=1e-5)
    # unit mean over valid chunks
    assert torch.allclose(w.mean(dim=1), torch.ones(2), atol=1e-5)


def test_weights_concentrate_on_high_deviation_chunk():
    dev = torch.tensor([[0.0, 0.0, 0.9, 0.0]])
    mask = torch.ones(1, 4, dtype=torch.bool)
    w = dgpo_redistribution_weights(dev, mask, tau=0.5, kappa=0.0)
    assert w[0, 2] > w[0, 0]
    assert torch.allclose(w.mean(dim=1), torch.ones(1), atol=1e-5)


def test_weights_respect_mask_unit_mean_over_valid_only():
    dev = torch.tensor([[0.1, 0.2, 0.3, 0.0]])
    mask = torch.tensor([[True, True, True, False]])
    w = dgpo_redistribution_weights(dev, mask, tau=0.5, kappa=0.0)
    # masked chunk weight is 0; mean over the 3 valid == 1
    assert w[0, 3].item() == 0.0
    assert abs(w[0, :3].mean().item() - 1.0) < 1e-5
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /vol/bitbucket/aa6622/project && PYTHONPATH=/vol/bitbucket/aa6622/project:/vol/bitbucket/aa6622/project/src /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python -m pytest tests/test_dgpo_math.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'smolvla_grpo.dgpo'`.
**(Every pytest command in this plan must be prefixed with `PYTHONPATH=/vol/bitbucket/aa6622/project:/vol/bitbucket/aa6622/project/src` — there is no editable install.)**

- [ ] **Step 3: Implement the module**

```python
# src/smolvla_grpo/dgpo.py
"""Distribution-Guided Policy Optimization (arXiv 2605.03327) math, adapted to
continuous diagonal-Gaussian action-chunk policies.

DGPO keeps the GRPO trajectory advantage and redistributes it across the
episode's chunks via a softmax of the per-chunk Hellinger deviation between the
current and a reference policy. This module is pure (CPU/GPU tensors only).
"""
from __future__ import annotations

import torch


def gaussian_hellinger_sq(
    mu_a: torch.Tensor,
    mu_b: torch.Tensor,
    log_std_a: torch.Tensor,
    log_std_b: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Squared Hellinger distance between two diagonal Gaussians, per chunk.

    Inputs are shape [B, T, D] (batch, chunks, action-dim). Returns [B, T] in
    [0, 1]. d = 1 - BC, BC = prod_k sqrt(2 s_a s_b/(s_a^2+s_b^2)) *
    exp(-(mu_a-mu_b)^2 / (4 (s_a^2+s_b^2))).
    """
    var_a = torch.exp(2.0 * log_std_a).clamp(min=eps)
    var_b = torch.exp(2.0 * log_std_b).clamp(min=eps)
    var_sum = var_a + var_b
    coef = torch.sqrt((2.0 * torch.sqrt(var_a * var_b)) / var_sum)
    expo = torch.exp(-((mu_a - mu_b) ** 2) / (2.0 * var_sum))
    bc = (coef * expo).prod(dim=-1)  # Bhattacharyya coefficient over D
    return (1.0 - bc).clamp(0.0, 1.0)


def _normalized_entropy(log_std: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-chunk diagonal-Gaussian differential entropy, min-max normalized to
    [0,1] within each trajectory's valid chunks. Shape [B, T, D] -> [B, T].
    Constant log_std -> constant -> normalized to ones (gate inert)."""
    ent = (log_std + 0.5 * (1.0 + torch.log(torch.tensor(2.0 * torch.pi)))).sum(dim=-1)  # [B,T]
    neg_inf = torch.finfo(ent.dtype).min
    masked = torch.where(mask, ent, torch.full_like(ent, neg_inf))
    e_max = masked.max(dim=1, keepdim=True).values
    masked_pos = torch.where(mask, ent, torch.full_like(ent, torch.finfo(ent.dtype).max))
    e_min = masked_pos.min(dim=1, keepdim=True).values
    rng = (e_max - e_min).clamp(min=eps)
    norm = ((ent - e_min) / rng).clamp(0.0, 1.0)
    # if range collapsed (constant entropy), treat gate as fully open (=1)
    collapsed = (e_max - e_min) <= eps
    return torch.where(collapsed, torch.ones_like(norm), norm)


def dgpo_redistribution_weights(
    deviations: torch.Tensor,
    mask: torch.Tensor,
    tau: float = 0.5,
    kappa: float = 0.0,
    entropy_norm: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Unit-mean softmax redistribution weights w_{i,t} per trajectory.

    deviations, mask: [B, T]. entropy_norm: [B, T] or None.
    Returns w: [B, T], masked chunks = 0, (1/T_i) sum_t w = 1 over valid chunks.
    kappa=0 -> pure deviation (default; SmolVLA entropy gate is inert).
    """
    dev = deviations
    if kappa != 0.0 and entropy_norm is not None:
        score = dev * entropy_norm.clamp(min=eps) ** kappa
    else:
        score = dev
    logits = score / max(float(tau), eps)
    neg_inf = torch.finfo(logits.dtype).min
    logits = torch.where(mask, logits, torch.full_like(logits, neg_inf))
    # numerically-stable softmax over chunk axis (valid only)
    logits = logits - logits.max(dim=1, keepdim=True).values
    exps = torch.where(mask, torch.exp(logits), torch.zeros_like(logits))
    denom = exps.sum(dim=1, keepdim=True).clamp(min=eps)
    counts = mask.sum(dim=1, keepdim=True).clamp(min=1).to(exps.dtype)
    w = counts * exps / denom
    return torch.where(mask, w, torch.zeros_like(w))
```

- [ ] **Step 4: Run to verify pass**

Run: `cd /vol/bitbucket/aa6622/project && PYTHONPATH=/vol/bitbucket/aa6622/project:/vol/bitbucket/aa6622/project/src /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python -m pytest tests/test_dgpo_math.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
cd /vol/bitbucket/aa6622/project
git add src/smolvla_grpo/dgpo.py tests/test_dgpo_math.py
git commit -s -m "feat(dgpo): add distribution-guided redistribution math"
```

---

### Task A2: Frozen SFT reference policy in the trainer

**Files:**
- Modify: `scripts/grpo/train_phase11_env_on_policy_grpo.py` — **inline in the `if __name__ == "__main__":` block, immediately after the `train_wrapper` setup block ends (after `:743` `freeze_all_but_grpo_trainables(bundle.policy)` / before `optimizer = ...` at `:745`).** The needed names (`load_bundle_for_grpo`, `MetaWorldSmolVLAGRPOPolicy`, `freeze_all_but_grpo_trainables`) are already imported at `:517-525` in this same block.

**Goal:** Build a second wrapper `ref_wrapper` = a **fresh bundle loaded from the SAME `args.checkpoint`** wrapped exactly like `train_wrapper`, fully frozen, never updated. Built only when `--dgpo` and `--dgpo-ref frozen_sft`. (Phase A implements only `frozen_sft`; `rollout`-ref needs a live-μ variant — out of scope, see A5.)

**Subagent decision:** edit inline — the construction code to mirror is already located (`:714-742`), no subagent needed.

- [ ] **Step 1: Insert the ref-builder inline (right after `:743`)**

```python
    # --- DGPO frozen SFT reference policy (for Hellinger deviation) ---
    ref_wrapper = None
    if getattr(args, "dgpo", False):
        if args.dgpo_ref != "frozen_sft":
            raise NotImplementedError(
                f"--dgpo-ref={args.dgpo_ref} not implemented in Phase A (use frozen_sft)"
            )
        ref_bundle, ref_action_dim = load_bundle_for_grpo(
            args.checkpoint,
            task=args.task,
            env_backend=args.env_backend,
            n_action_steps=(
                int(args.rollout_chunk_len)
                if args.rollout_unit == "chunk"
                else int(args.action_chunk_size)
            ),
        )
        ref_wrapper = MetaWorldSmolVLAGRPOPolicy(
            ref_bundle,
            task=args.task,
            task_text=task_text,
            camera_name=camera_name,
            flip_corner2=flip_corner2,
            action_dim=ref_action_dim,
            action_transform=args.action_transform,
            min_log_std=float(args.min_log_std),
            gaussian_logprob_action=args.gaussian_logprob_action,
            logprob_mode=args.logprob_mode,
            flow_sde_noise_level=float(args.flow_sde_noise_level),
            flow_sde_trace_step=int(args.flow_sde_trace_step),
        )
        ref_wrapper.assert_grpo_api()
        ref_wrapper.set_log_std(args.init_log_std)
        ref_wrapper.set_euler_step_noise_std(args.euler_step_noise_std)
        for prm in ref_bundle.policy.parameters():
            prm.requires_grad_(False)
        ref_bundle.policy.eval()
        print(f"[dgpo] frozen SFT reference ready (ref={args.dgpo_ref})", flush=True)
```

**Memory note (FLAG):** this loads a second full SmolVLA (~0.9 GB bf16 / ~1.8 GB fp32) onto the GPU. Fine on an A30 (24 GB) alongside the train policy + rollout, but if Phase A OOMs, move `ref_bundle.policy` to CPU and the helper in A3 will `.to(device)` per call (slower). Watch VRAM in the smoke (A6).

- [ ] **Step 2: Commit (exercised end-to-end in A4/A6; no standalone test — it needs real model weights)**

```bash
cd /vol/bitbucket/aa6622/project
git add scripts/grpo/train_phase11_env_on_policy_grpo.py
git commit -s -m "feat(dgpo): load frozen SFT reference policy for deviation"
```

---

### Task A3: Per-rollout chunk-weight helper

**Files:**
- Modify: `scripts/grpo/train_phase11_env_on_policy_grpo.py` — add a module-level helper near the other module-level helpers (e.g. right after `_log_std_telemetry`, ~`:230`). It is module-level (not inside the `__main__` block) so the A3 test can import it cheaply (top-of-file imports are light: only `torch` + `smolvla_grpo.process_memory`).
- Test: `tests/test_dgpo_chunk_weights.py`

**Goal:** For each rollout trajectory, compute one DGPO weight **per valid chunk**, in the SAME order the loss loop visits valid chunks (`for chunk in traj.chunks` then skip `if not chunk.valid_action_mask.any()`). Use the **stored** `chunk.distr_mean`/`chunk.distr_log_std` (current/old policy, `[chunk_len, action_dim]`) vs the **frozen-ref** μ/log_std recomputed from `chunk.flow_sde_trace`. Return weights aligned to valid chunks only.

**Subagent decision:** inline (location + data structures already mapped).

- [ ] **Step 1: Write the failing test (fakes — no GPU/model)**

```python
# tests/test_dgpo_chunk_weights.py
import torch
from scripts.grpo.train_phase11_env_on_policy_grpo import _compute_dgpo_chunk_weights


class _FakeChunk:
    def __init__(self, chunk_idx, *, chunk_len=5, adim=4, valid=True):
        # current/old-policy mean shifts with chunk index -> deviation grows
        self.distr_mean = torch.full((chunk_len, adim), float(chunk_idx))
        self.distr_log_std = torch.zeros(chunk_len, adim)
        self.valid_action_mask = torch.ones(chunk_len, dtype=torch.bool) if valid \
            else torch.zeros(chunk_len, dtype=torch.bool)
        self.proc_snapshot = chunk_idx
        self.flow_sde_trace = {"A_next": torch.zeros(1, chunk_len, adim)}


class _FakeTraj:
    def __init__(self, n, invalid_idx=()):
        self.chunks = [_FakeChunk(i, valid=(i not in invalid_idx)) for i in range(n)]


class _FakeRef:
    """Frozen ref returns mu=0 always -> deviation grows with chunk index."""
    def get_flow_sde_log_probs_for_chunk_from_proc_list(self, procs, traces, *, chunk_len):
        b = len(procs)
        adim = 4
        return (torch.zeros(b, chunk_len), torch.zeros(b, chunk_len, adim),
                torch.zeros(b, chunk_len, adim))


def test_weights_unit_mean_and_concentrate_on_high_deviation():
    rollouts = [_FakeTraj(4)]
    out = _compute_dgpo_chunk_weights(
        _FakeRef(), rollouts, chunk_len=5, tau=0.5, kappa=0.0, device=torch.device("cpu")
    )
    assert len(out) == 1
    w = out[0]                       # weights for the 4 valid chunks, in order
    assert len(w) == 4
    assert abs(sum(w) / len(w) - 1.0) < 1e-4   # unit mean over valid chunks
    assert w[3] > w[0]                          # bigger deviation -> bigger weight


def test_weights_align_to_valid_chunks_only():
    rollouts = [_FakeTraj(4, invalid_idx=(1,))]  # chunk 1 invalid -> skipped
    out = _compute_dgpo_chunk_weights(
        _FakeRef(), rollouts, chunk_len=5, tau=0.5, kappa=0.0, device=torch.device("cpu")
    )
    assert len(out[0]) == 3                      # only 3 valid chunks
    assert abs(sum(out[0]) / 3 - 1.0) < 1e-4
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /vol/bitbucket/aa6622/project && PYTHONPATH=/vol/bitbucket/aa6622/project:/vol/bitbucket/aa6622/project/src /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python -m pytest tests/test_dgpo_chunk_weights.py -v`
Expected: FAIL with `ImportError`/`AttributeError` (function not defined). (`PYTHONPATH` includes the repo root so `scripts.grpo....` imports; importing the trainer module is cheap because its heavy deps are lazy-imported inside `__main__`.)

- [ ] **Step 3: Implement `_compute_dgpo_chunk_weights` (module-level, ~`:230`)**

```python
@torch.no_grad()
def _compute_dgpo_chunk_weights(
    ref_wrapper,
    rollouts,
    *,
    chunk_len: int,
    tau: float,
    kappa: float,
    device,
):
    """One DGPO weight per VALID chunk, per trajectory, aligned to the loss loop.

    Current-policy Gaussian = stored chunk.distr_mean/distr_log_std (on-policy:
    equals pi_theta at update start). Reference = frozen SFT, recomputed from the
    stored flow-sde trace. Hellinger deviation -> unit-mean softmax over the
    trajectory's valid chunks. Returns list[list[float]] (per traj, valid order).
    """
    from smolvla_grpo.dgpo import (
        dgpo_redistribution_weights,
        gaussian_hellinger_sq,
        _normalized_entropy,
    )

    out: list[list[float]] = []
    for traj in rollouts:
        mu_c, ls_c, mu_r, ls_r = [], [], [], []
        for chunk in traj.chunks:
            if not bool(chunk.valid_action_mask.any()):
                continue  # MUST mirror the loss loop's skip (:1117-1118)
            mu_c.append(chunk.distr_mean.reshape(-1).float())       # [chunk_len*adim]
            ls_c.append(chunk.distr_log_std.reshape(-1).float())
            _, mr, lr = ref_wrapper.get_flow_sde_log_probs_for_chunk_from_proc_list(
                [chunk.proc_snapshot], [chunk.flow_sde_trace], chunk_len=int(chunk_len)
            )
            mu_r.append(mr.reshape(-1).float())
            ls_r.append(lr.reshape(-1).float())
        if not mu_c:
            out.append([])
            continue
        MC = torch.stack(mu_c).unsqueeze(0).to(device)   # [1, n_valid, D]
        LC = torch.stack(ls_c).unsqueeze(0).to(device)
        MR = torch.stack(mu_r).unsqueeze(0).to(device)
        LR = torch.stack(ls_r).unsqueeze(0).to(device)
        dev = gaussian_hellinger_sq(MC, MR, LC, LR)      # [1, n_valid]
        mask = torch.ones_like(dev, dtype=torch.bool)
        ent = _normalized_entropy(LC, mask) if kappa != 0.0 else None
        w = dgpo_redistribution_weights(dev, mask, tau=tau, kappa=kappa, entropy_norm=ent)
        out.append(w.reshape(-1).cpu().tolist())
    return out
```

**Caveat (minor):** for a partially-valid final chunk, the Hellinger uses the full `[chunk_len, action_dim]` padded means (the within-chunk action mask is not applied to the deviation). Fully-invalid chunks are skipped; partial chunks are rare (only an episode's last chunk). Acceptable for v1; refine later if `w_std` looks noisy.

- [ ] **Step 4: Run to verify pass**

Run: `cd /vol/bitbucket/aa6622/project && PYTHONPATH=/vol/bitbucket/aa6622/project:/vol/bitbucket/aa6622/project/src /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python -m pytest tests/test_dgpo_chunk_weights.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
cd /vol/bitbucket/aa6622/project
git add scripts/grpo/train_phase11_env_on_policy_grpo.py tests/test_dgpo_chunk_weights.py
git commit -s -m "feat(dgpo): per-valid-chunk weight helper"
```

---

### Task A4: Thread weights into the chunk-path inline loss loop

**Files:**
- Modify: `scripts/grpo/train_phase11_env_on_policy_grpo.py` — the **chunk-path inline loss loop at `:1108-1148`** (this is the live seam; the baseline runs `--rollout-unit chunk`). Do NOT touch `_backward_phase11_group_loss`.

**Goal:** When `--dgpo` is set, multiply the per-trajectory advantage `A` by the per-valid-chunk weight `w` so `A_eff = A * w`, applied at the `ratio * A` lines. When `--dgpo` is off, the code path is byte-for-byte unchanged (backward compatible).

**Subagent decision:** edit inline, then **before committing, spawn a `cavecrew-reviewer` subagent (prompt it to use `/caveman ultra`)** to review this exact diff — it is the one change that can silently corrupt the gradient (mis-aligned weights, lost backward-compat, wrong reset of the valid-chunk counter across epochs). Address findings, then commit.

- [ ] **Step 1: Compute weights once per update, just before the epoch loop**

Right after `bundle.policy.train()` (`:1102`) and before `for _epoch in range(args.update_epochs):` (`:1108`), insert:

```python
            dgpo_weights = None
            if getattr(args, "dgpo", False):
                dgpo_weights = _compute_dgpo_chunk_weights(
                    ref_wrapper,
                    rollouts,
                    chunk_len=int(args.rollout_chunk_len),
                    tau=float(args.dgpo_tau),
                    kappa=float(args.dgpo_kappa),
                    device=device,
                )
```

- [ ] **Step 2: Apply `A_eff = A * w` inside the loop, tracking a per-trajectory valid-chunk counter**

In the loop body (`:1113-1133`), change the trajectory/chunk iteration so it tracks a valid-chunk index `vc` and scales `A`. Replace:

```python
                for gi, traj in enumerate(rollouts):
                    A = advantages[gi].reshape(()).float()
                    for chunk in traj.chunks:
                        valid = chunk.valid_action_mask.reshape(1, -1).to(device)
                        if not bool(valid.any()):
                            continue
```
with:
```python
                for gi, traj in enumerate(rollouts):
                    A = advantages[gi].reshape(()).float()
                    w_list = dgpo_weights[gi] if dgpo_weights is not None else None
                    vc = 0  # valid-chunk index, MUST track the same skips as below
                    for chunk in traj.chunks:
                        valid = chunk.valid_action_mask.reshape(1, -1).to(device)
                        if not bool(valid.any()):
                            continue
                        A_eff = A
                        if w_list is not None:
                            if vc >= len(w_list):
                                raise RuntimeError("dgpo weight/valid-chunk misalignment")
                            A_eff = A * float(w_list[vc])
                        vc += 1
```
Then change the two advantage applications (`:1131-1132`) from `A` to `A_eff`:
```python
                        unclipped = ratio * A_eff
                        clipped = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * A_eff
```
Leave `apply_grpo_regularizers(...)` (`:1134-1141`) and the normalizer `valid_chunk_count` unchanged. **Note:** `dgpo_weights` is computed once (Step 1) outside the epoch loop, but `vc` is re-initialized inside `for _epoch` per trajectory — correct, because the same `w_list` is reused each epoch.

- [ ] **Step 3: Add a regression test that DGPO-off == GRPO (weights all 1.0)**

```python
# tests/test_dgpo_chunk_weights.py  (append)
def test_uniform_weights_equal_grpo_advantage():
    # all-equal deviation -> weights all 1.0 -> A*w == A (DGPO reduces to GRPO)
    from smolvla_grpo.dgpo import dgpo_redistribution_weights
    dev = torch.full((1, 8), 0.3)
    mask = torch.ones(1, 8, dtype=torch.bool)
    w = dgpo_redistribution_weights(dev, mask, tau=0.5, kappa=0.0)
    assert torch.allclose(w, torch.ones(1, 8), atol=1e-5)
```

- [ ] **Step 4: Run tests + a byte-level off-path check**

Run: `cd /vol/bitbucket/aa6622/project && PYTHONPATH=/vol/bitbucket/aa6622/project:/vol/bitbucket/aa6622/project/src /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python -m pytest tests/test_dgpo_math.py tests/test_dgpo_chunk_weights.py -v`
Expected: all pass.
Also confirm the trainer still parses/imports: `... /bin/python scripts/grpo/train_phase11_env_on_policy_grpo.py --help >/dev/null && echo OK`.

- [ ] **Step 5: Review (subagent) then commit**

Spawn the `cavecrew-reviewer` (with `/caveman ultra`) on the diff of `scripts/grpo/train_phase11_env_on_policy_grpo.py`. After addressing findings:
```bash
cd /vol/bitbucket/aa6622/project
git add scripts/grpo/train_phase11_env_on_policy_grpo.py tests/test_dgpo_chunk_weights.py
git commit -s -m "feat(dgpo): redistribute per-trajectory advantage across chunks"
```

---

### Task A5: CLI flags + telemetry

**Files:**
- Modify: `scripts/grpo/train_phase11_env_on_policy_grpo.py` (argparse section — search `add_argument` / `ArgumentParser`).

- [ ] **Step 1: Add args**

```python
    p.add_argument("--dgpo", action="store_true", help="enable Distribution-Guided DGPO advantage redistribution")
    p.add_argument("--dgpo-tau", type=float, default=0.5, help="DGPO softmax temperature")
    p.add_argument("--dgpo-kappa", type=float, default=0.0, help="DGPO entropy-gate exponent (0 = deviation only; inert with global log_std)")
    p.add_argument("--dgpo-ref", type=str, default="frozen_sft", choices=["frozen_sft", "rollout"], help="DGPO reference policy (Phase A implements frozen_sft only)")
```
Add these near the other `p.add_argument(...)` calls (anywhere in `:533-667`, before `args = p.parse_args()` at `:668`).

- [ ] **Step 2: Add telemetry — log weight stats each update**

In the chunk branch, immediately after the `dgpo_weights = _compute_dgpo_chunk_weights(...)` block from Task A4 Step 1 (so `dgpo_weights` and `update` are both in scope), add:

```python
                if dgpo_weights is not None and any(dgpo_weights):
                    import itertools
                    flat = list(itertools.chain.from_iterable(dgpo_weights))
                    if flat:
                        wt = torch.tensor(flat)
                        print(
                            f"[dgpo] update={update} w_min={wt.min():.3f} "
                            f"w_max={wt.max():.3f} w_std={wt.std():.3f} "
                            f"tau={args.dgpo_tau} kappa={args.dgpo_kappa} ref={args.dgpo_ref}",
                            flush=True,
                        )
```

`update` is the loop variable from `for update in range(start_u, end_u):` (`:847`). `w_std>0` here is the live confirmation that DGPO is actually redistributing (not collapsing to GRPO).

- [ ] **Step 3: Smoke-import the trainer to catch syntax/arg errors**

Run: `cd /vol/bitbucket/aa6622/project && /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python scripts/grpo/train_phase11_env_on_policy_grpo.py --help 2>&1 | grep -E "dgpo"`
Expected: the four `--dgpo*` flags printed.

- [ ] **Step 4: Commit**

```bash
cd /vol/bitbucket/aa6622/project
git add scripts/grpo/train_phase11_env_on_policy_grpo.py
git commit -s -m "feat(dgpo): CLI flags and weight telemetry"
```

---

### Task A6: SLURM job (clone proven flow-sde slurm + DGPO)

**Files:**
- Create: `scripts/grpo/submit_dgpo_chunk_grpo_train_eval_a30.slurm`

- [ ] **Step 1: Copy the proven slurm and add DGPO flags**

```bash
cd /vol/bitbucket/aa6622/project
cp scripts/grpo/submit_flow_sde_chunk_grpo_moonshot30_sparse_chain_a30.slurm \
   scripts/grpo/submit_dgpo_chunk_grpo_train_eval_a30.slurm
```

- [ ] **Step 2: Edit the new file**
- Change `#SBATCH --job-name=` to `dgpo-chunk-s30`.
- Change `OUT_ROOT` default dir to `.../artifacts/dgpo_chunk_grpo_sparse30`.
- In the trainer invocation block (the `train_phase11_env_on_policy_grpo.py` call), append:
  ```
    --dgpo \
    --dgpo-tau "${DGPO_TAU:-0.5}" \
    --dgpo-kappa "${DGPO_KAPPA:-0.0}" \
    --dgpo-ref "${DGPO_REF:-frozen_sft}" \
  ```
- Leave eval blocks unchanged (they evaluate checkpoints regardless of training algo).

- [ ] **Step 2b: Create a dedicated smoke slurm (2 updates) so the GPU smoke is one command**

```bash
cd /vol/bitbucket/aa6622/project
cp scripts/grpo/submit_dgpo_chunk_grpo_train_eval_a30.slurm \
   scripts/grpo/submit_dgpo_chunk_grpo_smoke_a30.slurm
```
Edit the smoke copy: `#SBATCH --job-name=dgpo-smoke`, `#SBATCH --mem=50G`, `#SBATCH --time=01:00:00`, set the trainer flags to `--num-updates 2 --save-every 1`, and **delete the eval100 block** (keep eval25 only) so the smoke is fast. This proves: DGPO weights compute, loss is finite, a checkpoint is written, eval runs.

- [ ] **Step 3: Queue the GPU smoke and monitor it autonomously (subagent: log-triage if it fails)**

Subagent note: run this inline; only spawn a `cavecrew-investigator` (with `/caveman ultra`) if the smoke fails and the `.err` is large.

Run: `cd /vol/bitbucket/aa6622/project && sbatch --parsable scripts/grpo/submit_dgpo_chunk_grpo_smoke_a30.slurm`
Capture the job id. Then poll until it leaves the queue:
```bash
JID=<jobid>; while squeue -j "$JID" -h | grep -q .; do sleep 60; done; \
echo "=== smoke done ==="; tail -40 dgpo_*"$JID".out dgpo_*"$JID".err 2>/dev/null
```
Expected: `[dgpo] update=...` telemetry with `w_std>0`, a written checkpoint, eval25 numbers, exit 0.
**If it breaks: full RCA from the log (check `docs/dgpo_overnight_log.md` for known modes — Ray collision, `QOSMaxMemoryPerUser`→`--mem=50G`, A30 OOM→offload), fix, resubmit, and log the failure+fix+new-jobid in `docs/dgpo_overnight_log.md`. Do not proceed to Step 4 until the smoke is green.**

- [ ] **Step 4: Once smoke is GREEN, queue the full E1 arm (30 updates + eval25 + eval100) and keep monitoring**

Run: `cd /vol/bitbucket/aa6622/project && DGPO_TAU=0.5 DGPO_KAPPA=0.0 DGPO_REF=frozen_sft sbatch --parsable scripts/grpo/submit_dgpo_chunk_grpo_train_eval_a30.slurm`
Record the job id in `docs/dgpo_overnight_log.md`. Poll as in Step 3 (longer cadence, ~270s, since it runs ~hours). This is the primary result arm — see Task A7 for the control + ablations to queue alongside (respecting the 1-then-2-parallel budget).

- [ ] **Step 5: Commit the slurm scripts**

```bash
cd /vol/bitbucket/aa6622/project
git add scripts/grpo/submit_dgpo_chunk_grpo_train_eval_a30.slurm scripts/grpo/submit_dgpo_chunk_grpo_smoke_a30.slurm
git commit -s -m "feat(dgpo): a30 smoke + train+eval slurm for distribution-guided dgpo"
```

---

### Task A7: Experiments + verification (beat 41%) — run autonomously overnight

**Orchestration (the user is asleep — drive this yourself):** queue arms with `sbatch`, poll `squeue -u $USER` on a ~270s cadence, tail each job's `.out`/`.err`, and on any failure do full RCA → fix → resubmit (log every step in `docs/dgpo_overnight_log.md`). **Sequencing under the budget:** queue **E0 (control) + E1 (DGPO)** first as the two parallel arms (both are needed for the apples-to-apples comparison); when a slot frees, queue **E2**, then **E3**. Never exceed 2 concurrent arms / 200 GB / 32 CPU. If `QOSMaxMemoryPerUser` trips, resubmit at `--mem=50G`. Reuse the `scripts/grpo/queue_flow_sde_moonshots_when_slots.sh` "submit when a slot is free" pattern for hands-off chaining if convenient.

- [ ] **E0 — control:** the existing flow-sde GRPO (no `--dgpo`) at the SAME seeds/updates, for a fresh apples-to-apples baseline on this machine. Use `submit_flow_sde_chunk_grpo_moonshot30_sparse_chain_a30.slurm` (already the 41% recipe).
- [ ] **E1 — DGPO primary:** `DGPO_TAU=0.5 DGPO_KAPPA=0.0 DGPO_REF=frozen_sft`, 30 updates, eval25+eval100 at 10/20/30.
- [ ] **E2 — τ flatter:** `DGPO_TAU=1.0` (closer to GRPO) vs E1.
- [ ] **E3 — τ sharper:** `DGPO_TAU=0.25` (more aggressive credit focus, higher variance) vs E1.
- [ ] **E4 (stretch) — entropy gate:** requires a state-dependent (heteroscedastic) std head; skip unless that is added, because κ is inert with the global `log_std`. Flag to supervisor as the top follow-up if E1 shows a lift.

- [ ] **Verification gate (record in `docs/dgpo_overnight_log.md`):**
  - DGPO E1 eval100 best-checkpoint success rate **> E0 baseline** (target: beat 41%).
  - Training stable: finite loss, `[dgpo] w_*` telemetry sane (`w_min>0`, `w_max` not exploding, `w_std>0` confirming redistribution is actually happening — if `w_std≈0` then deviation is ~uniform and DGPO≡GRPO, a real negative result worth noting).
  - `approx_kl` / `ratio_clip_fraction` comparable to the GRPO baseline (DGPO must not destabilize PPO).

- [ ] **Commit results doc** after each arm completes:
```bash
cd /vol/bitbucket/aa6622/project
git add docs/dgpo_overnight_log.md
git commit -s -m "docs(dgpo): record distribution-guided dgpo run <Ex>"
```

---

# PHASE B — Port validated DGPO into RLinf (SECONDARY, after Phase A beats baseline)

Only start once Phase A E1 has demonstrated a real lift. The math module and hyperparameters carry over; only the tensor plumbing differs.

### Task B1: RLinf DGPO math module

**Files:**
- Create: `RLinf-smolvla-metaworld-ppo-grpo/rlinf/algorithms/dgpo.py`
- Test: `RLinf-smolvla-metaworld-ppo-grpo/tests/unit_tests/test_dgpo_redistribution.py`

- [ ] **Step 1–4:** Copy `gaussian_hellinger_sq`, `_normalized_entropy`, `dgpo_redistribution_weights` verbatim from `src/smolvla_grpo/dgpo.py` into the new RLinf module (it is pure torch, no project deps). Port the same tests into `tests/unit_tests/test_dgpo_redistribution.py`.

Run: `cd /vol/bitbucket/aa6622/RLinf-smolvla-metaworld-ppo-grpo && python -m pytest tests/unit_tests/test_dgpo_redistribution.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**
```bash
cd /vol/bitbucket/aa6622/RLinf-smolvla-metaworld-ppo-grpo
git add rlinf/algorithms/dgpo.py tests/unit_tests/test_dgpo_redistribution.py
git commit -s -m "feat(dgpo): add distribution-guided redistribution math"
```

### Task B2: SmolVLA returns per-chunk action mean

**Files:**
- Modify: `rlinf/models/embodiment/smolvla/smolvla_action_model.py` (the `default_forward`/training forward that returns `output_dict` with `logprobs` — it already computes `(mean, log_std)` via `select_action_distr_params` at `:341-378`).

- [ ] **Step 1:** When `compute_logprobs=True`, also place `mean` and `log_std` (shape `[bsz, num_chunk, action_dim]`) into the returned `output_dict` as `output_dict["action_mean"]` and `output_dict["action_log_std"]`. Add a unit test asserting the keys + shapes given a synthetic proc (mirror the existing `test_smolvla_*` tests). Commit `feat(dgpo): expose per-chunk action mean/log_std from forward`.

### Task B3: Frozen-ref mean precompute in the actor worker

**Files:**
- Modify: `rlinf/workers/actor/fsdp_actor_worker.py` (the embodied training step around `:1406-1465`).

- [ ] **Step 1:** At trainer init, snapshot the initial (SFT) model state (mirror the NFT worker's `init_rollout_model` pattern in `fsdp_nft_policy_worker.py`). Add a `_precompute_dgpo_ref_means()` that, once per update, runs the frozen-ref weights over `forward_inputs` (offload/reload like `_recompute_v_old`) and stores `forward_inputs["dgpo_ref_action_mean"]` / `["dgpo_ref_action_log_std"]`. Gate on `cfg.algorithm.get("use_dgpo", False)`. Commit.

### Task B4: Redistribute advantages before `policy_loss`

**Files:**
- Modify: `rlinf/workers/actor/fsdp_actor_worker.py` (`:1448-1465`).

- [ ] **Step 1:** Between the model forward (`:1435`) and the `policy_loss(**kwargs)` call (`:1465`), when `use_dgpo`:
  ```python
  from rlinf.algorithms.dgpo import gaussian_hellinger_sq, dgpo_redistribution_weights
  # mean_cur: output_dict["action_mean"] [bsz, num_chunk, adim]; ref from forward_inputs
  dev = gaussian_hellinger_sq(mean_cur, mean_ref, ls_cur, ls_ref)   # [bsz, num_chunk]
  chunk_mask = loss_mask.reshape(bsz, num_chunk, -1).any(-1)        # [bsz, num_chunk]
  w = dgpo_redistribution_weights(dev, chunk_mask, tau=cfg.algorithm.dgpo_tau,
                                  kappa=cfg.algorithm.dgpo_kappa)     # [bsz, num_chunk]
  advantages = advantages * w.unsqueeze(-1)   # broadcast over action_dim
  ```
  Confirm `advantages` shape is `[bsz, num_chunk, action_dim]` here (from `postprocess_embodied_advantages_outputs`, `utils.py:167`); reshape `w` to match. Add metrics `actor/dgpo_w_std`, `actor/dgpo_dev_mean`. Commit `feat(dgpo): redistribute grpo advantage in embodied actor`.

### Task B5: Config + slurm + experiments

**Files:**
- Create: `examples/embodiment/config/metaworld_pushv3_dgpo_smolvla.yaml` — clone `metaworld_pushv3_native_grpo_smolvla.yaml`, set `noise_method: flow_sde`, `flow_sde_noise_level: 1.0`, and an `algorithm` block: `use_dgpo: True`, `dgpo_tau: 0.5`, `dgpo_kappa: 0.0`, `dgpo_ref: frozen_sft`. Keep `loss_type: actor`, `adv_type: grpo`, `kl_beta: 0.0`, `group_size: 16`. Scale `total_num_envs` to a multiple of 16.
- Create: `scripts/slurm/smolvla_rlinf_dgpo_{smoke,train,eval100}_a30.slurm` — clone the existing RLinf smolvla slurm templates, point `--config-name metaworld_pushv3_dgpo_smolvla`.

- [ ] Smoke (1 epoch) → train (20–30 ep) → eval100 (seeds 1000–1099). Verify RLinf-DGPO ≥ RLinf-GRPO control, and compare to the Phase A number. Record in `docs/dgpo_overnight_log.md`. Commit configs + results.

---

## Self-Review (run before declaring the plan done)

1. **Spec coverage:** paper math (A1/B1), Hellinger continuous adaptation (A1), entropy-gate caveat + κ=0 default (A1, context), frozen ref (A2/B3), redistribution into PPO loss (A4/B4), CLI/config (A5/B5), experiments incl. control + ablations (A7/B5), beat-41% verification gate (A7). ✔
2. **Placeholder scan:** the only deliberate "verify in code" notes are the checkpoint-loader method name (A2) and the update-counter variable name (A5) — both are explicit "read this file, use the real name" instructions, not silent TODOs. ✔
3. **Type consistency:** `gaussian_hellinger_sq(mu_a, mu_b, log_std_a, log_std_b)` and `dgpo_redistribution_weights(deviations, mask, tau, kappa, entropy_norm)` signatures are identical across A1, A3, B4. Weight tensors are `[B, T]`, unit-mean over valid chunks, masked→0 everywhere. ✔

## Risks / flags for the human
- **Entropy gate is inert** with SmolVLA's global `log_std` (κ has no effect). If E1 shows redistribution helps, a state-dependent std head is the highest-value follow-up to unlock the paper's full mechanism. **Flag for supervisor.**
- **Deviation may be near-uniform** early (π_θ≈π_ref) → DGPO≈GRPO for the first updates. Watch `w_std`; if it stays ~0 the whole run, DGPO has nothing to redistribute (a legitimate finding, not a bug). The frozen-SFT reference is what makes `w_std` grow over training.
- **τ sensitivity:** paper range 0.5–1.0; E2 covers it. Smaller τ = sharper credit focus (higher variance), larger τ = closer to GRPO.
- **Cost:** frozen-ref forward adds one extra no-grad pass per update (Phase A: cheap, reuses the chunk-batch recompute; Phase B: offload/reload like `_recompute_v_old`). On the A30, watch VRAM in Phase B.
