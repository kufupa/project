# Flow-SDE GRPO: Beat the push-v3 Baseline (Audit + Fix) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Get the Flow-SDE chunk GRPO line to clear the 21% (100-episode) push-v3 SmolVLA baseline, after establishing — with evidence — that the current "≤28%, oscillating" result is caused by toy training scale + a noise-floor eval, not a Flow-SDE math bug.

**Architecture:** The Flow-SDE port (`src/smolvla_grpo/flow_logprob.py`, lerobot `lerobot_mw_py310` fork) is a *faithful* implementation of π_RL Flow-SDE (verified against arXiv 2510.25889 §4.2 / Appendix J). The fix is not a rewrite — it is (1) measure on the real 100-episode metric, (2) raise the per-update data budget and iteration count to the regime where the already-proven Gaussian G8 line shows its +12pp, (3) remove a dead trainable, (4) only then ablate noise/denoise per the paper's own guidance.

**Tech Stack:** Python 3.10/3.12 venvs (`.envs/lerobot_mw_py310` = flow_sde train, `.envs/rlinf_smolvla_mw_py312_fresh_nodeps` = eval), forked LeRobot SmolVLA, MetaWorld push-v3 via official LeRobot adapter, Slurm (a30 partition, ≤2 GPUs), GRPO trainer `scripts/grpo/train_phase11_env_on_policy_grpo.py`.

---

## Background: Why the 16-update run looked like a failure (95% confidence)

Evidence gathered from code + run artifacts + the π_RL paper:

| Lever | π_RL Flow-SDE (paper, Appendix J) | Working Gaussian G8 (your latex, 33%@100ep) | Your 16-update Flow-SDE run |
|---|---|---|---|
| Parallel envs | 64 | vector_async (parallel) | **1 (serial), group of 8 sequential** |
| Rollout epochs | 8 | — | 1 |
| Update (PPO) epochs | 4 | 1 | **1** |
| Train iterations | 450–500 | ~100 (win by u10) | **16** |
| Global batch | 2048 | larger | ~192 chunks |
| Algorithm | PPO + critic + GAE | GRPO (no critic) | GRPO (no critic) |
| Actor LR | 5e-6 | **1e-5** | **5e-6** |
| Eval | 50 tasks × many states | **100 ep, seeds 1000–1099** | **25 ep, seeds 1000–1024** |
| Denoise steps K | 4–5 | n/a | 10 (fine) |
| Noise level a | 0.5 | n/a | 0.5 (fine) |
| GPUs | 8× H100 | 1× A30 | 1× A30 |

**Root causes (ranked):**

1. **Under-training by ~1000× in env-steps.** ~14k env steps total (8 ep × ~110 steps × 16 updates) vs the paper's ~12M+. Net parameter movement on the ~100M action expert ≈ 1e-4 (LR 5e-6 × grad clipped to norm 1 × 16 steps). The deterministic ODE action used at eval barely moves → eval ≈ SFT baseline. Paper §5.4 (line 446): the deterministic-eval baseline is *invariant* to the stochastic training unless the expert actually shifts. Even the proven Gaussian G8 is flat in its first 15 updates (`artifacts/phase111_grpo_correctness_g8_20260530_231533/progress.jsonl`).

2. **Eval was statistically blind.** n=25 → SE ≈ ±9pp around a ~21–28% baseline. The "oscillation 12–28%" is sampling noise. Your own latex names the decision metric as 100ep/seeds 1000–1099 and warns 25ep "looks much stronger/noisier." The G8 +12pp was only credible at n=100 (SE ±4pp). `best_update=0` (the baseline) is exactly what a noise-floor metric on an unchanged policy produces.

3. **Structurally weaker than the recipe that works.** `update_epochs=1` → importance ratio ≡ 1 every step (`parity_mean_ratio=1.0000` on all 16 updates — uninformative, not a bug), no PPO sample reuse. GRPO group-8 advantage is far higher-variance than the paper's PPO+GAE+critic@batch2048. Chunk-level credit assignment is the paper's *least* reliable setting (line 456: larger chunk → diminished explained variance). LR was 5e-6 vs the G8's proven 1e-5.

4. **Bug — dead trainable.** `freeze_all_but_grpo_trainables` (`src/smolvla_grpo/policy_wrapper.py:901`) unfreezes `model.log_std`, but in `flow_sde` mode exploration σ comes from the schedule `a·√(τ/(1-τ))` and the logprob is `_flow_sde_logprob(A_next, mu_tau, sigma_tau)` — `log_std` never enters the graph → zero gradient. Harmless to correctness, but wasted optimizer state and a removed knob that *did* help the Gaussian path.

5. **High-variance gradient from random trace-step.** `--flow-sde-trace-step -1` perturbs a random denoise step τ per chunk; σ_τ spans ~0.05→0.5 so per-sample gradient ∝ 1/σ² varies ~100×. Fine at batch 2048; vicious at 8 episodes.

**Verified NOT a bug:** the Flow-SDE transition math (`flow_logprob.py:sde_step_params`, lerobot `_flow_sde_transition_params`) matches paper Eq. 8 (σ_τ = a√(τ/(1-τ)), Euler mean + SDE correction) and §4.2.3 (one random SDE step, rest ODE). Gradients are healthy (grad_norm 7–18), advantages well-formed (±2). Codex implemented it faithfully; it was just run at toy scale and judged on a noisy metric.

**2-GPU reality check:** We cannot replicate 8×H100 / 64-env scale, so the MT50 78% number is out of reach. The achievable, defensible goal is the same one the Gaussian G8 already hit: **beat 21% on 100ep push-v3, target 30–40%.** Plan keeps to ≤2 GPUs.

---

## File Structure

- `scripts/grpo/train_phase11_env_on_policy_grpo.py` — GRPO trainer. Modify: drop `log_std` from flow_sde trainables; add `--lr-schedule cosine` (optional, paper Fig 14). No structural rewrite.
- `src/smolvla_grpo/policy_wrapper.py` — `freeze_all_but_grpo_trainables`. Modify: exclude `log_std` when `logprob_mode == "flow_sde"`.
- `scripts/grpo/submit_flow_sde_chunk_grpo_eval25_a30.slurm` — copy → `..._eval100_a30.slurm` (100ep, seeds 1000–1099).
- `scripts/grpo/submit_flow_sde_chunk_grpo_train16_a30.slurm` — copy → `..._train_fair_a30.slurm` (the scaled run).
- `project/tests/test_flow_logprob.py`, `test_smolvla_flow_sde_hook_static.py` — existing; reuse as regression guards.

---

## Task 1: Re-evaluate the EXISTING 16 checkpoints on the real 100-episode metric

Cheapest possible signal: does a hidden gain already exist that 25ep missed? ~4 min/ckpt.

**Files:**
- Create: `scripts/grpo/submit_flow_sde_chunk_grpo_eval100_a30.slurm`

- [ ] **Step 1: Copy the eval slurm and switch to the 100ep protocol**

Copy `scripts/grpo/submit_flow_sde_chunk_grpo_eval25_a30.slurm` to `..._eval100_a30.slurm`. In the new file, change the eval invocation so `num_episodes=100`, `nenvs=25`, `seed_base=1000` (seeds 1000–1099), pointing `--ckpt` at `artifacts/flow_sde_chunk_grpo_train16/246503/checkpoints` and out at `artifacts/flow_sde_chunk_grpo_eval100/246503`. Keep `--include-baseline`.

- [ ] **Step 2: Submit and capture job id**

Run: `cd /vol/bitbucket/aa6622/project && sbatch scripts/grpo/submit_flow_sde_chunk_grpo_eval100_a30.slurm`
Expected: `Submitted batch job <ID>`.

- [ ] **Step 3: Read results when done**

Run: `grep SMOLVLA_EVAL_RESULT artifacts/flow_sde_chunk_grpo_eval100/246503/eval_*/results.jsonl 2>/dev/null || cat flow_sde_chunk_grpo_eval100_<ID>.out`
Expected: per-checkpoint `success_rate` over 100 episodes + a `baseline` row.

- [ ] **Step 4: Decision gate (no code; record in `research_log.md`)**

- If any checkpoint ≥ baseline + 8pp at 100ep → a real signal already exists at 16 updates; jump to Task 4 to scale and lock it in.
- If all within ±5pp of baseline (expected) → confirms under-training; proceed Task 2 → 4.
Append the verdict + numbers to `research_log.md`.

---

## Task 2: Remove the dead `log_std` trainable in flow_sde mode

**Files:**
- Modify: `src/smolvla_grpo/policy_wrapper.py:901-919` (`freeze_all_but_grpo_trainables`)
- Test: `project/tests/test_grpo_policy_wrapper_static.py`

- [ ] **Step 1: Write the failing test**

Add to `project/tests/test_grpo_policy_wrapper_static.py`:

```python
def test_flow_sde_excludes_log_std_from_trainables():
    import torch.nn as nn
    from smolvla_grpo.policy_wrapper import freeze_all_but_grpo_trainables

    class _Expert(nn.Module):
        def __init__(self): super().__init__(); self.w = nn.Linear(2, 2)
    class _Vlm(nn.Module):
        def __init__(self): super().__init__(); self.lm_expert = _Expert()
    class _Model(nn.Module):
        def __init__(self):
            super().__init__(); self.vlm_with_expert = _Vlm()
            self.log_std = nn.Parameter(torch.zeros(1, 1, 4))
    class _Policy(nn.Module):
        def __init__(self): super().__init__(); self.model = _Model()

    pol = _Policy()
    trainables = freeze_all_but_grpo_trainables(pol, logprob_mode="flow_sde")
    assert all(t is not pol.model.log_std for t in trainables)
    assert pol.model.log_std.requires_grad is False
    # gaussian mode keeps it trainable
    trainables_g = freeze_all_but_grpo_trainables(pol, logprob_mode="gaussian")
    assert any(t is pol.model.log_std for t in trainables_g)
    assert pol.model.log_std.requires_grad is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /vol/bitbucket/aa6622/project && .envs/lerobot_mw_py310/bin/python -m pytest tests/test_grpo_policy_wrapper_static.py::test_flow_sde_excludes_log_std_from_trainables -v`
Expected: FAIL (`freeze_all_but_grpo_trainables() got an unexpected keyword argument 'logprob_mode'`).

- [ ] **Step 3: Implement the change**

In `src/smolvla_grpo/policy_wrapper.py`, change the signature and the `log_std` branch:

```python
def freeze_all_but_grpo_trainables(policy: Any, *, logprob_mode: str = "gaussian") -> list[nn.Parameter]:
    """lm_expert always trainable; log_std trainable only when it enters the logprob (gaussian)."""
    for p in policy.parameters():
        p.requires_grad = False
    trainable: list[nn.Parameter] = []
    model = getattr(policy, "model", None)
    if model is None:
        return trainable
    vlm = getattr(model, "vlm_with_expert", None)
    if vlm is not None:
        expert = getattr(vlm, "lm_expert", None)
        if expert is not None:
            for p in expert.parameters():
                p.requires_grad = True
            trainable.extend(p for p in expert.parameters() if p.requires_grad)
    if logprob_mode != "flow_sde" and hasattr(model, "log_std") and isinstance(model.log_std, nn.Parameter):
        model.log_std.requires_grad = True
        trainable.append(model.log_std)
    return trainable
```

- [ ] **Step 4: Pass `logprob_mode` at the call site**

In `scripts/grpo/train_phase11_env_on_policy_grpo.py`, find the `freeze_all_but_grpo_trainables(` call and pass `logprob_mode=args.logprob_mode`.

Run: `grep -n "freeze_all_but_grpo_trainables(" scripts/grpo/train_phase11_env_on_policy_grpo.py`
Then edit that call to `freeze_all_but_grpo_trainables(bundle.policy, logprob_mode=args.logprob_mode)`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /vol/bitbucket/aa6622/project && .envs/lerobot_mw_py310/bin/python -m pytest tests/test_grpo_policy_wrapper_static.py -v`
Expected: PASS (new test + existing tests green).

- [ ] **Step 6: Commit**

```bash
cd /vol/bitbucket/aa6622/project
git add src/smolvla_grpo/policy_wrapper.py scripts/grpo/train_phase11_env_on_policy_grpo.py tests/test_grpo_policy_wrapper_static.py
git commit -m "fix(flow_sde): drop dead log_std from trainables in flow_sde mode"
```

---

## Task 3: Smoke-test the scaled config (2 updates, update_epochs=4) before the long run

Confirms `update_epochs=4` produces a *non-trivial* ratio after epoch 1 (proves PPO reuse is actually active for flow_sde) and that nothing crashes.

**Files:**
- Create: `scripts/grpo/submit_flow_sde_chunk_grpo_smoke_fair_a30.slurm`

- [ ] **Step 1: Create the smoke slurm**

Copy `scripts/grpo/submit_flow_sde_chunk_grpo_train16_a30.slurm` to `..._smoke_fair_a30.slurm`, and set the trainer flags to:

```
  --group-size 16 --num-updates 2 --train-seed-base 9000 \
  --max-steps 120 --lr 1e-5 --clip-eps 0.2 --update-epochs 4 \
  --init-log-std -2.0 --euler-step-noise-std 0.0 --min-log-std -4.0 \
  --action-transform no_tanh --logprob-mode flow_sde \
  --flow-sde-noise-level 0.5 --flow-sde-trace-step -1 \
  --rollout-unit chunk --rollout-chunk-len 5 --rollout-execution serial \
  --run-label flow_sde_smoke_fair --save-every 1
```
Remove `--fail-on-parity-violation` (with update_epochs>1 the post-epoch ratio legitimately leaves 1.0; the pre-update parity check still runs).

- [ ] **Step 2: Submit**

Run: `cd /vol/bitbucket/aa6622/project && sbatch scripts/grpo/submit_flow_sde_chunk_grpo_smoke_fair_a30.slurm`
Expected: `Submitted batch job <ID>`.

- [ ] **Step 3: Verify ratio moves and grad is non-zero**

Run: `python3 -c "import json; [print(r['update'], r.get('grad_norm_before_clip'), r.get('ratio_mean')) for r in map(json.loads, open('artifacts/flow_sde_chunk_grpo_smoke_fair/<ID>/progress.jsonl'))]"`
Expected: `grad_norm_before_clip` > 0 on both updates; the post-update `ratio_mean` (from `update_metrics`) is **not** exactly 1.0 (PPO reuse is active). If `ratio_mean` is still exactly 1.0, inspect the chunk update loop — the recompute must run under `bundle.policy.train()` with grad.

- [ ] **Step 4: Record outcome in `research_log.md`** (no commit; slurm files committed in Task 5).

---

## Task 4: Launch the fair-scale Flow-SDE run (≤2 GPUs, overnight)

Match the data budget where the Gaussian G8 already showed +12pp: more iterations, real PPO reuse, proven LR, bigger groups. Keep seed rotation (`2000+update`) for cross-layout generalization.

**Files:**
- Create: `scripts/grpo/submit_flow_sde_chunk_grpo_train_fair_a30.slurm`

- [ ] **Step 1: Create the training slurm**

Copy `..._train16_a30.slurm` to `..._train_fair_a30.slurm`. Set `--time=20:00:00` and trainer flags:

```
  --task push-v3 --env-backend official_lerobot --rollout-execution serial \
  --rollout-unit chunk --rollout-chunk-len 5 \
  --group-size 16 --num-updates 150 --train-seed-base 2000 \
  --max-steps 120 --lr 1e-5 --clip-eps 0.2 --update-epochs 4 \
  --init-log-std -2.0 --euler-step-noise-std 0.0 --min-log-std -4.0 \
  --action-transform no_tanh --logprob-mode flow_sde \
  --flow-sde-noise-level 0.5 --flow-sde-trace-step -1 \
  --run-label flow_sde_chunk_fair_u150 --save-every 5
```

Rationale for each non-default vs the 16-update run: `--update-epochs 4` (paper value; free 4× sample reuse), `--lr 1e-5` (G8's proven value), `--group-size 16` (2× samples/update; H5 in your latex shows group-16 reaching 52%@u74 at 25ep), `--num-updates 150` (paper ablations use ≥100; G8 win window is u10–u100).

- [ ] **Step 2: Estimate + flag walltime**

Per-update from the 16-update logs ≈ 150 s at group-8/epoch-1. Group-16 ≈ 2× rollout, update_epochs=4 adds optimize time (~4× of ~30 s). Estimate ≈ (2×111 rollout + 4×33 opt) ≈ 350 s/update × 150 ≈ **~15 h on one A30**. Fits the 20 h cap; with a 2nd GPU, split into two 75-update halves (`--start-update`) to halve wall-clock. **Flag to user before launch.**

- [ ] **Step 3: Submit**

Run: `cd /vol/bitbucket/aa6622/project && sbatch scripts/grpo/submit_flow_sde_chunk_grpo_train_fair_a30.slurm`
Expected: `Submitted batch job <ID>`.

- [ ] **Step 4: Monitor train trend (not the decision metric)**

Run periodically: `grep phase111_grpo_update flow_sde_chunk_grpo_train_fair_<ID>.out | tail -20`
Watch `avg_return` and `success_rate` trend over a 10-update moving window — train success should drift up if learning. Per paper line 840: if train rises but later eval oscillates, raise denoise steps (Task 6).

---

## Task 5: Evaluate the fair run on 100ep and pick the best checkpoint

**Files:**
- Reuse: `scripts/grpo/submit_flow_sde_chunk_grpo_eval100_a30.slurm` (from Task 1)

- [ ] **Step 1: Point eval at the fair-run checkpoints**

Run the eval100 slurm with args: ckpt dir `artifacts/flow_sde_chunk_grpo_train_fair/<ID>/checkpoints`, out `artifacts/flow_sde_chunk_grpo_eval100_fair/<ID>`, `--include-baseline`, seeds 1000–1099, 100 episodes.

Run: `cd /vol/bitbucket/aa6622/project && sbatch scripts/grpo/submit_flow_sde_chunk_grpo_eval100_a30.slurm <fair_ckpt_dir> <fair_eval_out>`

- [ ] **Step 2: Tabulate success_rate ± SE per checkpoint**

Run:
```bash
python3 - <<'PY'
import json, glob, math
rows=[json.loads(l) for f in glob.glob('artifacts/flow_sde_chunk_grpo_eval100_fair/*/eval_*/results.jsonl') for l in open(f)]
for r in sorted(rows, key=lambda r:r.get('update',-1)):
    p=r['success_rate']; se=math.sqrt(p*(1-p)/100)
    print(r.get('update'), f"{p:.2%} ± {se:.2%}", r['checkpoint'].split('/')[-1])
PY
```
Expected: a `baseline` row (~21%) + per-update rows. SE ≈ ±4pp at n=100.

- [ ] **Step 3: Decision gate (record in `research_log.md` + add a latex row)**

- **Success:** best checkpoint ≥ baseline + 8pp (≥ ~29%, ~2 SE) → Flow-SDE clears baseline. Record the checkpoint, update, and curve. Per your H5, the best is likely a *mid* checkpoint — report it, not u150.
- **Marginal/flat:** within ±5pp → proceed to Task 6 (paper-guided ablation) before concluding.

---

## Task 6: (Conditional) Paper-guided ablation if eval still oscillates

Only if Task 5 is marginal. The paper gives explicit remedies for exactly this symptom.

**Files:**
- Reuse training/eval slurms with overridden flags.

- [ ] **Step 1: Increase denoise steps (paper line 840: "eval oscillates → increase denoising steps")**

SmolVLA default `num_steps=10` (`configuration_smolvla.py:66`). Raise to 16 for the run: set `SMOLVLA_NUM_STEPS=16` env (or pass through config) in a copy of the train slurm; re-run a 60-update probe (`--num-updates 60`). Re-eval at 100ep. Compare best vs Task 5.

- [ ] **Step 2: Lower noise level a (paper line 838: large train/eval gap → reduce noise; line 448: lower noise needs smaller LR)**

Run a 60-update probe with `--flow-sde-noise-level 0.3` AND `--lr 5e-6` (paper's coupling: lower noise ⇒ larger gradients ⇒ smaller LR). Re-eval at 100ep.

- [ ] **Step 3: Record both probes; keep the winning (noise, denoise, lr) triple as the new default.**

---

## Task 7: (Optional, larger lift) PPO + critic to match the paper's actual algorithm

The paper's 78.1% used PPO + GAE + a critic (V attached to VLM output, or V_expert averaged over the denoising trajectory), not GRPO. GRPO's group-relative advantage is the main remaining variance source on ≤2 GPUs. Only attempt if Tasks 4–6 stall below baseline and time allows; this is a real feature, not a flag flip. Scope it as its own brainstorming + plan (value head on expert/VLM features, GAE on the dense return, `loss_type=actor_critic`), mirroring the RLinf `add_value_head` path already present for Direct PPO. **Flag to user as a separate ~1–2 day effort.**

---

## Risks & Flags (for the user, ≤2 GPUs)

- **Scale ceiling:** 64-env / 8×H100 is unreachable. Realistic target = beat 21% (aim 30–40%), not 78%. Single-task push-v3, not MT50.
- **Wall-clock:** ~15 h for 150 updates on one A30 (serial chunk rollout dominates). Use the 2nd GPU to split the run by `--start-update`, or accept overnight.
- **Chunk rollout is serial-only** (`flow_sde requires --rollout-unit chunk`, `chunk requires --rollout-execution serial`). True 64-env parallelism would need a new vector_async chunk path (code work, not in this plan).
- **Mid-checkpoint > final** (your H5): always eval every 5 updates and report the best mid checkpoint; expect late collapse.
- **Always decide on 100ep / seeds 1000–1099.** Never cite 25ep as a final number — it cannot resolve a <15pp change.
- **Two venvs:** train in `.envs/lerobot_mw_py310` (has flow_sde hooks), eval in `.envs/rlinf_smolvla_mw_py312_fresh_nodeps`. Never run flow_sde *rollout* under the eval venv — its `select_action_distr_params` silently drops flow_sde kwargs (no crash, wrong behavior).

---

## Self-Review Notes

- Spec coverage: explains *why* results are bad (Background, 95% conf), and *how* to fix (Tasks 1–7) — both halves of the user's ask.
- All flags verified against `scripts/grpo/train_phase11_env_on_policy_grpo.py` argparse (group-size, update-epochs, lr, clip-eps, flow-sde-noise-level, flow-sde-trace-step, rollout-unit/chunk-len/execution, save-every, train-seed-base, num-updates, start-update).
- `update_epochs=4` is supported by the chunk update loop (`for _epoch in range(args.update_epochs)`), which recomputes flow_sde logprobs each epoch → real ratio after epoch 1.
- No placeholders; each code step shows the code; each run step shows the command + expected output.
