# Flow-SDE GRPO: Root-Cause Analysis + Remediation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to run this plan task-by-task. Steps use checkbox (`- [ ]`) syntax. This is a **research/experiment** plan, so most steps are *gated GPU runs* with explicit success markers, plus a small number of code tasks. **No code was edited to produce this document.**

**Goal:** Explain — with 95% confidence — why the 16-update Flow-SDE chunk GRPO run (`flow_sde_chunk_grpo_train16`, job 246503) failed to beat baseline, then lay out the cheapest, most elegant path back to a real >baseline (target ≥40% @100ep) result.

**Verdict up front:** The Flow-SDE implementation is **not broken**. There is **no parity bug, no shape bug, no sign bug, and the SDE math matches the OpenPI reference.** The run produced no signal because it is a **strictly higher-variance objective run at a tiny scale, with a halved learning rate, a frozen (non-learnable) exploration knob, dense-reward advantages dominated by one outlier per group, and it was judged on a noisy 25-episode eval that the thesis itself flags as unreliable.** The "33% that always improves" was a *fragile early peak that then collapsed*, not a stable bar.

**Scope note:** This plan is read-only analysis + a remediation roadmap. It deliberately does **not** edit code or queue jobs.

---

## Part 0 — What actually ran (ground truth)

The run the user is annoyed about:

| Item | Value | Source |
|---|---|---|
| Train job | `246503` `flow_sde_chunk_grpo_train16`, 16 updates, all parity `1.0000` | `artifacts/flow_sde_chunk_grpo_train16/246503/progress.jsonl` |
| Eval job | `246506`, 25 ep, seeds **1000–1024** (NOT 1000–1099) | `artifacts/flow_sde_chunk_grpo_eval25/246503/eval25_summary.json` |
| Eval result | baseline **28%**; u2=12, u4=24, u6=24, u8=20, u10=28, u12=16, u14=16, u16=28 | same |
| Hparams | `lr=5e-6`, `clip=0.2`, `group=8`, `chunk_len=5`, `n_action_steps=5`, dense env reward, `no_tanh`, `logprob_mode=flow_sde`, `flow_sde_noise_level=0.5`, `flow_sde_trace_step=-1` (random/chunk), `update_epochs` effectively 1 | `scripts/grpo/submit_flow_sde_chunk_grpo_train16_a30.slurm` |

The known-good baseline it is implicitly compared against (from the LaTeX ledger, the trusted external results record):

| Item | Value | Source |
|---|---|---|
| "33% @100ep" | env-dense **Gaussian** GRPO, **G8, lr 1e-5, clip 0.2, chunk 5, no_tanh**, checkpoint **10** | `imperial_latex/evaluation/evaluation.tex:284`, `appendix/appendix.tex:138` |
| Protocol | **100 episodes, seeds 1000–1099, n_envs=25** | `evaluation.tex:266` |
| Stability | "later checkpoints degraded strongly"; 48%@25ep → 33%@100ep | `evaluation.tex:298,309` |
| Action exec | "**True action chunking; each policy sample provides a 5-step action chunk**" | `evaluation.tex:265` |

---

## Part A — Root-cause analysis (ranked, evidence-backed)

### A0. The Flow-SDE code is correct (rules out the "obvious bug" theories)

Verified directly, not from codex's narration:

- **Sampling is consistent with scoring.** In the venv hook `VLAFlowMatching.sample_actions` (`.envs/lerobot_mw_py310/.../modeling_smolvla.py:949-1060`): at the chosen denoise step `tau_idx` it injects `A_next = mu_tau + sigma_tau * noise` and **continues denoising from `A_next`**, returning the final `x_t` as the action. So the *executed* action genuinely incorporates the injected noise, and the scored log-prob `N(A_next | mu_tau, sigma_tau)` is the probability of that injected transition. This is exactly the OpenPI "non-joint" single-transition Flow-SDE estimator. **Consistent.**
- **The SDE μ/σ formula matches the reference.** `_flow_sde_transition_params` (`modeling_smolvla.py:883-895`) is identical to OpenPI `sample_mean_var_val` flow_sde branch (`rlinf/models/embodiment/openpi/openpi_action_model.py:780-787`): same `x0_pred/x1_pred`, `x0_weight/x1_weight`, `sigma_ratio = tau/(1-denom)`, `x_t_std = sqrt(delta)*sigma_i`, and the same "use next timestep when tau==1" rule. **No math bug.** (`project/src/smolvla_grpo/flow_logprob.py` is a duplicate of the same formula; the runtime path uses the venv hook for both sampling and recompute, so parity is self-consistent.)
- **Gradient flows to the trainable expert.** Recompute (`flow_sde_logprob_from_trace` → `denoise_step` → `action_out_proj`) backprops through `lm_expert`. `grad_norm_before_clip ≈ 8–18` every update, and eval numbers differ per checkpoint → **the model is being updated**, not frozen.
- **Loss is a standard clipped GRPO surrogate** (`train_phase11_env_on_policy_grpo.py:744-758`): `ratio=exp(new_lp-old_lp)`, `-min(ratio*A, clip(ratio)*A)`, advantage broadcast per-trajectory to its chunks. Sign is correct (oscillation, not monotone collapse-to-zero).

**Conclusion:** the failure is **not** a correctness bug. It is the experiment design. The rest of Part A is *why a correct Flow-SDE still produced no signal.*

### A1. (LEAD) The objective is a high-variance single-transition estimator run at ~no scale

- Per update there is **one group of 8 trajectories from a single reset seed** (`phase11_chunk_rollout.py:134-141`, all 8 share `env_h.reset(reset_seed)`), a **single gradient step** (`ratio_min=ratio_max=1.0`, `clip_fraction=0.0`, `approx_kl=0.0` on every update), and only **16 updates** total. That is ~128 trajectories of experience for the entire run.
- The scored signal is **one randomly-chosen denoise transition per chunk** (`flow_sde_trace_step=-1` → uniform over 10 steps). Early steps (tau≈1) inject large noise that the remaining 9 deterministic denoise steps largely wash out; late steps inject small noise directly. So the relationship between "this transition's log-prob" and "the executed action's quality" changes wildly chunk-to-chunk → large gradient variance.
- For reference, πRL gets its 78–86% Flow-SDE MetaWorld numbers with **64 envs × rollout_epoch 8 × PPO epoch 4 × global batch 2048 × ~450 epochs on 8×H100** (`docs/papers/2510.25889v3.md`). The single-transition (non-joint) estimator is a *variance-reduction* trick that still assumes that scale. At G8 × 16 updates × single epoch it cannot move.

**Evidence it's "no signal" rather than "collapse":** training rollout success oscillates 0–37.5% with **no trend**; eval oscillates 12–28% around the 28% baseline. The model is wandering, not diverging.

### A2. Dense-reward GRPO advantage is dominated by one outlier per group

Every update's 8 advantages are ~**one big positive (+2.3) and seven ≈ −0.4** (`progress.jsonl`), e.g. u0 `[…, 2.37, …]`. `group_return_std` is huge (27–237) on returns averaging 32–147. Because advantages are z-normalised **dense episode return** (`grpo_math.py compute_group_advantages`), the gradient each update chases the single highest-dense-return trajectory — which in MetaWorld dense reward is often a trajectory that lingered near a high-reward region **without succeeding**. So the optimiser is pulled toward dense-reward outliers, not toward success. This afflicts the old Gaussian G8 too, but combined with A1's weak signal it leaves zero net progress.

### A3. Exploration is frozen, and the trainable exploration knob is dead

In Flow-SDE mode the exploration magnitude is the fixed `flow_sde_noise_level=0.5` SDE schedule. The trainable `model.log_std` parameter is still **unfrozen and in the optimiser**, but it does **not** appear in the Flow-SDE objective (the log-prob uses `sigma_tau`, not `log_std`). Confirmed by `log_std_mean` drifting only with noise (−1.65…−1.80) and never systematically moving. So:
- the policy cannot *learn* to anneal/raise exploration the way the Gaussian path can (the Gaussian G8 trained `log_std`), and
- a no-op parameter is sitting in the trainable set.

### A4. The run changed three things at once vs the known-good recipe (confounded)

| Knob | Working G8 (33%) | Flow-SDE run (≤28%) |
|---|---|---|
| Objective | Gaussian surrogate (trainable `log_std`) | Flow-SDE single transition (fixed noise) |
| Learning rate | **1e-5** | **5e-6** (halved) |
| Rollout impl | `vector_async` | `serial` |
| (same) group/clip/chunk/reward/no_tanh | G8 / 0.2 / 5 / dense / yes | identical |

Even if Flow-SDE were neutral, **halving the LR** alone materially weakens a 16-update run. Because three variables moved together, the null result **cannot** be attributed to Flow-SDE. The experiment is not a clean test of its own hypothesis.

### A5. The comparison is apples-to-oranges, and the bar was never real

- The Flow-SDE run was evaluated on **25 ep, seeds 1000–1024** (`eval25`). The "33%" and "21% baseline" are **100 ep, seeds 1000–1099** (`evaluation.tex:266`). The thesis repeatedly warns 25-ep sweeps overstate and are "not reliable enough for final claims" (`evaluation.tex:309,469`). The "28% baseline" here is 7/25 — pure small-N noise; on 100 ep it would likely sit near the true 21%.
- The "33% that always improves" is, per the thesis's own words, a **fragile early peak that then degraded strongly** (`evaluation.tex:298`; `466`: "36% best @u12, 16% @u70"). The user's intuition "G8 always increases" is partly an artifact of best-checkpoint-on-25ep selection. So "never above 28%" is being measured against a bar that was itself noise + selection bias.

**Net 95%-confidence statement:** *Flow-SDE chunk GRPO produced no above-baseline signal because (1) it is a higher-variance objective than the Gaussian surrogate, (2) run at far too small a scale with half the LR and a single gradient step, (3) with dense-reward advantages dominated by one outlier per group, (4) with exploration frozen, and (5) it was scored on a noisy 25-ep eval rather than the 100-ep decision metric. The code itself is correct.*

---

## Part B — Critical review of codex's work

**Wrong / misleading conclusions:**

1. **"Phase46 lost true chunk execution → that explains the collapse."** The LaTeX (`evaluation.tex:265,239`) states the winning G8 used **5-step true chunking**. The new Flow-SDE run *also* uses 5-step chunking. So chunk-vs-step execution is **not** the differentiator. The step/chunk confusion was about a *different* Gaussian ablation rollout path, mis-generalised into the headline explanation.
2. **Treated "beat 33%" as a stable target.** 33% was a one-checkpoint peak that collapsed. The right target is a *protocol-fair, multi-checkpoint* comparison on 100 ep.
3. **Confounded the experiment** (objective + LR + rollout impl changed together), making the null uninterpretable — then proposed *more* Flow-SDE work on top of it.
4. **Chased the parity gate (`246480` fail at 0.0232 > 0.02)** as if it were the scientific problem. It was a numerical tolerance on a non-issue; the real problem was never parity.

**Missed / not addressed:**

5. Dense-reward advantage outlier domination (A2) — the single biggest lever on *success*.
6. `log_std` is a dead no-op in flow mode (A3) but left in the trainable set.
7. Eval protocol mismatch (25ep vs 100ep decision metric) (A5).
8. πRL's Flow-SDE numbers are inseparable from massive scale; on ≤2 GPUs the Gaussian chunk GRPO at thesis-best hparams is the more reliable path to ≥40%.

**Done well (keep):**

9. SDE μ/σ math is a faithful OpenPI port. Parity self-consistency is genuinely clean. Full-padded-trace replay (32-dim for denoiser, slice to 4 for scoring) is correct. Chunk-axis shape handling is correct. The TDD/commit discipline and gated smoke→train→eval flow are good engineering.

---

## Part C — Remediation plan

Design principles: **measure on the decision metric first; change one variable at a time; attack the real bottleneck (reward/advantage + scale) before re-betting on Flow-SDE; keep Flow-SDE as a *clean A/B*, not the default.** All runs honour the cluster cap (≤3 GPUs / 32 CPU / 200 GB) and emit a completion marker. Autonomous-friendly: each phase has a gate; on failure, RCA then continue.

### Phase 0 — Measurement truth (no training; ~1 GPU-hour)

**Why:** Until the existing checkpoints are scored on 100 ep seeds 1000–1099, we cannot even claim the run failed.

**Files:**
- Use: `project/scripts/grpo/submit_phase111_eval_sweep.slurm` (the correct 100-ep gate)
- Eval driver: `project/scripts/grpo/eval_phase11_checkpoints.py` (needs `--base-checkpoint`, `--grpo-checkpoint`, `--eval-seed-start 1000`, `--episodes 100`)
- Checkpoints: `artifacts/flow_sde_chunk_grpo_train16/246503/checkpoints/`

- [ ] **Step 1:** Re-evaluate baseline + every saved Flow-SDE checkpoint (u2…u16) at **100 ep, seeds 1000–1099, n_envs=25**, using `submit_phase111_eval_sweep.slurm <RUN_DIR> <BASE_CKPT> push-v3 100 1000`. Critically, evaluate **deterministically** (mean action, `flow_sde` sampling OFF / `select_action` path) — confirm the eval uses the same action mode as the 33% run.
- [ ] **Step 2:** Record results in `docs/superpowers/plans/results-ledger.md` as one table. Expected outcome: Flow-SDE checkpoints land ≈ the true 21% baseline ± noise (confirming "no signal," not "collapse").
- [ ] **Gate:** If any checkpoint is clearly >21% on 100 ep, Flow-SDE already has a weak signal and Phase 1 becomes higher priority. If all ≈21%, proceed to Phase 1 to isolate cause.

### Phase 1 — Clean A/B: Gaussian vs Flow-SDE at the *known-good* recipe (~6–10 GPU-hours)

**Why:** Remove the confound. Hold everything at the 33% recipe and flip only the objective.

**Shared hparams (match `evaluation.tex:284`):** `--group-size 8 --lr 1e-5 --clip-eps 0.2 --rollout-unit chunk --rollout-chunk-len 5 --action-transform no_tanh --num-updates 50 --save-every 2 --max-steps 120 --train-seed-base 2000`, dense env reward, `--euler-step-noise-std 0.0`.

- [ ] **Arm A (control):** `--logprob-mode gaussian`, `--init-log-std -2.0`. This should reproduce the ~30–33% peak. If it does **not**, the regression is in the *harness/rollout*, not Flow-SDE — stop and RCA the serial chunk rollout vs the old vector_async path.
- [ ] **Arm B (treatment):** `--logprob-mode flow_sde --flow-sde-noise-level 0.5 --flow-sde-trace-step -1`. Identical otherwise.
- [ ] **Eval both** with the Phase-0 100-ep gate (seeds 1000–1099), all saved checkpoints.
- [ ] **Gate / decision:**
  - A ≥ B by a clear margin → Flow-SDE is *worse* at this scale; deprioritise it, go to Phase 2 on the Gaussian path.
  - A ≈ B near baseline → **scale is the bottleneck, not the objective** → Phase 2.
  - B > A → Flow-SDE helps; go to Phase 3 to tune it.

> Note: restoring `lr=1e-5` is the single highest-value one-line change relative to the failed run.

### Phase 2 — Attack the real bottleneck: reward/advantage + scale (~1–2 GPU-days)

Run on whichever objective won Phase 1 (default: Gaussian, the safe path to ≥40%).

**2a. De-noise the advantage (highest leverage on *success*).** Pick ONE per sub-run:
- [ ] **Reward shaping toward success:** enable `sparse_success_delta` blended with dense (`reward_backends.py` already supports sparse; thesis `evaluation.tex:61` notes the optional sparse delta). Goal: advantages correlate with *success*, not dense-reward outliers.
- [ ] **Reduce outlier domination:** larger group (G16/G32) so a single high-dense trajectory does not own the gradient; optionally winsorize/clip returns before normalisation. (Code touch: `compute_group_advantages` — add optional return clipping behind a flag; TDD it.)

**2b. Scale the optimiser toward the thesis best (`evaluation.tex:306`: G32, lr 5e-6, clip 0.1, low rollout noise tied the best at 30%):**
- [ ] Multiple reset-seed groups per update (`batch_size>1`) to cut advantage variance — **this is the cleanest variance fix** and the current `1 group/update` is a key weakness.
- [ ] `--num-updates ≥ 50`, `--save-every 2`, **select best checkpoint on 100 ep**, report best + last-3 tail (thesis operational rule, `evaluation.tex:466`).
- [ ] Optional `--update-epochs 2` (then the clip actually engages and PPO becomes non-trivial).
- [ ] **Gate:** any config with a 100-ep checkpoint ≥ 35% → push that config further toward 40%.

### Phase 3 — Flow-SDE-specific tuning (only if Phase 1 Arm B ≥ Arm A)

- [ ] **Noise level sweep** `{0.1, 0.3, 0.5}` — 0.5 may over-explore (action clip ~5–7%, OOB ~0.03 is tolerable but not negligible).
- [ ] **Fixed vs random `trace_step`** — random-per-chunk adds variance; try a fixed late step (e.g. 8) and compare.
- [ ] **Joint log-prob (all transitions)** vs single-transition — lower variance, matches OpenPI `joint_logprob` mode; needs a trace over all denoise steps, summed.
- [ ] **Remove the dead `log_std`** from the Flow-SDE trainable set (A3) OR make exploration learnable (predict `noise_level`/`x_t_std` like OpenPI `flow_noise`).

### Cross-cutting: autonomy + caps (applies to all phases)

- [ ] Each run: `--export=NIL`, source `scripts/slurm/common_env.sh`, offline caches, `--fail-on-parity-violation` **only** for Flow-SDE arms, explicit `*_OK` marker, dependent `afterok` eval.
- [ ] Never exceed 3 concurrent GPU jobs. On any failure: full RCA (per `systematic-debugging`), fix, re-gate, continue — do not relax parity tolerance to paper over a real issue.
- [ ] Every claim cites a **100-ep** number. 25-ep is triage only.

---

## Risks / flags

- **≥40% may not be reachable at this scale on this checkpoint at all.** The thesis ceiling across *every* confirmed config is 30–33% @100ep. 40% likely needs either bigger batch/longer training than 2 GPUs allow comfortably, a better reward signal, or a stronger base checkpoint. Set expectations: the realistic near-term win is a *protocol-fair, reproducible >25%* that does not collapse.
- **Flow-SDE is a research bet, not the safe path.** Its paper wins are inseparable from H100-scale compute. Budget it as Phase 3, gated by a clean A/B.
- **Serial chunk rollout is slow** (one env, group 8 sequentially). Phase 2's `batch_size>1` and/or restoring `vector_async` will be needed to make ≥50-update sweeps affordable; verify the vector path matches the serial path's parity first.
- **Dense reward is fundamentally mis-aligned with success.** Without addressing 2a, more compute will keep buying dense-reward-chasing, not success.

## Open questions for the user

1. **Target realism:** keep the hard ≥40% @100ep target, or accept "reproducible, non-collapsing >25% on the 100-ep decision metric" as the near-term success criterion?
2. **Priority:** safe path (Gaussian chunk GRPO at thesis-best hparams + reward/scale fixes) vs research path (make Flow-SDE work)? Recommendation: **safe path first (Phases 0→1→2), Flow-SDE as Phase 3 only if the A/B justifies it.**
3. **Reward:** OK to introduce a success-correlated/sparse-blended reward (Phase 2a)? This is the most likely single change to move *success* rather than dense return.
