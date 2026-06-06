# DGPO + SmolVLA + MetaWorld push-v3 — supervisor brief

**Date:** 2026-06-05  
**Status:** First clean end-to-end run complete (RLinf-native NFT path, no new trainer).

## One-liner

RLinf already had velocity-space preference RL (DiffusionNFT / `EmbodiedNFTFSDPPolicy`). We added **Direct Group Preference Optimization** (`nft_loss_form=dgpo`, arXiv 2510.08425 vendor path) and **SmolVLA NFT forward** so push-v3 training runs on the same stack as openpi+LIBERO — **no MDP, no policy gradient**, ODE-only rollouts, frozen per-update reference (`nft_tau=1.0`).

**2026-06-06 correctness fix:** prior runs before the grouping fix used shuffled positional micro-batch groups (not true GRPO groups) and sum-not-mean DSM energy. Treat pre-fix eval numbers as **proxy CFM runs**, not valid Direct-DGPO. Re-run required after fix lands.

## Why this track (vs Flow-SDE / FPO / per-sample NFT-DPO)

| | Flow-SDE GRPO | FPO | NFT-DPO (exists) | **DGPO (this work)** |
|---|---|---|---|---|
| Formalism | nested MDP + SDE | env MDP + CFM ratio | per-sample velocity pref | **group preference, no PG** |
| Sampling | SDE | ODE/SDE | ODE | **ODE** |
| Reference | old logprob | none | frozen/EMA | frozen/EMA |
| Answers "flow ≠ MDP" | weak | strong | strong | **strongest** |
| WM-rollout fit | poor (SDE off-manifold) | medium | good | **best (ODE, few steps)** |

> **Note:** Two unrelated papers use "DGPO". This branch implements **Direct Group Preference Optimization** (2510.08425 / vendor `project/vendor/DGPO`). **Distribution-Guided DGPO** (2605.03327) is a separate GRPO advantage-redistribution method — see `project/docs/superpowers/plans/2026-06-05-distribution-guided-dgpo-smolvla.md`.

Flow-SDE remains the **performance baseline** (41% push-v3). DGPO is the **narrative + WM-cost** bet.

> **Two "DGPO" papers:** this branch = **Direct Group Preference Optimization** (2510.08425 / vendor). **Distribution-Guided DGPO** (2605.03327) is separate — see `project/docs/superpowers/plans/2026-06-05-distribution-guided-dgpo-smolvla.md`.

## What we implemented (RLinf repo)

1. **`smolvla_action_model.py`:** `ForwardType.NFT`, `nft_forward` (prefix cache → `denoise_step` → `v_theta`), `flow_ode` rollout emits `nft_x0` + `nft_noise_level=0`.
2. **`fsdp_nft_policy_worker.py`:** `nft_loss_form=dgpo` — group-summed sigmoid over advantage-weighted CFM energy vs frozen ref (mirrors vendor DGPO).
3. **Config:** `metaworld_pushv3_nft_smolvla.yaml` — `embodied_nft`, `grpo`, G=16, `sparse_success_delta`, `filter_rewards`.
4. **Slurm:** smoke → 20-epoch train → 100ep eval (seeds 1000–1099, 25 envs, chunk 5, 150 steps).

**Gotcha fixed:** `actor.micro_batch_size` must be a multiple of `algorithm.group_size` (16) or DGPO group reshape fails.

## Eval numbers (100 episodes, push-v3)

| Checkpoint | Success rate |
|---|---|
| Pretrained baseline | 15.0% |
| DGPO @ update 5 | 15.0% |
| DGPO @ update 10 | 16.0% |
| **DGPO @ update 15** | **19.0% (best)** |
| DGPO @ update 20 | 18.0% |

**References:** pretrained HF ~14–17%; Flow-SDE GRPO **41%** (separate run, same task/reward family).

**Interpretation:** 20 global updates, `dpo_beta=10`, `nft_tau=1.0` → **stable** (finite loss, no collapse) and **+4pp** over baseline at best ckpt — not competitive with Flow-SDE yet. Training rollouts showed sparse group signal (`filter_rewards` drops all-fail/all-success groups); expected for push-v3.

**Human flag:** `nft_tau=1.0` was stable this run — **no EMA (0.995) swap required**. Next knobs: longer train, `dpo_beta` sweep (50/100), more envs.

## Artifacts

- Train ckpts: `RLinf-smolvla-metaworld-ppo-grpo/logs/results/rlinf_nft_dgpo_train_247286/`
- Eval sweep: `.../rlinf_nft_dgpo_eval100_247287/sweep/results.jsonl`
- Overnight log: `project/docs/dgpo_overnight_log.md`

## Next (plan Phase 5–6)

1. **JEPA-WM bridge:** swap real env for `wm_latent_progress` + oracle roots (`metaworld_smolvla_wm_pushv3.yaml`); DGPO ODE + resample should cut WM cost vs Flow-SDE.
2. **Hparam sweep:** `dpo_beta=50` train (queued autonomously if overnight continues).
3. Upstream PR to RLinf when numbers justify.
