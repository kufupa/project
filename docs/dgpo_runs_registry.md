# DGPO SmolVLA push-v3 — run registry

> Updated: 2026-06-06  
> Task: MetaWorld push-v3 | loss: `nft_loss_form=dgpo` | eval: 100ep seeds 1000–1099 | sampler: `flow_ode` unless noted  
> Baselines: pretrained ~15–17% | Flow-SDE RL ~41%

**Correctness (2026-06-06):** runs **before** the grouping fix used broken positional groups after shuffle → treat as **proxy CFM**, not valid Direct-DGPO. **`v3_groupfix_baseline` (247454)** is the first valid Direct-DGPO run (true group ids + mean DSM energy).

---

## Live queue (2026-06-06 V4 u10 grid)

| Bundle | Job | Tags (3×10u+eval100 each) | State |
|---|---|---|---|
| b1 | 247473 | v4_baseline, v4_fix_microbatch, v4_fix_filtergw | RUN |
| b2 | 247474 | v4_fix_advclip, v4_fix_all3, v4_beta100 | PEND |
| b3 | 247475 | v4_ema_tau85, v4_no_filter, v4_champion | PEND |
| b4 | 247476 | v4_fix_all3_ema, v4_fix_all3_open, v4_beta100_ema | PEND |
| b5 | 247477 | v4_flowsde25, v4_flowsde50_champ, v4_giant_g32 | PEND |
| b6 | 247478 | v4_roll2_upd2, v4_kl_tight, v4_lr15x | PEND |
| b7 | pending | v4_tau80_chaos, v4_beta200, v4_sde10_soft | QoS queue |
| b8 | pending | v4_peak_lite, v4_fix_mb_ema, v4_fix_mb_open | QoS queue |

Plan: `project/docs/dgpo_v4_u10_moonshot_plan.md` | Submit: `submit_rlinf_dgpo_v4_u10_grid.sh`

---

## Sparse reward — what we run

| Layer | Setting | All DGPO? | Notes |
|---|---|---|---|
| Env reward | `reward_mode=sparse_success_delta` | **YES** | +1 on first success in ep, 0 else. No dense shaping. |
| Train reward filter | `filter_rewards=True` [0.1, 0.9] | **NO** | Default in yaml. Wave1 m1–m6,m8=ON. m7 + all v2=OFF. |
| Adv type | `adv_type=grpo` (except m4 raw) | mostly | m4 uses `adv_type=raw` → direct sparse success as signed adv |

**Short answer:** sparse env reward on **every** DGPO run. Reward **band filter** not universal — v2 all unfiltered; wave1 mostly filtered.

Config lock: `metaworld_pushv3_nft_smolvla.yaml` → `reward_mode: sparse_success_delta`, `filter_rewards: True`, `rewards_lower_bound: 0.1`, `rewards_upper_bound: 0.9`.

---

## Master results table

| Wave | Tag | Job | Status | Ep | β | τ | filter | grp | LR | noise | train wall | best@100ep | best u | u30 | Δ vs base |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| init | baseline | 247286→287 | DONE | 20 | 10 | 1.0 | ON | 16 | 5e-6 | ode | 32m+14m eval | **19%** | u15 | 18% | +4pp |
| init | beta50 | 247289→290 | DONE | 20 | 50 | 1.0 | ON | 16 | 5e-6 | ode | 30m+eval | **19%** | u15 | 19% | +4pp |
| w1 | m1_nuclear_beta | 247291 | DONE | 30 | 200 | 1.0 | ON | 16 | 5e-6 | ode | 46m | 19% | u24 | 17% | +4pp |
| w1 | m2_giant_group | 247428† | DONE | 30 | 15 | 1.0 | ON | 32 | 5e-6 | ode | 50m | 19% | u27 | 18% | +4pp |
| w1 | m3_ema_chase | 247293 | DONE | 30 | 20 | 0.85 | ON | 16 | 5e-6 | ode | 53m | **22%** | u15 | 17% | **+7pp** |
| w1 | m4_raw_signal | 247294 | DONE | 30 | 30 | 1.0 | ON | 16 | 5e-6 | ode | 46m | 17% | u0 | **6%** | collapse |
| w1 | m5_throughput | 247429 | RUN u24 | 30 | 15 | 1.0 | ON | 16 | 5e-6 | ode | ~3h est | — | — | — | — |
| w1 | m6_lr_missile | 247301 | DONE | 30 | 25 | 1.0 | ON | 16 | **5e-5** | ode | 47m | 16% | u3 | **0%** | collapse |
| w1 | m7_chaos_open | 247302 | DONE | 30 | 75 | 1.0 | **OFF** | 16 | 5e-6 | ode | 50m | **22%** | u21 | 17% | **+7pp** |
| w1 | m8_flowsde_hybrid | 247303 | DONE | 30 | 15 | 1.0 | ON | 16 | 5e-6 | sde@0.25 | 56m | 21% | u24 | 16% | +6pp |
| v2 | v2_champion_fusion | 247430 | DONE | 30 | 50 | 0.85 | OFF | 16 | 5e-6 | ode | 49m | 21% | u21 | 20% | +6pp |
| v2 | v2_ema_flowsde_full | 247431 | EVAL u18 | 30 | 20 | 0.85 | OFF | 16 | 5e-6 | sde@1.0 | 60m+ | 17%‡ | u6 | — | +2pp‡ |
| v2 | v2_flowsde_flow_lr | 247432 | train DONE | 30 | 25 | 0.9 | OFF | 16 | 7.5e-6 | sde@0.5 | 73m | — | — | — | — |
| v2 | v2_peak_hunter | 247433 | PEND | 30 | 40 | 0.85 | OFF | 16 | 5e-6 | ode | — | — | — | — | — |
| v2 | v2_chaos_tau80 | 247434 | PEND | 30 | 60 | **0.80** | OFF | 16 | 5e-6 | ode | — | — | — | — | — |
| v2 | v2_sde_soft_ema | 247435 | PEND | 30 | 10 | 0.95 | OFF | 16 | 5e-6 | sde@1.0 | — | — | — | — | — |
| **v3** | **v3_groupfix_baseline** | **247454** | **DONE** | 30 | 10 | 1.0 | ON | 16 | 5e-6 | ode | **~50m+30m eval** | **21%** | **u3** | **1%** | **+6pp** ✓ valid DGPO |

† m2 retries: 247292,247299,247319,247321,247422 (Ray/OOM) → success 247428  
‡ partial eval only (u0–u18)

**Leaderboard:** m3_ema_chase = m7_chaos_open **22%** | v2_champion_fusion 21% | m8 21% | init baseline 19%

---

## Per-checkpoint eval curves (100ep %)

### Init runs (save@5)

| u | base β10 | beta50 |
|---|---|---|
| 0 | 15 | 15 |
| 5 | 15 | 17 |
| 10 | 16 | 15 |
| 15 | **19** | **19** |
| 20 | 18 | 19 |

### Wave1 moonshots (save@3)

| u | m1 β200 | m2 g32 | m3 τ85 | m4 raw | m6 lr5e5 | m7 noF | m8 sde.25 |
|---|---|---|---|---|---|---|---|
| 0 | 13 | 16 | 14 | **17** | 15 | 17 | 16 |
| 3 | 14 | 16 | 15 | 13 | **16** | 17 | 16 |
| 6 | 13 | 14 | 16 | 17 | **0** | 18 | 18 |
| 9 | 15 | 15 | 16 | 11 | 0 | 18 | 16 |
| 12 | 16 | 14 | 19 | 16 | 0 | 20 | 15 |
| 15 | 15 | 16 | **22** | 17 | 0 | 19 | 19 |
| 18 | 17 | 16 | **22** | 14 | 0 | 21 | 17 |
| 21 | 17 | 16 | **22** | 10 | 0 | **22** | 16 |
| 24 | **19** | 16 | 19 | 6 | 0 | 20 | **21** |
| 27 | 17 | **19** | 21 | 7 | 0 | 21 | 15 |
| 30 | 17 | 18 | 17 | 6 | 0 | 17 | 16 |

### V2 moonshots (save@3)

| u | v2_champion | v2_ema_sde1.0‡ | v2_flow_lr‡ |
|---|---|---|---|
| 0 | 16 | 13 | — |
| 3 | 14 | 16 | — |
| 6 | 17 | **17** | — |
| 9 | 17 | 15 | — |
| 12 | 20 | 15 | — |
| 15 | 19 | 17 | — |
| 18 | 20 | 17 | — |
| 21 | **21** | — | — |
| 24 | 20 | — | — |
| 27 | 17 | — | — |
| 30 | 20 | — | — |

### V3 groupfix — first valid Direct-DGPO (save@3, job 247454)

| u | v3_groupfix (valid) |
|---|---|
| 0/base | 15 |
| 3 | **21** |
| 6 | 15 |
| 9 | 16 |
| 12 | 15 |
| 15 | 12 |
| 18 | 10 |
| 21 | 10 |
| 24 | 6 |
| 27 | 6 |
| 30 | 1 |

Peak u3 **21%** (+6pp vs baseline); late-training collapse to u30 **1%**. Same yaml defaults as pre-fix runs but with correct grouping — best comparable to broken-run leaderboard (~19–22%) not Flow-SDE 41%.

---

## Hyperparam detail (moonshots)

| Tag | Key overrides beyond COMMON |
|---|---|
| COMMON | 32env, mbs16, gbs32, lr5e-6, τ1.0, grp16, 120 train steps, filter ON |
| m1 | dpo_beta=200 |
| m2 | group_size=32, mbs32, dpo_beta=15, offload+GC |
| m3 | nft_tau=0.85, dpo_beta=20 |
| m4 | adv_type=raw, normalize_advantages=False, dpo_beta=30 |
| m5 | rollout_epoch=2, update_epoch=4, dpo_beta=15, offload+GC |
| m6 | dpo_beta=25, lr=5e-5, value_lr=5e-4 |
| m7 | filter_rewards=False, adv_clip_max=1.5, dpo_beta=75 |
| m8 | dpo_beta=15, noise_method=flow_sde, noise_level=0.25 |
| v2_champion | τ0.85, no filter, beta50, adv_clip1.5 |
| v2_ema_flowsde | τ0.85, no filter, beta20, flow_sde@1.0 |
| v2_flowsde_lr | τ0.9, no filter, beta25, upd×2, lr7.5e-6, flow_sde@0.5 |
| v2_peak_hunter | τ0.85, no filter, beta40, adv_clip1.2, roll×2 upd×2 |
| v2_chaos_tau80 | τ0.80, no filter, beta60, adv_clip2.0 |
| v2_sde_soft_ema | τ0.95, no filter, beta10, flow_sde@1.0 |
| v3_groupfix_baseline | yaml defaults only — valid group ids + mean DSM (no extra overrides) |

---

## Failures / retries

| Tag | Job(s) | RCA | Fix |
|---|---|---|---|
| smoke | 247284 | mbs8 ≠ group16 | mbs16 |
| m2 | 247292 | Ray collision hopper | exclude node |
| m5–m8 | 247295–298 | QOSMaxMemoryPerUser 100G | 50G RAM |
| m2,m5 | 247299,320,321 | CUDA OOM | offload+GC, cluster.py list_actors try/except |
| m2 | 247422 | OOM group32 | same + resubmit 247428 OK |

---

## Patterns (evidence)

| Works | Fails |
|---|---|
| EMA ref τ0.85 (m3 22%) | lr 10× (m6 → 0% by u6) |
| no filter (m7 22%) | raw adv (m4 → 6% @ u30) |
| peak u15–u24, regress u30 | nuclear β200 no gain vs β10 |
| moderate β15–75 | Flow-SDE alone not magic (m8 21%, v2_ema 17% partial) |

Flow-SDE noise → rollout exploration only; DGPO loss still velocity-space group pref on collected rollouts.

---

## Artifact paths

| Type | Path |
|---|---|
| Config | `examples/embodiment/config/metaworld_pushv3_nft_smolvla.yaml` |
| Submit | `scripts/slurm/submit_rlinf_dgpo_moonshot_tags.sh` |
| Results | `logs/results/rlinf_nft_dgpo_ms_<tag>_<jobid>/eval100/sweep/results.jsonl` |
| Slurm out | `logs/slurm/dgpo-ms-<tag>_<jobid>.out` |
