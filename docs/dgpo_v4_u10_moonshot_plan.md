# DGPO V4 — 10-update moonshot grid (2026-06-06)

> **Goal:** Fast screen for peak-before-collapse (v3 peaked u3 @ 21%).  
> **Shape:** 24 configs, 8 GPU jobs × 3 sequential train+eval100 each.  
> **Train:** 10 updates, `save_interval=1`, valid Direct-DGPO (group-id fix).  
> **Eval:** 100ep seeds 1000–1099 on baseline + u1…u10.

---

## Job packing (QoS cap = 8 GPUs)

| Bundle | Job tags (sequential in one GPU) |
|---|---|
| b1 | v4_baseline, v4_fix_microbatch, v4_fix_filtergw |
| b2 | v4_fix_advclip, v4_fix_all3, v4_beta100 |
| b3 | v4_ema_tau85, v4_no_filter, v4_champion |
| b4 | v4_fix_all3_ema, v4_fix_all3_open, v4_beta100_ema |
| b5 | v4_flowsde25, v4_flowsde50_champ, v4_giant_g32 |
| b6 | v4_roll2_upd2, v4_kl_tight, v4_lr15x |
| b7 | v4_tau80_chaos, v4_beta200, v4_sde10_soft |
| b8 | v4_peak_lite, v4_fix_mb_ema, v4_fix_mb_open |

Submit: `bash scripts/slurm/submit_rlinf_dgpo_v4_u10_grid.sh`

---

## 24 moonshot specs

### Code-fix axis (new flags)

| Tag | Fix | Hydra |
|---|---|---|
| v4_fix_microbatch | Recompute DGPO weights per micro-batch | `dgpo_weight_precompute=False` |
| v4_fix_filtergw | Mask filtered envs in group-weight sum | `dgpo_apply_loss_mask_to_group_weights=True` |
| v4_fix_advclip | Clip signed GRPO adv before DGPO | `dgpo_clip_signed_adv=True` |
| v4_fix_all3 | All three fixes | all above |

### Full table

| # | Tag | Intent | Key overrides |
|---|---|---|---|
| 1 | v4_baseline | Post-fix yaml defaults | — |
| 2 | v4_fix_microbatch | Align weight/loss forward | `dgpo_weight_precompute=False` |
| 3 | v4_fix_filtergw | Filter-aware group pref | `dgpo_apply_loss_mask_to_group_weights=True` |
| 4 | v4_fix_advclip | Stabilize sparse adv outliers | `dgpo_clip_signed_adv=True` |
| 5 | v4_fix_all3 | Combined code fixes | all 3 flags |
| 6 | v4_beta100 | Paper β=100 | `dpo_beta=100` |
| 7 | v4_ema_tau85 | Wave1 winner m3 | `nft_tau=0.85`, `dpo_beta=20` |
| 8 | v4_no_filter | Wave1 winner m7 | `filter_rewards=False`, `dpo_beta=75` |
| 9 | v4_champion | v2 fusion | `τ0.85`, no filter, `β50` |
| 10 | v4_fix_all3_ema | Fixes + EMA | all3 + `τ0.85`, `β50` |
| 11 | v4_fix_all3_open | Fixes + open filter | all3 + no filter, `β75` |
| 12 | v4_beta100_ema | High β + EMA | `β100`, `τ0.85` |
| 13 | v4_flowsde25 | m8 hybrid | flow_sde@0.25, `β15` |
| 14 | v4_flowsde50_champ | SDE + champion | sde@0.5 + champion settings |
| 15 | v4_giant_g32 | m2 large groups | `group_size=32`, offload |
| 16 | v4_roll2_upd2 | Moderate throughput | `rollout×2`, `update×2` |
| 17 | v4_kl_tight | Drift control | `nft_beta=2`, `max_drift=0.15` |
| 18 | v4_lr15x | Gentle LR bump | `lr=7.5e-6`, no filter |
| 19 | v4_tau80_chaos | Aggressive EMA | `τ0.80`, no filter, `β60` |
| 20 | v4_beta200 | Nuclear β control | `dpo_beta=200` |
| 21 | v4_sde10_soft | Soft EMA + full SDE | `τ0.95`, sde@1.0 |
| 22 | v4_peak_lite | v2_peak_hunter lite | champion + roll×2 upd×2 |
| 23 | v4_fix_mb_ema | Microbatch fix only + EMA | `FIX_MB` + `τ0.85` |
| 24 | v4_fix_mb_open | Microbatch fix + open | `FIX_MB` + no filter |

COMMON: 32env, mbs16, gbs32, lr5e-6, grp16, sparse_success_delta, filter ON (unless overridden).

---

## Success criteria (10u screen)

| Metric | Pass |
|---|---|
| Best@100ep | ≥ 21% (beat v3 u3) |
| Peak update | ideally u3–u7 (not u1-only noise) |
| u10 vs peak | no >10pp collapse (v3 dropped 15pp u3→u30) |

**Resume:** `runner.resume_dir=<log_root>/<exp>/checkpoints/global_step_10/actor` + `max_epochs=30`.

---

## Artifacts

| Type | Path |
|---|---|
| Submit | `scripts/slurm/submit_rlinf_dgpo_v4_u10_grid.sh` |
| Single run | `scripts/slurm/smolvla_rlinf_nft_dgpo_moonshot_u10_train_eval100_a30.slurm` |
| Triple pack | `scripts/slurm/smolvla_rlinf_nft_dgpo_triple_u10_a30.slurm` |
| Results | `logs/results/rlinf_nft_dgpo_ms_<tag>_<jobid>/eval100/sweep/results.jsonl` |
