# DGPO SmolVLA + MetaWorld — overnight execution log

Autonomous pipeline: implement → unit test → GPU smoke → train → 100ep eval.

## Distribution-Guided DGPO Phase A (arXiv 2605.03327) — 2026-06-06

**Correct DGPO** — token-level credit redistribution via Hellinger deviation (NOT Direct-DGPO 2510.08425).

### Code landed (inline on cluster3, commit-per-task)

| Task | Commit | Status |
|------|--------|--------|
| A1 dgpo.py math | `feat(dgpo): add distribution-guided redistribution math` | 8/8 unit tests pass |
| A2 frozen SFT ref | `feat(dgpo): load frozen SFT reference policy for deviation` | done |
| A3 chunk weights | `feat(dgpo): per-valid-chunk weight helper` | done |
| A4 loss wiring | `feat(dgpo): redistribute per-trajectory advantage across chunks` | done |
| A5 CLI+telemetry | `feat(dgpo): CLI flags and weight telemetry` | done |
| A6 slurm | `feat(dgpo): a30 smoke + train+eval slurm` | queued |

### GPU jobs (Distribution-Guided)

| Job | Script | Status |
|-----|--------|--------|
| **247457** | `submit_dgpo_chunk_grpo_smoke_a30.slurm` | **FAILED** — see RCA below |
| **247459** | smoke retry (`dense_return`) | **CORE GREEN** — DGPO weights ok, train ok; slurm ckpt path check failed (fixed) |

### Smoke 247457 RCA (2026-06-06)

- **Symptom:** exit 20, `expected update_0002 checkpoint not found`
- **Root cause:** both updates `skipped=zero_advantages` — `sparse_success_delta` + 0% success → all G=16 returns identical (0) → GRPO adv=0 → no optimize, no ckpt, no `[dgpo]` weight telemetry
- **Fix:** smoke slurm → `--reward-mode dense_return` (train arms keep `sparse_success_delta`)

### Smoke 247459 RCA #2

- **Symptom:** exit 20 after successful train (`update_0002.pt` exists)
- **Root cause:** slurm guard looked for `update_0002/trainable_model/model.safetensors`; trainer writes `update_0002.pt`
- **Fix:** guard → `update_XXXX.pt` in smoke + train slurms
- **DGPO validated:** `w_std=0.045` update 1, finite loss, parity=1.0

### A7 experiment queue (2026-06-06)

| Arm | Job | Config |
|-----|-----|--------|
| E0 control (flow-sde GRPO) | **247467** | no `--dgpo`, sparse_success_delta |
| E1 DGPO primary | **247468** | tau=0.5 kappa=0 frozen_sft |
| E2 tau flat | pending | DGPO_TAU=1.0 |
| E3 tau sharp | pending | DGPO_TAU=0.25 |

## Session start

- **2026-06-05T00:00:00Z** Plan loaded. RLinf repo: `/vol/bitbucket/aa6622/RLinf-smolvla-metaworld-ppo-grpo`.
- Locked: `nft_loss_form=dgpo`, `group_size=16`, `nft_tau=1.0`, push-v3, `sparse_success_delta`.

## DGPO grouping fix (2026-06-06)

- **Review confirmed:** shuffle + positional `reshape(-1, group_size)` broke true GRPO groups; DSM energy used sum not mean.
- **Fix landed:** `rlinf/workers/actor/dgpo_group_utils.py` + `fsdp_nft_policy_worker.py`
  - `dgpo_group_id` from rollout batch `(chunk, env_seed_block)`
  - group-block shuffle (not row shuffle)
  - global-batch precompute of detached group weights via `scatter_add`
  - DSM energy → mean (vendor-aligned)
  - `world_size>1` hard-fail until distributed group all_reduce
- **Tests:** `tests/unit_tests/test_smolvla_nft_dgpo.py` — 10/10 pass
- **Prior moonshot/v1/v2 evals:** invalid Direct-DGPO — proxy CFM only. Re-run required.
- **Cancelled:** pending/running pre-fix moonshots (247429, 247433–247435)
- **2026-06-06 validation queued:**
  - `247452` smoke (groupfix code, 50G)
  - `247454` v3_groupfix_baseline train30+eval100 (afterok:247452, 50G) — yaml defaults, mean DSM + true group ids
- **2026-06-06 overnight:** `dgpo_v3_groupfix_overnight_loop.sh` nohup — **60s** wake, RCA/resubmit/eval-only, 50G RAM
- **2026-06-06 00:35Z smoke `247452` PASSED** — `RLINF_NFT_DGPO_SMOKE_OK`, dgpo_group_weight metrics finite
- **2026-06-06 train `247454` RUNNING** — v3_groupfix_baseline 30 updates + eval100 on hopper

- SmolVLA: `ForwardType.NFT`, `nft_forward`, `flow_ode` rollout + `nft_x0`/`nft_noise_level` emit.
- Worker: `nft_loss_form=dgpo` group-level branch in `fsdp_nft_policy_worker.py`.
- Config: `metaworld_pushv3_nft_smolvla.yaml`.
- Slurm: `smolvla_rlinf_nft_dgpo_{smoke,train,eval100}_a30.slurm`.

## GPU jobs

- **2026-06-05** smoke `nft-dgpo-smoke` job_id=**247284** — FAILED
  - **RCA:** `micro_batch_size=8` with `group_size=16` → `reshape(-1,16)` on 8 samples
  - **Fix:** `micro_batch_size=16` + guard in `_compute_dgpo_nft_loss`
- **2026-06-05** smoke retry job_id=**247285** — **GREEN** (`runner.run:done`, `dgpo_group_weight_mean=0.5`, finite loss)
- **2026-06-05** train job_id=**247286** — **GREEN** (~35min, `runner.run:done`, ckpts at steps 5/10/15/20)
- **2026-06-05** eval100 job_id=**247287** — **GREEN** (~14min, seeds 1000-1099)

## Eval results (100ep, push-v3, chunk=5, max_steps=150)

| Checkpoint | Success rate | vs baseline |
|---|---|---|
| baseline (pretrained) | **15.0%** | — |
| DGPO update_5 | 15.0% | flat |
| DGPO update_10 | 16.0% | +1pp |
| DGPO update_15 | **19.0%** | **+4pp (best)** |
| DGPO update_20 | 18.0% | +3pp |

Sweep: `/vol/bitbucket/aa6622/RLinf-smolvla-metaworld-ppo-grpo/logs/results/rlinf_nft_dgpo_eval100_247287/sweep/results.jsonl`

Refs: pretrained ~17%, Flow-SDE baseline 41%. Short 20-epoch DGPO run → modest lift, stable (no NaN/collapse). `nft_tau=1.0` OK this run — no EMA swap needed yet.

## Commits

- `feat(nft): add SmolVLA NFT forward and DGPO loss for push-v3`
- `fix(nft): align DGPO micro_batch with group_size`
- `chore(slurm): add dpo_beta=50 DGPO train and eval100 jobs`

## Follow-up (autonomous, user asleep)

- **Supervisor brief:** `project/docs/dgpo_smolvla_metaworld_supervisor_brief.md`
- **dpo_beta=50 sweep:** train **247289** → eval **247290** (chained, same 20ep protocol)
- **WM bridge:** deferred — needs config fork from `metaworld_pushv3_wm_flowsde_tf_smolvla.yaml` + NFT env reward alignment; logged for Phase 5

## Moonshot grid (2026-06-05) — 8 parallel @ 30ep + 100ep eval

Protocol: 30 updates, save@3, eval ckpts 3/6/9/12/15/18/21/24/27/30 (100ep, seeds 1000-1099), train+eval in same job.

| Tag | Job | Hypothesis |
|---|---|---|
| m1_nuclear_beta | 247291 | dpo_beta=200 — extreme preference sharpening |
| m2_giant_group | 247292 | group_size=32 — bigger group comparisons |
| m3_ema_chase | 247293 | nft_tau=0.85 — moving ref instead of frozen |
| m4_raw_signal | 247294 | adv_type=raw — direct success signal, no GRPO clip |
| m5_throughput | 247295 | rollout×2, update×4, 64 envs — exploit fast DGPO |
| m6_lr_missile | 247296 | lr=5e-5 — 10× policy LR |
| m7_chaos_open | 247297 | filter_rewards=False, beta=75 — full spectrum |
| m8_flowsde_hybrid | 247298 | flow_sde noise=0.25 — SDE rollouts + DGPO |

Submit: `scripts/slurm/submit_rlinf_dgpo_moonshot_grid.sh`
Results: `logs/results/rlinf_nft_dgpo_ms_<tag>_<jobid>/eval100/sweep/results.jsonl`

## Moonshot monitor (autonomous overnight)

- **2026-06-05** M2 job 247292 FAILED — Ray collision on hopper (M1+M4 co-located). RCA: multiple Ray instances same node.
- **2026-06-05** Pending M5–M8 (247295–298) **scancelled** — `QOSMaxMemoryPerUser` at 100G.
- **2026-06-05** Resubmitted @ **50G RAM**: M2=247299 (exclude hopper), M5=247300, M6=247301, M7=247302, M8=247303.
- Monitor: `scripts/slurm/dgpo_moonshot_overnight_loop.sh` (15m tick, auto-resubmit/eval-only).
- Eval-only script: `smolvla_rlinf_nft_dgpo_moonshot_eval100_only_a30.slurm`

## M2/M5 failure RCA + fix (2026-06-05)

**Symptom:** `ServerUnavailable: http://127.0.0.1:8265/api/v0/actors` — misleading.

**Root cause (from slurm logs):**
- **M2** (247299/247321): `torch.OutOfMemoryError` in NFT forward with `micro_batch_size=32`, `group_size=32` on A30 (23.6GB). Actor+rollout+env colocated.
- **M5** (247320): `torch.OutOfMemoryError` with `64 envs`, `micro_batch=32`, `update_epoch=4`.

**Secondary bug:** `cluster.py` SIGUSR1 handler called `list_actors()` which needs Ray dashboard API; jobs use `include_dashboard=false` → cleanup masked real OOM as ServerUnavailable.

**Fixes applied:**
1. `cluster.py`: try/except around `list_actors()` in signal handler.
2. **M2**: keep group32/mbs32 + `gradient_checkpointing=True`, `actor.enable_offload=True`, `rollout.enable_offload=True`.
3. **M5**: throughput via `rollout_epoch=2, update_epoch=4` at **32env/mbs16** (not 64/32) + same offload/GC.
4. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in moonshot slurm.
5. Resubmitted: **M2=247422**, **M5=247423** @ 50G.
- **2026-06-05T06:34:00Z** resubmit m2_giant_group job_id=submitted m2_giant_group job=247319 mem=50G exclude=none
RLINF_DGPO_MOONSHOT_SUBMIT_OK m2_giant_group=247319 rca=unknown mem=50G exclude=none
- **2026-06-05T06:34:00Z** resubmit m5_throughput job_id=submitted m5_throughput job=247320 mem=50G exclude=none
RLINF_DGPO_MOONSHOT_SUBMIT_OK m5_throughput=247320 rca=unknown mem=50G exclude=none
- **2026-06-05T07:49:02Z** resubmit m2_giant_group job_id=submitted m2_giant_group job=247321 mem=50G exclude=none
RLINF_DGPO_MOONSHOT_SUBMIT_OK m2_giant_group=247321 rca=unknown mem=50G exclude=none
