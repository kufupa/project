# GPU handoff — SmolVLA MetaWorld (done only, 9d)

**window:** 2026-05-19 → 2026-05-28 UTC · **queue:** PBS `v1_gpu72` · **GPU:** `RTX6000`  
**base ckpt:** HF `jadechoghari/smolvla_metaworld` · **task:** `push-v3`  
**train:** GRPO (`train_phase11_env_on_policy_grpo.py`) · Direct PPO (`run_smolvla_metaworld_direct_ppo.py`)  
**paths:** GRPO `.../ephemeral/smolvla_metaworld/` · Direct `.../ephemeral/rlinf_ray_runs/<jobid>/results/`

**collapse rule:** `last` ≥8pp below `best` → list **last 3 evaluated ckpts** under row.

---

## eval proto

| | GRPO | Direct |
|---|------|--------|
| ep | 25 (100ep holdout separate) | 25 |
| ckpt stride | 2 | 2 / every-other |
| n_env | 25 vector | 25 |
| seed | 1000+ | 50000+ |
| chunk / steps | 5 / 120 | 5 / 120 |
| metric | `pc_success` % | `success_rate` % |
| baseline | ~16–28% @u≤2 | ~12–16% @u0 |

---

## A — GRPO G16 lineage · *same hparams all segments*

**hparams:** `g16` · `chunk5` · `lr5e-6` · `clip0.2` · `save_every2` · `vector_async` · `official_lerobot`  
**run_dir (ephemeral):** `.../phase11_pushv3_chunk5_g16_lr5e6_clip02_u70` (continued from artifacts `..._u20`)

| segment | train | eval sweep | bl@25ep | best@25ep | last@25ep | collapse tail |
|---------|-------|------------|---------|-----------|-----------|---------------|
| fresh u0→20 | artifacts PBS | `u2–10`, `u12–20` | 20%@u2 | **44%@u10,u20** | **44%@u20** | — |
| resume u20→70 | resume PBS | `0022_0070` | 28%@u22 | **44%@u50,u70** | **44%@u70** | — |
| resume u70→170 | **2869628** ✓ | **2869761** ✓ | 40%@u72 | **52%@u74** | **20%@u170** | u166 **32%** · u168 **12%** · u170 **20%** |

**u70→170 mid-run (not collapse):** u158 40% · u160 32% · peak u74 52% · late u150–164 mostly 24–40%.

**read:** only strong ↑ chain; **never take last ckpt** on 70→170 segment.

---

## B — GRPO moonshot seedbatch · *fresh · lr diff only*

**hparams:** `b4×g16` · **`lr1.25e-6`** · `clip0.2` · `50u` · else = G16  
**run_dir:** `.../phase11_seedbatch_b4_g16_lr1p25e6_clip02_u50_2861115.pbs-7`

| PBS | train | bl | best | last | collapse tail |
|-----|-------|-----|------|------|---------------|
| **2861115/2861116** ✓ | u50 | 24%@u2 | 32%@u10,u50 | 32%@u50 | u42 **20%** · u46 **32%** · u50 **32%** (mild dip ok) |

---

## C — Direct stage3b sparse chain · *same hparams · single run_dir*

**hparams:** `4env` · `sparse_success_delta` · `rel_reward` · **`lr3e-7`** · `value_lr1e-4` · `chunk5` · `epochs2`  
**run_dir:** `.../2861110/.../smolvla_direct_stage3b_sparse_resume100_150_2861110.pbs-7`

| segment | date | train | eval | bl | best | last | collapse tail |
|---------|------|-------|------|-----|------|------|---------------|
| stage3 fresh | 05-24 | **2836701** | eval | ~12%@u0 | 32%@u50 | 32%@u50 | — |
| stage2 sparse | 05-24 | **2835191** | eval | 16%@u5 | **36%@u25** | 36%@u25 | — |
| stage2 sparse resume50 | 05-25 | **2839350** | eval | 16%@u0 | **36%@u25** | 28%@u75 | u0 16% · u25 **36%** · u75 **28%** |
| s3b resume50→100 | 05-25 | **2839351** | eval | 16%@u0 | **36%@u60** | 32%@u100 | u90 **28%** · u100 **32%** (mild) |
| s3b resume100→150 | 05-27 | **2861110** | — | — | — | u150 train | — |
| s3b resume110→150 | 05-27 | **2864660** ✓ | — | — | — | u150 | — |
| eval 100–150 | 05-28 | — | **2869622** ✓ | 16%@u0 | **40%@u120** | 36%@u140 | — |
| resume150→250 | 05-28 | **2869630** ✓ | **2869631** ✓ | 16%@u0 | **40%@u120** | **28%@u240** | u200 **32%** · u220 **24%** · u240 **28%** |

**read:** **40%@u120** stable best; **u240 collapse** vs u120 — same pattern as G16 late train.

---

## D — Direct moonshot 8env · *fresh · OOM fix*

**hparams:** `8env` · `lr5e-7` · `sparse` · `50u` · `save_every5` · **omit:** 16/12env OOM  
**run_dir:** `.../2869619/.../smolvla_direct_moonshot_8env_lr5e7_2869619.pbs-7`

| PBS | train | best | last | collapse tail |
|-----|-------|------|------|---------------|
| **2869619 / 2870105** ✓ | u50 | **36%@u40** | 28%@u50 | u30 **32%** · u40 **36%** · u50 **28%** |

---

## E — Direct 8env sparse tune · *train done · eval incomplete*

**hparams:** `8env` · `lr3e-7` · `sparse` · `100u`  
| **2869634** ✓ train u100 | eval **2872532** pending | no 25ep yet |

---

## F — GRPO reward-chain experiments (2026-05-19) · *each row = diff hparams*

Dir: `.../phase11_grpo_20260519_reward_chains/`

| run | hparams note | train u | bl | best@25ep | last@25ep | collapse tail |
|-----|--------------|---------|-----|-----------|-----------|---------------|
| `p11b4g16_ln_succ` | succ reward, ln | 30 | 28%@u2 | **40%@u12** | 20%@u30 | u26 **32%** · u28 **32%** · u30 **20%** |
| `p11b4g16_ln_succ_clip` | +clip variant | 10 | 20%@u2 | 28%@u4 | 16%@u10 | u6 **20%** · u8 **20%** · u10 **16%** |
| `p11b4g16_ln_dense` | dense reward | 70 | 16%@u2 | **36%@u12** | **16%@u70** | u66 **24%** · u68 **28%** · u70 **16%** |

---

## G — GRPO stability half-step (2026-05-20)

Dir: `.../phase11_grpo_20260520_stability/`

| run | note | train u | bl | best | last | collapse tail |
|-----|------|---------|-----|------|------|---------------|
| `p11b4g16_ln_halfstep` | half-step LN | 40 | 28%@u2 | **40%@u20** | 24%@u40 | u36 **32%** · u38 **36%** · u40 **24%** |

---

## H — GRPO `pushv3_*` grid (2026-05-23–25) · *fresh configs · stitched sweeps*

Dir: `.../smolvla_metaworld/checkpoints/<name>/`

| config | hparams (parse name) | u_max | bl | best | last | collapse tail |
|--------|----------------------|-------|-----|------|------|---------------|
| `b4g4_dense_lr5e6_d40` | b4g4 dense lr5e-6 | 70 | 24%@u2 | **40%@u20** | 28%@u70 | u66 **28%** · u68 **20%** · u70 **28%** |
| `b4g4_lr25e6_clip08` | lr2.5e-6 clip0.8 | 70 | 16%@u2 | **44%@u40** | 32%@u70 | u66 **28%** · u68 **32%** · u70 **32%** |
| `b4g8_dense_lr3e6_d40` | b4g8 dense lr3e-6 | 70 | 32%@u2 | **40%@u20** | **16%@u70** | u66 **16%** · u68 **16%** · u70 **16%** |
| `b4g4_dense_lr1e6_clip005` | lr1e-6 clip0.05 | 40 | 16%@u2 | **40%@u26** | 32%@u40 | — |
| `b4g4_dense_lr5e6` | short 10u run | 10 | 20%@u2 | 40%@u6 | 32%@u10 | u6 **40%** · u8 **36%** · u10 **32%** |
| `b4g4_sbonus500_lr5e6_clip01` | sbonus | 40 | 20%@u2 | 36%@u28 | 28%@u40 | — |
| `b8g4_sbonus500_lr5e6_clip01` | b8g4 sbonus | 40 | 20%@u2 | 32%@u34 | 28%@u40 | u36 **16%** · u38 **12%** · u40 **28%** |
| `b4g16_sbonus500_seedwave32` | b4g16 seedwave | 30 | 16%@u2 | 32%@u10 | 28%@u30 | — |
| `b4g16_..._sf1` | fork sf1 | 29 | — | 32%@u27 | **16%@u29** | u25 **28%** · u27 **32%** · u29 **16%** |
| `b8g8_sbonus500_seedwave32` | b8g8 | 30 | 12%@u2 | 32%@u4 | **8%@u30** | u26 **16%** · u28 **8%** · u30 **8%** |
| `b8g8_..._sf1` | b8g8 fork | 29 | 24%@u1 | 32%@u5 | 24%@u29 | — |

**pattern:** noisy **20–44%** band · **last ≪ best** common on long trains.

---

## I — Direct early / diag (2026-05-24) · *completed but weak*

| job dir | purpose | best@25ep | last@25ep | collapse tail |
|---------|---------|-----------|-----------|---------------|
| `2835766` stage3 dense | early s3 | 24%@u50 | 24%@u50 | — |
| `2835192` stage2 dense | dense reward | 28%@u10 | 24%@u25 | — |
| `2834108` fused diag | learn vs zero | ~16% | ~16% | flat |
| `2833854` eval stable | long eval sweep | 12%@u1 | **0%@u250** | u200 **0%** · u225 **0%** · u250 **0%** |
| `2833855` throughput | throughput test | 16%@u1 | **0%@u75** | u25 **0%** · u50 **0%** · u75 **0%** |

---

## J — 100ep holdout (legacy GRPO · completed)

Same ckpt as 25ep row · metric harder.

| config | u | 25ep best (ref) | 100ep % |
|--------|---|-----------------|---------|
| `b4g4_dense_lr5e6_d40` | 20 | 40% | 23 |
| same | 40 | 36% | 26 |
| `b4g4_lr25e6_clip08` | 38 | — | 22 |
| same | 40 | 44% | 23 |
| `b4g16_sbonus500_seedwave32` | 30 | 32% | 22 |
| `b4g4_sbonus500_lr5e6_clip01` | 40 | 36% | 20 |
| `b8g8_sbonus500_seedwave32` | 30 | 32% | **15** |
| `b4g8_dense_lr3e6_d40` | 20/40 | 40% | 22 / 24 |
| `b8g4_sbonus500_lr5e6_clip01` | 40 | 32% | 27 |
| `b4g4_dense_lr1e6_clip005` | 26/40 | 40% | 16 / 22 |

**read:** 100ep usually **below** 25ep best @ nearby u.

**also in artifacts (g8 vecasync u100):** 100ep@u85 **12%** vs 25ep peak ~29%@u15 — collapse at long holdout.

---

## infra fixes (May 27–28)

| bug | fix |
|-----|-----|
| Direct eval path used eval jobid | `RLINF_EPHEMERAL_BASE/<train_job>/results/...` |
| `--checkpoint-stride` missing GRPO | `eval_phase111_grpo_sweep.py` |
| `libpython3.12` eval | Python module + `LD_LIBRARY_PATH` |
| `smolvla_pipeline` import | `PYTHONPATH` in common.sh |
| moonshot OOM | 8env only |

---

## K — sparse GRPO moonshots (NEW)

**first env-reward GRPO with `sparse_success_delta` + `rel_reward`** (ported to Phase11 `reward_backends.py`).

| run | hparams | train PBS | eval PBS |
|-----|---------|-----------|----------|
| **A stability** | b4×g16, lr**2.5e-6**, clip**0.05**, sparse+rel, low noise, 50u | `phase11_sparse_b4g16_lr25e6_clip005_moonshot_0050_ephemeral.pbs` | `*_eval25_stride2.pbs` |
| **B perf** | b1×g16, lr**5e-6**, clip**0.2**, sparse+rel, 50u | `phase11_sparse_g16_lr5e6_clip02_moonshot_0050_ephemeral.pbs` | `*_eval25_stride2.pbs` |

Job IDs: `docs/sparse_grpo_moonshot_job_ids.txt` · monitor: `logs/overnight_sparse_grpo_status.log`

Submit: `bash scripts/grpo/submit_phase11_sparse_moonshots.sh`

---

## pending (NOT done)

| job | what |
|-----|------|
| **2872532** | Direct 8env eval Q |
| sparse A/B train+eval | see `sparse_grpo_moonshot_job_ids.txt` |

20min monitor loop **stopped** 2026-05-28 ~11:23 UTC · sparse monitor: `scripts/grpo/overnight_sparse_grpo_autonomous.sh`

---

## top @25ep (done, 9d)

| rank | result |
|------|--------|
| 1 | G16 **52%@u74** (70→170) |
| 2 | G16 chain **44%@u70** (20→70) |
| 3 | `b4g4_lr25e6_clip08` **44%@u40** |
| 4 | s3b Direct **40%@u120** |
| 5 | moonshot Direct **36%@u40** |
| 6 | GRPO moonshot **32%@u50** |

---

## artifact index

| what | path |
|------|------|
| tune job IDs | `project/docs/overnight_tune_job_ids.txt` |
| audit JSON | `project/docs/overnight_audit.json` |
| monitor log | `project/logs/overnight_status.log` |
| G16 eval 72–170 | `.../phase11_pushv3_chunk5_g16_lr5e6_clip02_u70/eval25_stride2_0072_0170/` |
| G16 eval 22–70 | `.../eval25_stride2_0022_0070/` |
| G16 u20 artifacts | `project/artifacts/phase11_pushv3_chunk5_g16_lr5e6_clip02_u20/` |
| s3b run + eval | `.../2861110/.../smolvla_direct_stage3b_sparse_resume100_150_2861110.pbs-7/` |
