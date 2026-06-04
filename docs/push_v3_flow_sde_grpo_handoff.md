# MetaWorld Push-v3 — Flow-SDE Chunk GRPO Results Handoff

> **Purpose:** Zero-context transfer doc for another LLM. Summarizes what was run on **push-v3**, best **100-episode** real-env evals, hyperparameters, infrastructure, and the main winning lineage (Jun 2026 experiments on Imperial DoC **gpucluster3**).

---

## TL;DR

| Metric | Value |
|--------|-------|
| **Best 100-ep success (any checkpoint, any run)** | **41%** @ **update 45** |
| **First checkpoint ≥40% on 100-ep eval** | **update 25** (40%) — moonshot sparse30 run |
| **Best 100-ep in extended u45 exploit chain** | **40%** @ u55 (lr3e6) and u60 (lr5e6); **39%** @ u75/u80 (lr3e6) |
| **Pretrained baseline (100-ep)** | ~13–17% depending on eval job |
| **Algorithm** | **Flow-SDE chunk GRPO** (`logprob_mode=flow_sde`, `rollout_unit=chunk`, chunk len **5**) |
| **Policy** | HuggingFace **jadechoghari/smolvla_metaworld** (frozen snapshot path below) |
| **Task** | MetaWorld **push-v3** — “Push the puck to a goal” |
| **Typical 100-ep eval wall time** | **~152–175 s** per checkpoint (25 parallel envs, 100 episodes) |
| **Typical 1000-ep eval wall time** | **~1550–1560 s** (~26 min) per checkpoint |

---

## What we did (high level)

1. **Started from** a pretrained SmolVLA MetaWorld checkpoint (no GRPO).
2. **Trained on-policy GRPO** in the real MetaWorld push-v3 environment (not world-model-only), using:
   - **Flow-SDE** for action log-probabilities (stochastic flow matching path, not plain Gaussian on actions alone).
   - **Chunked rollouts** — policy executes **5** action steps per chunk; GRPO groups compare chunk trajectories.
   - **Vector-async** parallel env rollouts during training (`rollout_execution=vector_async`).
   - **Sparse success delta** reward (binary success signal, delta form).
   - **`no_tanh`** action transform (important — older `tanh` runs peaked ~40% on **10-ep** only).
3. **Ran a moonshot sweep** (`flow_sde_chunk_grpo_moonshot_sparse30`) for updates **0→30**, then **resume chains** targeting >40% success on **100-episode** eval.
4. **Evaluated** checkpoints with `eval_smolvla_metaworld_ckpt_sweep.py` (**rlinf_fast** backend):
   - **100 ep:** seeds **1000–1099**, **25** envs, **150** max steps/ep, chunk len **5**.
   - **1000 ep (subset):** seeds **4000–4999**, same parallelism.
5. **Pruned disk:** deleted `.pt` files for runs whose **best** 100-ep eval was **<35%**; kept eval JSON/logs. Several strong runs kept; many checkpoint dirs are now empty except eval artifacts.
6. **Best peak** came from **`u25_aggressive_lr1e5`** resume (u25→u55), reaching **41% @ u45** before later exploit phases.

---

## Best results (100-episode eval)

### Global maximum (all Flow-SDE runs scanned)

| Success | Update | Run / eval artifact | Eval wall (s) |
|---------|--------|---------------------|---------------|
| **41%** | **45** | `u25_aggressive_lr1e5` — `eval100_246945` | 153.4 |

Path: `/vol/bitbucket/aa6622/project/artifacts/flow_sde_over40_moonshots_20260603/u25_aggressive_lr1e5/eval100_246945/eval100_summary.json`

Checkpoint (eval): `.../u25_aggressive_lr1e5/train_246945/checkpoints_eval/update_0045.pt`

### Other top 100-ep scores (same protocol)

| Success | Update | Run |
|---------|--------|-----|
| 40% | 25 | `flow_sde_chunk_grpo_moonshot_sparse30` (first ≥40%) |
| 40% | 40 | `u25_aggressive_lr1e5` |
| 40% | 55 | `u45_exploit_lr3e6` (eval100_246995) |
| 40% | 60 | `u45_exploit_lr5e6` (eval100_247023) |
| 40% | 30 | `u25_exploit_lr5` |
| 39% | 36, 75, 80 | `u45_exploit_lr3e6` (various resume jobs) |
| 39% | 35 | `u25_exploit_lr25` |

### Earliest update ≥40% (100-ep)

- **Update 25 @ 40%** — `flow_sde_chunk_grpo_moonshot_sparse30/train_246694`  
  This is the **first** RL checkpoint in the main lineage that cleared 40% under the standard 100-ep protocol.

### Pretrained baseline (100-ep)

Typically **13–17%** (eval job dependent; e.g. 14% in `u25_aggressive` eval, 17% in `u45_exploit_lr3e6` eval100_246995).  
Update **0** in eval JSON means **baseline policy**, not a GRPO checkpoint.

---

## 1000-episode eval (higher fidelity, seeds 4000–4999)

| Checkpoint | 1000-ep success | Eval wall (s) | Notes |
|------------|-----------------|---------------|-------|
| Baseline | 18.0% | 1545.1 | lineage batch job |
| u25 | 29.5% | 1552.2 | lineage batch job |
| **u80** | **35.1%** | 1559.7 | dedicated job; 100-ep was **39%** |
| u45, u60 | *not finished* | — | batch job `247030` incomplete for these two |

u80 summary: `.../u45_exploit_lr3e6/eval1000_u80_seeds4000_4999/eval1000_summary.json`  
**Note:** `update_0080.pt` was later removed from disk during checkpoint cleanup (eval JSON retained). 1000-ep result above was captured while the file existed.

---

## Main winning lineage (eval curve with wall times)

Canonical chain: **moonshot sparse30 (u0–25)** → **u25_aggressive (u30–55)** → **u45_exploit_lr3e6** resumes → optional **u45_exploit_lr5e6** sibling.

| Phase | Train job | Updates trained | 100-ep @ key updates | Eval wall (s) |
|-------|-----------|-----------------|----------------------|---------------|
| moonshot_sparse30 | 246694 | 0→30 | u25 **40%** | 160.6 |
| u25_aggressive_lr1e5 | 246945 | 25→55 (resume) | u40 **40%**, u45 **41%** | 153.3, 153.4 |
| u45_exploit_lr3e6 | 246995 | 45→55 (resume) | u55 **40%** | 154.6 |
| u45_exploit_lr3e6 | 247010 | 55→80 (resume) | u75/u80 **39%** | 154.1, 154.0 |
| u45_exploit_lr5e6 | 247023 | 55→70 (resume) | u60 **40%** | 152.9 |

### Full 100-ep table (main chain only)

| Update | Success | Eval wall (s) | Training phase |
|--------|---------|---------------|----------------|
| 0 (baseline) | 13% | 177.0 | pretrained (moonshot eval) |
| 5 | 24% | 159.6 | moonshot_sparse30 |
| 10 | 26% | 159.0 | moonshot_sparse30 |
| 15 | 34% | 158.5 | moonshot_sparse30 |
| 20 | 34% | 159.0 | moonshot_sparse30 |
| **25** | **40%** | 160.6 | moonshot_sparse30 — **first ≥40%** |
| 30 | 36% | 153.2 | u25_aggressive |
| 35 | 36% | 153.0 | u25_aggressive |
| 40 | 40% | 153.3 | u25_aggressive |
| **45** | **41%** | 153.4 | u25_aggressive — **global max** |
| 50 | 36% | 153.3 | u25_aggressive |
| 55 | 37% | 153.5 | u25_aggressive |
| 55 | 40% | 154.6 | u45_exploit_lr3e6 (re-eval same weights + continued train) |
| 60 | 38% | 153.6 | u45_exploit_lr3e6 (247010) |
| 65 | 34% | 154.3 | u45_exploit_lr3e6 (247010) |
| 70 | 37% | 153.8 | u45_exploit_lr3e6 (247010) |
| 75 | 39% | 154.1 | u45_exploit_lr3e6 (247010) |
| 80 | 39% | 154.0 | u45_exploit_lr3e6 (247010) |

Side experiment **u45_exploit_lr5e6** (lr **5e-6** instead of 3e-6): u60 **40%**, u65 37%, u70 33%.

---

## Hyperparameters (training)

### Shared core (all Flow-SDE chunk push-v3 runs)

| Parameter | Value |
|-----------|-------|
| Task | `push-v3` |
| Env backend | `official_lerobot` |
| Rollout execution | `vector_async` |
| Rollout unit | **`chunk`** |
| Rollout / policy chunk len | **5** |
| Logprob mode | **`flow_sde`** |
| Flow-SDE trace step | **9** |
| Flow-SDE noise level | **1.0** |
| Euler step noise std | **0.0** |
| Reward mode | **`sparse_success_delta`** |
| Action transform | **`no_tanh`** |
| Gaussian logprob action | `executed` |
| Max episode steps (train rollouts) | **120** |
| Group size (GRPO) | **16** |
| Batch size | **1** |
| Save every | **5** updates (some resume jobs: **2**) |
| Parity tolerance | **0.02** (`--fail-on-parity-violation`) |
| Init / min log-std | **-2.0** / **-4.0** |

### Per-phase learning rates & clipping

| Run label | Updates | LR | clip_eps | train_seed_base | Resume from |
|-----------|---------|-----|----------|-----------------|-------------|
| `flow_sde_chunk_grpo_moonshot_sparse30` | 0→30 | **7.5e-6** | **0.2** | 2000 | pretrained HF |
| `flow_sde_ms_u25_aggressive_lr1e5` | 25→55 | **1e-5** | **0.15** | 10002 | u25 moonshot ckpt |
| `flow_sde_u45_exploit_lr3e6` | 45→55, 55→80, … | **3e-6** | **0.1** | 11002 | u45 aggressive ckpt |
| `flow_sde_u45_exploit_lr5e6` | 45→55, 55→70 | **5e-6** | **0.1** | 11001 | u45 aggressive ckpt |

**Exploit-phase intent:** After peaking at u45 (41%), lower LR + tighter clip (0.1) to **stabilize/exploit** without destroying the policy. lr3e6 extended to u80; lr5e6 sibling tested slightly higher LR.

---

## Evaluation protocol (100-ep and 1000-ep)

| Setting | 100-ep | 1000-ep |
|---------|--------|---------|
| Script | `RLinf-smolvla-metaworld-ppo-grpo/scripts/eval_smolvla_metaworld_ckpt_sweep.py` |
| Backend | `rlinf_fast` |
| Task | push-v3 |
| Seeds | 1000–1099 | 4000–4999 |
| num_episodes | 100 | 1000 |
| num_envs | 25 | 25 |
| max_episode_steps | 150 | 150 |
| chunk_len | 5 | 5 |
| Summary file | `eval100_summary.json` (`rows_100ep`) | `eval1000_summary.json` (`rows_1000ep`) |
| Metric | `success_rate` (0–1) → **%** = ×100 | same |

**Wall time** = `eval_wall_s` in summary JSON (seconds for full eval of that checkpoint).

### Per-checkpoint fields in `eval100_summary.json`

Each element of `rows_100ep` is one plotted point:

| Field | Type | Meaning |
|-------|------|---------|
| `update` | int | Training update index (`0` = pretrained baseline, not a GRPO ckpt) |
| `success_rate` | float 0–1 | Fraction of successful episodes → **plot Y = `success_rate * 100`** |
| `eval_wall_s` | float | Wall-clock seconds for this eval job on that checkpoint |
| `checkpoint` | str | Path to `update_XXXX.pt` or `"baseline"` |
| `episode_len_mean` | float | Mean episode length (usually 150 at cap) |
| `eval_seeds` | int[] | Episode seeds used (length 100; should be 1000–1099) |

**X-axis for lineage plots:** use `update` as the global training step (continuous across resume jobs).

**Raw JSON paths** (machine-readable source of truth):

| Series label | `eval100_summary.json` path |
|--------------|----------------------------|
| moonshot_sparse30 | `.../flow_sde_chunk_grpo_moonshot_sparse30/eval100_all_ckpts_246694/eval100_summary.json` |
| u25_aggressive_lr1e5 | `.../flow_sde_over40_moonshots_20260603/u25_aggressive_lr1e5/eval100_246945/eval100_summary.json` |
| u25_exploit_lr5 | `.../flow_sde_over40_moonshots_20260603/u25_exploit_lr5/eval100_246752/eval100_summary.json` |
| u45_exploit_lr3e6 (job 246995) | `.../u45_exploit_lr3e6/eval100_246995/eval100_summary.json` |
| u45_exploit_lr3e6 (job 247010) | `.../u45_exploit_lr3e6/eval100_247010/eval100_summary.json` |
| u45_exploit_lr3e6 (job 247016, save/2) | `.../u45_exploit_lr3e6/eval100_247016/eval100_summary.json` |
| u45_exploit_lr3e6 (job 247021) | `.../u45_exploit_lr3e6/eval100_247021/eval100_summary.json` |
| u45_exploit_lr5e6 (job 246994) | `.../u45_exploit_lr5e6/eval100_246994/eval100_summary.json` |
| u45_exploit_lr5e6 (job 247023) | `.../u45_exploit_lr5e6/eval100_247023/eval100_summary.json` |

Prefix: `/vol/bitbucket/aa6622/project/artifacts/`

---

## 100-episode eval time series (plot-ready)

Copy any CSV block below into a spreadsheet or pandas (`pd.read_csv(StringIO(...))`).  
**Y-axis:** `success_pct` (already ×100). **Optional second Y:** `eval_wall_s`.

### How to plot (quick Python)

```python
import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt

csv = """update,success_pct,eval_wall_s
... paste a block below ...
"""
df = pd.read_csv(StringIO(csv.strip()))
fig, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(df["update"], df["success_pct"], "o-", label="success %")
ax1.set_xlabel("training update")
ax1.set_ylabel("100-ep success %")
ax1.set_ylim(0, 45)
ax1.grid(True, alpha=0.3)
if "eval_wall_s" in df.columns:
    ax2 = ax1.twinx()
    ax2.plot(df["update"], df["eval_wall_s"], "s--", color="gray", alpha=0.6, label="eval wall s")
    ax2.set_ylabel("eval wall time (s)")
plt.title("push-v3 Flow-SDE chunk GRPO — 100ep eval")
plt.tight_layout()
plt.savefig("push_v3_eval100_curve.png", dpi=150)
```

For **multiple runs** on one figure, plot each per-run CSV below with a different color/label.

---

### Series A — Primary winning lineage (single line, recommended)

One continuous curve: moonshot **u0–u30** → u25_aggressive **u35–u55** → u45_exploit_lr3e6 **u60–u80** (eval job 247010).  
**Peak:** u45 @ **41%**. Update 30 appears only once (moonshot eval; aggressive also trained through u30).

```csv
update,success_pct,eval_wall_s,eval_job,run_label
0,13.00,176.97,eval100_246694,moonshot_sparse30
5,24.00,159.585,eval100_246694,moonshot_sparse30
10,26.00,159.019,eval100_246694,moonshot_sparse30
15,34.00,158.481,eval100_246694,moonshot_sparse30
20,34.00,158.994,eval100_246694,moonshot_sparse30
25,40.00,160.581,eval100_246694,moonshot_sparse30
30,36.00,159.132,eval100_246694,moonshot_sparse30
35,36.00,153.038,eval100_246945,u25_aggressive_lr1e5
40,40.00,153.289,eval100_246945,u25_aggressive_lr1e5
45,41.00,153.417,eval100_246945,u25_aggressive_lr1e5
50,36.00,153.301,eval100_246945,u25_aggressive_lr1e5
55,37.00,153.502,eval100_246945,u25_aggressive_lr1e5
60,38.00,153.63,eval100_247010,u45_exploit_lr3e6
65,34.00,154.275,eval100_247010,u45_exploit_lr3e6
70,37.00,153.812,eval100_247010,u45_exploit_lr3e6
75,39.00,154.059,eval100_247010,u45_exploit_lr3e6
80,39.00,154.028,eval100_247010,u45_exploit_lr3e6
```

---

### Series B — `moonshot_sparse30` (full eval sweep, job 246694)

```csv
update,success_pct,eval_wall_s,checkpoint
0,13.00,176.97,baseline
5,24.00,159.585,update_0005
10,26.00,159.019,update_0010
15,34.00,158.481,update_0015
20,34.00,158.994,update_0020
25,40.00,160.581,update_0025
30,36.00,159.132,update_0030
```

---

### Series C — `u25_aggressive_lr1e5` (full eval, job 246945) — **contains global max**

```csv
update,success_pct,eval_wall_s,checkpoint
0,14.00,190.829,baseline
30,36.00,153.221,update_0030
35,36.00,153.038,update_0035
40,40.00,153.289,update_0040
45,41.00,153.417,update_0045
50,36.00,153.301,update_0050
55,37.00,153.502,update_0055
```

---

### Series D — `u25_exploit_lr5` (sibling top run, job 246752)

```csv
update,success_pct,eval_wall_s,checkpoint
0,16.00,189.309,baseline
30,40.00,168.178,update_0030
35,37.00,168.779,update_0035
40,35.00,168.379,update_0040
45,33.00,167.514,update_0045
50,37.00,168.377,update_0050
55,38.00,168.744,update_0055
```

---

### Series E — `u45_exploit_lr3e6` (all eval jobs merged; multiple points per update possible)

Use for fine-grained or branch comparisons. **247016** = resume u35→u50 with **save every 2** (dense evals u36–u50).

```csv
update,success_pct,eval_wall_s,eval_job,notes
50,36.00,154.924,eval100_246995,u45→u55 train phase
55,40.00,154.615,eval100_246995,u45→u55 train phase
36,39.00,152.232,eval100_247016,resume u35→u50 save/2
38,35.00,152.508,eval100_247016,resume u35→u50 save/2
40,35.00,152.528,eval100_247016,resume u35→u50 save/2
42,34.00,152.345,eval100_247016,resume u35→u50 save/2
44,37.00,153.254,eval100_247016,resume u35→u50 save/2
46,36.00,152.436,eval100_247016,resume u35→u50 save/2
48,35.00,152.244,eval100_247016,resume u35→u50 save/2
50,35.00,152.353,eval100_247016,resume u35→u50 save/2
60,38.00,153.63,eval100_247010,resume u55→u80
65,34.00,154.275,eval100_247010,resume u55→u80
70,37.00,153.812,eval100_247010,resume u55→u80
75,39.00,154.059,eval100_247010,resume u55→u80
80,39.00,154.028,eval100_247010,resume u55→u80
60,38.00,183.20,eval100_247021,resume u55→u70 (duplicate u60)
65,35.00,181.548,eval100_247021,resume u55→u70
70,37.00,179.868,eval100_247021,resume u55→u70
```

Baselines in u45-folder eval jobs (same pretrained policy, different eval runs): **14–19%** @ u0 — do not mix with GRPO checkpoint rows on the same axis without labeling.

---

### Series F — `u45_exploit_lr5e6` (lr 5e-6 sibling)

```csv
update,success_pct,eval_wall_s,eval_job,checkpoint
50,37.00,154.565,eval100_246994,update_0050
55,37.00,154.667,eval100_246994,update_0055
60,40.00,152.867,eval100_247023,update_0060
65,37.00,153.032,eval100_247023,update_0065
70,33.00,153.467,eval100_247023,update_0070
```

---

### Multi-series plot legend (suggested)

| Label | CSV section | Color idea |
|-------|-------------|------------|
| Primary lineage | Series A | solid blue |
| u25 aggressive only | Series C | orange |
| u25 exploit lr5 | Series D | green |
| u45 lr3e6 dense (247016) | Series E (u36–50 only) | purple dots |
| u45 lr5e6 | Series F | red dashed |

**Vertical reference lines:** u25 (first 40%), u45 (global max 41%), u55 (exploit restart), u80 (last lr3e6 ckpt before cleanup).

---

### Episode-level data (optional, not in CSV above)

Per-checkpoint episode outcomes live under each eval run directory, e.g.:

`eval100_<jobid>/eval_1000ep/update_XXXX/eval_info.json` and `episodes/*/actions.jsonl`

Use only if you need per-seed breakdown; the summary JSON `success_rate` is the mean over the 100 episodes (seeds `seed_base` … `seed_base + num_episodes - 1`).

---

## Infrastructure & how jobs were run

| Item | Detail |
|------|--------|
| Cluster | Imperial DoC **gpucluster3** (Slurm) |
| GPU partition | **a30** (NVIDIA A30), typically **1 GPU**, **16 CPU**, **64–100G RAM** |
| Job wrapper | `project/scripts/grpo/submit_flow_sde_*.slurm` |
| Env setup | `scripts/slurm/common_env.sh`, `--export=NIL`, HF/torch caches via `slurm_export_hf_torch_cache` |
| Training entrypoint | `project/scripts/grpo/train_phase11_env_on_policy_grpo.py` |
| Train Python | `/vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python` (typical) |
| Eval Python | RLinf env python via `rlinf_smolvla_common.sh` |
| Artifacts root | `/vol/bitbucket/aa6622/project/artifacts/` |

### Key Slurm scripts

- `submit_flow_sde_chunk_grpo_moonshot30_train_eval100_a30.slurm` — train + 100ep eval
- `submit_flow_sde_chunk_grpo_resume_train_eval100_a30.slurm` — parameterized resume (15/25 updates, save frequency, eval all ckpts)
- `submit_flow_sde_chunk_grpo_eval1000_a30.slurm` — 1000-ep single checkpoint
- `submit_flow_sde_lineage_eval1000x4_a30.slurm` — batch baseline/u25/u45/u60 @ 1000 ep

---

## Model & checkpoint paths

| Role | Path |
|------|------|
| Pretrained HF snapshot | `/vol/bitbucket/aa6622/.cache/huggingface/hub/models--jadechoghari--smolvla_metaworld/snapshots/ef3089ecb84eeeb7d33fedab24f6c76180a68900` |
| u25 (40% @ 100ep) | `.../flow_sde_chunk_grpo_moonshot_sparse30/train_246694/checkpoints_eval/update_0025.pt` |
| u45 (**41%** peak) | `.../flow_sde_over40_moonshots_20260603/u25_aggressive_lr1e5/train_246945/checkpoints_eval/update_0045.pt` |
| u80 (39% @ 100ep; 35.1% @ 1000ep) | **removed** from disk after eval; metrics in `eval100_247010` / `eval1000_u80_*` JSON |

Train manifests (exact hyperparams per job): `train_<jobid>/train_manifest.json` under each run directory.

---

## Experiment families (context)

| Family | Approx. scale | Outcome |
|--------|---------------|---------|
| `flow_sde_chunk_grpo_train{10,16,40}_*` | ~120 early Jun runs | Most **<35%** @ 100ep → **.pt deleted** |
| `flow_sde_chunk_grpo_moonshot_sparse30` | 30 updates from baseline | **40% @ u25** — kept |
| `flow_sde_over40_moonshots_20260603/*` | ~60 “over 40%” targeted resumes | **u25_aggressive 41%**, exploit variants high-30s/low-40s |
| Older `phase111_on_grpo_lerobot/push-v3/*` | Pre-flow-SDE / 10-ep eval | **40% @ u25 on 10-ep only** — not comparable to 100-ep protocol |

---

## What *not* to confuse

1. **Flow-SDE chunk GRPO** (Jun 2026, 100-ep eval) is the relevant stack — not WM-only training, not 10-ep smokes.
2. **Update numbers are global** across resume chains (u45 in exploit run = same weights as u45 from aggressive run).
3. **Different eval jobs** can report slightly different baseline % for the same pretrained weights (14% vs 17%) — always compare within the same `eval100_summary.json`.
4. **1000-ep** scores are lower variance but **not finished** for all lineage points; u80 1000-ep (35.1%) < 100-ep (39%) is within noise at N=1000 vs point estimate at N=100.
5. Many **checkpoint `.pt` files were deleted** to save space; **eval JSON/logs remain** as source of truth.

---

## Suggested next steps for a new LLM

1. Treat **`u25_aggressive` u45 @ 41%** as the best policy found under standard 100-ep eval (if `update_0045.pt` still exists).
2. If re-training: resume from u45 or u25 with **flow_sde + chunk5 + sparse_success_delta + no_tanh**, try lr **3e-6–1e-5**, clip **0.1–0.15**, group **16**.
3. For reporting: always cite **100-ep seeds 1000–1099**; use **1000-ep 4000–4999** for paper-grade numbers.
4. Re-run eval before claiming live performance if checkpoints were pruned.

---

## Key artifact paths (quick index)

```
project/artifacts/flow_sde_chunk_grpo_moonshot_sparse30/
  train_246694/train_manifest.json
  eval100_all_ckpts_246694/eval100_summary.json

project/artifacts/flow_sde_over40_moonshots_20260603/
  u25_aggressive_lr1e5/eval100_246945/eval100_summary.json   # 41% @ u45
  u45_exploit_lr3e6/eval100_{246995,247010,247016,247021}/eval100_summary.json
  u45_exploit_lr5e6/eval100_{246994,247023}/eval100_summary.json
  u45_exploit_lr5e6/eval1000_lineage_seeds4000_4999/         # partial 1000-ep

project/scripts/grpo/train_phase11_env_on_policy_grpo.py
RLinf-smolvla-metaworld-ppo-grpo/scripts/eval_smolvla_metaworld_ckpt_sweep.py
```

*Generated 2026-06-02 from on-disk `eval100_summary.json` / `eval1000_summary.json` and `train_manifest.json` files.*
