# Phase11 Recent Run Ledger

Date: 2026-05-19

Scope: recent Phase11 SmolVLA GRPO runs for MetaWorld `push-v3`, including the earlier G8 run, G16 run, Run A, R1/R2/R3, baseline eval, smoke tests, and later pop128 attempts. Every numeric result below points to an artifact/log/source path.

## Executive Summary

Best held-out 100 episode GRPO result so far is tied at `30%` success:

- Run A G32 update `0006`: `30%`, avg sum reward `83.65`.
- R2 G32 low-noise update `0018`: `30%`, avg sum reward `71.15`.
- Baseline SmolVLA on the same 100 eval seeds: `21%`, avg sum reward `70.22`.

Best 20/25 episode sweeps:

- G8 original 0-50 sweep: best success `48%` at update `0010`; best reward `93.69` at update `0015`.
- G8 full 0-100 sweep: best success `48%` at update `0015`; later updates did not recover.
- G8 100ep confirmation at update `0010`: `33%` success (was `48%` on 25ep); still above `21%` baseline; same 25ep→100ep drop as Run A.
- G16: best `44%` at update `0010` and again update `0020`; update `0020` had much higher avg sum reward `121.93`.
- Run A G32: best `45%` at update `0006` on 20 episodes.
- R1/R2: best 20 episode success both `40%`; R2 had stronger 100 episode confirmation (`30%` vs R1 `22%`).

Major operational finding:

- Batched logprob recompute worked: Run A slow section had median update time dominated by optimization; resumed batched section and R1/R2 had median update times near `190s`.
- G64 failed from CUDA OOM during SmolVLA image embedding. Pop128 with rollout policy microbatch got past GPU OOM but failed/killed after update 0 due host memory/process pressure.

## Common Setup

All main runs used:

- Base checkpoint: `/rds/general/user/aa6622/home/.cache/huggingface/hub/models--jadechoghari--smolvla_metaworld/snapshots/ef3089ecb84eeeb7d33fedab24f6c76180a68900`
- Task: `push-v3`
- Env backend: `official_lerobot`
- Rollout execution: `vector_async`
- Async start method / multiprocessing context: `forkserver`
- Action transform: `no_tanh`
- Action chunk size: `5`
- Optimizer chunk size: `5`
- Max steps: `120`
- Train reset mode: `SMOLVLA_METAWORLD_RESET_MODE=random_seeded`
- Train seeds: `train_seed_base + update`, generally `2000 + update`
- Eval seeds: `eval_seed_start=1000`, with `episodes` determining seed range

Primary source for common setup:

- `scripts/grpo/train_phase11_env_on_policy_grpo.py`
- `scripts/grpo/eval_phase111_grpo_sweep.py`
- Per-run `train_manifest.json` files listed below
- Per-run PBS scripts listed below

## Run Inventory

| Run | Status | Purpose | Main result | Primary sources |
| --- | --- | --- | --- | --- |
| G8 `no_tanh_main` | completed | Original Phase11 run, 100 train updates total | Best 25ep success `48%` u10; 100ep u10 `33%`; full sweep drifted by u100 | `artifacts/phase11_pushv3_chunk5_g8_vecasync_u100/` |
| G16 `lr5e-6 clip0.2` | completed | Mid-population + lower LR check | Best 25ep success `44%`; best reward `121.93` at u20 | `artifacts/phase11_pushv3_chunk5_g16_lr5e6_clip02_u20/` |
| Batched logprob smoke | passed | Validate batched logprob default, batch size 16 | 2 updates passed, mean optimize `93.55s` | `artifacts/phase11_pushv3_batched_logprob_smoke_u2/` |
| Run A G32 | completed after resume | First larger-pop run, lower LR/clip | 20ep best `45%` u6; 100ep u6 `30%` | `artifacts/phase11_pushv3_chunk5_g32_lr5e6_clip01_u30/` |
| R1 G32 | completed | Stable config, lower LR/clip | 20ep best `40%`; 100ep best `22%` u8 | `artifacts/phase11_pushv3_chunk5_g32_lr2e6_clip005_u50/` |
| R2 G32 low-noise | completed | Same LR/clip as A, lower policy noise | 20ep best `40%`; 100ep best `30%` u18 | `artifacts/phase11_pushv3_chunk5_g32_lr5e6_clip01_lownoise_u50/` |
| R3 G64 | failed | Larger population | OOM after update 0, no eval | `artifacts/phase11_pushv3_chunk5_g64_lr5e6_clip01_u50/` |
| Baseline 100ep | completed | Base SmolVLA reference | `21%` success, avg sum reward `70.22` | `artifacts/phase11_baseline_pushv3_100ep_s1000_nenv25_chunk5/` |
| Pop128 smoke | partial/failure after training | Validate `rollout_policy_batch_size=16` | 1 update wrote progress, post-check KeyError in script | `artifacts/phase11_pushv3_pop128_rolloutpbs32_smoke_u1/` |
| P128 A/B/C | failed after update 0 | Pop128 production attempts | 1 update each, killed/BrokenPipe/memory pressure | `artifacts/phase11_pushv3_chunk5_pop128_*` |

## G8 `no_tanh_main`

### Sources

- Manifest: `artifacts/phase11_pushv3_chunk5_g8_vecasync_u100/train_manifest.json`
- Progress: `artifacts/phase11_pushv3_chunk5_g8_vecasync_u100/progress.jsonl`
- Job IDs: `artifacts/phase11_pushv3_chunk5_g8_vecasync_u100/pbs_job_ids.txt`
- Eval sweep 5-50: `artifacts/phase11_pushv3_chunk5_g8_vecasync_u100/eval_sweep_0005_0050_25ep_nenv25_async/eval_sweep_summary.json`
- Eval sweep 5-100: `artifacts/phase11_pushv3_chunk5_g8_vecasync_u100/eval_sweep_0005_0100_25ep_nenv25_async/eval_sweep_summary.json`
- 100ep confirm u10:
  - Script: `scripts/grpo/phase11_g8_update0010_100ep_eval.pbs`
  - Sweep summary: `artifacts/phase11_pushv3_chunk5_g8_vecasync_u100/eval_update_0010_100ep_nenv25_async/eval_sweep_summary.json`
  - Per-checkpoint summary: `artifacts/phase11_pushv3_chunk5_g8_vecasync_u100/eval_update_0010_100ep_nenv25_async/update_0010/eval_summary.json`
  - Per-episode: `artifacts/phase11_pushv3_chunk5_g8_vecasync_u100/eval_update_0010_100ep_nenv25_async/update_0010/eval_episodes.jsonl`
  - Log: `logs/pbs/grpo/phase11_g8_update0010_100ep_eval.out`
  - PBS job: `2777665.pbs-7`
- Train 50-100 log: `logs/pbs/grpo/phase11_chunk5_train_0050_0100_resume.out`
- Eval 5-50 log: `logs/pbs/grpo/phase11_chunk5_eval_0005_0050.out`
- Eval 5-100 log: `logs/pbs/grpo/phase11_chunk5_eval_0005_0100.out`
- Original PBS scripts were created in earlier chat as `scripts/grpo/phase11_train_0000_0050.pbs`, `scripts/grpo/phase11_train_0050_0100_resume.pbs`, `scripts/grpo/phase11_eval_0005_0050.pbs`, and `scripts/grpo/phase11_eval_0005_0100.pbs`, but these files are not present in the current working tree. Exact script content remains visible in [Phase11 experiment setup](7a9d5d55-4766-4df8-b911-352e8d278b36), and runtime/log/artifact sources above remain present.

### Hyperparameters

| Param | Value | Source |
| --- | --- | --- |
| group_size | `8` | `train_manifest.json` |
| lr | `1e-5` | `train_manifest.json` |
| clip_eps | `0.2` | `train_manifest.json` |
| init_log_std | default `-2.0` | trainer default / not explicitly in manifest |
| euler_step_noise_std | default `0.2` | trainer default / not explicitly in manifest |
| action_chunk_size | `5` | `train_manifest.json` |
| chunk_size | `5` | `train_manifest.json` |
| max_steps | `120` | `train_manifest.json` |
| save_every | `5` | checkpoints every 5 updates; eval summaries |
| train_seed_base | `2000` | `train_manifest.json` |
| run_label | `no_tanh_main` | `train_manifest.json` |

### Runtime / Resources

| Segment | PBS job | Runtime | CPU | Peak memory | Source |
| --- | ---: | ---: | ---: | ---: | --- |
| train 0-50 | `2770909.pbs-7` | log not present now | `8 CPU / 32GB / 1 RTX6000` from prior script/transcript | unknown | `pbs_job_ids.txt`, transcript |
| eval 5-50 | `2770910.pbs-7` | `00:20:18` | `00:46:26` CPU time | `16,428,920kb` | `logs/pbs/grpo/phase11_chunk5_eval_0005_0050.out` |
| train 50-100 | `2770911.pbs-7` | `04:05:01` | `04:09:51` CPU time | `15,690,904kb` | `logs/pbs/grpo/phase11_chunk5_train_0050_0100_resume.out` |
| eval 5-100 | `2770912.pbs-7` | `00:43:23` | `01:28:57` CPU time | `17,461,112kb` | `logs/pbs/grpo/phase11_chunk5_eval_0005_0100.out` |

Progress aggregation from `progress.jsonl`:

| Metric | Value |
| --- | ---: |
| Updates logged | `100` (`0..99`) |
| Mean train rollout success | `14.00%` |
| Best train rollout success | `75.00%` at update `65` |
| Final train rollout success | `0.00%` at update `99` |
| Mean update seconds | `285.27s` |
| Median update seconds | `297.52s` |
| Mean rollout seconds | `53.62s` |
| Mean optimize seconds | `231.63s` |

### Eval Results

Eval sweep 5-50 (`25` episodes/checkpoint, `25` envs, seed start `1000`):

| Update | Success | Avg sum reward | Avg max reward |
| ---: | ---: | ---: | ---: |
| 5 | `32%` | `71.96` | `3.87` |
| 10 | `48%` | `71.51` | `5.16` |
| 15 | `44%` | `93.69` | `4.93` |
| 20 | `36%` | `68.08` | `4.34` |
| 25 | `28%` | `51.89` | `3.42` |
| 30 | `8%` | `39.30` | `1.95` |
| 35 | `8%` | `35.51` | `1.77` |
| 40 | `4%` | `35.79` | `1.49` |
| 45 | `8%` | `35.42` | `1.83` |
| 50 | `12%` | `38.36` | `2.46` |

Eval sweep 5-100 (`25` episodes/checkpoint, `25` envs, seed start `1000`):

| Best metric | Update | Value | Source |
| --- | ---: | ---: | --- |
| Best success | `15` | `48%` | `artifacts/phase11_pushv3_chunk5_g8_vecasync_u100/eval_sweep_0005_0100_25ep_nenv25_async/eval_sweep_summary.json` |
| Best avg sum reward | `90` | `92.36` with only `12%` success | same |
| Final checkpoint | `100` | `8%`, avg sum reward `55.36` | same |

Interpretation: G8 learned early then degraded. Full 0-100 sweep did not beat the early 0-50 peak.

### 100ep confirmation (update `0010`)

Follow-up eval for G8's best 25ep checkpoint (update `0010` from sweep 5-50). Same eval protocol as Run A / R1 / R2 100ep confirms: `100` episodes, `25` envs, `eval_seed_start=1000`, `vector_async`, no videos.

| Segment | PBS job | Status | Walltime | CPU time | Peak memory | Resources | Source |
| --- | ---: | --- | ---: | ---: | ---: | --- | --- |
| eval u10 100ep | `2777665.pbs-7` | finished (`F`) | `00:04:10` (req `00:30:00`) | `00:16:06` | `14,782,532kb` | `32 CPU / 32GB / 1 RTX6000` | `logs/pbs/grpo/phase11_g8_update0010_100ep_eval.out` |

| Metric | 25ep (sweep 5-50, u10) | 100ep (this run) | Source |
| --- | ---: | ---: | --- |
| Success | `48%` | `33%` | 25ep: `eval_sweep_0005_0050_25ep_nenv25_async/eval_sweep_summary.json`; 100ep: `eval_update_0010_100ep_nenv25_async/eval_sweep_summary.json` |
| Avg sum reward | `71.51` | `81.72` | same |
| Avg max reward | `5.16` | `4.08` | same |
| Episodes | `25` | `100` | same |

Interpretation: still above `21%` baseline on 100ep (`artifacts/phase11_baseline_pushv3_100ep_s1000_nenv25_chunk5/eval_summary.json`), but `48%` @ 25ep did not hold at 100ep — same pattern as Run A (`45%` → `30%`). Likely high variance on 25ep, not a stable peak.

## G16 `chunk5_g16_lr5e6_clip02`

### Sources

- Train scripts: `scripts/grpo/phase11_g16_lr5e6_clip02_train_0000_0010.pbs`, `scripts/grpo/phase11_g16_lr5e6_clip02_train_0010_0020_resume.pbs`
- Eval scripts: `scripts/grpo/phase11_g16_lr5e6_clip02_eval_0002_0010.pbs`, `scripts/grpo/phase11_g16_lr5e6_clip02_eval_0012_0020.pbs`
- Manifest: `artifacts/phase11_pushv3_chunk5_g16_lr5e6_clip02_u20/train_manifest.json`
- Progress: `artifacts/phase11_pushv3_chunk5_g16_lr5e6_clip02_u20/progress.jsonl`
- Eval summaries:
  - `artifacts/phase11_pushv3_chunk5_g16_lr5e6_clip02_u20/eval_sweep_0002_0010_25ep_nenv25_async/eval_sweep_summary.json`
  - `artifacts/phase11_pushv3_chunk5_g16_lr5e6_clip02_u20/eval_sweep_0012_0020_25ep_nenv25_async/eval_sweep_summary.json`
- Logs:
  - `logs/pbs/grpo/phase11_g16_lr5e6_clip02_train_0000_0010.out`
  - `logs/pbs/grpo/phase11_g16_lr5e6_clip02_train_0010_0020_resume.out`
  - `logs/pbs/grpo/phase11_g16_lr5e6_clip02_eval_0002_0010.out`
  - `logs/pbs/grpo/phase11_g16_lr5e6_clip02_eval_0012_0020.out`
- Prompt-mentioned doc `docs/findings/2026-05-18-phase11-g16-lr5e6-clip02-summary.md` is not present in `docs/findings/` now. Related plan reference exists at `docs/superpowers/plans/2026-05-18-phase11-cpu-memory-telemetry.md`.

### Hyperparameters

| Param | Value |
| --- | ---: |
| group_size | `16` |
| lr | `5e-6` |
| clip_eps | `0.2` |
| init_log_std | default `-2.0` |
| euler_step_noise_std | default `0.2` |
| action_chunk_size | `5` |
| chunk_size | `5` |
| num_updates | `20` total (`0..19`) |
| save_every | `2` inferred by eval checkpoints `2..20` |
| logprob_recompute_mode | `batched` in resumed manifest |
| logprob_batch_size | `16` |

### Runtime / Resources

| Segment | Runtime | CPU time | Peak memory | PBS resources | Source |
| --- | ---: | ---: | ---: | --- | --- |
| train 0-10 | `01:35:19` | `01:40:42` | `28,670,636kb` | `16 CPU / 48GB / 1 RTX6000`, wall `04:00:00` | train script/log |
| train 10-20 | `00:21:53` | `00:28:52` | `27,296,328kb` | `16 CPU / 48GB / 1 RTX6000`, wall `04:00:00` | resume script/log |
| eval 2-10 | `00:11:40` | `00:17:42` | `16,113,064kb` | `32 CPU / 32GB / 1 RTX6000`, wall `00:45:00` | eval script/log |
| eval 12-20 | `00:10:41` | `00:20:25` | `16,251,864kb` | `32 CPU / 32GB / 1 RTX6000`, wall `00:45:00` | eval script/log |

Progress aggregation:

| Metric | Value |
| --- | ---: |
| Updates logged | `20` (`0..19`) |
| Mean train rollout success | `12.81%` |
| Best train rollout success | `50.00%` at update `16` |
| Final train rollout success | `18.75%` at update `19` |
| Mean update seconds | `327.13s` |
| Mean rollout seconds | `69.32s` |
| Mean optimize seconds | `257.79s` |

### Eval Results

| Sweep | Best success | Best reward | Source |
| --- | --- | --- | --- |
| updates `2..10`, 25ep | `44%` at update `10`; avg sum reward `83.28` | same | `artifacts/phase11_pushv3_chunk5_g16_lr5e6_clip02_u20/eval_sweep_0002_0010_25ep_nenv25_async/eval_sweep_summary.json` |
| updates `12..20`, 25ep | `44%` at update `20`; avg sum reward `121.93` | same | `artifacts/phase11_pushv3_chunk5_g16_lr5e6_clip02_u20/eval_sweep_0012_0020_25ep_nenv25_async/eval_sweep_summary.json` |

Interpretation: G16 matched G8 best success only approximately (44% vs 48%) but update 20 had stronger dense reward.

## Batched Logprob Smoke

### Sources

- Script: `scripts/grpo/phase11_batched_logprob_smoke_u2.pbs`
- Manifest: `artifacts/phase11_pushv3_batched_logprob_smoke_u2/train_manifest.json`
- Progress: `artifacts/phase11_pushv3_batched_logprob_smoke_u2/progress.jsonl`
- Log: `logs/pbs/grpo/phase11_batched_logprob_smoke_u2.out`

### Result

| Metric | Value |
| --- | ---: |
| group_size | `32` |
| num_updates | `2` |
| lr / clip_eps | `5e-6 / 0.1` |
| logprob mode / batch size | `batched / 16` |
| Runtime | `00:08:44` |
| Peak memory | `45,036,228kb` |
| Mean rollout seconds | `71.21s` |
| Mean optimize seconds | `93.55s` |
| Mean update seconds | `164.88s` |
| Train success | `6.25%` at both updates |

Interpretation: smoke was for correctness/perf, not quality. It confirmed batched recompute plus telemetry fields (`ratio_clip_fraction`, `approx_kl`, `log_std_mean`, `num_logprob_forward_batches`) were emitted.

## Run A G32 `chunk5_g32_lr5e6_clip01`

### Sources

- Initial train script: `scripts/grpo/phase11_A_g32_lr5e6_clip01_train_0000_0030.pbs`
- Resume train script: `scripts/grpo/phase11_A_g32_lr5e6_clip01_resume_train_0014_0030.pbs`
- Eval script: `scripts/grpo/phase11_A_g32_lr5e6_clip01_eval_0002_0030.pbs`
- 100ep scripts: `scripts/grpo/phase11_A_update0006_100ep_eval.pbs`, `scripts/grpo/phase11_A_update0014_100ep_eval.pbs`
- Manifest: `artifacts/phase11_pushv3_chunk5_g32_lr5e6_clip01_u30/train_manifest.json`
- Progress: `artifacts/phase11_pushv3_chunk5_g32_lr5e6_clip01_u30/progress.jsonl`
- 20ep eval: `artifacts/phase11_pushv3_chunk5_g32_lr5e6_clip01_u30/eval_sweep_0002_0030_20ep_nenv25_async/eval_sweep_summary.json`
- 100ep evals:
  - `artifacts/phase11_pushv3_chunk5_g32_lr5e6_clip01_u30/eval_update_0006_100ep_nenv25_async/update_0006/eval_summary.json`
  - `artifacts/phase11_pushv3_chunk5_g32_lr5e6_clip01_u30/eval_update_0014_100ep_nenv25_async/update_0014/eval_summary.json`
- Logs:
  - `logs/pbs/grpo/phase11_A_g32_lr5e6_clip01_train_0000_0030.out`
  - `logs/pbs/grpo/phase11_A_g32_lr5e6_clip01_resume_train_0014_0030.out`
  - `logs/pbs/grpo/phase11_A_g32_lr5e6_clip01_eval_0002_0030.out`
  - `logs/pbs/grpo/phase11_A_update0006_100ep_eval.out`
  - `logs/pbs/grpo/phase11_A_update0014_100ep_eval.out`

### Hyperparameters

| Param | Value |
| --- | ---: |
| group_size | `32` |
| lr | `5e-6` |
| clip_eps | `0.1` |
| init_log_std | default `-2.0` |
| euler_step_noise_std | default `0.2` |
| action_chunk_size | `5` |
| chunk_size | `5` |
| save_every | `2` |
| num_updates | planned `30`; completed via initial + resume |
| logprob_recompute_mode | initial default/slow for updates 0-13; `batched` for resume updates 14-29 |
| logprob_batch_size | `16` on resume |

### Runtime / Resources

| Segment | Runtime | CPU time | Peak memory | PBS resources | Source |
| --- | ---: | ---: | ---: | --- | --- |
| initial train | `04:19:18` | `05:14:44` | `52,370,684kb` | `48 CPU / 64GB / 1 RTX6000`, wall `24:00:00` | initial script/log |
| resume train | `00:57:17` | `03:02:20` | `52,132,316kb` | `48 CPU / 64GB / 1 RTX6000`, wall `02:00:00` | resume script/log |
| 20ep eval sweep | `00:37:01` | `01:28:02` | `16,660,440kb` | `32 CPU / 32GB / 1 RTX6000`, wall `24:00:00` | eval script/log |
| 100ep u6 eval | `00:04:40` | `00:17:38` | `14,960,800kb` | `32 CPU / 32GB / 1 RTX6000` | u6 log |
| 100ep u14 eval | `00:04:42` | `00:17:32` | `14,948,592kb` | `32 CPU / 32GB / 1 RTX6000` | u14 log |

Progress aggregation:

| Metric | Value |
| --- | ---: |
| Updates logged | `30` (`0..29`) |
| Mean train rollout success | `15.83%` |
| Best train rollout success | `59.38%` at update `10` |
| Final train rollout success | `6.25%` at update `29` |
| Mean update seconds | `606.08s` |
| Median update seconds | `232.38s` |
| Mean optimize seconds | `504.51s` |
| Median optimize seconds | `101.83s` |

High mean vs median reflects slow initial non-batched section and faster resumed section.

### Eval Results

20 episode sweep, updates `2..30`, `25` envs, seeds `1000..1019`:

| Best metric | Update | Success | Avg sum reward | Source |
| --- | ---: | ---: | ---: | --- |
| Best success and reward | `6` | `45%` | `103.75` | `artifacts/phase11_pushv3_chunk5_g32_lr5e6_clip01_u30/eval_sweep_0002_0030_20ep_nenv25_async/eval_sweep_summary.json` |

100 episode confirmations:

| Checkpoint | Success | Avg sum reward | Avg max reward | Source |
| --- | ---: | ---: | ---: | --- |
| update `0006` | `30%` | `83.65` | `3.91` | `artifacts/phase11_pushv3_chunk5_g32_lr5e6_clip01_u30/eval_update_0006_100ep_nenv25_async/update_0006/eval_summary.json` |
| update `0014` | `26%` | `61.36` | `3.34` | `artifacts/phase11_pushv3_chunk5_g32_lr5e6_clip01_u30/eval_update_0014_100ep_nenv25_async/update_0014/eval_summary.json` |

Interpretation: update 6 is a real candidate; later checkpoint weaker on 100 episodes.

## R1 G32 `chunk5_g32_lr2e6_clip005`

### Sources

- Train script: `scripts/grpo/phase11_R1_g32_lr2e6_clip005_train_0001_0050.pbs`
- Eval script: `scripts/grpo/phase11_R1_g32_lr2e6_clip005_eval_0001_0050.pbs`
- Manifest/progress:
  - `artifacts/phase11_pushv3_chunk5_g32_lr2e6_clip005_u50/train_manifest.json`
  - `artifacts/phase11_pushv3_chunk5_g32_lr2e6_clip005_u50/progress.jsonl`
- Eval sweep: `artifacts/phase11_pushv3_chunk5_g32_lr2e6_clip005_u50/eval_sweep_0001_0050_20ep_nenv25_async/eval_sweep_summary.json`
- 100ep top-k:
  - `artifacts/phase11_pushv3_chunk5_g32_lr2e6_clip005_u50/eval_sweep_0001_0050_20ep_nenv25_async/topk_update_0008_100ep/eval_summary.json`
  - `artifacts/phase11_pushv3_chunk5_g32_lr2e6_clip005_u50/eval_sweep_0001_0050_20ep_nenv25_async/topk_update_0015_100ep/eval_summary.json`
- Logs:
  - `logs/pbs/grpo/phase11_R1_g32_lr2e6_clip005_train_0001_0050.out`
  - `logs/pbs/grpo/phase11_R1_g32_lr2e6_clip005_eval_0001_0050.out`

### Hyperparameters

| Param | Value |
| --- | ---: |
| group_size | `32` |
| lr | `2e-6` |
| clip_eps | `0.05` |
| init_log_std | default `-2.0` |
| euler_step_noise_std | default `0.2` |
| action_chunk_size | `5` |
| chunk_size | `5` |
| save_every | `1` |
| num_updates | `50` |
| logprob_recompute_mode / batch size | `batched / 16` |

### Runtime / Resources

| Segment | Runtime | CPU time | Peak memory | PBS resources |
| --- | ---: | ---: | ---: | --- |
| train | `02:43:49` | `08:21:56` | `52,634,548kb` | `48 CPU / 64GB / 1 RTX6000`, wall `12:00:00` |
| eval | `01:42:48` | `03:52:31` | `16,452,760kb` | `32 CPU / 32GB / 1 RTX6000`, wall `04:00:00` |

Progress aggregation:

| Metric | Value |
| --- | ---: |
| Updates logged | `50` |
| Mean train rollout success | `15.00%` |
| Best train rollout success | `71.88%` at update `44` |
| Final train rollout success | `43.75%` at update `49` |
| Mean update seconds | `190.05s` |
| Mean rollout seconds | `98.42s` |
| Mean optimize seconds | `91.61s` |

### Eval Results

20 episode sweep:

| Best metric | Update | Success | Avg sum reward | Source |
| --- | ---: | ---: | ---: | --- |
| Best success | `15` | `40%` | `102.72` | `eval_sweep_summary.json` |
| Best reward | `18` | `30%` | `112.22` | `eval_sweep_summary.json` |

100 episode top-k:

| Checkpoint | Success | Avg sum reward | Avg max reward |
| --- | ---: | ---: | ---: |
| update `0008` | `22%` | `69.64` | `3.00` |
| update `0015` | `21%` | `59.50` | `2.83` |

Interpretation: R1 was stable operationally but did not beat baseline convincingly on 100 episodes.

## R2 G32 Low-Noise `chunk5_g32_lr5e6_clip01_lownoise`

### Sources

- Train script: `scripts/grpo/phase11_R2_g32_lr5e6_clip01_lownoise_train_0001_0050.pbs`
- Eval script: `scripts/grpo/phase11_R2_g32_lr5e6_clip01_lownoise_eval_0001_0050.pbs`
- Manifest/progress:
  - `artifacts/phase11_pushv3_chunk5_g32_lr5e6_clip01_lownoise_u50/train_manifest.json`
  - `artifacts/phase11_pushv3_chunk5_g32_lr5e6_clip01_lownoise_u50/progress.jsonl`
- Eval sweep: `artifacts/phase11_pushv3_chunk5_g32_lr5e6_clip01_lownoise_u50/eval_sweep_0001_0050_20ep_nenv25_async/eval_sweep_summary.json`
- 100ep top-k:
  - `artifacts/phase11_pushv3_chunk5_g32_lr5e6_clip01_lownoise_u50/eval_sweep_0001_0050_20ep_nenv25_async/topk_update_0011_100ep/eval_summary.json`
  - `artifacts/phase11_pushv3_chunk5_g32_lr5e6_clip01_lownoise_u50/eval_sweep_0001_0050_20ep_nenv25_async/topk_update_0018_100ep/eval_summary.json`
- Logs:
  - `logs/pbs/grpo/phase11_R2_g32_lr5e6_clip01_lownoise_train_0001_0050.out`
  - `logs/pbs/grpo/phase11_R2_g32_lr5e6_clip01_lownoise_eval_0001_0050.out`

### Hyperparameters

| Param | Value |
| --- | ---: |
| group_size | `32` |
| lr | `5e-6` |
| clip_eps | `0.1` |
| init_log_std | `-2.5` |
| euler_step_noise_std | `0.1` |
| action_chunk_size | `5` |
| chunk_size | `5` |
| save_every | `1` |
| num_updates | `50` |
| logprob_recompute_mode / batch size | `batched / 16` |

Note: `train_manifest.json` does not record `init_log_std` / `euler_step_noise_std`; those are from `scripts/grpo/phase11_R2_g32_lr5e6_clip01_lownoise_train_0001_0050.pbs`.

### Runtime / Resources

| Segment | Runtime | CPU time | Peak memory | PBS resources |
| --- | ---: | ---: | ---: | --- |
| train | `02:45:04` | `09:54:10` | `52,405,796kb` | `48 CPU / 64GB / 1 RTX6000`, wall `12:00:00` |
| eval | `01:42:54` | `03:55:27` | `16,147,824kb` | `32 CPU / 32GB / 1 RTX6000`, wall `04:00:00` |

Progress aggregation:

| Metric | Value |
| --- | ---: |
| Updates logged | `50` |
| Mean train rollout success | `23.88%` |
| Best train rollout success | `96.88%` at update `49` |
| Final train rollout success | `96.88%` at update `49` |
| Mean update seconds | `191.41s` |
| Mean rollout seconds | `98.82s` |
| Mean optimize seconds | `92.57s` |

### Eval Results

20 episode sweep:

| Best metric | Update | Success | Avg sum reward | Source |
| --- | ---: | ---: | ---: | --- |
| Best success | `18` | `40%` | `80.50` | `eval_sweep_summary.json` |
| Best reward | `49` | `15%` | `145.62` | `eval_sweep_summary.json` |

100 episode top-k:

| Checkpoint | Success | Avg sum reward | Avg max reward |
| --- | ---: | ---: | ---: |
| update `0011` | `29%` | `78.22` | `3.68` |
| update `0018` | `30%` | `71.15` | `3.80` |

Interpretation: R2 best matches Run A update 6 on success (`30%`) and has stronger train rollout success late, but held-out eval did not validate the final update. Update 49 train success `96.88%` was one train seed group (`reset_seed=2049`), not “97% probability on same episode.”

## R3 G64 `chunk5_g64_lr5e6_clip01`

### Sources

- Train script: `scripts/grpo/phase11_R3_g64_lr5e6_clip01_train_0001_0050.pbs`
- Eval script: `scripts/grpo/phase11_R3_g64_lr5e6_clip01_eval_0001_0050.pbs`
- Manifest/progress:
  - `artifacts/phase11_pushv3_chunk5_g64_lr5e6_clip01_u50/train_manifest.json`
  - `artifacts/phase11_pushv3_chunk5_g64_lr5e6_clip01_u50/progress.jsonl`
- Failure log: `logs/pbs/grpo/phase11_R3_g64_lr5e6_clip01_train_0001_0050.out`
- GPU telemetry: `artifacts/phase11_pushv3_chunk5_g64_lr5e6_clip01_u50/gpu_telemetry/train/`

### Hyperparameters / Resources

| Param | Value |
| --- | ---: |
| group_size | `64` |
| lr | `5e-6` |
| clip_eps | `0.1` |
| init_log_std | default `-2.0` |
| euler_step_noise_std | default `0.2` |
| action_chunk_size | `5` |
| chunk_size | `5` |
| save_every | `1` |
| num_updates | planned `50`; failed after update 0 |
| logprob_recompute_mode / batch size | `batched / 16` |
| PBS resources | `64 CPU / 96GB / 1 RTX6000`, wall `12:00:00` |

### Runtime / Failure

| Metric | Value |
| --- | ---: |
| Runtime before failure | `00:06:30` |
| CPU time | `00:20:59` |
| Peak memory | `59,269,560kb` |
| Update 0 train success | `10.94%` |
| Update 0 update time | `322.30s` |
| GPU telemetry | `75` samples, max GPU `100%`, mean GPU `69.4%` |

Failure:

```text
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 3.00 GiB.
GPU 0 has a total capacity of 23.46 GiB of which 2.05 GiB is free.
```

Trace source: `logs/pbs/grpo/phase11_R3_g64_lr5e6_clip01_train_0001_0050.out`.

Root cause: rollout used full `group_size=64` as policy forward batch during SmolVLA image embedding, so GPU memory exceeded RTX6000 24GB. Batched logprob only helped optimization; it did not microbatch rollout policy forward before later fixes.

## Baseline 100 Episode Eval

### Sources

- Script: `scripts/grpo/phase11_baseline_pushv3_100ep_nenv25.pbs`
- Eval script: `scripts/grpo/eval_phase12_baseline_vector.py`
- Summary: `artifacts/phase11_baseline_pushv3_100ep_s1000_nenv25_chunk5/eval_summary.json`
- Info: `artifacts/phase11_baseline_pushv3_100ep_s1000_nenv25_chunk5/eval_info.json`
- Log: `logs/pbs/grpo/phase11_baseline_pushv3_100ep_nenv25.out`

### Settings / Runtime / Result

| Metric | Value |
| --- | ---: |
| checkpoint | base SmolVLA only, no GRPO checkpoint |
| episodes | `100` |
| eval_seed_start | `1000` |
| seeds | `1000..1099` |
| n_envs | `25` |
| chunk_len | `5` |
| max_steps | `120` |
| PBS resources | `32 CPU / 32GB / 1 RTX6000`, wall `00:30:00` |
| Runtime | `00:04:09` |
| Peak memory | `18,014,352kb` |
| Success | `21%` |
| Avg sum reward | `70.22` |
| Avg max reward | `2.93` |

## Pop128 / Rollout Policy Microbatch Attempts

These came after diagnosing G64 OOM. They tested `--rollout-policy-batch-size 16`, keeping GRPO `group_size=128` while microbatching SmolVLA rollout forwards.

### Pop128 Smoke

Sources:

- Script: `scripts/grpo/phase11_pop128_rolloutpbs32_smoke_u1.pbs`
- Manifest/progress: `artifacts/phase11_pushv3_pop128_rolloutpbs32_smoke_u1/train_manifest.json`, `progress.jsonl`
- Log: `logs/pbs/grpo/phase11_pop128_rolloutpbs32_smoke_u1.out`

Result:

| Metric | Value |
| --- | ---: |
| group_size | `128` |
| rollout_policy_batch_size | `16` |
| logprob_batch_size | `16` |
| num_updates | `1` |
| train success | `7.03%` |
| rollout / optimize / update seconds | `341.37 / 387.44 / 728.97` |
| Runtime | `00:12:44` |
| Peak memory | `116,366,344kb` |
| GPU telemetry | `151` samples, max GPU `100%`, mean GPU `68.0%` |
| Outcome | update wrote progress, but script post-check hit `KeyError: 'group_size'` |

### P128 A/B/C Production Attempts

Sources:

- Scripts:
  - `scripts/grpo/phase11_P128A_lr2e6_clip005_train_0001_0050.pbs`
  - `scripts/grpo/phase11_P128B_lr5e6_clip01_train_0001_0050.pbs`
  - `scripts/grpo/phase11_P128C_lr5e6_clip01_lownoise_train_0001_0050.pbs`
- Artifacts:
  - `artifacts/phase11_pushv3_chunk5_pop128_rpbs32_lr2e6_clip005_u50/`
  - `artifacts/phase11_pushv3_chunk5_pop128_rpbs32_lr5e6_clip01_u50/`
  - `artifacts/phase11_pushv3_chunk5_pop128_rpbs32_lr5e6_clip01_lownoise_u50/`
- Logs:
  - `logs/pbs/grpo/phase11_P128A_lr2e6_clip005_train_0001_0050.out`
  - `logs/pbs/grpo/phase11_P128B_lr5e6_clip01_train_0001_0050.out`
  - `logs/pbs/grpo/phase11_P128C_lr5e6_clip01_lownoise_train_0001_0050.out`

| Run | lr | clip | init_log_std | euler noise | Update 0 success | Update seconds | Runtime | Peak memory | Outcome |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| P128A | `2e-6` | `0.05` | default `-2.0` | default `0.2` | `7.03%` | `721.45s` | `00:18:28` | `129,656,332kb` | killed after update 0, worker BrokenPipe |
| P128B | `5e-6` | `0.1` | default `-2.0` | default `0.2` | `5.47%` | `745.61s` | `00:18:03` | `134,150,388kb` | killed after update 0 |
| P128C | `5e-6` | `0.1` | `-2.5` | `0.1` | `7.81%` | `721.49s` | `00:17:58` | `128,140,712kb` | killed after update 0 |

Interpretation: rollout microbatch solved the GPU OOM class of failure but not host memory/process pressure for full `group_size=128` with current vector env layout and PBS memory.

## Comparison Table

| Run | Pop | lr | clip | log_std | euler noise | Save every | Train updates | Best held-out eval | 100ep confirmation | Runtime note |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| G8 | 8 | `1e-5` | `0.2` | `-2.0` default | `0.2` default | 5 | 100 | 48% @ u10/u15 (25ep) | 33% @ u10 | train 50-100 took 4h05m; 100ep eval 4m10s |
| G16 | 16 | `5e-6` | `0.2` | `-2.0` default | `0.2` default | 2 | 20 | 44% @ u10/u20 (25ep) | none | train total logs 1h57m |
| A | 32 | `5e-6` | `0.1` | `-2.0` default | `0.2` default | 2 | 30 | 45% @ u6 (20ep) | 30% @ u6 | initial slow, resume fast |
| R1 | 32 | `2e-6` | `0.05` | `-2.0` default | `0.2` default | 1 | 50 | 40% @ u15 (20ep) | 22% @ u8 | train 2h44m |
| R2 | 32 | `5e-6` | `0.1` | `-2.5` | `0.1` | 1 | 50 | 40% @ u18 (20ep) | 30% @ u18 | train 2h45m |
| R3 | 64 | `5e-6` | `0.1` | `-2.0` default | `0.2` default | 1 | failed after u0 | none | none | CUDA OOM |
| P128 smoke | 128 | `5e-6` | `0.1` | `-2.0` default | `0.2` default | 1 | 1 | none | none | memory-heavy, post-check bug |
| P128A/B/C | 128 | varied | varied | varied | varied | 1 | killed after u0 | none | none | host memory/process pressure |

## Source Ledger

### Existing Canvases / Prior Summaries

- `~/.cursor/projects/rds-general-user-aa6622-home/canvases/phase11-GRPO-run-report.canvas.tsx`
- `~/.cursor/projects/rds-general-user-aa6622-home/canvases/phase11-grpo-results.canvas.tsx`
- Prompt-mentioned `docs/findings/2026-05-18-phase11-g16-lr5e6-clip02-summary.md` is absent from current `docs/findings/`.

### Artifact Roots

- `artifacts/phase11_pushv3_chunk5_g8_vecasync_u100/`
- `artifacts/phase11_pushv3_chunk5_g16_lr5e6_clip02_u20/`
- `artifacts/phase11_pushv3_batched_logprob_smoke_u2/`
- `artifacts/phase11_pushv3_chunk5_g32_lr5e6_clip01_u30/`
- `artifacts/phase11_pushv3_chunk5_g32_lr2e6_clip005_u50/`
- `artifacts/phase11_pushv3_chunk5_g32_lr5e6_clip01_lownoise_u50/`
- `artifacts/phase11_pushv3_chunk5_g64_lr5e6_clip01_u50/`
- `artifacts/phase11_baseline_pushv3_100ep_s1000_nenv25_chunk5/`
- `artifacts/phase11_pushv3_pop128_rolloutpbs32_smoke_u1/`
- `artifacts/phase11_pushv3_chunk5_pop128_rpbs32_lr2e6_clip005_u50/`
- `artifacts/phase11_pushv3_chunk5_pop128_rpbs32_lr5e6_clip01_u50/`
- `artifacts/phase11_pushv3_chunk5_pop128_rpbs32_lr5e6_clip01_lownoise_u50/`

### PBS Scripts

- `scripts/grpo/phase11_g16_lr5e6_clip02_train_0000_0010.pbs`
- `scripts/grpo/phase11_g16_lr5e6_clip02_train_0010_0020_resume.pbs`
- `scripts/grpo/phase11_g16_lr5e6_clip02_eval_0002_0010.pbs`
- `scripts/grpo/phase11_g16_lr5e6_clip02_eval_0012_0020.pbs`
- `scripts/grpo/phase11_batched_logprob_smoke_u2.pbs`
- `scripts/grpo/phase11_A_g32_lr5e6_clip01_train_0000_0030.pbs`
- `scripts/grpo/phase11_A_g32_lr5e6_clip01_resume_train_0014_0030.pbs`
- `scripts/grpo/phase11_A_g32_lr5e6_clip01_eval_0002_0030.pbs`
- `scripts/grpo/phase11_g8_update0010_100ep_eval.pbs`
- `scripts/grpo/phase11_A_update0006_100ep_eval.pbs`
- `scripts/grpo/phase11_A_update0014_100ep_eval.pbs`
- `scripts/grpo/phase11_R1_g32_lr2e6_clip005_train_0001_0050.pbs`
- `scripts/grpo/phase11_R1_g32_lr2e6_clip005_eval_0001_0050.pbs`
- `scripts/grpo/phase11_R2_g32_lr5e6_clip01_lownoise_train_0001_0050.pbs`
- `scripts/grpo/phase11_R2_g32_lr5e6_clip01_lownoise_eval_0001_0050.pbs`
- `scripts/grpo/phase11_R3_g64_lr5e6_clip01_train_0001_0050.pbs`
- `scripts/grpo/phase11_R3_g64_lr5e6_clip01_eval_0001_0050.pbs`
- `scripts/grpo/phase11_baseline_pushv3_100ep_nenv25.pbs`
- `scripts/grpo/phase11_pop128_rolloutpbs32_smoke_u1.pbs`
- `scripts/grpo/phase11_P128A_lr2e6_clip005_train_0001_0050.pbs`
- `scripts/grpo/phase11_P128B_lr5e6_clip01_train_0001_0050.pbs`
- `scripts/grpo/phase11_P128C_lr5e6_clip01_lownoise_train_0001_0050.pbs`

### PBS Logs

- `logs/pbs/grpo/phase11_g8_update0010_100ep_eval.out`
- `logs/pbs/grpo/phase11_chunk5_eval_0005_0050.out`
- `logs/pbs/grpo/phase11_chunk5_train_0050_0100_resume.out`
- `logs/pbs/grpo/phase11_chunk5_eval_0005_0100.out`
- `logs/pbs/grpo/phase11_g16_lr5e6_clip02_train_0000_0010.out`
- `logs/pbs/grpo/phase11_g16_lr5e6_clip02_train_0010_0020_resume.out`
- `logs/pbs/grpo/phase11_g16_lr5e6_clip02_eval_0002_0010.out`
- `logs/pbs/grpo/phase11_g16_lr5e6_clip02_eval_0012_0020.out`
- `logs/pbs/grpo/phase11_batched_logprob_smoke_u2.out`
- `logs/pbs/grpo/phase11_A_g32_lr5e6_clip01_train_0000_0030.out`
- `logs/pbs/grpo/phase11_A_g32_lr5e6_clip01_resume_train_0014_0030.out`
- `logs/pbs/grpo/phase11_A_g32_lr5e6_clip01_eval_0002_0030.out`
- `logs/pbs/grpo/phase11_A_update0006_100ep_eval.out`
- `logs/pbs/grpo/phase11_A_update0014_100ep_eval.out`
- `logs/pbs/grpo/phase11_R1_g32_lr2e6_clip005_train_0001_0050.out`
- `logs/pbs/grpo/phase11_R1_g32_lr2e6_clip005_eval_0001_0050.out`
- `logs/pbs/grpo/phase11_R2_g32_lr5e6_clip01_lownoise_train_0001_0050.out`
- `logs/pbs/grpo/phase11_R2_g32_lr5e6_clip01_lownoise_eval_0001_0050.out`
- `logs/pbs/grpo/phase11_R3_g64_lr5e6_clip01_train_0001_0050.out`
- `logs/pbs/grpo/phase11_baseline_pushv3_100ep_nenv25.out`
- `logs/pbs/grpo/phase11_pop128_rolloutpbs32_smoke_u1.out`
- `logs/pbs/grpo/phase11_P128A_lr2e6_clip005_train_0001_0050.out`
- `logs/pbs/grpo/phase11_P128B_lr5e6_clip01_train_0001_0050.out`
- `logs/pbs/grpo/phase11_P128C_lr5e6_clip01_lownoise_train_0001_0050.out`

## Caveats

- Train rollout success is not held-out eval. It is success over that update's sampled GRPO group with train reset seed `2000 + update`.
- Eval success uses fixed eval seeds starting `1000`, independent from train seeds.
- 20/25 episode sweeps are noisy; 100 episode confirmations are stronger but still finite-sample.
- G8 train 0-50 PBS log/script is absent in current files, though progress/eval artifacts and job IDs remain. Use G8 `progress.jsonl` as source for training metrics.
- Prompt-mentioned G16 markdown summary is absent. This ledger uses existing G16 PBS scripts, logs, progress, and eval summaries instead.
- Pop128 production runs did not produce eval sweeps; only update 0 training artifacts exist.
