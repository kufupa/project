# Slurm SBATCH notes for SmolVLA top-k launch

This note captures the current Slurm workflow and the mode interaction that previously caused `QOSMaxSubmitJobPerUserLimit`.

## Why nested sbatch failed

`run_oracle_topk_smolvla_full.sh` used to always honor:

`SMOLVLA_TOPK_LAUNCH_MODE=sbatch`

When a job submitted with `submit_smolvla_topk_full_one_node.slurm` also ran with `LAUNCH_MODE=sbatch`, the script executed:

- campaign creation (`launch_pushv3_smolvla_topk15.sh`)
- inner `sbatch --array ...` submission inside the already running batch job

The inner submission can be rejected by cluster policy before any task starts:

`QOSMaxSubmitJobPerUserLimit`
`Batch job submission failed: Job violates accounting/QOS policy (job submit limit, user's size and/or time limits)`

This is a scheduler-side submission limit, not an evaluation/runner failure.

## Current safe behavior

`run_oracle_topk_smolvla_full.sh` now auto-adjusts when running inside Slurm:

- Detects active allocation with `SLURM_JOB_ID`
- Uses `dry-run` mode by default to avoid another inner `sbatch`
- Runs campaign setup + per-target evaluation in the same allocation
- Logs resolved mode:
  - `smolvla-topk: launch_mode=<...> slurm_job_id=<...> allow_nested_sbatch=<0|1>`

`SMOLVLA_ALLOW_NESTED_SBATCH` can override this default.

## Recommended commands

### Use safe path (recommended, avoids nested submit):

```bash
SMOLVLA_TOPK_LAUNCH_MODE=dry-run sbatch scripts/smolvla/submit_smolvla_topk_full_one_node.slurm
```

You can also omit `SMOLVLA_TOPK_LAUNCH_MODE` since the script defaults to `dry-run` if unset.

### Opt-in nested-submission mode (explicit):

Use this only when you intentionally want inner array behavior and know your QOS submit budget:

```bash
SMOLVLA_TOPK_LAUNCH_MODE=sbatch \
SMOLVLA_ALLOW_NESTED_SBATCH=1 \
sbatch scripts/smolvla/submit_smolvla_topk_full_one_node.slurm
```

## Command patterns for monitoring

```bash
squeue -j <JOBID>
scontrol show job <JOBID>
tail -f smolvla_topk_full_<JOBID>.out
tail -f smolvla_topk_full_<JOBID>.err
```

Expected healthy startup markers:

- No `QOSMaxSubmitJobPerUserLimit` messages
- Campaign folder appears under `artifacts/phase07_smolvla_baseline/campaigns/`
- `smolvla_topk_best_summary.json` is created and populated

## Segment GRPO top-15 (chunk30 / K=5)

The launcher [`scripts/segment_grpo/submit_segment_grpo_topk15_chunk30_max30_k5_array.slurm`](../../scripts/segment_grpo/submit_segment_grpo_topk15_chunk30_max30_k5_array.slurm) is a normal array job (one `sbatch` from the login node). It does **not** call `sbatch` again from inside the job body.

If you still see `QOSMaxSubmitJobPerUserLimit` or submission failures, common causes match the SmolVLA case above: **submitting `sbatch` from inside another Slurm allocation** (nested submit), or hitting per-user submit caps when something upstream wraps your command in batch submission.

**Recommended:**

- From the **login** node, use **one** submit for all remaining targets without an inner queue:
  - [`scripts/segment_grpo/submit_segment_grpo_topk15_chunk30_max30_k5_serial.slurm`](../../scripts/segment_grpo/submit_segment_grpo_topk15_chunk30_max30_k5_serial.slurm) — one GPU job, sequential loop over ranks 2–15 (longer wall time).
- If you already have a GPU shell (`salloc` / `srun` / interactive node), avoid `sbatch` entirely:
  - `bash scripts/segment_grpo/segment_grpo_topk15_chunk30_max30_k5_run_all_local.sh`

Shared per-target logic lives in [`scripts/segment_grpo/segment_grpo_topk15_chunk30_max30_k5_run_task.sh`](../../scripts/segment_grpo/segment_grpo_topk15_chunk30_max30_k5_run_task.sh).

Each target’s `--goal-frame-index 25` is loaded from that row’s `frames/episode_{episode_index}/` under the resolved oracle baseline (same multi-episode run directory for all top-15 rows; different episode subfolder per seed). `run_segment_grpo.py` asserts the goal PNG path matches the requested episode and frame index.

Slurm may execute a **copy** of the batch script from `/var/spool/slurm/...`, so segment GRPO Slurm entrypoints use a fixed `PROJECT_ROOT` and `${PROJECT_ROOT}/scripts/segment_grpo/` for helper shells (do not resolve helper paths from `BASH_SOURCE` on the staged copy).

## segGRPO all-60 frame-50 k=3 (2026-04-16, subprocess orchestrator)

- Entry point: [`scripts/segment_grpo/run_all60_frame50_k3.py`](../../scripts/segment_grpo/run_all60_frame50_k3.py) (campaign orchestrator).
- Submit: `sbatch scripts/segment_grpo/submit_segment_grpo_all60_frame50_k3.slurm`.
- Local debug: `bash scripts/segment_grpo/run_all60_frame50_k3_local.sh` (override `EPISODES=2` for a smoke).
- Design: one subprocess per episode invoking `scripts/run_segment_grpo.py` with `--episodes 1 --episode-index i --reset-seed 1000+i --flat-output`. Memory isolation is real (fresh Python process each time).
- No edits to existing Python entrypoints: `run_segment_grpo.py`, `segment_grpo_loop.py`, `segment_grpo_reference.py`, and existing tests stay unchanged.
- Artifacts under `artifacts/phase08_segment_grpo_baseline/run_<UTC>_all60_f50_k3_s1000/` (auto `run_name` unless `--run-name` is passed):
  - `out_episode_NNNN.json` x 60 (full `EpisodeLog.to_dict()`, including `latent_scores`, `selected_scores`, `candidate_distances`, per-candidate `latent_distance` inside segment payloads).
  - Comparison-strip PNGs under each `out_episode_NNNN_artifacts/` when `--comparison-strip-overlay` is on.
  - `episodes.jsonl`: append-only orchestrator summary with latent-distance fields copied from each child JSON for quick post-processing.
  - `skipped_episodes.jsonl`: rows for any episode whose oracle goal PNG was missing under `frames/episode_NNNN/frame_000049.png` (same layout as `load_oracle_reference_frames`; expected empty for this campaign).
  - `failed_episodes.jsonl`: rows for any subprocess with non-zero exit or timeout (stderr tail included).
  - `segment_grpo_manifest.json`: final rollup with `counts` + `outcomes`.
- Resume: re-run with the same `--run-name`; episodes whose `out_episode_NNNN.json` already exist are skipped (`RESUME-SKIP` in log).
- Skip-on-missing-goal: orchestrator checks for the oracle goal frame before spawning; no child process starts for missing frames.
- Latent-distance JSON contract: unchanged in `segment_grpo_loop.py` dataclasses; orchestrator only reads child output.

## Quick troubleshooting

- If launch still fails, capture:
  - exact submission command
  - `squeue -j <JOBID>` and `scontrol show job <JOBID>`
  - `smolvla_topk_full_<JOBID>.out/.err`
- Re-check with cluster admin if `QOSMaxSubmitJobPerUserLimit` persists after the guard is in place.
