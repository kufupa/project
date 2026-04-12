# SmolVLA Top-K Full Run: Current State & Handoff Notes

Last updated: 2026-04-12

This document captures the current SmolVLA top-k campaign pipeline state for someone
continuing without this chat context.

## What this pipeline does

Goal: run a single Slurm job that evaluates oracle-selected Push-T push tasks
(`top_k=15`) with SmolVLA, generate 4 episodes per target, pick the best episode,
and emit campaign-level summaries + artifacts.

High-level flow:

1. Create campaign + targets list from oracle baseline output.
2. Loop over each target in the campaign.
3. For each target:
   - run SmolVLA evaluation over `EPISODES_PER_TARGET` episodes.
   - collect run artifacts and verify them.
   - extract best episode and copy `best_episode.mp4`.
4. Append best-episode summary to campaign aggregate.

## Files that implement the flow

- `project/scripts/smolvla/run_oracle_topk_smolvla_full.sh`
  - Main orchestrator for full one-node top-k campaign.
  - Creates campaign, loops all targets, invokes per-target script, aggregates summaries.
- `project/scripts/smolvla/launch_pushv3_smolvla_topk15.sh`
  - Builds campaign directory and `targets.json`.
- `project/scripts/smolvla/submit_smolvla_topk_full_one_node.slurm`
  - Slurm entrypoint used by users.
  - Sets env defaults and calls `run_oracle_topk_smolvla_full.sh`.
- `project/scripts/smolvla/run_smolvla_target_episode.sh`
  - Runs one target row from `targets.json`, calls `run_metaworld_smolvla_eval.py`.
- `project/src/smolvla_pipeline/evaluator.py`
  - Enforces episode/task selection and reset behavior via env overrides.
- `project/tests/test_smolvla_eval_artifacts.py`
  - Regression tests for artifact generation + fixed-seed behavior.

## Why output structure changed (last major logic change)

You requested fewer folders and clearer grouping by seed.
Current behavior now is:

- Campaign run root still holds campaign-level files.
- Each target run now uses a seed-based directory under campaign:
  - `${SMOLVLA_TARGET_RUN_ROOT}/${seed}/`
  - default target root currently resolves to campaign directory by default.
- This gives one folder per seed and groups all runs for that seed together.

### Mechanism added

- `run_oracle_topk_smolvla_full.sh`
  - sets `SMOLVLA_TARGET_RUN_ROOT="${CAMPAIGN_DIR}"` by default.
- `run_smolvla_target_episode.sh`
  - now respects `SMOLVLA_TARGET_RUN_ROOT`.
  - if set, uses `RUN_DIR="${SMOLVLA_TARGET_RUN_ROOT}/${SEED}"` (instead of random timestamped `run_*` name).
  - still creates that folder if needed (`mkdir -p`).

## Bug fixes completed recently

### 1) Environment reset consistency bug (critical)

Problem seen:
- same oracle target appeared with different goal/objective states across 4 episodes.

Fixes:
- In `evaluator.py`:
  - Added optional env override `SMOLVLA_TARGET_EPISODE_INDEX`.
  - Added optional env override `SMOLVLA_FIXED_RESET_SEED`.
  - For each run, task selection uses override episode index.
  - Reset seed uses override per episode.
- In `run_smolvla_target_episode.sh`:
  - Reads target row from `targets.json` and exports:
    - `SMOLVLA_TARGET_EPISODE_INDEX=${EPISODE_INDEX}`
    - `SMOLVLA_FIXED_RESET_SEED=${SEED}`
  - Passes `--episodes "${EPISODES_PER_TARGET}"` and `--seed "${SEED}"` correctly.
- Test added:
  - `test_smolvla_eval_artifacts.py` now validates fixed reset seed is used across episodes.

### 2) Single top-k campaign submission scaffolding

- Added/kept one-node entrypoint `submit_smolvla_topk_full_one_node.slurm`.
- Added Slurm-root fallback handling so scripts still locate project root when Slurm stages scripts.
- Uses:
  - `SMOLVLA_TOPK_LAUNCH_MODE=sbatch`
  - `SMOLVLA_TOPK_LAUNCH_MODE=dry-run` for validation mode.

## Current environment in practice

Working directories of interest:
- Artifacts: `project/artifacts/`
- Campaign outputs: `project/artifacts/phase07_smolvla_baseline/campaigns/...`

Recent submission attempts:
- `230465` (earlier)
- `230467` (failed at submission: `QOSMaxSubmitJobPerUserLimit`)
- `230487` (same failure)

Latest observed state:
- Job enters briefly then fails quickly during nested Slurm call.
- `squeue -u aa6622` is empty when checked after failures.
- Error in script output:
  - `QOSMaxSubmitJobPerUserLimit`
  - `Batch job submission failed: Job violates accounting/QOS policy (job submit limit, user's size and/or time limits)`

Important:
- This is a cluster policy/QOS issue, not a code logic regression in the SmolVLA runner.
- Same script can pass `sbatch --test-only` in some checks, so quota checks are timing/policy/context sensitive.

## Commands that were used repeatedly for status

- Submit:
  - `SMOLVLA_TOPK_LAUNCH_MODE=sbatch sbatch scripts/smolvla/submit_smolvla_topk_full_one_node.slurm`
- Queue check:
  - `squeue -u aa6622`
  - `squeue -j <JOBID>`
- Show failed job:
  - `scontrol show job <JOBID>`
- Tail output:
  - `tail -n 80 smolvla_topk_full_<JOBID>.out`
- Slurm qos/partition check (available today):
  - `scontrol show partition=t4`

## What to do next (recommended plan)

1. Confirm cluster-side policy with admin:
   - ask for exact `QOSMaxSubmitJobPerUserLimit` value on `multigpu` / user `aa6622`.
   - ask whether previous failed/active jobs are counting toward submit caps.
2. Retry once policy is cleared/reset:
   - prefer short controlled run first.
3. If needed, add explicit pacing/backoff/retry in launcher (between `sbatch` calls) instead of immediate retry loops.
4. When first full run is accepted, monitor:
   - job log for per-target output folder creation
   - generated `best_episode.mp4`, `run_manifest.json`, and campaign summary.

## Known caveats

- Slurm accounting DB (`sacctmgr`) appears intermittently unreachable from this login shell:
  - connection refused to `localhost:6819`.
- Some old command variants (e.g., some `scontrol`/`sacctmgr` forms) are not available on this cluster build.
- The queue limit/error behavior may differ between `sbatch` direct submit and `--test-only`.

## Planned continuation (short version)

The code is mostly aligned for the correct run semantics.
Only operational blocker remains: submit-side QOS/submit-cap policy.
Once that is lifted, we should be able to run the full 15-target campaign end-to-end. 
