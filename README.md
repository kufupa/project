# push-v3 Data Pipeline (Task 2)

This directory now includes a minimal, runnable path for:

- running smolvla_metaworld baseline on push-v3
- generating a compact top-k optimal trajectory report
- exporting top trajectory videos to an artifact `trajectories` folder

## Scripts

- `scripts/pushv3_data_pipeline.sh`
  - runs `vendor/pi05/run_baseline_eval.sh` with the requested `--task`
  - parses `eval_info.json` and prints reward/success summary
  - writes `optimal_report.json`
  - creates `trajectories/export_manifest.json`
- `scripts/summarize_pushv3_eval.py`
  - reads `eval_info.json`
  - writes compact `optimal_report.json` with top-k episodes by `max_reward`
  - enforces strict matching on `--task` and fails with a clear error when task is missing
- `scripts/extract_parquet_episode_video.py`
  - copied from pi05 shared utility
  - optional renderer fallback when videos are not already present

## 1) Running baseline + automatic trajectory export

```bash
bash scripts/pushv3_data_pipeline.sh \
  --episodes 45 \
  --seed 123 \
  --task push-v3 \
  --checkpoint jadechoghari/smolvla_metaworld \
  --output-root /path/to/artifacts/phase06_baseline \
  --top-k 15 \
  --video true
```

Environment aliases (equivalent):

```bash
export PUSHV3_EPISODES=45
export PUSHV3_SEED=123
export PUSHV3_TASK=push-v3
export PUSHV3_CHECKPOINT=jadechoghari/smolvla_metaworld
export PUSHV3_OUTPUT_ROOT=/path/to/artifacts/phase06_baseline
export PUSHV3_TOP_K=15
bash scripts/pushv3_data_pipeline.sh
```

Current defaults are still:

- episodes: 15
- top-k: 5

For best-coverage selection in push-v3, use `--episodes 45 --top-k 15`.

Dry run:

```bash
bash scripts/pushv3_data_pipeline.sh --dry-run
```

## 2) Generating optimal trajectories (standalone)

```bash
python3 scripts/summarize_pushv3_eval.py \
  --eval-info /path/to/phase06_baseline/run_2026.../eval_info.json \
  --task push-v3 \
  --top-k 5 \
  --output /path/to/optimal_report.json
```

You can also pass optional explicit video paths:

```bash
python3 scripts/summarize_pushv3_eval.py \
  --eval-info /path/to/eval_info.json \
  --task push-v3 \
  --top-k 5 \
  --video-path "0:/abs/path/top0.mp4" \
  --video-path "1:/abs/path/top1.mp4" \
  --output /tmp/optimal_report.json
```

## 3) Exporting top videos

The pipeline exports top-k episodes automatically when it runs baseline:

```bash
bash scripts/pushv3_data_pipeline.sh --top-k 5
```

`--task` is forwarded into the baseline command as `--env.task`.

Run output directories are now unique per invocation, so re-running with the same
inputs creates a new run folder and never overwrites earlier outputs.

If baseline videos are not available, provide a dataset root and the extractor
fallback can still be used:

```bash
bash scripts/pushv3_data_pipeline.sh \
  --top-k 5 \
  --dataset-root /path/to/dataset \
  --source-episodes-root /path/to/episodes_pt_dir
```

When available, this writes:

- `run_.../optimal_report.json`
- `run_.../trajectories/trajectory_XX_episode_YYYY.mp4` files
- `run_.../trajectories/export_manifest.json`
