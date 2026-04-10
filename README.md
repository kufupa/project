# push-v3 Oracle Data Pipeline

The active pipeline uses the Meta-World scripted policy (`metaworld.policies.ENV_POLICY_MAP`), not SmolVLA, for simulator rollouts on `push-v3`.

## Active Scripts (Oracle)

- `scripts/oracle/pushv3_oracle_data_pipeline.sh` — oracle eval, `optimal_report.json`, top-k video export, updates `run_manifest.json` with pipeline paths
- `scripts/oracle/run_oracle_baseline_eval.sh` — unique run directory + `xvfb-run` wrapper
- `scripts/oracle/run_metaworld_oracle_eval.py` — rollouts, per-step `actions.jsonl`, per-frame PNGs, `episode_meta.json`, `run_manifest.json`

## Legacy Scripts (SmolVLA-era)

- `scripts/legacy_pushv3_data_pipeline_smolvla.sh`
- `vendor/pi05/run_baseline_eval_legacy_smolvla.sh`
- `scripts/legacy_lerobot_eval_full_videos.py`

## Default artifact root

Runs go under the **project** directory:

- `<project_root>/artifacts/phase06_oracle_baseline/`

Example (this repo): `/vol/bitbucket/aa6622/project/artifacts/phase06_oracle_baseline/`

Override with `--output-root` or `PUSHV3_OUTPUT_ROOT` / `ORACLE_ARTIFACT_ROOT`.

## Run directory naming

Each run is a single child folder:

- `run_{UTC}_ep{episodes}_voracle_t{task}_s{seed}_r{nonce}`

Example: `run_20260410T130404Z_ep45_voracle_tpush_v3_s1000_r508367`

## Per-run layout

Inside each run folder:

| Path | Purpose |
|------|---------|
| `eval_info.json` | Aggregated rewards, success, video paths (summarizer input) |
| `run_manifest.json` | Run config + per-episode artifact index; `pipeline` block added after export |
| `videos/<task>_0/eval_episode_<i>.mp4` | Full-episode video (if `--video true`) |
| `episodes/episode_<i>/actions.jsonl` | One JSON line per step: `step`, `action`, `reward`, `terminated`, `truncated`, `success` |
| `episodes/episode_<i>/episode_meta.json` | Episode summary + relative paths to artifacts |
| `frames/episode_<i>/frame_<t>.png` | RGB frames (6-digit `t`; includes post-reset frame, then after each step) |
| `optimal_report.json` | Top-k episodes by `max_reward` |
| `trajectories/trajectory_<rank>_episode_<i>.mp4` | Copied top-k videos |
| `trajectories/export_manifest.json` | Export status; includes `run_manifest` path when present |

Disable per-frame PNGs (smaller disk): set `ORACLE_SAVE_FRAMES=false` or pass `--save-frames false` to `run_oracle_baseline_eval.sh` (pipeline would need a forward flag if you want it from the main script only; today the eval is invoked via that runner).

## Run Oracle Pipeline

```bash
bash scripts/oracle/pushv3_oracle_data_pipeline.sh \
  --episodes 45 \
  --seed 1000 \
  --task push-v3 \
  --output-root /vol/bitbucket/aa6622/project/artifacts/phase06_oracle_baseline \
  --top-k 15 \
  --video true
```

Defaults: episodes `15`, top-k `5`. Recommended trajectory bank: `--episodes 45 --top-k 15 --seed 1000`.

Dry run:

```bash
bash scripts/oracle/pushv3_oracle_data_pipeline.sh --dry-run
```

## Optional extractor fallback

```bash
bash scripts/oracle/pushv3_oracle_data_pipeline.sh \
  --top-k 5 \
  --dataset-root /path/to/dataset \
  --source-episodes-root /path/to/episodes_pt_dir
```

## Disk note

`45` episodes × `400` max steps × PNGs per timestep is large. Use shorter `--episode-length` for experiments, or `ORACLE_SAVE_FRAMES=false` when you only need video + `actions.jsonl`.
