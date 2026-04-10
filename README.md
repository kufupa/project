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

## SmolVLA Baseline

Active scripts:

- `scripts/smolvla/pushv3_smolvla_smoke.sh` - single-episode smoke run and writes `smoke_summary.json`
- `scripts/smolvla/run_metaworld_smolvla_eval.py` - SmolVLA evaluator entrypoint for Meta-World tasks
- `scripts/smolvla/run_pushv3_smolvla_preflight_1ep.sh` - single-episode GPU preflight run + artifact verification
- `scripts/smolvla/submit_pushv3_smolvla_preflight_1ep.slurm` - Slurm template for one-episode GPU preflight
- `scripts/smolvla/launch_pushv3_smolvla_topk15.sh` - prepares/submits top-k campaign runs from an oracle run
- `scripts/smolvla/run_pushv3_smolvla_parity_benchmark.sh` - 15-episode parity benchmark (`seed=123`, `max_steps=120`)
- `scripts/smolvla/compare_eval_info.py` - baseline vs candidate `eval_info.json` metric diff helper
- `scripts/smolvla/verify_smolvla_run_artifacts.py` - strict artifact validator for smoke/preflight gates

Artifact root:

- `artifacts/phase07_smolvla_baseline`

Run naming contract:

- `run_{UTC}_ep{episodes}_vsmolvla_t{task}_s{seed}_r{nonce}`

Per-run files:

| Path | Purpose |
|------|---------|
| `eval_info.json` | Aggregated metrics including `overall` and `video_paths` |
| `run_manifest.json` | Run config, checkpoint metadata, and per-episode artifact index |
| `smoke_summary.json` | Smoke-only status summary emitted by `pushv3_smolvla_smoke.sh` |
| `videos/<task>_0/eval_episode_<i>.mp4` | Episode videos when `--video true` |
| `episodes/episode_<i>/actions.jsonl` | Per-step action/reward/success rows |
| `episodes/episode_<i>/reward_curve.csv` | Step-wise reward and cumulative reward values |
| `episodes/episode_<i>/reward_curve.png` | Reward curve plot (or fallback placeholder image) |

### SmolVLA preflight before top-k

Run one GPU preflight episode (does not queue the top-k array):

```bash
bash scripts/smolvla/run_pushv3_smolvla_preflight_1ep.sh
```

This run must produce:

- `eval_info.json`
- `run_manifest.json`
- `episodes/episode_0000/actions.jsonl`
- `videos/push-v3_0/eval_episode_0000.mp4` (non-trivial size)

Parity/fast gate knobs used by SmolVLA scripts:

- `SMOLVLA_EVAL_MODE=parity|fast` (default `parity`)
- `SMOLVLA_METAWORLD_CAMERA_NAME` (default `corner2`)
- `SMOLVLA_FLIP_CORNER2` (default `true`)
- `SMOLVLA_LOAD_VLM_WEIGHTS` (defaults to `true` in parity mode)
- `SMOLVLA_PREFLIGHT_MAX_STEPS`, `SMOLVLA_SMOKE_MAX_STEPS`, `SMOLVLA_TARGET_MAX_STEPS`

Optional Slurm submission for the same preflight:

```bash
sbatch scripts/smolvla/submit_pushv3_smolvla_preflight_1ep.slurm
```

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
