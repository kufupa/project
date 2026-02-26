# SmolVLA top-k full run handoff for fresh cluster clone

This document is for the next LLM/operator running the run on a different cluster copy of this repo.

Goal: run the same one-node 15-target Slurm command that currently fails on the previous cluster due QOS, now on a cluster with higher submit capacity.

Primary command that blocked on previous cluster:

```bash
SMOLVLA_TOPK_LAUNCH_MODE=sbatch sbatch scripts/smolvla/submit_smolvla_topk_full_one_node.slurm
```

## 1) Pre-flight assumptions that are not portable

These are the biggest path/env traps when cloning to another host:

- `scripts/smolvla/submit_smolvla_topk_full_one_node.slurm`
  - `ORACLE_RUN_DIR` fallback is hardcoded to:
    - `/vol/bitbucket/aa6622/project/artifacts/phase06_oracle_baseline/run_20260411T131839Z_ep60_voracle_tpush_v3_s1000_r402093`
  - `SMOLVLA_LEROBOT_ENV_DIR` fallback is hardcoded to:
    - `${WORKSPACE_ROOT}/.envs/lerobot_mw_py310`
  - `SMOLVLA_TOPK_LAUNCH_MODE` defaults to `dry-run`.
- `scripts/smolvla/run_oracle_topk_smolvla_full.sh`
  - `ORACLE_RUN_DIR` is consumed through env but not defaulted in this file.
  - `PROJECT_ROOT` is resolved from script location; this is robust to clone path as long as mount layout is valid.
- `scripts/smolvla/launch_pushv3_smolvla_topk15.sh`
  - `--oracle-run-dir` is required and validated at runtime.
  - If run in `sbatch` mode this script creates an array and submits `submit_pushv3_smolvla_topk15.slurm`.
  - Uses hardcoded `SMOLVLA_SBATCH_*` overrides only if provided.
- `scripts/smolvla/run_smolvla_target_episode.sh`
  - Uses `TASK_FIELDS` from `targets.json` rows and requires either CLI arg or `TASK_INDEX`/`SLURM_ARRAY_TASK_ID`.
  - Exports `SMOLVLA_TARGET_RUN_ROOT` (seed-based subdir when set), `SMOLVLA_TARGET_EPISODE_INDEX`, `SMOLVLA_FIXED_RESET_SEED`.
  - `CAMPAIGN_DIR` is optional; when present writes `task_0000.json` to campaign directory.
- `scripts/smolvla/submit_pushv3_smolvla_topk15.slurm`
  - This is an array worker script, not the one-node full run path.

## 2) Required explicit exports before sbatch on new cluster

Set these explicitly once per session before submitting:

```bash
export REPO_ROOT="$(cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)" && pwd)"
export PYTHON_BIN="/path/to/cluster/venv/bin/python"  # must be the LeRobot MW Python
export SMOLVLA_PYTHON_BIN="${PYTHON_BIN}"
export SMOLVLA_LEROBOT_ENV_DIR="/path/to/cluster/.envs/lerobot_mw_py310"
export ORACLE_RUN_DIR="/path/to/repo/artifacts/phase06_oracle_baseline/run_20260411T131839Z_ep60_voracle_tpush_v3_s1000_r402093"
```

Optional explicit stability overrides:

```bash
export SMOLVLA_ARTIFACT_ROOT="${REPO_ROOT}/artifacts"
export SMOLVLA_TOPK_OUTPUT_ROOT="${SMOLVLA_ARTIFACT_ROOT}/phase07_smolvla_baseline"
export EPISODES_PER_TARGET=4
export SMOLVLA_SAVE_FRAMES=false
```

If your oracle run ID changed on this clone, update `ORACLE_RUN_DIR` before any run.

## 3) Dry-run before the real run (strongly recommended)

This avoids wasting quota and catches broken env defaults.

```bash
cd "${REPO_ROOT}"
SMOLVLA_TOPK_LAUNCH_MODE=dry-run \
ORACLE_RUN_DIR="${ORACLE_RUN_DIR}" \
SMOLVLA_LEROBOT_ENV_DIR="${SMOLVLA_LEROBOT_ENV_DIR}" \
SMOLVLA_PYTHON_BIN="${SMOLVLA_PYTHON_BIN}" \
SMOLVLA_TARGET_MAX_STEPS=120 \
EPISODES_PER_TARGET=1 \
bash scripts/smolvla/run_oracle_topk_smolvla_full.sh
```

Expected success: campaign directory created with `targets.json`, one or more target loop artifacts, and `SMOLVLA_TOPK_LAUNCH_MODE` staying dry-run (no actual nested `sbatch` submissions from launcher).

## 4) Real run command for another operator

After dry-run passes and cluster quota is enough:

```bash
cd "${REPO_ROOT}"
SMOLVLA_TOPK_LAUNCH_MODE=sbatch \
ORACLE_RUN_DIR="${ORACLE_RUN_DIR}" \
SMOLVLA_LEROBOT_ENV_DIR="${SMOLVLA_LEROBOT_ENV_DIR}" \
SMOLVLA_PYTHON_BIN="${SMOLVLA_PYTHON_BIN}" \
EPISODES_PER_TARGET=4 \
SMOLVLA_SAVE_FRAMES=false \
sbatch scripts/smolvla/submit_smolvla_topk_full_one_node.slurm
```

## 5) Inconsistency checklist to preempt before rerun

- `scripts/smolvla/submit_pushv3_smolvla_topk15.slurm` is an array template; do not use it for this command unless explicitly doing per-target array mode.
- `submit_smolvla_topk_full_one_node.slurm` exports `SMOLVLA_TOPK_LAUNCH_MODE` default `dry-run`; if this remains unset the script will not submit nested run work.
- `run_oracle_topk_smolvla_full.sh` defaults `TARGET_RUN_ROOT` to campaign directory. This means latest layout is:
  - `<campaign_dir>/<seed>/` (instead of timestamped run_*).
  - `task_####.json` and `smolvla_topk_best_summary.json` are under campaign directory.
- `run_smolvla_target_episode.sh` and evaluator rely on `SMOLVLA_TARGET_EPISODE_INDEX` / `SMOLVLA_FIXED_RESET_SEED`; they are exported by `run_oracle_topk_smolvla_full.sh` and must stay aligned with `targets.json`.
- `run_smolvla_target_episode.sh` will call `xvfb-run` if available; on clusters without Xvfb this branch is skipped and plain python is used.

## 6) Quick sanity checks after submit

- `squeue -u "$USER"` for status.
- `tail -n 120 smolvla_topk_full_<JOBID>.out` for campaign progress.
- locate latest campaign:
  - `ls -dt "${REPO_ROOT}/artifacts/phase07_smolvla_baseline/campaigns/pushv3_smolvla_topk15_"* | head -1`
- ensure campaign manifest has 15 summaries after full success:
  ```bash
  CAMPAIGN_DIR="/path/to/campaign"
  python3 - <<'PY'
  import json
  import os
  from pathlib import Path
  path = Path(os.environ["CAMPAIGN_DIR"]) / "smolvla_topk_best_summary.json"
  print(len(json.loads(path.read_text(encoding="utf-8"))))
  PY
  ```
- inspect `task_0000.json` -> `run_dir` for run-manifest checks.

## 7) Known external blocker from previous environment (not code)

Previous cluster blocked with `QOSMaxSubmitJobPerUserLimit` during submit. If new cluster still blocks, treat as scheduler policy, not these scripts:
- check queue policy docs, max submits per user, concurrent GPU jobs, and account/QOS overrides with admin.
- keep retry spacing if needed; `--test-only` can pass while real submit fails.

## 8) Quick variable source-of-truth map (for fast handoff)

- `ORACLE_RUN_DIR` → source oracle trajectories and `targets.json`.
- `SMOLVLA_PYTHON_BIN` / `SMOLVLA_LEROBOT_ENV_DIR` → active Python + torch install.
- `SMOLVLA_TARGET_RUN_ROOT` → run folder root for each target (currently defaults to campaign root).
- `SMOLVLA_CAMPAIGN_DIR` → campaign-level summary files.
- `SMOLVLA_TOPK_LAUNCH_MODE` → `dry-run` or `sbatch` in full orchestrator.
- `EPISODES_PER_TARGET` → episodes in evaluator.
