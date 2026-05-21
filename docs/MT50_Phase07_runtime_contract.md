# MT50 Phase07 Runtime Contract

Generated for this clone.

## Root variables

- `REPO_ROOT=/rds/general/user/aa6622/home/project`
- `WORKSPACE_ROOT=/rds/general/user/aa6622/home`
- `SMOLVLA_LEROBOT_ENV_DIR=/rds/general/user/aa6622/home/.envs/lerobot_mw_py312`
- `SMOLVLA_PYTHON_BIN=/rds/general/user/aa6622/home/.envs/lerobot_mw_py312/bin/python`
- `HF_HOME=/rds/general/user/aa6622/home/project/.cache/huggingface`
- `HUGGINGFACE_HUB_CACHE=/rds/general/user/aa6622/home/project/.cache/huggingface`
- `HF_DATASETS_CACHE=/rds/general/user/aa6622/home/project/.cache/huggingface/datasets`
- `XDG_CACHE_HOME=/rds/general/user/aa6622/home/project/.cache`
- `TORCH_HOME=/rds/general/user/aa6622/home/project/.cache/torch`
- `MT50_PHASE07_OUTPUT_ROOT=/rds/general/user/aa6622/home/project/artifacts/MT50_Phase07_500`

## Bootstrap (copy/paste)

```bash
export REPO_ROOT="/rds/general/user/aa6622/home/project"
export WORKSPACE_ROOT="/rds/general/user/aa6622/home"
export SMOLVLA_LEROBOT_ENV_DIR="/rds/general/user/aa6622/home/.envs/lerobot_mw_py312"
export SMOLVLA_PYTHON_BIN="/rds/general/user/aa6622/home/.envs/lerobot_mw_py312/bin/python"
export HF_HOME="/rds/general/user/aa6622/home/project/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="/rds/general/user/aa6622/home/project/.cache/huggingface"
export HF_DATASETS_CACHE="/rds/general/user/aa6622/home/project/.cache/huggingface/datasets"
export XDG_CACHE_HOME="/rds/general/user/aa6622/home/project/.cache"
export TORCH_HOME="/rds/general/user/aa6622/home/project/.cache/torch"
export MT50_PHASE07_OUTPUT_ROOT="/rds/general/user/aa6622/home/project/artifacts/MT50_Phase07_500"
export MT50_PHASE07_CHECKPOINT="jadechoghari/smolvla_metaworld"
export MT50_PHASE07_EPISODES=1
export MT50_PHASE07_SEED=1000
export MT50_PHASE07_MAX_STEPS=500
export MT50_PHASE07_VIDEO=false
```

## Cache + env sanity check commands

```bash
# Create cache dirs used by hf/torch
mkdir -p "/rds/general/user/aa6622/home/project/.cache/huggingface/hub" \
  "/rds/general/user/aa6622/home/project/.cache/huggingface/datasets" \
  "/rds/general/user/aa6622/home/project/.cache/torch"

# Confirm env binary and key imports
"${SMOLVLA_PYTHON_BIN}" -V
"${SMOLVLA_PYTHON_BIN}" -m pip check
```

## Readiness result (as executed)

- Venv created at `/rds/general/user/aa6622/home/.envs/lerobot_mw_py312`
- Cache dirs created under `/rds/general/user/aa6622/home/project/.cache`
- Core sim deps installed (`metaworld`, `gymnasium`, `mujoco`) under **Python 3.12** (`lerobot_mw_py312`)
- CX3 batch / headless: use **`MUJOCO_GL=glfw`** and run under **`xvfb-run`** (see `run_mt50_phase07_10ep_no_video.slurm` / `.pbs`); EGL/OSMesa often break on compute nodes
- `lerobot` + deps: use `project/requirements-smolvla-lock.txt` (includes **`transformers==5.3.0`** to avoid `GR00TN15Config` import crash with newer transformers)
- `huggingface_hub` installed and warm cache validated with:
  `google-bert/bert-base-uncased` → `${HF_HOME}/models--google-bert--bert-base-uncased/snapshots/.../config.json`
- Resolver check works:
  - `resolve_hf_hub_repo_to_local_snapshot('google-bert/bert-base-uncased')` returns local snapshot path
- `jadechoghari/smolvla_metaworld` checkpoint preloaded to cache:
  - `/rds/general/user/aa6622/home/project/.cache/huggingface/models--jadechoghari--smolvla_metaworld/snapshots/ef3089ecb84eeeb7d33fedab24f6c76180a68900`
