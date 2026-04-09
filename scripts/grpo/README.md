# Phase11/12 SmolVLA GRPO

## LeRobot fork (required for true GRPO)

Safe-robot-steering pins `jsnchon/lerobot` at commit `f30fc2a1b904bb2ccd752cfff94f6f4423bd523b` for `select_action_distr_params` and `model.log_std`.

**IC / Phase11 default:** use **`/vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python`** with SmolVLA GRPO hooks patched in that venv’s `site-packages/lerobot` (see Phase11 plan amendment). Below is only for **greenfield** setups elsewhere:

```bash
# Example: fresh venv + install forked lerobot at pinned commit (not required on IC if py310 patch is already applied)
pip install -U pip
pip install "git+https://github.com/jsnchon/lerobot.git@f30fc2a1b904bb2ccd752cfff94f6f4423bd523b"
```

## API smoke (CPU)

```bash
cd /vol/bitbucket/aa6622/project
PYTHONPATH=src ./path/to/venv/bin/python scripts/grpo/check_lerobot_grpo_api.py
```

## Forward + backward smoke (GPU)

```bash
PYTHONPATH=src python scripts/grpo/check_smolvla_grpo_forward.py \
  --checkpoint /path/to/smolvla/checkpoint
```

## Train Phase11 (GPU)

```bash
PYTHONPATH=src python scripts/grpo/train_phase11_env_on_policy_grpo.py \
  --checkpoint /path/to/smolvla/checkpoint \
  --output-dir artifacts/phase11_grpo/run_xxx \
  --train-seed-base 2000 \
  --start-update 0 \
  --num-updates 1
```

Slurm: see `scripts/grpo/submit_phase11_grpo.slurm` and `submit_phase11_chain.sh`.
