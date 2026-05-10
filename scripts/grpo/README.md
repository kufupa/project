# Phase111 SmolVLA GRPO

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

## Phase111 official LeRobot MetaWorld backend

Phase111 is new SmolVLA/MetaWorld GRPO implementation. It uses `--env-backend official_lerobot` so env setup, reset seeding, success termination, corner2 image correction, and horizon match official LeRobot `MetaworldEnv`.

Phase11 `custom` backend is retired legacy/reference material. Keep it runnable only to reproduce old artifacts or explain historical drift; do not use it for new training claims.

For single-task GRPO training, cap rollouts at 120 steps by default. Full official v3 horizon is 500 steps, but most failed `push-v3` rollouts run to truncation, making updates about 4x slower than legacy Phase11. The single-task Slurm runner therefore defaults `GRPO_PHASE111_MAX_STEPS=120` while still using the official LeRobot backend/preprocessing. Use `GRPO_PHASE111_MAX_STEPS=0` only for deliberate full-horizon training/evaluation.

Training logs report both signals:

- `avg_return` / `returns`: dense MetaWorld reward used for GRPO advantages.
- `success_rate` / `successes`: binary benchmark success used to judge task completion.

Smoke rollout:

```bash
cd /vol/bitbucket/aa6622/project
PYTHONPATH=src /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python scripts/grpo/smoke_phase11_rollout.py \
  --checkpoint /vol/bitbucket/aa6622/.cache/huggingface/hub/models--jadechoghari--smolvla_metaworld/snapshots/ef3089ecb84eeeb7d33fedab24f6c76180a68900 \
  --task assembly-v3 \
  --env-backend official_lerobot \
  --group-size 2 \
  --max-steps 0 \
  --seed 2000
```

One-update train smoke:

```bash
sbatch --chdir=/vol/bitbucket/aa6622/project --export=NIL scripts/grpo/submit_phase111_on_grpo_lerobot_smoke.slurm
```

Full trainer entrypoint remains `scripts/grpo/train_phase11_env_on_policy_grpo.py`; use `--env-backend official_lerobot --max-steps 120` for default single-task GRPO training, and reserve `--max-steps 0` for intentional full-horizon probes/evaluation.
