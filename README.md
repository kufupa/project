# SmolVLA Meta-World Research Stack

Training and evaluation for **SmolVLA** on **Meta-World** (LeRobot backend). Core code lives in `src/smolvla_grpo/` and `src/smolvla_pipeline/`. Runs write under `artifacts/`.

## Dependencies

This repo expects a **forked LeRobot** with SmolVLA GRPO hooks (`select_action_distr_params`, trainable `log_std`):

- Fork: [jsnchon/lerobot](https://github.com/jsnchon/lerobot) (Thomas Deng)
- Pin: [`f30fc2a`](https://github.com/jsnchon/lerobot/commit/f30fc2a1b904bb2ccd752cfff94f6f4423bd523b)

### Python environment

From repo root (`project/`):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel setuptools
pip install -r requirements-smolvla-lock.txt
pip install "git+https://github.com/jsnchon/lerobot.git@f30fc2a1b904bb2ccd752cfff94f6f4423bd523b"
```

On the IC cluster, a pre-built venv is often used instead:

```bash
export GRPO_PYTHON="${GRPO_PYTHON:-/vol/bitbucket/aa6622/.envs/lerobot_mw_py312/bin/python}"
```

Verify GRPO API (CPU):

```bash
cd project
PYTHONPATH=src "${GRPO_PYTHON}" scripts/grpo/check_lerobot_grpo_api.py
```

GPU forward smoke:

```bash
PYTHONPATH=src "${GRPO_PYTHON}" scripts/grpo/check_smolvla_grpo_forward.py \
  --checkpoint jadechoghari/smolvla_metaworld
```

Default checkpoint: `jadechoghari/smolvla_metaworld`. Override with `--checkpoint` or `GRPO_PHASE111_CHECKPOINT`.

Slurm jobs source `scripts/slurm/common_env.sh` for `PROJECT_ROOT`, `PYTHONPATH`, and HF/torch caches under `<workspace>/.cache/`.

## Swapping Meta-World tasks

Most scripts take `--task <name-v3>` (e.g. `assembly-v3`, `button-press-topdown-v3`, `reach-v3`).

| Context | How to set task |
|---------|-----------------|
| Python trainers / eval | `--task assembly-v3` |
| Slurm GRPO | `GRPO_PHASE111_TASK=assembly-v3` or positional arg to `submit_phase111_single_task_grpo.slurm` |
| Oracle eval | `ORACLE_BASELINE_TASK=reach-v3` or `--task` on pipeline script |
| SmolVLA smoke/eval | `--task` on `run_metaworld_smolvla_eval.py` |
| MT10 canonical list | `scripts/mt10/mt10_tasks.sh` |

MT1 scripted oracle policies and MT10/MT50 multi-task sweeps use the task lists in `scripts/mt10/` and `scripts/mt50/`.

## Active pipelines

### 1. Environment GRPO (Phase 11 / 111)

On-policy GRPO in the real Meta-World environment via LeRobot's `MetaworldEnv`.

**Trainer:** `scripts/grpo/train_phase11_env_on_policy_grpo.py`

```bash
PYTHONPATH=src "${GRPO_PYTHON}" scripts/grpo/train_phase11_env_on_policy_grpo.py \
  --checkpoint jadechoghari/smolvla_metaworld \
  --output-dir artifacts/phase111_grpo/run_local \
  --task assembly-v3 \
  --env-backend official_lerobot \
  --rollout-execution vector_async \
  --group-size 4 \
  --num-updates 10 \
  --max-steps 120
```

**Slurm (single-task):** `scripts/grpo/submit_phase111_single_task_grpo.slurm`

**Flow-SDE chunk GRPO** (default for chunk rollouts): add `--rollout-unit chunk --logprob-mode flow_sde`. See `scripts/grpo/submit_flow_sde_chunk_grpo_smoke_a30.slurm`.

**DGPO** (distribution-guided advantage redistribution): same trainer with `--dgpo` (+ optional `--dgpo-tau`, `--dgpo-kappa`). Some DGPO and Flow-SDE Slurm jobs source **RLinf** for eval (`RLINF_ROOT` → `/vol/bitbucket/aa6622/RLinf-smolvla-metaworld-ppo-grpo`, e.g. `scripts/grpo/submit_dgpo_chunk_grpo_smoke_a30.slurm`, several `submit_flow_sde_*` jobs).

Rollout smoke (no full train): `scripts/grpo/smoke_phase11_rollout.py`

### 2. World-model GRPO (Phase 12)

Chunk GRPO scored with a **JEPA world model** instead of (or alongside) dense env reward.

**Trainer:** `scripts/grpo/train_phase12_wm_chunk_grpo.py`

```bash
PYTHONPATH=src "${GRPO_PYTHON}" scripts/grpo/train_phase12_wm_chunk_grpo.py \
  --mode wm_grpo_train \
  --task button-press-topdown-v3 \
  --checkpoint jadechoghari/smolvla_metaworld \
  --jepa-ckpt /path/to/jepa_wm_metaworld.pth.tar \
  --output-dir artifacts/phase12_wm_chunk_grpo/run_local \
  --num-updates 10
```

Key modes: `--phase12-train-mode selected_env|wm_only`, `--action-profile official_jepa_mirror|bounded_executed`.

**Slurm smoke:** `scripts/grpo/submit_phase12_wm_chunk_grpo_smoke.slurm`

### 3. EGGROLL

Evolutionary / population-based SmolVLA fine-tuning (low-rank perturbations on action expert).

**Trainer:** `scripts/eggroll/train_smolvla_eggroll.py`

```bash
PYTHONPATH=src "${GRPO_PYTHON}" scripts/eggroll/train_smolvla_eggroll.py \
  --checkpoint jadechoghari/smolvla_metaworld \
  --output-dir artifacts/eggroll/run_local \
  --task assembly-v3 \
  --population-size 32 \
  --num-iterations 100
```

Queue helpers: `scripts/eggroll/queue_phase50_eggroll_*.sh` (PBS-oriented).

### 4. SmolVLA evaluation (no training)

Baseline rollouts, videos, `eval_info.json` / `run_manifest.json`.

- **Entry:** `scripts/smolvla/run_metaworld_smolvla_eval.py`
- **Smoke:** `scripts/smolvla/pushv3_smolvla_smoke.sh` (set task via eval script or env)
- **Artifact root:** `artifacts/phase07_smolvla_baseline/`

### 5. Oracle scripted baselines

Meta-World **scripted policies** (`metaworld.policies.ENV_POLICY_MAP`) for reference rollouts and top-k trajectory export. MT1-only.

- `scripts/oracle/run_metaworld_oracle_eval.py`
- `scripts/oracle/run_oracle_baseline_eval.sh`
- `scripts/oracle/pushv3_oracle_data_pipeline.sh` (eval + top-k export; pass `--task`)
- **Artifact root:** `artifacts/phase06_oracle_baseline/`

## Artifacts

Runs land under `artifacts/<phase>_<name>/` with naming like:

`run_{UTC}_ep{N}_v{kind}_t{task}_s{seed}_r{nonce}`

Common per-run files: `eval_info.json`, `run_manifest.json`, `episodes/episode_*/actions.jsonl`, optional `videos/`.

Override roots with `--output-dir` / `--output-root` or env vars (`ORACLE_ARTIFACT_ROOT`, `SMOLVLA_ARTIFACT_ROOT`, etc.).

## Secondary tracks

- **MT10 / MT50:** `scripts/mt10/`, `scripts/mt50/` — multi-task eval and phase sweeps
- **Segment GRPO:** `scripts/segment_grpo/` — older WM segment pipeline from oracle top-k targets
- **ManiSkill:** `scripts/maniskill_smolvla/` — separate sim stack

## Legacy (repro only — ignore for new work)

- **pi0.5:** `vendor/pi05/`, `scripts/legacy_pushv3_data_pipeline_smolvla.sh`, `scripts/legacy_lerobot_eval_full_videos.py`, `vendor/pi05/run_baseline_eval_legacy_smolvla.sh`
- **RL4VLA:** `RL4VLA/` — legacy ManiSkill track; not used for Meta-World GRPO
- Phase 11 `custom` env backend in `train_phase11_env_on_policy_grpo.py` — use `--env-backend official_lerobot` instead

## Tests

```bash
pytest
```

Config: `pytest.ini`, suites under `tests/`.
