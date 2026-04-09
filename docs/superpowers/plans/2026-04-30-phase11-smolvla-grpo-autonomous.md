# Phase11 SmolVLA GRPO Autonomous Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Execute task-by-task. Use checkbox steps for tracking. Use `/caveman` style for user-visible updates: terse, technical, low-token.

**Goal:** Build and test Phase11 env-on-policy GRPO for SmolVLA on MetaWorld Push-v3, starting with a **patched `lerobot_mw_py310` LeRobot API gate** (see amendment), then one-update smoke, checkpoint reload, short continuation, and baseline-aligned eval.

**Architecture:** First prove the LeRobot GRPO hooks work in the **chosen training venv** (see amendment below). Then implement true on-policy GRPO: same seed context, `GROUP_SIZE=4` rollouts, group-normalized returns, clipped log-prob ratio objective, checkpoint every 5 updates. Phase12 WM reward stays a later reward-backend swap.

**Tech Stack:** Python 3.12 **MetaWorld venv** [`lerobot_mw_py310`](/vol/bitbucket/aa6622/.envs/lerobot_mw_py310) (patched `site-packages/lerobot`), LeRobot SmolVLA, MetaWorld Push-v3, PyTorch, Slurm, `scripts/slurm/common_env.sh`, checkpoint `jadechoghari/smolvla_metaworld`.

---

## Amendment 2026-05-01 (runtime and Task 1)

**Decision:** Phase11 GRPO uses **`/vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python`** as the primary interpreter. The SmolVLA GRPO patch is applied under that venv’s **`site-packages/lerobot`**, with **package-local git** in that package directory for rollback. Any older “second venv only for patches” workflow has been **dropped** from active instructions.

**`sys.executable` (what it means):** In any Slurm or local run, `import sys; print(sys.executable)` is the **path to the Python binary running that process**. Together with `lerobot.policies.smolvla.modeling_smolvla.__file__`, it proves you are not on `/usr/bin/python3` and that SmolVLA is loaded from **`.../lerobot_mw_py310/.../site-packages`**. Slurm smokes and checker scripts should **assert** both paths share that prefix before heavy jobs.

**Task 1 status:** **API + GPU import:** Slurm job `237046` — see [`docs/slurm/2026-05-01-smolvla-lerobot-pkg-import-gpu-smoke-237046.md`](./slurm/2026-05-01-smolvla-lerobot-pkg-import-gpu-smoke-237046.md). **`from_pretrained` + GPU forward + GRPO paths:** Slurm job **`237048`** — see [`docs/slurm/2026-05-04-smolvla-pretrained-gpu-forward-smoke-237048.md`](./slurm/2026-05-04-smolvla-pretrained-gpu-forward-smoke-237048.md) and log `smolvla_pretrained_gpu_forward_smoke_237048.out` (`predict_action_chunk` + `select_action_distr_params`, real hub checkpoint, `LEROBOT_MW_PYTHON` = `lerobot_mw_py310`). Optional follow-up: checker scripts and manifests under `scripts/grpo/` and `artifacts/phase11_env_on_policy_grpo/api_gate/` for **repeatability**, not as a hard blocker for **CPU Task 2** (pure math).

---

## Non-Negotiable Rules

- LeRobot SmolVLA GRPO patch lives only under **`/vol/bitbucket/aa6622/.envs/lerobot_mw_py310/.../site-packages/lerobot`**; keep **package-local git** there for rollback. Do not hand-edit without recording a commit in that package repo.
- Do not delete or overwrite phase07/phase08/phase09 artifacts.
- Do not run long Phase11 jobs until LeRobot API + **GPU forward** gate, one-update smoke, checkpoint reload, and 3-seed eval pass.
- If a gate fails, debug that gate. Do not abandon; do not launch downstream jobs.
- Commit after each passing gate. Do not commit copied env, checkpoints, videos, caches, or large artifacts.
- Use artifact root: `/vol/bitbucket/aa6622/project/artifacts/phase11_env_on_policy_grpo/`.
- Use base checkpoint: `jadechoghari/smolvla_metaworld`.
- Use Phase11 Python: **`/vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python`** (set `GRPO_PYTHON` or `LEROBOT_MW_PYTHON` in Slurm scripts to this path; never bare `python`).
- Use baseline-aligned eval: seed `1000`, `20` episodes (`1000..1019`).
- Use train seed schedule: update `u` uses seed `2000 + u`.
- No local GPU assumption on `gpucluster3`. Anything requiring real SmolVLA GPU forward, MetaWorld rollout, training, or eval must run through Slurm/sbatch.
- Before queuing GPU work, apply `slurm-sbatch-runner` rules: `.slurm` is one allocation worker, login-node scripts orchestrate dependencies, `--export=NIL`, explicit Python, pinned caches, `bash -n`, and `sbatch --test-only`.
- Every `sbatch` command must include `--chdir=/vol/bitbucket/aa6622/project` unless the script proves `SLURM_SUBMIT_DIR` already points at the project root.

## Environment Responsibility Map

Do not mix these up:

```text
scripts/slurm/common_env.sh
  owns: PROJECT_ROOT, PYTHONPATH, HF_HOME, HUGGINGFACE_HUB_CACHE, TRANSFORMERS_CACHE, HF_DATASETS_CACHE, TORCH_HOME, XDG_CACHE_HOME
  does not own: Python interpreter, LeRobot version

scripts/smolvla/submit_smolvla_parity_eval.slurm
  owns: baseline/parity eval launcher
  default Python: /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python via run_pushv3_smolvla_parity_benchmark.sh
  copied-env override exists: SMOLVLA_LEROBOT_ENV_DIR or SMOLVLA_PYTHON_BIN
  Phase11 rule: do not rely on this submitter for GRPO unless override is explicit and logged

scripts/grpo/submit_api_gate_smoke.slurm
scripts/grpo/submit_phase11_grpo.slurm
  own: all patched-LeRobot API gate, GRPO rollout, GRPO train, GRPO eval
  required Python: /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python
  required runtime check: sys.executable and lerobot.policies.smolvla.modeling_smolvla.__file__ must both live under /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/
```

Safe rule: every file that imports patched SmolVLA GRPO APIs (`select_action_distr_params`, `_get_distr_params_chunk`, `log_std`, `euler_step_noise_std`) must execute under the **Phase11 Python** (`lerobot_mw_py310` as above), never bare `python` and never an interpreter whose `site-packages/lerobot` is unpatched.

## Existing WIP To Inspect First

These files were created prematurely. Treat them as WIP, not trusted. Inspect, fix, or replace them to match this plan.

- `src/smolvla_grpo/__init__.py`
- `src/smolvla_grpo/grpo_math.py`
- `src/smolvla_grpo/policy_wrapper.py`
- `src/smolvla_grpo/phase11_rollout.py`
- `src/smolvla_grpo/reward_backends.py`
- `src/smolvla_grpo/checkpointing.py`
- `scripts/grpo/README.md`
- `scripts/grpo/check_lerobot_grpo_api.py`
- `scripts/grpo/check_smolvla_grpo_forward.py`
- `scripts/grpo/train_phase11_env_on_policy_grpo.py`
- `scripts/grpo/eval_phase11_checkpoints.py`
- `scripts/grpo/submit_phase11_grpo.slurm`
- `scripts/grpo/submit_phase11_chain.sh`
- `tests/test_grpo_math.py`
- Required new tests: `tests/test_grpo_policy_wrapper_static.py`
- Required new tests: `tests/test_grpo_checkpointing.py`
- Required new tests: `tests/test_phase11_slurm_scripts.py`
- Missing but required by this plan: `scripts/grpo/submit_api_gate_smoke.slurm`
- Missing but required by this plan: `scripts/grpo/smoke_phase11_rollout.py`

Known WIP issue: default `/usr/bin/python3` lacks `torch`; run tests with Phase11 venv Python (`lerobot_mw_py310/bin/python`).
Known WIP issue: `scripts/grpo/submit_phase11_grpo.slurm` currently uses bare `python`; fix to explicit `GRPO_PYTHON=/vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python`.
Known WIP issue: `scripts/grpo/submit_phase11_chain.sh` currently omits `--chdir`; add `--chdir=/vol/bitbucket/aa6622/project` to every `sbatch`.
Known WIP issue: `src/smolvla_grpo/policy_wrapper.py` currently calls `Normal.rsample(generator=...)`; fix to explicit noise sampling because common PyTorch versions do not support that `generator` argument.
Known WIP issue: `scripts/grpo/eval_phase11_checkpoints.py` currently defaults `--eval-seed-start` to `3000`; change default to `1000`.

## Git Safety

- [ ] **Step 1: Record current state**

Run:

```bash
cd /vol/bitbucket/aa6622/project
git status --short
git diff --stat
```

Expected: output recorded in notes. Do not reset or discard user changes.

- [ ] **Step 2: Create or confirm branch**

Run:

```bash
cd /vol/bitbucket/aa6622/project
git switch -c phase11-smolvla-grpo || git switch phase11-smolvla-grpo
```

Expected: on branch `phase11-smolvla-grpo`.

- [ ] **Step 3: Commit frequently**

Commit after each passing gate:

```bash
git add scripts/grpo src/smolvla_grpo tests/test_grpo_math.py docs/superpowers/plans/2026-04-30-phase11-smolvla-grpo-autonomous.md
git commit -m "test: add phase11 grpo api smoke"
```

Use later commit subjects:

```text
feat: add phase11 grpo math helpers
feat: add phase11 policy wrapper
feat: add phase11 rollout collector
feat: add phase11 trainer checkpoint flow
chore: add phase11 slurm launch scripts
test: add phase11 smoke eval flow
```

Never commit:

```text
/vol/bitbucket/aa6622/.envs/
artifacts/
.cache/
*.pt
*.mp4
*.png frame dumps
```

## Task 1: LeRobot API gate (modified `lerobot_mw_py310` env)

**As of amendment 2026-05-01:** The SmolVLA GRPO patch lives in **`lerobot_mw_py310`** `site-packages/lerobot` with package-local git. The former **copy-to-second-venv** workflow has been **removed** from this plan; do not reintroduce a parallel patch tree unless you consciously fork the approach.

**Files:**
- Patched file: `.../lerobot_mw_py310/lib/python3.12/site-packages/lerobot/policies/smolvla/modeling_smolvla.py` (rollback via package-local git tag `baseline-production-lerobot`).
- Optional: `*.orig` backup beside `modeling_smolvla.py` only if you are not using package-local git (not recommended when git is present).
- Create/replace: `scripts/grpo/check_lerobot_grpo_api.py`
- Create/replace: `scripts/grpo/check_smolvla_grpo_forward.py`
- Artifact manifest: `artifacts/phase11_env_on_policy_grpo/api_gate/env_patch_manifest.json`

- [ ] **Step 1: Verify Phase11 interpreter + patched SmolVLA path**

Run (login node or Slurm prolog):

```bash
PHASE11_PY=/vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python
"${PHASE11_PY}" - <<'PY'
import sys
import lerobot.policies.smolvla.modeling_smolvla as m
print("executable", sys.executable)
print("modeling_smolvla", m.__file__)
assert sys.executable.startswith("/vol/bitbucket/aa6622/.envs/lerobot_mw_py310/")
assert "/vol/bitbucket/aa6622/.envs/lerobot_mw_py310/" in m.__file__
PY
```

Expected: `sys.executable` and `modeling_smolvla.__file__` both under `lerobot_mw_py310` (evidence you are on the patched venv, not system Python).

**Evidence:** Import-only: job **`237046`** (log + [`docs/slurm/2026-05-01-smolvla-lerobot-pkg-import-gpu-smoke-237046.md`](./slurm/2026-05-01-smolvla-lerobot-pkg-import-gpu-smoke-237046.md)). Pretrained forward + distr: job **`237048`** ([`docs/slurm/2026-05-04-smolvla-pretrained-gpu-forward-smoke-237048.md`](./slurm/2026-05-04-smolvla-pretrained-gpu-forward-smoke-237048.md)); log shows `using PYTHON=.../lerobot_mw_py310/bin/python` and ends with `smolvla_pretrained_gpu_forward_smoke_ok`.

- [ ] **Step 2: Fetch safe-robot fork file** (optional reference for diffs; patch is already applied in py310)

Run:

```bash
WORK=/vol/bitbucket/aa6622/project/artifacts/phase11_env_on_policy_grpo/api_gate
mkdir -p "${WORK}"
cd "${WORK}"
if [[ ! -d jsnchon_lerobot ]]; then
  git clone --filter=blob:none https://github.com/jsnchon/lerobot.git jsnchon_lerobot
fi
cd jsnchon_lerobot
git fetch origin f30fc2a1b904bb2ccd752cfff94f6f4423bd523b
git checkout f30fc2a1b904bb2ccd752cfff94f6f4423bd523b
```

Expected: fork checkout at pinned commit.

- [ ] **Step 3: (Removed)** Patch application and `.orig` backup are **superseded** by package-local git under `site-packages/lerobot`; use `git -C …/lerobot log` / `reset --hard baseline-production-lerobot` instead of ad hoc copies.

- [ ] **Step 4: (Removed)** Full-file replace from fork into venv is **not** the active procedure once the py310 patch is committed in the package-local repo.

- [ ] **Step 5: Replace API checker with exact code**

File: `scripts/grpo/check_lerobot_grpo_api.py`

```python
#!/usr/bin/env python3
from __future__ import annotations

import inspect
import json
from pathlib import Path


def main() -> int:
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, VLAFlowMatching

    checks = {
        "SmolVLAPolicy.select_action_distr_params": hasattr(SmolVLAPolicy, "select_action_distr_params"),
        "SmolVLAPolicy._get_distr_params_chunk": hasattr(SmolVLAPolicy, "_get_distr_params_chunk"),
        "VLAFlowMatching.euler_step_noise_std_source": "euler_step_noise_std" in inspect.getsource(VLAFlowMatching),
        "VLAFlowMatching.log_std_source": "log_std" in inspect.getsource(VLAFlowMatching),
    }
    out = {"checks": checks, "all_passed": all(checks.values())}
    print(json.dumps(out, indent=2))
    return 0 if out["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: Run CPU API check**

Run:

```bash
cd /vol/bitbucket/aa6622/project
PYTHONPATH="${PWD}:${PWD}/src" /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python scripts/grpo/check_lerobot_grpo_api.py
```

Expected:

```text
"all_passed": true
```

- [ ] **Step 7: If CPU API check fails, rollback and re-port**

Restore `modeling_smolvla.py` using **package-local git** under `.../site-packages/lerobot` (tag `baseline-production-lerobot`), or restore from `.orig` only if you still maintain that file. Then port smaller chunks from the fork reference in this order, rerunning the CPU API check after each chunk:

1. `VLAFlowMatching.__init__`: add `init_log_std`, `self.log_std`, `self.euler_step_noise_std`.
2. `VLAFlowMatching.sample_actions`: return `(x_t, self.log_std)` and inject Euler noise hook.
3. `SmolVLAPolicy._get_distr_params_chunk`.
4. `SmolVLAPolicy.select_action_distr_params`.

Expected: continue debugging until CPU API check passes. Do not proceed downstream until pass.

- [ ] **Step 8: GPU forward/backward smoke via Slurm**

Create/replace `scripts/grpo/submit_api_gate_smoke.slurm`:

```bash
#!/bin/bash
#SBATCH --job-name=grpo_api_gate
#SBATCH --output=logs/grpo_api_gate_%j.out
#SBATCH --error=logs/grpo_api_gate_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=t4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts/slurm/common_env.sh" ]]; then
  _COMMON="${SLURM_SUBMIT_DIR}/scripts/slurm/common_env.sh"
else
  _COMMON="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)/../slurm/common_env.sh"
fi
source "${_COMMON}"
slurm_resolve_project_root "scripts/grpo/check_smolvla_grpo_forward.py"
cd "${PROJECT_ROOT}"
mkdir -p logs
slurm_export_pythonpath
slurm_export_hf_torch_cache "grpo_api_gate"

export SMOLVLA_POLICY_DEVICE="${SMOLVLA_POLICY_DEVICE:-cuda}"
export SMOLVLA_MAX_STEPS="${SMOLVLA_MAX_STEPS:-20}"
GRPO_PYTHON="${GRPO_PYTHON:-/vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python}"
CHECKPOINT="${SMOLVLA_INIT_CHECKPOINT:-jadechoghari/smolvla_metaworld}"
MODE="${1:-forward}"
shift || true

test -x "${GRPO_PYTHON}"
"${GRPO_PYTHON}" - <<'PY'
import sys
import lerobot.policies.smolvla.modeling_smolvla as m
print("[grpo_api_gate] executable", sys.executable)
print("[grpo_api_gate] modeling", m.__file__)
assert sys.executable.startswith("/vol/bitbucket/aa6622/.envs/lerobot_mw_py310/")
assert "/vol/bitbucket/aa6622/.envs/lerobot_mw_py310/" in m.__file__
PY

if [[ "${MODE}" == "forward" ]]; then
  exec "${GRPO_PYTHON}" scripts/grpo/check_smolvla_grpo_forward.py --checkpoint "${CHECKPOINT}"
elif [[ "${MODE}" == "base-eval" ]]; then
  OUT="${1:?output_dir}"
  exec "${GRPO_PYTHON}" scripts/smolvla/run_metaworld_smolvla_eval.py \
    --task push-v3 \
    --episodes 1 \
    --seed 1000 \
    --checkpoint "${CHECKPOINT}" \
    --output-dir "${OUT}" \
    --video true
else
  echo "usage: $0 [forward|base-eval <output_dir>]" >&2
  exit 2
fi
```

Run:

```bash
cd /vol/bitbucket/aa6622/project
bash -n scripts/grpo/submit_api_gate_smoke.slurm
sbatch --test-only --chdir=/vol/bitbucket/aa6622/project --export=NIL scripts/grpo/submit_api_gate_smoke.slurm
JOBID=$(sbatch --parsable --chdir=/vol/bitbucket/aa6622/project --export=NIL scripts/grpo/submit_api_gate_smoke.slurm)
echo "${JOBID}" | tee artifacts/phase11_env_on_policy_grpo/api_gate/api_gate_jobid.txt
```

Monitor:

```bash
squeue -j "${JOBID}"
```

Read stdout/stderr when done. Expected stdout contains:

```text
OK: forward+backward on distr params completed.
```

- [ ] **Step 9: Run existing 1-episode eval under copied env**

Use existing evaluator before GRPO training.

Run via Slurm only:

```bash
cd /vol/bitbucket/aa6622/project
OUT=artifacts/phase11_env_on_policy_grpo/api_gate/eval_1ep_seed1000
JOBID=$(sbatch --parsable --chdir=/vol/bitbucket/aa6622/project --export=NIL \
  scripts/grpo/submit_api_gate_smoke.slurm base-eval "${OUT}")
echo "${JOBID}"
```

Expected files:

```text
artifacts/phase11_env_on_policy_grpo/api_gate/eval_1ep_seed1000/episodes/episode_0000/actions.jsonl
artifacts/phase11_env_on_policy_grpo/api_gate/eval_1ep_seed1000/episodes/episode_0000/episode_meta.json
```

Expected `episode_meta.json` has `sum_reward`.

- [ ] **Step 10: Commit API gate**

Only commit repo files, not env or artifacts:

```bash
cd /vol/bitbucket/aa6622/project
git add scripts/grpo/check_lerobot_grpo_api.py scripts/grpo/check_smolvla_grpo_forward.py scripts/grpo/submit_api_gate_smoke.slurm docs/superpowers/plans/2026-04-30-phase11-smolvla-grpo-autonomous.md
git commit -m "test: add lerobot grpo api gate"
```

## Task 2: GRPO Math Core

**Files:**
- Create/replace: `src/smolvla_grpo/grpo_math.py`
- Test: `tests/test_grpo_math.py`

- [ ] **Step 1: Implement math helpers**

Required behavior:

- `compute_group_advantages(torch.tensor([r0, r1, ...]))`
- zero variance returns all zeros.
- nonzero variance returns normalized tensor mean near 0.
- `compute_clipped_grpo_loss(new_log_probs, old_log_probs, advantage, epsilon=0.2)` returns scalar loss and stats.
- stats include `mean_ratio`, `max_ratio`, `clip_fraction`, `n`.

- [ ] **Step 2: Test with copied env Python**

Run:

```bash
cd /vol/bitbucket/aa6622/project
PYTHONPATH="${PWD}:${PWD}/src" /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python -m pytest tests/test_grpo_math.py -v
```

Expected: all tests pass.

- [ ] **Step 3: Commit**

```bash
git add src/smolvla_grpo/grpo_math.py tests/test_grpo_math.py
git commit -m "feat: add phase11 grpo math helpers"
```

## Task 3: Policy Wrapper

**Files:**
- Create/replace: `src/smolvla_grpo/policy_wrapper.py`
- Test through GPU smoke: `scripts/grpo/check_smolvla_grpo_forward.py`
- Test: `tests/test_grpo_policy_wrapper_static.py`

- [ ] **Step 1: Implement wrapper**

Required class: `MetaWorldSmolVLAGRPOPolicy`.

Required methods:

```python
assert_grpo_api()
set_log_std(value: float)
set_euler_step_noise_std(std: float)
build_proc_batch(obs, env)
sample_action_from_proc(proc, rng=None)
get_action_probs_from_proc_list(proc_snapshots, unsquashed_actions)
```

Required math:

```text
noise ~ Normal(0, 1)
unsquashed = mean + exp(log_std) * noise
action = tanh(unsquashed)
log_prob = gaussian_log_prob(unsquashed) - sum(log(1 - action^2 + eps))
```

Required implementation detail:

```python
std = torch.exp(log_std)
if rng is None:
    noise = torch.randn_like(mean)
else:
    noise = torch.randn(mean.shape, generator=rng, device=mean.device, dtype=mean.dtype)
unsquashed = mean + std * noise
squished = torch.tanh(unsquashed)
```

Do not call `torch.distributions.Normal(...).rsample(generator=rng)`; common PyTorch versions do not support that `generator` argument.

Required Euler noise setter:

```python
if hasattr(self._policy, "euler_step_noise_std"):
    self._policy.euler_step_noise_std = float(std)
elif hasattr(self._policy, "model") and hasattr(self._policy.model, "euler_step_noise_std"):
    self._policy.model.euler_step_noise_std = float(std)
else:
    raise AttributeError("SmolVLA policy has no euler_step_noise_std hook")
```

- [ ] **Step 2: Add static regression test for PyTorch generator bug**

Create `tests/test_grpo_policy_wrapper_static.py`:

```python
from pathlib import Path


def test_policy_wrapper_does_not_call_rsample_with_generator():
    src = Path("src/smolvla_grpo/policy_wrapper.py").read_text(encoding="utf-8")
    assert ".rsample(generator=" not in src
    assert "torch.randn(" in src or "torch.randn_like(" in src
    assert "generator=rng" in src


def test_policy_wrapper_checks_policy_and_model_for_euler_noise():
    src = Path("src/smolvla_grpo/policy_wrapper.py").read_text(encoding="utf-8")
    assert 'hasattr(self._policy, "euler_step_noise_std")' in src
    assert 'hasattr(self._policy.model, "euler_step_noise_std")' in src
```

Run:

```bash
cd /vol/bitbucket/aa6622/project
/vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python -m pytest tests/test_grpo_policy_wrapper_static.py -v
```

Expected: both tests pass.

- [ ] **Step 3: GPU smoke**

Run Slurm API smoke again:

```bash
cd /vol/bitbucket/aa6622/project
JOBID=$(sbatch --parsable --chdir=/vol/bitbucket/aa6622/project --export=NIL scripts/grpo/submit_api_gate_smoke.slurm)
squeue -j "${JOBID}"
```

Expected: forward/backward smoke passes.

- [ ] **Step 4: Commit**

```bash
git add src/smolvla_grpo/policy_wrapper.py scripts/grpo/check_smolvla_grpo_forward.py tests/test_grpo_policy_wrapper_static.py
git commit -m "feat: add phase11 smolvla logprob wrapper"
```

## Task 4: Rollout Collector

**Files:**
- Create/replace: `src/smolvla_grpo/phase11_rollout.py`
- Create optional smoke script: `scripts/grpo/smoke_phase11_rollout.py`

- [ ] **Step 1: Implement deterministic Push-v3 rollout group**

Required defaults:

```text
task=push-v3
reset_seed=2000
episode_index=0
group_size=2 for smoke, 4 for real
max_steps=10 for smoke, 120 for real
```

Required per-trajectory fields:

```text
reset_seed
rollout_index
proc_snapshots
exec_actions
unsquashed_actions
log_probs
rewards
successes
terminated
truncated
metadata
```

Use:

```python
seed_metaworld_process(seed)
env.set_task(...)
gymnasium_reset_strict(env, seed)
```

- [ ] **Step 2: GPU rollout smoke**

Run:

```bash
cd /vol/bitbucket/aa6622/project
JOBID=$(sbatch --parsable --chdir=/vol/bitbucket/aa6622/project --export=NIL \
  scripts/grpo/submit_phase11_grpo.slurm rollout-smoke \
  --checkpoint jadechoghari/smolvla_metaworld \
  --group-size 2 \
  --max-steps 10 \
  --seed 2000)
echo "${JOBID}"
```

Expected:

```text
2 trajectories
each has rewards length > 0
each has log_probs length == rewards length
each has unsquashed_actions length == rewards length
```

If local login node has no GPU, submit same via Slurm.

- [ ] **Step 3: Commit**

```bash
git add src/smolvla_grpo/phase11_rollout.py scripts/grpo/smoke_phase11_rollout.py
git commit -m "feat: add phase11 rollout collector"
```

## Task 5: Trainer And Checkpoint Flow

**Files:**
- Create/replace: `src/smolvla_grpo/checkpointing.py`
- Create/replace: `src/smolvla_grpo/reward_backends.py`
- Create/replace: `scripts/grpo/train_phase11_env_on_policy_grpo.py`

- [ ] **Step 1: Implement checkpointing**

Checkpoint must include:

```text
policy_state_dict
optimizer_state_dict
update_index
args
extra
```

Also write sidecar:

```text
update_0001.pt.meta.json
```

- [ ] **Step 2: Implement trainer**

Loop:

```text
for update in start_update..target:
  reset_seed = train_seed_base + update
  collect GROUP_SIZE rollouts with old policy
  returns = sum(rewards)
  advantages = group_normalize(returns)
  recompute current log_probs on stored proc_snapshots
  ratio = exp(new - old)
  clipped loss with eps=0.2
  backward in chunks of 5 timesteps
  grad clip 1.0
  optimizer step
  refresh old policy
  save latest.pt every update
  save update_XXXX.pt every 5 updates and end
  append progress.jsonl
```

- [ ] **Step 3: One-update GPU smoke**

Run:

```bash
cd /vol/bitbucket/aa6622/project
OUT=artifacts/phase11_env_on_policy_grpo/smoke_update1
JOBID=$(sbatch --parsable --chdir=/vol/bitbucket/aa6622/project --export=NIL \
  scripts/grpo/submit_phase11_grpo.slurm train \
  jadechoghari/smolvla_metaworld \
  "${OUT}" \
  2000 0 1)
echo "${JOBID}"
```

Expected:

```text
${OUT}/progress.jsonl
${OUT}/checkpoints/latest.pt
artifacts/phase11_env_on_policy_grpo/smoke_update1/checkpoints/update_0001.pt
```

This is always Slurm on `gpucluster3`.

- [ ] **Step 4: Reload checkpoint smoke**

Run:

```bash
cd /vol/bitbucket/aa6622/project
PY=/vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python
PYTHONPATH="${PWD}:${PWD}/src" "${PY}" - <<'PY'
from pathlib import Path
from smolvla_grpo.checkpointing import load_grpo_checkpoint
ck = load_grpo_checkpoint(Path("artifacts/phase11_env_on_policy_grpo/smoke_update1/checkpoints/update_0001.pt"), map_location="cpu")
assert "policy_state_dict" in ck
assert ck["update_index"] == 0
print("checkpoint reload ok")
PY
```

Expected: `checkpoint reload ok`.

- [ ] **Step 5: Commit**

```bash
git add src/smolvla_grpo/checkpointing.py src/smolvla_grpo/reward_backends.py scripts/grpo/train_phase11_env_on_policy_grpo.py
git commit -m "feat: add phase11 trainer checkpoint flow"
```

## Task 6: Eval And Slurm Automation

**Files:**
- Create/replace: `scripts/grpo/eval_phase11_checkpoints.py`
- Create/replace: `scripts/grpo/submit_phase11_grpo.slurm`
- Create/replace: `scripts/grpo/submit_phase11_chain.sh`
- Test: `tests/test_grpo_checkpointing.py`
- Test: `tests/test_phase11_slurm_scripts.py`

- [ ] **Step 1: Eval script**

Eval must:

- load base checkpoint `jadechoghari/smolvla_metaworld`
- load GRPO `.pt` state with `strict=False`
- run Push-v3 evaluator
- default `--eval-seed-start 1000`
- episodes `20`
- write `eval_summary.json` and `eval_episodes.jsonl`

- [ ] **Step 2: Smoke eval `update_0001.pt`**

Run:

```bash
cd /vol/bitbucket/aa6622/project
JOBID=$(sbatch --parsable --chdir=/vol/bitbucket/aa6622/project --export=NIL \
  scripts/grpo/submit_phase11_grpo.slurm eval \
  jadechoghari/smolvla_metaworld \
  artifacts/phase11_env_on_policy_grpo/smoke_update1/checkpoints/update_0001.pt \
  artifacts/phase11_env_on_policy_grpo/eval_update0001_seed1000_ep3 \
  1000 3)
echo "${JOBID}"
```

Expected:

```text
eval_summary.json includes pc_success and avg_sum_reward
```

- [ ] **Step 2b: Add checkpoint reload semantics test**

Create `tests/test_grpo_checkpointing.py`:

```python
from pathlib import Path

import torch

from smolvla_grpo.checkpointing import load_grpo_checkpoint, save_grpo_checkpoint


def test_checkpoint_round_trip_preserves_update_index(tmp_path: Path):
    path = tmp_path / "update_0001.pt"
    save_grpo_checkpoint(
        path,
        policy_state_dict={"w": torch.tensor([1.0])},
        optimizer_state_dict={"state": {}, "param_groups": []},
        update_index=0,
        args={"num_updates": 1},
        extra={"train_seed": 2000},
    )
    ck = load_grpo_checkpoint(path, map_location="cpu")
    assert ck["update_index"] == 0
    assert ck["args"]["num_updates"] == 1
    assert ck["extra"]["train_seed"] == 2000
    assert torch.equal(ck["policy_state_dict"]["w"], torch.tensor([1.0]))
```

Run:

```bash
cd /vol/bitbucket/aa6622/project
PYTHONPATH="${PWD}:${PWD}/src" /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python -m pytest tests/test_grpo_checkpointing.py -v
```

Expected: test passes.

- [ ] **Step 3: Slurm worker**

`.slurm` rules:

- source `scripts/slurm/common_env.sh`
- source via `SLURM_SUBMIT_DIR` first, then `BASH_SOURCE` fallback, matching `scripts/smolvla/submit_smolvla_parity_eval.slurm`
- `slurm_resolve_project_root "scripts/grpo/train_phase11_env_on_policy_grpo.py"`
- use `GRPO_PYTHON=/vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python`
- set `#SBATCH --partition=t4`, `#SBATCH --output=logs/...`, and `mkdir -p logs` after `cd "${PROJECT_ROOT}"`
- default eval seed start is `1000`, not `3000`
- support worker modes: `train`, `eval`, and `rollout-smoke`
- no nested `sbatch`
- one worker only

Replace `scripts/grpo/submit_phase11_grpo.slurm` with this structure:

```bash
#!/bin/bash
#SBATCH --job-name=phase11_grpo
#SBATCH --output=logs/phase11_grpo_%j.out
#SBATCH --error=logs/phase11_grpo_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=t4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts/slurm/common_env.sh" ]]; then
  _COMMON="${SLURM_SUBMIT_DIR}/scripts/slurm/common_env.sh"
else
  _COMMON="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)/../slurm/common_env.sh"
fi
source "${_COMMON}"
slurm_resolve_project_root "scripts/grpo/train_phase11_env_on_policy_grpo.py"
cd "${PROJECT_ROOT}"
mkdir -p logs
slurm_export_pythonpath
slurm_export_hf_torch_cache "phase11_grpo"

GRPO_PYTHON="${GRPO_PYTHON:-/vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python}"
MODE="${1:-train}"
shift || true

test -x "${GRPO_PYTHON}"
"${GRPO_PYTHON}" - <<'PY'
import sys
import lerobot.policies.smolvla.modeling_smolvla as m
print("[phase11_grpo] executable", sys.executable)
print("[phase11_grpo] modeling", m.__file__)
assert sys.executable.startswith("/vol/bitbucket/aa6622/.envs/lerobot_mw_py310/")
assert "/vol/bitbucket/aa6622/.envs/lerobot_mw_py310/" in m.__file__
PY

if [[ "${MODE}" == "train" ]]; then
  CKPT="${1:?checkpoint}"
  OUT="${2:?output_dir}"
  SEED_BASE="${3:-2000}"
  START="${4:-0}"
  NUPD="${5:-1}"
  RESUME="${6:-}"
  EXTRA=()
  if [[ -n "${RESUME}" ]]; then
    EXTRA+=(--resume "${RESUME}")
  fi
  exec "${GRPO_PYTHON}" scripts/grpo/train_phase11_env_on_policy_grpo.py \
    --checkpoint "${CKPT}" \
    --output-dir "${OUT}" \
    --train-seed-base "${SEED_BASE}" \
    --start-update "${START}" \
    --num-updates "${NUPD}" \
    "${EXTRA[@]}"
elif [[ "${MODE}" == "eval" ]]; then
  BASE="${1:?base checkpoint}"
  GRPO_PT="${2:?grpo pt}"
  EVAL_OUT="${3:?eval output}"
  EVAL_START="${4:-1000}"
  EPS="${5:-20}"
  exec "${GRPO_PYTHON}" scripts/grpo/eval_phase11_checkpoints.py \
    --base-checkpoint "${BASE}" \
    --grpo-checkpoint "${GRPO_PT}" \
    --output-dir "${EVAL_OUT}" \
    --eval-seed-start "${EVAL_START}" \
    --episodes "${EPS}"
elif [[ "${MODE}" == "rollout-smoke" ]]; then
  exec "${GRPO_PYTHON}" scripts/grpo/smoke_phase11_rollout.py "$@"
else
  echo "usage: $0 train <ckpt> <out> [seed_base] [start] [n_updates] [resume_pt]" >&2
  echo "       $0 eval <base_ckpt> <grpo.pt> <eval_out> [eval_seed_start] [episodes]" >&2
  echo "       $0 rollout-smoke <smoke_phase11_rollout.py args...>" >&2
  exit 2
fi
```

Update `scripts/grpo/submit_phase11_chain.sh` so every `sbatch` has `--chdir=/vol/bitbucket/aa6622/project` before the script path.

- [ ] **Step 3b: Add Slurm script content test**

Create `tests/test_phase11_slurm_scripts.py`:

```python
from pathlib import Path


def test_phase11_worker_uses_explicit_env_python_and_safe_root_resolution():
    src = Path("scripts/grpo/submit_phase11_grpo.slurm").read_text(encoding="utf-8")
    assert "GRPO_PYTHON" in src
    assert "/vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python" in src
    assert "exec python" not in src
    assert "SLURM_SUBMIT_DIR" in src
    assert "BASH_SOURCE[0]:-$0" in src
    assert "sys.executable.startswith" in src
    assert "lerobot_mw_py310" in src
    assert 'EVAL_START="${4:-1000}"' in src
    assert "#SBATCH --partition=t4" in src
    assert "mkdir -p logs" in src


def test_phase11_chain_uses_chdir_for_every_sbatch():
    src = Path("scripts/grpo/submit_phase11_chain.sh").read_text(encoding="utf-8")
    sbatch_lines = [line for line in src.splitlines() if "sbatch" in line and not line.strip().startswith("#")]
    assert sbatch_lines
    assert all("--chdir=/vol/bitbucket/aa6622/project" in line for line in sbatch_lines)
```

Run:

```bash
cd /vol/bitbucket/aa6622/project
/vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python -m pytest tests/test_phase11_slurm_scripts.py -v
```

Expected: both tests pass.

- [ ] **Step 4: Slurm syntax and test-only**

Run:

```bash
cd /vol/bitbucket/aa6622/project
bash -n scripts/grpo/submit_phase11_grpo.slurm
bash -n scripts/grpo/submit_phase11_chain.sh
sbatch --test-only --chdir=/vol/bitbucket/aa6622/project --export=NIL scripts/grpo/submit_phase11_grpo.slurm train jadechoghari/smolvla_metaworld artifacts/phase11_env_on_policy_grpo/slurm_test 2000 0 1
```

Expected: no syntax error and Slurm accepts test-only.

- [ ] **Step 5: Commit**

```bash
git add scripts/grpo/eval_phase11_checkpoints.py scripts/grpo/submit_phase11_grpo.slurm scripts/grpo/submit_phase11_chain.sh tests/test_grpo_checkpointing.py tests/test_phase11_slurm_scripts.py
git commit -m "chore: add phase11 slurm eval flow"
```

## Task 7: Progressive Training And Eval

- [ ] **Step 1: Submit smoke**

Run:

```bash
cd /vol/bitbucket/aa6622/project
JOB1=$(sbatch --parsable --chdir=/vol/bitbucket/aa6622/project --export=NIL \
  scripts/grpo/submit_phase11_grpo.slurm train \
  jadechoghari/smolvla_metaworld \
  artifacts/phase11_env_on_policy_grpo/run_progressive/smoke \
  2000 0 1)
echo "${JOB1}"
```

Monitor until done:

```bash
squeue -j "${JOB1}"
```

Expected:

```text
run_progressive/smoke/checkpoints/update_0001.pt
```

- [ ] **Step 2: Eval smoke**

Run eval on `1000..1002`:

```bash
cd /vol/bitbucket/aa6622/project
JOB_EVAL1=$(sbatch --parsable --chdir=/vol/bitbucket/aa6622/project --export=NIL \
  scripts/grpo/submit_phase11_grpo.slurm eval \
  jadechoghari/smolvla_metaworld \
  artifacts/phase11_env_on_policy_grpo/run_progressive/smoke/checkpoints/update_0001.pt \
  artifacts/phase11_env_on_policy_grpo/run_progressive/eval_update0001_seed1000_ep3 \
  1000 3)
echo "${JOB_EVAL1}"
```

Expected: `eval_summary.json` includes `pc_success` and `avg_sum_reward`.

- [ ] **Step 3: Submit short continuation**

Only after smoke eval passes:

```bash
JOB2=$(sbatch --parsable --chdir=/vol/bitbucket/aa6622/project --export=NIL --dependency=afterok:${JOB1} \
  scripts/grpo/submit_phase11_grpo.slurm train \
  jadechoghari/smolvla_metaworld \
  artifacts/phase11_env_on_policy_grpo/run_progressive/short_to5 \
  2000 1 4 \
  artifacts/phase11_env_on_policy_grpo/run_progressive/smoke/checkpoints/latest.pt)
echo "${JOB2}"
```

Expected:

```text
short_to5/checkpoints/update_0005.pt
```

- [ ] **Step 4: Eval `update_0005.pt`**

Eval seed `1000`, episodes `20`:

```bash
cd /vol/bitbucket/aa6622/project
JOB_EVAL5=$(sbatch --parsable --chdir=/vol/bitbucket/aa6622/project --export=NIL \
  scripts/grpo/submit_phase11_grpo.slurm eval \
  jadechoghari/smolvla_metaworld \
  artifacts/phase11_env_on_policy_grpo/run_progressive/short_to5/checkpoints/update_0005.pt \
  artifacts/phase11_env_on_policy_grpo/run_progressive/eval_update0005_seed1000_ep20 \
  1000 20)
echo "${JOB_EVAL5}"
```

- [ ] **Step 5: Submit long continuation**

Only after `update_0005.pt` eval passes:

```bash
JOB3=$(sbatch --parsable --chdir=/vol/bitbucket/aa6622/project --export=NIL --dependency=afterok:${JOB2} \
  scripts/grpo/submit_phase11_grpo.slurm train \
  jadechoghari/smolvla_metaworld \
  artifacts/phase11_env_on_policy_grpo/run_progressive/long_to25 \
  2000 5 20 \
  artifacts/phase11_env_on_policy_grpo/run_progressive/short_to5/checkpoints/latest.pt)
echo "${JOB3}"
```

Expected:

```text
long_to25/checkpoints/update_0025.pt
```

- [ ] **Step 6: Final eval**

Eval checkpoints:

```text
update_0005.pt
update_0010.pt
update_0015.pt
update_0020.pt
update_0025.pt
```

Eval config:

```text
seed=1000
episodes=20
max_steps=120
videos only for best/final unless debugging
```

Run:

```bash
cd /vol/bitbucket/aa6622/project
JOB_EVAL=$(sbatch --parsable --chdir=/vol/bitbucket/aa6622/project --export=NIL \
  scripts/grpo/submit_phase11_grpo.slurm eval \
  jadechoghari/smolvla_metaworld \
  artifacts/phase11_env_on_policy_grpo/run_progressive/short_to5/checkpoints/update_0005.pt \
  artifacts/phase11_env_on_policy_grpo/run_progressive/eval_update0005_seed1000_ep20 \
  1000 20)
echo "update_0005: ${JOB_EVAL}"

for U in 0010 0015 0020 0025; do
  JOB_EVAL=$(sbatch --parsable --chdir=/vol/bitbucket/aa6622/project --export=NIL \
    scripts/grpo/submit_phase11_grpo.slurm eval \
    jadechoghari/smolvla_metaworld \
    "artifacts/phase11_env_on_policy_grpo/run_progressive/long_to25/checkpoints/update_${U}.pt" \
    "artifacts/phase11_env_on_policy_grpo/run_progressive/eval_update${U}_seed1000_ep20" \
    1000 20)
  echo "update_${U}: ${JOB_EVAL}"
done
```

## Failure Debug Ladder

If gate fails, do not give up. Debug current gate only.

- API patch fails:
  - restore `.orig`
  - port smaller chunks in order: `log_std`, tuple return, `_get_distr_params_chunk`, `select_action_distr_params`
  - rerun CPU API check after each chunk

- checkpoint load fails:
  - verify `jadechoghari/smolvla_metaworld` works in baseline env
  - retry `strict=False`
  - inspect missing/unexpected keys
  - do not train until load works

- GPU smoke fails:
  - test `select_action` first
  - test `select_action_distr_params`
  - test log-prob
  - test backward
  - dump `mean`, `log_std`, `unsquashed`, `log_prob`

- GRPO loss NaN:
  - reduce to `max_steps=10`, `group_size=2`
  - print ratio stats
  - inspect `log_std`
  - add guard only after cause known

- Slurm fails:
  - inspect stdout/stderr
  - rerun smallest GPU smoke
  - do not queue long continuation until smoke passes

## Final Success Criteria

Minimum overnight success:

- copied env patched, baseline env untouched
- CPU API gate passed
- GPU forward/backward gate passed
- 1-episode eval under copied env passed
- one GRPO update completed
- `update_0001.pt` reload passed
- 3-seed eval on `1000..1002` completed

Good overnight success:

- `update_0005.pt` exists
- 20-episode eval on `1000..1019` completed

Best overnight success:

- `update_0025.pt` exists
- checkpoint evals complete for `0005/0010/0015/0020/0025`
- best checkpoint identified by `pc_success` then `avg_sum_reward`

