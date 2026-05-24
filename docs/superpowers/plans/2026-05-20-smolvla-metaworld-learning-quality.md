# SmolVLA MetaWorld Learning Quality Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix train seed diversity + reward contract + real eval, then launch three parallel RTX6000 GRPO chains (40 updates each, B=4) tuned for learning signal without collapse. Each chain runs four 10-update train chunks; after each chunk, launch a 25-episode eval sweep over the five checkpoints saved in that chunk while the next train chunk starts in parallel.

**Architecture:** Mirror Phase11 seed schedule inside `RayTrainer.fit` (override `trial_seed` per `global_step`, not static dataset row). Use existing `collect_rollout_seed_batch` when `train_batch_size>1`. Keep GRPO grouping on stable `uid=push-v3:{seed}` (same seed across G replicas; replica RNG stays `reset_seed*1000003 + r*7919` in Phase11 rollout). Add explicit `reward.mode` on rollout scores before GRPO. Eval: disable in-trainer eval for chained launch; run PBS 25/100-seed sweeps via new safetensors→policy loader bridge.

**Tech Stack:** SimpleVLA-RL/verl, Hydra, Ray+FSDP, Phase11 `smolvla_grpo` rollouts, PBS `v1_gpu72` RTX6000, ephemeral `/rds/general/user/aa6622/ephemeral/smolvla_metaworld/`.

**Branch:** `main` in `project/SimpleVLA-RL`. Commit after each task. Worktree may be dirty — do not reset.

---

## Root cause (why 20-update run flatlined)

| Issue | Evidence | Fix |
|-------|----------|-----|
| Same train seed every step | `num_trials_per_task=1` → always `trial_seed=2000`; ledger uses `2000+update` | `seed_schedule=global_step` override |
| Low seed diversity per update | `train_batch_size=1`, `n_samples=4` | `B=4`, `G=4` |
| Val not representative | Previous runs had disabled/tiny/inconsistent eval | External PBS eval sweeps every 10-update chunk, 25 episodes per checkpoint |
| Reward ambiguity | `token_level_scores`=dense sum; metrics use `acc` | `reward.mode` + audit script |
| Eval format mismatch | verl saves `model.safetensors`; Phase11 eval expects `.pt` | `eval_verl_smolvla_checkpoint.py` |

---

## File map

| File | Change |
|------|--------|
| `verl/utils/metaworld_seed_scheduler.py` | **Create** — Phase11 seed formulas |
| `verl/trainer/ppo/ray_trainer.py` | Override `trial_seed` + uid before rollout |
| `verl/workers/rollout/smolvla_metaworld_rollout.py` | `collect_rollout_seed_batch` when B>1; reward modes |
| `verl/trainer/config/smolvla_metaworld_grpo.yaml` | B=4, G=4, seed/reward/eval defaults |
| `verl/trainer/main_ppo.py` | Log reward mode in metrics (optional) |
| `tests/smolvla_metaworld/test_seed_scheduler.py` | **Create** |
| `tests/smolvla_metaworld/test_rollout_reward_modes.py` | **Create** |
| `tests/smolvla_metaworld/test_ray_trainer_seed_override.py` | **Create** |
| `scripts/grpo/audit_smolvla_reward_stats.py` | **Create** — one-batch reward histogram |
| `scripts/grpo/eval_verl_smolvla_checkpoint.py` | **Create** — load safetensors dir for 25/100 ep eval |
| `scripts/grpo/eval_verl_smolvla_sweep.pbs` | **Create** — post-train checkpoint sweep |
| `scripts/grpo/run_smolvla_rl_metaworld_pushv3.pbs` | Env vars for triplet runs |
| `scripts/grpo/submit_smolvla_learning_triplet.sh` | **Create** — qsub 3 train/eval chains |

---

## Seed + GRPO contract (do not break)

1. **Train reset seed** (per row `b` in batch, update `u` = `global_steps`):
   - `trial_seed = train_seed_base + u * train_batch_size + b`
   - Ledger default: `train_seed_base=2000` ([`docs/findings/2026-05-19-phase11-recent-run-ledger.md`](docs/findings/2026-05-19-phase11-recent-run-ledger.md))
2. **Replica sampling** (within group): unchanged in `phase11_rollout.py` / vector rollout — `reset_seed * 1000003 + r * 7919`. Do **not** put `trial_id` or `replica_idx` into `trial_seed`.
3. **GRPO uid**: `f"{task_name}:{trial_seed}"` — all G rollouts from same row share uid; advantages normalized within group ([`core_algos.compute_grpo_outcome_advantage`](project/SimpleVLA-RL/verl/trainer/ppo/core_algos.py)).
4. **Eval seeds**: `eval_seed_base + trial_id`, valid split `train_val=valid`, `num_trials_per_task=25` (quick) or `100` (final PBS).

---

## Three parallel GPU chains (after Tasks 1–5 green)

Submit together on `v1_gpu72` RTX6000. Shared: 40 updates per experiment, chunked as `0-10`, `10-20`, `20-30`, `30-40`; `save_freq=2`, external 25-episode eval sweeps over each chunk's five checkpoints, `B=4`, chunk_len=5, vector_sync, ephemeral checkpoints under `/rds/general/user/aa6622/ephemeral/smolvla_metaworld/`.

| Job | Experiment | G | LR | clip | reward.mode | Rationale |
|-----|------------|---|-----|------|-------------|-----------|
| **A** | `pushv3_b4g4_dense_lr5e6` | 4 | 5e-6 | 0.1 | `dense_return` | Phase11 parity + diversity fix (primary) |
| **B** | `pushv3_b4g8_dense_lr3e6` | 8 | 3e-6 | 0.1 | `dense_return` | Lower LR + wider group → stabler GRPO rank |
| **C** | `pushv3_b4g4_dense_lr1e6_clip005` | 4 | 1e-6 | 0.05 | `dense_return` | Anti-collapse conservative (small policy steps) |

**Not** running three identical configs. **Not** success-only on A/B (changes objective before dense baseline proven on verl path).

### Chain schedule + resources

Per experiment:

```text
train_00_10
  afterok -> eval_02_10   (checkpoints 2,4,6,8,10; 25 episodes each)
  afterok -> train_10_20
                 afterok -> eval_12_20
                 afterok -> train_20_30
                                afterok -> eval_22_30
                                afterok -> train_30_40
                                               afterok -> eval_32_40
```

Train jobs:
- PBS resources: `select=1:ncpus=48:mem=64gb:ngpus=1:gpu_type=RTX6000`
- Walltime: `02:00:00`
- Updates per job: `10`
- Checkpoint every `2` updates
- Resume from previous chunk's latest ephemeral checkpoint

Eval jobs:
- PBS resources: `select=1:ncpus=32:mem=32gb:ngpus=1:gpu_type=RTX6000`
- Walltime: `00:40:00`
- Sweep checkpoints from completed chunk only: `2..10`, `12..20`, `22..30`, `32..40`
- Standard quick eval: `25` episodes, `eval_seed_start=1000`, same valid protocol

Timing note: these resources match recent Phase11 chain scripts. `02:00:00` train walltime should be fine if verl batched rollout/logprob path is near Phase11 resume speed, but it is tighter than the old 3-5 min/update estimate (10 updates could exceed 2h at worst-case 5 min/update plus startup). First chunk may be slowest due Ray/HF startup. Keep `02:00:00` as requested, but treat first chain as timing smoke; bump to `03:00:00` if first chunk queues then hits walltime before update 10. Eval `00:40:00` is plausible for 5 checkpoints x 25 episodes based on prior Phase11 eval sweeps, but verl loader startup could make it tight; failed eval can be resubmitted without losing train progress.

---

### Task 1: Seed scheduler utility

**Files:**
- Create: `project/SimpleVLA-RL/verl/utils/metaworld_seed_scheduler.py`
- Create: `project/SimpleVLA-RL/tests/smolvla_metaworld/test_seed_scheduler.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/smolvla_metaworld/test_seed_scheduler.py
from verl.utils.metaworld_seed_scheduler import (
    train_trial_seed,
    train_trial_seeds_for_batch,
    eval_trial_seed,
)


def test_train_trial_seed_phase11_formula():
    assert train_trial_seed(train_seed_base=2000, global_step=7, batch_size=4, row_idx=2) == 2030


def test_train_trial_seeds_for_batch():
    assert train_trial_seeds_for_batch(
        train_seed_base=2000, global_step=3, batch_size=4
    ) == [2012, 2013, 2014, 2015]


def test_eval_trial_seed():
    assert eval_trial_seed(eval_seed_base=1000, trial_id=5) == 1005
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd project/SimpleVLA-RL && python -m pytest tests/smolvla_metaworld/test_seed_scheduler.py -v`  
Expected: FAIL `ModuleNotFoundError`

- [ ] **Step 3: Implement scheduler**

```python
# verl/utils/metaworld_seed_scheduler.py
from __future__ import annotations


def train_trial_seed(*, train_seed_base: int, global_step: int, batch_size: int, row_idx: int) -> int:
    return int(train_seed_base) + int(global_step) * int(batch_size) + int(row_idx)


def train_trial_seeds_for_batch(*, train_seed_base: int, global_step: int, batch_size: int) -> list[int]:
    return [train_trial_seed(
        train_seed_base=train_seed_base,
        global_step=global_step,
        batch_size=batch_size,
        row_idx=b,
    ) for b in range(int(batch_size))]


def eval_trial_seed(*, eval_seed_base: int, trial_id: int) -> int:
    return int(eval_seed_base) + int(trial_id)
```

- [ ] **Step 4: Run test — expect PASS**

- [ ] **Step 5: Commit**

```bash
cd project/SimpleVLA-RL && git add verl/utils/metaworld_seed_scheduler.py tests/smolvla_metaworld/test_seed_scheduler.py
git commit -m "feat(smolvla): add MetaWorld train/eval seed scheduler"
```

---

### Task 2: RayTrainer global_step seed override

**Files:**
- Modify: `project/SimpleVLA-RL/verl/trainer/ppo/ray_trainer.py` (~565–595, inside metaworld branch before `generate_sequences`)
- Create: `project/SimpleVLA-RL/tests/smolvla_metaworld/test_ray_trainer_seed_override.py`

- [ ] **Step 1: Write failing contract test (source grep)**

```python
# tests/smolvla_metaworld/test_ray_trainer_seed_override.py
from pathlib import Path


def test_ray_trainer_applies_global_step_seed_schedule():
    src = Path("verl/trainer/ppo/ray_trainer.py").read_text()
    assert "metaworld_seed_scheduler" in src
    assert "seed_schedule" in src
    assert "train_trial_seeds_for_batch" in src
```

- [ ] **Step 2: Run test — FAIL**

- [ ] **Step 3: Patch `ray_trainer.py`**

After `newbatch: DataProto = DataProto.from_single_dict(batch_dict)` and metaworld detection, when `self.config.data.get("seed_schedule", "dataset") == "global_step"`:

```python
from verl.utils.metaworld_seed_scheduler import train_trial_seeds_for_batch

batch_size_rows = len(newbatch)
scheduled = train_trial_seeds_for_batch(
    train_seed_base=int(self.config.data.get("train_seed_base", 2000)),
    global_step=int(global_steps),
    batch_size=int(self.config.data.train_batch_size),
)
if batch_size_rows != len(scheduled):
    raise ValueError(
        f"metaworld batch rows {batch_size_rows} != train_batch_size seeds {len(scheduled)}"
    )
newbatch.batch["trial_seed"] = torch.tensor(
    [[s] for s in scheduled[:batch_size_rows]],
    dtype=torch.int64,
)
for i in range(batch_size_rows):
    newbatch.batch["trial_id"][i] = torch.tensor([i], dtype=torch.int64)
task_names = newbatch.non_tensor_batch.get(
    "task_name",
    np.array([self.config.data.task_suite_name] * batch_size_rows, dtype=object),
)
trial_seeds = newbatch.batch["trial_seed"].reshape(-1).detach().cpu().numpy().tolist()
newbatch.non_tensor_batch["uid"] = np.array(
    [f"{str(task_names[i]).replace('metaworld_', '')}:{int(trial_seeds[i])}" for i in range(batch_size_rows)],
    dtype=object,
)
```

Remove duplicate uid assignment later in same block (single assignment path).

- [ ] **Step 4: Run test — PASS**

- [ ] **Step 5: Commit**

---

### Task 3: Rollout — seed batch + reward modes

**Files:**
- Modify: `project/SimpleVLA-RL/verl/workers/rollout/smolvla_metaworld_rollout.py`
- Modify: `project/SimpleVLA-RL/verl/trainer/config/smolvla_metaworld_grpo.yaml`
- Create: `project/SimpleVLA-RL/tests/smolvla_metaworld/test_rollout_reward_modes.py`

- [ ] **Step 1: Failing test for reward helper**

```python
# tests/smolvla_metaworld/test_rollout_reward_modes.py
from verl.workers.rollout.smolvla_metaworld_rollout import trajectory_outcome_score


def test_trajectory_outcome_score_dense_return():
    class T:
        rewards = [1.0, 2.0]
        successes = [False]
    assert trajectory_outcome_score(T(), mode="dense_return", success_bonus=0.0) == 3.0


def test_trajectory_outcome_score_success_only():
    class T:
        rewards = [100.0]
        successes = [True]
    assert trajectory_outcome_score(T(), mode="success_only", success_bonus=0.0) == 1.0
```

- [ ] **Step 2: Run — FAIL**

- [ ] **Step 3: Implement**

Add `trajectory_outcome_score(traj, *, mode, success_bonus)` in rollout module:

```python
def trajectory_outcome_score(traj, *, mode: str, success_bonus: float) -> float:
    dense = float(sum(traj.rewards)) if traj.rewards else 0.0
    success = bool(any(traj.successes)) if traj.successes else False
    if mode == "dense_return":
        return dense + (float(success_bonus) if success and success_bonus else 0.0)
    if mode == "success_only":
        return 1.0 if success else 0.0
    raise ValueError(f"unknown reward mode: {mode}")
```

Replace line `scores[...] = float(sum(traj.rewards))` with call using `self._cfg("reward_mode", "dense_return")` and `success_bonus`.

For `len(seeds) > 1`, replace per-row loop with one call:

```python
from smolvla_grpo.phase11_rollout import collect_rollout_seed_batch

rollouts = collect_rollout_seed_batch(
    bundle=bundle,
    policy_old=policy_module,
    task=task,
    task_text=_resolve_task_text(task),
    reset_seeds=[int(s) for s in seeds],
    episode_index=int(prompts.meta_info.get("global_step", 0)),
    max_steps=max_steps,
    group_size=group_size,
    ...
)
# then flatten rollouts into tensors (same as today per-traj loop)
```

Pass `global_step` from `ray_trainer` via `gen_batch.meta_info["global_step"] = global_steps` before `generate_sequences`.

- [ ] **Step 4: YAML**

```yaml
data:
  train_batch_size: 4
  n_samples: 4
  train_seed_base: 2000
  eval_seed_base: 1000
  seed_schedule: global_step
  num_trials_per_task: 25   # val dataloader size; train uses scheduler

reward:
  mode: dense_return
  success_bonus: 0.0

trainer:
  total_epochs: 10       # per train chunk; launcher resumes through 40 total updates
  save_freq: 2
  test_freq: -1          # external PBS eval sweeps handle checkpoint selection
```

- [ ] **Step 5: Run tests — PASS**

- [ ] **Step 6: Commit**

---

### Task 4: Reward audit script (Gate before PBS)

**Files:**
- Create: `project/SimpleVLA-RL/scripts/grpo/audit_smolvla_reward_stats.py`

- [ ] **Step 1: Script prints per-uid stats**

Run 1 rollout batch (CPU or 1-GPU): dense return min/max/mean, success rate, count trajectories with zero advantage group (all same return).

```bash
cd project/SimpleVLA-RL
SMOLVLA_MAIN_TASK_LOCAL=1 python scripts/grpo/audit_smolvla_reward_stats.py \
  --config-name smolvla_metaworld_grpo \
  data.train_batch_size=4 data.n_samples=4 data.seed_schedule=global_step
```

Expected stdout includes lines like:
`uid=push-v3:2012 n=4 dense_mean=... success_rate=... adv_std=...`

- [ ] **Step 2: Assert dense magnitudes documented**

If dense mean >> 100 while success_rate ≈ 0, log warning: GRPO still valid but scale large — note for later advantage clip, not blocking.

- [ ] **Step 3: Commit**

---

### Task 5: Eval bridge + PBS sweep

**Files:**
- Create: `project/scripts/grpo/eval_verl_smolvla_checkpoint.py`
- Create: `project/scripts/grpo/eval_verl_smolvla_sweep.pbs`
- Modify: `project/SimpleVLA-RL/verl/trainer/config/smolvla_metaworld_grpo.yaml` — `num_trials_per_task: 25` for valid

- [ ] **Step 1: Eval loader for verl HF dir**

```python
# project/scripts/grpo/eval_verl_smolvla_checkpoint.py
# Load base HF bundle, then:
from safetensors.torch import load_file
state = load_file(verl_ckpt_dir / "model.safetensors")
bundle.policy.load_state_dict(state, strict=False)
# Reuse official_lerobot eval loop from eval_phase11_checkpoints.py (episodes, eval_seed_start)
```

CLI: `--base-checkpoint`, `--verl-actor-dir` (e.g. `.../global_step_10/actor`), `--episodes 25|100`, `--eval-seed-start 1000`, `--output-dir`.

- [ ] **Step 2: PBS sweep template**

```bash
# eval_verl_smolvla_sweep.pbs — after each 10-update training chunk ends
for step in "${MIN_UPDATE}" "${MIN_UPDATE_PLUS_2}" ... "${MAX_UPDATE}"; do
  python scripts/grpo/eval_verl_smolvla_checkpoint.py \
    --verl-actor-dir "${CKPT_ROOT}/global_step_${step}/actor" \
    --episodes 25 --eval-seed-start 1000 ...
done
# Final best step after all 40 updates: --episodes 100
```

- [ ] **Step 3: Smoke eval 5 episodes locally**

- [ ] **Step 4: Commit** (both repos if script lives under `project/` not `SimpleVLA-RL`)

---

### Task 6: PBS triplet chain launcher

**Files:**
- Create: `project/SimpleVLA-RL/scripts/grpo/submit_smolvla_learning_triplet.sh`
- Create/modify: `project/SimpleVLA-RL/scripts/grpo/run_smolvla_rl_metaworld_pushv3.pbs`
- Create: `project/SimpleVLA-RL/scripts/grpo/eval_verl_smolvla_sweep.pbs`

- [ ] **Step 1: Submit script**

```bash
#!/usr/bin/env bash
set -euo pipefail
cd /rds/general/user/aa6622/home/project/SimpleVLA-RL
Q=v1_gpu72
# For each experiment A/B/C, submit four train chunks.
# Each train chunk afterok submits both:
#   1) eval sweep for checkpoints just produced
#   2) next train chunk
train0=$(qsub -q "$Q" \
  -l "select=1:ncpus=48:mem=64gb:ngpus=1:gpu_type=RTX6000" \
  -l "walltime=02:00:00" \
  -v SMOLVLA_EXPERIMENT_NAME=pushv3_b4g4_dense_lr5e6,\
SMOLVLA_START_UPDATE=0,SMOLVLA_TOTAL_EPOCHS=10,SMOLVLA_SAVE_FREQ=2,SMOLVLA_TEST_FREQ=-1,\
SMOLVLA_GROUP_SIZE=4,SMOLVLA_TRAIN_BATCH_SIZE=4,SMOLVLA_LR=5e-6,SMOLVLA_CLIP=0.1 \
  scripts/grpo/run_smolvla_rl_metaworld_pushv3.pbs)
# eval 2..10 and train 10..20 both depend on train0.
# Repeat through 40 updates; repeat chain for Job B and C configs.
```

Extend `run_smolvla_rl_metaworld_pushv3.pbs` to pass:
`data.train_batch_size`, `data.train_seed_base`, `data.seed_schedule`, `data.eval_seed_base`, `actor_rollout_ref.actor.optim.lr`, clip overrides, `reward.mode`, `trainer.total_epochs=10`, `trainer.save_freq=2`, `trainer.test_freq=-1`, resume/start-update metadata, and ephemeral checkpoint root.

Submit eval sweeps with:
`-l select=1:ncpus=32:mem=32gb:ngpus=1:gpu_type=RTX6000`, `-l walltime=00:40:00`, `SMOLVLA_MIN_UPDATE`, `SMOLVLA_MAX_UPDATE`, `SMOLVLA_EVAL_EPISODES=25`.

- [ ] **Step 2: Wire `examples/run_smolvla_rl_metaworld_pushv3.sh`** to forward new Hydra overrides from env.

- [ ] **Step 3: Dry-run CPU tests**

`cd project/SimpleVLA-RL && python -m pytest tests/smolvla_metaworld/ -q`

- [ ] **Step 4: Submit triplet chains** (user/agent autonomous)

- [ ] **Step 5: Queue eval PBS dependent on each train chunk job id** and next train chunk off the same `afterok` dependency.

- [ ] **Step 6: Commit**

---

### Task 7: OOM / throughput guard for B=4

**Files:**
- Modify: `project/SimpleVLA-RL/verl/trainer/config/smolvla_metaworld_grpo.yaml`

- [ ] **Step 1: If 1-update GPU smoke OOMs**, reduce `rollout_env_batch_size` to 4 (not G) or keep `smolvla_train_chunk_micro_batch_size: 1`.

- [ ] **Step 2: 1-update PBS smoke** `SMOLVLA_TOTAL_EPOCHS=1`, `B=4`, `G=4`, same train resources (`48cpu/64gb/1xRTX6000`, wall `02:00:00`) before triplet chains.

- [ ] **Step 3: Commit if config changed**

---

## Post-triplet metrics (success criteria)

| Metric | Target |
|--------|--------|
| Train `train_reward/success_rate` | Uptrend over 20+ updates; continue through 40 |
| PBS 25-episode eval success | Beat 21% baseline ([ledger](docs/findings/2026-05-19-phase11-recent-run-ledger.md)) |
| Final 100-seed eval | Best checkpoint from 40-update chain ≥ 25% (stretch 30%) |
| `actor/ppo_kl` | No explosion; Job C should have lowest KL |
| Checkpoint sanity | `model.safetensors` loads in eval bridge |

---

## Execution order

```mermaid
flowchart LR
  T1[Task1 Scheduler] --> T2[Task2 RayTrainer]
  T2 --> T3[Task3 Rollout+Reward]
  T3 --> T4[Task4 Audit]
  T4 --> T5[Task5 Eval bridge]
  T5 --> T7[Task7 Smoke]
  T7 --> T6[Task6 Triplet Chains]
  T6 --> E1[Chunked 25-seed sweeps]
  E1 --> E2[100-seed final]
```

---

## Spec coverage self-review

| Requirement | Task |
|-------------|------|
| Unique train seeds per update | 1, 2 |
| B=4, G=4/8 | 3, 6 |
| 25-seed eval per checkpoint | 5, 6 |
| 100-seed final eval | 5 |
| Reward mapping check | 3, 4 |
| LR tune after data fix | 6 (B/C lr) |
| 3 parallel anti-collapse runs | 6 |
| Replica seed offset G | unchanged Phase11 (documented) |

---

## LR / KL follow-up (after triplet — not blocking launch)

- If dense advantage std huge: add `algorithm.adv_clip: 5.0` (new knob) or reward normalize per group.
- If KL drifts: `algorithm.kl_ctrl.kl_coef: 0.01` with ref policy enabled.
- Do **not** enable `filter_warmup` / zero-advantage skip until val success > 0 on 25 seeds.
