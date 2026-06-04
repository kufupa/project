# Phase46 Logprob Action Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a two-way GRPO ablation that compares corrected executed-action Gaussian logprob against the old G8 unsquashed-action Gaussian logprob under identical seeds.

**Architecture:** Add an explicit `gaussian_logprob_action` switch to the SmolVLA GRPO policy wrapper, pass it through rollout collection and the trainer CLI, then submit two independent SLURM jobs. Each job trains 5 updates and immediately runs the same RLinf tiered eval on its produced checkpoints.

**Tech Stack:** Python, PyTorch, pytest, Bash, SLURM, RLinf SmolVLA MetaWorld eval.

---

### Task 1: Add Logprob Action Switch

**Files:**
- Modify: `src/smolvla_grpo/policy_wrapper.py`
- Modify: `src/smolvla_grpo/phase11_rollout.py`
- Modify: `src/smolvla_grpo/official_lerobot_vector_rollout.py`
- Modify: `scripts/grpo/train_phase11_env_on_policy_grpo.py`
- Test: `tests/test_grpo_logprob_correctness.py`

- [ ] **Step 1: Write the failing tests**

Add tests proving the default scores clipped executed action and `gaussian_logprob_action="unsquashed"` scores the sampled action.

- [ ] **Step 2: Run the focused test and confirm RED**

Run: `/vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python -m pytest tests/test_grpo_logprob_correctness.py -q`

Expected: fail because `gaussian_logprob_action` is not accepted yet.

- [ ] **Step 3: Implement the minimal switch**

Add `gaussian_logprob_action` with choices `executed` and `unsquashed`; default `executed`. In no-tanh Gaussian mode, store/recompute logprob against executed action for `executed`, and unsquashed action for `unsquashed`. Preserve tanh ablation behavior.

- [ ] **Step 4: Run the focused test and confirm GREEN**

Run: `/vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python -m pytest tests/test_grpo_logprob_correctness.py -q`

Expected: all tests pass.

### Task 2: Add Train+Eval SLURM Ablation Job

**Files:**
- Create: `scripts/grpo/submit_phase46_logprob_ablation_a30.slurm`

- [ ] **Step 1: Write the batch script**

The script accepts `executed` or `unsquashed`, trains 5 updates with G8 hyperparameters, saves every update, then runs `RLinf-smolvla-metaworld-ppo-grpo/scripts/run_phase46_tiered_eval_rlinf.sh` with first update `1` and last update `5`.

- [ ] **Step 2: Validate shell syntax**

Run: `bash -n scripts/grpo/submit_phase46_logprob_ablation_a30.slurm`

Expected: exit 0.

- [ ] **Step 3: Validate SLURM submission shape**

Run: `sbatch --test-only --chdir=/vol/bitbucket/aa6622/project --export=NIL scripts/grpo/submit_phase46_logprob_ablation_a30.slurm executed`

Expected: SLURM accepts the job specification.

### Task 3: Submit Two Parallel Jobs

**Files:**
- Runtime artifacts under `artifacts/phase46_logprob_ablation/<stamp>/{executed,unsquashed}`

- [ ] **Step 1: Submit executed-action job**

Run: `sbatch --parsable --chdir=/vol/bitbucket/aa6622/project --export=NIL scripts/grpo/submit_phase46_logprob_ablation_a30.slurm executed <outdir>/executed`

- [ ] **Step 2: Submit unsquashed-action job**

Run: `sbatch --parsable --chdir=/vol/bitbucket/aa6622/project --export=NIL scripts/grpo/submit_phase46_logprob_ablation_a30.slurm unsquashed <outdir>/unsquashed`

- [ ] **Step 3: Record status**

Run: `squeue -j <executed_job>,<unsquashed_job>`

Expected: both jobs are queued or running, each requesting one GPU.

### Eval Readout

Each job writes `eval/tiered_eval_summary.json`. Primary decision signal is:

- If `unsquashed` recovers G8-like train/eval behavior while `executed` collapses, A.3 objective change is confirmed as root cause.
- If both collapse, the culprit is elsewhere after G8 or the old 33% claim was eval-protocol noise.
- If both recover, Phase46 collapse was likely run-specific or downstream eval mismatch.
