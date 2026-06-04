---
name: slurm-sbatch-runner
description: Use when Slurm sbatch jobs miss env under --export=NIL, resolve wrong project roots from spooled batch paths, need HF/torch cache pinning, chain phases from the login node, or debug jobs that produce no artifacts for many minutes.
---

# Slurm / sbatch runner (IC HPC)

## Core rules

1. **`.slurm` = one allocation worker.** It should `exec bash …` or run `python …`. It must **not** call `sbatch` again (nested submit is usually policy-blocked and brittle).
2. **Orchestration lives on the login node.** Use a normal `submit_*.sh` that runs `sbatch` multiple times with `--dependency=afterok:<jobid>` when you need phase chains. Same pattern as `scripts/mt10/submit_mt10_p6_p8_p9_prefixed.sh`.
3. **Many tasks in one phase:** prefer **one GPU job that loops** (example: `run_phase8_mt10.sh` over `MT10_TASK_IDS`) or a **Slurm job array**, not one sbatch per task from inside another allocation.
4. **`--export=NIL`:** set everything the batch script needs itself: `PYTHONPATH`, HF caches, optional `SMOLVLA_*`. Do not rely on inherited login-shell env.
5. **Project root:** Slurm may execute the batch file from `/var/spool/slurm/...`. Never trust `BASH_SOURCE` alone as the repo path. Use `SLURM_SUBMIT_DIR` only after validating a **marker file** under it, or set `PROJECT_ROOT` / `SMOLVLA_PROJECT_ROOT` explicitly. Shared helper: `scripts/slurm/common_env.sh`.
6. **Caches:** pin `HF_HOME`, `HUGGINGFACE_HUB_CACHE`, `TRANSFORMERS_CACHE`, `HF_DATASETS_CACHE`, `TORCH_HOME`, `XDG_CACHE_HOME` under `${WORKSPACE_ROOT}/.cache` so compute nodes see the same trees as MT10 / segment_grpo jobs.
7. **Live progress:** long evals may not write `run_manifest.json` until the end. Prefer `progress.jsonl` + stdout episode lines from `run_smolvla_eval` (disable JSONL with `SMOLVLA_EVAL_PROGRESS_JSONL=false` if needed). Monitor helper: `scripts/smolvla/monitor_smolvla_parity_jobs.py`.

## First-run checklist

- `bash -n` on the `.slurm` and sourced helpers.
- `sbatch --test-only --export=NIL …` from `project/`.
- Clean import smoke: `PYTHONPATH=$PROJECT_ROOT:$PROJECT_ROOT/src` + project venv `python -c "import smolvla_obs_state"`.
- Submit one job, confirm stdout shows cache line + `parity benchmark run dir:` + `smolvla_eval:` lines.

## References in this repo

- Shared env: `scripts/slurm/common_env.sh`
- Parity submitter (args after script): `scripts/smolvla/submit_smolvla_parity_eval.slurm`
- MT10 phase-8 single-job loop: `scripts/mt10/run_phase8_mt10.sh`, `scripts/mt10/submit_phase8_mt10.slurm`
- Login-node phase chain: `scripts/mt10/submit_mt10_p6_p8_p9_prefixed.sh`

See [examples.md](examples.md) for copy-paste commands.
