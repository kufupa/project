# SmolVLA LeRobot package GPU import smoke — job 237046

**Date:** 2026-05-01  
**Slurm script:** [`scripts/slurm/smolvla_lerobot_pkg_import_gpu_smoke.slurm`](../scripts/slurm/smolvla_lerobot_pkg_import_gpu_smoke.slurm)  
**Submit dir:** `/vol/bitbucket/aa6622/project` (`sbatch --export=NIL …`)  
**Slurm job ID:** `237046`  
**Stdout log:** `smolvla_lerobot_pkg_import_gpu_smoke_237046.out` (in `project/` next to submit cwd)

## What ran

- Sourced [`scripts/slurm/common_env.sh`](../scripts/slurm/common_env.sh): `PROJECT_ROOT`, `PYTHONPATH`, pinned HF/torch caches under workspace `.cache`.
- Interpreter: `/vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python` (override with `LEROBOT_MW_PYTHON` if needed).
- Python work: **import** `SmolVLAPolicy` / `VLAFlowMatching`, print **CUDA** availability and device name, **inspect** `sample_actions` return type and presence of GRPO hooks. **No** `from_pretrained`, **no** Hugging Face checkpoint download, **no** full VLM forward pass.

## Did we load real SmolVLA weights?

**No.** This smoke only imports the **code** and checks **symbols/signatures**. It does not load `jadechoghari/smolvla_metaworld` or any other checkpoint. A follow-up gate should call `SmolVLAPolicy.from_pretrained(...)` (or your parity benchmark) under Slurm if you need weight + forward proof.

## Result

Log ends with `smolvla_gpu_symbol_check_ok`. Observed lines include `cuda_available True`, `cuda_device NVIDIA A16`, tuple return on `sample_actions`, and `select_action_distr_params` / `_get_distr_params_chunk` present.

## How long did it take?

**Not recorded in this workspace:** `sacct` against the Slurm DB returned `Connection refused` from the agent host, so elapsed time was not queried here. The job log has no timestamps; wall time is dominated by queue + allocation + import. For a rough order-of-magnitude: this is a **short** job (small `.out`, no model forward); expect **well under** the requested 30-minute limit once running.

## Re-run

```bash
cd /vol/bitbucket/aa6622/project
sbatch --export=NIL scripts/slurm/smolvla_lerobot_pkg_import_gpu_smoke.slurm
```

## Next (high level)

1. **Optional:** Add a second Slurm smoke that loads **base checkpoint** (`jadechoghari/smolvla_metaworld`) and runs one tiny **GPU forward** (policy or `sample_actions` with dummy batch), still under `--export=NIL` + `common_env.sh`.
2. **Phase11 GRPO:** Implement or harden `src/smolvla_grpo/policy_wrapper.py` (distribution hooks, `rsample` / noise handling per plan), `grpo_math`, rollout collector, trainer; run **API gate** and **forward** smokes under `scripts/grpo/` with **`/vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python`** (Phase11 primary interpreter per amended plan).
3. **Slurm hygiene:** Ensure every GRPO `sbatch` uses `--chdir=/vol/bitbucket/aa6622/project` (or equivalent) and pins caches like parity jobs.
