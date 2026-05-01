# SmolVLA pretrained GPU forward smoke — job 237048

**Checkpoint:** `jadechoghari/smolvla_metaworld` (same MetaWorld SmolVLA hub id as parity / Phase11)  
**Slurm script:** [`scripts/slurm/smolvla_pretrained_gpu_forward_smoke.slurm`](../scripts/slurm/smolvla_pretrained_gpu_forward_smoke.slurm)  
**Python driver:** [`scripts/slurm/run_smolvla_pretrained_gpu_forward_smoke.py`](../scripts/slurm/run_smolvla_pretrained_gpu_forward_smoke.py)  
**Submit:** `cd /vol/bitbucket/aa6622/project && sbatch --chdir=/vol/bitbucket/aa6622/project --export=NIL scripts/slurm/smolvla_pretrained_gpu_forward_smoke.slurm`  
**Job ID:** `237048`  
**Log:** `smolvla_pretrained_gpu_forward_smoke_237048.out` (under `project/`)

## What it proves (beyond import-only smoke)

- Hugging Face hub + pinned caches on a GPU node (`common_env.sh` line in log).
- `PolicyProcessorPipeline.from_pretrained` (pre/post) + `SmolVLAPolicy.from_pretrained` for the real checkpoint.
- One **`predict_action_chunk`** forward (patched `sample_actions` tuple path).
- One **`select_action_distr_params`** call (GRPO distribution queue path).

## Pass grep strings

- `smolvla_pretrained_load_ok`
- `smolvla_pretrained_forward_ok`
- `smolvla_pretrained_distr_params_ok`
- `smolvla_pretrained_gpu_forward_smoke_ok`

## Observed timings (this run, approximate)

- Bundle load (preprocessor + postprocessor + policy + VLM weights): **~215 s** on `gpuvm36` / A16.
- Forward `predict_action_chunk`: **~2.7 s**.
- `select_action_distr_params`: **~0.5 s**.

## Notes

- Log shows **`Missing key(s) when loading model: {'model.log_std'}`** — expected after the site-packages patch added `log_std`; the parameter is **initialized**, not loaded from the public checkpoint. Safe for this smoke; document if you ever `strict=True` load older checkpoints.

- Hub warned about **unauthenticated requests**; optional: `HF_TOKEN` for rate limits.

## Override checkpoint or Python

- `SMOLVLA_PRETRAINED_SMOKE_CHECKPOINT` — passed through the `.slurm` script to `--checkpoint`.
- `LEROBOT_MW_PYTHON` — defaults to `lerobot_mw_py310` venv python.

## Related

- Import-only sibling: [`2026-05-01-smolvla-lerobot-pkg-import-gpu-smoke-237046.md`](2026-05-01-smolvla-lerobot-pkg-import-gpu-smoke-237046.md)
