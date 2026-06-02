# Phase12 WM-GRPO Overnight Recovery

This note records the autonomous recovery contract for strict G8-u20 WM-GRPO runs.

## Primary Run

- PBS: `scripts/grpo/phase12_g8_u20_wm_train_eval100_stride5.pbs`
- Supervisor: `scripts/grpo/supervise_phase12_wm_grpo_overnight.py`
- Run dir: `artifacts/phase12_wm_g8_u20_strict_parity_20260602`
- Action profile: `bounded_executed`
- Train: `group_size=8`, `batch_size=1`, `chunk_len=5`, `num_updates=20`, `seed_base=2000`
- Eval100: updates `5,10,15,20`, seeds `1000..1099`, `chunk_len=5`

## Submit

```bash
cd /rds/general/user/aa6622/home/project
/rds/general/user/aa6622/home/.envs/lerobot_mw_py312/bin/python \
  scripts/grpo/supervise_phase12_wm_grpo_overnight.py \
  --submit
```

## Check / Resume

```bash
cd /rds/general/user/aa6622/home/project
/rds/general/user/aa6622/home/.envs/lerobot_mw_py312/bin/python \
  scripts/grpo/supervise_phase12_wm_grpo_overnight.py \
  --auto-resume
```

`--auto-resume` only resubmits known-safe walltime-style failures where a latest
checkpoint exists before `update_0020.pt`. Unknown failures stop with a blocker.

## Outputs

- `overnight_supervisor_state.json`: state machine, job ids, latest checkpoint.
- `overnight_root_cause_ledger.md`: timestamped diagnosis and action log.
- `progress.jsonl`: Phase12 update rows, including `action_profile`, `reward_key`,
  action clipping telemetry, WM timing, and memory telemetry.
- `eval100_u0005_0020_stride5_nenv25_async/eval_sweep_summary.json`: success curve.

## Failure Classes

- `pbs_walltime_timeout`: safe to resume from latest checkpoint.
- `hf_cache_failure`: safe for one resubmit after fixing cache env vars.
- `eval_cli_failure`: rerun eval only after flag mismatch fix.
- `oom_or_cuda_memory`: stop; reduce batch/scoring or inspect memory.
- `wm_load_failure`: stop; verify JEPA repo/checkpoint.
- `mujoco_egl_failure`: stop; inspect module/EGL setup.
- `nan_or_inf_loss`: stop; inspect optimizer/reward scale.
- `missing_checkpoint`: stop; train never reached checkpoint write.
- `unknown`: stop; full manual root-cause pass needed.
