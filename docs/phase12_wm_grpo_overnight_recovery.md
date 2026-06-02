# Phase12 WM-GRPO Overnight Recovery

This note records the autonomous recovery contract for strict G8-u20 WM-GRPO runs.

## Pure-WM Thesis Protocol

Preferred thesis path is pure-WM training:

- Train with `--phase12-train-mode wm_only`.
- Use `--wm-only-root-mode oracle_teacher_forced` so each 5-step segment starts from an offline oracle root frame/proprio/proc.
- Score candidate chunks only with JEPA-WM latent progress toward the local subgoal.
- Do not use policy-selected real environment transitions as training feedback. Real MetaWorld use is limited to offline oracle/cache generation and evaluation.
- Primary RCA hparams match the old strict run: `lr=1e-5`, `clip_eps=0.2`, `init_log_std=-2.0`, `euler_step_noise_std=0.2`, `group_size=8`, `chunk_len=5`, `train_seed_base=2000`.
- Default pure-WM regularization is `--wm-action-l2-penalty 0.003`; keep the explicit `0.0` ablation separate.

Primary autonomous PBS chain:

- PBS: `scripts/grpo/phase12_pure_wm_teacher_forced_train_eval.pbs`
- Supervisor: `scripts/grpo/supervise_phase12_pure_wm_overnight.py`
- Supervisor PBS: `scripts/grpo/phase12_pure_wm_supervisor_loop.pbs`
- Train: 20 updates, `--save-every-list 2,5`
- Eval25: explicit updates `2,4,5,6,8,10,12,14,15,16,18,20`
- Eval100: explicit updates `5,10,15,20`
- Selection: prefer best eval100 checkpoint; use eval25 only to decide where to spend eval100 if future runs expand beyond the fixed shortlist.

## Primary Run

- PBS: `scripts/grpo/phase12_g8_u20_wm_train_eval100_stride5.pbs`
- Supervisor: `scripts/grpo/supervise_phase12_wm_grpo_overnight.py`
- Supervisor PBS: `scripts/grpo/phase12_wm_grpo_supervisor_loop.pbs`
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

For unattended overnight recovery, submit the supervisor as a non-GPU PBS job
after the training job:

```bash
qsub -W depend=afterany:<TRAIN_JOB_ID> \
  -v PHASE12_RUN_DIR=/rds/general/user/aa6622/home/project/artifacts/phase12_wm_g8_u20_strict_parity_20260602 \
  scripts/grpo/phase12_wm_grpo_supervisor_loop.pbs
```

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
