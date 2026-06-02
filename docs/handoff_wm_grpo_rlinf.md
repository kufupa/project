# WM-GRPO / RLinf Handoff

## Current Decision

Use Phase12 as the production WM-GRPO path first. It already owns SmolVLA
chunk sampling, JEPA-WM scoring, GRPO update math, PBS launchers, checkpointing,
and evaluation sweeps. RLinf integration stays thin until Phase12 behavior is
stable enough to port into Ray workers without changing reward semantics.

## Action Contract

- `official_jepa_mirror`: execute and score raw post-processed SmolVLA actions.
- `bounded_executed`: clip actions to env bounds, execute clipped actions, score
  the same clipped actions in JEPA-WM.
- Phase57 decode results motivate `bounded_executed` for the strict WM parity
  run, but `official_jepa_mirror` remains available for ablation.
- Trainer progress rows must log `action_profile`, `reward_key`, `chunk_len`,
  `action_clip_fraction`, `action_clip_any_fraction`, `raw_action_max_abs`,
  `clipped_action_max_abs`, and `clip_delta_max_abs`.

## Training Contract

- Strict G8-u20 WM parity uses `group_size=8`, `batch_size=1`, `chunk_len=5`,
  `num_updates=20`, `seed_base=2000`, `clip_eps=0.2`, `lr=1e-5`.
- Run dir: `artifacts/phase12_wm_g8_u20_strict_parity_20260602`.
- PBS train+eval entrypoint:
  `scripts/grpo/phase12_g8_u20_wm_train_eval100_stride5.pbs`.
- Overnight supervisor:
  `scripts/grpo/supervise_phase12_wm_grpo_overnight.py`.

## RLinf Boundary

Initial RLinf work should launch or consume Phase12 runs rather than rewrite WM
reward inside Ray. Required bridge data:

- policy checkpoint path and update index,
- action profile and action bounds,
- selected candidate chunk and raw/clipped action tensors,
- WM latent reward key and score metadata,
- evaluation summary rows.

Ray-native WM reward remains a spike. Gate it on stable Phase12 telemetry and
clear storage for both raw post-processed and clipped executed action chunks.

## Safety Rules

- Unknown supervisor failures block instead of requeueing.
- Auto-resume requires scoped walltime evidence and a resume checkpoint.
- Eval-only recovery submits at most once per final checkpoint.
- Long-running monitoring must run under PBS, not on login nodes.
# WM-GRPO / RLinf Handoff

## Current Decision

Use Phase12 as the production WM-GRPO path first. It already owns SmolVLA
chunk sampling, JEPA-WM scoring, GRPO update math, PBS launchers, checkpointing,
and evaluation sweeps. RLinf integration stays thin until Phase12 behavior is
stable enough to port into Ray workers without changing reward semantics.

## Action Contract

- `official_jepa_mirror`: execute and score raw post-processed SmolVLA actions.
- `bounded_executed`: clip actions to env bounds, execute clipped actions, score
  the same clipped actions in JEPA-WM.
- Phase57 decode results motivate `bounded_executed` for the strict WM parity
  run, but `official_jepa_mirror` remains available for ablation.
- Trainer progress rows must log `action_profile`, `reward_key`, `chunk_len`,
  `action_clip_fraction`, `action_clip_any_fraction`, `raw_action_max_abs`,
  `clipped_action_max_abs`, and `clip_delta_max_abs`.

## Training Contract

- Strict G8-u20 WM parity uses `group_size=8`, `batch_size=1`, `chunk_len=5`,
  `num_updates=20`, `seed_base=2000`, `clip_eps=0.2`, `lr=1e-5`.
- Run dir: `artifacts/phase12_wm_g8_u20_strict_parity_20260602`.
- PBS train+eval entrypoint:
  `scripts/grpo/phase12_g8_u20_wm_train_eval100_stride5.pbs`.
- Overnight supervisor:
  `scripts/grpo/supervise_phase12_wm_grpo_overnight.py`.

## RLinf Boundary

Initial RLinf work should launch or consume Phase12 runs rather than rewrite WM
reward inside Ray. Required bridge data:

- policy checkpoint path and update index,
- action profile and action bounds,
- selected candidate chunk and raw/clipped action tensors,
- WM latent reward key and score metadata,
- evaluation summary rows.

Ray-native WM reward remains a spike. Gate it on stable Phase12 telemetry and
clear storage for both raw post-processed and clipped executed action chunks.

## Safety Rules

- Unknown supervisor failures block instead of requeueing.
- Auto-resume requires scoped walltime evidence and a resume checkpoint.
- Eval-only recovery submits at most once per final checkpoint.
- Long-running monitoring must run under PBS, not on login nodes.
