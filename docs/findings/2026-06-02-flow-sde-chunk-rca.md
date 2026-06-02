# Flow-SDE Chunk GRPO RCA

Date: 2026-06-02

## Scope

Run inspected:

- Train: `artifacts/flow_sde_chunk_grpo_train16/246503`
- Eval: `artifacts/flow_sde_chunk_grpo_eval25/246503/eval25_summary.json`
- Config: `push-v3`, `rollout_unit=chunk`, `rollout_chunk_len=5`, `group_size=8`, `num_updates=16`, `max_steps=120`, `logprob_mode=flow_sde`, `flow_sde_noise_level=0.5`, `flow_sde_trace_step=-1`

## Root Cause With Evidence

The strongest failure mode is reward/objective mismatch: the trainer used raw dense MetaWorld return for GRPO advantages, so high-dense failures were often ranked above successful episodes.

Parsed from `progress.jsonl`:

- 16 training updates, 128 rollout episodes.
- 28 successful episodes and 100 failed episodes.
- 9 successful episodes received negative advantages.
- 17 failed episodes received positive advantages.
- In 4 updates, the maximum failed dense return exceeded the maximum successful dense return.

Examples:

- Update 1: one successful rollout had return `39.75` and advantage `-0.278`; one failed rollout had return `663.95` and advantage `2.471`.
- Update 12: one successful rollout had return `33.29` and advantage `0.025`; one failed rollout had return `95.00` and advantage `2.297`.
- Update 15: successful rollout returns peaked at `197.86`; a failed rollout reached `707.84` and advantage `2.367`.

This means the optimizer was sometimes explicitly reinforcing non-success behavior over success behavior. That is enough to explain oscillation/stalling without requiring a Flow-SDE numerical parity bug.

## Supporting Observations

Eval did not improve beyond baseline:

- Baseline: `28%` success.
- Update 2: `12%`.
- Update 4: `24%`.
- Update 6: `24%`.
- Update 8: `20%`.
- Update 10: `28%`.
- Update 12: `16%`.
- Update 14: `16%`.
- Update 16: `28%`.

Flow-SDE parity itself was not the observed failure in this run:

- Logged `approx_kl=0.0`, `ratio_mean=1.0` at sampled updates before optimization.
- Logged parity checks stayed within tolerance.

Secondary risks still worth tracking:

- Flow-SDE scoring is tied to denoise trace transitions, not a conventional Gaussian action distribution over final executed actions.
- Eval uses the RLinf sweep path, so train/eval checkpoint resolution must be audited when comparing runs.
- Dense reward can still be useful, but it needs an ablation against success-aligned rewards because its per-episode ranking is demonstrably misaligned here.

## Fix Direction

Run a direct 10-update ablation under identical Flow-SDE chunk settings:

- `dense_return`: legacy behavior, to test whether dense is genuinely better in this implementation.
- `sparse_success_delta`: RLinf-style success-aligned signal.
- `success_first_dense`: success dominates; dense return only breaks ties.

Each run must log reward mode, success advantage alignment, and max success/failure returns so the next RCA does not rely on manual parsing.
