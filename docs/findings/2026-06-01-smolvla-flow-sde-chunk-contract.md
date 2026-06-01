# SmolVLA Flow-SDE Chunk Contract

## Verdict

Flow-SDE is implementable for SmolVLA because SmolVLA denoises full padded action chunks in flow-matching space. The current GRPO implementation is a first-action spike and not the correct final objective.

## Corrected Assumptions

- Public MetaWorld checkpoint evidence says `chunk_size=50`, `n_action_steps=1`.
- Training with `n_action_steps=chunk_len` is an intentional RL experiment, not checkpoint-native eval reproduction.
- Default chunk mode uses `chunk_len=5` and loads SmolVLA with `n_action_steps=5`.
- Current `--chunk-size` in `train_phase11_env_on_policy_grpo.py` is optimizer grouping only.
- Correct Flow-SDE GRPO unit is one sampled action chunk, one executed chunk, one chunk-summed logprob ratio.

## Source Files

- `src/smolvla_pipeline/evaluator.py`: `_load_smolvla_bundle(..., n_action_steps=1)`.
- `src/smolvla_grpo/policy_wrapper.py`: current first-action Flow-SDE scoring and broken chunk sampler.
- `src/smolvla_grpo/phase11_rollout.py`: current one-action-per-env-step rollout.
- `.envs/lerobot_mw_py310/lib/python3.12/site-packages/lerobot/policies/smolvla/modeling_smolvla.py`: full padded chunk denoise and local Flow-SDE hook.
- `RLinf-smolvla-metaworld-ppo-grpo/scripts/run_smolvla_metaworld_direct_ppo.py`: masked chunk logprob reference.
- `RLinf-smolvla-metaworld-ppo-grpo/rlinf/envs/metaworld/smolvla_metaworld_env.py`: valid action mask reference.

## First Implementation Scope

Implement serial official-LeRobot chunk rollout first with default `chunk_len=5`. Vector chunk rollout is deferred until Flow-SDE parity and 4-update diagnostic pass. After user approval, run implementation, tests, Slurm smoke, train16, and eval25 autonomously; if anything breaks, perform RCA, fix, rerun the smallest gate, and continue.
