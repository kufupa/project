# RLinf πRL PPO Hyperparameters and MetaWorld Results

## Scope

This note records answers about:

- Hyperparameters used in RLinf πRL PPO runs.
- MetaWorld performance reported for π₀ / π₀.₅.
- Source links for those claims.

## πRL PPO Hyperparameters

Mostly from the πRL paper configs shipped as `examples/embodiment/config/*_ppo_openpi*.yaml`. The paper reports results; exact hyperparameters live in YAML configs plus π₀ docs.

Shared PPO core for LIBERO / MetaWorld flow-SDE style runs:

- Algorithm: PPO + GAE.
- `gamma=0.99`.
- `gae_lambda=0.95`.
- `rollout_epoch=8`.
- Train envs: `64`.
- `update_epoch=4`.
- PPO clip: `clip_ratio=0.2`.
- Value clip: `value_clip=0.2`.
- KL beta: `kl_beta=0`.
- Batch: `micro_batch_size=128`, `global_batch_size=2048`.
- Optimizer: Adam `beta1=0.9`, `beta2=0.95`, `weight_decay=0.01`.
- Action chunks: `num_action_chunks=5`.
- Reward/logprob: chunk-level for LIBERO / MetaWorld.

π-specific differences versus our SmolVLA direct path:

- Not plain Gaussian PPO. It fine-tunes flow-matching π policies.
- Flow-SDE: `noise_method=flow_sde`, `noise_level=0.5`, `entropy_bonus=0`, `joint_logprob=False`.
- Flow-Noise: `noise_method=flow_noise`, `entropy_bonus=0.005`, `joint_logprob=True`.
- Critic: `add_value_head=True`, `detach_critic_input=True`.
- LoRA optional: rank 8.

Learning rates by benchmark:

| Benchmark | Actor LR | Value LR | Config |
|---|---:|---:|---|
| LIBERO | `5e-6` | `1e-4` | `libero_spatial_ppo_openpi.yaml` |
| MetaWorld MT50 | `1e-5` | `1e-4` | `metaworld_50_ppo_openpi.yaml` |
| ManiSkill flow-noise | `7.91e-6` | `1.55e-4` | `maniskill_ppo_openpi.yaml` |

Infrastructure:

- Typical layout: 8 GPUs.
- Env: GPUs 0-3.
- Rollout: GPUs 4-7.
- Actor: GPUs 0-7.
- `pipeline_stage_num=2` for rollout/env overlap.
- FSDP actor, distributed RLinf path.

Comparison to current SmolVLA direct runs:

- Our runs are much smaller: 4-8 envs, `update_epochs=2`, `minibatch_envs=4`, actor LR `1e-6` / `3e-7`.
- Our path has no flow-SDE / flow-noise machinery, no 2048 global batch, and no multi-GPU pipeline.
- Same broad family: PPO + GAE. Not same recipe as πRL paper runs.

Bottom line: paper results used 64-env x 8 rollout epochs x 4 PPO epochs x big batch plus flow-SDE / flow-noise π₀ fine-tuning. It was not only “increase num_envs”.

## MetaWorld Performance

The headline result: **π₀ + PPO Flow-Noise reaches 85.8% average success on MetaWorld MT50**.

πRL public result table:

| Model | Setting | MT50 Avg Success |
|---|---:|---:|
| π₀ SFT | before RL | 50.8% |
| π₀ + PPO Flow-SDE | RL | 78.1% |
| π₀ + PPO Flow-Noise | RL | 85.8% |
| π₀.₅ SFT | before RL | 43.8% |
| π₀.₅ + PPO Flow-SDE | RL | 70.7% |
| π₀.₅ + PPO Flow-Noise | RL | 66.1% |

MetaWorld docs also break down the π₀ Flow-SDE result by difficulty:

| Method | Easy | Medium | Hard | Very Hard | Avg |
|---|---:|---:|---:|---:|---:|
| SmolVLA | 87.1 | 51.8 | 70.0 | 64.0 | 68.2 |
| π₀ | 77.9 | 51.8 | 53.3 | 20.0 | 50.8 |
| π₀ + PPO | 92.1 | 74.6 | 61.7 | 84.0 | 78.1 |
| π₀.₅ | 68.2 | 37.3 | 41.7 | 28.0 | 43.8 |
| π₀.₅ + PPO | 86.4 | 55.5 | 75.0 | 66.0 | 70.7 |

## Sources

- πRL result page: <https://rlinf.readthedocs.io/en/release-v0.2/rst_source/publications/pi_rl.html>
- MetaWorld benchmark page: <https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/metaworld.html#metaworld-results>
- πRL paper link from docs: <https://arxiv.org/abs/2510.25889>
- Released best MetaWorld π₀ Flow-Noise model: <https://huggingface.co/RLinf/RLinf-Pi0-MetaWorld-RL-FlowNoise>
- Local docs checked:
  - `project/RLinf/docs/source-en/rst_source/publications/pi_rl.rst`
  - `project/RLinf/docs/source-en/rst_source/examples/embodied/pi0.rst`
  - `project/RLinf/examples/embodiment/config/metaworld_50_ppo_openpi.yaml`
  - `project/RLinf/examples/embodiment/config/libero_spatial_ppo_openpi.yaml`
  - `project/RLinf/examples/embodiment/config/maniskill_ppo_openpi.yaml`

