# ManiSkill PIRL Eval Suite

## Scope

- Run domain: ManiSkill pick-and-place.
- Task grid: 16 object types x 17 receptacles x 16 table scenes = 4,352 task combinations.
- Policy stack: pi0.5 with PPO + Flow-SDE.
- Target noise path: Flow-SDE, not Flow-Noise.
- Numeric baseline: video-off.
- Video sidecar: separate baseline capture of first successful SFT episode.

## Checkpoints

- SFT warm start: `RLinf/RLinf-Pi05-ManiSkill-25Main-SFT`.
- Local cache: `/vol/bitbucket/aa6622/eggroll/.cache/huggingface/rlinf/RLinf-Pi05-ManiSkill-25Main-SFT/`.
- Metadata:
  - `global_step=1000`
  - `name=pi05_maniskill`
  - `exp_name=PutOnPlateInScene25Main-v3`
- Use SFT checkpoint as warm start for RL. It is post-SFT, pre-RL.
- Do not confuse with RL-stage checkpoints:
  - `RLinf/RLinf-Pi05-ManiSkill-25Main-RL-FlowSDE`
  - `RLinf/RLinf-Pi05-ManiSkill-25Main-RL-FlowNoise`

## Data Contract

- SFT data: 16,384 MPLib episodes.
- Each SFT trajectory has exactly 15 terminal frames appended.
- If touching data pipeline, do not alter terminal-frame append count.

## RL Mechanics

- Algorithm: PPO + Flow-SDE + pi0.5.
- Flow-SDE chosen over Flow-Noise for target run.
- Flow-SDE knobs:
  - `noise_method=flow_sde`
  - `joint_logprob=False`
  - `noise_level=0.5`
  - `num_steps=4`
  - `action_horizon=8`
  - `num_action_chunks=5`
  - `entropy_bonus=0.0`
- Reward:
  - `1.0` for correct object placement.
  - Paper describes an auxiliary `0.1` reward for successful
    gripper-object attachment.
  - Current RLinf wrapper has shaped grasp-related terms in
    `rlinf/envs/maniskill/maniskill_env.py`; record exact resolved
    reward config before comparing runs.

## Batch And Resource Math

- Paper shape: `320` envs, `max_steps_per_rollout_epoch=80`, `num_action_chunks=5`, `rollout_epoch=1`.
- Paper samples/update: `320 * 1 * (80 / 5) = 5120`.
- 2x RTX6000 plan treats `320` envs as upper profiling bound, not target.
- Formula:

```text
train_samples =
  env.train.total_num_envs
  * algorithm.rollout_epoch
  * (env.train.max_steps_per_rollout_epoch / actor.model.num_action_chunks)
```

Tuple format below: `(envs, rollout_epoch, max_steps_per_rollout_epoch)`.

| Shape | Samples/update |
| --- | ---: |
| `(64, 5, 80)` | 5120 |
| `(80, 4, 80)` | 5120 |
| `(128, 5, 40)` | 5120 |
| `(160, 2, 80)` | 5120 |
| `(320, 1, 80)` | 5120 |

## Evaluation Suites

- ID (paper: IND): `Main-v3-train`
- Vision OOD:
  - `Instruct-v1-test`
  - `VisionImage-v1-test`
  - `VisionTexture03-v1-test`
  - `VisionTexture05-v1-test`
  - `VisionWhole03-v1-test`
  - `VisionWhole05-v1-test`
- Semantic OOD:
  - `MultiCarrot-v1-test`
  - `MultiCarrot-v1-train`
  - `MultiPlate-v1-test`
  - `MultiPlate-v1-train`
- Execution OOD:
  - `PositionChangeTo-v1-test`
  - `Position-v1-test`

## Paper Table 4 Reference Results For pi0.5 (Success Rate, Percent)

| Method | ID (paper: IND) | OOD Vision | Semantic | Execution | OOD Avg |
| --- | ---: | ---: | ---: | ---: | ---: |
| SFT | 40.1 | 40.2 | 16.6 | 22.4 | 26.4 |
| Flow-SDE | 90.9 | 68.0 | 34.5 | 45.4 | 49.3 |
| Flow-Noise reference | 89.7 | 69.9 | 35.5 | 54.9 | 53.4 |

## Eval Notes

- Keep `ignore_terminations` fixed across compared evals and record the value.
  Official Pi0.5 ManiSkill config uses `ignore_terminations=True`.
- Use `use_fixed_reset_state_ids=True` for paper-style fixed-state eval.
- If eval `total_num_envs < 320`, record state coverage and denominator.
- Keep video-off numeric baseline separate from video sidecar results.
