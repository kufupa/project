# Official LeRobot MetaWorld + GRPO Findings

Date: 2026-05-06

Purpose: thesis notes for SmolVLA MetaWorld benchmarking and planned GRPO integration.

## Short Verdict

- Official benchmark path now: LeRobot `lerobot-eval` + `env.type=metaworld`.
- Official output prefix: `MT50_Phase071`.
- Custom Phase07 evaluator: legacy/custom. Useful for debugging, not canonical reproduction.
- GRPO path currently not official. It uses raw/custom MetaWorld wrappers.
- Best GRPO fix: thin official rollout adapter around LeRobot `MetaworldEnv`.
- Do not put `lerobot-eval` inside GRPO trainer. Eval CLI lacks GRPO training data: old logprobs, unsquashed actions, stored processed observations.
- Adapter should reuse Phase071 eval preprocessing:
  - `MetaworldEnv.reset/step()`
  - `lerobot.envs.utils.preprocess_observation(obs)`
  - add task text from `env.task_description`
  - `bundle.preprocessor(...)`
  - GRPO `select_action_distr_params(...)`

## What Was Verified

- Smoke run used SmolVLA checkpoint:
  - Checkpoint: `/vol/bitbucket/aa6622/.cache/huggingface/hub/models--jadechoghari--smolvla_metaworld/snapshots/ef3089ecb84eeeb7d33fedab24f6c76180a68900`
  - Command path: `/vol/bitbucket/aa6622/project/scripts/mt50/run_official_lerobot_mt50_eval.sh`
  - Job: `239025`
  - Log: `/vol/bitbucket/aa6622/project/mt50_phase071_official_1task_239025.out`
- Smoke task:
  - `assembly-v3`
  - `1` episode
  - seed `1000`
  - official LeRobot env horizon, logged as expected `500`
  - video on
- Smoke artifacts:
  - `/vol/bitbucket/aa6622/project/artifacts/MT50_Phase071_official_lerobot_1task_1ep/eval_info.json`
  - `/vol/bitbucket/aa6622/project/artifacts/MT50_Phase071_official_lerobot_1task_1ep/videos/assembly-v3_0/eval_episode_0.mp4`
  - `/vol/bitbucket/aa6622/project/artifacts/MT50_Phase071_official_lerobot_1task_1ep/MT50_Phase071_official_index.json`
- Smoke result:
  - `avg_sum_reward`: `249.96597450084346`
  - `avg_max_reward`: `10.0`
  - `pc_success`: `100.0`
  - `n_episodes`: `1`
  - video path present

## Official LeRobot MetaWorld Semantics

- Source of truth in installed venv:
  - `/vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages/lerobot/envs/metaworld.py`
- `MetaworldEnv.__init__` accepts:
  - `task`
  - `camera_name="corner2"`
  - `obs_type="pixels"`
  - `render_mode="rgb_array"`
  - `observation_width=480`
  - `observation_height=480`
- For GRPO use:
  - `obs_type="pixels_agent_pos"`
  - returns image + 4-D proprio state
- Official action space:
  - `Box(low=-1, high=1, shape=(4,), dtype=float32)`
- Official observation modes:
  - `pixels`: `{"pixels": uint8[480,480,3]}`
  - `pixels_agent_pos`: `{"pixels": uint8[480,480,3], "agent_pos": float64[4]}`
  - `state`: not implemented
- Official task setup:
  - `metaworld.MT1(env_name, seed=42)`
  - `env = mt1.train_classes[env_name](render_mode="rgb_array", camera_name="corner2")`
  - `env.set_task(mt1.train_tasks[0])`
  - camera patch for `corner2`: `[0.75, 0.075, 0.7]`
  - `env.reset()`
  - `env._freeze_rand_vec = False`
- Important meaning:
  - `_freeze_rand_vec = False` enables seed-based randomization after task setup.
  - Old custom backend missed this, causing semantic drift.
- Official horizon:
  - `self._max_episode_steps = self._env.max_path_length`
  - v3 tasks observed as `500`
  - LeRobot eval reads via `env.call("_max_episode_steps")[0]`
- Official success/termination:
  - raw MetaWorld returns `info["success"]`
  - LeRobot maps to `info["is_success"]`
  - `terminated = done or is_success`
  - on termination, LeRobot adds `info["final_info"]`
  - then LeRobot wrapper calls `self.reset()`
- GRPO implication:
  - must break immediately on `terminated or truncated`
  - must not use returned obs after terminal step for another action

## Official LeRobot Eval Path

- Installed evaluator:
  - `/vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages/lerobot/scripts/lerobot_eval.py`
- Eval construction:
  - `make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)`
  - MetaWorld returns nested env map: `{task_group: {task_id: VectorEnv}}`
- Eval rollout:
  - `observation, info = env.reset(seed=seeds)`
  - `max_steps = env.call("_max_episode_steps")[0]`
  - loop until all envs done or `max_steps`
  - `preprocess_observation(observation)`
  - `add_envs_task(env, observation)`
  - `env_preprocessor(observation)`; MetaWorld has no special env processor
  - `bundle/policy preprocessor`
  - `policy.select_action(...)`
  - `postprocessor(action)`
  - `env.step(action_numpy)`
- Video:
  - local `eval_main` hardcodes `max_episodes_rendered=10`
  - `videos_dir = Path(cfg.output_dir) / "videos"`
  - per-task dirs like `videos/assembly-v3_0/eval_episode_0.mp4`
- Metrics:
  - `eval_info.json`
  - `per_task`
  - `per_group`
  - `overall`
  - `pc_success`
  - `avg_sum_reward`
  - `avg_max_reward`
  - `video_paths`

## Official Preprocessing Facts

- LeRobot utility:
  - `lerobot.envs.utils.preprocess_observation`
- Mapping:
  - raw `pixels` -> `observation.image`
  - raw `agent_pos` -> `observation.state`
  - raw `environment_state` -> `observation.environment_state`
- Image processing:
  - expects HWC uint8
  - converts to BCHW float32 in `[0,1]`
- State processing:
  - `agent_pos` becomes float tensor with batch dim
- Task text:
  - `add_envs_task()` reads `env.task_description`
  - adds `observation["task"]`
- SmolVLA preprocessor then:
  - renames if needed
  - adds batch dimension if needed
  - appends newline to task
  - tokenizes task
  - moves tensors to device
  - normalizes state/action where stats exist

## Environment State Finding

- Checkpoint config lists:
  - `observation.state`: shape `[4]`
  - `observation.environment_state`: shape `[39]`
  - `observation.image`: shape `[3,480,480]`
- But SmolVLA action path uses:
  - images
  - `OBS_STATE`
  - language tokens/masks
- `modeling_smolvla.py` does not use `OBS_ENV_STATE` in action selection path.
- Official eval succeeded without explicit `observation.environment_state`.
- Therefore:
  - adapter should not synthesize/pad `observation.environment_state` by default
  - reuse official `preprocess_observation(obs)`
  - only add env-state if future test proves needed
- This is more faithful than manually padding 39-D env state.

## Benchmark Protocol Notes

- SmolVLA/LeRobot MetaWorld benchmark is not strict original Farama MT50 RL eval.
- Practical LeRobot protocol:
  - task names are MetaWorld v3 tasks, e.g. `assembly-v3`
  - `env.task` accepts one task, difficulty group, or comma-separated tasks
  - `eval.batch_size` controls vector env copies
  - `eval.n_episodes` controls episodes per task
  - official docs recommend `10` episodes per task for reproducible benchmarking
- Full LeRobot MT50-style evaluation:
  - `50` tasks
  - `10` episodes each if reproducing recommended benchmark
  - `500` total episodes
  - horizon `500`
- Smoke / quick reproduction:
  - `50` tasks x `1` episode = paper-style quick check but not stable benchmark
  - `1` task x `1` episode = smoke only
- Strict original Farama MT50 RL protocol:
  - `50` tasks
  - often `50` goal positions per task
  - `2500` episodes total
  - not what current LeRobot SmolVLA eval wrapper is doing
- Thesis wording:
  - say "LeRobot MetaWorld MT50-style evaluation"
  - do not claim strict Farama MT50 unless running full task-goal protocol

## Current Custom Phase07 Drift

- Old custom evaluator path:
  - `src/smolvla_pipeline/evaluator.py`
  - `_LeRobotMetaWorldBackend`
- Name is misleading:
  - uses raw MetaWorld
  - not official `lerobot.envs.metaworld.MetaworldEnv`
- Known drift:
  - missed `env._freeze_rand_vec = False`
  - old defaults often `120` steps
  - richer artifacts but non-official semantics
- Policy:
  - keep artifacts historical
  - label scripts/results legacy/custom
  - do not compare numerically against `MT50_Phase071` without caveat

## Current GRPO Pipeline

- Main trainer:
  - `scripts/grpo/train_phase11_env_on_policy_grpo.py`
- Rollout collection:
  - `src/smolvla_grpo/phase11_rollout.py`
  - `collect_rollout_group()`
  - `PushV3GRPOEnv`
- Policy wrapper:
  - `src/smolvla_grpo/policy_wrapper.py`
  - `MetaWorldSmolVLAGRPOPolicy`
- Math:
  - `src/smolvla_grpo/grpo_math.py`
  - group-relative advantages
  - clipped GRPO/PPO-style ratio loss
- Rewards:
  - `src/smolvla_grpo/reward_backends.py`
  - `EnvRewardBackend` sums env rewards
- Checkpoint eval:
  - `scripts/grpo/eval_phase11_checkpoints.py`
  - currently custom evaluator backend
- Slurm:
  - `scripts/grpo/submit_phase11_grpo.slurm`

## Current GRPO Drift From Official

- `PushV3GRPOEnv` uses raw `metaworld.MT1(task)`.
- It does not use official `MetaworldEnv`.
- It does not mirror official `_freeze_rand_vec = False`.
- It cycles tasks by `episode_index % len(train_tasks)`.
- Official LeRobot uses `train_tasks[0]`.
- It builds policy inputs by flattening raw obs and rendering raw env.
- Official obs is dict: `pixels`, `agent_pos`.
- Naive swap would flatten image pixels into state. Bad.
- Current path flips `corner2` manually.
- Official LeRobot already flips `corner2`.
- Naive official swap could double flip.
- Current path records success but breaks only on `terminated or truncated`.
- Official path terminates on success.
- Current GRPO can keep accumulating rewards after success.
- This changes training signal.

## Why Adapter Is Best

- GRPO needs data `lerobot-eval` does not expose:
  - processed observations per step
  - old-policy log probabilities
  - unsquashed sampled actions
  - stochastic action distribution parameters
  - group returns
- `lerobot-eval` is evaluator, not online RL collector.
- Full LeRobot train-loop port is too invasive:
  - old policy copy
  - group sampling
  - custom loss
  - trainable subset
  - checkpoint flow
  - Slurm flow
- Raw MetaWorld patching is fragile:
  - duplicates official env semantics
  - risks more silent drift
- Thin adapter is clean boundary:
  - official env semantics from LeRobot
  - GRPO data contract from our trainer
  - minimal blast radius

## Better Adapter Design

- Name idea:
  - `OfficialLeRobotMetaWorldGRPOEnv`
  - or `LeRobotMetaWorldRolloutAdapter`
- Core env:
  - `from lerobot.envs.metaworld import MetaworldEnv`
  - `MetaworldEnv(task=task, obs_type="pixels_agent_pos", render_mode="rgb_array")`
- Reset:
  - `obs, info = env.reset(seed=reset_seed)`
  - use `obs` directly
- Step:
  - action is 1-D float32 shape `(4,)`
  - `obs, reward, terminated, truncated, info = env.step(action)`
  - success from `info["is_success"]` or terminal `info["final_info"]["is_success"]`
  - break immediately on `terminated or truncated`
- Proc batch:
  - `proc_obs = preprocess_observation(obs)`
  - `proc_obs["task"] = task_description`
  - `proc = bundle.preprocessor(proc_obs)`
- No manual corner2 flip.
- No pixel flattening.
- No default zero-padded env_state.
- Horizon:
  - if `--max-steps 0`, use `env._max_episode_steps`
  - for official MetaWorld this should be `500`

## Safe-Robot-Steering Repo: Useful Parts

- Path:
  - `/vol/bitbucket/aa6622/VGG JEPA/safe-robot-steering`
- Handoff:
  - `/vol/bitbucket/aa6622/VGG JEPA/safe-robot-steering/HANDOFF_GRPO_REPRODUCTION.md`
- Useful pattern:
  - on-policy data only, no offline dataset
  - group rollouts from same task/prompt
  - store obs
  - store unsquashed actions
  - store old logprobs
  - compute group-relative advantages
  - use clipped ratio objective
  - refresh `policy_old` after updates
- Useful hyperparams:
  - `MAX_STEPS = 520`
  - `GROUP_SIZE = 4`
  - `UPDATE_EPOCHS = 2`
  - `UPDATE_CHUNK_SIZE = 5`
  - `EULER_STEP_NOISE_STD = 0.2`
  - `INIT_LOG_STD = -2`
  - `GRPO_EPSILON = 0.2`
  - `lr = 1e-5`
- Useful warning:
  - rollout throughput is expensive
  - low reward diversity can cause zero-advantage skipped updates

## Safe-Robot-Steering Repo: Caveats / Bugs / Non-Portable Pieces

- LIBERO-specific, not MetaWorld.
- Action dim:
  - LIBERO: `7`
  - MetaWorld: `4`
- Reward:
  - LIBERO sparse-ish success reward
  - MetaWorld dense reward + success flag
- Success:
  - safe-robot uses `total_reward == 1`
  - invalid for MetaWorld
- Env:
  - `make_libero_env()`
  - LIBERO task suites and init states
  - not reusable for MetaWorld
- Preprocessing:
  - manual image/state/token mapping
  - bypasses official LeRobot processor pipeline
  - not ideal for Phase071
- Seeding bug:
  - `if seed:` skips seed `0`
- Randomness:
  - random task id
  - random init state
  - less controlled than benchmark eval
- Image flips:
  - LIBERO/OpenGL-specific
  - cannot copy into official MetaWorld adapter
- Possible code issue:
  - rollout logging f-string in handoff/code appears syntactically suspicious around nested quotes
- Treat safe-robot as GRPO shape inspiration, not benchmark truth.

## GRPO Reward Choice

- Initial Phase071 GRPO should use dense MetaWorld env reward sum.
- Also log success flags.
- Why:
  - dense reward gives more reward variance than sparse success only
  - GRPO group advantage needs reward diversity
  - safe-robot report had many zero-advantage updates even with sparse reward
- Risk:
  - optimizing dense reward may not maximize success perfectly
  - must evaluate success with official LeRobot protocol
- Possible later variant:
  - success-weighted return
  - terminal success bonus
  - sparse success-only return
  - compare zero-advantage frequency and official success rate

## Key Implementation Decisions

- Keep custom Phase11 backend.
  - reason: preserve old checkpoints/artifacts
  - mark as legacy/custom
- Add explicit flag:
  - `--env-backend custom|official_lerobot`
  - default `custom` until smoke stable
  - official Slurm uses `official_lerobot`
- Use official fixed task semantics:
  - `train_tasks[0]`
  - no episode-index cycling for Phase071
- Use direct single `MetaworldEnv` first.
  - reason: simpler than vector env
  - avoids vector same-step autoreset ambiguity
  - later vectorize only after correctness
- Use `--max-steps 0` to mean backend horizon.
  - official resolves to `500`
- Evaluate GRPO checkpoints with official backend.
  - write current `eval_summary.json`
  - also write official-style `eval_info.json`

## Tests Needed

- Unit tests:
  - fake LeRobot `MetaworldEnv` adapter contract
  - `reset(seed)` returns `pixels`, `agent_pos`
  - `max_episode_steps == 500`
  - `action_dim == 4`
  - success step returns `terminated=True`
- Preprocess tests:
  - `pixels` becomes `observation.image`
  - `agent_pos` becomes `observation.state`
  - no `observation.environment_state` inserted by default
  - task text present
  - no pixel flattening
- Rollout tests:
  - fake success at step 1 stops rollout
  - no reward after success
  - stored `log_probs`, `unsquashed_actions`, `proc_snapshots` lengths match rewards
- Flip tests:
  - official obs path does not call custom flip
- Script tests:
  - trainer exposes `--env-backend`
  - smoke exposes `--env-backend`
  - eval exposes `--env-backend`
  - Slurm uses `--export=NIL`
  - Slurm official smoke passes `--env-backend official_lerobot`
- GPU smoke:
  - one task
  - group size 2
  - one update
  - official env
  - horizon from env
  - writes checkpoint and progress
- Eval smoke:
  - evaluate latest checkpoint
  - one episode
  - official backend
  - write `eval_info.json`

## Open Risks For Thesis / Methods Section

- Official LeRobot MetaWorld benchmark differs from strict Farama MT50 RL benchmark.
- Phase071 `1ep` full run is smoke/reproduction check, not statistically stable benchmark.
- Official docs say recommended `10` episodes per task.
- Full `50 x 10` video eval is heavy.
- GRPO training is expensive:
  - safe-robot reported roughly `36.8` min/update in its setup
  - MetaWorld/SmolVLA throughput must be measured locally
- GRPO reward variance can be low:
  - zero advantages => skipped updates
  - group size matters
- Dense reward vs success metric mismatch:
  - train return not identical to eval success
- Official env auto-reset after terminal step:
  - dangerous if rollout loop accidentally continues
- Direct single env training vs vector eval:
  - same underlying env wrapper
  - but vector wrapper info format differs
  - eval should still use official eval path for final results

## Human Oversight Checklist

- Confirm Phase071 GRPO should train on official `train_tasks[0]`.
- Confirm dense MetaWorld reward sum is acceptable first training signal.
- Confirm no synthetic `observation.environment_state` by default.
- Confirm official backend remains explicit until smoke passes.
- Confirm single-task GRPO smoke before full MT50 scheduling.
- Confirm thesis wording distinguishes:
  - LeRobot MT50-style eval
  - strict Farama MT50 RL protocol
  - legacy/custom Phase07 runs

## Suggested Thesis Language

- "We use the LeRobot MetaWorld evaluation pathway as the canonical SmolVLA benchmark implementation, because it defines the policy preprocessing, task language injection, camera correction, success termination, and artifact schema used by the released SmolVLA checkpoint."
- "Earlier custom evaluators are retained as legacy diagnostic tooling but are not treated as benchmark-authoritative due to environment setup drift."
- "For GRPO, `lerobot-eval` is insufficient as a training loop because policy-gradient updates require the behavior-policy log probabilities and sampled latent actions for each rollout step. We therefore implement a thin rollout adapter that reuses LeRobot's official environment and observation preprocessing while preserving the GRPO rollout data contract."
- "All official MetaWorld evaluation results are reported separately from legacy custom runs."

## Source Pointers

- Official LeRobot MetaWorld env:
  - `/vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages/lerobot/envs/metaworld.py`
- Official LeRobot eval:
  - `/vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages/lerobot/scripts/lerobot_eval.py`
- Official preprocess utility:
  - `/vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages/lerobot/envs/utils.py`
- SmolVLA model action path:
  - `/vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages/lerobot/policies/smolvla/modeling_smolvla.py`
- Checkpoint processor/config:
  - `/vol/bitbucket/aa6622/.cache/huggingface/hub/models--jadechoghari--smolvla_metaworld/snapshots/ef3089ecb84eeeb7d33fedab24f6c76180a68900/policy_preprocessor.json`
  - `/vol/bitbucket/aa6622/.cache/huggingface/hub/models--jadechoghari--smolvla_metaworld/snapshots/ef3089ecb84eeeb7d33fedab24f6c76180a68900/config.json`
- Current GRPO:
  - `/vol/bitbucket/aa6622/project/src/smolvla_grpo/phase11_rollout.py`
  - `/vol/bitbucket/aa6622/project/src/smolvla_grpo/policy_wrapper.py`
  - `/vol/bitbucket/aa6622/project/scripts/grpo/train_phase11_env_on_policy_grpo.py`
- Safe-robot inspiration:
  - `/vol/bitbucket/aa6622/VGG JEPA/safe-robot-steering/HANDOFF_GRPO_REPRODUCTION.md`
  - `/vol/bitbucket/aa6622/VGG JEPA/safe-robot-steering/train/train_smolvla_grpo.py`
  - `/vol/bitbucket/aa6622/VGG JEPA/safe-robot-steering/model/smolvla_policy.py`
  - `/vol/bitbucket/aa6622/VGG JEPA/safe-robot-steering/env/env.py`
- Existing benchmark protocol note:
  - `/vol/bitbucket/aa6622/project/docs/smolvla_metaworld_benchmark_protocol.md`

## Web Sources Checked With Exa

- Hugging Face LeRobot MetaWorld docs:
  - `https://huggingface.co/docs/lerobot/metaworld`
- LeRobot MetaWorld source on GitHub:
  - `https://github.com/huggingface/lerobot/blob/4dbbcca4/src/lerobot/envs/metaworld.py`
- LeRobot eval source on GitHub:
  - `https://github.com/huggingface/lerobot/blob/main/src/lerobot/scripts/lerobot_eval.py`
- General GRPO environment/orchestrator references:
  - NVIDIA NeMo RL GRPO environment docs
  - Surogate GRPO docs

## Bottom Line

- Use official LeRobot eval for benchmark claims.
- Use official LeRobot env/preprocess inside GRPO adapter for training.
- Keep GRPO math/policy-logprob machinery.
- Do not copy safe-robot env/preprocess details.
- Do not compare legacy custom Phase07 numbers directly to Phase071 official numbers.
