# Parallel GRPO Rollout Report

Generated: 2026-05-12

## Executive Summary

We added opt-in batched rollout execution for official LeRobot MetaWorld GRPO runs, validated both `vector_sync` and `vector_async`, and launched a real `push-v3` GRPO run.

Final practical conclusion: `vector_async` works, but it is not meaningfully faster than `vector_sync` for the current `push-v3`, `group_size=6`, `max_steps=120` workload. The optimizer/log-prob recompute dominates update time, so env stepping mode only moves a small slice of the total runtime.

Recommended mode for future speed-focused runs: `vector_sync`.

## Why This Work Was Needed

The original GRPO collector used one trainer and serial rollout collection. For each GRPO group row, it reset and stepped an official LeRobot MetaWorld vector env with `n_envs=1`. That was correct but left possible wall-clock speed on the table because rollout rows were independent.

We wanted:

| Goal | Requirement |
|---|---|
| Keep one trainer | No multi-trainer/distributed GRPO rewrite |
| Preserve default behavior | `serial` remains default |
| Make parallel rollout opt-in | `--rollout-execution {serial,vector_sync,vector_async}` |
| Restrict to official backend | Vector modes only for `official_lerobot` |
| Avoid unsafe Linux async default | Do not use local LeRobot `make_env(..., use_async_envs=True)` directly |
| Smoke before real run | GPU Slurm smoke for sync and async |

## Relevant Upstream / Docs Sources

These were checked during implementation and debugging:

| Source | What It Established |
|---|---|
| Gymnasium `AsyncVectorEnv` docs: `https://gymnasium.farama.org/api/vector/async_vector_env/` | `AsyncVectorEnv` uses worker processes and does not expose sync-only `.envs` access in the way local LeRobot helper expected. |
| Gymnasium installed source: `/vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages/gymnasium/vector/async_vector_env.py` | Worker `.call()` uses `env.get_wrapper_attr(name)`, requiring envs to be proper `gym.Env` instances. |
| Local LeRobot helper: `/vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages/lerobot/envs/utils.py` | Local `add_envs_task()` assumes `env.envs[0]`, which breaks for `AsyncVectorEnv`. |
| Upstream LeRobot commit `b2765b39b82fa0d9de027cac90e9060e22513317` | Upstream added cached metadata/spaces for lazy async envs (`Cache lazy async env metadata for eval`). |
| Upstream LeRobot `main` `envs/utils.py` | `_LazyAsyncVectorEnv` exposes `call()`, `get_attr()`, `unwrapped`, cached `observation_space`, `action_space`, `metadata`. |
| Upstream LeRobot `main` `envs/metaworld.py` | MetaWorld env creation is deferred until first worker-side reset/use, avoiding parent-process MuJoCo/EGL construction before async workers. |

## What We Changed

| File | Change |
|---|---|
| `src/smolvla_grpo/lerobot_metaworld_adapter.py` | Added `OfficialBatchStep`, `LazyForkserverAsyncVectorEnv`, `DeferredLeRobotMetaworldEnv`, batched reset/step, per-env success parsing, async task injection via `vec_env.call()`. |
| `src/smolvla_grpo/official_lerobot_vector_rollout.py` | New batched collector for `vector_sync` and `vector_async`; returns existing `RolloutTrajectory` objects. |
| `src/smolvla_grpo/phase11_rollout.py` | Added `rollout_execution` dispatch while preserving serial path. |
| `src/smolvla_grpo/policy_wrapper.py` | Added `SampledBatchStep` and `sample_action_batch_from_proc()`. |
| `scripts/grpo/train_phase11_env_on_policy_grpo.py` | Added CLI flags, manifest fields, progress timing fields, per-update heartbeat logging. |
| `scripts/grpo/smoke_phase11_rollout.py` | Added rollout execution flags for quick smoke. |
| `scripts/grpo/submit_phase111_vector_rollout_smoke.slurm` | New GPU smoke: runs `vector_sync` then `vector_async`, asserts async `forkserver` manifest. |
| `scripts/grpo/submit_phase111_single_task_grpo.slurm` | Added 48h walltime, positional rollout mode args, stronger final asserts. |
| `tests/test_grpo_lerobot_adapter.py` | Extended adapter/static tests. |
| `tests/test_official_lerobot_vector_rollout.py` | Added vector rollout helper test. |
| `tests/test_phase11_slurm_scripts.py` | Added Slurm syntax/static assertions for vector smoke and real-run validation. |

## Async Adapter Design

Local LeRobot `0.5.1` was not safe enough for our async target because it creates the real MetaWorld/MuJoCo env in `MetaworldEnv.__init__`, and local `add_envs_task()` reaches into `env.envs[0]`. Both are poor fits for `AsyncVectorEnv` with `forkserver`.

We backported the relevant upstream pattern in project-local code:

| Component | Purpose |
|---|---|
| `LazyForkserverAsyncVectorEnv` | Defer real `AsyncVectorEnv` spawn until first use; force `context="forkserver"`, `shared_memory=True`, `AutoresetMode.SAME_STEP`; cache spaces/metadata. |
| `DeferredLeRobotMetaworldEnv` | Create observation/action spaces in parent, but create real MetaWorld/MuJoCo env inside worker on first reset/render/step. |
| `_task_batch()` | Use `vec_env.call("task_description")` or `vec_env.call("task")` instead of local LeRobot `add_envs_task()` sync-only `.envs[0]` path. |
| `_successes_from_vector_info()` | Handle vector `final_info` forms and legacy dict forms. |

## Verification

| Check | Result |
|---|---|
| Focused pytest | `20 passed, 2 warnings` |
| Python compile check | Passed |
| Slurm syntax check | Passed |
| Vector rollout smoke job `240278` | Passed |
| Smoke success markers | `PHASE111_VECTOR_ROLLOUT_SMOKE_OK`, `PHASE111_VECTOR_ROLLOUT_SLURM_OK` |

## Slurm Runs

| Run | Job | Mode | Task | Group | Max Steps | Updates | Seed Base | Node | Status at Report Time | Output |
|---|---:|---|---|---:|---:|---:|---:|---|---|---|
| Main GRPO | `240282` | `vector_async` | `push-v3` | 6 | 120 | 100 | 2000 | `parrot` | Running, 64 rows complete | `artifacts/phase111_on_grpo_lerobot/push-v3/g6_u100_seed2000_vector_async` |
| Sync bench 10 | `240569` | `vector_sync` | `push-v3` | 6 | 120 | 10 | 2000 | `parrot` | Running, 6 rows complete | `artifacts/phase111_on_grpo_lerobot/push-v3/g6_u10_seed2000_vector_sync_bench` |
| Sync bench 3 | `240320` | `vector_sync` | `push-v3` | 6 | 120 | 3 | 3000 | `gpuvm36` | Complete | `artifacts/phase111_on_grpo_lerobot/push-v3/g6_u3_seed3000_vector_sync_bench` |
| Vector smoke | `240278` | `vector_sync` + `vector_async` | `push-v3` | 2 | 8 | 1 each | 2000 | `parrot` | Complete | `artifacts/phase111_vector_rollout_smoke_vec_*` |

## Timing Summary

All times are seconds per update. `last10` is more representative for async because early async updates included warmup/outlier behavior.

| Run | Window | Rows | Rollout Mean | Rollout Median | Optimize Mean | Optimize Median | Total Mean | Total Median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Async main | all | 64 | 217.1 | 88.9 | 932.7 | 953.0 | 1149.8 | 1046.3 |
| Async main | first10 | 10 | 207.4 | 103.5 | 939.8 | 953.0 | 1147.2 | 1057.6 |
| Async main | last10 | 10 | 88.1 | 88.1 | 935.6 | 951.0 | 1023.8 | 1039.1 |
| Sync 10 bench | current all | 6 | 83.2 | 83.1 | 926.2 | 953.9 | 1009.4 | 1037.1 |
| Sync 3 bench | all | 3 | 108.4 | 93.6 | 1094.7 | 1093.6 | 1203.5 | 1185.2 |

## Sync vs Async Comparison

Best apples-to-apples comparison available is async main last10 vs sync 10 bench current rows. Same task, max steps, group size, seed base, node, code path, and checkpoint. Different run length and concurrent scheduler noise still apply.

| Metric | Async Main Last10 | Sync 10 Bench | Winner | Delta |
|---|---:|---:|---|---:|
| Rollout mean | 88.1s | 83.2s | `vector_sync` | 4.9s |
| Optimize mean | 935.6s | 926.2s | `vector_sync` | 9.4s |
| Total mean/update | 1023.8s | 1009.4s | `vector_sync` | 14.3s |
| Env steps/update | 708.1 | 698.0 | n/a | sync had slightly fewer env steps |
| Success mean | 0.033 | 0.056 | n/a | noisy |

Interpretation: `vector_sync` is slightly faster in this benchmark, but difference is small (~1.4% total update time). The dominant cost is optimization/log-prob recompute, not env rollout.

## Why Async Did Not Win

Both vector modes already batch policy action sampling across the group rows. The remaining difference is env stepping mechanics:

| Mode | Env stepping model | Expected advantage | Observed |
|---|---|---|---|
| `vector_sync` | One process, steps envs in a loop | Low overhead | Fastest in benchmark |
| `vector_async` | Forkserver workers + IPC + shared memory | Potential parallel env stepping | Works, but overhead cancels benefit |

At `group_size=6`, each update spends roughly:

| Phase | Typical Time | Share |
|---|---:|---:|
| Rollout | 83-88s | ~8% |
| Optimize/log-prob recompute | 926-936s | ~91% |
| Other | small | ~1% |

Even a large rollout improvement would only modestly affect total update time. Current async does not even improve rollout time; it is ~5s slower than sync in the same-node benchmark.

## Current Training Health

| Run | Updates Complete | Success Rows | Mean Success Rate | Latest Checkpoint |
|---|---:|---:|---:|---|
| Async main | 64/100 | 17/64 | 0.047 | `update_0060.pt` |
| Sync 10 bench | 6/10 | 2/6 | 0.056 | `update_0005.pt` |

Main run is healthy:

| Signal | Status |
|---|---|
| Slurm state | Running |
| Tracebacks | None seen |
| Checkpoints | Present through `update_0060.pt` |
| Progress rows | Appending |
| Async mode | `forkserver` |
| Walltime | 48h allocation; projected to fit |

## Recommendations

| Decision | Recommendation |
|---|---|
| Finish current async main? | Yes. It is healthy and already past 60%. |
| Future speed mode | Prefer `vector_sync`. |
| Keep async code? | Yes, but mark as experimental/fallback. It works and may help for heavier envs or larger groups. |
| Next optimization target | Optimize GRPO update/log-prob recompute. Rollout mode is not the bottleneck. |
| Further benchmark | Let sync 10 finish, then compare all 10 sync rows against async updates 0-9 and async last10. |

## Follow-Up Ideas

The next meaningful speed work is likely in optimizer/log-prob recomputation:

| Idea | Why |
|---|---|
| Batch `get_action_probs_from_proc_list` across more timesteps/rows | Current optimization appears dominated by repeated forward/log-prob recompute. |
| Increase `chunk_size` carefully | May reduce loop overhead, but GPU memory must be watched. |
| Profile one update | Need CPU/GPU profile before refactoring optimizer path. |
| Compare `group_size=4` vs `6` | If optimizer scales poorly with group size, smaller group could improve wall-clock per useful update. |

## Commands Used

Real async run:

```bash
cd /vol/bitbucket/aa6622/project
sbatch --export=NIL scripts/grpo/submit_phase111_single_task_grpo.slurm \
  push-v3 100 6 120 5 2000 0 \
  /vol/bitbucket/aa6622/project/artifacts/phase111_on_grpo_lerobot/push-v3/g6_u100_seed2000_vector_async \
  "" vector_async forkserver
```

Sync 10 benchmark:

```bash
cd /vol/bitbucket/aa6622/project
sbatch --export=NIL scripts/grpo/submit_phase111_single_task_grpo.slurm \
  push-v3 10 6 120 5 2000 0 \
  /vol/bitbucket/aa6622/project/artifacts/phase111_on_grpo_lerobot/push-v3/g6_u10_seed2000_vector_sync_bench \
  "" vector_sync forkserver
```

Monitor:

```bash
cd /vol/bitbucket/aa6622/project
squeue -j 240282,240569 -o "%.18i %.9P %.24j %.8u %.2t %.10M %.6D %R"
```

