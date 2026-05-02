# Push-v3 SmolVLA Chunk Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run five parallel 1-GPU Push-v3 baseline jobs, each evaluating 25 video episodes with `n_envs=3`, `max_steps=180`, random-seeded resets, and SmolVLA chunk lengths `2, 5, 10, 15, 20`.

**Architecture:** Extend the existing queue-free vector baseline runner so it can sample a full SmolVLA action chunk at each observation and execute that chunk open-loop until success, truncation, or the next chunk boundary. Add a PBS shard script plus all-five launcher that maps one chunk length to one GPU job and writes isolated outputs per chunk length. Each PBS job also runs lightweight GPU telemetry sampling so post-run summaries report max/mean GPU utilization, memory use, power draw, and per-process GPU use instead of relying on PBS CPU/RSS records.

**Tech Stack:** Python 3.12, PyTorch, LeRobot SmolVLA, MetaWorld v3, existing `OfficialLeRobotMetaWorldGRPORollout`, PBS.

---

## Autonomous Overnight Execution Policy

- User may be asleep/unavailable after saying to execute this plan. Treat execution approval as approval to code, test, review, fix scoped bugs, smoke, and submit the full five-job sweep without asking for intermediate confirmation.
- User grants 100% autonomy for this plan once execution begins. Do not pause for human prompting between steps. Keep moving through implementation, tests, reviews, root-cause diagnosis, fixes, smoke jobs, GPU telemetry checks, and full PBS submission until completion or a hard blocker is reached.
- Use subagent-driven development by default: implement each task with a fresh builder subagent, review each completed task with a reviewer subagent, then continue to the next task if review findings are clear and scoped. Main agent and all subagents should use `/caveman` mode for status, findings, and review summaries while keeping code, commit messages, and generated files normal.
- If a subagent, test, smoke job, or PBS submission exposes an obvious in-scope bug, fix it immediately and rerun the relevant tests/smoke. Examples: missing import, stale CLI signature, wrong output path, incorrect PBS variable, action tensor shape mismatch, missing executable bit, or summary field not written.
- Do not stop for approval on straightforward fixes that preserve the requested experiment contract: Push-v3, standard SmolVLA baseline checkpoint, chunk lengths `2,5,10,15,20`, 25 episodes, `n_envs=3`, `max_steps=180`, random-seeded resets, videos, five 1-GPU PBS jobs.
- If blocked by an unfamiliar failure, research autonomously before stopping: inspect logs/artifacts, search the repo, query PBS state, and use web/Exa search for current documentation when needed. Apply fixes when confidence is high and the fix stays within this plan.
- Monitor queued/running PBS jobs without waiting for user input. If a smoke or full job fails, collect PBS output, telemetry files, summaries, and relevant stack traces; root-cause the failure; patch in-scope bugs; rerun targeted tests; resubmit the failed smoke/job once.
- Stop and report only for blockers outside this plan: destructive git operations, credentials/manual cluster auth, unavailable PBS/GPU resources after one retry, missing checkpoint/cache that cannot be inferred from existing scripts, repeated job failure after an in-scope fix/resubmit, or a failure whose fix would change experiment semantics.
- After local tests pass, submit the 1-episode smoke. If smoke passes, immediately submit the full five-job sweep. Do not wait for the user between smoke and full submission.
- After full submission, report PBS job ids and output root. If jobs finish while the agent is still running, verify output summaries/videos and report chunk-level metrics.
- Do not commit generated artifacts/videos unless the user explicitly asks. Source/test/script commits are allowed when the executor is following this plan and repository policy permits commits.

## Scope Check

This is one subsystem: Push-v3 baseline evaluation. Do not modify Phase57 raw-vs-bounded decode, JEPA-WM metrics, MT50 sharding, training code, or action normalization semantics.

## File Structure

- Modify: `src/smolvla_grpo/phase12_vector_eval.py`
  - Add queue-free chunk selection helper.
  - Add postprocessed action chunk coercion helper.
  - Keep existing one-step eval helpers unchanged for prior tests and callers.

- Modify: `scripts/grpo/eval_smolvla_baseline_vector_video.py`
  - Add `--chunk-len`.
  - Load SmolVLA with `n_action_steps=chunk_len`.
  - Execute chunks open-loop inside existing vector wave loop.
  - Preserve per-episode videos and summary schema; add chunk metadata.

- Create: `scripts/grpo/submit_smolvla_baseline_pushv3_chunk_sweep.pbs`
  - One PBS job for one chunk length.
  - Defaults: Push-v3, 25 episodes, `n_envs=3`, `max_steps=180`, seed start `1000`, videos on.
  - Starts/stops nonfatal `nvidia-smi` telemetry sampling around the Python workload.
  - Writes PBS resource snapshot and GPU telemetry summaries under the chunk output directory.

- Create: `scripts/grpo/submit_smolvla_baseline_pushv3_chunk_sweep_all5.sh`
  - Submit five independent jobs with chunk lengths `2,5,10,15,20`.
  - Request one RTX6000 GPU per job.

- Create: `scripts/grpo/summarize_phase58_gpu_telemetry.py`
  - Parse `nvidia_smi_samples.csv`.
  - Write `gpu_telemetry_summary.json` with max/mean/p95 utilization, max memory, max power, and saturation counters.

- Modify: `tests/test_phase12_vector_eval.py`
  - Unit-test chunk action selection and coercion.
  - Unit-test chunked vector execution stops rows correctly and does not use SmolVLA action queue.

- Modify: `tests/test_phase12_pbs_static.py`
  - Static-test PBS resources, env defaults, output root, chunk list, and qsub env wiring.
  - Static-test telemetry sampler, cleanup trap, nonfatal behavior, and summary command.

- Create: `tests/test_phase58_gpu_telemetry.py`
  - Unit-test telemetry CSV parsing and summary math.

## Output Contract

Root:

`artifacts/phase58_smolvla_baseline_pushv3_chunk_sweep_25ep_s1000_max180_5x1gpu/`

Per chunk:

- `chunk_len_02/eval_summary.json`
- `chunk_len_02/episode_0000_seed_1000/selected_action_rollout.mp4`
- `chunk_len_02/episode_0000_seed_1000/episode_summary.json`
- `chunk_len_02/gpu_telemetry/nvidia_smi_start.txt`
- `chunk_len_02/gpu_telemetry/nvidia_smi_samples.csv`
- `chunk_len_02/gpu_telemetry/nvidia_smi_pmon.txt`
- `chunk_len_02/gpu_telemetry/nvidia_smi_end.txt`
- `chunk_len_02/gpu_telemetry/pbs_resource_snapshot.txt`
- `chunk_len_02/gpu_telemetry/gpu_telemetry_summary.json`
- `chunk_len_05/...`
- `chunk_len_10/...`
- `chunk_len_15/...`
- `chunk_len_20/...`

GPU telemetry summary must include:

```json
{
  "sample_count": 100,
  "max_gpu_utilization_pct": 98.0,
  "mean_gpu_utilization_pct": 61.4,
  "p95_gpu_utilization_pct": 93.0,
  "samples_gpu_utilization_ge_80_pct": 42,
  "max_memory_used_mib": 21234.0,
  "max_memory_used_fraction": 0.88,
  "max_memory_utilization_pct": 75.0,
  "max_power_draw_w": 247.5,
  "telemetry_csv": "/absolute/path/to/nvidia_smi_samples.csv"
}
```

GPU telemetry research basis:

- NVIDIA documents `nvidia-smi --query-gpu=... --format=csv -l <seconds>` for timestamped CSV logging of `utilization.gpu`, `utilization.memory`, `memory.used`, and related fields. `utilization.gpu` is percent of sample period with one or more GPU kernels executing, and sample period is product dependent around 1s to 1/6s. Source: <https://nvidia.custhelp.com/app/answers/detail/a_id/3751/~/useful-nvidia-smi-queries>
- NVIDIA documents `nvidia-smi dmon` for device-level monitoring of power, temperature, SM/memory clocks, utilization, framebuffer memory, and PCIe throughput, with configurable frequency and CSV output. Source: <https://docs.nvidia.com/deploy/nvidia-smi/>
- NVIDIA documents `nvidia-smi pmon` for per-process SM/memory utilization and framebuffer memory, with 1-10s monitoring frequency. Source: <https://docs.nvidia.com/deploy/nvidia-smi/index.html>
- NVIDIA DCGM supports job-level statistics and continuous GPU telemetry, but it usually depends on cluster DCGM availability/permissions. Treat `dcgmi` as optional and nonfatal in this PBS script. Source: <https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/index.html>

Each `eval_summary.json` must include:

```json
{
  "task": "push-v3",
  "episodes": 25,
  "eval_seed_start": 1000,
  "eval_seed_end": 1024,
  "n_envs": 3,
  "chunk_len": 10,
  "max_steps": 180,
  "reset_randomization_mode": "random_seeded",
  "video_enabled": true,
  "pc_success": 0.0,
  "avg_sum_reward": 0.0,
  "avg_max_reward": 0.0,
  "episodes_rows": []
}
```

## Task 1: Add Queue-Free Chunk Helpers

**Files:**
- Modify: `src/smolvla_grpo/phase12_vector_eval.py`
- Test: `tests/test_phase12_vector_eval.py`

- [ ] **Step 1: Add failing test for postprocessed chunk coercion**

Append to `tests/test_phase12_vector_eval.py`:

```python
def test_coerce_exec_action_chunk_batch_preserves_chunk_rows() -> None:
    from smolvla_grpo.phase12_vector_eval import coerce_exec_action_chunk_batch

    action = np.array(
        [
            [[2.0, 0.5, -0.5, -2.0], [0.1, 0.2, 0.3, 0.4]],
            [[-3.0, 0.0, 3.0, 0.5], [0.9, -0.9, 0.8, -0.8]],
        ],
        dtype=np.float32,
    )

    out = coerce_exec_action_chunk_batch(action, action_dim=4, n_envs=2, chunk_len=2)

    assert out.shape == (2, 2, 4)
    np.testing.assert_allclose(out[0, 0], np.array([1.0, 0.5, -0.5, -1.0], dtype=np.float32))
    np.testing.assert_allclose(out[1, 0], np.array([-1.0, 0.0, 1.0, 0.5], dtype=np.float32))
```

- [ ] **Step 2: Run test and verify it fails**

Run:

```bash
pytest tests/test_phase12_vector_eval.py::test_coerce_exec_action_chunk_batch_preserves_chunk_rows -q
```

Expected: `ImportError` or `cannot import name 'coerce_exec_action_chunk_batch'`.

- [ ] **Step 3: Implement action chunk coercion helper**

Add after `coerce_exec_action_batch()` in `src/smolvla_grpo/phase12_vector_eval.py`:

```python
def coerce_exec_action_chunk_batch(action: Any, *, action_dim: int, n_envs: int, chunk_len: int) -> np.ndarray:
    if hasattr(action, "detach"):
        action_np = action.detach().float().cpu().numpy()
    else:
        action_np = np.asarray(action, dtype=np.float32)
    action_np = np.asarray(action_np, dtype=np.float32)
    expected_size = int(n_envs) * int(chunk_len) * int(action_dim)
    if action_np.size != expected_size:
        raise RuntimeError(
            f"Policy action chunk dim mismatch: expected batch ({n_envs}, {chunk_len}, {action_dim}) "
            f"with {expected_size} values, got shape {tuple(action_np.shape)} and size {action_np.size}. "
            "Refusing silent pad/truncate."
        )
    return np.clip(action_np.reshape(int(n_envs), int(chunk_len), int(action_dim)), -1.0, 1.0).astype(
        np.float32,
        copy=False,
    )
```

- [ ] **Step 4: Run focused test and verify it passes**

Run:

```bash
pytest tests/test_phase12_vector_eval.py::test_coerce_exec_action_chunk_batch_preserves_chunk_rows -q
```

Expected: `1 passed`.

- [ ] **Step 5: Add failing test for queue-free chunk selection**

Append to `tests/test_phase12_vector_eval.py`:

```python
def test_select_eval_action_chunk_queue_free_uses_model_sample_actions() -> None:
    import torch
    from types import SimpleNamespace

    from smolvla_grpo.phase12_vector_eval import select_eval_action_chunk_queue_free

    class FakeModel:
        def __init__(self) -> None:
            self.calls = 0

        def sample_actions(self, images, img_masks, lang_tokens, lang_masks, state, noise=None):
            del images, img_masks, lang_tokens, lang_masks, state, noise
            self.calls += 1
            return torch.arange(2 * 5 * 4, dtype=torch.float32).reshape(2, 5, 4)

    class FakePolicy:
        def __init__(self) -> None:
            self.model = FakeModel()
            self.config = SimpleNamespace(action_feature=SimpleNamespace(shape=(4,)))
            self.select_action_calls = 0

        def _prepare_batch(self, proc):
            return proc

        def prepare_images(self, batch):
            return batch["images"], batch["img_masks"]

        def prepare_state(self, batch):
            return batch["state"]

        def select_action(self, proc):
            del proc
            self.select_action_calls += 1
            return torch.zeros(2, 4)

    proc = {
        "images": torch.zeros(2, 3, 8, 8),
        "img_masks": torch.ones(2, 1),
        "state": torch.zeros(2, 8),
        "observation.language.tokens": torch.zeros(2, 4, dtype=torch.long),
        "observation.language.attention_mask": torch.ones(2, 4, dtype=torch.long),
    }
    policy = FakePolicy()

    got = select_eval_action_chunk_queue_free(policy, proc, chunk_len=3)

    assert got.shape == (2, 3, 4)
    assert policy.model.calls == 1
    assert policy.select_action_calls == 0
    torch.testing.assert_close(got, torch.arange(2 * 5 * 4, dtype=torch.float32).reshape(2, 5, 4)[:, :3, :])
```

- [ ] **Step 6: Run test and verify it fails**

Run:

```bash
pytest tests/test_phase12_vector_eval.py::test_select_eval_action_chunk_queue_free_uses_model_sample_actions -q
```

Expected: `ImportError` or `cannot import name 'select_eval_action_chunk_queue_free'`.

- [ ] **Step 7: Implement queue-free chunk selection**

Add after `select_eval_action_queue_free()` in `src/smolvla_grpo/phase12_vector_eval.py`:

```python
def select_eval_action_chunk_queue_free(policy: Any, proc: dict[str, Any], *, chunk_len: int) -> torch.Tensor:
    """Return an eval action chunk without using SmolVLA's cross-step action queue."""

    if int(chunk_len) < 1:
        raise ValueError("chunk_len must be >= 1")
    if all(hasattr(policy, name) for name in ("_prepare_batch", "prepare_images", "prepare_state")) and hasattr(
        getattr(policy, "model", None), "sample_actions"
    ):
        batch = policy._prepare_batch(proc)
        images, img_masks = policy.prepare_images(batch)
        state = policy.prepare_state(batch)
        lang_tokens = batch["observation.language.tokens"]
        lang_masks = batch["observation.language.attention_mask"]
        actions = policy.model.sample_actions(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            noise=None,
        )
        if not torch.is_tensor(actions):
            raise RuntimeError("SmolVLA sample_actions must return a tensor during vector eval")
        action_dim = int(policy.config.action_feature.shape[0])
        actions = actions[:, :, :action_dim] if actions.ndim == 3 else actions.reshape(actions.shape[0], -1, action_dim)
        if int(actions.shape[1]) < int(chunk_len):
            raise RuntimeError(f"SmolVLA returned {int(actions.shape[1])} actions, requested chunk_len={int(chunk_len)}")
        return actions[:, : int(chunk_len), :]

    first = select_eval_action_queue_free(policy, proc)
    if int(chunk_len) != 1:
        raise RuntimeError("Chunked baseline eval requires SmolVLA model.sample_actions when chunk_len > 1")
    return first.unsqueeze(1)
```

- [ ] **Step 8: Run focused helper tests**

Run:

```bash
pytest tests/test_phase12_vector_eval.py::test_coerce_exec_action_chunk_batch_preserves_chunk_rows tests/test_phase12_vector_eval.py::test_select_eval_action_chunk_queue_free_uses_model_sample_actions -q
```

Expected: `2 passed`.

- [ ] **Step 9: Commit**

```bash
git add src/smolvla_grpo/phase12_vector_eval.py tests/test_phase12_vector_eval.py
git commit -m "feat: add queue-free SmolVLA chunk eval helpers"
```

## Task 2: Execute Chunks In Baseline Video Runner

**Files:**
- Modify: `scripts/grpo/eval_smolvla_baseline_vector_video.py`
- Test: `tests/test_phase12_vector_eval.py`

- [ ] **Step 1: Add failing test for chunked vector execution**

Append to `tests/test_phase12_vector_eval.py`:

```python
def test_vector_eval_chunked_execution_samples_once_per_chunk(tmp_path, monkeypatch):
    import sys
    import types
    from types import SimpleNamespace

    import torch

    from smolvla_grpo.phase12_vector_eval import evaluate_loaded_policy_vectorized

    class FakePolicy:
        def __init__(self):
            self.reset_calls = 0
            self.chunk_calls = []

        def reset(self):
            self.reset_calls += 1

    class FakeBundle:
        base_checkpoint = "base"
        grpo_checkpoint = None
        device = torch.device("cpu")

        def __init__(self):
            self.policy = FakePolicy()

        def postprocessor(self, action):
            return action

    class FakeEnv:
        action_dim = 4

        def __init__(self, *, task, n_envs=1):
            del task
            assert n_envs == 1
            self.seed = None
            self.steps = 0

        def reset(self, seed):
            self.seed = int(seed)
            self.steps = 0
            return {"state": np.array([float(seed), 0.0, 0.0, 0.0], dtype=np.float32)}

        def build_proc(self, obs, *, bundle):
            del bundle
            return {
                "observation.state": torch.tensor([[obs["state"][0], obs["state"][1], 0.0, 0.0]], dtype=torch.float32),
                "task": [f"seed-{self.seed}"],
            }

        def step(self, action):
            self.steps += 1
            return SimpleNamespace(
                observation={"state": np.array([float(self.seed), float(self.steps), 0.0, 0.0], dtype=np.float32)},
                reward=float(np.asarray(action).reshape(-1)[0]),
                success=False,
                terminated=False,
                truncated=False,
            )

        def close(self):
            return None

    fake_adapter = types.ModuleType("smolvla_grpo.lerobot_metaworld_adapter")
    fake_adapter.OfficialLeRobotMetaWorldGRPORollout = FakeEnv
    fake_adapter.resolve_lerobot_horizon = lambda env, max_steps: int(max_steps)
    monkeypatch.setitem(sys.modules, "smolvla_grpo.lerobot_metaworld_adapter", fake_adapter)
    monkeypatch.setattr("smolvla_grpo.phase12_vector_eval._resolve_action_dim", lambda task: 4)

    def fake_write_episode_artifacts(*, episode_dir, actions, rewards, successes, overlay_mode):
        del actions, rewards, successes, overlay_mode
        episode_dir.mkdir(parents=True, exist_ok=True)

    def fake_chunk(policy, proc, *, chunk_len):
        policy.chunk_calls.append((int(proc["observation.state"].shape[0]), int(chunk_len)))
        batch = int(proc["observation.state"].shape[0])
        return torch.ones(batch, int(chunk_len), 4)

    monkeypatch.setattr("smolvla_grpo.phase12_vector_eval.write_episode_artifacts", fake_write_episode_artifacts)
    monkeypatch.setattr("smolvla_grpo.phase12_vector_eval.select_eval_action_chunk_queue_free", fake_chunk)

    bundle = FakeBundle()
    summary = evaluate_loaded_policy_vectorized(
        bundle=bundle,
        base_checkpoint="base",
        grpo_checkpoint=None,
        output_dir=tmp_path,
        task="push-v3",
        episodes=3,
        eval_seed_start=1000,
        n_envs=3,
        rollout_execution="vector_sync",
        max_steps=5,
        chunk_len=2,
    )

    assert summary["episodes"] == 3
    assert summary["chunk_len"] == 2
    assert bundle.policy.chunk_calls == [(3, 2), (3, 2), (3, 2)]
```

- [ ] **Step 2: Run test and verify it fails**

Run:

```bash
pytest tests/test_phase12_vector_eval.py::test_vector_eval_chunked_execution_samples_once_per_chunk -q
```

Expected: `TypeError` because `evaluate_loaded_policy_vectorized()` has no `chunk_len` parameter.

- [ ] **Step 3: Update imports and function signature**

In `src/smolvla_grpo/phase12_vector_eval.py`, change imports already local and update signature:

```python
def evaluate_loaded_policy_vectorized(
    *,
    bundle: Any,
    base_checkpoint: str,
    grpo_checkpoint: Path | None,
    output_dir: Path,
    task: str,
    episodes: int,
    eval_seed_start: int,
    n_envs: int,
    rollout_execution: str,
    max_steps: int,
    chunk_len: int = 1,
) -> dict[str, Any]:
```

- [ ] **Step 4: Replace one-step loop with chunk-aware loop**

Inside `evaluate_loaded_policy_vectorized()`, replace the `for _step in range(int(resolved_steps)):` block with:

```python
            step_count = 0
            while step_count < int(resolved_steps):
                if not bool(np.any(active)):
                    break
                active_rows = [idx for idx in range(wave_n) if bool(active[idx])]
                proc_rows = [envs[idx].build_proc(obs_by_row[idx], bundle=bundle) for idx in active_rows]
                proc = concatenate_proc_rows(proc_rows)
                effective_chunk = min(int(chunk_len), int(resolved_steps) - int(step_count))
                with torch.inference_mode():
                    if int(chunk_len) == 1:
                        action = select_eval_action_queue_free(bundle.policy, proc)
                        post = bundle.postprocessor(action)
                        exec_action_np = coerce_exec_action_batch(post, action_dim=action_dim, n_envs=len(active_rows))[
                            :, None, :
                        ]
                    else:
                        action = select_eval_action_chunk_queue_free(bundle.policy, proc, chunk_len=effective_chunk)
                        post = bundle.postprocessor(action)
                        exec_action_np = coerce_exec_action_chunk_batch(
                            post,
                            action_dim=action_dim,
                            n_envs=len(active_rows),
                            chunk_len=effective_chunk,
                        )

                for chunk_step in range(effective_chunk):
                    if not bool(np.any(active)):
                        break
                    for batch_row, row in enumerate(active_rows):
                        if not bool(active[row]):
                            continue
                        step = envs[row].step(exec_action_np[batch_row, chunk_step : chunk_step + 1])
                        obs_by_row[row] = step.observation
                        actions[row].append(exec_action_np[batch_row, chunk_step].reshape(-1).tolist())
                        rewards[row].append(float(step.reward))
                        successes[row].append(bool(step.success))
                        if step.success or step.terminated or step.truncated:
                            active[row] = False
                            terminated[row] = bool(step.terminated)
                            truncated[row] = bool(step.truncated)
                    step_count += 1
                    if step_count >= int(resolved_steps):
                        break
```

- [ ] **Step 5: Add chunk metadata to vector eval summary**

After `write_eval_artifacts(...)` return value is computed, store summary in a variable and add chunk metadata:

```python
    summary = write_eval_artifacts(
        base_checkpoint=base_checkpoint,
        grpo_checkpoint=grpo_checkpoint,
        output_dir=output_dir,
        task=task,
        episodes=episodes,
        eval_seed_start=eval_seed_start,
        results=all_results,
    )
    summary["n_envs"] = int(n_envs)
    summary["max_steps"] = int(max_steps)
    summary["chunk_len"] = int(chunk_len)
    (output_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
```

- [ ] **Step 6: Run focused test**

Run:

```bash
pytest tests/test_phase12_vector_eval.py::test_vector_eval_chunked_execution_samples_once_per_chunk -q
```

Expected: `1 passed`.

- [ ] **Step 7: Update baseline video runner CLI**

In `scripts/grpo/eval_smolvla_baseline_vector_video.py`, add imports:

```python
    coerce_exec_action_chunk_batch,
    select_eval_action_chunk_queue_free,
```

Add parser argument:

```python
    parser.add_argument("--chunk-len", type=int, default=1)
```

Change bundle load:

```python
        n_action_steps=int(args.chunk_len),
```

- [ ] **Step 8: Update baseline video runner loop**

In `scripts/grpo/eval_smolvla_baseline_vector_video.py`, replace the per-step sampling loop with the same chunk loop from Step 4, plus frame capture after every env step:

```python
            step_count = 0
            while step_count < int(resolved_steps):
                if not bool(np.any(active)):
                    break
                active_rows = [idx for idx in range(wave_n) if bool(active[idx])]
                proc = concatenate_proc_rows(
                    [envs[idx].build_proc(obs_by_row[idx], bundle=bundle) for idx in active_rows]
                )
                effective_chunk = min(int(args.chunk_len), int(resolved_steps) - int(step_count))
                with torch.inference_mode():
                    if int(args.chunk_len) == 1:
                        action = select_eval_action_queue_free(bundle.policy, proc)
                        post = bundle.postprocessor(action)
                        exec_action_np = coerce_exec_action_batch(
                            post,
                            action_dim=int(envs[0].action_dim),
                            n_envs=len(active_rows),
                        )[:, None, :]
                    else:
                        action = select_eval_action_chunk_queue_free(bundle.policy, proc, chunk_len=effective_chunk)
                        post = bundle.postprocessor(action)
                        exec_action_np = coerce_exec_action_chunk_batch(
                            post,
                            action_dim=int(envs[0].action_dim),
                            n_envs=len(active_rows),
                            chunk_len=effective_chunk,
                        )
                for chunk_step in range(effective_chunk):
                    if not bool(np.any(active)):
                        break
                    for batch_row, row in enumerate(active_rows):
                        if not bool(active[row]):
                            continue
                        step = envs[row].step(exec_action_np[batch_row, chunk_step : chunk_step + 1])
                        obs_by_row[row] = step.observation
                        actions[row].append(exec_action_np[batch_row, chunk_step].reshape(-1).tolist())
                        rewards[row].append(float(step.reward))
                        successes[row].append(bool(step.success))
                        frame = _frame_from_obs(step.observation)
                        if frame is None:
                            frame = envs[row].render_frame()
                        frames[row].append(frame)
                        if step.success or step.terminated or step.truncated:
                            active[row] = False
                    step_count += 1
                    if step_count >= int(resolved_steps):
                        break
```

- [ ] **Step 9: Add chunk metadata to video summary**

In `scripts/grpo/eval_smolvla_baseline_vector_video.py`, add to `summary`:

```python
        "chunk_len": int(args.chunk_len),
        "rollout_execution": "chunk_open_loop" if int(args.chunk_len) > 1 else "one_step_queue_free",
```

- [ ] **Step 10: Commit**

```bash
git add src/smolvla_grpo/phase12_vector_eval.py scripts/grpo/eval_smolvla_baseline_vector_video.py tests/test_phase12_vector_eval.py
git commit -m "feat: run SmolVLA baseline eval with action chunks"
```

## Task 3: Add PBS Chunk Sweep Scripts

**Files:**
- Create: `scripts/grpo/submit_smolvla_baseline_pushv3_chunk_sweep.pbs`
- Create: `scripts/grpo/submit_smolvla_baseline_pushv3_chunk_sweep_all5.sh`
- Test: `tests/test_phase12_pbs_static.py`

- [ ] **Step 1: Add failing static test**

Append to `tests/test_phase12_pbs_static.py`:

```python
def test_smolvla_pushv3_chunk_sweep_pbs_contract() -> None:
    text = _read("submit_smolvla_baseline_pushv3_chunk_sweep.pbs")
    all5 = _read("submit_smolvla_baseline_pushv3_chunk_sweep_all5.sh")

    assert "#SBATCH" not in text
    assert "PBS_O_WORKDIR" in text
    assert "#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000" in text
    assert "#PBS -l walltime=02:00:00" in text
    assert 'export SMOLVLA_METAWORLD_RESET_MODE="${SMOLVLA_METAWORLD_RESET_MODE:-random_seeded}"' in text
    assert 'TASK="${PHASE58_TASK:-push-v3}"' in text
    assert 'EPISODES="${PHASE58_EPISODES:-25}"' in text
    assert 'N_ENVS="${PHASE58_N_ENVS:-3}"' in text
    assert 'MAX_STEPS="${PHASE58_MAX_STEPS:-180}"' in text
    assert 'CHUNK_LEN="${PHASE58_CHUNK_LEN:-2}"' in text
    assert "--chunk-len \"${CHUNK_LEN}\"" in text
    assert 'GPU_TELEMETRY_DIR="${OUT}/gpu_telemetry"' in text
    assert 'GPU_TELEMETRY_INTERVAL="${PHASE58_GPU_TELEMETRY_INTERVAL:-5}"' in text
    assert "nvidia_smi_samples.csv" in text
    assert "nvidia_smi_pmon.txt" in text
    assert "nvidia_smi_start.txt" in text
    assert "nvidia_smi_end.txt" in text
    assert "pbs_resource_snapshot.txt" in text
    assert "summarize_phase58_gpu_telemetry.py" in text
    assert "gpu_telemetry_summary.json" in text
    assert "command -v nvidia-smi" in text
    assert "command -v dcgmi" in text
    assert "stop_gpu_telemetry" in text
    assert "trap stop_gpu_telemetry EXIT" in text
    assert "|| true" in text
    assert "phase58_smolvla_baseline_pushv3_chunk_sweep_25ep_s1000_max180_5x1gpu" in text
    assert "SMOLVLA_PUSHV3_CHUNK_SWEEP_DONE" in text
    assert "chunks=(2 5 10 15 20)" in all5
    assert "qsub" in all5
    assert "PHASE58_CHUNK_LEN=${chunk}" in all5
```

- [ ] **Step 2: Run test and verify it fails**

Run:

```bash
pytest tests/test_phase12_pbs_static.py::test_smolvla_pushv3_chunk_sweep_pbs_contract -q
```

Expected: `FileNotFoundError`.

- [ ] **Step 3: Create one-chunk PBS script**

Create `scripts/grpo/submit_smolvla_baseline_pushv3_chunk_sweep.pbs`:

```bash
#!/usr/bin/env bash
# PBS (CX3): Push-v3 SmolVLA baseline chunk-length sweep, one chunk length per GPU job.
# Submit from repo root, usually through submit_smolvla_baseline_pushv3_chunk_sweep_all5.sh.
#PBS -N p58pushc
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -o logs/pbs/grpo/smolvla_pushv3_chunk_sweep.out

set -euo pipefail
export PYTHONUNBUFFERED=1

if [[ -f /etc/profile.d/modules.sh ]]; then . /etc/profile.d/modules.sh; fi
module purge >/dev/null 2>&1 || true
module load tools/prod
module load Python/3.12.3-GCCcore-13.3.0
module load Mesa/24.1.3-GCCcore-13.3.0

PROJECT_ROOT="$(cd "${PBS_O_WORKDIR:-/rds/general/user/aa6622/home/project}" && pwd)"
export SLURM_SUBMIT_DIR="${PROJECT_ROOT}"
cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/slurm/common_env.sh"
slurm_resolve_project_root "scripts/grpo/eval_smolvla_baseline_vector_video.py"
slurm_export_pythonpath
slurm_export_hf_torch_cache "phase58-pushv3-chunk-sweep"

export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"
export PATH="${PATH:-/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
export SMOLVLA_METAWORLD_RESET_MODE="${SMOLVLA_METAWORLD_RESET_MODE:-random_seeded}"

PYTHON_BIN="${SMOLVLA_PYTHON_BIN:-/rds/general/user/aa6622/home/.envs/lerobot_mw_py312/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "error: no usable PYTHON_BIN=${PYTHON_BIN}" >&2
  exit 2
fi

TASK="${PHASE58_TASK:-push-v3}"
SEED_START="${PHASE58_EVAL_SEED_START:-1000}"
EPISODES="${PHASE58_EPISODES:-25}"
N_ENVS="${PHASE58_N_ENVS:-3}"
MAX_STEPS="${PHASE58_MAX_STEPS:-180}"
CHUNK_LEN="${PHASE58_CHUNK_LEN:-2}"
FPS="${PHASE58_FPS:-20}"
SWEEP_ROOT="${PHASE58_OUT_ROOT:-${PROJECT_ROOT}/artifacts/phase58_smolvla_baseline_pushv3_chunk_sweep_25ep_s1000_max180_5x1gpu}"
OUT="${PHASE58_OUT:-${SWEEP_ROOT}/chunk_len_$(printf "%02d" "${CHUNK_LEN}")}"
BASE_CKPT="${PHASE58_CHECKPOINT:-/rds/general/user/aa6622/home/.cache/huggingface/hub/models--jadechoghari--smolvla_metaworld/snapshots/ef3089ecb84eeeb7d33fedab24f6c76180a68900}"
GPU_TELEMETRY_INTERVAL="${PHASE58_GPU_TELEMETRY_INTERVAL:-5}"
GPU_TELEMETRY_DIR="${OUT}/gpu_telemetry"
NVIDIA_SMI_PID=""
NVIDIA_PMON_PID=""
DCGMI_PID=""

stop_gpu_telemetry() {
  local status=$?
  set +e
  for pid in "${NVIDIA_SMI_PID}" "${NVIDIA_PMON_PID}" "${DCGMI_PID}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" >/dev/null 2>&1; then
      kill "${pid}" >/dev/null 2>&1 || true
      wait "${pid}" >/dev/null 2>&1 || true
    fi
  done
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi > "${GPU_TELEMETRY_DIR}/nvidia_smi_end.txt" 2>&1 || true
  fi
  if [[ -n "${PBS_JOBID:-}" ]] && command -v qstat >/dev/null 2>&1; then
    qstat -fx "${PBS_JOBID}" > "${GPU_TELEMETRY_DIR}/pbs_resource_snapshot.txt" 2>&1 || true
  fi
  if [[ -f "${GPU_TELEMETRY_DIR}/nvidia_smi_samples.csv" ]]; then
    "${PYTHON_BIN}" scripts/grpo/summarize_phase58_gpu_telemetry.py \
      --csv "${GPU_TELEMETRY_DIR}/nvidia_smi_samples.csv" \
      --output "${GPU_TELEMETRY_DIR}/gpu_telemetry_summary.json" || true
  fi
  exit "${status}"
}

start_gpu_telemetry() {
  mkdir -p "${GPU_TELEMETRY_DIR}"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi > "${GPU_TELEMETRY_DIR}/nvidia_smi_start.txt" 2>&1 || true
    nvidia-smi \
      --query-gpu=timestamp,index,uuid,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
      --format=csv,nounits \
      -l "${GPU_TELEMETRY_INTERVAL}" \
      > "${GPU_TELEMETRY_DIR}/nvidia_smi_samples.csv" \
      2> "${GPU_TELEMETRY_DIR}/nvidia_smi_samples.err" &
    NVIDIA_SMI_PID=$!
    nvidia-smi pmon -s um -d "${GPU_TELEMETRY_INTERVAL}" -o DT \
      > "${GPU_TELEMETRY_DIR}/nvidia_smi_pmon.txt" \
      2> "${GPU_TELEMETRY_DIR}/nvidia_smi_pmon.err" &
    NVIDIA_PMON_PID=$!
  else
    echo "nvidia-smi not found on PATH" > "${GPU_TELEMETRY_DIR}/nvidia_smi_start.txt"
  fi
  if command -v dcgmi >/dev/null 2>&1; then
    dcgmi dmon -d "${GPU_TELEMETRY_INTERVAL}" \
      > "${GPU_TELEMETRY_DIR}/dcgmi_dmon.txt" \
      2> "${GPU_TELEMETRY_DIR}/dcgmi_dmon.err" &
    DCGMI_PID=$!
  fi
}

mkdir -p "${OUT}" "${GPU_TELEMETRY_DIR}" logs/pbs/grpo
echo "[phase58] job=${PBS_JOBID:-local} task=${TASK} episodes=${EPISODES} n_envs=${N_ENVS} chunk_len=${CHUNK_LEN} max_steps=${MAX_STEPS} out=${OUT}"
trap stop_gpu_telemetry EXIT
start_gpu_telemetry

"${PYTHON_BIN}" scripts/grpo/eval_smolvla_baseline_vector_video.py \
  --checkpoint "${BASE_CKPT}" \
  --output-dir "${OUT}" \
  --task "${TASK}" \
  --episodes "${EPISODES}" \
  --eval-seed-start "${SEED_START}" \
  --n-envs "${N_ENVS}" \
  --max-steps "${MAX_STEPS}" \
  --chunk-len "${CHUNK_LEN}" \
  --fps "${FPS}"

test -f "${OUT}/eval_summary.json"
echo "SMOLVLA_PUSHV3_CHUNK_SWEEP_DONE out=${OUT}"
```

- [ ] **Step 4: Create all-five launcher**

Create `scripts/grpo/submit_smolvla_baseline_pushv3_chunk_sweep_all5.sh`:

```bash
#!/usr/bin/env bash
# Submit five independent Push-v3 SmolVLA chunk-length baseline jobs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

mkdir -p logs/pbs/grpo
chunks=(2 5 10 15 20)
for chunk in "${chunks[@]}"; do
  qsub \
    -N "p58c${chunk}" \
    -o "logs/pbs/grpo/smolvla_pushv3_chunk_${chunk}.out" \
    -v "PHASE58_CHUNK_LEN=${chunk}" \
    scripts/grpo/submit_smolvla_baseline_pushv3_chunk_sweep.pbs
done
```

- [ ] **Step 5: Make launcher executable**

Run:

```bash
chmod +x scripts/grpo/submit_smolvla_baseline_pushv3_chunk_sweep_all5.sh
```

- [ ] **Step 6: Run static test**

Run:

```bash
pytest tests/test_phase12_pbs_static.py::test_smolvla_pushv3_chunk_sweep_pbs_contract -q
```

Expected: `1 passed`.

- [ ] **Step 7: Commit**

```bash
git add scripts/grpo/submit_smolvla_baseline_pushv3_chunk_sweep.pbs scripts/grpo/submit_smolvla_baseline_pushv3_chunk_sweep_all5.sh tests/test_phase12_pbs_static.py
git commit -m "feat: add Push-v3 chunk sweep PBS jobs"
```

## Task 4: Add GPU Telemetry Summary Parser

**Files:**
- Create: `scripts/grpo/summarize_phase58_gpu_telemetry.py`
- Create: `tests/test_phase58_gpu_telemetry.py`

- [ ] **Step 1: Add failing parser tests**

Create `tests/test_phase58_gpu_telemetry.py`:

```python
from __future__ import annotations

import csv
import json
from pathlib import Path


def test_gpu_telemetry_summary_reports_max_mean_p95_and_memory(tmp_path: Path) -> None:
    from scripts.grpo.summarize_phase58_gpu_telemetry import summarize_nvidia_smi_csv

    csv_path = tmp_path / "nvidia_smi_samples.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "timestamp",
                "index",
                "uuid",
                "name",
                "utilization.gpu [%]",
                "utilization.memory [%]",
                "memory.used [MiB]",
                "memory.total [MiB]",
                "power.draw [W]",
                "temperature.gpu",
            ]
        )
        writer.writerow(["2026/05/17 10:00:00.000", "0", "GPU-a", "RTX6000", "10", "20", "1000", "24000", "70.5", "40"])
        writer.writerow(["2026/05/17 10:00:05.000", "0", "GPU-a", "RTX6000", "90", "70", "12000", "24000", "220.0", "62"])
        writer.writerow(["2026/05/17 10:00:10.000", "0", "GPU-a", "RTX6000", "100", "80", "18000", "24000", "250.0", "68"])

    summary = summarize_nvidia_smi_csv(csv_path)

    assert summary["sample_count"] == 3
    assert summary["max_gpu_utilization_pct"] == 100.0
    assert summary["mean_gpu_utilization_pct"] == 200.0 / 3.0
    assert summary["p95_gpu_utilization_pct"] == 100.0
    assert summary["samples_gpu_utilization_ge_80_pct"] == 2
    assert summary["max_memory_used_mib"] == 18000.0
    assert summary["max_memory_used_fraction"] == 0.75
    assert summary["max_memory_utilization_pct"] == 80.0
    assert summary["max_power_draw_w"] == 250.0


def test_gpu_telemetry_cli_writes_json(tmp_path: Path) -> None:
    from scripts.grpo.summarize_phase58_gpu_telemetry import main

    csv_path = tmp_path / "nvidia_smi_samples.csv"
    out_path = tmp_path / "gpu_telemetry_summary.json"
    csv_path.write_text(
        "timestamp,index,uuid,name,utilization.gpu [%],utilization.memory [%],memory.used [MiB],memory.total [MiB],power.draw [W],temperature.gpu\n"
        "2026/05/17 10:00:00.000,0,GPU-a,RTX6000,50,40,8000,24000,180,55\n",
        encoding="utf-8",
    )

    assert main(["--csv", str(csv_path), "--output", str(out_path)]) == 0

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["sample_count"] == 1
    assert payload["max_gpu_utilization_pct"] == 50.0
    assert payload["telemetry_csv"] == str(csv_path.resolve())
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
pytest tests/test_phase58_gpu_telemetry.py -q
```

Expected: `ModuleNotFoundError` or missing script.

- [ ] **Step 3: Create parser script**

Create `scripts/grpo/summarize_phase58_gpu_telemetry.py`:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Iterable


def _float_cell(row: dict[str, str], keys: Iterable[str]) -> float | None:
    for key in keys:
        if key not in row:
            continue
        raw = str(row[key]).strip()
        if not raw or raw == "-":
            return None
        raw = raw.replace("%", "").replace("MiB", "").replace("W", "").strip()
        try:
            return float(raw)
        except ValueError:
            return None
    return None


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(math.ceil((pct / 100.0) * len(ordered))) - 1
    idx = max(0, min(idx, len(ordered) - 1))
    return float(ordered[idx])


def summarize_nvidia_smi_csv(csv_path: Path) -> dict[str, object]:
    gpu_utils: list[float] = []
    mem_utils: list[float] = []
    mem_used: list[float] = []
    mem_total: list[float] = []
    power_draw: list[float] = []
    temperatures: list[float] = []
    gpu_names: set[str] = set()
    gpu_uuids: set[str] = set()

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gpu_name = str(row.get("name", "")).strip()
            gpu_uuid = str(row.get("uuid", "")).strip()
            if gpu_name:
                gpu_names.add(gpu_name)
            if gpu_uuid:
                gpu_uuids.add(gpu_uuid)

            gpu_util = _float_cell(row, ("utilization.gpu [%]", "utilization.gpu"))
            mem_util = _float_cell(row, ("utilization.memory [%]", "utilization.memory"))
            used = _float_cell(row, ("memory.used [MiB]", "memory.used"))
            total = _float_cell(row, ("memory.total [MiB]", "memory.total"))
            power = _float_cell(row, ("power.draw [W]", "power.draw"))
            temp = _float_cell(row, ("temperature.gpu",))

            if gpu_util is not None:
                gpu_utils.append(gpu_util)
            if mem_util is not None:
                mem_utils.append(mem_util)
            if used is not None:
                mem_used.append(used)
            if total is not None:
                mem_total.append(total)
            if power is not None:
                power_draw.append(power)
            if temp is not None:
                temperatures.append(temp)

    max_mem_used = max(mem_used) if mem_used else 0.0
    max_mem_total = max(mem_total) if mem_total else 0.0
    return {
        "telemetry_csv": str(csv_path.resolve()),
        "sample_count": len(gpu_utils),
        "gpu_names": sorted(gpu_names),
        "gpu_uuids": sorted(gpu_uuids),
        "max_gpu_utilization_pct": float(max(gpu_utils) if gpu_utils else 0.0),
        "mean_gpu_utilization_pct": float(mean(gpu_utils) if gpu_utils else 0.0),
        "p95_gpu_utilization_pct": _percentile(gpu_utils, 95.0),
        "samples_gpu_utilization_ge_80_pct": int(sum(1 for value in gpu_utils if value >= 80.0)),
        "samples_gpu_utilization_ge_95_pct": int(sum(1 for value in gpu_utils if value >= 95.0)),
        "max_memory_utilization_pct": float(max(mem_utils) if mem_utils else 0.0),
        "mean_memory_utilization_pct": float(mean(mem_utils) if mem_utils else 0.0),
        "max_memory_used_mib": float(max_mem_used),
        "max_memory_total_mib": float(max_mem_total),
        "max_memory_used_fraction": float(max_mem_used / max_mem_total) if max_mem_total > 0 else 0.0,
        "max_power_draw_w": float(max(power_draw) if power_draw else 0.0),
        "mean_power_draw_w": float(mean(power_draw) if power_draw else 0.0),
        "max_temperature_gpu_c": float(max(temperatures) if temperatures else 0.0),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = summarize_nvidia_smi_csv(args.csv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        "gpu_telemetry_summary_ok",
        f"samples={summary['sample_count']}",
        f"max_gpu={summary['max_gpu_utilization_pct']:.1f}",
        f"mean_gpu={summary['mean_gpu_utilization_pct']:.1f}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run parser tests**

Run:

```bash
pytest tests/test_phase58_gpu_telemetry.py -q
```

Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add scripts/grpo/summarize_phase58_gpu_telemetry.py tests/test_phase58_gpu_telemetry.py
git commit -m "feat: summarize Phase58 GPU telemetry"
```

## Task 5: Add Runner Parse/Smoke Guards

**Files:**
- Modify: `tests/test_phase12_vector_eval.py`
- Modify: `tests/test_phase12_pbs_static.py`

- [ ] **Step 1: Add CLI parse test for baseline runner**

Append to `tests/test_phase12_vector_eval.py`:

```python
def test_baseline_vector_video_parse_accepts_chunk_len(tmp_path):
    from scripts.grpo.eval_smolvla_baseline_vector_video import parse_args

    args = parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--task",
            "push-v3",
            "--episodes",
            "25",
            "--eval-seed-start",
            "1000",
            "--n-envs",
            "3",
            "--max-steps",
            "180",
            "--chunk-len",
            "20",
        ]
    )

    assert args.task == "push-v3"
    assert args.episodes == 25
    assert args.eval_seed_start == 1000
    assert args.n_envs == 3
    assert args.max_steps == 180
    assert args.chunk_len == 20
```

- [ ] **Step 2: If parse_args currently takes no argv, change it**

In `scripts/grpo/eval_smolvla_baseline_vector_video.py`, change:

```python
def parse_args() -> argparse.Namespace:
```

to:

```python
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
```

and change:

```python
    return parser.parse_args()
```

to:

```python
    return parser.parse_args(argv)
```

- [ ] **Step 3: Run parse and static tests**

Run:

```bash
pytest tests/test_phase12_vector_eval.py::test_baseline_vector_video_parse_accepts_chunk_len tests/test_phase12_pbs_static.py::test_smolvla_pushv3_chunk_sweep_pbs_contract -q
```

Expected: `2 passed`.

- [ ] **Step 4: Run broader targeted tests**

Run:

```bash
pytest tests/test_phase12_vector_eval.py tests/test_phase12_pbs_static.py tests/test_grpo_policy_wrapper_chunk.py -q
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/grpo/eval_smolvla_baseline_vector_video.py tests/test_phase12_vector_eval.py tests/test_phase12_pbs_static.py
git commit -m "test: guard Push-v3 chunk sweep contracts"
```

## Task 6: Smoke One Short GPU Job

**Files:**
- No source edits unless smoke exposes script bug.

- [ ] **Step 1: Submit 1-episode chunk-2 smoke**

Run from `project/`:

```bash
qsub \
  -N p58c2smoke \
  -o logs/pbs/grpo/smolvla_pushv3_chunk_2_smoke.out \
  -v "PHASE58_CHUNK_LEN=2,PHASE58_EPISODES=1,PHASE58_EVAL_SEED_START=1000,PHASE58_OUT_ROOT=/rds/general/user/aa6622/home/project/artifacts/phase58_smolvla_baseline_pushv3_chunk_sweep_smoke" \
  scripts/grpo/submit_smolvla_baseline_pushv3_chunk_sweep.pbs
```

Expected: PBS prints one job id, e.g. `2707000.cx3-pbs`.

- [ ] **Step 2: Monitor smoke**

Run:

```bash
qstat -u "$USER"
```

Expected while running: job name `p58c2smoke` appears in queue/running state.

- [ ] **Step 3: Verify smoke artifact**

After job exits, run:

```bash
python - <<'PY'
import json
from pathlib import Path
root = Path("artifacts/phase58_smolvla_baseline_pushv3_chunk_sweep_smoke/chunk_len_02")
summary = json.loads((root / "eval_summary.json").read_text())
assert summary["task"] == "push-v3"
assert summary["episodes"] == 1
assert summary["chunk_len"] == 2
assert summary["max_steps"] == 180
assert summary["n_envs"] == 3
assert summary["reset_randomization_mode"] == "random_seeded"
assert summary["video_enabled"] is True
video = root / "episode_0000_seed_1000" / "selected_action_rollout.mp4"
assert video.exists(), video
telemetry = root / "gpu_telemetry"
assert (telemetry / "nvidia_smi_start.txt").exists()
assert (telemetry / "nvidia_smi_end.txt").exists()
assert (telemetry / "pbs_resource_snapshot.txt").exists()
assert (telemetry / "gpu_telemetry_summary.json").exists()
gpu = json.loads((telemetry / "gpu_telemetry_summary.json").read_text())
assert gpu["sample_count"] >= 1
print(
    "phase58_smoke_ok",
    summary["pc_success"],
    "max_gpu",
    gpu["max_gpu_utilization_pct"],
    "mean_gpu",
    gpu["mean_gpu_utilization_pct"],
    video,
)
PY
```

Expected: `phase58_smoke_ok ... max_gpu ... mean_gpu ... selected_action_rollout.mp4`.

- [ ] **Step 4: Fix only scoped smoke failures**

If the smoke fails because `--chunk-len` is missing, import names are wrong, `n_action_steps` did not thread through, telemetry cleanup hangs, or telemetry summary parsing fails, fix the exact script/helper bug and rerun Task 4/Task 5 tests before resubmitting smoke. Do not change checkpoint, seeds, task, reset mode, or disable GPU telemetry to make smoke pass.

- [ ] **Step 5: Commit smoke bugfix if needed**

Only if Step 4 required a code edit:

```bash
git add src/smolvla_grpo/phase12_vector_eval.py scripts/grpo/eval_smolvla_baseline_vector_video.py scripts/grpo/submit_smolvla_baseline_pushv3_chunk_sweep.pbs tests/test_phase12_vector_eval.py tests/test_phase12_pbs_static.py
git commit -m "fix: stabilize Push-v3 chunk sweep smoke"
```

## Task 7: Submit Full Five-GPU Sweep

**Files:**
- No source edits.

- [ ] **Step 1: Submit five chunk jobs**

Run from `project/`:

```bash
scripts/grpo/submit_smolvla_baseline_pushv3_chunk_sweep_all5.sh
```

Expected: five PBS job ids, one each for chunk lengths `2`, `5`, `10`, `15`, `20`.

- [ ] **Step 2: Record job ids**

Run:

```bash
qstat -u "$USER"
```

Expected: jobs named `p58c2`, `p58c5`, `p58c10`, `p58c15`, `p58c20` are queued or running.

- [ ] **Step 3: Verify full output after completion**

Run:

```bash
python - <<'PY'
import json
from pathlib import Path
root = Path("artifacts/phase58_smolvla_baseline_pushv3_chunk_sweep_25ep_s1000_max180_5x1gpu")
for chunk in (2, 5, 10, 15, 20):
    d = root / f"chunk_len_{chunk:02d}"
    summary = json.loads((d / "eval_summary.json").read_text())
    assert summary["task"] == "push-v3"
    assert summary["episodes"] == 25
    assert summary["eval_seed_start"] == 1000
    assert summary["eval_seed_end"] == 1024
    assert summary["n_envs"] == 3
    assert summary["chunk_len"] == chunk
    assert summary["max_steps"] == 180
    assert summary["video_enabled"] is True
    assert len(summary["episodes_rows"]) == 25
    videos = list(d.glob("episode_*_seed_*/selected_action_rollout.mp4"))
    assert len(videos) == 25, (chunk, len(videos))
    telemetry = d / "gpu_telemetry" / "gpu_telemetry_summary.json"
    assert telemetry.exists(), telemetry
    gpu = json.loads(telemetry.read_text())
    assert gpu["sample_count"] >= 1
    print(
        chunk,
        "pc_success",
        summary["pc_success"],
        "avg_sum_reward",
        summary["avg_sum_reward"],
        "max_gpu",
        gpu["max_gpu_utilization_pct"],
        "mean_gpu",
        gpu["mean_gpu_utilization_pct"],
        "max_mem_mib",
        gpu["max_memory_used_mib"],
        "max_power_w",
        gpu["max_power_draw_w"],
    )
PY
```

Expected: five lines, one per chunk length.

- [ ] **Step 4: Write quick result note**

Create `artifacts/phase58_smolvla_baseline_pushv3_chunk_sweep_25ep_s1000_max180_5x1gpu/README.md`:

```markdown
# Phase58 Push-v3 SmolVLA Chunk Sweep

Config:
- Task: `push-v3`
- Episodes per chunk length: `25`
- Seeds: `1000..1024`
- `n_envs`: `3`
- `max_steps`: `180`
- Reset mode: `random_seeded`
- Videos: enabled
- Chunk lengths: `2, 5, 10, 15, 20`

Results are in:
- `chunk_len_02/eval_summary.json`
- `chunk_len_05/eval_summary.json`
- `chunk_len_10/eval_summary.json`
- `chunk_len_15/eval_summary.json`
- `chunk_len_20/eval_summary.json`

GPU utilization summaries are in:
- `chunk_len_02/gpu_telemetry/gpu_telemetry_summary.json`
- `chunk_len_05/gpu_telemetry/gpu_telemetry_summary.json`
- `chunk_len_10/gpu_telemetry/gpu_telemetry_summary.json`
- `chunk_len_15/gpu_telemetry/gpu_telemetry_summary.json`
- `chunk_len_20/gpu_telemetry/gpu_telemetry_summary.json`
```

- [ ] **Step 5: Do not commit artifacts unless requested**

Generated videos and summaries stay uncommitted unless the user explicitly asks to preserve artifacts in git.

## Final Verification

Run before reporting implementation complete:

```bash
pytest tests/test_phase12_vector_eval.py tests/test_phase12_pbs_static.py tests/test_grpo_policy_wrapper_chunk.py tests/test_phase58_gpu_telemetry.py -q
```

Expected: all pass.

Run after full jobs finish:

```bash
python - <<'PY'
import json
from pathlib import Path
root = Path("artifacts/phase58_smolvla_baseline_pushv3_chunk_sweep_25ep_s1000_max180_5x1gpu")
rows = []
for chunk in (2, 5, 10, 15, 20):
    summary = json.loads((root / f"chunk_len_{chunk:02d}" / "eval_summary.json").read_text())
    gpu = json.loads((root / f"chunk_len_{chunk:02d}" / "gpu_telemetry" / "gpu_telemetry_summary.json").read_text())
    rows.append((
        chunk,
        summary["pc_success"],
        summary["avg_sum_reward"],
        summary["avg_max_reward"],
        gpu["max_gpu_utilization_pct"],
        gpu["mean_gpu_utilization_pct"],
        gpu["p95_gpu_utilization_pct"],
        gpu["max_memory_used_mib"],
        gpu["max_power_draw_w"],
    ))
for row in rows:
    print(
        "chunk_len=%02d pc_success=%.1f avg_sum_reward=%.3f avg_max_reward=%.3f max_gpu=%.1f mean_gpu=%.1f p95_gpu=%.1f max_mem_mib=%.1f max_power_w=%.1f"
        % row
    )
PY
```

Expected: five metric rows.

## Self-Review

Spec coverage:
- 25 episodes: Task 3 PBS default `PHASE58_EPISODES=25`.
- V3 task: Task 3 default `PHASE58_TASK=push-v3`.
- Standard SmolVLA baseline: Task 2 keeps `load_bundle_for_grpo()` and baseline checkpoint path.
- Maximum parallelism with five GPUs: Task 3 launcher submits five independent 1-GPU PBS jobs.
- `n_envs=3`: Task 3 default `PHASE58_N_ENVS=3`.
- Chunk lengths `2,5,10,15,20`: Task 3 launcher chunk list.
- `max_steps=180`: Task 3 default `PHASE58_MAX_STEPS=180`.
- Videos: Task 2 preserves `write_phase12_episode_video()`, Task 6 verifies 25 videos per chunk.
- Default random seeded: Task 3 exports `SMOLVLA_METAWORLD_RESET_MODE=random_seeded`, Task 6 verifies summary.
- GPU usage visibility: Task 3 records `nvidia-smi` CSV, per-process `pmon`, optional `dcgmi`, and PBS resource snapshot per chunk; Task 4 summarizes max/mean/p95 GPU utilization, memory, and power; Task 6/7 verify telemetry artifacts exist.
- "Max it out" evidence: full run cannot force saturation by logging alone, but summaries reveal whether chunked batches actually hit high SM utilization. If utilization remains low, next plan should increase inference batch/work per GPU (`n_envs`, concurrent waves, or task multiplexing) based on telemetry.

Placeholder scan:
- No `TBD`, `TODO`, `similar to`, or unspecified error handling.

Type consistency:
- `chunk_len` name used in Python summaries.
- `CHUNK_LEN` shell variable maps to `--chunk-len`.
- `PHASE58_CHUNK_LEN` launcher env maps to PBS script.
- `GPU_TELEMETRY_DIR` shell variable maps to per-chunk `gpu_telemetry/`.
- `gpu_telemetry_summary.json` field names match `tests/test_phase58_gpu_telemetry.py`.
