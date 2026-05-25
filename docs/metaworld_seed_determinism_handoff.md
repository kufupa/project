# MetaWorld Seed Determinism Handoff (Phase111 / RLinf-style)

Purpose: give a zero-context LLM enough detail to (re)implement deterministic seeded MetaWorld rollouts and eval behavior.

## Problem We Had

`env.reset(seed=...)` alone was not enough to keep runs fully reproducible across long sweeps.

Why:
- MetaWorld / MuJoCo stack can still consume global RNG state (`random`, `numpy`, `torch`) in paths outside strict env reset.
- Task binding order matters (`set_task` before reset).
- Legacy fallback reset paths (without `seed`) silently break deterministic guarantees.
- Vectorized envs require explicit per-row seed lists; implicit defaults drift.

## Determinism Contract (Must Enforce)

Per episode, in this exact order:
1. `seed_metaworld_process(reset_seed)` (global RNGs)
2. `env.set_task(...)` (deterministic task choice)
3. `env.reset(seed=reset_seed)` (strict Gymnasium API)
4. `env.step(...)` with action clipping to env bounds

If any part is missing, determinism degrades.

---

## Relevant Implementation Snippets

These snippets are copied from the working implementation. A future LLM should be
able to rebuild the behavior from this document alone.

### Shared deterministic helpers

```python
def seed_metaworld_process(seed: int) -> None:
    s = int(seed)
    random.seed(s)
    np.random.seed(s)
    import torch

    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def gymnasium_reset_strict(env: Any, seed: int) -> Any:
    try:
        return env.reset(seed=seed)
    except TypeError as exc:
        raise RuntimeError(
            "env.reset(seed=...) is required for reproducible MetaWorld rollouts."
        ) from exc
```

### Oracle/eval episode loop

```python
reset_seed = int(args.seed) + int(episode_index)
seed_metaworld_process(reset_seed)

if tasks:
    env.set_task(tasks[episode_index % len(tasks)])

out = gymnasium_reset_strict(env, reset_seed)
obs = out[0] if isinstance(out, tuple) else out
```

### GRPO custom-env rollout loop

```python
for r in range(group_size):
    gen.manual_seed(int(reset_seed) * 1000003 + r * 7919)
    seed_metaworld_process(int(reset_seed))
    env_h.set_task_for_episode(episode_index)
    obs = env_h.reset(reset_seed)
```

### MetaWorld env wrapper reset mode

```python
def reset(self, seed: int | None = None, **kwargs):
    self._env._freeze_rand_vec = self.reset_randomization_mode == "fixed"
    if self.reset_randomization_mode == "random_seeded" and seed is not None:
        self._env.seed(int(seed))
        if hasattr(self._env, "seeded_rand_vec"):
            self._env.seeded_rand_vec = True
    elif hasattr(self._env, "seeded_rand_vec"):
        self._env.seeded_rand_vec = False

    raw_obs, _info = self._env.reset(seed=seed)
    return self._format_raw_obs(raw_obs), {"is_success": False}
```

### Vector env reset

```python
def reset(self, reset_seed: int):
    seeds = [int(reset_seed)] * self.n_envs
    obs, _info = self.vec_env.reset(seed=seeds)
    return obs


def reset_many(self, reset_seeds: Sequence[int]):
    seeds = [int(seed) for seed in reset_seeds]
    if len(seeds) != self.n_envs:
        raise ValueError(f"expected {self.n_envs} seeds; got {len(seeds)}")
    obs, _info = self.vec_env.reset(seed=seeds)
    return obs
```

### Action clipping before env step

```python
action_np = np.asarray(action, dtype=np.float32)
action_np = np.clip(action_np, env.action_space.low, env.action_space.high)
obs, reward, terminated, truncated, info = env.step(action_np)
```

### Minimal tests

```python
def test_seed_metaworld_process_numpy_repeatable():
    seed_metaworld_process(12345)
    a = float(np.random.random())
    seed_metaworld_process(12345)
    b = float(np.random.random())
    assert a == b


def test_gymnasium_reset_strict_raises_on_legacy_env():
    class LegacyEnv:
        def reset(self):
            return ({"obs": 1}, {})

    with pytest.raises(RuntimeError, match=r"env\.reset\(seed="):
        gymnasium_reset_strict(LegacyEnv(), seed=0)
```

---

## Common Failure Modes (Seen in Practice)

- Only calling `env.reset(seed=...)` (no global seeding): run-to-run drift.
- Setting task after reset: different internal initializations.
- Reusing stale env workers without explicit per-row seeds.
- Legacy env wrappers that drop the `seed` kwarg.
- Mixing fixed/random reset modes without explicit mode field.

---

## Verification Checklist

Run these checks after implementation:

1. Same seed twice yields identical first NumPy sample after reseed.
2. Strict reset wrapper raises on envs without `reset(seed=...)`.
3. Rollout A and rollout B with same `(task, episode_index, reset_seed)` produce matching first-step obs/action/reward signatures.
4. Vector reset with seed list reproduces per-row initial states.
5. Long sweep repeat (e.g., 25+ episodes) preserves deterministic ordering and metrics for same configuration.

---

## Practical Notes

- If you support multiple reset modes, default to `"random_seeded"` for deterministic randomized layouts.
- Persist seed metadata per episode (`base_seed`, `reset_seed`, `episode_index`, `task`) in artifacts.
- Keep deterministic behavior in one shared module (`metaworld_determinism.py`) and import it everywhere to avoid drift between codepaths.
