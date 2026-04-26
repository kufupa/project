# Phase12 WM View Final Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Phase12 train feed JEPA-WM the Phase08-correct V-only `corner2` view for start/goal/decode while preserving LeRobot V+H `corner2` view for SmolVLA policy and human videos.

**Architecture:** Use one captured policy frame per env state as source of truth. Phase12 reads `obs["pixels"][0]` from the LeRobot observation (V+H), stores it for policy/videos, derives `wm_rgb = np.flip(policy_rgb, axis=1)` for JEPA-WM V-only, and never performs a second MuJoCo render for the WM view. Oracle and selected rollout record both streams from the same captured policy frame. Selected rollout videos intentionally switch from separate `render_frame()` calls to captured obs pixels so policy/proc/video/WM debug all align to one frame source.

**Tech Stack:** Python 3.12, NumPy, PyTorch, pytest, LeRobot MetaWorld adapter, JEPA-WM, existing Phase12 trainer.

---

## Critical Review Of `/homes/aa6622/.cursor/plans/phase12-wm-view-fix_a2dc3fc6.plan.md`

Good:

- Correct root: Phase12 conflates policy V+H view with WM V-only view.
- Correct minimal behavior: keep videos/policy V+H, route WM score/goal/decode through V-only.
- Good red-test intent: boundary tests before implementation.

Problems:

- Helper lives in `train_phase12_wm_chunk_grpo.py`. That hides reusable pixel contract in huge trainer. Better: `src/smolvla_grpo/phase12_pixels.py`.
- A separate `render_frame_for_wm()` is only safe if it returns a pair from the same raw render. Calling `render_frame()` and `render_frame_for_wm()` separately risks two MuJoCo renders of the same state, reintroducing render jitter. Better: use `obs["pixels"][0]` / captured policy frame once, then derive WM frame by H-unflip.
- Static string tests are brittle. Use behavioral tests first; static guards optional.
- Plan does not record oracle `wm_frames`. It derives goal WM frames from policy frames each time. Better: oracle rollout records both streams once, then goal encode consumes `oracle["wm_frames"]`.
- Cleanup task removes debug instrumentation across five files. Too broad for root fix; risks deleting useful diagnostics. Keep debug logs until separate cleanup request.
- Verification lacks explicit blur/edge metric vs old known-bad strip. Add quick numeric check plus visual artifact path check.

Final plan below keeps good intent, lowers blast radius, makes contracts explicit.

## Files

- Create: `src/smolvla_grpo/phase12_pixels.py`
  - Pure RGB conversion helpers: observation -> policy V+H, raw -> policy V+H, raw -> WM V-only, policy V+H -> WM V-only.
- Create: `tests/test_phase12_pixels.py`
  - Fast unit tests for all pixel transforms.
- Modify: `src/smolvla_grpo/lerobot_metaworld_adapter.py`
  - No behavioral change. Only adapter tests assert existing policy-frame contract stays V+H.
- Modify: `tests/test_grpo_lerobot_adapter.py`
  - Make fake render asymmetric.
  - Assert `DeferredLeRobotMetaworldEnv.reset()` and `OfficialLeRobotMetaWorldGRPORollout.reset()` expose LeRobot V+H policy pixels.
- Modify: `scripts/grpo/train_phase12_wm_chunk_grpo.py`
  - Record oracle `wm_frames`.
  - Initialize `_Phase12SelectedRolloutEnv` from one captured policy frame.
  - Store `rollout_env.wm_frames`.
  - Feed WM frame as root `"image"` to scoring/decode.
  - Use `oracle["wm_frames"]` for goal encode.
  - Use `rollout_env.wm_frames` for real-vs-pred strip.
  - Record manifest/episode metadata for pixel contracts.
- Modify: `tests/test_phase12_training_loop.py`
  - Behavioral tests for root scoring image, oracle WM frames, decode real frame source.
- Modify: `tests/test_phase12_trainer_static.py`
  - Manifest contract test only.

## Non-Goals

- Do not change action profile semantics.
- Do not switch default objective from visual+proprio to visual-only.
- Do not replace Phase12 train with Phase08 `rollout_with_chunks`.
- Do not remove debug instrumentation in this fix.
- Do not add second env or replay reset. Same underlying env state stays source of truth.
- Do not call two render methods for the same state. One captured policy frame produces both policy/video and WM views.
- Do not call `env_h.render_frame()` in `_rollout_phase12_oracle()` or `_Phase12SelectedRolloutEnv.step()` after this fix. Captured observation pixels are canonical.
- Do not rebuild renderer to 224 yet. Orientation mismatch is primary. Add 224 render only if post-fix smoke still blurs.

## Task 1: Add Pixel Contract Helpers

**Files:**
- Create: `src/smolvla_grpo/phase12_pixels.py`
- Create: `tests/test_phase12_pixels.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_phase12_pixels.py`:

```python
from __future__ import annotations

import numpy as np

from smolvla_grpo.phase12_pixels import (
    policy_rgb_from_obs,
    policy_rgb_from_raw_corner2,
    to_rgb_uint8,
    wm_rgb_from_policy_rgb_corner2,
    wm_rgb_from_raw_corner2,
)


def _raw_frame() -> np.ndarray:
    return np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)


def test_policy_rgb_from_raw_corner2_is_vhflip() -> None:
    raw = _raw_frame()

    out = policy_rgb_from_raw_corner2(raw)

    np.testing.assert_array_equal(out, np.flip(raw, (0, 1)))
    assert out.flags.c_contiguous


def test_wm_rgb_from_raw_corner2_is_vflip_only() -> None:
    raw = _raw_frame()

    out = wm_rgb_from_raw_corner2(raw)

    np.testing.assert_array_equal(out, np.flip(raw, 0))
    assert out.flags.c_contiguous


def test_wm_rgb_from_policy_rgb_corner2_removes_horizontal_flip_only() -> None:
    raw = _raw_frame()
    policy = policy_rgb_from_raw_corner2(raw)

    out = wm_rgb_from_policy_rgb_corner2(policy)

    np.testing.assert_array_equal(out, wm_rgb_from_raw_corner2(raw))
    assert out.flags.c_contiguous


def test_to_rgb_uint8_drops_alpha_and_scales_unit_float() -> None:
    rgba = np.zeros((2, 2, 4), dtype=np.float32)
    rgba[..., 1] = 1.0
    rgba[..., 3] = 0.25

    out = to_rgb_uint8(rgba)

    assert out.dtype == np.uint8
    assert out.shape == (2, 2, 3)
    assert int(out[..., 1].max()) == 255


def test_policy_rgb_from_obs_extracts_single_env_pixels() -> None:
    raw = _raw_frame()
    obs = {"pixels": raw[None]}

    out = policy_rgb_from_obs(obs)

    np.testing.assert_array_equal(out, raw)
    assert out.flags.c_contiguous


def test_policy_rgb_from_obs_accepts_unbatched_pixels() -> None:
    raw = _raw_frame()
    obs = {"pixels": raw}

    out = policy_rgb_from_obs(obs)

    np.testing.assert_array_equal(out, raw)
    assert out.flags.c_contiguous
```

- [ ] **Step 2: Run test to verify fail**

Run:

```bash
cd /vol/bitbucket/aa6622/project
PYTHONPATH=src /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python -m pytest tests/test_phase12_pixels.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'smolvla_grpo.phase12_pixels'`.

- [ ] **Step 3: Add implementation**

Create `src/smolvla_grpo/phase12_pixels.py`:

```python
"""Phase12 pixel-contract helpers.

Policy RGB matches LeRobot/SmolVLA: corner2 + vertical+horizontal flip.
WM RGB matches JEPA-WM/Phase08: corner2 + vertical flip only.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def to_rgb_uint8(image: Any) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[-1] not in (3, 4):
        raise ValueError(f"image must be HxWx3/4, got {arr.shape}")
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating) and float(np.max(arr)) <= 1.5:
            arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def policy_rgb_from_raw_corner2(raw_rgb: Any) -> np.ndarray:
    return np.ascontiguousarray(np.flip(to_rgb_uint8(raw_rgb), (0, 1)))


def wm_rgb_from_raw_corner2(raw_rgb: Any) -> np.ndarray:
    return np.ascontiguousarray(np.flip(to_rgb_uint8(raw_rgb), 0))


def wm_rgb_from_policy_rgb_corner2(policy_rgb: Any) -> np.ndarray:
    return np.ascontiguousarray(np.flip(to_rgb_uint8(policy_rgb), 1))


def policy_rgb_from_obs(obs: Any) -> np.ndarray:
    if not isinstance(obs, dict) or "pixels" not in obs:
        raise KeyError("observation must contain 'pixels'")
    pixels = np.asarray(obs["pixels"])
    if pixels.ndim == 4:
        if pixels.shape[0] < 1:
            raise ValueError("observation pixels batch is empty")
        pixels = pixels[0]
    return to_rgb_uint8(pixels)
```

- [ ] **Step 4: Run test to verify pass**

Run:

```bash
cd /vol/bitbucket/aa6622/project
PYTHONPATH=src /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python -m pytest tests/test_phase12_pixels.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/smolvla_grpo/phase12_pixels.py tests/test_phase12_pixels.py
git commit -m "fix: add phase12 pixel contracts"
```

## Task 2: Lock Adapter Policy Frame Contract

**Files:**
- Modify: `tests/test_grpo_lerobot_adapter.py`

- [ ] **Step 1: Make fake render asymmetric**

In `tests/test_grpo_lerobot_adapter.py`, replace `FakeInner.render()` inside `_install_fake_deferred_deps()` with:

```python
        def render(self):
            return np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
```

- [ ] **Step 2: Add direct deferred-env contract test**

Append to `tests/test_grpo_lerobot_adapter.py`:

```python
def test_deferred_metaworld_reset_observation_pixels_are_lerobot_vh(monkeypatch):
    _install_fake_deferred_deps(monkeypatch)
    from smolvla_grpo.lerobot_metaworld_adapter import DeferredLeRobotMetaworldEnv

    env = DeferredLeRobotMetaworldEnv(task="push-v3", camera_name="corner2")
    try:
        obs, _info = env.reset(seed=123)
        raw = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
        np.testing.assert_array_equal(np.asarray(obs["pixels"]), np.flip(raw, (0, 1)))
    finally:
        env.close()
```

- [ ] **Step 3: Add vector wrapper contract test**

Append to `tests/test_grpo_lerobot_adapter.py`:

```python
def test_expert_oracle_rollout_reset_exposes_lerobot_vh_pixels(monkeypatch):
    _install_fake_deferred_deps(monkeypatch)
    from smolvla_grpo.lerobot_metaworld_adapter import OfficialLeRobotMetaWorldGRPORollout

    rollout = OfficialLeRobotMetaWorldGRPORollout(
        task="push-v3",
        n_envs=1,
        enable_expert_oracle=True,
    )
    try:
        obs = rollout.reset(123)
        raw = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
        policy_frame = np.asarray(obs["pixels"][0])

        np.testing.assert_array_equal(policy_frame, np.flip(raw, (0, 1)))
    finally:
        rollout.close()
```

- [ ] **Step 4: Run adapter tests**

Run:

```bash
cd /vol/bitbucket/aa6622/project
PYTHONPATH=src /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python -m pytest \
  tests/test_grpo_lerobot_adapter.py::test_deferred_metaworld_reset_observation_pixels_are_lerobot_vh \
  tests/test_grpo_lerobot_adapter.py::test_expert_oracle_rollout_reset_exposes_lerobot_vh_pixels \
  -v
```

Expected: PASS after Step 1. This task intentionally does not add a second WM render method.

- [ ] **Step 5: Run adapter suite**

Run:

```bash
cd /vol/bitbucket/aa6622/project
PYTHONPATH=src /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python -m pytest tests/test_grpo_lerobot_adapter.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tests/test_grpo_lerobot_adapter.py
git commit -m "test: lock phase12 adapter policy view"
```

## Task 3: Split Phase12 Rollout Root Frames

**Files:**
- Modify: `scripts/grpo/train_phase12_wm_chunk_grpo.py`
- Modify: `tests/test_phase12_training_loop.py`

- [ ] **Step 1: Add failing root boundary test**

Append to `tests/test_phase12_training_loop.py`:

```python
def test_phase12_root_uses_wm_image_for_scoring_and_policy_obs_for_proc() -> None:
    class EnvHarness:
        inner = SimpleNamespace(single_action_space=SimpleNamespace(shape=(4,)))

        def __init__(self) -> None:
            self.proc_obs = []

        def build_proc(self, obs, *, bundle):
            del bundle
            self.proc_obs.append(obs)
            return {"policy_proc": True}

    policy_frame = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    expected_wm_frame = np.flip(policy_frame, axis=1)
    obs = {"pixels": policy_frame[None], "agent_pos": np.zeros((1, 4), dtype=np.float32)}
    env_h = EnvHarness()

    rollout = trainer._Phase12SelectedRolloutEnv(
        env_h=env_h,
        bundle=object(),
        seed=7,
        initial_obs=obs,
        initial_frame=policy_frame,
        initial_proprio=np.zeros(4, dtype=np.float32),
    )

    root = rollout.reset()

    np.testing.assert_array_equal(root["image"], expected_wm_frame)
    np.testing.assert_array_equal(root["policy_image"], policy_frame)
    assert root["proc"] == {"policy_proc": True}
    assert env_h.proc_obs[0] is obs
```

- [ ] **Step 2: Run test to verify fail**

Run:

```bash
cd /vol/bitbucket/aa6622/project
PYTHONPATH=src /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python -m pytest tests/test_phase12_training_loop.py::test_phase12_root_uses_wm_image_for_scoring_and_policy_obs_for_proc -v
```

Expected: FAIL because `_root()["image"]` still returns the policy frame and `wm_frames` is missing.

- [ ] **Step 3: Update `_Phase12SelectedRolloutEnv.__init__`**

In `scripts/grpo/train_phase12_wm_chunk_grpo.py`, import:

```python
from smolvla_grpo.phase12_pixels import policy_rgb_from_obs, wm_rgb_from_policy_rgb_corner2
```

Then update constructor body:

```python
        self._frame = np.asarray(initial_frame, dtype=np.uint8)
        self._wm_frame = wm_rgb_from_policy_rgb_corner2(self._frame)
        self._proprio = initial_proprio
        self.frames: list[Any] = [self._frame]
        self.wm_frames: list[Any] = [self._wm_frame]
```

- [ ] **Step 4: Update `_root()`**

Replace `_root()` return with:

```python
    def _root(self) -> dict[str, Any]:
        return {
            "id": f"seed{self.seed}_step{len(self.rewards)}",
            "seed": self.seed,
            "obs": self._obs,
            "image": self._wm_frame,
            "policy_image": self._frame,
            "proprio": self._proprio,
            "proc": self.env_h.build_proc(self._obs, bundle=self.bundle),
        }
```

- [ ] **Step 5: Update `step()`**

In `_Phase12SelectedRolloutEnv.step()`, set both streams:

```python
        self._obs = step.observation
        self._frame = policy_rgb_from_obs(self._obs)
        self._wm_frame = wm_rgb_from_policy_rgb_corner2(self._frame)
        self._proprio = np.asarray(self.env_h.last_agent_pos(), dtype=np.float32)
        self.frames.append(self._frame)
        self.wm_frames.append(self._wm_frame)
```

- [ ] **Step 6: Run root test**

Run:

```bash
cd /vol/bitbucket/aa6622/project
PYTHONPATH=src /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python -m pytest tests/test_phase12_training_loop.py::test_phase12_root_uses_wm_image_for_scoring_and_policy_obs_for_proc -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add scripts/grpo/train_phase12_wm_chunk_grpo.py tests/test_phase12_training_loop.py
git commit -m "fix: split phase12 rollout frame streams"
```

## Task 4: Record Oracle WM Frames And Encode Goals From Them

**Files:**
- Modify: `scripts/grpo/train_phase12_wm_chunk_grpo.py`
- Modify: `tests/test_phase12_training_loop.py`

- [ ] **Step 1: Add failing oracle test**

Append to `tests/test_phase12_training_loop.py`:

```python
def test_phase12_oracle_records_policy_and_wm_frames(monkeypatch, tmp_path) -> None:
    reset_policy = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    step_policy = reset_policy + 20

    class Step:
        observation = {"pixels": step_policy[None]}
        reward = 1.0
        success = True
        terminated = True
        truncated = False
        info = {"success": True}

    class EnvHarness:
        def reset(self, seed):
            assert seed == 123
            return {"pixels": reset_policy[None]}

        def last_agent_pos(self):
            return np.arange(4, dtype=np.float32)

        def last_raw_obs(self):
            return np.arange(5, dtype=np.float64)

        def expert_action(self):
            return np.zeros(4, dtype=np.float32)

        def step(self, action):
            np.testing.assert_array_equal(action, np.zeros((1, 4), dtype=np.float32))
            return Step()

    monkeypatch.setattr(trainer, "write_phase12_episode_video", lambda **kwargs: kwargs["video_path"])

    oracle = trainer._rollout_phase12_oracle(
        env_h=EnvHarness(),
        seed=123,
        max_steps=1,
        output_dir=tmp_path,
        fps=6,
    )

    np.testing.assert_array_equal(oracle["frames"][0], reset_policy)
    np.testing.assert_array_equal(oracle["wm_frames"][0], np.flip(reset_policy, 1))
    np.testing.assert_array_equal(oracle["frames"][1], step_policy)
    np.testing.assert_array_equal(oracle["wm_frames"][1], np.flip(step_policy, 1))
    assert len(oracle["wm_frames"]) == len(oracle["frames"])
```

- [ ] **Step 2: Run test to verify fail**

Run:

```bash
cd /vol/bitbucket/aa6622/project
PYTHONPATH=src /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python -m pytest tests/test_phase12_training_loop.py::test_phase12_oracle_records_policy_and_wm_frames -v
```

Expected: FAIL with `KeyError: 'wm_frames'`.

- [ ] **Step 3: Record `wm_frames` in `_rollout_phase12_oracle()`**

Use the module-level helpers imported in Task 3. Do not import inside the loop.

```python
    # policy_rgb_from_obs and wm_rgb_from_policy_rgb_corner2 are imported at module scope.
```

Use reset observation pixels as the single source of truth:

```python
    obs = env_h.reset(int(seed))
    policy_frame = policy_rgb_from_obs(obs)
    frames: list[np.ndarray] = [policy_frame]
    wm_frames: list[np.ndarray] = [wm_rgb_from_policy_rgb_corner2(policy_frame)]
```

Inside step loop, append from `step.observation`:

```python
        policy_frame = policy_rgb_from_obs(step.observation)
        frames.append(policy_frame)
        wm_frames.append(wm_rgb_from_policy_rgb_corner2(policy_frame))
```

Return it:

```python
        "wm_frames": wm_frames,
```

- [ ] **Step 4: Encode goals from `oracle["wm_frames"]`**

In `collect_phase12_training_episode()`, change `_encode_structured()` image arg:

```python
                oracle["wm_frames"][frame_idx - 1],
```

Do not change saved `frame_path`; human PNGs stay policy/video view.

- [ ] **Step 5: Run tests**

Run:

```bash
cd /vol/bitbucket/aa6622/project
PYTHONPATH=src /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python -m pytest tests/test_phase12_training_loop.py::test_phase12_oracle_records_policy_and_wm_frames tests/test_phase12_wm_reward.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/grpo/train_phase12_wm_chunk_grpo.py tests/test_phase12_training_loop.py
git commit -m "fix: encode phase12 goals from wm frames"
```

## Task 5: Wire Initial WM Frame And Decode Against WM Real Frames

**Files:**
- Modify: `scripts/grpo/train_phase12_wm_chunk_grpo.py`
- Modify: `tests/test_phase12_training_loop.py`

- [ ] **Step 1: Add failing decode helper test**

Append to `tests/test_phase12_training_loop.py`:

```python
def test_phase12_selected_decode_uses_wm_frames_as_real_frames(monkeypatch, tmp_path) -> None:
    seen: dict[str, object] = {}
    policy_frame = np.full((2, 2, 3), 11, dtype=np.uint8)
    wm_frame = np.full((2, 2, 3), 22, dtype=np.uint8)

    def fake_build_decode_artifacts(**kwargs):
        seen["real_frames"] = kwargs["real_frames"]
        return SimpleNamespace(paths={}, metadata={"decode_status": "ok"})

    monkeypatch.setattr(trainer, "build_decode_artifacts", fake_build_decode_artifacts, raising=False)

    episode = SimpleNamespace(segments=[SimpleNamespace(selected_candidate_index=0)])
    rollout_env = SimpleNamespace(frames=[policy_frame], wm_frames=[wm_frame])
    score_inputs = {
        (0, 0): {
            "image": wm_frame,
            "proprio": np.zeros(4, dtype=np.float32),
            "actions": np.zeros((5, 4), dtype=np.float32),
        }
    }
    meta: dict[str, object] = {}

    trainer._build_phase12_selected_decode_artifacts(
        args=SimpleNamespace(save_wm_decodes=True, strict_decode=True, goal_latent_mode="visual_proprio", chunk_len=5),
        episode=episode,
        episode_dir=tmp_path,
        rollout_env=rollout_env,
        score_inputs=score_inputs,
        wm_bundle=SimpleNamespace(planner_action_dim=20),
        action_dim=4,
        meta=meta,
    )

    np.testing.assert_array_equal(seen["real_frames"][0], wm_frame)
    assert meta["wm_decode_real_frame_source"] == "wm_frames"
```

- [ ] **Step 2: Run test to verify fail**

Run:

```bash
cd /vol/bitbucket/aa6622/project
PYTHONPATH=src /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python -m pytest tests/test_phase12_training_loop.py::test_phase12_selected_decode_uses_wm_frames_as_real_frames -v
```

Expected: FAIL with `AttributeError: module 'scripts.grpo.train_phase12_wm_chunk_grpo' has no attribute '_build_phase12_selected_decode_artifacts'`.

- [ ] **Step 3: Add decode helper**

Add to `scripts/grpo/train_phase12_wm_chunk_grpo.py` near `_merge_phase12_decode_metadata()`:

```python
def _build_phase12_selected_decode_artifacts(
    *,
    args: argparse.Namespace,
    episode: Any,
    episode_dir: Path,
    rollout_env: Any,
    score_inputs: dict[tuple[int, int], dict[str, Any]],
    wm_bundle: Any,
    action_dim: int,
    meta: dict[str, Any],
) -> None:
    if not bool(args.save_wm_decodes) or not getattr(episode, "segments", None):
        meta.setdefault("wm_decode_status", "disabled")
        return
    first_segment = episode.segments[0]
    key = (0, int(first_segment.selected_candidate_index))
    decode_input = score_inputs.get(key)
    if decode_input is None:
        if bool(args.strict_decode):
            raise RuntimeError("strict Phase12 decode requested but selected decode input was not recorded")
        meta.setdefault("wm_decode_status", "missing_input")
        return
    real_frames = list(getattr(rollout_env, "wm_frames", rollout_env.frames))
    decode_result = build_decode_artifacts(
        decode_fn=lambda: _decode_phase12_prediction_frames(
            wm_bundle,
            image=decode_input["image"],
            proprio=decode_input["proprio"],
            actions=decode_input["actions"],
            mode=args.goal_latent_mode,
        ),
        output_dir=episode_dir,
        real_frames=real_frames,
        strict_decode=bool(args.strict_decode),
        segment_index=0,
        selected_candidate_index=int(first_segment.selected_candidate_index),
        env_steps_per_wm_step=max(1, int(wm_bundle.planner_action_dim) // max(1, action_dim)),
        carried_steps=min(int(args.chunk_len), max(0, len(real_frames) - 1)),
    )
    _merge_phase12_decode_metadata(meta, decode_result.metadata)
    meta["wm_decode_real_frame_source"] = "wm_frames"
```

- [ ] **Step 4: Replace inline decode block**

In `collect_phase12_training_episode()`, replace existing selected decode block with:

```python
        _build_phase12_selected_decode_artifacts(
            args=args,
            episode=episode,
            episode_dir=episode_dir,
            rollout_env=rollout_env,
            score_inputs=score_inputs,
            wm_bundle=wm_bundle,
            action_dim=action_dim,
            meta=meta,
        )
```

- [ ] **Step 5: Wire initial WM frame**

After reset in `collect_phase12_training_episode()`, use `reset_obs` pixels as source-of-truth:

```python
        reset_frame = policy_rgb_from_obs(reset_obs)
```

Update `_Phase12SelectedRolloutEnv(...)` call:

```python
            initial_frame=reset_frame,
            initial_proprio=reset_proprio,
```

`_Phase12SelectedRolloutEnv` derives its own `_wm_frame` from `initial_frame` to keep the invariant local. Do not pass a separately rendered or separately computed WM frame.

- [ ] **Step 6: Run tests**

Run:

```bash
cd /vol/bitbucket/aa6622/project
PYTHONPATH=src /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python -m pytest \
  tests/test_phase12_training_loop.py::test_phase12_root_uses_wm_image_for_scoring_and_policy_obs_for_proc \
  tests/test_phase12_training_loop.py::test_phase12_selected_decode_uses_wm_frames_as_real_frames \
  tests/test_phase12_diagnostics.py \
  -v
```

Expected: PASS.

- [ ] **Step 7: Add stale render guard tests**

Append to `tests/test_phase12_trainer_static.py`:

```python
def test_phase12_oracle_and_selected_rollout_do_not_use_render_frame_for_pixels() -> None:
    source = (trainer._REPO / "scripts" / "grpo" / "train_phase12_wm_chunk_grpo.py").read_text(
        encoding="utf-8"
    )

    assert "env_h.render_frame()" not in source
    assert "self.env_h.render_frame()" not in source
    assert "policy_rgb_from_obs(reset_obs)" in source
    assert "policy_rgb_from_obs(step.observation)" in source
```

Run:

```bash
cd /vol/bitbucket/aa6622/project
PYTHONPATH=src /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python -m pytest tests/test_phase12_trainer_static.py::test_phase12_oracle_and_selected_rollout_do_not_use_render_frame_for_pixels -v
```

Expected: PASS after inline render calls are removed.

- [ ] **Step 8: Commit**

```bash
git add scripts/grpo/train_phase12_wm_chunk_grpo.py tests/test_phase12_training_loop.py
git commit -m "fix: decode phase12 using wm real frames"
```

## Task 6: Add Pixel Contract Metadata

**Files:**
- Modify: `scripts/grpo/train_phase12_wm_chunk_grpo.py`
- Modify: `tests/test_phase12_trainer_static.py`

- [ ] **Step 1: Add failing manifest test**

Append to `tests/test_phase12_trainer_static.py`:

```python
def test_phase12_manifest_records_pixel_contracts(tmp_path) -> None:
    args = parse_args(["--output-dir", str(tmp_path), "--dry-run"])

    manifest = build_manifest(args)

    assert manifest["phase12_policy_frame_contract"] == "lerobot_corner2_vhflip"
    assert manifest["phase12_wm_frame_contract"] == "jepa_corner2_vflip"
    assert manifest["phase12_goal_frame_contract"] == "jepa_corner2_vflip"
    assert manifest["phase12_decode_real_frame_source"] == "wm_frames"
```

- [ ] **Step 2: Run test to verify fail**

Run:

```bash
cd /vol/bitbucket/aa6622/project
PYTHONPATH=src /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python -m pytest tests/test_phase12_trainer_static.py::test_phase12_manifest_records_pixel_contracts -v
```

Expected: FAIL with `KeyError: 'phase12_policy_frame_contract'`.

- [ ] **Step 3: Add manifest fields**

In `build_manifest()`, add:

```python
        "phase12_policy_frame_contract": "lerobot_corner2_vhflip",
        "phase12_wm_frame_contract": "jepa_corner2_vflip",
        "phase12_goal_frame_contract": "jepa_corner2_vflip",
        "phase12_decode_real_frame_source": "wm_frames",
```

- [ ] **Step 4: Add episode metadata fields**

In `collect_phase12_training_episode()` meta update, add same fields:

```python
                "phase12_policy_frame_contract": "lerobot_corner2_vhflip",
                "phase12_wm_frame_contract": "jepa_corner2_vflip",
                "phase12_goal_frame_contract": "jepa_corner2_vflip",
                "phase12_decode_real_frame_source": "wm_frames",
```

- [ ] **Step 5: Run static tests**

Run:

```bash
cd /vol/bitbucket/aa6622/project
PYTHONPATH=src /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python -m pytest tests/test_phase12_trainer_static.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/grpo/train_phase12_wm_chunk_grpo.py tests/test_phase12_trainer_static.py
git commit -m "fix: record phase12 wm view contract"
```

## Task 7: Post-Code CPU Regression Suite

**Files:**
- No code edits expected.

- [ ] **Step 1: Run focused CPU tests**

Run:

```bash
cd /vol/bitbucket/aa6622/project
PYTHONPATH=src /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python -m pytest \
  tests/test_phase12_pixels.py \
  tests/test_grpo_lerobot_adapter.py \
  tests/test_phase12_training_loop.py \
  tests/test_phase12_wm_reward.py \
  tests/test_phase12_diagnostics.py \
  tests/test_phase12_trainer_static.py \
  tests/test_phase12_artifacts.py \
  -q
```

Expected: PASS.

- [ ] **Step 2: Run Phase08/WM parity tests**

Run:

```bash
cd /vol/bitbucket/aa6622/project
PYTHONPATH=src /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python -m pytest \
  tests/test_run_segment_grpo_main.py \
  tests/test_segment_grpo_loop.py \
  tests/test_metaworld_jepa_render.py \
  -q
```

Expected: PASS. `test_metaworld_jepa_render.py` may skip if MetaWorld unavailable; in project env it should run.

- [ ] **Step 3: Commit any test-only mock fixes**

Only if tests expose stale mocks:

```bash
git add tests
git commit -m "test: update phase12 wm view mocks"
```

If behavior fails, stop and return to task that introduced it.

## Task 8: Bounded One-Update Strict Smoke And Push

**Files:**
- No code edits expected.

- [ ] **Step 1: Run bounded one-update smoke**

Run:

```bash
cd /vol/bitbucket/aa6622/project
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 JEPA_WM_DISABLE_IMAGE_HEAD=0 \
/vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python scripts/grpo/train_phase12_wm_chunk_grpo.py \
  --mode wm_grpo_train \
  --checkpoint /vol/bitbucket/aa6622/.cache/huggingface/hub/models--jadechoghari--smolvla_metaworld/snapshots/ef3089ecb84eeeb7d33fedab24f6c76180a68900 \
  --jepa-repo "/vol/bitbucket/aa6622/VGG JEPA/jepa-wms" \
  --jepa-ckpt jepa_wm_metaworld.pth.tar \
  --output-dir artifacts/phase12_wm_chunk_grpo_train/push-v3/wm_view_fix_bounded_u1_seed2000 \
  --action-profile bounded_executed \
  --num-episodes 1 \
  --num-updates 1 \
  --train-seed-base 2000 \
  --chunk-len 25 \
  --group-size 4 \
  --max-steps 120 \
  --strict-decode
```

Use offline env only if all weights are local.

Expected:

```text
PHASE12_WM_CHUNK_GRPO_TRAIN_DONE updates=1 out=...
```

- [ ] **Step 2: Verify artifact files**

Run:

```bash
python - <<'PY'
from pathlib import Path
import json

root = Path("/vol/bitbucket/aa6622/project/artifacts/phase12_wm_chunk_grpo_train/push-v3/wm_view_fix_bounded_u1_seed2000")
smoke = json.loads((root / "smoke_manifest.json").read_text())
for key in [
    "selected_action_rollout_video",
    "oracle_baseline_video",
    "wm_decode_selected_strip_path",
    "wm_real_vs_pred_selected_strip_path",
]:
    path = Path(smoke[key])
    if not path.is_file() or path.stat().st_size <= 0:
        raise SystemExit(f"bad artifact {key}: {path}")
print("artifacts ok")
PY
```

Expected:

```text
artifacts ok
```

- [ ] **Step 3: Verify metadata**

Run:

```bash
python - <<'PY'
from pathlib import Path
import json

root = Path("/vol/bitbucket/aa6622/project/artifacts/phase12_wm_chunk_grpo_train/push-v3/wm_view_fix_bounded_u1_seed2000")
manifest = json.loads((root / "train_manifest.json").read_text())
expected = {
    "phase12_policy_frame_contract": "lerobot_corner2_vhflip",
    "phase12_wm_frame_contract": "jepa_corner2_vflip",
    "phase12_goal_frame_contract": "jepa_corner2_vflip",
    "phase12_decode_real_frame_source": "wm_frames",
}
for key, value in expected.items():
    if manifest.get(key) != value:
        raise SystemExit(f"{key}: expected {value}, got {manifest.get(key)}")
print("metadata ok")
PY
```

Expected:

```text
metadata ok
```

- [ ] **Step 4: Inspect bounded artifacts**

Run:

```bash
python - <<'PY'
from pathlib import Path
import numpy as np
from PIL import Image

def edge_var(path: Path) -> tuple[float, float, float]:
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    h = arr.shape[0] // 2
    top = arr[:h]
    bottom = arr[h:]
    def score(x):
        gx = np.diff(x, axis=1)
        gy = np.diff(x, axis=0)
        return float(np.var(gx) + np.var(gy))
    real = score(top)
    pred = score(bottom)
    return real, pred, pred / max(real, 1e-6)

old = Path("/vol/bitbucket/aa6622/project/artifacts/phase12_wm_chunk_grpo_train/push-v3/debug_decode_postfix_bounded_u1_seed2000/rollouts/update_0000_episode_0000/segment_0000/wm_real_vs_pred_selected_strip.png")
new = Path("/vol/bitbucket/aa6622/project/artifacts/phase12_wm_chunk_grpo_train/push-v3/wm_view_fix_bounded_u1_seed2000/rollouts/update_0000_episode_0000/segment_0000/wm_real_vs_pred_selected_strip.png")
for label, path in [("old", old), ("new", new)]:
    real, pred, ratio = edge_var(path)
    print(label, "real", round(real, 2), "pred", round(pred, 2), "ratio", round(ratio, 3))
PY
```

Expected:

- `selected_action_rollout.mp4` is LeRobot V+H.
- `wm_real_vs_pred_selected_strip.png` top row is JEPA V-only.
- selected video and strip top row are horizontal mirrors, not identical.
- decode no longer looks "wack" or shows massive unphysical movement.
- debug log still shows `25x4 -> 5x20`.
- new blur ratio is higher than old and visual strip is no longer horizontally wrong/smeared.

If bounded still looks wrong, stop. Next diagnosis: compare `wm_rgb_from_policy_rgb_corner2(policy_rgb_from_obs(obs))` against Phase08 `render_jepa_rgb()` for the same seed/state.

- [ ] **Step 5: Push all commits**

Only after focused CPU tests, parity tests, and bounded artifact inspection pass:

```bash
cd /vol/bitbucket/aa6622/project
git push
```

## Task 9: Official Strict Smoke Then Bounded 100-Update

**Files:**
- No code edits expected.

- [ ] **Step 1: Run official one-update smoke**

Run:

```bash
cd /vol/bitbucket/aa6622/project
sbatch scripts/grpo/submit_phase12_wm_chunk_grpo_train.slurm official_jepa_mirror 1 artifacts/phase12_wm_chunk_grpo_train/push-v3/wm_view_fix_official_u1_seed2000
```

Expected: run only after bounded one-update looks good. Slurm job succeeds and writes `smoke_manifest.json`.

- [ ] **Step 2: Submit bounded 100-update run after bounded one-update passes**

Run:

```bash
cd /vol/bitbucket/aa6622/project
sbatch scripts/grpo/submit_phase12_wm_chunk_grpo_train.slurm bounded_executed 100 artifacts/phase12_wm_chunk_grpo_train/push-v3/wm_view_fix_bounded_u100_seed2000
```

Expected: launch automatically for bounded once bounded one-update inspection passes. If bounded still looks wrong, do not launch 100-update; run the Phase08 `render_jepa_rgb()` comparison diagnosis instead.

## Self-Review

- Spec coverage: fixes root pixel contract, preserves SmolVLA policy view, fixes goal encode, fixes decode real frame source, adds smoke gate.
- Placeholder scan: no `TBD`, no "add appropriate tests", no unspecified commands.
- Type consistency: `policy_rgb_from_obs()` extracts one captured policy frame; `wm_rgb_from_policy_rgb_corner2()` derives the WM view; `_Phase12SelectedRolloutEnv` stores `wm_frames`; root `"image"` is WM frame; policy proc still uses original observation.
- Risk handled: if LeRobot `obs["pixels"]` and `render_frame()` differ for the same reset, selected video and proc/debug views no longer diverge because Phase12 stored frames use `obs["pixels"][0]` only. Static guard rejects new `render_frame()` pixel plumbing in oracle/selected rollout.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-15-phase12-wm-view-final-plan.md`. Two execution options:

1. **Subagent-Driven (recommended)** - dispatch fresh subagent per task, review between tasks, fast iteration
2. **Inline Execution** - execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
