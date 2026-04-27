# Phase12 WM Pixel Contract Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix Phase12 train so JEPA-WM score/decode consumes the same pixel contract as Phase08: `corner2` camera, same cam patch, vertical-flip-only WM frames, while SmolVLA policy keeps LeRobot vertical+horizontal-flip frames.

**Architecture:** Split Phase12 pixels into two explicit streams at the environment boundary: `policy_image` for SmolVLA/video/oracle display and `wm_image` for JEPA-WM encode/score/decode/real-vs-pred strips. Reuse same underlying MetaWorld state/env; avoid a second env or replay shim. Add pure pixel helpers plus adapter/root-observation tests so future code cannot feed policy pixels into WM silently.

**Tech Stack:** Python 3.12, NumPy, PyTorch, MetaWorld, LeRobot SmolVLA, JEPA-WM, `pytest`, existing Phase12 trainer/adapter modules.

---

## Root Cause

Phase08 working path:

- WM image: `render_jepa_rgb(env)` = raw `corner2` render, cam patch, vertical flip only.
- Policy image: horizontal flip of WM image so SmolVLA sees LeRobot vertical+horizontal-flip view.
- Oracle goal image: stored oracle PNG is vertical+horizontal-flip, then horizontal-flipped before WM encode.

Phase12 train broken path:

- `DeferredLeRobotMetaworldEnv.render()` returns vertical+horizontal-flip for `corner2`.
- `_Phase12SelectedRolloutEnv._root()` stores that as `"image"`.
- `score_phase12_chunk_with_wm()` and `_decode_phase12_prediction_frames()` pass `"image"` into `_to_wm_visual()`.
- `_rollout_phase12_oracle()` encodes goal frames from same vertical+horizontal-flip stream.

Fix: keep vertical+horizontal-flip stream for policy/video, add vertical-only stream for WM.

## File Structure

- Create: `src/smolvla_grpo/phase12_pixels.py`
  - Pure uint8 RGB conversion and Phase12 camera-contract helpers.
  - No MetaWorld, torch, or LeRobot imports.
- Create: `tests/test_phase12_pixels.py`
  - Fast pure-unit tests for raw -> policy, raw -> WM, policy -> WM, and oracle goal -> WM transforms.
- Modify: `src/smolvla_grpo/lerobot_metaworld_adapter.py`
  - Refactor `DeferredLeRobotMetaworldEnv.render()` through raw render helper.
  - Add `render_frame_for_wm()` on deferred env and `OfficialLeRobotMetaWorldGRPORollout`.
- Modify: `tests/test_grpo_lerobot_adapter.py`
  - Extend fake render to non-symmetric pixels.
  - Test policy frame and WM frame are different and WM equals horizontal flip of policy for `corner2`.
- Modify: `scripts/grpo/train_phase12_wm_chunk_grpo.py`
  - Carry `initial_wm_frame` into `_Phase12SelectedRolloutEnv`.
  - Store `wm_frames` separately from policy/video frames.
  - Put WM frame in `root_observation["image"]`.
  - Decode real-vs-pred strips against `rollout_env.wm_frames`.
  - Encode oracle goals from `oracle["wm_frames"]`.
  - Add manifest/debug metadata for pixel contract.
- Modify: `tests/test_phase12_training_loop.py`
  - Test selected rollout root uses WM image for scoring but policy obs for SmolVLA proc.
  - Test decode artifacts receive WM real frames, not policy frames.
- Modify: `tests/test_phase12_trainer_static.py`
  - Lock manifest metadata so train runs record pixel contracts.
- Optional after main fix: `scripts/grpo/submit_phase12_wm_chunk_grpo_train.slurm`
  - No behavior change required. Only update echoed metadata if useful.

## Non-Goals

- Do not change action profile semantics in this plan.
- Do not switch Phase12 objective from visual+proprio to visual-only.
- Do not replace real Phase12 train loop with Phase08 `rollout_with_chunks`.
- Do not add a second MetaWorld env for WM. Same-state guarantee matters more.
- Do not rebuild renderer to 224 in first fix. `_to_wm_visual()` already normalizes size to `256x256`; orientation mismatch is primary root cause. Add 224 renderer only if post-fix smoke still shows blur.

## Task 1: Pure Phase12 Pixel Helpers

**Files:**
- Create: `src/smolvla_grpo/phase12_pixels.py`
- Create: `tests/test_phase12_pixels.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_phase12_pixels.py`:

```python
from __future__ import annotations

import numpy as np

from smolvla_grpo.phase12_pixels import (
    goal_rgb_for_wm_from_policy_rgb,
    policy_rgb_from_raw_corner2,
    to_rgb_uint8,
    wm_rgb_from_policy_rgb_corner2,
    wm_rgb_from_raw_corner2,
)


def _raw_frame() -> np.ndarray:
    return np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)


def test_policy_rgb_from_raw_corner2_matches_lerobot_vhflip() -> None:
    raw = _raw_frame()

    policy = policy_rgb_from_raw_corner2(raw)

    np.testing.assert_array_equal(policy, np.flip(raw, (0, 1)))


def test_wm_rgb_from_raw_corner2_matches_jepa_vflip_only() -> None:
    raw = _raw_frame()

    wm = wm_rgb_from_raw_corner2(raw)

    np.testing.assert_array_equal(wm, np.flip(raw, 0))


def test_wm_rgb_from_policy_rgb_corner2_removes_only_hflip() -> None:
    raw = _raw_frame()
    policy = policy_rgb_from_raw_corner2(raw)

    wm = wm_rgb_from_policy_rgb_corner2(policy)

    np.testing.assert_array_equal(wm, wm_rgb_from_raw_corner2(raw))


def test_goal_rgb_for_wm_from_policy_rgb_matches_phase08_goal_hflip() -> None:
    raw = _raw_frame()
    stored_oracle_policy = policy_rgb_from_raw_corner2(raw)

    wm_goal = goal_rgb_for_wm_from_policy_rgb(stored_oracle_policy)

    np.testing.assert_array_equal(wm_goal, wm_rgb_from_raw_corner2(raw))


def test_to_rgb_uint8_accepts_float_unit_rgb_and_drops_alpha() -> None:
    rgba = np.zeros((2, 2, 4), dtype=np.float32)
    rgba[..., 0] = 1.0
    rgba[..., 3] = 0.5

    rgb = to_rgb_uint8(rgba)

    assert rgb.dtype == np.uint8
    assert rgb.shape == (2, 2, 3)
    assert int(rgb[..., 0].max()) == 255
```

- [ ] **Step 2: Run tests to verify fail**

Run:

```bash
PYTHONPATH=src /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python -m pytest tests/test_phase12_pixels.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'smolvla_grpo.phase12_pixels'`.

- [ ] **Step 3: Add minimal implementation**

Create `src/smolvla_grpo/phase12_pixels.py`:

```python
"""Phase12 pixel-contract helpers.

Phase12 carries two RGB streams:
- policy RGB: LeRobot/SmolVLA corner2 contract, vertical+horizontal flip.
- WM RGB: JEPA-WM MetaWorld contract, vertical flip only.
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
    """LeRobot/SmolVLA corner2 policy contract: vertical+horizontal flip."""

    return np.ascontiguousarray(np.flip(to_rgb_uint8(raw_rgb), (0, 1)))


def wm_rgb_from_raw_corner2(raw_rgb: Any) -> np.ndarray:
    """JEPA-WM corner2 contract: vertical flip only."""

    return np.ascontiguousarray(np.flip(to_rgb_uint8(raw_rgb), 0))


def wm_rgb_from_policy_rgb_corner2(policy_rgb: Any) -> np.ndarray:
    """Convert stored LeRobot policy RGB (V+H) to JEPA-WM RGB (V-only)."""

    return np.ascontiguousarray(np.flip(to_rgb_uint8(policy_rgb), 1))


def goal_rgb_for_wm_from_policy_rgb(policy_rgb: Any) -> np.ndarray:
    """Phase12 oracle frames are saved in policy RGB; WM goal encode needs JEPA RGB."""

    return wm_rgb_from_policy_rgb_corner2(policy_rgb)
```

- [ ] **Step 4: Run tests to verify pass**

Run:

```bash
PYTHONPATH=src /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python -m pytest tests/test_phase12_pixels.py -v
```

Expected: PASS, 5 tests.

- [ ] **Step 5: Commit**

```bash
git add src/smolvla_grpo/phase12_pixels.py tests/test_phase12_pixels.py
git commit -m "fix: add phase12 wm pixel helpers"
```

## Task 2: Expose WM Render From LeRobot Adapter

**Files:**
- Modify: `src/smolvla_grpo/lerobot_metaworld_adapter.py`
- Modify: `tests/test_grpo_lerobot_adapter.py`

- [ ] **Step 1: Update fake render to be asymmetric**

In `tests/test_grpo_lerobot_adapter.py`, replace `FakeInner.render()` inside `_install_fake_deferred_deps()` with:

```python
        def render(self):
            return np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
```

- [ ] **Step 2: Add failing adapter test**

Append to `tests/test_grpo_lerobot_adapter.py`:

```python
def test_expert_oracle_adapter_exposes_distinct_policy_and_wm_frames(monkeypatch):
    _install_fake_deferred_deps(monkeypatch)
    from smolvla_grpo.lerobot_metaworld_adapter import OfficialLeRobotMetaWorldGRPORollout

    rollout = OfficialLeRobotMetaWorldGRPORollout(
        task="push-v3",
        n_envs=1,
        enable_expert_oracle=True,
    )
    try:
        obs = rollout.reset(123)
        policy_frame = np.asarray(obs["pixels"][0])
        wm_frame = rollout.render_frame_for_wm()
        raw = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)

        np.testing.assert_array_equal(policy_frame, np.flip(raw, (0, 1)))
        np.testing.assert_array_equal(wm_frame, np.flip(raw, 0))
        np.testing.assert_array_equal(wm_frame, np.flip(policy_frame, 1))
    finally:
        rollout.close()
```

- [ ] **Step 3: Run test to verify fail**

Run:

```bash
PYTHONPATH=src /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python -m pytest tests/test_grpo_lerobot_adapter.py::test_expert_oracle_adapter_exposes_distinct_policy_and_wm_frames -v
```

Expected: FAIL with `AttributeError: 'OfficialLeRobotMetaWorldGRPORollout' object has no attribute 'render_frame_for_wm'`.

- [ ] **Step 4: Refactor adapter render**

In `src/smolvla_grpo/lerobot_metaworld_adapter.py`, add import:

```python
from smolvla_grpo.phase12_pixels import policy_rgb_from_raw_corner2, to_rgb_uint8, wm_rgb_from_raw_corner2
```

Replace `DeferredLeRobotMetaworldEnv.render()` with:

```python
    def _render_raw_frame(self) -> np.ndarray:
        self._ensure_env()
        raw_image = np.asarray(self._env.render())
        return to_rgb_uint8(raw_image)

    def render(self) -> np.ndarray:
        raw_image = self._render_raw_frame()
        if self.camera_name == "corner2":
            image = policy_rgb_from_raw_corner2(raw_image)
            vflip_image = wm_rgb_from_raw_corner2(raw_image)
        else:
            image = raw_image
            vflip_image = raw_image
        # region agent log
        _agent_debug_log(
            hypothesis_id="H3",
            location="src/smolvla_grpo/lerobot_metaworld_adapter.py:DeferredLeRobotMetaworldEnv.render",
            message="adapter render orientation candidates before Phase12 WM encode",
            data={
                "camera_name": self.camera_name,
                "returned_contract": "vhflip_for_corner2" if self.camera_name == "corner2" else "raw",
                "jepa_metaworld_expected_contract": "vflip_for_corner2",
                "raw": _image_debug(raw_image),
                "vflip": _image_debug(vflip_image),
                "returned": _image_debug(image),
            },
        )
        # endregion
        return image

    def render_frame_for_wm(self) -> np.ndarray:
        raw_image = self._render_raw_frame()
        if self.camera_name == "corner2":
            return wm_rgb_from_raw_corner2(raw_image)
        return raw_image
```

Add method to `OfficialLeRobotMetaWorldGRPORollout`:

```python
    def render_frame_for_wm(self) -> np.ndarray:
        return np.asarray(self.vec_env.call("render_frame_for_wm")[0])
```

- [ ] **Step 5: Run adapter tests**

Run:

```bash
PYTHONPATH=src /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python -m pytest tests/test_grpo_lerobot_adapter.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/smolvla_grpo/lerobot_metaworld_adapter.py tests/test_grpo_lerobot_adapter.py
git commit -m "fix: expose phase12 wm render stream"
```

## Task 3: Carry Separate WM Frames Through Phase12 Rollout Env

**Files:**
- Modify: `scripts/grpo/train_phase12_wm_chunk_grpo.py`
- Modify: `tests/test_phase12_training_loop.py`

- [ ] **Step 1: Write failing root-observation test**

Append to `tests/test_phase12_training_loop.py`:

```python
def test_phase12_selected_rollout_root_uses_wm_image_for_scoring_but_policy_obs_for_proc() -> None:
    class EnvHarness:
        def __init__(self) -> None:
            self.proc_pixels: list[np.ndarray] = []

        def build_proc(self, obs, *, bundle):
            del bundle
            self.proc_pixels.append(np.asarray(obs["pixels"], dtype=np.uint8).copy())
            return {"proc": "policy"}

    policy_frame = np.full((2, 2, 3), 11, dtype=np.uint8)
    wm_frame = np.full((2, 2, 3), 22, dtype=np.uint8)
    obs = {"pixels": policy_frame.copy(), "agent_pos": np.zeros(4, dtype=np.float32)}
    env = trainer._Phase12SelectedRolloutEnv(
        env_h=EnvHarness(),
        bundle=object(),
        seed=7,
        initial_obs=obs,
        initial_frame=policy_frame,
        initial_wm_frame=wm_frame,
        initial_proprio=np.zeros(4, dtype=np.float32),
    )

    root = env.reset()

    np.testing.assert_array_equal(root["image"], wm_frame)
    np.testing.assert_array_equal(root["policy_image"], policy_frame)
    np.testing.assert_array_equal(env.env_h.proc_pixels[0], policy_frame)
```

- [ ] **Step 2: Run test to verify fail**

Run:

```bash
PYTHONPATH=src /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python -m pytest tests/test_phase12_training_loop.py::test_phase12_selected_rollout_root_uses_wm_image_for_scoring_but_policy_obs_for_proc -v
```

Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'initial_wm_frame'`.

- [ ] **Step 3: Update `_Phase12SelectedRolloutEnv` constructor and root**

In `scripts/grpo/train_phase12_wm_chunk_grpo.py`, replace `_Phase12SelectedRolloutEnv.__init__` with:

```python
    def __init__(
        self,
        *,
        env_h: Any,
        bundle: Any,
        seed: int,
        initial_obs: dict[str, Any],
        initial_frame: Any,
        initial_wm_frame: Any,
        initial_proprio: Any,
    ) -> None:
        self.env_h = env_h
        self.bundle = bundle
        self.seed = int(seed)
        self._obs = initial_obs
        self._frame = np.asarray(initial_frame, dtype=np.uint8)
        self._wm_frame = np.asarray(initial_wm_frame, dtype=np.uint8)
        self._proprio = initial_proprio
        self.frames: list[Any] = [self._frame]
        self.wm_frames: list[Any] = [self._wm_frame]
        self.rewards: list[float] = []
        self.successes: list[bool] = []
        self.action_space = getattr(env_h.inner, "single_action_space", None)
```

Replace `_root()` with:

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

- [ ] **Step 4: Update `step()` to refresh both streams**

In `_Phase12SelectedRolloutEnv.step()`, replace frame updates with:

```python
        self._obs = step.observation
        self._frame = np.asarray(self.env_h.render_frame(), dtype=np.uint8)
        self._wm_frame = np.asarray(self.env_h.render_frame_for_wm(), dtype=np.uint8)
        self._proprio = np.asarray(self.env_h.last_agent_pos(), dtype=np.float32)
        self.frames.append(self._frame)
        self.wm_frames.append(self._wm_frame)
```

- [ ] **Step 5: Run targeted test**

Run:

```bash
PYTHONPATH=src /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python -m pytest tests/test_phase12_training_loop.py::test_phase12_selected_rollout_root_uses_wm_image_for_scoring_but_policy_obs_for_proc -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/grpo/train_phase12_wm_chunk_grpo.py tests/test_phase12_training_loop.py
git commit -m "fix: split phase12 policy and wm root frames"
```

## Task 4: Use WM Frames For Oracle Goal Encode

**Files:**
- Modify: `scripts/grpo/train_phase12_wm_chunk_grpo.py`
- Modify: `tests/test_phase12_training_loop.py`

- [ ] **Step 1: Write failing oracle-rollout test**

Append to `tests/test_phase12_training_loop.py`:

```python
def test_phase12_oracle_rollout_records_policy_and_wm_frames_separately(monkeypatch, tmp_path) -> None:
    class Step:
        reward = 1.0
        success = True
        terminated = True
        truncated = False

    class EnvHarness:
        def __init__(self) -> None:
            self.policy_frame = np.full((2, 2, 3), 11, dtype=np.uint8)
            self.wm_frame = np.full((2, 2, 3), 22, dtype=np.uint8)

        def reset(self, seed):
            assert seed == 123
            return {"pixels": self.policy_frame}

        def render_frame(self):
            return self.policy_frame

        def render_frame_for_wm(self):
            return self.wm_frame

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

    np.testing.assert_array_equal(oracle["frames"][0], np.full((2, 2, 3), 11, dtype=np.uint8))
    np.testing.assert_array_equal(oracle["wm_frames"][0], np.full((2, 2, 3), 22, dtype=np.uint8))
    assert len(oracle["wm_frames"]) == len(oracle["frames"])
```

- [ ] **Step 2: Run test to verify fail**

Run:

```bash
PYTHONPATH=src /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python -m pytest tests/test_phase12_training_loop.py::test_phase12_oracle_rollout_records_policy_and_wm_frames_separately -v
```

Expected: FAIL with `KeyError: 'wm_frames'`.

- [ ] **Step 3: Update oracle rollout to record WM frames**

In `_rollout_phase12_oracle()`, replace initial frame setup with:

```python
    frames: list[np.ndarray] = [np.asarray(env_h.render_frame(), dtype=np.uint8)]
    wm_frames: list[np.ndarray] = [np.asarray(env_h.render_frame_for_wm(), dtype=np.uint8)]
    proprios: list[np.ndarray] = [np.asarray(env_h.last_agent_pos(), dtype=np.float32)]
    raw_obs: list[np.ndarray] = [np.asarray(env_h.last_raw_obs(), dtype=np.float64)]
```

Inside loop after `step = env_h.step(action)`, replace frame append block with:

```python
        frames.append(np.asarray(env_h.render_frame(), dtype=np.uint8))
        wm_frames.append(np.asarray(env_h.render_frame_for_wm(), dtype=np.uint8))
        proprios.append(np.asarray(env_h.last_agent_pos(), dtype=np.float32))
        raw_obs.append(np.asarray(env_h.last_raw_obs(), dtype=np.float64))
```

In returned dict, add:

```python
        "wm_frames": wm_frames,
```

- [ ] **Step 4: Use `oracle["wm_frames"]` for goal encode**

In `collect_phase12_training_episode()`, replace:

```python
                oracle["frames"][frame_idx - 1],
```

with:

```python
                oracle["wm_frames"][frame_idx - 1],
```

Keep `frame_path` pointing at saved policy/oracle PNGs for human artifact compatibility.

- [ ] **Step 5: Run targeted tests**

Run:

```bash
PYTHONPATH=src /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python -m pytest tests/test_phase12_training_loop.py::test_phase12_oracle_rollout_records_policy_and_wm_frames_separately tests/test_phase12_wm_reward.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/grpo/train_phase12_wm_chunk_grpo.py tests/test_phase12_training_loop.py
git commit -m "fix: encode phase12 goals with wm frames"
```

## Task 5: Wire Initial WM Frame And Decode Real Frames

**Files:**
- Modify: `scripts/grpo/train_phase12_wm_chunk_grpo.py`
- Modify: `tests/test_phase12_training_loop.py`

- [ ] **Step 1: Write failing decode-real-frame test**

Append to `tests/test_phase12_training_loop.py`:

```python
def test_phase12_decode_artifacts_use_wm_frames_for_real_vs_pred(monkeypatch, tmp_path) -> None:
    policy_frames = [np.full((2, 2, 3), 11, dtype=np.uint8)]
    wm_frames = [np.full((2, 2, 3), 22, dtype=np.uint8)]
    seen: dict[str, object] = {}

    def fake_build_decode_artifacts(**kwargs):
        seen["real_frames"] = kwargs["real_frames"]
        return SimpleNamespace(paths={}, metadata={"decode_status": "ok"})

    monkeypatch.setattr(trainer, "build_decode_artifacts", fake_build_decode_artifacts, raising=False)

    class Episode:
        segments = [SimpleNamespace(selected_candidate_index=0)]
        metadata = {}

    class RolloutEnv:
        frames = policy_frames
        wm_frames = wm_frames
        rewards = []
        successes = []

    score_inputs = {(0, 0): {"image": wm_frames[0], "proprio": np.zeros(4), "actions": np.zeros((5, 4))}}
    meta: dict[str, object] = {}

    trainer._build_phase12_selected_decode_artifacts(
        args=SimpleNamespace(save_wm_decodes=True, strict_decode=True, goal_latent_mode="visual_proprio", chunk_len=5),
        episode=Episode(),
        episode_dir=tmp_path,
        rollout_env=RolloutEnv(),
        score_inputs=score_inputs,
        wm_bundle=SimpleNamespace(planner_action_dim=20),
        action_dim=4,
        meta=meta,
    )

    np.testing.assert_array_equal(seen["real_frames"][0], wm_frames[0])
```

- [ ] **Step 2: Run test to verify fail**

Run:

```bash
PYTHONPATH=src /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python -m pytest tests/test_phase12_training_loop.py::test_phase12_decode_artifacts_use_wm_frames_for_real_vs_pred -v
```

Expected: FAIL with `AttributeError: module 'scripts.grpo.train_phase12_wm_chunk_grpo' has no attribute '_build_phase12_selected_decode_artifacts'`.

- [ ] **Step 3: Extract selected decode artifact helper**

In `scripts/grpo/train_phase12_wm_chunk_grpo.py`, add helper near `_merge_phase12_decode_metadata()`:

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
    real_frames = getattr(rollout_env, "wm_frames", None)
    if real_frames is None:
        real_frames = rollout_env.frames
    decode_result = build_decode_artifacts(
        decode_fn=lambda: _decode_phase12_prediction_frames(
            wm_bundle,
            image=decode_input["image"],
            proprio=decode_input["proprio"],
            actions=decode_input["actions"],
            mode=args.goal_latent_mode,
        ),
        output_dir=episode_dir,
        real_frames=list(real_frames),
        strict_decode=bool(args.strict_decode),
        segment_index=0,
        selected_candidate_index=int(first_segment.selected_candidate_index),
        env_steps_per_wm_step=max(1, int(wm_bundle.planner_action_dim) // max(1, action_dim)),
        carried_steps=min(int(args.chunk_len), max(0, len(real_frames) - 1)),
    )
    _merge_phase12_decode_metadata(meta, decode_result.metadata)
    meta.setdefault("wm_decode_real_frame_source", "wm_frames")
```

- [ ] **Step 4: Replace inline decode block**

In `collect_phase12_training_episode()`, replace existing `if bool(args.save_wm_decodes) and episode.segments:` block with:

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

In `collect_phase12_training_episode()`, after reset:

```python
        reset_frame = np.asarray(env_h.render_frame())
        reset_wm_frame = np.asarray(env_h.render_frame_for_wm())
```

Update `_Phase12SelectedRolloutEnv(...)` call:

```python
        rollout_env = _Phase12SelectedRolloutEnv(
            env_h=env_h,
            bundle=bundle,
            seed=reset_seed,
            initial_obs=reset_obs,
            initial_frame=reset_frame,
            initial_wm_frame=reset_wm_frame,
            initial_proprio=reset_proprio,
        )
```

- [ ] **Step 6: Run decode/root tests**

Run:

```bash
PYTHONPATH=src /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python -m pytest tests/test_phase12_training_loop.py::test_phase12_selected_rollout_root_uses_wm_image_for_scoring_but_policy_obs_for_proc tests/test_phase12_training_loop.py::test_phase12_decode_artifacts_use_wm_frames_for_real_vs_pred tests/test_phase12_diagnostics.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add scripts/grpo/train_phase12_wm_chunk_grpo.py tests/test_phase12_training_loop.py
git commit -m "fix: decode phase12 against wm frames"
```

## Task 6: Record Pixel Contract In Manifest And Progress Metadata

**Files:**
- Modify: `scripts/grpo/train_phase12_wm_chunk_grpo.py`
- Modify: `tests/test_phase12_trainer_static.py`

- [ ] **Step 1: Write failing manifest test**

Append to `tests/test_phase12_trainer_static.py`:

```python
def test_phase12_manifest_records_split_pixel_contract(tmp_path) -> None:
    args = parse_args(["--output-dir", str(tmp_path), "--dry-run"])

    manifest = build_manifest(args)

    assert manifest["policy_pixel_contract"] == "lerobot_corner2_vhflip"
    assert manifest["wm_pixel_contract"] == "jepa_corner2_vflip"
    assert manifest["wm_goal_pixel_source"] == "oracle_wm_frames"
    assert manifest["wm_decode_real_frame_source"] == "wm_frames"
```

- [ ] **Step 2: Run test to verify fail**

Run:

```bash
PYTHONPATH=src /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python -m pytest tests/test_phase12_trainer_static.py::test_phase12_manifest_records_split_pixel_contract -v
```

Expected: FAIL with `KeyError: 'policy_pixel_contract'`.

- [ ] **Step 3: Add manifest fields**

In `build_manifest()` returned dict, add:

```python
        "policy_pixel_contract": "lerobot_corner2_vhflip",
        "wm_pixel_contract": "jepa_corner2_vflip",
        "wm_goal_pixel_source": "oracle_wm_frames",
        "wm_decode_real_frame_source": "wm_frames",
```

- [ ] **Step 4: Add episode metadata fields**

In `collect_phase12_training_episode()`, when building `meta.update({...})`, add:

```python
                "policy_pixel_contract": "lerobot_corner2_vhflip",
                "wm_pixel_contract": "jepa_corner2_vflip",
                "wm_goal_pixel_source": "oracle_wm_frames",
                "wm_decode_real_frame_source": "wm_frames",
```

- [ ] **Step 5: Run static tests**

Run:

```bash
PYTHONPATH=src /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python -m pytest tests/test_phase12_trainer_static.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/grpo/train_phase12_wm_chunk_grpo.py tests/test_phase12_trainer_static.py
git commit -m "fix: record phase12 pixel contracts"
```

## Task 7: Full Unit Regression Sweep

**Files:**
- No code changes expected.

- [ ] **Step 1: Run focused Phase12 tests**

Run:

```bash
PYTHONPATH=src /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python -m pytest \
  tests/test_phase12_pixels.py \
  tests/test_grpo_lerobot_adapter.py \
  tests/test_phase12_training_loop.py \
  tests/test_phase12_wm_reward.py \
  tests/test_phase12_diagnostics.py \
  tests/test_phase12_trainer_static.py \
  tests/test_phase12_artifacts.py \
  -v
```

Expected: PASS.

- [ ] **Step 2: Run Phase08/WM parity tests**

Run:

```bash
PYTHONPATH=src /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python -m pytest \
  tests/test_run_segment_grpo_main.py \
  tests/test_segment_grpo_loop.py \
  tests/test_metaworld_jepa_render.py \
  -v
```

Expected: PASS. `test_metaworld_jepa_render.py` may skip only if MetaWorld missing; in project env it should run.

- [ ] **Step 3: Commit test-only fixes if needed**

If tests reveal stale mocks only, commit after fix:

```bash
git add tests scripts src
git commit -m "test: lock phase12 wm pixel contract"
```

If tests reveal behavior bug, stop and return to relevant earlier task.

## Task 8: One-Update Smoke Verification

**Files:**
- No code changes expected unless smoke reveals bug.

- [ ] **Step 1: Run bounded one-update smoke locally or via Slurm**

Run local only if GPU/MuJoCo env is ready:

```bash
cd /vol/bitbucket/aa6622/project
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 JEPA_WM_DISABLE_IMAGE_HEAD=0 \
/vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python scripts/grpo/train_phase12_wm_chunk_grpo.py \
  --mode wm_grpo_train \
  --checkpoint /vol/bitbucket/aa6622/.cache/huggingface/hub/models--jadechoghari--smolvla_metaworld/snapshots/ef3089ecb84eeeb7d33fedab24f6c76180a68900 \
  --jepa-repo "/vol/bitbucket/aa6622/VGG JEPA/jepa-wms" \
  --jepa-ckpt jepa_wm_metaworld.pth.tar \
  --output-dir artifacts/phase12_wm_chunk_grpo_train/push-v3/pixel_contract_fix_bounded_u1_seed2000 \
  --action-profile bounded_executed \
  --num-episodes 1 \
  --num-updates 1 \
  --train-seed-base 2000 \
  --chunk-len 25 \
  --group-size 4 \
  --max-steps 120 \
  --strict-decode
```

Expected:

```text
PHASE12_WM_CHUNK_GRPO_TRAIN_DONE updates=1 out=...
```

- [ ] **Step 2: Check smoke artifacts exist**

Run:

```bash
python - <<'PY'
from pathlib import Path
import json

root = Path("/vol/bitbucket/aa6622/project/artifacts/phase12_wm_chunk_grpo_train/push-v3/pixel_contract_fix_bounded_u1_seed2000")
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
print(json.dumps(smoke, indent=2))
PY
```

Expected: prints smoke manifest; no `bad artifact`.

- [ ] **Step 3: Check manifest pixel contract**

Run:

```bash
python - <<'PY'
from pathlib import Path
import json

root = Path("/vol/bitbucket/aa6622/project/artifacts/phase12_wm_chunk_grpo_train/push-v3/pixel_contract_fix_bounded_u1_seed2000")
manifest = json.loads((root / "train_manifest.json").read_text())
expected = {
    "policy_pixel_contract": "lerobot_corner2_vhflip",
    "wm_pixel_contract": "jepa_corner2_vflip",
    "wm_goal_pixel_source": "oracle_wm_frames",
    "wm_decode_real_frame_source": "wm_frames",
}
for key, value in expected.items():
    if manifest.get(key) != value:
        raise SystemExit(f"{key}: expected {value!r}, got {manifest.get(key)!r}")
print("pixel contract ok")
PY
```

Expected:

```text
pixel contract ok
```

- [ ] **Step 4: Compare blur metric before/after**

Run:

```bash
python - <<'PY'
from pathlib import Path
import numpy as np
from PIL import Image

def edge_var(path: Path) -> tuple[float, float]:
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    h = arr.shape[0] // 2
    top = arr[:h]
    bottom = arr[h:]
    def score(x):
        gx = np.diff(x, axis=1)
        gy = np.diff(x, axis=0)
        return float(np.var(gx) + np.var(gy))
    return score(top), score(bottom)

old = Path("/vol/bitbucket/aa6622/project/artifacts/phase12_wm_chunk_grpo_train/push-v3/debug_decode_postfix_bounded_u1_seed2000/rollouts/update_0000_episode_0000/segment_0000/wm_real_vs_pred_selected_strip.png")
new = Path("/vol/bitbucket/aa6622/project/artifacts/phase12_wm_chunk_grpo_train/push-v3/pixel_contract_fix_bounded_u1_seed2000/rollouts/update_0000_episode_0000/segment_0000/wm_real_vs_pred_selected_strip.png")
for label, path in [("old", old), ("new", new)]:
    real, pred = edge_var(path)
    print(label, "real_edge_var", round(real, 2), "pred_edge_var", round(pred, 2), "ratio", round(pred / max(real, 1e-6), 3))
PY
```

Expected: new `ratio` materially higher than old and visual strip no longer smeared/warped. If ratio stays low, do not continue to 100-update; inspect whether WM image hashes match `render_jepa_rgb` orientation.

- [ ] **Step 5: Commit smoke notes if project normally tracks findings**

If adding a finding note, create `docs/findings/2026-05-15-phase12-wm-pixel-contract-smoke.md` with paths and metric output, then:

```bash
git add docs/findings/2026-05-15-phase12-wm-pixel-contract-smoke.md
git commit -m "docs: record phase12 wm pixel smoke"
```

Skip commit if no docs note is created.

## Task 9: Official Profile Smoke Then 100-Update Gate

**Files:**
- No code changes expected.

- [ ] **Step 1: Run `official_jepa_mirror` one-update smoke**

Run:

```bash
cd /vol/bitbucket/aa6622/project
sbatch scripts/grpo/submit_phase12_wm_chunk_grpo_train.slurm official_jepa_mirror 1 artifacts/phase12_wm_chunk_grpo_train/push-v3/pixel_contract_fix_official_u1_seed2000
```

Expected: Slurm prints submitted job id.

- [ ] **Step 2: Run `bounded_executed` one-update smoke**

After official smoke passes:

```bash
cd /vol/bitbucket/aa6622/project
sbatch scripts/grpo/submit_phase12_wm_chunk_grpo_train.slurm bounded_executed 1 artifacts/phase12_wm_chunk_grpo_train/push-v3/pixel_contract_fix_bounded_u1_seed2000
```

Expected: Slurm prints submitted job id.

- [ ] **Step 3: Verify both smoke manifests**

Run:

```bash
python - <<'PY'
from pathlib import Path
import json

roots = [
    Path("/vol/bitbucket/aa6622/project/artifacts/phase12_wm_chunk_grpo_train/push-v3/pixel_contract_fix_official_u1_seed2000"),
    Path("/vol/bitbucket/aa6622/project/artifacts/phase12_wm_chunk_grpo_train/push-v3/pixel_contract_fix_bounded_u1_seed2000"),
]
for root in roots:
    smoke = json.loads((root / "smoke_manifest.json").read_text())
    if smoke["wm_decode_status"] != "ok":
        raise SystemExit(f"{root}: decode status {smoke['wm_decode_status']}")
    for key in ("wm_decode_selected_strip_path", "wm_real_vs_pred_selected_strip_path"):
        path = Path(smoke[key])
        if not path.is_file() or path.stat().st_size <= 0:
            raise SystemExit(f"{root}: missing {key} -> {path}")
    print(root.name, "ok")
PY
```

Expected:

```text
pixel_contract_fix_official_u1_seed2000 ok
pixel_contract_fix_bounded_u1_seed2000 ok
```

- [ ] **Step 4: Submit 100-update main run only after both smokes pass**

Run:

```bash
cd /vol/bitbucket/aa6622/project
sbatch scripts/grpo/submit_phase12_wm_chunk_grpo_train.slurm official_jepa_mirror 100 artifacts/phase12_wm_chunk_grpo_train/push-v3/pixel_contract_fix_official_u100_seed2000
```

Expected: Slurm prints submitted job id.

## Self-Review

- Spec coverage:
  - Root cause addressed: WM no longer receives policy V+H pixels.
  - Phase08 parity addressed: WM gets vertical-only `corner2`; policy keeps V+H.
  - Goal encode addressed: oracle goals encoded from WM stream.
  - Decode artifact addressed: real-vs-pred compares WM decode against WM real frames.
  - Tests addressed: pure helpers, adapter stream, rollout root, oracle frames, decode real frames, manifest.
- Placeholder scan:
  - No `TBD`, `TODO`, or unspecified "add tests" steps.
  - Every code-change task includes code snippets and exact commands.
- Type consistency:
  - `render_frame_for_wm()` exists on `DeferredLeRobotMetaworldEnv` and `OfficialLeRobotMetaWorldGRPORollout`.
  - `_Phase12SelectedRolloutEnv` receives `initial_wm_frame`, stores `wm_frames`, returns root `"image"` as WM frame and `"policy_image"` as policy frame.
  - `_build_phase12_selected_decode_artifacts()` consumes `rollout_env.wm_frames`.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-15-phase12-wm-pixel-contract-fix.md`. Two execution options:

1. **Subagent-Driven (recommended)** - dispatch fresh subagent per task, review between tasks, fast iteration
2. **Inline Execution** - execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
