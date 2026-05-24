# Phase12 Strict JEPA WM Render Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Phase12 WM pixel path that reproduces `jepa_wm_metaworld` native `224x224` same-env render contract while preserving LeRobot policy/eval pixels.

**Architecture:** Phase12 keeps the LeRobot-compatible rollout stack. The underlying MetaWorld simulator is reused for a second render view only: policy/eval gets current LeRobot `480x480` V+H pixels, WM/goal/decode can get isolated JEPA `224x224` V-only pixels from the same `self._env.model` and `self._env.data`. Renderer mutation is contained by saving and restoring `width`, `height`, `camera_name`, and `mujoco_renderer` around every WM render.

**Tech Stack:** Python, NumPy, Gymnasium vector envs, MetaWorld/MuJoCo renderer, LeRobot env preprocessing, PyTorch JEPA-WM.

---

## Agreement With LLM Review

- Agree with review: same-env render is correct; easy/no-risk framing is wrong.
- Main risk: renderer contamination. If `env.width`, `env.height`, `env.camera_name`, or `env.mujoco_renderer` stays at JEPA `224`, later LeRobot `obs["pixels"]` can become `224` and break SmolVLA input contract.
- Required invariant: after every `render_jepa_frame(224)`, normal `render()` still returns LeRobot policy pixels at configured observation size with V+H flip.
- Required timing: capture JEPA WM frame immediately after reset/step observation capture. No env step between policy obs and WM frame.

## Files

- Modify: `src/smolvla_grpo/lerobot_metaworld_adapter.py`
  - Add same-env `render_jepa_frame(img_size=224)` to `DeferredLeRobotMetaworldEnv`.
  - Add vector-call wrapper on `OfficialLeRobotMetaWorldGRPORollout`.
- Modify: `src/segment_grpo_loop.py`
  - Add optional `_to_wm_visual(..., resize_mode=...)` so strict JEPA path bypasses legacy `256` bridge.
- Modify: `src/smolvla_grpo/phase12_wm_reward.py`
  - Thread WM visual resize mode through encode/score.
- Modify: `scripts/grpo/train_phase12_wm_chunk_grpo.py`
  - Add CLI flags and manifest fields.
  - Route root/goal/decode WM frames from selected source.
  - Save diagnostic artifacts proving pixel contracts.
- Modify: `tests/test_grpo_lerobot_adapter.py`
  - Add renderer isolation tests.
- Modify: `tests/test_segment_grpo_loop.py`
  - Add passthrough WM visual resize test.
- Modify: `tests/test_phase12_wm_reward.py`
  - Add resize-mode threading test.
- Modify: `tests/test_phase12_trainer_static.py`
  - Add CLI/manifest/source-routing static tests.

---

### Task 1: Add Same-Env JEPA Render Tests

**Files:**
- Modify: `tests/test_grpo_lerobot_adapter.py`

- [ ] **Step 1: Extend fake MetaWorld env to expose mutable renderer state**

In `_install_fake_deferred_deps()`, replace `FakeInner` with this compatible version. It keeps default fake render size at `8x8` so existing tests keep passing, while allowing per-call `width`/`height`.

```python
    class FakeRenderer:
        def __init__(self, width=8, height=8, camera_name="corner2"):
            self.default_cam_config = {"trackbodyid": -1}
            self.max_geom = 1000
            self.width = int(width)
            self.height = int(height)
            self.camera_name = str(camera_name)

    class FakeInner:
        max_path_length = 500
        model = type("Model", (), {"cam_pos": {2: [0.0, 0.0, 0.0]}})()
        data = object()

        def __init__(self, *args, **kwargs):
            self.raw = np.array([1.0, 2.0, 3.0, 4.0, 9.0], dtype=np.float64)
            self.seeded_rand_vec = False
            self.seed_calls: list[int] = []
            self.reset_calls: list[int | None] = []
            self.render_calls: list[dict[str, object]] = []
            self.width = 8
            self.height = 8
            self.camera_name = kwargs.get("camera_name", "corner2")
            self.render_mode = kwargs.get("render_mode", "rgb_array")
            self.mujoco_renderer = FakeRenderer(self.width, self.height, self.camera_name)

        def set_task(self, task):
            self.task = task

        def seed(self, seed):
            self.seed_calls.append(int(seed))
            return [int(seed)]

        def reset(self, seed=None):
            self.reset_calls.append(None if seed is None else int(seed))
            self.raw = np.array([1.0, 2.0, 3.0, 4.0, 9.0], dtype=np.float64)
            return self.raw.copy(), {}

        def step(self, action):
            self.raw = np.asarray(action, dtype=np.float64).reshape(-1)
            self.raw = np.pad(self.raw, (0, max(0, 5 - self.raw.size)), constant_values=0.0)
            return self.raw.copy(), 1.0, False, False, {"success": False}

        def render(self, *args, **kwargs):
            width = int(kwargs.get("width", getattr(self, "width", 8)))
            height = int(kwargs.get("height", getattr(self, "height", 8)))
            self.render_calls.append(
                {
                    "width": width,
                    "height": height,
                    "camera_name": getattr(self, "camera_name", None),
                    "renderer": self.mujoco_renderer,
                }
            )
            return np.arange(height * width * 3, dtype=np.uint8).reshape(height, width, 3)

        def close(self):
            return None
```

- [ ] **Step 2: Patch fake `MujocoRenderer` dependency**

Still inside `_install_fake_deferred_deps()`, add this after the existing `monkeypatch.setattr(...)` calls:

```python
    import gymnasium.envs.mujoco.mujoco_rendering as mujoco_rendering

    monkeypatch.setattr(mujoco_rendering, "MujocoRenderer", FakeRenderer)
```

- [ ] **Step 3: Add direct env test for JEPA render contract and restoration**

Append test:

```python
def test_deferred_metaworld_render_jepa_frame_is_224_vflip_and_restores_policy_renderer(monkeypatch):
    _install_fake_deferred_deps(monkeypatch)
    from smolvla_grpo.lerobot_metaworld_adapter import DeferredLeRobotMetaworldEnv

    env = DeferredLeRobotMetaworldEnv(task="push-v3", camera_name="corner2")
    try:
        obs, _info = env.reset(seed=123)
        raw8 = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
        np.testing.assert_array_equal(obs["pixels"], np.flip(raw8, (0, 1)))

        inner = env._env  # noqa: SLF001
        original_renderer = inner.mujoco_renderer
        original_width = inner.width
        original_height = inner.height
        original_camera = inner.camera_name

        jepa = env.render_jepa_frame(img_size=4)
        raw4 = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)

        assert jepa.shape == (4, 4, 3)
        np.testing.assert_array_equal(jepa, np.flip(raw4, 0))
        assert inner.mujoco_renderer is original_renderer
        assert inner.width == original_width
        assert inner.height == original_height
        assert inner.camera_name == original_camera

        policy_after = env.render_frame()
        np.testing.assert_array_equal(policy_after, np.flip(raw8, (0, 1)))
    finally:
        env.close()
```

- [ ] **Step 4: Add vector wrapper test**

Append test:

```python
def test_expert_oracle_rollout_exposes_jepa_frame_without_changing_policy_pixels(monkeypatch):
    _install_fake_deferred_deps(monkeypatch)
    from smolvla_grpo.lerobot_metaworld_adapter import OfficialLeRobotMetaWorldGRPORollout

    rollout = OfficialLeRobotMetaWorldGRPORollout(
        task="push-v3",
        n_envs=1,
        enable_expert_oracle=True,
    )
    try:
        obs = rollout.reset(123)
        jepa = rollout.render_jepa_frame(img_size=4)
        raw4 = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
        raw8 = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)

        assert jepa.shape == (4, 4, 3)
        np.testing.assert_array_equal(jepa, np.flip(raw4, 0))
        np.testing.assert_array_equal(obs["pixels"][0], np.flip(raw8, (0, 1)))
        np.testing.assert_array_equal(rollout.render_frame(), np.flip(raw8, (0, 1)))
    finally:
        rollout.close()
```

- [ ] **Step 5: Run tests and verify failure**

Run:

```bash
pytest tests/test_grpo_lerobot_adapter.py::test_deferred_metaworld_render_jepa_frame_is_224_vflip_and_restores_policy_renderer tests/test_grpo_lerobot_adapter.py::test_expert_oracle_rollout_exposes_jepa_frame_without_changing_policy_pixels -q
```

Expected: FAIL with `AttributeError` mentioning `render_jepa_frame`.

---

### Task 2: Implement Isolated Same-Env JEPA Render

**Files:**
- Modify: `src/smolvla_grpo/lerobot_metaworld_adapter.py`

- [ ] **Step 1: Add RGB coercion helper near `_image_debug`**

Add:

```python
def _to_rgb_uint8(frame: Any) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim != 3 or arr.shape[-1] not in (3, 4):
        raise ValueError(f"expected HxWx3/4 RGB frame, got {arr.shape}")
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating) and float(np.max(arr)) <= 1.5:
            arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)
```

- [ ] **Step 2: Add raw render helper inside `DeferredLeRobotMetaworldEnv`**

Add method before `render()`:

```python
    def _render_raw_corner2_at_size(self, img_size: int) -> np.ndarray:
        self._ensure_env()
        assert self._env is not None
        env = self._env
        size = int(img_size)
        if size <= 0:
            raise ValueError(f"img_size must be positive, got {img_size}")

        old_width = getattr(env, "width", None)
        old_height = getattr(env, "height", None)
        old_camera_name = getattr(env, "camera_name", None)
        old_renderer = getattr(env, "mujoco_renderer", None)

        try:
            env.camera_name = "corner2"
            env.width = size
            env.height = size

            raw = self._try_render_with_size(size)
            if raw.shape[:2] != (size, size):
                raw = self._render_with_temporary_mujoco_renderer(size, old_renderer)
            if raw.shape[:2] != (size, size):
                raise RuntimeError(f"JEPA render produced {raw.shape[:2]}, expected {(size, size)}")
            return raw
        finally:
            if old_camera_name is not None:
                env.camera_name = old_camera_name
            if old_width is not None:
                env.width = old_width
            if old_height is not None:
                env.height = old_height
            if old_renderer is not None:
                env.mujoco_renderer = old_renderer
```

- [ ] **Step 3: Add size-aware render fallback helpers**

Add methods below `_render_raw_corner2_at_size`:

```python
    def _try_render_with_size(self, img_size: int) -> np.ndarray:
        assert self._env is not None
        try:
            frame = self._env.render(width=int(img_size), height=int(img_size))
        except TypeError:
            frame = self._env.render()
        return _to_rgb_uint8(frame)

    def _render_with_temporary_mujoco_renderer(self, img_size: int, old_renderer: Any) -> np.ndarray:
        assert self._env is not None
        if old_renderer is None:
            raise RuntimeError("MetaWorld env has no mujoco_renderer; cannot build isolated JEPA renderer")
        from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

        self._env.mujoco_renderer = MujocoRenderer(
            self._env.model,
            self._env.data,
            old_renderer.default_cam_config,
            width=int(img_size),
            height=int(img_size),
            max_geom=old_renderer.max_geom,
            camera_id=None,
            camera_name="corner2",
        )
        return _to_rgb_uint8(self._env.render())
```

- [ ] **Step 4: Add public JEPA render method**

Add method after `render()`:

```python
    def render_jepa_frame(self, img_size: int = 224) -> np.ndarray:
        """Same-state JEPA-WM MetaWorld RGB: corner2, square size, vertical flip only."""
        raw_image = self._render_raw_corner2_at_size(int(img_size))
        image = np.flip(raw_image, 0)
        return np.ascontiguousarray(image)
```

- [ ] **Step 5: Add rollout wrapper method**

Add to `OfficialLeRobotMetaWorldGRPORollout` after `render_frame()`:

```python
    def render_jepa_frame(self, img_size: int = 224, env_index: int = 0) -> np.ndarray:
        frames = self.vec_env.call("render_jepa_frame", int(img_size))
        return np.asarray(frames[int(env_index)])
```

- [ ] **Step 6: Run adapter tests**

Run:

```bash
pytest tests/test_grpo_lerobot_adapter.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/smolvla_grpo/lerobot_metaworld_adapter.py tests/test_grpo_lerobot_adapter.py
git commit -m "feat: add isolated JEPA render view"
```

---

### Task 3: Add WM Visual Resize Modes

**Files:**
- Modify: `src/segment_grpo_loop.py`
- Modify: `tests/test_segment_grpo_loop.py`

- [ ] **Step 1: Add failing passthrough test**

Append near `test_to_wm_visual_feeds_jepa_hub_encode_range`:

```python
def test_to_wm_visual_passthrough_keeps_input_resolution_for_jepa_preprocessor() -> None:
    torch = pytest.importorskip("torch")
    image = np.full((24, 32, 3), 255, dtype=np.uint8)

    t = _to_wm_visual(image, torch.device("cpu"), resize_mode="passthrough")

    assert t.shape == (1, 1, 3, 24, 32)
    assert float(t.max()) > 200.0
    assert float(t.min()) >= 0.0
```

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
pytest tests/test_segment_grpo_loop.py::test_to_wm_visual_passthrough_keeps_input_resolution_for_jepa_preprocessor -q
```

Expected: FAIL with unexpected keyword `resize_mode`.

- [ ] **Step 3: Implement resize mode**

Change `_to_wm_visual` signature and body:

```python
def _to_wm_visual(image: Any, device: torch.device, *, resize_mode: str = "legacy_256") -> torch.Tensor:
    _require_torch("WM visual conversion requires torch.")
    rgb = _to_rgb_uint8(image)
    if not rgb.flags.writeable:
        rgb = rgb.copy()
    # JEPA hub EncPredWM.encode divides by 255 once; feed float RGB in [0, 255], not [0, 1].
    tensor = torch.from_numpy(rgb).float()
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # 1,3,H,W
    mode = str(resize_mode)
    if mode == "legacy_256":
        tensor = torch.nn.functional.interpolate(
            tensor, size=(256, 256), mode="bilinear", align_corners=False
        )  # 1,3,256,256
    elif mode == "passthrough":
        pass
    else:
        raise ValueError(f"Unknown WM visual resize mode: {resize_mode}")
    return tensor.unsqueeze(0).to(device)  # 1,1,3,H,W
```

- [ ] **Step 4: Run targeted tests**

Run:

```bash
pytest tests/test_segment_grpo_loop.py::test_to_wm_visual_feeds_jepa_hub_encode_range tests/test_segment_grpo_loop.py::test_to_wm_visual_passthrough_keeps_input_resolution_for_jepa_preprocessor -q
```

Expected: PASS. Existing default stays `256x256`.

- [ ] **Step 5: Commit**

```bash
git add src/segment_grpo_loop.py tests/test_segment_grpo_loop.py
git commit -m "feat: support JEPA passthrough WM visuals"
```

---

### Task 4: Thread Resize Mode Through WM Reward

**Files:**
- Modify: `src/smolvla_grpo/phase12_wm_reward.py`
- Modify: `tests/test_phase12_wm_reward.py`

- [ ] **Step 1: Add failing threading test**

Append:

```python
def test_score_phase12_chunk_threads_wm_visual_resize_mode() -> None:
    class ShapeRecordingWM(FakeWM):
        class Model:
            action_dim = 4

            def __init__(self) -> None:
                self.visual_shape = None

            def encode(self, obs):
                self.visual_shape = tuple(obs["visual"].shape)
                return {"visual": torch.zeros(1, 1, 1), "proprio": obs["proprio"]}

            def unroll(self, z, *, act_suffix, debug=False):
                del act_suffix, debug
                return z

        model = Model()

    wm = ShapeRecordingWM()
    score_phase12_chunk_with_wm(
        wm_bundle=wm,
        image=np.zeros((24, 32, 3), dtype=np.uint8),
        proprio=np.zeros(2, dtype=np.float32),
        chunk_actions=np.zeros((1, 4), dtype=np.float32),
        goal={"visual": torch.zeros(1, 1, 1), "proprio": torch.zeros(1, 1, 2)},
        candidate_index=0,
        proprio_alpha=0.1,
        mode="visual_proprio",
        wm_visual_resize_mode="passthrough",
    )

    assert wm.model.visual_shape == (1, 1, 3, 24, 32)
```

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
pytest tests/test_phase12_wm_reward.py::test_score_phase12_chunk_threads_wm_visual_resize_mode -q
```

Expected: FAIL with unexpected keyword `wm_visual_resize_mode`.

- [ ] **Step 3: Update `_encode_structured`**

Change signature and visual call:

```python
def _encode_structured(
    wm_bundle: Any,
    image: np.ndarray,
    proprio: np.ndarray,
    *,
    mode: str,
    wm_visual_resize_mode: str = "legacy_256",
) -> dict[str, torch.Tensor]:
    obs = {
        "visual": _to_wm_visual(image, wm_bundle.device, resize_mode=wm_visual_resize_mode),
        "proprio": _to_wm_proprio(proprio, int(wm_bundle.proprio_dim), wm_bundle.device),
    }
```

- [ ] **Step 4: Update score function signature and call**

Change `score_phase12_chunk_with_wm` signature:

```python
def score_phase12_chunk_with_wm(
    *,
    wm_bundle: Any,
    image: np.ndarray,
    proprio: np.ndarray,
    chunk_actions: np.ndarray,
    goal: Mapping[str, torch.Tensor],
    candidate_index: int,
    proprio_alpha: float,
    mode: str,
    debug_npz_path: str | None = None,
    wm_visual_resize_mode: str = "legacy_256",
) -> Phase12Score:
```

Change start encode:

```python
    start = _encode_structured(
        wm_bundle,
        image,
        proprio,
        mode=mode,
        wm_visual_resize_mode=wm_visual_resize_mode,
    )
```

- [ ] **Step 5: Run tests**

Run:

```bash
pytest tests/test_phase12_wm_reward.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/smolvla_grpo/phase12_wm_reward.py tests/test_phase12_wm_reward.py
git commit -m "feat: thread WM visual resize mode"
```

---

### Task 5: Add Phase12 WM Frame Source CLI And Routing

**Files:**
- Modify: `scripts/grpo/train_phase12_wm_chunk_grpo.py`
- Modify: `tests/test_phase12_trainer_static.py`

- [ ] **Step 1: Add CLI defaults test**

In `test_phase12_cli_defaults`, add:

```python
    assert args.wm_frame_source == "lerobot_unflip"
    assert args.wm_render_img_size == 224
    assert args.wm_visual_resize_mode == "legacy_256"
```

- [ ] **Step 2: Add manifest test for strict JEPA mode**

Append:

```python
def test_manifest_records_strict_jepa_wm_frame_contract(tmp_path) -> None:
    args = parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--dry-run",
            "--wm-frame-source",
            "jepa_render_224",
            "--wm-visual-resize-mode",
            "passthrough",
        ]
    )

    manifest = build_manifest(args)

    assert manifest["phase12_policy_frame_contract"] == "lerobot_corner2_vhflip"
    assert manifest["phase12_wm_frame_source"] == "jepa_render_224"
    assert manifest["phase12_wm_frame_contract"] == "jepa_metaworld_corner2_vflip_224"
    assert manifest["phase12_goal_frame_contract"] == "jepa_metaworld_corner2_vflip_224"
    assert manifest["wm_visual_resize_mode"] == "passthrough"
    assert manifest["wm_render_img_size"] == 224
```

- [ ] **Step 3: Add parser args**

Add after `--save-wm-decodes`:

```python
    p.add_argument(
        "--wm-frame-source",
        choices=("lerobot_unflip", "jepa_render_224"),
        default="lerobot_unflip",
        help="Source for WM/goal/decode frames. lerobot_unflip preserves current Phase12 path; jepa_render_224 uses same-env JEPA-WM render.",
    )
    p.add_argument("--wm-render-img-size", type=int, default=224)
    p.add_argument(
        "--wm-visual-resize-mode",
        choices=("legacy_256", "passthrough"),
        default="legacy_256",
    )
```

- [ ] **Step 4: Update manifest**

Replace fixed frame contract fields with:

```python
        "phase12_policy_frame_contract": "lerobot_corner2_vhflip",
        "phase12_wm_frame_source": str(args.wm_frame_source),
        "phase12_wm_frame_contract": (
            "jepa_metaworld_corner2_vflip_224"
            if str(args.wm_frame_source) == "jepa_render_224"
            else "jepa_corner2_vflip_from_lerobot_unflip"
        ),
        "phase12_goal_frame_contract": (
            "jepa_metaworld_corner2_vflip_224"
            if str(args.wm_frame_source) == "jepa_render_224"
            else "jepa_corner2_vflip_from_lerobot_unflip"
        ),
        "phase12_decode_real_frame_source": "wm_frames",
        "wm_render_img_size": int(args.wm_render_img_size),
        "wm_visual_resize_mode": str(args.wm_visual_resize_mode),
```

- [ ] **Step 5: Add helper for WM frame source**

Add near `_write_selected_frames_png`:

```python
def _phase12_wm_frame_from_env(
    *,
    env_h: Any,
    policy_frame: np.ndarray,
    wm_frame_source: str,
    wm_render_img_size: int,
) -> np.ndarray:
    source = str(wm_frame_source)
    if source == "lerobot_unflip":
        return wm_rgb_from_policy_rgb_corner2(policy_frame)
    if source == "jepa_render_224":
        return np.asarray(env_h.render_jepa_frame(int(wm_render_img_size)), dtype=np.uint8)
    raise ValueError(f"Unknown WM frame source: {wm_frame_source}")
```

- [ ] **Step 6: Route selected rollout root and step frames**

In `_Phase12OfficialRolloutAdapter.__init__`, add parameters:

```python
        wm_frame_source: str,
        wm_render_img_size: int,
```

Store them:

```python
        self.wm_frame_source = str(wm_frame_source)
        self.wm_render_img_size = int(wm_render_img_size)
```

Replace:

```python
        self._wm_frame = wm_rgb_from_policy_rgb_corner2(self._frame)
```

with:

```python
        self._wm_frame = _phase12_wm_frame_from_env(
            env_h=self.env_h,
            policy_frame=self._frame,
            wm_frame_source=self.wm_frame_source,
            wm_render_img_size=self.wm_render_img_size,
        )
```

In `step()`, replace:

```python
        self._wm_frame = wm_rgb_from_policy_rgb_corner2(self._frame)
```

with:

```python
        self._wm_frame = _phase12_wm_frame_from_env(
            env_h=self.env_h,
            policy_frame=self._frame,
            wm_frame_source=self.wm_frame_source,
            wm_render_img_size=self.wm_render_img_size,
        )
```

- [ ] **Step 7: Pass routing options when constructing rollout adapter**

Where `_Phase12OfficialRolloutAdapter(...)` is constructed, add:

```python
            wm_frame_source=args.wm_frame_source,
            wm_render_img_size=int(args.wm_render_img_size),
```

- [ ] **Step 8: Thread resize mode into score**

In `score_fn`, add:

```python
                wm_visual_resize_mode=args.wm_visual_resize_mode,
```

to `score_phase12_chunk_with_wm(...)`.

- [ ] **Step 9: Run static tests**

Run:

```bash
pytest tests/test_phase12_trainer_static.py -q
```

Expected: PASS.

- [ ] **Step 10: Commit**

```bash
git add scripts/grpo/train_phase12_wm_chunk_grpo.py tests/test_phase12_trainer_static.py
git commit -m "feat: route Phase12 WM frames from JEPA view"
```

---

### Task 6: Route Oracle Goals And Decode Through Same Source

**Files:**
- Modify: `scripts/grpo/train_phase12_wm_chunk_grpo.py`
- Modify: `tests/test_phase12_trainer_static.py`

- [ ] **Step 1: Add static test that oracle collection uses helper**

Append:

```python
def test_oracle_and_rollout_use_shared_wm_frame_source_helper() -> None:
    source = (trainer._REPO / "scripts" / "grpo" / "train_phase12_wm_chunk_grpo.py").read_text(
        encoding="utf-8"
    )

    assert "_phase12_wm_frame_from_env(" in source
    assert source.count("_phase12_wm_frame_from_env(") >= 3
    assert "wm_rgb_from_policy_rgb_corner2(policy_frame)" not in source
```

- [ ] **Step 2: Update oracle baseline collection signature**

Change `_collect_official_oracle_baseline` signature to include:

```python
    wm_frame_source: str,
    wm_render_img_size: int,
```

- [ ] **Step 3: Use helper in oracle reset and step**

Replace oracle reset WM frame:

```python
    wm_frames: list[np.ndarray] = [
        _phase12_wm_frame_from_env(
            env_h=env_h,
            policy_frame=policy_frame,
            wm_frame_source=wm_frame_source,
            wm_render_img_size=wm_render_img_size,
        )
    ]
```

Replace oracle step append:

```python
        wm_frames.append(
            _phase12_wm_frame_from_env(
                env_h=env_h,
                policy_frame=policy_frame,
                wm_frame_source=wm_frame_source,
                wm_render_img_size=wm_render_img_size,
            )
        )
```

- [ ] **Step 4: Pass args into oracle baseline call**

Where `_collect_official_oracle_baseline(...)` is called, add:

```python
            wm_frame_source=args.wm_frame_source,
            wm_render_img_size=int(args.wm_render_img_size),
```

- [ ] **Step 5: Update decode encode resize mode**

In `_decode_phase12_prediction_frames`, add parameter:

```python
    wm_visual_resize_mode: str = "legacy_256",
```

Change encode call:

```python
        "visual": _to_wm_visual(image, wm_bundle.device, resize_mode=wm_visual_resize_mode),
```

Where `_decode_phase12_prediction_frames(...)` is called, pass:

```python
            wm_visual_resize_mode=args.wm_visual_resize_mode,
```

- [ ] **Step 6: Run tests**

Run:

```bash
pytest tests/test_phase12_trainer_static.py tests/test_phase12_wm_reward.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add scripts/grpo/train_phase12_wm_chunk_grpo.py tests/test_phase12_trainer_static.py
git commit -m "feat: align Phase12 goals and decodes with WM view"
```

---

### Task 7: Add Pixel Contract Diagnostics

**Files:**
- Modify: `scripts/grpo/train_phase12_wm_chunk_grpo.py`

- [ ] **Step 1: Add diagnostic writer**

Add near `_write_selected_frames_png`:

```python
def _write_phase12_pixel_contract_debug(
    *,
    out_dir: Path,
    policy_frame: np.ndarray,
    wm_frame: np.ndarray,
    wm_frame_source: str,
    wm_visual_resize_mode: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    import imageio.v2 as imageio

    policy = np.asarray(policy_frame, dtype=np.uint8)
    wm = np.asarray(wm_frame, dtype=np.uint8)
    imageio.imwrite(out_dir / "policy_frame0.png", policy)
    imageio.imwrite(out_dir / "wm_frame0.png", wm)
    (out_dir / "pixel_contract.json").write_text(
        json.dumps(
            {
                "policy_frame_shape": list(policy.shape),
                "wm_frame_shape": list(wm.shape),
                "wm_frame_source": str(wm_frame_source),
                "wm_visual_resize_mode": str(wm_visual_resize_mode),
                "policy_contract": "lerobot_corner2_vhflip",
                "wm_contract": (
                    "jepa_metaworld_corner2_vflip_224"
                    if str(wm_frame_source) == "jepa_render_224"
                    else "jepa_corner2_vflip_from_lerobot_unflip"
                ),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
```

- [ ] **Step 2: Call diagnostic writer after rollout adapter construction**

After `_Phase12OfficialRolloutAdapter` is created and before `collect_phase12_episode(...)`, add:

```python
        _write_phase12_pixel_contract_debug(
            out_dir=episode_dir / "pixel_contract",
            policy_frame=rollout_env.frames[0],
            wm_frame=rollout_env.wm_frames[0],
            wm_frame_source=args.wm_frame_source,
            wm_visual_resize_mode=args.wm_visual_resize_mode,
        )
```

- [ ] **Step 3: Run trainer static tests**

Run:

```bash
pytest tests/test_phase12_trainer_static.py -q
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add scripts/grpo/train_phase12_wm_chunk_grpo.py
git commit -m "feat: record Phase12 WM pixel contract"
```

---

### Task 8: Run Same-Seed Smoke Matrix

**Files:**
- No source changes unless smoke reveals bug.
- Output artifacts under `artifacts/phase12_wm_crop_ablation/`.

- [ ] **Step 1: Run current control**

Run:

```bash
module load tools/prod Python/3.12.3-GCCcore-13.3.0
"/rds/general/user/aa6622/home/.envs/lerobot_mw_py312/bin/python" scripts/grpo/train_phase12_wm_chunk_grpo.py \
  --mode rollout_validation \
  --jepa-repo /rds/general/user/aa6622/home/research/RESEARCH_PAPER_CLONES/jepa-wms \
  --jepa-ckpt /rds/general/user/aa6622/home/.cache/huggingface/hub/models--facebook--jepa-wms/snapshots/9b9c41ef249466630dbf1a20e78391865d07b3b9/jepa_wm_metaworld.pth.tar \
  --output-dir artifacts/phase12_wm_crop_ablation/control_legacy256 \
  --task push-v3 \
  --num-episodes 1 \
  --num-updates 1 \
  --max-steps 50 \
  --chunk-len 25 \
  --group-size 4 \
  --wm-frame-source lerobot_unflip \
  --wm-visual-resize-mode legacy_256 \
  --decode-candidates selected
```

Expected:
- Exit code `0`.
- `smoke_manifest.json` exists.
- `rollouts/update_0000_episode_0000/segment_0000/wm_real_vs_pred_selected_strip.png` exists.
- `rollouts/update_0000_episode_0000/pixel_contract/pixel_contract.json` says `wm_frame_shape` is `[480, 480, 3]`.

- [ ] **Step 2: Run strict JEPA render**

Run:

```bash
module load tools/prod Python/3.12.3-GCCcore-13.3.0
"/rds/general/user/aa6622/home/.envs/lerobot_mw_py312/bin/python" scripts/grpo/train_phase12_wm_chunk_grpo.py \
  --mode rollout_validation \
  --jepa-repo /rds/general/user/aa6622/home/research/RESEARCH_PAPER_CLONES/jepa-wms \
  --jepa-ckpt /rds/general/user/aa6622/home/.cache/huggingface/hub/models--facebook--jepa-wms/snapshots/9b9c41ef249466630dbf1a20e78391865d07b3b9/jepa_wm_metaworld.pth.tar \
  --output-dir artifacts/phase12_wm_crop_ablation/strict_jepa224 \
  --task push-v3 \
  --num-episodes 1 \
  --num-updates 1 \
  --max-steps 50 \
  --chunk-len 25 \
  --group-size 4 \
  --wm-frame-source jepa_render_224 \
  --wm-visual-resize-mode passthrough \
  --decode-candidates selected
```

Expected:
- Exit code `0`.
- `pixel_contract.json` says `wm_frame_shape` is `[224, 224, 3]`.
- Manifest says `phase12_wm_frame_contract = jepa_metaworld_corner2_vflip_224`.
- Selected decode strip exists.

- [ ] **Step 3: Compare artifacts**

Run:

```bash
"/rds/general/user/aa6622/home/.envs/lerobot_mw_py312/bin/python" - <<'PY'
import json
from pathlib import Path

roots = [
    Path("artifacts/phase12_wm_crop_ablation/control_legacy256"),
    Path("artifacts/phase12_wm_crop_ablation/strict_jepa224"),
]
for root in roots:
    manifest = json.loads((root / "smoke_manifest.json").read_text())
    contract = json.loads(
        next(root.glob("rollouts/update_0000_episode_0000/pixel_contract/pixel_contract.json")).read_text()
    )
    strip = next(root.glob("rollouts/update_0000_episode_0000/segment_0000/wm_real_vs_pred_selected_strip.png"))
    print(root.name)
    print("  wm_frame_source:", manifest.get("phase12_wm_frame_source"))
    print("  wm_visual_resize_mode:", manifest.get("wm_visual_resize_mode"))
    print("  wm_frame_shape:", contract.get("wm_frame_shape"))
    print("  decode_strip:", strip)
PY
```

Expected:
- Control reports `lerobot_unflip`, `legacy_256`, `[480, 480, 3]`.
- Strict reports `jepa_render_224`, `passthrough`, `[224, 224, 3]`.

- [ ] **Step 4: Commit smoke notes if repo tracks run notes**

Only if project convention wants run notes checked in:

```bash
git add docs/superpowers/plans/2026-05-16-phase12-strict-jepa-wm-render.md
git commit -m "docs: plan strict Phase12 JEPA render"
```

---

## Self-Review

- Spec coverage: plan keeps Phase12 LeRobot-compatible, uses same underlying MetaWorld env, avoids second physics env, isolates renderer mutation, routes WM/goal/decode to strict JEPA render, keeps policy/eval pixels unchanged, and includes smoke comparison.
- Placeholder scan: no `TBD`, no unspecified error handling, no "similar to" implementation steps.
- Type consistency: `wm_frame_source`, `wm_render_img_size`, and `wm_visual_resize_mode` names are consistent across CLI, manifest, trainer routing, reward scoring, and diagnostics.
- Risk left: actual MetaWorld `render(width=..., height=...)` behavior may vary by version. Plan handles this with temporary `MujocoRenderer` fallback and strict restore in `finally`.
