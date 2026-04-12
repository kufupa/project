# WM camera parity — single implementation handoff plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align **rollout RGB** (SmolVLA chunk sampling, **JEPA-WM** encode/score, comparison-strip “real” frames, goal-latent fallbacks) with **facebookresearch/jepa-wms** MetaWorld rendering semantics. Fix known bugs (wrong `train_tasks` index, goal/oracle vs WM flip mismatch). Does **not** change [`evaluator.py`](../../../src/smolvla_pipeline/evaluator.py) offline SmolVLA eval paths in this pass.

**Architecture:** (1) New module `metaworld_jepa_render.py` mirrors jepa-wms `MetaWorldWrapper` camera + V-flip render. (2) In `rollout_with_chunks` sim mode with parity on, **one** RGB buffer from `render_jepa_rgb(env)` — call it **`wm_image`** in code — feeds `_sample_smolvla_chunk`, WM paths, and strip frames (no separate legacy policy camera). (3) Before any WM `encode` from oracle-style goal pixels, apply **`np.flip(rgb, axis=1)`** on uint8 HWC so stored oracle PNGs (V+H) match jepa-wms effective view (V-only on raw buffer). (4) **Debug artifact:** copy that same prepared uint8 HWC goal (output of `_prepare_goal_image_for_wm`, i.e. what enters WM preprocess before resize/float) to a PNG under the run’s episode outputs so you can eyeball it against live `wm_image` / jepa-wms expectations. (5) Thread CLI flags from `run_segment_grpo.py`. (6) Tests + Slurm one-episode regeneration.

**Tech stack:** Python 3.10+, `numpy`, `torch`, `metaworld` (MT1), `gymnasium` (`gymnasium.envs.mujoco.mujoco_rendering.MujocoRenderer`), existing `segment_grpo_loop` WM bundle. Repo root: `/vol/bitbucket/aa6622/project` (adjust if forked).

---

## 0) Reader has zero context — read this first

### What this codebase is doing

- **Segment GRPO** ([`project/scripts/run_segment_grpo.py`](../../../scripts/run_segment_grpo.py)) runs episodes: SmolVLA proposes action chunks, **JEPA-WM** scores chunks by latent distance to a **goal**, best chunk is executed in MetaWorld **sim** (or replay).
- **Goal image** often comes from **pre-recorded oracle PNGs** under `artifacts/phase06_oracle_baseline/run_*/frames/episode_XXXX/frame_*.png` (see [`project/src/segment_grpo_reference.py`](../../../src/segment_grpo_reference.py) `load_oracle_reference_frames`). Default CLI `--goal-frame-index` is **25** (1-based) → file `frame_000024.png`.
- **WM** path: images go through `_to_wm_visual` in [`project/src/segment_grpo_loop.py`](../../../src/segment_grpo_loop.py) (resize to 256×256, float RGB) then `bundle.model.encode`.

### What is wrong today (must fix)

1. **Sim camera:** `rollout_with_chunks` builds `env_cls(render_mode="rgb_array")` **without** `corner2`, **without** `cam_pos[2]` patch, **without** jepa-wms V-flip — see ```1711:1730:project/src/segment_grpo_loop.py```. WM sees a different view than jepa-wms training/eval.
2. **Oracle vs WM flip:** Oracle script saves `np.flip(frame, (0, 1))` (V+H). jepa-wms `MetaWorldWrapper.render` uses **`env.render().copy()[::-1]`** (V-only). Same raw buffer → oracle PNG has an **extra horizontal mirror** vs WM contract.
3. **Math fix for goals:** If `oracle = flip_V(flip_H(raw))` in HWC, then **`flip_H(oracle)`** equals **`flip_V(raw)`**, i.e. matches jepa-wms post-render convention. Implement as `np.flip(rgb, axis=1)` on uint8 HWC **only** for WM goal encode paths.
4. **Task index bug:** Oracle uses `tasks[episode_index % len(tasks)]` ([`project/scripts/oracle/run_metaworld_oracle_eval.py`](../../../scripts/oracle/run_metaworld_oracle_eval.py) ~L208). Segment GRPO uses **`tasks[0]`** always — ```1727:1729:project/src/segment_grpo_loop.py```. Wrong for `episode_index > 0` vs phase06 alignment.
5. **Start-frame warning:** `start_frame` from oracle is V+H; with jepa-parity live RGB (`wm_image`), reset compare will false-alarm unless you **H-flip `start_frame`** before `_frame_similarity` when goal H-flip is on, or disable that warning in WM-parity mode.

### What is explicitly out of scope

- Do **not** edit [`project/src/smolvla_pipeline/evaluator.py`](../../../src/smolvla_pipeline/evaluator.py) SmolVLA flip/camera in this pass.
- Do **not** regenerate entire phase06 oracle corpus (optional follow-up).

### Reference implementation (upstream)

Copy semantics from workspace checkout (or GitHub `facebookresearch/jepa-wms`):

```32:58:VGG JEPA/jepa-wms/evals/simu_env_planning/envs/metaworld.py
        self.camera_name = "corner2"
        self.env.model.cam_pos[2] = [0.75, 0.075, 0.7]
        self.env.render_mode = "rgb_array"
        self.env.camera_name = self.camera_name
        self.env.width = cfg.task_specification.img_size
        self.env.height = cfg.task_specification.img_size
        ...
        self.env.mujoco_renderer = MujocoRenderer(
            self.env.model,
            self.env.data,
            self.env.mujoco_renderer.default_cam_config,
            width=self.env.width,
            height=self.env.height,
            max_geom=self.env.mujoco_renderer.max_geom,
            camera_id=None,
            camera_name=self.env.camera_name,
        )
```

```117:125:VGG JEPA/jepa-wms/evals/simu_env_planning/envs/metaworld.py
    def render(self, *args, **kwargs):
        result = self.env.render().copy()[::-1]  # flip vertically
        ...
        return result  # H W 3
```

**Note:** Upstream file imports `gym`; this project uses **gymnasium** MetaWorld — use `from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer`.

### Key functions to touch

| Location | Role |
|----------|------|
| `_load_goal_latent` | ```1504:1548:project/src/segment_grpo_loop.py``` — add goal image prep before `_encode_state_to_latent` for both `goal_frame` and dict `"image"` branch; optional write `wm_goal_for_encode.png` |
| `rollout_with_chunks` | ```1650:1937:project/src/segment_grpo_loop.py``` — sim env build, single `wm_image` from `render_jepa_rgb` for policy + WM + strip, task modulo; pass `comparison_root`-style path so goal debug PNG lands with episode artifacts |
| `_parse_args` + `main` | [`project/scripts/run_segment_grpo.py`](../../../scripts/run_segment_grpo.py) — new flags, pass to `rollout_with_chunks` |
| `run_first_episode_real_pipeline.sh` | [`project/scripts/segment_grpo/run_first_episode_real_pipeline.sh`](../../../scripts/segment_grpo/run_first_episode_real_pipeline.sh) — append new CLI args to the `exec ... run_segment_grpo.py` line so Slurm job picks up defaults |
| Docs | [`project/docs/wm_versus_smolvla_versus_environment_camera.md`](../../wm_versus_smolvla_versus_environment_camera.md) — document single jepa-parity RGB + goal H-flip + `wm_goal_for_encode.png` debug output |

### Tests layout

Existing tests insert `project/src` on `sys.path` — see ```10:16:project/tests/test_run_segment_grpo_main.py```. New tests should do the same.

### Slurm one-episode regeneration

After merge, queue:

- `cd /vol/bitbucket/aa6622/project && sbatch scripts/segment_grpo/submit_segment_grpo_first_episode_real.slurm`
- Logs: `/vol/bitbucket/aa6622/project/logs/seggrpo_first_ep_real_<JOBID>.out` and `.err`
- Expect new `artifacts/phase08_segment_grpo_baseline/run_*`, refreshed `segment_grpo_first_episode_real_artifacts/comparison/episode_0000_comparison_strip.png`, and **`.../comparison/episode_XXXX/wm_goal_for_encode.png`** (uint8 HWC goal **after** `axis=1` flip when enabled — same buffer WM encode path uses before `_to_wm_visual`)

Nested `sbatch` QOS issues: see [`project/docs/slurm/sbatch-notes.md`](../../slurm/sbatch-notes.md).

---

## 1) File map

| Action | Path |
|--------|------|
| **Create** | [`project/src/metaworld_jepa_render.py`](../../../src/metaworld_jepa_render.py) |
| **Modify** | [`project/src/segment_grpo_loop.py`](../../../src/segment_grpo_loop.py) |
| **Modify** | [`project/scripts/run_segment_grpo.py`](../../../scripts/run_segment_grpo.py) |
| **Modify** | [`project/scripts/segment_grpo/run_first_episode_real_pipeline.sh`](../../../scripts/segment_grpo/run_first_episode_real_pipeline.sh) |
| **Create** | [`project/tests/test_wm_goal_hflip.py`](../../../tests/test_wm_goal_hflip.py) |
| **Create** | [`project/tests/test_metaworld_jepa_render.py`](../../../tests/test_metaworld_jepa_render.py) |
| **Modify** | [`project/tests/test_run_segment_grpo_main.py`](../../../tests/test_run_segment_grpo_main.py) or new test file for rollout wiring |
| **Modify** | [`project/docs/wm_versus_smolvla_versus_environment_camera.md`](../../wm_versus_smolvla_versus_environment_camera.md) |

---

## 2) Implementation tasks (checkboxes)

### Task A: `metaworld_jepa_render.py`

**Files:** Create [`project/src/metaworld_jepa_render.py`](../../../src/metaworld_jepa_render.py)

- [ ] **A1.** Implement `build_jepa_metaworld_env(task: str, *, img_size: int, seed: int | None) -> Any`:

  - `import metaworld` → `MT1(task)` → `env_cls = mt1.train_classes[task]`.
  - Try `env_cls(render_mode="rgb_array", camera_name="corner2")`, fallback `env_cls()` + set `render_mode`.
  - `os.environ.setdefault("MUJOCO_GL", "egl")` (match `segment_grpo_loop`).
  - After env exists: `env.model.cam_pos[2] = [0.75, 0.075, 0.7]`, `env.camera_name = "corner2"`, `env.width = env.height = img_size`.
  - **Renderer init:** if `getattr(env, "mujoco_renderer", None)` is missing, call `env.render()` once then retry; else `MujocoRenderer(env.model, env.data, env.mujoco_renderer.default_cam_config, width=img_size, height=img_size, max_geom=env.mujoco_renderer.max_geom, camera_id=None, camera_name="corner2")`. Wrap in try/except; on failure log and re-raise with hint.
  - Return `env` (caller sets task + reset).

- [ ] **A2.** Implement `render_jepa_rgb(env) -> np.ndarray`:

```python
def render_jepa_rgb(env: Any) -> np.ndarray:
    arr = np.asarray(env.render().copy())[::-1]  # V-flip, match jepa-wms MetaWorldWrapper.render
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating) and float(np.max(arr)) <= 1.5:
            arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)
```

- [ ] **A3.** Module docstring: cite jepa-wms path + note parity sim uses this RGB for policy + WM + strip (one contract).

- [ ] **A4.** Commit: `feat(metaworld): jepa-wms parity render helper`

---

### Task B: Goal image H-flip for WM only

**Files:** Modify [`project/src/segment_grpo_loop.py`](../../../src/segment_grpo_loop.py)

- [ ] **B1.** Add helper next to `_to_wm_visual`:

```python
def _prepare_goal_image_for_wm(image: Any, *, flip_horizontal: bool) -> np.ndarray:
    rgb = _to_rgb_uint8(image)
    if not flip_horizontal:
        return rgb
    return np.ascontiguousarray(np.flip(rgb, axis=1))
```

- [ ] **B2.** Extend `_load_goal_latent(..., wm_goal_flip_horizontal: bool = True)`:

  - Before `_encode_state_to_latent(wm_bundle, img, ...)` in dict branch (lines ~1533–1536), set `img = _prepare_goal_image_for_wm(img, flip_horizontal=wm_goal_flip_horizontal)`.
  - Before `_encode_state_to_latent(wm_bundle, goal_frame, ...)` (lines ~1538–1544), use prepared image same way.

- [ ] **B3.** Extend `rollout_with_chunks` signature with `wm_goal_flip_horizontal: bool = True`; pass into `_load_goal_latent`.

- [ ] **B4.** **Start-frame warning:** where `start_frame` is compared to live RGB (~1751–1759), if `wm_goal_flip_horizontal` and parity sim uses `wm_image`, H-flip `start_frame` with `np.flip(start_frame, 1)` before `_frame_similarity` vs `wm_image` (document in comment).

- [ ] **B5.** Add `wm_goal_flip_horizontal` (and other new flags) to `episode_log.metadata` for traceability. Add **`wm_goal_for_encode_path`** (string or `null`) when the debug PNG is written.

- [ ] **B6.** **Save prepared goal for visual check:** After `_prepare_goal_image_for_wm` (uint8 HWC, channel-last), **copy** that array to disk as PNG when episode comparison artifacts are enabled (reuse same condition / root as comparison strips — today `comparison_root` → per-episode dirs under `.../comparison/episode_{idx:04d}/`). Suggested filename: **`wm_goal_for_encode.png`**. Must match exactly what goes into `_encode_state_to_latent` preprocessing (if flip off, file shows unflipped oracle crop). Use existing image write deps in tree (`PIL`, `imageio`, etc. — match comparison-strip writers). If no `comparison_root`, skip write (or gate behind a `--save-wm-goal-debug` flag only if you need runs without strips — default: tie to comparison root).

- [ ] **B7.** Commit: `feat(wm): H-flip oracle goal for WM encode + wm_goal_for_encode debug PNG`

---

### Task C: Single jepa-parity `wm_image` in sim rollout

**Files:** Modify [`project/src/segment_grpo_loop.py`](../../../src/segment_grpo_loop.py)

- [ ] **C1.** Add kwargs to `rollout_with_chunks`: `wm_sim_camera_parity: bool = True`, `wm_sim_img_size: int = 224`.

- [ ] **C2.** Sim branch (`carry_mode != "replay"`, not `dry_run`):

  - If `wm_sim_camera_parity`: `env = build_jepa_metaworld_env(task, img_size=wm_sim_img_size, seed=seed)` from `metaworld_jepa_render`.
  - **Task:** `tasks = getattr(mt1, "train_tasks", None) or []` — if non-empty: `env.set_task(tasks[int(episode_index) % len(tasks)])` (replace `tasks[0]`).
  - Else (parity off): keep **legacy** env construction for backward compat (existing single `current_image` path).

- [ ] **C3.** After `reset` / each step when parity on:

  - Use `_reset_env` / `_step_env` for **proprio** (and any non-RGB side effects) as today.
  - Set **`wm_image = render_jepa_rgb(env)`** after reset and after each step (ensure first post-reset frame covered — order vs `render()` side effects as needed).
  - Treat **`wm_image` as the only rollout RGB**: pass it into `_sample_smolvla_chunk`, WM scoring, goal fallback, strip — do **not** keep a second legacy camera image for the policy.

  When parity **off**, `wm_image` (or equivalent loop variable) equals existing `current_image` from legacy path.

- [ ] **C4.** Wire **`wm_image`** everywhere rollout previously used `current_image` for visuals when parity on:

  - `_load_goal_latent(..., fallback_image=wm_image, ...)`.
  - `score_chunk_by_goal_latent(..., wm_image, ...)`.
  - `segment_real_frames`: first frame + per-step frames from **`wm_image`**.
  - `_sample_smolvla_chunk(..., wm_image, ...)`.

- [ ] **C5.** Loop: update **`wm_image`** each step; one RGB buffer for parity mode (no parallel `policy_image`).

- [ ] **C6.** Commit: `feat(segment_grpo): jepa WM camera parity + single RGB stream`

---

### Task D: CLI + pipeline script

**Files:** [`project/scripts/run_segment_grpo.py`](../../../scripts/run_segment_grpo.py), [`project/scripts/segment_grpo/run_first_episode_real_pipeline.sh`](../../../scripts/segment_grpo/run_first_episode_real_pipeline.sh)

- [ ] **D1.** Add argparse:

  - `--wm-goal-hflip` default True — use `argparse.BooleanOptionalAction` so `--no-wm-goal-hflip` works (Python 3.9+).
  - `--wm-sim-camera-parity` default True — same pattern, `--no-wm-sim-camera-parity` for A/B.
  - `--wm-sim-img-size` type=int default 224.

- [ ] **D2.** Pass all three into `rollout_with_chunks(...)`.

- [ ] **D3.** Log once per run: values of the three flags.

- [ ] **D4.** Update `run_first_episode_real_pipeline.sh` — append to the `exec ... run_segment_grpo.py \` block (before `"$@"`) explicit args if you want Slurm defaults to always use parity, e.g.:

```bash
  --wm-sim-camera-parity \
  --wm-sim-img-size 224 \
  --wm-goal-hflip \
```

  (Adjust if using `BooleanOptionalAction` — only pass `--no-*` when disabling.)

- [ ] **D5.** Commit: `chore(segment_grpo): CLI flags for WM camera parity`

---

### Task E: Tests

**Files:** Create [`project/tests/test_wm_goal_hflip.py`](../../../tests/test_wm_goal_hflip.py), [`project/tests/test_metaworld_jepa_render.py`](../../../tests/test_metaworld_jepa_render.py), extend rollout tests as needed.

- [ ] **E1.** `test_wm_goal_hflip.py`: insert `SRC_ROOT` on `sys.path` like `test_run_segment_grpo_main.py`. Test `_prepare_goal_image_for_wm` width swap (2×2×3 toy array).

- [ ] **E2.** `test_metaworld_jepa_render.py`: `pytest.importorskip("metaworld")`. Build env `push-v3`, `reset(seed=0)`, `img = render_jepa_rgb(env)` → assert `img.ndim == 3`, `img.shape[2] == 3`, `img.shape[0] == img.shape[1] == 224` (or `wm_sim_img_size` passed in).

- [ ] **E3.** Mock-heavy test (optional): monkeypatch `score_chunk_by_goal_latent` / `_sample_smolvla_chunk` to capture image args; when parity on, assert **same** ndarray identity (or equal pixels) for policy vs WM image arg — single `wm_image`.

- [ ] **E4.** Run:

```bash
cd /vol/bitbucket/aa6622/project
PYTHONPATH=src pytest tests/test_wm_goal_hflip.py tests/test_metaworld_jepa_render.py tests/test_run_segment_grpo_main.py -v
```

- [ ] **E5.** Commit: `test: WM goal flip + jepa render smoke`

---

### Task F: Documentation

**Files:** [`project/docs/wm_versus_smolvla_versus_environment_camera.md`](../../wm_versus_smolvla_versus_environment_camera.md)

- [ ] **F1.** Add section: single jepa-parity `wm_image` for policy + WM + strip, CLI flags, goal H-flip rationale, **`wm_goal_for_encode.png`** location and meaning (one paragraph + pointer to this plan).

- [ ] **F2.** Commit: `docs: WM camera parity handoff notes`

---

### Task G: Verification (local)

- [ ] **G1.** `pytest` on touched tests + full `tests/test_segment_grpo_loop.py` if exists.

- [ ] **G2.** One local non-dry run (if GPU available): `python scripts/run_segment_grpo.py ... --episodes 1 --carry-mode sim` — inspect comparison strip orientation vs previous artifact.

- [ ] **G3.** Commit only if fixes needed from G2.

---

### Task H: Slurm one-episode artifact regeneration

- [ ] **H1.** From repo root: `sbatch scripts/segment_grpo/submit_segment_grpo_first_episode_real.slurm`

- [ ] **H2.** Monitor `logs/seggrpo_first_ep_real_<JOBID>.out/.err`.

- [ ] **H3.** Confirm new `artifacts/phase08_segment_grpo_baseline/run_*` and `episode_0000_comparison_strip.png` timestamp updated; JSON metadata contains new fields; **`comparison/episode_0000/wm_goal_for_encode.png`** present when `comparison_root` used and goal loaded from oracle path.

---

## 3) Acceptance criteria

- [ ] With default flags, WM goal latent from phase06 PNG uses H-flip at encode; `--no-wm-goal-hflip` restores old behavior.
- [ ] When comparison artifacts enabled, **`wm_goal_for_encode.png`** written per episode: uint8 HWC identical to post-`_prepare_goal_image_for_wm` input to WM goal encode (eyeball vs first `wm_image` / jepa contract).
- [ ] With `wm_sim_camera_parity` on, SmolVLA chunk sampling + WM scoring + strip real frames all use the same jepa-parity RGB (`render_jepa_rgb`).
- [ ] `env.set_task(tasks[episode_index % len(tasks)])` when tasks exist.
- [ ] Start-frame warning not systematically false-positive after change (or explicitly disabled with comment).
- [ ] Tests pass; Slurm job completes; new strip generated.

---

## 4) Risks (short)

| Risk | Mitigation |
|------|------------|
| `MujocoRenderer` / metaworld version skew | try/except + warm `render()`; clear error message |
| 224 render vs `_to_wm_visual` 256 | configurable `--wm-sim-img-size`; document |
| SmolVLA train/eval distribution vs jepa-parity sim pixels | parity mode changes what policy sees in sim vs old runs; document; `evaluator.py` unchanged this pass |

---

## 5) Self-review (plan author)

- Spec coverage: goal flip, goal debug PNG, sim parity, task index, start frame, CLI, tests, Slurm — all in tasks.
- No TBD/TODO placeholders in task bodies.
- Types: `wm_goal_flip_horizontal: bool` threaded consistently.

---

## 6) Execution handoff

**Plan complete:** [`project/docs/superpowers/plans/2026-04-13-wm-camera-parity-single-handoff.md`](2026-04-13-wm-camera-parity-single-handoff.md)

**Options:**

1. **Subagent-driven** — `superpowers:subagent-driven-development` (one subagent per task A–H).
2. **Inline** — `superpowers:executing-plans` in one session with checkpoints.

**Canonical plan = this file only.** Older fragments (`2026-04-12-jepa-wm-goal-hflip-and-sim-parity.md`, `.cursor/plans/wm-camera-parity-final_*.plan.md`) are superseded for implementation — keep only for history or delete manually.
