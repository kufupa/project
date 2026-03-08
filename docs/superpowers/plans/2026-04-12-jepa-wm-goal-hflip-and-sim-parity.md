# JEPA-WM goal H-flip + sim camera parity — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** (1) Apply a **horizontal flip** to oracle-derived goal RGB before WM `encode` for goal latent, undoing the extra horizontal mirror vs jepa-wms-style pixels. (2) Bring segment GRPO **live sim** rendering in line with **facebookresearch/jepa-wms** `MetaWorldWrapper` (corner2, `cam_pos`, square buffer, **vertical flip only** on render). **SmolVLA paths are explicitly out of scope** for this plan (no `evaluator.py` flip changes).

**Architecture:** One small module `metaworld_jepa_render` mirrors jepa-wms env init + `render()` contract. `segment_grpo_loop.rollout_with_chunks` uses it for `carry_mode='sim'`. Goal latent uses a narrow transform in `_load_goal_latent` (or adjacent helper) so existing phase06 PNGs stay on disk unchanged. Optional CLI flag disables goal H-flip for A/B.

**Tech stack:** Python 3, `metaworld`, `gymnasium` (`MujocoRenderer`), `numpy`, `torch`, existing `segment_grpo_loop` / `run_segment_grpo.py`.

---

## Review: `/homes/aa6622/.cursor/plans/jepa-wms_camera_parity_e37a2bd8.plan.md`

Paste into PR / issue as caveman-review:

- `L48-50`: 🔴 **risk:** “observation for policy + WM = rendered” — with SmolVLA **deferred**, changing sim pixels **still** feeds SmolVLA in `rollout_with_chunks` today; policy may degrade until SmolVLA parity pass. **Fix:** document in README + log one-line warning, or gate jepa sim behind `--wm-sim-parity` until policy re-aligned.
- `L50-51`: 🟡 **risk:** “oracle `_render_rgb_frame` V-only + regen phase06” **vs** goal-only H-flip ingest — both valid; doing **only** goal H-flip avoids regen; doing **only** oracle V-only avoids runtime special-case. **Fix:** this implementation plan uses **goal H-flip at WM** + **jepa sim for live**; leave oracle script **unchanged** unless you want artifact consistency (optional Task 6).
- `L40-42`: ❓ **q:** jepa configs use `img_size: 224`; this repo `_to_wm_visual` resizes to **256×256** — confirm checkpoint/preproc still OK. **Fix:** add one-line comment + optional `WM_VISUAL_SIZE` env if you need 224 later.
- `todos / smolvla-flip`: 🔵 **nit:** mark **cancelled / out of scope** per stakeholder (SmolVLA later).
- **Good:** clear `MetaWorldWrapper` reference, mermaid, concrete file paths.

---

## File map (what changes)

| File | Responsibility |
|------|----------------|
| **Create** [`project/src/metaworld_jepa_render.py`](../../../src/metaworld_jepa_render.py) | Build MetaWorld MT1 env + jepa-wms-style renderer (`corner2`, `cam_pos`, `MujocoRenderer`, `render()[::-1]`). |
| **Modify** [`project/src/segment_grpo_loop.py`](../../../src/segment_grpo_loop.py) | Goal: H-flip before WM encode from `goal_frame`. Sim: use `metaworld_jepa_render` in `rollout_with_chunks` when `carry_mode='sim'`. |
| **Modify** [`project/scripts/run_segment_grpo.py`](../../../scripts/run_segment_grpo.py) | Add `--wm-goal-hflip` (default `true`) or `--no-wm-goal-hflip`; pass through to `rollout_with_chunks`. |
| **Modify** [`project/docs/wm_versus_smolvla_versus_environment_camera.md`](../../wm_versus_smolvla_versus_environment_camera.md) | Document goal H-flip at WM ingest + new sim module. |
| **Create** [`project/tests/test_metaworld_jepa_render.py`](../../../tests/test_metaworld_jepa_render.py) | Smoke / shape tests (skip if `metaworld` missing). |
| **Create** [`project/tests/test_wm_goal_hflip.py`](../../../tests/test_wm_goal_hflip.py) | Unit test: flipped array `np.flip(x,1)` reverses width; encode path mocked or optional cuda. |

**Reference only (do not edit in this plan):** `VGG JEPA/jepa-wms/evals/simu_env_planning/envs/metaworld.py` (workspace copy) — source of truth for `MetaWorldWrapper`.

---

## Why horizontal flip on oracle goal fixes WM (math)

Oracle stores `np.flip(raw, (0, 1))` = vertical **and** horizontal. jepa-wms uses `raw` then `[::-1]` = **vertical only** of raw buffer.

For HWC image: `oracle = V(H(raw))`. Then `H(oracle) = H(V(H(raw)))` — flips along axes 0 and 1 commute, so `H(oracle) = V(H(H(raw))) = V(raw)` in terms of “undo H on stored oracle → leaves V-only of raw,” matching the jepa-wms **single** vertical flip convention relative to the same MuJoCo buffer orientation.

---

### Task 1: WM goal horizontal flip (oracle PNG path)

**Files:**

- Modify: [`project/src/segment_grpo_loop.py`](../../../src/segment_grpo_loop.py) (`_load_goal_latent`, `rollout_with_chunks` signature + call site for flag)
- Modify: [`project/scripts/run_segment_grpo.py`](../../../scripts/run_segment_grpo.py)
- Create: [`project/tests/test_wm_goal_hflip.py`](../../../tests/test_wm_goal_hflip.py)

- [ ] **Step 1: Add helper** near `_to_wm_visual` in `segment_grpo_loop.py`:

```python
def _prepare_goal_image_for_wm(
    goal_rgb: np.ndarray, *, flip_horizontal: bool
) -> np.ndarray:
    """Oracle PNGs use V+H vs jepa-wms V-only; optional H-flip aligns goal pixels for WM encode."""
    rgb = _to_rgb_uint8(goal_rgb)
    if not flip_horizontal:
        return rgb
    return np.ascontiguousarray(np.flip(rgb, axis=1))
```

- [ ] **Step 2: Thread flag** — add `wm_goal_flip_horizontal: bool = True` to `rollout_with_chunks`; pass into `_load_goal_latent` (add same param). Inside `_load_goal_latent`, when `goal_frame is not None` and about to call `_encode_state_to_latent`, use `img = _prepare_goal_image_for_wm(goal_frame, flip_horizontal=wm_goal_flip_horizontal)` instead of raw `goal_frame`.

- [ ] **Step 3: CLI** — in `run_segment_grpo.py` add `--no-wm-goal-hflip` (store_false pattern) or `--wm-goal-hflip` default true; pass to `rollout_with_chunks`. Log once per episode: `wm_goal_hflip=true|false`.

- [ ] **Step 4: Unit test** — `test_wm_goal_hflip.py`:

```python
import numpy as np

def test_prepare_goal_hflip_reverses_width():
    from segment_grpo_loop import _prepare_goal_image_for_wm
    x = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    y = _prepare_goal_image_for_wm(x, flip_horizontal=True)
    assert np.array_equal(y[:, 0, :], x[:, 1, :]) and np.array_equal(y[:, 1, :], x[:, 0, :])
```

Run: `pytest project/tests/test_wm_goal_hflip.py -v`  
Expected: PASS (adjust import path if package layout uses `src` on `PYTHONPATH`).

- [ ] **Step 5: Commit** — `feat(wm): optional H-flip on oracle goal before WM goal latent`

---

### Task 2: `metaworld_jepa_render` module (jepa-wms parity)

**Files:**

- Create: [`project/src/metaworld_jepa_render.py`](../../../src/metaworld_jepa_render.py)
- Modify: [`project/src/segment_grpo_loop.py`](../../../src/segment_grpo_loop.py)

- [ ] **Step 1: Implement** `build_jepa_metaworld_env(task: str, *, img_size: int = 224, seed: int | None = None)`:

  - `import metaworld` → `MT1(task)` → `train_classes[task]` → `env_cls(render_mode="rgb_array", camera_name="corner2")` with fallback `env_cls()` + set `render_mode`.
  - `env.model.cam_pos[2] = [0.75, 0.075, 0.7]` (same as jepa-wms).
  - `env.camera_name = "corner2"`, `env.width = env.height = img_size`.
  - `init_renderer()` copy from jepa-wms: `MujocoRenderer(model, data, env.mujoco_renderer.default_cam_config, width=img_size, height=img_size, max_geom=..., camera_id=None, camera_name="corner2")`.
  - If `seed is not None`, caller will `reset(seed=seed)` separately.

- [ ] **Step 2: Implement** `render_jepa_rgb(env) -> np.ndarray`:

```python
def render_jepa_rgb(env) -> np.ndarray:
    arr = np.asarray(env.render().copy())[::-1]  # V-flip only, match MetaWorldWrapper.render
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]
    return np.ascontiguousarray(arr)
```

- [ ] **Step 3: Export** `__all__` or keep minimal; document module docstring: “Parity with jepa-wms `evals/simu_env_planning/envs/metaworld.py` `MetaWorldWrapper`.”

- [ ] **Step 4: Commit** — `feat(metaworld): add jepa-wms-style render helper`

---

### Task 3: Wire segment GRPO sim branch

**Files:**

- Modify: [`project/src/segment_grpo_loop.py`](../../../src/segment_grpo_loop.py)

- [ ] **Step 1:** In `rollout_with_chunks`, when `carry_mode == "sim"` and not `dry_run`, replace manual `env_cls(render_mode="rgb_array")` block with:

  - `from metaworld_jepa_render import build_jepa_metaworld_env, render_jepa_rgb`
  - `env = build_jepa_metaworld_env(task, img_size=224, seed=None)` then `set_task` from `train_tasks[0]` or same logic as today.
  - After `reset`, set `current_image = render_jepa_rgb(env)` (do not rely on obs image if it differs).
  - After each `env.step`, set image from `render_jepa_rgb(env)` instead of `_extract_image_and_proprio` for **visual** (still need proprio from obs — keep `_extract_image_and_proprio` for proprio only or parse obs dict).

- [ ] **Step 2:** If obs has no separate proprio path, keep `_step_env` but swap image: after step, `_, proprio, info, done = ...` then `image = render_jepa_rgb(env)` — refactor `_step_env` or add `_step_env_jepa` to avoid duplication.

- [ ] **Step 3:** Add optional `metaworld_img_size: int = 224` to `rollout_with_chunks` + CLI `--metaworld-jepa-img-size` default 224 for future 256 experiments.

- [ ] **Step 4:** Run one dry smoke: `python project/scripts/run_segment_grpo.py --dry-run ...` unchanged; real run requires GPU/metaworld — manual or CI with deps.

- [ ] **Step 5: Commit** — `feat(segment_grpo): jepa-wms camera parity for sim carry mode`

---

### Task 4: Tests for `metaworld_jepa_render`

**Files:**

- Create: [`project/tests/test_metaworld_jepa_render.py`](../../../tests/test_metaworld_jepa_render.py)

- [ ] **Step 1:** `pytest.importorskip("metaworld")` at top of test file.

- [ ] **Step 2:** Test `build_jepa_metaworld_env("push-v3")` + `reset` + `render_jepa_rgb` returns `uint8` or float RGB, `ndim==3`, `shape[2]==3`, height==width==224.

- [ ] **Step 3:** Run `pytest project/tests/test_metaworld_jepa_render.py -v` — skip on CI without metaworld.

- [ ] **Step 4: Commit** — `test(metaworld): smoke jepa-wms render helper`

---

### Task 5: Documentation

**Files:**

- Modify: [`project/docs/wm_versus_smolvla_versus_environment_camera.md`](../../wm_versus_smolvla_versus_environment_camera.md)

- [ ] **Step 1:** Add subsection **“WM goal latent from oracle PNG”**: one sentence — H-flip before `encode` aligns stored oracle (V+H) with jepa-wms (V-only on raw).

- [ ] **Step 2:** Add subsection **“Segment GRPO sim (post-change)”** — points to `metaworld_jepa_render` + `img_size`.

- [ ] **Step 3: Commit** — `docs: WM goal hflip + jepa sim parity`

---

### Task 6 (optional): Oracle script V-only + regen phase06

**Files:**

- Modify: [`project/scripts/oracle/run_metaworld_oracle_eval.py`](../../../scripts/oracle/run_metaworld_oracle_eval.py)

- [ ] **Step 1:** Replace `np.flip(frame_np, (0, 1))` with `frame_np = np.asarray(frame_np)[::-1]` when `flip_corner2` (or delegate to `render_jepa_rgb`).

- [ ] **Step 2:** Re-run oracle campaign; invalidate old goal paths in docs.

**Skip** if goal H-flip + jepa sim are enough for experiments.

---

## Self-review

1. **Spec coverage:** Goal H-flip (user 1) → Task 1. jepa-wms sim parity (user 2) → Tasks 2–3. Cursor plan review → top section. SmolVLA ignored → no evaluator edits.
2. **Placeholders:** None; code snippets concrete.
3. **Consistency:** `img_size` 224 in new module vs `_to_wm_visual` 256 — called out in review; Task 3 allows CLI override for experiments.

---

## Execution handoff

**Plan complete and saved to** `project/docs/superpowers/plans/2026-04-12-jepa-wm-goal-hflip-and-sim-parity.md`.

**Execution options:**

1. **Subagent-driven (recommended)** — fresh subagent per task, review between tasks. **REQUIRED SUB-SKILL:** `superpowers:subagent-driven-development`.
2. **Inline execution** — same session, checkpoints. **REQUIRED SUB-SKILL:** `superpowers:executing-plans`.

**Which approach?**
