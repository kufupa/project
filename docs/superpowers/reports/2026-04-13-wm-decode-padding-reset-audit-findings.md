# WM Decode-Padding-Reset Pipeline Audit — Findings Report

**Date:** 2026-04-13  
**Audit plan:** `~/.cursor/plans/wm-decode-padding-reset-audit_e47ab801.plan.md`  
**Repo root:** `/vol/bitbucket/aa6622/project`  
**Scope:** Seven-task static audit (pipeline map, pack/decode alignment, reset/seed, metadata, upstream JEPA-WM YAML).  
**Status at time of audit:** 34 tests pass, 1 pre-existing FAIL (see BUG F6)

---

## 0. Zero-context primer — read this first

### What this codebase does

`scripts/run_segment_grpo.py` runs a segment-GRPO loop: SmolVLA proposes action chunks, a JEPA world model (WM) scores each chunk by predicting the resulting latent state and measuring distance to a goal latent, the best chunk is executed in the MetaWorld sim (or replayed), and optionally the WM's decoded predictions are compared side-by-side against real sim frames in PNG strips.

### Why this audit existed

The main concern was: **are decoded WM frames correctly aligned to sim frames?** The WM operates at a coarser timescale than the sim — one WM step consumes 5 consecutive 4-D env actions packed into a single 20-D vector. A visualization that treats "one WM step = one env step" would be wrong by a factor of 5.

### Vocabulary

| Term | Meaning |
|------|---------|
| `env_action_dim` | Width of `env.step` vector (4 for MetaWorld MT1) |
| `wm_dim` / `planner_action_dim` | WM `act_suffix` last dim (20 for MetaWorld hub) |
| `factor` | `wm_dim // env_dim` = 5. One WM step = 5 env steps |
| `T` | Number of env actions in a candidate chunk (`--chunk-len`) |
| `T_wm` | `ceil(T / factor)` — number of WM unroll steps |
| `carried_steps` | Env steps actually executed for the selected chunk |
| `segment_real_frames` | `[pre-execution frame] + [one frame per executed step]`; length = `carried_steps + 1` |
| `DecodeTrace` | Stores per-WM-step latents for decode; carries `env_steps_per_wm_step = factor` |
| comparison strip | PNG: real sim frame vs WM-decoded pred, side-by-side per column |
| parity mode | `--wm-sim-camera-parity` (default on): live sim renders via `render_jepa_rgb` (corner2 + V-flip) to match WM training camera |

---

## 1. Architecture summary

### One-segment control flow inside `rollout_with_chunks`

```
current_image, current_proprio (segment start state, pre-execution)
    │
    ├─ For each candidate chunk:
    │       _sample_smolvla_chunk  →  chunk (T, env_dim)
    │       score_chunk_by_goal_latent:
    │           _ensure_action_matrix      →  (T, env_dim)
    │           _normalize_env_actions_for_wm  →  (T, env_dim) normalized
    │           _pack_env_actions_for_wm   →  (T_wm, wm_dim)  [pads T to multiple of factor]
    │           unsqueeze                  →  actions_t (T_wm, 1, wm_dim)
    │           _to_wm_visual(image)       →  (1,1,3,256,256) float [0,255]
    │           model.encode(visual, proprio)  →  latent_state
    │           model.unroll(latent, act_suffix=actions_t)  →  latents per WM step
    │           extract scoring latent from final step
    │           L2 distance to goal_latent  →  score
    │           DecodeTrace: visual/proprio latents per WM step
    │
    ├─ Pick best candidate by score
    │
    ├─ Execute best_actions in sim (or replay) for effective_len steps (or until done):
    │       segment_real_frames.append(current_image) after each step
    │       carried_steps += 1 per step
    │
    ├─ _decode_latent_trace_to_frames(wm_bundle, selected_trace)
    │       → list of T_wm HWC uint8 frames (pred_frames)
    │
    └─ _write_comparison_segment_strip(
             segment_real_frames, pred_frames, carried_steps,
             env_steps_per_wm_step=factor)
             → comparison_strip.png
```

### Key tensor shapes

| Step | Shape | Notes |
|------|-------|-------|
| raw chunk | `(T, env_dim)` | env_dim=4 for MetaWorld |
| normalized | `(T, env_dim)` | preprocessor mean/std on device |
| packed | `(T_wm, wm_dim)` | T_wm = ceil(T/factor); wm_dim=20 |
| act_suffix to unroll | `(T_wm, 1, wm_dim)` | batch dim=1 |
| visual to encode | `(1, 1, 3, 256, 256)` | float [0,255]; hub divides by 255 internally |
| pred_frames from decode | `list[T_wm]` of HWC uint8 | one decoded frame per WM step |

### Files

| Path | Role |
|------|------|
| `scripts/run_segment_grpo.py` | CLI, episode loop, oracle loading, artifact dirs, calls `rollout_with_chunks` |
| `src/segment_grpo_loop.py` | ALL WM logic: scoring, packing, unroll, decode, strip builders, dataclasses |
| `src/segment_grpo_reference.py` | Oracle run discovery, `load_oracle_reference_frames` |
| `src/metaworld_jepa_render.py` | `build_jepa_metaworld_env` (corner2 + cam patch), `render_jepa_rgb` (V-flip) |
| `vendor/pi05/jepa_cem_paired_pushv3_export.py` | Hub WM load, `_infer_action_dims`, SmolVLA exec, image/proprio helpers |
| `tests/test_segment_grpo_loop.py` | Unit + integration tests for pack, unroll count, decode, comparison frames |

---

## 2. Camera architecture (critical — changed since original plan was drafted)

### Two image streams in parity mode

When `--wm-sim-camera-parity` is on (default):

```
sim render
    │
    └─ render_jepa_rgb(env)
          corner2 camera, cam_pos patch, square renderer
          env.render().copy()[::-1]   ← V-flip only
          → current_image   (used by WM encode + comparison strips)
          │
          └─ _derive_policy_rgb_for_smolvla(current_image, ...)
                  if --smolvla-policy-hflip-corner2 (default):
                      np.flip(current_image, axis=1)  ← H-flip added
                  → current_policy_image   (used by SmolVLA policy)
```

**WM encode** = corner2 + V-flip only.  
**SmolVLA** = corner2 + V-flip + H-flip.

This is intentional — documented in `docs/wm_versus_smolvla_versus_environment_camera.md`. Both streams derive from the same `render_jepa_rgb` call, so there is no silent mismatch. The H-flip on SmolVLA aligns it with its training oracle PNGs (V+H convention).

### Oracle start_frame similarity check

When checking if env reset matches oracle PNG:

```python
sf = np.asarray(start_frame)
if wm_goal_flip_horizontal and jepa_parity_sim:
    sf = np.flip(sf, axis=1)   # H-flip oracle to match WM's V-only live stream
start_frame_similarity = _frame_similarity(current_image, sf)
```

The oracle PNG (V+H) is H-flipped back to (V-only) before comparison with `current_image`. Correct.

### Vendor `main()` rollout — DIFFERENT camera

`vendor/pi05/jepa_cem_paired_pushv3_export.py` `main()` uses a plain `env_cls()` with no corner2/cam-patch/V-flip. If someone runs that script directly, WM sees different pixels than segment-GRPO's jepa parity mode. This is not used in the main pipeline, but worth knowing.

---

## 3. Packing math — verified correct

### `_pack_env_actions_for_wm` (`src/segment_grpo_loop.py` ~L411)

```python
n_pad = (factor - (t % factor)) % factor   # 0 when T exactly divisible
if n_pad:
    pad = np.repeat(arr[-1:], n_pad, axis=0)   # last-action repeat, NOT zeros
    arr = np.concatenate([arr, pad], axis=0)
n_blk = arr.shape[0] // factor             # = ceil(T / factor)
packed = arr.reshape(n_blk, wm_dim)
```

`T_wm = ceil(T/factor)` — correct. Verified by `test_normalize_and_pack_env_actions_for_wm_factor5`.

### T enumeration with factor=5

| T | n_pad | T_wm | Real actions in last block | Padded |
|---|-------|------|---------------------------|--------|
| 1 | 4 | 1 | 1 | 4 |
| 2 | 3 | 1 | 2 | 3 |
| 3 | 2 | 1 | 3 | 2 |
| 4 | 1 | 1 | 4 | 1 |
| 5 | 0 | 1 | 5 | 0 (exact) |
| 6 | 4 | 2 | 1 in last block | 4 |
| 7 | 3 | 2 | 2 in last block | 3 |
| 8 | 2 | 2 | 3 in last block | 2 |
| 9 | 1 | 2 | 4 in last block | 1 |

**Important semantic implication:** When `T % factor != 0`, the last WM step was generated from padded (phantom) actions the sim never executed. The WM predicts as if 5 steps ran; the sim may have run fewer. This can look like "WM moved too much" in strips. **This is a WARN, not a bug — see W-PAD-GHOST below.**

### Factor inference

`_wm_action_block_factor(env_dim, wm_dim)`:
- Returns `wm_dim // env_dim` when exactly divisible.
- Returns `1` silently when not divisible — then `_pack_env_actions_for_wm` **raises** `RuntimeError` if `env_dim != wm_dim`. No silent mispack.
- **WARN W-DIM-LATE:** Error is deferred to pack time, not at factor inference. No log warning at inference.

### Upstream confirmation

Upstream JEPA-WM YAML (`~/.cache/torch/hub/facebookresearch_jepa-wms_main/configs/vjepa_wm/mw_final_sweep/...yaml`):
```yaml
frameskip: 5
action_skip: 1
```
Upstream `plan_evaluator.py`: `action_ratio = frameskip // action_skip = 5`. Upstream reshapes `(t, f*d) → (t*f, d)` before `env.step_multiple`. Local `factor = wm_dim // env_dim = 20 // 4 = 5` is consistent with this.

---

## 4. Strip alignment — verified correct (with caveats)

### `_select_comparison_frames` (`src/segment_grpo_loop.py` ~L696)

**factor > 1 path:**
```python
for k in range(len(pred_frames)):
    ridx = min((k + 1) * factor, cs)   # cs = carried_steps
    if ridx < len(real_frames):
        out_real.append(real_frames[ridx])
        out_pred.append(pred_frames[k])
        out_ridx.append(ridx)
```

- `real_frames[0]` = pre-execution start frame.
- `real_frames[i]` for i≥1 = after i-th executed step.
- `len(real_frames) = carried_steps + 1`.
- `ridx = min((k+1)*factor, carried_steps) ≤ carried_steps < len(real_frames)` — **no OOB possible**.
- Final state (`real_frames[carried_steps]`) IS shown in strip (last column).

**factor == 1 path:**
```python
limit = min(len(real_frames)-1, len(pred_frames), carried_steps or len(pred_frames))
out_real = real_frames[:limit]
out_pred = pred_frames[:limit]
```
- Returns `real_frames[:limit]` starting at index 0 — the **pre-execution start frame**.
- Column 0: WM pred step 0 (after 1 step) vs `real_frames[0]` (before any step) — **1-step misalignment by design**.
- `real_frames[carried_steps]` (final state) is **never shown** in factor=1 strips.
- Test `test_select_comparison_frames_keeps_t0_as_context` validates this as intentional ("t0 as context").

### `env_steps_per_wm_step` propagation

`DecodeTrace.env_steps_per_wm_step = factor` (set at score time). The main loop reads:
```python
wm_stride = int(getattr(selected_trace, "env_steps_per_wm_step", 1) or 1)
```
And passes `env_steps_per_wm_step=wm_stride if wm_stride > 1 else None` into `_write_comparison_segment_strip`. Correct.

### Strip length for batched vs iterative

Both batched and iterative decode produce exactly `T_wm` pred frames — one per WM step. This is verified by tests:
- `test_iterative_decode_trace_one_step_per_action`
- `test_decode_selected_trace_prefers_fused_modal_and_accepts_structured_latents`

Strip does not assume `len(pred_frames) == carried_steps`. It uses `min(len(pred_frames), ...)` to bound columns. Correct.

### `carried_steps=0` handling

Double-guarded:
1. Main loop: `if comparison_root_path is not None and pred_frames and carried_steps > 0:` — never calls strip writer.
2. Strip writer: `if int(carried_steps) <= 0: return None, None` — early exit.

---

## 5. Reset, seed, oracle, episode-end

### `load_oracle_reference_frames` (`src/segment_grpo_reference.py` ~L96)

- `goal_frame_index` is **1-based** in the public API (e.g. 25 → `frame_000024.png`).
- Internally stored as `goal_idx = goal_frame_index - 1` (0-based).
- `OracleReferenceFrames.goal_frame_idx_zero_based` stores that 0-based PNG index (field name disambiguates from CLI 1-based `goal_frame_index`).
- `start_frame_index=0` (default) → `frame_000000.png`. Correct.

### Seed resolution (`scripts/run_segment_grpo.py` `_resolve_oracle_plan`)

Priority order:
1. `--episode-index` + `--reset-seed` → use both explicit.
2. `--episode-index` only → seed = `base_seed + episode_offset * 997` (synthetic, **not** the top-15 seed for that episode).
3. `--reset-seed` only (no `--episode-index`) → uses `episode_offset` (loop counter) as episode index → oracle path may point at wrong folder.
4. Neither → top-15 table row if `--oracle-run` provided; else synthetic seed.

**WARN W-SEED-MISMATCH:** Using `--episode-index` without `--reset-seed` can cause `start_frame_similarity` warning even when camera is correct — the sim's reset state doesn't match the oracle recording.

### `_frame_similarity` — warning only, no state change

Pure function. Returns mean absolute difference ∈ [0,1]. Callers only set `reset_frame_warning` and print. Does not affect WM encode or policy.

**WARN W-CAM-WARNING:** The warning text ("`reset frame mismatch`") cannot distinguish "wrong reset seed → different env state" from "wrong camera contract → pixels differ despite correct state". Check `wm_goal_for_encode.png` debug artifact or compare `goal_source` path to understand which oracle run was used.

### Goal latent encoding

```python
_encode_state_to_latent(wm_bundle,
    _prepare_goal_image_for_wm(goal_frame, flip_horizontal=wm_goal_flip_horizontal),
    fallback_proprio=current_proprio)   # ← live reset-time proprio, NOT oracle proprio
```

Goal visual comes from oracle PNG. Goal proprio is the **live sim proprio at reset time**, not whatever proprio the oracle agent had when it reached the goal. This is expected and documented, but means "goal latent" is a visual-only goal. Report as "goal looks wrong" → check `wm_goal_for_encode.png` artifact.

### `carried_steps` and `segment_real_frames` consistency

In the sim path:
```python
segment_real_frames = [np.asarray(current_image, copy=True)]  # index 0 = pre-execution
for i in range(effective_len):
    ...execute step...
    carried_steps += 1
    segment_real_frames.append(np.asarray(current_image, copy=True))
    if step_done:
        break
# → len(segment_real_frames) == carried_steps + 1  always
```

`carried_steps` never goes negative in the sim path (only `+= 1` inside loop). `done` on first step yields `carried_steps = 1`, not 0.

**WARN W-PRED-SHORT:** If `pred_frames` (decode trace) is shorter than `carried_steps` (e.g. decode partially failed), the strip silently shows fewer columns. The episode metadata `carried_steps` field still reflects the full number of executed steps.

---

## 6. Metadata and artifact layout

### JSON episode log fields (verified against real run)

Real run sample: `artifacts/phase08_segment_grpo_baseline/run_20260412T220041Z_.../segment_grpo_campaign1003_ep3_s1003_chunk20_overlaydist.json`

Per-candidate `meta` dict:
```json
{
  "planner_action_dim": 20,
  "env_action_dim": 4,
  "effective_chunk_len": 20,
  "wm_scoring_status": "ok",
  "wm_env_steps_per_wm_step": 5,
  "decode_status": "skipped",
  "latent_trace_len": 4
}
```

Notes:
- Non-selected candidates always have `"decode_status": "skipped"` — this means "not decoded" (decode only runs for selected candidate), not "decode failed".
- Selected candidate gets `decode_status` updated to `"ok"` or `"failed"` after decode.
- `decode_failure_reason` is added on failure.

Episode-level `metadata`:
```json
{
  "wm_loaded": true,
  "wm_scoring_statuses": ["ok", ...],
  "decode_statuses": ["ok", ...],
  "scoring_failure_reasons": [],
  "decode_failure_reasons": []
}
```

`comparison_strip_path` and per-segment `comparison_strip_path` are absolute paths → open PNGs directly from JSON.

### Artifact paths (corrected — plan doc was wrong)

Per-segment strip:
```
<artifact_dir>/comparison/episode_<XXXX>/<basename>.png
```
Where `basename = wmf05_comparison_strip_steps_<START>_to_<END>_seg<N>_cand<M>.png` (prefix `wmf05_` when factor>1).

Stitched episode strip:
```
<artifact_dir>/comparison/episode_<XXXX>_comparison_strip.png
```

**Note:** The plan document says `segment_XXXX/comparison_strip.png` — that's stale. Segments are NOT in subfolders; segment index is only in the filename.

`artifact_dir = <output_json_parent>/<output_json_stem>_artifacts`

### What's missing from single-episode JSON

- **Reset seed** — not in episode JSON; only in the multi-episode `segment_grpo_manifest.json`.
- **Oracle run root** — reconstructible from `goal_source` path prefix.
- **Startup log of `wm_dim`/`env_dim`/`factor`** — not emitted anywhere (see improvement W-NO-STARTUP-LOG).

---

## 7. H1–H5 hypothesis verdicts

### H1: Pred columns are sparse vs env when factor=5 — NOT A BUG

WM predictions correspond to groups of 5 env steps. Strip correctly pairs pred `k` with `real_frames[min((k+1)*5, carried_steps)]`. Filename includes `wmf05_` prefix indicating 5x factor. This is expected WM temporal resolution.

**Evidence:** `test_select_comparison_frames_with_wm_step_factor`, upstream YAML `frameskip=5`.

### H2: `T % factor != 0` → padded WM steps appear in strip — WARN, NOT CRASH

The last WM step (when `T % factor != 0`) was generated from phantom padded actions. It appears in the strip as a normal column. When additionally `carried_steps < factor`, multiple columns can pin to the same real frame. No visual annotation distinguishes ghost steps.

**Evidence:** `_pack_env_actions_for_wm` padding logic; `_select_comparison_frames` with `cs < factor` traces.

### H3: `wm_dim % env_dim != 0` → silent mismatch — SAFE, GUARDED

`_wm_action_block_factor` returns 1 silently. But `_pack_env_actions_for_wm` then raises `RuntimeError("WM dim {wm_dim} must equal factor*{env_dim}=...")` when `factor * env_dim != wm_dim`. No silent mispack.

**Evidence:** `_pack_env_actions_for_wm` L396-408.

### H4: Oracle start_frame mismatch → user blames WM — DESIGN LIMITATION

`_frame_similarity` fires a warning. Cannot distinguish camera mismatch from seed/state mismatch. WM still encodes live sim state regardless of warning. `_frame_similarity` warning threshold is configurable.

**Fix path:** Check `goal_source` in JSON to confirm oracle run, verify seed used matches oracle recording seed (from manifest or logs), check `wm_goal_for_encode.png` artifact for goal visual.

**Evidence:** `_frame_similarity` (~L421), similarity caller (~L2013-2024).

### H5: Batched vs iterative produce different decode trace lengths — OK

Both produce exactly `T_wm` pred frames. Strip length is `min(T_wm, carried_steps_limit)` columns in both modes. No assumption that `len(pred) == carried_steps`.

**Evidence:** `test_iterative_decode_trace_one_step_per_action`, `test_decode_selected_trace_prefers_fused_modal_and_accepts_structured_latents`.

---

## 8. Bug report

### BUG F6 — Iterative scoring uses wrong final_vector when model provides `"latent"` key

**Symptom:** 1 failing test: `test_iterative_rollout_keeps_structured_score_decode_traces_and_scores_from_final_latent`. In production: if the WM checkpoint returns a dict with a dedicated `"latent"` key that differs from `"visual"`, the wrong candidate may be selected.

**File:function:**
- `src/segment_grpo_loop.py` `_extract_scoring_latent` (~L1248–1287)
- `src/segment_grpo_loop.py` `_latent_vector_from_unroll_step` (~L1552)
- Default `wm_scoring_latent="visual"` (~L1388)

**Root cause:** `_latent_vector_from_unroll_step` routes through `_extract_scoring_latent(mode=wm_scoring_latent)`. When `mode="visual"` (default) and model output is a dict, it reads `z["visual"]`. If the model also provides `z["latent"]` as a fused representation, that is ignored. The `_extract_latent_with_fallback` call at L1551 correctly extracts `"latent"` for the trace storage — but this is separate from the scoring path.

**Test failure output:**
```
ACTUAL:  score_trace.final_vector = [3., 4., 5.]   # from "visual" key
EXPECTED: score_trace.final_vector = [3., 3., 3.]  # from "latent" key
```

**Impact:** Scoring skew — wrong candidate may be selected when `"latent"` ≠ `"visual"` projections diverge. Current Metaworld hub checkpoint may not expose a separate `"latent"` key in practice, so production behavior may be unaffected today, but this is a regression risk if checkpoint changes.

**Repro:**
```bash
cd /vol/bitbucket/aa6622/project
/vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python -m pytest tests/test_segment_grpo_loop.py::test_iterative_rollout_keeps_structured_score_decode_traces_and_scores_from_final_latent -v --tb=short
```

---

## 9. Warning catalogue

All warnings below are non-crashing issues. Ranked roughly by potential for silent incorrect behavior.

### W-PAD-GHOST — Ghost WM steps in strip, no annotation
**Location:** `_select_comparison_frames` (~L721-726), `_build_real_vs_pred_strip` overlay (~L875-892)  
**Issue:** When `T % factor != 0`, the final WM block is generated from padded phantom actions. When `carried_steps < factor`, multiple strip columns pin to the same `real_frames[carried_steps]`. Neither case is annotated in the strip image or overlay text. A human looking at the strip cannot tell "this column is from a ghost WM step."  
**Mitigation:** Check `wm_env_steps_per_wm_step` in JSON + `effective_chunk_len` + `carried_steps` to compute which columns are padded. Add overlay label if implementing fix.

### W-STRIP-PRED-SHORT — Strip silently shorter than carried_steps
**Location:** `_select_comparison_frames` factor=1 path (~L729-740)  
**Issue:** If `pred_frames` (decode trace) has fewer entries than `carried_steps`, strip has fewer columns than the episode executed. `carried_steps` in JSON metadata still reflects true executed count. No log entry from strip writer.  
**Mitigation:** Cross-check `segments[i].candidates[selected].meta.latent_trace_len` vs `carried_steps`.

### W-STRIP-NO-FINAL-STATE — factor=1 strip never shows final executed state
**Location:** `_select_comparison_frames` factor=1 path  
**Issue:** Factor=1 strip starts at `real_frames[0]` (pre-execution). `real_frames[carried_steps]` (final state) is never included. Factor>1 strip DOES show the final state in the last column.  
**Mitigation:** Design choice validated by `test_select_comparison_frames_keeps_t0_as_context`. Be aware when visually inspecting factor=1 strips — last column is not the final state.

### W-DIM-LATE — Non-divisible dim mismatch error deferred to pack time
**Location:** `_wm_action_block_factor` (~L362), `_pack_env_actions_for_wm` (~L396)  
**Issue:** `_wm_action_block_factor` silently returns 1 with no log when `wm_dim % env_dim != 0`. The `RuntimeError` only fires when `_pack_env_actions_for_wm` is called. Error message from pack is less clear than it would be if raised at factor inference.  
**Mitigation:** Not a bug — pack guard catches it. If debugging a mysterious `RuntimeError` from `_pack_env_actions_for_wm`, check `_wm_action_block_factor` inputs first.

### W-SEED-MISMATCH — `--episode-index` without `--reset-seed` uses synthetic seed
**Location:** `scripts/run_segment_grpo.py` `_resolve_oracle_plan` (~L258-261)  
**Issue:** When only `--episode-index` is passed, seed is computed as `base_seed + episode_offset * 997`, which is likely NOT the seed used when the oracle frames were recorded. This causes `start_frame_similarity` to warn even if camera parity is correct.  
**Fix:** Always pair `--episode-index` with `--reset-seed` from the top-15 table or oracle campaign manifest. Or use `--oracle-run` to let `_resolve_oracle_plan` pull the correct seed from the table.

### W-SEED-EPISODE — `--reset-seed` without `--episode-index` uses wrong oracle folder
**Location:** `scripts/run_segment_grpo.py` `_resolve_oracle_plan` (~L260)  
**Issue:** Without `--episode-index`, the episode index defaults to `episode_offset` (loop counter starting at 0). Oracle frames are loaded from `episode_0000/` etc. even if the seed corresponds to a different recorded episode.  
**Fix:** Always pair `--reset-seed` with `--episode-index`.

### W-GOAL-FRAME-INDEX-NAMING — ~~resolved~~
**Was:** `OracleReferenceFrames.goal_frame_index` stored 0-based index under a name that looked 1-based.  
**Now:** Field renamed to `goal_frame_idx_zero_based`. Episode JSON `goal_frame_index` remains the user’s 1-based CLI value (`EpisodeLog`), separate from the oracle loader dataclass.

### W-CAM-WARNING — Reset warning can't distinguish camera vs state mismatch
**Location:** `src/segment_grpo_loop.py` (~L2022-2024)  
**Issue:** `"reset frame mismatch: distance=..."` fires for both wrong seed/state AND camera contract mismatch. No additional diagnostic info in the log line.  
**Mitigation:** Check `wm_goal_for_encode.png` artifact for goal visual. Confirm `goal_source` path points to correct oracle run. If camera parity is on and oracle was recorded with the same jepa script, mismatch is likely a seed issue.

### W-TENSOR-RANK — `_to_tensor` rank-6 normalization brittle
**Location:** `src/segment_grpo_loop.py` `_to_tensor` (~L484-506)  
**Issue:** Hard-codes rank-6 as target shape `[T, B, V, H, W, D]`. Uses iterative squeeze/unsqueeze loops. If iterative trace stores flat 1-D latents (e.g. shape `[D]`), stacking gives `[T, D]` which is padded to `[T, 1, 1, 1, 1, D]` — may not be what `decode_unroll` expects. No test for this case.  
**Mitigation:** Currently not triggered by the standard Metaworld checkpoint. Watch for decode failures if checkpoint changes.

### W-NO-STARTUP-LOG — No `wm_dim`/`env_dim`/`factor` logged at startup
**Location:** `src/segment_grpo_loop.py` `load_wm_bundle` (~L1062-1126)  
**Issue:** `planner_action_dim` (=20) is inferred but never emitted as a single diagnostic log line along with env_dim and factor. Triage requires digging into JSON candidate meta or adding print statements.  
**Mitigation:** Check `segments[0].candidates[0].meta` in episode JSON: `planner_action_dim`, `env_action_dim`, `wm_env_steps_per_wm_step`.

### W-VENDOR-CAMERA — Vendor `main()` uses different camera than jepa parity
**Location:** `vendor/pi05/jepa_cem_paired_pushv3_export.py` `main()` (~L1064-1076)  
**Issue:** Vendor script's standalone rollout builds a plain `env_cls()` with no corner2 camera patch or V-flip. WM and SmolVLA both use `_collect_step_image` (obs RGB or raw `env.render()`). Different pixels than segment-GRPO's jepa parity mode. Not used by the main pipeline — only relevant if running the vendor script directly.

### W-INFER-FALLBACK — `_infer_action_dims` falls back to `[4]` if all probes fail
**Location:** `vendor/pi05/jepa_cem_paired_pushv3_export.py` `_infer_action_dims` (~L660-682)  
**Issue:** If `preprocessor.action_mean`, `model.model.action_dim`, and all `in_features` probes all return 0/None, returns `[4]` (env action dim) as fallback. This is wrong for a Metaworld WM checkpoint (needs 20-D). Results in `factor=1` and pack failure.  
**Mitigation:** The checkpoint probe chain normally succeeds for the standard hub model. Watch for this if using a non-standard checkpoint.

---

## 10. Test environment note

System Python 3.12 lacks `torch` and `tensordict`. All torch-dependent tests **silently skip** under `python -m pytest`. Always use:

```bash
/vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python -m pytest tests/test_segment_grpo_loop.py -v --tb=short
```

Full test run result at time of audit: **34 passed, 1 failed** (BUG F6 above).

---

## 11. Recommended actions (prioritised)

| Priority | Action | Effort |
|----------|--------|--------|
| HIGH | Fix BUG F6: `_extract_scoring_latent` must consult `"latent"` key when present before falling back to mode-specific key | Small — modify `_latent_vector_from_unroll_step` or `_extract_scoring_latent` |
| MEDIUM | Add ghost-step annotation to strip overlay when `(k+1)*factor > carried_steps` | Small — modify `_build_real_vs_pred_strip` overlay text |
| MEDIUM | Emit startup log line `[wm] loaded: wm_dim=20 env_dim=4 factor=5` when `load_wm_bundle` completes and factor is first computed | Tiny |
| MEDIUM | Add `reset_seed` field to single-episode `EpisodeLog` JSON for reproducibility | Small |
| LOW | Add early `logging.warning` in `_wm_action_block_factor` when `wm_dim % env_dim != 0` | Tiny |
| DONE | Renamed `OracleReferenceFrames.goal_frame_index` → `goal_frame_idx_zero_based` | Landed with audit handoff commit |
| LOW | Add `"padded_wm_block": true` flag to candidate meta when `T % factor != 0` for triage | Small |
| INFO | Update plan doc: strip paths are `episode_XXXX/<basename>.png`, not `episode_XXXX/segment_XXXX/comparison_strip.png` | Doc only |

---

## 12. Quick triage guide

**"WM preds look wrong scale / too many steps ahead"**
→ Check `wm_env_steps_per_wm_step` in JSON meta. If 5, strip columns are 5-env-step jumps — expected coarse resolution (H1).

**"Last strip column looks off / ghostly"**
→ Check `effective_chunk_len % 5`. If non-zero, last WM step had phantom padded actions (H2, W-PAD-GHOST).

**"strip has fewer columns than I expected"**
→ Check `carried_steps` in JSON vs `latent_trace_len` in selected candidate meta. If decode trace short → W-STRIP-PRED-SHORT.

**"reset_frame_warning in log"**
→ Check if `--episode-index` paired with correct `--reset-seed`. Check `goal_source` path. If camera recently changed, verify `--wm-sim-camera-parity` and `--smolvla-policy-hflip-corner2` flags (W-SEED-MISMATCH, W-CAM-WARNING).

**"wrong candidate being selected / scoring seems off"**
→ If iterative mode and WM checkpoint exposes `"latent"` key: BUG F6. Workaround: set `--wm-scoring-latent visual` explicitly.

**"RuntimeError from _pack_env_actions_for_wm"**
→ `wm_dim % env_dim != 0` AND `wm_dim != env_dim`. Check checkpoint `planner_action_dim` vs env dim. (W-DIM-LATE)
