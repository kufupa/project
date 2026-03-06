# WM vs SmolVLA vs environment camera

Reference for how **JEPA-WM**, **SmolVLA**, and **MetaWorld live rendering / oracle artifacts** relate on camera, flip, and resolution. Use this when debugging strip comparisons, goal images, or encode/decode mismatch.

---

## 1. Three pixel contracts (why they differ)

| Actor | Role | Where pixels come from | Canonical flip / camera in-repo |
|--------|------|------------------------|----------------------------------|
| **JEPA-WM** | Latent world model, encode / unroll / decode | Trained on **pre-recorded MP4** rows in `facebook/jepa-wms` Metaworld data (`METAWORLD_HF` → `MetaworldHFDataset`). Not a live camera at train time. | **Planning sim** in facebookresearch/jepa-wms: `MetaWorldWrapper` — `corner2`, `cam_pos[2] = [0.75, 0.075, 0.7]`, square `img_size`, render then **`[::-1]` (vertical flip only)**. |
| **SmolVLA** | Policy (chunk actions) | Fine-tune / eval must match **LeRobot dataset** convention (e.g. Hub Metaworld datasets that **reposition camera + flip** recorded frames). | This repo: `_LeRobotMetaWorldBackend` — `corner2`, same `cam_pos` patch, **`np.flip(frame, (0, 1))`** when `SMOLVLA_FLIP_CORNER2` is true (default **true**) → **vertical + horizontal** (180°-style alignment). |
| **Environment / oracle** | Scripted oracle rollouts, goal PNGs, optional sim for segment GRPO | Live MetaWorld `render()` + your scripts. | Oracle script matches **SmolVLA parity** by design (`run_oracle_parity_1ep.py` docstring: corner2 + flip). Segment GRPO **sim** historically used **weaker** setup (no `camera_name` / no full jepa-wms wrapper). |

**Takeaway:** WM and SmolVLA can **legitimately need different post-renders** if WM checkpoint follows jepa-wms data and SmolVLA checkpoint follows LeRobot-style Metaworld data. Unifying everything requires **proving** one pixel pipeline matches **both** training distributions (often you instead branch: **policy image** vs **WM image**).

---

## 2. JEPA-WM (facebookresearch / jepa-wms)

### Training

- Configs use `dataset_type: custom`, `datasets: [METAWORLD_HF]`, parquet + **video** column decoded in [`MetaworldHFDataset`](https://github.com/facebookresearch/jepa-wms) (`app/plan_common/datasets/metaworld_hf_dset.py`).
- Frames: decode MP4 → `/255` → `THWC → TCHW` → config transforms (e.g. normalize around 0.5).
- **Camera** is whatever was baked into those MP4s at dataset build time; the training loader does not call `env.render()`.

### Official MetaWorld sim bridge (eval / planning)

- File: `jepa-wms/evals/simu_env_planning/envs/metaworld.py`, class `MetaWorldWrapper`.
- Sets `camera_name = "corner2"`, patches `env.model.cam_pos[2] = [0.75, 0.075, 0.7]`, forces square width/height from `cfg.task_specification.img_size` (e.g. **224**), rebuilds `MujocoRenderer` with that camera.
- **`render()`:** `self.env.render().copy()[::-1]` — **vertical flip only** (comment: “flip vertically”).

**Implication:** For WM **encode** / fair comparison to WM **decode**, live sim should match this contract unless you verify HF MP4s used a different convention.

---

## 3. SmolVLA / LeRobot (this project)

### Evaluator behavior

- File: [`project/src/smolvla_pipeline/evaluator.py`](../src/smolvla_pipeline/evaluator.py).
- `SMOLVLA_METAWORLD_CAMERA_NAME` (default `corner2`).
- `SMOLVLA_FLIP_CORNER2` (default `true`) → `_maybe_flip_corner2_frame` applies **`np.flip(np.asarray(frame), (0, 1))`** for `corner2`.

### Why horizontal flip exists

- **SmolVLA parity:** Oracle parity runner explicitly targets **“SmolVLA-parity rendering (corner2 + flip, …)”** — see [`project/scripts/oracle/run_oracle_parity_1ep.py`](../scripts/oracle/run_oracle_parity_1ep.py).
- **LeRobot pattern:** Hub Metaworld-style datasets often document **camera reposition + flip** of rendered images so saved trajectories match eval. LeRobot **environment processors** docs describe similar “flip to match dataset” patterns (e.g. LIBERO 180° alignment via tensor flips). SmolVLA eval should match **the checkpoint’s training data**, not an abstract “best” camera.

### External references

- [lerobot/metaworld_mt50_push_v2_image](https://huggingface.co/datasets/lerobot/metaworld_mt50_push_v2_image) — dataset card mentions repositioning camera and flipping rendered images.
- [LeRobot — Environment processors](https://huggingface.co/docs/lerobot/en/env_processor) — pattern: env-specific flips to match Hub datasets.
- [MetaWorld issue #559](https://github.com/Farama-Foundation/Metaworld/issues/559) — raw `render()` orientation confusion; community uses various flips/rotations.

---

## 4. Oracle baseline (phase06) and goal frames

- Script: [`project/scripts/oracle/run_metaworld_oracle_eval.py`](../scripts/oracle/run_metaworld_oracle_eval.py).
- Defaults: `ORACLE_METAWORLD_CAMERA_NAME` / `--camera-name` → **corner2**; `ORACLE_FLIP_CORNER2` / `--flip-corner2` → **true**; same **`cam_pos[2]`** patch as above; `_render_rgb_frame` uses **`np.flip(..., (0, 1))`** when flip enabled.
- **Goal images** for segment GRPO (e.g. `--goal-frame-index 25`) load **PNG files** from a prior oracle run under `artifacts/phase06_oracle_baseline/run_*/frames/episode_XXXX/frame_000024.png` (1-based index 25 → 0-based filename `000024`). These are **pre-recorded**, not re-rendered during segment GRPO.
- **Start frame** from oracle is used for **similarity warning** vs live reset (`segment_grpo_loop`); it does not replace live sim pixels unless you change the code.

---

## 5. Segment GRPO sim — jepa camera parity

- [`project/src/metaworld_jepa_render.py`](../src/metaworld_jepa_render.py): `build_jepa_metaworld_env`, `render_jepa_rgb` — mirrors jepa-wms `MetaWorldWrapper` (corner2, `cam_pos[2]`, square `img_size`, vertical flip on raw render).
- Default sim path in [`project/src/segment_grpo_loop.py`](../src/segment_grpo_loop.py) `rollout_with_chunks`: **single** jepa-parity RGB feeds SmolVLA chunk sampling, WM scoring, goal-latent fallback from live frame, and comparison-strip “real” rows. Legacy obs-based RGB remains available via `--no-wm-sim-camera-parity`.
- **WM goal from oracle PNG:** Oracle saves V+H vs raw; jepa-wms live uses V-only. For WM encode from stored goal pixels, pipeline applies **`np.flip(rgb, axis=1)`** when `--wm-goal-hflip` is on (default). `--no-wm-goal-hflip` restores prior encode-without-H-flip behavior.
- **Debug PNG:** When comparison artifacts are enabled (`comparison_root` from `run_segment_grpo`), each episode may write **`comparison/episode_XXXX/wm_goal_for_encode.png`**: uint8 HWC equal to the goal tensor **after** `_prepare_goal_image_for_wm` (what WM sees before resize/float). Use to eyeball against first live parity frame / jepa contract.
- **CLI:** `scripts/run_segment_grpo.py` — `--wm-goal-hflip` / `--no-wm-goal-hflip`, `--wm-sim-camera-parity` / `--no-wm-sim-camera-parity`, `--wm-sim-img-size` (default 224). [`scripts/segment_grpo/run_first_episode_real_pipeline.sh`](../scripts/segment_grpo/run_first_episode_real_pipeline.sh) forwards parity defaults for Slurm.
- **Episode task:** `env.set_task(train_tasks[episode_index % len(train_tasks)])` when `train_tasks` is non-empty (aligned with oracle episode indexing).
- **Reset warning:** With parity sim + goal H-flip, oracle **start** frame is H-flipped before `_frame_similarity` vs live RGB so the warning does not systematically false-alarm.

Canonical implementation plan: [`docs/superpowers/plans/2026-04-13-wm-camera-parity-single-handoff.md`](superpowers/plans/2026-04-13-wm-camera-parity-single-handoff.md).

---

## 6. Quick comparison table

| Item | jepa-wms `MetaWorldWrapper` | Oracle / SmolVLA path (this repo) |
|------|---------------------------|-----------------------------------|
| Camera name | `corner2` | `corner2` |
| `cam_pos[2]` | `[0.75, 0.075, 0.7]` | Same |
| Flip after render | **V only** `[::-1]` | **V + H** `np.flip(..., (0,1))` |
| Square render size | `cfg.task_specification.img_size` (often 224) | Oracle env default unless changed; SmolVLA backend sets up env similarly to oracle |

---

## 7. Practical recommendations

1. **WM path:** Prefer pixels aligned with **jepa-wms `MetaWorldWrapper`** (corner2 + cam patch + **V-flip** + training resolution) before `encode` / when comparing to decode.
2. **SmolVLA path:** Match **checkpoint + dataset**; default here is **corner2 + H+V flip** via env vars.
3. **If one global pipeline:** Only safe after **pixel spot-check** (e.g. one HF MP4 frame vs your live render) for WM **and** policy eval accuracy for SmolVLA.
4. **Do not** expect a learned “camera normalization” layer in WM to fix systematic viewpoint mismatch — fix **inputs** to the encoder.

---

## 8. Related docs in this repo

- [`docs/superpowers/WM-vs-env-chunk-len.md`](superpowers/WM-vs-env-chunk-len.md) — chunk length / env steps vs WM (orthogonal to camera, but same integration surface).
- Oracle / SmolVLA campaign notes under [`docs/superpowers/`](superpowers/).

---

## 9. Citation / upstream

- JEPA-WM repo: [facebookresearch/jepa-wms](https://github.com/facebookresearch/jepa-wms)
- Datasets: [facebook/jepa-wms on Hugging Face](https://huggingface.co/datasets/facebook/jepa-wms)
- Paper (dataset / WM context): *What Drives Success in Physical Planning with Joint-Embedding Predictive World Models?* — see HF dataset README for arXiv link.

---

*Last consolidated: 2026-04-13 — camera / flip semantics for Metaworld + this codebase.*
