# SmolVLA and WM camera contracts (with sources)

This note captures the current camera/flip contract that should be treated as canonical for:

- SmolVLA policy input (checkpoint-specific behavior)
- JEPA-WM input/scoring (jepa-wms parity behavior)

## 1) SmolVLA contract (for `jadechoghari/smolvla_metaworld`)

Recommended setting in this repo:

- `camera_name = corner2`
- `flip_corner2 = true` (equivalent to `np.flip(frame, (0, 1))` in evaluator path)

### Local sources

- Campaign manifests for this checkpoint consistently record:
  - `checkpoint = "jadechoghari/smolvla_metaworld"`
  - `camera_name = "corner2"`
  - `flip_corner2 = true`
  - Example: `artifacts/phase07_smolvla_baseline/campaigns/pushv3_smolvla_topk15_20260412T100211Z_393558/1003/run_manifest.json`
- Evaluator default resolution:
  - `SMOLVLA_METAWORLD_CAMERA_NAME` defaults to `corner2`
  - `SMOLVLA_FLIP_CORNER2` defaults to `true`
  - Source: `src/smolvla_pipeline/evaluator.py`
- Project README parity knobs list the same defaults:
  - Source: `README.md`

### External sources

- Model card (checkpoint identity and base training context):
  - <https://huggingface.co/jadechoghari/smolvla_metaworld>
- LeRobot SmolVLA docs (input contract follows dataset/training setup):
  - <https://huggingface.co/docs/lerobot/smolvla>
- LeRobot issue clarifying model IO flexibility is data-dependent:
  - <https://github.com/huggingface/lerobot/issues/1447>

## 2) WM contract (JEPA-WM / jepa-wms parity)

Recommended setting in this repo:

- MetaWorld `camera_name = corner2`
- `cam_pos[2] = [0.75, 0.075, 0.7]`
- Square render size (`img_size`, default 224 in segment-GRPO CLI)
- Render flip: vertical-only (`[::-1]`)

### Local sources

- Parity renderer implementation:
  - `src/metaworld_jepa_render.py`
  - `build_jepa_metaworld_env(...)` sets `corner2`, camera position patch, square renderer config
  - `render_jepa_rgb(...)` applies vertical-only flip
- Segment-GRPO parity flag wiring:
  - `scripts/run_segment_grpo.py`
  - `--wm-sim-camera-parity` (default on), `--wm-sim-img-size` (default 224), `--wm-goal-hflip`
- Consolidated camera semantics doc:
  - `docs/wm_versus_smolvla_versus_environment_camera.md`

### External source

- jepa-wms reference behavior mirrored by parity renderer:
  - `facebookresearch/jepa-wms` MetaWorld wrapper semantics (corner2 + camera patch + vertical flip)

## 3) Practical integration rule

For this checkpoint, policy and WM contracts are not identical:

- SmolVLA policy path expects the established `corner2 + flip_corner2=true` behavior used in baseline campaign runs.
- WM scoring path expects jepa-wms parity frames (corner2 + vertical-only flip).

If a single frame contract is forced for both, either policy behavior or WM latent alignment can degrade.
