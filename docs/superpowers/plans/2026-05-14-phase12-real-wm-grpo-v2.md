# Phase12 Real WM-GRPO Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build honest Phase12: real SmolVLA chunk-GRPO training with JEPA-WM structured visual/proprio rewards, plus a one-update validation smoke that proves the full training path works before the 100-update run.

**Architecture:** Keep Phase8 as JEPA load/render/action-pack/decode plumbing, not as training logic. Keep Phase111 as optimizer/checkpoint/old-policy reference, not as rollout logic. Phase12 owns serial receding-horizon chunk sampling, WM scoring, best-chunk execution, chunk-level GRPO updates, artifacts, and Slurm contracts.

**Tech Stack:** Python 3.12, PyTorch, patched LeRobot SmolVLA in `/vol/bitbucket/aa6622/.envs/lerobot_mw_py310`, MetaWorld, JEPA-WM, TensorDict-like structured latents, imageio/Pillow, Slurm with `--export=NIL` and `scripts/slurm/common_env.sh`.

---

## Critical Review Verdict

Current Phase12 is not real WM-GRPO training. It is a rollout-validation bridge:

- `scripts/grpo/train_phase12_wm_chunk_grpo.py` calls Phase8 `rollout_with_chunks(...)`, uses `train_steps=0`, writes empty checkpoint policy/optimizer states, and prints a GRPO marker.
- `--num-updates` is recorded but no optimizer loop consumes it.
- `action_profile` is parsed but not applied to actual sampled candidate actions.
- `goal_latent_mode=visual_proprio` maps to Phase8 `wm_scoring_latent="concat"`, which is flattened latent distance, not `visual_mse + 0.1 * proprio_mse`.
- Smoke manifests fabricate artifact paths via `setdefault(...)`; audited run dirs contain no `.mp4` and no `.png`.
- Slurm markers and output dirs imply GRPO training even when only validation happened.

95% confidence source:

- Direct code audit of `train_phase12_wm_chunk_grpo.py`, `phase12_rollout.py`, `phase12_objective.py`, `phase12_actions.py`, `phase12_logging.py`, Phase111 trainer, Phase8 WM helpers, Slurm scripts, and Phase12 tests.
- Independent subagent audits agreed on same gaps: trainer is smoke bridge, artifacts fake, Slurm/static contracts weak, action profile unwired, Phase8 scoring unsuitable as default Phase12 reward.
- Web research is not needed for remaining decisions because remaining blockers are repo-local implementation/contracts, not external API uncertainty.

## Latest User Decisions For Coding Sprint

These override earlier plan wording:

- `wm_grpo_train` is default CLI mode.
- Focus of next sprint is real WM-GRPO training, not extended validation.
- Validation means real training smoke:
  - `--mode wm_grpo_train`
  - `--num-episodes 1`
  - `--num-updates 1`
  - real optimizer step,
  - real checkpoint,
  - real artifact files.
- Do not add or run a 10-episode validation path.
- After 1-update smoke passes, run Phase111-style 100-update WM-GRPO.
- Match non-WM Phase111 setup unless WM requires a change:
  - one GPU,
  - `official_lerobot_guarded`,
  - serial rollout first,
  - `group_size=4`,
  - `max_steps=120`,
  - `train_seed_base=2000`,
  - `save_every=5`,
  - `lr=1e-5`,
  - `init_log_std=-2.0`,
  - `euler_step_noise_std=0.2`,
  - `action_transform=no_tanh`.
- Use existing video/artifact utilities where possible:
  - call `smolvla_pipeline.evaluator._write_episode_video(...)` via thin Phase12 wrapper,
  - use existing overlay/reward/success behavior,
  - generate oracle baseline through the same LeRobot-backed MetaWorld env path as Phase12 selected rollouts,
  - add a narrow expert-action method to the local LeRobot adapter if needed,
  - write oracle MP4 directly to its final target path,
  - do not copy MP4 bytes,
  - do not call Phase06 oracle scripts as a fallback.
- One-update validation smoke must produce two real non-empty videos:
  - selected action rollout video for chosen SmolVLA/WM actions,
  - oracle baseline video for same task/seed/setup.
- Decode strictness:
  - one-update smoke runs strict decode for selected candidate diagnostics,
  - if selected WM decode/real-vs-pred strip fails in smoke, smoke fails,
  - strict decode smoke must run with `JEPA_WM_DISABLE_IMAGE_HEAD=0`,
  - reason: JEPA `hubconf.py` removes `heads_cfg.pretrain_dec_path` when `JEPA_WM_DISABLE_IMAGE_HEAD=1`, so `EncPredWM.decode_unroll(...)` has no `image_head` and cannot emit visual frames,
  - smoke uses `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`; a cache miss is a real smoke failure to diagnose, not something to hide,
  - 100-update run does not fail on decode failure by default,
  - 100-update run records `wm_decode_status`, `decode_failure_reason`, and continues if WM scoring itself is OK.
- Old Phase12 runs remain interpreted as rollout validation only.
  - Old names may exist on disk for audit.
  - New plan must not produce new misleading `phase12_wm_chunk_grpo_smoke` / `phase12_wm_chunk_grpo_10ep` validation runs.
- Real trainer environment contract:
  - step real env with the same official-LeRobot-style setup as Phase111 where possible,
  - use Phase8/JEPA helpers only for WM visual/proprio/action-pack parity,
  - do not fall back to Phase8 `rollout_with_chunks(...)` as training loop.
- Train-mode count invariant:
  - default `num_episodes == num_updates`,
  - each update collects one real episode/seed,
  - fail on mismatch unless an explicit future flag is added.
- Advantage grouping invariant:
  - compute GRPO advantages per segment over that segment's `K` candidates,
  - do not normalize globally across unrelated segment roots,
  - concatenate segment-candidate training records only after per-segment advantages are assigned.
- Optimizer cadence:
  - collect full receding-horizon episode first,
  - one optimizer step per update after collection,
  - no optimizer step between segments.
  - one update = one deterministic episode seed,
  - 100 episodes = `num_updates=100`, `num_episodes=100`, seeds `2000..2099`.
- Strict WM scoring:
  - `wm_grpo_train` must fail on WM scoring error/fallback,
  - no action-norm fallback in train mode,
  - if a debug fallback flag exists, production smoke/training must reject it.
- Zero-advantage rule:
  - one-update smoke fails if all per-segment advantages are zero,
  - 100-update run may skip zero-advantage updates if logged as `skipped=true`.
- Smoke-to-100 gate:
  - run two required one-update smoke profiles:
    - `official_jepa_mirror`: default/main profile, scores raw postprocessed actions like JEPA-WM mirror path,
    - `bounded_executed`: comparison profile, clips actions before both WM scoring and env execution,
  - submit `bounded_executed` with `afterok:<official_smoke_job_id>`,
  - both one-update smoke profiles must pass before the 100-update job,
  - inspect videos from both smoke profiles before trusting the 100-update run when possible.
  - run a single 100-update job using `official_jepa_mirror`.
- Goal-generation decision:
  - oracle rollout is a per-seed cached goal generator,
  - oracle produces subgoal frame/proprio latents and oracle baseline video,
  - oracle actions are diagnostics only, not training candidates,
  - SmolVLA samples all candidate chunks,
  - JEPA-WM scores each chunk against scheduled oracle subgoal,
  - selected chunk executes in real env,
  - next segment starts from fresh real image/proprio.
- Env-count decision:
  - Phase12 training uses one real LeRobot env per update, not `group_size` parallel envs,
  - `group_size=4` means four candidate chunks from the same real root observation,
  - `step_batch(...)` is not the Phase12 candidate axis.

## Non-Negotiable Decisions

- No CEM in Phase12 mainline. SmolVLA samples candidates; JEPA-WM scores them; GRPO trains SmolVLA.
- Default objective: `visual_mse + 0.1 * proprio_mse`.
- Objective must compute visual MSE and proprio MSE separately, then combine scalar losses.
- Do not concatenate visual/proprio latents before the distance calculation.
- Correct reward distance:
  - `visual_distance = mean((pred_visual - goal_visual) ** 2)`,
  - `proprio_distance = mean((pred_proprio - goal_proprio) ** 2)`,
  - `combined_distance = visual_distance + 0.1 * proprio_distance`.
- `wm_latent_progress` still makes sense: `start_combined_distance - final_combined_distance`.
- A chunk is rewarded when it reduces the official structured distance to the scheduled oracle subgoal.
- Default reward key: `wm_latent_progress = start_combined_distance - final_combined_distance`.
- Log ablation reward: `latent_return = -final_combined_distance`.
- Default action profile: `official_jepa_mirror`.
- First comparison action profile: `bounded_executed`, smoke-only unless explicitly requested.
- Default chunk length: `25`.
- Default group size: `K=4`.
- Default action transform: `no_tanh`.
- `tanh_norm_ablation` remains explicit ablation only.
- Default env backend: `official_lerobot_guarded`.
- `custom_oracle_aligned` remains ablation only.
- Strict reset mismatch default: fail.
- Intermediate oracle subgoals are mainline. Final-goal-only is deferred.
- GRPO ratio is chunk-level: `exp(sum_t new_logp_t - sum_t old_logp_t)`.
- Default CLI mode is real training: `wm_grpo_train`.
- Rollout validation is an explicit smoke mode only.
- Run one 1-episode/1-update validation smoke, then go straight to real 100-update WM-GRPO training.
- Match Phase111 setup unless Phase12 needs a WM-specific override:
  - `official_lerobot_guarded`,
  - serial rollout first,
  - `max_steps=120`,
  - `group_size=4`,
  - `train_seed_base=2000`,
  - `save_every=5`,
  - `lr=1e-5`,
  - `init_log_std=-2.0`,
  - `euler_step_noise_std=0.2`,
  - `action_transform=no_tanh`,
  - output shape like `artifacts/phase12_wm_chunk_grpo_train/push-v3/g4_u100_seed2000_<profile>`.
- After executing selected chunk, discard predicted WM state and re-render/re-encode real env state.

## File Structure

### Modify Existing Files

- `scripts/grpo/train_phase12_wm_chunk_grpo.py`
  - Add `--mode {rollout_validation,wm_grpo_train}`.
  - Default to `wm_grpo_train`.
  - Keep validation path honest.
  - Add real training path.
  - Remove fake artifact paths.
  - Remove empty real-training checkpoints.
- `src/smolvla_grpo/phase12_logging.py`
  - Enforce file-backed artifact contracts.
  - Write manifest/progress rows with status fields.
- `src/smolvla_grpo/phase12_diagnostics.py`
  - Reuse existing SmolVLA evaluator artifact/video code where possible.
  - Add a thin Phase12 wrapper only for Phase12-specific output naming/status.
  - Keep decode strip writer.
- `src/smolvla_grpo/phase12_rollout.py`
  - Carry clipped/raw actions.
  - Integrate action profiles.
  - Return enough training data for optimizer.
- `src/smolvla_grpo/phase12_objective.py`
  - Keep objective helpers.
  - Add shape normalization if needed for real WM tensors.
- `src/smolvla_grpo/phase12_goals.py`
  - Add concrete `Phase12Goal`.
  - Wire deterministic LeRobot-backed oracle goal collection.
  - No Phase06 script fallback in train/smoke path.
- `src/smolvla_grpo/lerobot_metaworld_adapter.py`
  - Add narrow expert-action/raw-observation support for Phase12 oracle baseline.
  - Add explicit single-env oracle construction flag, e.g. `enable_expert_oracle=True` or `use_deferred_env=True`.
  - When oracle flag is enabled, construct `DeferredLeRobotMetaworldEnv` even for `n_envs=1`, so our local adapter can expose full raw obs and expert action.
  - Keep existing Phase111 APIs stable.
- `tests/test_phase12_trainer_static.py`
  - Replace tests that assert old GRPO marker.
  - Add training-mode checkpoint/update tests.
- `tests/test_grpo_lerobot_adapter.py`
  - Add regression tests for Phase12 expert-oracle adapter path.
  - Prove `enable_expert_oracle=True` exposes raw obs, expert action, render frame, and agent proprio.
  - Prove default Phase111 path still uses installed LeRobot `make_env(...)` and existing `step_batch(...)` behavior.
- `tests/test_phase12_artifacts.py`
  - Require real existing non-empty artifact files.
- `tests/test_phase12_slurm_static.py`
  - Lock Slurm env, args, markers, no nested `sbatch`.
- `pytest.ini`
  - Add markers: `phase12`, `static`, `slurm_static`.

### Add New Files

- `src/smolvla_grpo/phase12_wm_reward.py`
  - Structured JEPA-WM encode/unroll/score backend.
- `scripts/grpo/submit_phase12_wm_chunk_grpo_train.slurm`
  - Real GRPO training script.
  - Default `100` updates, matching Phase111 single-task run.
  - Positional override `$2=1` gives one-update validation smoke.
- `tests/test_phase12_wm_reward.py`
  - Unit tests for structured reward backend.
- `tests/test_phase12_training_loop.py`
  - Local mocked tests proving optimizer/checkpoint/update rows.

### Preserve Existing Files

- `src/smolvla_grpo/policy_wrapper.py`
  - Reuse `sample_action_chunk_from_proc(...)`.
  - Reuse `get_action_probs_for_chunk_from_proc(...)`.
  - Keep one-step Phase111 APIs unchanged.
- `src/smolvla_grpo/checkpointing.py`
  - Reuse `save_grpo_checkpoint(...)` and `load_grpo_checkpoint(...)`.
- `scripts/grpo/train_phase11_env_on_policy_grpo.py`
  - Reference only. Do not mutate for Phase12.
- `src/segment_grpo_loop.py`
  - Reuse narrow WM helpers only:
    - `_to_wm_visual`
    - `_to_wm_proprio`
    - `_normalize_env_actions_for_wm`
    - `_pack_env_actions_for_wm`
    - `load_wm_bundle`
    - decode strip helpers
  - Do not use `score_chunk_by_goal_latent(...)` as default Phase12 scoring.
  - Do not use `update_grpo_step(...)` for SmolVLA training.

## Task 1: Mode Split And Honest Labels

**Files:**
- Modify: `scripts/grpo/train_phase12_wm_chunk_grpo.py`
- Modify: `tests/test_phase12_trainer_static.py`

- [ ] **Step 1: Write failing tests for mode defaults and labels**

```python
def test_phase12_default_mode_is_wm_grpo_train() -> None:
    args = parse_args([])
    assert args.mode == "wm_grpo_train"


def test_rollout_validation_manifest_is_not_grpo_training(tmp_path) -> None:
    args = parse_args(["--mode", "rollout_validation", "--output-dir", str(tmp_path), "--dry-run"])
    manifest = build_manifest(args)
    assert manifest["mode"] == "rollout_validation"
    assert manifest["method_label"] == "wm_scored_receding_horizon_rollout_validation"
    assert manifest["optimizer_updates"] == 0
    assert manifest["uses_cem"] is False


def test_training_manifest_requires_grpo_mode(tmp_path) -> None:
    args = parse_args(["--output-dir", str(tmp_path), "--dry-run"])
    manifest = build_manifest(args)
    assert manifest["mode"] == "wm_grpo_train"
    assert manifest["method_label"] == "wm_scored_receding_horizon_chunk_grpo"
    assert manifest["optimizer_updates"] == args.num_updates
```

- [ ] **Step 2: Run test to verify fail**

Run:

```bash
PYTHONPATH=src /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python -m pytest tests/test_phase12_trainer_static.py::test_phase12_default_mode_is_wm_grpo_train tests/test_phase12_trainer_static.py::test_rollout_validation_manifest_is_not_grpo_training tests/test_phase12_trainer_static.py::test_training_manifest_requires_grpo_mode -v
```

Expected: fail because `--mode` and honest labels do not exist.

- [ ] **Step 3: Add mode arg and manifest labels**

Patch `parse_args(...)`:

```python
p.add_argument(
    "--mode",
    choices=("rollout_validation", "wm_grpo_train"),
    default="wm_grpo_train",
    help="wm_grpo_train performs optimizer updates; rollout_validation is explicit smoke mode.",
)
```

Patch `build_manifest(...)`:

```python
is_train = str(args.mode) == "wm_grpo_train"
return {
    "created_at": utc_now_iso(),
    "mode": str(args.mode),
    "method_label": (
        "wm_scored_receding_horizon_chunk_grpo"
        if is_train
        else "wm_scored_receding_horizon_rollout_validation"
    ),
    "uses_cem": False,
    "optimizer_updates": int(args.num_updates) if is_train else 0,
    "checkpoint": str(args.checkpoint),
    "jepa_ckpt": str(args.jepa_ckpt),
    "jepa_repo": str(args.jepa_repo),
    "task": str(args.task),
    "env_backend": str(args.env_backend),
    "action_profile": str(args.action_profile),
    "chunk_len": int(args.chunk_len),
    "group_size": int(args.group_size),
    "num_episodes": int(args.num_episodes),
    "num_updates": int(args.num_updates),
    "max_steps": int(args.max_steps),
    "objective_type": "L2",
    "goal_latent_mode": str(args.goal_latent_mode),
    "proprio_alpha": float(args.proprio_alpha),
    "reward_key": str(args.reward_key),
    "ratio_mode": str(args.ratio_mode),
    "action_transform": str(args.action_transform),
    "reset_mismatch": str(args.reset_mismatch),
    "decode_candidates": str(args.decode_candidates),
    "save_wm_decodes": bool(args.save_wm_decodes),
}
```

- [ ] **Step 4: Replace startup event**

Use:

```python
write_jsonl_row(
    out / "progress.jsonl",
    {
        "created_at": utc_now_iso(),
        "event": "run_start" if not args.dry_run else "dry_run",
        "mode": args.mode,
        "action_profile": args.action_profile,
        "chunk_len": int(args.chunk_len),
        "goal_latent_mode": args.goal_latent_mode,
        "proprio_alpha": float(args.proprio_alpha),
    },
)
```

- [ ] **Step 5: Run tests**

Run:

```bash
PYTHONPATH=src /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python -m pytest tests/test_phase12_trainer_static.py -v
```

Expected: new tests pass; old tests that assert `PHASE12_WM_CHUNK_GRPO_DONE` are updated in Task 8.

## Task 2: File-Backed Artifact Contract

**Files:**
- Modify: `src/smolvla_grpo/phase12_logging.py`
- Modify: `tests/test_phase12_artifacts.py`
- Modify: `scripts/grpo/train_phase12_wm_chunk_grpo.py`

- [ ] **Step 1: Write failing artifact tests**

```python
from pathlib import Path
import pytest

from smolvla_grpo.phase12_logging import assert_smoke_manifest_contract


def test_smoke_manifest_rejects_nonexistent_files(tmp_path: Path) -> None:
    manifest = {
        "rollout_validation_video": str(tmp_path / "missing.mp4"),
        "selected_action_rollout_video": str(tmp_path / "missing_selected.mp4"),
        "wm_decode_selected_strip_path": str(tmp_path / "missing.png"),
        "wm_real_vs_pred_selected_strip_path": str(tmp_path / "missing2.png"),
        "oracle_baseline_video": str(tmp_path / "oracle_baseline.mp4"),
        "oracle_baseline_video_status": "ok",
        "success_any": False,
        "success_last": False,
    }
    with pytest.raises(AssertionError, match="does not exist"):
        assert_smoke_manifest_contract(manifest, base_dir=tmp_path)


def test_smoke_manifest_accepts_existing_nonempty_files(tmp_path: Path) -> None:
    video = tmp_path / "rollout_validation.mp4"
    oracle = tmp_path / "oracle_baseline.mp4"
    strip = tmp_path / "wm_decode_selected_strip.png"
    real_vs_pred = tmp_path / "wm_real_vs_pred_selected_strip.png"
    video.write_bytes(b"mp4")
    oracle.write_bytes(b"oracle")
    strip.write_bytes(b"png")
    real_vs_pred.write_bytes(b"png2")
    manifest = {
        "rollout_validation_video": str(video),
        "selected_action_rollout_video": str(video),
        "wm_decode_selected_strip_path": str(strip),
        "wm_real_vs_pred_selected_strip_path": str(real_vs_pred),
        "oracle_baseline_video": str(oracle),
        "oracle_baseline_video_status": "ok",
        "success_any": False,
        "success_last": False,
    }
    assert_smoke_manifest_contract(manifest, base_dir=tmp_path)
```

- [ ] **Step 2: Run test to verify fail**

Run:

```bash
PYTHONPATH=src /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python -m pytest tests/test_phase12_artifacts.py -v
```

Expected: fail because validator accepts path strings only and lacks `base_dir`.

- [ ] **Step 3: Implement strict validator**

Replace `assert_smoke_manifest_contract(...)` with:

```python
def _resolve_manifest_path(value: str, base_dir: Path | None) -> Path:
    path = Path(value)
    if not path.is_absolute() and base_dir is not None:
        path = Path(base_dir) / path
    return path


def _assert_nonempty_file(path: Path, key: str) -> None:
    assert path.exists(), f"smoke manifest {key} does not exist: {path}"
    assert path.is_file(), f"smoke manifest {key} is not a file: {path}"
    assert path.stat().st_size > 0, f"smoke manifest {key} is empty: {path}"


def assert_smoke_manifest_contract(manifest: dict[str, Any], *, base_dir: Path | None = None) -> None:
    required = (
        "rollout_validation_video",
        "selected_action_rollout_video",
        "success_any",
        "success_last",
        "oracle_baseline_video_status",
    )
    for key in required:
        assert key in manifest, f"smoke manifest missing {key}"
    _assert_nonempty_file(
        _resolve_manifest_path(str(manifest["rollout_validation_video"]), base_dir),
        "rollout_validation_video",
    )
    _assert_nonempty_file(
        _resolve_manifest_path(str(manifest["selected_action_rollout_video"]), base_dir),
        "selected_action_rollout_video",
    )
    if manifest.get("wm_decode_status") == "ok":
        for key in ("wm_decode_selected_strip_path", "wm_real_vs_pred_selected_strip_path"):
            assert key in manifest, f"smoke manifest missing {key}"
            _assert_nonempty_file(_resolve_manifest_path(str(manifest[key]), base_dir), key)
    assert manifest["oracle_baseline_video_status"] == "ok"
    assert "oracle_baseline_video" in manifest, "smoke manifest missing oracle_baseline_video"
    _assert_nonempty_file(
        _resolve_manifest_path(str(manifest["oracle_baseline_video"]), base_dir),
        "oracle_baseline_video",
    )
    assert isinstance(manifest["success_any"], bool), "success_any must be boolean"
    assert isinstance(manifest["success_last"], bool), "success_last must be boolean"
```

- [ ] **Step 4: Remove fake `setdefault(...)` artifact paths**

Delete these from `run_phase12_episode(...)`:

```python
meta.setdefault("oracle_baseline_video", ...)
meta.setdefault("smolvla_first_rollout_video", ...)
meta.setdefault("wm_decode_selected_strip_path", ...)
meta.setdefault("wm_real_vs_pred_selected_strip_path", ...)
```

Only use paths returned by writers.

- [ ] **Step 5: Run tests**

Run:

```bash
PYTHONPATH=src /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python -m pytest tests/test_phase12_artifacts.py tests/test_phase12_trainer_static.py -v
```

Expected: pass after trainer mocks create files or tests are updated to use real temp files.

## Task 3: Real Rollout Video Artifacts, Reusing Existing Pipeline

**Files:**
- Modify: `src/smolvla_grpo/phase12_diagnostics.py`
- Modify: `tests/test_phase12_diagnostics.py`
- Modify: `scripts/grpo/train_phase12_wm_chunk_grpo.py`

Existing code to reuse:

- `src/smolvla_pipeline/evaluator.py::_write_episode_video(...)`
  - Uses `imageio.v2`.
  - Adds reward/success overlays.
  - Verifies file exists and is non-empty.
- `src/smolvla_pipeline/evaluator.py::write_episode_artifacts(...)`
  - Writes action/reward/success metadata for evaluation episodes.
- `src/smolvla_pipeline/evaluator.py::run_smolvla_eval(...)`
  - Existing convention: per-episode dir plus `videos/<task>_0/eval_episode_XXXX.mp4`.
- `src/smolvla_pipeline/evaluator.py::_write_episode_frames_png(...)`
  - Useful if we want raw frame dumps later.
- `src/smolvla_grpo/lerobot_metaworld_adapter.py::OfficialLeRobotMetaWorldGRPORollout`
  - Existing LeRobot-backed MetaWorld env path used by Phase111.
  - Must be extended narrowly for Phase12 oracle baseline:
    - retain full raw MetaWorld observation after reset/step,
    - expose scripted expert action from LeRobot `expert_policy`,
    - expose render frame for the single env,
    - allow deterministic reset parity check on the same env.
- Installed LeRobot reference:
  - `lerobot.envs.metaworld.MetaworldEnv` has `expert_policy = TASK_POLICY_MAPPING[self.task]()`,
  - but it does not expose a public expert rollout/video API,
  - public observation is only `pixels` plus `agent_pos`,
  - Phase12 must add the raw-observation/expert-action bridge in our local adapter.

Decision:

- Prefer calling `_write_episode_video(...)` directly from Phase12 wrapper despite private underscore because it is local, already tested in practice, and avoids reinventing overlay/video code.
- Add only a thin Phase12 wrapper:
  - canonical Phase12 filename,
  - Phase12 metadata keys,
  - optional `overlay_mode`,
  - existence check delegated to evaluator writer.
- For selected action rollout video:
  - use actual frames from executed best chunks in the one-update training smoke.
  - write `rollouts/update_0000_episode_0000/selected_action_rollout.mp4`.
- For oracle baseline video:
  - use the same LeRobot-backed MetaWorld env adapter as Phase12 selected rollout.
  - run one scripted expert episode with same task, seed, max steps, camera, and flip settings as Phase12 smoke.
  - write video directly to `rollouts/update_0000_episode_0000/oracle_baseline.mp4` or `oracle/seed_<seed>/oracle_baseline.mp4`.
  - set `oracle_baseline_video` to that final path.
- Do not call `scripts/oracle/run_metaworld_oracle_eval.py` in Phase12 smoke/train path.
- Do not copy generated MP4 bytes into a second path.

- [ ] **Step 1: Write failing video test**

```python
def test_write_phase12_episode_video_writes_nonempty_mp4(tmp_path) -> None:
    from smolvla_grpo.phase12_diagnostics import write_phase12_episode_video

    video = tmp_path / "rollout_validation.mp4"
    frames = [_frame(0), _frame(20), _frame(40)]

    write_phase12_episode_video(
        video_path=video,
        frames=frames,
        rewards=[0.0, 1.0],
        successes=[False, True],
        fps=6,
        overlay_mode="cumulative_reward",
    )

    assert video.is_file()
    assert video.stat().st_size > 0
```

```python
def test_smoke_manifest_requires_selected_and_oracle_videos(tmp_path) -> None:
    from smolvla_grpo.phase12_logging import assert_smoke_manifest_contract

    selected = tmp_path / "selected_action_rollout.mp4"
    oracle = tmp_path / "oracle_baseline.mp4"
    selected.write_bytes(b"selected")
    oracle.write_bytes(b"oracle")
    manifest = {
        "selected_action_rollout_video": str(selected),
        "rollout_validation_video": str(selected),
        "oracle_baseline_video": str(oracle),
        "oracle_baseline_video_status": "ok",
        "success_any": False,
        "success_last": False,
        "wm_decode_status": "ok",
        "wm_decode_selected_strip_path": str(tmp_path / "wm_decode_selected_strip.png"),
        "wm_real_vs_pred_selected_strip_path": str(tmp_path / "wm_real_vs_pred_selected_strip.png"),
    }
    (tmp_path / "wm_decode_selected_strip.png").write_bytes(b"decode")
    (tmp_path / "wm_real_vs_pred_selected_strip.png").write_bytes(b"real-vs-pred")

    assert_smoke_manifest_contract(manifest, base_dir=tmp_path)
```

- [ ] **Step 2: Run test to verify fail**

Run:

```bash
PYTHONPATH=src /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python -m pytest tests/test_phase12_diagnostics.py::test_write_phase12_episode_video_writes_nonempty_mp4 -v
```

Expected: fail because function is missing.

- [ ] **Step 3: Implement thin wrapper around evaluator writer**

Add to `phase12_diagnostics.py`:

```python
def write_phase12_episode_video(
    *,
    video_path: Path,
    frames: list[np.ndarray],
    rewards: list[float],
    successes: list[bool],
    fps: int,
    overlay_mode: str = "cumulative_reward",
) -> Path:
    from smolvla_pipeline.evaluator import _write_episode_video

    if not frames:
        raise RuntimeError("No frames captured for Phase12 episode video.")
    video_path = Path(video_path)
    _write_episode_video(
        video_path=video_path,
        frames=frames,
        rewards=[float(x) for x in rewards],
        successes=[bool(x) for x in successes],
        overlay_mode=str(overlay_mode),
        fps=int(fps),
    )
    return video_path
```

- [ ] **Step 4: Integrate writer into validation/training episode outputs**

When episode result contains `frames`, `rewards`, `successes`, write selected action rollout video:

```python
episode_dir = out / "rollouts" / f"update_{update_index:04d}_episode_{episode_index:04d}"
video_path = write_phase12_episode_video(
    video_path=episode_dir / "selected_action_rollout.mp4",
    frames=list(episode.metadata["frames"]),
    rewards=list(episode.metadata["env_rewards"]),
    successes=list(episode.metadata["successes"]),
    fps=6,
    overlay_mode="cumulative_reward",
)
meta["rollout_validation_video"] = str(video_path)
meta["selected_action_rollout_video"] = str(video_path)
```

Write oracle baseline video during one-update smoke:

```python
oracle_dir = out / "oracle"
oracle_video = run_phase12_oracle_baseline_video(
    env=env_h,
    task=args.task,
    seed=int(args.train_seed_base),
    max_steps=int(args.max_steps),
    output_dir=oracle_dir,
    fps=6,
)
meta["oracle_baseline_video"] = str(oracle_video)
meta["oracle_baseline_video_status"] = "ok"
```

`run_phase12_oracle_baseline_video(...)` should:

```python
def run_phase12_oracle_baseline_video(*, env, task: str, seed: int, max_steps: int, output_dir: Path, fps: int) -> Path:
    output_dir = Path(output_dir)
    video_path = output_dir / "oracle_baseline.mp4"
    oracle = rollout_lerobot_expert_episode(
        env=env,
        seed=int(seed),
        max_steps=int(max_steps),
    )
    write_phase12_episode_video(
        video_path=video_path,
        frames=oracle.frames,
        rewards=oracle.rewards,
        successes=oracle.successes,
        fps=int(fps),
        overlay_mode="cumulative_reward",
    )
    oracle.write_manifest(output_dir / "oracle_manifest.json")
    return video_path
```

Why this is the right oracle-video path:

- User preference is parity: oracle baseline and selected SmolVLA rollout should use the same LeRobot-backed env construction.
- Installed LeRobot has scripted policy classes via `lerobot.envs.metaworld.TASK_POLICY_MAPPING`, but no public expert rollout/video API.
- Our adapter should expose exactly what Phase12 needs, not fork a second oracle environment path:
  - full raw observation for `expert_policy.get_action(raw_obs)`,
  - rendered RGB frame,
  - agent proprio,
  - deterministic reset,
  - same camera/flip behavior as selected rollout.
- This avoids MP4 copy IO: writer writes directly to final `oracle_baseline.mp4`.
- Phase06 remains historical reference only, not fallback.

- [ ] **Step 5: Run tests**

Run:

```bash
PYTHONPATH=src /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python -m pytest tests/test_phase12_diagnostics.py tests/test_phase12_artifacts.py -v
```

Expected: pass.

## Task 4: Structured WM Reward Backend

**Files:**
- Create: `src/smolvla_grpo/phase12_wm_reward.py`
- Create: `tests/test_phase12_wm_reward.py`
- Modify: `src/smolvla_grpo/phase12_objective.py` only if shape normalization needed

- [ ] **Step 1: Write failing unit tests with fake WM**

```python
import numpy as np
import torch

from smolvla_grpo.phase12_wm_reward import score_phase12_chunk_with_wm


class FakeWM:
    device = torch.device("cpu")
    proprio_dim = 2
    planner_action_dim = 4

    class Preprocessor:
        action_mean = np.zeros(4, dtype=np.float32)
        action_std = np.ones(4, dtype=np.float32)

    preprocessor = Preprocessor()

    class Model:
        def encode(self, obs):
            return {"visual": obs["visual"] * 0.0, "proprio": obs["proprio"]}

        def unroll(self, z, *, act_suffix, debug=False):
            delta = act_suffix.float().sum().reshape(1, 1, 1)
            return {
                "visual": z["visual"] + delta,
                "proprio": z["proprio"] + delta[..., :1],
            }

    model = Model()


def test_score_phase12_chunk_uses_structured_visual_plus_alpha_proprio() -> None:
    wm = FakeWM()
    score = score_phase12_chunk_with_wm(
        wm_bundle=wm,
        image=np.zeros((8, 8, 3), dtype=np.uint8),
        proprio=np.zeros(2, dtype=np.float32),
        chunk_actions=np.ones((1, 4), dtype=np.float32),
        goal={"visual": torch.zeros(1, 1, 1, 8, 8, 3), "proprio": torch.zeros(1, 1, 2)},
        candidate_index=3,
        proprio_alpha=0.1,
        mode="visual_proprio",
    )
    assert score.candidate_index == 3
    assert score.wm_status == "ok"
    assert score.final_combined_distance == score.final_visual_distance + 0.1 * score.final_proprio_distance
```

Reward sanity:

- This is not "add visual and proprio latents, then calculate distance".
- This is "calculate visual MSE, calculate proprio MSE, then add scalar losses with alpha".
- That matches JEPA-WM MetaWorld planning config:

```yaml
planner:
  planning_objective:
    objective_type: L2
    sum_all_diffs: false
    alpha: 0.1
```

Implementation invariant:

```python
visual_mse = (pred_visual.float() - goal_visual.float()).pow(2).mean()
proprio_mse = (pred_proprio.float() - goal_proprio.float()).pow(2).mean()
combined_distance = visual_mse + 0.1 * proprio_mse
reward = start_combined_distance - final_combined_distance
```

Do not use:

```python
flat_pred = torch.cat([pred_visual.reshape(-1), pred_proprio.reshape(-1)])
flat_goal = torch.cat([goal_visual.reshape(-1), goal_proprio.reshape(-1)])
combined_distance = (flat_pred - flat_goal).pow(2).mean()
```

That flatten/concat path changes modality weighting, hides structure, and no longer matches the official JEPA-WM objective.

- [ ] **Step 2: Run test to verify fail**

Run:

```bash
PYTHONPATH=src /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python -m pytest tests/test_phase12_wm_reward.py -v
```

Expected: fail because module is missing.

- [ ] **Step 3: Implement narrow structured reward backend**

Create `phase12_wm_reward.py`:

```python
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch

from segment_grpo_loop import (
    _infer_env_action_dim,
    _infer_model_action_dim,
    _normalize_env_actions_for_wm,
    _pack_env_actions_for_wm,
    _to_wm_proprio,
    _to_wm_visual,
    _wm_action_block_factor,
)
from smolvla_grpo.phase12_objective import Phase12Score, score_progress, split_structured_latent


def _encode_structured(wm_bundle: Any, image: np.ndarray, proprio: np.ndarray, *, mode: str) -> dict[str, torch.Tensor]:
    obs = {
        "visual": _to_wm_visual(image, wm_bundle.device),
        "proprio": _to_wm_proprio(proprio, int(wm_bundle.proprio_dim), wm_bundle.device),
    }
    with torch.no_grad():
        encoded = wm_bundle.model.encode(obs)
    return split_structured_latent(encoded, mode=mode)


def _final_structured_after_unroll(wm_bundle: Any, start_latent: Mapping[str, torch.Tensor], actions: np.ndarray, *, mode: str) -> dict[str, torch.Tensor]:
    env_dim = _infer_env_action_dim(wm_bundle, actions)
    model_action_dim = _infer_model_action_dim(wm_bundle.model)
    wm_dim = int(model_action_dim) if model_action_dim else int(wm_bundle.planner_action_dim)
    factor = _wm_action_block_factor(env_dim, wm_dim)
    normalized = _normalize_env_actions_for_wm(wm_bundle.preprocessor, actions[:, :env_dim], env_dim, wm_bundle.device)
    packed = _pack_env_actions_for_wm(normalized, factor, env_dim, wm_dim)
    action_t = torch.from_numpy(packed).to(wm_bundle.device).float().unsqueeze(1)
    latent: Any = start_latent
    with torch.no_grad():
        for t in range(int(action_t.shape[0])):
            latent = wm_bundle.model.unroll(latent, act_suffix=action_t[t : t + 1], debug=False)
    return split_structured_latent(latent, mode=mode)


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
) -> Phase12Score:
    actions = np.asarray(chunk_actions, dtype=np.float32)
    if actions.ndim != 2:
        raise ValueError(f"chunk_actions must be 2D, got {actions.shape}")
    start = _encode_structured(wm_bundle, image, proprio, mode=mode)
    final = _final_structured_after_unroll(wm_bundle, start, actions, mode=mode)
    return score_progress(
        candidate_index=int(candidate_index),
        start=start,
        final=final,
        goal=goal,
        proprio_alpha=float(proprio_alpha),
        mode=mode,
        debug_npz_path=debug_npz_path,
    )
```

- [ ] **Step 4: Add test that concat path is not used**

```python
def test_phase12_wm_reward_module_does_not_import_score_chunk_by_goal_latent() -> None:
    source = Path("src/smolvla_grpo/phase12_wm_reward.py").read_text(encoding="utf-8")
    assert "score_chunk_by_goal_latent" not in source
    assert "wm_scoring_latent" not in source
```

- [ ] **Step 5: Run tests**

Run:

```bash
PYTHONPATH=src /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python -m pytest tests/test_phase12_wm_reward.py tests/test_phase12_objective.py -v
```

Expected: pass.

## Task 5: Action Profiles In Candidate Sampling

**Files:**
- Modify: `src/smolvla_grpo/phase12_rollout.py`
- Modify: `tests/test_phase12_rollout.py`
- Use: `src/smolvla_grpo/phase12_actions.py`

- [ ] **Step 1: Write failing rollout action profile test**

```python
def test_collect_phase12_episode_applies_action_profile_before_score_and_env() -> None:
    seen_wm = []
    seen_env = []

    class DummyEnv:
        action_space = type("A", (), {"low": np.full((4,), -1.0), "high": np.full((4,), 1.0)})()
        def reset(self):
            return {"id": "root"}
        def step(self, action):
            seen_env.append(np.asarray(action).copy())
            return {"id": "next"}, 0.0, True, {}

    def sampler(*_args, **_kwargs):
        yield {
            "candidate_index": 0,
            "unsquashed_chunk": np.zeros((1, 4), dtype=np.float32),
            "old_logprob_steps": np.zeros(1, dtype=np.float32),
            "exec_actions_raw_postprocessed": np.array([[2.0, -2.0, 0.5, 1.5]], dtype=np.float32),
        }

    def score_fn(_root, candidate, _goal, **_kwargs):
        seen_wm.append(candidate.exec_actions_for_wm.copy())
        return _score(candidate_index=0, progress=1.0, final_distance=0.0)

    collect_phase12_episode(
        env=DummyEnv(),
        policy_sampler=sampler,
        score_fn=score_fn,
        goals=["goal"],
        num_candidates=1,
        action_profile="bounded_executed",
    )

    np.testing.assert_allclose(seen_wm[0], np.array([[1.0, -1.0, 0.5, 1.0]], dtype=np.float32))
    np.testing.assert_allclose(seen_env[0], np.array([1.0, -1.0, 0.5, 1.0], dtype=np.float32))
```

- [ ] **Step 2: Run test to verify fail**

Run:

```bash
PYTHONPATH=src /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python -m pytest tests/test_phase12_rollout.py::test_collect_phase12_episode_applies_action_profile_before_score_and_env -v
```

Expected: fail because `collect_phase12_episode(...)` does not accept/apply action profile.

- [ ] **Step 3: Extend `Phase12Candidate`**

Use:

```python
@dataclass(frozen=True)
class Phase12Candidate:
    candidate_index: int
    proc_root_snapshot: Any
    unsquashed_chunk: Any
    old_logprob_steps: Any
    old_logprob_sum: float
    exec_actions_raw_postprocessed: Any
    exec_actions_clipped: Any
    exec_actions_for_env: Any
    exec_actions_for_wm: Any
    action_metadata: dict[str, Any]
```

- [ ] **Step 4: Apply profile in `_candidate_from_sample(...)`**

Add args: `action_profile`, `action_low`, `action_high`, `preprocessor`, `env_action_dim`, `wm_action_dim`.

```python
raw = _optional_field(sample, "exec_actions_raw_postprocessed", _optional_field(sample, "exec_action_np", None))
if raw is None:
    raw = _optional_field(sample, "exec_actions_for_env", _optional_field(sample, "unsquashed_chunk", None))
profile = apply_phase12_action_profile(
    np.asarray(raw, dtype=np.float32),
    action_profile=action_profile,
    action_low=action_low,
    action_high=action_high,
    preprocessor=preprocessor,
    env_action_dim=env_action_dim,
    wm_action_dim=wm_action_dim,
)
return Phase12Candidate(
    candidate_index=candidate_index,
    proc_root_snapshot=_optional_field(sample, "proc_root_snapshot", root_snapshot),
    unsquashed_chunk=_optional_field(sample, "unsquashed_chunk", None),
    old_logprob_steps=old_logprob_steps,
    old_logprob_sum=float(_optional_field(sample, "old_logprob_sum", _sum_logprobs(old_logprob_steps))),
    exec_actions_raw_postprocessed=profile.exec_actions_raw_postprocessed,
    exec_actions_clipped=profile.exec_actions_clipped,
    exec_actions_for_env=profile.exec_actions_for_env,
    exec_actions_for_wm=profile.exec_actions_for_wm,
    action_metadata={**profile.metadata, **dict(_optional_field(sample, "action_metadata", {}) or {})},
)
```

- [ ] **Step 5: Run rollout/action tests**

Run:

```bash
PYTHONPATH=src /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python -m pytest tests/test_phase12_rollout.py tests/test_phase12_actions.py -v
```

Expected: pass.

## Task 6: Real GRPO Training Loop

**Files:**
- Modify: `scripts/grpo/train_phase12_wm_chunk_grpo.py`
- Modify: `src/smolvla_grpo/phase12_rollout.py`
- Create: `tests/test_phase12_training_loop.py`

- [ ] **Step 1: Write mocked training test proving non-empty checkpoint and update row**

```python
def test_wm_grpo_train_writes_update_row_and_nonempty_checkpoint(monkeypatch, tmp_path):
    import torch
    from scripts.grpo import train_phase12_wm_chunk_grpo as trainer

    class TinyPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.log_std = torch.nn.Parameter(torch.zeros(1, 4))
            self.w = torch.nn.Parameter(torch.zeros(1))
        def forward(self):
            return self.w

    class TinyBundle:
        device = torch.device("cpu")
        policy = TinyPolicy()
        preprocessor = object()
        postprocessor = lambda self, x: x
        obs_image_key = "observation.image"
        obs_state_key = "observation.state"
        obs_env_state_key = "observation.environment_state"

    monkeypatch.setattr(trainer, "load_phase12_train_resources", lambda args: (TinyBundle(), object(), 4))
    monkeypatch.setattr(trainer, "collect_phase12_training_episode", lambda **kwargs: kwargs["fake_episode"])

    code = trainer.main([
        "--mode", "wm_grpo_train",
        "--output-dir", str(tmp_path),
        "--jepa-repo", "/tmp/jepa",
        "--jepa-ckpt", "wm.pt",
        "--num-updates", "1",
        "--num-episodes", "1",
    ])

    assert code == 0
    rows = [json.loads(x) for x in (tmp_path / "progress.jsonl").read_text().splitlines() if x.strip()]
    assert any(row.get("event") == "update_complete" and row.get("optimizer_step") is True for row in rows)
    ckpt = torch.load(tmp_path / "checkpoints" / "latest.pt", map_location="cpu", weights_only=False)
    assert ckpt["policy_state_dict"]
    assert ckpt["optimizer_state_dict"]
```

Adjust mock names after Step 3; keep assertions.

- [ ] **Step 2: Run test to verify fail**

Run:

```bash
PYTHONPATH=src /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python -m pytest tests/test_phase12_training_loop.py -v
```

Expected: fail because training resource/loop functions are missing.

- [ ] **Step 3: Add CLI training args**

Add:

```python
p.add_argument("--lr", type=float, default=1e-5)
p.add_argument("--clip-eps", type=float, default=0.2)
p.add_argument("--grad-clip", type=float, default=1.0)
p.add_argument("--update-epochs", type=int, default=1)
p.add_argument("--init-log-std", type=float, default=-2.0)
p.add_argument("--euler-step-noise-std", type=float, default=0.2)
p.add_argument("--save-every", type=int, default=5)
p.add_argument("--resume", type=Path, default=None)
p.add_argument("--train-seed-base", type=int, default=2000)
p.add_argument("--allow-wm-fallback", action="store_true", default=False)
```

Add train-mode validation:

```python
def _validate_train_mode_args(args: argparse.Namespace) -> None:
    if args.mode != "wm_grpo_train":
        return
    if int(args.num_episodes) != int(args.num_updates):
        raise SystemExit(
            "wm_grpo_train requires --num-episodes == --num-updates. "
            "Each update collects one real episode, matching Phase111."
        )
    if bool(args.allow_wm_fallback):
        raise SystemExit("wm_grpo_train refuses --allow-wm-fallback for production smoke/training.")
```

- [ ] **Step 4: Add resource loader**

```python
def load_phase12_train_resources(args: argparse.Namespace):
    import torch
    from segment_grpo_loop import load_smolvla_bundle, load_wm_bundle

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wm_bundle = load_wm_bundle(args.jepa_repo, args.jepa_ckpt, device, required=True)
    smolvla_bundle = load_smolvla_bundle(args.checkpoint, device=device, n_action_steps=int(args.chunk_len))
    action_dim = 4
    return smolvla_bundle, wm_bundle, action_dim
```

If env exposes action dim, replace hardcoded `4` by reading official guarded env once.

- [ ] **Step 5: Add train wrapper setup**

```python
def build_train_wrapper(args, bundle, action_dim):
    from smolvla_pipeline.evaluator import _resolve_camera_name, _resolve_flip_corner2, _resolve_task_text
    from smolvla_grpo.policy_wrapper import MetaWorldSmolVLAGRPOPolicy, freeze_all_but_grpo_trainables

    wrapper = MetaWorldSmolVLAGRPOPolicy(
        bundle,
        task=args.task,
        task_text=_resolve_task_text(args.task),
        camera_name=_resolve_camera_name(),
        flip_corner2=_resolve_flip_corner2(),
        action_dim=int(action_dim),
        action_transform=args.action_transform,
    )
    wrapper.assert_grpo_api()
    wrapper.set_log_std(args.init_log_std)
    wrapper.set_euler_step_noise_std(args.euler_step_noise_std)
    trainable = freeze_all_but_grpo_trainables(bundle.policy)
    if not trainable:
        raise RuntimeError("No trainable GRPO parameters after freeze_all_but_grpo_trainables.")
    return wrapper, trainable
```

Add an old-policy wrapper helper that does not freeze/set train params again:

```python
def build_old_wrapper(args, bundle, old_policy, action_dim):
    from types import SimpleNamespace
    from smolvla_pipeline.evaluator import _resolve_camera_name, _resolve_flip_corner2, _resolve_task_text
    from smolvla_grpo.policy_wrapper import MetaWorldSmolVLAGRPOPolicy

    old_bundle = SimpleNamespace(
        policy=old_policy,
        preprocessor=bundle.preprocessor,
        postprocessor=bundle.postprocessor,
        device=bundle.device,
        obs_image_key=bundle.obs_image_key,
        obs_state_key=bundle.obs_state_key,
        obs_env_state_key=bundle.obs_env_state_key,
    )
    return MetaWorldSmolVLAGRPOPolicy(
        old_bundle,
        task=args.task,
        task_text=_resolve_task_text(args.task),
        camera_name=_resolve_camera_name(),
        flip_corner2=_resolve_flip_corner2(),
        action_dim=int(action_dim),
        action_transform=args.action_transform,
        policy_module=old_policy,
    )
```

- [ ] **Step 6: Add optimizer loop**

Optimizer cadence is Phase111-style:

- Freeze old policy for the whole update.
- Collect one complete receding-horizon episode under that old policy.
- During collection, for each segment:
  - sample `K` chunks from old policy,
  - score `K` chunks with JEPA-WM,
  - store that segment's `K` rewards,
  - select best chunk,
  - execute best chunk in real env,
  - refresh image/proprio from real env.
- After episode collection:
  - compute group advantages per segment,
  - concatenate segment-candidate records,
  - recompute new logprobs with current train policy,
  - run one optimizer step,
  - sync old policy from updated policy,
  - save checkpoint.
- Do not step optimizer between segments.
- Do not sync old policy mid-episode.
- Do not train before executing the selected chunk.

Core shape:

```python
def run_wm_grpo_train(args: argparse.Namespace, out: Path) -> int:
    import copy
    import time
    import torch
    from torch import nn
    from smolvla_grpo.checkpointing import load_grpo_checkpoint, save_grpo_checkpoint
    from smolvla_grpo.grpo_math import compute_group_advantages
    from smolvla_grpo.phase12_rollout import chunk_grpo_loss

    bundle, wm_bundle, action_dim = load_phase12_train_resources(args)
    train_wrapper, trainable = build_train_wrapper(args, bundle, action_dim)
    optimizer = torch.optim.AdamW(trainable, lr=float(args.lr), betas=(0.9, 0.95))
    _validate_train_mode_args(args)
    start_update = 0
    if args.resume is not None:
        ckpt = load_grpo_checkpoint(args.resume, map_location=bundle.device)
        bundle.policy.load_state_dict(ckpt["policy_state_dict"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_update = int(ckpt["update_index"]) + 1

    old_policy = copy.deepcopy(bundle.policy).eval().to(bundle.device)
    old_wrapper = build_old_wrapper(args, bundle, old_policy, action_dim)
    end_update = start_update + int(args.num_updates)
    for update_index in range(start_update, end_update):
        update_t0 = time.perf_counter()
        reset_seed = int(args.train_seed_base) + int(update_index)
        episode = collect_phase12_training_episode(
            args=args,
            bundle=bundle,
            wm_bundle=wm_bundle,
            old_policy=old_policy,
            old_wrapper=old_wrapper,
            action_dim=action_dim,
            update_index=update_index,
            reset_seed=reset_seed,
            output_dir=out,
        )
        segment_rewards = [
            torch.tensor(row, dtype=torch.float32, device=bundle.device)
            for row in episode.metadata["segment_candidate_rewards"]
        ]
        segment_advantages = [compute_group_advantages(row) for row in segment_rewards]
        if int(args.num_updates) == 1 and all(torch.allclose(a, torch.zeros_like(a)) for a in segment_advantages):
            raise RuntimeError("one-update WM-GRPO smoke produced zero advantages for every segment")
        rewards = torch.cat(segment_rewards, dim=0)
        advantages = torch.cat(segment_advantages, dim=0)
        old_lp = torch.tensor(episode.metadata["old_logprob_sums"], dtype=torch.float32, device=bundle.device)
        new_lp_rows = []
        for proc, chunk in zip(episode.metadata["proc_root_snapshots"], episode.metadata["unsquashed_chunks"], strict=True):
            new_lp_rows.append(train_wrapper.get_action_probs_for_chunk_from_proc(proc, chunk).sum())
        new_lp = torch.stack(new_lp_rows)
        if torch.allclose(advantages, torch.zeros_like(advantages)):
            write_jsonl_row(out / "progress.jsonl", {
                "created_at": utc_now_iso(),
                "event": "update_complete",
                "mode": "wm_grpo_train",
                "update_index": int(update_index),
                "reset_seed": int(reset_seed),
                "skipped": True,
                "reason": "zero_advantages",
                "optimizer_step": False,
            })
            continue
        loss, ratio_stats = chunk_grpo_loss(old_lp, new_lp, advantages, clip_eps=float(args.clip_eps))
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(trainable, float(args.grad_clip))
        optimizer.step()
        old_policy.load_state_dict({k: v.detach().clone() for k, v in bundle.policy.state_dict().items()})
        old_policy.eval()
        old_wrapper._policy = old_policy
        ckpt_path = out / "checkpoints" / "latest.pt"
        save_grpo_checkpoint(
            ckpt_path,
            policy_state=bundle.policy.state_dict(),
            optimizer_state=optimizer.state_dict(),
            update_index=update_index,
            args=vars(args),
            extra={"loss": float(loss.detach().cpu()), **ratio_stats},
        )
        write_jsonl_row(out / "progress.jsonl", {
            "created_at": utc_now_iso(),
            "event": "update_complete",
            "mode": "wm_grpo_train",
            "update_index": int(update_index),
            "reset_seed": int(reset_seed),
            "optimizer_step": True,
            "loss": float(loss.detach().cpu()),
            "advantages": advantages.detach().cpu().tolist(),
            "returns": rewards.detach().cpu().tolist(),
            "segment_candidate_rewards": [row.detach().cpu().tolist() for row in segment_rewards],
            "segment_advantages": [row.detach().cpu().tolist() for row in segment_advantages],
            "checkpoint_path": str(ckpt_path),
            "update_seconds": float(time.perf_counter() - update_t0),
            **ratio_stats,
        })
    print("PHASE12_WM_CHUNK_GRPO_TRAIN_DONE", f"updates={int(args.num_updates)}", f"out={out}", flush=True)
    return 0
```

- [ ] **Step 7: Route main by mode**

```python
if args.mode == "wm_grpo_train":
    return run_wm_grpo_train(args, out)
return run_rollout_validation(args, out)
```

- [ ] **Step 8: Run training tests**

Run:

```bash
PYTHONPATH=src /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python -m pytest tests/test_phase12_training_loop.py tests/test_phase12_trainer_static.py tests/test_grpo_checkpointing.py -v
```

Expected: pass.

## Task 7: Real Phase12 Episode Collection

**Files:**
- Modify: `scripts/grpo/train_phase12_wm_chunk_grpo.py`
- Modify: `src/smolvla_grpo/phase12_rollout.py`
- Modify: `src/smolvla_grpo/phase12_goals.py`
- Modify: `src/smolvla_grpo/lerobot_metaworld_adapter.py`
- Use: `src/smolvla_grpo/phase12_wm_reward.py`

Environment contract for this task:

- Build and step env using Phase111 official-LeRobot-style path wherever possible.
- Use one real LeRobot-backed env per update for Phase12 collection.
- Do not create `group_size` parallel envs for Phase12 candidate groups.
- `group_size=4` means four candidate chunks sampled from the same current real observation/proprio root.
- Capture render/proprio for WM using Phase8 JEPA parity helpers after each executed env step.
- Do not use Phase8 `rollout_with_chunks(...)` for training collection.
- Each update uses one deterministic seed: `train_seed_base + update_index`.
- Each segment starts from a fresh real observation after executing previous selected chunk.
- Frame/reward/success arrays must be captured for selected-action rollout video:
  - `episode.metadata["frames"]`
  - `episode.metadata["env_rewards"]`
  - `episode.metadata["successes"]`

Oracle goal/cache contract:

- For each update seed, create/cache an oracle bundle under `out / "oracle" / f"seed_{reset_seed}"`.
- Cache includes:
  - oracle video,
  - oracle frames,
  - oracle actions for diagnostics only,
  - oracle flat proprio JSONL,
  - encoded structured goal latents for scheduled subgoals.
- If cache exists and manifest matches task/seed/max_steps/chunk_len, reuse it.
- If cache is missing/stale, regenerate before collection.
- Training smoke fails if oracle goal collection or oracle video generation fails.
- Oracle action rows must not be used as policy training candidates.
- Oracle gives "where to go", not "what action to take".
- Goal latents must be encoded from oracle frame and oracle proprio at the same frame.
- Do not use fallback reset proprio for goal latents in train mode.
- Subgoal schedule:
  - frame `25`, `50`, `75`, ...
  - include first success frame if oracle succeeds before or between scheduled frames,
  - otherwise include final frame.
- Segment target schedule:
  - segment 0 targets frame 25,
  - segment 1 targets frame 50,
  - segment 2 targets frame 75,
  - continue by fixed schedule even if selected SmolVLA chunk undershoots/overshoots,
  - keep closest-subgoal retargeting for future ablation.
- Reset parity mismatch between oracle start and training env start fails train mode.

LeRobot-backed oracle baseline contract:

- Extend `OfficialLeRobotMetaWorldGRPORollout` or `DeferredLeRobotMetaworldEnv` with a minimal Phase12 expert interface.
- Do not patch installed LeRobot site-packages.
- Patch only local `src/smolvla_grpo/lerobot_metaworld_adapter.py`.
- Add regression tests in `tests/test_grpo_lerobot_adapter.py` before implementation:
  - `test_deferred_metaworld_env_stores_raw_obs_for_expert_action`,
  - `test_official_adapter_expert_oracle_uses_deferred_single_env`,
  - `test_official_adapter_default_path_still_uses_make_env`.
- `OfficialLeRobotMetaWorldGRPORollout(n_envs=1)` currently uses installed LeRobot `make_env(...)`, so add explicit construction flag:

```python
OfficialLeRobotMetaWorldGRPORollout(
    task=task,
    n_envs=1,
    enable_expert_oracle=True,
)
```

- When `enable_expert_oracle=True`, build a one-env vector wrapper around `DeferredLeRobotMetaworldEnv` instead of installed `make_env(...)`.
- Store full raw MetaWorld observation from reset/step before formatting to `pixels`/`agent_pos`.
- Add raw-observation/expert methods to `DeferredLeRobotMetaworldEnv`:

```python
def expert_action(self) -> np.ndarray:
    if self._last_raw_obs is None:
        raise RuntimeError("expert_action called before reset")
    return np.asarray(self.expert_policy.get_action(self._last_raw_obs), dtype=np.float32)


def last_agent_pos(self) -> np.ndarray:
    if self._last_raw_obs is None:
        raise RuntimeError("last_agent_pos called before reset")
    return np.asarray(self._last_raw_obs[:4], dtype=np.float32)


def render_frame(self) -> np.ndarray:
    return self.render()
```

- Access these through vector env `call(...)`:

```python
action = env_h.vec_env.call("expert_action")[0]
frame = env_h.vec_env.call("render_frame")[0]
proprio = env_h.vec_env.call("last_agent_pos")[0]
```

- Oracle baseline generation:
  - reset env with `reset_seed`,
  - record initial frame/proprio/raw obs,
  - loop `max_steps`:
    - action = expert action from stored raw obs,
    - clip to env action bounds,
    - step same LeRobot-backed env,
    - record frame/proprio/action/reward/success,
  - write oracle video directly to final path,
  - encode subgoal latents from recorded oracle frames/proprio.
- After oracle baseline collection, reset the same env with same `reset_seed`.
- Compare reset frame/proprio with cached oracle initial frame/proprio.
- If mismatch exceeds thresholds, fail train/smoke.
- Continue Phase12 selected rollout from that reset state.

- [ ] **Step 1: Add `Phase12Goal`**

```python
@dataclass(frozen=True)
class Phase12Goal:
    subgoal_index: int
    frame_index_1based: int
    frame_path: Path
    companion_frame_index_1based: int | None
    companion_frame_path: Path | None
    proprio: np.ndarray
    goal_visual: Any
    goal_proprio: Any
    source: str
```

- [ ] **Step 2: Add policy sampler for old policy**

```python
def make_old_policy_sampler(old_wrapper, *, chunk_len: int, action_profile: str, action_low, action_high, preprocessor, env_action_dim: int, wm_action_dim: int):
    def sampler(root_observation, *, root_id, num_candidates, segment_index, goal):
        proc = root_observation["proc"]
        for candidate_index in range(int(num_candidates)):
            gen = torch.Generator(device=old_wrapper.bundle.device)
            gen.manual_seed(int(root_observation["seed"]) * 1000003 + segment_index * 7919 + candidate_index)
            sample = old_wrapper.sample_action_chunk_from_proc(proc, chunk_len=int(chunk_len), rng=gen)
            yield {
                "candidate_index": candidate_index,
                "proc_root_snapshot": proc,
                "unsquashed_chunk": sample.unsquashed_chunk,
                "old_logprob_steps": sample.log_prob_steps,
                "old_logprob_sum": sample.log_prob_sum,
                "exec_actions_raw_postprocessed": sample.exec_action_np,
                "action_metadata": {
                    "sample_clip_fraction_mean": float(np.mean(sample.action_clip_fraction)),
                    "sample_clip_any_fraction": float(np.mean(sample.action_clip_any)),
                    "unique_action_rows": int(sample.unique_action_rows),
                },
            }
    return sampler
```

- [ ] **Step 3: Add score function**

```python
def make_phase12_score_fn(wm_bundle, *, proprio_alpha: float, mode: str, reward_key: str):
    from smolvla_grpo.phase12_wm_reward import score_phase12_chunk_with_wm

    def score_fn(root_observation, candidate, goal, *, root_id, segment_index):
        goal_latent = {"visual": goal.goal_visual}
        if mode == "visual_proprio":
            goal_latent["proprio"] = goal.goal_proprio
        try:
            return score_phase12_chunk_with_wm(
                wm_bundle=wm_bundle,
                image=root_observation["image"],
                proprio=root_observation["proprio"],
                chunk_actions=candidate.exec_actions_for_wm,
                goal=goal_latent,
                candidate_index=candidate.candidate_index,
                proprio_alpha=float(proprio_alpha),
                mode=mode,
            )
        except Exception:
            # Train mode must fail loudly. No action-norm fallback.
            raise
    return score_fn
```

WM scoring/carry invariant:

- Score candidate chunks one at a time by default.
- Each candidate score starts from the same encoded root state:
  - `root_observation["image"]`,
  - `root_observation["proprio"]`,
  - same scheduled oracle `goal`.
- Do not carry predicted WM state from candidate 0 into candidate 1.
- Do not carry predicted WM state across real segments as truth.
- For each candidate:
  - encode real root image/proprio into structured JEPA latent,
  - unroll that candidate's action chunk inside WM,
  - compute final structured distance to goal,
  - store score trace/debug paths on the candidate/segment record.
- After best chunk executes in real env:
  - discard all candidate predicted WM latents,
  - render/read new real env image/proprio,
  - next segment re-encodes from that real observation.
- Store between segments:
  - real env observation/proprio/frame,
  - policy `proc` snapshot for logprob recomputation,
  - candidate actions/logprobs/scores,
  - selected candidate index,
  - WM debug artifacts/NPZ paths.
- Do not store predicted WM latent as authoritative rollout state.

- [ ] **Step 4: Return training metadata for optimizer**

After `collect_phase12_episode(...)`, flatten:

```python
def phase12_episode_training_metadata(episode, reward_key: str) -> dict[str, Any]:
    candidates = [candidate for segment in episode.segments for candidate in segment.candidates]
    scores = [score for segment in episode.segments for score in segment.scores]
    segment_candidate_rewards = [
        [float(getattr(score, reward_key)) for score in segment.scores]
        for segment in episode.segments
    ]
    return {
        "segment_candidate_rewards": segment_candidate_rewards,
        "candidate_rewards": [float(getattr(score, reward_key)) for score in scores],
        "old_logprob_sums": [float(candidate.old_logprob_sum) for candidate in candidates],
        "proc_root_snapshots": [candidate.proc_root_snapshot for candidate in candidates],
        "unsquashed_chunks": [candidate.unsquashed_chunk for candidate in candidates],
        "segment_candidate_counts": [len(segment.candidates) for segment in episode.segments],
        "wm_status_counts": _count_values(getattr(score, "wm_status", "unknown") for score in scores),
        "action_clip_fraction": float(np.mean([candidate.action_metadata.get("clip_fraction", 0.0) for candidate in candidates])),
        "action_clip_any_fraction": float(np.mean([candidate.action_metadata.get("clip_any", False) for candidate in candidates])),
    }
```

Invariant:

- `segment_candidate_rewards[i]` length must equal `K`.
- GRPO advantages are normalized within each row of `segment_candidate_rewards`.
- Flattened `candidate_rewards` are for logging only, not for global advantage normalization.

- [ ] **Step 5: Run rollout and training tests**

Run:

```bash
PYTHONPATH=src /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python -m pytest tests/test_phase12_rollout.py tests/test_phase12_training_loop.py tests/test_phase12_goals.py -v
```

Expected: pass.

## Task 8: One-Update Real Training Validation Smoke

**Files:**
- Modify: `scripts/grpo/train_phase12_wm_chunk_grpo.py`
- Modify: `tests/test_phase12_trainer_static.py`

- [ ] **Step 1: Write test that one-update validation uses real training path**

```python
def test_one_update_training_validation_saves_real_checkpoint(monkeypatch, tmp_path):
    ckpt_path = tmp_path / "checkpoints" / "latest.pt"
    episode = SimpleNamespace(
        total_env_reward=1.0,
        success_any=False,
        success_last=False,
        metadata={
            "candidate_rewards": [0.0, 1.0, 2.0, 3.0],
            "old_logprob_sums": [-1.0, -1.1, -1.2, -1.3],
            "proc_root_snapshots": ["p0", "p1", "p2", "p3"],
            "unsquashed_chunks": ["u0", "u1", "u2", "u3"],
            "rollout_validation_video": str(tmp_path / "rollout_validation.mp4"),
            "selected_action_rollout_video": str(tmp_path / "rollout_validation.mp4"),
            "oracle_baseline_video": str(tmp_path / "oracle_baseline.mp4"),
            "oracle_baseline_video_status": "ok",
            "wm_decode_status": "failed",
        },
    )
    (tmp_path / "rollout_validation.mp4").write_bytes(b"mp4")
    (tmp_path / "oracle_baseline.mp4").write_bytes(b"oracle")
    monkeypatch.setattr(trainer, "collect_phase12_training_episode", lambda **_kwargs: episode)
    code = main([
        "--mode", "wm_grpo_train",
        "--output-dir", str(tmp_path),
        "--jepa-repo", "/tmp/jepa",
        "--jepa-ckpt", "wm.pt",
        "--num-episodes", "1",
        "--num-updates", "1",
    ])
    assert code == 0
    assert ckpt_path.exists()
    rows = [json.loads(x) for x in (tmp_path / "progress.jsonl").read_text().splitlines() if x.strip()]
    assert rows[-1]["event"] == "update_complete"
    assert rows[-1]["optimizer_step"] is True
```

- [ ] **Step 2: Training smoke marker**

Use:

```python
print("PHASE12_WM_CHUNK_GRPO_TRAIN_DONE", f"updates={int(args.num_updates)}", f"out={out}", flush=True)
```

- [ ] **Step 3: One-update artifact/checkpoint criteria**

For `--mode wm_grpo_train --num-episodes 1 --num-updates 1`, require:

- one `update_complete` row,
- `optimizer_step=true`,
- real `checkpoints/latest.pt`,
- non-empty `policy_state_dict`,
- non-empty `optimizer_state_dict`,
- `rollout_validation_video` exists and non-empty,
- `selected_action_rollout_video` exists and non-empty,
- `oracle_baseline_video` exists and non-empty,
- no `not_implemented` event,
- no rollout-validation marker.

- [ ] **Step 4: Run tests**

Run:

```bash
PYTHONPATH=src /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python -m pytest tests/test_phase12_trainer_static.py tests/test_phase12_artifacts.py -v
```

Expected: pass.

## Task 9: Phase111-Style Slurm Training Contract

**Files:**
- Create: `scripts/grpo/submit_phase12_wm_chunk_grpo_train.slurm`
- Modify: `tests/test_phase12_slurm_static.py`
- Modify: `pytest.ini`

- [ ] **Step 1: Write failing static assertions**

```python
def _body_without_sbatch_header(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if not line.startswith("#SBATCH"))


def test_train_slurm_contracts() -> None:
    text = _read("submit_phase12_wm_chunk_grpo_train.slurm")
    assert "#SBATCH --gres=gpu:1" in text
    assert "#SBATCH --export=NIL" in text
    assert 'slurm_resolve_project_root "scripts/grpo/train_phase12_wm_chunk_grpo.py"' in text
    assert "slurm_export_pythonpath" in text
    assert 'slurm_export_hf_torch_cache "phase12-wm-chunk-grpo-train"' in text
    assert 'export PATH="${PATH:-/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin}"' in text
    assert 'export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"' in text
    assert 'export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"' in text
    assert 'export JEPA_WM_DISABLE_IMAGE_HEAD="${JEPA_WM_DISABLE_IMAGE_HEAD:-0}"' in text
    assert "--mode wm_grpo_train" in text
    assert 'ACTION_PROFILE="${1:-${PHASE12_ACTION_PROFILE:-official_jepa_mirror}}"' in text
    assert 'UPDATES="${2:-${PHASE12_NUM_UPDATES:-100}}"' in text
    assert 'NUM_EPISODES="${PHASE12_NUM_EPISODES:-${UPDATES}}"' in text
    assert "--num-updates \"${UPDATES}\"" in text
    assert "--num-episodes \"${NUM_EPISODES}\"" in text
    assert "--max-steps \"${MAX_STEPS}\"" in text
    assert "--save-every \"${SAVE_EVERY}\"" in text
    assert "--lr \"${LR}\"" in text
    assert "--init-log-std \"${INIT_LOG_STD}\"" in text
    assert "--euler-step-noise-std \"${EULER_NOISE}\"" in text
    assert "PHASE12_WM_CHUNK_GRPO_TRAIN_DONE" in text
    assert "PHASE12_WM_CHUNK_ROLLOUT_VALIDATION_DONE" not in text
    assert "submit_phase12_wm_chunk_rollout_validation_10ep" not in text
    assert "sbatch" not in _body_without_sbatch_header(text)
```

- [ ] **Step 2: Run test to verify fail**

Run:

```bash
PYTHONPATH=src /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python -m pytest tests/test_phase12_slurm_static.py -v
```

Expected: fail because new train script/defaults do not exist.

- [ ] **Step 3: Add Phase111-style training script**

Use `scripts/grpo/submit_phase111_single_task_grpo.slurm` as template. Phase12 script must mirror these Phase111 defaults:

- `NUM_UPDATES=100`
- `GROUP_SIZE=4`
- `SAVE_EVERY=5`
- `MAX_STEPS=120`
- `SEED_BASE=2000`
- `LR=1e-5`
- `INIT_LOG_STD=-2.0`
- `EULER_NOISE=0.2`
- `ACTION_TRANSFORM=no_tanh`
- one GPU
- `--export=NIL`
- local checkpoint snapshot path

Base content:

```bash
#!/usr/bin/env bash
#SBATCH --job-name=phase12-wm-grpo-train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=phase12_wm_chunk_grpo_train_%j.out
#SBATCH --error=phase12_wm_chunk_grpo_train_%j.err
#SBATCH --export=NIL

set -euo pipefail
_PROJECT_FALLBACK="/vol/bitbucket/aa6622/project"
source "${_PROJECT_FALLBACK}/scripts/slurm/common_env.sh"
slurm_resolve_project_root "scripts/grpo/train_phase12_wm_chunk_grpo.py"
cd "${PROJECT_ROOT}"
slurm_export_pythonpath
slurm_export_hf_torch_cache "phase12-wm-chunk-grpo-train"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PATH="${PATH:-/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export JEPA_WM_DISABLE_IMAGE_HEAD="${JEPA_WM_DISABLE_IMAGE_HEAD:-0}"

GRPO_PYTHON="${GRPO_PYTHON:-/vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python}"
CHECKPOINT="${PHASE12_CHECKPOINT:-/vol/bitbucket/aa6622/.cache/huggingface/hub/models--jadechoghari--smolvla_metaworld/snapshots/ef3089ecb84eeeb7d33fedab24f6c76180a68900}"
JEPA_CKPT="${PHASE12_JEPA_CKPT:-jepa_wm_metaworld.pth.tar}"
JEPA_REPO="${PHASE12_JEPA_REPO:-/vol/bitbucket/aa6622/VGG JEPA/jepa-wms}"
ACTION_PROFILE="${1:-${PHASE12_ACTION_PROFILE:-official_jepa_mirror}}"
UPDATES="${2:-${PHASE12_NUM_UPDATES:-100}}"
NUM_EPISODES="${PHASE12_NUM_EPISODES:-${UPDATES}}"
GROUP_SIZE="${PHASE12_GROUP_SIZE:-4}"
SAVE_EVERY="${PHASE12_SAVE_EVERY:-5}"
MAX_STEPS="${PHASE12_MAX_STEPS:-120}"
SEED_BASE="${PHASE12_SEED_BASE:-2000}"
LR="${PHASE12_LR:-1e-5}"
INIT_LOG_STD="${PHASE12_INIT_LOG_STD:--2.0}"
EULER_NOISE="${PHASE12_EULER_NOISE:-0.2}"
ACTION_TRANSFORM="${PHASE12_ACTION_TRANSFORM:-no_tanh}"
STRICT_DECODE=()
if [[ "${UPDATES}" == "1" ]]; then
  STRICT_DECODE+=(--strict-decode)
fi
OUT="${PHASE12_OUT:-${PROJECT_ROOT}/artifacts/phase12_wm_chunk_grpo_train/push-v3/g${GROUP_SIZE}_u${UPDATES}_seed${SEED_BASE}_${ACTION_PROFILE}}"

"${GRPO_PYTHON}" scripts/grpo/train_phase12_wm_chunk_grpo.py \
  --mode wm_grpo_train \
  --checkpoint "${CHECKPOINT}" \
  --jepa-ckpt "${JEPA_CKPT}" \
  --jepa-repo "${JEPA_REPO}" \
  --output-dir "${OUT}" \
  --action-profile "${ACTION_PROFILE}" \
  --num-episodes "${NUM_EPISODES}" \
  --num-updates "${UPDATES}" \
  --train-seed-base "${SEED_BASE}" \
  --chunk-len 25 \
  --group-size "${GROUP_SIZE}" \
  --max-steps "${MAX_STEPS}" \
  --save-every "${SAVE_EVERY}" \
  --lr "${LR}" \
  --init-log-std "${INIT_LOG_STD}" \
  --euler-step-noise-std "${EULER_NOISE}" \
  --goal-latent-mode visual_proprio \
  --proprio-alpha 0.1 \
  --reward-key wm_latent_progress \
  --ratio-mode chunk \
  --action-transform "${ACTION_TRANSFORM}" \
  --reset-mismatch fail \
  --decode-candidates selected \
  "${STRICT_DECODE[@]}"

test -f "${OUT}/train_manifest.json"
test -f "${OUT}/progress.jsonl"
test -f "${OUT}/checkpoints/latest.pt"
rg -q '"event": "update_complete"' "${OUT}/progress.jsonl"
echo "PHASE12_WM_CHUNK_GRPO_TRAIN_DONE profile=${ACTION_PROFILE} updates=${UPDATES} out=${OUT}"
```

Slurm decode/offline rationale:

- One-update smoke must prove local cache completeness, so default smoke env is offline:
  - `HF_HUB_OFFLINE=1`,
  - `TRANSFORMERS_OFFLINE=1`.
- This can fail if SmolVLA, SmolVLM, JEPA main checkpoint, or JEPA image decoder are not cached. That failure is useful signal. Diagnose cache path/root cause, warm cache deliberately, then requeue.
- Strict decode requires image decoder, so default smoke env must keep `JEPA_WM_DISABLE_IMAGE_HEAD=0`.
- `JEPA_WM_DISABLE_IMAGE_HEAD=1` is allowed only for non-decode load/scoring probes or emergency 100-update scoring runs where decode is explicitly non-strict.
- Source of truth:
  - `VGG JEPA/jepa-wms/hubconf.py::_load_model_with_config(...)` pops `pretrain_dec_path` when `JEPA_WM_DISABLE_IMAGE_HEAD=1`,
  - `VGG JEPA/jepa-wms/app/vjepa_wm/modelcustom/simu_env_planning/vit_enc_preds.py::EncPredWM.decode_unroll(...)` returns frames only when `"image_head" in self.model.heads`,
  - MetaWorld JEPA-WM config points image head at `https://dl.fbaipublicfiles.com/jepa-wms/vm2m_lpips_dv2vits_vitldec_224_INet.pth.tar`.

- [ ] **Step 4: Add pytest markers**

Patch `pytest.ini`:

```ini
[pytest]
testpaths = tests
python_files = test_*.py
markers =
    phase12: Phase12 WM-GRPO tests
    static: static source/contract tests
    slurm_static: Slurm script static contract tests
```

- [ ] **Step 5: Run static checks**

Run:

```bash
bash -n scripts/grpo/submit_phase12_wm_chunk_grpo_train.slurm
PYTHONPATH=src /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python -m pytest tests/test_phase12_slurm_static.py -v
```

Expected: pass.

## Task 10: Verification And GPU Execution

**Files:**
- No source edits unless failures root-cause to code.

Autonomous execution requirement:

- After code/local tests pass, the agent must submit the one-update smoke chain to Slurm.
- The agent must monitor Slurm/stdout/artifact outputs, not stop after `sbatch`.
- If any smoke fails, the agent must:
  - inspect job state and stdout/stderr,
  - identify first failing invariant,
  - do root-cause analysis,
  - patch code/scripts/tests if needed,
  - rerun relevant local tests,
  - requeue from the failed smoke point.
- Do not launch the 100-update job until both one-update smokes pass by the pass criteria below.
- After both one-update smokes pass, submit the single 100-update `official_jepa_mirror` job and monitor it with the same failure policy.
- Keep Phase12 GPU usage to one GPU per job and do not push the user's total above three GPUs.

- [ ] **Step 1: Run local suite**

Run:

```bash
PYTHONPATH=src /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python -m pytest \
  tests/test_grpo_policy_wrapper_chunk.py \
  tests/test_phase12_objective.py \
  tests/test_phase12_actions.py \
  tests/test_phase12_goals.py \
  tests/test_phase12_rollout.py \
  tests/test_phase12_diagnostics.py \
  tests/test_phase12_artifacts.py \
  tests/test_phase12_wm_reward.py \
  tests/test_phase12_training_loop.py \
  tests/test_phase12_trainer_static.py \
  tests/test_phase12_slurm_static.py \
  tests/test_grpo_checkpointing.py -v
```

Expected: all pass.

- [ ] **Step 2: Check GPU capacity**

Run:

```bash
squeue -u "$USER" -o "%.18i %.9P %.20j %.8T %.10M %.6D %R"
```

Rule: each Phase12 job uses one GPU. Do not push user total above 3 GPUs. If uncertain, chain with `afterok`.

- [ ] **Step 3: Submit one-update real training validation smoke**

Run:

```bash
cd /vol/bitbucket/aa6622/project
jid_smoke_official=$(sbatch --parsable --chdir=/vol/bitbucket/aa6622/project --export=NIL \
  scripts/grpo/submit_phase12_wm_chunk_grpo_train.slurm official_jepa_mirror 1)
jid_smoke_bounded=$(sbatch --parsable --dependency=afterok:${jid_smoke_official} --chdir=/vol/bitbucket/aa6622/project --export=NIL \
  scripts/grpo/submit_phase12_wm_chunk_grpo_train.slurm bounded_executed 1)
echo "${jid_smoke_official} ${jid_smoke_bounded}"
```

Pass criteria:

- stdout contains `PHASE12_WM_CHUNK_GRPO_TRAIN_DONE`.
- `progress.jsonl` contains `event="update_complete"`.
- update row has `optimizer_step=true`.
- checkpoint has non-empty `policy_state_dict`.
- checkpoint has non-empty `optimizer_state_dict`.
- row has `loss`, `advantages`, `returns`, `ratio_mean`, `ratio_clip_fraction`, `approx_kl`.
- `rollout_validation_video` exists and non-empty for episode 0.
- `selected_action_rollout_video` exists and non-empty for episode 0.
- `oracle_baseline_video` exists and non-empty for same seed/task/max-steps.
- selected WM decode strip exists and non-empty.
- selected real-vs-pred strip exists and non-empty.
- manifest records `wm_decode_status="ok"`.
- stdout/logs show `JEPA_WM_DISABLE_IMAGE_HEAD=0` or equivalent manifest env capture.
- no cache-miss download attempt is hidden by online fallback; with `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`, missing model/decoder cache is a smoke failure.
- no traceback, NaN, WM fallback, or reset mismatch.
- Same pass criteria apply to `bounded_executed`.
- Both smoke directories should contain videos to inspect:
  - selected action rollout MP4,
  - oracle baseline MP4,
  - selected WM decode strip,
  - selected real-vs-pred strip.

- [ ] **Step 4: Submit Phase111-style 100-update WM-GRPO run**

Run only after both one-update smokes pass. Submit one 100-update job, using `official_jepa_mirror` only:

```bash
jid100_official=$(sbatch --parsable --chdir=/vol/bitbucket/aa6622/project --export=NIL \
  scripts/grpo/submit_phase12_wm_chunk_grpo_train.slurm official_jepa_mirror 100)
echo "${jid100_official}"
```

100-update pass criteria:

- stdout contains `PHASE12_WM_CHUNK_GRPO_TRAIN_DONE`.
- manifest `num_updates`/`end_update` reflect 100 updates.
- `progress.jsonl` has 100 update rows unless resume/start-update changes expected count.
- `checkpoints/latest.pt` exists.
- `checkpoints/update_0005.pt` exists because `save_every=5`, matching Phase111.
- final row has same Phase111-style fields plus WM reward fields.
- decode failures in 100-update run are logged, not fatal, unless WM scoring fails.

- [ ] **Step 5: Monitor failures by first invariant**

Run:

```bash
scontrol show job <jobid> | rg "JobState=|Reason=|RunTime=|TimeLimit=|StdOut=|StdErr=|NodeList=|Partition="
rg -n "PROJECT_ROOT=|HF_HOME=|TORCH_HOME=|PHASE12_|Traceback|CUDA|OOM|NaN|reset mismatch|WM scoring fallback|decode_status|decode_failure" "<stdout-path>"
```

Failure policy:

- Any traceback, NaN, missing required artifact, strict reset mismatch, WM scoring fallback, empty checkpoint state, missing update row, or missing marker is failure.
- Diagnose root cause before requeue.
- Re-run relevant unit/static tests after code fixes.
- Requeue from failed point.
- Continue monitor/fix/requeue loop until:
  - both one-update smokes pass and 100-update job is submitted, or
  - a blocker requires unavailable credentials/data/manual cluster intervention.

## Self-Review

Spec coverage:

- Current fake GRPO marker/checkpoint issue: Task 1, Task 6, Task 8, Task 9.
- Artifact/video issue: Task 2, Task 3, Task 10.
- Action profile no-op: Task 5.
- Structured official objective: Task 4.
- Chunk-level ratio and optimizer loop: Task 6.
- Slurm naming/env/static gaps: Task 9.
- GPU autonomous sequence: one-update real training smoke, then 100-update WM-GRPO run.
- Original side info retained: non-negotiables, reference behavior, root-cause findings, action/default choices.

Placeholder scan:

- No `TBD`.
- No `TODO`.
- No "implement later".
- Deferred ablations are explicitly labeled non-mainline.

Type consistency:

- `Phase12Score` comes from `phase12_objective.py`.
- Candidate action fields match `phase12_actions.py`.
- Training loop consumes `old_logprob_sums`, `unsquashed_chunks`, and `proc_root_snapshots` returned by Phase12 episode metadata.
- Artifact keys use canonical `rollout_validation_video`, with old `smolvla_first_rollout_video` removed or retained only as alias if tests require it.

