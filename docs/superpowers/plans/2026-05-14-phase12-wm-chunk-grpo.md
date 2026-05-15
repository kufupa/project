# Phase12 WM-Scored Chunk GRPO Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Use fresh implementer subagents per task, then spec-compliance review, then code-quality review. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Phase12: a serial receding-horizon SmolVLA GRPO trainer for MetaWorld `push-v3`, where JEPA-WM scores sampled SmolVLA action chunks against oracle subgoals using the official JEPA-WM MetaWorld objective.

**Architecture:** Phase12 is a new trainer, not a mutation of Phase111. It reuses Phase111's SmolVLA GRPO/logprob/checkpoint plumbing and Phase8's JEPA-WM action/render/decode plumbing, but replaces Phase8's flattened visual-only scoring with the official JEPA-WM objective: final-step mean-squared visual latent distance plus `0.1 *` final-step mean-squared proprio latent distance. Each segment samples `K` chunks from SmolVLA, scores all chunks with JEPA-WM, updates SmolVLA with chunk-level GRPO, executes only the best chunk in the real environment, then re-renders and re-encodes the real next state.

**Tech Stack:** Python 3.12, PyTorch, patched LeRobot SmolVLA in `/vol/bitbucket/aa6622/.envs/lerobot_mw_py310`, MetaWorld, JEPA-WM, TensorDict, imageio/Pillow, Slurm with `--export=NIL` and `scripts/slurm/common_env.sh`.

---

## Non-Negotiable Decisions

- [ ] **No CEM in Phase12 mainline.**
  - Official JEPA-WM uses CEM to optimize actions online.
  - Phase12 only borrows the official JEPA-WM objective.
  - Phase12 action candidates come from SmolVLA sampling.
  - Phase12 learning uses GRPO.
  - CEM is deferred to a later optional baseline: "JEPA-WM planner alone vs JEPA-WM reward for VLA training."

- [ ] **Default objective mirrors official JEPA-WM MetaWorld planning.**
  - `objective_type=L2`
  - `goal_latent_mode=visual_proprio`
  - `proprio_alpha=0.1`
  - `combined_distance = visual_mse + 0.1 * proprio_mse`
  - `wm_latent_progress = start_combined_distance - final_combined_distance`
  - `latent_return = -final_combined_distance`
  - GRPO default reward key: `wm_latent_progress`

- [ ] **Default action profile is `official_jepa_mirror`.**
  - Score raw postprocessed env-scale SmolVLA actions after JEPA-WM normalization/packing.
  - Execute the same raw postprocessed actions through the official-style env path.
  - Do not claim JEPA-WM was trained on clipped SmolVLA outputs.
  - Keep `bounded_executed` as a first-class safety/action-bound comparison.

- [ ] **Default rollout is serial.**
  - No vector/async Phase12 until serial smoke passes.
  - After executing selected chunk, discard predicted WM state.
  - Re-render/re-encode from the real environment state.

- [ ] **Default chunk/subgoal schedule.**
  - `chunk_len=25`, configurable.
  - `K=4` candidates for smoke, configurable.
  - Primary oracle subgoals at frames `25, 50, 75, ...`, plus first success or final frame.
  - Save companion frames `26, 51, 76, ...` for drift/debug only.
  - Use 1-based frame args in user-facing logs, mapped to zero-based filenames (`25 -> frame_000024.png`).

- [ ] **Default GRPO math is chunk-level.**
  - One chunk candidate = one sampled completion from one root state.
  - One scalar WM score per chunk.
  - Ratio: `exp(sum_t new_logp_t - sum_t old_logp_t)`.
  - Per-step ratio is only `--ratio-mode per_step_ablation`.

- [ ] **Method label must stay honest.**
  - Use: "WM-scored receding-horizon chunk GRPO with official-LeRobot-compatible rollouts."
  - Do not call it unbiased env-on-policy GRPO.
  - Non-selected chunks receive synthetic WM rewards and do not affect future real states.

## Official JEPA-WM Reference Behavior

Use these files as reference, not as code to copy wholesale:

- [ ] `VGG JEPA/jepa-wms/configs/evals/simu_env_planning/mw/jepa-wm/reach-wall_L2_cem_sourcexp_H6_nas3_ctxt2_r256_alpha0.1_ep48_decode.yaml`
  - `goal_source: expert`
  - `frameskip: 5`
  - `normalize_action: true`
  - `planner_name: cem`
  - `horizon: 6`
  - `num_act_stepped: 3`
  - `planning_objective.objective_type: L2`
  - `planning_objective.alpha: 0.1`

- [ ] `VGG JEPA/jepa-wms/evals/simu_env_planning/planning/planning/objectives.py::ReprTargetDistMPCObjective`
  - Structured `TensorDict` objective.
  - Visual MSE plus `alpha *` proprio MSE.
  - Uses final predicted WM step unless `sum_all_diffs`.

- [ ] `VGG JEPA/jepa-wms/evals/simu_env_planning/planning/gc_agent.py::GC_Agent.act`
  - Encodes current real obs.
  - Plans in latent space.
  - Replans after real env execution.

- [ ] `VGG JEPA/jepa-wms/evals/simu_env_planning/planning/plan_evaluator.py::PlanEvaluator.unroll_agent`
  - Denormalizes planned actions before environment step.
  - Saves expert/agent videos and optional decoded prediction artifacts.

Phase12 differs intentionally:

- [ ] SmolVLA supplies sampled chunks; CEM does not.
- [ ] GRPO updates SmolVLA; official JEPA-WM does no policy training.
- [ ] Intermediate oracle subgoals are mainline; final-goal-only is deferred.

## File Structure

### Modify Existing Files

- [ ] `src/smolvla_pipeline/evaluator.py`
  - Add optional `n_action_steps: int = 1` to `_load_smolvla_bundle()`.
  - Preserve Phase111/eval defaults.
  - Phase12 should load SmolVLA with `n_action_steps=chunk_len`.

- [ ] `src/smolvla_grpo/phase11_rollout.py`
  - Thread optional `n_action_steps` through `load_bundle_for_grpo()`.
  - Keep Phase111 callers defaulting to `1`.
  - Do not add Phase12 rollout code here.

- [ ] `src/smolvla_grpo/policy_wrapper.py`
  - Keep one-step Phase111 APIs unchanged.
  - Add full-chunk sampling and full-chunk logprob recompute APIs.
  - Use patched LeRobot `_get_distr_params_chunk()` or a thin local wrapper.
  - Do not use `predict_action_chunk()` for GRPO training samples unless it exposes old logprobs.
  - Keep `action_transform="no_tanh"` default.

- [ ] `src/smolvla_grpo/reward_backends.py`
  - Keep `EnvRewardBackend` unchanged.
  - Keep `WMLatentRewardBackend` as a metadata reader.
  - Do not put JEPA-WM scoring here.

- [ ] `src/segment_grpo_loop.py`
  - Reuse helpers where practical:
    - `_normalize_env_actions_for_wm()`
    - `_pack_env_actions_for_wm()`
    - `DecodeTrace`
    - `_decode_latent_trace_to_frames()`
    - `_build_real_vs_pred_strip()`
    - `_write_comparison_strip()`
    - `_stitch_comparison_strip()`
  - Avoid broad refactors.
  - If structured Phase12 scoring needs a new helper, add a narrow one without breaking Phase8.

### Create New Source Files

- [ ] `src/smolvla_grpo/phase12_objective.py`
  - Official JEPA-WM objective.
  - Structured visual/proprio latent extraction.
  - Distance/progress/return dataclasses.

- [ ] `src/smolvla_grpo/phase12_actions.py`
  - Action profiles.
  - Raw/clipped/scoring/execution action selection.
  - JEPA-normalized and packed-action telemetry.

- [ ] `src/smolvla_grpo/phase12_goals.py`
  - Oracle rollout and subgoal schedule.
  - Goal frame/proprio saving.
  - Reset parity guard.
  - Goal latent encoding.

- [ ] `src/smolvla_grpo/phase12_rollout.py`
  - Serial episode/segment loop.
  - Candidate sampling/scoring.
  - Chunk-level GRPO update.
  - Best-chunk execution.
  - Fresh real-state re-encoding.

- [ ] `src/smolvla_grpo/phase12_diagnostics.py`
  - WM decoded prediction strips.
  - Real-vs-pred strips.
  - Decode failure metadata.

- [ ] `src/smolvla_grpo/phase12_logging.py`
  - Progress rows.
  - Manifest helpers.
  - Timestamp/duration fields.

### Create New Scripts

- [ ] `scripts/grpo/train_phase12_wm_chunk_grpo.py`
  - Main CLI.
  - Loads SmolVLA, JEPA-WM, env backend.
  - Runs smoke/full training.
  - Writes checkpoints, progress, manifest, artifacts.

- [ ] `scripts/grpo/submit_phase12_wm_chunk_grpo_smoke.slurm`
  - One-episode GPU smoke.
  - One GPU per job.
  - Default action profile configurable with first script argument, falling back to `official_jepa_mirror`.
  - Emits `PHASE12_WM_CHUNK_SMOKE_OK`.

- [ ] `scripts/grpo/submit_phase12_wm_chunk_grpo.slurm`
  - 10-episode or full run.
  - One GPU per job.
  - Default action profile configurable with first script argument, falling back to `official_jepa_mirror`.
  - Episode count configurable with second script argument, falling back to `10`.
  - Emits `PHASE12_WM_CHUNK_GRPO_OK`.

### Create New Tests

- [ ] `tests/test_grpo_policy_wrapper_chunk.py`
- [ ] `tests/test_phase12_objective.py`
- [ ] `tests/test_phase12_actions.py`
- [ ] `tests/test_phase12_goals.py`
- [ ] `tests/test_phase12_rollout.py`
- [ ] `tests/test_phase12_diagnostics.py`
- [ ] `tests/test_phase12_trainer_static.py`
- [ ] `tests/test_phase12_artifacts.py`
- [ ] `tests/test_phase12_slurm_scripts.py`

## Core Data Contracts

### `Phase12Goal`

```python
@dataclass
class Phase12Goal:
    subgoal_index: int
    frame_index_1based: int
    frame_path: Path
    companion_frame_index_1based: int | None
    companion_frame_path: Path | None
    proprio: np.ndarray
    goal_visual: torch.Tensor
    goal_proprio: torch.Tensor
    source: str
```

### `Phase12Candidate`

```python
@dataclass
class Phase12Candidate:
    candidate_index: int
    proc_root_snapshot: dict[str, Any]
    unsquashed_chunk: torch.Tensor
    old_logprob_steps: torch.Tensor
    old_logprob_sum: torch.Tensor
    exec_actions_raw_postprocessed: np.ndarray
    exec_actions_for_env: np.ndarray
    exec_actions_for_wm: np.ndarray
    action_metadata: dict[str, Any]
```

### `Phase12Score`

```python
@dataclass
class Phase12Score:
    candidate_index: int
    start_visual_distance: float
    start_proprio_distance: float
    start_combined_distance: float
    final_visual_distance: float
    final_proprio_distance: float
    final_combined_distance: float
    wm_latent_progress: float
    latent_return: float
    wm_status: str
    debug_npz_path: str | None = None
```

### `Phase12SegmentRecord`

```python
@dataclass
class Phase12SegmentRecord:
    update_index: int
    episode_index: int
    segment_index: int
    goal_frame_index_1based: int
    selected_candidate_index: int
    scores: list[Phase12Score]
    candidates: list[Phase12Candidate]
    success_any: bool
    success_last: bool
    env_reward_sum: float
    decode_metadata: dict[str, Any]
```

## Task 0: Worktree, Git Safety, And Baseline

**Files:**
- Read-only: git metadata.
- Possible create: isolated worktree directory.

- [ ] **Step 0.1: Record current git state**

Run:

```bash
cd /vol/bitbucket/aa6622/project
git status --short
git diff --stat
git branch --show-current
```

Expected: record dirty files. Do not reset or discard user changes.

- [ ] **Step 0.2: Set up isolated worktree if practical**

Use `superpowers:using-git-worktrees`.

Preferred:

```bash
cd /vol/bitbucket/aa6622/project
git check-ignore -q .worktrees || git check-ignore -q worktrees || true
git worktree add .worktrees/phase12-wm-chunk-grpo -b phase12-wm-chunk-grpo
```

If `.worktrees/` is not ignored and adding `.gitignore` would require an unplanned commit, use current worktree and avoid destructive operations. Record reason.

- [ ] **Step 0.3: Confirm interpreter**

Run:

```bash
/vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python - <<'PY'
import sys
import lerobot.policies.smolvla.modeling_smolvla as m
print("executable", sys.executable)
print("modeling_smolvla", m.__file__)
assert sys.executable.startswith("/vol/bitbucket/aa6622/.envs/lerobot_mw_py310/")
assert "/vol/bitbucket/aa6622/.envs/lerobot_mw_py310/" in m.__file__
PY
```

Expected: interpreter and `modeling_smolvla.py` both under `lerobot_mw_py310`.

- [ ] **Step 0.4: Use TDD for all code tasks**

For each implementation task:

1. write failing test,
2. run targeted test and confirm expected failure,
3. implement minimal code,
4. run targeted test and confirm pass,
5. run relevant broader tests.

## Task 1: Full-Chunk SmolVLA Policy API

**Files:**
- Modify: `src/smolvla_pipeline/evaluator.py`
- Modify: `src/smolvla_grpo/phase11_rollout.py`
- Modify: `src/smolvla_grpo/policy_wrapper.py`
- Test: `tests/test_grpo_policy_wrapper_chunk.py`

- [ ] **Step 1.1: Write failing tests for `n_action_steps` threading**

Test desired behavior:

```python
def test_load_bundle_for_grpo_threads_n_action_steps(monkeypatch):
    captured = {}

    def fake_load(checkpoint, *, n_action_steps=1, **kwargs):
        captured["n_action_steps"] = n_action_steps
        return object()

    monkeypatch.setattr("smolvla_grpo.phase11_rollout._load_smolvla_bundle", fake_load)

    from smolvla_grpo.phase11_rollout import load_bundle_for_grpo

    load_bundle_for_grpo("dummy", n_action_steps=25)

    assert captured["n_action_steps"] == 25
```

Run:

```bash
PYTHONPATH=src /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python -m pytest tests/test_grpo_policy_wrapper_chunk.py -v
```

Expected: fails because `n_action_steps` is not threaded.

- [ ] **Step 1.2: Write failing tests for chunk sampling shapes**

Test desired behavior with fake policy:

```python
def test_sample_action_chunk_from_proc_returns_steps_and_sum_logprob(fake_chunk_wrapper):
    sample = fake_chunk_wrapper.sample_action_chunk_from_proc(
        {"observation.state": "dummy"},
        chunk_len=25,
        rng=None,
    )

    assert sample.unsquashed_chunk.shape == (25, 4)
    assert sample.log_prob_steps.shape == (25,)
    assert sample.log_prob_sum.shape == ()
    assert sample.exec_action_np.shape == (25, 4)
```

Expected: fails because chunk API does not exist.

- [ ] **Step 1.3: Implement `n_action_steps` threading**

Implement:

- `_load_smolvla_bundle(checkpoint, ..., n_action_steps: int = 1)`
- `load_bundle_for_grpo(checkpoint, ..., n_action_steps: int = 1)`

Rules:

- default remains `1`.
- Phase111 behavior unchanged.

- [ ] **Step 1.4: Implement chunk dataclass/API**

Add to `policy_wrapper.py`:

```python
@dataclass
class SampledActionChunk:
    exec_action_np: np.ndarray
    policy_tensor: torch.Tensor
    unsquashed_chunk: torch.Tensor
    log_prob_steps: torch.Tensor
    log_prob_sum: torch.Tensor
    action_clip_fraction: np.ndarray
    action_clip_any: np.ndarray
    unique_action_rows: int
```

Add methods:

```python
def sample_action_chunk_from_proc(self, proc: Any, *, chunk_len: int, rng: torch.Generator | None = None) -> SampledActionChunk:
    ...

def get_action_probs_for_chunk_from_proc(self, proc: Any, unsquashed_chunk: torch.Tensor) -> torch.Tensor:
    ...
```

Shape contract:

- mean/log_std: `[chunk_len, action_dim]`
- unsquashed: `[chunk_len, action_dim]`
- logprob steps: `[chunk_len]`
- sum logprob: scalar
- env action chunk: `[chunk_len, env_action_dim]`

- [ ] **Step 1.5: Run tests**

Run:

```bash
PYTHONPATH=src /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python -m pytest tests/test_grpo_policy_wrapper_chunk.py -v
```

Expected: pass.

## Task 2: Official JEPA-WM Objective

**Files:**
- Create: `src/smolvla_grpo/phase12_objective.py`
- Test: `tests/test_phase12_objective.py`

- [ ] **Step 2.1: Write failing formula tests**

Tests:

```python
def test_combined_l2_distance_uses_visual_plus_alpha_proprio():
    pred = {"visual": torch.tensor([1.0, 3.0]), "proprio": torch.tensor([2.0, 6.0])}
    goal = {"visual": torch.tensor([0.0, 1.0]), "proprio": torch.tensor([0.0, 2.0])}

    d = combined_l2_distance(pred, goal, proprio_alpha=0.1)

    assert d.visual_distance == pytest.approx(((1.0 ** 2 + 2.0 ** 2) / 2.0))
    assert d.proprio_distance == pytest.approx(((2.0 ** 2 + 4.0 ** 2) / 2.0))
    assert d.combined_distance == pytest.approx(d.visual_distance + 0.1 * d.proprio_distance)
```

```python
def test_score_progress_uses_start_minus_final_and_negative_final_return():
    score = score_progress(
        candidate_index=2,
        start={"visual": torch.ones(2), "proprio": torch.ones(2)},
        final={"visual": torch.zeros(2), "proprio": torch.zeros(2)},
        goal={"visual": torch.zeros(2), "proprio": torch.zeros(2)},
        proprio_alpha=0.1,
    )

    assert score.wm_latent_progress > 0
    assert score.latent_return == pytest.approx(-score.final_combined_distance)
```

Run:

```bash
PYTHONPATH=src /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python -m pytest tests/test_phase12_objective.py -v
```

Expected: import failure.

- [ ] **Step 2.2: Implement objective module**

Implement:

```python
@dataclass(frozen=True)
class Phase12Distance:
    visual_distance: float
    proprio_distance: float
    combined_distance: float

@dataclass(frozen=True)
class Phase12Score:
    candidate_index: int
    start_visual_distance: float
    start_proprio_distance: float
    start_combined_distance: float
    final_visual_distance: float
    final_proprio_distance: float
    final_combined_distance: float
    wm_latent_progress: float
    latent_return: float
    wm_status: str
    debug_npz_path: str | None = None
```

Implement:

```python
def split_structured_latent(encoded: Any, *, mode: str = "visual_proprio") -> dict[str, torch.Tensor]:
    ...

def combined_l2_distance(pred: Mapping[str, torch.Tensor], goal: Mapping[str, torch.Tensor], *, proprio_alpha: float = 0.1) -> Phase12Distance:
    ...

def score_progress(... ) -> Phase12Score:
    ...
```

Rules:

- default mode requires both `visual` and `proprio`.
- missing proprio in default mode raises a clear error.
- visual-only mode exists only as `visual_only_ablation`.
- return plain floats for JSON compatibility.

- [ ] **Step 2.3: Run objective tests**

Run:

```bash
PYTHONPATH=src /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python -m pytest tests/test_phase12_objective.py -v
```

Expected: pass.

## Task 3: Action Profiles And Telemetry

**Files:**
- Create: `src/smolvla_grpo/phase12_actions.py`
- Test: `tests/test_phase12_actions.py`

- [ ] **Step 3.1: Write failing action profile tests**

Tests:

```python
def test_official_jepa_mirror_scores_raw_postprocessed_actions():
    raw = np.array([[2.0, -2.0, 0.5, 1.5]], dtype=np.float32)
    result = apply_phase12_action_profile(raw, action_profile="official_jepa_mirror", action_low=-1, action_high=1)

    np.testing.assert_allclose(result.exec_actions_for_env, raw)
    np.testing.assert_allclose(result.exec_actions_for_wm, raw)
    assert result.metadata["clip_fraction"] > 0.0
```

```python
def test_bounded_executed_scores_exactly_what_it_executes():
    raw = np.array([[2.0, -2.0, 0.5, 1.5]], dtype=np.float32)
    result = apply_phase12_action_profile(raw, action_profile="bounded_executed", action_low=-1, action_high=1)

    np.testing.assert_allclose(result.exec_actions_for_env, np.clip(raw, -1, 1))
    np.testing.assert_allclose(result.exec_actions_for_wm, result.exec_actions_for_env)
```

Expected: import failure.

- [ ] **Step 3.2: Implement action profile module**

Implement:

```python
@dataclass(frozen=True)
class Phase12ActionProfileResult:
    exec_actions_raw_postprocessed: np.ndarray
    exec_actions_clipped: np.ndarray
    exec_actions_for_env: np.ndarray
    exec_actions_for_wm: np.ndarray
    metadata: dict[str, Any]
```

Implement:

```python
def apply_phase12_action_profile(
    raw_postprocessed_actions: np.ndarray,
    *,
    action_profile: str,
    action_low: float | np.ndarray,
    action_high: float | np.ndarray,
    preprocessor: Any | None = None,
    env_action_dim: int | None = None,
    wm_action_dim: int | None = None,
) -> Phase12ActionProfileResult:
    ...
```

Metadata fields:

- raw min/max/mean/std
- clipped min/max/mean/std
- clip fraction
- clip any
- JEPA-normalized min/max/mean/std if preprocessor supplied
- JEPA-normalized max absolute value if preprocessor supplied
- packed action shape if dimensions supplied
- env action dim
- WM action dim
- pack factor

- [ ] **Step 3.3: Run action tests**

Run:

```bash
PYTHONPATH=src /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python -m pytest tests/test_phase12_actions.py -v
```

Expected: pass.

## Task 4: Oracle Subgoals And Reset Guard

**Files:**
- Create: `src/smolvla_grpo/phase12_goals.py`
- Test: `tests/test_phase12_goals.py`

- [ ] **Step 4.1: Write failing schedule tests**

Tests:

```python
def test_goal_schedule_uses_primary_and_companion_frames():
    schedule = build_subgoal_schedule(max_frame_1based=80, chunk_len=25, success_frame_1based=None)

    assert schedule.primary_frames_1based == [25, 50, 75, 80]
    assert schedule.companion_frames_1based == [26, 51, 76]
    assert frame_index_to_filename(25) == "frame_000024.png"
```

```python
def test_reset_parity_reports_image_and_proprio_diffs():
    a_img = np.zeros((2, 2, 3), dtype=np.uint8)
    b_img = np.ones((2, 2, 3), dtype=np.uint8)
    a_prop = np.array([0.0, 1.0], dtype=np.float32)
    b_prop = np.array([0.5, 1.0], dtype=np.float32)

    metrics = compute_reset_parity(a_img, b_img, a_prop, b_prop)

    assert metrics["image_mean_abs_diff"] > 0
    assert metrics["image_max_abs_diff"] == 1
    assert metrics["proprio_max_abs_diff"] == pytest.approx(0.5)
```

Expected: import failure.

- [ ] **Step 4.2: Implement schedule/reset helpers**

Implement:

- `frame_index_to_filename(frame_index_1based: int) -> str`
- `build_subgoal_schedule(max_frame_1based: int, chunk_len: int, success_frame_1based: int | None) -> Phase12GoalSchedule`
- `compute_reset_parity(init_image, reset_image, init_proprio, reset_proprio) -> dict[str, float]`
- `should_fail_reset_parity(metrics, image_mean_threshold, image_max_threshold, proprio_max_threshold) -> bool`

Rules:

- schedule includes final/success frame if not already included.
- companion frame `primary + 1` only if within available frame range.
- strict mode fails if goal frame exists but matching proprio row is missing.

- [ ] **Step 4.3: Implement oracle rollout scaffold**

Implement public API:

```python
def collect_phase12_oracle_goals(
    *,
    env_backend: str,
    task: str,
    seed: int,
    chunk_len: int,
    output_dir: Path,
    strict_reset: bool,
    wm_bundle: Any | None = None,
) -> Phase12GoalBundle:
    ...
```

Keep actual env-specific code narrow and test helper behavior without requiring GPU.

- [ ] **Step 4.4: Run goal tests**

Run:

```bash
PYTHONPATH=src /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python -m pytest tests/test_phase12_goals.py -v
```

Expected: pass.

## Task 5: WM-Scored Receding-Horizon Rollout

**Files:**
- Create: `src/smolvla_grpo/phase12_rollout.py`
- Test: `tests/test_phase12_rollout.py`

- [ ] **Step 5.1: Write failing same-root and fresh-state tests**

Tests:

```python
def test_all_candidates_score_from_same_root_observation(dummy_phase12_rollout):
    result = dummy_phase12_rollout.run_one_segment(k=4)

    root_ids = {record.root_observation_id for record in result.candidate_score_calls}
    assert len(root_ids) == 1
```

```python
def test_next_segment_uses_fresh_observation_after_best_execution(dummy_phase12_rollout):
    result = dummy_phase12_rollout.run_two_segments(k=4)

    assert result.segment_root_observation_ids[0] != result.segment_root_observation_ids[1]
    assert result.env_execute_calls[0].candidate_index == result.segment_records[0].selected_candidate_index
```

Expected: import failure or missing API.

- [ ] **Step 5.2: Implement rollout data structures**

Implement:

- `Phase12Candidate`
- `Phase12SegmentRecord`
- `Phase12EpisodeResult`
- `collect_phase12_episode(...)`

Required segment flow:

1. choose current primary subgoal,
2. render current policy obs and WM obs from same simulator state,
3. sample `K` chunks from old policy root proc,
4. apply action profile,
5. score all candidates from same root WM latent,
6. compute group returns/advantages,
7. run chunk-level GRPO update,
8. sync old policy,
9. execute selected candidate in env,
10. collect real frames for video/strip,
11. re-render and re-encode real state for next segment.

- [ ] **Step 5.3: Implement chunk-level GRPO helper**

Implement:

```python
def chunk_grpo_loss(
    *,
    old_logprob_sums: torch.Tensor,
    new_logprob_sums: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    ...
```

Expected formula:

```python
ratio = torch.exp(new_logprob_sums - old_logprob_sums)
unclipped = ratio * advantages
clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
loss = -torch.min(unclipped, clipped).mean()
```

- [ ] **Step 5.4: Run rollout tests**

Run:

```bash
PYTHONPATH=src /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python -m pytest tests/test_phase12_rollout.py -v
```

Expected: pass.

## Task 6: WM Decode Diagnostics

**Files:**
- Create: `src/smolvla_grpo/phase12_diagnostics.py`
- Test: `tests/test_phase12_diagnostics.py`

- [ ] **Step 6.1: Write failing diagnostic tests**

Tests:

```python
def test_expected_decoded_future_frames_for_25_env_steps():
    assert expected_wm_decode_steps(chunk_len=25, env_steps_per_wm_step=5) == 5
```

```python
def test_decode_failure_nonfatal_unless_strict(tmp_path):
    result = build_decode_artifacts(
        decode_fn=lambda: (_ for _ in ()).throw(RuntimeError("decode broke")),
        output_dir=tmp_path,
        strict_decode=False,
    )

    assert result.metadata["decode_status"] == "failed"
    assert "decode broke" in result.metadata["decode_failure_reason"]
```

Expected: import failure.

- [ ] **Step 6.2: Implement diagnostics module**

Implement:

- `expected_wm_decode_steps(chunk_len, env_steps_per_wm_step) -> int`
- `build_decode_artifacts(...) -> Phase12DecodeArtifactResult`
- wrappers around Phase8 strip helpers.

Default smoke behavior:

- decode selected candidate only,
- save init/current frame plus decoded future frames,
- save real-vs-pred strip after execution if real frames available,
- stitch per-segment strips into episode strip.

Artifact paths:

- `segment_0000/wm_decode_selected_strip.png`
- `segment_0000/wm_real_vs_pred_selected_strip.png`
- `wm_decode_episode_stitched.png`

- [ ] **Step 6.3: Run diagnostics tests**

Run:

```bash
PYTHONPATH=src /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python -m pytest tests/test_phase12_diagnostics.py -v
```

Expected: pass.

## Task 7: Trainer, Manifest, Progress, And Checkpoints

**Files:**
- Create: `src/smolvla_grpo/phase12_logging.py`
- Create: `scripts/grpo/train_phase12_wm_chunk_grpo.py`
- Test: `tests/test_phase12_trainer_static.py`

- [ ] **Step 7.1: Write failing CLI/default tests**

Tests:

```python
def test_phase12_cli_defaults():
    args = parse_args([])

    assert args.env_backend == "official_lerobot_guarded"
    assert args.action_profile == "official_jepa_mirror"
    assert args.chunk_len == 25
    assert args.group_size == 4
    assert args.goal_latent_mode == "visual_proprio"
    assert args.proprio_alpha == pytest.approx(0.1)
    assert args.reward_key == "wm_latent_progress"
    assert args.ratio_mode == "chunk"
    assert args.action_transform == "no_tanh"
```

Expected: script/module missing.

- [ ] **Step 7.2: Implement logging helpers**

Implement:

- `utc_now_iso()`
- `duration_fields(...)`
- `build_progress_row(...)`
- `write_jsonl_row(path, row)`
- `write_manifest(path, manifest)`

Progress must include:

- timestamps/durations,
- objective fields,
- WM distance fields,
- GRPO fields,
- env success/reward diagnostics,
- reset metrics,
- action telemetry,
- decode metadata,
- artifact paths.

- [ ] **Step 7.3: Implement trainer CLI**

Required defaults:

```text
--env-backend official_lerobot_guarded
--action-profile official_jepa_mirror
--chunk-len 25
--group-size 4
--goal-latent-mode visual_proprio
--proprio-alpha 0.1
--reward-key wm_latent_progress
--ratio-mode chunk
--action-transform no_tanh
--reset-mismatch fail
--decode-candidates selected
--save-wm-decodes for smoke mode
```

Required output:

- `train_manifest.json`
- `progress.jsonl`
- `checkpoints/latest.pt`
- periodic `checkpoints/update_XXXX.pt`

Required smoke marker:

- `PHASE12_WM_CHUNK_SMOKE_OK`

Required full marker:

- `PHASE12_WM_CHUNK_GRPO_OK`

- [ ] **Step 7.4: Run trainer static tests**

Run:

```bash
PYTHONPATH=src /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python -m pytest tests/test_phase12_trainer_static.py -v
```

Expected: pass.

## Task 8: Smoke Artifacts

**Files:**
- Trainer/artifact code from Task 7.
- Test: `tests/test_phase12_artifacts.py`

- [ ] **Step 8.1: Write failing artifact manifest tests**

Tests:

```python
def test_smoke_manifest_requires_videos_and_decode_paths(tmp_path):
    manifest = {
        "oracle_baseline_video": "oracle/oracle_baseline.mp4",
        "smolvla_first_rollout_video": "rollouts/update_0000_episode_0000/smolvla_first_rollout.mp4",
        "wm_decode_selected_strip_path": "rollouts/update_0000_episode_0000/segment_0000/wm_decode_selected_strip.png",
        "wm_real_vs_pred_selected_strip_path": "rollouts/update_0000_episode_0000/segment_0000/wm_real_vs_pred_selected_strip.png",
        "success_any": False,
        "success_last": False,
    }

    assert_smoke_manifest_contract(manifest)
```

- [ ] **Step 8.2: Implement artifact contract helper**

Implement:

- `assert_smoke_manifest_contract(manifest)`
- optional `verify_smoke_artifacts(run_dir, manifest)`

Required smoke artifact layout:

```text
oracle/oracle_baseline.mp4
oracle/actions.jsonl
oracle/flat_obs.jsonl
oracle/goals/frame_000024.png
oracle/goals/frame_000025_companion.png
rollouts/update_0000_episode_0000/smolvla_first_rollout.mp4
rollouts/update_0000_episode_0000/candidates.jsonl
rollouts/update_0000_episode_0000/latents_debug.npz
rollouts/update_0000_episode_0000/segment_0000/wm_decode_selected_strip.png
rollouts/update_0000_episode_0000/segment_0000/wm_real_vs_pred_selected_strip.png
rollouts/update_0000_episode_0000/wm_decode_episode_stitched.png
train_manifest.json
progress.jsonl
```

- [ ] **Step 8.3: Run artifact tests**

Run:

```bash
PYTHONPATH=src /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python -m pytest tests/test_phase12_artifacts.py -v
```

Expected: pass.

## Task 9: Slurm Runners

**Files:**
- Create: `scripts/grpo/submit_phase12_wm_chunk_grpo_smoke.slurm`
- Create: `scripts/grpo/submit_phase12_wm_chunk_grpo.slurm`
- Test: `tests/test_phase12_slurm_scripts.py`

- [ ] **Step 9.1: Write failing Slurm static tests**

Tests:

```python
def test_phase12_slurm_uses_export_nil_and_common_env():
    text = Path("scripts/grpo/submit_phase12_wm_chunk_grpo_smoke.slurm").read_text()

    assert "#SBATCH --export=NIL" in text
    assert "common_env.sh" in text
    assert "slurm_resolve_project_root" in text
    assert "slurm_export_pythonpath" in text
    assert "slurm_export_hf_torch_cache" in text
    assert "sbatch" not in _body_without_comments(text)
```

Expected: files missing.

- [ ] **Step 9.2: Implement smoke Slurm**

Required:

- `#SBATCH --export=NIL`
- `#SBATCH --gres=gpu:1`
- source `scripts/slurm/common_env.sh`
- call `slurm_resolve_project_root`
- call `slurm_export_pythonpath`
- call `slurm_export_hf_torch_cache "phase12-wm-chunk-grpo-smoke"`
- explicit Python: `/vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python`
- action profile from `$1`, default `official_jepa_mirror`
- no nested `sbatch`
- emits `PHASE12_WM_CHUNK_SMOKE_OK`

- [ ] **Step 9.3: Implement 10-episode/full Slurm**

Required:

- same environment pattern as smoke,
- `#SBATCH --gres=gpu:1`,
- action profile from `$1`, default `official_jepa_mirror`,
- episode count from `$2`, default `10`,
- no nested `sbatch`,
- emits `PHASE12_WM_CHUNK_GRPO_OK`.

- [ ] **Step 9.4: Run Slurm static checks**

Run from the login/submit host. Use script arguments so `--export=NIL` stays simple:

```bash
bash -n scripts/grpo/submit_phase12_wm_chunk_grpo_smoke.slurm
bash -n scripts/grpo/submit_phase12_wm_chunk_grpo.slurm
PYTHONPATH=src /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python -m pytest tests/test_phase12_slurm_scripts.py -v
sbatch --test-only --chdir=/vol/bitbucket/aa6622/project --export=NIL scripts/grpo/submit_phase12_wm_chunk_grpo_smoke.slurm
sbatch --test-only --chdir=/vol/bitbucket/aa6622/project --export=NIL scripts/grpo/submit_phase12_wm_chunk_grpo.slurm
```

Expected:

- syntax checks pass,
- tests pass,
- Slurm test-only accepts scripts.

## Task 10: Integration Verification

**Files:** all Phase12 files.

- [ ] **Step 10.1: Run full unit/static suite**

Run:

```bash
PYTHONPATH=src /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python -m pytest \
  tests/test_grpo_policy_wrapper_chunk.py \
  tests/test_phase12_objective.py \
  tests/test_phase12_actions.py \
  tests/test_phase12_goals.py \
  tests/test_phase12_rollout.py \
  tests/test_phase12_diagnostics.py \
  tests/test_phase12_trainer_static.py \
  tests/test_phase12_artifacts.py \
  tests/test_phase12_slurm_scripts.py \
  -v
```

Expected: all pass.

- [ ] **Step 10.2: Run lints/diagnostics**

Use `ReadLints` on edited files.

Expected: no new linter errors from Phase12 files.

- [ ] **Step 10.3: Run CPU/dummy dry-run if supported**

Run a no-GPU or dummy path that checks CLI/logging failure behavior:

```bash
PYTHONPATH=src /vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python scripts/grpo/train_phase12_wm_chunk_grpo.py \
  --dry-run \
  --num-episodes 1 \
  --num-updates 1 \
  --group-size 4 \
  --chunk-len 25 \
  --strict-wm-scoring
```

Expected: either passes a dummy-mode contract or fails loudly because real WM scoring is unavailable. It must not silently fall back to synthetic reward.

## Task 11: GPU Smoke And Monitoring Loop

**Files:**
- Slurm scripts.
- Artifacts under `artifacts/phase12_wm_chunk_grpo/`.

### GPU Budget Rules

- [ ] Each Phase12 job requests one GPU.
- [ ] Smoke jobs use `afterok` dependency: `official_jepa_mirror` first, `bounded_executed` second.
- [ ] Monitor and fix during/after `official_jepa_mirror` smoke before trusting downstream jobs.
- [ ] 10-episode jobs may run concurrently only after checking live GPU usage and confirming user cap allows it.
- [ ] User cap: maximum 3 GPUs total across the user. Phase12 should not push total usage above 3.
- [ ] Never submit `sbatch` from inside a running allocation or batch script.

### Submit Smoke Chain

- [ ] **Step 11.1: Check current GPU usage**

Run from login/submit host:

```bash
squeue -u "$USER" -o "%.18i %.9P %.20j %.8T %.10M %.6D %R"
```

If needed, inspect jobs:

```bash
scontrol show job <jobid> | rg "JobState=|TRES=|gres/gpu|StdOut=|StdErr=|RunTime=|TimeLimit=|Partition=|NodeList="
```

Record whether there is space for one Phase12 GPU smoke.

- [ ] **Step 11.2: Submit official smoke**

Run:

```bash
cd /vol/bitbucket/aa6622/project
jid_official=$(sbatch --parsable --chdir=/vol/bitbucket/aa6622/project --export=NIL \
  scripts/grpo/submit_phase12_wm_chunk_grpo_smoke.slurm official_jepa_mirror)
echo "${jid_official}"
```

The Slurm script should read `$1` as the action profile. Do not compromise `common_env.sh`/cache setup.

- [ ] **Step 11.3: Submit bounded smoke with `afterok`**

Run:

```bash
cd /vol/bitbucket/aa6622/project
jid_bounded=$(sbatch --parsable --dependency=afterok:${jid_official} --chdir=/vol/bitbucket/aa6622/project --export=NIL \
  scripts/grpo/submit_phase12_wm_chunk_grpo_smoke.slurm bounded_executed)
echo "${jid_bounded}"
```

If environment passing is awkward under `--export=NIL`, provide two wrapper scripts or add Slurm script argument parsing before submitting. Do not submit from inside the first job.

### Monitor Official Smoke First

- [ ] **Step 11.4: Monitor official smoke**

Loop:

```bash
squeue -j "${jid_official}"
scontrol show job "${jid_official}" | rg "JobState=|Reason=|RunTime=|TimeLimit=|StdOut=|StdErr=|NodeList=|Partition="
```

Inspect stdout path from `scontrol`:

```bash
rg -n "PROJECT_ROOT=|HF_HOME=|TORCH_HOME=|PHASE12_WM_CHUNK_SMOKE_OK|Traceback|CUDA|OOM|NaN|reset mismatch|WM scoring fallback|decode_status|decode_failure" "<stdout-path>"
```

Do not wait silently if a clear traceback appears. Start diagnosis immediately.

### Smoke Pass Criteria

- [ ] job exits `0`.
- [ ] stdout contains `PHASE12_WM_CHUNK_SMOKE_OK`.
- [ ] `train_manifest.json` exists.
- [ ] `progress.jsonl` has at least one completed episode/update row.
- [ ] `oracle/oracle_baseline.mp4` exists and is non-empty.
- [ ] `rollouts/update_0000_episode_0000/smolvla_first_rollout.mp4` exists and is non-empty.
- [ ] `wm_decode_selected_strip.png` exists when `--save-wm-decodes` is enabled.
- [ ] reset parity passes strict thresholds.
- [ ] no WM scoring fallback occurred.
- [ ] candidate score std is non-zero, or skipped update reason is explicit.
- [ ] action telemetry includes raw/clipped/JEPA-normalized stats.

### Failure Policy

- [ ] Any traceback, NaN, missing required artifact, strict reset mismatch, WM scoring fallback, or missing completion marker is failure.
- [ ] Diagnose from first failing invariant.
- [ ] Use `/exa-web-search` only when live external docs/issues are needed.
- [ ] Fix code or Slurm script.
- [ ] Re-run relevant unit/static tests.
- [ ] Requeue from failed point.
- [ ] Do not claim success from import/load/checkpoint lines alone.

## Task 12: 10-Episode Runs

**Precondition:** both smoke jobs pass by Task 11 criteria.

- [ ] **Step 12.1: Check live GPU capacity**

Run:

```bash
squeue -u "$USER" -o "%.18i %.9P %.20j %.8T %.10M %.6D %R"
```

Decide:

- If total user GPU usage allows two more one-GPU jobs without exceeding 3 GPUs, submit both 10-episode jobs concurrently.
- Otherwise submit `bounded_executed` with `afterok` dependency on `official_jepa_mirror`.
- If uncertain, choose `afterok`.

- [ ] **Step 12.2: Submit official 10-episode run**

Run:

```bash
cd /vol/bitbucket/aa6622/project
jid10_official=$(sbatch --parsable --chdir=/vol/bitbucket/aa6622/project --export=NIL \
  scripts/grpo/submit_phase12_wm_chunk_grpo.slurm official_jepa_mirror 10)
echo "${jid10_official}"
```

The Slurm script should read `$1` as action profile and `$2` as episode count.

- [ ] **Step 12.3: Submit bounded 10-episode run**

Concurrent option, only if capacity allows:

```bash
jid10_bounded=$(sbatch --parsable --chdir=/vol/bitbucket/aa6622/project --export=NIL \
  scripts/grpo/submit_phase12_wm_chunk_grpo.slurm bounded_executed 10)
```

Chained option:

```bash
jid10_bounded=$(sbatch --parsable --dependency=afterok:${jid10_official} --chdir=/vol/bitbucket/aa6622/project --export=NIL \
  scripts/grpo/submit_phase12_wm_chunk_grpo.slurm bounded_executed 10)
```

- [ ] **Step 12.4: Monitor 10-episode jobs**

Use the same monitoring loop as smoke.

10-episode pass criteria:

- [ ] job exits `0`.
- [ ] stdout contains `PHASE12_WM_CHUNK_GRPO_OK`.
- [ ] `progress.jsonl` has 10 completed episode rows or configured completion marker.
- [ ] `checkpoints/latest.pt` exists.
- [ ] all episodes have WM score fields, selected candidate, action telemetry, and success diagnostics.
- [ ] no silent fallback to synthetic/non-WM reward.

## Subagent-Driven Execution Order

Use one implementer subagent per task. Do not run implementers in parallel if they touch overlapping files.

Suggested subagent order:

1. [ ] Task 1 implementer: full-chunk policy API.
2. [ ] Task 1 spec reviewer.
3. [ ] Task 1 code-quality reviewer.
4. [ ] Task 2 implementer: objective module.
5. [ ] Task 2 spec reviewer.
6. [ ] Task 2 code-quality reviewer.
7. [ ] Task 3 implementer: action profiles.
8. [ ] Task 3 spec reviewer.
9. [ ] Task 3 code-quality reviewer.
10. [ ] Task 4 implementer: goals/reset.
11. [ ] Task 4 spec reviewer.
12. [ ] Task 4 code-quality reviewer.
13. [ ] Task 6 implementer: diagnostics.
14. [ ] Task 6 spec reviewer.
15. [ ] Task 6 code-quality reviewer.
16. [ ] Task 5 implementer: rollout integration.
17. [ ] Task 5 spec reviewer.
18. [ ] Task 5 code-quality reviewer.
19. [ ] Task 7/8 implementer: trainer/logging/artifacts.
20. [ ] Task 7/8 spec reviewer.
21. [ ] Task 7/8 code-quality reviewer.
22. [ ] Task 9 implementer: Slurm scripts.
23. [ ] Task 9 spec reviewer.
24. [ ] Task 9 code-quality reviewer.
25. [ ] Task 10 integration verification.
26. [ ] Task 11/12 GPU execution/monitoring.
27. [ ] Final whole-branch code review.

Reviewer prompts must check:

- no CEM in mainline,
- objective formula exactly matches `visual_mse + 0.1 * proprio_mse`,
- `official_jepa_mirror` does not clip scoring actions,
- chunk ratio uses summed logprobs,
- no silent WM scoring fallback,
- no nested `sbatch`,
- smoke artifacts and markers exist.

## Final Morning Summary Requirements

When autonomous work stops, report:

- [ ] commit/status summary, without claiming commits if none were made,
- [ ] tests run and exact pass/fail result,
- [ ] job IDs submitted,
- [ ] stdout paths,
- [ ] artifact run dirs,
- [ ] smoke pass/fail evidence,
- [ ] 10-episode pass/fail evidence if reached,
- [ ] root causes and fixes for any failures,
- [ ] remaining blockers and next action.

