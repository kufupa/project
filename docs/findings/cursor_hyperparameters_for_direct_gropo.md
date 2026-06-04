# Hyperparameters for direct GROPO run
_Exported on 6/2/2026 at 16:56:29 PDT from Cursor (3.6.31)_

---

**User**

Here's the plan from another LLM. I believe it has an outdated version of the RLHF library that we are using though, so your goal is to redo the plan. Which has information about the hyperparameters for the direct GROPO run that we conducted that got very high results, specifically it got 35%. On a one hundred episode evaluation. So your goal is Read through our current implementation and figure out for me if I wanted to push the RL Inf library. There's And extend it for an official GitHub. Pull request, what kind of extension I need to modify to our local library? In order to allow for the run where we got thirty-five percent to be reproduced And I would use the information that I'll provide from the other LLM as a starting spot because I already did a bunch of exploring, but you should critically analyze all of it because it will most certainly be different to where our Library stores and operates because we are a few commits ahead.

ur goal isn't to convince me of anything its to keep researching till ur 95% confident u understand the resources u need i.e code, logs, doc and can explain whatever it does!
use /caveman ultra and subagents u deploy tell them to use the same!


---
name: RLinf GRPO Parity
overview: Map blockers between Phase11 direct GRPO (35% eval100 @ j2_n2 u25) and RLinf library paths; plan RLinf extensions to rerun that recipe without the direct wrapper.
todos:
  - id: parity-spec
    content: Write metaworld_pushv3_grpo_phase11_parity.md with N2+j2 hparams, eval protocol, behavior checklist
    status: pending
  - id: grpo-smolvla-yaml
    content: Add metaworld_pushv3_grpo_smolvla.yaml (SmolVLA env + GRPO + G32 + dense_return + clip0.05)
    status: pending
  - id: log-std-noise-config
    content: Expose init_log_std and euler_step_noise_std in SmolVLA model config + init
    status: pending
  - id: phase11-ckpt-adapter
    content: Implement Phase11 update_*.pt load/export + resume LR override in RLinf runner
    status: pending
  - id: zero-adv-skip
    content: Add zero-advantage skip parity in embodied GRPO actor path
    status: pending
  - id: pbs-n2-j2-chain
    content: Create RLinf PBS chain N2 u20 + j2 resume u25 + eval100 sweep
    status: pending
  - id: eval-parity
    content: Align RLinf eval script seeds/metric/updates with Phase11 eval100
    status: pending
  - id: loss-scale-audit
    content: Audit and test group_level loss scale vs Phase11 1/(n_chunks*G)
    status: pending
isProject: false
---

# RLinf GRPO Parity for 35% Push-v3 Run

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recreate the **35% eval100** Phase11 GRPO run inside **RLinf** (extend library, retire direct wrapper dependency).

**Architecture:** Phase11 = single-process `train_phase11_env_on_policy_grpo.py` + `src/smolvla_grpo/*`. RLinf has **3 lanes**; none today match the 35% recipe. Closest core math exists in Ray embodied GRPO (`adv_type: grpo`), but SmolVLA MetaWorld wiring, hyperparam surface, checkpoint/eval chain, and rollout scale are wrong or missing.

**Tech stack:** RLinf Ray embodied runner, Hydra YAML, `SmolVLAMetaWorldEnv`, `SmolVLAForRLActionPrediction`, Phase11 reference in [`project/scripts/grpo/`](project/scripts/grpo/) + [`project/src/smolvla_grpo/`](project/src/smolvla_grpo/).

---

## Target run (what “35%” actually is)

**Not** a single fresh train. **Two-phase chain:**

```mermaid
flowchart LR
  HF[HF smolvla_metaworld] --> N2
  N2["N2 fresh u0-20\nG32 lr6e-6 clip0.05 lownoise"] -->|resume u15| J2
  J2["j2 resume u15-25\nG32 lr2e-6 clip0.05"] --> E100["eval100 u25\npc_success=35%"]
```

| Phase | Run label | Updates | Key hparams | eval100 peak |
|-------|-----------|---------|-------------|--------------|
| 1 | `n2_g32_lr6e6_clip005_lownoise` | 0→20 | G32, lr **6e-6**, clip **0.05**, init_log_std **-2.5**, euler **0.1**, dense_return | **34% @ u15** (28% @ u20) |
| 2 | `j2_n2_resume_lr2_clip005` | 15→25 | resume N2 u15, lr **2e-6** (CLI override), clip 0.05, inherited log_std/noise | **35% @ u25** |

**Evidence:**
- [`project/artifacts/phase11_next8_lightweight/j2_n2_resume_lr2_clip005/eval100_u0020_0025_stride5_nenv25_async/eval_sweep_summary.json`](project/artifacts/phase11_next8_lightweight/j2_n2_resume_lr2_clip005/eval100_u0020_0025_stride5_nenv25_async/eval_sweep_summary.json) — u20=32%, u25=**35%**
- Manifests: [`.../j2_n2.../train_manifest.json`](project/artifacts/phase11_next8_lightweight/j2_n2_resume_lr2_clip005/train_manifest.json), [`.../n2.../train_manifest.json`](project/artifacts/phase11_pushv3_n2_g32_lr6e6_clip005_lownoise_u20_20260602/train_manifest.json)

**Eval protocol (selection metric):** 100 ep, seeds 1000–1099, n_envs=25, chunk=5, max_steps=120, metric **`pc_success`** via [`project/scripts/grpo/eval_phase111_grpo_sweep.py`](project/scripts/grpo/eval_phase111_grpo_sweep.py).

**Note:** Codebase has **GRPO**, not “GPO”. No separate GPO symbol found.

---

## RLinf lanes today

```mermaid
flowchart TB
  subgraph phase11 [Phase11 direct GRPO - 35% lives here]
    T11[train_phase11_env_on_policy_grpo.py]
    S11[src/smolvla_grpo]
  end
  subgraph rlinf [RLinf fork]
    RayGRPO["Ray embodied GRPO\ntrain_embodied_agent.py\nnative_grpo yaml"]
    RayPPO["Ray SmolVLA PPO\nmetaworld_pushv3_ppo_smolvla.yaml\nadv_type=gae"]
    DirectPPO["Direct PPO monolith\nrun_smolvla_metaworld_direct_ppo.py\nall smolvla_metaworld_*.pbs"]
  end
  phase11 -->|35% eval100| OK[works]
  RayGRPO -->|wrong env type| BAD1[blocked]
  RayPPO -->|GAE not GRPO| BAD2[blocked]
  DirectPPO -->|GAE not GRPO| BAD3[blocked]
```

| Lane | Entry | Algorithm | Env | Can hit 35% today? |
|------|-------|-----------|-----|-------------------|
| Phase11 direct | [`train_phase11_env_on_policy_grpo.py`](project/scripts/grpo/train_phase11_env_on_policy_grpo.py) | GRPO group adv + PPO clip | official LeRobot vector_async | **Yes** |
| RLinf Ray GRPO | [`metaworld_pushv3_native_grpo_smolvla.yaml`](project/RLinf/examples/embodiment/config/metaworld_pushv3_native_grpo_smolvla.yaml) | GRPO | **`MetaWorldEnv`** (native pixels, not SmolVLA) | **No** |
| RLinf Ray PPO | [`metaworld_pushv3_ppo_smolvla.yaml`](project/RLinf/examples/embodiment/config/metaworld_pushv3_ppo_smolvla.yaml) | GAE + critic | SmolVLA env | **No** (wrong algo) |
| RLinf direct PPO | [`run_smolvla_metaworld_direct_ppo.py`](project/RLinf/scripts/run_smolvla_metaworld_direct_ppo.py) | GAE | SmolVLA env | **No** (wrong algo) |

Handoff confirms mature SmolVLA PBS = PPO only: [`project/RLinf/docs/handoffs/2026-05-26-smolvla-metaworld-ppo-grpo.md`](project/RLinf/docs/handoffs/2026-05-26-smolvla-metaworld-ppo-grpo.md). Phase11 eval chain documented in [`project/RLinf/docs/handoffs/cursor_good_strips_for_push_v3_task.md`](project/RLinf/docs/handoffs/cursor_good_strips_for_push_v3_task.md) L1944–2007 but implemented outside RLinf PBS.

---

## Algorithm parity (Phase11 vs RLinf GRPO core)

| Piece | Phase11 direct | RLinf embodied GRPO | Parity? |
|-------|----------------|---------------------|---------|
| Group advantage | `(R-mean)/std` per G; skip update if std≈0 | [`compute_grpo_advantages`](project/RLinf/rlinf/algorithms/advantages.py) L89–121 | **Partial** — RLinf lacks zero-std skip |
| Return signal | sum step rewards (`dense_return`) | episode score via [`calculate_scores`](project/RLinf/rlinf/algorithms/utils.py) L134–152 | **Yes** if env reward matches |
| Clip | symmetric `clip_eps=0.05` | asymmetric `clip_ratio_high/low` (native yaml 0.28/0.2) | **No** |
| Loss unit | per **policy_chunk** (5-step logp sum) | `reward_type/logprob_type: chunk_level` | **Partial** — verify grad scale |
| Grad scale | divide by `n_chunks × G` | `group_level` loss scale in [`loss_scales.py`](project/RLinf/rlinf/algorithms/loss_scales.py) | **Needs audit** |
| Old policy | deepcopy refresh each update | rollout worker weight sync | **Different mechanism** |
| Trainable params | `lm_expert` + `log_std` only | [`freeze_for_ppo`](project/RLinf/rlinf/models/embodiment/smolvla/smolvla_action_model.py) L167–188 (+ value_head if PPO) | **Yes** for GRPO actor-only |
| Entropy bonus | none | `entropy_bonus: 0` | **Yes** |
| init_log_std / euler noise | CLI `--init-log-std`, `--euler-step-noise-std` | **Not in Hydra/model config** | **No** |
| Resume LR | CLI `--lr` overrides checkpoint opt LR | `runner.resume_dir` FSDP; no documented LR override | **No** |
| Updates per job | 20 + 10 chained | `runner.max_epochs` / save_interval | **Needs chain PBS** |

---

## Blockers (ranked)

### B1 — No SmolVLA + GRPO config (critical)
Only GRPO yaml for MetaWorld Push-v3 pulls **native** env:

```yaml
# metaworld_pushv3_native_grpo_smolvla.yaml L2-3
- env/metaworld_pushv3_native@env.train  # env_type: metaworld
```

SmolVLA model expects LeRobot pixels/proprio via [`env/metaworld_smolvla_pushv3.yaml`](project/RLinf/examples/embodiment/config/env/metaworld_smolvla_pushv3.yaml) → [`SmolVLAMetaWorldEnv`](project/RLinf/rlinf/envs/metaworld/smolvla_metaworld_env.py). **Model/env mismatch** on native GRPO path.

### B2 — Mature RLinf jobs are PPO not GRPO (critical)
All `smolvla_metaworld_direct_ppo_*.pbs` call direct PPO with `--advantage-mode gae`. PPO yaml sets `adv_type: gae`, `group_size: 1`. Cannot reproduce group-relative training without new config + runner path.

### B3 — Hyperparameter surface gap (critical for exact rerun)
Phase11 knobs with **no RLinf equivalent**:

| Phase11 CLI | 35% recipe value | RLinf today |
|-------------|------------------|-------------|
| `--group-size` | 32 | native yaml **2**; PPO yaml **1** |
| `--clip-eps` | 0.05 symmetric | high/low asymmetric |
| `--lr` | 6e-6 then 2e-6 resume | flat `actor.optim.lr` |
| `--init-log-std` | -2.5 (N2 only) | not configurable |
| `--euler-step-noise-std` | 0.1 (N2) | not configurable |
| `--train-seed-base` | 2000 + u | `reset_seed_base: 2000` exists but update coupling differs |
| `--save-every-list` | 2,5 union | `save_interval` only |
| `--resume` + LR override | j2 from N2 u15 | FSDP `global_step_*` vs `update_*.pt` |

### B4 — Rollout scale / topology (high)
Phase11: **G=32** rollouts/update, single GPU, `vector_async` + `forkserver`, full **120** steps/update.

RLinf native GRPO PBS: `total_num_envs=4`, `group_size=2`, Ray actor/env/rollout workers ([`native_metaworld_pushv3_grpo_learning_4env.pbs`](project/RLinf/scripts/pbs/native_metaworld_pushv3_grpo_learning_4env.pbs)). Default yaml `max_steps_per_rollout_epoch: **20**` (PBS overrides to 120 for that script only).

**Risk:** G=32 + 120 steps + SmolVLA forward may need new env batching / memory profile — not validated in RLinf.

### B5 — Train reward default mismatch (high)
35% recipe uses **`dense_return`** (Phase11 [`reward_backends.py`](project/src/smolvla_grpo/reward_backends.py)). RLinf SmolVLA yaml defaults **`sparse_success_delta`**. Env supports dense ([`smolvla_metaworld_env.py`](project/RLinf/rlinf/envs/metaworld/smolvla_metaworld_env.py) L206–208) but GRPO yaml never sets it.

### B6 — Eval protocol mismatch (high for apples-to-apples 35%)
| | Phase11 | RLinf |
|---|---------|-------|
| script | `eval_phase111_grpo_sweep.py` | `eval_smolvla_metaworld_ckpt_sweep.py` |
| seeds | **1000+** | **50000+** default |
| metric | **`pc_success`** | **`success_rate`** |
| chain | train→eval25→eval100 PBS | separate eval qsub |

Cannot claim parity without Phase11 eval wired post-train or RLinf eval aligned.

### B7 — Checkpoint / resume incompatibility (high)
Phase11: [`save_grpo_checkpoint`](project/src/smolvla_grpo/checkpointing.py) → `update_NNNN.pt` with `policy_state_dict`, `optimizer_state_dict`, `update_index`.

RLinf Ray: `checkpoints/global_step_{N}/actor` FSDP shards via [`embodied_runner`](project/RLinf/rlinf/runners/embodied_runner.py).

**Two-phase j2 recipe** needs: load N2 u15 → train 10 updates with new LR → eval u20/u25. No bridge loader exists RLinf←Phase11.

### B8 — Missing training behaviors (medium)
- **Zero-advantage skip:** Phase11 skips optimizer step, still saves ckpt ([`train_phase11_env_on_policy_grpo.py`](project/scripts/grpo/train_phase11_env_on_policy_grpo.py) L788–838). RLinf: no equivalent.
- **Duplicate adapter code:** Phase11 uses [`project/src/smolvla_grpo/lerobot_metaworld_adapter.py`](project/src/smolvla_grpo/lerobot_metaworld_adapter.py); RLinf has [`rlinf/envs/metaworld/lerobot_adapter.py`](project/RLinf/rlinf/envs/metaworld/lerobot_adapter.py). Handoff says wrapper moved to RLinf for PPO — Phase11 path still separate.

### B9 — Ray vector modes still open (medium)
Handoff scoreboard ([`cursor_good_strips_for_push_v3_task.md`](project/RLinf/docs/handoffs/cursor_good_strips_for_push_v3_task.md) L788): RLinf WM/vector/Ray spike **open**. Production parity lane not proven.

---

## Recommended RLinf extension strategy

**Do not** try to flip native GRPO yaml alone. **Add new first-class lane:**

`metaworld_pushv3_grpo_smolvla.yaml` = copy PPO smolvla env/model blocks + GRPO algorithm blocks + Phase11 hparam defaults.

```mermaid
flowchart LR
  YAML[new grpo_smolvla yaml] --> Runner[train_embodied_agent.py]
  Runner --> Env[SmolVLAMetaWorldEnv dense_return G32]
  Runner --> Actor[SmolVLA actor adv_type=grpo]
  Runner --> Ckpt[Phase11-compatible ckpt adapter]
  Ckpt --> Eval[eval_phase111 or RLinf parity eval]
```

**Fallback if Ray G32 too heavy:** extend [`run_smolvla_metaworld_direct_ppo.py`](project/RLinf/scripts/run_smolvla_metaworld_direct_ppo.py) → **`run_smolvla_metaworld_direct_grpo.py`** inside RLinf (library script, not `project/scripts/grpo/`), reusing RLinf env+model but Phase11 rollout loop — still “RLinf library” but avoids Ray until B9 closed.

---

## Implementation tasks

### Task 1: Parity spec doc (ground truth)

**Files:**
- Create: [`project/RLinf/docs/specs/metaworld_pushv3_grpo_phase11_parity.md`](project/RLinf/docs/specs/metaworld_pushv3_grpo_phase11_parity.md)

- [ ] **Step 1:** Copy hyperparam table from N2 + j2 manifests + eval100 summary (above).
- [ ] **Step 2:** Document eval protocol (seeds, n_envs, chunk, metric) verbatim from [`eval_phase111_grpo_sweep.py`](project/scripts/grpo/eval_phase111_grpo_sweep.py).
- [ ] **Step 3:** List Phase11-only behaviors (zero-adv skip, resume LR override, save_every_list).
- [ ] **Step 4:** Commit spec before code changes.

### Task 2: New Hydra config — SmolVLA + GRPO

**Files:**
- Create: [`project/RLinf/examples/embodiment/config/metaworld_pushv3_grpo_smolvla.yaml`](project/RLinf/examples/embodiment/config/metaworld_pushv3_grpo_smolvla.yaml)
- Modify: [`project/RLinf/examples/embodiment/config/env/metaworld_smolvla_pushv3.yaml`](project/RLinf/examples/embodiment/config/env/metaworld_smolvla_pushv3.yaml) (optional overrides)

- [ ] **Step 1:** Base on [`metaworld_pushv3_ppo_smolvla.yaml`](project/RLinf/examples/embodiment/config/metaworld_pushv3_ppo_smolvla.yaml) env/model sections.
- [ ] **Step 2:** Set `algorithm.adv_type: grpo`, `algorithm.group_size: 32`, `add_value_head: False`, `loss_type: actor`.
- [ ] **Step 3:** Set symmetric clip: `clip_ratio_high: 0.05`, `clip_ratio_low: 0.05`.
- [ ] **Step 4:** Set `env.train.reward_mode: dense_return`, `max_steps_per_rollout_epoch: 120`, `max_episode_steps: 120`, `total_num_envs: 32`, `group_size: 32`.
- [ ] **Step 5:** Set `actor.optim.lr: 6.0e-6`, `update_epoch: 1`, `rollout_epoch: 1`.
- [ ] **Step 6:** Smoke: `train_embodied_agent.py --config-name metaworld_pushv3_grpo_smolvla runner.max_epochs=1` on 1 GPU.

### Task 3: Expose init_log_std + euler noise in SmolVLA model config

**Files:**
- Modify: [`project/RLinf/rlinf/models/embodiment/smolvla/smolvla_action_model.py`](project/RLinf/rlinf/models/embodiment/smolvla/smolvla_action_model.py)
- Modify: [`project/RLinf/examples/embodiment/config/model/smolvla.yaml`](project/RLinf/examples/embodiment/config/model/smolvla.yaml)

- [ ] **Step 1:** Add config keys `init_log_std: -2.0`, `euler_step_noise_std: 0.2`.
- [ ] **Step 2:** On model init (fresh only): set `model.log_std` from `init_log_std`; store noise std on policy/model like Phase11 [`policy_wrapper.set_euler_step_noise_std`](project/src/smolvla_grpo/policy_wrapper.py) L372–378.
- [ ] **Step 3:** Unit test: loaded values match CLI/YAML.

### Task 4: Phase11 checkpoint adapter for RLinf resume

**Files:**
- Create: [`project/RLinf/rlinf/utils/phase11_grpo_checkpoint.py`](project/RLinf/rlinf/utils/phase11_grpo_checkpoint.py)
- Modify: [`project/RLinf/rlinf/runners/embodied_runner.py`](project/RLinf/rlinf/runners/embodied_runner.py) (or actor worker resume path)

- [ ] **Step 1:** Implement `load_phase11_checkpoint(path) -> policy_state, optimizer_state, update_index`.
- [ ] **Step 2:** Implement `export_phase11_checkpoint(...)` for round-trip eval with existing sweep script.
- [ ] **Step 3:** Support **LR override on resume** (mirror Phase11 L594+ resume behavior in [`train_phase11_env_on_policy_grpo.py`](project/scripts/grpo/train_phase11_env_on_policy_grpo.py)).
- [ ] **Step 4:** Test: load N2 u15 ckpt from ephemeral path, run 1 update, export `update_0016.pt`.

### Task 5: Zero-advantage skip (optional parity)

**Files:**
- Modify: [`project/RLinf/rlinf/workers/actor/fsdp_actor_worker.py`](project/RLinf/rlinf/workers/actor/fsdp_actor_worker.py)
- Modify: [`project/RLinf/rlinf/algorithms/advantages.py`](project/RLinf/rlinf/algorithms/advantages.py)

- [ ] **Step 1:** Detect group std < 1e-8 after GRPO adv compute.
- [ ] **Step 2:** Skip actor backward; still increment global step + save ckpt (match Phase11).
- [ ] **Step 3:** Log `zero_advantage_skips` in metrics JSON.

### Task 6: PBS chain — N2 then j2 resume + eval100

**Files:**
- Create: [`project/RLinf/scripts/pbs/smolvla_metaworld_grpo_n2_j2_chain.pbs`](project/RLinf/scripts/pbs/smolvla_metaworld_grpo_n2_j2_chain.pbs)
- Reuse: [`project/scripts/grpo/eval_phase111_grpo_sweep.py`](project/scripts/grpo/eval_phase111_grpo_sweep.py) (until Task 7 done)

- [ ] **Step 1:** Phase A — fresh N2: 20 updates, G32, lr 6e-6, init_log_std -2.5, euler 0.1, save every 5.
- [ ] **Step 2:** Phase B — resume u15, 10 updates, lr 2e-6, save 2+5 union.
- [ ] **Step 3:** Call eval100 on u20,u25 with seeds 1000+, n_envs 25.
- [ ] **Step 4:** Success gate: **≥34% @ u20, ≥35% @ u25** (match or beat Phase11).

### Task 7: RLinf-native eval100 parity (optional but recommended)

**Files:**
- Modify: [`project/RLinf/scripts/eval_smolvla_metaworld_ckpt_sweep.py`](project/RLinf/scripts/eval_smolvla_metaworld_ckpt_sweep.py)

- [ ] **Step 1:** Add `--eval-seed-start 1000`, `--metric-name pc_success` aligned with Phase11.
- [ ] **Step 2:** Add `--updates` explicit list (mirror eval sweep).
- [ ] **Step 3:** Cross-check: same ckpt scored by both eval scripts within ±1pp.

### Task 8: Grad-scale audit

**Files:**
- Modify: [`project/RLinf/rlinf/algorithms/loss_scales.py`](project/RLinf/rlinf/algorithms/loss_scales.py)
- Test: create [`project/RLinf/tests/unit_tests/test_grpo_phase11_loss_scale.py`](project/RLinf/tests/unit_tests/test_grpo_phase11_loss_scale.py)

- [ ] **Step 1:** Log effective loss multiplier for G=32, 24 chunks, chunk_level logprob.
- [ ] **Step 2:** Compare to Phase11 `1/(n_chunks*G)`; patch if mismatch >5%.

---

## What you can do today without code (partial)

| Action | Gets you | Missing |
|--------|----------|---------|
| Run RLinf direct PPO dense | dense reward SmolVLA train | GAE not GRPO; wrong eval seeds |
| Run native Ray GRPO PBS | GRPO math | wrong env; G=2; sparse reward |
| Eval Phase11 ckpt with RLinf eval script | quick sanity | seed/metric mismatch |
| Import Phase11 ckpt into RLinf actor | **blocked** — no adapter (B7) |

---

## Success criteria

1. Single RLinf PBS chain reproduces **≥35% pc_success @ u25** on eval100 (seeds 1000+, 100 ep).
2. Hyperparams live in **RLinf YAML/PBS**, not `project/scripts/grpo/` wrapper.
3. Checkpoint resume N2→j2 with LR override works inside RLinf.
4. Spec doc + tests lock parity behaviors (skip-adv, clip, dense_return, G32).

---Here's the plan from another LLM. I believe it has an outdated version of the RLHF library that we are using though, so your goal is to redo the plan. Which has information about the hyperparameters for the direct GROPO run that we conducted that got very high results, specifically it got 35%. On a one hundred episode evaluation. So your goal is Read through our current implementation and figure out for me if I wanted to push the RL Inf library. There's And extend it for an official GitHub. Pull request, what kind of extension I need to modify to our local library? In order to allow for the run where we got thirty-five percent to be reproduced And I would use the information that I'll provide from the other LLM as a starting spot because I already did a bunch of exploring, but you should critically analyze all of it because it will most certainly be different to where our Library stores and operates because we are a few commits ahead.

ur goal isn't to convince me of anything its to keep researching till ur 95% confident u understand the resources u need i.e code, logs, doc and can explain whatever it does!
use /caveman ultra and subagents u deploy tell them to use the same!




---

**Cursor**

Plan from other LLM = starting point only. Must verify against current code + 35% run artifacts. Launch parallel digs now.


---

**User**

the artifacts aren't on this machine ! thats why i gave u the llm context

---

**Cursor**

Artifacts off-box — dig local code only. LLM plan = hypothesis; verify in repo now.


Artifacts mostly off-box — code on disk is ground truth. Old plan partly stale; below = verified map + corrected PR plan.

---

## What 35% actually is (from your LLM context; not re-verifiable here)

Two-job chain, not one train:

| Phase | Label | Updates | Key knobs |
|-------|-------|---------|-----------|
| N2 | `n2_g32_lr6e6_clip005_lownoise` | 0→20 | G=32, lr=6e-6, clip=0.05, init_log_std=-2.5, euler=0.1, **dense_return** |
| j2 | `j2_n2_resume_lr2_clip005` | resume u15 → 25 | lr=2e-6, clip=0.05, inherit log_std/noise |

**Eval gate:** 100 ep, seeds 1000–1099, n_envs=25, chunk=5, max_steps=120, metric **`pc_success`** via `project/scripts/grpo/eval_phase111_grpo_sweep.py`.

One partial artifact *is* on this machine: `g6_u100.../update_0025/eval_summary.json` → **40% @ 10ep** (not your N2/j2 35% @ 100ep). Treat 35% as external ledger truth.

---

## Repo layout (old plan wrong here)

| Old plan says | Actual |
|---------------|--------|
| `project/RLinf/` | **Does not exist** |
| handoffs under `project/RLinf/docs/` | Docs live in **`RLinf-smolvla-metaworld-ppo-grpo/docs/`** |
| `metaworld_pushv3_grpo_smolvla.yaml` | **Missing** — only `metaworld_pushv3_native_grpo_smolvla.yaml` + `metaworld_pushv3_ppo_smolvla.yaml` |

**35% winner today:** `project/scripts/grpo/train_phase11_env_on_policy_grpo.py` + `project/src/smolvla_grpo/*` — **outside** RLinf library.

---

## RLinf lanes today (read from code)

```mermaid
flowchart TB
  P11[Phase11 direct GRPO] -->|35% target| OK
  R1[native_grpo_smolvla yaml] -->|env_type metaworld native pixels| BAD
  R2[ppo_smolvla yaml] -->|adv_type gae + value head| BAD
  R3[run_smolvla_metaworld_direct_ppo.py] -->|gae/mc only| BAD
```

| Lane | Entry | Algo | Env | Hit 35%? |
|------|-------|------|-----|----------|
| Phase11 | `train_phase11_env_on_policy_grpo.py` | GRPO | official LeRobot `vector_async` | **Yes** (only path) |
| Ray GRPO | `metaworld_pushv3_native_grpo_smolvla.yaml` | GRPO G=2 | **native** MetaWorld | No |
| Ray PPO | `metaworld_pushv3_ppo_smolvla.yaml` | GAE | SmolVLA | No |
| Direct PPO | `run_smolvla_metaworld_direct_ppo.py` | GAE | SmolVLA | No |

Native GRPO PBS (`scripts/pbs/native_metaworld_pushv3_grpo_learning_4env.pbs`): G=2, 4 envs, sparse reward default — far from N2 recipe.

---

## Algorithm parity: Phase11 vs RLinf (critical fixes to old plan)

| Piece | Phase11 (35% recipe) | RLinf embodied GRPO | Gap |
|-------|------------------------|---------------------|-----|
| Group adv | `(R-mean)/std`; **all-zero → skip step** (`grpo_math.compute_group_advantages`) | `advantages.py` L116–117: divide by `std+1e-6`, **no skip** | Yes |
| Return | `dense_return` sum step rewards (`reward_backends`) | `calculate_scores` sums steps — OK **if** env=`dense_return` | yaml defaults **sparse** |
| Clip | symmetric `clip_eps=0.05` | asymmetric high/low (native yaml 0.28/0.2) | Yes |
| Loss scale | step path: **`sum(loss) / (T×G)`** (`train_phase11` ~L926) | `loss_type: actor` → `masked_mean` over batch; **no** `group_level` loss_scales in embodied yaml | **Audit required** — old plan said `1/(n_chunks×G)` wrong for default step rollout |
| Old policy | `deepcopy` each update | rollout weight sync | Different |
| Trainable | `lm_expert` + `log_std` | `freeze_for_ppo` in `smolvla_action_model.py` L167–188 | OK if `add_value_head: False` |
| init_log_std / euler | CLI + `policy_wrapper.set_*` | **Not in** `model/smolvla.yaml` or RLinf model init | Yes |
| Resume + LR | `--resume` loads opt state; **`--lr` not re-applied after load** (L301–308) | FSDP `global_step_*`; no Phase11 `.pt` loader | Yes — j2 lr=2e-6 may need **explicit fix in both** |
| save_every_list | — | **Does not exist**; only `--save-every` | Old plan wrong |
| euler 0.1 | needs `--allow-euler-noise` (L254–257) | N/A | Must document in PBS |

Partial bridge **already exists** (Phase11 → RLinf eval format):

```57:76:project/src/smolvla_grpo/checkpointing.py
def save_rlinf_eval_checkpoint(...):
    payload = {
        "checkpoint_type": "trainable_delta",
        "update": int(update_index) + 1,
        "trainable_model": build_rlinf_eval_trainable_model(...),
```

Reverse: `RLinf-smolvla-metaworld-ppo-grpo/scripts/export_rlinf_ckpt_to_direct_ppo.py` (FSDP → `update_*.pt` for direct PPO eval). **No** Phase11 full `.pt` ↔ FSDP round-trip.

Eval mismatch:

| | Phase11 | RLinf |
|---|---------|-------|
| script | `eval_phase111_grpo_sweep.py` | `eval_smolvla_metaworld_ckpt_sweep.py` |
| seeds | **1000+** | default **50000+** |
| metric | **`pc_success`** | **`success_rate`** |

---

## RLinf PR: what to extend (ranked)

### P0 — must have for reproduction inside library

**1. SmolVLA + GRPO config (new yaml)**  
Create `examples/embodiment/config/metaworld_pushv3_grpo_smolvla.yaml`:
- Base env block from `metaworld_pushv3_ppo_smolvla.yaml` (`env/metaworld_smolvla_pushv3`)
- Algo: `adv_type: grpo`, `loss_type: actor`, `group_size: 32`, `add_value_head: False`
- `clip_ratio_high/low: 0.05`, `reward_mode: dense_return`, `max_steps_per_rollout_epoch: 120`, `total_num_envs: 32`
- `actor.optim.lr: 6e-6`

**2. Model hyperparam surface**  
`rlinf/models/embodiment/smolvla/smolvla_action_model.py` + `examples/embodiment/config/model/smolvla.yaml`:
- `init_log_std`, `euler_step_noise_std`
- Fresh-init set `log_std`; wire euler to policy like Phase11 `policy_wrapper.set_euler_step_noise_std`

**3. Training entry — pick one lane**

| Option | Pros | Cons |
|--------|------|------|
| **A: Ray** `train_embodied_agent.py` + new yaml | “Official” RLinf | G=32 × 120 steps × SmolVLA + Ray never validated; heavy |
| **B: Direct GRPO** `run_smolvla_metaworld_direct_grpo.py` | Mirrors mature `direct_ppo`; single-GPU like Phase11 | New script; less “pure Ray” |

**Recommendation:** PR ships **B first** (fastest path to 35%), **A as follow-up** once G32 Ray smoke passes. Old plan’s yaml-only Ray path is high risk without B.

**4. Phase11 checkpoint adapter**  
New `rlinf/utils/phase11_grpo_checkpoint.py` + hooks in `embodied_runner.py` (or direct runner):
- Load `update_*.pt` (`policy_state_dict`, `optimizer_state_dict`, `update_index`)
- Export same format for eval sweep
- **LR override on resume** (missing in Phase11 today — add for j2 parity)

**5. PBS/slurm chain**  
`scripts/slurm/smolvla_metaworld_grpo_n2_j2_chain.slurm` (or pbs):
- Phase A: N2 20 upd
- Phase B: resume u15, 10 upd, lr=2e-6
- Call Phase11 eval OR aligned RLinf eval

### P1 — parity polish (may matter for exact 35%)

**6. Zero-advantage skip** — `fsdp_actor_worker.py` + optionally `advantages.py` match Phase11 skip-optimize-still-save

**7. Loss-scale audit + unit test** — compare RLinf `masked_mean` + batch=G vs Phase11 `/(T×G)`; patch if >5% drift

**8. Eval parity** — extend `eval_smolvla_metaworld_ckpt_sweep.py`: `--seed-base 1000`, `--metric pc_success`, load Phase11 `.pt` format

### P2 — docs/tests for upstream PR

**9. Spec doc** — `docs/specs/metaworld_pushv3_grpo_phase11_parity.md` (hyperparams from your ledger, not guessed)

**10. Tests** — `tests/unit_tests/test_grpo_phase11_loss_scale.py`, smoke `train_embodied_agent` 1 epoch OR direct_grpo 1 update

---

## Resources you need (95% confidence list)

### Code (on disk)
| Resource | Role |
|----------|------|
| `project/scripts/grpo/train_phase11_env_on_policy_grpo.py` | Reference training loop, hparams, skip-adv |
| `project/src/smolvla_grpo/{grpo_math,policy_wrapper,checkpointing,reward_backends,phase11_rollout}.py` | GRPO math, log_std/euler, ckpt, dense reward |
| `project/scripts/grpo/eval_phase111_grpo_sweep.py` + `eval_phase11_checkpoints.py` | Eval100 protocol |
| `project/scripts/grpo/submit_phase111_single_task_grpo.slurm` | Slurm template (missing clip/reward/euler flags — extend) |
| `RLinf-smolvla-metaworld-ppo-grpo/rlinf/algorithms/{advantages,losses,utils}.py` | GRPO core to patch |
| `RLinf-smolvla-metaworld-ppo-grpo/rlinf/envs/metaworld/smolvla_metaworld_env.py` | dense_return already L213–215 |
| `RLinf-smolvla-metaworld-ppo-grpo/rlinf/models/embodiment/smolvla/smolvla_action_model.py` | Model to extend |
| `RLinf-smolvla-metaworld-ppo-grpo/scripts/run_smolvla_metaworld_direct_ppo.py` | Template for direct GRPO |
| `RLinf-smolvla-metaworld-ppo-grpo/scripts/export_rlinf_ckpt_to_direct_ppo.py` | Partial ckpt bridge |

### External (not in RLinf repo)
| Resource | Role |
|----------|------|
| LeRobot fork venv `lerobot_mw_py310` + `select_action_distr_params` / `log_std` | Required for GRPO (README L5–13) |
| HF `jadechoghari/smolvla_metaworld` | Base weights |
| Off-machine artifacts: N2/j2 manifests, `eval_sweep_summary.json` u25=35% | Ground-truth hparam proof |

### Docs (on disk, secondary)
| Resource | Role |
|----------|------|
| `project/scripts/grpo/README.md` | Phase111 backend, eval seeds |
| `RLinf-smolvla-metaworld-ppo-grpo/docs/superpowers/plans/2026-05-30-smolvla-grpo-flow-to-40pct.md` | 33% G8 line (different recipe than N2/j2 35%) |
| `RLinf-smolvla-metaworld-ppo-grpo/docs/old/cursor_2_comparison_of_flow_ppo_and_sm.md` | Stale-path warnings |

---

## Corrected task list (replaces old YAML todos)

1. **parity-spec.md** — N2+j2 hparams from your ledger + eval protocol verbatim  
2. **metaworld_pushv3_grpo_smolvla.yaml** — SmolVLA env + GRPO + G32 + dense + clip 0.05  
3. **init_log_std + euler** in model config/init  
4. **direct_grpo runner** (recommended) *or* Ray G32 smoke  
5. **phase11_grpo_checkpoint.py** + LR override on resume  
6. **zero-adv skip** (optional parity)  
7. **slurm N2→j2→eval100 chain**  
8. **eval script alignment** (pc_success, seed 1000)  
9. **loss-scale unit test** (T×G not n_chunks×G)

---

## Bottom line

Old plan **direction OK**, **paths and several details wrong**. RLinf fork is `RLinf-smolvla-metaworld-ppo-grpo`; no SmolVLA+GRPO yaml; no direct GRPO; checkpoint/eval bridges incomplete; Phase11 still owns 35% recipe.

Minimal upstream PR: **model knobs + checkpoint adapter + direct GRPO script + yaml + slurm chain + eval parity + spec**. Ray G32 is stretch — prove with smoke before betting reproduction on it.

Want next step: draft `metaworld_pushv3_grpo_smolvla.yaml` + skeleton `run_smolvla_metaworld_direct_grpo.py` in the fork?

