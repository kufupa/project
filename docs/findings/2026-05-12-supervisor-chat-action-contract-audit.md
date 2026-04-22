# Supervisor Chat Action-Contract Audit

Date: 2026-05-12  
Scope: SmolVLA -> MetaWorld -> JEPA-WM action claims from supervisor chat.  
Mode: code-first audit. Project docs are treated as low-trust notes unless backed by code, installed libs, official repos, or web/HF source.

## TL;DR Verdict

| Topic | Verdict | Short reason |
|---|---|---|
| SmolVLA checkpoint action contract | True | Checkpoint config says `ACTION=MEAN_STD`; LeRobot unnormalizer maps `a_norm * std + mean`. |
| MT50 actions exceed `[-1,1]` | True | HF dataset stats show XYZ min/max far outside MetaWorld action box. |
| MetaWorld clips all action dims internally | False / overbroad | MetaWorld clips XYZ in `set_xyz_action`; gripper action is passed raw to simulation/reward. |
| Official SmolVLA eval has no tanh | True | `select_action -> postprocessor -> env step`; SmolVLA flow returns `x_t`, no tanh. |
| Current local GRPO has no tanh | False | Live Phase11 GRPO path uses `MetaWorldSmolVLAGRPOPolicy`, which tanh-squashes before postprocessor. |
| JEPA-WM stats/normalization | True | JEPA-WM repo hardcodes MetaWorld action mean/std and uses `(a - mean) / std`. |
| 5x4 -> 20D WM packing | True | JEPA-WM config uses `frameskip: 5`, base action dim 4; hub loader multiplies dims; local scorer packs factor 5. |
| SmolVLA + JEPA-WM in current GRPO trainer | Not implemented | Phase11 trainer uses env rewards only; WM reward backend is placeholder. Segment/CEM WM scoring exists separately. |

## Source Trust Map

| Tier | Source | Trust note |
|---|---|---|
| T0 | HF raw URLs, MetaWorld docs, installed upstream package source | Highest for public contracts/stats. |
| T1 | `VGG JEPA/jepa-wms` official clone | High for JEPA-WM contract. |
| T2 | `VGG JEPA/safe-robot-steering` reference repo | Medium-high for LIBERO GRPO reference only. |
| T3 | Installed LeRobot/MetaWorld venv in `.envs/lerobot_mw_py310` | High for what this workspace runs, but version-specific. |
| T4 | Local `project/src`, `project/scripts` | Truth for current implementation, low trust for protocol claims. |
| T5 | Local docs/artifacts | Useful memory, not proof unless backed by T0-T4. |

## Claim Table

| ID | Human claim | Verdict | Implemented? | Source evidence | Audit note |
|---|---|---|---|---|---|
| C01 | `lerobot/metaworld_mt50` actions are not scaled to `[-1,1]`; they are expert/raw actions. | Mostly true, with caveat. | Dataset/source true. | HF stats via Exa: action min `[-11.47,-14.93,-17.28,-1]`, max `[19.46,12.47,24.63,1]`; local report cites same at `artifacts/curReporttext.tex:525-535`. | Data definitely not globally bounded. "Expert/raw before env clip" is strong inference, not maintainer-confirmed. |
| C02 | SmolVLA training stats for checkpoint are mean `[0.25,0.51,-0.13,0.38]`, std `[2.03,2.01,3.39,0.71]`. | True. | Config/data true. | HF stats via Exa; report snapshot `docs/findings/2026-05-10-action-normalization-wm-contract.md:151-160`. | Exact full values: mean `[0.2547,0.5097,-0.1269,0.3760]`, std `[2.0261,2.0075,3.3882,0.7099]`. |
| C03 | SmolVLA predicts normalized actions because trained with `ACTION=MEAN_STD`. | True. | Yes in LeRobot checkpoint path. | Checkpoint config: `config.json:43-47`; train config via Exa says `normalization_mapping.ACTION=MEAN_STD`; LeRobot processor does MEAN_STD at `.envs/.../normalize_processor.py:325-338`. | "Normalized" here means z-score, not bounded action. |
| C04 | LeRobot postprocessor maps policy output back to env-scale action: `a_raw = a_norm * std + mean`. | True. | Yes in evaluator and LeRobot processor. | Local evaluator loads postprocessor `src/smolvla_pipeline/evaluator.py:748-774`, then calls it at `src/smolvla_pipeline/evaluator.py:983-988`; unnormalizer inverse at `.envs/.../normalize_processor.py:336-338` and `.envs/.../normalize_processor.py:474-525`. | Local evaluator clips after postprocessor via `_coerce_exec_action`. |
| C05 | MetaWorld action space is `[-1,1]` with 4 dims. | True. | Yes. | MetaWorld docs via Exa say `Box(-1.0,1.0,(4,),float32)`; installed LeRobot wrapper sets this at `.envs/.../lerobot/envs/metaworld.py:137`; MetaWorld base env at `.envs/.../metaworld/sawyer_xyz_env.py:239-243`. | 4 dims = dx, dy, dz, gripper. |
| C06 | MetaWorld clips out-of-bound actions internally, so OOB is safe/clipped. | Partial / overbroad. | XYZ yes; gripper no equivalent clip. | XYZ clipped in `set_xyz_action` at `.envs/.../metaworld/sawyer_xyz_env.py:320-330`; step passes gripper raw into `do_simulation([action[-1], -action[-1]])` at `.envs/.../metaworld/sawyer_xyz_env.py:591-595`. | Local `_coerce_exec_action` clips all dims before env at `src/smolvla_pipeline/evaluator.py:330-340`, but installed LeRobot env step passes action through at `.envs/.../lerobot/envs/metaworld.py:221-240`. |
| C07 | Performance is best with raw actions passed from LeRobot to MetaWorld. | Not code-proven. | Official eval path approximates this, but local evaluator clips. | Official-like eval flow `select_action -> postprocessor -> _coerce_exec_action` at `src/smolvla_pipeline/evaluator.py:983-988`; report says no action transform in official-path local eval at `artifacts/curReporttext.tex:535`. | "Best" needs experiment table/log. Code can prove path, not performance claim. |
| C08 | There is no tanh for MetaWorld; tanh only in LIBERO because LIBERO requires `[-1,1]`. | True for official eval; false for current local GRPO. | Official eval no; Phase11 GRPO yes. | SmolVLA model returns `x_t` at `.envs/.../modeling_smolvla.py:861-936`; evaluator no tanh at `src/smolvla_pipeline/evaluator.py:983-988`. Current GRPO tanh at `src/smolvla_grpo/policy_wrapper.py:146-167` and `src/smolvla_grpo/policy_wrapper.py:175-218`. LIBERO reference tanh after unnormalizing at `VGG JEPA/safe-robot-steering/model/smolvla_policy.py:224-240`. | This is main bug: current Phase11 MetaWorld GRPO uses tanh in SmolVLA z-score space. |
| C09 | Current local GRPO wrapper is live, not legacy. | True. | Yes, Phase11 trainer + vector rollout use it. | Trainer imports/builds wrapper at `scripts/grpo/train_phase11_env_on_policy_grpo.py:29-36` and `scripts/grpo/train_phase11_env_on_policy_grpo.py:90-105`; rollout uses wrapper at `src/smolvla_grpo/phase11_rollout.py:179-187` and samples/steps at `src/smolvla_grpo/phase11_rollout.py:223-237`; vector path uses wrapper at `src/smolvla_grpo/official_lerobot_vector_rollout.py:77-85` and `src/smolvla_grpo/official_lerobot_vector_rollout.py:113-120`. | Answer to latest question: yes, live. |
| C10 | JEPA-WM consumes env action, then normalizes with its own stats. | True for JEPA-WM contract. | Local segment scorer does this when caller passes env actions. | JEPA `Preprocessor.normalize_actions` at `VGG JEPA/jepa-wms/app/plan_common/datasets/preprocessor.py:30-40`; stats registry at `VGG JEPA/jepa-wms/app/plan_common/datasets/__init__.py:24-36`; local normalizer at `src/segment_grpo_loop.py:374-399`. | Do not feed SmolVLA z-score directly to JEPA-WM. |
| C11 | JEPA-WM stats are mean `[0.0057,0.1574,-0.1396,0.1998]`, std `[0.7359,0.7341,0.7183,0.7432]`. | True. | Yes. | `VGG JEPA/jepa-wms/app/plan_common/datasets/__init__.py:24-36`; `hubconf.py` loads these into preprocessor at `VGG JEPA/jepa-wms/hubconf.py:251-261`. | These are not SmolVLA stats. |
| C12 | JEPA-WM was trained with normalized actions. | True for published config. | Yes in config/data loader. | JEPA config has `normalize_action: true` and `frameskip: 5` at `VGG JEPA/jepa-wms/configs/evals/simu_env_planning/mw/jepa-wm/reach-wall_L2_cem_sourcexp_H6_nas3_ctxt2_r256_alpha0.1_ep48_decode.yaml:127-133`; dataset normalizes actions at `VGG JEPA/jepa-wms/app/plan_common/datasets/metaworld_hf_dset.py:132-145`. | Config is eval model config; stats registry says values match model training at `VGG JEPA/jepa-wms/app/plan_common/datasets/__init__.py:7-18`. |
| C13 | JEPA-WM input action is 20D = pack 5 consecutive 4D actions. | True. | Segment scorer implements. | Hub loader multiplies base action dim by `frameskip` at `VGG JEPA/jepa-wms/hubconf.py:220-232`; JEPA `encode_act` doc says input is `frameskip * action_dim` at `VGG JEPA/jepa-wms/app/vjepa_wm/video_wm.py:199-214`; local pack at `src/segment_grpo_loop.py:402-426`. | For MetaWorld base action dim 4, frameskip 5 => 20. |
| C14 | SmolVLA -> WM should use actions actually sent/stepped in MetaWorld, normalize once, then pack. | Correct design; partially implemented. | Segment scorer normalizes/packs supplied env actions; Phase11 GRPO does not call WM. | Local score path normalizes then packs at `src/segment_grpo_loop.py:1620-1665`; current Phase11 trainer uses `EnvRewardBackend` at `scripts/grpo/train_phase11_env_on_policy_grpo.py:121` and returns from env at `scripts/grpo/train_phase11_env_on_policy_grpo.py:164-168`. | Guarantee depends on caller passing post-clip executed actions. Segment/CEM path can score candidate chunks before execution. |
| C15 | We have SmolVLA + JEPA-WM. | Partial. | Separate segment/CEM WM scoring yes; current Phase11 GRPO no. | WM scoring bridge in `src/segment_grpo_loop.py:1620-1665`; placeholder WM reward backend at `src/smolvla_grpo/reward_backends.py:32-52`; Phase11 trainer hardcodes `EnvRewardBackend` at `scripts/grpo/train_phase11_env_on_policy_grpo.py:121`. | Supervisor-safe wording: "WM scoring exists in segment planner; integrated on-policy SmolVLA GRPO currently uses env reward, not JEPA-WM reward." |
| C16 | "WM input" should be renamed "JEPA-WM input"; coordinate system is JEPA-WM z-score action space. | True. | Docs/code support. | JEPA preprocessor class at `VGG JEPA/jepa-wms/app/plan_common/datasets/preprocessor.py:9-40`; stats at `VGG JEPA/jepa-wms/app/plan_common/datasets/__init__.py:24-36`. | Use "JEPA-WM normalized 20D action" in report/table. |
| C17 | LIBERO tanh reference is valid precedent for MetaWorld tanh. | False as direct precedent. | Reference differs. | LIBERO reference unnormalizes mean then tanh-squashes raw-space sample at `VGG JEPA/safe-robot-steering/model/smolvla_policy.py:224-240`; LIBERO action space `[-1,1]^7` at `VGG JEPA/safe-robot-steering/env/env.py:10-12`. Local MetaWorld wrapper tanh-squashes before postprocessor at `src/smolvla_grpo/policy_wrapper.py:159-163`. | Same math style, wrong coordinate system locally. |

## Bugs / Mismatches Found

| Severity | Bug / mismatch | Evidence | Impact | Fix |
|---|---|---|---|---|
| High | Current Phase11 MetaWorld GRPO tanh-squashes SmolVLA z-score actions before LeRobot postprocessor. | `src/smolvla_grpo/policy_wrapper.py:159-163`, batch path `src/smolvla_grpo/policy_wrapper.py:201-205`; trainer uses wrapper at `scripts/grpo/train_phase11_env_on_policy_grpo.py:90-105`; rollout uses wrapper at `src/smolvla_grpo/phase11_rollout.py:223-237`. | Distorts pretrained action distribution; contradicts "no tanh for MetaWorld"; makes reported action-contract story false for GRPO runs. | Add no-tanh mode/default: sample Gaussian in SmolVLA normalized space, compute plain Gaussian logprob, postprocess, final env clip. Keep tanh as named ablation only. |
| High | Phase11 trainer does not implement JEPA-WM reward despite "SmolVLA + JEPA-WM" wording. | `EnvRewardBackend()` at `scripts/grpo/train_phase11_env_on_policy_grpo.py:121`; `WMLatentRewardBackend` raises unless metadata prefilled at `src/smolvla_grpo/reward_backends.py:32-52`. | Supervisor may think current GRPO optimizes WM score; it optimizes env return. | Wire `score_chunk_by_goal_latent` into rollout metadata, or rename Phase11 as env-reward GRPO only. |
| Medium | "MetaWorld clips actions internally" hides gripper issue. | XYZ clip at `.envs/.../metaworld/sawyer_xyz_env.py:320-330`; raw gripper simulation at `.envs/.../metaworld/sawyer_xyz_env.py:591-595`. | OOB gripper can violate env contract/reward assumptions; local full clip may change benchmark behavior vs raw LeRobot env. | State precisely: env action box is `[-1,1]`; MetaWorld clips XYZ; wrapper should clip all dims if it owns env safety, and report clip rate. |
| Medium | Local evaluator clips postprocessed actions, while installed LeRobot env step itself passes actions through. | Local clip `src/smolvla_pipeline/evaluator.py:330-340`; installed env step `.envs/.../lerobot/envs/metaworld.py:221-240`. | "Official LeRobot raw-to-env" vs "local parity evaluator" are not identical. | Separate official protocol, local safety clip, and ablation labels in report. |
| Medium | "Performance best with raw actions" lacks auditable source in code. | Found code paths/docs, no definitive result table for raw vs tanh/clip. | Claim is weak in supervisor discussion. | Add experiment table: no-tanh/no-extra-transform vs tanh wrapper vs clip-only, same seeds/tasks. |
| Low | Local `WMLatentRewardBackend` name suggests implemented backend, but it only reads existing metadata. | `src/smolvla_grpo/reward_backends.py:32-52`. | Future agent may assume WM scoring is wired. | Rename to `MetadataWMLatentRewardBackend` or implement scoring. |

## Current Live Pipeline Answer

The local GRPO wrapper is live. It is not only legacy.

Current path:

```text
scripts/grpo/train_phase11_env_on_policy_grpo.py
  -> load_bundle_for_grpo(...)
  -> MetaWorldSmolVLAGRPOPolicy(...)
  -> collect_rollout_group(...)
  -> old_wrapper.sample_action_from_proc(...)
  -> torch.tanh(unsquashed)
  -> bundle.postprocessor(...)
  -> _coerce_exec_action(... clip [-1,1])
  -> env.step(...)
```

Vector rollouts use same wrapper:

```text
official_lerobot_vector_rollout.py
  -> MetaWorldSmolVLAGRPOPolicy(...)
  -> sample_action_batch_from_proc(...)
  -> torch.tanh(...)
  -> postprocessor(...)
  -> env_h.step_batch(...)
```

Legacy/separate path:

```text
segment_grpo_loop.py
  -> JEPA-WM score_chunk_by_goal_latent(...)
  -> normalize env actions with JEPA stats
  -> pack 5x4 -> 20D
```

So: current Phase11 GRPO action bug is live; JEPA-WM bridge lives in segment/CEM code, not current Phase11 trainer.

## Correct Supervisor-Safe Wording

Use this:

> SmolVLA MetaWorld checkpoint outputs actions in its own normalized `ACTION=MEAN_STD` coordinate system. LeRobot postprocessing maps those to MetaWorld/env-scale actions using the SmolVLA dataset stats. For JEPA-WM, the semantic bridge should be the env-scale action actually executed; then JEPA-WM preprocessing converts that same physical action into JEPA-WM z-score coordinates and packs 5 consecutive 4D actions into one 20D WM input. Official SmolVLA eval has no tanh. Our current Phase11 GRPO wrapper still applies tanh before postprocessing, so that path should be treated as a LIBERO-inspired ablation/bug, not the clean MetaWorld action contract.

## Needed Fixes Before Paper/Supervisor Claim

| Priority | Action |
|---|---|
| P0 | Change current Phase11 default to no-tanh in SmolVLA normalized space; preserve tanh via explicit flag/name. |
| P0 | Update report/table: "MetaWorld clips XYZ; action space requires all 4 dims in `[-1,1]`; local final clip enforces contract." |
| P1 | If claiming SmolVLA+JEPA-WM GRPO, wire WM scoring into Phase11/Phase12 trainer or rename current trainer to env-reward GRPO. |
| P1 | Add action logging per rollout: normalized sample range, postprocessed raw range, executed clipped range, clip rate, JEPA z-score max. |
| P2 | Run same-seed ablation: no-tanh default vs current tanh-before-postprocessor vs clip-gripper/final-clip variants. |

## Web / External Sources Used

| Source | What it proved |
|---|---|
| `https://huggingface.co/datasets/lerobot/metaworld_mt50/raw/main/meta/stats.json` | MT50 action stats exceed `[-1,1]`; mean/std values. |
| `https://huggingface.co/jadechoghari/smolvla_metaworld/raw/main/train_config.json` | Checkpoint trained on `lerobot/metaworld_mt50`; action feature shape 4; `ACTION=MEAN_STD`; `n_action_steps=1`. |
| `https://metaworld.farama.org/benchmark/action_space/` | Official MetaWorld action-space contract: `Box(-1,1,(4,),float32)`. |
| `https://github.com/huggingface/lerobot/issues/2617` | Public ambiguity around MT50 dataset action values outside env box. |

