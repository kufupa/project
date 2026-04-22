# SmolVLA / MetaWorld / JEPA-WM Action Contract

Date: 2026-05-10  
Purpose: one source-of-truth for future LLM/supervisor questions about action normalization, clipping, tanh, and WM scoring.

## TL;DR

Actions move through **three coordinate systems**, not one:

```text
SmolVLA normalized action
  -> LeRobot postprocessor
MetaWorld/env-scale action
  -> execution clip / env action-box handling
executed MetaWorld action
  -> JEPA-WM preprocessor
JEPA-WM normalized + packed action
```

Best one-liner:

> Each learned module reads actions in its own training coordinate system. Env action is bridge. We convert SmolVLA-normalized -> env-scale -> WM-normalized, and execute/score actions actually sent to MetaWorld.

Formula:

```text
a_smol_norm = SmolVLA output
a_env       = a_smol_norm * smolvla_std + smolvla_mean
a_exec      = clip/action-bound(a_env)
a_wm_norm   = (a_exec - jepa_wm_mean) / jepa_wm_std
a_wm_20d    = concat(5 consecutive a_wm_norm)
```

Do **not** feed SmolVLA z-score output straight into JEPA-WM.  
Do **not** feed raw env numbers into JEPA-WM if checkpoint was trained with normalized actions.  
Correct bridge: **env-scale executed action**, then **JEPA-WM normalization**.

## Main Supervisor FAQ

| Question | Short answer | Evidence / why |
|---|---|---|
| Why not raw SmolVLA output to env? | SmolVLA output = z-score, not env action. | LeRobot config uses `ACTION=MEAN_STD`; postprocessor maps to env units. |
| Why unnormalize then normalize again for WM? | Different models, different coords. | SmolVLA coords -> env coords -> JEPA-WM coords. Not same tensor double-normalized. |
| Why JEPA-WM stats, not SmolVLA stats? | WM action encoder trained with JEPA-WM stats. | `jepa-wms` returns `(model, preprocessor)`; `Preprocessor.normalize_actions=(a-mean)/std`. |
| Why was WM trained with preprocessor? | Scale stability. | Dataset code normalizes actions/proprios before training; inputs near mean 0/std 1. |
| Does WM normalization hide OOD actions? | No. | OOD action becomes large z-score. Difference preserved. |
| What if SmolVLA action dist differs from WM train dist? | Real OOD risk. | Normalize still correct; log WM z-scores and treat large values as risk. |
| Why not use MetaWorld `[-1,1]` as universal scale? | No universal scale. | SmolVLA MT50 action stats exceed `[-1,1]`; JEPA-WM has separate stats. |
| Does MetaWorld clip everything? | Not exactly. | Action space is `Box(-1,1,(4,))`; source clips XYZ; gripper path less clean. Local final clip = safety/repro. |
| Is `tanh` official LeRobot? | No. | Official eval path is `preprocessor -> policy.select_action -> postprocessor -> env.step`; no tanh in SmolVLA model. |
| Why did LIBERO use tanh? | Differentiable bounded stochastic action. | LIBERO wrapper tanh/clamp handles `[-1,1]` action requirement. Good there, not automatically here. |
| Is current GRPO tanh suspect? | Yes. | Current wrapper tanh's SmolVLA z-score before postprocessor. Distorts normalized policy coord. |
| Can GRPO run no-tanh? | Yes. | Gaussian logprob can be in normalized action space; postprocess sampled action for env; final clip for execution. |
| If final clip non-diff, GRPO broken? | No, but mismatch risk. | Policy-gradient logprob is for sampled action; clip can make env action differ at bounds. Track clip rate. |
| Should WM score pre-clip or post-clip action? | Post-clip/executed action. | WM predicts next state from action env actually receives. |
| Should WM use 4D or 20D? | Both: 4D env actions packed to 20D. | JEPA-WM action input represents `5 x 4D` env steps. |
| Why pack 5 actions? | WM temporal stride. | JEPA-WM action input doc says `frameskip * action_dim`; local code packs `5*4=20`. |
| What if episode length not multiple of 5? | Pad normalized zeros. | Zero in WM z-space = mean action; local code pads zeros. |
| Are SmolVLA MT50 stats same as JEPA-WM stats? | No. | SmolVLA mean/std differ from JEPA-WM mean/std. |
| Why differ if both MetaWorld? | Different dataset/gen/execution contract likely. | SmolVLA stats likely include expert helper commands with large XYZ; JEPA-WM stats from its own MetaWorld data. |
| Is "expert actions not clipped" proven? | Strong inference, not author-confirmed. | Stats exceed env box; MetaWorld helper can emit OOB; HF/GitHub issue ambiguity remains. |
| Best next experiment? | 3-way ablation. | A: no tanh + executed-action WM norm. B: current tanh. C: raw-to-WM no-norm sanity fail/check. |

## Why Normalize / Clip?

| Link | Normalize? | Clip? | Why |
|---|---:|---:|---|
| MT50 data -> SmolVLA train | Y | N | SmolVLA trained with `ACTION=MEAN_STD`; target is z-score action, not bounded action. |
| SmolVLA internal output | Y | N | Output lives in normalized z-score space. `0` = dataset mean, `1` = +1 std. |
| SmolVLA -> MetaWorld eval | Y, inverse | Y, at execution | LeRobot postprocessor unnormalizes to raw/env-scale action. MetaWorld/action wrapper bounds final action. |
| Extra `tanh` before MetaWorld | N | N/A | Not official LeRobot path. It distorts z-score space before unnormalization. |
| Extra clip before MetaWorld | Maybe | Y if wrapper owns env call | Redundant with MetaWorld for XYZ, but useful reproducibility/safety. Must document if local. |
| MetaWorld internal action bounds | N/A | Y | Env action box is `[-1,1]`; source clips XYZ internally. Gripper less clear, so final wrapper clip is safer. |
| SmolVLA -> JEPA-WM | Y, JEPA stats | Use executed action | WM trained with its own action stats, not SmolVLA stats. Feed env-scale action, then JEPA-normalize. |
| JEPA-WM 4D -> 20D | N/A | N/A | WM expects packed action: `5 x 4D = 20D`, matching frameskip/stride 5. |
| LIBERO GRPO tanh | Y there | tanh bounds | LIBERO wrapper uses differentiable bounded stochastic action. Good for LIBERO, not automatically right for MetaWorld SmolVLA. |
| Current GRPO wrapper | Y but suspicious | final clip | It does `tanh` in SmolVLA normalized space before postprocessor. Main questionable bit. |

Bottom line:

- Normalize = match model training contract.
- Unnormalize = convert model output back to env action units.
- Clip = satisfy env action bounds / avoid illegal execution.
- Tanh = optional differentiable bounding trick; bad if applied in wrong space.
- MetaWorld best path: no tanh in SmolVLA z-score space; postprocess; execute clipped env-scale action; JEPA-normalize only when scoring WM.

## Critical "Why WM Normalize?" Answer

Challenge:

> Your policy outputs something, then you normalize with WM stats. Why? Isn't that wrong?

Answer:

We are **not** taking raw SmolVLA policy output and normalizing with WM stats. That would be wrong.

Correct chain:

| Step | Action space | Transform | Why |
|---|---|---|---|
| SmolVLA output | SmolVLA z-score | none | Policy predicts normalized action because trained with `ACTION=MEAN_STD`. |
| SmolVLA -> env action | raw MetaWorld action | inverse SmolVLA stats | LeRobot postprocessor maps policy output back to env-scale action. |
| Env execution | bounded MetaWorld action | clip/action bounds | MetaWorld action space is `[-1,1]`; execution must obey env contract. |
| Env action -> JEPA-WM | JEPA z-score | normalize with WM stats | WM was trained to consume normalized env actions using its own stats. |
| WM input | packed 20D | pack `5 x 4D` | WM action input represents 5 env steps. |

Supervisor wording:

> JEPA-WM normalization is part of the WM input contract, not extra rescaling of SmolVLA output. We first recover actual env-scale action, then express that action in coordinate system used to train WM.

Why raw env action into WM is risky:

```text
WM trained seeing normalized actions ~ mean 0, std 1.
Raw env action shifts scale/offset.
Action encoder weights interpret wrong numeric coords.
Prediction error may be unit mismatch, not model weakness.
```

OOD nuance:

```text
normal env action -> z-score near 0
unusual env action -> z-score large, e.g. 3 or -4
```

Normalization preserves OOD signal. It does not erase it.

Analogy:

> Image model trained on normalized RGB still needs same RGB normalization on new images. New domain may be OOD, but preprocessing contract remains.

## Why JEPA-WM Has Preprocessor

JEPA-WM was not trained on "API values" directly. It trained on **preprocessed tensors**:

- `metaworld_hf_dset.py`: when `normalize_action=True`, computes action/proprio/state stats, then stores `actions = (actions - action_mean) / action_std`.
- `preprocessor.py`: inference preprocessor exposes same transform: `normalize_actions(actions) = (actions - action_mean) / action_std`.
- `hubconf.py`: pretrained load returns `(model, preprocessor)` with hardcoded stats, so inference can match training without dataset access.

Reason:

- Optimizer stable: action dims, proprio dims, image features have different units/ranges.
- Action encoder weights learned normalized coordinates.
- Raw env numbers at inference break learned scale/offset.

Caveat:

If checkpoint was trained with `normalize_action=False`, raw would be correct. Current evidence says this MetaWorld JEPA-WM checkpoint has nontrivial `action_mean/std`, so normalize is correct.

## Stats Snapshot

SmolVLA / LeRobot MT50 action stats from `lerobot/metaworld_mt50`:

```text
mean = [0.25, 0.51, -0.13, 0.38]
std  = [2.03, 2.01, 3.39, 0.71]
min  = [-11.47, -14.93, -17.28, -1.0]
max  = [19.46, 12.47, 24.63, 1.0]
```

JEPA-WM MetaWorld action stats:

```text
mean = [0.0057235775, 0.1573565155, -0.1396457851, 0.1998193860]
std  = [0.7359239459, 0.7340804338, 0.7182977200, 0.7432057261]
```

Interpretation:

- SmolVLA stats: VLA training/postprocessing contract.
- JEPA-WM stats: WM action encoder/scoring contract.
- Different stats are expected because learned modules were trained with different preprocessing/data generation.

## Tanh Decision

Current `MetaWorldSmolVLAGRPOPolicy`:

```text
mean, log_std = policy.select_action_distr_params(...)
unsquashed = mean + std * noise
squished = tanh(unsquashed)
out = LeRobot postprocessor(squished)
exec_np = clip(out, -1, 1)
```

Problem:

- `mean` lives in SmolVLA normalized z-score space.
- `tanh` forces z-score into `(-1,1)` before unnormalization.
- This is not official LeRobot path.
- This may shrink/distort valid SmolVLA action distribution.

LIBERO inspiration differs:

```text
mean = unnormalize_action(mean)
unsquished_action ~ Normal(mean, std)
action = tanh(unsquished_action)
```

There tanh is applied after converting to raw action scale. Also LIBERO required bounded action. Good for LIBERO, not direct proof for MetaWorld.

Recommendation:

- Treat current tanh wrapper as ablation/design choice.
- Main MetaWorld GRPO path should test no-tanh in SmolVLA z-score space:

```text
sample normalized action
compute Gaussian logprob in normalized space
LeRobot postprocessor -> env-scale action
final execution clip
WM scoring uses executed action -> JEPA normalization -> pack
```

Risk:

- Final clip creates action/logprob mismatch if many samples hit bounds.
- Mitigation: log clip rate; lower `log_std`; consider tanh in env-space only if clip rate high.

## Research / Hypothesis Trail From Chat

This section records the hypotheses we considered, including the web-search angle, so a future LLM can reconstruct why recommendation changed.

### Initial Hypothesis: Tanh Needed Because Env Action Box Is `[-1,1]`

Status: partly true, but not sufficient.

- MetaWorld public action space is `Box(-1,1,(4,))`, so final env execution should be bounded.
- LIBERO also requires `[-1,1]`, and `safe-robot-steering` uses clamp/tanh for that reason.
- But the key question is **where** bounding is applied.

Finding:

```text
LIBERO:
SmolVLA normalized -> unnormalize to raw LIBERO action -> clamp/tanh raw action -> env

Our current MetaWorld GRPO:
SmolVLA normalized -> tanh normalized z-score -> LeRobot postprocessor -> raw action -> final clip -> env
```

Conclusion:

Current MetaWorld tanh is **not** "tanh raw action to satisfy MetaWorld". It is "tanh z-score, then unnormalize, then clip". That is different from LIBERO and different from official LeRobot eval.

### Hypothesis: Postprocessor Makes Tanh Equivalent To Official LeRobot

Status: false.

Evidence:

- Official/local eval path:

```text
proc = preprocessor(batch)
action = policy.select_action(proc)
action = postprocessor(action)
exec_action = _coerce_exec_action(action)
```

- Patched SmolVLA `sample_actions` returns denoised flow output `x_t`; no tanh on action chunk.
- Current GRPO wrapper explicitly adds `torch.tanh(unsquashed)` before postprocessor.

Conclusion:

LeRobot postprocessor is inverse normalization, not squashing. It maps normalized action to raw/env scale.

### Hypothesis: "Normalized" Means Bounded

Status: false.

Clarification:

```text
normalized action = z-score coordinate
0 = dataset mean
1 = one dataset std above mean
-1 = one dataset std below mean
```

Model can output `-2`, `0.5`, `3`, etc. in normalized space. This is not same as env `[-1,1]`.

Example from SmolVLA stats:

```text
gripper mean/std = 0.38 / 0.71
norm 0 -> raw 0.38
norm 1 -> raw 1.09
norm -1 -> raw -0.33
```

So even normalized `+1` can become raw env action above `1`.

### Hypothesis: Dataset Actions Are Already Env-Box Actions

Status: likely false for SmolVLA MT50, but not fully author-confirmed.

Evidence:

- `lerobot/metaworld_mt50` stats:

```text
min = [-11.47, -14.93, -17.28, -1.0]
max = [19.46, 12.47, 24.63, 1.0]
```

- XYZ far exceed `[-1,1]`; gripper stays in `[-1,1]`.
- MetaWorld expert helper can emit OOB XYZ and warns env clips response.
- Public issue/discussion ambiguity exists around LeRobot MetaWorld action scale.

Conclusion:

Say "evidence indicates dataset stores raw expert XYZ commands before MetaWorld internal XYZ clipping", not "proven author-confirmed".

### Exa / Web Search Performed

We used advanced Exa search in chat for these angles:

1. MetaWorld action space / clipping:
   - Query: `Meta-World action space clipping action values Box(-1,1) sawyer_xyz_env set_xyz_action clips actions`
   - Used to verify public action box and source-level clipping relevance.

2. Squashed Gaussian / SAC-style tanh:
   - Query: `SAC tanh squashed Gaussian policy log probability correction continuous control bounded action spaces why tanh instead of clipping`
   - Findings: tanh-squashed Gaussian is standard for bounded continuous-control policies because it gives bounded samples plus change-of-variables log-prob correction.
   - Source examples recorded in chat: `https://github.com/haarnoja/sac/blob/master/sac/policies/gaussian_policy.py`, `https://github.com/denisyarats/pytorch_sac_ae/blob/master/sac_ae.py`.

3. LeRobot MetaWorld dataset / stats / issue:
   - Query: `LeRobot MetaWorld mt50 dataset action stats raw expert action trajectories [-11.47 19.46] postprocessor MEAN_STD evaluation SmolVLA metaworld`
   - Used to root the ambiguity around MT50 action stats, postprocessor, and issue tracker.
   - Sources recorded in chat:
     - `https://huggingface.co/datasets/lerobot/metaworld_mt50`
     - `https://huggingface.co/docs/lerobot/v0.5.1/metaworld`
     - `https://github.com/huggingface/lerobot/issues/2617`
     - `https://metaworld.farama.org/benchmark/action_space/`

### Recommendation After Research

Main MetaWorld GRPO should be:

```text
a_norm ~ Normal(mean_norm, std_norm)
logprob = Gaussian logprob(a_norm)
a_raw = LeRobotPostprocessor(a_norm)
a_exec = clip/action-bound(a_raw, -1, 1)
env.step(a_exec)
WM score uses a_exec -> JEPA-WM normalize -> pack
```

Why:

- Most faithful to SmolVLA/LeRobot action contract.
- Avoids tanh in wrong coordinate system.
- Still respects MetaWorld execution bounds.
- Easier to explain/defend.
- With small `log_std=-2`, exploration should not explode.

Keep current tanh path as:

```text
LIBERO-inspired normalized-space tanh ablation
```

Not as official/protocol-faithful MetaWorld path.

### Still Worth Double-Checking

Before major claims/code changes:

- Run same-seed 1-update ablation:
  - `tanh_norm_then_postprocess`
  - `no_tanh`
  - `no_tanh_clip_gripper`
- Log three ranges:
  - normalized sampled action,
  - postprocessed raw action,
  - final executed action.
- Confirm exact installed `lerobot-eval` action clipping path, separate from local evaluator.
- Log WM-action z-score range to quantify OOD vs JEPA-WM stats.

## LaTeX Table Added To Report

Inserted in `project/artifacts/curReporttext.tex` after `Bottom-line normalization note`.

```latex
\begin{table}[t]
\centering
\small
\begin{tabular}{p{4.2cm} p{2.0cm} p{1.6cm} p{4.8cm}}
  \toprule
  \textbf{Stage} & \textbf{Normalize?} & \textbf{Clip?} & \textbf{Why}
  \\
  \midrule
  MT50 training target (dataset) & Y & N & SmolVLA trained on \texttt{MEAN\_STD} action stats from \texttt{meta/stats.json}. \\
  SmolVLA policy output (inference) & N (already z-scored) & N & Output is model coordinate in normalized action space. \\
  LeRobot postprocessor for env step & Y (inverse) & N/optional local & Converts normalized output to env units: \(a_{\mathrm{env}}=a_{\mathrm{norm}}\sigma+\mu\). \\
  Meta-World execution (\texttt{env.step}) & N & Y & Uses env action-box constraints (and internal XYZ clipping path). \\
  Env action \(\rightarrow\) JEPA-WM input & Y (JEPA stats) & N & JEPA-WM is trained on its own normalized action space before packing. \\
  Packing to WM action vector & N & N & \(5\times4\)-D steps become one \(20\)-D WM step. \\
  Current GRPO wrapper branch (\texttt{MetaWorldSmolVLAGRPOPolicy}) & Y (custom) & Y & Extra tanh/clip debug-local only; not required by LeRobot official path. \\
  \bottomrule
\end{tabular}
\caption{Normalization and clipping controls across training, evaluation, and scoring paths.}
\label{tab:normalization-clipping-audit}
\end{table}
```

Possible improvement before supervisor:

```text
Env action -> JEPA-WM input:
Use env action? Y
Normalize? Y, with JEPA stats
Why? Same physical action, converted into WM training coordinates.
```

## Source Map

### Report / audit text

- `project/artifacts/curReporttext.tex`
  - action protocol audit, bottom-line normalization note, table.
  - Key claims: MT50 stats, LeRobot postprocessor contract, MetaWorld action contract, Phase074 action debug, local additions vs official path.

External report URLs embedded there:

- SmolVLA checkpoint train config: `https://huggingface.co/jadechoghari/smolvla_metaworld/raw/main/train_config.json`
- SmolVLA checkpoint config: `https://huggingface.co/jadechoghari/smolvla_metaworld/raw/main/config.json`
- MT50 dataset stats: `https://huggingface.co/datasets/lerobot/metaworld_mt50/raw/main/meta/stats.json`
- MetaWorld action space: `https://metaworld.farama.org/benchmark/action_space/`
- MetaWorld Sawyer env source: `https://github.com/Farama-Foundation/Metaworld/blob/master/metaworld/sawyer_xyz_env.py`
- MetaWorld policy helper source: `https://github.com/Farama-Foundation/Metaworld/blob/master/metaworld/policies/policy.py`
- LeRobot MetaWorld env source: `https://github.com/huggingface/lerobot/blob/main/src/lerobot/envs/metaworld.py`
- LeRobot normalizer source: `https://github.com/huggingface/lerobot/blob/main/src/lerobot/processor/normalize_processor.py`
- Dataset discussion: `https://huggingface.co/datasets/lerobot/metaworld_mt50/discussions/5`
- LeRobot issue: `https://github.com/huggingface/lerobot/issues/2617`

### Official LeRobot local eval path

- `project/src/smolvla_pipeline/evaluator.py`
  - `_select_action`: `preprocessor -> policy.select_action -> postprocessor -> _coerce_exec_action`.
  - `_coerce_exec_action`: checks action dim, then `clip(action_np, -1.0, 1.0)`.

Relevant local flow:

```text
proc = bundle.preprocessor(batch)
action = bundle.policy.select_action(proc)
action = bundle.postprocessor(action)
exec_action = _coerce_exec_action(action, ...)
```

### Patched SmolVLA model

- `.envs/lerobot_mw_py310/lib/python3.12/site-packages/lerobot/policies/smolvla/modeling_smolvla.py`
  - `sample_actions`: denoising flow returns `x_t, log_std`; no tanh on returned action.
  - `select_action_distr_params`: local patch exposing mean/log_std for GRPO.

### Current MetaWorld GRPO wrapper

- `project/src/smolvla_grpo/policy_wrapper.py`
  - `sample_action_from_proc`: samples `unsquashed`, applies `torch.tanh`, postprocesses, then clips.
  - `get_action_probs_from_proc_list`: recomputes tanh-corrected logprob from stored unsquashed actions.

Key issue:

```text
tanh is applied before LeRobot postprocessor, i.e. in SmolVLA normalized z-score space.
```

### LIBERO inspiration

- `VGG JEPA/safe-robot-steering/model/smolvla_policy.py`
  - `get_action`: unnormalizes policy action then clamps for LIBERO.
  - `sample_action`: unnormalizes mean, samples Gaussian, applies tanh.

Key distinction:

```text
LIBERO tanh after unnormalizing mean -> raw action scale.
Our MetaWorld tanh before postprocessor -> normalized z-score scale.
```

### JEPA-WM preprocessor and stats

- `VGG JEPA/jepa-wms/app/plan_common/datasets/preprocessor.py`
  - `normalize_actions(actions) = (actions - action_mean) / action_std`.
  - `denormalize_actions(actions) = actions * action_std + action_mean`.

- `VGG JEPA/jepa-wms/app/plan_common/datasets/metaworld_hf_dset.py`
  - computes mean/std when `normalize_action=True`.
  - stores `self.actions = (self.actions - self.action_mean) / self.action_std`.

- `VGG JEPA/jepa-wms/app/plan_common/datasets/__init__.py`
  - hardcoded MetaWorld `DATA_STATS`, including `action_mean/action_std`.

- `VGG JEPA/jepa-wms/hubconf.py`
  - builds `Preprocessor` from hardcoded stats.
  - passes `preprocessor` into model init.
  - returns `(model, preprocessor)`.

### Local WM scoring bridge

- `project/src/segment_grpo_loop.py`
  - `_normalize_env_actions_for_wm`: env 4D action -> JEPA-WM normalized 4D action.
  - `_pack_env_actions_for_wm`: packed `factor * env_dim = wm_dim`; pads zeros in normalized space.
  - `score_chunk_by_goal_latent`: normalizes then packs before model unroll.

Core local code flow:

```text
chunk_env = ensure 4D env action matrix
actions_norm_np = _normalize_env_actions_for_wm(wm_bundle.preprocessor, chunk_env, ...)
packed_np = _pack_env_actions_for_wm(actions_norm_np, factor, env_dim, wm_dim)
actions_t = torch.from_numpy(packed_np).unsqueeze(1)
wm_bundle.model.unroll(..., act_suffix=actions_t)
```

### JEPA-WM action shape

- `VGG JEPA/jepa-wms/app/vjepa_wm/video_wm.py`
  - `encode_act` doc: input action shape uses `frameskip * action_dim`.
  - local MetaWorld: `frameskip=5`, env action dim `4`, WM action dim `20`.

## Confidence / Caveats

High confidence:

- SmolVLA official eval path uses postprocessor; no tanh.
- Current GRPO wrapper applies tanh in normalized space.
- JEPA-WM preprocessor normalizes actions with its stats.
- Local WM scoring code normalizes env actions then packs to WM dim.

Medium confidence:

- SmolVLA MT50 dataset stores raw expert helper commands before env internal clipping. Evidence: stats exceed `[-1,1]`, MetaWorld helper can emit OOB, public discussion unresolved.

Open risk:

- SmolVLA env actions may be OOD under JEPA-WM stats. Normalization is still correct contract, but WM scores may be unreliable. Need logs:
  - raw env action range,
  - executed/clipped action range,
  - WM z-score range,
  - clip rate,
  - WM score sensitivity to no-normalize sanity ablation.

## Proposed Next Ablations

1. **No-tanh MetaWorld GRPO**  
   Sample in SmolVLA z-score space, Gaussian logprob there, LeRobot postprocessor, final exec clip, WM normalize executed action.

2. **Current tanh wrapper**  
   Keep as comparison, label "LIBERO-inspired normalized-space tanh ablation".

3. **WM no-normalize sanity check**  
   Feed raw env actions to WM once. Expected worse / unstable if training contract matters. Useful as supervisor-proof negative control.

4. **Clip-rate/OOD logging**  
   For each rollout: count action dims clipped, record max abs WM z-score, compare reward/success/WM score.

## Supervisor-Safe Paragraph

> We do not feed SmolVLA's internal normalized action directly into JEPA-WM. SmolVLA first predicts in its own z-score action space, then LeRobot's checkpoint postprocessor maps that to env-scale MetaWorld action. The action actually executed in MetaWorld is the semantic bridge. For JEPA-WM scoring, we take that executed env action and apply JEPA-WM's own preprocessor, because the WM action encoder was trained on normalized actions using those stats. This is coordinate conversion into the WM training space, not double-normalizing or changing which action was taken.
