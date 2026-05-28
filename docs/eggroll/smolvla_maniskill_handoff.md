# SmolVLA x ManiSkill Handoff

Status: `2026-05-26 13:55 UTC+1`

Mode: caveman ultra. Substance dense. No fluff.

## TL;DR

Goal: generate RL4VLA/ManiSkill `PutOnPlateInScene25Main-v3` demos, convert to LeRobot, SFT SmolVLA, eval in ManiSkill. Target: any nonzero success.

Current status: data probe passed on RTX6000. Full data generation job `2852555.pbs-7` is running on `cx3-8-12` with `select=1:ncpus=16:mem=64gb:ngpus=1:gpu_type=RTX6000`.

Fixed root causes:

- `toppra._CythonUtils` wheel contained AVX512/EVEX instructions built on Intel login CPU. RTX6000 AMD EPYC nodes crashed with `Illegal instruction`, `Exit_status=132`.
  - Fix: rebuild `toppra==0.6.3` from source with portable flags:

```bash
CFLAGS="-O2 -fPIC -march=x86-64 -mtune=generic"
CXXFLAGS="-O2 -fPIC -march=x86-64 -mtune=generic"
```

- `mplib==0.1.1` segfaulted in `ArticulatedModel` with `numpy 2.4.6`.
  - Evidence: direct `ArticulatedModel` constructor crashed for all link/joint/SRDF variants; web RCA matches ManiSkill issues for MPLib + NumPy 2.x.
  - Fix: pin data env NumPy to `<2.0.0` (`1.26.4`) and rebuild TOPPRA after the NumPy pin.

Probe result: `01_data_probe.pbs` generated 4/4 demos, 123 frames, `action.shape == (*, 7)`, `state.shape == (*, 7)`.

## Original Ask

User wanted autonomous overnight pipeline:

- Generate `16,384` ManiSkill/RL4VLA motion-planning demos for `PutOnPlateInScene25Main-v3`.
- Include required 7D state, not only action/image.
- Convert demos into LeRobot dataset for SmolVLA.
- Fine-tune base `lerobot/smolvla_base` on 7D action/state task.
- Run baseline/eval in ManiSkill.
- Aim: "semi-success", meaning `success_rate > 0`.
- Use RTX6000 GPUs.
- Use ephemeral for big data/checkpoints.
- Do RCA, fix, continue autonomously.
- Commit small source changes often; never commit generated datasets/checkpoints.
- Run on `main`, no branch/subtree.

## Key Repo/Libs

- `RL4VLA`: data gen route. Contains RL4VLA fork of ManiSkill + motion planner scripts.
- `ManiSkill3`: sim env/task stack. Env: `PutOnPlateInScene25Main-v3`.
- `MPLib`: motion planner library imported by `collect_simpler`.
- `toppra`: trajectory time-param lib, imported by `mplib`. Current failing native extension.
- `LeRobot`: dataset + SmolVLA training stack.
- `SmolVLA`: base policy `lerobot/smolvla_base`.
- `PBS`: scheduler, queues `v1_gpu72` for RTX6000 GPU stages, `v1_small24` for current CPU rebuild.

## Important Paths

Project root:

```text
/rds/general/user/aa6622/home/project
```

Big artifact root:

```text
/rds/general/user/aa6622/ephemeral/eggroll/smolvla_maniskill
```

Pipeline scripts:

```text
scripts/maniskill_smolvla/
```

Main plan:

```text
/rds/general/user/aa6622/home/.cursor/plans/overnight-smolvla-maniskill_ad6f1d25.plan.md
```

Current doc:

```text
docs/eggroll/smolvla_maniskill_handoff.md
```

Data-gen Python:

```text
project/.venvs/pirl-rlinf-py312/bin/python
```

Train Python intended:

```text
/rds/general/user/aa6622/home/.envs/lerobot_mw_py312/bin/python
```

Ephemeral layout:

```text
raw_npz/             RL4VLA raw demos
lerobot_dataset/     converted local LeRobot dataset
checkpoints/         SmolVLA SFT outputs
videos/              eval videos
hf_home/             HF model cache
hf_lerobot_home/     LeRobot cache
torch_home/          torch cache
```

Project mirror/log layout:

```text
project/artifacts/smolvla_maniskill/
project/artifacts/smolvla_maniskill/runs/<run_id>/
```

## Planned Pipeline

Stage order in `autopilot.sh`:

```text
00_build_envs.pbs
01_data_probe.pbs
02_data_full.pbs
03_convert_full.pbs
04_sft_smoke.pbs
05_sft_train.pbs
06_benchmark.pbs
```

Intent:

1. Build/env smoke.
2. Small data probe, `4` demos.
3. Full data gen, ~`16,384` demos.
4. Convert `.npz` -> local LeRobot.
5. SmolVLA SFT smoke.
6. Full SFT.
7. ManiSkill eval.

Gating rule: smoke before scale. No full data/SFT until probe proves:

- RL4VLA planner imports.
- env assets load.
- raw `.npz` has `image`, `instruction`, `state`, `action`, `info`.
- `state.shape == (*, 7)`, `action.shape == (*, 7)`.
- converter accepts raw files.
- SmolVLA uses pretrained 32D padding dims.

## Implemented Files

### Shared Runtime

`scripts/maniskill_smolvla/common.sh`

Does:

- loads modules: `tools/prod`, `Python/3.12.3-GCCcore-13.3.0`, `Mesa/24.1.3-GCCcore-13.3.0`
- defines paths
- keeps big stuff in ephemeral
- sets `PYTHONPATH` for `RL4VLA` and `RL4VLA/ManiSkill`
- sets render/cache env:
  - `MUJOCO_GL=osmesa`
  - `PYOPENGL_PLATFORM=osmesa`
  - `LIBGL_ALWAYS_SOFTWARE=1`
  - `MS_SKIP_ASSET_DOWNLOAD_PROMPT=1`
- uses short temp dirs:
  - `RAY_TMPDIR=/tmp/msm_${USER}_${MSM_RUN_ID}/ray_tmp`
  - `TMPDIR=/tmp/msm_${USER}_${MSM_RUN_ID}/tmp`

### Env Build

`scripts/maniskill_smolvla/build_envs.sh`

Does:

- verifies data-gen Python.
- now attempts portable `toppra` rebuild before importing data stack:
  - `--force-reinstall --no-deps --no-cache-dir --no-binary=:all: toppra==0.6.3`
  - flags: `-march=x86-64 -mtune=generic`
- imports data deps:
  - `gymnasium`
  - `mani_skill`
  - `mplib`
  - `toppra.constraint.linear_joint_velocity`
  - `tyro`
  - `cv2`
  - `torch`
  - `sapien`
- verifies train env imports:
  - `torch`
  - `transformers`
  - `huggingface_hub`
  - `lerobot.datasets.lerobot_dataset`
  - `lerobot.policies.smolvla.configuration_smolvla`
  - `lerobot.scripts.lerobot_train`
- warms HF snapshots:
  - `lerobot/smolvla_base`
  - `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`

Concern: `--no-binary=:all:` also source-builds build deps. That can be slow. Better next version likely:

```bash
pip install Cython numpy wheel setuptools
pip install --no-build-isolation --no-deps --no-cache-dir --no-binary=toppra toppra==0.6.3
```

Use after current job result if current build too slow/fails.

### Autopilot

`scripts/maniskill_smolvla/autopilot.sh`

Does:

- submits stage list in order.
- monitors PBS job state.
- parses historical exit via `qstat -xf`/`qstat -Hf`.
- writes RCA markdown if exit status nonzero.
- supports resume via `MSM_START_STAGE_INDEX`.
- supports node pinning via `MSM_PBS_HOST`.

Important fix already patched:

```bash
vnode=${host}:gpu_type=RTX6000
```

Why: earlier host-pinned override forgot `gpu_type=RTX6000`. Queue default gpu type is `L40S`, so `vnode=cx3-8-12` with no `gpu_type` became impossible on RTX6000 vnode.

### Data Probe

`scripts/maniskill_smolvla/01_data_probe.pbs`

PBS:

```text
queue: v1_gpu72
select=1:ncpus=16:mem=128gb:ngpus=1:gpu_type=RTX6000
walltime=02:00:00
```

Command:

```bash
python -m mani_skill.examples.motionplanning.widowx.collect_simpler \
  -e PutOnPlateInScene25Main-v3 \
  --save_data \
  --num_procs 1 \
  --num_traj 4 \
  --record_dir ${MSM_PROBE_RECORD_DIR} \
  --seed 100 \
  --sim_backend cpu \
  --obs_mode rgb+segmentation \
  --render_mode rgb_array \
  --shader default
```

Then validates with `inspect_npz.py`.

### Full Data

`scripts/maniskill_smolvla/02_data_full.pbs`

Intent: generate full set into ephemeral `raw_npz/`.

Likely uses same `collect_simpler` path at larger `num_traj`.

Not reached yet.

### NPZ Inspection

`scripts/maniskill_smolvla/inspect_npz.py`

Intent:

- load `.npz`
- validate raw keys:
  - `image`
  - `instruction`
  - `state`
  - `action`
  - `info`
- verify 7D action/state.
- verify image HWC RGB.
- write manifest.

### Converter

`scripts/maniskill_smolvla/convert_npz_to_lerobot.py`

Input format:

```python
np.load(path, allow_pickle=True)["arr_0"].tolist()
```

Feature schema:

```text
observation.images.front: image, shape (480, 640, 3)
observation.state: float32, shape (7,)
action: float32, shape (7,)
task: instruction string
```

State names:

```text
eef_x, eef_y, eef_z, eef_roll, eef_pitch, eef_yaw, gripper
```

Action names:

```text
dx, dy, dz, droll, dpitch, dyaw, gripper
```

Filters tiny pose actions but preserves gripper toggles. Stops episode after sustained success count.

### SFT Scripts

`04_sft_smoke.pbs`

Intent:

- short `lerobot-train`.
- base: `lerobot/smolvla_base`.
- keep:
  - `--policy.max_state_dim=32`
  - `--policy.max_action_dim=32`

Why: pretrained SmolVLA base uses 32D padding/projection. Dataset has 7D vectors, policy pads internally. Do not shrink config to 7; checkpoint load likely breaks.

`05_sft_train.pbs`

Intent:

- larger supervised fine-tune.
- save checkpoints into ephemeral.
- latest symlink/manifest.

Not reached yet.

### Eval

`scripts/maniskill_smolvla/eval_maniskill_smolvla.py`

Does:

- resolves trained SmolVLA checkpoint.
- loads local LeRobot dataset for stats.
- loads `SmolVLAPolicy.from_pretrained`.
- builds ManiSkill env:
  - env `PutOnPlateInScene25Main-v3`
  - obs `rgb+segmentation`
  - control `arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos`
  - backend `cpu`
  - control freq `5`
- extracts current image and exact same 7D state contract.
- calls `policy.select_action`.
- clips/normalizes 7D action.
- reports `success_rate`.

Risk: eval env imports both LeRobot/SmolVLA and ManiSkill. Could hit dependency conflict after data issue solved.

### Static Tests

`tests/test_maniskill_smolvla_static.py`

Covers:

- PBS scripts use PBS, not Slurm.
- GPU stages use `v1_gpu72`.
- artifact roots in ephemeral.
- collector patch records `state`.
- converter uses LeRobot 7D contract.
- SFT keeps 32/32 padding dims.
- benchmark uses actual ManiSkill rollout.
- autopilot host pin retains `gpu_type=RTX6000`.
- env rebuilds `toppra` portable.
- `.gitignore` covers big artifacts.

## Collector Patch

File:

```text
RL4VLA/ManiSkill/mani_skill/examples/data_collector/vla_data_collector.py
```

Change:

- added `"state": []`
- imported:
  - `matrix_to_euler_angles`
  - `quaternion_to_matrix`
- added `update_state(action)`
- call `update_state(action)` before saving action.

State contract:

```python
state = [
    ee_pose.p[0],
    euler_xyz(ee_pose.q[0]),
    action[-1],
]
```

Shape: `(7,)`

Why action gripper used as state gripper: easiest consistent pre-step gripper scalar available in collector path. Risk: true robot gripper joint position may differ. But train/eval both use same state contract, so SFT/eval schema consistent.

## Problems Hit + RCA

### 1. No Public Exact Dataset

Initial assumption: maybe HF dataset exists for `16 objects x 17 receptacles x 16 scenes = 4352 combos`, `16,384` MPLib eps.

Research result:

- no exact standalone HF dataset found.
- `RLinf/RLinf-Pi05-ManiSkill-25Main-SFT` is model checkpoint, not demos.
- `haosulab/ManiSkill` has general assets/demos, not exact 25Main SFT grid.
- RL4VLA docs indicate dataset generated via motion-planning script, not downloaded.

Decision: clone/use RL4VLA and generate demos.

### 2. Raw RL4VLA Data Missing State

Problem:

- `VLADataCollector` saved only:
  - `image`
  - `instruction`
  - `action`
  - `info`
- SmolVLA SFT needs `observation.state`.

Fix:

- patched collector to store 7D state.
- converter expects `state`.

Risk:

- need probe to verify `.npz` actually includes state after generated with patched collector.

### 3. SmolVLA Action/State Dim Confusion

Problem:

- temptation: set `max_action_dim=7`, `max_state_dim=7`.
- likely wrong for pretrained base.

RCA:

- SmolVLA base trained with padded max dims (32/32).
- dataset feature can be 7D; model config max dims should remain 32.

Fix:

- SFT scripts keep:
  - `--policy.max_state_dim=32`
  - `--policy.max_action_dim=32`

### 4. Env Split Needed

Problem:

- RL4VLA/ManiSkill and SmolVLA/LeRobot deps not same clean env.
- Existing PIRL env was for RLinf/OpenPI/Pi0.5, with forced Python 3.12 and special pins.

Plan:

- reuse existing `pirl-rlinf-py312` for data generation because it already ran ManiSkill3/Pi0.5 path.
- use separate SmolVLA/LeRobot env for train.
- keep eval flexible; maybe needs third env or two-process bridge.

### 5. pytest/plugin Weirdness

Problem:

- tests hit Python path/plugin issues:
  - `pytest` not found
  - `libpython3.12.so.1.0` missing if module not loaded
  - pytest plugin import failures (`zarr`, `donfig`)

Fix:

```bash
module load Python/3.12.3-GCCcore-13.3.0
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 project/.venvs/pirl-rlinf-py312/bin/python -m pytest tests/test_maniskill_smolvla_static.py
```

### 6. Autopilot Exit Status Parse Bug

Problem:

- autopilot first reported `unknown` exit status.
- `qstat -f` on finished job insufficient; historical records need `qstat -xf` or `qstat -Hf`.
- `Exit_status` field capitalized; awk parse expected lowercase.

Fix:

```bash
qstat -xf "$job" || qstat -Hf "$job"
awk -F'= ' 'tolower($1) ~ /exit_status[[:space:]]*$/ {print $2; exit}'
```

### 7. `Illegal instruction` On Data Probe

First failed probe:

```text
job: 2847270.pbs-7
node: cx3-11-8
exit: 132
error: Illegal instruction (core dumped)
```

Initial hypothesis:

- bad vnode `cx3-11-8`.

Later proven wrong.

### 8. Host-Pinned Retry "Can Never Run"

Problem:

- pinned retry to `cx3-8-12` sat queued with:

```text
Can Never Run: can't fit in the largest placement set, and can't span psets
```

Bad select override:

```text
select=1:ncpus=4:mem=32gb:ngpus=1:vnode=cx3-8-12
```

RCA:

- queue default `gpu_type = L40S`.
- `cx3-8-12` is RTX6000.
- overriding `select` dropped `gpu_type=RTX6000`.
- scheduler tried impossible combo: RTX6000 vnode but default L40S chunk.

Fix:

```text
select=1:ncpus=4:mem=32gb:ngpus=1:vnode=cx3-8-12:gpu_type=RTX6000
```

Patched `autopilot.sh`.

### 9. Bad-Node Hypothesis Disproved

Submitted diagnostics with fixed selector to:

- `cx3-11-8`
- `cx3-11-4`
- `cx3-8-12`

All ran and all failed `Exit_status=132`.

Conclusion:

- not one bad node.
- not scheduler.
- shared runtime/native wheel issue.

### 10. Precise SIGILL Root Cause

Diagnostics used `python -X faulthandler`.

Crash stack:

```text
collect_simpler
-> motionplanner.py
-> mplib_non_convex.py
-> mplib
-> toppra
-> toppra.constraint.linear_joint_velocity
-> from toppra._CythonUtils import ...
-> Fatal Python error: Illegal instruction
```

Key facts:

- crash before env rollout.
- crash during import.
- `sapien` warnings shown before, but stack points to `toppra._CythonUtils`.
- login node import succeeded.
- compute nodes fail.

CPU evidence:

- login CPU: Intel Xeon Platinum 8358, supports AVX512.
- RTX6000 compute CPU: AMD EPYC 7742, supports AVX2, no AVX512.

Binary evidence:

```bash
objdump -d toppra/_CythonUtils*.so
```

Found EVEX/AVX512-like instr sample:

```text
vpbroadcastq %rax,%xmm0
vpbroadcastq %rax,%xmm4
```

RCA:

- `toppra._CythonUtils` wheel built/selected on AVX512-capable login CPU.
- same wheel stored in shared venv.
- compute AMD node executes unsupported instr.
- Linux raises `SIGILL`; PBS exit `132`.

Fix path:

- rebuild `toppra` native extension from source using generic x86-64 flags.
- verify on AMD compute node.

### 11. Source Build Accidentally Started On Login

Problem:

- rebuild command launched from login shell:

```bash
pip install --force-reinstall --no-deps --no-cache-dir --no-binary=:all: toppra==0.6.3
```

- got stuck in build deps.
- user asked if building on login node.

Fix:

- killed local process tree.
- submitted PBS CPU job instead:

```text
job: 2849726.pbs-7
queue: v1_small24
select=1:ncpus=4:mem=16gb
walltime=03:00:00
```

### 12. Current Build Slowness

Problem:

- current PBS build uses `--no-binary=:all:`.
- this forces source builds for build dependencies too.
- long no-output phase expected.

Better fallback:

```bash
pip install Cython numpy wheel setuptools
CFLAGS="-O2 -fPIC -march=x86-64 -mtune=generic" \
CXXFLAGS="-O2 -fPIC -march=x86-64 -mtune=generic" \
pip install --force-reinstall --no-deps --no-cache-dir --no-build-isolation --no-binary=toppra toppra==0.6.3
```

This should rebuild only `toppra`, not all build deps.

## Current PBS State

Job:

```text
2849726.pbs-7
```

Name:

```text
msm_toppra_rebuild
```

Queue/node:

```text
v1_small24 on cx3-3-22
```

Resources:

```text
4 CPU, 16GB, 0 GPU, 3h walltime
```

Last poll:

```text
job_state = R
resources_used.walltime = 00:28:44
resources_used.cput = 00:19:41
resources_used.mem = 1224436kb
```

Output path:

```text
project/artifacts/smolvla_maniskill/toppra_rebuild.out
```

As of last poll, output file absent. PBS may stage stdout only at end.

No background monitor currently active. Polling was stopped per user request.

## How To Check Current Job

```bash
qstat -f 2849726.pbs-7 || qstat -xf 2849726.pbs-7 || qstat -Hf 2849726.pbs-7
```

Read output after job finishes:

```bash
less project/artifacts/smolvla_maniskill/toppra_rebuild.out
```

Success sentinel:

```text
MSM_TOPPRA_REBUILD_DONE
```

Failure indicators:

```text
ERROR
Failed building wheel
Illegal instruction
```

## Verify Rebuild After Job

1. Check import on login:

```bash
project/.venvs/pirl-rlinf-py312/bin/python - <<'PY'
import toppra.constraint.linear_joint_velocity
print("login import ok")
PY
```

2. Check binary no obvious AVX512 sample:

```bash
so="$(project/.venvs/pirl-rlinf-py312/bin/python - <<'PY'
import importlib.util
print(importlib.util.find_spec("toppra._CythonUtils").origin)
PY
)"
objdump -d "$so" | awk '/zmm|vpbroadcastq|vmovdqu64|avx512/{print; found=1; if (++n>=20) exit} END{if(!found) print "no avx512 sample found"}'
```

3. Submit tiny RTX6000 import diag:

```bash
qsub -q v1_gpu72 -N msm_toppra_import_diag -j oe \
  -o project/artifacts/smolvla_maniskill/toppra_import_diag.out \
  -l walltime=00:10:00 \
  -l select=1:ncpus=4:mem=32gb:ngpus=1:gpu_type=RTX6000 <<'PBS'
#!/usr/bin/env bash
set -euo pipefail
source /rds/general/user/aa6622/home/project/scripts/maniskill_smolvla/common.sh
msm_setup_modules
msm_prepare_runtime
"$MSM_DATA_PYTHON" -X faulthandler - <<'PY'
import toppra.constraint.linear_joint_velocity
import mplib
import mani_skill.examples.motionplanning.widowx.collect_simpler
print("compute import ok")
PY
PBS
```

4. If import ok, rerun data probe:

```bash
MSM_START_STAGE_INDEX=1 /rds/general/user/aa6622/home/project/scripts/maniskill_smolvla/autopilot.sh
```

or direct:

```bash
qsub /rds/general/user/aa6622/home/project/scripts/maniskill_smolvla/01_data_probe.pbs
```

## If Current Rebuild Fails Or Drags

Recommended replacement PBS job:

```bash
qsub -q v1_small24 -N msm_toppra_rebuild_fast -j oe \
  -o /rds/general/user/aa6622/home/project/artifacts/smolvla_maniskill/toppra_rebuild_fast.out \
  -l walltime=01:00:00 \
  -l select=1:ncpus=4:mem=16gb <<'PBS'
#!/usr/bin/env bash
set -euo pipefail
export PROJECT_ROOT=/rds/general/user/aa6622/home/project
source "$PROJECT_ROOT/scripts/maniskill_smolvla/common.sh"
msm_setup_modules
msm_prepare_runtime
export TMPDIR="/tmp/msm_toppra_fast_${USER}_${PBS_JOBID:-manual}"
mkdir -p "$TMPDIR"
"$MSM_DATA_PYTHON" -m pip install --upgrade wheel setuptools Cython
CFLAGS="-O2 -fPIC -march=x86-64 -mtune=generic" \
CXXFLAGS="-O2 -fPIC -march=x86-64 -mtune=generic" \
  "$MSM_DATA_PYTHON" -m pip install \
    --force-reinstall \
    --no-deps \
    --no-cache-dir \
    --no-build-isolation \
    --no-binary=toppra \
    "toppra==0.6.3"
"$MSM_DATA_PYTHON" - <<'PY'
import toppra.constraint.linear_joint_velocity
print("MSM_TOPPRA_REBUILD_DONE")
PY
PBS
```

Why faster:

- allows wheels for build deps.
- only forces source for `toppra`.
- avoids compiling CMake/dependency stack unnecessarily.

## Web/Docs Findings Used

TOPPRA docs:

- Python install path exists via pip/Git.
- C++ side uses CMake/make.
- Python bindings/compiled parts require pybind/build tooling.

GitHub issue evidence:

- `toppra` source builds can fail/noisy under pip.
- Extension build compiles `_CythonUtils`.
- NumPy version/API can matter.

General native wheel fact:

- `Illegal instruction` often from binary built with instr unsupported by runtime CPU.
- Rebuilding from source with conservative `-march` fixes class of failure.

## Worktree / Git Notes

Working tree dirty with many unrelated changes before/around this work. Do not revert unrelated files.

Relevant files touched in this effort:

```text
RL4VLA/ManiSkill/mani_skill/examples/data_collector/vla_data_collector.py
scripts/maniskill_smolvla/common.sh
scripts/maniskill_smolvla/build_envs.sh
scripts/maniskill_smolvla/inspect_npz.py
scripts/maniskill_smolvla/convert_npz_to_lerobot.py
scripts/maniskill_smolvla/eval_maniskill_smolvla.py
scripts/maniskill_smolvla/autopilot.sh
scripts/maniskill_smolvla/00_build_envs.pbs
scripts/maniskill_smolvla/01_data_probe.pbs
scripts/maniskill_smolvla/02_data_full.pbs
scripts/maniskill_smolvla/03_convert_full.pbs
scripts/maniskill_smolvla/04_sft_smoke.pbs
scripts/maniskill_smolvla/05_sft_train.pbs
scripts/maniskill_smolvla/06_benchmark.pbs
tests/test_maniskill_smolvla_static.py
.gitignore
docs/eggroll/smolvla_maniskill_handoff.md
```

Existing commits reportedly made earlier:

```text
8abe217 feat(smolvla): add ManiSkill PBS pipeline
8b53fec feat(smolvla): add ManiSkill benchmark
4f986cb fix(smolvla): resume PBS autopilot
RL4VLA b1e4963 feat: save 7d collector state
```

Do not trust commit list blindly; verify:

```bash
git log --oneline -n 10
git -C RL4VLA log --oneline -n 5
```

## Open Risks

1. `toppra` rebuild may fail due Python 3.12/NumPy/Cython/build isolation.
2. Rebuilt `.so` may still include unsupported instr if compiler flags not honored.
3. After import fix, `collect_simpler` may hit asset/render/Vulkan issues.
4. RL4VLA `PutOnPlateInScene25Main-v3` uses fallback planner (`SolvePutCarrot` path) by design? Needs probe evidence.
5. Raw images may not be exactly `(480, 640, 3)` depending camera/render path; converter may need measured shape.
6. LeRobot API in train env may differ from converter assumptions.
7. SmolVLA eval in same Python env as ManiSkill may conflict; two-process bridge may be needed.
8. Full data generation may be slow; need shard/resume collision-free naming before scale.
9. PBS stdout stageout can delay log visibility; prefer in-job log to project artifacts via `tee`.
10. Current `build_envs.sh` source-rebuild line should be improved after current PBS result to avoid build-dep source compile.

## Next Actions

Immediate:

1. Poll `2849726.pbs-7`.
2. If success, run compute import diag.
3. If diag ok, rerun `01_data_probe.pbs`.
4. If probe ok, inspect `.npz` schema/state/action.
5. If probe fail, write concrete RCA before retry.

If `toppra` build fails:

1. Save `toppra_rebuild.out`.
2. Submit fast rebuild job using `--no-build-isolation --no-binary=toppra`.
3. Patch `build_envs.sh` to that better command.
4. Run static tests.

After data probe passes:

1. Generate full data, likely sharded dirs.
2. Convert to LeRobot.
3. Run SFT smoke.
4. Run eval smoke.
5. Scale SFT.

## Useful Commands

Poll current build:

```bash
qstat -f 2849726.pbs-7 || qstat -xf 2849726.pbs-7 || qstat -Hf 2849726.pbs-7
```

Run tests:

```bash
module load Python/3.12.3-GCCcore-13.3.0
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /rds/general/user/aa6622/home/project/.venvs/pirl-rlinf-py312/bin/python \
  -m pytest tests/test_maniskill_smolvla_static.py
```

Check RTX6000 availability:

```bash
python3 "$HOME/.agents/skills/checking-pbs-gpu-availability/scripts/pbs_gpu_snapshot.py" -q v1_gpu72
```

Resume autopilot from data probe:

```bash
MSM_START_STAGE_INDEX=1 \
  /rds/general/user/aa6622/home/project/scripts/maniskill_smolvla/autopilot.sh
```

Resume pinned after selector fix:

```bash
MSM_START_STAGE_INDEX=1 MSM_PBS_HOST=cx3-8-12 \
  /rds/general/user/aa6622/home/project/scripts/maniskill_smolvla/autopilot.sh
```

Do not pin unless needed. Generic `gpu_type=RTX6000` now OK once `toppra` fixed.

## 2026-05-26T19:47Z Autonomous Repair Run

- Committed repaired data/SmolVLA/eval contract and PBS supervisor changes.
- Cancelled stale pre-repair `msm_convert` job `2852597.pbs-7`; it was writing with old converter semantics.
- Submitted CPU PBS supervisor `2857972.pbs-7` via `07_autonomous_supervisor.pbs`.
- Supervisor is running on `cx3-13-3` and queued CPU audit child `2857984.pbs-7`.
- RTX6000 snapshot before queueing showed `count=0`; GPU stages must remain queued/waiting for `gpu_type=RTX6000`.

## 2026-05-26T20:50Z Full Audit RCA

- Full audit `2857984.pbs-7` failed intentionally on duplicate detection.
- Root cause: `96` decoded duplicate trajectory signatures among `16,400` raw demos.
- Shapes valid: `bad_shape_count=0`, `513,662` frames, RGB `480x640x3`, 7D state/action.
- Impact: duplicates are ~0.6% of episodes; not a schema blocker, but should not overweight SFT.
- Fix: converter now deduplicates decoded signatures by default and records skipped duplicate count.
- Validation: focused SmolVLA tests pass (`17 passed`).

## 2026-05-28T00:15Z CPU Regen + Stale Raw Purge

- Full data regen uses `v1_large72`, `select=1:ncpus=124:mem=256gb`, `MSM_FULL_NUM_PROCS=124`, `--sim_backend cpu`.
- New raw output dir: `${MSM_RAW_ROOT}/full_cpu124_v1` (legacy stays `${MSM_RAW_ROOT}/full` until purged).
- `02_data_full.pbs` calls `purge_stale_record_dir.sh` on the CPU job before collection; deletes only `MSM_STALE_RECORD_DIR` (default `full`) when it differs from the new record dir. Logs `du -sh` + `elapsed_s` (~206GB observed on login node; budget 10-30 min on ephemeral FS).
- Convert/audit default input record dir updated to `full_cpu124_v1`. Convert walltime raised to `24:00:00`.

## 2026-05-28T02:23Z Chain running (cuda sync fix)

- Active: `data=2869637` **R** on `cx3-2-26` (`v1_large24a`).
- Downstream: `convert=2869638` → `smoke=2869639` → `train=2869640` → `eval=2869641`.
- RCA `2869610` fail: `sapien_env._get_obs_sensor_data` called `torch.cuda.synchronize()` on CPU node. Patched guard `if torch.cuda.is_available()`.

## 2026-05-28T02:10Z Chain running (superseded)

- `2869610` failed NVIDIA driver after render_backend fix; see above.

## 2026-05-28T Overnight autonomous (render_backend fix)

- Prior `2869567` failed: `failed to find device cuda` on CPU node (default render_backend=gpu).
- Fix: `collect_simpler` + `02_data_full.pbs` pass `--render_backend cpu`; `MSM_PURGE_STALE_RECORD_DIR=0` (legacy `full` already purged).
- Re-submit chain after fix; monitor via `overnight_autonomous.sh` or `qstat`. GPU `Q`/placement-set = wait.

## 2026-05-28T Execute: CPU regen + 10k SFT chain

- PBS chain submitted: `data=2869567` → `convert=2869569` → `smoke=2869570` → `train=2869571` → `eval=2869572` (all `afterok`). **2869567 failed** (cuda render); superseded by requeue above.
- Raw output: `${MSM_RAW_ROOT}/full_cpu124_v1/.../16400/data`. Legacy `${MSM_RAW_ROOT}/full` purged at start of `02_data_full.pbs` (~206GB; logs `elapsed_s`).
- Convert input `full_cpu124_v1` → LeRobot `${MSM_LEROBOT_ROOT}/25main` (SFT/eval read this, not raw npz).
- Main SFT: `MSM_SFT_STEPS=10000`, `MSM_SFT_SAVE_FREQ=1000`, walltime `16:00:00`, out `${MSM_CHECKPOINT_ROOT}/sft_main_<jobid>`.
- PBS venv: `common.sh` `msm_setup_modules` exports `LD_LIBRARY_PATH="${EBROOTPython}/lib:..."` before `lerobot_mw_py312` python.
- GPU jobs may sit `Q` / "placement set too small" while queued — wait unless hard fail.

## 2026-05-28T04:12Z Active chain (32-proc CPU large24a)

| Stage | Job | State |
|-------|-----|-------|
| data | `2869684` | R `cx3-1-5` 32×CPU, `v1_large24a`, 12h wall |
| convert | `2869685` | H afterok |
| sft_smoke | `2869686` | H RTX6000 |
| sft_train | `2869687` | H 10k/1k/16h |
| eval | `2869688` | H |

- Raw: `full_cpu124_v1` (legacy `full` purged earlier; do not re-purge).
- Progress ~04:12Z: **947/16400** NPZ (~21–23/min); ETA ~11–12h total — tight vs 12h wall; contingency `02_data_full_fast.pbs` (64 procs, 16h) committed if walltime kills job.
- Probe `2869679`: 8/8 NPZ on large24a validated CPU path (`render_backend cpu`, `sapien_env` cuda sync guard).

## 2026-05-26T23:58Z Queue/Action-Step Update

- User approved queueing scarce-GPU stages together with PBS `afterok` dependencies.
- Added direct-login helper: `scripts/maniskill_smolvla/queue_afterok_gpu_tail.sh --after-job <convert_job>`.
- SmolVLA ManiSkill SFT contract now uses `n_action_steps=4` with `chunk_size=50` for walltime speedup.
- RTX6000 jobs may sit queued or show scheduler pressure comments like "placement set too small"; wait these out unless PBS marks an impossible run or failure.
- Do not edit the accepted plan file for this; executable scripts, tests, handoff, and Cursor rule are the source of truth.
