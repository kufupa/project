# SmolVLA Oracle Top-15 Full Run (One-Pass, GPU) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run SmolVLA for all 15 oracle top-k targets with 4 episodes each, deterministically pick the best episode per target, emit videos / actions / reward curves / campaign summary, with `SMOLVLA_SAVE_FRAMES=false`, in **one** Slurm GPU allocation—without duplicate Slurm array jobs and without over-engineered OOM mitigations.

**Architecture:** Reuse the existing evaluator (`run_smolvla_eval` in [`project/src/smolvla_pipeline/evaluator.py`](project/src/smolvla_pipeline/evaluator.py)), target runner ([`project/scripts/smolvla/run_smolvla_target_episode.sh`](project/scripts/smolvla/run_smolvla_target_episode.sh)), and campaign summarizer (inline Python in [`project/scripts/smolvla/run_oracle_topk_smolvla_full.sh`](project/scripts/smolvla/run_oracle_topk_smolvla_full.sh)). Sequential episodes already hold **one** episode’s frame list in RAM at a time; the smoke test validated the stack. Fix **campaign creation vs Slurm submission** so the full orchestrator does not submit a 15-way array **and** run the same work on the submit host. Submit **one** `sbatch` job that runs the full bash loop on a GPU node with the same resource profile that succeeded for the 1-episode pilot (`t4`, ~32G RAM, 4 CPUs, 8h wall).

**Tech Stack:** Bash, Slurm (`sbatch`), Python 3.10+ (`lerobot_mw_py310`), Meta-World + LeRobot SmolVLA evaluator, existing `pick_best_episode` in [`project/src/smolvla_pipeline/topk_selection.py`](project/src/smolvla_pipeline/topk_selection.py).

---

## Prerequisites (read first — weak agent checklist)

- **Where to run:** Submit `sbatch` from a **cluster login / submit node** with Slurm. Do **not** run the full 15×4 rollout on the small no-GPU lab VM (OOM / policy).
- **Filesystem:** `PROJECT_ROOT` and `ORACLE_RUN_DIR` must be visible **identical paths** on compute nodes (shared NFS/bitbucket mount). If paths differ, set `ORACLE_RUN_DIR` explicitly in the environment before `sbatch`.
- **Python:** Compute node must execute `SMOLVLA_LEROBOT_ENV_DIR/bin/python` (default `…/workspace/.envs/lerobot_mw_py310/bin/python`). If that path is wrong on the cluster, set `SMOLVLA_LEROBOT_ENV_DIR` or `SMOLVLA_PYTHON_BIN` in the Slurm script.
- **HF / checkpoints:** SmolVLA + VLM weights must resolve on the node (local HF cache or network). Unauthenticated Hub is OK but slower.
- **Launch stdout contract:** [`launch_pushv3_smolvla_topk15.sh`](project/scripts/smolvla/launch_pushv3_smolvla_topk15.sh) prints **four** lines at the end (not the raw Python prints): `campaign_dir=…`, `targets_json=…`, `targets_count=…`, `array=…`. [`run_oracle_topk_smolvla_full.sh`](project/scripts/smolvla/run_oracle_topk_smolvla_full.sh) parses `INFO[0]`/`INFO[1]` with `${var#campaign_dir=}` style strips — do **not** change launch to print only bare paths without updating the full script.

---

## File map (who does what)

| File | Responsibility |
|------|----------------|
| [`project/scripts/smolvla/run_oracle_topk_smolvla_full.sh`](project/scripts/smolvla/run_oracle_topk_smolvla_full.sh) | End-to-end: create campaign, loop 15 targets, verify artifacts, pick best, write `smolvla_topk_best_summary.json`. **Must not** submit duplicate array jobs when used as the body of a single GPU job. |
| [`project/scripts/smolvla/launch_pushv3_smolvla_topk15.sh`](project/scripts/smolvla/launch_pushv3_smolvla_topk15.sh) | Materialize `campaigns/.../targets.json` + `campaign_manifest.json`; optionally `sbatch` array. |
| [`project/scripts/smolvla/submit_pushv3_smolvla_topk15.slurm`](project/scripts/smolvla/submit_pushv3_smolvla_topk15.slurm) | **Array** worker: one target per task (still valid for future use; not used for the one-node full run). |
| **New:** [`project/scripts/smolvla/submit_smolvla_topk_full_one_node.slurm`](project/scripts/smolvla/submit_smolvla_topk_full_one_node.slurm) | Single GPU job: `cd` to repo, export env, run full orchestrator. |
| [`project/scripts/smolvla/run_smolvla_target_episode.sh`](project/scripts/smolvla/run_smolvla_target_episode.sh) | Per-target: `run_metaworld_smolvla_eval.py` + `verify_smolvla_run_artifacts.py` + `target_episode_summary.json`. |

---

### Task 1: Eliminate duplicate Slurm work in the full orchestrator

**Files:**

- Modify: [`project/scripts/smolvla/run_oracle_topk_smolvla_full.sh`](project/scripts/smolvla/run_oracle_topk_smolvla_full.sh) (lines 19–21 area)

**Problem (verified in repo):** `run_oracle_topk_smolvla_full.sh` calls `launch_pushv3_smolvla_topk15.sh` **without** `--dry-run`, which runs `sbatch --array 0-14 …` and then the same script runs `run_smolvla_target_episode.sh` for every index on the **current** machine. That duplicates work and wastes cluster budget.

**Design:** When `SMOLVLA_TOPK_LAUNCH_MODE=dry-run` (default for full one-node run), call launch with `--dry-run` so only campaign files are created. When `SMOLVLA_TOPK_LAUNCH_MODE=sbatch`, keep current behavior for operators who want array-only execution (document as advanced).

- [ ] **Step 1: Patch `run_oracle_topk_smolvla_full.sh` launch invocation**

Replace the fixed launch call with a mode switch:

```bash
LAUNCH_MODE="${SMOLVLA_TOPK_LAUNCH_MODE:-dry-run}"
LAUNCH_ARGS=(--oracle-run-dir "${ORACLE_RUN_DIR}" --top-k "${TOP_K}")
if [[ "${LAUNCH_MODE}" == "dry-run" ]]; then
  LAUNCH_ARGS+=(--dry-run)
elif [[ "${LAUNCH_MODE}" != "sbatch" ]]; then
  echo "error: SMOLVLA_TOPK_LAUNCH_MODE must be dry-run or sbatch (got: ${LAUNCH_MODE})" >&2
  exit 2
fi

CAMPAIGN_OUT="$(
  bash "${SCRIPT_DIR}/launch_pushv3_smolvla_topk15.sh" "${LAUNCH_ARGS[@]}"
)"
```

**Scope:** Replace only the **`CAMPAIGN_OUT=$( bash … launch … )` block** (currently lines 19–21 in `run_oracle_topk_smolvla_full.sh`). Keep the next lines (`readarray`, `CAMPAIGN_DIR`, `TARGETS_JSON`) unchanged.

- [ ] **Step 2: Syntax-check**

Run:

```bash
bash -n /vol/bitbucket/aa6622/project/scripts/smolvla/run_oracle_topk_smolvla_full.sh
```

Expected: exit code `0`, no output.

- [ ] **Step 3: Dry-run smoke (no Slurm)**

Run:

```bash
cd /vol/bitbucket/aa6622/project
SMOLVLA_TOPK_LAUNCH_MODE=dry-run TOP_K=1 EPISODES_PER_TARGET=1 bash scripts/smolvla/run_oracle_topk_smolvla_full.sh
```

Expected: creates a new `campaigns/pushv3_smolvla_topk1_*` dir with `targets.json`; runs **one** target loop iteration. `EPISODES_PER_TARGET=1` keeps smoke fast (production remains default `4` in the script / Slurm file). Confirm stdout contains **no** line starting with `Submitted batch job` (dry-run must not call `sbatch`). A line like `dry-run sbatch command:` is OK.

- [ ] **Step 4: Commit**

```bash
cd /vol/bitbucket/aa6622/project
git add scripts/smolvla/run_oracle_topk_smolvla_full.sh
git commit -m "fix(smolvla): default full top-k orchestrator to launch dry-run to avoid duplicate sbatch"
```

---

### Task 2: Single-node Slurm entrypoint (proven resource profile)

**Files:**

- Create: [`project/scripts/smolvla/submit_smolvla_topk_full_one_node.slurm`](project/scripts/smolvla/submit_smolvla_topk_full_one_node.slurm)

**Rationale:** One job runs all 15 targets × 4 episodes on **one** GPU with enough RAM; avoids login-node OOM and avoids array + local double execution.

- [ ] **Step 1: Add Slurm script**

Create `submit_smolvla_topk_full_one_node.slurm` with:

```bash
#!/bin/bash
#SBATCH --job-name=smolvla-topk-full
#SBATCH --partition=t4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=smolvla_topk_full_%j.out

set -euo pipefail

# Repo root = project/scripts/smolvla -> project
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export SMOLVLA_TOPK_LAUNCH_MODE="${SMOLVLA_TOPK_LAUNCH_MODE:-dry-run}"
export SMOLVLA_SAVE_FRAMES=false
export EPISODES_PER_TARGET="${EPISODES_PER_TARGET:-4}"
export ORACLE_RUN_DIR="${ORACLE_RUN_DIR:-/vol/bitbucket/aa6622/project/artifacts/phase06_oracle_baseline/run_20260411T131839Z_ep60_voracle_tpush_v3_s1000_r402093}"
export SMOLVLA_LEROBOT_ENV_DIR="${SMOLVLA_LEROBOT_ENV_DIR:-${WORKSPACE_ROOT}/.envs/lerobot_mw_py310}"

bash "${PROJECT_ROOT}/scripts/smolvla/run_oracle_topk_smolvla_full.sh"
```

Adjust `ORACLE_RUN_DIR` default if your canonical oracle run path differs.

- [ ] **Step 2: Executable bit (if needed)**

```bash
chmod +x /vol/bitbucket/aa6622/project/scripts/smolvla/submit_smolvla_topk_full_one_node.slurm
```

(`sbatch` accepts non-executable scripts on many sites; harmless if redundant.)

- [ ] **Step 3: Submit**

From a cluster login node (not the 8GB no-GPU VM):

```bash
cd /vol/bitbucket/aa6622/project
sbatch scripts/smolvla/submit_smolvla_topk_full_one_node.slurm
```

Expected: `Submitted batch job <JOBID>`. Slurm writes `smolvla_topk_full_<JOBID>.out` in the **submission working directory** (here: `project/`), not necessarily under `scripts/smolvla/`.

- [ ] **Step 4: Commit**

```bash
git add scripts/smolvla/submit_smolvla_topk_full_one_node.slurm
git commit -m "chore(smolvla): add one-node Slurm entrypoint for full top-k campaign"
```

---

### Task 3: Monitor and validate artifacts (acceptance)

**Files:** (read-only checks under `project/artifacts/phase07_smolvla_baseline/campaigns/`)

- [ ] **Step 1: Watch job**

```bash
squeue -u "$USER" -o '%.18i %.2t %.10M %.30R %j'
```

Expected: job `RUNNING` then gone with exit 0 (check `smolvla_topk_full_<JOBID>.out` in the directory from which you ran `sbatch`).

- [ ] **Step 2: Resolve `CAMPAIGN_DIR`**

Newest campaign directory under artifacts (run on submit host):

```bash
CAMPAIGN_DIR="$(ls -dt /vol/bitbucket/aa6622/project/artifacts/phase07_smolvla_baseline/campaigns/pushv3_smolvla_topk15_* 2>/dev/null | head -1)"
test -n "${CAMPAIGN_DIR}" && test -f "${CAMPAIGN_DIR}/targets.json"
echo "CAMPAIGN_DIR=${CAMPAIGN_DIR}"
```

If empty, list `campaigns/` and pick the run that matches your job’s log timestamps.

- [ ] **Step 3: Campaign summary JSON + videos**

```bash
SUMMARY="${CAMPAIGN_DIR}/smolvla_topk_best_summary.json"
test -f "${SUMMARY}"

python3 - "${SUMMARY}" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
rows = json.loads(p.read_text(encoding="utf-8"))
assert len(rows) == 15, len(rows)
for r in rows:
    assert "best_video" in r and Path(r["best_video"]).is_file(), r
print("ok", len(rows))
PY
```

**Critical:** Use **`python3 - "$SUMMARY"`** (dash = read program from stdin). The path is **`sys.argv[1]`**. Wrong: `python3 "$SUMMARY" <<'PY'` (that executes the JSON file as Python and breaks).

Expected: `ok 15`.

- [ ] **Step 4: No raw frames (spot-check)**

Ensure `CAMPAIGN_DIR` is still set (Step 2). For each row, `best_video` lives under a `run_*` tree. Check those run roots for stray PNG dumps:

```bash
python3 <<PY
import json
from pathlib import Path
campaign = Path("${CAMPAIGN_DIR}")
summary = json.loads((campaign / "smolvla_topk_best_summary.json").read_text())
seen = []
for r in summary:
    # best_video = .../run_*/videos/<task_camera>/eval_episode_XXXX.mp4 -> run root is three parents up
    run_dir = Path(r["best_video"]).resolve().parent.parent.parent
    pngs = list(run_dir.rglob("frame_*.png"))
    if pngs:
        seen.extend(pngs[:3])
if seen:
    raise SystemExit(f"unexpected frame_*.png: {seen[:5]}")
print("no frame_*.png under best run roots")
PY
```

Expected: `no frame_*.png under best run roots`.

- [ ] **Step 5: Per-target `run_manifest`**

Read `task_0000.json` → `run_dir` → open `run_manifest.json`. Assert: `"save_frames": false`, `"episodes"` length **4**, each episode has `sum_reward`, `max_reward`, `paths.video`, `paths.actions`.

- [ ] **Step 6: `campaign_manifest.json` submission mode**

Open `${CAMPAIGN_DIR}/campaign_manifest.json` → `submission.mode` should be `dry-run` for the one-node path (proves no accidental array submit from launch).

---

## Failure modes (if something breaks)

- **`sbatch: error … Requested node configuration is not available`:** Partition full or bad `--mem`/`--cpus-per-task` combo for that partition. Try another GPU partition your site documents, or lower CPUs (some clusters reserve CPU per GPU).
- **`user_env_retrieval_failed` / job stuck pending:** Prior cluster issue with propagating login env; this plan uses `SMOLVLA_TOPK_LAUNCH_MODE=dry-run` inside the job so launch does not need to submit nested `sbatch`. If problems persist, run `run_oracle_topk_smolvla_full.sh` from an interactive GPU allocation instead of nested batch.
- **`python not found` / wrong torch:** Fix `SMOLVLA_LEROBOT_ENV_DIR` in the Slurm script to the real venv on compute nodes.
- **Summary has fewer than 15 rows or job died mid-loop:** Re-submit a fresh job; partial campaigns stay under `campaigns/` for forensics — do not delete until you know you have a full success.

---

## Explicit non-goals (YAGNI)

- No incremental `run_manifest.json` rewriting, no `SMOLVLA_KEEP_BEST_ONLY`, no duplicate `best_episode` logic inside `run_smolvla_eval`—the existing per-episode disk writes and end-of-target `pick_best_episode` in the shell loop are sufficient for this campaign.
- No streaming video refactor unless a **measured** OOM reproduces on the GPU node after Task 2’s resource profile.

---

## Self-review (spec coverage)

| Requirement | Task |
|---------------|------|
| 15 oracle top-k environments | Default `TOP_K=15` in full script + 15 rows in summary |
| 4 SmolVLA episodes per env | `EPISODES_PER_TARGET=4` default in full script + slurm exports |
| Best episode per env (deterministic) | Existing `pick_best_episode` in full script loop |
| Video + actions + reward curves per episode | Existing evaluator + verifier |
| No `frame_*.png` | `SMOLVLA_SAVE_FRAMES=false` hard/default in slurm + full script |
| GPU / enough RAM | `t4`, `32G`, `gpu:1`, 8h in new slurm file |
| One pass, no duplicate cluster work | Task 1 `dry-run` default + Task 2 single job |

**Placeholder scan:** None intentional; oracle path default is explicit—change if your run id differs.

**Type/consistency:** Summary JSON schema unchanged; `task_*.json` + `smolvla_topk_best_summary.json` remain the contract.

---

## Execution handoff

**Plan complete and saved to** `project/docs/superpowers/plans/2026-04-11-smolvla-topk15-full-run.md` (full path: `/vol/bitbucket/aa6622/project/docs/superpowers/plans/2026-04-11-smolvla-topk15-full-run.md`).

**Two execution options:**

1. **Subagent-Driven (recommended)** — Dispatch a fresh subagent per task, review between tasks, fast iteration. **REQUIRED SUB-SKILL:** superpowers:subagent-driven-development.

2. **Inline Execution** — Run tasks in this session with checkpoints. **REQUIRED SUB-SKILL:** superpowers:executing-plans.

**Which approach?**
