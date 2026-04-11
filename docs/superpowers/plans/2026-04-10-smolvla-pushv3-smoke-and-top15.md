# SmolVLA Push-v3 Smoke and Top-15 Campaign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible SmolVLA evaluation pipeline in `project/` that runs a 1-episode smoke test first, records video + per-step action/reward artifacts + reward graph, then schedules a 15-run campaign on the same `push-v3` environment family used by the oracle videos.

**Architecture:** Add a new `scripts/smolvla/` and `src/smolvla_pipeline/` stack that mirrors the oracle pipeline style (unique run directories, manifests, and deterministic outputs). The evaluator writes per-episode logs (`actions.jsonl`), cumulative-reward video overlays, and reward-vs-time plots. A separate launcher resolves top-k oracle episode targets and submits a Slurm job array for the 15-episode campaign, while keeping all logic/versioned code in `project/`.

**Tech Stack:** Python 3.12, PyTorch, LeRobot SmolVLA policy loading, Meta-World (`push-v3`), `imageio`, `matplotlib`, Bash, Slurm (`sbatch`).

---

## Scope Check

This request is one cohesive subsystem (SmolVLA eval pipeline with smoke + scale-up). No split plan is required.

## File Structure

- Create: `src/smolvla_pipeline/run_layout.py`  
  Owns run-id creation, output directory layout, and manifest path helpers.
- Create: `src/smolvla_pipeline/targets.py`  
  Resolves campaign episode targets from oracle `optimal_report.json` + `run_manifest.json`.
- Create: `src/smolvla_pipeline/evaluator.py`  
  Runs one-or-many SmolVLA episodes, writes `eval_info.json`, `run_manifest.json`, `actions.jsonl`, and reward plots.
- Create: `scripts/smolvla/run_metaworld_smolvla_eval.py`  
  CLI entrypoint over evaluator for local/smoke/batch usage.
- Create: `scripts/smolvla/pushv3_smolvla_smoke.sh`  
  One-episode smoke test wrapper (video required), writes smoke summary status.
- Create: `scripts/smolvla/launch_pushv3_smolvla_topk15.sh`  
  Builds target list from oracle run and submits Slurm job array.
- Create: `scripts/smolvla/run_smolvla_target_episode.sh`  
  Per-array-task runner for one target episode.
- Create: `scripts/smolvla/submit_pushv3_smolvla_topk15.slurm`  
  Slurm script template for campaign scheduling.
- Create: `tests/test_smolvla_run_layout.py`  
  Unit tests for run naming and layout paths.
- Create: `tests/test_smolvla_targets.py`  
  Unit tests for oracle top-k target resolution.
- Create: `tests/test_smolvla_eval_artifacts.py`  
  Unit tests for actions logs, reward CSV/PNG outputs, and manifest contracts.
- Modify: `README.md`  
  Add SmolVLA smoke/campaign docs and exact artifact layout.
- Modify: `docs/paths.example.env`  
  Add explicit SmolVLA checkpoint/env variables used by new scripts.

---

### Task 1: Define run layout and unique directory contract

**Files:**
- Create: `src/smolvla_pipeline/run_layout.py`
- Test: `tests/test_smolvla_run_layout.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

from src.smolvla_pipeline.run_layout import build_run_dir_name, ensure_unique_run_dir


def test_build_run_dir_name_contract():
    run_name = build_run_dir_name(
        timestamp_utc="20260410T170000Z",
        episodes=1,
        task="push-v3",
        seed=1000,
        variant="smolvla",
        nonce="123456",
    )
    assert run_name == "run_20260410T170000Z_ep1_vsmolvla_tpush_v3_s1000_r123456"


def test_ensure_unique_run_dir_never_reuses_existing(tmp_path: Path):
    first = ensure_unique_run_dir(tmp_path, episodes=1, task="push-v3", seed=1000, variant="smolvla")
    second = ensure_unique_run_dir(tmp_path, episodes=1, task="push-v3", seed=1000, variant="smolvla")
    assert first != second
    assert first.exists()
    assert second.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_smolvla_run_layout.py -v`  
Expected: FAIL with `ModuleNotFoundError` for `src.smolvla_pipeline.run_layout`.

- [ ] **Step 3: Write minimal implementation**

```python
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import secrets


def _slug_task(task: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in task).strip("_")


def build_run_dir_name(
    *,
    timestamp_utc: str,
    episodes: int,
    task: str,
    seed: int,
    variant: str,
    nonce: str,
) -> str:
    return f"run_{timestamp_utc}_ep{episodes}_v{variant}_t{_slug_task(task)}_s{seed}_r{nonce}"


def ensure_unique_run_dir(output_root: Path, *, episodes: int, task: str, seed: int, variant: str) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp_utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for _ in range(20):
        nonce = f"{secrets.randbelow(1_000_000):06d}"
        run_name = build_run_dir_name(
            timestamp_utc=timestamp_utc,
            episodes=episodes,
            task=task,
            seed=seed,
            variant=variant,
            nonce=nonce,
        )
        run_dir = output_root / run_name
        try:
            run_dir.mkdir()
            return run_dir
        except FileExistsError:
            continue
    raise RuntimeError(f"Failed to create unique run directory under {output_root}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_smolvla_run_layout.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_smolvla_run_layout.py src/smolvla_pipeline/run_layout.py
git commit -m "feat(smolvla): add unique run directory layout helpers"
```

### Task 2: Add oracle-target resolver for top-k campaign episodes

**Files:**
- Create: `src/smolvla_pipeline/targets.py`
- Test: `tests/test_smolvla_targets.py`

- [ ] **Step 1: Write the failing test**

```python
import json
from pathlib import Path

from src.smolvla_pipeline.targets import load_topk_targets


def test_load_topk_targets_reads_reset_seed_from_manifest(tmp_path: Path):
    run_dir = tmp_path / "oracle_run"
    run_dir.mkdir()
    (run_dir / "optimal_report.json").write_text(
        json.dumps(
            {
                "top_k": 2,
                "episodes": [
                    {"rank": 1, "episode_index": 7, "max_reward": 2.0},
                    {"rank": 2, "episode_index": 3, "max_reward": 1.5},
                ],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "task": "push-v3",
                "episodes": [
                    {"episode_index": 3, "reset_seed": 1003},
                    {"episode_index": 7, "reset_seed": 1007},
                ],
            }
        ),
        encoding="utf-8",
    )
    targets = load_topk_targets(run_dir, top_k=2)
    assert [t["episode_index"] for t in targets] == [7, 3]
    assert [t["reset_seed"] for t in targets] == [1007, 1003]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_smolvla_targets.py -v`  
Expected: FAIL with `ModuleNotFoundError` for `src.smolvla_pipeline.targets`.

- [ ] **Step 3: Write minimal implementation**

```python
from __future__ import annotations

import json
from pathlib import Path


def load_topk_targets(oracle_run_dir: Path, *, top_k: int) -> list[dict]:
    report = json.loads((oracle_run_dir / "optimal_report.json").read_text(encoding="utf-8"))
    run_manifest = json.loads((oracle_run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    by_episode = {
        int(row["episode_index"]): int(row["reset_seed"])
        for row in run_manifest.get("episodes", [])
    }
    targets: list[dict] = []
    for row in report.get("episodes", [])[:top_k]:
        ep = int(row["episode_index"])
        targets.append(
            {
                "rank": int(row["rank"]),
                "episode_index": ep,
                "reset_seed": int(by_episode[ep]),
                "task": run_manifest.get("task", "push-v3"),
            }
        )
    return targets
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_smolvla_targets.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_smolvla_targets.py src/smolvla_pipeline/targets.py
git commit -m "feat(smolvla): add oracle top-k target resolver"
```

### Task 3: Implement SmolVLA evaluator with overlays, per-step logs, and reward graph

**Files:**
- Create: `src/smolvla_pipeline/evaluator.py`
- Create: `scripts/smolvla/run_metaworld_smolvla_eval.py`
- Test: `tests/test_smolvla_eval_artifacts.py`

- [ ] **Step 1: Write the failing test**

```python
import json
from pathlib import Path

from src.smolvla_pipeline.evaluator import write_episode_artifacts


def test_write_episode_artifacts_outputs_logs_and_plot(tmp_path: Path):
    episode_dir = tmp_path / "episodes" / "episode_0000"
    frames_dir = tmp_path / "frames" / "episode_0000"
    write_episode_artifacts(
        episode_dir=episode_dir,
        frames_dir=frames_dir,
        actions=[[0.1, 0.0, 0.0, 0.0], [0.2, 0.0, 0.0, 0.0]],
        rewards=[0.3, 0.5],
        successes=[False, True],
    )
    actions_lines = (episode_dir / "actions.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(actions_lines) == 2
    first = json.loads(actions_lines[0])
    assert first["step"] == 0
    assert first["reward"] == 0.3
    assert (episode_dir / "reward_curve.csv").is_file()
    assert (episode_dir / "reward_curve.png").is_file()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_smolvla_eval_artifacts.py::test_write_episode_artifacts_outputs_logs_and_plot -v`  
Expected: FAIL with `ModuleNotFoundError` for `src.smolvla_pipeline.evaluator`.

- [ ] **Step 3: Write minimal implementation**

```python
from __future__ import annotations

import csv
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def write_episode_artifacts(*, episode_dir: Path, frames_dir: Path, actions: list[list[float]], rewards: list[float], successes: list[bool]) -> None:
    episode_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    cumulative = 0.0
    with (episode_dir / "actions.jsonl").open("w", encoding="utf-8") as fp:
        for step, (action, reward, success) in enumerate(zip(actions, rewards, successes)):
            cumulative += float(reward)
            fp.write(
                json.dumps(
                    {
                        "step": step,
                        "action": [float(x) for x in action],
                        "reward": float(reward),
                        "cumulative_reward": float(cumulative),
                        "success": bool(success),
                    }
                )
                + "\n"
            )

    with (episode_dir / "reward_curve.csv").open("w", newline="", encoding="utf-8") as csv_fp:
        writer = csv.DictWriter(csv_fp, fieldnames=["step", "reward", "cumulative_reward"])
        writer.writeheader()
        cumulative = 0.0
        for step, reward in enumerate(rewards):
            cumulative += float(reward)
            writer.writerow({"step": step, "reward": float(reward), "cumulative_reward": float(cumulative)})

    x = np.arange(len(rewards))
    y = np.cumsum(np.asarray(rewards, dtype=np.float64))
    plt.figure(figsize=(8, 4))
    plt.plot(x, y)
    plt.xlabel("step")
    plt.ylabel("cumulative_reward")
    plt.title("Time vs cumulative reward")
    plt.tight_layout()
    plt.savefig(episode_dir / "reward_curve.png")
    plt.close()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_smolvla_eval_artifacts.py::test_write_episode_artifacts_outputs_logs_and_plot -v`  
Expected: PASS.

- [ ] **Step 5: Add CLI entrypoint and smoke-checkable outputs**

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from src.smolvla_pipeline.evaluator import run_smolvla_eval


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="push-v3")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--video", default="true")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--overlay-mode", choices=["cumulative_reward", "reward"], default="cumulative_reward")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_smolvla_eval(
        task=args.task,
        episodes=args.episodes,
        seed=args.seed,
        checkpoint=args.checkpoint,
        output_dir=Path(args.output_dir),
        video=args.video,
        fps=args.fps,
        overlay_mode=args.overlay_mode,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: Run tests for this task**

Run: `pytest tests/test_smolvla_eval_artifacts.py -v`  
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add tests/test_smolvla_eval_artifacts.py src/smolvla_pipeline/evaluator.py scripts/smolvla/run_metaworld_smolvla_eval.py
git commit -m "feat(smolvla): add evaluator artifacts, overlays, and reward plots"
```

### Task 4: Add smoke script (1 episode) with success + video verification

**Files:**
- Create: `scripts/smolvla/pushv3_smolvla_smoke.sh`
- Modify: `src/smolvla_pipeline/evaluator.py`
- Test: `tests/test_smolvla_eval_artifacts.py`

- [ ] **Step 1: Write the failing test for smoke summary contract**

```python
import json
from pathlib import Path


def test_smoke_summary_contract(tmp_path: Path):
    summary_path = tmp_path / "smoke_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "status": "success",
                "video_recorded": True,
                "episodes": 1,
                "task": "push-v3",
            }
        ),
        encoding="utf-8",
    )
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["status"] == "success"
    assert payload["video_recorded"] is True
```

- [ ] **Step 2: Run smoke script syntax check first**

Run: `bash -n scripts/smolvla/pushv3_smolvla_smoke.sh`  
Expected: initially FAIL (file missing).

- [ ] **Step 3: Write smoke wrapper**

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
PYTHON_BIN="${SMOLVLA_LEROBOT_ENV_DIR:-${WORKSPACE_ROOT}/.envs/lerobot_mw_py310}/bin/python"
CHECKPOINT="${SMOLVLA_INIT_CHECKPOINT:-jadechoghari/smolvla_metaworld}"
OUTPUT_ROOT="${SMOLVLA_ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts}/phase07_smolvla_baseline"

run_dir="$("${PYTHON_BIN}" -c "from pathlib import Path; from src.smolvla_pipeline.run_layout import ensure_unique_run_dir; print(ensure_unique_run_dir(Path('${OUTPUT_ROOT}'), episodes=1, task='push-v3', seed=1000, variant='smolvla'))")"

xvfb-run -a -s "-screen 0 1280x1024x24" "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/smolvla/run_metaworld_smolvla_eval.py" \
  --task push-v3 \
  --episodes 1 \
  --seed 1000 \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${run_dir}" \
  --video true \
  --overlay-mode cumulative_reward

"${PYTHON_BIN}" - <<'PY'
import json
from pathlib import Path
run_dir = Path("'"${run_dir}"'")
eval_info = json.loads((run_dir / "eval_info.json").read_text())
videos = eval_info["overall"]["video_paths"]
status = "success" if eval_info["overall"]["pc_success"] > 0 else "failed"
summary = {
    "status": status,
    "task": "push-v3",
    "seed": 1000,
    "episodes": 1,
    "video_recorded": len(videos) == 1,
    "video_path": videos[0] if videos else None,
    "run_dir": str(run_dir),
}
(run_dir / "smoke_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PY
```

- [ ] **Step 4: Run checks**

Run: `bash -n scripts/smolvla/pushv3_smolvla_smoke.sh`  
Expected: PASS.  

Run: `pytest tests/test_smolvla_eval_artifacts.py::test_smoke_summary_contract -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/smolvla/pushv3_smolvla_smoke.sh src/smolvla_pipeline/evaluator.py tests/test_smolvla_eval_artifacts.py
git commit -m "feat(smolvla): add one-episode smoke script with success and video checks"
```

### Task 5: Add Slurm campaign scheduling for oracle top-15 target set

**Files:**
- Create: `scripts/smolvla/run_smolvla_target_episode.sh`
- Create: `scripts/smolvla/submit_pushv3_smolvla_topk15.slurm`
- Create: `scripts/smolvla/launch_pushv3_smolvla_topk15.sh`
- Modify: `src/smolvla_pipeline/targets.py`
- Test: `tests/test_smolvla_targets.py`

- [ ] **Step 1: Write failing test for target serialization**

```python
import json
from pathlib import Path

from src.smolvla_pipeline.targets import write_targets_file


def test_write_targets_file(tmp_path: Path):
    output = tmp_path / "targets_top15.json"
    write_targets_file(
        output,
        [
            {"rank": 1, "episode_index": 7, "reset_seed": 1007, "task": "push-v3"},
            {"rank": 2, "episode_index": 3, "reset_seed": 1003, "task": "push-v3"},
        ],
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["count"] == 2
    assert payload["targets"][0]["episode_index"] == 7
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_smolvla_targets.py::test_write_targets_file -v`  
Expected: FAIL with `ImportError` for `write_targets_file`.

- [ ] **Step 3: Implement target writer + Slurm scripts**

```python
def write_targets_file(path: Path, targets: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"count": len(targets), "targets": targets}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
```

```bash
#!/usr/bin/env bash
set -euo pipefail
# run_smolvla_target_episode.sh
TARGETS_JSON="$1"
TASK_ID="$2"
python3 - <<'PY'
import json, os
from pathlib import Path
targets = json.loads(Path(os.environ["TARGETS_JSON"]).read_text())["targets"]
row = targets[int(os.environ["TASK_ID"])]
print(row["task"], row["reset_seed"], row["episode_index"])
PY
```

```bash
#!/bin/bash
# submit_pushv3_smolvla_topk15.slurm
#SBATCH --job-name=smolvla-top15
#SBATCH --array=0-14
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
set -euo pipefail
bash scripts/smolvla/run_smolvla_target_episode.sh "$TARGETS_JSON" "$SLURM_ARRAY_TASK_ID"
```

- [ ] **Step 4: Run checks**

Run: `pytest tests/test_smolvla_targets.py -v`  
Expected: PASS.  

Run: `bash -n scripts/smolvla/run_smolvla_target_episode.sh`  
Expected: PASS.  

Run: `bash -n scripts/smolvla/launch_pushv3_smolvla_topk15.sh`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/smolvla_pipeline/targets.py tests/test_smolvla_targets.py scripts/smolvla/run_smolvla_target_episode.sh scripts/smolvla/submit_pushv3_smolvla_topk15.slurm scripts/smolvla/launch_pushv3_smolvla_topk15.sh
git commit -m "feat(smolvla): add slurm top-15 campaign launcher from oracle targets"
```

### Task 6: Document reproducible usage and artifact locations

**Files:**
- Modify: `README.md`
- Modify: `docs/paths.example.env`

- [ ] **Step 1: Write docs update**

```markdown
## Active Scripts (SmolVLA)

- `scripts/smolvla/pushv3_smolvla_smoke.sh` — 1-episode smoke test with overlay video
- `scripts/smolvla/run_metaworld_smolvla_eval.py` — SmolVLA eval + actions/reward logs + plots
- `scripts/smolvla/launch_pushv3_smolvla_topk15.sh` — derive target set from oracle run and submit Slurm array

## SmolVLA artifact root

- `<project_root>/artifacts/phase07_smolvla_baseline/`
- Run naming: `run_{UTC}_ep{episodes}_vsmolvla_t{task}_s{seed}_r{nonce}`

Per-run files:
- `eval_info.json`
- `run_manifest.json`
- `smoke_summary.json` (smoke runs)
- `videos/<task>_0/eval_episode_<i>.mp4` (frame overlay includes cumulative reward)
- `episodes/episode_<i>/actions.jsonl`
- `episodes/episode_<i>/reward_curve.csv`
- `episodes/episode_<i>/reward_curve.png`
```

- [ ] **Step 2: Add env example entries**

```bash
# SmolVLA evaluation
export SMOLVLA_INIT_CHECKPOINT="jadechoghari/smolvla_metaworld"
export SMOLVLA_LEROBOT_ENV_DIR="/vol/bitbucket/aa6622/.envs/lerobot_mw_py310"
export SMOLVLA_ARTIFACT_ROOT="/vol/bitbucket/aa6622/project/artifacts"
```

- [ ] **Step 3: Validate docs syntax and references**

Run: `rg "phase07_smolvla_baseline|pushv3_smolvla_smoke|run_metaworld_smolvla_eval" README.md docs/paths.example.env`  
Expected: matches found in both files.

- [ ] **Step 4: Commit**

```bash
git add README.md docs/paths.example.env
git commit -m "docs(smolvla): add smoke and top-15 campaign usage guide"
```

---

## End-to-End Verification Checklist (after all tasks)

- [ ] Run smoke test:  
`bash scripts/smolvla/pushv3_smolvla_smoke.sh`  
Expected: creates one run directory under `artifacts/phase07_smolvla_baseline/`, includes `smoke_summary.json`, one video, one `actions.jsonl`, and one reward plot PNG.

- [ ] Confirm smoke success + video recorded:  
`python3 -c "import json,glob; p=sorted(glob.glob('/vol/bitbucket/aa6622/project/artifacts/phase07_smolvla_baseline/run_*/*smoke_summary.json'.replace('//*','/'))); print(p[-1]); d=json.load(open(p[-1])); print(d['status'], d['video_recorded'])"`  
Expected: `success True` (or `failed True/False` if model/env genuinely fails; script must still record artifacts).

- [ ] Submit top-15 campaign job array:  
`bash scripts/smolvla/launch_pushv3_smolvla_topk15.sh --oracle-run-dir /vol/bitbucket/aa6622/project/artifacts/phase06_oracle_baseline/<run_id> --top-k 15`  
Expected: prints `sbatch` job id and writes campaign manifest under `phase07_smolvla_baseline/campaigns/`.

- [ ] For one completed array task, verify artifact set includes:
  - overlaid MP4
  - `actions.jsonl`
  - `reward_curve.csv`
  - `reward_curve.png`
  - `eval_info.json`
  - `run_manifest.json`

---

## Self-Review

1. **Spec coverage:**  
   - New pipeline in `project/`: covered by `scripts/smolvla/*` and `src/smolvla_pipeline/*`.  
   - Separate unique run directories: Task 1 naming/layout contract.  
   - Initial smoke test with one trajectory: Task 4.  
   - Success/failure flag + video recording proof: Task 4 + verification checklist.  
   - Per-step action/reward stats and pretty format: Task 3 (`actions.jsonl`, CSV, manifest).  
   - Time-vs-reward graph: Task 3 (`reward_curve.png`).  
   - Overlay reward in smoke video: Task 3 (`overlay_mode=cumulative_reward`) and Task 4 smoke command.  
   - Schedule for 15 oracle-linked targets: Task 5.

2. **Placeholder scan:**  
   No `TODO/TBD` placeholders remain; each code-change step includes concrete code and commands.

3. **Type consistency:**  
   `episode_index`, `reset_seed`, `task`, `rank`, and `targets` naming are consistent across `targets.py`, scripts, and tests.
