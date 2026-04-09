# Minimal paired trajectory schema — push-v3 + CEM + JEPA-WM export

This document locks the **minimal** on-disk contract for phase07 exports consumed by phase08 (`bridge_builder.py`). It is the single source of truth for **Task 0 Step 1**; implementation may store tensors inline or by reference, but **field names and semantics** below are stable.

---

## Versioning

| Field | Type | Value / notes |
|-------|------|----------------|
| `schema_version` | string | e.g. `cem_paired_push_v3_v0` — bump suffix when breaking reader compatibility. |
| `export_mode` | string (constant) | **`cem_paired_push_v3`** — bridge and manifest checks must accept only this mode for the paired pipeline (no legacy random-rollout export). |

---

## Per-episode (required)

| Field | Type | Semantics |
|-------|------|-----------|
| `pair_key` | string | Stable identifier linking the **executed (real / sim push-v3)** arm and the **CEM + JEPA-WM latent** arm for the same episode. Use a UUID v4 or a **deterministic hash** over fixed episode inputs (e.g. seed + task id + init signature) so replays reproduce the same key. |
| `task_id` | string (recommended) | e.g. Meta-World task name `push-v3` for validation and filtering. |
| `episode_index` | int (optional) | Monotonic or dataset-local episode id for debugging. |

---

## Per timestep or chunk — **executed** arm (push-v3)

Aligned with the **same** `step_index` (or chunk start index) as the predicted arm.

| Field | Type | Semantics |
|-------|------|-----------|
| `step_index` | int | Zero-based index within the episode (or chunk start if records are chunked). |
| Observation | dict or refs | At minimum, keys or tensor refs the bridge can resolve to LeRobot-style **`images`**, **`state`** (and any policy-specific keys). May be `obs_keys` + paths, inline tensors, or nested `executed.obs` — exporter and bridge must agree on one layout per `schema_version`. |
| `action` | tensor or array ref | **Actually executed** control at this step (or chunk label), not CEM proposal. |
| `reward` | float (optional) | Environment reward if logged. |
| `success` | bool or float (optional) | Task success signal if available (terminal or per-step). |
| `done` | bool (optional) | Episode termination at this step. |

---

## Per timestep or chunk — **predicted / CEM** arm (JEPA-WM)

Same `step_index` (or chunk alignment) as the executed arm for this `pair_key`.

| Field | Type | Semantics |
|-------|------|-----------|
| `latent_pred` | tensor ref or id | WM-predicted latent (or identifier to load it from a sidecar), paired to this step. |
| `cem_action_sequence` | optional | Best or mean action sequence from CEM at this planner invocation (shape/layout exporter-defined). |
| `cem_iterations` | int | Number of CEM iterations used (or effective sample count). |
| `cem_cost` | float | Scalar cost / negative return CEM optimized (definition fixed per exporter version). |
| `cem_seed` | int | RNG seed for reproducibility of CEM + WM rollouts for this episode or step. |
| `planner_metadata` | object (optional) | Free-form but JSON-serializable: population size, horizon, elite fraction, etc. |

---

## On-disk artifacts (under `SMOLVLA_JEPA_EXPORT_OUT`, default `SMOLVLA_JEPA_SOURCE`)

| Filename | Role |
|----------|------|
| **`trajectories.pt`** or shard payload files | Primary payload: serialized structure containing **paired** executed + CEM fields. Single-file mode may write `trajectories.pt`; memory-bounded mode may write shard files (for example `trajectories_shard_00000.pt`, `..._00001.pt`). Manifest must declare the concrete payload files (see shard contract below). |
| **`export_manifest.json`** | Sidecar describing export run and schema so `bridge_builder.py` can fail fast. |

### `export_manifest.json` — required keys

| Key | Type | Notes |
|-----|------|--------|
| `schema_version` | string | Must match paired schema (e.g. `cem_paired_push_v3_v0`). |
| `export_mode` | string | Must be **`cem_paired_push_v3`**. |
| `trajectories_file` | string | Primary payload root. Current contract uses `episodes` (directory of episode shards). |
| `trajectories_format` | string | Current contract: `pt_per_episode`. |
| `trajectories_glob` | string | Current contract: `episodes/episode_*.pt`. |
| `created_at` | string | ISO-8601 timestamp. |
| `task_id` | string | Expected task, e.g. `push-v3`. |
| `jepa_ckpt` | string | Path or id of JEPA-WM weights used (audit). |
| `pairing` | string | e.g. `executed_latent_aligned` — documents that real and latent arms share `pair_key` + `step_index`. |

**Recommended (optional) keys:** `git_sha`, `exporter_script`, `num_episodes`, `policy_checkpoint`, `cem_config` (horizon, population, etc.).

---

## Shard contract (memory-bounded export)

`--episodes-per-shard` controls flush cadence and is recorded in manifest metadata.

| Key | Type | Notes |
|-----|------|-------|
| `episodes_per_shard` | int | Requested flush cadence (default `1` in phase07 wiring). |
| `shard_count` | int | Number of completed payload files under `episodes/`. |
| `shard_files` | array[string] | Ordered list of payload files consumed by phase08. |
| `complete_episodes` | int | Number of episodes fully materialized; currently equals `shard_count` (`pt_per_episode` format). |

Bridge readers must treat `shard_files` as the source of truth when present, and preserve list order for deterministic replay and split assignment.

### Compact payload layout

Each payload file currently stores a single compact episode dict with paired records:

- Episode envelope: `pair_key`, `task_id`, `episode_index`
- Executed arm: aligned step records with observation refs/payload, executed `action`, optional `reward`/`success`/`done`
- Predicted arm: aligned `latent_pred`, optional `cem_action_sequence`, and CEM diagnostics (`cem_iterations`, `cem_cost`, `cem_seed`, optional `planner_metadata`)

Writers may choose inline tensors or sidecar refs, but field names and pairing semantics above remain stable for a given `schema_version`.

---

## Bridge contract

- **`bridge_builder.py` must read `export_manifest.json` and `trajectories_file`, validate `export_mode == cem_paired_push_v3` and `schema_version`, then assign each episode to `train/` vs `val/` by a deterministic function of `pair_key` (or `episode_index`) documented in `bridge_summary.json`, with default intent: `train/` = executed-heavy frames, `val/` = latent/CEM-heavy frames from the same paired export.**

---

## References

- Baseline eval entrypoint (local): `../run_baseline_eval.sh`.
- Producer contract source for this repository copy: `../jepa_cem_paired_pushv3_export.py`.
