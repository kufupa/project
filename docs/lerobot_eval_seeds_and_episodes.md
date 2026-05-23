# LeRobot eval: seeds vs episodes (MetaWorld / `lerobot-eval`)

This note describes how **`--seed`** and **`--eval.n_episodes`** interact in Hugging Face **LeRobot**’s evaluation script, and how to **find the source** in any environment (venv, conda, editable install, or git clone).

## Behaviour (what actually happens)

LeRobot’s evaluator passes a **`start_seed`** into `eval_policy()`. For each eval batch (indexed by `batch_ix`), it builds a Python `range` of environment seeds:

- **First batch:** seeds `start_seed + 0*num_envs` … `start_seed + 1*num_envs - 1`
- **Next batch:** continues with consecutive integers, step **`num_envs`** per batch.

So with the usual MetaWorld setup **`--eval.batch_size=1`** (one vectorised env → `num_envs == 1`) and:

```text
--seed=1000
--eval.n_episodes=10
```

you get **one episode per seed**, in order:

```text
1000, 1001, 1002, …, 1009
```

That is **not** “one seed for the whole run and ten random episodes”; it is **explicit per-episode env reset seeds** driven by that arithmetic.

If **`start_seed` is `None`**, LeRobot does **not** pass a manual seed list into `env.reset(seed=…)` for those batches (env default randomness applies).

### Multi-task (`--env.task` comma list)

`eval_policy_all` runs **each task** via the same pipeline. With a fixed CLI `--seed`, **each task** typically gets the **same** episode seed block (e.g. `1000..1009` for 10 episodes per task), not a single global sequence stretched across all tasks.

### Where this lives in code (names only)

- Episode seeding loop: `eval_policy()` in **`lerobot.scripts.lerobot_eval`**
- Env reset: `rollout()` in the same module calls `env.reset(seed=seeds)`
- MetaWorld adapter: **`lerobot.envs.metaworld`** (`MetaworldEnv.reset` forwards `seed` to the underlying env)

---

## How to find these files (general)

### 1) Installed package (pip / uv / conda into a venv)

From the **same interpreter** you use to run eval:

```bash
python -c "import lerobot.scripts.lerobot_eval as m; print(m.__file__)"
python -c "import lerobot.envs.metaworld as m; print(m.__file__)"
```

Optional: see install location and version:

```bash
python -c "import importlib.metadata as im; print(im.version('lerobot'))"
python -m pip show lerobot
```

Typical layout on disk:

```text
<site-packages>/lerobot/scripts/lerobot_eval.py
<site-packages>/lerobot/envs/metaworld.py
```

`<site-packages>` depends on OS and env (venv, conda prefix, user `--user` install, etc.).

### 2) Editable install (`pip install -e`)

If LeRobot was installed editable from a checkout, `__file__` from the snippet above points into **your clone** (not a copy under `site-packages`). Same commands apply.

### 3) Git clone (no install, or you browse GitHub)

Upstream project: **Hugging Face LeRobot** (`huggingface/lerobot` on GitHub). In a source tree the same modules are usually under:

```text
src/lerobot/scripts/lerobot_eval.py
src/lerobot/envs/metaworld.py
```

Exact top-level folder (`src/` vs `lerobot/`) can vary by branch; use repo search for `def eval_policy` or `start_seed`.

### 4) Repo-specific wrapper (this project)

Official MT50 Phase072 driver in **this** repo:

- `project/scripts/mt50/run_official_lerobot_mt50_eval.sh` — passes `--seed` and `--eval.n_episodes` to `lerobot_eval` (or the configurable rendering adapter).
- `project/scripts/mt50/lerobot_eval_configurable_rendering.py` — imports `lerobot.scripts.lerobot_eval` and calls `main()`; does **not** change seed semantics unless you change CLI/env.

---

## Quick sanity check from logs

LeRobot’s aggregated `eval_info.json` includes **`per_episode`** entries with a **`seed`** field when seeds were supplied. Comparing those values to `start_seed` and `n_episodes` confirms the incrementing pattern for your run configuration.
