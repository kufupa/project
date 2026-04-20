# Meta-World MT10 — sim install + verify

Sidecar bundle: pins, verifier script, optional Slurm. Does not modify SmolVLA / segment GRPO code.

## Policy

- **Venv:** Reuse existing LeRobot env (`SMOLVLA_LEROBOT_ENV_DIR`). New venv only if you need isolation or `pip check` shows unresolvable conflicts.
- **Audit installs:** Before mutating shared venv:

  ```bash
  pip freeze > /tmp/freeze.before.txt
  pip freeze | rg -i 'metaworld|gymnasium|mujoco|imageio|scipy' || true
  ```

  After `pip install -r …/requirements-sim.txt`:

  ```bash
  pip freeze > /tmp/freeze.after.txt
  diff -u /tmp/freeze.before.txt /tmp/freeze.after.txt | head -200
  pip check
  ```

  `verify_env.py` prints runtime versions on stdout.

## Setup + run

```bash
export MUJOCO_GL="${MUJOCO_GL:-egl}"
cd /path/to/aa6622/repo   # or: cd "$(git rev-parse --show-toplevel)"
REPO_ROOT="$(git rev-parse --show-toplevel)"
source "${SMOLVLA_LEROBOT_ENV_DIR:?set SMOLVLA_LEROBOT_ENV_DIR}/bin/activate"
pip install -r "${REPO_ROOT}/project/mt10/requirements-sim.txt"
python "${REPO_ROOT}/project/mt10/verify_env.py"
```

Expect last line: `mt10_ok`.

If EGL fails on a headless node, try `export MUJOCO_GL=osmesa` (site-dependent).

## Tests

From repo root (with same venv activated):

```bash
pytest project/tests/mt10/test_vec_env_smoke.py -v
```

Skips when `metaworld` is not installed.

## Optional Slurm

```bash
sbatch project/mt10/sbatch_verify_cpu.sh
```

Edit `#SBATCH` lines in that script for your cluster.
