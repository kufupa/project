# scripts/pbs/ — agent-owned PBS mirror

PBS submission scripts live here. The user owns Slurm under scripts/<area>/; this tree mirrors them for PBS-only sites (CX3, OpenPBS).

Conventions:
- Bash payload is NEVER duplicated. Each .pbs `exec`s the shared scripts/<area>/run_*.sh; only #PBS directives + site env differ.
- qsub from repo root: `cd /path/to/project && qsub scripts/pbs/<area>/foo.pbs` so PBS_O_WORKDIR is the repo.
- PROJECT_ROOT resolves from PBS_O_WORKDIR; fallback `${SCRIPT_DIR}/../../..` because files sit at scripts/pbs/<area>/.
- Site env (CX3): module load tools/prod + Python/3.12.3-GCCcore-13.3.0; MUJOCO_GL=glfw; SMOLVLA_LEROBOT_ENV_DIR pointing to lerobot_mw_py312.
- Helpers (submit_*_pbs.sh) live next to the .pbs files they fan out to and create their log dir before qsub.
