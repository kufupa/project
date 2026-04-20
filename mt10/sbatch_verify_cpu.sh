#!/usr/bin/bash
# One-node CPU smoke: MT10 verify script. Edit #SBATCH lines for your site.
#SBATCH --job-name=mt10_verify
#SBATCH --output=mt10_verify_%j.out
#SBATCH --error=mt10_verify_%j.err
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --partition=REPLACE_ME

set -euo pipefail

: "${SMOLVLA_LEROBOT_ENV_DIR:?export SMOLVLA_LEROBOT_ENV_DIR to your LeRobot venv}"

REPO_ROOT="$(git rev-parse --show-toplevel)"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

cd "${REPO_ROOT}"
# shellcheck source=/dev/null
source "${SMOLVLA_LEROBOT_ENV_DIR}/bin/activate"
exec python "${REPO_ROOT}/project/mt10/verify_env.py"
