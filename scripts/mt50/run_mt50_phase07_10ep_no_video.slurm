#!/usr/bin/env bash
#SBATCH --job-name=smolvla-mt50-10ep
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=smolvla_mt50_phase07_10ep_%A_%a.out

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
PYTHON_BIN="${SMOLVLA_LEROBOT_ENV_DIR:-${WORKSPACE_ROOT}/.envs/lerobot_mw_py310}/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "error: python executable not found: ${PYTHON_BIN}" >&2
  exit 2
fi

MT50_ROOT="${PROJECT_ROOT}/scripts/mt50"
mkdir -p "${MT50_ROOT}"
DIFF_JSON="${MT50_ROOT}/mt50_phase07_task_difficulties.json"
if [[ ! -f "${DIFF_JSON}" ]]; then
  cat <<'JSON' > "${DIFF_JSON}"
{
  "default": "unclassified",
  "task_difficulties": {}
}
JSON
fi

export PROJECT_ROOT
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export MT50_TASK_DIFFICULTY_JSON="${DIFF_JSON}"
export MT50_PHASE07_OUTPUT_ROOT="${PROJECT_ROOT}/artifacts/MT50_Phase07_500_10ep"
export MT50_PHASE07_EPISODES=10
export MT50_PHASE07_SEED=1000
export MT50_PHASE07_MAX_STEPS=500
export MT50_PHASE07_VIDEO=false
export MT50_PHASE07_FPS=30
export MT50_PHASE07_OVERLAY_MODE=reward_delta
export MT50_PHASE07_SAVE_FRAMES=false
export MT50_PHASE07_SAVE_ACTIONS=false
export MT50_PHASE07_RESUME=false
export MT50_PHASE07_PURGE_PARTIAL=true
export MT50_PHASE07_CHECKPOINT=jadechoghari/smolvla_metaworld
export SMOLVLA_PYTHON_BIN="${PYTHON_BIN}"

echo "start phase07 10ep no-video run"
echo "output_root=${MT50_PHASE07_OUTPUT_ROOT}"
echo "episodes=${MT50_PHASE07_EPISODES}"
echo "seed=${MT50_PHASE07_SEED}"
echo "max_steps=${MT50_PHASE07_MAX_STEPS}"
echo "video=${MT50_PHASE07_VIDEO}"

"${PYTHON_BIN}" -m src.smolvla_pipeline.mt50_phase07_campaign
