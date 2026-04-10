#!/bin/bash
#SBATCH --job-name=smolvla-preflight
#SBATCH --array=0-0
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --output=smolvla_preflight_%A_%a.out

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

bash "${PROJECT_ROOT}/scripts/smolvla/run_pushv3_smolvla_preflight_1ep.sh"
