#!/usr/bin/env bash
# Run PIRL static tests under the CX3 Python 3.12 module stack.

set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-/rds/general/user/aa6622/home/project}"
COMMON="${PROJECT_ROOT}/scripts/pirl/pirl_maniskill_common.sh"
# shellcheck source=pirl_maniskill_common.sh
source "${COMMON}"

pirl_setup_modules

export PIRL_TEST_VENV="${PIRL_TEST_VENV:-${PROJECT_ROOT}/.venvs/pirl-py312}"
if [[ ! -x "${PIRL_TEST_VENV}/bin/python" ]]; then
  python3 -m venv "${PIRL_TEST_VENV}"
fi

export PIRL_PYTHON="${PIRL_TEST_VENV}/bin/python"
pirl_assert_python312

if ! "${PIRL_PYTHON}" - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("pytest") and importlib.util.find_spec("yaml") else 1)
PY
then
  "${PIRL_PYTHON}" -m pip install --upgrade pip
  "${PIRL_PYTHON}" -m pip install pytest pyyaml
fi

cd "${PROJECT_ROOT}"
"${PIRL_PYTHON}" -m pytest tests/test_pirl_maniskill_pbs_static.py -q
cd "${PROJECT_ROOT}/RLinf"
"${PIRL_PYTHON}" -m pytest tests/unit_tests/test_pirl_maniskill_config.py -q

echo "PIRL_PYTHON312_TESTS_DONE python=${PIRL_PYTHON}"
