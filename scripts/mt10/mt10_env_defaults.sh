#!/usr/bin/env bash
# Reproducibility defaults for MT10 paper runs. Source from phase6/8/9/preflight after mt10_tasks.sh.
# Override any variable in the environment before sourcing this file.
#
# Thread caps (OMP/MKL/OpenBLAS/NumExpr) are not set here—use library defaults or export before sourcing.

export METAWORLD_STRICT_CTOR="${METAWORLD_STRICT_CTOR:-1}"
