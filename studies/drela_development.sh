#!/bin/bash
# Temporary foreground runner for developing the Drela DAE-11 case.
#
# This calls run_optimizer.sh directly. It creates only the normal
# optimization_i directory required by the optimizer: no study directory,
# copied results, nohup process, or redirected log is created.

set -euo pipefail

STUDY_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$(cd "$STUDY_DIR/.." && pwd)"
cd "$BASE_DIR"

# ---------------------------------------------------------------------------
# Development configuration: edit these values between experiments.
# ---------------------------------------------------------------------------
export PARAM_METHOD="${PARAM_METHOD:-ffd}"
export OPTIMIZER="${OPTIMIZER:-cmaes}"
export STUDY_SEED="${STUDY_SEED:-1}"
export OPTIMIZER_QUIET="${OPTIMIZER_QUIET:-1}"

# Development run currently exercises feasibility-first constraint handling.
export DRELA_DAE11_CONSTRAINTS="${DRELA_DAE11_CONSTRAINTS:-1}"

# Cumulative examples:
#   t33
#   t33,t90
#   t33,t90,te_angle
#   t33,t90,te_angle,cm
#   t33,t90,te_angle,cm,le_mismatch
export DRELA_DAE11_ACTIVE_CONSTRAINTS="${DRELA_DAE11_ACTIVE_CONSTRAINTS:-t33}"
export DRELA_DAE11_CONSTRAINT_MODE="${DRELA_DAE11_CONSTRAINT_MODE:-deb}"
export DRELA_DAE11_CONSTRAINT_WEIGHT="${DRELA_DAE11_CONSTRAINT_WEIGHT:-10.0}"

# Development-only adaptive penalty schedule. An individual is constraint
# feasible when every active normalized violation is <= FEASIBILITY_TOL.
#
#   weight = min + (max - min) * feasible_fraction^power
#
# With population 100 and the defaults below, 0 feasible members gives 0.01,
# 50 feasible gives 5.005, and 100 feasible gives 10.0.
export DRELA_DAE11_ADAPTIVE_PENALTY="${DRELA_DAE11_ADAPTIVE_PENALTY:-1}"
export DRELA_DAE11_PENALTY_WEIGHT_MIN="${DRELA_DAE11_PENALTY_WEIGHT_MIN:-0.01}"
export DRELA_DAE11_PENALTY_WEIGHT_MAX="${DRELA_DAE11_PENALTY_WEIGHT_MAX:-10.0}"
export DRELA_DAE11_PENALTY_SCHEDULE_POWER="${DRELA_DAE11_PENALTY_SCHEDULE_POWER:-1.0}"
export DRELA_DAE11_FEASIBILITY_TOL="${DRELA_DAE11_FEASIBILITY_TOL:-1.0}"

AOA_MIN="${CASE_AOA_MIN:--2}"
AOA_MAX="${CASE_AOA_MAX:-10}"
AOA_N_POINTS="${CASE_AOA_N_POINTS:-49}"
MACH="${CASE_MACH:-0.03}"
REYNOLDS="${CASE_RE:-500000}"
CL_TARGET="${CASE_CL_TARGET:-1.25}"
NUM_GENERATIONS="${CASE_NUM_GEN:-500}"
POPULATION_SIZE="${CASE_POP_SIZE:-100}"

# The selected FFD convention retains leading-edge radius authority.
unset FFD_PIN_END_COLUMNS
export FFD_N_ROWS="${FFD_N_ROWS:-4}"
export FFD_M="${FFD_M:-4}"
export FFD_PIN_MODE="${FFD_PIN_MODE:-corners}"

# Keep the established CMA-ES configuration explicit for repeatability.
export CMAES_SIGMA="${CMAES_SIGMA:-0.30}"
export CMAES_MU="${CMAES_MU:-0}"
export CMAES_CM="${CMAES_CM:-1.0}"
export CMAES_ELITISM="${CMAES_ELITISM:-0}"
export CMAES_MAX_RESAMPLE="${CMAES_MAX_RESAMPLE:-100}"

if [ -n "${DAE11_DATA_FILE:-}" ]; then
    DAE11_PATH="$DAE11_DATA_FILE"
elif [ -f "/home/erene/simulations/rAIFoil/DAE11.dat" ]; then
    DAE11_PATH="/home/erene/simulations/rAIFoil/DAE11.dat"
elif [ -f "/home/erene/simulations/rAIFoil/dae11.dat" ]; then
    DAE11_PATH="/home/erene/simulations/rAIFoil/dae11.dat"
elif [ -f "$BASE_DIR/DAE11.dat" ]; then
    DAE11_PATH="$BASE_DIR/DAE11.dat"
else
    DAE11_PATH="$BASE_DIR/parameterizations/data/dae11.dat"
fi

if [ ! -f "$DAE11_PATH" ]; then
    echo "ERROR: missing DAE-11 baseline: $DAE11_PATH"
    exit 1
fi

export AOA_N_POINTS
export DAE11_DATA_FILE="$DAE11_PATH"
export PARAM_REFERENCE_AIRFOIL="dae11"
export DEFORMATION_BASELINE="dae11"
export INCLUDE_REFERENCE_IN_GEN0="1"
unset DEFORMATION_TARGET_TE_UPPER DEFORMATION_TARGET_TE_LOWER

# Use the same DAE-11 inverse-study setup and constructive reference vector as
# the production matrix. Hicks-Henne and FFD retain zero deformation around
# DAE-11; only their selected structural hyperparameters are imported.
INVERSE_SELECTED_METHODS="${INVERSE_SELECTED_METHODS:-$BASE_DIR/dae11_inverse/selected_methods.csv}"
unset PARSEC_REFERENCE_PARAMS CST_REFERENCE_PARAMS BSPLINE_REFERENCE_PARAMS
unset CST_RAE_MARGIN CST_A0_RAE_MARGIN
unset BSPLINE_RAE_MARGIN BSPLINE_RAE_LE_MARGIN BSPLINE_RAE_TE_MARGIN
unset BSPLINE_RAE_LE_WIDTH BSPLINE_RAE_TE_WIDTH
unset BSPLINE_RAE_LE_RADIUS_REF BSPLINE_RAE_LE_RADIUS_MIN
unset BSPLINE_RAE_LE_RADIUS_MAX BSPLINE_RAE_LE_SIGN_EPS

if [ -f "$INVERSE_SELECTED_METHODS" ]; then
    inverse_exports=$(python3 - "$INVERSE_SELECTED_METHODS" "$PARAM_METHOD" << 'PYEOF'
import csv
import json
import shlex
import sys

path, method = sys.argv[1:3]
for row in csv.DictReader(open(path, newline="")):
    if row.get("method") != method:
        continue

    hyperparams = json.loads(row.get("hyperparams") or "{}")
    params = json.loads(row.get("params_json") or "null")
    passes = str(row.get("passes_kulfan", "")).strip().lower() in {
        "1", "true", "yes", "y",
    }
    for key, value in hyperparams.items():
        print(f"export {key}={shlex.quote(str(value))}")

    reference_variables = {
        "parsec": "PARSEC_REFERENCE_PARAMS",
        "cst": "CST_REFERENCE_PARAMS",
        "bspline": "BSPLINE_REFERENCE_PARAMS",
    }
    reference_variable = reference_variables.get(method)
    if reference_variable and isinstance(params, list) and passes:
        encoded = json.dumps(params, separators=(",", ":"))
        print(f"export {reference_variable}={shlex.quote(encoded)}")
    elif reference_variable and isinstance(params, list):
        print(
            f"WARNING: selected {method} inverse fit failed Kulfan tolerance; "
            "using its built-in DAE-11 reference.",
            file=sys.stderr,
        )
    break
PYEOF
    )
    eval "$inverse_exports"
else
    echo "WARNING: missing inverse-study selections: $INVERSE_SELECTED_METHODS"
fi

# Retain only the generic non-intersection geometry filter.
unset GEOM_MIN_AREA GEOM_MIN_AREA_REFERENCE GEOM_MIN_AREA_RATIO

export DRELA_DAE11_CM_TARGET="${DRELA_DAE11_CM_TARGET:-0.053639}"
export DRELA_DAE11_CM_SCALE="${DRELA_DAE11_CM_SCALE:-0.016}"
export DRELA_DAE11_T33_TARGET="${DRELA_DAE11_T33_TARGET:-0.128}"
export DRELA_DAE11_T90_TARGET="${DRELA_DAE11_T90_TARGET:-0.014}"
export DRELA_DAE11_TE_ANGLE_TARGET="${DRELA_DAE11_TE_ANGLE_TARGET:-6.25}"
export DRELA_DAE11_TE_ANGLE_SCALE="${DRELA_DAE11_TE_ANGLE_SCALE:-0.25}"
export DRELA_DAE11_LE_MISMATCH_TARGET="${DRELA_DAE11_LE_MISMATCH_TARGET:-0.0}"
export DRELA_DAE11_T33_SCALE="${DRELA_DAE11_T33_SCALE:-0.0004}"
export DRELA_DAE11_T90_SCALE="${DRELA_DAE11_T90_SCALE:-0.0004}"
export DRELA_DAE11_LE_MISMATCH_SCALE="${DRELA_DAE11_LE_MISMATCH_SCALE:-0.003}"

# Select the operating point closest to the CM target while keeping CL inside
# the permitted envelope. Set CL_TARGET_TOLERANCE=0 to recover exact-CL mode.
export CL_TARGET_TOLERANCE="${CL_TARGET_TOLERANCE:-0.01}"
export CM_SELECTION_TARGET="${CM_SELECTION_TARGET:-$DRELA_DAE11_CM_TARGET}"
export CM_SELECTION_TOLERANCE="${CM_SELECTION_TOLERANCE:-$DRELA_DAE11_CM_SCALE}"

bash "${OPTIMIZER_RUN_SCRIPT:-run_optimizer.sh}" <<EOF
$AOA_MIN
$AOA_MAX
1
$MACH
$REYNOLDS
$CL_TARGET
$NUM_GENERATIONS
$POPULATION_SIZE
y
EOF
