#!/bin/bash
# Run constrained and CL-only Section 3.1 DAE-11 cases through separate 5x5
# matrices, each with its own logs, convergence copies, and study index.
#
# Paper operating point and constraints:
#   minimize Cd at M=0.03, Re=500,000, Cl=1.25
#   CMZ=0.053639, TE angle=6.25 deg, LE closure mismatch=0
#   t/c=0.128 at x/c=0.33 and t/c=0.014 at x/c=0.90
# The constrained case uses Deb feasibility-first ranking and CM-guided
# selection inside its CL envelope. The CL-only case interpolates at exact CL.

set -euo pipefail

STUDY_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$(cd "$STUDY_DIR/.." && pwd)"

export STUDY_SEED="${STUDY_SEED:-1}"
export OPTIMIZER_QUIET="${OPTIMIZER_QUIET:-1}"
export OPTIMIZER_PLOT="${OPTIMIZER_PLOT:-0}"

CASE_RUN_TAG="${CASE_RUN_TAG:-dae11_seed_${STUDY_SEED}}"
CASE_RESULTS_ROOT="${CASE_RESULTS_ROOT:-$STUDY_DIR/$CASE_RUN_TAG}"
read -r -a study_cases <<< "${CASE_STUDY_CASES:-constrained unconstrained}"
if [ "${#study_cases[@]}" -eq 0 ]; then
    echo "ERROR: CASE_STUDY_CASES must include constrained and/or unconstrained."
    exit 1
fi
for study_case in "${study_cases[@]}"; do
    case "$study_case" in
        constrained|unconstrained) ;;
        *)
            echo "ERROR: unknown study case: $study_case"
            exit 1
            ;;
    esac
done

CASE_AOA_MIN="${CASE_AOA_MIN:-2}"
CASE_AOA_MAX="${CASE_AOA_MAX:-10}"
CASE_AOA_N_POINTS="${CASE_AOA_N_POINTS:-30}"
CASE_MACH="${CASE_MACH:-0.2}"
CASE_RE="${CASE_RE:-500000}"
CASE_CL_TARGET="${CASE_CL_TARGET:-1.25}"
CASE_NUM_GEN="${CASE_NUM_GEN:-500}"
CASE_POP_SIZE="${CASE_POP_SIZE:-100}"
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
    echo "Download dae11.dat from:"
    echo "https://m-selig.web.engr.illinois.edu/ads/coord/dae11.dat"
    exit 1
fi
if [ -e "$CASE_RESULTS_ROOT" ] && [ "${CASE_ALLOW_EXISTING_RESULTS:-0}" != "1" ]; then
    echo "ERROR: results root already exists: $CASE_RESULTS_ROOT"
    echo "       Change CASE_RUN_TAG or set CASE_ALLOW_EXISTING_RESULTS=1."
    exit 1
fi

mkdir -p "$CASE_RESULTS_ROOT"

export AOA_N_POINTS="$CASE_AOA_N_POINTS"
export DAE11_DATA_FILE="$DAE11_PATH"
export PARAM_REFERENCE_AIRFOIL="dae11"
export DEFORMATION_BASELINE="dae11"
export INVERSE_SELECTED_METHODS="${INVERSE_SELECTED_METHODS:-$BASE_DIR/dae11_inverse/selected_methods.csv}"
unset DEFORMATION_TARGET_TE_UPPER DEFORMATION_TARGET_TE_LOWER
unset CST_RAE_MARGIN CST_A0_RAE_MARGIN
unset BSPLINE_RAE_MARGIN BSPLINE_RAE_LE_MARGIN BSPLINE_RAE_TE_MARGIN
unset BSPLINE_RAE_LE_WIDTH BSPLINE_RAE_TE_WIDTH
unset BSPLINE_RAE_LE_RADIUS_REF BSPLINE_RAE_LE_RADIUS_MIN
unset BSPLINE_RAE_LE_RADIUS_MAX BSPLINE_RAE_LE_SIGN_EPS
export INCLUDE_REFERENCE_IN_GEN0="1"

# FFD uses the structure selected by the DAE-11 inverse study. An explicitly
# supplied MATRIX_FFD_*_OVERRIDE still takes precedence in matrix_5x5.sh.
unset FFD_PIN_END_COLUMNS
export FFD_N_ROWS="${FFD_N_ROWS:-4}"
export FFD_M="${FFD_M:-4}"
export FFD_PIN_MODE="${FFD_PIN_MODE:-corners}"

# Keep the global geometry policy requested for the main optimizer: reject only
# intersections. The paper-specific equalities are handled by their own ranking.
unset GEOM_MIN_AREA GEOM_MIN_AREA_REFERENCE GEOM_MIN_AREA_RATIO
export DRELA_DAE11_ACTIVE_CONSTRAINTS="${DRELA_DAE11_ACTIVE_CONSTRAINTS:-te_angle,t90,t33,le_mismatch,cm}"
export DRELA_DAE11_CONSTRAINT_MODE="${DRELA_DAE11_CONSTRAINT_MODE:-deb}"
export DRELA_DAE11_CONSTRAINT_WEIGHT="${DRELA_DAE11_CONSTRAINT_WEIGHT:-10.0}"
export DRELA_DAE11_ADAPTIVE_PENALTY="${DRELA_DAE11_ADAPTIVE_PENALTY:-1}"
export DRELA_DAE11_PENALTY_WEIGHT_MIN="${DRELA_DAE11_PENALTY_WEIGHT_MIN:-0.01}"
export DRELA_DAE11_PENALTY_WEIGHT_MAX="${DRELA_DAE11_PENALTY_WEIGHT_MAX:-10.0}"
export DRELA_DAE11_PENALTY_SCHEDULE_POWER="${DRELA_DAE11_PENALTY_SCHEDULE_POWER:-1.0}"
export DRELA_DAE11_FEASIBILITY_TOL="${DRELA_DAE11_FEASIBILITY_TOL:-1.0}"

export DRELA_DAE11_CM_TARGET="${DRELA_DAE11_CM_TARGET:-0.053639}"
export DRELA_DAE11_CM_SCALE="${DRELA_DAE11_CM_SCALE:-0.02}"
export DRELA_DAE11_T33_TARGET="${DRELA_DAE11_T33_TARGET:-0.128}"
export DRELA_DAE11_T90_TARGET="${DRELA_DAE11_T90_TARGET:-0.014}"
export DRELA_DAE11_TE_ANGLE_TARGET="${DRELA_DAE11_TE_ANGLE_TARGET:-6.25}"
export DRELA_DAE11_LE_MISMATCH_TARGET="${DRELA_DAE11_LE_MISMATCH_TARGET:-0.0}"

# Use Kulfan's wind-tunnel ordinate tolerance for both prescribed thicknesses.
export DRELA_DAE11_T33_SCALE="${DRELA_DAE11_T33_SCALE:-0.0004}"
export DRELA_DAE11_T90_SCALE="${DRELA_DAE11_T90_SCALE:-0.0004}"
export DRELA_DAE11_TE_ANGLE_SCALE="${DRELA_DAE11_TE_ANGLE_SCALE:-1.0}"
export DRELA_DAE11_LE_MISMATCH_SCALE="${DRELA_DAE11_LE_MISMATCH_SCALE:-0.005}"

# The constrained case selects CM inside this CL envelope. The CL-only case
# uses exact-CL interpolation, which also disables CM-guided point selection.
CONSTRAINED_CL_TOLERANCE="${CL_TARGET_TOLERANCE:-0.02}"
CONSTRAINED_CM_TARGET="${CM_SELECTION_TARGET:-$DRELA_DAE11_CM_TARGET}"
CONSTRAINED_CM_TOLERANCE="${CM_SELECTION_TOLERANCE:-$DRELA_DAE11_CM_SCALE}"

echo "Drela DAE-11 constrained vs. CL-only 5x5 study"
echo "Results root         : $CASE_RESULTS_ROOT"
echo "Seed                 : $STUDY_SEED"
echo "Baseline             : $DAE11_PATH"
echo "Inverse selections   : $INVERSE_SELECTED_METHODS"
echo "AoA sweep            : $CASE_AOA_MIN to $CASE_AOA_MAX ($CASE_AOA_N_POINTS samples)"
echo "Mach / Reynolds      : $CASE_MACH / $CASE_RE"
echo "Target Cl            : $CASE_CL_TARGET"
echo "Generations / pop.   : $CASE_NUM_GEN / $CASE_POP_SIZE"

cd "$BASE_DIR"

matrix_args=(
    "MATRIX_METHODS=${CASE_MATRIX_METHODS:-parsec cst bspline hickshenne ffd}"
    "MATRIX_OPTIMIZERS=${CASE_MATRIX_OPTIMIZERS:-ga pso de cmaes abc}"
    "MATRIX_CASES=${CASE_MATRIX_CASES:-}"
)

for study_case in "${study_cases[@]}"; do
    RESULTS_DIR="$CASE_RESULTS_ROOT/$study_case"
    INPUT_FILE="$RESULTS_DIR/input_questions_1point.txt"
    mkdir -p "$RESULTS_DIR"
    printf '%s\n' \
        "$CASE_AOA_MIN" "$CASE_AOA_MAX" 1 "$CASE_MACH" "$CASE_RE" \
        "$CASE_CL_TARGET" "$CASE_NUM_GEN" "$CASE_POP_SIZE" y > "$INPUT_FILE"

    if [ "$study_case" = constrained ]; then
        export DRELA_DAE11_CONSTRAINTS=1
        export CL_TARGET_TOLERANCE="$CONSTRAINED_CL_TOLERANCE"
        export CM_SELECTION_TARGET="$CONSTRAINED_CM_TARGET"
        export CM_SELECTION_TOLERANCE="$CONSTRAINED_CM_TOLERANCE"
        echo "Constrained case: $RESULTS_DIR"
        echo "  DAE-11 constraints: $DRELA_DAE11_ACTIVE_CONSTRAINTS ($DRELA_DAE11_CONSTRAINT_MODE)"
        echo "  CL envelope: $CASE_CL_TARGET +/- $CL_TARGET_TOLERANCE"
        echo "  Scales: CM=$DRELA_DAE11_CM_SCALE t33=$DRELA_DAE11_T33_SCALE t90=$DRELA_DAE11_T90_SCALE TE=$DRELA_DAE11_TE_ANGLE_SCALE LE=$DRELA_DAE11_LE_MISMATCH_SCALE"
    else
        export DRELA_DAE11_CONSTRAINTS=0
        export CL_TARGET_TOLERANCE=0
        unset CM_SELECTION_TARGET CM_SELECTION_TOLERANCE
        echo "Unconstrained case: $RESULTS_DIR"
        echo "  Constraint: CL=$CASE_CL_TARGET (exact interpolation)"
    fi

    env "${matrix_args[@]}" \
        MATRIX_RESULTS_DIR="$RESULTS_DIR" \
        MATRIX_INPUT_FILE="$INPUT_FILE" \
        bash "$STUDY_DIR/matrix_5x5.sh"

    if [ "${CASE_PLOT_AFTER:-0}" = "1" ]; then
        MATRIX_RESULTS_DIR="$RESULTS_DIR" \
            python3 "$STUDY_DIR/plot_matrix_5x5.py"
    fi
done

echo "Study complete: $CASE_RESULTS_ROOT"
