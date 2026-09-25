#!/bin/bash
# Run the Chen-Fidkowski (2017) low-speed turbulent operating condition
# through this repository's 5x5 stochastic optimization matrix.
#
# Reproduced paper conditions:
#   NACA0012 baseline, M=0.15, Re=1e6, target Cl=0.6, and drag minimization
#   at fixed lift. Parameterization setups come from the NACA0012 inverse study.
#
# Framework differences:
#   rAIFoil replaces the paper's adaptive DG RANS solver and this repository's
#   stochastic optimizers replace SLSQP. The paper treats incidence as a design
#   variable starting at 6 deg; here it is eliminated as a trim variable by
#   interpolating each geometry's AoA sweep at Cl=0.6. The paper does not report
#   Amin, so the default assumption here is
#   A >= 1.0 * area(NACA0012), overridable below.

set -euo pipefail

STUDY_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$(cd "$STUDY_DIR/.." && pwd)"

export STUDY_SEED="${STUDY_SEED:-1}"

CASE_RUN_TAG="${CASE_RUN_TAG:-chen_fidkowski_low_speed_seed${STUDY_SEED}}"
CASE_RESULTS_ROOT="${CASE_RESULTS_ROOT:-$STUDY_DIR/point_studies_$CASE_RUN_TAG}"
RESULTS_DIR="$CASE_RESULTS_ROOT/matrix_5x5_1point"
INPUT_FILE="$RESULTS_DIR/input_questions_1point.txt"

CASE_AOA_MIN="${CASE_AOA_MIN:-0}"
CASE_AOA_MAX="${CASE_AOA_MAX:-8}"
CASE_AOA_N_POINTS="${CASE_AOA_N_POINTS:-33}"
CASE_MACH="${CASE_MACH:-0.15}"
CASE_RE="${CASE_RE:-1000000}"
CASE_CL_TARGET="${CASE_CL_TARGET:-0.6}"
CASE_NUM_GEN="${CASE_NUM_GEN:-500}"
CASE_POP_SIZE="${CASE_POP_SIZE:-100}"
CASE_MIN_AREA_RATIO="${CASE_MIN_AREA_RATIO:-1.0}"
NACA0012_PATH="${NACA0012_DATA_FILE:-/home/erene/simulations/rAIFoil/NACA0012.dat}"

if [ ! -f "$NACA0012_PATH" ]; then
    echo "ERROR: missing NACA0012 baseline: $NACA0012_PATH"
    exit 1
fi
if [ -e "$CASE_RESULTS_ROOT" ] && [ "${CASE_ALLOW_EXISTING_RESULTS:-0}" != "1" ]; then
    echo "ERROR: results root already exists: $CASE_RESULTS_ROOT"
    echo "       Change CASE_RUN_TAG or set CASE_ALLOW_EXISTING_RESULTS=1."
    exit 1
fi

mkdir -p "$RESULTS_DIR"
{
    echo "$CASE_AOA_MIN"
    echo "$CASE_AOA_MAX"
    echo "1"
    echo "$CASE_MACH"
    echo "$CASE_RE"
    echo "$CASE_CL_TARGET"
    echo "$CASE_NUM_GEN"
    echo "$CASE_POP_SIZE"
    echo "y"
} > "$INPUT_FILE"

export AOA_N_POINTS="$CASE_AOA_N_POINTS"
export NACA0012_DATA_FILE="$NACA0012_PATH"
export PARAM_REFERENCE_AIRFOIL="naca0012"
export DEFORMATION_BASELINE="naca0012"
unset DEFORMATION_TARGET_TE_UPPER DEFORMATION_TARGET_TE_LOWER
export INCLUDE_REFERENCE_IN_GEN0="1"

# Optional paper area constraint. The numerical Amin is omitted in the paper;
# 1.0 preserves at least the baseline enclosed area and can be overridden.
export GEOM_MIN_AREA_REFERENCE="$NACA0012_PATH"
export GEOM_MIN_AREA_RATIO="$CASE_MIN_AREA_RATIO"

echo "Chen-Fidkowski low-speed turbulent analogue"
echo "Results root         : $CASE_RESULTS_ROOT"
echo "Seed                 : $STUDY_SEED"
echo "Baseline             : $NACA0012_PATH"
echo "AoA sweep            : $CASE_AOA_MIN to $CASE_AOA_MAX ($CASE_AOA_N_POINTS samples)"
echo "Paper initial AoA    : 6 deg"
echo "Incidence treatment  : solved per geometry by target-Cl interpolation"
echo "Mach / Reynolds      : $CASE_MACH / $CASE_RE"
echo "Target Cl            : $CASE_CL_TARGET"
echo "Minimum area ratio   : $CASE_MIN_AREA_RATIO (assumption; paper omits Amin)"
echo "Generations / pop.   : $CASE_NUM_GEN / $CASE_POP_SIZE"
echo "Method setups        : naca0012_inverse/selected_methods.csv"

cd "$BASE_DIR"

matrix_args=()
if [ -n "${CASE_MATRIX_CASES:-}" ]; then
    matrix_args+=("MATRIX_CASES=$CASE_MATRIX_CASES")
fi

env "${matrix_args[@]}" \
    MATRIX_RESULTS_DIR="$RESULTS_DIR" \
    MATRIX_INPUT_FILE="$INPUT_FILE" \
    bash "$STUDY_DIR/matrix_5x5.sh"

if [ "${CASE_PLOT_AFTER:-0}" = "1" ]; then
    MATRIX_RESULTS_DIR="$RESULTS_DIR" \
        python3 "$STUDY_DIR/plot_matrix_5x5.py"
fi

echo "Study complete: $CASE_RESULTS_ROOT"
