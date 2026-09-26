#!/bin/bash
# studies/run_point_studies.sh
#
# Sequentially run separate 5x5 matrix studies for the abstract's 1-point,
# 2-point, and 4-point Cd objective setups.
#
# Run from optimizer/:
#   nohup bash studies/run_point_studies.sh > studies/point_studies.nohup.log 2>&1 &
#
# Main editable defaults can be overridden before launch:
#   POINT_NUM_GEN=500 \
#   POINT_MACHES_4="0.68 0.71 0.74 0.76" \
#   nohup bash studies/run_point_studies.sh > studies/point_studies.nohup.log 2>&1 &
#
# Each launch writes to a fresh timestamped root by default:
#   studies/point_studies_<timestamp>_seed<seed>/
#
# To choose the folder explicitly:
#   POINT_RUN_TAG="hp_v2_seed42" bash studies/run_point_studies.sh

set -euo pipefail

STUDY_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$(cd "$STUDY_DIR/.." && pwd)"

read -r -a POINT_STUDIES_ARR <<< "${POINT_STUDIES:-1 2 4}"

# Mach definitions from the abstract:
#   1-point: M = 0.74
#   2-point: M = 0.68, 0.74
#   4-point: M = 0.68, 0.71, 0.74, 0.76
POINT_MACHES_1="${POINT_MACHES_1:-0.74}"
POINT_MACHES_2="${POINT_MACHES_2:-0.68 0.74}"
POINT_MACHES_4="${POINT_MACHES_4:-0.68 0.71 0.74 0.76}"

POINT_AOA_MIN="${POINT_AOA_MIN:-0}"
POINT_AOA_MAX="${POINT_AOA_MAX:-3}"
POINT_RE="${POINT_RE:-2700000}"
POINT_CL_TARGET="${POINT_CL_TARGET:-0.733}"
POINT_NUM_GEN="${POINT_NUM_GEN:-500}"
POINT_POP_SIZE="${POINT_POP_SIZE:-100}"
POINT_RESULTS_PREFIX="${POINT_RESULTS_PREFIX:-matrix_5x5}"
POINT_PLOT_AFTER_EACH="${POINT_PLOT_AFTER_EACH:-1}"
export STUDY_SEED="${STUDY_SEED:-42}"

POINT_RUN_TAG="${POINT_RUN_TAG:-$(date +%Y%m%d_%H%M%S)_seed${STUDY_SEED}}"
POINT_RESULTS_ROOT="${POINT_RESULTS_ROOT:-$STUDY_DIR/point_studies_$POINT_RUN_TAG}"

if [ -e "$POINT_RESULTS_ROOT" ] && [ "${POINT_ALLOW_EXISTING_RESULTS:-0}" != "1" ]; then
    echo "ERROR: results root already exists: $POINT_RESULTS_ROOT"
    echo "       Pick a new POINT_RUN_TAG or set POINT_ALLOW_EXISTING_RESULTS=1 to reuse it."
    exit 1
fi

make_input_file() {
    local n_points="$1"
    local out_file="$2"
    local mach_string=""
    local -a maches=()

    case "$n_points" in
        1) mach_string="$POINT_MACHES_1" ;;
        2) mach_string="$POINT_MACHES_2" ;;
        4) mach_string="$POINT_MACHES_4" ;;
        *)
            echo "ERROR: no default Mach setup configured for ${n_points}-point studies"
            exit 1
            ;;
    esac

    read -r -a maches <<< "$mach_string"
    if [ "${#maches[@]}" -ne "$n_points" ]; then
        echo "ERROR: ${n_points}-point study needs exactly ${n_points} Mach values; got: $mach_string"
        exit 1
    fi

    {
        echo "$POINT_AOA_MIN"
        echo "$POINT_AOA_MAX"
        echo "$n_points"

        for ((i = 0; i < n_points; i++)); do
            echo "${maches[$i]}"
        done

        echo "$POINT_RE"
        echo "$POINT_CL_TARGET"
        echo "$POINT_NUM_GEN"
        echo "$POINT_POP_SIZE"

        echo "y"
    } > "$out_file"
}

echo "Point-study sweep"
echo "Base dir       : $BASE_DIR"
echo "Point studies  : ${POINT_STUDIES_ARR[*]}"
echo "Mach setup     :"
echo "  1-point      : $POINT_MACHES_1"
echo "  2-point      : $POINT_MACHES_2"
echo "  4-point      : $POINT_MACHES_4"
echo "Generations    : $POINT_NUM_GEN"
echo "Population     : $POINT_POP_SIZE"
echo "Study seed     : $STUDY_SEED"
echo "Results root   : $POINT_RESULTS_ROOT"

mkdir -p "$POINT_RESULTS_ROOT"
printf "%s\n" "$POINT_RESULTS_ROOT" > "$STUDY_DIR/point_studies_latest.txt"

cd "$BASE_DIR"

for n_points in "${POINT_STUDIES_ARR[@]}"; do
    if ! [[ "$n_points" =~ ^[0-9]+$ ]] || [ "$n_points" -lt 1 ]; then
        echo "ERROR: invalid point count '$n_points'"
        exit 1
    fi
    case "$n_points" in
        1|2|4) ;;
        *)
            echo "ERROR: this study runner is configured for the abstract cases only: 1, 2, and 4 points; got '$n_points'"
            exit 1
            ;;
    esac

    results_name="${POINT_RESULTS_PREFIX}_${n_points}point"
    results_dir="$POINT_RESULTS_ROOT/$results_name"
    input_file="$results_dir/input_questions_${n_points}point.txt"

    mkdir -p "$results_dir"
    make_input_file "$n_points" "$input_file"

    echo ""
    echo "============================================================"
    echo "Launching ${n_points}-point matrix study"
    echo "Results : $results_dir"
    echo "Input   : $input_file"
    echo "============================================================"

    MATRIX_RESULTS_NAME="$results_name" \
    MATRIX_RESULTS_DIR="$results_dir" \
    MATRIX_INPUT_FILE="$input_file" \
    bash "$STUDY_DIR/matrix_5x5.sh"

    if [ "$POINT_PLOT_AFTER_EACH" = "1" ]; then
        MATRIX_RESULTS_NAME="$results_name" \
        MATRIX_RESULTS_DIR="$results_dir" \
        python3 "$STUDY_DIR/plot_matrix_5x5.py"
    fi
done

echo ""
echo "All point studies complete."
