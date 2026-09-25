#!/bin/bash
set -e

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"

OPTIMIZER_PLOT="${OPTIMIZER_PLOT:-1}"
if [ "$OPTIMIZER_PLOT" != 0 ] && [ "$OPTIMIZER_PLOT" != 1 ]; then
    echo "ERROR: OPTIMIZER_PLOT must be 0 or 1; got '$OPTIMIZER_PLOT'" >&2
    exit 1
fi

RAIFOIL_BACKEND="${RAIFOIL_BACKEND:-cpu}"
case "$RAIFOIL_BACKEND" in
    cpu) RAIFOIL_BIN="${RAIFOIL_CPU_BIN:-$HOME/rAIFoil_CPU_bin}" ;;
    gpu) RAIFOIL_BIN="rAIFoil" ;;
    *)
        echo "ERROR: RAIFOIL_BACKEND must be cpu or gpu; got '$RAIFOIL_BACKEND'" >&2
        exit 1
        ;;
esac
if ! command -v "$RAIFOIL_BIN" >/dev/null 2>&1; then
    echo "ERROR: rAIFoil $RAIFOIL_BACKEND executable not found: $RAIFOIL_BIN" >&2
    exit 1
fi

is_quiet() {
    case "${OPTIMIZER_QUIET:-0}" in
        1|true|TRUE|yes|YES|y|Y) return 0 ;;
        *) return 1 ;;
    esac
}

status() {
    is_quiet || echo "$@"
}

run_quietly() {
    if is_quiet; then
        "$@" >/dev/null
    else
        "$@"
    fi
}

prompt_read() {
    local variable_name="$1"
    local prompt="$2"
    if is_quiet; then
        IFS= read -r "$variable_name"
    else
        IFS= read -r -p "$prompt" "$variable_name"
    fi
}

# User input
prompt_read AOA_MIN "Enter minimum AoA: "
prompt_read AOA_MAX "Enter maximum AoA: "
prompt_read N_CD_POINTS "Enter number of Cd objective points: "

if ! [[ "$N_CD_POINTS" =~ ^[0-9]+$ ]] || [ "$N_CD_POINTS" -lt 1 ]; then
    echo "ERROR: number of Cd objective points must be a positive integer; got '$N_CD_POINTS'"
    exit 1
fi
if [ "$N_CD_POINTS" -gt 4 ]; then
    echo "ERROR: this fitness setup supports at most 4 Cd objective points; got '$N_CD_POINTS'"
    exit 1
fi

MACH_VALUES=()
for i in $(seq 1 "$N_CD_POINTS"); do
    prompt_read mach_i "Enter Mach for Cd objective $i: "
    MACH_VALUES+=("$mach_i")
done

prompt_read RE "Enter Reynolds: "
prompt_read CL_TARGET "Enter target Cl: "
prompt_read NUM_GEN "Enter number of generations: "
prompt_read POP_SIZE "Enter population size: "

prompt_read CONFIRM "Proceed? (y/n): "

[[ "$CONFIRM" != "y" ]] && echo "Aborted." && exit 1

export AOA_MIN AOA_MAX RE CL_TARGET NUM_GEN POP_SIZE N_CD_POINTS
export AOA_N_POINTS="${AOA_N_POINTS:-10}"

if ! [[ "$AOA_N_POINTS" =~ ^[0-9]+$ ]] || [ "$AOA_N_POINTS" -lt 2 ]; then
    echo "ERROR: AOA_N_POINTS must be an integer >= 2; got '$AOA_N_POINTS'"
    exit 1
fi

for i in $(seq 1 "$N_CD_POINTS"); do
    idx=$((i - 1))
    export "MACH_CD${i}=${MACH_VALUES[$idx]}"
done

MACH_CD_LIST=$(IFS=,; echo "${MACH_VALUES[*]}")
export MACH_CD_LIST

# Backward-compatible names used by older helper code.
export MACH_CD="$MACH_CD1"

# Create optimization directory
cd "$BASE_DIR"
last_index=$(ls -d optimization_* 2>/dev/null | sed 's/optimization_//' | sort -n | tail -n 1)
next_index=$( [[ -z "$last_index" ]] && echo 1 || echo $((last_index + 1)) )

while true; do
    OPT_DIR="$BASE_DIR/optimization_${next_index}"
    if mkdir "$OPT_DIR" 2>/dev/null; then
        break
    fi
    if [ ! -d "$OPT_DIR" ]; then
        echo "ERROR: failed to create optimization directory: $OPT_DIR"
        exit 1
    fi
    next_index=$((next_index + 1))
done
export OPT_DIR

DESIGN_SPACE="$OPT_DIR/Design_space"
export DESIGN_SPACE

EVAL_DIRS=()
INPUT_CDS=()
for idx in "${!MACH_VALUES[@]}"; do
    mach="${MACH_VALUES[$idx]}"
    eval_dir="$OPT_DIR/evaluation_mach_$(printf "%.2f" "$mach")"
    EVAL_DIRS+=("$eval_dir")
    INPUT_CDS+=("$eval_dir/input")
done

mkdir -p "$DESIGN_SPACE" "${EVAL_DIRS[@]}"
status "Optimization folder: $OPT_DIR"
status "rAIFoil backend: $RAIFOIL_BACKEND ($RAIFOIL_BIN)"
status "Cd objective points: $N_CD_POINTS"
status "AoA samples per Cd point: $AOA_N_POINTS"
for idx in "${!MACH_VALUES[@]}"; do
    point=$((idx + 1))
    status "Cd point $point: Mach ${MACH_VALUES[$idx]} -> ${EVAL_DIRS[$idx]}"
done

write_cd_input() {
    local geom_dir="$1"
    local eval_dir="$2"
    local mach="$3"
    local input_path="$4"
    local aoa_step

    aoa_step=$(awk "BEGIN {print ($AOA_MAX - $AOA_MIN) / ($AOA_N_POINTS - 1)}")

    mkdir -p "$eval_dir"
    cat > "$input_path" << EOF
GEOMETRY_FOLDER = $geom_dir
AOA_MIN = $AOA_MIN
AOA_MAX = $AOA_MAX
AOA_STEP = $aoa_step
MACH_MIN = $mach
MACH_MAX = $mach
MACH_STEP = $mach
REYNOLDS_MIN = $RE
REYNOLDS_MAX = $RE
REYNOLDS_STEP = $RE
CI_LEVEL = 0.95
OUTPUT_FILE = DD_80
OUTPUT_FOLDER = $eval_dir
OUTPUT_FORMAT = csv
EOF
}

evaluate_cd_points() {
    local geom_dir="$1"

    for idx in "${!MACH_VALUES[@]}"; do
        write_cd_input \
            "$geom_dir" \
            "${EVAL_DIRS[$idx]}" \
            "${MACH_VALUES[$idx]}" \
            "${INPUT_CDS[$idx]}"

        run_quietly "$RAIFOIL_BIN" -i "${INPUT_CDS[$idx]}"
        run_quietly python3 run_concat.py "${EVAL_DIRS[$idx]}" "$CL_TARGET"
    done

    primary_eval_dir="${EVAL_DIRS[0]}"
    for ((idx = 1; idx < N_CD_POINTS; idx++)); do
        run_quietly python3 run_cd2.py \
            "$primary_eval_dir" \
            "${EVAL_DIRS[$idx]}" \
            "${MACH_VALUES[$idx]}" \
            "$CL_TARGET"
    done
}

status "Setup complete"

# Generation 0
run_quietly python3 run_gen0.py "$DESIGN_SPACE" "$POP_SIZE"
evaluate_cd_points "$DESIGN_SPACE"

# Generation loop
gen=1
while [ $gen -le $NUM_GEN ]; do
    # 1. Rank current population using the configured Cd points
    python3 run_rank.py "$gen"

    # 2. Optimizer step + decode phenotypes
    run_quietly python3 run_step.py
    GEN_DIR="$OPT_DIR/generation_$gen"

    # 3. Delete previous generation's .dat files, keeping the best pointer target
    BEST_AIRFOIL_FILE=$(cat "$OPT_DIR/best_airfoil.dat" | tr -d '[:space:]')
    BEST_AIRFOIL_NAME=$(basename "$BEST_AIRFOIL_FILE")

    if [ $gen -gt 1 ]; then
        PREV_GEN_DIR="$OPT_DIR/generation_$((gen - 1))"
        if [ -d "$PREV_GEN_DIR" ]; then
            for dat_file in "$PREV_GEN_DIR"/*.dat; do
                [ -f "$dat_file" ] || continue
                [ "$(basename "$dat_file")" != "$BEST_AIRFOIL_NAME" ] && rm -f "$dat_file"
            done
        fi
    fi

    # 4. Optionally plot the best airfoil; its .dat file is retained above.
    if [ "$OPTIMIZER_PLOT" = 1 ]; then
        run_quietly python3 run_plot.py "$BEST_AIRFOIL_FILE" "$gen"
    fi

    # 5. Evaluate this generation at every Cd Mach point
    evaluate_cd_points "$GEN_DIR"

    gen=$((gen + 1))
done

# Final ranking and optional plot
python3 run_rank.py $((NUM_GEN + 1))
BEST_AIRFOIL_FILE=$(cat "$OPT_DIR/best_airfoil.dat" | tr -d '[:space:]')
if [ "$OPTIMIZER_PLOT" = 1 ]; then
    run_quietly python3 run_plot.py "$BEST_AIRFOIL_FILE" $((NUM_GEN + 1))
fi

status "Optimization complete: $OPT_DIR"
