#!/bin/bash
# Run the complete Chen-Fidkowski NACA0012 study sequentially for five seeds.
#
# Optional overrides:
#   CHEN_N_SEEDS=5
#   CHEN_SEEDS="1 2 3 4 5"       # use explicit seeds instead of random ones
#   CHEN_BATCH_TAG=my_batch       # controls the manifest/log basename
#   CASE_MATRIX_CASES="ffd:cmaes" # restrict each seed to selected cases

set -euo pipefail

STUDY_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$(cd "$STUDY_DIR/.." && pwd)"

N_SEEDS="${CHEN_N_SEEDS:-5}"
BATCH_TAG="${CHEN_BATCH_TAG:-chen_benchmark_random5}"
SEED_MANIFEST="$STUDY_DIR/${BATCH_TAG}_seeds.csv"

if ! [[ "$N_SEEDS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: CHEN_N_SEEDS must be a positive integer, got: $N_SEEDS"
    exit 1
fi

declare -a seeds=()
declare -A seen=()

if [ -n "${CHEN_SEEDS:-}" ]; then
    read -r -a seeds <<< "$CHEN_SEEDS"
    if [ "${#seeds[@]}" -ne "$N_SEEDS" ]; then
        echo "ERROR: CHEN_SEEDS supplied ${#seeds[@]} values; expected $N_SEEDS."
        exit 1
    fi
    for seed in "${seeds[@]}"; do
        if ! [[ "$seed" =~ ^[1-9][0-9]*$ ]]; then
            echo "ERROR: seeds must be positive integers, got: $seed"
            exit 1
        fi
        if [ -n "${seen[$seed]:-}" ]; then
            echo "ERROR: duplicate seed: $seed"
            exit 1
        fi
        seen[$seed]=1
    done
else
    while [ "${#seeds[@]}" -lt "$N_SEEDS" ]; do
        seed="$(od -An -N4 -tu4 /dev/urandom | awk '{print $1}')"
        seed=$((seed % 2147483646 + 1))
        run_tag="chen_benchmark_seed_${seed}"
        result_root="$STUDY_DIR/$run_tag"
        if [ -z "${seen[$seed]:-}" ] && [ ! -e "$result_root" ]; then
            seeds+=("$seed")
            seen[$seed]=1
        fi
    done
fi

if [ -e "$SEED_MANIFEST" ]; then
    echo "ERROR: seed manifest already exists: $SEED_MANIFEST"
    echo "       Change CHEN_BATCH_TAG before starting a new batch."
    exit 1
fi

{
    echo "seed,run_tag,results_root"
    for seed in "${seeds[@]}"; do
        run_tag="chen_benchmark_seed_${seed}"
        echo "$seed,$run_tag,$STUDY_DIR/$run_tag"
    done
} > "$SEED_MANIFEST"

echo "Chen-Fidkowski five-seed batch"
echo "Manifest: $SEED_MANIFEST"
echo "Seeds   : ${seeds[*]}"
echo "Runs are sequential."

cd "$BASE_DIR"

for position in "${!seeds[@]}"; do
    seed="${seeds[$position]}"
    run_number=$((position + 1))
    run_tag="chen_benchmark_seed_${seed}"
    result_root="$STUDY_DIR/$run_tag"

    echo
    echo "============================================================"
    echo "Chen study $run_number / ${#seeds[@]}: seed $seed"
    echo "Run tag: $run_tag"
    echo "============================================================"

    env \
        -u CASE_ALLOW_EXISTING_RESULTS \
        -u DAE11_DATA_FILE \
        -u DRELA_DAE11_CONSTRAINTS \
        -u DRELA_DAE11_CONSTRAINT_MODE \
        -u DRELA_DAE11_CONSTRAINT_WEIGHT \
        -u DRELA_DAE11_INCLUDE_LE_PENALTY \
        -u DRELA_DAE11_EPSILON_START \
        -u DRELA_DAE11_EPSILON_END \
        -u DRELA_DAE11_CM_TARGET \
        -u DRELA_DAE11_T33_TARGET \
        -u DRELA_DAE11_T90_TARGET \
        -u DRELA_DAE11_TE_ANGLE_TARGET \
        -u DRELA_DAE11_LE_MISMATCH_TARGET \
        -u DRELA_DAE11_CM_SCALE \
        -u DRELA_DAE11_T33_SCALE \
        -u DRELA_DAE11_T90_SCALE \
        -u DRELA_DAE11_TE_ANGLE_SCALE \
        -u DRELA_DAE11_LE_MISMATCH_SCALE \
        STUDY_SEED="$seed" \
        CASE_RUN_TAG="$run_tag" \
        CASE_RESULTS_ROOT="$result_root" \
        bash "$STUDY_DIR/run_chen_fidkowski_low_speed.sh"
done

echo
echo "All ${#seeds[@]} Chen-Fidkowski studies completed."
echo "Seed manifest: $SEED_MANIFEST"
