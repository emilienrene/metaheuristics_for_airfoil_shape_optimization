#!/bin/bash
# studies/plot_point_studies.sh
#
# Replot all point-study matrix result folders.
#
# Run from optimizer/:
#   bash studies/plot_point_studies.sh
#
# By default this replots the latest timestamped point-study root written by
# run_point_studies.sh. To plot a specific run:
#   POINT_RESULTS_ROOT="studies/point_studies_hp_v2_seed42" bash studies/plot_point_studies.sh

set -euo pipefail

STUDY_DIR="$(cd "$(dirname "$0")" && pwd)"

read -r -a POINT_STUDIES_ARR <<< "${POINT_STUDIES:-1 2 4}"
POINT_RESULTS_PREFIX="${POINT_RESULTS_PREFIX:-matrix_5x5}"
POINT_RESULTS_ROOT="${POINT_RESULTS_ROOT:-}"

if [ -z "$POINT_RESULTS_ROOT" ]; then
    if [ -f "$STUDY_DIR/point_studies_latest.txt" ]; then
        POINT_RESULTS_ROOT="$(cat "$STUDY_DIR/point_studies_latest.txt")"
    else
        POINT_RESULTS_ROOT="$STUDY_DIR"
    fi
fi

echo "Point-study results root: $POINT_RESULTS_ROOT"

for n_points in "${POINT_STUDIES_ARR[@]}"; do
    results_name="${POINT_RESULTS_PREFIX}_${n_points}point"
    results_dir="$POINT_RESULTS_ROOT/$results_name"

    if [ ! -f "$results_dir/study_index.csv" ] && [ -f "$STUDY_DIR/$results_name/study_index.csv" ]; then
        results_dir="$STUDY_DIR/$results_name"
    fi

    if [ ! -f "$results_dir/study_index.csv" ]; then
        echo "Skipping ${n_points}-point study; missing $results_dir/study_index.csv"
        continue
    fi

    echo "Plotting ${n_points}-point study -> $results_dir"
    MATRIX_RESULTS_NAME="$results_name" \
    MATRIX_RESULTS_DIR="$results_dir" \
    python3 "$STUDY_DIR/plot_matrix_5x5.py"
done

echo "Point-study plots complete."
