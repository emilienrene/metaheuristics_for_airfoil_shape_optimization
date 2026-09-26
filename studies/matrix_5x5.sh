#!/bin/bash
# studies/matrix_5x5.sh
#
# Runs the first 5x5 matrix study:
#   5 parameterizations x 5 optimizers
#
# Run from optimizer/:
#   bash studies/matrix_5x5.sh
#
# Targeted reruns can be selected with MATRIX_CASES:
#   MATRIX_UPDATE_INDEX=1 \
#   MATRIX_CASES="parsec:pso cst:pso bspline:pso hickshenne:pso ffd:pso ffd:cmaes ffd:abc" \
#   bash studies/matrix_5x5.sh
#
# The script reads NACA0012 inverse-study method-complexity choices from:
#   naca0012_inverse/selected_methods.csv
# Override with INVERSE_SELECTED_METHODS to reproduce another inverse study.
#
# Each run uses the current run_optimizer.sh and feeds MATRIX_INPUT_FILE
# if set, otherwise:
#   input_questions.txt

set -u

STUDY_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$(cd "$STUDY_DIR/.." && pwd)"
RESULTS_NAME="${MATRIX_RESULTS_NAME:-matrix_5x5}"
RESULTS_DIR="${MATRIX_RESULTS_DIR:-$STUDY_DIR/$RESULTS_NAME}"
SELECTED_METHODS="${INVERSE_SELECTED_METHODS:-$BASE_DIR/naca0012_inverse/selected_methods.csv}"
INPUT_FILE="${MATRIX_INPUT_FILE:-$BASE_DIR/input_questions.txt}"
INDEX="$RESULTS_DIR/study_index.csv"
INDEX_HEADER="run_index,method,optimizer,setup,n_genes,opt_dir,status,final_best,convergence_csv,log_file"

case "$SELECTED_METHODS" in
    /*) ;;
    *) SELECTED_METHODS="$BASE_DIR/$SELECTED_METHODS" ;;
esac

read -r -a METHODS <<< "${MATRIX_METHODS:-parsec cst bspline hickshenne ffd}"
read -r -a OPTIMIZERS <<< "${MATRIX_OPTIMIZERS:-ga pso de cmaes abc}"
read -r -a CASES <<< "${MATRIX_CASES:-}"

export STUDY_SEED="${STUDY_SEED:-42}"

# Canonical/near-canonical settings for the matrix study. Values can still be
# overridden by exporting them before launching this script.
export CMAES_SIGMA="${CMAES_SIGMA:-0.30}"
export CMAES_MU="${CMAES_MU:-0}"
export CMAES_CM="${CMAES_CM:-1.0}"
export CMAES_ELITISM="${CMAES_ELITISM:-0}"
export CMAES_MAX_RESAMPLE="${CMAES_MAX_RESAMPLE:-100}"

export ABC_MR="${ABC_MR:-0.0}"
export ABC_EMPLOYED_FRACTION="${ABC_EMPLOYED_FRACTION:-0.5}"
export ABC_ELITISM="${ABC_ELITISM:-1}"
export ABC_SCOUT_PERIOD="${ABC_SCOUT_PERIOD:-0}"
export ABC_SCOUT_COUNT="${ABC_SCOUT_COUNT:-1}"

mkdir -p "$RESULTS_DIR/logs" "$RESULTS_DIR/convergence"

if [ ! -f "$INPUT_FILE" ]; then
    echo "ERROR: missing input file: $INPUT_FILE"
    exit 1
fi

if [ ! -f "$SELECTED_METHODS" ]; then
    echo "WARNING: missing inverse-study selections: $SELECTED_METHODS"
    echo "         Runs will use each parameterization's default hyperparameters."
fi

if [ "${MATRIX_UPDATE_INDEX:-0}" = "1" ] && [ -f "$INDEX" ]; then
    :
else
    echo "$INDEX_HEADER" > "$INDEX"
fi

latest_opt_dir() {
    ls -d "$BASE_DIR"/optimization_* 2>/dev/null \
        | sort -t_ -k2 -n \
        | tail -n 1
}

clear_method_env() {
    unset PARSEC_REFERENCE_PARAMS
    unset CST_REFERENCE_PARAMS
    unset BSPLINE_REFERENCE_PARAMS
    unset CST_ORDER
    unset BSPLINE_N_CTRL
    unset HH_N_BUMPS
    unset FFD_N_ROWS
    unset FFD_M
    unset FFD_PIN_MODE
    unset FFD_PIN_END_COLUMNS
}

apply_selected_method_setup() {
    local method="$1"
    METHOD_SETUP="default"
    N_GENES=""

    clear_method_env

    if [ ! -f "$SELECTED_METHODS" ]; then
        return
    fi

    local exports
    exports=$(python3 - "$SELECTED_METHODS" "$method" << 'PYEOF'
import csv
import json
import shlex
import sys

path, method = sys.argv[1], sys.argv[2]

for row in csv.DictReader(open(path, newline="")):
    if row.get("method") != method:
        continue

    setup = row.get("setup", "selected")
    n_params = row.get("n_params", "")
    hyperparams = json.loads(row.get("hyperparams") or "{}")
    params = json.loads(row.get("params_json") or "null")
    passes_kulfan = str(row.get("passes_kulfan", "")).strip().lower() in {
        "1", "true", "yes", "y",
    }

    print(f"METHOD_SETUP={shlex.quote(setup)}")
    print(f"N_GENES={shlex.quote(str(n_params))}")
    reference_variables = {
        "parsec": "PARSEC_REFERENCE_PARAMS",
        "cst": "CST_REFERENCE_PARAMS",
        "bspline": "BSPLINE_REFERENCE_PARAMS",
    }
    reference_variable = reference_variables.get(method)
    if reference_variable and isinstance(params, list) and passes_kulfan:
        print(
            f"export {reference_variable}="
            + shlex.quote(json.dumps(params, separators=(",", ":")))
        )
    elif reference_variable and isinstance(params, list):
        print(
            f"WARNING: selected {method} inverse fit did not pass Kulfan "
            "tolerance; using the built-in reference instead.",
            file=sys.stderr,
        )
    for key, value in hyperparams.items():
        print(f"export {key}={shlex.quote(str(value))}")
    break
PYEOF
)
    eval "$exports"
}

method_gene_count() {
    local method="$1"
    python3 - "$method" << 'PYEOF'
import os
import sys
sys.path.insert(0, ".")
from parameterizations import get_parameterization
print(get_parameterization(sys.argv[1]).n_params)
PYEOF
}

write_index_row() {
    if [ "${MATRIX_UPDATE_INDEX:-0}" = "1" ]; then
        python3 - "$INDEX" "$run_index" "$method" "$optimizer" "$METHOD_SETUP" "$N_GENES" "$opt_dir" "$status" "$final_best" "$conv_dest" "$log_file" << 'PYEOF'
import csv
import os
import sys

path = sys.argv[1]
header = [
    "run_index",
    "method",
    "optimizer",
    "setup",
    "n_genes",
    "opt_dir",
    "status",
    "final_best",
    "convergence_csv",
    "log_file",
]
row = dict(zip(header, sys.argv[2:]))

rows = []
if os.path.isfile(path):
    with open(path, newline="") as f:
        for old in csv.DictReader(f):
            if old.get("method") == row["method"] and old.get("optimizer") == row["optimizer"]:
                continue
            rows.append({key: old.get(key, "") for key in header})

rows.append(row)

with open(path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=header)
    writer.writeheader()
    writer.writerows(rows)
PYEOF
    else
        echo "$run_index,$method,$optimizer,$METHOD_SETUP,$N_GENES,$opt_dir,$status,$final_best,$conv_dest,$log_file" >> "$INDEX"
    fi
}

cd "$BASE_DIR"

run_case() {
    local method="$1"
    local optimizer="$2"

    apply_selected_method_setup "$method"

    if [ "$method" = "hickshenne" ] && [ -n "${MATRIX_HH_N_BUMPS_OVERRIDE:-}" ]; then
        export HH_N_BUMPS="$MATRIX_HH_N_BUMPS_OVERRIDE"
        METHOD_SETUP="${METHOD_SETUP};HH_N_BUMPS_OVERRIDE=${HH_N_BUMPS}"
    fi

    if [ "$method" = "ffd" ]; then
        ffd_override_setup=""
        if [ -n "${MATRIX_FFD_N_ROWS_OVERRIDE:-}" ]; then
            export FFD_N_ROWS="$MATRIX_FFD_N_ROWS_OVERRIDE"
            ffd_override_setup="${ffd_override_setup};FFD_N_ROWS_OVERRIDE=${FFD_N_ROWS}"
        fi
        if [ -n "${MATRIX_FFD_M_OVERRIDE:-}" ]; then
            export FFD_M="$MATRIX_FFD_M_OVERRIDE"
            ffd_override_setup="${ffd_override_setup};FFD_M_OVERRIDE=${FFD_M}"
        fi
        if [ -n "${MATRIX_FFD_PIN_MODE_OVERRIDE:-}" ]; then
            export FFD_PIN_MODE="$MATRIX_FFD_PIN_MODE_OVERRIDE"
            ffd_override_setup="${ffd_override_setup};FFD_PIN_MODE_OVERRIDE=${FFD_PIN_MODE}"
        fi
        METHOD_SETUP="${METHOD_SETUP}${ffd_override_setup}"
    fi

    N_GENES=$(method_gene_count "$method")

    export PARAM_METHOD="$method"
    export OPTIMIZER="$optimizer"

    label="${method}__${optimizer}"
    log_file="$RESULTS_DIR/logs/${label}.log"
    conv_dest="$RESULTS_DIR/convergence/${label}.csv"

    echo ""
    echo "============================================================"
    echo "Method    : $method"
    echo "Optimizer : $optimizer"
    echo "Setup     : $METHOD_SETUP"
    echo "n_genes   : $N_GENES"
    echo "Log       : $log_file"
    if [ "$optimizer" = "pso" ]; then
        echo "PSO       : omega_max=${PSO_OMEGA_MAX:-default} omega_min=${PSO_OMEGA_MIN:-default} c1=${PSO_C1:-default} c2=${PSO_C2:-default}"
    fi
    if [ "$optimizer" = "cmaes" ]; then
        echo "CMAES     : sigma=$CMAES_SIGMA mu=$CMAES_MU cm=$CMAES_CM elitism=$CMAES_ELITISM max_resample=$CMAES_MAX_RESAMPLE"
    fi
    if [ "$optimizer" = "abc" ]; then
        echo "ABC       : mr=$ABC_MR employed_fraction=$ABC_EMPLOYED_FRACTION elitism=$ABC_ELITISM scout_period=$ABC_SCOUT_PERIOD scout_count=$ABC_SCOUT_COUNT"
    fi
    echo "============================================================"

    before_opt_dir=$(latest_opt_dir)

    if bash run_optimizer.sh < "$INPUT_FILE" > "$log_file" 2>&1; then
        status="ok"
    else
        status="failed"
    fi

    opt_dir=$(sed -n 's/^Optimization folder: //p' "$log_file" | tail -n 1)
    if [ -z "$opt_dir" ]; then
        opt_dir=$(latest_opt_dir)
    fi
    if [ "$opt_dir" = "$before_opt_dir" ]; then
        opt_dir=""
    fi
    run_index=""
    final_best=""

    if [ -n "$opt_dir" ]; then
        run_index=$(basename "$opt_dir" | sed 's/optimization_//')
    fi

    if [ "$status" = "ok" ] && [ -n "$opt_dir" ] && [ -f "$opt_dir/convergence.csv" ]; then
        cp "$opt_dir/convergence.csv" "$conv_dest"
        final_best=$(awk -F, '
            NR == 1 {
                value_col = 2
                for (i = 1; i <= NF; i++) {
                    if ($i == "cd") value_col = i
                }
                next
            }
            { value = $value_col }
            END { print value }
        ' "$conv_dest")
    else
        conv_dest=""
    fi

    write_index_row

    if [ "$status" = "ok" ]; then
        echo "Completed: $label  final_best=$final_best"
    else
        echo "FAILED: $label  see $log_file"
    fi

    unset PARAM_METHOD
    unset OPTIMIZER
}

if [ "${#CASES[@]}" -gt 0 ]; then
    for case_name in "${CASES[@]}"; do
        method="${case_name%%:*}"
        optimizer="${case_name#*:}"
        if [ "$method" = "$case_name" ] || [ -z "$method" ] || [ -z "$optimizer" ]; then
            echo "ERROR: invalid MATRIX_CASES entry '$case_name'; expected method:optimizer"
            exit 1
        fi
        run_case "$method" "$optimizer"
    done
else
    for method in "${METHODS[@]}"; do
        for optimizer in "${OPTIMIZERS[@]}"; do
            run_case "$method" "$optimizer"
        done
    done
fi

clear_method_env

echo ""
echo "Matrix study complete."
echo "Results: $RESULTS_DIR"
echo "Index  : $INDEX"
