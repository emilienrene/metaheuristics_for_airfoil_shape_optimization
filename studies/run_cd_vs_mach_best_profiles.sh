#!/usr/bin/env bash
set -euo pipefail

# Run rAIFoil Mach sweeps for RAE2822 and the selected best profile from each
# point study, then plot Cd versus Mach.
#
# Run from optimizer/:
#   bash studies/run_cd_vs_mach_best_profiles.sh
#
# Required first:
#   python3 studies/plot_seed_matrix_summary.py
#
# Main overrides:
#   SUMMARY_DIR=studies/seed_matrix_summary
#   RAE2822_PATH=/home/erene/simulations/rAIFoil/RAE2822.dat
#   MACH_MIN=0.65 MACH_MAX=0.80 MACH_STEP=0.01
#   MACH_VALUES="0.65 0.66 0.67 0.68 0.69 0.70 0.71 0.72 0.73 0.74 0.75 0.76 0.77 0.78 0.79 0.80"
#   AOA_MIN=0 AOA_MAX=10 AOA_N_POINTS=10 RE=2700000 CL_TARGET=0.733

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE_DIR"

SUMMARY_DIR="${SUMMARY_DIR:-$BASE_DIR/studies/seed_matrix_summary}"
SELECTION_CSV="${SELECTION_CSV:-$SUMMARY_DIR/best_seed_airfoil_selection.csv}"
OUT_DIR="${OUT_DIR:-$SUMMARY_DIR/cd_vs_mach_best_profiles}"
RAE2822_PATH="${RAE2822_PATH:-/home/erene/simulations/rAIFoil/RAE2822.dat}"
POINT_COUNTS="${POINT_COUNTS:-1 2 4}"

MACH_MIN="${MACH_MIN:-0.65}"
MACH_MAX="${MACH_MAX:-0.80}"
MACH_STEP="${MACH_STEP:-0.01}"
AOA_MIN="${AOA_MIN:-0}"
AOA_MAX="${AOA_MAX:-10}"
AOA_N_POINTS="${AOA_N_POINTS:-10}"
RE="${RE:-2700000}"
CL_TARGET="${CL_TARGET:-0.733}"

if ! command -v rAIFoil >/dev/null 2>&1; then
    echo "ERROR: rAIFoil is not in PATH."
    exit 1
fi

if [ ! -f "$SELECTION_CSV" ]; then
    echo "ERROR: missing selection CSV: $SELECTION_CSV"
    echo "Run first: python3 studies/plot_seed_matrix_summary.py"
    exit 1
fi

if [ ! -f "$RAE2822_PATH" ] && [ -f "$BASE_DIR/RAE2822.dat" ]; then
    RAE2822_PATH="$BASE_DIR/RAE2822.dat"
fi

if [ ! -f "$RAE2822_PATH" ]; then
    echo "ERROR: missing RAE2822 baseline: $RAE2822_PATH"
    exit 1
fi

if ! [[ "$AOA_N_POINTS" =~ ^[0-9]+$ ]] || [ "$AOA_N_POINTS" -lt 2 ]; then
    echo "ERROR: AOA_N_POINTS must be an integer >= 2; got '$AOA_N_POINTS'"
    exit 1
fi

mkdir -p "$OUT_DIR"

MANIFEST="$OUT_DIR/profile_manifest.tsv"
MACH_VALUES_FILE="$OUT_DIR/mach_values.txt"
RESULTS_CSV="$OUT_DIR/cd_vs_mach.csv"

python3 - "$SELECTION_CSV" "$RAE2822_PATH" "$MANIFEST" "$POINT_COUNTS" <<'PY'
import csv
import os
import sys

selection_csv, rae2822_path, manifest_path, point_counts_raw = sys.argv[1:5]
point_counts = [int(part) for part in point_counts_raw.replace(",", " ").split()]

method_labels = {
    "parsec": "PARSEC",
    "cst": "CST",
    "bspline": "B-spline",
    "hickshenne": "Hicks-Henne",
    "ffd": "FFD",
}

rows = [{
    "case_id": "rae2822",
    "n_points": "0",
    "label": "RAE2822",
    "method": "none",
    "optimizer": "none",
    "seed": "none",
    "airfoil_path": os.path.abspath(rae2822_path),
    "point_study_dir": "none",
    "final_best": "none",
    "mean_drag_reduction_percent": "none",
}]

with open(selection_csv, newline="") as f:
    reader = csv.DictReader(f)
    by_points = {}
    for row in reader:
        try:
            n_points = int(float(row.get("n_points", "")))
        except ValueError:
            continue
        by_points[n_points] = row

for n_points in point_counts:
    if n_points not in by_points:
        raise SystemExit(
            f"Missing {n_points}-point best profile in {selection_csv}. "
            "Regenerate it with plot_seed_matrix_summary.py."
        )
    row = by_points[n_points]
    method = row.get("method", "").strip().lower()
    optimizer = row.get("optimizer", "").strip().lower()
    method_label = method_labels.get(method, method)
    optimizer_label = optimizer.upper()
    label = f"{n_points}-point best"
    rows.append({
        "case_id": f"{n_points}point",
        "n_points": str(n_points),
        "label": label,
        "method": method,
        "optimizer": optimizer,
        "seed": row.get("seed", "").strip() or "none",
        "airfoil_path": row.get("best_airfoil_path", "").strip(),
        "point_study_dir": row.get("point_study_dir", "").strip() or "none",
        "final_best": row.get("final_best", "").strip() or "none",
        "mean_drag_reduction_percent": row.get("mean_drag_reduction_percent", "").strip() or "none",
    })
    print(f"Selected {label}: {method_label}-{optimizer_label}, seed {row.get('seed', '?')}")

fieldnames = [
    "case_id",
    "n_points",
    "label",
    "method",
    "optimizer",
    "seed",
    "airfoil_path",
    "point_study_dir",
    "final_best",
    "mean_drag_reduction_percent",
]
with open(manifest_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)
PY

if [ -n "${MACH_VALUES:-}" ]; then
    printf "%s\n" $MACH_VALUES > "$MACH_VALUES_FILE"
else
    python3 - "$MACH_MIN" "$MACH_MAX" "$MACH_STEP" "$MACH_VALUES_FILE" <<'PY'
import sys

start = float(sys.argv[1])
stop = float(sys.argv[2])
step = float(sys.argv[3])
out_path = sys.argv[4]

if step <= 0.0:
    raise SystemExit("MACH_STEP must be positive.")

n = int(round((stop - start) / step))
values = [start + i * step for i in range(n + 1)]
if values[-1] < stop - 1.0e-10:
    values.append(stop)

with open(out_path, "w") as f:
    for value in values:
        f.write(f"{value:.5f}".rstrip("0").rstrip(".") + "\n")
PY
fi

AOA_STEP=$(python3 - "$AOA_MIN" "$AOA_MAX" "$AOA_N_POINTS" <<'PY'
import sys
aoa_min = float(sys.argv[1])
aoa_max = float(sys.argv[2])
n_points = int(sys.argv[3])
print((aoa_max - aoa_min) / (n_points - 1))
PY
)

python3 - "$RESULTS_CSV" <<'PY'
import csv
import sys

path = sys.argv[1]
fieldnames = [
    "case_id",
    "n_points",
    "label",
    "method",
    "optimizer",
    "seed",
    "mach",
    "aoa",
    "cl",
    "cd",
    "cl_target_reached",
    "airfoil_path",
]
with open(path, "w", newline="") as f:
    csv.DictWriter(f, fieldnames=fieldnames).writeheader()
PY

echo "Cd-vs-Mach output: $OUT_DIR"
echo "Mach values:"
sed 's/^/  /' "$MACH_VALUES_FILE"

tail -n +2 "$MANIFEST" | while IFS=$'\t' read -r case_id n_points label method optimizer seed airfoil_path point_study_dir final_best mean_drag_reduction_percent; do
    if [ ! -f "$airfoil_path" ]; then
        echo "ERROR: missing airfoil for $label: $airfoil_path"
        exit 1
    fi

    geom_dir="$OUT_DIR/geometries/$case_id"
    mkdir -p "$geom_dir"
    cp "$airfoil_path" "$geom_dir/airfoil_1.dat"

    while read -r mach; do
        [ -n "$mach" ] || continue
        mach_tag=$(printf "%.3f" "$mach" | sed 's/\./p/g')
        eval_dir="$OUT_DIR/evaluations/$case_id/mach_$mach_tag"
        rm -rf "$eval_dir"
        mkdir -p "$eval_dir"

        input_path="$eval_dir/input"
        cat > "$input_path" <<EOF_INPUT
GEOMETRY_FOLDER = $geom_dir
AOA_MIN = $AOA_MIN
AOA_MAX = $AOA_MAX
AOA_STEP = $AOA_STEP
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
EOF_INPUT

        echo "Evaluating $label at Mach $mach"
        rAIFoil -i "$input_path" > "$eval_dir/rAIFoil.log" 2>&1
        python3 run_concat.py "$eval_dir" "$CL_TARGET" >> "$eval_dir/rAIFoil.log" 2>&1

        python3 - "$RESULTS_CSV" "$eval_dir/DD_80_combined.csv" "$case_id" "$n_points" "$label" "$method" "$optimizer" "$seed" "$mach" "$airfoil_path" <<'PY'
import csv
import math
import sys

out_csv, combined_csv, case_id, n_points, label, method, optimizer, seed, mach, airfoil_path = sys.argv[1:11]

with open(combined_csv, newline="") as f:
    rows = list(csv.DictReader(f))

if not rows:
    raise SystemExit(f"No rows in {combined_csv}")

row = rows[0]
out = {
    "case_id": case_id,
    "n_points": n_points,
    "label": label,
    "method": method,
    "optimizer": optimizer,
    "seed": seed,
    "mach": mach,
    "aoa": row.get("AOA", ""),
    "cl": row.get("CL", ""),
    "cd": row.get("CD", ""),
    "cl_target_reached": row.get("CL_target_reached", ""),
    "airfoil_path": airfoil_path,
}

fieldnames = [
    "case_id",
    "n_points",
    "label",
    "method",
    "optimizer",
    "seed",
    "mach",
    "aoa",
    "cl",
    "cd",
    "cl_target_reached",
    "airfoil_path",
]
with open(out_csv, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writerow(out)

try:
    cd = float(out["cd"])
except ValueError:
    cd = math.nan
print(f"  Cd={cd:.6g}  CL_target_reached={out['cl_target_reached']}")
PY
    done < "$MACH_VALUES_FILE"
done

python3 - "$RESULTS_CSV" "$OUT_DIR" "$CL_TARGET" "$POINT_COUNTS" <<'PY'
import csv
import math
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

results_csv, out_dir, cl_target, point_counts_raw = sys.argv[1:5]
point_counts = [int(part) for part in point_counts_raw.replace(",", " ").split()]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 13,
    "axes.labelsize": 13,
    "axes.labelweight": "bold",
    "axes.linewidth": 1.1,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "lines.linewidth": 1.8,
    "savefig.dpi": 220,
})

groups = defaultdict(list)
metadata = {}
with open(results_csv, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            mach = float(row["mach"])
            cd = float(row["cd"])
        except (ValueError, TypeError):
            continue
        if not math.isfinite(mach) or not math.isfinite(cd):
            continue
        case_id = row["case_id"]
        groups[case_id].append((mach, cd))
        metadata[case_id] = row

colors = {
    "rae2822": "black",
    "1point": "#1f77b4",
    "2point": "#ff7f0e",
    "4point": "#2ca02c",
}
linestyles = {
    "rae2822": "--",
    "1point": "-",
    "2point": "-",
    "4point": "-",
}
markers = {
    "rae2822": "o",
    "1point": "s",
    "2point": "^",
    "4point": "D",
}

operating_machs = {
    "1point": [0.74],
    "2point": [0.68, 0.74],
    "4point": [0.68, 0.71, 0.74, 0.76],
}
order = ["rae2822"] + [f"{n_points}point" for n_points in point_counts]

fig, ax = plt.subplots(figsize=(7.6, 4.8))

plotted = 0
for case_id in order:
    points = sorted(groups.get(case_id, []))
    if not points:
        continue

    x = [point[0] for point in points]
    y = [point[1] for point in points]
    label = metadata[case_id]["label"]
    ax.plot(
        x,
        y,
        color=colors.get(case_id, "#4b5563"),
        linestyle=linestyles.get(case_id, "-"),
        linewidth=1.8 if case_id == "rae2822" else 2.1,
        label=label,
    )
    plotted += 1

    if case_id == "rae2822":
        continue

    values_by_mach = {round(mx, 8): cd for mx, cd in points}
    highlight_x = []
    highlight_y = []
    for mach in operating_machs.get(case_id, []):
        key = round(mach, 8)
        if key in values_by_mach:
            highlight_x.append(mach)
            highlight_y.append(values_by_mach[key])

    if highlight_x:
        ax.scatter(
            highlight_x,
            highlight_y,
            s=62,
            facecolors="white",
            edgecolors=colors.get(case_id, "#4b5563"),
            linewidths=1.8,
            marker="o",
            zorder=5,
        )

if plotted < 2:
    raise SystemExit("Not enough Cd-vs-Mach curves were available to plot.")

ax.set_xlim(0.65, 0.80)
ax.set_xticks([0.65, 0.70, 0.75, 0.80])
ax.xaxis.set_minor_locator(MultipleLocator(0.01))
ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))

ax.set_xlabel("M", fontweight="bold")
ax.set_ylabel(r"$C_D$ at $C_L = " + str(cl_target) + "$", fontweight="bold")
ax.grid(True, which="major", axis="both", color="black", alpha=0.35, linewidth=0.75)
ax.grid(True, which="minor", axis="x", color="black", alpha=0.25, linewidth=0.55, linestyle=":")
ax.tick_params(
    axis="both",
    which="major",
    direction="in",
    width=1.25,
    length=5,
)
ax.tick_params(
    axis="x",
    which="minor",
    direction="in",
    width=0.9,
    length=3,
)
for spine in ax.spines.values():
    spine.set_linewidth(1.25)

handles, labels = ax.get_legend_handles_labels()
handles.append(
    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="none",
        markerfacecolor="white",
        markeredgecolor="black",
        markeredgewidth=1.8,
        markersize=6.5,
        label="Operating Machs",
    )
)
labels.append("Operating Machs")

legend = ax.legend(
    handles,
    labels,
    loc="upper left",
    frameon=True,
    borderaxespad=0.4,
)
legend.get_frame().set_linewidth(1.0)
for text in legend.get_texts():
    text.set_fontweight("bold")

plot_path = os.path.join(out_dir, "cd_vs_mach_best_profiles.png")
fig.subplots_adjust(left=0.13, right=0.985, top=0.96, bottom=0.15)
fig.savefig(plot_path, dpi=220, bbox_inches="tight", pad_inches=0.08)
plt.close(fig)
print(f"Saved plot: {plot_path}")
PY

echo "Saved data: $RESULTS_CSV"
echo "Saved plot: $OUT_DIR/cd_vs_mach_best_profiles.png"
