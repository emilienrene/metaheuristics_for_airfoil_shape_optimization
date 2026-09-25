"""
plot_sweep.py

Plots convergence curves for the parameterization parameter sweep study.
One figure per method, each showing all parameter configurations as overlaid
lines on a single set of axes.

Usage (run from optimizer/):
    python3 studies/plot_sweep.py

Output:
    studies/parameterization_parameter_sweep/<method>/<method>_convergence.png
"""

from __future__ import annotations

import os
import re
import glob
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from itertools import cycle

matplotlib.use("Agg")

# ── Paths ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "parameterization_parameter_sweep")

# ── Style ─────────────────────────────────────────────────────────────────────

PALETTE = [
    "#2563eb",  # blue
    "#dc2626",  # red
    "#16a34a",  # green
    "#9333ea",  # purple
    "#ea580c",  # orange
    "#0891b2",  # cyan
    "#db2777",  # pink
    "#65a30d",  # lime
    "#d97706",  # amber
    "#7c3aed",  # violet
    "#0f766e",  # teal
    "#b45309",  # brown
    "#e11d48",  # rose
    "#1d4ed8",  # indigo
    "#15803d",  # dark green
]

plt.rcParams.update({
    "font.family"      : "DejaVu Sans",
    "font.size"        : 10,
    "axes.linewidth"   : 0.8,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "grid.color"       : "#e5e7eb",
    "grid.linewidth"   : 0.6,
    "legend.frameon"   : False,
    "legend.fontsize"  : 9,
    "figure.dpi"       : 150,
})

# ── Label helpers ─────────────────────────────────────────────────────────────

PARAM_LABELS = {
    "NONE"          : "fixed",
    "CST_ORDER"     : "order",
    "BSPLINE_N_CTRL": "n_ctrl",
    "HH_N_BUMPS"    : "n_bumps",
    "FFD_M"         : "m",
    "FFD_N_ROWS"    : "n_rows",
}

METHOD_TITLES = {
    "parsec"    : "PARSEC",
    "cst"       : "CST  (Kulfan + LEM)",
    "bspline"   : "B-Spline  (cubic, √x reparameterization)",
    "hickshenne": "Hicks-Henne",
    "ffd"       : "Free-Form Deformation  (Bézier, 3-row)",
}

def parse_label(filename: str) -> str:
    """Extract a short legend label from the CSV filename."""
    stem = os.path.splitext(os.path.basename(filename))[0]
    # e.g. cst_CST_ORDER=4  ->  order = 4
    m = re.search(r"_([A-Z_]+)=(\d+)$", stem)
    if m:
        env_var, value = m.group(1), m.group(2)
        param = PARAM_LABELS.get(env_var, env_var.lower())
        return f"{param} = {value}"
    if stem.endswith("_fixed"):
        return "fixed (11 genes)"
    return stem

def n_genes_from_label(label: str) -> int:
    """Extract numeric value for sort key."""
    m = re.search(r"=\s*(\d+)", label)
    return int(m.group(1)) if m else 0

# ── Main ──────────────────────────────────────────────────────────────────────

method_dirs = sorted([
    d for d in glob.glob(os.path.join(RESULTS_DIR, "*"))
    if os.path.isdir(d)
])

if not method_dirs:
    print(f"No method subdirectories found in {RESULTS_DIR}")
    print("Run the sweep first: bash studies/param_sweep.sh")
    raise SystemExit(1)

for method_dir in method_dirs:
    method = os.path.basename(method_dir)
    csv_files = sorted(glob.glob(os.path.join(method_dir, "*.csv")))

    if not csv_files:
        print(f"  [{method}] no CSV files found, skipping")
        continue

    # load all curves
    curves: list[tuple[str, pd.DataFrame]] = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            if df.shape[1] < 2:
                continue
            df.columns = ["generation", "best_fitness"]
            label = parse_label(f)
            curves.append((label, df))
        except Exception as e:
            print(f"  WARNING: could not read {f}: {e}")

    if not curves:
        print(f"  [{method}] no readable curves, skipping")
        continue

    # sort by parameter value ascending
    curves.sort(key=lambda t: n_genes_from_label(t[0]))

    # ── Figure ────────────────────────────────────────────────────────────────

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for (label, df), color in zip(curves, cycle(PALETTE)):
        generations = df["generation"].to_numpy()
        fitness     = df["best_fitness"].to_numpy()

        # replace zeros with NaN for cleaner display (penalty generations)
        fitness_plot = np.where(fitness == 0, np.nan, fitness)

        ax.plot(
            generations,
            fitness_plot,
            color     = color,
            linewidth = 1.8,
            label     = label,
        )
        # mark the final value
        last_valid = np.where(~np.isnan(fitness_plot))[0]
        if len(last_valid):
            i = last_valid[-1]
            ax.scatter(
                generations[i], fitness_plot[i],
                color=color, s=30, zorder=5
            )

    ax.set_xlabel("Generation", labelpad=6)
    ax.set_ylabel("Best fitness", labelpad=6)
    ax.set_title(
        METHOD_TITLES.get(method, method.upper()) + " — parameter sweep",
        fontsize=11, fontweight="semibold", pad=10
    )
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=10))
    ax.grid(True, axis="y")
    ax.legend(title="Design variables", title_fontsize=9, loc="lower right")

    fig.tight_layout()

    out_path = os.path.join(method_dir, f"{method}_convergence.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [{method}] saved → {out_path}")

print("\nDone.")
