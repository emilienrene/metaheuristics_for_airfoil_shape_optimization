"""
Plot seed-averaged convergence/performance Pareto fronts.

Run from optimizer/ after plot_seed_matrix_summary.py:
    python3 studies/plot_pareto_tradeoff_summary.py

Inputs:
    studies/seed_matrix_summary/final_results_matrix_summary.csv
    studies/seed_matrix_summary/convergence_cauchy_seed_summary.csv

Outputs:
    studies/seed_matrix_summary/pareto_tradeoff_1point.png
    studies/seed_matrix_summary/pareto_tradeoff_2point.png
    studies/seed_matrix_summary/pareto_tradeoff_4point.png
    studies/seed_matrix_summary/pareto_tradeoff_summary.csv
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 13,
    "axes.labelsize": 13,
    "axes.labelweight": "bold",
    "axes.linewidth": 1.1,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "lines.linewidth": 2.0,
    "savefig.dpi": 220,
})


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = Path(os.environ.get(
    "SEED_SUMMARY_DIR",
    str(SCRIPT_DIR / "seed_matrix_summary"),
))
PERFORMANCE_CSV = Path(os.environ.get(
    "PARETO_PERFORMANCE_CSV",
    str(OUTPUT_DIR / "final_results_matrix_summary.csv"),
))
CAUCHY_CSV = Path(os.environ.get(
    "PARETO_CAUCHY_CSV",
    str(OUTPUT_DIR / "convergence_cauchy_seed_summary.csv"),
))

POINT_COUNTS = tuple(
    int(part)
    for part in os.environ.get("PARETO_POINT_STUDIES", "1 2 4").replace(",", " ").split()
)
FIGURE_PREFIX = os.environ.get("PARETO_FIGURE_PREFIX", "pareto_tradeoff")
PARETO_SUMMARY_CSV = Path(os.environ.get(
    "PARETO_SUMMARY_CSV",
    str(OUTPUT_DIR / "pareto_tradeoff_summary.csv"),
))

PARETO_FONT_SIZE = float(os.environ.get("PLOT_PARETO_FONT_SIZE", 13))
PARETO_TICK_FONT_SIZE = float(os.environ.get("PLOT_PARETO_TICK_FONT_SIZE", 11))
PARETO_LEGEND_FONT_SIZE = float(os.environ.get("PLOT_PARETO_LEGEND_FONT_SIZE", 11))
PARETO_SPINE_LINEWIDTH = float(os.environ.get("PLOT_PARETO_SPINE_LINEWIDTH", 1.25))
PARETO_MARKER_SIZE = float(os.environ.get("PLOT_PARETO_MARKER_SIZE", 90))
PARETO_SHOW_ERRORBARS = os.environ.get("PLOT_PARETO_ERRORBARS", "1").strip() == "1"
PARETO_ERRORBAR_LINEWIDTH = float(os.environ.get("PLOT_PARETO_ERRORBAR_LINEWIDTH", 1.0))
PARETO_ERRORBAR_CAPSIZE = float(os.environ.get("PLOT_PARETO_ERRORBAR_CAPSIZE", 3.0))

XMIN = os.environ.get("PLOT_PARETO_XMIN", "0")
XMAX = os.environ.get("PLOT_PARETO_XMAX", "250")
YMIN = os.environ.get("PLOT_PARETO_YMIN", "50")
YMAX = os.environ.get("PLOT_PARETO_YMAX")
SHARED_Y_AXIS = os.environ.get("PLOT_PARETO_SHARED_Y", "1").strip() == "1"

METHODS = ["parsec", "cst", "bspline", "hickshenne", "ffd"]
OPTIMIZERS = ["ga", "pso", "de", "cmaes", "abc"]

METHOD_LABELS = {
    "parsec": "PARSEC",
    "cst": "CST",
    "bspline": "B-spline",
    "hickshenne": "Hicks-Henne",
    "ffd": "FFD",
}
OPTIMIZER_LABELS = {optimizer: optimizer.upper() for optimizer in OPTIMIZERS}
METHOD_COLORS = {
    "parsec": "#1f77b4",
    "cst": "#ff7f0e",
    "bspline": "#2ca02c",
    "hickshenne": "#d62728",
    "ffd": "#9467bd",
}
OPTIMIZER_MARKERS = {
    "ga": "o",
    "pso": "s",
    "de": "^",
    "cmaes": "D",
    "abc": "P",
}


def main() -> None:
    require_input(PERFORMANCE_CSV)
    require_input(CAUCHY_CSV)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_pareto_data()
    if data.empty:
        raise SystemExit("No finite convergence/performance rows found for Pareto plotting.")

    for n_points in POINT_COUNTS:
        point_mask = data["n_points"] == n_points
        if not point_mask.any():
            print(f"Skipping {n_points}-point study; no merged rows.")

    data.to_csv(PARETO_SUMMARY_CSV, index=False)

    shared_ylim = performance_ylim(data) if SHARED_Y_AXIS else None
    for n_points in POINT_COUNTS:
        subset = data[data["n_points"] == n_points].copy()
        if subset.empty:
            continue
        plot_tradeoff(subset, n_points, shared_ylim)

    print(f"Wrote Pareto summary CSV: {PARETO_SUMMARY_CSV}")


def require_input(path: Path) -> None:
    if path.is_file():
        return
    raise SystemExit(
        f"Missing input CSV: {path}\n"
        "Run first: python3 studies/plot_seed_matrix_summary.py"
    )


def load_pareto_data() -> pd.DataFrame:
    perf = pd.read_csv(PERFORMANCE_CSV)
    cauchy = pd.read_csv(CAUCHY_CSV)

    required_perf = {
        "n_points",
        "method",
        "optimizer",
        "mean_drag_reduction_percent",
        "std_drag_reduction_percent",
    }
    required_cauchy = {
        "n_points",
        "method",
        "optimizer",
        "mean_iterations_to_converge",
        "std_iterations_to_converge",
        "n_converged",
        "n_completed_runs",
        "convergence_rate",
    }
    missing_perf = sorted(required_perf - set(perf.columns))
    missing_cauchy = sorted(required_cauchy - set(cauchy.columns))
    if missing_perf or missing_cauchy:
        messages = []
        if missing_perf:
            messages.append(f"{PERFORMANCE_CSV} missing columns: {', '.join(missing_perf)}")
        if missing_cauchy:
            messages.append(f"{CAUCHY_CSV} missing columns: {', '.join(missing_cauchy)}")
        raise SystemExit("\n".join(messages))

    for df in (perf, cauchy):
        df["n_points"] = pd.to_numeric(df["n_points"], errors="coerce")
        df["method"] = df["method"].astype(str).str.strip().str.lower()
        df["optimizer"] = df["optimizer"].astype(str).str.strip().str.lower()

    merged = pd.merge(
        perf,
        cauchy,
        on=["n_points", "method", "optimizer"],
        how="inner",
        suffixes=("_performance", "_cauchy"),
    )

    numeric_columns = [
        "n_points",
        "mean_drag_reduction_percent",
        "std_drag_reduction_percent",
        "mean_iterations_to_converge",
        "std_iterations_to_converge",
        "n_converged",
        "n_completed_runs",
        "convergence_rate",
    ]
    for column in numeric_columns:
        merged[column] = pd.to_numeric(merged[column], errors="coerce")

    keep = (
        merged["n_points"].isin(POINT_COUNTS)
        & np.isfinite(merged["mean_drag_reduction_percent"])
        & np.isfinite(merged["mean_iterations_to_converge"])
    )
    merged = merged[keep].copy()
    merged["n_points"] = merged["n_points"].astype(int)
    return merged


def plot_tradeoff(
    df: pd.DataFrame,
    n_points: int,
    shared_ylim: tuple[float, float] | None,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.2))

    for method in METHODS:
        for optimizer in OPTIMIZERS:
            rows = df[(df["method"] == method) & (df["optimizer"] == optimizer)]
            if rows.empty:
                continue

            row = rows.iloc[-1]
            x = float(row["mean_iterations_to_converge"])
            y = float(row["mean_drag_reduction_percent"])
            xerr = finite_or_none(row.get("std_iterations_to_converge"))
            yerr = finite_or_none(row.get("std_drag_reduction_percent"))

            if PARETO_SHOW_ERRORBARS:
                ax.errorbar(
                    x,
                    y,
                    xerr=xerr,
                    yerr=yerr,
                    fmt="none",
                    ecolor="black",
                    elinewidth=PARETO_ERRORBAR_LINEWIDTH,
                    capsize=PARETO_ERRORBAR_CAPSIZE,
                    alpha=0.35,
                    zorder=1,
                )

            ax.scatter(
                x,
                y,
                s=PARETO_MARKER_SIZE,
                marker=OPTIMIZER_MARKERS.get(optimizer, "o"),
                facecolor=METHOD_COLORS.get(method, "#4b5563"),
                edgecolor="black",
                linewidth=0.7,
                alpha=0.85,
                zorder=3,
            )

    style_axis(ax, df, shared_ylim)
    add_legends(fig, ax)

    fig.subplots_adjust(left=0.16, right=0.98, top=0.97, bottom=0.31)
    output_path = OUTPUT_DIR / f"{FIGURE_PREFIX}_{n_points}point.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"Saved {n_points}-point convergence-performance tradeoff: {output_path}")


def style_axis(
    ax,
    df: pd.DataFrame,
    shared_ylim: tuple[float, float] | None,
) -> None:
    ax.set_xlabel("Iterations to convergence", fontweight="bold", fontsize=PARETO_FONT_SIZE)
    ax.set_ylabel("Mean drag reduction (%)", fontweight="bold", fontsize=PARETO_FONT_SIZE)

    xlim = explicit_or_default_xlim(df)
    ylim = explicit_or_default_ylim(df, shared_ylim)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    ax.grid(True, axis="both", alpha=0.3, linewidth=0.75)
    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        width=PARETO_SPINE_LINEWIDTH,
        length=5,
        labelsize=PARETO_TICK_FONT_SIZE,
    )
    for spine in ax.spines.values():
        spine.set_linewidth(PARETO_SPINE_LINEWIDTH)


def explicit_or_default_xlim(df: pd.DataFrame) -> tuple[float, float]:
    if XMIN.strip().lower() not in {"", "auto", "none"}:
        xmin = float(XMIN)
    else:
        xmin = float(df["mean_iterations_to_converge"].min())

    if XMAX.strip().lower() not in {"", "auto", "none"}:
        xmax = float(XMAX)
    else:
        xmax = float(df["mean_iterations_to_converge"].max())

    if np.isclose(xmin, xmax):
        pad = max(1.0, abs(xmin) * 0.05)
        xmin -= pad
        xmax += pad
    return xmin, xmax


def explicit_or_default_ylim(
    df: pd.DataFrame,
    shared_ylim: tuple[float, float] | None,
) -> tuple[float, float]:
    if YMIN is not None and YMIN.strip().lower() not in {"", "auto", "none"}:
        ymin = float(YMIN)
    elif shared_ylim is not None:
        ymin = shared_ylim[0]
    else:
        ymin = float(df["mean_drag_reduction_percent"].min())

    if YMAX is not None and YMAX.strip().lower() not in {"", "auto", "none"}:
        ymax = float(YMAX)
    elif shared_ylim is not None:
        ymax = shared_ylim[1]
    else:
        ymax = float(df["mean_drag_reduction_percent"].max())

    if np.isclose(ymin, ymax):
        pad = max(1.0, abs(ymin) * 0.05)
        ymin -= pad
        ymax += pad
    return ymin, ymax


def performance_ylim(df: pd.DataFrame) -> tuple[float, float]:
    values = df["mean_drag_reduction_percent"].to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.0

    ymin = float(np.min(values))
    ymax = float(np.max(values))
    if np.isclose(ymin, ymax):
        pad = max(1.0, abs(ymin) * 0.05)
    else:
        pad = 0.06 * (ymax - ymin)
    return ymin - pad, ymax + pad


def add_legends(fig, ax) -> None:
    method_handles = [
        Patch(
            facecolor=METHOD_COLORS[method],
            edgecolor="none",
            label=METHOD_LABELS[method],
        )
        for method in METHODS
    ]
    optimizer_handles = [
        Line2D(
            [0],
            [0],
            marker=OPTIMIZER_MARKERS[optimizer],
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=1.1,
            markersize=7.5,
            label=OPTIMIZER_LABELS[optimizer],
        )
        for optimizer in OPTIMIZERS
    ]
    legend = fig.legend(
        handles=method_handles + optimizer_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025),
        ncol=6,
        frameon=False,
        handlelength=1.8,
        columnspacing=1.1,
        borderaxespad=0.0,
    )
    for text in legend.get_texts():
        text.set_fontweight("bold")
        text.set_fontsize(PARETO_LEGEND_FONT_SIZE)


def finite_or_none(value) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return value


def combo_label(method: str, optimizer: str) -> str:
    return f"{METHOD_LABELS.get(method, method)}-{OPTIMIZER_LABELS.get(optimizer, optimizer.upper())}"


if __name__ == "__main__":
    main()
