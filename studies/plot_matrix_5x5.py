"""
Plot the 5x5 parameterization x optimizer matrix study.

Run from optimizer/:
    python3 studies/plot_matrix_5x5.py

Inputs:
    studies/matrix_5x5/study_index.csv
    studies/matrix_5x5/convergence/*.csv

Outputs:
    studies/matrix_5x5/final_best_heatmap.png
    studies/matrix_5x5/convergence_by_optimizer_wide.png
    studies/matrix_5x5/best_airfoils_by_method.png
    studies/matrix_5x5/convergence_cauchy_summary.csv
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
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


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.environ.get(
    "MATRIX_RESULTS_DIR",
    os.path.join(SCRIPT_DIR, os.environ.get("MATRIX_RESULTS_NAME", "matrix_5x5")),
)
INDEX_PATH = os.path.join(RESULTS_DIR, "study_index.csv")
CONVERGENCE_DIR = os.path.join(RESULTS_DIR, "convergence")

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

PENALTY_THRESHOLD = float(os.environ.get("PLOT_PENALTY_THRESHOLD", 99.0))
HEATMAP_POINT_COUNTS = (1, 2, 4)
HEATMAP_CMAP = os.environ.get("PLOT_HEATMAP_CMAP", "RdYlGn")
HEATMAP_FONT_SIZE = float(os.environ.get("PLOT_HEATMAP_FONT_SIZE", 13))
AIRFOIL_FONT_SIZE = float(os.environ.get("PLOT_AIRFOIL_FONT_SIZE", 13))
AIRFOIL_TICK_FONT_SIZE = float(os.environ.get("PLOT_AIRFOIL_TICK_FONT_SIZE", 11))
AIRFOIL_TITLE_FONT_SIZE = float(os.environ.get("PLOT_AIRFOIL_TITLE_FONT_SIZE", 15))
AIRFOIL_LINEWIDTH = float(os.environ.get("PLOT_AIRFOIL_LINEWIDTH", 2.0))
AIRFOIL_SPINE_LINEWIDTH = float(os.environ.get("PLOT_AIRFOIL_SPINE_LINEWIDTH", 1.35))
CONVERGENCE_FONT_SIZE = float(os.environ.get("PLOT_CONV_FONT_SIZE", 13))
CONVERGENCE_TICK_FONT_SIZE = float(os.environ.get("PLOT_CONV_TICK_FONT_SIZE", 10))
CONVERGENCE_PANEL_FONT_SIZE = float(os.environ.get("PLOT_CONV_PANEL_FONT_SIZE", 13))
CONVERGENCE_LEGEND_FONT_SIZE = float(os.environ.get("PLOT_CONV_LEGEND_FONT_SIZE", 13))
CONVERGENCE_PANEL_LABEL_FONT_SIZE = float(os.environ.get("PLOT_CONV_PANEL_LABEL_FONT_SIZE", 13))
CONVERGENCE_LINEWIDTH = float(os.environ.get("PLOT_CONV_LINEWIDTH", 2.0))
CONVERGENCE_PANEL_LABEL = os.environ.get("PLOT_CONV_PANEL_LABEL", "").strip()
CAUCHY_DRAG_COUNT = float(os.environ.get("PLOT_CAUCHY_DRAG_COUNT", 1.0e-4))
CAUCHY_EPS_REL = float(os.environ.get("PLOT_CAUCHY_EPS_REL", 0.0))
CONVERGENCE_XMIN = float(os.environ.get("PLOT_CONV_XMIN", 0.0))
CONVERGENCE_XMAX = float(os.environ.get("PLOT_CONV_XMAX", 500.0))
CONVERGENCE_YMIN = os.environ.get("PLOT_CONV_YMIN")
CONVERGENCE_YMAX = os.environ.get("PLOT_CONV_YMAX")
_DAE11_PLOT_CONTEXT = re.sub(
    r"[^a-z0-9]+",
    "",
    " ".join((
        RESULTS_DIR,
        os.environ.get("PARAM_REFERENCE_AIRFOIL", ""),
        os.environ.get("DEFORMATION_BASELINE", ""),
        os.environ.get("DAE11_DATA_FILE", ""),
    )).lower(),
)
IS_DAE11_PLOT = "dae11" in _DAE11_PLOT_CONTEXT
REQUIRE_RAW_OBJECTIVE = (
    os.environ.get(
        "PLOT_REQUIRE_RAW_OBJECTIVE",
        "1" if IS_DAE11_PLOT else "0",
    ).strip().lower()
    in {"1", "true", "yes", "on"}
)
CONVERGENCE_YLABEL = os.environ.get(
    "PLOT_CONV_YLABEL",
    (
        r"Best-historic $C_D$"
        if IS_DAE11_PLOT
        else r"Best-historic objective, $\mathcal{F}^*$"
    ),
)
CONVERGENCE_YLIMITS_BY_N_POINTS = {
    1: (0.02, 0.06),
    2: (0.04, 0.10),
    4: (0.10, 0.20),
}
CONVERGENCE_YTICKS_BY_N_POINTS = {
    1: np.array([0.02, 0.03, 0.04, 0.05, 0.06]),
    2: np.array([0.04, 0.06, 0.08, 0.10]),
    4: np.array([0.10, 0.12, 0.14, 0.16, 0.18, 0.20]),
}

# Edit these values to the baseline drag sum you want to compare against.
# For multi-point studies, use the sum of the baseline Cd values across
# the objective Mach points, e.g. Cd(M1) + Cd(M2) [+ Cd(M3) + Cd(M4)].
DAE11_BASELINE_CD = float(os.environ.get("DAE11_BASELINE_CD", "0.050505"))
INITIAL_DRAG_SUM_BY_N_POINTS = {
    1: float(os.environ.get(
        "INITIAL_DRAG_SUM_1POINT",
        str(DAE11_BASELINE_CD) if IS_DAE11_PLOT else "0.103416",
    )),
    2: 0.14673,
    4: 0.346914,
}


def main() -> None:
    if not os.path.isfile(INDEX_PATH):
        raise SystemExit(
            f"No matrix index found: {INDEX_PATH}\n"
            "Run first: bash studies/matrix_5x5.sh"
        )

    index = pd.read_csv(INDEX_PATH)
    index["status"] = index["status"].astype(str).str.strip().str.lower()
    index["final_best"] = pd.to_numeric(index["final_best"], errors="coerce")

    plot_heatmap(index)
    plot_convergence_by_optimizer_wide(index)
    plot_best_airfoils_by_method(index)
    write_cauchy_convergence_summary(index)
    print(f"Wrote plots to {RESULTS_DIR}")


def plot_heatmap(index: pd.DataFrame) -> None:
    n_points = infer_n_points()
    matrix = heatmap_matrix(index, n_points, Path(RESULTS_DIR))
    vmin, vmax = shared_heatmap_limits(matrix)

    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    masked = np.ma.masked_invalid(matrix)
    image = ax.imshow(
        masked,
        cmap=HEATMAP_CMAP,
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_xticks(np.arange(len(OPTIMIZERS)))
    ax.set_yticks(np.arange(len(METHODS)))
    ax.set_xticklabels(
        [o.upper() for o in OPTIMIZERS],
        fontweight="bold",
        fontsize=HEATMAP_FONT_SIZE,
        rotation=30,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_yticklabels(
        [METHOD_LABELS[m] for m in METHODS],
        fontweight="bold",
        fontsize=HEATMAP_FONT_SIZE,
        rotation=30,
        ha="right",
        va="center",
        rotation_mode="anchor",
    )
    ax.set_xticks(np.arange(-0.5, len(OPTIMIZERS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(METHODS), 1), minor=True)
    ax.grid(which="minor", color="black", linestyle="-", linewidth=0.9)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(axis="both", which="major", length=0, pad=7, width=1.4)
    for spine in ax.spines.values():
        spine.set_linewidth(1.4)

    for i in range(len(METHODS)):
        for j in range(len(OPTIMIZERS)):
            value = matrix[i, j]
            text = "-" if np.isnan(value) else f"{value:.1f}%"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="black",
                fontsize=HEATMAP_FONT_SIZE,
                fontweight="bold",
            )

    cbar = fig.colorbar(image, ax=ax)
    cbar.ax.set_xlabel(
        "Objective\nreduction",
        fontweight="bold",
        fontsize=HEATMAP_FONT_SIZE,
        labelpad=12,
    )
    cbar.outline.set_linewidth(1.4)
    cbar.ax.tick_params(width=1.4, labelsize=HEATMAP_FONT_SIZE)
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontweight("bold")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "final_best_heatmap.png"), dpi=220)
    plt.close(fig)


def heatmap_matrix(
    index: pd.DataFrame,
    n_points: int,
    results_dir: Path,
) -> np.ndarray:
    matrix = np.full((len(METHODS), len(OPTIMIZERS)), np.nan)
    initial_drag_sum = INITIAL_DRAG_SUM_BY_N_POINTS[n_points]

    for i, method in enumerate(METHODS):
        for j, optimizer in enumerate(OPTIMIZERS):
            rows = index[
                (index["method"] == method)
                & (index["optimizer"] == optimizer)
                & (index["status"] == "ok")
            ]
            if not rows.empty:
                row = rows.iloc[-1]
                value = best_ever_from_convergence(row, results_dir / "convergence")
                if value is None and not IS_DAE11_PLOT:
                    value = float(row["final_best"])
                if value is None or not np.isfinite(value) or value >= initial_drag_sum:
                    matrix[i, j] = 0.0
                else:
                    matrix[i, j] = drag_reduction_percent(value, initial_drag_sum)

    return matrix


def shared_heatmap_limits(current_matrix: np.ndarray) -> tuple[float, float]:
    env_vmin = os.environ.get("PLOT_HEATMAP_VMIN")
    env_vmax = os.environ.get("PLOT_HEATMAP_VMAX")
    if env_vmin is not None and env_vmax is not None:
        return float(env_vmin), float(env_vmax)

    matrices = [current_matrix]
    for results_dir in sibling_point_result_dirs():
        n_points = infer_n_points_from_dir(results_dir)
        if n_points is None or n_points not in INITIAL_DRAG_SUM_BY_N_POINTS:
            continue

        index_path = results_dir / "study_index.csv"
        if not index_path.is_file():
            continue

        try:
            index = pd.read_csv(index_path)
        except Exception as exc:
            print(f"  unreadable matrix index {index_path}: {exc}")
            continue

        index["status"] = index["status"].astype(str).str.strip().str.lower()
        index["final_best"] = pd.to_numeric(index["final_best"], errors="coerce")
        matrices.append(heatmap_matrix(index, n_points, results_dir))

    values = np.concatenate([
        matrix[np.isfinite(matrix)].ravel()
        for matrix in matrices
        if np.isfinite(matrix).any()
    ]) if any(np.isfinite(matrix).any() for matrix in matrices) else np.array([])

    if values.size == 0:
        return 0.0, 1.0

    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if np.isclose(vmin, vmax):
        pad = max(1.0, abs(vmin) * 0.05)
        return vmin - pad, vmax + pad
    return vmin, vmax


def sibling_point_result_dirs() -> list[Path]:
    results_dir = Path(RESULTS_DIR).resolve()
    parent = results_dir.parent
    prefix_match = re.match(r"(.+)_\d+point$", results_dir.name)
    prefix = prefix_match.group(1) if prefix_match else "matrix_5x5"
    return [parent / f"{prefix}_{n_points}point" for n_points in HEATMAP_POINT_COUNTS]


def convergence_ylim(n_points: int | None = None) -> tuple[float, float] | None:
    ymin = "" if CONVERGENCE_YMIN is None else str(CONVERGENCE_YMIN).strip().lower()
    ymax = "" if CONVERGENCE_YMAX is None else str(CONVERGENCE_YMAX).strip().lower()
    ymin_auto = ymin in {"", "auto", "none"}
    ymax_auto = ymax in {"", "auto", "none"}

    default_ylim = CONVERGENCE_YLIMITS_BY_N_POINTS.get(n_points)
    if default_ylim is not None:
        return (
            default_ylim[0] if ymin_auto else float(ymin),
            default_ylim[1] if ymax_auto else float(ymax),
        )

    if ymin_auto or ymax_auto:
        return None
    return float(ymin), float(ymax)


def add_convergence_panel_label(fig) -> None:
    if not CONVERGENCE_PANEL_LABEL:
        return
    fig.text(
        0.012,
        0.965,
        CONVERGENCE_PANEL_LABEL,
        ha="left",
        va="top",
        fontweight="bold",
        fontsize=CONVERGENCE_PANEL_LABEL_FONT_SIZE,
    )


def style_convergence_axis(
    ax,
    label: str,
    show_ylabel: bool = False,
    n_points: int | None = None,
) -> None:
    ax.text(
        0.94,
        0.88,
        label,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontweight="bold",
        fontsize=CONVERGENCE_PANEL_FONT_SIZE,
    )
    ax.set_xlim(CONVERGENCE_XMIN, CONVERGENCE_XMAX)
    first_tick = np.ceil(CONVERGENCE_XMIN / 100.0) * 100.0
    ax.set_xticks(np.arange(first_tick, CONVERGENCE_XMAX + 1.0e-9, 100.0))
    ylim = convergence_ylim(n_points)
    if ylim is not None:
        ax.set_ylim(*ylim)
        yticks = CONVERGENCE_YTICKS_BY_N_POINTS.get(n_points)
        if yticks is not None:
            ymin, ymax = sorted(ylim)
            ax.set_yticks(yticks[(yticks >= ymin) & (yticks <= ymax)])
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))

    if show_ylabel:
        ax.set_ylabel(CONVERGENCE_YLABEL, fontsize=CONVERGENCE_FONT_SIZE, labelpad=8)

    ax.grid(True, axis="y", alpha=0.35, linewidth=0.75)
    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        width=1.2,
        length=4,
        labelsize=CONVERGENCE_TICK_FONT_SIZE,
        top=False,
        right=False,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.25)
    ax.spines["bottom"].set_linewidth(1.25)


def method_legend_handles(methods: list[str]) -> list[Patch]:
    return [
        Patch(
            facecolor=METHOD_COLORS.get(method, "#4b5563"),
            edgecolor="none",
            label=METHOD_LABELS.get(method, method),
        )
        for method in methods
    ]


def style_convergence_legend(legend) -> None:
    if legend is None:
        return

    legend.get_frame().set_linewidth(0.9)
    for text in legend.get_texts():
        text.set_fontweight("bold")
        text.set_fontsize(CONVERGENCE_LEGEND_FONT_SIZE)


def plot_convergence_by_optimizer(index: pd.DataFrame) -> None:
    n_points = infer_n_points()
    fig, axes = plt.subplots(
        len(OPTIMIZERS),
        1,
        figsize=(8, 12),
        sharex=True,
        sharey=True,
    )
    plotted_total = 0

    for ax, optimizer in zip(axes, OPTIMIZERS):
        subset = index[(index["optimizer"] == optimizer) & (index["status"] == "ok")]
        plotted_here = 0
        legend_methods = []

        for _, row in subset.iterrows():
            method = str(row["method"])
            conv_path = resolve_convergence_path(row)
            if conv_path is None:
                print(
                    f"  missing convergence: "
                    f"{row.get('method', '?')} / {row.get('optimizer', '?')}"
                )
                continue

            df = load_best_ever_convergence(conv_path)
            if df.empty:
                #print(f"  only penalty/no numeric convergence rows: {conv_path}")
                continue

            ax.plot(
                df["generation"],
                df["best_fitness"],
                linewidth=CONVERGENCE_LINEWIDTH,
                color=METHOD_COLORS.get(method),
                label=METHOD_LABELS.get(method, method),
            )
            if method not in legend_methods:
                legend_methods.append(method)
            plotted_here += 1
            plotted_total += 1

        style_convergence_axis(
            ax,
            OPTIMIZER_LABELS.get(optimizer, optimizer.upper()),
            show_ylabel=True,
            n_points=n_points,
        )
        if plotted_here:
            legend = ax.legend(
                handles=method_legend_handles(legend_methods),
                loc="lower left",
                fontsize=CONVERGENCE_FONT_SIZE,
                frameon=False,
            )
            style_convergence_legend(legend)
        else:
            ax.text(
                0.5, 0.5, "no completed runs yet",
                transform=ax.transAxes,
                ha="center", va="center",
                color="#6b7280",
            )

    axes[-1].set_xlabel("Generation")
    fig.tight_layout()
    add_convergence_panel_label(fig)
    out_path = os.path.join(RESULTS_DIR, "convergence_by_optimizer.png")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"  plotted convergence curves: {plotted_total}")
    print(f"  saved: {out_path}")


def plot_convergence_by_optimizer_wide(index: pd.DataFrame) -> None:
    n_points = infer_n_points()
    fig, axes = plt.subplots(
        1,
        len(OPTIMIZERS),
        figsize=(11.5, 4.2),
        sharex=True,
        sharey=True,
    )
    plotted_total = 0
    legend_handles = {}

    for ax, optimizer in zip(axes, OPTIMIZERS):
        subset = index[(index["optimizer"] == optimizer) & (index["status"] == "ok")]
        plotted_here = 0

        for method in METHODS:
            rows = subset[subset["method"] == method]
            if rows.empty:
                continue

            row = rows.sort_values("run_index", kind="mergesort").iloc[-1]
            conv_path = resolve_convergence_path(row)
            if conv_path is None:
                print(
                    f"  missing convergence: "
                    f"{row.get('method', '?')} / {row.get('optimizer', '?')}"
                )
                continue

            df = load_best_ever_convergence(conv_path)
            if df.empty:
                #print(f"  only penalty/no numeric convergence rows: {conv_path}")
                continue

            line, = ax.plot(
                df["generation"],
                df["best_fitness"],
                linewidth=CONVERGENCE_LINEWIDTH,
                color=METHOD_COLORS.get(method),
                label=METHOD_LABELS.get(method, method),
            )
            legend_handles.setdefault(method, line)
            plotted_here += 1
            plotted_total += 1

        style_convergence_axis(
            ax,
            OPTIMIZER_LABELS.get(optimizer, optimizer.upper()),
            show_ylabel=ax is axes[0],
            n_points=n_points,
        )

        if not plotted_here:
            ax.text(
                0.5,
                0.5,
                "no completed runs yet",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="#6b7280",
            )

    fig.supxlabel(
        "Generation",
        fontweight="bold",
        fontsize=CONVERGENCE_FONT_SIZE,
        y=0.125,
    )

    if legend_handles:
        legend_methods = [method for method in METHODS if method in legend_handles]
        legend = fig.legend(
            handles=method_legend_handles(legend_methods),
            loc="lower center",
            bbox_to_anchor=(0.5, 0.025),
            ncol=len(legend_methods),
            frameon=False,
            handlelength=2.0,
            handleheight=0.75,
            columnspacing=1.35,
            borderaxespad=0.0,
        )
        style_convergence_legend(legend)

    fig.align_ylabels(axes)
    add_convergence_panel_label(fig)
    fig.subplots_adjust(
        left=0.085,
        right=0.995,
        top=0.90,
        bottom=0.27 if legend_handles else 0.17,
        wspace=0.22,
    )
    out_path = os.path.join(RESULTS_DIR, "convergence_by_optimizer_wide.png")
    fig.savefig(out_path, dpi=220, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"  plotted wide optimizer-grouped convergence curves: {plotted_total}")
    print(f"  saved: {out_path}")


def plot_best_airfoils_by_method(index: pd.DataFrame) -> None:
    airfoils = collect_best_airfoils(index)
    fig, axes = plt.subplots(
        len(METHODS),
        1,
        figsize=(8.4, 10.2),
        sharex=True,
        sharey=True,
    )

    if airfoils:
        all_y = np.concatenate([coords[1] for coords in airfoils.values()])
        y_min = float(np.nanmin(all_y))
        y_max = float(np.nanmax(all_y))
        y_pad = max(0.006, 0.08 * (y_max - y_min))
    else:
        y_min, y_max, y_pad = -0.08, 0.08, 0.01

    plotted_total = 0
    legend_handles = {}
    for ax, method in zip(axes, METHODS):
        plotted_here = 0
        for optimizer in OPTIMIZERS:
            coords = airfoils.get((method, optimizer))
            if coords is None:
                continue

            x, z = coords
            line, = ax.plot(
                x,
                z,
                linewidth=AIRFOIL_LINEWIDTH,
                label=OPTIMIZER_LABELS.get(optimizer, optimizer.upper()),
            )
            legend_handles.setdefault(optimizer, line)
            plotted_here += 1
            plotted_total += 1

        ax.set_title(
            METHOD_LABELS[method],
            fontweight="bold",
            fontsize=AIRFOIL_TITLE_FONT_SIZE,
            pad=5,
        )
        ax.set_ylabel(
            "z/c",
            fontweight="bold",
            fontsize=AIRFOIL_FONT_SIZE,
            labelpad=6,
        )
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, axis="both", alpha=0.3, linewidth=0.75)
        ax.tick_params(
            axis="both",
            which="major",
            direction="in",
            width=AIRFOIL_SPINE_LINEWIDTH,
            length=5,
            labelsize=AIRFOIL_TICK_FONT_SIZE,
        )
        for spine in ax.spines.values():
            spine.set_linewidth(AIRFOIL_SPINE_LINEWIDTH)

        if not plotted_here:
            ax.text(
                0.5,
                0.5,
                "no completed airfoils yet",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="#6b7280",
            )

    axes[-1].set_xlabel("x/c", fontweight="bold", fontsize=AIRFOIL_FONT_SIZE)
    if legend_handles:
        handles = [
            legend_handles[optimizer]
            for optimizer in OPTIMIZERS
            if optimizer in legend_handles
        ]
        labels = [
            OPTIMIZER_LABELS.get(optimizer, optimizer.upper())
            for optimizer in OPTIMIZERS
            if optimizer in legend_handles
        ]
        legend = fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.992),
            ncol=len(handles),
            frameon=True,
            borderaxespad=0.0,
        )
        legend.get_frame().set_linewidth(AIRFOIL_SPINE_LINEWIDTH)
        for text in legend.get_texts():
            text.set_fontweight("bold")
            text.set_fontsize(AIRFOIL_FONT_SIZE)

    fig.align_ylabels(axes)
    fig.subplots_adjust(
        left=0.115,
        right=0.985,
        top=0.925 if legend_handles else 0.965,
        bottom=0.075,
        hspace=0.70,
    )
    out_path = os.path.join(RESULTS_DIR, "best_airfoils_by_method.png")
    fig.savefig(out_path, dpi=220, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"  plotted best airfoils: {plotted_total}")
    print(f"  saved: {out_path}")


def collect_best_airfoils(index: pd.DataFrame) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]:
    airfoils: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}

    for method in METHODS:
        for optimizer in OPTIMIZERS:
            rows = index[
                (index["method"] == method)
                & (index["optimizer"] == optimizer)
                & (index["status"] == "ok")
            ]
            if rows.empty:
                continue

            row = rows.iloc[-1]
            conv_path = resolve_convergence_path(row)
            if conv_path is None:
                print(f"  missing convergence: {method} / {optimizer}")
                continue
            
            history = pd.read_csv(conv_path)
            if "feasible" not in history.columns:
                raise ValueError(f"Missing feasible column: {conv_path}")
            
            final_feasible = (
                not history.empty
                and str(history.iloc[-1]["feasible"]).strip().lower()
                in {"1", "true", "yes", "y"}
            )
            if not final_feasible:
                #print(f"  skipping infeasible airfoil: {method} / {optimizer}")
                continue
            
            best_path = resolve_best_airfoil_path(row)
            if best_path is None:
                print(f"  missing best airfoil: {method} / {optimizer}")
                continue

            try:
                airfoils[(method, optimizer)] = load_airfoil_xy(best_path)
            except Exception as exc:
                print(f"  unreadable best airfoil {best_path}: {exc}")

    return airfoils


def resolve_best_airfoil_path(row: pd.Series) -> Path | None:
    candidates: list[Path] = []
    opt_dir_raw = row.get("opt_dir", "")
    if not pd.isna(opt_dir_raw):
        opt_dir = Path(str(opt_dir_raw).strip())
        if str(opt_dir):
            pointer = opt_dir / "best_airfoil.dat"
            if pointer.is_file():
                lines = [
                    line.strip()
                    for line in pointer.read_text(errors="ignore").splitlines()
                    if line.strip()
                ]
                if lines:
                    target = lines[-1]
                    target_path = Path(target)
                    candidates.append(target_path)
                    if not target_path.is_absolute():
                        candidates.append(opt_dir / target_path)

    log_raw = row.get("log_file", "")
    if not pd.isna(log_raw):
        log_path = Path(str(log_raw).strip())
        if log_path.is_file():
            for line in reversed(log_path.read_text(errors="ignore").splitlines()):
                match = re.search(r"Best airfoil:\s*(.+)$", line)
                if match:
                    candidates.append(Path(match.group(1).strip()))
                    break

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def load_airfoil_xy(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    rows: list[tuple[float, float]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = re.split(r"[\s,]+", line)
            if len(parts) < 2:
                continue

            try:
                rows.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue

    if len(rows) < 2:
        raise ValueError("not enough coordinate rows")

    coords = np.asarray(rows, dtype=float)
    return coords[:, 0], coords[:, 1]


def resolve_convergence_path(
    row: pd.Series,
    convergence_dir: str | Path = CONVERGENCE_DIR,
) -> str | None:
    raw = row.get("convergence_csv", "")
    if pd.isna(raw):
        raw = ""

    path = str(raw).strip()
    candidates: list[Path] = []
    convergence_dir = Path(convergence_dir)

    if path:
        candidates.append(Path(path))
        candidates.append(convergence_dir / Path(path).name)

    method = str(row.get("method", "")).strip()
    optimizer = str(row.get("optimizer", "")).strip()
    if method and optimizer:
        candidates.append(convergence_dir / f"{method}__{optimizer}.csv")

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


def load_best_ever_convergence(path: str | Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"  unreadable convergence {path}: {exc}")
        return pd.DataFrame(columns=["generation", "best_fitness"])

    if df.shape[1] < 2 or df.empty:
        print(f"  empty convergence: {path}")
        return pd.DataFrame(columns=["generation", "best_fitness"])

    if IS_DAE11_PLOT:
        if "feasible" not in df.columns:
            raise SystemExit(
                "DAE-11 convergence data must contain the feasible column: "
                f"{path}\nRe-run with the updated core/fitness.py."
            )
        feasible = (
            df["feasible"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin({"1", "true", "yes", "y"})
        )
        df = df.loc[feasible].copy()
        if df.empty:
            return pd.DataFrame(columns=["generation", "best_fitness"])

    generation_column = "generation" if "generation" in df.columns else df.columns[0]
    if "cd" in df.columns:
        objective_column = "cd"
    elif "best_historic_objective" in df.columns:
        objective_column = "best_historic_objective"
    elif "best_objective" in df.columns:
        objective_column = "best_objective"
    elif "best_fitness" in df.columns:
        if REQUIRE_RAW_OBJECTIVE:
            raise SystemExit(
                "Raw-objective convergence data is required, but this legacy "
                f"file only contains best_fitness: {path}\n"
                "Re-run with the updated core/fitness.py."
            )
        # In the DAE-11 schema this compatibility column is pure Cd. Legacy
        # files also reach this branch because they contain no raw diagnostic.
        objective_column = "best_fitness"
    else:
        objective_column = df.columns[1]

    df = df[[generation_column, objective_column]].copy()
    df.columns = ["generation", "best_fitness"]
    df["generation"] = pd.to_numeric(df["generation"], errors="coerce")
    df["best_fitness"] = pd.to_numeric(df["best_fitness"], errors="coerce")
    df = df.dropna(subset=["generation", "best_fitness"])
    df = df[df["best_fitness"] > 0.0]
    df.loc[df["best_fitness"] >= PENALTY_THRESHOLD, "best_fitness"] = np.nan
    df = df.dropna(subset=["best_fitness"])
    if df.empty:
        return df

    df = df.sort_values("generation", kind="mergesort")
    df["best_fitness"] = df["best_fitness"].cummin()
    return df


def best_ever_from_convergence(
    row: pd.Series,
    convergence_dir: str | Path = CONVERGENCE_DIR,
) -> float | None:
    conv_path = resolve_convergence_path(row, convergence_dir)
    if conv_path is None:
        return None

    df = load_best_ever_convergence(conv_path)
    if df.empty:
        return None
    return float(df["best_fitness"].iloc[-1])


def write_cauchy_convergence_summary(index: pd.DataFrame) -> None:
    n_points = infer_n_points()
    rows = []

    for method in METHODS:
        for optimizer in OPTIMIZERS:
            completed = index[
                (index["method"] == method)
                & (index["optimizer"] == optimizer)
                & (index["status"] == "ok")
            ]

            if completed.empty:
                rows.append({
                    "method": method,
                    "optimizer": optimizer,
                    "n_points": n_points,
                    "status": "missing",
                    "converged": False,
                })
                continue

            row = completed.iloc[-1]
            conv_path = resolve_convergence_path(row)
            if conv_path is None:
                rows.append({
                    "method": method,
                    "optimizer": optimizer,
                    "n_points": n_points,
                    "status": "missing_convergence_csv",
                    "converged": False,
                })
                continue

            df = load_best_ever_convergence(conv_path)
            if df.empty:
                rows.append({
                    "method": method,
                    "optimizer": optimizer,
                    "n_points": n_points,
                    "status": "empty_convergence",
                    "converged": False,
                    "convergence_csv": conv_path,
                })
                continue

            final_best = float(df["best_fitness"].iloc[-1])
            eps = cauchy_epsilon(n_points, final_best)
            conv = cauchy_convergence_point(df, eps)
            first_generation = float(df["generation"].iloc[0])
            total_generations = float(df["generation"].iloc[-1])

            rows.append({
                "method": method,
                "optimizer": optimizer,
                "n_points": n_points,
                "status": "ok",
                "converged": conv is not None,
                "convergence_generation": np.nan if conv is None else conv["generation"],
                "iterations_to_converge": np.nan if conv is None else conv["generation"] - first_generation,
                "total_generations": total_generations,
                "convergence_fraction": (
                    np.nan if conv is None or total_generations <= first_generation
                    else (conv["generation"] - first_generation) / (total_generations - first_generation)
                ),
                "final_best": final_best,
                "cost_at_convergence": np.nan if conv is None else conv["best_fitness"],
                "remaining_improvement": np.nan if conv is None else conv["remaining_improvement"],
                "cauchy_epsilon": eps,
                "convergence_csv": conv_path,
            })

    out_path = os.path.join(RESULTS_DIR, "convergence_cauchy_summary.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"  saved: {out_path}")


def cauchy_epsilon(n_points: int, final_best: float) -> float:
    eps_abs_env = os.environ.get("PLOT_CAUCHY_EPS_ABS")
    if eps_abs_env is None:
        eps_abs = CAUCHY_DRAG_COUNT * n_points
    else:
        eps_abs = float(eps_abs_env)

    return eps_abs + CAUCHY_EPS_REL * max(abs(final_best), 1.0e-12)


def cauchy_convergence_point(
    df: pd.DataFrame,
    eps: float,
) -> dict[str, float] | None:
    values = df["best_fitness"].to_numpy(dtype=float)
    generations = df["generation"].to_numpy(dtype=float)
    final_best = values[-1]

    remaining_improvement = values - final_best
    hits = np.flatnonzero(remaining_improvement <= eps)
    if len(hits) == 0:
        return None

    idx = int(hits[0])
    return {
        "generation": float(generations[idx]),
        "best_fitness": float(values[idx]),
        "remaining_improvement": float(remaining_improvement[idx]),
    }


def infer_n_points() -> int:
    n_points = infer_n_points_from_dir(Path(RESULTS_DIR))
    if n_points is None:
        n_points = int(os.environ.get("PLOT_N_POINTS", "1"))

    if n_points not in INITIAL_DRAG_SUM_BY_N_POINTS:
        known = ", ".join(str(key) for key in sorted(INITIAL_DRAG_SUM_BY_N_POINTS))
        raise SystemExit(
            f"No initial drag sum configured for {n_points}-point study. "
            f"Known point counts: {known}"
        )
    return n_points


def infer_n_points_from_dir(results_dir: Path) -> int | None:
    match = re.search(r"matrix_5x5_(\d+)point", str(results_dir))
    if match:
        return int(match.group(1))
    return None


def drag_reduction_percent(best_drag_sum: float, initial_drag_sum: float) -> float:
    if initial_drag_sum <= 0.0:
        raise SystemExit("INITIAL_DRAG_SUM must be positive.")
    reduction = 100.0 * (initial_drag_sum - float(best_drag_sum)) / initial_drag_sum
    return max(0.0, reduction)


if __name__ == "__main__":
    main()
