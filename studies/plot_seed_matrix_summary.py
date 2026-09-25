"""
Aggregate final matrix results across multiple seeded point studies.

Run from optimizer/:
    python3 studies/plot_seed_matrix_summary.py

For any custom ``<name>_seed_<integer>`` directory family:
    SEED_STUDY_NAME=<name> python3 studies/plot_seed_matrix_summary.py

With SEED_STUDY_NAME set, point counts are discovered automatically, the
heatmap reports the final objective, and output is written to
``studies/<name>_seed_summary`` unless explicitly overridden.

For DAE-11 studies, either include ``dae11`` in ``SEED_STUDY_NAME`` or set
``SEED_SUMMARY_REFERENCE=dae11``.  The one-point heatmap then reports drag
reduction relative to the DAE-11 baseline Cd of 0.050505.

For DAE-11 constrained or CL-only results stored directly under each seed:
    SEED_STUDY_NAME=dae11 SEED_STUDY_CASE=constrained \
        python3 studies/plot_seed_matrix_summary.py
Set SEED_STUDY_CASE=unconstrained for the CL-only matrix.

Inputs:
    studies/point_studies_full_study_seed_*/matrix_5x5_{1,2,4}point/study_index.csv
    studies/point_studies_full_study_seed_*/matrix_5x5_{1,2,4}point/convergence/*.csv

Outputs:
    studies/seed_matrix_summary/final_results_matrix_1point.png
    studies/seed_matrix_summary/final_results_matrix_2point.png
    studies/seed_matrix_summary/final_results_matrix_4point.png
    studies/seed_matrix_summary/final_results_matrix_summary.csv
    studies/seed_matrix_summary/convergence_by_optimizer_wide_seed_mean_1point.png
    studies/seed_matrix_summary/convergence_by_optimizer_wide_seed_mean_2point.png
    studies/seed_matrix_summary/convergence_by_optimizer_wide_seed_mean_4point.png
    studies/seed_matrix_summary/best_seed_airfoil_1point.png
    studies/seed_matrix_summary/best_seed_airfoil_2point.png
    studies/seed_matrix_summary/best_seed_airfoil_4point.png
    studies/seed_matrix_summary/best_seed_airfoil_selection.csv
    studies/seed_matrix_summary/convergence_seed_summary.csv
    studies/seed_matrix_summary/convergence_cauchy_seed_summary.csv
    studies/seed_matrix_summary/convergence_cauchy_seed_raw.csv
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter
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
SEED_STUDY_NAME = os.environ.get("SEED_STUDY_NAME", "").strip()
SEED_STUDY_CASE = os.environ.get("SEED_STUDY_CASE", "").strip()
SEED_SUMMARY_REFERENCE = os.environ.get(
    "SEED_SUMMARY_REFERENCE",
    "",
).strip().lower()
_NORMALIZED_STUDY_NAME = re.sub(r"[^a-z0-9]+", "", SEED_STUDY_NAME.lower())
_NORMALIZED_REFERENCE = re.sub(r"[^a-z0-9]+", "", SEED_SUMMARY_REFERENCE)
IS_DAE11_SUMMARY = (
    _NORMALIZED_REFERENCE in {"dae11", "dreladae11"}
    or "dae11" in _NORMALIZED_STUDY_NAME
)
ALLOW_LEGACY_DAE11_FITNESS = (
    os.environ.get("SEED_ALLOW_LEGACY_DAE11_FITNESS", "0").strip().lower()
    in {"1", "true", "yes", "on"}
)
DEFAULT_SEED_STUDY_GLOB = (
    f"{SEED_STUDY_NAME}_seed_*"
    if SEED_STUDY_NAME
    else "point_studies_full_study_seed_*"
)
SEED_STUDY_GLOB = os.environ.get(
    "SEED_STUDY_GLOB",
    DEFAULT_SEED_STUDY_GLOB,
)
SEED_STUDY_REGEX = os.environ.get("SEED_STUDY_REGEX", "").strip()
ALLOW_SEED_SUFFIX = os.environ.get("SEED_ALLOW_SUFFIX", "0").strip() == "1"
SUMMARY_METRIC = os.environ.get(
    "SEED_SUMMARY_METRIC",
    "reduction" if IS_DAE11_SUMMARY or not SEED_STUDY_NAME else "objective",
).strip().lower()
PLOT_BEST_SEED_AIRFOILS = (
    os.environ.get(
        "SEED_PLOT_BEST_AIRFOILS",
        "0" if SEED_STUDY_NAME else "1",
    ).strip().lower()
    not in {"0", "false", "no", "off"}
)
OUTPUT_DIR = Path(os.environ.get(
    "SEED_SUMMARY_DIR",
    str(
        SCRIPT_DIR
        / (
            f"{SEED_STUDY_NAME}_{SEED_STUDY_CASE}_seed_summary"
            if SEED_STUDY_NAME and SEED_STUDY_CASE
            else f"{SEED_STUDY_NAME}_seed_summary"
            if SEED_STUDY_NAME
            else "seed_matrix_summary"
        )
    ),
))
PENALTY_THRESHOLD = float(os.environ.get("PLOT_PENALTY_THRESHOLD", 99.0))
HEATMAP_CMAP = os.environ.get(
    "PLOT_HEATMAP_CMAP",
    "RdYlGn_r" if SUMMARY_METRIC == "objective" else "RdYlGn",
)
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
CONVERGENCE_XMIN = float(os.environ.get("PLOT_CONV_XMIN", 0.0))
CONVERGENCE_XMAX = float(os.environ.get("PLOT_CONV_XMAX", 500.0))
CONVERGENCE_YMIN = os.environ.get("PLOT_CONV_YMIN")
CONVERGENCE_YMAX = os.environ.get("PLOT_CONV_YMAX")
CONVERGENCE_USE_POINT_LIMITS = (
    os.environ.get(
        "PLOT_CONV_USE_POINT_LIMITS",
        "0" if SEED_STUDY_NAME else "1",
    ).strip().lower()
    not in {"0", "false", "no", "off"}
)
CONVERGENCE_YLABEL = os.environ.get(
    "PLOT_CONV_YLABEL",
    (
        r"Best-historic $C_D$"
        if IS_DAE11_SUMMARY
        else r"Best-historic objective, $\mathcal{F}^*$"
    ),
)
CONVERGENCE_STD_ALPHA = float(os.environ.get("SEED_CONV_STD_ALPHA", 0.0))
CAUCHY_DRAG_COUNT = float(os.environ.get("PLOT_CAUCHY_DRAG_COUNT", 1.0e-4))
CAUCHY_EPS_REL = float(os.environ.get("PLOT_CAUCHY_EPS_REL", 0.0))
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
REFERENCE_AIRFOIL_DIR = Path(os.environ.get(
    "REFERENCE_AIRFOIL_DIR",
    "/home/erene/simulations/rAIFoil",
))
RAE2822_BASELINE_PATH = Path(os.environ.get(
    "RAE2822_BASELINE_PATH",
    str(REFERENCE_AIRFOIL_DIR / "RAE2822.dat"),
))
DRELA_OPTIMUM_PATHS = {
    1: Path(os.environ.get(
        "DRELA_OPT_1POINT_PATH",
        str(REFERENCE_AIRFOIL_DIR / "opt_1point.dat"),
    )),
    2: Path(os.environ.get(
        "DRELA_OPT_2POINT_PATH",
        str(REFERENCE_AIRFOIL_DIR / "opt_2point.dat"),
    )),
    4: Path(os.environ.get(
        "DRELA_OPT_4POINT_PATH",
        str(REFERENCE_AIRFOIL_DIR / "opt_4point.dat"),
    )),
}

# Edit these values to the baseline drag sum you want to compare against.
# For multi-point studies, use the sum of the baseline Cd values across
# the objective Mach points, e.g. Cd(M1) + Cd(M2) [+ Cd(M3) + Cd(M4)].
DAE11_BASELINE_CD = float(os.environ.get("DAE11_BASELINE_CD", "0.050505"))
_ONE_POINT_BASELINE_DEFAULT = (
    str(DAE11_BASELINE_CD) if IS_DAE11_SUMMARY else "0.103416"
)
INITIAL_DRAG_SUM_BY_N_POINTS = {
    1: float(os.environ.get(
        "INITIAL_DRAG_SUM_1POINT",
        _ONE_POINT_BASELINE_DEFAULT,
    )),
    2: float(os.environ.get("INITIAL_DRAG_SUM_2POINT", "0.14673")),
    4: float(os.environ.get("INITIAL_DRAG_SUM_4POINT", "0.346914")),
}


def main() -> None:
    if SEED_STUDY_CASE and SEED_STUDY_CASE not in {"constrained", "unconstrained"}:
        raise SystemExit("SEED_STUDY_CASE must be 'constrained' or 'unconstrained'.")
    if SUMMARY_METRIC not in {"reduction", "objective"}:
        raise SystemExit(
            "SEED_SUMMARY_METRIC must be 'reduction' or 'objective', "
            f"got {SUMMARY_METRIC!r}."
        )
    point_counts = requested_point_counts()
    seed_dirs = discover_seed_dirs()
    if not seed_dirs:
        raise SystemExit(
            "No seed study directories found.\n"
            f"Looked in: {SCRIPT_DIR / SEED_STUDY_GLOB}\n"
            f"Seed-name regex: {SEED_STUDY_REGEX or 'default full-study pattern'}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = build_summary(seed_dirs, point_counts)
    summary_path = OUTPUT_DIR / "final_results_matrix_summary.csv"
    summary.to_csv(summary_path, index=False)

    convergence_summary = build_convergence_summary(seed_dirs, point_counts)
    convergence_summary_path = OUTPUT_DIR / "convergence_seed_summary.csv"
    convergence_summary.to_csv(convergence_summary_path, index=False)

    cauchy_raw, cauchy_summary = build_cauchy_seed_summary(seed_dirs, point_counts)
    cauchy_raw_path = OUTPUT_DIR / "convergence_cauchy_seed_raw.csv"
    cauchy_summary_path = OUTPUT_DIR / "convergence_cauchy_seed_summary.csv"
    cauchy_raw.to_csv(cauchy_raw_path, index=False)
    cauchy_summary.to_csv(cauchy_summary_path, index=False)

    vmin, vmax = shared_heatmap_limits(summary)
    for n_points in point_counts:
        plot_summary_heatmap(summary, n_points, vmin, vmax)
        plot_seed_convergence_by_optimizer_wide(convergence_summary, n_points)
    if PLOT_BEST_SEED_AIRFOILS:
        plot_best_seed_airfoils(summary, seed_dirs, point_counts)

    seeds = ", ".join(str(seed) for seed, _ in seed_dirs)
    print(f"Aggregated seeds: {seeds}")
    if SUMMARY_METRIC == "reduction":
        baselines = ", ".join(
            f"{n_points}-point={INITIAL_DRAG_SUM_BY_N_POINTS[n_points]:.6f}"
            for n_points in point_counts
        )
        print(f"Drag-reduction baseline(s): {baselines}")
        print("Drag reduction [%] = 100 * (baseline - final objective) / baseline")
    print(f"Wrote summary CSV: {summary_path}")
    print(f"Wrote convergence CSV: {convergence_summary_path}")
    print(f"Wrote Cauchy raw CSV: {cauchy_raw_path}")
    print(f"Wrote Cauchy seed-summary CSV: {cauchy_summary_path}")
    print(f"Wrote plots to: {OUTPUT_DIR}")


def requested_point_counts() -> tuple[int, ...]:
    raw = os.environ.get("SEED_POINT_STUDIES")
    if raw is None and SEED_STUDY_CASE:
        values = [1]
    elif raw is None and SEED_STUDY_NAME:
        discovered = set()
        point_re = re.compile(r"^matrix_5x5_(\d+)point$")
        for seed_dir in SCRIPT_DIR.glob(SEED_STUDY_GLOB):
            if not seed_dir.is_dir():
                continue
            for point_dir in seed_dir.glob("matrix_5x5_*point"):
                match = point_re.match(point_dir.name)
                if match is not None:
                    discovered.add(int(match.group(1)))
        if not discovered:
            raise SystemExit(
                f"No matrix_5x5_<n>point directories found under "
                f"{SCRIPT_DIR / SEED_STUDY_GLOB}"
            )
        values = sorted(discovered)
    else:
        raw = "1 2 4" if raw is None else raw
        values = [int(part) for part in raw.replace(",", " ").split()]
    unknown = (
        [value for value in values if value not in INITIAL_DRAG_SUM_BY_N_POINTS]
        if SUMMARY_METRIC == "reduction" else []
    )
    if unknown:
        known = ", ".join(str(key) for key in sorted(INITIAL_DRAG_SUM_BY_N_POINTS))
        raise SystemExit(
            f"No initial drag sum configured for point count(s): {unknown}. "
            f"Known point counts: {known}"
        )
    if SEED_STUDY_CASE and values != [1]:
        raise SystemExit("SEED_STUDY_CASE supports only the one-point matrix.")
    return tuple(values)


def seed_point_dir(seed_dir: Path, n_points: int) -> Path:
    if SEED_STUDY_CASE:
        return seed_dir / SEED_STUDY_CASE
    return seed_dir / f"matrix_5x5_{n_points}point"


def discover_seed_dirs() -> list[tuple[int, Path]]:
    if SEED_STUDY_REGEX:
        seed_re = re.compile(SEED_STUDY_REGEX)
    elif SEED_STUDY_NAME:
        seed_re = re.compile(
            rf"^{re.escape(SEED_STUDY_NAME)}_seed_(\d+)$"
        )
    elif ALLOW_SEED_SUFFIX:
        seed_re = re.compile(r"^point_studies_full_study_seed_(\d+)(?:$|_)")
    else:
        seed_re = re.compile(r"^point_studies_full_study_seed_(\d+)$")

    seed_dirs: list[tuple[int, Path]] = []
    for path in sorted(SCRIPT_DIR.glob(SEED_STUDY_GLOB)):
        if not path.is_dir():
            continue

        match = seed_re.match(path.name)
        if match is None:
            print(f"Skipping non-seed study directory: {path}")
            continue

        seed_dirs.append((int(match.group(1)), path))

    seed_dirs.sort(key=lambda item: item[0])
    return seed_dirs


def build_summary(
    seed_dirs: list[tuple[int, Path]],
    point_counts: tuple[int, ...],
) -> pd.DataFrame:
    rows = []
    for n_points in point_counts:
        values_by_case: dict[tuple[str, str], list[tuple[int, float, float]]] = {
            (method, optimizer): []
            for method in METHODS
            for optimizer in OPTIMIZERS
        }

        for seed, seed_dir in seed_dirs:
            point_dir = seed_point_dir(seed_dir, n_points)
            index_path = point_dir / "study_index.csv"
            if not index_path.is_file():
                print(f"Missing {n_points}-point index for seed {seed}: {index_path}")
                continue

            try:
                index = pd.read_csv(index_path)
            except Exception as exc:
                print(f"Unreadable index for seed {seed}: {index_path}: {exc}")
                continue

            normalize_index(index)
            for method in METHODS:
                for optimizer in OPTIMIZERS:
                    row = latest_ok_row(index, method, optimizer)
                    if row is None:
                        continue

                    value = best_drag_sum(row, point_dir / "convergence")
                    baseline = INITIAL_DRAG_SUM_BY_N_POINTS[n_points]
                    if IS_DAE11_SUMMARY and (
                        value is None
                        or not np.isfinite(value)
                        or value >= baseline
                    ):
                        value = baseline
                    elif value is None or not np.isfinite(value):
                        continue
                    if value >= PENALTY_THRESHOLD:
                        if IS_DAE11_SUMMARY:
                            value = baseline
                        else:
                            continue

                    baseline_key = f"INITIAL_DRAG_SUM_{n_points}POINT"
                    if SUMMARY_METRIC == "objective" and baseline_key not in os.environ:
                        reduction = np.nan
                    else:
                        reduction = drag_reduction_percent(
                            value,
                            INITIAL_DRAG_SUM_BY_N_POINTS[n_points],
                        )
                    values_by_case[(method, optimizer)].append(
                        (seed, float(value), reduction)
                    )

        for method in METHODS:
            for optimizer in OPTIMIZERS:
                seed_values = values_by_case[(method, optimizer)]
                objectives = np.asarray(
                    [objective for _, objective, _ in seed_values], dtype=float
                )
                reductions = np.asarray(
                    [reduction for _, _, reduction in seed_values], dtype=float
                )
                seeds = [seed for seed, _, _ in seed_values]

                finite_reductions = reductions[np.isfinite(reductions)]
                if finite_reductions.size:
                    mean = float(np.mean(finite_reductions))
                    std = (
                        float(np.std(finite_reductions, ddof=1))
                        if finite_reductions.size > 1 else 0.0
                    )
                    vmin = float(np.min(finite_reductions))
                    vmax = float(np.max(finite_reductions))
                else:
                    mean = np.nan
                    std = np.nan
                    vmin = np.nan
                    vmax = np.nan

                if objectives.size:
                    objective_mean = float(np.mean(objectives))
                    objective_std = (
                        float(np.std(objectives, ddof=1))
                        if objectives.size > 1 else 0.0
                    )
                    objective_min = float(np.min(objectives))
                    objective_max = float(np.max(objectives))
                else:
                    objective_mean = np.nan
                    objective_std = np.nan
                    objective_min = np.nan
                    objective_max = np.nan

                rows.append({
                    "n_points": n_points,
                    "method": method,
                    "optimizer": optimizer,
                    "n_seeds": int(reductions.size),
                    "seeds": " ".join(str(seed) for seed in seeds),
                    "drag_reduction_baseline": (
                        INITIAL_DRAG_SUM_BY_N_POINTS[n_points]
                        if SUMMARY_METRIC == "reduction" else np.nan
                    ),
                    "mean_final_objective": objective_mean,
                    "std_final_objective": objective_std,
                    "min_final_objective": objective_min,
                    "max_final_objective": objective_max,
                    "mean_drag_reduction_percent": mean,
                    "std_drag_reduction_percent": std,
                    "min_drag_reduction_percent": vmin,
                    "max_drag_reduction_percent": vmax,
                })

    return pd.DataFrame(rows)


def build_convergence_summary(
    seed_dirs: list[tuple[int, Path]],
    point_counts: tuple[int, ...],
) -> pd.DataFrame:
    columns = [
        "n_points",
        "method",
        "optimizer",
        "generation",
        "n_seeds",
        "seeds",
        "mean_best_fitness",
        "std_best_fitness",
        "min_best_fitness",
        "max_best_fitness",
    ]
    rows = []

    for n_points in point_counts:
        curves_by_case: dict[tuple[str, str], list[tuple[int, pd.DataFrame]]] = {
            (method, optimizer): []
            for method in METHODS
            for optimizer in OPTIMIZERS
        }

        for seed, seed_dir in seed_dirs:
            point_dir = seed_point_dir(seed_dir, n_points)
            index_path = point_dir / "study_index.csv"
            if not index_path.is_file():
                continue

            try:
                index = pd.read_csv(index_path)
            except Exception as exc:
                print(f"Unreadable index for seed {seed}: {index_path}: {exc}")
                continue

            normalize_index(index)
            for method in METHODS:
                for optimizer in OPTIMIZERS:
                    row = latest_ok_row(index, method, optimizer)
                    if row is None:
                        continue

                    conv_path = resolve_convergence_path(row, point_dir / "convergence")
                    if conv_path is None:
                        continue

                    df = load_best_ever_convergence(conv_path)
                    if df.empty:
                        continue

                    curves_by_case[(method, optimizer)].append((seed, df))

        for method in METHODS:
            for optimizer in OPTIMIZERS:
                seed_curves = curves_by_case[(method, optimizer)]
                aggregate = aggregate_seed_curves(seed_curves)
                if aggregate.empty:
                    continue

                seeds = " ".join(str(seed) for seed, _ in seed_curves)
                for _, row in aggregate.iterrows():
                    rows.append({
                        "n_points": n_points,
                        "method": method,
                        "optimizer": optimizer,
                        "generation": int(row["generation"]),
                        "n_seeds": int(row["n_seeds"]),
                        "seeds": seeds,
                        "mean_best_fitness": float(row["mean_best_fitness"]),
                        "std_best_fitness": float(row["std_best_fitness"]),
                        "min_best_fitness": float(row["min_best_fitness"]),
                        "max_best_fitness": float(row["max_best_fitness"]),
                    })

    return pd.DataFrame(rows, columns=columns)


def build_cauchy_seed_summary(
    seed_dirs: list[tuple[int, Path]],
    point_counts: tuple[int, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_columns = [
        "seed",
        "n_points",
        "method",
        "optimizer",
        "status",
        "converged",
        "convergence_generation",
        "iterations_to_converge",
        "total_generations",
        "convergence_fraction",
        "final_best",
        "cost_at_convergence",
        "remaining_improvement",
        "cauchy_epsilon",
        "convergence_csv",
    ]
    raw_rows = []

    for n_points in point_counts:
        for seed, seed_dir in seed_dirs:
            point_dir = seed_point_dir(seed_dir, n_points)
            index_path = point_dir / "study_index.csv"
            if not index_path.is_file():
                continue

            try:
                index = pd.read_csv(index_path)
            except Exception as exc:
                print(f"Unreadable index for seed {seed}: {index_path}: {exc}")
                continue

            normalize_index(index)
            for method in METHODS:
                for optimizer in OPTIMIZERS:
                    row = latest_ok_row(index, method, optimizer)
                    if row is None:
                        raw_rows.append(cauchy_missing_row(
                            seed,
                            n_points,
                            method,
                            optimizer,
                            "missing_completed_run",
                        ))
                        continue

                    conv_path = resolve_convergence_path(row, point_dir / "convergence")
                    if conv_path is None:
                        raw_rows.append(cauchy_missing_row(
                            seed,
                            n_points,
                            method,
                            optimizer,
                            "missing_convergence_csv",
                        ))
                        continue

                    df = load_best_ever_convergence(conv_path)
                    if df.empty:
                        raw_rows.append(cauchy_missing_row(
                            seed,
                            n_points,
                            method,
                            optimizer,
                            "empty_convergence",
                            str(conv_path),
                        ))
                        continue

                    final_best = float(df["best_fitness"].iloc[-1])
                    eps = cauchy_epsilon(n_points, final_best)
                    conv = cauchy_convergence_point(df, eps)
                    first_generation = float(df["generation"].iloc[0])
                    total_generations = float(df["generation"].iloc[-1])

                    raw_rows.append({
                        "seed": seed,
                        "n_points": n_points,
                        "method": method,
                        "optimizer": optimizer,
                        "status": "ok",
                        "converged": conv is not None,
                        "convergence_generation": np.nan if conv is None else conv["generation"],
                        "iterations_to_converge": (
                            np.nan if conv is None
                            else conv["generation"] - first_generation
                        ),
                        "total_generations": total_generations,
                        "convergence_fraction": (
                            np.nan if conv is None or total_generations <= first_generation
                            else (conv["generation"] - first_generation)
                            / (total_generations - first_generation)
                        ),
                        "final_best": final_best,
                        "cost_at_convergence": np.nan if conv is None else conv["best_fitness"],
                        "remaining_improvement": (
                            np.nan if conv is None else conv["remaining_improvement"]
                        ),
                        "cauchy_epsilon": eps,
                        "convergence_csv": str(conv_path),
                    })

    raw = pd.DataFrame(raw_rows, columns=raw_columns)
    return raw, aggregate_cauchy_table(raw)


def cauchy_missing_row(
    seed: int,
    n_points: int,
    method: str,
    optimizer: str,
    status: str,
    convergence_csv: str = "",
) -> dict[str, object]:
    return {
        "seed": seed,
        "n_points": n_points,
        "method": method,
        "optimizer": optimizer,
        "status": status,
        "converged": False,
        "convergence_generation": np.nan,
        "iterations_to_converge": np.nan,
        "total_generations": np.nan,
        "convergence_fraction": np.nan,
        "final_best": np.nan,
        "cost_at_convergence": np.nan,
        "remaining_improvement": np.nan,
        "cauchy_epsilon": np.nan,
        "convergence_csv": convergence_csv,
    }


def aggregate_cauchy_table(raw: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "n_points",
        "method",
        "optimizer",
        "n_seed_dirs",
        "n_completed_runs",
        "n_converged",
        "convergence_rate",
        "seeds",
        "converged_seeds",
        "mean_iterations_to_converge",
        "std_iterations_to_converge",
        "mean_convergence_generation",
        "std_convergence_generation",
        "mean_convergence_fraction",
        "std_convergence_fraction",
        "mean_final_best",
        "std_final_best",
        "mean_cost_at_convergence",
        "std_cost_at_convergence",
        "mean_remaining_improvement",
        "std_remaining_improvement",
        "mean_cauchy_epsilon",
    ]
    if raw.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for n_points in sorted(raw["n_points"].dropna().astype(int).unique()):
        point_raw = raw[raw["n_points"] == n_points]
        for method in METHODS:
            for optimizer in OPTIMIZERS:
                case = point_raw[
                    (point_raw["method"] == method)
                    & (point_raw["optimizer"] == optimizer)
                ].copy()
                if case.empty:
                    continue

                completed = case[case["status"] == "ok"].copy()
                converged = completed[completed["converged"].astype(bool)].copy()
                seeds = sorted(case["seed"].dropna().astype(int).unique())
                converged_seeds = sorted(converged["seed"].dropna().astype(int).unique())

                rows.append({
                    "n_points": n_points,
                    "method": method,
                    "optimizer": optimizer,
                    "n_seed_dirs": int(len(seeds)),
                    "n_completed_runs": int(len(completed)),
                    "n_converged": int(len(converged)),
                    "convergence_rate": (
                        np.nan if len(completed) == 0
                        else float(len(converged) / len(completed))
                    ),
                    "seeds": " ".join(str(seed) for seed in seeds),
                    "converged_seeds": " ".join(str(seed) for seed in converged_seeds),
                    "mean_iterations_to_converge": numeric_mean(converged, "iterations_to_converge"),
                    "std_iterations_to_converge": numeric_std(converged, "iterations_to_converge"),
                    "mean_convergence_generation": numeric_mean(converged, "convergence_generation"),
                    "std_convergence_generation": numeric_std(converged, "convergence_generation"),
                    "mean_convergence_fraction": numeric_mean(converged, "convergence_fraction"),
                    "std_convergence_fraction": numeric_std(converged, "convergence_fraction"),
                    "mean_final_best": numeric_mean(completed, "final_best"),
                    "std_final_best": numeric_std(completed, "final_best"),
                    "mean_cost_at_convergence": numeric_mean(converged, "cost_at_convergence"),
                    "std_cost_at_convergence": numeric_std(converged, "cost_at_convergence"),
                    "mean_remaining_improvement": numeric_mean(converged, "remaining_improvement"),
                    "std_remaining_improvement": numeric_std(converged, "remaining_improvement"),
                    "mean_cauchy_epsilon": numeric_mean(completed, "cauchy_epsilon"),
                })

    return pd.DataFrame(rows, columns=columns)


def numeric_mean(df: pd.DataFrame, column: str) -> float:
    if df.empty or column not in df.columns:
        return np.nan
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    return np.nan if values.empty else float(values.mean())


def numeric_std(df: pd.DataFrame, column: str) -> float:
    if df.empty or column not in df.columns:
        return np.nan
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    if len(values) <= 1:
        return 0.0 if len(values) == 1 else np.nan
    return float(values.std(ddof=1))


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
) -> Optional[dict[str, float]]:
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


def aggregate_seed_curves(
    seed_curves: list[tuple[int, pd.DataFrame]],
) -> pd.DataFrame:
    columns = [
        "generation",
        "n_seeds",
        "mean_best_fitness",
        "std_best_fitness",
        "min_best_fitness",
        "max_best_fitness",
    ]
    if not seed_curves:
        return pd.DataFrame(columns=columns)

    generations = sorted({
        int(round(float(generation)))
        for _, df in seed_curves
        for generation in df["generation"].dropna()
    })
    if not generations:
        return pd.DataFrame(columns=columns)

    aligned = pd.DataFrame(index=generations)
    for seed, df in seed_curves:
        curve = df.copy()
        curve["generation"] = pd.to_numeric(curve["generation"], errors="coerce")
        curve["best_fitness"] = pd.to_numeric(curve["best_fitness"], errors="coerce")
        curve = curve.dropna(subset=["generation", "best_fitness"])
        if curve.empty:
            continue

        curve["generation"] = curve["generation"].round().astype(int)
        series = (
            curve.sort_values("generation", kind="mergesort")
            .groupby("generation")["best_fitness"]
            .last()
        )
        aligned[str(seed)] = series.reindex(generations).ffill()

    if aligned.empty:
        return pd.DataFrame(columns=columns)

    result = pd.DataFrame({
        "generation": generations,
        "n_seeds": aligned.count(axis=1).astype(int).to_numpy(),
        "mean_best_fitness": aligned.mean(axis=1, skipna=True).to_numpy(),
        "std_best_fitness": aligned.std(axis=1, ddof=1, skipna=True).fillna(0.0).to_numpy(),
        "min_best_fitness": aligned.min(axis=1, skipna=True).to_numpy(),
        "max_best_fitness": aligned.max(axis=1, skipna=True).to_numpy(),
    })
    result = result[result["n_seeds"] > 0].reset_index(drop=True)
    return result


def normalize_index(index: pd.DataFrame) -> None:
    index["method"] = index["method"].astype(str).str.strip().str.lower()
    index["optimizer"] = index["optimizer"].astype(str).str.strip().str.lower()
    index["status"] = index["status"].astype(str).str.strip().str.lower()
    if "final_best" in index.columns:
        index["final_best"] = pd.to_numeric(index["final_best"], errors="coerce")


def latest_ok_row(
    index: pd.DataFrame,
    method: str,
    optimizer: str,
) -> Optional[pd.Series]:
    rows = index[
        (index["method"] == method)
        & (index["optimizer"] == optimizer)
        & (index["status"] == "ok")
    ]
    if rows.empty:
        return None

    if "run_index" in rows.columns:
        rows = rows.copy()
        rows["_run_index_numeric"] = pd.to_numeric(
            rows["run_index"],
            errors="coerce",
        )
        rows = rows.sort_values("_run_index_numeric", kind="mergesort")

    return rows.iloc[-1]


def best_drag_sum(row: pd.Series, convergence_dir: Path) -> Optional[float]:
    conv_path = resolve_convergence_path(row, convergence_dir)
    if conv_path is not None:
        df = load_best_ever_convergence(conv_path)
        if not df.empty:
            return float(df["best_fitness"].iloc[-1])

    if IS_DAE11_SUMMARY:
        return None

    if "final_best" not in row.index:
        return None

    value = pd.to_numeric(pd.Series([row["final_best"]]), errors="coerce").iloc[0]
    if not np.isfinite(value):
        return None
    return float(value)


def plot_best_seed_airfoils(
    summary: pd.DataFrame,
    seed_dirs: list[tuple[int, Path]],
    point_counts: tuple[int, ...],
) -> None:
    selected = select_best_seed_airfoil_runs(summary, seed_dirs, point_counts)
    selection_path = OUTPUT_DIR / "best_seed_airfoil_selection.csv"
    selected.to_csv(selection_path, index=False)

    if selected.empty:
        print("No best seed airfoil selections available.")
        print(f"Wrote empty selection CSV: {selection_path}")
        return

    reference_airfoils: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    rae2822 = load_optional_airfoil(RAE2822_BASELINE_PATH, "RAE2822 baseline")
    if rae2822 is not None:
        reference_airfoils["rae2822"] = rae2822

    drela_airfoils: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for n_points in point_counts:
        path = DRELA_OPTIMUM_PATHS.get(n_points)
        if path is None:
            continue
        coords = load_optional_airfoil(path, f"Drela {n_points}-point optimum")
        if coords is not None:
            drela_airfoils[n_points] = coords

    best_airfoils: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for _, row in selected.iterrows():
        n_points = int(row["n_points"])
        path = Path(str(row["best_airfoil_path"]))
        coords = load_optional_airfoil(path, f"{n_points}-point selected airfoil")
        if coords is not None:
            best_airfoils[n_points] = coords

    for n_points in point_counts:
        fig, ax = plt.subplots(figsize=(8.4, 2.9))
        row = selected[selected["n_points"] == n_points]
        legend_handles: list[Line2D] = []

        if not row.empty and n_points in best_airfoils:
            record = row.iloc[0]
            x, z = best_airfoils[n_points]
            ax.plot(
                x,
                z,
                color="#1f77b4",
                linewidth=AIRFOIL_LINEWIDTH,
                label=method_combo_label(record),
                zorder=3,
            )
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color="#1f77b4",
                    lw=AIRFOIL_LINEWIDTH,
                    label=method_combo_label(record),
                )
            )

        if "rae2822" in reference_airfoils:
            x, z = reference_airfoils["rae2822"]
            ax.plot(
                x,
                z,
                color="black",
                linewidth=AIRFOIL_LINEWIDTH,
                linestyle="--",
                label="RAE2822 baseline",
                zorder=2,
            )
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color="black",
                    lw=AIRFOIL_LINEWIDTH,
                    ls="--",
                    label="RAE2822 baseline",
                )
            )

        if n_points in drela_airfoils:
            x, z = drela_airfoils[n_points]
            ax.plot(
                x,
                z,
                color="#d62728",
                linewidth=AIRFOIL_LINEWIDTH,
                linestyle="-.",
                label="Drela optimum",
                zorder=4,
            )
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color="#d62728",
                    lw=AIRFOIL_LINEWIDTH,
                    ls="-.",
                    label="Drela optimum",
                )
            )

        y_min, y_max, y_pad = airfoil_ylim([
            best_airfoils.get(n_points),
            reference_airfoils.get("rae2822"),
            drela_airfoils.get(n_points),
        ])
        style_airfoil_axis(ax)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.set_xlabel("x/c", fontweight="bold", fontsize=AIRFOIL_FONT_SIZE)
        ax.set_ylabel("z/c", fontweight="bold", fontsize=AIRFOIL_FONT_SIZE, labelpad=8)

        if legend_handles:
            legend = fig.legend(
                handles=legend_handles,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.99),
                ncol=len(legend_handles),
                frameon=True,
                borderaxespad=0.0,
            )
            legend.get_frame().set_linewidth(AIRFOIL_SPINE_LINEWIDTH)
            for text in legend.get_texts():
                text.set_fontweight("bold")
                text.set_fontsize(AIRFOIL_FONT_SIZE)

        fig.subplots_adjust(left=0.115, right=0.985, top=0.80, bottom=0.23)
        output_path = OUTPUT_DIR / f"best_seed_airfoil_{n_points}point.png"
        fig.savefig(output_path, dpi=220, bbox_inches="tight", pad_inches=0.08)
        plt.close(fig)
        print(f"Saved {n_points}-point best seed airfoil comparison: {output_path}")

    print(f"Wrote best seed airfoil selection CSV: {selection_path}")


def method_combo_label(row: pd.Series) -> str:
    method = str(row["method"])
    optimizer = str(row["optimizer"])
    return (
        f"{METHOD_LABELS.get(method, method)}-"
        f"{OPTIMIZER_LABELS.get(optimizer, optimizer.upper())}"
    )


def airfoil_ylim(
    curves: list[Optional[tuple[np.ndarray, np.ndarray]]],
) -> tuple[float, float, float]:
    y_values = [coords[1] for coords in curves if coords is not None]
    if not y_values:
        return -0.08, 0.08, 0.01

    y = np.concatenate(y_values)
    y_min = float(np.nanmin(y))
    y_max = float(np.nanmax(y))
    y_pad = max(0.006, 0.08 * (y_max - y_min))
    return y_min, y_max, y_pad


def select_best_seed_airfoil_runs(
    summary: pd.DataFrame,
    seed_dirs: list[tuple[int, Path]],
    point_counts: tuple[int, ...],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for n_points in point_counts:
        summary_rows = summary[
            (summary["n_points"] == n_points)
            & np.isfinite(pd.to_numeric(
                summary["mean_final_objective"],
                errors="coerce",
            ))
        ].copy()
        if summary_rows.empty:
            print(f"No finite summary rows for {n_points}-point best-airfoil selection.")
            continue

        summary_rows["mean_final_objective"] = pd.to_numeric(
            summary_rows["mean_final_objective"],
            errors="coerce",
        )
        best_combo = summary_rows.sort_values(
            "mean_final_objective",
            ascending=True,
            kind="mergesort",
        ).iloc[0]
        method = str(best_combo["method"])
        optimizer = str(best_combo["optimizer"])

        best_run: Optional[dict[str, object]] = None
        for seed, seed_dir in seed_dirs:
            point_dir = seed_point_dir(seed_dir, n_points)
            index_path = point_dir / "study_index.csv"
            if not index_path.is_file():
                continue

            try:
                index = pd.read_csv(index_path)
            except Exception as exc:
                print(f"Unreadable index for seed {seed}: {index_path}: {exc}")
                continue

            normalize_index(index)
            row = latest_ok_row(index, method, optimizer)
            if row is None:
                continue

            value = best_drag_sum(row, point_dir / "convergence")
            if value is None or not np.isfinite(value) or value >= PENALTY_THRESHOLD:
                continue

            best_airfoil_path = resolve_best_airfoil_path(row)
            if best_airfoil_path is None:
                print(f"Missing best airfoil for seed {seed}: {n_points}-point {method}/{optimizer}")
                continue

            candidate = {
                "n_points": n_points,
                "seed": seed,
                "method": method,
                "optimizer": optimizer,
                "mean_drag_reduction_percent": float(best_combo["mean_drag_reduction_percent"]),
                "final_best": float(value),
                "best_airfoil_path": str(best_airfoil_path),
                "point_study_dir": str(point_dir),
            }
            if best_run is None or float(candidate["final_best"]) < float(best_run["final_best"]):
                best_run = candidate

        if best_run is not None:
            rows.append(best_run)
        else:
            print(f"No plottable best seed run found for {n_points}-point {method}/{optimizer}.")

    return pd.DataFrame(rows)


def resolve_best_airfoil_path(row: pd.Series) -> Optional[Path]:
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


def load_optional_airfoil(
    path: str | Path,
    label: str,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    path = Path(path)
    if not path.is_file():
        print(f"Missing {label}: {path}")
        return None

    try:
        return load_airfoil_xy(path)
    except Exception as exc:
        print(f"Unreadable {label} {path}: {exc}")
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


def style_airfoil_axis(ax) -> None:
    ax.set_xlim(-0.02, 1.02)
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


def resolve_convergence_path(
    row: pd.Series,
    convergence_dir: Path,
) -> Optional[Path]:
    raw = row.get("convergence_csv", "")
    if pd.isna(raw):
        raw = ""

    path = str(raw).strip()
    candidates: list[Path] = []
    if path:
        candidates.append(Path(path))
        candidates.append(convergence_dir / Path(path).name)

    method = str(row.get("method", "")).strip()
    optimizer = str(row.get("optimizer", "")).strip()
    if method and optimizer:
        candidates.append(convergence_dir / f"{method}__{optimizer}.csv")

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def load_best_ever_convergence(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"Unreadable convergence {path}: {exc}")
        return pd.DataFrame(columns=["generation", "best_fitness"])

    if df.shape[1] < 2 or df.empty:
        return pd.DataFrame(columns=["generation", "best_fitness"])

    if IS_DAE11_SUMMARY:
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
    if IS_DAE11_SUMMARY:
        if "cd" in df.columns:
            value_column = "cd"
        elif "best_historic_objective" in df.columns:
            value_column = "best_historic_objective"
        elif "best_objective" in df.columns:
            value_column = "best_objective"
        elif ALLOW_LEGACY_DAE11_FITNESS:
            value_column = "best_fitness" if "best_fitness" in df.columns else df.columns[1]
            print(
                "WARNING: treating legacy best_fitness as raw DAE-11 Cd for "
                f"{path}; this may include a constraint penalty."
            )
        else:
            raise SystemExit(
                "DAE-11 convergence data does not contain a raw objective: "
                f"{path}\n"
                "The legacy best_fitness column may include a constraint penalty, "
                "which cannot be reconstructed after the run. Re-run with the "
                "updated core/fitness.py. To knowingly use the legacy value, set "
                "SEED_ALLOW_LEGACY_DAE11_FITNESS=1."
            )
    else:
        if "cd" in df.columns:
            value_column = "cd"
        else:
            value_column = "best_fitness" if "best_fitness" in df.columns else df.columns[1]

    df = df[[generation_column, value_column]].copy()
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


def convergence_ylim(
    point_data: pd.DataFrame,
    n_points: int | None = None,
) -> Optional[tuple[float, float]]:
    ymin_raw = "" if CONVERGENCE_YMIN is None else str(CONVERGENCE_YMIN).strip().lower()
    ymax_raw = "" if CONVERGENCE_YMAX is None else str(CONVERGENCE_YMAX).strip().lower()
    ymin_auto = ymin_raw in {"", "auto", "none"}
    ymax_auto = ymax_raw in {"", "auto", "none"}

    default_ylim = (
        CONVERGENCE_YLIMITS_BY_N_POINTS.get(n_points)
        if CONVERGENCE_USE_POINT_LIMITS else None
    )
    if default_ylim is not None:
        return (
            default_ylim[0] if ymin_auto else float(ymin_raw),
            default_ylim[1] if ymax_auto else float(ymax_raw),
        )

    if not ymin_auto and not ymax_auto:
        return float(ymin_raw), float(ymax_raw)

    if point_data.empty:
        return None

    values = []
    for column in ("mean_best_fitness", "std_best_fitness"):
        point_data[column] = pd.to_numeric(point_data[column], errors="coerce")

    in_window = point_data[
        (pd.to_numeric(point_data["generation"], errors="coerce") >= CONVERGENCE_XMIN)
        & (pd.to_numeric(point_data["generation"], errors="coerce") <= CONVERGENCE_XMAX)
    ]
    if in_window.empty:
        in_window = point_data

    mean = in_window["mean_best_fitness"].to_numpy(dtype=float)
    std = in_window["std_best_fitness"].fillna(0.0).to_numpy(dtype=float)
    values.extend((mean - std)[np.isfinite(mean - std)])
    values.extend((mean + std)[np.isfinite(mean + std)])

    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return None

    ymin = float(np.min(values))
    ymax = float(np.max(values))
    if np.isclose(ymin, ymax):
        pad = max(1.0e-3, abs(ymin) * 0.05)
    else:
        pad = 0.04 * (ymax - ymin)

    if ymin_auto:
        ymin = ymin - pad
    else:
        ymin = float(ymin_raw)

    if ymax_auto:
        ymax = ymax + pad
    else:
        ymax = float(ymax_raw)

    return ymin, ymax


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
    ylim: Optional[tuple[float, float]] = None,
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
    if ylim is not None:
        ax.set_ylim(*ylim)
        yticks = CONVERGENCE_YTICKS_BY_N_POINTS.get(n_points)
        if yticks is not None:
            ymin, ymax = sorted(ylim)
            visible_ticks = yticks[(yticks >= ymin) & (yticks <= ymax)]
            if visible_ticks.size:
                ax.set_yticks(visible_ticks)
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


def plot_seed_convergence_by_optimizer(
    convergence_summary: pd.DataFrame,
    n_points: int,
) -> None:
    point_data = convergence_summary[convergence_summary["n_points"] == n_points].copy()
    ylim = convergence_ylim(point_data, n_points)
    fig, axes = plt.subplots(
        len(OPTIMIZERS),
        1,
        figsize=(8, 12),
        sharex=True,
        sharey=True,
    )
    plotted_total = 0

    for ax, optimizer in zip(axes, OPTIMIZERS):
        optimizer_data = point_data[point_data["optimizer"] == optimizer]
        plotted_here = 0
        legend_methods = []

        for method in METHODS:
            curve = optimizer_data[optimizer_data["method"] == method]
            if curve.empty:
                continue

            curve = curve.sort_values("generation", kind="mergesort")
            x = curve["generation"].to_numpy(dtype=float)
            mean = curve["mean_best_fitness"].to_numpy(dtype=float)
            std = curve["std_best_fitness"].fillna(0.0).to_numpy(dtype=float)
            color = METHOD_COLORS.get(method, "#4b5563")

            if CONVERGENCE_STD_ALPHA > 0.0:
                ax.fill_between(
                    x,
                    mean - std,
                    mean + std,
                    color=color,
                    alpha=CONVERGENCE_STD_ALPHA,
                    linewidth=0,
                )
            ax.plot(
                x,
                mean,
                linewidth=CONVERGENCE_LINEWIDTH,
                color=color,
                label=METHOD_LABELS.get(method, method),
            )
            legend_methods.append(method)
            plotted_here += 1
            plotted_total += 1

        style_convergence_axis(
            ax,
            OPTIMIZER_LABELS.get(optimizer, optimizer.upper()),
            show_ylabel=True,
            ylim=ylim,
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
                0.5,
                0.5,
                "no completed runs yet",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="#6b7280",
            )

    axes[-1].set_xlabel("Generation")
    fig.tight_layout()
    add_convergence_panel_label(fig)
    output_path = OUTPUT_DIR / f"convergence_by_optimizer_seed_mean_{n_points}point.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"Saved {n_points}-point seed-mean convergence ({plotted_total} curves): {output_path}")


def plot_seed_convergence_by_optimizer_wide(
    convergence_summary: pd.DataFrame,
    n_points: int,
) -> None:
    point_data = convergence_summary[convergence_summary["n_points"] == n_points].copy()
    ylim = convergence_ylim(point_data, n_points)
    fig, axes = plt.subplots(
        1,
        len(OPTIMIZERS),
        figsize=(11.5, 4.2),
        sharex=True,
        sharey=True,
    )
    plotted_total = 0
    legend_methods_seen = set()

    for ax, optimizer in zip(axes, OPTIMIZERS):
        optimizer_data = point_data[point_data["optimizer"] == optimizer]
        plotted_here = 0

        for method in METHODS:
            curve = optimizer_data[optimizer_data["method"] == method]
            if curve.empty:
                continue

            curve = curve.sort_values("generation", kind="mergesort")
            x = curve["generation"].to_numpy(dtype=float)
            mean = curve["mean_best_fitness"].to_numpy(dtype=float)
            std = curve["std_best_fitness"].fillna(0.0).to_numpy(dtype=float)
            color = METHOD_COLORS.get(method, "#4b5563")

            if CONVERGENCE_STD_ALPHA > 0.0:
                ax.fill_between(
                    x,
                    mean - std,
                    mean + std,
                    color=color,
                    alpha=CONVERGENCE_STD_ALPHA,
                    linewidth=0,
                )
            ax.plot(
                x,
                mean,
                linewidth=CONVERGENCE_LINEWIDTH,
                color=color,
                label=METHOD_LABELS.get(method, method),
            )
            legend_methods_seen.add(method)
            plotted_here += 1
            plotted_total += 1

        style_convergence_axis(
            ax,
            OPTIMIZER_LABELS.get(optimizer, optimizer.upper()),
            show_ylabel=ax is axes[0],
            ylim=ylim,
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

    if legend_methods_seen:
        legend_methods = [method for method in METHODS if method in legend_methods_seen]
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
        bottom=0.27 if legend_methods_seen else 0.17,
        wspace=0.22,
    )
    output_path = OUTPUT_DIR / f"convergence_by_optimizer_wide_seed_mean_{n_points}point.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"Saved {n_points}-point wide seed-mean convergence ({plotted_total} curves): {output_path}")


def shared_heatmap_limits(summary: pd.DataFrame) -> tuple[float, float]:
    env_vmin = os.environ.get("PLOT_HEATMAP_VMIN")
    env_vmax = os.environ.get("PLOT_HEATMAP_VMAX")
    if env_vmin is not None and env_vmax is not None:
        return float(env_vmin), float(env_vmax)

    mean_column = (
        "mean_final_objective"
        if SUMMARY_METRIC == "objective"
        else "mean_drag_reduction_percent"
    )
    values = pd.to_numeric(summary[mean_column], errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.0

    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if np.isclose(vmin, vmax):
        pad = max(1.0, abs(vmin) * 0.05)
        return vmin - pad, vmax + pad
    return vmin, vmax


def plot_summary_heatmap(
    summary: pd.DataFrame,
    n_points: int,
    vmin: float,
    vmax: float,
) -> None:
    if SUMMARY_METRIC == "objective":
        mean_column = "mean_final_objective"
        std_column = "std_final_objective"
        colorbar_label = "Mean final\nobjective"
    else:
        mean_column = "mean_drag_reduction_percent"
        std_column = "std_drag_reduction_percent"
        colorbar_label = "Objective\nreduction"

    mean_matrix = matrix_from_summary(summary, n_points, mean_column)
    std_matrix = matrix_from_summary(summary, n_points, std_column)

    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    masked = np.ma.masked_invalid(mean_matrix)
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
        [OPTIMIZER_LABELS[optimizer] for optimizer in OPTIMIZERS],
        fontweight="bold",
        fontsize=HEATMAP_FONT_SIZE,
        rotation=30,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_yticklabels(
        [METHOD_LABELS[method] for method in METHODS],
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
            mean = mean_matrix[i, j]
            std = std_matrix[i, j]
            if np.isfinite(mean):
                if SUMMARY_METRIC == "objective":
                    text = f"{mean:.5f}\n+- {std:.5f}"
                else:
                    text = f"{mean:.1f}%\n+- {std:.1f}"
            else:
                text = "-\n-"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="black",
                fontsize=HEATMAP_FONT_SIZE,
                fontweight="bold",
                linespacing=1.05,
            )

    cbar = fig.colorbar(image, ax=ax)
    cbar.ax.set_xlabel(
        colorbar_label,
        fontweight="bold",
        fontsize=HEATMAP_FONT_SIZE,
        labelpad=12,
    )
    cbar.outline.set_linewidth(1.4)
    cbar.ax.tick_params(width=1.4, labelsize=HEATMAP_FONT_SIZE)
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontweight("bold")

    fig.tight_layout()
    output_path = OUTPUT_DIR / f"final_results_matrix_{n_points}point.png"
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    print(f"Saved {n_points}-point matrix: {output_path}")


def matrix_from_summary(
    summary: pd.DataFrame,
    n_points: int,
    column: str,
) -> np.ndarray:
    matrix = np.full((len(METHODS), len(OPTIMIZERS)), np.nan)
    subset = summary[summary["n_points"] == n_points]

    for i, method in enumerate(METHODS):
        for j, optimizer in enumerate(OPTIMIZERS):
            rows = subset[
                (subset["method"] == method)
                & (subset["optimizer"] == optimizer)
            ]
            if not rows.empty:
                matrix[i, j] = float(rows.iloc[-1][column])

    return matrix


def drag_reduction_percent(best_drag_sum: float, initial_drag_sum: float) -> float:
    if initial_drag_sum <= 0.0:
        raise SystemExit("INITIAL_DRAG_SUM must be positive.")
    reduction = 100.0 * (initial_drag_sum - float(best_drag_sum)) / initial_drag_sum
    return max(0.0, reduction)


if __name__ == "__main__":
    main()
