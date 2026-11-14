"""Plot seed-averaged 5x5 results and report Cauchy convergence.

From the optimizer directory, for the current DAE11 study:

    SEED_STUDY_NAME=dae11 SEED_STUDY_CASE=constrained \
        python3 studies/plot_seed_matrix_summary.py

Use SEED_STUDY_CASE=unconstrained for the CL-only runs. Set PLOT=25 for a
5x5 airfoil grid or PLOT=5 for the best airfoil per parameterization.
The DAE11 reference geometry is read from studies/reference_airfoils/.
No optimizer data is modified.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.markers import TICKRIGHT, TICKUP
from matplotlib.patches import Patch
from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter, LogLocator, NullFormatter
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

STUDY_DIR = Path(__file__).resolve().parent
STUDY_NAME = os.environ.get("SEED_STUDY_NAME", "").strip()
STUDY_CASE = os.environ.get("SEED_STUDY_CASE", "").strip()
STUDY_GLOB = os.environ.get(
    "SEED_STUDY_GLOB",
    f"{STUDY_NAME}_seed_*" if STUDY_NAME else "point_studies_full_study_seed_*",
)
STUDY_REGEX = os.environ.get("SEED_STUDY_REGEX", "").strip()
REFERENCE = os.environ.get("SEED_SUMMARY_REFERENCE", "").strip().lower()
IS_DAE11 = "dae11" in re.sub(r"[^a-z0-9]", "", STUDY_NAME.lower()) or (
    re.sub(r"[^a-z0-9]", "", REFERENCE) in {"dae11", "dreladae11"}
)
METRIC = os.environ.get(
    "SEED_SUMMARY_METRIC",
    "reduction" if IS_DAE11 or not STUDY_NAME else "objective",
).strip().lower()
OUTPUT_DIR = Path(os.environ.get(
    "SEED_SUMMARY_DIR",
    str(STUDY_DIR / (
        f"{STUDY_NAME}_{STUDY_CASE}_seed_summary" if STUDY_CASE
        else f"{STUDY_NAME}_seed_summary" if STUDY_NAME
        else "seed_matrix_summary"
    )),
))
AIRFOIL_PLOT = os.environ.get("PLOT", "0").strip()
REFERENCE_DIR = Path(os.environ.get(
    "SEED_REFERENCE_AIRFOIL_DIR", str(STUDY_DIR / "reference_airfoils")
))

METHODS = ("parsec", "cst", "bspline", "hickshenne", "ffd")
OPTIMIZERS = ("ga", "pso", "de", "cmaes", "abc")
METHOD_LABELS = {
    "parsec": "PARSEC", "cst": "CST", "bspline": "B-Splines",
    "hickshenne": "Hicks-Henne", "ffd": "FFD",
}
OPTIMIZER_LABELS = {
    "ga": "GA", "pso": "PSO", "de": "DE", "cmaes": "CMA-ES", "abc": "ABC",
}
METHOD_COLORS = {
    "parsec": "#1f77b4", "cst": "#ff7f0e", "bspline": "#2ca02c",
    "hickshenne": "#d62728", "ffd": "#9467bd",
}
DAE11_BASELINE_CD = float(os.environ.get("DAE11_BASELINE_CD", "0.050505"))
BASELINES = {
    1: float(os.environ.get(
        "INITIAL_DRAG_SUM_1POINT", str(DAE11_BASELINE_CD) if IS_DAE11 else "0.103416"
    )),
    2: float(os.environ.get("INITIAL_DRAG_SUM_2POINT", "0.14673")),
    4: float(os.environ.get("INITIAL_DRAG_SUM_4POINT", "0.346914")),
}
PENALTY_THRESHOLD = float(os.environ.get("PLOT_PENALTY_THRESHOLD", "99"))
CAUCHY_DRAG_COUNT = float(os.environ.get("PLOT_CAUCHY_DRAG_COUNT", "0.00005"))
CAUCHY_EPS_REL = float(os.environ.get("PLOT_CAUCHY_EPS_REL", "0"))
ALLOW_LEGACY_DAE11_FITNESS = (
    os.environ.get("SEED_ALLOW_LEGACY_DAE11_FITNESS", "0").lower()
    in {"1", "true", "yes", "on"}
)

HEATMAP_FONT_SIZE = float(os.environ.get("PLOT_HEATMAP_FONT_SIZE", "13"))
HEATMAP_CMAP = os.environ.get(
    "PLOT_HEATMAP_CMAP", "RdYlGn_r" if METRIC == "objective" else "RdYlGn"
)
CAUCHY_HEATMAP_CMAP = os.environ.get("PLOT_CAUCHY_HEATMAP_CMAP", "bwr")
CONV_FONT_SIZE = float(os.environ.get("PLOT_CONV_FONT_SIZE", "13"))
CONV_TICK_SIZE = float(os.environ.get("PLOT_CONV_TICK_FONT_SIZE", "10"))
CONV_PANEL_SIZE = float(os.environ.get("PLOT_CONV_PANEL_FONT_SIZE", "13"))
CONV_LEGEND_SIZE = float(os.environ.get("PLOT_CONV_LEGEND_FONT_SIZE", "13"))
CONV_LINEWIDTH = float(os.environ.get("PLOT_CONV_LINEWIDTH", "2"))
CONV_Z_MINOR_GRID = 0.25
CONV_Z_MAJOR_GRID = 0.5
CONV_Z_MINOR_TICK = 1.5
CONV_Z_CURVE = 2.0
CONV_Z_LABEL = 5.0
CONV_Z_MAJOR_TICK = 6.0
CONV_Z_SPINE = 7.0
CONV_XSCALE = os.environ.get("PLOT_CONV_XSCALE", "log").strip().lower()
CONV_XMIN = max(1.0, float(os.environ.get("PLOT_CONV_XMIN", "1")))
CONV_XMAX = float(os.environ.get("PLOT_CONV_XMAX", "500"))
DEFAULT_CONV_YMIN = (
    "0.03" if IS_DAE11 and STUDY_CASE == "constrained" else
    "0.01" if IS_DAE11 and STUDY_CASE == "unconstrained" else
    "0" if IS_DAE11 else ""
)
DEFAULT_CONV_YMAX = (
    "0.05" if IS_DAE11 and STUDY_CASE == "constrained" else
    "0.06" if IS_DAE11 and STUDY_CASE == "unconstrained" else
    "0.055" if IS_DAE11 else ""
)
CONV_YMIN = os.environ.get("PLOT_CONV_YMIN", DEFAULT_CONV_YMIN).strip()
CONV_YMAX = os.environ.get("PLOT_CONV_YMAX", DEFAULT_CONV_YMAX).strip()
CONV_YLABEL = os.environ.get(
    "PLOT_CONV_YLABEL", r"Best-historic objective, $\mathcal{F}^*$"
)
CONV_POINT_LIMITS = os.environ.get(
    "PLOT_CONV_USE_POINT_LIMITS", "0" if STUDY_NAME else "1"
).lower() not in {"0", "false", "no", "off"}
CONV_YLIMITS = {1: (0.02, 0.06), 2: (0.04, 0.10), 4: (0.10, 0.20)}
CONV_YTICKS = {
    1: np.arange(0.0, 0.061, 0.01) if IS_DAE11 else
       np.array([0.02, 0.03, 0.04, 0.05, 0.06]),
    2: np.array([0.04, 0.06, 0.08, 0.10]),
    4: np.array([0.10, 0.12, 0.14, 0.16, 0.18, 0.20]),
}
AIRFOIL_FONT_SIZE = float(os.environ.get("PLOT_AIRFOIL_FONT_SIZE", "13"))
AIRFOIL_TICK_SIZE = float(os.environ.get("PLOT_AIRFOIL_TICK_FONT_SIZE", "10"))
AIRFOIL_LEGEND_SIZE = float(os.environ.get(
    "PLOT_AIRFOIL_LEGEND_FONT_SIZE",
    "24" if AIRFOIL_PLOT == "25" else str(CONV_LEGEND_SIZE),
))
AIRFOIL_METHOD_FONT_SIZE = float(os.environ.get(
    "PLOT_AIRFOIL_METHOD_FONT_SIZE", str(AIRFOIL_LEGEND_SIZE)
))
AIRFOIL_ANNOTATION_SIZE = float(os.environ.get(
    "PLOT_AIRFOIL_ANNOTATION_FONT_SIZE", "9" if AIRFOIL_PLOT == "25" else "11"
))
AIRFOIL_LINEWIDTH = float(os.environ.get("PLOT_AIRFOIL_LINEWIDTH", "2"))
AIRFOIL_EVAL_POINTS = int(os.environ.get(
    "PLOT_AOA_N_POINTS", os.environ.get("CASE_AOA_N_POINTS", "30")
))


@dataclass
class Run:
    seed: int
    status: str
    path: Path | None
    curve: pd.DataFrame
    opt_dir: Path | None = None


def point_counts() -> tuple[int, ...]:
    raw = os.environ.get("SEED_POINT_STUDIES")
    if STUDY_CASE:
        points = (1,) if raw is None else tuple(
            int(value) for value in raw.replace(",", " ").split()
        )
        if points != (1,):
            raise SystemExit("SEED_STUDY_CASE supports only the one-point matrix.")
    elif raw is not None:
        points = tuple(int(value) for value in raw.replace(",", " ").split())
    elif STUDY_NAME:
        discovered = set()
        for seed_dir in STUDY_DIR.glob(STUDY_GLOB):
            for directory in seed_dir.glob("matrix_5x5_*point"):
                match = re.fullmatch(r"matrix_5x5_(\d+)point", directory.name)
                if match:
                    discovered.add(int(match.group(1)))
        points = tuple(sorted(discovered))
    else:
        points = (1, 2, 4)
    if not points:
        raise SystemExit(f"No point-study matrices found under {STUDY_DIR / STUDY_GLOB}")
    if METRIC == "reduction" and any(point not in BASELINES for point in points):
        raise SystemExit(f"No drag baseline configured for point counts {points}.")
    return points


def discover_seeds() -> list[tuple[int, Path]]:
    if STUDY_REGEX:
        pattern = re.compile(STUDY_REGEX)
    elif STUDY_NAME:
        pattern = re.compile(rf"^{re.escape(STUDY_NAME)}_seed_(\d+)$")
    else:
        suffix = r"(?:$|_)" if os.environ.get("SEED_ALLOW_SUFFIX") == "1" else r"$"
        pattern = re.compile(rf"^point_studies_full_study_seed_(\d+){suffix}")
    seeds = []
    for directory in STUDY_DIR.glob(STUDY_GLOB):
        match = pattern.match(directory.name)
        if directory.is_dir() and match:
            seeds.append((int(match.group(1)), directory))
    return sorted(seeds)


def matrix_dir(seed_dir: Path, n_points: int) -> Path:
    return seed_dir / STUDY_CASE if STUDY_CASE else seed_dir / f"matrix_5x5_{n_points}point"


def convergence_path(row: pd.Series, matrix: Path) -> Path | None:
    copied = matrix / "convergence"
    raw = row.get("convergence_csv", "")
    candidates = []
    if pd.notna(raw) and str(raw).strip():
        path = Path(str(raw).strip())
        candidates.extend((path, copied / path.name))
    candidates.append(copied / f"{row['method']}__{row['optimizer']}.csv")
    return next((path for path in candidates if path.is_file()), None)


def optimization_dir(row: pd.Series, copied_convergence: Path) -> Path | None:
    def matches(path: Path) -> bool:
        source = path / "convergence.csv"
        return (
            source.is_file()
            and source.read_bytes() == copied_convergence.read_bytes()
        )

    raw = row.get("opt_dir", "")
    if pd.notna(raw) and str(raw).strip():
        path = Path(str(raw).strip())
        if not path.is_absolute():
            path = STUDY_DIR.parent / path
        if matches(path):
            return path
    run_index = pd.to_numeric(row.get("run_index", np.nan), errors="coerce")
    if np.isfinite(run_index):
        path = STUDY_DIR.parent / f"optimization_{int(run_index)}"
        if matches(path):
            return path
    return None


def load_curve(path: Path) -> pd.DataFrame:
    columns = ["generation", "best_fitness"]
    try:
        history = pd.read_csv(path)
    except Exception as exc:
        print(f"Unreadable convergence {path}: {exc}")
        return pd.DataFrame(columns=columns)
    if history.empty or history.shape[1] < 2:
        return pd.DataFrame(columns=columns)

    if IS_DAE11:
        if "feasible" not in history:
            raise SystemExit(f"DAE11 convergence lacks feasible column: {path}")
        feasible = history["feasible"].astype(str).str.strip().str.lower().isin(
            {"1", "true", "yes", "y"}
        )
        history = history.loc[feasible].copy()
        if "cd" in history:
            value_column = "cd"
        elif "best_historic_objective" in history:
            value_column = "best_historic_objective"
        elif "best_objective" in history:
            value_column = "best_objective"
        elif ALLOW_LEGACY_DAE11_FITNESS and "best_fitness" in history:
            value_column = "best_fitness"
            print(f"WARNING: legacy best_fitness may include penalties: {path}")
        else:
            raise SystemExit(
                f"DAE11 convergence lacks raw Cd: {path}\n"
                "Set SEED_ALLOW_LEGACY_DAE11_FITNESS=1 only if that is acceptable."
            )
    else:
        value_column = (
            "cd" if "cd" in history else
            "best_fitness" if "best_fitness" in history else history.columns[1]
        )

    generation_column = "generation" if "generation" in history else history.columns[0]
    curve = history[[generation_column, value_column]].copy()
    curve.columns = columns
    for column in columns:
        curve[column] = pd.to_numeric(curve[column], errors="coerce")
    curve = curve.dropna().sort_values("generation", kind="mergesort")
    curve = curve[
        (curve["best_fitness"] > 0)
        & (curve["best_fitness"] < PENALTY_THRESHOLD)
    ].copy()
    curve["raw_cd"] = curve["best_fitness"]
    curve["best_fitness"] = curve["best_fitness"].cummin()
    return curve


def collect_runs(
    seeds: list[tuple[int, Path]], points: tuple[int, ...]
) -> dict[tuple[int, str, str], list[Run]]:
    runs = {
        (n, method, optimizer): []
        for n in points for method in METHODS for optimizer in OPTIMIZERS
    }
    for n in points:
        for seed, seed_dir in seeds:
            matrix = matrix_dir(seed_dir, n)
            index_path = matrix / "study_index.csv"
            if not index_path.is_file():
                print(f"Missing index for seed {seed}: {index_path}")
                continue
            try:
                index = pd.read_csv(index_path)
                for column in ("method", "optimizer", "status"):
                    index[column] = index[column].astype(str).str.strip().str.lower()
            except Exception as exc:
                print(f"Unreadable index for seed {seed}: {index_path}: {exc}")
                continue

            for method in METHODS:
                for optimizer in OPTIMIZERS:
                    rows = index[
                        (index["method"] == method)
                        & (index["optimizer"] == optimizer)
                        & (index["status"] == "ok")
                    ]
                    if rows.empty:
                        run = Run(seed, "missing_completed_run", None, pd.DataFrame())
                    else:
                        if "run_index" in rows:
                            rows = rows.assign(
                                _number=pd.to_numeric(rows["run_index"], errors="coerce")
                            ).sort_values("_number", kind="mergesort")
                        path = convergence_path(rows.iloc[-1], matrix)
                        if path is None:
                            run = Run(seed, "missing_convergence_csv", None, pd.DataFrame())
                        else:
                            curve = load_curve(path)
                            run = Run(
                                seed, "ok" if not curve.empty else "empty_convergence",
                                path, curve, optimization_dir(rows.iloc[-1], path),
                            )
                    runs[(n, method, optimizer)].append(run)
    return runs


def sample_stats(values: list[float]) -> tuple[float, float, float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return np.nan, np.nan, np.nan, np.nan
    return (
        float(finite.mean()),
        float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
        float(finite.min()),
        float(finite.max()),
    )


def build_matrix_summary(
    runs: dict[tuple[int, str, str], list[Run]], points: tuple[int, ...]
) -> pd.DataFrame:
    rows = []
    for n in points:
        baseline = BASELINES[n]
        for method in METHODS:
            for optimizer in OPTIMIZERS:
                objectives = []
                seeds = []
                for run in runs[(n, method, optimizer)]:
                    if run.status == "ok":
                        value = float(run.curve["best_fitness"].min())
                    elif IS_DAE11 and run.status == "empty_convergence":
                        value = baseline
                    else:
                        continue
                    if IS_DAE11:
                        value = min(value, baseline)
                    objectives.append(value)
                    seeds.append(run.seed)
                reductions = [
                    max(0.0, 100.0 * (baseline - value) / baseline)
                    for value in objectives
                ] if baseline > 0 else []
                obj_mean, obj_std, obj_min, obj_max = sample_stats(objectives)
                red_mean, red_std, red_min, red_max = sample_stats(reductions)
                rows.append({
                    "n_points": n, "method": method, "optimizer": optimizer,
                    "n_seeds": len(seeds), "seeds": " ".join(map(str, seeds)),
                    "drag_reduction_baseline": baseline if METRIC == "reduction" else np.nan,
                    "mean_final_objective": obj_mean,
                    "std_final_objective": obj_std,
                    "min_final_objective": obj_min,
                    "max_final_objective": obj_max,
                    "mean_drag_reduction_percent": red_mean,
                    "std_drag_reduction_percent": red_std,
                    "min_drag_reduction_percent": red_min,
                    "max_drag_reduction_percent": red_max,
                })
    return pd.DataFrame(rows)


def build_convergence_summary(
    runs: dict[tuple[int, str, str], list[Run]]
) -> pd.DataFrame:
    rows = []
    for (n, method, optimizer), case in runs.items():
        valid = [run for run in case if run.status == "ok"]
        generations = sorted({
            int(round(generation))
            for run in valid for generation in run.curve["generation"]
        })
        if not generations:
            continue
        aligned = pd.DataFrame(index=generations)
        for run in valid:
            curve = run.curve.copy()
            curve["generation"] = curve["generation"].round().astype(int)
            series = curve.groupby("generation")["best_fitness"].last()
            aligned[str(run.seed)] = series.reindex(generations).ffill()
        for generation, values in aligned.iterrows():
            finite = values.dropna().astype(float).to_list()
            if not finite:
                continue
            mean, std, minimum, maximum = sample_stats(finite)
            rows.append({
                "n_points": n, "method": method, "optimizer": optimizer,
                "generation": generation, "mean_best_fitness": mean,
                "std_best_fitness": std, "min_best_fitness": minimum,
                "max_best_fitness": maximum,
            })
    return pd.DataFrame(rows, columns=[
        "n_points", "method", "optimizer", "generation",
        "mean_best_fitness", "std_best_fitness",
        "min_best_fitness", "max_best_fitness",
    ])


def cauchy_epsilon(n_points: int, final_best: float) -> float:
    absolute = os.environ.get("PLOT_CAUCHY_EPS_ABS")
    eps = float(absolute) if absolute is not None else CAUCHY_DRAG_COUNT * n_points
    return eps + CAUCHY_EPS_REL * max(abs(final_best), 1.0e-12)


def lognormal_iteration_interval(values: list[float]) -> tuple[float, float, float, float]:
    """Return median and one-sigma bounds after fitting log(iterations + 1)."""
    iterations = np.asarray(values, dtype=float)
    iterations = iterations[np.isfinite(iterations) & (iterations >= 1)]
    if not iterations.size:
        return np.nan, np.nan, np.nan, np.nan
    logs = np.log1p(iterations)
    mu = float(np.mean(logs))
    sigma = float(np.std(logs, ddof=1)) if logs.size > 1 else 0.0
    median = float(np.expm1(mu))
    lower = max(1.0, float(np.expm1(mu - sigma)))
    upper = float(np.expm1(mu + sigma))
    return median, lower, upper, sigma


def build_cauchy_summary(
    runs: dict[tuple[int, str, str], list[Run]]
) -> pd.DataFrame:
    rows = []
    for (n, method, optimizer), case in runs.items():
        completed = []
        converged = []
        for run in case:
            if run.status != "ok":
                continue
            curve = run.curve
            final = float(curve["best_fitness"].iloc[-1])
            eps = cauchy_epsilon(n, final)
            values = curve["best_fitness"].to_numpy(dtype=float)
            generations = curve["generation"].to_numpy(dtype=float)
            hits = np.flatnonzero((generations >= 1) & (values - final <= eps))
            last = float(curve["generation"].iloc[-1])
            completed.append({"seed": run.seed, "final_best": final, "epsilon": eps})
            if hits.size:
                position = int(hits[0])
                generation = float(generations[position])
                iterations = generation
                converged.append({
                    "seed": run.seed,
                    "iterations": iterations,
                    "generation": generation,
                    "fraction": iterations / last if last > 0 else np.nan,
                    "cost": float(values[position]),
                    "remaining": float(values[position] - final),
                })

        row = {
            "n_points": n, "method": method, "optimizer": optimizer,
            "n_seed_dirs": len(case), "n_completed_runs": len(completed),
            "n_converged": len(converged),
            "convergence_rate": (
                len(converged) / len(completed) if completed else np.nan
            ),
            "seeds": " ".join(str(run.seed) for run in case),
            "converged_seeds": " ".join(str(run["seed"]) for run in converged),
        }
        for name, data, key in (
            ("iterations_to_converge", converged, "iterations"),
            ("convergence_generation", converged, "generation"),
            ("convergence_fraction", converged, "fraction"),
            ("final_best", completed, "final_best"),
            ("cost_at_convergence", converged, "cost"),
            ("remaining_improvement", converged, "remaining"),
        ):
            mean, std, _, _ = sample_stats([item[key] for item in data])
            row[f"mean_{name}"] = mean
            row[f"std_{name}"] = std
        median, lower, upper, sigma = lognormal_iteration_interval(
            [item["iterations"] for item in converged]
        )
        row["lognormal_median_iterations_to_converge"] = median
        row["lognormal_lower_iterations_to_converge"] = lower
        row["lognormal_upper_iterations_to_converge"] = upper
        row["lognormal_sigma_log1p_iterations_to_converge"] = sigma
        row["mean_cauchy_epsilon"] = sample_stats(
            [item["epsilon"] for item in completed]
        )[0]
        rows.append(row)
    return pd.DataFrame(rows)


def heatmap_limits(summary: pd.DataFrame) -> tuple[float, float]:
    lower = os.environ.get("PLOT_HEATMAP_VMIN")
    upper = os.environ.get("PLOT_HEATMAP_VMAX")
    if lower is not None and upper is not None:
        return float(lower), float(upper)
    column = "mean_final_objective" if METRIC == "objective" else "mean_drag_reduction_percent"
    values = pd.to_numeric(summary[column], errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return 0.0, 1.0
    vmin, vmax = float(values.min()), float(values.max())
    if np.isclose(vmin, vmax):
        pad = max(1.0, abs(vmin) * 0.05)
        return vmin - pad, vmax + pad
    return vmin, vmax


def cauchy_heatmap_limits(summary: pd.DataFrame) -> tuple[float, float]:
    lower = os.environ.get("PLOT_CAUCHY_HEATMAP_VMIN")
    upper = os.environ.get("PLOT_CAUCHY_HEATMAP_VMAX")
    if lower is not None and upper is not None:
        return float(lower), float(upper)
    values = pd.to_numeric(
        summary["lognormal_median_iterations_to_converge"], errors="coerce"
    ).to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return 1.0, 2.0
    vmin, vmax = float(values.min()), float(values.max())
    if np.isclose(vmin, vmax):
        return max(1.0, vmin - 1.0), vmax + 1.0
    return vmin, vmax


def matrix_values(summary: pd.DataFrame, n: int, column: str) -> np.ndarray:
    matrix = np.full((len(METHODS), len(OPTIMIZERS)), np.nan)
    selected = summary[summary["n_points"] == n].set_index(["method", "optimizer"])
    for i, method in enumerate(METHODS):
        for j, optimizer in enumerate(OPTIMIZERS):
            matrix[i, j] = float(selected.loc[(method, optimizer), column])
    return matrix


def plot_matrix(
    summary: pd.DataFrame, n: int, limits: tuple[float, float],
    *, iterations: bool = False,
) -> None:
    objective = METRIC == "objective" and not iterations
    if iterations:
        mean_column = "lognormal_median_iterations_to_converge"
        lower = matrix_values(summary, n, "lognormal_lower_iterations_to_converge")
        upper = matrix_values(summary, n, "lognormal_upper_iterations_to_converge")
        cmap = CAUCHY_HEATMAP_CMAP
    else:
        mean_column = "mean_final_objective" if objective else "mean_drag_reduction_percent"
        std_column = "std_final_objective" if objective else "std_drag_reduction_percent"
        std = matrix_values(summary, n, std_column)
        cmap = HEATMAP_CMAP
    values = matrix_values(summary, n, mean_column)
    display = np.full((len(METHODS) + 1, len(OPTIMIZERS) + 1), np.nan)
    display[:-1, :-1] = values
    for i in range(len(METHODS)):
        finite = values[i, np.isfinite(values[i])]
        if finite.size:
            display[i, -1] = finite.mean()
    for j in range(len(OPTIMIZERS)):
        finite = values[np.isfinite(values[:, j]), j]
        if finite.size:
            display[-1, j] = finite.mean()
    finite = values[np.isfinite(values)]
    if finite.size:
        display[-1, -1] = finite.mean()

    fig, ax = plt.subplots(figsize=(9.4, 6.7))
    image = ax.imshow(
        np.ma.masked_invalid(display), cmap=cmap,
        aspect="auto", vmin=limits[0], vmax=limits[1],
    )
    xlabels = [OPTIMIZER_LABELS[name] for name in OPTIMIZERS] + ["Average"]
    ylabels = [METHOD_LABELS[name] for name in METHODS] + ["Average"]
    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_xticklabels(
        xlabels, fontweight="bold", fontsize=HEATMAP_FONT_SIZE,
        rotation=30, ha="right", rotation_mode="anchor",
    )
    ax.set_yticklabels(
        ylabels, fontweight="bold", fontsize=HEATMAP_FONT_SIZE,
        rotation=30, ha="right", va="center", rotation_mode="anchor",
    )
    ax.set_xticks(np.arange(-0.5, len(xlabels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ylabels), 1), minor=True)
    ax.grid(which="minor", color="black", linestyle="-", linewidth=0.9)
    ax.axvline(len(OPTIMIZERS) - 0.5, color="black", linewidth=4.0)
    ax.axhline(len(METHODS) - 0.5, color="black", linewidth=4.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(axis="both", which="major", length=0, pad=7, width=1.4)
    for spine in ax.spines.values():
        spine.set_linewidth(1.4)

    for i in range(len(ylabels)):
        for j in range(len(xlabels)):
            mean = display[i, j]
            if np.isfinite(mean):
                if i == len(METHODS) or j == len(OPTIMIZERS):
                    label = (
                        f"{mean:.0f}" if iterations else
                        f"{mean:.5f}" if objective else f"{mean:.1f}%"
                    )
                elif iterations:
                    below = max(0.0, mean - lower[i, j])
                    above = max(0.0, upper[i, j] - mean)
                    label = (
                        rf"$\mathbf{{{mean:.0f}}}"
                        rf"_{{\mathbf{{-{below:.0f}}}}}"
                        rf"^{{\mathbf{{+{above:.0f}}}}}$"
                    )
                else:
                    label = (
                        f"{mean:.5f}\n\u00b1 {std[i, j]:.5f}" if objective
                        else f"{mean:.1f}%\n\u00b1 {std[i, j]:.1f}"
                    )
            else:
                label = "-\n-"
            color = "black"
            if iterations and np.isfinite(mean):
                red, green, blue, _ = image.cmap(image.norm(mean))
                if 0.2126 * red + 0.7152 * green + 0.0722 * blue < 0.45:
                    color = "white"
            ax.text(
                j, i, label, ha="center", va="center", color=color,
                fontsize=HEATMAP_FONT_SIZE, fontweight="bold", linespacing=1.05,
            )

    bar = fig.colorbar(image, ax=ax)
    bar.ax.set_xlabel(
        "Iterations to\nconvergence" if iterations else
        "Mean final\nobjective" if objective else "Objective\nreduction",
        fontweight="bold", fontsize=HEATMAP_FONT_SIZE, labelpad=12,
    )
    bar.outline.set_linewidth(1.4)
    bar.ax.tick_params(width=1.4, labelsize=HEATMAP_FONT_SIZE)
    for tick in bar.ax.get_yticklabels():
        tick.set_fontweight("bold")
    fig.tight_layout()
    if iterations:
        name = (
            f"matrix_iterations_{STUDY_CASE}.png" if STUDY_CASE else
            f"iterations_to_convergence_matrix_{n}point.png"
        )
    else:
        name = f"matrix_{STUDY_CASE}.png" if STUDY_CASE else f"final_results_matrix_{n}point.png"
    output = OUTPUT_DIR / name
    fig.savefig(output, dpi=220)
    pdf_output = output.with_suffix(".pdf")
    fig.savefig(pdf_output)
    plt.close(fig)
    print(f"Saved matrix: {output}")
    print(f"Saved PDF: {pdf_output}")


def convergence_ylim(data: pd.DataFrame, n: int) -> tuple[float, float] | None:
    lower_auto = CONV_YMIN.lower() in {"", "auto", "none"}
    upper_auto = CONV_YMAX.lower() in {"", "auto", "none"}
    default = CONV_YLIMITS.get(n) if CONV_POINT_LIMITS else None
    if default is not None:
        upper = default[1] if upper_auto else float(CONV_YMAX)
        if upper_auto and IS_DAE11:
            upper = max(upper, 0.055)
        return (
            default[0] if lower_auto else float(CONV_YMIN),
            upper,
        )
    if not lower_auto and not upper_auto:
        return float(CONV_YMIN), float(CONV_YMAX)
    if data.empty:
        return None
    window = data[
        (data["generation"] >= CONV_XMIN) & (data["generation"] <= CONV_XMAX)
    ]
    if window.empty:
        window = data
    mean = window["mean_best_fitness"].to_numpy(dtype=float)
    std = window["std_best_fitness"].fillna(0).to_numpy(dtype=float)
    values = np.r_[mean - std, mean + std]
    values = values[np.isfinite(values)]
    if not values.size:
        return None
    low, high = float(values.min()), float(values.max())
    pad = (
        max(0.001, abs(low) * 0.05) if np.isclose(low, high)
        else 0.04 * (high - low)
    )
    upper = high + pad if upper_auto else float(CONV_YMAX)
    if upper_auto and IS_DAE11:
        upper = max(upper, 0.055)
    return (low - pad if lower_auto else float(CONV_YMIN), upper)


def plot_wide_convergence(curves: pd.DataFrame, n: int) -> None:
    if CONV_XSCALE not in {"linear", "log"}:
        raise ValueError("PLOT_CONV_XSCALE must be 'linear' or 'log'")
    data = curves[curves["n_points"] == n]
    ylim = convergence_ylim(data, n)
    fig, axes = plt.subplots(
        1, len(OPTIMIZERS), figsize=(11.5, 4.2), sharex=True, sharey=True,
    )
    plotted_methods = set()
    for ax, optimizer in zip(axes, OPTIMIZERS):
        optimizer_data = data[data["optimizer"] == optimizer]
        for method in METHODS:
            curve = optimizer_data[optimizer_data["method"] == method]
            if curve.empty:
                continue
            curve = curve[curve["generation"] >= CONV_XMIN]
            if curve.empty:
                continue
            curve = curve.sort_values("generation", kind="mergesort")
            ax.plot(
                curve["generation"], curve["mean_best_fitness"],
                linewidth=CONV_LINEWIDTH, color=METHOD_COLORS[method],
                linestyle="-", zorder=CONV_Z_CURVE,
            )
            plotted_methods.add(method)
        if CONV_XSCALE == "log":
            ax.set_xscale("log")
            ticks = np.array([1, 10, 100, 500], dtype=float)
            ax.set_xticks(ticks[(ticks >= CONV_XMIN) & (ticks <= CONV_XMAX)])
            ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
        else:
            first_tick = np.ceil(CONV_XMIN / 100.0) * 100.0
            ax.set_xticks(np.arange(first_tick, CONV_XMAX + 1.0e-9, 100.0))
        ax.set_xlim(CONV_XMIN, CONV_XMAX)
        if ylim is not None:
            ax.set_ylim(*ylim)
            ticks = CONV_YTICKS.get(n)
            if ticks is not None:
                visible = ticks[(ticks >= min(ylim)) & (ticks <= max(ylim))]
                if visible.size:
                    ax.set_yticks(visible)
            ax.yaxis.set_major_formatter(FormatStrFormatter("%g" if IS_DAE11 else "%.3f"))
        if ax is axes[0]:
            ax.set_ylabel(CONV_YLABEL, fontsize=CONV_FONT_SIZE, labelpad=8)
        ax.minorticks_on()
        if CONV_XSCALE == "log":
            ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10)))
            ax.xaxis.set_minor_formatter(NullFormatter())
        else:
            ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.grid(False, which="both")
        xlow, xhigh = ax.get_xlim()
        ylow, yhigh = ax.get_ylim()
        for minor, color, width, zorder in (
            (False, "#cbd2d9", 0.75, CONV_Z_MAJOR_GRID),
            (True, "#e6e9ed", 0.45, CONV_Z_MINOR_GRID),
        ):
            for x in ax.get_xticks(minor=minor):
                if xlow < x < xhigh:
                    ax.axvline(x, color=color, linewidth=width, zorder=zorder)
            for y in ax.get_yticks(minor=minor):
                if ylow < y < yhigh:
                    ax.axhline(y, color=color, linewidth=width, zorder=zorder)
        ax.tick_params(
            axis="both", which="major", direction="in",
            width=1.2, length=4, labelsize=CONV_TICK_SIZE,
            top=False, right=False, zorder=CONV_Z_MAJOR_TICK,
        )
        ax.tick_params(
            axis="both", which="minor", direction="in", color="#aeb8c2",
            bottom=False, left=False, top=False, right=False,
        )
        x_minor = np.asarray(ax.get_xticks(minor=True), dtype=float)
        x_minor = x_minor[(xlow < x_minor) & (x_minor < xhigh)]
        y_minor = np.asarray(ax.get_yticks(minor=True), dtype=float)
        y_minor = y_minor[(ylow < y_minor) & (y_minor < yhigh)]
        ax.plot(
            x_minor, np.zeros_like(x_minor), linestyle="none", marker=TICKUP,
            markersize=2.5, markeredgewidth=0.7, color="#aeb8c2",
            transform=ax.get_xaxis_transform(), zorder=CONV_Z_MINOR_TICK,
        )
        ax.plot(
            np.zeros_like(y_minor), y_minor, linestyle="none", marker=TICKRIGHT,
            markersize=2.5, markeredgewidth=0.7, color="#aeb8c2",
            transform=ax.get_yaxis_transform(), zorder=CONV_Z_MINOR_TICK,
        )
        ax.xaxis.set_zorder(CONV_Z_MAJOR_TICK)
        ax.yaxis.set_zorder(CONV_Z_MAJOR_TICK)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.25)
        ax.spines["bottom"].set_linewidth(1.25)
        ax.spines["left"].set_zorder(CONV_Z_SPINE)
        ax.spines["bottom"].set_zorder(CONV_Z_SPINE)
        ax.text(
            0.04, 0.02, OPTIMIZER_LABELS[optimizer],
            transform=ax.transAxes, ha="left", va="bottom",
            fontweight="bold", fontsize=CONV_PANEL_SIZE, zorder=CONV_Z_LABEL,
            bbox={"facecolor": "white", "edgecolor": "none",
                  "boxstyle": "square,pad=0.2"},
        )
        if optimizer_data.empty:
            ax.text(
                0.5, 0.5, "no completed runs yet", transform=ax.transAxes,
                ha="center", va="center", color="#6b7280",
            )

    fig.supxlabel("Iteration", fontweight="bold", fontsize=CONV_FONT_SIZE, y=0.125)
    if plotted_methods:
        handles = [
            Patch(facecolor=METHOD_COLORS[method], edgecolor="none",
                  label=METHOD_LABELS[method])
            for method in METHODS if method in plotted_methods
        ]
        legend = fig.legend(
            handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.025),
            ncol=len(handles), frameon=False, handlelength=1.5, handleheight=0.75,
            columnspacing=1.35, borderaxespad=0.0,
        )
        for label in legend.get_texts():
            label.set_fontweight("bold")
            label.set_fontsize(CONV_LEGEND_SIZE)
    fig.align_ylabels(axes)
    fig.subplots_adjust(
        left=0.085, right=0.995, top=0.90,
        bottom=0.27 if plotted_methods else 0.17, wspace=0.22,
    )
    name = (
        f"conv_{STUDY_CASE}.png" if STUDY_CASE
        else f"convergence_by_optimizer_wide_seed_mean_{n}point.png"
    )
    output = OUTPUT_DIR / name
    fig.savefig(output, dpi=220, bbox_inches="tight", pad_inches=0.08)
    pdf_output = output.with_suffix(".pdf")
    fig.savefig(pdf_output, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"Saved wide convergence: {output}")
    print(f"Saved PDF: {pdf_output}")


def load_airfoil(path: Path) -> tuple[np.ndarray, np.ndarray]:
    coordinates = []
    for line in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        fields = re.split(r"[\s,]+", line.strip())
        if len(fields) < 2:
            continue
        try:
            x, z = float(fields[0]), float(fields[1])
        except ValueError:
            continue
        if np.isfinite(x) and np.isfinite(z):
            coordinates.append((x, z))
    if len(coordinates) < 3:
        raise ValueError(f"Expected two-column airfoil coordinates: {path}")
    coordinates = np.asarray(coordinates, dtype=float)
    return coordinates[:, 0], coordinates[:, 1]


def final_airfoil(opt_dir: Path) -> Path | None:
    pointer = opt_dir / "best_airfoil.dat"
    if not pointer.is_file():
        return None
    lines = [line.strip() for line in pointer.read_text(errors="replace").splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        return None
    path = Path(lines[-1])
    candidates = [path, opt_dir / path, opt_dir / path.parent.name / path.name]
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def airfoil_at_generation(opt_dir: Path, generation: int) -> Path | None:
    directory = (
        opt_dir / "Design_space" if generation == 1
        else opt_dir / f"generation_{generation - 1}"
    )
    final = final_airfoil(opt_dir)
    if final is not None and final.parent.resolve() == directory.resolve():
        return final
    candidates = list(directory.glob("airfoil_*.dat"))
    return candidates[0] if len(candidates) == 1 else None


def best_airfoil_for_case(case: list[Run]) -> dict | None:
    candidates = []
    for run in case:
        if run.status != "ok" or run.opt_dir is None:
            continue
        history = run.curve.sort_values(
            ["raw_cd", "generation"], kind="mergesort"
        )
        for _, row in history.iterrows():
            generation = int(round(float(row["generation"])))
            path = airfoil_at_generation(run.opt_dir, generation)
            if path is None:
                continue
            try:
                x, z = load_airfoil(path)
            except (OSError, ValueError) as exc:
                print(f"Skipping unreadable airfoil {path}: {exc}")
                continue
            candidates.append({
                "seed": run.seed, "generation": generation,
                "cd": float(row["raw_cd"]), "path": path, "x": x, "z": z,
                "case_dir": run.path.parent.parent,
            })
            break
    return min(
        candidates, key=lambda candidate: (
            candidate["cd"], candidate["seed"], candidate["generation"]
        ),
    ) if candidates else None


def airfoil_eval_settings(pick: dict) -> dict:
    input_path = pick["case_dir"] / "input_questions_1point.txt"
    if not input_path.is_file():
        raise SystemExit(f"Missing study operating conditions: {input_path}")
    values = [line.strip() for line in input_path.read_text().splitlines()]
    if len(values) < 6 or values[2] != "1":
        raise SystemExit(f"Expected one-point study input: {input_path}")
    if AIRFOIL_EVAL_POINTS < 2:
        raise SystemExit("PLOT_AOA_N_POINTS must be at least 2")
    settings = {
        "aoa_min": float(values[0]), "aoa_max": float(values[1]),
        "mach": float(values[3]), "reynolds": float(values[4]),
        "cl_target": float(values[5]), "n_points": AIRFOIL_EVAL_POINTS,
    }
    if settings["aoa_max"] <= settings["aoa_min"]:
        raise SystemExit(f"Invalid AoA sweep in {input_path}")
    if STUDY_CASE == "constrained":
        settings["cl_tolerance"] = float(os.environ.get(
            "PLOT_CL_TOLERANCE", os.environ.get("CL_TARGET_TOLERANCE", "0.02")
        ))
        settings["cm_target"] = float(os.environ.get(
            "PLOT_CM_TARGET", os.environ.get("CM_SELECTION_TARGET", "0.053639")
        ))
        settings["cm_tolerance"] = float(os.environ.get(
            "PLOT_CM_TOLERANCE", os.environ.get("CM_SELECTION_TOLERANCE", "0.02")
        ))
    else:
        settings["cl_tolerance"] = 0.0
        settings["cm_target"] = None
        settings["cm_tolerance"] = None
    return settings


def raifoil_executable() -> Path:
    backend = os.environ.get("RAIFOIL_BACKEND", "cpu").lower()
    if backend == "cpu":
        binary = os.environ.get("RAIFOIL_CPU_BIN", str(Path.home() / "rAIFoil_CPU_bin"))
    elif backend == "gpu":
        binary = os.environ.get("RAIFOIL_GPU_BIN", "rAIFoil")
    else:
        raise SystemExit("RAIFOIL_BACKEND must be cpu or gpu")
    resolved = shutil.which(binary)
    if resolved is None:
        raise SystemExit(f"rAIFoil executable not found: {binary}")
    return Path(resolved).resolve()


def evaluate_airfoil_cd(
    source: Path, settings: dict, binary: Path, fixed_aoa: float | None = None
) -> dict:
    binary_stat = binary.stat()
    key = hashlib.sha256(json.dumps({
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "settings": settings, "fixed_aoa": fixed_aoa,
        "binary": str(binary), "binary_size": binary_stat.st_size,
        "binary_mtime_ns": binary_stat.st_mtime_ns,
        "evaluation_sha256": hashlib.sha256(
            (STUDY_DIR.parent / "core" / "evaluation.py").read_bytes()
        ).hexdigest(),
    }, sort_keys=True).encode()).hexdigest()
    cache_dir = OUTPUT_DIR / "airfoil_cd_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{key}.json"
    if cache_path.is_file():
        return json.loads(cache_path.read_text())

    with tempfile.TemporaryDirectory(prefix="airfoil_eval_", dir=cache_dir) as tmp:
        work = Path(tmp)
        geometry_dir = work / "geometry"
        evaluation_dir = work / "evaluation"
        geometry_dir.mkdir()
        evaluation_dir.mkdir()
        shutil.copyfile(source, geometry_dir / "airfoil_1.dat")
        aoa_min = settings["aoa_min"] if fixed_aoa is None else fixed_aoa
        aoa_max = settings["aoa_max"] if fixed_aoa is None else fixed_aoa
        aoa_step = (
            (aoa_max - aoa_min) / (settings["n_points"] - 1)
            if fixed_aoa is None else 1.0
        )
        input_path = work / "input.txt"
        input_path.write_text(
            f"GEOMETRY_FOLDER = {geometry_dir}\n"
            f"AOA_MIN = {aoa_min:.17g}\n"
            f"AOA_MAX = {aoa_max:.17g}\n"
            f"AOA_STEP = {aoa_step:.17g}\n"
            f"MACH_MIN = {settings['mach']:.17g}\n"
            f"MACH_MAX = {settings['mach']:.17g}\n"
            f"MACH_STEP = {settings['mach']:.17g}\n"
            f"REYNOLDS_MIN = {settings['reynolds']:.17g}\n"
            f"REYNOLDS_MAX = {settings['reynolds']:.17g}\n"
            f"REYNOLDS_STEP = {settings['reynolds']:.17g}\n"
            "CI_LEVEL = 0.95\n"
            "OUTPUT_FILE = DD_80\n"
            f"OUTPUT_FOLDER = {evaluation_dir}\n"
            "OUTPUT_FORMAT = csv\n"
        )
        result = subprocess.run(
            [str(binary), "-i", str(input_path)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"rAIFoil failed for {source}:\n{result.stdout}\n{result.stderr}"
            )
        if fixed_aoa is None:
            env = os.environ.copy()
            env["CL_TARGET_TOLERANCE"] = str(settings["cl_tolerance"])
            env["CM_SELECTION_TARGET"] = (
                "" if settings["cm_target"] is None else str(settings["cm_target"])
            )
            env["CM_SELECTION_TOLERANCE"] = (
                "" if settings["cm_tolerance"] is None else str(settings["cm_tolerance"])
            )
            subprocess.run(
                [sys.executable, str(STUDY_DIR.parent / "run_concat.py"),
                 str(evaluation_dir), str(settings["cl_target"])],
                check=True, env=env, capture_output=True, text=True,
            )
            frame = pd.read_csv(evaluation_dir / "DD_80_combined.csv")
            if len(frame) != 1 or str(frame.iloc[0]["CL_target_reached"]).lower() != "true":
                raise RuntimeError(f"CL target not reached for {source}")
            if settings["cm_target"] is not None and str(
                frame.iloc[0]["CM_target_reached"]
            ).lower() != "true":
                raise RuntimeError(f"CM target not reached for {source}")
        else:
            frame = pd.read_csv(evaluation_dir / "DD_80_airfoil_1.csv")
            if len(frame) != 1:
                raise RuntimeError(f"Expected one baseline operating point at AoA={fixed_aoa}")
        row = frame.iloc[0]
        result = {
            "aoa": float(row["AOA"]),
            "cl": float(row["CL"]),
            "cd": float(row["CD"]),
        }
        if not all(np.isfinite(value) for value in result.values()) or result["cd"] <= 0:
            raise RuntimeError(f"Invalid rAIFoil result for {source}: {result}")
        if fixed_aoa is not None and not np.isclose(
            result["aoa"], fixed_aoa, rtol=0.0, atol=1e-5
        ):
            raise RuntimeError(f"Baseline AoA differs from requested {fixed_aoa}: {result}")
        cache_path.write_text(json.dumps(result, indent=2) + "\n")
        return result


def airfoil_comparison_label(optimized: dict, baseline: dict) -> str:
    return (
        rf"$\alpha_{{\mathrm{{opt.}}}}$ = {optimized['aoa']:.3f}$^\circ$" "\n"
        rf"$C_{{D,\mathrm{{opt.}}}}$ = {optimized['cd']:.5f}" "\n"
        rf"$\alpha_{{\mathrm{{DAE11}}}}$ = {baseline['aoa']:.3f}$^\circ$" "\n"
        rf"$C_{{D,\mathrm{{DAE11}}}}$ = {baseline['cd']:.5f}"
    )


def plot_airfoils(
    runs: dict[tuple[int, str, str], list[Run]], n: int
) -> None:
    baseline_path = REFERENCE_DIR / "DAE11.dat"
    if not baseline_path.is_file():
        raise SystemExit(f"Missing reference airfoil for PLOT={AIRFOIL_PLOT}: {baseline_path}")
    baseline = load_airfoil(baseline_path)

    combos = {
        (method, optimizer): best_airfoil_for_case(runs[(n, method, optimizer)])
        for method in METHODS for optimizer in OPTIMIZERS
    }
    if AIRFOIL_PLOT == "25":
        selected = combos
        fig, axes = plt.subplots(
            len(METHODS), len(OPTIMIZERS), figsize=(21, 8.8),
            sharex=True, sharey=True, squeeze=False,
        )
    else:
        selected = {
            method: min(
                (
                    (optimizer, combos[(method, optimizer)])
                    for optimizer in OPTIMIZERS
                    if combos[(method, optimizer)] is not None
                ),
                key=lambda item: (item[1]["cd"], OPTIMIZERS.index(item[0])),
                default=None,
            )
            for method in METHODS
        }
        fig, axes = plt.subplots(
            len(METHODS), 1, figsize=(9, 11), sharex=True,
            sharey=True, squeeze=False,
        )

    if AIRFOIL_PLOT == "25":
        plot_choices = [
            (method, optimizer, selected[(method, optimizer)])
            for method in METHODS for optimizer in OPTIMIZERS
        ]
    else:
        plot_choices = [
            (method, choice[0], choice[1])
            for method, choice in selected.items() if choice is not None
        ]
    operating_points = {}
    if any(pick is not None for _, _, pick in plot_choices):
        binary = raifoil_executable()
        for method, optimizer, pick in plot_choices:
            if pick is None:
                continue
            settings = airfoil_eval_settings(pick)
            optimized = evaluate_airfoil_cd(pick["path"], settings, binary)
            baseline_settings = {
                **settings, "cl_target": 1.25, "cl_tolerance": 0.0,
                "cm_target": None, "cm_tolerance": None,
            }
            baseline_cd = evaluate_airfoil_cd(baseline_path, baseline_settings, binary)
            operating_points[(method, optimizer)] = optimized, baseline_cd
            print(
                f"{METHOD_LABELS[method]}/{OPTIMIZER_LABELS[optimizer]}: "
                f"optimized AoA={optimized['aoa']:.6f} deg, "
                f"Cd={optimized['cd']:.8f}; "
                f"DAE11 at CL={baseline_cd['cl']:.6f}: "
                f"AoA={baseline_cd['aoa']:.6f} deg, Cd={baseline_cd['cd']:.8f}"
            )
            if not np.isclose(optimized["cd"], pick["cd"], rtol=0.01, atol=1e-4):
                print(
                    f"WARNING: re-evaluated Cd {optimized['cd']:.8f} differs "
                    f"from recorded Cd {pick['cd']:.8f} for {pick['path']}"
                )

    visible_picks = (
        combos.values() if AIRFOIL_PLOT == "25"
        else (choice[1] for choice in selected.values() if choice is not None)
    )
    all_z = [baseline[1]]
    all_z.extend(pick["z"] for pick in visible_picks if pick is not None)
    z = np.concatenate(all_z)
    pad = max(0.008, 0.08 * (float(z.max()) - float(z.min())))
    top_pad = 12 if AIRFOIL_PLOT == "25" else 10
    ylim = (float(z.min()) - pad, float(z.max()) + top_pad * pad)

    for i, method in enumerate(METHODS):
        for j, optimizer in enumerate(OPTIMIZERS if AIRFOIL_PLOT == "25" else ("best",)):
            ax = axes[i, j]
            ax.plot(
                *baseline, color="black",
                linestyle="-" if AIRFOIL_PLOT == "25" else "--",
                linewidth=1.45, alpha=1.0 if AIRFOIL_PLOT == "25" else 0.65,
                zorder=1,
            )
            pick = (
                selected[(method, optimizer)] if AIRFOIL_PLOT == "25"
                else selected[method][1] if selected[method] is not None else None
            )
            if pick is not None:
                ax.plot(
                    pick["x"], pick["z"],
                    color="#d62728" if AIRFOIL_PLOT == "25" else "#1f77b4",
                    linewidth=AIRFOIL_LINEWIDTH, zorder=3,
                )
                label = (
                    f"{METHOD_LABELS[method]} | "
                    f"{OPTIMIZER_LABELS[selected[method][0]]}"
                    if AIRFOIL_PLOT == "5" else ""
                )
                print(
                    f"Airfoil {METHOD_LABELS[method]}/{OPTIMIZER_LABELS[optimizer] if AIRFOIL_PLOT == '25' else OPTIMIZER_LABELS[selected[method][0]]}: "
                    f"seed={pick['seed']} generation={pick['generation']} "
                    f"Cd={pick['cd']:.8g} {pick['path']}"
                )
            else:
                label = "No feasible airfoil"
            if AIRFOIL_PLOT == "25":
                if i == len(METHODS) - 1:
                    ax.text(
                        0.5, -0.13, OPTIMIZER_LABELS[optimizer],
                        transform=ax.transAxes, ha="center", va="top",
                        clip_on=False,
                        fontsize=AIRFOIL_METHOD_FONT_SIZE, fontweight="bold",
                    )
                if j == 0:
                    ax.text(
                        -0.04, 0.5, METHOD_LABELS[method],
                        transform=ax.transAxes, ha="right", va="center",
                        rotation=30, rotation_mode="anchor", clip_on=False,
                        fontsize=AIRFOIL_METHOD_FONT_SIZE, fontweight="bold",
                    )
                if pick is None:
                    ax.text(
                        0.5, 0.5, label, transform=ax.transAxes,
                        ha="center", va="center", fontsize=AIRFOIL_TICK_SIZE,
                    )
                elif (method, optimizer) in operating_points:
                    ax.text(
                        0.98, 0.98,
                        airfoil_comparison_label(
                            *operating_points[(method, optimizer)]
                        ),
                        transform=ax.transAxes, ha="right", va="top",
                        fontsize=AIRFOIL_ANNOTATION_SIZE,
                        fontweight="bold", linespacing=1.15,
                    )
            else:
                ax.text(
                    0.97, 0.58, label, transform=ax.transAxes,
                    ha="right", va="top", fontsize=AIRFOIL_METHOD_FONT_SIZE,
                    fontweight="bold",
                )
                if selected[method] is not None and (
                    method, selected[method][0]
                ) in operating_points:
                    ax.text(
                        0.97, 0.98,
                        airfoil_comparison_label(
                            *operating_points[(method, selected[method][0])]
                        ),
                        transform=ax.transAxes, ha="right", va="top",
                        fontsize=AIRFOIL_ANNOTATION_SIZE, fontweight="bold",
                        linespacing=1.15,
                    )
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(*ylim)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(False)
            if AIRFOIL_PLOT == "25":
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
            else:
                ax.tick_params(
                    axis="both", direction="in", width=1.2, length=4,
                    labelsize=AIRFOIL_TICK_SIZE,
                )
                for spine in ax.spines.values():
                    spine.set_linewidth(1.25)

    handles = [
        Patch(
            facecolor="#d62728" if AIRFOIL_PLOT == "25" else "#1f77b4",
            edgecolor="none",
            label="Optimized airfoil" if AIRFOIL_PLOT == "25" else "Best seed",
        ),
        Patch(facecolor="black", edgecolor="none", label="DAE11 baseline"),
    ]
    legend = fig.legend(
        handles=handles, loc="lower center",
        bbox_to_anchor=(0.5, 0.005 if AIRFOIL_PLOT == "25" else 0.018),
        ncol=2, frameon=False, fontsize=AIRFOIL_LEGEND_SIZE,
        handlelength=1.15 if AIRFOIL_PLOT == "25" else 1.5,
        handleheight=0.6 if AIRFOIL_PLOT == "25" else 0.75,
        columnspacing=1.35, borderaxespad=0.0,
    )
    for text in legend.get_texts():
        text.set_fontweight("bold")
    if AIRFOIL_PLOT == "5":
        fig.supxlabel("x/c", fontsize=AIRFOIL_FONT_SIZE, fontweight="bold", y=0.08)
        fig.supylabel(
            "z/c", fontsize=AIRFOIL_FONT_SIZE, fontweight="bold", x=0.07,
        )
    fig.subplots_adjust(
        left=0.085 if AIRFOIL_PLOT == "25" else 0.12,
        right=0.995 if AIRFOIL_PLOT == "25" else 0.99,
        top=0.93 if AIRFOIL_PLOT == "25" else 0.95,
        bottom=0.15 if AIRFOIL_PLOT == "25" else 0.13,
        wspace=0.03 if AIRFOIL_PLOT == "25" else 0.12,
        hspace=0.05 if AIRFOIL_PLOT == "25" else 0.16,
    )
    suffix = f"_{STUDY_CASE}" if STUDY_CASE else ""
    output = OUTPUT_DIR / f"airfoils_{AIRFOIL_PLOT}{suffix}.png"
    fig.savefig(output, dpi=220, bbox_inches="tight", pad_inches=0.08)
    pdf_output = output.with_suffix(".pdf")
    fig.savefig(pdf_output, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"Saved {AIRFOIL_PLOT}-airfoil comparison: {output}")
    print(f"Saved PDF: {pdf_output}")


def main() -> None:
    if STUDY_CASE not in {"", "constrained", "unconstrained"}:
        raise SystemExit("SEED_STUDY_CASE must be constrained or unconstrained.")
    if METRIC not in {"reduction", "objective"}:
        raise SystemExit("SEED_SUMMARY_METRIC must be reduction or objective.")
    if AIRFOIL_PLOT not in {"0", "5", "25"}:
        raise SystemExit("PLOT must be 0, 5, or 25.")
    points = point_counts()
    if AIRFOIL_PLOT != "0":
        if not IS_DAE11 or points != (1,):
            raise SystemExit("PLOT=5/25 requires the one-point DAE11 study.")
        path = REFERENCE_DIR / "DAE11.dat"
        if not path.is_file():
            raise SystemExit(f"Missing reference airfoil for PLOT={AIRFOIL_PLOT}: {path}")
    seeds = discover_seeds()
    if not seeds:
        raise SystemExit(f"No seed directories found under {STUDY_DIR / STUDY_GLOB}")

    runs = collect_runs(seeds, points)
    matrix = build_matrix_summary(runs, points)
    curves = build_convergence_summary(runs)
    cauchy = build_cauchy_summary(runs)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(OUTPUT_DIR / "final_results_matrix_summary.csv", index=False)
    cauchy.to_csv(OUTPUT_DIR / "convergence_cauchy_seed_summary.csv", index=False)

    limits = heatmap_limits(matrix)
    cauchy_limits = cauchy_heatmap_limits(cauchy)
    for n in points:
        plot_matrix(matrix, n, limits)
        plot_matrix(cauchy, n, cauchy_limits, iterations=True)
        plot_wide_convergence(curves, n)
        if AIRFOIL_PLOT != "0":
            plot_airfoils(runs, n)
    print(f"Aggregated seeds: {', '.join(str(seed) for seed, _ in seeds)}")
    print(f"Wrote matrix and Cauchy CSVs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
