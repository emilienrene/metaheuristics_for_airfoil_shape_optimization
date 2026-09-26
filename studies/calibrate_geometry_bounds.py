#!/usr/bin/env python3
"""Calibrate parameterization bounds to a common geometry-space magnitude.

The numerical widths of PARSEC, CST, B-spline, Hicks-Henne, and FFD genes are
not directly comparable. This study samples every method's current bounds,
decodes the airfoils, and measures displacement from that method's reference
geometry on a common chordwise grid.

By default, the common target is the smallest initial 95th-percentile RMS
displacement. Consequently, calibration only contracts design spaces; it never
expands a method beyond its current bounds. The study does not modify the live
parameterization files. It writes proposed bounds and scale factors for review.

Run from optimizer/:

    python3 studies/calibrate_geometry_bounds.py

Useful overrides:

    python3 studies/calibrate_geometry_bounds.py --samples 5000 --seed 1
    python3 studies/calibrate_geometry_bounds.py --target-rms 0.012
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from parameterizations import get_parameterization  # noqa: E402
from parameterizations.airfoil_references import reference_table  # noqa: E402


METHODS = ("parsec", "cst", "bspline", "hickshenne", "ffd")
METHOD_LABELS = {
    "parsec": "PARSEC",
    "cst": "CST",
    "bspline": "B-spline",
    "hickshenne": "Hicks-Henne",
    "ffd": "FFD",
}
METHOD_COLORS = {
    "parsec": "#1f77b4",
    "cst": "#ff7f0e",
    "bspline": "#2ca02c",
    "hickshenne": "#d62728",
    "ffd": "#9467bd",
}

FONT_SIZE = 13
LINE_WIDTH = 2.2


@dataclass
class MethodStudy:
    name: str
    method: object
    setup: str
    hyperparams: Dict[str, object]
    center: np.ndarray
    original_bounds: np.ndarray
    unit_samples: np.ndarray
    reference_upper: np.ndarray
    reference_lower: np.ndarray
    representation_rms: float


@dataclass
class Evaluation:
    scale: float
    rms: np.ndarray
    maximum: np.ndarray
    valid: np.ndarray
    pointwise_p95: np.ndarray
    decode_failures: int

    @property
    def finite(self) -> np.ndarray:
        return np.isfinite(self.rms)

    def percentile_rms(self, percentile: float) -> float:
        values = self.rms[self.finite]
        return float(np.percentile(values, percentile)) if len(values) else float("nan")

    def percentile_maximum(self, percentile: float) -> float:
        values = self.maximum[np.isfinite(self.maximum)]
        return float(np.percentile(values, percentile)) if len(values) else float("nan")

    @property
    def valid_fraction(self) -> float:
        finite = self.finite
        return float(np.mean(self.valid[finite])) if np.any(finite) else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--stations", type=int, default=257)
    parser.add_argument("--percentile", type=float, default=95.0)
    parser.add_argument(
        "--target-rms",
        type=float,
        default=None,
        help="Common RMS target. Default: smallest initial percentile across methods.",
    )
    parser.add_argument("--max-iterations", type=int, default=8)
    parser.add_argument(
        "--relative-tolerance",
        type=float,
        default=0.005,
        help="Relative tolerance on the calibrated RMS percentile.",
    )
    parser.add_argument(
        "--selected-methods",
        type=Path,
        default=ROOT / "naca0012_inverse" / "selected_methods.csv",
        help="Optional inverse-study CSV defining each method's structural setup.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "geometry_bound_calibration",
    )
    args = parser.parse_args()
    if args.samples < 20:
        parser.error("--samples must be at least 20")
    if args.stations < 33:
        parser.error("--stations must be at least 33")
    if not 0.0 < args.percentile < 100.0:
        parser.error("--percentile must be between 0 and 100")
    if args.target_rms is not None and args.target_rms <= 0.0:
        parser.error("--target-rms must be positive")
    return args


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    structural_setups = load_selected_method_setups(args.selected_methods)
    x = cosine_grid(args.stations)
    rng = np.random.default_rng(args.seed)

    studies: Dict[str, MethodStudy] = {}
    initial: Dict[str, Evaluation] = {}

    print("Geometry-space bound calibration")
    print(f"  samples per method : {args.samples}")
    print(f"  RMS percentile     : {args.percentile:g}")
    print(f"  reference airfoil  : {os.environ.get('PARAM_REFERENCE_AIRFOIL', 'naca0012')}")
    if args.selected_methods.is_file():
        print(f"  method setups      : {args.selected_methods}")
    else:
        print("  method setups      : current/default environment")

    for name in METHODS:
        setup, hyperparams = structural_setups.get(name, ("current/default", {}))
        study = make_method_study(
            name=name,
            setup=setup,
            hyperparams=hyperparams,
            samples=args.samples,
            x=x,
            rng=rng,
        )
        studies[name] = study
        initial[name] = evaluate(study, scale=1.0, x=x)
        p_rms = initial[name].percentile_rms(args.percentile)
        print(
            f"  {METHOD_LABELS[name]:12s}: n={study.method.n_params:2d}, "
            f"initial p{args.percentile:g} RMS={p_rms:.6f}, "
            f"valid={100.0 * initial[name].valid_fraction:.1f}%"
        )

    initial_targets = [
        result.percentile_rms(args.percentile)
        for result in initial.values()
        if np.isfinite(result.percentile_rms(args.percentile))
    ]
    if not initial_targets:
        raise RuntimeError("No method produced finite decoded geometries.")

    target_rms = float(args.target_rms) if args.target_rms is not None else min(initial_targets)
    print(f"  common target      : {target_rms:.6f}")

    final: Dict[str, Evaluation] = {}
    convergence_rows: List[Dict[str, object]] = []
    for name in METHODS:
        scale, result, rows = calibrate_scale(
            studies[name],
            initial[name],
            target_rms=target_rms,
            percentile=args.percentile,
            x=x,
            max_iterations=args.max_iterations,
            relative_tolerance=args.relative_tolerance,
        )
        final[name] = result
        convergence_rows.extend(rows)
        print(
            f"  {METHOD_LABELS[name]:12s}: scale={scale:.6f}, "
            f"calibrated p{args.percentile:g} RMS="
            f"{result.percentile_rms(args.percentile):.6f}, "
            f"valid={100.0 * result.valid_fraction:.1f}%"
        )

    write_summary(
        args.output_dir / "geometry_bound_calibration_summary.csv",
        studies,
        initial,
        final,
        target_rms,
        args.percentile,
    )
    write_bounds(
        args.output_dir / "calibrated_parameter_bounds.csv",
        studies,
        final,
    )
    write_rows(
        args.output_dir / "calibration_convergence.csv",
        convergence_rows,
    )
    plot_comparison(
        args.output_dir / "geometry_bound_calibration.png",
        x,
        initial,
        final,
        target_rms,
        args.percentile,
    )

    print(f"Wrote calibration study to: {args.output_dir}")


def make_method_study(
    name: str,
    setup: str,
    hyperparams: Mapping[str, object],
    samples: int,
    x: np.ndarray,
    rng: np.random.Generator,
) -> MethodStudy:
    reference_name = os.environ.get("PARAM_REFERENCE_AIRFOIL", "naca0012")
    deformation_name = os.environ.get("DEFORMATION_BASELINE", reference_name)
    environment = {
        "PARAM_REFERENCE_AIRFOIL": reference_name,
        "DEFORMATION_BASELINE": deformation_name,
    }
    environment.update({key: str(value) for key, value in hyperparams.items()})

    with temporary_environment(environment):
        method = get_parameterization(name, n_points=max(401, len(x)))
        original_bounds = np.asarray(method.bounds, dtype=float).copy()

    if method.reference is None:
        center = np.mean(original_bounds, axis=1)
    else:
        center = np.asarray(method.reference, dtype=float).copy()
    if len(center) != method.n_params:
        raise ValueError(f"{name}: reference length does not match n_params")
    if np.any(center < original_bounds[:, 0] - 1e-12) or np.any(
        center > original_bounds[:, 1] + 1e-12
    ):
        raise ValueError(f"{name}: reference genotype lies outside its current bounds")

    reference_upper, reference_lower = decoded_surfaces_at(method, center, x)
    x_external, z_upper_external, z_lower_external = reference_table(reference_name)
    z_upper_external = np.interp(x, x_external, z_upper_external)
    z_lower_external = np.interp(x, x_external, z_lower_external)
    representation_rms = rms_displacement(
        x,
        reference_upper - z_upper_external,
        reference_lower - z_lower_external,
    )

    return MethodStudy(
        name=name,
        method=method,
        setup=setup,
        hyperparams=dict(hyperparams),
        center=center,
        original_bounds=original_bounds,
        unit_samples=latin_hypercube(samples, method.n_params, rng),
        reference_upper=reference_upper,
        reference_lower=reference_lower,
        representation_rms=representation_rms,
    )


def evaluate(study: MethodStudy, scale: float, x: np.ndarray) -> Evaluation:
    bounds = scaled_bounds(study.original_bounds, study.center, scale)
    params = bounds[:, 0] + study.unit_samples * (bounds[:, 1] - bounds[:, 0])
    n_samples = len(params)
    rms = np.full(n_samples, np.nan)
    maximum = np.full(n_samples, np.nan)
    valid = np.zeros(n_samples, dtype=bool)
    pointwise = np.full((n_samples, len(x)), np.nan)
    decode_failures = 0

    for index, genotype in enumerate(params):
        try:
            upper, lower = decoded_surfaces_at(study.method, genotype, x)
            delta_upper = upper - study.reference_upper
            delta_lower = lower - study.reference_lower
            if not np.all(np.isfinite(delta_upper)) or not np.all(np.isfinite(delta_lower)):
                raise ValueError("non-finite geometry")
            rms[index] = rms_displacement(x, delta_upper, delta_lower)
            maximum[index] = float(
                max(np.max(np.abs(delta_upper)), np.max(np.abs(delta_lower)))
            )
            pointwise[index] = np.maximum(np.abs(delta_upper), np.abs(delta_lower))
            valid[index] = bool(np.all(upper + 1e-10 >= lower))
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            decode_failures += 1

    finite_rows = np.any(np.isfinite(pointwise), axis=1)
    if np.any(finite_rows):
        pointwise_p95 = np.nanpercentile(pointwise[finite_rows], 95.0, axis=0)
    else:
        pointwise_p95 = np.full(len(x), np.nan)

    return Evaluation(
        scale=float(scale),
        rms=rms,
        maximum=maximum,
        valid=valid,
        pointwise_p95=pointwise_p95,
        decode_failures=decode_failures,
    )


def calibrate_scale(
    study: MethodStudy,
    initial: Evaluation,
    target_rms: float,
    percentile: float,
    x: np.ndarray,
    max_iterations: int,
    relative_tolerance: float,
) -> Tuple[float, Evaluation, List[Dict[str, object]]]:
    current = initial
    current_value = current.percentile_rms(percentile)
    if not np.isfinite(current_value) or current_value <= 0.0:
        raise RuntimeError(f"{study.name}: cannot calibrate a non-positive RMS reach")

    # The default target is the smallest original reach. Capping at one keeps
    # an explicit target from unexpectedly expanding the current design space.
    scale = min(1.0, target_rms / current_value)
    rows: List[Dict[str, object]] = []

    for iteration in range(max_iterations + 1):
        if iteration == 0 and abs(scale - 1.0) < 1e-14:
            current = initial
        else:
            current = evaluate(study, scale=scale, x=x)
        current_value = current.percentile_rms(percentile)
        relative_error = (
            (current_value - target_rms) / target_rms
            if np.isfinite(current_value)
            else float("nan")
        )
        rows.append({
            "method": study.name,
            "iteration": iteration,
            "scale": scale,
            "percentile_rms": current_value,
            "target_rms": target_rms,
            "relative_error": relative_error,
            "valid_fraction": current.valid_fraction,
        })
        if np.isfinite(relative_error) and abs(relative_error) <= relative_tolerance:
            break
        if not np.isfinite(current_value) or current_value <= 0.0:
            raise RuntimeError(f"{study.name}: calibration produced invalid RMS reach")

        proposed = scale * target_rms / current_value
        scale = float(np.clip(proposed, 1e-6, 1.0))

    return current.scale, current, rows


def scaled_bounds(bounds: np.ndarray, center: np.ndarray, scale: float) -> np.ndarray:
    return center[:, None] + float(scale) * (bounds - center[:, None])


def decoded_surfaces_at(
    method: object,
    params: np.ndarray,
    x_grid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    coords = np.asarray(method.decode(np.asarray(params, dtype=float)), dtype=float)
    n_upper = (len(coords) + 1) // 2
    upper = coords[:n_upper][::-1]
    lower_tail = coords[n_upper:]
    lower = np.vstack([upper[0], lower_tail])

    x_upper, z_upper = unique_surface(upper)
    x_lower, z_lower = unique_surface(lower)
    return (
        np.interp(x_grid, x_upper, z_upper),
        np.interp(x_grid, x_lower, z_lower),
    )


def unique_surface(surface: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    order = np.argsort(surface[:, 0])
    x = surface[order, 0]
    z = surface[order, 1]
    unique_x, inverse = np.unique(x, return_inverse=True)
    z_sum = np.zeros_like(unique_x)
    counts = np.zeros_like(unique_x)
    np.add.at(z_sum, inverse, z)
    np.add.at(counts, inverse, 1.0)
    return unique_x, z_sum / counts


def rms_displacement(x: np.ndarray, delta_upper: np.ndarray, delta_lower: np.ndarray) -> float:
    squared = 0.5 * (delta_upper ** 2 + delta_lower ** 2)
    return float(np.sqrt(np.trapz(squared, x)))


def cosine_grid(n_stations: int) -> np.ndarray:
    theta = np.linspace(0.0, np.pi, n_stations)
    return 0.5 * (1.0 - np.cos(theta))


def latin_hypercube(
    n_samples: int,
    n_dimensions: int,
    rng: np.random.Generator,
) -> np.ndarray:
    sample = np.empty((n_samples, n_dimensions), dtype=float)
    for column in range(n_dimensions):
        sample[:, column] = (rng.permutation(n_samples) + rng.random(n_samples)) / n_samples
    return sample


def load_selected_method_setups(
    path: Path,
) -> Dict[str, Tuple[str, Dict[str, object]]]:
    if not path.is_file():
        return {}
    setups: Dict[str, Tuple[str, Dict[str, object]]] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            name = (row.get("method") or "").strip().lower()
            if name not in METHODS:
                continue
            raw = row.get("hyperparams") or "{}"
            setups[name] = (
                row.get("setup") or "selected",
                json.loads(raw),
            )
    return setups


@contextmanager
def temporary_environment(values: Mapping[str, str]) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in values}
    try:
        for key, value in values.items():
            os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def write_summary(
    path: Path,
    studies: Mapping[str, MethodStudy],
    initial: Mapping[str, Evaluation],
    final: Mapping[str, Evaluation],
    target_rms: float,
    percentile: float,
) -> None:
    rows: List[Dict[str, object]] = []
    for name in METHODS:
        study = studies[name]
        before = initial[name]
        after = final[name]
        rows.append({
            "method": name,
            "setup": study.setup,
            "hyperparams": json.dumps(study.hyperparams, sort_keys=True),
            "n_params": study.method.n_params,
            "percentile": percentile,
            "target_rms": target_rms,
            "bound_scale": after.scale,
            "initial_percentile_rms": before.percentile_rms(percentile),
            "calibrated_percentile_rms": after.percentile_rms(percentile),
            "initial_percentile_max_displacement": before.percentile_maximum(percentile),
            "calibrated_percentile_max_displacement": after.percentile_maximum(percentile),
            "initial_valid_fraction": before.valid_fraction,
            "calibrated_valid_fraction": after.valid_fraction,
            "reference_representation_rms": study.representation_rms,
            "initial_decode_failures": before.decode_failures,
            "calibrated_decode_failures": after.decode_failures,
        })
    write_rows(path, rows)


def write_bounds(
    path: Path,
    studies: Mapping[str, MethodStudy],
    final: Mapping[str, Evaluation],
) -> None:
    rows: List[Dict[str, object]] = []
    for name in METHODS:
        study = studies[name]
        calibrated = scaled_bounds(
            study.original_bounds,
            study.center,
            final[name].scale,
        )
        for index, parameter_name in enumerate(study.method.param_names):
            rows.append({
                "method": name,
                "parameter": parameter_name,
                "reference": study.center[index],
                "original_lower": study.original_bounds[index, 0],
                "original_upper": study.original_bounds[index, 1],
                "bound_scale": final[name].scale,
                "calibrated_lower": calibrated[index, 0],
                "calibrated_upper": calibrated[index, 1],
            })
    write_rows(path, rows)


def write_rows(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_comparison(
    path: Path,
    x: np.ndarray,
    initial: Mapping[str, Evaluation],
    final: Mapping[str, Evaluation],
    target_rms: float,
    percentile: float,
) -> None:
    plt.rcParams.update({
        "font.size": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "axes.titlesize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE - 2,
        "ytick.labelsize": FONT_SIZE - 2,
        "legend.fontsize": FONT_SIZE - 2,
        "axes.linewidth": 1.5,
        "font.weight": "normal",
    })

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.2))
    plot_ecdf(axes[0, 0], initial, target_rms, "Original bounds")
    plot_ecdf(axes[0, 1], final, target_rms, "RMS-equivalent bounds")
    plot_chordwise_reach(axes[1, 0], x, initial, "Original bounds")
    plot_chordwise_reach(axes[1, 1], x, final, "RMS-equivalent bounds")

    axes[0, 0].set_ylabel("Cumulative fraction")
    axes[1, 0].set_ylabel(r"95th-percentile $|\Delta z|/c$")
    for ax in axes[0]:
        ax.set_xlabel(r"RMS displacement, $d_{\mathrm{RMS}}/c$")
    for ax in axes[1]:
        ax.set_xlabel(r"$x/c$")

    handles = [
        plt.Line2D(
            [0],
            [0],
            color=METHOD_COLORS[name],
            linewidth=LINE_WIDTH + 0.5,
            label=METHOD_LABELS[name],
        )
        for name in METHODS
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=len(METHODS),
        frameon=False,
        handlelength=2.5,
        columnspacing=1.8,
    )
    fig.suptitle(
        f"Geometric bound calibration at the {percentile:g}th RMS percentile",
        y=0.955,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.10, top=0.88, wspace=0.23, hspace=0.33)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_ecdf(
    ax: plt.Axes,
    evaluations: Mapping[str, Evaluation],
    target_rms: float,
    title: str,
) -> None:
    for name in METHODS:
        values = evaluations[name].rms[evaluations[name].finite]
        values = np.sort(values)
        cumulative = np.arange(1, len(values) + 1, dtype=float) / len(values)
        ax.plot(values, cumulative, color=METHOD_COLORS[name], linewidth=LINE_WIDTH)
    ax.axvline(target_rms, color="black", linewidth=1.4, linestyle="--")
    ax.set_ylim(0.0, 1.0)
    style_axis(ax, title)


def plot_chordwise_reach(
    ax: plt.Axes,
    x: np.ndarray,
    evaluations: Mapping[str, Evaluation],
    title: str,
) -> None:
    for name in METHODS:
        ax.plot(
            x,
            evaluations[name].pointwise_p95,
            color=METHOD_COLORS[name],
            linewidth=LINE_WIDTH,
        )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(bottom=0.0)
    style_axis(ax, title)


def style_axis(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, fontweight="bold")
    ax.grid(True, color="#d9d9d9", linewidth=0.8, alpha=0.8)
    ax.tick_params(direction="in", length=5, width=1.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


if __name__ == "__main__":
    main()
