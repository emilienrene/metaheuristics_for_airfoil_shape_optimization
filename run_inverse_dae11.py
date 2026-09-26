"""
Inverse-design DAE-11 while deformative methods start from another baseline.

This script is intentionally separate from the aerodynamic optimization loop.
It answers two geometry-only questions:

1. What genotype represents DAE-11 for each parameterization when
   Hicks-Henne and FFD deform a NACA0012 baseline?
2. What is the smallest hyperparameter setup that satisfies Kulfan's
   wind-tunnel tolerance?

Outputs are written to ``dae11_inverse/`` by default:

    summary.csv
        every attempted method/hyperparameter fit
    selected_methods.csv
        one selected fit per method: the minimal passing setup, or the closest
        failed setup if no attempted setup passes
    params/<method>_<setup>_params.csv
        parameter names and values for the selected fit
    fits/<method>_<setup>.dat
        decoded airfoil coordinates using the existing method.decode()
    errors/<method>_<setup>_errors.csv
        target/fitted values and errors at cosine-spaced DAE-11 stations

The target coordinates default to
``/home/erene/simulations/rAIFoil/dae11.dat``. Override that location with
the ``DAE11_DATA_FILE`` environment variable when needed.

Run from the optimizer directory:

    python3 run_inverse_dae11.py

By default each sweep tests the finest admissible hyperparameter increments and
stops at the first Kulfan pass. Use ``--full-sweep`` to continue through the
configured maximums.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import BSpline
from scipy.optimize import least_squares, lsq_linear

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from parameterizations.parsec import PARSEC
from parameterizations.cst import CST
from parameterizations.bspline import BSplineParam
from parameterizations.hickshenne import HicksHenne, _bump_matrix
from parameterizations.ffd import FFD, ffd_design_matrices as ffd_parameterization_design_matrices
from parameterizations.airfoil_references import (
    dae11_data_file,
    dae11_table,
    deformation_baseline_surfaces,
)


DAE11_N_STATIONS = 65


@dataclass
class FitResult:
    method: str
    setup: str
    hyperparams: dict[str, object]
    n_params: int
    n_points: int
    param_names: list[str]
    params: np.ndarray
    passes: bool
    max_abs_error: float
    max_upper_error: float
    max_lower_error: float
    upper_te_error: float
    lower_te_error: float
    worst_x: float
    worst_surface: str
    max_tol_ratio: float
    weighted_rmse: float
    mse: float
    success: bool
    message: str


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    make_output_dirs(out_dir)

    # Constructive methods are centred on the DAE-11 target. Deformative
    # methods use a distinct baseline so their representation test is not the
    # trivial zero-deformation solution.
    os.environ.pop("PARSEC_REFERENCE_PARAMS", None)
    os.environ["PARAM_REFERENCE_AIRFOIL"] = "dae11"
    os.environ["DEFORMATION_BASELINE"] = args.deformation_baseline
    os.environ["FFD_PIN_MODE"] = args.ffd_pin_mode
    target = target_airfoil()
    os.environ["DEFORMATION_TARGET_TE_UPPER"] = f"{target['zu'][-1]:.17g}"
    os.environ["DEFORMATION_TARGET_TE_LOWER"] = f"{target['zl'][-1]:.17g}"

    print("Constructive-method reference: DAE-11")
    print(f"DAE-11 coordinate file: {dae11_data_file()}")
    print(
        "Hicks-Henne/FFD deformation baseline: "
        f"{args.deformation_baseline}"
    )
    print(f"FFD pin mode: {args.ffd_pin_mode}")
    print(
        "Deformation-baseline trailing edge: "
        f"upper={target['zu'][-1]:.8f}, lower={target['zl'][-1]:.8f}"
    )

    if args.respect_bounds:
        print("Fitting mode: respecting current optimizer bounds.")
    else:
        print("Fitting mode: unconstrained representation fit.")
        print("Use --respect-bounds to test whether the fit also lies inside method.bounds.")

    write_target_outputs(out_dir, target)

    all_results: list[FitResult] = []
    selected: list[FitResult] = []

    for method_name, fitter in [
        ("parsec", fit_parsec_sweep),
        ("cst", fit_cst_sweep),
        ("bspline", fit_bspline_sweep),
        ("hickshenne", fit_hickshenne_sweep),
        ("ffd", fit_ffd_sweep),
    ]:
        print(f"\n=== {method_name} ===")
        results = fitter(args, target)
        all_results.extend(results)
        best = select_result(results)
        selected.append(best)
        status = "PASS" if best.passes else "FAIL"
        print(
            f"{status}: {best.setup}, n_params={best.n_params}, "
            f"max_error={best.max_abs_error:.6e}, "
            f"max_tol_ratio={best.max_tol_ratio:.3f}"
        )
        write_selected_outputs(out_dir, best, target)

    write_summary(out_dir / "summary.csv", all_results)
    write_summary(out_dir / "selected_methods.csv", selected)

    print(f"\nWrote inverse-design outputs to: {out_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit DAE-11 geometry while Hicks-Henne and FFD deform a "
            "separately selected baseline."
        )
    )
    parser.add_argument("--out-dir", default="dae11_inverse")
    parser.add_argument(
        "--deformation-baseline",
        default="naca0012",
        choices=("naca0012", "rae2822", "modified_rae2822", "dae11"),
        help=(
            "Baseline deformed by Hicks-Henne and FFD. The default NACA0012 "
            "makes their DAE-11 representation test nontrivial; selecting "
            "dae11 intentionally tests the zero-deformation case."
        ),
    )
    parser.add_argument(
        "--ffd-pin-mode",
        default="corners",
        choices=("corners", "end_columns"),
        help="FFD topology used during the inverse sweep (default: corners).",
    )
    parser.add_argument("--max-cst-order", type=int, default=12)
    parser.add_argument("--max-bspline-n-ctrl", type=int, default=21)
    parser.add_argument("--max-hh-bumps", type=int, default=20)
    parser.add_argument("--max-ffd-m", type=int, default=14)
    parser.add_argument("--parsec-starts", type=int, default=40)
    parser.add_argument(
        "--full-sweep",
        action="store_true",
        help="Continue testing larger setups after the first Kulfan pass.",
    )
    parser.add_argument(
        "--fit-n-points",
        type=int,
        default=20001,
        help=(
            "Number of points used when decoding fitted geometry for tolerance "
            "checks and output .dat files. A fine grid avoids false failures "
            "near the leading edge."
        ),
    )
    parser.add_argument(
        "--respect-bounds",
        action="store_true",
        help="Constrain the inverse fit to the current optimizer method.bounds.",
    )
    parser.add_argument("--no-bounds", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def make_output_dirs(out_dir: Path) -> None:
    (out_dir / "params").mkdir(parents=True, exist_ok=True)
    (out_dir / "fits").mkdir(parents=True, exist_ok=True)
    (out_dir / "errors").mkdir(parents=True, exist_ok=True)


def target_airfoil() -> dict[str, np.ndarray]:
    # Resample the coordinate file on the same 65-point cosine distribution
    # used by the representation-tolerance comparison.
    x, z_upper, z_lower = dae11_table(DAE11_N_STATIONS)
    tol = kulfan_tolerance(x)
    return {"x": x, "zu": z_upper, "zl": z_lower, "tol": tol}


def kulfan_tolerance(x: np.ndarray) -> np.ndarray:
    return np.where(x < 0.2, 4.0e-4, 8.0e-4)


def least_squares_residual_weights(x: np.ndarray) -> np.ndarray:
    """Residual multipliers corresponding to Masters et al. equation 9."""
    return np.sqrt(np.where(x < 0.2, 2.0, 1.0))


def fit_parsec_sweep(args: argparse.Namespace, target: dict[str, np.ndarray]) -> list[FitResult]:
    method = PARSEC(n_points=args.fit_n_points)
    params, success, message = fit_parsec(method, target, args)
    return [assess_fit(method, params, "parsec", "fixed", {}, success, message, target)]


def fit_cst_sweep(args: argparse.Namespace, target: dict[str, np.ndarray]) -> list[FitResult]:
    results = []
    for order in range(2, args.max_cst_order + 1):
        with temp_env({"CST_ORDER": str(order)}):
            method = CST(order=order, n_points=args.fit_n_points)
        x = target["x"]
        basis = cst_basis(x, order)
        stride = order + 2
        n_params = method.n_params

        a_up = np.zeros((len(x), n_params))
        a_lo = np.zeros((len(x), n_params))
        a_up[:, :stride] = basis
        a_lo[:, stride:2 * stride] = basis
        a_up[:, -1] = x
        a_lo[:, -1] = -x

        params, success, message = fit_linear_system(
            a_up=a_up,
            a_lo=a_lo,
            offset_up=np.zeros_like(x),
            offset_lo=np.zeros_like(x),
            method=method,
            target=target,
            use_bounds=args.respect_bounds,
        )
        results.append(assess_fit(
            method, params, "cst", f"CST_ORDER={order}", {"CST_ORDER": order},
            success, message, target
        ))
        if results[-1].passes and not args.full_sweep:
            break
    return results


def fit_bspline_sweep(args: argparse.Namespace, target: dict[str, np.ndarray]) -> list[FitResult]:
    results = []
    for n_ctrl in range(5, args.max_bspline_n_ctrl + 1):
        with temp_env({"BSPLINE_N_CTRL": str(n_ctrl)}):
            method = BSplineParam(n_ctrl=n_ctrl, n_points=args.fit_n_points)

        x = target["x"]
        ctrl_x = bspline_control_x(n_ctrl, getattr(method, "spacing", "cosine"))
        u = bspline_u_for_x(x, n_ctrl, method.degree, ctrl_x)
        b = bspline_basis_matrix(n_ctrl, method.degree, u)

        n_free = n_ctrl - 2
        n_params = 2 * n_free

        a_up = np.zeros((len(x), n_params))
        a_lo = np.zeros((len(x), n_params))

        a_up[:, :n_free] = b[:, 1:-1]
        a_lo[:, n_free:] = b[:, 1:-1]

        offset_up = b[:, 0] * target["zu"][0] + b[:, -1] * target["zu"][-1]
        offset_lo = b[:, 0] * target["zl"][0] + b[:, -1] * target["zl"][-1]

        params, success, message = fit_linear_system(
            a_up=a_up,
            a_lo=a_lo,
            offset_up=offset_up,
            offset_lo=offset_lo,
            method=method,
            target=target,
            use_bounds=args.respect_bounds,
        )
        results.append(assess_fit(
            method, params, "bspline", f"BSPLINE_N_CTRL={n_ctrl}",
            {"BSPLINE_N_CTRL": n_ctrl}, success, message, target
        ))
        if results[-1].passes and not args.full_sweep:
            break
    return results


def fit_hickshenne_sweep(args: argparse.Namespace, target: dict[str, np.ndarray]) -> list[FitResult]:
    results = []
    for n_bumps in range(1, args.max_hh_bumps + 1):
        with temp_env({"HH_N_BUMPS": str(n_bumps)}):
            method = HicksHenne(n_bumps=n_bumps, n_points=args.fit_n_points)

        x = target["x"]
        bumps = _bump_matrix(x, n_bumps)
        z_base_up, z_base_lo = deformation_baseline_surfaces(
            x, args.deformation_baseline
        )
        n_params = method.n_params

        a_up = np.zeros((len(x), n_params))
        a_lo = np.zeros((len(x), n_params))
        a_up[:, :n_bumps] = bumps
        a_lo[:, n_bumps:] = bumps

        params, success, message = fit_linear_system(
            a_up=a_up,
            a_lo=a_lo,
            offset_up=z_base_up,
            offset_lo=z_base_lo,
            method=method,
            target=target,
            use_bounds=args.respect_bounds,
        )
        results.append(assess_fit(
            method, params, "hickshenne", f"HH_N_BUMPS={n_bumps}",
            {"HH_N_BUMPS": n_bumps}, success, message, target
        ))
        if results[-1].passes and not args.full_sweep:
            break
    return results


def fit_ffd_sweep(args: argparse.Namespace, target: dict[str, np.ndarray]) -> list[FitResult]:
    results = []
    configs = []
    for n_rows in (2, 3, 4):
        for m in range(2, args.max_ffd_m + 1):
            with temp_env({"FFD_N_ROWS": str(n_rows), "FFD_M": str(m)}):
                method = FFD(n_rows=n_rows, m=m, n_points=25)
            configs.append((method.n_params, n_rows, m))

    for _, n_rows, m in sorted(configs):
        with temp_env({"FFD_N_ROWS": str(n_rows), "FFD_M": str(m)}):
            method = FFD(n_rows=n_rows, m=m, n_points=args.fit_n_points)

        x = target["x"]
        z_base_up, z_base_lo = deformation_baseline_surfaces(
            x, args.deformation_baseline
        )
        a_up, a_lo = ffd_parameterization_design_matrices(
            x=x,
            z_base_up=z_base_up,
            z_base_lo=z_base_lo,
            n_rows=n_rows,
            m=m,
            z_min=method.z_min,
            z_max=method.z_max,
            pin_mode=method.pin_mode,
        )

        params, success, message = fit_linear_system(
            a_up=a_up,
            a_lo=a_lo,
            offset_up=z_base_up,
            offset_lo=z_base_lo,
            method=method,
            target=target,
            use_bounds=args.respect_bounds,
        )
        setup = f"FFD_PIN_MODE={method.pin_mode};FFD_N_ROWS={n_rows};FFD_M={m}"
        results.append(assess_fit(
            method, params, "ffd", setup,
            {"FFD_N_ROWS": n_rows, "FFD_M": m, "FFD_PIN_MODE": method.pin_mode},
            success, message, target
        ))
        if results[-1].passes and not args.full_sweep:
            break
    return results


def fit_linear_system(
    a_up: np.ndarray,
    a_lo: np.ndarray,
    offset_up: np.ndarray,
    offset_lo: np.ndarray,
    method,
    target: dict[str, np.ndarray],
    use_bounds: bool,
) -> tuple[np.ndarray, bool, str]:
    w = least_squares_residual_weights(target["x"])

    a = np.vstack([a_up * w[:, None], a_lo * w[:, None]])
    b = np.concatenate([
        (target["zu"] - offset_up) * w,
        (target["zl"] - offset_lo) * w,
    ])

    bounds = np.asarray(method.bounds, dtype=float)
    lower = bounds[:, 0].copy()
    upper = bounds[:, 1].copy()
    fixed = np.isclose(lower, upper)

    if not use_bounds:
        # "Unconstrained" means that variable design parameters may leave
        # their optimizer bounds. Structurally fixed parameters (for example
        # CST/PARSEC trailing-edge thickness) must remain fixed, otherwise the
        # inverse result cannot be used as a generation-zero reference.
        lower[~fixed] = -np.inf
        upper[~fixed] = np.inf

    active = ~fixed

    params = np.zeros(method.n_params)
    params[fixed] = lower[fixed]

    if not np.any(active):
        return params, True, "all parameters fixed"

    b_active = b - a[:, fixed] @ params[fixed]
    a_active = a[:, active]
    lb_active = lower[active]
    ub_active = upper[active]

    try:
        if use_bounds:
            result = lsq_linear(
                a_active,
                b_active,
                bounds=(lb_active, ub_active),
                tol=1e-12,
                lsmr_tol="auto",
                max_iter=2000,
            )
            params[active] = result.x
            return params, bool(result.success), str(result.message)

        params[active], *_ = np.linalg.lstsq(a_active, b_active, rcond=None)
        return params, True, "unbounded least squares"
    except Exception as exc:
        return np.clip(params, lower, upper), False, repr(exc)


def fit_parsec(
    method: PARSEC,
    target: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> tuple[np.ndarray, bool, str]:
    bounds = np.asarray(method.bounds, dtype=float)
    lower = bounds[:, 0].copy()
    upper = bounds[:, 1].copy()
    fixed = np.isclose(lower, upper)

    if not args.respect_bounds:
        span = upper - lower
        lower[~fixed] = lower[~fixed] - 2.0 * span[~fixed]
        upper[~fixed] = upper[~fixed] + 2.0 * span[~fixed]

    active = ~fixed

    seeds = parsec_initial_guesses(method, target, lower, upper)
    rng = np.random.default_rng(12345)
    for _ in range(max(0, args.parsec_starts)):
        seeds.append(lower + rng.random(method.n_params) * (upper - lower))

    best_params = None
    best_score = np.inf
    best_success = False
    best_message = "not run"

    for seed in seeds:
        x0_full = clip_with_fixed(seed, lower, upper)
        x0 = x0_full[active]

        try:
            result = least_squares(
                lambda p: parsec_residual(p, x0_full, active, method, target),
                x0,
                bounds=(lower[active], upper[active]),
                ftol=1e-12,
                xtol=1e-12,
                gtol=1e-12,
                x_scale="jac",
                max_nfev=20000,
            )
            params = x0_full.copy()
            params[active] = result.x
            score = float(np.sum(parsec_residual(result.x, x0_full, active, method, target) ** 2))
            if score < best_score:
                best_params = params
                best_score = score
                best_success = bool(result.success)
                best_message = str(result.message)
        except Exception as exc:
            if best_params is None:
                best_params = x0_full
                best_message = repr(exc)

    assert best_params is not None
    return clip_with_fixed(best_params, lower, upper), best_success, best_message


def parsec_residual(
    active_params: np.ndarray,
    base_params: np.ndarray,
    active: np.ndarray,
    method: PARSEC,
    target: dict[str, np.ndarray],
) -> np.ndarray:
    params = base_params.copy()
    params[active] = active_params

    try:
        zu, zl = decoded_surfaces_at(method, params, target["x"])
        w = least_squares_residual_weights(target["x"])
        return np.concatenate([
            (zu - target["zu"]) * w,
            (zl - target["zl"]) * w,
        ])
    except Exception:
        return np.full(2 * len(target["x"]), 1.0e6)


def parsec_initial_guesses(
    method: PARSEC,
    target: dict[str, np.ndarray],
    lower: np.ndarray,
    upper: np.ndarray,
) -> list[np.ndarray]:
    x = target["x"]
    zu = target["zu"]
    zl = target["zl"]

    i_up = int(np.argmax(zu))
    i_lo = int(np.argmin(zl))

    r_le_vals = []
    for xi, zui, zli in zip(x[1:8], zu[1:8], zl[1:8]):
        if xi > 0:
            r_le_vals.append(zui ** 2 / (2.0 * xi))
            r_le_vals.append(zli ** 2 / (2.0 * xi))
    r_le = float(np.median(r_le_vals)) if r_le_vals else method.reference[0]

    dzu = np.gradient(zu, x, edge_order=2)
    dzl = np.gradient(zl, x, edge_order=2)
    d2zu = np.gradient(dzu, x, edge_order=2)
    d2zl = np.gradient(dzl, x, edge_order=2)

    theta_u = np.arctan(dzu[-1])
    theta_l = np.arctan(dzl[-1])
    alpha = 0.5 * (theta_u + theta_l)
    beta = theta_l - theta_u

    geom = np.array([
        r_le,
        x[i_up],
        zu[i_up],
        d2zu[i_up],
        x[i_lo],
        zl[i_lo],
        d2zl[i_lo],
        0.0,
        0.0,
        alpha,
        beta,
    ])

    midpoint = 0.5 * (lower + upper)
    return [
        clip_with_fixed(geom, lower, upper),
        clip_with_fixed(method.reference.copy(), lower, upper),
        clip_with_fixed(midpoint, lower, upper),
    ]


def assess_fit(
    method,
    params: np.ndarray,
    method_name: str,
    setup: str,
    hyperparams: dict[str, object],
    success: bool,
    message: str,
    target: dict[str, np.ndarray],
) -> FitResult:
    zu_fit, zl_fit = decoded_surfaces_at(method, params, target["x"])
    err_up = np.abs(zu_fit - target["zu"])
    err_lo = np.abs(zl_fit - target["zl"])
    tol = target["tol"]
    max_upper = float(np.max(err_up))
    max_lower = float(np.max(err_lo))
    upper_te = float(err_up[-1])
    lower_te = float(err_lo[-1])
    max_abs = float(max(max_upper, max_lower))
    ratios = np.concatenate([err_up / tol, err_lo / tol])
    worst_i = int(np.argmax(ratios))
    if worst_i < len(target["x"]):
        worst_x = float(target["x"][worst_i])
        worst_surface = "upper"
    else:
        worst_x = float(target["x"][worst_i - len(target["x"])])
        worst_surface = "lower"
    max_ratio = float(np.max(ratios))
    residual = np.concatenate([
        (zu_fit - target["zu"]) / tol,
        (zl_fit - target["zl"]) / tol,
    ])
    weighted_rmse = float(np.sqrt(np.mean(residual ** 2)))
    mse = float(np.mean(np.concatenate([
        zu_fit - target["zu"],
        zl_fit - target["zl"],
    ]) ** 2))
    passes = bool(np.all(err_up <= tol) and np.all(err_lo <= tol))

    return FitResult(
        method=method_name,
        setup=setup,
        hyperparams=hyperparams,
        n_params=method.n_params,
        n_points=int(getattr(method, "n_points", 1000)),
        param_names=method.param_names,
        params=np.asarray(params, dtype=float),
        passes=passes,
        max_abs_error=max_abs,
        max_upper_error=max_upper,
        max_lower_error=max_lower,
        upper_te_error=upper_te,
        lower_te_error=lower_te,
        worst_x=worst_x,
        worst_surface=worst_surface,
        max_tol_ratio=max_ratio,
        weighted_rmse=weighted_rmse,
        mse=mse,
        success=success,
        message=message,
    )


def select_result(results: list[FitResult]) -> FitResult:
    passing = [r for r in results if r.passes]
    if passing:
        return min(passing, key=lambda r: (r.n_params, r.max_tol_ratio, r.weighted_rmse))
    return min(results, key=lambda r: (r.max_tol_ratio, r.weighted_rmse, r.n_params))


def decoded_surfaces_at(method, params: np.ndarray, x_ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coords = np.asarray(method.decode(np.asarray(params, dtype=float)), dtype=float)
    n_upper = (len(coords) + 1) // 2

    upper = coords[:n_upper][::-1]
    lower = coords[n_upper:]

    x_upper = upper[:, 0]
    z_upper = upper[:, 1]

    x_lower = np.concatenate([[0.0], lower[:, 0]])
    z_lower = np.concatenate([[0.0], lower[:, 1]])

    zu = np.interp(x_ref, x_upper, z_upper)
    zl = np.interp(x_ref, x_lower, z_lower)
    return zu, zl


def cst_basis(x: np.ndarray, order: int) -> np.ndarray:
    c = np.sqrt(x) * (1.0 - x)
    bernstein = np.column_stack([
        c * comb(order, i) * x ** i * (1.0 - x) ** (order - i)
        for i in range(order + 1)
    ])
    # Match the augmented CST basis used by parameterizations.cst.CST.decode().
    lem = (c * np.sqrt(x) * (1.0 - x) ** (order - 0.5)).reshape(-1, 1)
    return np.hstack([bernstein, lem])


def clamped_knots(n_ctrl: int, degree: int) -> np.ndarray:
    n_inner = n_ctrl - degree - 1
    inner = np.linspace(0.0, 1.0, n_inner + 2)[1:-1]
    return np.concatenate([
        np.zeros(degree + 1),
        inner,
        np.ones(degree + 1),
    ])


def bspline_basis_matrix(n_ctrl: int, degree: int, u_eval: np.ndarray) -> np.ndarray:
    knots = clamped_knots(n_ctrl, degree)
    basis = np.zeros((len(u_eval), n_ctrl))
    for j in range(n_ctrl):
        coeff = np.zeros(n_ctrl)
        coeff[j] = 1.0
        basis[:, j] = BSpline(knots, coeff, degree)(u_eval)
    return basis


def bspline_control_x(n_ctrl: int, spacing: str = "cosine") -> np.ndarray:
    spacing = spacing.strip().lower()
    ctrl_x = np.zeros(n_ctrl)
    ctrl_x[-1] = 1.0

    i = np.arange(1, n_ctrl - 1, dtype=float)
    denom = float(n_ctrl - 2)
    if spacing == "uniform":
        ctrl_x[1:-1] = (i - 1.0) / denom
    elif spacing == "cosine":
        ctrl_x[1:-1] = 0.5 * (1.0 - np.cos(np.pi * (i - 1.0) / denom))
    else:
        raise ValueError(
            f"Unknown BSPLINE_X_SPACING={spacing!r}; use 'cosine' or 'uniform'."
        )

    return ctrl_x


def bspline_u_for_x(
    x_target: np.ndarray,
    n_ctrl: int,
    degree: int,
    ctrl_x: np.ndarray,
) -> np.ndarray:
    u_dense = np.linspace(0.0, 1.0, max(20001, 400 * n_ctrl))
    x_dense = bspline_basis_matrix(n_ctrl, degree, u_dense) @ ctrl_x
    x_dense = np.maximum.accumulate(np.clip(x_dense, 0.0, 1.0))
    x_unique, unique_idx = np.unique(x_dense, return_index=True)
    return np.interp(np.clip(x_target, 0.0, 1.0), x_unique, u_dense[unique_idx])


def hicks_henne_bumps(x: np.ndarray, n_bumps: int) -> np.ndarray:
    return _bump_matrix(x, n_bumps)


def bernstein_matrix(degree: int, x: np.ndarray) -> np.ndarray:
    return np.column_stack([
        comb(degree, i) * x ** i * (1.0 - x) ** (degree - i)
        for i in range(degree + 1)
    ])


def write_target_outputs(out_dir: Path, target: dict[str, np.ndarray]) -> None:
    ref_path = out_dir / "dae11_target_signed.dat"
    x = target["x"]
    upper = np.column_stack([x[::-1], target["zu"][::-1]])
    lower = np.column_stack([x[1:], target["zl"][1:]])
    coords = np.vstack([upper, lower])
    np.savetxt(ref_path, coords, fmt="%.8f", header="X Z", comments="")


def write_selected_outputs(out_dir: Path, result: FitResult, target: dict[str, np.ndarray]) -> None:
    tag = safe_tag(result.method, result.setup)

    params_path = out_dir / "params" / f"{tag}_params.csv"
    with params_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "value"])
        for name, value in zip(result.param_names, result.params):
            writer.writerow([name, f"{value:.16g}"])

    method = instantiate_for_result(result)
    coords = method.decode(result.params)
    np.savetxt(
        out_dir / "fits" / f"{tag}.dat",
        coords,
        fmt="%.8f",
        header="X Z",
        comments="",
    )

    zu_fit, zl_fit = decoded_surfaces_at(method, result.params, target["x"])
    err = pd.DataFrame({
        "x": target["x"],
        "tol": target["tol"],
        "z_upper_ref": target["zu"],
        "z_upper_fit": zu_fit,
        "upper_error": zu_fit - target["zu"],
        "upper_abs_error": np.abs(zu_fit - target["zu"]),
        "z_lower_ref": target["zl"],
        "z_lower_fit": zl_fit,
        "lower_error": zl_fit - target["zl"],
        "lower_abs_error": np.abs(zl_fit - target["zl"]),
    })
    err.to_csv(out_dir / "errors" / f"{tag}_errors.csv", index=False)


def instantiate_for_result(result: FitResult):
    if result.method == "parsec":
        return PARSEC(n_points=result.n_points)
    if result.method == "cst":
        with temp_env({"CST_ORDER": str(result.hyperparams["CST_ORDER"])}):
            return CST(n_points=result.n_points)
    if result.method == "bspline":
        with temp_env({"BSPLINE_N_CTRL": str(result.hyperparams["BSPLINE_N_CTRL"])}):
            return BSplineParam(n_points=result.n_points)
    if result.method == "hickshenne":
        with temp_env({"HH_N_BUMPS": str(result.hyperparams["HH_N_BUMPS"])}):
            return HicksHenne(n_points=result.n_points)
    if result.method == "ffd":
        env = {
            "FFD_N_ROWS": str(result.hyperparams["FFD_N_ROWS"]),
            "FFD_M": str(result.hyperparams["FFD_M"]),
            "FFD_PIN_MODE": str(result.hyperparams.get("FFD_PIN_MODE", "corners")),
        }
        with temp_env(env):
            return FFD(n_points=result.n_points)
    raise ValueError(f"Unknown method: {result.method}")


def write_summary(path: Path, results: list[FitResult]) -> None:
    rows = []
    for result in results:
        rows.append({
            "method": result.method,
            "setup": result.setup,
            "hyperparams": json.dumps(result.hyperparams, sort_keys=True),
            "n_params": result.n_params,
            "n_points": result.n_points,
            "passes_kulfan": result.passes,
            "max_abs_error": result.max_abs_error,
            "max_upper_error": result.max_upper_error,
            "max_lower_error": result.max_lower_error,
            "upper_te_error": result.upper_te_error,
            "lower_te_error": result.lower_te_error,
            "worst_x": result.worst_x,
            "worst_surface": result.worst_surface,
            "max_tol_ratio": result.max_tol_ratio,
            "weighted_rmse": result.weighted_rmse,
            "mse": result.mse,
            "success": result.success,
            "message": result.message,
            "params_json": json.dumps(result.params.tolist()),
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def safe_tag(method: str, setup: str) -> str:
    tag = f"{method}_{setup}"
    return (
        tag.replace("=", "")
        .replace(";", "_")
        .replace(",", "_")
        .replace(" ", "_")
    )


def clip_with_fixed(values: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).copy()
    fixed = np.isclose(lower, upper)
    values[fixed] = lower[fixed]
    values[~fixed] = np.clip(values[~fixed], lower[~fixed], upper[~fixed])
    return values


@contextmanager
def temp_env(values: dict[str, str]):
    old = {key: os.environ.get(key) for key in values}
    try:
        for key, value in values.items():
            os.environ[key] = value
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


if __name__ == "__main__":
    main()
