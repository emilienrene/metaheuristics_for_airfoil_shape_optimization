"""
Parametric B-spline airfoil parameterization.

This follows the constructive B-spline setup used in the reference study:
each surface is a parametric curve

    X(u), Z(u) = sum_i N_i,k(u) P_i

with fixed chordwise control-point coordinates and optimized vertical
coordinates only. The leading edge and trailing edge are pinned, while every
internal upper/lower ordinate is an independent gene.

The default chordwise control-point distribution is cosine spacing, with the
first free control point sharing x=0 with the pinned leading-edge point. That
duplicate leading-edge x coordinate gives the curve enough freedom to form a
rounded nose without tying the lower surface to the upper surface.

Gene layout (2*(N_CTRL - 2) total, default N_CTRL=9 gives 14 genes):
    0       ... N-3      z1_up ... z(N-2)_up
    N-2     ... 2N-5     z1_lo ... z(N-2)_lo

References:
    Lepine, J. et al. (2000). Optimized Nonuniform Rational B-Spline Geometrical
    Representation for Aerodynamic Design of Wings. AIAA Journal, 39(11).
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import BSpline

from .base import Parameterization
from .rae2822 import rae2822_table


def _clamped_knots(n_ctrl: int, degree: int) -> np.ndarray:
    n_inner = n_ctrl - degree - 1
    inner = np.linspace(0, 1, n_inner + 2)[1:-1]
    return np.concatenate([
        np.zeros(degree + 1),
        inner,
        np.ones(degree + 1),
    ])


def _basis_matrix(n_ctrl: int, degree: int, u_eval: np.ndarray) -> np.ndarray:
    knots = _clamped_knots(n_ctrl, degree)
    B = np.zeros((len(u_eval), n_ctrl))
    for j in range(n_ctrl):
        c = np.zeros(n_ctrl)
        c[j] = 1.0
        B[:, j] = BSpline(knots, c, degree)(u_eval)
    return B


def _control_x(n_ctrl: int, spacing: str = "cosine") -> np.ndarray:
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


def _u_for_x(
    x_target: np.ndarray,
    n_ctrl: int,
    degree: int,
    ctrl_x: np.ndarray,
) -> np.ndarray:
    u_dense = np.linspace(0.0, 1.0, max(20001, 400 * n_ctrl))
    x_dense = _basis_matrix(n_ctrl, degree, u_dense) @ ctrl_x
    x_dense = np.maximum.accumulate(np.clip(x_dense, 0.0, 1.0))
    x_unique, unique_idx = np.unique(x_dense, return_index=True)
    return np.interp(np.clip(x_target, 0.0, 1.0), x_unique, u_dense[unique_idx])


def _fit_rae2822_bspline_reference(
    n_ctrl: int,
    degree: int,
    spacing: str = "cosine",
) -> np.ndarray:
    x, z_up, z_lo = rae2822_table()
    ctrl_x = _control_x(n_ctrl, spacing)
    u_fit = _u_for_x(x, n_ctrl, degree, ctrl_x)
    B = _basis_matrix(n_ctrl, degree, u_fit)

    A = B[:, 1:-1]
    offset_up = B[:, 0] * z_up[0] + B[:, -1] * z_up[-1]
    offset_lo = B[:, 0] * z_lo[0] + B[:, -1] * z_lo[-1]

    params_up, _, _, _ = np.linalg.lstsq(A, z_up - offset_up, rcond=None)
    params_lo, _, _, _ = np.linalg.lstsq(A, z_lo - offset_lo, rcond=None)
    return np.concatenate([params_up, params_lo])


class BSplineParam(Parameterization):
    """
    Cubic clamped parametric B-spline airfoil parameterization.

    Parameters
    ----------
    n_ctrl : int
        Number of control points per surface including pinned endpoints (default 9).
    degree : int
        B-spline degree (default 3).
    n_points : int
        Number of chord-wise evaluation points (default 1000).
    """

    name = "bspline"
    mutation_mode = "additive"

    def __init__(self, n_ctrl: int = 9, degree: int = 3, n_points: int = 1000):
        import os
        n_ctrl = int(os.environ.get("BSPLINE_N_CTRL", n_ctrl))
        spacing = os.environ.get("BSPLINE_X_SPACING", "cosine")
        if n_ctrl < degree + 2:
            raise ValueError(
                f"n_ctrl={n_ctrl} too small for degree={degree}; need >= {degree + 2}"
            )

        self.n_ctrl   = n_ctrl
        self.degree   = degree
        self.n_points = n_points
        self.spacing  = spacing.strip().lower()

        # Parametric evaluation grid; x/c is itself a B-spline curve.
        u_eval = np.linspace(0.0, 1.0, n_points)
        self._u_eval = u_eval

        # precompute collocation matrix in u-space
        self._B = _basis_matrix(n_ctrl, degree, u_eval)
        self._ctrl_x = _control_x(n_ctrl, self.spacing)
        self._x_eval = np.maximum.accumulate(np.clip(self._B @ self._ctrl_x, 0.0, 1.0))

        x_ref, z_up_ref, z_lo_ref = rae2822_table()
        self._z_le_up = float(z_up_ref[0])
        self._z_le_lo = float(z_lo_ref[0])
        self._z_te_up = float(z_up_ref[-1])
        self._z_te_lo = float(z_lo_ref[-1])

        # RAE2822 reference in the same independent upper/lower control layout.
        self.reference = _fit_rae2822_bspline_reference(n_ctrl, degree, self.spacing)

    @property
    def param_names(self) -> list[str]:
        up = [f"z{i}_up" for i in range(1, self.n_ctrl-1)]
        lo = [f"z{i}_lo" for i in range(1, self.n_ctrl-1)]
        return up + lo

    @property
    def bounds(self) -> np.ndarray:
        import os
        body_margin = float(os.environ.get("BSPLINE_RAE_MARGIN", 0.04))
        le_margin = float(os.environ.get("BSPLINE_RAE_LE_MARGIN", 0.008))
        te_margin = float(os.environ.get("BSPLINE_RAE_TE_MARGIN", 0.008))
        le_width = float(os.environ.get("BSPLINE_RAE_LE_WIDTH", 0.08))
        te_width = float(os.environ.get("BSPLINE_RAE_TE_WIDTH", 0.12))

        x_free = self._ctrl_x[1:-1]
        margins = np.full_like(x_free, body_margin, dtype=float)

        if le_width > 0.0:
            le_weight = np.clip(1.0 - x_free / le_width, 0.0, 1.0)
            margins = margins * (1.0 - le_weight) + le_margin * le_weight

        if te_width > 0.0:
            te_weight = np.clip(1.0 - (1.0 - x_free) / te_width, 0.0, 1.0)
            margins = margins * (1.0 - te_weight) + te_margin * te_weight

        if len(margins):
            margins[0] = min(margins[0], le_margin)
            margins[-1] = min(margins[-1], te_margin)

        all_margins = np.concatenate([margins, margins])
        lower = self.reference - all_margins
        upper = self.reference + all_margins

        if len(margins):
            n_free = self.n_ctrl - 2
            r_le_ref = float(os.environ.get("BSPLINE_RAE_LE_RADIUS_REF", 0.008460265))
            r_le_min = float(os.environ.get(
                "BSPLINE_RAE_LE_RADIUS_MIN",
                max(0.001, r_le_ref - 0.006),
            ))
            r_le_max = float(os.environ.get("BSPLINE_RAE_LE_RADIUS_MAX", r_le_ref + 0.006))
            le_min_scale = np.sqrt(max(r_le_min, 1e-12) / r_le_ref)
            le_max_scale = np.sqrt(max(r_le_max, 1e-12) / r_le_ref)
            eps = float(os.environ.get("BSPLINE_RAE_LE_SIGN_EPS", 1e-4))

            up_ref = max(float(self.reference[0]), eps)
            lo_ref = min(float(self.reference[n_free]), -eps)
            lo_mag = abs(lo_ref)

            lower[0] = max(lower[0], up_ref * le_min_scale)
            upper[0] = min(upper[0], up_ref * le_max_scale)
            lower[n_free] = max(lower[n_free], -lo_mag * le_max_scale)
            upper[n_free] = min(upper[n_free], -lo_mag * le_min_scale)

        return np.column_stack([lower, upper])

    def decode(self, params: np.ndarray) -> np.ndarray:
        n_free = self.n_ctrl - 2

        z_up_free = np.asarray(params[:n_free], dtype=float)
        z_lo_free = np.asarray(params[n_free:2*n_free], dtype=float)

        ctrl_z_up = np.concatenate([[self._z_le_up], z_up_free, [self._z_te_up]])
        ctrl_z_lo = np.concatenate([[self._z_le_lo], z_lo_free, [self._z_te_lo]])

        Z_up  = self._B @ ctrl_z_up
        Z_low = self._B @ ctrl_z_lo

        x = self._x_eval
        X_upper = x[::-1].reshape(-1, 1)
        Z_upper = Z_up[::-1].reshape(-1, 1)
        X_lower = x[1:].reshape(-1, 1)
        Z_lower = Z_low[1:].reshape(-1, 1)

        return np.hstack((
            np.vstack((X_upper, X_lower)),
            np.vstack((Z_upper, Z_lower)),
        ))
