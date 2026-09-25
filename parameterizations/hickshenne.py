"""
Hicks-Henne bump function airfoil parameterization.

The airfoil is represented as a baseline airfoil plus a linear combination
of bump functions applied independently to the upper and lower surfaces:

    z_up(x)  = z_base_up(x)  + sum_i  a_i * f_i(x)
    z_low(x) = z_base_low(x) + sum_i  b_i * f_i(x)

Each bump function follows the augmented sine formulation (Hicks & Henne 1978):

    f_i(x) = sin( pi * x ^ (ln(0.5) / ln(h_i)) ) ^ t_i

Bump centres h_i follow the cosine placement used in the paper:

    h_i = 0.5 * (1 - cos( i*pi / (n+1) ))    i = 1, ..., n

This is the default. A uniform experimental spacing is available with
``HH_CENTER_MODE=uniform`` but should not be used for paper-conformant runs.

Bump powers follow the paper's fixed graded schedule:

    t_i = 2 * ((n-i) / (n-1))^3 + 1

Only the amplitudes a_i / b_i are genes. Centres and powers are fully
determined by n_bumps and environment settings, and are fixed throughout the
optimization.

The first and last bump amplitudes can be capped separately to protect the
leading- and trailing-edge tangents while keeping the paper's fixed basis.

Gene layout  (2 * N_BUMPS total, default N_BUMPS=8 gives 16 genes):
    0   ... N-1    a0 ... a(N-1)    upper surface amplitudes
    N   ... 2N-1   b0 ... b(N-1)    lower surface amplitudes

References:
    Hicks, R. & Henne, P. (1978). Wing Design by Numerical Optimization.
    Journal of Aircraft, 15(7), 407-412.
    Wu, H. et al. (1991). Practical applications of adjoint-based aerodynamic
    shape optimization. AIAA Paper.
"""

from __future__ import annotations

import numpy as np
import os

from .base import Parameterization
from .airfoil_references import deformation_baseline_name, deformation_baseline_surfaces


def _centres(n: int) -> np.ndarray:
    """Bump centres, i = 1 ... n."""
    mode = os.environ.get("HH_CENTER_MODE", "cosine").strip().lower()
    i = np.arange(1, n + 1)
    if mode in {"cos", "cosine", "wu"}:
        return 0.5 * (1 - np.cos(i * np.pi / (n + 1)))
    if mode in {"uniform", "linear"}:
        return i / (n + 1)
    raise ValueError(f"Unknown HH_CENTER_MODE={mode!r}. Use 'uniform' or 'cosine'.")


def _widths(n: int) -> np.ndarray:
    """Bump powers, i = 1 ... n."""
    mode = os.environ.get("HH_WIDTH_MODE", "graded").strip().lower()
    if mode in {"constant", "fixed"}:
        power = float(os.environ.get("HH_BUMP_POWER", 1.0))
        return np.full(n, power)
    if mode not in {"graded", "current", "old"}:
        raise ValueError(
            f"Unknown HH_WIDTH_MODE={mode!r}. Use 'constant' or 'graded'."
        )
    if n == 1:
        return np.array([float(os.environ.get("HH_BUMP_POWER", 1.0))])
    i = np.arange(1, n + 1)
    return 2 * ((n - i) / (n - 1)) ** 3 + 1


def _bump_matrix(x: np.ndarray, n: int) -> np.ndarray:
    """
    Build (n_points, n) bump basis matrix.
    Each column is one bump function evaluated over x.
    """
    h = _centres(n)
    t = _widths(n)
    x_safe = np.clip(x, 1e-10, 1 - 1e-10)
    B = np.zeros((len(x), n))
    for i in range(n):
        exp = np.log(0.5) / np.log(h[i])
        B[:, i] = np.sin(np.pi * x_safe ** exp) ** t[i]
    B[0, :] = 0.0
    B[-1, :] = 0.0
    return B


class HicksHenne(Parameterization):
    """
    Hicks-Henne bump-function parameterization on a configurable baseline.

    The normal optimization default is NACA0012. Set
    DEFORMATION_BASELINE=rae2822 for the RAE2822-to-NACA0012 inverse study.

    Parameters
    ----------
    n_bumps : int
        Number of bump functions per surface (default 8).
    n_points : int
        Number of chord-wise evaluation points (default 1000).
    """

    name = "hickshenne"
    mutation_mode = "additive"

    def __init__(self, n_bumps: int = 8, n_points: int = 1000):
        n_bumps = int(os.environ.get("HH_N_BUMPS", n_bumps))
        self.n_bumps  = n_bumps
        self.n_points = n_points

        x = np.linspace(0, 1, n_points)
        self._x = x
        self.baseline_name = deformation_baseline_name()
        self._z_base_up, self._z_base_lo = deformation_baseline_surfaces(
            x,
            self.baseline_name,
        )
        self._B = _bump_matrix(x, n_bumps)

        # Zero amplitudes reproduce the selected deformation baseline.
        self.reference = np.zeros(2 * n_bumps)

    @property
    def param_names(self) -> list[str]:
        up = [f"a{i}_up" for i in range(self.n_bumps)]
        lo = [f"a{i}_lo" for i in range(self.n_bumps)]
        return up + lo

    @property
    def bounds(self) -> np.ndarray:
        amp = float(os.environ.get(
            "HH_AMPLITUDE",
            os.environ.get("HH_RAE_AMPLITUDE", 0.05),
        ))
        amps = np.full(self.n_bumps, amp)

        if self.n_bumps > 0:
            h = _centres(self.n_bumps)

            le_amp = float(os.environ.get(
                "HH_LE_AMPLITUDE",
                os.environ.get("HH_RAE_LE_AMPLITUDE", 0.005),
            ))
            amps[0] = min(amp, le_amp)

            exp_te = np.log(0.5) / np.log(h[-1])
            te_slope_delta = float(os.environ.get(
                "HH_TE_SLOPE_DELTA",
                os.environ.get("HH_RAE_TE_SLOPE_DELTA", 0.15),
            ))
            te_amp = float(os.environ.get(
                "HH_TE_AMPLITUDE",
                os.environ.get(
                    "HH_RAE_TE_AMPLITUDE",
                    te_slope_delta / (np.pi * exp_te),
                ),
            ))
            amps[-1] = min(amp, te_amp)

        all_amps = np.concatenate([amps, amps])
        return np.column_stack([
            -all_amps,
            all_amps,
        ])

    def decode(self, params: np.ndarray) -> np.ndarray:
        a_up = np.asarray(params[:self.n_bumps], dtype=float)
        a_lo = np.asarray(params[self.n_bumps:], dtype=float)

        Z_up  = self._z_base_up + self._B @ a_up
        Z_low = self._z_base_lo + self._B @ a_lo

        x = self._x
        X_upper = x[::-1].reshape(-1, 1)
        Z_upper = Z_up[::-1].reshape(-1, 1)
        X_lower = x[1:].reshape(-1, 1)
        Z_lower = Z_low[1:].reshape(-1, 1)

        return np.hstack((
            np.vstack((X_upper, X_lower)),
            np.vstack((Z_upper, Z_lower)),
        ))
