from __future__ import annotations
"""
PARSEC parameterization.

The decode is lifted verbatim from the original Random_Gen0_Generation.py /
Create_Next_Gen_Phenotype.py so geometry output is identical to the current
pipeline. It now lives in exactly one place instead of two.

Gene order (canonical, do not reorder):
    0  R_LE        leading-edge radius
    1  X_UP        upper-crest x
    2  Z_UP        upper-crest z
    3  Z_XXUP      upper-crest curvature
    4  X_LO        lower-crest x
    5  Z_LO        lower-crest z
    6  Z_XXLO      lower-crest curvature
    7  Z_TE        trailing-edge z
    8  delta_Z_TE  trailing-edge thickness
    9  alpha_TE    trailing-edge direction
   10  beta_TE     trailing-edge wedge angle
"""

import json
import os

import numpy as np
from numpy.linalg import solve

from .base import Parameterization
from .airfoil_references import NACA0012_LE_RADIUS, parameterization_reference_name

_CURV_SCALE_3 = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
_CURV_SCALE_4 = np.array([-1 / 4, 3 / 4, 15 / 4, 15 / 4, 63 / 4, 99 / 4])


_RAE2822_PARSEC_REFERENCE = np.array([
    0.008460265,    # R_LE
    0.431618294,    # X_UP
    0.063070436,    # Z_UP
    0.658283043,    # Z_XXUP
    0.343022668,    # X_LO
   -0.058797639,    # Z_LO
   -1.227121705,    # Z_XXLO
    0.000246375,    # Z_TE
   -0.000033200,    # delta_Z_TE
   -0.116662805,    # alpha_TE
    0.164290466,    # beta_TE
], dtype=float)

_NACA0012_PARSEC_REFERENCE = np.array([
    NACA0012_LE_RADIUS,  # R_LE
    0.299528435,         # X_UP
    0.060007111,         # Z_UP
   -0.453952234,         # Z_XXUP
    0.299528435,         # X_LO
   -0.060007111,         # Z_LO
    0.453952234,         # Z_XXLO
    0.0,                 # Z_TE
    0.0,                 # delta_Z_TE
    0.0,                 # alpha_TE
    0.288678395,         # beta_TE
], dtype=float)

# Least-squares PARSEC representation of the UIUC DAE-11 coordinates. The
# coordinate file remains the source of truth for the other parameterizations;
# PARSEC needs an explicit set of its eleven geometric descriptors.
_DAE11_PARSEC_REFERENCE = np.array([
    0.013830882138,   # R_LE
    0.356519944368,   # X_UP
    0.129726031456,   # Z_UP
   -10.150703102150,  # Z_XXUP
    0.047700360855,   # X_LO
   -0.022571610046,   # Z_LO
    4.899264313913,   # Z_XXLO
    0.0,              # Z_TE
    0.0,              # delta_Z_TE
   -0.023470919136,   # alpha_TE
    0.068960269631,   # beta_TE
], dtype=float)


class PARSEC(Parameterization):

    name = "parsec"
    mutation_mode = "additive"

    def __init__(self, n_points: int = 1000):
        self.n_points = n_points
        reference_name = parameterization_reference_name()
        if reference_name == "naca0012":
            self.reference = _NACA0012_PARSEC_REFERENCE.copy()
        elif reference_name == "dae11":
            self.reference = _DAE11_PARSEC_REFERENCE.copy()
        else:
            self.reference = _RAE2822_PARSEC_REFERENCE.copy()

        configured_reference = os.environ.get("PARSEC_REFERENCE_PARAMS")
        if configured_reference:
            try:
                values = np.asarray(json.loads(configured_reference), dtype=float)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(
                    "PARSEC_REFERENCE_PARAMS must be a JSON array of 11 numbers"
                ) from exc
            if values.shape != (11,) or not np.all(np.isfinite(values)):
                raise ValueError(
                    "PARSEC_REFERENCE_PARAMS must contain exactly 11 finite numbers"
                )
            self.reference = values.copy()
        x = np.linspace(0, 1, n_points).reshape([n_points, 1])
        self._x = x
        x_matrix = np.zeros([n_points, 6])
        for i in range(n_points):
            for j in range(6):
                x_matrix[i, j] = x[i, 0] ** (j + 0.5)
        self._x_matrix = x_matrix

    @property
    def param_names(self) -> list[str]:
        return [
            "R_LE", "X_UP", "Z_UP", "Z_XXUP",
            "X_LO", "Z_LO", "Z_XXLO",
            "Z_TE", "delta_Z_TE", "alpha_TE", "beta_TE",
        ]

    @property
    def bounds(self) -> np.ndarray:
        r = self.reference
        return np.array([
            [max(0.001, r[0] - 0.006), r[0] + 0.006],       # R_LE
            [max(0.02,  r[1] - 0.08),  min(0.95, r[1] + 0.08)],  # X_UP
            [max(0.00,  r[2] - 0.03),  min(0.16, r[2] + 0.03)],  # Z_UP
            [r[3] - 0.90, r[3] + 0.90],                     # Z_XXUP
            [max(0.02,  r[4] - 0.08),  min(0.95, r[4] + 0.08)],  # X_LO
            [max(-0.16, r[5] - 0.03),  min(0.02, r[5] + 0.03)],  # Z_LO
            [r[6] - 0.90, r[6] + 0.90],                     # Z_XXLO
            [r[7] - 0.004, r[7] + 0.004],                   # Z_TE
            [r[8], r[8]],                                   # delta_Z_TE
            [r[9] - 0.25, r[9] + 0.25],                     # alpha_TE
            [max(0.01, r[10] - 0.12), r[10] + 0.12],        # beta_TE
        ])

    @staticmethod
    def _coeff_matrix(x_crest: float) -> np.ndarray:
        C = np.ones([6, 6])
        for i in range(1, 6):
            C[5, i] = 0
        for i in range(6):
            C[1, i] = x_crest ** (0.5 + i)
            C[2, i] = 0.5 + i
            C[3, i] = x_crest ** (-0.5 + i)
            C[4, i] = x_crest ** (-1.5 + i)
        C[3, :] *= _CURV_SCALE_3
        C[4, :] *= _CURV_SCALE_4
        return C

    def decode(self, params: np.ndarray) -> np.ndarray:
        (R_LE, X_UP, Z_UP, Z_XXUP, X_LO, Z_LO, Z_XXLO,
         Z_TE, delta_Z_TE, alpha_TE, beta_TE) = (float(p) for p in params)

        C_up = self._coeff_matrix(X_UP)
        C_low = self._coeff_matrix(X_LO)

        b_up = np.zeros(6)
        b_up[0] = Z_TE + delta_Z_TE / 2
        b_up[1] = Z_UP
        b_up[2] = np.tan(alpha_TE - beta_TE / 2)
        b_up[3] = 0
        b_up[4] = Z_XXUP
        b_up[5] = (2 * R_LE) ** 0.5

        b_low = np.zeros(6)
        b_low[0] = Z_TE - delta_Z_TE / 2
        b_low[1] = Z_LO
        b_low[2] = np.tan(alpha_TE + beta_TE / 2)
        b_low[3] = 0
        b_low[4] = Z_XXLO
        b_low[5] = -(2 * R_LE) ** 0.5

        a_up = solve(C_up, b_up)
        a_low = solve(C_low, b_low)

        Z_up = np.dot(self._x_matrix, a_up).reshape([self.n_points, 1])
        Z_low = np.dot(self._x_matrix, a_low).reshape([self.n_points, 1])

        X_upper = self._x[::-1]      # TE -> LE
        Z_upper = Z_up[::-1]
        X_lower = self._x[1:]        # LE -> TE
        Z_lower = Z_low[1:]

        X_total = np.vstack((X_upper, X_lower))
        Z_total = np.vstack((Z_upper, Z_lower))
        return np.hstack((X_total, Z_total))
