"""
CST (Class-Shape Transformation) parameterization — Kulfan (2008)
with Leading Edge Modification (LEM) term.

Surface shape function:

    S(x) = Σ_{i=0}^{N} aᵢ · Bᵢₙ(x)  +  a_{N+1} · x^0.5 · (1-x)^(N-0.5)
                                          #_________________________________#
                                                     LEM term

Full surface equation:

    z(x) = C(x) · S(x) + x · dz_te
    C(x) = x^0.5 · (1-x)          (airfoil class function, N1=0.5 N2=1.0)
    Bᵢₙ(x) = C(n,i) · xⁱ · (1-x)^(n-i)   (Bernstein basis)

Upper and lower surfaces are independent. With N=5 the gene layout is:

    0   … N       A0_up … A5_up      upper Bernstein coefficients  (6)
    N+1            Alem_up            upper LEM coefficient          (1)
    N+2 … 2N+2    A0_lo … A5_lo      lower Bernstein coefficients  (6)
    2N+3           Alem_lo            lower LEM coefficient          (1)
    2N+4           dz_te              trailing-edge half-thickness   (1)
                                                              total: 15

References:
    Kulfan, B. (2008). Universal Parametric Geometry Representation Method.
    Journal of Aircraft, 45(1), 142-158.
"""

from __future__ import annotations

import json
import os

import numpy as np
from math import comb as _comb

from .base import Parameterization
from .airfoil_references import reference_table


def _surface_basis(x: np.ndarray, order: int) -> np.ndarray:
    c = x ** 0.5 * (1 - x)
    bernstein = np.column_stack([
        c * _comb(order, i) * x ** i * (1 - x) ** (order - i)
        for i in range(order + 1)
    ])
    lem = (c * x ** 0.5 * (1 - x) ** (order - 0.5)).reshape(-1, 1)
    return np.hstack([bernstein, lem])


def _fit_cst_reference(order: int) -> tuple[np.ndarray, np.ndarray]:
    x, z_up, z_lo = reference_table()
    basis = _surface_basis(x, order)
    ref_up, _, _, _ = np.linalg.lstsq(basis, z_up, rcond=None)
    ref_lo, _, _, _ = np.linalg.lstsq(basis, z_lo, rcond=None)
    return ref_up, ref_lo


class CST(Parameterization):

    name = "cst"
    mutation_mode = "additive"

    def __init__(self, order: int = 5, n_points: int = 1000):
        order = int(os.environ.get("CST_ORDER", order))
        self.order = order
        self.n_points = n_points

        x = np.linspace(0, 1, n_points)
        self._x = x

        # class function
        C = x ** 0.5 * (1 - x)

        # Bernstein basis  (n_points, order+1)
        n = order
        B = np.column_stack([
            C * _comb(n, i) * x ** i * (1 - x) ** (n - i)
            for i in range(n + 1)
        ])

        # LEM term is part of S(x), so it is multiplied by C(x).
        L = (C * x ** 0.5 * (1 - x) ** (n - 0.5)).reshape(-1, 1)

        # full augmented basis  (n_points, order+2)
        self._basis = np.hstack([B, L])
        
        ref_up, ref_lo = _fit_cst_reference(order)
        self.reference = np.concatenate([
            ref_up,          # Bernstein upper + Alem_up
            ref_lo,          # Bernstein lower + Alem_lo
            [0.0],           # dz_te: keep sharp trailing edge
        ])

        configured_reference = os.environ.get("CST_REFERENCE_PARAMS")
        if configured_reference:
            try:
                values = np.asarray(json.loads(configured_reference), dtype=float)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(
                    "CST_REFERENCE_PARAMS must be a JSON array of numbers"
                ) from exc
            if values.shape != self.reference.shape or not np.all(np.isfinite(values)):
                raise ValueError(
                    f"CST_REFERENCE_PARAMS must contain exactly "
                    f"{len(self.reference)} finite numbers"
                )
            self.reference = values.copy()

    @property
    def param_names(self) -> list[str]:
        n = self.order
        return (
            [f"A{i}_up" for i in range(n + 1)] + ["Alem_up"] +
            [f"A{i}_lo" for i in range(n + 1)] + ["Alem_lo"] +
            ["dz_te"]
        )

    @property
    def bounds(self) -> np.ndarray:
        n_coeffs = self.order + 1      # 6 Bernstein per surface
        stride = n_coeffs + 1
        margin = float(os.environ.get(
            "CST_MARGIN",
            os.environ.get("CST_RAE_MARGIN", 0.12),
        ))
        a0_margin = float(os.environ.get(
            "CST_A0_MARGIN",
            os.environ.get("CST_A0_RAE_MARGIN", 0.04),
        ))

        ref_up = self.reference[:stride]
        ref_lo = self.reference[stride:2 * stride]

        upper = np.column_stack([ref_up - margin, ref_up + margin])
        lower = np.column_stack([ref_lo - margin, ref_lo + margin])
        upper[0] = [ref_up[0] - a0_margin, ref_up[0] + a0_margin]
        lower[0] = [ref_lo[0] - a0_margin, ref_lo[0] + a0_margin]

        # trailing edge: sharp only for now
        te = np.array([[0.0, 0.0]])

        return np.vstack([upper, lower, te])

    def decode(self, params: np.ndarray) -> np.ndarray:
        n_coeffs = self.order + 1       # 6 Bernstein per surface
        stride   = n_coeffs + 1         # 7 genes per surface (Bernstein + LEM)

        a_up  = np.asarray(params[:stride],           dtype=float)   # 7 genes
        a_lo  = np.asarray(params[stride:2*stride],   dtype=float)   # 7 genes
        dz_te = float(params[2 * stride])

        x = self._x
        Z_up  = self._basis @ a_up  + x * dz_te
        Z_low = self._basis @ a_lo  - x * dz_te

        # enforce exact closure at leading edge
        Z_up[0]  = 0.0
        Z_low[0] = 0.0

        X_upper = x[::-1].reshape(-1, 1)    # TE -> LE
        Z_upper = Z_up[::-1].reshape(-1, 1)
        X_lower = x[1:].reshape(-1, 1)      # LE -> TE (skip duplicate LE)
        Z_lower = Z_low[1:].reshape(-1, 1)

        return np.hstack((
            np.vstack((X_upper, X_lower)),
            np.vstack((Z_upper, Z_lower)),
        ))
