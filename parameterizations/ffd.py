"""
Free-Form Deformation (FFD) airfoil parameterization.

Follows the Bézier surface FFD formulation of Sederberg & Parry (1986) as
described in equations 22-23 of the reference. A rectangular (M+1) x (N+1)
control point lattice is embedded around a configurable baseline. Each surface
point is mapped to lattice coordinates (s, t) and the deformed position is:

    z_def(x) = z_base(x) + sum_{j=0}^{N} sum_{i=0}^{M} B_{i,M}(s) * B_{j,N}(t) * dP_z[j,i]

where s = x (chordwise) and t = (z_base - z_min) / (z_max - z_min) (normal).
By default, z_min and z_max are a tight box around the baseline airfoil. Set
FFD_BOX_MODE=fixed to recover the old symmetric box z in [-z_margin, z_margin].
Control points are only allowed to move in z. By default, only the four
lattice corners are pinned (zero displacement), matching the paper-style FFD
setup. Set FFD_PIN_MODE=end_columns to recover the older stricter behavior
where every leading/trailing lattice-column point is pinned.

Both 3-row (N=2) and 4-row (N=3) configurations are supported via the
n_rows parameter.

Default gene layout  ((N+1) * (M+1) - 4 total):
    Row 0 interior cols:   dz[0,1] ... dz[0,M-1]
    Middle rows all cols:  dz[j,0] ... dz[j,M]
    ...
    Row N interior cols:   dz[N,1] ... dz[N,M-1]

Default: n_rows=3, m=8  ->  3 * 9 - 4 = 23 genes.

References:
    Sederberg, T. & Parry, S. (1986). Free-Form Deformation of Solid
    Geometric Models. SIGGRAPH Computer Graphics, 20(4), 151-160.
    Samareh, J. (2001). Survey of Shape Parameterization Techniques for
    High-Fidelity Multidisciplinary Shape Optimization. AIAA Journal, 39(5).
"""

from __future__ import annotations

import numpy as np
from math import comb as _comb

from .base import Parameterization
from .airfoil_references import deformation_baseline_name, deformation_baseline_surfaces


def _bernstein_matrix(m: int, t: np.ndarray) -> np.ndarray:
    """(len(t), m+1) matrix of degree-m Bernstein basis values."""
    return np.column_stack([
        _comb(m, i) * t**i * (1 - t)**(m - i)
        for i in range(m + 1)
    ])


def _box_limits(
    z_base_up: np.ndarray,
    z_base_lo: np.ndarray,
    z_margin: float,
) -> tuple[float, float]:
    """
    Return the normal-direction FFD box limits.

    Default mode is a tight box around the baseline plus FFD_Z_PADDING. This
    keeps upper and lower surfaces separated in the normal Bernstein
    coordinate. FFD_BOX_MODE=fixed restores the old symmetric half-height box.
    """
    import os

    z_min_override = os.environ.get("FFD_Z_MIN")
    z_max_override = os.environ.get("FFD_Z_MAX")
    if z_min_override is not None or z_max_override is not None:
        if z_min_override is None or z_max_override is None:
            raise ValueError("Set both FFD_Z_MIN and FFD_Z_MAX, or neither.")
        z_min = float(z_min_override)
        z_max = float(z_max_override)
    else:
        mode = os.environ.get("FFD_BOX_MODE", "tight").strip().lower()
        if mode in {"fixed", "legacy", "symmetric"}:
            z_min, z_max = -float(z_margin), float(z_margin)
        elif mode in {"tight", "baseline"}:
            padding = float(os.environ.get("FFD_Z_PADDING", 0.01))
            z_all = np.concatenate([z_base_up, z_base_lo])
            z_min = float(np.min(z_all) - padding)
            z_max = float(np.max(z_all) + padding)
        else:
            raise ValueError(
                f"Unknown FFD_BOX_MODE={mode!r}. Use 'tight' or 'fixed'."
            )

    if not z_min < z_max:
        raise ValueError(f"Invalid FFD box: z_min={z_min}, z_max={z_max}")
    return z_min, z_max


def _pin_mode() -> str:
    """Return the FFD pinning convention."""
    import os

    legacy = os.environ.get("FFD_PIN_END_COLUMNS")
    if legacy is not None:
        return "end_columns" if int(legacy) else "corners"

    mode = os.environ.get("FFD_PIN_MODE", "corners").strip().lower()
    aliases = {
        "corner": "corners",
        "corners": "corners",
        "four_corners": "corners",
        "paper": "corners",
        "end_columns": "end_columns",
        "columns": "end_columns",
        "legacy": "end_columns",
    }
    if mode not in aliases:
        raise ValueError(
            f"Unknown FFD_PIN_MODE={mode!r}. Use 'corners' or 'end_columns'."
        )
    return aliases[mode]


def _free_control_indices(n_rows: int, m: int, pin_mode: str) -> list[tuple[int, int]]:
    """Control-lattice indices that are represented by genotype genes."""
    if pin_mode == "corners":
        pinned = {
            (0, 0),
            (0, m),
            (n_rows - 1, 0),
            (n_rows - 1, m),
        }
    elif pin_mode == "end_columns":
        pinned = {(row, col) for row in range(n_rows) for col in (0, m)}
    else:
        raise ValueError(
            f"Unknown FFD pin mode {pin_mode!r}. Use 'corners' or 'end_columns'."
        )

    return [
        (row, col)
        for row in range(n_rows)
        for col in range(m + 1)
        if (row, col) not in pinned
    ]


def ffd_design_matrices(
    x: np.ndarray,
    z_base_up: np.ndarray,
    z_base_lo: np.ndarray,
    n_rows: int,
    m: int,
    z_min: float,
    z_max: float,
    pin_mode: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return upper/lower linear design matrices matching FFD.decode()."""
    if pin_mode is None:
        pin_mode = _pin_mode()

    normal_degree = n_rows - 1
    t_up = (z_base_up - z_min) / (z_max - z_min)
    t_lo = (z_base_lo - z_min) / (z_max - z_min)

    b_s = _bernstein_matrix(m, x)
    b_t_up = _bernstein_matrix(normal_degree, t_up)
    b_t_lo = _bernstein_matrix(normal_degree, t_lo)

    free_indices = _free_control_indices(n_rows, m, pin_mode)
    n_params = len(free_indices)
    a_up = np.zeros((len(x), n_params))
    a_lo = np.zeros((len(x), n_params))

    for gene, (row, chord_col) in enumerate(free_indices):
        a_up[:, gene] = b_t_up[:, row] * b_s[:, chord_col]
        a_lo[:, gene] = b_t_lo[:, row] * b_s[:, chord_col]

    return a_up, a_lo


class FFD(Parameterization):
    """
    Bezier surface FFD parameterization on a configurable baseline.

    The normal optimization default is NACA0012. Set
    DEFORMATION_BASELINE=rae2822 for the RAE2822-to-NACA0012 inverse study.

    Parameters
    ----------
    n_rows : int
        Number of lattice rows (default 3). Must be >= 2.
        3 rows is compact; 4 rows gives more independent upper/lower control.
        4 rows is recommended when upper and lower surfaces are unconstrained.
    m : int
        Chordwise Bernstein degree (m+1 control columns). Default 8.
    z_margin : float
        Legacy half-height used only when FFD_BOX_MODE=fixed. Default 0.15.
    n_points : int
        Number of chord-wise evaluation points. Default 1000.
    """

    name = "ffd"
    mutation_mode = "additive"

    def __init__(
        self,
        n_rows: int = 3,
        m: int = 8,
        z_margin: float = 0.15,
        n_points: int = 1000,
    ):
        import os
        n_rows = int(os.environ.get("FFD_N_ROWS", n_rows))
        m      = int(os.environ.get("FFD_M",      m))
        if n_rows < 2:
            raise ValueError(f"n_rows={n_rows}; need >= 2")
        if m < 2:
            raise ValueError(f"m={m}; need >= 2 for at least one free interior column")

        self.n_rows   = n_rows
        self.m        = m
        self.z_margin = z_margin
        self.n_points = n_points
        self.pin_mode = _pin_mode()
        self._free_indices = _free_control_indices(n_rows, m, self.pin_mode)

        N = n_rows - 1   # normal Bernstein degree

        x = np.linspace(0, 1, n_points)
        self._x = x
        self.baseline_name = deformation_baseline_name()
        self._z_base_up, self._z_base_lo = deformation_baseline_surfaces(
            x,
            self.baseline_name,
        )

        z_min, z_max = _box_limits(self._z_base_up, self._z_base_lo, z_margin)
        self.z_min = z_min
        self.z_max = z_max

        # map each surface to its normal lattice coordinate t in [0,1]
        t_up = (self._z_base_up - z_min) / (z_max - z_min)
        t_lo = (self._z_base_lo - z_min) / (z_max - z_min)

        # precompute basis matrices
        Bs   = _bernstein_matrix(m, x)           # (n_pts, m+1)  chordwise
        Bt_up = _bernstein_matrix(N, t_up)        # (n_pts, N+1)  normal, upper
        Bt_lo = _bernstein_matrix(N, t_lo)        # (n_pts, N+1)  normal, lower

        # precompute row-weighted chordwise bases for fast decode
        # dz = sum_j Bt[:,j] * (Bs @ dP_z[j])
        # store as list of (n_pts, m+1) matrices, one per row
        self._K_up = [Bt_up[:, j:j+1] * Bs for j in range(N + 1)]
        self._K_lo = [Bt_lo[:, j:j+1] * Bs for j in range(N + 1)]

        self.reference = np.zeros(len(self._free_indices))

    @property
    def param_names(self) -> list[str]:
        return [
            f"dz_r{j}_c{i}"
            for j, i in self._free_indices
        ]

    @property
    def bounds(self) -> np.ndarray:
        import os
        n_genes = len(self._free_indices)
        amp = float(os.environ.get(
            "FFD_AMPLITUDE",
            os.environ.get("FFD_RAE_AMPLITUDE", 0.08),
        ))
        return np.column_stack([
            np.full(n_genes, -amp),
            np.full(n_genes,  amp),
        ])

    def _unpack(self, params: np.ndarray) -> np.ndarray:
        """Unpack flat genes into the full control-lattice z displacement array."""
        params = np.asarray(params, dtype=float)
        if len(params) != len(self._free_indices):
            raise ValueError(
                f"FFD expected {len(self._free_indices)} params, got {len(params)}"
            )

        dz = np.zeros((self.n_rows, self.m + 1))
        for value, (row, col) in zip(params, self._free_indices):
            dz[row, col] = value
        return dz

    def decode(self, params: np.ndarray) -> np.ndarray:
        dz = self._unpack(params)

        dz_up = sum(self._K_up[j] @ dz[j] for j in range(self.n_rows))
        dz_lo = sum(self._K_lo[j] @ dz[j] for j in range(self.n_rows))

        Z_up  = self._z_base_up + dz_up
        Z_low = self._z_base_lo + dz_lo

        x = self._x
        X_upper = x[::-1].reshape(-1, 1)
        Z_upper = Z_up[::-1].reshape(-1, 1)
        X_lower = x[1:].reshape(-1, 1)
        Z_lower = Z_low[1:].reshape(-1, 1)

        return np.hstack((
            np.vstack((X_upper, X_lower)),
            np.vstack((Z_upper, Z_lower)),
        ))
