"""
RAE2822 reference geometry utilities.

The source table lists a lower-surface ordinate. For the signed geometry used
by the optimizer, the lower surface is therefore ``z_lower = -lower_ordinate``.
"""

from __future__ import annotations

import os
import numpy as np


RAE2822_TABLE = np.array([
    [0.00000,  0.00000, 0.00000],
    [0.00060,  0.00317, 0.00323],
    [0.00241,  0.00658, 0.00642],
    [0.00541,  0.00957, 0.00945],
    [0.00961,  0.01273, 0.01269],
    [0.01498,  0.01580, 0.01579],
    [0.02153,  0.01880, 0.01875],
    [0.02923,  0.02180, 0.02163],
    [0.03806,  0.02472, 0.02445],
    [0.04801,  0.02761, 0.02726],
    [0.05904,  0.03042, 0.03004],
    [0.07114,  0.03315, 0.03280],
    [0.08427,  0.03584, 0.03552],
    [0.09840,  0.03844, 0.03817],
    [0.11349,  0.04094, 0.04073],
    [0.12952,  0.04333, 0.04321],
    [0.14645,  0.04561, 0.04558],
    [0.16422,  0.04775, 0.04778],
    [0.18280,  0.04977, 0.04987],
    [0.20215,  0.05167, 0.05187],
    [0.22221,  0.05340, 0.05377],
    [0.24295,  0.05498, 0.05556],
    [0.26430,  0.05638, 0.05713],
    [0.28622,  0.05753, 0.05848],
    [0.30866,  0.05843, 0.05967],
    [0.33156,  0.05900, 0.06070],
    [0.35486,  0.05919, 0.06155],
    [0.37851,  0.05893, 0.06220],
    [0.40245,  0.05817, 0.06263],
    [0.42663,  0.05689, 0.06285],
    [0.45099,  0.05515, 0.06286],
    [0.47547,  0.05297, 0.06261],
    [0.50000,  0.05044, 0.06212],
    [0.52453,  0.04761, 0.06135],
    [0.54901,  0.04452, 0.06030],
    [0.57336,  0.04127, 0.05895],
    [0.59754,  0.03791, 0.05733],
    [0.62149,  0.03463, 0.05547],
    [0.64514,  0.03110, 0.05339],
    [0.66845,  0.02770, 0.05112],
    [0.69134,  0.02438, 0.04857],
    [0.71378,  0.02118, 0.04612],
    [0.73570,  0.01812, 0.04338],
    [0.75705,  0.01524, 0.04075],
    [0.77778,  0.01256, 0.03795],
    [0.79785,  0.01013, 0.03514],
    [0.81720,  0.00792, 0.03231],
    [0.83578,  0.00594, 0.02948],
    [0.85355,  0.00422, 0.02670],
    [0.87048,  0.00273, 0.02397],
    [0.88651,  0.00149, 0.02131],
    [0.90160,  0.00049, 0.01874],
    [0.91574, -0.00027, 0.01627],
    [0.92886, -0.00081, 0.01393],
    [0.94096, -0.00113, 0.01170],
    [0.95200, -0.00125, 0.00964],
    [0.96194, -0.00125, 0.00775],
    [0.97077, -0.00113, 0.00606],
    [0.97847, -0.00094, 0.00455],
    [0.98502, -0.00071, 0.00326],
    [0.99039, -0.00048, 0.00218],
    [0.99459, -0.00026, 0.00132],
    [0.99759, -0.00009, 0.00069],
    [0.99940,  0.00001, 0.00030],
    [1.00000,  0.00000, 0.00000],
], dtype=float)


def rae2822_table() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = RAE2822_TABLE[:, 0].copy()
    z_upper = RAE2822_TABLE[:, 2].copy()
    z_lower = -RAE2822_TABLE[:, 1].copy()
    return x, z_upper, z_lower


def rae2822_surfaces(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_ref, z_upper_ref, z_lower_ref = rae2822_table()
    x = np.asarray(x, dtype=float)
    z_upper = np.interp(x, x_ref, z_upper_ref)
    z_lower = np.interp(x, x_ref, z_lower_ref)
    return z_upper, z_lower


def rae2822_half_te_thickness() -> float:
    _, z_upper, z_lower = rae2822_table()
    return 0.5 * float(z_upper[-1] - z_lower[-1])


def modified_rae2822_surfaces(
    x: np.ndarray,
    half_te_thickness: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Modified RAE2822 baseline for deformative methods.

    The baseline starts from the measured RAE2822 camber/thickness
    decomposition, scales both components, and then optionally applies the
    same linear half-trailing-edge-thickness correction used in modified
    baseline deformation studies:

        z_upper = s_c * camber_rae + s_t * thickness_rae + x * delta_te
        z_lower = s_c * camber_rae - s_t * thickness_rae - x * delta_te

    Defaults intentionally do not reproduce RAE2822 exactly; this avoids a
    trivial zero-parameter pass for deformative methods while keeping the
    starting shape RAE-like and topologically compatible.

    Environment overrides:
        RAE_BASELINE_CAMBER_SCALE     default 0.70
        RAE_BASELINE_THICKNESS_SCALE  default 0.90
    """
    x = np.asarray(x, dtype=float)
    z_upper, z_lower = rae2822_surfaces(x)

    camber = 0.5 * (z_upper + z_lower)
    thickness = 0.5 * (z_upper - z_lower)

    camber_scale = float(os.environ.get("RAE_BASELINE_CAMBER_SCALE", 0.70))
    thickness_scale = float(os.environ.get("RAE_BASELINE_THICKNESS_SCALE", 0.90))

    current_half_te = rae2822_half_te_thickness()
    if half_te_thickness is None:
        half_te_thickness = current_half_te

    delta_te = float(half_te_thickness) - current_half_te
    z_upper = camber_scale * camber + thickness_scale * thickness + x * delta_te
    z_lower = camber_scale * camber - thickness_scale * thickness - x * delta_te
    return z_upper, z_lower


def modified_naca0012_surfaces(
    x: np.ndarray,
    half_te_thickness: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Modified NACA0012 baseline for deformative methods.

    This follows the common deformative-parameterization setup:

        y_t = 0.6 * (0.2969*sqrt(x) - 0.1260*x - 0.3516*x^2
                     + 0.2843*x^3 - 0.1036*x^4)

    then adds the target half trailing-edge thickness linearly:

        z_upper =  y_t + x * z_te
        z_lower = -y_t - x * z_te
    """
    x = np.asarray(x, dtype=float)

    thickness = 0.6 * (
        0.2969 * np.sqrt(x)
        - 0.1260 * x
        - 0.3516 * x ** 2
        + 0.2843 * x ** 3
        - 0.1036 * x ** 4
    )

    z_upper = thickness + x * half_te_thickness
    z_lower = -thickness - x * half_te_thickness
    return z_upper, z_lower
