"""Reference and deformation-baseline airfoil geometry utilities."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np

from .rae2822 import modified_rae2822_surfaces, rae2822_surfaces, rae2822_table


NACA0012_THICKNESS_RATIO = 0.12
NACA0012_LE_RADIUS = (5.0 * NACA0012_THICKNESS_RATIO * 0.2969) ** 2 / 2.0
DEFAULT_NACA0012_DATA_FILE = Path("/home/erene/simulations/rAIFoil/NACA0012.dat")
DEFAULT_DAE11_DATA_FILE = Path("/home/erene/simulations/rAIFoil/dae11.dat")
BUNDLED_DAE11_DATA_FILE = Path(__file__).with_name("data") / "dae11.dat"
RAE2822_LE_RADIUS = 0.008460265


def _normalise_name(name: str) -> str:
    key = name.strip().lower().replace("-", "").replace("_", "")
    aliases = {
        "naca0012": "naca0012",
        "0012": "naca0012",
        "rae2822": "rae2822",
        "2822": "rae2822",
        "modifiedrae2822": "modified_rae2822",
        "dae11": "dae11",
    }
    if key not in aliases:
        raise ValueError(
            f"Unknown airfoil reference {name!r}; use 'naca0012', "
            "'rae2822', 'modified_rae2822', or 'dae11'."
        )
    return aliases[key]


def parameterization_reference_name() -> str:
    """Reference used to centre constructive-method bounds."""
    return _normalise_name(os.environ.get("PARAM_REFERENCE_AIRFOIL", "naca0012"))


def deformation_baseline_name() -> str:
    """Zero-deformation geometry used by Hicks-Henne and FFD."""
    return _normalise_name(os.environ.get("DEFORMATION_BASELINE", "naca0012"))


def _analytic_naca0012_surfaces(
    x: np.ndarray,
    half_te_thickness: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return symmetric, closed-TE NACA0012 surfaces plus an optional TE term."""
    x = np.asarray(x, dtype=float)
    t = NACA0012_THICKNESS_RATIO
    half_thickness = 5.0 * t * (
        0.2969 * np.sqrt(x)
        - 0.1260 * x
        - 0.3516 * x ** 2
        + 0.2843 * x ** 3
        - 0.1036 * x ** 4
    )
    te_term = x * float(half_te_thickness)
    return half_thickness + te_term, -half_thickness - te_term


def naca0012_data_file() -> Path:
    """Return the configured NACA0012 coordinate-file path."""
    return Path(os.environ.get("NACA0012_DATA_FILE", DEFAULT_NACA0012_DATA_FILE))


def _numeric_airfoil_rows(path: Path) -> np.ndarray:
    rows: list[tuple[float, float]] = []
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            fields = line.replace(",", " ").split()
            if len(fields) < 2:
                continue
            try:
                rows.append((float(fields[0]), float(fields[1])))
            except ValueError:
                continue
    if len(rows) < 5:
        raise ValueError(f"Could not read an airfoil contour from {path}")
    coords = np.asarray(rows, dtype=float)
    # Some Lednicer files place the upper/lower point counts on a numeric line.
    if np.all(coords[0] > 1.5):
        coords = coords[1:]
    if len(coords) < 5:
        raise ValueError(f"Could not read an airfoil contour from {path}")
    return coords


def _unique_sorted_surface(surface: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(surface[:, 0])
    x = surface[order, 0]
    z = surface[order, 1]
    unique_x, inverse = np.unique(x, return_inverse=True)
    z_sum = np.zeros_like(unique_x)
    counts = np.zeros_like(unique_x)
    np.add.at(z_sum, inverse, z)
    np.add.at(counts, inverse, 1.0)
    return unique_x, z_sum / counts


@lru_cache(maxsize=None)
def _load_airfoil_file(path_string: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load common Selig or two-block airfoil coordinate ordering."""
    coords = _numeric_airfoil_rows(Path(path_string))
    x_raw = coords[:, 0]

    x_min = float(np.min(x_raw))
    x_max = float(np.max(x_raw))
    chord = x_max - x_min
    if chord <= 0.0:
        raise ValueError(f"Degenerate chord coordinates in {path_string}")
    coords[:, 0] = (x_raw - x_min) / chord
    coords[:, 1] /= chord

    x = coords[:, 0]
    i_min = int(np.argmin(x))
    i_max = int(np.argmax(x))
    if 0 < i_min < len(coords) - 1:
        first, second = coords[: i_min + 1], coords[i_min:]
    elif 0 < i_max < len(coords) - 1:
        first, second = coords[: i_max + 1], coords[i_max:]
    else:
        reset = int(np.argmin(np.diff(x)))
        if np.diff(x)[reset] >= -0.25:
            raise ValueError(f"Could not identify upper/lower surfaces in {path_string}")
        first, second = coords[: reset + 1], coords[reset + 1 :]

    x_first, z_first = _unique_sorted_surface(first)
    x_second, z_second = _unique_sorted_surface(second)
    probe = np.linspace(0.05, 0.95, 101)
    first_mean = float(np.mean(np.interp(probe, x_first, z_first)))
    second_mean = float(np.mean(np.interp(probe, x_second, z_second)))
    if first_mean >= second_mean:
        return x_first, z_first, np.interp(x_first, x_second, z_second)
    return x_second, z_second, np.interp(x_second, x_first, z_first)


def naca0012_surfaces(
    x: np.ndarray,
    half_te_thickness: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate the NACA0012 file, with an analytic local fallback."""
    x = np.asarray(x, dtype=float)
    path = naca0012_data_file()
    if path.is_file():
        x_ref, z_upper_ref, z_lower_ref = _load_airfoil_file(str(path.resolve()))
        z_upper = np.interp(x, x_ref, z_upper_ref)
        z_lower = np.interp(x, x_ref, z_lower_ref)
        if half_te_thickness is not None:
            current_half_te = 0.5 * float(z_upper_ref[-1] - z_lower_ref[-1])
            delta_te = float(half_te_thickness) - current_half_te
            z_upper = z_upper + x * delta_te
            z_lower = z_lower - x * delta_te
        return z_upper, z_lower
    return _analytic_naca0012_surfaces(
        x,
        half_te_thickness=0.0 if half_te_thickness is None else half_te_thickness,
    )


def naca0012_table(n_stations: int = 65) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n_stations < 2:
        raise ValueError("n_stations must be at least 2")
    theta = np.linspace(0.0, np.pi, n_stations)
    x = 0.5 * (1.0 - np.cos(theta))
    z_upper, z_lower = naca0012_surfaces(x)
    return x, z_upper, z_lower


def dae11_data_file() -> Path:
    """Return the configured DAE-11 coordinate-file path."""
    configured = os.environ.get("DAE11_DATA_FILE")
    if configured:
        return Path(configured)
    if DEFAULT_DAE11_DATA_FILE.is_file():
        return DEFAULT_DAE11_DATA_FILE
    return BUNDLED_DAE11_DATA_FILE


def dae11_surfaces(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate the configured DAE-11 coordinate file."""
    x = np.asarray(x, dtype=float)
    path = dae11_data_file()
    if not path.is_file():
        raise FileNotFoundError(
            f"DAE-11 coordinate file not found: {path}. Set DAE11_DATA_FILE."
        )
    x_ref, z_upper_ref, z_lower_ref = _load_airfoil_file(str(path.resolve()))
    return np.interp(x, x_ref, z_upper_ref), np.interp(x, x_ref, z_lower_ref)


def dae11_table(n_stations: int = 65) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n_stations < 2:
        raise ValueError("n_stations must be at least 2")
    theta = np.linspace(0.0, np.pi, n_stations)
    x = 0.5 * (1.0 - np.cos(theta))
    z_upper, z_lower = dae11_surfaces(x)
    return x, z_upper, z_lower


def dae11_le_radius() -> float:
    """Estimate DAE-11 nose radius from the leading sqrt(x) coefficient."""
    x, z_upper, z_lower = dae11_table()
    mask = (x > 0.0) & (x <= 0.05)
    basis = np.column_stack((np.sqrt(x[mask]), x[mask], x[mask] ** 1.5))
    a_upper = float(np.linalg.lstsq(basis, z_upper[mask], rcond=None)[0][0])
    a_lower = float(np.linalg.lstsq(basis, z_lower[mask], rcond=None)[0][0])
    sqrt_coefficient = 0.5 * (a_upper - a_lower)
    return 0.5 * sqrt_coefficient ** 2


def reference_table(name: str | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    key = _normalise_name(name) if name is not None else parameterization_reference_name()
    if key == "naca0012":
        return naca0012_table()
    if key == "dae11":
        return dae11_table()
    return rae2822_table()


def reference_le_radius(name: str | None = None) -> float:
    key = _normalise_name(name) if name is not None else parameterization_reference_name()
    if key == "naca0012":
        return NACA0012_LE_RADIUS
    if key == "dae11":
        return dae11_le_radius()
    return RAE2822_LE_RADIUS


def deformation_baseline_surfaces(
    x: np.ndarray,
    name: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    key = _normalise_name(name) if name is not None else deformation_baseline_name()
    if key == "naca0012":
        z_upper, z_lower = naca0012_surfaces(x)
    elif key == "dae11":
        z_upper, z_lower = dae11_surfaces(x)
    elif key == "rae2822":
        z_upper, z_lower = rae2822_surfaces(x)
    else:
        z_upper, z_lower = modified_rae2822_surfaces(x)

    target_te_upper = os.environ.get("DEFORMATION_TARGET_TE_UPPER")
    target_te_lower = os.environ.get("DEFORMATION_TARGET_TE_LOWER")
    if target_te_upper is None and target_te_lower is None:
        return z_upper, z_lower
    if target_te_upper is None or target_te_lower is None:
        raise ValueError(
            "Set both DEFORMATION_TARGET_TE_UPPER and "
            "DEFORMATION_TARGET_TE_LOWER, or neither."
        )

    # Deformative methods cannot change endpoint topology when their basis is
    # pinned there. Supply the known target trailing edge through the baseline,
    # as in the modified-NACA treatment used by Masters et al.
    i_te = int(np.argmax(x))
    z_upper = z_upper + x * (float(target_te_upper) - z_upper[i_te])
    z_lower = z_lower + x * (float(target_te_lower) - z_lower[i_te])
    return z_upper, z_lower
