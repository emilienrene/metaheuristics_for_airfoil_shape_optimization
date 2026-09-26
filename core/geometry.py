"""
Geometry utilities — airfoil visualisation.
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


_DAE11_THICKNESS_GUIDES = (
    (0.33, "DRELA_DAE11_T33_TARGET", 0.128, "#e67e22"),
    (0.90, "DRELA_DAE11_T90_TARGET", 0.014, "#2ca02c"),
)


def plot_best_airfoil(airfoil_path: str, generation: int) -> str:
    """
    Plot the airfoil at airfoil_path and save a PNG next to it.
    generation is the display generation number (1-indexed, printed as gen-1).
    Returns the path to the saved PNG.
    """
    coords = np.loadtxt(airfoil_path, skiprows=1)
    x, z   = coords[:, 0], coords[:, 1]

    fig, ax = plt.subplots(figsize=(8, 4))
    show_dae11_guides = _plot_dae11_thickness_guides()
    if show_dae11_guides:
        _plot_dae11_reference(ax)
    ax.scatter(x, z, s=10, color="blue", label="Best airfoil", zorder=4)

    if show_dae11_guides:
        x_up, z_up, x_lo, z_lo = _split_airfoil_surfaces(coords)
        for x_station, target_variable, default_target, color in _DAE11_THICKNESS_GUIDES:
            target = float(os.environ.get(target_variable, default_target))
            upper = float(np.interp(x_station, x_up, z_up))
            lower = float(np.interp(x_station, x_lo, z_lo))
            camber = 0.5 * (upper + lower)
            actual = upper - lower

            ax.axvline(
                x_station,
                color=color,
                linestyle=":",
                linewidth=1.8,
                alpha=0.9,
                zorder=1,
            )
            ax.plot(
                [x_station, x_station],
                [camber - 0.5 * target, camber + 0.5 * target],
                color=color,
                linewidth=8.0,
                alpha=0.55,
                solid_capstyle="butt",
                label=(
                    f"x/c={x_station:.2f}: target t/c={target:.3f}, "
                    f"current={actual:.3f}"
                ),
                zorder=2,
            )

    ax.set_aspect("equal")
    ax.set_xlabel("x/c")
    ax.set_ylabel("z/c")
    ax.set_ylim(-0.05, 0.30)
    ax.set_title(
        f"Best Airfoil — Generation {generation - 1}  |  "
        f"{os.path.basename(airfoil_path)}"
    )
    ax.grid(True, linestyle="--", alpha=0.5)
    if show_dae11_guides:
        ax.legend(loc="upper right", frameon=True, fontsize=8)

    out_path = os.path.join(
        os.path.dirname(airfoil_path),
        f"best_airfoil_gen{generation - 1}.png",
    )
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def _plot_dae11_thickness_guides() -> bool:
    configured = os.environ.get("PLOT_DAE11_THICKNESS_GUIDES")
    if configured is not None:
        return configured.strip().lower() in {"1", "true", "yes", "y"}
    reference = os.environ.get("PARAM_REFERENCE_AIRFOIL", "")
    return reference.strip().lower().replace("-", "").replace("_", "") == "dae11"


def _plot_dae11_reference(ax: plt.Axes) -> None:
    """Draw a deliberately subdued DAE-11 baseline beneath the best airfoil."""
    from parameterizations.airfoil_references import dae11_table

    x_ref, z_upper, z_lower = dae11_table(n_stations=501)
    ax.plot(
        x_ref,
        z_upper,
        color="#666666",
        linestyle="--",
        linewidth=1.0,
        alpha=0.55,
        label="DAE-11 baseline",
        zorder=3,
    )
    ax.plot(
        x_ref,
        z_lower,
        color="#666666",
        linestyle="--",
        linewidth=1.0,
        alpha=0.55,
        zorder=3,
    )


def _split_airfoil_surfaces(
    coords: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return upper and lower surfaces with monotonically increasing x."""
    leading_edge = int(np.argmin(coords[:, 0]))
    if leading_edge <= 0 or leading_edge >= len(coords) - 1:
        raise ValueError("Could not split airfoil contour at the leading edge.")

    first = coords[: leading_edge + 1]
    second = coords[leading_edge:]
    first = first[np.argsort(first[:, 0])]
    second = second[np.argsort(second[:, 0])]

    probe = np.linspace(0.05, 0.95, 101)
    first_mean = float(np.mean(np.interp(probe, first[:, 0], first[:, 1])))
    second_mean = float(np.mean(np.interp(probe, second[:, 0], second[:, 1])))
    if first_mean >= second_mean:
        upper, lower = first, second
    else:
        upper, lower = second, first
    return upper[:, 0], upper[:, 1], lower[:, 0], lower[:, 1]
