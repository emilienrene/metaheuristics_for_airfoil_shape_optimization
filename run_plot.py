"""
Plot the best airfoil of the current generation.

Usage:
    python3 run_plot.py <airfoil_path> <generation>
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.geometry import plot_best_airfoil

airfoil_path = sys.argv[1]
generation   = int(sys.argv[2])

out = plot_best_airfoil(airfoil_path, generation)
print(f"Plot saved: {out}")
