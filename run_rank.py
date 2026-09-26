"""
Rank the current population by fitness.

Usage:
    python3 run_rank.py <current_gen>

Environment variables read:
    AOA_MIN, AOA_MAX, N_CD_POINTS
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.fitness import rank

current_gen  = int(sys.argv[1])
coeffs       = [float(value) for value in sys.argv[2:]] or None
aoa_min      = float(os.environ["AOA_MIN"])
aoa_max      = float(os.environ["AOA_MAX"])

rank(
    current_gen  = current_gen,
    coeff_cd     = coeffs,
    aoa_min      = aoa_min,
    aoa_max      = aoa_max,
)
