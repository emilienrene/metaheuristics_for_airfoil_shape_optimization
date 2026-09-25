"""
Compute Cl_max from angle-sweep CSVs and append to the combined CSV.

Usage:
    python3 run_clmax.py
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.evaluation import append_clmax

append_clmax()
