"""
Combine per-airfoil solver output CSVs and insert interpolated CL row.

Usage:
    python3 run_concat.py <eval_dir> <cl_target>
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.evaluation import combine_and_interpolate

eval_dir  = sys.argv[1]
cl_target = float(sys.argv[2])

combine_and_interpolate(eval_dir, cl_target)
