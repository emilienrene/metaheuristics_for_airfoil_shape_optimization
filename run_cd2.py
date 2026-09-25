"""
Append Cd from an additional Mach-point evaluation into the primary combined CSV.

Usage:
    python3 run_cd2.py <primary_eval_dir> <extra_eval_dir> [mach] [cl_target]
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.evaluation import append_second_cd


primary_eval_dir = sys.argv[1]
second_eval_dir = sys.argv[2]
mach2 = sys.argv[3] if len(sys.argv) > 3 else os.environ.get("MACH_CD2")

if len(sys.argv) > 4:
    cl_target = float(sys.argv[4])
elif os.environ.get("CL_TARGET") is not None:
    cl_target = float(os.environ["CL_TARGET"])
else:
    cl_target = None

append_second_cd(
    primary_eval_dir=primary_eval_dir,
    second_eval_dir=second_eval_dir,
    mach=mach2,
    cl_target=cl_target,
)
