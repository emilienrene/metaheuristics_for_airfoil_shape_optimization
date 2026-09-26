"""
Append Cd columns from additional Mach-point evaluations into the primary CSV.

Usage:
    python3 run_cd_points.py <primary_eval_dir> <cl_target>
    python3 run_cd_points.py <primary_eval_dir> <cl_target> <mach2> <eval_dir2>
    python3 run_cd_points.py <primary_eval_dir> <cl_target> <mach2> <eval_dir2> <mach3> <eval_dir3> ...

The primary directory keeps its normal DD_80_combined.csv with the first
Mach-point CD column named "CD". Each extra Mach point is appended as:

    CD_mach_<mach>

Example:
    python3 run_cd_points.py evaluation_mach_0.65 0.65 0.75 evaluation_mach_0.75 0.85 evaluation_mach_0.85
"""

from __future__ import annotations

import os
import sys

import pandas as pd


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(
            "Usage: python3 run_cd_points.py <primary_eval_dir> <cl_target> "
            "[<mach> <eval_dir> ...]"
        )

    primary_eval_dir = sys.argv[1]
    cl_target = float(sys.argv[2])
    extra_args = sys.argv[3:]

    if len(extra_args) % 2 != 0:
        raise SystemExit("Extra Mach-point args must be pairs: <mach> <eval_dir>.")

    primary_csv = os.path.join(primary_eval_dir, "DD_80_combined.csv")
    if not os.path.isfile(primary_csv):
        raise RuntimeError(f"Primary combined CSV not found: {primary_csv}")

    df_primary = pd.read_csv(primary_csv)

    for i in range(0, len(extra_args), 2):
        mach = float(extra_args[i])
        eval_dir = extra_args[i + 1]
        column = f"CD_mach_{mach:.2f}"

        second_csv = os.path.join(eval_dir, "DD_80_combined.csv")
        if not os.path.isfile(second_csv):
            raise RuntimeError(f"Combined CSV not found for Mach {mach}: {second_csv}")

        df_second = pd.read_csv(second_csv)
        cd_map = _cd_by_airfoil(df_second, cl_target)
        df_primary[column] = df_primary["Airfoil_idx"].map(cd_map)
        print(f"Appended {column} from {second_csv}")

    df_primary.to_csv(primary_csv, index=False)
    print(f"Updated primary combined CSV: {primary_csv}")


def _cd_by_airfoil(df: pd.DataFrame, cl_target: float) -> dict[int, float]:
    if "Airfoil_idx" not in df.columns:
        raise RuntimeError("Combined CSV is missing Airfoil_idx.")
    if "CD" not in df.columns:
        raise RuntimeError("Combined CSV is missing CD.")

    result: dict[int, float] = {}
    for airfoil_idx, group in df.groupby("Airfoil_idx"):
        if "CL" in group.columns:
            distances = (pd.to_numeric(group["CL"], errors="coerce") - cl_target).abs()
            row = group.loc[distances.idxmin()]
        else:
            row = group.iloc[0]
        result[int(airfoil_idx)] = float(row["CD"])
    return result


if __name__ == "__main__":
    main()
