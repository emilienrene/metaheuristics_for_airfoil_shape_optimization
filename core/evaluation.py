"""
Post-processing of rAIFoil solver output.

Two responsibilities:
    combine_and_interpolate  — merge per-airfoil DD_80 CSVs and write one
                               target-CL objective row per airfoil.
    append_clmax             — read per-airfoil angle-sweep CSVs, compute
                               Cl_max, and append it as a column to the
                               combined CSV.
"""

from __future__ import annotations

import os
import re
import glob
import numpy as np
import pandas as pd

from .paths import latest_opt_dir, latest_eval_dir, clmax_eval_dir


def combine_and_interpolate(eval_dir: str, cl_target: float) -> str:
    """
    Merge all DD_80_airfoil_*.csv files in eval_dir, then write one operating
    row per airfoil. By default the row is interpolated at cl_target. When
    CL_TARGET_TOLERANCE is positive and a CM target is configured, the row is
    instead selected inside the permitted CL envelope to minimize CM error.

    DD_80_sweep.csv keeps the raw AoA sweep for diagnostics. DD_80_combined.csv
    is the optimizer-facing table. If an airfoil does not bracket cl_target, its
    row is retained with CL_target_reached=False and CD=NaN so fitness assigns a
    penalty instead of comparing drag at the wrong lift.

    Returns the path to the combined CSV.
    """
    csv_files = sorted([
        f for f in os.listdir(eval_dir)
        if f.startswith("DD_80_airfoil_") and f.endswith(".csv")
    ])

    if not csv_files:
        raise RuntimeError(f"No DD_80_airfoil_*.csv files found in {eval_dir}")

    frames = []
    for f in csv_files:
        match = re.search(r"DD_80_airfoil_(\d+)\.csv", f)
        airfoil_idx = int(match.group(1)) if match else -1
        df = pd.read_csv(os.path.join(eval_dir, f))
        df.insert(0, "Airfoil_idx", airfoil_idx)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined_path = os.path.join(eval_dir, "DD_80_combined.csv")
    sweep_path = os.path.join(eval_dir, "DD_80_sweep.csv")
    combined.to_csv(sweep_path, index=False)
    print(f"Raw AoA sweep CSV saved: {sweep_path}")

    target_rows = _target_cl_rows(combined, cl_target)
    target_rows.to_csv(combined_path, index=False)
    print(f"Target-CL combined CSV saved: {combined_path}")

    # delete individual files
    for f in csv_files:
        path = os.path.join(eval_dir, f)
        if os.path.exists(path):
            os.remove(path)

    return combined_path


def _target_cl_rows(df: pd.DataFrame, target_cl: float) -> pd.DataFrame:
    first_col = df.columns[0]
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if first_col in numeric_cols:
        numeric_cols.remove(first_col)

    rows: list[dict] = []

    for airfoil_idx, group in df.groupby(first_col):
        row = _interpolate_airfoil_at_cl(group, target_cl, numeric_cols)
        row[first_col] = int(airfoil_idx)
        rows.append(row)

    out = pd.DataFrame(rows)
    out[first_col] = out[first_col].astype(int)
    return out


def _interpolate_airfoil_at_cl(
    group: pd.DataFrame,
    target_cl: float,
    numeric_cols: list[str],
) -> dict:
    if "AOA" in group.columns:
        group = group.sort_values("AOA", kind="mergesort")

    group = group.reset_index(drop=True)
    fallback = _fallback_target_row(group, target_cl)

    if len(group) < 2 or "CL" not in group.columns:
        return fallback

    joint_row = _interpolate_in_cl_envelope_for_cm(
        group,
        target_cl,
        numeric_cols,
    )
    if joint_row is not None:
        return joint_row

    cl_values = pd.to_numeric(group["CL"], errors="coerce")

    for i in range(len(group)):
        cl_i = cl_values.iloc[i]
        if pd.notna(cl_i) and np.isclose(float(cl_i), target_cl, rtol=0.0, atol=1e-12):
            row = group.iloc[i].to_dict()
            row["CL_target"] = float(target_cl)
            row["CL_target_reached"] = True
            return row

    for i in range(len(group) - 1):
        cl0 = cl_values.iloc[i]
        cl1 = cl_values.iloc[i + 1]
        if pd.isna(cl0) or pd.isna(cl1) or float(cl0) == float(cl1):
            continue
        if not ((cl0 <= target_cl <= cl1) or (cl1 <= target_cl <= cl0)):
            continue

        frac = (target_cl - float(cl0)) / (float(cl1) - float(cl0))
        new_row = _interpolate_rows(
            group.iloc[i],
            group.iloc[i + 1],
            frac,
            numeric_cols,
        )

        new_row["CL"] = float(target_cl)
        new_row["CL_target"] = float(target_cl)
        new_row["CL_target_tolerance"] = 0.0
        new_row["CL_target_error"] = 0.0
        new_row["CL_target_reached"] = True
        return new_row

    return fallback


def _interpolate_in_cl_envelope_for_cm(
    group: pd.DataFrame,
    target_cl: float,
    numeric_cols: list[str],
) -> dict | None:
    """Choose the point closest to CM target within the permitted CL band."""
    settings = _cl_cm_selection_settings(group)
    if settings is None:
        return None

    cl_tolerance, cm_column, cm_target, cm_tolerance = settings
    cl_low = target_cl - cl_tolerance
    cl_high = target_cl + cl_tolerance
    candidates: list[tuple[tuple[float, float, float, float], dict]] = []

    cl_values = pd.to_numeric(group["CL"], errors="coerce")
    cm_values = pd.to_numeric(group[cm_column], errors="coerce")

    for index in range(len(group) - 1):
        cl0 = cl_values.iloc[index]
        cl1 = cl_values.iloc[index + 1]
        cm0 = cm_values.iloc[index]
        cm1 = cm_values.iloc[index + 1]
        if any(pd.isna(value) for value in (cl0, cl1, cm0, cm1)):
            continue

        interval = _fraction_interval_in_band(
            float(cl0),
            float(cl1),
            cl_low,
            cl_high,
        )
        if interval is None:
            continue
        lower_fraction, upper_fraction = interval

        cm_delta = float(cm1) - float(cm0)
        if abs(cm_delta) > 1e-15:
            fraction = np.clip(
                (cm_target - float(cm0)) / cm_delta,
                lower_fraction,
                upper_fraction,
            )
        else:
            fraction = _fraction_closest_to_target(
                float(cl0),
                float(cl1),
                target_cl,
                lower_fraction,
                upper_fraction,
            )

        row = _interpolate_rows(
            group.iloc[index],
            group.iloc[index + 1],
            float(fraction),
            numeric_cols,
        )
        selected_cl = float(row["CL"])
        selected_cm = float(row[cm_column])
        cm_error = abs(selected_cm - cm_target)
        cl_error = abs(selected_cl - target_cl)
        cd = float(row.get("CD", np.inf))
        if not np.isfinite(cd):
            cd = np.inf
        aoa = float(row.get("AOA", np.inf))
        if not np.isfinite(aoa):
            aoa = np.inf

        row["CL_target"] = float(target_cl)
        row["CL_target_tolerance"] = float(cl_tolerance)
        row["CL_target_error"] = float(cl_error)
        row["CL_target_reached"] = True
        row["CM_target"] = float(cm_target)
        row["CM_target_tolerance"] = float(cm_tolerance)
        row["CM_target_error"] = float(cm_error)
        row["CM_target_reached"] = bool(cm_error <= cm_tolerance)
        candidates.append(((cm_error, cl_error, cd, aoa), row))

    if not candidates:
        return None
    return min(candidates, key=lambda candidate: candidate[0])[1]


def _cl_cm_selection_settings(
    group: pd.DataFrame,
) -> tuple[float, str, float, float] | None:
    cl_tolerance = float(os.environ.get("CL_TARGET_TOLERANCE", "0.0"))
    if not np.isfinite(cl_tolerance) or cl_tolerance < 0.0:
        raise ValueError(
            f"CL_TARGET_TOLERANCE must be finite and nonnegative; got {cl_tolerance}."
        )
    if cl_tolerance == 0.0:
        return None

    normalized_columns = {
        str(column).strip().lower().replace("_", ""): column
        for column in group.columns
    }
    cm_column = normalized_columns.get("cmz") or normalized_columns.get("cm")
    if cm_column is None:
        return None

    cm_target_raw = os.environ.get(
        "CM_SELECTION_TARGET",
        os.environ.get("DRELA_DAE11_CM_TARGET"),
    )
    if cm_target_raw is None:
        return None
    cm_target = float(cm_target_raw)

    cm_tolerance = float(os.environ.get(
        "CM_SELECTION_TOLERANCE",
        os.environ.get("DRELA_DAE11_CM_SCALE", "0.005"),
    ))
    if not np.isfinite(cm_target):
        raise ValueError(f"CM selection target must be finite; got {cm_target}.")
    if not np.isfinite(cm_tolerance) or cm_tolerance <= 0.0:
        raise ValueError(
            "CM_SELECTION_TOLERANCE must be finite and positive; "
            f"got {cm_tolerance}."
        )
    return cl_tolerance, cm_column, cm_target, cm_tolerance


def _fraction_interval_in_band(
    value0: float,
    value1: float,
    lower: float,
    upper: float,
) -> tuple[float, float] | None:
    delta = value1 - value0
    if abs(delta) <= 1e-15:
        return (0.0, 1.0) if lower <= value0 <= upper else None

    first = (lower - value0) / delta
    second = (upper - value0) / delta
    interval_lower = max(0.0, min(first, second))
    interval_upper = min(1.0, max(first, second))
    if interval_lower > interval_upper + 1e-15:
        return None
    return float(interval_lower), float(interval_upper)


def _fraction_closest_to_target(
    value0: float,
    value1: float,
    target: float,
    lower_fraction: float,
    upper_fraction: float,
) -> float:
    delta = value1 - value0
    if abs(delta) <= 1e-15:
        return float(lower_fraction)
    return float(np.clip(
        (target - value0) / delta,
        lower_fraction,
        upper_fraction,
    ))


def _interpolate_rows(
    row0: pd.Series,
    row1: pd.Series,
    fraction: float,
    numeric_cols: list[str],
) -> dict:
    row: dict = {}
    for column in row0.index:
        if column in numeric_cols:
            value0 = pd.to_numeric(
                pd.Series([row0[column]]), errors="coerce"
            ).iloc[0]
            value1 = pd.to_numeric(
                pd.Series([row1[column]]), errors="coerce"
            ).iloc[0]
            if pd.notna(value0) and pd.notna(value1):
                row[column] = float(value0 + fraction * (value1 - value0))
            else:
                row[column] = np.nan
        else:
            row[column] = row0[column]
    return row


def _fallback_target_row(group: pd.DataFrame, target_cl: float) -> dict:
    if group.empty:
        row: dict = {}
    elif "CL" in group.columns:
        cl = pd.to_numeric(group["CL"], errors="coerce")
        if cl.notna().any():
            row = group.loc[(cl - target_cl).abs().idxmin()].to_dict()
        else:
            row = group.iloc[0].to_dict()
    else:
        row = group.iloc[0].to_dict()

    row["CL_target"] = float(target_cl)
    row["CL_target_tolerance"] = float(
        os.environ.get("CL_TARGET_TOLERANCE", "0.0")
    )
    if "CL" in row and pd.notna(row["CL"]):
        row["CL_target_error"] = abs(float(row["CL"]) - float(target_cl))
    else:
        row["CL_target_error"] = np.nan
    row["CL_target_reached"] = False
    if "CD" in row:
        row["CD"] = np.nan
    return row


def append_clmax(opt_dir: str | None = None) -> None:
    """
    Read per-airfoil angle-sweep CSVs from evaluation_Mach_* and append
    Cl_max as a column to the DD_80_combined.csv in evaluation_mach_*.
    """
    opt = opt_dir or latest_opt_dir()
    combined_csv = os.path.join(latest_eval_dir(opt), "DD_80_combined.csv")
    airfoil_dir  = clmax_eval_dir(opt)

    airfoil_files = sorted(glob.glob(os.path.join(airfoil_dir, "DD_80_airfoil_*.csv")))

    cl_max_dict: dict[int, float] = {}
    for af_file in airfoil_files:
        idx = int(os.path.basename(af_file).split("_")[-1].split(".")[0])
        df  = pd.read_csv(af_file)
        cl_max_dict[idx] = float(df.iloc[:, 3].max())

    lines: list[str] = []
    with open(combined_csv) as f:
        lines.append(f.readline().rstrip() + ",Cl_max")
        for line in f:
            parts   = line.rstrip().split(",")
            af_idx  = int(parts[0])
            cl_max  = cl_max_dict.get(af_idx, "")
            lines.append(line.rstrip() + f",{cl_max}")

    with open(combined_csv, "w") as f:
        f.write("\n".join(lines))

    print(f"Cl_max appended to {combined_csv}")


def append_second_cd(
    primary_eval_dir: str,
    second_eval_dir: str,
    mach: float | str | None = None,
    cl_target: float | None = None,
) -> str:
    """
    Append the Cd measured at a second Mach point to the primary combined CSV.

    Both evaluation directories are expected to have already been processed by
    run_concat.py, so each contains a DD_80_combined.csv. The second Mach Cd is
    matched by Airfoil_idx and written into the primary combined file.
    """
    primary_csv = os.path.join(primary_eval_dir, "DD_80_combined.csv")
    second_csv = os.path.join(second_eval_dir, "DD_80_combined.csv")

    if not os.path.isfile(primary_csv):
        raise RuntimeError(f"Primary combined CSV not found: {primary_csv}")
    if not os.path.isfile(second_csv):
        raise RuntimeError(f"Second Mach combined CSV not found: {second_csv}")

    df_primary = pd.read_csv(primary_csv)
    df_second = pd.read_csv(second_csv)

    if "Airfoil_idx" not in df_primary.columns or "Airfoil_idx" not in df_second.columns:
        raise RuntimeError("Both combined CSVs must contain Airfoil_idx.")
    if "CD" not in df_second.columns:
        raise RuntimeError(f"Second Mach CSV has no CD column: {second_csv}")

    if mach is None:
        column = "CD_mach2"
        reached_column = "CL_target_reached_mach2"
    else:
        column = f"CD_mach_{float(mach):.2f}"
        reached_column = f"CL_target_reached_mach_{float(mach):.2f}"

    cd_map = _cd_by_airfoil(df_second, cl_target)
    reached_map = _target_reached_by_airfoil(df_second)
    df_primary[column] = df_primary["Airfoil_idx"].map(cd_map)
    df_primary[reached_column] = (
        df_primary["Airfoil_idx"].map(reached_map).fillna(False).astype(bool)
    )
    df_primary.to_csv(primary_csv, index=False)

    print(f"Second Mach Cd appended to {primary_csv} as {column}")
    return primary_csv


def _cd_by_airfoil(df: pd.DataFrame, cl_target: float | None) -> dict[int, float]:
    """
    Return one Cd value per airfoil from a combined rAIFoil CSV.

    If cl_target is available, choose the row closest to that target CL. In the
    normal pipeline this is the interpolated row inserted by run_concat.py.
    """
    result: dict[int, float] = {}
    for airfoil_idx, group in df.groupby("Airfoil_idx"):
        if group.empty:
            continue
        if "CL_target_reached" in group.columns and not bool(group["CL_target_reached"].any()):
            result[int(airfoil_idx)] = np.nan
            continue
        if cl_target is not None and "CL" in group.columns:
            distances = (group["CL"].astype(float) - float(cl_target)).abs()
            row = group.loc[distances.idxmin()]
        else:
            row = group.iloc[0]
        result[int(airfoil_idx)] = float(row["CD"])
    return result


def _target_reached_by_airfoil(df: pd.DataFrame) -> dict[int, bool]:
    result: dict[int, bool] = {}
    if "CL_target_reached" not in df.columns:
        return result

    reached = df["CL_target_reached"].astype(str).str.lower().isin(
        {"1", "true", "yes", "y"}
    )
    for airfoil_idx, group_reached in reached.groupby(df["Airfoil_idx"]):
        result[int(airfoil_idx)] = bool(group_reached.any())
    return result
