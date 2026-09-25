"""
Fitness ranking and convergence tracking.

Multi-point Cd minimization version.

The important editable part is intentionally simple:

    obj1 = ...
    obj2 = ...
    obj3 = ...
    obj4 = ...
    coeff1 = ...
    coeff2 = ...
    coeff3 = ...
    coeff4 = ...
    cost = ...

Lower cost is better. Invalid airfoils receive a large penalty.
"""

from __future__ import annotations

import csv
import os
import re
from collections.abc import Sequence

import numpy as np
import pandas as pd

from .paths import latest_opt_dir, latest_eval_dir, design_space_dir, next_gen_index

AirfoilSurfaces = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def rank(
    current_gen: int,
    coeff_cd: float | Sequence[float] | None = None,
    coeff_cd2: float | None = None,
    aoa_min: float | None = None,
    aoa_max: float | None = None,
    opt_dir: str | None = None,
) -> str:
    coeffs = _coefficients(coeff_cd, coeff_cd2)
    if aoa_min is None:
        aoa_min = float(os.environ["AOA_MIN"])
    if aoa_max is None:
        aoa_max = float(os.environ["AOA_MAX"])

    opt = opt_dir or latest_opt_dir()
    eval_dir = _primary_cd_eval_dir(opt)
    ds_dir = design_space_dir(opt)
    geom_dir = _geometry_dir_for_rank(opt, ds_dir, current_gen)

    combined_path = os.path.join(eval_dir, "DD_80_combined.csv")
    population_path = os.path.join(ds_dir, "population.csv")

    df_perf = pd.read_csv(combined_path)
    df_pop = pd.read_csv(population_path)

    optimizer = os.environ.get("OPTIMIZER", "ga").lower()
    is_de = optimizer == "de"
    surface_cache: dict[str, AirfoilSurfaces | None] = {}

    if is_de and current_gen > 1:
        pop_size = len(df_pop) // 2
        df_perf = _de_selection(
            df_perf,
            pop_size,
            coeffs,
            aoa_min,
            aoa_max,
            geom_dir,
            current_gen,
            surface_cache=surface_cache,
        )
        df_pop = _de_pop_selection(df_pop, df_perf)

        df_perf = df_perf.copy()
        df_perf["source_airfoil_idx"] = df_perf["Airfoil_idx"].astype(int)
        winner_ids = sorted(df_perf["Airfoil_idx"].unique())
        id_remap = {old: new for new, old in enumerate(winner_ids, start=1)}
        df_perf["Airfoil_idx"] = df_perf["Airfoil_idx"].map(id_remap)

    df_ranked = _rank_rows(
        df_perf,
        coeffs,
        aoa_min,
        aoa_max,
        geom_dir,
        current_gen,
        surface_cache=surface_cache,
    )
    df_ranked.to_csv(combined_path, index=False)

    sorted_idx = df_ranked["Airfoil_idx"]
    df_pop = df_pop.set_index("Airfoil_idx").loc[sorted_idx].reset_index()
    df_pop.to_csv(population_path, index=False)
    _write_optimizer_costs(opt, df_pop, df_ranked)

    best_idx = int(df_ranked.iloc[0]["Airfoil_idx"])
    best_file_idx = int(df_ranked.iloc[0].get("source_airfoil_idx", best_idx))
    if current_gen == 1:
        best_file = os.path.join(ds_dir, f"airfoil_{best_file_idx}.dat")
    else:
        best_file = os.path.join(
            opt,
            f"generation_{current_gen - 1}",
            f"airfoil_{best_file_idx}.dat",
        )

    best_pointer = os.path.join(opt, "best_airfoil.dat")
    with open(best_pointer, "w") as f:
        f.write(best_file + "\n")

    # The first row is the selected best. This differs from the minimum raw
    # objective when hierarchical constraint ranking is active. Keep the raw
    # aerodynamic objective separate from the ranking fitness so downstream
    # drag-reduction reports never interpret a constraint penalty as Cd.
    best_row = df_ranked.iloc[0]
    best_fitness = float(best_row["cost"])
    best_total_cost = float(best_row.get("selection_cost", best_fitness))
    # Re-evaluate the already-selected row with the basic Cd objective. This is
    # deliberately done after ranking and cannot affect optimizer selection.
    best_objective = float(
        _cost(best_row.to_frame().T, len(coeffs)).iloc[0]
    )
    best_penalty = best_fitness - best_objective
    best_feasible = (
        bool(best_row.get("evaluation_valid", False))
        and np.isfinite(best_objective)
        and best_objective > 0.0
    )
    if _dae11_constraints_enabled():
        feasibility_tolerance = float(os.environ.get(
            "DRELA_DAE11_FEASIBILITY_TOL", "1.0"
        ))
        normalized_violation = float(
            best_row.get("dae11_constraint_violation", np.inf)
        )
        best_feasible = (
            best_feasible
            and np.isfinite(normalized_violation)
            and normalized_violation <= feasibility_tolerance
        )
    convergence_path = os.path.join(opt, "convergence.csv")
    gen_index = next_gen_index(opt)
    _append_convergence_row(
        convergence_path,
        generation=gen_index,
        total_cost=best_total_cost,
        cd=best_objective,
        feasible=best_feasible,
    )
    _print_generation_summary(gen_index, best_fitness, best_penalty, best_row)
    return best_file


_CONVERGENCE_FIELDS = [
    "generation",
    "total_cost",
    "cd",
    "feasible",
]


def _append_convergence_row(
    path: str,
    *,
    generation: int,
    total_cost: float,
    cd: float,
    feasible: bool,
) -> None:
    """Append one compact convergence row without rereading prior history."""
    file_exists = os.path.isfile(path) and os.path.getsize(path) > 0
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CONVERGENCE_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "generation": generation,
            "total_cost": total_cost,
            "cd": cd,
            "feasible": "yes" if feasible else "no",
        })


def _print_generation_summary(
    generation: int,
    best_cost: float,
    penalty: float,
    best_row: pd.Series,
) -> None:
    """Print only the compact progress block intended for live monitoring."""
    print(f"Generation {generation} best cost: {best_cost:.10g}", flush=True)
    print(f"Penalty on best cost: {penalty:.10g}", flush=True)

    selected_cl = float(best_row.get("CL", np.nan))
    if np.isfinite(selected_cl):
        print(f"CL: {selected_cl:.10g}", flush=True)

    if not _dae11_constraints_enabled():
        return

    labels = {
        "cm": "CM",
        "t33": "t/c at x/c=0.33",
        "t90": "t/c at x/c=0.90",
        "te_angle": "TE angle (deg)",
        "le_mismatch": "LE mismatch",
    }
    for metric in _dae11_active_constraints():
        value = float(best_row[f"dae11_{metric}"])
        print(f"{labels[metric]}: {value:.10g}", flush=True)


def _rank_rows(
    df: pd.DataFrame,
    coeffs: Sequence[float],
    aoa_min: float,
    aoa_max: float,
    geometry_dir: str | None = None,
    current_gen: int | None = None,
    surface_cache: dict[str, AirfoilSurfaces | None] | None = None,
) -> pd.DataFrame:
    if surface_cache is None:
        surface_cache = {}
    df = df.copy()
    n_points = len(coeffs)
    df["cost"] = _cost(df, n_points)
    constraint_ranking = False
    constraint_mode = "penalty"
    if _dae11_constraints_enabled():
        diagnostics = _dae11_constraint_diagnostics(df, geometry_dir, surface_cache)
        for column in diagnostics.columns:
            df[column] = diagnostics[column]
        constraint_mode = _dae11_constraint_mode()
        constraint_ranking = constraint_mode in {"hierarchical", "deb"}
        if constraint_mode == "penalty":
            df["cost"] = df["cost"] + df["dae11_constraint_penalty"]

    valid = (
        (pd.to_numeric(df["AOA"], errors="coerce") > aoa_min)
        & (pd.to_numeric(df["AOA"], errors="coerce") < aoa_max)
        & pd.to_numeric(df["CD"], errors="coerce").notna()
        & (pd.to_numeric(df["CD"], errors="coerce") > 0)
    )
    primary_reached_col = _target_reached_name(df, 1)
    if primary_reached_col is not None:
        valid &= _truthy(df[primary_reached_col])

    for point_idx in range(2, n_points + 1):
        cd_col = _cd_name(df, point_idx)
        valid &= (
            pd.to_numeric(df[cd_col], errors="coerce").notna()
            & (pd.to_numeric(df[cd_col], errors="coerce") > 0)
        )
        reached_col = _target_reached_name(df, point_idx)
        if reached_col is not None:
            valid &= _truthy(df[reached_col])

    geometry_valid = _geometry_valid_mask(df, geometry_dir, surface_cache)
    valid &= geometry_valid
    df["geometry_valid"] = geometry_valid

    if constraint_ranking:
        violation = pd.to_numeric(
            df["dae11_constraint_violation"], errors="coerce"
        )
        violation_l2 = pd.to_numeric(
            df["dae11_constraint_violation_l2"], errors="coerce"
        )
        constraint_metrics_valid = np.isfinite(violation) & np.isfinite(violation_l2)
        valid &= constraint_metrics_valid

        epsilon = (
            _dae11_constraint_epsilon(current_gen)
            if constraint_mode == "hierarchical"
            else float(os.environ.get("DRELA_DAE11_FEASIBILITY_TOL", "1.0"))
        )
        if not np.isfinite(epsilon) or epsilon < 0.0:
            raise ValueError(f"Invalid DAE-11 feasibility threshold: {epsilon}")
        feasible = valid & (violation <= epsilon)
        df["dae11_constraint_epsilon"] = epsilon
        df["dae11_constraint_feasible"] = feasible
        df["dae11_constraint_tier"] = np.where(feasible, 0, np.where(valid, 1, 2))

    valid_costs = pd.to_numeric(df.loc[valid, "cost"], errors="coerce")
    valid_costs = valid_costs[np.isfinite(valid_costs)]
    penalty = 100.0 if valid_costs.empty else max(100.0, float(valid_costs.max()) * 10.0)

    rows = []
    for _, group in df.groupby("Airfoil_idx", sort=False):
        valid_group = group[valid.loc[group.index]]
        if valid_group.empty:
            row = group.iloc[0].copy()
            row["cost"] = penalty
            row["evaluation_valid"] = False
            if constraint_ranking:
                row["dae11_constraint_feasible"] = False
                row["dae11_constraint_tier"] = 2
        else:
            if constraint_ranking:
                feasible_group = valid_group[
                    valid_group["dae11_constraint_feasible"].astype(bool)
                ]
                if not feasible_group.empty:
                    row = feasible_group.sort_values(
                        ["cost", "dae11_constraint_violation"],
                        ascending=True,
                        kind="mergesort",
                    ).iloc[0].copy()
                else:
                    row = valid_group.sort_values(
                        [
                            "dae11_constraint_violation",
                            "dae11_constraint_violation_l2",
                            "cost",
                        ],
                        ascending=True,
                        kind="mergesort",
                    ).iloc[0].copy()
            else:
                row = valid_group.loc[valid_group["cost"].idxmin()].copy()
            row["evaluation_valid"] = True
        rows.append(row)

    ranked = pd.DataFrame(rows)
    ranked["Airfoil_idx"] = ranked["Airfoil_idx"].astype(int)
    if "source_airfoil_idx" in ranked.columns:
        ranked["source_airfoil_idx"] = ranked["source_airfoil_idx"].astype(int)
    if constraint_ranking:
        objective = pd.to_numeric(ranked["cost"], errors="coerce")
        violation = pd.to_numeric(
            ranked["dae11_constraint_violation"], errors="coerce"
        )
        violation_l2 = pd.to_numeric(
            ranked["dae11_constraint_violation_l2"], errors="coerce"
        )
        tier = pd.to_numeric(ranked["dae11_constraint_tier"], errors="coerce")

        ranked["_hierarchy_primary"] = np.where(tier == 0, objective, violation)
        ranked["_hierarchy_secondary"] = np.where(
            tier == 0, violation, violation_l2
        )
        ranked["_hierarchy_tertiary"] = objective
        ranked = ranked.sort_values(
            [
                "dae11_constraint_tier",
                "_hierarchy_primary",
                "_hierarchy_secondary",
                "_hierarchy_tertiary",
            ],
            ascending=True,
            kind="mergesort",
        ).reset_index(drop=True)

        # Stateful optimizers consume a scalar sidecar cost. These disjoint
        # intervals preserve the hierarchy without allowing objective value to
        # compensate for infeasibility.
        sorted_objective = pd.to_numeric(ranked["cost"], errors="coerce")
        sorted_violation = pd.to_numeric(
            ranked["dae11_constraint_violation"], errors="coerce"
        )
        sorted_tier = pd.to_numeric(
            ranked["dae11_constraint_tier"], errors="coerce"
        )
        objective_score = _bounded_nonnegative(sorted_objective)
        violation_score = _bounded_nonnegative(sorted_violation)
        ranked["selection_cost"] = np.where(
            sorted_tier == 0,
            objective_score,
            np.where(sorted_tier == 1, 1.0 + violation_score, 2.0),
        )
        return ranked.drop(
            columns=[
                "_hierarchy_primary",
                "_hierarchy_secondary",
                "_hierarchy_tertiary",
            ]
        )

    ranked = ranked.sort_values("cost", ascending=True, kind="mergesort")
    ranked = ranked.reset_index(drop=True)
    ranked["selection_cost"] = pd.to_numeric(ranked["cost"], errors="coerce")
    return ranked


def _cost(df: pd.DataFrame, n_points: int) -> pd.Series:
    """
    Edit this function however you want.

    obj1, obj2, obj3, and obj4 are the objective variables. The cost line is
    deliberately one line so you can hard-code experiments quickly.
    """
    coeff1 = 1.0
    coeff2 = 1.0
    coeff3 = 1.0
    coeff4 = 1.0

    cd1 = pd.to_numeric(df["CD"], errors="coerce")
    cd2 = pd.to_numeric(df[_cd_name(df, 2)], errors="coerce") if n_points >= 2 else 0.0
    cd3 = pd.to_numeric(df[_cd_name(df, 3)], errors="coerce") if n_points >= 3 else 0.0
    cd4 = pd.to_numeric(df[_cd_name(df, 4)], errors="coerce") if n_points >= 4 else 0.0

    obj1 = cd1
    obj2 = cd2
    obj3 = cd3
    obj4 = cd4
    cost = obj1 * coeff1 + obj2 * coeff2 + obj3 * coeff3 + obj4 * coeff4

    return cost


def _dae11_constraints_enabled() -> bool:
    return os.environ.get("DRELA_DAE11_CONSTRAINTS", "0").strip().lower() in {
        "1", "true", "yes", "y",
    }


def _dae11_active_constraints() -> list[str]:
    """Return the enabled DAE-11 equality metrics in deterministic order."""
    configured = os.environ.get("DRELA_DAE11_ACTIVE_CONSTRAINTS")
    if configured is None:
        metrics = ["cm", "t33", "t90", "te_angle"]
        include_le = os.environ.get(
            "DRELA_DAE11_INCLUDE_LE_PENALTY", "0"
        ).strip().lower() in {"1", "true", "yes", "y"}
        if include_le:
            metrics.append("le_mismatch")
        return metrics

    aliases = {
        "cm": "cm",
        "cmz": "cm",
        "moment": "cm",
        "t33": "t33",
        "thickness33": "t33",
        "thickness_33": "t33",
        "t90": "t90",
        "thickness90": "t90",
        "thickness_90": "t90",
        "te": "te_angle",
        "teangle": "te_angle",
        "te_angle": "te_angle",
        "le": "le_mismatch",
        "leangle": "le_mismatch",
        "le_angle": "le_mismatch",
        "le_mismatch": "le_mismatch",
    }
    requested = [
        item.strip().lower()
        for item in configured.replace(";", ",").replace(" ", ",").split(",")
        if item.strip()
    ]
    metrics = []
    for item in requested:
        if item not in aliases:
            raise ValueError(
                "Unknown DRELA_DAE11_ACTIVE_CONSTRAINTS entry "
                f"{item!r}; use cm,t33,t90,te_angle,le_mismatch."
            )
        metric = aliases[item]
        if metric not in metrics:
            metrics.append(metric)
    if not metrics:
        raise ValueError(
            "DRELA_DAE11_CONSTRAINTS=1 requires at least one entry in "
            "DRELA_DAE11_ACTIVE_CONSTRAINTS."
        )
    return metrics


def _dae11_constraint_mode() -> str:
    mode = os.environ.get("DRELA_DAE11_CONSTRAINT_MODE", "penalty").strip().lower()
    aliases = {
        "penalty": "penalty",
        "quadratic": "penalty",
        "hierarchical": "hierarchical",
        "hierarchy": "hierarchical",
        "rank": "hierarchical",
        "ranking": "hierarchical",
        "deb": "deb",
        "feasibility": "deb",
        "feasibility_first": "deb",
        "feasibility-first": "deb",
    }
    if mode not in aliases:
        raise ValueError(
            "DRELA_DAE11_CONSTRAINT_MODE must be 'penalty', 'hierarchical', "
            "or 'deb', "
            f"got {mode!r}."
        )
    return aliases[mode]


def _dae11_constraint_epsilon(current_gen: int | None) -> float:
    start = float(os.environ.get("DRELA_DAE11_EPSILON_START", "5.0"))
    end = float(os.environ.get("DRELA_DAE11_EPSILON_END", "1.0"))
    total_generations = int(os.environ.get("NUM_GEN", "500"))
    if start <= 0.0 or end <= 0.0 or not np.isfinite(start + end):
        raise ValueError(
            "DRELA_DAE11_EPSILON_START and DRELA_DAE11_EPSILON_END "
            "must be finite positive values."
        )
    if total_generations <= 0:
        return end

    generation = 0 if current_gen is None else max(0, int(current_gen) - 1)
    progress = min(float(generation) / float(total_generations), 1.0)
    return float(start * (end / start) ** progress)


def _bounded_nonnegative(values: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    numeric = np.where(np.isfinite(numeric), np.maximum(numeric, 0.0), np.inf)
    return np.divide(
        numeric,
        1.0 + numeric,
        out=np.ones_like(numeric),
        where=np.isfinite(numeric),
    )


def _dae11_penalty_weight(
    df: pd.DataFrame,
    normalized_violation: pd.Series,
) -> tuple[float, int, int, float]:
    """Return fixed or population-feasibility-adaptive penalty diagnostics."""
    fixed_weight = float(os.environ.get("DRELA_DAE11_CONSTRAINT_WEIGHT", "0.01"))
    if not np.isfinite(fixed_weight) or fixed_weight <= 0.0:
        raise ValueError(
            f"Invalid DRELA_DAE11_CONSTRAINT_WEIGHT: {fixed_weight}"
        )

    tolerance = float(os.environ.get("DRELA_DAE11_FEASIBILITY_TOL", "1.0"))
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError(
            f"Invalid DRELA_DAE11_FEASIBILITY_TOL: {tolerance}"
        )

    population = pd.DataFrame({
        "Airfoil_idx": pd.to_numeric(df["Airfoil_idx"], errors="coerce"),
        "violation": pd.to_numeric(normalized_violation, errors="coerce"),
    }).dropna(subset=["Airfoil_idx"])
    # A one-point run has one row per airfoil. Taking the minimum also handles
    # a future multi-row evaluation by asking whether that individual has any
    # operating row satisfying all active constraints.
    per_airfoil = population.groupby("Airfoil_idx", sort=False)["violation"].min()
    population_count = int(per_airfoil.size)
    feasible_count = int(
        (np.isfinite(per_airfoil.to_numpy(dtype=float)) & (per_airfoil <= tolerance)).sum()
    )
    feasible_fraction = (
        float(feasible_count) / float(population_count)
        if population_count else 0.0
    )

    adaptive = os.environ.get(
        "DRELA_DAE11_ADAPTIVE_PENALTY", "0"
    ).strip().lower() in {"1", "true", "yes", "y", "on"}
    if not adaptive:
        return fixed_weight, feasible_count, population_count, feasible_fraction

    minimum = float(os.environ.get(
        "DRELA_DAE11_PENALTY_WEIGHT_MIN",
        str(0.1 * fixed_weight),
    ))
    maximum = float(os.environ.get(
        "DRELA_DAE11_PENALTY_WEIGHT_MAX",
        str(fixed_weight),
    ))
    power = float(os.environ.get("DRELA_DAE11_PENALTY_SCHEDULE_POWER", "1.0"))
    if (
        not np.isfinite(minimum)
        or not np.isfinite(maximum)
        or minimum < 0.0
        or maximum <= 0.0
        or minimum > maximum
    ):
        raise ValueError(
            "Adaptive DAE-11 penalty weights must satisfy "
            f"0 <= min <= max and max > 0; got min={minimum}, max={maximum}."
        )
    if not np.isfinite(power) or power <= 0.0:
        raise ValueError(
            f"DRELA_DAE11_PENALTY_SCHEDULE_POWER must be positive; got {power}."
        )

    weight = minimum + (maximum - minimum) * feasible_fraction ** power
    return float(weight), feasible_count, population_count, feasible_fraction


def _dae11_constraint_diagnostics(
    df: pd.DataFrame,
    geometry_dir: str | None,
    surface_cache: dict[str, AirfoilSurfaces | None] | None = None,
) -> pd.DataFrame:
    """Return the Section 3.1 DAE-11 equality-constraint penalty terms."""
    if geometry_dir is None:
        raise RuntimeError("DAE-11 constraints require an airfoil geometry directory.")

    normalized_columns = {
        str(column).strip().lower().replace("_", ""): column
        for column in df.columns
    }
    # rAIFoil writes the section pitching-moment coefficient as CMZ (sometimes
    # CM_Z). Normalization maps either spelling to cmz. Retain CM as a fallback.
    active_constraints = _dae11_active_constraints()
    cm_column = normalized_columns.get("cmz") or normalized_columns.get("cm")
    if cm_column is None and "cm" in active_constraints:
        raise RuntimeError(
            "The active DAE-11 CM constraint requires a CMZ/CM_Z (or legacy "
            "CM) column in rAIFoil output."
        )

    id_column = (
        "source_airfoil_idx" if "source_airfoil_idx" in df.columns else "Airfoil_idx"
    )
    file_ids = pd.to_numeric(df[id_column], errors="coerce")
    metrics_by_id: dict[int, dict[str, float]] = {}
    for file_id in sorted(file_ids.dropna().astype(int).unique()):
        airfoil_path = os.path.join(geometry_dir, f"airfoil_{file_id}.dat")
        try:
            metrics_by_id[file_id] = _dae11_geometry_metrics(airfoil_path, surface_cache)
        except Exception:
            metrics_by_id[file_id] = {
                "t33": np.nan,
                "t90": np.nan,
                "te_angle": np.nan,
                "le_mismatch": np.nan,
            }

    result = pd.DataFrame(index=df.index)
    if cm_column is None:
        result["dae11_cm"] = np.nan
    else:
        result["dae11_cm"] = pd.to_numeric(df[cm_column], errors="coerce")
    for metric in ("t33", "t90", "te_angle", "le_mismatch"):
        values = {file_id: data[metric] for file_id, data in metrics_by_id.items()}
        result[f"dae11_{metric}"] = file_ids.map(values)

    targets = {
        "cm": float(os.environ.get("DRELA_DAE11_CM_TARGET", "0.053639")),
        "t33": float(os.environ.get("DRELA_DAE11_T33_TARGET", "0.128")),
        "t90": float(os.environ.get("DRELA_DAE11_T90_TARGET", "0.014")),
        "te_angle": float(os.environ.get("DRELA_DAE11_TE_ANGLE_TARGET", "6.25")),
        "le_mismatch": float(os.environ.get("DRELA_DAE11_LE_MISMATCH_TARGET", "0.0")),
    }
    scales = {
        "cm": float(os.environ.get("DRELA_DAE11_CM_SCALE", "0.005")),
        "t33": float(os.environ.get("DRELA_DAE11_T33_SCALE", "0.003")),
        "t90": float(os.environ.get("DRELA_DAE11_T90_SCALE", "0.001")),
        "te_angle": float(os.environ.get("DRELA_DAE11_TE_ANGLE_SCALE", "1.0")),
        "le_mismatch": float(os.environ.get("DRELA_DAE11_LE_MISMATCH_SCALE", "0.003")),
    }
    if any(not np.isfinite(value) or value <= 0.0 for value in scales.values()):
        raise ValueError(f"Invalid DAE-11 constraint scales: {scales}")

    squared_violation = pd.Series(0.0, index=df.index)
    normalized_violations = []
    for metric in active_constraints:
        residual = (result[f"dae11_{metric}"] - targets[metric]) / scales[metric]
        normalized = residual.abs()
        result[f"dae11_{metric}_normalized_violation"] = normalized
        normalized_violations.append(normalized)
        squared_violation = squared_violation + residual ** 2

    result["dae11_constraint_violation"] = pd.concat(
        normalized_violations, axis=1
    ).max(axis=1, skipna=False)
    result["dae11_constraint_violation_l2"] = squared_violation

    weight, feasible_count, population_count, feasible_fraction = (
        _dae11_penalty_weight(df, result["dae11_constraint_violation"])
    )
    result["dae11_penalty_weight"] = weight
    result["dae11_feasible_count"] = feasible_count
    result["dae11_population_count"] = population_count
    result["dae11_feasible_fraction"] = feasible_fraction
    result["dae11_constraint_penalty"] = (
        weight * squared_violation
    ).where(np.isfinite(squared_violation), 1.0e6)
    return result


def _dae11_geometry_metrics(
    airfoil_path: str,
    surface_cache: dict[str, AirfoilSurfaces | None] | None = None,
) -> dict[str, float]:
    x_up, z_up, x_lo, z_lo = _load_airfoil_surfaces(airfoil_path, surface_cache)

    t33 = float(np.interp(0.33, x_up, z_up) - np.interp(0.33, x_lo, z_lo))
    t90 = float(np.interp(0.90, x_up, z_up) - np.interp(0.90, x_lo, z_lo))

    slope_up = _endpoint_slope(x_up, z_up)
    slope_lo = _endpoint_slope(x_lo, z_lo)
    te_angle = abs(np.degrees(np.arctan(slope_lo) - np.arctan(slope_up)))

    a_upper = _leading_sqrt_coefficient(x_up, z_up)
    a_lower = _leading_sqrt_coefficient(x_lo, z_lo)
    le_mismatch = a_upper + a_lower
    return {
        "t33": t33,
        "t90": t90,
        "te_angle": float(te_angle),
        "le_mismatch": float(le_mismatch),
    }


def _endpoint_slope(x: np.ndarray, z: np.ndarray) -> float:
    n_fit = min(3, len(x))
    if n_fit < 2:
        raise ValueError("Too few trailing-edge points to estimate slope.")
    degree = min(2, n_fit - 1)
    polynomial = np.polyfit(x[-n_fit:], z[-n_fit:], degree)
    return float(np.polyval(np.polyder(polynomial), 1.0))


def _leading_sqrt_coefficient(x: np.ndarray, z: np.ndarray) -> float:
    x0 = float(x[0])
    local_x = x - x0
    mask = (local_x > 0.0) & (local_x <= 0.05)
    if int(np.count_nonzero(mask)) < 3:
        positive = np.flatnonzero(local_x > 0.0)[:5]
        mask = np.zeros_like(local_x, dtype=bool)
        mask[positive] = True
    if int(np.count_nonzero(mask)) < 3:
        raise ValueError("Too few leading-edge points to estimate nose slope.")

    x_fit = local_x[mask]
    z_fit = z[mask] - float(z[0])
    basis = np.column_stack((np.sqrt(x_fit), x_fit, x_fit ** 1.5))
    return float(np.linalg.lstsq(basis, z_fit, rcond=None)[0][0])


def _geometry_dir_for_rank(opt_dir: str, ds_dir: str, current_gen: int) -> str:
    if current_gen <= 1:
        return ds_dir
    return os.path.join(opt_dir, f"generation_{current_gen - 1}")


def _geometry_valid_mask(
    df: pd.DataFrame,
    geometry_dir: str | None,
    surface_cache: dict[str, AirfoilSurfaces | None] | None = None,
) -> pd.Series:
    if geometry_dir is None:
        return pd.Series(True, index=df.index)

    id_col = "source_airfoil_idx" if "source_airfoil_idx" in df.columns else "Airfoil_idx"
    file_ids = pd.to_numeric(df[id_col], errors="coerce")
    minimum_area = _configured_minimum_area()

    valid_by_id = {}
    for file_id in sorted(file_ids.dropna().astype(int).unique()):
        airfoil_path = os.path.join(geometry_dir, f"airfoil_{file_id}.dat")
        valid_by_id[file_id] = _airfoil_geometry_is_valid(
            airfoil_path,
            minimum_area=minimum_area,
            surface_cache=surface_cache,
        )

    return file_ids.map(valid_by_id).fillna(False).astype(bool)


def _airfoil_geometry_is_valid(
    airfoil_path: str,
    minimum_area: float | None = None,
    surface_cache: dict[str, AirfoilSurfaces | None] | None = None,
) -> bool:
    try:
        x_up, z_up, x_lo, z_lo = _load_airfoil_surfaces(airfoil_path, surface_cache)
        if not _surfaces_are_non_intersecting(x_up, z_up, x_lo, z_lo):
            return False
        if minimum_area is None:
            return True
        return _airfoil_enclosed_area(x_up, z_up, x_lo, z_lo) >= minimum_area
    except Exception:
        return False


def _minimum_area_is_configured() -> bool:
    return bool(
        os.environ.get("GEOM_MIN_AREA")
        or os.environ.get("GEOM_MIN_AREA_REFERENCE")
    )


def _configured_minimum_area() -> float | None:
    absolute = os.environ.get("GEOM_MIN_AREA")
    if absolute is not None:
        minimum_area = float(absolute)
    else:
        reference_path = os.environ.get("GEOM_MIN_AREA_REFERENCE")
        if not reference_path:
            return None
        ratio = float(os.environ.get("GEOM_MIN_AREA_RATIO", "1.0"))
        reference = _load_airfoil_coords(reference_path)
        x_up, z_up, x_lo, z_lo = _split_airfoil_surfaces(reference)
        minimum_area = ratio * _airfoil_enclosed_area(x_up, z_up, x_lo, z_lo)

    if not np.isfinite(minimum_area) or minimum_area <= 0.0:
        raise ValueError(f"Invalid configured minimum airfoil area: {minimum_area}")
    return float(minimum_area)


def _airfoil_enclosed_area(
    x_up: np.ndarray,
    z_up: np.ndarray,
    x_lo: np.ndarray,
    z_lo: np.ndarray,
) -> float:
    x_min = max(float(x_up[0]), float(x_lo[0]))
    x_max = min(float(x_up[-1]), float(x_lo[-1]))
    if x_max <= x_min:
        raise ValueError("Upper and lower surfaces have no common chord interval.")

    n_grid = max(3, int(os.environ.get("GEOM_AREA_N_GRID", "1001")))
    x_grid = np.linspace(x_min, x_max, n_grid)
    thickness = np.interp(x_grid, x_up, z_up) - np.interp(x_grid, x_lo, z_lo)
    if not np.all(np.isfinite(thickness)):
        raise ValueError("Non-finite thickness while calculating airfoil area.")
    return float(np.trapz(thickness, x_grid))


def _surfaces_are_non_intersecting(
    x_up: np.ndarray,
    z_up: np.ndarray,
    x_lo: np.ndarray,
    z_lo: np.ndarray,
) -> bool:
    x_min = max(float(x_up[0]), float(x_lo[0]))
    x_max = min(float(x_up[-1]), float(x_lo[-1]))

    x_eps = float(os.environ.get("GEOM_INTERSECTION_X_EPS", "1e-5"))
    x_min += x_eps
    x_max -= x_eps
    if x_max <= x_min:
        return False

    n_grid = int(os.environ.get("GEOM_INTERSECTION_N_GRID", "501"))
    x_grid = np.linspace(x_min, x_max, max(3, n_grid))
    thickness = np.interp(x_grid, x_up, z_up) - np.interp(x_grid, x_lo, z_lo)
    return bool(np.all(np.isfinite(thickness)) and np.min(thickness) > 0.0)


def _load_airfoil_coords(airfoil_path: str) -> np.ndarray:
    rows = []
    with open(airfoil_path, encoding="utf-8", errors="replace") as stream:
        for line in stream:
            fields = line.replace(",", " ").split()
            if len(fields) < 2:
                continue
            try:
                rows.append((float(fields[0]), float(fields[1])))
            except ValueError:
                continue

    coords = np.asarray(rows, dtype=float)
    coords = coords[np.all(np.isfinite(coords), axis=1)]
    if len(coords) < 5:
        raise ValueError(f"Airfoil file has too few coordinate rows: {airfoil_path}")

    # Lednicer files begin with upper/lower point counts and then store two
    # independent LE-to-TE blocks. Convert them to the TE-upper-LE-lower-TE
    # contour expected by the geometry checks.
    if np.all(coords[0] > 1.5):
        n_upper = int(round(coords[0, 0]))
        n_lower = int(round(coords[0, 1]))
        points = coords[1:]
        if n_upper < 2 or n_lower < 2 or len(points) < n_upper + n_lower:
            raise ValueError(f"Invalid Lednicer point counts in {airfoil_path}")
        upper = points[:n_upper]
        lower = points[n_upper:n_upper + n_lower]
        if float(np.mean(upper[:, 1])) < float(np.mean(lower[:, 1])):
            upper, lower = lower, upper
        upper_te_to_le = upper if upper[0, 0] > upper[-1, 0] else upper[::-1]
        lower_le_to_te = lower if lower[0, 0] < lower[-1, 0] else lower[::-1]
        coords = np.vstack([upper_te_to_le, lower_le_to_te[1:]])
    return coords


def _load_airfoil_surfaces(
    airfoil_path: str,
    surface_cache: dict[str, AirfoilSurfaces | None] | None = None,
) -> AirfoilSurfaces:
    if surface_cache is not None and airfoil_path in surface_cache:
        surfaces = surface_cache[airfoil_path]
        if surfaces is None:
            raise ValueError(f"Invalid airfoil geometry: {airfoil_path}")
        return surfaces

    try:
        surfaces = _split_airfoil_surfaces(_load_airfoil_coords(airfoil_path))
    except Exception:
        if surface_cache is not None:
            surface_cache[airfoil_path] = None
        raise
    if surface_cache is not None:
        surface_cache[airfoil_path] = surfaces
    return surfaces


def _split_airfoil_surfaces(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    le_idx = int(np.argmin(coords[:, 0]))
    if le_idx < 2 or le_idx > len(coords) - 3:
        raise ValueError("Could not split airfoil into upper/lower surfaces.")

    upper = coords[:le_idx + 1]
    lower = coords[le_idx:]

    x_up, z_up = _surface_by_increasing_x(upper[:, 0], upper[:, 1])
    x_lo, z_lo = _surface_by_increasing_x(lower[:, 0], lower[:, 1])
    if len(x_up) < 2 or len(x_lo) < 2:
        raise ValueError("Airfoil surface has too few unique x/c stations.")

    return x_up, z_up, x_lo, z_lo


def _surface_by_increasing_x(x: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(x)
    x_sorted = np.asarray(x, dtype=float)[order]
    z_sorted = np.asarray(z, dtype=float)[order]

    if np.all(np.diff(x_sorted) > 0.0):
        return x_sorted, z_sorted

    x_unique, starts, counts = np.unique(
        x_sorted, return_index=True, return_counts=True
    )
    z_unique = np.add.reduceat(z_sorted, starts) / counts
    return x_unique, z_unique


def _write_optimizer_costs(
    opt_dir: str,
    df_pop_ranked: pd.DataFrame,
    df_ranked: pd.DataFrame,
) -> None:
    """
    Persist the true ranked cost for optimizers with state across generations.

    PSO needs the actual minimization cost for each stable particle_id. Keeping
    this in a sidecar file avoids adding non-gene columns to population.csv.
    """
    airfoil_ids = pd.to_numeric(df_pop_ranked["Airfoil_idx"], errors="coerce").astype(int)
    cost_column = "selection_cost" if "selection_cost" in df_ranked else "cost"
    cost_by_airfoil = pd.Series(
        pd.to_numeric(df_ranked[cost_column], errors="coerce").to_numpy(),
        index=pd.to_numeric(df_ranked["Airfoil_idx"], errors="coerce").astype(int),
    )

    out = pd.DataFrame({"Airfoil_idx": airfoil_ids})
    if "particle_id" in df_pop_ranked.columns:
        out["particle_id"] = pd.to_numeric(
            df_pop_ranked["particle_id"],
            errors="coerce",
        ).astype(int)
    else:
        out["particle_id"] = out["Airfoil_idx"]
    out["cost"] = out["Airfoil_idx"].map(cost_by_airfoil)
    out.to_csv(os.path.join(opt_dir, "optimizer_costs.csv"), index=False)


def _de_selection(
    df_perf: pd.DataFrame,
    pop_size: int,
    coeffs: Sequence[float],
    aoa_min: float,
    aoa_max: float,
    geometry_dir: str | None = None,
    current_gen: int | None = None,
    surface_cache: dict[str, AirfoilSurfaces | None] | None = None,
) -> pd.DataFrame:
    ranked = _rank_rows(
        df_perf,
        coeffs,
        aoa_min,
        aoa_max,
        geometry_dir,
        current_gen,
        surface_cache=surface_cache,
    )

    targets = ranked[ranked["Airfoil_idx"] <= pop_size].set_index("Airfoil_idx")
    trials = ranked[ranked["Airfoil_idx"] > pop_size].set_index("Airfoil_idx")

    winners = []
    for i in range(1, pop_size + 1):
        trial_idx = i + pop_size
        has_target = i in targets.index
        has_trial = trial_idx in trials.index

        if not has_target and not has_trial:
            continue
        if not has_trial:
            winners.append(i)
            continue
        if not has_target:
            winners.append(trial_idx)
            continue

        comparison_column = (
            "selection_cost" if "selection_cost" in ranked.columns else "cost"
        )
        target_cost = float(targets.loc[i, comparison_column])
        trial_cost = float(trials.loc[trial_idx, comparison_column])
        winners.append(trial_idx if trial_cost < target_cost else i)

    return df_perf[df_perf["Airfoil_idx"].isin(winners)]


def _de_pop_selection(df_pop: pd.DataFrame, df_perf_winners: pd.DataFrame) -> pd.DataFrame:
    winner_ids = df_perf_winners["Airfoil_idx"].unique()
    df_pop_idx = df_pop.set_index("Airfoil_idx")

    rows = []
    for new_id, winner_id in enumerate(sorted(winner_ids), start=1):
        row = df_pop_idx.loc[winner_id].copy()
        row.name = new_id
        rows.append(row)

    result = pd.DataFrame(rows)
    result.index.name = "Airfoil_idx"
    return result.reset_index()


def _coefficients(
    coeff_cd: float | Sequence[float] | None,
    coeff_cd2: float | None,
) -> list[float]:
    if coeff_cd is None:
        n_points = int(os.environ.get("N_CD_POINTS", "1"))
        return [1.0] * n_points

    if isinstance(coeff_cd, Sequence) and not isinstance(coeff_cd, (str, bytes)):
        return [float(value) for value in coeff_cd]

    coeffs = [float(coeff_cd)]
    if coeff_cd2 is not None:
        coeffs.append(float(coeff_cd2))
    return coeffs


def _primary_cd_eval_dir(opt_dir: str) -> str:
    mach = os.environ.get("MACH_CD1") or os.environ.get("MACH_CD")
    if mach is None:
        return latest_eval_dir(opt_dir)
    return os.path.join(opt_dir, f"evaluation_mach_{float(mach):.2f}")


def _cd_name(df: pd.DataFrame, point_idx: int) -> str:
    if point_idx == 1:
        return "CD"

    mach = os.environ.get(f"MACH_CD{point_idx}")
    if mach is not None:
        mach_token = f"{float(mach):.2f}"
        normalized_expected = {
            f"cd_mach_{mach_token}",
            f"cdmach_{mach_token}",
            f"cd_mach{mach_token}",
        }
        for col in df.columns:
            normalized = str(col).strip().lower().replace(" ", "_")
            if normalized in normalized_expected:
                return col

    exact_candidates = (
        f"CD_mach{point_idx}",
        f"CD_Mach{point_idx}",
        f"CD_{point_idx}",
        f"CD{point_idx}",
    )
    for col in exact_candidates:
        if col in df.columns:
            return col

    extra_cd_cols = _extra_cd_names(df)
    extra_idx = point_idx - 2
    if 0 <= extra_idx < len(extra_cd_cols):
        return extra_cd_cols[extra_idx]

    raise RuntimeError(
        f"No Cd column found for objective point {point_idx}. Columns are: "
        + ", ".join(str(c) for c in df.columns)
    )


def _extra_cd_names(df: pd.DataFrame) -> list[str]:
    candidates = []
    for col in df.columns:
        name = str(col).strip()
        lower = name.lower()
        if lower in {"cd", "cost"}:
            continue
        if re.match(r"^cd($|[_\s-]*(mach|m|2)|2)", lower):
            candidates.append(col)

    if candidates:
        return candidates

    return []


def _second_cd_name(df: pd.DataFrame) -> str:
    return _cd_name(df, 2)


def _target_reached_name(df: pd.DataFrame, point_idx: int) -> str | None:
    if point_idx == 1:
        return "CL_target_reached" if "CL_target_reached" in df.columns else None

    mach = os.environ.get(f"MACH_CD{point_idx}")
    if mach is not None:
        mach_token = f"{float(mach):.2f}"
        expected = f"cl_target_reached_mach_{mach_token}"
        for col in df.columns:
            if str(col).strip().lower() == expected:
                return col

    candidates = (
        f"CL_target_reached_mach{point_idx}",
        f"CL_target_reached_{point_idx}",
    )
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _truthy(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})
