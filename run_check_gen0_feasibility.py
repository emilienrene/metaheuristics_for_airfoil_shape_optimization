"""Check an evaluated LHS population using the normal ranking criteria."""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pandas as pd

from core.fitness import (
    _dae11_constraint_epsilon,
    _dae11_constraint_mode,
    _dae11_constraints_enabled,
    _rank_rows,
)


def main() -> int:
    design_space, combined_csv, threshold_raw = sys.argv[1:4]
    threshold = float(threshold_raw)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("GEN0_MIN_FEASIBLE_FRACTION must be between 0 and 1.")

    population = pd.read_csv(os.path.join(design_space, "population.csv"))
    ranked = _rank_rows(
        pd.read_csv(combined_csv),
        [1.0] * int(os.environ.get("N_CD_POINTS", "1")),
        float(os.environ["AOA_MIN"]),
        float(os.environ["AOA_MAX"]),
        design_space,
        current_gen=1,
    )
    if "dae11_constraint_feasible" in ranked:
        feasible = ranked["dae11_constraint_feasible"].fillna(False).astype(bool)
    else:
        feasible = ranked["evaluation_valid"].fillna(False).astype(bool)
        if _dae11_constraints_enabled():
            tolerance = (
                _dae11_constraint_epsilon(1)
                if _dae11_constraint_mode() == "hierarchical"
                else float(os.environ.get("DRELA_DAE11_FEASIBILITY_TOL", "1.0"))
            )
            violation = pd.to_numeric(
                ranked["dae11_constraint_violation"], errors="coerce"
            )
            feasible &= np.isfinite(violation) & (violation <= tolerance)

    count = int(feasible.sum())
    total = len(population)
    if total == 0:
        raise ValueError("Generation-0 population is empty.")
    required = math.ceil(threshold * total)
    print(
        f"Gen0 LHS feasible: {count}/{total} "
        f"({100.0 * count / total:.1f}%; target {100.0 * threshold:g}%)",
        flush=True,
    )
    return 0 if count >= required else 10


if __name__ == "__main__":
    sys.exit(main())
