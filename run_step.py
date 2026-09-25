"""
Advance the optimization by one generation.

For GA/PSO: returns pop_size new genotypes, decoded to generation_N/.
For DE:     returns 2*pop_size rows (targets + trials), both decoded.
            Selection happens in run_rank.py after evaluation.

Usage:
    python3 run_step.py
"""

from __future__ import annotations

import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from parameterizations  import get_parameterization
from optimizers         import get_optimizer
from core.paths         import latest_opt_dir, design_space_dir, next_gen_index, generation_dir
from core.population    import load_population_with_ids, save_population, write_dat_files

method    = get_parameterization()
optimizer = get_optimizer()
is_de     = optimizer.name == "de"
is_pso    = optimizer.name == "pso"

print(f"Parameterization : {method.name}  ({method.n_params} genes)")
print(f"Optimizer        : {optimizer.name}")

opt    = latest_opt_dir()
ds_dir = design_space_dir(opt)
pop_path = os.path.join(ds_dir, "population.csv")

population, ids, gene_names = load_population_with_ids(pop_path)

if population.shape[1] != method.n_params:
    raise ValueError(
        f"population.csv has {population.shape[1]} gene columns but "
        f"'{method.name}' expects {method.n_params}."
    )

def load_optimizer_costs(opt_dir: str, particle_ids: np.ndarray) -> np.ndarray | None:
    path = os.path.join(opt_dir, "optimizer_costs.csv")
    if not os.path.isfile(path):
        return None

    df = pd.read_csv(path)
    if "cost" not in df.columns:
        return None

    if "particle_id" in df.columns:
        key_col = "particle_id"
        keys = particle_ids
    elif "Airfoil_idx" in df.columns:
        key_col = "Airfoil_idx"
        keys = np.arange(1, len(particle_ids) + 1, dtype=int)
    else:
        return None

    costs_by_id = {
        int(row[key_col]): float(row["cost"])
        for _, row in df.iterrows()
        if pd.notna(row[key_col]) and pd.notna(row["cost"])
    }
    values = np.array([costs_by_id.get(int(pid), np.nan) for pid in keys], dtype=float)
    return None if np.all(np.isnan(values)) else values


fitness_values = load_optimizer_costs(opt, ids)

next_pop = optimizer.step(
    population    = population,
    bounds        = method.bounds,
    mutation_mode = method.mutation_mode,
    particle_ids  = ids,
    fitness_values = fitness_values,
)

pop_size = len(population)

if is_de:
    target_ids = np.arange(1, pop_size + 1)
    trial_ids  = np.arange(pop_size + 1, 2 * pop_size + 1)
    dat_indices = list(target_ids) + list(trial_ids)
    output_particle_ids = np.array(dat_indices, dtype=int)
elif is_pso and len(next_pop) == len(ids):
    dat_indices = list(range(1, len(next_pop) + 1))
    output_particle_ids = ids
else:
    dat_indices = list(range(1, len(next_pop) + 1))
    output_particle_ids = np.array(dat_indices, dtype=int)

# decode phenotypes into the new generation folder
gen_idx = next_gen_index(opt)
gen_dir = generation_dir(gen_idx, opt)
indices = dat_indices

print(f"Decoding {len(next_pop)} airfoils -> {gen_dir}")
write_dat_files(
    genotypes  = next_pop,
    indices    = indices,
    output_dir = gen_dir,
    decode_fn  = method.decode,
)

# save population.csv with combined genotypes
save_population(next_pop, method.header, pop_path, particle_ids=output_particle_ids)
print("population.csv updated.")
