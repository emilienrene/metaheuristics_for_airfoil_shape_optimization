"""
Generate the initial population (generation 0) via Latin Hypercube Sampling.

Usage:
    python3 run_gen0.py <design_space_dir> <pop_size>
"""

from __future__ import annotations

import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from parameterizations import get_parameterization
from core.population   import lhs_sample, save_population, write_dat_files

output_dir = sys.argv[1]
pop_size   = int(sys.argv[2])

if not os.path.isdir(output_dir):
    raise RuntimeError(f"Design_space directory not found: {output_dir}")

method = get_parameterization()
print(f"Parameterization : {method.name}  ({method.n_params} genes)")
print(f"Output directory : {output_dir}")

samples = lhs_sample(pop_size, method.bounds)
print(f"LHS samples generated: {pop_size}")

include_reference = os.environ.get("INCLUDE_REFERENCE_IN_GEN0", "0").strip().lower()
if include_reference in {"1", "true", "yes", "y"}:
    if method.reference is None:
        raise RuntimeError(
            f"INCLUDE_REFERENCE_IN_GEN0 is enabled, but {method.name} has no reference genotype."
        )
    reference = np.asarray(method.reference, dtype=float)
    if reference.shape != (method.n_params,):
        raise RuntimeError(
            f"{method.name} reference has shape {reference.shape}; "
            f"expected ({method.n_params},)."
        )
    bounds = np.asarray(method.bounds, dtype=float)
    outside = (reference < bounds[:, 0]) | (reference > bounds[:, 1])
    if np.any(outside):
        names = getattr(method, "param_names", None)
        if names is None or len(names) != method.n_params:
            names = [f"parameter_{i}" for i in range(method.n_params)]
        details = "\n".join(
            f"  {names[i]}: reference={reference[i]:.16g}, "
            f"bounds=[{bounds[i, 0]:.16g}, {bounds[i, 1]:.16g}]"
            for i in np.flatnonzero(outside)
        )
        raise RuntimeError(
            f"{method.name} reference genotype lies outside its bounds:\n{details}"
        )
    samples[0] = reference
    print("Reference genotype inserted as airfoil_1.")

write_dat_files(
    genotypes  = samples,
    indices    = list(range(1, pop_size + 1)),
    output_dir = output_dir,
    decode_fn  = method.decode,
)

save_population(
    genotypes = samples,
    header    = method.header,
    path      = os.path.join(output_dir, "population.csv"),
)

print("Generation 0 complete.")
