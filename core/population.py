"""
Population I/O and sampling.

Handles Latin Hypercube Sampling for generation 0, reading and writing
population CSV files, and writing airfoil .dat geometry files.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import os

def lhs_sample(n_samples: int, bounds: np.ndarray) -> np.ndarray:
    """
    Latin Hypercube Sample of shape (n_samples, n_params) scaled to bounds.
    """
    seed = os.environ.get("STUDY_SEED")
    if seed is not None:
        np.random.seed(int(seed))
    n_params = bounds.shape[0]
    lhs = np.zeros((n_samples, n_params))
    for j in range(n_params):
        perm = np.random.permutation(n_samples)
        lhs[:, j] = (perm + np.random.rand(n_samples)) / n_samples
    return bounds[:, 0] + lhs * (bounds[:, 1] - bounds[:, 0])


def save_population(genotypes, header, path, particle_ids=None):
    n = len(genotypes)
    if particle_ids is None:
        particle_ids = np.arange(1, n + 1)
    table = np.hstack([
        np.arange(1, n + 1).reshape(-1, 1),   # Airfoil_idx
        particle_ids.reshape(-1, 1),            # particle_id
        genotypes,
    ])
    open(path, "w").close()
    np.savetxt(path, table, delimiter=",", fmt="%.8f",
               header="Airfoil_idx,particle_id," + ",".join(header.split(",")[1:]),
               comments="")

def load_population(path: str) -> tuple[np.ndarray, list[str]]:
    """
    Load a population CSV.
    Returns (genotypes, gene_names) where genotypes is (pop_size, n_genes).
    """
    df = pd.read_csv(path)
    gene_cols = [c for c in df.columns if c not in ("Airfoil_idx", "particle_id")]
    return df[gene_cols].to_numpy(dtype=float), gene_cols

def load_population_with_ids(path):
    df = pd.read_csv(path)
    gene_cols = [c for c in df.columns if c not in ("Airfoil_idx", "particle_id")]
    ids = df["particle_id"].to_numpy(dtype=int)
    genes = df[gene_cols].to_numpy(dtype=float)
    return genes, ids, gene_cols

def load_population_df(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def write_dat_files(
    genotypes: np.ndarray,
    indices: list[int],
    output_dir: str,
    decode_fn,
    log_every: int = 50,
) -> None:
    """
    Decode each genotype and write a .dat coordinate file.

    Parameters
    ----------
    genotypes   : (n, n_genes) array
    indices     : airfoil index for each row (used in filename)
    output_dir  : directory to write files into
    decode_fn   : callable(params) -> (M, 2) array of [X, Z]
    """
    os.makedirs(output_dir, exist_ok=True)
    n = len(genotypes)
    for i, (params, idx) in enumerate(zip(genotypes, indices)):
        coords = decode_fn(params)
        path = os.path.join(output_dir, f"airfoil_{idx}.dat")
        np.savetxt(path, coords, fmt="%.6f", header="X Z", comments="")
        if i % log_every == 0:
            print(f"  [{i+1}/{n}] airfoil_{idx}.dat")
