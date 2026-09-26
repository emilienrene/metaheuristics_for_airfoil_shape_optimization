"""
Differential Evolution optimizer - DE/best/2/bin.

Implements the strategy used by Carrigan et al. (2012):
    - Mutation   : v_i = x_best + F * ((x_r1 - x_r2) + (x_r3 - x_r4))
    - Crossover  : binomial with forced gene exchange
    - Selection  : greedy one-to-one, handled in core/fitness.py

step() returns a (2*pop_size, n_genes) array:
    rows 0        .. pop_size-1   current population  (targets)
    rows pop_size .. 2*pop_size-1 trial vectors       (candidates)

Row i and row pop_size+i are always a pair. Selection is deferred to
run_rank.py after rAIFoil has evaluated both sets.

Environment variables (with defaults):
    DE_F    float   0.8    mutation scale factor
    DE_CR   float   0.6    crossover rate

References:
    Carrigan, T. J., Dennis, B. H., Han, Z. X., & Wang, B. P. (2012).
    Aerodynamic Shape Optimization of a Vertical-Axis Wind Turbine Using
    Differential Evolution. ISRN Renewable Energy.
"""

from __future__ import annotations

import os
import numpy as np

from .base import Optimizer


def _env_float(key: str, default: float) -> float:
    return float(os.environ.get(key, default))


class DifferentialEvolution(Optimizer):

    name = "de"
    mutation_mode = "additive"

    def __init__(self) -> None:
        self.F   = _env_float("DE_F",  0.8)
        self.CR  = _env_float("DE_CR", 0.6)
        seed = int(os.environ.get("STUDY_SEED", 0))
        self.seed: int | None = seed or None

    def step(
        self,
        population: np.ndarray,
        bounds: np.ndarray,
        mutation_mode: str,
        **kwargs,
    ) -> np.ndarray:
        """
        Returns (2*pop_size, n_genes):
            rows 0        .. pop_size-1   targets  (current population)
            rows pop_size .. 2*pop_size-1 trials   (mutant + crossover)
        """
        generation = int(kwargs.get("generation", 0))
        rng = self._rng_for_generation(generation)
        pop_size, n_genes = population.shape

        trials = np.empty_like(population)
        best_idx = 0
        x_best = population[best_idx]

        for i in range(pop_size):

            candidates = [
                j for j in range(pop_size)
                if j != i and j != best_idx
            ]
            if len(candidates) < 4:
                raise ValueError(
                    "DE/best/2/bin requires at least 6 population members "
                    "so four random vectors can be distinct from the target "
                    "and best vector."
                )
            r1, r2, r3, r4 = rng.choice(candidates, size=4, replace=False)

            mutant = x_best + self.F * (
                (population[r1] - population[r2])
                + (population[r3] - population[r4])
            )

            mutant = np.clip(mutant, bounds[:, 0], bounds[:, 1])

            rnbr = int(rng.integers(0, n_genes))
            mask = rng.random(n_genes) <= self.CR
            mask[rnbr] = True
            trial = np.where(mask, mutant, population[i])

            trials[i] = trial

        # stack targets then trials: row i pairs with row pop_size+i
        return np.vstack([population, trials])

    def _rng_for_generation(self, generation: int) -> np.random.Generator:
        if self.seed is None:
            return np.random.default_rng()
        return np.random.default_rng(self.seed + int(generation))