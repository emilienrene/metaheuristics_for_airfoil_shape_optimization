"""
Genetic Algorithm optimizer.

Implements elitism + uniform crossover + per-gene mutation.
All hyperparameters are readable from environment variables so they can be
overridden at launch without touching the code.

Environment variables (with defaults):
    GA_ELITE_FRACTION   float   0.05   fraction of population kept as elites
    GA_PARENT_POOL      float   0.20   fraction eligible as parents
    GA_MUTATION_RATE    float   0.10   per-gene mutation probability
    GA_MUTATION_STD     float   0.15   mutation step size (fraction of span)
"""

from __future__ import annotations

import os
import numpy as np

from .base import Optimizer


def _env_float(key: str, default: float) -> float:
    return float(os.environ.get(key, default))


class GeneticAlgorithm(Optimizer):

    name = "ga"

    def __init__(self) -> None:
        self.elite_fraction = _env_float("GA_ELITE_FRACTION", 0.533)
        self.parent_pool    = _env_float("GA_PARENT_POOL",    0.533)
        self.mutation_rate  = _env_float("GA_MUTATION_RATE",  0.3)
        self.mutation_std   = _env_float("GA_MUTATION_STD",   0.15)
        seed = int(os.environ.get("STUDY_SEED", 0))
        self.seed: int | None = seed or None

    def step(
        self,
        population: np.ndarray,
        bounds: np.ndarray,
        mutation_mode: str,
        **kwargs,
    ) -> np.ndarray:
        generation = int(kwargs.get("generation", 0))
        rng = self._rng_for_generation(generation)

        pop_size, n_genes = population.shape
        elite_n  = max(1, int(self.elite_fraction * pop_size))
        parent_n = max(2, int(self.parent_pool    * pop_size))

        elites  = population[:elite_n]
        parents = population[:parent_n]

        children = []
        while len(children) < pop_size - elite_n:
            i1, i2 = rng.integers(0, parent_n, size=2)
            child = self._crossover(parents[i1], parents[i2], n_genes, rng)
            child = self._mutate(child, bounds, mutation_mode, rng)
            children.append(child)

        next_pop = np.vstack([elites, np.array(children)])
        return next_pop

    def _crossover(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        n_genes: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        mask = rng.random(n_genes) < 0.5
        return np.where(mask, p1, p2)

    def _mutate(
        self,
        child: np.ndarray,
        bounds: np.ndarray,
        mutation_mode: str,
        rng: np.random.Generator,
    ) -> np.ndarray:
        child = child.copy()
        for i in range(len(child)):
            if rng.random() < self.mutation_rate:
                if mutation_mode == "multiplicative":
                    child[i] *= 1 + rng.normal(0, self.mutation_std)
                else:
                    span = bounds[i, 1] - bounds[i, 0]
                    child[i] += rng.normal(0, self.mutation_std) * span
        return np.clip(child, bounds[:, 0], bounds[:, 1])

    def _rng_for_generation(self, generation: int) -> np.random.Generator:
        if self.seed is None:
            return np.random.default_rng()
        return np.random.default_rng(self.seed + int(generation))
