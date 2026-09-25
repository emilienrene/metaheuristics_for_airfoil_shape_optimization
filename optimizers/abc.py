"""
Artificial Bee Colony optimizer.

This version follows the ABC setup used by Gabor et al. for morphing-airfoil
optimization, itself based on Karaboga and Basturk's ABC mechanics:

    v_ij = x_ij + SF * phi_ij * (x_ij - x_kj)
           + psi_ij * (x_best,j - x_ij)

where k != i, phi_ij is sampled uniformly from [-1, 1], and psi_ij is sampled
from [0, ABC_BEST_ATTRACTION]. The extra attraction term reflects the paper's
"attraction factor of the best solution"; setting ABC_BEST_ATTRACTION=0
recovers the usual ABC/MABC neighbourhood update.

The paper used 30 employed bees, 30 onlooker bees, 50 cycles, initial
modification rate MR=1.0, initial scaling factor SF=1.0, dynamic parameter
updates every 10 cycles, abandonment limit equal to the maximum cycles, and
best-solution stagnation monitoring over 20 cycles. In this generation-wise
pipeline, objective evaluations happen outside optimizer.step(), so greedy
same-cycle replacement and the final ALM-BFGS refinement are not reproduced
inside this optimizer.

Environment variables:
    ABC_MR                 float  1.0   initial per-gene modification rate
    ABC_MR_FINAL           float  0.2   final scheduled modification rate
    ABC_SF                 float  1.0   initial ABC scaling factor
    ABC_SF_FINAL           float  0.2   final scheduled scaling factor
    ABC_UPDATE_PERIOD      int    10    cycles between MR/SF schedule updates
    ABC_BEST_ATTRACTION    float  1.5   best-solution attraction factor
    ABC_EMPLOYED_FRACTION  float  0.5   employed/onlooker candidate split
    ABC_ELITISM            int    1     keep current best in next generation
    ABC_LIMIT              int    NUM_GEN cycles before scout replacement
    ABC_SCOUT_PERIOD       int    ABC_LIMIT override for scout period
    ABC_SCOUT_COUNT        int    1     random scouts at each scout period
    ABC_STAGNATION_LIMIT   int    20    paper's convergence monitor
    STUDY_SEED             int    0     random seed; 0 means unseeded
"""

from __future__ import annotations

import glob
import json
import os
import re
from typing import Any

import numpy as np

from .base import Optimizer


def _env_float(key: str, default: float) -> float:
    return float(os.environ.get(key, default))


def _env_int(key: str, default: int) -> int:
    return int(os.environ.get(key, default))


def _env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


def _latest_opt_dir() -> str:
    env_opt_dir = os.environ.get("OPT_DIR")
    if env_opt_dir:
        return os.path.abspath(env_opt_dir)

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dirs = glob.glob(os.path.join(base, "optimization_*"))
    if not dirs:
        raise RuntimeError("No optimization_* directories found.")
    return max(dirs, key=lambda p: int(re.search(r"optimization_(\d+)", p).group(1)))


def _state_path() -> str:
    return os.path.join(_latest_opt_dir(), "abc_state.json")


class ArtificialBeeColony(Optimizer):
    """
    Rank-driven ABC generator for the one-generation-at-a-time pipeline.
    """

    name = "abc"

    def __init__(self) -> None:
        num_gen = _env_int("NUM_GEN", 50)

        self.initial_modification_rate = _env_float("ABC_MR", 1.0)
        self.final_modification_rate = _env_float("ABC_MR_FINAL", 0.2)
        self.initial_scaling_factor = _env_float("ABC_SF", 1.0)
        self.final_scaling_factor = _env_float("ABC_SF_FINAL", 0.2)
        self.update_period = _env_int("ABC_UPDATE_PERIOD", 10)
        self.best_attraction = _env_float("ABC_BEST_ATTRACTION", 1.5)
        self.employed_fraction = _env_float("ABC_EMPLOYED_FRACTION", 0.5)
        self.elitism = _env_bool("ABC_ELITISM", True)
        self.limit = _env_int("ABC_LIMIT", num_gen)
        self.scout_period = _env_int("ABC_SCOUT_PERIOD", self.limit)
        self.scout_count = _env_int("ABC_SCOUT_COUNT", 1)
        self.stagnation_limit = _env_int("ABC_STAGNATION_LIMIT", 20)
        self.num_gen = num_gen

        seed = int(os.environ.get("STUDY_SEED", 0))
        self.seed: int | None = seed or None

    # -- state persistence -------------------------------------------------

    def _load_state(self) -> dict[str, Any]:
        path = _state_path()
        if not os.path.isfile(path):
            return {
                "generation": 0,
                "best_cost": float("inf"),
                "stagnant_cycles": 0,
            }
        with open(path) as f:
            raw = json.load(f)
        return {
            "generation": int(raw.get("generation", 0)),
            "best_cost": float(raw.get("best_cost", float("inf"))),
            "stagnant_cycles": int(raw.get("stagnant_cycles", 0)),
        }

    def _save_state(self, state: dict[str, Any]) -> None:
        path = _state_path()
        with open(path, "w") as f:
            json.dump(
                {
                    "generation": int(state["generation"]),
                    "best_cost": float(state["best_cost"]),
                    "stagnant_cycles": int(state["stagnant_cycles"]),
                },
                f,
            )

    # -- optimizer interface -----------------------------------------------

    def step(
        self,
        population: np.ndarray,
        bounds: np.ndarray,
        mutation_mode: str,
        particle_ids=None,
        fitness_values=None,
        **kwargs,
    ) -> np.ndarray:
        _ = mutation_mode, particle_ids

        population = np.asarray(population, dtype=float)
        bounds = np.asarray(bounds, dtype=float)
        pop_size, n_genes = population.shape

        lower = bounds[:, 0]
        upper = bounds[:, 1]
        span = upper - lower
        active_idx = np.flatnonzero(span > 0.0)

        if len(active_idx) == 0:
            return np.tile(lower, (pop_size, 1))

        state = self._load_state()
        generation = int(state["generation"]) + 1
        rng_generation = int(kwargs.get("generation", generation))
        rng = self._rng_for_generation(rng_generation)
        state = self._update_state_with_costs(state, fitness_values, generation)
        modification_rate, scaling_factor = self._scheduled_parameters(generation)
        best_idx = self._best_index(fitness_values, pop_size)
        best_source = population[best_idx].copy()

        next_pop: list[np.ndarray] = []
        if self.elitism:
            next_pop.append(np.clip(best_source.copy(), lower, upper))

        remaining = pop_size - len(next_pop)
        employed_n = int(round(self.employed_fraction * remaining))
        employed_n = min(max(0, employed_n), remaining)
        onlooker_n = remaining - employed_n

        for i in range(employed_n):
            source_idx = i % pop_size
            next_pop.append(
                self._neighbor(
                    population=population,
                    source_idx=source_idx,
                    lower=lower,
                    upper=upper,
                    active_idx=active_idx,
                    best_source=best_source,
                    modification_rate=modification_rate,
                    scaling_factor=scaling_factor,
                    rng=rng,
                )
            )

        probs = self._nectar_probabilities(fitness_values, pop_size)
        for _ in range(onlooker_n):
            source_idx = int(rng.choice(pop_size, p=probs))
            next_pop.append(
                self._neighbor(
                    population=population,
                    source_idx=source_idx,
                    lower=lower,
                    upper=upper,
                    active_idx=active_idx,
                    best_source=best_source,
                    modification_rate=modification_rate,
                    scaling_factor=scaling_factor,
                    rng=rng,
                )
            )

        out = np.vstack(next_pop) if next_pop else np.empty_like(population)
        out = self._apply_scouts(out, lower, upper, active_idx, generation, rng)

        state["generation"] = generation
        self._save_state(state)
        return np.clip(out[:, :n_genes], lower, upper)

    # -- ABC operators -----------------------------------------------------

    def _neighbor(
        self,
        population: np.ndarray,
        source_idx: int,
        lower: np.ndarray,
        upper: np.ndarray,
        active_idx: np.ndarray,
        best_source: np.ndarray,
        modification_rate: float,
        scaling_factor: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        pop_size = population.shape[0]
        if pop_size < 2:
            return np.clip(population[source_idx].copy(), lower, upper)

        partner_idx = int(rng.integers(0, pop_size - 1))
        if partner_idx >= source_idx:
            partner_idx += 1

        candidate = population[source_idx].copy()
        mask = rng.random(len(active_idx)) < modification_rate
        if not np.any(mask):
            mask[int(rng.integers(0, len(active_idx)))] = True

        dims = active_idx[mask]
        phi = rng.uniform(-1.0, 1.0, size=len(dims))
        psi = rng.uniform(0.0, self.best_attraction, size=len(dims))
        candidate[dims] = (
            population[source_idx, dims]
            + scaling_factor
            * phi
            * (population[source_idx, dims] - population[partner_idx, dims])
            + psi * (best_source[dims] - population[source_idx, dims])
        )
        return np.clip(candidate, lower, upper)

    def _apply_scouts(
        self,
        population: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        active_idx: np.ndarray,
        generation: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        scout_period = self.scout_period
        if scout_period <= 0:
            scout_period = max(1, self.limit)

        if generation % scout_period != 0:
            return population

        scout_n = min(max(0, self.scout_count), population.shape[0])
        if scout_n == 0:
            return population

        out = population.copy()
        for row in range(population.shape[0] - scout_n, population.shape[0]):
            out[row] = self._random_food_source(lower, upper, active_idx, rng)
        return out

    def _random_food_source(
        self,
        lower: np.ndarray,
        upper: np.ndarray,
        active_idx: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        source = lower.copy()
        source[active_idx] = rng.uniform(lower[active_idx], upper[active_idx])
        return source

    def _scheduled_parameters(self, generation: int) -> tuple[float, float]:
        if self.update_period <= 0:
            return self.initial_modification_rate, self.initial_scaling_factor

        scheduled_generation = ((max(generation, 1) - 1) // self.update_period) * self.update_period
        denominator = max(self.num_gen - self.update_period, 1)
        progress = min(max(scheduled_generation / denominator, 0.0), 1.0)

        modification_rate = (
            self.initial_modification_rate
            + progress * (self.final_modification_rate - self.initial_modification_rate)
        )
        scaling_factor = (
            self.initial_scaling_factor
            + progress * (self.final_scaling_factor - self.initial_scaling_factor)
        )
        modification_rate = float(np.clip(modification_rate, 0.0, 1.0))
        scaling_factor = max(0.0, float(scaling_factor))
        return modification_rate, scaling_factor

    def _update_state_with_costs(
        self,
        state: dict[str, Any],
        fitness_values,
        generation: int,
    ) -> dict[str, Any]:
        costs = self._costs_for_minimization(fitness_values)
        if costs is None:
            state["generation"] = generation
            return state

        best_cost = float(np.min(costs))
        previous_best = float(state.get("best_cost", float("inf")))
        if best_cost < previous_best:
            state["best_cost"] = best_cost
            state["stagnant_cycles"] = 0
        else:
            state["stagnant_cycles"] = int(state.get("stagnant_cycles", 0)) + 1
        state["generation"] = generation
        return state

    @staticmethod
    def _costs_for_minimization(fitness_values) -> np.ndarray | None:
        if fitness_values is None:
            return None

        costs = np.asarray(fitness_values, dtype=float).copy()
        if costs.ndim != 1 or costs.size == 0:
            return None
        costs[~np.isfinite(costs)] = np.inf
        if not np.any(np.isfinite(costs)):
            return None
        return costs

    @classmethod
    def _best_index(cls, fitness_values, pop_size: int) -> int:
        costs = cls._costs_for_minimization(fitness_values)
        if costs is None or costs.shape[0] != pop_size:
            return 0
        return int(np.argmin(costs))

    @classmethod
    def _nectar_probabilities(cls, fitness_values, pop_size: int) -> np.ndarray:
        costs = cls._costs_for_minimization(fitness_values)
        if costs is None or costs.shape[0] != pop_size:
            return cls._rank_probabilities(pop_size)

        finite = np.isfinite(costs)
        if not np.any(finite):
            return cls._rank_probabilities(pop_size)

        worst = float(np.max(costs[finite]))
        shifted = worst - costs
        shifted[~finite] = 0.0
        nectar = shifted + 1.0e-12
        total = float(np.sum(nectar))
        if total <= 0.0 or not np.isfinite(total):
            return cls._rank_probabilities(pop_size)
        return nectar / total

    @staticmethod
    def _rank_probabilities(pop_size: int) -> np.ndarray:
        scores = np.arange(pop_size, 0, -1, dtype=float)
        return scores / np.sum(scores)

    def _rng_for_generation(self, generation: int) -> np.random.Generator:
        if self.seed is None:
            return np.random.default_rng()
        return np.random.default_rng(self.seed + int(generation))
