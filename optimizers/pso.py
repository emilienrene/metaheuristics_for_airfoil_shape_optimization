"""
Particle Swarm Optimisation optimizer.

This implementation is adapted for the one-generation-at-a-time pipeline:

    - particle_id is stable across generations and is used to reconnect each
      evaluated particle to its velocity and personal best.
    - optimizer_costs.csv supplies the true minimization cost from ranking.
    - the global-best particle is copied into the next population by default,
      so the best-cost convergence curve is best-so-far instead of forgetting
      a good design for one generation.

Environment variables:
    PSO_OMEGA_MAX   float  0.9   inertia weight at generation 0
    PSO_OMEGA_MIN   float  0.4   inertia weight near final generation
    PSO_C1          float  2.0   cognitive coefficient
    PSO_C2          float  2.0   social coefficient
    PSO_ELITISM     int    1     keep the known global best in the swarm
    NUM_GEN         int    20    total generations for omega schedule
    STUDY_SEED      int    0     random seed; 0 means unseeded
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
    return os.path.join(_latest_opt_dir(), "pso_state.json")


class PSO(Optimizer):
    """
    Global-best PSO for minimization with persistent particle memory.
    """

    name = "pso"

    def __init__(self) -> None:
        self.omega_max = _env_float("PSO_OMEGA_MAX", 0.9)
        self.omega_min = _env_float("PSO_OMEGA_MIN", 0.4)
        self.c1 = _env_float("PSO_C1", 2.0)
        self.c2 = _env_float("PSO_C2", 2.0)
        self.elitism = _env_bool("PSO_ELITISM", True)
        self.num_gen = _env_int("NUM_GEN", 20)

        seed = int(os.environ.get("STUDY_SEED", 0))
        self.seed: int | None = seed or None

    # -- state persistence -------------------------------------------------

    def _load_state(self) -> dict[str, Any] | None:
        path = _state_path()
        if not os.path.isfile(path):
            return None

        with open(path) as f:
            raw = json.load(f)

        particle_ids = [int(pid) for pid in raw["particle_ids"]]
        pbest_costs = raw.get("personal_best_costs")
        if pbest_costs is None:
            # Older PSO state used synthetic rank fitness. Force refresh from
            # true optimizer_costs.csv on the next step.
            pbest_costs = [float("inf")] * len(particle_ids)

        return {
            "particle_ids": particle_ids,
            "positions": np.asarray(raw["positions"], dtype=float),
            "velocities": np.asarray(raw["velocities"], dtype=float),
            "personal_best": np.asarray(raw["personal_best"], dtype=float),
            "personal_best_costs": np.asarray(pbest_costs, dtype=float),
            "generation": int(raw.get("generation", 0)),
        }

    def _save_state(self, state: dict[str, Any]) -> None:
        path = _state_path()
        with open(path, "w") as f:
            json.dump(
                {
                    "particle_ids": [int(pid) for pid in state["particle_ids"]],
                    "positions": np.asarray(state["positions"], dtype=float).tolist(),
                    "velocities": np.asarray(state["velocities"], dtype=float).tolist(),
                    "personal_best": np.asarray(state["personal_best"], dtype=float).tolist(),
                    "personal_best_costs": np.asarray(
                        state["personal_best_costs"],
                        dtype=float,
                    ).tolist(),
                    "generation": int(state["generation"]),
                    "global_best": np.asarray(state["global_best"], dtype=float).tolist(),
                    "global_best_cost": float(state["global_best_cost"]),
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
        _ = mutation_mode
        rng_generation = kwargs.get("generation")

        population = np.asarray(population, dtype=float)
        bounds = np.asarray(bounds, dtype=float)
        pop_size, n_genes = population.shape

        if particle_ids is None:
            particle_ids = np.arange(1, pop_size + 1, dtype=int)
        else:
            particle_ids = np.asarray(particle_ids, dtype=int)

        lower = bounds[:, 0]
        upper = bounds[:, 1]
        span = upper - lower
        v_max = span

        current_costs = self._costs_for_minimization(fitness_values, pop_size)
        state = self._load_state()

        if state is None or not self._state_matches_particles(state, particle_ids):
            generation = 0
            rng = self._rng_for_generation(
                int(rng_generation) if rng_generation is not None else generation
            )
            velocities = rng.uniform(-v_max, v_max, size=(pop_size, n_genes))
            personal_best = population.copy()
            personal_best_costs = current_costs.copy()
        else:
            generation = int(state["generation"])
            order = self._state_order(state, particle_ids)
            velocities = state["velocities"][order]
            personal_best = state["personal_best"][order]
            personal_best_costs = state["personal_best_costs"][order]

            improved = current_costs < personal_best_costs
            personal_best[improved] = population[improved]
            personal_best_costs[improved] = current_costs[improved]
            rng = self._rng_for_generation(
                int(rng_generation) if rng_generation is not None else generation
            )

        best_idx = self._best_index(personal_best_costs)
        global_best = personal_best[best_idx].copy()
        global_best_cost = float(personal_best_costs[best_idx])

        omega = self._omega(generation)
        r1 = rng.random((pop_size, n_genes))
        r2 = rng.random((pop_size, n_genes))

        new_velocities = (
            omega * velocities
            + self.c1 * r1 * (personal_best - population)
            + self.c2 * r2 * (global_best - population)
        )
        new_velocities = np.clip(new_velocities, -v_max, v_max)

        new_positions = np.clip(population + new_velocities, lower, upper)

        if self.elitism:
            new_positions[best_idx] = global_best
            new_velocities[best_idx] = 0.0

        self._save_state(
            {
                "particle_ids": particle_ids.tolist(),
                "positions": new_positions,
                "velocities": new_velocities,
                "personal_best": personal_best,
                "personal_best_costs": personal_best_costs,
                "generation": generation + 1,
                "global_best": global_best,
                "global_best_cost": global_best_cost,
            }
        )

        return new_positions

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _costs_for_minimization(fitness_values, pop_size: int) -> np.ndarray:
        if fitness_values is None:
            # Fallback for older pipelines: population is sorted best-first, so
            # use a synthetic minimization cost based on rank.
            return np.arange(pop_size, dtype=float)

        costs = np.asarray(fitness_values, dtype=float)
        if costs.shape[0] != pop_size:
            return np.arange(pop_size, dtype=float)
        costs = costs.copy()
        costs[~np.isfinite(costs)] = np.inf
        return costs

    @staticmethod
    def _state_matches_particles(state: dict[str, Any], particle_ids: np.ndarray) -> bool:
        state_ids = {int(pid) for pid in state["particle_ids"]}
        return all(int(pid) in state_ids for pid in particle_ids)

    @staticmethod
    def _state_order(state: dict[str, Any], particle_ids: np.ndarray) -> np.ndarray:
        id_to_idx = {int(pid): i for i, pid in enumerate(state["particle_ids"])}
        return np.array([id_to_idx[int(pid)] for pid in particle_ids], dtype=int)

    @staticmethod
    def _best_index(costs: np.ndarray) -> int:
        finite = np.isfinite(costs)
        if not np.any(finite):
            return 0
        finite_indices = np.flatnonzero(finite)
        return int(finite_indices[np.argmin(costs[finite])])

    def _omega(self, generation: int) -> float:
        progress = generation / max(self.num_gen, 1)
        progress = min(max(progress, 0.0), 1.0)
        return self.omega_max - (self.omega_max - self.omega_min) * progress

    def _rng_for_generation(self, generation: int) -> np.random.Generator:
        if self.seed is None:
            return np.random.default_rng()
        return np.random.default_rng(self.seed + int(generation))
