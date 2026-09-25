"""
Covariance Matrix Adaptation Evolution Strategy optimizer.

This implements the core "pure CMA-ES" update from Hansen's CMA-ES tutorial:
weighted recombination, cumulation for the covariance and step-size evolution
paths, rank-one and rank-mu covariance adaptation, and cumulative step-size
adaptation.

The optimizer is adapted to this pipeline in three practical ways:

1. The pipeline minimizes cost and passes the population already sorted
   best-first. CMA-ES only needs the ranking, so the first rows are treated as
   the selected parents.
2. Every generation is a new Python process. The strategy state and the
   sampled mutation vectors are persisted in the latest optimization directory
   as ``cmaes_state.json``.
3. Airfoil genes can have very different scales and some fixed bounds. CMA-ES
   runs internally on active genes normalized to [0, 1]; fixed genes are kept
   at their bound values when offspring are decoded back to genotype space.

Environment variables:
    CMAES_SIGMA          float  0.12   initial normalized step-size
    CMAES_MU             int    0      selected parents; 0 means lambda // 2
    CMAES_CM             float  1.0    mean learning rate
    CMAES_ELITISM        int    1      copy current best into next generation
    CMAES_MAX_RESAMPLE   int    100    rejection attempts for box feasibility
    CMAES_MIN_SIGMA      float  1e-12  lower safety clamp for sigma
    CMAES_MAX_SIGMA      float  2.0    upper safety clamp for sigma
    STUDY_SEED           int    0      random seed; 0 means unseeded
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
    return os.path.join(_latest_opt_dir(), "cmaes_state.json")


class CMAES(Optimizer):
    """
    Rank-based CMA-ES with JSON state persistence between pipeline calls.
    """

    name = "cmaes"

    def __init__(self) -> None:
        self.initial_sigma = _env_float("CMAES_SIGMA", 0.12)
        self.mu_override = _env_int("CMAES_MU", 0)
        self.cm = _env_float("CMAES_CM", 1.0)
        self.elitism = _env_bool("CMAES_ELITISM", True)
        self.max_resample = _env_int("CMAES_MAX_RESAMPLE", 100)
        self.min_sigma = _env_float("CMAES_MIN_SIGMA", 1e-12)
        self.max_sigma = _env_float("CMAES_MAX_SIGMA", 2.0)

        seed = int(os.environ.get("STUDY_SEED", 0))
        self.seed: int | None = seed or None

    # -- state persistence -------------------------------------------------

    def _load_state(self) -> dict[str, Any] | None:
        path = _state_path()
        if not os.path.isfile(path):
            return None
        with open(path) as f:
            raw = json.load(f)

        state: dict[str, Any] = {
            "n": int(raw["n"]),
            "lambda": int(raw["lambda"]),
            "generation": int(raw["generation"]),
            "active_idx": [int(i) for i in raw["active_idx"]],
            "offspring_ids": [int(i) for i in raw["offspring_ids"]],
            "mean": np.array(raw["mean"], dtype=float),
            "sigma": float(raw["sigma"]),
            "C": np.array(raw["C"], dtype=float),
            "B": np.array(raw["B"], dtype=float),
            "D": np.array(raw["D"], dtype=float),
            "pc": np.array(raw["pc"], dtype=float),
            "ps": np.array(raw["ps"], dtype=float),
            "arz": np.array(raw["arz"], dtype=float),
            "ary": np.array(raw["ary"], dtype=float),
        }
        return state

    def _save_state(self, state: dict[str, Any]) -> None:
        path = _state_path()
        payload = {
            "n": int(state["n"]),
            "lambda": int(state["lambda"]),
            "generation": int(state["generation"]),
            "active_idx": [int(i) for i in state["active_idx"]],
            "offspring_ids": [int(i) for i in state["offspring_ids"]],
            "mean": np.asarray(state["mean"], dtype=float).tolist(),
            "sigma": float(state["sigma"]),
            "C": np.asarray(state["C"], dtype=float).tolist(),
            "B": np.asarray(state["B"], dtype=float).tolist(),
            "D": np.asarray(state["D"], dtype=float).tolist(),
            "pc": np.asarray(state["pc"], dtype=float).tolist(),
            "ps": np.asarray(state["ps"], dtype=float).tolist(),
            "arz": np.asarray(state["arz"], dtype=float).tolist(),
            "ary": np.asarray(state["ary"], dtype=float).tolist(),
        }
        with open(path, "w") as f:
            json.dump(payload, f)

    # -- optimizer interface -----------------------------------------------

    def step(
        self,
        population: np.ndarray,
        bounds: np.ndarray,
        mutation_mode: str,
        particle_ids=None,
        **kwargs,
    ) -> np.ndarray:
        _ = mutation_mode

        population = np.asarray(population, dtype=float)
        bounds = np.asarray(bounds, dtype=float)
        pop_size, n_genes = population.shape

        lower = bounds[:, 0]
        upper = bounds[:, 1]
        span = upper - lower
        active_idx = np.flatnonzero(span > 0.0)

        if len(active_idx) == 0:
            return np.tile(lower, (pop_size, 1))

        if particle_ids is None:
            particle_ids = np.arange(1, pop_size + 1, dtype=int)
        else:
            particle_ids = np.asarray(particle_ids, dtype=int)

        norm_pop = self._normalize(population, lower, span, active_idx)
        params = self._strategy_parameters(pop_size, len(active_idx))

        state = self._load_state()
        if self._state_matches(state, pop_size, active_idx):
            updated = self._tell(norm_pop, particle_ids, state, params)
            if updated is None:
                updated = self._initial_state(norm_pop, params, active_idx)
        else:
            updated = self._initial_state(norm_pop, params, active_idx)

        elite = norm_pop[0] if self.elitism else None
        rng_generation = int(kwargs.get("generation", int(updated["generation"])))
        next_norm, sampled_state = self._ask(
            updated,
            pop_size,
            active_idx,
            elite,
            rng_generation,
        )
        self._save_state(sampled_state)
        return self._denormalize(next_norm, lower, span, active_idx, n_genes)

    # -- CMA-ES internals ---------------------------------------------------

    def _strategy_parameters(self, pop_size: int, n: int) -> dict[str, Any]:
        if self.mu_override > 0:
            mu = min(pop_size, self.mu_override)
        else:
            mu = max(1, pop_size // 2)

        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mueff = float(np.sum(weights) ** 2 / np.sum(weights ** 2))

        cc = (4.0 + mueff / n) / (n + 4.0 + 2.0 * mueff / n)
        cs = (mueff + 2.0) / (n + mueff + 5.0)
        c1 = 2.0 / ((n + 1.3) ** 2 + mueff)
        cmu_num = 2.0 * (mueff - 2.0 + 1.0 / mueff)
        cmu_den = (n + 2.0) ** 2 + mueff
        cmu = min(1.0 - c1, cmu_num / cmu_den)
        cmu = max(0.0, cmu)
        damps = 1.0 + 2.0 * max(0.0, np.sqrt((mueff - 1.0) / (n + 1.0)) - 1.0) + cs
        chi_n = np.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n))

        return {
            "mu": mu,
            "weights": weights,
            "mueff": mueff,
            "cc": cc,
            "cs": cs,
            "c1": c1,
            "cmu": cmu,
            "damps": damps,
            "chi_n": chi_n,
        }

    def _initial_state(
        self,
        norm_pop: np.ndarray,
        params: dict[str, Any],
        active_idx: np.ndarray,
    ) -> dict[str, Any]:
        n = len(active_idx)
        mu = params["mu"]
        weights = params["weights"]
        mean = weights @ norm_pop[:mu]
        mean = np.clip(mean, 0.0, 1.0)

        return {
            "n": n,
            "lambda": int(norm_pop.shape[0]),
            "generation": 0,
            "active_idx": [int(i) for i in active_idx],
            "offspring_ids": list(range(1, norm_pop.shape[0] + 1)),
            "mean": mean,
            "sigma": float(np.clip(self.initial_sigma, self.min_sigma, self.max_sigma)),
            "C": np.eye(n),
            "B": np.eye(n),
            "D": np.ones(n),
            "pc": np.zeros(n),
            "ps": np.zeros(n),
            "arz": np.zeros((norm_pop.shape[0], n)),
            "ary": np.zeros((norm_pop.shape[0], n)),
        }

    def _tell(
        self,
        norm_pop: np.ndarray,
        particle_ids: np.ndarray,
        state: dict[str, Any],
        params: dict[str, Any],
    ) -> dict[str, Any] | None:
        id_to_idx = {pid: i for i, pid in enumerate(state["offspring_ids"])}
        if any(int(pid) not in id_to_idx for pid in particle_ids):
            return None

        order = np.array([id_to_idx[int(pid)] for pid in particle_ids], dtype=int)

        mu = params["mu"]
        weights = params["weights"]
        mueff = params["mueff"]
        cc = params["cc"]
        cs = params["cs"]
        c1 = params["c1"]
        cmu = params["cmu"]
        damps = params["damps"]
        chi_n = params["chi_n"]

        mean_old = np.asarray(state["mean"], dtype=float)
        sigma_old = float(state["sigma"])
        C_old = np.asarray(state["C"], dtype=float)
        B = np.asarray(state["B"], dtype=float)
        D = np.asarray(state["D"], dtype=float)
        pc_old = np.asarray(state["pc"], dtype=float)
        ps_old = np.asarray(state["ps"], dtype=float)

        ranked_y = np.asarray(state["ary"], dtype=float)[order]
        ranked_z = np.asarray(state["arz"], dtype=float)[order]

        selected_x = norm_pop[:mu]
        selected_y = ranked_y[:mu]
        selected_z = ranked_z[:mu]

        mean = (1.0 - self.cm) * mean_old + self.cm * (weights @ selected_x)
        mean = np.clip(mean, 0.0, 1.0)

        ymean = (mean - mean_old) / max(sigma_old, self.min_sigma)
        zmean = weights @ selected_z

        ps = (1.0 - cs) * ps_old + np.sqrt(cs * (2.0 - cs) * mueff) * (B @ zmean)

        generation = int(state["generation"]) + 1
        hsig_den = np.sqrt(1.0 - (1.0 - cs) ** (2.0 * generation)) * chi_n
        hsig = (np.linalg.norm(ps) / max(hsig_den, 1e-30)) < (1.4 + 2.0 / (state["n"] + 1.0))

        pc = (1.0 - cc) * pc_old
        if hsig:
            pc = pc + np.sqrt(cc * (2.0 - cc) * mueff) * ymean

        rank_mu = np.zeros_like(C_old)
        for w, y in zip(weights, selected_y):
            rank_mu += w * np.outer(y, y)

        C = (
            (1.0 - c1 - cmu) * C_old
            + c1 * (np.outer(pc, pc) + (1.0 - float(hsig)) * cc * (2.0 - cc) * C_old)
            + cmu * rank_mu
        )
        C = self._symmetrize(C)

        sigma = sigma_old * np.exp((cs / damps) * (np.linalg.norm(ps) / chi_n - 1.0))
        if not np.isfinite(sigma):
            sigma = self.initial_sigma
        sigma = float(np.clip(sigma, self.min_sigma, self.max_sigma))

        C, B, D = self._eigendecompose(C)

        return {
            "n": int(state["n"]),
            "lambda": int(state["lambda"]),
            "generation": generation,
            "active_idx": state["active_idx"],
            "offspring_ids": state["offspring_ids"],
            "mean": mean,
            "sigma": sigma,
            "C": C,
            "B": B,
            "D": D,
            "pc": pc,
            "ps": ps,
            "arz": state["arz"],
            "ary": state["ary"],
        }

    def _ask(
        self,
        state: dict[str, Any],
        pop_size: int,
        active_idx: np.ndarray,
        elite: np.ndarray | None = None,
        rng_generation: int | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        n = len(active_idx)
        mean = np.asarray(state["mean"], dtype=float)
        sigma = float(state["sigma"])
        B = np.asarray(state["B"], dtype=float)
        D = np.asarray(state["D"], dtype=float)
        if rng_generation is None:
            rng_generation = int(state["generation"])
        rng = self._rng_for_generation(rng_generation)

        arz = np.empty((pop_size, n))
        ary = np.empty((pop_size, n))
        arx = np.empty((pop_size, n))

        start = 0
        if elite is not None:
            elite = np.clip(np.asarray(elite, dtype=float), 0.0, 1.0)
            y = (elite - mean) / max(sigma, self.min_sigma)
            z = self._z_from_y(y, B, D)
            arz[0] = z
            ary[0] = y
            arx[0] = elite
            start = 1

        for k in range(start, pop_size):
            accepted = False
            for _ in range(max(1, self.max_resample)):
                z = rng.normal(size=n)
                y = B @ (D * z)
                x = mean + sigma * y
                if np.all((0.0 <= x) & (x <= 1.0)):
                    accepted = True
                    break

            if not accepted:
                z = rng.normal(size=n)
                y = B @ (D * z)
                x = np.clip(mean + sigma * y, 0.0, 1.0)
                y = (x - mean) / max(sigma, self.min_sigma)
                z = self._z_from_y(y, B, D)

            arz[k] = z
            ary[k] = y
            arx[k] = x

        sampled_state = {
            "n": n,
            "lambda": pop_size,
            "generation": int(state["generation"]),
            "active_idx": [int(i) for i in active_idx],
            "offspring_ids": list(range(1, pop_size + 1)),
            "mean": mean,
            "sigma": sigma,
            "C": state["C"],
            "B": B,
            "D": D,
            "pc": state["pc"],
            "ps": state["ps"],
            "arz": arz,
            "ary": ary,
        }
        return arx, sampled_state

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _normalize(
        population: np.ndarray,
        lower: np.ndarray,
        span: np.ndarray,
        active_idx: np.ndarray,
    ) -> np.ndarray:
        active_lower = lower[active_idx]
        active_span = span[active_idx]
        normalized = (population[:, active_idx] - active_lower) / active_span
        return np.clip(normalized, 0.0, 1.0)

    @staticmethod
    def _denormalize(
        norm_pop: np.ndarray,
        lower: np.ndarray,
        span: np.ndarray,
        active_idx: np.ndarray,
        n_genes: int,
    ) -> np.ndarray:
        out = np.tile(lower, (norm_pop.shape[0], 1))
        out[:, active_idx] = lower[active_idx] + norm_pop * span[active_idx]
        upper = lower + span
        return np.clip(out[:, :n_genes], lower, upper)

    @staticmethod
    def _state_matches(
        state: dict[str, Any] | None,
        pop_size: int,
        active_idx: np.ndarray,
    ) -> bool:
        if state is None:
            return False
        return (
            int(state.get("n", -1)) == len(active_idx)
            and int(state.get("lambda", -1)) == pop_size
            and list(state.get("active_idx", [])) == [int(i) for i in active_idx]
            and len(state.get("offspring_ids", [])) == pop_size
        )

    @staticmethod
    def _symmetrize(C: np.ndarray) -> np.ndarray:
        return 0.5 * (C + C.T)

    @staticmethod
    def _eigendecompose(C: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        C = CMAES._symmetrize(C)
        eigvals, B = np.linalg.eigh(C)
        eigvals = np.maximum(eigvals, 1e-30)
        D = np.sqrt(eigvals)
        C = (B * eigvals) @ B.T
        C = CMAES._symmetrize(C)
        return C, B, D

    @staticmethod
    def _z_from_y(y: np.ndarray, B: np.ndarray, D: np.ndarray) -> np.ndarray:
        safe_D = np.maximum(D, 1e-30)
        return (B.T @ y) / safe_D

    def _rng_for_generation(self, generation: int) -> np.random.Generator:
        if self.seed is None:
            return np.random.default_rng()
        return np.random.default_rng(self.seed + int(generation))
