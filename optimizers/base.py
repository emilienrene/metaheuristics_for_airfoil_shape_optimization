"""
Optimizer contract.

Every metaheuristic implements this interface. The pipeline never references
a specific algorithm — it calls optimizer.step() and gets the next generation.

The optimizer receives the current population as a numpy array sorted
best-first by fitness, plus the gene bounds and mutation mode from the active
parameterization. It returns the next generation as a numpy array of the
same shape.

File I/O, path resolution, and genotype-to-geometry decoding are handled by
the pipeline layer (run_step.py + core/). The optimizer does pure numpy.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class Optimizer(ABC):

    #: short identifier used by the registry and OPTIMIZER env var
    name: str = "base"

    @abstractmethod
    def step(
        self,
        population: np.ndarray,   # (pop_size, n_genes), sorted best-first
        bounds: np.ndarray,        # (n_genes, 2)  [min, max] per gene
        mutation_mode: str,        # "additive" | "multiplicative"
        **kwargs,
    ) -> np.ndarray:               # (pop_size, n_genes) next generation
        """Produce the next generation from the current ranked population."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
