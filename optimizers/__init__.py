"""
Optimizer registry.

Active optimizer selected by OPTIMIZER env var (default: 'ga').
To add a new method: implement Optimizer in a new module, register below.
"""

from __future__ import annotations

import os

from .base import Optimizer
from .ga   import GeneticAlgorithm
from .pso  import PSO
from .de   import DifferentialEvolution
from .cmaes import CMAES
from .abc import ArtificialBeeColony

_REGISTRY: dict[str, type[Optimizer]] = {
    "ga": GeneticAlgorithm,
    "pso": PSO,
    "de": DifferentialEvolution,
    "cmaes": CMAES,
    "abc": ArtificialBeeColony,
}

def register(name: str, cls: type[Optimizer]) -> None:
    _REGISTRY[name.strip().lower()] = cls

def available() -> list[str]:
    return sorted(_REGISTRY)

def get_optimizer(name: str | None = None, **kwargs) -> Optimizer:
    if name is None:
        name = os.environ.get("OPTIMIZER", "ga")
    key = name.strip().lower()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown optimizer {name!r}. Available: {available()}"
        )
    return _REGISTRY[key](**kwargs)

__all__ = ["Optimizer", "get_optimizer", "register", "available"]
