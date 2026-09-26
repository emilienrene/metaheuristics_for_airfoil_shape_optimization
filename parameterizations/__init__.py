"""
Parameterization registry.

Active method selected by PARAM_METHOD env var (default: 'parsec').
To add a method: implement Parameterization in a new module, register below.
"""

from __future__ import annotations
import os

from .base       import Parameterization
from .parsec     import PARSEC
from .cst        import CST
from .bspline    import BSplineParam
from .hickshenne import HicksHenne
from .ffd        import FFD

_REGISTRY: dict[str, type[Parameterization]] = {
    "parsec":     PARSEC,
    "cst":        CST,
    "bspline":    BSplineParam,
    "hickshenne": HicksHenne,
    "ffd":        FFD,
}

def register(name: str, cls: type[Parameterization]) -> None:
    _REGISTRY[name.strip().lower()] = cls

def available() -> list[str]:
    return sorted(_REGISTRY)

def get_parameterization(name: str | None = None, **kwargs) -> Parameterization:
    if name is None:
        name = os.environ.get("PARAM_METHOD", "parsec")
    key = name.strip().lower()
    if key not in _REGISTRY:
        raise ValueError(f"Unknown parameterization {name!r}. Available: {available()}")
    method = _REGISTRY[key](**kwargs)
    method.validate()
    return method

__all__ = ["Parameterization", "get_parameterization", "register", "available"]
