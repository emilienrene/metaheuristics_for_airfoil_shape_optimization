from __future__ import annotations
"""
Parameterization contract.

Every airfoil parameterization (PARSEC, CST, B-spline, NURBS, ...) implements
this interface. The rest of the pipeline never references a specific method:
it asks the active method for its bounds, its parameter names, and a decode
function, and treats the genotype as an opaque vector of length ``n_params``.

A method must define:
    param_names   -> list[str]      column names for the genotype CSV
    bounds        -> (n_params, 2)  [min, max] per gene, used by LHS sampling
    decode(params)-> (M, 2) ndarray surface coordinates [X, Z]

and may override:
    mutation_mode -> "multiplicative" | "additive"
    reference     -> ndarray | None  a known genotype (e.g. NACA0012) for sanity checks
"""

from abc import ABC, abstractmethod
import numpy as np


class Parameterization(ABC):

    #: short identifier, also the key used to select the method at runtime
    name: str = "base"

    #: "multiplicative" preserves PARSEC's historical GA behaviour.
    #: "additive" (bound-scaled) is the safe default for genes that may be
    #: zero or sign-crossing, such as CST coefficients or control-point offsets.
    mutation_mode: str = "additive"

    #: optional reference genotype, purely for validation / seeding
    reference: np.ndarray | None = None

    # ----- required interface -------------------------------------------------

    @property
    @abstractmethod
    def param_names(self) -> list[str]:
        """Ordered gene names. Defines the genotype CSV column order."""

    @property
    @abstractmethod
    def bounds(self) -> np.ndarray:
        """(n_params, 2) array of [min, max] per gene."""

    @abstractmethod
    def decode(self, params: np.ndarray) -> np.ndarray:
        """
        Map a genotype vector to surface coordinates.

        Returns an (M, 2) array of [X, Z] ordered TE -> LE (upper) then
        LE -> TE (lower), i.e. the same point ordering the solver expects in
        the .dat files. Implementations are responsible for their own
        discretization.
        """

    # ----- derived / shared ---------------------------------------------------

    @property
    def n_params(self) -> int:
        return len(self.param_names)

    @property
    def header(self) -> str:
        """Genotype CSV header line, e.g. 'Airfoil_idx,R_LE,X_UP,...'."""
        return "Airfoil_idx," + ",".join(self.param_names)

    def validate(self) -> None:
        """Cheap structural checks so a broken method fails loudly at startup."""
        b = np.asarray(self.bounds, dtype=float)
        if b.shape != (self.n_params, 2):
            raise ValueError(
                f"{self.name}: bounds shape {b.shape} != ({self.n_params}, 2)"
            )
        if np.any(b[:, 1] < b[:, 0]):
            raise ValueError(f"{self.name}: found a bound with max < min")
        if self.mutation_mode not in ("multiplicative", "additive"):
            raise ValueError(f"{self.name}: bad mutation_mode {self.mutation_mode!r}")
