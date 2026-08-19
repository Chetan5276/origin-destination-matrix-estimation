"""Stage 7: marginal perturbation with grand-total conservation."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PerturbedMarginals:
    """Target production and attraction vectors for IPF."""

    target_productions: np.ndarray
    target_attractions: np.ndarray
    total_demand: float
    perturbation: float


def perturb_marginals(
    productions: np.ndarray,
    attractions: np.ndarray,
    total_demand: float,
    perturbation: float,
    rng: np.random.Generator,
) -> PerturbedMarginals:
    """
    Apply multiplicative shocks and rescale to preserve total demand Q.

    P'_i = P_i (1 + eps_i),  eps_i ~ U(-delta, delta)
    A'_j = A_j (1 + eta_j),  eta_j ~ U(-delta, delta)

    Both vectors are then normalized to sum to Q.
    """
    if not 0 <= perturbation < 1:
        raise ValueError("perturbation must be in [0, 1)")

    p_tilde = productions * (
        1.0 + rng.uniform(-perturbation, perturbation, size=productions.shape)
    )
    a_tilde = attractions * (
        1.0 + rng.uniform(-perturbation, perturbation, size=attractions.shape)
    )

    target_productions = total_demand * p_tilde / p_tilde.sum()
    target_attractions = total_demand * a_tilde / a_tilde.sum()

    return PerturbedMarginals(
        target_productions=target_productions,
        target_attractions=target_attractions,
        total_demand=total_demand,
        perturbation=perturbation,
    )
