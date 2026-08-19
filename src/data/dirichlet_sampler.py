"""Stage 3: Dirichlet sampling around the base OD probability distribution."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from src.data.od_probability import BaseProbabilityDistribution
from src.data.sparse_mask import (
    DEFAULT_GAMMA_SCALE,
    DEFAULT_GAMMA_SHAPE,
    apply_gamma_sparse_mask,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DirichletSample:
    """One draw from Dir(alpha * p) on the base support."""

    probability_vector: np.ndarray
    alpha: float


def sample_dirichlet_probability(
    base_dist: BaseProbabilityDistribution,
    alpha: float,
    rng: np.random.Generator,
) -> DirichletSample:
    """Draw ``p_tilde ~ Dirichlet(alpha * p)`` on the support of the base OD."""
    if alpha <= 0:
        raise ValueError("alpha must be positive")

    support_p = base_dist.support_probabilities
    concentration = alpha * support_p
    if np.any(concentration <= 0):
        raise ValueError("All concentration parameters must be positive on support")

    sampled_support = rng.dirichlet(concentration)

    full = np.zeros_like(base_dist.probability_vector)
    full[base_dist.support_indices] = sampled_support

    return DirichletSample(probability_vector=full, alpha=alpha)


def sample_demand_matrix(
    support_indices: np.ndarray,
    support_probabilities: np.ndarray,
    alpha: float,
    total_demand: float,
    num_zones: int,
    rng: np.random.Generator,
    out: np.ndarray | None = None,
    *,
    gamma_shape: float = DEFAULT_GAMMA_SHAPE,
    gamma_scale: float = DEFAULT_GAMMA_SCALE,
    apply_sparse_mask: bool = True,
) -> np.ndarray:
    """
    Dirichlet sample → optional Gamma sparse reweight → scale to demand.

    Pipeline
    --------
    1. ``p ~ Dir(alpha * p_prior)`` on support
    2. ``w_ij ~ Gamma(shape, scale)``; ``p_ij ∝ w_ij p_ij``
    3. Scale probabilities to ``total_demand``
    """
    concentration = alpha * support_probabilities
    sampled = rng.dirichlet(concentration)

    if out is None:
        matrix = np.zeros((num_zones, num_zones), dtype=np.float64)
    else:
        matrix = out
        matrix.fill(0.0)

    matrix.ravel()[support_indices] = sampled

    if apply_sparse_mask:
        support_mask = np.zeros((num_zones, num_zones), dtype=bool)
        support_mask.ravel()[support_indices] = True
        reweighted = apply_gamma_sparse_mask(
            matrix,
            rng,
            shape=gamma_shape,
            scale=gamma_scale,
            support_mask=support_mask,
        )
        matrix[:] = reweighted

    matrix *= total_demand
    return matrix


def reconstruct_demand(
    sample: DirichletSample,
    total_demand: float,
    num_zones: int,
) -> np.ndarray:
    """Convert sampled probabilities back to a demand matrix (Stage 4)."""
    demand = sample.probability_vector.reshape(num_zones, num_zones, order="C")
    return demand * total_demand
