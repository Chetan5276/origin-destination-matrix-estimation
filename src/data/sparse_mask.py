"""Heavy-tailed sparse reweighting after Dirichlet sampling."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Gamma(shape < 1) is heavy near zero with a long right tail → many tiny, few large flows
DEFAULT_GAMMA_SHAPE = 0.5
DEFAULT_GAMMA_SCALE = 1.0


def sample_gamma_weights(
    size: int,
    rng: np.random.Generator,
    *,
    shape: float = DEFAULT_GAMMA_SHAPE,
    scale: float = DEFAULT_GAMMA_SCALE,
) -> np.ndarray:
    """Sample positive heavy-tailed weights ``w ~ Gamma(shape, scale)``."""
    if shape <= 0 or scale <= 0:
        raise ValueError("Gamma shape and scale must be positive")
    weights = rng.gamma(shape, scale, size=size)
    # Guard against rare exact zeros under floating-point underflow
    return np.maximum(weights, 1e-30)


def apply_gamma_sparse_mask(
    probabilities: np.ndarray,
    rng: np.random.Generator,
    *,
    shape: float = DEFAULT_GAMMA_SHAPE,
    scale: float = DEFAULT_GAMMA_SCALE,
    support_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Reweight Dirichlet probabilities with independent Gamma weights:

        p_ij = (w_ij * p_ij^prior) / sum_{k,l} (w_kl * p_kl^prior)

    with ``w_ij ~ Gamma(shape, scale)`` on the support (diagonal stays zero).

    Heavy-tailed Gamma (shape < 1) naturally produces many tiny OD flows and
    a few large ones.
    """
    p = np.asarray(probabilities, dtype=np.float64).copy()
    if support_mask is None:
        support = p > 0
    else:
        support = np.asarray(support_mask, dtype=bool)
        if support.shape != p.shape:
            raise ValueError("support_mask shape must match probabilities")

    n_support = int(support.sum())
    if n_support == 0:
        raise ValueError("No support cells to reweight")

    weights = np.zeros_like(p)
    weights[support] = sample_gamma_weights(n_support, rng, shape=shape, scale=scale)

    reweighted = weights * p
    total = float(reweighted.sum())
    if total <= 0:
        raise ValueError("Gamma-reweighted probabilities have zero mass")
    reweighted /= total
    reweighted[~support] = 0.0
    return reweighted
