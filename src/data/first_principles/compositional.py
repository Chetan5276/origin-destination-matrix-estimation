"""Stage D: Dirichlet compositional noise on the probability simplex."""

from __future__ import annotations

import numpy as np


def seed_to_probability(seed: np.ndarray, support_mask: np.ndarray) -> np.ndarray:
    """Normalize nonnegative seed to a probability matrix on support."""
    p = np.asarray(seed, dtype=np.float64).copy()
    p[~support_mask] = 0.0
    np.fill_diagonal(p, 0.0)
    total = float(p.sum())
    if total <= 0:
        # Uniform fallback on support
        p = support_mask.astype(np.float64)
        total = float(p.sum())
    p /= total
    return p


def dirichlet_reweight(
    probability: np.ndarray,
    support_mask: np.ndarray,
    alpha: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Draw ``p ~ Dir(α * p_prior)`` on support cells; zeros elsewhere.

    Mean equals the gravity/heavy-tail prior; variance decreases with α.
    """
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    p = seed_to_probability(probability, support_mask)
    idx = np.flatnonzero(support_mask.ravel(order="C"))
    prior = p.ravel(order="C")[idx]
    # Guard tiny prior mass
    prior = np.maximum(prior, 1e-18)
    prior /= prior.sum()
    concentration = alpha * prior
    # Numerical floor for scipy/numpy dirichlet
    concentration = np.maximum(concentration, 1e-12)
    sample = rng.dirichlet(concentration)
    out = np.zeros_like(p)
    out.ravel(order="C")[idx] = sample
    np.fill_diagonal(out, 0.0)
    return out
