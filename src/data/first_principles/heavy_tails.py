"""Stage C: heavy-tailed corridor reweighting."""

from __future__ import annotations

import numpy as np


def apply_gamma_weights(
    seed: np.ndarray,
    rng: np.random.Generator,
    support_mask: np.ndarray,
    *,
    shape: float = 0.5,
    scale: float = 1.0,
) -> np.ndarray:
    """
    Independent Gamma weights on support:

        S_ij ← w_ij * S_ij,   w_ij ~ Gamma(shape, scale)

    Shape < 1 yields many tiny and few large corridors.
    """
    if shape <= 0 or scale <= 0:
        raise ValueError("Gamma shape and scale must be positive")
    out = np.asarray(seed, dtype=np.float64).copy()
    n = int(support_mask.sum())
    weights = rng.gamma(shape, scale, size=n)
    weights = np.maximum(weights, 1e-30)
    out[support_mask] *= weights
    np.fill_diagonal(out, 0.0)
    return out
