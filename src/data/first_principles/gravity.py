"""Stage B: gravity interaction kernel and reciprocity mixing."""

from __future__ import annotations

import numpy as np


def distance_impedance(
    distance: np.ndarray,
    *,
    decay: str = "exponential",
    lambda_decay: float = 500.0,
    power_gamma: float = 1.5,
) -> np.ndarray:
    """
    Monotone decreasing impedance ``f(d)``.

    exponential: f(d) = exp(-d / λ)
    power:       f(d) = (d + ε)^(-γ)
    """
    d = np.asarray(distance, dtype=np.float64)
    if decay == "exponential":
        if lambda_decay <= 0:
            raise ValueError("lambda_decay must be positive")
        f = np.exp(-d / lambda_decay)
    elif decay == "power":
        if power_gamma <= 0:
            raise ValueError("power_gamma must be positive")
        f = np.power(d + 1.0, -power_gamma)
    else:
        raise ValueError(f"Unknown decay mode: {decay}")
    np.fill_diagonal(f, 0.0)
    return f


def gravity_seed(
    u: np.ndarray,
    v: np.ndarray,
    impedance: np.ndarray,
    support_mask: np.ndarray,
) -> np.ndarray:
    """
    Gravity interaction:

        S_ij = u_i * v_j * f(d_ij)   for (i,j) on support, else 0.
    """
    s = np.outer(u, v) * impedance
    s = np.where(support_mask, s, 0.0)
    np.fill_diagonal(s, 0.0)
    return np.maximum(s, 0.0)


def apply_reciprocity(seed: np.ndarray, reciprocity: float) -> np.ndarray:
    """
    Mix toward transpose for approximate daily reciprocity:

        S ← (1 - ρ) S + ρ Sᵀ

    with ρ ∈ [0, 0.5]. ρ = 0 disables; ρ = 0.5 fully symmetrizes the mean structure.
    """
    rho = float(np.clip(reciprocity, 0.0, 0.5))
    if rho <= 0:
        return seed
    out = (1.0 - rho) * seed + rho * seed.T
    np.fill_diagonal(out, 0.0)
    return np.maximum(out, 0.0)


def build_gravity_matrix(
    u: np.ndarray,
    v: np.ndarray,
    distance: np.ndarray,
    support_mask: np.ndarray,
    *,
    decay: str = "exponential",
    lambda_decay: float = 500.0,
    power_gamma: float = 1.5,
    reciprocity: float = 0.35,
) -> np.ndarray:
    """Full Stage B: impedance → gravity → reciprocity mix."""
    f = distance_impedance(
        distance, decay=decay, lambda_decay=lambda_decay, power_gamma=power_gamma
    )
    s = gravity_seed(u, v, f, support_mask)
    return apply_reciprocity(s, reciprocity)
