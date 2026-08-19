"""Stage A: low-dimensional latent production / attraction factors."""

from __future__ import annotations

import numpy as np


def _spatial_kernel(distance: np.ndarray, length: float) -> np.ndarray:
    """Gaussian spatial kernel K_ij = exp(-d_ij^2 / (2 λ^2)); rows normalized."""
    if length <= 0:
        return np.eye(distance.shape[0], dtype=np.float64)
    k = np.exp(-(distance**2) / (2.0 * length**2))
    np.fill_diagonal(k, 1.0)
    row_sum = k.sum(axis=1, keepdims=True)
    return k / np.maximum(row_sum, 1e-15)


def sample_type_intensities(
    latent_dim: int,
    rng: np.random.Generator,
    *,
    shape: float = 2.0,
    scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample city-level land-use / activity intensities.

    ``prod_types[t]`` scales production of type t; ``attr_types[t]`` attraction.
    """
    prod = rng.gamma(shape, scale, size=latent_dim).astype(np.float64)
    attr = rng.gamma(shape, scale, size=latent_dim).astype(np.float64)
    return prod, attr


def sample_zone_memberships(
    num_zones: int,
    latent_dim: int,
    coords: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Soft zone→type memberships via distances to random type centers.

    Returns ``M`` shape (N, r) with rows summing to 1 (simplex).
    """
    coords = np.asarray(coords, dtype=np.float64)
    # Place type centers near random zones + small jitter
    idx = rng.choice(num_zones, size=latent_dim, replace=True)
    centers = coords[idx] + rng.normal(0.0, 50.0, size=(latent_dim, 2))
    # Squared distances (N, r)
    diff = coords[:, None, :] - centers[None, :, :]
    dist2 = np.sum(diff * diff, axis=-1)
    # Softmax(-dist2 / temp)
    temp = max(float(np.median(dist2)) * 0.25, 1.0)
    logits = -dist2 / temp
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-15)


def build_zone_factors(
    memberships: np.ndarray,
    type_intensities: np.ndarray,
) -> np.ndarray:
    """Zone strength = membership-weighted type intensities. Shape (N,)."""
    return memberships @ type_intensities


def smooth_factors(
    factors: np.ndarray,
    distance: np.ndarray,
    *,
    length: float,
    strength: float,
) -> np.ndarray:
    """
    Spatially smooth zone factors for coherence.

    ``out = (1-s) * factors + s * K @ factors``, then clip to positive.
    """
    s = float(np.clip(strength, 0.0, 1.0))
    if s <= 0 or length <= 0:
        return np.maximum(factors, 1e-12)
    k = _spatial_kernel(distance, length)
    mixed = (1.0 - s) * factors + s * (k @ factors)
    return np.maximum(mixed, 1e-12)


def sample_latent_factors(
    num_zones: int,
    latent_dim: int,
    coords: np.ndarray,
    distance: np.ndarray,
    rng: np.random.Generator,
    *,
    factor_shape: float = 2.0,
    factor_scale: float = 1.0,
    spatial_smooth_length: float = 400.0,
    spatial_smooth_strength: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample production ``u`` and attraction ``v`` on a low-dimensional manifold.

    Returns
    -------
    u, v : (N,)
    memberships : (N, r)
    """
    prod_t, attr_t = sample_type_intensities(
        latent_dim, rng, shape=factor_shape, scale=factor_scale
    )
    m = sample_zone_memberships(num_zones, latent_dim, coords, rng)
    u = build_zone_factors(m, prod_t)
    v = build_zone_factors(m, attr_t)
    u = smooth_factors(
        u, distance, length=spatial_smooth_length, strength=spatial_smooth_strength
    )
    v = smooth_factors(
        v, distance, length=spatial_smooth_length, strength=spatial_smooth_strength
    )
    return u, v, m
