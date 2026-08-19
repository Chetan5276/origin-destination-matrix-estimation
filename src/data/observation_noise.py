"""Stage 10: observation noise models for turning counts."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)

NoiseModel = Literal["none", "gaussian", "poisson"]


@dataclass(frozen=True)
class NoiseConfig:
    """Configuration for turning-count observation noise."""

    model: NoiseModel = "poisson"
    gaussian_sigma: float = 0.05
    poisson_scale: float = 1.0  # 1.0 = pure Poisson; lower = more noise


def apply_observation_noise(
    clean_counts: np.ndarray,
    config: NoiseConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Apply noise to turning counts.

    Parameters
    ----------
    clean_counts:
        Shape (N, M) or (M,) non-negative counts.
    """
    x = np.asarray(clean_counts, dtype=float)
    x = np.clip(x, 0, None)

    if config.model == "none":
        return x.copy()

    if config.model == "gaussian":
        sigma = config.gaussian_sigma
        if sigma <= 0:
            return x.copy()
        noise = rng.normal(0.0, sigma, size=x.shape)
        return np.clip(x + noise * np.maximum(x, 1.0), 0, None)

    if config.model == "poisson":
        scale = config.poisson_scale
        if scale >= 1.0 - 1e-12:
            # Standard Poisson: x_obs ~ Poisson(x)
            flat = x.ravel()
            noisy = rng.poisson(flat)
            return noisy.reshape(x.shape).astype(float)
        # Partial noise interpolation toward Poisson(scale * x)
        lam = np.clip(x * scale, 0, None)
        noisy = rng.poisson(lam)
        return noisy.astype(float)

    raise ValueError(f"Unknown noise model: {config.model}")


def noise_level_to_scale(noise_pct: float) -> float:
    """
    Map a nominal noise percentage to Poisson scale parameter.

    0% -> scale=1 (no distortion beyond Poisson sampling at same mean)
    Higher pct -> lower effective scale (more dispersion relative to mean)
    """
    if noise_pct <= 0:
        return 1.0
    return max(0.5, 1.0 - noise_pct / 100.0)
