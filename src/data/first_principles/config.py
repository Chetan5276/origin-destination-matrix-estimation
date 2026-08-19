"""Configuration for first-principles OD generation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FPGeneratorConfig:
    """
    Hyperparameters for latent-gravity OD synthesis.

    Pipeline
    --------
    A. Latent production / attraction factors
    B. Gravity interaction kernel + optional reciprocity mix
    C. Gamma heavy-tailed corridor weights
    D. Dirichlet compositional noise
    E. Scale to G + IPF to marginal targets
    F. Dataset quality filters (handled in CLI)
    """

    total_demand: float = 365_475.0  # Sioux Falls–scale default; override via CLI
    num_zones: int = 24

    # Stage A: latent factors
    latent_dim: int = 4
    factor_shape: float = 2.0  # Gamma shape for zone loadings / type intensities
    factor_scale: float = 1.0
    spatial_smooth_length: float = 400.0  # meters; 0 disables spatial smoothing
    spatial_smooth_strength: float = 0.5  # blend weight in [0, 1]

    # Stage B: gravity / distance decay
    decay: str = "exponential"  # "exponential" | "power"
    lambda_decay: float = 500.0  # meters (exponential)
    power_gamma: float = 1.5  # power-law exponent
    reciprocity: float = 0.35  # mix toward S^T; 0 = none, 0.5 = fully symmetrized mean

    # Stage C: heavy tails
    gamma_shape: float = 0.5
    gamma_scale: float = 1.0
    apply_gamma_mask: bool = True

    # Stage D: Dirichlet
    alpha: float = 200.0  # lower than base-OD pipeline → more compositional diversity

    # Stage E: IPF
    ipf_tol: float = 1e-3
    ipf_max_iter: int = 1000
    seed_floor: float = 1e-6
    # Marginal targets: draw from latent u,v then rescale to G (True),
    # or use fixed equal-share marginals (False).
    latent_marginals: bool = True
