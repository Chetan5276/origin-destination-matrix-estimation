"""Stage 2: OD probability prior on full off-diagonal support."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src import NETWORK_PATH, NUM_ZONES

logger = logging.getLogger(__name__)

# Gravity-style prior: ε_ij = β exp(-d_ij / λ)
DEFAULT_EPSILON_BETA = 1.0
DEFAULT_EPSILON_LAMBDA = 500.0  # ≈ median inter-zone Euclidean distance (meters)
DEFAULT_NETWORK_PATH = NETWORK_PATH
_DISTANCE_CACHE: dict[str, np.ndarray] = {}


@dataclass(frozen=True)
class BaseProbabilityDistribution:
    """Probability vector derived from the base OD matrix + distance prior."""

    probability_vector: np.ndarray
    support_mask: np.ndarray
    total_demand: float
    num_zones: int
    epsilon_matrix: np.ndarray | None = None
    distance_matrix: np.ndarray | None = None

    @property
    def num_cells(self) -> int:
        return self.probability_vector.size

    @property
    def support_indices(self) -> np.ndarray:
        return np.flatnonzero(self.support_mask.ravel(order="C"))

    @property
    def support_probabilities(self) -> np.ndarray:
        return self.probability_vector[self.support_indices]


def full_off_diagonal_support(num_zones: int = NUM_ZONES) -> np.ndarray:
    """Boolean (N, N) mask: True iff i ≠ j."""
    mask = np.ones((num_zones, num_zones), dtype=bool)
    np.fill_diagonal(mask, False)
    return mask


def load_zone_coordinates(
    network_path: Path | None = None,
) -> np.ndarray:
    """Return zone coordinates shape (N, 2) ordered by zone index 1..N."""
    from src.data.network_parser import parse_sumo_network

    path = network_path or DEFAULT_NETWORK_PATH
    network = parse_sumo_network(path)
    coords = np.zeros((NUM_ZONES, 2), dtype=float)
    for zone in range(1, NUM_ZONES + 1):
        jid = network.zone_to_junction[zone]
        coords[zone - 1] = network.junction_coords[jid]
    return coords


def zone_distance_matrix(
    coords: np.ndarray | None = None,
    network_path: Path | None = None,
) -> np.ndarray:
    """Euclidean inter-zone distance matrix (N, N); diagonal is 0."""
    path = str(network_path or DEFAULT_NETWORK_PATH)
    if coords is None and path in _DISTANCE_CACHE:
        return _DISTANCE_CACHE[path]
    if coords is None:
        coords = load_zone_coordinates(network_path)
    coords = np.asarray(coords, dtype=float)
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    if coords is not None:
        _DISTANCE_CACHE[path] = dist
    return dist


def distance_dependent_epsilon(
    distance_matrix: np.ndarray,
    beta: float = DEFAULT_EPSILON_BETA,
    lambda_decay: float = DEFAULT_EPSILON_LAMBDA,
) -> np.ndarray:
    """
    Gravity-style additive prior mass:

        ε_ij = β exp(-d_ij / λ)  for i ≠ j
        ε_ii = 0
    """
    if beta < 0:
        raise ValueError("beta must be non-negative")
    if lambda_decay <= 0:
        raise ValueError("lambda_decay must be positive")

    d = np.asarray(distance_matrix, dtype=float)
    eps = beta * np.exp(-d / lambda_decay)
    np.fill_diagonal(eps, 0.0)
    return eps


def build_prior_od(
    od_matrix: np.ndarray,
    *,
    beta: float = DEFAULT_EPSILON_BETA,
    lambda_decay: float = DEFAULT_EPSILON_LAMBDA,
    distance_matrix: np.ndarray | None = None,
    network_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build T_prior = T_base + ε with full off-diagonal support.

    Returns
    -------
    prior_od, epsilon, distance_matrix
    """
    od = np.asarray(od_matrix, dtype=float).copy()
    np.fill_diagonal(od, 0.0)

    if distance_matrix is None:
        distance_matrix = zone_distance_matrix(network_path=network_path)
    eps = distance_dependent_epsilon(distance_matrix, beta=beta, lambda_decay=lambda_decay)
    prior = od + eps
    np.fill_diagonal(prior, 0.0)
    return prior, eps, distance_matrix


def build_base_probability(
    od_matrix: np.ndarray,
    *,
    beta: float = DEFAULT_EPSILON_BETA,
    lambda_decay: float = DEFAULT_EPSILON_LAMBDA,
    distance_matrix: np.ndarray | None = None,
    network_path: Path | None = None,
) -> BaseProbabilityDistribution:
    """
    Flatten OD into a Dirichlet-safe prior on full off-diagonal support.

    Current (legacy) support was S = {(i,j): T_ij > 0}.
    New support is S = {(i,j): i ≠ j}, with

        p^prior = (T^base + ε) / sum(T^base + ε),
        ε_ij = β exp(-d_ij / λ)  (i ≠ j),  ε_ii = 0.

    ``total_demand`` remains the observed base OD total (not prior mass).
    """
    od = np.asarray(od_matrix, dtype=float).copy()
    np.fill_diagonal(od, 0.0)

    total_demand = float(od.sum())
    if total_demand <= 0:
        raise ValueError("Base OD matrix has zero total demand")

    prior, eps, dist = build_prior_od(
        od,
        beta=beta,
        lambda_decay=lambda_decay,
        distance_matrix=distance_matrix,
        network_path=network_path,
    )
    prior_total = float(prior.sum())
    if prior_total <= 0:
        raise ValueError("Prior OD has zero total mass")

    probability = prior.ravel(order="C") / prior_total
    support_mask = full_off_diagonal_support(od.shape[0])

    if not np.isclose(probability.sum(), 1.0):
        raise ValueError(f"Probability vector sums to {probability.sum()}, not 1")

    # Diagonal must be zero probability; all off-diagonal must be strictly positive.
    diag_prob = probability.reshape(od.shape)[np.diag_indices(od.shape[0])]
    if np.any(diag_prob > 1e-15):
        raise ValueError("Diagonal cells must have zero probability")
    off_diag = probability.reshape(od.shape)[support_mask]
    if np.any(off_diag <= 0):
        raise ValueError("Off-diagonal support cells must have positive probability")

    logger.info(
        "Built full-support prior: length=%d, support=%d, beta=%.4f, lambda=%.1f, "
        "eps_mean=%.4f, eps_max=%.4f",
        probability.size,
        int(support_mask.sum()),
        beta,
        lambda_decay,
        float(eps[support_mask].mean()),
        float(eps.max()),
    )
    return BaseProbabilityDistribution(
        probability_vector=probability,
        support_mask=support_mask,
        total_demand=total_demand,
        num_zones=od.shape[0],
        epsilon_matrix=eps,
        distance_matrix=dist,
    )
