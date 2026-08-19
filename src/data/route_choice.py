"""Stage 5: Multinomial logit route choice model."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from src.data.k_shortest_paths import RouteCatalog, RoutePath

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RouteWithProbability:
    """Route path annotated with logit choice probability."""

    route: RoutePath
    probability: float
    utility: float


@dataclass
class RouteChoiceResult:
    """Logit probabilities for all routes in the catalog."""

    choices: dict[tuple[int, int], list[RouteWithProbability]]
    theta: float


def logit_route_probabilities(
    routes: list[RoutePath],
    theta: float,
) -> list[RouteWithProbability]:
    """
    Compute multinomial logit probabilities.

    Utility: U_r = -theta * c_r
    Probability: P_r = exp(U_r) / sum_k exp(U_k)
    """
    if not routes:
        return []
    if theta <= 0:
        raise ValueError("theta must be positive")

    costs = np.array([r.path_cost for r in routes], dtype=float)
    utilities = -theta * costs
    # Stable softmax
    utilities -= utilities.max()
    exp_u = np.exp(utilities)
    probs = exp_u / exp_u.sum()

    return [
        RouteWithProbability(route=r, probability=float(p), utility=float(u))
        for r, p, u in zip(routes, probs, utilities)
    ]


def apply_logit_choice(
    catalog: RouteCatalog,
    theta: float = 0.1,
) -> RouteChoiceResult:
    """Attach logit probabilities to every route in the catalog."""
    choices: dict[tuple[int, int], list[RouteWithProbability]] = {}
    for od_pair, route_list in catalog.routes.items():
        choices[od_pair] = logit_route_probabilities(route_list, theta)

    logger.info("Applied logit route choice with theta=%.4f", theta)
    return RouteChoiceResult(choices=choices, theta=theta)
