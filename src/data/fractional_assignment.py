"""Stages 6--7: route turn vectors and fractional assignment matrix."""

from __future__ import annotations

import logging

import numpy as np

from src.data.k_shortest_paths import RoutePath
from src.data.od_pairs import OdPairIndex
from src.data.route_choice import RouteChoiceResult, RouteWithProbability
from src.data.turning_movements import TurningMovementIndex

logger = logging.getLogger(__name__)


def route_turn_indices(
    route: RoutePath,
    turning_index: TurningMovementIndex,
) -> np.ndarray:
    """Return sorted turn indices used by a route (binary incidence on turns)."""
    indices: list[int] = []
    for inc, out in zip(route.path_edges[:-1], route.path_edges[1:]):
        tid = turning_index.id_for(inc, out)
        if tid is not None:
            indices.append(tid)
    return np.array(sorted(set(indices)), dtype=np.int32)


def route_turn_vector(
    route: RoutePath,
    turning_index: TurningMovementIndex,
) -> np.ndarray:
    """Binary turn vector of shape (num_turns,)."""
    vec = np.zeros(turning_index.num_turning_movements, dtype=float)
    for idx in route_turn_indices(route, turning_index):
        vec[idx] = 1.0
    return vec


def build_fractional_assignment_matrix(
    turning_index: TurningMovementIndex,
    od_index: OdPairIndex,
    route_choices: RouteChoiceResult,
) -> np.ndarray:
    """
    Construct probabilistic A_turn.

    A_turn[i, j] = sum_r P(r) * I(turn i is on route r)
    """
    n_turns = turning_index.num_turning_movements
    n_od = od_index.num_od_pairs
    a_turn = np.zeros((n_turns, n_od), dtype=np.float64)

    for (origin, destination), col_idx in od_index.od_pair_to_index.items():
        route_probs: list[RouteWithProbability] = route_choices.choices.get(
            (origin, destination), []
        )
        if not route_probs:
            continue

        col = np.zeros(n_turns, dtype=np.float64)
        for rp in route_probs:
            for tid in route_turn_indices(rp.route, turning_index):
                col[tid] += rp.probability
        a_turn[:, col_idx] = col

    logger.info(
        "Built fractional A_turn shape=%s nnz=%d max=%.4f",
        a_turn.shape,
        np.count_nonzero(a_turn),
        a_turn.max(),
    )
    return a_turn


def validate_fractional_matrix(a_turn: np.ndarray) -> dict[str, float | bool]:
    """Check entries are in [0, 1] and columns sum reasonably."""
    col_sums = a_turn.sum(axis=0)
    return {
        "min_entry": float(a_turn.min()),
        "max_entry": float(a_turn.max()),
        "entries_in_01": bool(a_turn.min() >= -1e-12 and a_turn.max() <= 1.0 + 1e-9),
        "max_column_sum": float(col_sums.max()),
        "mean_column_sum": float(col_sums[col_sums > 0].mean()) if np.any(col_sums > 0) else 0.0,
    }
