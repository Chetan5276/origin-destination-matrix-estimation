"""Build the turning-movement assignment matrix A_turn."""

from __future__ import annotations

import logging

import numpy as np
from scipy import sparse

from src.data.od_pairs import OdPairIndex
from src.data.route_assignment import RouteAssignment
from src.data.turning_movements import TurningMovementIndex

logger = logging.getLogger(__name__)


def build_assignment_matrix(
    turning_index: TurningMovementIndex,
    od_index: OdPairIndex,
    routes: RouteAssignment,
    weighted: bool = False,
) -> np.ndarray:
    """
    Construct A_turn with shape (num_turns, num_od_pairs).

    Entry A_turn[i, j] indicates whether turning movement i is used by OD pair j.
    Initially binary (0/1); set ``weighted=True`` for future fractional splits.
    """
    n_turns = turning_index.num_turning_movements
    n_od = od_index.num_od_pairs
    a_turn = np.zeros((n_turns, n_od), dtype=float)

    for (origin, destination), col_idx in od_index.od_pair_to_index.items():
        edge_path = routes.path_edges(origin, destination)
        if len(edge_path) < 2:
            continue

        for inc, out in zip(edge_path[:-1], edge_path[1:]):
            row_idx = turning_index.id_for(inc, out)
            if row_idx is None:
                logger.debug(
                    "Turn (%s, %s) on OD (%d,%d) not in index",
                    inc,
                    out,
                    origin,
                    destination,
                )
                continue
            a_turn[row_idx, col_idx] = 1.0 if not weighted else a_turn[row_idx, col_idx] + 1.0

    logger.info("Built A_turn with shape %s (nnz=%d)", a_turn.shape, np.count_nonzero(a_turn))
    return a_turn


def assignment_matrix_sparse(a_turn: np.ndarray) -> sparse.csr_matrix:
    """Return a CSR sparse view of A_turn."""
    return sparse.csr_matrix(a_turn)
