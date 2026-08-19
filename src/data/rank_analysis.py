"""Rank and conditioning analysis of the assignment matrix."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from src import NUM_OD_PAIRS

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RankAnalysisResult:
    """Linear-algebra diagnostics for A_turn."""

    shape: tuple[int, int]
    rank: int
    condition_number: float
    nullity: int
    num_turning_movements: int
    num_od_pairs: int

    def as_dict(self) -> dict[str, float | int | tuple[int, int]]:
        return {
            "shape": self.shape,
            "rank": self.rank,
            "condition_number": self.condition_number,
            "nullity": self.nullity,
            "num_turning_movements": self.num_turning_movements,
            "num_od_pairs": self.num_od_pairs,
        }


def analyze_rank(a_turn: np.ndarray, od_pairs: int = NUM_OD_PAIRS) -> RankAnalysisResult:
    """Compute rank, condition number, and nullity of A_turn."""
    matrix = np.asarray(a_turn, dtype=float)
    rank = int(np.linalg.matrix_rank(matrix))
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    positive_sv = singular_values[singular_values > 1e-12]
    if len(positive_sv) >= 2:
        condition_number = float(positive_sv[0] / positive_sv[-1])
    elif len(positive_sv) == 1:
        condition_number = float("inf")
    else:
        condition_number = float("nan")

    nullity = od_pairs - rank
    result = RankAnalysisResult(
        shape=matrix.shape,
        rank=rank,
        condition_number=condition_number,
        nullity=nullity,
        num_turning_movements=matrix.shape[0],
        num_od_pairs=matrix.shape[1],
    )
    logger.info(
        "A_turn shape=%s rank=%d nullity=%d cond=%.4g",
        result.shape,
        result.rank,
        result.nullity,
        result.condition_number,
    )
    return result
