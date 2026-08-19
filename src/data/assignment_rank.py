"""Stage 8: rank analysis with binary vs probabilistic comparison."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import numpy as np

from src import NUM_OD_PAIRS
from src.data.rank_analysis import RankAnalysisResult, analyze_rank

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExtendedRankAnalysis:
    """Rank diagnostics including singular values."""

    result: RankAnalysisResult
    singular_values: list[float]
    effective_rank_99: int  # number of SVs to capture 99% energy

    def to_dict(self) -> dict:
        d = self.result.as_dict()
        d["singular_values"] = self.singular_values[:20]  # truncate for JSON
        d["effective_rank_99"] = self.effective_rank_99
        return d


def extended_rank_analysis(
    a_turn: np.ndarray,
    od_pairs: int = NUM_OD_PAIRS,
) -> ExtendedRankAnalysis:
    """Compute rank, condition number, nullity, and singular values."""
    matrix = np.asarray(a_turn, dtype=float)
    sv = np.linalg.svd(matrix, compute_uv=False)
    base = analyze_rank(matrix, od_pairs=od_pairs)

    energy = np.cumsum(sv**2) / (sv**2).sum() if sv.size else np.array([1.0])
    eff_rank = int(np.searchsorted(energy, 0.99) + 1) if sv.size else 0

    return ExtendedRankAnalysis(
        result=base,
        singular_values=sv.tolist(),
        effective_rank_99=eff_rank,
    )


@dataclass(frozen=True)
class AssignmentComparison:
    """Compare binary (shortest-path) vs probabilistic assignment matrices."""

    binary: ExtendedRankAnalysis
    probabilistic: ExtendedRankAnalysis
    rank_improvement: int
    nullity_reduction: int

    def to_dict(self) -> dict:
        return {
            "binary": self.binary.to_dict(),
            "probabilistic": self.probabilistic.to_dict(),
            "rank_improvement": self.rank_improvement,
            "nullity_reduction": self.nullity_reduction,
        }


def compare_assignment_matrices(
    binary_a: np.ndarray,
    probabilistic_a: np.ndarray,
) -> AssignmentComparison:
    """Compare rank structure of binary and fractional assignment matrices."""
    bin_analysis = extended_rank_analysis(binary_a)
    prob_analysis = extended_rank_analysis(probabilistic_a)
    rank_improvement = prob_analysis.result.rank - bin_analysis.result.rank
    nullity_reduction = bin_analysis.result.nullity - prob_analysis.result.nullity

    logger.info(
        "Rank comparison: binary=%d probabilistic=%d (+%d)",
        bin_analysis.result.rank,
        prob_analysis.result.rank,
        rank_improvement,
    )
    return AssignmentComparison(
        binary=bin_analysis,
        probabilistic=prob_analysis,
        rank_improvement=rank_improvement,
        nullity_reduction=nullity_reduction,
    )
