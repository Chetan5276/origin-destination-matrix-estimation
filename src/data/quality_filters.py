"""Quality filters for synthetic OD candidate selection (Constraints 3--4).

Workflow after IPF
------------------
Candidate pool → filters (sparsity, correlation, trip-length, entropy,
pairwise Frobenius diversity) → accept / reject → final dataset.

Constraint 3 (High Entropy)
    Dataset-level cell-wise entropy across samples. We generate a large
    candidate pool and greedily keep matrices that improve mean cell entropy.

Constraint 4 (High Variability)
    Pairwise Frobenius distance: reject a candidate if
    ``||T - T'||_F < threshold`` for any already accepted matrix.
    Mean cell CV is reported alongside.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import numpy as np

from src.data.od_probability import zone_distance_matrix

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QualityFilterConfig:
    """Thresholds and knobs for candidate accept / reject."""

    # Pool sizing: generate oversample_factor * target_size candidates
    oversample_factor: float = 10.0
    # Sparsity: fraction of exact zeros (diagonal always zero)
    min_sparsity: float = 0.0
    max_sparsity: float = 0.95
    # Correlation with base OD (too high → clones; too low → unrealistic)
    min_correlation: float = 0.30
    max_correlation: float = 0.95
    # Trip-length: relative error vs base mean trip length
    max_trip_length_rel_error: float = 0.35
    # Pairwise Frobenius diversity (Constraint 4)
    min_frobenius_distance: float = 0.0  # 0 → auto = auto_frobenius_fraction * median
    auto_frobenius_fraction: float = 0.35
    # Soft entropy contribution weight in ranking (Constraint 3)
    entropy_bins: int = 20
    # Hard requirements
    require_zero_diagonal: bool = True
    max_diagonal_mass: float = 1e-6


def _auto_frobenius_threshold(
    flat: np.ndarray,
    fraction: float,
    seed: int = 42,
    n_pairs: int = 2_000,
) -> float:
    """Estimate diversity threshold from median pairwise Frobenius distance."""
    n = flat.shape[0]
    if n < 2:
        return 0.0
    rng = np.random.default_rng(seed)
    n_pairs = min(n_pairs, n * (n - 1) // 2)
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    same = i == j
    while same.any():
        j[same] = rng.integers(0, n, size=int(same.sum()))
        same = i == j
    dists = np.linalg.norm(flat[i] - flat[j], axis=1)
    return float(fraction * np.median(dists))


@dataclass
class FilterStats:
    """Summary of accept / reject decisions."""

    n_candidates: int
    n_accepted: int
    n_rejected_sparsity: int = 0
    n_rejected_correlation: int = 0
    n_rejected_trip_length: int = 0
    n_rejected_diagonal: int = 0
    n_rejected_pairwise: int = 0
    n_rejected_pool_exhausted: int = 0
    mean_cell_entropy: float = float("nan")
    mean_cell_cv: float = float("nan")
    mean_pairwise_frobenius: float = float("nan")
    min_pairwise_frobenius: float = float("nan")

    def to_dict(self) -> dict:
        return asdict(self)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def matrix_sparsity(od: np.ndarray) -> float:
    return float(np.mean(od <= 1e-12))


def mean_trip_length(od: np.ndarray, distance: np.ndarray) -> float:
    """Flow-weighted mean inter-zone distance."""
    mass = float(od.sum())
    if mass <= 0:
        return float("nan")
    return float(np.sum(od * distance) / mass)


def frobenius_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def cell_wise_entropy(
    batch: np.ndarray,
    n_bins: int = 20,
    support_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Per-cell Shannon entropy across samples.

    ``batch`` shape (K, N, N). Returns ``H`` of shape (N, N).
    """
    k, n, _ = batch.shape
    flat = batch.reshape(k, n * n)
    ent = np.zeros(n * n, dtype=np.float64)
    for c in range(n * n):
        col = flat[:, c]
        if col.max() - col.min() < 1e-12:
            continue
        hist, _ = np.histogram(col, bins=n_bins, density=False)
        p = hist.astype(np.float64)
        p = p[p > 0]
        p /= p.sum()
        ent[c] = -np.sum(p * np.log(p + 1e-30))
    h = ent.reshape(n, n)
    if support_mask is not None:
        h = np.where(support_mask, h, 0.0)
    else:
        np.fill_diagonal(h, 0.0)
    return h


def passes_per_matrix_filters(
    candidate: np.ndarray,
    base_od: np.ndarray,
    distance: np.ndarray,
    base_trip_length: float,
    config: QualityFilterConfig,
) -> tuple[bool, str]:
    """Return (ok, reason) for single-matrix hard filters."""
    if config.require_zero_diagonal and float(np.sum(np.diag(candidate))) > config.max_diagonal_mass:
        return False, "diagonal"

    spars = matrix_sparsity(candidate)
    if spars < config.min_sparsity or spars > config.max_sparsity:
        return False, "sparsity"

    corr = _safe_corr(base_od, candidate)
    if not np.isfinite(corr) or corr < config.min_correlation or corr > config.max_correlation:
        return False, "correlation"

    tl = mean_trip_length(candidate, distance)
    if not np.isfinite(tl) or not np.isfinite(base_trip_length) or base_trip_length <= 0:
        return False, "trip_length"
    rel = abs(tl - base_trip_length) / base_trip_length
    if rel > config.max_trip_length_rel_error:
        return False, "trip_length"

    return True, "ok"


def _batch_per_matrix_mask(
    candidates: np.ndarray,
    base_od: np.ndarray,
    distance: np.ndarray,
    base_trip_length: float,
    config: QualityFilterConfig,
) -> tuple[np.ndarray, dict[str, int]]:
    """
    Vectorized hard filters over a candidate pool.

    Returns
    -------
    keep : bool array shape (K,)
    reject_counts : dict with sparsity/correlation/trip_length/diagonal counts
    """
    k = candidates.shape[0]
    flat = candidates.reshape(k, -1)
    base_flat = base_od.ravel()

    keep = np.ones(k, dtype=bool)
    counts = {"sparsity": 0, "correlation": 0, "trip_length": 0, "diagonal": 0}

    if config.require_zero_diagonal:
        diag_mass = np.einsum("kii->k", candidates)
        bad = diag_mass > config.max_diagonal_mass
        counts["diagonal"] = int(bad.sum())
        keep &= ~bad

    # Sparsity: fraction of near-zeros
    spars = np.mean(candidates <= 1e-12, axis=(1, 2))
    bad = (spars < config.min_sparsity) | (spars > config.max_sparsity)
    counts["sparsity"] = int((keep & bad).sum())
    keep &= ~bad

    # Correlation with base (vectorized pearson)
    base_c = base_flat - base_flat.mean()
    base_norm = np.linalg.norm(base_c)
    centered = flat - flat.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1)
    valid = (norms > 1e-12) & (base_norm > 1e-12)
    corr = np.full(k, np.nan)
    corr[valid] = (centered[valid] @ base_c) / (norms[valid] * base_norm)
    bad = (~np.isfinite(corr)) | (corr < config.min_correlation) | (corr > config.max_correlation)
    counts["correlation"] = int((keep & bad).sum())
    keep &= ~bad

    # Trip length
    mass = candidates.sum(axis=(1, 2))
    tl = np.full(k, np.nan)
    positive = mass > 0
    tl[positive] = np.sum(candidates[positive] * distance[None, :, :], axis=(1, 2)) / mass[positive]
    if np.isfinite(base_trip_length) and base_trip_length > 0:
        rel = np.abs(tl - base_trip_length) / base_trip_length
        bad = (~np.isfinite(rel)) | (rel > config.max_trip_length_rel_error)
    else:
        bad = np.ones(k, dtype=bool)
    counts["trip_length"] = int((keep & bad).sum())
    keep &= ~bad

    return keep, counts


def select_diverse_subset(
    candidates: np.ndarray,
    base_od: np.ndarray,
    target_size: int,
    config: QualityFilterConfig | None = None,
    distance_matrix: np.ndarray | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, FilterStats]:
    """
    Apply quality filters and greedily select a diverse subset.

    Strategy
    --------
    1. Soft-rank surviving candidates by entropy-gain proxy (Constraint 3).
    2. Greedily accept if pairwise Frobenius distance to all accepted
       matrices exceeds ``min_frobenius_distance`` (Constraint 4).
    """
    config = config or QualityFilterConfig()
    candidates = np.asarray(candidates, dtype=np.float64)
    base_od = np.asarray(base_od, dtype=np.float64)
    n_cand = candidates.shape[0]

    if distance_matrix is None:
        distance_matrix = zone_distance_matrix()
    base_tl = mean_trip_length(base_od, distance_matrix)

    stats = FilterStats(n_candidates=n_cand, n_accepted=0)
    keep, reject_counts = _batch_per_matrix_mask(
        candidates, base_od, distance_matrix, base_tl, config
    )
    stats.n_rejected_sparsity = reject_counts["sparsity"]
    stats.n_rejected_correlation = reject_counts["correlation"]
    stats.n_rejected_trip_length = reject_counts["trip_length"]
    stats.n_rejected_diagonal = reject_counts["diagonal"]
    survivors = np.flatnonzero(keep).tolist()

    if not survivors:
        logger.warning("No candidates passed per-matrix filters")
        empty = np.zeros((0, *candidates.shape[1:]), dtype=candidates.dtype)
        return empty, stats

    surv = candidates[survivors]
    flat = surv.reshape(len(survivors), -1)

    min_d = config.min_frobenius_distance
    if min_d <= 0:
        min_d = _auto_frobenius_threshold(
            flat, config.auto_frobenius_fraction, seed=seed
        )
        logger.info("Auto Frobenius diversity threshold: %.2f", min_d)

    # Rank by entropy-gain proxy (Constraint 3): prefer high deviation from survivor mean
    mean_surv = flat.mean(axis=0)
    scores = np.mean(np.abs(flat - mean_surv), axis=1)
    order = np.argsort(-scores)  # descending

    # Negligible pairwise threshold → take top-scoring survivors (avoids O(n²)).
    # For large accept sets, check each candidate against a capped reservoir of
    # already-accepted matrices (approximate diversity, O(n · R)).
    pairwise_reservoir = 256
    skip_pairwise = min_d <= 1e-6

    accepted_idx: list[int] = []
    accepted_flat: list[np.ndarray] = []

    if skip_pairwise:
        take = order[: min(target_size, len(order))]
        accepted_idx = take.tolist()
    else:
        for local in order:
            if len(accepted_idx) >= target_size:
                break
            cand = flat[local]
            if accepted_flat:
                refs = accepted_flat[-pairwise_reservoir:]
                ref_mat = np.stack(refs, axis=0)
                dists = np.linalg.norm(ref_mat - cand[None, :], axis=1)
                if float(dists.min()) < min_d:
                    stats.n_rejected_pairwise += 1
                    continue
            accepted_idx.append(int(local))
            accepted_flat.append(cand)

    if len(accepted_idx) < target_size:
        stats.n_rejected_pool_exhausted = target_size - len(accepted_idx)
        logger.warning(
            "Accepted %d / %d requested (pool exhausted after filters)",
            len(accepted_idx),
            target_size,
        )

    selected = surv[accepted_idx]
    stats.n_accepted = selected.shape[0]

    if stats.n_accepted >= 2:
        sel_flat = selected.reshape(stats.n_accepted, -1)
        # Sample pairwise Frobenius for reporting
        rng = np.random.default_rng(seed)
        n_pairs = min(5_000, stats.n_accepted * (stats.n_accepted - 1) // 2)
        i = rng.integers(0, stats.n_accepted, size=n_pairs)
        j = rng.integers(0, stats.n_accepted, size=n_pairs)
        same = i == j
        while same.any():
            j[same] = rng.integers(0, stats.n_accepted, size=int(same.sum()))
            same = i == j
        pair_d = np.linalg.norm(sel_flat[i] - sel_flat[j], axis=1)
        stats.mean_pairwise_frobenius = float(pair_d.mean())
        stats.min_pairwise_frobenius = float(pair_d.min())

        cell_mean = sel_flat.mean(axis=0)
        cell_std = sel_flat.std(axis=0)
        stats.mean_cell_cv = float(np.mean(cell_std / (cell_mean + 1e-9)))

        # Entropy on a subsample for speed at large K
        ent_sample = selected
        if stats.n_accepted > 5_000:
            idx = rng.choice(stats.n_accepted, size=5_000, replace=False)
            ent_sample = selected[idx]
        h = cell_wise_entropy(ent_sample, n_bins=config.entropy_bins)
        off = ~np.eye(selected.shape[1], dtype=bool)
        stats.mean_cell_entropy = float(h[off].mean()) if off.any() else float("nan")

    logger.info(
        "Quality filter: candidates=%d survivors=%d accepted=%d | "
        "reject sparsity=%d corr=%d trip=%d diag=%d pairwise=%d | "
        "H̄=%.3f CV̄=%.3f F̄=%.1f",
        n_cand,
        len(survivors),
        stats.n_accepted,
        stats.n_rejected_sparsity,
        stats.n_rejected_correlation,
        stats.n_rejected_trip_length,
        stats.n_rejected_diagonal,
        stats.n_rejected_pairwise,
        stats.mean_cell_entropy,
        stats.mean_cell_cv,
        stats.mean_pairwise_frobenius,
    )
    return selected.astype(candidates.dtype, copy=False), stats


def filter_candidate_pool(
    candidates: np.ndarray,
    base_od: np.ndarray,
    target_size: int,
    config: QualityFilterConfig | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, FilterStats]:
    """Public entry: select ``target_size`` matrices from a candidate pool."""
    return select_diverse_subset(
        candidates,
        base_od,
        target_size=target_size,
        config=config,
        seed=seed,
    )
