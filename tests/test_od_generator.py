"""Unit tests for Phase 1 synthetic OD generation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dirichlet_sampler import reconstruct_demand, sample_dirichlet_probability
from src.data.ipf import ipf
from src.data.marginal_perturbation import perturb_marginals
from src.data.od_analysis import analyze_base_od
from src.data.od_generator import (
    GeneratorConfig,
    apply_sparsity_mask,
    enforce_zero_diagonal,
    generate_one_synthetic_od,
    generate_synthetic_od_batch,
)
from src.data.od_metrics import compute_matrix_metrics
from src.data.od_probability import (
    build_base_probability,
    distance_dependent_epsilon,
    full_off_diagonal_support,
    zone_distance_matrix,
)

BASE_OD_PATH = PROJECT_ROOT / "EstimatedODMatrix.npy"


@pytest.fixture(scope="module")
def base_od() -> np.ndarray:
    od = np.load(BASE_OD_PATH).astype(float)
    np.fill_diagonal(od, 0.0)
    return od


@pytest.fixture(scope="module")
def base_dist(base_od):
    return build_base_probability(base_od, beta=1.0, lambda_decay=500.0)


def test_base_probability_sums_to_one(base_dist):
    assert np.isclose(base_dist.probability_vector.sum(), 1.0)
    assert base_dist.probability_vector.size == 576


def test_full_off_diagonal_support(base_dist):
    expected = full_off_diagonal_support(24)
    assert base_dist.support_mask.shape == (24, 24)
    assert np.array_equal(base_dist.support_mask, expected)
    assert int(base_dist.support_mask.sum()) == 24 * 23
    assert np.all(np.diag(base_dist.support_mask) == False)  # noqa: E712


def test_distance_dependent_epsilon_decays_with_distance():
    dist = zone_distance_matrix()
    eps = distance_dependent_epsilon(dist, beta=1.0, lambda_decay=500.0)
    assert eps.shape == (24, 24)
    assert np.allclose(np.diag(eps), 0.0)
    # Nearest off-diagonal pairs should get larger epsilon than farthest
    i, j = np.unravel_index(np.argmin(np.where(dist > 0, dist, np.inf)), dist.shape)
    k, l = np.unravel_index(np.argmax(dist), dist.shape)
    assert eps[i, j] > eps[k, l]


def test_dirichlet_sample_on_support(base_dist):
    rng = np.random.default_rng(0)
    sample = sample_dirichlet_probability(base_dist, alpha=500, rng=rng)
    assert np.isclose(sample.probability_vector.sum(), 1.0)
    off_support = ~base_dist.support_mask.ravel(order="C")
    assert np.all(sample.probability_vector[off_support] == 0)
    on_support = base_dist.support_mask.ravel(order="C")
    # Dirichlet may underflow tiny concentrations to ~0; most mass stays on support
    assert sample.probability_vector[on_support].sum() > 0.999
    assert (sample.probability_vector[on_support] > 0).mean() > 0.9


def test_reconstruct_demand_preserves_total(base_dist):
    rng = np.random.default_rng(1)
    sample = sample_dirichlet_probability(base_dist, alpha=500, rng=rng)
    matrix = reconstruct_demand(sample, base_dist.total_demand, base_dist.num_zones)
    assert np.isclose(matrix.sum(), base_dist.total_demand)


def test_perturb_marginals_conserve_total(base_od):
    rng = np.random.default_rng(2)
    p = base_od.sum(axis=1)
    a = base_od.sum(axis=0)
    q = base_od.sum()
    result = perturb_marginals(p, a, q, 0.20, rng)
    assert np.isclose(result.target_productions.sum(), q)
    assert np.isclose(result.target_attractions.sum(), q)


def test_ipf_converges(base_od):
    p = base_od.sum(axis=1)
    a = base_od.sum(axis=0)
    seed = base_od.copy() + 1e-6 * (base_od > 0)
    result = ipf(seed, p, a, tol=1e-3)
    assert result.converged
    assert np.allclose(result.matrix.sum(axis=1), p, atol=1e-2)
    assert np.allclose(result.matrix.sum(axis=0), a, atol=1e-2)


def test_sparsity_and_diagonal(base_od):
    support = full_off_diagonal_support(24)
    noisy = base_od.copy()
    noisy[0, 0] = 999.0
    masked = apply_sparsity_mask(noisy, support)
    assert masked[0, 0] == 0.0
    assert masked[~support].sum() == 0
    diag = enforce_zero_diagonal(noisy)
    assert diag.diagonal().sum() == 0


def test_gamma_sparse_mask_heavy_tailed():
    from src.data.sparse_mask import apply_gamma_sparse_mask

    rng = np.random.default_rng(7)
    # Uniform prior on off-diagonal
    p = np.ones((24, 24), dtype=float)
    np.fill_diagonal(p, 0.0)
    p /= p.sum()
    reweighted = apply_gamma_sparse_mask(p, rng, shape=0.5, scale=1.0)
    assert np.isclose(reweighted.sum(), 1.0)
    assert np.allclose(np.diag(reweighted), 0.0)
    # Heavy-tailed: top 5% of cells should hold a large share of mass
    flat = reweighted[reweighted > 0]
    top = np.sort(flat)[-max(1, len(flat) // 20) :]
    assert top.sum() / flat.sum() > 0.25


def test_generate_one_synthetic_od(base_od, base_dist):
    rng = np.random.default_rng(3)
    config = GeneratorConfig(alpha=500, perturbation=0.20, apply_gamma_mask=True)
    result = generate_one_synthetic_od(base_od, base_dist, config, rng)
    syn = result.matrix
    assert syn.shape == (24, 24)
    assert syn.diagonal().sum() == 0
    assert np.all(syn[base_dist.support_mask] >= 0)
    assert np.all(syn[~base_dist.support_mask] == 0)
    assert np.isclose(syn.sum(), base_od.sum(), rtol=0.01)


def test_prior_boosts_nearby_relative_to_far(base_od):
    """Distance-dependent ε should raise near OD relative probability vs far OD."""
    dist = zone_distance_matrix()
    flat = build_base_probability(base_od, beta=0.0, lambda_decay=500.0)
    gravity = build_base_probability(base_od, beta=50.0, lambda_decay=200.0)
    # Pick a near and far pair among originally zero (or small) cells if any;
    # otherwise compare ratios on the nearest vs farthest off-diagonal pair.
    i_near, j_near = np.unravel_index(np.argmin(np.where(dist > 0, dist, np.inf)), dist.shape)
    i_far, j_far = np.unravel_index(np.argmax(dist), dist.shape)
    p_flat = flat.probability_vector.reshape(24, 24)
    p_grav = gravity.probability_vector.reshape(24, 24)
    ratio_flat = p_flat[i_near, j_near] / (p_flat[i_far, j_far] + 1e-15)
    ratio_grav = p_grav[i_near, j_near] / (p_grav[i_far, j_far] + 1e-15)
    assert ratio_grav >= ratio_flat


def test_higher_alpha_closer_correlation(base_od, base_dist):
    rng_low = np.random.default_rng(10)
    rng_high = np.random.default_rng(11)
    low = generate_one_synthetic_od(
        base_od, base_dist, GeneratorConfig(alpha=100), rng_low
    ).matrix
    high = generate_one_synthetic_od(
        base_od, base_dist, GeneratorConfig(alpha=2000), rng_high
    ).matrix
    corr_low = compute_matrix_metrics(base_od, low).correlation
    corr_high = compute_matrix_metrics(base_od, high).correlation
    assert corr_high > corr_low


def test_analyze_base_od(base_od):
    analysis = analyze_base_od(base_od)
    assert analysis.num_zones == 24
    assert analysis.total_demand > 0
    assert len(analysis.top_10_pairs) == 10


def test_quality_filters_select_diverse_subset(base_od, base_dist):
    from src.data.quality_filters import QualityFilterConfig, filter_candidate_pool

    config = GeneratorConfig(alpha=500, perturbation=0.0, apply_gamma_mask=True)
    pool = generate_synthetic_od_batch(
        base_od, 200, config, seed=5, workers=1, show_progress=False
    ).matrices
    selected, stats = filter_candidate_pool(
        pool,
        base_od,
        target_size=20,
        config=QualityFilterConfig(
            oversample_factor=10.0,
            min_correlation=0.2,
            max_correlation=0.99,
            auto_frobenius_fraction=0.2,
        ),
        seed=5,
    )
    assert selected.shape[0] > 0
    assert selected.shape[0] <= 20
    assert stats.n_accepted == selected.shape[0]
    assert stats.n_candidates == 200
    if selected.shape[0] >= 2:
        assert np.isfinite(stats.mean_cell_entropy)
        assert stats.min_pairwise_frobenius >= 0


def test_parallel_matches_sequential(base_od):
    config = GeneratorConfig(alpha=500, perturbation=0.20)
    seq = generate_synthetic_od_batch(
        base_od, 32, config, seed=99, workers=1, show_progress=False
    ).matrices
    par = generate_synthetic_od_batch(
        base_od, 32, config, seed=99, workers=4, show_progress=False
    ).matrices
    np.testing.assert_allclose(seq, par, rtol=0, atol=0)

