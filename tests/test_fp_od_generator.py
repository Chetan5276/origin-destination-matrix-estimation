"""Unit tests for first-principles OD generation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.first_principles import (
    FPGeneratorConfig,
    build_reference_od,
    generate_fp_od_batch,
    generate_one_fp_od,
)
from src.data.first_principles.compositional import dirichlet_reweight, seed_to_probability
from src.data.first_principles.gravity import build_gravity_matrix, distance_impedance
from src.data.first_principles.heavy_tails import apply_gamma_weights
from src.data.first_principles.latent import sample_latent_factors
from src.data.od_probability import (
    full_off_diagonal_support,
    load_zone_coordinates,
    zone_distance_matrix,
)


@pytest.fixture(scope="module")
def geo():
    coords = load_zone_coordinates()
    dist = zone_distance_matrix(coords=coords)
    support = full_off_diagonal_support(24)
    return coords, dist, support


def test_latent_factors_positive(geo):
    coords, dist, _ = geo
    rng = np.random.default_rng(0)
    u, v, m = sample_latent_factors(24, 4, coords, dist, rng)
    assert u.shape == (24,) and v.shape == (24,)
    assert m.shape == (24, 4)
    assert np.all(u > 0) and np.all(v > 0)
    assert np.allclose(m.sum(axis=1), 1.0)


def test_impedance_decays(geo):
    _, dist, _ = geo
    f = distance_impedance(dist, decay="exponential", lambda_decay=500.0)
    i, j = np.unravel_index(np.argmin(np.where(dist > 0, dist, np.inf)), dist.shape)
    k, l = np.unravel_index(np.argmax(dist), dist.shape)
    assert f[i, j] > f[k, l]
    assert np.allclose(np.diag(f), 0.0)


def test_gravity_and_reciprocity(geo):
    coords, dist, support = geo
    rng = np.random.default_rng(1)
    u, v, _ = sample_latent_factors(24, 3, coords, dist, rng)
    s0 = build_gravity_matrix(u, v, dist, support, reciprocity=0.0)
    s1 = build_gravity_matrix(u, v, dist, support, reciprocity=0.5)
    assert np.all(s0[~support] == 0)
    # Fully symmetrized mean structure
    assert np.allclose(s1, s1.T, atol=1e-8)


def test_gamma_heavy_tail(geo):
    _, _, support = geo
    rng = np.random.default_rng(2)
    seed = support.astype(float)
    out = apply_gamma_weights(seed, rng, support, shape=0.5, scale=1.0)
    flat = out[support]
    top = np.sort(flat)[-max(1, flat.size // 20) :]
    assert top.sum() / flat.sum() > 0.25


def test_dirichlet_on_simplex(geo):
    _, _, support = geo
    rng = np.random.default_rng(3)
    prior = support.astype(float)
    prior = seed_to_probability(prior, support)
    sample = dirichlet_reweight(prior, support, alpha=100.0, rng=rng)
    assert np.isclose(sample.sum(), 1.0)
    assert np.allclose(sample[~support], 0.0)


def test_generate_one_fp_od(geo):
    coords, dist, support = geo
    rng = np.random.default_rng(4)
    cfg = FPGeneratorConfig(total_demand=1e5, alpha=150.0, reciprocity=0.3)
    res = generate_one_fp_od(cfg, rng, distance=dist, coords=coords, support_mask=support)
    T = res.matrix
    assert T.shape == (24, 24)
    assert np.allclose(np.diag(T), 0.0)
    assert np.isclose(T.sum(), cfg.total_demand, rtol=1e-2)
    assert np.all(T >= -1e-8)


def test_reference_od():
    cfg = FPGeneratorConfig(total_demand=1e5)
    ref = build_reference_od(cfg, seed=0)
    assert ref.shape == (24, 24)
    assert np.isclose(ref.sum(), cfg.total_demand, rtol=1e-2)


def test_batch_parallel_matches_sequential():
    cfg = FPGeneratorConfig(total_demand=5e4, alpha=100.0)
    seq = generate_fp_od_batch(16, cfg, seed=7, workers=1, show_progress=False).matrices
    par = generate_fp_od_batch(16, cfg, seed=7, workers=4, show_progress=False).matrices
    np.testing.assert_allclose(seq, par, rtol=0, atol=0)
