"""Unit tests for Phase 2 probabilistic turning-count pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.fractional_assignment import (
    build_fractional_assignment_matrix,
    validate_fractional_matrix,
)
from src.data.k_shortest_paths import enumerate_k_shortest_paths
from src.data.network_parser import parse_sumo_network
from src.data.observation_noise import NoiseConfig, apply_observation_noise
from src.data.od_pairs import OdPairIndex, flatten_od_matrix
from src.data.route_choice import apply_logit_choice, logit_route_probabilities
from src.data.statistics import generate_turning_counts
from src.data.turning_movements import enumerate_turning_movements

NET_PATH = PROJECT_ROOT / "sioux-falls.net.xml"


@pytest.fixture(scope="module")
def network():
    return parse_sumo_network(NET_PATH)


@pytest.fixture(scope="module")
def turning_index(network):
    return enumerate_turning_movements(network)


@pytest.fixture(scope="module")
def od_index():
    return OdPairIndex.build()


@pytest.fixture(scope="module")
def route_catalog(network, od_index):
    return enumerate_k_shortest_paths(network, od_index, k_paths=5)


@pytest.fixture(scope="module")
def route_choices(route_catalog):
    return apply_logit_choice(route_catalog, theta=0.1)


@pytest.fixture(scope="module")
def a_turn_frac(turning_index, od_index, route_choices):
    return build_fractional_assignment_matrix(
        turning_index, od_index, route_choices
    )


def test_logit_probabilities_sum_to_one(route_catalog):
    routes = route_catalog.get(1, 24)
    assert len(routes) >= 1
    probs = logit_route_probabilities(routes, theta=0.1)
    total = sum(rp.probability for rp in probs)
    assert abs(total - 1.0) < 1e-9


def test_fractional_matrix_bounds(a_turn_frac):
    assert a_turn_frac.shape == (178, 576)
    validation = validate_fractional_matrix(a_turn_frac)
    assert validation["entries_in_01"] is True
    assert validation["max_entry"] <= 1.0 + 1e-9


def test_fractional_rank_higher_than_binary(network, turning_index, od_index, a_turn_frac):
    from src.data.assignment_rank import compare_assignment_matrices
    from src.data.build_assignment_matrix import build_assignment_matrix
    from src.data.route_assignment import assign_routes

    binary_routes = assign_routes(network, od_index, weight_metric="length")
    a_binary = build_assignment_matrix(turning_index, od_index, binary_routes)
    comparison = compare_assignment_matrices(a_binary, a_turn_frac)
    assert comparison.probabilistic.result.rank >= comparison.binary.result.rank


def test_matrix_multiplication(a_turn_frac, route_choices, turning_index):
    od = np.zeros((24, 24))
    od[0, 23] = 50.0
    y = flatten_od_matrix(od)
    x = generate_turning_counts(y.reshape(1, -1), a_turn_frac)[0]

    expected = np.zeros(178)
    for rp in route_choices.choices[(1, 24)]:
        for inc, out in zip(rp.route.path_edges[:-1], rp.route.path_edges[1:]):
            tid = turning_index.id_for(inc, out)
            if tid is not None:
                expected[tid] += rp.probability * 50.0

    np.testing.assert_allclose(x, a_turn_frac @ y, rtol=1e-9)
    np.testing.assert_allclose(x, expected, rtol=1e-9)


def test_poisson_noise_nonnegative():
    clean = np.array([[10.0, 20.0, 0.0]])
    rng = np.random.default_rng(0)
    noisy = apply_observation_noise(clean, NoiseConfig(model="poisson"), rng)
    assert noisy.min() >= 0


def test_no_noise_identity():
    clean = np.array([[1.5, 2.5]])
    rng = np.random.default_rng(0)
    out = apply_observation_noise(clean, NoiseConfig(model="none"), rng)
    np.testing.assert_array_equal(out, clean)


def test_batched_generation(tmp_path):
    from src.data.batch_turning import generate_turning_counts_batched

    od = np.random.default_rng(0).random((500, 24, 24)).astype(np.float32)
    od_path = tmp_path / "od.npy"
    np.save(od_path, od)

    # Minimal A_turn: identity-like mapping for test
    a_turn = np.zeros((178, 576), dtype=np.float32)
    a_turn[0, 0] = 1.0

    result = generate_turning_counts_batched(
        od_path, a_turn, tmp_path, batch_size=100, show_progress=False
    )
    out = np.load(tmp_path / "turning_counts.npy", mmap_mode="r")
    assert out.shape == (500, 178)
    assert result["matrix_mult_consistent"]
