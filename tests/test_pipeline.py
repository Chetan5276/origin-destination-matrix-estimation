"""Unit tests for the OD → turning-count pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import NETWORK_PATH
from src.data.build_assignment_matrix import build_assignment_matrix
from src.data.network_parser import parse_sumo_network
from src.data.od_pairs import OdPairIndex, flatten_od_matrix, unflatten_od_vector
from src.data.route_assignment import assign_routes
from src.data.statistics import build_dataset, generate_turning_counts, validate_dataset
from src.data.turning_movements import enumerate_turning_movements

NET_PATH = NETWORK_PATH


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
def routes(network, od_index):
    return assign_routes(network, od_index, weight_metric="length")


@pytest.fixture(scope="module")
def a_turn(turning_index, od_index, routes):
    return build_assignment_matrix(turning_index, od_index, routes)


def test_turning_movement_count(turning_index):
    assert turning_index.num_turning_movements == 178


def test_od_pair_count(od_index):
    assert od_index.num_od_pairs == 576
    assert od_index.flat_index(1, 1) == 0
    assert od_index.flat_index(24, 24) == 575


def test_route_generation(routes):
    path = routes.path_edges(1, 24)
    assert len(path) >= 1
    assert all(isinstance(e, str) for e in path)


def test_route_turns_are_valid(turning_index, routes):
    turns = routes.path_turns(1, 24)
    assert len(turns) == len(routes.path_edges(1, 24)) - 1
    for inc, out in turns:
        assert (inc, out) in turning_index.turning_id_map


def test_assignment_matrix_shape(a_turn):
    assert a_turn.shape == (178, 576)
    assert np.all((a_turn == 0) | (a_turn == 1))


def test_assignment_matrix_populated(a_turn):
    assert np.count_nonzero(a_turn) > 0
    assert a_turn.sum(axis=0).max() >= 1


def test_matrix_multiplication_correctness(a_turn, turning_index, routes):
    """X = A_turn @ Y for a single OD matrix with one nonzero cell."""
    od = np.zeros((24, 24))
    origin, destination = 1, 24
    od[origin - 1, destination - 1] = 100.0
    y = flatten_od_matrix(od)
    x = generate_turning_counts(y.reshape(1, -1), a_turn)[0]

    expected = np.zeros(178)
    for inc, out in routes.path_turns(origin, destination):
        row = turning_index.turning_id_map[(inc, out)]
        expected[row] = 100.0

    np.testing.assert_allclose(x, a_turn @ y)
    np.testing.assert_allclose(x, expected)


def test_flatten_unflatten_roundtrip():
    od = np.arange(576).reshape(24, 24).astype(float)
    vec = flatten_od_matrix(od)
    restored = unflatten_od_vector(vec)
    np.testing.assert_array_equal(od, restored)


def test_build_dataset_shapes(a_turn):
    od_batch = np.random.default_rng(0).random((5, 24, 24))
    dataset = build_dataset(od_batch, a_turn)
    assert dataset.x.shape == (5, 178)
    assert dataset.y.shape == (5, 576)
    checks = validate_dataset(dataset)
    assert checks["matrix_mult_consistent"] is True
