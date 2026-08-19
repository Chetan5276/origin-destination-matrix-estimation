"""Shortest-path route assignment between OD zones."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import networkx as nx

from src.data.network_parser import SumoNetwork, WeightAttribute
from src.data.od_pairs import OdPairIndex

logger = logging.getLogger(__name__)


@dataclass
class RouteAssignment:
    """Shortest-path routes for all zone OD pairs."""

    routes: dict[tuple[int, int], list[str]]
    weight_metric: WeightAttribute
    unreachable_pairs: list[tuple[int, int]]

    def path_edges(self, origin: int, destination: int) -> list[str]:
        return self.routes.get((origin, destination), [])

    def path_turns(self, origin: int, destination: int) -> list[tuple[str, str]]:
        edges = self.path_edges(origin, destination)
        return list(zip(edges[:-1], edges[1:]))


def _path_to_edge_list(
    network: SumoNetwork,
    node_path: list[str],
) -> list[str]:
    edge_ids: list[str] = []
    for from_node, to_node in zip(node_path[:-1], node_path[1:]):
        edge_ids.append(network.graph[from_node][to_node]["edge_id"])
    return edge_ids


def assign_routes(
    network: SumoNetwork,
    od_index: OdPairIndex,
    weight_metric: WeightAttribute = "length",
) -> RouteAssignment:
    """
    Compute shortest paths for every zone OD pair.

    Parameters
    ----------
    weight_metric:
        ``'length'`` (meters) or ``'time'`` (length / speed).
    """
    logger.info("Assigning routes using shortest-path metric=%s", weight_metric)
    routes: dict[tuple[int, int], list[str]] = {}
    unreachable: list[tuple[int, int]] = []

    for origin, destination in od_index.od_pair_to_index:
        from_junction = network.zone_to_junction[origin]
        to_junction = network.zone_to_junction[destination]

        if origin == destination:
            routes[(origin, destination)] = []
            continue

        try:
            node_path = nx.shortest_path(
                network.graph,
                from_junction,
                to_junction,
                weight=weight_metric,
            )
        except nx.NetworkXNoPath:
            logger.warning("No path for OD (%d, %d)", origin, destination)
            routes[(origin, destination)] = []
            unreachable.append((origin, destination))
            continue

        routes[(origin, destination)] = _path_to_edge_list(network, node_path)

    logger.info(
        "Assigned routes for %d OD pairs (%d unreachable)",
        len(routes),
        len(unreachable),
    )
    return RouteAssignment(
        routes=routes,
        weight_metric=weight_metric,
        unreachable_pairs=unreachable,
    )
