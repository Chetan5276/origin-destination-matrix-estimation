"""Stage 4: K-shortest path enumeration for all OD pairs."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import networkx as nx

from src.data.network_parser import SumoNetwork, WeightAttribute
from src.data.od_pairs import OdPairIndex

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RoutePath:
    """One feasible route between an OD pair."""

    origin: int
    destination: int
    path_nodes: list[str]
    path_edges: list[str]
    path_cost: float


@dataclass
class RouteCatalog:
    """K-shortest routes and metadata for every OD pair."""

    routes: dict[tuple[int, int], list[RoutePath]]
    k_paths: int
    weight_metric: WeightAttribute
    unreachable_pairs: list[tuple[int, int]] = field(default_factory=list)

    def get(self, origin: int, destination: int) -> list[RoutePath]:
        return self.routes.get((origin, destination), [])


def _node_path_to_edges(network: SumoNetwork, node_path: list[str]) -> tuple[list[str], float]:
    edges: list[str] = []
    cost = 0.0
    weight_key = "length"  # caller passes metric via graph edge attrs
    for u, v in zip(node_path[:-1], node_path[1:]):
        data = network.graph[u][v]
        edges.append(data["edge_id"])
        cost += float(data["length"])  # default; overridden below
    return edges, cost


def _path_cost(network: SumoNetwork, edges: list[str], metric: WeightAttribute) -> float:
    total = 0.0
    for eid in edges:
        info = network.edges[eid]
        total += info.length if metric == "length" else info.time
    return total


def enumerate_k_shortest_paths(
    network: SumoNetwork,
    od_index: OdPairIndex,
    k_paths: int = 5,
    weight_metric: WeightAttribute = "length",
    path_limit: int = 50,
) -> RouteCatalog:
    """
    Enumerate up to ``k_paths`` simple shortest paths per OD pair.

    Uses ``networkx.shortest_simple_paths`` which yields paths in
    non-decreasing cost order.
    """
    logger.info(
        "Enumerating up to K=%d shortest paths per OD pair (metric=%s)",
        k_paths,
        weight_metric,
    )
    routes: dict[tuple[int, int], list[RoutePath]] = {}
    unreachable: list[tuple[int, int]] = []

    for origin, destination in od_index.od_pair_to_index:
        if origin == destination:
            routes[(origin, destination)] = []
            continue

        source = network.zone_to_junction[origin]
        target = network.zone_to_junction[destination]
        pair_routes: list[RoutePath] = []

        try:
            path_gen = nx.shortest_simple_paths(
                network.graph,
                source,
                target,
                weight=weight_metric,
            )
            for idx, node_path in enumerate(path_gen):
                if idx >= k_paths:
                    break
                if idx >= path_limit:
                    break
                edges = []
                for u, v in zip(node_path[:-1], node_path[1:]):
                    edges.append(network.graph[u][v]["edge_id"])
                cost = _path_cost(network, edges, weight_metric)
                pair_routes.append(
                    RoutePath(
                        origin=origin,
                        destination=destination,
                        path_nodes=list(node_path),
                        path_edges=edges,
                        path_cost=cost,
                    )
                )
        except nx.NetworkXNoPath:
            unreachable.append((origin, destination))

        routes[(origin, destination)] = pair_routes

    total_routes = sum(len(v) for v in routes.values())
    logger.info(
        "Route catalog: %d OD pairs, %d total routes, %d unreachable",
        len(routes),
        total_routes,
        len(unreachable),
    )
    return RouteCatalog(
        routes=routes,
        k_paths=k_paths,
        weight_metric=weight_metric,
        unreachable_pairs=unreachable,
    )
