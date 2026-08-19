"""Parse SUMO network files into a NetworkX graph representation."""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import networkx as nx

logger = logging.getLogger(__name__)

WeightAttribute = Literal["length", "time"]


@dataclass(frozen=True)
class EdgeInfo:
    """Directed network edge (SUMO regular edge)."""

    edge_id: str
    from_junction: str
    to_junction: str
    length: float
    speed: float
    time: float


@dataclass
class SumoNetwork:
    """Parsed Sioux Falls SUMO network."""

    junctions: list[str]
    edges: dict[str, EdgeInfo]
    connections: list[tuple[str, str]]
    graph: nx.DiGraph = field(repr=False)
    zone_to_junction: dict[int, str] = field(default_factory=dict)
    junction_to_zone: dict[str, int] = field(default_factory=dict)
    junction_coords: dict[str, tuple[float, float]] = field(default_factory=dict)

    @property
    def num_junctions(self) -> int:
        return len(self.junctions)

    @property
    def num_edges(self) -> int:
        return len(self.edges)


def _parse_lane_metrics(edge_elem: ET.Element) -> tuple[float, float]:
    lane = edge_elem.find("lane")
    if lane is None:
        raise ValueError(f"Edge {edge_elem.get('id')} has no lane element")
    length = float(lane.get("length"))
    speed = float(lane.get("speed"))
    if speed <= 0:
        raise ValueError(f"Edge {edge_elem.get('id')} has non-positive speed")
    return length, speed


def parse_sumo_network(net_path: Path) -> SumoNetwork:
    """Parse a SUMO ``.net.xml`` file into structured objects and a DiGraph."""
    logger.info("Parsing SUMO network from %s", net_path)
    root = ET.parse(net_path).getroot()

    junction_coords: dict[str, tuple[float, float]] = {}
    junctions: list[str] = []
    for j_elem in root.findall("junction"):
        jid = j_elem.get("id", "")
        if jid.startswith(":"):
            continue
        junctions.append(jid)
        junction_coords[jid] = (float(j_elem.get("x", 0.0)), float(j_elem.get("y", 0.0)))
    junctions = sorted(junctions, key=lambda jid: int(jid[1:]))

    edges: dict[str, EdgeInfo] = {}
    graph = nx.DiGraph()

    for edge_elem in root.findall("edge"):
        if edge_elem.get("function") == "internal":
            continue
        edge_id = edge_elem.get("id")
        from_j = edge_elem.get("from")
        to_j = edge_elem.get("to")
        length, speed = _parse_lane_metrics(edge_elem)
        time = length / speed
        info = EdgeInfo(
            edge_id=edge_id,
            from_junction=from_j,
            to_junction=to_j,
            length=length,
            speed=speed,
            time=time,
        )
        edges[edge_id] = info
        graph.add_edge(
            from_j,
            to_j,
            edge_id=edge_id,
            length=length,
            time=time,
        )

    connections = [
        (conn.get("from"), conn.get("to"))
        for conn in root.findall("connection")
        if conn.get("from") and conn.get("to")
    ]

    zone_to_junction = {zone: f"J{zone}" for zone in range(1, len(junctions) + 1)}
    junction_to_zone = {junction: zone for zone, junction in zone_to_junction.items()}

    network = SumoNetwork(
        junctions=junctions,
        edges=edges,
        connections=connections,
        graph=graph,
        zone_to_junction=zone_to_junction,
        junction_to_zone=junction_to_zone,
        junction_coords=junction_coords,
    )
    logger.info(
        "Parsed %d junctions, %d edges, %d connections",
        network.num_junctions,
        network.num_edges,
        len(connections),
    )
    return network


def edge_weight(network: SumoNetwork, edge_id: str, metric: WeightAttribute) -> float:
    """Return edge weight for shortest-path routing."""
    info = network.edges[edge_id]
    return info.length if metric == "length" else info.time
