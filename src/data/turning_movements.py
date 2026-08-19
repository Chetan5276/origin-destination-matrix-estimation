"""Enumerate valid turning movements on the Sioux Falls network."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.data.network_parser import SumoNetwork

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TurningMovement:
    """A single incoming-edge → outgoing-edge turn at a junction."""

    incoming_edge: str
    outgoing_edge: str
    junction: str
    turn_id: int


@dataclass
class TurningMovementIndex:
    """Index of all valid turning movements."""

    movements: list[TurningMovement]
    turning_id_map: dict[tuple[str, str], int]
    num_turning_movements: int

    def id_for(self, incoming_edge: str, outgoing_edge: str) -> int | None:
        return self.turning_id_map.get((incoming_edge, outgoing_edge))


def enumerate_turning_movements(network: SumoNetwork) -> TurningMovementIndex:
    """
    Identify turning movements as valid junction turns.

  A turn is an incoming edge → outgoing edge pair at the same junction,
  excluding U-turns on the same directed edge, and restricted to turns
  explicitly allowed by SUMO ``<connection>`` elements. For Sioux Falls this
  yields 178 movements.
    """
    valid_connections = set(network.connections)
    candidate_turns: set[tuple[str, str, str]] = set()

    for junction in network.junctions:
        incoming = [
            edge_id
            for edge_id, info in network.edges.items()
            if info.to_junction == junction
        ]
        outgoing = [
            edge_id
            for edge_id, info in network.edges.items()
            if info.from_junction == junction
        ]
        for inc in incoming:
            for out in outgoing:
                if inc == out:
                    continue
                if (inc, out) in valid_connections:
                    candidate_turns.add((inc, out, junction))

    sorted_turns = sorted(candidate_turns, key=lambda t: (t[2], t[0], t[1]))
    movements: list[TurningMovement] = []
    turning_id_map: dict[tuple[str, str], int] = {}

    for turn_id, (inc, out, junction) in enumerate(sorted_turns):
        movements.append(
            TurningMovement(
                incoming_edge=inc,
                outgoing_edge=out,
                junction=junction,
                turn_id=turn_id,
            )
        )
        turning_id_map[(inc, out)] = turn_id

    index = TurningMovementIndex(
        movements=movements,
        turning_id_map=turning_id_map,
        num_turning_movements=len(movements),
    )
    logger.info("Enumerated %d turning movements", index.num_turning_movements)
    return index


def export_turning_movements_csv(
    turning_index: TurningMovementIndex,
    output_path: "Path",
) -> None:
    """Write turning movement catalog to CSV."""
    from pathlib import Path

    import pandas as pd

    rows = [
        {
            "turn_id": m.turn_id,
            "incoming_edge": m.incoming_edge,
            "outgoing_edge": m.outgoing_edge,
            "junction": m.junction,
        }
        for m in turning_index.movements
    ]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
