"""OD pair indexing for zone-level demand matrices."""

from __future__ import annotations

from dataclasses import dataclass

from src import NUM_ZONES


@dataclass(frozen=True)
class OdPairIndex:
    """Maps between (origin, destination) zones and flat vector indices."""

    od_pair_to_index: dict[tuple[int, int], int]
    index_to_od_pair: dict[int, tuple[int, int]]
    num_od_pairs: int = NUM_ZONES * NUM_ZONES

    @staticmethod
    def build(num_zones: int = NUM_ZONES) -> "OdPairIndex":
        od_pair_to_index: dict[tuple[int, int], int] = {}
        index_to_od_pair: dict[int, tuple[int, int]] = {}
        idx = 0
        for origin in range(1, num_zones + 1):
            for destination in range(1, num_zones + 1):
                od_pair_to_index[(origin, destination)] = idx
                index_to_od_pair[idx] = (origin, destination)
                idx += 1
        return OdPairIndex(
            od_pair_to_index=od_pair_to_index,
            index_to_od_pair=index_to_od_pair,
            num_od_pairs=num_zones * num_zones,
        )

    def flat_index(self, origin: int, destination: int) -> int:
        return self.od_pair_to_index[(origin, destination)]


def flatten_od_matrix(od_matrix) -> "np.ndarray":
    """Flatten a zone OD matrix to a length-576 vector (row-major zones)."""
    import numpy as np

    od = np.asarray(od_matrix, dtype=float)
    if od.shape != (NUM_ZONES, NUM_ZONES):
        raise ValueError(f"Expected OD shape ({NUM_ZONES}, {NUM_ZONES}), got {od.shape}")
    return od.reshape(-1, order="C")


def unflatten_od_vector(vector) -> "np.ndarray":
    """Restore a flattened OD vector to a 24×24 matrix."""
    import numpy as np

    vec = np.asarray(vector, dtype=float)
    if vec.shape != (NUM_ZONES * NUM_ZONES,):
        raise ValueError(
            f"Expected vector length {NUM_ZONES * NUM_ZONES}, got {vec.shape}"
        )
    return vec.reshape(NUM_ZONES, NUM_ZONES, order="C")
