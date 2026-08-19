"""Physical constraints on predicted OD matrices."""

from __future__ import annotations

import numpy as np

from src import NUM_OD_PAIRS, NUM_ZONES

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


def base_support_mask(base_od: np.ndarray | None = None) -> np.ndarray:
    """
    Boolean mask (576,) for allowed OD cells.

    Full off-diagonal support: S = {(i, j) : i ≠ j}.
    ``base_od`` is accepted for API compatibility but no longer restricts support
    to positive base cells.
    """
    mask = np.ones((NUM_ZONES, NUM_ZONES), dtype=bool)
    np.fill_diagonal(mask, False)
    return mask.reshape(NUM_OD_PAIRS)


def zero_diagonal_flat(y_flat: np.ndarray) -> np.ndarray:
    """Zero intrazonal flows in flattened OD batch."""
    out = np.asarray(y_flat, dtype=float).copy()
    n = out.shape[0]
    mat = out.reshape(n, NUM_ZONES, NUM_ZONES)
    idx = np.arange(NUM_ZONES)
    mat[:, idx, idx] = 0.0
    return mat.reshape(n, NUM_OD_PAIRS)


def apply_od_constraints_numpy(
    y_flat: np.ndarray,
    *,
    support_mask: np.ndarray | None = None,
    zero_diagonal: bool = True,
) -> np.ndarray:
    """Non-negative OD with optional zero diagonal and sparsity mask."""
    y = np.clip(np.asarray(y_flat, dtype=float), 0, None)
    if zero_diagonal:
        y = zero_diagonal_flat(y)
    if support_mask is not None:
        y = y * support_mask.astype(float)
    return y


if torch is not None:

    class ODConstraintLayer(nn.Module):
        """Softplus non-negativity, zero diagonal, optional sparsity mask."""

        def __init__(
            self,
            support_mask: np.ndarray | None = None,
            zero_diagonal: bool = True,
            activation: str = "softplus",
        ) -> None:
            super().__init__()
            self.zero_diagonal = zero_diagonal
            self.activation = activation
            if support_mask is not None:
                mask = torch.tensor(support_mask.astype(np.float32))
            else:
                mask = torch.ones(NUM_OD_PAIRS, dtype=torch.float32)
            self.register_buffer("support_mask", mask)
            if zero_diagonal:
                diag = torch.ones(NUM_OD_PAIRS, dtype=torch.float32)
                for i in range(NUM_ZONES):
                    diag[i * NUM_ZONES + i] = 0.0
                self.register_buffer("diagonal_mask", diag)
            else:
                self.register_buffer("diagonal_mask", torch.ones(NUM_OD_PAIRS))

        def forward(self, y_raw: torch.Tensor) -> torch.Tensor:
            if self.activation == "softplus":
                y = F.softplus(y_raw)
            else:
                y = F.relu(y_raw)
            y = y * self.diagonal_mask * self.support_mask
            return y
