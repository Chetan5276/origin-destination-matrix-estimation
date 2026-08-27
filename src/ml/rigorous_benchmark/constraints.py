"""Constraint strategies and ablation helpers (no silent clipping)."""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np

from src import NUM_ZONES
from src.data.ipf import ipf
from src.ml.od_constraints import apply_od_constraints_numpy, zero_diagonal_flat

logger = logging.getLogger(__name__)

ConstraintStrategy = Literal["none", "relu", "softplus", "clip", "clip_ipf"]


def apply_constraint_strategy(
    y_flat: np.ndarray,
    *,
    strategy: str,
    support_mask: np.ndarray | None = None,
    target_productions: np.ndarray | None = None,
    target_attractions: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Apply a named constraint strategy and return (y_out, metadata).

    Strategies
    ----------
    none:
        Pass-through (may contain negatives / diagonal mass).
    relu:
        max(y, 0), zero diagonal, optional support mask.
    softplus:
        softplus(y) via log1p(exp) for stability, zero diagonal, mask.
    clip:
        Same as legacy apply_od_constraints_numpy (clip negatives).
    clip_ipf:
        clip then optional IPF to production/attraction targets when provided.
    """
    y = np.asarray(y_flat, dtype=np.float64)
    meta = {"constraint_strategy": strategy, "clipped": False, "ipf_applied": False}

    if strategy == "none":
        return y.astype(np.float32), meta

    if strategy == "softplus":
        # Numerically stable softplus
        out = np.where(y > 20, y, np.log1p(np.exp(np.clip(y, -20, 20))))
        out = zero_diagonal_flat(out)
        if support_mask is not None:
            out = out * support_mask.astype(float)
        meta["clipped"] = False
        return out.astype(np.float32), meta

    if strategy == "relu":
        out = np.maximum(y, 0.0)
        out = zero_diagonal_flat(out)
        if support_mask is not None:
            out = out * support_mask.astype(float)
        meta["note"] = "relu = max(0,·); not silent post-hoc clip of a different model"
        return out.astype(np.float32), meta

    if strategy in ("clip", "clip_ipf"):
        out = apply_od_constraints_numpy(y, support_mask=support_mask, zero_diagonal=True)
        meta["clipped"] = True
        if strategy == "clip_ipf" and target_productions is not None and target_attractions is not None:
            n = out.shape[0]
            mats = out.reshape(n, NUM_ZONES, NUM_ZONES).copy()
            for i in range(n):
                tp = target_productions[i]
                ta = target_attractions[i]
                res = ipf(mats[i], tp, ta, tol=1e-2, max_iter=50)
                mats[i] = res.matrix
            out = mats.reshape(n, -1)
            meta["ipf_applied"] = True
        return out.astype(np.float32), meta

    raise ValueError(f"Unknown constraint strategy: {strategy}")


ABLATION_STRATEGIES: tuple[str, ...] = ("none", "relu", "softplus", "clip", "clip_ipf")
