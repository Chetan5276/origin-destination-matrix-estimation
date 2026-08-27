"""SVD / pseudoinverse / null-space of A_turn."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from src import NUM_OD_PAIRS
from src.data.rank_analysis import analyze_rank

logger = logging.getLogger(__name__)


@dataclass
class OperatorInfo:
    """Cached linear-algebra facts about A_turn."""

    a_turn: np.ndarray
    u: np.ndarray
    s: np.ndarray
    vt: np.ndarray
    rank: int
    nullity: int
    a_pinv: np.ndarray
    null_basis: np.ndarray  # (n_od, nullity) columns of V for zero singular values
    condition_number: float
    effective_rank_99: int

    @property
    def n_turns(self) -> int:
        return int(self.a_turn.shape[0])

    @property
    def n_od(self) -> int:
        return int(self.a_turn.shape[1])


def compute_operator(
    a_turn: np.ndarray,
    *,
    rtol: float = 1e-10,
    od_pairs: int = NUM_OD_PAIRS,
) -> OperatorInfo:
    """Full SVD, truncated Moore–Penrose inverse, and null-space basis."""
    a = np.asarray(a_turn, dtype=np.float64)
    u, s, vt = np.linalg.svd(a, full_matrices=True)
    smax = float(s[0]) if s.size else 0.0
    thresh = rtol * smax if smax > 0 else rtol
    rank = int(np.sum(s > thresh))
    nullity = int(a.shape[1] - rank)

    # Moore–Penrose via truncated SVD (stable; no explicit invert of ATA)
    s_inv = np.zeros_like(s)
    s_inv[:rank] = 1.0 / s[:rank]
    # A+ = V @ S+ @ U.T  with shapes: V is vt.T (n_od, n_od), U is (m, m)
    a_pinv = (vt[:rank, :].T * s_inv[:rank]) @ u[:, :rank].T

    # Null space: right singular vectors for singular values ≈ 0
    # full_matrices=True → vt is (n_od, n_od); null columns are vt[rank:]
    if nullity > 0:
        null_basis = vt[rank:, :].T.copy()  # (n_od, nullity)
    else:
        null_basis = np.zeros((a.shape[1], 0), dtype=np.float64)

    positive = s[s > thresh]
    if len(positive) >= 2:
        cond = float(positive[0] / positive[-1])
    elif len(positive) == 1:
        cond = float("inf")
    else:
        cond = float("nan")

    energy = np.cumsum(s**2) / (s**2).sum() if s.size else np.array([1.0])
    eff_rank = int(np.searchsorted(energy, 0.99) + 1) if s.size else 0

    # Cross-check with existing rank_analysis
    base = analyze_rank(a, od_pairs=od_pairs)
    logger.info(
        "Operator: shape=%s svd_rank=%d analyze_rank=%d nullity=%d cond=%.4g",
        a.shape,
        rank,
        base.rank,
        nullity,
        cond,
    )
    return OperatorInfo(
        a_turn=a.astype(np.float32),
        u=u.astype(np.float64),
        s=s.astype(np.float64),
        vt=vt.astype(np.float64),
        rank=rank,
        nullity=nullity,
        a_pinv=a_pinv.astype(np.float64),
        null_basis=null_basis.astype(np.float64),
        condition_number=cond,
        effective_rank_99=eff_rank,
    )


def save_operator(info: OperatorInfo, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "singular_values.npy", info.s)
    np.save(out_dir / "a_pinv.npy", info.a_pinv.astype(np.float32))
    np.save(out_dir / "null_basis.npy", info.null_basis.astype(np.float32))
    meta = {
        "shape": list(info.a_turn.shape),
        "rank": info.rank,
        "nullity": info.nullity,
        "condition_number": info.condition_number,
        "effective_rank_99": info.effective_rank_99,
        "singular_values_head": info.s[:30].tolist(),
    }
    (out_dir / "operator_meta.json").write_text(json.dumps(meta, indent=2))
    logger.info("Wrote operator artifacts to %s", out_dir)


def pinv_predict(a_pinv: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Batched y = A+ @ x  for X shape (N, m)."""
    x = np.asarray(x, dtype=np.float64)
    return (x @ a_pinv.T).astype(np.float32)


def operator_summary_dict(info: OperatorInfo) -> dict:
    return {
        "shape": list(info.a_turn.shape),
        "rank": info.rank,
        "nullity": info.nullity,
        "condition_number": info.condition_number,
        "effective_rank_99": info.effective_rank_99,
    }
