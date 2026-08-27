"""Validation-only model selection and ranking tables."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.ml.rigorous_benchmark.metrics_suite import composite_score


FAMILY_MAP: dict[str, str] = {
    "moore_penrose": "operator",
    "tikhonov": "operator",
    "ridge": "classical",
    "physics_ridge": "classical",
    "pls": "classical",
    "mlp": "neural",
    "residual_mlp": "neural",
    "physics_residual_mlp": "neural",
    "ae_32": "autoencoder",
    "ae_64": "autoencoder",
    "ae_128": "autoencoder",
    "ae_64_finetune": "autoencoder",
    "nullspace_mlp": "neural",
}


def val_refs_from_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Mean of each metric across models on validation (for normalization)."""
    keys = ["mae", "forward_mae", "production_mae", "attraction_mae"]
    refs: dict[str, float] = {}
    for k in keys:
        vals = [r["metrics"].get(k, np.nan) for r in rows if "metrics" in r]
        if not vals:
            vals = [r.get(k, np.nan) for r in rows]
        arr = np.asarray(vals, dtype=float)
        refs[k] = float(np.nanmean(arr)) if np.isfinite(arr).any() else 1.0
    return refs


def attach_composite(
    rows: list[dict[str, Any]],
    *,
    alpha: float,
    beta: float,
    gamma: float,
    refs: dict[str, float] | None = None,
    split_key: str = "val",
) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        m = r.get("metrics") or r
        score = composite_score(m, alpha=alpha, beta=beta, gamma=gamma, refs=refs)
        nr = dict(r)
        nr["composite_score"] = score
        nr["family"] = FAMILY_MAP.get(r.get("model", ""), "other")
        nr["split"] = r.get("split", split_key)
        out.append(nr)
    return out


def ranking_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Four ranking tables: OD / forward / marginals / overall."""
    if df.empty:
        empty = pd.DataFrame()
        return {"od": empty, "forward": empty, "marginals": empty, "overall": empty}

    od = df.sort_values("mae").reset_index(drop=True)
    fwd_col = "forward_mae" if "forward_mae" in df.columns else "forward_rmse"
    forward = df.sort_values(fwd_col).reset_index(drop=True)
    df = df.copy()
    df["marginal_mae"] = df["production_mae"] + df["attraction_mae"]
    marginals = df.sort_values("marginal_mae").reset_index(drop=True)
    overall = df.sort_values("composite_score").reset_index(drop=True)
    return {"od": od, "forward": forward, "marginals": marginals, "overall": overall}


def results_to_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    flat = []
    for r in rows:
        row = {
            "model": r.get("model"),
            "family": r.get("family", FAMILY_MAP.get(r.get("model", ""), "other")),
            "split": r.get("split"),
            "composite_score": r.get("composite_score"),
            "constraint_strategy": r.get("constraint_strategy"),
        }
        metrics = r.get("metrics") or {}
        for k, v in metrics.items():
            if isinstance(v, (int, float, np.floating, np.integer)) or v is None:
                row[k] = float(v) if v is not None and np.isfinite(v) else v
        # also allow already-flat rows
        for k, v in r.items():
            if k in ("model", "family", "split", "composite_score", "metrics", "constraint_strategy"):
                continue
            if isinstance(v, (int, float, np.floating, np.integer)):
                row.setdefault(k, float(v))
        flat.append(row)
    return pd.DataFrame(flat)
