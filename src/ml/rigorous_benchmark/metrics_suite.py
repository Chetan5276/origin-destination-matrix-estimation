"""Extended metrics for OD, forward, marginals, and inverse-problem distances."""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src import NUM_ZONES
from src.ml.metrics import (
    attraction_vector,
    evaluate_predictions,
    production_vector,
)

# Forward RMSE is in turning space: ||X - Y_hat @ A_turn.T||


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size < 3 or a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    rho, _ = spearmanr(a, b)
    return float(rho)


def forward_predictions(y_pred: np.ndarray, a_turn: np.ndarray) -> np.ndarray:
    """X_hat = Y_flat @ A_turn.T  (turning space)."""
    return np.asarray(y_pred, dtype=np.float64) @ np.asarray(a_turn, dtype=np.float64).T


def forward_metrics(
    y_pred: np.ndarray,
    x_turn: np.ndarray,
    a_turn: np.ndarray,
) -> dict[str, float]:
    """Forward-consistency metrics in turning-count space."""
    x_hat = forward_predictions(y_pred, a_turn)
    x = np.asarray(x_turn, dtype=np.float64)
    resid = x - x_hat
    mae = float(np.mean(np.abs(resid)))
    rmse = float(np.sqrt(np.mean(resid**2)))
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((x - x.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")
    corr = _safe_corr(x, x_hat)
    return {
        "forward_mae": mae,
        "forward_rmse": rmse,
        "forward_r2": r2,
        "forward_corr": corr,
        "forward_residual_l2_mean": float(np.mean(np.linalg.norm(resid, axis=1))),
    }


def structural_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Sparsity / support agreement and diagonal leakage."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    eps = 1e-6
    true_pos = yt > eps
    pred_pos = yp > eps
    inter = np.logical_and(true_pos, pred_pos).sum()
    union = np.logical_or(true_pos, pred_pos).sum()
    jaccard = float(inter / union) if union > 0 else float("nan")
    n = yt.shape[0]
    mats = yp.reshape(n, NUM_ZONES, NUM_ZONES)
    diag_abs = float(np.mean(np.abs(np.diagonal(mats, axis1=1, axis2=2))))
    return {
        "support_jaccard": jaccard,
        "pred_sparsity": float(1.0 - pred_pos.mean()),
        "true_sparsity": float(1.0 - true_pos.mean()),
        "mean_abs_diagonal": diag_abs,
    }


def inverse_problem_metrics(
    y_pred: np.ndarray,
    y_pinv: np.ndarray,
    a_turn: np.ndarray,
    x_turn: np.ndarray,
) -> dict[str, float]:
    """Distance to Moore–Penrose solution and residual on the range of A."""
    yp = np.asarray(y_pred, dtype=np.float64)
    ypin = np.asarray(y_pinv, dtype=np.float64)
    delta = yp - ypin
    dist = float(np.mean(np.linalg.norm(delta, axis=1)))
    # Component in null space of A: A @ (y - y_pinv) should be ~0 for particular solution
    # relative null deviation: ||A (y_pred - y_pinv)|| / ||x||
    a = np.asarray(a_turn, dtype=np.float64)
    ax = (yp - ypin) @ a.T
    x = np.asarray(x_turn, dtype=np.float64)
    null_leak = float(np.mean(np.linalg.norm(ax, axis=1) / np.maximum(np.linalg.norm(x, axis=1), 1e-6)))
    return {
        "mean_l2_to_pinv": dist,
        "mean_rel_nullspace_leak_via_A": null_leak,
    }


def spearman_od(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    max_samples: int | None = 500,
    seed: int = 0,
) -> float:
    """Mean per-sample Spearman ρ over OD cells (optionally subsampled)."""
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    n = yt.shape[0]
    if max_samples is not None and n > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_samples, replace=False)
        yt = yt[idx]
        yp = yp[idx]
    rhos = [_safe_spearman(t, p) for t, p in zip(yt, yp)]
    arr = np.asarray(rhos, dtype=float)
    return float(np.nanmean(arr))


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    x_turn: np.ndarray,
    a_turn: np.ndarray,
    *,
    y_pinv: np.ndarray | None = None,
    clip_for_legacy: bool = False,
    spearman_max_samples: int | None = 500,
) -> dict[str, float]:
    """
    Full §16-style metric dict.

    Does **not** silently clip predictions for primary metrics unless
    ``clip_for_legacy=True`` (legacy ``evaluate_predictions`` clips).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if clip_for_legacy:
        base = evaluate_predictions(y_true, y_pred)
    else:
        # Unclipped cell-level metrics
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        try:
            if y_true.shape[0] < 2:
                r2 = float("nan")
            else:
                r2 = float(r2_score(y_true, y_pred, multioutput="uniform_average"))
        except ValueError:
            r2 = float("nan")
        frob = float(np.linalg.norm(y_true - y_pred) / np.sqrt(y_true.size))
        prod_err = float(
            mean_absolute_error(production_vector(y_true), production_vector(y_pred))
        )
        attr_err = float(
            mean_absolute_error(attraction_vector(y_true), attraction_vector(y_pred))
        )
        # Subsample correlations for speed/memory on large test sets
        n = y_true.shape[0]
        if spearman_max_samples is not None and n > spearman_max_samples:
            rng = np.random.default_rng(0)
            idx = rng.choice(n, size=spearman_max_samples, replace=False)
            yt_c, yp_c = y_true[idx], y_pred[idx]
        else:
            yt_c, yp_c = y_true, y_pred
        corrs = [
            _safe_corr(t, p)
            for t, p in zip(yt_c, yp_c)
            if np.asarray(t).std() > 1e-12 and np.asarray(p).std() > 1e-12
        ]
        corr = float(np.nanmean(corrs)) if corrs else float("nan")
        # Relative OD error
        denom = np.maximum(np.abs(y_true), 1e-6)
        rel_err = float(np.mean(np.abs(y_true - y_pred) / denom))
        # Total demand / negatives / diagonal
        tot_true = y_true.sum(axis=1)
        tot_pred = y_pred.sum(axis=1)
        total_demand_error = float(np.mean(np.abs(tot_true - tot_pred)))
        negative_cells = float(np.mean(np.sum(y_pred < -1e-8, axis=1)))
        mats = y_pred.reshape(y_pred.shape[0], NUM_ZONES, NUM_ZONES)
        diagonal_violation = float(np.mean(np.abs(np.diagonal(mats, axis1=1, axis2=2))))
        base = {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "frobenius": frob,
            "production_mae": prod_err,
            "attraction_mae": attr_err,
            "correlation": corr,
            "pearson": corr,
            "relative_error": rel_err,
            "total_demand_error": total_demand_error,
            "negative_cells": negative_cells,
            "diagonal_violation": diagonal_violation,
        }

    out = dict(base)
    out["spearman"] = spearman_od(y_true, y_pred, max_samples=spearman_max_samples)
    # production/attraction RMSE
    pt, pp = production_vector(y_true), production_vector(y_pred)
    at, ap = attraction_vector(y_true), attraction_vector(y_pred)
    out["production_rmse"] = float(np.sqrt(mean_squared_error(pt, pp)))
    out["attraction_rmse"] = float(np.sqrt(mean_squared_error(at, ap)))
    out["production_rel_error"] = float(
        np.mean(np.abs(pt - pp) / np.maximum(np.abs(pt), 1e-6))
    )
    out["attraction_rel_error"] = float(
        np.mean(np.abs(at - ap) / np.maximum(np.abs(at), 1e-6))
    )
    out["sparsity_error"] = float(
        abs((y_pred <= 1e-6).mean() - (y_true <= 1e-6).mean())
    )
    out.update(forward_metrics(y_pred, x_turn, a_turn))
    out.update(structural_metrics(y_true, y_pred))
    if y_pinv is not None:
        out.update(inverse_problem_metrics(y_pred, y_pinv, a_turn, x_turn))
    out["forward_space_definition"] = 0.0  # placeholder; documented in report
    # Use a sentinel note via separate key for JSON consumers that want strings
    return out


def normalize_metric(value: float, ref: float, *, higher_is_better: bool = False) -> float:
    """Scale metric relative to a reference (typically mean or max on val)."""
    if ref is None or not np.isfinite(ref) or abs(ref) < 1e-12:
        return float("nan")
    if not np.isfinite(value):
        return float("nan")
    if higher_is_better:
        return float(ref / value) if abs(value) > 1e-12 else float("nan")
    return float(value / ref)


def composite_score(
    metrics: dict[str, float],
    *,
    alpha: float = 1.0,
    beta: float = 0.5,
    gamma: float = 0.5,
    refs: dict[str, float] | None = None,
) -> float:
    """
    score = nMAE_OD + α nFwd + β nProd + γ nAttr

    If ``refs`` provided, normalize each term by the corresponding reference
    (validation-set normalization for selection). Otherwise use raw values.
    """
    mae = metrics["mae"]
    fwd = metrics.get("forward_mae", metrics.get("forward_rmse", float("nan")))
    prod = metrics["production_mae"]
    attr = metrics["attraction_mae"]
    if refs:
        mae = normalize_metric(mae, refs.get("mae", mae))
        fwd = normalize_metric(fwd, refs.get("forward_mae", refs.get("forward_rmse", fwd)))
        prod = normalize_metric(prod, refs.get("production_mae", prod))
        attr = normalize_metric(attr, refs.get("attraction_mae", attr))
    return float(mae + alpha * fwd + beta * prod + gamma * attr)
