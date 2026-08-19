"""Evaluation metrics for OD estimation."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src import NUM_ZONES


def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    mask = np.abs(y_true) > eps
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def production_vector(y_flat: np.ndarray) -> np.ndarray:
    """Row sums (productions) from flattened OD."""
    n = y_flat.shape[0]
    mat = y_flat.reshape(n, NUM_ZONES, NUM_ZONES)
    return mat.sum(axis=2)


def attraction_vector(y_flat: np.ndarray) -> np.ndarray:
    """Column sums (attractions) from flattened OD."""
    n = y_flat.shape[0]
    mat = y_flat.reshape(n, NUM_ZONES, NUM_ZONES)
    return mat.sum(axis=1)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute cell-level, matrix-level, and marginal metrics."""
    y_true = np.clip(np.asarray(y_true, dtype=float), 0, None)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred, multioutput="uniform_average"))
    mape = _safe_mape(y_true, y_pred)

    frob = float(np.linalg.norm(y_true - y_pred, ord="fro") / np.sqrt(y_true.size))

    prod_true = production_vector(y_true)
    prod_pred = production_vector(y_pred)
    attr_true = attraction_vector(y_true)
    attr_pred = attraction_vector(y_pred)

    prod_err = float(mean_absolute_error(prod_true, prod_pred))
    attr_err = float(mean_absolute_error(attr_true, attr_pred))

    corrs = [
        float(np.corrcoef(t, p)[0, 1])
        for t, p in zip(y_true, y_pred)
        if t.std() > 1e-12 and p.std() > 1e-12
    ]
    corr = float(np.mean(corrs)) if corrs else float("nan")

    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "r2": r2,
        "frobenius": frob,
        "production_mae": prod_err,
        "attraction_mae": attr_err,
        "correlation": corr,
    }


def forward_consistency_error(
    y_pred: np.ndarray,
    x_turn: np.ndarray,
    a_turn: np.ndarray,
) -> float:
    """RMSE between observed turning counts and A_turn @ predicted OD."""
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    x_turn = np.asarray(x_turn, dtype=float)
    a_turn = np.asarray(a_turn, dtype=float)
    x_hat = y_pred @ a_turn.T
    return float(np.sqrt(np.mean((x_turn - x_hat) ** 2)))


def evaluate_predictions_with_forward(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    x_turn: np.ndarray | None = None,
    a_turn: np.ndarray | None = None,
) -> dict[str, float]:
    """OD metrics plus optional forward-consistency RMSE."""
    metrics = evaluate_predictions(y_true, y_pred)
    if x_turn is not None and a_turn is not None:
        metrics["forward_rmse"] = forward_consistency_error(y_pred, x_turn, a_turn)
    return metrics
