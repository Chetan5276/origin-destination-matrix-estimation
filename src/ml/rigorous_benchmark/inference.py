"""Inference API: estimate_od / estimate_od_batch with frozen scalers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src import NUM_ZONES
from src.data.od_pairs import unflatten_od_vector
from src.ml.rigorous_benchmark.data import BenchmarkData, transform_x
from src.ml.rigorous_benchmark.models.base import BenchmarkModel
from src.ml.rigorous_benchmark.operator import OperatorInfo

logger = logging.getLogger(__name__)


def estimate_od(
    model: BenchmarkModel,
    x_turn: np.ndarray,
    data: BenchmarkData,
    operator: OperatorInfo,
) -> np.ndarray:
    """
    Estimate a single OD matrix from turning counts.

    Parameters
    ----------
    x_turn : array shape (n_turns,) or (1, n_turns)

    Returns
    -------
    od : (24, 24) float32
    """
    x = np.asarray(x_turn, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    y = model.predict(x, data, operator)
    return unflatten_od_vector(y[0]).astype(np.float32)


def estimate_od_batch(
    model: BenchmarkModel,
    x_batch: np.ndarray,
    data: BenchmarkData,
    operator: OperatorInfo,
) -> np.ndarray:
    """Batch OD estimation → (N, 24, 24)."""
    x = np.asarray(x_batch, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    y = model.predict(x, data, operator)
    return y.reshape(-1, NUM_ZONES, NUM_ZONES).astype(np.float32)


def survey_scale_diagnostics(
    data: BenchmarkData,
) -> dict[str, float]:
    """Compare survey X/Y moments to train (no refit)."""
    x_s = data.survey_x_raw.ravel()
    y_s = data.survey_y_raw.ravel()
    x_tr = data.x_train_raw
    y_tr = data.y_train_raw

    def _moments(a, b_mean, b_std):
        return {
            "mean": float(np.mean(a)),
            "std": float(np.std(a)),
            "mean_shift_vs_train": float(np.mean(a) - b_mean),
            "std_ratio_vs_train": float(np.std(a) / max(b_std, 1e-6)),
        }

    # Approximate 1D Wasserstein between survey and a train subsample (pooled cells)
    rng = np.random.default_rng(0)
    n = min(5000, x_tr.size)
    xt = rng.choice(x_tr.ravel(), size=n, replace=False)
    xs = np.resize(x_s, n) if x_s.size < n else rng.choice(x_s, size=n, replace=False)
    xt_s = np.sort(xt)
    xs_s = np.sort(xs.astype(float))
    w1_x = float(np.mean(np.abs(xt_s - xs_s)))

    n_y = min(5000, y_tr.size)
    yt = rng.choice(y_tr.ravel(), size=n_y, replace=False)
    ys = np.resize(y_s, n_y) if y_s.size < n_y else rng.choice(y_s, size=n_y, replace=False)
    w1_y = float(np.mean(np.abs(np.sort(yt) - np.sort(ys.astype(float)))))

    return {
        "x": _moments(x_s, float(x_tr.mean()), float(x_tr.std())),
        "y": _moments(y_s, float(y_tr.mean()), float(y_tr.std())),
        "wasserstein1_x_approx": w1_x,
        "wasserstein1_y_approx": w1_y,
        "note": "Scalers never refit on survey; diagnostics only",
    }


def run_survey_inference(
    models: dict[str, BenchmarkModel],
    data: BenchmarkData,
    operator: OperatorInfo,
    out_dir: Path,
) -> dict[str, Any]:
    """Evaluate all frozen models on the held-out survey pair."""
    import json

    from src.ml.rigorous_benchmark.metrics_suite import compute_all_metrics
    from src.ml.rigorous_benchmark.operator import pinv_predict

    out_dir.mkdir(parents=True, exist_ok=True)
    diag = survey_scale_diagnostics(data)
    (out_dir / "scale_diagnostics.json").write_text(json.dumps(diag, indent=2, default=str))

    x = data.survey_x_raw.reshape(1, -1)
    y_true = data.survey_y_raw.reshape(1, -1)
    y_pinv = pinv_predict(operator.a_pinv, x)
    results = {}
    for name, model in models.items():
        pred = model.predict(x, data, operator)
        metrics = compute_all_metrics(
            y_true, pred, x, operator.a_turn, y_pinv=y_pinv
        )
        od = unflatten_od_vector(pred[0])
        np.save(out_dir / f"{name}_od.npy", od)
        results[name] = {
            "metrics": {k: v for k, v in metrics.items() if isinstance(v, (int, float))},
            "constraint_strategy": (
                model.fit_result.constraint_strategy if model.fit_result else "unknown"
            ),
        }
        logger.info("Survey %s MAE=%.4g forward_rmse=%.4g", name, metrics["mae"], metrics["forward_rmse"])

    (out_dir / "survey_results.json").write_text(json.dumps(results, indent=2, default=str))
    return results


def load_scalers(path: Path) -> dict:
    return joblib.load(path)
