"""Tests for Phase 3 ML pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.config import NeuralTrainConfig
from src.ml.dataset import split_dataset
from src.ml.metrics import evaluate_predictions_with_forward
from src.ml.models_classical import classical_models, fit_model, predict_model
from src.ml.neural_trainer import predict_direct_model, train_direct_model
from src.ml.od_constraints import apply_od_constraints_numpy, base_support_mask


def test_evaluate_predictions_with_forward():
    rng = np.random.default_rng(0)
    y = rng.random((10, 576)) * 100
    y_hat = y + rng.normal(0, 5, y.shape)
    x_turn = rng.random((10, 178)) * 50
    a_turn = rng.random((178, 576))
    m = evaluate_predictions_with_forward(y, y_hat, x_turn, a_turn)
    assert m["mae"] > 0
    assert "forward_rmse" in m


def test_ridge_fits_small_data():
    rng = np.random.default_rng(1)
    x = rng.random((200, 178)).astype(np.float32)
    y = rng.random((200, 576)).astype(np.float32)
    spec = next(s for s in classical_models(n_jobs=1) if s.name == "ridge")
    model = fit_model(spec, x, y, cv=2, n_jobs=1)
    pred = predict_model(model, x[:5])
    assert pred.shape == (5, 576)


def test_od_constraints_and_mlp_smoke():
    rng = np.random.default_rng(2)
    base = np.zeros((24, 24), dtype=np.float32)
    base[0, 1] = 10.0
    mask = base_support_mask(base)
    assert mask.sum() == 24 * 23
    x = rng.random((64, 178)).astype(np.float32)
    y = rng.random((64, 576)).astype(np.float32) * 100
    ds = split_dataset(x, y, base_od=base, seed=0, standardize_x=True, standardize_y=True)
    a_turn = rng.random((178, 576)).astype(np.float32)
    cfg = NeuralTrainConfig(epochs=1, batch_size=16)
    model = train_direct_model(ds, a_turn, cfg, residual_blocks=False, residual_learning=False)
    pred = predict_direct_model(model, ds, ds.x_test[:4], cfg)
    assert pred.shape == (4, 576)
    assert pred.min() >= 0
    constrained = apply_od_constraints_numpy(pred, support_mask=mask)
    assert constrained[:, mask].min() >= 0
    # Diagonal must be zero under full off-diagonal support
    mat = constrained.reshape(4, 24, 24)
    assert np.allclose(np.diagonal(mat, axis1=1, axis2=2), 0.0)
