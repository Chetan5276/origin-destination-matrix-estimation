"""Tests for the rigorous OD estimation benchmark."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import NUM_ZONES
from src.data.statistics import flatten_od_batch
from src.ml.rigorous_benchmark.config import smoke_config
from src.ml.rigorous_benchmark.constraints import apply_constraint_strategy
from src.ml.rigorous_benchmark.data import (
    assert_disjoint,
    make_split_indices,
    verify_forward_map,
    write_leakage_report,
)
from src.ml.rigorous_benchmark.inference import estimate_od
from src.ml.rigorous_benchmark.metrics_suite import compute_all_metrics, forward_metrics
from src.ml.rigorous_benchmark.models.pinv import MoorePenroseModel
from src.ml.rigorous_benchmark.models.tikhonov import tikhonov_solve
from src.ml.rigorous_benchmark.operator import compute_operator, pinv_predict


def test_forward_orientation():
    rng = np.random.default_rng(0)
    a = rng.random((20, 30))
    y = rng.random((5, 30))
    x = y @ a.T
    check = verify_forward_map(y, x, a, atol=1e-8, rtol=1e-8)
    assert check["allclose"]
    assert check["forward_formula"] == "X = Y_flat @ A_turn.T"


def test_split_disjointness():
    tr, va, te = make_split_indices(1000, train_frac=0.7, val_frac=0.15, test_frac=0.15, seed=42)
    assert_disjoint(tr, va, te)
    assert len(tr) + len(va) + len(te) == 1000


def test_survey_index_exclusion(tmp_path):
    tr, va, te = make_split_indices(1000, train_frac=0.7, val_frac=0.15, test_frac=0.15, seed=42)
    report = write_leakage_report(
        tmp_path / "leakage_report.json",
        train_idx=tr,
        val_idx=va,
        test_idx=te,
        survey_index=100000,
        n_synthetics=100000,
        a_turn_sha256="abc",
        forward_check={"allclose": True},
        max_samples=1000,
    )
    assert report["survey_excluded_from_development"] is True
    assert report["survey_in_train_val_test"] is False


def test_pinv_residual_noise_free():
    rng = np.random.default_rng(1)
    # Fat underdetermined-ish: m < n
    m, n = 40, 60
    a = rng.standard_normal((m, n))
    y_true = rng.standard_normal(n)
    x = a @ y_true
    op = compute_operator(a, rtol=1e-10)
    y_hat = pinv_predict(op.a_pinv, x.reshape(1, -1))[0]
    # Minimum-norm solution: A y_hat ≈ x
    resid = np.linalg.norm(a @ y_hat - x)
    assert resid < 1e-5
    # Among solutions, pinv is min-norm
    assert np.linalg.norm(y_hat) <= np.linalg.norm(y_true) + 1e-5


def test_tikhonov_cho_solve_stable():
    rng = np.random.default_rng(2)
    a = rng.standard_normal((30, 50))
    y = rng.standard_normal((4, 50))
    x = y @ a.T
    y_hat = tikhonov_solve(a, x, lam=1e-2)
    assert y_hat.shape == (4, 50)
    assert np.isfinite(y_hat).all()


def test_metrics_forward_space_definition():
    rng = np.random.default_rng(3)
    a = rng.random((10, 24 * 24))
    y_true = rng.random((8, 24 * 24)) * 10
    y_pred = y_true + rng.normal(0, 0.5, y_true.shape)
    x = y_true @ a.T
    fwd = forward_metrics(y_pred, x, a)
    x_hat = y_pred @ a.T
    rmse_manual = float(np.sqrt(np.mean((x - x_hat) ** 2)))
    assert abs(fwd["forward_rmse"] - rmse_manual) < 1e-9
    m = compute_all_metrics(y_true, y_pred, x, a)
    assert "forward_rmse" in m and "spearman" in m


def test_inference_shape_24x24():
    rng = np.random.default_rng(4)
    a = rng.random((20, 576)).astype(np.float32)
    op = compute_operator(a)
    cfg = smoke_config(models=("moore_penrose",))
    model = MoorePenroseModel(cfg)
    # Minimal fake BenchmarkData-like object via simple namespace
    from types import SimpleNamespace

    data = SimpleNamespace()
    x = rng.random(20).astype(np.float32)
    model.fit(data, op, {})
    od = estimate_od(model, x, data, op)
    assert od.shape == (NUM_ZONES, NUM_ZONES)


def test_constraint_strategy_reported():
    rng = np.random.default_rng(5)
    y = rng.normal(0, 1, (3, 576))
    y_pos, meta = apply_constraint_strategy(y, strategy="relu")
    assert meta["constraint_strategy"] == "relu"
    assert (y_pos >= 0).all()
    y_none, meta2 = apply_constraint_strategy(y, strategy="none")
    assert meta2["constraint_strategy"] == "none"
    assert y_none.min() < 0  # likely negatives preserved


@pytest.mark.slow
def test_smoke_benchmark_subset(tmp_path):
    """Optional end-to-end smoke (requires FP data on disk)."""
    od = PROJECT_ROOT / "outputs/od_generator_fp/synthetic_od_fp_synthetics_only.npy"
    if not od.exists():
        pytest.skip("FP data not present")
    from src.ml.rigorous_benchmark.run import run_benchmark

    cfg = smoke_config(
        models=("moore_penrose", "tikhonov", "ridge"),
        output_dir=tmp_path / "bench",
        n_trials=1,
        max_samples=256,
        hpo_train_subsample=128,
        run_ablations=False,
    )
    summary = run_benchmark(cfg, skip_hpo=False, skip_survey=False)
    assert (tmp_path / "bench" / "final_report.md").exists()
    assert summary["n_models"] == 3
