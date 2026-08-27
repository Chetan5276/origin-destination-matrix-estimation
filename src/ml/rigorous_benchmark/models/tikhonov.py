"""Tikhonov regularization: solve (AᵀA + λI) y = Aᵀx via cho_solve / lstsq."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.linalg import cho_factor, cho_solve, lstsq

from src.ml.rigorous_benchmark.data import BenchmarkData
from src.ml.rigorous_benchmark.hpo import run_study, save_best_params, val_composite_from_preds
from src.ml.rigorous_benchmark.models.base import BenchmarkModel, FitResult
from src.ml.rigorous_benchmark.operator import OperatorInfo


def tikhonov_solve(a: np.ndarray, x: np.ndarray, lam: float) -> np.ndarray:
    """
    Batched Tikhonov: y = argmin ||Ay - x||^2 + λ||y||^2
    Equivalent to (AᵀA + λI) y = Aᵀx.
    """
    a64 = np.asarray(a, dtype=np.float64)
    x64 = np.asarray(x, dtype=np.float64)
    n_od = a64.shape[1]
    ata = a64.T @ a64
    ata.flat[:: n_od + 1] += float(lam)
    rhs = x64 @ a64  # (N, n_od) = X @ A
    try:
        c, lower = cho_factor(ata, lower=True, check_finite=False)
        y = cho_solve((c, lower), rhs.T, check_finite=False).T
    except np.linalg.LinAlgError:
        # Fall back to least squares on augmented system per batch via ATA lstsq
        y, _, _, _ = lstsq(ata, rhs.T, lapack_driver="gelsy")
        y = np.asarray(y.T)
    return y.astype(np.float32)


class TikhonovModel(BenchmarkModel):
    name = "tikhonov"

    def __init__(self, config):
        super().__init__(config)
        self.lam: float = 1.0

    def hyperopt(self, data: BenchmarkData, operator: OperatorInfo) -> dict[str, Any]:
        a = operator.a_turn
        x_va = data.x_val_raw
        y_va = data.y_val_raw
        alpha, beta, gamma = self.config.alpha_fwd, self.config.beta_prod, self.config.gamma_attr
        y_pinv = (data.x_val_raw @ operator.a_pinv.T).astype(np.float32)

        def objective(trial):
            lam = trial.suggest_float("lambda", 1e-6, 1e6, log=True)
            pred = tikhonov_solve(a, x_va, lam)
            return val_composite_from_preds(
                y_va, pred, x_va, a, alpha=alpha, beta=beta, gamma=gamma, y_pinv=y_pinv
            )

        study = run_study(self.name, objective, self.config.n_trials, seed=self.config.seed)
        params = dict(study.best_params)
        save_best_params(self.model_dir() / "best_params.json", params, {"best_value": study.best_value})
        return params

    def fit(
        self,
        data: BenchmarkData,
        operator: OperatorInfo,
        params: dict[str, Any],
        *,
        use_train_val: bool = True,
    ) -> FitResult:
        self.lam = float(params.get("lambda", 1.0))
        result = FitResult(
            model_name=self.name,
            best_params={"lambda": self.lam},
            constraint_strategy="none",
            notes="Tikhonov closed form via cho_solve; identity L; no silent clip",
        )
        self.fit_result = result
        self.save_metadata(result)
        return result

    def predict(
        self,
        x_raw: np.ndarray,
        data: BenchmarkData,
        operator: OperatorInfo,
    ) -> np.ndarray:
        return tikhonov_solve(operator.a_turn, x_raw, self.lam)
