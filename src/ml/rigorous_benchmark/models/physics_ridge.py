"""Physics-informed Ridge: sklearn Ridge + separate forward-augmented closed form.

Distinction from Tikhonov
------------------------
- Tikhonov solves (AᵀA + λI)y = Aᵀx using the known operator only (no OD labels).
- Physics Ridge here fits a supervised multi-output Ridge on (X, Y), then reports
  an optional closed-form that adds a forward penalty against A when refining:
  y ← argmin ||y - y_ridge||^2 + μ ||A y - x||^2
  which is (I + μ AᵀA) y = y_ridge + μ Aᵀx  (cho_solve).
"""

from __future__ import annotations

from typing import Any

import joblib
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from sklearn.linear_model import Ridge

from src.ml.rigorous_benchmark.data import BenchmarkData, transform_x
from src.ml.rigorous_benchmark.hpo import run_study, save_best_params, val_composite_from_preds
from src.ml.rigorous_benchmark.models.base import BenchmarkModel, FitResult
from src.ml.rigorous_benchmark.operator import OperatorInfo


def physics_refine(y_ridge: np.ndarray, x: np.ndarray, a: np.ndarray, mu: float) -> np.ndarray:
    """One-shot closed-form forward refinement of a supervised prediction."""
    if mu <= 0:
        return y_ridge.astype(np.float32)
    a64 = np.asarray(a, dtype=np.float64)
    y0 = np.asarray(y_ridge, dtype=np.float64)
    x64 = np.asarray(x, dtype=np.float64)
    n_od = a64.shape[1]
    ata = a64.T @ a64
    m = np.eye(n_od) + mu * ata
    rhs = y0 + mu * (x64 @ a64)
    c, lower = cho_factor(m, lower=True, check_finite=False)
    y = cho_solve((c, lower), rhs.T, check_finite=False).T
    return y.astype(np.float32)


class PhysicsRidgeModel(BenchmarkModel):
    name = "physics_ridge"

    def __init__(self, config):
        super().__init__(config)
        self.model: Ridge | None = None
        self.mu: float = 0.0
        self.alpha: float = 1.0

    def hyperopt(self, data: BenchmarkData, operator: OperatorInfo) -> dict[str, Any]:
        x_tr, _, _, y_tr_raw = data.hpo_train_subset(
            self.config.hpo_train_subsample, seed=self.config.seed
        )
        a = operator.a_turn
        alpha_c, beta, gamma = self.config.alpha_fwd, self.config.beta_prod, self.config.gamma_attr
        y_pinv = (data.x_val_raw @ operator.a_pinv.T).astype(np.float32)

        def objective(trial):
            alpha = trial.suggest_float("alpha", 1e-4, 1e5, log=True)
            mu = trial.suggest_float("mu", 1e-6, 1e2, log=True)
            est = Ridge(alpha=alpha, random_state=self.config.seed)
            est.fit(x_tr, y_tr_raw)
            pred0 = est.predict(data.x_val).astype(np.float32)
            pred = physics_refine(pred0, data.x_val_raw, a, mu)
            return val_composite_from_preds(
                data.y_val_raw,
                pred,
                data.x_val_raw,
                a,
                alpha=alpha_c,
                beta=beta,
                gamma=gamma,
                y_pinv=y_pinv,
            )

        study = run_study(self.name, objective, self.config.n_trials, seed=self.config.seed)
        params = dict(study.best_params)
        save_best_params(self.model_dir() / "hpo_best.json", params, {"best_value": study.best_value})
        return params

    def fit(
        self,
        data: BenchmarkData,
        operator: OperatorInfo,
        params: dict[str, Any],
        *,
        use_train_val: bool = True,
    ) -> FitResult:
        self.alpha = float(params.get("alpha", 1.0))
        self.mu = float(params.get("mu", 0.1))
        if use_train_val:
            x = np.concatenate([data.x_train, data.x_val], axis=0)
            y = np.concatenate([data.y_train_raw, data.y_val_raw], axis=0)
        else:
            x, y = data.x_train, data.y_train_raw
        self.model = Ridge(alpha=self.alpha, random_state=self.config.seed)
        self.model.fit(x, y)
        path = self.model_dir() / "model.joblib"
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "mu": self.mu, "alpha": self.alpha}, path)
        result = FitResult(
            model_name=self.name,
            best_params={"alpha": self.alpha, "mu": self.mu},
            artifact_paths={"model": str(path)},
            constraint_strategy="none",
            notes=(
                "Supervised Ridge + forward closed-form refine (I+μAᵀA)y = y_ridge+μAᵀx; "
                "distinct from unsupervised Tikhonov"
            ),
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
        assert self.model is not None
        xs = transform_x(data, x_raw)
        pred0 = np.asarray(self.model.predict(xs), dtype=np.float32)
        return physics_refine(pred0, x_raw, operator.a_turn, self.mu)
