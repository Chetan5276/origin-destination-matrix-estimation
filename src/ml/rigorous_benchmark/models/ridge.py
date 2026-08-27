"""Sklearn multi-output Ridge with Optuna HPO."""

from __future__ import annotations

from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import Ridge

from src.ml.rigorous_benchmark.constraints import apply_constraint_strategy
from src.ml.rigorous_benchmark.data import BenchmarkData, transform_x
from src.ml.rigorous_benchmark.hpo import run_study, save_best_params, val_composite_from_preds
from src.ml.rigorous_benchmark.models.base import BenchmarkModel, FitResult
from src.ml.rigorous_benchmark.operator import OperatorInfo


class RidgeModel(BenchmarkModel):
    name = "ridge"

    def __init__(self, config):
        super().__init__(config)
        self.model: Ridge | None = None

    def hyperopt(self, data: BenchmarkData, operator: OperatorInfo) -> dict[str, Any]:
        x_tr, _, x_tr_raw, y_tr_raw = data.hpo_train_subset(
            self.config.hpo_train_subsample, seed=self.config.seed
        )
        # Fit on raw Y for interpretability; X scaled
        alpha_c, beta, gamma = self.config.alpha_fwd, self.config.beta_prod, self.config.gamma_attr
        a = operator.a_turn
        y_pinv = (data.x_val_raw @ operator.a_pinv.T).astype(np.float32)

        def objective(trial):
            alpha = trial.suggest_float("alpha", 1e-4, 1e5, log=True)
            est = Ridge(alpha=alpha, random_state=self.config.seed)
            est.fit(x_tr, y_tr_raw)
            pred = est.predict(data.x_val).astype(np.float32)
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
        alpha = float(params.get("alpha", 1.0))
        if use_train_val:
            x, y = data.train_val_scaled()
            # need raw Y
            y = np.concatenate([data.y_train_raw, data.y_val_raw], axis=0)
        else:
            x, y = data.x_train, data.y_train_raw
        self.model = Ridge(alpha=alpha, random_state=self.config.seed)
        self.model.fit(x, y)
        path = self.model_dir() / "model.joblib"
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        result = FitResult(
            model_name=self.name,
            best_params={"alpha": alpha},
            artifact_paths={"model": str(path)},
            constraint_strategy="none",
            notes="sklearn Ridge multi-output on scaled X → raw Y; no silent clip",
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
        return np.asarray(self.model.predict(xs), dtype=np.float32)
