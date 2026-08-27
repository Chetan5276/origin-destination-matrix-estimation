"""PLS regression with Optuna HPO on n_components."""

from __future__ import annotations

from typing import Any

import joblib
import numpy as np
from sklearn.cross_decomposition import PLSRegression

from src.ml.rigorous_benchmark.data import BenchmarkData, transform_x
from src.ml.rigorous_benchmark.hpo import run_study, save_best_params, val_composite_from_preds
from src.ml.rigorous_benchmark.models.base import BenchmarkModel, FitResult
from src.ml.rigorous_benchmark.operator import OperatorInfo


class PLSModel(BenchmarkModel):
    name = "pls"

    def __init__(self, config):
        super().__init__(config)
        self.model: PLSRegression | None = None

    def hyperopt(self, data: BenchmarkData, operator: OperatorInfo) -> dict[str, Any]:
        x_tr, _, _, y_tr_raw = data.hpo_train_subset(
            self.config.hpo_train_subsample, seed=self.config.seed
        )
        max_c = min(20, x_tr.shape[0] - 1, x_tr.shape[1], y_tr_raw.shape[1])
        a = operator.a_turn
        alpha_c, beta, gamma = self.config.alpha_fwd, self.config.beta_prod, self.config.gamma_attr
        y_pinv = (data.x_val_raw @ operator.a_pinv.T).astype(np.float32)

        def objective(trial):
            n_comp = trial.suggest_int("n_components", 2, max(2, max_c))
            est = PLSRegression(n_components=n_comp, scale=False, max_iter=200)
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
        n_comp = int(params.get("n_components", 25))
        if use_train_val:
            x = np.concatenate([data.x_train, data.x_val], axis=0)
            y = np.concatenate([data.y_train_raw, data.y_val_raw], axis=0)
        else:
            x, y = data.x_train, data.y_train_raw
        n_comp = min(n_comp, x.shape[0] - 1, x.shape[1], y.shape[1])
        self.model = PLSRegression(n_components=n_comp, scale=False, max_iter=500)
        self.model.fit(x, y)
        path = self.model_dir() / "model.joblib"
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        result = FitResult(
            model_name=self.name,
            best_params={"n_components": n_comp},
            artifact_paths={"model": str(path)},
            constraint_strategy="none",
            notes="PLSRegression; no silent clip",
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
