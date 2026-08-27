"""Residual MLP OD regressor."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from src.ml.neural_core import DirectODRegressor
from src.ml.rigorous_benchmark.data import BenchmarkData, transform_x
from src.ml.rigorous_benchmark.hpo import run_study, save_best_params, val_composite_from_preds
from src.ml.rigorous_benchmark.models.base import BenchmarkModel, FitResult
from src.ml.rigorous_benchmark.neural_train import (
    bundle_from_data,
    capped_hpo_val,
    capped_train_arrays,
    neural_config_from_benchmark,
    predict_regressor,
    suggest_neural_params,
    train_regressor_with_early_stopping,
)
from src.ml.rigorous_benchmark.operator import OperatorInfo


class ResidualMLPModel(BenchmarkModel):
    name = "residual_mlp"

    def __init__(self, config):
        super().__init__(config)
        self.torch_model: DirectODRegressor | None = None
        self.cfg = None
        self.bundle = None
        self.params: dict[str, Any] = {}

    def hyperopt(self, data: BenchmarkData, operator: OperatorInfo) -> dict[str, Any]:
        x_tr, _, _, y_tr_raw = data.hpo_train_subset(
            self.config.hpo_train_subsample, seed=self.config.seed
        )
        x_va, y_va_raw, x_va_raw = capped_hpo_val(
            data, self.config.hpo_val_cap, self.config.seed + 1
        )
        bundle = bundle_from_data(data)
        a = operator.a_turn
        alpha_c, beta, gamma = self.config.alpha_fwd, self.config.beta_prod, self.config.gamma_attr
        y_pinv = (x_va_raw @ operator.a_pinv.T).astype(np.float32)

        def objective(trial):
            p = suggest_neural_params(trial, self.config)
            p["num_res_blocks"] = trial.suggest_int("num_res_blocks", 1, 3)
            p["hidden0"] = trial.suggest_categorical("width", [128, 256, 512])
            p["hidden1"] = p["hidden0"]
            cfg = neural_config_from_benchmark(self.config, p)
            model = DirectODRegressor(data.n_features, data.n_targets, cfg, residual_blocks=True)
            model, _ = train_regressor_with_early_stopping(
                model,
                x_train=x_tr,
                y_train_raw=y_tr_raw,
                x_val=x_va,
                y_val_raw=y_va_raw,
                a_turn=a,
                bundle=bundle,
                cfg=cfg,
                patience=self.config.early_stopping_patience,
            )
            pred = predict_regressor(model, x_va, bundle, cfg)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return val_composite_from_preds(
                y_va_raw, pred, x_va_raw, a,
                alpha=alpha_c, beta=beta, gamma=gamma, y_pinv=y_pinv,
            )

        study = run_study(self.name, objective, self.config.n_trials, seed=self.config.seed)
        params = dict(study.best_params)
        if "width" in params:
            params["hidden0"] = params["width"]
            params["hidden1"] = params["width"]
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
        p = dict(params)
        if "width" in p:
            p["hidden0"] = p["width"]
            p["hidden1"] = p["width"]
        self.params = p
        self.cfg = neural_config_from_benchmark(self.config, p)
        self.bundle = bundle_from_data(data)
        x_tr, y_tr, x_va, y_va = capped_train_arrays(
            data,
            use_train_val=use_train_val,
            cap=self.config.final_train_cap,
            seed=self.config.seed,
        )

        self.torch_model = DirectODRegressor(
            data.n_features, data.n_targets, self.cfg, residual_blocks=True
        )
        self.torch_model, history = train_regressor_with_early_stopping(
            self.torch_model,
            x_train=x_tr,
            y_train_raw=y_tr,
            x_val=x_va,
            y_val_raw=y_va,
            a_turn=operator.a_turn,
            bundle=self.bundle,
            cfg=self.cfg,
            patience=self.config.early_stopping_patience,
        )
        path = self.model_dir() / "model.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.torch_model.state_dict(), "params": p}, path)
        result = FitResult(
            model_name=self.name,
            best_params=p,
            history=history,
            artifact_paths={"model": str(path)},
            constraint_strategy=self.cfg.output_activation,
            notes="Residual MLP blocks; constraints via output_activation",
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
        assert self.torch_model is not None and self.cfg is not None and self.bundle is not None
        xs = transform_x(data, x_raw)
        return predict_regressor(self.torch_model, xs, self.bundle, self.cfg)
