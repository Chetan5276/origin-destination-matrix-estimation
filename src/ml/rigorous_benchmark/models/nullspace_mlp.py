"""Null-space-aware MLP: y = A+ x + N z(x)."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from src.ml.config import NeuralTrainConfig
from src.ml.neural_core import DirectODRegressor
from src.ml.neural_trainer import get_device
from src.ml.rigorous_benchmark.data import BenchmarkData, transform_x
from src.ml.rigorous_benchmark.hpo import run_study, save_best_params, val_composite_from_preds
from src.ml.rigorous_benchmark.models.base import BenchmarkModel, FitResult
from src.ml.rigorous_benchmark.neural_train import (
    bundle_from_data,
    neural_config_from_benchmark,
    suggest_neural_params,
    _x_raw_from_scaled,
)
from src.ml.neural_trainer import composite_loss
from src.ml.rigorous_benchmark.operator import OperatorInfo, pinv_predict


class NullspaceNet(nn.Module):
    """Maps scaled turning counts → null-space coefficients z."""

    def __init__(self, n_in: int, nullity: int, cfg: NeuralTrainConfig):
        super().__init__()
        # Reuse MLP trunk with output = nullity
        self.net = DirectODRegressor(n_in, nullity, cfg, residual_blocks=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NullspaceMLPModel(BenchmarkModel):
    name = "nullspace_mlp"

    def __init__(self, config):
        super().__init__(config)
        self.torch_model: NullspaceNet | None = None
        self.cfg = None
        self.bundle = None
        self.params: dict[str, Any] = {}
        self.null_penalty: float = 0.1

    def _predict_od(
        self,
        model: NullspaceNet,
        x_scaled: np.ndarray,
        x_raw: np.ndarray,
        operator: OperatorInfo,
        bundle,
        cfg,
    ) -> np.ndarray:
        device = get_device()
        model = model.to(device).eval()
        n_basis = torch.tensor(operator.null_basis, dtype=torch.float32, device=device)
        y_part = torch.tensor(pinv_predict(operator.a_pinv, x_raw), dtype=torch.float32, device=device)
        with torch.no_grad():
            z = model(torch.tensor(x_scaled, dtype=torch.float32, device=device))
            if operator.nullity == 0:
                y = y_part
            else:
                y = y_part + z @ n_basis.T
            # Apply nonnegativity softplus in OD space (report strategy)
            if cfg.output_activation == "softplus":
                y = torch.nn.functional.softplus(y)
            elif cfg.output_activation == "relu":
                y = torch.relu(y)
            # zero diagonal
            from src import NUM_ZONES

            n = NUM_ZONES
            mat = y.view(-1, n, n)
            diag = torch.eye(n, device=device).bool()
            mat = mat.masked_fill(diag.unsqueeze(0), 0.0)
            y = mat.view(y.shape[0], -1)
            if cfg.enforce_sparsity_mask and bundle.support_mask is not None:
                mask = torch.tensor(bundle.support_mask.astype(np.float32), device=device)
                y = y * mask
        return y.cpu().numpy().astype(np.float32)

    def _train(
        self,
        data: BenchmarkData,
        operator: OperatorInfo,
        cfg,
        bundle,
        x_tr,
        y_tr_raw,
        x_tr_raw,
        x_va,
        y_va_raw,
        x_va_raw,
        null_penalty: float,
    ) -> tuple[NullspaceNet, dict]:
        device = get_device()
        nullity = max(operator.nullity, 1)
        # If nullity==0, still train a dummy 1-dim head but ignore it
        model = NullspaceNet(data.n_features, nullity if operator.nullity > 0 else 1, cfg).to(device)
        n_basis = torch.tensor(operator.null_basis, dtype=torch.float32, device=device)
        a_t = torch.tensor(operator.a_turn, dtype=torch.float32, device=device)
        a_pinv = torch.tensor(operator.a_pinv, dtype=torch.float32, device=device)
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

        from torch.utils.data import DataLoader, TensorDataset
        import copy

        loader = DataLoader(
            TensorDataset(
                torch.tensor(x_tr, dtype=torch.float32),
                torch.tensor(x_tr_raw, dtype=torch.float32),
                torch.tensor(y_tr_raw, dtype=torch.float32),
            ),
            batch_size=cfg.batch_size,
            shuffle=True,
        )
        best = copy.deepcopy(model.state_dict())
        best_val = float("inf")
        stale = 0
        history: dict = {"val_mae": []}

        for epoch in range(cfg.epochs):
            model.train()
            for xs, xr, yr in loader:
                xs, xr, yr = xs.to(device), xr.to(device), yr.to(device)
                z = model(xs)
                y_part = xr @ a_pinv.T
                if operator.nullity > 0:
                    y = y_part + z @ n_basis.T
                    # soft penalty ||A (N z)|| ≈ ||A y - x|| for null component
                    null_term = (z @ n_basis.T) @ a_t.T
                    penalty = null_penalty * torch.mean(null_term**2)
                else:
                    y = y_part
                    penalty = 0.0 * z.sum()
                if cfg.output_activation == "softplus":
                    y_c = torch.nn.functional.softplus(y)
                elif cfg.output_activation == "relu":
                    y_c = torch.relu(y)
                else:
                    y_c = y
                from src import NUM_ZONES

                n = NUM_ZONES
                mat = y_c.view(-1, n, n)
                diag = torch.eye(n, device=device).bool()
                mat = mat.masked_fill(diag.unsqueeze(0), 0.0)
                y_c = mat.view(y_c.shape[0], -1)
                loss = composite_loss(y_c, yr, xr, a_t, cfg) + penalty
                opt.zero_grad()
                loss.backward()
                opt.step()

            model.eval()
            with torch.no_grad():
                pred = self._predict_od(
                    model, x_va, x_va_raw, operator, bundle, cfg
                )
                val_mae = float(np.mean(np.abs(pred - y_va_raw)))
            history["val_mae"].append(val_mae)
            if val_mae < best_val - 1e-6:
                best_val = val_mae
                best = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
                if stale >= self.config.early_stopping_patience:
                    break
        model.load_state_dict(best)
        history["best_val_mae"] = best_val
        return model, history

    def hyperopt(self, data: BenchmarkData, operator: OperatorInfo) -> dict[str, Any]:
        from src.ml.rigorous_benchmark.neural_train import capped_hpo_val

        x_tr, _, x_tr_raw, y_tr_raw = data.hpo_train_subset(
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
            null_penalty = trial.suggest_float("null_penalty", 1e-4, 10.0, log=True)
            cfg = neural_config_from_benchmark(self.config, p)
            model, _ = self._train(
                data,
                operator,
                cfg,
                bundle,
                x_tr,
                y_tr_raw,
                x_tr_raw,
                x_va,
                y_va_raw,
                x_va_raw,
                null_penalty,
            )
            pred = self._predict_od(model, x_va, x_va_raw, operator, bundle, cfg)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return val_composite_from_preds(
                y_va_raw, pred, x_va_raw, a,
                alpha=alpha_c, beta=beta, gamma=gamma, y_pinv=y_pinv,
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
        self.params = params
        self.null_penalty = float(params.get("null_penalty", 0.1))
        self.cfg = neural_config_from_benchmark(self.config, params)
        self.bundle = bundle_from_data(data)
        if use_train_val:
            x = np.concatenate([data.x_train, data.x_val], axis=0)
            xr = np.concatenate([data.x_train_raw, data.x_val_raw], axis=0)
            y = np.concatenate([data.y_train_raw, data.y_val_raw], axis=0)
        else:
            x, xr, y = data.x_train, data.x_train_raw, data.y_train_raw
        cap = self.config.final_train_cap
        if len(x) > cap:
            rng = np.random.default_rng(self.config.seed)
            idx = rng.choice(len(x), size=cap, replace=False)
            x, xr, y = x[idx], xr[idx], y[idx]
        n_hold = max(32, int(0.1 * len(x)))
        x_tr, xr_tr, y_tr = x[:-n_hold], xr[:-n_hold], y[:-n_hold]
        x_va, xr_va, y_va = x[-n_hold:], xr[-n_hold:], y[-n_hold:]

        self.torch_model, history = self._train(
            data,
            operator,
            self.cfg,
            self.bundle,
            x_tr,
            y_tr,
            xr_tr,
            x_va,
            y_va,
            xr_va,
            self.null_penalty,
        )
        path = self.model_dir() / "model.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.torch_model.state_dict(), "params": params}, path)
        result = FitResult(
            model_name=self.name,
            best_params=params,
            history=history,
            artifact_paths={"model": str(path)},
            constraint_strategy=self.cfg.output_activation,
            notes="y = A+x + N z(x); soft ||A N z|| penalty; constraints via activation",
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
        return self._predict_od(self.torch_model, xs, x_raw, operator, self.bundle, self.cfg)
