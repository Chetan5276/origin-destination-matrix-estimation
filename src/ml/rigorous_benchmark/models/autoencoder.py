"""Autoencoder latent OD models (32 / 64 / 128 / 64-finetune)."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.ml.neural_core import LatentODModel, LatentPredictor, ODAutoencoder, ODDecoder, ODEncoder
from src.ml.neural_trainer import get_device
from src.ml.rigorous_benchmark.data import BenchmarkData, transform_x
from src.ml.rigorous_benchmark.hpo import run_study, save_best_params, val_composite_from_preds
from src.ml.rigorous_benchmark.models.base import BenchmarkModel, FitResult
from src.ml.rigorous_benchmark.neural_train import (
    bundle_from_data,
    logits_to_od,
    neural_config_from_benchmark,
    suggest_neural_params,
    _x_raw_from_scaled,
)
from src.ml.neural_trainer import composite_loss
from src.ml.rigorous_benchmark.operator import OperatorInfo


class AutoencoderModel(BenchmarkModel):
    def __init__(self, config, *, latent_dim: int, finetune: bool):
        super().__init__(config)
        self.latent_dim = latent_dim
        self.finetune = finetune
        self.name = f"ae_{latent_dim}" + ("_finetune" if finetune else "")
        self.torch_model: LatentODModel | None = None
        self.cfg = None
        self.bundle = None
        self.params: dict[str, Any] = {}

    def hyperopt(self, data: BenchmarkData, operator: OperatorInfo) -> dict[str, Any]:
        x_tr, y_tr_s, _, y_tr_raw = data.hpo_train_subset(
            self.config.hpo_train_subsample, seed=self.config.seed
        )
        bundle = bundle_from_data(data)
        a = operator.a_turn
        alpha_c, beta, gamma = self.config.alpha_fwd, self.config.beta_prod, self.config.gamma_attr
        y_pinv = (data.x_val_raw @ operator.a_pinv.T).astype(np.float32)

        def objective(trial):
            p = suggest_neural_params(trial, self.config)
            cfg = neural_config_from_benchmark(self.config, p)
            model = self._train_pipeline(
                data,
                operator,
                cfg,
                x_tr=x_tr,
                y_tr_scaled=y_tr_s,
                y_tr_raw=y_tr_raw,
                x_va=data.x_val,
                y_va_scaled=data.y_val,
                y_va_raw=data.y_val_raw,
                bundle=bundle,
                do_finetune=self.finetune,
            )
            pred = self._predict_np(model, data.x_val, bundle, cfg)
            return val_composite_from_preds(
                data.y_val_raw, pred, data.x_val_raw, a,
                alpha=alpha_c, beta=beta, gamma=gamma, y_pinv=y_pinv,
            )

        study = run_study(self.name, objective, self.config.n_trials, seed=self.config.seed)
        params = dict(study.best_params)
        save_best_params(self.model_dir() / "hpo_best.json", params, {"best_value": study.best_value})
        return params

    def _pretrain_ae(
        self,
        y_scaled: np.ndarray,
        y_raw: np.ndarray,
        y_val_scaled: np.ndarray,
        y_val_raw: np.ndarray,
        cfg,
        bundle,
    ) -> tuple[ODEncoder, ODDecoder, float]:
        device = get_device()
        encoder = ODEncoder(self.latent_dim, cfg).to(device)
        decoder = ODDecoder(self.latent_dim, cfg).to(device)
        ae = ODAutoencoder(encoder, decoder).to(device)
        opt = torch.optim.Adam(ae.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        loader = DataLoader(
            TensorDataset(
                torch.tensor(y_scaled, dtype=torch.float32),
                torch.tensor(y_raw, dtype=torch.float32),
            ),
            batch_size=cfg.batch_size,
            shuffle=True,
        )
        mse = nn.MSELoss()
        for _ in range(cfg.autoencoder_epochs):
            ae.train()
            for yb, yb_raw in loader:
                yb = yb.to(device)
                yb_raw = yb_raw.to(device)
                logits = ae(yb)
                pred = logits_to_od(
                    logits,
                    bundle,
                    residual=False,
                    output_activation=cfg.output_activation,
                    enforce_mask=cfg.enforce_sparsity_mask,
                )
                loss = mse(pred, yb_raw)
                opt.zero_grad()
                loss.backward()
                opt.step()
        ae.eval()
        with torch.no_grad():
            logits = ae(torch.tensor(y_val_scaled, dtype=torch.float32, device=device))
            pred = logits_to_od(
                logits,
                bundle,
                residual=False,
                output_activation=cfg.output_activation,
                enforce_mask=cfg.enforce_sparsity_mask,
            )
            rmse = float(torch.sqrt(torch.mean((pred - torch.tensor(y_val_raw, device=device)) ** 2)))
        return encoder, decoder, rmse

    def _train_pipeline(
        self,
        data: BenchmarkData,
        operator: OperatorInfo,
        cfg,
        *,
        x_tr,
        y_tr_scaled,
        y_tr_raw,
        x_va,
        y_va_scaled,
        y_va_raw,
        bundle,
        do_finetune: bool,
    ) -> LatentODModel:
        device = get_device()
        encoder, decoder, _ = self._pretrain_ae(
            y_tr_scaled, y_tr_raw, y_va_scaled, y_va_raw, cfg, bundle
        )
        encoder.eval()
        with torch.no_grad():
            z_train = encoder(torch.tensor(y_tr_scaled, device=device)).cpu().numpy()

        for p in decoder.parameters():
            p.requires_grad = False
        latent_mlp = LatentPredictor(data.n_features, self.latent_dim, cfg).to(device)
        model = LatentODModel(latent_mlp, decoder).to(device)
        a_t = torch.tensor(operator.a_turn, dtype=torch.float32, device=device)
        opt = torch.optim.Adam(latent_mlp.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        loader = DataLoader(
            TensorDataset(
                torch.tensor(x_tr, dtype=torch.float32),
                torch.tensor(z_train, dtype=torch.float32),
                torch.tensor(y_tr_raw, dtype=torch.float32),
            ),
            batch_size=cfg.batch_size,
            shuffle=True,
        )
        mse = nn.MSELoss()
        for _ in range(cfg.epochs):
            model.train()
            for xb, zb, yb in loader:
                xb, zb, yb = xb.to(device), zb.to(device), yb.to(device)
                z_pred = latent_mlp(xb)
                logits = decoder(z_pred)
                pred = logits_to_od(
                    logits,
                    bundle,
                    residual=False,
                    output_activation=cfg.output_activation,
                    enforce_mask=cfg.enforce_sparsity_mask,
                )
                x_raw = _x_raw_from_scaled(xb, bundle)
                loss = mse(z_pred, zb) + composite_loss(pred, yb, x_raw, a_t, cfg)
                opt.zero_grad()
                loss.backward()
                opt.step()

        if do_finetune:
            for p in model.decoder.parameters():
                p.requires_grad = True
            opt = torch.optim.Adam(model.parameters(), lr=cfg.finetune_lr, weight_decay=cfg.weight_decay)
            loader2 = DataLoader(
                TensorDataset(
                    torch.tensor(x_tr, dtype=torch.float32),
                    torch.tensor(y_tr_raw, dtype=torch.float32),
                ),
                batch_size=cfg.batch_size,
                shuffle=True,
            )
            for _ in range(cfg.finetune_epochs):
                model.train()
                for xb, yb in loader2:
                    xb, yb = xb.to(device), yb.to(device)
                    logits = model(xb)
                    pred = logits_to_od(
                        logits,
                        bundle,
                        residual=False,
                        output_activation=cfg.output_activation,
                        enforce_mask=cfg.enforce_sparsity_mask,
                    )
                    x_raw = _x_raw_from_scaled(xb, bundle)
                    loss = composite_loss(pred, yb, x_raw, a_t, cfg)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
        return model

    def _predict_np(self, model, x_scaled, bundle, cfg) -> np.ndarray:
        device = get_device()
        model = model.to(device).eval()
        with torch.no_grad():
            logits = model(torch.tensor(x_scaled, dtype=torch.float32, device=device))
            pred = logits_to_od(
                logits,
                bundle,
                residual=False,
                output_activation=cfg.output_activation,
                enforce_mask=cfg.enforce_sparsity_mask,
            )
        return pred.cpu().numpy().astype(np.float32)

    def fit(
        self,
        data: BenchmarkData,
        operator: OperatorInfo,
        params: dict[str, Any],
        *,
        use_train_val: bool = True,
    ) -> FitResult:
        self.params = params
        self.cfg = neural_config_from_benchmark(self.config, params)
        self.bundle = bundle_from_data(data)
        # Cap final train size to avoid OOM
        if use_train_val:
            x = np.concatenate([data.x_train, data.x_val], axis=0)
            ys = np.concatenate([data.y_train, data.y_val], axis=0)
            yr = np.concatenate([data.y_train_raw, data.y_val_raw], axis=0)
        else:
            x, ys, yr = data.x_train, data.y_train, data.y_train_raw
        cap = self.config.final_train_cap
        if len(x) > cap:
            rng = np.random.default_rng(self.config.seed)
            idx = rng.choice(len(x), size=cap, replace=False)
            x, ys, yr = x[idx], ys[idx], yr[idx]
        n_hold = max(32, int(0.1 * len(x)))
        x_tr, ys_tr, yr_tr = x[:-n_hold], ys[:-n_hold], yr[:-n_hold]
        x_va, ys_va, yr_va = x[-n_hold:], ys[-n_hold:], yr[-n_hold:]

        self.torch_model = self._train_pipeline(
            data,
            operator,
            self.cfg,
            x_tr=x_tr,
            y_tr_scaled=ys_tr,
            y_tr_raw=yr_tr,
            x_va=x_va,
            y_va_scaled=ys_va,
            y_va_raw=yr_va,
            bundle=self.bundle,
            do_finetune=self.finetune,
        )
        path = self.model_dir() / "model.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.torch_model.state_dict(),
                "params": params,
                "latent_dim": self.latent_dim,
                "finetune": self.finetune,
            },
            path,
        )
        result = FitResult(
            model_name=self.name,
            best_params=params,
            artifact_paths={"model": str(path)},
            constraint_strategy=self.cfg.output_activation,
            notes=f"AE latent={self.latent_dim} finetune={self.finetune}",
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
        return self._predict_np(self.torch_model, xs, self.bundle, self.cfg)
