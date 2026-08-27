"""Neural training with early stopping on validation (reuses neural_core / composite_loss)."""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src import NUM_ZONES
from src.ml.config import NeuralTrainConfig
from src.ml.neural_trainer import composite_loss, get_device
from src.ml.od_constraints import apply_od_constraints_numpy
from src.ml.rigorous_benchmark.config import BenchmarkConfig
from src.ml.rigorous_benchmark.data import BenchmarkData

logger = logging.getLogger(__name__)


def neural_config_from_benchmark(
    config: BenchmarkConfig,
    overrides: dict[str, Any] | None = None,
) -> NeuralTrainConfig:
    kw = dict(
        hidden=(512, 1024),
        num_res_blocks=2,
        latent_hidden=(256, 128),
        activation="gelu",
        lr=config.neural_lr,
        finetune_lr=config.finetune_lr,
        epochs=config.neural_epochs,
        autoencoder_epochs=config.autoencoder_epochs,
        finetune_epochs=config.finetune_epochs,
        batch_size=config.neural_batch_size,
        weight_decay=config.neural_weight_decay,
        od_loss_weight=config.od_loss_weight,
        forward_weight=config.forward_weight,
        production_weight=config.production_weight,
        attraction_weight=config.attraction_weight,
        enforce_sparsity_mask=config.enforce_sparsity_mask,
        output_activation=config.constraint_strategy
        if config.constraint_strategy in ("softplus", "relu")
        else "softplus",
    )
    if overrides:
        # map Optuna keys
        if "hidden0" in overrides and "hidden1" in overrides:
            kw["hidden"] = (int(overrides["hidden0"]), int(overrides["hidden1"]))
        if "num_res_blocks" in overrides:
            kw["num_res_blocks"] = int(overrides["num_res_blocks"])
        if "lr" in overrides:
            kw["lr"] = float(overrides["lr"])
        if "weight_decay" in overrides:
            kw["weight_decay"] = float(overrides["weight_decay"])
        if "batch_size" in overrides:
            kw["batch_size"] = int(overrides["batch_size"])
        if "forward_weight" in overrides:
            kw["forward_weight"] = float(overrides["forward_weight"])
        if "production_weight" in overrides:
            kw["production_weight"] = float(overrides["production_weight"])
        if "attraction_weight" in overrides:
            kw["attraction_weight"] = float(overrides["attraction_weight"])
        if "epochs" in overrides:
            kw["epochs"] = int(overrides["epochs"])
        if "output_activation" in overrides:
            kw["output_activation"] = str(overrides["output_activation"])
    return NeuralTrainConfig(**kw)


@dataclass
class ScalerBundle:
    x_mean: np.ndarray | None
    x_scale: np.ndarray | None
    y_mean: np.ndarray | None
    y_scale: np.ndarray | None
    base_od_flat: np.ndarray
    support_mask: np.ndarray | None


def bundle_from_data(data: BenchmarkData) -> ScalerBundle:
    return ScalerBundle(
        x_mean=None if data.x_scaler is None else data.x_scaler.mean_.astype(np.float32),
        x_scale=None if data.x_scaler is None else data.x_scaler.scale_.astype(np.float32),
        y_mean=None if data.y_scaler is None else data.y_scaler.mean_.astype(np.float32),
        y_scale=None if data.y_scaler is None else data.y_scaler.scale_.astype(np.float32),
        base_od_flat=data.base_od_flat.astype(np.float32),
        support_mask=data.support_mask if True else None,
    )


def logits_to_od(
    logits: torch.Tensor,
    bundle: ScalerBundle,
    *,
    residual: bool,
    output_activation: str,
    enforce_mask: bool,
) -> torch.Tensor:
    device = logits.device
    if bundle.y_mean is not None:
        mean = torch.tensor(bundle.y_mean, device=device)
        scale = torch.tensor(bundle.y_scale, device=device)
        orig = logits * scale + mean
    else:
        orig = logits
    if residual:
        orig = orig + torch.tensor(bundle.base_od_flat, device=device)

    if output_activation == "softplus":
        orig = torch.nn.functional.softplus(orig)
    elif output_activation == "relu":
        orig = torch.relu(orig)
    # else: none — leave unconstrained

    n = NUM_ZONES
    mat = orig.view(-1, n, n)
    diag = torch.eye(n, device=device).bool()
    mat = mat.masked_fill(diag.unsqueeze(0), 0.0)
    orig = mat.view(orig.shape[0], -1)
    if enforce_mask and bundle.support_mask is not None:
        mask = torch.tensor(bundle.support_mask.astype(np.float32), device=device)
        orig = orig * mask
    return orig


def _x_raw_from_scaled(xb: torch.Tensor, bundle: ScalerBundle) -> torch.Tensor:
    if bundle.x_mean is None:
        return xb
    mean = torch.tensor(bundle.x_mean, device=xb.device)
    scale = torch.tensor(bundle.x_scale, device=xb.device)
    return xb * scale + mean


def train_regressor_with_early_stopping(
    model: nn.Module,
    *,
    x_train: np.ndarray,
    y_train_raw: np.ndarray,
    x_val: np.ndarray,
    y_val_raw: np.ndarray,
    a_turn: np.ndarray,
    bundle: ScalerBundle,
    cfg: NeuralTrainConfig,
    patience: int,
    residual_learning: bool = False,
    forward_extra_weight: float | None = None,
) -> tuple[nn.Module, dict]:
    """Train until val OD MAE stops improving; restore best weights."""
    import gc
    from dataclasses import replace

    device = get_device()
    model = model.to(device)
    a_t = torch.tensor(a_turn, dtype=torch.float32, device=device)
    if forward_extra_weight is not None:
        cfg = replace(cfg, forward_weight=float(forward_extra_weight))

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loader = DataLoader(
        TensorDataset(
            torch.tensor(x_train, dtype=torch.float32),
            torch.tensor(y_train_raw, dtype=torch.float32),
        ),
        batch_size=min(cfg.batch_size, 256),
        shuffle=True,
        pin_memory=device.type == "cuda",
    )

    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    stale = 0
    history: dict[str, list] = {"train_loss": [], "val_mae": []}

    def _val_mae() -> float:
        model.eval()
        errs = []
        with torch.no_grad():
            for i in range(0, len(x_val), cfg.batch_size):
                xb = torch.tensor(
                    x_val[i : i + cfg.batch_size], dtype=torch.float32, device=device
                )
                yb = torch.tensor(
                    y_val_raw[i : i + cfg.batch_size], dtype=torch.float32, device=device
                )
                logits = model(xb)
                pred = logits_to_od(
                    logits,
                    bundle,
                    residual=residual_learning,
                    output_activation=cfg.output_activation,
                    enforce_mask=cfg.enforce_sparsity_mask,
                )
                errs.append(torch.mean(torch.abs(pred - yb)).item())
        return float(np.mean(errs)) if errs else float("inf")

    for epoch in range(cfg.epochs):
        model.train()
        losses = []
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            pred = logits_to_od(
                logits,
                bundle,
                residual=residual_learning,
                output_activation=cfg.output_activation,
                enforce_mask=cfg.enforce_sparsity_mask,
            )
            x_raw = _x_raw_from_scaled(xb, bundle)
            loss = composite_loss(pred, yb, x_raw, a_t, cfg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        history["train_loss"].append(float(np.mean(losses)) if losses else float("nan"))

        val_mae = _val_mae()
        history["val_mae"].append(val_mae)
        if val_mae < best_val - 1e-6:
            best_val = val_mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                logger.info("Early stop at epoch %d (best val MAE=%.4g)", epoch + 1, best_val)
                break

    model.load_state_dict(best_state)
    history["best_val_mae"] = best_val
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return model, history


def predict_regressor(
    model: nn.Module,
    x_scaled: np.ndarray,
    bundle: ScalerBundle,
    cfg: NeuralTrainConfig,
    *,
    residual_learning: bool = False,
    post_constraint: str | None = None,
    batch_size: int | None = None,
) -> np.ndarray:
    device = get_device()
    model = model.to(device)
    model.eval()
    bs = batch_size or min(cfg.batch_size, 512)
    outs = []
    with torch.no_grad():
        for i in range(0, len(x_scaled), bs):
            xt = torch.tensor(x_scaled[i : i + bs], dtype=torch.float32, device=device)
            logits = model(xt)
            pred = logits_to_od(
                logits,
                bundle,
                residual=residual_learning,
                output_activation=cfg.output_activation,
                enforce_mask=cfg.enforce_sparsity_mask,
            )
            outs.append(pred.cpu().numpy().astype(np.float32))
    out = np.concatenate(outs, axis=0) if outs else np.zeros((0, 576), dtype=np.float32)
    if post_constraint == "clip":
        out = apply_od_constraints_numpy(out, support_mask=bundle.support_mask)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def suggest_neural_params(trial, config: BenchmarkConfig) -> dict[str, Any]:
    return {
        "hidden0": trial.suggest_categorical("hidden0", [128, 256, 512]),
        "hidden1": trial.suggest_categorical("hidden1", [256, 512]),
        "lr": trial.suggest_float("lr", 1e-4, 2e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
        "forward_weight": trial.suggest_float("forward_weight", 0.0, 1.5),
        "output_activation": trial.suggest_categorical(
            "output_activation", ["softplus", "relu"]
        ),
    }


def capped_train_arrays(
    data: BenchmarkData,
    *,
    use_train_val: bool,
    cap: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (x_tr, y_tr_raw, x_va, y_va_raw) with training capped for RAM."""
    if use_train_val:
        x = np.concatenate([data.x_train, data.x_val], axis=0)
        y = np.concatenate([data.y_train_raw, data.y_val_raw], axis=0)
    else:
        x, y = data.x_train, data.y_train_raw
    if len(x) > cap:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(x), size=cap, replace=False)
        x, y = x[idx], y[idx]
    n_hold = max(32, int(0.1 * len(x)))
    return x[:-n_hold], y[:-n_hold], x[-n_hold:], y[-n_hold:]


def capped_hpo_val(
    data: BenchmarkData, cap: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Subsample validation for Optuna objectives."""
    n = data.x_val.shape[0]
    if n <= cap:
        return data.x_val, data.y_val_raw, data.x_val_raw
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=cap, replace=False)
    return data.x_val[idx], data.y_val_raw[idx], data.x_val_raw[idx]
