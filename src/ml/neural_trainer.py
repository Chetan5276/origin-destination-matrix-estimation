"""Training loops, composite losses, and OD autoencoder pretraining."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src import NUM_ZONES
from src.ml.config import NeuralTrainConfig
from src.ml.dataset import ODDataset, inverse_transform_y
from src.ml.metrics import evaluate_predictions_with_forward
from src.ml.neural_core import (
    DirectODRegressor,
    LatentODModel,
    LatentPredictor,
    ODAutoencoder,
    ODDecoder,
    ODEncoder,
)
from src.ml.od_constraints import apply_od_constraints_numpy

logger = logging.getLogger(__name__)


@dataclass
class PretrainResult:
    latent_dim: int
    reconstruction_rmse: float
    encoder_path: Path
    decoder_path: Path
    encoder: ODEncoder
    decoder: ODDecoder


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _scaler_tensors(dataset: ODDataset, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if dataset.y_scaler is None:
        return (
            torch.zeros(dataset.n_targets, device=device),
            torch.ones(dataset.n_targets, device=device),
        )
    mean = torch.tensor(dataset.y_scaler.mean_, dtype=torch.float32, device=device)
    scale = torch.tensor(dataset.y_scaler.scale_, dtype=torch.float32, device=device)
    return mean, scale


def _inverse_scale(y_scaled: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return y_scaled * scale + mean


def _scaled_residual_targets(y_raw: np.ndarray, dataset: ODDataset) -> np.ndarray:
    resid = y_raw - dataset.base_od_flat
    if dataset.y_scaler is not None:
        return dataset.y_scaler.transform(resid).astype(np.float32)
    return resid.astype(np.float32)


def logits_to_od_orig(
    logits: torch.Tensor,
    dataset: ODDataset,
    *,
    residual: bool,
    support_mask: np.ndarray | None,
    output_activation: str,
) -> torch.Tensor:
    """Map network logits → constrained OD in original units."""
    mean, scale = _scaler_tensors(dataset, logits.device)
    if residual:
        resid = _inverse_scale(logits, mean, scale)
        orig = resid + torch.tensor(dataset.base_od_flat, dtype=torch.float32, device=logits.device)
    else:
        orig = _inverse_scale(logits, mean, scale)

    if output_activation == "softplus":
        orig = torch.nn.functional.softplus(orig)
    else:
        orig = torch.relu(orig)

    n = NUM_ZONES
    mat = orig.view(-1, n, n)
    diag = torch.eye(n, device=orig.device).bool()
    mat = mat.masked_fill(diag.unsqueeze(0), 0.0)
    orig = mat.view(orig.shape[0], -1)

    if support_mask is not None:
        mask = torch.tensor(support_mask.astype(np.float32), device=orig.device)
        orig = orig * mask
    return orig


def _x_raw_batch(x_scaled: torch.Tensor, dataset: ODDataset) -> torch.Tensor:
    if dataset.x_scaler is None:
        return x_scaled
    arr = dataset.x_scaler.inverse_transform(x_scaled.detach().cpu().numpy())
    return torch.tensor(arr, dtype=torch.float32, device=x_scaled.device)


def composite_loss(
    pred_od: torch.Tensor,
    true_od: torch.Tensor,
    x_turn_raw: torch.Tensor,
    a_turn: torch.Tensor,
    config: NeuralTrainConfig,
) -> torch.Tensor:
    mse = nn.MSELoss()
    loss = config.od_loss_weight * mse(pred_od, true_od)
    if config.forward_weight > 0:
        turn_pred = pred_od @ a_turn.T
        loss = loss + config.forward_weight * mse(turn_pred, x_turn_raw)
    if config.production_weight > 0 or config.attraction_weight > 0:
        n = NUM_ZONES
        pred_m = pred_od.view(-1, n, n)
        true_m = true_od.view(-1, n, n)
        if config.production_weight > 0:
            loss = loss + config.production_weight * mse(pred_m.sum(2), true_m.sum(2))
        if config.attraction_weight > 0:
            loss = loss + config.attraction_weight * mse(pred_m.sum(1), true_m.sum(1))
    return loss


def _train_loop(
    model: nn.Module,
    dataset: ODDataset,
    a_turn: np.ndarray,
    config: NeuralTrainConfig,
    *,
    forward_fn,
    train_pairs: tuple[np.ndarray, np.ndarray],
    epochs: int,
    lr: float,
    trainable: list[nn.Module] | None = None,
) -> None:
    device = get_device()
    model = model.to(device)
    a_t = torch.tensor(a_turn, dtype=torch.float32, device=device)
    mask = dataset.support_mask if config.enforce_sparsity_mask else None

    x_all, y_all = train_pairs
    x_t = torch.tensor(x_all, dtype=torch.float32)
    y_t = torch.tensor(y_all, dtype=torch.float32)
    y_raw_t = torch.tensor(dataset.y_train_raw, dtype=torch.float32)
    loader = DataLoader(TensorDataset(x_t, y_t, y_raw_t), batch_size=config.batch_size, shuffle=True)

    params = [p for m in (trainable or [model]) for p in m.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr, weight_decay=config.weight_decay)

    for _ in range(epochs):
        model.train()
        for xb, yb, yb_raw in loader:
            xb = xb.to(device)
            yb_raw = yb_raw.to(device)
            pred_od = forward_fn(model, xb, dataset, config, mask)
            x_raw = _x_raw_batch(xb, dataset)
            loss = composite_loss(pred_od, yb_raw, x_raw, a_t, config)
            opt.zero_grad()
            loss.backward()
            opt.step()


def train_direct_model(
    dataset: ODDataset,
    a_turn: np.ndarray,
    config: NeuralTrainConfig,
    *,
    residual_blocks: bool = False,
    residual_learning: bool = False,
) -> DirectODRegressor:
    model = DirectODRegressor(
        dataset.n_features,
        dataset.n_targets,
        config,
        residual_blocks=residual_blocks,
    )
    mask = dataset.support_mask if config.enforce_sparsity_mask else None

    def forward_fn(m, xb, ds, cfg, msk):
        logits = m(xb)
        return logits_to_od_orig(logits, ds, residual=residual_learning, support_mask=msk, output_activation=cfg.output_activation)

    if residual_learning:
        y_target = _scaled_residual_targets(dataset.y_train_raw, dataset)
    else:
        y_target = dataset.y_train

    _train_loop(
        model,
        dataset,
        a_turn,
        config,
        forward_fn=forward_fn,
        train_pairs=(dataset.x_train, y_target),
        epochs=config.epochs,
        lr=config.lr,
    )
    return model


def predict_direct_model(
    model: DirectODRegressor,
    dataset: ODDataset,
    x: np.ndarray,
    config: NeuralTrainConfig,
    *,
    residual_learning: bool = False,
) -> np.ndarray:
    device = get_device()
    model = model.to(device)
    model.eval()
    mask = dataset.support_mask if config.enforce_sparsity_mask else None
    with torch.no_grad():
        xt = torch.tensor(x, dtype=torch.float32, device=device)
        logits = model(xt)
        pred = logits_to_od_orig(
            logits, dataset, residual=residual_learning, support_mask=mask, output_activation=config.output_activation
        )
    return apply_od_constraints_numpy(pred.cpu().numpy(), support_mask=mask)


def pretrain_od_autoencoder(
    dataset: ODDataset,
    latent_dim: int,
    config: NeuralTrainConfig,
    output_dir: Path,
) -> PretrainResult:
    device = get_device()
    encoder = ODEncoder(latent_dim, config).to(device)
    decoder = ODDecoder(latent_dim, config).to(device)
    ae = ODAutoencoder(encoder, decoder).to(device)
    mask = dataset.support_mask if config.enforce_sparsity_mask else None

    y_tr = torch.tensor(dataset.y_train, dtype=torch.float32)
    y_raw_tr = torch.tensor(dataset.y_train_raw, dtype=torch.float32)
    y_va = torch.tensor(dataset.y_val, dtype=torch.float32)
    loader = DataLoader(TensorDataset(y_tr, y_raw_tr), batch_size=config.batch_size, shuffle=True)
    opt = torch.optim.Adam(ae.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    mse = nn.MSELoss()

    for _ in range(config.autoencoder_epochs):
        ae.train()
        for yb, yb_raw in loader:
            yb = yb.to(device)
            yb_raw = yb_raw.to(device)
            logits = ae(yb)
            pred = logits_to_od_orig(
                logits, dataset, residual=False, support_mask=mask, output_activation=config.output_activation
            )
            loss = mse(pred, yb_raw)
            opt.zero_grad()
            loss.backward()
            opt.step()

    ae.eval()
    with torch.no_grad():
        logits = ae(y_va.to(device))
        pred = logits_to_od_orig(logits, dataset, residual=False, support_mask=mask, output_activation=config.output_activation)
        pred_np = pred.cpu().numpy()
    rmse = float(np.sqrt(np.mean((dataset.y_val_raw - pred_np) ** 2)))

    output_dir.mkdir(parents=True, exist_ok=True)
    enc_path = output_dir / f"od_encoder_lat{latent_dim}.pt"
    dec_path = output_dir / f"od_decoder_lat{latent_dim}.pt"
    torch.save(encoder.state_dict(), enc_path)
    torch.save(decoder.state_dict(), dec_path)
    logger.info("AE latent=%d val reconstruction RMSE=%.2f", latent_dim, rmse)
    return PretrainResult(latent_dim, rmse, enc_path, dec_path, encoder, decoder)


def train_latent_predictor(
    dataset: ODDataset,
    a_turn: np.ndarray,
    encoder: ODEncoder,
    decoder: ODDecoder,
    config: NeuralTrainConfig,
    *,
    freeze_decoder: bool = True,
    epochs: int | None = None,
    lr: float | None = None,
) -> LatentODModel:
    device = get_device()
    encoder = encoder.to(device).eval()
    decoder = decoder.to(device)
    if freeze_decoder:
        for p in decoder.parameters():
            p.requires_grad = False

    with torch.no_grad():
        z_train = encoder(torch.tensor(dataset.y_train, device=device)).cpu().numpy()

    latent_mlp = LatentPredictor(dataset.n_features, z_train.shape[1], config).to(device)
    model = LatentODModel(latent_mlp, decoder).to(device)
    mask = dataset.support_mask if config.enforce_sparsity_mask else None
    a_t = torch.tensor(a_turn, dtype=torch.float32, device=device)

    x_t = torch.tensor(dataset.x_train, dtype=torch.float32)
    z_t = torch.tensor(z_train, dtype=torch.float32)
    y_raw_t = torch.tensor(dataset.y_train_raw, dtype=torch.float32)
    loader = DataLoader(TensorDataset(x_t, z_t, y_raw_t), batch_size=config.batch_size, shuffle=True)
    params = list(latent_mlp.parameters()) if freeze_decoder else list(model.parameters())
    opt = torch.optim.Adam(params, lr=lr or config.lr, weight_decay=config.weight_decay)
    mse = nn.MSELoss()
    n_epochs = epochs or config.epochs

    for _ in range(n_epochs):
        model.train()
        for xb, zb, yb_raw in loader:
            xb = xb.to(device)
            zb = zb.to(device)
            yb_raw = yb_raw.to(device)
            z_pred = latent_mlp(xb)
            logits = decoder(z_pred)
            pred_od = logits_to_od_orig(
                logits, dataset, residual=False, support_mask=mask, output_activation=config.output_activation
            )
            latent_loss = mse(z_pred, zb)
            x_raw = _x_raw_batch(xb, dataset)
            od_loss = composite_loss(pred_od, yb_raw, x_raw, a_t, config)
            loss = latent_loss + od_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def finetune_latent_model(
    model: LatentODModel,
    dataset: ODDataset,
    a_turn: np.ndarray,
    config: NeuralTrainConfig,
) -> LatentODModel:
    for p in model.decoder.parameters():
        p.requires_grad = True
    mask = dataset.support_mask if config.enforce_sparsity_mask else None

    def forward_fn(m, xb, ds, cfg, msk):
        logits = m(xb)
        return logits_to_od_orig(logits, ds, residual=False, support_mask=msk, output_activation=cfg.output_activation)

    _train_loop(
        model,
        dataset,
        a_turn,
        config,
        forward_fn=forward_fn,
        train_pairs=(dataset.x_train, dataset.y_train),
        epochs=config.finetune_epochs,
        lr=config.finetune_lr,
        trainable=[model],
    )
    return model


def predict_latent_model(
    model: LatentODModel,
    dataset: ODDataset,
    x: np.ndarray,
    config: NeuralTrainConfig,
) -> np.ndarray:
    device = get_device()
    model = model.to(device)
    model.eval()
    mask = dataset.support_mask if config.enforce_sparsity_mask else None
    with torch.no_grad():
        xt = torch.tensor(x, dtype=torch.float32, device=device)
        logits = model(xt)
        pred = logits_to_od_orig(
            logits, dataset, residual=False, support_mask=mask, output_activation=config.output_activation
        )
    return apply_od_constraints_numpy(pred.cpu().numpy(), support_mask=mask)


def evaluate_model(
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    x_turn_raw: np.ndarray,
    a_turn: np.ndarray,
) -> dict:
    metrics = evaluate_predictions_with_forward(y_true, y_pred, x_turn_raw, a_turn)
    metrics["model"] = name
    return metrics
