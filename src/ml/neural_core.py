"""PyTorch modules for OD estimation and OD-only autoencoders."""

from __future__ import annotations

import torch
import torch.nn as nn

from src import NUM_OD_PAIRS
from src.ml.config import NeuralTrainConfig


def _activation(name: str):
    return nn.GELU if name == "gelu" else nn.ReLU


class ODEncoder(nn.Module):
    """OD matrix → latent (trained on OD only)."""

    def __init__(self, latent_dim: int, config: NeuralTrainConfig, n_od: int = NUM_OD_PAIRS):
        super().__init__()
        h1, h2 = config.latent_hidden
        act = _activation(config.activation)
        self.net = nn.Sequential(
            nn.Linear(n_od, h1),
            act(),
            nn.Linear(h1, h2),
            act(),
            nn.Linear(h2, latent_dim),
        )

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.net(y)


class ODDecoder(nn.Module):
    """Latent → OD (scaled / logit space)."""

    def __init__(self, latent_dim: int, config: NeuralTrainConfig, n_od: int = NUM_OD_PAIRS):
        super().__init__()
        h1, h2 = config.latent_hidden
        act = _activation(config.activation)
        self.net = nn.Sequential(
            nn.Linear(latent_dim, h2),
            act(),
            nn.Linear(h2, h1),
            act(),
            nn.Linear(h1, n_od),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class ODAutoencoder(nn.Module):
    """576 → latent → 576 autoencoder on OD matrices."""

    def __init__(self, encoder: ODEncoder, decoder: ODDecoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(y))

    def encode(self, y: torch.Tensor) -> torch.Tensor:
        return self.encoder(y)


class ResBlock(nn.Module):
    def __init__(self, dim: int, activation: str):
        super().__init__()
        act = _activation(activation)
        self.fc = nn.Sequential(nn.Linear(dim, dim), act(), nn.Linear(dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc(x)


class DirectODRegressor(nn.Module):
    """Turning counts → OD logits (MLP or residual MLP with variable depth)."""

    def __init__(
        self,
        n_in: int,
        n_out: int,
        config: NeuralTrainConfig,
        *,
        residual_blocks: bool = False,
    ):
        super().__init__()
        act = _activation(config.activation)
        hidden = list(config.hidden)
        if len(hidden) < 1:
            raise ValueError("config.hidden must have at least one layer width")

        self.residual_blocks = residual_blocks
        if residual_blocks:
            width = hidden[0]
            self.in_proj = nn.Linear(n_in, width)
            self.blocks = nn.ModuleList(
                [ResBlock(width, config.activation) for _ in range(config.num_res_blocks)]
            )
            self.trunk = None
            self.out_proj = nn.Linear(width, n_out)
        else:
            dims = [n_in, *hidden]
            trunk: list[nn.Module] = []
            for i in range(len(dims) - 1):
                trunk.append(nn.Linear(dims[i], dims[i + 1]))
                trunk.append(act())
            self.trunk = nn.Sequential(*trunk)
            self.in_proj = None
            self.blocks = nn.ModuleList()
            self.out_proj = nn.Linear(dims[-1], n_out)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.residual_blocks:
            assert self.in_proj is not None
            h = self.act(self.in_proj(x))
            for block in self.blocks:
                h = block(h)
            return self.out_proj(h)
        assert self.trunk is not None
        return self.out_proj(self.trunk(x))


class LatentPredictor(nn.Module):
    """Turning counts → latent vector."""

    def __init__(self, n_in: int, latent_dim: int, config: NeuralTrainConfig):
        super().__init__()
        act = _activation(config.activation)
        h0, h1 = config.hidden
        self.net = nn.Sequential(
            nn.Linear(n_in, h0),
            act(),
            nn.Linear(h0, h1),
            act(),
            nn.Linear(h1, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LatentODModel(nn.Module):
    """Turning → latent MLP → decoder."""

    def __init__(self, latent_predictor: LatentPredictor, decoder: ODDecoder):
        super().__init__()
        self.latent_predictor = latent_predictor
        self.decoder = decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.latent_predictor(x))
