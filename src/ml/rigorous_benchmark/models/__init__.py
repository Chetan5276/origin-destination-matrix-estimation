"""Model registry."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.ml.rigorous_benchmark.config import BenchmarkConfig
    from src.ml.rigorous_benchmark.models.base import BenchmarkModel


def build_model(name: str, config: "BenchmarkConfig") -> "BenchmarkModel":
    from src.ml.rigorous_benchmark.models.autoencoder import AutoencoderModel
    from src.ml.rigorous_benchmark.models.mlp import MLPModel
    from src.ml.rigorous_benchmark.models.nullspace_mlp import NullspaceMLPModel
    from src.ml.rigorous_benchmark.models.physics_residual_mlp import PhysicsResidualMLPModel
    from src.ml.rigorous_benchmark.models.physics_ridge import PhysicsRidgeModel
    from src.ml.rigorous_benchmark.models.pinv import MoorePenroseModel
    from src.ml.rigorous_benchmark.models.pls import PLSModel
    from src.ml.rigorous_benchmark.models.residual_mlp import ResidualMLPModel
    from src.ml.rigorous_benchmark.models.ridge import RidgeModel
    from src.ml.rigorous_benchmark.models.tikhonov import TikhonovModel

    registry = {
        "moore_penrose": MoorePenroseModel,
        "tikhonov": TikhonovModel,
        "ridge": RidgeModel,
        "physics_ridge": PhysicsRidgeModel,
        "pls": PLSModel,
        "mlp": MLPModel,
        "residual_mlp": ResidualMLPModel,
        "physics_residual_mlp": PhysicsResidualMLPModel,
        "ae_32": lambda cfg: AutoencoderModel(cfg, latent_dim=32, finetune=False),
        "ae_64": lambda cfg: AutoencoderModel(cfg, latent_dim=64, finetune=False),
        "ae_128": lambda cfg: AutoencoderModel(cfg, latent_dim=128, finetune=False),
        "ae_64_finetune": lambda cfg: AutoencoderModel(cfg, latent_dim=64, finetune=True),
        "nullspace_mlp": NullspaceMLPModel,
    }
    if name not in registry:
        raise KeyError(f"Unknown model {name}")
    factory = registry[name]
    return factory(config)
