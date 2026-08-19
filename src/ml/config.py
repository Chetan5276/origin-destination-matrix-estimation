"""Configuration and paths for Phase 3 ML pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src import NUM_OD_PAIRS, OUTPUT_DIR, PROJECT_ROOT, REPORT_DIR

ML_OUTPUT_DIR = OUTPUT_DIR / "ml"
ML_REPORT_DIR = REPORT_DIR / "ml"


@dataclass(frozen=True)
class TrainConfig:
    """Training and benchmarking configuration."""

    seed: int = 42
    train_frac: float = 0.8
    val_frac: float = 0.1
    test_frac: float = 0.1
    max_samples: int | None = 50_000
    standardize_x: bool = True
    standardize_y: bool = True
    cv_folds: int = 3
    n_jobs: int = -1
    use_clean_turning: bool = True


@dataclass(frozen=True)
class NeuralTrainConfig:
    """Neural OD estimator hyperparameters."""

    hidden: tuple[int, ...] = (512, 1024)
    num_res_blocks: int = 1
    latent_hidden: tuple[int, ...] = (256, 128)
    activation: str = "gelu"
    lr: float = 1e-3
    finetune_lr: float = 1e-4
    epochs: int = 40
    autoencoder_epochs: int = 40
    finetune_epochs: int = 15
    batch_size: int = 256
    weight_decay: float = 1e-5
    od_loss_weight: float = 1.0
    forward_weight: float = 0.5
    production_weight: float = 0.1
    attraction_weight: float = 0.1
    latent_dims: tuple[int, ...] = (32, 64)
    residual_learning: bool = False
    enforce_sparsity_mask: bool = True
    output_activation: str = "softplus"


NUM_TURNS = 178
