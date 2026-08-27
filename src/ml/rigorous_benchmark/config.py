"""Configuration for the rigorous OD estimation benchmark."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from src import BASE_OD_PATH, NUM_OD_PAIRS, NUM_ZONES, OUTPUT_DIR

BENCHMARK_OUTPUT_DIR = OUTPUT_DIR / "benchmark"

ALL_MODEL_NAMES: tuple[str, ...] = (
    "moore_penrose",
    "tikhonov",
    "ridge",
    "physics_ridge",
    "pls",
    "mlp",
    "residual_mlp",
    "physics_residual_mlp",
    "ae_32",
    "ae_64",
    "ae_128",
    "ae_64_finetune",
    "nullspace_mlp",
)


@dataclass
class BenchmarkConfig:
    """Paths, split, Optuna budget, loss weights, and model enable list."""

    # Paths
    synthetic_od_path: Path = field(
        default_factory=lambda: OUTPUT_DIR
        / "od_generator_fp"
        / "synthetic_od_fp_synthetics_only.npy"
    )
    turning_counts_path: Path = field(
        default_factory=lambda: OUTPUT_DIR / "turning_counts_fp" / "turning_counts.npy"
    )
    a_turn_path: Path = field(
        default_factory=lambda: OUTPUT_DIR / "turning_counts_fp" / "A_turn.npy"
    )
    survey_od_path: Path = field(default_factory=lambda: BASE_OD_PATH)
    output_dir: Path = field(default_factory=lambda: BENCHMARK_OUTPUT_DIR)

    # Split
    seed: int = 42
    train_frac: float = 0.70
    val_frac: float = 0.15
    test_frac: float = 0.15
    n_synthetics: int = 100_000
    survey_turning_index: int = 100_000

    # HPO / final protocol (defaults tuned to avoid OOM on ~16GB hosts)
    hpo_train_subsample: int = 5_000
    n_trials: int = 8
    max_samples: int | None = None  # override for smoke (caps synthetics used)
    final_retrain_on_train_val: bool = True
    final_train_cap: int = 25_000  # max train(+val) rows for neural final fit
    hpo_val_cap: int = 3_000  # val rows used inside Optuna objectives
    metrics_spearman_samples: int = 500  # subsample for expensive Spearman

    # Composite selection weights (val only)
    alpha_fwd: float = 1.0
    beta_prod: float = 0.5
    gamma_attr: float = 0.5

    # Preprocessing
    standardize_x: bool = True
    standardize_y: bool = True

    # Constraints (reported explicitly; default softplus for neural)
    constraint_strategy: str = "softplus"  # none | relu | softplus | clip | clip_ipf

    # Neural defaults (overridden by HPO)
    neural_epochs: int = 25
    neural_batch_size: int = 128
    neural_lr: float = 1e-3
    neural_weight_decay: float = 1e-5
    early_stopping_patience: int = 6
    autoencoder_epochs: int = 20
    finetune_epochs: int = 10
    finetune_lr: float = 1e-4
    od_loss_weight: float = 1.0
    forward_weight: float = 0.5
    production_weight: float = 0.1
    attraction_weight: float = 0.1
    enforce_sparsity_mask: bool = True

    # Operator
    svd_rtol: float = 1e-10

    # Models
    models: tuple[str, ...] = ALL_MODEL_NAMES
    run_ablations: bool = True
    n_jobs: int = -1
    device: str | None = None  # auto

    # Smoke / pragmatic flags
    smoke: bool = False

    def __post_init__(self) -> None:
        self.synthetic_od_path = Path(self.synthetic_od_path)
        self.turning_counts_path = Path(self.turning_counts_path)
        self.a_turn_path = Path(self.a_turn_path)
        self.survey_od_path = Path(self.survey_od_path)
        self.output_dir = Path(self.output_dir)
        if abs(self.train_frac + self.val_frac + self.test_frac - 1.0) > 1e-9:
            raise ValueError("train_frac + val_frac + test_frac must equal 1")
        unknown = set(self.models) - set(ALL_MODEL_NAMES)
        if unknown:
            raise ValueError(f"Unknown models: {sorted(unknown)}")

    @property
    def n_od(self) -> int:
        return NUM_OD_PAIRS

    @property
    def n_zones(self) -> int:
        return NUM_ZONES

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k, v in list(d.items()):
            if isinstance(v, Path):
                d[k] = str(v)
            elif isinstance(v, tuple):
                d[k] = list(v)
        return d

    def with_updates(self, **kwargs: Any) -> "BenchmarkConfig":
        return replace(self, **kwargs)

    def data_dir(self) -> Path:
        return self.output_dir / "data"

    def preprocessing_dir(self) -> Path:
        return self.output_dir / "preprocessing"

    def operator_dir(self) -> Path:
        return self.output_dir / "operator"

    def model_dir(self, name: str) -> Path:
        return self.output_dir / name

    def ablations_dir(self) -> Path:
        return self.output_dir / "ablations"

    def survey_dir(self) -> Path:
        return self.output_dir / "survey_inference"

    def figures_dir(self) -> Path:
        return self.output_dir / "figures"


def smoke_config(**overrides: Any) -> BenchmarkConfig:
    """Tiny config for end-to-end smoke tests."""
    defaults = dict(
        smoke=True,
        max_samples=512,
        hpo_train_subsample=256,
        n_trials=2,
        neural_epochs=2,
        autoencoder_epochs=2,
        finetune_epochs=1,
        early_stopping_patience=2,
        neural_batch_size=64,
        run_ablations=True,
        n_jobs=1,
    )
    defaults.update(overrides)
    return BenchmarkConfig(**defaults)
