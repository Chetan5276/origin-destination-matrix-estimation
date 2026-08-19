"""Dataset statistics, validation, and visualization."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

from src import NUM_OD_PAIRS, NUM_ZONES
from src.data.od_pairs import unflatten_od_vector

logger = logging.getLogger(__name__)


@dataclass
class DatasetArrays:
    """ML-ready turning-count inputs and OD targets."""

    x: np.ndarray
    y: np.ndarray
    a_turn: np.ndarray

    @property
    def num_samples(self) -> int:
        return self.x.shape[0]


def generate_turning_counts(y_flat: np.ndarray, a_turn: np.ndarray) -> np.ndarray:
    """Compute X = A_turn @ Y for each sample. y_flat shape (N, 576)."""
    return y_flat @ a_turn.T


def flatten_od_batch(od_matrices: np.ndarray) -> np.ndarray:
    """Flatten (N, 24, 24) OD tensors to (N, 576)."""
    if od_matrices.ndim != 3:
        raise ValueError("Expected OD array with shape (N, 24, 24)")
    if od_matrices.shape[1:] != (NUM_ZONES, NUM_ZONES):
        raise ValueError(f"Expected zone dimensions ({NUM_ZONES}, {NUM_ZONES})")
    return od_matrices.reshape(od_matrices.shape[0], -1, order="C")


def build_dataset(
    od_matrices: np.ndarray,
    a_turn: np.ndarray,
) -> DatasetArrays:
    """Build X (N, num_turns) and Y (N, 576) from synthetic OD matrices."""
    y = flatten_od_batch(od_matrices)
    x = generate_turning_counts(y, a_turn)
    return DatasetArrays(x=x, y=y, a_turn=a_turn)


def validate_dataset(dataset: DatasetArrays, atol: float = 1e-6) -> dict[str, bool | int]:
    """Run consistency checks on shapes and X = A_turn @ Y."""
    n, n_turns = dataset.x.shape
    _, n_od = dataset.y.shape
    checks: dict[str, bool | int] = {}

    checks["x_shape_ok"] = dataset.x.shape == (n, n_turns)
    checks["y_shape_ok"] = dataset.y.shape == (n, NUM_OD_PAIRS)
    checks["od_matrix_shape_ok"] = all(
        unflatten_od_vector(row).shape == (NUM_ZONES, NUM_ZONES) for row in dataset.y[: min(10, n)]
    )
    checks["turning_vector_shape_ok"] = dataset.x.shape[1] == dataset.a_turn.shape[0]

    reconstructed = generate_turning_counts(dataset.y, dataset.a_turn)
    max_err = float(np.max(np.abs(reconstructed - dataset.x)))
    checks["matrix_mult_consistent"] = max_err <= atol
    checks["max_reconstruction_error"] = max_err

    logger.info("Validation: %s", checks)
    return checks


def turning_count_statistics(x: np.ndarray) -> pd.DataFrame:
    """Per-turning-movement descriptive statistics."""
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    cv = std / (mean + 1e-9)
    return pd.DataFrame(
        {
            "mean": mean,
            "std": std,
            "min": x.min(axis=0),
            "max": x.max(axis=0),
            "cv": cv,
        }
    )


def od_statistics(y: np.ndarray) -> pd.DataFrame:
    """Per-OD-pair descriptive statistics."""
    mean = y.mean(axis=0)
    std = y.std(axis=0)
    cv = std / (mean + 1e-9)
    return pd.DataFrame({"mean": mean, "std": std, "cv": cv})


def _sampled_pairwise_mean_distance(
    data: np.ndarray,
    n_samples: int = 10_000,
    seed: int = 42,
) -> float:
    n = data.shape[0]
    if n < 2:
        return 0.0
    rng = np.random.default_rng(seed)
    idx_i = rng.integers(0, n, size=n_samples)
    idx_j = rng.integers(0, n, size=n_samples)
    same = idx_i == idx_j
    while same.any():
        idx_j[same] = rng.integers(0, n, size=int(same.sum()))
        same = idx_i == idx_j
    return float(np.mean(np.abs(data[idx_i] - data[idx_j])))


def correlation_summary(
    data: np.ndarray,
    max_features: int = 50,
    seed: int = 42,
) -> dict[str, float]:
    """Correlation summary on a random subset of varying features."""
    rng = np.random.default_rng(seed)
    std = data.std(axis=0)
    varying_idx = np.flatnonzero(std > 1e-12)
    if len(varying_idx) < 2:
        return {
            "mean_correlation": float("nan"),
            "min_correlation": float("nan"),
            "max_correlation": float("nan"),
            "features_used": len(varying_idx),
        }
    n_pick = min(max_features, len(varying_idx))
    chosen = rng.choice(varying_idx, size=n_pick, replace=False)
    subset = data[:, chosen]
    corr = np.corrcoef(subset, rowvar=False)
    off_diag = corr[np.triu_indices(n_pick, k=1)]
    return {
        "mean_correlation": float(np.mean(off_diag)),
        "min_correlation": float(np.min(off_diag)),
        "max_correlation": float(np.max(off_diag)),
        "features_used": n_pick,
    }


def diversity_comparison(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Compare OD vs turning-count diversity via sampled pairwise distance."""
    return {
        "mean_pairwise_od_distance": _sampled_pairwise_mean_distance(y),
        "mean_pairwise_turning_distance": _sampled_pairwise_mean_distance(x),
    }


def pca_explained_variance(
    data: np.ndarray,
    n_components: int = 10,
) -> np.ndarray:
    """Return explained variance ratio from PCA."""
    n_components = min(n_components, data.shape[0], data.shape[1])
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=42)
    pca.fit(data)
    return pca.explained_variance_ratio_


def create_visualizations(
    dataset: DatasetArrays,
    figures_dir: Path,
    turning_stats: pd.DataFrame,
    od_stats: pd.DataFrame,
) -> list[Path]:
    """Generate all required plots and return saved file paths."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    sns.set_theme(style="whitegrid")

    # Turning count histogram (all values pooled)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(dataset.x.ravel(), bins=50, kde=True, ax=ax)
    ax.set_title("Turning Count Distribution (all samples)")
    ax.set_xlabel("Turning count")
    path = figures_dir / "turning_counts_histogram.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved.append(path)

    # Distribution of per-turn means and variances
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(turning_stats["mean"], bins=40, ax=axes[0])
    axes[0].set_title("Distribution of Turning-Count Means")
    axes[0].set_xlabel("Mean turning count")
    turning_var = turning_stats["std"] ** 2
    sns.histplot(turning_var, bins=40, ax=axes[1])
    axes[1].set_title("Distribution of Turning-Count Variances")
    axes[1].set_xlabel("Variance")
    path = figures_dir / "turning_means_variances.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved.append(path)

    # OD mean and std heatmaps
    mean_od = od_stats["mean"].values.reshape(NUM_ZONES, NUM_ZONES)
    std_od = od_stats["std"].values.reshape(NUM_ZONES, NUM_ZONES)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(mean_od, ax=axes[0], cmap="YlOrRd")
    axes[0].set_title("Mean OD Matrix")
    sns.heatmap(std_od, ax=axes[1], cmap="Blues")
    axes[1].set_title("OD Std Dev")
    path = figures_dir / "od_mean_std_heatmaps.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved.append(path)

    # A_turn sparse heatmap (subsample columns if needed)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(dataset.a_turn[:, :min(100, dataset.a_turn.shape[1])], cmap="Greys", ax=ax)
    ax.set_title("A_turn (first 100 OD columns)")
    ax.set_xlabel("OD pair index")
    ax.set_ylabel("Turning movement")
    path = figures_dir / "a_turn_heatmap.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved.append(path)

    # PCA explained variance
    for label, data in [("turning_counts", dataset.x), ("od_targets", dataset.y)]:
        evr = pca_explained_variance(data)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(range(1, len(evr) + 1), evr)
        ax.set_xlabel("Principal component")
        ax.set_ylabel("Explained variance ratio")
        ax.set_title(f"PCA Explained Variance — {label}")
        path = figures_dir / f"pca_{label}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(path)

    logger.info("Saved %d figures to %s", len(saved), figures_dir)
    return saved
