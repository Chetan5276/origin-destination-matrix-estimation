"""Stage 10: dataset-level evaluation, visualization, and reporting."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from src.data.od_metrics import MatrixMetrics, aggregate_matrix_metrics

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetEvaluation:
    """Dataset-level statistics and diagnostics."""

    num_samples: int
    mean_pairwise_distance: float
    min_pairwise_distance: float
    max_pairwise_distance: float
    mean_cell_cv: float
    max_cell_cv: float
    demand_cv: float
    pca_explained_variance: list[float]
    pca_cumulative_variance: list[float]
    pca_components_for_90pct: int
    kmeans_inertia: dict[int, float]
    aggregate_metrics: dict[str, float]

    def to_dict(self) -> dict:
        return asdict(self)


def _sampled_pairwise_distances(
    flat_batch: np.ndarray,
    n_samples: int = 10_000,
    seed: int = 42,
) -> np.ndarray:
    n = flat_batch.shape[0]
    if n < 2:
        return np.array([0.0])
    rng = np.random.default_rng(seed)
    idx_i = rng.integers(0, n, size=n_samples)
    idx_j = rng.integers(0, n, size=n_samples)
    same = idx_i == idx_j
    while same.any():
        idx_j[same] = rng.integers(0, n, size=int(same.sum()))
        same = idx_i == idx_j
    return np.mean(np.abs(flat_batch[idx_i] - flat_batch[idx_j]), axis=1)


def evaluate_dataset(
    base_od: np.ndarray,
    synthetic_batch: np.ndarray,
    matrix_metrics: list[MatrixMetrics] | None = None,
    diversity_samples: int = 10_000,
    pca_components: int = 20,
    kmeans_k: list[int] | None = None,
    eval_sample_size: int = 50_000,
    fast_aggregate: dict[str, float] | None = None,
    seed: int = 42,
) -> DatasetEvaluation:
    """Run PCA, clustering, diversity, and variability analysis."""
    if kmeans_k is None:
        kmeans_k = [2, 5, 10]

    n_total = synthetic_batch.shape[0]
    flat = synthetic_batch.reshape(n_total, -1)

    # Subsample for PCA / KMeans on very large datasets
    eval_n = min(eval_sample_size, n_total)
    if eval_n < n_total:
        rng = np.random.default_rng(seed)
        eval_idx = rng.choice(n_total, size=eval_n, replace=False)
        flat_eval = flat[eval_idx]
    else:
        flat_eval = flat

    pairwise = _sampled_pairwise_distances(flat, n_samples=diversity_samples, seed=seed)

    cell_mean = flat.mean(axis=0)
    cell_std = flat.std(axis=0)
    cell_cv = cell_std / (cell_mean + 1e-9)

    totals = flat.sum(axis=1)
    demand_cv = float(totals.std() / (totals.mean() + 1e-12))

    n_comp = min(pca_components, flat_eval.shape[0], flat_eval.shape[1])
    pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=42)
    pca.fit(flat_eval.astype(np.float64))
    evr = pca.explained_variance_ratio_.tolist()
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    components_90 = int(np.searchsorted(cumulative, 0.90) + 1)

    kmeans_inertia: dict[int, float] = {}
    for k in kmeans_k:
        km = KMeans(n_clusters=k, n_init=10, random_state=42, algorithm="elkan")
        km.fit(flat_eval.astype(np.float64))
        kmeans_inertia[k] = float(km.inertia_)

    if fast_aggregate is not None:
        agg = fast_aggregate
    elif matrix_metrics:
        agg = aggregate_matrix_metrics(matrix_metrics)
    else:
        agg = {}

    return DatasetEvaluation(
        num_samples=n_total,
        mean_pairwise_distance=float(pairwise.mean()),
        min_pairwise_distance=float(pairwise.min()),
        max_pairwise_distance=float(pairwise.max()),
        mean_cell_cv=float(cell_cv.mean()),
        max_cell_cv=float(cell_cv.max()),
        demand_cv=demand_cv,
        pca_explained_variance=evr,
        pca_cumulative_variance=cumulative.tolist(),
        pca_components_for_90pct=components_90,
        kmeans_inertia=kmeans_inertia,
        aggregate_metrics=agg,
    )


def create_evaluation_figures(
    base_od: np.ndarray,
    synthetic_batch: np.ndarray,
    matrix_metrics: list[MatrixMetrics],
    evaluation: DatasetEvaluation,
    output_dir: Path,
) -> list[Path]:
    """Generate required visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    sns.set_theme(style="whitegrid")

    flat = synthetic_batch.reshape(synthetic_batch.shape[0], -1)
    n = base_od.shape[0]
    mean_od = flat.mean(axis=0).reshape(n, n)
    std_od = flat.std(axis=0).reshape(n, n)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(mean_od, ax=axes[0], cmap="YlOrRd")
    axes[0].set_title("Mean Synthetic OD")
    sns.heatmap(std_od, ax=axes[1], cmap="Blues")
    axes[1].set_title("Std Dev Synthetic OD")
    path = output_dir / "od_mean_std_heatmap.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved.append(path)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(
        range(1, len(evaluation.pca_explained_variance) + 1),
        evaluation.pca_explained_variance,
    )
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance ratio")
    ax.set_title("PCA Explained Variance")
    path = output_dir / "pca_variance.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved.append(path)

    pairwise = _sampled_pairwise_distances(flat, n_samples=min(5000, flat.shape[0] * 10))
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(pairwise, bins=40, kde=True, ax=ax)
    ax.set_title("Pairwise OD Distance Distribution")
    ax.set_xlabel("Mean absolute difference")
    path = output_dir / "diversity_histogram.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved.append(path)

    correlations = [m.correlation for m in matrix_metrics]
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(correlations, bins=40, kde=True, ax=ax)
    ax.set_title("Correlation with Base OD")
    ax.set_xlabel("Pearson correlation")
    path = output_dir / "correlation_histogram.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved.append(path)

    logger.info("Saved %d figures to %s", len(saved), output_dir)
    return saved


def write_dataset_summary(
    evaluation: DatasetEvaluation,
    config: dict,
    output_path: Path,
) -> None:
    """Write markdown summary report."""
    agg = evaluation.aggregate_metrics
    lines = [
        "# Synthetic OD Dataset Summary",
        "",
        "## Configuration",
        "",
    ]
    for key, value in config.items():
        lines.append(f"- **{key}**: {value}")

    lines.extend(
        [
            "",
            "## Dataset Size",
            f"- Samples: **{evaluation.num_samples:,}**",
            "",
            "## Similarity (mean over samples)",
            f"- Correlation: **{agg.get('mean_correlation', float('nan')):.4f}**",
            f"- MAE: **{agg.get('mean_mae', float('nan')):.2f}**",
            f"- Relative error: **{agg.get('mean_relative_error', float('nan')):.4f}**",
            "",
            "## Diversity",
            f"- Mean pairwise distance: **{evaluation.mean_pairwise_distance:.2f}**",
            f"- Min / max pairwise distance: **{evaluation.min_pairwise_distance:.2f}** / "
            f"**{evaluation.max_pairwise_distance:.2f}**",
            f"- Mean cell CV: **{evaluation.mean_cell_cv:.4f}**",
            "",
            "## Demand Consistency",
            f"- Demand CV: **{evaluation.demand_cv:.2e}**",
            "",
            "## Structural Validity",
            f"- Max new connections: **{agg.get('max_new_connections', 0):.0f}**",
            f"- Max diagonal violation: **{agg.get('max_diagonal_violation', 0):.2e}**",
            "",
            "## PCA",
            f"- Components for 90% variance: **{evaluation.pca_components_for_90pct}**",
            f"- PC1 explained variance: **{evaluation.pca_explained_variance[0]:.4f}**",
            "",
            "## KMeans Inertia",
        ]
    )
    for k, inertia in evaluation.kmeans_inertia.items():
        lines.append(f"- k={k}: {inertia:.2f}")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_dataset_metrics(
    evaluation: DatasetEvaluation,
    matrix_metrics: list[MatrixMetrics],
    output_path: Path,
) -> None:
    """Save JSON metrics file."""
    payload = {
        "dataset_evaluation": evaluation.to_dict(),
        "per_matrix_metrics_sample": [m.to_dict() for m in matrix_metrics[:10]],
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
