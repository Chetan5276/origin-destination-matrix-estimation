"""Stage 11: hyperparameter search over alpha and perturbation."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.dataset_evaluation import evaluate_dataset
from src.data.od_generator import GeneratorConfig, generate_synthetic_od_batch
from src.data.od_metrics import compute_batch_metrics

logger = logging.getLogger(__name__)

DEFAULT_ALPHAS = [100, 250, 500, 1000, 2000]
DEFAULT_PERTURBATIONS = [0.05, 0.10, 0.20, 0.30]


@dataclass(frozen=True)
class HyperparameterResult:
    """Evaluation of one (alpha, perturbation) configuration."""

    alpha: float
    perturbation: float
    mean_correlation: float
    mean_relative_error: float
    mean_pairwise_distance: float
    mean_cell_cv: float
    pca_pc1_variance: float
    pca_components_90pct: int
    demand_cv: float
    score: float

    def to_dict(self) -> dict:
        return asdict(self)


def _recommendation_score(
    mean_correlation: float,
    mean_pairwise_distance: float,
    mean_cell_cv: float,
    pca_components_90pct: int,
    target_corr: float = 0.85,
) -> float:
    """
    Higher is better: reward diversity and PCA spread, penalize excessive
    correlation with base (too little diversity for ML).
    """
    diversity_term = mean_pairwise_distance / 200.0 + mean_cell_cv
    pca_term = pca_components_90pct / 20.0
    corr_penalty = max(0.0, mean_correlation - target_corr) * 2.0
    return diversity_term + pca_term - corr_penalty


def run_hyperparameter_search(
    base_od: np.ndarray,
    samples_per_config: int = 200,
    alphas: list[float] | None = None,
    perturbations: list[float] | None = None,
    seed: int = 42,
    workers: int | None = None,
) -> list[HyperparameterResult]:
    """Evaluate grid of alpha and perturbation values."""
    alphas = alphas or DEFAULT_ALPHAS
    perturbations = perturbations or DEFAULT_PERTURBATIONS
    results: list[HyperparameterResult] = []

    for i, alpha in enumerate(alphas):
        for j, perturbation in enumerate(perturbations):
            config = GeneratorConfig(alpha=alpha, perturbation=perturbation)
            config_seed = seed + i * 1000 + j * 100
            batch_result = generate_synthetic_od_batch(
                base_od,
                samples_per_config,
                config,
                seed=config_seed,
                workers=workers,
                show_progress=False,
            )
            batch = batch_result.matrices
            matrix_metrics = compute_batch_metrics(
                base_od,
                batch,
                batch_result=batch_result,
                sample_size=min(500, samples_per_config),
            )
            evaluation = evaluate_dataset(base_od, batch, matrix_metrics)

            agg = evaluation.aggregate_metrics
            mean_corr = agg.get("mean_correlation", float("nan"))
            result = HyperparameterResult(
                alpha=alpha,
                perturbation=perturbation,
                mean_correlation=mean_corr,
                mean_relative_error=agg.get("mean_relative_error", float("nan")),
                mean_pairwise_distance=evaluation.mean_pairwise_distance,
                mean_cell_cv=evaluation.mean_cell_cv,
                pca_pc1_variance=evaluation.pca_explained_variance[0],
                pca_components_90pct=evaluation.pca_components_for_90pct,
                demand_cv=evaluation.demand_cv,
                score=_recommendation_score(
                    mean_corr,
                    evaluation.mean_pairwise_distance,
                    evaluation.mean_cell_cv,
                    evaluation.pca_components_for_90pct,
                ),
            )
            results.append(result)
            logger.info(
                "alpha=%.0f pert=%.2f corr=%.3f diversity=%.1f score=%.3f",
                alpha,
                perturbation,
                mean_corr,
                evaluation.mean_pairwise_distance,
                result.score,
            )

    return results


def write_hyperparameter_report(
    results: list[HyperparameterResult],
    output_dir: Path,
) -> HyperparameterResult:
    """Save CSV/JSON and return the recommended configuration."""
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([r.to_dict() for r in results])
    df = df.sort_values("score", ascending=False)
    df.to_csv(output_dir / "hyperparameter_search.csv", index=False)

    best = df.iloc[0]
    recommended = HyperparameterResult(
        alpha=float(best["alpha"]),
        perturbation=float(best["perturbation"]),
        mean_correlation=float(best["mean_correlation"]),
        mean_relative_error=float(best["mean_relative_error"]),
        mean_pairwise_distance=float(best["mean_pairwise_distance"]),
        mean_cell_cv=float(best["mean_cell_cv"]),
        pca_pc1_variance=float(best["pca_pc1_variance"]),
        pca_components_90pct=int(best["pca_components_90pct"]),
        demand_cv=float(best["demand_cv"]),
        score=float(best["score"]),
    )

    payload = {
        "recommended": recommended.to_dict(),
        "all_results": [r.to_dict() for r in results],
    }
    (output_dir / "hyperparameter_search.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    md = [
        "# Hyperparameter Search Report",
        "",
        "## Recommended Configuration",
        f"- **alpha**: {recommended.alpha:.0f}",
        f"- **perturbation**: {recommended.perturbation:.2f}",
        f"- mean correlation: {recommended.mean_correlation:.4f}",
        f"- mean pairwise distance: {recommended.mean_pairwise_distance:.2f}",
        f"- PCA components (90%): {recommended.pca_components_90pct}",
        f"- score: {recommended.score:.4f}",
        "",
        "## All Configurations (sorted by score)",
        "",
        "| alpha | perturbation | correlation | diversity | cell CV | PC1 var | score |",
        "|-------|--------------|-------------|-----------|---------|---------|-------|",
    ]
    for _, row in df.iterrows():
        md.append(
            f"| {row['alpha']:.0f} | {row['perturbation']:.2f} | "
            f"{row['mean_correlation']:.3f} | {row['mean_pairwise_distance']:.1f} | "
            f"{row['mean_cell_cv']:.3f} | {row['pca_pc1_variance']:.3f} | "
            f"{row['score']:.3f} |"
        )

    (output_dir / "hyperparameter_search.md").write_text(
        "\n".join(md) + "\n",
        encoding="utf-8",
    )
    return recommended
