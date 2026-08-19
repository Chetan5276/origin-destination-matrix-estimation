"""Stage 11--12: turning dataset statistics and sensitivity analysis."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TurningDatasetStats:
    """Summary statistics for turning-count dataset."""

    num_samples: int
    num_turns: int
    mean_pairwise_od_distance: float
    mean_pairwise_turning_distance: float
    mean_turn_correlation: float
    max_turn_correlation: float
    per_turn_mean: list[float]
    per_turn_std: list[float]
    per_turn_cv: list[float]


def _sampled_pairwise(flat: np.ndarray, n_samples: int, seed: int) -> float:
    n = flat.shape[0]
    if n < 2:
        return 0.0
    rng = np.random.default_rng(seed)
    idx_i = rng.integers(0, n, size=n_samples)
    idx_j = rng.integers(0, n, size=n_samples)
    same = idx_i == idx_j
    while same.any():
        idx_j[same] = rng.integers(0, n, size=int(same.sum()))
        same = idx_i == idx_j
    return float(np.mean(np.abs(flat[idx_i] - flat[idx_j])))


def compute_turning_stats(
    od_batch: np.ndarray,
    turning_batch: np.ndarray,
    diversity_samples: int = 10_000,
    seed: int = 42,
) -> TurningDatasetStats:
    """Compute turning-count and OD diversity statistics."""
    od_flat = od_batch.reshape(od_batch.shape[0], -1)
    n_turns = turning_batch.shape[1]

    turn_mean = turning_batch.mean(axis=0)
    turn_std = turning_batch.std(axis=0)
    turn_cv = turn_std / (turn_mean + 1e-9)

    # Correlation among turning movements (sample features if M large)
    n_feat = min(50, n_turns)
    rng = np.random.default_rng(seed)
    feat_idx = rng.choice(n_turns, size=n_feat, replace=False)
    subset = turning_batch[:, feat_idx]
    std = subset.std(axis=0)
    varying = std > 1e-12
    if varying.sum() >= 2:
        corr = np.corrcoef(subset[:, varying], rowvar=False)
        off = corr[np.triu_indices(corr.shape[0], k=1)]
        mean_corr = float(np.mean(off))
        max_corr = float(np.max(np.abs(off)))
    else:
        mean_corr = float("nan")
        max_corr = float("nan")

    return TurningDatasetStats(
        num_samples=turning_batch.shape[0],
        num_turns=n_turns,
        mean_pairwise_od_distance=_sampled_pairwise(od_flat, diversity_samples, seed),
        mean_pairwise_turning_distance=_sampled_pairwise(
            turning_batch, diversity_samples, seed + 1
        ),
        mean_turn_correlation=mean_corr,
        max_turn_correlation=max_corr,
        per_turn_mean=turn_mean.tolist(),
        per_turn_std=turn_std.tolist(),
        per_turn_cv=turn_cv.tolist(),
    )


def run_sensitivity_analysis(
    network,
    od_index,
    turning_index,
    k_values: list[int],
    theta_values: list[float],
    weight_metric: str = "length",
) -> pd.DataFrame:
    """Grid search over K and theta; return comparison table."""
    from src.data.assignment_rank import extended_rank_analysis
    from src.data.fractional_assignment import build_fractional_assignment_matrix
    from src.data.k_shortest_paths import enumerate_k_shortest_paths
    from src.data.route_choice import apply_logit_choice

    rows: list[dict] = []
    for k in k_values:
        catalog = enumerate_k_shortest_paths(
            network, od_index, k_paths=k, weight_metric=weight_metric
        )
        for theta in theta_values:
            choices = apply_logit_choice(catalog, theta=theta)
            a_turn = build_fractional_assignment_matrix(
                turning_index, od_index, choices
            )
            rank = extended_rank_analysis(a_turn)
            rows.append(
                {
                    "k_paths": k,
                    "theta": theta,
                    "rank": rank.result.rank,
                    "nullity": rank.result.nullity,
                    "condition_number": rank.result.condition_number,
                    "effective_rank_99": rank.effective_rank_99,
                    "max_entry": float(a_turn.max()),
                }
            )
    return pd.DataFrame(rows)


def create_turning_figures(
    a_turn: np.ndarray,
    turning_clean: np.ndarray | None,
    turning_noisy: np.ndarray | None,
    rank_comparison,
    route_probs: list[float],
    output_dir: Path,
    *,
    turn_mean: np.ndarray | None = None,
    turn_std: np.ndarray | None = None,
    turn_cv: np.ndarray | None = None,
) -> list[Path]:
    """Generate Phase 2 visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    sns.set_theme(style="whitegrid")

    if turn_mean is None:
        if turning_clean is None:
            raise ValueError("Provide turn_mean or turning_clean for histograms")
        turn_mean = turning_clean.mean(axis=0)
        turn_std = turning_clean.std(axis=0)
        turn_cv = turn_std / (turn_mean + 1e-9)
    elif turn_std is None or turn_cv is None:
        raise ValueError("turn_std and turn_cv required when turn_mean is provided")

    for data, name in [
        (turn_mean, "turning_mean_histogram.png"),
        (turn_std, "turning_std_histogram.png"),
        (turn_cv, "turning_cv_histogram.png"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(data, bins=40, kde=True, ax=ax)
        ax.set_title(name.replace("_", " ").replace(".png", ""))
        fig.tight_layout()
        path = output_dir / name
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(path)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        a_turn[:, : min(100, a_turn.shape[1])],
        cmap="viridis",
        ax=ax,
        vmin=0,
        vmax=1,
    )
    ax.set_title("A_turn (fractional, first 100 OD columns)")
    path = output_dir / "A_turn_heatmap.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved.append(path)

    sv = rank_comparison.probabilistic.singular_values
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(range(1, len(sv) + 1), sv, "o-", label="probabilistic")
    sv_bin = rank_comparison.binary.singular_values
    ax.semilogy(range(1, len(sv_bin) + 1), sv_bin, "s--", label="binary")
    ax.set_xlabel("Index")
    ax.set_ylabel("Singular value")
    ax.set_title("Singular values")
    ax.legend()
    path = output_dir / "singular_values.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved.append(path)

    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["Binary", "Probabilistic"]
    ranks = [rank_comparison.binary.result.rank, rank_comparison.probabilistic.result.rank]
    ax.bar(labels, ranks, color=["steelblue", "darkorange"])
    ax.set_ylabel("Matrix rank")
    ax.set_title("Rank comparison")
    path = output_dir / "rank_analysis.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved.append(path)

    if route_probs:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(route_probs, bins=40, kde=True, ax=ax)
        ax.set_title("Route probability distribution")
        ax.set_xlabel("P(route)")
        path = output_dir / "route_probability_distribution.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(path)

    logger.info("Saved %d figures to %s", len(saved), output_dir)
    return saved


def write_turning_summary(
    stats: TurningDatasetStats,
    rank_comparison,
    config: dict,
    output_path: Path,
) -> None:
    """Write markdown summary report."""
    lines = [
        "# Turning-Count Dataset Summary (Phase 2)",
        "",
        "## Configuration",
    ]
    for k, v in config.items():
        lines.append(f"- **{k}**: {v}")

    lines.extend(
        [
            "",
            "## Dataset",
            f"- Samples: **{stats.num_samples:,}**",
            f"- Turning movements: **{stats.num_turns}**",
            "",
            "## Diversity",
            f"- OD pairwise distance: **{stats.mean_pairwise_od_distance:.2f}**",
            f"- Turning pairwise distance: **{stats.mean_pairwise_turning_distance:.2f}**",
            f"- Mean turn correlation: **{stats.mean_turn_correlation:.4f}**",
            "",
            "## Rank Analysis",
            f"- Binary rank: **{rank_comparison.binary.result.rank}** "
            f"(nullity {rank_comparison.binary.result.nullity})",
            f"- Probabilistic rank: **{rank_comparison.probabilistic.result.rank}** "
            f"(nullity {rank_comparison.probabilistic.result.nullity})",
            f"- Rank improvement: **+{rank_comparison.rank_improvement}**",
            f"- Condition number (prob.): **{rank_comparison.probabilistic.result.condition_number:.4g}**",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
