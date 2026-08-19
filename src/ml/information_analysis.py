"""Stage 1--2: dataset and information content analysis."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression

from src import NUM_OD_PAIRS
from src.data.assignment_rank import extended_rank_analysis

logger = logging.getLogger(__name__)


def feature_statistics(x: np.ndarray) -> pd.DataFrame:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    return pd.DataFrame(
        {
            "turn_id": np.arange(x.shape[1]),
            "mean": mean,
            "std": std,
            "min": x.min(axis=0),
            "max": x.max(axis=0),
            "cv": std / (mean + 1e-9),
        }
    )


def target_statistics(y: np.ndarray) -> pd.DataFrame:
    mean = y.mean(axis=0)
    std = y.std(axis=0)
    sparsity = (y == 0).mean(axis=0)
    return pd.DataFrame(
        {
            "od_index": np.arange(y.shape[1]),
            "mean": mean,
            "std": std,
            "sparsity": sparsity,
        }
    )


def correlation_summary_matrix(data: np.ndarray, max_features: int = 50, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_feat = min(max_features, data.shape[1])
    idx = rng.choice(data.shape[1], size=n_feat, replace=False)
    sub = data[:, idx]
    return np.corrcoef(sub, rowvar=False)


def pca_analysis(data: np.ndarray, max_components: int = 50) -> dict:
    n_comp = min(max_components, data.shape[0], data.shape[1])
    pca = PCA(n_components=n_comp, random_state=42)
    pca.fit(data)
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    n90 = int(np.searchsorted(cum, 0.90) + 1)
    return {
        "n_components_90pct": n90,
        "explained_variance_ratio": evr.tolist(),
        "cumulative_variance": cum.tolist(),
    }


def mutual_information_matrix(
    x: np.ndarray,
    y: np.ndarray,
    n_turn_sample: int = 50,
    n_od_sample: int = 50,
    seed: int = 42,
) -> pd.DataFrame:
    """Sample-based MI(turn_i, OD_j) for tractability."""
    rng = np.random.default_rng(seed)
    turn_idx = rng.choice(x.shape[1], size=min(n_turn_sample, x.shape[1]), replace=False)
    od_idx = rng.choice(y.shape[1], size=min(n_od_sample, y.shape[1]), replace=False)

    rows = []
    for ti in turn_idx:
        for oj in od_idx:
            mi = mutual_info_regression(
                x[:, [ti]], y[:, oj], random_state=seed, n_neighbors=5
            )[0]
            rows.append({"turn_id": int(ti), "od_index": int(oj), "mutual_info": float(mi)})
    return pd.DataFrame(rows)


def rank_turns_by_information(
    x: np.ndarray,
    y: np.ndarray,
    mi_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if mi_df is None:
        mi_df = mutual_information_matrix(x, y)

    mi_agg = mi_df.groupby("turn_id")["mutual_info"].mean()
    var = x.var(axis=0)
    corr_proxy = np.array(
        [abs(np.corrcoef(x[:, i], y.mean(axis=1))[0, 1]) if x[:, i].std() > 0 else 0.0
         for i in range(x.shape[1])]
    )

    df = pd.DataFrame(
        {
            "turn_id": np.arange(x.shape[1]),
            "mi_mean": mi_agg.reindex(range(x.shape[1]), fill_value=0.0).values,
            "variance": var,
            "corr_proxy": corr_proxy,
        }
    )
    df["importance_score"] = (
        df["mi_mean"] / (df["mi_mean"].max() + 1e-9)
        + df["variance"] / (df["variance"].max() + 1e-9)
        + df["corr_proxy"]
    ) / 3.0
    return df.sort_values("importance_score", ascending=False)


def identifiability_analysis(a_turn_path: Path, output_dir: Path) -> dict:
    a_turn = np.load(a_turn_path)
    analysis = extended_rank_analysis(a_turn)
    output_dir.mkdir(parents=True, exist_ok=True)

    sv = analysis.singular_values
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(range(1, len(sv) + 1), sv, "o-")
    ax.set_xlabel("Singular value index")
    ax.set_ylabel("Singular value")
    ax.set_title("A_turn singular value spectrum")
    fig.tight_layout()
    fig.savefig(output_dir / "singular_value_spectrum.png", dpi=150)
    plt.close(fig)

    return analysis.to_dict()


def run_dataset_analysis(
    x: np.ndarray,
    y: np.ndarray,
    a_turn_path: Path | None,
    output_dir: Path,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    x_stats = feature_statistics(x)
    y_stats = target_statistics(y)
    x_stats.to_csv(output_dir / "turning_statistics.csv", index=False)
    y_stats.to_csv(output_dir / "od_target_statistics.csv", index=False)

    pca_x = pca_analysis(x)
    pca_y = pca_analysis(y)

    for name, pca_res, data in [
        ("pca_X", pca_x, x),
        ("pca_Y", pca_y, y),
    ]:
        n_show = min(20, len(pca_res["explained_variance_ratio"]))
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(range(1, n_show + 1), pca_res["explained_variance_ratio"][:n_show])
        ax.set_xlabel("Component")
        ax.set_ylabel("Explained variance ratio")
        ax.set_title(name)
        fig.tight_layout()
        fig.savefig(fig_dir / f"{name.lower()}.png", dpi=150)
        plt.close(fig)

    mi_df = mutual_information_matrix(x, y)
    importance = rank_turns_by_information(x, y, mi_df)
    importance.to_csv(output_dir / "important_turns.csv", index=False)
    importance.to_csv(output_dir / "feature_importance.csv", index=False)

    corr_x = correlation_summary_matrix(x)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_x, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Turning count correlation (sampled features)")
    fig.tight_layout()
    fig.savefig(fig_dir / "correlation_heatmap.png", dpi=150)
    plt.close(fig)

    ident = None
    if a_turn_path and Path(a_turn_path).exists():
        ident = identifiability_analysis(Path(a_turn_path), fig_dir)

    summary = {
        "n_samples": int(x.shape[0]),
        "n_turns": int(x.shape[1]),
        "n_od_pairs": int(y.shape[1]),
        "pca_x": pca_x,
        "pca_y": pca_y,
        "top_turns": importance.head(20).to_dict(orient="records"),
        "identifiability": ident,
    }
    (output_dir / "dataset_analysis.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    logger.info("Dataset analysis saved to %s", output_dir)
    return summary
