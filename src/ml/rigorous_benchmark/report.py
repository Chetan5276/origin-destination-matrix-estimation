"""Benchmark tables, figures, and final_report.md."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.ml.rigorous_benchmark.selection import ranking_tables, results_to_dataframe

logger = logging.getLogger(__name__)

# Plan §29 column order for the single test leaderboard CSV
BENCHMARK_CSV_COLUMNS = [
    "model",
    "family",
    "parameters",
    "train_samples",
    "val_samples",
    "test_samples",
    "best_hyperparameters",
    "constraint_strategy",
    "train_time",
    "inference_time",
    "mae",
    "rmse",
    "r2",
    "pearson",
    "spearman",
    "relative_error",
    "forward_mae",
    "forward_rmse",
    "forward_r2",
    "production_mae",
    "production_rmse",
    "attraction_mae",
    "attraction_rmse",
    "total_demand_error",
    "negative_cells",
    "diagonal_violation",
    "sparsity_error",
    "composite_score",
]


def write_results(
    rows: list[dict[str, Any]],
    output_dir: Path,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = results_to_dataframe(rows)
    df.to_csv(output_dir / "benchmark_results_all_splits.csv", index=False)

    test_df = df[df["split"] == "test"].copy() if "split" in df.columns else df.copy()
    for col in BENCHMARK_CSV_COLUMNS:
        if col not in test_df.columns:
            test_df[col] = np.nan
    if "pearson" not in test_df.columns or test_df["pearson"].isna().all():
        if "correlation" in test_df.columns:
            test_df["pearson"] = test_df["correlation"]
    leaderboard = test_df[BENCHMARK_CSV_COLUMNS].sort_values("mae", na_position="last")
    leaderboard.to_csv(output_dir / "benchmark_results.csv", index=False)

    payload = []
    for r in rows:
        item = {k: v for k, v in r.items() if k != "metrics"}
        m = r.get("metrics") or {}
        item["metrics"] = {
            k: (
                float(v)
                if isinstance(v, (float, np.floating, int, np.integer)) and np.isfinite(v)
                else v
            )
            for k, v in m.items()
            if not isinstance(v, str)
        }
        payload.append(item)
    (output_dir / "benchmark_results.json").write_text(json.dumps(payload, indent=2, default=str))
    logger.info("Wrote benchmark_results.csv (%d test models)", len(leaderboard))
    return df


def make_figures(df: pd.DataFrame, figures_dir: Path) -> list[Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    if df.empty:
        return paths

    test_df = df[df["split"] == "test"] if "split" in df.columns else df
    if test_df.empty:
        test_df = df

    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(10, 5))
    order = test_df.sort_values("mae")["model"]
    sns.barplot(data=test_df, x="model", y="mae", order=order, ax=ax, color="steelblue")
    ax.tick_params(axis="x", rotation=45)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    ax.set_title("Synthetic test OD MAE")
    ax.set_ylabel("MAE")
    fig.tight_layout()
    p = figures_dir / "test_od_mae.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths.append(p)

    fwd = "forward_mae" if "forward_mae" in test_df.columns else "forward_rmse"
    fig, ax = plt.subplots(figsize=(10, 5))
    order = test_df.sort_values(fwd)["model"]
    sns.barplot(data=test_df, x="model", y=fwd, order=order, ax=ax, color="darkorange")
    ax.tick_params(axis="x", rotation=45)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    ax.set_title(f"Synthetic test {fwd} (turning space)")
    fig.tight_layout()
    p = figures_dir / "test_forward.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths.append(p)

    if "mae" in test_df.columns and fwd in test_df.columns:
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.scatterplot(data=test_df, x="mae", y=fwd, hue="family", s=80, ax=ax)
        for _, row in test_df.iterrows():
            ax.annotate(row["model"], (row["mae"], row[fwd]), fontsize=7, alpha=0.8)
        ax.set_title("OD MAE vs forward error (test)")
        fig.tight_layout()
        p = figures_dir / "od_vs_forward.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        paths.append(p)

    return paths


def write_final_report(
    output_dir: Path,
    *,
    df: pd.DataFrame,
    leakage: dict,
    operator_meta: dict,
    survey_results: dict | None,
    ablation_summary: dict | None,
    config_dict: dict,
) -> Path:
    ranks = ranking_tables(
        df[df["split"] == "test"] if "split" in df.columns and not df.empty else df
    )
    lines = [
        "# Rigorous OD Estimation Benchmark — Final Report",
        "",
        "## Protocol",
        "",
        "- Data: first-principles synthetics only for train/val/test/HPO.",
        "- Split: 70/15/15, seed 42; indices in `data/split_indices.npz`.",
        "- Survey OD held out until final inference (`survey_inference/`).",
        "- Forward map: `X = Y_flat @ A_turn.T` (forward RMSE/MAE in **turning space**).",
        "- Composite selection on **validation only**; one-shot synthetic test; then survey.",
        "- No silent clipping: constraint strategy reported per model.",
        "- Memory-safe defaults: capped HPO/final train sizes; sequential model training.",
        "",
        "## Leakage",
        "",
        "```json",
        json.dumps(leakage, indent=2)[:2000],
        "```",
        "",
        "## Operator (A_turn)",
        "",
        "```json",
        json.dumps(operator_meta, indent=2),
        "```",
        "",
        "## Synthetic test rankings",
        "",
    ]
    for name, rdf in ranks.items():
        lines.append(f"### {name}")
        lines.append("")
        if rdf.empty:
            lines.append("_empty_")
        else:
            cols = [
                c
                for c in [
                    "model",
                    "mae",
                    "forward_mae",
                    "forward_rmse",
                    "production_mae",
                    "attraction_mae",
                    "composite_score",
                    "constraint_strategy",
                ]
                if c in rdf.columns
            ]
            lines.append(rdf[cols].head(20).to_string(index=False))
        lines.append("")

    lines.extend(
        [
            "## Single-file leaderboard",
            "",
            "See `benchmark_results.csv` (test split; columns per plan §29).",
            "",
            "## Survey held-out",
            "",
        ]
    )
    if survey_results:
        rows = []
        for m, info in survey_results.items():
            met = info.get("metrics", {})
            rows.append(
                {
                    "model": m,
                    "mae": met.get("mae"),
                    "forward_rmse": met.get("forward_rmse"),
                    "constraint": info.get("constraint_strategy"),
                }
            )
        sdf = pd.DataFrame(rows).sort_values("mae")
        lines.append(sdf.to_string(index=False))
    else:
        lines.append("_not run_")
    lines.append("")

    lines.extend(["## Ablations", ""])
    if ablation_summary:
        lines.append("```json")
        lines.append(json.dumps(ablation_summary, indent=2, default=str)[:3000])
        lines.append("```")
    else:
        lines.append("_not run_")
    lines.append("")

    lines.extend(
        [
            "## Config",
            "",
            "```json",
            json.dumps(config_dict, indent=2, default=str)[:2500],
            "```",
            "",
            "## Figures",
            "",
            "- `figures/test_od_mae.png`",
            "- `figures/test_forward.png`",
            "- `figures/od_vs_forward.png`",
            "",
            "## Notes",
            "",
            "- Physics Ridge ≠ Tikhonov: supervised Ridge + optional forward closed-form refine.",
            "- Null-space MLP: `y = A+ x + N z(x)` with soft `||A N z||` penalty.",
            "- Do not rank solely by OD MAE; compare OD, forward, and marginal consistency.",
            "",
        ]
    )
    path = output_dir / "final_report.md"
    path.write_text("\n".join(lines))
    logger.info("Wrote %s", path)
    return path
