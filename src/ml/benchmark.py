"""Stage 10: benchmark orchestration and leaderboard."""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.ml.config import ML_OUTPUT_DIR, NeuralTrainConfig, TrainConfig
from src.ml.dataset import ODDataset, inverse_transform_y, load_turning_and_od, split_dataset
from src.ml.explainability import generate_shap_summary
from src.ml.information_analysis import run_dataset_analysis
from src.ml.metrics import evaluate_predictions_with_forward
from src.ml.models_classical import classical_models, fit_model, predict_model, tree_models
from src.ml.neural_trainer import (
    evaluate_model,
    finetune_latent_model,
    predict_direct_model,
    predict_latent_model,
    pretrain_od_autoencoder,
    train_direct_model,
    train_latent_predictor,
)
from src.ml.od_constraints import apply_od_constraints_numpy

logger = logging.getLogger(__name__)

METRIC_COLUMNS = [
    "model",
    "mae",
    "rmse",
    "r2",
    "correlation",
    "production_mae",
    "attraction_mae",
    "forward_rmse",
]


def _plot_leaderboard(df: pd.DataFrame, metric: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    valid = df.dropna(subset=[metric])
    order = valid.sort_values(metric, ascending=(metric != "r2"))["model"]
    sns.barplot(data=valid, x="model", y=metric, order=order, ax=ax)
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_horizontalalignment("right")
    ax.set_title(f"{metric.upper()} by model")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_pred_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, path: Path) -> None:
    n = min(5000, y_true.size)
    idx = np.random.default_rng(0).choice(y_true.size, size=n, replace=False)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true.ravel()[idx], y_pred.ravel()[idx], alpha=0.2, s=5)
    lim = max(y_true.max(), y_pred.max())
    ax.plot([0, lim], [0, lim], "r--")
    ax.set_xlabel("Actual OD")
    ax.set_ylabel("Predicted OD")
    ax.set_title("Predicted vs actual (sampled cells)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_best_model(path: Path, name: str, model) -> None:
    with path.open("wb") as fh:
        pickle.dump({"name": name, "model": model}, fh)


def _ridge_predict(dataset: ODDataset, model) -> np.ndarray:
    pred_scaled = np.asarray(model.predict(dataset.x_test), dtype=np.float32)
    pred = inverse_transform_y(dataset, pred_scaled)
    return apply_od_constraints_numpy(pred, support_mask=dataset.support_mask)


def run_neural_benchmark(
    turning_path: Path,
    od_path: Path,
    a_turn_path: Path,
    output_dir: Path,
    train_config: TrainConfig,
    neural_config: NeuralTrainConfig | None = None,
    skip_analysis: bool = False,
) -> pd.DataFrame:
    """
    Benchmark Ridge + redesigned neural OD estimators on clean turning counts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    models_dir = output_dir / "models"
    models_dir.mkdir(exist_ok=True)

    ncfg = neural_config or NeuralTrainConfig()
    use_noisy = not train_config.use_clean_turning
    x, y = load_turning_and_od(turning_path, od_path, max_samples=train_config.max_samples, use_noisy=use_noisy)
    a_turn = np.load(a_turn_path).astype(np.float32)

    if not skip_analysis:
        run_dataset_analysis(x, y, a_turn_path, output_dir / "analysis")

    dataset = split_dataset(
        x,
        y,
        seed=train_config.seed,
        train_frac=train_config.train_frac,
        val_frac=train_config.val_frac,
        standardize_x=train_config.standardize_x,
        standardize_y=train_config.standardize_y,
    )

    results: list[dict] = []
    ae_reconstruction: dict[int, float] = {}
    best_mae = float("inf")
    best_name = ""
    best_pred = None

    # --- Ridge ---
    logger.info("Training ridge ...")
    ridge_spec = next(s for s in classical_models(train_config.n_jobs) if s.name == "ridge")
    ridge = fit_model(ridge_spec, dataset.x_train, dataset.y_train, cv=train_config.cv_folds, n_jobs=train_config.n_jobs)
    ridge_pred = _ridge_predict(dataset, ridge)
    row = evaluate_model("ridge", dataset.y_test_raw, ridge_pred, dataset.x_test_raw, a_turn)
    results.append(row)
    if row["mae"] < best_mae:
        best_mae, best_name, best_pred = row["mae"], "ridge", ridge_pred

    # --- Direct MLP ---
    logger.info("Training mlp ...")
    mlp = train_direct_model(dataset, a_turn, ncfg, residual_blocks=False, residual_learning=False)
    mlp_pred = predict_direct_model(mlp, dataset, dataset.x_test, ncfg, residual_learning=False)
    row = evaluate_model("mlp", dataset.y_test_raw, mlp_pred, dataset.x_test_raw, a_turn)
    results.append(row)
    if row["mae"] < best_mae:
        best_mae, best_name, best_pred = row["mae"], "mlp", mlp_pred

    # --- Residual MLP ---
    logger.info("Training residual_mlp ...")
    res_mlp = train_direct_model(dataset, a_turn, ncfg, residual_blocks=True, residual_learning=False)
    res_pred = predict_direct_model(res_mlp, dataset, dataset.x_test, ncfg, residual_learning=False)
    row = evaluate_model("residual_mlp", dataset.y_test_raw, res_pred, dataset.x_test_raw, a_turn)
    results.append(row)
    if row["mae"] < best_mae:
        best_mae, best_name, best_pred = row["mae"], "residual_mlp", res_pred

    # --- Residual learning (MLP) ---
    logger.info("Training mlp_residual ...")
    mlp_res = train_direct_model(dataset, a_turn, ncfg, residual_blocks=False, residual_learning=True)
    mlp_res_pred = predict_direct_model(mlp_res, dataset, dataset.x_test, ncfg, residual_learning=True)
    row = evaluate_model("mlp_residual", dataset.y_test_raw, mlp_res_pred, dataset.x_test_raw, a_turn)
    results.append(row)
    if row["mae"] < best_mae:
        best_mae, best_name, best_pred = row["mae"], "mlp_residual", mlp_res_pred

    # --- Two-stage autoencoder + latent predictor (per latent dim) ---
    latent_test_mae: dict[int, float] = {}
    for latent_dim in ncfg.latent_dims:
        logger.info("Pretraining OD autoencoder (latent=%d) ...", latent_dim)
        pre = pretrain_od_autoencoder(dataset, latent_dim, ncfg, models_dir)
        ae_reconstruction[latent_dim] = pre.reconstruction_rmse

        logger.info("Training latent predictor (latent=%d) ...", latent_dim)
        latent_model = train_latent_predictor(
            dataset, a_turn, pre.encoder, pre.decoder, ncfg, freeze_decoder=True
        )
        lat_pred = predict_latent_model(latent_model, dataset, dataset.x_test, ncfg)
        name = f"ae_latent_{latent_dim}"
        row = evaluate_model(name, dataset.y_test_raw, lat_pred, dataset.x_test_raw, a_turn)
        results.append(row)
        latent_test_mae[latent_dim] = row["mae"]
        if row["mae"] < best_mae:
            best_mae, best_name, best_pred = row["mae"], name, lat_pred

        logger.info("Fine-tuning ae_latent_%d ...", latent_dim)
        finetuned = finetune_latent_model(latent_model, dataset, a_turn, ncfg)
        ft_pred = predict_latent_model(finetuned, dataset, dataset.x_test, ncfg)
        ft_name = f"ae_latent_{latent_dim}_finetuned"
        row = evaluate_model(ft_name, dataset.y_test_raw, ft_pred, dataset.x_test_raw, a_turn)
        results.append(row)
        if row["mae"] < best_mae:
            best_mae, best_name, best_pred = row["mae"], ft_name, ft_pred

    best_latent = min(latent_test_mae, key=latent_test_mae.get) if latent_test_mae else None

    df = pd.DataFrame(results)
    if "mae" in df.columns:
        df = df.sort_values("mae", na_position="last")
    df.to_csv(output_dir / "benchmark_results.csv", index=False)

    if len(df) and "mae" in df.columns:
        valid = df.dropna(subset=["mae"])
        _plot_leaderboard(valid, "mae", fig_dir / "mae_by_model.png")
        _plot_leaderboard(valid, "rmse", fig_dir / "rmse_by_model.png")
        if "forward_rmse" in valid.columns:
            _plot_leaderboard(valid, "forward_rmse", fig_dir / "forward_rmse_by_model.png")

    if best_pred is not None:
        np.save(output_dir / "predictions.npy", best_pred.astype(np.float32))
        _plot_pred_vs_actual(dataset.y_test_raw, best_pred, fig_dir / "predicted_vs_actual.png")

    summary = {
        "config": asdict(train_config),
        "neural_config": asdict(ncfg),
        "best_model": best_name,
        "best_mae": best_mae,
        "ae_reconstruction_rmse": ae_reconstruction,
        "best_latent_dim_od_mae": best_latent,
        "leaderboard": df.to_dict(orient="records"),
    }
    (output_dir / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    _write_evaluation_report(output_dir, df, best_name, train_config, ae_reconstruction, best_latent)
    logger.info("Benchmark complete. Best model: %s (MAE=%.4f)", best_name, best_mae)
    return df


def run_benchmark(
    turning_path: Path,
    od_path: Path,
    a_turn_path: Path | None,
    output_dir: Path,
    config: TrainConfig,
    models: list[str] | None = None,
    include_neural: bool = True,
    skip_analysis: bool = False,
    neural_config: NeuralTrainConfig | None = None,
) -> pd.DataFrame:
    """Run Phase 3 benchmark (neural redesign + Ridge by default)."""
    if a_turn_path is None:
        raise ValueError("a_turn_path is required for the redesigned benchmark")
    if include_neural or models is None or "auto" in (models or []):
        return run_neural_benchmark(
            turning_path,
            od_path,
            a_turn_path,
            output_dir,
            config,
            neural_config=neural_config,
            skip_analysis=skip_analysis,
        )

    # Classical-only fallback
    output_dir.mkdir(parents=True, exist_ok=True)
    x, y = load_turning_and_od(
        turning_path, od_path, max_samples=config.max_samples, use_noisy=not config.use_clean_turning
    )
    dataset = split_dataset(
        x, y, seed=config.seed, train_frac=config.train_frac, val_frac=config.val_frac,
        standardize_x=config.standardize_x, standardize_y=config.standardize_y,
    )
    results = []
    for spec in classical_models(config.n_jobs):
        if models and spec.name not in models:
            continue
        model = fit_model(spec, dataset.x_train, dataset.y_train, cv=config.cv_folds, n_jobs=config.n_jobs)
        pred = _ridge_predict(dataset, model) if spec.name == "ridge" else predict_model(model, dataset.x_test)
        if spec.name != "ridge":
            pred = apply_od_constraints_numpy(pred, support_mask=dataset.support_mask)
        results.append(evaluate_model(spec.name, dataset.y_test_raw, pred, dataset.x_test_raw, np.load(a_turn_path)))
    df = pd.DataFrame(results).sort_values("mae")
    df.to_csv(output_dir / "benchmark_results.csv", index=False)
    return df


def _write_evaluation_report(
    output_dir: Path,
    df: pd.DataFrame,
    best: str,
    config: TrainConfig,
    ae_reconstruction: dict[int, float] | None = None,
    best_latent: int | None = None,
) -> None:
    lines = [
        "# OD Estimation Benchmark Report (Phase 3)",
        "",
        "## Configuration",
        f"- Seed: {config.seed}",
        f"- Max samples: {config.max_samples}",
        f"- Clean turning counts: {config.use_clean_turning}",
        f"- Standardize Y: {config.standardize_y}",
        "",
    ]
    if ae_reconstruction:
        lines.append("## OD Autoencoder reconstruction RMSE (validation)")
        for dim, rmse in sorted(ae_reconstruction.items()):
            lines.append(f"- Latent {dim}: {rmse:.2f}")
        if best_latent is not None:
            lines.append(f"- **Best latent dim (test MAE):** {best_latent}")
        lines.append("")

    lines.extend(
        [
            "## Leaderboard",
            "",
            "| Model | MAE | RMSE | R² | Correlation | Production MAE | Attraction MAE | Forward RMSE |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for _, row in df.iterrows():
        if pd.isna(row.get("mae")):
            continue
        lines.append(
            f"| {row['model']} | {row.get('mae', float('nan')):.4f} | "
            f"{row.get('rmse', float('nan')):.4f} | {row.get('r2', float('nan')):.4f} | "
            f"{row.get('correlation', float('nan')):.4f} | "
            f"{row.get('production_mae', float('nan')):.4f} | "
            f"{row.get('attraction_mae', float('nan')):.4f} | "
            f"{row.get('forward_rmse', float('nan')):.4f} |"
        )
    lines.extend(["", f"**Best model:** {best}", ""])
    (output_dir / "evaluation_report.md").write_text("\n".join(lines), encoding="utf-8")
