#!/usr/bin/env python3
"""Generate documentation figures from outputs/benchmark/ artifacts.

Writes PNGs into docs/figures/od_benchmark/. All metrics come from saved
CSV/JSON/NPY artifacts — this script does not retrain models.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCH = REPO_ROOT / "outputs" / "benchmark"
DEFAULT_OUT = REPO_ROOT / "docs" / "figures" / "od_benchmark"

HEATMAP_MODELS = [
    "moore_penrose",
    "tikhonov",
    "physics_residual_mlp",
    "ae_64_finetune",
    "nullspace_mlp",
]

EXISTING_FIGURES = [
    "test_od_mae.png",
    "test_forward.png",
    "od_vs_forward.png",
]


def _barplot(df: pd.DataFrame, x: str, y: str, title: str, path: Path, color: str) -> None:
    order = df.sort_values(y)[x]
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=df, x=x, y=y, order=order, ax=ax, color=color)
    ax.tick_params(axis="x", rotation=45)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _scatter_od_vs_forward(
    df: pd.DataFrame,
    mae_col: str,
    fwd_col: str,
    title: str,
    path: Path,
    hue_col: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6))
    if hue_col and hue_col in df.columns:
        sns.scatterplot(data=df, x=mae_col, y=fwd_col, hue=hue_col, s=90, ax=ax)
    else:
        sns.scatterplot(data=df, x=mae_col, y=fwd_col, s=90, ax=ax, color="steelblue")
    for _, row in df.iterrows():
        ax.annotate(str(row["model"]), (row[mae_col], row[fwd_col]), fontsize=7, alpha=0.85)
    ax.set_xlabel("OD MAE")
    ax.set_ylabel("Forward RMSE (turning space)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _load_survey_df(survey_json: Path) -> pd.DataFrame:
    raw = json.loads(survey_json.read_text())
    rows = []
    for model, payload in raw.items():
        m = dict(payload.get("metrics", {}))
        m["model"] = model
        m["constraint_strategy"] = payload.get("constraint_strategy")
        rows.append(m)
    return pd.DataFrame(rows)


def _heatmap_triplet(
    true_mat: np.ndarray,
    pred_mat: np.ndarray,
    title_prefix: str,
    path: Path,
) -> None:
    err = np.abs(true_mat - pred_mat)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, mat, ttl, cmap in zip(
        axes,
        [true_mat, pred_mat, err],
        ["True", "Pred", "Abs error"],
        ["viridis", "viridis", "magma"],
    ):
        im = ax.imshow(mat, cmap=cmap, aspect="equal")
        ax.set_title(f"{title_prefix}: {ttl}")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _reshape_od(flat_or_mat: np.ndarray) -> np.ndarray:
    arr = np.asarray(flat_or_mat, dtype=np.float64)
    if arr.ndim == 2 and arr.shape == (24, 24):
        return arr
    if arr.ndim == 1 and arr.size == 576:
        return arr.reshape(24, 24)
    if arr.ndim == 2 and arr.shape[1] == 576:
        return arr[0].reshape(24, 24)
    raise ValueError(f"Unexpected OD shape: {arr.shape}")


def generate_bar_scatter(bench: Path, out: Path) -> list[Path]:
    paths: list[Path] = []
    sns.set_theme(style="whitegrid")
    test_df = pd.read_csv(bench / "benchmark_results.csv")

    specs = [
        ("mae", "test_od_mae.png", "Synthetic test OD MAE", "steelblue"),
        ("forward_rmse", "test_forward_rmse.png", "Synthetic test forward RMSE (turning)", "darkorange"),
        ("pearson", "test_pearson.png", "Synthetic test Pearson r (OD)", "seagreen"),
        ("production_mae", "test_production_mae.png", "Synthetic test production MAE", "slateblue"),
        ("attraction_mae", "test_attraction_mae.png", "Synthetic test attraction MAE", "teal"),
    ]
    for col, fname, title, color in specs:
        p = out / fname
        _barplot(test_df, "model", col, title, p, color)
        paths.append(p)

    p = out / "test_od_vs_forward.png"
    _scatter_od_vs_forward(
        test_df,
        "mae",
        "forward_rmse",
        "Synthetic test: OD MAE vs forward RMSE",
        p,
        hue_col="family",
    )
    paths.append(p)

    survey_df = _load_survey_df(bench / "survey_inference" / "survey_results.json")
    for col, fname, title, color in [
        ("mae", "survey_od_mae.png", "Survey OD MAE", "steelblue"),
        ("forward_rmse", "survey_forward_rmse.png", "Survey forward RMSE (turning)", "darkorange"),
    ]:
        p = out / fname
        _barplot(survey_df, "model", col, title, p, color)
        paths.append(p)

    p = out / "survey_od_vs_forward.png"
    _scatter_od_vs_forward(
        survey_df,
        "mae",
        "forward_rmse",
        "Survey: OD MAE vs forward RMSE",
        p,
    )
    paths.append(p)
    return paths


def generate_survey_heatmaps(bench: Path, out: Path, survey_od_path: Path) -> list[Path]:
    paths: list[Path] = []
    true = np.load(survey_od_path).astype(np.float64)
    np.fill_diagonal(true, 0.0)
    survey_dir = bench / "survey_inference"
    for model in HEATMAP_MODELS:
        pred_path = survey_dir / f"{model}_od.npy"
        if not pred_path.exists():
            continue
        pred = _reshape_od(np.load(pred_path))
        p = out / f"survey_heatmap_{model}.png"
        _heatmap_triplet(true, pred, model, p)
        paths.append(p)
    return paths


def generate_synthetic_heatmaps(bench: Path, out: Path, synth_od_path: Path) -> list[Path]:
    paths: list[Path] = []
    if not synth_od_path.exists():
        return paths
    splits = np.load(bench / "data" / "split_indices.npz")
    test_idx = int(splits["test_idx"][0])
    y_mm = np.load(synth_od_path, mmap_mode="r")
    true = np.asarray(y_mm[test_idx], dtype=np.float64).copy()
    np.fill_diagonal(true, 0.0)

    # Locate this sample within the test_pred rows (same order as test_idx)
    local_i = 0  # first test index → first row of test_pred
    meta = {
        "global_test_idx": test_idx,
        "local_test_row": local_i,
        "synth_od_path": str(synth_od_path),
    }
    (out / "synthetic_heatmap_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    for model in HEATMAP_MODELS:
        pred_path = bench / model / "test_pred.npy"
        if not pred_path.exists():
            continue
        pred_flat = np.load(pred_path, mmap_mode="r")[local_i]
        pred = _reshape_od(np.asarray(pred_flat))
        p = out / f"synthetic_heatmap_{model}.png"
        _heatmap_triplet(true, pred, f"{model} (test[{local_i}])", p)
        paths.append(p)
    return paths


def generate_nullspace_diagnostics(bench: Path, out: Path, survey_od_path: Path) -> list[Path]:
    """Survey-only nullspace decomposition using saved survey OD predictions."""
    paths: list[Path] = []
    survey_dir = bench / "survey_inference"
    pinv_path = survey_dir / "moore_penrose_od.npy"
    ns_path = survey_dir / "nullspace_mlp_od.npy"
    a_path = Path(
        json.loads((bench / "run_config.json").read_text()).get(
            "a_turn_path",
            str(REPO_ROOT / "outputs" / "turning_counts_fp" / "A_turn.npy"),
        )
    )
    if not (pinv_path.exists() and ns_path.exists() and a_path.exists()):
        return paths

    y_pinv = _reshape_od(np.load(pinv_path)).ravel()
    y_ns = _reshape_od(np.load(ns_path)).ravel()
    true = np.load(survey_od_path).astype(np.float64)
    np.fill_diagonal(true, 0.0)
    y_true = true.ravel()
    a = np.load(a_path).astype(np.float64)

    delta = y_ns - y_pinv
    leak = delta @ a.T  # (n_turns,)
    x_from_true = y_true @ a.T

    # Prefer saved survey turning counts if available via run_config turning path
    run_cfg = json.loads((bench / "run_config.json").read_text())
    turning_path = Path(run_cfg["turning_counts_path"])
    survey_i = int(run_cfg.get("survey_turning_index", 100000))
    if turning_path.exists():
        x_mm = np.load(turning_path, mmap_mode="r")
        if x_mm.shape[0] > survey_i:
            x_survey = np.asarray(x_mm[survey_i], dtype=np.float64)
        else:
            x_survey = x_from_true
    else:
        x_survey = x_from_true

    a_pinv = np.load(bench / "operator" / "a_pinv.npy").astype(np.float64)
    y_part = x_survey @ a_pinv.T  # A+ x

    stats = {
        "survey_od_mae_moore_penrose": float(np.mean(np.abs(y_pinv - y_true))),
        "survey_od_mae_nullspace_mlp": float(np.mean(np.abs(y_ns - y_true))),
        "norm_y_ns_minus_Aplus_x": float(np.linalg.norm(y_ns - y_part)),
        "norm_y_ns_minus_y_pinv": float(np.linalg.norm(delta)),
        "norm_A_times_y_ns_minus_y_pinv": float(np.linalg.norm(leak)),
        "rel_nullspace_leak_via_A": float(
            np.linalg.norm(leak) / max(np.linalg.norm(x_survey), 1e-6)
        ),
        "mean_abs_null_component_cell": float(np.mean(np.abs(delta))),
        "note": (
            "Computed from survey_inference/*_od.npy and operator/a_pinv.npy; "
            "no checkpoint restore."
        ),
    }
    (out / "nullspace_diagnostics.json").write_text(json.dumps(stats, indent=2) + "\n")

    # Bar summary figure
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = [
        "OD MAE\npinv",
        "OD MAE\nnullspace",
        "||ŷ_ns−A⁺x||\n/1000",
        "||A(ŷ_ns−ŷ_pinv)||",
        "rel leak\nvia A",
    ]
    values = [
        stats["survey_od_mae_moore_penrose"],
        stats["survey_od_mae_nullspace_mlp"],
        stats["norm_y_ns_minus_Aplus_x"] / 1000.0,
        stats["norm_A_times_y_ns_minus_y_pinv"],
        stats["rel_nullspace_leak_via_A"],
    ]
    ax.bar(labels, values, color=["#4c72b0", "#55a868", "#c44e52", "#8172b2", "#ccb974"])
    ax.set_title("Survey null-space diagnostics (artifact-derived)")
    ax.set_ylabel("Value (see labels)")
    fig.tight_layout()
    p = out / "nullspace_diagnostics.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(p)

    # Component heatmap: nullspace correction
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    mats = [
        y_pinv.reshape(24, 24),
        y_ns.reshape(24, 24),
        delta.reshape(24, 24),
    ]
    titles = ["Moore–Penrose ŷ", "Nullspace MLP ŷ", "ŷ_ns − ŷ_pinv"]
    for ax, mat, ttl in zip(axes, mats, titles):
        im = ax.imshow(mat, cmap="coolwarm" if ttl.startswith("ŷ") else "viridis", aspect="equal")
        ax.set_title(ttl)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Survey null-space correction structure", y=1.02)
    fig.tight_layout()
    p2 = out / "nullspace_correction_heatmap.png"
    fig.savefig(p2, dpi=140, bbox_inches="tight")
    plt.close(fig)
    paths.append(p2)
    return paths


def copy_existing(bench: Path, out: Path) -> list[Path]:
    paths: list[Path] = []
    src_dir = bench / "figures"
    for name in EXISTING_FIGURES:
        src = src_dir / name
        if src.exists():
            dst = out / f"benchmark_report_{name}"
            shutil.copy2(src, dst)
            paths.append(dst)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, default=DEFAULT_BENCH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--survey-od",
        type=Path,
        default=REPO_ROOT / "data" / "EstimatedODMatrix.npy",
    )
    parser.add_argument(
        "--synthetic-od",
        type=Path,
        default=REPO_ROOT
        / "outputs"
        / "od_generator_fp"
        / "synthetic_od_fp_synthetics_only.npy",
    )
    args = parser.parse_args()

    bench = args.benchmark_dir
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")
    written: list[Path] = []
    written += generate_bar_scatter(bench, out)
    written += generate_survey_heatmaps(bench, out, args.survey_od)
    written += generate_synthetic_heatmaps(bench, out, args.synthetic_od)
    written += generate_nullspace_diagnostics(bench, out, args.survey_od)
    written += copy_existing(bench, out)

    manifest = {
        "n_figures": len(written),
        "figures": [str(p.relative_to(REPO_ROOT)) for p in written],
        "benchmark_dir": str(bench),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(written)} figures to {out}")
    for p in written:
        print(f"  {p.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
