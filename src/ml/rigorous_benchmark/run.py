"""CLI orchestrator for the rigorous OD estimation benchmark.

Usage
-----
python -m src.ml.rigorous_benchmark.run --smoke
python -m src.ml.rigorous_benchmark.run --models ridge,mlp --n-trials 5
python -m src.ml.rigorous_benchmark.run   # full pragmatic defaults
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.ml.metrics import attraction_vector, production_vector
from src.ml.rigorous_benchmark.config import ALL_MODEL_NAMES, BenchmarkConfig, smoke_config
from src.ml.rigorous_benchmark.constraints import ABLATION_STRATEGIES, apply_constraint_strategy
from src.ml.rigorous_benchmark.data import load_benchmark_data
from src.ml.rigorous_benchmark.inference import run_survey_inference
from src.ml.rigorous_benchmark.metrics_suite import compute_all_metrics
from src.ml.rigorous_benchmark.models import build_model
from src.ml.rigorous_benchmark.operator import (
    compute_operator,
    operator_summary_dict,
    pinv_predict,
    save_operator,
)
from src.ml.rigorous_benchmark.report import make_figures, write_final_report, write_results
from src.ml.rigorous_benchmark.selection import attach_composite, val_refs_from_rows

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("rigorous_benchmark")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rigorous OD estimation benchmark")
    p.add_argument("--smoke", action="store_true", help="Tiny N / few trials end-to-end")
    p.add_argument(
        "--models",
        type=str,
        default=",".join(ALL_MODEL_NAMES),
        help="Comma-separated model names (default: all 13)",
    )
    p.add_argument("--n-trials", type=int, default=None)
    p.add_argument("--hpo-subsample", type=int, default=None)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--neural-epochs", type=int, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-ablations", action="store_true")
    p.add_argument("--skip-survey", action="store_true")
    p.add_argument("--skip-hpo", action="store_true", help="Use default params (debug)")
    return p.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> BenchmarkConfig:
    models = tuple(m.strip() for m in args.models.split(",") if m.strip())
    if args.smoke:
        cfg = smoke_config(models=models, seed=args.seed, run_ablations=not args.no_ablations)
    else:
        cfg = BenchmarkConfig(models=models, seed=args.seed, run_ablations=not args.no_ablations)
    updates: dict[str, Any] = {}
    if args.n_trials is not None:
        updates["n_trials"] = args.n_trials
    if args.hpo_subsample is not None:
        updates["hpo_train_subsample"] = args.hpo_subsample
    if args.max_samples is not None:
        updates["max_samples"] = args.max_samples
    if args.neural_epochs is not None:
        updates["neural_epochs"] = args.neural_epochs
        updates["autoencoder_epochs"] = max(2, args.neural_epochs)
        updates["finetune_epochs"] = max(1, args.neural_epochs // 3)
    if args.output_dir is not None:
        updates["output_dir"] = Path(args.output_dir)
    if updates:
        cfg = cfg.with_updates(**updates)
    return cfg


def evaluate_split(
    name: str,
    model,
    data,
    operator,
    *,
    split: str,
    constraint_strategy: str,
) -> dict[str, Any]:
    if split == "val":
        x_raw, y_true = data.x_val_raw, data.y_val_raw
    elif split == "test":
        x_raw, y_true = data.x_test_raw, data.y_test_raw
    else:
        raise ValueError(split)
    pred = model.predict(x_raw, data, operator)
    y_pinv = pinv_predict(operator.a_pinv, x_raw)
    metrics = compute_all_metrics(y_true, pred, x_raw, operator.a_turn, y_pinv=y_pinv)
    return {
        "model": name,
        "split": split,
        "metrics": metrics,
        "constraint_strategy": constraint_strategy,
        "predictions": pred,
    }


def _metrics_from_saved_preds(
    name: str,
    mdir: Path,
    data,
    operator,
    *,
    constraint_strategy: str,
) -> tuple[dict | None, dict]:
    """Build val/test metric rows from on-disk predictions (no refit)."""
    test_pred = np.load(mdir / "test_pred.npy")
    y_pinv_te = pinv_predict(operator.a_pinv, data.x_test_raw)
    test_row = {
        "model": name,
        "split": "test",
        "metrics": compute_all_metrics(
            data.y_test_raw, test_pred, data.x_test_raw, operator.a_turn, y_pinv=y_pinv_te
        ),
        "constraint_strategy": constraint_strategy,
    }
    val_row = None
    if (mdir / "val_pred.npy").exists():
        val_pred = np.load(mdir / "val_pred.npy")
        y_pinv_va = pinv_predict(operator.a_pinv, data.x_val_raw)
        val_row = {
            "model": name,
            "split": "val",
            "metrics": compute_all_metrics(
                data.y_val_raw, val_pred, data.x_val_raw, operator.a_turn, y_pinv=y_pinv_va
            ),
            "constraint_strategy": constraint_strategy,
        }
    return val_row, test_row


def try_restore_model(model, name: str, mdir: Path, params: dict, data, operator) -> bool:
    """Restore fitted weights from disk for survey/ablations (no HPO/retrain)."""
    import joblib
    import torch

    from src.ml.neural_core import DirectODRegressor, LatentODModel, LatentPredictor, ODDecoder
    from src.ml.rigorous_benchmark.models.base import FitResult
    from src.ml.rigorous_benchmark.neural_train import (
        bundle_from_data,
        neural_config_from_benchmark,
    )

    if name == "moore_penrose":
        model.fit(data, operator, {}, use_train_val=False)
        return True
    if name == "tikhonov":
        model.fit(data, operator, params or {"lambda": 1.0}, use_train_val=False)
        return True

    joblib_path = mdir / "model.joblib"
    pt_path = mdir / "model.pt"
    if joblib_path.exists():
        payload = joblib.load(joblib_path)
        if name == "ridge":
            model.model = payload
        elif name == "pls":
            model.model = payload
        elif name == "physics_ridge":
            model.model = payload["model"]
            model.mu = float(payload["mu"])
            model.alpha = float(payload["alpha"])
        else:
            return False
        model.fit_result = FitResult(
            model_name=name,
            best_params=params,
            constraint_strategy="none",
            notes="restored from joblib",
        )
        return True

    if pt_path.exists():
        ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
        p = ckpt.get("params") or params or {}
        cfg = neural_config_from_benchmark(model.config, p)
        model.params = p
        model.cfg = cfg
        model.bundle = bundle_from_data(data)
        if name == "mlp":
            model.torch_model = DirectODRegressor(
                data.n_features, data.n_targets, cfg, residual_blocks=False
            )
            model.torch_model.load_state_dict(ckpt["state_dict"])
        elif name in ("residual_mlp", "physics_residual_mlp"):
            from dataclasses import replace

            n_blocks = int(p.get("num_res_blocks", getattr(cfg, "num_res_blocks", 1)))
            cfg = replace(cfg, num_res_blocks=n_blocks)
            model.cfg = cfg
            model.torch_model = DirectODRegressor(
                data.n_features, data.n_targets, cfg, residual_blocks=True
            )
            model.torch_model.load_state_dict(ckpt["state_dict"])
        elif name.startswith("ae_"):
            latent = int(getattr(model, "latent_dim", 64))
            dec = ODDecoder(latent, cfg)
            pred_net = LatentPredictor(data.n_features, latent, cfg)
            model.torch_model = LatentODModel(pred_net, dec)
            model.torch_model.load_state_dict(ckpt["state_dict"])
            model.latent_dim = latent
            model.params = p
            model.cfg = cfg
            model.bundle = bundle_from_data(data)
        elif name == "nullspace_mlp":
            from src.ml.rigorous_benchmark.models.nullspace_mlp import NullspaceNet

            nullity = int(operator.null_basis.shape[1])
            net = NullspaceNet(data.n_features, nullity, cfg)
            net.load_state_dict(ckpt["state_dict"])
            model.torch_model = net
            model.null_penalty = float(p.get("null_penalty", 0.1))
            model.params = p
            model.cfg = cfg
            model.bundle = bundle_from_data(data)
        else:
            return False
        model.fit_result = FitResult(
            model_name=name,
            best_params=p,
            constraint_strategy=getattr(cfg, "output_activation", "softplus"),
            notes="restored from checkpoint",
        )
        return True
    return False


def run_ablations(model, name: str, data, operator, out_dir: Path) -> dict:
    """Constraint strategy ablations on a frozen model's raw-ish predictions."""
    out_dir.mkdir(parents=True, exist_ok=True)
    x_raw, y_true = data.x_test_raw, data.y_test_raw
    # Get unconstrained-ish prediction: use model predict then re-apply strategies
    # For fair ablation, start from model output before additional projection when possible.
    base_pred = model.predict(x_raw, data, operator)
    y_pinv = pinv_predict(operator.a_pinv, x_raw)
    prod = production_vector(y_true)
    attr = attraction_vector(y_true)
    summary = {}
    for strat in ABLATION_STRATEGIES:
        y_out, meta = apply_constraint_strategy(
            base_pred,
            strategy=strat,
            support_mask=data.support_mask,
            target_productions=prod,
            target_attractions=attr,
        )
        metrics = compute_all_metrics(y_true, y_out, x_raw, operator.a_turn, y_pinv=y_pinv)
        summary[strat] = {
            "meta": meta,
            "mae": metrics["mae"],
            "forward_mae": metrics["forward_mae"],
            "production_mae": metrics["production_mae"],
            "attraction_mae": metrics["attraction_mae"],
        }
        np.save(out_dir / f"{name}_{strat}_pred.npy", y_out[: min(100, len(y_out))])
    (out_dir / f"{name}_ablation.json").write_text(json.dumps(summary, indent=2))
    return summary


def run_benchmark(config: BenchmarkConfig, *, skip_hpo: bool = False, skip_survey: bool = False) -> dict:
    import gc

    t0 = time.time()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "run_config.json").write_text(json.dumps(config.to_dict(), indent=2))

    logger.info("Loading data…")
    data = load_benchmark_data(config)
    leakage = json.loads((config.data_dir() / "leakage_report.json").read_text())

    logger.info("Computing operator SVD…")
    operator = compute_operator(data.a_turn, rtol=config.svd_rtol)
    save_operator(operator, config.operator_dir())
    op_meta = operator_summary_dict(operator)

    n_train = int(data.x_train.shape[0])
    n_val = int(data.x_val.shape[0])
    n_test = int(data.x_test.shape[0])

    def _enrich(row: dict, *, params: dict | None = None, train_time: float | None = None, n_params=None) -> dict:
        out = dict(row)
        out["train_samples"] = n_train
        out["val_samples"] = n_val
        out["test_samples"] = n_test
        if params is not None:
            out["best_hyperparameters"] = json.dumps(params, default=str)
        if train_time is not None:
            out["train_time"] = float(train_time)
        if n_params is not None:
            out["parameters"] = int(n_params)
        return out

    def _count_params(m) -> int | None:
        tm = getattr(m, "torch_model", None)
        if tm is None:
            return None
        try:
            import torch

            return int(sum(p.numel() for p in tm.parameters()))
        except Exception:
            return None

    fitted = {}
    val_rows = []
    test_rows = []

    for name in config.models:
        logger.info("=== Model: %s ===", name)
        mdir = config.model_dir(name)
        pred_path = mdir / "test_pred.npy"
        params_path = mdir / "best_params.json"
        model = build_model(name, config)

        if pred_path.exists() and params_path.exists() and not config.smoke:
            logger.info("Skipping finished model %s (using saved preds; restore weights if present)", name)
            meta = json.loads(params_path.read_text())
            params = meta.get("best_params") or {}
            cstrat = meta.get("constraint_strategy", "unknown")
            val_row, test_row = _metrics_from_saved_preds(
                name, mdir, data, operator, constraint_strategy=cstrat
            )
            test_rows.append(_enrich(test_row, params=params))
            if val_row is not None:
                val_rows.append(_enrich(val_row, params=params))
            if try_restore_model(model, name, mdir, params, data, operator):
                fitted[name] = model
            else:
                logger.warning("Could not restore %s for survey/ablations; metrics still included", name)
            gc.collect()
            continue

        t_model = time.time()
        if skip_hpo or name == "moore_penrose":
            params = {}
            if name == "tikhonov":
                params = {"lambda": 1.0}
            elif name == "ridge":
                params = {"alpha": 1.0}
            elif name == "physics_ridge":
                params = {"alpha": 1.0, "mu": 0.1}
            elif name == "pls":
                params = {"n_components": min(15, data.n_features)}
            elif name == "nullspace_mlp":
                params = {"null_penalty": 0.1, "hidden0": 256, "hidden1": 512, "lr": 1e-3, "batch_size": 128}
            else:
                params = {
                    "hidden0": 256,
                    "hidden1": 512,
                    "lr": 1e-3,
                    "weight_decay": 1e-5,
                    "batch_size": config.neural_batch_size,
                    "forward_weight": config.forward_weight,
                    "output_activation": "softplus",
                    "num_res_blocks": 2,
                    "width": 256,
                }
        else:
            params = model.hyperopt(data, operator)

        model.fit(data, operator, params, use_train_val=config.final_retrain_on_train_val)
        fitted[name] = model
        train_time = time.time() - t_model
        n_params = _count_params(model)

        cstrat = model.fit_result.constraint_strategy if model.fit_result else "unknown"
        t_inf = time.time()
        val_res = evaluate_split(name, model, data, operator, split="val", constraint_strategy=cstrat)
        val_row = {k: v for k, v in val_res.items() if k != "predictions"}
        val_row["inference_time"] = time.time() - t_inf
        val_rows.append(_enrich(val_row, params=params, train_time=train_time, n_params=n_params))
        mdir.mkdir(parents=True, exist_ok=True)
        np.save(mdir / "val_pred.npy", val_res["predictions"])
        del val_res

        t_inf = time.time()
        test_res = evaluate_split(name, model, data, operator, split="test", constraint_strategy=cstrat)
        test_row = {k: v for k, v in test_res.items() if k != "predictions"}
        test_row["inference_time"] = time.time() - t_inf
        np.save(mdir / "test_pred.npy", test_res["predictions"])
        test_rows.append(_enrich(test_row, params=params, train_time=train_time, n_params=n_params))
        logger.info(
            "%s test MAE=%.4g forward_mae=%.4g (%.1fs)",
            name,
            test_res["metrics"]["mae"],
            test_res["metrics"]["forward_mae"],
            train_time,
        )
        del test_res
        # Free GPU/CPU between sequential models
        if name not in ("ridge", "mlp", "moore_penrose"):
            # keep a few for ablations; drop heavy torch graphs otherwise after save
            pass
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    refs = val_refs_from_rows(val_rows)
    val_rows = attach_composite(
        val_rows,
        alpha=config.alpha_fwd,
        beta=config.beta_prod,
        gamma=config.gamma_attr,
        refs=refs,
        split_key="val",
    )
    test_rows = attach_composite(
        test_rows,
        alpha=config.alpha_fwd,
        beta=config.beta_prod,
        gamma=config.gamma_attr,
        refs=refs,
        split_key="test",
    )
    all_rows = val_rows + test_rows
    df = write_results(all_rows, config.output_dir)
    make_figures(df, config.figures_dir())

    ablation_summary = {}
    if config.run_ablations and fitted:
        for key in ("ridge", "mlp", "moore_penrose"):
            if key in fitted:
                ablation_summary[key] = run_ablations(
                    fitted[key],
                    key,
                    data,
                    operator,
                    config.ablations_dir(),
                )

    survey_results = None
    if not skip_survey:
        survey_results = run_survey_inference(fitted, data, operator, config.survey_dir())

    write_final_report(
        config.output_dir,
        df=df,
        leakage=leakage,
        operator_meta=op_meta,
        survey_results=survey_results,
        ablation_summary=ablation_summary,
        config_dict=config.to_dict(),
    )

    elapsed = time.time() - t0
    summary = {
        "elapsed_sec": elapsed,
        "n_models": len(test_rows),
        "output_dir": str(config.output_dir),
    }
    (config.output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("Done in %.1fs → %s", elapsed, config.output_dir)
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = config_from_args(args)
    run_benchmark(config, skip_hpo=args.skip_hpo, skip_survey=args.skip_survey)
    return 0


if __name__ == "__main__":
    sys.exit(main())
