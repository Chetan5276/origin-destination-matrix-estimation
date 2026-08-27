"""Optuna HPO helpers — objective = validation composite score (minimize)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np

from src.ml.rigorous_benchmark.metrics_suite import composite_score, compute_all_metrics

logger = logging.getLogger(__name__)


def _import_optuna():
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return optuna


def run_study(
    study_name: str,
    objective_fn: Callable[[Any], float],
    n_trials: int,
    *,
    seed: int = 42,
    direction: str = "minimize",
) -> Any:
    optuna = _import_optuna()
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(study_name=study_name, direction=direction, sampler=sampler)
    study.optimize(objective_fn, n_trials=n_trials, show_progress_bar=False)
    logger.info(
        "Study %s best=%.6g params=%s",
        study_name,
        study.best_value,
        study.best_params,
    )
    return study


def val_composite_from_preds(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    x_turn: np.ndarray,
    a_turn: np.ndarray,
    *,
    alpha: float,
    beta: float,
    gamma: float,
    y_pinv: np.ndarray | None = None,
) -> float:
    m = compute_all_metrics(y_true, y_pred, x_turn, a_turn, y_pinv=y_pinv)
    return composite_score(m, alpha=alpha, beta=beta, gamma=gamma, refs=None)


def save_best_params(path: Path, params: dict[str, Any], extra: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"best_params": params}
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, indent=2, default=str))
