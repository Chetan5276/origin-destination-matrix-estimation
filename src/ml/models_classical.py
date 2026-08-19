"""Stage 3--4: classical and tree-based OD estimators."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: object
    tune: bool = False
    param_grid: dict | None = None


def _multi(est, n_jobs: int = -1) -> MultiOutputRegressor:
    return MultiOutputRegressor(est, n_jobs=n_jobs)


def classical_models(n_jobs: int = -1, include_sparse: bool = False) -> list[ModelSpec]:
    specs = [
        ModelSpec("linear", LinearRegression(n_jobs=n_jobs)),
        ModelSpec(
            "ridge",
            Ridge(),
            tune=True,
            param_grid={"alpha": [0.1, 1.0, 10.0, 100.0, 1000.0]},
        ),
        ModelSpec(
            "pls",
            PLSRegression(n_components=min(25, 178)),
            tune=False,
        ),
    ]
    if include_sparse:
        specs.extend(
            [
                ModelSpec("lasso", _multi(Lasso(max_iter=2000), n_jobs), tune=False),
                ModelSpec(
                    "elastic_net",
                    _multi(ElasticNet(max_iter=2000, alpha=0.01, l1_ratio=0.5), n_jobs),
                    tune=False,
                ),
            ]
        )
    return specs


def tree_models(n_jobs: int = -1, seed: int = 42) -> list[ModelSpec]:
    specs = [
        ModelSpec(
            "random_forest",
            RandomForestRegressor(
                n_estimators=50, max_depth=16, random_state=seed, n_jobs=n_jobs
            ),
        ),
        ModelSpec(
            "extra_trees",
            ExtraTreesRegressor(
                n_estimators=50, max_depth=16, random_state=seed, n_jobs=n_jobs
            ),
        ),
    ]
    try:
        from xgboost import XGBRegressor

        specs.append(
            ModelSpec(
                "xgboost",
                _multi(
                    XGBRegressor(
                        n_estimators=100,
                        max_depth=8,
                        learning_rate=0.1,
                        random_state=seed,
                        n_jobs=n_jobs,
                        verbosity=0,
                    ),
                    n_jobs=1,
                ),
            )
        )
    except ImportError:
        logger.warning("XGBoost not installed; skipping xgboost model")
    return specs


def fit_model(
    spec: ModelSpec,
    x_train: np.ndarray,
    y_train: np.ndarray,
    cv: int = 3,
    n_jobs: int = -1,
) -> object:
    if spec.tune and spec.param_grid:
        search = GridSearchCV(
            spec.estimator,
            spec.param_grid,
            cv=cv,
            n_jobs=n_jobs,
            scoring="neg_mean_absolute_error",
        )
        search.fit(x_train, y_train)
        logger.info("%s best params: %s", spec.name, search.best_params_)
        return search.best_estimator_
    spec.estimator.fit(x_train, y_train)
    return spec.estimator


def predict_model(model: object, x: np.ndarray) -> np.ndarray:
    pred = model.predict(x)
    return np.clip(np.asarray(pred, dtype=np.float32), 0, None)
