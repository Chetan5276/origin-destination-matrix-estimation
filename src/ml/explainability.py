"""Stage 9: SHAP explainability for tree models."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def generate_shap_summary(
    model,
    x_sample: np.ndarray,
    output_path: Path,
    max_samples: int = 500,
) -> Path | None:
    try:
        import shap
    except ImportError:
        logger.warning("SHAP not installed; skipping explainability")
        return None

    n = min(max_samples, x_sample.shape[0])
    xs = x_sample[:n]

    # MultiOutputRegressor: explain first estimator as proxy
    base = model
    if hasattr(model, "estimators_") and model.estimators_:
        base = model.estimators_[0]

    try:
        explainer = shap.Explainer(base, xs)
        shap_values = explainer(xs)
    except Exception:
        explainer = shap.KernelExplainer(base.predict, shap.sample(xs, min(100, n)))
        shap_values = explainer.shap_values(xs[: min(50, n)])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, xs, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved SHAP summary to %s", output_path)
    return output_path
