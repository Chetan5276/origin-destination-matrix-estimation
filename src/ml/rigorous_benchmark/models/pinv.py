"""Moore–Penrose pseudoinverse baseline (no HPO)."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.ml.rigorous_benchmark.constraints import apply_constraint_strategy
from src.ml.rigorous_benchmark.data import BenchmarkData
from src.ml.rigorous_benchmark.models.base import BenchmarkModel, FitResult
from src.ml.rigorous_benchmark.operator import OperatorInfo, pinv_predict


class MoorePenroseModel(BenchmarkModel):
    name = "moore_penrose"

    def hyperopt(self, data: BenchmarkData, operator: OperatorInfo) -> dict[str, Any]:
        return {}

    def fit(
        self,
        data: BenchmarkData,
        operator: OperatorInfo,
        params: dict[str, Any],
        *,
        use_train_val: bool = True,
    ) -> FitResult:
        result = FitResult(
            model_name=self.name,
            best_params={},
            constraint_strategy="none",
            notes="y = A+ @ x; no trainable params; constraints not applied by default",
        )
        self.fit_result = result
        self.save_metadata(result)
        np.save(self.model_dir() / "singular_values.npy", operator.s)
        return result

    def predict(
        self,
        x_raw: np.ndarray,
        data: BenchmarkData,
        operator: OperatorInfo,
    ) -> np.ndarray:
        y = pinv_predict(operator.a_pinv, x_raw)
        # Report strategy: none (raw pinv may be negative)
        return y
