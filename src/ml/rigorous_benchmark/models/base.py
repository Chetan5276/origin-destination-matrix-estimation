"""Shared model protocol for the rigorous benchmark."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.ml.rigorous_benchmark.config import BenchmarkConfig
from src.ml.rigorous_benchmark.data import BenchmarkData
from src.ml.rigorous_benchmark.operator import OperatorInfo


@dataclass
class FitResult:
    model_name: str
    best_params: dict[str, Any] = field(default_factory=dict)
    history: dict[str, Any] = field(default_factory=dict)
    artifact_paths: dict[str, str] = field(default_factory=dict)
    constraint_strategy: str = "none"
    notes: str = ""


class BenchmarkModel(ABC):
    """Thin interface: HPO on val → retrain → predict in original OD units."""

    name: str = "base"

    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config
        self.fit_result: FitResult | None = None

    @abstractmethod
    def hyperopt(
        self,
        data: BenchmarkData,
        operator: OperatorInfo,
    ) -> dict[str, Any]:
        """Tune on validation; return best hyperparams. Must not touch test."""

    @abstractmethod
    def fit(
        self,
        data: BenchmarkData,
        operator: OperatorInfo,
        params: dict[str, Any],
        *,
        use_train_val: bool = True,
    ) -> FitResult:
        """Final fit (typically train+val)."""

    @abstractmethod
    def predict(
        self,
        x_raw: np.ndarray,
        data: BenchmarkData,
        operator: OperatorInfo,
    ) -> np.ndarray:
        """Predict OD in original units, shape (N, 576)."""

    def model_dir(self) -> Path:
        return self.config.model_dir(self.name)

    def save_metadata(self, result: FitResult) -> None:
        d = self.model_dir()
        d.mkdir(parents=True, exist_ok=True)
        (d / "config.json").write_text(json.dumps(self.config.to_dict(), indent=2))
        (d / "best_params.json").write_text(
            json.dumps(
                {
                    "best_params": result.best_params,
                    "constraint_strategy": result.constraint_strategy,
                    "notes": result.notes,
                    "history": result.history,
                },
                indent=2,
                default=str,
            )
        )
