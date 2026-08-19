"""Stage 5--6: neural model configuration (training in neural_trainer.py)."""

from __future__ import annotations

from src.ml.config import NeuralTrainConfig

# Re-export for backward compatibility
TorchMLPConfig = NeuralTrainConfig

__all__ = ["NeuralTrainConfig", "TorchMLPConfig"]
