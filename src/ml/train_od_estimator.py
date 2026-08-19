#!/usr/bin/env python3
"""Phase 3 CLI: train and benchmark OD estimation models."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import OUTPUT_DIR
from src.ml.benchmark import run_benchmark
from src.ml.config import ML_OUTPUT_DIR, NeuralTrainConfig, TrainConfig

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train OD estimators from turning counts (Phase 3).")
    parser.add_argument(
        "--turning",
        type=Path,
        default=OUTPUT_DIR / "turning_counts_1m" / "turning_counts.npy",
        help="Clean turning counts (default: turning_counts.npy)",
    )
    parser.add_argument(
        "--turning-noisy",
        action="store_true",
        help="Use turning_counts_noisy.npy instead of clean counts",
    )
    parser.add_argument(
        "--od",
        type=Path,
        default=OUTPUT_DIR / "od_generator" / "synthetic_od_1m.npy",
    )
    parser.add_argument(
        "--a-turn",
        type=Path,
        default=OUTPUT_DIR / "turning_counts_1m" / "A_turn.npy",
    )
    parser.add_argument("--output-dir", type=Path, default=ML_OUTPUT_DIR)
    parser.add_argument("--max-samples", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-neural", action="store_true")
    parser.add_argument("--skip-analysis", action="store_true")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--ae-epochs", type=int, default=40)
    parser.add_argument("--finetune-epochs", type=int, default=15)
    parser.add_argument("--forward-weight", type=float, default=0.5)
    parser.add_argument("--production-weight", type=float, default=0.1)
    parser.add_argument("--attraction-weight", type=float, default=0.1)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    train_config = TrainConfig(
        seed=args.seed,
        max_samples=args.max_samples,
        use_clean_turning=not args.turning_noisy,
        standardize_y=True,
    )
    neural_config = NeuralTrainConfig(
        epochs=args.epochs,
        autoencoder_epochs=args.ae_epochs,
        finetune_epochs=args.finetune_epochs,
        forward_weight=args.forward_weight,
        production_weight=args.production_weight,
        attraction_weight=args.attraction_weight,
    )

    run_benchmark(
        turning_path=args.turning,
        od_path=args.od,
        a_turn_path=args.a_turn,
        output_dir=args.output_dir,
        config=train_config,
        include_neural=not args.no_neural,
        skip_analysis=args.skip_analysis,
        neural_config=neural_config,
    )


if __name__ == "__main__":
    main()
