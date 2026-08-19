#!/usr/bin/env python3
"""Phase 2: Probabilistic OD → turning-count generation pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import DATA_DIR, OUTPUT_DIR, REPORT_DIR
from src.data.assignment_rank import compare_assignment_matrices, extended_rank_analysis
from src.data.build_assignment_matrix import build_assignment_matrix
from src.data.fractional_assignment import (
    build_fractional_assignment_matrix,
    validate_fractional_matrix,
)
from src.data.k_shortest_paths import enumerate_k_shortest_paths
from src.data.network_parser import parse_sumo_network
from src.data.observation_noise import NoiseConfig, apply_observation_noise, noise_level_to_scale
from src.data.od_pairs import OdPairIndex
from src.data.route_assignment import assign_routes
from src.data.route_choice import apply_logit_choice
from src.data.batch_turning import (
    DEFAULT_BATCH_SIZE,
    LARGE_DATASET_THRESHOLD,
    generate_turning_counts_batched,
    sample_pairwise_distance,
    sample_turn_correlation,
)
from src.data.turning_movements import enumerate_turning_movements, export_turning_movements_csv
from src.data.turning_statistics import (
    TurningDatasetStats,
    compute_turning_stats,
    create_turning_figures,
    run_sensitivity_analysis,
    write_turning_summary,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TurningPipelineConfig:
    """Configuration for probabilistic turning-count generation."""

    k_paths: int = 5
    theta: float = 0.1
    weight_metric: str = "length"
    noise_model: str = "poisson"
    noise_level_pct: float = 0.0
    seed: int = 42
    run_sensitivity: bool = False


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def _collect_route_probabilities(route_choices) -> list[float]:
    probs: list[float] = []
    for route_list in route_choices.choices.values():
        for rp in route_list:
            probs.append(rp.probability)
    return probs


def run_turning_pipeline(
    net_path: Path,
    od_path: Path,
    output_dir: Path,
    report_dir: Path,
    config: TurningPipelineConfig,
    max_samples: int | None = None,
    skip_plots: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> None:
    """Execute the probabilistic OD → turning-count pipeline."""
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    rng = np.random.default_rng(config.seed)

    # --- Stage 1: Network ---
    network = parse_sumo_network(net_path)
    turning_index = enumerate_turning_movements(network)
    od_index = OdPairIndex.build()
    timings["network_parse"] = time.perf_counter() - t0

    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    export_turning_movements_csv(
        turning_index, output_dir / "turning_movements.csv"
    )

    # --- Stage 4: K-shortest paths ---
    t1 = time.perf_counter()
    catalog = enumerate_k_shortest_paths(
        network,
        od_index,
        k_paths=config.k_paths,
        weight_metric=config.weight_metric,  # type: ignore[arg-type]
    )
    timings["k_shortest_paths"] = time.perf_counter() - t1

    # --- Stage 5: "Logit route choice ---
    t2 = time.perf_counter()
    route_choices = apply_logit_choice(catalog, theta=config.theta)
    timings["route_choice"] = time.perf_counter() - t2

    with (output_dir / "route_catalog.pkl").open("wb") as fh:
        pickle.dump({"catalog": catalog, "choices": route_choices}, fh)

    # --- Stages 6--7: Fractional A_turn ---
    t3 = time.perf_counter()
    a_turn_prob = build_fractional_assignment_matrix(
        turning_index, od_index, route_choices
    )
    frac_validation = validate_fractional_matrix(a_turn_prob)
    timings["fractional_assignment"] = time.perf_counter() - t3

    # Binary baseline for comparison
    t4 = time.perf_counter()
    binary_routes = assign_routes(
        network, od_index, weight_metric=config.weight_metric  # type: ignore[arg-type]
    )
    a_turn_binary = build_assignment_matrix(turning_index, od_index, binary_routes)
    rank_comparison = compare_assignment_matrices(a_turn_binary, a_turn_prob)
    timings["binary_baseline"] = time.perf_counter() - t4

    np.save(output_dir / "A_turn.npy", a_turn_prob.astype(np.float32))

    # --- Stages 9--10: Batched turning counts (memory-safe for 1M+ samples) ---
    t5 = time.perf_counter()
    od_probe = np.load(od_path, mmap_mode="r")
    n_samples = od_probe.shape[0]
    if max_samples is not None:
        n_samples = min(n_samples, max_samples)
    use_batched = n_samples >= LARGE_DATASET_THRESHOLD
    logger.info(
        "Processing %d OD matrices (batched=%s, batch_size=%d)",
        n_samples,
        use_batched,
        batch_size,
    )

    noise_config = NoiseConfig(
        model=config.noise_model,  # type: ignore[arg-type]
        poisson_scale=noise_level_to_scale(config.noise_level_pct),
    )

    if use_batched:
        batch_result = generate_turning_counts_batched(
            od_path,
            a_turn_prob,
            output_dir,
            batch_size=batch_size,
            max_samples=max_samples,
            noise_config=noise_config,
            seed=config.seed,
        )
        validation = {
            "matrix_mult_consistent": batch_result["matrix_mult_consistent"],
            "max_reconstruction_error": batch_result["max_reconstruction_error"],
            "batched": True,
            "batch_size": batch_size,
        }
        mean_corr, max_corr = sample_turn_correlation(
            output_dir / "turning_counts.npy", seed=config.seed
        )
        stats = TurningDatasetStats(
            num_samples=batch_result["num_samples"],
            num_turns=batch_result["num_turns"],
            mean_pairwise_od_distance=sample_pairwise_distance(
                od_path, batch_result["num_samples"], seed=config.seed
            ),
            mean_pairwise_turning_distance=sample_pairwise_distance(
                output_dir / "turning_counts.npy",
                batch_result["num_samples"],
                seed=config.seed + 1,
            ),
            mean_turn_correlation=mean_corr,
            max_turn_correlation=max_corr,
            per_turn_mean=batch_result["per_turn_mean"].tolist(),
            per_turn_std=batch_result["per_turn_std"].tolist(),
            per_turn_cv=batch_result["per_turn_cv"].tolist(),
        )
        turn_mean = batch_result["per_turn_mean"]
        turn_std = batch_result["per_turn_std"]
        turn_cv = batch_result["per_turn_cv"]
    else:
        from src.data.statistics import build_dataset, validate_dataset

        od_matrices = np.asarray(od_probe[:n_samples])
        dataset = build_dataset(od_matrices, a_turn_prob)
        clean_counts = dataset.x
        np.save(output_dir / "turning_counts.npy", clean_counts.astype(np.float32))
        np.save(output_dir / "clean_turn_counts.npy", clean_counts.astype(np.float32))
        noisy_counts = apply_observation_noise(clean_counts, noise_config, rng)
        np.save(output_dir / "turning_counts_noisy.npy", noisy_counts.astype(np.float32))
        np.save(output_dir / "noisy_turn_counts.npy", noisy_counts.astype(np.float32))
        validation = validate_dataset(dataset)
        stats = compute_turning_stats(od_matrices, clean_counts, seed=config.seed)
        turn_mean = np.array(stats.per_turn_mean)
        turn_std = np.array(stats.per_turn_std)
        turn_cv = np.array(stats.per_turn_cv)

    timings["turning_counts"] = time.perf_counter() - t5

    # --- Stage 11: Statistics ---
    t8 = time.perf_counter()
    route_probs = _collect_route_probabilities(route_choices)
    timings["statistics"] = time.perf_counter() - t8

    # --- Stage 12: Sensitivity (optional) ---
    sensitivity_df = None
    if config.run_sensitivity:
        t9 = time.perf_counter()
        sensitivity_df = run_sensitivity_analysis(
            network,
            od_index,
            turning_index,
            k_values=[1, 3, 5, 10],
            theta_values=[0.05, 0.1, 0.2, 0.5],
            weight_metric=config.weight_metric,
        )
        sensitivity_df.to_csv(output_dir / "sensitivity_analysis.csv", index=False)
        timings["sensitivity"] = time.perf_counter() - t9

    # --- Visualizations ---
    if not skip_plots:
        t10 = time.perf_counter()
        create_turning_figures(
            a_turn_prob,
            None,
            None,
            rank_comparison,
            route_probs,
            output_dir / "figures",
            turn_mean=turn_mean,
            turn_std=turn_std,
            turn_cv=turn_cv,
        )
        timings["visualization"] = time.perf_counter() - t10

    timings["total"] = time.perf_counter() - t0

    config_dict = asdict(config)
    write_turning_summary(
        stats,
        rank_comparison,
        config_dict,
        report_dir / "turning_dataset_summary.md",
    )

    analysis = {
        "config": config_dict,
        "num_turning_movements": turning_index.num_turning_movements,
        "num_od_pairs": od_index.num_od_pairs,
        "num_samples": stats.num_samples,
        "fractional_validation": frac_validation,
        "rank_comparison": rank_comparison.to_dict(),
        "dataset_validation": validation,
        "turning_stats": {
            "mean_pairwise_od_distance": stats.mean_pairwise_od_distance,
            "mean_pairwise_turning_distance": stats.mean_pairwise_turning_distance,
            "mean_turn_correlation": stats.mean_turn_correlation,
            "max_turn_correlation": stats.max_turn_correlation,
        },
        "route_catalog": {
            "total_routes": sum(len(v) for v in catalog.routes.values()),
            "unreachable_pairs": len(catalog.unreachable_pairs),
            "mean_routes_per_od": sum(len(v) for v in catalog.routes.values())
            / max(len(catalog.routes), 1),
        },
        "timings": timings,
    }
    if sensitivity_df is not None:
        best = sensitivity_df.loc[sensitivity_df["rank"].idxmax()]
        analysis["sensitivity_recommendation"] = {
            "k_paths": int(best["k_paths"]),
            "theta": float(best["theta"]),
            "rank": int(best["rank"]),
        }

    (output_dir / "assignment_analysis.json").write_text(
        json.dumps(analysis, indent=2, default=str),
        encoding="utf-8",
    )

    print("=" * 60)
    print("PHASE 2 PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Samples:              {stats.num_samples:,}")
    print(f"  Turning movements:    {turning_index.num_turning_movements}")
    print(f"  A_turn shape:         {a_turn_prob.shape}")
    print(f"  Binary rank:          {rank_comparison.binary.result.rank}")
    print(f"  Probabilistic rank:   {rank_comparison.probabilistic.result.rank}")
    print(f"  Rank improvement:     +{rank_comparison.rank_improvement}")
    print(f"  Total time:           {timings['total']:.2f}s")
    print(f"  Outputs:              {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probabilistic OD → turning-count generation (Phase 2)."
    )
    parser.add_argument(
        "--network",
        type=Path,
        default=PROJECT_ROOT / "sioux-falls.net.xml",
        help="Path to SUMO network file",
    )
    parser.add_argument(
        "--od",
        type=Path,
        default=OUTPUT_DIR / "od_generator" / "synthetic_od.npy",
        help="Path to synthetic OD matrices (N, 24, 24)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR / "turning_counts",
        help="Directory for outputs",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=REPORT_DIR,
        help="Directory for markdown report",
    )
    parser.add_argument("--k-paths", "--k_paths", type=int, default=5, dest="k_paths", help="K shortest paths")
    parser.add_argument("--theta", type=float, default=0.1, help="Logit sensitivity")
    parser.add_argument(
        "--weight-metric",
        choices=["length", "time"],
        default="length",
    )
    parser.add_argument(
        "--noise",
        choices=["none", "gaussian", "poisson"],
        default="poisson",
        help="Observation noise model",
    )
    parser.add_argument(
        "--noise-level",
        type=float,
        default=0.0,
        help="Nominal noise level percent (0, 2, 5, 10)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit OD matrices (for testing)",
    )
    parser.add_argument(
        "--run-sensitivity",
        action="store_true",
        help="Run K/theta sensitivity grid (slow)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size for large datasets (default {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    _configure_logging(args.verbose)
    config = TurningPipelineConfig(
        k_paths=args.k_paths,
        theta=args.theta,
        weight_metric=args.weight_metric,
        noise_model=args.noise,
        noise_level_pct=args.noise_level,
        seed=args.seed,
        run_sensitivity=args.run_sensitivity,
    )
    run_turning_pipeline(
        net_path=args.network,
        od_path=args.od,
        output_dir=args.output_dir,
        report_dir=args.report_dir,
        config=config,
        max_samples=args.max_samples,
        skip_plots=args.skip_plots,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
