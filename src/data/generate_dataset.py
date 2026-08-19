#!/usr/bin/env python3
"""End-to-end OD → turning-count dataset generator for Sioux Falls."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

# Ensure project root is importable when run as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import DATA_DIR, OUTPUT_DIR, REPORT_DIR
from src.data.build_assignment_matrix import build_assignment_matrix
from src.data.network_parser import parse_sumo_network
from src.data.od_pairs import OdPairIndex
from src.data.rank_analysis import analyze_rank
from src.data.route_assignment import assign_routes
from src.data.statistics import (
    build_dataset,
    correlation_summary,
    create_visualizations,
    diversity_comparison,
    od_statistics,
    pca_explained_variance,
    turning_count_statistics,
    validate_dataset,
)
from src.data.turning_movements import enumerate_turning_movements

logger = logging.getLogger(__name__)


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def _write_report(
    report_path: Path,
    rank_result,
    validation: dict,
    turning_stats,
    od_stats,
    diversity: dict,
    x_corr: dict,
    y_corr: dict,
    pca_x,
    pca_y,
    timings: dict,
    num_samples: int,
    num_turns: int,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Sioux Falls OD → Turning Count Dataset Summary",
        "",
        "## Dataset",
        f"- Samples (N): **{num_samples:,}**",
        f"- Turning movements: **{num_turns}**",
        f"- OD pairs: **576** (24×24)",
        f"- X shape: `(N, {num_turns})`",
        f"- Y shape: `(N, 576)`",
        "",
        "## Assignment Matrix A_turn",
        f"- Shape: `{rank_result.shape}`",
        f"- Rank: **{rank_result.rank}**",
        f"- Nullity (576 − rank): **{rank_result.nullity}**",
        f"- Condition number: **{rank_result.condition_number:.4g}**",
        "",
        "> Nullity indicates the dimension of the OD solution space not constrained "
        "by turning counts alone — relevant for OD identifiability.",
        "",
        "## Validation",
    ]
    for key, value in validation.items():
        lines.append(f"- {key}: `{value}`")

    lines.extend(
        [
            "",
            "## Turning Count Statistics (summary)",
            f"- Mean of per-turn means: {turning_stats['mean'].mean():.4f}",
            f"- Mean CV: {turning_stats['cv'].mean():.4f}",
            f"- Global min / max: {turning_stats['min'].min():.4f} / {turning_stats['max'].max():.4f}",
            "",
            "## OD Statistics (summary)",
            f"- Mean of per-pair means: {od_stats['mean'].mean():.4f}",
            f"- Mean CV: {od_stats['cv'].mean():.4f}",
            "",
            "## Diversity (sampled pairwise L1 distance)",
            f"- OD diversity: {diversity['mean_pairwise_od_distance']:.4f}",
            f"- Turning diversity: {diversity['mean_pairwise_turning_distance']:.4f}",
            "",
            "## Correlation (first 50 features)",
            f"- Turning counts — mean: {x_corr['mean_correlation']:.4f}, "
            f"min: {x_corr['min_correlation']:.4f}, max: {x_corr['max_correlation']:.4f}",
            f"- OD targets — mean: {y_corr['mean_correlation']:.4f}, "
            f"min: {y_corr['min_correlation']:.4f}, max: {y_corr['max_correlation']:.4f}",
            "",
            "## PCA Explained Variance (first components)",
            f"- Turning: {', '.join(f'{v:.3f}' for v in pca_x[:5])}",
            f"- OD: {', '.join(f'{v:.3f}' for v in pca_y[:5])}",
            "",
            "## Timings (seconds)",
        ]
    )
    for step, seconds in timings.items():
        lines.append(f"- {step}: {seconds:.2f}s")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote report to %s", report_path)


def run_pipeline(
    net_path: Path,
    od_path: Path,
    output_dir: Path,
    report_dir: Path,
    weight_metric: str = "length",
    max_samples: int | None = None,
    skip_plots: bool = False,
) -> None:
    """Execute the full OD → turning-count pipeline."""
    timings: dict[str, float] = {}
    t0 = time.perf_counter()

    # --- Network & static structures ---
    network = parse_sumo_network(net_path)
    turning_index = enumerate_turning_movements(network)
    od_index = OdPairIndex.build()
    routes = assign_routes(network, od_index, weight_metric=weight_metric)  # type: ignore[arg-type]
    a_turn = build_assignment_matrix(turning_index, od_index, routes)
    rank_result = analyze_rank(a_turn)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "A_turn.npy", a_turn)

    timings["network_and_assignment"] = time.perf_counter() - t0

    # --- Load synthetic OD data ---
    t1 = time.perf_counter()
    logger.info("Loading synthetic OD matrices from %s", od_path)
    od_matrices = np.load(od_path)
    if max_samples is not None:
        od_matrices = od_matrices[:max_samples]
    logger.info("Loaded %d OD matrices with shape %s", od_matrices.shape[0], od_matrices.shape[1:])
    timings["load_od"] = time.perf_counter() - t1

    # --- Generate dataset ---
    t2 = time.perf_counter()
    dataset = build_dataset(od_matrices, a_turn)
    timings["generate_dataset"] = time.perf_counter() - t2

    # --- Save outputs ---
    t3 = time.perf_counter()
    np.save(output_dir / "turning_counts.npy", dataset.x)
    np.save(output_dir / "od_targets.npy", dataset.y)
    np.savez_compressed(
        output_dir / "dataset.npz",
        X=dataset.x.astype(np.float32),
        Y=dataset.y.astype(np.float32),
        A_turn=a_turn.astype(np.float32),
    )
    timings["save_outputs"] = time.perf_counter() - t3

    # --- Validation ---
    t4 = time.perf_counter()
    validation = validate_dataset(dataset)
    timings["validation"] = time.perf_counter() - t4

    # --- Statistics ---
    t5 = time.perf_counter()
    turning_stats = turning_count_statistics(dataset.x)
    od_stats = od_statistics(dataset.y)
    diversity = diversity_comparison(dataset.x, dataset.y)
    x_corr = correlation_summary(dataset.x)
    y_corr = correlation_summary(dataset.y)
    pca_x = pca_explained_variance(dataset.x)
    pca_y = pca_explained_variance(dataset.y)
    turning_stats.to_csv(output_dir / "turning_statistics.csv", index_label="turn_id")
    od_stats.to_csv(output_dir / "od_statistics.csv", index_label="od_index")
    timings["statistics"] = time.perf_counter() - t5

    # --- Visualizations ---
    t6 = time.perf_counter()
    if not skip_plots:
        create_visualizations(
            dataset,
            output_dir / "figures",
            turning_stats,
            od_stats,
        )
    timings["visualization"] = time.perf_counter() - t6

    timings["total"] = time.perf_counter() - t0

    _write_report(
        report_dir / "dataset_summary.md",
        rank_result,
        validation,
        turning_stats,
        od_stats,
        diversity,
        x_corr,
        y_corr,
        pca_x,
        pca_y,
        timings,
        dataset.num_samples,
        turning_index.num_turning_movements,
    )

    summary = {
        "num_samples": dataset.num_samples,
        "num_turning_movements": turning_index.num_turning_movements,
        "a_turn_shape": list(a_turn.shape),
        "rank": rank_result.rank,
        "nullity": rank_result.nullity,
        "condition_number": rank_result.condition_number,
        "validation": validation,
        "diversity": diversity,
        "timings": timings,
    }
    (output_dir / "pipeline_summary.json").write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )

    print("=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Samples:           {dataset.num_samples:,}")
    print(f"  X shape:           {dataset.x.shape}")
    print(f"  Y shape:           {dataset.y.shape}")
    print(f"  A_turn shape:      {a_turn.shape}")
    print(f"  Rank / nullity:    {rank_result.rank} / {rank_result.nullity}")
    print(f"  Total time:        {timings['total']:.2f}s")
    print(f"  Outputs:           {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate OD → turning-count ML dataset for Sioux Falls."
    )
    parser.add_argument(
        "--net",
        type=Path,
        default=DATA_DIR / "sioux-falls.net.xml",
        help="Path to SUMO network file",
    )
    parser.add_argument(
        "--od",
        type=Path,
        default=DATA_DIR / "synthetic_od_100000.npy",
        help="Path to synthetic OD matrices (N, 24, 24)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for outputs",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=REPORT_DIR,
        help="Directory for markdown report",
    )
    parser.add_argument(
        "--weight-metric",
        choices=["length", "time"],
        default="length",
        help="Shortest-path edge weight",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of OD matrices (for testing)",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip figure generation",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    _configure_logging(args.verbose)
    run_pipeline(
        net_path=args.net,
        od_path=args.od,
        output_dir=args.output_dir,
        report_dir=args.report_dir,
        weight_metric=args.weight_metric,
        max_samples=args.max_samples,
        skip_plots=args.skip_plots,
    )


if __name__ == "__main__":
    main()
