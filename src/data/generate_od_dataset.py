#!/usr/bin/env python3
"""CLI for Phase 1: robust synthetic OD matrix generation (optimized for large N)."""

from __future__ import annotations

from dataclasses import asdict
import argparse
import json
import logging
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_evaluation import (
    create_evaluation_figures,
    evaluate_dataset,
    save_dataset_metrics,
    write_dataset_summary,
)
from src.data.hyperparameter_search import (
    run_hyperparameter_search,
    write_hyperparameter_report,
)
from src.data.od_analysis import analyze_base_od, write_analysis_report
from src.data.od_generator import GeneratorConfig, generate_synthetic_od_batch
from src.data.od_metrics import compute_batch_metrics, compute_fast_aggregate_metrics
from src.data.quality_filters import QualityFilterConfig, filter_candidate_pool

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "od_generator"
LARGE_DATASET_THRESHOLD = 100_000


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def load_base_od(path: Path) -> np.ndarray:
    """Load and sanitize the base OD matrix."""
    od = np.load(path).astype(float)
    od = np.clip(od, 0, None)
    np.fill_diagonal(od, 0.0)
    return od


def run_pipeline(args: argparse.Namespace) -> None:
    """Execute the full synthetic OD generation pipeline."""
    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    timings: dict[str, float] = {}
    t0 = time.perf_counter()

    base_od = load_base_od(Path(args.input))
    analysis = analyze_base_od(base_od)
    if not args.skip_base_analysis:
        write_analysis_report(analysis, output_dir)
    timings["base_analysis"] = time.perf_counter() - t0

    alpha = args.alpha
    perturbation = args.perturbation
    workers = args.workers if args.workers is not None else mp.cpu_count()

    if args.hyperparameter_search:
        t_hp = time.perf_counter()
        logger.info("Running hyperparameter search...")
        hp_results = run_hyperparameter_search(
            base_od,
            samples_per_config=args.search_samples,
            seed=args.seed,
            workers=workers,
        )
        recommended = write_hyperparameter_report(hp_results, output_dir)
        alpha = recommended.alpha
        perturbation = recommended.perturbation
        timings["hyperparameter_search"] = time.perf_counter() - t_hp

    config = GeneratorConfig(
        alpha=alpha,
        perturbation=perturbation,
        ipf_tol=args.ipf_tol,
        ipf_max_iter=args.ipf_max_iter,
        epsilon_beta=args.epsilon_beta,
        epsilon_lambda=args.epsilon_lambda,
        gamma_shape=args.gamma_shape,
        gamma_scale=args.gamma_scale,
        apply_gamma_mask=not args.no_gamma_mask,
    )

    dtype = np.float32 if args.float32 else np.float64
    store_ipf_stats = args.store_ipf_stats or args.samples <= LARGE_DATASET_THRESHOLD

    use_filters = not args.no_quality_filters
    filter_cfg = QualityFilterConfig(
        oversample_factor=args.oversample_factor,
        min_correlation=args.min_correlation,
        max_correlation=args.max_correlation,
        max_sparsity=args.max_sparsity,
        max_trip_length_rel_error=args.max_trip_length_rel_error,
        min_frobenius_distance=args.min_frobenius_distance,
        auto_frobenius_fraction=args.auto_frobenius_fraction,
    )
    n_candidates = (
        int(np.ceil(args.samples * filter_cfg.oversample_factor))
        if use_filters
        else args.samples
    )

    t_gen = time.perf_counter()
    logger.info(
        "Generating %d candidate OD matrices → target %d "
        "(alpha=%.0f, perturbation=%.2f, epsilon_beta=%.4f, epsilon_lambda=%.1f, "
        "gamma_shape=%.3f, gamma_scale=%.3f, gamma_mask=%s, quality_filters=%s, "
        "workers=%d, dtype=%s)...",
        n_candidates,
        args.samples,
        config.alpha,
        config.perturbation,
        config.epsilon_beta,
        config.epsilon_lambda,
        config.gamma_shape,
        config.gamma_scale,
        config.apply_gamma_mask,
        use_filters,
        workers,
        dtype,
    )
    batch_result = generate_synthetic_od_batch(
        base_od,
        n_candidates,
        config,
        seed=args.seed,
        workers=workers,
        dtype=dtype,
        store_ipf_stats=store_ipf_stats and not use_filters,
        show_progress=not args.no_progress,
    )
    candidates = batch_result.matrices
    timings["generation"] = time.perf_counter() - t_gen

    filter_stats = None
    if use_filters:
        t_filt = time.perf_counter()
        synthetic_batch, filter_stats = filter_candidate_pool(
            candidates,
            base_od,
            target_size=args.samples,
            config=filter_cfg,
            seed=args.seed,
        )
        timings["quality_filters"] = time.perf_counter() - t_filt
        if synthetic_batch.shape[0] == 0:
            raise RuntimeError(
                "Quality filters rejected all candidates. "
                "Relax thresholds or increase --oversample-factor."
            )
        if synthetic_batch.shape[0] < args.samples:
            logger.warning(
                "Only accepted %d / %d requested matrices",
                synthetic_batch.shape[0],
                args.samples,
            )
        # Free candidate pool memory
        del candidates
        batch_result = None
    else:
        synthetic_batch = candidates
        timings["quality_filters"] = 0.0

    t_save = time.perf_counter()
    od_output = output_dir / args.output_name
    if args.memmap_save:
        mm = np.lib.format.open_memmap(
            od_output, mode="w+", dtype=dtype, shape=synthetic_batch.shape
        )
        mm[:] = synthetic_batch
        del mm
    else:
        np.save(od_output, synthetic_batch)
    timings["save"] = time.perf_counter() - t_save
    logger.info("Saved %s with shape %s", od_output, synthetic_batch.shape)

    if filter_stats is not None:
        (output_dir / "quality_filter_stats.json").write_text(
            json.dumps(filter_stats.to_dict(), indent=2),
            encoding="utf-8",
        )

    run_full_eval = args.full_evaluation or synthetic_batch.shape[0] <= LARGE_DATASET_THRESHOLD
    skip_plots = args.skip_plots or synthetic_batch.shape[0] > LARGE_DATASET_THRESHOLD

    t_eval = time.perf_counter()
    eval_sample = args.eval_samples
    fast_agg = compute_fast_aggregate_metrics(
        base_od, synthetic_batch, sample_size=eval_sample, seed=args.seed
    )

    matrix_metrics = []
    if run_full_eval:
        matrix_metrics = compute_batch_metrics(
            base_od,
            synthetic_batch,
            batch_result=batch_result if store_ipf_stats and not use_filters else None,
            sample_size=eval_sample,
            seed=args.seed,
        )

    evaluation = evaluate_dataset(
        base_od,
        synthetic_batch,
        matrix_metrics=matrix_metrics if matrix_metrics else None,
        diversity_samples=args.diversity_samples,
        eval_sample_size=eval_sample,
        fast_aggregate=fast_agg,
        seed=args.seed,
    )

    save_dataset_metrics(
        evaluation,
        matrix_metrics if matrix_metrics else [],
        output_dir / "dataset_metrics.json",
    )
    config_dict = {
        "input": str(args.input),
        "samples": int(synthetic_batch.shape[0]),
        "requested_samples": args.samples,
        "n_candidates": n_candidates,
        "quality_filters": use_filters,
        "alpha": config.alpha,
        "perturbation": config.perturbation,
        "epsilon_beta": config.epsilon_beta,
        "epsilon_lambda": config.epsilon_lambda,
        "gamma_shape": config.gamma_shape,
        "gamma_scale": config.gamma_scale,
        "apply_gamma_mask": config.apply_gamma_mask,
        "ipf_tol": config.ipf_tol,
        "seed": args.seed,
        "workers": workers,
        "dtype": str(dtype),
        "eval_sample_size": eval_sample,
        "filter_config": asdict(filter_cfg) if use_filters else None,
        "filter_stats": filter_stats.to_dict() if filter_stats is not None else None,
    }
    write_dataset_summary(
        evaluation,
        config_dict,
        output_dir / "dataset_summary.md",
    )

    if not skip_plots:
        if matrix_metrics:
            create_evaluation_figures(
                base_od,
                synthetic_batch[: min(50_000, len(synthetic_batch))],
                matrix_metrics,
                evaluation,
                figures_dir,
            )
        else:
            logger.info("Skipping plots: no per-matrix metrics collected")

    timings["evaluation"] = time.perf_counter() - t_eval
    timings["total"] = time.perf_counter() - t0

    summary = {
        "output_file": str(od_output),
        "shape": list(synthetic_batch.shape),
        "config": config_dict,
        "fast_aggregate_metrics": fast_agg,
        "evaluation": evaluation.to_dict(),
        "timings": timings,
    }
    (output_dir / "pipeline_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("=" * 60)
    print("SYNTHETIC OD GENERATION COMPLETE")
    print("=" * 60)
    print(f"  Output:      {od_output}")
    print(f"  Shape:       {synthetic_batch.shape}")
    print(f"  alpha:       {config.alpha}")
    print(f"  perturbation:{config.perturbation}")
    print(f"  eps beta:    {config.epsilon_beta}")
    print(f"  eps lambda:  {config.epsilon_lambda}")
    print(f"  gamma mask:  {config.apply_gamma_mask} (shape={config.gamma_shape}, scale={config.gamma_scale})")
    print(f"  filters:     {use_filters}")
    if filter_stats is not None:
        print(
            f"  accepted:    {filter_stats.n_accepted}/{filter_stats.n_candidates} "
            f"(H̄={filter_stats.mean_cell_entropy:.3f}, "
            f"CV̄={filter_stats.mean_cell_cv:.3f}, "
            f"F̄={filter_stats.mean_pairwise_frobenius:.1f})"
        )
    print(f"  workers:     {workers}")
    print(f"  Correlation: {fast_agg.get('mean_correlation', float('nan')):.4f}")
    print(f"  Diversity:   {evaluation.mean_pairwise_distance:.2f}")
    print(f"  Demand CV:   {evaluation.demand_cv:.2e}")
    print(f"  Generation:  {timings['generation']:.2f}s")
    if use_filters:
        print(f"  Filters:     {timings.get('quality_filters', 0.0):.2f}s")
    print(f"  Save:        {timings['save']:.2f}s")
    print(f"  Total time:  {timings['total']:.2f}s")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate synthetic OD matrices via Dirichlet sampling + IPF."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(PROJECT_ROOT / "EstimatedODMatrix.npy"),
        help="Path to base OD matrix (.npy)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5000,
        help="Number of synthetic matrices to generate",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=500.0,
        help="Dirichlet concentration multiplier",
    )
    parser.add_argument(
        "--perturbation",
        type=float,
        default=0.20,
        help="Marginal perturbation level in [0,1)",
    )
    parser.add_argument(
        "--epsilon-beta",
        type=float,
        default=1.0,
        help="Distance-dependent prior scale β in ε_ij = β exp(-d_ij / λ)",
    )
    parser.add_argument(
        "--epsilon-lambda",
        type=float,
        default=500.0,
        help="Distance decay λ (meters) in ε_ij = β exp(-d_ij / λ)",
    )
    parser.add_argument(
        "--gamma-shape",
        type=float,
        default=0.5,
        help="Gamma shape for sparse mask (shape<1 → many tiny / few large flows)",
    )
    parser.add_argument(
        "--gamma-scale",
        type=float,
        default=1.0,
        help="Gamma scale for sparse mask weights",
    )
    parser.add_argument(
        "--no-gamma-mask",
        action="store_true",
        help="Disable Gamma sparse reweighting after Dirichlet sampling",
    )
    parser.add_argument(
        "--no-quality-filters",
        action="store_true",
        help="Skip quality filters (keep all generated candidates)",
    )
    parser.add_argument(
        "--oversample-factor",
        type=float,
        default=10.0,
        help="Generate this many × target samples as candidate pool",
    )
    parser.add_argument(
        "--min-correlation",
        type=float,
        default=0.30,
        help="Min Pearson correlation with base OD",
    )
    parser.add_argument(
        "--max-correlation",
        type=float,
        default=0.95,
        help="Max Pearson correlation with base OD",
    )
    parser.add_argument(
        "--max-sparsity",
        type=float,
        default=0.95,
        help="Max fraction of zero cells",
    )
    parser.add_argument(
        "--max-trip-length-rel-error",
        type=float,
        default=0.35,
        help="Max relative error of flow-weighted mean trip length vs base",
    )
    parser.add_argument(
        "--min-frobenius-distance",
        type=float,
        default=0.0,
        help="Min pairwise Frobenius distance (0 = auto from candidate median)",
    )
    parser.add_argument(
        "--auto-frobenius-fraction",
        type=float,
        default=0.35,
        help="Auto threshold = fraction × median pairwise Frobenius",
    )
    parser.add_argument(
        "--ipf-tol",
        type=float,
        default=1e-3,
        help="IPF convergence tolerance",
    )
    parser.add_argument(
        "--ipf-max-iter",
        type=int,
        default=1000,
        help="Maximum IPF iterations",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel worker processes (default: all CPU cores)",
    )
    parser.add_argument(
        "--float32",
        dest="float32",
        action="store_true",
        default=True,
        help="Save matrices as float32 (default, ~50%% less disk/RAM)",
    )
    parser.add_argument(
        "--float64",
        dest="float32",
        action="store_false",
        help="Save matrices as float64 instead of float32",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="synthetic_od.npy",
        help="Filename for synthetic OD array",
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=10_000,
        help="Random sample size for quality metrics / PCA / KMeans",
    )
    parser.add_argument(
        "--diversity-samples",
        type=int,
        default=10_000,
        help="Random pairs for diversity estimate",
    )
    parser.add_argument(
        "--store-ipf-stats",
        action="store_true",
        help="Store per-sample IPF iteration counts (uses extra memory at large N)",
    )
    parser.add_argument(
        "--full-evaluation",
        action="store_true",
        help="Run full per-matrix evaluation even for large N",
    )
    parser.add_argument(
        "--memmap-save",
        action="store_true",
        help="Write output .npy via memmap",
    )
    parser.add_argument(
        "--hyperparameter-search",
        action="store_true",
        help="Run grid search over alpha and perturbation before generation",
    )
    parser.add_argument(
        "--search-samples",
        type=int,
        default=200,
        help="Samples per config during hyperparameter search",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip figure generation",
    )
    parser.add_argument(
        "--skip-base-analysis",
        action="store_true",
        help="Skip base OD analysis reports",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _configure_logging(args.verbose)
    run_pipeline(args)


if __name__ == "__main__":
    main()
