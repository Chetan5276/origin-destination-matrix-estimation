#!/usr/bin/env python3
"""CLI: first-principles synthetic OD generation (no historical base OD)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import NETWORK_PATH, NUM_ZONES, OUTPUT_DIR
from src.data.first_principles import (
    FPGeneratorConfig,
    build_reference_od,
    generate_fp_od_batch,
)
from src.data.quality_filters import QualityFilterConfig, filter_candidate_pool

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = OUTPUT_DIR / "od_generator_fp"
LARGE_DATASET_THRESHOLD = 50_000


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="First-principles OD generation: latent gravity → Gamma → Dirichlet → IPF."
    )
    p.add_argument("--samples", type=int, default=5000)
    p.add_argument(
        "--total-demand",
        type=float,
        default=365_475.0,
        help="Grand total trips G (default ≈ Sioux Falls base total)",
    )
    p.add_argument("--latent-dim", type=int, default=4)
    p.add_argument("--factor-shape", type=float, default=2.0)
    p.add_argument("--factor-scale", type=float, default=1.0)
    p.add_argument("--spatial-smooth-length", type=float, default=400.0)
    p.add_argument("--spatial-smooth-strength", type=float, default=0.5)
    p.add_argument("--decay", choices=["exponential", "power"], default="exponential")
    p.add_argument("--lambda-decay", type=float, default=500.0)
    p.add_argument("--power-gamma", type=float, default=1.5)
    p.add_argument("--reciprocity", type=float, default=0.35)
    p.add_argument("--gamma-shape", type=float, default=0.5)
    p.add_argument("--gamma-scale", type=float, default=1.0)
    p.add_argument("--no-gamma-mask", action="store_true")
    p.add_argument("--alpha", type=float, default=200.0, help="Dirichlet concentration")
    p.add_argument("--equal-marginals", action="store_true", help="Use equal P/A instead of latent")
    p.add_argument("--ipf-tol", type=float, default=1e-3)
    p.add_argument("--ipf-max-iter", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--float64", dest="float32", action="store_false")
    p.add_argument("--float32", dest="float32", action="store_true", default=True)
    p.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--output-name", type=str, default="synthetic_od_fp.npy")
    p.add_argument("--memmap-save", action="store_true")
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--no-quality-filters", action="store_true")
    p.add_argument("--oversample-factor", type=float, default=2.0)
    p.add_argument("--min-correlation", type=float, default=-1.0,
                   help="vs reference gravity OD; default disables corr filter")
    p.add_argument("--max-correlation", type=float, default=1.0)
    p.add_argument("--max-sparsity", type=float, default=0.95)
    p.add_argument("--max-trip-length-rel-error", type=float, default=0.50)
    p.add_argument("--min-frobenius-distance", type=float, default=0.0)
    p.add_argument("--auto-frobenius-fraction", type=float, default=0.25)
    p.add_argument("--network", type=str, default=str(NETWORK_PATH))
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    _configure_logging(args.verbose)
    t0 = time.perf_counter()
    timings: dict[str, float] = {}

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = FPGeneratorConfig(
        total_demand=args.total_demand,
        num_zones=NUM_ZONES,
        latent_dim=args.latent_dim,
        factor_shape=args.factor_shape,
        factor_scale=args.factor_scale,
        spatial_smooth_length=args.spatial_smooth_length,
        spatial_smooth_strength=args.spatial_smooth_strength,
        decay=args.decay,
        lambda_decay=args.lambda_decay,
        power_gamma=args.power_gamma,
        reciprocity=args.reciprocity,
        gamma_shape=args.gamma_shape,
        gamma_scale=args.gamma_scale,
        apply_gamma_mask=not args.no_gamma_mask,
        alpha=args.alpha,
        ipf_tol=args.ipf_tol,
        ipf_max_iter=args.ipf_max_iter,
        latent_marginals=not args.equal_marginals,
    )

    dtype = np.float32 if args.float32 else np.float64
    use_filters = not args.no_quality_filters
    n_candidates = (
        int(np.ceil(args.samples * args.oversample_factor)) if use_filters else args.samples
    )
    workers = args.workers

    logger.info(
        "FP OD generation: candidates=%d → target=%d | latent_dim=%d decay=%s "
        "λ=%.1f reciprocity=%.2f gamma=%s α=%.0f G=%.1f",
        n_candidates,
        args.samples,
        config.latent_dim,
        config.decay,
        config.lambda_decay,
        config.reciprocity,
        config.apply_gamma_mask,
        config.alpha,
        config.total_demand,
    )

    t_gen = time.perf_counter()
    batch = generate_fp_od_batch(
        n_candidates,
        config,
        seed=args.seed,
        workers=workers,
        dtype=dtype,
        store_ipf_stats=n_candidates <= LARGE_DATASET_THRESHOLD,
        show_progress=not args.no_progress,
        network_path=Path(args.network),
    )
    candidates = batch.matrices
    timings["generation"] = time.perf_counter() - t_gen

    # Reference OD for trip-length / optional correlation filters (not a survey matrix)
    reference = build_reference_od(
        config, network_path=Path(args.network), seed=args.seed
    ).astype(np.float64)

    filter_stats = None
    if use_filters:
        t_filt = time.perf_counter()
        filter_cfg = QualityFilterConfig(
            oversample_factor=args.oversample_factor,
            min_correlation=args.min_correlation,
            max_correlation=args.max_correlation,
            max_sparsity=args.max_sparsity,
            max_trip_length_rel_error=args.max_trip_length_rel_error,
            min_frobenius_distance=args.min_frobenius_distance,
            auto_frobenius_fraction=args.auto_frobenius_fraction,
        )
        synthetic, filter_stats = filter_candidate_pool(
            candidates,
            reference,
            target_size=args.samples,
            config=filter_cfg,
            seed=args.seed,
        )
        timings["quality_filters"] = time.perf_counter() - t_filt
        del candidates
        if synthetic.shape[0] == 0:
            raise RuntimeError("Quality filters rejected all FP candidates.")
        if synthetic.shape[0] < args.samples:
            logger.warning(
                "Accepted %d / %d requested", synthetic.shape[0], args.samples
            )
    else:
        synthetic = candidates
        timings["quality_filters"] = 0.0

    t_save = time.perf_counter()
    od_output = output_dir / args.output_name
    if args.memmap_save:
        mm = np.lib.format.open_memmap(
            od_output, mode="w+", dtype=dtype, shape=synthetic.shape
        )
        mm[:] = synthetic
        del mm
    else:
        np.save(od_output, synthetic)
    timings["save"] = time.perf_counter() - t_save
    logger.info("Saved %s shape=%s", od_output, synthetic.shape)

    # Lightweight summary metrics (no dependence on survey base OD)
    flat = synthetic.reshape(synthetic.shape[0], -1).astype(np.float64)
    totals = flat.sum(axis=1)
    # Sampled pairwise diversity
    rng = np.random.default_rng(args.seed)
    n_pairs = min(2000, synthetic.shape[0] * (synthetic.shape[0] - 1) // 2)
    i = rng.integers(0, synthetic.shape[0], size=n_pairs)
    j = rng.integers(0, synthetic.shape[0], size=n_pairs)
    same = i == j
    while same.any():
        j[same] = rng.integers(0, synthetic.shape[0], size=int(same.sum()))
        same = i == j
    pair_l1 = np.mean(np.abs(flat[i] - flat[j]).sum(axis=1))

    summary = {
        "pipeline": "first_principles",
        "output_file": str(od_output),
        "shape": list(synthetic.shape),
        "config": asdict(config),
        "n_candidates": n_candidates,
        "n_accepted": int(synthetic.shape[0]),
        "quality_filters": use_filters,
        "filter_stats": filter_stats.to_dict() if filter_stats is not None else None,
        "demand_mean": float(totals.mean()),
        "demand_cv": float(totals.std() / (totals.mean() + 1e-15)),
        "mean_pairwise_l1": float(pair_l1),
        "zero_diag": bool(np.allclose(np.einsum("kii->k", synthetic), 0)),
        "timings": timings,
        "seed": args.seed,
        "network": args.network,
    }
    timings["total"] = time.perf_counter() - t0
    summary["timings"] = timings

    (output_dir / "pipeline_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    if filter_stats is not None:
        (output_dir / "quality_filter_stats.json").write_text(
            json.dumps(filter_stats.to_dict(), indent=2), encoding="utf-8"
        )
    np.save(output_dir / "reference_od.npy", reference.astype(np.float32))

    md = [
        "# First-Principles OD Generation Summary",
        "",
        f"- Output: `{od_output}`",
        f"- Shape: `{synthetic.shape}`",
        f"- Total demand G: `{config.total_demand}` (CV={summary['demand_cv']:.2e})",
        f"- Latent dim: `{config.latent_dim}`, decay: `{config.decay}`, reciprocity: `{config.reciprocity}`",
        f"- Gamma mask: `{config.apply_gamma_mask}`, Dirichlet α: `{config.alpha}`",
        f"- Mean pairwise L1: `{pair_l1:.2f}`",
        f"- Total time: `{timings['total']:.2f}s`",
        "",
        "Next: generate turning counts with the existing Phase-2 pipeline, e.g.",
        "```",
        f"python -m src.data.generate_turning_counts --network {args.network} \\",
        f"  --od {od_output} --output-dir outputs/turning_counts_fp",
        "```",
        "",
    ]
    (output_dir / "dataset_summary.md").write_text("\n".join(md), encoding="utf-8")

    print("=" * 60)
    print("FIRST-PRINCIPLES OD GENERATION COMPLETE")
    print("=" * 60)
    print(f"  Output:   {od_output}")
    print(f"  Shape:    {synthetic.shape}")
    print(f"  Demand CV:{summary['demand_cv']:.2e}")
    print(f"  Diversity:{pair_l1:.2f}")
    print(f"  Time:     {timings['total']:.2f}s")


if __name__ == "__main__":
    main()
