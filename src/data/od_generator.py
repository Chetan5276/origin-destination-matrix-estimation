"""Stages 4--8: high-performance synthetic OD matrix generation."""

from __future__ import annotations

import logging
import multiprocessing as mp
from dataclasses import dataclass

import numpy as np

from src.data.dirichlet_sampler import sample_demand_matrix
from src.data.ipf import IpfResult, ipf
from src.data.marginal_perturbation import PerturbedMarginals, perturb_marginals
from src.data.od_probability import (
    DEFAULT_EPSILON_BETA,
    DEFAULT_EPSILON_LAMBDA,
    BaseProbabilityDistribution,
    build_base_probability,
)
from src.data.sparse_mask import DEFAULT_GAMMA_SCALE, DEFAULT_GAMMA_SHAPE

logger = logging.getLogger(__name__)

DEFAULT_SEED_FLOOR = 1e-6

# Worker globals for multiprocessing (initialized once per process).
_WORKER: dict | None = None


@dataclass(frozen=True)
class GeneratorConfig:
    """Hyperparameters for synthetic OD generation."""

    alpha: float = 500.0
    perturbation: float = 0.20
    ipf_tol: float = 1e-3
    ipf_max_iter: int = 1000
    seed_floor: float = DEFAULT_SEED_FLOOR
    # Distance-dependent prior: ε_ij = β exp(-d_ij / λ)
    epsilon_beta: float = DEFAULT_EPSILON_BETA
    epsilon_lambda: float = DEFAULT_EPSILON_LAMBDA
    # Heavy-tailed Gamma sparse mask after Dirichlet
    gamma_shape: float = DEFAULT_GAMMA_SHAPE
    gamma_scale: float = DEFAULT_GAMMA_SCALE
    apply_gamma_mask: bool = True


@dataclass(frozen=True)
class GenerationResult:
    """Output of generating one synthetic OD matrix (detailed / single-sample mode)."""

    matrix: np.ndarray
    ipf_result: IpfResult
    marginals: PerturbedMarginals
    alpha: float


@dataclass(frozen=True)
class BatchGenerationResult:
    """Output of a large batch run."""

    matrices: np.ndarray
    ipf_iterations: np.ndarray | None = None
    ipf_final_errors: np.ndarray | None = None


def apply_sparsity_mask_inplace(matrix: np.ndarray, support_mask: np.ndarray) -> None:
    """Zero all cells outside the base support in-place."""
    matrix[~support_mask] = 0.0


def enforce_zero_diagonal_inplace(matrix: np.ndarray) -> None:
    """Force intrazonal flows to zero in-place."""
    np.fill_diagonal(matrix, 0.0)


def apply_sparsity_mask(matrix: np.ndarray, support_mask: np.ndarray) -> np.ndarray:
    """Zero all cells outside the base support (copying variant)."""
    out = matrix.copy()
    apply_sparsity_mask_inplace(out, support_mask)
    return out


def enforce_zero_diagonal(matrix: np.ndarray) -> np.ndarray:
    """Force intrazonal flows to zero (copying variant)."""
    out = matrix.copy()
    enforce_zero_diagonal_inplace(out)
    return out


def prepare_ipf_seed_inplace(
    matrix: np.ndarray,
    support_mask: np.ndarray,
    seed_floor: float = DEFAULT_SEED_FLOOR,
) -> None:
    """Apply sparsity, zero diagonal, and IPF floor on support in-place."""
    apply_sparsity_mask_inplace(matrix, support_mask)
    enforce_zero_diagonal_inplace(matrix)
    matrix[support_mask] += seed_floor
    enforce_zero_diagonal_inplace(matrix)


def _perturb_targets_fast(
    productions: np.ndarray,
    attractions: np.ndarray,
    total_demand: float,
    perturbation: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Return IPF target row/column sums without dataclass overhead."""
    p_tilde = productions * (
        1.0 + rng.uniform(-perturbation, perturbation, size=productions.shape)
    )
    a_tilde = attractions * (
        1.0 + rng.uniform(-perturbation, perturbation, size=attractions.shape)
    )
    target_row = total_demand * p_tilde / p_tilde.sum()
    target_col = total_demand * a_tilde / a_tilde.sum()
    return target_row, target_col


def _generate_core(
    base_dist: BaseProbabilityDistribution,
    productions: np.ndarray,
    attractions: np.ndarray,
    config: GeneratorConfig,
    rng: np.random.Generator,
    seed_buf: np.ndarray,
) -> tuple[np.ndarray, IpfResult]:
    support_mask = base_dist.support_mask

    sample_demand_matrix(
        base_dist.support_indices,
        base_dist.support_probabilities,
        config.alpha,
        base_dist.total_demand,
        base_dist.num_zones,
        rng,
        out=seed_buf,
        gamma_shape=config.gamma_shape,
        gamma_scale=config.gamma_scale,
        apply_sparse_mask=config.apply_gamma_mask,
    )
    prepare_ipf_seed_inplace(seed_buf, support_mask, config.seed_floor)

    target_row, target_col = _perturb_targets_fast(
        productions,
        attractions,
        base_dist.total_demand,
        config.perturbation,
        rng,
    )

    ipf_result = ipf(
        seed_buf,
        target_row,
        target_col,
        tol=config.ipf_tol,
        max_iter=config.ipf_max_iter,
        record_history=False,
        out=seed_buf,
    )

    apply_sparsity_mask_inplace(ipf_result.matrix, support_mask)
    enforce_zero_diagonal_inplace(ipf_result.matrix)
    return ipf_result.matrix, ipf_result


def generate_one_synthetic_od(
    base_od: np.ndarray,
    base_dist: BaseProbabilityDistribution,
    config: GeneratorConfig,
    rng: np.random.Generator,
    seed_buf: np.ndarray | None = None,
) -> GenerationResult:
    """Generate a single synthetic OD matrix (detailed result for tests / debugging)."""
    if seed_buf is None:
        seed_buf = np.zeros((base_dist.num_zones, base_dist.num_zones), dtype=np.float64)

    productions = base_od.sum(axis=1)
    attractions = base_od.sum(axis=0)
    support_mask = base_dist.support_mask

    sample_demand_matrix(
        base_dist.support_indices,
        base_dist.support_probabilities,
        config.alpha,
        base_dist.total_demand,
        base_dist.num_zones,
        rng,
        out=seed_buf,
        gamma_shape=config.gamma_shape,
        gamma_scale=config.gamma_scale,
        apply_sparse_mask=config.apply_gamma_mask,
    )
    prepare_ipf_seed_inplace(seed_buf, support_mask, config.seed_floor)

    target_row, target_col = _perturb_targets_fast(
        productions,
        attractions,
        base_dist.total_demand,
        config.perturbation,
        rng,
    )

    ipf_result = ipf(
        seed_buf,
        target_row,
        target_col,
        tol=config.ipf_tol,
        max_iter=config.ipf_max_iter,
        record_history=True,
        out=seed_buf,
    )

    apply_sparsity_mask_inplace(ipf_result.matrix, support_mask)
    enforce_zero_diagonal_inplace(ipf_result.matrix)

    marginals = PerturbedMarginals(
        target_productions=target_row,
        target_attractions=target_col,
        total_demand=base_dist.total_demand,
        perturbation=config.perturbation,
    )

    return GenerationResult(
        matrix=ipf_result.matrix,
        ipf_result=ipf_result,
        marginals=marginals,
        alpha=config.alpha,
    )


def _build_dist_from_config(base_od: np.ndarray, config: GeneratorConfig) -> BaseProbabilityDistribution:
    return build_base_probability(
        base_od,
        beta=config.epsilon_beta,
        lambda_decay=config.epsilon_lambda,
    )


def _init_worker(
    base_od: np.ndarray,
    config: GeneratorConfig,
    seed_base: int,
) -> None:
    """Initialize per-process buffers and cached base statistics."""
    global _WORKER
    base_dist = _build_dist_from_config(base_od, config)
    _WORKER = {
        "base_dist": base_dist,
        "config": config,
        "seed_base": seed_base,
        "seed_buf": np.zeros((base_dist.num_zones, base_dist.num_zones), dtype=np.float64),
        "productions": base_od.sum(axis=1),
        "attractions": base_od.sum(axis=0),
    }


def _generate_worker_fixed(index: int) -> tuple[int, np.ndarray, int, float]:
    """Worker using initialized _WORKER state."""
    assert _WORKER is not None
    rng = np.random.default_rng(_WORKER["seed_base"] + index)
    matrix, ipf_result = _generate_core(
        _WORKER["base_dist"],
        _WORKER["productions"],
        _WORKER["attractions"],
        _WORKER["config"],
        rng,
        _WORKER["seed_buf"],
    )
    return index, matrix.copy(), ipf_result.iterations, ipf_result.final_error


def generate_synthetic_od_batch(
    base_od: np.ndarray,
    num_samples: int,
    config: GeneratorConfig,
    seed: int = 42,
    workers: int | None = None,
    dtype: np.dtype = np.float32,
    store_ipf_stats: bool = False,
    show_progress: bool = True,
) -> BatchGenerationResult:
    """
    Generate ``num_samples`` synthetic OD matrices.

    Uses multiprocessing when ``workers > 1``. Output dtype defaults to
    float32 (~2.3 GB for 1M samples vs ~4.6 GB for float64).
    """
    base_od = np.asarray(base_od, dtype=np.float64)
    np.fill_diagonal(base_od, 0.0)

    n_zones = base_od.shape[0]
    matrices = np.zeros((num_samples, n_zones, n_zones), dtype=dtype)

    n_workers = workers if workers is not None else mp.cpu_count()
    n_workers = max(1, min(n_workers, num_samples))

    ipf_iterations = np.zeros(num_samples, dtype=np.int16) if store_ipf_stats else None
    ipf_final_errors = np.zeros(num_samples, dtype=np.float32) if store_ipf_stats else None

    if n_workers == 1:
        base_dist = _build_dist_from_config(base_od, config)
        seed_buf = np.zeros((n_zones, n_zones), dtype=np.float64)
        productions = base_od.sum(axis=1)
        attractions = base_od.sum(axis=0)
        iterator = range(num_samples)
        if show_progress and num_samples >= 10_000:
            from tqdm import tqdm

            iterator = tqdm(iterator, desc="Generating OD", unit="matrix")

        for k in iterator:
            rng = np.random.default_rng(seed + k)
            matrix, ipf_result = _generate_core(
                base_dist,
                productions,
                attractions,
                config,
                rng,
                seed_buf,
            )
            matrices[k] = matrix.astype(dtype, copy=False)
            if store_ipf_stats:
                ipf_iterations[k] = ipf_result.iterations  # type: ignore[index]
                ipf_final_errors[k] = ipf_result.final_error  # type: ignore[index]
    else:
        chunksize = max(1, num_samples // (n_workers * 8))
        logger.info(
            "Parallel generation: %d samples, %d workers, chunksize=%d",
            num_samples,
            n_workers,
            chunksize,
        )
        with mp.Pool(
            processes=n_workers,
            initializer=_init_worker,
            initargs=(base_od, config, seed),
        ) as pool:
            iterator = pool.imap(_generate_worker_fixed, range(num_samples), chunksize=chunksize)
            if show_progress:
                from tqdm import tqdm

                iterator = tqdm(
                    iterator,
                    total=num_samples,
                    desc="Generating OD",
                    unit="matrix",
                )
            for index, matrix, n_iter, final_err in iterator:
                matrices[index] = matrix.astype(dtype, copy=False)
                if store_ipf_stats:
                    ipf_iterations[index] = n_iter  # type: ignore[index]
                    ipf_final_errors[index] = final_err  # type: ignore[index]

    logger.info("Generated %d synthetic OD matrices (dtype=%s)", num_samples, dtype)
    return BatchGenerationResult(
        matrices=matrices,
        ipf_iterations=ipf_iterations,
        ipf_final_errors=ipf_final_errors,
    )
