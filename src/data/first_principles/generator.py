"""First-principles OD generator: Stages A–E (latent → gravity → Gamma → Dir → IPF)."""

from __future__ import annotations

import logging
import multiprocessing as mp
from dataclasses import dataclass

import numpy as np

from src import PROJECT_ROOT
from src.data.first_principles.compositional import dirichlet_reweight, seed_to_probability
from src.data.first_principles.config import FPGeneratorConfig
from src.data.first_principles.gravity import build_gravity_matrix
from src.data.first_principles.heavy_tails import apply_gamma_weights
from src.data.first_principles.latent import sample_latent_factors
from src.data.ipf import IpfResult, ipf
from src.data.od_probability import (
    full_off_diagonal_support,
    load_zone_coordinates,
    zone_distance_matrix,
)

logger = logging.getLogger(__name__)

DEFAULT_NETWORK_PATH = PROJECT_ROOT / "sioux-falls.net.xml"
_WORKER: dict | None = None


@dataclass(frozen=True)
class FPGenerationResult:
    matrix: np.ndarray
    ipf_result: IpfResult
    productions: np.ndarray
    attractions: np.ndarray


@dataclass(frozen=True)
class FPBatchResult:
    matrices: np.ndarray
    ipf_iterations: np.ndarray | None = None
    ipf_final_errors: np.ndarray | None = None


def build_reference_od(
    config: FPGeneratorConfig,
    *,
    distance: np.ndarray | None = None,
    coords: np.ndarray | None = None,
    network_path=None,
    seed: int = 0,
) -> np.ndarray:
    """
    Deterministic-ish reference table for quality filters / trip-length checks.

    Uses fixed RNG seed and skips Gamma + Dirichlet noise (gravity + IPF only).
    """
    rng = np.random.default_rng(seed)
    net = network_path or DEFAULT_NETWORK_PATH
    if coords is None:
        coords = load_zone_coordinates(net)
    if distance is None:
        distance = zone_distance_matrix(coords=coords, network_path=net)
    support = full_off_diagonal_support(config.num_zones)
    u, v, _ = sample_latent_factors(
        config.num_zones,
        config.latent_dim,
        coords,
        distance,
        rng,
        factor_shape=config.factor_shape,
        factor_scale=config.factor_scale,
        spatial_smooth_length=config.spatial_smooth_length,
        spatial_smooth_strength=config.spatial_smooth_strength,
    )
    s = build_gravity_matrix(
        u,
        v,
        distance,
        support,
        decay=config.decay,
        lambda_decay=config.lambda_decay,
        power_gamma=config.power_gamma,
        reciprocity=config.reciprocity,
    )
    p = seed_to_probability(s, support)
    seed_mat = config.total_demand * p
    seed_mat[support] += config.seed_floor
    np.fill_diagonal(seed_mat, 0.0)
    row = u / u.sum() * config.total_demand
    col = v / v.sum() * config.total_demand
    return ipf(seed_mat, row, col, tol=config.ipf_tol, max_iter=config.ipf_max_iter).matrix


def _marginal_targets(
    u: np.ndarray,
    v: np.ndarray,
    total_demand: float,
    latent_marginals: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if latent_marginals:
        row = u / u.sum() * total_demand
        col = v / v.sum() * total_demand
    else:
        n = u.size
        row = np.full(n, total_demand / n)
        col = np.full(n, total_demand / n)
    return row, col


def generate_one_fp_od(
    config: FPGeneratorConfig,
    rng: np.random.Generator,
    *,
    distance: np.ndarray,
    coords: np.ndarray,
    support_mask: np.ndarray | None = None,
) -> FPGenerationResult:
    """Generate a single first-principles OD matrix (Stages A–E)."""
    support = support_mask if support_mask is not None else full_off_diagonal_support(config.num_zones)

    # A. Latent factors
    u, v, _ = sample_latent_factors(
        config.num_zones,
        config.latent_dim,
        coords,
        distance,
        rng,
        factor_shape=config.factor_shape,
        factor_scale=config.factor_scale,
        spatial_smooth_length=config.spatial_smooth_length,
        spatial_smooth_strength=config.spatial_smooth_strength,
    )

    # B. Gravity + reciprocity
    seed = build_gravity_matrix(
        u,
        v,
        distance,
        support,
        decay=config.decay,
        lambda_decay=config.lambda_decay,
        power_gamma=config.power_gamma,
        reciprocity=config.reciprocity,
    )

    # C. Heavy tails
    if config.apply_gamma_mask:
        seed = apply_gamma_weights(
            seed,
            rng,
            support,
            shape=config.gamma_shape,
            scale=config.gamma_scale,
        )

    # D. Dirichlet compositional noise
    prob = dirichlet_reweight(seed, support, config.alpha, rng)

    # E. Scale + IPF
    mat = config.total_demand * prob
    mat[support] += config.seed_floor
    np.fill_diagonal(mat, 0.0)
    row, col = _marginal_targets(u, v, config.total_demand, config.latent_marginals)
    result = ipf(mat, row, col, tol=config.ipf_tol, max_iter=config.ipf_max_iter)
    out = result.matrix
    out[~support] = 0.0
    np.fill_diagonal(out, 0.0)
    return FPGenerationResult(
        matrix=out,
        ipf_result=result,
        productions=row,
        attractions=col,
    )


def _init_worker(config: FPGeneratorConfig, distance: np.ndarray, coords: np.ndarray) -> None:
    global _WORKER
    _WORKER = {
        "config": config,
        "distance": distance,
        "coords": coords,
        "support": full_off_diagonal_support(config.num_zones),
    }


def _worker_task_seeded(args: tuple[int, int]) -> tuple[int, np.ndarray, int, float]:
    """args = (index, seed)."""
    assert _WORKER is not None
    idx, seed = args
    cfg: FPGeneratorConfig = _WORKER["config"]
    rng = np.random.default_rng(seed)
    res = generate_one_fp_od(
        cfg,
        rng,
        distance=_WORKER["distance"],
        coords=_WORKER["coords"],
        support_mask=_WORKER["support"],
    )
    return idx, res.matrix.astype(np.float64), res.ipf_result.iterations, res.ipf_result.final_error


def generate_fp_od_batch(
    n_samples: int,
    config: FPGeneratorConfig,
    *,
    seed: int = 42,
    workers: int | None = None,
    dtype=np.float32,
    store_ipf_stats: bool = False,
    show_progress: bool = True,
    network_path=None,
) -> FPBatchResult:
    """Parallel batch generation of first-principles OD matrices."""
    net = network_path or DEFAULT_NETWORK_PATH
    coords = load_zone_coordinates(net)
    distance = zone_distance_matrix(coords=coords, network_path=net)

    n_workers = workers if workers is not None else max(1, mp.cpu_count())
    tasks = [(k, seed + k) for k in range(n_samples)]
    out = np.zeros((n_samples, config.num_zones, config.num_zones), dtype=dtype)
    iters = np.zeros(n_samples, dtype=np.int32) if store_ipf_stats else None
    errs = np.zeros(n_samples, dtype=np.float64) if store_ipf_stats else None

    iterator: list | object
    if n_workers <= 1:
        _init_worker(config, distance, coords)
        results = []
        it = tasks
        if show_progress:
            try:
                from tqdm import tqdm

                it = tqdm(tasks, desc="FP OD", unit="matrix")
            except ImportError:
                pass
        for args in it:
            results.append(_worker_task_seeded(args))
    else:
        logger.info("FP parallel generation: %d samples, %d workers", n_samples, n_workers)
        chunksize = max(1, n_samples // (n_workers * 8))
        with mp.Pool(
            processes=n_workers,
            initializer=_init_worker,
            initargs=(config, distance, coords),
        ) as pool:
            map_iter = pool.imap_unordered(_worker_task_seeded, tasks, chunksize=chunksize)
            if show_progress:
                try:
                    from tqdm import tqdm

                    map_iter = tqdm(map_iter, total=n_samples, desc="FP OD", unit="matrix")
                except ImportError:
                    pass
            results = list(map_iter)

    for idx, mat, n_iter, err in results:
        out[idx] = mat.astype(dtype, copy=False)
        if store_ipf_stats and iters is not None and errs is not None:
            iters[idx] = n_iter
            errs[idx] = err

    logger.info("Generated %d first-principles OD matrices (dtype=%s)", n_samples, dtype)
    return FPBatchResult(matrices=out, ipf_iterations=iters, ipf_final_errors=errs)
