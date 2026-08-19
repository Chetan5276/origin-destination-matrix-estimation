"""Stage 9: per-matrix and fast batch quality metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from src.data.ipf import IpfResult
from src.data.od_generator import BatchGenerationResult, GenerationResult


@dataclass(frozen=True)
class MatrixMetrics:
    """Quality metrics for one synthetic OD matrix."""

    mae: float
    rmse: float
    relative_error: float
    correlation: float
    production_error: float
    attraction_error: float
    new_connections: int
    diagonal_violation: float
    sparsity: float
    total_demand: float
    ipf_iterations: int
    ipf_final_error: float
    ipf_converged: bool

    def to_dict(self) -> dict:
        return asdict(self)


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x.ravel(), y.ravel())[0, 1])


def compute_matrix_metrics(
    base_od: np.ndarray,
    synthetic_od: np.ndarray,
    ipf_result: IpfResult | None = None,
    ipf_iterations: int = -1,
    ipf_final_error: float = float("nan"),
) -> MatrixMetrics:
    """Compute similarity, marginal drift, and structural metrics."""
    base = np.asarray(base_od, dtype=np.float64)
    syn = np.asarray(synthetic_od, dtype=np.float64)
    diff = syn - base

    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))
    relative_error = float(np.sum(np.abs(diff)) / (base.sum() + 1e-12))
    correlation = _safe_corr(base, syn)

    p_base = base.sum(axis=1)
    a_base = base.sum(axis=0)
    p_syn = syn.sum(axis=1)
    a_syn = syn.sum(axis=0)

    production_error = float(
        np.mean(np.abs((p_syn - p_base) / (p_base + 1e-9)))
    )
    attraction_error = float(
        np.mean(np.abs((a_syn - a_base) / (a_base + 1e-9)))
    )

    support = base > 0
    new_connections = int(np.sum((~support) & (syn > 1e-6)))
    diagonal_violation = float(np.sum(np.diag(syn)))
    sparsity = float(np.mean(syn == 0))
    total_demand = float(syn.sum())

    if ipf_result is not None:
        ipf_iterations = ipf_result.iterations
        ipf_final_error = ipf_result.final_error
        ipf_converged = ipf_result.converged
    else:
        ipf_converged = ipf_final_error < 1e-3 if np.isfinite(ipf_final_error) else False

    return MatrixMetrics(
        mae=mae,
        rmse=rmse,
        relative_error=relative_error,
        correlation=correlation,
        production_error=production_error,
        attraction_error=attraction_error,
        new_connections=new_connections,
        diagonal_violation=diagonal_violation,
        sparsity=sparsity,
        total_demand=total_demand,
        ipf_iterations=ipf_iterations,
        ipf_final_error=ipf_final_error,
        ipf_converged=ipf_converged,
    )


def compute_batch_metrics(
    base_od: np.ndarray,
    synthetic_batch: np.ndarray,
    generation_results: list[GenerationResult] | None = None,
    batch_result: BatchGenerationResult | None = None,
    sample_size: int | None = None,
    seed: int = 42,
) -> list[MatrixMetrics]:
    """
    Compute metrics for a batch.

    When ``sample_size`` is set, evaluate only a random subset (for large N).
    """
    n = synthetic_batch.shape[0]
    if sample_size is not None and sample_size < n:
        rng = np.random.default_rng(seed)
        indices = rng.choice(n, size=sample_size, replace=False)
    else:
        indices = np.arange(n)

    metrics: list[MatrixMetrics] = []
    for k in indices:
        ipf_result = None
        ipf_iter = -1
        ipf_err = float("nan")
        if generation_results is not None:
            ipf_result = generation_results[k].ipf_result
        elif batch_result is not None and batch_result.ipf_iterations is not None:
            ipf_iter = int(batch_result.ipf_iterations[k])
            ipf_err = float(batch_result.ipf_final_errors[k])  # type: ignore[index]
        metrics.append(
            compute_matrix_metrics(
                base_od,
                synthetic_batch[k],
                ipf_result=ipf_result,
                ipf_iterations=ipf_iter,
                ipf_final_error=ipf_err,
            )
        )
    return metrics


def compute_fast_aggregate_metrics(
    base_od: np.ndarray,
    synthetic_batch: np.ndarray,
    sample_size: int = 10_000,
    seed: int = 42,
) -> dict[str, float]:
    """
    Vectorized aggregate metrics on a random sample — O(sample_size) not O(N).
    """
    n = synthetic_batch.shape[0]
    sample_size = min(sample_size, n)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=sample_size, replace=False)
    sample = synthetic_batch[idx].astype(np.float64)

    base = np.asarray(base_od, dtype=np.float64)
    base_flat = base.ravel()
    flat = sample.reshape(sample_size, -1)
    diff = flat - base_flat

    correlations = np.array([
        _safe_corr(base, sample[k]) for k in range(sample_size)
    ])

    support = base > 0
    new_connections = int(np.max([
        np.sum((~support) & (sample[k] > 1e-6)) for k in range(sample_size)
    ]))

    return {
        "mean_mae": float(np.mean(np.mean(np.abs(diff), axis=1))),
        "mean_rmse": float(np.mean(np.sqrt(np.mean(diff**2, axis=1)))),
        "mean_relative_error": float(np.mean(np.sum(np.abs(diff), axis=1) / base.sum())),
        "mean_correlation": float(np.nanmean(correlations)),
        "std_correlation": float(np.nanstd(correlations)),
        "max_new_connections": float(new_connections),
        "mean_total_demand": float(np.mean(flat.sum(axis=1))),
        "std_total_demand": float(np.std(flat.sum(axis=1))),
        "eval_sample_size": float(sample_size),
    }


def aggregate_matrix_metrics(metrics: list[MatrixMetrics]) -> dict[str, float]:
    """Return mean values across a list of per-matrix metrics."""
    if not metrics:
        return {}
    keys = metrics[0].to_dict().keys()
    agg: dict[str, float] = {}
    for key in keys:
        values = [getattr(m, key) for m in metrics]
        if isinstance(values[0], bool):
            agg[key] = float(np.mean(values))
        elif isinstance(values[0], int):
            agg[f"mean_{key}"] = float(np.mean(values))
            agg[f"max_{key}"] = float(np.max(values))
        else:
            agg[f"mean_{key}"] = float(np.mean(values))
            agg[f"std_{key}"] = float(np.std(values))
    return agg
