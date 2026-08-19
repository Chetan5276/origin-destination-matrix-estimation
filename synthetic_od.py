import argparse
import multiprocessing as mp
import time

import numpy as np


# ==========================================================
# Iterative Proportional Fitting (IPF) / Furness Algorithm
# ==========================================================
def ipf(seed_matrix,
        target_row,
        target_col,
        tol=1e-3,
        max_iter=1000):

    T = seed_matrix.astype(float).copy()

    history = []

    for it in range(max_iter):

        row_sum = T.sum(axis=1)

        row_factor = np.divide(
            target_row,
            row_sum,
            out=np.ones_like(target_row),
            where=row_sum > 0
        )

        T *= row_factor[:, None]

        col_sum = T.sum(axis=0)

        col_factor = np.divide(
            target_col,
            col_sum,
            out=np.ones_like(target_col),
            where=col_sum > 0
        )

        T *= col_factor[None, :]

        row_err = np.max(
            np.abs(T.sum(axis=1) - target_row)
        )

        col_err = np.max(
            np.abs(T.sum(axis=0) - target_col)
        )

        err = max(row_err, col_err)

        history.append(err)

        if err < tol:
            return T, it + 1, history

    return T, max_iter, history


# ==========================================================
# Generate Perturbed Marginals
# ==========================================================
def perturb_marginals(
        productions,
        attractions,
        perturbation=0.20,
        rng=None):

    if rng is None:
        rng = np.random.default_rng()

    # perturb productions
    p_new = productions * (
        1 + rng.uniform(
            -perturbation,
            perturbation,
            size=productions.shape
        )
    )

    # perturb attractions
    a_new = attractions * (
        1 + rng.uniform(
            -perturbation,
            perturbation,
            size=attractions.shape
        )
    )

    # enforce same grand total
    total = productions.sum()

    p_new *= total / p_new.sum()
    a_new *= total / a_new.sum()

    return p_new, a_new


# ==========================================================
# Generate One Synthetic OD Matrix
# ==========================================================
def generate_synthetic_od(
        base_od,
        perturbation=0.20,
        seed=None):

    rng = np.random.default_rng(seed)

    N = base_od.shape[0]

    productions = base_od.sum(axis=1)
    attractions = base_od.sum(axis=0)

    # Preserve sparsity structure
    mask = (base_od > 0).astype(float)

    # Perturb marginals
    target_row, target_col = perturb_marginals(
        productions,
        attractions,
        perturbation,
        rng
    )

    # Seed matrix
    seed_matrix = base_od.copy().astype(float)

    # Small multiplicative perturbation
    noise = rng.uniform(
        0.8,
        1.2,
        size=seed_matrix.shape
    )

    seed_matrix *= noise

    # Preserve sparsity
    seed_matrix *= mask

    # No intrazonal trips
    np.fill_diagonal(seed_matrix, 0)

    # Tiny value avoids divide-by-zero in IPF
    seed_matrix += mask * 1e-6

    # Run IPF
    synthetic, n_iter, history = ipf(
        seed_matrix,
        target_row,
        target_col
    )

    # Restore exact sparsity
    synthetic *= mask

    # Force diagonal zero again
    np.fill_diagonal(synthetic, 0)

    return synthetic, n_iter, history

# ==========================================================
# Similarity Metrics
# ==========================================================

def similarity_metrics(base, generated):

    diff = generated - base

    mae = np.mean(np.abs(diff))

    rmse = np.sqrt(
        np.mean(diff**2)
    )

    relative_error = (
        np.sum(np.abs(diff))
        /
        np.sum(base)
    )

    corr = np.corrcoef(
        base.flatten(),
        generated.flatten()
    )[0,1]

    return {
        "mae": mae,
        "rmse": rmse,
        "relative_error": relative_error,
        "correlation": corr
    }

# ==========================================================
# Production and Attraction Metrics
# ==========================================================


def marginal_metrics(base, generated):

    p0 = base.sum(axis=1)
    a0 = base.sum(axis=0)

    p1 = generated.sum(axis=1)
    a1 = generated.sum(axis=0)

    production_error = np.mean(
        np.abs(
            (p1 - p0) / (p0 + 1e-9)
        )
    )

    attraction_error = np.mean(
        np.abs(
            (a1 - a0) / (a0 + 1e-9)
        )
    )

    return {
        "production_error": production_error,
        "attraction_error": attraction_error
    }


# ==========================================================
# Structural Violation Metrics
# ==========================================================
def structural_metrics(base, generated):

    mask = base > 0

    new_connections = np.sum(
        (~mask) & (generated > 1e-6)
    )

    diagonal_violation = np.sum(
        np.diag(generated)
    )

    sparsity = (
        np.sum(generated == 0)
        /
        generated.size
    )

    return {
        "new_connections": int(new_connections),
        "diagonal_violation": diagonal_violation,
        "sparsity": sparsity
    }

# ==========================================================
# Diversity Metrics
# ==========================================================

def diversity_metrics(dataset, n_samples=10_000, seed=42):

    data = np.asarray(dataset)
    K = data.shape[0]

    if K < 2:
        return {
            "mean_distance": 0.0,
            "min_distance": 0.0,
            "max_distance": 0.0,
            "n_pairs_sampled": 0,
        }

    rng = np.random.default_rng(seed)
    max_pairs = K * (K - 1) // 2
    n_pairs = min(n_samples, max_pairs)

    idx_i = rng.integers(0, K, size=n_pairs)
    idx_j = rng.integers(0, K, size=n_pairs)
    same = idx_i == idx_j
    while same.any():
        idx_j[same] = rng.integers(0, K, size=same.sum())
        same = idx_i == idx_j

    flat = data.reshape(K, -1)
    pairwise = np.mean(
        np.abs(flat[idx_i] - flat[idx_j]),
        axis=1,
    )

    return {
        "mean_distance": float(pairwise.mean()),
        "min_distance": float(pairwise.min()),
        "max_distance": float(pairwise.max()),
        "n_pairs_sampled": int(n_pairs),
    }


# ==========================================================
# OD Variability Metrics
# ==========================================================
def od_variability(dataset):

    data = np.array(dataset)

    mean_od = data.mean(axis=0)

    std_od = data.std(axis=0)

    cv = std_od / (mean_od + 1e-9)

    return {
        "mean_cv": np.mean(cv),
        "max_cv": np.max(cv),
        "min_cv": np.min(cv)
    }

# ==========================================================
# Demand Consistency Metrics
# ==========================================================
def demand_consistency(dataset):

    data = np.asarray(dataset)
    totals = data.reshape(data.shape[0], -1).sum(axis=1)

    return {
        "mean_total": totals.mean(),
        "std_total": totals.std(),
        "cv_total":
            totals.std()
            /
            totals.mean()
    }

# ==========================================================
# Convergence Metrics
# ==========================================================
def convergence_metrics(iterations, final_errors):

    iterations = np.asarray(iterations)
    final_errors = np.asarray(final_errors)

    return {
        "mean_iterations": iterations.mean(),
        "max_iterations": iterations.max(),
        "min_iterations": iterations.min(),
        "mean_final_error": final_errors.mean(),
        "max_final_error": final_errors.max(),
    }


def matrix_metrics(base, generated):

    return {
        **similarity_metrics(base, generated),
        **marginal_metrics(base, generated),
        **structural_metrics(base, generated),
    }


# Worker globals (set once per process via pool initializer)
_WORKER_BASE_OD = None
_WORKER_PERTURBATION = None


def _init_worker(base_od, perturbation):
    global _WORKER_BASE_OD, _WORKER_PERTURBATION
    _WORKER_BASE_OD = base_od
    _WORKER_PERTURBATION = perturbation


def _generate_one(i):
    od, n_iter, history = generate_synthetic_od(
        _WORKER_BASE_OD,
        perturbation=_WORKER_PERTURBATION,
        seed=1000 + i,
    )
    return i, od, n_iter, history[-1]


def _metrics_one(od):
    return matrix_metrics(_WORKER_BASE_OD, od)



# ==========================================================
# Metric display hints
# ==========================================================
METRIC_HINTS = {
    # diversity
    "mean_distance": "moderate↑ = synthetics differ; good for augmentation",
    "min_distance": "↑ = even the most similar pair differs enough",
    "max_distance": "moderate = some variety without extreme outliers",
    # od variability
    "mean_cv": "moderate = healthy per-cell variation across samples",
    "max_cv": "↓ = no single OD cell is wildly unstable",
    "min_cv": "0 expected where all synthetics stay zero",
    # demand consistency
    "mean_total": "should match base grand total",
    "std_total": "↓ = consistent total demand across samples",
    "cv_total": "↓ (near 0) = total trips preserved by IPF",
    # convergence
    "mean_iterations": "↓ (with low error) = fast, reliable IPF",
    "max_iterations": "↓ = no slow or stuck fits",
    "min_iterations": "informational; small spread means stable generation",
    "mean_final_error": "↓ = row/column targets met within tolerance",
    "max_final_error": "↓ = worst matrix still acceptable",
    # per-matrix vs base
    "mae": "↓ = closer to base; ↑ = stronger perturbation",
    "rmse": "↓ = fewer large cell-level errors",
    "relative_error": "↓ = less total drift from base demand",
    "correlation": "↑ (→1) = OD spatial pattern preserved",
    "production_error": "↓ = origin (row) totals stay realistic",
    "attraction_error": "↓ = destination (col) totals stay realistic",
    "new_connections": "↓ (0) = no spurious nonzero links invented",
    "diagonal_violation": "↓ (0) = no intrazonal trips",
    "sparsity": "match base = realistic zero structure",
}


def print_metrics_block(title, metrics):
    print(f"\n{title}")
    print("-" * len(title))
    for key, value in metrics.items():
        hint = METRIC_HINTS.get(key, "")
        print(f"  {key}: {value}  — {hint}")


# ==========================================================
# Main
# ==========================================================
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Generate synthetic OD matrices via IPF."
    )
    parser.add_argument(
        "-n", "--n-matrices",
        type=int,
        default=100_000,
        help="number of synthetic matrices to generate (default: 100000)",
    )
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=None,
        help="parallel worker processes (default: all CPU cores)",
    )
    parser.add_argument(
        "--diversity-samples",
        type=int,
        default=10_000,
        help="random pairs for diversity estimate (default: 10000)",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="output .npy path (default: synthetic_od_<n>.npy)",
    )
    args = parser.parse_args()

    n_matrices = args.n_matrices
    n_workers = args.workers or mp.cpu_count()
    output_path = args.output or f"synthetic_od_{n_matrices}.npy"
    perturbation = 0.20

    t_total = time.perf_counter()

    base_od = np.load("EstimatedODMatrix.npy").astype(float)
    base_od = np.clip(base_od, 0, None)
    np.fill_diagonal(base_od, 0)

    n_zones = base_od.shape[0]
    synthetic_matrices = np.zeros((n_matrices, n_zones, n_zones), dtype=float)
    iteration_list = np.zeros(n_matrices, dtype=int)
    final_errors = np.zeros(n_matrices, dtype=float)

    print(
        f"Generating {n_matrices} matrices "
        f"({n_zones}x{n_zones}) using {n_workers} workers..."
    )

    t_gen = time.perf_counter()
    chunksize = max(1, n_matrices // (n_workers * 4))

    with mp.Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=(base_od, perturbation),
    ) as pool:
        for i, od, n_iter, final_err in pool.imap_unordered(
            _generate_one,
            range(n_matrices),
            chunksize=chunksize,
        ):
            synthetic_matrices[i] = od
            iteration_list[i] = n_iter
            final_errors[i] = final_err

    gen_elapsed = time.perf_counter() - t_gen

    print(
        f"Generated {synthetic_matrices.shape[0]} matrices "
        f"of size {synthetic_matrices.shape[1]}x"
        f"{synthetic_matrices.shape[2]} in {gen_elapsed:.1f}s "
        f"({n_matrices / gen_elapsed:.0f} matrices/s)"
    )

    t_save = time.perf_counter()
    np.save(output_path, synthetic_matrices)
    save_elapsed = time.perf_counter() - t_save
    print(f"Saved {output_path} in {save_elapsed:.1f}s")

    t_metrics = time.perf_counter()
    with mp.Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=(base_od, perturbation),
    ) as pool:
        all_metrics = list(
            pool.imap(_metrics_one, synthetic_matrices, chunksize=chunksize)
        )

    metrics_elapsed = time.perf_counter() - t_metrics

    print("=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    print_metrics_block(
        "Diversity (sampled pairwise difference between synthetics)",
        diversity_metrics(
            synthetic_matrices,
            n_samples=args.diversity_samples,
        ),
    )

    print_metrics_block(
        "OD variability (per-cell spread across synthetics)",
        od_variability(synthetic_matrices),
    )

    print_metrics_block(
        "Demand consistency (grand total per matrix)",
        demand_consistency(synthetic_matrices),
    )

    print_metrics_block(
        "IPF convergence (fitting quality)",
        convergence_metrics(iteration_list, final_errors),
    )

    avg_metrics = {
        key: np.mean([m[key] for m in all_metrics])
        for key in all_metrics[0]
    }
    print_metrics_block(
        "Per-matrix vs base (averaged over all synthetics)",
        avg_metrics,
    )

    total_elapsed = time.perf_counter() - t_total
    print("\n" + "=" * 60)
    print("TIMING")
    print("=" * 60)
    print(f"  generation:  {gen_elapsed:.1f}s")
    print(f"  save:        {save_elapsed:.1f}s")
    print(f"  metrics:     {metrics_elapsed:.1f}s")
    print(f"  total:       {total_elapsed:.1f}s")



    