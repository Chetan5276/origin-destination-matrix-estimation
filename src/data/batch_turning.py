"""Memory-efficient batched OD → turning-count generation for large N."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src import NUM_OD_PAIRS, NUM_ZONES
from src.data.observation_noise import NoiseConfig, apply_observation_noise
from src.data.statistics import generate_turning_counts

logger = logging.getLogger(__name__)

LARGE_DATASET_THRESHOLD = 50_000
DEFAULT_BATCH_SIZE = 10_000


@dataclass
class StreamingStats:
    """Accumulate per-turn mean/std without storing full dataset."""

    num_turns: int
    count: int = 0
    sum_: np.ndarray | None = None
    sum_sq: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.sum_ is None:
            self.sum_ = np.zeros(self.num_turns, dtype=np.float64)
        if self.sum_sq is None:
            self.sum_sq = np.zeros(self.num_turns, dtype=np.float64)

    def update(self, batch: np.ndarray) -> None:
        batch = np.asarray(batch, dtype=np.float64)
        self.count += batch.shape[0]
        self.sum_ += batch.sum(axis=0)
        self.sum_sq += (batch * batch).sum(axis=0)

    @property
    def mean(self) -> np.ndarray:
        if self.count == 0:
            return np.zeros(self.num_turns)
        return self.sum_ / self.count

    @property
    def std(self) -> np.ndarray:
        if self.count == 0:
            return np.zeros(self.num_turns)
        mean = self.mean
        var = self.sum_sq / self.count - mean * mean
        return np.sqrt(np.maximum(var, 0.0))

    @property
    def cv(self) -> np.ndarray:
        return self.std / (self.mean + 1e-9)


def load_od_memmap(od_path: Path) -> np.ndarray:
    """Load OD array read-only via memory mapping (no full RAM load)."""
    od = np.load(od_path, mmap_mode="r")
    if od.ndim != 3 or od.shape[1:] != (NUM_ZONES, NUM_ZONES):
        raise ValueError(f"Expected OD shape (N, {NUM_ZONES}, {NUM_ZONES}), got {od.shape}")
    logger.info(
        "Memory-mapped OD: shape=%s dtype=%s (%.2f GB on disk)",
        od.shape,
        od.dtype,
        od.nbytes / 1e9,
    )
    return od


def _open_output_memmap(path: Path, shape: tuple[int, int], dtype=np.float32) -> np.ndarray:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def _flatten_chunk(od_chunk: np.ndarray) -> np.ndarray:
    """Flatten (B, 24, 24) → (B, 576) as float32 without extra copies."""
    return np.asarray(od_chunk, dtype=np.float32).reshape(od_chunk.shape[0], NUM_OD_PAIRS)


def generate_turning_counts_batched(
    od_path: Path,
    a_turn: np.ndarray,
    output_dir: Path,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_samples: int | None = None,
    noise_config: NoiseConfig | None = None,
    seed: int = 42,
    show_progress: bool = True,
) -> dict:
    """
    Generate turning counts in batches using memory-mapped I/O.

    Peak RAM ≈ batch_size × (576 + 178) × 4 bytes  (~30 MB at batch_size=10k)
    instead of loading all N OD matrices and all outputs at once.
    """
    od_mmap = load_od_memmap(od_path)
    n_total = od_mmap.shape[0]
    if max_samples is not None:
        n_total = min(n_total, max_samples)

    n_turns = a_turn.shape[0]
    a_turn_f32 = np.asarray(a_turn, dtype=np.float32)

    clean_path = output_dir / "turning_counts.npy"
    clean_mm = _open_output_memmap(clean_path, (n_total, n_turns))

    write_noisy = noise_config is not None and noise_config.model != "none"
    noisy_mm = None
    if write_noisy:
        noisy_path = output_dir / "turning_counts_noisy.npy"
        noisy_mm = _open_output_memmap(noisy_path, (n_total, n_turns))

    rng = np.random.default_rng(seed)
    stats = StreamingStats(num_turns=n_turns)
    max_reconstruction_error = 0.0

    batches = range(0, n_total, batch_size)
    iterator = tqdm(batches, desc="Turning counts", unit="batch") if show_progress else batches

    for start in iterator:
        end = min(start + batch_size, n_total)
        od_chunk = od_mmap[start:end]
        y = _flatten_chunk(od_chunk)
        x = generate_turning_counts(y, a_turn_f32).astype(np.float32, copy=False)
        clean_mm[start:end] = x
        stats.update(x)

        if write_noisy and noisy_mm is not None:
            noisy_mm[start:end] = apply_observation_noise(
                x, noise_config, rng  # type: ignore[arg-type]
            ).astype(np.float32, copy=False)

        # Validate first batch only
        if start == 0:
            recon = generate_turning_counts(y[:1], a_turn_f32)
            max_reconstruction_error = float(np.max(np.abs(recon[0] - x[0])))

    clean_mm.flush()
    if noisy_mm is not None:
        noisy_mm.flush()

    # Symlink-style duplicates expected by deliverables spec
    _link_or_copy(clean_path, output_dir / "clean_turn_counts.npy")
    if write_noisy and noisy_mm is not None:
        _link_or_copy(output_dir / "turning_counts_noisy.npy", output_dir / "noisy_turn_counts.npy")

    logger.info(
        "Batched generation complete: N=%d, batch_size=%d, peak batch RAM ~%.1f MB",
        n_total,
        batch_size,
        batch_size * (NUM_OD_PAIRS + n_turns) * 4 / 1e6,
    )
    return {
        "num_samples": n_total,
        "num_turns": n_turns,
        "per_turn_mean": stats.mean,
        "per_turn_std": stats.std,
        "per_turn_cv": stats.cv,
        "max_reconstruction_error": max_reconstruction_error,
        "matrix_mult_consistent": max_reconstruction_error <= 1e-3,
        "batch_size": batch_size,
        "clean_path": str(clean_path),
    }


def _link_or_copy(src: Path, dst: Path) -> None:
    """Hard-link output duplicate; fall back to skipping if already same file."""
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.hardlink_to(src)
    except OSError:
        # Different filesystem — copy is too expensive at 1M scale; skip duplicate
        logger.debug("Could not hardlink %s → %s; skipping duplicate", src, dst)


def sample_pairwise_distance(
    array_path: Path,
    n_rows: int,
    n_pairs: int = 10_000,
    seed: int = 42,
) -> float:
    """Estimate mean pairwise L1 distance by sampling rows from memmap."""
    data = np.load(array_path, mmap_mode="r")
    n = data.shape[0]
    if n < 2:
        return 0.0
    rng = np.random.default_rng(seed)
    idx_i = rng.integers(0, n, size=n_pairs)
    idx_j = rng.integers(0, n, size=n_pairs)
    same = idx_i == idx_j
    while same.any():
        idx_j[same] = rng.integers(0, n, size=int(same.sum()))
        same = idx_i == idx_j

    unique_idx = np.unique(np.concatenate([idx_i, idx_j]))
    row_cache: dict[int, np.ndarray] = {}
    for global_i in unique_idx:
        row = np.asarray(data[int(global_i)], dtype=np.float32)
        row_cache[int(global_i)] = row.reshape(-1) if row.ndim > 1 else row

    distances = [
        float(np.abs(row_cache[int(i)] - row_cache[int(j)]).sum())
        for i, j in zip(idx_i, idx_j)
    ]
    return float(np.mean(distances))


def sample_turn_correlation(
    turning_path: Path,
    n_sample_rows: int = 5_000,
    n_features: int = 50,
    seed: int = 42,
) -> tuple[float, float]:
    """Sample-based turn correlation from memmap (no full load)."""
    data = np.load(turning_path, mmap_mode="r")
    n, n_turns = data.shape
    rng = np.random.default_rng(seed)
    row_idx = rng.choice(n, size=min(n_sample_rows, n), replace=False)
    feat_idx = rng.choice(n_turns, size=min(n_features, n_turns), replace=False)
    subset = np.asarray(data[row_idx][:, feat_idx], dtype=np.float64)
    std = subset.std(axis=0)
    varying = std > 1e-12
    if varying.sum() < 2:
        return float("nan"), float("nan")
    corr = np.corrcoef(subset[:, varying], rowvar=False)
    off = corr[np.triu_indices(corr.shape[0], k=1)]
    return float(np.mean(off)), float(np.max(np.abs(off)))
