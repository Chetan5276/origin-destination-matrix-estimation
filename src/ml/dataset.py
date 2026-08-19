"""Load turning counts and OD targets with train/val/test splits."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src import BASE_OD_PATH, NUM_OD_PAIRS, NUM_ZONES
from src.data.statistics import flatten_od_batch
from src.ml.od_constraints import base_support_mask

logger = logging.getLogger(__name__)


@dataclass
class ODDataset:
    """ML-ready arrays with scaling and physical metadata."""

    x_train: np.ndarray
    x_val: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    x_train_raw: np.ndarray
    x_val_raw: np.ndarray
    x_test_raw: np.ndarray
    y_train_raw: np.ndarray
    y_val_raw: np.ndarray
    y_test_raw: np.ndarray
    x_scaler: StandardScaler | None
    y_scaler: StandardScaler | None
    base_od_flat: np.ndarray
    support_mask: np.ndarray

    @property
    def n_features(self) -> int:
        return self.x_train.shape[1]

    @property
    def n_targets(self) -> int:
        return self.y_train.shape[1]


def load_base_od(path: Path | None = None) -> np.ndarray:
    """Load base OD matrix (24, 24)."""
    path = path or BASE_OD_PATH
    od = np.load(path).astype(np.float32)
    np.fill_diagonal(od, 0.0)
    return od


def load_turning_and_od(
    turning_path: str | Path,
    od_path: str | Path,
    max_samples: int | None = None,
    use_noisy: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Load X (turning counts) and Y (flattened OD) from disk."""
    turning_path = Path(turning_path)
    od_path = Path(od_path)

    if use_noisy:
        noisy = turning_path.parent / "turning_counts_noisy.npy"
        if noisy.exists():
            turning_path = noisy

    x = np.load(turning_path, mmap_mode="r")
    od = np.load(od_path, mmap_mode="r")

    n = min(x.shape[0], od.shape[0])
    if max_samples is not None:
        n = min(n, max_samples)

    x_arr = np.asarray(x[:n], dtype=np.float32)
    y_arr = flatten_od_batch(np.asarray(od[:n], dtype=np.float32))
    logger.info("Loaded dataset: X=%s Y=%s", x_arr.shape, y_arr.shape)
    return x_arr, y_arr


def split_dataset(
    x: np.ndarray,
    y: np.ndarray,
    *,
    base_od: np.ndarray | None = None,
    seed: int = 42,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    standardize_x: bool = True,
    standardize_y: bool = True,
) -> ODDataset:
    """Random split with optional standardization of X and Y."""
    test_frac = 1.0 - train_frac - val_frac
    if test_frac <= 0:
        raise ValueError("train_frac + val_frac must be < 1")

    if base_od is None:
        base_od = load_base_od()
    base_flat = flatten_od_batch(base_od.reshape(1, NUM_ZONES, NUM_ZONES))[0]
    support_mask = base_support_mask(base_od)

    x_temp, x_test, y_temp, y_test = train_test_split(
        x, y, test_size=test_frac, random_state=seed
    )
    val_ratio = val_frac / (train_frac + val_frac)
    x_train, x_val, y_train, y_val = train_test_split(
        x_temp, y_temp, test_size=val_ratio, random_state=seed
    )

    x_train_raw = x_train.copy()
    x_val_raw = x_val.copy()
    x_test_raw = x_test.copy()
    y_train_raw = y_train.copy()
    y_val_raw = y_val.copy()
    y_test_raw = y_test.copy()

    x_scaler = y_scaler = None
    if standardize_x:
        x_scaler = StandardScaler()
        x_train = x_scaler.fit_transform(x_train)
        x_val = x_scaler.transform(x_val)
        x_test = x_scaler.transform(x_test)
    if standardize_y:
        y_scaler = StandardScaler()
        y_train = y_scaler.fit_transform(y_train)
        y_val = y_scaler.transform(y_val)
        y_test = y_scaler.transform(y_test)

    return ODDataset(
        x_train=x_train.astype(np.float32),
        x_val=x_val.astype(np.float32),
        x_test=x_test.astype(np.float32),
        y_train=y_train.astype(np.float32),
        y_val=y_val.astype(np.float32),
        y_test=y_test.astype(np.float32),
        x_train_raw=x_train_raw.astype(np.float32),
        x_val_raw=x_val_raw.astype(np.float32),
        x_test_raw=x_test_raw.astype(np.float32),
        y_train_raw=y_train_raw.astype(np.float32),
        y_val_raw=y_val_raw.astype(np.float32),
        y_test_raw=y_test_raw.astype(np.float32),
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        base_od_flat=base_flat.astype(np.float32),
        support_mask=support_mask,
    )


def inverse_transform_y(dataset: ODDataset, y: np.ndarray) -> np.ndarray:
    if dataset.y_scaler is not None:
        return dataset.y_scaler.inverse_transform(y)
    return y
