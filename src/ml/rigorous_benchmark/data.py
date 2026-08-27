"""FP data loading, 70/15/15 split, leakage report, scalers."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

from src import NUM_ZONES
from src.data.od_pairs import OdPairIndex
from src.data.statistics import flatten_od_batch
from src.ml.od_constraints import base_support_mask
from src.ml.rigorous_benchmark.config import BenchmarkConfig

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkData:
    """In-memory split arrays + fitted train-only scalers."""

    x_train_raw: np.ndarray
    x_val_raw: np.ndarray
    x_test_raw: np.ndarray
    y_train_raw: np.ndarray
    y_val_raw: np.ndarray
    y_test_raw: np.ndarray
    x_train: np.ndarray
    x_val: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    a_turn: np.ndarray
    a_turn_sha256: str
    x_scaler: StandardScaler | None
    y_scaler: StandardScaler | None
    support_mask: np.ndarray
    base_od_flat: np.ndarray
    survey_x_raw: np.ndarray
    survey_y_raw: np.ndarray
    od_index: OdPairIndex

    @property
    def n_features(self) -> int:
        return int(self.x_train.shape[1])

    @property
    def n_targets(self) -> int:
        return int(self.y_train.shape[1])

    def train_val_raw(self) -> tuple[np.ndarray, np.ndarray]:
        x = np.concatenate([self.x_train_raw, self.x_val_raw], axis=0)
        y = np.concatenate([self.y_train_raw, self.y_val_raw], axis=0)
        return x, y

    def train_val_scaled(self) -> tuple[np.ndarray, np.ndarray]:
        x = np.concatenate([self.x_train, self.x_val], axis=0)
        y = np.concatenate([self.y_train, self.y_val], axis=0)
        return x, y

    def hpo_train_subset(
        self, n: int, seed: int = 42
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Subsample train for Optuna; returns scaled + raw X/Y."""
        n = min(n, len(self.x_train))
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(self.x_train), size=n, replace=False)
        return (
            self.x_train[idx],
            self.y_train[idx],
            self.x_train_raw[idx],
            self.y_train_raw[idx],
        )


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _sha256_array(arr: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def verify_forward_map(
    y_flat: np.ndarray,
    x: np.ndarray,
    a_turn: np.ndarray,
    *,
    atol: float = 1e-2,
    rtol: float = 1e-3,
    n_check: int = 32,
) -> dict:
    """Sanity-check X ≈ Y_flat @ A_turn.T on a few samples."""
    n = min(n_check, len(y_flat))
    y = np.asarray(y_flat[:n], dtype=np.float64)
    x_obs = np.asarray(x[:n], dtype=np.float64)
    a = np.asarray(a_turn, dtype=np.float64)
    x_hat = y @ a.T
    err = np.abs(x_obs - x_hat)
    rel = err / np.maximum(np.abs(x_obs), 1e-6)
    ok = bool(np.allclose(x_obs, x_hat, atol=atol, rtol=rtol))
    return {
        "n_checked": n,
        "max_abs_err": float(err.max()),
        "mean_abs_err": float(err.mean()),
        "max_rel_err": float(rel.max()),
        "allclose": ok,
        "forward_formula": "X = Y_flat @ A_turn.T",
    }


def make_split_indices(
    n: int,
    *,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-9:
        raise ValueError("fractions must sum to 1")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    # remainder to test to guarantee sum == n
    n_test = n - n_train - n_val
    if n_test < 0:
        raise ValueError("invalid split sizes")
    train_idx = np.sort(perm[:n_train])
    val_idx = np.sort(perm[n_train : n_train + n_val])
    test_idx = np.sort(perm[n_train + n_val :])
    return train_idx, val_idx, test_idx


def assert_disjoint(train_idx, val_idx, test_idx) -> None:
    s_tr, s_va, s_te = set(train_idx.tolist()), set(val_idx.tolist()), set(test_idx.tolist())
    if s_tr & s_va or s_tr & s_te or s_va & s_te:
        raise AssertionError("Split indices are not disjoint")
    if len(s_tr) + len(s_va) + len(s_te) != len(s_tr | s_va | s_te):
        raise AssertionError("Duplicate indices within splits")


def write_leakage_report(
    path: Path,
    *,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    survey_index: int,
    n_synthetics: int,
    a_turn_sha256: str,
    forward_check: dict,
    max_samples: int | None,
) -> dict:
    assert_disjoint(train_idx, val_idx, test_idx)
    all_idx = set(train_idx.tolist()) | set(val_idx.tolist()) | set(test_idx.tolist())
    report = {
        "protocol": "FP synthetics only for train/val/test/HPO; survey held out until inference",
        "n_synthetics_available": n_synthetics,
        "max_samples": max_samples,
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "splits_disjoint": True,
        "survey_turning_index": survey_index,
        "survey_in_train_val_test": survey_index in all_idx,
        "survey_excluded_from_development": survey_index not in all_idx,
        "generator_aware_split": "N/A — no per-sample seed metadata in FP artifacts; random split v1",
        "a_turn_sha256": a_turn_sha256,
        "forward_sanity": forward_check,
        "scalers_fit_on": "train only (never refit on survey)",
        "hpo_uses": "train subsample + validation only; test never used for selection",
    }
    if survey_index in all_idx:
        raise AssertionError("Survey index leaked into development splits")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    return report


def load_benchmark_data(config: BenchmarkConfig) -> BenchmarkData:
    """Load FP memmaps, split, fit scalers on train only, load survey separately."""
    od_path = config.synthetic_od_path
    turn_path = config.turning_counts_path
    a_path = config.a_turn_path

    od_mm = np.load(od_path, mmap_mode="r")
    x_mm = np.load(turn_path, mmap_mode="r")
    a_turn = np.asarray(np.load(a_path), dtype=np.float32)

    n_avail = min(int(od_mm.shape[0]), config.n_synthetics, int(x_mm.shape[0]))
    n = n_avail if config.max_samples is None else min(n_avail, config.max_samples)
    logger.info("Loading %d synthetics from %s / %s", n, od_path, turn_path)

    y_mat = np.asarray(od_mm[:n], dtype=np.float32)
    y_flat = flatten_od_batch(y_mat)
    x_all = np.asarray(x_mm[:n], dtype=np.float32)

    a_sha = _sha256_file(a_path) if a_path.exists() else _sha256_array(a_turn)
    fwd = verify_forward_map(y_flat, x_all, a_turn)

    train_idx, val_idx, test_idx = make_split_indices(
        n,
        train_frac=config.train_frac,
        val_frac=config.val_frac,
        test_frac=config.test_frac,
        seed=config.seed,
    )
    assert_disjoint(train_idx, val_idx, test_idx)

    data_dir = config.data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        data_dir / "split_indices.npz",
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        seed=np.array([config.seed]),
        n=np.array([n]),
    )
    (data_dir / "a_turn.sha256").write_text(a_sha + "\n")

    od_index = OdPairIndex.build(NUM_ZONES)
    od_map = {
        "num_zones": NUM_ZONES,
        "num_od_pairs": od_index.num_od_pairs,
        "index_to_od_pair": {
            str(k): list(v) for k, v in od_index.index_to_od_pair.items()
        },
        "note": "Zones are 1-indexed in OdPairIndex; flat storage is row-major 0..23",
    }
    (data_dir / "od_index_map.json").write_text(json.dumps(od_map))

    write_leakage_report(
        data_dir / "leakage_report.json",
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        survey_index=config.survey_turning_index,
        n_synthetics=n_avail,
        a_turn_sha256=a_sha,
        forward_check=fwd,
        max_samples=config.max_samples,
    )

    x_train_raw = x_all[train_idx].copy()
    x_val_raw = x_all[val_idx].copy()
    x_test_raw = x_all[test_idx].copy()
    y_train_raw = y_flat[train_idx].copy()
    y_val_raw = y_flat[val_idx].copy()
    y_test_raw = y_flat[test_idx].copy()

    x_scaler = StandardScaler() if config.standardize_x else None
    y_scaler = StandardScaler() if config.standardize_y else None
    if x_scaler is not None:
        x_train = x_scaler.fit_transform(x_train_raw).astype(np.float32)
        x_val = x_scaler.transform(x_val_raw).astype(np.float32)
        x_test = x_scaler.transform(x_test_raw).astype(np.float32)
    else:
        x_train, x_val, x_test = x_train_raw.copy(), x_val_raw.copy(), x_test_raw.copy()
    if y_scaler is not None:
        y_train = y_scaler.fit_transform(y_train_raw).astype(np.float32)
        y_val = y_scaler.transform(y_val_raw).astype(np.float32)
        y_test = y_scaler.transform(y_test_raw).astype(np.float32)
    else:
        y_train, y_val, y_test = y_train_raw.copy(), y_val_raw.copy(), y_test_raw.copy()

    prep = config.preprocessing_dir()
    prep.mkdir(parents=True, exist_ok=True)
    joblib.dump({"x_scaler": x_scaler, "y_scaler": y_scaler}, prep / "scalers.joblib")

    # Survey — loaded only for later inference; never mixed into splits
    survey_od = np.load(config.survey_od_path).astype(np.float32)
    np.fill_diagonal(survey_od, 0.0)
    survey_y = flatten_od_batch(survey_od.reshape(1, NUM_ZONES, NUM_ZONES))[0]
    if x_mm.shape[0] > config.survey_turning_index:
        survey_x = np.asarray(x_mm[config.survey_turning_index], dtype=np.float32)
    else:
        # Reconstruct from forward map if turning row missing
        survey_x = (survey_y @ a_turn.T).astype(np.float32)
        logger.warning("Survey turning index missing; used forward map reconstruction")

    base_flat = survey_y.copy()  # survey/base OD for residual learning baseline
    support_mask = base_support_mask(survey_od)

    return BenchmarkData(
        x_train_raw=x_train_raw,
        x_val_raw=x_val_raw,
        x_test_raw=x_test_raw,
        y_train_raw=y_train_raw,
        y_val_raw=y_val_raw,
        y_test_raw=y_test_raw,
        x_train=x_train,
        x_val=x_val,
        x_test=x_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        a_turn=a_turn,
        a_turn_sha256=a_sha,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        support_mask=support_mask,
        base_od_flat=base_flat,
        survey_x_raw=survey_x,
        survey_y_raw=survey_y,
        od_index=od_index,
    )


def transform_x(data: BenchmarkData, x_raw: np.ndarray) -> np.ndarray:
    if data.x_scaler is None:
        return np.asarray(x_raw, dtype=np.float32)
    return data.x_scaler.transform(np.asarray(x_raw, dtype=np.float32)).astype(np.float32)


def inverse_transform_y(data: BenchmarkData, y_scaled: np.ndarray) -> np.ndarray:
    if data.y_scaler is None:
        return np.asarray(y_scaled, dtype=np.float32)
    return data.y_scaler.inverse_transform(np.asarray(y_scaled, dtype=np.float32)).astype(
        np.float32
    )
