"""Iterative proportional fitting (Furness algorithm)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class IpfResult:
    """Result of an IPF run."""

    matrix: np.ndarray
    iterations: int
    history: list[float]
    converged: bool
    final_error: float


def ipf(
    seed_matrix: np.ndarray,
    target_row: np.ndarray,
    target_col: np.ndarray,
    tol: float = 1e-3,
    max_iter: int = 1000,
    record_history: bool = False,
    out: np.ndarray | None = None,
) -> IpfResult:
    """
    Balance ``seed_matrix`` to match row and column targets via IPF.

    Parameters
    ----------
    record_history:
        When False (default for large batches), skip storing per-iteration errors.
    out:
        Optional output buffer; IPF runs in-place on this array when provided.
    """
    if out is None:
        matrix = seed_matrix.astype(np.float64, copy=True)
    else:
        matrix = out
        np.copyto(matrix, seed_matrix)

    history: list[float] = []
    final_error = float("inf")

    for iteration in range(1, max_iter + 1):
        row_sum = matrix.sum(axis=1)
        row_factor = np.divide(
            target_row,
            row_sum,
            out=np.ones_like(target_row),
            where=row_sum > 0,
        )
        matrix *= row_factor[:, None]

        col_sum = matrix.sum(axis=0)
        col_factor = np.divide(
            target_col,
            col_sum,
            out=np.ones_like(target_col),
            where=col_sum > 0,
        )
        matrix *= col_factor[None, :]

        row_err = np.max(np.abs(matrix.sum(axis=1) - target_row))
        col_err = np.max(np.abs(matrix.sum(axis=0) - target_col))
        final_error = float(max(row_err, col_err))

        if record_history:
            history.append(final_error)

        if final_error < tol:
            return IpfResult(
                matrix=matrix,
                iterations=iteration,
                history=history,
                converged=True,
                final_error=final_error,
            )

    return IpfResult(
        matrix=matrix,
        iterations=max_iter,
        history=history,
        converged=final_error < tol,
        final_error=final_error,
    )
