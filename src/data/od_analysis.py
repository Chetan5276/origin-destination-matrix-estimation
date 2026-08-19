"""Stage 1: base OD matrix analysis."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BaseOdAnalysis:
    """Summary statistics of the observed base OD matrix."""

    num_zones: int
    total_demand: float
    productions: np.ndarray
    attractions: np.ndarray
    sparsity_fraction: float
    num_nonzero: int
    top_10_pairs: list[dict[str, float | int]]
    top_20_pairs: list[dict[str, float | int]]

    def to_dict(self) -> dict:
        data = asdict(self)
        data["productions"] = self.productions.tolist()
        data["attractions"] = self.attractions.tolist()
        return data


def _top_od_pairs(
    od_matrix: np.ndarray,
    k: int,
) -> list[dict[str, float | int]]:
    flat = od_matrix.ravel(order="C")
    n = od_matrix.shape[0]
    indices = np.argsort(flat)[::-1]
    pairs: list[dict[str, float | int]] = []
    for idx in indices:
        value = float(flat[idx])
        if value <= 0:
            break
        origin = idx // n + 1
        destination = idx % n + 1
        pairs.append(
            {
                "origin": int(origin),
                "destination": int(destination),
                "flow": value,
            }
        )
        if len(pairs) >= k:
            break
    return pairs


def analyze_base_od(od_matrix: np.ndarray) -> BaseOdAnalysis:
    """Compute productions, attractions, sparsity, and dominant OD pairs."""
    od = np.asarray(od_matrix, dtype=float)
    np.fill_diagonal(od, 0.0)

    n = od.shape[0]
    productions = od.sum(axis=1)
    attractions = od.sum(axis=0)
    total_demand = float(od.sum())
    num_nonzero = int(np.count_nonzero(od))
    sparsity_fraction = float(1.0 - num_nonzero / od.size)

    analysis = BaseOdAnalysis(
        num_zones=n,
        total_demand=total_demand,
        productions=productions,
        attractions=attractions,
        sparsity_fraction=sparsity_fraction,
        num_nonzero=num_nonzero,
        top_10_pairs=_top_od_pairs(od, 10),
        top_20_pairs=_top_od_pairs(od, 20),
    )
    logger.info(
        "Base OD: Q=%.2f, sparsity=%.1f%%, nonzero=%d",
        total_demand,
        sparsity_fraction * 100,
        num_nonzero,
    )
    return analysis


def write_analysis_report(
    analysis: BaseOdAnalysis,
    output_dir: Path,
) -> None:
    """Write JSON and markdown summaries of base OD analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "base_od_analysis.json"
    json_path.write_text(json.dumps(analysis.to_dict(), indent=2), encoding="utf-8")

    top10 = pd.DataFrame(analysis.top_10_pairs)
    top20 = pd.DataFrame(analysis.top_20_pairs)
    top10.to_csv(output_dir / "top_10_od_pairs.csv", index=False)
    top20.to_csv(output_dir / "top_20_od_pairs.csv", index=False)

    md_lines = [
        "# Base OD Analysis",
        "",
        f"- Zones: **{analysis.num_zones}**",
        f"- Total demand Q: **{analysis.total_demand:,.2f}**",
        f"- Nonzero cells: **{analysis.num_nonzero}** "
        f"({(1 - analysis.sparsity_fraction) * 100:.1f}% dense)",
        f"- Sparsity: **{analysis.sparsity_fraction * 100:.1f}%** zeros",
        "",
        "## Top 10 OD Pairs",
        "",
        "| Origin | Destination | Flow |",
        "|--------|-------------|------|",
    ]
    for row in analysis.top_10_pairs:
        md_lines.append(
            f"| {row['origin']} | {row['destination']} | {row['flow']:,.2f} |"
        )

    (output_dir / "base_od_analysis.md").write_text(
        "\n".join(md_lines) + "\n",
        encoding="utf-8",
    )
    logger.info("Wrote base OD analysis to %s", output_dir)
