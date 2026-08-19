"""First-principles synthetic OD generation (no historical base OD required)."""

from src.data.first_principles.config import FPGeneratorConfig
from src.data.first_principles.generator import (
    generate_fp_od_batch,
    generate_one_fp_od,
    build_reference_od,
)

__all__ = [
    "FPGeneratorConfig",
    "generate_fp_od_batch",
    "generate_one_fp_od",
    "build_reference_od",
]
