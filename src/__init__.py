"""OD estimation and dataset generation for Sioux Falls."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DOCS_DIR = PROJECT_ROOT / "docs"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
REPORT_DIR = PROJECT_ROOT / "report"
OD_GENERATOR_OUTPUT_DIR = OUTPUT_DIR / "od_generator"

NUM_ZONES = 24
NUM_OD_PAIRS = NUM_ZONES * NUM_ZONES
