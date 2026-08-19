# Origin-Destination Matrix Estimation

Pipelines for synthetic origin-destination (OD) generation, turning-movement assignment on the Sioux Falls network, and ML-based OD matrix estimation from turning counts.

Use the conda environment **ODEstimation** as the runtime.

```bash
conda activate ODEstimation
cd ~/repos/origin-destination-matrix-estimation
pip install -e ".[dev]"   # first time only
```

## Layout

```
data/                 # Input network and base OD matrix
src/data/             # OD generation, route assignment, turning counts
src/ml/               # Classical and neural OD estimators
notebooks/            # Exploratory notebooks
tests/
docs/                 # Methodology (LaTeX + PDF)
outputs/              # Generated datasets (local; not in git)
```

## Pipelines

Run from the repo root:

```bash
# Phase 1 — synthetic OD matrices from the estimated base OD
python -m src.data.generate_od_dataset

# Phase 1b — first-principles OD generation (no historical base OD)
python -m src.data.generate_fp_od_dataset

# Phase 2 — turning counts from OD via assignment
python -m src.data.generate_turning_counts
python -m src.data.generate_dataset

# Phase 3 — train / benchmark OD estimators
python -m src.ml.train_od_estimator
```

## Tests

```bash
pytest tests/ -q
```

## Data

Tracked inputs live in [`data/`](data/README.md). Large generated `.npy` files stay under `outputs/` and are not committed.
