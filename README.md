# Origin-Destination Matrix Estimation

Pipelines for synthetic origin-destination (OD) generation, turning-movement assignment on the Sioux Falls network, and ML-based OD matrix estimation from turning counts.

Canonical project location: `~/repos/origin-destination-matrix-estimation`. Runtime is the existing conda environment **ODEstimation**.

## Setup

```bash
conda activate ODEstimation
cd ~/repos/origin-destination-matrix-estimation
```

Optional (fresh Python env instead of conda):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

SUMO is not required to run the current assignment code; the network file `sioux-falls.net.xml` is parsed directly.

## Layout

```
src/data/          # OD generation, route assignment, turning counts
src/ml/            # Classical and neural OD estimators
tests/
docs/              # Methodology notes (LaTeX/PDF)
report/            # Dataset summaries
data/              # Network + base OD (large synthetics live under outputs/)
outputs/           # Generated datasets and training artifacts (local, gitignored binaries)
```

## Pipelines

From the repo root, with `ODEstimation` activated:

```bash
# Phase 1 — synthetic OD matrices from the estimated base OD
python generate_od_dataset.py

# Phase 2 — turning counts from OD via assignment
python generate_turning_counts.py
python run_dataset_pipeline.py

# Phase 3 — train / benchmark OD estimators
python train_od_estimator.py
```

Notebook: `OD_generation.ipynb`.

## Tests

```bash
conda activate ODEstimation
pytest tests/ -q
```

## Data

Input network and base OD are in the repo. Large generated `.npy` datasets stay on disk under `outputs/` and are not committed. See [`data/README.md`](data/README.md) and [`outputs/README.md`](outputs/README.md).

## Original workspace

The conda environment `~/anaconda3/envs/ODEstimation` remains the Python runtime. Project files were moved here; the environment directory keeps compatibility symlinks to this repo.
