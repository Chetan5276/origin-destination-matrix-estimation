# Synthetic OD Dataset Summary

## Configuration

- **input**: EstimatedODMatrix.npy
- **samples**: 100000
- **requested_samples**: 100000
- **n_candidates**: 200000
- **quality_filters**: True
- **alpha**: 500.0
- **perturbation**: 0.0
- **epsilon_beta**: 1.0
- **epsilon_lambda**: 500.0
- **gamma_shape**: 0.5
- **gamma_scale**: 1.0
- **apply_gamma_mask**: True
- **ipf_tol**: 0.001
- **seed**: 42
- **workers**: 20
- **dtype**: <class 'numpy.float32'>
- **eval_sample_size**: 5000
- **filter_config**: {'oversample_factor': 2.0, 'min_sparsity': 0.0, 'max_sparsity': 0.95, 'min_correlation': 0.3, 'max_correlation': 0.95, 'max_trip_length_rel_error': 0.35, 'min_frobenius_distance': 0.0, 'auto_frobenius_fraction': 0.35, 'entropy_bins': 20, 'require_zero_diagonal': True, 'max_diagonal_mass': 1e-06}
- **filter_stats**: {'n_candidates': 200000, 'n_accepted': 100000, 'n_rejected_sparsity': 0, 'n_rejected_correlation': 1, 'n_rejected_trip_length': 0, 'n_rejected_diagonal': 0, 'n_rejected_pairwise': 0, 'n_rejected_pool_exhausted': 0, 'mean_cell_entropy': 1.1698114528466463, 'mean_cell_cv': 2.4456722739986656, 'mean_pairwise_frobenius': 42851.08321914817, 'min_pairwise_frobenius': 33760.57371292537}

## Dataset Size
- Samples: **100,000**

## Similarity (mean over samples)
- Correlation: **0.4708**
- MAE: **680.59**
- Relative error: **1.0726**

## Diversity
- Mean pairwise distance: **865.82**
- Min / max pairwise distance: **714.97** / **983.47**
- Mean cell CV: **2.4457**

## Demand Consistency
- Demand CV: **2.59e-09**

## Structural Validity
- Max new connections: **0**
- Max diagonal violation: **0.00e+00**

## PCA
- Components for 90% variance: **21**
- PC1 explained variance: **0.0277**

## KMeans Inertia
- k=2: 4500952237476.47
- k=5: 4317263217264.81
- k=10: 4170063392348.62
