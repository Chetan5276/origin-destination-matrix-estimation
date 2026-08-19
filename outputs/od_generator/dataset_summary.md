# Synthetic OD Dataset Summary

## Configuration

- **input**: /home/chetan/anaconda3/envs/ODEstimation/EstimatedODMatrix.npy
- **samples**: 50
- **requested_samples**: 50
- **n_candidates**: 250
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
- **eval_sample_size**: 10000
- **filter_config**: {'oversample_factor': 5.0, 'min_sparsity': 0.0, 'max_sparsity': 0.95, 'min_correlation': 0.3, 'max_correlation': 0.95, 'max_trip_length_rel_error': 0.35, 'min_frobenius_distance': 0.0, 'auto_frobenius_fraction': 0.35, 'entropy_bins': 20, 'require_zero_diagonal': True, 'max_diagonal_mass': 1e-06}
- **filter_stats**: {'n_candidates': 250, 'n_accepted': 50, 'n_rejected_sparsity': 0, 'n_rejected_correlation': 0, 'n_rejected_trip_length': 0, 'n_rejected_diagonal': 0, 'n_rejected_pairwise': 0, 'n_rejected_pool_exhausted': 0, 'mean_cell_entropy': 1.3468555843772272, 'mean_cell_cv': 2.0030727683264207, 'mean_pairwise_frobenius': 44313.86768508055, 'min_pairwise_frobenius': 35607.645935197725}

## Dataset Size
- Samples: **50**

## Similarity (mean over samples)
- Correlation: **0.4578**
- MAE: **695.19**
- Relative error: **1.0956**

## Diversity
- Mean pairwise distance: **880.51**
- Min / max pairwise distance: **758.92** / **987.78**
- Mean cell CV: **2.0031**

## Demand Consistency
- Demand CV: **2.26e-09**

## Structural Validity
- Max new connections: **0**
- Max diagonal violation: **0.00e+00**

## PCA
- Components for 90% variance: **21**
- PC1 explained variance: **0.0670**

## KMeans Inertia
- k=2: 46130899998.16
- k=5: 41162577439.21
- k=10: 35655947312.36
