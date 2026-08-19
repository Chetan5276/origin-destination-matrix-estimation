# Outputs

Pipeline artifacts written by the OD generator, turning-count builder, and ML trainer.

Typical subdirectories:

- `od_generator/`, `od_generator_100k/`, `od_generator_fp/` — synthetic OD datasets
- `turning_counts_100k/`, `turning_counts_fp/`, `turning_counts_1m/` — assignment matrices and counts
- `ml_neural/`, `ml_100k_ridge_resmlp/`, `ml_fp_ridge_resmlp/` — training reports and models

`.npy`, `.pkl`, and trained weights are **not** tracked in git. Markdown/JSON summaries may be committed.
