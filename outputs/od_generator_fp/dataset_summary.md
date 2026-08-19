# First-Principles OD Generation Summary

- Output: `outputs/od_generator_fp/synthetic_od_fp.npy`
- Shape: `(100000, 24, 24)`
- Total demand G: `365475.0` (CV=2.55e-09)
- Latent dim: `4`, decay: `exponential`, reciprocity: `0.35`
- Gamma mask: `True`, Dirichlet α: `200.0`
- Mean pairwise L1: `586918.71`
- Total time: `77.23s`

Next: generate turning counts with the existing Phase-2 pipeline, e.g.
```
python -m src.data.generate_turning_counts --network /home/chetan/anaconda3/envs/ODEstimation/sioux-falls.net.xml \
  --od outputs/od_generator_fp/synthetic_od_fp.npy --output-dir outputs/turning_counts_fp
```
