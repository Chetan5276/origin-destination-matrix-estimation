# OD Estimation Benchmark Report (Phase 3)

## Configuration
- Seed: 42
- Max samples: 50000
- Clean turning counts: True
- Standardize Y: True

## OD Autoencoder reconstruction RMSE (validation)
- Latent 32: 511.62
- Latent 64: 469.10
- **Best latent dim (test MAE):** 64

## Leaderboard

| Model | MAE | RMSE | R² | Correlation | Production MAE | Attraction MAE | Forward RMSE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| residual_mlp | 296.9241 | 455.2022 | 0.3593 | 0.8708 | 53.5701 | 53.0230 | 68.1184 |
| mlp | 302.0500 | 460.4718 | 0.3385 | 0.8676 | 59.1917 | 60.1977 | 94.8284 |
| mlp_residual | 341.9775 | 520.9554 | -0.8196 | 0.8294 | 82.2984 | 64.8250 | 140.9438 |
| ae_latent_64_finetuned | 355.6271 | 535.4931 | 0.1521 | 0.8160 | 17.8996 | 18.9731 | 402.3410 |
| ridge | 357.2643 | 509.5795 | 0.2825 | 0.8474 | 3019.5690 | 3019.5690 | 1026.0500 |
| ae_latent_32_finetuned | 376.3623 | 572.2610 | 0.0928 | 0.7867 | 13.7252 | 13.3005 | 618.3232 |
| ae_latent_64 | 380.0031 | 576.5793 | 0.0666 | 0.7855 | 85.6331 | 97.0611 | 627.3894 |
| ae_latent_32 | 386.4781 | 590.6973 | 0.0528 | 0.7731 | 22.3434 | 21.6617 | 728.3765 |

**Best model:** residual_mlp
