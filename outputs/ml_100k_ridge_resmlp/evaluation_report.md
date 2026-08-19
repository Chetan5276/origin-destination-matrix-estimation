# OD Estimation Benchmark Report (Phase 3)

## Configuration
- Seed: 42
- Max samples: 50000
- Clean turning counts: True
- Standardize Y: True

## Leaderboard

| Model | MAE | RMSE | R² | Correlation | Production MAE | Attraction MAE | Forward RMSE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| residual_mlp_1blk | 434.9342 | 781.7900 | 0.4201 | 0.8431 | 246.9686 | 248.5004 | 194.3255 |
| ridge | 483.4611 | 871.3330 | 0.3843 | 0.8015 | 693.0201 | 693.0201 | 306.2410 |

**Best model:** residual_mlp_1blk
