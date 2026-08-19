# OD Estimation Benchmark Report (Phase 3)

## Configuration
- Seed: 42
- Max samples: 50000
- Clean turning counts: True
- Standardize Y: True

## Leaderboard

| Model | MAE | RMSE | R² | Correlation | Production MAE | Attraction MAE | Forward RMSE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| residual_mlp_1blk | 645.2904 | 1102.0412 | 0.3751 | 0.6307 | 1124.9018 | 1122.6935 | 397.1566 |
| ridge | 662.0275 | 1109.3181 | 0.3789 | 0.6267 | 1563.2421 | 1560.8378 | 562.4203 |

**Best model:** residual_mlp_1blk
