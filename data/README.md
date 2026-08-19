# Data directory

| Path | In git | Description |
|------|--------|-------------|
| `sioux-falls.net.xml` | yes (repo root; linked here) | Sioux Falls SUMO network |
| `EstimatedODMatrix.npy` | yes (repo root; linked here) | Base 24×24 estimated OD |
| `synthetic_od_100000.npy` | no (symlink into `outputs/`) | 100k synthetic OD tensor when generated |

Large generated arrays live under `../outputs/` and are gitignored.

If you still have artifacts only in an older checkout, restore them with:

```bash
bash scripts/setup_local_data.sh
```
