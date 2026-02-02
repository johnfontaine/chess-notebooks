# Engine Cache

Cached Stockfish evaluations to avoid recomputation across analysis runs.

## Files

| File | Description |
|------|-------------|
| `move_evals.parquet` | Individual move evaluations |
| `position_evals.parquet` | Position-level evaluations |
| `gap_metrics.parquet` | Gap/fragility metrics |

These files are generated automatically during engine analysis phases and shared across all player reports.
