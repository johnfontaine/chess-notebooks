# Expanded Cheaters Dataset

Consolidated dataset of all known cheaters from multiple sources.

## Sources

1. `data/cheaters/config.json` - Manually curated cheaters
2. `data/opponent_cache.db` - Banned opponents encountered during baseline generation
3. `data/titled-cheaters/titled_cheaters.json` - Titled players banned from Titled Tuesday

## Output

`cheaters.parquet` with columns:

| Column | Description |
|--------|-------------|
| `username` | Chess.com username |
| `status` | Account status (e.g. `closed:fair_play_violations`) |
| `sources` | Comma-separated list of sources |
| `title` | Chess title (GM, IM, FM, etc.) or null |
| `joined` | Unix timestamp of account creation |
| `last_online` | Unix timestamp of last activity (ban date for closed accounts) |
| `days_cheating` | Days between account creation and ban |
| `rating_blitz`, `rating_rapid`, `rating_bullet` | Ratings at time of ban |
| `category_blitz`, `category_rapid`, `category_bullet` | Player category (title or rating bucket) |

## Building

```bash
python scripts/build_expanded_cheaters.py

# With API refresh for missing data
python scripts/build_expanded_cheaters.py --refresh-missing
```
