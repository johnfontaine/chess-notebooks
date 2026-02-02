# Individual Player Analysis Data

## Purpose

Stores cached game data for specific players being analyzed. This directory holds data for players who are neither in the trusted baseline nor the known cheater dataset - typically players under investigation or for comparative analysis.

## Directory Structure

```
data/other-users/
└── <username>/
    ├── games_cache.json     # Cached Chess.com API responses
    ├── games.parquet        # Processed game data (if analyzed)
    └── positions.parquet    # Position evaluations (if deep analysis run)
```

## How Data Is Created

Data is fetched when analyzing a player in the notebook:

```python
# In updated-player-fairness.ipynb
username = "player_to_analyze"
games = fetch_player_games(username, cache_dir=f"data/other-users/{username}")
```

### API Calls Made

1. **Player Profile:** `/pub/player/{username}`
   - Account status, country, joined date

2. **Player Stats:** `/pub/player/{username}/stats`
   - Ratings, RD values, game counts, best ratings

3. **Game Archives:** `/pub/player/{username}/games/archives`
   - List of monthly archive URLs

4. **Monthly Games:** `/pub/player/{username}/games/{YYYY}/{MM}`
   - PGN data for each month

## Cache Structure

`games_cache.json` contains:
```json
{
  "profile": { "username": "...", "joined": 1234567890, ... },
  "stats": { "chess_blitz": { "last": { "rating": 1500, "rd": 65 }, ... } },
  "archives": ["https://api.chess.com/pub/player/.../games/2024/01", ...],
  "games": {
    "2024/01": [ { "pgn": "...", "time_control": "180", ... } ],
    "2024/02": [ ... ]
  },
  "last_updated": "2024-12-01T10:30:00Z"
}
```

## Data Retention

- Cache files are kept to avoid repeated API calls
- Old cache can be refreshed by deleting the cache file
- Large datasets may accumulate; periodically clean unused player directories

## Privacy Note

This data is fetched from Chess.com's public API. Only publicly available game data is stored. No private information or authentication is required or used.
