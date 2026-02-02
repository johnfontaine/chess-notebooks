# Trusted Player Baseline Dataset

## Purpose

Reference data from trusted players to establish normal play patterns. This baseline is used to compare metrics from analyzed players against expected values for fair play.

## How It's Built

1. **Configure trusted players** in `config.json`
2. **Run baseline generation:**
   ```bash
   python scripts/generate_baseline.py --config data/trusted/config.json --output data/trusted/
   ```
3. **Process:**
   - Fetches 2 years of rapid/blitz games per player from Chess.com API
   - Validates opponents against fair play status (flags banned opponents)
   - Calculates per-player statistics
   - Generates combined statistics across all trusted players

## Directory Structure

```
data/trusted/
├── config.json              # List of trusted players and settings
├── combined_baseline.json   # Aggregated statistics across all players
└── <username>/              # Per-player data directories
    ├── summary.json         # Player statistics summary
    ├── games.parquet        # Game metadata (date, result, opponent, etc.)
    ├── positions.parquet    # Position-level data with evaluations
    ├── frequent_opponents.json  # Opponent analysis
    ├── elo_segments.json    # Performance by opponent rating tier
    └── sessions.json        # Play session analysis
```

## config.json Structure

```json
{
  "trusted_users": [
    "DinaBelenkaya",
    "imrosen",
    "danny",
    "...15 trusted players total"
  ],
  "time_controls": ["blitz", "rapid"],
  "months_back": 24,
  "min_games": 100
}
```

## Key Metrics in combined_baseline.json

### Elo Baseline (`elo_baseline`)
- `win_rate_mean/std` - Expected win rate
- `elo_manipulation_score_mean/std` - Rating manipulation indicator

### Result Baseline (`result_baseline`)
- `win_rate_mean/std` - Win percentage
- `checkmate_rate_mean/std` - Games ending in checkmate
- `win_checkmate_rate_mean/std` - Wins by checkmate (14.3% typical)
- `loss_checkmate_rate_mean/std` - Losses by checkmate (14.0% typical)
- `result_entropy_mean/std` - Diversity of game outcomes

### Termination Baseline (`termination_baseline`)
- `resignation_rate_mean/std` - Games ending by resignation
- `timeout_rate_mean/std` - Games ending by timeout
- `draw_rate_mean/std` - Draw percentage

### Upset Baseline (`upset_baseline`) - To Be Added
- `upset_rate_mean/std` - Rate of wins against higher-rated opponents
- `major_upset_rate_mean/std` - Rate of significant upsets (>200 Elo)
- `losses_to_lower_rate_mean/std` - Rate of losses to lower-rated opponents

## Trusted Player Selection Criteria

Players in the baseline were selected based on:
1. **Verified identity** - Known streamers, titled players, public figures
2. **Good standing** - No fair play violations
3. **Sufficient games** - 100+ games in time controls
4. **Active accounts** - Recent activity within analysis period

## Usage

The baseline is loaded in analysis notebooks:
```python
with open('data/trusted/combined_baseline.json') as f:
    trusted_baseline = json.load(f)

# Compare player metrics against baseline
if player_loss_cm_rate < trusted_baseline['result_baseline']['loss_checkmate_rate_mean']:
    flags.append("Low loss-by-checkmate rate")
```

## Expanding the Baseline

To add more trusted users:
1. Run `scripts/expand_trusted_baseline.py` to find candidates from frequent opponents
2. Review candidates (accounts 3+ years old, not banned)
3. Add approved usernames to `config.json`
4. Re-run `generate_baseline.py`
