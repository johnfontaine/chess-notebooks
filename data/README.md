# Data Directory

This directory stores all game data, baselines, and analysis outputs. Contents are git-ignored; only README files are tracked to document the expected structure.

## Directory Structure

```text
data/
├── my-games/              # Personal game files in PGN format
├── other-users/           # Individual player analysis data (Chess.com API)
│   └── <username>/
│       └── report/        # Fairness report outputs (HTML, JSON, notebooks)
├── trusted/               # Trusted player baseline statistics
│   ├── config.json        # List of trusted players and settings
│   ├── combined_baseline.json
│   └── <username>/        # Per-player game data and metrics
├── cheaters/              # Known cheater baseline statistics
│   ├── config.json        # List of known cheaters
│   ├── combined_baseline.json
│   └── <username>/        # Per-player game data and metrics
├── titled-cheaters/       # Titled players banned from Titled Tuesday
├── expanded-cheaters/     # Consolidated cheaters from all sources (parquet)
├── engine_cache/          # Cached Stockfish evaluations (parquet)
├── game_assessments/      # Individual game assessment reports
├── puzzles/               # Lichess puzzle database
└── opponent_cache.db      # SQLite cache of opponent profiles
```

## Generating Data

### Baselines

```bash
# Generate trusted player baseline
python scripts/generate_baseline.py --config data/trusted/config.json --output data/trusted/

# Generate cheater baseline
python scripts/generate_baseline.py --config data/cheaters/config.json --output data/cheaters/

# Regenerate all baselines
./scripts/regenerate_baselines.sh
```

### Player Analysis

```bash
# Full fairness report for a player
python fairness_report/run_analysis.py USERNAME
```

See README files in each subdirectory for details.
