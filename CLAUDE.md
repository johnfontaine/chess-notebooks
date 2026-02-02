# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains Jupyter notebooks and Python scripts for analyzing chess games, with a focus on player fairness evaluation using metrics like centipawn loss, Lichess accuracy, and Ken Regan's method. The analysis draws from academic research on cheat detection and fair play in online chess.

## Environment Setup

- Python 3.12 virtual environment located in `.venv/`
- Activate: `source .venv/bin/activate`
- Install packages: `pip3 install <package>` (or `python -m pip`)

## Running Notebooks

```bash
source .venv/bin/activate
jupyter notebook
```

## Main Notebooks

**`updated-player-fairness.ipynb`** - The original monolithic analysis notebook (archived for reference).

**`fairness_report/`** - Modular fairness analysis system (recommended). See section below.

## Data Structure

- `data/my-games/` - Personal game files in PGN format (dated files like `2025-09-26.pgn`)
- `data/other-users/<username>/` - Games from other users pulled via chess.com pub data API
  - `games.parquet` - Game metadata with `opponent_is_banned` flag
  - `positions.parquet` - Position-level data
  - `report/` - Fairness report outputs (HTML, JSON, executed notebooks)
- `data/trusted/` - Trusted player baseline statistics
- `data/cheaters/` - Known cheater statistics for comparison
- `data/titled-cheaters/` - Dataset of titled players banned from Titled Tuesday
- `data/expanded-cheaters/` - Consolidated cheaters from all sources (parquet)
- `data/engine_cache/` - Cached Stockfish evaluations (parquet)
- `data/opponent_cache.db` - SQLite cache of opponent profiles (shared across baselines)

## Chess Analysis Module (`chess_analysis/`)

Core analysis library with the following components:

### Engine Analysis (`engine.py`)
- `EngineAnalyzer` - Multi-depth Stockfish analysis
- `MultiDepthResult` - Results from multi-depth evaluation

### Metrics (`metrics.py`)
- `calculate_acpl()` - Average centipawn loss
- `calculate_game_accuracy()` - Lichess-style accuracy (win percentage based)
- `classify_move_by_accuracy()` - Classify moves: Best/Excellent/Good/Inaccuracy/Mistake/Blunder
- `classify_advantage()` - Position advantage: Winning/Slight Advantage/Equal/Slight Disadvantage/Losing
- `analyze_errors_by_advantage()` - Error rates by position type

### Time Analysis (`time_analysis.py`)
- `extract_clock_times()` - Parse clock data from PGN
- `analyze_time_patterns()` - Time usage statistics
- `classify_time_spent()` - Classify: Instant/Quick/Short/Normal/Long/Very Long
- `analyze_time_distribution()` - Distribution by time classification
- `detect_bot_patterns()` - Bot-like timing detection

### Ken Regan Analysis (`regan_analysis.py`)
- `analyze_game_regan()` - IPR (Ideal Performance Rating) and Z-score calculation
- FIDE-style cheat detection methodology

### Tablebase (`tablebase.py`)
- `TablebaseClient` - Lichess tablebase API client
- `analyze_endgame_accuracy()` - Single game endgame analysis
- `analyze_tablebase_consistency()` - Multi-game tablebase accuracy report

### Dataset Building (`dataset.py`)
- `build_game_dataset()` - Extract game metadata
- `build_position_dataset()` - Extract positions from games
- `analyze_elo_patterns()` - Elo distribution analysis
- `analyze_opponent_segments()` - Performance by opponent strength tier
- `EloSegmentStats` - Statistics for Player Favored/Fair Match/Opponent Favored segments

### Visualization (`visualization.py`)
- `render_position()` - SVG board rendering with arrows
- `render_key_position()` - Annotated key position rendering
- `find_key_positions()` - Find fragile peaks, complex positions, mistakes, brilliant moves
- Arrow colors: green (#15781B) for played, blue (#003088/#0066ff) for best, red (#882020) for traps

### Other Modules
- `fragility.py` - Position fragility calculation
- `complexity.py` - Position complexity metrics
- `themes.py` - Tactical/positional theme detection
- `game_phase.py` - Opening/middlegame/endgame detection
- `games.py` - Chess.com API fetching, PGN parsing

## Scripts

- `scripts/generate_baseline.py` - Generate baseline statistics from trusted players
- `scripts/find_titled_cheaters.py` - Find titled players banned from Titled Tuesday tournaments
- `scripts/regenerate_baselines.sh` - Regenerate all baselines (trusted + cheaters)
- `scripts/refresh_opponent_cache.py` - Refresh stale/null entries in opponent cache
- `scripts/build_expanded_cheaters.py` - Build consolidated cheaters dataset

## Key References

The analysis methodology is based on:

- Centipawn loss evaluation via [PGN Spy](https://github.com/MGleason1/PGN-Spy)
- [Chess.com pub data API](https://www.chess.com/news/view/published-data-api)
- Lichess accuracy formula (win percentage based)
- Ken Regan's FIDE cheat detection methodology
- Academic papers on fair play detection and position fragility in chess

## Fairness Report System (`fairness_report/`)

Modular notebook system for player fairness analysis. Each phase can run independently with proper inputs.

### Structure

- `run_analysis.py` - CLI runner: `python fairness_report/run_analysis.py USERNAME`
- `master.ipynb` - Interactive orchestrator notebook
- `common.py` - Shared setup (imports from `chess_analysis/`, path helpers)
- `phases/` - Individual analysis notebooks (can run independently)
- `templates/` - Jinja2 HTML report templates

### Phases

1. **01_data_collection.ipynb** - Fetch games from Chess.com API
2. **02_quick_analysis.ipynb** - Elo/result patterns, session analysis
3. **03_game_prioritization.ipynb** - Score games for suspiciousness
4. **04a_engine_analysis.ipynb** - Multi-depth Stockfish analysis
5. **04b_regan_analysis.ipynb** - Ken Regan IPR/Z-score
6. **04c_tablebase_analysis.ipynb** - Endgame accuracy
7. **05_time_analysis.ipynb** - Time usage patterns
8. **06_maia2_analysis.ipynb** - Maia2 humanness scoring
9. **07_cheater_comparison.ipynb** - Compare vs cheater baseline
10. **08_final_report.ipynb** - Generate HTML report

### Output Location

Reports are saved to `data/other-users/{username}/report/`

### Running

```bash
# Full analysis via CLI
python fairness_report/run_analysis.py USERNAME

# Or interactively via master notebook
jupyter notebook fairness_report/master.ipynb

# Or run individual phases
jupyter notebook fairness_report/phases/01_data_collection.ipynb
```

### Notes

- Each notebook has a `parameters` cell for Papermill injection
- `common.py` re-exports `chess_analysis` module - no code duplication
- Engine evaluations are cached in `data/engine_cache/` for reuse across runs

## Expanded Cheaters Dataset (`data/expanded-cheaters/`)

Consolidated dataset of all known cheaters from multiple sources:

1. `data/cheaters/config.json` - Manually curated cheaters
2. `data/opponent_cache.db` - Banned opponents encountered during baseline generation
3. `data/titled-cheaters/titled_cheaters.json` - Titled players banned from Titled Tuesday

### Schema

`cheaters.parquet` columns:

- `username` - Chess.com username
- `status` - Account status (e.g., "closed:fair_play_violations")
- `sources` - Comma-separated list of sources where player appears
- `title` - Chess title (GM, IM, FM, etc.) or null
- `joined` - Unix timestamp of account creation
- `last_online` - Unix timestamp of last activity (= ban date for closed accounts)
- `days_cheating` - Days between account creation and ban
- `rating_blitz`, `rating_rapid`, `rating_bullet` - Ratings at time of ban
- `category_blitz`, `category_rapid`, `category_bullet` - Player category (title or rating bucket)

### Player Categories

- Titled players: Use their title (GM, IM, FM, CM, NM, etc.)
- Untitled players: Rating bucket = `int(rating / 200)` (e.g., 1655 -> "8")

### Building

```bash
# Build expanded cheaters dataset
python scripts/build_expanded_cheaters.py

# With API refresh for missing data
python scripts/build_expanded_cheaters.py --refresh-missing

# Or via regenerate_baselines.sh
./scripts/regenerate_baselines.sh --include-expanded
```

### Refreshing Opponent Cache

```bash
# Show cache statistics
python scripts/refresh_opponent_cache.py --stats

# Fix entries with null last_online/title/player_id
python scripts/refresh_opponent_cache.py --fix-nulls

# Refresh banned accounts only
python scripts/refresh_opponent_cache.py --fix-nulls --banned-only
```
