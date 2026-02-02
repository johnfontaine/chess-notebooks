# Chess Analysis Library API Reference

This document provides a comprehensive reference for the `chess_analysis` Python library, which provides tools for chess game analysis, player fairness evaluation, and cheat detection.

## Table of Contents

1. [Installation & Setup](#installation--setup)
2. [Core Modules](#core-modules)
3. [Metrics & Definitions](#metrics--definitions)
4. [Engine Analysis](#engine-analysis)
5. [Position Assessment](#position-assessment)
6. [Game Metrics](#game-metrics)
7. [Maia2 Analysis](#maia2-analysis)
8. [Ken Regan-esque Analysis](#ken-regan-esque-analysis)
9. [Time Analysis](#time-analysis)
10. [Tablebase Analysis](#tablebase-analysis)
11. [Data Structures](#data-structures)

---

## Installation & Setup

```python
# Activate virtual environment
source .venv/bin/activate

# Import the library
from chess_analysis import (
    EngineAnalyzer,
    calculate_acpl,
    calculate_game_accuracy,
    # ... other imports
)
```

### Requirements

- Python 3.12+
- Stockfish engine installed at `/opt/homebrew/bin/stockfish` (configurable)
- Optional: Maia2 models in `maia2_models/` directory

---

## Core Modules

| Module | Description |
|--------|-------------|
| `engine.py` | Stockfish engine integration and multi-depth analysis |
| `metrics.py` | CPL, accuracy, win percentage calculations |
| `position_assessment.py` | Comprehensive position evaluation |
| `complexity.py` | Position complexity scoring |
| `fragility.py` | Graph-theoretic fragility analysis |
| `maia2_analysis.py` | Human-like move prediction |
| `regan_analysis.py` | Ken Regan-esque IPR and Z-score |
| `time_analysis.py` | Time usage pattern detection |
| `tablebase.py` | Endgame tablebase analysis |
| `game_phase.py` | Opening/middlegame/endgame detection |
| `themes.py` | Tactical and positional theme detection |
| `openings.py` | Opening book lookup |
| `dataset.py` | Dataset building and aggregation |
| `visualization.py` | SVG board rendering |
| `caching.py` | Game and opponent caching |
| `engine_cache.py` | Engine evaluation caching |
| `baseline.py` | Player baseline generation |
| `glicko2.py` | Glicko-2 rating calculations |

---

## Metrics & Definitions

### Evaluation Metrics

| Metric | Definition | Range | Interpretation |
|--------|------------|-------|----------------|
| **CPL (Centipawn Loss)** | Difference between best move eval and played move eval | 0 - 1000+ cp | Lower = better; 0 = best move |
| **ACPL (Average CPL)** | Mean CPL across all moves in a game | 0 - 200+ cp | <20 = excellent, <40 = good, >80 = poor |
| **Accuracy** | Lichess formula based on win% preservation | 0 - 100% | Higher = better move quality |
| **Win Percentage** | Expected win probability from eval | 0 - 100% | 50% = equal position |

### Complexity Metrics

| Metric | Definition | Range | Interpretation |
|--------|------------|-------|----------------|
| **Eval Volatility** | Std dev of evals across depths [6,10,14,20] | 0 - 500+ cp | >150cp = very unstable |
| **SF Branching Factor** | Engine search branching from node counts | 1.5 - 10 | Typical: 2-4 |
| **Raw Branching Factor** | Brute-force legal move tree at 3-ply depth | 5 - 40+ | ~20 for opening positions |
| **Gap to 2nd Best** | Eval difference between best and 2nd best move | 0 - 500+ cp | <50cp = many viable moves |
| **Complexity Score** | Composite score from engine heuristics | 0 - 1 (0-100%) | Categories: LOW/MEDIUM/HIGH/VERY_HIGH |

### Position Metrics

| Metric | Definition | Range | Interpretation |
|--------|------------|-------|----------------|
| **Fragility** | Sum of betweenness centrality of attacked pieces | 0 - 1+ | High = cascade-prone position |
| **Pure Material** | Material count (P=1, N/B=3, R=5, Q=9) | -39 to +39 | From White's perspective |
| **Game Phase** | Position classification | opening/middlegame/endgame | Based on piece count and development |
| **Distance from Book** | Plys since last book move | 0 - 100+ | 0 = still in book |

### Human-like Metrics

| Metric | Definition | Range | Interpretation |
|--------|------------|-------|----------------|
| **Humanness** | Maia2 probability for played move | 0 - 100% | Higher = more human-like |
| **Maia Rank** | Rank of played move in Maia2 predictions | 1 - N | 1 = most human move |
| **Num Human Moves** | Moves with >=1% Maia2 probability | 1 - 10+ | More = more options |
| **CP Adjustment** | Eval difference vs top Maia2 move | -500 to +500 cp | Negative = played better than human |

### Fair Play Metrics

| Metric | Definition | Range | Interpretation |
|--------|------------|-------|----------------|
| **IPR** | Intrinsic Performance Rating | 0 - 3500+ | Player's effective strength |
| **Z-Score** | Std deviations above expected | -3 to +5+ | >2.0 = suspicious, >3.0 = highly suspicious |
| **Move Match Rate** | % of moves matching engine top choice | 0 - 100% | >60% across games = suspicious |
| **Tablebase Accuracy** | % optimal moves in TB positions | 0 - 100% | Perfect play in known positions |

---

## Engine Analysis

### EngineAnalyzer

Main class for Stockfish integration.

```python
from chess_analysis import EngineAnalyzer

# Basic usage
with EngineAnalyzer("/opt/homebrew/bin/stockfish", depth=20) as analyzer:
    result = analyzer.analyze(board, depth=20)
    print(f"Eval: {result['score']} cp")
    print(f"Best move: {result['pv'][0]}")
```

#### Methods

| Method | Description | Returns |
|--------|-------------|---------|
| `analyze(board, depth)` | Single-depth analysis | `dict` with score, pv, wdl |
| `analyze_multi_depth(board, depths)` | Multi-depth analysis | `MultiDepthResult` |
| `analyze_multi_depth_extended(board, depths, multipv, capture_search_stats)` | Extended analysis with complexity heuristics | `MultiDepthResult` with `complexity_heuristics` |

#### MultiDepthResult Fields

```python
@dataclass
class MultiDepthResult:
    fen: str
    depths: list[int]
    evaluations: dict[int, int]       # depth -> centipawns
    best_moves: dict[int, str]        # depth -> UCI move
    pvs: dict[int, list[str]]         # depth -> PV moves
    move_consistency: float           # 0-1, higher = more stable
    first_consistent_depth: int       # depth where move stabilized
    eval_swing: int                   # max - min eval across depths
    depth_transitions: list[DepthTransition]
    search_metrics: dict[int, EngineSearchMetrics]  # nodes, nps, time
    gap_metrics: dict[int, GapMetricResult]         # gap to 2nd best
    complexity_heuristics: PositionComplexityHeuristics
```

#### PositionComplexityHeuristics Fields

```python
@dataclass
class PositionComplexityHeuristics:
    eval_volatility: float            # Std dev of evals (centipawns)
    eval_volatility_normalized: float # 0-1 scale
    gap_at_max_depth: int             # Gap in cp at highest depth
    avg_gap: float                    # Average gap across depths
    convergence_depth: Optional[int]  # Depth where eval stabilized
    total_nodes: int                  # Total nodes searched
    nodes_per_depth: dict[int, int]   # Nodes at each depth
    branching_factor_estimate: float  # SF branching factor
    complexity_score: float           # 0-1 composite score
    complexity_category: str          # LOW/MEDIUM/HIGH/VERY_HIGH
```

---

## Position Assessment

### Functions

#### calculate_brute_force_branching

Calculates raw branching factor by counting all legal moves at 3-ply depth.

```python
from chess_analysis import calculate_brute_force_branching

result = calculate_brute_force_branching(board, depth=3)
print(f"Branching factor: {result['branching_factor']:.2f}")
print(f"Nodes at depth 1: {result['nodes_by_depth'][0]}")
print(f"Nodes at depth 2: {result['nodes_by_depth'][1]}")
print(f"Nodes at depth 3: {result['nodes_by_depth'][2]}")
print(f"Total nodes: {result['total_nodes']}")
```

**Returns:**
```python
{
    'nodes_by_depth': [20, 400, 8902],  # Legal positions at each depth
    'total_nodes': 9322,
    'branching_factor': 20.73,          # Geometric mean
    'initial_legal_moves': 20,
}
```

#### calculate_pure_material

Calculates material balance using standard piece values.

```python
from chess_analysis import calculate_pure_material

material = calculate_pure_material(board)
# Returns: positive = White advantage, negative = Black advantage
# P=1, N=3, B=3, R=5, Q=9
```

#### assess_position

Comprehensive position assessment combining all metrics.

```python
from chess_analysis import assess_position

result = assess_position(
    board=board,
    move=move,
    ply=ply,
    analyzer=engine_analyzer,
    maia2_result=maia_result,  # Optional
)
```

---

## Game Metrics

### Centipawn Loss

```python
from chess_analysis import calculate_centipawn_loss, calculate_acpl

# Single move CPL
cpl = calculate_centipawn_loss(eval_before, eval_after, is_white_move)

# Game average
acpl = calculate_acpl(move_cpls)
```

### Accuracy (Lichess Formula)

```python
from chess_analysis import (
    centipawns_to_win_percent,
    calculate_move_accuracy,
    calculate_game_accuracy,
)

# Win percentage from eval
win_pct = centipawns_to_win_percent(centipawns)
# Formula: 50 + 50 * (2 / (1 + exp(-0.00368208 * cp)) - 1)

# Move accuracy
accuracy = calculate_move_accuracy(win_pct_before, win_pct_after)
# Returns 0-100% based on win% preservation

# Game accuracy
game_acc = calculate_game_accuracy(move_accuracies)
```

### Move Classification

```python
from chess_analysis import classify_move_by_cpl, classify_move

# Simple CPL-based classification
move_class = classify_move_by_cpl(cpl)
# Returns: "best" | "excellent" | "good" | "inaccuracy" | "mistake" | "blunder"

# Context-aware classification (includes Critical Mistake, Missed Win, etc.)
move_class = classify_move(cpl, eval_before, eval_after, is_white_move)
```

**Classification Thresholds:**

| Classification | CPL Range |
|---------------|-----------|
| Best | 0 |
| Excellent | 1-9 |
| Good | 10-24 |
| Inaccuracy | 25-49 |
| Mistake | 50-299 |
| Blunder | 300+ |

---

## Maia2 Analysis

Maia2 provides human-like move predictions calibrated to different rating levels.

```python
from chess_analysis import analyze_position_maia2, Maia2Result

result = analyze_position_maia2(
    fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    played_move="e2e4",
    elo_self=1500,
    elo_oppo=1500,
    game_type="blitz",  # or "rapid"
)

print(f"Humanness: {result.move_probability * 100:.1f}%")
print(f"Rank: {result.move_rank}")
print(f"Top Maia move: {result.top_move}")
print(f"Top probability: {result.top_move_probability * 100:.1f}%")
```

### Maia2Result Fields

```python
@dataclass
class Maia2Result:
    move_probability: float      # Probability of played move (0-1)
    move_rank: int               # Rank of played move (1 = most human)
    top_move: str                # Most human move (UCI)
    top_move_probability: float  # Probability of top move
    all_probabilities: dict[str, float]  # All move probabilities
```

---

## Ken Regan-esque Analysis

Analysis inspired by Ken Regan's FIDE cheat detection methodology.

```python
from chess_analysis import (
    analyze_game_regan,
    calculate_z_score,
    identify_suspicious_positions,
    ReganAnalysisResult,
)

# Analyze a game
result = analyze_game_regan(
    game=game,  # chess.pgn.Game
    analyzer=engine_analyzer,
    player_elo=1500,
)

print(f"IPR: {result.ipr}")
print(f"Z-Score: {result.z_score:.2f}")
print(f"Move Match Rate: {result.move_match_rate * 100:.1f}%")

# Get suspicious positions
suspicious = identify_suspicious_positions(result)
for pos in suspicious:
    print(f"Move {pos.ply}: {pos.move} - Suspicion: {pos.suspicion_score:.2f}")
```

### Z-Score Interpretation

| Z-Score | Interpretation |
|---------|---------------|
| < 1.0 | Normal play |
| 1.0 - 2.0 | Above average |
| 2.0 - 3.0 | Warrants investigation |
| > 3.0 | Highly suspicious |

---

## Time Analysis

```python
from chess_analysis import (
    extract_clock_times,
    classify_time_spent,
    analyze_time_distribution,
    detect_bot_patterns,
)

# Extract clock times from PGN
clock_times = extract_clock_times(game)

# Classify individual move time
classification = classify_time_spent(seconds_spent)
# Returns: "instant" | "quick" | "short" | "normal" | "long" | "very_long"

# Analyze distribution
distribution = analyze_time_distribution(move_times)

# Detect bot-like patterns
bot_score = detect_bot_patterns(move_times)
```

### Time Classification Thresholds

| Classification | Duration |
|---------------|----------|
| Instant | < 1 second |
| Quick | 1-3 seconds |
| Short | 3-10 seconds |
| Normal | 10-30 seconds |
| Long | 30-60 seconds |
| Very Long | > 60 seconds |

---

## Tablebase Analysis

Analysis of endgame positions using Syzygy tablebases (via Lichess API).

```python
from chess_analysis import (
    TablebaseClient,
    probe_tablebase,
    check_tablebase_move,
    analyze_endgame_accuracy,
)

# Check if position is in tablebase (<=7 pieces)
from chess_analysis import is_tablebase_position
if is_tablebase_position(board):
    result = probe_tablebase(board)
    print(f"WDL: {result.wdl}")  # win/draw/loss
    print(f"DTZ: {result.dtz}")  # distance to zeroing

# Check if a move maintains the result
move_check = check_tablebase_move(board, move)
print(f"Optimal: {move_check.is_optimal}")

# Analyze endgame accuracy for a game
accuracy = analyze_endgame_accuracy(game, player_color)
print(f"TB Accuracy: {accuracy * 100:.1f}%")
```

---

## Data Structures

### GameMetadata

```python
@dataclass
class GameMetadata:
    game_id: str
    white_username: str
    black_username: str
    white_elo: int
    black_elo: int
    time_control: str
    result: str  # "1-0", "0-1", "1/2-1/2"
    termination: str
    date: datetime
```

### FragilityAnalysis

```python
@dataclass
class FragilityAnalysis:
    scores: list[float]      # Fragility score per ply
    peaks: list[int]         # Ply indices of local maxima
    valleys: list[int]       # Ply indices of local minima
    max_fragility: float     # Maximum fragility score
    max_fragility_ply: int   # Ply of maximum fragility
```

### OpeningInfo

```python
@dataclass
class OpeningInfo:
    eco: str        # e.g., "B90"
    name: str       # e.g., "Sicilian Defense: Najdorf Variation"
    pgn: str        # Move sequence
    ply_count: int  # Number of half-moves
```

---

## Common Workflows

### Analyzing a Single Game

```python
import chess.pgn
from chess_analysis import (
    EngineAnalyzer,
    calculate_acpl,
    analyze_game_regan,
    analyze_position_maia2,
)

# Load game
with open("game.pgn") as f:
    game = chess.pgn.read_game(f)

# Analyze with Stockfish
with EngineAnalyzer("/opt/homebrew/bin/stockfish", depth=20) as analyzer:
    # Calculate ACPL
    cpls = []
    board = game.board()
    for move in game.mainline_moves():
        result = analyzer.analyze(board, depth=20)
        # ... calculate CPL
        board.push(move)

    acpl = calculate_acpl(cpls)
    print(f"ACPL: {acpl:.1f}")

    # Ken Regan-esque analysis
    regan_result = analyze_game_regan(game, analyzer, player_elo=1500)
    print(f"Z-Score: {regan_result.z_score:.2f}")
```

### Building a Player Baseline

```python
from chess_analysis import generate_player_baseline

baseline = generate_player_baseline(
    username="player123",
    time_classes=["blitz", "rapid"],
    max_games=1000,
)

print(f"Games analyzed: {baseline['total_games']}")
print(f"Average ACPL: {baseline['avg_acpl']:.1f}")
print(f"Average Accuracy: {baseline['avg_accuracy']:.1f}%")
```

---

## Error Handling

Most functions return `None` or default values on errors rather than raising exceptions. Check return values:

```python
result = analyze_position_maia2(...)
if result is None:
    print("Maia2 analysis failed")
else:
    print(f"Humanness: {result.move_probability}")
```

For engine analysis, use context managers to ensure proper cleanup:

```python
with EngineAnalyzer(engine_path) as analyzer:
    # ... use analyzer
# Engine automatically closed
```

---

## Performance Tips

1. **Use caching**: Engine evaluations are cached in `data/engine_cache/`
2. **Batch analysis**: Use `analyze_position_complexity_batch()` for multiple positions
3. **Limit depth**: Use depth 12-14 for move ranking (fast), depth 20+ for final evaluation
4. **Parallel execution**: Run independent analyses in separate processes

---

## Version History

- **v1.0**: Initial release with CPL, accuracy, fragility
- **v1.1**: Added Maia2 integration
- **v1.2**: Added Ken Regan-esque analysis
- **v1.3**: Added multi-depth complexity heuristics
- **v1.4**: Added brute-force branching factor, improved documentation
