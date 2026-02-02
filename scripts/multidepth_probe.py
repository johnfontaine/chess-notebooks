#!/usr/bin/env python3
"""
Multi-Depth Probe Analysis

Samples positions from trusted players to analyze:
1. Move emergence patterns - when does the best move first appear?
2. Eval vs material - how much positional advantage exists?
3. Move complexity - how stable is the best move across depths?

This establishes baseline patterns for what "normal" human play looks like
at different depth levels, helping identify suspicious depth-dependent patterns.
"""

import argparse
import hashlib
import json
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
from datetime import datetime

import chess
import chess.pgn
import pandas as pd
import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from chess_analysis import (
    EngineAnalyzer,
    MultiDepthResult,
    DepthTransition,
    DEFAULT_MULTI_DEPTHS,
    calculate_material_score,
    detect_game_phase,
    GamePhase,
    calculate_fragility_simple,
)


# =============================================================================
# TRAP MOVE DETECTION
# =============================================================================

@dataclass
class TrapMove:
    """A move that looks good at shallow depth but drops under deeper analysis."""
    move_uci: str
    eval_at_shallow: int      # Eval at shallowest depth (e.g., depth 3)
    eval_at_deep: int         # Eval at deepest depth (e.g., depth 21)
    eval_drop: int            # How much eval dropped (shallow - deep)
    best_move_at_shallow: str # What engine preferred at shallow depth
    best_move_at_deep: str    # What engine preferred at deep depth
    was_best_at_shallow: bool # Was this move the engine's choice at shallow depth?


def find_trap_moves_in_position(
    engine: EngineAnalyzer,
    board: chess.Board,
    depths: Optional[list[int]] = None,
    min_eval_drop: int = 100,  # Must drop 100+cp from shallow to deep
    max_initial_deficit: int = 50,  # Must look "good" initially (within 50cp of best)
) -> list[TrapMove]:
    """
    Find moves that look good at shallow depth but degrade under analysis.

    A "trap move" is one that:
    1. Looks promising at shallow depth (close to best move's eval)
    2. But degrades significantly under deeper analysis

    These are moves that low-rated or time-pressured players might play,
    but engine-assisted players would avoid.

    Args:
        engine: EngineAnalyzer instance (must be started)
        board: Position to analyze
        depths: Depths to analyze at (default: DEFAULT_MULTI_DEPTHS)
        min_eval_drop: Minimum eval drop from shallow to deep to qualify as trap
        max_initial_deficit: Max deficit from best at shallow to be "good-looking"

    Returns:
        List of TrapMove objects
    """
    if depths is None:
        depths = DEFAULT_MULTI_DEPTHS

    trap_moves = []
    sorted_depths = sorted(depths)
    shallow_depth = sorted_depths[0]  # e.g., 3
    deep_depth = sorted_depths[-1]    # e.g., 21

    # First, get the position's evaluation at each depth (without making a move)
    # This tells us what the "best" eval looks like at shallow depth
    position_result = engine.analyze_multi_depth(board, depths)
    best_shallow_eval = position_result.evaluations.get(shallow_depth, 0)

    for move in board.legal_moves:
        # Make the move
        board.push(move)

        # Analyze the resulting position at multiple depths
        result = engine.analyze_multi_depth(board, depths)

        # Get evals from the moving side's perspective
        # After the move, it's opponent's turn, so we negate
        shallow_eval = -result.evaluations.get(shallow_depth, 0)
        deep_eval = -result.evaluations.get(deep_depth, 0)

        board.pop()

        # Check if this is a trap move:
        # 1. Initial eval looks "good" (close to best move's eval at shallow depth)
        # 2. Deep eval drops significantly
        initial_deficit = best_shallow_eval - shallow_eval
        eval_drop = shallow_eval - deep_eval

        if initial_deficit <= max_initial_deficit and eval_drop >= min_eval_drop:
            trap_moves.append(TrapMove(
                move_uci=move.uci(),
                eval_at_shallow=shallow_eval,
                eval_at_deep=deep_eval,
                eval_drop=eval_drop,
                best_move_at_shallow=position_result.best_moves.get(shallow_depth, ""),
                best_move_at_deep=position_result.best_moves.get(deep_depth, ""),
                was_best_at_shallow=(move.uci() == position_result.best_moves.get(shallow_depth, "")),
            ))

    return trap_moves


def analyze_position_trap_density(
    engine: EngineAnalyzer,
    board: chess.Board,
    depths: Optional[list[int]] = None,
) -> dict:
    """
    Analyze a position for how many trap moves exist.

    Returns dict with:
    - trap_count: Number of trap moves in the position
    - trap_density: trap_count / total_legal_moves
    - worst_trap: The move with biggest eval drop
    - fragility: Position fragility score
    - legal_move_count: Total legal moves
    - material_score: Material balance in centipawns
    """
    legal_moves = list(board.legal_moves)
    trap_moves = find_trap_moves_in_position(engine, board, depths)
    fragility = calculate_fragility_simple(board)
    material = calculate_material_score(board)

    worst_trap = None
    if trap_moves:
        worst_trap = max(trap_moves, key=lambda t: t.eval_drop)

    return {
        'trap_count': len(trap_moves),
        'trap_density': len(trap_moves) / len(legal_moves) if legal_moves else 0,
        'legal_move_count': len(legal_moves),
        'traps': [asdict(t) for t in trap_moves],
        'worst_trap': asdict(worst_trap) if worst_trap else None,
        'fragility': fragility,
        'material_score': material,
    }


# =============================================================================
# DEPTH EMERGENCE STATISTICS
# =============================================================================

@dataclass
class DepthEmergenceStats:
    """Statistics about when best moves first emerge."""
    depth: int
    count: int  # How many moves first became best at this depth
    pct: float  # Percentage of total moves
    avg_eval_at_emergence: float  # Average eval when move first becomes best
    stayed_best: int  # How many stayed best through max depth
    changed_later: int  # How many were replaced by deeper search


@dataclass
class MoveComplexityStats:
    """Statistics about move complexity based on depth behavior."""
    total_positions: int

    # Move depth classification
    shallow_moves: int  # Best move emerges at min depth (depth 5)
    shallow_pct: float
    deep_moves: int  # Best move only emerges at max depth (depth 21)
    deep_pct: float
    mid_depth_moves: int  # Best move emerges at intermediate depth
    mid_depth_pct: float

    # Move stability
    stable_moves: int  # Same best move at all depths
    stable_pct: float
    avg_best_move_changes: float

    # Eval swings
    avg_eval_swing: float
    max_eval_swing: int
    high_swing_count: int  # >100cp swing between depths
    high_swing_pct: float


@dataclass
class MaterialAnalysisStats:
    """Statistics about eval vs material differences."""
    total_positions: int

    # Material balance
    avg_material: float
    material_equal_count: int  # Within 50cp
    white_ahead_count: int
    black_ahead_count: int

    # Eval vs material
    avg_eval_vs_material: float  # Positional component
    positional_advantage_count: int  # Eval > material by 50+
    positional_disadvantage_count: int  # Eval < material by 50+

    # Correlation
    eval_material_correlation: float

    # By game phase
    opening_avg_positional: float
    middlegame_avg_positional: float
    endgame_avg_positional: float


@dataclass
class MultiDepthProbeResult:
    """Complete results from multi-depth probe analysis."""
    timestamp: str
    dataset: str
    games_sampled: int
    positions_analyzed: int
    depths_used: list[int]

    emergence_stats: list[DepthEmergenceStats]
    complexity_stats: MoveComplexityStats
    material_stats: MaterialAnalysisStats

    # Per-phase breakdown
    phase_complexity: dict  # GamePhase -> MoveComplexityStats

    # Example positions
    example_shallow_moves: list[dict]  # Easy tactical moves
    example_deep_moves: list[dict]  # Subtle strategic moves
    example_high_positional: list[dict]  # High eval vs material


def load_game_from_json(game_dict: dict) -> Optional[chess.pgn.Game]:
    """Convert a game dict from JSON to a chess.pgn.Game object."""
    pgn_text = game_dict.get('pgn', '')
    if not pgn_text:
        return None

    try:
        import io
        game = chess.pgn.read_game(io.StringIO(pgn_text))
        return game
    except Exception:
        return None


def sample_games_from_baseline(
    baseline_dir: Path,
    n_games: int = 20,
    seed: int = 42
) -> list[tuple[str, chess.pgn.Game]]:
    """Sample games from trusted baseline players."""
    random.seed(seed)

    games = []

    # Find all player directories
    player_dirs = [d for d in baseline_dir.iterdir() if d.is_dir()]

    if not player_dirs:
        print(f"No player directories found in {baseline_dir}")
        return games

    # Sample games from each player proportionally
    games_per_player = max(1, n_games // len(player_dirs))

    for player_dir in player_dirs:
        player_name = player_dir.name
        player_games_added = 0

        # Try loading from games_cache.json (has PGN strings)
        cache_file = player_dir / "games_cache.json"
        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    data = json.load(f)

                game_list = data.get('games', [])
                if isinstance(data, list):
                    game_list = data

                # Shuffle and sample
                sampled = random.sample(game_list, min(games_per_player * 2, len(game_list)))

                for game_dict in sampled:
                    if player_games_added >= games_per_player:
                        break

                    game = load_game_from_json(game_dict)
                    if game is None:
                        continue

                    # Skip very short games
                    moves = list(game.mainline_moves())
                    if len(moves) < 20:
                        continue

                    games.append((player_name, game))
                    player_games_added += 1

            except Exception as e:
                print(f"Error reading {cache_file}: {e}")
                continue

    # Shuffle and limit
    random.shuffle(games)
    return games[:n_games]


def find_material_change_positions(
    game: chess.pgn.Game,
    min_material_change: int = 100,  # At least 1 pawn worth
) -> list[tuple[int, chess.Board, chess.Move, int]]:
    """
    Find positions where material changed (captures).

    Returns list of (move_idx, board_before, move, material_change) tuples.
    These are interesting positions to analyze with multi-depth because:
    1. The decision to capture/exchange is critical
    2. Material imbalances are easy to detect quickly
    3. Humans are biased towards material over position
    """
    positions = []
    board = game.board()
    prev_material = calculate_material_score(board)

    for move_idx, move in enumerate(game.mainline_moves()):
        # Calculate material BEFORE the move for analysis
        board_before = board.copy()

        # Make the move
        board.push(move)

        # Calculate material after
        new_material = calculate_material_score(board)
        material_change = abs(new_material - prev_material)

        # If significant material change, this is an interesting position
        if material_change >= min_material_change:
            positions.append((move_idx, board_before, move, material_change))

        prev_material = new_material

    return positions


def find_critical_positions(
    game: chess.pgn.Game,
    n_positions: int = 10,
    bias_material_change: float = 0.7,  # 70% from material changes, 30% random
) -> list[tuple[int, chess.Board, chess.Move]]:
    """
    Find critical positions to analyze, biased towards material changes.

    Args:
        game: The chess game
        n_positions: Number of positions to return
        bias_material_change: Fraction of positions that should be material changes

    Returns:
        List of (move_idx, board_before, move) tuples
    """
    moves = list(game.mainline_moves())
    if len(moves) < 10:
        return []

    # Find material change positions
    material_positions = find_material_change_positions(game, min_material_change=100)

    # Number of positions from each source
    n_material = min(int(n_positions * bias_material_change), len(material_positions))
    n_random = n_positions - n_material

    selected = []

    # Sample from material change positions
    if material_positions and n_material > 0:
        sampled_material = random.sample(
            material_positions,
            min(n_material, len(material_positions))
        )
        for move_idx, board, move, _ in sampled_material:
            selected.append((move_idx, board, move))

    # Fill remaining with random positions (skip first 5 and last 5 moves)
    if n_random > 0:
        material_indices = {p[0] for p in selected}
        available_indices = [
            i for i in range(5, len(moves) - 5)
            if i not in material_indices
        ]

        if available_indices:
            random_indices = random.sample(
                available_indices,
                min(n_random, len(available_indices))
            )

            board = game.board()
            for move_idx, move in enumerate(moves):
                if move_idx in random_indices:
                    selected.append((move_idx, board.copy(), move))
                board.push(move)

    # Sort by move index
    selected.sort(key=lambda x: x[0])

    return selected


def analyze_position_multidepth(
    engine: EngineAnalyzer,
    board: chess.Board,
    played_move: Optional[chess.Move] = None,
) -> dict:
    """Analyze a single position at multiple depths."""

    # Get multi-depth analysis
    result = engine.analyze_multi_depth(board)

    # Determine game phase
    phase_result = detect_game_phase(board)
    # detect_game_phase returns GamePhase enum directly
    phase = phase_result.value if hasattr(phase_result, 'value') else str(phase_result)

    # Find when best move first emerged
    sorted_depths = sorted(result.depths)
    final_best = result.best_moves.get(sorted_depths[-1], "")

    first_emergence_depth = None
    for depth in sorted_depths:
        if result.best_moves.get(depth) == final_best:
            first_emergence_depth = depth
            break

    # Check if played move matches best at each depth
    played_match_depths = []
    if played_move:
        played_uci = played_move.uci()
        for depth in sorted_depths:
            if result.best_moves.get(depth) == played_uci:
                played_match_depths.append(depth)

    return {
        'fen': board.fen(),
        'phase': phase,
        'depths': result.depths,
        'evaluations': result.evaluations,
        'best_moves': result.best_moves,
        'final_best_move': final_best,
        'first_emergence_depth': first_emergence_depth,
        'move_consistency': result.move_consistency,
        'best_move_changes': result.best_move_changes,
        'eval_swing': result.eval_swing,
        'material_score': result.material_score,
        'eval_vs_material': result.eval_vs_material,
        'max_eval_change': result.max_eval_change,
        'unstable_depths': result.unstable_depths,
        'depth_transitions': [
            {
                'from_depth': t.from_depth,
                'to_depth': t.to_depth,
                'eval_change': t.eval_change,
                'move_changed': t.move_changed,
            }
            for t in result.depth_transitions
        ],
        'played_move': played_move.uci() if played_move else None,
        'played_match_depths': played_match_depths,
    }


def compute_emergence_stats(
    positions: list[dict],
    depths: list[int]
) -> list[DepthEmergenceStats]:
    """Compute statistics about when best moves first emerge."""

    stats = []
    sorted_depths = sorted(depths)
    max_depth = sorted_depths[-1]

    for depth in sorted_depths:
        # Count moves that first became best at this depth
        emerged_at_depth = [
            p for p in positions
            if p.get('first_emergence_depth') == depth
        ]
        count = len(emerged_at_depth)

        # How many stayed best through max depth?
        stayed_best = count  # By definition, if it emerged here and is final best

        # Average eval when move emerges
        evals = [
            p['evaluations'].get(depth, 0)
            for p in emerged_at_depth
        ]
        avg_eval = np.mean(evals) if evals else 0

        stats.append(DepthEmergenceStats(
            depth=depth,
            count=count,
            pct=count / len(positions) * 100 if positions else 0,
            avg_eval_at_emergence=round(avg_eval, 1),
            stayed_best=stayed_best,
            changed_later=0,  # These are all final best moves
        ))

    return stats


def compute_complexity_stats(
    positions: list[dict],
    depths: list[int]
) -> MoveComplexityStats:
    """Compute move complexity statistics."""

    if not positions:
        return MoveComplexityStats(
            total_positions=0,
            shallow_moves=0, shallow_pct=0,
            deep_moves=0, deep_pct=0,
            mid_depth_moves=0, mid_depth_pct=0,
            stable_moves=0, stable_pct=0,
            avg_best_move_changes=0,
            avg_eval_swing=0, max_eval_swing=0,
            high_swing_count=0, high_swing_pct=0,
        )

    sorted_depths = sorted(depths)
    min_depth = sorted_depths[0]
    max_depth = sorted_depths[-1]

    total = len(positions)

    # Depth of emergence (shallow vs deep)
    shallow = sum(1 for p in positions if p.get('first_emergence_depth') == min_depth)
    deep = sum(1 for p in positions if p.get('first_emergence_depth') == max_depth)
    mid_depth = total - shallow - deep

    # Stability (whether best move changes across depths)
    stable = sum(1 for p in positions if p.get('move_consistency', False))
    avg_changes = np.mean([p.get('best_move_changes', 0) for p in positions])

    # Eval swings
    swings = [p.get('eval_swing', 0) for p in positions]
    avg_swing = np.mean(swings)
    max_swing = max(swings) if swings else 0
    high_swing = sum(1 for s in swings if s > 100)

    return MoveComplexityStats(
        total_positions=total,
        shallow_moves=shallow,
        shallow_pct=round(shallow / total * 100, 1),
        deep_moves=deep,
        deep_pct=round(deep / total * 100, 1),
        mid_depth_moves=mid_depth,
        mid_depth_pct=round(mid_depth / total * 100, 1),
        stable_moves=stable,
        stable_pct=round(stable / total * 100, 1),
        avg_best_move_changes=round(avg_changes, 2),
        avg_eval_swing=round(avg_swing, 1),
        max_eval_swing=max_swing,
        high_swing_count=high_swing,
        high_swing_pct=round(high_swing / total * 100, 1),
    )


def compute_material_stats(positions: list[dict]) -> MaterialAnalysisStats:
    """Compute material vs evaluation statistics."""

    if not positions:
        return MaterialAnalysisStats(
            total_positions=0,
            avg_material=0, material_equal_count=0,
            white_ahead_count=0, black_ahead_count=0,
            avg_eval_vs_material=0,
            positional_advantage_count=0, positional_disadvantage_count=0,
            eval_material_correlation=0,
            opening_avg_positional=0, middlegame_avg_positional=0, endgame_avg_positional=0,
        )

    total = len(positions)

    materials = [p.get('material_score', 0) for p in positions]
    eval_vs_mats = [p.get('eval_vs_material', 0) for p in positions]

    # Get final evals (at max depth)
    final_evals = []
    for p in positions:
        evals = p.get('evaluations', {})
        if evals:
            max_d = max(evals.keys())
            final_evals.append(evals[max_d])
        else:
            final_evals.append(0)

    # Material balance
    avg_material = np.mean(materials)
    material_equal = sum(1 for m in materials if abs(m) <= 50)
    white_ahead = sum(1 for m in materials if m > 50)
    black_ahead = sum(1 for m in materials if m < -50)

    # Positional component
    avg_positional = np.mean(eval_vs_mats)
    pos_advantage = sum(1 for e in eval_vs_mats if e > 50)
    pos_disadvantage = sum(1 for e in eval_vs_mats if e < -50)

    # Correlation
    if len(materials) > 2:
        correlation = np.corrcoef(materials, final_evals)[0, 1]
    else:
        correlation = 0

    # By phase
    phase_positional = {'opening': [], 'middlegame': [], 'endgame': []}
    for p in positions:
        phase = p.get('phase', 'middlegame').lower()
        if phase in phase_positional:
            phase_positional[phase].append(p.get('eval_vs_material', 0))

    return MaterialAnalysisStats(
        total_positions=total,
        avg_material=round(avg_material, 1),
        material_equal_count=material_equal,
        white_ahead_count=white_ahead,
        black_ahead_count=black_ahead,
        avg_eval_vs_material=round(avg_positional, 1),
        positional_advantage_count=pos_advantage,
        positional_disadvantage_count=pos_disadvantage,
        eval_material_correlation=round(correlation, 3) if not np.isnan(correlation) else 0,
        opening_avg_positional=round(np.mean(phase_positional['opening']), 1) if phase_positional['opening'] else 0,
        middlegame_avg_positional=round(np.mean(phase_positional['middlegame']), 1) if phase_positional['middlegame'] else 0,
        endgame_avg_positional=round(np.mean(phase_positional['endgame']), 1) if phase_positional['endgame'] else 0,
    )


def find_example_positions(
    positions: list[dict],
    n_examples: int = 5
) -> tuple[list[dict], list[dict], list[dict]]:
    """Find example positions for each category."""

    sorted_depths = sorted(DEFAULT_MULTI_DEPTHS)
    min_depth = sorted_depths[0]
    max_depth = sorted_depths[-1]

    # Shallow moves - best at min depth, stable
    shallow = [
        p for p in positions
        if p.get('first_emergence_depth') == min_depth
        and p.get('move_consistency', False)
    ]
    shallow = sorted(shallow, key=lambda x: abs(x.get('eval_swing', 0)))[:n_examples]

    # Deep moves - only best at max depth
    deep = [
        p for p in positions
        if p.get('first_emergence_depth') == max_depth
    ]
    deep = sorted(deep, key=lambda x: -abs(x.get('eval_swing', 0)))[:n_examples]

    # High positional - large eval vs material difference
    high_pos = sorted(
        positions,
        key=lambda x: -abs(x.get('eval_vs_material', 0))
    )[:n_examples]

    # Simplify for output
    def simplify(p):
        return {
            'fen': p['fen'],
            'phase': p['phase'],
            'final_best': p['final_best_move'],
            'first_emergence': p['first_emergence_depth'],
            'eval_swing': p['eval_swing'],
            'material': p['material_score'],
            'eval_vs_material': p['eval_vs_material'],
        }

    return (
        [simplify(p) for p in shallow],
        [simplify(p) for p in deep],
        [simplify(p) for p in high_pos],
    )


# =============================================================================
# CACHE FUNCTIONS
# =============================================================================

def get_games_hash(games: list[tuple[str, chess.pgn.Game]]) -> str:
    """Generate a hash of the game list for cache validation."""
    game_ids = []
    for player, game in games:
        # Use headers to identify game
        headers = game.headers
        game_id = f"{player}:{headers.get('Site', '')}:{headers.get('Date', '')}:{headers.get('White', '')}:{headers.get('Black', '')}"
        game_ids.append(game_id)

    game_ids.sort()
    combined = "|".join(game_ids)
    return hashlib.md5(combined.encode()).hexdigest()


def save_to_cache(
    cache_dir: Path,
    positions_by_category: dict[str, list[dict]],
    statistics: dict,
    metadata: dict,
    games_analyzed: list[tuple[str, str]],
):
    """Save analysis data to cache directory."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    positions_dir = cache_dir / "positions"
    positions_dir.mkdir(exist_ok=True)

    # Save metadata
    with open(cache_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    # Save games list
    with open(cache_dir / "games_analyzed.json", 'w') as f:
        json.dump(games_analyzed, f, indent=2)

    # Save statistics
    with open(cache_dir / "statistics.json", 'w') as f:
        json.dump(statistics, f, indent=2)

    # Save positions by category
    for category, positions in positions_by_category.items():
        with open(positions_dir / f"{category}.json", 'w') as f:
            json.dump(positions, f, indent=2)

    print(f"  Cache saved to {cache_dir}")


def load_from_cache(cache_dir: Path) -> tuple[dict, dict, dict[str, list[dict]]]:
    """Load analysis data from cache directory."""
    metadata = json.load(open(cache_dir / "metadata.json"))
    statistics = json.load(open(cache_dir / "statistics.json"))

    positions = {}
    positions_dir = cache_dir / "positions"
    if positions_dir.exists():
        for f in positions_dir.glob("*.json"):
            category = f.stem
            positions[category] = json.load(open(f))

    return metadata, statistics, positions


def is_cache_valid(cache_dir: Path, games_hash: str) -> bool:
    """Check if cache exists and matches the games hash."""
    metadata_file = cache_dir / "metadata.json"
    if not metadata_file.exists():
        return False

    try:
        metadata = json.load(open(metadata_file))
        return metadata.get('games_hash') == games_hash
    except Exception:
        return False


# =============================================================================
# INTERESTING POSITION FINDERS
# =============================================================================

def find_high_fragility_positions(
    game: chess.pgn.Game,
    min_fragility: float = 0.5,
    max_positions: int = 5,
) -> list[tuple[int, chess.Board, chess.Move, float]]:
    """
    Find middlegame positions with high fragility scores.

    High fragility = important pieces under attack, position is tense.
    These are interesting because small mistakes have big consequences.

    Returns list of (move_idx, board_before, move, fragility) tuples.
    """
    positions = []
    board = game.board()

    for move_idx, move in enumerate(game.mainline_moves()):
        # Check position BEFORE move
        board_before = board.copy()

        # Only check middlegame positions
        phase = detect_game_phase(board_before)
        if phase != GamePhase.MIDDLEGAME:
            board.push(move)
            continue

        # Calculate fragility
        try:
            fragility = calculate_fragility_simple(board_before)

            if fragility >= min_fragility:
                positions.append((move_idx, board_before, move, fragility))
        except Exception:
            pass

        board.push(move)

    # Sort by fragility (highest first) and limit
    positions.sort(key=lambda x: -x[3])
    return positions[:max_positions]


def find_comeback_positions(
    game: chess.pgn.Game,
    player_color: chess.Color,
    game_result: str,  # 'win', 'loss', 'draw'
    min_disadvantage: int = 300,  # centipawns down
    max_positions: int = 5,
) -> list[tuple[int, chess.Board, chess.Move, int]]:
    """
    Find positions where player was down material but won the game.

    These are interesting because player found resources despite material deficit,
    suggesting either deep calculation or opponent mistakes.

    Only applicable for games the player won.

    Returns list of (move_idx, board_before, move, material_deficit) tuples.
    """
    if game_result != 'win':
        return []

    positions = []
    board = game.board()

    for move_idx, move in enumerate(game.mainline_moves()):
        board_before = board.copy()

        # Only check middlegame
        phase = detect_game_phase(board_before)
        if phase != GamePhase.MIDDLEGAME:
            board.push(move)
            continue

        # Calculate material from player's perspective
        material = calculate_material_score(board_before)
        if player_color == chess.BLACK:
            material = -material  # Flip for black's perspective

        # If player is significantly down material
        if material <= -min_disadvantage:
            positions.append((move_idx, board_before, move, material))

        board.push(move)

    # Sort by deficit (most down first)
    positions.sort(key=lambda x: x[3])
    return positions[:max_positions]


def get_game_result_and_color(
    game: chess.pgn.Game,
    player_name: str,
) -> tuple[str, chess.Color]:
    """Extract game result and player color from game headers."""
    headers = game.headers
    white = headers.get('White', '').lower()
    black = headers.get('Black', '').lower()
    result = headers.get('Result', '*')

    player_lower = player_name.lower()

    if player_lower in white:
        player_color = chess.WHITE
        if result == '1-0':
            game_result = 'win'
        elif result == '0-1':
            game_result = 'loss'
        else:
            game_result = 'draw'
    elif player_lower in black:
        player_color = chess.BLACK
        if result == '0-1':
            game_result = 'win'
        elif result == '1-0':
            game_result = 'loss'
        else:
            game_result = 'draw'
    else:
        # Default
        player_color = chess.WHITE
        game_result = 'draw'

    return game_result, player_color


def get_game_id(game: chess.pgn.Game) -> str:
    """Extract a unique game ID from headers."""
    headers = game.headers
    site = headers.get('Site', '')
    # Chess.com URLs contain game ID
    if 'chess.com' in site:
        # Extract game ID from URL like https://www.chess.com/game/live/123456
        parts = site.split('/')
        if parts:
            return parts[-1]
    return f"{headers.get('Date', '')}_{headers.get('White', '')}_{headers.get('Black', '')}"


def run_probe(
    baseline_dir: Path,
    output_dir: Path,
    n_games: int = 100,
    positions_per_game: int = 10,
    seed: int = 42,
    stockfish_path: Optional[str] = None,
    include_interesting: bool = True,
    force_rerun: bool = False,
) -> Optional[dict]:
    """Run the multi-depth probe analysis with caching support."""

    print(f"Multi-Depth Probe Analysis")
    print(f"=" * 60)
    print(f"Baseline directory: {baseline_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Games to sample: {n_games}")
    print(f"Positions per game: {positions_per_game}")
    print(f"Include interesting positions: {include_interesting}")
    print(f"Depths: {DEFAULT_MULTI_DEPTHS}")
    print()

    # Sample games first to check cache
    print("Sampling games from baseline...")
    games = sample_games_from_baseline(baseline_dir, n_games, seed)
    print(f"  Sampled {len(games)} games")

    if not games:
        print("No games found!")
        sys.exit(1)

    # Check cache
    games_hash = get_games_hash(games)
    if not force_rerun and is_cache_valid(output_dir, games_hash):
        print(f"\n✓ Valid cache found at {output_dir}")
        print("  Use --force to rerun analysis")
        metadata, statistics, positions = load_from_cache(output_dir)
        print(f"  Loaded {sum(len(p) for p in positions.values())} positions from cache")
        return {'metadata': metadata, 'statistics': statistics, 'positions': positions}

    # Prepare position collection by category
    positions_by_category: dict[str, list[dict]] = {
        'standard': [],
        'high_fragility': [],
        'material_comebacks': [],
    }

    # Track games analyzed
    games_analyzed = []

    print(f"\nAnalyzing positions at depths {DEFAULT_MULTI_DEPTHS}...")

    with EngineAnalyzer(
        stockfish_path=stockfish_path,
        depth=max(DEFAULT_MULTI_DEPTHS),
        threads=1,
        hash_mb=256,
    ) as engine:

        material_change_count = 0
        random_count = 0
        fragility_count = 0
        comeback_count = 0

        for i, (player, game) in enumerate(games):
            game_id = get_game_id(game)
            game_result, player_color = get_game_result_and_color(game, player)
            games_analyzed.append({'player': player, 'game_id': game_id, 'result': game_result})

            # ===== STANDARD POSITIONS =====
            # Find critical positions (biased towards material changes)
            critical_positions = find_critical_positions(
                game,
                n_positions=positions_per_game,
                bias_material_change=0.7,  # 70% material changes, 30% random
            )

            for move_idx, board, move in critical_positions:
                try:
                    result = analyze_position_multidepth(engine, board, move)
                    result['player'] = player
                    result['game_id'] = game_id
                    result['game_result'] = game_result
                    result['move_idx'] = move_idx
                    result['category'] = 'standard'

                    # Track if this was a material change position
                    if board.is_capture(move):
                        result['is_capture'] = True
                        material_change_count += 1
                    else:
                        result['is_capture'] = False
                        random_count += 1

                    positions_by_category['standard'].append(result)
                except Exception as e:
                    print(f"  Error analyzing standard position: {e}")

            # ===== INTERESTING POSITIONS =====
            if include_interesting:
                # High fragility positions
                fragile_positions = find_high_fragility_positions(
                    game,
                    min_fragility=0.5,
                    max_positions=3,
                )
                for move_idx, board, move, fragility in fragile_positions:
                    try:
                        result = analyze_position_multidepth(engine, board, move)
                        result['player'] = player
                        result['game_id'] = game_id
                        result['game_result'] = game_result
                        result['move_idx'] = move_idx
                        result['category'] = 'high_fragility'
                        result['fragility'] = fragility
                        result['is_capture'] = board.is_capture(move)

                        positions_by_category['high_fragility'].append(result)
                        fragility_count += 1
                    except Exception as e:
                        print(f"  Error analyzing fragility position: {e}")

                # Material comeback positions (only for wins)
                comeback_positions = find_comeback_positions(
                    game,
                    player_color,
                    game_result,
                    min_disadvantage=300,
                    max_positions=3,
                )
                for move_idx, board, move, material_deficit in comeback_positions:
                    try:
                        result = analyze_position_multidepth(engine, board, move)
                        result['player'] = player
                        result['game_id'] = game_id
                        result['game_result'] = game_result
                        result['move_idx'] = move_idx
                        result['category'] = 'material_comeback'
                        result['material_deficit'] = material_deficit
                        result['is_capture'] = board.is_capture(move)

                        positions_by_category['material_comebacks'].append(result)
                        comeback_count += 1
                    except Exception as e:
                        print(f"  Error analyzing comeback position: {e}")

            if (i + 1) % 10 == 0:
                total_pos = sum(len(p) for p in positions_by_category.values())
                print(f"  Processed {i + 1}/{len(games)} games, {total_pos} positions")

        print(f"\n  Standard positions: {len(positions_by_category['standard'])}")
        print(f"    Material changes: {material_change_count}")
        print(f"    Random: {random_count}")
        if include_interesting:
            print(f"  High fragility positions: {fragility_count}")
            print(f"  Material comeback positions: {comeback_count}")

    total_positions = sum(len(p) for p in positions_by_category.values())
    print(f"\nTotal positions analyzed: {total_positions}")

    # Compute statistics on all positions combined
    print("\nComputing statistics...")

    all_positions = []
    for positions in positions_by_category.values():
        all_positions.extend(positions)

    emergence_stats = compute_emergence_stats(all_positions, DEFAULT_MULTI_DEPTHS)
    complexity_stats = compute_complexity_stats(all_positions, DEFAULT_MULTI_DEPTHS)
    material_stats = compute_material_stats(all_positions)

    # Phase breakdown
    phase_complexity = {}
    for phase in ['opening', 'middlegame', 'endgame']:
        phase_positions = [p for p in all_positions if p.get('phase', '').lower() == phase]
        if phase_positions:
            phase_complexity[phase] = asdict(
                compute_complexity_stats(phase_positions, DEFAULT_MULTI_DEPTHS)
            )

    # Capture vs non-capture breakdown (standard positions only)
    standard_positions = positions_by_category['standard']
    capture_positions = [p for p in standard_positions if p.get('is_capture', False)]
    non_capture_positions = [p for p in standard_positions if not p.get('is_capture', False)]

    capture_complexity = asdict(compute_complexity_stats(capture_positions, DEFAULT_MULTI_DEPTHS)) if capture_positions else None
    non_capture_complexity = asdict(compute_complexity_stats(non_capture_positions, DEFAULT_MULTI_DEPTHS)) if non_capture_positions else None

    # Category-specific stats
    category_stats = {}
    for category, positions in positions_by_category.items():
        if positions:
            category_stats[category] = asdict(
                compute_complexity_stats(positions, DEFAULT_MULTI_DEPTHS)
            )

    # Find examples
    shallow_ex, deep_ex, high_pos_ex = find_example_positions(all_positions)

    # Build statistics dict
    statistics = {
        'emergence_stats': [asdict(s) for s in emergence_stats],
        'complexity_stats': asdict(complexity_stats),
        'material_stats': asdict(material_stats),
        'phase_complexity': phase_complexity,
        'capture_complexity': capture_complexity,
        'non_capture_complexity': non_capture_complexity,
        'category_stats': category_stats,
        'example_shallow_moves': shallow_ex,
        'example_deep_moves': deep_ex,
        'example_high_positional': high_pos_ex,
    }

    # Build metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'dataset': str(baseline_dir),
        'games_sampled': len(games),
        'positions_analyzed': total_positions,
        'depths_used': DEFAULT_MULTI_DEPTHS,
        'games_hash': games_hash,
        'config': {
            'n_games': n_games,
            'positions_per_game': positions_per_game,
            'include_interesting': include_interesting,
            'seed': seed,
        },
        'position_counts': {
            category: len(positions)
            for category, positions in positions_by_category.items()
        },
    }

    # Save to cache
    print(f"\nSaving results to {output_dir}...")
    save_to_cache(
        cache_dir=output_dir,
        positions_by_category=positions_by_category,
        statistics=statistics,
        metadata=metadata,
        games_analyzed=games_analyzed,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print("\n📊 Move Emergence by Depth:")
    for s in emergence_stats:
        print(f"  Depth {s.depth:2d}: {s.count:4d} moves ({s.pct:5.1f}%) - avg eval {s.avg_eval_at_emergence:+.0f}cp")

    print(f"\n📈 Move Depth Classification:")
    print(f"  Shallow moves (depth {DEFAULT_MULTI_DEPTHS[0]}): {complexity_stats.shallow_moves} ({complexity_stats.shallow_pct:.1f}%)")
    print(f"  Mid-depth moves: {complexity_stats.mid_depth_moves} ({complexity_stats.mid_depth_pct:.1f}%)")
    print(f"  Deep moves (depth {DEFAULT_MULTI_DEPTHS[-1]}): {complexity_stats.deep_moves} ({complexity_stats.deep_pct:.1f}%)")
    print(f"  Stable across all depths: {complexity_stats.stable_moves} ({complexity_stats.stable_pct:.1f}%)")
    print(f"  Avg eval swing: {complexity_stats.avg_eval_swing:.1f}cp")
    print(f"  High swing (>100cp): {complexity_stats.high_swing_count} ({complexity_stats.high_swing_pct:.1f}%)")

    print(f"\n💰 Material vs Evaluation:")
    print(f"  Avg material balance: {material_stats.avg_material:+.1f}cp")
    print(f"  Avg positional component: {material_stats.avg_eval_vs_material:+.1f}cp")
    print(f"  Eval-material correlation: {material_stats.eval_material_correlation:.3f}")
    print(f"  Positional advantage (>50cp): {material_stats.positional_advantage_count} positions")

    print(f"\n📍 By Game Phase:")
    for phase, stats in phase_complexity.items():
        print(f"  {phase.capitalize()}:")
        print(f"    Shallow: {stats['shallow_pct']:.1f}%, Deep: {stats['deep_pct']:.1f}%")

    print(f"\n🎯 Captures vs Non-Captures:")
    if capture_complexity:
        print(f"  Captures ({capture_complexity['total_positions']} positions):")
        print(f"    Shallow: {capture_complexity['shallow_pct']:.1f}%, Deep: {capture_complexity['deep_pct']:.1f}%")
        print(f"    Avg eval swing: {capture_complexity['avg_eval_swing']:.1f}cp")
    if non_capture_complexity:
        print(f"  Non-captures ({non_capture_complexity['total_positions']} positions):")
        print(f"    Shallow: {non_capture_complexity['shallow_pct']:.1f}%, Deep: {non_capture_complexity['deep_pct']:.1f}%")
        print(f"    Avg eval swing: {non_capture_complexity['avg_eval_swing']:.1f}cp")

    if include_interesting:
        print(f"\n🔍 Interesting Positions:")
        for category, count in metadata['position_counts'].items():
            if category != 'standard':
                print(f"  {category}: {count} positions")

    print(f"\nResults saved to: {output_dir}")

    return {'metadata': metadata, 'statistics': statistics, 'positions': positions_by_category}


def main():
    parser = argparse.ArgumentParser(
        description="Multi-depth probe analysis on trusted baseline"
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=Path("data/trusted"),
        help="Directory containing trusted player baselines",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/multidepth_cache"),
        help="Output directory for cached results",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=100,
        help="Number of games to sample (default: 100)",
    )
    parser.add_argument(
        "--positions-per-game",
        type=int,
        default=10,
        help="Standard positions to analyze per game",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--stockfish",
        type=str,
        default=None,
        help="Path to Stockfish binary",
    )
    parser.add_argument(
        "--no-interesting",
        action="store_true",
        help="Skip finding interesting positions (high fragility, comebacks)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rerun even if cache is valid",
    )

    args = parser.parse_args()

    run_probe(
        baseline_dir=args.baseline_dir,
        output_dir=args.output,
        n_games=args.games,
        positions_per_game=args.positions_per_game,
        seed=args.seed,
        stockfish_path=args.stockfish,
        include_interesting=not args.no_interesting,
        force_rerun=args.force,
    )


if __name__ == "__main__":
    main()
