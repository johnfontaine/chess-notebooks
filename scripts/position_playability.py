#!/usr/bin/env python3
"""
Position Playability Analysis

Analyzes positions to understand how "playable" they are by examining
the quality distribution of all legal moves at multiple depths.

Key metrics:
- How many moves are "good enough" (within X cp of best)?
- How does move quality distribute? (one clear best vs many options)
- How does playability change with analysis depth?
- Correlation with fragility, material, game phase

Usage:
    python scripts/position_playability.py --games 2 --positions 20
    python scripts/position_playability.py --games 3 --output reports/playability.json
"""

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import chess
import numpy as np
import pandas as pd

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from chess_analysis import (
    EngineAnalyzer,
    DEFAULT_MULTI_DEPTHS,
    calculate_material_score,
    detect_game_phase,
    calculate_fragility_simple,
)
from scripts.multidepth_probe import sample_games_from_baseline


@dataclass
class MoveEval:
    """Evaluation of a single move at multiple depths."""
    move_uci: str
    is_capture: bool
    evals: dict[int, int]  # depth -> eval in cp (from moving side's perspective)


@dataclass
class PlayabilityResult:
    """Playability analysis for a single position."""
    fen: str
    phase: str
    legal_move_count: int
    fragility: float
    material_score: int

    # Best move info
    best_move: str
    best_eval: int  # At max depth

    # Playability metrics at each depth
    # "good" = within 50cp, "reasonable" = within 100cp, "playable" = within 200cp
    good_moves: dict[int, int]      # depth -> count within 50cp
    reasonable_moves: dict[int, int]  # depth -> count within 100cp
    playable_moves: dict[int, int]   # depth -> count within 200cp

    # Distribution stats at max depth
    eval_std: float          # Standard deviation of move evals
    eval_range: int          # Best - worst eval
    median_loss: int         # Median cp loss vs best move

    # Move quality percentiles at max depth
    pct_good: float          # % of moves within 50cp
    pct_reasonable: float    # % of moves within 100cp
    pct_playable: float      # % of moves within 200cp

    # Depth dynamics
    good_moves_change: int   # Change in good move count from min to max depth
    complexity_revealed: bool  # Did deeper analysis reveal fewer good options?


def analyze_all_moves(
    engine: EngineAnalyzer,
    board: chess.Board,
    depths: list[int],
) -> list[MoveEval]:
    """
    Analyze all legal moves at multiple depths.

    Returns list of MoveEval with evaluations from the moving side's perspective.
    """
    results = []

    for move in board.legal_moves:
        is_capture = board.is_capture(move)

        # Make the move
        board.push(move)

        # Analyze resulting position at all depths
        analysis = engine.analyze_multi_depth(board, depths)

        board.pop()

        # Convert evals to moving side's perspective (negate since it's opponent's turn after move)
        evals = {}
        for depth in depths:
            eval_cp = analysis.evaluations.get(depth, 0)
            evals[depth] = -eval_cp  # Negate for moving side's perspective

        results.append(MoveEval(
            move_uci=move.uci(),
            is_capture=is_capture,
            evals=evals,
        ))

    return results


def compute_playability(
    board: chess.Board,
    move_evals: list[MoveEval],
    depths: list[int],
) -> PlayabilityResult:
    """
    Compute playability metrics from move evaluations.
    """
    if not move_evals:
        raise ValueError("No moves to analyze")

    sorted_depths = sorted(depths)
    min_depth = sorted_depths[0]
    max_depth = sorted_depths[-1]

    # Get best eval at each depth
    best_evals = {}
    for depth in depths:
        best_evals[depth] = max(m.evals[depth] for m in move_evals)

    # Find best move (by max depth eval)
    best_move = max(move_evals, key=lambda m: m.evals[max_depth])

    # Count moves within thresholds at each depth
    good_moves = {}      # within 50cp
    reasonable_moves = {}  # within 100cp
    playable_moves = {}   # within 200cp

    for depth in depths:
        best = best_evals[depth]
        good_moves[depth] = sum(1 for m in move_evals if best - m.evals[depth] <= 50)
        reasonable_moves[depth] = sum(1 for m in move_evals if best - m.evals[depth] <= 100)
        playable_moves[depth] = sum(1 for m in move_evals if best - m.evals[depth] <= 200)

    # Distribution stats at max depth
    max_depth_evals = [m.evals[max_depth] for m in move_evals]
    best_max = best_evals[max_depth]
    losses = [best_max - e for e in max_depth_evals]

    eval_std = float(np.std(max_depth_evals))
    eval_range = max(max_depth_evals) - min(max_depth_evals)
    median_loss = int(np.median(losses))

    # Percentages
    n_moves = len(move_evals)
    pct_good = good_moves[max_depth] / n_moves * 100
    pct_reasonable = reasonable_moves[max_depth] / n_moves * 100
    pct_playable = playable_moves[max_depth] / n_moves * 100

    # Depth dynamics
    good_change = good_moves[max_depth] - good_moves[min_depth]
    complexity_revealed = good_change < 0  # Fewer good moves at depth

    # Position metrics
    fragility = calculate_fragility_simple(board)
    material = calculate_material_score(board)
    phase = detect_game_phase(board)

    return PlayabilityResult(
        fen=board.fen(),
        phase=phase.value,
        legal_move_count=n_moves,
        fragility=fragility,
        material_score=material,
        best_move=best_move.move_uci,
        best_eval=best_move.evals[max_depth],
        good_moves=good_moves,
        reasonable_moves=reasonable_moves,
        playable_moves=playable_moves,
        eval_std=eval_std,
        eval_range=eval_range,
        median_loss=median_loss,
        pct_good=pct_good,
        pct_reasonable=pct_reasonable,
        pct_playable=pct_playable,
        good_moves_change=good_change,
        complexity_revealed=complexity_revealed,
    )


def analyze_position_playability(
    engine: EngineAnalyzer,
    board: chess.Board,
    depths: Optional[list[int]] = None,
) -> tuple[PlayabilityResult, list[MoveEval]]:
    """
    Full playability analysis for a position.

    Returns (PlayabilityResult, list of MoveEval for all moves)
    """
    if depths is None:
        depths = DEFAULT_MULTI_DEPTHS

    move_evals = analyze_all_moves(engine, board, depths)
    playability = compute_playability(board, move_evals, depths)

    return playability, move_evals


def run_exploration(games, n_games=2, positions_per_game=20, depths=None):
    """
    Run playability analysis across multiple games.
    """
    if depths is None:
        depths = DEFAULT_MULTI_DEPTHS

    results = []
    all_move_evals = []  # Optional: store all move evals for detailed analysis

    with EngineAnalyzer(depth=max(depths)) as engine:
        for game_idx, (player, game) in enumerate(games[:n_games]):
            print(f"\nGame {game_idx + 1}/{n_games}: {player}")
            board = game.board()

            for move_idx, move in enumerate(game.mainline_moves()):
                if move_idx >= positions_per_game:
                    break

                # Skip book moves
                if move_idx < 6:
                    board.push(move)
                    continue

                print(f"  Position {move_idx}: ", end="", flush=True)

                playability, move_evals = analyze_position_playability(engine, board, depths)

                # Add context
                result = asdict(playability)
                result['game_idx'] = game_idx
                result['player'] = player
                result['move_idx'] = move_idx
                result['played_move'] = move.uci()

                # Check if played move was among the good/reasonable options
                played_eval = None
                for me in move_evals:
                    if me.move_uci == move.uci():
                        played_eval = me.evals[max(depths)]
                        break

                if played_eval is not None:
                    played_loss = playability.best_eval - played_eval
                    result['played_loss'] = played_loss
                    result['played_was_good'] = played_loss <= 50
                    result['played_was_reasonable'] = played_loss <= 100
                    result['played_was_playable'] = played_loss <= 200

                print(f"good={playability.good_moves[max(depths)]}, "
                      f"reasonable={playability.reasonable_moves[max(depths)]}, "
                      f"playable={playability.playable_moves[max(depths)]}/{playability.legal_move_count}, "
                      f"fragility={playability.fragility:.2f}")

                results.append(result)
                board.push(move)

    return results


def analyze_results(df):
    """Analyze and display results."""
    max_depth = max(DEFAULT_MULTI_DEPTHS)

    print("\n" + "=" * 70)
    print("POSITION PLAYABILITY ANALYSIS")
    print("=" * 70)

    # Extract max-depth metrics for easier analysis
    df['good_at_max'] = df['good_moves'].apply(lambda x: x.get(str(max_depth)) or x.get(max_depth, 0))
    df['reasonable_at_max'] = df['reasonable_moves'].apply(lambda x: x.get(str(max_depth)) or x.get(max_depth, 0))
    df['playable_at_max'] = df['playable_moves'].apply(lambda x: x.get(str(max_depth)) or x.get(max_depth, 0))

    print("\n=== Summary Statistics ===")
    print(f"  Total positions:           {len(df)}")
    print(f"  Avg legal moves:           {df['legal_move_count'].mean():.1f}")
    print(f"  Avg good moves (≤50cp):    {df['good_at_max'].mean():.1f} ({df['pct_good'].mean():.1f}%)")
    print(f"  Avg reasonable (≤100cp):   {df['reasonable_at_max'].mean():.1f} ({df['pct_reasonable'].mean():.1f}%)")
    print(f"  Avg playable (≤200cp):     {df['playable_at_max'].mean():.1f} ({df['pct_playable'].mean():.1f}%)")
    print(f"  Avg median loss:           {df['median_loss'].mean():.0f}cp")
    print(f"  Avg eval std:              {df['eval_std'].mean():.0f}cp")

    # Complexity revealed
    revealed = df['complexity_revealed'].sum()
    print(f"\n  Complexity revealed by depth: {revealed}/{len(df)} ({revealed/len(df)*100:.1f}%)")
    print(f"  (Positions where deeper analysis found fewer good options)")

    # Correlations
    print("\n=== Correlations with Good Move Count ===")
    for col in ['fragility', 'legal_move_count', 'material_score', 'eval_std']:
        if col in df.columns:
            corr = df[col].corr(df['good_at_max'])
            if not pd.isna(corr):
                print(f"  {col:20s}: {corr:+.3f}")

    print("\n=== Correlations with Pct Good ===")
    for col in ['fragility', 'legal_move_count', 'material_score', 'eval_std']:
        if col in df.columns:
            corr = df[col].corr(df['pct_good'])
            if not pd.isna(corr):
                print(f"  {col:20s}: {corr:+.3f}")

    # By phase
    print("\n=== Playability by Game Phase ===")
    for phase in ['opening', 'middlegame', 'endgame']:
        phase_data = df[df['phase'] == phase]
        if len(phase_data) > 0:
            print(f"\n  {phase.title()} ({len(phase_data)} positions):")
            print(f"    Avg good moves:      {phase_data['good_at_max'].mean():.1f} ({phase_data['pct_good'].mean():.1f}%)")
            print(f"    Avg reasonable:      {phase_data['reasonable_at_max'].mean():.1f} ({phase_data['pct_reasonable'].mean():.1f}%)")
            print(f"    Avg legal moves:     {phase_data['legal_move_count'].mean():.1f}")
            print(f"    Avg fragility:       {phase_data['fragility'].mean():.3f}")

    # Critical positions (few good moves)
    critical = df[df['good_at_max'] <= 2]
    if len(critical) > 0:
        print(f"\n=== Critical Positions (≤2 good moves) ===")
        print(f"  Count: {len(critical)} ({len(critical)/len(df)*100:.1f}%)")
        print(f"  Avg fragility: {critical['fragility'].mean():.3f}")
        print(f"  Avg legal moves: {critical['legal_move_count'].mean():.1f}")

    # Forgiving positions (many good moves)
    forgiving = df[df['good_at_max'] >= 5]
    if len(forgiving) > 0:
        print(f"\n=== Forgiving Positions (≥5 good moves) ===")
        print(f"  Count: {len(forgiving)} ({len(forgiving)/len(df)*100:.1f}%)")
        print(f"  Avg fragility: {forgiving['fragility'].mean():.3f}")
        print(f"  Avg legal moves: {forgiving['legal_move_count'].mean():.1f}")

    # Played move quality (if available)
    if 'played_was_good' in df.columns:
        print("\n=== Played Move Quality ===")
        print(f"  Good (≤50cp loss):       {df['played_was_good'].sum()} ({df['played_was_good'].mean()*100:.1f}%)")
        print(f"  Reasonable (≤100cp):     {df['played_was_reasonable'].sum()} ({df['played_was_reasonable'].mean()*100:.1f}%)")
        print(f"  Playable (≤200cp):       {df['played_was_playable'].sum()} ({df['played_was_playable'].mean()*100:.1f}%)")
        print(f"  Avg played loss:         {df['played_loss'].mean():.0f}cp")

    return df


def main():
    parser = argparse.ArgumentParser(description="Analyze position playability")
    parser.add_argument("--games", type=int, default=2, help="Number of games to analyze")
    parser.add_argument("--positions", type=int, default=20, help="Positions per game")
    parser.add_argument("--baseline", type=str, default="data/trusted", help="Trusted baseline directory")
    parser.add_argument("--output", type=str, help="Output JSON file for results")
    args = parser.parse_args()

    print("=" * 70)
    print("POSITION PLAYABILITY EXPLORATION")
    print("=" * 70)
    print(f"  Games: {args.games}")
    print(f"  Positions per game: {args.positions}")
    print(f"  Depths: {DEFAULT_MULTI_DEPTHS}")
    print()

    # Sample games
    baseline_path = Path(args.baseline)
    games = sample_games_from_baseline(baseline_path, n_games=args.games + 1)
    print(f"Sampled {len(games)} games")

    # Run analysis
    results = run_exploration(games, n_games=args.games, positions_per_game=args.positions)

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Analyze
    df = analyze_results(df)

    # Save if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert dict columns for JSON serialization
        output_records = []
        for r in results:
            record = dict(r)
            # Convert depth keys to strings for JSON
            for key in ['good_moves', 'reasonable_moves', 'playable_moves']:
                if key in record and isinstance(record[key], dict):
                    record[key] = {str(k): v for k, v in record[key].items()}
            output_records.append(record)

        output_data = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'games': args.games,
                'positions_per_game': args.positions,
                'depths': DEFAULT_MULTI_DEPTHS,
            },
            'summary': {
                'total_positions': len(df),
                'avg_good_moves': float(df['good_at_max'].mean()),
                'avg_pct_good': float(df['pct_good'].mean()),
                'avg_reasonable_moves': float(df['reasonable_at_max'].mean()),
                'avg_pct_reasonable': float(df['pct_reasonable'].mean()),
                'avg_median_loss': float(df['median_loss'].mean()),
                'complexity_revealed_count': int(df['complexity_revealed'].sum()),
            },
            'positions': output_records,
        }

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\n✓ Results saved to {output_path}")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
