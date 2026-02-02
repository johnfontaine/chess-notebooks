#!/usr/bin/env python3
"""
Trap Move Exploration Script

Explores the relationship between "trap moves" and position characteristics
like fragility, legal move count, and material balance.

A "trap move" is one that looks good at shallow depth but degrades under analysis.

Usage:
    python scripts/trap_move_exploration.py --games 2 --positions 20
    python scripts/trap_move_exploration.py --games 3 --positions 30 --output reports/trap_analysis.json
"""

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

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
from scripts.multidepth_probe import (
    analyze_position_trap_density,
    sample_games_from_baseline,
)


def explore_trap_moves(games, n_games=2, positions_per_game=20):
    """
    Explore trap move patterns across games.

    For each position:
    1. Analyze for trap moves (moves that look good shallow but degrade)
    2. Calculate fragility, legal move count, material score
    3. Return DataFrame for correlation analysis
    """
    results = []
    total_positions = 0

    with EngineAnalyzer(depth=21) as engine:
        for game_idx, (player, game) in enumerate(games[:n_games]):
            print(f"\nGame {game_idx + 1}/{n_games}: {player}")
            board = game.board()
            game_positions = 0

            for move_idx, move in enumerate(game.mainline_moves()):
                if move_idx >= positions_per_game:
                    break

                # Skip very early positions (book moves)
                if move_idx < 6:
                    board.push(move)
                    continue

                print(f"  Position {move_idx}: ", end="", flush=True)

                # Analyze position for traps
                trap_info = analyze_position_trap_density(engine, board)

                # Add position info
                trap_info['fen'] = board.fen()
                trap_info['game_idx'] = game_idx
                trap_info['player'] = player
                trap_info['move_idx'] = move_idx
                trap_info['played_move'] = move.uci()
                trap_info['phase'] = detect_game_phase(board).value

                # Check if played move was a trap
                trap_info['played_was_trap'] = any(
                    t['move_uci'] == move.uci() for t in trap_info.get('traps', [])
                )

                print(f"{trap_info['trap_count']} traps, density={trap_info['trap_density']:.3f}, "
                      f"fragility={trap_info['fragility']:.2f}, moves={trap_info['legal_move_count']}")

                results.append(trap_info)
                game_positions += 1
                total_positions += 1

                board.push(move)

            print(f"  Analyzed {game_positions} positions")

    print(f"\n✓ Total positions analyzed: {total_positions}")
    return results


def analyze_correlations(df):
    """Compute and display correlation analysis."""
    print("\n" + "=" * 60)
    print("CORRELATION ANALYSIS: What predicts trap move density?")
    print("=" * 60)

    # Add derived column
    df['abs_material'] = df['material_score'].abs()

    print("\n=== Correlation with Trap Density ===")
    for col in ['fragility', 'legal_move_count', 'material_score', 'abs_material']:
        corr = df[col].corr(df['trap_density'])
        if pd.isna(corr):
            print(f"  {col:20s}: N/A (no variance)")
        else:
            print(f"  {col:20s}: {corr:+.3f}")

    print("\n=== Correlation with Trap Count ===")
    for col in ['fragility', 'legal_move_count', 'material_score', 'abs_material']:
        corr = df[col].corr(df['trap_count'])
        if pd.isna(corr):
            print(f"  {col:20s}: N/A (no variance)")
        else:
            print(f"  {col:20s}: {corr:+.3f}")

    return df


def compute_summary_stats(df):
    """Compute and display summary statistics."""
    print("\n=== Summary Statistics ===")
    print(f"  Total positions:         {len(df)}")
    print(f"  Positions with traps:    {(df['trap_count'] > 0).sum()} ({(df['trap_count'] > 0).mean()*100:.1f}%)")
    print(f"  Avg traps per position:  {df['trap_count'].mean():.2f}")
    print(f"  Avg trap density:        {df['trap_density'].mean():.3f}")
    print(f"  Avg legal moves:         {df['legal_move_count'].mean():.1f}")
    print(f"  Avg fragility:           {df['fragility'].mean():.3f}")
    print(f"  Played moves that were traps: {df['played_was_trap'].sum()}")


def analyze_by_phase(df):
    """Analyze trap patterns by game phase."""
    print("\n=== Trap Move Patterns by Game Phase ===")
    for phase in ['opening', 'middlegame', 'endgame']:
        phase_data = df[df['phase'] == phase]
        if len(phase_data) > 0:
            print(f"\n{phase.title()}:")
            print(f"  Positions: {len(phase_data)}")
            print(f"  With traps: {(phase_data['trap_count'] > 0).sum()} "
                  f"({(phase_data['trap_count'] > 0).mean()*100:.1f}%)")
            print(f"  Avg trap count: {phase_data['trap_count'].mean():.2f}")
            print(f"  Avg trap density: {phase_data['trap_density'].mean():.3f}")
            print(f"  Avg fragility: {phase_data['fragility'].mean():.3f}")


def compare_trap_positions(df):
    """Compare positions with and without traps."""
    has_traps = df[df['trap_count'] > 0]
    no_traps = df[df['trap_count'] == 0]

    print("\n=== Positions With vs Without Traps ===")

    if len(has_traps) > 0:
        print(f"\nPositions WITH traps ({len(has_traps)}):")
        print(f"  Avg fragility:      {has_traps['fragility'].mean():.3f}")
        print(f"  Avg legal moves:    {has_traps['legal_move_count'].mean():.1f}")
        print(f"  Avg abs material:   {has_traps['abs_material'].mean():.0f}cp")

    if len(no_traps) > 0:
        print(f"\nPositions WITHOUT traps ({len(no_traps)}):")
        print(f"  Avg fragility:      {no_traps['fragility'].mean():.3f}")
        print(f"  Avg legal moves:    {no_traps['legal_move_count'].mean():.1f}")
        print(f"  Avg abs material:   {no_traps['abs_material'].mean():.0f}cp")


def main():
    parser = argparse.ArgumentParser(description="Explore trap move patterns")
    parser.add_argument("--games", type=int, default=2, help="Number of games to analyze")
    parser.add_argument("--positions", type=int, default=25, help="Positions per game")
    parser.add_argument("--baseline", type=str, default="data/trusted", help="Trusted baseline directory")
    parser.add_argument("--output", type=str, help="Output JSON file for results")
    args = parser.parse_args()

    print("=" * 60)
    print("TRAP MOVE EXPLORATION")
    print("=" * 60)
    print(f"  Games to analyze: {args.games}")
    print(f"  Positions per game: {args.positions}")
    print(f"  Baseline: {args.baseline}")
    print(f"  Multi-depth levels: {DEFAULT_MULTI_DEPTHS}")
    print()

    # Sample games
    baseline_path = Path(args.baseline)
    games = sample_games_from_baseline(baseline_path, n_games=args.games + 1)  # Extra for safety
    print(f"Sampled {len(games)} games from baseline")

    # Run exploration
    results = explore_trap_moves(games, n_games=args.games, positions_per_game=args.positions)

    # Convert to DataFrame (dropping trap details for analysis)
    df = pd.DataFrame(results)
    if 'traps' in df.columns:
        df = df.drop(columns=['traps'])
    if 'worst_trap' in df.columns:
        df = df.drop(columns=['worst_trap'])

    # Run analyses
    df = analyze_correlations(df)
    compute_summary_stats(df)
    analyze_by_phase(df)
    compare_trap_positions(df)

    # Show top trap positions
    if (df['trap_count'] > 0).any():
        print("\n=== Top 5 Positions by Trap Count ===")
        top_traps = df[df['trap_count'] > 0].nlargest(5, 'trap_count')
        for _, row in top_traps.iterrows():
            print(f"  {row['phase']:12s} | move {row['move_idx']:2d} | "
                  f"{row['trap_count']} traps | density={row['trap_density']:.3f} | "
                  f"fragility={row['fragility']:.3f} | moves={row['legal_move_count']}")

    # Save results if output specified
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_data = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'games': args.games,
                'positions_per_game': args.positions,
                'baseline': args.baseline,
                'depths': DEFAULT_MULTI_DEPTHS,
            },
            'summary': {
                'total_positions': len(df),
                'positions_with_traps': int((df['trap_count'] > 0).sum()),
                'positions_with_traps_pct': float((df['trap_count'] > 0).mean() * 100),
                'avg_trap_count': float(df['trap_count'].mean()),
                'avg_trap_density': float(df['trap_density'].mean()),
                'avg_fragility': float(df['fragility'].mean()),
                'avg_legal_moves': float(df['legal_move_count'].mean()),
                'played_was_trap_count': int(df['played_was_trap'].sum()),
            },
            'correlations': {
                'trap_density': {
                    'fragility': float(df['fragility'].corr(df['trap_density'])) if not pd.isna(df['fragility'].corr(df['trap_density'])) else None,
                    'legal_move_count': float(df['legal_move_count'].corr(df['trap_density'])) if not pd.isna(df['legal_move_count'].corr(df['trap_density'])) else None,
                    'material_score': float(df['material_score'].corr(df['trap_density'])) if not pd.isna(df['material_score'].corr(df['trap_density'])) else None,
                    'abs_material': float(df['abs_material'].corr(df['trap_density'])) if not pd.isna(df['abs_material'].corr(df['trap_density'])) else None,
                },
                'trap_count': {
                    'fragility': float(df['fragility'].corr(df['trap_count'])) if not pd.isna(df['fragility'].corr(df['trap_count'])) else None,
                    'legal_move_count': float(df['legal_move_count'].corr(df['trap_count'])) if not pd.isna(df['legal_move_count'].corr(df['trap_count'])) else None,
                    'material_score': float(df['material_score'].corr(df['trap_count'])) if not pd.isna(df['material_score'].corr(df['trap_count'])) else None,
                    'abs_material': float(df['abs_material'].corr(df['trap_count'])) if not pd.isna(df['abs_material'].corr(df['trap_count'])) else None,
                },
            },
            'positions': df.to_dict(orient='records'),
        }

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\n✓ Results saved to {output_path}")

    print("\n" + "=" * 60)
    print("EXPLORATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
