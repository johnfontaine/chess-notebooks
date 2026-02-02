#!/usr/bin/env python3
"""
Test script for position metrics calculations.

Tests:
- Raw branching factor (sampling-based)
- Brute-force branching factor (all nodes)
- Complexity calculation
- Win percentage conversion
- Move accuracy calculation
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import chess
from chess_analysis import (
    calculate_raw_branching_factor,
    calculate_brute_force_branching,
    centipawns_to_win_percent,
    calculate_move_accuracy,
    calculate_complexity_fast,
    EngineAnalyzer,
)


def test_brute_force_branching():
    """Test brute-force branching factor calculation."""
    print("=" * 60)
    print("Testing Brute-Force Branching Factor")
    print("=" * 60)

    # Starting position
    board = chess.Board()
    result = calculate_brute_force_branching(board, depth=3)
    print(f"\nStarting position:")
    print(f"  Legal moves: {result['initial_legal_moves']}")
    print(f"  Nodes at depth 1: {result['nodes_by_depth'][0]}")
    print(f"  Nodes at depth 2: {result['nodes_by_depth'][1]}")
    print(f"  Nodes at depth 3: {result['nodes_by_depth'][2]}")
    print(f"  Total nodes: {result['total_nodes']}")
    print(f"  Branching factor: {result['branching_factor']:.2f}")

    # After 1.e4
    board_e4 = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
    result_e4 = calculate_brute_force_branching(board_e4, depth=3)
    print(f"\nAfter 1.e4:")
    print(f"  Legal moves: {result_e4['initial_legal_moves']}")
    print(f"  Nodes by depth: {result_e4['nodes_by_depth']}")
    print(f"  Total nodes: {result_e4['total_nodes']}")
    print(f"  Branching factor: {result_e4['branching_factor']:.2f}")

    # Endgame position (fewer pieces)
    endgame = chess.Board("8/8/8/4k3/8/4K3/4P3/8 w - - 0 1")
    result_end = calculate_brute_force_branching(endgame, depth=3)
    print(f"\nEndgame (K+P vs K):")
    print(f"  Legal moves: {result_end['initial_legal_moves']}")
    print(f"  Nodes by depth: {result_end['nodes_by_depth']}")
    print(f"  Total nodes: {result_end['total_nodes']}")
    print(f"  Branching factor: {result_end['branching_factor']:.2f}")

    return True


def test_raw_branching_factor():
    """Test sampling-based branching factor."""
    print("\n" + "=" * 60)
    print("Testing Raw Branching Factor (sampling-based)")
    print("=" * 60)

    board = chess.Board()
    raw_bf = calculate_raw_branching_factor(board, depth=3)
    print(f"\nStarting position:")
    print(f"  Raw branching factor: {raw_bf:.2f}")

    # After 1.e4
    board_e4 = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
    raw_bf_e4 = calculate_raw_branching_factor(board_e4, depth=3)
    print(f"\nAfter 1.e4:")
    print(f"  Raw branching factor: {raw_bf_e4:.2f}")

    return True


def test_win_percentage():
    """Test centipawn to win percentage conversion."""
    print("\n" + "=" * 60)
    print("Testing Win Percentage Conversion")
    print("=" * 60)

    test_cases = [
        (0, 50.0),      # Equal position
        (100, None),    # +1 pawn
        (-100, None),   # -1 pawn
        (300, None),    # +3 pawns
        (500, None),    # +5 pawns
        (1000, None),   # +10 pawns (near winning)
    ]

    print("\n  CP   | Win %")
    print("  -----+-------")
    for cp, expected in test_cases:
        win_pct = centipawns_to_win_percent(cp)
        print(f"  {cp:+5d} | {win_pct:.1f}%")

    # Verify equal position gives 50%
    assert abs(centipawns_to_win_percent(0) - 50.0) < 0.1, "Equal position should be ~50%"
    print("\n  [PASS] Equal position correctly gives 50% win chance")

    return True


def test_move_accuracy():
    """Test move accuracy calculation."""
    print("\n" + "=" * 60)
    print("Testing Move Accuracy Calculation")
    print("=" * 60)

    test_cases = [
        (50.0, 50.0, "Perfect move (no loss)"),
        (50.0, 45.0, "Small loss (5%)"),
        (50.0, 40.0, "Moderate loss (10%)"),
        (50.0, 30.0, "Large loss (20%)"),
        (70.0, 70.0, "Perfect move from advantage"),
        (70.0, 50.0, "Blunder from advantage"),
    ]

    print("\n  Before | After | Accuracy | Description")
    print("  -------+-------+----------+-------------")
    for before, after, desc in test_cases:
        acc = calculate_move_accuracy(before, after)
        print(f"  {before:5.1f}% | {after:5.1f}% | {acc:6.1f}%  | {desc}")

    # Verify perfect move gives ~100%
    perfect = calculate_move_accuracy(50.0, 50.0)
    assert perfect > 99.0, "Perfect move should be ~100% accuracy"
    print("\n  [PASS] Perfect move correctly gives ~100% accuracy")

    return True


def test_complexity_with_engine():
    """Test complexity calculation with Stockfish."""
    print("\n" + "=" * 60)
    print("Testing Complexity Calculation with Stockfish")
    print("=" * 60)

    engine_path = "/opt/homebrew/bin/stockfish"

    positions = [
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "Starting position"),
        ("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", "Italian Game"),
        ("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4", "Italian with Bc5"),
        ("8/8/8/4k3/8/4K3/4P3/8 w - - 0 1", "Simple endgame"),
    ]

    with EngineAnalyzer(engine_path, depth=14) as analyzer:
        print("\n  Position              | Complexity | Category")
        print("  ----------------------+------------+----------")
        for fen, desc in positions:
            board = chess.Board(fen)
            try:
                complexity = calculate_complexity_fast(
                    board,
                    analyzer._engine,
                    shallow_depth=6,
                    deep_depth=14,
                )
                if complexity < 0.2:
                    category = "LOW"
                elif complexity < 0.4:
                    category = "MEDIUM"
                elif complexity < 0.6:
                    category = "HIGH"
                else:
                    category = "VERY_HIGH"
                print(f"  {desc:21} | {complexity*100:8.1f}%  | {category}")
            except Exception as e:
                print(f"  {desc:21} | ERROR: {e}")

    return True


def test_multi_depth_heuristics():
    """Test multi-depth analysis with complexity heuristics."""
    print("\n" + "=" * 60)
    print("Testing Multi-Depth Complexity Heuristics")
    print("=" * 60)

    engine_path = "/opt/homebrew/bin/stockfish"

    positions = [
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "Starting position"),
        ("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", "After 1.e4 e5 2.Nf3 Nc6"),
        ("8/8/8/4k3/8/4K3/4P3/8 w - - 0 1", "Simple endgame"),
    ]

    with EngineAnalyzer(engine_path, depth=20) as analyzer:
        print("\n  Position              | Eval Vol | SF Branch | Complexity | Category")
        print("  ----------------------+----------+-----------+------------+---------")
        for fen, desc in positions:
            board = chess.Board(fen)
            try:
                result = analyzer.analyze_multi_depth_extended(
                    board,
                    depths=[6, 10, 14, 20],
                    multipv=2,
                    capture_search_stats=True,
                )
                ch = result.complexity_heuristics
                if ch:
                    vol = f"{ch.eval_volatility:6.1f}cp"
                    bf = f"{ch.branching_factor_estimate:5.2f}"
                    cmplx = f"{ch.complexity_score*100:6.1f}%"
                    cat = ch.complexity_category
                    print(f"  {desc:21} | {vol:>8} | {bf:>9} | {cmplx:>10} | {cat}")
                else:
                    print(f"  {desc:21} | NO HEURISTICS")
            except Exception as e:
                print(f"  {desc:21} | ERROR: {e}")

    print("\n  [PASS] Multi-depth heuristics test completed")
    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("POSITION METRICS TEST SUITE")
    print("=" * 60)

    tests = [
        ("Brute-Force Branching", test_brute_force_branching),
        ("Raw Branching Factor", test_raw_branching_factor),
        ("Win Percentage", test_win_percentage),
        ("Move Accuracy", test_move_accuracy),
        ("Complexity", test_complexity_with_engine),
        ("Multi-Depth Heuristics", test_multi_depth_heuristics),
    ]

    results = []
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, "PASS" if success else "FAIL"))
        except Exception as e:
            print(f"\n  [ERROR] {name}: {e}")
            results.append((name, "ERROR"))

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for name, status in results:
        print(f"  {name:30} [{status}]")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
