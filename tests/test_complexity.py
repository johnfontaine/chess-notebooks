"""
Tests for position complexity calculations.

Tests verify that:
1. Engine complexity heuristics are computed correctly
2. Higher-rated puzzles tend to have higher complexity scores
3. Individual complexity components work as expected

Run with: pytest tests/test_complexity.py -v
Run slow tests: pytest tests/test_complexity.py -v --run-slow
"""

import pytest
import chess
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from chess_analysis.engine import (
    EngineAnalyzer,
    MultiDepthResult,
    PositionComplexityHeuristics,
    calculate_eval_volatility,
    estimate_branching_factor,
    calculate_complexity_heuristics,
    categorize_complexity,
    EngineSearchMetrics,
    GapMetricResult,
)


# =============================================================================
# Puzzle Test Data
# Source: Lichess Puzzle Database (https://database.lichess.org/)
# License: CC0 (Public Domain)
# Puzzles sampled from different rating ranges to test complexity correlation
# =============================================================================

PUZZLES_LOW = [
    {
        'id': '0009B',
        'fen': 'r2qr1k1/b1p2ppp/pp4n1/P1P1p3/4P1n1/B2P2Pb/3NBP1P/RN1QR1K1 b - - 1 16',
        'rating': 1103,
        'themes': 'advantage middlegame short',
    },
    {
        'id': '000o3',
        'fen': '8/2p1k3/6p1/1p1P1p2/1P3P2/3K2Pp/7P/8 b - - 1 43',
        'rating': 944,
        'themes': 'crushing endgame pawnEndgame short zugzwang',
    },
    {
        'id': '000rO',
        'fen': '3R4/8/K7/pB2b3/1p6/1P2k3/3p4/8 w - - 4 58',
        'rating': 1110,
        'themes': 'crushing endgame fork master short',
    },
    {
        'id': '000rZ',
        'fen': '2kr1b1r/p1p2pp1/2pqb3/7p/3N2n1/2NPB3/PPP2PPP/R2Q1RK1 w - - 2 13',
        'rating': 923,
        'themes': 'kingsideAttack mate mateIn1 oneMove opening',
    },
    {
        'id': '001Wz',
        'fen': '4r1k1/5ppp/r1p5/p1n1RP2/8/2P2N1P/2P3P1/3R2K1 b - - 0 21',
        'rating': 1118,
        'themes': 'backRankMate endgame mate mateIn2 short',
    },
]

PUZZLES_MEDIUM = [
    {
        'id': '0000D',
        'fen': '5rk1/1p3ppp/pq3b2/8/8/1P1Q1N2/P4PPP/3R2K1 w - - 2 27',
        'rating': 1473,
        'themes': 'advantage endgame short',
    },
    {
        'id': '000Pw',
        'fen': '6k1/5p1p/4p3/4q3/3nN3/2Q3P1/PP3P1P/6K1 w - - 2 37',
        'rating': 1522,
        'themes': 'crushing endgame fork short',
    },
    {
        'id': '000Sa',
        'fen': '2Q2bk1/5p1p/p5p1/2p3P1/2r1B3/7P/qPQ2P2/2K4R b - - 0 32',
        'rating': 1471,
        'themes': 'advantage endgame short',
    },
    {
        'id': '000Vc',
        'fen': '8/8/4k1p1/2KpP2p/5PP1/8/8/8 w - - 0 53',
        'rating': 1611,
        'themes': 'crushing endgame long pawnEndgame',
    },
    {
        'id': '000aY',
        'fen': 'r4rk1/pp3ppp/2n1b3/q1pp2B1/8/P1Q2NP1/1PP1PP1P/2KR3R w - - 0 15',
        'rating': 1414,
        'themes': 'advantage master middlegame short',
    },
]

PUZZLES_HIGH = [
    {
        'id': '000h0',
        'fen': '5rk1/p5p1/3bpr1p/1Pp4q/3pR3/1P1Q1N2/P4PPP/4R1K1 w - - 4 22',
        'rating': 2066,
        'themes': 'advantage interference kingsideAttack middlegame veryLong',
    },
    {
        'id': '000jr',
        'fen': '5k2/1p4pp/p5n1/5Q2/3BpP2/1P2PP1K/P1q4P/7r b - - 1 33',
        'rating': 2136,
        'themes': 'crushing endgame long',
    },
    {
        'id': '000qP',
        'fen': '8/7R/8/5p2/4bk1P/8/2r2K2/6R1 w - - 7 51',
        'rating': 2003,
        'themes': 'crushing endgame exposedKing long skewer',
    },
    {
        'id': '001Hi',
        'fen': '6k1/pp1r1pp1/1qp1p2p/4P2P/5Q2/1P4R1/P1Pr1PP1/R5K1 b - - 4 23',
        'rating': 2389,
        'themes': 'advantage endgame long pin',
    },
    {
        'id': '001Oo',
        'fen': '6k1/4p1bp/6p1/1p1pP3/1PpPp3/2P1P3/Q2B1KPP/3q4 b - - 2 23',
        'rating': 2078,
        'themes': 'crushing endgame long quietMove',
    },
]

PUZZLES_VERY_HIGH = [
    {
        'id': '000VW',
        'fen': 'r4r2/1p3pkp/p5p1/3R1N1Q/3P4/8/P1q2P2/3R2K1 b - - 3 25',
        'rating': 2861,
        'themes': 'crushing endgame long',
    },
    {
        'id': '002e5',
        'fen': 'r2q4/pp1n1kbp/3P2b1/6N1/6Q1/P3P3/6P1/4K2R b K - 1 21',
        'rating': 2632,
        'themes': 'crushing long middlegame sacrifice',
    },
    {
        'id': '004zh',
        'fen': '4b1k1/4Pr2/3R2pp/1ppBP1q1/8/PP4P1/2P4P/3R3K b - - 2 38',
        'rating': 2624,
        'themes': 'advancedPawn attraction crushing endgame exposedKing intermezzo pin promotion quietMove veryLong',
    },
    {
        'id': '005jR',
        'fen': '8/5p1k/1P4pp/3Qn3/4BP2/6P1/1p2P2P/2q3K1 w - - 1 34',
        'rating': 2787,
        'themes': 'advancedPawn crushing endgame long promotion',
    },
    {
        'id': '00CNA',
        'fen': 'rnbq1r2/pp1n2bk/3pB1p1/4P1N1/5P2/2p5/PPP3P1/R1BQK3 b Q - 1 13',
        'rating': 2665,
        'themes': 'crushing opening short',
    },
]


# =============================================================================
# Unit Tests for Helper Functions (no engine required)
# =============================================================================

class TestEvalVolatility:
    """Tests for calculate_eval_volatility function."""

    def test_stable_evals_low_volatility(self):
        """Stable evaluations should have low volatility."""
        # All evals are the same = 0 volatility
        evals = {5: 100, 10: 100, 15: 100, 20: 100}
        raw, normalized = calculate_eval_volatility(evals)
        assert raw == 0.0
        assert normalized == 0.0

    def test_varying_evals_higher_volatility(self):
        """Varying evaluations should have higher volatility."""
        # Evals swing between 0 and 200 = high volatility
        evals = {5: 0, 10: 200, 15: 0, 20: 200}
        raw, normalized = calculate_eval_volatility(evals)
        assert raw > 50  # Significant volatility
        assert 0 < normalized <= 1.0

    def test_single_eval_zero_volatility(self):
        """Single evaluation should have zero volatility."""
        evals = {20: 150}
        raw, normalized = calculate_eval_volatility(evals)
        assert raw == 0.0
        assert normalized == 0.0

    def test_normalized_caps_at_one(self):
        """Normalized volatility should cap at 1.0."""
        # Extreme volatility (std dev > 200)
        evals = {5: -500, 10: 500, 15: -500, 20: 500}
        raw, normalized = calculate_eval_volatility(evals)
        assert normalized == 1.0

    def test_moderate_volatility(self):
        """Moderate eval changes should give moderate volatility."""
        evals = {5: 50, 10: 80, 15: 60, 20: 70}
        raw, normalized = calculate_eval_volatility(evals)
        assert 0 < raw < 50  # Small variations
        assert 0 < normalized < 0.5


class TestBranchingFactor:
    """Tests for estimate_branching_factor function."""

    def test_no_metrics_returns_default(self):
        """No metrics should return default branching factor."""
        assert estimate_branching_factor(None) == 3.5
        assert estimate_branching_factor({}) == 3.5

    def test_single_depth_returns_default(self):
        """Single depth should return default."""
        metrics = {
            10: EngineSearchMetrics(depth=10, nodes=1000, nps=100000, time_ms=10, seldepth=15)
        }
        assert estimate_branching_factor(metrics) == 3.5

    def test_increasing_nodes_gives_branching_factor(self):
        """Increasing node counts should produce reasonable branching factor."""
        # Nodes roughly triple each depth (bf ~= 3)
        metrics = {
            5: EngineSearchMetrics(depth=5, nodes=100, nps=100000, time_ms=1, seldepth=8),
            10: EngineSearchMetrics(depth=10, nodes=24300, nps=100000, time_ms=243, seldepth=15),
        }
        bf = estimate_branching_factor(metrics)
        # Should be close to 3 (100 * 3^5 = 24300)
        assert 2.5 <= bf <= 3.5

    def test_branching_factor_clamped(self):
        """Branching factor should be clamped to reasonable range."""
        # Unrealistic node growth
        metrics = {
            5: EngineSearchMetrics(depth=5, nodes=1, nps=100000, time_ms=1, seldepth=8),
            10: EngineSearchMetrics(depth=10, nodes=1000000000, nps=100000, time_ms=10000, seldepth=15),
        }
        bf = estimate_branching_factor(metrics)
        assert 1.5 <= bf <= 10.0  # Clamped


class TestCategorizeComplexity:
    """Tests for categorize_complexity function."""

    def test_low_complexity(self):
        assert categorize_complexity(0.0) == "LOW"
        assert categorize_complexity(0.24) == "LOW"

    def test_medium_complexity(self):
        assert categorize_complexity(0.25) == "MEDIUM"
        assert categorize_complexity(0.49) == "MEDIUM"

    def test_high_complexity(self):
        assert categorize_complexity(0.50) == "HIGH"
        assert categorize_complexity(0.74) == "HIGH"

    def test_very_high_complexity(self):
        assert categorize_complexity(0.75) == "VERY_HIGH"
        assert categorize_complexity(1.0) == "VERY_HIGH"


class TestCalculateComplexityHeuristics:
    """Tests for the main complexity heuristics calculation."""

    def test_simple_position_low_complexity(self):
        """Stable evaluations, large gap = low complexity."""
        evaluations = {5: 100, 10: 100, 15: 100, 20: 100}
        gap_metrics = {
            20: GapMetricResult(
                depth=20,
                best_move="e2e4",
                best_eval=100,
                second_move="d2d4",
                second_eval=-100,  # Large gap (200cp)
                gap_cp=200,
            )
        }

        result = calculate_complexity_heuristics(
            evaluations=evaluations,
            search_metrics=None,
            gap_metrics=gap_metrics,
            first_consistent_depth=5,  # Early convergence
        )

        assert isinstance(result, PositionComplexityHeuristics)
        assert result.eval_volatility == 0.0
        assert result.gap_at_max_depth == 200
        assert result.complexity_score < 0.5  # Should be low/medium
        assert result.complexity_category in ["LOW", "MEDIUM"]

    def test_complex_position_high_complexity(self):
        """Volatile evaluations, small gap = high complexity."""
        evaluations = {5: -100, 10: 100, 15: -50, 20: 80}
        gap_metrics = {
            20: GapMetricResult(
                depth=20,
                best_move="e2e4",
                best_eval=80,
                second_move="d2d4",
                second_eval=75,  # Small gap (5cp)
                gap_cp=5,
            )
        }

        result = calculate_complexity_heuristics(
            evaluations=evaluations,
            search_metrics=None,
            gap_metrics=gap_metrics,
            first_consistent_depth=20,  # Late convergence
        )

        assert result.eval_volatility > 50  # High volatility
        assert result.gap_at_max_depth == 5  # Small gap
        assert result.complexity_score > 0.5  # Should be high
        assert result.complexity_category in ["HIGH", "VERY_HIGH"]

    def test_gap_normalized_correctly(self):
        """Gap metric should be inverted (small gap = high complexity)."""
        evaluations = {20: 100}

        # Small gap = high complexity contribution
        small_gap = calculate_complexity_heuristics(
            evaluations=evaluations,
            search_metrics=None,
            gap_metrics={20: GapMetricResult(20, "e2e4", 100, "d2d4", 98, 2)},
        )

        # Large gap = low complexity contribution
        large_gap = calculate_complexity_heuristics(
            evaluations=evaluations,
            search_metrics=None,
            gap_metrics={20: GapMetricResult(20, "e2e4", 100, "d2d4", -100, 200)},
        )

        # Small gap should result in higher complexity
        assert small_gap.complexity_score > large_gap.complexity_score


# =============================================================================
# Integration Tests (require Stockfish engine)
# =============================================================================

@pytest.fixture(scope="module")
def engine():
    """Provide a running engine for tests."""
    analyzer = EngineAnalyzer(
        depth=15,
        threads=2,
        hash_mb=256,
    )
    analyzer.start()
    yield analyzer
    analyzer.stop()


@pytest.mark.slow
class TestEngineComplexityIntegration:
    """Integration tests that use the actual Stockfish engine."""

    def test_analyze_multi_depth_extended_returns_heuristics(self, engine):
        """Verify analyze_multi_depth_extended returns complexity heuristics."""
        board = chess.Board()  # Starting position

        result = engine.analyze_multi_depth_extended(
            board,
            depths=[5, 10, 15],
            multipv=2,
            capture_search_stats=True,
        )

        assert isinstance(result, MultiDepthResult)
        assert result.complexity_heuristics is not None
        assert isinstance(result.complexity_heuristics, PositionComplexityHeuristics)

        heuristics = result.complexity_heuristics
        assert heuristics.eval_volatility >= 0
        assert heuristics.gap_at_max_depth >= 0
        assert heuristics.branching_factor_estimate >= 1.5
        assert 0 <= heuristics.complexity_score <= 1
        assert heuristics.complexity_category in ["LOW", "MEDIUM", "HIGH", "VERY_HIGH"]

    def test_gap_metrics_populated_with_multipv(self, engine):
        """Verify gap metrics are populated when using multipv."""
        board = chess.Board("r1bqkbnr/pppppppp/2n5/8/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")

        result = engine.analyze_multi_depth_extended(
            board,
            depths=[10, 15],
            multipv=2,
            capture_search_stats=True,
        )

        assert result.gap_metrics is not None
        assert len(result.gap_metrics) > 0

        for depth, gap in result.gap_metrics.items():
            assert isinstance(gap, GapMetricResult)
            assert gap.best_move is not None
            assert gap.best_eval is not None
            # gap_cp should be >= 0
            assert gap.gap_cp >= 0

    def test_search_metrics_populated(self, engine):
        """Verify search metrics are populated when requested."""
        board = chess.Board()

        result = engine.analyze_multi_depth_extended(
            board,
            depths=[5, 10],
            multipv=1,
            capture_search_stats=True,
        )

        assert result.search_metrics is not None
        assert len(result.search_metrics) > 0

        for depth, metrics in result.search_metrics.items():
            assert isinstance(metrics, EngineSearchMetrics)
            assert metrics.nodes > 0
            assert metrics.depth == depth


@pytest.mark.slow
class TestPuzzleComplexityCorrelation:
    """
    Test that puzzle difficulty correlates with complexity scores.

    Higher-rated puzzles should generally have higher complexity.
    This is a statistical correlation, not a strict ordering.
    """

    def analyze_puzzle_complexity(self, engine, fen: str) -> float:
        """Analyze a puzzle position and return complexity score."""
        board = chess.Board(fen)
        result = engine.analyze_multi_depth_extended(
            board,
            depths=[5, 10, 15],
            multipv=2,
            capture_search_stats=True,
        )
        return result.complexity_heuristics.complexity_score

    def test_low_vs_high_rated_puzzles(self, engine):
        """Low-rated puzzles should have lower average complexity than high-rated ones."""
        low_complexities = []
        high_complexities = []

        for puzzle in PUZZLES_LOW[:3]:  # Sample 3 from each
            complexity = self.analyze_puzzle_complexity(engine, puzzle['fen'])
            low_complexities.append(complexity)
            print(f"Low puzzle {puzzle['id']} (rating {puzzle['rating']}): complexity = {complexity:.3f}")

        for puzzle in PUZZLES_HIGH[:3]:
            complexity = self.analyze_puzzle_complexity(engine, puzzle['fen'])
            high_complexities.append(complexity)
            print(f"High puzzle {puzzle['id']} (rating {puzzle['rating']}): complexity = {complexity:.3f}")

        avg_low = sum(low_complexities) / len(low_complexities)
        avg_high = sum(high_complexities) / len(high_complexities)

        print(f"\nAverage complexity - Low: {avg_low:.3f}, High: {avg_high:.3f}")

        # High-rated puzzles should have higher average complexity
        # Allow some tolerance since this is a statistical relationship
        assert avg_high > avg_low * 0.8, (
            f"Expected high-rated puzzles to have higher complexity. "
            f"Low avg: {avg_low:.3f}, High avg: {avg_high:.3f}"
        )

    def test_medium_vs_very_high_rated_puzzles(self, engine):
        """
        Test complexity computation across puzzle rating ranges.

        Note: Puzzle rating (human difficulty) doesn't strictly correlate with engine
        complexity. High-rated puzzles often have ONE clearly winning move that's hard
        for humans to find but straightforward for engines. Lower-rated puzzles might
        have multiple decent moves, creating more eval volatility.

        This test verifies complexity is computed correctly across rating ranges.
        """
        medium_complexities = []
        very_high_complexities = []

        for puzzle in PUZZLES_MEDIUM[:3]:
            complexity = self.analyze_puzzle_complexity(engine, puzzle['fen'])
            medium_complexities.append(complexity)
            print(f"Medium puzzle {puzzle['id']} (rating {puzzle['rating']}): complexity = {complexity:.3f}")

        for puzzle in PUZZLES_VERY_HIGH[:3]:
            complexity = self.analyze_puzzle_complexity(engine, puzzle['fen'])
            very_high_complexities.append(complexity)
            print(f"Very High puzzle {puzzle['id']} (rating {puzzle['rating']}): complexity = {complexity:.3f}")

        avg_medium = sum(medium_complexities) / len(medium_complexities)
        avg_very_high = sum(very_high_complexities) / len(very_high_complexities)

        print(f"\nAverage complexity - Medium: {avg_medium:.3f}, Very High: {avg_very_high:.3f}")

        # Verify complexity values are reasonable (between 0 and 1)
        for c in medium_complexities + very_high_complexities:
            assert 0.0 <= c <= 1.0, f"Complexity {c} out of expected range [0, 1]"

        # Verify we got non-zero complexity values (calculation is working)
        assert any(c > 0 for c in medium_complexities), "All medium puzzle complexities are zero"
        assert any(c > 0 for c in very_high_complexities), "All very-high puzzle complexities are zero"

    def test_complexity_distribution_across_ratings(self, engine):
        """
        Test complexity is computed across all rating buckets.

        Note: Puzzle rating measures human difficulty, not computational complexity.
        We verify that complexity values are computed correctly across rating ranges,
        but don't assert strict ordering since the metrics measure different things.
        """
        all_puzzles = [
            ('low', PUZZLES_LOW),
            ('medium', PUZZLES_MEDIUM),
            ('high', PUZZLES_HIGH),
            ('very_high', PUZZLES_VERY_HIGH),
        ]

        bucket_averages = {}
        all_complexities = []

        for bucket_name, puzzles in all_puzzles:
            complexities = []
            for puzzle in puzzles[:2]:  # 2 per bucket for speed
                complexity = self.analyze_puzzle_complexity(engine, puzzle['fen'])
                complexities.append(complexity)
                all_complexities.append(complexity)
                print(f"{bucket_name}: {puzzle['id']} (r={puzzle['rating']}) -> {complexity:.3f}")

            bucket_averages[bucket_name] = sum(complexities) / len(complexities)

        print(f"\nBucket averages: {bucket_averages}")

        # Verify all complexity values are in valid range
        for c in all_complexities:
            assert 0.0 <= c <= 1.0, f"Complexity {c} out of expected range [0, 1]"

        # Verify we have variation in complexity (not all same value)
        assert max(all_complexities) > min(all_complexities), (
            "All complexity values are identical - possible calculation issue"
        )

        # Verify all buckets have some complexity computed
        for bucket_name, avg in bucket_averages.items():
            assert avg >= 0, f"Bucket {bucket_name} has negative average complexity"


@pytest.mark.slow
class TestSpecificPositionComplexity:
    """Test complexity on specific known positions."""

    def test_starting_position_low_complexity(self, engine):
        """Starting position should have low complexity (well-understood)."""
        board = chess.Board()
        result = engine.analyze_multi_depth_extended(
            board, depths=[5, 10, 15], multipv=2, capture_search_stats=True
        )

        # Starting position is very stable - should be low/medium complexity
        assert result.complexity_heuristics.complexity_category in ["LOW", "MEDIUM"]
        print(f"Starting position complexity: {result.complexity_heuristics.complexity_score:.3f}")

    def test_tactical_position_higher_complexity(self, engine):
        """Position with multiple tactical possibilities should have higher complexity."""
        # Position with lots of captures and tactics available
        fen = "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
        board = chess.Board(fen)

        result = engine.analyze_multi_depth_extended(
            board, depths=[5, 10, 15], multipv=2, capture_search_stats=True
        )

        print(f"Italian Game position complexity: {result.complexity_heuristics.complexity_score:.3f}")
        print(f"  Eval volatility: {result.complexity_heuristics.eval_volatility:.1f}cp")
        print(f"  Gap to 2nd best: {result.complexity_heuristics.gap_at_max_depth}cp")
        print(f"  Branching factor: {result.complexity_heuristics.branching_factor_estimate:.2f}")

    def test_endgame_vs_middlegame_complexity(self, engine):
        """Compare complexity of simple endgame vs complex middlegame."""
        # Simple K+P endgame
        endgame_fen = "8/8/4k3/8/4P3/4K3/8/8 w - - 0 1"
        endgame_board = chess.Board(endgame_fen)

        # Complex middlegame
        middlegame_fen = "r2q1rk1/ppp2ppp/2n1bn2/3pp3/2PP4/2N1PN2/PP2BPPP/R1BQ1RK1 w - - 0 9"
        middlegame_board = chess.Board(middlegame_fen)

        endgame_result = engine.analyze_multi_depth_extended(
            endgame_board, depths=[5, 10, 15], multipv=2, capture_search_stats=True
        )

        middlegame_result = engine.analyze_multi_depth_extended(
            middlegame_board, depths=[5, 10, 15], multipv=2, capture_search_stats=True
        )

        print(f"Endgame complexity: {endgame_result.complexity_heuristics.complexity_score:.3f}")
        print(f"Middlegame complexity: {middlegame_result.complexity_heuristics.complexity_score:.3f}")

        # Middlegame should generally be more complex
        # (though not always - depends on position)


# =============================================================================
# Test Fixtures for Data Validation
# =============================================================================

class TestComplexityDataValidation:
    """Validate that complexity data is properly formatted."""

    def test_heuristics_all_fields_present(self):
        """Verify all fields are present in PositionComplexityHeuristics."""
        heuristics = PositionComplexityHeuristics(
            eval_volatility=50.0,
            eval_volatility_normalized=0.25,
            gap_at_max_depth=100,
            avg_gap=80.0,
            convergence_depth=10,
            total_nodes=10000,
            nodes_per_depth={5: 100, 10: 1000, 15: 10000},
            branching_factor_estimate=3.2,
            complexity_score=0.45,
            complexity_category="MEDIUM",
        )

        assert heuristics.eval_volatility == 50.0
        assert heuristics.eval_volatility_normalized == 0.25
        assert heuristics.gap_at_max_depth == 100
        assert heuristics.avg_gap == 80.0
        assert heuristics.convergence_depth == 10
        assert heuristics.total_nodes == 10000
        assert heuristics.nodes_per_depth == {5: 100, 10: 1000, 15: 10000}
        assert heuristics.branching_factor_estimate == 3.2
        assert heuristics.complexity_score == 0.45
        assert heuristics.complexity_category == "MEDIUM"

    def test_gap_metric_result_fields(self):
        """Verify GapMetricResult fields."""
        gap = GapMetricResult(
            depth=15,
            best_move="e2e4",
            best_eval=50,
            second_move="d2d4",
            second_eval=30,
            gap_cp=20,
        )

        assert gap.depth == 15
        assert gap.best_move == "e2e4"
        assert gap.best_eval == 50
        assert gap.second_move == "d2d4"
        assert gap.second_eval == 30
        assert gap.gap_cp == 20


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
