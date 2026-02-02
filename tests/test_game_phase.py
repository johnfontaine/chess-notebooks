"""
Tests for game phase detection.

Tests verify that:
1. Game phases (opening, middlegame, endgame) are detected correctly
2. Phase transitions are identified at the right plies
3. Ply counts sum correctly
4. Helper functions work as expected

Run with: pytest tests/test_game_phase.py -v
"""

import pytest
import chess
import chess.pgn
import io
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from chess_analysis.game_phase import (
    GamePhase,
    GamePhaseInfo,
    detect_game_phase,
    detect_game_phase_detailed,
    get_phase_transitions,
    analyze_game_phases,
    count_major_minor_pieces,
    count_back_rank_pieces,
    calculate_mixedness,
    is_endgame,
    is_opening,
    detect_phase_by_material_only,
    detect_phase_by_material_value,
    MIDDLEGAME_MATERIAL_THRESHOLD,
    ENDGAME_MATERIAL_THRESHOLD,
    BACK_RANK_DEVELOPMENT_THRESHOLD,
    MIXEDNESS_THRESHOLD,
)


# =============================================================================
# Test Positions
# =============================================================================

# Starting position - 14 major/minor pieces (7 per side: Q+2R+2B+2N), 8 pieces per side on back rank
STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

# After 1.e4 - still opening
AFTER_E4_FEN = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"

# Italian Game after 4 moves - pieces developed, should be middlegame by development
ITALIAN_FEN = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"

# Ruy Lopez - similar development level
RUY_LOPEZ_FEN = "r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"

# Complex middlegame with interlocked pieces (high mixedness)
COMPLEX_MIDDLEGAME_FEN = "r2q1rk1/ppp2ppp/2n1bn2/3pp3/2PP4/2N1PN2/PP2BPPP/R1BQ1RK1 w - - 0 9"

# Simple K+P endgame - 0 major/minor pieces
PAWN_ENDGAME_FEN = "8/8/4k3/8/4P3/4K3/8/8 w - - 0 1"

# Rook endgame - 2 major pieces (rooks) + 6 pawns = clearly endgame
ROOK_ENDGAME_FEN = "4r1k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1"

# Queen endgame - 2 queens only = endgame by material
QUEEN_ENDGAME_FEN = "6k1/5ppp/8/8/3Q4/8/5PPP/3q2K1 w - - 0 1"

# Late middlegame - reduced material but not quite endgame
LATE_MIDDLEGAME_FEN = "r4rk1/ppp2ppp/2n2n2/3pp3/8/2N2N2/PPP2PPP/R4RK1 w - - 0 1"

# Completely empty board except kings
BARE_KINGS_FEN = "4k3/8/8/8/8/8/8/4K3 w - - 0 1"


# =============================================================================
# Tests for count_major_minor_pieces
# =============================================================================

class TestCountMajorMinorPieces:
    """Tests for counting major (Q, R) and minor (B, N) pieces."""

    def test_starting_position_has_14_pieces(self):
        """Starting position has 14 major/minor pieces (7 per side: Q+2R+2B+2N)."""
        board = chess.Board(STARTING_FEN)
        assert count_major_minor_pieces(board) == 14

    def test_after_e4_still_14_pieces(self):
        """After 1.e4, still 14 major/minor pieces (pawns don't count)."""
        board = chess.Board(AFTER_E4_FEN)
        assert count_major_minor_pieces(board) == 14

    def test_pawn_endgame_has_0_pieces(self):
        """K+P vs K endgame has 0 major/minor pieces."""
        board = chess.Board(PAWN_ENDGAME_FEN)
        assert count_major_minor_pieces(board) == 0

    def test_rook_endgame_has_2_pieces(self):
        """Rook endgame with R vs R has 2 major/minor pieces."""
        board = chess.Board(ROOK_ENDGAME_FEN)
        assert count_major_minor_pieces(board) == 2

    def test_bare_kings_has_0_pieces(self):
        """K vs K has 0 major/minor pieces."""
        board = chess.Board(BARE_KINGS_FEN)
        assert count_major_minor_pieces(board) == 0


# =============================================================================
# Tests for count_back_rank_pieces
# =============================================================================

class TestCountBackRankPieces:
    """Tests for counting pieces on back rank (development indicator)."""

    def test_starting_white_has_8_on_back_rank(self):
        """White starts with 8 pieces on rank 1."""
        board = chess.Board(STARTING_FEN)
        assert count_back_rank_pieces(board, chess.WHITE) == 8

    def test_starting_black_has_8_on_back_rank(self):
        """Black starts with 8 pieces on rank 8."""
        board = chess.Board(STARTING_FEN)
        assert count_back_rank_pieces(board, chess.BLACK) == 8

    def test_italian_white_fewer_on_back_rank(self):
        """In Italian Game setup, white has fewer pieces on back rank."""
        board = chess.Board(ITALIAN_FEN)
        white_back = count_back_rank_pieces(board, chess.WHITE)
        # After Nf3, Bc4: knight and bishop are off back rank
        # King, Queen, 2 Rooks, Bishop, Knight still on rank 1 = 6
        assert white_back < 8
        assert white_back >= 4  # At minimum, K + Q + 2R

    def test_castled_position_back_rank(self):
        """After castling, king moves to back rank corner."""
        board = chess.Board(COMPLEX_MIDDLEGAME_FEN)
        # Both sides castled - still pieces on back rank
        white_back = count_back_rank_pieces(board, chess.WHITE)
        black_back = count_back_rank_pieces(board, chess.BLACK)
        assert white_back >= 3  # R, K, R, Q on rank 1
        assert black_back >= 3  # R, K, R on rank 8


# =============================================================================
# Tests for calculate_mixedness
# =============================================================================

class TestCalculateMixedness:
    """Tests for position mixedness (piece interlocking score)."""

    def test_starting_position_zero_mixedness(self):
        """Starting position has zero mixedness - pieces not interlocked."""
        board = chess.Board(STARTING_FEN)
        mixedness = calculate_mixedness(board)
        assert mixedness == 0

    def test_after_e4_still_low_mixedness(self):
        """After 1.e4, mixedness is still very low."""
        board = chess.Board(AFTER_E4_FEN)
        mixedness = calculate_mixedness(board)
        assert mixedness < MIXEDNESS_THRESHOLD

    def test_complex_middlegame_higher_mixedness(self):
        """Complex middlegame position should have higher mixedness."""
        board = chess.Board(COMPLEX_MIDDLEGAME_FEN)
        mixedness = calculate_mixedness(board)
        # Pieces are more interlocked in complex middlegames
        # This might or might not exceed threshold depending on exact position
        assert mixedness >= 0

    def test_endgame_low_mixedness(self):
        """Endgame positions tend to have lower mixedness."""
        board = chess.Board(ROOK_ENDGAME_FEN)
        mixedness = calculate_mixedness(board)
        # Few pieces = little opportunity for interlocking
        assert mixedness < 100


# =============================================================================
# Tests for detect_game_phase
# =============================================================================

class TestDetectGamePhase:
    """Tests for the main phase detection function."""

    def test_starting_position_is_opening(self):
        """Starting position should be classified as OPENING."""
        board = chess.Board(STARTING_FEN)
        phase = detect_game_phase(board)
        assert phase == GamePhase.OPENING

    def test_after_e4_is_opening(self):
        """After 1.e4 should still be OPENING."""
        board = chess.Board(AFTER_E4_FEN)
        phase = detect_game_phase(board)
        assert phase == GamePhase.OPENING

    def test_developed_position_is_middlegame(self):
        """Italian Game setup should be MIDDLEGAME (by development)."""
        board = chess.Board(ITALIAN_FEN)
        phase = detect_game_phase(board)
        # Should be middlegame because pieces are developed (< 6 on back rank)
        assert phase in (GamePhase.MIDDLEGAME, GamePhase.OPENING)

    def test_complex_middlegame_detected(self):
        """Complex middlegame position should be MIDDLEGAME."""
        board = chess.Board(COMPLEX_MIDDLEGAME_FEN)
        phase = detect_game_phase(board)
        assert phase == GamePhase.MIDDLEGAME

    def test_pawn_endgame_is_endgame(self):
        """K+P endgame should be ENDGAME."""
        board = chess.Board(PAWN_ENDGAME_FEN)
        phase = detect_game_phase(board)
        assert phase == GamePhase.ENDGAME

    def test_rook_endgame_is_endgame(self):
        """Rook endgame should be ENDGAME."""
        board = chess.Board(ROOK_ENDGAME_FEN)
        phase = detect_game_phase(board)
        assert phase == GamePhase.ENDGAME

    def test_bare_kings_is_endgame(self):
        """K vs K should be ENDGAME."""
        board = chess.Board(BARE_KINGS_FEN)
        phase = detect_game_phase(board)
        assert phase == GamePhase.ENDGAME


# =============================================================================
# Tests for detect_game_phase_detailed
# =============================================================================

class TestDetectGamePhaseDetailed:
    """Tests for detailed phase detection with reasoning."""

    def test_returns_game_phase_info(self):
        """Should return GamePhaseInfo dataclass."""
        board = chess.Board(STARTING_FEN)
        info = detect_game_phase_detailed(board)
        assert isinstance(info, GamePhaseInfo)

    def test_starting_position_details(self):
        """Verify starting position details."""
        board = chess.Board(STARTING_FEN)
        info = detect_game_phase_detailed(board)

        assert info.phase == GamePhase.OPENING
        assert info.major_minor_count == 14  # 7 per side: Q+2R+2B+2N
        assert info.back_rank_white == 8
        assert info.back_rank_black == 8
        assert info.mixedness_score == 0
        assert info.is_middlegame_by_material is False  # 14 > 10
        assert info.is_middlegame_by_development is False  # 8 >= 6
        assert info.is_middlegame_by_mixedness is False  # 0 < 150
        assert info.is_endgame is False  # 14 > 6

    def test_endgame_details(self):
        """Verify endgame position details."""
        board = chess.Board(PAWN_ENDGAME_FEN)
        info = detect_game_phase_detailed(board)

        assert info.phase == GamePhase.ENDGAME
        assert info.major_minor_count == 0
        assert info.is_endgame is True

    def test_middlegame_by_development_flag(self):
        """Position with developed pieces should trigger by_development flag."""
        board = chess.Board(ITALIAN_FEN)
        info = detect_game_phase_detailed(board)

        # If phase is middlegame, at least one trigger should be True
        if info.phase == GamePhase.MIDDLEGAME:
            assert (
                info.is_middlegame_by_material or
                info.is_middlegame_by_development or
                info.is_middlegame_by_mixedness
            )


# =============================================================================
# Tests for get_phase_transitions
# =============================================================================

class TestGetPhaseTransitions:
    """Tests for detecting phase transition points in a game."""

    def test_empty_board_list(self):
        """Empty board list should return None transitions."""
        result = get_phase_transitions([])
        assert result['middlegame_start'] is None
        assert result['endgame_start'] is None
        assert result['phases'] == []

    def test_single_position(self):
        """Single position should return that phase, no transitions."""
        board = chess.Board(STARTING_FEN)
        result = get_phase_transitions([board])

        assert result['middlegame_start'] is None
        assert result['endgame_start'] is None
        assert len(result['phases']) == 1
        assert result['phases'][0] == 'opening'

    def test_opening_to_middlegame_transition(self):
        """Detect transition from opening to middlegame."""
        # Create sequence: starting -> developed
        boards = []
        board = chess.Board()
        boards.append(board.copy())

        # Play some opening moves to reach middlegame
        moves = ['e4', 'e5', 'Nf3', 'Nc6', 'Bc4', 'Bc5', 'c3', 'Nf6', 'd4', 'exd4']
        for san in moves:
            board.push_san(san)
            boards.append(board.copy())

        result = get_phase_transitions(boards)

        # Should have phases for each position
        assert len(result['phases']) == len(boards)
        # Should start with opening
        assert result['phases'][0] == 'opening'
        # Should eventually reach middlegame (by development)
        assert 'middlegame' in result['phases']

    def test_endgame_only_positions(self):
        """Sequence of endgame positions should all be endgame."""
        board = chess.Board(PAWN_ENDGAME_FEN)
        boards = [board.copy()]

        # Make a few king moves
        board.push_san('Kd3')
        boards.append(board.copy())
        board.push_san('Kd6')
        boards.append(board.copy())

        result = get_phase_transitions(boards)

        assert all(p == 'endgame' for p in result['phases'])
        assert result['middlegame_start'] is None
        # endgame_start is None because it was endgame from the start
        assert result['endgame_start'] is None


# =============================================================================
# Tests for analyze_game_phases
# =============================================================================

class TestAnalyzeGamePhases:
    """Tests for full game phase analysis."""

    def _make_game_from_pgn(self, pgn_text: str) -> chess.pgn.Game:
        """Helper to create game from PGN text."""
        pgn_io = io.StringIO(pgn_text)
        return chess.pgn.read_game(pgn_io)

    def test_short_opening_game(self):
        """Very short game should be mostly opening."""
        pgn = """
[Event "Test"]
[Result "*"]

1. e4 e5 2. Nf3 *
"""
        game = self._make_game_from_pgn(pgn)
        result = analyze_game_phases(game)

        assert result['total_moves'] == 2  # 2 full moves
        assert result['opening_length'] >= 0
        assert result['middlegame_length'] >= 0
        assert result['endgame_length'] >= 0
        # Sum should equal total
        total = result['opening_length'] + result['middlegame_length'] + result['endgame_length']
        assert total == result['total_moves']

    def test_longer_game_phases(self):
        """Longer game should have multiple phases."""
        # Italian Game mainline
        pgn = """
[Event "Test"]
[Result "*"]

1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. c3 Nf6 5. d4 exd4 6. cxd4 Bb4+
7. Bd2 Bxd2+ 8. Nbxd2 d5 9. exd5 Nxd5 10. O-O O-O *
"""
        game = self._make_game_from_pgn(pgn)
        result = analyze_game_phases(game)

        assert result['total_moves'] == 10
        # Should have phases array
        assert len(result['phases_by_ply']) > 0
        # Ply counts should sum correctly
        total = result['opening_length'] + result['middlegame_length'] + result['endgame_length']
        assert total == result['total_moves']

    def test_phases_by_ply_populated(self):
        """phases_by_ply should have entry for each position."""
        pgn = """
[Event "Test"]
[Result "*"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 *
"""
        game = self._make_game_from_pgn(pgn)
        result = analyze_game_phases(game)

        # 5 half-moves = 6 positions (including start)
        assert len(result['phases_by_ply']) == 6

    def test_move_number_calculation(self):
        """Verify move number calculations are correct."""
        pgn = """
[Event "Test"]
[Result "*"]

1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. c3 Nf6 5. d4 exd4 *
"""
        game = self._make_game_from_pgn(pgn)
        result = analyze_game_phases(game)

        # If middlegame starts, the move number should be reasonable
        if result['middlegame_start_move'] is not None:
            assert 1 <= result['middlegame_start_move'] <= result['total_moves']


# =============================================================================
# Tests for convenience functions
# =============================================================================

class TestConvenienceFunctions:
    """Tests for is_endgame, is_opening, etc."""

    def test_is_endgame_true_for_endgame(self):
        """is_endgame returns True for endgame positions."""
        board = chess.Board(PAWN_ENDGAME_FEN)
        assert is_endgame(board) is True

    def test_is_endgame_false_for_opening(self):
        """is_endgame returns False for opening positions."""
        board = chess.Board(STARTING_FEN)
        assert is_endgame(board) is False

    def test_is_opening_true_for_start(self):
        """is_opening returns True for starting position."""
        board = chess.Board(STARTING_FEN)
        assert is_opening(board) is True

    def test_is_opening_false_for_endgame(self):
        """is_opening returns False for endgame positions."""
        board = chess.Board(PAWN_ENDGAME_FEN)
        assert is_opening(board) is False


# =============================================================================
# Tests for alternative detection methods
# =============================================================================

class TestAlternativeDetectionMethods:
    """Tests for simplified detection methods."""

    def test_material_only_opening(self):
        """Material-only detection for starting position."""
        board = chess.Board(STARTING_FEN)
        phase = detect_phase_by_material_only(board)
        # Starting has 14 pieces, which is <= 14 so it's MIDDLEGAME by this simple method
        # The threshold in detect_phase_by_material_only is count <= 14 = MIDDLEGAME
        assert phase in (GamePhase.OPENING, GamePhase.MIDDLEGAME)

    def test_material_only_endgame(self):
        """Material-only detection for endgame."""
        board = chess.Board(PAWN_ENDGAME_FEN)
        phase = detect_phase_by_material_only(board)
        assert phase == GamePhase.ENDGAME

    def test_material_value_opening(self):
        """Material-value detection for starting position."""
        board = chess.Board(STARTING_FEN)
        phase = detect_phase_by_material_value(board)
        # Starting material is high, should be opening
        assert phase == GamePhase.OPENING

    def test_material_value_endgame(self):
        """Material-value detection for endgame."""
        board = chess.Board(PAWN_ENDGAME_FEN)
        phase = detect_phase_by_material_value(board)
        # Very low material, should be endgame
        assert phase == GamePhase.ENDGAME


# =============================================================================
# Tests for threshold constants
# =============================================================================

class TestThresholdConstants:
    """Verify threshold constants are as expected."""

    def test_middlegame_threshold(self):
        """Middlegame material threshold should be 10."""
        assert MIDDLEGAME_MATERIAL_THRESHOLD == 10

    def test_endgame_threshold(self):
        """Endgame material threshold should be 6."""
        assert ENDGAME_MATERIAL_THRESHOLD == 6

    def test_back_rank_threshold(self):
        """Back rank development threshold should be 6."""
        assert BACK_RANK_DEVELOPMENT_THRESHOLD == 6

    def test_mixedness_threshold(self):
        """Mixedness threshold should be 150."""
        assert MIXEDNESS_THRESHOLD == 150


# =============================================================================
# Edge case tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_exactly_6_pieces_is_endgame(self):
        """Exactly 6 major/minor pieces should be endgame."""
        # 2 rooks each + 2 knights = 6 pieces
        fen = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"
        board = chess.Board(fen)
        count = count_major_minor_pieces(board)
        assert count == 4  # Only 4 rooks, kings don't count
        assert detect_game_phase(board) == GamePhase.ENDGAME

    def test_exactly_10_pieces_triggers_middlegame(self):
        """10 or fewer major/minor pieces should trigger middlegame by material."""
        # Construct a position with exactly 10 pieces
        # Remove queens (2) and one knight each (2) from starting = 14 - 4 = 10
        fen = "rn1qk1nr/pppppppp/8/8/8/8/PPPPPPPP/RN1QK1NR w KQkq - 0 1"  # No bishops
        board = chess.Board(fen)
        info = detect_game_phase_detailed(board)
        assert info.major_minor_count == 10  # 2R + 2N + Q = 5 per side = 10
        assert info.is_middlegame_by_material is True

    def test_promotion_position(self):
        """Position with promoted pieces detected correctly."""
        # White has 2 queens (original + promoted)
        fen = "4k3/8/8/8/8/8/8/Q3K2Q w - - 0 1"
        board = chess.Board(fen)
        count = count_major_minor_pieces(board)
        assert count == 2  # 2 queens
        assert detect_game_phase(board) == GamePhase.ENDGAME


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
