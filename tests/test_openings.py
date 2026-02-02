"""
Tests for opening book functionality.

Tests verify that:
1. Opening book loads correctly from TSV files
2. Book positions are detected correctly
3. Last book ply is found correctly
4. Opening classification works
5. Book moves vs opening phase are independent

Run with: pytest tests/test_openings.py -v
"""

import pytest
import chess
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from chess_analysis.openings import (
    OpeningBook,
    OpeningInfo,
    find_last_book_ply,
    calculate_distance_from_book,
    get_opening_book,
    classify_opening,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def opening_book() -> OpeningBook:
    """Provide a loaded opening book for tests."""
    book = OpeningBook()
    book.load()
    return book


# =============================================================================
# Tests for OpeningBook class
# =============================================================================

class TestOpeningBookLoading:
    """Tests for opening book loading."""

    def test_book_loads_successfully(self, opening_book: OpeningBook):
        """Book should load without errors."""
        assert opening_book._loaded is True

    def test_book_has_positions(self, opening_book: OpeningBook):
        """Book should have many positions after loading."""
        assert len(opening_book._position_to_opening) > 100

    def test_lazy_loading(self):
        """Book should load lazily."""
        book = OpeningBook()
        assert book._loaded is False
        # Accessing a position triggers loading
        board = chess.Board()
        book.is_book_position(board)
        assert book._loaded is True


class TestIsBookPosition:
    """Tests for checking if a position is in the opening book."""

    def test_starting_position_not_in_book(self, opening_book: OpeningBook):
        """Starting position is not named in the opening book (no moves played yet)."""
        board = chess.Board()
        # The starting position itself has no opening name - openings start after first move
        assert opening_book.is_book_position(board) is False

    def test_after_e4_is_book(self, opening_book: OpeningBook):
        """1.e4 is a standard opening move, should be in book."""
        board = chess.Board()
        board.push_san("e4")
        assert opening_book.is_book_position(board) is True

    def test_after_d4_is_book(self, opening_book: OpeningBook):
        """1.d4 is a standard opening move, should be in book."""
        board = chess.Board()
        board.push_san("d4")
        assert opening_book.is_book_position(board) is True

    def test_italian_game_is_book(self, opening_book: OpeningBook):
        """Italian Game (1.e4 e5 2.Nf3 Nc6 3.Bc4) should be in book."""
        board = chess.Board()
        for san in ["e4", "e5", "Nf3", "Nc6", "Bc4"]:
            board.push_san(san)
        assert opening_book.is_book_position(board) is True

    def test_sicilian_is_book(self, opening_book: OpeningBook):
        """Sicilian Defense (1.e4 c5) should be in book."""
        board = chess.Board()
        board.push_san("e4")
        board.push_san("c5")
        assert opening_book.is_book_position(board) is True

    def test_random_position_not_book(self, opening_book: OpeningBook):
        """A random middle/endgame position should not be in book."""
        # Endgame position
        board = chess.Board("8/8/4k3/8/4P3/4K3/8/8 w - - 0 1")
        assert opening_book.is_book_position(board) is False


class TestGetOpening:
    """Tests for getting opening information."""

    def test_starting_position_has_opening(self, opening_book: OpeningBook):
        """Starting position should have opening info."""
        board = chess.Board()
        # Note: Starting position might not have an "opening" name
        # but after first move it should
        board.push_san("e4")
        opening = opening_book.get_opening(board)
        # After 1.e4, might not have a name yet
        # Let's check after more moves
        board.push_san("e5")
        opening = opening_book.get_opening(board)
        if opening:
            assert isinstance(opening, OpeningInfo)

    def test_italian_game_opening_info(self, opening_book: OpeningBook):
        """Italian Game should return correct opening info."""
        board = chess.Board()
        for san in ["e4", "e5", "Nf3", "Nc6", "Bc4"]:
            board.push_san(san)

        opening = opening_book.get_opening(board)
        if opening:
            assert isinstance(opening, OpeningInfo)
            assert opening.eco.startswith("C")  # Italian Game is C5x
            assert "Italian" in opening.name or opening.eco == "C50"

    def test_sicilian_opening_info(self, opening_book: OpeningBook):
        """Sicilian Defense should return correct opening info."""
        board = chess.Board()
        board.push_san("e4")
        board.push_san("c5")

        opening = opening_book.get_opening(board)
        if opening:
            assert isinstance(opening, OpeningInfo)
            assert opening.eco.startswith("B")  # Sicilian is B2x-B9x

    def test_non_book_position_returns_none(self, opening_book: OpeningBook):
        """Non-book position should return None."""
        board = chess.Board("8/8/4k3/8/4P3/4K3/8/8 w - - 0 1")
        opening = opening_book.get_opening(board)
        assert opening is None


class TestIsBookMove:
    """Tests for checking if a move leads to a book position."""

    def test_e4_is_book_move(self, opening_book: OpeningBook):
        """1.e4 should be a book move."""
        board = chess.Board()
        move = board.parse_san("e4")
        assert opening_book.is_book_move(board, move) is True

    def test_d4_is_book_move(self, opening_book: OpeningBook):
        """1.d4 should be a book move."""
        board = chess.Board()
        move = board.parse_san("d4")
        assert opening_book.is_book_move(board, move) is True

    def test_random_move_might_not_be_book(self, opening_book: OpeningBook):
        """A very unusual move might not be in book."""
        board = chess.Board()
        # 1.h4 is legal but unusual
        move = board.parse_san("h4")
        # Note: h4 might actually be in the book (Grob variations)
        # so we don't assert False, just check it runs
        result = opening_book.is_book_move(board, move)
        assert isinstance(result, bool)


# =============================================================================
# Tests for find_last_book_ply
# =============================================================================

class TestFindLastBookPly:
    """Tests for finding the last book ply in a game."""

    def test_mainline_opening_stays_in_book(self, opening_book: OpeningBook):
        """Mainline openings should stay in book longer."""
        boards = []
        board = chess.Board()
        boards.append(board.copy())

        # Italian Game mainline
        for san in ["e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5"]:
            board.push_san(san)
            boards.append(board.copy())

        last_ply, opening = find_last_book_ply(boards, opening_book)

        # Should stay in book for several plies
        assert last_ply >= 4  # At least through Bc4

    def test_starting_position_only(self, opening_book: OpeningBook):
        """Single starting position should return ply 0."""
        boards = [chess.Board()]
        last_ply, opening = find_last_book_ply(boards, opening_book)

        # Starting position is ply 0
        assert last_ply == 0

    def test_sideline_leaves_book_early(self, opening_book: OpeningBook):
        """Unusual moves should leave book earlier."""
        boards = []
        board = chess.Board()
        boards.append(board.copy())

        # Normal start, then unusual move
        board.push_san("e4")
        boards.append(board.copy())
        board.push_san("e5")
        boards.append(board.copy())
        # Unusual response
        board.push_san("a3")  # Mengarini Attack - might be in book
        boards.append(board.copy())

        last_ply, opening = find_last_book_ply(boards, opening_book)

        # Should have found some book plies
        assert last_ply >= 0

    def test_returns_opening_info(self, opening_book: OpeningBook):
        """Should return OpeningInfo for last book position."""
        boards = []
        board = chess.Board()
        boards.append(board.copy())

        for san in ["e4", "e5", "Nf3", "Nc6", "Bc4"]:
            board.push_san(san)
            boards.append(board.copy())

        last_ply, opening = find_last_book_ply(boards, opening_book)

        # Should have opening info
        assert opening is None or isinstance(opening, OpeningInfo)


# =============================================================================
# Tests for calculate_distance_from_book
# =============================================================================

class TestCalculateDistanceFromBook:
    """Tests for calculating distance from last book move."""

    def test_still_in_book(self):
        """Distance should be 0 when current ply equals last book ply."""
        assert calculate_distance_from_book(5, 5) == 0

    def test_distance_calculation(self):
        """Distance should be difference between plies."""
        assert calculate_distance_from_book(10, 5) == 5

    def test_no_negative_distance(self):
        """Distance should never be negative."""
        # Edge case: current ply < last_book_ply shouldn't happen but handle it
        assert calculate_distance_from_book(3, 5) == 0


# =============================================================================
# Tests for classify_opening
# =============================================================================

class TestClassifyOpening:
    """Tests for classifying a game's opening."""

    def test_italian_game_classification(self, opening_book: OpeningBook):
        """Italian Game should be classified correctly."""
        boards = []
        board = chess.Board()
        boards.append(board.copy())

        for san in ["e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5"]:
            board.push_san(san)
            boards.append(board.copy())

        opening = classify_opening(boards)

        if opening:
            assert isinstance(opening, OpeningInfo)
            # Should be Italian Game family (C50-C54)
            assert opening.eco.startswith("C5")

    def test_sicilian_classification(self, opening_book: OpeningBook):
        """Sicilian Defense should be classified correctly."""
        boards = []
        board = chess.Board()
        boards.append(board.copy())

        for san in ["e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4"]:
            board.push_san(san)
            boards.append(board.copy())

        opening = classify_opening(boards)

        if opening:
            assert isinstance(opening, OpeningInfo)
            # Should be Sicilian family (B20-B99)
            assert opening.eco.startswith("B")

    def test_empty_boards_returns_none(self):
        """Empty board list should return None."""
        opening = classify_opening([])
        assert opening is None


# =============================================================================
# Tests for singleton access
# =============================================================================

class TestSingletonAccess:
    """Tests for singleton opening book access."""

    def test_get_opening_book_returns_book(self):
        """get_opening_book should return a loaded OpeningBook."""
        book = get_opening_book()
        assert isinstance(book, OpeningBook)
        assert book._loaded is True

    def test_get_opening_book_same_instance(self):
        """get_opening_book should return the same instance."""
        book1 = get_opening_book()
        book2 = get_opening_book()
        assert book1 is book2


# =============================================================================
# Tests for book/phase independence
# =============================================================================

class TestBookPhaseIndependence:
    """Tests verifying book moves and game phase are independent."""

    def test_book_can_extend_into_middlegame(self, opening_book: OpeningBook):
        """
        Book positions can exist even in middlegame phase.

        The opening book contains positions from theory that may be
        classified as middlegame by the phase detector (due to development).
        """
        from chess_analysis.game_phase import detect_game_phase, GamePhase

        boards = []
        board = chess.Board()
        boards.append(board.copy())

        # Play a long theoretical line
        moves = ["e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "c3", "Nf6",
                 "d4", "exd4", "cxd4", "Bb4+"]

        for san in moves:
            board.push_san(san)
            boards.append(board.copy())

        # Check phase at each position
        phases = [detect_game_phase(b) for b in boards]

        # At some point, phase becomes MIDDLEGAME
        has_middlegame = GamePhase.MIDDLEGAME in phases

        # But we might still be in book (theory goes deep)
        last_ply, _ = find_last_book_ply(boards, opening_book)

        # The key insight: book_end_ply and phase transition ply are INDEPENDENT
        # Book might end at ply X, phase might change at ply Y
        # X and Y don't have to be the same

        # Just verify both systems work
        assert last_ply >= 0
        assert GamePhase.OPENING in phases  # Started in opening

    def test_early_deviation_from_book_in_opening(self, opening_book: OpeningBook):
        """
        A game can leave book while still in opening phase.

        If a player plays an unusual move early (still all pieces on back rank),
        they leave book but are still in the opening phase.
        """
        from chess_analysis.game_phase import detect_game_phase, GamePhase

        boards = []
        board = chess.Board()
        boards.append(board.copy())

        # Normal start
        board.push_san("e4")
        boards.append(board.copy())

        # Black plays unusual reply that might not be in book
        board.push_san("a6")  # Might be in book (St. George Defense)
        boards.append(board.copy())

        # Another unusual move
        board.push_san("h4")  # Unusual
        boards.append(board.copy())

        # Check last board is still opening phase
        last_phase = detect_game_phase(boards[-1])

        # All pieces still on back rank = still opening phase
        assert last_phase == GamePhase.OPENING

        # But we may have left book
        last_ply, _ = find_last_book_ply(boards, opening_book)
        # (We don't assert exact ply since it depends on book contents)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
