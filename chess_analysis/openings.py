"""
Opening book lookup using Lichess chess-openings database.

Provides functions to:
- Load opening book from TSV files (a-e.tsv)
- Check if a move sequence is book
- Find the last book move in a game
- Calculate distance from book
- Get opening classification (ECO code + name)

Source: https://github.com/lichess-org/chess-openings
"""

import chess
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
import re


# Default path relative to project root
DEFAULT_OPENINGS_PATH = Path(__file__).parent.parent / "submodules" / "chess-openings"


@dataclass
class OpeningInfo:
    """Information about an opening."""
    eco: str        # ECO code (e.g., "B90")
    name: str       # Opening name (e.g., "Sicilian Defense: Najdorf Variation")
    pgn: str        # Move sequence in SAN
    ply_count: int  # Number of half-moves in the opening line


class OpeningBook:
    """
    Opening book loaded from Lichess chess-openings TSV files.

    Stores openings by position hash for O(1) lookup.
    Each position maps to the most specific (longest) opening name for that position.
    """

    def __init__(self, openings_path: Optional[Path] = None):
        self.openings_path = openings_path or DEFAULT_OPENINGS_PATH
        self._position_to_opening: dict[str, OpeningInfo] = {}
        self._loaded = False

    def load(self) -> None:
        """Load all opening TSV files (a.tsv through e.tsv)."""
        if self._loaded:
            return

        for letter in "abcde":
            tsv_path = self.openings_path / f"{letter}.tsv"
            if tsv_path.exists():
                self._load_tsv(tsv_path)

        self._loaded = True
        print(f"Loaded {len(self._position_to_opening)} opening positions")

    def _load_tsv(self, path: Path) -> None:
        """Parse a single TSV file and index by position."""
        with open(path, "r", encoding="utf-8") as f:
            # Skip header
            header = next(f, None)

            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split("\t")
                if len(parts) < 3:
                    continue

                eco, name, pgn = parts[0], parts[1], parts[2]

                # Play through moves to get final position
                board = chess.Board()
                try:
                    # Parse PGN moves - strip move numbers
                    moves_str = pgn.strip()
                    # Remove move numbers like "1." "2." etc.
                    san_moves = re.findall(r'[A-Za-z][A-Za-z0-9+#=\-]*', moves_str)

                    for san in san_moves:
                        if san in ('O-O', 'O-O-O'):
                            # Handle castling notation
                            move = board.parse_san(san)
                        else:
                            move = board.parse_san(san)
                        board.push(move)

                    # Index by board FEN (position only, not castling/ep for simpler matching)
                    fen_key = board.board_fen()
                    ply = len(board.move_stack)

                    # Keep the longest (most specific) opening for each position
                    existing = self._position_to_opening.get(fen_key)
                    if existing is None or ply >= existing.ply_count:
                        self._position_to_opening[fen_key] = OpeningInfo(
                            eco=eco,
                            name=name,
                            pgn=pgn,
                            ply_count=ply
                        )

                except (ValueError, chess.InvalidMoveError, chess.IllegalMoveError) as e:
                    # Skip invalid move sequences
                    continue

    def is_book_position(self, board: chess.Board) -> bool:
        """Check if current position is in the opening book."""
        if not self._loaded:
            self.load()
        return board.board_fen() in self._position_to_opening

    def get_opening(self, board: chess.Board) -> Optional[OpeningInfo]:
        """
        Get opening info for current position if it's a book position.

        Returns:
            OpeningInfo with ECO code, name, and PGN, or None if not in book.
        """
        if not self._loaded:
            self.load()
        return self._position_to_opening.get(board.board_fen())

    def is_book_move(self, board: chess.Board, move: chess.Move) -> bool:
        """
        Check if making this move leads to a book position.

        Args:
            board: Position before the move
            move: Move to check

        Returns:
            True if the resulting position is in the opening book
        """
        if not self._loaded:
            self.load()
        board_copy = board.copy()
        board_copy.push(move)
        return board_copy.board_fen() in self._position_to_opening

    def get_opening_after_move(self, board: chess.Board, move: chess.Move) -> Optional[OpeningInfo]:
        """
        Get opening info for position after making a move.

        Args:
            board: Position before the move
            move: Move to make

        Returns:
            OpeningInfo if resulting position is in book, None otherwise
        """
        if not self._loaded:
            self.load()
        board_copy = board.copy()
        board_copy.push(move)
        return self._position_to_opening.get(board_copy.board_fen())


def find_last_book_ply(boards: list[chess.Board], book: Optional[OpeningBook] = None) -> tuple[int, Optional[OpeningInfo]]:
    """
    Find the last ply that is still in the opening book.

    Args:
        boards: List of board positions throughout the game (including starting position)
        book: Optional OpeningBook instance (uses singleton if not provided)

    Returns:
        Tuple of (last_book_ply, opening_info) where:
        - last_book_ply: Ply number (0 = starting position) of last book position
        - opening_info: OpeningInfo for the last book position, or None
    """
    if book is None:
        book = get_opening_book()

    last_book_ply = 0
    last_opening: Optional[OpeningInfo] = None

    for ply, board in enumerate(boards):
        opening = book.get_opening(board)
        if opening is not None:
            last_book_ply = ply
            last_opening = opening

    return last_book_ply, last_opening


def calculate_distance_from_book(current_ply: int, last_book_ply: int) -> int:
    """
    Calculate half-moves (plys) since the last book move.

    Args:
        current_ply: Current ply number
        last_book_ply: Ply of last book move

    Returns:
        Number of plys since last book move (0 if still in book)
    """
    return max(0, current_ply - last_book_ply)


# Singleton instance for convenience
_default_book: Optional[OpeningBook] = None


def get_opening_book() -> OpeningBook:
    """
    Get the singleton opening book instance.

    The book is loaded lazily on first access.
    """
    global _default_book
    if _default_book is None:
        _default_book = OpeningBook()
        _default_book.load()
    return _default_book


def classify_opening(boards: list[chess.Board]) -> Optional[OpeningInfo]:
    """
    Classify a game's opening based on the board positions.

    Returns the most specific opening classification found.

    Args:
        boards: List of board positions throughout the game

    Returns:
        OpeningInfo for the game's opening, or None if no book positions found
    """
    _, opening = find_last_book_ply(boards)
    return opening
