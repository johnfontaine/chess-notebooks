"""
Tactical and positional theme detection for chess positions.

This module provides functions to detect common tactical motifs like forks, pins,
skewers, and discovered attacks, as well as positional themes.

Attribution:
-----------
The detection algorithms in this module are adapted from the Lichess Puzzler project:
    https://github.com/ornicar/lichess-puzzler
    License: AGPL-3.0

The original tagger code can be found at:
    https://github.com/ornicar/lichess-puzzler/tree/master/tagger

Key source files referenced:
    - cook.py: Tactical theme detection (fork, pin, skewer, discovered attack, etc.)
    - tagger.py: Main tagging logic
    - model.py: Puzzle model definitions
    - util.py: Utility functions

Theme definitions are documented at:
    https://github.com/lichess-org/lila/blob/master/translation/source/puzzleTheme.xml

Lichess puzzle database with labeled themes:
    https://database.lichess.org/#puzzles
    https://huggingface.co/datasets/Lichess/chess-puzzles

References:
----------
- ChessTempo Tactical Motifs: https://chesstempo.com/tactical-motifs
- ChessTempo Positional Motifs: https://chesstempo.com/positional-motifs
- Lichess Puzzle Themes: https://lichess.org/training/themes
"""

import chess
from chess import (
    KING, QUEEN, ROOK, BISHOP, KNIGHT, PAWN,
    SquareSet, Square, Board, Move, Color, Piece,
    WHITE, BLACK,
)
from dataclasses import dataclass
from typing import Optional
from enum import Enum


# =============================================================================
# Constants and piece values
# =============================================================================

# Piece values including king (for comparison purposes)
PIECE_VALUES = {
    PAWN: 1,
    KNIGHT: 3,
    BISHOP: 3,
    ROOK: 5,
    QUEEN: 9,
    KING: 100,  # High value for attack detection, not material counting
}

# Piece types that can create ray attacks (pins, skewers, x-rays)
RAY_PIECE_TYPES = {BISHOP, ROOK, QUEEN}


# =============================================================================
# Theme enumeration
# =============================================================================

class TacticalTheme(Enum):
    """Tactical themes that can be detected in a position or move sequence."""
    # Attack patterns
    FORK = "fork"
    PIN = "pin"
    SKEWER = "skewer"
    DISCOVERED_ATTACK = "discoveredAttack"
    DISCOVERED_CHECK = "discoveredCheck"
    DOUBLE_CHECK = "doubleCheck"
    XRAY_ATTACK = "xRayAttack"

    # Capture themes
    HANGING_PIECE = "hangingPiece"
    TRAPPED_PIECE = "trappedPiece"
    CAPTURING_DEFENDER = "capturingDefender"

    # Forcing moves
    ATTRACTION = "attraction"
    DEFLECTION = "deflection"
    INTERFERENCE = "interference"
    INTERMEZZO = "intermezzo"
    CLEARANCE = "clearance"

    # Sacrifices
    SACRIFICE = "sacrifice"
    QUIET_MOVE = "quietMove"

    # Checkmate patterns
    BACK_RANK_MATE = "backRankMate"
    SMOTHERED_MATE = "smotheredMate"

    # Special moves
    EN_PASSANT = "enPassant"
    PROMOTION = "promotion"
    UNDERPROMOTION = "underpromotion"
    CASTLING = "castling"

    # Position types
    ZUGZWANG = "zugzwang"


class PositionalTheme(Enum):
    """Positional themes that can be detected in a position."""
    ADVANCED_PAWN = "advancedPawn"
    EXPOSED_KING = "exposedKing"
    KINGSIDE_ATTACK = "kingsideAttack"
    QUEENSIDE_ATTACK = "queensideAttack"
    ATTACKING_F2_F7 = "attackingF2F7"

    # Endgame types
    PAWN_ENDGAME = "pawnEndgame"
    KNIGHT_ENDGAME = "knightEndgame"
    BISHOP_ENDGAME = "bishopEndgame"
    ROOK_ENDGAME = "rookEndgame"
    QUEEN_ENDGAME = "queenEndgame"


@dataclass
class ThemeDetectionResult:
    """Result of theme detection for a position or move."""
    themes: list[str]
    details: dict  # Additional details about detected themes


# =============================================================================
# Utility functions
# =============================================================================

def get_piece_value(piece_type: int) -> int:
    """Get the value of a piece type."""
    return PIECE_VALUES.get(piece_type, 0)


def is_hanging(board: Board, piece: Piece, square: Square) -> bool:
    """
    Check if a piece is hanging (attacked and not adequately defended).

    A piece is considered hanging if:
    - It's attacked by the opponent
    - Either not defended, or attacked by a less valuable piece

    Adapted from lichess-puzzler/tagger/util.py
    """
    dominated_squares = get_dominated_squares(board, not piece.color)
    if square in dominated_squares:
        return True

    attackers = board.attackers(not piece.color, square)
    if not attackers:
        return False

    defenders = board.attackers(piece.color, square)
    if not defenders:
        return True

    # Check if attacked by less valuable piece
    min_attacker_value = min(
        get_piece_value(board.piece_type_at(sq))
        for sq in attackers
    )

    return min_attacker_value < get_piece_value(piece.piece_type)


def get_dominated_squares(board: Board, color: Color) -> SquareSet:
    """
    Get squares where color has attacking superiority.

    A square is dominated if:
    - Color attacks it more times than opponent defends it
    - Or color attacks with a less valuable piece
    """
    dominated = SquareSet()

    for square in chess.SQUARES:
        attackers = board.attackers(color, square)
        defenders = board.attackers(not color, square)

        if attackers and not defenders:
            dominated.add(square)
        elif attackers and defenders:
            min_attacker = min(get_piece_value(board.piece_type_at(sq)) for sq in attackers)
            min_defender = min(get_piece_value(board.piece_type_at(sq)) for sq in defenders)
            if min_attacker < min_defender:
                dominated.add(square)

    return dominated


def is_in_bad_spot(board: Board, square: Square) -> bool:
    """
    Check if a square is attacked by opponent and not safe.

    Adapted from lichess-puzzler/tagger/util.py
    """
    dominated = get_dominated_squares(board, board.turn)
    return square in dominated


def attacked_opponent_pieces(
    board: Board,
    from_square: Square,
    pov: Color
) -> list[tuple[Piece, Square]]:
    """
    Get list of opponent pieces attacked from a square.

    Returns list of (piece, square) tuples for opponent pieces under attack.

    Adapted from lichess-puzzler/tagger/util.py
    """
    result = []
    piece = board.piece_at(from_square)
    if not piece:
        return result

    # Get attacks from this square
    attacks = board.attacks(from_square)

    for target_square in attacks:
        target_piece = board.piece_at(target_square)
        if target_piece and target_piece.color != pov:
            result.append((target_piece, target_square))

    return result


def is_capture(board: Board, move: Move) -> bool:
    """Check if a move is a capture."""
    return board.is_capture(move)


def moved_piece_type(board: Board, move: Move) -> int:
    """Get the piece type that was moved."""
    piece = board.piece_at(move.from_square)
    return piece.piece_type if piece else PAWN


# =============================================================================
# Tactical theme detection functions
# =============================================================================

def detect_fork(
    board: Board,
    move: Move,
    pov: Color,
) -> bool:
    """
    Detect if a move creates a fork (attacking multiple pieces).

    A fork occurs when a single piece attacks two or more opponent pieces
    simultaneously, typically winning material.

    Adapted from lichess-puzzler/tagger/cook.py

    Args:
        board: Board state BEFORE the move
        move: The move to analyze
        pov: Point of view (color making the move)

    Returns:
        True if the move creates a fork
    """
    # Make the move on a copy
    test_board = board.copy()
    test_board.push(move)

    piece_type = moved_piece_type(board, move)

    # King can't really fork (checking multiple pieces is check, not fork)
    if piece_type == KING:
        return False

    # Check if destination square is safe
    if is_in_bad_spot(test_board, move.to_square):
        return False

    # Count valuable pieces attacked
    attacked = attacked_opponent_pieces(test_board, move.to_square, pov)

    valuable_attacked = 0
    moving_piece_value = get_piece_value(piece_type)

    for target_piece, target_square in attacked:
        # Skip pawns for fork counting
        if target_piece.piece_type == PAWN:
            continue

        target_value = get_piece_value(target_piece.piece_type)

        # Count if target is more valuable, or if it's hanging
        if target_value > moving_piece_value:
            valuable_attacked += 1
        elif is_hanging(test_board, target_piece, target_square):
            # Also count if the piece is hanging and not defended by attacker
            if target_square not in test_board.attackers(not pov, move.to_square):
                valuable_attacked += 1

    return valuable_attacked > 1


def detect_pin(board: Board, color: Color) -> list[tuple[Square, Square, Square]]:
    """
    Detect all pins on the board for a given color.

    A pin occurs when a piece cannot move because it would expose a more
    valuable piece (often the king) to attack.

    Returns list of (pinned_square, pinner_square, pinned_to_square) tuples.

    Adapted from lichess-puzzler/tagger/cook.py
    """
    pins = []

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if not piece or piece.color != color:
            continue

        # Check if this piece is pinned
        pin_mask = board.pin(color, square)

        # BB_ALL means not pinned
        if pin_mask == chess.BB_ALL:
            continue

        # Find the pinner and what the piece is pinned to
        # The pin mask shows the ray of the pin
        for pinner_square in chess.SQUARES:
            pinner = board.piece_at(pinner_square)
            if not pinner or pinner.color == color:
                continue
            if pinner.piece_type not in RAY_PIECE_TYPES:
                continue

            # Check if this piece is creating the pin
            if board.is_pinned(color, square):
                # Find what's behind the pinned piece
                ray = SquareSet.ray(pinner_square, square)
                for behind_square in ray:
                    if behind_square == square or behind_square == pinner_square:
                        continue
                    behind_piece = board.piece_at(behind_square)
                    if behind_piece and behind_piece.color == color:
                        pins.append((square, pinner_square, behind_square))
                        break

    return pins


def detect_skewer(
    board: Board,
    move: Move,
    pov: Color,
) -> bool:
    """
    Detect if a move creates or exploits a skewer.

    A skewer is like a reverse pin - a valuable piece is attacked and must move,
    exposing a less valuable piece behind it to capture.

    Adapted from lichess-puzzler/tagger/cook.py

    Args:
        board: Board state BEFORE the move
        move: The move to analyze
        pov: Point of view (color making the move)

    Returns:
        True if the move involves a skewer
    """
    piece_type = moved_piece_type(board, move)

    # Only ray pieces can skewer
    if piece_type not in RAY_PIECE_TYPES:
        return False

    test_board = board.copy()
    test_board.push(move)

    # Check if this is capturing after opponent moved away from skewer
    capture = board.piece_at(move.to_square)
    if capture:
        between = SquareSet.between(move.from_square, move.to_square)
        # This might be completing a skewer
        return True

    # Check if creating a new skewer
    for attacked_piece, attacked_square in attacked_opponent_pieces(test_board, move.to_square, pov):
        if attacked_piece.piece_type == KING or get_piece_value(attacked_piece.piece_type) >= get_piece_value(QUEEN):
            # High value piece attacked - check what's behind it
            ray = SquareSet.ray(move.to_square, attacked_square)
            for behind_square in ray:
                if behind_square == move.to_square or behind_square == attacked_square:
                    continue
                behind_piece = board.piece_at(behind_square)
                if behind_piece and behind_piece.color != pov:
                    # Found a piece behind - it's a skewer if behind piece is less valuable
                    if get_piece_value(behind_piece.piece_type) < get_piece_value(attacked_piece.piece_type):
                        return True
                    break

    return False


def detect_discovered_attack(
    board: Board,
    move: Move,
    pov: Color,
) -> bool:
    """
    Detect if a move creates a discovered attack.

    A discovered attack occurs when moving a piece reveals an attack by another
    piece that was behind it.

    Adapted from lichess-puzzler/tagger/cook.py

    Args:
        board: Board state BEFORE the move
        move: The move to analyze
        pov: Point of view (color making the move)

    Returns:
        True if the move creates a discovered attack
    """
    # Check if discovered check first
    if detect_discovered_check(board, move, pov):
        return True

    # Find pieces that could be unblocked
    from_square = move.from_square

    for attacker_square in chess.SQUARES:
        attacker = board.piece_at(attacker_square)
        if not attacker or attacker.color != pov:
            continue
        if attacker.piece_type not in RAY_PIECE_TYPES:
            continue
        if attacker_square == from_square:
            continue

        # Check if moving piece was blocking this attacker
        between = SquareSet.between(attacker_square, from_square)

        # Check what's on the ray beyond the moving piece
        ray = SquareSet.ray(attacker_square, from_square)

        for target_square in ray:
            if target_square == attacker_square or target_square == from_square:
                continue
            if target_square in between:
                continue

            target = board.piece_at(target_square)
            if target and target.color != pov:
                # There's an enemy piece on the ray
                # Check if moving piece was actually blocking
                if from_square in SquareSet.between(attacker_square, target_square):
                    # And make sure move doesn't block again
                    if move.to_square not in SquareSet.between(attacker_square, target_square):
                        return True
                break

    return False


def detect_discovered_check(
    board: Board,
    move: Move,
    pov: Color,
) -> bool:
    """
    Detect if a move creates a discovered check.

    A discovered check is a discovered attack where the revealed attack is on the king.

    Args:
        board: Board state BEFORE the move
        move: The move to analyze
        pov: Point of view (color making the move)

    Returns:
        True if the move creates a discovered check
    """
    test_board = board.copy()
    test_board.push(move)

    if not test_board.is_check():
        return False

    # Check if it's discovered (the moved piece is not giving check)
    moved_piece_square = move.to_square
    opponent_king_square = test_board.king(not pov)

    if opponent_king_square is None:
        return False

    # If the moved piece is not attacking the king, it's discovered
    moved_attacks = test_board.attacks(moved_piece_square)
    return opponent_king_square not in moved_attacks


def detect_double_check(board: Board, move: Move, pov: Color) -> bool:
    """
    Detect if a move creates a double check.

    A double check is when two pieces give check simultaneously, which forces
    the king to move (can't block or capture both attackers).

    Args:
        board: Board state BEFORE the move
        move: The move to analyze
        pov: Point of view (color making the move)

    Returns:
        True if the move creates double check
    """
    test_board = board.copy()
    test_board.push(move)

    if not test_board.is_check():
        return False

    opponent_king_square = test_board.king(not pov)
    if opponent_king_square is None:
        return False

    # Count attackers on the king
    attackers = test_board.attackers(pov, opponent_king_square)
    return len(attackers) >= 2


def detect_hanging_piece(board: Board, color: Color) -> list[tuple[Piece, Square]]:
    """
    Detect all hanging pieces for a color.

    A hanging piece is one that is undefended or insufficiently defended
    while being attacked.

    Args:
        board: Current board state
        color: Color whose hanging pieces to find

    Returns:
        List of (piece, square) tuples for hanging pieces
    """
    hanging = []

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if not piece or piece.color != color:
            continue

        if is_hanging(board, piece, square):
            hanging.append((piece, square))

    return hanging


def detect_back_rank_mate_threat(board: Board, color: Color) -> bool:
    """
    Detect if there's a back rank mate threat.

    A back rank mate occurs when a rook or queen checkmates a king trapped
    on the back rank by its own pawns.

    Args:
        board: Current board state
        color: Color that might deliver back rank mate

    Returns:
        True if there's a back rank vulnerability
    """
    opponent = not color
    king_square = board.king(opponent)

    if king_square is None:
        return False

    # Check if king is on back rank
    back_rank = 0 if opponent == WHITE else 7
    if chess.square_rank(king_square) != back_rank:
        return False

    # Check if king is blocked by own pieces (typically pawns)
    escape_rank = 1 if opponent == WHITE else 6
    king_file = chess.square_file(king_square)

    blocked = True
    for df in [-1, 0, 1]:
        escape_file = king_file + df
        if 0 <= escape_file <= 7:
            escape_square = chess.square(escape_file, escape_rank)
            piece = board.piece_at(escape_square)
            if not piece or piece.color != opponent:
                # Escape square is not blocked by own piece
                # But check if it's attacked
                if not board.is_attacked_by(color, escape_square):
                    blocked = False
                    break

    return blocked


def detect_sacrifice(
    board: Board,
    move: Move,
    pov: Color,
) -> bool:
    """
    Detect if a move is a sacrifice (giving up material).

    A sacrifice is intentionally giving up material for compensation,
    often tactical (checkmate, winning back more material).

    Args:
        board: Board state BEFORE the move
        move: The move to analyze
        pov: Point of view (color making the move)

    Returns:
        True if the move appears to be a sacrifice
    """
    piece = board.piece_at(move.from_square)
    if not piece:
        return False

    piece_value = get_piece_value(piece.piece_type)

    # Check if piece moves to an attacked square
    test_board = board.copy()
    test_board.push(move)

    # If our piece is now attacked by less valuable piece, it's a sacrifice
    attackers = test_board.attackers(not pov, move.to_square)
    if attackers:
        min_attacker_value = min(
            get_piece_value(test_board.piece_type_at(sq))
            for sq in attackers
        )
        if min_attacker_value < piece_value:
            return True

    # Check if we captured something less valuable while being taken
    captured = board.piece_at(move.to_square)
    if captured:
        captured_value = get_piece_value(captured.piece_type)
        if captured_value < piece_value and attackers:
            return True

    return False


# =============================================================================
# Positional theme detection
# =============================================================================

def detect_advanced_pawn(board: Board, color: Color) -> list[Square]:
    """
    Detect advanced pawns (pawns past the 4th rank).

    Advanced pawns are strategically important as they can become
    passed pawns or create promotion threats.

    Args:
        board: Current board state
        color: Color whose advanced pawns to find

    Returns:
        List of squares with advanced pawns
    """
    advanced = []
    threshold_rank = 4 if color == WHITE else 3  # 5th rank (0-indexed: 4) for white

    for square in board.pieces(PAWN, color):
        rank = chess.square_rank(square)
        if color == WHITE and rank >= threshold_rank:
            advanced.append(square)
        elif color == BLACK and rank <= threshold_rank:
            advanced.append(square)

    return advanced


def detect_exposed_king(board: Board, color: Color) -> bool:
    """
    Detect if a king is exposed (lacking pawn shelter).

    An exposed king is vulnerable to attacks, especially in the middlegame.

    Args:
        board: Current board state
        color: Color whose king to check

    Returns:
        True if the king appears exposed
    """
    king_square = board.king(color)
    if king_square is None:
        return False

    king_file = chess.square_file(king_square)
    king_rank = chess.square_rank(king_square)

    # Count friendly pawns near the king
    pawn_shield = 0
    shield_rank = king_rank + (1 if color == WHITE else -1)

    if 0 <= shield_rank <= 7:
        for df in [-1, 0, 1]:
            shield_file = king_file + df
            if 0 <= shield_file <= 7:
                shield_square = chess.square(shield_file, shield_rank)
                piece = board.piece_at(shield_square)
                if piece and piece.piece_type == PAWN and piece.color == color:
                    pawn_shield += 1

    # King is exposed if fewer than 2 pawns shield it
    return pawn_shield < 2


def detect_endgame_type(board: Board) -> Optional[PositionalTheme]:
    """
    Detect what type of endgame the position is.

    Args:
        board: Current board state

    Returns:
        PositionalTheme for the endgame type, or None if not an endgame
    """
    # Count material
    white_pieces = {pt: len(board.pieces(pt, WHITE)) for pt in range(1, 7)}
    black_pieces = {pt: len(board.pieces(pt, BLACK)) for pt in range(1, 7)}

    total_white = sum(v for k, v in white_pieces.items() if k != KING)
    total_black = sum(v for k, v in black_pieces.items() if k != KING)

    # Not an endgame if too much material
    if total_white > 5 or total_black > 5:
        return None

    # Pure pawn endgame
    white_non_pawns = sum(v for k, v in white_pieces.items() if k not in [PAWN, KING])
    black_non_pawns = sum(v for k, v in black_pieces.items() if k not in [PAWN, KING])

    if white_non_pawns == 0 and black_non_pawns == 0:
        return PositionalTheme.PAWN_ENDGAME

    # Single piece endgames
    if white_non_pawns <= 1 and black_non_pawns <= 1:
        for pt, theme in [
            (QUEEN, PositionalTheme.QUEEN_ENDGAME),
            (ROOK, PositionalTheme.ROOK_ENDGAME),
            (BISHOP, PositionalTheme.BISHOP_ENDGAME),
            (KNIGHT, PositionalTheme.KNIGHT_ENDGAME),
        ]:
            if white_pieces.get(pt, 0) > 0 or black_pieces.get(pt, 0) > 0:
                return theme

    return None


# =============================================================================
# High-level analysis functions
# =============================================================================

def analyze_position_themes(
    board: Board,
    pov: Color,
) -> ThemeDetectionResult:
    """
    Analyze a position for all detectable themes.

    Args:
        board: Current board state
        pov: Point of view (which color we're analyzing for)

    Returns:
        ThemeDetectionResult with detected themes and details
    """
    themes = []
    details = {}

    # Positional themes
    advanced_pawns = detect_advanced_pawn(board, pov)
    if advanced_pawns:
        themes.append(PositionalTheme.ADVANCED_PAWN.value)
        details['advanced_pawns'] = [chess.square_name(sq) for sq in advanced_pawns]

    if detect_exposed_king(board, not pov):
        themes.append(PositionalTheme.EXPOSED_KING.value)

    endgame_type = detect_endgame_type(board)
    if endgame_type:
        themes.append(endgame_type.value)

    # Tactical themes - check for immediate threats
    hanging = detect_hanging_piece(board, not pov)
    if hanging:
        themes.append(TacticalTheme.HANGING_PIECE.value)
        details['hanging_pieces'] = [
            f"{piece.symbol()} on {chess.square_name(sq)}"
            for piece, sq in hanging
        ]

    pins = detect_pin(board, not pov)
    if pins:
        themes.append(TacticalTheme.PIN.value)
        details['pins'] = [
            f"{chess.square_name(pinned)} pinned by {chess.square_name(pinner)}"
            for pinned, pinner, _ in pins
        ]

    if detect_back_rank_mate_threat(board, pov):
        themes.append(TacticalTheme.BACK_RANK_MATE.value)

    return ThemeDetectionResult(themes=themes, details=details)


def analyze_move_themes(
    board: Board,
    move: Move,
    pov: Color,
) -> ThemeDetectionResult:
    """
    Analyze a move for tactical themes it creates.

    Args:
        board: Board state BEFORE the move
        move: The move to analyze
        pov: Point of view (color making the move)

    Returns:
        ThemeDetectionResult with detected themes and details
    """
    themes = []
    details = {}

    # Check various tactical themes
    if detect_fork(board, move, pov):
        themes.append(TacticalTheme.FORK.value)

    if detect_skewer(board, move, pov):
        themes.append(TacticalTheme.SKEWER.value)

    if detect_double_check(board, move, pov):
        themes.append(TacticalTheme.DOUBLE_CHECK.value)
    elif detect_discovered_check(board, move, pov):
        themes.append(TacticalTheme.DISCOVERED_CHECK.value)
    elif detect_discovered_attack(board, move, pov):
        themes.append(TacticalTheme.DISCOVERED_ATTACK.value)

    if detect_sacrifice(board, move, pov):
        themes.append(TacticalTheme.SACRIFICE.value)

    # Special moves
    if board.is_en_passant(move):
        themes.append(TacticalTheme.EN_PASSANT.value)

    if move.promotion:
        if move.promotion != QUEEN:
            themes.append(TacticalTheme.UNDERPROMOTION.value)
        else:
            themes.append(TacticalTheme.PROMOTION.value)

    if board.is_castling(move):
        themes.append(TacticalTheme.CASTLING.value)

    details['move'] = board.san(move)
    details['from'] = chess.square_name(move.from_square)
    details['to'] = chess.square_name(move.to_square)

    return ThemeDetectionResult(themes=themes, details=details)


def get_all_theme_names() -> dict:
    """
    Get all available theme names grouped by category.

    Returns:
        Dictionary with 'tactical' and 'positional' theme lists
    """
    return {
        'tactical': [t.value for t in TacticalTheme],
        'positional': [t.value for t in PositionalTheme],
    }
