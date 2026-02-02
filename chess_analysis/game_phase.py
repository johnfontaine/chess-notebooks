"""
Game phase detection for chess positions.

Classifies positions into Opening, Middlegame, or Endgame phases based on
material count, piece development, and position "mixedness".

Attribution:
-----------
The algorithm in this module is adapted from the Lichess scalachess library:
    https://github.com/lichess-org/scalachess
    License: MIT

Original implementation:
    https://github.com/lichess-org/scalachess/blob/master/core/src/main/scala/Divider.scala

The Divider algorithm determines game phases based on:
1. Major/minor piece count (excluding kings and pawns)
2. Back rank piece count (development indicator)
3. Position "mixedness" score (how interlocked the pieces are)

References:
----------
- Lichess forum discussion on phase detection:
  https://lichess.org/forum/lichess-feedback/query-what-formula-used-in-lichess-determines-the-move-from-which-middle-game-and-end-game-start
- Chess Stack Exchange on game phase calculation:
  https://chess.stackexchange.com/questions/19317/calculation-for-game-phase
"""

import chess
from chess import Board, Color, WHITE, BLACK, SquareSet
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class GamePhase(Enum):
    """The three phases of a chess game."""
    OPENING = "opening"
    MIDDLEGAME = "middlegame"
    ENDGAME = "endgame"


@dataclass
class GamePhaseInfo:
    """Detailed information about game phase detection."""
    phase: GamePhase
    major_minor_count: int  # Count of major and minor pieces (no kings/pawns)
    back_rank_white: int    # White pieces on rank 1
    back_rank_black: int    # Black pieces on rank 8
    mixedness_score: int    # How interlocked the pieces are
    is_middlegame_by_material: bool
    is_middlegame_by_development: bool
    is_middlegame_by_mixedness: bool
    is_endgame: bool


# =============================================================================
# Constants - based on Lichess scalachess Divider.scala
# =============================================================================

# Middlegame starts when major+minor pieces <= this threshold
MIDDLEGAME_MATERIAL_THRESHOLD = 10

# Endgame starts when major+minor pieces <= this threshold
ENDGAME_MATERIAL_THRESHOLD = 6

# Back rank threshold for development (fewer than this = developed = middlegame)
# Starting position has 8 pieces on back rank, so <6 means significant development
BACK_RANK_DEVELOPMENT_THRESHOLD = 6

# Mixedness score threshold for middlegame
MIXEDNESS_THRESHOLD = 150


# =============================================================================
# Mixedness scoring - how interlocked are the pieces?
# =============================================================================

def _score_quadrant(white_count: int, black_count: int, rank_factor: int) -> int:
    """
    Score a 2x2 quadrant based on piece distribution.

    Higher scores indicate more mixing of white and black pieces.
    Only scores when BOTH colors have pieces in the quadrant (actual mixing).
    The rank_factor increases score for advanced pieces.

    Adapted from scalachess Divider.scala score() function.

    Args:
        white_count: Number of white pieces in quadrant (0-4)
        black_count: Number of black pieces in quadrant (0-4)
        rank_factor: Multiplier based on rank (higher for middle ranks)

    Returns:
        Score for this quadrant
    """
    # Only score when both colors have pieces (actual mixing)
    # Single-color quadrants don't indicate mixedness
    if white_count == 0 or black_count == 0:
        return 0

    # Score based on how many pieces of each color are mixed
    # More of each = more interesting/complex
    base_score = white_count * black_count * 4
    return base_score * rank_factor


def calculate_mixedness(board: Board) -> int:
    """
    Calculate how mixed/interlocked the pieces are on the board.

    This measures how much the two sides' pieces are intermingled,
    which indicates whether the game has entered the middlegame phase.

    Adapted from scalachess Divider.scala mixedness() function.

    Args:
        board: Current board position

    Returns:
        Mixedness score (higher = more mixed, threshold is ~150 for middlegame)
    """
    total_score = 0

    # Scan 2x2 quadrants across the board (7x7 = 49 quadrants)
    for start_file in range(7):
        for start_rank in range(7):
            white_count = 0
            black_count = 0

            # Count pieces in 2x2 quadrant
            for df in range(2):
                for dr in range(2):
                    sq = chess.square(start_file + df, start_rank + dr)
                    piece = board.piece_at(sq)
                    if piece:
                        if piece.color == WHITE:
                            white_count += 1
                        else:
                            black_count += 1

            # Rank factor: middle ranks get higher weight
            # Ranks 2-5 (indices 1-4) are most important for mixedness
            avg_rank = start_rank + 0.5
            if 2 <= avg_rank <= 5:
                rank_factor = 3
            elif 1 <= avg_rank <= 6:
                rank_factor = 2
            else:
                rank_factor = 1

            total_score += _score_quadrant(white_count, black_count, rank_factor)

    return total_score


# =============================================================================
# Material counting
# =============================================================================

def count_major_minor_pieces(board: Board) -> int:
    """
    Count major and minor pieces on the board (excluding kings and pawns).

    Major pieces: Queens, Rooks
    Minor pieces: Bishops, Knights

    This is the primary material indicator for phase detection.

    Args:
        board: Current board position

    Returns:
        Total count of major and minor pieces (both colors)
    """
    # All occupied squares minus kings and pawns
    major_minor = board.occupied & ~(board.kings | board.pawns)
    return chess.popcount(major_minor)


def count_back_rank_pieces(board: Board, color: Color) -> int:
    """
    Count pieces on the back rank for a given color.

    Back rank = rank 1 for white, rank 8 for black.
    Fewer pieces on back rank indicates better development.

    Args:
        board: Current board position
        color: Which color's back rank to check

    Returns:
        Number of pieces (of that color) on their back rank
    """
    if color == WHITE:
        back_rank = chess.BB_RANK_1
    else:
        back_rank = chess.BB_RANK_8

    color_pieces = board.occupied_co[color]
    return chess.popcount(color_pieces & back_rank)


def calculate_material_value(board: Board, color: Color) -> int:
    """
    Calculate total material value for a color.

    Uses standard piece values:
    - Pawn: 1
    - Knight: 3
    - Bishop: 3
    - Rook: 5
    - Queen: 9

    Args:
        board: Current board position
        color: Which color to calculate for

    Returns:
        Total material value in pawns
    """
    values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
    }

    total = 0
    for piece_type, value in values.items():
        count = len(board.pieces(piece_type, color))
        total += count * value

    return total


# =============================================================================
# Phase detection
# =============================================================================

def detect_game_phase(board: Board) -> GamePhase:
    """
    Detect the game phase (opening, middlegame, endgame) for a position.

    Algorithm based on Lichess scalachess Divider.scala:
    - Endgame: ≤6 major/minor pieces
    - Middlegame: ≤10 major/minor pieces OR poor development OR high mixedness
    - Opening: Otherwise

    Args:
        board: Current board position

    Returns:
        GamePhase enum value
    """
    info = detect_game_phase_detailed(board)
    return info.phase


def detect_game_phase_detailed(board: Board) -> GamePhaseInfo:
    """
    Detect game phase with detailed reasoning.

    Provides full information about why a position is classified
    as opening, middlegame, or endgame.

    Args:
        board: Current board position

    Returns:
        GamePhaseInfo with phase and all detection metrics
    """
    # Count material
    major_minor = count_major_minor_pieces(board)

    # Count back rank pieces (development indicator)
    back_rank_white = count_back_rank_pieces(board, WHITE)
    back_rank_black = count_back_rank_pieces(board, BLACK)

    # Calculate mixedness
    mixedness = calculate_mixedness(board)

    # Check endgame condition first
    is_endgame = major_minor <= ENDGAME_MATERIAL_THRESHOLD

    # Check middlegame conditions
    is_middlegame_by_material = major_minor <= MIDDLEGAME_MATERIAL_THRESHOLD
    is_middlegame_by_development = (
        back_rank_white < BACK_RANK_DEVELOPMENT_THRESHOLD or
        back_rank_black < BACK_RANK_DEVELOPMENT_THRESHOLD
    )
    is_middlegame_by_mixedness = mixedness >= MIXEDNESS_THRESHOLD

    # Determine phase
    if is_endgame:
        phase = GamePhase.ENDGAME
    elif is_middlegame_by_material or is_middlegame_by_development or is_middlegame_by_mixedness:
        phase = GamePhase.MIDDLEGAME
    else:
        phase = GamePhase.OPENING

    return GamePhaseInfo(
        phase=phase,
        major_minor_count=major_minor,
        back_rank_white=back_rank_white,
        back_rank_black=back_rank_black,
        mixedness_score=mixedness,
        is_middlegame_by_material=is_middlegame_by_material,
        is_middlegame_by_development=is_middlegame_by_development,
        is_middlegame_by_mixedness=is_middlegame_by_mixedness,
        is_endgame=is_endgame,
    )


def get_phase_transitions(boards: list[Board]) -> dict:
    """
    Find where phase transitions occur in a game.

    Analyzes a sequence of positions to identify when the game
    transitions from opening to middlegame and middlegame to endgame.

    Args:
        boards: List of Board positions in game order

    Returns:
        Dictionary with:
        - 'middlegame_start': Ply where middlegame starts (or None)
        - 'endgame_start': Ply where endgame starts (or None)
        - 'phases': List of phase for each position
    """
    if not boards:
        return {
            'middlegame_start': None,
            'endgame_start': None,
            'phases': [],
        }

    phases = []
    middlegame_start = None
    endgame_start = None

    prev_phase = None

    for i, board in enumerate(boards):
        phase = detect_game_phase(board)
        phases.append(phase.value)

        # Detect transitions
        if prev_phase == GamePhase.OPENING and phase == GamePhase.MIDDLEGAME:
            if middlegame_start is None:
                middlegame_start = i

        if prev_phase in (GamePhase.OPENING, GamePhase.MIDDLEGAME) and phase == GamePhase.ENDGAME:
            if endgame_start is None:
                endgame_start = i

        prev_phase = phase

    return {
        'middlegame_start': middlegame_start,
        'endgame_start': endgame_start,
        'phases': phases,
    }


def analyze_game_phases(game: "chess.pgn.Game") -> dict:
    """
    Analyze phase transitions throughout a game.

    Args:
        game: A parsed PGN game

    Returns:
        Dictionary with:
        - 'middlegame_start_ply': Ply where middlegame starts
        - 'middlegame_start_move': Move number where middlegame starts
        - 'endgame_start_ply': Ply where endgame starts
        - 'endgame_start_move': Move number where endgame starts
        - 'opening_length': Number of moves in opening
        - 'middlegame_length': Number of moves in middlegame
        - 'endgame_length': Number of moves in endgame
        - 'total_moves': Total game length in moves
        - 'phases_by_ply': Phase at each ply
    """
    board = game.board()
    boards = [board.copy()]

    for move in game.mainline_moves():
        board.push(move)
        boards.append(board.copy())

    transitions = get_phase_transitions(boards)

    total_plies = len(boards) - 1  # Exclude starting position
    total_moves = (total_plies + 1) // 2

    middlegame_ply = transitions['middlegame_start']
    endgame_ply = transitions['endgame_start']

    # Calculate move numbers (1-indexed, from white's perspective)
    middlegame_move = (middlegame_ply + 2) // 2 if middlegame_ply is not None else None
    endgame_move = (endgame_ply + 2) // 2 if endgame_ply is not None else None

    # Calculate phase lengths
    if middlegame_ply is None:
        # Never left opening
        opening_length = total_moves
        middlegame_length = 0
        endgame_length = 0
    elif endgame_ply is None:
        # Never reached endgame
        opening_length = middlegame_move - 1 if middlegame_move else total_moves
        middlegame_length = total_moves - opening_length
        endgame_length = 0
    else:
        opening_length = middlegame_move - 1 if middlegame_move else 0
        endgame_start = endgame_move if endgame_move else total_moves
        middlegame_length = endgame_start - opening_length - 1
        endgame_length = total_moves - opening_length - middlegame_length

    return {
        'middlegame_start_ply': middlegame_ply,
        'middlegame_start_move': middlegame_move,
        'endgame_start_ply': endgame_ply,
        'endgame_start_move': endgame_move,
        'opening_length': max(0, opening_length),
        'middlegame_length': max(0, middlegame_length),
        'endgame_length': max(0, endgame_length),
        'total_moves': total_moves,
        'phases_by_ply': transitions['phases'],
    }


# =============================================================================
# Alternative/simplified detection methods
# =============================================================================

def detect_phase_by_material_only(board: Board) -> GamePhase:
    """
    Simple phase detection based purely on material count.

    This is a simplified heuristic that doesn't consider development
    or piece placement, just raw material.

    Thresholds:
    - Opening: All 16 major/minor pieces present (both sides combined)
    - Middlegame: 7-15 major/minor pieces
    - Endgame: ≤6 major/minor pieces

    Args:
        board: Current board position

    Returns:
        GamePhase enum value
    """
    count = count_major_minor_pieces(board)

    if count <= 6:
        return GamePhase.ENDGAME
    elif count <= 14:  # Some pieces traded
        return GamePhase.MIDDLEGAME
    else:
        return GamePhase.OPENING


def detect_phase_by_material_value(board: Board) -> GamePhase:
    """
    Phase detection based on total material value remaining.

    Uses standard piece values to determine phase based on
    how much material has been traded.

    Starting material per side: 39 points (1Q + 2R + 2B + 2N + 8P = 9+10+6+6+8)
    Total starting material: 78 points

    Thresholds:
    - Opening: >70 points total (little trading)
    - Middlegame: 30-70 points
    - Endgame: <30 points

    Args:
        board: Current board position

    Returns:
        GamePhase enum value
    """
    white_material = calculate_material_value(board, WHITE)
    black_material = calculate_material_value(board, BLACK)
    total = white_material + black_material

    if total >= 70:
        return GamePhase.OPENING
    elif total >= 30:
        return GamePhase.MIDDLEGAME
    else:
        return GamePhase.ENDGAME


def is_endgame(board: Board) -> bool:
    """
    Quick check if position is an endgame.

    Uses the Lichess threshold of ≤6 major/minor pieces.

    Args:
        board: Current board position

    Returns:
        True if position is an endgame
    """
    return count_major_minor_pieces(board) <= ENDGAME_MATERIAL_THRESHOLD


def is_opening(board: Board) -> bool:
    """
    Quick check if position is still in the opening.

    Uses Lichess criteria - not middlegame by any metric.

    Args:
        board: Current board position

    Returns:
        True if position is in the opening
    """
    return detect_game_phase(board) == GamePhase.OPENING
