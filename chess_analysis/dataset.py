"""
Dataset building for large-scale chess game analysis.

Provides functions to extract cheap-to-compute features from games
without requiring engine analysis. Used to build a dataset for
prioritizing which games deserve expensive deep analysis.
"""

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional
from collections import defaultdict

import chess
import chess.pgn

from .game_phase import (
    detect_game_phase,
    detect_game_phase_detailed,
    calculate_mixedness,
    count_major_minor_pieces,
    GamePhase,
)
from .fragility import calculate_fragility_simple
from .tablebase import is_tablebase_position
from .games import parse_pgn_string


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class GameMetadata:
    """Metadata extracted from a chess.com game."""

    game_id: str
    url: str

    # Players
    white: str
    black: str
    player_color: str  # 'white' or 'black'
    player_username: str
    opponent_username: str

    # Ratings
    player_elo: int
    opponent_elo: int
    elo_diff: int  # player_elo - opponent_elo

    # Time control
    time_control: str
    time_class: str  # 'rapid', 'blitz', 'bullet', 'daily'
    base_time: int  # seconds
    increment: int  # seconds

    # Game result
    result: str  # '1-0', '0-1', '1/2-1/2'
    player_result: str  # 'win', 'loss', 'draw'
    termination: str  # 'checkmate', 'resignation', 'timeout', 'stalemate', etc.
    resolution: str  # 'win_checkmate', 'loss_resignation', etc.

    # Game info
    date: str
    end_time: int  # unix timestamp
    num_moves: int  # total plies / 2
    num_plies: int  # total half-moves

    # Opening
    eco: Optional[str]
    opening_name: Optional[str]


@dataclass
class PositionFeatures:
    """Cheap-to-compute features for a single position."""

    game_id: str
    ply: int
    move_number: int

    # FEN (without move counters for deduplication)
    fen: str
    fen_board_only: str  # Just piece placement for opening book

    # Move info
    move_uci: Optional[str]
    move_san: Optional[str]
    is_capture: bool
    is_check: bool
    is_castling: bool
    is_promotion: bool

    # Material (raw piece values: P=1, N/B=3, R=5, Q=9)
    material_white: int
    material_black: int
    material_player: int
    material_opponent: int
    material_balance: int  # player - opponent

    # Piece counts
    total_pieces: int
    is_tablebase: bool

    # Game phase
    phase: str  # 'opening', 'middlegame', 'endgame'
    mixedness: int
    major_minor_count: int

    # Position features
    legal_moves_count: int
    fragility: float

    # Side to move
    side_to_move: str  # 'white' or 'black'
    is_player_move: bool


@dataclass
class GameFeatures:
    """
    Comprehensive per-game features for baseline analysis.

    Combines time control, game phase, fragility, and material metrics
    extracted without engine analysis.
    """

    game_id: str

    # Time control analysis
    time_class: str  # bullet/blitz/rapid
    base_time: int  # seconds
    increment: int  # seconds
    avg_time_per_move: Optional[float]  # None if no clock data
    time_cv: Optional[float]  # coefficient of variation
    fast_move_pct: Optional[float]  # % moves under 2s

    # Game phase duration (in moves, not plies)
    total_moves: int
    opening_moves: int
    middlegame_moves: int
    endgame_moves: int
    reached_endgame: bool

    # Fragility metrics
    fragility_mean: float
    fragility_max: float
    fragility_std: float
    high_fragility_positions: int  # positions above threshold (0.5)
    fragility_category: str  # "low", "medium", "high", "extreme"

    # Material balance (in centipawns, P=100, N/B=300, R=500, Q=900)
    material_swings: int  # number of material changes > 100cp
    max_material_advantage: int  # max positive balance (cp)
    max_material_disadvantage: int  # max negative balance (cp)
    avg_material_balance: float
    material_trajectory: str  # "gained", "lost", "stable", "volatile"


@dataclass
class EloRangeStats:
    """Statistics for games within a 200-Elo range bucket (e.g., 1000-1200)."""

    segment_name: str  # e.g., "1000-1200"
    elo_min: int
    elo_max: int

    # Game counts
    total_games: int
    trustworthy_games: int
    untrustworthy_games: int

    # Results
    wins: int
    losses: int
    draws: int
    win_rate: float

    # Average metrics
    avg_opponent_elo: float
    avg_game_length: float


# =============================================================================
# Game Metadata Extraction
# =============================================================================

def parse_time_control(time_control: str) -> tuple[int, int]:
    """
    Parse time control string into base time and increment.

    Args:
        time_control: String like "600", "180+2", "1/86400"

    Returns:
        Tuple of (base_seconds, increment_seconds)
    """
    if not time_control:
        return 0, 0

    # Daily format: "1/86400" means 1 day per move
    if "/" in time_control:
        parts = time_control.split("/")
        try:
            return int(parts[1]), 0
        except (ValueError, IndexError):
            return 0, 0

    # Standard format: "600" or "180+2"
    if "+" in time_control:
        parts = time_control.split("+")
        try:
            return int(parts[0]), int(parts[1])
        except (ValueError, IndexError):
            return 0, 0

    try:
        return int(time_control), 0
    except ValueError:
        return 0, 0


def determine_resolution(
    player_result: str,
    termination: str,
    player_color: str,
) -> str:
    """
    Determine the game resolution category.

    Args:
        player_result: 'win', 'loss', or 'draw'
        termination: Raw termination reason from chess.com
        player_color: 'white' or 'black'

    Returns:
        Resolution string like 'win_checkmate', 'loss_timeout', etc.
    """
    term_lower = termination.lower() if termination else ""

    if player_result == "draw":
        if "stalemate" in term_lower:
            return "draw_stalemate"
        elif "repetition" in term_lower:
            return "draw_repetition"
        elif "insufficient" in term_lower:
            return "draw_insufficient"
        elif "50" in term_lower or "fifty" in term_lower:
            return "draw_50move"
        elif "agreed" in term_lower or "agreement" in term_lower:
            return "draw_agreement"
        else:
            return "draw_other"

    # Win or loss
    prefix = player_result  # 'win' or 'loss'

    if "checkmate" in term_lower or "mate" in term_lower:
        return f"{prefix}_checkmate"
    elif "resign" in term_lower:
        return f"{prefix}_resignation"
    elif "time" in term_lower or "timeout" in term_lower:
        return f"{prefix}_timeout"
    elif "abandon" in term_lower:
        return f"{prefix}_abandonment"
    else:
        return f"{prefix}_other"


def extract_game_metadata(
    game_data: dict,
    target_username: str,
) -> Optional[GameMetadata]:
    """
    Extract metadata from a chess.com game API response.

    Args:
        game_data: Raw game data from chess.com API
        target_username: Username of the player being analyzed

    Returns:
        GameMetadata object or None if parsing fails
    """
    try:
        # Parse the PGN
        pgn = game_data.get("pgn", "")
        if not pgn:
            return None

        games = list(parse_pgn_string(pgn))
        if not games:
            return None

        game = games[0]

        # Extract player info
        white = game.headers.get("White", "")
        black = game.headers.get("Black", "")

        target_lower = target_username.lower()
        if white.lower() == target_lower:
            player_color = "white"
            player_username = white
            opponent_username = black
        elif black.lower() == target_lower:
            player_color = "black"
            player_username = black
            opponent_username = white
        else:
            return None

        # Get Elo ratings
        try:
            white_elo = int(game.headers.get("WhiteElo", "0"))
            black_elo = int(game.headers.get("BlackElo", "0"))
        except ValueError:
            white_elo = 0
            black_elo = 0

        if player_color == "white":
            player_elo = white_elo
            opponent_elo = black_elo
        else:
            player_elo = black_elo
            opponent_elo = white_elo

        # Parse time control
        time_control = game_data.get("time_control", "")
        time_class = game_data.get("time_class", "unknown")
        base_time, increment = parse_time_control(time_control)

        # Determine result
        result = game.headers.get("Result", "*")
        if result == "1-0":
            player_result = "win" if player_color == "white" else "loss"
        elif result == "0-1":
            player_result = "win" if player_color == "black" else "loss"
        elif result == "1/2-1/2":
            player_result = "draw"
        else:
            player_result = "unknown"

        # Get termination reason
        termination = game.headers.get("Termination", "")
        resolution = determine_resolution(player_result, termination, player_color)

        # Count moves
        num_plies = len(list(game.mainline_moves()))
        num_moves = (num_plies + 1) // 2

        # Extract opening info
        eco = game.headers.get("ECO")
        opening_name = game.headers.get("ECOUrl", "")
        if opening_name:
            # Extract opening name from URL like "/openings/Kings-Pawn-Opening"
            opening_name = opening_name.split("/")[-1].replace("-", " ")
        else:
            opening_name = game.headers.get("Opening")

        # Build game ID from URL
        url = game_data.get("url", "")
        game_id = url.split("/")[-1] if url else str(game_data.get("end_time", ""))

        return GameMetadata(
            game_id=game_id,
            url=url,
            white=white,
            black=black,
            player_color=player_color,
            player_username=player_username,
            opponent_username=opponent_username,
            player_elo=player_elo,
            opponent_elo=opponent_elo,
            elo_diff=player_elo - opponent_elo,
            time_control=time_control,
            time_class=time_class,
            base_time=base_time,
            increment=increment,
            result=result,
            player_result=player_result,
            termination=termination,
            resolution=resolution,
            date=game.headers.get("Date", ""),
            end_time=game_data.get("end_time", 0),
            num_moves=num_moves,
            num_plies=num_plies,
            eco=eco,
            opening_name=opening_name,
        )

    except Exception:
        return None


# =============================================================================
# Material Calculation
# =============================================================================

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,  # Don't count kings
}


def calculate_material(board: chess.Board, color: chess.Color) -> int:
    """
    Calculate total material value for a color.

    Uses standard piece values: P=1, N/B=3, R=5, Q=9

    Args:
        board: Current position
        color: Which color to calculate for

    Returns:
        Total material value
    """
    total = 0
    for piece_type, value in PIECE_VALUES.items():
        count = len(board.pieces(piece_type, color))
        total += count * value
    return total


def get_fen_board_only(fen: str) -> str:
    """
    Extract just the piece placement from a FEN string.

    Used for opening book deduplication where we don't care
    about castling rights, en passant, or move counters.

    Args:
        fen: Full FEN string

    Returns:
        Just the piece placement and side to move
    """
    parts = fen.split()
    if len(parts) >= 2:
        return f"{parts[0]} {parts[1]}"
    return parts[0] if parts else fen


# =============================================================================
# Position Feature Extraction
# =============================================================================

def extract_position_features(
    board: chess.Board,
    move: Optional[chess.Move],
    game_id: str,
    ply: int,
    player_color: str,
) -> PositionFeatures:
    """
    Extract cheap-to-compute features from a position.

    Args:
        board: Position AFTER the move was made
        move: The move that led to this position (None for starting position)
        game_id: Identifier for the game
        ply: Half-move number (0 = starting position)
        player_color: 'white' or 'black'

    Returns:
        PositionFeatures with all computed features
    """
    fen = board.fen()
    fen_board_only = get_fen_board_only(fen)

    # Move info
    if move:
        move_uci = move.uci()
        # Need to get SAN from position before move
        move_san = None  # Would need board before move
        is_capture = board.is_capture(move) if board.move_stack else False
        # For captures, check the previous position
        is_castling = board.is_castling(move) if hasattr(board, 'is_castling') else False
        is_promotion = move.promotion is not None
    else:
        move_uci = None
        move_san = None
        is_capture = False
        is_castling = False
        is_promotion = False

    # Check if position is check (for the side to move)
    is_check = board.is_check()

    # Material calculation
    material_white = calculate_material(board, chess.WHITE)
    material_black = calculate_material(board, chess.BLACK)

    is_white_player = player_color == "white"
    if is_white_player:
        material_player = material_white
        material_opponent = material_black
    else:
        material_player = material_black
        material_opponent = material_white

    # Piece counts
    total_pieces = chess.popcount(board.occupied)
    is_tablebase = is_tablebase_position(board)

    # Game phase
    phase_info = detect_game_phase_detailed(board)
    phase = phase_info.phase.value
    mixedness = phase_info.mixedness_score
    major_minor = count_major_minor_pieces(board)

    # Legal moves
    legal_moves_count = board.legal_moves.count()

    # Fragility (this is cheap - no engine needed)
    fragility = calculate_fragility_simple(board)

    # Side to move
    side_to_move = "white" if board.turn == chess.WHITE else "black"
    is_player_move = (ply % 2 == 1 and is_white_player) or (ply % 2 == 0 and not is_white_player)
    # Actually: ply 1 is white's first move result, ply 2 is black's first move result
    # After white moves (ply 1, 3, 5...), it's black's turn
    # So if ply is odd and board.turn is BLACK, white just moved
    if ply > 0:
        # The move at this ply was made by the opposite of whose turn it is now
        just_moved = "black" if board.turn == chess.WHITE else "white"
        is_player_move = just_moved == player_color
    else:
        is_player_move = False  # Starting position, no move made

    return PositionFeatures(
        game_id=game_id,
        ply=ply,
        move_number=(ply + 1) // 2,
        fen=fen,
        fen_board_only=fen_board_only,
        move_uci=move_uci,
        move_san=move_san,
        is_capture=is_capture,
        is_check=is_check,
        is_castling=is_castling,
        is_promotion=is_promotion,
        material_white=material_white,
        material_black=material_black,
        material_player=material_player,
        material_opponent=material_opponent,
        material_balance=material_player - material_opponent,
        total_pieces=total_pieces,
        is_tablebase=is_tablebase,
        phase=phase,
        mixedness=mixedness,
        major_minor_count=major_minor,
        legal_moves_count=legal_moves_count,
        fragility=fragility,
        side_to_move=side_to_move,
        is_player_move=is_player_move,
    )


def extract_game_positions(
    game: chess.pgn.Game,
    game_id: str,
    player_color: str,
) -> list[PositionFeatures]:
    """
    Extract position features for all positions in a game.

    Args:
        game: Parsed PGN game
        game_id: Identifier for the game
        player_color: 'white' or 'black'

    Returns:
        List of PositionFeatures for each position
    """
    positions = []
    board = game.board()

    # Starting position
    positions.append(extract_position_features(
        board=board,
        move=None,
        game_id=game_id,
        ply=0,
        player_color=player_color,
    ))

    # Process each move
    for ply, move in enumerate(game.mainline_moves(), start=1):
        board.push(move)
        positions.append(extract_position_features(
            board=board,
            move=move,
            game_id=game_id,
            ply=ply,
            player_color=player_color,
        ))

    return positions


# =============================================================================
# Dataset Building
# =============================================================================

def build_game_dataset(
    games: list[dict],
    target_username: str,
    time_classes: Optional[list[str]] = None,
    rated_only: bool = True,
    opponent_profiles: Optional[dict] = None,
) -> list[dict]:
    """
    Build a dataset of game-level metadata.

    Args:
        games: List of game data from chess.com API
        target_username: Username of player being analyzed
        time_classes: Filter to specific time classes (e.g., ['rapid', 'blitz'])
        rated_only: If True, only include rated games (default True)
        opponent_profiles: Optional dict mapping opponent username (lowercase) to
            profile object with is_banned attribute. If provided, adds
            opponent_is_banned flag to each game.

    Returns:
        List of game metadata dictionaries
    """
    dataset = []

    for game_data in games:
        # Filter by rated status
        if rated_only and not game_data.get("rated", False):
            continue

        # Filter by time class if specified
        if time_classes:
            time_class = game_data.get("time_class", "")
            if time_class not in time_classes:
                continue

        metadata = extract_game_metadata(game_data, target_username)
        if metadata:
            game_dict = asdict(metadata)

            # Add opponent ban status if profiles provided
            if opponent_profiles is not None:
                opponent = metadata.opponent_username.lower()
                profile = opponent_profiles.get(opponent)
                game_dict["opponent_is_banned"] = profile.is_banned if profile else False

            dataset.append(game_dict)

    return dataset


def build_position_dataset(
    games: list[dict],
    target_username: str,
    time_classes: Optional[list[str]] = None,
    max_games: Optional[int] = None,
    rated_only: bool = True,
) -> tuple[list[dict], list[dict]]:
    """
    Build datasets of game and position features.

    Args:
        games: List of game data from chess.com API
        target_username: Username of player being analyzed
        time_classes: Filter to specific time classes
        max_games: Maximum number of games to process
        rated_only: If True, only include rated games (default True)

    Returns:
        Tuple of (games_dataset, positions_dataset) as lists of dicts
    """
    games_dataset = []
    positions_dataset = []

    processed = 0

    for game_data in games:
        if max_games and processed >= max_games:
            break

        # Filter by rated status
        if rated_only and not game_data.get("rated", False):
            continue

        # Filter by time class if specified
        if time_classes:
            time_class = game_data.get("time_class", "")
            if time_class not in time_classes:
                continue

        # Extract game metadata
        metadata = extract_game_metadata(game_data, target_username)
        if not metadata:
            continue

        games_dataset.append(asdict(metadata))

        # Parse game and extract positions
        pgn = game_data.get("pgn", "")
        parsed_games = list(parse_pgn_string(pgn))
        if not parsed_games:
            continue

        game = parsed_games[0]
        positions = extract_game_positions(
            game=game,
            game_id=metadata.game_id,
            player_color=metadata.player_color,
        )

        for pos in positions:
            positions_dataset.append(asdict(pos))

        processed += 1

    return games_dataset, positions_dataset


# =============================================================================
# Opening Book Generation
# =============================================================================

def build_opening_book(
    positions: list[dict],
    min_occurrences: int = 2,
    max_ply: int = 30,
) -> dict:
    """
    Build a player's personal opening book from position frequencies.

    Args:
        positions: List of position feature dictionaries
        min_occurrences: Minimum times a position must appear
        max_ply: Maximum ply to consider (opening phase only)

    Returns:
        Dictionary with:
        - 'positions': {fen_board_only: count}
        - 'common_positions': Positions appearing >= min_occurrences
        - 'opening_depth_avg': Average ply where player leaves common positions
    """
    position_counts = defaultdict(int)
    position_games = defaultdict(set)  # Track which games reached each position

    for pos in positions:
        if pos["ply"] > max_ply:
            continue

        fen_board = pos["fen_board_only"]
        position_counts[fen_board] += 1
        position_games[fen_board].add(pos["game_id"])

    # Find common positions (player's repertoire)
    common_positions = {
        fen: count
        for fen, count in position_counts.items()
        if count >= min_occurrences
    }

    # Calculate average opening depth (where player leaves their repertoire)
    opening_depths = []
    current_game = None
    last_common_ply = 0

    for pos in sorted(positions, key=lambda p: (p["game_id"], p["ply"])):
        if pos["ply"] > max_ply:
            continue

        if pos["game_id"] != current_game:
            if current_game is not None:
                opening_depths.append(last_common_ply)
            current_game = pos["game_id"]
            last_common_ply = 0

        if pos["fen_board_only"] in common_positions:
            last_common_ply = pos["ply"]

    # Don't forget the last game
    if current_game is not None:
        opening_depths.append(last_common_ply)

    avg_opening_depth = sum(opening_depths) / len(opening_depths) if opening_depths else 0

    return {
        "positions": dict(position_counts),
        "common_positions": common_positions,
        "num_unique_positions": len(position_counts),
        "num_common_positions": len(common_positions),
        "opening_depth_avg": round(avg_opening_depth, 2),
        "total_games_analyzed": len(set(p["game_id"] for p in positions)),
    }


# =============================================================================
# Game-Level Aggregations
# =============================================================================

def aggregate_game_features(
    positions: list[dict],
    game_id: str,
) -> dict:
    """
    Aggregate position-level features to game level.

    Args:
        positions: List of position features for a single game
        game_id: The game identifier

    Returns:
        Dictionary with aggregated game features
    """
    if not positions:
        return {"game_id": game_id}

    # Filter to just this game
    game_positions = [p for p in positions if p["game_id"] == game_id]
    if not game_positions:
        return {"game_id": game_id}

    # Material swings
    balances = [p["material_balance"] for p in game_positions]

    # Fragility stats
    fragilities = [p["fragility"] for p in game_positions]

    # Phase distribution
    phases = [p["phase"] for p in game_positions]
    phase_counts = {
        "opening": phases.count("opening"),
        "middlegame": phases.count("middlegame"),
        "endgame": phases.count("endgame"),
    }
    total_positions = len(phases)

    # Move types (player moves only)
    player_moves = [p for p in game_positions if p["is_player_move"]]

    return {
        "game_id": game_id,
        "num_positions": len(game_positions),
        "num_player_moves": len(player_moves),

        # Material
        "material_max": max(balances),
        "material_min": min(balances),
        "material_swing": max(balances) - min(balances),
        "material_final": balances[-1] if balances else 0,

        # Fragility
        "fragility_avg": sum(fragilities) / len(fragilities) if fragilities else 0,
        "fragility_max": max(fragilities) if fragilities else 0,
        "fragility_std": _std(fragilities) if len(fragilities) > 1 else 0,

        # Phase distribution
        "opening_moves": phase_counts["opening"],
        "middlegame_moves": phase_counts["middlegame"],
        "endgame_moves": phase_counts["endgame"],
        "opening_pct": phase_counts["opening"] / total_positions if total_positions else 0,
        "middlegame_pct": phase_counts["middlegame"] / total_positions if total_positions else 0,
        "endgame_pct": phase_counts["endgame"] / total_positions if total_positions else 0,

        # Tablebase positions
        "tablebase_positions": sum(1 for p in game_positions if p["is_tablebase"]),

        # Move types
        "captures": sum(1 for p in player_moves if p["is_capture"]),
        "checks": sum(1 for p in player_moves if p["is_check"]),
        "castled": any(p["is_castling"] for p in player_moves),
    }


def _std(values: list[float]) -> float:
    """Calculate standard deviation."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return variance ** 0.5


def aggregate_all_games(positions: list[dict]) -> list[dict]:
    """
    Aggregate position features for all games.

    Args:
        positions: List of all position features

    Returns:
        List of game-level aggregation dictionaries
    """
    # Group by game
    games = defaultdict(list)
    for pos in positions:
        games[pos["game_id"]].append(pos)

    return [
        aggregate_game_features(positions, game_id)
        for game_id, positions in games.items()
    ]


# =============================================================================
# Save/Load Functions
# =============================================================================

def save_dataset_parquet(
    data: list[dict],
    path: Path,
):
    """
    Save dataset to parquet file.

    Args:
        data: List of dictionaries
        path: Output file path
    """
    try:
        import pandas as pd
        df = pd.DataFrame(data)
        df.to_parquet(path, index=False)
    except ImportError:
        # Fallback to JSON if pandas not available
        with open(path.with_suffix(".json"), "w") as f:
            json.dump(data, f, indent=2)


def save_opening_book(
    book: dict,
    path: Path,
):
    """
    Save opening book to JSON file.

    Args:
        book: Opening book dictionary
        path: Output file path
    """
    with open(path, "w") as f:
        json.dump(book, f, indent=2)


def load_dataset_parquet(path: Path) -> list[dict]:
    """
    Load dataset from parquet file.

    Args:
        path: Input file path

    Returns:
        List of dictionaries
    """
    try:
        import pandas as pd
        df = pd.read_parquet(path)
        return df.to_dict(orient="records")
    except ImportError:
        # Fallback to JSON
        with open(path.with_suffix(".json")) as f:
            return json.load(f)


# =============================================================================
# Elo & Result Pattern Analysis
# =============================================================================

@dataclass
class EloAnalysis:
    """Analysis of Elo rating patterns over time."""

    # Overall stats
    elo_start: int
    elo_end: int
    elo_change: int
    elo_min: int
    elo_max: int
    elo_range: int
    elo_avg: float
    elo_std: float

    # Win/loss stats
    total_games: int
    wins: int
    losses: int
    draws: int
    win_rate: float
    loss_rate: float

    # Streak analysis
    current_streak: int  # Positive = wins, negative = losses
    longest_win_streak: int
    longest_loss_streak: int
    streak_volatility: float  # How often streaks change

    # Upset analysis (wins against higher rated)
    upsets: int  # Wins against opponents 100+ Elo higher
    upset_rate: float
    major_upsets: int  # Wins against opponents 200+ Elo higher

    # Sandbagging indicators
    losses_to_lower: int  # Losses to opponents 100+ Elo lower
    losses_to_lower_rate: float
    suspicious_losses: int  # Losses to opponents 200+ Elo lower

    # Elo volatility
    elo_drops: int  # Number of significant Elo drops (>50 points)
    elo_recoveries: int  # Number of recoveries after drops
    rating_manipulation_score: float  # 0-1, higher = more suspicious pattern


@dataclass
class ResultPatterns:
    """Analysis of win/loss/draw patterns."""

    # Resolution breakdown
    wins_by_checkmate: int
    wins_by_resignation: int
    wins_by_timeout: int
    losses_by_checkmate: int
    losses_by_resignation: int
    losses_by_timeout: int
    draws_total: int

    # Pattern analysis
    checkmate_rate: float  # How often games end in checkmate
    resignation_rate: float  # How often games end in resignation
    timeout_rate: float

    # Consistency metrics
    result_entropy: float  # Randomness of results (higher = more random)
    run_test_p_value: float  # Wald-Wolfowitz runs test for randomness

    # Time of day patterns (if available)
    games_by_hour: dict  # hour -> count

    # Session analysis
    avg_games_per_session: float
    max_games_per_session: int


@dataclass
class TerminationPatterns:
    """Detailed analysis of game termination types from PGN Termination field."""

    # Total games
    total_games: int

    # Checkmate terminations
    win_checkmate: int
    loss_checkmate: int
    combined_checkmate: int
    checkmate_rate: float  # combined_checkmate / total

    # Timeout terminations
    win_timeout: int
    loss_timeout: int
    draw_timeout: int  # Rare but possible (both flag)
    combined_timeout: int
    timeout_rate: float  # combined_timeout / total

    # Resignation terminations
    win_resignation: int
    loss_resignation: int
    combined_resignation: int
    resignation_rate: float  # combined_resignation / total

    # Abandonment terminations
    win_abandonment: int
    loss_abandonment: int
    combined_abandonment: int
    abandonment_rate: float  # combined_abandonment / total

    # Draw terminations (detailed)
    draw_stalemate: int
    draw_repetition: int
    draw_insufficient: int
    draw_50move: int
    draw_agreement: int
    draw_other: int
    combined_draw: int
    draw_rate: float  # combined_draw / total

    # Other/unknown terminations
    other_terminations: int
    other_rate: float


def analyze_termination_patterns(games: list[dict]) -> TerminationPatterns:
    """
    Analyze game termination patterns from the resolution field.

    Args:
        games: List of game metadata dictionaries with 'resolution' field

    Returns:
        TerminationPatterns with detailed breakdown
    """
    total = len(games)
    if total == 0:
        return TerminationPatterns(
            total_games=0,
            win_checkmate=0, loss_checkmate=0, combined_checkmate=0, checkmate_rate=0.0,
            win_timeout=0, loss_timeout=0, draw_timeout=0, combined_timeout=0, timeout_rate=0.0,
            win_resignation=0, loss_resignation=0, combined_resignation=0, resignation_rate=0.0,
            win_abandonment=0, loss_abandonment=0, combined_abandonment=0, abandonment_rate=0.0,
            draw_stalemate=0, draw_repetition=0, draw_insufficient=0, draw_50move=0,
            draw_agreement=0, draw_other=0, combined_draw=0, draw_rate=0.0,
            other_terminations=0, other_rate=0.0,
        )

    resolutions = [g.get("resolution", "") for g in games]

    # Checkmate
    win_checkmate = sum(1 for r in resolutions if r == "win_checkmate")
    loss_checkmate = sum(1 for r in resolutions if r == "loss_checkmate")
    combined_checkmate = win_checkmate + loss_checkmate

    # Timeout
    win_timeout = sum(1 for r in resolutions if r == "win_timeout")
    loss_timeout = sum(1 for r in resolutions if r == "loss_timeout")
    draw_timeout = sum(1 for r in resolutions if r == "draw_timeout")
    combined_timeout = win_timeout + loss_timeout + draw_timeout

    # Resignation
    win_resignation = sum(1 for r in resolutions if r == "win_resignation")
    loss_resignation = sum(1 for r in resolutions if r == "loss_resignation")
    combined_resignation = win_resignation + loss_resignation

    # Abandonment
    win_abandonment = sum(1 for r in resolutions if r == "win_abandonment")
    loss_abandonment = sum(1 for r in resolutions if r == "loss_abandonment")
    combined_abandonment = win_abandonment + loss_abandonment

    # Draw types
    draw_stalemate = sum(1 for r in resolutions if r == "draw_stalemate")
    draw_repetition = sum(1 for r in resolutions if r == "draw_repetition")
    draw_insufficient = sum(1 for r in resolutions if r == "draw_insufficient")
    draw_50move = sum(1 for r in resolutions if r == "draw_50move")
    draw_agreement = sum(1 for r in resolutions if r == "draw_agreement")
    draw_other = sum(1 for r in resolutions if r == "draw_other")
    combined_draw = (draw_stalemate + draw_repetition + draw_insufficient +
                     draw_50move + draw_agreement + draw_other + draw_timeout)

    # Other (win_other, loss_other, or unrecognized)
    known_resolutions = {
        "win_checkmate", "loss_checkmate", "win_timeout", "loss_timeout", "draw_timeout",
        "win_resignation", "loss_resignation", "win_abandonment", "loss_abandonment",
        "draw_stalemate", "draw_repetition", "draw_insufficient", "draw_50move",
        "draw_agreement", "draw_other",
    }
    other_terminations = sum(1 for r in resolutions if r and r not in known_resolutions)

    return TerminationPatterns(
        total_games=total,
        win_checkmate=win_checkmate,
        loss_checkmate=loss_checkmate,
        combined_checkmate=combined_checkmate,
        checkmate_rate=combined_checkmate / total,
        win_timeout=win_timeout,
        loss_timeout=loss_timeout,
        draw_timeout=draw_timeout,
        combined_timeout=combined_timeout,
        timeout_rate=combined_timeout / total,
        win_resignation=win_resignation,
        loss_resignation=loss_resignation,
        combined_resignation=combined_resignation,
        resignation_rate=combined_resignation / total,
        win_abandonment=win_abandonment,
        loss_abandonment=loss_abandonment,
        combined_abandonment=combined_abandonment,
        abandonment_rate=combined_abandonment / total,
        draw_stalemate=draw_stalemate,
        draw_repetition=draw_repetition,
        draw_insufficient=draw_insufficient,
        draw_50move=draw_50move,
        draw_agreement=draw_agreement,
        draw_other=draw_other,
        combined_draw=combined_draw,
        draw_rate=combined_draw / total,
        other_terminations=other_terminations,
        other_rate=other_terminations / total,
    )


def analyze_elo_patterns(
    games: list[dict],
    elo_upset_threshold: int = 100,
    elo_major_upset_threshold: int = 200,
    elo_drop_threshold: int = 50,
) -> EloAnalysis:
    """
    Analyze Elo rating patterns for signs of sandbagging or manipulation.

    Args:
        games: List of game metadata dictionaries (sorted by date)
        elo_upset_threshold: Elo difference to count as upset (default 100)
        elo_major_upset_threshold: Elo difference for major upset (default 200)
        elo_drop_threshold: Elo drop to count as significant (default 50)

    Returns:
        EloAnalysis with pattern metrics
    """
    if not games:
        return _empty_elo_analysis()

    # Sort games by end_time
    sorted_games = sorted(games, key=lambda g: g.get("end_time", 0))

    # Extract Elo ratings
    elos = [g["player_elo"] for g in sorted_games if g.get("player_elo")]
    if not elos:
        return _empty_elo_analysis()

    # Basic Elo stats
    elo_start = elos[0]
    elo_end = elos[-1]
    elo_min = min(elos)
    elo_max = max(elos)

    # Win/loss counts
    results = [g.get("player_result", "") for g in sorted_games]
    wins = results.count("win")
    losses = results.count("loss")
    draws = results.count("draw")
    total = len(results)

    # Streak analysis
    current_streak, longest_win, longest_loss, streak_changes = _analyze_streaks(results)
    streak_volatility = streak_changes / max(1, total - 1)

    # Upset analysis
    upsets = 0
    major_upsets = 0
    losses_to_lower = 0
    suspicious_losses = 0

    for g in sorted_games:
        elo_diff = g.get("elo_diff", 0)
        result = g.get("player_result", "")

        if result == "win" and elo_diff < -elo_upset_threshold:
            upsets += 1
            if elo_diff < -elo_major_upset_threshold:
                major_upsets += 1

        if result == "loss" and elo_diff > elo_upset_threshold:
            losses_to_lower += 1
            if elo_diff > elo_major_upset_threshold:
                suspicious_losses += 1

    # Elo volatility (drops and recoveries)
    elo_drops = 0
    elo_recoveries = 0
    in_drop = False
    drop_low = None

    for i in range(1, len(elos)):
        diff = elos[i] - elos[i-1]
        if diff <= -elo_drop_threshold:
            elo_drops += 1
            in_drop = True
            drop_low = elos[i]
        elif in_drop and drop_low and elos[i] >= drop_low + elo_drop_threshold:
            elo_recoveries += 1
            in_drop = False

    # Rating manipulation score (heuristic)
    manipulation_score = _calculate_manipulation_score(
        losses_to_lower=losses_to_lower,
        suspicious_losses=suspicious_losses,
        elo_drops=elo_drops,
        elo_recoveries=elo_recoveries,
        total_games=total,
    )

    return EloAnalysis(
        elo_start=elo_start,
        elo_end=elo_end,
        elo_change=elo_end - elo_start,
        elo_min=elo_min,
        elo_max=elo_max,
        elo_range=elo_max - elo_min,
        elo_avg=sum(elos) / len(elos),
        elo_std=_std(elos),
        total_games=total,
        wins=wins,
        losses=losses,
        draws=draws,
        win_rate=wins / total if total else 0,
        loss_rate=losses / total if total else 0,
        current_streak=current_streak,
        longest_win_streak=longest_win,
        longest_loss_streak=longest_loss,
        streak_volatility=streak_volatility,
        upsets=upsets,
        upset_rate=upsets / wins if wins else 0,
        major_upsets=major_upsets,
        losses_to_lower=losses_to_lower,
        losses_to_lower_rate=losses_to_lower / losses if losses else 0,
        suspicious_losses=suspicious_losses,
        elo_drops=elo_drops,
        elo_recoveries=elo_recoveries,
        rating_manipulation_score=manipulation_score,
    )


def analyze_result_patterns(
    games: list[dict],
    session_gap_minutes: int = 60,
) -> ResultPatterns:
    """
    Analyze win/loss/draw patterns for randomness and consistency.

    Args:
        games: List of game metadata dictionaries
        session_gap_minutes: Gap between games to consider new session

    Returns:
        ResultPatterns with analysis metrics
    """
    if not games:
        return _empty_result_patterns()

    # Resolution breakdown
    resolutions = [g.get("resolution", "") for g in games]

    wins_checkmate = sum(1 for r in resolutions if r == "win_checkmate")
    wins_resignation = sum(1 for r in resolutions if r == "win_resignation")
    wins_timeout = sum(1 for r in resolutions if r == "win_timeout")
    losses_checkmate = sum(1 for r in resolutions if r == "loss_checkmate")
    losses_resignation = sum(1 for r in resolutions if r == "loss_resignation")
    losses_timeout = sum(1 for r in resolutions if r == "loss_timeout")
    draws = sum(1 for r in resolutions if r.startswith("draw"))

    total = len(games)

    # Resolution rates
    checkmates = wins_checkmate + losses_checkmate
    resignations = wins_resignation + losses_resignation
    timeouts = wins_timeout + losses_timeout

    # Result entropy (measure of randomness)
    results = [g.get("player_result", "") for g in games]
    entropy = _calculate_entropy(results)

    # Runs test for randomness
    p_value = _runs_test(results)

    # Time of day analysis
    games_by_hour = defaultdict(int)
    for g in games:
        end_time = g.get("end_time", 0)
        if end_time:
            hour = datetime.fromtimestamp(end_time).hour
            games_by_hour[hour] += 1

    # Session analysis
    sorted_games = sorted(games, key=lambda g: g.get("end_time", 0))
    sessions = _identify_sessions(sorted_games, session_gap_minutes)
    session_sizes = [len(s) for s in sessions]

    return ResultPatterns(
        wins_by_checkmate=wins_checkmate,
        wins_by_resignation=wins_resignation,
        wins_by_timeout=wins_timeout,
        losses_by_checkmate=losses_checkmate,
        losses_by_resignation=losses_resignation,
        losses_by_timeout=losses_timeout,
        draws_total=draws,
        checkmate_rate=checkmates / total if total else 0,
        resignation_rate=resignations / total if total else 0,
        timeout_rate=timeouts / total if total else 0,
        result_entropy=entropy,
        run_test_p_value=p_value,
        games_by_hour=dict(games_by_hour),
        avg_games_per_session=sum(session_sizes) / len(sessions) if sessions else 0,
        max_games_per_session=max(session_sizes) if session_sizes else 0,
    )


def _analyze_streaks(results: list[str]) -> tuple[int, int, int, int]:
    """
    Analyze win/loss streaks.

    Returns:
        Tuple of (current_streak, longest_win, longest_loss, streak_changes)
    """
    if not results:
        return 0, 0, 0, 0

    current = 0
    longest_win = 0
    longest_loss = 0
    streak_changes = 0
    prev_result = None

    for result in results:
        if result == "win":
            if current >= 0:
                current += 1
            else:
                streak_changes += 1
                current = 1
            longest_win = max(longest_win, current)
        elif result == "loss":
            if current <= 0:
                current -= 1
            else:
                streak_changes += 1
                current = -1
            longest_loss = max(longest_loss, abs(current))
        else:  # draw
            if current != 0:
                streak_changes += 1
            current = 0

        prev_result = result

    return current, longest_win, longest_loss, streak_changes


def _calculate_entropy(results: list[str]) -> float:
    """
    Calculate Shannon entropy of results (measure of randomness).

    Higher entropy = more random/unpredictable results.
    """
    import math

    if not results:
        return 0.0

    counts = defaultdict(int)
    for r in results:
        counts[r] += 1

    total = len(results)
    entropy = 0.0

    for count in counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)

    return entropy


def _runs_test(results: list[str]) -> float:
    """
    Wald-Wolfowitz runs test for randomness.

    Returns p-value. Low p-value (<0.05) suggests non-random pattern.
    """
    import math

    if len(results) < 10:
        return 1.0  # Not enough data

    # Convert to binary (win vs not-win)
    binary = [1 if r == "win" else 0 for r in results]

    n1 = sum(binary)
    n2 = len(binary) - n1

    if n1 == 0 or n2 == 0:
        return 1.0  # All same result

    # Count runs
    runs = 1
    for i in range(1, len(binary)):
        if binary[i] != binary[i-1]:
            runs += 1

    # Expected runs and variance
    n = n1 + n2
    expected = (2 * n1 * n2) / n + 1
    variance = (2 * n1 * n2 * (2 * n1 * n2 - n)) / (n * n * (n - 1))

    if variance <= 0:
        return 1.0

    # Z-score
    z = (runs - expected) / math.sqrt(variance)

    # Two-tailed p-value (approximation)
    p_value = 2 * (1 - _normal_cdf(abs(z)))

    return p_value


def _normal_cdf(z: float) -> float:
    """Approximate standard normal CDF."""
    import math
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def _identify_sessions(
    games: list[dict],
    gap_minutes: int,
) -> list[list[dict]]:
    """
    Group games into sessions based on time gaps.
    """
    if not games:
        return []

    sessions = []
    current_session = [games[0]]
    gap_seconds = gap_minutes * 60

    for i in range(1, len(games)):
        prev_time = games[i-1].get("end_time", 0)
        curr_time = games[i].get("end_time", 0)

        if curr_time - prev_time > gap_seconds:
            sessions.append(current_session)
            current_session = [games[i]]
        else:
            current_session.append(games[i])

    sessions.append(current_session)
    return sessions


def _calculate_manipulation_score(
    losses_to_lower: int,
    suspicious_losses: int,
    elo_drops: int,
    elo_recoveries: int,
    total_games: int,
) -> float:
    """
    Calculate a heuristic score for potential rating manipulation.

    Returns value 0-1, higher = more suspicious.
    """
    if total_games < 10:
        return 0.0

    # Factors that increase suspicion
    score = 0.0

    # Losing to much lower rated players
    if suspicious_losses > 0:
        score += min(0.3, suspicious_losses * 0.1)

    # Regular losses to lower rated
    loss_rate = losses_to_lower / total_games
    if loss_rate > 0.1:
        score += min(0.2, loss_rate)

    # Pattern of drops and recoveries (sandbagging)
    if elo_recoveries > 0 and elo_drops > 0:
        recovery_ratio = elo_recoveries / elo_drops
        if recovery_ratio > 0.5:
            score += min(0.3, recovery_ratio * 0.2)

    # Frequent significant Elo drops
    drop_rate = elo_drops / total_games
    if drop_rate > 0.05:
        score += min(0.2, drop_rate * 2)

    return min(1.0, score)


def _empty_elo_analysis() -> EloAnalysis:
    """Return empty EloAnalysis."""
    return EloAnalysis(
        elo_start=0, elo_end=0, elo_change=0, elo_min=0, elo_max=0,
        elo_range=0, elo_avg=0, elo_std=0, total_games=0, wins=0,
        losses=0, draws=0, win_rate=0, loss_rate=0, current_streak=0,
        longest_win_streak=0, longest_loss_streak=0, streak_volatility=0,
        upsets=0, upset_rate=0, major_upsets=0, losses_to_lower=0,
        losses_to_lower_rate=0, suspicious_losses=0, elo_drops=0,
        elo_recoveries=0, rating_manipulation_score=0,
    )


def _empty_result_patterns() -> ResultPatterns:
    """Return empty ResultPatterns."""
    return ResultPatterns(
        wins_by_checkmate=0, wins_by_resignation=0, wins_by_timeout=0,
        losses_by_checkmate=0, losses_by_resignation=0, losses_by_timeout=0,
        draws_total=0, checkmate_rate=0, resignation_rate=0, timeout_rate=0,
        result_entropy=0, run_test_p_value=1.0, games_by_hour={},
        avg_games_per_session=0, max_games_per_session=0,
    )


@dataclass
class EloSegmentStats:
    """Statistics for a segment of games by Elo advantage."""
    segment_name: str  # "Player Favored", "Fair Match", "Opponent Favored"
    games: int
    wins: int
    losses: int
    draws: int
    win_rate: float
    expected_win_rate: float  # Based on Elo difference
    performance_vs_expected: float  # Actual - Expected
    avg_elo_diff: float
    avg_game_length: float
    checkmate_wins: int
    resignation_wins: int
    timeout_wins: int
    checkmate_losses: int
    resignation_losses: int
    timeout_losses: int


def calculate_expected_win_rate(elo_diff: int) -> float:
    """
    Calculate expected win rate based on Elo difference.

    Uses the standard Elo formula:
    Expected = 1 / (1 + 10^(-elo_diff/400))

    Args:
        elo_diff: Player Elo - Opponent Elo

    Returns:
        Expected win rate (0-1)
    """
    import math
    return 1 / (1 + math.pow(10, -elo_diff / 400))


def analyze_opponent_segments(
    games: list[dict],
    elo_threshold: int = 200,
) -> dict:
    """
    Analyze games segmented by Elo advantage (Hybrid approach).

    Segments:
    - Player Favored: Player Elo > Opponent Elo + threshold
    - Fair Match: |Elo difference| <= threshold
    - Opponent Favored: Opponent Elo > Player Elo + threshold

    Args:
        games: List of game metadata dicts with player_elo, opponent_elo, result, etc.
        elo_threshold: Elo difference threshold for favoritism (default 200).

    Returns:
        Dictionary with:
        - segments: {segment_name: EloSegmentStats}
        - prioritized_games: List of games worth deeper analysis
        - flags: List of suspicious patterns found
    """
    from collections import defaultdict
    import math

    segments = {
        "Player Favored": [],
        "Fair Match": [],
        "Opponent Favored": [],
    }

    # Classify games into segments
    for game in games:
        player_elo = game.get("player_elo", 0)
        opponent_elo = game.get("opponent_elo", 0)

        if not player_elo or not opponent_elo:
            continue

        elo_diff = player_elo - opponent_elo

        if elo_diff > elo_threshold:
            segments["Player Favored"].append(game)
        elif elo_diff < -elo_threshold:
            segments["Opponent Favored"].append(game)
        else:
            segments["Fair Match"].append(game)

    # Analyze each segment
    segment_stats = {}
    for segment_name, segment_games in segments.items():
        if not segment_games:
            continue

        wins = sum(1 for g in segment_games if g.get("outcome") == "won")
        losses = sum(1 for g in segment_games if g.get("outcome") == "lost")
        draws = len(segment_games) - wins - losses

        # Result types
        checkmate_wins = sum(1 for g in segment_games
                            if g.get("outcome") == "won" and "checkmate" in g.get("resolution_type", "").lower())
        resignation_wins = sum(1 for g in segment_games
                              if g.get("outcome") == "won" and "resign" in g.get("resolution_type", "").lower())
        timeout_wins = sum(1 for g in segment_games
                          if g.get("outcome") == "won" and "timeout" in g.get("resolution_type", "").lower())

        checkmate_losses = sum(1 for g in segment_games
                              if g.get("outcome") == "lost" and "checkmate" in g.get("resolution_type", "").lower())
        resignation_losses = sum(1 for g in segment_games
                                if g.get("outcome") == "lost" and "resign" in g.get("resolution_type", "").lower())
        timeout_losses = sum(1 for g in segment_games
                            if g.get("outcome") == "lost" and "timeout" in g.get("resolution_type", "").lower())

        # Calculate expected win rate based on average Elo diff
        elo_diffs = [g.get("player_elo", 0) - g.get("opponent_elo", 0) for g in segment_games]
        avg_elo_diff = sum(elo_diffs) / len(elo_diffs)
        expected_win_rate = calculate_expected_win_rate(int(avg_elo_diff))

        actual_win_rate = wins / len(segment_games) if segment_games else 0
        game_lengths = [g.get("move_count", 0) for g in segment_games]
        avg_game_length = sum(game_lengths) / len(game_lengths) if game_lengths else 0

        segment_stats[segment_name] = EloSegmentStats(
            segment_name=segment_name,
            games=len(segment_games),
            wins=wins,
            losses=losses,
            draws=draws,
            win_rate=round(actual_win_rate, 3),
            expected_win_rate=round(expected_win_rate, 3),
            performance_vs_expected=round(actual_win_rate - expected_win_rate, 3),
            avg_elo_diff=round(avg_elo_diff, 1),
            avg_game_length=round(avg_game_length, 1),
            checkmate_wins=checkmate_wins,
            resignation_wins=resignation_wins,
            timeout_wins=timeout_wins,
            checkmate_losses=checkmate_losses,
            resignation_losses=resignation_losses,
            timeout_losses=timeout_losses,
        )

    # Identify games worth deeper analysis (Hybrid prioritization)
    prioritized_games = []
    for game in games:
        player_elo = game.get("player_elo", 0)
        opponent_elo = game.get("opponent_elo", 0)
        outcome = game.get("outcome", "")

        if not player_elo or not opponent_elo:
            continue

        elo_diff = player_elo - opponent_elo

        # Prioritize: Upset wins (won as significant underdog)
        if elo_diff < -300 and outcome == "won":
            game["priority_reason"] = "Major upset win (300+ Elo underdog)"
            prioritized_games.append(game)
        # Prioritize: Won against much higher rated opponent
        elif elo_diff < -200 and outcome == "won":
            game["priority_reason"] = "Upset win (200+ Elo underdog)"
            prioritized_games.append(game)

    # Detect suspicious patterns
    flags = []

    # Check Opponent Favored segment for unusual performance
    if "Opponent Favored" in segment_stats:
        stats = segment_stats["Opponent Favored"]
        if stats.games >= 5:
            # Significantly outperforming expectations
            if stats.performance_vs_expected > 0.15:
                flags.append(f"High performance vs opponents 200+ Elo higher: "
                           f"{stats.win_rate:.0%} actual vs {stats.expected_win_rate:.0%} expected")
            # Win rate > 50% against higher-rated opponents
            if stats.win_rate > 0.5:
                flags.append(f"Winning majority ({stats.win_rate:.0%}) against 200+ Elo stronger opponents")

    # Check Fair Match segment
    if "Fair Match" in segment_stats:
        stats = segment_stats["Fair Match"]
        if stats.games >= 10 and stats.performance_vs_expected > 0.20:
            flags.append(f"Outperforming expectations in fair matches: "
                        f"{stats.win_rate:.0%} actual vs {stats.expected_win_rate:.0%} expected")

    return {
        "segments": segment_stats,
        "prioritized_games": prioritized_games,
        "flags": flags,
        "total_games_analyzed": len(games),
    }


# =============================================================================
# Feature Extraction Functions
# =============================================================================

def categorize_fragility(
    fragility_max: float,
    high_frag_count: int,
    total_positions: int,
) -> str:
    """
    Categorize a game by its fragility pattern.

    Categories based on max fragility and % of high-fragility positions:
    - "low": max < 0.3 or very few high-fragility positions
    - "medium": moderate fragility
    - "high": high max fragility or many high-fragility positions
    - "extreme": very high max fragility AND many high-fragility positions

    Args:
        fragility_max: Maximum fragility score in the game
        high_frag_count: Number of positions with fragility > 0.5
        total_positions: Total number of positions analyzed

    Returns:
        Category string: "low", "medium", "high", "extreme"
    """
    if total_positions == 0:
        return "low"

    high_frag_pct = high_frag_count / total_positions

    # Extreme: both very high max AND many high positions
    if fragility_max > 0.8 and high_frag_pct > 0.15:
        return "extreme"
    # High: high max OR many high positions
    elif fragility_max > 0.6 or high_frag_pct > 0.10:
        return "high"
    # Medium: moderate fragility
    elif fragility_max > 0.3 or high_frag_pct > 0.05:
        return "medium"
    else:
        return "low"


def categorize_material_trajectory(
    material_values: list[int],
    player_color: str,
) -> str:
    """
    Categorize how material balance changed throughout the game.

    Categories:
    - "gained": Ended with significantly more material than started
    - "lost": Ended with significantly less material than started
    - "stable": Little change throughout
    - "volatile": Large swings in material balance

    Args:
        material_values: List of material balance values (player - opponent) per position
        player_color: "white" or "black"

    Returns:
        Category string: "gained", "lost", "stable", "volatile"
    """
    if not material_values or len(material_values) < 2:
        return "stable"

    import statistics

    # Calculate change from start to end (in centipawns)
    start_material = material_values[0]
    end_material = material_values[-1]
    net_change = end_material - start_material

    # Calculate volatility (standard deviation of changes)
    changes = [material_values[i] - material_values[i-1]
               for i in range(1, len(material_values))]
    if changes:
        volatility = statistics.stdev(changes) if len(changes) > 1 else 0
    else:
        volatility = 0

    # Thresholds in centipawns (100 = 1 pawn)
    GAIN_THRESHOLD = 200  # 2 pawns
    VOLATILITY_THRESHOLD = 150  # High volatility

    # Volatile: lots of back-and-forth exchanges
    if volatility > VOLATILITY_THRESHOLD:
        return "volatile"
    # Gained: ended significantly ahead
    elif net_change > GAIN_THRESHOLD:
        return "gained"
    # Lost: ended significantly behind
    elif net_change < -GAIN_THRESHOLD:
        return "lost"
    else:
        return "stable"


def extract_game_features(
    game_metadata: dict,
    pgn_game: "chess.pgn.Game",
    player_color: str,
    position_features: Optional[list[dict]] = None,
) -> GameFeatures:
    """
    Extract comprehensive game features combining existing analysis functions.

    Uses existing functions from chess_analysis:
    - analyze_game_phases() for phase durations
    - calculate_fragility_simple() for fragility per position
    - extract_clock_times() + analyze_time_patterns() for time analysis

    Args:
        game_metadata: Game metadata dict (from build_game_dataset)
        pgn_game: Parsed PGN game object
        player_color: "white" or "black"
        position_features: Optional pre-computed position features (from build_position_dataset)

    Returns:
        GameFeatures dataclass with all extracted features
    """
    from .game_phase import analyze_game_phases, GamePhase
    from .time_analysis import extract_clock_times, analyze_time_patterns

    game_id = game_metadata.get("game_id", "unknown")
    time_class = game_metadata.get("time_class", "unknown")
    base_time = game_metadata.get("base_time", 0)
    increment = game_metadata.get("increment", 0)

    # =================
    # Game Phase Analysis
    # =================
    phase_info = analyze_game_phases(pgn_game)
    total_moves = phase_info.get("total_moves", 0)
    opening_moves = phase_info.get("opening_length", 0)
    middlegame_moves = phase_info.get("middlegame_length", 0)
    endgame_moves = phase_info.get("endgame_length", 0)
    reached_endgame = phase_info.get("endgame_start_ply") is not None

    # =================
    # Time Analysis
    # =================
    avg_time_per_move = None
    time_cv = None
    fast_move_pct = None

    try:
        clock_data = extract_clock_times(pgn_game)
        if clock_data:
            time_stats = analyze_time_patterns(clock_data, player_color)
            if time_stats and time_stats.get("num_moves", 0) > 0:
                avg_time_per_move = time_stats.get("avg_time")
                time_cv = time_stats.get("cv_time")
                fast_move_pct = time_stats.get("fast_move_pct")
    except Exception:
        pass  # Clock data not available

    # =================
    # Fragility Analysis
    # =================
    fragility_values = []
    material_values = []

    if position_features:
        # Use pre-computed position features
        is_white = player_color == "white"
        for pos in position_features:
            if pos.get("is_player_move", False) or True:  # Include all positions
                frag = pos.get("fragility", 0.0)
                fragility_values.append(frag)

                # Material balance (player perspective)
                mat_balance = pos.get("material_balance", 0)
                # Convert to centipawns (already in pawn units, multiply by 100)
                material_values.append(mat_balance * 100)
    else:
        # Compute fragility on-the-fly
        board = pgn_game.board()
        for move in pgn_game.mainline_moves():
            # Calculate fragility for side to move
            frag = calculate_fragility_simple(board)
            fragility_values.append(frag)

            # Material balance
            mat_white = calculate_material(board, chess.WHITE)
            mat_black = calculate_material(board, chess.BLACK)
            if player_color == "white":
                mat_balance = (mat_white - mat_black) * 100
            else:
                mat_balance = (mat_black - mat_white) * 100
            material_values.append(mat_balance)

            board.push(move)

    # Fragility stats
    import statistics
    if fragility_values:
        fragility_mean = statistics.mean(fragility_values)
        fragility_max = max(fragility_values)
        fragility_std = statistics.stdev(fragility_values) if len(fragility_values) > 1 else 0.0
        high_fragility_positions = sum(1 for f in fragility_values if f > 0.5)
    else:
        fragility_mean = 0.0
        fragility_max = 0.0
        fragility_std = 0.0
        high_fragility_positions = 0

    fragility_category = categorize_fragility(
        fragility_max, high_fragility_positions, len(fragility_values)
    )

    # =================
    # Material Analysis
    # =================
    if material_values:
        avg_material_balance = statistics.mean(material_values)
        max_material_advantage = max(material_values)
        max_material_disadvantage = min(material_values)

        # Count material swings (changes > 100cp = 1 pawn)
        material_swings = 0
        for i in range(1, len(material_values)):
            if abs(material_values[i] - material_values[i-1]) > 100:
                material_swings += 1
    else:
        avg_material_balance = 0.0
        max_material_advantage = 0
        max_material_disadvantage = 0
        material_swings = 0

    material_trajectory = categorize_material_trajectory(material_values, player_color)

    return GameFeatures(
        game_id=game_id,
        time_class=time_class,
        base_time=base_time,
        increment=increment,
        avg_time_per_move=round(avg_time_per_move, 2) if avg_time_per_move else None,
        time_cv=round(time_cv, 3) if time_cv else None,
        fast_move_pct=round(fast_move_pct, 3) if fast_move_pct else None,
        total_moves=total_moves,
        opening_moves=opening_moves,
        middlegame_moves=middlegame_moves,
        endgame_moves=endgame_moves,
        reached_endgame=reached_endgame,
        fragility_mean=round(fragility_mean, 4),
        fragility_max=round(fragility_max, 4),
        fragility_std=round(fragility_std, 4),
        high_fragility_positions=high_fragility_positions,
        fragility_category=fragility_category,
        material_swings=material_swings,
        max_material_advantage=max_material_advantage,
        max_material_disadvantage=max_material_disadvantage,
        avg_material_balance=round(avg_material_balance, 1),
        material_trajectory=material_trajectory,
    )


def segment_games_by_elo_range(
    games: list[dict],
    segment_size: int = 200,
    segment_by: str = "player",
    trustworthy_opponents: Optional[set[str]] = None,
) -> dict[str, EloRangeStats]:
    """
    Segment games into Elo range buckets (e.g., 1000-1200, 1200-1400).

    Args:
        games: List of game metadata dicts
        segment_size: Elo range per bucket (default 200)
        segment_by: "player" to segment by player's Elo, "opponent" for opponent's Elo
        trustworthy_opponents: Set of opponent usernames known to be trustworthy

    Returns:
        Dict mapping segment name to EloRangeStats
    """
    segments: dict[str, list[dict]] = defaultdict(list)

    elo_key = "player_elo" if segment_by == "player" else "opponent_elo"

    for game in games:
        elo = game.get(elo_key, 0)
        if not elo:
            continue

        # Calculate segment boundaries
        elo_min = (elo // segment_size) * segment_size
        elo_max = elo_min + segment_size
        segment_name = f"{elo_min}-{elo_max}"

        segments[segment_name].append(game)

    # Calculate stats for each segment
    result = {}
    for segment_name, segment_games in sorted(segments.items()):
        if not segment_games:
            continue

        # Parse segment name to get elo range
        parts = segment_name.split("-")
        elo_min = int(parts[0])
        elo_max = int(parts[1])

        # Count results
        wins = sum(1 for g in segment_games if g.get("player_result") == "win")
        losses = sum(1 for g in segment_games if g.get("player_result") == "loss")
        draws = sum(1 for g in segment_games if g.get("player_result") == "draw")

        # Count trustworthy games
        if trustworthy_opponents is not None:
            trustworthy_games = sum(
                1 for g in segment_games
                if g.get("opponent_username", "").lower() in trustworthy_opponents
            )
            untrustworthy_games = len(segment_games) - trustworthy_games
        else:
            trustworthy_games = len(segment_games)
            untrustworthy_games = 0

        # Calculate averages
        opponent_elos = [g.get("opponent_elo", 0) for g in segment_games if g.get("opponent_elo")]
        avg_opponent_elo = sum(opponent_elos) / len(opponent_elos) if opponent_elos else 0

        game_lengths = [g.get("num_moves", 0) for g in segment_games if g.get("num_moves")]
        avg_game_length = sum(game_lengths) / len(game_lengths) if game_lengths else 0

        total = len(segment_games)
        win_rate = wins / total if total > 0 else 0.0

        result[segment_name] = EloRangeStats(
            segment_name=segment_name,
            elo_min=elo_min,
            elo_max=elo_max,
            total_games=total,
            trustworthy_games=trustworthy_games,
            untrustworthy_games=untrustworthy_games,
            wins=wins,
            losses=losses,
            draws=draws,
            win_rate=round(win_rate, 3),
            avg_opponent_elo=round(avg_opponent_elo, 1),
            avg_game_length=round(avg_game_length, 1),
        )

    return result
