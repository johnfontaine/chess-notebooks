"""
Lichess Tablebase API integration.

Provides access to 7-piece Syzygy tablebases via the Lichess API.
Used to check if endgame positions are played optimally.

API Documentation:
    https://lichess.org/api#tag/Tablebase

Rate Limits:
    - Reasonable usage expected (no formal limit published)
    - This module includes basic rate limiting
"""

import time
import urllib.parse
from dataclasses import dataclass
from typing import Optional

import chess
import requests


# Lichess tablebase API endpoint
TABLEBASE_API_URL = "https://tablebase.lichess.ovh/standard"

# Rate limiting: minimum seconds between API calls
MIN_REQUEST_INTERVAL = 0.1

# Maximum pieces for tablebase lookup (Syzygy supports up to 7)
MAX_TABLEBASE_PIECES = 7


@dataclass
class TablebaseResult:
    """Result from a tablebase probe."""

    fen: str
    category: str  # "win", "loss", "draw", "maybe-win", "maybe-loss", "cursed-win", "blessed-loss"
    dtz: Optional[int]  # Distance to zeroing (pawn move or capture)
    dtm: Optional[int]  # Distance to mate (not always available)
    best_move: Optional[str]  # Best move in UCI notation
    wdl: int  # Win/Draw/Loss from side to move: 2=win, 1=cursed-win, 0=draw, -1=blessed-loss, -2=loss
    insufficient_material: bool
    is_stalemate: bool
    is_checkmate: bool
    is_variant_end: bool
    moves: list[dict]  # All legal moves with their WDL/DTZ values


@dataclass
class TablebaseMoveCheck:
    """Result of checking if a move matches tablebase."""

    move: str  # UCI notation
    is_tablebase_position: bool
    is_winning_move: bool  # True if move maintains/achieves winning position
    is_drawing_move: bool  # True if move maintains/achieves drawing position
    is_losing_move: bool  # True if move leads to loss
    is_best_move: bool  # True if move is the best tablebase move
    move_dtz: Optional[int]  # DTZ after this move
    best_dtz: Optional[int]  # DTZ of the best move
    tablebase_category: Optional[str]  # Category after the move


class TablebaseClient:
    """
    Client for querying the Lichess tablebase API.

    Usage:
        client = TablebaseClient()
        result = client.probe("8/8/8/8/8/5k2/8/4K2R w - - 0 1")
        print(result.best_move)  # "h1h8" or similar

    Or with rate limiting for bulk queries:
        with TablebaseClient() as client:
            for fen in positions:
                result = client.probe(fen)
    """

    def __init__(self, min_interval: float = MIN_REQUEST_INTERVAL):
        """
        Initialize the tablebase client.

        Args:
            min_interval: Minimum seconds between API requests (rate limiting).
        """
        self.min_interval = min_interval
        self._last_request_time: float = 0
        self._session: Optional[requests.Session] = None

    def __enter__(self):
        self._session = requests.Session()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._session:
            self._session.close()
            self._session = None

    def _get_session(self) -> requests.Session:
        """Get the HTTP session, creating one if needed."""
        if self._session is None:
            return requests.Session()
        return self._session

    def _rate_limit(self):
        """Apply rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_request_time = time.time()

    def probe(self, fen: str) -> Optional[TablebaseResult]:
        """
        Query the tablebase for a position.

        Args:
            fen: FEN string of the position.

        Returns:
            TablebaseResult with WDL, DTZ, best move, etc.
            None if the position has too many pieces or API error.
        """
        # Check piece count first
        board = chess.Board(fen)
        if not is_tablebase_position(board):
            return None

        self._rate_limit()

        # URL encode the FEN
        encoded_fen = urllib.parse.quote(fen, safe="")
        url = f"{TABLEBASE_API_URL}?fen={encoded_fen}"

        try:
            session = self._get_session()
            response = session.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException:
            return None
        except ValueError:  # JSON decode error
            return None

        # Parse the response
        category = data.get("category", "unknown")
        dtz = data.get("dtz")
        dtm = data.get("dtm")
        moves = data.get("moves", [])

        # Find the best move (first in the list is best)
        best_move = None
        if moves:
            best_move = moves[0].get("uci")

        # Convert category to WDL value
        wdl = _category_to_wdl(category)

        return TablebaseResult(
            fen=fen,
            category=category,
            dtz=dtz,
            dtm=dtm,
            best_move=best_move,
            wdl=wdl,
            insufficient_material=data.get("insufficient_material", False),
            is_stalemate=data.get("stalemate", False),
            is_checkmate=data.get("checkmate", False),
            is_variant_end=data.get("variant_end", False),
            moves=moves,
        )

    def check_move(
        self,
        board: chess.Board,
        move: chess.Move,
    ) -> TablebaseMoveCheck:
        """
        Check if a move is optimal according to tablebases.

        Args:
            board: Current position (before the move).
            move: The move to check.

        Returns:
            TablebaseMoveCheck with analysis of the move quality.
        """
        uci = move.uci()

        if not is_tablebase_position(board):
            return TablebaseMoveCheck(
                move=uci,
                is_tablebase_position=False,
                is_winning_move=False,
                is_drawing_move=False,
                is_losing_move=False,
                is_best_move=False,
                move_dtz=None,
                best_dtz=None,
                tablebase_category=None,
            )

        result = self.probe(board.fen())
        if result is None:
            return TablebaseMoveCheck(
                move=uci,
                is_tablebase_position=True,
                is_winning_move=False,
                is_drawing_move=False,
                is_losing_move=False,
                is_best_move=False,
                move_dtz=None,
                best_dtz=None,
                tablebase_category=None,
            )

        # Find the move in the tablebase results
        move_info = None
        best_move_info = result.moves[0] if result.moves else None
        for m in result.moves:
            if m.get("uci") == uci:
                move_info = m
                break

        if move_info is None:
            # Move not found in tablebase (shouldn't happen for legal moves)
            return TablebaseMoveCheck(
                move=uci,
                is_tablebase_position=True,
                is_winning_move=False,
                is_drawing_move=False,
                is_losing_move=False,
                is_best_move=False,
                move_dtz=None,
                best_dtz=result.dtz,
                tablebase_category=result.category,
            )

        move_category = move_info.get("category", "unknown")
        move_wdl = _category_to_wdl(move_category)
        move_dtz = move_info.get("dtz")

        # Determine move quality
        # Note: WDL after move is from opponent's perspective, so we negate
        opponent_wdl = -move_wdl
        is_winning = opponent_wdl == 2 or opponent_wdl == 1  # We win or cursed-win
        is_drawing = opponent_wdl == 0
        is_losing = opponent_wdl == -2 or opponent_wdl == -1

        is_best = best_move_info is not None and best_move_info.get("uci") == uci

        return TablebaseMoveCheck(
            move=uci,
            is_tablebase_position=True,
            is_winning_move=is_winning,
            is_drawing_move=is_drawing,
            is_losing_move=is_losing,
            is_best_move=is_best,
            move_dtz=move_dtz,
            best_dtz=best_move_info.get("dtz") if best_move_info else None,
            tablebase_category=move_category,
        )


def _category_to_wdl(category: str) -> int:
    """
    Convert tablebase category string to WDL value.

    WDL values (from side to move perspective):
        2: win
        1: cursed-win (win but 50-move rule may apply)
        0: draw
       -1: blessed-loss (loss but 50-move rule may save)
       -2: loss
    """
    mapping = {
        "win": 2,
        "maybe-win": 2,  # Treat as win
        "cursed-win": 1,
        "draw": 0,
        "blessed-loss": -1,
        "maybe-loss": -2,  # Treat as loss
        "loss": -2,
    }
    return mapping.get(category, 0)


def is_tablebase_position(board: chess.Board) -> bool:
    """
    Check if a position is eligible for tablebase lookup.

    Syzygy tablebases support positions with up to 7 pieces total
    (including kings).

    Args:
        board: The chess position to check.

    Returns:
        True if the position has 7 or fewer pieces.
    """
    piece_count = chess.popcount(board.occupied)
    return piece_count <= MAX_TABLEBASE_PIECES


def probe_tablebase(fen: str) -> Optional[TablebaseResult]:
    """
    Convenience function to probe the tablebase for a single position.

    For bulk queries, use TablebaseClient with context manager for
    connection reuse and rate limiting.

    Args:
        fen: FEN string of the position.

    Returns:
        TablebaseResult or None if not a tablebase position.
    """
    client = TablebaseClient()
    return client.probe(fen)


def check_tablebase_move(
    board: chess.Board,
    move: chess.Move,
) -> TablebaseMoveCheck:
    """
    Convenience function to check a single move against tablebases.

    For bulk queries, use TablebaseClient with context manager.

    Args:
        board: Current position (before the move).
        move: The move to check.

    Returns:
        TablebaseMoveCheck with move quality analysis.
    """
    client = TablebaseClient()
    return client.check_move(board, move)


def analyze_endgame_accuracy(
    boards: list[chess.Board],
    moves: list[chess.Move],
    player_color: chess.Color,
) -> dict:
    """
    Analyze a sequence of endgame moves for tablebase accuracy.

    Args:
        boards: List of board positions (before each move).
        moves: List of moves made.
        player_color: The player whose moves to analyze.

    Returns:
        Dictionary with:
        - 'total_positions': Number of tablebase positions
        - 'player_moves': Number of moves by player in TB positions
        - 'optimal_moves': Number of optimal moves by player
        - 'accuracy': Percentage of optimal moves
        - 'mistakes': List of positions where player deviated from optimal
    """
    if len(boards) != len(moves):
        raise ValueError("boards and moves must have same length")

    results = {
        "total_positions": 0,
        "player_moves": 0,
        "optimal_moves": 0,
        "mistakes": [],
    }

    with TablebaseClient() as client:
        for i, (board, move) in enumerate(zip(boards, moves)):
            # Skip if not tablebase position
            if not is_tablebase_position(board):
                continue

            results["total_positions"] += 1

            # Check if this is the player's move
            if board.turn != player_color:
                continue

            results["player_moves"] += 1

            # Check the move
            check = client.check_move(board, move)

            if check.is_best_move:
                results["optimal_moves"] += 1
            else:
                results["mistakes"].append({
                    "ply": i,
                    "fen": board.fen(),
                    "move_played": move.uci(),
                    "best_move": client.probe(board.fen()).best_move if client.probe(board.fen()) else None,
                    "move_dtz": check.move_dtz,
                    "best_dtz": check.best_dtz,
                    "category": check.tablebase_category,
                })

    # Calculate accuracy
    if results["player_moves"] > 0:
        results["accuracy"] = round(
            100 * results["optimal_moves"] / results["player_moves"], 2
        )
    else:
        results["accuracy"] = 100.0  # No tablebase positions, perfect by default

    return results


@dataclass
class TablebaseConsistencyReport:
    """Summary of tablebase consistency across multiple games."""
    games_with_endgames: int  # Games reaching tablebase territory
    total_tablebase_positions: int  # Total positions with ≤7 pieces
    total_player_moves: int  # Moves by player in TB positions
    optimal_moves: int  # Moves that were optimal
    suboptimal_moves: int  # Moves that weren't optimal
    accuracy: float  # % of optimal moves
    perfect_endgame_games: int  # Games with 100% TB accuracy and 5+ TB moves
    games_with_mistakes: int  # Games with at least one TB mistake
    all_mistakes: list[dict]  # All individual mistakes


def analyze_tablebase_consistency(
    games_data: list[dict],
    min_tb_moves_for_perfect: int = 5,
) -> TablebaseConsistencyReport:
    """
    Analyze tablebase consistency across multiple games.

    Args:
        games_data: List of game data dicts, each containing:
            - 'boards': List of chess.Board positions
            - 'moves': List of chess.Move objects
            - 'player_color': chess.WHITE or chess.BLACK
            - 'game_id' or 'url': Identifier for the game
        min_tb_moves_for_perfect: Minimum TB moves to count as "perfect endgame"

    Returns:
        TablebaseConsistencyReport with aggregate statistics.
    """
    total_positions = 0
    total_player_moves = 0
    optimal_moves = 0
    all_mistakes = []
    games_with_endgames = 0
    perfect_endgame_games = 0
    games_with_mistakes = 0

    for game in games_data:
        boards = game.get('boards', [])
        moves = game.get('moves', [])
        player_color = game.get('player_color')
        game_id = game.get('game_id', game.get('url', 'unknown'))

        if not boards or not moves or len(boards) != len(moves):
            continue

        try:
            result = analyze_endgame_accuracy(boards, moves, player_color)
        except Exception:
            continue

        if result['total_positions'] == 0:
            continue

        games_with_endgames += 1
        total_positions += result['total_positions']
        total_player_moves += result['player_moves']
        optimal_moves += result['optimal_moves']

        # Track mistakes with game context
        for mistake in result['mistakes']:
            mistake['game_id'] = game_id
            all_mistakes.append(mistake)

        if result['mistakes']:
            games_with_mistakes += 1
        elif result['player_moves'] >= min_tb_moves_for_perfect:
            perfect_endgame_games += 1

    suboptimal_moves = total_player_moves - optimal_moves
    accuracy = (100 * optimal_moves / total_player_moves) if total_player_moves > 0 else 100.0

    return TablebaseConsistencyReport(
        games_with_endgames=games_with_endgames,
        total_tablebase_positions=total_positions,
        total_player_moves=total_player_moves,
        optimal_moves=optimal_moves,
        suboptimal_moves=suboptimal_moves,
        accuracy=round(accuracy, 2),
        perfect_endgame_games=perfect_endgame_games,
        games_with_mistakes=games_with_mistakes,
        all_mistakes=all_mistakes,
    )
