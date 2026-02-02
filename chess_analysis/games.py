"""
Game fetching and parsing utilities.
"""

import os

import chess.pgn
from pathlib import Path
from typing import Optional, Iterator
from datetime import datetime
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Build User-Agent from environment variables
# Format: ProjectName/Version (username: your_username; contact: your_email)
_project = os.getenv("CHESSCOM_PROJECT_NAME", "chess-notebooks")
_version = os.getenv("CHESSCOM_PROJECT_VERSION", "0.1")
_username = os.getenv("CHESSCOM_USERNAME", "")
_contact = os.getenv("CHESSCOM_CONTACT_EMAIL", "")

if _username and _contact:
    USER_AGENT = f"{_project}/{_version} (username: {_username}; contact: {_contact})"
else:
    USER_AGENT = f"{_project}/{_version}"

HEADERS = {"User-Agent": USER_AGENT}


def fetch_archives(username: str) -> list[str]:
    """
    Fetch the list of monthly game archives for a player.

    Args:
        username: Chess.com username.

    Returns:
        List of archive URLs (e.g., ["https://api.chess.com/pub/player/username/games/2024/01", ...])
    """
    archives_url = f"https://api.chess.com/pub/player/{username}/games/archives"
    response = requests.get(archives_url, headers=HEADERS)
    response.raise_for_status()
    return response.json().get("archives", [])


def get_last_game_date(username: str) -> Optional[datetime]:
    """
    Get the date of the player's most recent game.

    Useful for banned accounts to determine the date range for analysis.

    Args:
        username: Chess.com username.

    Returns:
        datetime of the last game, or None if no games found.
    """
    # Get archives list
    archives = fetch_archives(username)
    if not archives:
        return None

    # Fetch the most recent archive (last in list)
    response = requests.get(archives[-1], headers=HEADERS)
    response.raise_for_status()
    games = response.json().get("games", [])

    if not games:
        return None

    # Find the most recent game by end_time
    last_game = max(games, key=lambda g: g.get("end_time", 0))
    end_time = last_game.get("end_time", 0)

    if end_time:
        return datetime.fromtimestamp(end_time)
    return None


def fetch_games(
    username: str,
    year: Optional[int] = None,
    month: Optional[int] = None,
) -> list[dict]:
    """
    Fetch games from chess.com API for a given user.

    Args:
        username: Chess.com username.
        year: Optional year filter (e.g., 2024).
        month: Optional month filter (1-12). Requires year.

    Returns:
        List of game dictionaries from the API.
    """
    if year and month:
        # Fetch specific month
        url = f"https://api.chess.com/pub/player/{username}/games/{year}/{month:02d}"
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        return response.json().get("games", [])

    # Fetch archives list using fetch_archives function
    archives = fetch_archives(username)

    # Filter archives if year specified
    if year:
        archives = [a for a in archives if f"/{year}/" in a]

    # Fetch all matching archives
    all_games = []
    for archive_url in archives:
        response = requests.get(archive_url, headers=HEADERS)
        response.raise_for_status()
        all_games.extend(response.json().get("games", []))

    return all_games


def parse_pgn_file(filepath: str | Path) -> Iterator[chess.pgn.Game]:
    """
    Parse a PGN file and yield games.

    Args:
        filepath: Path to the PGN file.

    Yields:
        chess.pgn.Game objects.
    """
    filepath = Path(filepath)
    with open(filepath) as pgn_file:
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            yield game


def parse_pgn_string(pgn_text: str) -> Iterator[chess.pgn.Game]:
    """
    Parse PGN text and yield games.

    Args:
        pgn_text: PGN formatted string.

    Yields:
        chess.pgn.Game objects.
    """
    import io
    pgn_io = io.StringIO(pgn_text)
    while True:
        game = chess.pgn.read_game(pgn_io)
        if game is None:
            break
        yield game


def save_games_to_pgn(games: list[dict], filepath: str | Path) -> None:
    """
    Save chess.com API game data to a PGN file.

    Args:
        games: List of game dictionaries from chess.com API.
        filepath: Output PGN file path.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w") as f:
        for game in games:
            if "pgn" in game:
                f.write(game["pgn"])
                f.write("\n\n")


def find_last_book_move(
    game: "chess.pgn.Game",
    min_games: int = 1,
    database: str = "masters",
) -> int:
    """
    Find the last move that appears in the Lichess opening database.

    Uses the Lichess Masters database API to determine where the game
    leaves "book" (known opening theory).

    API docs: https://lichess.org/api#tag/opening-explorer

    Args:
        game: A parsed PGN game.
        min_games: Minimum number of games in database to consider it "book".
                   Default 1 means any position that has been played counts.
        database: Which database to use - "masters" or "lichess".

    Returns:
        The ply number (half-move) of the last book move.
        Returns 0 if the starting position is not in book.
    """
    import chess
    import time
    import urllib.parse

    base_url = f"https://explorer.lichess.ovh/{database}"
    headers = {
        "User-Agent": "ChessAnalysis/1.0 (github.com/chess-notebooks)",
        "Accept": "application/json",
    }

    board = game.board()
    moves_uci = []
    last_book_ply = 0

    for ply, move in enumerate(game.mainline_moves(), start=1):
        moves_uci.append(move.uci())

        # Build the API request
        # Use starting FEN and play parameter for moves
        start_fen = chess.STARTING_FEN
        play_param = ",".join(moves_uci)

        params = {
            "fen": start_fen,
            "play": play_param,
            "moves": 1,  # We only need to know if position exists
        }

        url = f"{base_url}?{urllib.parse.urlencode(params)}"

        try:
            response = requests.get(url, headers=headers, timeout=10)

            if response.status_code == 429:
                # Rate limited - wait and retry once
                time.sleep(1)
                response = requests.get(url, headers=headers, timeout=10)

            if response.status_code != 200:
                # API error - assume we've left book
                break

            data = response.json()

            # Check if this position has enough games
            total_games = data.get("white", 0) + data.get("draws", 0) + data.get("black", 0)

            if total_games >= min_games:
                last_book_ply = ply
            else:
                # Position not in book - stop searching
                break

        except (requests.RequestException, ValueError):
            # Network error or JSON parse error - assume we've left book
            break

        # Small delay to be respectful to the API
        time.sleep(0.1)

        # Safety limit - don't check beyond move 30
        if ply >= 60:
            break

        board.push(move)

    return last_book_ply


def find_last_book_move_fast(
    game: "chess.pgn.Game",
    min_games: int = 1,
    database: str = "masters",
) -> dict:
    """
    Find the last book move using binary search for efficiency.

    Makes fewer API calls by using binary search instead of checking
    every position sequentially.

    Args:
        game: A parsed PGN game.
        min_games: Minimum number of games in database to consider it "book".
        database: Which database to use - "masters" or "lichess".

    Returns:
        Dictionary with:
        - last_book_ply: The ply number (half-move) of the last book move
        - opening_name: Name of the opening (if detected)
        - opening_eco: ECO code (if detected)
        - book_moves: List of moves that were in book
        - book_depth: Number of full moves in book (last_book_ply // 2)
        - total_book_games: Number of master games with this position
    """
    import chess
    import time
    import urllib.parse

    base_url = f"https://explorer.lichess.ovh/{database}"
    headers = {
        "User-Agent": "ChessAnalysis/1.0 (github.com/chess-notebooks)",
        "Accept": "application/json",
    }

    moves_list = list(game.mainline_moves())
    total_moves = len(moves_list)

    # Default result for empty games
    empty_result = {
        'last_book_ply': 0,
        'opening_name': None,
        'opening_eco': None,
        'book_moves': [],
        'book_depth': 0,
        'total_book_games': 0,
        'white_wins': 0,
        'draws': 0,
        'black_wins': 0,
    }

    if total_moves == 0:
        return empty_result

    # Store API response data for later use
    last_book_data = None

    def check_book_position(ply: int) -> tuple[bool, dict]:
        """Check if position at given ply is in book. Returns (is_in_book, api_data)."""
        if ply == 0:
            return True, {}

        moves_uci = [moves_list[i].uci() for i in range(ply)]
        play_param = ",".join(moves_uci)

        params = {
            "fen": chess.STARTING_FEN,
            "play": play_param,
            "moves": 1,
        }

        url = f"{base_url}?{urllib.parse.urlencode(params)}"

        try:
            response = requests.get(url, headers=headers, timeout=10)

            if response.status_code == 429:
                time.sleep(1)
                response = requests.get(url, headers=headers, timeout=10)

            if response.status_code != 200:
                return False, {}

            data = response.json()
            total_games = data.get("white", 0) + data.get("draws", 0) + data.get("black", 0)
            return total_games >= min_games, data

        except (requests.RequestException, ValueError):
            return False, {}

    # Binary search for last book move
    # Cap at move 40 (ply 80) for efficiency
    max_ply = min(total_moves, 80)

    # First check if opening move is in book
    in_book, data = check_book_position(1)
    if not in_book:
        return empty_result

    last_book_data = data
    time.sleep(0.1)

    # Binary search
    low, high = 1, max_ply
    last_known_book = 1

    while low <= high:
        mid = (low + high) // 2

        time.sleep(0.1)  # Rate limiting

        in_book, data = check_book_position(mid)
        if in_book:
            last_known_book = mid
            last_book_data = data
            low = mid + 1
        else:
            high = mid - 1

    # Build the book moves list (SAN notation)
    board = chess.Board()
    book_moves_san = []
    for i in range(last_known_book):
        move = moves_list[i]
        book_moves_san.append(board.san(move))
        board.push(move)

    # Extract opening info from last book position
    opening_name = None
    opening_eco = None
    if last_book_data:
        opening_info = last_book_data.get("opening")
        if opening_info:
            opening_name = opening_info.get("name")
            opening_eco = opening_info.get("eco")

    return {
        'last_book_ply': last_known_book,
        'opening_name': opening_name,
        'opening_eco': opening_eco,
        'book_moves': book_moves_san,
        'book_depth': (last_known_book + 1) // 2,  # Full moves
        'total_book_games': last_book_data.get("white", 0) + last_book_data.get("draws", 0) + last_book_data.get("black", 0) if last_book_data else 0,
        'white_wins': last_book_data.get("white", 0) if last_book_data else 0,
        'draws': last_book_data.get("draws", 0) if last_book_data else 0,
        'black_wins': last_book_data.get("black", 0) if last_book_data else 0,
    }
