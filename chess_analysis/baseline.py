"""
Baseline generation utilities for chess fairness analysis.

This module provides functions for:
- Generating player baselines from Chess.com games
- Opponent profile fetching and validation
- Frequent opponent analysis
- Timeout pattern analysis
- Opening analysis by ECO code
- Combined baseline aggregation
"""

import json
import os
import sqlite3
import statistics
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv

from .caching import (
    OpponentProfile,
    get_cached_opponent_profile,
    save_opponent_profile_to_cache,
    load_cached_games_v2,
    analyze_cache_state,
    build_fetch_plan,
    merge_games,
    save_cache_with_metadata,
    query_cache_filtered,
    extract_opponents_from_games,
)
from .games import fetch_games, fetch_archives, get_last_game_date
from .dataset import (
    build_game_dataset,
    build_position_dataset,
    build_opening_book,
    aggregate_all_games,
    analyze_elo_patterns,
    analyze_result_patterns,
    analyze_termination_patterns,
    save_dataset_parquet,
    save_opening_book,
    segment_games_by_elo_range,
)


# =============================================================================
# API Configuration
# =============================================================================

# Load environment variables from .env file
load_dotenv()

API_BASE = "https://api.chess.com/pub"

# Build User-Agent from environment variables
_project = os.getenv("CHESSCOM_PROJECT_NAME", "chess-notebooks")
_version = os.getenv("CHESSCOM_PROJECT_VERSION", "0.1")
_username = os.getenv("CHESSCOM_USERNAME", "")
_contact = os.getenv("CHESSCOM_CONTACT_EMAIL", "")

if _username and _contact:
    USER_AGENT = f"{_project}/{_version} (username: {_username}; contact: {_contact})"
else:
    USER_AGENT = f"{_project}/{_version}"

HEADERS = {"User-Agent": USER_AGENT}
REQUEST_DELAY = 0.3  # 300ms between requests


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class FrequentOpponent:
    """Statistics for a frequently-played opponent."""
    username: str
    games_played: int
    std_devs_above_mean: float
    current_elos: dict = field(default_factory=dict)
    account_status: str = "unknown"
    is_trustworthy: bool = True
    wins: int = 0
    losses: int = 0
    draws: int = 0


@dataclass
class GameSession:
    """
    A session of games played consecutively.

    Sessions are useful for fairness analysis to detect:
    - Consistency patterns within a session vs across sessions
    - Potential account sharing (different play styles between sessions)
    - Performance variations that might indicate external assistance
    """
    session_id: int
    start_time: int  # Unix timestamp of first game
    end_time: int  # Unix timestamp of last game
    duration_minutes: float  # Total session duration
    game_count: int
    game_ids: list = field(default_factory=list)

    # Session statistics
    time_classes: list = field(default_factory=list)  # Time classes played
    is_tournament: bool = False  # If all games are from same tournament
    tournament_url: Optional[str] = None

    # Performance in session
    wins: int = 0
    losses: int = 0
    draws: int = 0
    elo_start: Optional[int] = None  # Elo at session start
    elo_end: Optional[int] = None  # Elo at session end
    elo_change: int = 0

    # Aggregate stats (can be populated later with engine analysis)
    avg_accuracy: Optional[float] = None
    avg_time_per_move: Optional[float] = None


# =============================================================================
# Session Detection
# =============================================================================

DEFAULT_SESSION_GAP_MINUTES = 15  # Gap threshold between sessions


def detect_sessions(
    games: list[dict],
    username: str,
    gap_minutes: int = DEFAULT_SESSION_GAP_MINUTES,
) -> list[GameSession]:
    """
    Detect playing sessions from a list of games.

    A session is a series of consecutive games where the gap between
    any two adjacent games is less than gap_minutes.

    Args:
        games: List of game dicts (must have 'end_time' field)
        username: Player's username (to determine win/loss/draw)
        gap_minutes: Maximum gap between games in same session (default 15)

    Returns:
        List of GameSession objects, sorted by start_time descending (newest first)
    """
    if not games:
        return []

    # Sort games by end_time ascending (oldest first) for session building
    sorted_games = sorted(games, key=lambda g: g.get('end_time', 0))

    gap_seconds = gap_minutes * 60
    sessions = []
    current_session_games = []
    username_lower = username.lower()

    for game in sorted_games:
        end_time = game.get('end_time', 0)
        if end_time == 0:
            continue

        if not current_session_games:
            # Start new session
            current_session_games.append(game)
        else:
            # Check gap from previous game
            prev_end = current_session_games[-1].get('end_time', 0)
            gap = end_time - prev_end

            if gap <= gap_seconds:
                # Same session
                current_session_games.append(game)
            else:
                # New session - finalize current one
                session = _build_session(
                    current_session_games,
                    len(sessions),
                    username_lower
                )
                sessions.append(session)
                current_session_games = [game]

    # Don't forget the last session
    if current_session_games:
        session = _build_session(
            current_session_games,
            len(sessions),
            username_lower
        )
        sessions.append(session)

    # Return newest first
    sessions.reverse()
    return sessions


def _build_session(
    games: list[dict],
    session_id: int,
    username_lower: str
) -> GameSession:
    """Build a GameSession object from a list of games."""
    start_time = games[0].get('end_time', 0)
    end_time = games[-1].get('end_time', 0)
    duration_minutes = (end_time - start_time) / 60.0

    # Extract game IDs
    game_ids = []
    for g in games:
        url = g.get('url', '')
        game_id = url.split('/')[-1] if url else ''
        if game_id:
            game_ids.append(game_id)

    # Get time classes
    time_classes = list(set(g.get('time_class', 'unknown') for g in games))

    # Check for tournament (all games have same tournament URL)
    tournament_urls = set()
    for g in games:
        if 'tournament' in g:
            tournament_urls.add(g.get('tournament'))
    is_tournament = len(tournament_urls) == 1 and None not in tournament_urls
    tournament_url = list(tournament_urls)[0] if is_tournament else None

    # Calculate W/L/D and Elo changes
    wins, losses, draws = 0, 0, 0
    elos = []

    for g in games:
        white = g.get('white', {})
        black = g.get('black', {})

        white_username = white.get('username', '').lower() if isinstance(white, dict) else ''
        black_username = black.get('username', '').lower() if isinstance(black, dict) else ''

        if white_username == username_lower:
            player_data = white
            result = white.get('result', '')
        elif black_username == username_lower:
            player_data = black
            result = black.get('result', '')
        else:
            continue

        # Track Elo
        rating = player_data.get('rating')
        if rating:
            elos.append(rating)

        # Count results
        if result == 'win':
            wins += 1
        elif result in ('checkmated', 'timeout', 'resigned', 'lose', 'abandoned'):
            losses += 1
        elif result in ('agreed', 'stalemate', 'repetition', 'insufficient',
                       'timevsinsufficient', '50move', 'draw'):
            draws += 1

    elo_start = elos[0] if elos else None
    elo_end = elos[-1] if elos else None
    elo_change = (elo_end - elo_start) if (elo_start and elo_end) else 0

    return GameSession(
        session_id=session_id,
        start_time=start_time,
        end_time=end_time,
        duration_minutes=duration_minutes,
        game_count=len(games),
        game_ids=game_ids,
        time_classes=time_classes,
        is_tournament=is_tournament,
        tournament_url=tournament_url,
        wins=wins,
        losses=losses,
        draws=draws,
        elo_start=elo_start,
        elo_end=elo_end,
        elo_change=elo_change,
    )


def assign_session_ids_to_games(
    games: list[dict],
    sessions: list[GameSession],
) -> list[dict]:
    """
    Add session_id field to each game dict.

    Args:
        games: List of game dicts
        sessions: List of GameSession objects

    Returns:
        Games with 'session_id' field added
    """
    # Build game_id -> session_id mapping
    game_to_session = {}
    for session in sessions:
        for game_id in session.game_ids:
            game_to_session[game_id] = session.session_id

    # Add session_id to games
    for game in games:
        url = game.get('url', '')
        game_id = url.split('/')[-1] if url else ''
        game['session_id'] = game_to_session.get(game_id)

    return games


def analyze_session_patterns(sessions: list[GameSession]) -> dict:
    """
    Analyze patterns across sessions for fairness indicators.

    Returns statistics that could indicate account sharing or external assistance.
    """
    if not sessions:
        return {}

    # Session size distribution
    game_counts = [s.game_count for s in sessions]
    durations = [s.duration_minutes for s in sessions]

    # Elo volatility across sessions
    elo_changes = [s.elo_change for s in sessions if s.elo_change != 0]

    # Win rates per session
    session_win_rates = []
    for s in sessions:
        total = s.wins + s.losses + s.draws
        if total > 0:
            session_win_rates.append(s.wins / total)

    return {
        "total_sessions": len(sessions),
        "total_games_in_sessions": sum(game_counts),
        "avg_session_length": statistics.mean(game_counts) if game_counts else 0,
        "max_session_length": max(game_counts) if game_counts else 0,
        "min_session_length": min(game_counts) if game_counts else 0,
        "avg_session_duration_minutes": statistics.mean(durations) if durations else 0,
        "max_session_duration_minutes": max(durations) if durations else 0,
        "tournament_sessions": sum(1 for s in sessions if s.is_tournament),
        # Elo analysis
        "avg_elo_change_per_session": statistics.mean(elo_changes) if elo_changes else 0,
        "elo_change_std": statistics.stdev(elo_changes) if len(elo_changes) > 1 else 0,
        "max_elo_gain_session": max(elo_changes) if elo_changes else 0,
        "max_elo_loss_session": min(elo_changes) if elo_changes else 0,
        # Win rate consistency
        "session_win_rate_std": statistics.stdev(session_win_rates) if len(session_win_rates) > 1 else 0,
        "session_win_rate_avg": statistics.mean(session_win_rates) if session_win_rates else 0,
    }


# =============================================================================
# Chess.com API Functions
# =============================================================================

def api_get(url: str, retries: int = 3) -> Optional[dict]:
    """Make a GET request to the chess.com API with retries and rate limiting."""
    for attempt in range(retries):
        try:
            time.sleep(REQUEST_DELAY)
            response = requests.get(url, headers=HEADERS, timeout=60)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                return None
            elif response.status_code == 429:
                wait_time = 10 * (attempt + 1)
                print(f"  Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"  HTTP {response.status_code} for {url}")
                if attempt < retries - 1:
                    time.sleep(2)
        except requests.exceptions.Timeout:
            print(f"  Timeout for {url} (attempt {attempt + 1})")
            time.sleep(5)
        except Exception as e:
            print(f"  Request error: {e}")
            time.sleep(2)
    return None


def is_banned_account(status: str) -> bool:
    """Check if account status indicates fair play violation or closed account."""
    status_lower = status.lower()
    return (
        "fair_play" in status_lower or
        "closed" in status_lower or
        "violation" in status_lower
    )


def fetch_player_profile(username: str) -> Optional[OpponentProfile]:
    """Fetch player profile from chess.com API."""
    profile_data = api_get(f"{API_BASE}/player/{username}")
    if not profile_data:
        return None

    status = profile_data.get("status", "unknown")
    joined = profile_data.get("joined")  # Unix timestamp of account creation
    last_online = profile_data.get("last_online")  # Unix timestamp (= ban date for closed accounts)
    title = profile_data.get("title")  # Chess title (GM, IM, FM, etc.)
    player_id = profile_data.get("player_id")  # Chess.com player ID

    stats_data = api_get(f"{API_BASE}/player/{username}/stats")

    rating_blitz = None
    rating_rapid = None
    rating_bullet = None
    blitz_rd = None
    rapid_rd = None
    total_games_blitz = None
    total_games_rapid = None

    if stats_data:
        # Blitz stats
        if "chess_blitz" in stats_data:
            blitz_data = stats_data["chess_blitz"]
            rating_blitz = blitz_data.get("last", {}).get("rating")
            blitz_rd = blitz_data.get("last", {}).get("rd")
            # Calculate total games from record
            record = blitz_data.get("record", {})
            total_games_blitz = (
                record.get("win", 0) + record.get("loss", 0) + record.get("draw", 0)
            )

        # Rapid stats
        if "chess_rapid" in stats_data:
            rapid_data = stats_data["chess_rapid"]
            rating_rapid = rapid_data.get("last", {}).get("rating")
            rapid_rd = rapid_data.get("last", {}).get("rd")
            # Calculate total games from record
            record = rapid_data.get("record", {})
            total_games_rapid = (
                record.get("win", 0) + record.get("loss", 0) + record.get("draw", 0)
            )

        # Bullet stats
        if "chess_bullet" in stats_data:
            rating_bullet = stats_data["chess_bullet"].get("last", {}).get("rating")

    return OpponentProfile(
        username=username.lower(),
        status=status,
        is_banned=is_banned_account(status),
        rating_blitz=rating_blitz,
        rating_rapid=rating_rapid,
        rating_bullet=rating_bullet,
        fetched_at=datetime.now().isoformat(),
        joined=joined,
        blitz_rd=blitz_rd,
        rapid_rd=rapid_rd,
        total_games_blitz=total_games_blitz,
        total_games_rapid=total_games_rapid,
        last_online=last_online,
        title=title,
        player_id=player_id,
    )


def batch_fetch_opponent_profiles(
    usernames: list[str],
    cache_conn: Optional[sqlite3.Connection] = None,
    progress_interval: int = 50,
) -> dict[str, OpponentProfile]:
    """
    Fetch profiles for multiple opponents with caching and rate limiting.

    Args:
        usernames: List of opponent usernames to fetch
        cache_conn: SQLite connection for caching (optional)
        progress_interval: How often to print progress

    Returns:
        Dictionary mapping username to OpponentProfile
    """
    profiles = {}
    total = len(usernames)
    fetched_count = 0
    cached_count = 0

    for i, username in enumerate(usernames):
        username_lower = username.lower()

        if cache_conn:
            cached = get_cached_opponent_profile(cache_conn, username_lower)
            if cached:
                profiles[username_lower] = cached
                cached_count += 1
                continue

        profile = fetch_player_profile(username_lower)
        if profile:
            profiles[username_lower] = profile
            fetched_count += 1

            if cache_conn:
                save_opponent_profile_to_cache(cache_conn, profile)

        if (i + 1) % progress_interval == 0:
            print(f"  Progress: {i + 1}/{total} opponents checked ({cached_count} cached, {fetched_count} fetched)")

    print(f"  Completed: {cached_count} from cache, {fetched_count} fetched from API")
    return profiles


# =============================================================================
# Game Splitting and Opponent Analysis Functions
# =============================================================================

def extract_opponent_from_game(game: dict, player_username: str) -> Optional[str]:
    """Extract opponent username from a game dict."""
    player_lower = player_username.lower()
    white = game.get("white", {})
    black = game.get("black", {})

    white_username = white.get("username", "").lower() if isinstance(white, dict) else ""
    black_username = black.get("username", "").lower() if isinstance(black, dict) else ""

    if white_username == player_lower:
        return black_username
    elif black_username == player_lower:
        return white_username
    return None


def split_games_by_opponent_trust(
    games: list[dict],
    player_username: str,
    opponent_profiles: dict[str, OpponentProfile],
) -> tuple[list[dict], list[dict]]:
    """
    Split games into trustworthy and untrustworthy based on opponent status.

    Returns:
        Tuple of (trustworthy_games, untrustworthy_games)
    """
    trustworthy = []
    untrustworthy = []

    for game in games:
        opponent = extract_opponent_from_game(game, player_username)
        if not opponent:
            trustworthy.append(game)
            continue

        profile = opponent_profiles.get(opponent.lower())
        if profile and profile.is_banned:
            untrustworthy.append(game)
        else:
            trustworthy.append(game)

    return trustworthy, untrustworthy


def analyze_frequent_opponents(
    games: list[dict],
    player_username: str,
    opponent_profiles: dict[str, OpponentProfile],
    std_threshold: float = 2.0,
) -> dict:
    """
    Identify frequent opponents using statistical threshold.

    Uses +N standard deviations above mean games-per-opponent as threshold.

    Returns:
        Dictionary with distribution_stats, all_opponents, frequent_opponents,
        trustable_frequent_opponents, banned_opponents_encountered
    """
    opponent_stats = {}

    for game in games:
        opponent = extract_opponent_from_game(game, player_username)
        if not opponent:
            continue

        opponent_lower = opponent.lower()
        if opponent_lower not in opponent_stats:
            opponent_stats[opponent_lower] = {"games": 0, "wins": 0, "losses": 0, "draws": 0}

        opponent_stats[opponent_lower]["games"] += 1

        white = game.get("white", {})
        black = game.get("black", {})
        player_lower = player_username.lower()

        if isinstance(white, dict) and white.get("username", "").lower() == player_lower:
            player_result = white.get("result", "")
        elif isinstance(black, dict):
            player_result = black.get("result", "")
        else:
            player_result = ""

        if player_result == "win":
            opponent_stats[opponent_lower]["wins"] += 1
        elif player_result in ("checkmated", "timeout", "resigned", "lose", "abandoned"):
            opponent_stats[opponent_lower]["losses"] += 1
        else:
            opponent_stats[opponent_lower]["draws"] += 1

    if not opponent_stats:
        return {
            "distribution_stats": {
                "total_unique_opponents": 0,
                "mean_games_per_opponent": 0,
                "std_dev": 0,
                "threshold_value": 0,
                "frequent_count": 0,
            },
            "all_opponents": [],
            "frequent_opponents": [],
            "trustable_frequent_opponents": [],
            "banned_opponents_encountered": [],
        }

    games_counts = [s["games"] for s in opponent_stats.values()]
    mean_games = statistics.mean(games_counts)
    std_dev = statistics.stdev(games_counts) if len(games_counts) > 1 else 0
    threshold_value = mean_games + (std_threshold * std_dev)

    all_opponents = []
    frequent_opponents = []
    trustable_frequent = []
    banned_encountered = []

    for username, stats in opponent_stats.items():
        profile = opponent_profiles.get(username)

        current_elos = {}
        if profile:
            if profile.rating_blitz:
                current_elos["blitz"] = profile.rating_blitz
            if profile.rating_rapid:
                current_elos["rapid"] = profile.rating_rapid
            if profile.rating_bullet:
                current_elos["bullet"] = profile.rating_bullet

        is_banned = profile.is_banned if profile else False
        account_status = profile.status if profile else "unknown"
        # Account is trustworthy only if not banned AND not closed for any reason
        is_closed = account_status.startswith("closed") if account_status else False
        is_trustworthy = not is_banned and not is_closed

        if is_banned:
            banned_encountered.append(username)

        std_devs_above = (stats["games"] - mean_games) / std_dev if std_dev > 0 else 0

        opponent_data = FrequentOpponent(
            username=username,
            games_played=stats["games"],
            std_devs_above_mean=round(std_devs_above, 2),
            current_elos=current_elos,
            account_status=account_status,
            is_trustworthy=is_trustworthy,
            wins=stats["wins"],
            losses=stats["losses"],
            draws=stats["draws"],
        )

        all_opponents.append(opponent_data)

        if stats["games"] >= threshold_value:
            frequent_opponents.append(opponent_data)
            if is_trustworthy:
                trustable_frequent.append(username)

    all_opponents.sort(key=lambda x: x.games_played, reverse=True)
    frequent_opponents.sort(key=lambda x: x.games_played, reverse=True)

    return {
        "distribution_stats": {
            "total_unique_opponents": len(opponent_stats),
            "mean_games_per_opponent": round(mean_games, 2),
            "std_dev": round(std_dev, 2),
            "threshold_value": round(threshold_value, 2),
            "frequent_count": len(frequent_opponents),
        },
        "all_opponents": all_opponents,
        "frequent_opponents": frequent_opponents,
        "trustable_frequent_opponents": trustable_frequent,
        "banned_opponents_encountered": banned_encountered,
    }


# =============================================================================
# Analysis Functions
# =============================================================================

def analyze_timeout_patterns(game_dataset: list[dict], game_aggregates: list[dict]) -> dict:
    """
    Analyze timeout wins while behind on material.

    Returns:
        Dictionary with timeout analysis metrics
    """
    material_by_game = {}
    for agg in game_aggregates:
        game_id = agg.get('game_id')
        if game_id:
            material_by_game[game_id] = agg.get('material_final', 0)

    timeout_wins = [g for g in game_dataset if g.get('resolution') == 'win_timeout']
    total_wins = sum(1 for g in game_dataset if g.get('player_result') == 'win')
    total_timeout_wins = len(timeout_wins)

    if total_timeout_wins == 0:
        return {
            'total_wins': total_wins,
            'timeout_wins': 0,
            'timeout_win_rate': 0.0,
            'timeout_wins_behind': 0,
            'timeout_wins_way_behind': 0,
            'behind_rate': 0.0,
            'way_behind_rate': 0.0,
        }

    timeout_wins_behind = 0
    timeout_wins_way_behind = 0

    for game in timeout_wins:
        game_id = game.get('game_id')
        material = material_by_game.get(game_id, 0)

        if material < -2:
            timeout_wins_behind += 1
        if material < -5:
            timeout_wins_way_behind += 1

    behind_rate = timeout_wins_behind / total_timeout_wins if total_timeout_wins > 0 else 0
    way_behind_rate = timeout_wins_way_behind / total_timeout_wins if total_timeout_wins > 0 else 0
    timeout_win_rate = total_timeout_wins / total_wins if total_wins > 0 else 0

    return {
        'total_wins': total_wins,
        'timeout_wins': total_timeout_wins,
        'timeout_win_rate': timeout_win_rate,
        'timeout_wins_behind': timeout_wins_behind,
        'timeout_wins_way_behind': timeout_wins_way_behind,
        'behind_rate': behind_rate,
        'way_behind_rate': way_behind_rate,
    }


def analyze_banned_opponent_games(
    trustworthy_games: list[dict],
    untrustworthy_games: list[dict],
    game_aggregates: list[dict],
    banned_opponents: list[str],
) -> dict:
    """
    Analyze games played against banned opponents to detect potential engine vs engine patterns.

    Args:
        trustworthy_games: Games against opponents in good standing
        untrustworthy_games: Games against banned opponents
        game_aggregates: Aggregated game data with ACPL and accuracy
        banned_opponents: List of banned opponent usernames

    Returns:
        Dictionary with banned opponent analysis metrics
    """
    if not untrustworthy_games:
        return {
            'games_vs_banned': 0,
            'games_vs_trustworthy': len(trustworthy_games),
            'banned_percentage': 0.0,
            'banned_opponent_count': len(banned_opponents),
            'has_banned_games': False,
        }

    # Build game_id -> aggregate mapping
    agg_by_game = {}
    for agg in game_aggregates:
        game_id = agg.get('game_id')
        if game_id:
            agg_by_game[game_id] = agg

    # Calculate W/L/D for trustworthy games
    trust_wins = sum(1 for g in trustworthy_games if g.get('player_result') == 'win')
    trust_losses = sum(1 for g in trustworthy_games if g.get('player_result') == 'loss')
    trust_draws = len(trustworthy_games) - trust_wins - trust_losses

    # Calculate W/L/D for banned opponent games
    banned_wins = sum(1 for g in untrustworthy_games if g.get('player_result') == 'win')
    banned_losses = sum(1 for g in untrustworthy_games if g.get('player_result') == 'loss')
    banned_draws = len(untrustworthy_games) - banned_wins - banned_losses

    # Calculate ACPL for each category
    trust_acpls = []
    banned_acpls = []

    for game in trustworthy_games:
        game_id = game.get('game_id')
        agg = agg_by_game.get(game_id)
        if agg and agg.get('acpl') is not None:
            trust_acpls.append(agg['acpl'])

    for game in untrustworthy_games:
        game_id = game.get('game_id')
        agg = agg_by_game.get(game_id)
        if agg and agg.get('acpl') is not None:
            banned_acpls.append(agg['acpl'])

    # Count games by opponent
    games_per_banned_opponent = {}
    for game in untrustworthy_games:
        opponent = game.get('opponent_username', '').lower()
        if opponent:
            if opponent not in games_per_banned_opponent:
                games_per_banned_opponent[opponent] = {
                    'games': 0, 'wins': 0, 'losses': 0, 'draws': 0, 'acpls': []
                }
            games_per_banned_opponent[opponent]['games'] += 1

            result = game.get('player_result')
            if result == 'win':
                games_per_banned_opponent[opponent]['wins'] += 1
            elif result == 'loss':
                games_per_banned_opponent[opponent]['losses'] += 1
            else:
                games_per_banned_opponent[opponent]['draws'] += 1

            game_id = game.get('game_id')
            agg = agg_by_game.get(game_id)
            if agg and agg.get('acpl') is not None:
                games_per_banned_opponent[opponent]['acpls'].append(agg['acpl'])

    # Identify potential engine vs engine games (both players with very low ACPL)
    # This would require opponent ACPL which we may not have, so flag games with
    # suspiciously low player ACPL
    low_acpl_threshold = 15  # ACPL below this is engine-level
    engine_level_games = []

    for game in untrustworthy_games:
        game_id = game.get('game_id')
        agg = agg_by_game.get(game_id)
        if agg and agg.get('acpl') is not None and agg['acpl'] < low_acpl_threshold:
            engine_level_games.append({
                'game_id': game_id,
                'opponent': game.get('opponent_username'),
                'result': game.get('player_result'),
                'acpl': agg['acpl'],
                'end_time': game.get('end_time'),
            })

    # Sort banned opponents by game count
    opponent_summaries = [
        {
            'username': opp,
            'games': data['games'],
            'wins': data['wins'],
            'losses': data['losses'],
            'draws': data['draws'],
            'win_rate': data['wins'] / data['games'] if data['games'] > 0 else 0,
            'avg_acpl': statistics.mean(data['acpls']) if data['acpls'] else None,
        }
        for opp, data in games_per_banned_opponent.items()
    ]
    opponent_summaries.sort(key=lambda x: x['games'], reverse=True)
    top_banned_opponents = opponent_summaries[:10]

    total_games = len(trustworthy_games) + len(untrustworthy_games)

    return {
        'games_vs_banned': len(untrustworthy_games),
        'games_vs_trustworthy': len(trustworthy_games),
        'banned_percentage': len(untrustworthy_games) / total_games * 100 if total_games > 0 else 0,
        'banned_opponent_count': len(banned_opponents),
        'unique_banned_opponents_played': len(games_per_banned_opponent),
        'has_banned_games': True,

        # W/L/D comparison
        'trustworthy_results': {
            'wins': trust_wins,
            'losses': trust_losses,
            'draws': trust_draws,
            'win_rate': trust_wins / len(trustworthy_games) if trustworthy_games else 0,
        },
        'banned_results': {
            'wins': banned_wins,
            'losses': banned_losses,
            'draws': banned_draws,
            'win_rate': banned_wins / len(untrustworthy_games) if untrustworthy_games else 0,
        },

        # ACPL comparison
        'trustworthy_acpl': {
            'mean': statistics.mean(trust_acpls) if trust_acpls else None,
            'median': statistics.median(trust_acpls) if trust_acpls else None,
            'std': statistics.stdev(trust_acpls) if len(trust_acpls) > 1 else None,
            'games_with_acpl': len(trust_acpls),
        },
        'banned_acpl': {
            'mean': statistics.mean(banned_acpls) if banned_acpls else None,
            'median': statistics.median(banned_acpls) if banned_acpls else None,
            'std': statistics.stdev(banned_acpls) if len(banned_acpls) > 1 else None,
            'games_with_acpl': len(banned_acpls),
        },

        # Potential engine vs engine
        'engine_level_games_count': len(engine_level_games),
        'engine_level_games': engine_level_games[:20],  # Limit to 20

        # Top banned opponents
        'top_banned_opponents': top_banned_opponents,

        # Flags
        'flags': _get_banned_analysis_flags(
            len(untrustworthy_games), total_games,
            banned_wins, len(untrustworthy_games),
            trust_acpls, banned_acpls,
            len(engine_level_games)
        ),
    }


def _get_banned_analysis_flags(
    banned_games: int,
    total_games: int,
    banned_wins: int,
    banned_total: int,
    trust_acpls: list,
    banned_acpls: list,
    engine_level_count: int,
) -> list[str]:
    """Generate flags for banned opponent analysis."""
    flags = []

    # Flag: High percentage of games vs banned opponents
    if total_games > 0:
        banned_pct = banned_games / total_games
        if banned_pct > 0.20:
            flags.append(f"High banned opponent rate: {banned_pct:.1%} of games")

    # Flag: Unusually high win rate vs banned opponents
    if banned_total >= 5:
        banned_win_rate = banned_wins / banned_total
        if banned_win_rate > 0.75:
            flags.append(f"High win rate vs banned: {banned_win_rate:.1%}")

    # Flag: Lower ACPL in banned games (playing better against cheaters?)
    if trust_acpls and banned_acpls:
        trust_mean = statistics.mean(trust_acpls)
        banned_mean = statistics.mean(banned_acpls)
        if banned_mean < trust_mean * 0.7:  # 30% lower ACPL
            flags.append(f"Lower ACPL vs banned: {banned_mean:.1f} vs {trust_mean:.1f}")

    # Flag: Many engine-level games
    if engine_level_count >= 3:
        flags.append(f"Engine-level accuracy in {engine_level_count} games vs banned")

    return flags


def limit_games_per_time_class(
    game_dataset: list[dict],
    max_games_per_class: int,
    time_classes: list[str],
) -> list[dict]:
    """
    Limit games to max_games_per_class for each time class.

    Games are sorted by end_time descending (newest first) before limiting.
    """
    if max_games_per_class is None:
        return game_dataset

    games_by_class = {tc: [] for tc in time_classes}
    for game in game_dataset:
        tc = game.get("time_class", "")
        if tc in games_by_class:
            games_by_class[tc].append(game)

    limited_games = []
    for tc in time_classes:
        tc_games = games_by_class[tc]
        tc_games.sort(key=lambda g: g.get("end_time", 0), reverse=True)
        limited = tc_games[:max_games_per_class]
        limited_games.extend(limited)
        if len(tc_games) > max_games_per_class:
            print(f"  Limited {tc}: {len(tc_games)} -> {len(limited)} games")

    limited_games.sort(key=lambda g: g.get("end_time", 0), reverse=True)
    return limited_games


def analyze_openings_by_eco(game_dataset: list[dict]) -> dict:
    """
    Analyze opening distribution by ECO code.

    Returns:
        Dictionary with ECO distribution and top openings
    """
    total_games = len(game_dataset)
    games_with_eco = 0
    eco_stats = {}
    opening_stats = {}

    for game in game_dataset:
        eco = game.get("eco", "")
        opening_name = game.get("opening_name", "Unknown")
        player_result = game.get("player_result", "")

        if eco:
            games_with_eco += 1

            if eco not in eco_stats:
                eco_stats[eco] = {"count": 0, "wins": 0, "losses": 0, "draws": 0}

            eco_stats[eco]["count"] += 1
            if player_result == "win":
                eco_stats[eco]["wins"] += 1
            elif player_result == "loss":
                eco_stats[eco]["losses"] += 1
            else:
                eco_stats[eco]["draws"] += 1

            opening_key = f"{eco}: {opening_name}"
            if opening_key not in opening_stats:
                opening_stats[opening_key] = {"count": 0, "wins": 0, "losses": 0, "draws": 0, "eco": eco, "name": opening_name}

            opening_stats[opening_key]["count"] += 1
            if player_result == "win":
                opening_stats[opening_key]["wins"] += 1
            elif player_result == "loss":
                opening_stats[opening_key]["losses"] += 1
            else:
                opening_stats[opening_key]["draws"] += 1

    # Calculate ECO category distribution (A, B, C, D, E)
    eco_categories = {}
    for eco, stats in eco_stats.items():
        if eco:
            category = eco[0].upper()
            if category not in eco_categories:
                eco_categories[category] = {"count": 0, "wins": 0, "losses": 0, "draws": 0}
            eco_categories[category]["count"] += stats["count"]
            eco_categories[category]["wins"] += stats["wins"]
            eco_categories[category]["losses"] += stats["losses"]
            eco_categories[category]["draws"] += stats["draws"]

    # Get top 10 openings
    top_openings = sorted(opening_stats.values(), key=lambda x: x["count"], reverse=True)[:10]

    # Calculate win rates
    for opening in top_openings:
        total = opening["count"]
        opening["win_rate"] = opening["wins"] / total if total > 0 else 0

    return {
        "total_games": total_games,
        "games_with_eco": games_with_eco,
        "eco_coverage": games_with_eco / total_games if total_games > 0 else 0,
        "unique_eco_codes": len(eco_stats),
        "eco_distribution": eco_stats,
        "eco_category_distribution": eco_categories,
        "top_openings": top_openings,
    }


# =============================================================================
# Baseline Generation Functions
# =============================================================================

def generate_player_baseline(
    username: str,
    output_dir: Path,
    time_classes: list[str],
    days_back: Optional[int] = None,
    max_games: int | None = None,
    use_last_game_date: bool = False,
    validate_opponents: bool = True,
    opponent_cache_conn: Optional[sqlite3.Connection] = None,
) -> dict:
    """
    Generate baseline data for a single player.

    Args:
        username: Chess.com username
        output_dir: Directory to save output files
        time_classes: List of time classes to include
        days_back: Number of days of history to fetch (None = all available)
        max_games: Maximum games per time class to process (None = all)
        use_last_game_date: If True, use player's last game date as end date
        validate_opponents: If True, fetch opponent profiles to check for bans
        opponent_cache_conn: SQLite connection for opponent caching

    Returns:
        Summary dictionary with stats
    """
    print(f"\n{'='*60}")
    print(f"Processing: {username}")
    print(f"{'='*60}")

    # Create player directory
    player_dir = output_dir / username
    player_dir.mkdir(parents=True, exist_ok=True)

    # --- PHASE 1: Analyze Cache State ---
    cache_state = analyze_cache_state(player_dir, username)
    if cache_state.total_games > 0:
        print(f"  Cache: {cache_state.total_games} games (v{cache_state.cache_version})")

    # Determine reference date
    if use_last_game_date:
        reference_date = get_last_game_date(username)
        if reference_date is None:
            print("Could not determine last game date!")
            return {"username": username, "status": "no_games"}
        print(f"  Using last game date: {reference_date.strftime('%Y-%m-%d')}")
    else:
        reference_date = datetime.now()

    # --- PHASE 2: Build Fetch Plan ---
    archives = fetch_archives(username)
    if not archives and cache_state.total_games == 0:
        print("No archives found!")
        return {"username": username, "status": "no_games"}

    fetch_plan = build_fetch_plan(
        cache_state=cache_state,
        username=username,
        days_back=days_back,
        time_classes=time_classes,
        current_date=reference_date,
        all_archives=archives,
    )

    # --- PHASE 3: Fetch and Merge Games ---
    cached_games, _ = load_cached_games_v2(player_dir)
    archives_fetched = list(cache_state.archives_fetched)

    if fetch_plan.archives_to_fetch:
        print(f"Fetching {len(fetch_plan.archives_to_fetch)} archives ({fetch_plan.fetch_reason})...")
        all_new_games = []
        for archive_month in fetch_plan.archives_to_fetch:
            try:
                parts = archive_month.split("/")
                year, month = int(parts[0]), int(parts[1])
                games = fetch_games(username, year=year, month=month)
                all_new_games.extend(games)
                print(f"  {year}-{month:02d}: {len(games)} games")
                archives_fetched.append(archive_month)
            except Exception as e:
                print(f"  {archive_month}: Error - {e}")

        all_games = merge_games(cached_games, all_new_games)
        save_cache_with_metadata(player_dir, all_games, username, archives_fetched)
    else:
        all_games = cached_games
        print(f"  Cache is current, no fetch needed")

    # --- PHASE 4: Query Cache with Filters ---
    recent_games = query_cache_filtered(
        games=all_games,
        days_back=days_back,
        time_classes=time_classes,
        max_games_per_class=max_games,
        reference_date=reference_date,
    )

    print(f"Total games after filtering: {len(recent_games)}")

    if not recent_games:
        print("No games found!")
        return {"username": username, "status": "no_games"}

    # --- PHASE 5: Extract and Update Opponents ---
    opponents = extract_opponents_from_games(recent_games, username)

    print(f"Unique opponents: {len(opponents)}")

    opponent_profiles = {}
    trustworthy_opponents = set()
    banned_opponents = set()

    if validate_opponents and opponents:
        print("Validating opponent accounts...")
        opponent_profiles = batch_fetch_opponent_profiles(
            list(opponents),
            cache_conn=opponent_cache_conn,
        )

        for opp_username, profile in opponent_profiles.items():
            if profile.is_banned:
                banned_opponents.add(opp_username)
            else:
                trustworthy_opponents.add(opp_username)

        print(f"  Trustworthy opponents: {len(trustworthy_opponents)}")
        print(f"  Banned opponents: {len(banned_opponents)}")

    # Split games by opponent trust
    if validate_opponents:
        trustworthy_games, untrustworthy_games = split_games_by_opponent_trust(
            recent_games, username, opponent_profiles
        )
        print(f"Games vs trustworthy opponents: {len(trustworthy_games)}")
        print(f"Games vs banned opponents: {len(untrustworthy_games)}")
    else:
        trustworthy_games = recent_games
        untrustworthy_games = []

    # Analyze frequent opponents
    frequent_analysis = {}
    if validate_opponents:
        print("Analyzing frequent opponents...")
        frequent_analysis = analyze_frequent_opponents(
            recent_games, username, opponent_profiles, std_threshold=2.0
        )
        print(f"  Frequent opponents (+2σ): {frequent_analysis['distribution_stats']['frequent_count']}")

    # Build game dataset (with opponent ban status flag)
    print("Building game dataset...")
    game_dataset = build_game_dataset(
        recent_games, username,
        time_classes=time_classes,
        opponent_profiles=opponent_profiles if validate_opponents else None,
    )
    print(f"  Filtered games ({', '.join(time_classes)}): {len(game_dataset)}")

    if not game_dataset:
        print("No games match criteria!")
        return {"username": username, "status": "no_matching_games"}

    # Apply max_games limit per time class
    if max_games:
        print(f"Applying max {max_games} games per time class limit...")
        game_dataset = limit_games_per_time_class(game_dataset, max_games, time_classes)
        print(f"  After limiting: {len(game_dataset)} games")

    # Count trustworthy vs banned games (for summary stats)
    if validate_opponents:
        trustworthy_count = sum(1 for g in game_dataset if not g.get("opponent_is_banned", False))
        banned_count = sum(1 for g in game_dataset if g.get("opponent_is_banned", False))
        print(f"  Games vs fair opponents: {trustworthy_count}")
        print(f"  Games vs banned opponents: {banned_count}")
    else:
        trustworthy_count = len(game_dataset)
        banned_count = 0

    # Build position dataset
    print("Building position dataset...")
    limited_game_urls = {g.get("url") for g in game_dataset if g.get("url")}
    limited_recent_games = [g for g in recent_games if g.get("url") in limited_game_urls]
    game_ds, pos_ds = build_position_dataset(
        limited_recent_games, username,
        time_classes=time_classes,
        max_games=None,
    )
    print(f"  Games processed: {len(game_ds)}")
    print(f"  Positions extracted: {len(pos_ds)}")

    # Build opening book
    print("Building opening book...")
    opening_book = build_opening_book(pos_ds, min_occurrences=2)
    print(f"  Common positions: {opening_book['num_common_positions']}")

    # Aggregate game features
    print("Aggregating game features...")
    game_aggregates = aggregate_all_games(pos_ds)

    # Segment games by Elo ranges
    print("Segmenting by Elo ranges...")
    elo_segments_by_player = segment_games_by_elo_range(
        game_dataset, segment_size=200, segment_by="player",
        trustworthy_opponents=trustworthy_opponents if validate_opponents else None
    )
    elo_segments_by_opponent = segment_games_by_elo_range(
        game_dataset, segment_size=200, segment_by="opponent",
        trustworthy_opponents=trustworthy_opponents if validate_opponents else None
    )
    print(f"  Player Elo segments: {len(elo_segments_by_player)}")
    print(f"  Opponent Elo segments: {len(elo_segments_by_opponent)}")

    # Analyze patterns
    print("Analyzing patterns...")
    elo_analysis = analyze_elo_patterns(game_dataset)
    result_patterns = analyze_result_patterns(game_dataset)
    termination_patterns = analyze_termination_patterns(game_dataset)

    # Analyze timeout wins while behind
    print("Analyzing timeout patterns...")
    timeout_analysis = analyze_timeout_patterns(game_dataset, game_aggregates)

    # Analyze openings by ECO code
    print("Analyzing openings by ECO code...")
    opening_analysis = analyze_openings_by_eco(game_dataset)
    print(f"  Unique ECO codes: {opening_analysis['unique_eco_codes']}")
    print(f"  ECO coverage: {opening_analysis['eco_coverage']:.1%}")

    # Detect sessions
    print("Detecting sessions...")
    sessions = detect_sessions(limited_recent_games, username)
    session_patterns = analyze_session_patterns(sessions)
    print(f"  Sessions detected: {len(sessions)}")
    if sessions:
        print(f"  Avg session length: {session_patterns['avg_session_length']:.1f} games")
        print(f"  Longest session: {session_patterns['max_session_length']} games")

    # Add session_id to game datasets
    assign_session_ids_to_games(limited_recent_games, sessions)
    # Also update game_dataset with session IDs
    game_id_to_session = {}
    for g in limited_recent_games:
        url = g.get('url', '')
        game_id = url.split('/')[-1] if url else ''
        if game_id and g.get('session_id') is not None:
            game_id_to_session[game_id] = g['session_id']

    for g in game_dataset:
        game_id = g.get('game_id', '')
        g['session_id'] = game_id_to_session.get(game_id)

    # Save datasets
    # Note: games.parquet now includes opponent_is_banned flag for filtering
    print("Saving datasets...")
    save_dataset_parquet(game_dataset, player_dir / "games.parquet")
    save_dataset_parquet(pos_ds, player_dir / "positions.parquet")
    save_dataset_parquet(game_aggregates, player_dir / "game_aggregates.parquet")
    save_opening_book(opening_book, player_dir / "opening_book.json")

    # Save frequent opponents analysis
    if frequent_analysis:
        frequent_opponents_output = {
            "distribution_stats": frequent_analysis["distribution_stats"],
            "frequent_opponents": [
                asdict(fo) for fo in frequent_analysis["frequent_opponents"]
            ],
            "trustable_frequent_opponents": frequent_analysis["trustable_frequent_opponents"],
            "banned_opponents_encountered": frequent_analysis["banned_opponents_encountered"],
        }
        with open(player_dir / "frequent_opponents.json", "w") as f:
            json.dump(frequent_opponents_output, f, indent=2)

    # Save Elo segments
    elo_segments_output = {
        "by_player_elo": {
            name: asdict(stats) for name, stats in elo_segments_by_player.items()
        },
        "by_opponent_elo": {
            name: asdict(stats) for name, stats in elo_segments_by_opponent.items()
        },
    }
    with open(player_dir / "elo_segments.json", "w") as f:
        json.dump(elo_segments_output, f, indent=2)

    # Save sessions
    if sessions:
        sessions_output = {
            "session_gap_minutes": DEFAULT_SESSION_GAP_MINUTES,
            "patterns": session_patterns,
            "sessions": [asdict(s) for s in sessions],
        }
        with open(player_dir / "sessions.json", "w") as f:
            json.dump(sessions_output, f, indent=2)

    # Count games by time class
    games_by_time_class = {}
    for tc in time_classes:
        tc_games = [g for g in game_dataset if g.get("time_class") == tc]
        tc_fair = sum(1 for g in tc_games if not g.get("opponent_is_banned", False))
        tc_cheat = sum(1 for g in tc_games if g.get("opponent_is_banned", False))
        games_by_time_class[tc] = {
            "total": len(tc_games),
            "fair": tc_fair,
            "cheat": tc_cheat,
        }

    # Save analysis summary
    summary = {
        "username": username,
        "status": "success",
        "generated_at": datetime.now().isoformat(),
        "days_back": days_back,
        "fetch_all_history": days_back is None,
        "time_classes": time_classes,
        "total_games": len(game_dataset),
        "games_vs_fair_opponents": trustworthy_count,
        "games_vs_banned_opponents": banned_count,
        "games_by_time_class": games_by_time_class,
        "positions_extracted": len(pos_ds),
        "unique_opponents": len(opponents),
        "banned_opponents_count": len(banned_opponents),
        "frequent_opponents_count": frequent_analysis.get("distribution_stats", {}).get("frequent_count", 0),
        "elo_segments_by_player": len(elo_segments_by_player),
        "elo_segments_by_opponent": len(elo_segments_by_opponent),
        "elo_analysis": asdict(elo_analysis),
        "result_patterns": asdict(result_patterns),
        "termination_patterns": asdict(termination_patterns),
        "timeout_analysis": timeout_analysis,
        "opening_book_summary": {
            "unique_positions": opening_book["num_unique_positions"],
            "common_positions": opening_book["num_common_positions"],
            "avg_opening_depth": opening_book["opening_depth_avg"],
        },
        "opening_analysis": opening_analysis,
        "session_patterns": session_patterns,
    }

    with open(player_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Save opening analysis separately
    with open(player_dir / "opening_analysis.json", "w") as f:
        json.dump(opening_analysis, f, indent=2)

    print(f"Saved to: {player_dir}")

    return summary


def generate_combined_baseline(
    player_summaries: list[dict],
    output_dir: Path,
) -> dict:
    """
    Combine individual player baselines into aggregate statistics.

    Args:
        player_summaries: List of summary dicts from generate_player_baseline
        output_dir: Directory to save combined baseline

    Returns:
        Combined baseline dictionary
    """
    print(f"\n{'='*60}")
    print("Generating combined baseline statistics")
    print(f"{'='*60}")

    # Filter to successful summaries
    successful = [s for s in player_summaries if s.get("status") == "success"]

    if not successful:
        print("No successful player baselines to combine!")
        return {"status": "no_successful_baselines"}

    # Aggregate statistics
    def avg(lst):
        return sum(lst) / len(lst) if lst else 0

    def std(lst):
        if len(lst) < 2:
            return 0
        mean = avg(lst)
        return (sum((x - mean) ** 2 for x in lst) / (len(lst) - 1)) ** 0.5

    # Collect metrics across players
    win_rates = [s["elo_analysis"]["win_rate"] for s in successful]
    manipulation_scores = [s["elo_analysis"]["rating_manipulation_score"] for s in successful]
    checkmate_rates = [s["result_patterns"]["checkmate_rate"] for s in successful]
    entropy_values = [s["result_patterns"]["result_entropy"] for s in successful]

    # Upset metrics
    upset_rates = [s["elo_analysis"]["upset_rate"] for s in successful]
    losses_to_lower_rates = [s["elo_analysis"]["losses_to_lower_rate"] for s in successful]

    # Timeout metrics
    timeout_win_rates = [s["timeout_analysis"]["timeout_win_rate"] for s in successful]
    timeout_behind_rates = [s["timeout_analysis"]["behind_rate"] for s in successful]

    # Termination rates
    termination_checkmate = [s["termination_patterns"]["checkmate_rate"] for s in successful]
    termination_timeout = [s["termination_patterns"]["timeout_rate"] for s in successful]
    termination_resign = [s["termination_patterns"]["resignation_rate"] for s in successful]
    termination_draw = [s["termination_patterns"]["draw_rate"] for s in successful]

    # Win/Loss by checkmate for detailed analysis
    win_checkmate_rates = []
    loss_checkmate_rates = []
    for summary in successful:
        rp = summary["result_patterns"]
        total_games = summary["total_games"]
        if total_games > 0:
            win_cm_rate = rp.get("wins_by_checkmate", 0) / total_games
            loss_cm_rate = rp.get("losses_by_checkmate", 0) / total_games
            win_checkmate_rates.append(win_cm_rate)
            loss_checkmate_rates.append(loss_cm_rate)

    # Collect all trustable frequent opponents (deduplicated)
    all_trustable_opponents = set()
    all_banned_opponents = set()
    for summary in successful:
        # Load frequent opponents from file
        player_dir = output_dir / summary["username"]
        freq_file = player_dir / "frequent_opponents.json"
        if freq_file.exists():
            with open(freq_file) as f:
                freq_data = json.load(f)
            all_trustable_opponents.update(freq_data.get("trustable_frequent_opponents", []))
            all_banned_opponents.update(freq_data.get("banned_opponents_encountered", []))

    # Collect opening statistics
    combined_eco_distribution = {}
    for summary in successful:
        eco_dist = summary.get("opening_analysis", {}).get("eco_distribution", {})
        for eco, stats in eco_dist.items():
            if eco not in combined_eco_distribution:
                combined_eco_distribution[eco] = {"count": 0, "wins": 0, "losses": 0, "draws": 0}
            combined_eco_distribution[eco]["count"] += stats["count"]
            combined_eco_distribution[eco]["wins"] += stats["wins"]
            combined_eco_distribution[eco]["losses"] += stats["losses"]
            combined_eco_distribution[eco]["draws"] += stats["draws"]

    # Aggregate games by time class across all players
    combined_by_time_class = {}
    for summary in successful:
        for tc, tc_stats in summary.get("games_by_time_class", {}).items():
            if tc not in combined_by_time_class:
                combined_by_time_class[tc] = {"total": 0, "fair": 0, "cheat": 0}
            combined_by_time_class[tc]["total"] += tc_stats.get("total", 0)
            combined_by_time_class[tc]["fair"] += tc_stats.get("fair", 0)
            combined_by_time_class[tc]["cheat"] += tc_stats.get("cheat", 0)

    combined = {
        "generated_at": datetime.now().isoformat(),
        "num_players": len(successful),
        "total_games": sum(s["total_games"] for s in successful),
        "total_fair_games": sum(s.get("games_vs_fair_opponents", 0) for s in successful),
        "total_cheat_games": sum(s.get("games_vs_banned_opponents", 0) for s in successful),
        "games_by_time_class": combined_by_time_class,
        "total_positions": sum(s["positions_extracted"] for s in successful),
        "unique_opponents_total": sum(s["unique_opponents"] for s in successful),
        "banned_opponents_total": sum(s["banned_opponents_count"] for s in successful),

        "elo_baseline": {
            "win_rate_mean": avg(win_rates),
            "win_rate_std": std(win_rates),
            "manipulation_score_mean": avg(manipulation_scores),
            "manipulation_score_std": std(manipulation_scores),
            "manipulation_score_max": max(manipulation_scores) if manipulation_scores else 0,
        },

        "result_baseline": {
            "checkmate_rate_mean": avg(checkmate_rates),
            "checkmate_rate_std": std(checkmate_rates),
            "win_checkmate_rate_mean": avg(win_checkmate_rates),
            "win_checkmate_rate_std": std(win_checkmate_rates),
            "loss_checkmate_rate_mean": avg(loss_checkmate_rates),
            "loss_checkmate_rate_std": std(loss_checkmate_rates),
            "entropy_mean": avg(entropy_values),
            "entropy_std": std(entropy_values),
        },

        "upset_baseline": {
            "upset_rate_mean": avg(upset_rates),
            "upset_rate_std": std(upset_rates),
            "losses_to_lower_rate_mean": avg(losses_to_lower_rates),
            "losses_to_lower_rate_std": std(losses_to_lower_rates),
        },

        "timeout_baseline": {
            "timeout_win_rate_mean": avg(timeout_win_rates),
            "timeout_win_rate_std": std(timeout_win_rates),
            "timeout_behind_rate_mean": avg(timeout_behind_rates),
            "timeout_behind_rate_std": std(timeout_behind_rates),
        },

        "termination_baseline": {
            "checkmate_rate_mean": avg(termination_checkmate),
            "checkmate_rate_std": std(termination_checkmate),
            "timeout_rate_mean": avg(termination_timeout),
            "timeout_rate_std": std(termination_timeout),
            "resignation_rate_mean": avg(termination_resign),
            "resignation_rate_std": std(termination_resign),
            "draw_rate_mean": avg(termination_draw),
            "draw_rate_std": std(termination_draw),
        },

        "opening_baseline": {
            "unique_eco_codes": len(combined_eco_distribution),
            "eco_distribution": combined_eco_distribution,
        },

        "opponent_analysis": {
            "trustable_frequent_opponents": sorted(all_trustable_opponents),
            "banned_opponents_encountered": sorted(all_banned_opponents),
        },

        "players": [s["username"] for s in successful],
    }

    # Save combined baseline
    output_path = output_dir / "combined_baseline.json"
    with open(output_path, "w") as f:
        json.dump(combined, f, indent=2)

    print(f"\nCombined baseline:")
    print(f"  Players: {combined['num_players']}")
    print(f"  Total games: {combined['total_games']}")
    fair_pct = combined['total_fair_games'] / combined['total_games'] * 100 if combined['total_games'] > 0 else 0
    print(f"  Fair games (vs good standing): {combined['total_fair_games']} ({fair_pct:.1f}%)")
    print(f"  Cheat games (vs banned): {combined['total_cheat_games']}")
    print(f"  Games by time class:")
    for tc, tc_stats in combined.get("games_by_time_class", {}).items():
        print(f"    {tc}: {tc_stats['total']} total ({tc_stats['fair']} fair, {tc_stats['cheat']} cheat)")
    print(f"  Unique opponents: {combined['unique_opponents_total']}")
    print(f"  Banned opponents: {combined['banned_opponents_total']}")
    print(f"  Win rate: {combined['elo_baseline']['win_rate_mean']:.1%} ± {combined['elo_baseline']['win_rate_std']:.1%}")
    print(f"  Manipulation score: {combined['elo_baseline']['manipulation_score_mean']:.3f} (max: {combined['elo_baseline']['manipulation_score_max']:.3f})")
    print(f"  Timeout win rate: {combined['timeout_baseline']['timeout_win_rate_mean']:.1%} ± {combined['timeout_baseline']['timeout_win_rate_std']:.1%}")
    print(f"  Timeout wins while behind: {combined['timeout_baseline']['timeout_behind_rate_mean']:.1%} ± {combined['timeout_baseline']['timeout_behind_rate_std']:.1%}")
    print(f"  Trustable frequent opponents: {len(all_trustable_opponents)}")
    print(f"  Saved to: {output_path}")

    return combined
