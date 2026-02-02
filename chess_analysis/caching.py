"""
Game and opponent caching utilities for chess analysis.

This module provides:
- Game cache management (v1 and v2 formats)
- Opponent profile caching with SQLite
- Smart fetching based on cache state
- Cache migration utilities
"""

import json
import shutil
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class TimeClassCacheState:
    """Cache state for a specific time class."""
    time_class: str
    newest_game_timestamp: int
    oldest_game_timestamp: int
    game_count: int


@dataclass
class CacheState:
    """Represents the current state of a player's game cache."""
    username: str
    cache_version: int  # 1 for legacy, 2 for new format
    cached_at: Optional[datetime]
    last_fetch_by_time_class: dict  # str -> TimeClassCacheState
    archives_fetched: set
    total_games: int
    needs_migration: bool  # True for version 1 caches


@dataclass
class FetchPlan:
    """Plan for what to fetch from the API."""
    archives_to_fetch: list  # Archive URLs to fetch
    fetch_reason: str  # "initial", "incremental", "expanded_range"
    estimated_api_calls: int


@dataclass
class OpponentProfile:
    """Profile information for an opponent fetched from chess.com API."""
    username: str
    status: str  # "premium", "closed:fair_play_violations", etc.
    is_banned: bool
    rating_blitz: Optional[int] = None
    rating_rapid: Optional[int] = None
    rating_bullet: Optional[int] = None
    fetched_at: str = ""  # ISO timestamp
    # Extended fields (added for baseline expansion)
    joined: Optional[int] = None  # Unix timestamp of account creation
    blitz_rd: Optional[int] = None  # Glicko Rating Deviation for blitz
    rapid_rd: Optional[int] = None  # Glicko Rating Deviation for rapid
    total_games_blitz: Optional[int] = None  # Total blitz games played
    total_games_rapid: Optional[int] = None  # Total rapid games played
    # Additional fields for expanded cheaters dataset
    last_online: Optional[int] = None  # Unix timestamp (= ban date for closed accounts)
    title: Optional[str] = None  # Chess title (GM, IM, FM, etc.)
    player_id: Optional[int] = None  # Chess.com player ID


# =============================================================================
# Opponent Cache (SQLite)
# =============================================================================

def get_shared_opponent_cache_path(data_dir: Path = Path("data")) -> Path:
    """Get path to shared opponent cache at data/opponent_cache.db."""
    return data_dir / "opponent_cache.db"


def migrate_opponent_cache_if_needed(old_path: Path, new_path: Path) -> bool:
    """
    Migrate opponent cache from baseline-specific to shared location.

    Strategy:
    1. If new_path exists: use it (already migrated)
    2. If old_path exists and new_path doesn't: copy old to new
    3. If neither exists: will be created fresh

    Returns True if migration occurred.
    """
    if new_path.exists():
        return False  # Already migrated or using new location

    if old_path.exists():
        new_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(old_path, new_path)
        print(f"  Migrated opponent cache: {old_path} -> {new_path}")
        return True

    return False


def init_opponent_cache(cache_path: Path) -> sqlite3.Connection:
    """Initialize SQLite database for opponent profile caching."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(cache_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS opponent_profiles (
            username TEXT PRIMARY KEY,
            status TEXT,
            is_banned INTEGER,
            rating_blitz INTEGER,
            rating_rapid INTEGER,
            rating_bullet INTEGER,
            fetched_at TEXT,
            joined INTEGER,
            blitz_rd INTEGER,
            rapid_rd INTEGER,
            total_games_blitz INTEGER,
            total_games_rapid INTEGER,
            last_online INTEGER,
            title TEXT,
            player_id INTEGER
        )
    """)
    conn.commit()

    # Migrate existing tables to add new columns if they don't exist
    _migrate_opponent_cache_schema(conn)

    return conn


def _migrate_opponent_cache_schema(conn: sqlite3.Connection):
    """Add new columns to opponent_profiles table if they don't exist."""
    cursor = conn.execute("PRAGMA table_info(opponent_profiles)")
    existing_columns = {row[1] for row in cursor.fetchall()}

    new_columns = [
        ("joined", "INTEGER"),
        ("blitz_rd", "INTEGER"),
        ("rapid_rd", "INTEGER"),
        ("total_games_blitz", "INTEGER"),
        ("total_games_rapid", "INTEGER"),
        ("last_online", "INTEGER"),
        ("title", "TEXT"),
        ("player_id", "INTEGER"),
    ]

    for col_name, col_type in new_columns:
        if col_name not in existing_columns:
            conn.execute(f"ALTER TABLE opponent_profiles ADD COLUMN {col_name} {col_type}")

    conn.commit()


def get_cached_opponent_profile(conn: sqlite3.Connection, username: str) -> Optional[OpponentProfile]:
    """Retrieve opponent profile from cache if exists."""
    cursor = conn.execute(
        "SELECT username, status, is_banned, rating_blitz, rating_rapid, rating_bullet, "
        "fetched_at, joined, blitz_rd, rapid_rd, total_games_blitz, total_games_rapid, "
        "last_online, title, player_id "
        "FROM opponent_profiles WHERE username = ?",
        (username.lower(),)
    )
    row = cursor.fetchone()
    if row:
        return OpponentProfile(
            username=row[0],
            status=row[1],
            is_banned=bool(row[2]),
            rating_blitz=row[3],
            rating_rapid=row[4],
            rating_bullet=row[5],
            fetched_at=row[6],
            joined=row[7],
            blitz_rd=row[8],
            rapid_rd=row[9],
            total_games_blitz=row[10],
            total_games_rapid=row[11],
            last_online=row[12],
            title=row[13],
            player_id=row[14],
        )
    return None


def save_opponent_profile_to_cache(conn: sqlite3.Connection, profile: OpponentProfile):
    """Save opponent profile to cache."""
    conn.execute("""
        INSERT OR REPLACE INTO opponent_profiles
        (username, status, is_banned, rating_blitz, rating_rapid, rating_bullet, fetched_at,
         joined, blitz_rd, rapid_rd, total_games_blitz, total_games_rapid,
         last_online, title, player_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        profile.username.lower(),
        profile.status,
        int(profile.is_banned),
        profile.rating_blitz,
        profile.rating_rapid,
        profile.rating_bullet,
        profile.fetched_at,
        profile.joined,
        profile.blitz_rd,
        profile.rapid_rd,
        profile.total_games_blitz,
        profile.total_games_rapid,
        profile.last_online,
        profile.title,
        profile.player_id,
    ))
    conn.commit()


# =============================================================================
# Game Cache Functions (v2 with metadata)
# =============================================================================

def get_cache_path(player_dir: Path) -> Path:
    """Get path to the games cache file."""
    return player_dir / "games_cache.json"


def load_cached_games_v2(player_dir: Path) -> tuple[list[dict], dict]:
    """
    Load cached games from player directory, supporting both v1 and v2 formats.

    Returns:
        Tuple of (cached_games_list, cache_metadata_dict)
        For v1 caches, metadata will have version=1 and minimal info.
    """
    cache_path = get_cache_path(player_dir)
    if not cache_path.exists():
        return [], {"version": 0}

    try:
        with open(cache_path) as f:
            cached = json.load(f)

        games = cached.get("games", [])

        # Check for v2 format
        if "version" in cached and cached["version"] >= 2:
            metadata = {
                "version": cached["version"],
                "cached_at": cached.get("cached_at"),
                "cache_metadata": cached.get("cache_metadata", {}),
            }
        else:
            # Legacy v1 format - extract minimal metadata
            metadata = {
                "version": 1,
                "cached_at": cached.get("cached_at"),
                "cache_metadata": {},
            }

        print(f"  Loaded {len(games)} cached games (v{metadata['version']})")
        return games, metadata
    except Exception as e:
        print(f"  Warning: Could not load cache: {e}")
        return [], {"version": 0}


def analyze_cache_state(
    player_dir: Path,
    username: str,
) -> CacheState:
    """
    Analyze the current state of a player's game cache.

    Returns CacheState describing cache contents and whether migration is needed.
    """
    games, metadata = load_cached_games_v2(player_dir)

    if not games:
        return CacheState(
            username=username,
            cache_version=0,
            cached_at=None,
            last_fetch_by_time_class={},
            archives_fetched=set(),
            total_games=0,
            needs_migration=False,
        )

    # Calculate per-time-class statistics from games
    time_class_stats = {}
    for game in games:
        tc = game.get("time_class", "unknown")
        end_time = game.get("end_time", 0)

        if tc not in time_class_stats:
            time_class_stats[tc] = TimeClassCacheState(
                time_class=tc,
                newest_game_timestamp=end_time,
                oldest_game_timestamp=end_time,
                game_count=0,
            )

        stats = time_class_stats[tc]
        stats.game_count += 1
        if end_time > stats.newest_game_timestamp:
            stats.newest_game_timestamp = end_time
        if end_time < stats.oldest_game_timestamp:
            stats.oldest_game_timestamp = end_time

    # Get archives from metadata if v2, otherwise empty
    cache_meta = metadata.get("cache_metadata", {})
    archives_fetched = set(cache_meta.get("archives_fetched", []))

    # Parse cached_at
    cached_at = None
    if metadata.get("cached_at"):
        try:
            cached_at = datetime.fromisoformat(metadata["cached_at"])
        except (ValueError, TypeError):
            pass

    return CacheState(
        username=username,
        cache_version=metadata.get("version", 1),
        cached_at=cached_at,
        last_fetch_by_time_class=time_class_stats,
        archives_fetched=archives_fetched,
        total_games=len(games),
        needs_migration=metadata.get("version", 1) < 2,
    )


def build_fetch_plan(
    cache_state: CacheState,
    username: str,
    days_back: Optional[int],
    time_classes: list[str],
    current_date: Optional[datetime] = None,
    all_archives: Optional[list[str]] = None,
) -> FetchPlan:
    """
    Build a plan for what archives need to be fetched.

    Args:
        cache_state: Current cache state
        username: Chess.com username
        days_back: Number of days of history (None = all)
        time_classes: Time classes to fetch
        current_date: Reference date (defaults to now)
        all_archives: List of all available archive URLs (if already fetched)

    Returns:
        FetchPlan with archives to fetch and reason
    """
    if current_date is None:
        current_date = datetime.now()

    # Calculate date range
    if days_back is not None:
        start_date = current_date - timedelta(days=days_back)
    else:
        start_date = None  # Fetch all available

    # Determine which months we need
    needed_months = set()
    if start_date:
        current = start_date
        while current <= current_date:
            needed_months.add(f"{current.year}/{current.month:02d}")
            current += timedelta(days=15)
        needed_months.add(f"{current_date.year}/{current_date.month:02d}")
    elif all_archives:
        # Extract months from all archive URLs
        for url in all_archives:
            parts = url.rstrip("/").split("/")
            if len(parts) >= 2:
                needed_months.add(f"{parts[-2]}/{parts[-1]}")

    # No cache - fetch everything needed
    if cache_state.total_games == 0:
        return FetchPlan(
            archives_to_fetch=sorted(needed_months),
            fetch_reason="initial",
            estimated_api_calls=len(needed_months),
        )

    # Have cache - determine incremental fetch
    # Find newest game timestamp across all time classes
    newest_cached = 0
    for tc, stats in cache_state.last_fetch_by_time_class.items():
        if stats.newest_game_timestamp > newest_cached:
            newest_cached = stats.newest_game_timestamp

    if newest_cached > 0:
        newest_date = datetime.fromtimestamp(newest_cached)
        # Fetch from the month of newest cached game to now
        incremental_months = set()
        current = newest_date.replace(day=1)
        while current <= current_date:
            incremental_months.add(f"{current.year}/{current.month:02d}")
            # Move to next month
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)

        # Filter to only months we need
        archives_to_fetch = sorted(incremental_months & needed_months) if needed_months else sorted(incremental_months)

        if archives_to_fetch:
            return FetchPlan(
                archives_to_fetch=archives_to_fetch,
                fetch_reason="incremental",
                estimated_api_calls=len(archives_to_fetch),
            )

    # Cache is up to date
    return FetchPlan(
        archives_to_fetch=[],
        fetch_reason="cache_current",
        estimated_api_calls=0,
    )


def merge_games(
    cached_games: list[dict],
    new_games: list[dict],
) -> list[dict]:
    """
    Merge cached games with newly fetched games, avoiding duplicates.

    Returns:
        Combined list of games sorted by end_time descending (newest first)
    """
    cached_urls = {g.get("url", "") for g in cached_games if g.get("url")}

    # Add new games that aren't in cache
    new_count = 0
    for game in new_games:
        url = game.get("url", "")
        if url and url not in cached_urls:
            cached_games.append(game)
            cached_urls.add(url)
            new_count += 1

    if new_count > 0:
        print(f"  Added {new_count} new games to cache")

    # Sort by end_time descending (newest first)
    cached_games.sort(key=lambda g: g.get("end_time", 0), reverse=True)
    return cached_games


def save_cache_with_metadata(
    player_dir: Path,
    games: list[dict],
    username: str,
    archives_fetched: list[str],
) -> None:
    """
    Save games to cache with version 2 metadata.
    """
    # Calculate per-time-class stats
    time_class_stats = {}
    for game in games:
        tc = game.get("time_class", "unknown")
        end_time = game.get("end_time", 0)

        if tc not in time_class_stats:
            time_class_stats[tc] = {
                "newest_game_timestamp": end_time,
                "oldest_game_timestamp": end_time,
                "game_count": 0,
            }

        stats = time_class_stats[tc]
        stats["game_count"] += 1
        if end_time > stats["newest_game_timestamp"]:
            stats["newest_game_timestamp"] = end_time
        if end_time < stats["oldest_game_timestamp"]:
            stats["oldest_game_timestamp"] = end_time

    cache_data = {
        "version": 2,
        "cached_at": datetime.now().isoformat(),
        "cache_metadata": {
            "username": username,
            "last_fetch_by_time_class": time_class_stats,
            "archives_fetched": sorted(set(archives_fetched)),
            "total_games": len(games),
        },
        "games": games,
    }

    cache_path = get_cache_path(player_dir)
    with open(cache_path, "w") as f:
        json.dump(cache_data, f)
    print(f"  Saved {len(games)} games to cache (v2)")


def query_cache_filtered(
    games: list[dict],
    days_back: Optional[int],
    time_classes: list[str],
    max_games_per_class: Optional[int] = None,
    reference_date: Optional[datetime] = None,
) -> list[dict]:
    """
    Filter cached games by request parameters.

    Args:
        games: List of game dicts
        days_back: Number of days from reference_date (None = no date filter)
        time_classes: List of time classes to include
        max_games_per_class: Max games per time class (None = no limit)
        reference_date: Reference date for days_back (defaults to now)

    Returns:
        Filtered list of games
    """
    if reference_date is None:
        reference_date = datetime.now()

    # Filter by date range
    if days_back is not None:
        cutoff_timestamp = int((reference_date - timedelta(days=days_back)).timestamp())
        games = [g for g in games if g.get("end_time", 0) >= cutoff_timestamp]

    # Filter by time class
    if time_classes:
        games = [g for g in games if g.get("time_class", "") in time_classes]

    # Apply max_games limit per time class
    if max_games_per_class is not None:
        games_by_class = {}
        for game in games:
            tc = game.get("time_class", "unknown")
            if tc not in games_by_class:
                games_by_class[tc] = []
            games_by_class[tc].append(game)

        limited_games = []
        for tc in time_classes:
            tc_games = games_by_class.get(tc, [])
            # Sort by end_time descending and take newest
            tc_games.sort(key=lambda g: g.get("end_time", 0), reverse=True)
            limited = tc_games[:max_games_per_class]
            limited_games.extend(limited)
            if len(tc_games) > max_games_per_class:
                print(f"  Limited {tc}: {len(tc_games)} -> {len(limited)} games")

        games = limited_games

    # Final sort by end_time descending
    games.sort(key=lambda g: g.get("end_time", 0), reverse=True)
    return games


def extract_opponents_from_games(games: list[dict], username: str) -> set[str]:
    """
    Extract unique opponent usernames from games.
    """
    opponents = set()
    username_lower = username.lower()

    for game in games:
        white = game.get("white", {})
        black = game.get("black", {})

        white_username = white.get("username", "") if isinstance(white, dict) else ""
        black_username = black.get("username", "") if isinstance(black, dict) else ""

        if white_username.lower() == username_lower:
            if black_username:
                opponents.add(black_username.lower())
        elif black_username.lower() == username_lower:
            if white_username:
                opponents.add(white_username.lower())

    return opponents
