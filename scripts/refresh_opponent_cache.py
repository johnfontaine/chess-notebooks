#!/usr/bin/env python3
"""
Refresh opponent cache entries with missing or stale data.

This script finds profiles in the opponent cache that have null fields
(like last_online, title, player_id) and re-fetches them from the API.

Usage:
    python scripts/refresh_opponent_cache.py --fix-nulls
    python scripts/refresh_opponent_cache.py --refresh-all
    python scripts/refresh_opponent_cache.py --older-than 30
    python scripts/refresh_opponent_cache.py --banned-only --fix-nulls
"""

import argparse
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from chess_analysis.baseline import fetch_player_profile, REQUEST_DELAY
from chess_analysis.caching import (
    init_opponent_cache,
    get_shared_opponent_cache_path,
    save_opponent_profile_to_cache,
    OpponentProfile,
)


def get_profiles_with_null_fields(conn: sqlite3.Connection, banned_only: bool = False) -> list[str]:
    """Get usernames with null last_online, title, or player_id fields."""
    query = """
        SELECT username FROM opponent_profiles
        WHERE (last_online IS NULL OR title IS NULL OR player_id IS NULL)
    """
    if banned_only:
        query += " AND is_banned = 1"

    cursor = conn.execute(query)
    return [row[0] for row in cursor.fetchall()]


def get_all_profiles(conn: sqlite3.Connection, banned_only: bool = False) -> list[str]:
    """Get all usernames in the cache."""
    query = "SELECT username FROM opponent_profiles"
    if banned_only:
        query += " WHERE is_banned = 1"

    cursor = conn.execute(query)
    return [row[0] for row in cursor.fetchall()]


def get_stale_profiles(
    conn: sqlite3.Connection,
    days: int,
    banned_only: bool = False,
) -> list[str]:
    """Get usernames with fetched_at older than N days."""
    cutoff = (datetime.now() - timedelta(days=days)).isoformat()

    query = """
        SELECT username FROM opponent_profiles
        WHERE (fetched_at < ? OR fetched_at IS NULL)
    """
    if banned_only:
        query += " AND is_banned = 1"

    cursor = conn.execute(query, (cutoff,))
    return [row[0] for row in cursor.fetchall()]


def get_cache_stats(conn: sqlite3.Connection) -> dict:
    """Get statistics about the opponent cache."""
    cursor = conn.execute("SELECT COUNT(*) FROM opponent_profiles")
    total = cursor.fetchone()[0]

    cursor = conn.execute("SELECT COUNT(*) FROM opponent_profiles WHERE is_banned = 1")
    banned = cursor.fetchone()[0]

    cursor = conn.execute("SELECT COUNT(*) FROM opponent_profiles WHERE last_online IS NULL")
    null_last_online = cursor.fetchone()[0]

    cursor = conn.execute("SELECT COUNT(*) FROM opponent_profiles WHERE title IS NULL")
    null_title = cursor.fetchone()[0]

    cursor = conn.execute("SELECT COUNT(*) FROM opponent_profiles WHERE player_id IS NULL")
    null_player_id = cursor.fetchone()[0]

    cursor = conn.execute("SELECT COUNT(*) FROM opponent_profiles WHERE joined IS NULL")
    null_joined = cursor.fetchone()[0]

    cursor = conn.execute("""
        SELECT COUNT(*) FROM opponent_profiles
        WHERE last_online IS NULL OR player_id IS NULL
    """)
    missing_key_fields = cursor.fetchone()[0]

    return {
        "total": total,
        "banned": banned,
        "not_banned": total - banned,
        "null_last_online": null_last_online,
        "null_title": null_title,
        "null_player_id": null_player_id,
        "null_joined": null_joined,
        "missing_key_fields": missing_key_fields,
    }


def refresh_profiles(
    conn: sqlite3.Connection,
    usernames: list[str],
    dry_run: bool = False,
    progress_interval: int = 50,
) -> dict:
    """
    Refresh profiles for the given usernames.

    Returns:
        Dictionary with refresh statistics
    """
    total = len(usernames)
    refreshed = 0
    failed = 0
    not_found = 0

    print(f"Refreshing {total} profiles...")
    if dry_run:
        print("(DRY RUN - no changes will be made)")

    for i, username in enumerate(usernames):
        if (i + 1) % progress_interval == 0:
            print(f"  Progress: {i + 1}/{total} ({refreshed} refreshed, {failed} failed)")

        if dry_run:
            refreshed += 1
            continue

        profile = fetch_player_profile(username)
        if profile:
            save_opponent_profile_to_cache(conn, profile)
            refreshed += 1
        else:
            # Profile not found (404) - could be deleted account
            not_found += 1
            failed += 1

    return {
        "total": total,
        "refreshed": refreshed,
        "failed": failed,
        "not_found": not_found,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Refresh opponent cache entries with missing or stale data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Show cache statistics
    python scripts/refresh_opponent_cache.py --stats

    # Fix entries with null last_online/title/player_id
    python scripts/refresh_opponent_cache.py --fix-nulls

    # Fix nulls for banned accounts only
    python scripts/refresh_opponent_cache.py --fix-nulls --banned-only

    # Refresh all entries older than 30 days
    python scripts/refresh_opponent_cache.py --older-than 30

    # Refresh all entries (full cache refresh)
    python scripts/refresh_opponent_cache.py --refresh-all

    # Dry run (show what would be refreshed)
    python scripts/refresh_opponent_cache.py --fix-nulls --dry-run
        """,
    )

    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show cache statistics and exit",
    )
    parser.add_argument(
        "--fix-nulls",
        action="store_true",
        help="Refresh entries with null last_online, title, or player_id",
    )
    parser.add_argument(
        "--refresh-all",
        action="store_true",
        help="Refresh all entries in the cache",
    )
    parser.add_argument(
        "--older-than",
        type=int,
        metavar="DAYS",
        help="Refresh entries older than N days",
    )
    parser.add_argument(
        "--banned-only",
        action="store_true",
        help="Only process banned accounts",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be refreshed without making changes",
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=None,
        help="Path to opponent cache (default: data/opponent_cache.db)",
    )

    args = parser.parse_args()

    # Determine cache path
    if args.cache_path:
        cache_path = args.cache_path
    else:
        cache_path = get_shared_opponent_cache_path()

    if not cache_path.exists():
        print(f"Error: Cache not found at {cache_path}")
        sys.exit(1)

    print(f"Cache: {cache_path}")

    # Initialize connection (this also runs migrations)
    conn = init_opponent_cache(cache_path)

    # Show stats
    stats = get_cache_stats(conn)
    print(f"\n{'='*60}")
    print("Cache Statistics")
    print(f"{'='*60}")
    print(f"  Total profiles: {stats['total']}")
    print(f"  Banned accounts: {stats['banned']}")
    print(f"  Not banned: {stats['not_banned']}")
    print(f"  Missing last_online: {stats['null_last_online']}")
    print(f"  Missing title: {stats['null_title']} (note: most accounts don't have titles)")
    print(f"  Missing player_id: {stats['null_player_id']}")
    print(f"  Missing joined: {stats['null_joined']}")
    print(f"  Missing key fields (last_online or player_id): {stats['missing_key_fields']}")

    if args.stats:
        conn.close()
        return

    # Determine which profiles to refresh
    usernames = []

    if args.fix_nulls:
        usernames = get_profiles_with_null_fields(conn, args.banned_only)
        print(f"\nFound {len(usernames)} profiles with null fields to refresh")
    elif args.refresh_all:
        usernames = get_all_profiles(conn, args.banned_only)
        print(f"\nRefreshing all {len(usernames)} profiles")
    elif args.older_than:
        usernames = get_stale_profiles(conn, args.older_than, args.banned_only)
        print(f"\nFound {len(usernames)} profiles older than {args.older_than} days")
    else:
        print("\nNo action specified. Use --fix-nulls, --refresh-all, or --older-than")
        print("Use --stats to just show statistics")
        conn.close()
        return

    if not usernames:
        print("No profiles to refresh!")
        conn.close()
        return

    # Confirm before proceeding (unless dry run)
    if not args.dry_run:
        print(f"\nThis will make approximately {len(usernames) * 2} API calls")
        print(f"Estimated time: {len(usernames) * REQUEST_DELAY * 2 / 60:.1f} minutes")
        response = input("Continue? [y/N] ")
        if response.lower() != 'y':
            print("Aborted.")
            conn.close()
            return

    # Refresh profiles
    print(f"\n{'='*60}")
    print("Refreshing Profiles")
    print(f"{'='*60}")

    result = refresh_profiles(conn, usernames, args.dry_run)

    print(f"\n{'='*60}")
    print("Results")
    print(f"{'='*60}")
    print(f"  Total processed: {result['total']}")
    print(f"  Refreshed: {result['refreshed']}")
    print(f"  Failed: {result['failed']}")
    if result['not_found'] > 0:
        print(f"  Not found (deleted accounts?): {result['not_found']}")

    # Show updated stats
    if not args.dry_run:
        print(f"\n{'='*60}")
        print("Updated Cache Statistics")
        print(f"{'='*60}")
        new_stats = get_cache_stats(conn)
        print(f"  Missing last_online: {stats['null_last_online']} -> {new_stats['null_last_online']}")
        print(f"  Missing player_id: {stats['null_player_id']} -> {new_stats['null_player_id']}")

    conn.close()


if __name__ == "__main__":
    main()
