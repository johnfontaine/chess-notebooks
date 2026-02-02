#!/usr/bin/env python3
"""
Build expanded cheaters dataset from multiple sources.

Consolidates cheaters from:
1. data/cheaters/config.json (manually curated)
2. data/opponent_cache.db (banned opponents)
3. data/titled-cheaters/titled_cheaters.json (Titled Tuesday bans)

Outputs:
- data/expanded-cheaters/cheaters.parquet
- data/expanded-cheaters/summary.json

Usage:
    python scripts/build_expanded_cheaters.py
    python scripts/build_expanded_cheaters.py --refresh-missing
    python scripts/build_expanded_cheaters.py --output-dir data/expanded-cheaters
"""

import argparse
import json
import sqlite3
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from chess_analysis.baseline import fetch_player_profile
from chess_analysis.caching import (
    init_opponent_cache,
    get_shared_opponent_cache_path,
    get_cached_opponent_profile,
    save_opponent_profile_to_cache,
    OpponentProfile,
)


def load_cheaters_config(config_path: Path) -> list[dict]:
    """Load cheaters from config.json."""
    if not config_path.exists():
        print(f"  Warning: {config_path} not found")
        return []

    with open(config_path) as f:
        config = json.load(f)

    players = config.get("players", [])
    print(f"  Loaded {len(players)} from cheaters/config.json")
    return players


def load_titled_cheaters(json_path: Path) -> list[dict]:
    """Load titled cheaters from JSON file."""
    if not json_path.exists():
        print(f"  Warning: {json_path} not found")
        return []

    with open(json_path) as f:
        data = json.load(f)

    players = data.get("players", [])
    print(f"  Loaded {len(players)} from titled_cheaters.json")
    return players


def load_banned_from_cache(conn: sqlite3.Connection) -> list[dict]:
    """Load banned profiles from opponent cache."""
    cursor = conn.execute("""
        SELECT username, status, is_banned, rating_blitz, rating_rapid, rating_bullet,
               fetched_at, joined, blitz_rd, rapid_rd, total_games_blitz, total_games_rapid,
               last_online, title, player_id
        FROM opponent_profiles
        WHERE is_banned = 1
    """)

    profiles = []
    for row in cursor.fetchall():
        profiles.append({
            "username": row[0],
            "status": row[1],
            "is_banned": bool(row[2]),
            "rating_blitz": row[3],
            "rating_rapid": row[4],
            "rating_bullet": row[5],
            "fetched_at": row[6],
            "joined": row[7],
            "blitz_rd": row[8],
            "rapid_rd": row[9],
            "total_games_blitz": row[10],
            "total_games_rapid": row[11],
            "last_online": row[12],
            "title": row[13],
            "player_id": row[14],
        })

    print(f"  Loaded {len(profiles)} banned profiles from opponent_cache.db")
    return profiles


def get_rating_category(rating: Optional[int], title: Optional[str]) -> Optional[str]:
    """
    Get category for a player based on title or rating bucket.

    Titled players use their title (GM, IM, FM, etc.)
    Untitled players use rating bucket = int(rating / 200)
    """
    if title:
        return title
    if rating is not None:
        return str(rating // 200)
    return None


def calculate_days_cheating(joined: Optional[int], last_online: Optional[int]) -> Optional[int]:
    """Calculate days between account creation and ban (last_online)."""
    if joined is None or last_online is None:
        return None
    if last_online <= joined:
        return 0
    return (last_online - joined) // 86400  # Convert seconds to days


def merge_cheaters(
    cheaters_config: list[dict],
    titled_cheaters: list[dict],
    banned_cache: list[dict],
    cache_conn: sqlite3.Connection,
    refresh_missing: bool = False,
) -> list[dict]:
    """
    Merge cheaters from all sources, deduplicating by username.

    Priority for data: cache > titled_cheaters > config
    (Cache has most complete data from API)
    """
    merged = {}  # username -> merged data

    # Track sources for each user
    sources = defaultdict(set)

    # First, add from cheaters/config.json
    for player in cheaters_config:
        username = player.get("username", "").lower()
        if not username:
            continue

        sources[username].add("cheaters_config")
        merged[username] = {
            "username": username,
            "note": player.get("note"),
            "title": player.get("title"),  # Some configs include title
        }

    # Add from titled_cheaters.json
    for player in titled_cheaters:
        username = player.get("username", "").lower()
        if not username:
            continue

        sources[username].add("titled_cheaters")

        if username not in merged:
            merged[username] = {"username": username}

        # Update with titled cheater data (has title, player_id)
        merged[username].update({
            "status": player.get("status"),
            "title": player.get("title") or merged[username].get("title"),
            "player_id": player.get("player_id"),
        })

    # Add from opponent cache (has most complete data)
    for profile in banned_cache:
        username = profile.get("username", "").lower()
        if not username:
            continue

        sources[username].add("opponent_cache")

        if username not in merged:
            merged[username] = {"username": username}

        # Update with cache data (preserving note from config if present)
        note = merged[username].get("note")
        merged[username].update(profile)
        if note:
            merged[username]["note"] = note

    # For users not in cache, optionally fetch from API
    users_needing_fetch = []
    for username, data in merged.items():
        # Check if we have key fields
        if data.get("joined") is None or data.get("last_online") is None:
            # Check cache first
            cached = get_cached_opponent_profile(cache_conn, username)
            if cached:
                # Update from cache
                merged[username].update({
                    "status": cached.status,
                    "is_banned": cached.is_banned,
                    "rating_blitz": cached.rating_blitz,
                    "rating_rapid": cached.rating_rapid,
                    "rating_bullet": cached.rating_bullet,
                    "fetched_at": cached.fetched_at,
                    "joined": cached.joined,
                    "blitz_rd": cached.blitz_rd,
                    "rapid_rd": cached.rapid_rd,
                    "total_games_blitz": cached.total_games_blitz,
                    "total_games_rapid": cached.total_games_rapid,
                    "last_online": cached.last_online,
                    "title": cached.title or merged[username].get("title"),
                    "player_id": cached.player_id,
                })
            else:
                users_needing_fetch.append(username)

    # Fetch missing profiles if requested
    if refresh_missing and users_needing_fetch:
        print(f"\nFetching {len(users_needing_fetch)} missing profiles from API...")
        fetched = 0
        for i, username in enumerate(users_needing_fetch):
            if (i + 1) % 50 == 0:
                print(f"  Progress: {i + 1}/{len(users_needing_fetch)}")

            profile = fetch_player_profile(username)
            if profile:
                save_opponent_profile_to_cache(cache_conn, profile)
                merged[username].update({
                    "status": profile.status,
                    "is_banned": profile.is_banned,
                    "rating_blitz": profile.rating_blitz,
                    "rating_rapid": profile.rating_rapid,
                    "rating_bullet": profile.rating_bullet,
                    "fetched_at": profile.fetched_at,
                    "joined": profile.joined,
                    "blitz_rd": profile.blitz_rd,
                    "rapid_rd": profile.rapid_rd,
                    "total_games_blitz": profile.total_games_blitz,
                    "total_games_rapid": profile.total_games_rapid,
                    "last_online": profile.last_online,
                    "title": profile.title or merged[username].get("title"),
                    "player_id": profile.player_id,
                })
                fetched += 1

        print(f"  Fetched {fetched}/{len(users_needing_fetch)} profiles")

    # Add sources and calculate derived fields
    result = []
    for username, data in merged.items():
        data["sources"] = ",".join(sorted(sources[username]))

        # Calculate days_cheating
        data["days_cheating"] = calculate_days_cheating(
            data.get("joined"),
            data.get("last_online"),
        )

        # Calculate categories for each time control
        title = data.get("title")
        data["category_blitz"] = get_rating_category(data.get("rating_blitz"), title)
        data["category_rapid"] = get_rating_category(data.get("rating_rapid"), title)
        data["category_bullet"] = get_rating_category(data.get("rating_bullet"), title)

        result.append(data)

    return result


def calculate_summary_stats(cheaters: list[dict]) -> dict:
    """Calculate summary statistics for the cheaters dataset."""
    # Count by source
    source_counts = defaultdict(int)
    for c in cheaters:
        for source in c.get("sources", "").split(","):
            if source:
                source_counts[source] += 1

    # Calculate overlaps
    multi_source = sum(1 for c in cheaters if "," in c.get("sources", ""))

    # Count by title
    title_counts = defaultdict(int)
    for c in cheaters:
        title = c.get("title") or "untitled"
        title_counts[title] += 1

    # Count by category (per time control)
    category_counts = {
        "blitz": defaultdict(int),
        "rapid": defaultdict(int),
        "bullet": defaultdict(int),
    }
    for c in cheaters:
        for tc in ["blitz", "rapid", "bullet"]:
            cat = c.get(f"category_{tc}")
            if cat:
                category_counts[tc][cat] += 1

    # Days cheating statistics
    days_values = [c["days_cheating"] for c in cheaters if c.get("days_cheating") is not None]

    days_stats = {}
    if days_values:
        days_stats = {
            "count": len(days_values),
            "mean": statistics.mean(days_values),
            "median": statistics.median(days_values),
            "std": statistics.stdev(days_values) if len(days_values) > 1 else 0,
            "min": min(days_values),
            "max": max(days_values),
        }

    # Days cheating by category (per time control)
    days_by_category = {
        "blitz": defaultdict(list),
        "rapid": defaultdict(list),
        "bullet": defaultdict(list),
    }
    for c in cheaters:
        days = c.get("days_cheating")
        if days is None:
            continue
        for tc in ["blitz", "rapid", "bullet"]:
            cat = c.get(f"category_{tc}")
            if cat:
                days_by_category[tc][cat].append(days)

    # Calculate stats per category
    days_stats_by_category = {}
    for tc in ["blitz", "rapid", "bullet"]:
        days_stats_by_category[tc] = {}
        for cat, values in days_by_category[tc].items():
            if values:
                days_stats_by_category[tc][cat] = {
                    "count": len(values),
                    "mean": statistics.mean(values),
                    "median": statistics.median(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0,
                }

    return {
        "generated_at": datetime.now().isoformat(),
        "total_cheaters": len(cheaters),
        "sources": {
            "cheaters_config": source_counts.get("cheaters_config", 0),
            "opponent_cache": source_counts.get("opponent_cache", 0),
            "titled_cheaters": source_counts.get("titled_cheaters", 0),
            "multi_source": multi_source,
        },
        "by_title": dict(title_counts),
        "by_category": {tc: dict(counts) for tc, counts in category_counts.items()},
        "days_cheating_stats": days_stats,
        "days_cheating_by_category": {
            tc: dict(stats) for tc, stats in days_stats_by_category.items()
        },
        "missing_data": {
            "missing_joined": sum(1 for c in cheaters if c.get("joined") is None),
            "missing_last_online": sum(1 for c in cheaters if c.get("last_online") is None),
            "missing_days_cheating": sum(1 for c in cheaters if c.get("days_cheating") is None),
            "missing_ratings": sum(
                1 for c in cheaters
                if c.get("rating_blitz") is None
                and c.get("rating_rapid") is None
                and c.get("rating_bullet") is None
            ),
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build expanded cheaters dataset from multiple sources",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Build expanded cheaters dataset
    python scripts/build_expanded_cheaters.py

    # Refresh missing profiles from API
    python scripts/build_expanded_cheaters.py --refresh-missing

    # Custom output directory
    python scripts/build_expanded_cheaters.py --output-dir data/my-cheaters
        """,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/expanded-cheaters"),
        help="Output directory (default: data/expanded-cheaters)",
    )
    parser.add_argument(
        "--refresh-missing",
        action="store_true",
        help="Fetch missing profiles from API",
    )
    parser.add_argument(
        "--cheaters-config",
        type=Path,
        default=Path("data/cheaters/config.json"),
        help="Path to cheaters config.json",
    )
    parser.add_argument(
        "--titled-cheaters",
        type=Path,
        default=Path("data/titled-cheaters/titled_cheaters.json"),
        help="Path to titled_cheaters.json",
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=None,
        help="Path to opponent cache (default: data/opponent_cache.db)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Building Expanded Cheaters Dataset")
    print("=" * 60)

    # Determine paths
    cache_path = args.cache_path or get_shared_opponent_cache_path()

    print(f"\nSources:")
    print(f"  Cheaters config: {args.cheaters_config}")
    print(f"  Titled cheaters: {args.titled_cheaters}")
    print(f"  Opponent cache: {cache_path}")
    print(f"  Output: {args.output_dir}")

    # Load from sources
    print(f"\n{'='*60}")
    print("Loading Sources")
    print(f"{'='*60}")

    cheaters_config = load_cheaters_config(args.cheaters_config)
    titled_cheaters = load_titled_cheaters(args.titled_cheaters)

    # Initialize cache connection
    if not cache_path.exists():
        print(f"  Warning: Cache not found at {cache_path}")
        banned_cache = []
        cache_conn = init_opponent_cache(cache_path)
    else:
        cache_conn = init_opponent_cache(cache_path)
        banned_cache = load_banned_from_cache(cache_conn)

    # Merge sources
    print(f"\n{'='*60}")
    print("Merging Sources")
    print(f"{'='*60}")

    cheaters = merge_cheaters(
        cheaters_config,
        titled_cheaters,
        banned_cache,
        cache_conn,
        refresh_missing=args.refresh_missing,
    )

    print(f"\nTotal unique cheaters: {len(cheaters)}")

    # Calculate summary statistics
    print(f"\n{'='*60}")
    print("Calculating Statistics")
    print(f"{'='*60}")

    summary = calculate_summary_stats(cheaters)

    print(f"\nSummary:")
    print(f"  Total cheaters: {summary['total_cheaters']}")
    print(f"  From cheaters_config: {summary['sources']['cheaters_config']}")
    print(f"  From opponent_cache: {summary['sources']['opponent_cache']}")
    print(f"  From titled_cheaters: {summary['sources']['titled_cheaters']}")
    print(f"  In multiple sources: {summary['sources']['multi_source']}")

    print(f"\n  By title:")
    for title, count in sorted(summary['by_title'].items(), key=lambda x: -x[1]):
        print(f"    {title}: {count}")

    if summary['days_cheating_stats']:
        ds = summary['days_cheating_stats']
        print(f"\n  Days cheating:")
        print(f"    Mean: {ds['mean']:.1f}")
        print(f"    Median: {ds['median']:.1f}")
        print(f"    Std: {ds['std']:.1f}")
        print(f"    Range: {ds['min']} - {ds['max']}")

    # Save outputs
    print(f"\n{'='*60}")
    print("Saving Outputs")
    print(f"{'='*60}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Convert to DataFrame and save as parquet
    df = pd.DataFrame(cheaters)

    # Reorder columns for clarity
    column_order = [
        "username", "status", "sources", "title", "player_id",
        "joined", "last_online", "days_cheating",
        "rating_blitz", "rating_rapid", "rating_bullet",
        "category_blitz", "category_rapid", "category_bullet",
        "total_games_blitz", "total_games_rapid",
        "blitz_rd", "rapid_rd",
        "note", "fetched_at", "is_banned",
    ]
    # Only include columns that exist
    columns = [c for c in column_order if c in df.columns]
    # Add any remaining columns
    for c in df.columns:
        if c not in columns:
            columns.append(c)
    df = df[columns]

    parquet_path = args.output_dir / "cheaters.parquet"
    df.to_parquet(parquet_path, index=False)
    print(f"  Saved {len(df)} cheaters to {parquet_path}")

    # Save summary JSON
    summary_path = args.output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved summary to {summary_path}")

    # Close cache connection
    cache_conn.close()

    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
