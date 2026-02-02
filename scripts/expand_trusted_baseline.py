#!/usr/bin/env python3
"""
Expand Trusted Baseline Dataset

This script identifies candidate trusted users from frequent opponents of existing
trusted players. It filters for accounts that are:
1. At least 3 years old
2. Not banned for fair play violations
3. Have sufficient games played

Usage:
    python scripts/expand_trusted_baseline.py --config data/trusted/config.json --output candidates.json
    python scripts/expand_trusted_baseline.py --min-age-years 3 --min-games 500
"""

import argparse
import json
import sqlite3
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from chess_analysis.baseline import api_get, fetch_player_profile, is_banned_account, API_BASE
from chess_analysis.caching import (
    init_opponent_cache,
    get_cached_opponent_profile,
    get_shared_opponent_cache_path,
    save_opponent_profile_to_cache,
    OpponentProfile,
)


@dataclass
class CandidateTrustedUser:
    """A candidate for addition to the trusted baseline."""
    username: str
    account_age_years: float
    joined_timestamp: int
    joined_date: str
    status: str
    is_banned: bool
    rating_blitz: Optional[int]
    rating_rapid: Optional[int]
    blitz_rd: Optional[int]
    rapid_rd: Optional[int]
    total_games_blitz: Optional[int]
    total_games_rapid: Optional[int]
    total_games: int  # blitz + rapid
    found_via: list  # List of trusted users who have this as a frequent opponent
    games_against_trusted: int  # Total games played against trusted users


def get_account_age_years(joined_timestamp: int) -> float:
    """Calculate account age in years from joined timestamp."""
    if not joined_timestamp:
        return 0
    joined_date = datetime.fromtimestamp(joined_timestamp)
    age_days = (datetime.now() - joined_date).days
    return age_days / 365.25


def load_existing_trusted_users(config_path: Path) -> set[str]:
    """Load existing trusted users from config."""
    if not config_path.exists():
        return set()

    with open(config_path) as f:
        config = json.load(f)

    # Config uses "players" array with objects containing "username" field
    players = config.get("players", [])
    return set(p["username"].lower() for p in players if isinstance(p, dict) and "username" in p)


def load_frequent_opponents(baseline_dir: Path, trusted_users: set[str]) -> dict[str, list[str]]:
    """
    Load frequent opponents for all trusted users.

    Returns:
        dict mapping opponent username -> list of trusted users they frequently play
    """
    opponent_to_trusted = {}

    for username in trusted_users:
        freq_file = baseline_dir / username / "frequent_opponents.json"
        if not freq_file.exists():
            print(f"  Warning: No frequent_opponents.json for {username}")
            continue

        with open(freq_file) as f:
            freq_data = json.load(f)

        # Get trustable frequent opponents (not banned)
        trustable = freq_data.get("trustable_frequent_opponents", [])

        for opponent in trustable:
            opponent_lower = opponent.lower()
            if opponent_lower not in opponent_to_trusted:
                opponent_to_trusted[opponent_lower] = []
            opponent_to_trusted[opponent_lower].append(username)

    return opponent_to_trusted


def collect_all_opponents_from_cache(cache_path: Path) -> set[str]:
    """Get all opponent usernames from the opponent cache database."""
    if not cache_path.exists():
        return set()

    conn = sqlite3.connect(cache_path)
    cursor = conn.execute("SELECT username FROM opponent_profiles")
    opponents = {row[0] for row in cursor.fetchall()}
    conn.close()

    return opponents


def evaluate_candidate(
    username: str,
    cache_conn: sqlite3.Connection,
    min_age_years: float,
    min_games: int,
    force_refresh: bool = False,
) -> Optional[tuple[OpponentProfile, float]]:
    """
    Evaluate if an opponent qualifies as a candidate trusted user.

    Returns:
        Tuple of (OpponentProfile, account_age_years) if candidate qualifies, None otherwise
    """
    # Check cache first (unless force refresh)
    profile = None
    if not force_refresh:
        profile = get_cached_opponent_profile(cache_conn, username)

    # Fetch fresh profile if needed or if missing new fields
    if profile is None or profile.joined is None:
        fresh_profile = fetch_player_profile(username)
        if fresh_profile:
            save_opponent_profile_to_cache(cache_conn, fresh_profile)
            profile = fresh_profile

    if profile is None:
        return None

    # Check if banned or closed for any reason
    if profile.is_banned:
        return None

    # Exclude any closed account (not just banned)
    if profile.status and profile.status.startswith("closed"):
        return None

    # Check account age
    if not profile.joined:
        return None

    age_years = get_account_age_years(profile.joined)
    if age_years < min_age_years:
        return None

    # Check total games
    total_games = (profile.total_games_blitz or 0) + (profile.total_games_rapid or 0)
    if total_games < min_games:
        return None

    return profile, age_years


def main():
    parser = argparse.ArgumentParser(
        description="Find candidate trusted users from frequent opponents"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("data/trusted/config.json"),
        help="Path to trusted baseline config.json",
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=Path("data/trusted"),
        help="Path to trusted baseline data directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/trusted/candidate_trusted_users.json"),
        help="Output path for candidate list",
    )
    parser.add_argument(
        "--min-age-years",
        type=float,
        default=3.0,
        help="Minimum account age in years (default: 3)",
    )
    parser.add_argument(
        "--min-games",
        type=int,
        default=500,
        help="Minimum total games (blitz + rapid) (default: 500)",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force refresh profiles from API even if cached",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=50,
        help="Number of top candidates to output (default: 50)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Expand Trusted Baseline - Candidate Identification")
    print("=" * 60)

    # Load existing trusted users
    print(f"\nLoading existing trusted users from {args.config}...")
    existing_trusted = load_existing_trusted_users(args.config)
    print(f"  Found {len(existing_trusted)} existing trusted users")

    # Load frequent opponents
    print(f"\nLoading frequent opponents from {args.baseline_dir}...")
    opponent_to_trusted = load_frequent_opponents(args.baseline_dir, existing_trusted)
    print(f"  Found {len(opponent_to_trusted)} unique frequent opponents")

    # Filter out existing trusted users
    candidates_to_check = {
        opp: trusted_list
        for opp, trusted_list in opponent_to_trusted.items()
        if opp not in existing_trusted
    }
    print(f"  {len(candidates_to_check)} candidates after excluding existing trusted users")

    # Initialize shared opponent cache (data/opponent_cache.db)
    cache_path = get_shared_opponent_cache_path()
    cache_conn = init_opponent_cache(cache_path)

    # Evaluate each candidate
    print(f"\nEvaluating candidates (min age: {args.min_age_years} years, min games: {args.min_games})...")

    qualified_candidates = []
    checked = 0

    for username, trusted_list in candidates_to_check.items():
        result = evaluate_candidate(
            username,
            cache_conn,
            args.min_age_years,
            args.min_games,
            args.force_refresh,
        )

        checked += 1
        if checked % 50 == 0:
            print(f"  Checked {checked}/{len(candidates_to_check)} opponents...")

        if result is None:
            continue

        profile, age_years = result

        candidate = CandidateTrustedUser(
            username=profile.username,
            account_age_years=round(age_years, 2),
            joined_timestamp=profile.joined,
            joined_date=datetime.fromtimestamp(profile.joined).strftime("%Y-%m-%d"),
            status=profile.status,
            is_banned=profile.is_banned,
            rating_blitz=profile.rating_blitz,
            rating_rapid=profile.rating_rapid,
            blitz_rd=profile.blitz_rd,
            rapid_rd=profile.rapid_rd,
            total_games_blitz=profile.total_games_blitz,
            total_games_rapid=profile.total_games_rapid,
            total_games=(profile.total_games_blitz or 0) + (profile.total_games_rapid or 0),
            found_via=trusted_list,
            games_against_trusted=len(trusted_list),  # Simplified - could sum actual games
        )
        qualified_candidates.append(candidate)

    cache_conn.close()

    print(f"\nFound {len(qualified_candidates)} qualified candidates")

    # Sort by number of trusted connections (more connections = more confident)
    # Then by total games
    qualified_candidates.sort(
        key=lambda c: (len(c.found_via), c.total_games),
        reverse=True
    )

    # Take top N
    top_candidates = qualified_candidates[:args.top_n]

    # Output results
    output_data = {
        "generated_at": datetime.now().isoformat(),
        "criteria": {
            "min_age_years": args.min_age_years,
            "min_games": args.min_games,
        },
        "existing_trusted_users": sorted(existing_trusted),
        "total_candidates_checked": len(candidates_to_check),
        "total_qualified": len(qualified_candidates),
        "top_candidates": [asdict(c) for c in top_candidates],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved {len(top_candidates)} candidates to {args.output}")

    # Print summary
    print("\n" + "=" * 60)
    print("TOP CANDIDATES FOR REVIEW")
    print("=" * 60)

    print(f"\n{'Username':<20} {'Age (yrs)':<10} {'Games':<10} {'Blitz':<8} {'Rapid':<8} {'Connections':<12}")
    print("-" * 78)

    for c in top_candidates[:20]:
        connections = len(c.found_via)
        blitz_str = str(c.rating_blitz) if c.rating_blitz else "-"
        rapid_str = str(c.rating_rapid) if c.rating_rapid else "-"
        print(f"{c.username:<20} {c.account_age_years:<10.1f} {c.total_games:<10} {blitz_str:<8} {rapid_str:<8} {connections:<12}")

    if len(top_candidates) > 20:
        print(f"... and {len(top_candidates) - 20} more candidates in output file")

    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("""
1. Review the candidates in the output file
2. Add approved usernames to data/trusted/config.json
3. Re-run baseline generation:
   python scripts/generate_baseline.py --config data/trusted/config.json --output data/trusted/
""")


if __name__ == "__main__":
    main()
