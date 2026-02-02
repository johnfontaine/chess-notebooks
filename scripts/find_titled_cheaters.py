#!/usr/bin/env python3
"""
Find titled players who have been caught cheating in Titled Tuesday tournaments.

This script:
1. Fetches tournaments for known titled players
2. Finds all Titled Tuesday tournaments from the last 2 years
3. Gets all participants from each tournament's rounds
4. Checks each player's status for "closed:fair_play_violations"
5. Saves the list of titled cheaters to JSON

Usage:
    python scripts/find_titled_cheaters.py
"""

import json
import os
import sys
import time
import requests
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

from dotenv import load_dotenv


def log(msg: str):
    """Print with flush to ensure output is visible."""
    print(msg, flush=True)

# Load environment variables from .env file
load_dotenv()

# Chess.com API base
API_BASE = "https://api.chess.com/pub"

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

# Sample players to find Titled Tuesday tournaments
SAMPLE_PLAYERS = [
    "hikaru",
    "imrosen",
    "wonderfultime",
    "gmbenjaminbok",
    "magnuscarlsen",
    "gothamchess",
]

# Rate limiting - be respectful of Chess.com API
REQUEST_DELAY = 0.3  # 300ms between requests


def api_get(url: str, retries: int = 3) -> dict | None:
    """Make a GET request to the chess.com API with retries."""
    for attempt in range(retries):
        try:
            time.sleep(REQUEST_DELAY)
            response = requests.get(url, headers=HEADERS, timeout=60)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                return None
            elif response.status_code == 429:
                # Rate limited - wait longer
                wait_time = 10 * (attempt + 1)
                log(f"  Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
            elif response.status_code == 301:
                # Handle redirects
                log(f"  Redirect for {url}")
                return None
            else:
                log(f"  HTTP {response.status_code} for {url}")
                if attempt < retries - 1:
                    time.sleep(2)
        except requests.exceptions.Timeout:
            log(f"  Timeout for {url} (attempt {attempt + 1})")
            time.sleep(5)
        except Exception as e:
            log(f"  Request error: {e}")
            time.sleep(2)
    return None


def get_player_tournaments(username: str) -> list[dict]:
    """Get all tournaments for a player."""
    url = f"{API_BASE}/player/{username}/tournaments"
    data = api_get(url)
    if data and "finished" in data:
        return data["finished"]
    return []


def is_titled_tuesday(tournament: dict) -> bool:
    """Check if a tournament is a Titled Tuesday blitz event."""
    tournament_id = tournament.get("@id", "")
    return "titled-tuesday-blitz" in tournament_id.lower()


def get_tournament_date(tournament_id: str) -> datetime | None:
    """Extract date from tournament ID if possible, otherwise return None."""
    # Try to parse date from tournament ID like:
    # "titled-tuesday-blitz-november-11-2025-6031699"
    parts = tournament_id.split("/")[-1].split("-")

    months = {
        "january": 1, "february": 2, "march": 3, "april": 4,
        "may": 5, "june": 6, "july": 7, "august": 8,
        "september": 9, "october": 10, "november": 11, "december": 12
    }

    for i, part in enumerate(parts):
        if part.lower() in months:
            try:
                month = months[part.lower()]
                day = int(parts[i + 1])
                year = int(parts[i + 2])
                return datetime(year, month, day)
            except (IndexError, ValueError):
                pass

    return None


def get_tournament_players(tournament_id: str) -> list[str]:
    """Get all players from a tournament by fetching round data."""
    # First get the tournament info to find round URLs
    data = api_get(tournament_id)
    if not data:
        return []

    players = set()

    # Get players from main tournament data
    if "players" in data:
        for p in data["players"]:
            if isinstance(p, dict) and "username" in p:
                players.add(p["username"].lower())

    # Get players from rounds
    rounds = data.get("rounds", [])
    for round_url in rounds:
        round_data = api_get(round_url)
        if round_data and "players" in round_data:
            for p in round_data["players"]:
                if isinstance(p, dict) and "username" in p:
                    players.add(p["username"].lower())

    return list(players)


def get_player_status(username: str) -> dict:
    """Get a player's profile to check their status and title."""
    url = f"{API_BASE}/player/{username}"
    data = api_get(url)
    if data:
        return {
            "username": username,
            "status": data.get("status", "unknown"),
            "title": data.get("title"),
            "player_id": data.get("player_id"),
            "url": data.get("url"),
        }
    return {
        "username": username,
        "status": "not_found",
        "title": None,
        "player_id": None,
        "url": None,
    }


def main():
    log("=" * 60)
    log("Finding Titled Cheaters from Titled Tuesday Tournaments")
    log("=" * 60)

    # Calculate date cutoff (2 years ago)
    cutoff_date = datetime.now() - timedelta(days=730)
    log(f"\nLooking for tournaments since: {cutoff_date.strftime('%Y-%m-%d')}")

    # Step 1: Find all Titled Tuesday tournaments from sample players
    log(f"\n{'='*60}")
    log("Step 1: Finding Titled Tuesday tournaments")
    log("=" * 60)

    tournament_ids = set()

    for username in SAMPLE_PLAYERS:
        log(f"\nFetching tournaments for {username}...")
        tournaments = get_player_tournaments(username)

        titled_tuesdays = [t for t in tournaments if is_titled_tuesday(t)]
        log(f"  Found {len(titled_tuesdays)} Titled Tuesday tournaments")

        recent_count = 0
        for t in titled_tuesdays:
            t_id = t.get("@id", "")
            t_date = get_tournament_date(t_id)

            # Include if date is recent or if we couldn't parse date
            if t_date is None or t_date >= cutoff_date:
                tournament_ids.add(t_id)
                recent_count += 1

        log(f"  {recent_count} from last 2 years")

    log(f"\nTotal unique Titled Tuesday tournaments: {len(tournament_ids)}")

    # Step 2: Get all players from each tournament
    log(f"\n{'='*60}")
    log("Step 2: Collecting players from tournaments")
    log("=" * 60)

    all_players = set()
    tournament_count = 0

    for t_id in sorted(tournament_ids):
        tournament_count += 1
        # Extract readable name from URL
        t_name = t_id.split("/")[-1]
        log(f"\n[{tournament_count}/{len(tournament_ids)}] {t_name}")

        players = get_tournament_players(t_id)
        log(f"  Players: {len(players)}")
        all_players.update(players)

    log(f"\nTotal unique players across all tournaments: {len(all_players)}")

    # Step 3: Check each player's status
    log(f"\n{'='*60}")
    log("Step 3: Checking player statuses for fair play violations")
    log("=" * 60)

    cheaters = []
    checked_count = 0

    for username in sorted(all_players):
        checked_count += 1
        if checked_count % 100 == 0:
            log(f"  Checked {checked_count}/{len(all_players)} players, found {len(cheaters)} cheaters...")

        player_info = get_player_status(username)

        if "fair_play" in player_info["status"].lower():
            log(f"  FOUND: {username} - {player_info['status']} (title: {player_info['title']})")
            cheaters.append(player_info)

    log(f"\n{'='*60}")
    log("Results")
    log("=" * 60)
    log(f"Total players checked: {len(all_players)}")
    log(f"Players with fair play violations: {len(cheaters)}")

    # Count by title
    title_counts = defaultdict(int)
    for c in cheaters:
        title = c.get("title") or "untitled"
        title_counts[title] += 1

    log("\nBreakdown by title:")
    for title, count in sorted(title_counts.items(), key=lambda x: -x[1]):
        log(f"  {title}: {count}")

    # Step 4: Save to JSON
    output_dir = Path("data/titled-cheaters")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_data = {
        "description": "Titled players caught cheating in Titled Tuesday tournaments",
        "generated_at": datetime.now().isoformat(),
        "tournaments_checked": len(tournament_ids),
        "players_checked": len(all_players),
        "cheaters_found": len(cheaters),
        "title_breakdown": dict(title_counts),
        "players": cheaters,
    }

    output_path = output_dir / "titled_cheaters.json"
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    log(f"\nSaved to: {output_path}")

    # Also save the tournament list for reference
    tournaments_path = output_dir / "tournaments_checked.json"
    with open(tournaments_path, "w") as f:
        json.dump({
            "description": "Titled Tuesday tournaments checked",
            "cutoff_date": cutoff_date.strftime("%Y-%m-%d"),
            "tournaments": sorted(list(tournament_ids)),
        }, f, indent=2)

    log(f"Tournament list saved to: {tournaments_path}")


if __name__ == "__main__":
    main()
