#!/usr/bin/env python3
"""
Generate baseline dataset from trusted (non-cheating) players.

This script fetches games from players who are presumed to be fair players
and generates a baseline dataset for comparison. The baseline helps establish
what "normal" patterns look like across various metrics.

Usage:
    python scripts/generate_baseline.py --players player1,player2 --output data/trusted/
    python scripts/generate_baseline.py --config baseline_config.json

The trusted player list should include players across different Elo ranges
to capture natural variation in playing patterns.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from chess_analysis import (
    # Caching
    get_shared_opponent_cache_path,
    migrate_opponent_cache_if_needed,
    init_opponent_cache,
    # Baseline generation
    generate_player_baseline,
    generate_combined_baseline,
)


# =============================================================================
# Configuration
# =============================================================================

# Example baseline configuration
EXAMPLE_CONFIG = {
    "description": "Baseline dataset from trusted players",
    "created": None,  # Will be filled in
    "time_classes": ["blitz", "rapid", "bullet"],
    "days_back": None,  # None = fetch all available history
    "max_games": 1000,  # Maximum games per time class per player
    "validate_opponents": True,
    "players": [
        # Add players like:
        # {"username": "player1", "elo_range": "1000-1200", "note": "friend, known fair player"},
        # {"username": "player2", "elo_range": "1400-1600", "note": "club player, verified"},
    ],
}


def create_example_config(output_path: Path):
    """Create an example configuration file."""
    config = EXAMPLE_CONFIG.copy()
    config["created"] = datetime.now().isoformat()

    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Created example config at: {output_path}")
    print("\nEdit this file to add your trusted players, then run:")
    print(f"  python scripts/generate_baseline.py --config {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate baseline dataset from trusted players",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Create example config file
    python scripts/generate_baseline.py --create-config baseline_config.json

    # Generate baseline from config
    python scripts/generate_baseline.py --config baseline_config.json

    # Generate baseline for specific players (all history)
    python scripts/generate_baseline.py --players friend1,friend2 --output data/trusted/

    # Generate baseline with limited history (30 days)
    python scripts/generate_baseline.py --players friend1 --days 30

    # Skip opponent validation for faster runs
    python scripts/generate_baseline.py --players friend1 --skip-opponent-validation
        """,
    )

    parser.add_argument(
        "--config",
        type=Path,
        help="Path to JSON config file with player list",
    )
    parser.add_argument(
        "--players",
        type=str,
        help="Comma-separated list of trusted player usernames",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/trusted"),
        help="Output directory for baseline data",
    )
    parser.add_argument(
        "--time-classes",
        type=str,
        default="blitz,rapid,bullet",
        help="Comma-separated time classes to include (default: blitz,rapid,bullet)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="Number of days of history to fetch (default: all available)",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=None,
        help="Maximum games per time class to process (e.g., 1000 rapid + 1000 blitz)",
    )
    parser.add_argument(
        "--skip-opponent-validation",
        action="store_true",
        help="Skip fetching opponent profiles (faster but no trust filtering)",
    )
    parser.add_argument(
        "--create-config",
        type=Path,
        help="Create example config file at specified path",
    )

    args = parser.parse_args()

    # Handle create-config option
    if args.create_config:
        create_example_config(args.create_config)
        return

    # Determine player list and settings
    players = []
    validate_opponents = not args.skip_opponent_validation

    if args.config:
        if not args.config.exists():
            print(f"Error: Config file not found: {args.config}")
            sys.exit(1)

        with open(args.config) as f:
            config = json.load(f)

        players = [p["username"] for p in config.get("players", [])]
        time_classes = config.get("time_classes", args.time_classes.split(","))
        days_back = config.get("days_back", args.days)
        # Handle empty string as None (fetch all history)
        if days_back == "":
            days_back = None
        max_games = config.get("max_games", args.max_games)
        if max_games is None:
            max_games = 1000  # Default to 1000 games per time class
        use_last_game_date = config.get("use_last_game_date", False)
        validate_opponents = config.get("validate_opponents", validate_opponents)
    elif args.players:
        players = [p.strip() for p in args.players.split(",")]
        time_classes = args.time_classes.split(",")
        days_back = args.days
        max_games = args.max_games if args.max_games else 1000
        use_last_game_date = False
    else:
        print("Error: Must specify --config or --players")
        print("Use --create-config to create an example configuration file")
        sys.exit(1)

    if not players:
        print("Error: No players specified")
        sys.exit(1)

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Initialize opponent cache if validating
    # Uses shared cache at data/opponent_cache.db (migrated from baseline-specific location)
    opponent_cache_conn = None
    if validate_opponents:
        old_cache_path = args.output / "opponent_cache.db"
        new_cache_path = get_shared_opponent_cache_path()

        # Migrate from old location if needed
        if migrate_opponent_cache_if_needed(old_cache_path, new_cache_path):
            print(f"Migrated opponent cache to shared location: {new_cache_path}")

        print(f"Initializing opponent cache: {new_cache_path}")
        opponent_cache_conn = init_opponent_cache(new_cache_path)

    print(f"\nGenerating baseline for {len(players)} players")
    print(f"Time classes: {time_classes}")
    if days_back is None:
        print("History: ALL available")
    else:
        print(f"Days back: {days_back}")
    print(f"Max games per time class: {max_games}")
    print(f"Validate opponents: {validate_opponents}")
    print(f"Output: {args.output}")
    if use_last_game_date:
        print("Mode: Banned account (using last game date as end date)")

    # Process each player
    summaries = []
    for username in players:
        try:
            summary = generate_player_baseline(
                username=username,
                output_dir=args.output,
                time_classes=time_classes,
                days_back=days_back,
                max_games=max_games,
                use_last_game_date=use_last_game_date,
                validate_opponents=validate_opponents,
                opponent_cache_conn=opponent_cache_conn,
            )
            summaries.append(summary)
        except Exception as e:
            print(f"Error processing {username}: {e}")
            import traceback
            traceback.print_exc()
            summaries.append({"username": username, "status": "error", "error": str(e)})

    # Close opponent cache
    if opponent_cache_conn:
        opponent_cache_conn.close()

    # Generate combined baseline
    generate_combined_baseline(summaries, args.output)

    print("\n" + "="*60)
    print("Baseline generation complete!")
    print(f"Output directory: {args.output}")


if __name__ == "__main__":
    main()
