#!/usr/bin/env python3
"""
CLI runner for fairness analysis.

Usage:
    python run_analysis.py USERNAME [OPTIONS]

Examples:
    python run_analysis.py hikaru
    python run_analysis.py magnus --days-back 90 --time-classes rapid blitz
    python run_analysis.py player123 --skip-engine  # Skip slow engine analysis
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(
        description="Run fairness analysis on a Chess.com player",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_analysis.py hikaru
    python run_analysis.py magnus --days-back 90
    python run_analysis.py player123 --skip-engine --skip-maia2
        """
    )

    parser.add_argument("username", help="Chess.com username to analyze")
    parser.add_argument("--days-back", type=int, default=None,
                        help="Days of history to analyze (default: all)")
    parser.add_argument("--time-classes", nargs="+", default=["rapid", "blitz"],
                        help="Time controls to include (default: rapid blitz)")
    parser.add_argument("--max-games", type=int, default=50,
                        help="Maximum games for deep analysis (default: 50)")
    parser.add_argument("--engine-threads", type=int, default=4,
                        help="CPU threads for Stockfish (default: 4)")
    parser.add_argument("--engine-hash", type=int, default=512,
                        help="Hash table size in MB (default: 512)")
    parser.add_argument("--skip-engine", action="store_true",
                        help="Skip engine analysis (Phase 4a)")
    parser.add_argument("--skip-regan", action="store_true",
                        help="Skip Regan analysis (Phase 4b)")
    parser.add_argument("--skip-tablebase", action="store_true",
                        help="Skip tablebase analysis (Phase 4c)")
    parser.add_argument("--skip-maia2", action="store_true",
                        help="Skip Maia2 analysis (Phase 6)")
    parser.add_argument("--no-validate-opponents", action="store_true",
                        help="Skip opponent ban status validation")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Custom output directory")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")

    args = parser.parse_args()

    # Check for papermill
    try:
        import papermill as pm
    except ImportError:
        print("ERROR: Papermill is required. Install with: pip install papermill")
        sys.exit(1)

    # Setup paths
    username = args.username.lower()
    phases_dir = Path(__file__).parent / "phases"

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = PROJECT_ROOT / "data" / "other-users" / username / "report"

    executed_dir = output_dir / "executed_notebooks"
    executed_dir.mkdir(parents=True, exist_ok=True)

    print(f"="*60)
    print(f"Fairness Analysis: {args.username}")
    print(f"="*60)
    print(f"Output: {output_dir}")
    print(f"Time classes: {args.time_classes}")
    print(f"Days back: {args.days_back or 'All history'}")
    print()

    def run_phase(name: str, notebook: str, parameters: dict):
        """Run a single phase notebook."""
        input_path = phases_dir / notebook
        output_path = executed_dir / notebook

        if not input_path.exists():
            print(f"ERROR: {input_path} not found")
            return False

        print(f"\n{'-'*40}")
        print(f"Phase: {name}")
        print(f"{'-'*40}")

        if args.verbose:
            print(f"Parameters: {parameters}")

        try:
            pm.execute_notebook(
                str(input_path),
                str(output_path),
                parameters=parameters,
                cwd=str(phases_dir),  # Set working directory to phases/ so '../common.py' works
            )
            print(f"✓ {name} complete")
            return True
        except Exception as e:
            print(f"✗ {name} failed: {e}")
            return False

    # Build report options string for display in final report
    report_options = []
    if args.days_back is not None:
        report_options.append(f"--days-back {args.days_back}")
    if args.time_classes != ["rapid", "blitz"]:
        report_options.append(f"--time-classes {' '.join(args.time_classes)}")
    if args.max_games != 50:
        report_options.append(f"--max-games {args.max_games}")
    if args.engine_threads != 4:
        report_options.append(f"--engine-threads {args.engine_threads}")
    if args.engine_hash != 512:
        report_options.append(f"--engine-hash {args.engine_hash}")
    if args.skip_engine:
        report_options.append("--skip-engine")
    if args.skip_regan:
        report_options.append("--skip-regan")
    if args.skip_tablebase:
        report_options.append("--skip-tablebase")
    if args.skip_maia2:
        report_options.append("--skip-maia2")
    if args.no_validate_opponents:
        report_options.append("--no-validate-opponents")

    report_options_str = " ".join(report_options) if report_options else "Default"

    # Run phases
    phases_run = 0
    phases_failed = 0

    # Phase 1: Data Collection
    phase1_params = {
        "username": args.username,
        "time_classes": args.time_classes,
        "validate_opponents": not args.no_validate_opponents,
    }
    # Only pass days_back if explicitly set (avoid passing None)
    if args.days_back is not None:
        phase1_params["days_back"] = args.days_back

    if run_phase("Data Collection", "01_data_collection.ipynb", phase1_params):
        phases_run += 1
    else:
        phases_failed += 1
        print("ERROR: Data collection failed. Cannot continue.")
        sys.exit(1)

    # Phase 2: Quick Analysis
    if run_phase("Quick Analysis", "02_quick_analysis.ipynb", {
        "username": args.username,
    }):
        phases_run += 1
    else:
        phases_failed += 1

    # Phase 3: Game Prioritization
    if run_phase("Game Prioritization", "03_game_prioritization.ipynb", {
        "username": args.username,
        "min_suspicion_score": 3,
        "max_games_to_analyze": args.max_games,
    }):
        phases_run += 1
    else:
        phases_failed += 1

    # Phase 4a: Engine Analysis
    if not args.skip_engine:
        if run_phase("Engine Analysis", "04a_engine_analysis.ipynb", {
            "username": args.username,
            "analysis_depths": [5, 10, 20],
            "engine_threads": args.engine_threads,
            "engine_hash_mb": args.engine_hash,
        }):
            phases_run += 1
        else:
            phases_failed += 1
    else:
        print("\nSkipping Engine Analysis (--skip-engine)")

    # Phase 4b: Regan Analysis
    if not args.skip_regan and not args.skip_engine:
        if run_phase("Regan Analysis", "04b_regan_analysis.ipynb", {
            "username": args.username,
            "z_score_threshold": 2.0,
        }):
            phases_run += 1
        else:
            phases_failed += 1
    else:
        print("\nSkipping Regan Analysis")

    # Phase 4c: Tablebase Analysis
    if not args.skip_tablebase:
        if run_phase("Tablebase Analysis", "04c_tablebase_analysis.ipynb", {
            "username": args.username,
            "max_pieces": 7,
        }):
            phases_run += 1
        else:
            phases_failed += 1
    else:
        print("\nSkipping Tablebase Analysis (--skip-tablebase)")

    # Phase 5: Time Analysis
    if run_phase("Time Analysis", "05_time_analysis.ipynb", {
        "username": args.username,
    }):
        phases_run += 1
    else:
        phases_failed += 1

    # Phase 6: Maia2 Analysis
    if not args.skip_maia2:
        if run_phase("Maia2 Analysis", "06_maia2_analysis.ipynb", {
            "username": args.username,
        }):
            phases_run += 1
        else:
            phases_failed += 1
    else:
        print("\nSkipping Maia2 Analysis (--skip-maia2)")

    # Phase 7: Cheater Comparison
    if run_phase("Cheater Comparison", "07_cheater_comparison.ipynb", {
        "username": args.username,
    }):
        phases_run += 1
    else:
        phases_failed += 1

    # Phase 8: Final Report
    if run_phase("Final Report", "08_final_report.ipynb", {
        "username": args.username,
        "report_title": f"Fairness Analysis: {args.username}",
        "report_options": report_options_str,
    }):
        phases_run += 1
    else:
        phases_failed += 1

    # Summary
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Phases run: {phases_run}")
    print(f"Phases failed: {phases_failed}")

    report_path = output_dir / "report.html"
    if report_path.exists():
        print(f"\nReport: {report_path}")
    else:
        print(f"\nReport not generated (check executed notebooks for errors)")

    return 0 if phases_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
