#!/usr/bin/env python3
"""
Stockfish evaluation timeline logger.

Logs how Stockfish's lines and evaluations evolve as search depth increases.
Captures every info update from the engine, showing when specific moves first
appear and how evaluations shift over time.

Usage:
    python scripts/stockfish_timeline.py [FEN] [--depth DEPTH] [--multipv N] [--threads N] [--hash N]

Example:
    python scripts/stockfish_timeline.py \
        "rnb1k1nr/1p1p3p/1qpP2p1/p7/1PP5/P1BB1Q2/5PPP/R3K2R w KQ - 2 18" \
        --depth 30 --multipv 5
"""

import argparse
import csv
import io
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chess
import chess.engine
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

DEFAULT_FEN = "rnb1k1nr/1p1p3p/1qpP2p1/p7/1PP5/P1BB1Q2/5PPP/R3K2R w KQ - 2 18"


def format_score(score: chess.engine.PovScore, board: chess.Board) -> str:
    """Format a score from the side-to-move's perspective."""
    white = score.white()
    if white.is_mate():
        return f"M{white.mate():+d}"
    return f"{white.score() / 100:+.2f}"


def format_pv(board: chess.Board, pv: list[chess.Move], max_moves: int = 8) -> str:
    """Format a principal variation in SAN notation."""
    san_moves = []
    b = board.copy()
    for move in pv[:max_moves]:
        try:
            san_moves.append(b.san(move))
            b.push(move)
        except Exception:
            break
    if len(pv) > max_moves:
        san_moves.append("...")
    return " ".join(san_moves)


def format_move(board: chess.Board, move: chess.Move) -> str:
    """Format a single move in SAN."""
    return board.san(move)


def run_timeline(
    fen: str,
    max_depth: int = 30,
    multipv: int = 5,
    threads: int = 1,
    hash_mb: int = 256,
    track_move: str | None = None,
    output_csv: str | None = None,
):
    """Run the evaluation timeline analysis."""
    stockfish_path = os.environ.get("STOCKFISH_PATH", "stockfish")

    board = chess.Board(fen)
    print(f"Position: {fen}")
    print(f"Side to move: {'White' if board.turn == chess.WHITE else 'Black'}")
    print(f"Search depth: {max_depth}, MultiPV: {multipv}")
    print()

    # If tracking a specific move, resolve it
    track_uci = None
    if track_move:
        try:
            parsed = board.parse_san(track_move)
            track_uci = parsed.uci()
            print(f"Tracking move: {track_move} ({track_uci})")
        except ValueError:
            # Try as UCI
            try:
                parsed = chess.Move.from_uci(track_move)
                if parsed in board.legal_moves:
                    track_uci = track_move
                    track_move = board.san(parsed)
                    print(f"Tracking move: {track_move} ({track_uci})")
                else:
                    print(f"Warning: {track_move} is not legal in this position")
            except ValueError:
                print(f"Warning: Could not parse move '{track_move}'")
        print()

    # Print board
    print(board)
    print()

    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    engine.configure({
        "Threads": threads,
        "Hash": hash_mb,
        "UCI_ShowWDL": True,
    })

    # Print engine identity
    engine_name = engine.id.get("name", "Unknown")
    print(f"Engine: {engine_name}")
    print(f"Threads: {threads}, Hash: {hash_mb}MB")
    print()

    # Storage for timeline entries
    timeline = []
    seen_depths = set()
    track_appearances = []

    header = f"{'Depth':>5} {'PV#':>3} {'Score':>8} {'Nodes':>12} {'Time':>8} {'Best Move':<12} {'Principal Variation'}"
    separator = "-" * 120

    print(header)
    print(separator)

    start_time = time.time()

    try:
        with engine.analysis(
            board,
            chess.engine.Limit(depth=max_depth),
            multipv=multipv,
        ) as analysis:
            for info in analysis:
                depth = info.get("depth")
                seldepth = info.get("seldepth")
                multipv_idx = info.get("multipv", 1)
                score = info.get("score")
                pv = info.get("pv", [])
                nodes = info.get("nodes", 0)
                elapsed = info.get("time", 0)
                nps = info.get("nps", 0)

                if depth is None or score is None or not pv:
                    continue

                best_move_san = format_move(board, pv[0])
                pv_str = format_pv(board, pv)
                score_str = format_score(score, board)

                entry = {
                    "engine": engine_name,
                    "depth": depth,
                    "seldepth": seldepth,
                    "pv_rank": multipv_idx,
                    "score": score_str,
                    "best_move": best_move_san,
                    "pv": pv_str,
                    "nodes": nodes,
                    "time_s": round(elapsed, 3),
                    "nps": nps,
                    "wall_time": round(time.time() - start_time, 3),
                }
                timeline.append(entry)

                # Check if tracked move appears
                is_tracked = False
                if track_uci and pv[0].uci() == track_uci:
                    is_tracked = True
                    track_appearances.append(entry)

                # Print with highlight if tracked move
                marker = " <<<" if is_tracked else ""
                print(
                    f"{depth:>5} {multipv_idx:>3} {score_str:>8} {nodes:>12,} {elapsed:>7.2f}s {best_move_san:<12} {pv_str}{marker}"
                )

                # Stop after reaching max depth on the last PV line
                if depth >= max_depth and multipv_idx >= multipv:
                    break

    except KeyboardInterrupt:
        print("\n[Interrupted]")
    finally:
        engine.quit()

    total_time = time.time() - start_time
    print(separator)
    print(f"Total wall time: {total_time:.2f}s")
    print()

    # Summary for tracked move
    if track_move and track_appearances:
        print(f"=== Timeline for {track_move} ===")
        print()
        print(f"{'Depth':>5} {'Rank':>4} {'Score':>8} {'PV'}")
        print("-" * 80)
        for e in track_appearances:
            print(f"{e['depth']:>5} {e['pv_rank']:>4} {e['score']:>8} {e['pv']}")
        print()
        first = track_appearances[0]
        print(f"First appearance: depth {first['depth']} as PV #{first['pv_rank']} (score: {first['score']})")
        best_rank = min(e["pv_rank"] for e in track_appearances)
        best_at = [e for e in track_appearances if e["pv_rank"] == best_rank]
        if best_rank == 1:
            depths_at_top = [e["depth"] for e in track_appearances if e["pv_rank"] == 1]
            print(f"Reached #1 at depth: {depths_at_top[0]}")
            print(f"Was #1 at depths: {depths_at_top}")
        else:
            print(f"Best rank achieved: #{best_rank} (at depth {best_at[0]['depth']})")
    elif track_move:
        print(f"=== {track_move} never appeared in top {multipv} lines ===")
        print()

    # Final position summary
    final_by_rank = {}
    for e in reversed(timeline):
        max_d = max(e2["depth"] for e2 in timeline)
        if e["depth"] == max_d and e["pv_rank"] not in final_by_rank:
            final_by_rank[e["pv_rank"]] = e

    if final_by_rank:
        print(f"=== Final evaluation at depth {max_d} ===")
        print()
        for rank in sorted(final_by_rank):
            e = final_by_rank[rank]
            print(f"  #{rank}: {e['score']:>8}  {e['pv']}")
        print()

    # CSV export
    if output_csv:
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "engine", "depth", "seldepth", "pv_rank", "score", "best_move",
                "pv", "nodes", "time_s", "nps", "wall_time",
            ])
            writer.writeheader()
            writer.writerows(timeline)
        print(f"Timeline saved to {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Log Stockfish evaluation timeline for a position"
    )
    parser.add_argument(
        "fen",
        nargs="?",
        default=DEFAULT_FEN,
        help="FEN string of the position to analyze",
    )
    parser.add_argument(
        "--depth", type=int, default=30,
        help="Maximum search depth (default: 30)",
    )
    parser.add_argument(
        "--multipv", type=int, default=5,
        help="Number of principal variations to track (default: 5)",
    )
    parser.add_argument(
        "--threads", type=int, default=None,
        help="Number of threads (default: from .env or 1)",
    )
    parser.add_argument(
        "--hash", type=int, default=None,
        help="Hash table size in MB (default: from .env or 256)",
    )
    parser.add_argument(
        "--track", type=str, default=None,
        help="Track a specific move (SAN or UCI), e.g. 'Bxg6+' or 'd3g6'",
    )
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Export timeline to CSV file",
    )

    args = parser.parse_args()

    threads = args.threads or int(os.environ.get("STOCKFISH_THREADS", "1"))
    hash_mb = args.hash or int(os.environ.get("STOCKFISH_HASH", "256"))

    run_timeline(
        fen=args.fen,
        max_depth=args.depth,
        multipv=args.multipv,
        threads=threads,
        hash_mb=hash_mb,
        track_move=args.track,
        output_csv=args.csv,
    )


if __name__ == "__main__":
    main()
