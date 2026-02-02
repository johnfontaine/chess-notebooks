#!/usr/bin/env python3
"""
Game Assessment CLI Tool

Generates a detailed move-by-move analysis report for a single chess game.

Usage:
    python scripts/game_assessment.py USERNAME GAME_ID

Example:
    python scripts/game_assessment.py tryingtolearn1234 147728336594

Output:
    data/game_assessments/{username}_{game_id}/
      - assessment.html (move-by-move visualization)
      - assessment.json (raw data)
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
import statistics

import chess
import chess.pgn
import requests
from jinja2 import Environment, FileSystemLoader

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from chess_analysis import (
    # Engine analysis
    EngineAnalyzer,
    # Metrics
    classify_move_by_cpl,
    centipawns_to_win_percent,
    calculate_move_accuracy,
    # Fragility
    calculate_game_fragility,
    get_fragility_trend,
    FragilityTrend,
    # Openings
    get_opening_book,
    find_last_book_ply,
    calculate_distance_from_book,
    # Game phase
    detect_game_phase,
    # Position assessment
    calculate_pure_material,
    calculate_brute_force_branching,
    get_tablebase_status,
    # Visualization
    render_position,
    # Maia2 analysis
    analyze_position_maia2,
)


# Chess.com API for fetching games
CHESS_COM_API = "https://api.chess.com/pub"


def get_game_type(time_control: str) -> str:
    """
    Determine Maia2 game type from Chess.com time control.

    Time control format: "base+increment" or "base" in seconds.
    - Rapid: base >= 600 (10+ minutes)
    - Blitz: base < 600
    """
    try:
        if "+" in time_control:
            base = int(time_control.split("+")[0])
        else:
            base = int(time_control)

        return "rapid" if base >= 600 else "blitz"
    except:
        return "rapid"  # Default to rapid


def fetch_game_pgn(username: str, game_id: str) -> str | None:
    """Fetch a specific game from Chess.com by game ID."""
    # First, we need to find which archive contains this game
    archives_url = f"{CHESS_COM_API}/player/{username}/games/archives"

    try:
        resp = requests.get(archives_url, headers={"User-Agent": "GameAssessment/1.0"})
        resp.raise_for_status()
        archives = resp.json().get("archives", [])
    except Exception as e:
        print(f"Error fetching archives: {e}")
        return None

    # Search archives in reverse (most recent first)
    for archive_url in reversed(archives):
        try:
            resp = requests.get(archive_url, headers={"User-Agent": "GameAssessment/1.0"})
            resp.raise_for_status()
            games = resp.json().get("games", [])

            for game in games:
                # Extract game ID from URL
                url = game.get("url", "")
                if game_id in url:
                    return game.get("pgn")
        except Exception as e:
            print(f"Error fetching archive {archive_url}: {e}")
            continue

    return None


def parse_game(pgn_text: str) -> chess.pgn.Game | None:
    """Parse PGN text into a chess game."""
    import io
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    return game


def analyze_game(
    game: chess.pgn.Game,
    username: str,
    engine_path: str = "/opt/homebrew/bin/stockfish"
) -> dict:
    """Analyze a game and return comprehensive assessment data."""

    # Extract game metadata
    headers = dict(game.headers)
    white = headers.get("White", "Unknown")
    black = headers.get("Black", "Unknown")
    white_elo = int(headers.get("WhiteElo", 0)) if headers.get("WhiteElo", "").isdigit() else 0
    black_elo = int(headers.get("BlackElo", 0)) if headers.get("BlackElo", "").isdigit() else 0
    result = headers.get("Result", "*")
    date = headers.get("Date", "")
    time_control = headers.get("TimeControl", "")
    termination = headers.get("Termination", "")

    # Determine which color the user is playing
    is_white = white.lower() == username.lower()

    # Build list of positions
    board = game.board()
    positions = [board.copy()]
    moves = []

    for node in game.mainline():
        move = node.move
        san = board.san(move)
        moves.append({
            "move": move,
            "san": san,
            "fen_before": board.fen(),
        })
        board.push(move)
        positions.append(board.copy())
        moves[-1]["fen_after"] = board.fen()

    # Load opening book
    print("Loading opening book...")
    book = get_opening_book()
    last_book_ply, opening_info = find_last_book_ply(positions, book)

    # Calculate fragility for both colors throughout the game
    print("Calculating fragility...")
    white_fragility = calculate_game_fragility(positions, chess.WHITE)
    black_fragility = calculate_game_fragility(positions, chess.BLACK)

    # Analyze each move using engine
    print(f"Analyzing {len(moves)} moves...")
    analyzed_moves = []

    # Determine game type for Maia2
    game_type = get_game_type(time_control)
    print(f"Game type for Maia2: {game_type}")

    # Check if Maia2 is available
    maia2_available = False
    try:
        from maia2 import model, inference
        maia2_available = True
        print("Maia2 models available - will analyze humanness")
    except ImportError:
        print("Maia2 not installed - skipping humanness analysis")

    # Use context manager for engine
    print("Initializing engine...")
    with EngineAnalyzer(engine_path, depth=20) as analyzer:
        for i, move_data in enumerate(moves):
            ply = i + 1
            is_white_move = (ply % 2 == 1)

            board_before = chess.Board(move_data["fen_before"])
            board_after = chess.Board(move_data["fen_after"])
            move = move_data["move"]
            san = move_data["san"]

            print(f"  Move {ply}: {san}")

            # Initialize defaults
            eval_before = None
            eval_after = None
            eval_before_cp = None
            eval_after_cp = None
            cpl = 0
            win_pct_before = None
            win_pct_after = None
            move_accuracy = None
            best_move_san = None
            best_move_uci = None
            move_rank = None
            gap_cp = None
            num_playable = None
            complexity_score = None
            complexity_category = None
            eval_volatility = None
            sf_branching_factor = None
            brute_force_branching = None
            wdl_before = None  # Stockfish WDL (win, draw, loss) in permille
            wdl_after = None
            legal_moves = list(board_before.legal_moves)
            candidate_moves = []  # Engine candidate moves with evals
            maia_moves = []  # Maia2 candidate moves with probabilities

            # Engine analysis
            try:
                # Use multi-depth analysis for before-position to get complexity heuristics
                multi_result = analyzer.analyze_multi_depth_extended(
                    board_before,
                    depths=[6, 10, 14, 20],  # Depths for volatility/branching calculation
                    multipv=2,  # For gap metric
                    capture_search_stats=True,  # For node-based branching factor
                )

                # Extract primary eval at max depth
                max_depth = max(multi_result.depths)
                eval_before = multi_result.evaluations.get(max_depth, 0) / 100.0  # Convert to pawns
                best_move_uci = multi_result.best_moves.get(max_depth)
                if best_move_uci:
                    best_move_obj = chess.Move.from_uci(best_move_uci)
                    best_move_san = board_before.san(best_move_obj)

                # Extract complexity heuristics (includes volatility and branching factor)
                if multi_result.complexity_heuristics:
                    ch = multi_result.complexity_heuristics
                    eval_volatility = ch.eval_volatility
                    sf_branching_factor = ch.branching_factor_estimate
                    complexity_score = ch.complexity_score
                    complexity_category = ch.complexity_category

                # Get WDL from single-depth analysis (multi-depth doesn't capture WDL)
                result_before_single = analyzer.analyze(board_before, depth=20)
                wdl_before = result_before_single.get("wdl")  # (win, draw, loss) in permille

                # Analyze position after move
                result_after = analyzer.analyze(board_after, depth=20)
                eval_after = result_after.get("score", 0) / 100.0  # Convert to pawns
                wdl_after = result_after.get("wdl")  # (win, draw, loss) in permille

                # Calculate CPL (from the side that moved)
                eval_before_cp = int(eval_before * 100)
                eval_after_cp = int(eval_after * 100)

                if is_white_move:
                    cpl = max(0, eval_before_cp - eval_after_cp)
                else:
                    cpl = max(0, eval_after_cp - eval_before_cp)

                # Calculate win percentages (Lichess formula)
                win_pct_before = centipawns_to_win_percent(eval_before_cp)
                win_pct_after = centipawns_to_win_percent(eval_after_cp)

                # Calculate move accuracy (Lichess formula)
                if is_white_move:
                    move_accuracy = calculate_move_accuracy(win_pct_before, win_pct_after)
                else:
                    # For black, we need to flip perspective (100 - win%)
                    move_accuracy = calculate_move_accuracy(100 - win_pct_before, 100 - win_pct_after)

                # Move rank - analyze each legal move at lower depth for speed
                move_evals = []
                for lm in legal_moves[:30]:  # Limit to 30 moves for speed
                    board_copy = board_before.copy()
                    board_copy.push(lm)
                    try:
                        lm_result = analyzer.analyze(board_copy, depth=12)
                        lm_eval = lm_result.get("score", 0)
                        move_evals.append((lm, lm_eval))
                    except:
                        pass

                # Sort by eval (best first for white, worst first for black)
                move_evals.sort(key=lambda x: x[1], reverse=is_white_move)
                for idx, (lm, _) in enumerate(move_evals):
                    if lm == move:
                        move_rank = idx + 1
                        break

                # Store all candidate moves for display
                candidate_moves = []
                for idx, (lm, lm_eval) in enumerate(move_evals):
                    candidate_moves.append({
                        "uci": lm.uci(),
                        "san": board_before.san(lm),
                        "eval": lm_eval / 100.0,  # Convert to pawns
                        "rank": idx + 1,
                        "is_played": lm == move,
                    })

                # Gap to 2nd best
                if len(move_evals) >= 2:
                    gap_cp = abs(move_evals[0][1] - move_evals[1][1])

                # Playable moves (within 50cp of best)
                if move_evals:
                    best_eval_cp = move_evals[0][1]
                    num_playable = sum(1 for _, e in move_evals if abs(e - best_eval_cp) <= 50)

            except Exception as e:
                print(f"    Engine error: {e}")

            # Book status (check early for classification)
            is_book = ply <= last_book_ply

            # Move classification (simple CPL-based)
            if is_book:
                move_class = "book"
            elif cpl is not None:
                move_class = classify_move_by_cpl(cpl)
            else:
                move_class = None

            # Game phase
            game_phase = detect_game_phase(board_before).name.lower()

            # Pure material
            pure_material = calculate_pure_material(board_before)

            # Brute-force branching factor (counts all legal moves at 3-ply depth)
            try:
                bf_result = calculate_brute_force_branching(board_before, depth=3)
                brute_force_branching = bf_result['branching_factor']
            except:
                brute_force_branching = None

            # Fragility
            fragility_analysis = white_fragility if is_white_move else black_fragility
            fragility = fragility_analysis.scores[ply - 1] if ply - 1 < len(fragility_analysis.scores) else 0
            trend, distance = get_fragility_trend(ply - 1, fragility_analysis)
            fragility_trend = trend.value if trend != FragilityTrend.UNKNOWN else None
            distance_to_peak = distance if distance != 0 else None

            # Distance from book (is_book already set above for classification)
            distance_from_book = calculate_distance_from_book(ply, last_book_ply)

            # Tablebase status
            tablebase_status = None
            piece_count = len(board_before.piece_map())
            if piece_count <= 7:
                try:
                    tb_result = get_tablebase_status(board_before)
                    tablebase_status = tb_result
                except:
                    pass

            # Note: Complexity is now calculated via multi-depth analysis above
            # (complexity_score and complexity_category from complexity_heuristics)

            # Maia2 analysis
            humanness = None
            maia_rank = None
            num_human_moves = None
            top_maia_move = None
            top_maia_prob = None
            cp_adjustment = None

            if maia2_available:
                try:
                    # Determine player and opponent ELOs based on who is moving
                    if is_white_move:
                        elo_self = white_elo
                        elo_oppo = black_elo
                    else:
                        elo_self = black_elo
                        elo_oppo = white_elo

                    maia_result = analyze_position_maia2(
                        fen=move_data["fen_before"],
                        played_move=move.uci(),
                        elo_self=elo_self,
                        elo_oppo=elo_oppo,
                        game_type=game_type,
                    )

                    humanness = maia_result.move_probability * 100  # Convert to percentage
                    maia_rank = maia_result.move_rank
                    top_maia_move = maia_result.top_move
                    top_maia_prob = maia_result.top_move_probability * 100

                    # Count moves with >=1% probability
                    num_human_moves = sum(1 for p in maia_result.all_probabilities.values() if p >= 0.01)

                    # Store all Maia2 candidate moves (sorted by probability)
                    # Filter to only legal moves
                    sorted_maia = sorted(
                        maia_result.all_probabilities.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )
                    rank = 0
                    for m_uci, m_prob in sorted_maia:
                        try:
                            m_move = chess.Move.from_uci(m_uci)
                            # Only include legal moves
                            if m_move not in board_before.legal_moves:
                                continue
                            m_san = board_before.san(m_move)
                            rank += 1
                            maia_moves.append({
                                "uci": m_uci,
                                "san": m_san,
                                "prob": m_prob * 100,  # Convert to percentage
                                "rank": rank,
                                "is_played": m_uci == move.uci(),
                            })
                        except:
                            # Skip moves that can't be parsed
                            continue

                    # Calculate CP adjustment (eval difference vs top Maia2 move)
                    # If played move is different from top Maia2 move, calculate the CP difference
                    if top_maia_move and top_maia_move != move.uci() and eval_after_cp is not None:
                        try:
                            # Analyze position after top Maia2 move
                            board_after_maia = board_before.copy()
                            top_maia_move_obj = chess.Move.from_uci(top_maia_move)
                            if top_maia_move_obj in board_after_maia.legal_moves:
                                board_after_maia.push(top_maia_move_obj)
                                maia_result_eval = analyzer.analyze(board_after_maia, depth=12)
                                maia_eval_cp = maia_result_eval.get("score", 0)
                                # CP adjustment: how much worse is played move vs Maia2's top move
                                if is_white_move:
                                    cp_adjustment = eval_after_cp - maia_eval_cp
                                else:
                                    cp_adjustment = maia_eval_cp - eval_after_cp
                        except:
                            pass
                    elif top_maia_move == move.uci():
                        cp_adjustment = 0  # Played the expected human move

                except Exception as e:
                    # Maia2 analysis failed for this position - continue without it
                    pass

            # Generate board SVG
            try:
                svg = render_position(
                    fen=move_data["fen_before"],
                    played_move=move.uci() if move else None,
                    best_move=best_move_uci if best_move_uci and best_move_uci != move.uci() else None,
                    size=250,
                )
            except:
                svg = None

            analyzed_move = {
                "ply": ply,
                "san": san,
                "uci": move.uci(),
                "fen": move_data["fen_after"],  # FEN after the move was played
                "fen_before": move_data["fen_before"],  # FEN before the move (for arrows)
                "is_white": is_white_move,
                "eval_before": eval_before,
                "eval_after": eval_after,
                "win_pct_before": win_pct_before,
                "win_pct_after": win_pct_after,
                "move_accuracy": move_accuracy,
                "cpl": cpl,
                "move_class": move_class,
                "best_move": best_move_san,
                "best_move_uci": best_move_uci,
                "move_rank": move_rank,
                "total_legal_moves": len(legal_moves),
                "candidate_moves": candidate_moves,  # Engine candidate moves with evals
                "gap_cp": gap_cp,
                "num_playable": num_playable,
                "complexity_score": complexity_score,
                "complexity_category": complexity_category,
                "volatility": eval_volatility,
                "sf_branching_factor": sf_branching_factor,
                "brute_force_branching": brute_force_branching,
                "game_phase": game_phase,
                "pure_material": pure_material,
                "fragility": fragility,
                "fragility_trend": fragility_trend,
                "distance_to_peak": distance_to_peak,
                "is_book": is_book,
                "distance_from_book": distance_from_book,
                "tablebase_status": tablebase_status,
                "svg": svg,
                "depth": 20,
                # Stockfish WDL
                "wdl_before": wdl_before,
                "wdl_after": wdl_after,
                # Maia2 metrics
                "humanness": humanness,
                "maia_rank": maia_rank,
                "num_human_moves": num_human_moves,
                "top_maia_move": top_maia_move,
                "top_maia_prob": top_maia_prob,
                "cp_adjustment": cp_adjustment,
                "maia_moves": maia_moves,  # Maia2 candidate moves with probabilities
                "has_trap": False,
                "is_tricky": False,
            }

            analyzed_moves.append(analyzed_move)

    # Calculate player summaries
    white_moves = [m for m in analyzed_moves if m["is_white"]]
    black_moves = [m for m in analyzed_moves if not m["is_white"]]

    def calc_histogram(values, bins):
        """Calculate histogram counts for given bin edges."""
        counts = [0] * (len(bins) - 1)
        for v in values:
            for i in range(len(bins) - 1):
                if bins[i] <= v < bins[i + 1]:
                    counts[i] += 1
                    break
            else:
                # Value >= last bin edge goes in last bucket
                if v >= bins[-1]:
                    counts[-1] += 1
        return counts

    def calc_summary(moves_list, color_result):
        cpls = [m["cpl"] for m in moves_list if m["cpl"] is not None]
        humanness_values = [m["humanness"] for m in moves_list if m["humanness"] is not None]
        accuracy_values = [m["move_accuracy"] for m in moves_list if m.get("move_accuracy") is not None]

        if not cpls:
            return {
                "accuracy": 0,
                "acpl": 0,
                "acpl_stdev": 0,
                "cpl_25": 0,
                "cpl_75": 0,
                "humanness": 0,
                "z_score": 0,
                "result": color_result,
                "cpl_histogram": {"bins": [], "counts": []},
                "accuracy_histogram": {"bins": [], "counts": []},
                "humanness_histogram": {"bins": [], "counts": []},
            }

        acpl = statistics.mean(cpls)
        acpl_stdev = statistics.stdev(cpls) if len(cpls) > 1 else 0
        cpls_sorted = sorted(cpls)
        cpl_25 = cpls_sorted[len(cpls_sorted) // 4] if cpls_sorted else 0
        cpl_75 = cpls_sorted[3 * len(cpls_sorted) // 4] if cpls_sorted else 0

        # Simple accuracy approximation
        accuracy = max(0, 100 - acpl / 2)

        # Average humanness from Maia2
        avg_humanness = statistics.mean(humanness_values) if humanness_values else 0

        # Calculate histograms for distribution charts
        # CPL bins: 0, 10, 25, 50, 100, 200, 500+
        cpl_bins = [0, 10, 25, 50, 100, 200, 500, 10000]
        cpl_labels = ["0-10", "10-25", "25-50", "50-100", "100-200", "200-500", "500+"]
        cpl_counts = calc_histogram(cpls, cpl_bins)

        # Accuracy bins: 0-50, 50-70, 70-85, 85-95, 95-100
        acc_bins = [0, 50, 70, 85, 95, 101]
        acc_labels = ["<50", "50-70", "70-85", "85-95", "95-100"]
        acc_counts = calc_histogram(accuracy_values, acc_bins) if accuracy_values else [0] * len(acc_labels)

        # Humanness bins: 0-1, 1-5, 5-15, 15-30, 30-50, 50+
        human_bins = [0, 1, 5, 15, 30, 50, 101]
        human_labels = ["<1%", "1-5%", "5-15%", "15-30%", "30-50%", "50%+"]
        human_counts = calc_histogram(humanness_values, human_bins) if humanness_values else [0] * len(human_labels)

        return {
            "accuracy": accuracy,
            "acpl": acpl,
            "acpl_stdev": acpl_stdev,
            "cpl_25": int(cpl_25),
            "cpl_75": int(cpl_75),
            "humanness": avg_humanness,
            "z_score": 0,  # Would need baseline comparison
            "result": color_result,
            "cpl_histogram": {"labels": cpl_labels, "counts": cpl_counts},
            "accuracy_histogram": {"labels": acc_labels, "counts": acc_counts},
            "humanness_histogram": {"labels": human_labels, "counts": human_counts},
        }

    # Determine results
    if result == "1-0":
        white_result, black_result = "win", "loss"
    elif result == "0-1":
        white_result, black_result = "loss", "win"
    else:
        white_result, black_result = "draw", "draw"

    white_summary = calc_summary(white_moves, white_result)
    black_summary = calc_summary(black_moves, black_result)

    # Collect chart data (indexed by move number, not ply)
    chart_data = {
        "moves": [],  # Move numbers (1, 2, 3, ...)
        "white_win_pct": [],
        "black_win_pct": [],
        "white_wdl_win": [],
        "white_wdl_draw": [],
        "white_wdl_loss": [],
        "black_wdl_win": [],
        "black_wdl_draw": [],
        "black_wdl_loss": [],
        "white_accuracy": [],
        "black_accuracy": [],
        "white_humanness": [],
        "black_humanness": [],
        "fragility": [],  # Position metric - one value per move
        "complexity": [],  # Position metric - one value per move
    }

    # Phase transition move numbers
    phase_transitions = {
        "book_end": (last_book_ply + 1) // 2 if last_book_ply else None,
        "opening_end": None,
        "middlegame_end": None,
    }

    last_phase = None
    for m in analyzed_moves:
        ply = m["ply"]
        move_num = (ply + 1) // 2  # Convert ply to move number

        # Only add move number once per move (on white's turn)
        if m["is_white"]:
            chart_data["moves"].append(move_num)

        # Detect phase transitions (convert to move numbers)
        current_phase = m.get("game_phase", "").lower()
        if last_phase == "opening" and current_phase == "middlegame":
            phase_transitions["opening_end"] = (ply - 1 + 1) // 2
        elif last_phase == "middlegame" and current_phase == "endgame":
            phase_transitions["middlegame_end"] = (ply - 1 + 1) // 2
        last_phase = current_phase

        # Win percentages (from white's perspective) - capture on white's turn
        if m["is_white"]:
            wp_before = m.get("win_pct_before")
            chart_data["white_win_pct"].append(wp_before if wp_before is not None else None)
            chart_data["black_win_pct"].append(100 - wp_before if wp_before is not None else None)

            # WDL from Stockfish
            wdl = m.get("wdl_before")
            if wdl:
                chart_data["white_wdl_win"].append(wdl[0] / 10)
                chart_data["white_wdl_draw"].append(wdl[1] / 10)
                chart_data["white_wdl_loss"].append(wdl[2] / 10)
                chart_data["black_wdl_win"].append(wdl[2] / 10)
                chart_data["black_wdl_draw"].append(wdl[1] / 10)
                chart_data["black_wdl_loss"].append(wdl[0] / 10)
            else:
                chart_data["white_wdl_win"].append(None)
                chart_data["white_wdl_draw"].append(None)
                chart_data["white_wdl_loss"].append(None)
                chart_data["black_wdl_win"].append(None)
                chart_data["black_wdl_draw"].append(None)
                chart_data["black_wdl_loss"].append(None)

            # Fragility and Complexity (position metrics - capture once per move)
            frag = m.get("fragility")
            chart_data["fragility"].append(frag)

            comp = m.get("complexity_score")
            chart_data["complexity"].append(comp * 100 if comp is not None else None)

        # Accuracy per move (for player who moved)
        acc = m.get("move_accuracy")
        if m["is_white"]:
            chart_data["white_accuracy"].append(acc)
        else:
            # Append to existing move's black accuracy
            if chart_data["black_accuracy"] and len(chart_data["black_accuracy"]) < len(chart_data["moves"]):
                chart_data["black_accuracy"].append(acc)
            elif len(chart_data["black_accuracy"]) == len(chart_data["moves"]) - 1:
                chart_data["black_accuracy"].append(acc)

        # Humanness per move
        human = m.get("humanness")
        if m["is_white"]:
            chart_data["white_humanness"].append(human)
        else:
            if len(chart_data["black_humanness"]) < len(chart_data["moves"]):
                chart_data["black_humanness"].append(human)

    # Ensure black arrays match white arrays in length (pad with None for last move if needed)
    while len(chart_data["black_accuracy"]) < len(chart_data["moves"]):
        chart_data["black_accuracy"].append(None)
    while len(chart_data["black_humanness"]) < len(chart_data["moves"]):
        chart_data["black_humanness"].append(None)

    # Organize moves into pairs for template
    move_pairs = []
    for i in range(0, len(analyzed_moves), 2):
        pair = {
            "number": (i // 2) + 1,
            "white": analyzed_moves[i] if i < len(analyzed_moves) else None,
            "black": analyzed_moves[i + 1] if i + 1 < len(analyzed_moves) else None,
        }
        move_pairs.append(pair)

    # Calculate phase ply counts
    opening_plies = 0
    middlegame_plies = 0
    endgame_plies = 0
    book_plies = last_book_ply if last_book_ply else 0

    for m in analyzed_moves:
        phase = m.get("game_phase", "").lower()
        if phase == "opening":
            opening_plies += 1
        elif phase == "middlegame":
            middlegame_plies += 1
        elif phase == "endgame":
            endgame_plies += 1

    total_plies = len(analyzed_moves)
    phase_ply_counts = {
        "opening_plies": opening_plies,
        "middlegame_plies": middlegame_plies,
        "endgame_plies": endgame_plies,
        "book_plies": book_plies,
        "total_plies": total_plies,
        # Also include move counts (plies / 2, rounded up)
        "opening_moves": (opening_plies + 1) // 2,
        "middlegame_moves": (middlegame_plies + 1) // 2,
        "endgame_moves": (endgame_plies + 1) // 2,
        "book_moves": (book_plies + 1) // 2,
        "total_moves": (total_plies + 1) // 2,
    }

    return {
        "game": {
            "game_id": "",  # Will be set by caller
            "white": white,
            "black": black,
            "white_elo": white_elo,
            "black_elo": black_elo,
            "result": result,
            "date": date,
            "time_control": time_control,
            "termination": termination,
        },
        "opening": {
            "eco": opening_info.eco if opening_info else None,
            "name": opening_info.name if opening_info else None,
        } if opening_info else None,
        "last_book_ply": last_book_ply,
        "white_summary": white_summary,
        "black_summary": black_summary,
        "moves": move_pairs,
        "all_moves": analyzed_moves,
        "chart_data": chart_data,
        "phase_transitions": phase_transitions,
        "phase_ply_counts": phase_ply_counts,
    }


def render_report(data: dict, output_dir: Path) -> None:
    """Render the HTML report using Jinja2 template."""
    template_dir = Path(__file__).parent.parent / "fairness_report" / "templates"
    env = Environment(loader=FileSystemLoader(str(template_dir)))
    template = env.get_template("game_assessment.html.jinja2")

    html = template.render(**data)

    output_file = output_dir / "assessment.html"
    output_file.write_text(html)
    print(f"HTML report saved to: {output_file}")


def save_json(data: dict, output_dir: Path) -> None:
    """Save raw data as JSON."""
    # Remove SVG from JSON (too large)
    json_data = {
        "game": data["game"],
        "opening": data["opening"],
        "last_book_ply": data["last_book_ply"],
        "white_summary": data["white_summary"],
        "black_summary": data["black_summary"],
        "phase_transitions": data.get("phase_transitions", {}),
        "phase_ply_counts": data.get("phase_ply_counts", {}),
        "chart_data": data.get("chart_data", {}),
        "moves": [
            {
                "ply": m["ply"],
                "san": m["san"],
                "uci": m["uci"],
                "fen_before": m.get("fen_before"),
                "is_white": m["is_white"],
                "eval_before": m["eval_before"],
                "eval_after": m["eval_after"],
                "win_pct_before": m.get("win_pct_before"),
                "win_pct_after": m.get("win_pct_after"),
                "move_accuracy": m.get("move_accuracy"),
                "cpl": m["cpl"],
                "move_class": m["move_class"],
                "best_move": m["best_move"],
                "best_move_uci": m.get("best_move_uci"),
                "move_rank": m["move_rank"],
                "total_legal_moves": m["total_legal_moves"],
                "candidate_moves": m.get("candidate_moves", []),
                "gap_cp": m["gap_cp"],
                "num_playable": m["num_playable"],
                "complexity_score": m["complexity_score"],
                "complexity_category": m["complexity_category"],
                "game_phase": m["game_phase"],
                "fragility": m["fragility"],
                "is_book": m["is_book"],
                # WDL from Stockfish
                "wdl_before": m.get("wdl_before"),
                "wdl_after": m.get("wdl_after"),
                # Maia2 metrics
                "humanness": m.get("humanness"),
                "maia_rank": m.get("maia_rank"),
                "num_human_moves": m.get("num_human_moves"),
                "top_maia_move": m.get("top_maia_move"),
                "top_maia_prob": m.get("top_maia_prob"),
                "cp_adjustment": m.get("cp_adjustment"),
                "maia_moves": m.get("maia_moves", []),
            }
            for m in data["all_moves"]
        ],
    }

    output_file = output_dir / "assessment.json"
    output_file.write_text(json.dumps(json_data, indent=2))
    print(f"JSON data saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate a detailed game assessment report",
        epilog="Example: python scripts/game_assessment.py tryingtolearn1234 147728336594"
    )
    parser.add_argument("username", help="Chess.com username")
    parser.add_argument("game_id", help="Chess.com game ID")
    parser.add_argument(
        "--engine",
        default="/opt/homebrew/bin/stockfish",
        help="Path to Stockfish engine"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: data/game_assessments/{username}_{game_id})"
    )

    args = parser.parse_args()

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).parent.parent / "data" / "game_assessments" / f"{args.username}_{args.game_id}"

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching game {args.game_id} for {args.username}...")
    pgn_text = fetch_game_pgn(args.username, args.game_id)

    if not pgn_text:
        print(f"Error: Could not find game {args.game_id}")
        sys.exit(1)

    print("Parsing game...")
    game = parse_game(pgn_text)

    if not game:
        print("Error: Could not parse game PGN")
        sys.exit(1)

    print("Analyzing game...")
    data = analyze_game(game, args.username, args.engine)
    data["game"]["game_id"] = args.game_id

    # Add report metadata
    data["report_generated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data["report_username"] = args.username

    print("Generating report...")
    render_report(data, output_dir)
    save_json(data, output_dir)

    print(f"\nDone! Report saved to: {output_dir}")


if __name__ == "__main__":
    main()
