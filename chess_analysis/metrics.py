"""
Chess fairness metrics calculation.

Based on centipawn loss methodology used in cheat detection research.
Accuracy calculation uses the Lichess formula:
https://lichess.org/page/accuracy
"""

import math
from typing import Optional


def centipawns_to_win_percent(centipawns: int) -> float:
    """
    Convert centipawn evaluation to win percentage using Lichess formula.

    Formula: winPercent = 50 + 50 * (2 / (1 + exp(-0.00368208 * centipawns)) - 1)

    This is based on statistical analysis of millions of games correlating
    engine evaluations with actual game outcomes.

    Args:
        centipawns: Position evaluation in centipawns (from side-to-move perspective).

    Returns:
        Win probability as a percentage (0-100).
    """
    return 50 + 50 * (2 / (1 + math.exp(-0.00368208 * centipawns)) - 1)


def mate_to_centipawns(mate_in_n: int) -> int:
    """
    Convert mate-in-N to centipawn equivalent using Lichess approach.

    Lichess uses a formula where mate values are converted to very high
    centipawn values that decay slightly with distance to mate.
    This allows comparing positions with and without forced mates.

    Formula: cp = sign(mate) * (10000 - |mate| * 10)
    - Mate in 1 = ±9990 cp
    - Mate in 10 = ±9900 cp
    - Mate in 100 = ±9000 cp

    This ensures:
    1. All mates are worth more than any material advantage
    2. Shorter mates are worth more than longer mates
    3. The sign indicates who is winning

    Args:
        mate_in_n: Mate in N moves. Positive = side to move wins.
                   Negative = side to move loses.

    Returns:
        Centipawn equivalent (positive = winning, negative = losing).
    """
    if mate_in_n == 0:
        return 0
    sign = 1 if mate_in_n > 0 else -1
    # Cap at 1000 moves to prevent edge cases
    distance = min(abs(mate_in_n), 1000)
    return sign * (10000 - distance * 10)


def eval_to_centipawns(score: Optional[int], mate: Optional[int]) -> Optional[int]:
    """
    Convert any evaluation (score or mate) to centipawns.

    Uses Lichess approach where mate values are converted to very high
    centipawn values.

    Args:
        score: Centipawn evaluation (None if mate).
        mate: Mate in N moves (None if not mate).

    Returns:
        Centipawn value, or None if both inputs are None.
    """
    if mate is not None:
        return mate_to_centipawns(mate)
    return score


def calculate_centipawn_loss(
    score_before: int,
    score_after: int,
    is_white_move: bool,
) -> int:
    """
    Calculate centipawn loss for a single move.

    Centipawn loss measures how much worse a move is compared to the
    best available move. A perfect move has 0 centipawn loss.

    Args:
        score_before: Position evaluation before the move (white's perspective).
        score_after: Position evaluation after the move (white's perspective).
        is_white_move: True if white made the move.

    Returns:
        Centipawn loss (always >= 0). Higher values indicate worse moves.
    """
    if is_white_move:
        # White wants score to stay high or increase
        # Loss = how much the evaluation dropped
        loss = score_before - score_after
    else:
        # Black wants score to decrease (become more negative)
        # Loss = how much the evaluation increased (bad for black)
        loss = score_after - score_before

    # CPL is always non-negative
    return max(0, loss)


def calculate_acpl(
    evaluations: list[dict],
    player_color: str,
    exclude_book_moves: int = 0,
    exclude_endgame_moves: int = 0,
    cap_cpl: Optional[int] = None,
) -> dict:
    """
    Calculate Average Centipawn Loss (ACPL) for a player.

    ACPL is a key metric for evaluating move quality and is used in
    fair play analysis. Lower ACPL indicates more accurate play.

    Args:
        evaluations: List of position evaluations from engine analysis.
                    Each dict should have 'ply', 'score', and optionally 'mate'.
        player_color: 'white' or 'black'.
        exclude_book_moves: Number of opening moves to exclude from calculation.
        exclude_endgame_moves: Number of ending moves to exclude.
        cap_cpl: Optional cap on individual CPL values to reduce outlier impact.

    Returns:
        Dictionary with:
        - 'acpl': Average centipawn loss
        - 'move_count': Number of moves included in calculation
        - 'cpl_values': List of individual CPL values
        - 'total_cpl': Sum of all CPL values
    """
    is_white = player_color.lower() == "white"

    # Filter to relevant plies (player's moves)
    # White moves on odd plies (1, 3, 5...), Black on even (2, 4, 6...)
    start_ply = 1 if is_white else 2

    cpl_values = []

    for i in range(len(evaluations) - 1):
        current = evaluations[i]
        next_pos = evaluations[i + 1]

        ply = next_pos["ply"]

        # Check if this is the player's move
        if is_white and ply % 2 == 0:
            continue
        if not is_white and ply % 2 == 1:
            continue

        # Exclude book/endgame moves
        move_number = (ply + 1) // 2
        if move_number <= exclude_book_moves:
            continue
        if exclude_endgame_moves > 0:
            total_moves = len(evaluations) // 2
            if move_number > total_moves - exclude_endgame_moves:
                continue

        # Convert evaluations to centipawns (handles both score and mate)
        score_before = eval_to_centipawns(current.get("score"), current.get("mate"))
        score_after = eval_to_centipawns(next_pos.get("score"), next_pos.get("mate"))

        if score_before is None or score_after is None:
            continue

        cpl = calculate_centipawn_loss(score_before, score_after, is_white)

        if cap_cpl is not None:
            cpl = min(cpl, cap_cpl)

        cpl_values.append(cpl)

    total_cpl = sum(cpl_values)
    move_count = len(cpl_values)
    acpl = total_cpl / move_count if move_count > 0 else 0.0

    return {
        "acpl": round(acpl, 2),
        "move_count": move_count,
        "cpl_values": cpl_values,
        "total_cpl": total_cpl,
    }


def calculate_move_accuracy_from_cpl(cpl: int) -> float:
    """
    Convert centipawn loss to accuracy percentage (legacy method).

    Uses a formula based on CPL directly.

    Args:
        cpl: Centipawn loss for a move.

    Returns:
        Accuracy percentage (0-100).
    """
    # Formula: accuracy = 103.1668 * e^(-0.04354 * cpl) - 3.1668
    # Clamped to [0, 100]
    accuracy = 103.1668 * math.exp(-0.04354 * cpl) - 3.1668
    return max(0.0, min(100.0, accuracy))


def calculate_move_accuracy(
    win_percent_before: float,
    win_percent_after: float,
) -> float:
    """
    Calculate move accuracy using the Lichess formula.

    Formula: accuracy = 103.1668 * exp(-0.04354 * (winPercentBefore - winPercentAfter)) - 3.1669

    The formula measures how much winning probability was lost with the move.
    A perfect move (no loss) gives ~100% accuracy.

    Args:
        win_percent_before: Win percentage before the move (0-100).
        win_percent_after: Win percentage after the move (0-100).

    Returns:
        Accuracy percentage (0-100), clamped.
    """
    win_percent_loss = win_percent_before - win_percent_after
    accuracy = 103.1668 * math.exp(-0.04354 * win_percent_loss) - 3.1669
    return max(0.0, min(100.0, accuracy))


def calculate_game_accuracy(
    evaluations: list[dict],
    player_color: str,
    exclude_book_moves: int = 0,
    use_harmonic_mean: bool = True,
    use_volatility_weighting: bool = True,
    window_size: int = 2,
) -> dict:
    """
    Calculate overall game accuracy using the Lichess formula.

    The Lichess accuracy methodology:
    1. Convert evaluations to win percentages
    2. Calculate accuracy for each move based on win% change
    3. Use sliding windows to smooth volatility
    4. Apply volatility-based weighting (more weight to stable positions)
    5. Combine using harmonic mean (penalizes bad moves more than arithmetic mean)

    Args:
        evaluations: List of position evaluations with 'score', 'mate', 'ply' keys.
        player_color: 'white' or 'black'.
        exclude_book_moves: Number of opening moves to exclude.
        use_harmonic_mean: Use harmonic mean instead of arithmetic (default True).
        use_volatility_weighting: Weight by position stability (default True).
        window_size: Size of sliding window for smoothing (default 2).

    Returns:
        Dictionary with:
        - 'accuracy': Overall game accuracy (0-100)
        - 'move_accuracies': List of per-move accuracies
        - 'move_count': Number of moves included
        - 'weighted_accuracies': Accuracies after volatility weighting (if enabled)
    """
    is_white = player_color.lower() == "white"

    move_data = []

    for i in range(len(evaluations) - 1):
        current = evaluations[i]
        next_pos = evaluations[i + 1]

        ply = next_pos["ply"]

        # Check if this is the player's move
        if is_white and ply % 2 == 0:
            continue
        if not is_white and ply % 2 == 1:
            continue

        # Exclude book moves
        move_number = (ply + 1) // 2
        if move_number <= exclude_book_moves:
            continue

        # Get scores (handle mate as extreme values)
        score_before = current.get("score")
        score_after = next_pos.get("score")
        mate_before = current.get("mate")
        mate_after = next_pos.get("mate")

        # Convert to centipawns from player's perspective
        if mate_before is not None:
            # Mate values: positive = player mating, negative = being mated
            if is_white:
                cp_before = 10000 if mate_before > 0 else -10000
            else:
                cp_before = 10000 if mate_before < 0 else -10000
        elif score_before is not None:
            cp_before = score_before if is_white else -score_before
        else:
            continue

        if mate_after is not None:
            if is_white:
                cp_after = 10000 if mate_after > 0 else -10000
            else:
                cp_after = 10000 if mate_after < 0 else -10000
        elif score_after is not None:
            cp_after = score_after if is_white else -score_after
        else:
            continue

        # Convert to win percentages
        win_pct_before = centipawns_to_win_percent(cp_before)
        win_pct_after = centipawns_to_win_percent(cp_after)

        # Calculate move accuracy
        accuracy = calculate_move_accuracy(win_pct_before, win_pct_after)

        move_data.append({
            "ply": ply,
            "move_number": move_number,
            "win_pct_before": win_pct_before,
            "win_pct_after": win_pct_after,
            "accuracy": accuracy,
        })

    if not move_data:
        return {
            "accuracy": 0.0,
            "move_accuracies": [],
            "move_count": 0,
            "weighted_accuracies": [],
        }

    # Apply sliding window smoothing (Lichess uses windows to handle volatility)
    if window_size > 1 and len(move_data) >= window_size:
        smoothed_accuracies = []
        for i in range(len(move_data)):
            start = max(0, i - window_size + 1)
            window = [move_data[j]["accuracy"] for j in range(start, i + 1)]
            smoothed_accuracies.append(sum(window) / len(window))
    else:
        smoothed_accuracies = [m["accuracy"] for m in move_data]

    # Calculate volatility weights (more weight to stable positions)
    if use_volatility_weighting:
        weights = []
        for i, m in enumerate(move_data):
            # Volatility = how much the win% changed
            volatility = abs(m["win_pct_before"] - m["win_pct_after"])
            # Higher volatility = lower weight (positions with big swings are less reliable)
            # Use inverse relationship: weight = 1 / (1 + volatility/50)
            weight = 1 / (1 + volatility / 50)
            weights.append(weight)

        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight * len(weights) for w in weights]
        else:
            weights = [1.0] * len(move_data)
    else:
        weights = [1.0] * len(move_data)

    # Apply weights
    weighted_accuracies = [a * w for a, w in zip(smoothed_accuracies, weights)]

    # Calculate final accuracy using harmonic mean or arithmetic mean
    if use_harmonic_mean and len(weighted_accuracies) > 0:
        # Harmonic mean: n / sum(1/x)
        # Add small epsilon to avoid division by zero
        epsilon = 0.01
        harmonic_sum = sum(1 / max(a, epsilon) for a in weighted_accuracies)
        if harmonic_sum > 0:
            game_accuracy = len(weighted_accuracies) / harmonic_sum
        else:
            game_accuracy = 0.0
    else:
        # Arithmetic mean
        game_accuracy = sum(weighted_accuracies) / len(weighted_accuracies) if weighted_accuracies else 0.0

    return {
        "accuracy": round(game_accuracy, 2),
        "move_accuracies": [round(m["accuracy"], 2) for m in move_data],
        "move_count": len(move_data),
        "weighted_accuracies": [round(a, 2) for a in weighted_accuracies],
    }


def calculate_game_accuracy_simple(cpl_values: list[int]) -> float:
    """
    Calculate overall game accuracy from CPL values (legacy simple method).

    Args:
        cpl_values: List of centipawn loss values per move.

    Returns:
        Average accuracy percentage.
    """
    if not cpl_values:
        return 0.0

    accuracies = [calculate_move_accuracy_from_cpl(cpl) for cpl in cpl_values]
    return round(sum(accuracies) / len(accuracies), 2)


def classify_position(
    score: Optional[int],
    mate: Optional[int],
    player_is_white: bool,
) -> str:
    """
    Classify a position based on evaluation.

    Classification thresholds (from player's perspective):
    - Even: +-100 cp
    - Advantage (White/Black): +-100 to +-200 cp
    - Up Minor Piece (White/Black): +-200 to +-500 cp
    - Up Rook (White/Black): +-500 to +-600 cp
    - Up Two Pieces (White/Black): +-600 to +-900 cp
    - Up Queen (White/Black): > +-900 cp
    - Mating (White/Black): Mate in N

    Args:
        score: Position evaluation in centipawns (from white's perspective), or None if mate.
        mate: Mate in N moves (positive = white mates, negative = black mates), or None.
        player_is_white: True if classifying for the white player.

    Returns:
        Position classification string.
    """
    # Handle mate
    if mate is not None:
        if mate > 0:
            return "Mating (White)"
        else:
            return "Mating (Black)"

    if score is None:
        return "Unknown"

    # Get absolute score and determine who is better
    abs_score = abs(score)
    white_is_better = score > 0

    # Determine the advantage level
    if abs_score <= 100:
        return "Even"
    elif abs_score <= 200:
        level = "Advantage"
    elif abs_score <= 500:
        level = "Up Minor Piece"
    elif abs_score <= 600:
        level = "Up Rook"
    elif abs_score <= 900:
        level = "Up Two Pieces"
    else:
        level = "Up Queen"

    # Add color
    color = "White" if white_is_better else "Black"
    return f"{level} ({color})"


def calculate_acpl_by_position(
    evaluations: list[dict],
    player_color: str,
    exclude_book_moves: int = 0,
    cap_cpl: Optional[int] = None,
) -> dict:
    """
    Calculate ACPL segmented by position type.

    Args:
        evaluations: List of position evaluations from engine analysis.
        player_color: 'white' or 'black'.
        exclude_book_moves: Number of opening moves to exclude.
        cap_cpl: Optional cap on individual CPL values.

    Returns:
        Dictionary with ACPL stats for each position type and overall.
    """
    is_white = player_color.lower() == "white"

    # Collect CPL values by position type
    cpl_by_type: dict[str, list] = {}

    for i in range(len(evaluations) - 1):
        current = evaluations[i]
        next_pos = evaluations[i + 1]

        ply = next_pos["ply"]

        # Check if this is the player's move
        if is_white and ply % 2 == 0:
            continue
        if not is_white and ply % 2 == 1:
            continue

        # Exclude book moves
        move_number = (ply + 1) // 2
        if move_number <= exclude_book_moves:
            continue

        score_before = current.get("score")
        score_after = next_pos.get("score")
        mate_before = current.get("mate")
        mate_after = next_pos.get("mate")

        # Classify position BEFORE the move
        position_type = classify_position(score_before, mate_before, is_white)

        # Calculate CPL using Lichess approach (convert mate to centipawns)
        cp_before = eval_to_centipawns(score_before, mate_before)
        cp_after = eval_to_centipawns(score_after, mate_after)

        if cp_before is not None and cp_after is not None:
            cpl = calculate_centipawn_loss(cp_before, cp_after, is_white)
            if cap_cpl is not None:
                cpl = min(cpl, cap_cpl)
        else:
            cpl = 0

        if position_type not in cpl_by_type:
            cpl_by_type[position_type] = []
        cpl_by_type[position_type].append(cpl)

    # Calculate stats for each type
    result = {}
    all_cpl = []

    for pos_type, cpl_values in cpl_by_type.items():
        all_cpl.extend(cpl_values)

        if cpl_values:
            acpl = sum(cpl_values) / len(cpl_values)
        else:
            acpl = 0.0

        result[pos_type] = {
            "acpl": round(acpl, 2),
            "move_count": len(cpl_values),
            "cpl_values": cpl_values,
        }

    # Overall stats
    if all_cpl:
        overall_acpl = sum(all_cpl) / len(all_cpl)
    else:
        overall_acpl = 0.0

    result["overall"] = {
        "acpl": round(overall_acpl, 2),
        "move_count": len(all_cpl),
    }

    return result


def classify_move_by_cpl(cpl: int) -> str:
    """
    Classify a move based on centipawn loss (simple thresholds).

    This is a straightforward classification based only on CPL magnitude,
    without considering the overall position context.

    Classification thresholds:
    - Best: 0 CPL
    - Excellent: 1-9 CPL
    - Good: 10-24 CPL
    - Inaccuracy: 25-49 CPL
    - Mistake: 50-99 CPL
    - Blunder: ≥100 CPL

    Args:
        cpl: Centipawn loss for the move.

    Returns:
        Move classification string.
    """
    if cpl == 0:
        return "Best"
    elif cpl < 10:
        return "Excellent"
    elif cpl < 25:
        return "Good"
    elif cpl < 50:
        return "Inaccuracy"
    elif cpl < 100:
        return "Mistake"
    else:
        return "Blunder"


def classify_move(
    cpl: int,
    mate_before: Optional[int],
    mate_after: Optional[int],
    is_white: bool,
    eval_before_cp: Optional[int] = None,
    eval_after_cp: Optional[int] = None,
) -> str:
    """
    Classify a move based on centipawn loss and position change.

    Classification considers both the magnitude of the error AND whether
    the move changes who has the advantage. "Critical" errors are those
    that shift the position from favorable to unfavorable.

    Classification thresholds:
    - Best: 0 CPL
    - Excellent: 1-9 CPL
    - Good: 10-24 CPL
    - Inaccuracy: 25-49 CPL
    - Mistake: 50-299 CPL, still favorable/equal position
    - Blunder: ≥300 CPL, still favorable/equal position
    - Critical Mistake: 50-299 CPL, position now unfavorable
    - Critical Blunder: ≥300 CPL, position now unfavorable (or allowing mate)
    - Missed Win: Had winning position (>300cp), now equal or worse

    Args:
        cpl: Centipawn loss for the move.
        mate_before: Mate evaluation before the move (None if no mate).
        mate_after: Mate evaluation after the move (None if no mate).
        is_white: True if the player is white.
        eval_before_cp: Centipawn evaluation before move (from player's perspective).
        eval_after_cp: Centipawn evaluation after move (from player's perspective).

    Returns:
        Move classification string.
    """
    # Convert mate to centipawns if evals not provided
    if eval_before_cp is None:
        if mate_before is not None:
            eval_before_cp = mate_to_centipawns(mate_before)
            if not is_white:
                eval_before_cp = -eval_before_cp
    if eval_after_cp is None:
        if mate_after is not None:
            eval_after_cp = mate_to_centipawns(mate_after)
            if not is_white:
                eval_after_cp = -eval_after_cp

    # Check for critical blunder: allowing mate when not already being mated
    if mate_after is not None:
        player_being_mated_after = (is_white and mate_after < 0) or (not is_white and mate_after > 0)

        if player_being_mated_after:
            if mate_before is not None:
                player_being_mated_before = (is_white and mate_before < 0) or (not is_white and mate_before > 0)
                if not player_being_mated_before:
                    return "Critical Blunder"
            else:
                return "Critical Blunder"

    # Check for missed win: had mate, now don't
    if mate_before is not None:
        player_had_mate = (is_white and mate_before > 0) or (not is_white and mate_before < 0)
        if player_had_mate:
            if mate_after is None:
                return "Missed Win"
            player_still_has_mate = (is_white and mate_after > 0) or (not is_white and mate_after < 0)
            if not player_still_has_mate:
                return "Critical Blunder"

    # Determine position characteristics (from player's perspective)
    # Favorable: eval > 100 (slight advantage or better)
    # Winning: eval > 300 (clear advantage)
    # Unfavorable: eval < -100 (opponent has advantage)
    position_was_favorable = eval_before_cp is not None and eval_before_cp > 100
    position_was_winning = eval_before_cp is not None and eval_before_cp > 300
    position_now_unfavorable = eval_after_cp is not None and eval_after_cp < -100
    position_now_equal_or_worse = eval_after_cp is not None and eval_after_cp < 100

    # Missed win: was winning, now equal or worse
    if position_was_winning and position_now_equal_or_worse and cpl >= 50:
        return "Missed Win"

    # Standard CPL-based classification with critical modifier
    if cpl == 0:
        return "Best"
    elif cpl < 10:
        return "Excellent"
    elif cpl < 25:
        return "Good"
    elif cpl < 50:
        return "Inaccuracy"
    elif cpl < 300:
        # Check if this mistake changed the advantage
        if position_was_favorable and position_now_unfavorable:
            return "Critical Mistake"
        return "Mistake"
    else:
        # Blunder - check if critical
        if position_was_favorable and position_now_unfavorable:
            return "Critical Blunder"
        return "Blunder"


def classify_move_by_accuracy(accuracy: float) -> str:
    """
    Classify a move based on accuracy percentage (Lichess-style).

    Classification thresholds:
    - Best: ≥95% accuracy
    - Excellent: 80-95% accuracy
    - Good: 60-80% accuracy
    - Inaccuracy: 40-60% accuracy
    - Mistake: 20-40% accuracy
    - Blunder: <20% accuracy

    Args:
        accuracy: Move accuracy percentage (0-100).

    Returns:
        Move classification string.
    """
    if accuracy >= 95:
        return "Best"
    elif accuracy >= 80:
        return "Excellent"
    elif accuracy >= 60:
        return "Good"
    elif accuracy >= 40:
        return "Inaccuracy"
    elif accuracy >= 20:
        return "Mistake"
    else:
        return "Blunder"


def count_move_types_in_position(
    board: "chess.Board",
    engine_eval: int,
    engine_mate: Optional[int],
    is_white: bool,
    move_evals: Optional[dict] = None,
) -> dict:
    """
    Count legal moves by classification type.

    This is a simplified version that estimates move quality distribution
    based on position characteristics. For accurate counts, you'd need
    to evaluate each legal move with an engine.

    Args:
        board: Chess position.
        engine_eval: Engine evaluation of position (centipawns).
        engine_mate: Mate in N if applicable.
        is_white: True if counting for white's moves.
        move_evals: Optional dict of {move_uci: (score, mate)} for each legal move.

    Returns:
        Dictionary with counts: {legal_moves, best, good, inaccuracy, mistake, blunder}
    """
    legal_moves = list(board.legal_moves)
    num_legal = len(legal_moves)

    # If we don't have individual move evaluations, just return the count
    if move_evals is None:
        return {
            "legal_moves": num_legal,
            "best": 0,
            "good": 0,
            "inaccuracy": 0,
            "mistake": 0,
            "blunder": 0,
        }

    # Count moves by type based on their evaluations
    counts = {
        "legal_moves": num_legal,
        "best": 0,
        "good": 0,
        "inaccuracy": 0,
        "mistake": 0,
        "blunder": 0,
    }

    # Find best move evaluation
    best_score = None
    for move_uci, (score, mate) in move_evals.items():
        if mate is not None:
            # Mate is best for the player if positive (white) or negative (black)
            if is_white and mate > 0:
                best_score = 10000  # Very high
            elif not is_white and mate < 0:
                best_score = 10000
        elif best_score is None or score > best_score:
            best_score = score if is_white else -score

    if best_score is None:
        return counts

    # Classify each move
    for move_uci, (score, mate) in move_evals.items():
        if mate is not None:
            move_score = 10000 if ((is_white and mate > 0) or (not is_white and mate < 0)) else -10000
        else:
            move_score = score if is_white else -score

        cpl = max(0, best_score - move_score)
        move_class = classify_move(cpl, engine_mate, mate, is_white)
        counts[move_class.lower()] = counts.get(move_class.lower(), 0) + 1

    return counts


def extract_position_data(
    evaluations: list[dict],
    player_color: str,
    exclude_book_moves: int = 0,
    cap_cpl: Optional[int] = None,
    include_fragility: bool = True,
) -> list[dict]:
    """
    Extract detailed position data for each move.

    Args:
        evaluations: List of position evaluations from engine analysis.
        player_color: 'white' or 'black'.
        exclude_book_moves: Number of opening moves to exclude.
        cap_cpl: Optional cap on individual CPL values.
        include_fragility: Whether to calculate fragility scores (slower).

    Returns:
        List of dictionaries with position data for each player move:
        - fen: Position in FEN format
        - move: Selected move in UCI format
        - eval_before: Evaluation before move (centipawns or mate)
        - eval_after: Evaluation after move
        - cpl: Centipawn loss for this move
        - move_class: Classification (Best/Good/Inaccuracy/Mistake/Blunder)
        - position_type: Classification of position
        - fragility: Position fragility score (if include_fragility=True)
        - legal_moves: Number of legal moves in position
        - move_number: Full move number
        - ply: Half-move number
    """
    import chess
    from .fragility import calculate_fragility_simple

    is_white = player_color.lower() == "white"
    player_chess_color = chess.WHITE if is_white else chess.BLACK
    positions = []

    for i in range(len(evaluations) - 1):
        current = evaluations[i]
        next_pos = evaluations[i + 1]

        ply = next_pos["ply"]

        # Check if this is the player's move
        if is_white and ply % 2 == 0:
            continue
        if not is_white and ply % 2 == 1:
            continue

        # Exclude book moves
        move_number = (ply + 1) // 2
        if move_number <= exclude_book_moves:
            continue

        score_before = current.get("score")
        score_after = next_pos.get("score")
        mate_before = current.get("mate")
        mate_after = next_pos.get("mate")

        # Format evaluation strings
        if mate_before is not None:
            eval_before_str = f"M{mate_before}"
        else:
            eval_before_str = str(score_before) if score_before is not None else "?"

        if mate_after is not None:
            eval_after_str = f"M{mate_after}"
        else:
            eval_after_str = str(score_after) if score_after is not None else "?"

        # Classify position
        position_type = classify_position(score_before, mate_before, is_white)

        # Calculate CPL using Lichess approach (convert mate to centipawns)
        cp_before = eval_to_centipawns(score_before, mate_before)
        cp_after = eval_to_centipawns(score_after, mate_after)

        if cp_before is not None and cp_after is not None:
            cpl = calculate_centipawn_loss(cp_before, cp_after, is_white)
        else:
            cpl = 0

        # Classify the move (CPL-based with position context)
        # Get evaluation from player's perspective for position assessment
        eval_before_player = cp_before if is_white else (-cp_before if cp_before is not None else None)
        eval_after_player = cp_after if is_white else (-cp_after if cp_after is not None else None)
        move_class = classify_move(cpl, mate_before, mate_after, is_white, eval_before_player, eval_after_player)

        # Calculate move accuracy (Lichess formula)
        # Convert scores to win percentages for accuracy calculation
        if mate_before is not None:
            # Mate in N: use extreme win probability
            if is_white:
                win_pct_before = 100.0 if mate_before > 0 else 0.0
            else:
                win_pct_before = 100.0 if mate_before < 0 else 0.0
        elif score_before is not None:
            # From player's perspective
            cp_for_player = score_before if is_white else -score_before
            win_pct_before = centipawns_to_win_percent(cp_for_player)
        else:
            win_pct_before = 50.0

        if mate_after is not None:
            if is_white:
                win_pct_after = 100.0 if mate_after > 0 else 0.0
            else:
                win_pct_after = 100.0 if mate_after < 0 else 0.0
        elif score_after is not None:
            cp_for_player = score_after if is_white else -score_after
            win_pct_after = centipawns_to_win_percent(cp_for_player)
        else:
            win_pct_after = 50.0

        move_accuracy = calculate_move_accuracy(win_pct_before, win_pct_after)

        # Classify by accuracy (Lichess-style)
        accuracy_class = classify_move_by_accuracy(move_accuracy)

        # Apply CPL cap after classification
        if cap_cpl is not None:
            cpl = min(cpl, cap_cpl)

        # Get legal moves count and calculate fragility
        legal_moves_count = 0
        fragility = 0.0
        fen = current.get("fen", "")
        if fen:
            try:
                board = chess.Board(fen)
                legal_moves_count = len(list(board.legal_moves))
                if include_fragility:
                    fragility = calculate_fragility_simple(board, player_chess_color)
            except Exception:
                pass

        positions.append({
            "fen": fen,
            "move": next_pos.get("move", ""),
            "eval_before": eval_before_str,
            "eval_after": eval_after_str,
            "cpl": cpl,
            "move_class": move_class,
            "accuracy": round(move_accuracy, 1),
            "accuracy_class": accuracy_class,
            "position_type": position_type,
            "fragility": round(fragility, 4),
            "legal_moves": legal_moves_count,
            "move_number": move_number,
            "ply": ply,
        })

    return positions


def classify_advantage(eval_cp: Optional[int], mate: Optional[int], is_white: bool) -> str:
    """
    Classify position by player's advantage level.

    Classification thresholds:
    - Winning: Player advantage > 200cp (or has mate)
    - Equal: |advantage| <= 100cp
    - Losing: Opponent advantage > 200cp (or opponent has mate)
    - Slight Advantage: 100-200cp for player
    - Slight Disadvantage: 100-200cp for opponent

    Args:
        eval_cp: Evaluation in centipawns (from white's perspective).
        mate: Mate in N moves (positive = white wins, negative = black wins).
        is_white: True if classifying for white player.

    Returns:
        Advantage classification string.
    """
    # Handle mate situations
    if mate is not None:
        if is_white:
            return "Winning" if mate > 0 else "Losing"
        else:
            return "Winning" if mate < 0 else "Losing"

    if eval_cp is None:
        return "Unknown"

    # Convert to player's perspective
    player_eval = eval_cp if is_white else -eval_cp

    if player_eval > 200:
        return "Winning"
    elif player_eval > 100:
        return "Slight Advantage"
    elif player_eval >= -100:
        return "Equal"
    elif player_eval >= -200:
        return "Slight Disadvantage"
    else:
        return "Losing"


def analyze_errors_by_advantage(position_data: list[dict]) -> dict:
    """
    Analyze error rates segmented by position advantage.

    Args:
        position_data: List of position data from extract_position_data().

    Returns:
        Dictionary with error analysis by advantage class:
        - error_rates: {advantage_class: {blunders, mistakes, inaccuracies, total, rate}}
        - summary: Overall statistics
    """
    from collections import defaultdict

    # Group moves by advantage class
    by_advantage = defaultdict(lambda: {
        'total': 0,
        'blunders': 0,
        'mistakes': 0,
        'inaccuracies': 0,
        'cpl_sum': 0,
        'accuracy_sum': 0,
    })

    for pos in position_data:
        # Parse eval_before to get centipawns
        eval_str = pos.get('eval_before', '?')
        if eval_str.startswith('M'):
            mate = int(eval_str[1:]) if eval_str[1:].lstrip('-').isdigit() else None
            eval_cp = None
        else:
            try:
                eval_cp = int(eval_str)
                mate = None
            except (ValueError, TypeError):
                eval_cp = None
                mate = None

        # Determine if this position was for white (based on ply)
        ply = pos.get('ply', 0)
        is_white_move = (ply % 2 == 1)

        advantage = classify_advantage(eval_cp, mate, is_white_move)
        stats = by_advantage[advantage]

        stats['total'] += 1
        stats['cpl_sum'] += pos.get('cpl', 0)
        stats['accuracy_sum'] += pos.get('accuracy', 50)

        accuracy_class = pos.get('accuracy_class', '')
        if accuracy_class == 'Blunder':
            stats['blunders'] += 1
        elif accuracy_class == 'Mistake':
            stats['mistakes'] += 1
        elif accuracy_class == 'Inaccuracy':
            stats['inaccuracies'] += 1

    # Calculate rates
    results = {}
    for adv_class, stats in by_advantage.items():
        total = stats['total']
        if total > 0:
            results[adv_class] = {
                'total_moves': total,
                'blunders': stats['blunders'],
                'mistakes': stats['mistakes'],
                'inaccuracies': stats['inaccuracies'],
                'error_count': stats['blunders'] + stats['mistakes'] + stats['inaccuracies'],
                'error_rate': (stats['blunders'] + stats['mistakes'] + stats['inaccuracies']) / total,
                'blunder_rate': stats['blunders'] / total,
                'avg_cpl': stats['cpl_sum'] / total,
                'avg_accuracy': stats['accuracy_sum'] / total,
            }

    return results
