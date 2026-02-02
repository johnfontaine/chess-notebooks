"""
Ken Regan's Statistical Cheat Detection System.

Implements the methodology used by FIDE for detecting chess cheating,
as developed by Prof. Kenneth Regan at University at Buffalo.

Reference:
- https://www.chess.com/blog/Jordi641/advanced-cheat-detection-algorithms
- https://cse.buffalo.edu/~regan/publications.html#chess

The Regan system works by:
1. Computing the "drop-off" (difference between best move and played move eval)
2. Using sensitivity and consistency parameters to compute partial credit
3. Converting sensitivity/consistency into an Intrinsic Performance Rating (IPR)
4. Computing a z-score comparing IPR to official Elo rating
5. Flagging z-scores above 4.5 as suspicious (1 in 300,000 chance)
"""

import math
from dataclasses import dataclass
from typing import Optional
import statistics


# Sensitivity and consistency to IPR mapping table
# Based on Table 1 from the article
# Format: (consistency, sensitivity) -> IPR
IPR_CALIBRATION_TABLE = [
    # (consistency, sensitivity, IPR)
    (0.300, 0.200, 1200),
    (0.350, 0.170, 1400),
    (0.400, 0.140, 1600),
    (0.450, 0.110, 1800),
    (0.480, 0.095, 2000),
    (0.500, 0.088, 2200),
    (0.505, 0.085, 2400),
    (0.510, 0.083, 2600),
    (0.515, 0.082, 2700),
    (0.520, 0.080, 2800),
]

# Expected move matching rates by Elo (from ChessBase article)
# These represent the expected % of moves matching engine's top choice
EXPECTED_MOVE_MATCH_BY_ELO = {
    1400: 0.39,
    1600: 0.42,
    1800: 0.45,
    2000: 0.48,
    2100: 0.50,
    2200: 0.52,
    2400: 0.55,
    2600: 0.58,
    2800: 0.61,
}


@dataclass
class PartialCreditResult:
    """Result of partial credit calculation for a single move."""
    fen: str
    played_move: str
    best_move: str
    drop_off: float  # Centipawn difference between best and played move
    partial_credit: float  # Credit assigned to the played move (0-1)
    is_best_move: bool


@dataclass
class ReganAnalysisResult:
    """Complete result of Regan analysis for a game or set of games."""
    num_positions: int
    total_partial_credit: float
    avg_partial_credit: float

    # Move matching metrics
    move_match_rate: float  # % of moves matching engine's top choice
    equal_top_rate: float  # % matching when multiple moves are equally good

    # Derived parameters
    sensitivity: float
    consistency: float

    # Intrinsic Performance Rating
    ipr: float

    # Comparison to official rating
    official_elo: int
    elo_difference: float  # IPR - official Elo

    # Z-score (key metric)
    z_score: float

    # Interpretation
    is_suspicious: bool  # z-score >= 4.5
    suspicion_level: str  # "normal", "elevated", "high", "extreme"
    false_positive_rate: float  # 1 in X chance this is legitimate


def calculate_drop_off(best_eval_cp: float, played_eval_cp: float) -> float:
    """
    Calculate the drop-off between the best move and played move.

    The drop-off represents the loss in centipawns from not playing
    the engine's top recommendation.

    Args:
        best_eval_cp: Engine evaluation of best move in centipawns
        played_eval_cp: Engine evaluation of played move in centipawns

    Returns:
        Drop-off in centipawns (always >= 0)
    """
    return max(0, best_eval_cp - played_eval_cp)


def calculate_partial_credit(
    drop_off: float,
    sensitivity: float = 0.088,
    consistency: float = 0.500,
) -> float:
    """
    Calculate partial credit for a move based on its drop-off.

    The partial credit curve is determined by sensitivity and consistency:
    - Higher consistency = curve hugs y-axis (rarely choose bad moves)
    - Lower sensitivity = better at distinguishing similar moves

    The formula approximates the logistic-style decay shown in the article.

    Args:
        drop_off: Centipawn loss from best move
        sensitivity: Player's sensitivity parameter (lower = stronger)
        consistency: Player's consistency parameter (higher = stronger)

    Returns:
        Partial credit score (0-1)
    """
    if drop_off <= 0:
        return 1.0

    # The partial credit decays based on drop-off
    # Using a logistic-style function that matches the curves in the article
    # Higher consistency makes the curve steeper (less tolerance for drop-off)
    # Higher sensitivity makes the curve shift right (more tolerance)

    # Normalize drop-off to pawns (100 cp = 1 pawn)
    d_pawns = drop_off / 100.0

    # The decay rate depends on consistency (higher = faster decay)
    decay_rate = 2.0 + (consistency - 0.5) * 20  # Ranges roughly 0-4

    # Sensitivity affects the threshold (lower = less tolerant of drop-off)
    threshold = sensitivity * 5  # Ranges roughly 0.4-1.0

    # Logistic decay
    credit = 1.0 / (1.0 + math.exp(decay_rate * (d_pawns - threshold)))

    return max(0.0, min(1.0, credit))


def estimate_parameters_from_moves(
    drop_offs: list[float],
    is_best_moves: list[bool],
) -> tuple[float, float]:
    """
    Estimate sensitivity and consistency parameters from move data.

    Args:
        drop_offs: List of drop-off values for each move
        is_best_moves: List of booleans indicating if each move was the best

    Returns:
        Tuple of (sensitivity, consistency)
    """
    if not drop_offs:
        return 0.088, 0.500  # Default values

    # Consistency: related to how often the player avoids large drop-offs
    # Higher consistency = fewer large errors
    large_error_threshold = 100  # 1 pawn
    large_errors = sum(1 for d in drop_offs if d > large_error_threshold)
    error_rate = large_errors / len(drop_offs)

    # Map error rate to consistency (fewer errors = higher consistency)
    # Error rate of 0% -> consistency ~0.52 (super GM)
    # Error rate of 50% -> consistency ~0.35 (weak player)
    consistency = 0.52 - (error_rate * 0.35)
    consistency = max(0.30, min(0.52, consistency))

    # Sensitivity: related to how well the player distinguishes between moves
    # Lower sensitivity = better discrimination
    # We estimate this from the distribution of drop-offs for non-best moves

    non_best_dropoffs = [d for d, is_best in zip(drop_offs, is_best_moves) if not is_best and d > 0]

    if non_best_dropoffs:
        # Average drop-off when not playing best move indicates sensitivity
        avg_non_best_dropoff = statistics.mean(non_best_dropoffs)
        # Lower average = player is more sensitive (finds close-to-best moves)
        # Map to sensitivity range (0.08-0.20)
        sensitivity = 0.08 + (min(avg_non_best_dropoff, 200) / 200) * 0.12
    else:
        sensitivity = 0.082  # Very strong player (always plays best)

    return sensitivity, consistency


def sensitivity_consistency_to_ipr(sensitivity: float, consistency: float) -> float:
    """
    Convert sensitivity and consistency parameters to an IPR.

    Uses interpolation based on the calibration table from Regan's work.

    Note: This is a simplified model. Real IPR calculation requires
    proper calibration against a database of known games at various
    rating levels. This implementation uses a conservative mapping
    to avoid false positives.

    Args:
        sensitivity: Player's sensitivity parameter
        consistency: Player's consistency parameter

    Returns:
        Intrinsic Performance Rating (Elo-equivalent)
    """
    # Find closest calibration points and interpolate
    # The IPR is primarily determined by consistency, with sensitivity as secondary

    # Compute a combined score (higher consistency, lower sensitivity = higher IPR)
    # Normalize both to 0-1 range
    c_norm = (consistency - 0.30) / (0.52 - 0.30)  # 0 at 0.30, 1 at 0.52
    s_norm = 1 - (sensitivity - 0.08) / (0.20 - 0.08)  # 1 at 0.08, 0 at 0.20

    # Combine with more weight on consistency
    combined = 0.6 * c_norm + 0.4 * s_norm
    combined = max(0, min(1, combined))

    # Map to IPR range (1000-2400) - more conservative than before
    # Even perfect play in a single game shouldn't imply 2800 strength
    # A player having a good game might play at ~300 above their rating
    ipr = 1000 + combined * 1400

    return round(ipr)


def calculate_z_score(
    ipr: float,
    official_elo: int,
    num_moves: int,
    elo_std_dev: float = 200.0,
) -> float:
    """
    Calculate z-score comparing IPR to official Elo.

    The z-score measures how many standard deviations the player's
    performance (IPR) is above their expected performance (Elo).

    Note: Single-game performance has HIGH variance. A player can easily
    perform 200-300 points above or below their rating in any given game.
    We use a conservative standard deviation to avoid false positives.

    For reference:
    - Single game performance SD is ~200-300 Elo
    - With 30 moves, standard error is still ~150 (200/sqrt(2))
    - A Z-score of 2.0 means ~200+ Elo above rating, which can happen naturally

    Args:
        ipr: Intrinsic Performance Rating from move analysis
        official_elo: Player's official Elo rating
        num_moves: Number of moves analyzed
        elo_std_dev: Standard deviation of performance rating (default 200)

    Returns:
        Z-score (positive = performing above Elo)
    """
    # Standard error decreases with more moves (Central Limit Theorem)
    # More moves = more confident in the estimate
    # Use a more conservative scaling: sqrt(num_moves / 40) instead of /10
    # This means ~40 moves needed for one "unit" of confidence
    standard_error = elo_std_dev / math.sqrt(max(1, num_moves / 40))

    # Cap the minimum standard error to avoid over-confidence
    # Even with many moves, single-game variance is high
    standard_error = max(standard_error, 100)

    # Z-score = (observed - expected) / standard_error
    z_score = (ipr - official_elo) / standard_error

    return z_score


def interpret_z_score(z_score: float) -> tuple[str, float, bool]:
    """
    Interpret a z-score in terms of suspicion level.

    Args:
        z_score: The calculated z-score

    Returns:
        Tuple of (suspicion_level, false_positive_rate, is_suspicious)
    """
    if z_score < 2.0:
        return "normal", 1/20, False
    elif z_score < 3.0:
        return "elevated", 1/370, False
    elif z_score < 4.0:
        return "high", 1/15787, False
    elif z_score < 4.5:
        return "very_high", 1/31574, False
    else:
        # FIDE threshold: 4.5 = 1 in 300,000
        return "extreme", 1/300000, True


def get_expected_move_match_rate(elo: int) -> float:
    """
    Get the expected move match rate for a given Elo.

    Based on the ChessBase article showing expected engine move-matching
    ranges from ~39% (1400) to ~61% (2800).

    Args:
        elo: Player's Elo rating

    Returns:
        Expected move match rate (0-1)
    """
    # Interpolate from the table
    elos = sorted(EXPECTED_MOVE_MATCH_BY_ELO.keys())

    if elo <= elos[0]:
        return EXPECTED_MOVE_MATCH_BY_ELO[elos[0]]
    if elo >= elos[-1]:
        return EXPECTED_MOVE_MATCH_BY_ELO[elos[-1]]

    # Find bracketing values
    for i in range(len(elos) - 1):
        if elos[i] <= elo < elos[i + 1]:
            # Linear interpolation
            t = (elo - elos[i]) / (elos[i + 1] - elos[i])
            rate_low = EXPECTED_MOVE_MATCH_BY_ELO[elos[i]]
            rate_high = EXPECTED_MOVE_MATCH_BY_ELO[elos[i + 1]]
            return rate_low + t * (rate_high - rate_low)

    return 0.50  # Default


def analyze_game_regan(
    positions: list[dict],
    official_elo: int,
    exclude_book_moves: int = 0,
) -> ReganAnalysisResult:
    """
    Perform Regan analysis on a game's positions.

    Args:
        positions: List of position dicts with 'best_move', 'move',
                   'eval_before', 'eval_after', 'best_eval' keys
        official_elo: Player's official Elo rating
        exclude_book_moves: Number of opening moves to exclude

    Returns:
        ReganAnalysisResult with full analysis
    """
    # Filter to analyzed positions (after book moves)
    analyzed = positions[exclude_book_moves:]

    if not analyzed:
        return ReganAnalysisResult(
            num_positions=0,
            total_partial_credit=0,
            avg_partial_credit=0,
            move_match_rate=0,
            equal_top_rate=0,
            sensitivity=0.088,
            consistency=0.500,
            ipr=official_elo,
            official_elo=official_elo,
            elo_difference=0,
            z_score=0,
            is_suspicious=False,
            suspicion_level="normal",
            false_positive_rate=1.0,
        )

    # Calculate drop-offs and track best move matches
    drop_offs = []
    is_best_moves = []
    partial_credits = []

    for pos in analyzed:
        played_move = pos.get('move')
        best_move = pos.get('best_move')

        # Get evaluations
        best_eval = pos.get('best_eval', pos.get('eval_before', 0))
        played_eval = pos.get('eval_after', best_eval)

        # Handle centipawn values
        if isinstance(best_eval, str):
            # Mate score
            best_eval = 10000 if best_eval.startswith('M') else -10000
        if isinstance(played_eval, str):
            played_eval = 10000 if played_eval.startswith('M') else -10000

        # Calculate drop-off (CPL essentially)
        cpl = pos.get('cpl', 0)
        drop_off = cpl if cpl is not None else calculate_drop_off(best_eval, played_eval)
        drop_offs.append(drop_off)

        # Track if this was the best move
        is_best = (played_move == best_move) or (drop_off == 0)
        is_best_moves.append(is_best)

    # Estimate sensitivity and consistency from the data
    sensitivity, consistency = estimate_parameters_from_moves(drop_offs, is_best_moves)

    # Calculate partial credits
    for drop_off in drop_offs:
        credit = calculate_partial_credit(drop_off, sensitivity, consistency)
        partial_credits.append(credit)

    # Aggregate statistics
    total_credit = sum(partial_credits)
    avg_credit = total_credit / len(partial_credits) if partial_credits else 0

    # Move matching rates
    move_match_rate = sum(is_best_moves) / len(is_best_moves) if is_best_moves else 0

    # Equal-top rate (moves within 10cp of best - roughly equivalent moves)
    equal_top_moves = sum(1 for d in drop_offs if d <= 10)
    equal_top_rate = equal_top_moves / len(drop_offs) if drop_offs else 0

    # Convert to IPR
    ipr = sensitivity_consistency_to_ipr(sensitivity, consistency)

    # Calculate z-score
    z_score = calculate_z_score(ipr, official_elo, len(analyzed))

    # Interpret
    suspicion_level, false_positive_rate, is_suspicious = interpret_z_score(z_score)

    return ReganAnalysisResult(
        num_positions=len(analyzed),
        total_partial_credit=round(total_credit, 2),
        avg_partial_credit=round(avg_credit, 4),
        move_match_rate=round(move_match_rate, 4),
        equal_top_rate=round(equal_top_rate, 4),
        sensitivity=round(sensitivity, 4),
        consistency=round(consistency, 4),
        ipr=ipr,
        official_elo=official_elo,
        elo_difference=ipr - official_elo,
        z_score=round(z_score, 2),
        is_suspicious=is_suspicious,
        suspicion_level=suspicion_level,
        false_positive_rate=false_positive_rate,
    )


def analyze_multiple_games_regan(
    games_positions: list[list[dict]],
    official_elo: int,
    exclude_book_moves: int = 0,
) -> ReganAnalysisResult:
    """
    Perform Regan analysis across multiple games.

    Aggregating across games provides more statistical power and
    reduces variance in the estimate.

    Args:
        games_positions: List of games, each containing list of positions
        official_elo: Player's official Elo rating
        exclude_book_moves: Number of opening moves to exclude per game

    Returns:
        ReganAnalysisResult with aggregated analysis
    """
    # Flatten all positions
    all_positions = []
    for game_positions in games_positions:
        # Skip book moves for each game
        all_positions.extend(game_positions[exclude_book_moves:])

    return analyze_game_regan(all_positions, official_elo, exclude_book_moves=0)


def compare_to_expected(
    move_match_rate: float,
    official_elo: int,
    num_moves: int,
) -> dict:
    """
    Compare observed move match rate to expected for the rating.

    Args:
        move_match_rate: Observed rate of matching engine's top move
        official_elo: Player's official Elo
        num_moves: Number of moves analyzed

    Returns:
        Dictionary with comparison metrics
    """
    expected_rate = get_expected_move_match_rate(official_elo)

    # Standard deviation for move matching (binomial)
    # SD = sqrt(p * (1-p) / n)
    std_dev = math.sqrt(expected_rate * (1 - expected_rate) / max(1, num_moves))

    # Z-score for move matching
    z_match = (move_match_rate - expected_rate) / std_dev if std_dev > 0 else 0

    return {
        'expected_rate': round(expected_rate, 4),
        'observed_rate': round(move_match_rate, 4),
        'difference': round(move_match_rate - expected_rate, 4),
        'z_score_match': round(z_match, 2),
        'std_dev': round(std_dev, 4),
    }


@dataclass
class SuspiciousPosition:
    """A position flagged as suspicious by Regan analysis."""
    fen: str
    move_number: int
    played_move: str
    best_move: str
    cpl: float
    complexity: float  # Position complexity (from legal moves, fragility, etc.)
    expected_accuracy: float  # Expected accuracy for player's Elo
    actual_accuracy: float  # Actual accuracy achieved
    suspicion_score: float  # How suspicious (0-1)
    reason: str  # Why this position was flagged


def calculate_position_difficulty(
    legal_moves: int,
    fragility: float,
    eval_cp: Optional[int],
) -> float:
    """
    Calculate how difficult a position is to play correctly.

    Difficulty is based on:
    - Number of legal moves (more = harder to find best)
    - Fragility (higher = easier to go wrong)
    - Evaluation sharpness (positions close to 0 are harder)

    Returns:
        Difficulty score (0-1, higher = more difficult)
    """
    # More legal moves = more difficult
    move_difficulty = min(legal_moves / 40, 1.0)  # Cap at 40 moves

    # Higher fragility = more difficult (easier to make mistakes)
    fragility_factor = min(fragility, 1.0)

    # Positions near equality are harder to navigate
    eval_factor = 0.5
    if eval_cp is not None:
        abs_eval = abs(eval_cp)
        if abs_eval < 50:
            eval_factor = 1.0  # Equal positions are hardest
        elif abs_eval < 200:
            eval_factor = 0.7  # Slight advantages
        else:
            eval_factor = 0.3  # Clear advantages are easier

    # Combine factors
    difficulty = 0.4 * move_difficulty + 0.3 * fragility_factor + 0.3 * eval_factor
    return round(difficulty, 3)


def get_expected_accuracy_for_difficulty(
    elo: int,
    difficulty: float,
) -> float:
    """
    Get expected accuracy for a player of given Elo in a position of given difficulty.

    Higher Elo players are expected to find good moves more often,
    but even strong players struggle in very difficult positions.

    Args:
        elo: Player's Elo rating
        difficulty: Position difficulty (0-1)

    Returns:
        Expected accuracy (0-1)
    """
    # Base expected accuracy by Elo (from move match rate table)
    base_accuracy = get_expected_move_match_rate(elo)

    # Difficulty reduces expected accuracy
    # In easy positions (difficulty=0), accuracy should be higher
    # In hard positions (difficulty=1), accuracy drops
    difficulty_penalty = difficulty * 0.3  # Up to 30% reduction

    expected = base_accuracy * (1 - difficulty_penalty)
    return max(0.2, min(0.95, expected))


def identify_suspicious_positions(
    positions: list[dict],
    official_elo: int,
    min_difficulty: float = 0.4,
    suspicion_threshold: float = 0.6,
) -> list[SuspiciousPosition]:
    """
    Identify positions where the player found difficult moves suspiciously well.

    A position is suspicious when:
    1. The position was difficult (high complexity, fragility, many legal moves)
    2. The player found a strong move (low CPL)
    3. This performance exceeds what's expected for their rating

    Args:
        positions: List of position dicts with engine analysis
        official_elo: Player's official Elo rating
        min_difficulty: Minimum difficulty to consider (0-1)
        suspicion_threshold: Threshold for flagging (0-1)

    Returns:
        List of SuspiciousPosition objects, sorted by suspicion score
    """
    suspicious = []

    for i, pos in enumerate(positions):
        # Get position data
        fen = pos.get('fen', '')
        played_move = pos.get('move', '')
        best_move = pos.get('best_move', '')
        cpl = pos.get('cpl', 0) or 0

        # Get complexity factors
        legal_moves = pos.get('legal_moves', 20)
        fragility = pos.get('fragility', 0)
        eval_cp = pos.get('eval_before')

        # Calculate position difficulty
        difficulty = calculate_position_difficulty(legal_moves, fragility, eval_cp)

        # Skip easy positions - nothing suspicious about finding good moves there
        if difficulty < min_difficulty:
            continue

        # Calculate actual accuracy for this move
        # 0 CPL = 100% accurate, 100 CPL = ~50% accurate, 300+ CPL = poor
        if cpl <= 0:
            actual_accuracy = 1.0
        elif cpl <= 10:
            actual_accuracy = 0.95
        elif cpl <= 25:
            actual_accuracy = 0.85
        elif cpl <= 50:
            actual_accuracy = 0.70
        elif cpl <= 100:
            actual_accuracy = 0.50
        else:
            actual_accuracy = max(0.1, 0.50 - (cpl - 100) / 400)

        # Get expected accuracy for this difficulty
        expected_accuracy = get_expected_accuracy_for_difficulty(official_elo, difficulty)

        # Calculate suspicion: how much better than expected?
        # High suspicion = good move in difficult position by lower-rated player
        if actual_accuracy > expected_accuracy:
            # Outperformed expectation
            outperformance = actual_accuracy - expected_accuracy

            # Scale by difficulty (more suspicious in harder positions)
            suspicion_score = outperformance * (1 + difficulty)

            # Bonus for finding the exact best move in difficult positions
            if played_move == best_move and difficulty > 0.6:
                suspicion_score *= 1.3

            suspicion_score = min(1.0, suspicion_score)
        else:
            # Underperformed or met expectations - not suspicious
            suspicion_score = 0

        if suspicion_score >= suspicion_threshold:
            reason = _build_suspicion_reason(
                difficulty, actual_accuracy, expected_accuracy,
                played_move == best_move, cpl
            )

            suspicious.append(SuspiciousPosition(
                fen=fen,
                move_number=i + 1,
                played_move=played_move,
                best_move=best_move,
                cpl=cpl,
                complexity=difficulty,
                expected_accuracy=round(expected_accuracy, 3),
                actual_accuracy=round(actual_accuracy, 3),
                suspicion_score=round(suspicion_score, 3),
                reason=reason,
            ))

    # Sort by suspicion score (most suspicious first)
    suspicious.sort(key=lambda x: x.suspicion_score, reverse=True)
    return suspicious


def _build_suspicion_reason(
    difficulty: float,
    actual: float,
    expected: float,
    is_best: bool,
    cpl: float,
) -> str:
    """Build a human-readable reason for why a position is suspicious."""
    parts = []

    if is_best:
        parts.append("Found best move")
    elif cpl < 10:
        parts.append("Found excellent move")
    else:
        parts.append(f"Found good move (CPL: {cpl:.0f})")

    if difficulty >= 0.7:
        parts.append("in very difficult position")
    elif difficulty >= 0.5:
        parts.append("in difficult position")
    else:
        parts.append("in moderately difficult position")

    diff_pct = (actual - expected) * 100
    parts.append(f"(+{diff_pct:.0f}% above expected)")

    return " ".join(parts)


def get_regan_key_positions(
    positions: list[dict],
    official_elo: int,
    n_suspicious: int = 5,
    n_best_moves_in_complex: int = 5,
) -> dict:
    """
    Get key positions for visualization based on Regan analysis.

    Returns positions that warrant closer inspection:
    1. Suspicious positions (unexpectedly good play in difficult spots)
    2. Best moves found in complex positions (not necessarily suspicious
       but worth reviewing)

    Args:
        positions: List of position dicts with engine analysis
        official_elo: Player's official Elo rating
        n_suspicious: Number of suspicious positions to return
        n_best_moves_in_complex: Number of best-move-in-complex positions

    Returns:
        Dictionary with:
        - suspicious: Most suspicious positions
        - best_in_complex: Best moves found in complex positions
        - summary: Summary statistics
    """
    # Identify suspicious positions
    all_suspicious = identify_suspicious_positions(
        positions, official_elo,
        min_difficulty=0.4,
        suspicion_threshold=0.5
    )

    # Find all positions where player found best move in complex position
    best_in_complex = []
    for i, pos in enumerate(positions):
        played = pos.get('move', '')
        best = pos.get('best_move', '')
        cpl = pos.get('cpl', 0) or 0

        if played == best and cpl == 0:
            legal_moves = pos.get('legal_moves', 20)
            fragility = pos.get('fragility', 0)
            eval_cp = pos.get('eval_before')
            difficulty = calculate_position_difficulty(legal_moves, fragility, eval_cp)

            if difficulty >= 0.5:
                best_in_complex.append({
                    **pos,
                    'difficulty': difficulty,
                    'move_number': i + 1,
                    'flag_reason': f"Best move in difficult position (complexity: {difficulty:.2f})"
                })

    # Sort by difficulty
    best_in_complex.sort(key=lambda x: x.get('difficulty', 0), reverse=True)

    # Convert suspicious positions to dict format for visualization
    suspicious_for_viz = []
    for sp in all_suspicious[:n_suspicious]:
        # Find original position data
        pos_data = next(
            (p for p in positions if p.get('fen') == sp.fen),
            {'fen': sp.fen}
        )
        suspicious_for_viz.append({
            **pos_data,
            'suspicion_score': sp.suspicion_score,
            'expected_accuracy': sp.expected_accuracy,
            'actual_accuracy': sp.actual_accuracy,
            'complexity': sp.complexity,
            'flag_reason': f"Suspicious: {sp.reason}"
        })

    return {
        'suspicious': suspicious_for_viz,
        'best_in_complex': best_in_complex[:n_best_moves_in_complex],
        'summary': {
            'total_suspicious': len(all_suspicious),
            'high_suspicion_count': len([s for s in all_suspicious if s.suspicion_score >= 0.7]),
            'avg_suspicion_score': (
                sum(s.suspicion_score for s in all_suspicious) / len(all_suspicious)
                if all_suspicious else 0
            ),
            'best_moves_in_complex_positions': len(best_in_complex),
        }
    }
