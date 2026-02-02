"""
Clock time analysis for chess games.

Analyzes time usage patterns to detect:
- Correlation between move complexity and think time
- Suspiciously regular time patterns (bot-like behavior)
- Time pressure situations
"""

import re
import chess.pgn
from typing import Optional
from dataclasses import dataclass


@dataclass
class TimeControl:
    """Parsed time control information."""
    base_seconds: int  # Initial time in seconds
    increment_seconds: int  # Time added per move

    @classmethod
    def parse(cls, tc_string: str) -> "TimeControl":
        """
        Parse time control string (e.g., "600", "180+2", "300+5").

        Args:
            tc_string: Time control string from PGN header.

        Returns:
            TimeControl object.
        """
        if not tc_string:
            return cls(0, 0)

        # Handle formats: "600", "180+2", "300+5"
        if "+" in tc_string:
            parts = tc_string.split("+")
            base = int(parts[0])
            increment = int(parts[1])
        else:
            base = int(tc_string)
            increment = 0

        return cls(base, increment)


def parse_clock_time(clock_str: str) -> float:
    """
    Parse clock time string to seconds.

    Args:
        clock_str: Clock string like "0:09:59.9" or "0:05:30"

    Returns:
        Time in seconds as float.
    """
    if not clock_str:
        return 0.0

    # Format: H:MM:SS.d or H:MM:SS
    parts = clock_str.split(":")
    if len(parts) != 3:
        return 0.0

    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])

    return hours * 3600 + minutes * 60 + seconds


def extract_clock_times(game: chess.pgn.Game) -> list[dict]:
    """
    Extract clock times for each move in a game.

    Args:
        game: Parsed PGN game with clock annotations.

    Returns:
        List of dicts with ply, move, clock_remaining, time_spent for each move.
    """
    # Parse time control
    tc_str = game.headers.get("TimeControl", "")
    time_control = TimeControl.parse(tc_str)

    results = []
    node = game
    ply = 0

    # Track previous clock for each color
    prev_white_clock = None
    prev_black_clock = None

    while node.variations:
        node = node.variations[0]
        ply += 1
        move = node.move

        # Extract clock from comment
        clock_remaining = None
        time_spent = None
        comment = node.comment or ""

        # Parse clock annotation: [%clk H:MM:SS.d]
        clock_match = re.search(r'\[%clk\s+(\d+:\d+:\d+(?:\.\d+)?)\]', comment)
        if clock_match:
            clock_remaining = parse_clock_time(clock_match.group(1))

            # Calculate time spent
            is_white = (ply % 2 == 1)

            if is_white:
                if prev_white_clock is not None:
                    # Time spent = previous clock - current clock + increment
                    time_spent = prev_white_clock - clock_remaining + time_control.increment_seconds
                    # Handle first move (may use premove, so time_spent could be negative)
                    if time_spent < 0:
                        time_spent = 0
                prev_white_clock = clock_remaining
            else:
                if prev_black_clock is not None:
                    time_spent = prev_black_clock - clock_remaining + time_control.increment_seconds
                    if time_spent < 0:
                        time_spent = 0
                prev_black_clock = clock_remaining

        results.append({
            'ply': ply,
            'move': move.uci(),
            'san': node.san(),
            'is_white': (ply % 2 == 1),
            'clock_remaining': clock_remaining,
            'time_spent': time_spent,
        })

    return results


def analyze_time_patterns(
    clock_data: list[dict],
    player_color: str,
) -> dict:
    """
    Analyze time usage patterns for a player.

    Args:
        clock_data: List of clock data from extract_clock_times().
        player_color: 'white' or 'black'.

    Returns:
        Dictionary with time pattern statistics.
    """
    import statistics

    is_white = player_color.lower() == 'white'

    # Filter to player's moves with valid time_spent
    player_times = [
        d['time_spent'] for d in clock_data
        if d['is_white'] == is_white and d['time_spent'] is not None
    ]

    if len(player_times) < 3:
        return {
            'avg_time': 0,
            'median_time': 0,
            'std_time': 0,
            'min_time': 0,
            'max_time': 0,
            'cv_time': 0,  # Coefficient of variation
            'regularity_score': 0,
            'num_moves': len(player_times),
            'fast_moves': 0,
            'slow_moves': 0,
        }

    avg_time = statistics.mean(player_times)
    median_time = statistics.median(player_times)
    std_time = statistics.stdev(player_times) if len(player_times) > 1 else 0
    min_time = min(player_times)
    max_time = max(player_times)

    # Coefficient of variation (std/mean) - lower = more regular
    cv_time = std_time / avg_time if avg_time > 0 else 0

    # Regularity score: percentage of moves within 1 second of median
    regular_threshold = 1.0  # seconds
    regular_moves = sum(1 for t in player_times if abs(t - median_time) <= regular_threshold)
    regularity_score = regular_moves / len(player_times)

    # Fast moves (< 2 seconds) and slow moves (> 30 seconds)
    fast_moves = sum(1 for t in player_times if t < 2.0)
    slow_moves = sum(1 for t in player_times if t > 30.0)

    return {
        'avg_time': round(avg_time, 2),
        'median_time': round(median_time, 2),
        'std_time': round(std_time, 2),
        'min_time': round(min_time, 2),
        'max_time': round(max_time, 2),
        'cv_time': round(cv_time, 3),
        'regularity_score': round(regularity_score, 3),
        'num_moves': len(player_times),
        'fast_moves': fast_moves,
        'slow_moves': slow_moves,
        'fast_move_pct': round(fast_moves / len(player_times) * 100, 1),
        'slow_move_pct': round(slow_moves / len(player_times) * 100, 1),
    }


def detect_bot_patterns(time_stats: dict) -> dict:
    """
    Analyze time statistics for bot-like patterns.

    Args:
        time_stats: Output from analyze_time_patterns().

    Returns:
        Dictionary with bot detection flags and scores.
    """
    flags = []
    suspicion_score = 0

    # Very low coefficient of variation (< 0.3) suggests mechanical timing
    if time_stats['cv_time'] < 0.3 and time_stats['num_moves'] >= 10:
        flags.append("Very regular time usage (CV < 0.3)")
        suspicion_score += 30
    elif time_stats['cv_time'] < 0.5 and time_stats['num_moves'] >= 10:
        flags.append("Somewhat regular time usage (CV < 0.5)")
        suspicion_score += 15

    # High regularity score (> 0.5 of moves within 1 second of median)
    if time_stats['regularity_score'] > 0.5 and time_stats['num_moves'] >= 10:
        flags.append(f"High move time regularity ({time_stats['regularity_score']:.0%} within 1s of median)")
        suspicion_score += 25

    # Very few fast moves (humans often premove or play quickly in obvious positions)
    if time_stats['fast_move_pct'] < 5 and time_stats['num_moves'] >= 15:
        flags.append(f"Very few fast moves ({time_stats['fast_move_pct']:.1f}%)")
        suspicion_score += 10

    # No slow moves in long games (humans think longer on critical moves)
    if time_stats['slow_move_pct'] == 0 and time_stats['num_moves'] >= 20:
        flags.append("No slow deliberation moves (>30s)")
        suspicion_score += 15

    # Minimum time is suspiciously consistent (e.g., always > 1 second)
    if time_stats['min_time'] > 1.5 and time_stats['num_moves'] >= 10:
        flags.append(f"Minimum move time suspiciously high ({time_stats['min_time']:.1f}s)")
        suspicion_score += 20

    return {
        'flags': flags,
        'suspicion_score': min(100, suspicion_score),
        'is_suspicious': suspicion_score >= 40,
    }


def classify_time_spent(time_seconds: float) -> str:
    """
    Classify move time into descriptive buckets.

    Classification thresholds:
    - Instant: <1s (premoves, obvious recaptures)
    - Quick: 1-2s (book moves, forcing moves)
    - Short: 2-5s (routine decisions)
    - Normal: 5-15s (thoughtful moves)
    - Long: 15-30s (complex decisions)
    - Very Long: >30s (critical moments)

    Args:
        time_seconds: Time spent on the move in seconds.

    Returns:
        Time classification string.
    """
    if time_seconds < 1:
        return "Instant"
    elif time_seconds < 2:
        return "Quick"
    elif time_seconds < 5:
        return "Short"
    elif time_seconds < 15:
        return "Normal"
    elif time_seconds < 30:
        return "Long"
    else:
        return "Very Long"


def analyze_time_distribution(clock_data: list[dict], player_color: str) -> dict:
    """
    Analyze distribution of move times by classification.

    Args:
        clock_data: List of clock data from extract_clock_times().
        player_color: 'white' or 'black'.

    Returns:
        Dictionary with time distribution by classification:
        - counts: {classification: count}
        - percentages: {classification: percentage}
        - examples: {classification: list of (move_num, time, move)}
    """
    is_white = player_color.lower() == 'white'

    # Initialize counts for each classification
    classifications = ["Instant", "Quick", "Short", "Normal", "Long", "Very Long"]
    counts = {c: 0 for c in classifications}
    examples = {c: [] for c in classifications}

    # Filter to player's moves with valid time_spent
    player_moves = [
        d for d in clock_data
        if d['is_white'] == is_white and d['time_spent'] is not None
    ]

    for d in player_moves:
        time_class = classify_time_spent(d['time_spent'])
        counts[time_class] += 1
        # Keep first 3 examples of each type
        if len(examples[time_class]) < 3:
            examples[time_class].append({
                'move_number': (d['ply'] + 1) // 2,
                'time': d['time_spent'],
                'move': d.get('san', d.get('move', '')),
            })

    total = len(player_moves)
    percentages = {
        c: round(counts[c] / total * 100, 1) if total > 0 else 0
        for c in classifications
    }

    return {
        'counts': counts,
        'percentages': percentages,
        'examples': examples,
        'total_moves': total,
    }


def merge_time_with_positions(
    position_data: list[dict],
    clock_data: list[dict],
    player_color: str,
) -> list[dict]:
    """
    Merge time data with position analysis data.

    Args:
        position_data: Position data from extract_position_data().
        clock_data: Clock data from extract_clock_times().
        player_color: 'white' or 'black'.

    Returns:
        Position data with time_spent added.
    """
    is_white = player_color.lower() == 'white'

    # Create lookup by ply
    clock_by_ply = {d['ply']: d for d in clock_data}

    for pos in position_data:
        ply = pos.get('ply')
        if ply and ply in clock_by_ply:
            clock_info = clock_by_ply[ply]
            pos['time_spent'] = clock_info.get('time_spent')
            pos['clock_remaining'] = clock_info.get('clock_remaining')
        else:
            pos['time_spent'] = None
            pos['clock_remaining'] = None

    return position_data


def calculate_time_complexity_correlation(
    position_data: list[dict],
) -> dict:
    """
    Calculate correlation between time spent and position metrics.

    Args:
        position_data: Position data with time_spent, complexity, cpl, etc.

    Returns:
        Dictionary with correlation statistics.
    """
    import statistics

    # Filter positions with valid time data
    valid_positions = [
        p for p in position_data
        if p.get('time_spent') is not None and p['time_spent'] > 0
    ]

    if len(valid_positions) < 5:
        return {
            'time_cpl_correlation': None,
            'time_complexity_correlation': None,
            'time_fragility_correlation': None,
            'time_legal_moves_correlation': None,
            'num_positions': len(valid_positions),
        }

    def pearson_correlation(x_vals, y_vals):
        """Calculate Pearson correlation coefficient."""
        n = len(x_vals)
        if n < 3:
            return None

        mean_x = statistics.mean(x_vals)
        mean_y = statistics.mean(y_vals)

        numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_vals, y_vals))

        std_x = statistics.stdev(x_vals)
        std_y = statistics.stdev(y_vals)

        if std_x == 0 or std_y == 0:
            return None

        denominator = (n - 1) * std_x * std_y

        return numerator / denominator if denominator != 0 else None

    times = [p['time_spent'] for p in valid_positions]
    cpls = [p.get('cpl', 0) for p in valid_positions]
    fragilities = [p.get('fragility', 0) for p in valid_positions]
    legal_moves = [p.get('legal_moves', 0) for p in valid_positions]

    # Complexity may not be available for all positions
    complexity_positions = [p for p in valid_positions if p.get('complexity') is not None]

    result = {
        'time_cpl_correlation': pearson_correlation(times, cpls),
        'time_fragility_correlation': pearson_correlation(times, fragilities),
        'time_legal_moves_correlation': pearson_correlation(times, legal_moves),
        'num_positions': len(valid_positions),
    }

    if len(complexity_positions) >= 5:
        comp_times = [p['time_spent'] for p in complexity_positions]
        complexities = [p['complexity'] for p in complexity_positions]
        result['time_complexity_correlation'] = pearson_correlation(comp_times, complexities)
    else:
        result['time_complexity_correlation'] = None

    return result
