"""
Position fragility calculation based on graph-theoretic analysis.

Based on: "Position Fragility in Chess" (arXiv:2410.02333v3)
https://arxiv.org/html/2410.02333v3

Fragility measures how vulnerable a position is by identifying important
pieces (high betweenness centrality) that are under attack.

Formula: F = Σ g(p) × a(p)
where:
  - g(p) = betweenness centrality of piece p
  - a(p) = 1 if piece is under attack, 0 otherwise
"""

import chess
from typing import Optional
from dataclasses import dataclass
from enum import Enum


def build_interaction_graph(board: chess.Board) -> dict:
    """
    Build a directed graph of piece interactions (attacks/defenses).

    An edge exists from piece A to piece B if:
    - A can legally capture B (attack), or
    - A defends B (same color, A can move to B's square if B weren't there)

    Args:
        board: Chess position.

    Returns:
        Dictionary with:
        - 'nodes': list of (square, piece) tuples
        - 'edges': list of (from_square, to_square) tuples
        - 'adjacency': dict mapping square -> list of squares it connects to
    """
    nodes = []
    edges = []
    adjacency = {sq: [] for sq in chess.SQUARES}

    # Find all pieces
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            nodes.append((square, piece))

    piece_squares = {sq for sq, _ in nodes}

    # Build edges based on attacks and defenses
    for from_sq, from_piece in nodes:
        # Get all squares this piece attacks
        attacks = board.attacks(from_sq)

        for to_sq in attacks:
            to_piece = board.piece_at(to_sq)
            if to_piece is not None:
                # Edge exists: either attack (different color) or defense (same color)
                edges.append((from_sq, to_sq))
                adjacency[from_sq].append(to_sq)

    return {
        'nodes': nodes,
        'edges': edges,
        'adjacency': adjacency,
        'piece_squares': piece_squares,
    }


def calculate_betweenness_centrality(graph: dict) -> dict[int, float]:
    """
    Calculate betweenness centrality for each piece in the interaction graph.

    Betweenness centrality measures how often a node lies on shortest paths
    between pairs of other nodes. Higher values indicate more "important"
    pieces that mediate many interactions.

    Uses Brandes' algorithm for efficiency.

    Args:
        graph: Interaction graph from build_interaction_graph().

    Returns:
        Dictionary mapping square -> betweenness centrality value.
    """
    piece_squares = graph['piece_squares']
    adjacency = graph['adjacency']

    # Initialize centrality scores
    centrality = {sq: 0.0 for sq in piece_squares}

    if len(piece_squares) < 2:
        return centrality

    # Brandes' algorithm for betweenness centrality
    for source in piece_squares:
        # BFS from source
        stack = []
        predecessors = {sq: [] for sq in piece_squares}
        sigma = {sq: 0.0 for sq in piece_squares}
        sigma[source] = 1.0
        distance = {sq: -1 for sq in piece_squares}
        distance[source] = 0

        queue = [source]
        while queue:
            v = queue.pop(0)
            stack.append(v)

            for w in adjacency[v]:
                if w not in piece_squares:
                    continue

                # First visit to w?
                if distance[w] < 0:
                    queue.append(w)
                    distance[w] = distance[v] + 1

                # Shortest path to w via v?
                if distance[w] == distance[v] + 1:
                    sigma[w] += sigma[v]
                    predecessors[w].append(v)

        # Accumulation
        delta = {sq: 0.0 for sq in piece_squares}
        while stack:
            w = stack.pop()
            for v in predecessors[w]:
                delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
            if w != source:
                centrality[w] += delta[w]

    # Normalize by number of pairs (for undirected interpretation)
    n = len(piece_squares)
    if n > 2:
        norm = 1.0 / ((n - 1) * (n - 2))
        for sq in centrality:
            centrality[sq] *= norm

    return centrality


def get_attacked_pieces(board: chess.Board, color: chess.Color) -> set[int]:
    """
    Find all pieces of the given color that are under attack.

    Args:
        board: Chess position.
        color: Color of pieces to check (chess.WHITE or chess.BLACK).

    Returns:
        Set of squares containing attacked pieces.
    """
    attacked = set()
    opponent_color = not color

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None and piece.color == color:
            # Check if any opponent piece attacks this square
            if board.is_attacked_by(opponent_color, square):
                attacked.add(square)

    return attacked


def calculate_fragility(board: chess.Board, color: Optional[chess.Color] = None) -> dict:
    """
    Calculate the fragility score for a position.

    Fragility = Σ g(p) × a(p)
    where g(p) is betweenness centrality and a(p) is 1 if piece is attacked.

    High fragility indicates important pieces are under attack, suggesting
    the position is critical/volatile.

    Args:
        board: Chess position.
        color: If specified, only consider pieces of this color.
               If None, calculate for the side to move.

    Returns:
        Dictionary with:
        - 'fragility': Total fragility score
        - 'attacked_pieces': List of attacked pieces with their centrality
        - 'max_centrality': Highest centrality piece under attack
        - 'centrality': Full centrality dict for all pieces
    """
    if color is None:
        color = board.turn

    # Build interaction graph
    graph = build_interaction_graph(board)

    # Calculate betweenness centrality
    centrality = calculate_betweenness_centrality(graph)

    # Find attacked pieces
    attacked_squares = get_attacked_pieces(board, color)

    # Calculate fragility score
    fragility = 0.0
    attacked_pieces_info = []
    max_centrality = 0.0

    for square in attacked_squares:
        piece = board.piece_at(square)
        if piece is not None and piece.color == color:
            c = centrality.get(square, 0.0)
            fragility += c
            attacked_pieces_info.append({
                'square': chess.square_name(square),
                'piece': piece.symbol(),
                'centrality': round(c, 4),
            })
            if c > max_centrality:
                max_centrality = c

    return {
        'fragility': round(fragility, 4),
        'attacked_pieces': attacked_pieces_info,
        'max_centrality': round(max_centrality, 4),
        'num_attacked': len(attacked_squares),
        'centrality': {chess.square_name(sq): round(c, 4) for sq, c in centrality.items()},
    }


def calculate_fragility_simple(board: chess.Board, color: Optional[chess.Color] = None) -> float:
    """
    Calculate fragility score (simple version returning just the score).

    Args:
        board: Chess position.
        color: Side to calculate fragility for (default: side to move).

    Returns:
        Fragility score as a float.
    """
    result = calculate_fragility(board, color)
    return result['fragility']


class FragilityTrend(Enum):
    """Direction of fragility change."""
    INCREASING = "increasing"  # Heading toward a peak
    DECREASING = "decreasing"  # Past peak, heading toward valley
    STABLE = "stable"          # No significant change
    UNKNOWN = "unknown"        # Not enough data


@dataclass
class FragilityAnalysis:
    """Complete fragility analysis for a game."""
    scores: list[float]      # Fragility at each ply
    peaks: list[int]         # Ply indices of local maxima
    valleys: list[int]       # Ply indices of local minima
    max_fragility: float     # Highest fragility in the game
    max_fragility_ply: int   # Ply where max fragility occurred


def calculate_game_fragility(
    boards: list[chess.Board],
    color: chess.Color
) -> FragilityAnalysis:
    """
    Calculate fragility scores throughout a game and identify peaks/valleys.

    Args:
        boards: List of board positions throughout the game
        color: Color to calculate fragility for

    Returns:
        FragilityAnalysis with scores, peaks, valleys, and max info
    """
    scores = []
    for board in boards:
        scores.append(calculate_fragility_simple(board, color))

    # Find local peaks and valleys (need at least 3 points)
    peaks = []
    valleys = []

    for i in range(1, len(scores) - 1):
        if scores[i] > scores[i - 1] and scores[i] > scores[i + 1]:
            peaks.append(i)
        elif scores[i] < scores[i - 1] and scores[i] < scores[i + 1]:
            valleys.append(i)

    # Handle edge cases for max
    if scores:
        max_frag = max(scores)
        max_ply = scores.index(max_frag)
    else:
        max_frag = 0.0
        max_ply = 0

    return FragilityAnalysis(
        scores=scores,
        peaks=peaks,
        valleys=valleys,
        max_fragility=max_frag,
        max_fragility_ply=max_ply,
    )


def get_fragility_trend(
    current_ply: int,
    fragility_analysis: FragilityAnalysis
) -> tuple[FragilityTrend, int]:
    """
    Determine fragility trend and distance to nearest peak.

    The trend indicates whether fragility is increasing (heading toward peak)
    or decreasing (past peak). Distance is negative before peak, positive after.

    When between two peaks:
    - If trend is increasing, use the next peak
    - If trend is decreasing, use the previous peak

    Args:
        current_ply: Current ply number
        fragility_analysis: FragilityAnalysis from calculate_game_fragility()

    Returns:
        Tuple of (trend, distance) where:
        - trend: FragilityTrend enum value
        - distance: negative = plys before peak, positive = plys after peak
    """
    scores = fragility_analysis.scores
    peaks = fragility_analysis.peaks

    # Need at least 2 scores for trend
    if current_ply >= len(scores) or current_ply < 1:
        return (FragilityTrend.UNKNOWN, 0)

    # Determine trend from comparison with previous score
    prev_score = scores[current_ply - 1]
    curr_score = scores[current_ply]

    if curr_score > prev_score + 0.01:  # Small threshold for stability
        trend = FragilityTrend.INCREASING
    elif curr_score < prev_score - 0.01:
        trend = FragilityTrend.DECREASING
    else:
        trend = FragilityTrend.STABLE

    # Find distance to nearest relevant peak
    if not peaks:
        # No peaks found, use max fragility ply
        return (trend, current_ply - fragility_analysis.max_fragility_ply)

    # Find the previous and next peaks relative to current position
    prev_peak = None
    next_peak = None

    for peak_ply in peaks:
        if peak_ply <= current_ply:
            prev_peak = peak_ply
        elif next_peak is None:
            next_peak = peak_ply
            break

    # Determine which peak to use based on trend
    if trend == FragilityTrend.INCREASING:
        # Heading toward a peak - use next peak (or current if we're at it)
        if next_peak is not None:
            distance = current_ply - next_peak  # Negative (before peak)
        elif prev_peak is not None:
            distance = current_ply - prev_peak  # Positive (after last peak)
        else:
            distance = 0
    elif trend == FragilityTrend.DECREASING:
        # Past a peak - use previous peak
        if prev_peak is not None:
            distance = current_ply - prev_peak  # Positive (after peak)
        elif next_peak is not None:
            distance = current_ply - next_peak  # Negative (before next peak)
        else:
            distance = 0
    else:
        # Stable - use nearest peak
        if prev_peak is not None and next_peak is not None:
            # Use whichever is closer
            if (current_ply - prev_peak) <= (next_peak - current_ply):
                distance = current_ply - prev_peak
            else:
                distance = current_ply - next_peak
        elif prev_peak is not None:
            distance = current_ply - prev_peak
        elif next_peak is not None:
            distance = current_ply - next_peak
        else:
            distance = 0

    return (trend, distance)


def is_pre_fragility_peak(
    current_ply: int,
    fragility_analysis: FragilityAnalysis,
    lookahead: int = 5
) -> bool:
    """
    Check if current position is in the critical zone before a fragility peak.

    Positions before peaks are critical decision points where errors
    can have outsized consequences.

    Args:
        current_ply: Current ply number
        fragility_analysis: FragilityAnalysis from calculate_game_fragility()
        lookahead: Number of plys to look ahead for peaks (default 5)

    Returns:
        True if a peak is within the lookahead window
    """
    for peak_ply in fragility_analysis.peaks:
        if current_ply < peak_ply <= current_ply + lookahead:
            return True
    return False
