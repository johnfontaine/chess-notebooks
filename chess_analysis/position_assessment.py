"""
Unified position assessment combining all metrics.

Provides a comprehensive position analysis that includes:
- Stockfish metrics (eval, CPL, rank among legal moves)
- Maia2 metrics (humanness, human moves, CP adjustment)
- Positional assessment (phase, traps, trickiness, material, etc.)

This module serves as the single source of truth for position metrics,
used by both the fairness report and game assessment report.
"""

import chess
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

from .fragility import (
    calculate_fragility_simple,
    FragilityAnalysis,
    FragilityTrend,
    get_fragility_trend,
    is_pre_fragility_peak,
)
from .game_phase import detect_game_phase, GamePhase
from .tablebase import is_tablebase_position, TablebaseClient
from .openings import (
    get_opening_book,
    calculate_distance_from_book,
    OpeningInfo,
)


# =============================================================================
# Data Classes for Stockfish Metrics
# =============================================================================

@dataclass
class StockfishMetrics:
    """Stockfish-derived metrics for a position."""
    eval_before: float           # Centipawns from player's perspective (before move)
    eval_after: float            # Centipawns after move
    analysis_depth: int          # Depth used for analysis
    cpl: int                     # Centipawn loss for this move
    move_rank: int               # Position of played move among sorted legal moves (1 = best)
    total_legal_moves: int       # Total number of legal moves
    best_move: str               # UCI notation of engine's best move
    best_move_san: str = ""      # SAN notation of best move (optional)


# =============================================================================
# Data Classes for Maia2 Metrics
# =============================================================================

@dataclass
class Maia2Metrics:
    """Maia2 humanness metrics for a position."""
    humanness_probability: float  # 0-1, probability Maia2 assigns to played move
    rank: int                     # Rank among Maia2 predictions (1 = most likely)
    num_human_moves: int          # Count of moves with >=1% Maia2 probability
    total_legal_moves: int        # Total number of legal moves
    cp_adjustment: float          # CP diff between played move and top Maia2 move
    top_maia_move: str            # Most probable human move (UCI)
    top_maia_move_san: str = ""   # SAN notation (optional)
    top_maia_probability: float = 0.0  # Probability of top Maia2 move


# =============================================================================
# Data Classes for Positional Assessment
# =============================================================================

@dataclass
class TrapInfo:
    """Information about traps in a position."""
    has_trap_in_candidates: bool  # True if trap among probable Maia2 moves
    trap_moves: list[str]         # UCI notation of trap moves
    is_tricky_position: bool      # Position after player's move has traps for opponent


@dataclass
class StockfishMoveDistribution:
    """Metrics about the distribution of move evaluations from Stockfish."""
    gap_to_second: int            # CP difference between best and 2nd best move
    num_playable_moves: int       # Moves within threshold of best (typically 50cp)
    eval_volatility: float        # Standard deviation of eval across depths
    node_branching_factor: float  # Estimated branching factor from node counts


@dataclass
class PositionalAssessment:
    """Comprehensive positional assessment metrics."""
    # Game context
    game_phase: GamePhase

    # Traps and trickiness
    trap_info: TrapInfo

    # Material evaluation
    pure_material_eval: int       # P=1, N=3, B=3, R=5, Q=9 (from white's perspective)
    engine_eval_cp: int           # Engine evaluation for comparison

    # Branching analysis
    raw_branching_factor: float   # Ratio of legal moves after 3 plys vs current

    # Fragility analysis
    fragility: float
    fragility_trend: FragilityTrend
    distance_to_peak: int         # Negative=before peak, Positive=after peak

    # Stockfish move distribution
    move_distribution: StockfishMoveDistribution

    # Complexity
    complexity_estimate: float    # 0-1 composite score
    complexity_category: str      # LOW/MEDIUM/HIGH/VERY_HIGH

    # Opening book
    is_book_move: bool
    distance_from_book: int       # Plys since last book move

    # Tablebase (for endgame positions)
    tablebase_status: Optional[str]  # "win"/"loss"/"draw" or None if not tablebase position


# =============================================================================
# Main Position Assessment Result
# =============================================================================

@dataclass
class PositionAssessmentResult:
    """Complete position assessment with all three metric categories."""
    fen: str                              # FEN of position before move
    move: str                             # Move played (UCI notation)
    move_san: str                         # Move played (SAN notation)
    ply: int                              # Half-move number (0 = starting position)

    stockfish: StockfishMetrics
    maia2: Optional[Maia2Metrics]         # May be None if Maia2 not available
    positional: PositionalAssessment

    # Convenience fields
    opening_info: Optional[OpeningInfo] = None  # Opening classification if in book


# =============================================================================
# Helper Functions
# =============================================================================

# Material values: P=1, N=3, B=3, R=5, Q=9
MATERIAL_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}


def calculate_pure_material(board: chess.Board) -> int:
    """
    Calculate pure material balance in pawns (P=1, N=3, B=3, R=5, Q=9).

    Returns value from white's perspective (positive = white ahead).

    Args:
        board: Chess position

    Returns:
        Material balance in pawn units
    """
    score = 0
    for piece_type, value in MATERIAL_VALUES.items():
        white_count = len(board.pieces(piece_type, chess.WHITE))
        black_count = len(board.pieces(piece_type, chess.BLACK))
        score += (white_count - black_count) * value
    return score


def calculate_raw_branching_factor(board: chess.Board, depth: int = 3) -> float:
    """
    Calculate raw branching factor: ratio of avg legal moves over next N plys vs current.

    This measures how the position's tactical density changes.

    Args:
        board: Chess position
        depth: Number of half-moves to sample (default 3)

    Returns:
        Branching ratio (>1 means position opens up, <1 means it simplifies)
    """
    current_legal = len(list(board.legal_moves))
    if current_legal == 0:
        return 0.0

    # Sample a few top moves to estimate branching
    total_next_moves = 0
    samples = min(current_legal, 5)  # Sample up to 5 moves

    for i, move in enumerate(board.legal_moves):
        if i >= samples:
            break
        board_copy = board.copy()
        board_copy.push(move)
        total_next_moves += len(list(board_copy.legal_moves))

    if samples > 0:
        avg_next = total_next_moves / samples
        return avg_next / current_legal

    return 1.0


def calculate_brute_force_branching(board: chess.Board, depth: int = 3) -> dict:
    """
    Calculate brute-force branching factor by counting all reachable nodes at each ply.

    This counts the total number of reachable positions at each depth level
    (1 ply = legal moves from current, 2 ply = legal moves of legal moves, etc.)

    Args:
        board: Chess position
        depth: Number of half-moves to explore (default 3)

    Returns:
        Dictionary with:
        - 'nodes_by_depth': List of node counts at each depth [1, 2, 3...]
        - 'total_nodes': Total reachable nodes
        - 'branching_factor': Average branching factor (geometric mean)
    """
    nodes_by_depth = []
    current_positions = [board.copy()]

    for d in range(depth):
        next_positions = []
        for pos in current_positions:
            legal_moves = list(pos.legal_moves)
            for move in legal_moves:
                new_pos = pos.copy()
                new_pos.push(move)
                next_positions.append(new_pos)

        nodes_by_depth.append(len(next_positions))
        current_positions = next_positions

        # Safety limit - stop if we have too many positions
        if len(next_positions) > 100000:
            break

    total_nodes = sum(nodes_by_depth)

    # Calculate average branching factor (geometric mean of ratios)
    if len(nodes_by_depth) >= 1 and nodes_by_depth[0] > 0:
        # Use the Nth root of total leaf nodes / initial legal moves
        initial_moves = len(list(board.legal_moves))
        if initial_moves > 0 and len(nodes_by_depth) > 0:
            final_nodes = nodes_by_depth[-1] if nodes_by_depth else 1
            if final_nodes > 0:
                branching_factor = final_nodes ** (1.0 / len(nodes_by_depth))
            else:
                branching_factor = 0.0
        else:
            branching_factor = 0.0
    else:
        branching_factor = 0.0

    return {
        'nodes_by_depth': nodes_by_depth,
        'total_nodes': total_nodes,
        'branching_factor': branching_factor,
        'initial_legal_moves': len(list(board.legal_moves)),
    }


def detect_traps_in_candidates(
    board: chess.Board,
    maia2_probs: dict[str, float],
    engine_evals: dict[str, int],
    prob_threshold: float = 0.01,
    blunder_threshold_cp: int = 200
) -> TrapInfo:
    """
    Detect traps among Maia2's probable moves.

    A trap is a move that:
    - Has >= prob_threshold in Maia2 predictions (looks human)
    - Loses >= blunder_threshold_cp according to engine (is actually bad)

    Args:
        board: Chess position
        maia2_probs: Dict of move_uci -> probability from Maia2
        engine_evals: Dict of move_uci -> centipawn evaluation
        prob_threshold: Minimum Maia2 probability to consider (default 1%)
        blunder_threshold_cp: CP loss threshold for "trap" (default 200cp)

    Returns:
        TrapInfo with detected traps
    """
    if not maia2_probs or not engine_evals:
        return TrapInfo(
            has_trap_in_candidates=False,
            trap_moves=[],
            is_tricky_position=False,
        )

    trap_moves = []
    best_eval = max(engine_evals.values()) if engine_evals else 0

    for move_uci, prob in maia2_probs.items():
        if prob >= prob_threshold:
            move_eval = engine_evals.get(move_uci, 0)
            if best_eval - move_eval >= blunder_threshold_cp:
                trap_moves.append(move_uci)

    return TrapInfo(
        has_trap_in_candidates=len(trap_moves) > 0,
        trap_moves=trap_moves,
        is_tricky_position=False,  # Set after analyzing resulting position
    )


def check_trickiness(
    board_after: chess.Board,
    maia2_probs_after: Optional[dict[str, float]],
    engine_evals_after: Optional[dict[str, int]],
) -> bool:
    """
    Check if the resulting position after a move is "tricky" for the opponent.

    A position is tricky if it contains traps - moves that look good to humans
    but are actually bad.

    Args:
        board_after: Position after the move
        maia2_probs_after: Maia2 probabilities for opponent's replies
        engine_evals_after: Engine evaluations for opponent's replies

    Returns:
        True if the position contains traps for the opponent
    """
    if maia2_probs_after is None or engine_evals_after is None:
        return False

    trap_info = detect_traps_in_candidates(
        board_after,
        maia2_probs_after,
        engine_evals_after,
    )
    return trap_info.has_trap_in_candidates


def get_tablebase_status(board: chess.Board) -> Optional[str]:
    """
    Get tablebase status for a position if it's a tablebase position.

    Args:
        board: Chess position

    Returns:
        "win", "loss", "draw", or None if not a tablebase position (>7 pieces)
    """
    if not is_tablebase_position(board):
        return None

    try:
        client = TablebaseClient()
        result = client.probe(board.fen())
        if result:
            return result.category
    except Exception:
        pass

    return None


# =============================================================================
# Main Assessment Function
# =============================================================================

def assess_position(
    board: chess.Board,
    move_uci: str,
    ply: int,
    # Engine data
    eval_before_cp: int,
    eval_after_cp: int,
    cpl: int,
    best_move_uci: str,
    analysis_depth: int,
    # Multi-PV data for move ranking
    move_evals: Optional[dict[str, int]] = None,
    # Complexity heuristics
    gap_to_second: int = 0,
    eval_volatility: float = 0.0,
    node_branching_factor: float = 3.5,
    complexity_score: float = 0.0,
    complexity_category: str = "UNKNOWN",
    # Maia2 data (optional)
    maia2_probability: Optional[float] = None,
    maia2_rank: Optional[int] = None,
    maia2_probs: Optional[dict[str, float]] = None,
    maia2_top_move: Optional[str] = None,
    maia2_top_prob: Optional[float] = None,
    # Fragility data
    fragility_analysis: Optional[FragilityAnalysis] = None,
    # Book data
    last_book_ply: int = 0,
    opening_info: Optional[OpeningInfo] = None,
    # For trickiness analysis
    board_after_maia2_probs: Optional[dict[str, float]] = None,
    board_after_engine_evals: Optional[dict[str, int]] = None,
) -> PositionAssessmentResult:
    """
    Perform comprehensive position assessment.

    Args:
        board: Position before the move
        move_uci: Move played in UCI notation
        ply: Current ply number
        eval_before_cp: Engine eval before move (centipawns, from white's perspective)
        eval_after_cp: Engine eval after move
        cpl: Centipawn loss for the move
        best_move_uci: Engine's best move (UCI)
        analysis_depth: Depth used for analysis
        move_evals: Dict of move_uci -> eval for ranking
        gap_to_second: CP gap between best and 2nd best
        eval_volatility: Eval volatility across depths
        node_branching_factor: Branching factor from node counts
        complexity_score: Complexity score (0-1)
        complexity_category: Complexity category string
        maia2_probability: Probability Maia2 assigned to played move
        maia2_rank: Rank of played move in Maia2 predictions
        maia2_probs: Full dict of move -> probability from Maia2
        maia2_top_move: Most probable move according to Maia2
        maia2_top_prob: Probability of top Maia2 move
        fragility_analysis: Pre-computed FragilityAnalysis for the game
        last_book_ply: Ply of last book move
        opening_info: Opening classification
        board_after_maia2_probs: Maia2 probs for resulting position (for trickiness)
        board_after_engine_evals: Engine evals for resulting position (for trickiness)

    Returns:
        PositionAssessmentResult with all metrics organized into categories
    """
    is_white = board.turn == chess.WHITE
    move = chess.Move.from_uci(move_uci)
    move_san = board.san(move)

    # Make the move to get resulting position
    board_after = board.copy()
    board_after.push(move)

    # Get best move SAN
    try:
        best_move = chess.Move.from_uci(best_move_uci)
        best_move_san = board.san(best_move)
    except (ValueError, chess.InvalidMoveError):
        best_move_san = best_move_uci

    # === Calculate move rank among legal moves ===
    legal_moves = list(board.legal_moves)
    total_legal = len(legal_moves)

    if move_evals:
        # Sort moves by evaluation (from perspective of side to move)
        sorted_moves = sorted(
            move_evals.items(),
            key=lambda x: x[1] if is_white else -x[1],
            reverse=True
        )
        move_rank = 1
        for rank, (m_uci, _) in enumerate(sorted_moves, 1):
            if m_uci == move_uci:
                move_rank = rank
                break
    else:
        move_rank = 1  # Default if no multi-PV data

    # Count playable moves (within 50cp of best)
    num_playable = 1
    if move_evals:
        best_eval = max(move_evals.values()) if is_white else min(move_evals.values())
        for m_uci, m_eval in move_evals.items():
            if is_white:
                if best_eval - m_eval <= 50:
                    num_playable += 1
            else:
                if m_eval - best_eval <= 50:
                    num_playable += 1
        num_playable = max(1, num_playable - 1)  # Don't double count best

    # === Stockfish Metrics ===
    # Convert evals to player's perspective
    eval_before_player = eval_before_cp / 100.0 if is_white else -eval_before_cp / 100.0
    eval_after_player = eval_after_cp / 100.0 if is_white else -eval_after_cp / 100.0

    stockfish = StockfishMetrics(
        eval_before=eval_before_player,
        eval_after=eval_after_player,
        analysis_depth=analysis_depth,
        cpl=cpl,
        move_rank=move_rank,
        total_legal_moves=total_legal,
        best_move=best_move_uci,
        best_move_san=best_move_san,
    )

    # === Maia2 Metrics ===
    maia2_metrics = None
    if maia2_probability is not None:
        # Count human moves (>=1% probability)
        num_human_moves = 0
        if maia2_probs:
            num_human_moves = sum(1 for p in maia2_probs.values() if p >= 0.01)

        # Calculate CP adjustment (diff between played move and top Maia2 move)
        cp_adjustment = 0.0
        if maia2_top_move and move_evals:
            played_eval = move_evals.get(move_uci, 0)
            top_eval = move_evals.get(maia2_top_move, 0)
            cp_adjustment = played_eval - top_eval

        # Get top Maia2 move SAN
        top_maia_san = ""
        if maia2_top_move:
            try:
                top_move = chess.Move.from_uci(maia2_top_move)
                top_maia_san = board.san(top_move)
            except (ValueError, chess.InvalidMoveError):
                top_maia_san = maia2_top_move

        maia2_metrics = Maia2Metrics(
            humanness_probability=maia2_probability,
            rank=maia2_rank or 0,
            num_human_moves=num_human_moves,
            total_legal_moves=total_legal,
            cp_adjustment=cp_adjustment,
            top_maia_move=maia2_top_move or "",
            top_maia_move_san=top_maia_san,
            top_maia_probability=maia2_top_prob or 0.0,
        )

    # === Positional Assessment ===

    # Game phase
    phase = detect_game_phase(board)

    # Material
    pure_material = calculate_pure_material(board)

    # Branching
    raw_bf = calculate_raw_branching_factor(board)

    # Fragility
    if fragility_analysis:
        fragility = fragility_analysis.scores[ply] if ply < len(fragility_analysis.scores) else 0.0
        trend, distance = get_fragility_trend(ply, fragility_analysis)
    else:
        fragility = calculate_fragility_simple(board, board.turn)
        trend = FragilityTrend.UNKNOWN
        distance = 0

    # Traps detection
    trap_info = TrapInfo(
        has_trap_in_candidates=False,
        trap_moves=[],
        is_tricky_position=False,
    )
    if maia2_probs and move_evals:
        trap_info = detect_traps_in_candidates(board, maia2_probs, move_evals)
        # Check trickiness of resulting position
        trap_info.is_tricky_position = check_trickiness(
            board_after,
            board_after_maia2_probs,
            board_after_engine_evals,
        )

    # Move distribution
    move_dist = StockfishMoveDistribution(
        gap_to_second=gap_to_second,
        num_playable_moves=num_playable,
        eval_volatility=eval_volatility,
        node_branching_factor=node_branching_factor,
    )

    # Opening book
    book = get_opening_book()
    is_book = book.is_book_move(board, move)
    dist_from_book = calculate_distance_from_book(ply, last_book_ply)

    # Tablebase
    tb_status = get_tablebase_status(board_after)

    positional = PositionalAssessment(
        game_phase=phase,
        trap_info=trap_info,
        pure_material_eval=pure_material,
        engine_eval_cp=eval_before_cp,
        raw_branching_factor=raw_bf,
        fragility=fragility,
        fragility_trend=trend,
        distance_to_peak=distance,
        move_distribution=move_dist,
        complexity_estimate=complexity_score,
        complexity_category=complexity_category,
        is_book_move=is_book,
        distance_from_book=dist_from_book,
        tablebase_status=tb_status,
    )

    return PositionAssessmentResult(
        fen=board.fen(),
        move=move_uci,
        move_san=move_san,
        ply=ply,
        stockfish=stockfish,
        maia2=maia2_metrics,
        positional=positional,
        opening_info=opening_info,
    )


def assessment_to_dict(result: PositionAssessmentResult) -> dict:
    """
    Convert PositionAssessmentResult to a flat dictionary for templates.

    This creates a backwards-compatible dict that can be used directly
    in Jinja2 templates.

    Args:
        result: PositionAssessmentResult from assess_position()

    Returns:
        Flat dictionary with all metrics
    """
    d = {
        # Basic info
        'fen': result.fen,
        'move': result.move,
        'move_san': result.move_san,
        'ply': result.ply,

        # Stockfish metrics
        'eval_before': result.stockfish.eval_before,
        'eval_after': result.stockfish.eval_after,
        'analysis_depth': result.stockfish.analysis_depth,
        'cpl': result.stockfish.cpl,
        'move_rank': result.stockfish.move_rank,
        'total_legal_moves': result.stockfish.total_legal_moves,
        'best_move': result.stockfish.best_move,
        'best_move_san': result.stockfish.best_move_san,

        # Positional assessment
        'phase': result.positional.game_phase.value,
        'has_trap_in_candidates': result.positional.trap_info.has_trap_in_candidates,
        'trap_moves': result.positional.trap_info.trap_moves,
        'is_tricky_position': result.positional.trap_info.is_tricky_position,
        'pure_material_eval': result.positional.pure_material_eval,
        'engine_eval_cp': result.positional.engine_eval_cp,
        'raw_branching_factor': result.positional.raw_branching_factor,
        'fragility': result.positional.fragility,
        'fragility_trend': result.positional.fragility_trend.value,
        'distance_to_peak': result.positional.distance_to_peak,
        'gap_cp': result.positional.move_distribution.gap_to_second,
        'num_playable_moves': result.positional.move_distribution.num_playable_moves,
        'eval_volatility': result.positional.move_distribution.eval_volatility,
        'branching_factor': result.positional.move_distribution.node_branching_factor,
        'engine_complexity_score': result.positional.complexity_estimate,
        'engine_complexity_category': result.positional.complexity_category,
        'is_book_move': result.positional.is_book_move,
        'distance_from_book': result.positional.distance_from_book,
        'tablebase_status': result.positional.tablebase_status,

        # Opening info
        'opening_eco': result.opening_info.eco if result.opening_info else None,
        'opening_name': result.opening_info.name if result.opening_info else None,
    }

    # Maia2 metrics (if available)
    if result.maia2:
        d.update({
            'probability': result.maia2.humanness_probability,
            'rank': result.maia2.rank,
            'num_human_moves': result.maia2.num_human_moves,
            'maia_cp_adjustment': result.maia2.cp_adjustment,
            'top_move': result.maia2.top_maia_move,
            'top_move_san': result.maia2.top_maia_move_san,
            'top_move_probability': result.maia2.top_maia_probability,
        })

    return d
