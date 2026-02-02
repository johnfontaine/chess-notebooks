"""
Position complexity calculation based on move evaluation analysis.

Complexity measures how difficult a position is for a human to navigate:
1. Evaluation Instability: Does the best move change at different depths?
2. Move Clustering: Are many moves similarly evaluated (easy to pick wrong one)?
3. Trap Detection: Are there moves that look good but are actually bad?

References:
- "Complexity and Satisficing" (Salant & Spenkuch, Northwestern)
- Chess.com forum discussions on position complexity
"""

import chess
import chess.engine
from typing import Optional
from dataclasses import dataclass


@dataclass
class MoveEvaluation:
    """Evaluation of a single move at multiple depths."""
    move: chess.Move
    uci: str
    shallow_score: int  # Score at low depth
    shallow_mate: Optional[int]
    deep_score: int  # Score at high depth
    deep_mate: Optional[int]

    @property
    def shallow_cp(self) -> int:
        """Get shallow evaluation in centipawns (mate converted to large value)."""
        if self.shallow_mate is not None:
            return 10000 if self.shallow_mate > 0 else -10000
        return self.shallow_score

    @property
    def deep_cp(self) -> int:
        """Get deep evaluation in centipawns (mate converted to large value)."""
        if self.deep_mate is not None:
            return 10000 if self.deep_mate > 0 else -10000
        return self.deep_score

    @property
    def eval_swing(self) -> int:
        """How much did the evaluation change from shallow to deep?"""
        return abs(self.deep_cp - self.shallow_cp)

    @property
    def is_trap(self) -> bool:
        """Is this move a 'trap' - looks good shallow but bad deep?"""
        # Trap: shallow looks good (>0 for white) but deep is significantly worse
        return self.shallow_cp > 0 and self.deep_cp < self.shallow_cp - 100


@dataclass
class ComplexityResult:
    """Results of complexity analysis for a position."""
    # Core metrics
    instability_score: float  # 0-1, how much best move changes with depth
    cluster_score: float  # 0-1, how many moves are similarly evaluated
    trap_score: float  # 0-1, presence of deceptive moves

    # Combined complexity
    complexity: float  # 0-1, weighted combination

    # Details
    num_legal_moves: int
    best_move_shallow: str
    best_move_deep: str
    best_move_changed: bool
    eval_swing: int  # How much eval changed for position
    num_competitive_moves: int  # Moves within threshold of best
    num_trap_moves: int  # Moves that look good but are bad
    move_evaluations: list  # All move evals for detailed analysis

    def to_dict(self) -> dict:
        """Convert to dictionary for DataFrame/CSV export."""
        return {
            'complexity': round(self.complexity, 4),
            'instability_score': round(self.instability_score, 4),
            'cluster_score': round(self.cluster_score, 4),
            'trap_score': round(self.trap_score, 4),
            'num_legal_moves': self.num_legal_moves,
            'best_move_shallow': self.best_move_shallow,
            'best_move_deep': self.best_move_deep,
            'best_move_changed': self.best_move_changed,
            'eval_swing': self.eval_swing,
            'num_competitive_moves': self.num_competitive_moves,
            'num_trap_moves': self.num_trap_moves,
        }


def analyze_move_at_depth(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    move: chess.Move,
    depth: int,
) -> tuple[int, Optional[int]]:
    """
    Analyze a specific move at a given depth.

    Returns (score_cp, mate_in_n) from the perspective of the side to move.
    """
    # Make the move
    board_copy = board.copy()
    board_copy.push(move)

    # Analyze resulting position
    info = engine.analyse(board_copy, chess.engine.Limit(depth=depth))
    score = info["score"].white()

    # Flip score to be from original side's perspective
    if board.turn == chess.BLACK:
        score = -score

    if score.is_mate():
        # Negate mate because we analyzed AFTER the move
        mate = -score.mate() if score.mate() else None
        return (0, mate)
    else:
        return (score.score(), None)


def calculate_complexity(
    board: chess.Board,
    engine: chess.engine.SimpleEngine,
    shallow_depth: int = 8,
    deep_depth: int = 18,
    competitive_threshold: int = 50,
    top_n_moves: Optional[int] = None,
) -> ComplexityResult:
    """
    Calculate complexity score for a position.

    The complexity score measures how difficult the position is for a human:

    1. Instability Score: Does the best move change between shallow/deep analysis?
       - If best move is different at different depths, position is tricky

    2. Cluster Score: Are many moves similarly evaluated?
       - If 5 moves are all within 50cp of each other, hard to choose
       - If one move is clearly best by 200cp, easier

    3. Trap Score: Are there moves that look good but are bad?
       - Moves that evaluate well at shallow depth but poorly at deep depth

    Args:
        board: Chess position to analyze.
        engine: Running Stockfish engine.
        shallow_depth: Low depth for initial move evaluation (default 8).
        deep_depth: High depth for accurate evaluation (default 18).
        competitive_threshold: CPL threshold for "competitive" moves (default 50cp).
        top_n_moves: Only analyze top N moves at shallow depth (for speed). None = all.

    Returns:
        ComplexityResult with all metrics.
    """
    legal_moves = list(board.legal_moves)
    num_legal = len(legal_moves)

    if num_legal == 0:
        return ComplexityResult(
            instability_score=0.0,
            cluster_score=0.0,
            trap_score=0.0,
            complexity=0.0,
            num_legal_moves=0,
            best_move_shallow="",
            best_move_deep="",
            best_move_changed=False,
            eval_swing=0,
            num_competitive_moves=0,
            num_trap_moves=0,
            move_evaluations=[],
        )

    # Analyze all moves at both depths
    move_evals: list[MoveEvaluation] = []

    for move in legal_moves:
        shallow_score, shallow_mate = analyze_move_at_depth(engine, board, move, shallow_depth)
        deep_score, deep_mate = analyze_move_at_depth(engine, board, move, deep_depth)

        move_evals.append(MoveEvaluation(
            move=move,
            uci=move.uci(),
            shallow_score=shallow_score,
            shallow_mate=shallow_mate,
            deep_score=deep_score,
            deep_mate=deep_mate,
        ))

        # Optional: only analyze top N moves at shallow for speed
        if top_n_moves and len(move_evals) >= top_n_moves:
            # Sort by shallow eval and only keep analyzing top N
            move_evals.sort(key=lambda m: m.shallow_cp, reverse=(board.turn == chess.WHITE))
            if len(move_evals) > top_n_moves:
                move_evals = move_evals[:top_n_moves]

    # Find best moves at each depth
    if board.turn == chess.WHITE:
        best_shallow = max(move_evals, key=lambda m: m.shallow_cp)
        best_deep = max(move_evals, key=lambda m: m.deep_cp)
    else:
        best_shallow = min(move_evals, key=lambda m: m.shallow_cp)
        best_deep = min(move_evals, key=lambda m: m.deep_cp)

    best_move_changed = best_shallow.uci != best_deep.uci

    # Calculate instability score
    # Based on: did best move change? how much did best move's eval swing?
    eval_swing = abs(best_deep.deep_cp - best_shallow.shallow_cp)

    # Instability: combination of best move changing and eval swing
    instability_from_change = 1.0 if best_move_changed else 0.0
    instability_from_swing = min(1.0, eval_swing / 300)  # Normalize: 300cp swing = max
    instability_score = 0.6 * instability_from_change + 0.4 * instability_from_swing

    # Calculate cluster score
    # How many moves are within competitive_threshold of the best?
    best_deep_eval = best_deep.deep_cp
    competitive_moves = []
    for m in move_evals:
        diff = abs(m.deep_cp - best_deep_eval)
        if diff <= competitive_threshold:
            competitive_moves.append(m)

    num_competitive = len(competitive_moves)
    # Normalize: 1 competitive move = 0 cluster, 5+ = 1.0
    cluster_score = min(1.0, (num_competitive - 1) / 4) if num_competitive > 0 else 0.0

    # Calculate trap score
    # How many moves look good at shallow but are bad at deep?
    trap_moves = []
    for m in move_evals:
        # Trap: shallow looks good (within 100cp of shallow best) but deep is bad (>150cp worse than deep best)
        shallow_diff = abs(m.shallow_cp - best_shallow.shallow_cp)
        deep_diff = abs(m.deep_cp - best_deep_eval)

        if shallow_diff <= 100 and deep_diff > 150:
            trap_moves.append(m)

    num_traps = len(trap_moves)
    # Normalize: 0 traps = 0, 3+ traps = 1.0
    trap_score = min(1.0, num_traps / 3)

    # Combined complexity score (weighted average)
    # Instability is most important, then traps, then clusters
    complexity = (
        0.5 * instability_score +
        0.3 * trap_score +
        0.2 * cluster_score
    )

    return ComplexityResult(
        instability_score=instability_score,
        cluster_score=cluster_score,
        trap_score=trap_score,
        complexity=complexity,
        num_legal_moves=num_legal,
        best_move_shallow=best_shallow.uci,
        best_move_deep=best_deep.uci,
        best_move_changed=best_move_changed,
        eval_swing=eval_swing,
        num_competitive_moves=num_competitive,
        num_trap_moves=num_traps,
        move_evaluations=[{
            'move': m.uci,
            'shallow_cp': m.shallow_cp,
            'deep_cp': m.deep_cp,
            'swing': m.eval_swing,
        } for m in move_evals],
    )


def calculate_complexity_fast(
    board: chess.Board,
    engine: chess.engine.SimpleEngine,
    shallow_depth: int = 6,
    deep_depth: int = 14,
    top_n: int = 5,
) -> float:
    """
    Fast complexity calculation - returns just the score.

    Only analyzes top N moves for speed.

    Args:
        board: Chess position.
        engine: Running engine.
        shallow_depth: Low depth (default 6).
        deep_depth: High depth (default 14).
        top_n: Only analyze top N moves (default 5).

    Returns:
        Complexity score 0-1.
    """
    result = calculate_complexity(
        board=board,
        engine=engine,
        shallow_depth=shallow_depth,
        deep_depth=deep_depth,
        top_n_moves=top_n,
    )
    return result.complexity


def analyze_position_complexity_batch(
    positions: list[str],
    engine: chess.engine.SimpleEngine,
    shallow_depth: int = 8,
    deep_depth: int = 18,
    top_n: int = 10,
) -> list[ComplexityResult]:
    """
    Analyze complexity for multiple positions.

    Args:
        positions: List of FEN strings.
        engine: Running engine.
        shallow_depth: Low analysis depth.
        deep_depth: High analysis depth.
        top_n: Only analyze top N moves per position.

    Returns:
        List of ComplexityResult objects.
    """
    results = []
    for fen in positions:
        try:
            board = chess.Board(fen)
            result = calculate_complexity(
                board=board,
                engine=engine,
                shallow_depth=shallow_depth,
                deep_depth=deep_depth,
                top_n_moves=top_n,
            )
            results.append(result)
        except Exception as e:
            # Return zero complexity on error
            results.append(ComplexityResult(
                instability_score=0.0,
                cluster_score=0.0,
                trap_score=0.0,
                complexity=0.0,
                num_legal_moves=0,
                best_move_shallow="",
                best_move_deep="",
                best_move_changed=False,
                eval_swing=0,
                num_competitive_moves=0,
                num_trap_moves=0,
                move_evaluations=[],
            ))

    return results
