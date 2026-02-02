"""
Stockfish engine wrapper for position analysis.

Supports single-depth and multi-depth analysis for complexity assessment.
"""

import chess
import chess.engine
import chess.pgn
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# Default depths for multi-depth analysis
# Includes depth 3 for detecting "trap moves" - moves that look good shallow but degrade
# Fibonacci-inspired sequence for meaningful depth progression (capped at 21 for speed)
DEFAULT_MULTI_DEPTHS = [3, 5, 8, 13, 21]

# Material values in centipawns
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,  # King has no material value
}

# Threshold for "significant" eval change between depths (in centipawns)
SIGNIFICANT_EVAL_CHANGE = 50


def calculate_material_score(board: chess.Board) -> int:
    """
    Calculate material balance from white's perspective in centipawns.

    Returns:
        Positive = white ahead, negative = black ahead
    """
    score = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            value = PIECE_VALUES.get(piece.piece_type, 0)
            if piece.color == chess.WHITE:
                score += value
            else:
                score -= value
    return score


@dataclass
class DepthTransition:
    """Tracks changes between two consecutive analysis depths."""
    from_depth: int
    to_depth: int
    eval_change: int  # Centipawn change in evaluation
    move_changed: bool  # Whether best move changed
    from_move: str
    to_move: str
    from_eval: int
    to_eval: int


@dataclass
class EngineSearchMetrics:
    """Search statistics from engine analysis at a specific depth."""

    depth: int
    nodes: int  # Total nodes searched
    nps: int  # Nodes per second
    time_ms: int  # Time spent in milliseconds
    seldepth: int  # Selective search depth


@dataclass
class GapMetricResult:
    """Gap between best and second-best moves at a specific depth."""

    depth: int
    best_move: str  # UCI notation
    best_eval: int  # Centipawns
    second_move: Optional[str]  # UCI notation (None if only 1 legal move)
    second_eval: Optional[int]  # Centipawns
    gap_cp: int  # Difference in centipawns (0 if only 1 legal move)


@dataclass
class PositionComplexityHeuristics:
    """Aggregate engine-derived complexity metrics for a position."""

    # Evaluation Volatility
    eval_volatility: float  # Standard deviation of evals across depths
    eval_volatility_normalized: float  # Normalized 0-1 scale

    # Gap Metric
    gap_at_max_depth: int  # Gap in cp at highest analyzed depth
    avg_gap: float  # Average gap across analyzed depths

    # Search Depth Velocity / Convergence
    convergence_depth: Optional[int]  # Depth where eval stabilized

    # Node Count per Ply
    total_nodes: int  # Total nodes searched across all depths
    nodes_per_depth: dict[int, int]  # Nodes at each depth
    branching_factor_estimate: float  # Estimated branching factor

    # Composite score
    complexity_score: float  # 0-1 weighted combination
    complexity_category: str  # LOW, MEDIUM, HIGH, VERY_HIGH


@dataclass
class MultiDepthResult:
    """Result from multi-depth position analysis."""

    fen: str
    depths: list[int]
    evaluations: dict[int, int]  # depth -> centipawns (from white's perspective)
    mates: dict[int, Optional[int]]  # depth -> mate in N (or None)
    best_moves: dict[int, str]  # depth -> best move UCI
    pvs: dict[int, list[str]]  # depth -> principal variation
    move_consistency: bool  # True if best move same across all depths
    first_consistent_depth: Optional[int]  # Depth where move stabilized
    best_move_changes: int  # Number of times best move changed
    eval_swing: int  # Max evaluation difference between depths

    # New fields for tracking changes between depths
    depth_transitions: list[DepthTransition]  # Changes between consecutive depths
    material_score: int  # Material balance in centipawns
    eval_vs_material: int  # Difference between engine eval and material (positional component)
    max_eval_change: int  # Largest eval change between any two consecutive depths
    unstable_depths: list[int]  # Depths where eval or move changed significantly

    # Engine heuristics (optional for backward compatibility)
    search_metrics: Optional[dict[int, EngineSearchMetrics]] = None
    gap_metrics: Optional[dict[int, GapMetricResult]] = None
    complexity_heuristics: Optional[PositionComplexityHeuristics] = None


class EngineAnalyzer:
    """Wrapper for Stockfish engine analysis."""

    def __init__(
        self,
        stockfish_path: Optional[str] = None,
        depth: int = 20,
        threads: int = 1,
        hash_mb: int = 256,
    ):
        """
        Initialize the engine analyzer.

        Args:
            stockfish_path: Path to stockfish binary. If None, uses 'stockfish' from PATH.
            depth: Analysis depth (higher = more accurate but slower).
            threads: Number of CPU threads to use.
            hash_mb: Hash table size in MB.
        """
        self.stockfish_path = stockfish_path or "stockfish"
        self.depth = depth
        self.threads = threads
        self.hash_mb = hash_mb
        self._engine: Optional[chess.engine.SimpleEngine] = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def start(self):
        """Start the engine."""
        if self._engine is None:
            self._engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
            self._engine.configure({
                "Threads": self.threads,
                "Hash": self.hash_mb,
                "UCI_ShowWDL": True,  # Enable Win-Draw-Loss statistics
            })

    def stop(self):
        """Stop the engine."""
        if self._engine is not None:
            self._engine.quit()
            self._engine = None

    def analyze(
        self,
        board: chess.Board,
        depth: Optional[int] = None,
        include_material: bool = True,
    ) -> dict:
        """
        Analyze a position.

        Args:
            board: The chess position to analyze.
            depth: Override default depth for this analysis.
            include_material: Include material score (cheap, enabled by default).

        Returns:
            Dictionary with:
            - 'score': Engine eval in centipawns from white's perspective
            - 'mate': Mate-in-N if mate found, else None
            - 'pv': Principal variation (list of moves)
            - 'depth': Actual analysis depth
            - 'material': Material balance in centipawns (if include_material=True)
            - 'eval_vs_material': Engine eval minus material (positional component)
            - 'wdl': Win-Draw-Loss tuple (win, draw, loss) in permille from white's perspective
        """
        if self._engine is None:
            raise RuntimeError("Engine not started. Use 'with' context or call start().")

        info = self._engine.analyse(
            board,
            chess.engine.Limit(depth=depth or self.depth),
        )

        score = info["score"].white()
        result = {
            "pv": info.get("pv", []),
            "depth": info.get("depth", 0),
        }

        if score.is_mate():
            result["mate"] = score.mate()
            # Convert mate to centipawn equivalent (large value)
            mate_moves = score.mate()
            if mate_moves > 0:
                result["score"] = 10000 - (mate_moves * 10)
            else:
                result["score"] = -10000 - (mate_moves * 10)
        else:
            result["score"] = score.score()
            result["mate"] = None

        # Add material analysis (inexpensive)
        if include_material:
            material = calculate_material_score(board)
            result["material"] = material
            result["eval_vs_material"] = result["score"] - material

        # Extract WDL (Win-Draw-Loss) statistics if available
        wdl = info.get("wdl")
        if wdl:
            # WDL comes as PovWdl, get from white's perspective
            wdl_white = wdl.white()
            result["wdl"] = (wdl_white.wins, wdl_white.draws, wdl_white.losses)
        else:
            result["wdl"] = None

        return result

    def analyze_game(
        self,
        game: chess.pgn.Game,
        depth: Optional[int] = None,
    ) -> list[dict]:
        """
        Analyze all positions in a game.

        Args:
            game: A parsed PGN game.
            depth: Override default depth for this analysis.

        Returns:
            List of analysis results for each position.
        """
        results = []
        board = game.board()

        # Analyze starting position
        results.append({
            "ply": 0,
            "move": None,
            "fen": board.fen(),
            **self.analyze(board, depth),
        })

        for ply, move in enumerate(game.mainline_moves(), start=1):
            board.push(move)
            analysis = self.analyze(board, depth)
            results.append({
                "ply": ply,
                "move": move.uci(),
                "fen": board.fen(),
                **analysis,
            })

        return results

    def analyze_multi_depth(
        self,
        board: chess.Board,
        depths: Optional[list[int]] = None,
    ) -> MultiDepthResult:
        """
        Analyze a position at multiple depths.

        This is useful for detecting:
        - Position complexity (best move changes at different depths)
        - Horizon effects (evaluation swings between depths)
        - Move depth requirements (when move becomes clearly best)
        - Positional vs material imbalance (eval vs material score)

        Args:
            board: The chess position to analyze.
            depths: List of depths to analyze at. Defaults to [5, 8, 13, 21].

        Returns:
            MultiDepthResult with evaluations, best moves, and depth transitions.
        """
        if self._engine is None:
            raise RuntimeError("Engine not started. Use 'with' context or call start().")

        if depths is None:
            depths = DEFAULT_MULTI_DEPTHS

        evaluations: dict[int, int] = {}
        mates: dict[int, Optional[int]] = {}
        best_moves: dict[int, str] = {}
        pvs: dict[int, list[str]] = {}

        sorted_depths = sorted(depths)

        for depth in sorted_depths:
            info = self._engine.analyse(
                board,
                chess.engine.Limit(depth=depth),
            )

            score = info["score"].white()
            pv = info.get("pv", [])

            # Store principal variation as UCI strings
            pvs[depth] = [m.uci() for m in pv]

            # Store best move (first move of PV)
            if pv:
                best_moves[depth] = pv[0].uci()

            # Store evaluation
            if score.is_mate():
                mates[depth] = score.mate()
                mate_moves = score.mate()
                if mate_moves > 0:
                    evaluations[depth] = 10000 - (mate_moves * 10)
                else:
                    evaluations[depth] = -10000 - (mate_moves * 10)
            else:
                mates[depth] = None
                evaluations[depth] = score.score() or 0

        # Analyze move consistency
        move_consistency, first_consistent, changes = self._analyze_move_consistency(
            depths, best_moves
        )

        # Calculate evaluation swing
        if evaluations:
            eval_swing = max(evaluations.values()) - min(evaluations.values())
        else:
            eval_swing = 0

        # Calculate depth transitions (changes between consecutive depths)
        depth_transitions = []
        max_eval_change = 0
        unstable_depths = []

        for i in range(1, len(sorted_depths)):
            prev_depth = sorted_depths[i - 1]
            curr_depth = sorted_depths[i]

            prev_eval = evaluations.get(prev_depth, 0)
            curr_eval = evaluations.get(curr_depth, 0)
            eval_change = curr_eval - prev_eval

            prev_move = best_moves.get(prev_depth, "")
            curr_move = best_moves.get(curr_depth, "")
            move_changed = prev_move != curr_move

            transition = DepthTransition(
                from_depth=prev_depth,
                to_depth=curr_depth,
                eval_change=eval_change,
                move_changed=move_changed,
                from_move=prev_move,
                to_move=curr_move,
                from_eval=prev_eval,
                to_eval=curr_eval,
            )
            depth_transitions.append(transition)

            # Track max eval change
            if abs(eval_change) > max_eval_change:
                max_eval_change = abs(eval_change)

            # Flag unstable depths (significant eval or move change)
            if abs(eval_change) >= SIGNIFICANT_EVAL_CHANGE or move_changed:
                unstable_depths.append(curr_depth)

        # Calculate material score (cheap operation, always do it)
        material_score = calculate_material_score(board)

        # Compare engine eval (at max depth) to material
        max_depth_eval = evaluations.get(sorted_depths[-1], 0) if sorted_depths else 0
        eval_vs_material = max_depth_eval - material_score

        return MultiDepthResult(
            fen=board.fen(),
            depths=depths,
            evaluations=evaluations,
            mates=mates,
            best_moves=best_moves,
            pvs=pvs,
            move_consistency=move_consistency,
            first_consistent_depth=first_consistent,
            best_move_changes=changes,
            eval_swing=eval_swing,
            depth_transitions=depth_transitions,
            material_score=material_score,
            eval_vs_material=eval_vs_material,
            max_eval_change=max_eval_change,
            unstable_depths=unstable_depths,
        )

    def _analyze_move_consistency(
        self,
        depths: list[int],
        best_moves: dict[int, str],
    ) -> tuple[bool, Optional[int], int]:
        """
        Analyze how consistent the best move is across depths.

        Returns:
            Tuple of:
            - move_consistency: True if same move at all depths
            - first_consistent_depth: Depth where move stopped changing
            - best_move_changes: Number of times the best move changed
        """
        sorted_depths = sorted(depths)
        if not sorted_depths or not best_moves:
            return True, None, 0

        changes = 0
        first_consistent = None
        prev_move = None
        final_move = best_moves.get(sorted_depths[-1])

        for depth in sorted_depths:
            move = best_moves.get(depth)
            if move is None:
                continue

            if prev_move is not None and move != prev_move:
                changes += 1

            # Track when move becomes consistent with final answer
            if first_consistent is None and move == final_move:
                first_consistent = depth
            elif move != final_move:
                first_consistent = None  # Reset if it changes again

            prev_move = move

        # If we never found consistency, use max depth
        if first_consistent is None and final_move is not None:
            first_consistent = sorted_depths[-1]

        move_consistency = changes == 0

        return move_consistency, first_consistent, changes

    def analyze_game_multi_depth(
        self,
        game: chess.pgn.Game,
        depths: Optional[list[int]] = None,
        skip_first_n: int = 0,
    ) -> list[dict]:
        """
        Analyze all positions in a game at multiple depths.

        Args:
            game: A parsed PGN game.
            depths: List of depths to analyze at. Defaults to [1,2,5,8,13,21,30].
            skip_first_n: Skip first N positions (e.g., for book moves).

        Returns:
            List of analysis results with multi-depth data for each position.
        """
        if depths is None:
            depths = DEFAULT_MULTI_DEPTHS

        results = []
        board = game.board()

        # Analyze starting position if not skipping
        if skip_first_n == 0:
            multi_result = self.analyze_multi_depth(board, depths)
            results.append({
                "ply": 0,
                "move": None,
                "fen": board.fen(),
                "multi_depth": multi_result,
                # Also include max-depth analysis for compatibility
                "score": multi_result.evaluations.get(max(depths), 0),
                "mate": multi_result.mates.get(max(depths)),
                "pv": multi_result.pvs.get(max(depths), []),
                "depth": max(depths),
            })

        for ply, move in enumerate(game.mainline_moves(), start=1):
            board.push(move)

            if ply <= skip_first_n:
                continue

            multi_result = self.analyze_multi_depth(board, depths)
            results.append({
                "ply": ply,
                "move": move.uci(),
                "fen": board.fen(),
                "multi_depth": multi_result,
                # Also include max-depth analysis for compatibility
                "score": multi_result.evaluations.get(max(depths), 0),
                "mate": multi_result.mates.get(max(depths)),
                "pv": multi_result.pvs.get(max(depths), []),
                "depth": max(depths),
            })

        return results

    def check_move_at_depths(
        self,
        board: chess.Board,
        move: chess.Move,
        depths: Optional[list[int]] = None,
    ) -> dict:
        """
        Check if a specific move is the best move at each depth.

        This helps identify:
        - Shallow moves (correct at low depth)
        - Deep moves (only correct at high depth, possible engine use)
        - Consistent moves (correct at all depths)

        Args:
            board: Position before the move.
            move: The move to check.
            depths: Depths to check at. Defaults to [1,2,5,8,13,21,30].

        Returns:
            Dictionary with:
            - 'move': The move in UCI notation
            - 'is_best_at': List of depths where this is the best move
            - 'first_best_depth': First depth where this becomes best
            - 'always_best': True if best at all depths
            - 'never_best': True if never best at any depth
            - 'depth_analysis': Detailed analysis at each depth
        """
        if depths is None:
            depths = DEFAULT_MULTI_DEPTHS

        multi_result = self.analyze_multi_depth(board, depths)

        move_uci = move.uci()
        is_best_at = []
        depth_analysis = {}

        for depth in depths:
            best_at_depth = multi_result.best_moves.get(depth)
            is_best = best_at_depth == move_uci

            if is_best:
                is_best_at.append(depth)

            depth_analysis[depth] = {
                "is_best": is_best,
                "best_move": best_at_depth,
                "evaluation": multi_result.evaluations.get(depth),
                "mate": multi_result.mates.get(depth),
            }

        first_best = min(is_best_at) if is_best_at else None

        return {
            "move": move_uci,
            "is_best_at": is_best_at,
            "first_best_depth": first_best,
            "always_best": len(is_best_at) == len(depths),
            "never_best": len(is_best_at) == 0,
            "depth_analysis": depth_analysis,
            "multi_depth_result": multi_result,
        }

    def analyze_multi_depth_extended(
        self,
        board: chess.Board,
        depths: Optional[list[int]] = None,
        multipv: int = 2,
        capture_search_stats: bool = True,
    ) -> MultiDepthResult:
        """
        Extended multi-depth analysis with search statistics and gap metrics.

        This method provides additional engine heuristics for complexity assessment:
        - Evaluation volatility (std dev across depths)
        - Gap metric (difference between best and second-best move)
        - Search statistics (nodes, nps, time)
        - Convergence analysis

        Args:
            board: The chess position to analyze.
            depths: List of depths to analyze at. Defaults to DEFAULT_MULTI_DEPTHS.
            multipv: Number of principal variations for gap metric (default 2).
            capture_search_stats: Whether to capture nodes/nps/time (default True).

        Returns:
            MultiDepthResult with populated search_metrics, gap_metrics,
            and complexity_heuristics fields.
        """
        if self._engine is None:
            raise RuntimeError("Engine not started. Use 'with' context or call start().")

        if depths is None:
            depths = DEFAULT_MULTI_DEPTHS

        evaluations: dict[int, int] = {}
        mates: dict[int, Optional[int]] = {}
        best_moves: dict[int, str] = {}
        pvs: dict[int, list[str]] = {}
        search_metrics: dict[int, EngineSearchMetrics] = {}
        gap_metrics: dict[int, GapMetricResult] = {}

        sorted_depths = sorted(depths)

        # Note: MultiPV is passed directly to analyse(), not via configure()
        # The python-chess library manages MultiPV automatically

        for depth in sorted_depths:
            info = self._engine.analyse(
                board,
                chess.engine.Limit(depth=depth),
                multipv=multipv if multipv > 1 else None,
            )

            # Handle MultiPV results (returns list of infos)
            if multipv > 1 and isinstance(info, list):
                infos = info
                primary_info = infos[0] if infos else {}
            else:
                infos = [info]
                primary_info = info

            # Extract primary PV data
            score = primary_info.get("score")
            if score:
                score = score.white()
            pv = primary_info.get("pv", [])

            # Store principal variation as UCI strings
            pvs[depth] = [m.uci() for m in pv]

            # Store best move (first move of PV)
            if pv:
                best_moves[depth] = pv[0].uci()

            # Store evaluation
            if score and score.is_mate():
                mates[depth] = score.mate()
                mate_moves = score.mate()
                if mate_moves > 0:
                    evaluations[depth] = 10000 - (mate_moves * 10)
                else:
                    evaluations[depth] = -10000 - (mate_moves * 10)
            elif score:
                mates[depth] = None
                evaluations[depth] = score.score() or 0
            else:
                mates[depth] = None
                evaluations[depth] = 0

            # Capture search statistics
            if capture_search_stats:
                search_metrics[depth] = EngineSearchMetrics(
                    depth=depth,
                    nodes=primary_info.get("nodes", 0),
                    nps=primary_info.get("nps", 0),
                    time_ms=int(primary_info.get("time", 0) * 1000)
                    if primary_info.get("time")
                    else 0,
                    seldepth=primary_info.get("seldepth", depth),
                )

            # Calculate gap metric from MultiPV
            if multipv > 1 and len(infos) >= 2:
                first_info = infos[0]
                second_info = infos[1]

                first_pv = first_info.get("pv", [])
                second_pv = second_info.get("pv", [])

                first_score = first_info.get("score")
                second_score = second_info.get("score")

                if first_pv and first_score:
                    first_eval = _score_to_cp(first_score.white())
                    second_eval = (
                        _score_to_cp(second_score.white())
                        if second_pv and second_score
                        else None
                    )

                    gap_metrics[depth] = GapMetricResult(
                        depth=depth,
                        best_move=first_pv[0].uci(),
                        best_eval=first_eval,
                        second_move=second_pv[0].uci() if second_pv else None,
                        second_eval=second_eval,
                        gap_cp=abs(first_eval - second_eval)
                        if second_eval is not None
                        else 0,
                    )
            elif pv:
                # No MultiPV, just record best move with no gap
                gap_metrics[depth] = GapMetricResult(
                    depth=depth,
                    best_move=pv[0].uci(),
                    best_eval=evaluations[depth],
                    second_move=None,
                    second_eval=None,
                    gap_cp=0,
                )

        # Analyze move consistency
        move_consistency, first_consistent, changes = self._analyze_move_consistency(
            depths, best_moves
        )

        # Calculate evaluation swing
        if evaluations:
            eval_swing = max(evaluations.values()) - min(evaluations.values())
        else:
            eval_swing = 0

        # Calculate depth transitions
        depth_transitions = []
        max_eval_change = 0
        unstable_depths = []

        for i in range(1, len(sorted_depths)):
            prev_depth = sorted_depths[i - 1]
            curr_depth = sorted_depths[i]

            prev_eval = evaluations.get(prev_depth, 0)
            curr_eval = evaluations.get(curr_depth, 0)
            eval_change = curr_eval - prev_eval

            prev_move = best_moves.get(prev_depth, "")
            curr_move = best_moves.get(curr_depth, "")
            move_changed = prev_move != curr_move

            transition = DepthTransition(
                from_depth=prev_depth,
                to_depth=curr_depth,
                eval_change=eval_change,
                move_changed=move_changed,
                from_move=prev_move,
                to_move=curr_move,
                from_eval=prev_eval,
                to_eval=curr_eval,
            )
            depth_transitions.append(transition)

            if abs(eval_change) > max_eval_change:
                max_eval_change = abs(eval_change)

            if abs(eval_change) >= SIGNIFICANT_EVAL_CHANGE or move_changed:
                unstable_depths.append(curr_depth)

        # Calculate material score
        material_score = calculate_material_score(board)

        # Compare engine eval (at max depth) to material
        max_depth_eval = evaluations.get(sorted_depths[-1], 0) if sorted_depths else 0
        eval_vs_material = max_depth_eval - material_score

        # Calculate complexity heuristics
        complexity_heuristics = calculate_complexity_heuristics(
            evaluations=evaluations,
            search_metrics=search_metrics if capture_search_stats else None,
            gap_metrics=gap_metrics,
            first_consistent_depth=first_consistent,
        )

        return MultiDepthResult(
            fen=board.fen(),
            depths=depths,
            evaluations=evaluations,
            mates=mates,
            best_moves=best_moves,
            pvs=pvs,
            move_consistency=move_consistency,
            first_consistent_depth=first_consistent,
            best_move_changes=changes,
            eval_swing=eval_swing,
            depth_transitions=depth_transitions,
            material_score=material_score,
            eval_vs_material=eval_vs_material,
            max_eval_change=max_eval_change,
            unstable_depths=unstable_depths,
            search_metrics=search_metrics if capture_search_stats else None,
            gap_metrics=gap_metrics,
            complexity_heuristics=complexity_heuristics,
        )


# Helper functions for complexity heuristics


def _score_to_cp(score: chess.engine.Score) -> int:
    """Convert a chess.engine.Score to centipawns."""
    if score.is_mate():
        mate_moves = score.mate()
        if mate_moves > 0:
            return 10000 - (mate_moves * 10)
        else:
            return -10000 - (mate_moves * 10)
    return score.score() or 0


def calculate_eval_volatility(evaluations: dict[int, int]) -> tuple[float, float]:
    """
    Calculate standard deviation of evaluations across depths.

    Args:
        evaluations: Dict mapping depth -> centipawn evaluation

    Returns:
        (raw_std_dev, normalized_score) where normalized is 0-1
        Normalization: std_dev / 200, capped at 1.0
    """
    vals = list(evaluations.values())
    if len(vals) < 2:
        return 0.0, 0.0

    # Calculate standard deviation manually to avoid numpy dependency
    mean_val = sum(vals) / len(vals)
    variance = sum((v - mean_val) ** 2 for v in vals) / len(vals)
    std_dev = variance**0.5

    # Normalize: 200cp std dev = 1.0 complexity
    normalized = min(1.0, std_dev / 200.0)

    return float(std_dev), float(normalized)


def estimate_branching_factor(
    search_metrics: Optional[dict[int, EngineSearchMetrics]],
) -> float:
    """
    Estimate branching factor from node counts at different depths.

    Uses: b = (nodes_d2 / nodes_d1)^(1/(d2-d1))
    Averaged across depth pairs.

    Args:
        search_metrics: Dict mapping depth -> EngineSearchMetrics

    Returns:
        Estimated branching factor (typical chess: 2-8, average ~3.5)
    """
    if not search_metrics or len(search_metrics) < 2:
        return 3.5  # Default average branching factor for chess

    sorted_depths = sorted(search_metrics.keys())
    branching_factors = []

    for i in range(1, len(sorted_depths)):
        d1 = sorted_depths[i - 1]
        d2 = sorted_depths[i]

        nodes1 = search_metrics[d1].nodes
        nodes2 = search_metrics[d2].nodes

        if nodes1 > 0 and nodes2 > nodes1:
            depth_diff = d2 - d1
            if depth_diff > 0:
                # b = (nodes2/nodes1)^(1/depth_diff)
                ratio = nodes2 / nodes1
                bf = ratio ** (1.0 / depth_diff)
                # Clamp to reasonable range
                bf = max(1.5, min(10.0, bf))
                branching_factors.append(bf)

    if branching_factors:
        return sum(branching_factors) / len(branching_factors)

    return 3.5  # Default


def categorize_complexity(score: float) -> str:
    """
    Categorize complexity score into human-readable level.

    Args:
        score: Complexity score 0-1

    Returns:
        "LOW" (< 0.25), "MEDIUM" (0.25-0.5),
        "HIGH" (0.5-0.75), "VERY_HIGH" (>= 0.75)
    """
    if score < 0.25:
        return "LOW"
    elif score < 0.50:
        return "MEDIUM"
    elif score < 0.75:
        return "HIGH"
    else:
        return "VERY_HIGH"


def calculate_complexity_heuristics(
    evaluations: dict[int, int],
    search_metrics: Optional[dict[int, EngineSearchMetrics]],
    gap_metrics: Optional[dict[int, GapMetricResult]],
    first_consistent_depth: Optional[int] = None,
) -> PositionComplexityHeuristics:
    """
    Aggregate all engine metrics into a composite complexity score.

    Weights:
    - Eval volatility: 30% (high volatility = complex)
    - Gap metric: 25% (small gap = complex, inverted)
    - Convergence: 20% (late convergence = complex)
    - Branching factor: 25% (high branching = complex)

    Args:
        evaluations: Dict mapping depth -> centipawn evaluation
        search_metrics: Dict mapping depth -> EngineSearchMetrics (optional)
        gap_metrics: Dict mapping depth -> GapMetricResult (optional)
        first_consistent_depth: Depth where move stabilized (optional)

    Returns:
        PositionComplexityHeuristics with all metrics and composite score
    """
    # Calculate eval volatility
    eval_volatility, eval_volatility_normalized = calculate_eval_volatility(evaluations)

    # Calculate gap metrics
    if gap_metrics:
        sorted_depths = sorted(gap_metrics.keys())
        max_depth = sorted_depths[-1] if sorted_depths else 0
        gap_at_max_depth = gap_metrics[max_depth].gap_cp if max_depth in gap_metrics else 0
        avg_gap = (
            sum(gm.gap_cp for gm in gap_metrics.values()) / len(gap_metrics)
            if gap_metrics
            else 0
        )
    else:
        gap_at_max_depth = 0
        avg_gap = 0.0

    # Gap normalized: small gap = high complexity (inverted)
    # 200cp gap = 0 complexity, 0cp gap = 1.0 complexity
    gap_normalized = max(0.0, min(1.0, 1.0 - (avg_gap / 200.0)))

    # Convergence metric: late convergence = complex
    # Normalize based on typical depth range
    if first_consistent_depth is not None:
        # Later convergence = higher complexity
        # Depth 3 = 0, Depth 21 = 1.0
        convergence_normalized = min(1.0, max(0.0, (first_consistent_depth - 3) / 18.0))
    else:
        convergence_normalized = 0.5  # Default if unknown

    # Calculate branching factor
    branching_factor = estimate_branching_factor(search_metrics)
    # Normalize: 2 = 0, 8 = 1.0
    branching_normalized = min(1.0, max(0.0, (branching_factor - 2.0) / 6.0))

    # Node statistics
    if search_metrics:
        total_nodes = sum(sm.nodes for sm in search_metrics.values())
        nodes_per_depth = {depth: sm.nodes for depth, sm in search_metrics.items()}
    else:
        total_nodes = 0
        nodes_per_depth = {}

    # Composite complexity score (weighted average)
    complexity_score = (
        0.30 * eval_volatility_normalized
        + 0.25 * gap_normalized
        + 0.20 * convergence_normalized
        + 0.25 * branching_normalized
    )

    return PositionComplexityHeuristics(
        eval_volatility=eval_volatility,
        eval_volatility_normalized=eval_volatility_normalized,
        gap_at_max_depth=gap_at_max_depth,
        avg_gap=avg_gap,
        convergence_depth=first_consistent_depth,
        total_nodes=total_nodes,
        nodes_per_depth=nodes_per_depth,
        branching_factor_estimate=branching_factor,
        complexity_score=complexity_score,
        complexity_category=categorize_complexity(complexity_score),
    )
