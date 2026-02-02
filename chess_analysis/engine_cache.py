"""
Engine evaluation caching for chess analysis.

This module provides parquet-based caching for:
1. Position evaluations: FEN + depth → eval, best move, WDL percentages
2. Position move evaluations: FEN + depth → all legal moves with their evaluations

The cache includes engine name and version for compatibility when engines are updated.
"""

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd


# =============================================================================
# Constants
# =============================================================================

DEFAULT_CACHE_DIR = Path("data/engine_cache")

# Cache file names
POSITION_EVAL_CACHE = "position_evals.parquet"
MOVE_EVALS_CACHE = "move_evals.parquet"
GAP_METRICS_CACHE = "gap_metrics.parquet"


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class PositionEval:
    """Cached evaluation for a single position at a specific depth."""
    fen: str
    depth: int
    engine_name: str
    engine_version: str

    # Evaluation results
    eval_cp: int  # Centipawns from white's perspective
    mate_in: Optional[int]  # Mate in N moves (None if no mate)
    best_move_uci: str  # Best move in UCI notation
    best_move_san: Optional[str]  # Best move in SAN notation (if available)
    pv_uci: str  # Principal variation as space-separated UCI moves

    # Win/Draw/Loss percentages (if available from engine)
    wdl_win: Optional[int]  # Win percentage * 10 (e.g., 534 = 53.4%)
    wdl_draw: Optional[int]
    wdl_loss: Optional[int]

    # Metadata
    cached_at: str  # ISO timestamp
    nodes_searched: Optional[int] = None
    time_ms: Optional[int] = None

    # Extended search stats (optional, for complexity heuristics)
    nps: Optional[int] = None  # Nodes per second
    seldepth: Optional[int] = None  # Selective search depth


@dataclass
class MoveEval:
    """Cached evaluation for a single move in a position."""
    fen: str
    depth: int
    engine_name: str
    engine_version: str
    move_uci: str
    move_san: Optional[str]

    # Evaluation after this move
    eval_cp: int
    mate_in: Optional[int]

    # Move ranking
    rank: int  # 1 = best move, 2 = second best, etc.
    eval_loss_cp: int  # Centipawn loss vs best move

    # Metadata
    cached_at: str


# =============================================================================
# Cache Key Generation
# =============================================================================

def normalize_fen(fen: str) -> str:
    """
    Normalize FEN for consistent cache keys.

    Removes the halfmove and fullmove counters as they don't affect evaluation.
    Keeps: piece placement, active color, castling rights, en passant square.
    """
    parts = fen.split()
    if len(parts) >= 4:
        # Keep first 4 parts: position, turn, castling, en passant
        return " ".join(parts[:4])
    return fen


def make_cache_key(fen: str, depth: int, engine_name: str, engine_version: str) -> str:
    """Create a unique cache key for a position evaluation."""
    normalized = normalize_fen(fen)
    key_string = f"{normalized}|{depth}|{engine_name}|{engine_version}"
    return hashlib.sha256(key_string.encode()).hexdigest()[:16]


# =============================================================================
# Position Evaluation Cache
# =============================================================================

class PositionEvalCache:
    """
    Cache for single-position evaluations at a given depth.

    Usage:
        cache = PositionEvalCache()

        # Check cache
        cached = cache.get(fen, depth, engine_name, engine_version)
        if cached:
            print(f"Cached eval: {cached['eval_cp']}")
        else:
            # Run engine analysis
            result = engine.analyze(board, depth)
            cache.put(fen, depth, engine_name, engine_version, result)
    """

    def __init__(self, cache_dir: Path = DEFAULT_CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_file = self.cache_dir / POSITION_EVAL_CACHE
        self._df: Optional[pd.DataFrame] = None
        self._dirty = False

    def _ensure_loaded(self):
        """Load cache from disk if not already loaded."""
        if self._df is None:
            if self.cache_file.exists():
                self._df = pd.read_parquet(self.cache_file)
            else:
                self._df = pd.DataFrame(columns=[
                    "cache_key", "fen", "fen_normalized", "depth",
                    "engine_name", "engine_version",
                    "eval_cp", "mate_in", "best_move_uci", "best_move_san", "pv_uci",
                    "wdl_win", "wdl_draw", "wdl_loss",
                    "cached_at", "nodes_searched", "time_ms",
                ])

    def get(
        self,
        fen: str,
        depth: int,
        engine_name: str,
        engine_version: str,
    ) -> Optional[dict]:
        """
        Get cached evaluation for a position.

        Args:
            fen: Position FEN string
            depth: Analysis depth
            engine_name: Engine name (e.g., "stockfish")
            engine_version: Engine version (e.g., "16.1")

        Returns:
            Dictionary with cached evaluation, or None if not found.
        """
        self._ensure_loaded()

        cache_key = make_cache_key(fen, depth, engine_name, engine_version)
        matches = self._df[self._df["cache_key"] == cache_key]

        if len(matches) == 0:
            return None

        row = matches.iloc[0]
        return row.to_dict()

    def get_any_depth(
        self,
        fen: str,
        min_depth: int,
        engine_name: str,
        engine_version: str,
    ) -> Optional[dict]:
        """
        Get cached evaluation at any depth >= min_depth.

        Useful when you want at least a certain depth but will accept deeper.

        Returns:
            Dictionary with cached evaluation at highest available depth, or None.
        """
        self._ensure_loaded()

        fen_normalized = normalize_fen(fen)
        matches = self._df[
            (self._df["fen_normalized"] == fen_normalized) &
            (self._df["depth"] >= min_depth) &
            (self._df["engine_name"] == engine_name) &
            (self._df["engine_version"] == engine_version)
        ]

        if len(matches) == 0:
            return None

        # Return the deepest analysis
        best = matches.loc[matches["depth"].idxmax()]
        return best.to_dict()

    def put(
        self,
        fen: str,
        depth: int,
        engine_name: str,
        engine_version: str,
        eval_cp: int,
        best_move_uci: str,
        mate_in: Optional[int] = None,
        best_move_san: Optional[str] = None,
        pv_uci: Optional[str] = None,
        wdl_win: Optional[int] = None,
        wdl_draw: Optional[int] = None,
        wdl_loss: Optional[int] = None,
        nodes_searched: Optional[int] = None,
        time_ms: Optional[int] = None,
    ):
        """
        Store an evaluation in the cache.

        Args:
            fen: Position FEN string
            depth: Analysis depth
            engine_name: Engine name
            engine_version: Engine version
            eval_cp: Evaluation in centipawns (from white's perspective)
            best_move_uci: Best move in UCI notation
            mate_in: Mate in N moves (positive = white mates, negative = black mates)
            best_move_san: Best move in SAN notation (optional)
            pv_uci: Principal variation as space-separated UCI moves
            wdl_win: Win percentage * 10 (optional)
            wdl_draw: Draw percentage * 10 (optional)
            wdl_loss: Loss percentage * 10 (optional)
            nodes_searched: Nodes searched (optional)
            time_ms: Time spent in milliseconds (optional)
        """
        self._ensure_loaded()

        cache_key = make_cache_key(fen, depth, engine_name, engine_version)
        fen_normalized = normalize_fen(fen)

        # Remove existing entry if present
        self._df = self._df[self._df["cache_key"] != cache_key]

        # Add new entry
        new_row = pd.DataFrame([{
            "cache_key": cache_key,
            "fen": fen,
            "fen_normalized": fen_normalized,
            "depth": depth,
            "engine_name": engine_name,
            "engine_version": engine_version,
            "eval_cp": eval_cp,
            "mate_in": mate_in,
            "best_move_uci": best_move_uci,
            "best_move_san": best_move_san,
            "pv_uci": pv_uci or "",
            "wdl_win": wdl_win,
            "wdl_draw": wdl_draw,
            "wdl_loss": wdl_loss,
            "cached_at": datetime.now().isoformat(),
            "nodes_searched": nodes_searched,
            "time_ms": time_ms,
        }])

        self._df = pd.concat([self._df, new_row], ignore_index=True)
        self._dirty = True

    def save(self):
        """Save cache to disk."""
        if self._dirty and self._df is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._df.to_parquet(self.cache_file, index=False)
            self._dirty = False

    def stats(self) -> dict:
        """Get cache statistics."""
        self._ensure_loaded()
        return {
            "total_positions": len(self._df),
            "unique_fens": self._df["fen_normalized"].nunique() if len(self._df) > 0 else 0,
            "engines": self._df["engine_name"].unique().tolist() if len(self._df) > 0 else [],
            "depth_range": (
                int(self._df["depth"].min()),
                int(self._df["depth"].max())
            ) if len(self._df) > 0 else (0, 0),
        }

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.save()


# =============================================================================
# Move Evaluations Cache
# =============================================================================

class MoveEvalsCache:
    """
    Cache for all legal moves evaluated at a given position and depth.

    This is useful for:
    - Move quality analysis (how does played move compare to alternatives)
    - Finding second-best moves for complexity analysis
    - Caching multipv analysis results

    Usage:
        cache = MoveEvalsCache()

        # Check cache
        moves = cache.get(fen, depth, engine_name, engine_version)
        if moves:
            print(f"Cached {len(moves)} moves")
        else:
            # Run multipv engine analysis
            results = engine.analyze_all_moves(board, depth)
            cache.put(fen, depth, engine_name, engine_version, results)
    """

    def __init__(self, cache_dir: Path = DEFAULT_CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_file = self.cache_dir / MOVE_EVALS_CACHE
        self._df: Optional[pd.DataFrame] = None
        self._dirty = False

    def _ensure_loaded(self):
        """Load cache from disk if not already loaded."""
        if self._df is None:
            if self.cache_file.exists():
                self._df = pd.read_parquet(self.cache_file)
            else:
                self._df = pd.DataFrame(columns=[
                    "position_key", "fen", "fen_normalized", "depth",
                    "engine_name", "engine_version",
                    "move_uci", "move_san",
                    "eval_cp", "mate_in",
                    "rank", "eval_loss_cp",
                    "cached_at",
                ])

    def get(
        self,
        fen: str,
        depth: int,
        engine_name: str,
        engine_version: str,
    ) -> Optional[list[dict]]:
        """
        Get cached move evaluations for a position.

        Args:
            fen: Position FEN string
            depth: Analysis depth
            engine_name: Engine name
            engine_version: Engine version

        Returns:
            List of move evaluation dicts sorted by rank, or None if not found.
        """
        self._ensure_loaded()

        position_key = make_cache_key(fen, depth, engine_name, engine_version)
        matches = self._df[self._df["position_key"] == position_key]

        if len(matches) == 0:
            return None

        # Sort by rank and return as list of dicts
        matches = matches.sort_values("rank")
        return matches.to_dict("records")

    def get_best_move(
        self,
        fen: str,
        depth: int,
        engine_name: str,
        engine_version: str,
    ) -> Optional[dict]:
        """Get just the best move from cached evaluations."""
        moves = self.get(fen, depth, engine_name, engine_version)
        if moves:
            return moves[0]  # Already sorted by rank
        return None

    def put(
        self,
        fen: str,
        depth: int,
        engine_name: str,
        engine_version: str,
        moves: list[dict],
    ):
        """
        Store move evaluations in the cache.

        Args:
            fen: Position FEN string
            depth: Analysis depth
            engine_name: Engine name
            engine_version: Engine version
            moves: List of move dicts, each containing:
                - move_uci: Move in UCI notation (required)
                - move_san: Move in SAN notation (optional)
                - eval_cp: Evaluation in centipawns (required)
                - mate_in: Mate in N (optional)
                - rank: Move rank, 1=best (optional, will be computed if missing)
        """
        self._ensure_loaded()

        position_key = make_cache_key(fen, depth, engine_name, engine_version)
        fen_normalized = normalize_fen(fen)

        # Remove existing entries for this position
        self._df = self._df[self._df["position_key"] != position_key]

        # Sort moves by eval (best first) and assign ranks if missing
        sorted_moves = sorted(
            moves,
            key=lambda m: m.get("eval_cp", -99999),
            reverse=True  # Higher eval = better for white
        )

        best_eval = sorted_moves[0].get("eval_cp", 0) if sorted_moves else 0
        cached_at = datetime.now().isoformat()

        new_rows = []
        for i, move in enumerate(sorted_moves):
            rank = move.get("rank", i + 1)
            eval_cp = move.get("eval_cp", 0)
            eval_loss = best_eval - eval_cp

            new_rows.append({
                "position_key": position_key,
                "fen": fen,
                "fen_normalized": fen_normalized,
                "depth": depth,
                "engine_name": engine_name,
                "engine_version": engine_version,
                "move_uci": move["move_uci"],
                "move_san": move.get("move_san"),
                "eval_cp": eval_cp,
                "mate_in": move.get("mate_in"),
                "rank": rank,
                "eval_loss_cp": eval_loss,
                "cached_at": cached_at,
            })

        if new_rows:
            new_df = pd.DataFrame(new_rows)
            self._df = pd.concat([self._df, new_df], ignore_index=True)
            self._dirty = True

    def save(self):
        """Save cache to disk."""
        if self._dirty and self._df is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._df.to_parquet(self.cache_file, index=False)
            self._dirty = False

    def stats(self) -> dict:
        """Get cache statistics."""
        self._ensure_loaded()
        if len(self._df) == 0:
            return {
                "total_moves": 0,
                "unique_positions": 0,
                "avg_moves_per_position": 0,
                "engines": [],
            }

        return {
            "total_moves": len(self._df),
            "unique_positions": self._df["position_key"].nunique(),
            "avg_moves_per_position": len(self._df) / self._df["position_key"].nunique(),
            "engines": self._df["engine_name"].unique().tolist(),
            "depth_range": (
                int(self._df["depth"].min()),
                int(self._df["depth"].max())
            ),
        }

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.save()


# =============================================================================
# Gap Metric Cache (for MultiPV analysis)
# =============================================================================

class GapMetricCache:
    """
    Cache for gap metrics between best and second-best moves.

    This stores MultiPV analysis results for calculating the gap metric,
    which measures how clear the best move is (large gap = clear best,
    small gap = multiple viable options).

    Usage:
        cache = GapMetricCache()

        # Check cache
        gap = cache.get(fen, depth, multipv, engine_name, engine_version)
        if gap:
            print(f"Cached gap: {gap['gap_cp']}cp")
        else:
            # Run multipv engine analysis
            result = engine.analyze_multipv(board, depth, multipv=2)
            cache.put(fen, depth, multipv, engine_name, engine_version, result)
    """

    def __init__(self, cache_dir: Path = DEFAULT_CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_file = self.cache_dir / GAP_METRICS_CACHE
        self._df: Optional[pd.DataFrame] = None
        self._dirty = False

    def _ensure_loaded(self):
        """Load cache from disk if not already loaded."""
        if self._df is None:
            if self.cache_file.exists():
                self._df = pd.read_parquet(self.cache_file)
            else:
                self._df = pd.DataFrame(columns=[
                    "cache_key", "fen", "fen_normalized", "depth", "multipv",
                    "engine_name", "engine_version",
                    "best_move", "best_eval",
                    "second_move", "second_eval",
                    "gap_cp",
                    "cached_at",
                ])

    def _make_cache_key(
        self,
        fen: str,
        depth: int,
        multipv: int,
        engine_name: str,
        engine_version: str,
    ) -> str:
        """Create a unique cache key for gap metric."""
        normalized = normalize_fen(fen)
        key_string = f"{normalized}|{depth}|{multipv}|{engine_name}|{engine_version}"
        return hashlib.sha256(key_string.encode()).hexdigest()[:16]

    def get(
        self,
        fen: str,
        depth: int,
        multipv: int,
        engine_name: str,
        engine_version: str,
    ) -> Optional[dict]:
        """
        Get cached gap metric for a position.

        Args:
            fen: Position FEN string
            depth: Analysis depth
            multipv: Number of PVs analyzed
            engine_name: Engine name
            engine_version: Engine version

        Returns:
            Dictionary with gap metric data, or None if not found.
        """
        self._ensure_loaded()

        cache_key = self._make_cache_key(fen, depth, multipv, engine_name, engine_version)
        matches = self._df[self._df["cache_key"] == cache_key]

        if len(matches) == 0:
            return None

        row = matches.iloc[0]
        return row.to_dict()

    def put(
        self,
        fen: str,
        depth: int,
        multipv: int,
        engine_name: str,
        engine_version: str,
        best_move: str,
        best_eval: int,
        second_move: Optional[str] = None,
        second_eval: Optional[int] = None,
        gap_cp: int = 0,
    ):
        """
        Store gap metric in the cache.

        Args:
            fen: Position FEN string
            depth: Analysis depth
            multipv: Number of PVs analyzed
            engine_name: Engine name
            engine_version: Engine version
            best_move: Best move in UCI notation
            best_eval: Best move evaluation in centipawns
            second_move: Second-best move in UCI notation (optional)
            second_eval: Second-best move evaluation in centipawns (optional)
            gap_cp: Gap between best and second-best in centipawns
        """
        self._ensure_loaded()

        cache_key = self._make_cache_key(fen, depth, multipv, engine_name, engine_version)
        fen_normalized = normalize_fen(fen)

        # Remove existing entry if present
        self._df = self._df[self._df["cache_key"] != cache_key]

        # Add new entry
        new_row = pd.DataFrame([{
            "cache_key": cache_key,
            "fen": fen,
            "fen_normalized": fen_normalized,
            "depth": depth,
            "multipv": multipv,
            "engine_name": engine_name,
            "engine_version": engine_version,
            "best_move": best_move,
            "best_eval": best_eval,
            "second_move": second_move,
            "second_eval": second_eval,
            "gap_cp": gap_cp,
            "cached_at": datetime.now().isoformat(),
        }])

        self._df = pd.concat([self._df, new_row], ignore_index=True)
        self._dirty = True

    def save(self):
        """Save cache to disk."""
        if self._dirty and self._df is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._df.to_parquet(self.cache_file, index=False)
            self._dirty = False

    def stats(self) -> dict:
        """Get cache statistics."""
        self._ensure_loaded()
        if len(self._df) == 0:
            return {
                "total_entries": 0,
                "unique_positions": 0,
                "engines": [],
            }

        return {
            "total_entries": len(self._df),
            "unique_positions": self._df["fen_normalized"].nunique(),
            "engines": self._df["engine_name"].unique().tolist(),
            "depth_range": (
                int(self._df["depth"].min()),
                int(self._df["depth"].max())
            ) if len(self._df) > 0 else (0, 0),
            "avg_gap_cp": float(self._df["gap_cp"].mean()) if len(self._df) > 0 else 0,
        }

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.save()


# =============================================================================
# Utility Functions
# =============================================================================

def get_engine_version(engine_path: str = "stockfish") -> tuple[str, str]:
    """
    Get engine name and version from the engine binary.

    Returns:
        Tuple of (engine_name, engine_version)
    """
    import subprocess
    try:
        result = subprocess.run(
            [engine_path],
            input="uci\nquit\n",
            capture_output=True,
            text=True,
            timeout=5,
        )
        # Parse "id name Stockfish 16.1" from output
        for line in result.stdout.split("\n"):
            if line.startswith("id name"):
                parts = line.replace("id name", "").strip().split()
                if parts:
                    name = parts[0].lower()
                    version = parts[1] if len(parts) > 1 else "unknown"
                    return name, version
    except Exception:
        pass

    return "stockfish", "unknown"


def clear_cache(cache_dir: Path = DEFAULT_CACHE_DIR):
    """Remove all cached evaluations."""
    import shutil
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        print(f"Cleared cache at {cache_dir}")


# =============================================================================
# Cached Engine Analyzer Wrapper
# =============================================================================

class CachedEngineAnalyzer:
    """
    Wrapper around EngineAnalyzer that caches results to parquet files.

    Usage:
        with CachedEngineAnalyzer(depth=20) as engine:
            result = engine.analyze(board)  # Cached!
            multi = engine.analyze_multi_depth(board, [5, 10, 20])  # Also cached!

    The cache persists across sessions, so repeated analysis of the same
    positions (common when re-running notebooks) is instant.
    """

    def __init__(
        self,
        stockfish_path: Optional[str] = None,
        depth: int = 20,
        threads: int = 1,
        hash_mb: int = 256,
        cache_dir: Path = DEFAULT_CACHE_DIR,
        enable_cache: bool = True,
    ):
        """
        Initialize cached engine analyzer.

        Args:
            stockfish_path: Path to stockfish binary
            depth: Default analysis depth
            threads: CPU threads for engine
            hash_mb: Hash table size in MB
            cache_dir: Directory for cache files
            enable_cache: If False, bypass cache (for testing)
        """
        # Import here to avoid circular import
        from .engine import EngineAnalyzer

        self.engine = EngineAnalyzer(
            stockfish_path=stockfish_path,
            depth=depth,
            threads=threads,
            hash_mb=hash_mb,
        )
        self.depth = depth
        self.enable_cache = enable_cache

        # Get engine version for cache keys
        self.engine_name, self.engine_version = get_engine_version(
            stockfish_path or "stockfish"
        )

        # Initialize caches
        self.position_cache = PositionEvalCache(cache_dir) if enable_cache else None
        self.moves_cache = MoveEvalsCache(cache_dir) if enable_cache else None
        self.gap_cache = GapMetricCache(cache_dir) if enable_cache else None

        # Stats
        self.cache_hits = 0
        self.cache_misses = 0

    def __enter__(self):
        self.engine.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.engine.stop()
        if self.position_cache:
            self.position_cache.save()
        if self.moves_cache:
            self.moves_cache.save()
        if self.gap_cache:
            self.gap_cache.save()

    def analyze(
        self,
        board,
        depth: Optional[int] = None,
        include_material: bool = True,
    ) -> dict:
        """
        Analyze a position with caching.

        Args:
            board: chess.Board position to analyze
            depth: Analysis depth (uses default if None)
            include_material: Include material score calculation

        Returns:
            Dictionary with score, mate, pv, depth, material, eval_vs_material
        """
        actual_depth = depth or self.depth
        fen = board.fen()

        # Check cache
        if self.enable_cache and self.position_cache:
            cached = self.position_cache.get(
                fen, actual_depth, self.engine_name, self.engine_version
            )
            if cached is not None:
                self.cache_hits += 1
                # Reconstruct result format from cache
                pv_moves = []
                if cached.get("pv_uci"):
                    import chess
                    temp_board = chess.Board(fen)
                    for uci in cached["pv_uci"].split():
                        try:
                            pv_moves.append(chess.Move.from_uci(uci))
                        except ValueError:
                            break

                result = {
                    "score": cached["eval_cp"],
                    "mate": cached["mate_in"],
                    "pv": pv_moves,
                    "depth": cached["depth"],
                }

                if include_material:
                    from .engine import calculate_material_score
                    material = calculate_material_score(board)
                    result["material"] = material
                    result["eval_vs_material"] = cached["eval_cp"] - material

                return result

        # Cache miss - run engine
        self.cache_misses += 1
        result = self.engine.analyze(board, actual_depth, include_material)

        # Store in cache
        if self.enable_cache and self.position_cache:
            pv_uci = " ".join(m.uci() for m in result.get("pv", []))
            self.position_cache.put(
                fen=fen,
                depth=actual_depth,
                engine_name=self.engine_name,
                engine_version=self.engine_version,
                eval_cp=result["score"],
                best_move_uci=result["pv"][0].uci() if result.get("pv") else "",
                mate_in=result.get("mate"),
                pv_uci=pv_uci,
            )

        return result

    def analyze_multi_depth(
        self,
        board,
        depths: Optional[list[int]] = None,
    ):
        """
        Analyze a position at multiple depths with caching.

        Note: This caches individual depth results, not the full MultiDepthResult.
        This allows partial cache hits when some depths are cached but not others.

        Args:
            board: chess.Board position to analyze
            depths: List of depths to analyze (uses DEFAULT_MULTI_DEPTHS if None)

        Returns:
            MultiDepthResult with evaluations at each depth
        """
        from .engine import DEFAULT_MULTI_DEPTHS

        if depths is None:
            depths = DEFAULT_MULTI_DEPTHS

        fen = board.fen()
        sorted_depths = sorted(depths)

        # Check which depths are cached
        cached_depths = {}
        uncached_depths = []

        if self.enable_cache and self.position_cache:
            for d in sorted_depths:
                cached = self.position_cache.get(
                    fen, d, self.engine_name, self.engine_version
                )
                if cached is not None:
                    cached_depths[d] = cached
                    self.cache_hits += 1
                else:
                    uncached_depths.append(d)
                    self.cache_misses += 1
        else:
            uncached_depths = sorted_depths

        # Run engine for uncached depths
        if uncached_depths:
            # Analyze at max uncached depth (engine gives all depths up to max)
            max_uncached = max(uncached_depths)
            for d in uncached_depths:
                # Analyze each uncached depth individually
                result = self.engine.analyze(board, d, include_material=False)

                # Cache the result
                if self.enable_cache and self.position_cache:
                    pv_uci = " ".join(m.uci() for m in result.get("pv", []))
                    self.position_cache.put(
                        fen=fen,
                        depth=d,
                        engine_name=self.engine_name,
                        engine_version=self.engine_version,
                        eval_cp=result["score"],
                        best_move_uci=result["pv"][0].uci() if result.get("pv") else "",
                        mate_in=result.get("mate"),
                        pv_uci=pv_uci,
                    )
                    cached_depths[d] = {
                        "eval_cp": result["score"],
                        "mate_in": result.get("mate"),
                        "best_move_uci": result["pv"][0].uci() if result.get("pv") else "",
                        "pv_uci": pv_uci,
                        "depth": d,
                    }

        # Build MultiDepthResult from cached data
        from .engine import MultiDepthResult, DepthTransition, calculate_material_score, SIGNIFICANT_EVAL_CHANGE

        evaluations = {}
        mates = {}
        best_moves = {}
        pvs = {}

        for d in sorted_depths:
            cached = cached_depths.get(d, {})
            evaluations[d] = cached.get("eval_cp", 0)
            mates[d] = cached.get("mate_in")
            best_moves[d] = cached.get("best_move_uci", "")
            pv_str = cached.get("pv_uci", "")
            pvs[d] = pv_str.split() if pv_str else []

        # Calculate consistency metrics
        move_consistency, first_consistent, changes = self._analyze_move_consistency(
            sorted_depths, best_moves
        )

        # Calculate eval swing
        eval_values = list(evaluations.values())
        eval_swing = max(eval_values) - min(eval_values) if eval_values else 0

        # Calculate depth transitions
        depth_transitions = []
        max_eval_change = 0
        unstable_depths = []

        for i in range(1, len(sorted_depths)):
            prev_d = sorted_depths[i - 1]
            curr_d = sorted_depths[i]
            prev_eval = evaluations.get(prev_d, 0)
            curr_eval = evaluations.get(curr_d, 0)
            eval_change = curr_eval - prev_eval
            prev_move = best_moves.get(prev_d, "")
            curr_move = best_moves.get(curr_d, "")
            move_changed = prev_move != curr_move

            transition = DepthTransition(
                from_depth=prev_d,
                to_depth=curr_d,
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
                unstable_depths.append(curr_d)

        # Material score
        material_score = calculate_material_score(board)
        max_depth_eval = evaluations.get(sorted_depths[-1], 0) if sorted_depths else 0
        eval_vs_material = max_depth_eval - material_score

        return MultiDepthResult(
            fen=fen,
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
        """Analyze how consistent the best move is across depths."""
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

            if first_consistent is None and move == final_move:
                first_consistent = depth
            elif move != final_move:
                first_consistent = None

            prev_move = move

        if first_consistent is None and final_move is not None:
            first_consistent = sorted_depths[-1]

        return changes == 0, first_consistent, changes

    def analyze_multi_depth_extended(
        self,
        board,
        depths: Optional[list[int]] = None,
        multipv: int = 2,
        capture_search_stats: bool = True,
    ):
        """
        Extended multi-depth analysis with caching for search stats and gap metrics.

        This method delegates to the engine's analyze_multi_depth_extended() but
        caches individual depth results and gap metrics for efficiency.

        Args:
            board: chess.Board position to analyze
            depths: List of depths to analyze (uses DEFAULT_MULTI_DEPTHS if None)
            multipv: Number of principal variations for gap metric (default 2)
            capture_search_stats: Whether to capture nodes/nps/time (default True)

        Returns:
            MultiDepthResult with populated search_metrics, gap_metrics,
            and complexity_heuristics fields
        """
        from .engine import DEFAULT_MULTI_DEPTHS

        if depths is None:
            depths = DEFAULT_MULTI_DEPTHS

        fen = board.fen()
        sorted_depths = sorted(depths)

        # Check gap cache for all depths
        cached_gaps = {}
        uncached_gap_depths = []

        if self.enable_cache and self.gap_cache:
            for d in sorted_depths:
                cached = self.gap_cache.get(
                    fen, d, multipv, self.engine_name, self.engine_version
                )
                if cached is not None:
                    cached_gaps[d] = cached
                    self.cache_hits += 1
                else:
                    uncached_gap_depths.append(d)
                    self.cache_misses += 1
        else:
            uncached_gap_depths = sorted_depths

        # If any depths need analysis, run the full extended analysis
        if uncached_gap_depths:
            result = self.engine.analyze_multi_depth_extended(
                board,
                depths=depths,
                multipv=multipv,
                capture_search_stats=capture_search_stats,
            )

            # Cache the gap metrics for uncached depths
            if self.enable_cache and self.gap_cache and result.gap_metrics:
                for d in uncached_gap_depths:
                    if d in result.gap_metrics:
                        gm = result.gap_metrics[d]
                        self.gap_cache.put(
                            fen=fen,
                            depth=d,
                            multipv=multipv,
                            engine_name=self.engine_name,
                            engine_version=self.engine_version,
                            best_move=gm.best_move,
                            best_eval=gm.best_eval,
                            second_move=gm.second_move,
                            second_eval=gm.second_eval,
                            gap_cp=gm.gap_cp,
                        )

            return result
        else:
            # All depths were cached - reconstruct from cache
            # Run standard multi-depth analysis (non-extended) and merge gap data
            result = self.analyze_multi_depth(board, depths)

            # Reconstruct gap_metrics from cache
            from .engine import (
                GapMetricResult,
                EngineSearchMetrics,
                calculate_complexity_heuristics,
            )

            gap_metrics = {}
            for d, cached in cached_gaps.items():
                gap_metrics[d] = GapMetricResult(
                    depth=d,
                    best_move=cached.get("best_move", ""),
                    best_eval=cached.get("best_eval", 0),
                    second_move=cached.get("second_move"),
                    second_eval=cached.get("second_eval"),
                    gap_cp=cached.get("gap_cp", 0),
                )

            # Calculate complexity heuristics
            complexity_heuristics = calculate_complexity_heuristics(
                evaluations=result.evaluations,
                search_metrics=None,  # No search metrics from cache
                gap_metrics=gap_metrics,
                first_consistent_depth=result.first_consistent_depth,
            )

            # Update result with gap metrics and complexity heuristics
            result.gap_metrics = gap_metrics
            result.complexity_heuristics = complexity_heuristics

            return result

    def cache_stats(self) -> dict:
        """Get cache statistics for this session."""
        hit_rate = (
            self.cache_hits / (self.cache_hits + self.cache_misses)
            if (self.cache_hits + self.cache_misses) > 0
            else 0
        )
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": f"{hit_rate:.1%}",
            "engine": f"{self.engine_name} {self.engine_version}",
            "position_cache": self.position_cache.stats() if self.position_cache else {},
            "moves_cache": self.moves_cache.stats() if self.moves_cache else {},
            "gap_cache": self.gap_cache.stats() if self.gap_cache else {},
        }
