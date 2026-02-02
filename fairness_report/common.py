"""
Shared setup and utilities for fairness report notebooks.

This module provides:
- Path configuration for the project
- Common imports from chess_analysis module
- Notebook display helpers
- Phase data loading/saving utilities

Usage in notebooks:
    import sys
    sys.path.insert(0, '..')
    from common import *
    setup_notebook()
"""

import sys
from pathlib import Path
from datetime import datetime

# =============================================================================
# Path Configuration
# =============================================================================

# Project root (parent of fairness_report/)
# Note: resolve() must come FIRST to handle relative __file__ paths correctly
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Add project root to path for chess_analysis imports
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Key directories
DATA_DIR = PROJECT_ROOT / "data"
BASELINE_DIR = DATA_DIR / "baseline"
CHEATER_DIR = DATA_DIR / "cheaters"
OTHER_USERS_DIR = DATA_DIR / "other-users"
ENGINE_CACHE_DIR = DATA_DIR / "engine_cache"


def get_user_data_dir(username: str) -> Path:
    """Get the data directory for a specific user."""
    return OTHER_USERS_DIR / username.lower()


def get_report_dir(username: str) -> Path:
    """Get the report output directory for a user."""
    report_dir = get_user_data_dir(username) / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    return report_dir


def get_phase_dir(username: str, phase: str) -> Path:
    """Get the directory for a specific phase's outputs."""
    phase_dir = get_report_dir(username) / phase
    phase_dir.mkdir(parents=True, exist_ok=True)
    return phase_dir


# =============================================================================
# Chess Analysis Module Imports
# =============================================================================

# Re-export commonly used items from chess_analysis
# Notebooks can use: from common import *

from chess_analysis import (
    # Games & API
    fetch_games,
    fetch_archives,
    get_last_game_date,
    parse_pgn_file,
    find_last_book_move,
    find_last_book_move_fast,

    # Dataset building
    GameMetadata,
    PositionFeatures,
    GameFeatures,
    EloAnalysis,
    ResultPatterns,
    TerminationPatterns,
    EloRangeStats,
    EloSegmentStats,
    extract_game_metadata,
    parse_time_control,
    determine_resolution,
    extract_position_features,
    extract_game_positions,
    calculate_material,
    build_game_dataset,
    build_position_dataset,
    build_opening_book,
    aggregate_game_features,
    aggregate_all_games,
    analyze_elo_patterns,
    analyze_result_patterns,
    analyze_termination_patterns,
    save_dataset_parquet,
    save_opening_book,
    load_dataset_parquet,
    calculate_expected_win_rate,
    analyze_opponent_segments,
    extract_game_features,
    segment_games_by_elo_range,
    categorize_fragility,
    categorize_material_trajectory,

    # Engine analysis
    EngineAnalyzer,
    CachedEngineAnalyzer,
    MultiDepthResult,
    DepthTransition,
    DEFAULT_MULTI_DEPTHS,
    PIECE_VALUES,
    SIGNIFICANT_EVAL_CHANGE,
    calculate_material_score,
    # Engine complexity heuristics
    EngineSearchMetrics,
    GapMetricResult,
    PositionComplexityHeuristics,
    calculate_eval_volatility,
    estimate_branching_factor,
    categorize_complexity,
    calculate_complexity_heuristics,
    GapMetricCache,

    # Metrics
    calculate_centipawn_loss,
    calculate_acpl,
    calculate_game_accuracy,
    calculate_game_accuracy_simple,
    calculate_move_accuracy,
    calculate_move_accuracy_from_cpl,
    centipawns_to_win_percent,
    mate_to_centipawns,
    eval_to_centipawns,
    calculate_acpl_by_position,
    extract_position_data,
    classify_position,
    classify_move,
    classify_move_by_accuracy,
    classify_advantage,
    analyze_errors_by_advantage,

    # Ken Regan analysis
    analyze_game_regan,
    analyze_multiple_games_regan,
    calculate_partial_credit,
    calculate_z_score,
    compare_to_expected,
    get_expected_move_match_rate,
    ReganAnalysisResult,
    SuspiciousPosition,
    identify_suspicious_positions,
    get_regan_key_positions,
    calculate_position_difficulty,

    # Maia2 analysis
    analyze_position_maia2,
    analyze_game_maia2,
    calculate_humanness_score,
    get_surprising_moves,
    compare_to_stockfish,
    Maia2Result,

    # Time analysis
    extract_clock_times,
    analyze_time_patterns,
    detect_bot_patterns,
    merge_time_with_positions,
    calculate_time_complexity_correlation,
    TimeControl,
    classify_time_spent,
    analyze_time_distribution,

    # Tablebase
    TablebaseResult,
    TablebaseMoveCheck,
    TablebaseClient,
    probe_tablebase,
    check_tablebase_move,
    is_tablebase_position,
    analyze_endgame_accuracy,
    TablebaseConsistencyReport,
    analyze_tablebase_consistency,

    # Themes
    TacticalTheme,
    PositionalTheme,
    ThemeDetectionResult,
    detect_fork,
    detect_pin,
    detect_skewer,
    detect_discovered_attack,
    detect_discovered_check,
    detect_double_check,
    detect_hanging_piece,
    detect_back_rank_mate_threat,
    detect_sacrifice,
    detect_advanced_pawn,
    detect_exposed_king,
    detect_endgame_type,
    analyze_position_themes,
    analyze_move_themes,
    get_all_theme_names,

    # Game phase
    GamePhase,
    GamePhaseInfo,
    detect_game_phase,
    detect_game_phase_detailed,
    get_phase_transitions,
    analyze_game_phases,
    count_major_minor_pieces,
    calculate_mixedness,
    is_endgame,
    is_opening,

    # Fragility & Complexity
    calculate_fragility,
    calculate_fragility_simple,
    build_interaction_graph,
    calculate_betweenness_centrality,
    calculate_complexity,
    calculate_complexity_fast,
    analyze_position_complexity_batch,
    ComplexityResult,
    # Fragility trend/distance
    FragilityTrend,
    FragilityAnalysis,
    calculate_game_fragility,
    get_fragility_trend,
    is_pre_fragility_peak,

    # Openings
    OpeningBook,
    OpeningInfo,
    get_opening_book,
    find_last_book_ply,
    calculate_distance_from_book,
    classify_opening,

    # Position assessment
    StockfishMetrics,
    Maia2Metrics,
    TrapInfo,
    StockfishMoveDistribution,
    PositionalAssessment,
    PositionAssessmentResult,
    MATERIAL_VALUES,
    calculate_pure_material,
    calculate_raw_branching_factor,
    detect_traps_in_candidates,
    check_trickiness,
    get_tablebase_status,
    assess_position,
    assessment_to_dict,

    # Visualization
    render_position,
    render_key_position,
    display_key_position,
    find_key_positions,
    MoveArrow,
    COLORS,
    SQUARE_COLORS,

    # Caching
    TimeClassCacheState,
    CacheState,
    FetchPlan,
    OpponentProfile,
    get_shared_opponent_cache_path,
    migrate_opponent_cache_if_needed,
    init_opponent_cache,
    get_cached_opponent_profile,
    save_opponent_profile_to_cache,
    get_cache_path,
    load_cached_games_v2,
    analyze_cache_state,
    build_fetch_plan,
    merge_games,
    save_cache_with_metadata,
    query_cache_filtered,
    extract_opponents_from_games,

    # Baseline generation
    FrequentOpponent,
    GameSession,
    fetch_player_profile,
    batch_fetch_opponent_profiles,
    is_banned_account,
    extract_opponent_from_game,
    split_games_by_opponent_trust,
    analyze_frequent_opponents,
    analyze_timeout_patterns,
    analyze_banned_opponent_games,
    limit_games_per_time_class,
    analyze_openings_by_eco,
    generate_player_baseline,
    generate_combined_baseline,

    # Session detection
    detect_sessions,
    assign_session_ids_to_games,
    analyze_session_patterns,
    DEFAULT_SESSION_GAP_MINUTES,

    # Glicko-2 rating
    Glicko2Rating,
    Glicko2Result,
    RatingHistory,
    ImprovementAnalysis,
    calculate_rating_period,
    calculate_glicko2,
    track_rating_over_time,
    analyze_rating_improvement,
    correlate_with_regan_zscore,
    DEFAULT_RATING,
    DEFAULT_RD,
    DEFAULT_VOLATILITY,
    DEFAULT_TAU,
)


# =============================================================================
# Notebook Display Helpers
# =============================================================================

def setup_notebook():
    """
    Standard notebook setup - call at the start of each phase notebook.

    Sets up:
    - Warning filters
    - Pandas display options
    - Matplotlib defaults
    """
    import warnings
    warnings.filterwarnings('ignore')

    import pandas as pd
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 50)

    # Matplotlib setup
    try:
        import matplotlib.pyplot as plt
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['figure.figsize'] = (12, 6)
        plt.rcParams['font.size'] = 10
    except Exception:
        pass  # Matplotlib not available or style not found


def print_section(title: str, width: int = 70):
    """Print a section header."""
    print("=" * width)
    print(title)
    print("=" * width)


def print_subsection(title: str, width: int = 60):
    """Print a subsection header."""
    print("-" * width)
    print(title)
    print("-" * width)


# =============================================================================
# Data Loading/Saving Utilities
# =============================================================================

def load_baseline(baseline_name: str = "trusted") -> dict:
    """
    Load a baseline configuration.

    Args:
        baseline_name: "trusted" or "cheater"

    Returns:
        Baseline data dictionary
    """
    import json

    if baseline_name == "trusted":
        baseline_path = BASELINE_DIR / "combined_baseline.json"
    elif baseline_name == "cheater":
        baseline_path = CHEATER_DIR / "combined_baseline.json"
    else:
        raise ValueError(f"Unknown baseline: {baseline_name}")

    if baseline_path.exists():
        with open(baseline_path) as f:
            return json.load(f)
    return {}


def save_phase_output(username: str, phase: str, filename: str, data):
    """
    Save output from a phase.

    Args:
        username: Target username
        phase: Phase identifier (e.g., "phase1", "phase2")
        filename: Output filename
        data: Data to save (DataFrame or dict/list for JSON)
    """
    import json
    import pandas as pd

    phase_dir = get_phase_dir(username, phase)
    output_path = phase_dir / filename

    if isinstance(data, pd.DataFrame):
        if filename.endswith('.parquet'):
            data.to_parquet(output_path, index=False)
        elif filename.endswith('.csv'):
            data.to_csv(output_path, index=False)
        else:
            raise ValueError(f"Unknown format for DataFrame: {filename}")
    else:
        # Assume JSON-serializable
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    print(f"Saved: {output_path}")
    return output_path


def load_phase_output(username: str, phase: str, filename: str):
    """
    Load output from a phase.

    Args:
        username: Target username
        phase: Phase identifier
        filename: Output filename

    Returns:
        DataFrame or dict depending on file type
    """
    import json
    import pandas as pd

    phase_dir = get_phase_dir(username, phase)
    input_path = phase_dir / filename

    if not input_path.exists():
        raise FileNotFoundError(f"Phase output not found: {input_path}")

    if filename.endswith('.parquet'):
        return pd.read_parquet(input_path)
    elif filename.endswith('.csv'):
        return pd.read_csv(input_path)
    elif filename.endswith('.json'):
        with open(input_path) as f:
            return json.load(f)
    else:
        raise ValueError(f"Unknown format: {filename}")


# =============================================================================
# Papermill Parameter Helpers
# =============================================================================

def validate_parameters(username: str, **kwargs):
    """
    Validate required parameters at the start of a notebook.

    Args:
        username: Required username parameter
        **kwargs: Additional parameters to validate

    Raises:
        ValueError if parameters are invalid
    """
    if not username or username == "default_user":
        raise ValueError("USERNAME parameter must be set to a valid Chess.com username")

    for key, value in kwargs.items():
        if value is None:
            raise ValueError(f"Parameter {key} must not be None")


def get_run_timestamp() -> str:
    """Get a timestamp string for the current run."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")
