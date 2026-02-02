"""
Maia2 human move prediction analysis.

Maia2 is a skill-aware chess engine that predicts human moves based on
the player's rating level. It can be used to assess how "human-like"
a player's moves are.

Reference: https://github.com/CSSLab/maia2
"""

import chess
from dataclasses import dataclass
from typing import Optional


# Lazy loading of maia2 to avoid import errors if not installed
_maia2_models = {}
_prepared = None


def _get_model(game_type: str):
    """
    Get or load a Maia2 model for the specified game type.

    Args:
        game_type: "rapid" or "blitz"

    Returns:
        Loaded Maia2 model
    """
    global _maia2_models, _prepared

    if game_type not in ["rapid", "blitz"]:
        raise ValueError(f"Invalid game type: {game_type}. Must be 'rapid' or 'blitz'")

    if game_type not in _maia2_models:
        from maia2 import model, inference

        print(f"Loading Maia2 {game_type} model...")
        _maia2_models[game_type] = model.from_pretrained(
            type=game_type,
            device="cpu",
            save_root="./maia2_models"
        )

        if _prepared is None:
            _prepared = inference.prepare()

    return _maia2_models[game_type], _prepared


@dataclass
class Maia2Result:
    """Result from Maia2 analysis of a single position."""
    fen: str
    played_move: str
    move_probability: float  # Probability Maia2 assigns to the played move
    top_move: str  # Most likely move according to Maia2
    top_move_probability: float
    win_probability: float
    move_rank: int  # Rank of played move in Maia2's predictions (1 = most likely)
    is_top_choice: bool  # Whether played move was Maia2's top prediction
    all_probabilities: dict  # Full move probability distribution


def analyze_position_maia2(
    fen: str,
    played_move: str,
    elo_self: int,
    elo_oppo: int,
    game_type: str = "rapid",
) -> Maia2Result:
    """
    Analyze a single position with Maia2.

    Args:
        fen: Position in FEN notation
        played_move: The move that was played (UCI format, e.g., "e2e4")
        elo_self: Rating of the player to move
        elo_oppo: Rating of the opponent
        game_type: "rapid" or "blitz"

    Returns:
        Maia2Result with move probabilities and humanness metrics
    """
    from maia2 import inference

    maia2_model, prepared = _get_model(game_type)

    # Get move probabilities from Maia2
    move_probs, win_prob = inference.inference_each(
        maia2_model, prepared, fen, elo_self, elo_oppo
    )

    # Sort moves by probability
    sorted_moves = sorted(move_probs.items(), key=lambda x: x[1], reverse=True)

    # Find the played move's probability and rank
    played_prob = move_probs.get(played_move, 0.0)
    move_rank = 1
    for i, (move, prob) in enumerate(sorted_moves):
        if move == played_move:
            move_rank = i + 1
            break

    top_move, top_prob = sorted_moves[0] if sorted_moves else (None, 0.0)

    return Maia2Result(
        fen=fen,
        played_move=played_move,
        move_probability=played_prob,
        top_move=top_move,
        top_move_probability=top_prob,
        win_probability=win_prob,
        move_rank=move_rank,
        is_top_choice=(played_move == top_move),
        all_probabilities=move_probs,
    )


def analyze_game_maia2(
    positions: list[dict],
    player_elo: int,
    opponent_elo: int,
    game_type: str = "rapid",
) -> list[Maia2Result]:
    """
    Analyze all positions in a game with Maia2.

    Args:
        positions: List of position dicts with 'fen' and 'move' keys
        player_elo: Player's rating
        opponent_elo: Opponent's rating
        game_type: "rapid" or "blitz"

    Returns:
        List of Maia2Result for each position
    """
    results = []

    for pos in positions:
        fen = pos.get('fen')
        move = pos.get('move')

        if not fen or not move:
            continue

        try:
            result = analyze_position_maia2(
                fen=fen,
                played_move=move,
                elo_self=player_elo,
                elo_oppo=opponent_elo,
                game_type=game_type,
            )
            results.append(result)
        except Exception as e:
            print(f"Error analyzing position: {e}")
            continue

    return results


def calculate_humanness_score(maia2_results: list[Maia2Result]) -> dict:
    """
    Calculate overall humanness metrics from Maia2 analysis.

    Args:
        maia2_results: List of Maia2Result from analyze_game_maia2

    Returns:
        Dictionary with humanness metrics:
        - avg_move_probability: Average probability assigned to played moves
        - top_choice_rate: Percentage of moves that were Maia2's top choice
        - avg_move_rank: Average rank of played moves in Maia2's predictions
        - log_likelihood: Sum of log probabilities (higher = more human-like)
        - humanness_score: Composite score (0-100, higher = more human-like)
    """
    import math

    if not maia2_results:
        return {
            'avg_move_probability': 0,
            'top_choice_rate': 0,
            'avg_move_rank': 0,
            'log_likelihood': 0,
            'humanness_score': 0,
            'num_positions': 0,
        }

    probabilities = [r.move_probability for r in maia2_results]
    ranks = [r.move_rank for r in maia2_results]
    top_choices = [r.is_top_choice for r in maia2_results]

    avg_prob = sum(probabilities) / len(probabilities)
    top_choice_rate = sum(top_choices) / len(top_choices)
    avg_rank = sum(ranks) / len(ranks)

    # Log likelihood (avoid log(0) by using small epsilon)
    epsilon = 1e-10
    log_likelihood = sum(math.log(max(p, epsilon)) for p in probabilities)
    avg_log_likelihood = log_likelihood / len(probabilities)

    # Composite humanness score (0-100)
    # Based on:
    # - Average move probability (higher = more human)
    # - Top choice rate (higher = more human)
    # - Average rank (lower = more human)
    #
    # A "typical" human at their rating should score around 50-70
    # Very high scores (>80) might indicate the player is playing
    # unusually predictably, while low scores (<30) suggest
    # moves that are atypical for their rating level

    # Normalize components
    prob_score = min(avg_prob * 200, 100)  # avg_prob of 0.5 = 100
    rank_score = max(0, 100 - (avg_rank - 1) * 15)  # rank 1 = 100, rank 7+ = 0
    top_score = top_choice_rate * 100

    # Weighted combination
    humanness_score = 0.4 * prob_score + 0.3 * rank_score + 0.3 * top_score

    return {
        'avg_move_probability': round(avg_prob, 4),
        'top_choice_rate': round(top_choice_rate, 4),
        'avg_move_rank': round(avg_rank, 2),
        'log_likelihood': round(log_likelihood, 2),
        'avg_log_likelihood': round(avg_log_likelihood, 4),
        'humanness_score': round(humanness_score, 1),
        'num_positions': len(maia2_results),
    }


def get_surprising_moves(
    maia2_results: list[Maia2Result],
    probability_threshold: float = 0.05,
    rank_threshold: int = 5,
) -> list[Maia2Result]:
    """
    Find moves that were surprising (low probability according to Maia2).

    These are moves that a human at the given rating would rarely play,
    which could indicate either:
    - Very strong play (found a non-obvious good move)
    - A blunder (atypical mistake)
    - Computer assistance (playing moves outside human patterns)

    Args:
        maia2_results: List of Maia2Result
        probability_threshold: Moves below this probability are surprising
        rank_threshold: Moves ranked worse than this are surprising

    Returns:
        List of surprising Maia2Result objects
    """
    surprising = []

    for result in maia2_results:
        if result.move_probability < probability_threshold or result.move_rank > rank_threshold:
            surprising.append(result)

    return surprising


def compare_to_stockfish(
    maia2_results: list[Maia2Result],
    stockfish_best_moves: list[str],
) -> dict:
    """
    Compare Maia2 predictions with Stockfish best moves.

    This helps identify if a player is:
    - Playing human-like AND strong (high Maia2 prob, matches Stockfish)
    - Playing human-like but weak (high Maia2 prob, doesn't match Stockfish)
    - Playing computer-like (low Maia2 prob, matches Stockfish) - SUSPICIOUS
    - Playing randomly (low Maia2 prob, doesn't match Stockfish)

    Args:
        maia2_results: List of Maia2Result
        stockfish_best_moves: List of best moves from Stockfish (UCI format)

    Returns:
        Dictionary with comparison metrics
    """
    if len(maia2_results) != len(stockfish_best_moves):
        raise ValueError("Maia2 results and Stockfish moves must have same length")

    human_and_strong = 0  # High Maia2 prob, matches Stockfish
    human_but_weak = 0    # High Maia2 prob, doesn't match Stockfish
    computer_like = 0     # Low Maia2 prob, matches Stockfish (SUSPICIOUS)
    random_like = 0       # Low Maia2 prob, doesn't match Stockfish

    prob_threshold = 0.1  # Above this = "human-like"

    computer_like_moves = []

    for result, sf_best in zip(maia2_results, stockfish_best_moves):
        is_human_like = result.move_probability >= prob_threshold
        matches_stockfish = result.played_move == sf_best

        if is_human_like and matches_stockfish:
            human_and_strong += 1
        elif is_human_like and not matches_stockfish:
            human_but_weak += 1
        elif not is_human_like and matches_stockfish:
            computer_like += 1
            computer_like_moves.append(result)
        else:
            random_like += 1

    total = len(maia2_results)

    return {
        'human_and_strong': human_and_strong,
        'human_and_strong_pct': round(human_and_strong / total * 100, 1) if total > 0 else 0,
        'human_but_weak': human_but_weak,
        'human_but_weak_pct': round(human_but_weak / total * 100, 1) if total > 0 else 0,
        'computer_like': computer_like,
        'computer_like_pct': round(computer_like / total * 100, 1) if total > 0 else 0,
        'random_like': random_like,
        'random_like_pct': round(random_like / total * 100, 1) if total > 0 else 0,
        'computer_like_moves': computer_like_moves,
        'total_moves': total,
    }
