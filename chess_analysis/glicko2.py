"""
Glicko-2 Rating System Implementation

Based on Professor Mark Glickman's paper:
"Example of the Glicko-2 system" (2013)
http://www.glicko.net/glicko/glicko2.pdf

This module provides:
- Glicko-2 rating calculation with rating, RD (deviation), and volatility
- Rating tracking over time (per-game, weekly, monthly periods)
- Rating improvement analysis
- Correlation with Ken Regan Z-scores
"""

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any
import pandas as pd
import numpy as np
from scipy import stats


# Constants from Glickman paper
GLICKO2_SCALE = 173.7178  # Conversion factor between Glicko and Glicko-2 scales
DEFAULT_RATING = 1500.0
DEFAULT_RD = 350.0
DEFAULT_VOLATILITY = 0.06
DEFAULT_TAU = 0.5  # System constant - constrains volatility change
CONVERGENCE_TOLERANCE = 0.000001


@dataclass
class Glicko2Rating:
    """
    Represents a Glicko-2 rating with three components.

    Attributes:
        rating: The player's skill estimate (μ on Glicko-2 scale = (rating-1500)/173.7178)
        rd: Rating Deviation - uncertainty in the rating (φ on Glicko-2 scale = rd/173.7178)
        volatility: σ - degree of expected fluctuation in rating (erratic vs consistent)
    """
    rating: float = DEFAULT_RATING
    rd: float = DEFAULT_RD
    volatility: float = DEFAULT_VOLATILITY

    def to_glicko2_scale(self) -> Tuple[float, float, float]:
        """Convert to Glicko-2 internal scale (μ, φ, σ)."""
        mu = (self.rating - 1500) / GLICKO2_SCALE
        phi = self.rd / GLICKO2_SCALE
        return mu, phi, self.volatility

    @classmethod
    def from_glicko2_scale(cls, mu: float, phi: float, sigma: float) -> 'Glicko2Rating':
        """Create from Glicko-2 internal scale values."""
        return cls(
            rating=mu * GLICKO2_SCALE + 1500,
            rd=phi * GLICKO2_SCALE,
            volatility=sigma
        )

    def confidence_interval(self, z: float = 1.96) -> Tuple[float, float]:
        """
        Calculate confidence interval for the rating.

        Args:
            z: Z-score for confidence level (1.96 = 95%, 2.58 = 99%)

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        margin = z * self.rd
        return (self.rating - margin, self.rating + margin)

    def __str__(self) -> str:
        ci_low, ci_high = self.confidence_interval()
        return f"Rating: {self.rating:.0f} ± {self.rd:.0f} (95% CI: {ci_low:.0f}-{ci_high:.0f}), σ={self.volatility:.4f}"


@dataclass
class Glicko2Result:
    """Result of Glicko-2 calculation for a rating period."""
    rating: Glicko2Rating
    games_processed: int
    period_start: datetime
    period_end: datetime
    wins: int = 0
    losses: int = 0
    draws: int = 0

    @property
    def win_rate(self) -> float:
        """Calculate win rate for the period."""
        total = self.wins + self.losses + self.draws
        if total == 0:
            return 0.0
        return self.wins / total


@dataclass
class RatingHistory:
    """Complete rating history for a player."""
    username: str
    results: List[Glicko2Result] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert rating history to a DataFrame for analysis."""
        if not self.results:
            return pd.DataFrame()

        records = []
        for r in self.results:
            records.append({
                'period_start': r.period_start,
                'period_end': r.period_end,
                'rating': r.rating.rating,
                'rd': r.rating.rd,
                'volatility': r.rating.volatility,
                'games_played': r.games_processed,
                'wins': r.wins,
                'losses': r.losses,
                'draws': r.draws,
                'win_rate': r.win_rate,
            })

        return pd.DataFrame(records)


def g(phi: float) -> float:
    """
    The g function from Glicko-2.
    Reduces the impact of an opponent's rating based on their uncertainty.

    g(φ) = 1 / √(1 + 3φ²/π²)
    """
    return 1.0 / math.sqrt(1.0 + 3.0 * phi * phi / (math.pi * math.pi))


def expected_score(mu: float, mu_j: float, phi_j: float) -> float:
    """
    Calculate expected score against an opponent.

    E(μ, μⱼ, φⱼ) = 1 / (1 + exp(-g(φⱼ)(μ - μⱼ)))
    """
    return 1.0 / (1.0 + math.exp(-g(phi_j) * (mu - mu_j)))


def compute_variance(mu: float, opponents: List[Tuple[float, float, float]]) -> float:
    """
    Compute the variance v (Step 3 of Glicko-2 algorithm).

    v = [Σ g(φⱼ)² × E × (1 - E)]⁻¹

    Args:
        mu: Player's rating on Glicko-2 scale
        opponents: List of (mu_j, phi_j, score) tuples

    Returns:
        Variance v
    """
    variance_sum = 0.0
    for mu_j, phi_j, _ in opponents:
        g_phi = g(phi_j)
        e = expected_score(mu, mu_j, phi_j)
        variance_sum += g_phi * g_phi * e * (1.0 - e)

    if variance_sum == 0:
        return float('inf')
    return 1.0 / variance_sum


def compute_delta(mu: float, opponents: List[Tuple[float, float, float]], v: float) -> float:
    """
    Compute the estimated improvement delta (Step 4 of Glicko-2 algorithm).

    Δ = v × Σ g(φⱼ) × (sⱼ - E)

    Args:
        mu: Player's rating on Glicko-2 scale
        opponents: List of (mu_j, phi_j, score) tuples
        v: Variance from compute_variance

    Returns:
        Delta improvement estimate
    """
    delta_sum = 0.0
    for mu_j, phi_j, score in opponents:
        g_phi = g(phi_j)
        e = expected_score(mu, mu_j, phi_j)
        delta_sum += g_phi * (score - e)

    return v * delta_sum


def compute_new_volatility(
    sigma: float,
    phi: float,
    v: float,
    delta: float,
    tau: float = DEFAULT_TAU
) -> float:
    """
    Compute new volatility using Illinois algorithm (Step 5 of Glicko-2).

    This finds σ' such that f(σ') = 0 using iterative method.
    """
    a = math.log(sigma * sigma)
    phi_sq = phi * phi

    def f(x: float) -> float:
        ex = math.exp(x)
        num = ex * (delta * delta - phi_sq - v - ex)
        denom = 2.0 * (phi_sq + v + ex) ** 2
        return num / denom - (x - a) / (tau * tau)

    # Find initial bounds
    A = a
    if delta * delta > phi_sq + v:
        B = math.log(delta * delta - phi_sq - v)
    else:
        k = 1
        while f(a - k * tau) < 0:
            k += 1
        B = a - k * tau

    # Illinois algorithm iteration
    f_A = f(A)
    f_B = f(B)

    while abs(B - A) > CONVERGENCE_TOLERANCE:
        C = A + (A - B) * f_A / (f_B - f_A)
        f_C = f(C)

        if f_C * f_B <= 0:
            A = B
            f_A = f_B
        else:
            f_A = f_A / 2.0

        B = C
        f_B = f_C

    return math.exp(A / 2.0)


def calculate_rating_period(
    current: Glicko2Rating,
    opponents: List[Tuple[float, float, float]],
    tau: float = DEFAULT_TAU
) -> Glicko2Rating:
    """
    Apply Glicko-2 algorithm for a single rating period.

    Implements the 8-step algorithm from Glickman's paper.

    Args:
        current: Current Glicko2Rating
        opponents: List of (opponent_rating, opponent_rd, score) tuples
                   Score is 1 for win, 0.5 for draw, 0 for loss
        tau: System constant (default 0.5)

    Returns:
        New Glicko2Rating after the rating period
    """
    if not opponents:
        # No games played - only RD increases (Step 6 special case)
        mu, phi, sigma = current.to_glicko2_scale()
        # RD increases over time when not playing
        # Using simplified version - increase by small factor
        phi_star = math.sqrt(phi * phi + sigma * sigma)
        return Glicko2Rating.from_glicko2_scale(mu, phi_star, sigma)

    # Step 1-2: Convert to Glicko-2 scale
    mu, phi, sigma = current.to_glicko2_scale()

    # Convert opponents to Glicko-2 scale
    opponents_g2 = []
    for opp_rating, opp_rd, score in opponents:
        mu_j = (opp_rating - 1500) / GLICKO2_SCALE
        phi_j = opp_rd / GLICKO2_SCALE
        opponents_g2.append((mu_j, phi_j, score))

    # Step 3: Compute variance v
    v = compute_variance(mu, opponents_g2)

    # Step 4: Compute delta
    delta = compute_delta(mu, opponents_g2, v)

    # Step 5: Compute new volatility
    sigma_new = compute_new_volatility(sigma, phi, v, delta, tau)

    # Step 6: Update RD (phi* then phi')
    phi_star = math.sqrt(phi * phi + sigma_new * sigma_new)
    phi_new = 1.0 / math.sqrt(1.0 / (phi_star * phi_star) + 1.0 / v)

    # Step 7: Update rating (mu')
    update_sum = 0.0
    for mu_j, phi_j, score in opponents_g2:
        e = expected_score(mu, mu_j, phi_j)
        update_sum += g(phi_j) * (score - e)

    mu_new = mu + phi_new * phi_new * update_sum

    # Step 8: Convert back to Glicko scale
    return Glicko2Rating.from_glicko2_scale(mu_new, phi_new, sigma_new)


def extract_game_result(
    game: dict,
    player_username: str
) -> Optional[Tuple[float, float, float, datetime]]:
    """
    Extract opponent rating and game result from a game dict.

    Args:
        game: Game dictionary from Chess.com API
        player_username: Username of the player we're tracking

    Returns:
        Tuple of (opponent_rating, opponent_rd, score, game_date) or None
    """
    username_lower = player_username.lower()

    # Determine if player is white or black
    white_player = game.get('white', {})
    black_player = game.get('black', {})

    white_username = white_player.get('username', '').lower()
    black_username = black_player.get('username', '').lower()

    if white_username == username_lower:
        player_color = 'white'
        opponent = black_player
    elif black_username == username_lower:
        player_color = 'black'
        opponent = white_player
    else:
        return None

    # Get opponent rating
    opponent_rating = opponent.get('rating', DEFAULT_RATING)
    # Chess.com doesn't provide RD, use default based on account status
    opponent_rd = DEFAULT_RD / 2  # Assume established players

    # Determine game result
    player_result = game.get(player_color, {}).get('result', '')

    if player_result == 'win':
        score = 1.0
    elif player_result in ('checkmated', 'timeout', 'resigned', 'lose', 'abandoned'):
        score = 0.0
    else:  # Draw variations
        score = 0.5

    # Get game timestamp
    end_time = game.get('end_time', 0)
    game_date = datetime.fromtimestamp(end_time) if end_time else datetime.now()

    return (opponent_rating, opponent_rd, score, game_date)


def calculate_glicko2(
    games: List[dict],
    player_username: str,
    initial_rating: Optional[Glicko2Rating] = None,
    tau: float = DEFAULT_TAU,
    period: str = "session",
    max_games_per_period: int = 50,
    session_gap_minutes: int = 15
) -> RatingHistory:
    """
    Calculate Glicko-2 ratings over time from game history.

    Args:
        games: List of game dicts from Chess.com API (should be sorted by date)
        player_username: Username of the player
        initial_rating: Starting rating (if None, uses player's first game Elo)
        tau: System constant (default 0.5)
        period: Rating period - "session", "monthly", "weekly", or "per_game"
        max_games_per_period: Maximum games per period before splitting (default 50)
        session_gap_minutes: Gap between games to define session boundary (default 15)

    Returns:
        RatingHistory with results for each rating period
    """
    history = RatingHistory(username=player_username)

    if not games:
        return history

    # Extract game data
    game_data = []
    for game in games:
        result = extract_game_result(game, player_username)
        if result:
            game_data.append(result)

    if not game_data:
        return history

    # Sort by date
    game_data.sort(key=lambda x: x[3])

    # If no initial rating provided, use player's Elo from first game
    if initial_rating is None:
        # Get player's rating from first game
        first_game = games[0] if games else None
        player_first_elo = None
        if first_game:
            username_lower = player_username.lower()
            white = first_game.get('white', {})
            black = first_game.get('black', {})
            if white.get('username', '').lower() == username_lower:
                player_first_elo = white.get('rating')
            elif black.get('username', '').lower() == username_lower:
                player_first_elo = black.get('rating')

        if player_first_elo:
            # Start with player's actual Elo, high RD (new to Glicko-2 tracking)
            initial_rating = Glicko2Rating(
                rating=player_first_elo,
                rd=DEFAULT_RD,  # High uncertainty initially
                volatility=DEFAULT_VOLATILITY
            )
        else:
            initial_rating = Glicko2Rating()

    # Group by period
    if period == "per_game":
        # Each game is its own period
        periods = [[g] for g in game_data]
    elif period == "session":
        # Group by session - games within session_gap_minutes of each other
        from datetime import timedelta
        periods = []
        current_session = []

        for game in game_data:
            if not current_session:
                current_session.append(game)
            else:
                # Check time gap from last game
                last_game_time = current_session[-1][3]  # date is index 3
                current_game_time = game[3]
                gap = (current_game_time - last_game_time).total_seconds() / 60

                if gap <= session_gap_minutes:
                    current_session.append(game)
                else:
                    # New session - save current and start new
                    if current_session:
                        periods.append(current_session)
                    current_session = [game]

        # Don't forget the last session
        if current_session:
            periods.append(current_session)
    else:
        # Group by month or week
        df = pd.DataFrame(game_data, columns=['opp_rating', 'opp_rd', 'score', 'date'])

        if period == "monthly":
            df['period'] = df['date'].dt.to_period('M')
        elif period == "weekly":
            df['period'] = df['date'].dt.to_period('W')
        else:
            raise ValueError(f"Unknown period: {period}. Use 'session', 'monthly', 'weekly', or 'per_game'")

        periods = []
        for _, group in df.groupby('period'):
            period_games = [
                (row['opp_rating'], row['opp_rd'], row['score'], row['date'])
                for _, row in group.iterrows()
            ]
            # Split large periods into smaller chunks to prevent algorithm instability
            # Glicko-2 is designed for rating periods with reasonable game counts
            if len(period_games) > max_games_per_period:
                for i in range(0, len(period_games), max_games_per_period):
                    chunk = period_games[i:i + max_games_per_period]
                    periods.append(chunk)
            else:
                periods.append(period_games)

    # Calculate ratings for each period
    current_rating = initial_rating

    for period_games in periods:
        if not period_games:
            continue

        # Extract opponent data for this period
        opponents = [(g[0], g[1], g[2]) for g in period_games]

        # Calculate new rating
        new_rating = calculate_rating_period(current_rating, opponents, tau)

        # Count results
        wins = sum(1 for g in period_games if g[2] == 1.0)
        losses = sum(1 for g in period_games if g[2] == 0.0)
        draws = len(period_games) - wins - losses

        # Create result
        result = Glicko2Result(
            rating=new_rating,
            games_processed=len(period_games),
            period_start=min(g[3] for g in period_games),
            period_end=max(g[3] for g in period_games),
            wins=wins,
            losses=losses,
            draws=draws
        )

        history.results.append(result)
        current_rating = new_rating

    return history


def track_rating_over_time(
    games: List[dict],
    player_username: str,
    period: str = "monthly"
) -> pd.DataFrame:
    """
    Track both Elo (from Chess.com) and calculated Glicko-2 over time.

    Args:
        games: List of game dicts from Chess.com API
        player_username: Username of the player
        period: Rating period - "monthly", "weekly", or "per_game"

    Returns:
        DataFrame with columns:
        - period_start, period_end
        - elo_start, elo_end, elo_change
        - glicko2_rating, glicko2_rd, glicko2_volatility
        - games_played, wins, losses, draws
    """
    # Calculate Glicko-2 history
    glicko_history = calculate_glicko2(games, player_username, period=period)
    glicko_df = glicko_history.to_dataframe()

    if glicko_df.empty:
        return pd.DataFrame()

    # Extract Chess.com Elo history
    username_lower = player_username.lower()
    elo_data = []

    for game in games:
        white = game.get('white', {})
        black = game.get('black', {})

        if white.get('username', '').lower() == username_lower:
            player_rating = white.get('rating', DEFAULT_RATING)
        elif black.get('username', '').lower() == username_lower:
            player_rating = black.get('rating', DEFAULT_RATING)
        else:
            continue

        end_time = game.get('end_time', 0)
        game_date = datetime.fromtimestamp(end_time) if end_time else None

        if game_date:
            elo_data.append({'date': game_date, 'elo': player_rating})

    if not elo_data:
        return glicko_df

    elo_df = pd.DataFrame(elo_data)

    # Group Elo by same periods
    if period == "monthly":
        elo_df['period'] = elo_df['date'].dt.to_period('M')
    elif period == "weekly":
        elo_df['period'] = elo_df['date'].dt.to_period('W')
    else:  # per_game - already aligned
        glicko_df['elo'] = [d['elo'] for d in elo_data[:len(glicko_df)]]
        return glicko_df

    # Aggregate Elo by period
    elo_agg = elo_df.groupby('period').agg(
        elo_start=('elo', 'first'),
        elo_end=('elo', 'last'),
        elo_min=('elo', 'min'),
        elo_max=('elo', 'max'),
    ).reset_index()
    elo_agg['elo_change'] = elo_agg['elo_end'] - elo_agg['elo_start']

    # Merge with Glicko-2 data
    glicko_df['period'] = glicko_df['period_start'].dt.to_period('M' if period == "monthly" else 'W')

    combined = glicko_df.merge(elo_agg, on='period', how='left')
    combined = combined.drop(columns=['period'])

    return combined


@dataclass
class ImprovementAnalysis:
    """Results of rating improvement analysis."""
    trend: str  # "improving", "stable", "declining"
    elo_slope: float  # Elo change per 100 games
    glicko2_slope: float  # Glicko-2 change per 100 games
    r_squared: float  # How well the trend fits
    volatility_trend: str  # "increasing", "stable", "decreasing"
    avg_volatility: float
    rapid_improvement_periods: List[Tuple[datetime, datetime]]  # Periods with unusually fast improvement


def analyze_rating_improvement(
    rating_history: pd.DataFrame,
    min_games: int = 50
) -> Optional[ImprovementAnalysis]:
    """
    Analyze if player is improving over time.

    Args:
        rating_history: DataFrame from track_rating_over_time
        min_games: Minimum games required for analysis

    Returns:
        ImprovementAnalysis or None if insufficient data
    """
    if rating_history.empty or rating_history['games_played'].sum() < min_games:
        return None

    # Calculate cumulative game count for x-axis
    rating_history = rating_history.copy()
    rating_history['cumulative_games'] = rating_history['games_played'].cumsum()

    # Linear regression on Glicko-2 rating
    x = rating_history['cumulative_games'].values
    y = rating_history['rating'].values

    if len(x) < 3:
        return None

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    r_squared = r_value ** 2

    # Slope per 100 games
    glicko2_slope = slope * 100

    # Elo slope if available
    if 'elo_end' in rating_history.columns:
        elo_y = rating_history['elo_end'].dropna().values
        if len(elo_y) >= 3:
            elo_slope, _, _, _, _ = stats.linregress(x[:len(elo_y)], elo_y)
            elo_slope = elo_slope * 100
        else:
            elo_slope = 0.0
    else:
        elo_slope = 0.0

    # Determine trend
    if glicko2_slope > 10:
        trend = "improving"
    elif glicko2_slope < -10:
        trend = "declining"
    else:
        trend = "stable"

    # Analyze volatility trend
    vol = rating_history['volatility'].values
    if len(vol) >= 3:
        vol_slope, _, _, _, _ = stats.linregress(x, vol)
        if vol_slope > 0.001:
            volatility_trend = "increasing"
        elif vol_slope < -0.001:
            volatility_trend = "decreasing"
        else:
            volatility_trend = "stable"
    else:
        volatility_trend = "stable"

    avg_volatility = rating_history['volatility'].mean()

    # Find rapid improvement periods
    rapid_periods = []
    if len(rating_history) >= 3:
        # Calculate per-period improvement rate
        rating_history['rating_change'] = rating_history['rating'].diff()
        rating_history['improvement_rate'] = rating_history['rating_change'] / rating_history['games_played']

        # Flag periods with improvement > 2 std dev above mean
        mean_rate = rating_history['improvement_rate'].mean()
        std_rate = rating_history['improvement_rate'].std()

        if std_rate > 0:
            threshold = mean_rate + 2 * std_rate
            rapid = rating_history[rating_history['improvement_rate'] > threshold]

            for _, row in rapid.iterrows():
                rapid_periods.append((row['period_start'], row['period_end']))

    return ImprovementAnalysis(
        trend=trend,
        elo_slope=elo_slope,
        glicko2_slope=glicko2_slope,
        r_squared=r_squared,
        volatility_trend=volatility_trend,
        avg_volatility=avg_volatility,
        rapid_improvement_periods=rapid_periods
    )


def correlate_with_regan_zscore(
    rating_history: pd.DataFrame,
    regan_results: List[Any],  # List[ReganAnalysisResult]
    games_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Correlate rating improvement periods with Ken Regan Z-scores.

    Hypothesis: Rapidly improving players may show higher Z-scores.

    Args:
        rating_history: DataFrame from track_rating_over_time
        regan_results: List of ReganAnalysisResult from regan_analysis module
        games_df: DataFrame with game data including dates

    Returns:
        Dict with correlation coefficient and flagged periods
    """
    if rating_history.empty or not regan_results:
        return {
            'correlation': None,
            'p_value': None,
            'flagged_periods': [],
            'message': 'Insufficient data for correlation analysis'
        }

    # Calculate improvement rate per period
    rating_history = rating_history.copy()
    rating_history['rating_change'] = rating_history['rating'].diff().fillna(0)
    rating_history['improvement_rate'] = rating_history['rating_change'] / rating_history['games_played'].replace(0, 1)

    # Extract Z-scores from Regan results
    zscores = []
    for result in regan_results:
        if hasattr(result, 'z_score') and result.z_score is not None:
            zscores.append(result.z_score)

    if len(zscores) < 5:
        return {
            'correlation': None,
            'p_value': None,
            'flagged_periods': [],
            'message': f'Only {len(zscores)} Z-scores available, need at least 5'
        }

    # Match Z-scores to rating periods (simplified - uses overall correlation)
    improvement_rates = rating_history['improvement_rate'].dropna().values

    # Ensure same length for correlation
    min_len = min(len(improvement_rates), len(zscores))
    if min_len < 5:
        return {
            'correlation': None,
            'p_value': None,
            'flagged_periods': [],
            'message': 'Insufficient matching data points'
        }

    improvement_rates = improvement_rates[:min_len]
    zscores = zscores[:min_len]

    # Calculate correlation
    corr, p_value = stats.pearsonr(improvement_rates, zscores)

    # Flag periods where both improvement rate AND Z-score are high
    flagged = []
    improvement_threshold = np.percentile(improvement_rates, 75)
    zscore_threshold = 1.5  # Moderately high Z-score

    for i, (imp_rate, zscore) in enumerate(zip(improvement_rates, zscores)):
        if imp_rate > improvement_threshold and zscore > zscore_threshold:
            if i < len(rating_history):
                row = rating_history.iloc[i]
                flagged.append({
                    'period_start': row.get('period_start'),
                    'period_end': row.get('period_end'),
                    'improvement_rate': imp_rate,
                    'z_score': zscore
                })

    return {
        'correlation': corr,
        'p_value': p_value,
        'flagged_periods': flagged,
        'interpretation': _interpret_correlation(corr, p_value),
        'message': None
    }


def _interpret_correlation(corr: float, p_value: float) -> str:
    """Provide interpretation of correlation results."""
    if p_value > 0.05:
        return "No significant correlation between rating improvement and Z-scores"

    strength = "weak" if abs(corr) < 0.3 else "moderate" if abs(corr) < 0.6 else "strong"
    direction = "positive" if corr > 0 else "negative"

    if corr > 0.3:
        return (f"A {strength} {direction} correlation (r={corr:.2f}, p={p_value:.3f}) suggests "
                "periods of rapid improvement coincide with higher-than-expected engine match rates")
    elif corr < -0.3:
        return (f"A {strength} {direction} correlation (r={corr:.2f}, p={p_value:.3f}) suggests "
                "improving players actually have lower engine match rates (expected for learning)")
    else:
        return f"Weak correlation (r={corr:.2f}) - no clear relationship between improvement and Z-scores"


# Export all public functions and classes
__all__ = [
    'Glicko2Rating',
    'Glicko2Result',
    'RatingHistory',
    'ImprovementAnalysis',
    'calculate_rating_period',
    'calculate_glicko2',
    'track_rating_over_time',
    'analyze_rating_improvement',
    'correlate_with_regan_zscore',
    'DEFAULT_RATING',
    'DEFAULT_RD',
    'DEFAULT_VOLATILITY',
    'DEFAULT_TAU',
]
