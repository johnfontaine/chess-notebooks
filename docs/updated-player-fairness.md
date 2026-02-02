# Updated Player Fairness Report

## General Requirements

- Update so that all sections are formatted and use notebook style tables where tables are displayed.
- Add Definitions at the start of each section/analysis to explain calculated columns, rows and outputs.
- Use `pd.DataFrame.style` for styled tables with consistent column naming.
- Add summary rows where appropriate.

---

## 1. Setup and Configuration

- Environment setup, imports, configuration variables
- Define `USERNAME`, `DATA_DIR`, analysis parameters
- Load Stockfish engine path

---

## 2. Data Collection

- Fetch games via `generate_player_baseline()`
- Load parquet/JSON datasets
- Load raw games for PGN access

---

## 3. Quick Analysis

### 3.1 Session Analysis

**Purpose**: Understand how the player plays - in sessions or individual games.

**Definitions**:

| Term | Definition |
|------|------------|
| Session | A group of consecutive games where gaps between games are <15 minutes |
| Session Performance Rating | Elo calculated based on session results only |
| Session Win Rate | Wins / Total games in session |

**Metrics**:

- Games per session distribution
- W/L/D per session
- Sessions vs Individual Game Play (what % of the games are in sessions)
- Session Performance Rating
- Session win rate consistency (std dev across sessions)

**Flags**:

- Sessions with abnormally high / low performance ratings
- Sessions with abnormally high win rates
- Sessions wtih abnormally low win rates

---

### 3.2 Rating Analysis (formerly Elo Analysis)

**Purpose**: Track player rating over time using multiple rating systems to identify improvement patterns and consistency.

**Definitions**:

| Term | Definition |
|------|------------|
| Elo | Standard rating from Chess.com |
| Glicko | Rating system with Rating + RD (Rating Deviation) |
| Glicko-2 | Rating system with Rating + RD + Volatility (σ) |
| RD (Rating Deviation) | Uncertainty in rating; high RD = unreliable rating |
| Volatility (σ) | Measures how erratic a player's performance is |
| Upset | A win or loss where Elo difference was >200 |
| Rating Stability | Standard deviation of rating over time period |

**Rating Systems Tracked**:

- **Elo**: Standard rating from Chess.com (tracked over time)
- **Glicko-2**: Calculated rating with RD (uncertainty) and σ (volatility)

**Lifetime Metrics**:

- Rating range (floor/ceiling)
- Rating stability (std dev)
- Upsets count (wins/losses with >200 rating diff)
- Glicko-2 RD trend (is uncertainty decreasing over time?)
- Glicko-2 volatility (σ) pattern (consistent or erratic player?)

**Monthly Metrics**: Same as lifetime, aggregated by month

**Session Metrics**: Same as lifetime, aggregated by session

**Rating Improvement Analysis**:

- Rating trajectory over time (improving/stable/declining)
- Rate of improvement (Elo gain per month/100 games)
- Correlation with Ken Regan Z-score (do improving players have higher Z-scores?)

**Segment Outliers**:

- Did any segments show a high performance variance from the monthly statistics?

---

### 3.3 Result Patterns

**Definitions**:

| Term | Definition |
|------|------------|
| Win Streak | Consecutive wins |
| Loss Streak | Consecutive losses |
| Termination | How the game ended (checkmate, timeout, resignation, etc.) |

**Metrics**:

- Win/Loss/Draw rates
- Streak analysis (win/loss streaks)
- Termination patterns (checkmate, timeout, resignation rates)

---

### 3.4 Baseline Comparison

**Purpose**: Compare player metrics against known baselines.

**Definitions**:

| Term | Definition |
|------|------------|
| Trusted Baseline | Statistics from verified fair players |
| Cheater Baseline | Statistics from known cheaters for comparison |
| Z-score | Standard deviations from the mean (>2 is unusual, >3 is highly unusual) |

**Metrics**:

- Compare player metrics to trusted baseline
- Compare to known cheater baseline
- Z-score calculations for key metrics

---

## 4. Game Prioritization

**Purpose**: Score and rank games for deep analysis based on suspicion indicators.

**Definitions**:

| Term | Definition |
|------|------------|
| Suspicion Score | Composite score based on multiple indicators |

**Metrics**:

- Suspicion scoring algorithm
- Ranked game list for deep analysis
- Filters for specific time controls or opponent types

---

## 5. Deep Analysis

### 5.1 Multi-Depth Engine Analysis

**Definitions**:

| Term | Definition |
|------|------------|
| CPL (Centipawn Loss) | Difference between best move evaluation and played move evaluation |
| ACPL (Average CPL) | Mean CPL across all moves |
| Move Accuracy | Move quality (0-100%) based on win% preservation using Lichess formula |
| Win Percentage | Expected win probability from eval: 50 + 50 × (2/(1+exp(-0.00368208×cp)) - 1) |
| Mate Evaluation | Mate-in-N converted to centipawns: ±(10000 - N×10). Mate in 1 = ±9990 cp |
| Best | Move matches engine's top choice (0 CPL) |
| Excellent | Move loses 1-9 centipawns |
| Good | Move loses 10-24 centipawns |
| Inaccuracy | Move loses 25-49 centipawns |
| Mistake | Move loses 50-299 centipawns, position still favorable or equal |
| Blunder | Move loses ≥300 centipawns, position still favorable or equal |
| Critical Mistake | Move loses 50-299 CPL AND shifts position from favorable (>100cp) to unfavorable (<-100cp) |
| Critical Blunder | Move loses ≥300 CPL AND shifts position favorable to unfavorable, or allows forced mate |
| Missed Win | Had winning position (>300cp or mate) but now equal or worse |
| Move Rank | Position of played move among legal moves sorted by eval (1 = best move) |
| Gap to 2nd Best | Eval difference between best and second-best move (via MultiPV) |
| Playable Moves | Count of moves within 50cp of best move |
| Eval Volatility | Std dev of evals across depths [6, 10, 14, 20] - higher = unstable position |
| SF Branching Factor | Engine branching from node counts: (nodes_d2/nodes_d1)^(1/(d2-d1)). Typical: 1.5-4 |
| Raw Branching Factor | Brute-force branching counting legal moves at 3-ply depth. Typical: 20-25 opening, 5-10 endgame |
| Complexity Score | Composite (0-100%): 30% volatility + 25% gap + 20% convergence + 25% branching |
| Complexity Category | LOW (<25%), MEDIUM (25-50%), HIGH (50-75%), VERY_HIGH (>75%) |

**Metrics**:
- Stockfish analysis at configurable depth (default: 20)
- Multi-depth analysis at [6, 10, 14, 20] for complexity heuristics
- Move-by-move centipawn loss and accuracy
- Move classification distribution
- ACPL by game phase (opening/middlegame/endgame)
- Position complexity scoring
- Branching factor analysis (SF and Raw)

---

### 5.2 Ken Regan Analysis

**Definitions**:

| Term | Definition |
|------|------------|
| IPR (Ideal Performance Rating) | Rating needed to achieve observed move match rate |
| Z-score | Standard deviations from expected performance |
| Expected Move Match Rate | Probability of matching engine move based on Elo |
| Sensitivity | How well player distinguishes between similar moves (lower = stronger) |
| Consistency | How rarely player makes large errors (higher = stronger) |
| Partial Credit | Credit assigned to a move based on drop-off from best (0-1) |
| Position Difficulty | Combined score from legal moves, fragility, and evaluation (0-1) |
| Suspicious Position | Position where player found a difficult move at a rate exceeding expected for their Elo |

**Metrics**:
- FIDE methodology implementation
- IPR calculation
- Z-score for deviation from expected performance
- Correlation with rating improvement (from Rating Analysis)
- **Suspicious position identification** (positions where player outperformed expectations)
- **Best moves in complex positions** (engine-matching moves in difficult situations)

---

### 5.3 Tablebase Accuracy

**Definitions**:
| Term | Definition |
|------|------------|
| Tablebase | Database of perfect endgame play for positions with <=7 pieces |
| DTZ (Distance to Zeroing) | Moves to capture or pawn move (50-move rule) |
| Perfect Play | Move that does not change the theoretical result |

**Metrics**:
- Endgame accuracy analysis
- Tablebase-optimal move percentage
- Conversion efficiency in winning positions

---

## 6. Analysis Reports

### 6.1 Accuracy Report

**Metrics**:
- ACPL statistics (mean, median, std dev)
- Move classification distribution
- Error analysis by position advantage
- Comparison to baseline

---

### 6.2 Time Usage Report

**Definitions**:
| Term | Definition |
|------|------------|
| Instant | <1 second |
| Quick | 1-3 seconds |
| Short | 3-10 seconds |
| Normal | 10-30 seconds |
| Long | 30-60 seconds |
| Very Long | >60 seconds |

**Metrics**:
- Time per move distribution
- Bot pattern detection (too-consistent timing)
- Time classification breakdown
- Time vs complexity correlation

---

### 6.3 Opponent Segment Analysis

**Definitions**:
| Term | Definition |
|------|------------|
| Player Favored | Player Elo > Opponent Elo by 100+ |
| Fair Match | Elo difference within 100 |
| Opponent Favored | Opponent Elo > Player Elo by 100+ |
| Expected Win Rate | Calculated from Elo difference |

**Metrics**:
- Performance by opponent Elo tier
- Expected vs actual win rates
- Upset frequency by segment

---

### 6.4 Banned Opponent Analysis

**Purpose**: Examine games played against opponents who have since been banned for fair play violations. This can reveal engine-vs-engine patterns or suspicious collaboration.

**Definitions**:

| Term | Definition |
|------|------------|
| Banned Opponent | Player whose account was closed for fair play violations |
| Trustworthy Game | Game against opponent with good standing account |
| Untrustworthy Game | Game against banned opponent |
| Engine vs Engine | Pattern where both players show unusually high accuracy |

**Metrics**:

- Count and percentage of games vs banned opponents
- Win/Loss/Draw rate vs banned opponents compared to overall
- ACPL comparison: player's ACPL in banned vs trustworthy games
- Accuracy distribution in banned opponent games
- Suspicious patterns (both players with low ACPL)
- Timeline: when were these games played relative to opponent's ban

**Flags**:

- Unusually high win rate vs banned opponents
- Both players showing engine-level accuracy
- Concentrated games against same banned opponent

---

### 6.5 Tablebase Consistency Report

**Metrics**:
- Cross-game endgame accuracy
- Consistency of tablebase accuracy across games
- Flagged games with unusually high/low accuracy

---

### 6.6 Key Positions Visualization

**Definitions**:

| Term | Definition |
|------|------------|
| Fragility | Graph-theoretic position vulnerability (betweenness centrality of attacked pieces) |
| Fragility Peak | Position where fragility reaches a local maximum - decisive moments |
| Fragility Trend | Direction of fragility change: ↑ increasing, ↓ decreasing, → stable |
| Distance to Peak | Plys before (-) or after (+) the game's maximum fragility point |
| Pre-Peak Position | Position before max fragility - finding best moves here suggests strong calculation |
| Complex Position | High complexity score (>50%) combining volatility, gap, convergence, branching |
| Position Difficulty | Combined score from legal moves, fragility, and evaluation (0-1) |
| Suspicious Position | Position where player found difficult move at rate exceeding expected for their Elo |
| Suspicion Score | How much player outperformed expectation in difficult position (0-1) |
| SF Branching | Engine search tree expansion factor (typical 1.5-4) |
| Raw Branching | True legal move tree branching at 3-ply depth (typical 20-25 opening) |
| Eval Volatility | Std dev of evals across depths - unstable positions have high volatility |

**Metrics**:
- Fragility peaks (positions where small errors have large consequences)
- Complex positions (high complexity score from engine heuristics)
- Mistakes and blunders (with CPL and accuracy)
- Brilliant moves
- **Suspicious positions** (from Ken Regan analysis - unexpectedly good play in difficult spots)
- **Best moves in complex positions** (positions where player found engine's top choice despite high difficulty)
- Position complexity breakdown (volatility, gap, branching factors)

---

### 6.7 Final Summary

**Metrics**:
- Aggregate fairness assessment
- Key indicators summary
- Recommended actions (if any)
