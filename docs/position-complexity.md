# Computing Position Complexity and Human Cognatigve Load

The goal is to be able to assign some scores to a position that can be used to determine how "complex" a position is for a human.  By complexity I mean the relative cognative load for a human.  We are looking for a z-score as well as a categorization.  Where the load is LOW, MEDIUM, HIGH, EXTREMELY HIGH, IMPOSSIBLE.

## Techniques and Underlying Metrics

1. Engine Based Heuristics -- these are gatehrs by observing engine behavior at is evaluates a position
   - Evaluation Volatility: If the engine's evaluation ($E$) swings wildly as depth increases, the position is likely complex.
   - The "Gap" Metric ($\Delta$): Measure the difference between the best move and the second-best move. A small $\Delta$ (e.g., $< 100$ centipawns) suggests a "high-entropy" position where many moves are viable but only one might be correct, increasing cognitive load.
   - Search Depth Velocity: Track how quickly the engine reaches higher depths. If the "Nodes Per Second" stays high but the depth increases slowly, the engine is struggling with a massive branching factor—a hallmark of complexity.
   - Node Count per Ply: A higher number of nodes searched to reach a specific depth indicates a more "tactically dense" position.

2. Graph Theory & Network Analysis -- model the chess position as a inteaction graph where peices are nodes and attacks/defenses are eges.
   - Fragility Score: Calculated by summing the betweenness centrality (BC) of all attacked pieces. A high score indicates a "fragile" position where a single exchange could trigger a massive cascade of changes (this is our existing fragility score) [More research here](../references/Chess%20and%20Complexity%20Science_%20Understanding%20the%20Fragility%20of%20Positions%20-%20Chess.com.pdf)
   - Strategic Tension: This metric quantifies the [piece-to-piece interaction](../references/2508.13213v1.pdf). Research shows that AI can sustain much higher tension than humans, who tend to "resolve" tension to lower their cognitive load.
3. Machine Learning & Behavioral Models
   - *Elocator Model*: [An open-source project](https://github.com/cmwetherell/elocator) that uses a model trained on GM games to predict [human lose-rate](../references/Analyzing%20The%20Complexity%20of%20a%20Chess%20Position%20_%20Pawnalyze.pdf) based on position complexity.
   - [GlickFormer](../references/2410.11078v1.pdf): A transformer-based architecture that predicts the difficulty of a position by approximating the Glicko-2 rating required to solve it.
   - Maia2 Complexity predictions as decsribed in [this paper](../references/Estimating_the_Difficulty_of_Chess_Puzzles_by_Comb.pdf)
4. Additional Maia2 Items
   - High Entropy: If Maia-2 assigns similar probabilities to 5 or 6 different moves, the position is "high entropy." This correlates with a higher cognitive load because there is no clear, intuitive "human" move.
   - Low Perplexity: If Maia-2 is very certain about a move (e.g., 85% probability for one move), the position is likely a "one-move" puzzle or a simple tactical recapture with low cognitive load.
   - *The "Trap" Metric*: If Maia-2’s #1 predicted move is an engine blunder ($-3.0$), the position has a very high "relative cognitive load" because the most intuitive human response is actually a mistake.
   - *Skill-Aware Attention*: Maia-2 uses a "skill-aware attention mechanism." This means you can query the engine for different ratings (e.g., 1000 vs. 2000) for the same position. If the 2000-rated model finds the correct move with 60% probability but the 1000-rated model only finds it with 10%, you have a quantified measure of how much "skill" is required to resolve the position's complexity.
   - *Maia bludner prediction*: [discussed in the project's github repository](https://github.com/CSSLab/maia-chess)

## Summary Table: Complexity Metrics

| Category | Metric | Description | Complexity Signal |
|----------|--------|-------------|-------------------|
| **Engine Heuristics** | Evaluation Volatility | Swing in engine eval ($E$) as depth increases | High volatility → High complexity |
| | Gap Metric ($\Delta$) | Difference between best and second-best move | Small $\Delta$ (< 100cp) → High complexity |
| | Search Depth Velocity | Rate of depth increase vs NPS | Slow depth increase → High branching factor |
| | Node Count per Ply | Nodes searched to reach specific depth | More nodes → Tactically dense |
| **Graph Theory** | Fragility Score | Sum of betweenness centrality of attacked pieces | High score → Fragile, cascade-prone position |
| | Strategic Tension | Piece-to-piece interaction intensity | High tension → High cognitive load (humans resolve tension) |
| **ML Models** | Elocator | Predicts human lose-rate from GM training data | Higher lose-rate → More complex |
| | GlickFormer | Predicts Glicko-2 rating required to solve position | Higher rating → More difficult |
| | Maia2 Complexity | Difficulty predictions from transformer model | Direct complexity estimate |
| **Maia2 Entropy** | High Entropy | Similar probabilities across 5-6 moves | No clear intuitive move → High load |
| | Low Perplexity | High certainty on single move (e.g., 85%) | Simple tactical/recapture → Low load |
| | Trap Metric | Maia's top move is engine blunder ($\leq -3.0$) | Intuitive move is wrong → Very high load |
| | Skill-Aware Attention | Probability gap between rating levels (e.g., 1000 vs 2000) | Large gap → Skill-dependent complexity |
| | Blunder Prediction | Maia blunder probability estimate | Higher probability → Error-prone position |

## Implementation Details

### Engine Heuristics Implementation

The following engine-based heuristics are implemented in `chess_analysis/engine.py`:

#### 1. Evaluation Volatility

- **Calculation**: Standard deviation of centipawn evaluations across depths [5, 10, 15, 20]
- **Normalization**: Raw std dev divided by 200 (capped at 1.0)
- **Implementation**: `calculate_eval_volatility(evaluations: dict[int, int]) -> tuple[float, float]`
- **Interpretation**:
  - < 30cp: Stable position (LOW complexity signal)
  - 30-80cp: Moderate instability (MEDIUM)
  - 80-150cp: High instability (HIGH)
  - > 150cp: Very unstable (VERY_HIGH)

#### 2. Gap Metric (Delta)
- **Calculation**: Difference between MultiPV 1 and MultiPV 2 evaluations at max depth
- **Requirements**: Stockfish configured with MultiPV=2
- **Implementation**: `GapMetricResult` dataclass populated during `analyze_multi_depth_extended()`
- **Interpretation**:
  - > 150cp: Clear best move (LOW complexity signal)
  - 100-150cp: Fairly clear (MEDIUM)
  - 50-100cp: Multiple options (HIGH)
  - < 50cp: Many viable moves (VERY_HIGH)

#### 3. Search Depth Velocity / Convergence
- **Calculation**: First depth where best move matches the final best move
- **Implementation**: `first_consistent_depth` field in `MultiDepthResult`
- **Normalization**: `(convergence_depth - 3) / 18` mapped to 0-1
- **Interpretation**: Later convergence indicates a more complex position

#### 4. SF Branching Factor (Engine-derived)
- **Calculation**: `(nodes_d2 / nodes_d1)^(1/(d2-d1))` averaged across depth pairs
- **Implementation**: `estimate_branching_factor(search_metrics) -> float`
- **Typical values**: 1.5-4 for chess positions (average ~2-3)
- **Normalization**: `(branching_factor - 1.5) / 2.5` mapped to 0-1
- **Interpretation**: Measures engine search tree expansion at each level

#### 5. Raw Branching Factor (Brute-Force)
- **Calculation**: Count all legal positions at 3-ply depth (your moves × opponent responses × your replies)
- **Implementation**: `calculate_brute_force_branching(board, depth=3) -> dict`
- **Returns**: `{'branching_factor': float, 'nodes_by_depth': list, 'total_nodes': int, 'initial_legal_moves': int}`
- **Typical values**:
  - Opening positions: ~20-25
  - Middlegame: ~15-20
  - Simple endgames: ~5-10
- **Interpretation**: True positional branching - how many continuations exist

### Composite Complexity Score

The final `complexity_score` (0-1) is calculated in `calculate_complexity_heuristics()` as:

```
complexity = (
    0.30 * eval_volatility_normalized +
    0.25 * gap_normalized +           # Inverted: small gap = complex
    0.20 * convergence_normalized +   # Later convergence = complex
    0.25 * branching_factor_normalized
)
```

### Categories

The `categorize_complexity(score: float) -> str` function assigns:
- **LOW**: complexity < 0.25
- **MEDIUM**: 0.25 <= complexity < 0.50
- **HIGH**: 0.50 <= complexity < 0.75
- **VERY_HIGH**: complexity >= 0.75

### Key Classes and Functions

| Item | Location | Description |
|------|----------|-------------|
| `EngineSearchMetrics` | `engine.py` | Dataclass for nodes, nps, time, seldepth per depth |
| `GapMetricResult` | `engine.py` | Dataclass for best/second-best move gap |
| `PositionComplexityHeuristics` | `engine.py` | Aggregate dataclass with all heuristics |
| `analyze_multi_depth_extended()` | `engine.py` | Extended analysis method with MultiPV |
| `calculate_eval_volatility()` | `engine.py` | Std dev calculation helper |
| `estimate_branching_factor()` | `engine.py` | SF branching factor from node counts |
| `calculate_brute_force_branching()` | `position_assessment.py` | Raw branching factor from legal move tree |
| `calculate_complexity_heuristics()` | `engine.py` | Aggregate all metrics into score |
| `GapMetricCache` | `engine_cache.py` | Parquet-based cache for gap metrics |

## Position Complexity Classifier Proposal

### Objective

Train a classifier to categorize positions as **LOW**, **MEDIUM**, **HIGH**, **EXTREME**, or **IMPOSSIBLE** based on human cognitive load. The classifier combines engine heuristics with human performance data.

### Training Data Source

**Lichess Puzzle Database** (https://database.lichess.org/#puzzles)
- 3+ million rated puzzles with difficulty ratings
- Puzzle rating serves as proxy for human difficulty
- Ratings range from ~400 to 3000+
- Each puzzle has a FEN and solution moves

### Rating-to-Category Mapping

| Puzzle Rating | Category | Interpretation |
|---------------|----------|----------------|
| < 1200 | LOW | Most players can solve |
| 1200-1600 | MEDIUM | Intermediate difficulty |
| 1600-2000 | HIGH | Requires strong calculation |
| 2000-2400 | EXTREME | Expert-level difficulty |
| > 2400 | IMPOSSIBLE | Near-impossible for humans |

### Input Features

| Feature | Type | Source | Description |
|---------|------|--------|-------------|
| `eval_volatility` | float | Engine multi-depth | Std dev of eval across depths |
| `eval_volatility_normalized` | float | Engine | Normalized 0-1 |
| `gap_cp` | int | Engine MultiPV | Gap to second-best move |
| `branching_factor` | float | Engine node counts | Estimated branching factor |
| `convergence_depth` | int | Engine | Depth where move stabilized |
| `num_legal_moves` | int | Board state | Total legal moves |
| `fragility_score` | float | Graph theory | Betweenness centrality sum |
| `material_imbalance` | int | Board state | Material difference (cp) |
| `game_phase` | enum | Board state | opening/middlegame/endgame |
| `pieces_remaining` | int | Board state | Total pieces on board |
| `maia2_entropy` | float | Maia2 | Move distribution entropy |
| `maia2_top_prob` | float | Maia2 | Probability of top move |

### Model Architecture Options

#### Option A: Gradient Boosting (Recommended for interpretability)
- **Model**: XGBoost or LightGBM
- **Advantages**:
  - Feature importance rankings
  - Fast inference
  - Handles missing values
  - Works well with mixed feature types
- **Hyperparameters**: 100-500 trees, max_depth 4-8, learning_rate 0.05-0.1

#### Option B: Neural Network (Higher accuracy potential)
- **Architecture**:
  - Input: Feature vector (12-15 features)
  - Hidden layers: 2-3 layers, 64-128 units each
  - Output: 5-class softmax
- **Training**: Cross-entropy loss, Adam optimizer
- **Regularization**: Dropout 0.2-0.3, batch normalization

#### Option C: Ensemble
- Combine XGBoost for structured features with a small CNN for board state
- Board CNN processes 8x8x12 piece placement tensor
- Concatenate embeddings before final classification

### Training Pipeline

```python
# Pseudocode for training pipeline

1. Download and parse Lichess puzzle database
   puzzles = load_lichess_puzzles("lichess_db_puzzle.csv")

2. Extract FEN positions and compute features
   for puzzle in puzzles:
       board = chess.Board(puzzle.fen)
       features = {
           # Engine features (batch process for efficiency)
           'eval_volatility': compute_eval_volatility(board, engine),
           'gap_cp': compute_gap_metric(board, engine),
           'branching_factor': estimate_branching_factor(board, engine),
           # Board features
           'num_legal_moves': len(list(board.legal_moves)),
           'fragility_score': calculate_fragility_simple(board),
           'game_phase': detect_game_phase(board).value,
           # ...
       }
       features['label'] = rating_to_category(puzzle.rating)

3. Train/test split (stratified by category)
   X_train, X_test, y_train, y_test = train_test_split(
       features, labels,
       test_size=0.2,
       stratify=labels
   )

4. Train classifier
   model = XGBClassifier(
       n_estimators=200,
       max_depth=6,
       learning_rate=0.1,
       objective='multi:softmax',
       num_class=5,
   )
   model.fit(X_train, y_train)

5. Evaluate
   y_pred = model.predict(X_test)
   print(classification_report(y_test, y_pred))
```

### Validation Strategy

1. **Stratified K-Fold Cross-Validation**
   - 5-fold CV stratified by category
   - Ensures representation of all difficulty levels in each fold

2. **Correlation with External Metrics**
   - Compare predictions vs Maia2 entropy scores
   - Compare vs Elocator human error rate predictions
   - Compare vs GlickFormer difficulty estimates

3. **Human Performance Validation**
   - Sample positions from each predicted category
   - Measure actual solve rate by rating tier
   - Higher-rated players should solve more HIGH/EXTREME positions

4. **Calibration Analysis**
   - Plot predicted probability vs actual accuracy
   - Ensure model confidence aligns with true difficulty

### Evaluation Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Accuracy | Overall correct classifications | > 70% |
| Macro F1 | Average F1 across categories | > 0.65 |
| Confusion Matrix | Category-wise errors | Low off-diagonal |
| Top-2 Accuracy | Correct within 1 category | > 90% |
| Correlation with Rating | Spearman rho with puzzle rating | > 0.7 |

### Integration Points

Once trained, the classifier can be used in:

1. **Phase 4a Engine Analysis**
   - Assign complexity category to each position
   - Store `complexity_category` field in position data

2. **Phase 7 Risk Assessment**
   - Weight metrics by position complexity
   - Strong moves in EXTREME positions are more concerning than in LOW

3. **Phase 8 Final Report**
   - Display complexity badges on key positions
   - Explain why a move is flagged based on position difficulty

### Future Enhancements

1. **Rating-Specific Models**: Train separate models for different player rating ranges
2. **Time-Pressure Adjustment**: Account for time control effects on complexity perception
3. **Opening Book Integration**: Lower complexity for known opening positions
4. **Maia2 Humanness**: Incorporate Maia2's prediction of "human-like" moves
5. **Active Learning**: Continuously improve model with user feedback on position difficulty
