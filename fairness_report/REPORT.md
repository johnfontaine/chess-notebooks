# Fairness Analysis Report

The final report will be organized into the following sections

## Page Header

This section will be formatted with a dark background and a title and metadata elements. Each metadata element should be formatted `<Field>: <Value>`. The title of the report will be "Fairness Analysis: [Username]" where the username is linked to the chess.com profile e.g. https://www.chess.com/member/[username]

Metadata Fields:

- Generated: Datetime the report was generated
- Player Status: Current status of the player from chess.com player endpoint e.g. "premium", "closed:fair_play_violations"

## Summary Grid

This will be key metrics for a quick view. This section should not have a subtitle label.

Summary items:

- Games Analyzed
- Accuracy
- ACPL
- Humanness
- Average Z-Score
- Risk Level: High/Medium/Low

## Phase Results Overview

This section will be labeled with a subtitle/subheading "Phase Results". It will provide metric cards for each phase showing status and key metrics.

Metrics should be laid out horizontally from left to right in a grid using metric cards within the section. Layout should adapt to additional rows when the number of metric cards would overflow within the larger container for the phases.

### Phase Results Metrics per phase

- Data Collection
  - Total Games
  - Time Period (the first and last dates that games were played)
- Quick Analysis
  - Current Rating (elo_end)
  - Peak Rating (elo_max)
  - Elo Range (elo_max and elo_min)
  - Wins
  - Losses
  - Draws
  - Rating Manipulation Score
- Prioritization
  - Priority Games (these are games flagged as high priority, up to ten)
  - Min Suspicion Score
- Engine Analysis
  - Games Analyzed
  - Average Centipawn Loss
  - Average Accuracy
  - Best Move Rate
- Ken Regan-esque Analysis
  - Games Analyzed
  - Average Z-Score
  - Max Z-Score
  - Flagged Games (games identified as having a possible issue)
  - Flagged Moves (moves that had z-score indicating possible computer assistance)
- Tablebase
  - Games Analyzed (these are the games that had a tablebase position)
  - Tablebase Accuracy
- Time Analysis
  - Games Analyzed
  - Average Move Time
  - Std Deviation
  - Flagged Games (games with unusually consistent move timings indicating bot play)
- Maia2
  - Games Analyzed
  - Average Humanness
  - Flagged Games (games with inhuman moves)
  - Flagged Moves (moves with low humanness score)
- Risk Assessment
  - Risk Score
  - Risk Level

## Risk Factors

This section will have a subheading title. It should highlight the risk factors if any.

## Phase Details

This section provides detailed metrics tables from individual analysis phases. Each phase shows a table with columns:

- Metric
- Value
- Assessment / Comments

**Note:** Phase Details should NOT include key positions or flagged games inline. These are consolidated in the Key Games and Positions section below.

## Key Games and Positions

This is a **consolidated section** that combines all flagged games and key positions from multiple analysis phases into a single view. This allows reviewers to see overlapping flags and understand the full picture of any suspicious activity.

### Organization

Games are listed in descending order by number of flags/reasons. Each game entry shows:

1. **Game Header**
   - Game ID (linked to Chess.com: `https://www.chess.com/game/live/<game_id>?username=<username>`)
   - Total number of flags/reasons this game was highlighted

2. **Flags Summary** - A list of which analyses flagged this game:
   - Engine Analysis: Shows if game has key positions (blunders/mistakes with CPL > 100)
   - Ken Regan-esque: Shows if game was flagged (Z-score, IPR, suspicion level)
   - Maia2: Shows if game has surprising moves (low probability moves)

3. **Key Positions** (if any) - SVG board renderings with:
   - Board position with arrows (green = played move, blue = best move if different)
   - Move number/ply
   - Error classification badge (Blunder/Mistake/Inaccuracy) for engine positions
   - "Low Probability" badge for Maia2 positions
   - Context explaining why the position was flagged
   - Metrics (CPL, accuracy, Maia2 probability/rank as applicable)
   - Link to specific move on Chess.com: `https://www.chess.com/game/live/<game_id>?username=<username>&move=<ply>`

### Data Sources

The consolidated view pulls from:

1. **Engine Analysis** (`report.key_positions`)
   - Positions with CPL > 100 (significant mistakes)
   - Fields: fen, move, best_move, cpl, error_class, context, svg

2. **Ken Regan-esque Analysis** (`report.regan_flagged_games`)
   - Games where Z-score exceeds threshold (typically 2.0)
   - Fields: z_score, ipr, official_elo, elo_difference, move_match_rate, suspicion_level, acpl

3. **Maia2 Analysis** (`report.maia2_positions`)
   - Moves with low Maia2 probability (unusual for rating level)
   - Fields: fen, move, probability, rank, context, svg

### Benefits

- **Cross-reference**: Easily see if a game is flagged by multiple analyses
- **Holistic view**: Understand the full context of suspicious activity
- **Reduced scrolling**: All flagged content in one place rather than scattered across phase details
- **Overlap detection**: Highlights games that appear in multiple analyses (strongest indicators)

## Glossary

Provide a detailed definition of metrics used in the report.

Note we are using "Ken Regan-esque" because that section is inspired by his work but is not his actual code. Be sure that the report uses this nomenclature when referring to the metrics in that section.

## Citations

Please link to third party projects used in this report:

- [Stockfish](https://stockfishchess.org/) - Chess engine for position evaluation
- [Maia Chess / Maia2](https://github.com/CSSLab/maia-chess) - Human-like neural network for humanness scoring
- [python-chess](https://python-chess.readthedocs.io/) - Python chess library
- [Syzygy Tablebases](https://lichess.org/blog/W3WeMyQAACQAdfAL/7-piece-syzygy-tablebases-are-complete) - Endgame tablebases via Lichess API
- [Ken Regan's Chess Fidelity Research](https://cse.buffalo.edu/~regan/chess/fidelity/) - Inspiration for IPR and Z-score methodology
- [PGN Spy](https://github.com/MGleason1/PGN-Spy) - Reference for centipawn loss calculation
- [Lichess Accuracy Formula](https://lichess.org/page/accuracy) - Win percentage based accuracy formula
