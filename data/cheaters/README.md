# Known Cheater Dataset

## Purpose

Game data from players who have been banned for fair play violations. This dataset provides comparison metrics to identify suspicious patterns that deviate from normal play.

## How It's Built

1. **Configure known cheaters** in `config.json`
2. **Run baseline generation:**
   ```bash
   python scripts/generate_baseline.py --config data/cheaters/config.json --output data/cheaters/
   ```
3. **Data source:** Games played before the account was banned

## Directory Structure

```
data/cheaters/
├── config.json              # List of known cheaters
├── combined_baseline.json   # Aggregated statistics
└── <username>/              # Per-player data (same structure as baseline)
```

## Key Findings vs Trusted Baseline

| Metric | Trusted | Cheaters | Difference | Significance |
|--------|---------|----------|------------|--------------|
| **Loss by checkmate** | 14.0% | 8.8% | -5.2% | **KEY INDICATOR** |
| Win by checkmate | 14.3% | 14.4% | +0.1% | No difference |
| Resignation rate | 43.7% | 48.9% | +5.2% | Cheaters resign more |
| Timeout rate | 20.0% | 16.9% | -3.1% | Cheaters timeout less |

## Critical Insight: Loss by Checkmate

The most significant difference between cheaters and trusted players is in **loss by checkmate rate**:

- **Trusted players:** 14.0% of games lost end in checkmate
- **Cheaters:** Only 8.8% of games lost end in checkmate

This makes sense because:
1. Engine assistance sees threats coming that humans might miss
2. Cheaters are more likely to resign when losing rather than play into checkmate
3. Cheaters don't "miss" back-rank mates, queen traps, etc.

Interestingly, win-by-checkmate rates are virtually identical (14.3% vs 14.4%), suggesting cheaters don't specifically seek checkmate wins.

## Usage in Analysis

```python
# Load cheater baseline for comparison
with open('data/cheaters/combined_baseline.json') as f:
    cheater_baseline = json.load(f)

# Flag if player's loss-by-checkmate rate is suspiciously low
cheater_loss_cm_avg = 0.088  # 8.8%
if player_loss_cm_rate < cheater_loss_cm_avg:
    red_flags.append(f"Very low loss-by-checkmate rate: {player_loss_cm_rate:.1%}")
```

## Caveats

- Small sample size (fewer known cheaters than trusted players)
- Self-selection bias (only caught cheaters are included)
- Historical data (patterns may evolve as detection improves)
- Different skill levels may have different natural checkmate rates
