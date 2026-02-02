# Titled Tuesday Cheater Identification

## Purpose

Dataset of titled players who were banned from Titled Tuesday tournaments for fair play violations. This represents a higher-level cheating dataset as these are players with verified chess titles.

## How It's Built

Run the identification script:
```bash
python scripts/find_titled_cheaters.py
```

### Process

1. **Fetches Titled Tuesday tournament data** from Chess.com API
2. **Scans tournament standings** for players with titles (GM, IM, FM, etc.)
3. **Checks player status** via Chess.com profile API
4. **Identifies banned accounts** - players whose accounts are closed for fair play
5. **Records findings** with tournament context

## Directory Structure

```
data/titled-cheaters/
├── titled_cheaters.json     # List of identified titled cheaters
└── tournaments_scanned.json # Log of tournaments processed
```

## Output Format

```json
{
  "username": "example_player",
  "title": "IM",
  "banned_date": "2024-03-15",
  "tournaments_participated": [
    "titled-tuesday-blitz-march-2024",
    "titled-tuesday-blitz-february-2024"
  ],
  "best_finish": 15,
  "total_prize_money": 500
}
```

## Significance

Titled player cheating is particularly notable because:
1. These players have verified OTB (over-the-board) strength
2. They passed FIDE title requirements legitimately
3. Cheating suggests financial motivation (prize money) or rating protection
4. Provides insight into sophisticated cheating patterns

## Usage

This dataset can be:
- Added to the cheater baseline for analysis
- Used for pattern identification unique to skilled cheaters
- Referenced when analyzing titled player games

## API Endpoints Used

- `/pub/tournament/{tournament-id}` - Tournament info and rounds
- `/pub/tournament/{tournament-id}/round/{n}` - Round results
- `/pub/player/{username}` - Player profile and status
