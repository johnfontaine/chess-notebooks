# Chess Notebooks

A collection of Jupyter notebooks and Python scripts for analyzing chess games, with a focus on player fairness evaluation using metrics like centipawn loss, Lichess accuracy, and Ken Regan's method.

## Features

- **Game Analysis**: Fetch and analyze games from Chess.com API
- **Engine Analysis**: Multi-depth Stockfish analysis with move quality metrics
- **Accuracy Metrics**: Centipawn loss, Lichess-style accuracy, position complexity
- **Cheat Detection**: Ken Regan's IPR method, Z-score analysis, suspicious position detection
- **Human Play Analysis**: Maia 2 integration for humanness scoring
- **Visualization**: Board rendering, key position highlighting, statistical charts

## Prerequisites

### Python 3.12+

**macOS:**
```bash
brew install python@3.12
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install python3.12 python3.12-venv python3-pip
```

### Stockfish Chess Engine

The analysis requires Stockfish for position evaluation.

**macOS:**
```bash
brew install stockfish
```

**Ubuntu/Debian:**
```bash
sudo apt install stockfish
```

**Verify installation:**
```bash
which stockfish
# Should output: /opt/homebrew/bin/stockfish (macOS) or /usr/games/stockfish (Linux)
```

### Syzygy Tablebases (Optional)

For perfect endgame analysis with 6 or fewer pieces. Download from [lichess.org/page/tablebases](https://lichess.org/page/tablebases).

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/chess-notebooks.git
   cd chess-notebooks
   ```

2. **Create and activate virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment (optional):**
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

5. **Download Maia 2 models (optional, for humanness analysis):**
   ```bash
   mkdir -p maia2_models
   # Download models from https://github.com/CSSLab/maia2
   ```

## Quick Start

1. **Activate the environment:**
   ```bash
   source .venv/bin/activate
   ```

2. **Start Jupyter:**
   ```bash
   jupyter notebook
   ```

3. **Open a notebook:**
   - `updated-player-fairness.ipynb` - Main analysis notebook
   - `position-playability.ipynb` - Single position deep dive
   - `multidepth-exploration.ipynb` - Multi-depth analysis patterns

## Project Structure

```
chess-notebooks/
├── chess_analysis/          # Core analysis library
│   ├── engine.py           # Stockfish integration
│   ├── metrics.py          # Accuracy calculations
│   ├── regan_analysis.py   # Ken Regan's method
│   ├── maia2_analysis.py   # Maia 2 humanness
│   ├── fragility.py        # Position fragility
│   ├── tablebase.py        # Endgame tablebases
│   └── ...
├── scripts/                 # Command-line tools
│   ├── generate_baseline.py
│   ├── multidepth_probe.py
│   └── position_playability.py
├── data/                    # Game data (gitignored)
│   ├── trusted/            # Trusted player baselines
│   └── cheaters/           # Known cheater data
├── reports/                 # Analysis output
├── maia2_models/           # Maia 2 model files
├── requirements.txt
├── .env.example
└── README.md
```

## Configuration

Copy `.env.example` to `.env` and configure:

```bash
# Stockfish path (auto-detected if in PATH)
STOCKFISH_PATH=/opt/homebrew/bin/stockfish

# Syzygy tablebase path (optional)
SYZYGY_PATH=/path/to/syzygy

# Chess.com API (no auth required for public data)
# CHESS_COM_USERNAME=your_username
```

## Notebooks

### `updated-player-fairness.ipynb`
The main analysis notebook with a phased approach:
1. Data Collection - Fetch games from Chess.com API
2. Quick Analysis - Elo/result patterns, baseline comparison
3. Game Prioritization - Score games for suspiciousness
4. Deep Analysis - Multi-depth Stockfish, Ken Regan analysis
5. Reports - Accuracy, time usage, key positions

### `position-playability.ipynb`
Analyze a single position in depth:
- Evaluate ALL legal moves at multiple depths
- Track how move quality changes with analysis depth
- Identify "trap" moves that look good but degrade

### `multidepth-exploration.ipynb`
Explore patterns across many positions:
- When do best moves emerge (shallow vs deep)?
- Correlation with fragility, material, game phase
- Statistical analysis of move complexity

## Scripts

```bash
# Generate baseline statistics from trusted players
python scripts/generate_baseline.py --config data/trusted/config.json

# Run multi-depth analysis on baseline games
python scripts/multidepth_probe.py --games 100

# Analyze position playability
python scripts/position_playability.py --games 2 --output reports/playability.json
```

## References

### Chess APIs
- [Chess.com Published Data API](https://www.chess.com/news/view/published-data-api)
- [Lichess API](https://lichess.org/api)

### Cheat Detection Research
- [Ken Regan's Publications on Chess](https://cse.buffalo.edu/~regan/publications.html#chess)
- [FIDE Advanced Cheat Detection](https://www.chess.com/blog/Jordi641/advanced-cheat-detection-algorithms)
- [Towards Transparent Cheat Detection](https://dl.acm.org/doi/10.1007/978-3-031-34017-8_14)
- [Detecting Fair Play Violations](https://ceur-ws.org/Vol-3885/paper13.pdf)

### Position Analysis
- [Fragility in Chess Positions](https://arxiv.org/html/2410.02333v3)
- [Maia 2: Human-like Chess AI](https://www.cs.toronto.edu/~ashton/pubs/maia2-neurips2024.pdf)
- [Maia Chess](https://www.maiachess.com) | [GitHub](https://github.com/CSSLab/maia2)

### Metrics & Rating
- [Centipawn Loss & Elo Correlation](https://medium.com/@enzo.leon/data-science-and-chess-centipawn-loss-elo-correlation-e06089efd8b8)
- [Average Centipawn Loss](https://www.chess.com/blog/raync910/average-centipawn-loss-chess-acpl)
- [PGN Spy](https://github.com/MGleason1/PGN-Spy)
- [Glicko-2 Rating System](https://www.glicko.net/glicko/glicko2.pdf)

### Datasets
- [Titled Tuesday Statistical Analysis](https://github.com/golkir/titled-tuesday-chess-statistical-analysis)

## License

© 2025 John Fontaine

This project is licensed under the MIT License. See the LICENSE file for details.
