"""
Board visualization for key positions using python-chess SVG.

Provides functions for rendering chess positions with move annotations
including arrows for played moves, best moves, and trap moves.
"""

import chess
import chess.svg
from typing import Optional, Tuple
from dataclasses import dataclass


# Arrow colors as specified in the plan
COLORS = {
    'played': '#15781B',        # Green - move played
    'best_dark': '#003088',     # Dark blue - best move (primary)
    'best_light': '#0066ff',    # Light blue - best move (secondary)
    'trap': '#882020',          # Red - trap move
    'mistake': '#c43432',       # Red variant - mistake/blunder
}

# Square fill colors
SQUARE_COLORS = {
    'from': '#ccf',             # Light blue for origin square
    'to': '#ccf',               # Light blue for destination square
    'highlight': '#fce94f',     # Yellow for important squares
}


@dataclass
class MoveArrow:
    """Represents an arrow to draw on the board."""
    from_square: chess.Square
    to_square: chess.Square
    color: str
    weight: float = 1.0  # Line thickness multiplier


def render_position(
    fen: str,
    played_move: Optional[str] = None,
    best_move: Optional[str] = None,
    alt_moves: Optional[list[str]] = None,
    trap_move: Optional[str] = None,
    flipped: bool = False,
    size: int = 400,
    show_coordinates: bool = True,
) -> str:
    """
    Render a chess position as SVG with annotated arrows.

    Args:
        fen: Position in FEN format.
        played_move: Move played (UCI notation) - shown with green arrow.
        best_move: Best engine move (UCI) - shown with dark blue arrow.
        alt_moves: Alternative good moves (UCI list) - shown with lighter blue arrows.
        trap_move: Trap/bad move to avoid (UCI) - shown with red arrow.
        flipped: If True, show board from black's perspective.
        size: Size of the SVG in pixels.
        show_coordinates: If True, show file/rank labels.

    Returns:
        SVG string for the position.
    """
    board = chess.Board(fen)

    arrows = []
    fill = {}

    # Add arrow for played move (green)
    if played_move:
        try:
            move = chess.Move.from_uci(played_move)
            arrows.append(chess.svg.Arrow(
                move.from_square,
                move.to_square,
                color=COLORS['played']
            ))
            # Highlight move squares
            fill[move.from_square] = SQUARE_COLORS['from']
            fill[move.to_square] = SQUARE_COLORS['to']
        except (ValueError, AttributeError):
            pass

    # Add arrow for best move (dark blue)
    if best_move:
        try:
            move = chess.Move.from_uci(best_move)
            arrows.append(chess.svg.Arrow(
                move.from_square,
                move.to_square,
                color=COLORS['best_dark']
            ))
        except (ValueError, AttributeError):
            pass

    # Add arrows for alternative moves (lighter blue)
    if alt_moves:
        for uci in alt_moves[:3]:  # Limit to 3 alternatives
            try:
                move = chess.Move.from_uci(uci)
                arrows.append(chess.svg.Arrow(
                    move.from_square,
                    move.to_square,
                    color=COLORS['best_light']
                ))
            except (ValueError, AttributeError):
                pass

    # Add arrow for trap move (red)
    if trap_move:
        try:
            move = chess.Move.from_uci(trap_move)
            arrows.append(chess.svg.Arrow(
                move.from_square,
                move.to_square,
                color=COLORS['trap']
            ))
        except (ValueError, AttributeError):
            pass

    return chess.svg.board(
        board,
        arrows=arrows,
        fill=fill,
        flipped=flipped,
        size=size,
        coordinates=show_coordinates,
    )


def render_key_position(
    position_data: dict,
    is_white: bool = True,
    size: int = 400,
) -> Tuple[str, dict]:
    """
    Render a key position with full context.

    Args:
        position_data: Dictionary containing:
            - fen: Position FEN
            - move: Move played (UCI)
            - best_move: Best engine move (UCI, optional)
            - alt_moves: Alternative moves (list of UCI, optional)
            - trap_move: Trap move to avoid (UCI, optional)
            - cpl: Centipawn loss
            - accuracy: Move accuracy %
            - accuracy_class: Classification (Best/Excellent/Good/etc.)
            - time_spent: Time spent on move (optional)
            - fragility: Fragility score (optional)
            - complexity: Complexity score (optional)
            - flag_reason: Why this position was flagged (optional)
        is_white: True if player is white (for board orientation).
        size: Size of the SVG in pixels.

    Returns:
        Tuple of (svg_string, summary_dict) where summary_dict contains
        formatted strings for display below the board.
    """
    fen = position_data.get('fen', '')
    played_move = position_data.get('move', '')
    best_move = position_data.get('best_move')
    alt_moves = position_data.get('alt_moves', [])
    trap_move = position_data.get('trap_move')

    svg = render_position(
        fen=fen,
        played_move=played_move,
        best_move=best_move,
        alt_moves=alt_moves,
        trap_move=trap_move,
        flipped=not is_white,  # Flip if player is black
        size=size,
    )

    # Build summary for display below board
    summary = {
        'move_played': played_move,
        'move_san': _uci_to_san(fen, played_move),
    }

    if best_move and best_move != played_move:
        summary['best_move'] = best_move
        summary['best_san'] = _uci_to_san(fen, best_move)

    if 'cpl' in position_data:
        summary['cpl'] = position_data['cpl']

    if 'accuracy' in position_data:
        summary['accuracy'] = f"{position_data['accuracy']:.1f}%"

    if 'accuracy_class' in position_data:
        summary['classification'] = position_data['accuracy_class']

    if 'time_spent' in position_data and position_data['time_spent']:
        summary['time'] = f"{position_data['time_spent']:.1f}s"

    if 'fragility' in position_data and position_data['fragility']:
        summary['fragility'] = f"{position_data['fragility']:.3f}"

    if 'complexity' in position_data and position_data['complexity']:
        summary['complexity'] = f"{position_data['complexity']:.1f}"

    if 'maia_probability' in position_data:
        summary['maia_prob'] = f"{position_data['maia_probability']*100:.1f}%"

    if 'flag_reason' in position_data:
        summary['flag'] = position_data['flag_reason']

    return svg, summary


def _uci_to_san(fen: str, uci: str) -> str:
    """Convert UCI move to SAN notation."""
    if not uci:
        return ''
    try:
        board = chess.Board(fen)
        move = chess.Move.from_uci(uci)
        return board.san(move)
    except (ValueError, AttributeError):
        return uci


def display_key_position(
    position_data: dict,
    is_white: bool = True,
    size: int = 400,
    title: Optional[str] = None,
) -> None:
    """
    Display a key position in Jupyter notebook with SVG and summary.

    Args:
        position_data: Position data dictionary (see render_key_position).
        is_white: True if player is white.
        size: Size of the SVG in pixels.
        title: Optional title for the position.
    """
    from IPython.display import display, HTML, SVG

    svg, summary = render_key_position(position_data, is_white, size)

    # Build HTML output
    html_parts = []

    if title:
        html_parts.append(f'<h4 style="margin-bottom: 10px;">{title}</h4>')

    html_parts.append(svg)

    # Summary table
    html_parts.append('<table style="margin-top: 10px; font-size: 12px;">')

    if 'move_san' in summary:
        played_label = f"Played: <b>{summary['move_san']}</b>"
        if 'best_san' in summary:
            played_label += f" (Best: {summary['best_san']})"
        html_parts.append(f'<tr><td>{played_label}</td></tr>')

    metrics = []
    if 'accuracy' in summary:
        metrics.append(f"Accuracy: {summary['accuracy']}")
    if 'classification' in summary:
        metrics.append(f"[{summary['classification']}]")
    if 'cpl' in summary:
        metrics.append(f"CPL: {summary['cpl']}")
    if metrics:
        html_parts.append(f'<tr><td>{" | ".join(metrics)}</td></tr>')

    context = []
    if 'time' in summary:
        context.append(f"Time: {summary['time']}")
    if 'fragility' in summary:
        context.append(f"Fragility: {summary['fragility']}")
    if 'complexity' in summary:
        context.append(f"Complexity: {summary['complexity']}")
    if 'maia_prob' in summary:
        context.append(f"Maia: {summary['maia_prob']}")
    if context:
        html_parts.append(f'<tr><td>{" | ".join(context)}</td></tr>')

    if 'flag' in summary:
        html_parts.append(f'<tr><td style="color: #c43432;"><b>{summary["flag"]}</b></td></tr>')

    html_parts.append('</table>')

    display(HTML('\n'.join(html_parts)))


def find_key_positions(
    position_data: list[dict],
    evaluations: list[dict] = None,
    n_fragile: int = 3,
    n_complex: int = 3,
    n_mistakes: int = 3,
    n_brilliant: int = 3,
    official_elo: Optional[int] = None,
    n_suspicious: int = 3,
) -> dict:
    """
    Find key positions from analyzed game data.

    Args:
        position_data: List of position dictionaries from extract_position_data().
        evaluations: Optional engine evaluations for best move info.
        n_fragile: Number of fragile peaks to return.
        n_complex: Number of complex positions to return.
        n_mistakes: Number of mistakes/blunders to return.
        n_brilliant: Number of potential brilliant moves to return.
        official_elo: Player's Elo rating (for Regan suspicious position detection).
        n_suspicious: Number of suspicious positions to return (Regan analysis).

    Returns:
        Dictionary with:
        - fragile_peaks: Positions with highest fragility
        - complex_positions: Positions with highest complexity
        - mistakes: Biggest mistakes/blunders
        - brilliant_moves: Potential brilliant moves (low CPL in complex positions)
        - suspicious: Positions flagged by Regan analysis (if official_elo provided)
    """
    # Sort by fragility for peaks
    fragile = sorted(
        [p for p in position_data if p.get('fragility', 0) > 0],
        key=lambda x: x.get('fragility', 0),
        reverse=True
    )[:n_fragile]

    for p in fragile:
        p['flag_reason'] = f"Fragility peak: {p.get('fragility', 0):.3f}"

    # Sort by complexity (if available in position data or can be computed)
    complex_positions = sorted(
        [p for p in position_data if p.get('complexity', 0) > 0],
        key=lambda x: x.get('complexity', 0),
        reverse=True
    )[:n_complex]

    for p in complex_positions:
        p['flag_reason'] = f"High complexity: {p.get('complexity', 0):.1f}"

    # Find mistakes (lowest accuracy)
    mistakes = sorted(
        [p for p in position_data if p.get('accuracy_class') in ['Blunder', 'Mistake']],
        key=lambda x: x.get('accuracy', 100)
    )[:n_mistakes]

    for p in mistakes:
        p['flag_reason'] = f"{p.get('accuracy_class', 'Error')}: CPL {p.get('cpl', 0)}"

    # Find potential brilliant moves (high accuracy in complex/fragile positions)
    brilliant = sorted(
        [p for p in position_data
         if p.get('accuracy', 0) >= 95
         and (p.get('fragility', 0) > 0.5 or p.get('legal_moves', 0) > 30)],
        key=lambda x: x.get('fragility', 0) + (x.get('legal_moves', 0) / 50),
        reverse=True
    )[:n_brilliant]

    for p in brilliant:
        p['flag_reason'] = f"Brilliant: {p.get('accuracy', 0):.0f}% in complex position"

    result = {
        'fragile_peaks': fragile,
        'complex_positions': complex_positions,
        'mistakes': mistakes,
        'brilliant_moves': brilliant,
    }

    # Add Regan suspicious positions if Elo is provided
    if official_elo is not None:
        from .regan_analysis import get_regan_key_positions
        regan_positions = get_regan_key_positions(
            position_data, official_elo,
            n_suspicious=n_suspicious,
            n_best_moves_in_complex=n_suspicious
        )
        result['suspicious'] = regan_positions['suspicious']
        result['best_in_complex'] = regan_positions['best_in_complex']
        result['regan_summary'] = regan_positions['summary']

    return result
