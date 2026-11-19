"""Local chord diagram of who spoke immediately after whom.

Nothing leaves the machine. Ribbon width and arc size are defined in
CHORD_DEFINITION and drawn with matplotlib only.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use('Agg', force=False)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import PathPatch, Wedge
from matplotlib.path import Path as MplPath

NAVY = '#1C2C5B'
SKY = '#6CABDD'
PARTICIPANT_COLORS = (
    '#1F4E79',
    '#C4554D',
    '#2F6F4E',
    '#E09F3E',
    '#6B4C9A',
    '#D4783A',
    '#0E7C7B',
    '#8C3A4A',
    '#4F6F3A',
    '#3E5A8C',
    '#B86B3D',
    '#7A4E2D',
    '#5C6B8A',
    '#A34B6B',
    '#3F7D56',
)

CHORD_DEFINITION = (
    'Ribbon width is the number of times one person\'s turn was immediately followed '
    'by the other\'s, in either direction, counted from the diarized turns. '
    'Arc size is that person\'s total of those transitions. '
    'A turn followed by the same person is not drawn. '
    'Facilitators are marked with a navy hatch and actors with a sky hatch. '
    'Every arc is labeled with the name confirmed for that person.'
)


def transition_ribbons(
    sequence: list[str], names: list[str] | None = None
) -> tuple[list[str], list[list[int]], list[int]]:
    """Undirected ribbons and arc sizes for a sequence of speaker names.

    Returns (names, symmetric ribbon matrix, arc sizes). A self-follow is ignored.
    Names with no transitions stay in the list with an arc size of 0.
    """
    if names is None:
        names = list(dict.fromkeys(sequence))
    index = {name: position for position, name in enumerate(names)}
    size = len(names)
    ribbon = [[0 for _ in range(size)] for _ in range(size)]
    for previous, following in zip(sequence, sequence[1:]):
        if previous not in index or following not in index or previous == following:
            continue
        left = index[previous]
        right = index[following]
        ribbon[left][right] += 1
        ribbon[right][left] += 1
    arcs = [sum(row) for row in ribbon]
    return names, ribbon, arcs


def _color_for(position: int, count: int) -> str:
    if count <= len(PARTICIPANT_COLORS):
        return PARTICIPANT_COLORS[position % len(PARTICIPANT_COLORS)]
    # Spread extra speakers around the same hues so 10 or more stay distinct.
    base = PARTICIPANT_COLORS[position % len(PARTICIPANT_COLORS)]
    red = int(base[1:3], 16)
    green = int(base[3:5], 16)
    blue = int(base[5:7], 16)
    shift = 28 if (position // len(PARTICIPANT_COLORS)) % 2 else -18
    parts = [max(0, min(255, channel + shift)) for channel in (red, green, blue)]
    return '#{:02X}{:02X}{:02X}'.format(*parts)


def _arc_layout(arcs: list[int]) -> list[tuple[float, float]]:
    """Clockwise arcs as (low, high) radians, starting at the top."""
    count = len(arcs)
    if count == 0:
        return []
    positive = [weight for weight in arcs if weight > 0]
    sliver = (sum(positive) / len(positive) * 0.18) if positive else 1.0
    visual = [weight if weight > 0 else sliver for weight in arcs]
    gap = min(0.06, (2 * math.pi) / (count * 8))
    available = (2 * math.pi) - gap * count
    if available <= 0:
        gap = (2 * math.pi) / (count * 12)
        available = (2 * math.pi) - gap * count
    total = sum(visual)
    cursor = math.pi / 2
    spans = []
    for weight in visual:
        width = available * (weight / total)
        low = cursor - width
        spans.append((low, cursor))
        cursor = low - gap
    return spans


def _ribbon_slices(
    low: float, high: float, weights: list[tuple[int, int]], arc_total: int
) -> dict[int, tuple[float, float]]:
    """Map each partner index to the angular slice it occupies on this arc."""
    slices = {}
    if arc_total <= 0 or high <= low:
        return slices
    cursor = high
    span = high - low
    for partner, weight in weights:
        width = span * (weight / arc_total)
        slices[partner] = (cursor - width, cursor)
        cursor -= width
    return slices


def _polar(angle: float, radius: float) -> tuple[float, float]:
    return (radius * math.cos(angle), radius * math.sin(angle))


def _arc_points(start: float, end: float, radius: float, steps: int = 12) -> list[tuple[float, float]]:
    if end < start:
        start, end = end, start
    return [_polar(angle, radius) for angle in np.linspace(start, end, steps)]


def _ribbon_path(first: tuple[float, float], second: tuple[float, float], radius: float) -> MplPath:
    a0, a1 = sorted(first)
    b0, b1 = sorted(second)
    vertices: list[tuple[float, float]] = []
    codes: list[int] = []

    def add(code: int, point: tuple[float, float]) -> None:
        codes.append(code)
        vertices.append(point)

    # Each edge of the ribbon keeps its own control point so the band stays wide
    # instead of collapsing into a pie slice at the center.
    add(MplPath.MOVETO, _polar(a1, radius))
    add(MplPath.CURVE4, _polar(a1, radius * 0.42))
    add(MplPath.CURVE4, _polar(b1, radius * 0.42))
    add(MplPath.CURVE4, _polar(b1, radius))
    for point in reversed(_arc_points(b0, b1, radius)[:-1]):
        add(MplPath.LINETO, point)
    add(MplPath.CURVE4, _polar(b0, radius * 0.42))
    add(MplPath.CURVE4, _polar(a0, radius * 0.42))
    add(MplPath.CURVE4, _polar(a0, radius))
    for point in _arc_points(a0, a1, radius)[1:]:
        add(MplPath.LINETO, point)
    add(MplPath.CLOSEPOLY, vertices[0])
    return MplPath(vertices, codes)


def _blend(left: str, right: str) -> str:
    channels = []
    for start in (1, 3, 5):
        one = int(left[start : start + 2], 16)
        two = int(right[start : start + 2], 16)
        channels.append((one + two) // 2)
    return '#{:02X}{:02X}{:02X}'.format(*channels)


def save_chord_diagram(
    sequence: list[str],
    path: Path,
    names: list[str] | None = None,
    roles: dict[str, str] | None = None,
    title: str = 'Who talks with whom',
) -> tuple[list[str], list[list[int]], list[int]]:
    """Draw arcs and ribbons to `path`. Returns the same matrices the drawing uses."""
    roles = roles or {}
    names, ribbon, arcs = transition_ribbons(sequence, names)
    count = len(names)
    spans = _arc_layout(arcs)
    colors = [_color_for(position, count) for position in range(count)]
    slices = []
    for position, (low, high) in enumerate(spans):
        partners = [
            (other, ribbon[position][other])
            for other in range(count)
            if other != position and ribbon[position][other] > 0
        ]
        partners.sort(key=lambda item: names[item[0]])
        slices.append(_ribbon_slices(low, high, partners, arcs[position]))

    side = 8 if count <= 8 else 11
    figure, axis = plt.subplots(figsize=(side, side + 0.6))
    axis.set_aspect('equal')
    axis.axis('off')
    axis.set_xlim(-1.55, 1.55)
    axis.set_ylim(-1.55, 1.45)
    outer = 1.0
    thickness = 0.09
    inner = outer - thickness

    for left in range(count):
        for right in range(left + 1, count):
            weight = ribbon[left][right]
            if weight <= 0 or right not in slices[left] or left not in slices[right]:
                continue
            patch = PathPatch(
                _ribbon_path(slices[left][right], slices[right][left], inner),
                facecolor=_blend(colors[left], colors[right]),
                edgecolor=SKY,
                linewidth=0.6,
                alpha=0.55,
                zorder=1,
            )
            axis.add_patch(patch)

    for position, (low, high) in enumerate(spans):
        role = roles.get(names[position], 'participant')
        degrees_low = math.degrees(low)
        degrees_high = math.degrees(high)
        wedge = Wedge(
            (0, 0),
            outer,
            degrees_low,
            degrees_high,
            width=thickness,
            facecolor='#F4F7FB' if role != 'participant' else colors[position],
            edgecolor=NAVY if role != 'actor' else SKY,
            linewidth=1.2 if role != 'participant' else 0.4,
            zorder=2,
        )
        if role == 'facilitator':
            wedge.set_hatch('///')
            wedge.set_edgecolor(NAVY)
        elif role == 'actor':
            wedge.set_hatch('xxx')
            wedge.set_edgecolor(SKY)
        axis.add_patch(wedge)
        mid = (low + high) / 2
        label_radius = 1.14
        axis.text(
            * _polar(mid, label_radius),
            names[position],
            ha='center',
            va='center',
            rotation=_upright_rotation(mid),
            rotation_mode='anchor',
            fontsize=8 if count <= 8 else 7,
            color=NAVY,
            fontweight='medium',
            zorder=3,
        )
        if role in {'facilitator', 'actor'}:
            axis.text(
                *_polar(mid, 1.28),
                role,
                ha='center',
                va='center',
                rotation=_upright_rotation(mid),
                rotation_mode='anchor',
                fontsize=6.5,
                color=NAVY if role == 'facilitator' else '#2E6F99',
                zorder=3,
            )

    axis.set_title(title, color=NAVY, fontsize=14, pad=12, fontweight='semibold')
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, facecolor='white', bbox_inches='tight')
    plt.close(figure)
    return names, ribbon, arcs


def _upright_rotation(angle: float) -> float:
    degrees = math.degrees(angle)
    rotation = degrees - 90
    while rotation > 90:
        rotation -= 180
    while rotation < -90:
        rotation += 180
    return rotation
