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
# Dark enough to stay distinct in print. Each person keeps one of these, unmixed.
PARTICIPANT_COLORS = (
    '#0072B2',
    '#D55E00',
    '#009E73',
    '#882255',
    '#E69F00',
    '#332288',
    '#CC79A7',
    '#117733',
    '#AA4499',
    '#44AA99',
    '#8C510A',
    '#4D4D4D',
)

HOW_TO_READ = (
    'Each colored arc is one person. A bigger arc means more back-and-forth.',
    'A ribbon connects two people. Wider means they spoke right after each other more often.',
    'A ribbon takes the color of the person who spoke first, and narrows toward who spoke next.',
    'This counts turn order, not who addressed whom.',
)

CHORD_DEFINITION = ' '.join(HOW_TO_READ)

# A single follow is easy to over-read. Fade those unless the pair is one of the strongest.
FADE_BELOW = 2
STRONG_PAIRS = 3


def roster_names(sequence: list[str], names: list[str] | None = None) -> list[str]:
    if names is None:
        return list(dict.fromkeys(sequence))
    return list(names)


def directed_follows(
    sequence: list[str], names: list[str] | None = None
) -> tuple[list[str], list[list[int]]]:
    """matrix[i][j] is how many times names[i] was immediately followed by names[j].

    A person following themselves is not counted. names[i] is the person who spoke first.
    """
    names = roster_names(sequence, names)
    index = {name: position for position, name in enumerate(names)}
    size = len(names)
    matrix = [[0 for _ in range(size)] for _ in range(size)]
    for previous, following in zip(sequence, sequence[1:]):
        if previous not in index or following not in index or previous == following:
            continue
        matrix[index[previous]][index[following]] += 1
    return names, matrix


def transition_ribbons(
    sequence: list[str], names: list[str] | None = None
) -> tuple[list[str], list[list[int]], list[int]]:
    """Symmetric totals and arc sizes.

    ribbon[i][j] is the back-and-forth between the two people (both directions).
    Arc size is that person's total of those transitions. A self-follow is ignored.
    """
    names, directed = directed_follows(sequence, names)
    size = len(names)
    ribbon = [[0 for _ in range(size)] for _ in range(size)]
    for left in range(size):
        for right in range(left + 1, size):
            total = directed[left][right] + directed[right][left]
            ribbon[left][right] = total
            ribbon[right][left] = total
    arcs = [sum(row) for row in ribbon]
    return names, ribbon, arcs


def _short_name(name: str, names: list[str]) -> str:
    first = name.split(' ', 1)[0]
    if sum(other.split(' ', 1)[0] == first for other in names) == 1:
        return first
    return name


def _join_names(labels: list[str]) -> str:
    if not labels:
        return ''
    if len(labels) == 1:
        return labels[0]
    if len(labels) == 2:
        return f'{labels[0]} and {labels[1]}'
    return ', '.join(labels[:-1]) + f', and {labels[-1]}'


def pair_directions(
    names: list[str], directed: list[list[int]]
) -> list[tuple[str, str, int, int, int]]:
    """Each pair as (name_left, name_right, left_then_right, right_then_left, total).

    The person who spoke first more often is named on the left. Ties keep roster order.
    Pairs are sorted from the most back-and-forth to the least.
    """
    pairs = []
    for left in range(len(names)):
        for right in range(left + 1, len(names)):
            left_then_right = directed[left][right]
            right_then_left = directed[right][left]
            total = left_then_right + right_then_left
            if total <= 0:
                continue
            if right_then_left > left_then_right:
                pairs.append(
                    (names[right], names[left], right_then_left, left_then_right, total)
                )
            else:
                pairs.append(
                    (names[left], names[right], left_then_right, right_then_left, total)
                )
    pairs.sort(key=lambda item: (-item[4], item[0], item[1]))
    return pairs


def chord_caption(names: list[str], directed: list[list[int]]) -> str:
    """One sentence naming the strongest pairs and the least connected people."""
    pairs = pair_directions(names, directed)
    if not pairs:
        return 'No one spoke immediately after anyone else.'
    shown = pairs[:2]
    bits = [
        f'{_short_name(left, names)} ↔ {_short_name(right, names)} ({total})'
        for left, right, _forward, _back, total in shown
    ]
    size = len(names)
    arcs = [
        sum(directed[person][other] + directed[other][person] for other in range(size) if other != person)
        for person in range(size)
    ]
    smallest = min(arcs) if arcs else 0
    largest = max(arcs) if arcs else 0
    if smallest == largest:
        ending = 'Everyone has the same amount of back-and-forth.'
    else:
        least = [_short_name(names[person], names) for person, arc in enumerate(arcs) if arc == smallest]
        if len(least) > 3:
            ending = f'Least connected: {", ".join(least[:3])}, and {len(least) - 3} others.'
        else:
            ending = f'Least connected: {_join_names(least)}.'
    return f'Most back-and-forth: {", ".join(bits)}. {ending}'


def strongest_pair_indexes(directed: list[list[int]], limit: int = STRONG_PAIRS) -> set[tuple[int, int]]:
    scored = []
    size = len(directed)
    for left in range(size):
        for right in range(left + 1, size):
            total = directed[left][right] + directed[right][left]
            if total > 0:
                scored.append((total, left, right))
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    return {(left, right) for _total, left, right in scored[:limit]}


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


def _directed_ribbon(
    source: tuple[float, float], target_angle: float, attach_radius: float
) -> MplPath:
    """Wide on the person who spoke first, narrow toward the person who spoke next."""
    start, end = sorted(source)
    tip_radius = attach_radius - 0.085
    tip_left = _polar(target_angle + 0.02, tip_radius)
    tip_right = _polar(target_angle - 0.02, tip_radius)
    vertices = [
        _polar(end, attach_radius),
        _polar(end, attach_radius * 0.58),
        _polar(target_angle + 0.02, tip_radius + 0.12),
        tip_left,
        tip_right,
        _polar(target_angle - 0.02, tip_radius + 0.12),
        _polar(start, attach_radius * 0.58),
        _polar(start, attach_radius),
    ]
    vertices.extend(_arc_points(start, end, attach_radius)[1:-1])
    vertices.append(vertices[0])
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
    ]
    codes.extend([MplPath.LINETO] * (len(vertices) - 9))
    codes.append(MplPath.CLOSEPOLY)
    return MplPath(vertices, codes)


def _arrowhead(angle: float, attach_radius: float) -> list[tuple[float, float]]:
    return [
        _polar(angle, attach_radius - 0.012),
        _polar(angle + 0.055, attach_radius - 0.072),
        _polar(angle - 0.055, attach_radius - 0.072),
    ]


def _label_alignment(angle: float) -> tuple[str, str]:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    if cosine > 0.25:
        horizontal = 'left'
    elif cosine < -0.25:
        horizontal = 'right'
    else:
        horizontal = 'center'
    if sine > 0.35:
        vertical = 'bottom'
    elif sine < -0.35:
        vertical = 'top'
    else:
        vertical = 'center'
    return horizontal, vertical


def save_chord_diagram(
    sequence: list[str],
    path: Path,
    names: list[str] | None = None,
    roles: dict[str, str] | None = None,
    title: str = 'Who talks with whom',
) -> tuple[list[str], list[list[int]], list[int]]:
    """Draw directed ribbons and return the symmetric totals used for arc size."""
    roles = roles or {}
    names, directed = directed_follows(sequence, names)
    _roster, ribbon, arcs = transition_ribbons(sequence, names)
    count = len(names)
    spans = _arc_layout(arcs)
    colors = [_color_for(position, count) for position in range(count)]
    strong = strongest_pair_indexes(directed)
    peak = max((value for row in directed for value in row), default=0)
    slices = []
    for position, (low, high) in enumerate(spans):
        outgoing = [
            (other, directed[position][other])
            for other in range(count)
            if directed[position][other] > 0
        ]
        outgoing.sort(key=lambda item: names[item[0]])
        slices.append(_ribbon_slices(low, high, outgoing, arcs[position]))

    landings: list[dict[int, float]] = []
    for target, (low, high) in enumerate(spans):
        sources = [source for source in range(count) if directed[source][target] > 0]
        spots = {}
        if len(sources) == 1:
            spots[sources[0]] = (low + high) / 2
        elif sources:
            pad = (high - low) * 0.18
            usable = max(high - low - 2 * pad, (high - low) * 0.5)
            start = low + (high - low - usable) / 2
            step = usable / (len(sources) - 1)
            for index, source in enumerate(sources):
                spots[source] = start + step * index
        landings.append(spots)

    side = 8.4 if count <= 8 else 10.6
    figure = plt.figure(figsize=(side, side + 2.15), facecolor='white')
    axis = figure.add_axes([0.08, 0.30, 0.84, 0.64])
    guide = figure.add_axes([0.08, 0.035, 0.84, 0.24])
    axis.set_aspect('equal')
    axis.axis('off')
    axis.set_xlim(-1.72, 1.72)
    axis.set_ylim(-1.72, 1.72)
    outer = 1.0
    thickness = 0.10
    inner = outer - thickness

    drawings = []
    for source in range(count):
        for target in range(count):
            weight = directed[source][target]
            if weight <= 0 or target not in slices[source]:
                continue
            pair = tuple(sorted((source, target)))
            emphasized = pair in strong
            faded = peak >= FADE_BELOW and weight < FADE_BELOW and not emphasized
            drawings.append((0 if faded else 1 if not emphasized else 2, source, target, weight, emphasized, faded))
    drawings.sort(key=lambda item: item[0])

    for _rank, source, target, _weight, emphasized, faded in drawings:
        alpha = 0.16 if faded else (0.95 if emphasized else 0.78)
        patch = PathPatch(
            _directed_ribbon(slices[source][target], landings[target][source], inner),
            facecolor=colors[source],
            edgecolor=NAVY if emphasized else 'white',
            linewidth=1.15 if emphasized else 0.35,
            alpha=alpha,
            zorder=1 + _rank,
        )
        axis.add_patch(patch)
        if not faded:
            head = plt.Polygon(
                _arrowhead(landings[target][source], inner),
                closed=True,
                facecolor=colors[source],
                edgecolor=NAVY if emphasized else 'none',
                linewidth=0.6 if emphasized else 0,
                alpha=alpha,
                zorder=2 + _rank,
            )
            axis.add_patch(head)

    for position, (low, high) in enumerate(spans):
        role = roles.get(names[position], 'participant')
        wedge = Wedge(
            (0, 0),
            outer,
            math.degrees(low),
            math.degrees(high),
            width=thickness,
            facecolor=colors[position],
            edgecolor='white',
            linewidth=0.8,
            zorder=6,
        )
        if role == 'facilitator':
            wedge.set_hatch('///')
            wedge.set_edgecolor(NAVY)
            wedge.set_linewidth(1.1)
        elif role == 'actor':
            wedge.set_hatch('xxx')
            wedge.set_edgecolor(SKY)
            wedge.set_linewidth(1.1)
        axis.add_patch(wedge)
        mid = (low + high) / 2
        horizontal, vertical = _label_alignment(mid)
        label_x, label_y = _polar(mid, 1.28 if count <= 8 else 1.34)
        axis.plot(
            [_polar(mid, outer + 0.01)[0], label_x * 0.96],
            [_polar(mid, outer + 0.01)[1], label_y * 0.96],
            color=NAVY,
            linewidth=0.4,
            alpha=0.45,
            zorder=3,
        )
        axis.scatter(
            [_polar(mid, outer + 0.02)[0]],
            [_polar(mid, outer + 0.02)[1]],
            s=18,
            color=colors[position],
            zorder=7,
        )
        label = f'{names[position]} ({arcs[position]})'
        axis.text(
            label_x,
            label_y,
            label,
            ha=horizontal,
            va=vertical,
            fontsize=8 if count <= 8 else 7,
            color=NAVY,
            zorder=8,
        )
        if role in {'facilitator', 'actor'}:
            role_x, role_y = _polar(mid, 1.46 if count <= 8 else 1.52)
            axis.text(
                role_x,
                role_y,
                role,
                ha=horizontal,
                va=vertical,
                fontsize=6.5,
                color=NAVY,
                zorder=8,
            )

    axis.set_title(title, color=NAVY, fontsize=15, pad=8, fontweight='semibold')
    _draw_reading_guide(guide, chord_caption(names, directed))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, facecolor='white')
    plt.close(figure)
    return names, ribbon, arcs


def _draw_reading_guide(axis, caption: str) -> None:
    import textwrap
    from matplotlib.patches import FancyBboxPatch

    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis('off')
    axis.add_patch(
        FancyBboxPatch(
            (0.01, 0.40),
            0.98,
            0.58,
            boxstyle='round,pad=0.012,rounding_size=0.02',
            facecolor='#F4F7FB',
            edgecolor=NAVY,
            linewidth=1.1,
            mutation_aspect=0.35,
        )
    )
    axis.text(0.04, 0.88, 'How to read this chart', color=NAVY, fontsize=11, fontweight='bold')
    for index, line in enumerate(HOW_TO_READ):
        axis.text(0.04, 0.76 - index * 0.09, line, color=NAVY, fontsize=8.2, va='center')
    wrapped = textwrap.wrap(caption, width=108) or ['']
    for index, line in enumerate(wrapped[:3]):
        axis.text(0.01, 0.22 - index * 0.09, line, color=NAVY, fontsize=9, fontweight='medium', va='center')
