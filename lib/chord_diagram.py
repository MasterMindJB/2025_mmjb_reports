"""Local chord diagram of who spoke immediately after whom.

Nothing leaves the machine. Ribbons are drawn with matplotlib, set in the
vendored Poppins files, and colored from the HTS navy-to-sky palette.
"""

from __future__ import annotations

import math
import re
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use('Agg', force=False)

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch, PathPatch, Rectangle
from matplotlib.path import Path as MplPath
from fontTools.ttLib import TTFont

NAVY = '#1C2C5B'
NAVY_DEEP = '#121D3D'
INK = '#14214A'
SKY = '#6CABDD'
ON_NAVY_MUTED = '#AFC3DE'
LIGHT = '#F4F8FC'
MUTED = '#55628A'
LINE = '#D8E3F0'
BLUEPRINT = '#AFCBEA'
WHITE = '#FFFFFF'

# Tonal, print-safe, and ordered so neighbors contrast. Amber is heat only, and red
# belongs on the website, so neither is a person color.
PARTICIPANT_COLORS = (
    '#1C2C5B',
    '#AFCBEA',
    '#2F4A86',
    '#8FC0E6',
    '#3F6FB0',
    '#7F8FB5',
    '#121D3D',
    '#AFC3DE',
    '#4A6794',
    '#6CABDD',
    '#55628A',
    '#9BB8D4',
)

HOW_TO_READ = (
    'Each colored arc is one person. A bigger arc means more back-and-forth.',
    'A ribbon connects two people. Wider means they spoke right after each other more often.',
    'A ribbon takes the color of the person who spoke first, and narrows toward who spoke next.',
    'This counts turn order, not who addressed whom.',
)

CHORD_DEFINITION = ' '.join(HOW_TO_READ)

FADE_BELOW = 2
STRONG_PAIRS = 3
KICKER = 'CONVERSATION FLOW'
SUBTITLE = 'Who spoke right after whom'

_FONT_DIR = Path(__file__).resolve().parents[1] / 'assets' / 'fonts' / 'Poppins'
_FONT_FILES = {
    'regular': 'Poppins-Regular.ttf',
    'medium': 'Poppins-Medium.ttf',
    'semibold': 'Poppins-SemiBold.ttf',
    'bold': 'Poppins-Bold.ttf',
    'extrabold': 'Poppins-ExtraBold.ttf',
}

_PAIR_PATTERN = re.compile(r'[^:,\n]+ and [^:,\n]+\(\d+\)')


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
        f'{_short_name(left, names)} and {_short_name(right, names)} ({total})'
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


def font_dir() -> Path:
    return _FONT_DIR


@lru_cache(maxsize=None)
def load_poppins(weight: str) -> FontProperties:
    """Load one vendored Poppins face and register it with matplotlib.

    Raises if the file is missing or matplotlib would have to substitute another family.
    Static Poppins files use their own family names for Medium, SemiBold, and ExtraBold,
    so each face is loaded from its file rather than by a CSS weight.
    """
    filename = _FONT_FILES.get(weight)
    if filename is None:
        known = ', '.join(_FONT_FILES)
        raise ValueError(f'Unknown Poppins weight {weight!r}. Expected one of: {known}.')
    path = _FONT_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(
            f'Poppins {weight} is not at {path}. The chord diagram requires the vendored font.'
        )
    font_manager.fontManager.addfont(str(path))
    prop = FontProperties(fname=str(path))
    resolved = Path(font_manager.findfont(prop, fallback_to_default=False)).resolve()
    if resolved != path.resolve() or not prop.get_name().startswith('Poppins'):
        raise RuntimeError(
            f'Poppins {weight} did not load from {path}. Matplotlib resolved {resolved} ({prop.get_name()}).'
        )
    return prop


def _color_for(position: int) -> str:
    return PARTICIPANT_COLORS[position % len(PARTICIPANT_COLORS)]


def _contour(color: str) -> str:
    """A slightly deeper edge so a pale arc still reads on the light page."""
    red = int(color[1:3], 16)
    green = int(color[3:5], 16)
    blue = int(color[5:7], 16)
    mix = 0.32
    deep = (0x12, 0x1D, 0x3D)
    channels = [
        int(channel * (1 - mix) + target * mix) for channel, target in zip((red, green, blue), deep)
    ]
    return '#{:02X}{:02X}{:02X}'.format(*channels)


def _arc_layout(arcs: list[int]) -> list[tuple[float, float]]:
    """Clockwise arcs as (low, high) radians, starting at the top, with even gaps."""
    count = len(arcs)
    if count == 0:
        return []
    positive = [weight for weight in arcs if weight > 0]
    sliver = (sum(positive) / len(positive) * 0.2) if positive else 1.0
    visual = [weight if weight > 0 else sliver for weight in arcs]
    gap = 0.055
    available = (2 * math.pi) - gap * count
    if available <= 0.4:
        gap = (2 * math.pi) / (count * 14)
        available = (2 * math.pi) - gap * count
    total = sum(visual) or 1.0
    cursor = math.pi / 2
    spans = []
    for weight in visual:
        width = available * (weight / total)
        spans.append((cursor - width, cursor))
        cursor -= width + gap
    return spans


def _ribbon_slices(
    low: float, high: float, weights: list[tuple[int, int]], arc_total: int
) -> dict[int, tuple[float, float]]:
    """Map each partner index to the angular slice it occupies on this arc."""
    slices: dict[int, tuple[float, float]] = {}
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


def _angular_separation(start: float, end: float) -> float:
    return abs((start - end + math.pi) % (2 * math.pi) - math.pi)


def _arc_points(start: float, end: float, radius: float, steps: int = 10) -> list[tuple[float, float]]:
    if end < start:
        start, end = end, start
    if steps < 2:
        steps = 2
    return [
        _polar(start + (end - start) * step / (steps - 1), radius) for step in range(steps)
    ]


def _directed_ribbon(source: tuple[float, float], target_angle: float, radius: float) -> MplPath:
    """A tapered bezier from the speaker to who spoke next.

    Control points stay inside the disk. A bezier stays in the convex hull of its
    controls, and that hull cannot reach a neighboring arc, so a ribbon between
    neighbors cannot loop back along the rim.
    """
    start, end = sorted(source)
    mid = (start + end) / 2
    separation = _angular_separation(mid, target_angle)
    # A moderate bow: deep enough to stay off the rim, shallow enough to read as a band.
    fraction = 0.20 + 0.24 * (separation / math.pi) ** 0.85
    if separation < 0.55:
        fraction = 0.14
    tip_radius = radius - 0.045
    source_width = max(end - start, 0.045)
    # Gentle taper. The far end stays a band, not a point or an arrowhead.
    half_tip = min(max(source_width * 0.34, 0.02), 0.12)
    tip_low = target_angle - half_tip
    tip_high = target_angle + half_tip

    def control(angle: float) -> tuple[float, float]:
        return _polar(angle, radius * fraction)

    vertices: list[tuple[float, float]] = [
        _polar(end, radius),
        control(end),
        control(tip_high),
        _polar(tip_high, tip_radius),
    ]
    codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4]
    # Short rounded end. The high tip is already the current point.
    for point in list(reversed(_arc_points(tip_low, tip_high, tip_radius, steps=7)))[1:]:
        vertices.append(point)
        codes.append(MplPath.LINETO)
    vertices.extend([control(tip_low), control(start), _polar(start, radius)])
    codes.extend([MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])
    for point in _arc_points(start, end, radius, steps=16)[1:-1]:
        vertices.append(point)
        codes.append(MplPath.LINETO)
    vertices.append(vertices[0])
    codes.append(MplPath.CLOSEPOLY)
    return MplPath(vertices, codes)


@lru_cache(maxsize=None)
def _font_metrics(font_path: str):
    font = TTFont(font_path)
    return font['head'].unitsPerEm, font.getBestCmap(), font['hmtx']


def _text_width(text: str, prop: FontProperties, size: float) -> float:
    """Advance width in points, including spaces. TextPath drops spaces."""
    if text == '':
        return 0.0
    units, cmap, metrics = _font_metrics(str(Path(prop.get_file()).resolve()))
    total = 0
    for character in text:
        glyph = cmap.get(ord(character))
        if glyph is None:
            total += int(units * 0.33)
        else:
            total += metrics[glyph][0]
    return total / units * size


def _rounded_arc_path(low: float, high: float, outer: float, thickness: float) -> MplPath:
    """One filled ring segment with round caps, so the ends don't read as bubbles."""
    cap = thickness / 2
    midline = outer - cap
    inner = outer - thickness
    cap_angle = math.atan2(cap, midline)
    if high - low <= cap_angle * 2.2:
        center = _polar((low + high) / 2, midline)
        steps = 16
        points = [
            (
                center[0] + cap * math.cos(2 * math.pi * step / steps),
                center[1] + cap * math.sin(2 * math.pi * step / steps),
            )
            for step in range(steps)
        ]
        points.append(points[0])
        return MplPath(points, [MplPath.MOVETO] + [MplPath.LINETO] * (steps - 1) + [MplPath.CLOSEPOLY])
    start = low + cap_angle
    end = high - cap_angle
    points: list[tuple[float, float]] = []
    points.extend(_arc_points(start, end, outer, steps=20))
    cap_center = _polar(end, midline)
    for step in range(1, 8):
        theta = end + math.pi * step / 7
        points.append((cap_center[0] + cap * math.cos(theta), cap_center[1] + cap * math.sin(theta)))
    points.extend(list(reversed(_arc_points(start, end, inner, steps=20)))[1:])
    cap_center = _polar(start, midline)
    for step in range(1, 8):
        theta = start + math.pi + math.pi * step / 7
        points.append((cap_center[0] + cap * math.cos(theta), cap_center[1] + cap * math.sin(theta)))
    points.append(points[0])
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(points) - 2) + [MplPath.CLOSEPOLY]
    return MplPath(points, codes)


def _add_arc(axis, low: float, high: float, outer: float, thickness: float, color: str) -> None:
    if high <= low:
        return
    axis.add_patch(
        PathPatch(
            _rounded_arc_path(low, high, outer, thickness),
            facecolor=color,
            edgecolor=_contour(color),
            linewidth=0.55,
            joinstyle='round',
            capstyle='round',
            zorder=6,
        )
    )


def _horizontal_alignment(angle: float) -> tuple[str, str, float]:
    """Horizontal label alignment and how far outside the arc the block starts."""
    normalized = math.atan2(math.sin(angle), math.cos(angle))
    if math.sin(normalized) > 0.93:
        return 'center', 'bottom', 1.11
    if math.sin(normalized) < -0.93:
        return 'center', 'top', 1.11
    if math.cos(normalized) >= 0:
        return 'left', 'center', 1.07
    return 'right', 'center', 1.07


def _draw_labels(axis, names, arcs, spans, roles, colors) -> None:
    count = len(names)
    name_size = 10.0 if count <= 8 else 8.2
    count_size = 7.4 if count <= 8 else 6.4
    role_size = 6.1 if count <= 8 else 5.4
    name_font = load_poppins('medium')
    count_font = load_poppins('regular')
    role_font = load_poppins('semibold')
    pill = {
        'boxstyle': 'round,pad=0.22',
        'facecolor': SKY,
        'edgecolor': SKY,
        'linewidth': 0.4,
    }
    for position, (low, high) in enumerate(spans):
        angle = (low + high) / 2
        align, vertical, radius = _horizontal_alignment(angle)
        anchor = _polar(angle, radius)
        name = names[position]
        exchanges = arcs[position]
        exchange_word = 'exchange' if exchanges == 1 else 'exchanges'
        role = roles.get(name, 'participant')
        has_role = role in {'facilitator', 'actor'}
        axis.text(
            anchor[0],
            anchor[1],
            name,
            ha=align,
            va=vertical,
            fontproperties=name_font,
            fontsize=name_size,
            color=INK,
            clip_on=False,
            zorder=8,
        )
        if vertical == 'bottom':
            first_line = -14
        elif vertical == 'top':
            first_line = -(name_size + 8)
        else:
            first_line = -(name_size * 0.62 + 9)
        second_line = first_line - 17
        if has_role:
            axis.annotate(
                'FACILITATOR' if role == 'facilitator' else 'ACTOR',
                xy=anchor,
                xytext=(0, first_line),
                textcoords='offset points',
                ha=align,
                va='center',
                fontproperties=role_font,
                fontsize=role_size,
                color=WHITE,
                bbox=pill,
                clip_on=False,
                zorder=8,
                annotation_clip=False,
            )
        axis.annotate(
            f'{exchanges} {exchange_word}',
            xy=anchor,
            xytext=(0, second_line if has_role else first_line),
            textcoords='offset points',
            ha=align,
            va='center',
            fontproperties=count_font,
            fontsize=count_size,
            color=MUTED,
            clip_on=False,
            zorder=8,
            annotation_clip=False,
        )


def _tracked(figure, x: float, y: float, text: str, prop: FontProperties, size: float, color: str, tracking: float) -> None:
    cursor = x
    width = figure.get_figwidth() * 72
    for character in text:
        figure.text(
            cursor,
            y,
            character,
            fontproperties=prop,
            fontsize=size,
            color=color,
            ha='left',
            va='center',
        )
        cursor += (_text_width(character, prop, size) + tracking) / width


def _caption_runs(caption: str) -> list[tuple[str, str]]:
    """Whole phrases, so each run keeps the font's own spacing."""
    runs: list[tuple[str, str]] = []
    position = 0
    for match in _PAIR_PATTERN.finditer(caption):
        if match.start() > position:
            runs.append((caption[position : match.start()], 'regular'))
        runs.append((match.group().strip(), 'semibold'))
        position = match.end()
    if position < len(caption):
        runs.append((caption[position:], 'regular'))
    return [(text, weight) for text, weight in runs if text]


def _draw_caption(figure, caption: str, x: float, y: float, width: float) -> float:
    """One measured line. Pair phrases are SemiBold and do not collide with the rest."""
    regular = load_poppins('regular')
    semibold = load_poppins('semibold')
    size = 9.0
    figure_points = figure.get_figwidth() * 72
    line_gap = 14 / (figure.get_figheight() * 72)
    cursor_x = x
    cursor_y = y
    limit = x + width
    runs = _caption_runs(caption)
    for index, (text, weight) in enumerate(runs):
        prop = semibold if weight == 'semibold' else regular
        piece = text
        token_width = _text_width(piece, prop, size) / figure_points
        if cursor_x > x and cursor_x + token_width > limit:
            cursor_x = x
            cursor_y -= line_gap
            piece = text.lstrip()
            if piece == '':
                continue
            token_width = _text_width(piece, prop, size) / figure_points
        figure.text(
            cursor_x,
            cursor_y,
            piece,
            fontproperties=prop,
            fontsize=size,
            color=INK,
            ha='left',
            va='center',
            zorder=4,
        )
        cursor_x += token_width
        following = runs[index + 1][0] if index + 1 < len(runs) else ''
        if following and following[0] not in ',.;: ':
            cursor_x += 1.5 / figure_points
    return cursor_y - line_gap


def _draw_guide(figure, caption: str) -> None:
    semibold = load_poppins('semibold')
    regular = load_poppins('regular')
    card_x, card_y, card_w, card_h = 0.075, 0.055, 0.85, 0.152
    figure.add_artist(
        FancyBboxPatch(
            (card_x, card_y),
            card_w,
            card_h,
            boxstyle='round,pad=0,rounding_size=0.012',
            mutation_aspect=figure.get_figheight() / figure.get_figwidth(),
            transform=figure.transFigure,
            facecolor=WHITE,
            edgecolor=LINE,
            linewidth=0.9,
            clip_on=False,
            zorder=2,
        )
    )
    bar_w = 3.4 / (figure.get_figwidth() * 72)
    figure.add_artist(
        Rectangle(
            (card_x + 0.012, card_y + 0.014),
            bar_w,
            card_h - 0.028,
            transform=figure.transFigure,
            facecolor=SKY,
            edgecolor='none',
            linewidth=0,
            clip_on=False,
            zorder=3,
        )
    )
    text_x = card_x + 0.034
    figure.text(
        text_x,
        card_y + card_h - 0.022,
        'How to read this chart',
        fontproperties=semibold,
        fontsize=11,
        color=INK,
        ha='left',
        va='center',
        zorder=4,
    )
    for index, line in enumerate(HOW_TO_READ):
        figure.text(
            text_x,
            card_y + card_h - 0.046 - index * 0.022,
            line,
            fontproperties=regular,
            fontsize=8.3,
            color=INK,
            ha='left',
            va='center',
            zorder=4,
        )
    _draw_caption(figure, caption, 0.075, 0.038, 0.78)


def _draw_wordmark(figure) -> None:
    prop = load_poppins('extrabold')
    text = 'HARKNESS TRAINING'
    size = 7.6
    width = _text_width(text, prop, size)
    figure_width = figure.get_figwidth() * 72
    x_right = 0.925
    x_left = x_right - width / figure_width
    figure.text(
        x_left,
        0.022,
        text,
        fontproperties=prop,
        fontsize=size,
        color=NAVY,
        ha='left',
        va='bottom',
        zorder=4,
    )
    bar_width = min(width * 0.38, 78) / figure_width
    figure.add_artist(
        Rectangle(
            (x_left, 0.014),
            bar_width,
            2.4 / (figure.get_figheight() * 72),
            transform=figure.transFigure,
            facecolor=SKY,
            edgecolor='none',
            linewidth=0,
            clip_on=False,
            zorder=4,
        )
    )


def _draw_header(figure, title: str) -> None:
    _tracked(figure, 0.075, 0.972, KICKER, load_poppins('semibold'), 8.2, SKY, 1.15)
    figure.text(
        0.075,
        0.942,
        title,
        fontproperties=load_poppins('semibold'),
        fontsize=21,
        color=INK,
        ha='left',
        va='center',
    )
    figure.text(
        0.075,
        0.912,
        SUBTITLE,
        fontproperties=load_poppins('regular'),
        fontsize=10.5,
        color=MUTED,
        ha='left',
        va='center',
    )
    figure.add_artist(
        Rectangle(
            (0.075, 0.892),
            0.85,
            1.15 / (figure.get_figheight() * 72),
            transform=figure.transFigure,
            facecolor=LINE,
            edgecolor='none',
            linewidth=0,
            clip_on=False,
            zorder=2,
        )
    )


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
    colors = [_color_for(position) for position in range(count)]
    strong = strongest_pair_indexes(directed)
    slices = []
    for position, (low, high) in enumerate(spans):
        outgoing = [
            (other, directed[position][other]) for other in range(count) if directed[position][other] > 0
        ]
        outgoing.sort(key=lambda item: names[item[0]])
        slices.append(_ribbon_slices(low, high, outgoing, arcs[position]))

    landings: list[dict[int, float]] = []
    for target, (low, high) in enumerate(spans):
        sources = [source for source in range(count) if directed[source][target] > 0]
        spots: dict[int, float] = {}
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

    figure_width, figure_height = 10.0, 11.0
    figure = plt.figure(figsize=(figure_width, figure_height), facecolor=LIGHT)
    # Square axes in inches, tight around the circle, so the chord is about two thirds of the page.
    axes_inches = 7.05
    axis = figure.add_axes(
        [
            (figure_width - axes_inches) / 2 / figure_width,
            2.45 / figure_height,
            axes_inches / figure_width,
            axes_inches / figure_height,
        ]
    )
    axis.set_aspect('equal')
    axis.axis('off')
    axis.set_xlim(-1.06, 1.06)
    axis.set_ylim(-1.06, 1.06)
    axis.set_facecolor('none')
    outer = 1.0
    thickness = 0.082
    inner = outer - thickness

    drawings = []
    for source in range(count):
        for target in range(count):
            weight = directed[source][target]
            if weight <= 0 or target not in slices[source] or source not in landings[target]:
                continue
            pair = tuple(sorted((source, target)))
            emphasized = pair in strong
            drawings.append((2 if emphasized else 1, source, target, emphasized))
    drawings.sort(key=lambda item: item[0])

    for rank, source, target, emphasized in drawings:
        alpha = 0.82 if emphasized else 0.45
        axis.add_patch(
            PathPatch(
                _directed_ribbon(slices[source][target], landings[target][source], inner),
                facecolor=colors[source],
                edgecolor=WHITE,
                linewidth=0.8,
                alpha=alpha,
                joinstyle='round',
                capstyle='round',
                zorder=1 + rank,
            )
        )

    for position, (low, high) in enumerate(spans):
        _add_arc(axis, low, high, outer, thickness, colors[position])
    _draw_labels(axis, names, arcs, spans, roles, colors)
    _draw_header(figure, title)
    _draw_guide(figure, chord_caption(names, directed))
    _draw_wordmark(figure)

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, facecolor=LIGHT)
    plt.close(figure)
    return names, ribbon, arcs
