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
    from lib.names import short_name

    label = short_name(name)
    if sum(short_name(other) == label for other in names) == 1:
        return label
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


def visible_chord_pairs(names: list[str], directed: list[list[int]]) -> list[tuple]:
    """Every pair the diagram draws, including a single exchange on a group of seven or fewer.

    Larger groups hide flows under 3 percent of the turns. The table uses the same cutoff.
    """
    total = sum(sum(row) for row in directed)
    fade = 0.03 * total if len(names) > 7 else 0
    shown = []
    for left, right, forward, back, weight in pair_directions(names, directed):
        if weight > 0 and weight + 1e-9 >= fade:
            shown.append((left, right, forward, back))
    return shown


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


def _clockwise_from(origin: float, angle: float) -> float:
    return (origin - angle) % (2 * math.pi)


def _allocate_arc_ends(
    spans: list[tuple[float, float]], directed: list[list[int]]
) -> list[dict[tuple[str, int], tuple[float, float]]]:
    """Subdivide each arc into the directed flows that touch that person.

    Outgoing and incoming segments are ordered by the other person's angle so
    neighboring ribbons sit next to each other instead of crossing.
    """
    allocated = []
    for person, (low, high) in enumerate(spans):
        flows = []
        for other, (other_low, other_high) in enumerate(spans):
            if other == person:
                continue
            other_mid = (other_low + other_high) / 2
            if directed[person][other] > 0:
                flows.append((other_mid, 0, other, 'out', directed[person][other]))
            if directed[other][person] > 0:
                flows.append((other_mid, 1, other, 'in', directed[other][person]))
        flows.sort(key=lambda item: (_clockwise_from(high, item[0]), item[1]))
        total = sum(item[4] for item in flows) or 1
        span = high - low
        cursor = high
        slots: dict[tuple[str, int], tuple[float, float]] = {}
        for _mid, _order, other, kind, weight in flows:
            width = span * (weight / total)
            slots[(kind, other)] = (cursor - width, cursor)
            cursor -= width
        allocated.append(slots)
    return allocated


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


def _target_span(source: tuple[float, float], incoming: tuple[float, float]) -> tuple[float, float]:
    """A narrower segment centered on the incoming slot, still on that arc."""
    source_width = max(source[1] - source[0], 1e-4)
    slot_low, slot_high = incoming
    slot_width = max(slot_high - slot_low, 1e-4)
    desired = source_width * 0.42
    if desired < 0.01:
        desired = min(source_width, slot_width)
    width = min(desired, slot_width)
    center = (slot_low + slot_high) / 2
    return (center - width / 2, center + width / 2)


def _directed_ribbon(
    source: tuple[float, float],
    target: tuple[float, float] | float,
    radius: float,
) -> MplPath:
    """A chord band. Both ends lie on the inner radius; the sides bow inward.

    ``target`` may be a span ``(low, high)`` or a single angle, which becomes a
    short span. Neighboring arcs get a shallower bow so the band stays near the rim.
    """
    source_low, source_high = sorted(source)
    if isinstance(target, tuple):
        target_low, target_high = sorted(target)
    else:
        half = max((source_high - source_low) * 0.21, 0.012)
        target_low, target_high = target - half, target + half
    source_mid = (source_low + source_high) / 2
    target_mid = (target_low + target_high) / 2
    separation = _angular_separation(source_mid, target_mid)
    # One shared control keeps the band from pinching. Neighbors stay shallower.
    pull = 0.72 if separation < 0.9 else 0.30
    delta = (target_mid - source_mid + math.pi) % (2 * math.pi) - math.pi
    shared = _polar(source_mid + delta / 2, radius * pull)

    def add_arc(vertices, codes, start: float, end: float) -> None:
        points = _arc_points(start, end, radius, steps=18)
        if end < start:
            points.reverse()
        for point in points[1:]:
            vertices.append(point)
            codes.append(MplPath.LINETO)

    def add_quadratic(vertices, codes, end_angle: float) -> None:
        vertices.append(shared)
        vertices.append(_polar(end_angle, radius))
        codes.extend([MplPath.CURVE3, MplPath.CURVE3])

    # Connecting low-to-low can twist the band into a bowtie. Swap ends when the chords cross.
    if _chords_cross(source_low, target_low, source_high, target_high):
        first_target, second_target = target_high, target_low
    else:
        first_target, second_target = target_low, target_high
    vertices = [_polar(source_high, radius)]
    codes = [MplPath.MOVETO]
    add_arc(vertices, codes, source_high, source_low)
    add_quadratic(vertices, codes, first_target)
    add_arc(vertices, codes, first_target, second_target)
    add_quadratic(vertices, codes, source_high)
    vertices.append(vertices[0])
    codes.append(MplPath.CLOSEPOLY)
    return MplPath(vertices, codes)


def _chords_cross(a0: float, b0: float, a1: float, b1: float) -> bool:
    def orient(p, q, r) -> float:
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    start_a, end_a = _polar(a0, 1.0), _polar(b0, 1.0)
    start_b, end_b = _polar(a1, 1.0), _polar(b1, 1.0)
    return (
        orient(start_a, end_a, start_b) * orient(start_a, end_a, end_b) < 0
        and orient(start_b, end_b, start_a) * orient(start_b, end_b, end_a) < 0
    )


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


def _horizontal_alignment(angle: float) -> tuple[str, str]:
    """Horizontal label alignment. The anchor sits outside the arc at one radius."""
    normalized = math.atan2(math.sin(angle), math.cos(angle))
    if math.sin(normalized) > 0.78:
        return 'center', 'bottom'
    if math.sin(normalized) < -0.78:
        return 'center', 'top'
    if math.cos(normalized) >= 0:
        return 'left', 'center'
    return 'right', 'center'


def _label_box(item: dict) -> tuple[float, float, float, float]:
    """Display-space box: left, bottom, right, top."""
    width = item['width']
    height = item['height']
    px, py = item['px'], item['py']
    if item['ha'] == 'left':
        left, right = px, px + width
    elif item['ha'] == 'right':
        left, right = px - width, px
    else:
        left, right = px - width / 2, px + width / 2
    if item['va'] == 'bottom':
        bottom, top = py, py + height
    elif item['va'] == 'top':
        bottom, top = py - height, py
    else:
        bottom, top = py - height / 2, py + height / 2
    return left, bottom, right, top


def _separate_labels(axis, placed: list[dict]) -> None:
    """Nudge overlapping labels apart vertically, then keep them outside the circle."""
    if len(placed) < 2:
        return
    axis.figure.canvas.draw()
    transform = axis.transData
    inverse = transform.inverted()
    for item in placed:
        item['px'], item['py'] = transform.transform((item['x'], item['y']))
        item['home_y'] = item['py']
    for _ in range(16):
        placed.sort(key=lambda item: item['py'])
        moved = False
        for earlier, later in zip(placed, placed[1:]):
            left_a, bottom_a, right_a, top_a = _label_box(earlier)
            left_b, bottom_b, right_b, top_b = _label_box(later)
            if right_a < left_b - 4 or right_b < left_a - 4:
                continue
            overlap_y = min(top_a, top_b) - max(bottom_a, bottom_b)
            overlap_x = min(right_a, right_b) - max(left_a, left_b)
            if overlap_y <= 2 or overlap_x <= 2:
                continue
            # Split on the shorter overlap so neighbors move just far enough to clear.
            if overlap_x < overlap_y:
                shift = overlap_x / 2 + 4
                if earlier['px'] <= later['px']:
                    earlier['px'] -= shift
                    later['px'] += shift
                else:
                    earlier['px'] += shift
                    later['px'] -= shift
            else:
                shift = overlap_y / 2 + 4
                # A label under the circle hangs toward the legend. Nudge the other one up.
                if earlier['va'] != 'top':
                    earlier['py'] -= shift
                later['py'] += shift
            moved = True
        if not moved:
            break
    for item in placed:
        data_x, data_y = inverse.transform((item['px'], item['py']))
        # A vertical nudge must not drop the label back onto the ring.
        radius = math.hypot(data_x, data_y)
        if radius < 1.14:
            scale = 1.14 / max(radius, 1e-6)
            data_x *= scale
            data_y *= scale
        item['x'], item['y'] = float(data_x), float(data_y)


def _draw_labels(axis, names, arcs, spans, roles, colors) -> None:
    count = len(names)
    name_size = 10.0 if count <= 8 else 8.4
    count_size = 7.4 if count <= 8 else 6.6
    role_size = 6.1 if count <= 8 else 5.5
    name_font = load_poppins('medium')
    count_font = load_poppins('regular')
    role_font = load_poppins('semibold')
    label_radius = 1.12 if count <= 8 else 1.22
    pill = {
        'boxstyle': 'round,pad=0.22',
        'facecolor': SKY,
        'edgecolor': SKY,
        'linewidth': 0.4,
    }
    placed = []
    for position, (low, high) in enumerate(spans):
        angle = (low + high) / 2
        align, vertical = _horizontal_alignment(angle)
        # Bottom labels hang into the gap under the circle. The others sit further
        # out so a second line cannot fall back onto the ring.
        normalized = math.atan2(math.sin(angle), math.cos(angle))
        if vertical == 'top':
            radius = 1.06
        elif vertical == 'bottom':
            radius = 1.10 if count <= 8 else 1.05
        elif count > 7 and math.sin(normalized) > 0.2:
            radius = max(label_radius, 1.28)
        else:
            radius = label_radius
        # Neighbors on the same ring collide. Step every other label out a little.
        # Labels under the circle already hang toward the legend, so they stay put.
        if count >= 6 and position % 2 == 1 and vertical != 'top':
            radius += 0.10
        anchor = _polar(angle, radius)
        name = names[position]
        role = roles.get(name, 'participant')
        has_role = role in {'facilitator', 'actor'}
        width = max(
            _text_width(name, name_font, name_size),
            _text_width(f'{arcs[position]} exchanges', count_font, count_size),
        )
        height = name_size + count_size + (role_size + 10 if has_role else 0) + 16
        placed.append(
            {
                'x': anchor[0],
                'y': anchor[1],
                'ha': align,
                'va': vertical,
                'width': width * axis.figure.dpi / 72.0,
                'height': height * axis.figure.dpi / 72.0,
                'position': position,
            }
        )
    if count >= 2:
        _separate_labels(axis, placed)
    for item in placed:
        position = item['position']
        name = names[position]
        exchanges = arcs[position]
        exchange_word = 'exchange' if exchanges == 1 else 'exchanges'
        role = roles.get(name, 'participant')
        has_role = role in {'facilitator', 'actor'}
        anchor = (item['x'], item['y'])
        align, vertical = item['ha'], item['va']
        name_artist = axis.text(
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
        name_artist.set_gid(f'chord-label-{position}')
        if vertical == 'bottom':
            first_line = -12
        elif vertical == 'top':
            first_line = -(name_size + 4)
        else:
            first_line = -(name_size * 0.55 + 8)
        # Near the top, a line placed straight under the name falls onto the ring.
        outward_x = 0
        if count > 7 and vertical == 'center' and math.sin(normalized) > 0.2:
            outward_x = 16 if align == 'left' else -16
        if has_role:
            role_artist = axis.annotate(
                'FACILITATOR' if role == 'facilitator' else 'ACTOR',
                xy=anchor,
                xytext=(outward_x, first_line),
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
            role_artist.set_gid(f'chord-label-{position}')
            count_offset = first_line - 16
        else:
            count_offset = first_line
        count_artist = axis.annotate(
            f'{exchanges} {exchange_word}',
            xy=anchor,
            xytext=(outward_x, count_offset),
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
        count_artist.set_gid(f'chord-label-{position}')


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
        runs.append((match.group().lstrip(), 'semibold'))
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
        piece = text.rstrip()
        if piece == '':
            continue
        token_width = _text_width(piece, prop, size) / figure_points
        if cursor_x > x and cursor_x + token_width > limit:
            cursor_x = x
            cursor_y -= line_gap
            piece = piece.lstrip()
            if piece == '':
                continue
            token_width = _text_width(piece, prop, size) / figure_points
        caption_artist = figure.text(
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
        caption_artist.set_gid('chord-caption')
        cursor_x += token_width
        # A trailing space is easy to lose between a bold pair and the next word.
        # Keep the comma tight, then advance one explicit space.
        if piece.endswith((',', ';', ':')):
            cursor_x += max(_text_width(' ', regular, size), size * 0.33) / figure_points
    return cursor_y - line_gap


def _draw_guide(figure, caption: str) -> None:
    semibold = load_poppins('semibold')
    regular = load_poppins('regular')
    card_x, card_y, card_w, card_h = 0.075, 0.040, 0.85, 0.148
    legend = figure.add_artist(
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
    legend.set_gid('chord-legend')
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
    """The mark sits in the header, clear of the legend card at the bottom."""
    prop = load_poppins('extrabold')
    text = 'HARKNESS TRAINING'
    size = 7.6
    width = _text_width(text, prop, size)
    figure_width = figure.get_figwidth() * 72
    x_right = 0.925
    bar_width = min(width * 0.38, 78) / figure_width
    mark = figure.text(
        x_right,
        0.942,
        text,
        fontproperties=prop,
        fontsize=size,
        color=NAVY,
        ha='right',
        va='center',
        zorder=4,
    )
    mark.set_gid('chord-wordmark')
    figure.add_artist(
        Rectangle(
            (x_right - bar_width, 0.928),
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
            (0.075, 0.905),
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


def _window_box(artist, renderer) -> tuple[float, float, float, float]:
    extent = artist.get_window_extent(renderer)
    return (extent.x0, extent.y0, extent.x1, extent.y1)


def _record_layout_boxes(figure, axis, sink: dict) -> None:
    """Pixel boxes for the overlap check. Ribbons and arcs are not included."""
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    grouped: dict[str, list[tuple[float, float, float, float]]] = {}
    for text in list(axis.texts):
        gid = text.get_gid() or ''
        if gid.startswith('chord-label-'):
            grouped.setdefault(gid, []).append(_window_box(text, renderer))
    sink['labels'] = [
        (
            min(box[0] for box in boxes),
            min(box[1] for box in boxes),
            max(box[2] for box in boxes),
            max(box[3] for box in boxes),
        )
        for boxes in grouped.values()
    ]
    sink['canvas'] = (float(renderer.width), float(renderer.height))
    sink['wordmark'] = []
    sink['legend'] = []
    sink['captions'] = []
    for text in list(figure.texts):
        gid = text.get_gid() or ''
        if gid == 'chord-wordmark':
            sink['wordmark'].append(_window_box(text, renderer))
        elif gid == 'chord-caption':
            sink['captions'].append(_window_box(text, renderer))
    for artist in list(figure.artists):
        if artist.get_gid() == 'chord-legend':
            sink['legend'].append(_window_box(artist, renderer))


def save_chord_diagram(
    sequence: list[str],
    path: Path,
    names: list[str] | None = None,
    roles: dict[str, str] | None = None,
    title: str = 'Who talks with whom',
    layout_boxes: dict | None = None,
) -> tuple[list[str], list[list[int]], list[int]]:
    """Draw directed ribbons and return the symmetric totals used for arc size."""
    roles = roles or {}
    names, directed = directed_follows(sequence, names)
    _roster, ribbon, arcs = transition_ribbons(sequence, names)
    count = len(names)
    spans = _arc_layout(arcs)
    colors = [_color_for(position) for position in range(count)]
    strong = strongest_pair_indexes(directed)
    ends = _allocate_arc_ends(spans, directed)

    figure_width, figure_height = 10.0, 11.0
    figure = plt.figure(figsize=(figure_width, figure_height), facecolor=LIGHT)
    # Square axes in inches, tight around the circle, so the chord is about two thirds of the page.
    axes_inches = 7.05
    axis = figure.add_axes(
        [
            (figure_width - axes_inches) / 2 / figure_width,
            2.68 / figure_height,
            axes_inches / figure_width,
            axes_inches / figure_height,
        ]
    )
    axis.set_aspect('equal')
    axis.axis('off')
    # Past eight people the names need margin. A wider view shrinks the circle
    # on the canvas without changing ribbon geometry or colors.
    view = 1.06 if count <= 8 else 1.72
    axis.set_xlim(-view, view)
    axis.set_ylim(-view, view)
    axis.set_facecolor('none')
    axis.set_clip_on(False)
    axis.set_zorder(4)
    outer = 1.0
    thickness = 0.082
    inner = outer - thickness

    total_turns = sum(sum(row) for row in directed) or 1
    fade_below = 0.03 * total_turns if count > 7 else 0
    drawings = []
    weak_by_source: dict[int, list[int]] = {source: [] for source in range(count)}
    for source in range(count):
        for target in range(count):
            weight = directed[source][target]
            source_span = ends[source].get(('out', target))
            incoming_span = ends[target].get(('in', source))
            if weight <= 0 or source_span is None or incoming_span is None:
                continue
            pair = tuple(sorted((source, target)))
            emphasized = pair in strong
            if fade_below and weight < fade_below and not emphasized:
                weak_by_source[source].append(target)
                continue
            alpha = 0.80 if emphasized else 0.45
            drawings.append((2 if emphasized else 1, source, source_span, incoming_span, alpha))
    for source, targets in weak_by_source.items():
        targets.sort(key=lambda target: sum(spans[target]) / 2)
        clusters: list[list[int]] = []
        for target in targets:
            midpoint = sum(spans[target]) / 2
            if clusters and _angular_separation(sum(spans[clusters[-1][-1]]) / 2, midpoint) < 0.8:
                clusters[-1].append(target)
            else:
                clusters.append([target])
        for cluster in clusters:
            source_parts = [ends[source][('out', target)] for target in cluster]
            incoming_parts = [ends[target][('in', source)] for target in cluster]
            source_span = (min(part[0] for part in source_parts), max(part[1] for part in source_parts))
            incoming_span = (min(part[0] for part in incoming_parts), max(part[1] for part in incoming_parts))
            drawings.append((0, source, source_span, incoming_span, 0.20))
    drawings.sort(key=lambda item: item[0])

    for rank, source, source_span, incoming_span, alpha in drawings:
        axis.add_patch(
            PathPatch(
                _directed_ribbon(source_span, _target_span(source_span, incoming_span), inner),
                facecolor=colors[source],
                edgecolor=WHITE,
                linewidth=0.6,
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
    if layout_boxes is not None:
        _record_layout_boxes(figure, axis, layout_boxes)
    figure.savefig(path, dpi=300, facecolor=LIGHT)
    plt.close(figure)
    return names, ribbon, arcs
