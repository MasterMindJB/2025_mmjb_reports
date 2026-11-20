"""Draw the personal and group data stories in the HTS palette."""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use('Agg', force=False)

import matplotlib.pyplot as plt

from lib.chord_diagram import load_poppins
from lib.scorecard import round_half_up

NAVY = '#1C2C5B'
INK = '#14214A'
SKY = '#6CABDD'
MUTED = '#55628A'
LINE = '#D8E3F0'
LIGHT = '#F4F8FC'


def _room(pdf, needed: float = 36) -> None:
    if pdf.h - pdf.get_y() - pdf.b_margin < needed:
        pdf.add_page()


def _remaining(pdf) -> float:
    return pdf.h - pdf.get_y() - pdf.b_margin


def _break_before(pdf, needed: float) -> None:
    """Start a new page only when this page cannot hold the block and a new one can."""
    room = pdf.h - pdf.t_margin - pdf.b_margin
    if _remaining(pdf) < needed and _remaining(pdf) < room - 1:
        pdf.add_page()


def _kicker(pdf, text: str) -> None:
    pdf.set_text_color(108, 171, 221)
    pdf.set_font('Poppins', 'b', 9)
    pdf.cell(w=0, h=5, text=text.upper(), ln=True)
    pdf.set_text_color(20, 33, 74)


def _heading(pdf, text: str) -> None:
    """Queue a heading so it is drawn on the same page as the block that follows."""
    _flush_heading(pdf)
    pdf._pending_heading = text


def _flush_heading(pdf, following: float = 0) -> None:
    text = getattr(pdf, '_pending_heading', None)
    if not text:
        return
    pdf._pending_heading = None
    _break_before(pdf, 14 + following)
    pdf._heading_page = pdf.page
    pdf.ln(2)
    pdf.set_x(pdf.l_margin)
    pdf.set_font('Poppins', 'b', 14)
    pdf.set_text_color(20, 33, 74)
    pdf.cell(w=0, h=8, text=text, ln=True)
    if hasattr(pdf, 'note_y'):
        pdf.note_y()


def _paragraph_height(pdf, text: str, size: int) -> float:
    pdf.set_font('Poppins', '', size)
    width = max(pdf.epw, 1)
    lines = 0
    for paragraph in (text or '').split('\n'):
        words = paragraph.split()
        if not words:
            lines += 1
            continue
        current = ''
        lines += 1
        for word in words:
            trial = f'{current} {word}'.strip()
            if pdf.get_string_width(trial) <= width:
                current = trial
            else:
                lines += 1
                current = word
    return max(lines, 1) * pdf.font_size * 1.4


def _gap(pdf) -> float:
    return float(getattr(pdf, '_story_gap', 1.0) or 1.0)


def _plain(text: str) -> str:
    """Customer text does not say bigram or phatic marker."""
    text = re.sub(r'\bphatic markers\b', 'filler words', text or '', flags=re.I)
    text = re.sub(r'\bphatic marker\b', 'filler word', text, flags=re.I)
    text = re.sub(r'\bbigrams\b', 'phrases', text, flags=re.I)
    text = re.sub(r'\bbigram\b', 'phrase', text, flags=re.I)
    return text


def _clocks(text: str) -> list[str]:
    return re.findall(r'\b\d+:\d{2}(?::\d{2})?\b', text or '')


def _keep_fresh(text: str, used: set[str]) -> str:
    """Drop sentences whose timestamp was already used in this report."""
    if not text:
        return ''
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    kept = []
    for sentence in sentences:
        stamps = _clocks(sentence)
        if stamps and any(stamp in used for stamp in stamps):
            continue
        kept.append(sentence)
        used.update(stamps)
    return ' '.join(kept)


def _body(pdf, text: str, size: int = 11) -> None:
    from lib.story_render import lead_sentence

    from lib.story_render import close_quotes

    text = _strip_control_ids(close_quotes(lead_sentence(_plain(text))))
    if size == 11:
        size = int(getattr(pdf, '_body_size', 11) or 11)
    if not text:
        _flush_heading(pdf)
        return
    height = _paragraph_height(pdf, text, size)
    if getattr(pdf, '_pending_heading', None):
        _break_before(pdf, min(height, 48) + 14)
        _flush_heading(pdf)
    elif _remaining(pdf) < height:
        _break_before(pdf, height)
    pdf.set_x(pdf.l_margin)
    pdf.set_font('Poppins', '', size)
    pdf.set_text_color(20, 33, 74)
    pdf.multi_cell(w=0, h=pdf.font_size * 1.4, text=text, align='L')
    pdf.ln(1 * _gap(pdf))
    if hasattr(pdf, 'note_y'):
        pdf.note_y()


def _pull_quote(pdf, text: str, when: str = '') -> None:
    """An indented quote under its heading. At most 40 words, on a sentence boundary."""
    from lib.data_story import sentence_around
    from lib.story_render import bounded_quote, lead_sentence

    if getattr(pdf, '_pending_heading', None):
        _flush_heading(pdf)
    bounded = bounded_quote(sentence_around(text, text))
    if not bounded:
        return
    text = lead_sentence(_plain(bounded))
    if not text:
        return
    from lib.story_render import close_quotes

    inner = text.strip().strip('"')
    text = close_quotes(f'"{inner}"')
    pdf.set_font('Poppins', '', 11)
    pdf.set_text_color(20, 33, 74)
    indent = 8
    pdf.set_x(pdf.l_margin + indent)
    pdf.multi_cell(w=pdf.epw - indent, h=pdf.font_size * 1.4, text=text, align='L')
    if when:
        pdf.set_x(pdf.l_margin + indent)
        pdf.set_text_color(85, 98, 138)
        pdf.multi_cell(w=pdf.epw - indent, h=pdf.font_size * 1.4, text=when, align='L')
        pdf.set_text_color(20, 33, 74)
    pdf.ln(1 * _gap(pdf))
    if hasattr(pdf, 'note_y'):
        pdf.note_y()


def _transition(pdf, text: str) -> None:
    if not text:
        _flush_heading(pdf)
        return
    if getattr(pdf, '_pending_heading', None):
        _break_before(pdf, 24)
        _flush_heading(pdf)
    pdf.set_x(pdf.l_margin)
    pdf.set_font('Poppins', 'i', 11)
    pdf.set_text_color(85, 98, 138)
    pdf.multi_cell(w=0, text=text, align='L')
    pdf.ln(1)
    pdf.set_text_color(20, 33, 74)


def _callout(pdf, title: str, text: str) -> None:
    if getattr(pdf, '_pending_heading', None):
        _break_before(pdf, 56)
        _flush_heading(pdf)
    else:
        _room(pdf, 40)
    pdf.set_x(pdf.l_margin)
    pdf.set_fill_color(244, 248, 252)
    pdf.set_font('Poppins', 'b', 16)
    pdf.set_text_color(28, 44, 91)
    pdf.multi_cell(w=0, h=8, text=title, fill=True, align='L')
    pdf.set_x(pdf.l_margin)
    pdf.set_font('Poppins', '', 12)
    pdf.set_text_color(20, 33, 74)
    pdf.multi_cell(w=0, text=text, fill=True, align='L')
    pdf.ln(2)


_OVERLAPS: list[str] = []


def _note_overlaps(figure, name: str) -> None:
    """Record text boxes that land on each other. Tests read this list."""
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    from matplotlib.transforms import Bbox

    boxes = []
    for artist in figure.findobj(lambda item: hasattr(item, 'get_text') and callable(item.get_text)):
        if not str(artist.get_text()).strip() or not artist.get_visible():
            continue
        box = artist.get_window_extent(renderer)
        if box.width < 4 or box.height < 4:
            continue
        boxes.append(box)
    for index, box in enumerate(boxes):
        for other in boxes[index + 1:]:
            shared = Bbox.intersection(box, other)
            if shared is not None and shared.width > 2 and shared.height > 2:
                _OVERLAPS.append(name)
                return


def _save_chart(figure, folder: Path, name: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    figure.savefig(path, dpi=140, facecolor='white')
    plt.close(figure)
    return path


def _plain_axes(axis) -> None:
    """No panel fill, no tick marks, no spines."""
    axis.set_facecolor('white')
    axis.tick_params(length=0, colors=INK, labelsize=8)
    for spine in axis.spines.values():
        spine.set_visible(False)


def _talk_title(share: float, even: float) -> str:
    if share > even + 2:
        return 'You spoke more than an even share.'
    if share + 2 < even:
        return 'You spoke less than an even share.'
    return 'You spoke near an even share.'


def _talk_share(story: dict, folder: Path) -> Path:
    """Ranked talk time. You is navy, everyone else is gray, with a labeled even-share line."""
    share = max(0, min(100, float(story.get('share') or 0)))
    roster = list(story.get('roster_shares') or [])
    name = _display_name(story.get('name') or 'You')
    if not any(_same_person(item[0], name) for item in roster):
        roster.append((name, share))
    roster.sort(key=lambda item: item[1])
    count = max(1, len(roster))
    even = 100 / count
    figure, axis = plt.subplots(figsize=(5.4, max(1.6, 0.38 * count + 0.6)), facecolor='white')
    _plain_axes(axis)
    labels = ['You' if _same_person(item[0], name) else item[0] for item in roster]
    values = [item[1] for item in roster]
    colors = [NAVY if label == 'You' else '#C5CDD8' for label in labels]
    bars = axis.barh(labels, values, color=colors, height=0.55)
    axis.axvline(even, color=SKY, linestyle='--', linewidth=1)
    axis.set_ylim(-0.7, len(labels) - 0.4)
    axis.text(
        even, -0.55, f'Even share {even:.0f}%',
        color=SKY, fontsize=8, fontproperties=load_poppins('regular'), ha='center', va='top', clip_on=False,
    )
    axis.bar_label(
        bars,
        labels=[f'{round_half_up(value)}%' for value in values],
        padding=3,
        color=INK,
        fontproperties=load_poppins('regular'),
        fontsize=8,
    )
    axis.get_xaxis().set_visible(False)
    finding = _talk_title(share, even)
    axis.set_title(_wrap_title(finding, 42), color=INK, fontproperties=load_poppins('semibold'), fontsize=11, loc='left')
    for label in axis.get_yticklabels():
        label.set_fontproperties(load_poppins('regular'))
    figure.tight_layout()
    _note_overlaps(figure, 'talk-share.png')
    _remember(folder, 'talk', 'horizontal_bar', [round_half_up(value) for value in values])
    return _save_chart(figure, folder, 'talk-share.png')


def _wrap_title(title: str, width: int = 32) -> str:
    import textwrap

    return '\n'.join(textwrap.wrap(title, width=width)) or title


def _ranked_bars(pairs, folder: Path, filename: str, title: str, percent: bool = False) -> Path | None:
    """Top bar is navy. The other bars are gray. No panel and no tick marks."""
    rows = [(str(name), float(value or 0)) for name, value in (pairs or []) if str(name) != 'Average']
    if not rows:
        return None
    ranked = sorted(rows, key=lambda item: (-item[1], item[0]))
    ordered = list(reversed(ranked))
    top_value = ranked[0][1]
    labels = [name for name, _value in ordered]
    values = [value for _name, value in ordered]
    colors = [NAVY if value == top_value and top_value > 0 else '#C5CDD8' for value in values]
    figure, axis = plt.subplots(figsize=(5.4, max(1.8, 0.42 * len(labels) + 0.7)), facecolor='white')
    _plain_axes(axis)
    bars = axis.barh(labels, values, color=colors, height=0.55)
    axis.bar_label(
        bars,
        labels=[f'{int(value)}%' if percent else str(int(value)) for value in values],
        padding=3,
        color=INK,
        fontproperties=load_poppins('regular'),
        fontsize=8,
    )
    axis.get_xaxis().set_visible(False)
    axis.set_title(_wrap_title(title, 42), color=INK, fontproperties=load_poppins('semibold'), fontsize=11, loc='left')
    for tick in axis.get_yticklabels():
        weight = 'bold' if float(dict(ordered).get(tick.get_text(), -1)) == top_value else 'regular'
        tick.set_fontproperties(load_poppins(weight))
    figure.tight_layout()
    _remember(folder, 'talk-ranked' if 'talked' in filename else filename, 'horizontal_bar', [int(value) for value in values])
    return _save_chart(figure, folder, filename)


def _remember(folder: Path, key: str, kind: str, values: list[int], labels: list[str] | None = None, value_labels: list[str] | None = None) -> None:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'chart-types.json'
    data = {}
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
        except json.JSONDecodeError:
            data = {}
    if not isinstance(data, dict):
        data = {}
    entry = {'type': kind, 'values': [int(value) for value in values]}
    if labels is not None:
        entry['labels'] = list(labels)
    if value_labels is not None:
        entry['value_labels'] = list(value_labels)
    data[key] = entry
    path.write_text(json.dumps(data, indent=2), encoding='utf-8')


def chart_types(folder: Path) -> dict:
    path = Path(folder) / 'chart-types.json'
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding='utf-8'))
    return data if isinstance(data, dict) else {}


def _same_person(left: str, right: str) -> bool:
    return _display_name(left).casefold() == _display_name(right).casefold()


def _pie_slices(pairs, limit: int = 6) -> list[tuple[str, int]]:
    """Drop zero counts. More than about six slices fold the tail into Others."""
    rows = [
        (str(name), int(value or 0))
        for name, value in (pairs or [])
        if str(name) != 'Average' and int(value or 0) > 0
    ]
    rows.sort(key=lambda item: (-item[1], item[0]))
    if len(rows) <= limit:
        return rows
    head = rows[: limit - 1]
    rest = sum(value for _name, value in rows[limit - 1:])
    if rest:
        head.append(('Others', rest))
    return head


def _column_label(name: str, highlight: str = '') -> str:
    """First name, Moderator, or Speaker F. The reader is You."""
    text = str(name or '').strip()
    if highlight and (_same_person(text, highlight) or text == 'You'):
        return 'You'
    if re.search(r'\b(moderator|host|facilitator)\b', text, re.I):
        return 'Moderator'
    match = re.match(r'(?i)^speaker\s+([A-Za-z0-9]+)$', text)
    if match:
        token = match.group(1)
        return f'Speaker {token.upper() if len(token) == 1 else token}'
    match = re.match(r'(?i)^aud\.?\s*member\s+(\d+)$', text)
    if match:
        return f'Aud. member {match.group(1)}'
    return text.split()[0] if text else 'Speaker'


def _unique_column_labels(names: list[str], highlight: str = '') -> list[str]:
    """Labels that do not collapse two people into one word. Speaker F wraps."""
    raw = [_column_label(name, highlight) for name in names]
    counts: dict[str, int] = {}
    for label in raw:
        counts[label] = counts.get(label, 0) + 1
    labels = []
    for name, label in zip(names, raw):
        if counts[label] > 1 and label not in {'You', 'Moderator'} and not label.startswith('Speaker'):
            parts = str(name).split()
            label = parts[0] if len(parts) == 1 else f'{parts[0]} {parts[-1][:1]}'
        if label.startswith('Speaker '):
            label = label.replace(' ', '\n', 1)
        labels.append(label)
    return labels


def _complete_pairs(pairs, roster) -> list[tuple[str, int]]:
    """Every person in the roster gets a column, including a zero."""
    values: dict[str, int] = {}
    order: list[str] = []
    for name, _value in roster or []:
        key = str(name)
        if key == 'Average' or key in values:
            continue
        values[key] = 0
        order.append(key)
    for name, value in pairs or []:
        key = str(name)
        if key == 'Average':
            continue
        if key not in values:
            order.append(key)
        values[key] = int(value or 0)
    return [(name, values[name]) for name in order]


def _vertical_bars(pairs, folder: Path, filename: str, title: str, highlight: str = '', roster=None) -> Path | None:
    """Columns. The subject, or every tied leader, is navy. A zero has no value label."""
    rows = _complete_pairs(pairs, roster)
    if not rows:
        return None
    ranked = sorted(rows, key=lambda item: (-item[1], item[0]))
    labels = _unique_column_labels([name for name, _value in ranked], highlight)
    values = [value for _name, value in ranked]
    top = max(values) if values else 0
    if highlight:
        colors = [NAVY if label == 'You' else '#C5CDD8' for label in labels]
    else:
        colors = [NAVY if value == top and top > 0 else '#C5CDD8' for value in values]
    figure, axis = plt.subplots(figsize=(max(5.4, 0.72 * len(labels)), 3.4), facecolor='white')
    _plain_axes(axis)
    bars = axis.bar(labels, values, color=colors, width=0.62)
    shown = ['' if int(value) == 0 else str(int(value)) for value in values]
    axis.bar_label(
        bars,
        labels=shown,
        padding=2,
        color=INK,
        fontproperties=load_poppins('regular'),
        fontsize=8,
    )
    for bar, value in zip(bars, values):
        if int(value) == 0:
            axis.text(
                bar.get_x() + bar.get_width() / 2, 0, '0',
                ha='center', va='bottom', color=INK,
                fontproperties=load_poppins('regular'), fontsize=8,
            )
    axis.set_ylim(0, (max(values) or 1) * 1.45)
    axis.get_yaxis().set_visible(False)
    for tick in axis.get_xticklabels():
        tick.set_fontproperties(load_poppins('regular'))
        tick.set_rotation(0)
        tick.set_ha('center')
    axis.set_title(_wrap_title(title, 42), color=INK, fontproperties=load_poppins('semibold'), fontsize=11, loc='left')
    figure.tight_layout()
    _note_overlaps(figure, filename)
    _remember(folder, filename, 'vertical_bar', values, labels, [label if label else '0' for label in shown])
    return _save_chart(figure, folder, filename)


def _lollipop(pairs, folder: Path, filename: str, title: str, highlight: str = '') -> Path | None:
    """A dot on a stem. Not a bar."""
    rows = [(str(name), int(value or 0)) for name, value in (pairs or []) if str(name) != 'Average']
    if not rows:
        return None
    ranked = sorted(rows, key=lambda item: (item[1], item[0]))
    labels = ['You' if highlight and _same_person(name, highlight) else name for name, _value in ranked]
    values = [value for _name, value in ranked]
    top = max(values) if values else 0
    figure, axis = plt.subplots(figsize=(5.4, max(1.8, 0.38 * len(labels) + 0.6)), facecolor='white')
    _plain_axes(axis)
    axis.hlines(range(len(labels)), 0, values, color='#C5CDD8', linewidth=1.5)
    colors = []
    for label, value in zip(labels, values):
        if highlight and label == 'You':
            colors.append(NAVY)
        elif not highlight and value == top and top > 0:
            colors.append(NAVY)
        else:
            colors.append('#C5CDD8')
    axis.scatter(values, range(len(labels)), color=colors, s=36, zorder=3)
    ceiling = max(values) or 1
    pad = ceiling * 0.08 + 1
    axis.set_xlim(0, ceiling + pad * 3)
    for index, value in enumerate(values):
        axis.text(value + pad, index, str(int(value)), va='center', ha='left', color=INK, fontproperties=load_poppins('regular'), fontsize=8)
    axis.set_yticks(range(len(labels)))
    axis.set_yticklabels(labels)
    axis.get_xaxis().set_visible(False)
    for tick in axis.get_yticklabels():
        tick.set_fontproperties(load_poppins('regular'))
    axis.set_title(_wrap_title(title, 42), color=INK, fontproperties=load_poppins('semibold'), fontsize=11, loc='left')
    figure.tight_layout()
    _note_overlaps(figure, filename)
    _remember(folder, filename, 'lollipop', values)
    return _save_chart(figure, folder, filename)


def _donut(pairs, folder: Path, filename: str, title: str, highlight: str = '', center: str = '') -> Path | None:
    """A donut. The key slice starts at 12 o'clock. Zero slices are left out."""
    slices = _pie_slices(pairs)
    if not slices:
        return None
    labels = [f'{name} {value}' for name, value in slices]
    values = [value for _name, value in slices]
    top = max(values)
    colors = []
    for name, value in slices:
        if highlight and (_same_person(name, highlight) or name == highlight):
            colors.append(NAVY)
        elif not highlight and value == top:
            colors.append(NAVY)
        else:
            colors.append('#C5CDD8')
    figure, axis = plt.subplots(figsize=(5.2, 3.4), facecolor='white')
    axis.set_facecolor('white')
    axis.pie(
        values,
        labels=labels,
        colors=colors,
        startangle=90,
        wedgeprops={'width': 0.42, 'edgecolor': 'white'},
        textprops={'fontproperties': load_poppins('regular'), 'fontsize': 8, 'color': INK},
    )
    if center:
        axis.text(0, 0, center, ha='center', va='center', color=NAVY, fontproperties=load_poppins('semibold'), fontsize=12)
    axis.set_title(_wrap_title(title, 42), color=INK, fontproperties=load_poppins('semibold'), fontsize=11, loc='left')
    figure.tight_layout()
    _remember(folder, filename, 'donut', values)
    return _save_chart(figure, folder, filename)


def _big_number(you: int, top: int, folder: Path, filename: str, title: str, left: str = 'You', right: str = 'Room top') -> Path:
    """One tile. The personal page says You. The group page says the person's name."""
    figure, axis = plt.subplots(figsize=(5.2, 2.2), facecolor='white')
    axis.set_facecolor('white')
    axis.axis('off')
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_title(_wrap_title(title, 42), color=INK, fontproperties=load_poppins('semibold'), fontsize=11, loc='left')
    axis.text(0.28 if right else 0.5, 0.48, str(int(you)), ha='center', va='center', color=NAVY, fontproperties=load_poppins('semibold'), fontsize=28)
    axis.text(0.28 if right else 0.5, 0.16, left, ha='center', va='center', color=INK, fontproperties=load_poppins('regular'), fontsize=10)
    if right:
        axis.text(0.72, 0.48, str(int(top)), ha='center', va='center', color='#8B95A8', fontproperties=load_poppins('semibold'), fontsize=28)
        axis.text(0.72, 0.16, right, ha='center', va='center', color=INK, fontproperties=load_poppins('regular'), fontsize=10)
    figure.tight_layout()
    _remember(folder, filename, 'big_number', [int(you), int(top)])
    return _save_chart(figure, folder, filename)


def _question_share(counts, subject: str) -> list[tuple[str, int]]:
    you = 0
    others = 0
    for name, value in counts or []:
        if _same_person(str(name), subject):
            you += int(value or 0)
        else:
            others += int(value or 0)
    return [('You', you), ('Everyone else', others)]


def _strip_control_ids(text: str) -> str:
    cleaned = re.sub(r'\b[QAU]\d+\b', '', str(text or ''))
    return re.sub(r'\s{2,}', ' ', cleaned).strip()


def _without_next_time(text: str) -> str:
    """The only countable action lives in Your next rep."""
    kept = []
    for sentence in re.split(r'(?<=[.!?])\s+', str(text or '').strip()):
        if sentence and not re.match(r'(?i)next time\b', sentence.strip()):
            kept.append(sentence.strip())
    return ' '.join(kept)


def _finding_line(story: dict) -> str:
    """Twelve words or fewer, with no verdict words."""
    share = float(story.get('share') or 0)
    roster = story.get('roster_shares') or []
    even = 100 / max(1, len(roster) or 1)
    if share > even + 2:
        line = 'You spoke more than an even share.'
    elif share + 2 < even:
        line = 'You spoke less than an even share.'
    else:
        line = 'You spoke near an even share.'
    words = line.replace('.', '').split()
    banned = {'too', 'bad'}
    words = [word for word in words if word.lower() not in banned]
    return ' '.join(words[:12]) + '.'


def _group_tiles(story: dict) -> list[tuple[str, str]]:
    """People, turn passing, questions, and ideas. Grades use this meeting only."""
    shares = story.get('shares') or []
    spoke = sum(1 for _name, value in shares if float(value or 0) > 0)
    total = len(shares) or spoke
    questions = sum(int(value or 0) for _name, value in (story.get('question_counts') or []))
    ideas = sum(int(value or 0) for _name, value in (story.get('proposal_counts') or []))
    numbers = ' '.join(story.get('numbers') or [])
    grade = 'This meeting'
    if 'below an even mix' in numbers:
        grade = 'Below an even mix'
    elif 'at an even mix' in numbers:
        grade = 'At an even mix'
    elif 'near an even mix' in numbers:
        grade = 'Near an even mix'
    moved = ''
    import re as _re
    found = _re.search(r'(\d+)% of turns moved to someone new, (\w+) an even mix of (\d+)%', numbers)
    if found:
        moved = f'{found.group(1)}%'
        grade = f'{found.group(2).capitalize()} an even mix ({found.group(3)}%)'
    else:
        moved = '—'
    return [
        ('People who spoke', f'{spoke} of {total}', ''),
        ('Turns that moved to someone new', moved, grade if found else ''),
        ('Questions', str(questions), ''),
        ('Ideas', str(ideas), _idea_grade(story)),
    ]


def _idea_grade(story: dict) -> str:
    """One idea names the person. Several ideas need no extra line."""
    proposed = [
        (str(name), int(value or 0))
        for name, value in (story.get('proposal_counts') or [])
        if int(value or 0) > 0
    ]
    if len(proposed) == 1:
        return f'from {proposed[0][0]}'
    return ''


def _number_lines(story: dict) -> list[str]:
    """Drop lines the tiles already show. An empty list omits the section."""
    kept = []
    for line in story.get('numbers') or []:
        low = str(line).lower()
        if 'moved to someone new' in low or 'people spoke' in low or 'people were in the room' in low:
            continue
        kept.append(str(line))
    return kept


def _personal_tiles(story: dict) -> list[tuple[str, str]]:
    """Four numbers, each graded against this meeting only."""
    share = int(round(float(story.get('share') or 0)))
    roster = story.get('roster_shares') or [(story.get('name') or 'You', share)]
    even = int(round(100 / max(1, len(roster))))
    if share > even + 2:
        share_grade = f'Above even share ({even}%)'
    elif share + 2 < even:
        share_grade = f'Below even share ({even}%)'
    else:
        share_grade = f'Near even share ({even}%)'
    questions = story.get('question_counts') or []
    you_q = 0
    for name, value in questions:
        if _same_person(str(name), str(story.get('name') or '')):
            you_q = int(value or 0)
    others = [int(value or 0) for name, value in questions if not _same_person(str(name), str(story.get('name') or ''))]
    typical = int(round(sum(others) / len(others))) if others else you_q
    if you_q > typical:
        question_grade = f'Above the room ({typical})'
    elif you_q < typical:
        question_grade = f'Below the room ({typical})'
    else:
        question_grade = f'Same as the room ({typical})'
    turns = story.get('longest_turns') or []
    you_turn = 0
    for name, value in turns:
        if _same_person(str(name), str(story.get('name') or '')):
            you_turn = int(value or 0)
    other_turns = [int(value or 0) for name, value in turns if not _same_person(str(name), str(story.get('name') or ''))]
    middle = sorted(other_turns)[len(other_turns) // 2] if other_turns else you_turn
    if you_turn > middle:
        turn_grade = f'Above the room middle ({middle})'
    elif you_turn < middle:
        turn_grade = f'Below the room middle ({middle})'
    else:
        turn_grade = f'Same as the room middle ({middle})'
    ideas = story.get('proposal_counts') or []
    you_ideas = 0
    for name, value in ideas:
        if _same_person(str(name), str(story.get('name') or '')):
            you_ideas = int(value or 0)
    top = max((int(value or 0) for _name, value in ideas), default=you_ideas)
    if you_ideas >= top and top > 0:
        idea_grade = f'Matches the room top ({top})'
    else:
        idea_grade = f'Below the room top ({top})'
    return [
        ('Talk share', f'{share}%', share_grade),
        ('Questions', str(you_q), question_grade),
        ('Longest turn (words)', str(you_turn), turn_grade),
        ('Ideas', str(you_ideas), idea_grade),
    ]


def _score_tiles(pdf, tiles: list[tuple[str, str, str]]) -> None:
    """A gray label, a navy number, and one gray grade. No traffic-light colors."""
    if not tiles:
        return
    _flush_heading(pdf)
    width = pdf.epw / len(tiles)
    y = pdf.get_y()
    for index, (label, number, grade) in enumerate(tiles):
        x = pdf.l_margin + index * width
        pdf.set_xy(x, y)
        pdf.set_text_color(120, 128, 142)
        pdf.set_font('Poppins', '', 8)
        pdf.multi_cell(w=width - 1, h=3.4, text=str(label), align='L')
        pdf.set_xy(x, y + 8)
        pdf.set_text_color(28, 44, 91)
        pdf.set_font('Poppins', 'b', 22)
        pdf.cell(w=width - 1, h=9, text=str(number), align='L')
        pdf.set_xy(x, y + 17)
        pdf.set_text_color(120, 128, 142)
        pdf.set_font('Poppins', '', 8)
        pdf.multi_cell(w=width - 1, h=3.4, text=str(grade), align='L')
    pdf.set_y(y + 30)
    pdf.set_text_color(20, 33, 74)


def _anchored(text: str, action: str) -> str:
    """Keep a real section, and end it on the one action. An empty section stays dropped."""
    body = str(text or '').strip()
    if not body:
        return ''
    for banned in ('here is your personal report', 'you spoke for too long'):
        if banned in body.lower():
            body = re.sub(re.escape(banned), '', body, flags=re.I).strip(' .')
    if not body:
        return action
    if action and action.lower() not in body.lower():
        return f'{body.rstrip()} {action}'
    return body


def _personal_charts(pdf, story: dict, folder: Path) -> None:
    """Four different chart types. A 0% or 100% question share stays a tile, not a donut."""
    subject = str(story.get('name') or 'You')
    talk = _talk_share(story, folder)
    questions = story.get('question_counts') or []
    columns = None
    donut = None
    if questions:
        askers = _join_names(_subject_names(_leaders(questions), subject))
        columns = _vertical_bars(
            questions, folder, 'who-asked.png',
            f'{askers} asked the most questions.' if askers else 'No one asked a question.',
            subject,
            roster=story.get('roster_shares') or story.get('shares'),
        )
        share = _question_share(questions, subject)
        you, others = share[0][1], share[1][1]
        total = you + others
        if total and 0 < you < total:
            donut = _donut(
                share, folder, 'question-share.png',
                f'You asked {you} of {total} questions.',
                'You',
                center=f'{you} of {total}',
            )
    turns = story.get('longest_turns') or story.get('word_counts') or []
    lollipop = None
    if turns:
        leaders = _join_names(_subject_names(_leaders(turns), subject))
        verb = 'had the longest turn.' if story.get('longest_turns') else 'used the most words.'
        lollipop = _lollipop(
            turns, folder, 'words.png',
            f'{leaders} {verb}' if leaders else 'No one spoke.',
            subject,
        )
    ideas = story.get('proposal_counts') or []
    tile = None
    if ideas:
        you_ideas = next((int(value) for name, value in ideas if _same_person(str(name), subject)), 0)
        top = max((int(value or 0) for _name, value in ideas), default=0)
        tile = _big_number(you_ideas, top, folder, 'ideas-you.png', 'Ideas put forward')
    _image_row(pdf, talk, columns)
    if donut:
        _image_row(pdf, donut, lollipop)
        if tile:
            _image(pdf, tile)
    elif lollipop:
        # No question donut. Keep the lollipop on this page at the paired width.
        tile_fits = bool(tile) and _remaining(pdf) >= 62
        _paired_row(pdf, lollipop, tile if tile_fits else None)
        if tile and not tile_fits:
            _image(pdf, tile)


def _leaders(pairs) -> list[str]:
    ranked = sorted(
        [(str(name), float(value or 0)) for name, value in (pairs or [])],
        key=lambda item: (-item[1], item[0]),
    )
    if not ranked or ranked[0][1] <= 0:
        return []
    top = ranked[0][1]
    return [name for name, value in ranked if value == top]


def _subject_names(names: list[str], highlight: str = '') -> list[str]:
    """The person this report is about is You, or you after another name."""
    shown = []
    for index, name in enumerate(names):
        if highlight and (_same_person(name, highlight) or str(name) == 'You'):
            shown.append('You' if index == 0 else 'you')
        else:
            shown.append(name)
    return shown


def _join_names(names: list[str]) -> str:
    if not names:
        return ''
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f'{names[0]} and {names[1]}'
    return ', '.join(names[:-1]) + f', and {names[-1]}'


def _footnote(pdf, text: str) -> None:
    pdf.set_x(pdf.l_margin)
    pdf.set_font('Poppins', '', 9)
    pdf.set_text_color(85, 98, 138)
    pdf.multi_cell(w=0, h=pdf.font_size * 1.3, text=text, align='L')
    pdf.ln(1)
    pdf.set_text_color(20, 33, 74)
    if hasattr(pdf, 'note_y'):
        pdf.note_y()


def _clock_label(seconds: float) -> str:
    from lib.story_evidence import format_clock_seconds

    return format_clock_seconds(int(seconds))


def timeline_bounds(story: dict) -> tuple[float, float]:
    """Left edge is the window start, not 0:00, when the panel begins later."""
    lines = story.get('lines') or []
    start = float(story.get('session_start_ms') or 0) / 1000
    end = float(story.get('session_end_ms') or 0) / 1000
    if lines:
        if start <= 0:
            start = min(float(row.get('start') or 0) for row in lines) / 1000
        if end <= 0:
            end = max(float(row.get('end') or 0) for row in lines) / 1000
    if end <= start:
        end = start + 1
    return start, end


def _timeline(story: dict, folder: Path) -> Path | None:
    lines = story.get('lines') or []
    if not lines:
        return None
    session_start, session_end = timeline_bounds(story)
    figure, axis = plt.subplots(figsize=(5.4, 1.6), facecolor='white')
    axis.set_facecolor('white')
    for row in lines:
        start = float(row.get('start') or 0) / 1000
        end = float(row.get('end') or start) / 1000
        axis.barh(0, max(end - start, 0.2), left=start, height=0.45, color=NAVY)
    axis.set_xlim(session_start, session_end)
    axis.set_xticks([session_start, session_end])
    axis.set_xticklabels([_clock_label(session_start), _clock_label(session_end)])
    axis.set_yticks([0])
    axis.set_yticklabels(['You'])
    axis.set_xlabel('When you spoke', color=MUTED, fontproperties=load_poppins('regular'), fontsize=8)
    axis.tick_params(colors=INK, labelsize=8)
    for spine in ('top', 'right'):
        axis.spines[spine].set_visible(False)
    axis.spines['left'].set_color(LINE)
    axis.spines['bottom'].set_color(LINE)
    figure.tight_layout()
    return _save_chart(figure, folder, 'turns.png')


def _phrase_chart(story: dict, folder: Path) -> Path | None:
    phrase = story['signature'].get('phrase')
    count = story['signature'].get('count') or 0
    if not phrase:
        return None
    figure, axis = plt.subplots(figsize=(5.2, 1.2), facecolor='white')
    axis.set_facecolor('white')
    axis.barh([phrase], [count], color=SKY, height=0.45)
    axis.tick_params(colors=INK, labelsize=8)
    for label in axis.get_yticklabels():
        label.set_fontproperties(load_poppins('medium'))
    for spine in ('top', 'right'):
        axis.spines[spine].set_visible(False)
    axis.spines['left'].set_color(LINE)
    axis.spines['bottom'].set_color(LINE)
    figure.tight_layout()
    return _save_chart(figure, folder, 'phrase.png')


def _paired_row(pdf, left: Path | None, right: Path | None = None) -> None:
    """One or two charts at the paired-row width, kept on the current page."""
    paths = [path for path in (left, right) if path and Path(path).is_file()]
    if not paths:
        return
    _flush_heading(pdf)
    previous = pdf.auto_page_break
    margin = pdf.b_margin
    pdf.set_auto_page_break(False)
    y = pdf.get_y()
    width = pdf.epw * 0.48
    pdf.image(str(paths[0]), x=pdf.l_margin, y=y, w=width)
    if len(paths) > 1:
        pdf.image(str(paths[1]), x=pdf.l_margin + pdf.epw * 0.52, y=y, w=width)
    pdf.set_y(y + 62)
    pdf.set_auto_page_break(previous, margin)
    if hasattr(pdf, 'note_y'):
        pdf.note_y()


def _image_row(pdf, left: Path | None, right: Path | None) -> None:
    """Two charts on one row when both exist. One chart uses the full width."""
    paths = [path for path in (left, right) if path and Path(path).is_file()]
    if not paths:
        return
    _flush_heading(pdf)
    y = pdf.get_y()
    if len(paths) == 1:
        pdf.image(str(paths[0]), x='C', w=pdf.epw * 0.72)
        pdf.ln(1)
        return
    width = pdf.epw * 0.48
    pdf.image(str(paths[0]), x=pdf.l_margin, y=y, w=width)
    pdf.image(str(paths[1]), x=pdf.l_margin + pdf.epw * 0.52, y=y, w=width)
    pdf.set_y(y + 62)
    if hasattr(pdf, 'note_y'):
        pdf.note_y()


def _image_height(pdf, path: Path, width: float) -> float:
    try:
        from PIL import Image

        with Image.open(path) as image:
            px_w, px_h = image.size
        if px_w and px_h:
            return width * px_h / px_w
    except (OSError, ValueError):
        pass
    return 48


def _image(pdf, path: Path | None, width: float = 0.72) -> None:
    if path is None or not Path(path).is_file():
        _flush_heading(pdf)
        return
    draw_w = pdf.epw * width
    needed = _image_height(pdf, Path(path), draw_w) + 6
    if getattr(pdf, '_pending_heading', None):
        _break_before(pdf, needed + 8)
        _flush_heading(pdf)
    else:
        _break_before(pdf, needed)
    pdf.image(str(path), x='C', w=draw_w)
    pdf.ln(1)
    if hasattr(pdf, 'note_y'):
        pdf.note_y()


APPENDIX_FONT_FLOOR = 8
GAP_FLOOR = 0.7


def _appendix_block(pdf, lines: list[str], size: int, columns: int) -> None:
    _flush_heading(pdf)
    pdf.ln(2 * _gap(pdf))
    pdf.set_x(pdf.l_margin)
    pdf.set_font('Poppins', 'b', 14)
    pdf.set_text_color(20, 33, 74)
    pdf.cell(w=0, h=8, text='Appendix: style lenses', ln=True)
    pdf.set_font('Poppins', '', size)
    pdf.multi_cell(w=0, text='Top result for each lens.', align='L')
    if columns >= 2 and len(lines) > 1:
        col_w = (pdf.epw - 6) / 2
        midpoint = (len(lines) + 1) // 2
        left, right = lines[:midpoint], lines[midpoint:]
        start = pdf.get_y()
        for line in left:
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(w=col_w, text=line, align='L')
        left_end = pdf.get_y()
        pdf.set_y(start)
        for line in right:
            pdf.set_x(pdf.l_margin + col_w + 6)
            pdf.multi_cell(w=col_w, text=line, align='L')
        pdf.set_y(max(left_end, pdf.get_y()))
    else:
        for line in lines:
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(w=0, text=line, align='L')
    if hasattr(pdf, 'note_y'):
        pdf.note_y()


def _display_name(name: str) -> str:
    """Drop a leading honorific so a chart title does not stop at 'Dr.'."""
    parts = str(name or 'You').split()
    honorifics = {'dr', 'prof', 'mr', 'ms', 'mrs', 'professor'}
    while parts and parts[0].lower().rstrip('.') in honorifics:
        parts.pop(0)
    return ' '.join(parts) or str(name or 'You')


def _specific_headline(story: dict) -> str:
    source = f"{story.get('drama') or ''} {story.get('goal') or ''}"
    word = next(
        (token.lower() for token in re.findall(r"[A-Za-z']+", source) if len(token) > 5),
        'point',
    )
    return f'You name {word}.'


def _short_headline(share: int, role: str) -> str:
    role = (role or '').lower()
    if role in {'facilitator', 'host', 'moderator'}:
        return 'You carried the room. Share it.'
    if int(share) >= 20:
        return 'You held the floor. Hand it.'
    return 'You set the point. Name it.'


def _bullet(pdf, text: str) -> None:
    """One hanging bullet. The three parts stay in a single paragraph."""
    if not text:
        return
    _flush_heading(pdf)
    pdf.set_x(pdf.l_margin)
    pdf.set_font('Poppins', '', getattr(pdf, '_body_size', 11))
    pdf.set_text_color(20, 33, 74)
    pdf.multi_cell(w=0, h=pdf.font_size * 1.35, text=f'• {text}', align='L')
    pdf.ln(1)


def _observed(text: str, observation_only: bool) -> str:
    """Drop a prescriptive sentence when the next-rep flag is off."""
    if not observation_only:
        return _without_next_time(text)
    from lib.personal_narrative import advice_problems

    kept = []
    for sentence in re.split(r'(?<=[.!?])\s+', str(text or '').strip()):
        if sentence and not advice_problems(sentence):
            kept.append(sentence.strip())
    return ' '.join(kept)


def _metric_box(pdf, metrics: list[dict]) -> None:
    """One thin box. A short name, the number, and one gray grade. No fill."""
    metrics = list(metrics or [])[:4]
    if not metrics:
        return
    _flush_heading(pdf)
    height = 24
    if _remaining(pdf) < height + 2:
        pdf.add_page()
    y = pdf.get_y()
    pdf.set_draw_color(197, 205, 216)
    pdf.set_line_width(0.2)
    pdf.rect(pdf.l_margin, y, pdf.epw, height)
    width = pdf.epw / len(metrics)
    for index, metric in enumerate(metrics):
        x = pdf.l_margin + 1.2 + index * width
        pdf.set_xy(x, y + 1.2)
        pdf.set_text_color(120, 128, 142)
        pdf.set_font('Poppins', '', 7)
        pdf.multi_cell(w=width - 2, h=3, text=str(metric.get('label') or ''), align='L')
        pdf.set_xy(x, y + 7.5)
        pdf.set_text_color(28, 44, 91)
        pdf.set_font('Poppins', 'b', 14)
        pdf.cell(w=width - 2, h=6, text=str(metric.get('number') or ''), align='L')
        pdf.set_xy(x, y + 14)
        pdf.set_text_color(120, 128, 142)
        pdf.set_font('Poppins', '', 7)
        pdf.multi_cell(w=width - 2, h=3, text=str(metric.get('grade') or ''), align='L')
    pdf.set_y(y + height + 1)
    pdf.set_text_color(20, 33, 74)


def _clear_footer(pdf) -> None:
    """Keep the last line above the footer, which is drawn at 15mm."""
    pdf.set_auto_page_break(auto=True, margin=18)


def write_personal_report(
    pdf,
    story: dict,
    chart_dir: Path,
    *,
    gap: float = 1.0,
    appendix_font: int = 10,
    appendix_first: bool = False,
    columns: int = 1,
    body_size: int = 11,
) -> None:
    pdf._story_gap = max(0.5, float(gap))
    pdf._body_size = max(9, int(body_size))
    pdf._hts_story = True
    appendix_font = max(APPENDIX_FONT_FLOOR, int(appendix_font))
    _clear_footer(pdf)
    pdf.add_page()
    pdf.set_font('Poppins', '', 11)
    pdf.set_text_color(20, 33, 74)
    from lib.names import header_name, is_placeholder_name, report_header

    if 'header_name' in story:
        header = report_header(story.get('header_name'))
    else:
        candidate = str(story.get('name') or '')
        headline = str(story.get('headline') or '')
        if not candidate and headline and len(headline.split()) <= 3 and not is_placeholder_name(headline):
            candidate = headline
        header = report_header(header_name(fallback=candidate))
    pdf.set_text_color(28, 44, 91)
    pdf.set_font('Poppins', 'b', 22)
    pdf.multi_cell(w=0, h=10, text=header, align='L')
    pdf.ln(1)
    pdf.set_text_color(20, 33, 74)
    from lib.personal_narrative import personal_next_rep_enabled
    from lib.scorecard import personal_metrics

    observe = not personal_next_rep_enabled()
    bullets = [line for line in (story.get('bullets') or []) if line]
    if bullets:
        for line in bullets:
            _bullet(pdf, _observed(line, observe))
    else:
        character = _observed(story.get('character') or '', observe)
        goal = str(story.get('goal') or '')
        if not goal.startswith('It seemed like you were trying to '):
            goal = ''
        challenge = _observed(story.get('challenge') or '', observe)
        for line in (character, goal, challenge):
            if line:
                _body(pdf, line)
    dropped = set(story.get('fallback_sections') or [])
    _heading(pdf, 'By the numbers')
    _metric_box(pdf, story.get('scorecard') or personal_metrics(story))
    _heading(pdf, 'Your charts')
    _personal_charts(pdf, story, chart_dir)
    _heading(pdf, 'What you said')
    if story.get('said_quote'):
        _pull_quote(pdf, str(story['said_quote']))
    caveat_text = '' if 'yes_but' in dropped else (story.get('yes_but') or '')
    caveat_text = _observed(caveat_text, observe)
    if caveat_text:
        _heading(pdf, 'Yes, but')
        _body(pdf, caveat_text)
    if personal_next_rep_enabled():
        action = str(story.get('action') or story.get('next_rep') or story.get('one_thing') or '')
        _heading(pdf, 'Your next rep')
        _body(pdf, action)
    appendix = [f'{name}: {line}' for name, line in (story.get('appendix') or {}).items()]
    _appendix_block(pdf, appendix, appendix_font, columns)


def fit_personal_report(story: dict, chart_dir: Path, make_pdf):
    """Re-render when the last page is thin or One Thing is left on a third page.

    Tighter spacing and a smaller body come first. Text is wrapped, never cut.
    """
    plans = (
        {'gap': 1.0, 'appendix_font': 10, 'appendix_first': False, 'columns': 1, 'body_size': 11},
        {'gap': 0.75, 'appendix_font': 9, 'appendix_first': False, 'columns': 1, 'body_size': 10},
        {'gap': GAP_FLOOR, 'appendix_font': APPENDIX_FONT_FLOOR, 'appendix_first': True, 'columns': 1, 'body_size': 10},
        {'gap': GAP_FLOOR, 'appendix_font': APPENDIX_FONT_FLOOR, 'appendix_first': False, 'columns': 2, 'body_size': 9},
        {'gap': 0.5, 'appendix_font': APPENDIX_FONT_FLOOR, 'appendix_first': True, 'columns': 2, 'body_size': 9},
    )
    chosen = None
    for plan in plans:
        pdf = make_pdf()
        write_personal_report(pdf, story, chart_dir, **plan)
        ratios = pdf.page_fill_ratios()
        chosen = pdf
        pages = len(ratios)
        thin = pages > 1 and ratios[-1] < 0.15
        if pages <= 2 and not thin:
            break
    return chosen


def _top_exchange(pairs) -> str:
    scores: dict[str, int] = {}
    for left, right, forward, back, *rest in pairs or []:
        weight = int(forward or 0) + int(back or 0)
        scores[str(left)] = scores.get(str(left), 0) + weight
        scores[str(right)] = scores.get(str(right), 0) + weight
    if not scores:
        return ''
    return max(scores, key=lambda name: (scores[name], name))


def _chord_hero(pdf, chord_path, pairs=None) -> None:
    """The diagram sits under the headline. Its own title already states the finding."""
    del pairs
    pdf._chord_page = pdf.page
    if not chord_path or not Path(chord_path).is_file():
        return
    width = pdf.epw * 0.92
    try:
        from PIL import Image

        with Image.open(chord_path) as image:
            px_w, px_h = image.size
        if px_w and px_h:
            height = width * px_h / px_w
            cap = max(120, min(160, _remaining(pdf) - 48))
            if height > cap:
                width = width * cap / height
    except (OSError, ValueError):
        pass
    pdf.image(str(chord_path), x='C', w=width)
    pdf.ln(1)
    if hasattr(pdf, 'note_y'):
        pdf.note_y()


def write_group_story(pdf, story: dict, chord_path, chord_pairs, *, gap: float = 1.0) -> None:
    pdf._story_gap = gap
    pdf._hts_story = True
    _clear_footer(pdf)
    pdf.add_page()
    pdf.set_text_color(20, 33, 74)
    from lib.story_render import group_page_headline

    headline = group_page_headline(story.get('character') or '', str(story.get('headline') or ''))
    pdf.set_font('Poppins', 'b', 16)
    pdf.multi_cell(w=0, h=pdf.font_size * 1.2, text=headline, align='L')
    pdf.ln(1)
    _score_tiles(pdf, _group_tiles(story))
    if hasattr(pdf, 'note_y'):
        pdf.note_y()
    from lib.story_render import narrative_text

    _heading(pdf, '1 big thing')
    _body(pdf, _without_next_time(story.get('character') or ''))
    number_lines = _number_lines(story)
    if number_lines:
        _heading(pdf, 'By the numbers')
        for line in number_lines:
            _body(pdf, line)
    folder = Path(chord_path).parent if chord_path else Path('.')
    shares = story.get('shares') or []
    questions = story.get('question_counts') or []
    talk_path = None
    question_path = None
    if shares:
        talkers = _join_names(_leaders(shares))
        talk_path = _ranked_bars(
            shares,
            folder,
            'who-talked.png',
            f'{talkers} talked the most.' if talkers else 'No one talked.',
            percent=True,
        )
    if questions:
        askers = _join_names(_leaders(questions))
        question_path = _vertical_bars(
            questions,
            folder,
            'who-asked.png',
            f'{askers} asked the most questions.' if askers else 'No one asked a question.',
            roster=shares,
        )
    if talk_path and question_path and _remaining(pdf) >= 70:
        _image_row(pdf, talk_path, question_path)
    else:
        _image(pdf, talk_path)
        _image(pdf, question_path)
    if questions:
        from lib.questions import QUESTION_FOOTNOTE

        _footnote(pdf, QUESTION_FOOTNOTE)
    proposals = story.get('proposal_counts') or []
    proposed = [(name, int(value or 0)) for name, value in proposals if int(value or 0) > 0]
    if len(proposed) >= 2:
        _image(pdf, _donut(proposals, folder, 'ideas.png', 'Ideas put forward'))
    words = story.get('word_counts') or []
    if words:
        leaders = _join_names(_leaders(words))
        _image(
            pdf,
            _lollipop(
                words,
                folder,
                'words.png',
                f'{leaders} used the most words.' if leaders else 'No one spoke.',
            ),
        )
    if hasattr(pdf, 'note_y'):
        pdf.note_y()
    why = _without_next_time(narrative_text(story.get('pattern_why') or story.get('goal') or ''))
    if why and 'you can change this next time' not in why.lower():
        _heading(pdf, 'Why it matters')
        _body(pdf, why)
    caveat = _without_next_time(story.get('caveat') or story.get('challenge') or '')
    if caveat and 'you can change this next time' not in caveat.lower():
        _heading(pdf, 'Yes, but')
        _body(pdf, caveat)
    _heading(pdf, 'Your next rep')
    _body(pdf, story.get('one_thing') or '')
    pdf.add_page()
    _heading(pdf, 'Appendix')
    _flush_heading(pdf)
    title_file = Path(str(chord_path)).with_suffix('.title') if chord_path else None
    if title_file and title_file.is_file():
        _body(pdf, title_file.read_text(encoding='utf-8').strip())
    _remember(folder, 'structure', 'chord', [])
    _chord_hero(pdf, chord_path, chord_pairs)
