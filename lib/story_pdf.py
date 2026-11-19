"""Draw the personal and group data stories in the HTS palette."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use('Agg', force=False)

import matplotlib.pyplot as plt

from lib.chord_diagram import load_poppins

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

    text = lead_sentence(_plain(text))
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


def _save_chart(figure, folder: Path, name: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    figure.savefig(path, dpi=140, facecolor=LIGHT)
    plt.close(figure)
    return path


def _talk_share(story: dict, folder: Path) -> Path:
    """Ranked talk time. This person is navy, everyone else is gray, with an even-share line."""
    share = max(0, min(100, float(story.get('share') or 0)))
    roster = list(story.get('roster_shares') or [])
    name = _display_name(story.get('name') or 'You')
    if not any(item[0] == name for item in roster):
        roster.append((name, share))
    roster.sort(key=lambda item: item[1])
    count = max(1, len(roster))
    even = 100 / count
    figure, axis = plt.subplots(figsize=(5.4, max(1.6, 0.38 * count + 0.5)), facecolor=LIGHT)
    axis.set_facecolor(LIGHT)
    labels = [item[0] for item in roster]
    values = [item[1] for item in roster]
    colors = [NAVY if label == name else '#C5CDD8' for label in labels]
    bars = axis.barh(labels, values, color=colors, height=0.55)
    axis.axvline(even, color=SKY, linestyle='--', linewidth=1)
    axis.bar_label(
        bars,
        labels=[str(int(round(value))) for value in values],
        padding=3,
        color=INK,
        fontproperties=load_poppins('regular'),
        fontsize=8,
    )
    axis.get_xaxis().set_visible(False)
    axis.spines['bottom'].set_visible(False)
    finding = (
        f'{name} spoke more than an even share of {even:.0f}%.'
        if share >= even
        else f'{name} spoke less than an even share of {even:.0f}%.'
    )
    axis.set_title(_wrap_title(finding), color=INK, fontproperties=load_poppins('semibold'), fontsize=9, loc='left')
    axis.tick_params(colors=INK, labelsize=8)
    for label in axis.get_xticklabels() + axis.get_yticklabels():
        label.set_fontproperties(load_poppins('regular'))
    for spine in axis.spines.values():
        spine.set_color(LINE)
    figure.tight_layout()
    return _save_chart(figure, folder, 'talk-share.png')


def _wrap_title(title: str, width: int = 32) -> str:
    import textwrap

    return '\n'.join(textwrap.wrap(title, width=width)) or title


def _ranked_bars(pairs, folder: Path, filename: str, title: str) -> Path | None:
    """Top bar is navy with a bold label. The other bars are gray."""
    rows = [(str(name), float(value or 0)) for name, value in (pairs or []) if str(name) != 'Average']
    if not rows:
        return None
    ranked = sorted(rows, key=lambda item: (-item[1], item[0]))
    ordered = list(reversed(ranked))
    top_value = ranked[0][1]
    labels = [name for name, _value in ordered]
    values = [value for _name, value in ordered]
    colors = [NAVY if value == top_value else '#C5CDD8' for value in values]
    figure, axis = plt.subplots(figsize=(5.4, max(1.8, 0.42 * len(labels) + 0.7)), facecolor=LIGHT)
    axis.set_facecolor(LIGHT)
    bars = axis.barh(labels, values, color=colors, height=0.55)
    axis.bar_label(
        bars,
        labels=[str(int(value)) for value in values],
        padding=3,
        color=INK,
        fontproperties=load_poppins('regular'),
        fontsize=8,
    )
    axis.get_xaxis().set_visible(False)
    axis.spines['bottom'].set_visible(False)
    axis.set_title(_wrap_title(title), color=INK, fontproperties=load_poppins('semibold'), fontsize=11, loc='left')
    axis.tick_params(colors=INK, labelsize=8)
    for tick in axis.get_yticklabels():
        weight = 'bold' if float(dict(ordered).get(tick.get_text(), -1)) == top_value else 'regular'
        tick.set_fontproperties(load_poppins(weight))
    for tick in axis.get_xticklabels():
        tick.set_fontproperties(load_poppins('regular'))
    for spine in axis.spines.values():
        spine.set_color(LINE)
    figure.tight_layout()
    return _save_chart(figure, folder, filename)


def _leaders(pairs) -> list[str]:
    ranked = sorted(
        [(str(name), float(value or 0)) for name, value in (pairs or [])],
        key=lambda item: (-item[1], item[0]),
    )
    if not ranked or ranked[0][1] <= 0:
        return []
    top = ranked[0][1]
    return [name for name, value in ranked if value == top]


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
    figure, axis = plt.subplots(figsize=(5.4, 1.6), facecolor=LIGHT)
    axis.set_facecolor(LIGHT)
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
    figure, axis = plt.subplots(figsize=(5.2, 1.2), facecolor=LIGHT)
    axis.set_facecolor(LIGHT)
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


def _image(pdf, path: Path | None, width: float = 0.72) -> None:
    if path is None or not Path(path).is_file():
        _flush_heading(pdf)
        return
    if getattr(pdf, '_pending_heading', None):
        _break_before(pdf, 46)
        _flush_heading(pdf)
    else:
        _room(pdf, 32)
    pdf.image(str(path), x='C', w=pdf.epw * width)
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
    headline = str(story.get('headline') or 'You spoke.')
    pdf.set_font('Poppins', 'b', 16)
    pdf.multi_cell(w=0, h=pdf.font_size * 1.2, text=headline, align='L')
    pdf.ln(1)
    if story.get('short_mode'):
        count = int(story.get('turns') or 0)
        if count == 1:
            spoken = 'You spoke once.'
        elif count > 1:
            spoken = f'You spoke {count} times.'
        else:
            spoken = headline
        _body(pdf, spoken)
        _image(pdf, _talk_share(story, chart_dir))
        if story.get('said_quote'):
            _pull_quote(pdf, str(story['said_quote']))
        _heading(pdf, 'Your next rep')
        _body(pdf, story.get('next_rep') or story.get('one_thing') or '')
    else:
        _heading(pdf, '1 big thing')
        _body(pdf, story.get('one_big_thing') or story.get('drama') or '')
        _heading(pdf, 'Why it matters')
        _body(pdf, story.get('why_it_matters') or story.get('goal') or '')
        _heading(pdf, 'By the numbers')
        for line in (story.get('numbers') or [])[:3]:
            _body(pdf, line)
        _image(pdf, _talk_share(story, chart_dir))
        _heading(pdf, 'What you said')
        if story.get('said_quote'):
            _pull_quote(pdf, str(story['said_quote']))
        _heading(pdf, 'Yes, but')
        _body(pdf, story.get('yes_but') or story.get('challenge') or '')
        _heading(pdf, 'Your next rep')
        _body(pdf, story.get('next_rep') or story.get('one_thing') or '')
    used_turns: set[str] = set()
    card = story.get('signature') or {}
    # Keep the signature label with the lines under it, and leave a gap after Drama.
    if _remaining(pdf) < 24:
        pdf.add_page()
    else:
        pdf.ln(3)
    _body(pdf, 'Your signature phrase:')
    if card.get('phrase'):
        from lib.report_stats import display_bigram

        _body(pdf, f'You said "{display_bigram(str(card["phrase"]))}" {card.get("count") or 0} times.')
    shown_examples = 0
    for quote in card.get('quotes') or []:
        if shown_examples >= 2:
            break
        fresh = _keep_fresh(str(quote), used_turns)
        if not fresh:
            continue
        _pull_quote(pdf, fresh)
        shown_examples += 1
    _image(pdf, _phrase_chart(story, chart_dir), width=0.55)
    for label, key in (('What?', 'what'), ('So, what?', 'so_what'), ('Now, what?', 'now_what')):
        if card.get(key):
            _body(pdf, f'{label} {card[key]}')
    if story.get('handed_to'):
        _body(pdf, story['handed_to'])
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
    """The diagram sits under the headline. It stays on this page, scaled to leave room below."""
    top = _display_name(_top_exchange(pairs))
    finding = (
        f'Most exchanges ran through {top}.'
        if top
        else 'Who spoke right after whom. A wider ribbon means those two followed each other more often.'
    )
    _body(pdf, finding)
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
    headline = str(story.get('headline') or '')
    from lib.story_render import fit_headline

    headline = fit_headline(headline, 'A few people talked most.')
    pdf.set_font('Poppins', 'b', 16)
    pdf.multi_cell(w=0, h=pdf.font_size * 1.2, text=headline, align='L')
    pdf.ln(1)
    if hasattr(pdf, 'note_y'):
        pdf.note_y()
    _chord_hero(pdf, chord_path, chord_pairs)
    from lib.story_render import narrative_text

    _heading(pdf, '1 big thing')
    _body(pdf, narrative_text(story['character'], big=True))
    _heading(pdf, 'By the numbers')
    for line in story.get('numbers') or []:
        _body(pdf, line)
    folder = Path(chord_path).parent if chord_path else Path('.')
    shares = story.get('shares') or []
    if shares:
        talkers = _join_names(_leaders(shares))
        _image(
            pdf,
            _ranked_bars(
                shares,
                folder,
                'who-talked.png',
                f'{talkers} talked the most.' if talkers else 'No one talked.',
            ),
        )
    questions = story.get('question_counts') or []
    if questions:
        from lib.questions import QUESTION_FOOTNOTE

        askers = _join_names(_leaders(questions))
        _image(
            pdf,
            _ranked_bars(
                questions,
                folder,
                'who-asked.png',
                f'{askers} asked the most questions.' if askers else 'No one asked a question.',
            ),
        )
        _footnote(pdf, QUESTION_FOOTNOTE)
    # The proposal chart is only the bars. No finding sentence and no footnote.
    proposals = story.get('proposal_counts')
    if proposals:
        _image(
            pdf,
            _ranked_bars(proposals, folder, 'ideas.png', 'Ideas put forward'),
        )
    if hasattr(pdf, 'note_y'):
        pdf.note_y()
    _heading(pdf, 'Why it matters')
    _body(pdf, narrative_text(story.get('pattern_why') or story['goal']))
    _heading(pdf, 'Yes, but')
    _body(pdf, narrative_text(story.get('caveat') or story['challenge']))
    _heading(pdf, 'Your next rep')
    _body(pdf, narrative_text(story['one_thing']))
    if story.get('why'):
        _body(pdf, narrative_text(story['why']))
    _flush_heading(pdf)
