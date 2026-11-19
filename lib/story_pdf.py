"""Draw the personal and group data stories in the HTS palette."""

from __future__ import annotations

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
    return max(lines, 1) * pdf.font_size * 1.45


def _gap(pdf) -> float:
    return float(getattr(pdf, '_story_gap', 1.0) or 1.0)


def _body(pdf, text: str, size: int = 11) -> None:
    from lib.story_render import capitalize_sentences

    text = capitalize_sentences(text)
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
    pdf.multi_cell(w=0, text=text, align='L')
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
    figure, axis = plt.subplots(figsize=(5.2, 1.6), facecolor=LIGHT)
    axis.set_facecolor(LIGHT)
    share = max(0, min(100, story['share']))
    axis.barh(['You', 'Others'], [share, 100 - share], color=[NAVY, SKY], height=0.55)
    axis.set_xlim(0, 100)
    axis.tick_params(colors=INK, labelsize=8)
    for label in axis.get_xticklabels() + axis.get_yticklabels():
        label.set_fontproperties(load_poppins('regular'))
    axis.set_xlabel('Share of speaking time', color=MUTED, fontproperties=load_poppins('regular'), fontsize=8)
    for spine in axis.spines.values():
        spine.set_color(LINE)
    figure.tight_layout()
    return _save_chart(figure, folder, 'talk-share.png')


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
    appendix_font = max(APPENDIX_FONT_FLOOR, int(appendix_font))
    _clear_footer(pdf)
    pdf.add_page()
    pdf.set_font('Poppins', '', 11)
    pdf.set_text_color(20, 33, 74)
    _kicker(pdf, 'Your conversation')
    pdf.set_font('Poppins', 'b', 16)
    pdf.multi_cell(w=0, text=story['headline'], align='L')
    pdf.ln(1)
    _heading(pdf, 'Character')
    for line in story.get('numbers') or []:
        _body(pdf, line)
    _body(pdf, story.get('character_body') or '')
    _image(pdf, _talk_share(story, chart_dir))
    _callout(pdf, 'One thing', story['one_thing'])
    _transition(pdf, (story.get('transitions') or {}).get('after_character'))
    _heading(pdf, 'Goal')
    _body(pdf, story['goal'])
    _transition(pdf, (story.get('transitions') or {}).get('after_goal'))
    _heading(pdf, 'Challenge')
    _body(pdf, story['challenge'])
    _image(pdf, _timeline(story, chart_dir))
    _transition(pdf, (story.get('transitions') or {}).get('after_challenge'))
    _heading(pdf, 'Drama')
    _body(pdf, story['drama'])
    card = story['signature']
    # Keep the signature label with the lines under it, and leave a gap after Drama.
    if _remaining(pdf) < 24:
        pdf.add_page()
    else:
        pdf.ln(3)
    _body(pdf, 'Your signature phrase:')
    if card.get('phrase'):
        from lib.report_stats import display_bigram

        _body(pdf, f'You said "{display_bigram(str(card["phrase"]))}" {card.get("count") or 0} times.')
    from lib.story_render import clip_clause

    shown = 0
    for quote in card.get('quotes') or []:
        if shown >= 2:
            break
        _body(pdf, clip_clause(str(quote), 25))
        shown += 1
    _image(pdf, _phrase_chart(story, chart_dir), width=0.55)
    for label, key in (('What?', 'what'), ('So, what?', 'so_what'), ('Now, what?', 'now_what')):
        if card.get(key):
            _body(pdf, f'{label} {card[key]}')
    _transition(pdf, (story.get('transitions') or {}).get('after_drama'))
    _heading(pdf, 'Resolution')
    _body(pdf, story['resolution'])
    _transition(pdf, (story.get('transitions') or {}).get('after_resolution'))
    appendix = [f'{name}: {line}' for name, line in (story.get('appendix') or {}).items()]

    def _one_thing() -> None:
        _heading(pdf, 'One thing')
        _callout(pdf, story['one_thing'], story['why'])
        _flush_heading(pdf)

    if appendix_first:
        _appendix_block(pdf, appendix, appendix_font, columns)
        _one_thing()
    else:
        _one_thing()
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


def _chord_hero(pdf, chord_path) -> None:
    """The diagram sits under the headline. It stays on this page, scaled to leave room below."""
    _body(
        pdf,
        'Who spoke right after whom. A wider ribbon means those two followed each other more often. '
        'Color is the person who spoke first.',
    )
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
            cap = max(70, _remaining(pdf) - 46)
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
    _clear_footer(pdf)
    pdf.add_page()
    pdf.set_text_color(20, 33, 74)
    _kicker(pdf, 'The session')
    pdf.set_font('Poppins', 'b', 16)
    pdf.multi_cell(w=0, text=story['headline'], align='L')
    pdf.ln(1)
    if hasattr(pdf, 'note_y'):
        pdf.note_y()
    _chord_hero(pdf, chord_path)
    _heading(pdf, 'Character')
    _body(pdf, story['character'])
    _transition(pdf, (story.get('transitions') or {}).get('after_character'))
    from lib.pdf_functions import _add_direction_table

    _add_direction_table(pdf, (chord_pairs or [])[:4])
    if hasattr(pdf, 'note_y'):
        pdf.note_y()
    _heading(pdf, 'Goal')
    _body(pdf, story['goal'])
    _heading(pdf, 'Challenge')
    _body(pdf, story['challenge'])
    _transition(pdf, (story.get('transitions') or {}).get('after_challenge'))
    _heading(pdf, 'Drama')
    _body(pdf, story.get('drama') or '')
    _transition(pdf, (story.get('transitions') or {}).get('after_drama'))
    _heading(pdf, 'Resolution')
    _body(pdf, story['resolution'])
    _transition(pdf, (story.get('transitions') or {}).get('after_resolution'))
    _heading(pdf, 'One thing')
    _callout(pdf, story['one_thing'], story['why'])
    _flush_heading(pdf)
