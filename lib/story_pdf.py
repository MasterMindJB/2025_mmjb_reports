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


def _kicker(pdf, text: str) -> None:
    pdf.set_text_color(108, 171, 221)
    pdf.set_font('Poppins', 'b', 9)
    pdf.cell(w=0, h=5, text=text.upper(), ln=True)
    pdf.set_text_color(20, 33, 74)


def _heading(pdf, text: str) -> None:
    _room(pdf, 28)
    pdf.ln(2)
    pdf.set_x(pdf.l_margin)
    pdf.set_font('Poppins', 'b', 14)
    pdf.set_text_color(20, 33, 74)
    pdf.cell(w=0, h=8, text=text, ln=True)


def _body(pdf, text: str) -> None:
    pdf.set_x(pdf.l_margin)
    pdf.set_font('Poppins', '', 11)
    pdf.set_text_color(20, 33, 74)
    pdf.multi_cell(w=0, text=text or '')
    pdf.ln(1)


def _callout(pdf, title: str, text: str) -> None:
    _room(pdf, 40)
    pdf.set_x(pdf.l_margin)
    pdf.set_fill_color(244, 248, 252)
    pdf.set_font('Poppins', 'b', 12)
    pdf.set_text_color(28, 44, 91)
    pdf.multi_cell(w=0, h=6, text=title, fill=True)
    pdf.set_x(pdf.l_margin)
    pdf.set_font('Poppins', '', 11)
    pdf.set_text_color(20, 33, 74)
    pdf.multi_cell(w=0, text=text, fill=True)
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


def _timeline(story: dict, folder: Path) -> Path | None:
    lines = story.get('lines') or []
    if not lines:
        return None
    figure, axis = plt.subplots(figsize=(5.2, 1.3), facecolor=LIGHT)
    axis.set_facecolor(LIGHT)
    for row in lines:
        start = float(row.get('start') or 0) / 1000
        end = float(row.get('end') or start) / 1000
        axis.barh(0, max(end - start, 0.2), left=start, height=0.4, color=NAVY)
    axis.set_yticks([])
    axis.set_xlabel('Seconds', color=MUTED, fontproperties=load_poppins('regular'), fontsize=8)
    axis.tick_params(colors=INK, labelsize=8)
    for spine in ('top', 'right', 'left'):
        axis.spines[spine].set_visible(False)
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
        return
    _room(pdf, 32)
    pdf.image(str(path), x='C', w=pdf.epw * width)
    pdf.ln(1)


def write_personal_report(pdf, story: dict, chart_dir: Path) -> None:
    pdf.add_page()
    pdf.set_font('Poppins', '', 11)
    pdf.set_text_color(20, 33, 74)
    _kicker(pdf, 'Your conversation')
    pdf.set_font('Poppins', 'b', 16)
    pdf.multi_cell(w=0, text=story['headline'])
    pdf.ln(1)
    _heading(pdf, 'Character')
    _body(
        pdf,
        f"Share of talk {story['share']:.0f}%. Turns {story['turns']}. "
        f"Questions {story['questions']}. Pace {story['wpm']:.0f} words per minute.",
    )
    _image(pdf, _talk_share(story, chart_dir))
    _callout(pdf, 'One thing', story['one_thing'])
    _heading(pdf, 'Goal')
    _body(pdf, story['goal'])
    _heading(pdf, 'Challenge')
    _body(pdf, story['challenge'])
    for item in story.get('contributions') or []:
        quote = item.get('quote') or ''
        tag = item.get('tagline') or ''
        if quote or tag:
            _body(pdf, f'{tag} "{quote}"'.strip())
    _image(pdf, _timeline(story, chart_dir))
    _heading(pdf, 'Drama')
    _body(pdf, story['drama'])
    card = story['signature']
    _room(pdf, 40)
    _body(pdf, 'Your signature phrase:')
    _kicker(pdf, 'Your signature phrase')
    if card.get('phrase'):
        _body(pdf, f'You said "{card["phrase"]}" {card.get("count") or 0} times.')
    for quote in card.get('quotes') or []:
        _body(pdf, quote)
    _image(pdf, _phrase_chart(story, chart_dir), width=0.55)
    for label, key in (('What?', 'what'), ('So, what?', 'so_what'), ('Now, what?', 'now_what')):
        if card.get(key):
            _body(pdf, f'{label} {card[key]}')
    _heading(pdf, 'Resolution')
    _body(pdf, story['resolution'])
    _heading(pdf, 'One thing')
    _callout(pdf, story['one_thing'], story['why'])
    if story.get('actions'):
        _body(pdf, 'Supporting moves, all about the next conversation:')
        for item in story['actions']:
            _body(pdf, f"{item.get('title') or 'Try this'}. {item.get('action') or ''}".strip())
    _heading(pdf, 'Appendix: style lenses')
    _body(pdf, 'These lenses stay in the file for now. Each line is only the top result.')
    for name, line in story['appendix'].items():
        _body(pdf, f'{name}: {line}')


def write_group_story(pdf, story: dict, chord_path, chord_pairs) -> None:
    pdf.add_page()
    pdf.set_text_color(20, 33, 74)
    _kicker(pdf, 'The session')
    pdf.set_font('Poppins', 'b', 16)
    pdf.multi_cell(w=0, text=story['headline'])
    _heading(pdf, 'Character')
    _body(pdf, story['character'])
    _heading(pdf, 'Goal')
    _body(pdf, story['goal'])
    _heading(pdf, 'Challenge')
    _body(pdf, story['challenge'])
    _heading(pdf, 'Drama')
    _body(
        pdf,
        'Who spoke right after whom. A wider ribbon means those two followed each other more often. '
        'Color is the person who spoke first.',
    )
    if chord_path and Path(chord_path).is_file():
        _room(pdf, 70)
        pdf.image(str(chord_path), x='C', w=pdf.epw * 0.92)
        pdf.ln(1)
    from lib.pdf_functions import _add_direction_table

    _add_direction_table(pdf, chord_pairs or [])
    _heading(pdf, 'Resolution')
    _body(pdf, story['resolution'])
    _heading(pdf, 'One thing')
    _callout(pdf, story['one_thing'], story['why'])
