import io
import logging
from pathlib import Path
from typing import Optional, TypeAlias

import pandas as pd

from lib.helper_functions import spider_chart_0to5

logger_PIL = logging.getLogger('PIL')
logger_PIL.setLevel(logging.INFO)
logger_fontTools = logging.getLogger('fontTools')
logger_fontTools.setLevel(logging.WARNING)
logger_fpdf = logging.getLogger('fpdf')
logger_fpdf.setLevel(logging.INFO)


import PIL.Image
from fpdf import FPDF
from fpdf.fonts import FontFace

JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None


class MM_PDF(FPDF):

    def set_font_to_recall(self):
        self.font_family_to_recall, self.font_style_to_recall, self.font_sizpt_to_recall = (
            self.font_family,
            self.font_style,
            self.font_size_pt,
        )

    def set_font_recall(self):
        self.set_font(
            self.font_family_to_recall, self.font_style_to_recall, self.font_sizpt_to_recall
        )

    def __init__(
        self,
        path_report_assets: Path,
        orientation="portrait",
        unit="mm",
        format="A4",
        font_cache_dir="DEPRECATED",
        link_feedback: str = None,
    ):
        # print('__init__')
        # print('wow')
        super().__init__(orientation, unit, format, font_cache_dir)

        self.set_margins(20, 10)

        self.link_feedback = link_feedback

        path_kulim_dir = path_report_assets.joinpath('fonts').joinpath('Kulim_Park')
        self.add_font('KulimPark', style='', fname=path_kulim_dir.joinpath('KulimPark-Regular.ttf'))
        self.add_font('KulimPark', style='b', fname=path_kulim_dir.joinpath('KulimPark-Bold.ttf'))
        self.add_font('KulimPark', style='i', fname=path_kulim_dir.joinpath('KulimPark-Italic.ttf'))
        self.add_font(
            'KulimPark', style='bi', fname=path_kulim_dir.joinpath('KulimPark-BoldItalic.ttf')
        )
        self.add_font(
            'KulimPark-light', style='', fname=path_kulim_dir.joinpath('KulimPark-Light.ttf')
        )
        self.add_font(
            'KulimPark-light', style='i', fname=path_kulim_dir.joinpath('KulimPark-LightItalic.ttf')
        )

        path_zilla_dir = path_report_assets.joinpath('fonts').joinpath('Zilla_Slab')
        self.add_font('ZillaSlab', style='', fname=path_zilla_dir.joinpath('ZillaSlab-Regular.ttf'))
        self.add_font('ZillaSlab', style='b', fname=path_zilla_dir.joinpath('ZillaSlab-Bold.ttf'))
        self.add_font('ZillaSlab', style='i', fname=path_zilla_dir.joinpath('ZillaSlab-Italic.ttf'))
        self.add_font(
            'ZillaSlab', style='bi', fname=path_zilla_dir.joinpath('ZillaSlab-BoldItalic.ttf')
        )
        self.add_font(
            'ZillaSlab-light', style='', fname=path_zilla_dir.joinpath('ZillaSlab-Light.ttf')
        )
        self.add_font(
            'ZillaSlab-light', style='i', fname=path_zilla_dir.joinpath('ZillaSlab-LightItalic.ttf')
        )

        path_dejavusans_dir = path_report_assets.joinpath('fonts').joinpath('DejaVuSans')
        self.add_font('DejaVuSans', style='', fname=path_dejavusans_dir.joinpath('DejaVuSans.ttf'))

        poppins_dir = Path(__file__).resolve().parents[1] / 'assets' / 'fonts' / 'Poppins'
        self.add_font('Poppins', '', fname=poppins_dir / 'Poppins-Regular.ttf')
        self.add_font('Poppins', 'b', fname=poppins_dir / 'Poppins-SemiBold.ttf')
        self.add_font('PoppinsMedium', '', fname=poppins_dir / 'Poppins-Medium.ttf')
        self.brand_canvas = False
        self._brand_pages: set[int] = set()

        # Font settings
        self.set_font('KulimPark', '', 12)
        self.set_font_to_recall()

    def header(self):
        if getattr(self, 'brand_canvas', False):
            self._brand_pages.add(self.page)
            self.brand_canvas = False
            self.set_fill_color(244, 248, 252)
            self.rect(0, 0, self.w, self.h, 'F')
            self.set_fill_color(255, 255, 255)
        branded = self.page in getattr(self, '_brand_pages', ())
        self.set_text_color(20, 33, 74) if branded else self.set_text_color(0, 0, 0)
        self.set_font('KulimPark', '', 11)
        line_height = self.font_size * 2
        self.cell(w=self.epw * 0.72, text='Harkness Training Solutions', align='L', h=line_height)
        self.set_text_color(85, 98, 138) if branded else self.set_text_color(0, 0, 0)
        self.set_font('KulimPark', '', 9)
        self.cell(w=self.epw * 0.28, text='Confidential', align='R', h=line_height)
        self.set_text_color(20, 33, 74) if branded else self.set_text_color(0, 0, 0)
        self.ln(12)

    # Page footer
    def footer(self):
        self.set_y(-15)  # Set position of the footer
        branded = self.page in getattr(self, '_brand_pages', ())
        if branded:
            self.set_text_color(85, 98, 138)
        if self.link_feedback:
            self.set_font('DejaVuSans', '', 13)
            self.cell(6, 10, '😊', align='L')
            self.set_font('KulimPark-light', 'U', 11)
            self.cell(
                0,
                10,
                text="Leave a Comment!",
                align='L',
                link=self.link_feedback,
            )
        self.set_font('KulimPark-light', '', 11)
        self.cell(0, 10, f'{self.page_no()}/{{nb}}', align='R')  # Page number
        self.set_text_color(0, 0, 0)

    def add_title_document(self, title: str, subtitle: str = None):
        self.set_font_to_recall()
        self.set_font('KulimPark-light', '', 29)
        self.cell(w=0, h=self.font_size * 1.4, text=title, align='C', ln=True)
        if subtitle is not None:
            self.set_font('KulimPark-light', '', 14)
            self.set_text_color((160, 160, 160))
            self.cell(
                w=0,
                text=subtitle,
                align='C',
                ln=True,
                markdown=True,
            )
            self.set_text_color((0, 0, 0))
        self.set_font_recall()

    def add_title_1(self, text: str, link_dest: int = None):
        self.set_font_to_recall()
        self.set_font('KulimPark', '', 25)
        if link_dest is not None:
            self.set_link(link_dest)
        self.cell(
            w=0, h=self.font_size * 2, text=text, align='L', markdown=True, ln=True, border='B'
        )
        self.ln(h=self.font_size / 2)
        self.set_font_recall()

    def add_title_2(self, text: str, link_dest: int = None):
        self.set_font_to_recall()
        self.set_font('KulimPark', '', 18)
        self.cell(w=0, h=self.font_size * 2, text=text, align='L', markdown=True, ln=True)
        if link_dest is not None:
            self.set_link(link_dest)
        self.ln(0)
        self.set_font_recall()

    def add_speaker_stats_table(self, df_nice_stats_data):
        self.set_font_to_recall()
        col_count = len(df_nice_stats_data.columns) + 1
        col_widths = [15] + [85 / col_count for c in range(col_count - 1)]
        text_align = ['left'] + ['CENTER' for c in range(col_count - 1)]
        # Headings are bold. KulimPark-light has no bold face, so the table uses KulimPark.
        self.set_font('KulimPark', '', 8)
        with self.table(
            text_align=text_align,
            col_widths=col_widths,
            padding=1,
            line_height=self.font_size,
            borders_layout='MINIMAL',
        ) as table:
            row = table.row()
            titles = ['Name'] + list(df_nice_stats_data.columns)
            for title in titles:
                row.cell(title)
            for index, rowdata in df_nice_stats_data.iterrows():
                # Hack, see https://github.com/pandas-dev/pandas/issues/58543#issuecomment-2101240339
                rowdata = rowdata.convert_dtypes().astype('object').infer_objects()
                datamax = max(rowdata)
                datamin = min(rowdata)
                row = table.row()
                row.cell(index)
                for data_i in range(len(rowdata)):
                    val = rowdata.iloc[data_i]
                    style = FontFace()
                    if data_i == len(rowdata) - 1:
                        style = FontFace(color=(80, 80, 80), fill_color=(245, 245, 245))
                    else:
                        if val == datamax:
                            style = FontFace(fill_color=(255, 192, 203))  # pink
                        elif val == datamin:
                            style = FontFace(fill_color=(176, 224, 230))  # powderblue
                    row.cell(str(val), style=style)
        self.set_font_recall()

    def add_comments_section(self):
        self.add_page()
        self.set_font_to_recall()
        self.set_font('KulimPark-light', '', 20)
        self.multi_cell(
            w=0,
            padding=(60, 0, 2.5, 0),
            text="How can we improve?",
            align='C',
            ln=True,
            markdown=True,
        )
        self.set_font('KulimPark-light', 'U', 26)
        self.multi_cell(
            w=0,
            padding=2.5,
            text="Leave a Comment!",
            align='C',
            ln=True,
            markdown=True,
            link=self.link_feedback,
        )
        self.set_font_recall()


def pdf_filename_for(name: str) -> str:
    """A single path segment. Display names may contain slashes; file names may not."""
    cleaned = ''.join('_' if char in '\\/:*?"<>|' or ord(char) < 32 else char for char in name)
    cleaned = cleaned.strip().strip('.')
    return f'{cleaned or "participant"}.pdf'


def _place_chord_image(pdf: 'MM_PDF', path: Path) -> None:
    with PIL.Image.open(path) as image:
        aspect = image.height / max(image.width, 1)
    reserve = 64
    available_h = pdf.h - pdf.get_y() - pdf.b_margin - reserve
    available_w = pdf.epw
    width = available_w
    if available_h > 20 and width * aspect > available_h:
        width = available_h / aspect
    pdf.image(path, x='C', w=max(width, 40))
    pdf.ln(2)


def _add_direction_table(pdf: 'MM_PDF', pairs: list) -> None:
    ink = (20, 33, 74)
    muted = (85, 98, 138)
    navy = (28, 44, 91)
    line = (216, 227, 240)
    light = (244, 248, 252)
    white = (255, 255, 255)
    y = pdf.get_y()
    pdf.set_fill_color(108, 171, 221)
    pdf.rect(pdf.l_margin, y, 16, 1.1, 'F')
    pdf.ln(4)
    pdf.set_text_color(*ink)
    pdf.set_font('Poppins', 'b', 12)
    pdf.cell(w=0, h=7, text='Top pairs, each direction', align='L', ln=True)
    pdf.set_font('Poppins', '', 9)
    pdf.set_text_color(*muted)
    pdf.multi_cell(
        w=0,
        text='Each number is how many times the first person was immediately followed by the second.',
    )
    pdf.ln(1)
    if not pairs:
        pdf.set_font('Poppins', '', 10)
        pdf.set_text_color(*ink)
        pdf.multi_cell(w=0, text='No turn was immediately followed by a different person.')
        pdf.set_text_color(0, 0, 0)
        pdf.set_font('KulimPark', '', 12)
        return
    pdf.set_font('Poppins', '', 8)
    pdf.set_text_color(*ink)
    pdf.set_draw_color(*line)
    pdf.set_line_width(0.2)
    heading = FontFace(family='Poppins', emphasis='BOLD', size_pt=8, color=white, fill_color=navy)
    with pdf.table(
        text_align=['LEFT', 'LEFT', 'LEFT'],
        col_widths=(3, 2, 2),
        borders_layout='HORIZONTAL_LINES',
        line_height=pdf.font_size * 1.45,
        first_row_as_headings=True,
    ) as table:
        header = table.row()
        header.cell('Pair', style=heading)
        header.cell('One way', style=heading)
        header.cell('The other way', style=heading)
        for index, (name_a, name_b, first_way, other_way) in enumerate(pairs[:6]):
            short_a = name_a.split(' ', 1)[0]
            short_b = name_b.split(' ', 1)[0]
            fill = light if index % 2 == 0 else white
            style = FontFace(family='Poppins', size_pt=8, color=ink, fill_color=fill)
            row = table.row()
            row.cell(f'{name_a}  and  {name_b}', style=style)
            row.cell(f'{short_a} then {short_b}: {first_way}', style=style)
            row.cell(f'{short_b} then {short_a}: {other_way}', style=style)
    pdf.set_draw_color(0, 0, 0)
    pdf.set_text_color(0, 0, 0)
    pdf.set_line_width(0.2)
    pdf.set_font('KulimPark', '', 12)
    pdf.ln(2)


def _sorted_speakers(speaker_map: dict):
    return sorted(
        [(speaker, speaker_map[speaker]['name']) for speaker in speaker_map], key=lambda item: item[1]
    )


def _clean_llm_markdown(text: str) -> str:
    """Make Gemini markdown safe for fpdf2, which only understands bold and italic.

    Headings become bold lines, bullets become dots, rules are dropped, and
    markdown tables become plain rows so the pipe characters are not printed raw.
    """
    import re

    lines = []
    for raw in str(text or '').splitlines():
        line = raw.rstrip()
        heading = re.match(r'^\s*#{1,6}\s*(.+)$', line)
        if heading:
            title = heading.group(1).strip().strip('*').strip()
            line = f'**{title}**'
        elif re.match(r'^\s*[-*+]\s+', line):
            line = re.sub(r'^(\s*)[-*+]\s+', lambda match: match.group(1) + '\u2022 ', line)
        elif re.match(r'^\s*(-{3,}|\*{3,}|_{3,})\s*$', line):
            line = ''
        elif re.match(r'^\s*\|', line):
            cells = [cell.strip() for cell in line.strip().strip('|').split('|')]
            if cells and all(re.fullmatch(r':?-{2,}:?', cell or '--') for cell in cells):
                continue
            line = ' \u2014 '.join(cell for cell in cells if cell)
        line = line.replace('\u2192', '->')
        lines.append(line)
    cleaned = '\n'.join(lines)
    return re.sub(r'\n{3,}', '\n\n', cleaned).strip()


def _add_signature_phrase(pdf: 'MM_PDF', signature: dict | None, link_dest=None):
    if not signature:
        pdf.add_title_1(text='Your signature phrase', link_dest=link_dest)
        pdf.set_font('KulimPark-light', '', 12)
        pdf.multi_cell(
            w=0,
            text='Not enough speech in this session to find a repeated phrase.',
            ln=True,
        )
        return
    bigram = signature.get('bigram', '')
    count = int(signature.get('count') or 0)
    pdf.add_title_1(text=f'Your signature phrase: \u201c{bigram}\u201d', link_dest=link_dest)
    pdf.set_font('KulimPark', 'b', 12)
    times = 'once' if count == 1 else f'{count} times'
    pdf.multi_cell(
        w=0,
        text=f'Your most frequent two-word phrase in this session was \u201c{bigram}\u201d, used {times}.',
        ln=True,
    )
    pdf.ln(2)
    pdf.set_font('KulimPark', '', 11)
    pdf.multi_cell(
        w=0,
        text=_clean_llm_markdown(signature.get('analysis', '')),
        align='L',
        ln=True,
        markdown=True,
    )
    pdf.ln(4)


def _add_own_quotes(pdf: MM_PDF, quotes):
    pdf.set_font_to_recall()
    if not quotes:
        pdf.set_font('KulimPark-light', '', 12)
        pdf.multi_cell(w=0, text='None noted.', ln=True)
        pdf.set_font_recall()
        return
    for quote in quotes:
        pdf.set_font('KulimPark', '', 12)
        pdf.set_x(pdf.l_margin + pdf.epw * 0.05)
        pdf.multi_cell(
            w=0,
            h=1.5 * pdf.font_size,
            ln=True,
            markdown=True,
            text=f'**{_clean_llm_markdown(quote["tagline"])}**',
        )
        pdf.set_font('KulimPark-light', 'i', 12)
        pdf.set_x(pdf.l_margin + pdf.epw * 0.05)
        pdf.multi_cell(
            w=0,
            align='J',
            ln=True,
            markdown=True,
            text=f'"{_clean_llm_markdown(quote["quote"])}"',
        )
        pdf.ln(5)
    pdf.set_font_recall()


def _add_own_actions(pdf: MM_PDF, actions):
    if not actions:
        pdf.set_font('KulimPark-light', '', 12)
        pdf.multi_cell(w=0, text='None noted.', ln=True)
        pdf.set_font_recall()
        return
    for action in actions:
        pdf.set_x(pdf.l_margin + pdf.epw * 0.05)
        pdf.multi_cell(
            w=0,
            h=1.5 * pdf.font_size,
            ln=True,
            markdown=True,
            text=f'**{_clean_llm_markdown(action["title"])}**',
        )
        pdf.set_font_to_recall()
        pdf.set_font('KulimPark-light', '', 12)
        pdf.set_x(pdf.l_margin + pdf.epw * 0.05)
        pdf.multi_cell(
            w=pdf.epw * 0.8,
            align='J',
            ln=True,
            markdown=True,
            text=_clean_llm_markdown(action['action']),
        )
        pdf.set_font_recall()
        pdf.ln(5)


def _add_group_individual_sections(
    pdf: MM_PDF,
    speaker_map: dict,
    image_action_items: PIL.Image,
    link_key_contribs: int,
    link_action_items: int,
    link_participant_summ: int,
):
    pdf.add_page()
    pdf.add_title_1(text='Key Contributions', link_dest=link_key_contribs)
    for speaker, _name in _sorted_speakers(speaker_map):
        pdf.add_title_2(text=speaker_map[speaker]['name'])
        _add_own_quotes(pdf, speaker_map[speaker]['personal_key_contributions'])

    pdf.add_page()
    pdf.add_title_1(text='Action Items', link_dest=link_action_items)
    pdf.image(image_action_items, x='C', w=pdf.epw * 1 / 3)
    for speaker, _name in _sorted_speakers(speaker_map):
        pdf.add_title_2(text=speaker_map[speaker]['name'])
        _add_own_actions(pdf, speaker_map[speaker]['personal_action_items'])

    pdf.add_page()
    pdf.add_title_1(text='Participant Summary', link_dest=link_participant_summ)
    for speaker, _name in _sorted_speakers(speaker_map):
        pdf.add_title_2(text=speaker_map[speaker]['name'])
        pdf.multi_cell(
            w=0,
            text=_clean_llm_markdown(speaker_map[speaker]['personal_summary']),
            align='J',
            ln=True,
            markdown=True,
        )
        pdf.ln(5)


def create_report_group(
    images_barplot: [PIL.Image],
    image_convo_sentiment_evo: PIL.Image,
    image_speaker_sentiment_evo: PIL.Image,
    image_convo_wpm: PIL.Image,
    image_speaker_wpm: PIL.Image,
    image_action_items: PIL.Image,
    path_image_chord: Path,
    path_report_assets: Path,
    path_file: Path,
    summaries: JSON,
    gini_reverse_interventions: float,
    gini_reverse_interactions: float,
    gini_reverse_interactions_both_wrd_cnt: float,
    df_nice_stats_data: pd.DataFrame,
    speaker_map: dict,
    link_feedback: str,
    include_individual_sections: bool = False,
    path_image_matrix: Path | None = None,
    chord_pairs: list | None = None,
):
    logging.info('Creating PDF report.')

    pdf = MM_PDF(
        path_report_assets=path_report_assets,
        orientation='P',
        unit='mm',
        format='letter',
        link_feedback=link_feedback,
    )

    pdf.add_page()
    pdf.set_font('KulimPark', '', 12)

    # Document title
    pdf.add_title_document(
        title='Conversation Health Report',
        subtitle='Feedback & actionable recommendations to enhance your communication.',
    )
    pdf.set_font('KulimPark-light', '', 10)
    pdf.multi_cell(
        w=0,
        text=(
            'Confidential group report. This document summarizes the session. '
            'It does not include any participant\'s individual report.'
        ),
        align='C',
    )
    pdf.set_font('KulimPark', '', 12)

    # Introduction, conversation short summary
    pdf.set_font_to_recall()
    pdf.ln(15)
    pdf.multi_cell(w=pdf.epw * 0.5, padding=0, text='', border='B', center=True)
    pdf.ln(0)
    pdf.set_font('KulimPark', '', 14)
    pdf.multi_cell(
        w=0,
        padding=(8, 10, 0),
        text=_clean_llm_markdown(summaries['conversation_title']),
        align='C',
    )
    pdf.ln(0)
    pdf.set_font('KulimPark-light', '', 12)
    pdf.multi_cell(
        w=0,
        padding=(6, 10, 8),
        text=_clean_llm_markdown(summaries['conversation_summary_short']),
    )
    pdf.ln(0)
    pdf.multi_cell(w=pdf.epw * 0.5, padding=0, text='', border='T', center=True)
    pdf.ln(5)
    pdf.set_font_recall()

    # The chord diagram is the first view of the session, ahead of the scores.
    pdf.brand_canvas = True
    pdf.add_page()
    link_chord = pdf.add_link()
    pdf.set_link(link_chord)
    _place_chord_image(pdf, path_image_chord)
    _add_direction_table(pdf, chord_pairs or [])

    # Table of Contents
    link_convo_stats = pdf.add_link()
    link_convo_summ = pdf.add_link()
    link_speaker_stats = pdf.add_link()
    link_action_items = pdf.add_link()
    link_key_contribs = pdf.add_link()
    link_participant_summ = pdf.add_link()
    link_convo_evo = pdf.add_link()
    link_wpm_evo = pdf.add_link()

    pdf.set_font('KulimPark', '', 18)
    pdf.cell(w=0, h=pdf.font_size * 1.5, text='Table of Contents', align='C', ln=True)
    pdf.ln(h=pdf.font_size * 3 / 4)
    pdf.set_font('KulimPark', '', 12)

    with pdf.table(
        width=pdf.epw * 0.8,
        text_align=['R', 'L'],
        # col_widths=col_widths,
        padding=(2, 2),
        line_height=pdf.font_size,
        # borders_layout="INTERNAL"
        borders_layout='NONE',
        first_row_as_headings=False,
    ) as table:
        style_link = FontFace(family='KulimPark', emphasis='BOLD', size_pt=12)
        style_desc = FontFace(family='KulimPark-light', emphasis='ITALICS', size_pt=10)
        row = table.row()
        row.cell('Who talks with whom', style=style_link, link=link_chord)
        row.cell(
            'Who spoke immediately after whom. Color shows who spoke first.',
            style=style_desc,
        )
        row = table.row()
        row.cell('Conversation Statistics', style=style_link, link=link_convo_stats)
        row.cell(
            'The measure of how evenly participants engaged in the conversation.', style=style_desc
        )
        row = table.row()
        row.cell('Speaker Statistics', style=style_link, link=link_speaker_stats)
        row.cell('Some interesting stats about the way each participant spoke.', style=style_desc)
        row = table.row()
        row.cell('Conversation Summary', style=style_link, link=link_convo_summ)
        row.cell('A brief recount of what happened in the session.', style=style_desc)
        if include_individual_sections:
            row = table.row()
            row.cell('Key Contributions', style=style_link, link=link_key_contribs)
            row.cell('Three important ideas brought by each participant.', style=style_desc)
            row = table.row()
            row.cell('Action Items', style=style_link, link=link_action_items)
            row.cell('Actionable ideas discussed during the session.', style=style_desc)
            row = table.row()
            row.cell('Participant Summary', style=style_link, link=link_participant_summ)
            row.cell('A summary of each participant\'s contribution.', style=style_desc)
        row = table.row()
        row.cell('Sentiment Evolution', style=style_link, link=link_convo_evo)
        row.cell(
            'How the sentiment of the conversation evolved throughout the conversation.',
            style=style_desc,
        )
        row = table.row()
        row.cell('Words Per Minute Evolution', style=style_link, link=link_wpm_evo)
        row.cell('How the rate of speech evolved throughout the conversation.', style=style_desc)

    # Conversation Statistics
    pdf.add_page()
    pdf.add_title_1(text='Conversation Statistics', link_dest=link_convo_stats)

    pdf.ln(3)
    if path_image_matrix is not None:
        pdf.set_font('KulimPark-light', '', 10)
        pdf.multi_cell(
            w=0,
            text=(
                'Secondary view. Each cell counts how many times the person in the row '
                'was immediately preceded by the person in the column.'
            ),
        )
        pdf.image(path_image_matrix, x='C', w=pdf.epw * 0.62)
    else:
        pdf.image(path_image_chord, x='C', w=pdf.epw * 0.5)
    pdf.set_font('KulimPark', '', 12)
    pdf.ln(3)

    pdf.add_title_2(text=f'Group **Participation** Score: {round(100*gini_reverse_interventions)}%')
    pdf.set_font_to_recall()
    pdf.set_font('KulimPark-light', '', 10)
    pdf.multi_cell(
        w=0,
        h=pdf.font_size,
        text='The Group Participation Score measures how evenly participants engage in a conversation, taking into account the number of interventions by each participant. So for instance, if every participant in the conversation spoke 10 times, the score will be 100%. However, if all participants except one spoke 10 times and the other one spoke 50 times, the score will be much lower.',
        align='J',
        markdown=True,
    )
    pdf.set_font_recall()
    pdf.ln()

    pdf.add_title_2(text=f'Group **Interaction** Score: {round(100*gini_reverse_interactions)}%')
    pdf.set_font_to_recall()
    pdf.set_font('KulimPark-light', '', 10)
    pdf.multi_cell(
        w=0,
        h=pdf.font_size,
        text='The Group Interaction Score measures how evenly participants engage in a conversation between each other, taking into account the number of interactions between each participant. So for instance if every participant spoke to every other participant 5 times, the score will be 100%. However, if all participants except one spoke to each other once and the other one spoke to a specific participant 20 times, the score will be much lower.',
        align='J',
        markdown=True,
    )
    pdf.set_font_recall()
    pdf.ln()

    pdf.add_title_2(
        text=f'Group **Word Interaction** Score: {round(100*gini_reverse_interactions_both_wrd_cnt)}%'
    )
    pdf.set_font_to_recall()
    pdf.set_font('KulimPark-light', '', 10)
    pdf.multi_cell(
        w=0,
        h=pdf.font_size,
        text='The Group Word Interaction Score measures how evenly participants engage in a conversation between each other and how much they talked to each other, taking into account the word count of each interaction. So for instance if every participant spoke to every other participant 100 words each time, the score will be 100%. However, if all participants spoke to each other 100 words, but one participant spoke a lot more words to the others, the score will be much lower.',
        align='J',
        markdown=True,
    )
    pdf.set_font_recall()
    pdf.ln()
    # pdf.ln(5)

    # Speaker Statistics
    pdf.add_page()
    pdf.add_title_1(text='Speaker Statistics', link_dest=link_speaker_stats)
    pdf.add_title_2(text='Statistics Plots')
    for _img in images_barplot:
        pdf.image(_img, x='C', w=pdf.epw)
    pdf.ln()
    # pdf.add_page()
    pdf.add_title_2(text='Statistics Table')
    pdf.add_speaker_stats_table(df_nice_stats_data=df_nice_stats_data)

    # Conversation Summary for each speaker
    pdf.add_page()
    pdf.add_title_1(text='Conversation Summary', link_dest=link_convo_summ)
    pdf.set_font_to_recall()
    pdf.set_font('KulimPark-light', '', 12)
    pdf.multi_cell(
        w=0,
        text=_clean_llm_markdown(summaries['conversation_summary_long']),
        align='J',
        ln=True,
        markdown=True,
    )
    pdf.set_font_recall()

    # Per-person quotes, action items, and summaries are individual sections.
    # They stay out of the group report unless a caller explicitly opts in.
    if include_individual_sections:
        _add_group_individual_sections(
            pdf,
            speaker_map,
            image_action_items,
            link_key_contribs,
            link_action_items,
            link_participant_summ,
        )

    # Sentiment Evolution
    pdf.add_page()
    pdf.add_title_1(text='Sentiment Evolution', link_dest=link_convo_evo)
    pdf.image(image_convo_sentiment_evo, x='C', w=pdf.epw)
    pdf.ln(2)
    pdf.image(image_speaker_sentiment_evo, x='C', w=pdf.epw)

    # Words per minute Evolution
    pdf.add_page()
    pdf.add_title_1(text='Words per minute Evolution', link_dest=link_wpm_evo)
    pdf.image(image_convo_wpm, x='C', w=pdf.epw)
    pdf.ln(2)
    pdf.image(image_speaker_wpm, x='C', w=pdf.epw)

    pdf.add_comments_section()

    pdf.output(path_file)
    logging.info('...created pdf [%s].', path_file)


def create_report_personal(
    path_report_assets: Path,
    path_reports_dir: Path,
    df_nice_stats_data: pd.DataFrame,  # Used for personal statistics
    speaker_map: dict,  # Used for(1) personal summary
    link_feedback: str,
    speaker_analysis: dict,  # Used for(1) personal super power and (2) Errors and Corrections
    df_framework_amase_values: pd.DataFrame,
    df_framework_amase_explanation: pd.DataFrame,
    df_framework_enneagram_values: pd.DataFrame,
    df_framework_enneagram_explanation: pd.DataFrame,
    df_framework_doshas_values: pd.DataFrame,
    df_framework_doshas_explanation: pd.DataFrame,
    file_name_prefix: Optional[str] = None,
):
    logging.info('Creating personal PDF reports for each speaker.')

    # Create a PDF for each speaker
    for speaker, speaker_info in speaker_map.items():
        speaker_name = speaker_info['name']

        # Create PDF filename for this speaker
        if file_name_prefix:
            speaker_filename = path_reports_dir.joinpath(
                f"{file_name_prefix} - {pdf_filename_for(speaker_name)}"
            )
        else:
            speaker_filename = path_reports_dir.joinpath(pdf_filename_for(speaker_name))

        logging.info(f'Creating personal PDF report for {speaker_name}.')

        # Initialize PDF
        pdf = MM_PDF(
            path_report_assets=path_report_assets,
            orientation='P',
            unit='mm',
            format='letter',
            link_feedback=link_feedback,
        )

        pdf.add_page()
        pdf.set_font('KulimPark', '', 12)

        # Title page
        # pdf.add_title_document(
        #     title=f'{speaker_name}\'s Personal Report',
        #     subtitle='Your personal insights and communication patterns.',
        # )
        pdf.add_title_document(
            title=f'{speaker_name}\'s Personal Report',
            subtitle='Feedback & actionable recommendations to enhance your communication.',
        )
        pdf.set_font('KulimPark-light', '', 10)
        pdf.multi_cell(
            w=0,
            text=(
                f'Confidential. This individual report was prepared for {speaker_name} '
                'and is not for redistribution.'
            ),
            align='C',
        )
        pdf.set_font('KulimPark', '', 12)

        # Introduction, conversation short summary
        pdf.set_font_to_recall()
        pdf.ln(15)
        pdf.multi_cell(w=pdf.epw * 0.5, padding=0, text='', border='B', center=True)
        pdf.ln(0)
        pdf.set_font('KulimPark', '', 14)
        pdf.multi_cell(w=0, padding=(8, 10, 0), text=f'Your Personal Super Power', align='C')
        pdf.ln(0)
        pdf.set_font('KulimPark-light', '', 12)
        pdf.multi_cell(
            w=0,
            padding=(6, 10, 8),
            text=_clean_llm_markdown(speaker_analysis[speaker]['personal_super_power']),
        )
        pdf.ln(0)
        pdf.multi_cell(w=pdf.epw * 0.5, padding=0, text='', border='T', center=True)
        pdf.ln(5)
        pdf.set_font_recall()

        # Table of Contents
        link_personal_summary = pdf.add_link()
        link_key_contribs = pdf.add_link()
        link_action_items = pdf.add_link()
        link_signature_phrase = pdf.add_link()
        link_statistics = pdf.add_link()
        link_amase_framework = pdf.add_link()
        link_enneagram_framework = pdf.add_link()
        link_doshas_framework = pdf.add_link()
        # link_errors_corrections = pdf.add_link()

        pdf.set_font('KulimPark', '', 18)
        pdf.cell(w=0, h=pdf.font_size * 1.5, text='Table of Contents', align='C', ln=True)
        pdf.ln(h=pdf.font_size * 3 / 4)
        pdf.set_font('KulimPark', '', 12)

        with pdf.table(
            width=pdf.epw * 0.8,
            text_align=['R', 'L'],
            padding=(2, 2),
            line_height=pdf.font_size,
            borders_layout='NONE',
            first_row_as_headings=False,
        ) as table:
            style_link = FontFace(family='KulimPark', emphasis='BOLD', size_pt=12)
            style_desc = FontFace(family='KulimPark-light', emphasis='ITALICS', size_pt=10)
            row = table.row()
            row.cell('Personal Summary', style=style_link, link=link_personal_summary)
            row.cell('A comprehensive overview of your communication style.', style=style_desc)
            row = table.row()
            row.cell('Your Key Contributions', style=style_link, link=link_key_contribs)
            row.cell('Ideas from your part of the conversation.', style=style_desc)
            row = table.row()
            row.cell('Your Action Items', style=style_link, link=link_action_items)
            row.cell('Commitments drawn from your part of the conversation.', style=style_desc)
            row = table.row()
            row.cell('Your Signature Phrase', style=style_link, link=link_signature_phrase)
            row.cell('Your most frequent two-word phrase and what it may signal.', style=style_desc)
            row = table.row()
            row.cell('Your Statistics', style=style_link, link=link_statistics)
            row.cell('Key metrics and data about your speaking patterns.', style=style_desc)
            row = table.row()
            row.cell('AMASE Framework Analysis', style=style_link, link=link_amase_framework)
            row.cell(
                'Analysis of your communication through the AMASE framework.', style=style_desc
            )
            row = table.row()
            row.cell(
                'Enneagram Framework Analysis', style=style_link, link=link_enneagram_framework
            )
            row.cell(
                'Understanding your communication through the Enneagram lens.', style=style_desc
            )
            row = table.row()
            row.cell('Doshas Framework Analysis', style=style_link, link=link_doshas_framework)
            row.cell(
                'Analysis of your communication through the Doshas framework.', style=style_desc
            )
            row = table.row()
            # row.cell('Errors and Corrections', style=style_link, link=link_errors_corrections)
            # row.cell(
            #     'Communication patterns that could be improved and suggestions for better interactions.',
            #     style=style_desc,
            # )

        # 1. Personal Summary
        pdf.add_page()
        pdf.add_title_1(text='Personal Summary', link_dest=link_personal_summary)
        pdf.set_font_to_recall()
        pdf.set_font('KulimPark-light', '', 12)
        pdf.multi_cell(
            w=0,
            text=_clean_llm_markdown(speaker_info['personal_summary']),
            align='J',
            ln=True,
            markdown=True,
        )
        pdf.set_font_recall()

        pdf.add_title_1(text='Your Key Contributions', link_dest=link_key_contribs)
        _add_own_quotes(pdf, speaker_info.get('personal_key_contributions') or [])

        pdf.add_title_1(text='Your Action Items', link_dest=link_action_items)
        _add_own_actions(pdf, speaker_info.get('personal_action_items') or [])

        pdf.set_font_to_recall()
        _add_signature_phrase(
            pdf,
            (speaker_analysis.get(speaker) or {}).get('personal_signature_phrase'),
            link_dest=link_signature_phrase,
        )
        pdf.set_font_recall()

        # 2. Speaker Statistics (only for this speaker)
        # pdf.add_page() # TODO: Remove comment
        pdf.add_title_1(text='Your Statistics', link_dest=link_statistics)

        # print(df_nice_stats_data, flush=True) # TODO: Remove

        # Filter stats for this speaker only
        speaker_stats = (
            df_nice_stats_data.loc[[speaker_name]]
            if speaker_name in df_nice_stats_data.index
            else pd.DataFrame()
        )

        if not speaker_stats.empty:
            pdf.add_title_2(text='Statistics Table')
            pdf.add_speaker_stats_table(df_nice_stats_data=speaker_stats)
        else:
            pdf.set_font_to_recall()
            pdf.set_font('KulimPark-light', '', 12)
            pdf.multi_cell(
                w=0,
                text='No statistics available for this speaker.',
                align='C',
                ln=True,
            )
            pdf.set_font_recall()

        # 3. Print Framework Analysis for each framework

        for _values_df, _explanation_df, _title, _link in [
            (
                df_framework_amase_values,
                df_framework_amase_explanation,
                'AMASE Framework',
                link_amase_framework,
            ),
            (
                df_framework_enneagram_values,
                df_framework_enneagram_explanation,
                'Enneagram Framework',
                link_enneagram_framework,
            ),
            (
                df_framework_doshas_values,
                df_framework_doshas_explanation,
                'Doshas Framework',
                link_doshas_framework,
            ),
        ]:
            # print(_explanation_df, flush=True)
            pdf.add_page()
            pdf.add_title_1(text=f'{_title} Analysis', link_dest=_link)

            if speaker_name in _values_df.index:
                # Create spider chart for this speaker
                fig = spider_chart_0to5(
                    _values_df,
                    row_names=[speaker_name],
                    show_mean=True,
                    title_prefix=_title,
                )

                # Convert figure to PIL Image
                buf = io.BytesIO()
                fig.savefig(buf, format='png', bbox_inches='tight', dpi=300)
                buf.seek(0)
                spider_image = PIL.Image.open(buf)

                # Add chart to PDF
                pdf.image(spider_image, x='C', w=pdf.epw * 0.5)
                pdf.ln(5)

                # Add explanation for each column
                if speaker_name in _explanation_df.index:
                    speaker_explanations = _explanation_df.loc[speaker_name]

                    for column_name in _explanation_df.columns:
                        if column_name in speaker_explanations and pd.notna(
                            speaker_explanations[column_name]
                        ):
                            # Add column title
                            pdf.add_title_2(text=column_name.title())

                            # Add column content
                            pdf.set_font_to_recall()
                            pdf.set_font('KulimPark-light', '', 10)
                            pdf.cell(text=f'{_values_df.loc[speaker_name, column_name]}/5', w=pdf.epw * 0.05, ln=False, align='R')
                            pdf.set_font_recall()
                            pdf.set_font_to_recall()
                            pdf.set_font('KulimPark-light', '', 10)
                            pdf.multi_cell(
                                w=pdf.epw * 0.95,
                                text=_clean_llm_markdown(str(speaker_explanations[column_name])),
                                align='J',
                                markdown=True,
                            )
                            pdf.set_font_recall()
                            pdf.ln(3)  # Add some spacing between columns
            else:
                pdf.set_font_to_recall()
                pdf.set_font('KulimPark-light', '', 12)
                pdf.multi_cell(
                    w=0,
                    text='No AMASE framework data available for this speaker.',
                    align='C',
                    ln=True,
                )
                pdf.set_font_recall()

        # 4. Errors and Corrections

        #         pdf.add_page()
        #         pdf.add_title_1(text='Errors and Corrections', link_dest=link_errors_corrections)

        #         pdf.add_title_2(text='Friction Factor')
        #         # pdf.set_text_color((65, 141, 251))
        #         pdf.set_font_to_recall()
        #         pdf.set_font('KulimPark-light', '', 8)
        #         pdf.set_text_color((255, 108, 114))
        #         pdf.multi_cell(
        #                     w=0,
        #                     text='Please note that this section includes subjective observations generated by an AI model. While the suggestions aim to highlight potential friction points or areas for improvement, they may not all be accurate or applicable to your specific context. Use your best judgment, and feel free to disregard any points that don’t resonate with your experience or intent.',
        #                     align='J',
        #                     ln=True,
        #                 )
        #         pdf.set_font_recall()
        #         pdf.set_text_color((0, 0, 0))
        #         pdf.set_text_color((14, 95, 164))
        #         pdf.set_font('KulimPark', '', 12)
        #         pdf.multi_cell(
        #                     w=0,
        #                     text='''
        # While we often speak with the best of intentions, the impact of our words can land very differently—especially in high-stakes or emotionally charged conversations. This score is designed to bring awareness to that gap between intent and impact, helping us spot moments where language might unintentionally cause friction, confusion, or harm.

        # For each issue, we assign an **Friction Factor** (0–5) based on its potential to negatively affect the conversation or its participants:
        # 0. No noticeable impact.
        # 1. Minor, negligible effect.
        # 2. Mild tension or confusion.
        # 3. Noticeable disruption or discomfort.
        # 4. Strong negative effect, potential to derail or damage trust.
        # 5. Severe impact, likely to cause harm or major breakdown in communication.

        # Use this score not as a judgment, but as a tool for reflection and refinement.
        # ''',
        #                     align='J',
        #                     markdown=True,
        #                     ln=True,
        #                 )
        #         pdf.set_font_recall()
        #         pdf.set_text_color((0, 0, 0))

        #         speaker_map_name_to_id = {speaker_map[key]['name']: key for key in speaker_map.keys()}

        #         errors_and_corrections = speaker_analysis[speaker_map_name_to_id[speaker_name]][
        #             'personal_errors_and_corrections'
        #         ]

        #         # Loop through each topic in the errors and corrections
        #         for topic, corrections in errors_and_corrections.items():
        #             # Only show topics that have corrections (non-empty arrays)
        #             if corrections and len(corrections) > 0:
        #                 # Add topic title
        #                 pdf.add_title_2(text=topic.title())

        #                 # Loop through each correction for this topic
        #                 for correction in corrections:
        #                     if (
        #                         isinstance(correction, dict)
        #                         and 'observation' in correction
        #                         and 'friction_factor' in correction
        #                     ):
        #                         # Add gravity score
        #                         gravity_score = correction.get('friction_factor', 0)
        #                         pdf.set_font('KulimPark', 'b', 11)
        #                         pdf.multi_cell(
        #                             w=0,
        #                             text=f'Friction Factor: {gravity_score}/5',
        #                             align='L',
        #                             ln=True,
        #                         )

        #                         # Add observation
        #                         if correction.get('observation'):
        #                             pdf.set_font('KulimPark', '', 11)
        #                             pdf.multi_cell(
        #                                 w=0,
        #                                 text=f"**Observation:** {correction['observation']}",
        #                                 align='J',
        #                                 ln=True,
        #                                 markdown=True,
        #                             )

        #                         # Add advice if available
        #                         if correction.get('advice'):
        #                             pdf.set_font('KulimPark', 'i', 11)
        #                             pdf.multi_cell(
        #                                 w=0,
        #                                 text=f"**Advice:** {correction['advice']}",
        #                                 align='J',
        #                                 ln=True,
        #                                 markdown=True,
        #                             )

        #                         pdf.ln(3)  # Add spacing between corrections

        #                 pdf.ln(5)  # Add spacing between topics

        #         pdf.set_font_recall()

        # 5. Comments section
        pdf.add_comments_section()

        # Output the PDF
        pdf.output(speaker_filename)
        logging.info(f'...created personal pdf for {speaker_name} [%s].', speaker_filename)
