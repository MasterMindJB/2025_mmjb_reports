import io
import logging
from pathlib import Path
from typing import Optional, TypeAlias

import pandas as pd

from lib.data_story import group_story, personal_story
from lib.story_pdf import fit_personal_report, write_group_story

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

        poppins_dir = Path(__file__).resolve().parents[1] / 'assets' / 'fonts' / 'Poppins'
        regular = poppins_dir / 'Poppins-Regular.ttf'
        semibold = poppins_dir / 'Poppins-SemiBold.ttf'
        self.add_font('Poppins', style='', fname=regular)
        self.add_font('Poppins', style='b', fname=semibold)
        self.add_font('Poppins', style='i', fname=regular)
        self.add_font('Poppins', style='bi', fname=semibold)

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

        self.add_font('PoppinsMedium', '', fname=poppins_dir / 'Poppins-Medium.ttf')
        self.brand_canvas = False
        self._brand_pages: set[int] = set()
        self.content_bottom: dict[int, float] = {}

    def output(self, name: str = '', *args, **kwargs):
        MM_PDF.latest_ratios = self.page_fill_ratios()
        return super().output(name, *args, **kwargs)

    def note_y(self) -> None:
        page = int(self.page)
        y = float(self.get_y())
        # The footer parks the cursor below the content box, which used to report ~1.02.
        if y >= self.h - self.b_margin - 0.5:
            return
        self.content_bottom[page] = max(self.content_bottom.get(page, self.t_margin), y)

    def page_fill_ratios(self) -> list[float]:
        if float(self.get_y()) < self.h - self.b_margin - 0.5:
            self.note_y()
        usable = max(self.h - self.t_margin - self.b_margin, 1)
        limit = self.h - self.b_margin
        ratios = []
        for page in range(1, int(self.page) + 1):
            bottom = min(self.content_bottom.get(page, self.t_margin), limit)
            ratios.append(max(0.0, min(1.0, (bottom - self.t_margin) / usable)))
        return ratios

        # Font settings
        self.set_font('Poppins', '', 12)
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
        self.set_font('Poppins', '', 11)
        line_height = self.font_size * 2
        self.cell(w=self.epw * 0.72, text='Harkness Training Solutions', align='L', h=line_height)
        self.set_text_color(85, 98, 138) if branded else self.set_text_color(0, 0, 0)
        self.set_font('Poppins', '', 9)
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
            self.set_font('Poppins', 'U', 11)
            self.cell(
                0,
                10,
                text="Leave a Comment!",
                align='L',
                link=self.link_feedback,
            )
        self.set_font('Poppins', '', 11)
        self.cell(0, 10, f'{self.page_no()}/{{nb}}', align='R')  # Page number
        self.set_text_color(0, 0, 0)

    def add_title_document(self, title: str, subtitle: str = None):
        self.set_font_to_recall()
        self.set_font('Poppins', '', 29)
        self.cell(w=0, h=self.font_size * 1.4, text=title, align='C', ln=True)
        if subtitle is not None:
            self.set_font('Poppins', '', 14)
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
        self.set_font('Poppins', '', 25)
        if link_dest is not None:
            self.set_link(link_dest)
        self.cell(
            w=0, h=self.font_size * 2, text=text, align='L', markdown=True, ln=True, border='B'
        )
        self.ln(h=self.font_size / 2)
        self.set_font_recall()

    def add_title_2(self, text: str, link_dest: int = None):
        self.set_font_to_recall()
        self.set_font('Poppins', '', 18)
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
        # Headings are bold. Poppins has no bold face, so the table uses Poppins.
        self.set_font('Poppins', '', 8)
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
        self.set_font('Poppins', '', 20)
        self.multi_cell(
            w=0,
            padding=(60, 0, 2.5, 0),
            text="How can we improve?",
            align='C',
            ln=True,
            markdown=True,
        )
        self.set_font('Poppins', 'U', 26)
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
    rows = len(pairs or []) + 2
    needed = 28 + rows * 7
    usable = pdf.h - pdf.t_margin - pdf.b_margin
    if needed < usable and pdf.h - pdf.get_y() - pdf.b_margin < needed:
        pdf.add_page()
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
        pdf.set_font('Poppins', '', 12)
        return
    pdf.set_font('Poppins', '', 8)
    pdf.set_text_color(*ink)
    pdf.set_draw_color(*line)
    pdf.set_line_width(0.2)
    heading = FontFace(family='Poppins', emphasis='BOLD', size_pt=8, color=white, fill_color=navy)
    with pdf.table(
        text_align=['LEFT', 'LEFT', 'LEFT'],
        col_widths=(2.4, 2.3, 2.3),
        borders_layout='HORIZONTAL_LINES',
        line_height=pdf.font_size * 1.45,
        first_row_as_headings=True,
    ) as table:
        header = table.row()
        header.cell('Pair', style=heading)
        header.cell('One way', style=heading)
        header.cell('The other way', style=heading)
        for index, (name_a, name_b, first_way, other_way) in enumerate(pairs):
            from lib.names import table_name

            short_a = table_name(name_a)
            short_b = table_name(name_b)
            fill = light if index % 2 == 0 else white
            style = FontFace(family='Poppins', size_pt=8, color=ink, fill_color=fill)
            row = table.row()
            row.cell(f'{short_a} and {short_b}', style=style)
            row.cell(f'{short_a} then {short_b}: {first_way}', style=style)
            row.cell(f'{short_b} then {short_a}: {other_way}', style=style)
    pdf.set_draw_color(0, 0, 0)
    pdf.set_text_color(0, 0, 0)
    pdf.set_line_width(0.2)
    pdf.set_font('Poppins', '', 12)
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
        pdf.set_font('Poppins', '', 12)
        pdf.multi_cell(
            w=0,
            text='Not enough speech in this session to find a repeated phrase.',
            ln=True,
        )
        return
    bigram = signature.get('bigram', '')
    count = int(signature.get('count') or 0)
    pdf.add_title_1(text=f'Your signature phrase: \u201c{bigram}\u201d', link_dest=link_dest)
    pdf.set_font('Poppins', 'b', 12)
    times = 'once' if count == 1 else f'{count} times'
    pdf.multi_cell(
        w=0,
        text=f'Your most frequent two-word phrase in this session was \u201c{bigram}\u201d, used {times}.',
        ln=True,
    )
    pdf.ln(2)
    pdf.set_font('Poppins', '', 11)
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
        pdf.set_font('Poppins', '', 12)
        pdf.multi_cell(w=0, text='None noted.', ln=True)
        pdf.set_font_recall()
        return
    for quote in quotes:
        pdf.set_font('Poppins', '', 12)
        pdf.set_x(pdf.l_margin + pdf.epw * 0.05)
        pdf.multi_cell(
            w=0,
            h=1.5 * pdf.font_size,
            ln=True,
            markdown=True,
            text=f'**{_clean_llm_markdown(quote["tagline"])}**',
        )
        pdf.set_font('Poppins', 'i', 12)
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
        pdf.set_font('Poppins', '', 12)
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
        pdf.set_font('Poppins', '', 12)
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
    story_draft: dict | None = None,
    transcript: dict | None = None,
):
    logging.info('Creating PDF report.')

    pdf = MM_PDF(
        path_report_assets=path_report_assets,
        orientation='P',
        unit='mm',
        format='letter',
        link_feedback=link_feedback,
    )

    cast = 'Who talks to whom, and who nobody answered.'
    if chord_pairs:
        left, right, forward, back = chord_pairs[0][:4]
        cast = f'Who talks to whom: {left} and {right} lead with {int(forward) + int(back)}.'
    shares = []
    if df_nice_stats_data is not None and 'Speaking Time (%)' in getattr(df_nice_stats_data, 'columns', []):
        for name, value in df_nice_stats_data['Speaking Time (%)'].items():
            if str(name) == 'Average':
                continue
            shares.append((str(name), int(round(float(value or 0)))))
    names = {label: info.get('name') for label, info in speaker_map.items() if info.get('name')}
    story = group_story(
        summaries=summaries,
        participation=gini_reverse_interventions,
        interaction=gini_reverse_interactions,
        word_interaction=gini_reverse_interactions_both_wrd_cnt,
        cast=cast,
        draft=story_draft,
        transcript=transcript,
        names=names,
        pairs=chord_pairs,
        shares=shares,
    )
    write_group_story(pdf, story, path_image_chord, chord_pairs)
    if pdf.link_feedback:
        pdf.add_comments_section()
    pdf.output(path_file)
    logging.info('...created pdf [%s].', path_file)
    return



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
    transcript: dict | None = None,
    story_drafts: dict | None = None,
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

        story = personal_story(
            label=speaker,
            info=speaker_info,
            analysis=speaker_analysis.get(speaker) or {},
            transcript=transcript,
            stats_frame=df_nice_stats_data,
            framework={
                'amase_values': df_framework_amase_values,
                'amase_notes': df_framework_amase_explanation,
                'enneagram_values': df_framework_enneagram_values,
                'enneagram_notes': df_framework_enneagram_explanation,
                'dosha_values': df_framework_doshas_values,
                'dosha_notes': df_framework_doshas_explanation,
            },
            draft=(story_drafts or {}).get(speaker),
        )
        chart_dir = path_reports_dir / '_story' / pdf_filename_for(speaker_name).replace('.pdf', '')

        def make_pdf():
            return MM_PDF(
                path_report_assets=path_report_assets,
                orientation='P',
                unit='mm',
                format='letter',
                link_feedback=link_feedback,
            )

        pdf = fit_personal_report(story, chart_dir, make_pdf)
        if pdf.link_feedback:
            pdf.add_comments_section()
        pdf.output(speaker_filename)
        logging.info(f'...created personal pdf for {speaker_name} [%s].', speaker_filename)
