import logging
import pandas as pd

logger_PIL = logging.getLogger('PIL')
logger_PIL.setLevel(logging.INFO)
logger_fontTools = logging.getLogger('fontTools')
logger_fontTools.setLevel(logging.WARNING)
logger_fpdf = logging.getLogger('fpdf')
logger_fpdf.setLevel(logging.INFO)

from pathlib import Path
from typing import TypeAlias

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

        # Font settings
        self.set_font('KulimPark', '', 12)
        self.set_font_to_recall()

    def header(self):
        self.set_font('KulimPark', '', 13)
        line_height = self.font_size * 2
        self.cell(w=0, text='MasterMind', align='L', h=line_height, markdown=True)
        self.set_font('KulimPark', '', 8)
        self.cell(w=0, text='jason@jasonrodneybrooks.com', align='R', h=line_height)
        self.ln(16)  # Line break

    # Page footer
    def footer(self):
        self.set_y(-15)  # Set position of the footer
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
        self.set_font(size=8)
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
        # subtitle='Feedback & actionable recommendations to **enhance your communication**.',
        subtitle='Feedback & actionable recommendations to enhance your communication.',
    )

    # Introduction, conversation short summary
    pdf.set_font_to_recall()
    pdf.ln(15)
    pdf.multi_cell(w=pdf.epw * 0.5, padding=0, text='', border='B', center=True)
    pdf.ln(0)
    pdf.set_font('KulimPark', '', 14)
    pdf.multi_cell(w=0, padding=(8, 10, 0), text=summaries['conversation_title'], align='C')
    pdf.ln(0)
    pdf.set_font('KulimPark-light', '', 12)
    pdf.multi_cell(w=0, padding=(6, 10, 8), text=summaries['conversation_summary_short'])
    pdf.ln(0)
    pdf.multi_cell(w=pdf.epw * 0.5, padding=0, text='', border='T', center=True)
    pdf.ln(5)
    pdf.set_font_recall()

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
        row.cell('Conversation Statistics', style=style_link, link=link_convo_stats)
        row.cell(
            'The measure of how evenly participants engaged in the conversation.', style=style_desc
        )
        row = table.row()
        row.cell('Speaker Statistics', style=style_link, link=link_speaker_stats)
        row.cell('Some interesting stats about the way each participant spoke.', style=style_desc)
        row = table.row()
        row.cell('Conversation Summary', style=style_link, link=link_convo_summ)
        row.cell('A brief recount of what happened during the MasterMind.', style=style_desc)
        row = table.row()
        row.cell('Key Contributions', style=style_link, link=link_key_contribs)
        row.cell('Three important ideas brought by each participant.', style=style_desc)
        row = table.row()
        row.cell('Action Items', style=style_link, link=link_action_items)
        row.cell('Actionable ideas discussed during the MasterMind.', style=style_desc)
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
    pdf.image(path_image_chord, x='C', w=pdf.epw * 0.5)
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
    pdf.multi_cell(w=0, text=summaries['conversation_summary_long'], align='J', ln=True)
    pdf.set_font_recall()

    # Key Contributions for each speaker
    pdf.add_page()
    pdf.add_title_1(text='Key Contributions', link_dest=link_key_contribs)
    for speaker, name in sorted(
        [(speaker, speaker_map[speaker]['name']) for speaker in speaker_map], key=lambda x: x[1]
    ):
        pdf.add_title_2(text=speaker_map[speaker]['name'])
        pdf.set_font_to_recall()
        for quote in speaker_map[speaker]['personal_key_contributions']:
            pdf.set_font('KulimPark', '', 12)
            pdf.set_x(pdf.l_margin + pdf.epw * 0.05)
            pdf.multi_cell(
                # padding=(0, 7),
                w=0,
                h=1.5 * pdf.font_size,
                ln=True,
                markdown=True,
                text=f'**{quote["tagline"]}**',
            )
            pdf.set_font_to_recall()
            pdf.set_font('KulimPark-light', 'i', 12)
            pdf.set_x(pdf.l_margin + pdf.epw * 0.05)
            pdf.multi_cell(
                w=0,
                align='J',
                ln=True,
                markdown=True,
                text=f'"{quote["quote"]}"',
            )
            pdf.set_font_recall()
            pdf.ln(5)
        pdf.set_font_recall()

    # Action Items for each speaker
    pdf.add_page()
    pdf.add_title_1(text='Action Items', link_dest=link_action_items)
    pdf.image(image_action_items, x='C', w=pdf.epw * 1 / 3)
    for speaker, name in sorted(
        [(speaker, speaker_map[speaker]['name']) for speaker in speaker_map], key=lambda x: x[1]
    ):
        pdf.add_title_2(text=speaker_map[speaker]['name'])
        for action in speaker_map[speaker]['personal_action_items']:
            pdf.set_x(pdf.l_margin + pdf.epw * 0.05)
            pdf.multi_cell(
                # padding=(0, 7),
                w=0,
                h=1.5 * pdf.font_size,
                ln=True,
                markdown=True,
                text=f'**{action["title"]}**',
            )
            pdf.set_font_to_recall()
            pdf.set_font('KulimPark-light', '', 12)
            pdf.set_x(pdf.l_margin + pdf.epw * 0.05)
            pdf.multi_cell(
                w=pdf.epw * 0.8,
                align='J',
                ln=True,
                markdown=True,
                text=f'{action["action"]}',
            )
            pdf.set_font_recall()
            pdf.ln(5)

    # Participant Summary for each speaker
    pdf.add_page()
    pdf.add_title_1(text='Participant Summary', link_dest=link_participant_summ)
    for speaker, name in sorted(
        [(speaker, speaker_map[speaker]['name']) for speaker in speaker_map], key=lambda x: x[1]
    ):
        pdf.add_title_2(text=speaker_map[speaker]['name'])
        pdf.multi_cell(
            w=0,
            # padding=(0, 10),
            # padding=(0, 0),
            text=f'{speaker_map[speaker]["personal_summary"]}',
            align='J',
            ln=True,
            markdown=True,
        )
        pdf.ln(5)

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
