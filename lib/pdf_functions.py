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
            speaker_filename = path_reports_dir.joinpath(f"{file_name_prefix} - {speaker_name}.pdf")
        else:
            speaker_filename = path_reports_dir.joinpath(f"{speaker_name}.pdf")

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
            w=0, padding=(6, 10, 8), text=speaker_analysis[speaker]['personal_super_power']
        )
        pdf.ln(0)
        pdf.multi_cell(w=pdf.epw * 0.5, padding=0, text='', border='T', center=True)
        pdf.ln(5)
        pdf.set_font_recall()

        # Table of Contents
        link_personal_summary = pdf.add_link()
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
            text=speaker_info['personal_summary'],
            align='J',
            ln=True,
            markdown=True,
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
                                text=str(speaker_explanations[column_name]),
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
