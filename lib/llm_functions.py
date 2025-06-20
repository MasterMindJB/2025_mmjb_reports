import json
import logging
from pathlib import Path
from typing import TypeAlias

import pandas as pd
from langchain_core.language_models.chat_models import BaseChatModel

from langchain.globals import set_debug
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from google.api_core.exceptions import ResourceExhausted
from langchain_core.exceptions import OutputParserException

set_debug(False)  # Set to True to see communication information

JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None


class ConversationSummarizer:

    stop_after_attempt = 7

    parser_json = JsonOutputParser()
    parser_str = StrOutputParser()

    # The file name is kept without the '.txt' extension to declutter the code
    file_parser_map = {
        'personal_key_contributions': parser_json,
        'personal_summary': parser_str,
        'personal_action_items': parser_json,
        #
        'personal_errors_and_corrections': parser_json,
        'personal_framework_amase': parser_json,
        'personal_framework_enneagram': parser_json,
        'personal_framework_doshas': parser_json,
        #
        'conversation_summary_long': parser_str,
        'conversation_summary_short': parser_str,
        'conversation_title': parser_str,
    }

    def get_prompt_template(self, file_name: str) -> str:
        """Get a template by filename.

        Parameters
        ----------
        file_name : str
            File name.

        Returns
        -------
        str
            The template as a string.
        """
        path_file = self.path_prompt_templates.joinpath(file_name)
        with open(path_file, 'r', encoding='UTF-8') as file:
            return file.read()

    def __init__(self, path_save_dir: Path, path_prompt_templates: Path):
        self.path_save_dir = path_save_dir
        self.path_prompt_templates = path_prompt_templates

        # self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro") # Free quota/day = 50
        self.llm = ChatGoogleGenerativeAI(  # Free quota/day = 1500
            # model="gemini-1.5-flash",
            model="gemini-2.5-flash",
            # model="gemini-2.0-flash-lite",
            temperature=0,
        )

        # Create dictionary to store prompt templates for each type of conversation information generation
        self.prompt_templates = {}
        for partial_file_name, parser in self.file_parser_map.items():
            # Create a ChatPromptTemplate for each type of conversation information generation using the corresponding template file
            self.prompt_templates[partial_file_name] = ChatPromptTemplate.from_template(
                self.get_prompt_template(Path(partial_file_name).with_suffix('.txt'))
            )

        # Create dictionary to store processing chains for each type of conversation information generation
        self.chains: dict[str, BaseChatModel] = {}
        for partial_file_name, parser in self.file_parser_map.items():
            # Create a processing chain that combines:
            # 1. The prompt template
            # 2. The language model (LLM)
            # 3. The appropriate parser (JSON or String)
            self.chains[partial_file_name] = (
                self.prompt_templates[partial_file_name] | self.llm | parser
            )

        # Create dictionary to store retry-enabled chains for each type of conversation information generation
        self.chains_with_retry = {}
        for partial_file_name, parser in self.file_parser_map.items():
            # Add retry capability to each chain to handle potential API rate limits
            # Will retry up to stop_after_attempt times if ResourceExhausted error occurs
            self.chains_with_retry[partial_file_name] = self.chains[partial_file_name].with_retry(
                retry_if_exception_type=[
                    ResourceExhausted,  # Google only
                    OutputParserException,
                ],
                stop_after_attempt=self.stop_after_attempt,
            )

    def speaker_map_rename(self) -> JSON:
        """Shows some information from the speaker maps and asks for the names of the speakers so that the speaker map has real names.

        Returns
        -------
        JSON
            The speaker map with the new names.
        """
        path_file_speaker_map = self.path_save_dir.joinpath('speaker_map.json')

        logging.debug('Rename speakers in speaker map.')
        logging.debug('...loading from json file [%s].', path_file_speaker_map)
        if not path_file_speaker_map.is_file():
            logging.error(
                '...speaker map does not exist, create it with `ConversationSummarizer.speaker_map()`,  file[%s].',
                path_file_speaker_map,
            )
            raise FileNotFoundError(
                f'Speaker map does not exist, create it with `ConversationSummarizer.speaker_map()`,  file[{path_file_speaker_map}].'
            )

        with open(path_file_speaker_map, 'r', encoding='UTF-8') as file:
            speaker_map = json.load(file)

        added_names = []
        for speaker_id in speaker_map:
            input_txt = ''
            input_txt = f"Speaker[{speaker_id}] / Name[{speaker_map[speaker_id]['name']}]"
            input_txt += (
                ' ' + '_' * 82 + ' '
            )  # Workaround to allow visual spacing, in CLI could use """txt"""
            input_txt += f'Summary [{speaker_map[speaker_id]["personal_summary"]}]'
            input_txt += (
                ' ' + '_' * 82 + ' '
            )  # Workaround to allow visual spacing, in CLI could use """txt"""
            input_txt += f'Quotes [{speaker_map[speaker_id]["personal_key_contributions"]}]'
            input_response = input(input_txt)
            while (input_response == '') or (input_response in added_names):
                if input_response == '':
                    already_added_msg = ''
                    already_added_msg += 'ERROR: Name cannot be empty, add a valid name.'
                    already_added_msg += (
                        ' ' + '_' * 82 + ' '
                    )  # Workaround to allow visual spacing, in CLI could use """txt"""
                if input_response in added_names:
                    already_added_msg = ''
                    already_added_msg += (
                        f'ERROR: Name already added [{input_response}], choose another one.'
                    )
                    already_added_msg += (
                        ' ' + '_' * 82 + ' '
                    )  # Workaround to allow visual spacing, in CLI could use """txt"""
                input_response = input(already_added_msg + input_txt)

            added_names.append(input_response)
            speaker_map[speaker_id]['name'] = input_response

        with open(path_file_speaker_map, 'w', encoding='UTF-8') as file:
            logging.debug('...saving speaker map to json file [%s]', file.name)
            json.dump(speaker_map, fp=file, indent=4)

        return speaker_map

    def speaker_map(
        self,
        utterances: JSON,
        speakers_expected: int,
        force_regen: bool = False,
        force_regen_personal_summary: bool = False,
        force_regen_personal_key_contributions: bool = False,
        force_regen_personal_action_items: bool = False,
    ) -> JSON:
        """Creates or updates a speaker map containing detailed information for each speaker.

        This function processes conversation utterances to generate a comprehensive speaker map with personal summaries, key contributions, and action items for each participant.

        The function implements caching logic to avoid regenerating data when not necessary.

        The speaker map contains the following information for each speaker:
            - name: Speaker identifier (initially set to speaker ID, can be renamed later).
            - personal_summary: A concise summary of the speaker's participation and contributions.
            - personal_key_contributions: Key quotes and notable contributions from the speaker.
            - personal_action_items: Action items and commitments mentioned by the speaker.

        Parameters
        ----------
        utterances : JSON
            A JSON object containing conversation utterances from AssemblyAI transcript.
            Expected format: list of dictionaries with 'speaker' and 'text' fields.
        speakers_expected : int
            The expected number of speakers in the conversation. Used for validation
            and to determine if cached data needs regeneration.
        force_regen : bool, optional
            If True, forces complete regeneration of all speaker data, ignoring
            existing cached data. Default is False.
        force_regen_personal_summary : bool, optional
            If True, forces regeneration of personal summaries. Default is False.
        force_regen_personal_key_contributions : bool, optional
            If True, forces regeneration of personal key contributions. Default is False.
        force_regen_personal_action_items : bool, optional
            If True, forces regeneration of personal action items. Default is False.

        Returns
        -------
        JSON
            A dictionary mapping speaker IDs to their detailed information. Each speaker
            entry contains 'name', 'personal_summary', 'personal_key_contributions', and
            'personal_action_items' fields. The data is also saved to 'speaker_map.json'
            in the path_save_dir (when class is instantiated) for future use.

        Notes
        -----
        - The function automatically filters out 'UNK' speakers from the utterances.
        - Cached data is loaded from 'speaker_map.json' if available and valid.
        - Data is regenerated if the number of speakers doesn't match what is expected.
        - Individual regeneration flags allow selective updates of specific data types.
        - Uses LLM chains with retry logic for robust API calls.
        """
        path_file_speaker_map = self.path_save_dir.joinpath('speaker_map.json')

        speaker_map = None
        generate_all = False

        logging.debug('Creating speaker map...')
        if force_regen:
            speaker_map = {}
            generate_all = True
            logging.warning('...forcing regeneration (not taking into account if file exists).')
        elif not path_file_speaker_map.is_file():  # File doesn't exist
            speaker_map = {}
            generate_all = True
            logging.debug('...speaker map not created, creating.')
        else:  # File exists
            return_unchanged = True
            logging.debug(
                '...speaker map already created, loading from json file [%s].',
                path_file_speaker_map,
            )
            with open(path_file_speaker_map, 'r', encoding='UTF-8') as file:
                speaker_map = json.load(file)

            if speakers_expected != len(speaker_map.keys()):
                return_unchanged = False
                speaker_map = {}
                generate_all = True
                logging.warning('......speakers_expected differs from what was stored on file.')
                logging.warning('......forcing regeneration.')

            if force_regen_personal_summary:
                return_unchanged = False
                logging.debug('...forcing personal summary regeneration.')
            if force_regen_personal_key_contributions:
                return_unchanged = False
                logging.debug('...forcing personal key contributions regeneration.')
            if force_regen_personal_action_items:
                return_unchanged = False
                logging.debug('...forcing personal action items regeneration.')

            if return_unchanged:
                return speaker_map

        # Create a dataframe from the utterances
        _df_utter = pd.DataFrame.from_dict(utterances)

        # Remove speaker 'UNK', from _df_utter
        UNK_index = _df_utter.query('speaker == "UNK"').index
        _df_utter.drop(index=UNK_index, inplace=True)

        _df_utter_reduced = (
            _df_utter[['speaker', 'text']].groupby(['speaker']).agg({'text': lambda x: ' '.join(x)})
        )

        # Process all speakers
        for i in range(_df_utter_reduced.shape[0]):
            speaker = _df_utter_reduced.index[i]
            logging.debug('...processing speaker [%s].', speaker)

            _tmp_speaker = speaker_map.get(speaker, {})

            name = _tmp_speaker.get('name', speaker)

            logging.debug('name: %s', name)

            if generate_all or force_regen_personal_summary:
                logging.debug('......creating speaker summary.')
                _tmp_speaker['personal_summary'] = self.chains_with_retry[
                    'personal_summary'
                ].invoke({'participation': _df_utter_reduced.iloc[i]['text']})
                logging.info('personal_summary: %s', _tmp_speaker['personal_summary'])

            if generate_all or force_regen_personal_key_contributions:
                logging.debug('......creating speaker contributions.')
                _tmp_speaker['personal_key_contributions'] = self.chains_with_retry[
                    'personal_key_contributions'
                ].invoke({'participation': _df_utter_reduced.iloc[i]['text']})
                logging.info(
                    'personal_key_contributions: %s', _tmp_speaker['personal_key_contributions']
                )

            if generate_all or force_regen_personal_action_items:
                logging.debug('......creating speaker actions.')
                _tmp_speaker['personal_action_items'] = self.chains_with_retry[
                    'personal_action_items'
                ].invoke({'participation': _df_utter_reduced.iloc[i]['text']})
                logging.info('personal_action_items: %s', _tmp_speaker['personal_action_items'])

            _tmp_speaker_map_sorted = {}
            _tmp_speaker_map_sorted['name'] = name
            _tmp_speaker_map_sorted['personal_summary'] = _tmp_speaker['personal_summary']
            _tmp_speaker_map_sorted['personal_key_contributions'] = _tmp_speaker['personal_key_contributions']
            _tmp_speaker_map_sorted['personal_action_items'] = _tmp_speaker['personal_action_items']

            # Fusion _tmp_speaker with speaker_map[speaker]
            speaker_map[speaker] = _tmp_speaker_map_sorted

        with open(path_file_speaker_map, 'w', encoding='UTF-8') as file:
            logging.debug('...saving speaker map to json file [%s]', file.name)
            json.dump(speaker_map, fp=file, indent=4)

        return speaker_map

    def conversation_summaries(
        self,
        conversation_script: str,
        speakers_expected: int,
        force_regen: bool = False,
        force_regen_conversation_title: bool = False,
        force_regen_conversation_summary_short: bool = False,
        force_regen_conversation_summary_long: bool = False,
    ) -> JSON:
        """Creates or updates comprehensive conversation summaries and metadata.

        This function processes a conversation script to generate multiple levels of
        conversation analysis including a title, short summary, and long summary.
        The function implements caching logic to avoid regenerating data when not necessary.

        The conversation summaries contain the following information:
            - speaker_count: The number of participants in the conversation.
            - conversation_title: The conversation generated title.
            - conversation_summary_short: The conversation generated short summary.
            - conversation_summary_long: The conversation generated long summary.

        Parameters
        ----------
        conversation_script : str
            The conversation script to be analyzed.
        speakers_expected : int
            The expected number of speakers in the conversation. Used for validation
            and to determine if cached data needs regeneration.
        force_regen : bool, optional
            If True, forces complete regeneration of all conversation summaries, ignoring existing cached data. Default is False.
        force_regen_conversation_title : bool, optional
            If True, forces regeneration of the conversation title. Default is False.
        force_regen_conversation_summary_short : bool, optional
            If True, forces regeneration of the short summary. Default is False.
        force_regen_conversation_summary_long : bool, optional
            If True, forces regeneration of the long summary. Default is False.

        Returns
        -------
        JSON
            A dictionary containing conversation metadata and summaries with the following
            structure:
            {
                'speaker_count': int,
                'conversation_title': str,
                'conversation_summary_short': str,
                'conversation_summary_long': str
            }
            The data is saved to 'conversation_summaries.json' in the path_save_dir
            (when class is instantiated) for future use.

        Notes
        -----
        - Cached data is loaded from 'conversation_summaries.json' if available and valid.
        - Data is regenerated if the number of speakers doesn't match what is expected.
        - Individual regeneration flags allow selective updates of specific summary types.
        - Uses LLM chains with retry logic for robust API calls.
        - The summaries are generated using different prompt templates optimized for
          each type of analysis (title, short summary, long summary).
        """
        path_file_convo_summs = self.path_save_dir.joinpath('conversation_summaries.json')

        summaries = None
        generate_all = False

        logging.debug('Creating conversation summaries...')
        if force_regen:
            summaries = {}
            generate_all = True
            logging.warning('...forcing regeneration (not taking into account if file exists).')
        elif not path_file_convo_summs.is_file():  # File doesn't exist
            summaries = {}
            generate_all = True
            logging.debug('...conversation summaries not created, creating.')
        else:  # File exists
            return_unchanged = True
            logging.debug(
                '...conversation summaries already created, loading from json file [%s].',
                path_file_convo_summs,
            )
            with open(path_file_convo_summs, 'r', encoding='UTF-8') as file:
                summaries = json.load(file)

            if speakers_expected != summaries['speaker_count']:
                return_unchanged = False
                summaries = {}
                generate_all = True
                logging.warning('......speakers_expected differs from what was stored on file.')
                logging.warning('......forcing regeneration.')

            if force_regen_conversation_title:
                return_unchanged = False
                logging.debug('...forcing conversation title regeneration.')
            if force_regen_conversation_summary_short:
                return_unchanged = False
                logging.debug('...forcing conversation summary short regeneration.')
            if force_regen_conversation_summary_long:
                return_unchanged = False
                logging.debug('...forcing conversation summary long regeneration.')

            if return_unchanged:
                return summaries

        summaries['speaker_count'] = speakers_expected

        # Initialize missing fields to avoid KeyError when individual force flags are True
        if 'conversation_title' not in summaries:
            summaries['conversation_title'] = ''
        if 'conversation_summary_short' not in summaries:
            summaries['conversation_summary_short'] = ''
        if 'conversation_summary_long' not in summaries:
            summaries['conversation_summary_long'] = ''

        if 'conversation_title' not in summaries or generate_all or force_regen_conversation_title:
            logging.debug('...creating conversation title.')
            summaries['conversation_title'] = self.chains_with_retry['conversation_title'].invoke(
                {'conversation': conversation_script}
            )

        if (
            'conversation_summary_short' not in summaries
            or generate_all
            or force_regen_conversation_summary_short
        ):
            logging.debug('...creating conversation short summary.')
            summaries['conversation_summary_short'] = self.chains_with_retry[
                'conversation_summary_short'
            ].invoke({'conversation': conversation_script})

        if (
            'conversation_summary_long' not in summaries
            or generate_all
            or force_regen_conversation_summary_long
        ):
            logging.debug('...creating conversation long summary.')
            summaries['conversation_summary_long'] = self.chains_with_retry[
                'conversation_summary_long'
            ].invoke({'conversation': conversation_script})
        
        summaries_sorted = {}
        summaries_sorted['speaker_count'] = summaries['speaker_count']
        summaries_sorted['conversation_title'] = summaries['conversation_title']
        summaries_sorted['conversation_summary_short'] = summaries['conversation_summary_short']
        summaries_sorted['conversation_summary_long'] = summaries['conversation_summary_long']
        summaries = summaries_sorted

        with open(path_file_convo_summs, 'w', encoding='UTF-8') as file:
            logging.debug('...Saving json to file [%s].', path_file_convo_summs)
            json.dump(summaries, fp=file, indent=4)
        return summaries

    def speaker_analysis(
        self,
        df_speaker_stats: pd.DataFrame,
        df_nice_stats_data: pd.DataFrame,
        conversation_script: str,
        speakers_expected: int,
        force_regen: bool = False,
        force_regen_personal_errors_and_corrections: bool = False,
        force_regen_personal_framework_amase: bool = False,
        force_regen_personal_framework_enneagram: bool = False,
        force_regen_personal_framework_doshas: bool = False,
    ) -> JSON:
        """Creates or updates detailed psychological and behavioral analysis for each speaker.

        This function performs comprehensive analysis of each speaker's behavior, communication
        patterns, and psychological frameworks based on conversation data and statistical metrics.
        The function implements caching logic to avoid regenerating data when not necessary.

        The speaker analysis contains the following information for each speaker:
            - name: The speaker's name.
            - personal_framework_amase: Analysis using the AMASE framework.
            - personal_framework_enneagram: Analysis using the Enneagram framework.
            - personal_framework_doshas: Analysis using Ayurvedic doshas framework.
            - personal_errors_and_corrections: Identification and analysis of communication errors and corrections.

        Parameters
        ----------
        df_speaker_stats : pd.DataFrame
            DataFrame containing speaker statistics and metrics.
        df_nice_stats_data : pd.DataFrame
            DataFrame containing formatted statistical data.
        conversation_script : str
            The conversation script to be analyzed.
        speakers_expected : int
            The expected number of speakers in the conversation. Used for validation
            and to determine if cached data needs regeneration.
        force_regen : bool, optional
            If True, forces complete regeneration of all speaker analysis data, ignoring
            existing cached data. Default is False.
        force_regen_personal_errors_and_corrections : bool, optional
            If True, forces regeneration of personal errors and corrections analysis. Default is False.
        force_regen_personal_framework_amase : bool, optional
            If True, forces regeneration of AMASE framework analysis. Default is False.
        force_regen_personal_framework_enneagram : bool, optional
            If True, forces regeneration of Enneagram framework analysis. Default is False.
        force_regen_personal_framework_doshas : bool, optional
            If True, forces regeneration of Ayurvedic doshas framework analysis. Default is False.

        Returns
        -------
        JSON
            A dictionary mapping speaker IDs to their detailed analysis with the following
            structure for each speaker:
            {
                'name': str,
                'personal_framework_amase': dict,
                'personal_framework_enneagram': dict,
                'personal_framework_doshas': dict,
                'personal_errors_and_corrections': dict
            }
            The data is saved to 'speaker_analysis.json' in the path_save_dir (when class is instantiated) for future use.

        Notes
        -----
        - Cached data is loaded from 'speaker_analysis.json' if available and valid.
        - Data is regenerated if the number of speakers doesn't match what is expected.
        - Individual regeneration flags allow selective updates of specific analysis types.
        - Uses LLM chains with retry logic for robust API calls.
        - Statistical data is formatted as markdown tables for LLM processing.
        - Each framework analysis uses specialized prompt templates for accurate assessment.
        """
        path_file_speaker_analysis = self.path_save_dir.joinpath('speaker_analysis.json')

        speaker_analysis = None
        generate_all = False

        logging.debug('Creating speaker analysis...')
        if force_regen:
            speaker_analysis = {}
            generate_all = True
            logging.warning('...forcing regeneration (not taking into account if file exists).')
        elif not path_file_speaker_analysis.is_file():  # File doesn't exist
            speaker_analysis = {}
            generate_all = True
            logging.debug('...speaker analysis not created, creating.')
        else:  # File exists
            return_unchanged = True
            logging.debug(
                '...speaker analysis already created, loading from json file [%s].',
                path_file_speaker_analysis,
            )
            with open(path_file_speaker_analysis, 'r', encoding='UTF-8') as file:
                speaker_analysis = json.load(file)

            if speakers_expected != len(speaker_analysis.keys()):
                return_unchanged = False
                speaker_analysis = {}
                generate_all = True
                logging.warning('......speakers_expected differs from what was stored on file.')
                logging.warning('......forcing regeneration.')

            if force_regen_personal_errors_and_corrections:
                return_unchanged = False
                logging.debug('...forcing personal errors and corrections regeneration.')
            if force_regen_personal_framework_amase:
                return_unchanged = False
                logging.debug('...forcing personal framework amase regeneration.')
            if force_regen_personal_framework_enneagram:
                return_unchanged = False
                logging.debug('...forcing personal framework enneagram regeneration.')
            if force_regen_personal_framework_doshas:
                return_unchanged = False
                logging.debug('...forcing personal framework doshas regeneration.')

            if return_unchanged:
                return speaker_analysis

        # Process all speakers
        for speaker in df_speaker_stats.index:
            speaker_id = df_speaker_stats.loc[speaker]['speaker']

            logging.debug('...processing speaker [%s].', speaker_id)

            _tmp_speaker = speaker_analysis.get(speaker_id, {})

            # Initialize missing fields to avoid KeyError when individual force flags are True
            if 'personal_framework_amase' not in _tmp_speaker:
                _tmp_speaker['personal_framework_amase'] = {}
            if 'personal_framework_enneagram' not in _tmp_speaker:
                _tmp_speaker['personal_framework_enneagram'] = {}
            if 'personal_framework_doshas' not in _tmp_speaker:
                _tmp_speaker['personal_framework_doshas'] = {}
            if 'personal_errors_and_corrections' not in _tmp_speaker:
                _tmp_speaker['personal_errors_and_corrections'] = {}

            logging.debug('...name: %s.', speaker)

            # TODO: avoid column selection here, do it when calling this function
            _stats_markdown = (
                df_nice_stats_data[
                    [
                        # 'Action Items',
                        # 'Idea Generation',
                        'Questions Asked',
                        'Speaking Time (%)',
                        'Total Interactions',
                        # 'Unique Words % (Idea Density)',
                        'Word Count',
                        'Words Per Minute (WPM)',
                    ]
                ]
                .to_markdown()
            )

            if generate_all or force_regen_personal_framework_amase:
                logging.debug('......creating personal framework amase.')
                _tmp_speaker['personal_framework_amase'] = self.chains_with_retry[
                    'personal_framework_amase'
                ].invoke(
                    {
                        'stats_markdown': _stats_markdown,
                        'name': speaker,
                        'conversation_script': conversation_script,
                    }
                )
                logging.info(
                    'personal_framework_amase: %s', _tmp_speaker['personal_framework_amase']
                )

            if generate_all or force_regen_personal_framework_enneagram:
                logging.debug('......creating personal framework enneagram.')
                _tmp_speaker['personal_framework_enneagram'] = self.chains_with_retry[
                    'personal_framework_enneagram'
                ].invoke(
                    {
                        'stats_markdown': _stats_markdown,
                        'name': speaker,
                        'conversation_script': conversation_script,
                    }
                )
                logging.info(
                    'personal_framework_enneagram: %s', _tmp_speaker['personal_framework_enneagram']
                )

            if generate_all or force_regen_personal_framework_doshas:
                logging.debug('......creating personal framework doshas.')
                _tmp_speaker['personal_framework_doshas'] = self.chains_with_retry[
                    'personal_framework_doshas'
                ].invoke(
                    {
                        'stats_markdown': _stats_markdown,
                        'name': speaker,
                        'conversation_script': conversation_script,
                    }
                )
                logging.info(
                    'personal_framework_doshas: %s', _tmp_speaker['personal_framework_doshas']
                )

            if generate_all or force_regen_personal_errors_and_corrections:
                logging.debug('......creating speaker errors and corrections.')
                _tmp_speaker['personal_errors_and_corrections'] = self.chains_with_retry[
                    'personal_errors_and_corrections'
                ].invoke(
                    {
                        # 'participation': df_speaker_stats.loc[speaker, 'text'],
                        'name': speaker,
                        'conversation_script': conversation_script,
                    }
                )
                logging.info(
                    'personal_errors_and_corrections: %s',
                    _tmp_speaker['personal_errors_and_corrections'],
                )

            _tmp_speaker_sorted = {}
            _tmp_speaker_sorted['name'] = speaker
            _tmp_speaker_sorted['personal_framework_amase'] = _tmp_speaker['personal_framework_amase']
            _tmp_speaker_sorted['personal_framework_enneagram'] = _tmp_speaker['personal_framework_enneagram']
            _tmp_speaker_sorted['personal_framework_doshas'] = _tmp_speaker['personal_framework_doshas']
            _tmp_speaker_sorted['personal_errors_and_corrections'] = _tmp_speaker['personal_errors_and_corrections']

            # Fusion _tmp_speaker with speaker_analysis[speaker]
            speaker_analysis[speaker_id] = _tmp_speaker_sorted

        with open(path_file_speaker_analysis, 'w', encoding='UTF-8') as file:
            logging.debug('...saving speaker analysis to json file [%s]', file.name)
            json.dump(speaker_analysis, fp=file, indent=4)

        return speaker_analysis
