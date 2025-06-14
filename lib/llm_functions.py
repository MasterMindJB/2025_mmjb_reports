import json
import logging
from pathlib import Path
from typing import TypeAlias, Dict

import pandas as pd
from langchain_core.language_models.chat_models import BaseChatModel

from langchain.globals import set_debug
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from google.api_core.exceptions import ResourceExhausted

set_debug(False)  # Set to True to see communication information

JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None


class ConversationSummarizer:

    stop_after_attempt = 15

    parser_json = JsonOutputParser()
    parser_str = StrOutputParser()

    # The file name is kept without the '.txt' extension to declutter the code
    file_parser_map = {
        'personal_key_contributions': parser_json,
        'personal_summary': parser_str,
        'personal_action_items': parser_json,
        #
        'personal_errors_and_corrections': parser_str,
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
            model="gemini-1.5-flash",
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
                ],
                stop_after_attempt=self.stop_after_attempt,
            )

    def _rename_speaker_map(self, speaker_map: JSON) -> JSON:
        """Shows some information from the speaker maps and asks for the names of the speakers so that the speaker map has real names.

        Parameters
        ----------
        speaker_map : JSON
            The speaker map.

        Returns
        -------
        JSON
            The speaker map with the new names.
        """
        added_names = []
        for speaker_id in speaker_map:
            input_txt = ''
            input_txt = f'Speaker[{speaker_id}]'
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

        return speaker_map

    def speaker_mapping(
        self, utterances: JSON, speakers_expected: int, force_regen: bool = False
    ) -> JSON:
        """Creates a speaker mapping, it creates a map of each speaker with this information:
        - name
        - personal_summary
        - personal_key_contributions
        - personal_action_items

        Parameters
        ----------
        utterances : JSON
            A JSON object created by assemblyai.
        speakers_expected : int
            The number of speakers expected.
        force_regen : bool, optional
            Force the regeneration, by default False.

        Returns
        -------
        JSON
            The JSON mapping.
        """
        path_file_speaker_map = self.path_save_dir.joinpath('speaker_mapping.json')

        logging.info('Creating speaker mapping...')
        if path_file_speaker_map.is_file() and not force_regen:
            with open(path_file_speaker_map, 'r', encoding='UTF-8') as file:
                logging.debug(
                    '...speaker mapping already created, loading mapping from json file [%s]',
                    file.name,
                )
                speaker_map = json.load(file)
                # check speakers_expected
                if speakers_expected == len(speaker_map.keys()):
                    return speaker_map
                else:
                    force_regen = True
                    logging.warning('...speakers_expected differs from what was stored on file.')

        if path_file_speaker_map.is_file() and force_regen:
            logging.info('...speaker mapping already created, but forcing recreation.')
            path_file_speaker_map.unlink(missing_ok=True)
        else:
            logging.info('...speaker mapping not created, creating.')

        _df_utter = pd.DataFrame.from_dict(utterances)

        # Remove speaker 'UNK', from _df_utter
        UNK_index = _df_utter.query('speaker == "UNK"').index
        _df_utter.drop(index=UNK_index, inplace=True)

        _df_utter_reduced = (
            _df_utter[['speaker', 'text']].groupby(['speaker']).agg({'text': lambda x: ' '.join(x)})
        )

        speaker_map = {}

        for i in range(_df_utter_reduced.shape[0]):
            speaker = _df_utter_reduced.index[i]
            logging.info('...processing speaker [%s].', speaker)
            speaker_map[speaker] = {}
            speaker_map[speaker]['name'] = speaker
            logging.info('......creating speaker summary.')
            speaker_map[speaker]['personal_summary'] = self.chains_with_retry[
                'personal_summary'
            ].invoke({'participation': _df_utter_reduced.iloc[i]['text']})
            logging.info('......creating speaker contributions.')
            speaker_map[speaker]['personal_key_contributions'] = self.chains_with_retry[
                'personal_key_contributions'
            ].invoke({'participation': _df_utter_reduced.iloc[i]['text']})
            logging.info('......creating speaker actions.')
            speaker_map[speaker]['personal_action_items'] = self.chains_with_retry[
                'personal_action_items'
            ].invoke({'participation': _df_utter_reduced.iloc[i]['text']})
            # logging.info('......creating speaker errors and corrections.')
            # speaker_map[speaker]['personal_errors_and_corrections'] = self.chains_with_retry[
            #     'personal_errors_and_corrections'
            # ].invoke({'participation': _df_utter_reduced.iloc[i]['text']})

            print('*' * 20)
            print(f'speaker[{speaker}]')
            print("speaker_map[speaker]['name']", speaker_map[speaker]['name'])
            print(speaker_map[speaker]['personal_summary'])
            print(speaker_map[speaker]['personal_key_contributions'])
            print(speaker_map[speaker]['personal_action_items'])
            # print(speaker_map[speaker]['personal_errors_and_corrections'])
            print(flush=True)

        speaker_map = self._rename_speaker_map(speaker_map=speaker_map)

        with open(path_file_speaker_map, 'x', encoding='UTF-8') as file:
            logging.debug('...saving speaker mapping to json file [%s]', file.name)
            json.dump(speaker_map, fp=file, indent=4)

        return speaker_map

    def conversation_summaries(
        self, conversation_script: str, speakers_expected: int, force_regen: bool = False
    ) -> JSON:
        """Creates a summary of the conversation.

        Parameters
        ----------
        conversation_script : str
            The conversation script.
        speakers_expected : int
            The number of speakers expected.
        force_regen : bool, optional
            Force the regeneration, by default False.

        Returns
        -------
        JSON
            A JSON object containing the summaries (long and short) and the speaker count.
        """
        path_file_convo_summs = self.path_save_dir.joinpath('conversation_summaries.json')

        logging.info('Creating conversation summaries...')
        if path_file_convo_summs.is_file() and not force_regen:
            with open(path_file_convo_summs, 'r', encoding='UTF-8') as file:
                logging.debug(
                    '...conversation summaries already created, loading  from json file [%s]',
                    file.name,
                )
                summaries = json.load(file)
                if speakers_expected == summaries['speaker_count']:
                    return summaries
                else:
                    force_regen = True
                    logging.warning('...speakers_expected differs from what was stored on file.')

        if path_file_convo_summs.is_file() and force_regen:
            logging.info('...conversation summaries already created, forcing recreation.')
            path_file_convo_summs.unlink(missing_ok=True)
        else:
            logging.info('...conversation summaries not created, creating.')

        logging.info('...creating conversation title.')
        convo_title = self.chains_with_retry['conversation_title'].invoke(
            {'conversation': conversation_script}
        )
        logging.info('...creating conversation short summary.')
        convo_summ_short = self.chains_with_retry['conversation_summary_short'].invoke(
            {'conversation': conversation_script}
        )
        logging.info('...creating conversation long summary.')
        convo_summ_long = self.chains_with_retry['conversation_summary_long'].invoke(
            {'conversation': conversation_script}
        )

        summaries = {
            'speaker_count': speakers_expected,
            'conversation_title': convo_title,
            'conversation_summary_short': convo_summ_short,
            'conversation_summary_long': convo_summ_long,
        }
        with open(path_file_convo_summs, 'x', encoding='UTF-8') as file:
            logging.debug('...Saving json to file [%s].', path_file_convo_summs)
            json.dump(summaries, fp=file, indent=4)
        return summaries
