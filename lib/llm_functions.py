import json
import logging
from pathlib import Path
from typing import TypeAlias

import pandas as pd
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None


class ConversationSummarizer:

    stop_after_attempt = 15

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
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0,
        )  # Free quota/day = 1500

        self.prompt_templates = {
            'contribs': ChatPromptTemplate.from_template(
                self.get_prompt_template('personal_key_contributions.txt')
            ),
            'summary': ChatPromptTemplate.from_template(
                self.get_prompt_template('personal_summarize_participation.txt')
            ),
            'next_steps': ChatPromptTemplate.from_template(
                self.get_prompt_template('personal_action_items.txt')
            ),
            'errors_and_corrections': ChatPromptTemplate.from_template(
                self.get_prompt_template('personal_errors_and_corrections.txt')
            ),
            'convo_summ_long': ChatPromptTemplate.from_template(
                self.get_prompt_template('conversation_summarize_long.txt')
            ),
            'convo_summ_short': ChatPromptTemplate.from_template(
                self.get_prompt_template('conversation_summarize_short.txt')
            ),
            'convo_title': ChatPromptTemplate.from_template(
                self.get_prompt_template('conversation_create_title.txt')
            ),
        }

        parser_json = JsonOutputParser()
        parser_str = StrOutputParser()

        self.chains = {
            'contribs': self.prompt_templates['contribs'] | self.llm | parser_json,
            'summary': self.prompt_templates['summary'] | self.llm | parser_str,
            'next_steps': self.prompt_templates['next_steps'] | self.llm | parser_json,
            'errors_and_corrections': (
                self.prompt_templates['errors_and_corrections'] | self.llm | parser_str
            ),
            'convo_summ_long': self.prompt_templates['convo_summ_long'] | self.llm | parser_str,
            'convo_summ_short': self.prompt_templates['convo_summ_short'] | self.llm | parser_str,
            'convo_title': self.prompt_templates['convo_title'] | self.llm | parser_str,
        }

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
            input_txt += f'Summary [{speaker_map[speaker_id]["summary"]}]'
            input_txt += (
                ' ' + '_' * 82 + ' '
            )  # Workaround to allow visual spacing, in CLI could use """txt"""
            input_txt += f'Quotes [{speaker_map[speaker_id]["quotes"]}]'
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

    def speaker_summarize_and_map(
        self, utterances: JSON, speakers_expected: int, force_regen: bool = False
    ) -> JSON:
        """Creates a speaker mapping, it creates a map of each speaker with this information:
        - name
        - summary
        - quotes
        - actions

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
            speaker_map[speaker]['summary'] = (
                self.chains['summary']
                .with_retry(stop_after_attempt=self.stop_after_attempt)
                .invoke({'participation': _df_utter_reduced.iloc[i]['text']})
            )
            logging.info('......creating speaker contributions.')
            speaker_map[speaker]['quotes'] = (
                self.chains['contribs']
                .with_retry(stop_after_attempt=self.stop_after_attempt)
                .invoke({'participation': _df_utter_reduced.iloc[i]['text']})
            )
            logging.info('......creating speaker actions.')
            speaker_map[speaker]['actions'] = (
                self.chains['next_steps']
                .with_retry(stop_after_attempt=self.stop_after_attempt)
                .invoke({'participation': _df_utter_reduced.iloc[i]['text']})
            )
            logging.info('......creating speaker errors and corrections.')
            speaker_map[speaker]['errors_corrections'] = (
                self.chains['errors_and_corrections']
                .with_retry(stop_after_attempt=self.stop_after_attempt)
                .invoke({'participation': _df_utter_reduced.iloc[i]['text']})
            )

            print('*' * 20)
            print(f'speaker[{speaker}]')
            print("speaker_map[speaker]['name']", speaker_map[speaker]['name'])
            print(speaker_map[speaker]['summary'])
            print(speaker_map[speaker]['quotes'])
            print(speaker_map[speaker]['actions'])
            print(speaker_map[speaker]['errors_corrections'])
            print(flush=True)

        speaker_map = self._rename_speaker_map(speaker_map=speaker_map)

        with open(path_file_speaker_map, 'x', encoding='UTF-8') as file:
            logging.debug('...saving speaker mapping to json file [%s]', file.name)
            json.dump(speaker_map, fp=file, indent=4)

        return speaker_map

    def conversation_summarize(
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
        path_file_convo_summs = self.path_save_dir.joinpath('conversations_summaries.json')

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
        convo_title = (
            self.chains['convo_title']
            .with_retry(stop_after_attempt=self.stop_after_attempt)
            .invoke({'conversation': conversation_script})
        )
        logging.info('...creating conversation short summary.')
        convo_summ_short = (
            self.chains['convo_summ_short']
            .with_retry(stop_after_attempt=self.stop_after_attempt)
            .invoke({'conversation': conversation_script})
        )
        logging.info('...creating conversation long summary.')
        convo_summ_long = (
            self.chains['convo_summ_long']
            .with_retry(stop_after_attempt=self.stop_after_attempt)
            .invoke({'conversation': conversation_script})
        )

        summaries = {
            'speaker_count': speakers_expected,
            'title': convo_title,
            'short': convo_summ_short,
            'long': convo_summ_long,
        }
        with open(path_file_convo_summs, 'x', encoding='UTF-8') as file:
            logging.debug('...Saving json to file [%s].', path_file_convo_summs)
            json.dump(summaries, fp=file, indent=4)
        return summaries
