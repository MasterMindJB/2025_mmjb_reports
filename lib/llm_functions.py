import json
import logging
from pathlib import Path
from typing import TypeAlias

import pandas as pd
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None


def get_prompt_template(path_prompt_templates: Path, file_name: str):
    path_file = path_prompt_templates.joinpath(file_name)
    with open(path_file, 'r', encoding='UTF-8') as file:
        return file.read()


def rename_speaker_map(speaker_map: JSON):
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


def summarize_and_map_speaker(
    utterances: JSON, path_work_dir: Path, path_prompt_templates: Path, force: bool = False
):
    path_file_speaker_map = path_work_dir.joinpath('speaker_mapping.json')

    logging.info('Creating speaker mapping...')
    if path_file_speaker_map.is_file() and not force:
        with open(path_file_speaker_map, 'r', encoding='UTF-8') as file:
            logging.debug(
                '...speaker mapping already created, loading mapping from json file [%s]',
                file.name,
            )
            return json.load(file)
    else:
        if path_file_speaker_map.is_file() and force:
            logging.info('...speaker mapping already created, forcing recreation.')
            path_file_speaker_map.unlink(missing_ok=True)
        else:
            logging.info('...speaker mapping not created, creating.')

        _df_utter = pd.DataFrame.from_dict(utterances)
        # Remove speaker 'UNK', from _df_utter
        UNK_index = _df_utter.query('speaker == "UNK"').index
        _df_utter.drop(index=UNK_index, inplace=True)
        _df_speaker_stats = (
            _df_utter[['speaker', 'text']].groupby(['speaker']).agg({'text': lambda x: ' '.join(x)})
        )

        # llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro") # Free quota/day = 50
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", temperature=0
        )  # Free quota/day = 1500

        prompt_template_contribs = ChatPromptTemplate.from_template(
            get_prompt_template(path_prompt_templates, 'key_personal_contributions.txt')
        )
        prompt_template_summary = ChatPromptTemplate.from_template(
            get_prompt_template(path_prompt_templates, 'summarize_participation.txt')
        )
        prompt_template_next_steps = ChatPromptTemplate.from_template(
            get_prompt_template(path_prompt_templates, 'next_steps.txt')
        )

        parser_json = JsonOutputParser()
        parser_str = StrOutputParser()

        chain_contribs = prompt_template_contribs | llm | parser_json
        chain_summary = prompt_template_summary | llm | parser_str
        chain_next_steps = prompt_template_next_steps | llm | parser_json

        speaker_map = {}

        for i in range(_df_speaker_stats.shape[0]):
            speaker = _df_speaker_stats.index[i]
            logging.info('...processing speaker [%s].', speaker)
            speaker_map[speaker] = {}
            speaker_map[speaker]['name'] = speaker

            speaker_map[speaker]['summary'] = chain_summary.with_retry(
                stop_after_attempt=15
            ).invoke({'participation': _df_speaker_stats.iloc[i]['text']})

            speaker_map[speaker]['quotes'] = chain_contribs.with_retry(
                stop_after_attempt=15
            ).invoke({'participation': _df_speaker_stats.iloc[i]['text']})

            speaker_map[speaker]['actions'] = chain_next_steps.with_retry(
                stop_after_attempt=15
            ).invoke({'participation': _df_speaker_stats.iloc[i]['text']})

            print('*' * 20)
            print(f'speaker[{speaker}]')
            print("speaker_map[speaker]['name']", speaker_map[speaker]['name'])
            print(speaker_map[speaker]['summary'])
            print(speaker_map[speaker]['quotes'])
            print(speaker_map[speaker]['actions'])
            print()

        speaker_map = rename_speaker_map(speaker_map=speaker_map)

        with open(path_file_speaker_map, 'x', encoding='UTF-8') as file:
            logging.debug('...saving speaker mapping to json file [%s]', file.name)
            json.dump(speaker_map, fp=file, indent=4)

        return speaker_map
