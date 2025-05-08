import json
import logging
import os
from pathlib import Path
from time import sleep

import requests


def _yield_file_chunks(file_path, chunk_size=5242880):
    """Reads a file in chunks."""
    if not Path(file_path).is_file():
        raise ValueError(f'File doesn\'t exist: {file_path}.')
    with open(file_path, 'rb') as file:
        while True:
            data = file.read(chunk_size)
            if not data:
                break
            yield data


def _upload_file(
    path_file: Path,
    path_response_upload_json: Path,
    force: bool = False,
):
    logging.info('File uploading...')
    if path_response_upload_json.is_file() and not force:
        with open(path_response_upload_json, 'r', encoding='UTF-8') as file:
            logging.debug(
                '...file already uploaded, loading from json file [%s].', path_response_upload_json
            )
            return json.load(file)
    else:
        if path_response_upload_json.is_file() and force:
            logging.info('...file already uploaded, forcing reupload.')
            path_response_upload_json.unlink(missing_ok=True)
        else:
            logging.info('...file not uploaded, uploading.')

        # The actual uploading
        headers = {'authorization': os.getenv('ASSEMBLYAI_API_KEY')}
        response_upload = requests.post(
            'https://api.assemblyai.com/v2/upload',
            headers=headers,
            data=_yield_file_chunks(path_file),
            timeout=None,
        )
        json_response = response_upload.json()
        with open(path_response_upload_json, 'x', encoding='UTF-8') as file:
            logging.debug('...saving to json file [%s].', path_response_upload_json)
            json.dump(json_response, fp=file)
        return json_response


def _create_transcript(
    url_upload: str,
    path_response_create_json: Path,
    speakers_expected: int,
    force=False,
):
    logging.info('Asking for transcript generation...')
    if path_response_create_json.is_file() and not force:
        with open(path_response_create_json, 'r', encoding='UTF-8') as file:
            logging.debug(
                '...transcript already generated, loading from json file [%s].',
                path_response_create_json,
            )
            return json.load(file)
    else:
        if path_response_create_json.is_file() and force:
            logging.info('...transcript already generated, forcing regeneration.')
            path_response_create_json.unlink(missing_ok=True)
        else:
            logging.info('...transcript not generated, asking to generate.')

        # The actual transcript generation request
        json_payload = {
            'audio_url': f'{url_upload}',
            'iab_categories': True,
            'auto_chapters': True,
            'sentiment_analysis': True,
            'entity_detection': True,
            'auto_highlights': True,
            'speaker_labels': True,
            'speakers_expected': speakers_expected,
        }
        headers = {
            'authorization': os.getenv('ASSEMBLYAI_API_KEY'),
            'content-type': 'application/json',
        }
        response_gen_transcript = requests.post(
            'https://api.assemblyai.com/v2/transcript',
            json=json_payload,
            headers=headers,
            timeout=None,
        )
        json_response = response_gen_transcript.json()
        with open(path_response_create_json, 'x', encoding='UTF-8') as file:
            logging.debug('...Saving json to file [%s].', path_response_create_json)
            json.dump(json_response, fp=file)
        return json_response


def _obtain_transcript(
    id_transcript: str,
    path_response_transcript_json: Path,
    force=False,
):
    logging.info('Obtaining transcript...')
    if path_response_transcript_json.is_file() and not force:
        with open(path_response_transcript_json, 'r', encoding='UTF-8') as file:
            logging.debug(
                '...transcript already obtained, loading transcript from json file [%s].',
                path_response_transcript_json,
            )
            return json.load(file)
    else:
        if path_response_transcript_json.is_file() and force:
            logging.info('...transcript already obtained, forcing request.')
            path_response_transcript_json.unlink(missing_ok=True)
        else:
            logging.info('...transcript not obtained, obtaining.')

        # The actual obtaining
        headers = {
            'authorization': os.getenv('ASSEMBLYAI_API_KEY'),
        }

        while True:
            response_get_transcript = requests.get(
                f'https://api.assemblyai.com/v2/transcript/{id_transcript}',
                headers=headers,
                timeout=None,
            ).json()

            if response_get_transcript['status'] == 'completed':
                break

            elif response_get_transcript['status'] == 'error':
                raise RuntimeError(f"Transcription failed: {response_get_transcript['error']}")

            else:
                sleep(3)

        with open(path_response_transcript_json, 'x', encoding='UTF-8') as file:
            logging.debug('...saving transcript to json file [%s]', path_response_transcript_json)
            json.dump(response_get_transcript, fp=file)

        return response_get_transcript


def get_audio_transcript(
    path_file: Path,
    path_response_upload_json: Path,
    path_response_create_json: Path,
    path_response_transcript_json: Path,
    speakers_expected: int,
    force=False,
):
    if not Path(path_file).is_file():
        raise ValueError(f'File doesn\'t exist: {path_file}.')

    logging.info('\x1b[0;1mCreating Audio transcript\x1b[0m')

    force_regen = False
    if path_response_transcript_json.is_file() and not force:
        with open(path_response_transcript_json, 'r', encoding='UTF-8') as file:
            logging.info('Transcript already stored, retrieving from json file.')
            response_transcript_json = json.load(file)
        if speakers_expected == response_transcript_json['speakers_expected']:
            return response_transcript_json
        else:
            logging.warning(
                '...speakers_expected differs from what was in the stored transcript, regenerating.'
            )
            force_regen = True

    response_upload_json = _upload_file(path_file, path_response_upload_json, force)
    url_upload = response_upload_json['upload_url']
    logging.debug('Uploaded file URL: %s', url_upload)
    force = force or force_regen
    response_create_json = _create_transcript(
        url_upload, path_response_create_json, speakers_expected, force
    )
    id_transcript = response_create_json['id']
    logging.debug('Transcript id: %s', id_transcript)
    response_transcript_json = _obtain_transcript(
        id_transcript, path_response_transcript_json, force
    )
    return response_transcript_json
