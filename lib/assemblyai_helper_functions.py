import json
import logging
import os
from pathlib import Path
from time import monotonic
from time import sleep as _sleep
from typing import TypeAlias

import requests

JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None

# One create-transcript call per attempt. Three attempts, then stop.
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_MAX_POLL_SECONDS = 30 * 60
DEFAULT_POLL_INTERVAL_SECONDS = 10
_JSON_TIMEOUT = (10, 60)
_UPLOAD_TIMEOUT = (10, 300)


def _yield_file_chunks(file_path, chunk_size=5242880):
    """Reads a file in chunks."""
    if not Path(file_path).is_file():
        raise ValueError(f"File doesn't exist: {file_path}.")
    with open(file_path, 'rb') as file:
        while True:
            data = file.read(chunk_size)
            if not data:
                break
            yield data


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with open(temporary, 'w', encoding='UTF-8') as file:
        json.dump(payload, file, indent=4)
    temporary.replace(path)


def _load_json_cache(path: Path, accept) -> dict | None:
    """Return cached JSON only when `accept` says it is a real success payload."""
    if not path.is_file():
        return None
    try:
        with open(path, 'r', encoding='UTF-8') as file:
            payload = json.load(file)
    except (OSError, json.JSONDecodeError):
        path.unlink(missing_ok=True)
        return None
    if not isinstance(payload, dict) or not accept(payload):
        logging.warning('Ignoring invalid cached AssemblyAI response [%s].', path)
        path.unlink(missing_ok=True)
        return None
    return payload


def _read_json(response: requests.Response, what: str) -> dict:
    excerpt = (response.text or '')[:300]
    if response.status_code >= 400:
        raise RuntimeError(f'{what} failed with HTTP {response.status_code}: {excerpt}')
    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError(f'{what} returned non-JSON (HTTP {response.status_code})') from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f'{what} returned unexpected JSON')
    if payload.get('error'):
        raise RuntimeError(f'{what} error: {payload.get("error")}')
    return payload


def _api_key() -> str:
    key = os.getenv('ASSEMBLYAI_API_KEY')
    if not key:
        raise RuntimeError('ASSEMBLYAI_API_KEY is not set')
    return key


def _valid_upload(payload: dict) -> bool:
    return isinstance(payload.get('upload_url'), str) and bool(payload.get('upload_url')) and not payload.get('error')


def _valid_create(payload: dict) -> bool:
    return isinstance(payload.get('id'), str) and bool(payload.get('id')) and not payload.get('error')


def _valid_transcript(payload: dict) -> bool:
    return (
        payload.get('status') == 'completed'
        and isinstance(payload.get('utterances'), list)
        and not payload.get('error')
    )


def _upload_file(
    path_file: Path,
    path_response_upload_json: Path,
    force: bool = False,
):
    logging.info('File uploading...')
    if force and path_response_upload_json.is_file():
        logging.info('...file already uploaded but forcing reupload.')
        path_response_upload_json.unlink(missing_ok=True)
    cached = _load_json_cache(path_response_upload_json, _valid_upload)
    if cached is not None:
        logging.debug('...file already uploaded, loading from json file [%s].', path_response_upload_json)
        return cached

    logging.info('...file not uploaded, uploading.')
    response_upload = requests.post(
        'https://api.assemblyai.com/v2/upload',
        headers={'authorization': _api_key()},
        data=_yield_file_chunks(path_file),
        timeout=_UPLOAD_TIMEOUT,
    )
    json_response = _read_json(response_upload, 'AssemblyAI upload')
    if not _valid_upload(json_response):
        raise RuntimeError('AssemblyAI upload response did not include upload_url')
    _write_json(path_response_upload_json, json_response)
    return json_response


def _create_transcript(
    url_upload: str,
    path_response_create_json: Path,
    speakers_expected: int | None,
    force=False,
):
    logging.info('Asking for transcript generation...')
    if force and path_response_create_json.is_file():
        logging.info('...transcript already generated but forcing regeneration.')
        path_response_create_json.unlink(missing_ok=True)
    cached = _load_json_cache(path_response_create_json, _valid_create)
    if cached is not None:
        logging.debug(
            '...transcript already generated, loading from json file [%s].',
            path_response_create_json,
        )
        return cached

    logging.info('...transcript not generated, asking to generate.')
    json_payload = {
        'audio_url': f'{url_upload}',
        'speaker_labels': True,
    }
    if speakers_expected is not None:
        json_payload['speakers_expected'] = speakers_expected
    response_gen_transcript = requests.post(
        'https://api.assemblyai.com/v2/transcript',
        json=json_payload,
        headers={
            'authorization': _api_key(),
            'content-type': 'application/json',
        },
        timeout=_JSON_TIMEOUT,
    )
    json_response = _read_json(response_gen_transcript, 'AssemblyAI transcript create')
    if not _valid_create(json_response):
        raise RuntimeError('AssemblyAI transcript create response did not include id')
    _write_json(path_response_create_json, json_response)
    return json_response


def _obtain_transcript(
    id_transcript: str,
    path_response_transcript_json: Path,
    force=False,
    max_poll_seconds: int = DEFAULT_MAX_POLL_SECONDS,
    poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
):
    logging.info('Obtaining transcript...')
    if force and path_response_transcript_json.is_file():
        logging.info('...transcript already obtained but forcing request.')
        path_response_transcript_json.unlink(missing_ok=True)
    cached = _load_json_cache(path_response_transcript_json, _valid_transcript)
    if cached is not None:
        logging.debug(
            '...transcript already obtained, loading from json file [%s].',
            path_response_transcript_json,
        )
        return cached

    logging.info('...transcript not obtained, obtaining.')
    headers = {'authorization': _api_key()}
    deadline = monotonic() + max_poll_seconds
    while True:
        response = requests.get(
            f'https://api.assemblyai.com/v2/transcript/{id_transcript}',
            headers=headers,
            timeout=_JSON_TIMEOUT,
        )
        # A poll can legitimately be "processing" with no error field. HTTP failures
        # and explicit error payloads still raise, and those are not cached.
        if response.status_code >= 400:
            excerpt = (response.text or '')[:300]
            raise RuntimeError(
                f'AssemblyAI transcript poll failed with HTTP {response.status_code}: {excerpt}'
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError('AssemblyAI transcript poll returned non-JSON') from exc
        if not isinstance(payload, dict):
            raise RuntimeError('AssemblyAI transcript poll returned unexpected JSON')
        status = payload.get('status')
        if status == 'completed':
            if payload.get('error'):
                raise RuntimeError(f"Transcription failed: {payload.get('error')}")
            if not isinstance(payload.get('utterances'), list):
                raise RuntimeError('Completed transcript did not include utterances')
            _write_json(path_response_transcript_json, payload)
            return payload
        if status == 'error':
            raise RuntimeError(f"Transcription failed: {payload.get('error')}")
        if monotonic() >= deadline:
            raise RuntimeError(
                f'Transcription polling timed out after {max_poll_seconds} seconds '
                f'(last status: {status})'
            )
        _sleep(poll_interval_seconds)


def get_audio_transcript(
    path_file: Path,
    path_response_upload_json: Path,
    path_response_create_json: Path,
    path_response_transcript_json: Path,
    speakers_expected: int | None,
    force_upload: bool = False,
    force_regen: bool = False,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    max_poll_seconds: int = DEFAULT_MAX_POLL_SECONDS,
    poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
) -> JSON:
    """Upload audio, request a diarized transcript, and poll until it completes.

    At most `max_attempts` create calls are made. Error payloads are not cached.
    """
    if not Path(path_file).is_file():
        raise ValueError(f"File doesn't exist: {path_file}.")
    if max_attempts < 1:
        raise ValueError('max_attempts must be at least 1')
    if speakers_expected is not None and speakers_expected < 1:
        raise ValueError('speakers_expected must be a positive integer when set')

    last_error: Exception | None = None
    force_upload_now = force_upload
    force_regen_now = force_regen
    for attempt in range(1, max_attempts + 1):
        try:
            logging.info('Creating Audio transcript (attempt %s/%s)', attempt, max_attempts)
            response_upload_json = _upload_file(
                path_file, path_response_upload_json, force_upload_now
            )
            url_upload = response_upload_json['upload_url']
            logging.debug('Uploaded file URL: %s', url_upload)
            force_regen_final = force_upload_now or force_regen_now
            response_create_json = _create_transcript(
                url_upload,
                path_response_create_json,
                speakers_expected,
                force_regen_final,
            )
            id_transcript = response_create_json['id']
            logging.debug('Transcript id: %s', id_transcript)
            response_transcript_json = _obtain_transcript(
                id_transcript,
                path_response_transcript_json,
                force_regen_final,
                max_poll_seconds=max_poll_seconds,
                poll_interval_seconds=poll_interval_seconds,
            )
            reported = response_transcript_json.get('speakers_expected')
            if (
                speakers_expected is not None
                and reported is not None
                and reported != speakers_expected
            ):
                path_response_transcript_json.unlink(missing_ok=True)
                raise RuntimeError(
                    'speakers_expected differs from what was stored on file '
                    f'(requested {speakers_expected}, transcript has {reported})'
                )
            return response_transcript_json
        except Exception as exc:
            last_error = exc
            logging.warning('Transcription attempt %s/%s failed: %s', attempt, max_attempts, exc)
            if 'File not found' in str(exc):
                force_upload_now = True
            else:
                force_regen_now = True
            if attempt >= max_attempts:
                break
    raise RuntimeError(
        f'Transcription failed after {max_attempts} attempts: {last_error}'
    ) from last_error


def delete_transcript(transcript_id: str) -> None:
    """Delete a transcript from AssemblyAI. Missing transcripts are treated as gone."""
    if not transcript_id or not str(transcript_id).strip():
        raise ValueError('transcript_id is required')
    response = requests.delete(
        f'https://api.assemblyai.com/v2/transcript/{transcript_id}',
        headers={'authorization': _api_key()},
        timeout=_JSON_TIMEOUT,
    )
    if response.status_code == 404:
        return
    if response.status_code not in (200, 202, 204):
        excerpt = (response.text or '')[:300]
        raise RuntimeError(
            f'AssemblyAI transcript delete failed with HTTP {response.status_code}: {excerpt}'
        )
    check = requests.get(
        f'https://api.assemblyai.com/v2/transcript/{transcript_id}',
        headers={'authorization': _api_key()},
        timeout=_JSON_TIMEOUT,
    )
    if check.status_code == 404:
        return
    payload = {}
    try:
        payload = check.json()
    except ValueError:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    text = str(payload.get('text') or '')
    status = str(payload.get('status') or '').lower()
    confirmed = (
        status in {'deleted', 'redacted'}
        or payload.get('deleted') is True
        or 'redacted' in text.lower()
        or text.strip() in {'', 'redacted'}
    )
    if not confirmed:
        logging.warning(
            'AssemblyAI delete was not confirmed for %s: HTTP %s text=%s',
            transcript_id,
            check.status_code,
            text[:120],
        )
