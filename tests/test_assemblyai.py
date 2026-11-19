import json
from pathlib import Path

import pytest

from lib.assemblyai_helper_functions import delete_transcript, get_audio_transcript


class _Response:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload
        self.text = payload if isinstance(payload, str) else json.dumps(payload)

    def json(self):
        if isinstance(self._payload, str):
            raise ValueError('not json')
        return self._payload


def _paths(tmp_path: Path):
    audio = tmp_path / 'clip.wav'
    audio.write_bytes(b'RIFFfake')
    return (
        audio,
        tmp_path / 'upload.json',
        tmp_path / 'create.json',
        tmp_path / 'transcript.json',
    )


def _completed(speakers_expected=2):
    return {
        'id': 'trx-1',
        'status': 'completed',
        'speakers_expected': speakers_expected,
        'utterances': [{'speaker': 'A', 'text': 'Hello there.', 'start': 0, 'end': 1000}],
    }


def test_http_error_is_not_cached_and_stops_after_three_uploads(tmp_path, monkeypatch):
    posts = []

    def post(url, **kwargs):
        posts.append(url)
        return _Response(500, {'error': 'upstream'})

    monkeypatch.setenv('ASSEMBLYAI_API_KEY', 'test-key')
    monkeypatch.setattr('lib.assemblyai_helper_functions.requests.post', post)
    audio, upload, create, transcript = _paths(tmp_path)
    with pytest.raises(RuntimeError, match='3 attempts'):
        get_audio_transcript(audio, upload, create, transcript, speakers_expected=2)
    assert len(posts) == 3
    assert all(url.endswith('/upload') for url in posts)
    assert list(tmp_path.glob('*.json')) == []


def test_create_errors_retry_three_times_and_reuse_a_good_upload(tmp_path, monkeypatch):
    posts = []

    def post(url, **kwargs):
        posts.append(url)
        if url.endswith('/upload'):
            return _Response(200, {'upload_url': 'https://cdn.example/audio'})
        return _Response(400, {'error': 'rejected'})

    monkeypatch.setenv('ASSEMBLYAI_API_KEY', 'test-key')
    monkeypatch.setattr('lib.assemblyai_helper_functions.requests.post', post)
    audio, upload, create, transcript = _paths(tmp_path)
    with pytest.raises(RuntimeError, match='3 attempts'):
        get_audio_transcript(audio, upload, create, transcript, speakers_expected=2)
    assert sum(url.endswith('/transcript') for url in posts) == 3
    assert sum(url.endswith('/upload') for url in posts) == 1
    assert not create.exists()
    assert not transcript.exists()
    cached = json.loads(upload.read_text())
    assert cached['upload_url'].startswith('https://')
    assert 'error' not in cached


def test_poll_error_is_not_cached(tmp_path, monkeypatch):
    def post(url, **kwargs):
        if url.endswith('/upload'):
            return _Response(200, {'upload_url': 'https://cdn.example/audio'})
        return _Response(200, {'id': 'trx-err', 'status': 'queued'})

    def get(url, **kwargs):
        return _Response(200, {'id': 'trx-err', 'status': 'error', 'error': 'File not found'})

    monkeypatch.setenv('ASSEMBLYAI_API_KEY', 'test-key')
    monkeypatch.setattr('lib.assemblyai_helper_functions.requests.post', post)
    monkeypatch.setattr('lib.assemblyai_helper_functions.requests.get', get)
    monkeypatch.setattr('lib.assemblyai_helper_functions._sleep', lambda _seconds: None)
    audio, upload, create, transcript = _paths(tmp_path)
    with pytest.raises(RuntimeError, match='3 attempts'):
        get_audio_transcript(audio, upload, create, transcript, speakers_expected=2, max_poll_seconds=5)
    assert not transcript.exists()


def test_success_caches_only_a_completed_transcript(tmp_path, monkeypatch):
    polls = {'n': 0}

    def post(url, **kwargs):
        if url.endswith('/upload'):
            return _Response(200, {'upload_url': 'https://cdn.example/audio'})
        return _Response(200, {'id': 'trx-ok', 'status': 'queued'})

    def get(url, **kwargs):
        polls['n'] += 1
        if polls['n'] == 1:
            return _Response(200, {'id': 'trx-ok', 'status': 'processing'})
        payload = _completed(2)
        payload['id'] = 'trx-ok'
        return _Response(200, payload)

    monkeypatch.setenv('ASSEMBLYAI_API_KEY', 'test-key')
    monkeypatch.setattr('lib.assemblyai_helper_functions.requests.post', post)
    monkeypatch.setattr('lib.assemblyai_helper_functions.requests.get', get)
    monkeypatch.setattr('lib.assemblyai_helper_functions._sleep', lambda _seconds: None)
    audio, upload, create, transcript = _paths(tmp_path)
    result = get_audio_transcript(
        audio, upload, create, transcript, speakers_expected=2, poll_interval_seconds=0
    )
    assert result['status'] == 'completed'
    assert json.loads(transcript.read_text())['id'] == 'trx-ok'
    cached_again = get_audio_transcript(audio, upload, create, transcript, speakers_expected=2)
    assert cached_again['id'] == 'trx-ok'
    assert polls['n'] == 2


def test_speaker_count_mismatch_does_not_keep_the_transcript(tmp_path, monkeypatch):
    creates = []

    def post(url, **kwargs):
        if url.endswith('/upload'):
            return _Response(200, {'upload_url': 'https://cdn.example/audio'})
        creates.append(1)
        return _Response(200, {'id': f'trx-{len(creates)}', 'status': 'queued'})

    def get(url, **kwargs):
        return _Response(200, _completed(speakers_expected=9))

    monkeypatch.setenv('ASSEMBLYAI_API_KEY', 'test-key')
    monkeypatch.setattr('lib.assemblyai_helper_functions.requests.post', post)
    monkeypatch.setattr('lib.assemblyai_helper_functions.requests.get', get)
    monkeypatch.setattr('lib.assemblyai_helper_functions._sleep', lambda _seconds: None)
    audio, upload, create, transcript = _paths(tmp_path)
    with pytest.raises(RuntimeError, match='3 attempts'):
        get_audio_transcript(audio, upload, create, transcript, speakers_expected=2)
    assert len(creates) == 3
    assert not transcript.exists()


def test_polling_times_out(tmp_path, monkeypatch):
    clock = {'now': 0}

    def monotonic():
        clock['now'] += 100
        return clock['now']

    def post(url, **kwargs):
        if url.endswith('/upload'):
            return _Response(200, {'upload_url': 'https://cdn.example/audio'})
        return _Response(200, {'id': 'trx-slow', 'status': 'queued'})

    def get(url, **kwargs):
        return _Response(200, {'id': 'trx-slow', 'status': 'processing'})

    monkeypatch.setenv('ASSEMBLYAI_API_KEY', 'test-key')
    monkeypatch.setattr('lib.assemblyai_helper_functions.requests.post', post)
    monkeypatch.setattr('lib.assemblyai_helper_functions.requests.get', get)
    monkeypatch.setattr('lib.assemblyai_helper_functions._sleep', lambda _seconds: None)
    monkeypatch.setattr('lib.assemblyai_helper_functions.monotonic', monotonic)
    audio, upload, create, transcript = _paths(tmp_path)
    with pytest.raises(RuntimeError, match='timed out|3 attempts'):
        get_audio_transcript(audio, upload, create, transcript, speakers_expected=2, max_poll_seconds=30)
    assert not transcript.exists()


def test_invalid_cached_upload_is_discarded(tmp_path, monkeypatch):
    posts = []

    def post(url, **kwargs):
        posts.append(url)
        if url.endswith('/upload'):
            return _Response(200, {'upload_url': 'https://cdn.example/audio'})
        return _Response(200, {'id': 'trx-ok', 'status': 'queued'})

    def get(url, **kwargs):
        return _Response(200, _completed(2))

    monkeypatch.setenv('ASSEMBLYAI_API_KEY', 'test-key')
    monkeypatch.setattr('lib.assemblyai_helper_functions.requests.post', post)
    monkeypatch.setattr('lib.assemblyai_helper_functions.requests.get', get)
    audio, upload, create, transcript = _paths(tmp_path)
    upload.write_text(json.dumps({'error': 'cached failure'}), encoding='utf-8')
    result = get_audio_transcript(audio, upload, create, transcript, speakers_expected=2)
    assert result['status'] == 'completed'
    assert json.loads(upload.read_text())['upload_url'].startswith('https://')
    assert posts


def test_delete_transcript_accepts_404(monkeypatch):
    calls = []

    def delete(url, **kwargs):
        calls.append(url)
        return _Response(404, {'error': 'not found'})

    monkeypatch.setenv('ASSEMBLYAI_API_KEY', 'test-key')
    monkeypatch.setattr('lib.assemblyai_helper_functions.requests.delete', delete)
    delete_transcript('trx-gone')
    assert calls and calls[0].endswith('/trx-gone')


def test_delete_transcript_raises_on_server_error(monkeypatch):
    def delete(url, **kwargs):
        return _Response(503, {'error': 'unavailable'})

    monkeypatch.setenv('ASSEMBLYAI_API_KEY', 'test-key')
    monkeypatch.setattr('lib.assemblyai_helper_functions.requests.delete', delete)
    with pytest.raises(RuntimeError, match='503'):
        delete_transcript('trx-1')
