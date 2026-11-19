import json
import logging
import re
from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient
from pypdf import PdfReader

from app.analysis import GeminiAnalysis
from app.config import Settings
from app.main import create_app
from tests.fakes import FakeLLM, FakeTranscription
from tests.synthetic import synthetic_transcript


def settings(tmp_path: Path, retention_days: int = 14) -> Settings:
    return Settings(
        admin_username='admin',
        admin_password='test-password',
        secret_key='test-secret-key-value',
        data_dir=tmp_path,
        retention_days=retention_days,
        email_provider='dry-run',
        smtp_host='',
        smtp_port=587,
        smtp_username='',
        smtp_password='',
        smtp_from='',
        smtp_use_tls=True,
        email_api_url='',
        email_api_key='',
        email_from='',
        feedback_url='',
        max_upload_mb=5,
        https_only=False,
    )


def pdf_text(payload: bytes) -> str:
    reader = PdfReader(BytesIO(payload))
    return '\n'.join(page.extract_text() or '' for page in reader.pages)


def csrf_from(html: str) -> str:
    match = re.search(r'name="csrf_token" value="([^"]+)"', html)
    assert match, 'csrf token missing from the page'
    return match.group(1)


def login(client: TestClient) -> str:
    page = client.get('/login')
    token = csrf_from(page.text)
    response = client.post(
        '/login',
        params={'csrf_token': token},
        data={'username': 'admin', 'password': 'test-password'},
        follow_redirects=False,
    )
    assert response.status_code == 303
    return token


def current_csrf(client: TestClient) -> str:
    return csrf_from(client.get('/').text)


def build_client(tmp_path: Path, retention_days: int = 14):
    transcription = FakeTranscription(synthetic_transcript())
    app = create_app(
        settings=settings(tmp_path, retention_days),
        transcription=transcription,
        analysis=GeminiAnalysis(llm=FakeLLM()),
        start_retention_thread=False,
    )
    return app, transcription


def upload(client: TestClient, consent: str = 'yes'):
    return client.post(
        '/sessions',
        params={'csrf_token': current_csrf(client)},
        data={
            'client': 'Northwind',
            'group_name': 'Cohort A',
            'session_name': 'Take a Position',
            'session_date': '2026-11-18',
            'speakers_expected': '3',
            'consent': consent,
        },
        files={'recording': ('session.wav', b'RIFFfake-audio', 'audio/wav')},
        follow_redirects=False,
    )


def test_login_is_required(tmp_path):
    app, _transcription = build_client(tmp_path)
    with TestClient(app) as client:
        home = client.get('/', follow_redirects=False)
        assert home.status_code == 303
        assert home.headers['location'] == '/login'
        page = client.get('/login')
        denied = client.post(
            '/login',
            params={'csrf_token': csrf_from(page.text)},
            data={'username': 'admin', 'password': 'wrong-password'},
            follow_redirects=False,
        )
        assert denied.status_code == 401
        health = client.get('/health')
        assert health.json()['status'] == 'ok'


def test_consent_is_required(tmp_path):
    app, _transcription = build_client(tmp_path)
    with TestClient(app) as client:
        login(client)
        response = upload(client, consent='')
        assert response.status_code == 400
        assert 'consented' in response.text
        assert app.state.db.list_sessions() == []


def test_full_flow_keeps_individual_reports_private(tmp_path):
    app, transcription = build_client(tmp_path)
    with TestClient(app) as client:
        login(client)
        created = upload(client)
        assert created.status_code == 303
        session_url = created.headers['location']
        page = client.get(session_url)
        assert 'Speaker A' in page.text
        assert 'onboarding plan' in page.text
        assert '0:00' in page.text

        saved = client.post(
            f'{session_url}/speakers',
            params={'csrf_token': current_csrf(client)},
            data={
                'action': 'generate',
                'label': ['A', 'B', 'C'],
                'name': ['Alex Rivera', 'Blake Chen', 'Casey Quinn'],
                'email': ['alex@example.com', 'blake@example.com', 'casey@example.com'],
                'role': ['participant', 'participant', 'facilitator'],
            },
            follow_redirects=True,
        )
        assert 'Preview every report' in saved.text
        assert 'Los Angeles' in saved.text
        assert '+00:00' not in saved.text
        assert 'alex@example.com' in saved.text
        assert 'does not receive a report' in saved.text
        assert 'Casey Quinn' in saved.text
        reports = app.state.db.list_reports(1)
        by_name = {report['filename']: report for report in reports}
        assert 'group.pdf' in by_name
        assert 'Alex Rivera.pdf' in by_name
        assert 'Blake Chen.pdf' in by_name
        assert 'Casey Quinn.pdf' not in by_name
        assert not list((tmp_path / 'outbox').glob('*.json'))

        def text_for(filename: str) -> str:
            report = by_name[filename]
            response = client.get(f"/sessions/1/reports/{report['id']}")
            assert response.status_code == 200
            assert response.headers['content-type'] == 'application/pdf'
            return pdf_text(response.content)

        alex = text_for('Alex Rivera.pdf')
        blake = text_for('Blake Chen.pdf')
        group = text_for('group.pdf')
        assert 'PERSONAL_SUMMARY_TOKEN_ALEX' not in alex
        assert 'QUOTE_TAG_ALEX' not in alex
        assert 'ACTION_TITLE_ALEX' not in alex
        assert 'SUPERPOWER_TOKEN_Alex_Rivera' not in alex
        assert 'AMASE_TOKEN_Alex_Rivera' in alex
        assert 'Your signature phrase:' in alex
        assert 'SIGNATURE_PHRASE_TOKEN' in alex
        assert 'Your signature phrase' in blake
        for foreign in (
            'PERSONAL_SUMMARY_TOKEN_BLAKE',
            'QUOTE_TAG_BLAKE',
            'ACTION_TITLE_BLAKE',
            'SUPERPOWER_TOKEN_Blake_Chen',
            'AMASE_TOKEN_Blake_Chen',
            'Blake Chen',
            'Casey Quinn',
        ):
            assert foreign not in alex
        assert 'PERSONAL_SUMMARY_TOKEN_ALEX' not in blake
        assert 'QUOTE_TAG_ALEX' not in blake
        assert 'Alex Rivera' not in blake
        assert 'GROUP_LONG_TOKEN' not in group
        assert 'GROUP_SHORT_TOKEN' not in group
        for private in (
            'PERSONAL_SUMMARY_TOKEN_ALEX',
            'PERSONAL_SUMMARY_TOKEN_BLAKE',
            'PERSONAL_SUMMARY_TOKEN_CASEY',
            'QUOTE_TAG_ALEX',
            'QUOTE_TAG_BLAKE',
            'ACTION_TITLE_ALEX',
            'ACTION_BODY_BLAKE',
            'SUPERPOWER_TOKEN_Alex_Rivera',
            'AMASE_TOKEN_Alex_Rivera',
            'ENNEAGRAM_TOKEN_Blake_Chen',
            'SIGNATURE_PHRASE_TOKEN',
        ):
            assert private not in group
        assert 'MasterMind' not in group
        assert 'Harkness Training Solutions' in group
        assert 'Top pairs, each direction' in group

        sent = client.post(
            '/sessions/1/send',
            params={'csrf_token': current_csrf(client)},
            follow_redirects=True,
        )
        assert 'saved locally' in sent.text
        messages = sorted((tmp_path / 'outbox').glob('*.json'))
        assert len(messages) == 2
        delivered = []
        for path in messages:
            record = json.loads(path.read_text())
            delivered.append(record)
            assert record['attachment_filenames'][1] == 'group.pdf'
            assert len(record['attachment_filenames']) == 2
        by_recipient = {record['to_email']: record for record in delivered}
        assert set(by_recipient) == {'alex@example.com', 'blake@example.com'}
        assert by_recipient['alex@example.com']['attachment_filenames'] == [
            'Alex Rivera.pdf',
            'group.pdf',
        ]
        assert by_recipient['blake@example.com']['attachment_filenames'] == [
            'Blake Chen.pdf',
            'group.pdf',
        ]
        assert 'Blake' not in by_recipient['alex@example.com']['body']
        assert 'Alex' not in by_recipient['blake@example.com']['body']
        again = client.post(
            '/sessions/1/send',
            params={'csrf_token': current_csrf(client)},
            follow_redirects=False,
        )
        assert again.status_code == 303
        assert len(list((tmp_path / 'outbox').glob('*.json'))) == 2
        assert transcription.deleted == []
        recording = Path(app.state.db.get_session(1)['recording_path'])
        assert recording.is_file()


def test_merge_and_retention_delete_sources(tmp_path):
    app, transcription = build_client(tmp_path, retention_days=0)
    with TestClient(app) as client:
        login(client)
        created = upload(client)
        session_url = created.headers['location']
        merged = client.post(
            f'{session_url}/merge',
            params={'csrf_token': current_csrf(client)},
            data={'source': 'C', 'target': 'A'},
            follow_redirects=True,
        )
        assert 'value="C"' not in merged.text
        assert 'Speaker A' in merged.text
        transcript = json.loads(Path(app.state.db.get_session(1)['transcript_path']).read_text())
        assert {item['speaker'] for item in transcript['utterances']} == {'A', 'B'}

        client.post(
            f'{session_url}/speakers',
            params={'csrf_token': current_csrf(client)},
            data={
                'action': 'generate',
                'label': ['A', 'B'],
                'name': ['Alex Rivera', 'Blake Chen'],
                'email': ['alex@example.com', 'blake@example.com'],
                'role': ['participant', 'participant'],
            },
            follow_redirects=True,
        )
        session = app.state.db.get_session(1)
        assert session['status'] == 'review'
        assert session['recording_path'] is None
        assert session['recording_deleted_at']
        assert session['transcript_deleted_at']
        assert not (tmp_path / 'sessions' / '1' / 'source.wav').exists()
        assert not (tmp_path / 'sessions' / '1' / 'transcript').exists()
        assert (tmp_path / 'sessions' / '1' / 'reports' / 'group.pdf').is_file()
        assert (tmp_path / 'sessions' / '1' / 'reports' / 'Alex Rivera.pdf').is_file()
        assert transcription.deleted == ['synthetic-transcript-001']


def test_csrf_token_is_required(tmp_path):
    app, _transcription = build_client(tmp_path)
    with TestClient(app) as client:
        page = client.get('/login')
        missing = client.post(
            '/login',
            data={'username': 'admin', 'password': 'test-password'},
            follow_redirects=False,
        )
        assert missing.status_code == 403
        token = csrf_from(page.text)
        accepted = client.post(
            '/login',
            params={'csrf_token': token},
            data={'username': 'admin', 'password': 'test-password'},
            follow_redirects=False,
        )
        assert accepted.status_code == 303


def test_names_allow_punctuation_and_name_the_speaker():
    from app.main import validate_assignments

    cleaned = validate_assignments(
        ['Alex Rivera (he/him)', 'Dr. Blake Chen'],
        ['alex@example.com', 'blake@example.com'],
        ['participant', 'participant'],
        ['A', 'B'],
    )
    assert cleaned[0]['name'] == 'Alex Rivera (he/him)'
    try:
        validate_assignments(['bad\nname'], ['alex@example.com'], ['participant'], ['A'])
    except ValueError as exc:
        assert 'Speaker A' in str(exc)
    else:
        raise AssertionError('a line break in a name should be rejected')
    unnamed = validate_assignments(['Alex'], [''], ['participant'], ['C'])
    assert unnamed[0]['name'] == 'Alex'
    assert unnamed[0]['email'] == ''
    skipped = validate_assignments([''], [''], ['participant'], ['C'], ['C'])
    assert skipped[0]['name_status'] == 'skipped'


def test_generation_records_progress_and_logs_the_traceback(tmp_path, caplog):
    class StoppingAnalysis:
        def analyze(self, **kwargs):
            kwargs['on_progress']('Writing speaker summaries')
            raise RuntimeError('stopped on purpose')

    app, _transcription = build_client(tmp_path)
    app.state.analysis = StoppingAnalysis()
    with TestClient(app) as client:
        login(client)
        created = upload(client)
        with caplog.at_level(logging.ERROR):
            client.post(
                f"{created.headers['location']}/speakers",
                params={'csrf_token': current_csrf(client)},
                data={
                    'action': 'generate',
                    'label': ['A', 'B', 'C'],
                    'name': ['Alex Rivera', 'Blake Chen', 'Casey Quinn'],
                    'email': ['alex@example.com', 'blake@example.com', 'casey@example.com'],
                    'role': ['participant', 'participant', 'facilitator'],
                },
                follow_redirects=True,
            )
    assert 'Traceback' in caplog.text
    assert 'stopped on purpose' in caplog.text
    session = app.state.db.get_session(1)
    assert session['status'] == 'error'
    assert session['progress'] == 'Writing speaker summaries'


def test_upload_form_accepts_an_optional_session_window(tmp_path):
    app, _transcription = build_client(tmp_path)
    with TestClient(app) as client:
        login(client)
        page = client.get('/upload')
        assert 'name="session_start"' in page.text
        assert 'name="session_end"' in page.text
        created = client.post(
            '/sessions',
            params={'csrf_token': current_csrf(client)},
            data={
                'client': 'Northwind',
                'group_name': 'Cohort A',
                'session_name': 'Take a Position',
                'session_date': '2026-11-18',
                'speakers_expected': '3',
                'consent': 'yes',
                'session_start': '0:30',
                'session_end': '1:48:00',
            },
            files={'recording': ('session.wav', b'RIFFfake-audio', 'audio/wav')},
            follow_redirects=False,
        )
        assert created.status_code == 303
        stored = app.state.db.get_session(1)
        assert stored['window_start_ms'] == 30_000
        assert stored['window_end_ms'] == (1 * 3600 + 48 * 60) * 1000


def test_samples_are_clipped_to_200_characters():
    from app.services import speaker_previews

    long_line = 'word ' * 80
    previews = speaker_previews(
        {'utterances': [{'speaker': 'A', 'text': long_line, 'start': 0, 'end': 1000}]}
    )
    assert len(previews[0]['samples'][0]['text']) <= 200
    assert previews[0]['samples'][0]['text'].endswith('…')


def test_deletion_time_is_shown_in_los_angeles():
    from app.main import format_pacific

    rendered = format_pacific('2026-11-19T00:00:00+00:00')
    assert rendered == 'Nov 18, 2026, 4:00 PM PST'


def test_retention_over_14_days_is_rejected(monkeypatch):
    monkeypatch.setenv('HTS_ADMIN_USERNAME', 'admin')
    monkeypatch.setenv('HTS_ADMIN_PASSWORD', 'test-password')
    monkeypatch.setenv('HTS_SECRET_KEY', 'test-secret-key-value')
    monkeypatch.setenv('RETENTION_DAYS', '15')
    from app.config import load_settings

    try:
        load_settings()
    except RuntimeError as exc:
        assert '14' in str(exc)
    else:
        raise AssertionError('RETENTION_DAYS=15 should be rejected')


def test_gemini_settings_are_checked_at_startup(monkeypatch):
    from app.config import load_settings

    monkeypatch.setenv('HTS_ADMIN_USERNAME', 'admin')
    monkeypatch.setenv('HTS_ADMIN_PASSWORD', 'test-password')
    monkeypatch.setenv('HTS_SECRET_KEY', 'test-secret-key-value')
    monkeypatch.delenv('GEMINI_MODEL', raising=False)
    monkeypatch.delenv('GEMINI_TIMEOUT_SECONDS', raising=False)
    settings = load_settings()
    assert settings.gemini_model == 'gemini-3.1-flash-lite'
    assert settings.gemini_timeout_seconds == 120
    monkeypatch.setenv('GEMINI_MODEL', 'gemini 2.5 flash')
    try:
        load_settings()
    except RuntimeError as exc:
        assert 'GEMINI_MODEL' in str(exc)
    else:
        raise AssertionError('a model id with spaces should be rejected')
    monkeypatch.setenv('GEMINI_MODEL', 'gemini-3.1-flash-lite')
    monkeypatch.setenv('GEMINI_TIMEOUT_SECONDS', '0')
    try:
        load_settings()
    except RuntimeError as exc:
        assert 'GEMINI_TIMEOUT_SECONDS' in str(exc)
    else:
        raise AssertionError('a zero Gemini timeout should be rejected')


def test_generate_session_substitutes_a_repeated_drama_turn(tmp_path):
    """The live story path, not a direct render call, replaces a repeated counter-moment."""
    from app.analysis import GeminiAnalysis
    from app.services import generate_session
    from lib.session_window import parse_clock_ms
    from tests.fakes import FakeLLM

    class SameTurnDramaLLM(FakeLLM):
        def _reply(self, text: str) -> str:
            reply = super()._reply(text)
            if 'BATCH:' in text and 'BATCH: group' not in text and 'FACTS' in text:
                payload = json.loads(reply)
                drama = payload.get('drama') or {}
                quotes = ((drama.get('patterns') or [{}])[0].get('quotes') or [{}])
                if quotes and quotes[0].get('text'):
                    drama['exception'] = {
                        'quote': quotes[0]['text'],
                        'ts': quotes[0]['ts'],
                        'what_happened': 'you named the plan',
                    }
                    payload['drama'] = drama
                return json.dumps(payload)
            return reply

    captured = {}

    class CaptureRenderer:
        def render(self, **kwargs):
            captured['stories'] = getattr(kwargs['bundle'], 'stories', {})
            return []

    rows = [
        {
            'speaker': 'S',
            'text': 'We should try the new onboarding plan before the hour ends.',
            'start': 0,
            'end': 4000,
        },
        {
            'speaker': 'S',
            'text': 'the budget can wait until Monday',
            'start': 8000,
            'end': 12000,
        },
        {
            'speaker': 'H',
            'text': 'Thank you all for coming to the session today.',
            'start': 16000,
            'end': 20000,
        },
    ]
    app, _transcription = build_client(tmp_path)
    app.state.analysis = GeminiAnalysis(llm=SameTurnDramaLLM())
    app.state.renderer = CaptureRenderer()
    db = app.state.db
    transcript_path = tmp_path / 'transcript.json'
    transcript_path.write_text(json.dumps({'utterances': rows}), encoding='utf-8')
    session_id = db.create_session(
        client='Northwind',
        group_name='Full Sail',
        session_name='Take a Position',
        session_date='2026-11-18',
        speakers_expected=2,
        recording_path='',
        recording_filename='',
        created_at='2026-11-19T00:00:00+00:00',
        window_end_ms=parse_clock_ms('1:00'),
    )
    db.update_session(session_id, transcript_path=str(transcript_path), status='naming')
    db.replace_speakers(session_id, [{'label': 'S', 'samples': []}, {'label': 'H', 'samples': []}])
    for speaker in db.list_speakers(session_id):
        if speaker['label'] == 'S':
            db.update_speaker(
                speaker['id'], name='Stephen', email='stephen@example.com',
                role='participant', name_status='confirmed',
            )
        else:
            db.update_speaker(
                speaker['id'], name='Host', email='', role='host', name_status='confirmed',
            )
    generate_session(
        db, app.state.analysis, app.state.renderer, app.state.transcription, app.state.settings, session_id,
    )
    drama = captured['stories']['personal']['S']['drama']
    assert 'budget can wait' in drama
    assert 'different turn' not in drama


def test_generate_session_stamps_the_host_before_the_closer(tmp_path):
    """Unannotated AssemblyAI rows plus the speaker map. Annie's thanks is the closer."""
    from app.interfaces import AnalysisBundle
    from app.services import generate_session
    from lib.session_window import closer_turn, parse_clock_ms

    captured = {}

    class CaptureAnalysis:
        def analyze(self, **kwargs):
            captured['transcript'] = kwargs['transcript']
            return AnalysisBundle({}, {}, {}, None, {})

    class QuietRenderer:
        def render(self, **kwargs):
            return []

    def turn(clock, speaker, text):
        start = parse_clock_ms(clock)
        return {'speaker': speaker, 'text': text, 'start': start, 'end': start + 20_000}

    rows = [
        turn(
            f'2:{30 + index}:00',
            'A',
            'The knowledge systems still hide the design until the workflow is named by the room.',
        )
        for index in range(8)
    ]
    rows.append(turn(
        '2:42:24',
        'A',
        'The knowledge systems still hide the design until the workflow is named. Thank you all for coming.',
    ))
    fragment = (
        'this era, like as users, as participants, et cetera the knowledge systems '
        'still hide the design until the workflow ends today.'
    )
    assert len(fragment.split()) == 21
    rows.append(turn('2:46:37', 'E', fragment))
    assert all('name' not in row and 'role' not in row for row in rows)

    app, _transcription = build_client(tmp_path)
    app.state.analysis = CaptureAnalysis()
    app.state.renderer = QuietRenderer()
    db = app.state.db
    transcript_path = tmp_path / 'transcript.json'
    transcript_path.write_text(json.dumps({'utterances': rows}), encoding='utf-8')
    session_id = db.create_session(
        client='Northwind',
        group_name='Cohort A',
        session_name='Take a Position',
        session_date='2026-11-18',
        speakers_expected=2,
        recording_path='',
        recording_filename='',
        created_at='2026-11-19T00:00:00+00:00',
        window_end_ms=parse_clock_ms('2:47:30'),
    )
    db.update_session(session_id, transcript_path=str(transcript_path), status='naming')
    db.replace_speakers(session_id, [{'label': 'A', 'samples': []}, {'label': 'E', 'samples': []}])
    for speaker in db.list_speakers(session_id):
        if speaker['label'] == 'A':
            db.update_speaker(
                speaker['id'], name='Dr. Annie Green', email='', role='host', name_status='confirmed',
            )
        else:
            db.update_speaker(
                speaker['id'], name='', email='', role='participant', name_status='skipped',
            )
    generate_session(
        db, app.state.analysis, app.state.renderer, app.state.transcription, app.state.settings, session_id,
    )
    stamped = captured['transcript']['utterances']
    host = next(row for row in stamped if row['start'] == parse_clock_ms('2:42:24'))
    assert host['role'] == 'host'
    assert host['name'] == 'Dr. Annie Green'
    later = next(row for row in stamped if row['start'] == parse_clock_ms('2:46:37'))
    assert later['name'] == 'Speaker E'
    assert later['role'] == 'participant'
    chosen = closer_turn(stamped)
    assert chosen['start'] == parse_clock_ms('2:42:24')
    assert chosen['text'].startswith('The knowledge systems still hide the design')
    assert 'Thank you all for coming' not in chosen['text']
    assert 'et cetera' not in chosen['text']
