import json
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


def login(client: TestClient) -> None:
    response = client.post(
        '/login',
        data={'username': 'admin', 'password': 'test-password'},
        follow_redirects=False,
    )
    assert response.status_code == 303


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
        denied = client.post(
            '/login',
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
        assert 'PERSONAL_SUMMARY_TOKEN_ALEX' in alex
        assert 'QUOTE_TAG_ALEX' in alex
        assert 'ACTION_TITLE_ALEX' in alex
        assert 'SUPERPOWER_TOKEN_Alex_Rivera' in alex
        assert 'AMASE_TOKEN_Alex_Rivera' in alex
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
        assert 'GROUP_LONG_TOKEN' in group
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
        ):
            assert private not in group

        sent = client.post('/sessions/1/send', follow_redirects=True)
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
        again = client.post('/sessions/1/send', follow_redirects=False)
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
            data={'source': 'C', 'target': 'A'},
            follow_redirects=True,
        )
        assert 'value="C"' not in merged.text
        assert 'Speaker A' in merged.text
        transcript = json.loads(Path(app.state.db.get_session(1)['transcript_path']).read_text())
        assert {item['speaker'] for item in transcript['utterances']} == {'A', 'B'}

        client.post(
            f'{session_url}/speakers',
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
