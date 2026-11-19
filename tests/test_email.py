import smtplib
from email.message import EmailMessage
from pathlib import Path

from app.config import Settings
from app.emailer import ApiEmailer, SmtpEmailer
from app.interfaces import OutgoingEmail


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        admin_username='admin',
        admin_password='test-password',
        secret_key='test-secret-key-value',
        data_dir=tmp_path,
        retention_days=14,
        email_provider='smtp',
        smtp_host='smtp.example',
        smtp_port=587,
        smtp_username='mailer',
        smtp_password='local-secret',
        smtp_from='reports@example.com',
        smtp_use_tls=True,
        email_api_url='https://api.example.test/emails',
        email_api_key='short-key',
        email_from='reports@example.com',
        feedback_url='',
        max_upload_mb=5,
        https_only=False,
    )


def test_smtp_sends_only_the_given_attachments(tmp_path, monkeypatch):
    attachment = tmp_path / 'Alex Rivera.pdf'
    group = tmp_path / 'group.pdf'
    attachment.write_bytes(b'%PDF-1.4 alex')
    group.write_bytes(b'%PDF-1.4 group')
    sent = {}

    class FakeSMTP:
        def __init__(self, host, port, timeout):
            sent['host'] = host
            sent['port'] = port

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def starttls(self):
            sent['tls'] = True

        def login(self, username, password):
            sent['login'] = username

        def send_message(self, mime: EmailMessage):
            sent['to'] = mime['To']
            sent['attachments'] = [part.get_filename() for part in mime.iter_attachments()]

    monkeypatch.setattr(smtplib, 'SMTP', FakeSMTP)
    SmtpEmailer(_settings(tmp_path)).send(
        OutgoingEmail(
            to_email='alex@example.com',
            to_name='Alex Rivera',
            subject='Your report',
            body='Hello Alex',
            attachments=[attachment, group],
        )
    )
    assert sent['to'] == 'alex@example.com'
    assert sent['attachments'] == ['Alex Rivera.pdf', 'group.pdf']
    assert sent['tls'] is True


def test_api_email_posts_only_the_given_attachments(tmp_path, monkeypatch):
    attachment = tmp_path / 'Blake Chen.pdf'
    group = tmp_path / 'group.pdf'
    attachment.write_bytes(b'%PDF blake')
    group.write_bytes(b'%PDF group')
    captured = {}

    class Response:
        status_code = 200
        text = '{"id":"msg_1"}'

        def json(self):
            return {'id': 'msg_1'}

    def post(url, json=None, headers=None, timeout=None):
        captured['url'] = url
        captured['json'] = json
        captured['headers'] = headers
        return Response()

    monkeypatch.setattr('app.emailer.requests.post', post)
    result = ApiEmailer(_settings(tmp_path)).send(
        OutgoingEmail(
            to_email='blake@example.com',
            to_name='Blake Chen',
            subject='Your report',
            body='Hello Blake',
            attachments=[attachment, group],
        )
    )
    assert result == 'msg_1'
    assert captured['json']['to'] == ['blake@example.com']
    assert [item['filename'] for item in captured['json']['attachments']] == [
        'Blake Chen.pdf',
        'group.pdf',
    ]
