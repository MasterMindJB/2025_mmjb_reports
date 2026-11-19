"""Email delivery. The default provider writes the message locally and sends nothing."""

from __future__ import annotations

import base64
import json
import smtplib
from email.message import EmailMessage as MimeMessage
from pathlib import Path

import requests

from app.config import Settings
from app.interfaces import OutgoingEmail


class DryRunEmailer:
    name = 'dry-run'

    def __init__(self, outbox: Path):
        self.outbox = Path(outbox)
        self.outbox.mkdir(parents=True, exist_ok=True)

    def send(self, message: OutgoingEmail) -> str:
        stamp = f'{len(list(self.outbox.glob("*.json"))) + 1:04d}'
        safe = ''.join(ch if ch.isalnum() else '-' for ch in message.to_email)[:40]
        path = self.outbox / f'{stamp}-{safe}.json'
        record = {
            'to_email': message.to_email,
            'to_name': message.to_name,
            'subject': message.subject,
            'body': message.body,
            'attachment_filenames': [item.name for item in message.attachments],
            'provider': self.name,
        }
        path.write_text(json.dumps(record, indent=2), encoding='utf-8')
        return str(path)


class SmtpEmailer:
    name = 'smtp'

    def __init__(self, settings: Settings):
        self.settings = settings

    def send(self, message: OutgoingEmail) -> str:
        if not self.settings.smtp_host or not self.settings.smtp_from:
            raise RuntimeError('SMTP is not configured. Set SMTP_HOST and SMTP_FROM.')
        mime = MimeMessage()
        mime['From'] = self.settings.smtp_from
        mime['To'] = message.to_email
        mime['Subject'] = message.subject
        mime.set_content(message.body)
        for attachment in message.attachments:
            mime.add_attachment(
                attachment.read_bytes(),
                maintype='application',
                subtype='pdf',
                filename=attachment.name,
            )
        with smtplib.SMTP(self.settings.smtp_host, self.settings.smtp_port, timeout=30) as client:
            if self.settings.smtp_use_tls:
                client.starttls()
            if self.settings.smtp_username:
                client.login(self.settings.smtp_username, self.settings.smtp_password)
            client.send_message(mime)
        return f'smtp:{message.to_email}'


class ApiEmailer:
    """Transactional JSON API. The body matches Resend's email endpoint."""

    name = 'api'

    def __init__(self, settings: Settings):
        self.settings = settings

    def send(self, message: OutgoingEmail) -> str:
        if not self.settings.email_api_url or not self.settings.email_api_key or not self.settings.email_from:
            raise RuntimeError('Email API is not configured. Set EMAIL_API_URL, EMAIL_API_KEY, and EMAIL_FROM.')
        payload = {
            'from': self.settings.email_from,
            'to': [message.to_email],
            'subject': message.subject,
            'text': message.body,
            'attachments': [
                {
                    'filename': attachment.name,
                    'content': base64.b64encode(attachment.read_bytes()).decode('ascii'),
                }
                for attachment in message.attachments
            ],
        }
        response = requests.post(
            self.settings.email_api_url,
            json=payload,
            headers={
                'Authorization': f'Bearer {self.settings.email_api_key}',
                'Content-Type': 'application/json',
            },
            timeout=(10, 60),
        )
        if response.status_code >= 400:
            raise RuntimeError(
                f'Email API failed with HTTP {response.status_code}: {(response.text or "")[:200]}'
            )
        try:
            body = response.json()
        except ValueError:
            return f'api:{response.status_code}'
        return str(body.get('id') or f'api:{response.status_code}')


def build_emailer(settings: Settings):
    if settings.email_provider == 'smtp':
        return SmtpEmailer(settings)
    if settings.email_provider == 'api':
        return ApiEmailer(settings)
    return DryRunEmailer(settings.data_dir / 'outbox')
