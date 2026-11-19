"""Runtime configuration. Secrets come from the environment, never from the repo."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    admin_username: str
    admin_password: str
    secret_key: str
    data_dir: Path
    retention_days: int
    email_provider: str
    smtp_host: str
    smtp_port: int
    smtp_username: str
    smtp_password: str
    smtp_from: str
    smtp_use_tls: bool
    email_api_url: str
    email_api_key: str
    email_from: str
    feedback_url: str
    max_upload_mb: int
    https_only: bool

    @property
    def sends_email(self) -> bool:
        return self.email_provider in {'smtp', 'api'}


def _flag(name: str, default: str = 'false') -> bool:
    return os.getenv(name, default).strip().lower() in {'1', 'true', 'yes', 'on'}


def load_settings() -> Settings:
    load_dotenv()
    username = os.getenv('HTS_ADMIN_USERNAME', '').strip()
    password = os.getenv('HTS_ADMIN_PASSWORD', '')
    secret = os.getenv('HTS_SECRET_KEY', '')
    if not username:
        raise RuntimeError('Set HTS_ADMIN_USERNAME before starting the app.')
    if len(password) < 8:
        raise RuntimeError('Set HTS_ADMIN_PASSWORD to at least 8 characters.')
    if len(secret) < 16:
        raise RuntimeError('Set HTS_SECRET_KEY to at least 16 characters.')
    try:
        retention_days = int(os.getenv('RETENTION_DAYS', '14'))
    except ValueError as exc:
        raise RuntimeError('RETENTION_DAYS must be a whole number from 0 to 14.') from exc
    if retention_days < 0 or retention_days > 14:
        raise RuntimeError('RETENTION_DAYS must be a whole number from 0 to 14.')
    provider = os.getenv('EMAIL_PROVIDER', 'dry-run').strip().lower()
    if provider not in {'dry-run', 'smtp', 'api'}:
        raise RuntimeError('EMAIL_PROVIDER must be dry-run, smtp, or api.')
    try:
        max_upload_mb = int(os.getenv('HTS_MAX_UPLOAD_MB', '500'))
    except ValueError as exc:
        raise RuntimeError('HTS_MAX_UPLOAD_MB must be a positive whole number.') from exc
    if max_upload_mb < 1:
        raise RuntimeError('HTS_MAX_UPLOAD_MB must be a positive whole number.')
    try:
        smtp_port = int(os.getenv('SMTP_PORT', '587'))
    except ValueError as exc:
        raise RuntimeError('SMTP_PORT must be a whole number.') from exc
    data_dir = Path(os.getenv('HTS_DATA_DIR', 'data')).expanduser().resolve()
    return Settings(
        admin_username=username,
        admin_password=password,
        secret_key=secret,
        data_dir=data_dir,
        retention_days=retention_days,
        email_provider=provider,
        smtp_host=os.getenv('SMTP_HOST', '').strip(),
        smtp_port=smtp_port,
        smtp_username=os.getenv('SMTP_USERNAME', ''),
        smtp_password=os.getenv('SMTP_PASSWORD', ''),
        smtp_from=os.getenv('SMTP_FROM', '').strip(),
        smtp_use_tls=_flag('SMTP_USE_TLS', 'true'),
        email_api_url=os.getenv('EMAIL_API_URL', '').strip(),
        email_api_key=os.getenv('EMAIL_API_KEY', ''),
        email_from=os.getenv('EMAIL_FROM', '').strip(),
        feedback_url=os.getenv('HTS_FEEDBACK_URL', '').strip(),
        max_upload_mb=max_upload_mb,
        https_only=_flag('HTS_HTTPS_ONLY', 'false'),
    )
