"""Runtime configuration. Secrets come from the environment, never from the repo."""

from __future__ import annotations

import os
import re
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
    gemini_model: str = 'gemini-3.1-flash-lite'
    gemini_timeout_seconds: float = 120.0
    include_personality: bool = False

    @property
    def sends_email(self) -> bool:
        return self.email_provider in {'smtp', 'api'}


def _flag(name: str, default: str = 'false') -> bool:
    return os.getenv(name, default).strip().lower() in {'1', 'true', 'yes', 'on'}


_MODEL_RE = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]{0,80}\Z')


def parse_gemini_model(raw: str | None) -> str:
    model = (raw if raw is not None else 'gemini-3.1-flash-lite').strip()
    if not _MODEL_RE.fullmatch(model):
        raise RuntimeError(
            'GEMINI_MODEL must be a model id such as gemini-3.1-flash-lite, '
            'with no spaces or slashes.'
        )
    return model


def parse_gemini_timeout(raw: str | None) -> float:
    try:
        timeout = float(raw if raw not in (None, '') else '120')
    except ValueError as exc:
        raise RuntimeError('GEMINI_TIMEOUT_SECONDS must be a number of seconds.') from exc
    if timeout < 1 or timeout > 600:
        raise RuntimeError('GEMINI_TIMEOUT_SECONDS must be between 1 and 600.')
    return timeout


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
    gemini_model = parse_gemini_model(os.getenv('GEMINI_MODEL'))
    gemini_timeout_seconds = parse_gemini_timeout(os.getenv('GEMINI_TIMEOUT_SECONDS'))
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
        gemini_model=gemini_model,
        gemini_timeout_seconds=gemini_timeout_seconds,
        include_personality=_flag('HTS_PERSONALITY', 'false'),
    )
