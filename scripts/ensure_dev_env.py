"""Fill a local .env so `make dev` can sign in. Does not print API keys."""

from __future__ import annotations

import secrets
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / '.env'
EXAMPLE = ROOT / '.env.example'


def _lines(text: str) -> list[str]:
    return text.splitlines()


def _get(lines: list[str], key: str) -> str:
    prefix = f'{key}='
    for line in lines:
        if line.startswith(prefix):
            return line[len(prefix):]
    return ''


def _set(lines: list[str], key: str, value: str) -> None:
    prefix = f'{key}='
    for index, line in enumerate(lines):
        if line.startswith(prefix):
            lines[index] = f'{key}={value}'
            return
    lines.append(f'{key}={value}')


def main() -> None:
    if ENV_PATH.exists():
        lines = _lines(ENV_PATH.read_text(encoding='utf-8'))
    elif EXAMPLE.exists():
        lines = _lines(EXAMPLE.read_text(encoding='utf-8'))
    else:
        lines = []
    created = []
    if not _get(lines, 'HTS_ADMIN_USERNAME').strip():
        _set(lines, 'HTS_ADMIN_USERNAME', 'jason')
        created.append('HTS_ADMIN_USERNAME')
    if len(_get(lines, 'HTS_ADMIN_PASSWORD')) < 8:
        password = secrets.token_urlsafe(18)
        _set(lines, 'HTS_ADMIN_PASSWORD', password)
        created.append('HTS_ADMIN_PASSWORD')
        print(f'Local admin password: {password}')
    if len(_get(lines, 'HTS_SECRET_KEY')) < 16:
        _set(lines, 'HTS_SECRET_KEY', secrets.token_urlsafe(32))
        created.append('HTS_SECRET_KEY')
    if not _get(lines, 'EMAIL_PROVIDER').strip():
        _set(lines, 'EMAIL_PROVIDER', 'dry-run')
    ENV_PATH.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    username = _get(lines, 'HTS_ADMIN_USERNAME').strip()
    print(f'Local admin username: {username}')
    if created:
        print(f'Wrote {", ".join(created)} to .env')
    else:
        print('.env already has the local sign-in values')
    print('make dev forces EMAIL_PROVIDER=dry-run. It does not send mail.')
    if not _get(lines, 'ASSEMBLYAI_API_KEY').strip():
        print('Transcription still needs ASSEMBLYAI_API_KEY in .env.')
    if not (_get(lines, 'GOOGLE_API_KEY').strip() or _get(lines, 'GEMINI_API_KEY').strip()):
        print('Report generation still needs GOOGLE_API_KEY in .env.')


if __name__ == '__main__':
    main()
