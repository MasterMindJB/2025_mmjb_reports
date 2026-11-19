"""Small SQLite store for sessions, speakers, reports, and the email log."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path


SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY,
    client TEXT NOT NULL,
    group_name TEXT NOT NULL,
    session_name TEXT NOT NULL,
    session_date TEXT NOT NULL,
    consent_confirmed INTEGER NOT NULL,
    speakers_expected INTEGER,
    status TEXT NOT NULL,
    error_message TEXT,
    recording_path TEXT,
    recording_filename TEXT,
    transcript_path TEXT,
    assemblyai_transcript_id TEXT,
    created_at TEXT NOT NULL,
    reports_generated_at TEXT,
    purge_due_at TEXT,
    recording_deleted_at TEXT,
    transcript_deleted_at TEXT,
    assemblyai_deleted_at TEXT,
    assemblyai_delete_error TEXT,
    sent_at TEXT,
    window_start_ms INTEGER,
    window_end_ms INTEGER
);

CREATE TABLE IF NOT EXISTS speakers (
    id INTEGER PRIMARY KEY,
    session_id INTEGER NOT NULL,
    label TEXT NOT NULL,
    name TEXT,
    email TEXT,
    role TEXT NOT NULL DEFAULT 'participant',
    samples_json TEXT NOT NULL,
    UNIQUE(session_id, label),
    FOREIGN KEY(session_id) REFERENCES sessions(id)
);

CREATE TABLE IF NOT EXISTS reports (
    id INTEGER PRIMARY KEY,
    session_id INTEGER NOT NULL,
    speaker_id INTEGER,
    kind TEXT NOT NULL,
    path TEXT NOT NULL,
    filename TEXT NOT NULL,
    FOREIGN KEY(session_id) REFERENCES sessions(id)
);

CREATE TABLE IF NOT EXISTS email_log (
    id INTEGER PRIMARY KEY,
    session_id INTEGER NOT NULL,
    recipient_name TEXT NOT NULL,
    recipient_email TEXT NOT NULL,
    attachment_names TEXT NOT NULL,
    status TEXT NOT NULL,
    detail TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(session_id) REFERENCES sessions(id)
);
"""

SESSION_FIELDS = {
    'status',
    'error_message',
    'recording_path',
    'recording_filename',
    'transcript_path',
    'assemblyai_transcript_id',
    'reports_generated_at',
    'purge_due_at',
    'recording_deleted_at',
    'transcript_deleted_at',
    'assemblyai_deleted_at',
    'assemblyai_delete_error',
    'sent_at',
    'speakers_expected',
    'progress',
    'window_start_ms',
    'window_end_ms',
}


def _row(row) -> dict | None:
    if row is None:
        return None
    return dict(row)


class Database:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.executescript(SCHEMA)
            columns = {row['name'] for row in conn.execute('PRAGMA table_info(sessions)')}
            if 'progress' not in columns:
                conn.execute('ALTER TABLE sessions ADD COLUMN progress TEXT')
            for column in ('window_start_ms', 'window_end_ms'):
                if column not in columns:
                    conn.execute(f'ALTER TABLE sessions ADD COLUMN {column} INTEGER')
            speaker_columns = {row['name'] for row in conn.execute('PRAGMA table_info(speakers)')}
            if 'name_status' not in speaker_columns:
                conn.execute(
                    "ALTER TABLE speakers ADD COLUMN name_status TEXT NOT NULL DEFAULT 'confirmed'"
                )
            conn.execute('PRAGMA journal_mode=WAL')

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute('PRAGMA foreign_keys = ON')
        return conn

    def create_session(
        self,
        *,
        client: str,
        group_name: str,
        session_name: str,
        session_date: str,
        speakers_expected: int | None,
        recording_path: str,
        recording_filename: str,
        created_at: str,
        window_start_ms: int | None = None,
        window_end_ms: int | None = None,
    ) -> int:
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO sessions (
                    client, group_name, session_name, session_date, consent_confirmed,
                    speakers_expected, status, recording_path, recording_filename, created_at,
                    window_start_ms, window_end_ms
                ) VALUES (?, ?, ?, ?, 1, ?, 'transcribing', ?, ?, ?, ?, ?)
                """,
                (
                    client,
                    group_name,
                    session_name,
                    session_date,
                    speakers_expected,
                    recording_path,
                    recording_filename,
                    created_at,
                    window_start_ms,
                    window_end_ms,
                ),
            )
            return int(cursor.lastrowid)

    def get_session(self, session_id: int) -> dict | None:
        with self._lock, self._connect() as conn:
            return _row(conn.execute('SELECT * FROM sessions WHERE id = ?', (session_id,)).fetchone())

    def list_sessions(self) -> list[dict]:
        with self._lock, self._connect() as conn:
            rows = conn.execute('SELECT * FROM sessions ORDER BY id DESC').fetchall()
            return [dict(row) for row in rows]

    def update_session(self, session_id: int, **fields) -> None:
        unknown = set(fields) - SESSION_FIELDS
        if unknown:
            raise ValueError(f'Unknown session fields: {sorted(unknown)}')
        if not fields:
            return
        assignments = ', '.join(f'{name} = ?' for name in fields)
        values = list(fields.values()) + [session_id]
        with self._lock, self._connect() as conn:
            conn.execute(f'UPDATE sessions SET {assignments} WHERE id = ?', values)

    def replace_speakers(self, session_id: int, speakers: list[dict]) -> None:
        with self._lock, self._connect() as conn:
            conn.execute('DELETE FROM speakers WHERE session_id = ?', (session_id,))
            conn.executemany(
                """
                INSERT INTO speakers (session_id, label, name, email, role, samples_json)
                VALUES (?, ?, NULL, NULL, 'participant', ?)
                """,
                [
                    (session_id, speaker['label'], json.dumps(speaker['samples']))
                    for speaker in speakers
                ],
            )

    def list_speakers(self, session_id: int) -> list[dict]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                'SELECT * FROM speakers WHERE session_id = ? ORDER BY label',
                (session_id,),
            ).fetchall()
            speakers = []
            for row in rows:
                item = dict(row)
                item['samples'] = json.loads(item['samples_json'])
                speakers.append(item)
            return speakers

    def update_speaker(
        self, speaker_id: int, *, name: str, email: str, role: str, name_status: str = 'confirmed'
    ) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                'UPDATE speakers SET name = ?, email = ?, role = ?, name_status = ? WHERE id = ?',
                (name, email, role, name_status, speaker_id),
            )

    def update_speaker_samples(self, speaker_id: int, samples: list) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                'UPDATE speakers SET samples_json = ? WHERE id = ?',
                (json.dumps(samples), speaker_id),
            )

    def delete_speaker(self, speaker_id: int) -> None:
        with self._lock, self._connect() as conn:
            conn.execute('DELETE FROM speakers WHERE id = ?', (speaker_id,))

    def clear_reports(self, session_id: int) -> None:
        with self._lock, self._connect() as conn:
            conn.execute('DELETE FROM reports WHERE session_id = ?', (session_id,))

    def add_report(
        self, *, session_id: int, speaker_id: int | None, kind: str, path: str, filename: str
    ) -> int:
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO reports (session_id, speaker_id, kind, path, filename)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, speaker_id, kind, path, filename),
            )
            return int(cursor.lastrowid)

    def list_reports(self, session_id: int) -> list[dict]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM reports
                WHERE session_id = ?
                ORDER BY CASE kind WHEN 'group' THEN 0 ELSE 1 END, filename
                """,
                (session_id,),
            ).fetchall()
            return [dict(row) for row in rows]

    def get_report(self, report_id: int) -> dict | None:
        with self._lock, self._connect() as conn:
            return _row(conn.execute('SELECT * FROM reports WHERE id = ?', (report_id,)).fetchone())

    def add_email(
        self,
        *,
        session_id: int,
        recipient_name: str,
        recipient_email: str,
        attachment_names: list[str],
        status: str,
        detail: str,
        created_at: str,
    ) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO email_log (
                    session_id, recipient_name, recipient_email, attachment_names,
                    status, detail, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    recipient_name,
                    recipient_email,
                    json.dumps(attachment_names),
                    status,
                    detail,
                    created_at,
                ),
            )

    def list_emails(self, session_id: int) -> list[dict]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                'SELECT * FROM email_log WHERE session_id = ? ORDER BY id',
                (session_id,),
            ).fetchall()
            emails = []
            for row in rows:
                item = dict(row)
                item['attachment_names'] = json.loads(item['attachment_names'])
                emails.append(item)
            return emails

    def sessions_due(self, now_iso: str) -> list[dict]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM sessions
                WHERE purge_due_at IS NOT NULL
                  AND purge_due_at <= ?
                  AND status IN ('review', 'sent')
                  AND (
                        recording_deleted_at IS NULL
                        OR (
                            assemblyai_transcript_id IS NOT NULL
                            AND assemblyai_deleted_at IS NULL
                        )
                  )
                ORDER BY id
                """,
                (now_iso,),
            ).fetchall()
            return [dict(row) for row in rows]
