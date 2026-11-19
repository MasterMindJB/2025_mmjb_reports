"""Session workflow: transcribe, name speakers, render, send, then delete source media."""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.config import Settings
from app.db import Database
from app.interfaces import OutgoingEmail

ROLES = {'participant', 'facilitator', 'actor', 'host', 'audience', 'noise'}
REPORT_ROLES = {'participant'}


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def iso(moment: datetime) -> str:
    return moment.isoformat()


def session_root(settings: Settings, session_id: int) -> Path:
    return settings.data_dir / 'sessions' / str(session_id)


def clip_text(text: str, limit: int = 200) -> str:
    flattened = ' '.join(str(text or '').split())
    if len(flattened) <= limit:
        return flattened
    return flattened[: limit - 1].rstrip() + '…'


def speaker_previews(transcript: dict, limit: int = 3) -> list[dict]:
    grouped: dict[str, list] = {}
    for utterance in transcript.get('utterances') or []:
        label = utterance.get('speaker')
        if not label or label == 'UNK':
            continue
        grouped.setdefault(label, [])
        if len(grouped[label]) < limit:
            grouped[label].append(
                {
                    'text': clip_text(utterance.get('text') or ''),
                    'start': utterance.get('start'),
                    'end': utterance.get('end'),
                }
            )
    return [{'label': label, 'samples': grouped[label]} for label in sorted(grouped)]


def short_error(exc: Exception) -> str:
    text = f'{type(exc).__name__}: {exc}'
    return text.replace('\n', ' ')[:300]


def transcribe_session(db: Database, provider, settings: Settings, session_id: int) -> None:
    session = db.get_session(session_id)
    if session is None:
        return
    try:
        recording = Path(session['recording_path'])
        dest = session_root(settings, session_id) / 'transcript'
        payload = provider.transcribe(recording, dest, session['speakers_expected'])
        transcript_path = dest / 'response_transcript.json'
        transcript_path.write_text(json.dumps(payload), encoding='utf-8')
        previews = speaker_previews(payload)
        if not previews:
            raise RuntimeError('No speakers were detected in the recording.')
        db.replace_speakers(session_id, previews)
        db.update_session(
            session_id,
            status='naming',
            error_message=None,
            transcript_path=str(transcript_path),
            assemblyai_transcript_id=payload.get('id'),
        )
    except Exception as exc:
        logging.exception('Transcription failed for session %s', session_id)
        db.update_session(session_id, status='error', error_message=short_error(exc), progress=None)


def _load_transcript(session: dict) -> dict:
    path = session.get('transcript_path')
    if not path or not Path(path).is_file():
        raise RuntimeError('The transcript is no longer on disk.')
    return json.loads(Path(path).read_text(encoding='utf-8'))


def merge_speakers(db: Database, session: dict, source: str, target: str) -> None:
    if session['status'] != 'naming':
        raise RuntimeError('Speakers can be merged only before reports are generated.')
    if source == target:
        raise RuntimeError('Choose two different speakers to merge.')
    speakers = {item['label']: item for item in db.list_speakers(session['id'])}
    if source not in speakers or target not in speakers:
        raise RuntimeError('Those speakers are not part of this session.')
    transcript = _load_transcript(session)
    for key in ('utterances', 'sentiment_analysis_results'):
        for item in transcript.get(key) or []:
            if item.get('speaker') == source:
                item['speaker'] = target
    Path(session['transcript_path']).write_text(json.dumps(transcript), encoding='utf-8')
    combined = (speakers[target]['samples'] + speakers[source]['samples'])[:4]
    db.update_speaker_samples(speakers[target]['id'], combined)
    db.delete_speaker(speakers[source]['id'])


def generate_session(
    db: Database,
    analysis,
    renderer,
    transcription,
    settings: Settings,
    session_id: int,
) -> None:
    session = db.get_session(session_id)
    if session is None:
        return
    try:
        speakers = db.list_speakers(session_id)
        from lib.naming import assign_display_names

        assigned = assign_display_names(speakers)
        included = [speaker for speaker in assigned if speaker['include']]
        if not included:
            raise RuntimeError('Mark at least one speaker to include. Noise labels are left out.')
        speaker_names = {speaker['label']: speaker['display_name'] for speaker in included}
        recipients = [speaker for speaker in included if speaker['receives_report']]
        participant_labels = {speaker['label']: speaker for speaker in recipients}
        participant_labels = set(participant_labels)
        transcript = _load_transcript(session)
        noise = {speaker['label'] for speaker in assigned if not speaker['include']}
        if noise:
            for key in ('utterances', 'sentiment_analysis_results'):
                transcript[key] = [
                    item for item in (transcript.get(key) or []) if item.get('speaker') not in noise
                ]
        work_dir = session_root(settings, session_id) / 'work'

        def note(message: str) -> None:
            logging.info('Session %s: %s', session_id, message)
            db.update_session(session_id, progress=message[:240])

        note('Starting report generation')
        bundle = analysis.analyze(
            transcript=transcript,
            speaker_names=speaker_names,
            participant_labels=participant_labels,
            work_dir=work_dir,
            on_progress=note,
        )
        note('Rendering the PDFs')
        staging = session_root(settings, session_id) / 'reports_staging'
        reports_dir = session_root(settings, session_id) / 'reports'
        if staging.exists():
            shutil.rmtree(staging)
        rendered = renderer.render(
            transcript=transcript,
            bundle=bundle,
            reports_dir=staging,
            participant_labels=participant_labels,
            feedback_url=settings.feedback_url,
            roles_by_label={speaker['label']: speaker['role'] for speaker in speakers},
        )
        if reports_dir.exists():
            shutil.rmtree(reports_dir)
        staging.rename(reports_dir)
        db.clear_reports(session_id)
        by_label = {speaker['label']: speaker for speaker in speakers}
        for report in rendered:
            final_path = reports_dir / report.filename
            speaker_id = by_label[report.speaker_label]['id'] if report.kind == 'individual' else None
            db.add_report(
                session_id=session_id,
                speaker_id=speaker_id,
                kind=report.kind,
                path=str(final_path),
                filename=report.filename,
            )
        generated = utcnow()
        due = generated if settings.retention_days == 0 else generated + timedelta(days=settings.retention_days)
        db.update_session(
            session_id,
            status='review',
            error_message=None,
            reports_generated_at=iso(generated),
            purge_due_at=iso(due),
            progress=None,
        )
        if settings.retention_days == 0:
            purge_session(db, transcription, settings, db.get_session(session_id))
    except Exception as exc:
        logging.exception('Report generation failed for session %s', session_id)
        db.update_session(session_id, status='error', error_message=short_error(exc))


def _under(root: Path, path: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def purge_session(db: Database, transcription, settings: Settings, session: dict, now: datetime | None = None) -> None:
    if session is None:
        return
    moment = now or utcnow()
    root = session_root(settings, session['id'])
    if not session.get('recording_deleted_at'):
        recording = session.get('recording_path')
        if recording:
            path = Path(recording)
            if path.exists() and _under(root, path):
                path.unlink()
        for folder_name in ('transcript', 'work', 'reports_staging'):
            folder = root / folder_name
            if folder.exists() and _under(root, folder):
                shutil.rmtree(folder)
        db.update_session(
            session['id'],
            recording_path=None,
            transcript_path=None,
            recording_deleted_at=iso(moment),
            transcript_deleted_at=iso(moment),
        )
        session = db.get_session(session['id'])
    transcript_id = session.get('assemblyai_transcript_id')
    if transcript_id and not session.get('assemblyai_deleted_at'):
        try:
            transcription.delete_transcript(transcript_id)
        except Exception as exc:
            logging.error('AssemblyAI delete failed for session %s (%s)', session['id'], type(exc).__name__)
            db.update_session(session['id'], assemblyai_delete_error=short_error(exc))
            return
        db.update_session(
            session['id'],
            assemblyai_deleted_at=iso(moment),
            assemblyai_delete_error=None,
        )


def purge_due(db: Database, transcription, settings: Settings, now: datetime | None = None) -> None:
    moment = now or utcnow()
    for session in db.sessions_due(iso(moment)):
        purge_session(db, transcription, settings, session, now=moment)


def attachments_for(participant: dict, reports: list[dict]) -> list[Path]:
    own = [
        report
        for report in reports
        if report['kind'] == 'individual' and report['speaker_id'] == participant['id']
    ]
    group = [report for report in reports if report['kind'] == 'group']
    if len(own) != 1 or len(group) != 1:
        raise RuntimeError(f'Report set is incomplete for {participant["name"]}.')
    paths = [Path(own[0]['path']), Path(group[0]['path'])]
    for path in paths:
        if not path.is_file():
            raise RuntimeError(f'Missing report file {path.name}.')
    return paths


def send_session(db: Database, emailer, session: dict) -> str:
    if session['status'] != 'review':
        raise RuntimeError('Reports can be sent only from the review screen.')
    speakers = db.list_speakers(session['id'])
    reports = db.list_reports(session['id'])
    participants = [speaker for speaker in speakers if speaker['role'] in REPORT_ROLES]
    if not participants:
        raise RuntimeError('There are no participants to email.')
    already = {
        item['recipient_email']
        for item in db.list_emails(session['id'])
        if item['status'] in {'sent', 'dry-run'}
    }
    for participant in participants:
        if participant['email'] in already:
            continue
        paths = attachments_for(participant, reports)
        message = OutgoingEmail(
            to_email=participant['email'],
            to_name=participant['name'],
            subject=f"Your Harkness report — {session['session_name']}",
            body=(
                f"Hello {participant['name']},\n\n"
                "Attached are your individual report and the group report for "
                f"{session['client']} / {session['group_name']} / {session['session_name']} "
                f"({session['session_date']}).\n\n"
                "Your individual report is only for you. The group report does not include "
                "anyone else's individual report.\n\n"
                "Harkness Training Solutions\n"
            ),
            attachments=paths,
        )
        detail = emailer.send(message)
        status = 'dry-run' if emailer.name == 'dry-run' else 'sent'
        db.add_email(
            session_id=session['id'],
            recipient_name=participant['name'],
            recipient_email=participant['email'],
            attachment_names=[path.name for path in paths],
            status=status,
            detail=detail,
            created_at=iso(utcnow()),
        )
    db.update_session(session['id'], status='sent', sent_at=iso(utcnow()), error_message=None)
    if emailer.name == 'dry-run':
        return 'Emails were saved locally. Nothing was sent.'
    return 'Reports were emailed. Each participant received only their own report and the group report.'
