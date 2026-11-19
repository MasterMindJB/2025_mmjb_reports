"""Harkness Training Solutions: upload a recording, name the speakers, review, send."""

from __future__ import annotations

import logging
import os
import re
import shutil
import threading
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from app.analysis import GeminiAnalysis
from app.config import Settings, load_settings
from app.db import Database
from app.emailer import build_emailer
from app.rendering import PdfReportRenderer
from app.services import (
    ROLES,
    generate_session,
    merge_speakers,
    purge_due,
    send_session,
    transcribe_session,
    utcnow,
    iso,
    session_root,
)
from app.transcription import AssemblyAITranscription

ALLOWED_SUFFIXES = {'.mp3', '.wav', '.m4a', '.mp4', '.webm', '.aac', '.ogg', '.flac', '.mov', '.mpeg', '.mpga'}
EMAIL_RE = re.compile(r'^[^@\s]+@[^@\s]+\.[^@\s]+$')
NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9 .'\-]{0,80}\Z")
APP_DIR = Path(__file__).resolve().parent


class NotAuthenticated(Exception):
    pass


def format_ms(value) -> str:
    if value is None or value == '':
        return ''
    seconds = int(value) // 1000
    return f'{seconds // 60}:{seconds % 60:02d}'


def require_admin(request: Request) -> None:
    if not request.session.get('admin'):
        raise NotAuthenticated()


def render(request: Request, name: str, status_code: int = 200, **extra):
    context = {
        'flash': request.session.pop('flash', None),
        'email_mode': request.app.state.settings.email_provider,
        'retention_days': request.app.state.settings.retention_days,
        **extra,
    }
    return request.app.state.templates.TemplateResponse(
        request, name, context, status_code=status_code
    )


def flash(request: Request, message: str) -> None:
    request.session['flash'] = message


def clean_text(value: str, field: str, limit: int = 200) -> str:
    text = ' '.join((value or '').split())
    if not text:
        raise ValueError(f'{field} is required.')
    if len(text) > limit:
        raise ValueError(f'{field} must be {limit} characters or fewer.')
    return text


def clean_person_name(value: str) -> str:
    text = ' '.join((value or '').split())
    if text.casefold() == 'average':
        raise ValueError('Average is reserved. Choose another name.')
    if not NAME_RE.fullmatch(text):
        raise ValueError('Names can use letters, numbers, spaces, apostrophes, and hyphens.')
    return text


def validate_assignments(names: list[str], emails: list[str], roles: list[str]) -> list[dict]:
    if not (len(names) == len(emails) == len(roles)):
        raise ValueError('The speaker form was incomplete. Save it again.')
    cleaned = []
    seen_names = set()
    seen_emails = set()
    participants = 0
    for raw_name, raw_email, role in zip(names, emails, roles):
        name = clean_person_name(raw_name)
        if name.casefold() in seen_names:
            raise ValueError(f'{name} is used more than once.')
        seen_names.add(name.casefold())
        if role not in ROLES:
            raise ValueError('Each speaker is a participant, facilitator, or actor.')
        email = (raw_email or '').strip()
        if role == 'participant':
            if not EMAIL_RE.fullmatch(email):
                raise ValueError(f'Enter an email address for {name}.')
            if email.casefold() in seen_emails:
                raise ValueError(f'{email} is used more than once. Each participant needs their own inbox.')
            seen_emails.add(email.casefold())
            participants += 1
        elif email and not EMAIL_RE.fullmatch(email):
            raise ValueError(f'The email for {name} is not valid.')
        cleaned.append({'name': name, 'email': email, 'role': role})
    if participants < 1:
        raise ValueError('At least one speaker must be a participant who receives a report.')
    return cleaned


def _session_or_404(request: Request, session_id: int) -> dict:
    session = request.app.state.db.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail='Session not found')
    return session


def create_app(
    settings: Settings | None = None,
    transcription=None,
    analysis=None,
    emailer=None,
    renderer=None,
    start_retention_thread: bool = True,
) -> FastAPI:
    os.environ.setdefault('MPLBACKEND', 'Agg')
    settings = settings or load_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    (settings.data_dir / 'outbox').mkdir(parents=True, exist_ok=True)
    (settings.data_dir / 'sessions').mkdir(parents=True, exist_ok=True)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        stop = threading.Event()

        def loop() -> None:
            while not stop.wait(3600):
                try:
                    purge_due(app.state.db, app.state.transcription, app.state.settings)
                except Exception:
                    logging.error('Retention check failed')

        thread = None
        if app.state.start_retention_thread:
            thread = threading.Thread(target=loop, name='hts-retention', daemon=True)
            thread.start()
        yield
        stop.set()
        if thread is not None:
            thread.join(timeout=1)

    app = FastAPI(title='Harkness Training Solutions', lifespan=lifespan)
    app.state.settings = settings
    app.state.db = Database(settings.data_dir / 'hts.sqlite3')
    app.state.transcription = transcription or AssemblyAITranscription()
    app.state.analysis = analysis or GeminiAnalysis()
    app.state.emailer = emailer or build_emailer(settings)
    app.state.renderer = renderer or PdfReportRenderer()
    app.state.start_retention_thread = start_retention_thread
    templates = Jinja2Templates(directory=str(APP_DIR / 'templates'))
    templates.env.filters['mmss'] = format_ms
    app.state.templates = templates

    @app.middleware('http')
    async def housekeeping(request: Request, call_next):
        try:
            purge_due(request.app.state.db, request.app.state.transcription, request.app.state.settings)
        except Exception:
            logging.error('Retention check failed')
        response = await call_next(request)
        response.headers['Cache-Control'] = 'no-store'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Referrer-Policy'] = 'same-origin'
        response.headers.setdefault('X-Frame-Options', 'SAMEORIGIN')
        return response

    app.add_middleware(
        SessionMiddleware,
        secret_key=settings.secret_key,
        session_cookie='hts_session',
        max_age=60 * 60 * 12,
        same_site='lax',
        https_only=settings.https_only,
    )
    app.mount('/static', StaticFiles(directory=str(APP_DIR / 'static')), name='static')

    @app.exception_handler(NotAuthenticated)
    async def _unauthenticated(request: Request, _exc: NotAuthenticated):
        return RedirectResponse('/login', status_code=303)

    public = APIRouter()
    private = APIRouter(dependencies=[Depends(require_admin)])

    @public.get('/health')
    def health():
        return {'status': 'ok'}

    @public.get('/login')
    def login_form(request: Request):
        if request.session.get('admin'):
            return RedirectResponse('/', status_code=303)
        return render(request, 'login.html')

    @public.post('/login')
    def login(request: Request, username: str = Form(''), password: str = Form('')):
        user_ok = secrets_compare(username, settings.admin_username)
        pass_ok = secrets_compare(password, settings.admin_password)
        if not (user_ok and pass_ok):
            return render(request, 'login.html', status_code=401, error='Those sign-in details were not recognized.')
        request.session['admin'] = True
        return RedirectResponse('/', status_code=303)

    @private.post('/logout')
    def logout(request: Request):
        request.session.clear()
        return RedirectResponse('/login', status_code=303)

    @private.get('/')
    def home(request: Request):
        return render(request, 'sessions.html', sessions=request.app.state.db.list_sessions())

    @private.get('/upload')
    def upload_form(request: Request):
        return render(request, 'upload.html')

    @private.post('/sessions')
    async def create_session(
        request: Request,
        background: BackgroundTasks,
        client: str = Form(''),
        group_name: str = Form(''),
        session_name: str = Form(''),
        session_date: str = Form(''),
        speakers_expected: str = Form(''),
        consent: str = Form(''),
        recording: UploadFile | None = None,
    ):
        try:
            if consent != 'yes':
                raise ValueError('Confirm that everyone in the recording has consented.')
            client_name = clean_text(client, 'Client')
            group = clean_text(group_name, 'Group')
            title = clean_text(session_name, 'Session')
            try:
                datetime.strptime(session_date, '%Y-%m-%d')
            except ValueError as exc:
                raise ValueError('Enter the session date.') from exc
            expected = None
            if speakers_expected.strip():
                if not speakers_expected.strip().isdigit():
                    raise ValueError('Expected speakers must be a whole number.')
                expected = int(speakers_expected.strip())
                if expected < 1 or expected > 25:
                    raise ValueError('Expected speakers must be between 1 and 25, or left blank.')
            if recording is None or not recording.filename:
                raise ValueError('Choose a recording to upload.')
            suffix = Path(recording.filename).suffix.lower()
            if suffix not in ALLOWED_SUFFIXES:
                raise ValueError('Upload an audio or video file (mp3, wav, m4a, mp4, webm, and similar).')
        except ValueError as exc:
            return render(request, 'upload.html', status_code=400, error=str(exc))

        db = request.app.state.db
        session_id = db.create_session(
            client=client_name,
            group_name=group,
            session_name=title,
            session_date=session_date,
            speakers_expected=expected,
            recording_path='',
            recording_filename=Path(recording.filename).name,
            created_at=iso(utcnow()),
        )
        root = session_root(settings, session_id)
        root.mkdir(parents=True, exist_ok=True)
        destination = root / f'source{suffix}'
        limit = settings.max_upload_mb * 1024 * 1024
        size = 0
        try:
            with destination.open('wb') as handle:
                while True:
                    chunk = await recording.read(1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > limit:
                        raise ValueError(f'Recordings must be {settings.max_upload_mb} MB or smaller.')
                    handle.write(chunk)
            if size == 0:
                raise ValueError('The recording file is empty.')
        except ValueError as exc:
            destination.unlink(missing_ok=True)
            db.update_session(session_id, status='error', error_message=str(exc), recording_path=None)
            flash(request, str(exc))
            return RedirectResponse(f'/sessions/{session_id}', status_code=303)
        db.update_session(session_id, recording_path=str(destination))
        background.add_task(
            transcribe_session,
            db,
            request.app.state.transcription,
            settings,
            session_id,
        )
        return RedirectResponse(f'/sessions/{session_id}', status_code=303)

    @private.get('/sessions/{session_id}')
    def session_page(request: Request, session_id: int):
        session = _session_or_404(request, session_id)
        speakers = request.app.state.db.list_speakers(session_id)
        reports = request.app.state.db.list_reports(session_id)
        emails = request.app.state.db.list_emails(session_id)
        return render(
            request,
            'session.html',
            session=session,
            speakers=speakers,
            reports=reports,
            emails=emails,
        )

    @private.post('/sessions/{session_id}/speakers')
    def save_speakers(
        request: Request,
        background: BackgroundTasks,
        session_id: int,
        action: str = Form('save'),
        label: list[str] = Form(default=[]),
        name: list[str] = Form(default=[]),
        email: list[str] = Form(default=[]),
        role: list[str] = Form(default=[]),
    ):
        session = _session_or_404(request, session_id)
        if session['status'] not in {'naming', 'error'}:
            flash(request, 'Speakers can be edited before reports are generated.')
            return RedirectResponse(f'/sessions/{session_id}', status_code=303)
        db = request.app.state.db
        current = db.list_speakers(session_id)
        if [item['label'] for item in current] != label:
            flash(request, 'The speaker list changed. Review it and save again.')
            return RedirectResponse(f'/sessions/{session_id}', status_code=303)
        try:
            cleaned = validate_assignments(name, email, role)
        except ValueError as exc:
            flash(request, str(exc))
            return RedirectResponse(f'/sessions/{session_id}', status_code=303)
        for speaker, assignment in zip(current, cleaned):
            db.update_speaker(
                speaker['id'],
                name=assignment['name'],
                email=assignment['email'],
                role=assignment['role'],
            )
        if action == 'generate':
            db.update_session(session_id, status='generating', error_message=None)
            background.add_task(
                generate_session,
                db,
                request.app.state.analysis,
                request.app.state.renderer,
                request.app.state.transcription,
                settings,
                session_id,
            )
            return RedirectResponse(f'/sessions/{session_id}', status_code=303)
        db.update_session(session_id, status='naming', error_message=None)
        flash(request, 'Speakers saved.')
        return RedirectResponse(f'/sessions/{session_id}', status_code=303)

    @private.post('/sessions/{session_id}/merge')
    def merge(request: Request, session_id: int, source: str = Form(''), target: str = Form('')):
        session = _session_or_404(request, session_id)
        try:
            merge_speakers(request.app.state.db, session, source, target)
        except RuntimeError as exc:
            flash(request, str(exc))
        else:
            flash(request, f'Speaker {source} was merged into {target}.')
        return RedirectResponse(f'/sessions/{session_id}', status_code=303)

    @private.post('/sessions/{session_id}/reopen')
    def reopen(request: Request, session_id: int):
        session = _session_or_404(request, session_id)
        if session['status'] != 'review':
            flash(request, 'Names can be changed again only before the reports are sent.')
            return RedirectResponse(f'/sessions/{session_id}', status_code=303)
        reports_dir = session_root(settings, session_id) / 'reports'
        if reports_dir.exists():
            shutil.rmtree(reports_dir)
        request.app.state.db.clear_reports(session_id)
        request.app.state.db.update_session(
            session_id,
            status='naming',
            error_message=None,
            reports_generated_at=None,
            purge_due_at=None,
        )
        flash(request, 'Report files were removed. Update the names, then generate again.')
        return RedirectResponse(f'/sessions/{session_id}', status_code=303)

    @private.post('/sessions/{session_id}/retry')
    def retry(request: Request, background: BackgroundTasks, session_id: int):
        session = _session_or_404(request, session_id)
        if session['status'] not in {'error', 'transcribing', 'generating'}:
            return RedirectResponse(f'/sessions/{session_id}', status_code=303)
        db = request.app.state.db
        speakers = db.list_speakers(session_id)
        transcript_ready = bool(session.get('transcript_path') and Path(session['transcript_path']).is_file())
        if session['status'] == 'transcribing':
            db.update_session(session_id, status='transcribing', error_message=None)
            background.add_task(
                transcribe_session, db, request.app.state.transcription, settings, session_id
            )
            return RedirectResponse(f'/sessions/{session_id}', status_code=303)
        if session['status'] == 'generating':
            db.update_session(session_id, status='generating', error_message=None)
            background.add_task(
                generate_session,
                db,
                request.app.state.analysis,
                request.app.state.renderer,
                request.app.state.transcription,
                settings,
                session_id,
            )
            return RedirectResponse(f'/sessions/{session_id}', status_code=303)
        if transcript_ready and speakers and all(speaker.get('name') for speaker in speakers):
            try:
                validate_assignments(
                    [speaker['name'] or '' for speaker in speakers],
                    [speaker['email'] or '' for speaker in speakers],
                    [speaker['role'] for speaker in speakers],
                )
            except ValueError as exc:
                db.update_session(session_id, status='naming', error_message=str(exc))
                return RedirectResponse(f'/sessions/{session_id}', status_code=303)
            db.update_session(session_id, status='generating', error_message=None)
            background.add_task(
                generate_session,
                db,
                request.app.state.analysis,
                request.app.state.renderer,
                request.app.state.transcription,
                settings,
                session_id,
            )
        elif transcript_ready and speakers:
            db.update_session(session_id, status='naming', error_message=None)
        else:
            db.update_session(session_id, status='transcribing', error_message=None)
            background.add_task(
                transcribe_session, db, request.app.state.transcription, settings, session_id
            )
        return RedirectResponse(f'/sessions/{session_id}', status_code=303)

    @private.get('/sessions/{session_id}/reports/{report_id}')
    def download_report(request: Request, session_id: int, report_id: int):
        _session_or_404(request, session_id)
        report = request.app.state.db.get_report(report_id)
        if report is None or report['session_id'] != session_id:
            raise HTTPException(status_code=404, detail='Report not found')
        path = Path(report['path']).resolve()
        root = (session_root(settings, session_id) / 'reports').resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail='Report not found') from exc
        if not path.is_file():
            raise HTTPException(status_code=404, detail='Report not found')
        return FileResponse(
            path,
            media_type='application/pdf',
            filename=report['filename'],
            content_disposition_type='inline',
        )

    @private.post('/sessions/{session_id}/send')
    def send(request: Request, session_id: int):
        session = _session_or_404(request, session_id)
        try:
            message = send_session(request.app.state.db, request.app.state.emailer, session)
        except RuntimeError as exc:
            flash(request, str(exc))
        else:
            flash(request, message)
        return RedirectResponse(f'/sessions/{session_id}', status_code=303)

    app.include_router(public)
    app.include_router(private)
    return app


def secrets_compare(left: str, right: str) -> bool:
    import secrets

    return secrets.compare_digest(left, right)
