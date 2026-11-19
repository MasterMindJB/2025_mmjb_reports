"""Provider boundaries. The web app depends on these, not on a specific vendor."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from lib.report_stats import SessionTables


class TranscriptionProvider(Protocol):
    def transcribe(
        self, audio_path: Path, dest_dir: Path, speakers_expected: int | None
    ) -> dict: ...

    def delete_transcript(self, transcript_id: str) -> None: ...


class AnalysisBundle:
    def __init__(
        self,
        speaker_map: dict,
        summaries: dict,
        speaker_analysis: dict,
        tables: SessionTables,
        stories: dict | None = None,
    ):
        self.speaker_map = speaker_map
        self.summaries = summaries
        self.speaker_analysis = speaker_analysis
        self.tables = tables
        self.stories = stories or {}


class AnalysisProvider(Protocol):
    def analyze(
        self,
        *,
        transcript: dict,
        speaker_names: dict[str, str],
        participant_labels: set[str],
        work_dir: Path,
    ) -> AnalysisBundle: ...


class RenderedReport:
    def __init__(self, kind: str, path: Path, filename: str, speaker_label: str | None):
        self.kind = kind
        self.path = path
        self.filename = filename
        self.speaker_label = speaker_label


class ReportRenderer(Protocol):
    def render(
        self,
        *,
        transcript: dict,
        bundle: AnalysisBundle,
        reports_dir: Path,
        participant_labels: set[str],
        feedback_url: str,
        roles_by_label: dict[str, str] | None = None,
    ) -> list[RenderedReport]: ...


class OutgoingEmail:
    def __init__(self, to_email: str, to_name: str, subject: str, body: str, attachments: list[Path]):
        self.to_email = to_email
        self.to_name = to_name
        self.subject = subject
        self.body = body
        self.attachments = attachments


class EmailProvider(Protocol):
    name: str

    def send(self, message: OutgoingEmail) -> str: ...
