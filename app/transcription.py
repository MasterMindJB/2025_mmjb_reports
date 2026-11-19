"""AssemblyAI adapter. The web app talks to TranscriptionProvider, not the HTTP client."""

from __future__ import annotations

import os
from pathlib import Path

from lib.assemblyai_helper_functions import delete_transcript, get_audio_transcript


class AssemblyAITranscription:
    def transcribe(self, audio_path: Path, dest_dir: Path, speakers_expected: int | None) -> dict:
        if not os.getenv('ASSEMBLYAI_API_KEY'):
            raise RuntimeError('Set ASSEMBLYAI_API_KEY before transcribing a recording.')
        dest_dir.mkdir(parents=True, exist_ok=True)
        return get_audio_transcript(
            path_file=audio_path,
            path_response_upload_json=dest_dir / 'response_upload.json',
            path_response_create_json=dest_dir / 'response_create.json',
            path_response_transcript_json=dest_dir / 'response_transcript.json',
            speakers_expected=speakers_expected,
        )

    def delete_transcript(self, transcript_id: str) -> None:
        delete_transcript(transcript_id)
