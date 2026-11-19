"""Gemini analysis adapter around the existing prompt chains."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from langchain_google_genai import ChatGoogleGenerativeAI

from app.interfaces import AnalysisBundle
from lib.llm_functions import ConversationSummarizer
from lib.report_stats import compute_session_tables

PROMPT_DIR = Path(__file__).resolve().parents[1] / 'prompt_templates'
LLM_COLUMNS = [
    'Questions Asked',
    'Speaking Time (%)',
    'Total Interactions',
    'Word Count',
    'Words Per Minute (WPM)',
]


def build_gemini_llm():
    key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
    if not key:
        raise RuntimeError('Set GOOGLE_API_KEY before generating reports.')
    model = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')
    return ChatGoogleGenerativeAI(model=model, temperature=0, google_api_key=key)


class GeminiAnalysis:
    def __init__(self, llm=None, prompt_dir: Path = PROMPT_DIR):
        self.llm = llm
        self.prompt_dir = prompt_dir

    def analyze(
        self,
        *,
        transcript: dict,
        speaker_names: dict[str, str],
        participant_labels: set[str],
        work_dir: Path,
    ) -> AnalysisBundle:
        if work_dir.exists():
            shutil.rmtree(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        llm = self.llm if self.llm is not None else build_gemini_llm()
        summarizer = ConversationSummarizer(work_dir, self.prompt_dir, llm=llm)
        labels = sorted(
            {
                utterance.get('speaker')
                for utterance in transcript.get('utterances') or []
                if utterance.get('speaker') not in (None, 'UNK')
            }
        )
        missing = [label for label in labels if label not in speaker_names]
        if missing:
            raise RuntimeError(f'Speakers still need names: {", ".join(missing)}')
        speaker_map = summarizer.speaker_map(
            utterances=transcript.get('utterances') or [],
            speakers_expected=len(labels),
            force_regen=True,
        )
        for label in labels:
            if label not in speaker_map:
                raise RuntimeError(f'The analysis did not return speaker {label}.')
            speaker_map[label]['name'] = speaker_names[label]
        (work_dir / 'speaker_map.json').write_text(json.dumps(speaker_map, indent=4), encoding='utf-8')

        tables = compute_session_tables(transcript, speaker_map)
        summaries = summarizer.conversation_summaries(
            conversation_script=tables.conversation_script,
            speakers_expected=len(labels),
            force_regen=True,
        )
        participant_names = [
            speaker_map[label]['name'] for label in labels if label in participant_labels
        ]
        if not participant_names:
            raise RuntimeError('At least one participant is required before reports can be generated.')
        df_participants = tables.df_speaker_stats.loc[participant_names]
        nice = tables.df_nice_stats_data[LLM_COLUMNS]
        speaker_analysis = summarizer.speaker_analysis(
            df_speaker_stats=df_participants,
            df_nice_stats_data=nice,
            conversation_script=tables.conversation_script,
            speakers_expected=len(participant_names),
            force_regen=True,
        )
        return AnalysisBundle(speaker_map, summaries, speaker_analysis, tables)
