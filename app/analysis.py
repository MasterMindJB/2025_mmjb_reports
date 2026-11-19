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
from lib.story_generation import generate_report_stories

PROMPT_DIR = Path(__file__).resolve().parents[1] / 'prompt_templates'
LLM_COLUMNS = [
    'Questions Asked',
    'Speaking Time (%)',
    'Total Interactions',
    'Word Count',
    'Words Per Minute (WPM)',
]


def build_gemini_llm(settings=None):
    key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
    if not key:
        raise RuntimeError('Set GOOGLE_API_KEY before generating reports.')
    if settings is None:
        from app.config import parse_gemini_model, parse_gemini_timeout

        model = parse_gemini_model(os.getenv('GEMINI_MODEL'))
        timeout = parse_gemini_timeout(os.getenv('GEMINI_TIMEOUT_SECONDS'))
    else:
        model = settings.gemini_model
        timeout = settings.gemini_timeout_seconds
    # gemini-2.5-flash returns 404 for new API keys. gemini-3.1-flash-lite answered
    # on the free tier. A timeout stops a stalled call, and max_retries bounds the
    # client backoff (the library default is 6, which spent about a minute on errors
    # that could not succeed).
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=0,
        google_api_key=key,
        timeout=timeout,
        max_retries=2,
    )


def apply_speaker_roles(speaker_map: dict, roles_by_label: dict[str, str] | None, participant_labels: set[str]) -> dict:
    """The stock pipeline has to see facilitator and host before One Thing is chosen."""
    for label, info in speaker_map.items():
        role = (roles_by_label or {}).get(label)
        if not role and label not in participant_labels:
            role = 'facilitator'
        if role:
            info['role'] = role
    return speaker_map


class GeminiAnalysis:
    def __init__(self, llm=None, prompt_dir: Path = PROMPT_DIR, settings=None):
        self.llm = llm
        self.prompt_dir = prompt_dir
        self.settings = settings

    def analyze(
        self,
        *,
        transcript: dict,
        speaker_names: dict[str, str],
        participant_labels: set[str],
        work_dir: Path,
        on_progress=None,
        roles_by_label: dict[str, str] | None = None,
    ) -> AnalysisBundle:
        if work_dir.exists():
            shutil.rmtree(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        llm = self.llm if self.llm is not None else build_gemini_llm(self.settings)
        summarizer = ConversationSummarizer(
            work_dir, self.prompt_dir, llm=llm, on_progress=on_progress
        )
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
        apply_speaker_roles(speaker_map, roles_by_label, participant_labels)
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
            speaker_analysis = {}
        else:
            df_participants = tables.df_speaker_stats.loc[participant_names]
            nice = tables.df_nice_stats_data[LLM_COLUMNS]
            speaker_analysis = summarizer.speaker_analysis(
                df_speaker_stats=df_participants,
                df_nice_stats_data=nice,
                conversation_script=tables.conversation_script,
                speakers_expected=len(participant_names),
                force_regen=True,
            )
        stories = generate_report_stories(
            llm,
            transcript=transcript,
            speaker_map=speaker_map,
            participant_labels=participant_labels,
            participation=tables.gini_reverse_interventions,
            interaction=tables.gini_reverse_interactions,
            word_interaction=tables.gini_reverse_interactions_both_wrd_cnt,
            progress_dir=work_dir / 'story_progress',
        )
        return AnalysisBundle(speaker_map, summaries, speaker_analysis, tables, stories)
