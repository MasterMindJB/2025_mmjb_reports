"""Ask the model for structured story fields, then render the sentences ourselves."""

from __future__ import annotations

import json
import logging
import re

from langchain_core.exceptions import OutputParserException

from lib.chord_diagram import directed_follows
from lib.story_craft import render_section_prompt
from lib.story_evidence import cast_line, quote_events
from lib.story_render import (
    SCHEMAS,
    facts_prompt,
    group_facts,
    personal_facts,
    pick_transitions,
    render_group_section,
    render_personal_section,
    transcript_excerpt,
)

ATTEMPTS = 2
PERSONAL_SECTIONS = ('character', 'goal', 'challenge', 'drama', 'resolution', 'one_thing')
GROUP_SECTIONS = PERSONAL_SECTIONS

# Before this batching, each section was its own call and each call retried 3 times.
PREVIOUS_ATTEMPTS = 3
PREVIOUS_SECTION_CALLS = 6


def _content(raw) -> str:
    content = raw.content if hasattr(raw, 'content') else raw
    if isinstance(content, list):
        return ' '.join(str(part) for part in content)
    return str(content)


def _parse_json(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith('```'):
        cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
        cleaned = re.sub(r'\s*```$', '', cleaned)
    data = json.loads(cleaned)
    if not isinstance(data, dict):
        raise OutputParserException('story section must be a JSON object')
    return data


def story_request_estimate(participants: int, attempts: int = ATTEMPTS) -> dict:
    """Gemini calls for the story plus the signature card.

    The older path called each of the six sections on its own, up to three times,
    and called the signature prompt up to three times per person.
    """
    people = max(0, int(participants))
    before = people * PREVIOUS_SECTION_CALLS * PREVIOUS_ATTEMPTS + PREVIOUS_SECTION_CALLS * PREVIOUS_ATTEMPTS
    before += people * PREVIOUS_ATTEMPTS
    after_worst = (people + 1 + people) * attempts
    return {
        'before_worst': before,
        'after_typical': people + 1 + people,
        'after_worst': after_worst,
    }


def _ask(llm, prompt: str, accept) -> dict:
    last_error = None
    for _attempt in range(ATTEMPTS):
        try:
            return accept(_parse_json(_content(llm.invoke(prompt))))
        except (OutputParserException, json.JSONDecodeError, TypeError, ValueError) as exc:
            last_error = exc
            logging.warning('story reject: %s', exc)
    logging.warning('Story section kept the computed draft after %s', last_error)
    return {}


def _prompt(audience: str, section: str, evidence: str) -> str:
    schema = SCHEMAS[audience][section]
    base = render_section_prompt(audience=audience, section=section, evidence=evidence)
    return (
        f'{base}\n\nJSON schema, the only output. Do not write a digit inside a string. '
        f'Do not quote another person. Refer to facts by key.\n{schema}\n'
    )


def speaker_stats_for_story(transcript, speaker_map, label: str) -> dict:
    """Share, turns, and questions from the transcript itself."""
    rows = [row for row in (transcript.get('utterances') or []) if str(row.get('text') or '').strip()]

    def duration(row):
        return max(0.0, float(row.get('end') or 0) - float(row.get('start') or 0))

    total = sum(duration(row) for row in rows) or 1
    mine = [row for row in rows if row.get('speaker') == label]
    share = int(round(100 * sum(duration(row) for row in mine) / total))
    questions = [
        sum(str(row.get('text') or '').count('?') for row in rows if row.get('speaker') == key)
        for key in speaker_map
    ]
    average = int(round(sum(questions) / len(questions))) if questions else 0
    speakers = {row.get('speaker') for row in rows if row.get('speaker')}
    return {
        'share': share,
        'turns': len(mine),
        'questions': sum(str(row.get('text') or '').count('?') for row in mine),
        'session_turns': len(rows),
        'session_question_average': average,
        'speaker_count': max(1, len(speakers)),
        'events': quote_events(transcript, label),
    }


def _quote_lists(events: list[dict]) -> tuple[list[dict], list[dict]]:
    own = [{'time': event['time'], 'text': event['text']} for event in events]
    following = [
        {'time': event['next_time'], 'text': event['next_text']}
        for event in events
        if event.get('next_text')
    ]
    return own, following


def _progress_file(progress_dir, key: str):
    from pathlib import Path

    if not progress_dir:
        return None
    directory = Path(progress_dir)
    directory.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r'[^A-Za-z0-9._-]+', '_', key)
    return directory / f'story-{safe}.json'


def _load_progress(path) -> dict:
    if path is None or not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _store_progress(path, payload: dict) -> None:
    if path is None:
        return
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')


def _batch_prompt(audience: str, sections: tuple[str, ...], evidence: str) -> str:
    schemas = '\n'.join(f'{section}: {SCHEMAS[audience][section]}' for section in sections)
    return (
        f'BATCH: {audience}\n'
        f'Sections still needed: {", ".join(sections)}\n'
        f'{evidence}\n\n'
        'Return one JSON object whose keys are the section names still needed. '
        'Do not write a digit inside a string unless it is one of the fact values. '
        'Do not quote another person.\n'
        f'{schemas}\n'
    )


def _fill_sections(llm, *, audience, sections, evidence, render, progress_path, draft) -> None:
    pending = [section for section in sections if section not in (draft.get('accepted') or [])]
    if not pending:
        return
    for _attempt in range(ATTEMPTS):
        if not pending:
            break
        raw = _parse_safely(llm, _batch_prompt(audience, tuple(pending), evidence))
        if not isinstance(raw, dict):
            logging.warning('story reject [batch %s]: response was not an object', audience)
            continue
        still = []
        for section in pending:
            if section not in raw:
                logging.warning('story reject [%s]: missing from the batch', section)
                still.append(section)
                continue
            try:
                rendered = render(section, raw[section])
            except (OutputParserException, TypeError, ValueError, KeyError) as exc:
                logging.warning('story reject [%s]: %s', section, exc)
                still.append(section)
                continue
            draft.update(rendered)
            draft.setdefault('accepted', [])
            if section not in draft['accepted']:
                draft['accepted'].append(section)
            _store_progress(progress_path, draft)
        pending = still
    draft['fallback_sections'] = [section for section in sections if section not in (draft.get('accepted') or [])]


def _parse_safely(llm, prompt: str):
    try:
        return _parse_json(_content(llm.invoke(prompt)))
    except (OutputParserException, json.JSONDecodeError, TypeError, ValueError) as exc:
        logging.warning('story reject [batch]: %s', exc)
        return None


def generate_personal_draft(
    llm, *, transcript, label, info, stats, forbidden_names, progress_dir=None
) -> dict:
    facts = personal_facts(
        transcript=transcript,
        label=label,
        share=stats['share'],
        turns=stats['turns'],
        questions=stats['questions'],
        question_average=stats['session_question_average'],
        speaker_count=stats['speaker_count'],
    )
    own, following = _quote_lists(stats['events'])
    evidence = facts_prompt(facts, own, following)
    path = _progress_file(progress_dir, f'personal-{label}')
    saved = _load_progress(path)
    draft = {
        'transitions': pick_transitions(label),
        'fallback_sections': [],
        'accepted': list(saved.get('accepted') or []),
        'name_seed': info.get('name') or label,
    }
    for key, value in saved.items():
        if key not in ('accepted',):
            draft.setdefault(key, value)
    _fill_sections(
        llm,
        audience='personal',
        sections=PERSONAL_SECTIONS,
        evidence=evidence,
        render=lambda section, data: render_personal_section(
            section,
            data,
            transcript=transcript,
            label=label,
            facts=facts,
            forbidden=forbidden_names,
        ),
        progress_path=path,
        draft=draft,
    )
    return draft


def generate_group_draft(
    llm, *, transcript, speaker_map, participation, interaction, word_interaction, progress_dir=None
) -> dict:
    names = []
    name_by_label = {}
    for label, info in speaker_map.items():
        person = info.get('name')
        if person:
            name_by_label[label] = person
    for utterance in transcript.get('utterances') or []:
        person = name_by_label.get(utterance.get('speaker'))
        if person and (not names or names[-1] != person):
            names.append(person)
    roster = list(dict.fromkeys(name_by_label.values()))
    order, directed = directed_follows(names, roster)
    cast = cast_line(order, directed)
    facts = group_facts(
        participation=participation,
        interaction=interaction,
        word_interaction=word_interaction,
    )
    evidence = facts_prompt(
        facts,
        [],
        [],
        excerpt=transcript_excerpt(transcript, name_by_label),
    )
    evidence = f'{evidence}\nCAST\n{cast}'
    path = _progress_file(progress_dir, 'group')
    saved = _load_progress(path)
    draft = {
        'transitions': pick_transitions('group', group=True),
        'cast': cast,
        'fallback_sections': [],
        'accepted': list(saved.get('accepted') or []),
    }
    for key, value in saved.items():
        if key != 'accepted':
            draft.setdefault(key, value)
    _fill_sections(
        llm,
        audience='group',
        sections=GROUP_SECTIONS,
        evidence=evidence,
        render=lambda section, data: render_group_section(
            section, data, transcript=transcript, facts=facts, cast=cast
        ),
        progress_path=path,
        draft=draft,
    )
    return draft


def generate_report_stories(
    llm,
    *,
    transcript,
    speaker_map,
    participant_labels,
    participation,
    interaction,
    word_interaction,
    progress_dir=None,
) -> dict:
    """Personal drafts for participants, plus one group draft."""
    names_by_label = {label: info.get('name') or label for label, info in speaker_map.items()}
    personal = {}
    fallbacks = []
    for label in speaker_map:
        if label not in participant_labels:
            continue
        forbidden = [name for other, name in names_by_label.items() if other != label]
        draft = generate_personal_draft(
            llm,
            transcript=transcript,
            label=label,
            info=speaker_map[label],
            stats=speaker_stats_for_story(transcript, speaker_map, label),
            forbidden_names=forbidden,
            progress_dir=progress_dir,
        )
        personal[label] = draft
        fallbacks.extend(f'{label}:{section}' for section in draft.get('fallback_sections') or [])
    group = generate_group_draft(
        llm,
        transcript=transcript,
        speaker_map=speaker_map,
        participation=participation,
        interaction=interaction,
        word_interaction=word_interaction,
        progress_dir=progress_dir,
    )
    fallbacks.extend(f'group:{section}' for section in group.get('fallback_sections') or [])
    return {'personal': personal, 'group': group, 'fallbacks': fallbacks}
