"""Ask the model for each story section, then check the craft rules."""

from __future__ import annotations

import json
import logging
import re

from langchain_core.exceptions import OutputParserException

from lib.chord_diagram import directed_follows
from lib.story_craft import render_section_prompt
from lib.story_evidence import (
    cast_line,
    group_evidence,
    halfway_target,
    personal_evidence,
    quote_events,
)
from lib.story_validation import reject_craft_violations

PERSONAL_SECTIONS = ('character', 'goal', 'challenge', 'drama', 'resolution', 'one_thing')
GROUP_SECTIONS = PERSONAL_SECTIONS
ATTEMPTS = 3


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


def _check(text, **kwargs) -> str:
    return reject_craft_violations(text, **kwargs)


def _ask(llm, prompt: str, checks: list[tuple[str, dict]]) -> dict:
    last_error = None
    for _attempt in range(ATTEMPTS):
        try:
            data = _parse_json(_content(llm.invoke(prompt)))
            cleaned = {}
            for key, options in checks:
                if key not in data:
                    raise OutputParserException(f'story section is missing {key}')
                cleaned[key] = _check(str(data[key]), **options)
            return cleaned
        except (OutputParserException, json.JSONDecodeError, TypeError, ValueError) as exc:
            last_error = exc
            logging.info('Story section rejected, asking again: %s', exc)
    logging.warning('Story section kept the computed draft after %s', last_error)
    return {}


def _personal_packet(transcript, label, info, stats) -> tuple[str, set[str], int]:
    return personal_evidence(
        transcript=transcript,
        label=label,
        name=info.get('name') or label,
        share=stats['share'],
        turns=stats['turns'],
        questions=stats['questions'],
        session_turns=stats['session_turns'],
        session_question_average=stats['session_question_average'],
    )


def generate_personal_draft(llm, *, transcript, label, info, stats, forbidden_names) -> dict:
    evidence, allowed, _target = _personal_packet(transcript, label, info, stats)
    common = {
        'transcript': transcript,
        'allowed_numbers': allowed,
        'forbidden_names': forbidden_names,
    }
    draft = {'transitions': {}}
    character = _ask(
        llm,
        render_section_prompt(audience='personal', section='character', evidence=evidence),
        [
            ('headline', {**common, 'headline': True}),
            ('body', common),
            ('transition', common),
        ],
    )
    if character:
        draft['headline'] = character['headline']
        draft['character_body'] = character['body']
        draft['transitions']['after_character'] = character['transition']
    for section, field, seam in (
        ('goal', 'goal', 'after_goal'),
        ('challenge', 'challenge', 'after_challenge'),
        ('drama', 'drama', 'after_drama'),
        ('resolution', 'resolution', 'after_resolution'),
    ):
        piece = _ask(
            llm,
            render_section_prompt(audience='personal', section=section, evidence=evidence),
            [('body', common), ('transition', common)],
        )
        if piece:
            draft[field] = piece['body']
            draft['transitions'][seam] = piece['transition']
    one = _ask(
        llm,
        render_section_prompt(audience='personal', section='one_thing', evidence=evidence),
        [
            ('one_thing', {**common, 'one_thing': True}),
            ('why', common),
        ],
    )
    if one:
        draft['one_thing'] = one['one_thing']
        draft['why'] = one['why']
    return draft


def generate_group_draft(llm, *, transcript, speaker_map, participation, interaction, word_interaction) -> dict:
    names = []
    for utterance in transcript.get('utterances') or []:
        label = utterance.get('speaker')
        name = (speaker_map.get(label) or {}).get('name')
        if name and (not names or names[-1] != name):
            names.append(name)
    roster = []
    for info in speaker_map.values():
        person = info.get('name')
        if person and person not in roster:
            roster.append(person)
    _order, directed = directed_follows(names, roster)
    cast = cast_line(_order, directed)
    current = int(round(100 * interaction))
    target = halfway_target(current, 70 if current < 70 else current)
    evidence, allowed = group_evidence(
        participation=participation,
        interaction=interaction,
        word_interaction=word_interaction,
        cast=cast,
        target=target,
    )
    common = {'transcript': transcript, 'allowed_numbers': allowed, 'forbidden_names': []}
    draft = {'transitions': {}, 'cast': cast}
    character = _ask(
        llm,
        render_section_prompt(audience='group', section='character', evidence=evidence),
        [
            ('headline', {**common, 'headline': True}),
            ('body', common),
            ('transition', common),
        ],
    )
    if character:
        draft['headline'] = character['headline']
        draft['character'] = character['body']
        draft['transitions']['after_character'] = character['transition']
    for section, field, seam in (
        ('goal', 'goal', 'after_goal'),
        ('challenge', 'challenge', 'after_challenge'),
        ('drama', 'drama', 'after_drama'),
        ('resolution', 'resolution', 'after_resolution'),
    ):
        piece = _ask(
            llm,
            render_section_prompt(audience='group', section=section, evidence=evidence),
            [('body', common), ('transition', common)],
        )
        if piece:
            draft[field] = piece['body']
            draft['transitions'][seam] = piece['transition']
    one = _ask(
        llm,
        render_section_prompt(audience='group', section='one_thing', evidence=evidence),
        [
            ('one_thing', {**common, 'one_thing': True}),
            ('why', common),
        ],
    )
    if one:
        draft['one_thing'] = one['one_thing']
        draft['why'] = one['why']
    return draft


def speaker_stats_for_story(transcript, speaker_map, label: str) -> dict:
    """Share, turns, and questions from the transcript itself."""
    rows = [row for row in (transcript.get('utterances') or []) if str(row.get('text') or '').strip()]
    def duration(row):
        return max(0.0, float(row.get('end') or 0) - float(row.get('start') or 0))

    total = sum(duration(row) for row in rows) or 1
    mine = [row for row in rows if row.get('speaker') == label]
    share = int(round(100 * sum(duration(row) for row in mine) / total))
    questions = [sum(str(row.get('text') or '').count('?') for row in rows if row.get('speaker') == key) for key in speaker_map]
    average = int(round(sum(questions) / len(questions))) if questions else 0
    return {
        'share': share,
        'turns': len(mine),
        'questions': sum(str(row.get('text') or '').count('?') for row in mine),
        'session_turns': len(rows),
        'session_question_average': average,
        'events': quote_events(transcript, label),
    }


def generate_report_stories(
    llm,
    *,
    transcript,
    speaker_map,
    participant_labels,
    participation,
    interaction,
    word_interaction,
) -> dict:
    """Personal drafts for participants, plus one group draft."""
    names_by_label = {label: info.get('name') or label for label, info in speaker_map.items()}
    personal = {}
    for label in speaker_map:
        if label not in participant_labels:
            continue
        forbidden = [name for other, name in names_by_label.items() if other != label]
        personal[label] = generate_personal_draft(
            llm,
            transcript=transcript,
            label=label,
            info=speaker_map[label],
            stats=speaker_stats_for_story(transcript, speaker_map, label),
            forbidden_names=forbidden,
        )
    group = generate_group_draft(
        llm,
        transcript=transcript,
        speaker_map=speaker_map,
        participation=participation,
        interaction=interaction,
        word_interaction=word_interaction,
    )
    return {'personal': personal, 'group': group}
