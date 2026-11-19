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

ATTEMPTS = 3
PERSONAL_SECTIONS = ('character', 'goal', 'challenge', 'drama', 'resolution', 'one_thing')
GROUP_SECTIONS = PERSONAL_SECTIONS


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


def _ask(llm, prompt: str, accept) -> dict:
    last_error = None
    for _attempt in range(ATTEMPTS):
        try:
            return accept(_parse_json(_content(llm.invoke(prompt))))
        except (OutputParserException, json.JSONDecodeError, TypeError, ValueError) as exc:
            last_error = exc
            logging.info('Story section rejected, asking again: %s', exc)
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


def generate_personal_draft(llm, *, transcript, label, info, stats, forbidden_names) -> dict:
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
    draft = {'transitions': pick_transitions(label), 'fallback_sections': []}
    for section in PERSONAL_SECTIONS:
        rendered = _ask(
            llm,
            _prompt('personal', section, evidence),
            lambda data, section=section: render_personal_section(
                section,
                data,
                transcript=transcript,
                label=label,
                facts=facts,
                forbidden=forbidden_names,
            ),
        )
        if rendered:
            draft.update(rendered)
        else:
            draft['fallback_sections'].append(section)
    draft['name_seed'] = info.get('name') or label
    return draft


def generate_group_draft(llm, *, transcript, speaker_map, participation, interaction, word_interaction) -> dict:
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
    draft = {'transitions': pick_transitions('group'), 'cast': cast, 'fallback_sections': []}
    for section in GROUP_SECTIONS:
        rendered = _ask(
            llm,
            _prompt('group', section, evidence),
            lambda data, section=section: render_group_section(
                section, data, transcript=transcript, facts=facts, cast=cast
            ),
        )
        if rendered:
            draft.update(rendered)
        else:
            draft['fallback_sections'].append(section)
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
    )
    fallbacks.extend(f'group:{section}' for section in group.get('fallback_sections') or [])
    return {'personal': personal, 'group': group, 'fallbacks': fallbacks}
