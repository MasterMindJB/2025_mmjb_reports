"""Ask the model for structured story fields, then render the sentences ourselves."""

from __future__ import annotations

import json
import logging
import re

from langchain_core.exceptions import OutputParserException

from lib.chord_diagram import directed_follows
from lib.gemini_retry import invoke_llm
from lib.story_validation import ban_instruction
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
    unique_habits,
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
    mine_words = [len(str(row.get('text') or '').split()) for row in mine]
    all_words = [len(str(row.get('text') or '').split()) for row in rows]
    from lib.story_evidence import before_after

    early = late = None
    contrast = before_after(transcript, label)
    if contrast and ' then ' in contrast:
        early = int(re.search(r'\d+', contrast).group(0))
        late = int(re.search(r'then (\d+)', contrast).group(1))
    return {
        'share': share,
        'turns': len(mine),
        'questions': sum(str(row.get('text') or '').count('?') for row in mine),
        'session_turns': len(rows),
        'session_question_average': average,
        'speaker_count': max(1, len(speakers)),
        'turn_words': (sum(mine_words) / len(mine_words)) if mine_words else 0,
        'session_turn_words': (sum(all_words) / len(all_words)) if all_words else 0,
        'early': early,
        'late': late,
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


def _batch_prompt(audience: str, sections: tuple[str, ...], evidence: str, feedback: dict | None = None, turns: str = '') -> str:
    schemas = '\n'.join(f'{section}: {SCHEMAS[audience][section]}' for section in sections)
    if audience == 'personal':
        voice = (
            'Write every string to the reader as you. '
            'Do not say the speaker, the student, the participant, or the individual. '
            'Do not use he, she, or they for the reader, and do not capitalize a word mid-sentence. '
            'Do not write I will outside quotation marks. '
            'If the speaker recounts someone else\'s story, write "you told a student\'s story about" '
            'and do not treat that story as the reader\'s own life. '
            'A quote must be an exact substring of that speaker\'s turn, with its original capitalization. '
            'Do not drop or recapitalize a leading word. '
            f'{ban_instruction()} '
        )
    else:
        voice = (
            'Write about the room. Do not address a reader as you. '
            'The character headline describes the room, not one person and not a role. '
            'Do not say creative, professional, you, or the speaker. '
            'Character and challenge each need at least one rate fact, not all three. '
            'Challenge uses words. The rate is already in the facts. '
            'Resolution names who said the closing line, inside quote marks. '
            f'{ban_instruction()} '
        )
    rejected = ''
    if feedback:
        lines = ['The previous draft was rejected. Fix each reason and do not repeat it.']
        for section, reason in feedback.items():
            lines.append(f'- {section}: {reason}')
        if turns and any('quote' in str(reason).lower() for reason in feedback.values()):
            lines.append('That quote is not from this speaker. Here are this speaker\'s turns:')
            lines.append(turns)
        rejected = '\n'.join(lines) + '\n'
    return (
        f'BATCH: {audience}\n'
        f'Sections still needed: {", ".join(sections)}\n'
        f'{evidence}\n\n'
        'Return one JSON object whose keys are the section names still needed. '
        'Do not write a digit inside a string unless it is one of the fact values. '
        'Do not quote another person. '
        f'{voice}\n'
        f'{rejected}'
        f'{schemas}\n'
    )


def _coerce_payload(payload, section: str):
    """A JSON string, a fenced block, or a list of one object is still an object."""
    if isinstance(payload, dict):
        return payload
    logging.warning('story section %s arrived as %s', section, type(payload).__name__)
    if isinstance(payload, list) and len(payload) == 1:
        return _coerce_payload(payload[0], section)
    if isinstance(payload, str):
        text = payload.strip()
        if text.startswith('```'):
            text = re.sub(r'^```(?:json)?\s*', '', text)
            text = re.sub(r'\s*```$', '', text).strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            if section == 'goal' and text:
                from lib.story_render import verb_phrase

                try:
                    verb_phrase(text)
                except OutputParserException:
                    return payload
                logging.warning('story section goal arrived as non-JSON string')
                return {'claim': text}
            return payload
        if isinstance(parsed, (dict, list)):
            return _coerce_payload(parsed, section)
    return payload


def _fill_sections(llm, *, audience, sections, evidence, render, progress_path, draft, turns: str = '') -> None:
    pending = [section for section in sections if section not in (draft.get('accepted') or [])]
    if not pending:
        return
    reasons: dict[str, str] = {}
    for _attempt in range(ATTEMPTS):
        if not pending:
            break
        raw = _parse_safely(llm, _batch_prompt(audience, tuple(pending), evidence, reasons, turns))
        if not isinstance(raw, dict):
            logging.warning('story reject [batch %s]: response was not an object', audience)
            reasons['batch'] = 'the response was not a JSON object'
            continue
        still = []
        for section in pending:
            if section not in raw:
                logging.warning('story reject [%s]: missing from the batch', section)
                reasons[section] = 'missing from the batch'
                still.append(section)
                continue
            payload = _coerce_payload(raw[section], section)
            if not isinstance(payload, dict):
                logging.warning(
                    'story reject [%s]: section payload was not an object (raw type %s)',
                    section,
                    type(raw[section]).__name__,
                )
                reasons[section] = f'section payload must be an object (raw type {type(raw[section]).__name__})'
                still.append(section)
                continue
            try:
                rendered = render(section, payload)
            except (OutputParserException, TypeError, ValueError, KeyError, AttributeError) as exc:
                logging.warning('story reject [%s]: %s', section, exc)
                reasons[section] = str(exc)
                still.append(section)
                continue
            draft.update(rendered)
            draft.setdefault('accepted', [])
            if section not in draft['accepted']:
                draft['accepted'].append(section)
            _store_progress(progress_path, draft)
        pending = still
    draft['fallback_sections'] = [section for section in sections if section not in (draft.get('accepted') or [])]
    draft['rejection_reasons'] = {
        section: reasons[section]
        for section in draft['fallback_sections']
        if reasons.get(section)
    }


def _parse_safely(llm, prompt: str):
    try:
        return _parse_json(_content(invoke_llm(llm, prompt)))
    except (OutputParserException, json.JSONDecodeError, TypeError, ValueError) as exc:
        logging.warning('story reject [batch]: %s', exc)
        return None
    except Exception as exc:
        logging.warning('story call failed: %s', exc)
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
    own, _following = _quote_lists(stats['events'])
    # Other people's lines stay out of this prompt. Exact-quote checks still use the transcript.
    evidence = facts_prompt(facts, own, [])
    substantive = 0
    for event in stats['events']:
        words = re.findall(r"[A-Za-z']+", str(event.get('text') or ''))
        if len(words) >= 8 or (len(words) >= 4 and words and words[0].lower() not in {
            'yeah', 'yes', 'yep', 'right', 'ok', 'okay', 'sure', 'mm', 'hmm', 'uh', 'um',
        }):
            substantive += 1
    if substantive < 4:
        evidence += (
            '\nTHIN DATA: fewer than 4 substantive turns. '
            'Drama is about those short entries and when this person chose to come in. '
            'Resolution stays to two sentences. Use only the turns listed above.\n'
        )
    turns = '\n'.join(f'{item["time"]} {item["text"]}' for item in own)
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
        turns=turns,
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
    from lib.story_render import clip_clause

    durations = {}
    for row in transcript.get('utterances') or []:
        text = str(row.get('text') or '').strip()
        if not text:
            continue
        who = name_by_label.get(row.get('speaker')) or row.get('speaker') or 'a voice'
        span = max(0.0, float(row.get('end') or 0) - float(row.get('start') or 0))
        durations[who] = durations.get(who, 0.0) + span
    total = sum(durations.values()) or 1
    shares = [(name, int(round(100 * value / total))) for name, value in durations.items()]
    best = None
    best_count = -1
    for source, row in enumerate(directed):
        for target, count in enumerate(row):
            if source != target and count > best_count:
                best_count = count
                best = (order[source], order[target]) if source < len(order) and target < len(order) else None
    portrait = {}
    if best and best[0] and best[1]:
        portrait['pair'] = best
    if shares:
        portrait['least'] = min(shares, key=lambda item: item[1])[0]
        portrait['low_share'] = min(item[1] for item in shares)
        portrait['high_share'] = max(item[1] for item in shares)
    facts['_portrait'] = portrait
    from lib.story_render import closing_clip, last_substantive_utterance

    closing_row = last_substantive_utterance(transcript.get('utterances') or [])
    if closing_row:
        spoken = closing_clip(str(closing_row.get('text') or ''))
        if spoken:
            facts['_closing'] = {
                'name': name_by_label.get(closing_row.get('speaker')) or 'the last voice',
                'text': spoken,
            }
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
    profiles = []
    fallbacks = []
    for label in speaker_map:
        if label not in participant_labels:
            continue
        forbidden = [name for other, name in names_by_label.items() if other != label]
        stats = speaker_stats_for_story(transcript, speaker_map, label)
        draft = generate_personal_draft(
            llm,
            transcript=transcript,
            label=label,
            info=speaker_map[label],
            stats=stats,
            forbidden_names=forbidden,
            progress_dir=progress_dir,
        )
        personal[label] = draft
        profiles.append({'label': label, 'role': speaker_map[label].get('role'), **stats})
        fallbacks.extend(f'{label}:{section}' for section in draft.get('fallback_sections') or [])
    for label, (sentence, why) in unique_habits(profiles).items():
        personal[label]['one_thing'] = sentence
        personal[label]['why'] = why
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
    reasons = {}
    for label, draft in personal.items():
        for section, reason in (draft.get('rejection_reasons') or {}).items():
            reasons[f'{label}:{section}'] = reason
    for section, reason in (group.get('rejection_reasons') or {}).items():
        reasons[f'group:{section}'] = reason
    write_fallback_reasons(progress_dir, reasons)
    return {'personal': personal, 'group': group, 'fallbacks': fallbacks, 'rejection_reasons': reasons}


def write_fallback_reasons(progress_dir, reasons: dict) -> None:
    """The last rejection for each section that still fell back."""
    if not progress_dir:
        return
    from pathlib import Path

    directory = Path(progress_dir)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'fallbacks.json').write_text(json.dumps(reasons, indent=2), encoding='utf-8')
