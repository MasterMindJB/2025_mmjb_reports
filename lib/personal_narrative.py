"""Personal narrative: code picks the evidence, one call writes the sections."""

from __future__ import annotations

import json
import logging
import re

from lib.readability import reading_grade
from lib.story_render import (
    question_number_line,
    room_average,
    room_median,
    talk_number_line,
    turns_sentence,
)

PROMPT_MARK = 'PERSONAL_NARRATIVE'
_DANGLING = {
    'to', 'and', 'the', 'a', 'an', 'of', 'for', 'with', 'in', 'on', 'at',
    'from', 'by', 'or', 'as', 'if', 'that', 'so',
}
_BANNED = (
    'interaction', 'participation', 'talk time', 'focused dialogue',
    'this is the part to change', 'who else should speak about',
    'you name ',
)


def _words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z']+", text or '')


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.findall(r'[^.!?]*[.!?]', text or '') if part.strip()]


def good_sentence(sentence: str) -> bool:
    """A usable sentence is complete, at least six words, and does not end on 'to.'"""
    words = _words(sentence)
    if len(words) < 6 or not sentence.endswith(('.', '!', '?')):
        return False
    return words[-1].lower() not in _DANGLING


def candidate_quotes(turns: list[str]) -> list[str]:
    """Up to three quotes. Each is one or two complete sentences, at most 40 words."""
    found: list[str] = []
    for turn in turns:
        sentences = [sentence for sentence in _sentences(turn) if good_sentence(sentence)]
        options = list(sentences)
        for first, second in zip(sentences, sentences[1:]):
            options.append(f'{first} {second}')
        for option in options:
            if len(_words(option)) > 40 or option in found:
                continue
            found.append(option)
            if len(found) == 3:
                return found
    return found


def speaker_rows(transcript: dict | None, label: str) -> list[dict]:
    return [
        row for row in (transcript or {}).get('utterances') or []
        if row.get('speaker') == label and str(row.get('text') or '').strip()
    ]


def word_total(rows: list[dict]) -> int:
    return sum(len(_words(str(row.get('text') or ''))) for row in rows)


def substantive_count(rows: list[dict]) -> int:
    count = 0
    for row in rows:
        if any(good_sentence(sentence) for sentence in _sentences(str(row.get('text') or ''))):
            count += 1
    return count


def needs_short_report(rows: list[dict]) -> bool:
    return substantive_count(rows) < 3 or word_total(rows) < 60


def turn_counts(transcript: dict | None) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in (transcript or {}).get('utterances') or []:
        if not str(row.get('text') or '').strip():
            continue
        label = str(row.get('speaker') or '')
        if not label:
            continue
        counts[label] = counts.get(label, 0) + 1
    return counts


def computed_numbers(transcript, label: str, stats: dict) -> tuple[list[str], set[str]]:
    """Number lines built only from counts. The median is the room's own middle value."""
    counts = turn_counts(transcript)
    mine = counts.get(label, int(stats.get('turns') or 0))
    median = room_median(counts.values()) if counts else int(stats.get('turns') or 0)
    share = int(stats.get('share') or 0)
    speakers = int(stats.get('speaker_count') or len(counts) or 1)
    questions = int(stats.get('questions') or 0)
    average = int(stats.get('session_question_average') or 0)
    lines = [
        talk_number_line(share, speakers),
        question_number_line(questions, average),
        turns_sentence(mine, median if median is not None else mine),
    ]
    allowed = {
        str(share), str(room_average(speakers)), str(mine), str(median if median is not None else mine),
        str(questions), str(average),
    }
    return lines, allowed


def _prompt(name: str, lines: list[str], quotes: list[str], failures: list[str]) -> str:
    quote_block = '\n'.join(f'{index + 1}. "{quote}"' for index, quote in enumerate(quotes)) or 'None.'
    number_block = '\n'.join(f'- {line}' for line in lines)
    retry = ''
    if failures:
        retry = 'The last draft failed:\n' + '\n'.join(f'- {item}' for item in failures) + '\n'
    return (
        f'{PROMPT_MARK}\n'
        f'Write the personal report for {name}. Use only the evidence below.\n'
        'Style: fifth-grade words, short sentences, second person. '
        'Headline: one real sentence of 6 words or fewer. '
        'one_big_thing: the pattern, then quote 1, then a question that names a person or a concrete thing from that quote. '
        'Do not write "who else should speak about". '
        'why_it_matters and yes_but: second person, and end with what to try next time. '
        'next_rep: a script plus one target number copied from the numbers below. Do not invent a word target. '
        'number_lines: copy the number lines exactly. '
        'said_quote: copy quote 2 exactly, or "" if there is no quote 2. Do not reuse quote 1. '
        'Never name a metric. Never use interaction, participation, or talk time.\n'
        f'{retry}'
        f'NUMBERS\n{number_block}\n'
        f'QUOTES\n{quote_block}\n'
        'Return JSON only with keys headline, one_big_thing, why_it_matters, yes_but, next_rep, number_lines, said_quote.\n'
    )


def _numbers_outside(sentence: str) -> list[str]:
    outside = re.sub(r'"[^"]*"', ' ', sentence or '')
    outside = re.sub(r'\b\d+:\d{2}(?::\d{2})?\b', ' ', outside)
    return re.findall(r'\d+', outside)


def validate_narrative(payload: dict, quotes: list[str], allowed: set[str]) -> list[str]:
    problems = []
    if not isinstance(payload, dict):
        return ['response was not an object']
    headline = str(payload.get('headline') or '').strip()
    if len(headline.split()) > 6 or reading_grade(headline) > 6 or 'focused dialogue' in headline.lower():
        problems.append('headline must be one plain sentence of 6 words or fewer')
    fields = {
        'one_big_thing': str(payload.get('one_big_thing') or ''),
        'why_it_matters': str(payload.get('why_it_matters') or ''),
        'yes_but': str(payload.get('yes_but') or ''),
        'next_rep': str(payload.get('next_rep') or ''),
    }
    if '?' not in re.sub(r'"[^"]*"', ' ', fields['one_big_thing']):
        problems.append('one_big_thing needs a question')
    for name, text in fields.items():
        plain = re.sub(r'"[^"]*"', ' ', text)
        if reading_grade(text) > 6:
            problems.append(f'{name} is above a sixth-grade reading level')
        lowered = plain.lower()
        for phrase in _BANNED:
            if phrase in lowered:
                problems.append(f'{name} uses {phrase}')
        if re.search(r"\b(I|I'll|I'm|I've|I'd)\b", plain):
            problems.append(f'{name} has an unquoted first-person line')
        for sentence in re.split(r'(?<=[.!?])\s+', text.strip()):
            if len(_numbers_outside(sentence)) > 1:
                problems.append(f'{name} has two numbers in one sentence')
        if text and 'next time' not in text.lower() and '?' not in plain:
            problems.append(f'{name} does not end with a next step')
        for span in re.findall(r'"([^"]*)"', text):
            if span not in quotes:
                problems.append(f'{name} quotes text that is not a candidate')
            if len(_words(span)) > 40 or not good_sentence(span.split('. ')[0] if span else ''):
                if span not in quotes:
                    problems.append(f'{name} quote is not a complete candidate')
    said = str(payload.get('said_quote') or '').strip().strip('"')
    expected = quotes[1] if len(quotes) > 1 else ''
    if said != expected:
        problems.append('said_quote must be the second candidate')
    for line in payload.get('number_lines') or []:
        for number in _numbers_outside(str(line)):
            if number not in allowed:
                problems.append(f'number_lines uses {number}, which was not computed')
    next_numbers = _numbers_outside(fields['next_rep'])
    if any(number not in allowed for number in next_numbers):
        problems.append('next_rep uses a number that was not computed')
    return problems


def _parse(raw) -> dict | None:
    content = raw.content if hasattr(raw, 'content') else raw
    if isinstance(content, list):
        content = ' '.join(str(part) for part in content)
    text = str(content or '').strip()
    if text.startswith('```'):
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def short_story(name: str, rows: list[dict], lines: list[str], quotes: list[str]) -> dict:
    count = len(rows)
    spoken = 'You spoke once.' if count == 1 else f'You spoke {count} times.'
    quote = quotes[0] if quotes else ''
    return {
        'short_mode': True,
        'headline': spoken,
        'one_big_thing': '',
        'why_it_matters': '',
        'yes_but': '',
        'next_rep': 'Next time, add one sentence before you stop.',
        'numbers': lines,
        'said_quote': quote,
        'big_quote': quote,
        'drama': quote,
        'goal': '',
        'challenge': '',
        'one_thing': 'Next time, add one sentence before you stop.',
        'why': '',
        'turns': count,
        'fallback_sections': [],
        'name_seed': name,
    }


def full_story(payload: dict, lines: list[str], name: str) -> dict:
    one_big = str(payload.get('one_big_thing') or '').strip()
    why = str(payload.get('why_it_matters') or '').strip()
    yes_but = str(payload.get('yes_but') or '').strip()
    next_rep = str(payload.get('next_rep') or '').strip()
    return {
        'short_mode': False,
        'headline': str(payload.get('headline') or '').strip(),
        'one_big_thing': one_big,
        'why_it_matters': why,
        'yes_but': yes_but,
        'next_rep': next_rep,
        'numbers': lines,
        'said_quote': str(payload.get('said_quote') or '').strip().strip('"'),
        'big_quote': '',
        'drama': one_big,
        'goal': why,
        'challenge': yes_but,
        'one_thing': next_rep,
        'why': '',
        'fallback_sections': [],
        'name_seed': name,
    }


def build_personal_story(llm, transcript, label: str, info: dict, stats: dict) -> dict:
    """One call when there is enough speech. Otherwise, or after a failed retry, the short report."""
    rows = speaker_rows(transcript, label)
    quotes = candidate_quotes([str(row.get('text') or '') for row in rows])
    lines, allowed = computed_numbers(transcript, label, stats)
    name = info.get('name') or label
    if needs_short_report(rows):
        return short_story(name, rows, lines, quotes)
    failures: list[str] = []
    for _attempt in range(2):
        try:
            payload = _parse(llm.invoke(_prompt(name, lines, quotes, failures)))
        except Exception as exc:
            logging.warning('personal narrative call failed: %s', exc)
            payload = None
        if payload is None:
            failures = ['response was not JSON']
            continue
        failures = validate_narrative(payload, quotes, allowed)
        if not failures:
            return full_story(payload, lines, name)
    logging.warning('personal narrative fell back to the short report: %s', failures)
    return short_story(name, rows, lines, quotes)
