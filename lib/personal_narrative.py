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
_FUNCTION = {
    'the', 'a', 'an', 'of', 'for', 'with', 'in', 'on', 'at', 'from', 'by', 'or',
    'as', 'if', 'that', 'so', 'to', 'and', 'we', 'you', 'i', 'it', 'is', 'are',
    'was', 'were', 'be', 'been', 'will', 'would', 'could', 'should', 'all', 'us',
    'our', 'probably', 'just', 'really', 'very', 'well', 'yeah', 'yes', 'right',
    'think', 'know', 'mean', 'like', 'this', 'these', 'those',
}
_FILLER_PREFIX = re.compile(
    r'^(?:(?:well|so|okay|ok|yeah|yes|right|um+|uh+)\s*,?\s*|(?:i think|you know|i mean)\s*,?\s*)+',
    re.I,
)
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


def _content_words(sentence: str) -> list[str]:
    return [word for word in _words(sentence) if word.lower() not in _FUNCTION and len(word) > 3]


def _strip_filler(sentence: str) -> str:
    cleaned = _FILLER_PREFIX.sub('', sentence or '').strip()
    if cleaned[:1].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned


def candidate_quotes(turns: list[str]) -> list[str]:
    """Up to three quotes, preferring a real claim over filler. Mid-turn sentences count."""
    scored: list[tuple[int, str]] = []
    for turn in turns:
        sentences = [_strip_filler(sentence) for sentence in _sentences(turn)]
        sentences = [sentence for sentence in sentences if good_sentence(sentence)]
        options = list(sentences)
        for first, second in zip(sentences, sentences[1:]):
            options.append(f'{first} {second}')
        for option in options:
            if len(_words(option)) > 40:
                continue
            if re.search(r'\ball of us will probably agree\b', option, re.I):
                continue
            content = _content_words(option)
            if len(content) < 3:
                continue
            scored.append((len(content), option))
    scored.sort(key=lambda item: (-item[0], len(item[1])))
    found: list[str] = []
    for _score, option in scored:
        if option not in found:
            found.append(option)
        if len(found) == 3:
            break
    return found


def suggested_next_rep(rows: list[dict], stats: dict) -> str:
    """A next step that matches how this person actually spoke."""
    turns = len(rows)
    words = word_total(rows)
    share = int(stats.get('share') or 0)
    speakers = max(int(stats.get('speaker_count') or 1), 1)
    if turns <= 2 and words >= 60:
        return 'Next time, stop after two sentences and ask the room a question.'
    if share > room_average(speakers) + 2:
        return 'Next time, stop after one point and let someone else speak.'
    if turns < 3:
        return 'Next time, add one more turn before the close.'
    return 'Next time, ask the room one question before you add a point.'


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
    """Only a small amount of speech. A long monologue is still a full report."""
    return word_total(rows) < 60


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


def _prompt(name: str, lines: list[str], quotes: list[str], failures: list[str], phrase: str, next_rep: str) -> str:
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
        'next_rep should follow the suggested next step unless the evidence shows a better fit. '
        'Never name a metric. Never use interaction, participation, or talk time.\n'
        f'{retry}'
        f'NUMBERS\n{number_block}\n'
        f'QUOTES\n{quote_block}\n'
        f'REPEATED PHRASE\n{phrase or "None."}\n'
        f'SUGGESTED NEXT STEP\n{next_rep}\n'
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


def short_story(name: str, rows: list[dict], lines: list[str], quotes: list[str], stats: dict) -> dict:
    count = len(rows)
    quote = quotes[0] if quotes else ''
    return {
        'short_mode': True,
        'headline': 'A short turn.' if count <= 1 else 'A few short turns.',
        'one_big_thing': '',
        'why_it_matters': '',
        'yes_but': '',
        'next_rep': suggested_next_rep(rows, stats),
        'numbers': lines,
        'said_quote': quote,
        'big_quote': quote,
        'drama': quote,
        'goal': '',
        'challenge': '',
        'one_thing': suggested_next_rep(rows, stats),
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
    step = suggested_next_rep(rows, stats)
    phrase = _repeated_phrase(rows)
    if needs_short_report(rows):
        return short_story(name, rows, lines, quotes, stats)
    failures: list[str] = []
    for _attempt in range(2):
        try:
            payload = _parse(llm.invoke(_prompt(name, lines, quotes, failures, phrase, step)))
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
    return short_story(name, rows, lines, quotes, stats)


def _repeated_phrase(rows: list[dict]) -> str:
    from lib.report_stats import display_bigram, top_bigram

    found = top_bigram(' '.join(str(row.get('text') or '') for row in rows))
    if not found:
        return ''
    phrase, count = found
    return f'{display_bigram(phrase)} ({count})'
