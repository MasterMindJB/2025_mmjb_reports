"""Reject a story draft that breaks the craft rules.

A hit raises OutputParserException so the generator can ask again.
"""

from __future__ import annotations

import re

from langchain_core.exceptions import OutputParserException

BAN_PHRASES = (
    'leverage',
    'synergy',
    'actionable insights',
    'growth mindset',
    'journey',
    'unlock',
    'empower',
    'best practices',
    'communication style',
    'stakeholder alignment',
    'coaching',
    'workshop',
    'soft skills',
    'you value connection',
    'you sometimes doubt yourself',
    'you failed to',
    'you should have',
    'unfortunately',
    'great job',
    'amazing',
    'crushed it',
    'in summary',
    'people who say',
)
HEDGES = ('maybe', 'might', 'perhaps', 'kind of', 'sort of')
ALLOWED_ONE_THING_VERBS = ('pause', 'ask', 'name', 'repeat', 'count', 'stop', 'wait', 'say')
BANNED_ONE_THING_VERBS = ('be', 'try', 'consider', 'improve', 'leverage')
EMOJI = re.compile('[\U0001f300-\U0001faff\U00002600-\U000027bf]')


def _fail(message: str) -> None:
    raise OutputParserException(message)


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9']+", text or ''))


def _normalize(text: str) -> str:
    return re.sub(r'[^a-z0-9]+', '', (text or '').lower())


def quoted_spans(text: str) -> list[str]:
    """Double quotes, curly quotes, and single-quoted spans of three or more words.

    Apostrophes inside words, as in "Here's", are not quotes.
    """
    spans = re.findall(r'"([^"]+)"', text or '')
    spans += re.findall(r'“([^”]+)”', text or '')
    for match in re.finditer(r"(?<![A-Za-z])'([^']+)'(?![A-Za-z])", text or ''):
        if len(match.group(1).split()) >= 3:
            spans.append(match.group(1))
    unique = []
    for span in spans:
        cleaned = span.strip()
        if cleaned and cleaned not in unique:
            unique.append(cleaned)
    return unique


def quotes_are_in_transcript(text: str, transcript: dict | None) -> None:
    blob = _normalize(
        ' '.join(str(row.get('text') or '') for row in (transcript or {}).get('utterances') or [])
    )
    for span in quoted_spans(text):
        needle = _normalize(span)
        if len(needle) < 4:
            continue
        if needle not in blob:
            _fail(f'quoted span is not in the transcript: {span}')


def reject_craft_violations(
    text: str,
    *,
    transcript: dict | None = None,
    allowed_numbers: set[str] | None = None,
    forbidden_names: list[str] | None = None,
    headline: bool = False,
    one_thing: bool = False,
) -> str:
    """Return the cleaned text, or raise OutputParserException."""
    cleaned = (text or '').strip()
    if not cleaned:
        _fail('story text is empty')
    lowered = cleaned.lower()
    for phrase in BAN_PHRASES:
        if phrase in lowered:
            _fail(f'story text contains {phrase!r}')
    if '!' in cleaned:
        _fail('story text contains an exclamation point')
    if EMOJI.search(cleaned):
        _fail('story text contains an emoji')
    hedge_hits = 0
    for hedge in HEDGES:
        hedge_hits += len(re.findall(rf'\b{re.escape(hedge)}\b', lowered))
    if hedge_hits > 1:
        _fail('story text stacks hedges')
    for name in forbidden_names or []:
        if name and name.lower() in lowered:
            _fail(f'story text names a peer: {name}')
    if allowed_numbers is not None:
        extra = [number for number in re.findall(r'\d+', cleaned) if number not in allowed_numbers]
        if extra:
            _fail(f'story text invents numbers: {extra}')
    if transcript is not None:
        quotes_are_in_transcript(cleaned, transcript)
    if headline and word_count(cleaned) > 12:
        _fail(f'headline is {word_count(cleaned)} words; the maximum is 12')
    if one_thing:
        if word_count(cleaned) > 25:
            _fail(f'one thing is {word_count(cleaned)} words; the maximum is 25')
        if not re.search(r'\d', cleaned):
            _fail('one thing needs a target number')
        if re.search(r'\balso\b', lowered):
            _fail("one thing must not say 'also'")
        first = re.findall(r"[A-Za-z']+", cleaned)
        opening = first[0].lower() if first else ''
        if opening not in ALLOWED_ONE_THING_VERBS:
            _fail(f"one thing must start with an allowed verb, not {opening!r}")
        for verb in BANNED_ONE_THING_VERBS:
            if re.search(rf'\b{verb}\b', lowered):
                _fail(f'one thing uses banned verb {verb!r}')
        if re.search(r'\bwork on\b', lowered) or re.search(r'\bfocus on\b', lowered):
            _fail('one thing uses a banned verb')
    return cleaned
