"""Three personal sentences from one Gemini 3.6 call. No template filler."""

from __future__ import annotations

import logging
import re

from lib.names import is_placeholder_name

_BANNED = ('should', 'try to', 'next time')
_OPENERS = (
    'You seem to care about',
    'When you said "',
    'As a point of curiosity, I wonder how',
)


def _norm(text: str) -> str:
    return re.sub(r'[^a-z0-9]+', '', str(text or '').lower())


def _quote_span(sentence: str) -> str:
    match = re.search(r'When you said "(.+?)"\s*,\s*it felt', str(sentence or ''), re.I | re.S)
    return match.group(1) if match else ''


def bullet_problems(sentences: list[str], speech: str) -> list[str]:
    """The only gates: opener, a verbatim quote, one 'I wonder', and no advice."""
    reasons = []
    lines = [str(item or '').strip() for item in (sentences or [])]
    while len(lines) < 3:
        lines.append('')
    for line, opener in zip(lines, _OPENERS):
        if not line.startswith(opener):
            reasons.append(f'does not start with {opener}')
    quote = _quote_span(lines[1])
    if not quote or _norm(quote) not in _norm(speech):
        reasons.append('quote is not a verbatim substring of this person\'s speech')
    wonder = ' '.join(lines).count('I wonder')
    if wonder != 1:
        reasons.append(f'I wonder appears {wonder} times')
    lowered = ' '.join(lines).lower()
    for banned in _BANNED:
        if banned in lowered:
            reasons.append(f'contains {banned}')
    return reasons


def passing_sentences(sentences: list[str], speech: str) -> list[str]:
    """Keep a sentence only when it passes on its own. Never invent a replacement."""
    lines = [str(item or '').strip() for item in (sentences or [])]
    while len(lines) < 3:
        lines.append('')
    kept = []
    for index, line in enumerate(lines):
        if not line.startswith(_OPENERS[index]):
            continue
        if index == 1 and _norm(_quote_span(line)) not in _norm(speech):
            continue
        if any(banned in line.lower() for banned in _BANNED):
            continue
        if index == 2 and line.count('I wonder') != 1:
            continue
        if index != 2 and 'I wonder' in line:
            continue
        kept.append(line)
    if sum(line.count('I wonder') for line in kept) > 1:
        return [line for line in kept if 'I wonder' not in line][:1]
    return kept


def nearby_context(transcript: dict | None, label: str, limit: int = 2) -> list[str]:
    """One or two lines spoken by other people next to this person's turns."""
    rows = list((transcript or {}).get('utterances') or [])
    found: list[str] = []
    for index, row in enumerate(rows):
        if str(row.get('speaker') or '') != str(label):
            continue
        for neighbor in (index - 1, index + 1):
            if not 0 <= neighbor < len(rows):
                continue
            other = rows[neighbor]
            if str(other.get('speaker') or '') == str(label):
                continue
            text = ' '.join(str(other.get('text') or '').split())
            if text and text not in found:
                found.append(text)
            if len(found) >= limit:
                return found
    return found


def simple_prompt(speech: str, context: list[str], errors: list[str] | None = None) -> str:
    context_block = '\n'.join(f'- {line}' for line in context) or 'None.'
    retry = ''
    if errors:
        retry = 'The last draft failed:\n' + '\n'.join(f'- {item}' for item in errors) + '\nWrite the three sentences again.\n'
    return (
        'SIMPLE_BULLETS\n'
        'Write three plain sentences at about a fifth-grade level. Address the person as you. '
        'Be charitable. Do not give advice. Do not use should, try to, or next time.\n'
        'Return JSON only with keys s1, s2, and s3.\n'
        's1: You seem to care about {a 2 to 4 word value}, and it seemed like you were trying to '
        '{a 3 to 6 word goal that starts with a verb}.\n'
        's2: When you said "{exact verbatim words from their speech, 8 to 20 words, a complete thought}", '
        'it felt like {a short generous reading in second person, 12 words or fewer}.\n'
        's3: As a point of curiosity, I wonder how {your plus an observed behavior, 3 to 7 words} '
        'affects {a plain group or team effect, 2 to 5 words}.\n'
        'Use I wonder only in s3.\n'
        'Example from a made-up speaker named Dana:\n'
        's1: You seem to care about clear work, and it seemed like you were trying to show the point.\n'
        's2: When you said "the draft can wait until the room is ready", it felt like you wanted the work to be honest.\n'
        's3: As a point of curiosity, I wonder how your care for the draft affects how the group decides.\n'
        f'{retry}'
        f'THEIR SPEECH\n{speech or "None."}\n'
        f'NEARBY LINES FROM OTHER PEOPLE\n{context_block}\n'
    )


def _sentences_from(payload: dict | None) -> list[str]:
    data = payload or {}
    return [str(data.get(key) or '').strip() for key in ('s1', 's2', 's3')]


def generate_simple_bullets(llm, transcript: dict | None, label: str, name: str, speech: str) -> list[str]:
    """One 3.6 call. One retry with the error. Then only the sentences that passed."""
    if not speech.strip() or is_placeholder_name(name):
        return []
    from lib.personal_narrative import _invoke_personal, _parse, _raw_text, upgrade_model_name

    context = nearby_context(transcript, label)
    errors: list[str] = []
    last = ['', '', '']
    for attempt in range(2):
        prompt = simple_prompt(speech, context, errors if attempt else None)
        try:
            raw = _invoke_personal(
                llm, prompt, upgrade_model_name(), max_output=1024, response_mime='application/json',
            )
            payload = _parse(_raw_text(raw))
        except Exception as exc:
            logging.warning('simple bullets call failed for %s: %s', name, exc)
            errors = [f'the call failed: {exc}']
            continue
        last = _sentences_from(payload)
        errors = bullet_problems(last, speech)
        if not errors:
            return last
        logging.warning('simple bullets rejected for %s: %s', name, '; '.join(errors))
    kept = passing_sentences(last, speech)
    if errors:
        logging.warning('simple bullets shipped %s of 3 for %s: %s', len(kept), name, '; '.join(errors))
    return kept
