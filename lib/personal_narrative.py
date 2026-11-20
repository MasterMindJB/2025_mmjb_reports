"""Personal narrative: code picks the evidence, one call writes the sections."""

from __future__ import annotations

import json
import logging
import os
import re

from lib.readability import reading_grade
from lib.story_render import (
    question_number_line,
    room_average,
    room_median,
    session_talk_phrase,
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
    """Up to three quotes. Each one is a single sentence copied from the turn."""
    scored: list[tuple[int, str]] = []
    for turn in turns:
        for sentence in _sentences(turn):
            if sentence not in turn or not good_sentence(sentence):
                continue
            if re.search(r'\ball of us will probably agree\b', sentence, re.I):
                continue
            content = _content_words(sentence)
            # "All of us will probably agree" has no concrete claim. A real noun keeps it.
            if len(content) < 2:
                continue
            # Filler can stay in the words we print. It only loses the ranking.
            score = len(content)
            if _FILLER_PREFIX.match(sentence):
                score -= 2
            scored.append((score, sentence))
    scored.sort(key=lambda item: (-item[0], len(item[1])))
    found: list[str] = []
    for _score, option in scored:
        if option not in found:
            found.append(option)
        if len(found) == 3:
            break
    return found


_MOMENT_SKIP = _FUNCTION | {
    'about', 'after', 'before', 'there', 'their', 'which', 'where', 'while', 'being',
    'other', 'every', 'still', 'those', 'first', 'going', 'gonna', 'people', 'really',
    'thing', 'would', 'could', 'should', 'something', 'anything', 'please', 'thank',
}


def moment_word(quotes: list[str], rows: list[dict]) -> str:
    """One concrete word from this person's own words, so the next step is not shared."""
    source = ' '.join(quotes) if quotes else ' '.join(str(row.get('text') or '') for row in rows)
    best = ''
    for word in re.findall(r"[A-Za-z']{5,}", source):
        token = word.lower()
        if token in _MOMENT_SKIP:
            continue
        if len(token) > len(best):
            best = token
    return best


def suggested_next_rep(rows: list[dict], stats: dict, moment: str = '') -> str:
    """The one next step that fits this person's numbers. A low share does not get a stop-early script."""
    del moment
    from lib.story_render import session_action

    return session_action(stats, rows)


def speech_headline(rows: list[dict]) -> str:
    """What they actually did. A long monologue is not 'a short turn'."""
    count = len(rows)
    if count <= 1:
        opening = 'You spoke once'
    elif count == 2:
        opening = 'You spoke twice'
    else:
        opening = f'You spoke {count} times'
    seconds = 0.0
    for row in rows:
        seconds += max(0.0, float(row.get('end') or 0) - float(row.get('start') or 0))
    if seconds >= 1000:
        seconds /= 1000.0
    elif seconds <= 0:
        seconds = word_total(rows) / 130 * 60
    if seconds < 50:
        span = 'under a minute'
    elif seconds < 90:
        span = 'about a minute'
    else:
        minutes = max(2, int(round(seconds / 60)))
        span = 'about two minutes' if minutes == 2 else f'about {minutes} minutes'
    return f'{opening}, for {span}.'


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
    span = session_talk_phrase(transcript)
    own_words = word_total(speaker_rows(transcript, label))
    lines = [
        talk_number_line(share, speakers, span),
        question_number_line(questions, average),
        turns_sentence(mine, median if median is not None else mine, own_words if mine == 1 else None),
    ]
    allowed = {
        str(share), str(room_average(speakers)), str(mine), str(median if median is not None else mine),
        str(questions), str(average),
    }
    return lines, allowed


def personal_next_rep_enabled() -> bool:
    """Off for this Full Sail run. Set PERSONAL_NEXT_REP=on to bring the section back."""
    return os.getenv('PERSONAL_NEXT_REP', 'off').strip().lower() in {'1', 'on', 'true', 'yes'}


def _prompt(name: str, lines: list[str], quotes: list[str], failures: list[str], phrase: str, next_rep: str, units: list[dict] | None = None) -> str:
    quote_block = '\n'.join(f'Q{index + 1}: {quote}' for index, quote in enumerate(quotes)) or 'None.'
    retry = ''
    if failures:
        retry = 'The last draft failed:\n' + '\n'.join(f'- {item}' for item in failures) + '\n'
    del name, lines, phrase, next_rep
    return (
        f'{PROMPT_MARK}\n'
        'Write in second person. Do not write the person\'s name. Say you. Do not give advice.\n'
        'Do not write a headline, a count, or a next step.\n'
        'character: one sentence on what you seem to care about. No digits and no counts. '
        'Ground it in a quote. Put that quote id in character_quote, such as "Q1". Do not retype the quote.\n'
        'Good: You seem to care a lot about doing work you love.\n'
        'Bad: You said "life" 5 times.\n'
        'goal: one sentence that starts exactly "It seemed like you were trying to ".\n'
        'challenge: one thing that happened. No advice, no question, no "what if", no "you should". '
        'If you use a quote, put its id in challenge_quote. Do not retype it.\n'
        'why_it_matters: one specific consequence of this pattern, using a word from their quote. '
        'Do not say below, above, next page, or step below. No generic praise.\n'
        'yes_but: one observed caveat. No advice and no next step.\n'
        'said_quote: a quote id such as "Q2", or "" if there is no Q2.\n'
        f'{retry}'
        f'QUOTES\n{quote_block}\n'
        f'STYLE UNITS\n{_style_block(units)}\n'
        'Return JSON only with keys character, character_quote, goal, challenge, challenge_quote, '
        'why_it_matters, yes_but, said_quote.\n'
    )


def _style_block(units: list[dict] | None) -> str:
    if not units:
        return 'None.'
    return '\n'.join(f'{item["id"]}: "{item["text"]}" ({item["count"]})' for item in units)


def _quote_by_id(token: str, quotes: list[str], default_index: int) -> str:
    """Q2 means the second candidate. A retyped sentence is not used."""
    raw = str(token or '').strip().strip('"').strip('“”')
    match = re.fullmatch(r'Q([1-3])', raw, re.I)
    if match:
        index = int(match.group(1)) - 1
        return quotes[index] if 0 <= index < len(quotes) else ''
    if raw in quotes:
        return raw
    if 0 <= default_index < len(quotes):
        return quotes[default_index]
    return ''


def _strip_typed_quotes(text: str) -> str:
    cleaned = re.sub(r'(?:"[^"]*"|“[^”]*”)', ' ', text or '')
    return re.sub(r'\s+', ' ', cleaned).strip()


def _place_quote(text: str, quote: str) -> str:
    """Pattern, then the verbatim quote, then the question. The model does not type it."""
    sentences = [part.strip() for part in re.split(r'(?<=[.!?])\s+', _strip_typed_quotes(text)) if part.strip()]
    sentences = [re.sub(r'\s+you said$', '', part, flags=re.I).strip() for part in sentences]
    sentences = [part for part in sentences if part]
    spoken = f'You said "{quote}"'
    if not sentences:
        return spoken
    if len(sentences) == 1:
        return f'{sentences[0]} {spoken}'
    return f'{sentences[0]} {spoken} ' + ' '.join(sentences[1:])


def advice_problems(text: str) -> list[str]:
    """Advice, a question, or a suggestion. Quoted speech is not judged."""
    plain = _outside_quotes(text)
    reasons = []
    if '?' in plain:
        reasons.append('a question is advice here')
    if re.search(r'\b(what if|next time|you should|you could|you might|ought|try to|try this|will you|who will)\b', plain, re.I):
        reasons.append('this tells the person what to do')
    if re.search(r'(?:^|[.!?]\s+)(ask|stop|name|come|put|give|let|invite|try|hand|pause)\b', plain, re.I):
        reasons.append('an imperative is advice here')
    return reasons


def apply_quote_ids(payload: dict, quotes: list[str]) -> dict:
    """Replace quote ids with the candidate text. Drop anything the model retyped."""
    data = dict(payload or {})
    said_default = 1 if len(quotes) > 1 else 0
    said = _quote_by_id(str(data.get('said_quote') or ''), quotes, said_default if quotes else -1)
    challenge_quote = _quote_by_id(str(data.get('challenge_quote') or ''), quotes, -1)
    data['character'] = _strip_typed_quotes(str(data.get('character') or ''))
    data['goal'] = _strip_typed_quotes(str(data.get('goal') or ''))
    challenge = _strip_typed_quotes(str(data.get('challenge') or ''))
    if challenge_quote:
        challenge = f'{challenge} You said "{challenge_quote}"'.strip()
    data['challenge'] = challenge
    data['said_quote'] = said
    for key in ('why_it_matters', 'yes_but'):
        data[key] = _strip_typed_quotes(str(data.get(key) or ''))
    if data.get('one_big_thing'):
        big = _quote_by_id(str(data.get('one_big_thing_quote') or ''), quotes, 0)
        data['one_big_thing'] = _place_quote(str(data.get('one_big_thing') or ''), big) if big else _strip_typed_quotes(str(data.get('one_big_thing') or ''))
    return data


def _outside_quotes(text: str) -> str:
    """First person and number checks ignore a quoted script, straight or curly."""
    return re.sub(r'(?:"[^"]*"|“[^”]*”)', ' ', text or '')


def _numbers_outside(sentence: str) -> list[str]:
    outside = _outside_quotes(sentence)
    outside = re.sub(r'\b\d+:\d{2}(?::\d{2})?\b', ' ', outside)
    return re.findall(r'\d+', outside)


_FIRST_PERSON = re.compile(r"\b(I|I'll|I'm|I've|I'd|I’ll|I’m|I’ve|I’d)\b")
_IMPERATIVE = re.compile(r'\b(stop|ask|name|say|pause|invite|hand|add|let|leave|pass)\b', re.I)


def _quote_is_verbatim(span: str, quotes: list[str]) -> bool:
    """One contiguous sentence that appears inside a candidate. A splice is not."""
    cleaned = (span or '').strip()
    if not cleaned or len(_sentences(cleaned)) != 1:
        return False
    return any(cleaned == quote or cleaned in quote for quote in quotes)


def _content_words(text: str) -> list[str]:
    skip = {'about', 'their', 'there', 'which', 'would', 'could', 'should', 'while', 'still', 'being'}
    return [word.lower() for word in re.findall(r"[A-Za-z']{5,}", text or '') if word.lower() not in skip]


def _for_grade(text: str) -> str:
    """The mandated header is not part of the reading-grade check."""
    body = re.sub(r"^[^.\n]{0,80}, here's your report\s*", '', str(text or ''), count=1)
    return re.sub(r"^Here's your report\s*", '', body, count=1)


def section_problems(payload: dict, quotes: list[str], allowed: set[str], moment: str = '', share: int | None = None) -> dict[str, list[str]]:
    """Problems by section. Number lines are computed in code and are not judged here."""
    if isinstance(moment, int) and share is None:
        share = moment
        moment = ''
    found: dict[str, list[str]] = {}
    if not isinstance(payload, dict):
        return {'one_big_thing': ['response was not an object']}

    def add(name: str, reason: str) -> None:
        found.setdefault(name, []).append(reason)

    fields = {
        'character': str(payload.get('character') or ''),
        'goal': str(payload.get('goal') or ''),
        'challenge': str(payload.get('challenge') or ''),
        'why_it_matters': str(payload.get('why_it_matters') or ''),
        'yes_but': str(payload.get('yes_but') or ''),
    }
    if personal_next_rep_enabled():
        fields['next_rep'] = str(payload.get('next_rep') or '')
    character = fields['character']
    if not character.strip():
        add('character', 'character is missing')
    elif re.search(r'\d|%|\b(times|words|turns|percent)\b', _outside_quotes(character), re.I):
        add('character', 'character has a count')
    token = str(payload.get('character_quote') or '').strip()
    if quotes and not re.fullmatch(r'Q[1-3]', token, re.I):
        add('character', 'character must point at a quote id')
    elif quotes and re.fullmatch(r'Q[1-3]', token, re.I) and int(token[1]) > len(quotes):
        add('character', 'character quote id is out of range')
    if not fields['goal'].startswith('It seemed like you were trying to '):
        add('goal', 'goal must start with the set phrase')
    for reason in advice_problems(fields['challenge']):
        add('challenge', reason)
    for reason in advice_problems(fields['yes_but']):
        add('yes_but', reason)
    why = fields['why_it_matters'].lower()
    if any(phrase in why for phrase in ('your voice brings', 'good ideas', 'brings good ideas', 'to the team', 'your aim')):
        add('why_it_matters', 'why_it_matters is generic praise')
    if re.search(r'\b(below|above|next page|step below)\b', why):
        add('why_it_matters', 'why_it_matters points at the page')
    heard = _content_words(str(payload.get('said_quote') or ''))
    if heard and not any(word in why for word in heard):
        add('why_it_matters', 'why_it_matters does not use this person\'s words')
    if share is not None and re.search(r'\b(most of|almost all)\b', fields['yes_but'], re.I) and int(share) <= 50:
        add('yes_but', 'yes_but says most of the talk, but the share is not a majority')
    for name, text in fields.items():
        plain = _outside_quotes(text)
        limit = 5 if name == 'next_rep' else 6
        if reading_grade(_for_grade(text)) > limit:
            add(name, f'{name} is above a sixth-grade reading level')
        lowered = plain.lower()
        for phrase in _BANNED:
            if phrase in lowered:
                add(name, f'{name} uses {phrase}')
        if _FIRST_PERSON.search(plain):
            add(name, f'{name} has an unquoted first-person line')
        # One number per sentence. The number lines are not part of this text.
        for sentence in re.split(r'(?<=[.!?])\s+', text.strip()):
            if len(_numbers_outside(sentence)) > 1:
                add(name, f'{name} has two numbers in one sentence')
        if re.search(r'\b[QAU]\d+\b', text):
            add(name, f'{name} contains a control id')
        if name == 'next_rep':
            continue
        for span in re.findall(r'(?:"([^"]*)"|“([^”]*)”)', text):
            quoted = span[0] or span[1]
            if not _quote_is_verbatim(quoted, quotes):
                add(name, f'{name} quotes text that is not one verbatim sentence')
    if personal_next_rep_enabled():
        next_rep = fields.get('next_rep') or ''
        if next_rep and (
            re.search(r'\bask about \w+\b', next_rep, re.I)
            or (not _IMPERATIVE.search(next_rep) and 'next time' not in next_rep.lower())
            or len(_words(next_rep)) < 8
        ):
            add('next_rep', 'next_rep must be a full script line')
        next_numbers = _numbers_outside(next_rep)
        if any(number not in allowed for number in next_numbers):
            add('next_rep', 'next_rep uses a number that was not computed')
    said = str(payload.get('said_quote') or '').strip().strip('"').strip('“”')
    expected = quotes[1] if len(quotes) > 1 else ''
    if said != expected:
        add('said_quote', 'said_quote must be the second candidate')
    return found


def validate_narrative(payload: dict, quotes: list[str], allowed: set[str], moment: str = '') -> list[str]:
    found = section_problems(payload, quotes, allowed, moment)
    return [reason for reasons in found.values() for reason in reasons]


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
    step = suggested_next_rep(rows, stats, moment_word(quotes, rows))
    return {
        'short_mode': True,
        'headline': speech_headline(rows),
        'one_big_thing': '',
        'character': character_template(),
        'why_it_matters': why_template(rows, stats, quote),
        'yes_but': 'The turn was short.',
        'next_rep': step if personal_next_rep_enabled() else '',
        'numbers': lines,
        'said_quote': quote,
        'big_quote': quote,
        'drama': quote,
        'goal': goal_template(),
        'challenge': challenge_template([quote] if quote else []),
        'one_thing': step,
        'why': '',
        'turns': count,
        'fallback_sections': [],
        'name_seed': name,
    }


def full_story(payload: dict, lines: list[str], name: str) -> dict:
    character = str(payload.get('character') or '').strip()
    goal = str(payload.get('goal') or '').strip()
    challenge = str(payload.get('challenge') or '').strip()
    why = str(payload.get('why_it_matters') or '').strip()
    yes_but = str(payload.get('yes_but') or '').strip()
    next_rep = str(payload.get('next_rep') or '').strip() if personal_next_rep_enabled() else ''
    return {
        'short_mode': False,
        'headline': str(payload.get('headline') or '').strip(),
        'character': character,
        'one_big_thing': '',
        'why_it_matters': why,
        'yes_but': yes_but,
        'next_rep': next_rep,
        'numbers': lines,
        'said_quote': str(payload.get('said_quote') or '').strip().strip('"'),
        'big_quote': '',
        'drama': character,
        'goal': goal,
        'challenge': challenge,
        'one_thing': next_rep,
        'why': '',
        'fallback_sections': [],
        'name_seed': name,
    }


def _fallback_line(section: str, rows, lines, quotes, stats, moment: str) -> str:
    del lines
    if section == 'headline':
        return speech_headline(rows)
    if section == 'character':
        return character_template()
    if section == 'goal':
        return goal_template()
    if section == 'challenge':
        return challenge_template(quotes)
    if section == 'why_it_matters':
        return why_template(rows, stats, quotes[0] if quotes else '')
    if section == 'yes_but':
        return 'The turn still left one thing unsaid.'
    if section == 'next_rep' and personal_next_rep_enabled():
        return suggested_next_rep(rows, stats, moment)
    if section == 'said_quote':
        return quotes[1] if len(quotes) > 1 else (quotes[0] if quotes else '')
    return ''


def _record_attempt(progress_dir, label: str, raw: str, reasons: list[str], extra: dict | None = None) -> None:
    """Raw personal responses stay in the local checkpoint file, mode 600."""
    if not progress_dir:
        return
    from pathlib import Path

    directory = Path(progress_dir)
    directory.mkdir(parents=True, exist_ok=True)
    os.chmod(directory, 0o700)
    path = directory / 'checkpoints.json'
    existing = []
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding='utf-8'))
        except json.JSONDecodeError:
            existing = []
    if not isinstance(existing, list):
        existing = []
    entry = {'section': f'personal:{label}', 'raw': raw, 'reasons': reasons}
    if extra:
        entry.update(extra)
    existing.append(entry)
    path.write_text(json.dumps(existing, indent=2), encoding='utf-8')
    os.chmod(path, 0o600)


THINKING_BUDGET = 256
# Gemini 3.x ignores thinkingBudget. The live call uses thinkingLevel instead.
# Flash-lite at 'low' spent 119–192 thought tokens. Charge that, not 256.
THINKING_LEVEL = 'low'
THOUGHT_TOKENS = 192
# REST shape. The proto client has no thinkingLevel field and drops it.
THINKING_REQUEST = {
    'generationConfig': {
        'thinkingConfig': {'thinkingLevel': THINKING_LEVEL},
    },
}
INPUT_PER_MILLION = 0.30
OUTPUT_PER_MILLION = 3.50


def personal_model_name() -> str:
    """The first personal call. Gemini 3.6 does not honor a thinking-token cap."""
    return os.getenv('GEMINI_PERSONAL_MODEL', 'gemini-3.1-flash-lite')


def upgrade_model_name() -> str:
    """Used once, only after the flash-lite draft fails a validator."""
    return os.getenv('GEMINI_UPGRADE_MODEL', 'gemini-3.6-flash')


def personal_request(prompt: str, level: str = THINKING_LEVEL) -> dict:
    """The JSON body generateContent accepts. thinkingLevel, not thinkingBudget."""
    return {
        'contents': [{'role': 'user', 'parts': [{'text': prompt}]}],
        'generationConfig': {
            'temperature': 0,
            'thinkingConfig': {'thinkingLevel': level},
        },
    }


def personal_client_settings(llm) -> dict:
    """Flash-lite first. Thinking is a REST thinkingLevel, not a token budget."""
    return {
        'model': personal_model_name(),
        'temperature': 0,
        'timeout': getattr(llm, 'timeout', 120),
        'max_retries': 2,
        'thinking_level': THINKING_LEVEL,
    }


def estimate_personal_cost(
    people: int,
    *,
    attempts: int = 1,
    prompt_tokens: int = 900,
    answer_tokens: int = 180,
    thinking_budget: int = THOUGHT_TOKENS,
) -> float:
    """Each personal call, including the 3.6 upgrade, is priced at thinkingLevel low."""
    people = max(0, int(people))
    one = (
        prompt_tokens * INPUT_PER_MILLION
        + (answer_tokens + thinking_budget) * OUTPUT_PER_MILLION
    ) / 1_000_000
    return people * max(1, int(attempts)) * one


def _model_not_found(exc: BaseException) -> bool:
    """A 404 from the personal model. Other failures stay failures."""
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if getattr(current, 'status_code', None) == 404 or getattr(current, 'code', None) == 404:
            return True
        if type(current).__name__.lower() == 'notfound':
            return True
        text = str(current).lower()
        if '404' in text and 'not found' in text:
            return True
        current = current.__cause__ or current.__context__
    return False


def for_personal_narrative(llm):
    """The first personal call stays on the client the app built. A test double is unchanged."""
    return llm


def _api_key(llm) -> str:
    """A key on the client, or on a wrapper's inner client. Not the process environment."""
    seen: set[int] = set()
    current = llm
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        for attr in ('google_api_key', 'api_key'):
            value = getattr(current, attr, None)
            if hasattr(value, 'get_secret_value'):
                try:
                    value = value.get_secret_value()
                except Exception:
                    value = None
            if isinstance(value, str) and value:
                return value
        inner = getattr(current, 'llm', None)
        if inner is None or inner is current:
            break
        current = inner
    return ''


def _use_model(llm, model: str) -> None:
    """The next invoke on this client, and a wrapper that reads .model, uses this id."""
    try:
        llm.model = model
    except Exception:
        logging.warning('personal model stayed on the client; wanted %s', model)


class _Reply:
    def __init__(self, content: str, thoughts: int | None = None):
        self.content = content
        self.usage_metadata = {'thoughts_token_count': thoughts}


def _rest_generate(model: str, key: str, prompt: str, level: str = THINKING_LEVEL):
    """POST generateContent. Retries once with thinkingLevel minimal when low is refused."""
    import urllib.error
    import urllib.request

    def _post(thinking: str):
        body = personal_request(prompt, thinking)
        url = f'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}'
        request = urllib.request.Request(
            url,
            data=json.dumps(body).encode(),
            headers={'Content-Type': 'application/json'},
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors='replace')
            error = RuntimeError(f'{exc.code} {detail}')
            error.status_code = exc.code
            error.body = detail
            raise error from exc

    try:
        data = _post(level)
    except RuntimeError as exc:
        detail = str(getattr(exc, 'body', '') or exc).lower()
        if level != 'minimal' and 'thinkinglevel' in detail:
            logging.warning('thinkingLevel %s was refused; retrying minimal', level)
            data = _post('minimal')
        else:
            raise
    parts = (((data.get('candidates') or [{}])[0].get('content') or {}).get('parts') or [])
    text = ''.join(str(part.get('text') or '') for part in parts if isinstance(part, dict))
    usage = data.get('usageMetadata') or {}
    thoughts = usage.get('thoughtsTokenCount')
    logging.warning('personal call model=%s thinkingLevel=%s thought tokens=%s', model, level, thoughts)
    return _Reply(text, thoughts)


def _invoke_personal(llm, prompt, model: str | None = None):
    """REST thinkingLevel low when this client has a key. Otherwise pass that config to invoke."""
    model = model or getattr(llm, 'model', None) or personal_model_name()
    _use_model(llm, model)
    key = _api_key(llm)
    if key:
        return _rest_generate(model, key, prompt)
    config = THINKING_REQUEST['generationConfig']
    try:
        raw = llm.invoke(prompt, generationConfig=config)
    except TypeError:
        raw = llm.invoke(prompt)
    meta = getattr(raw, 'usage_metadata', None) or {}
    thoughts = meta.get('thoughts_token_count') if isinstance(meta, dict) else getattr(meta, 'thoughts_token_count', None)
    logging.warning('personal call model=%s thinkingLevel=%s thought tokens=%s', model, THINKING_LEVEL, thoughts)
    return raw


def _simplify(llm, text: str) -> str:
    prompt = (
        'Simplify this to about a fifth-grade reading level. Short words. One or two short sentences. '
        'Keep the meaning. Do not add Q1, A1, or any id. Return only the new text.\n'
        f'{text}'
    )
    try:
        return _raw_text(_invoke_personal(llm, prompt)).strip()
    except Exception as exc:
        logging.warning('simplify call failed: %s', exc)
        return text


def character_template() -> str:
    return 'You seem to care about the work in front of you.'


def goal_template() -> str:
    return 'It seemed like you were trying to make one point.'


def challenge_template(quotes: list[str]) -> str:
    spoken = str(quotes[0] if quotes else '').strip().strip('"').strip('“”')
    if spoken:
        return f'You said "{spoken}"'
    return 'You held one turn and then stopped.'


def why_template(rows: list[dict], stats: dict, quote: str = '') -> str:
    """A grade-5 line that does not point at the page."""
    del rows, stats
    words = _content_words(quote)
    if words:
        return f'The room can still hear you on {words[0]}.'
    return 'The room can still hear that moment.'


def _templated_story(name, rows, lines, quotes, stats, step, units, best_payload, best_found) -> dict:
    """Full report from the observation templates. No next step."""
    payload = dict(best_payload or {})
    payload['character'] = character_template()
    payload['character_quote'] = 'Q1' if quotes else ''
    payload['goal'] = goal_template()
    payload['challenge'] = challenge_template(quotes)
    payload['one_big_thing'] = ''
    why = _strip_ids(str(payload.get('why_it_matters') or ''))
    if not why or advice_problems(why) or re.search(r'\b(below|above|next page|step below)\b', why, re.I):
        why = why_template(rows, stats, quotes[0] if quotes else '')
    payload['why_it_matters'] = why
    yes_but = '' if (best_found or {}).get('yes_but') else str(payload.get('yes_but') or '')
    payload['yes_but'] = '' if advice_problems(yes_but) else yes_but
    payload['said_quote'] = quotes[1] if len(quotes) > 1 else (quotes[0] if quotes else '')
    story = _finish(full_story(payload, lines, name), step, units, str(payload.get('style_id') or ''))
    story['short_mode'] = False
    story['fallback_sections'] = ['character']
    story['rejection_reasons'] = {'character': '; '.join((best_found or {}).get('character') or [])}
    return story


def build_personal_story(llm, transcript, label: str, info: dict, stats: dict, progress_dir=None) -> dict:
    """Flash-lite first. One 3.6 upgrade if validators fail. SHORT is only under 60 words."""
    original = llm
    llm = for_personal_narrative(original)
    active = personal_model_name() if llm is original else (getattr(llm, 'model', None) or personal_model_name())
    _use_model(llm, active)
    upgraded = False
    from lib.style_units import style_candidates

    rows = speaker_rows(transcript, label)
    quotes = candidate_quotes([str(row.get('text') or '') for row in rows])
    lines, allowed = computed_numbers(transcript, label, stats)
    name = info.get('name') or label
    moment = moment_word(quotes, rows)
    step = suggested_next_rep(rows, stats, moment)
    names = [
        str(row.get('name') or '')
        for row in (transcript or {}).get('utterances') or []
        if row.get('name')
    ]
    units = style_candidates(transcript, label, names)
    style_note = {'style_candidates': units}
    if needs_short_report(rows):
        story = short_story(name, rows, lines, quotes, stats)
        _attach_style(story, units, units[0]['id'] if units else '')
        return _stamp(story, progress_dir)
    failures: list[str] = []
    best_payload = None
    best_found: dict[str, list[str]] = {'one_big_thing': ['response was not JSON']}
    attempts = 0
    simplified = False
    while attempts < 2:
        raw_text = ''
        try:
            raw_text = _raw_text(_invoke_personal(llm, _prompt(name, lines, quotes, failures, '', step, units), active))
            payload = _parse(raw_text)
            if payload is not None:
                payload = apply_quote_ids(payload, quotes)
                payload = _scrub_own_name(payload, name)
        except Exception as exc:
            if llm is not original and _model_not_found(exc):
                logging.warning(
                    'personal model %s returned 404; falling back to %s',
                    active,
                    getattr(original, 'model', None) or 'gemini-3.1-flash-lite',
                )
                llm = original
                active = personal_model_name()
                _use_model(llm, active)
                continue
            logging.warning('personal narrative call failed: %s', exc)
            payload = None
            raw_text = raw_text or str(exc)
        attempts += 1
        if payload is None:
            found = {'one_big_thing': ['response was not JSON']}
        else:
            found = section_problems(payload, quotes, allowed, moment, int(stats.get('share') or 0))
        reasons = [reason for items in found.values() for reason in items]
        pick = str((payload or {}).get('style_id') or '')
        _record_attempt(progress_dir, label, raw_text, reasons, {**style_note, 'style_pick': pick, 'model': active})
        if payload is not None and _grade_failures(found) and not simplified:
            payload = _rewrite_grade(llm, payload, found)
            simplified = True
            found = section_problems(payload, quotes, allowed, moment, int(stats.get('share') or 0))
            reasons = [reason for items in found.values() for reason in items]
        if payload is not None and reasons and not upgraded:
            active = upgrade_model_name()
            _use_model(llm, active)
            logging.warning('personal narrative retry uses %s', active)
            upgraded = True
            failures = reasons
            continue
        if payload is not None and not reasons:
            story = _finish(full_story(payload, lines, name), step, units, pick)
            story['fallback_sections'] = []
            return _stamp(story, progress_dir)
        if payload is not None and (best_payload is None or len(found) < len(best_found)):
            best_payload = payload
            best_found = found
        failures = reasons
    if best_payload is None or best_found.get('character') or best_found.get('goal'):
        logging.warning('personal narrative used the code template: %s', failures)
        return _stamp(_templated_story(name, rows, lines, quotes, stats, step, units, best_payload, best_found), progress_dir)
    patched = dict(best_payload)
    replaced = []
    for section, reasons in best_found.items():
        if section in {'one_big_thing', 'character', 'goal'} or not reasons:
            continue
        patched[section] = _fallback_line(section, rows, lines, quotes, stats, moment)
        replaced.append(section)
    story = full_story(patched, lines, name)
    story['fallback_sections'] = replaced
    story['rejection_reasons'] = {section: '; '.join(best_found[section]) for section in replaced}
    story['next_rep'] = step if personal_next_rep_enabled() else ''
    story['one_thing'] = story['next_rep']
    story['short_mode'] = False
    _attach_style(story, units, str((best_payload or {}).get('style_id') or ''))
    return _stamp(story, progress_dir)


def _grade_failures(found: dict) -> bool:
    return any('reading level' in reason or 'grade' in reason for reasons in found.values() for reason in reasons)


def _rewrite_grade(llm, payload: dict, found: dict) -> dict:
    """One flash-lite rewrite of a section that failed the reading grade."""
    data = dict(payload)
    for section, reasons in found.items():
        if any('reading level' in reason or 'grade' in reason for reason in reasons):
            data[section] = _simplify(llm, str(data.get(section) or ''))
    return data


def _strip_ids(text: str) -> str:
    cleaned = re.sub(r'\b[QAU]\d+\b', '', str(text or ''))
    return re.sub(r'\s{2,}', ' ', cleaned).strip()


def _finish(story: dict, step: str, units: list[dict], pick: str) -> dict:
    story['next_rep'] = step if personal_next_rep_enabled() else ''
    story['one_thing'] = story['next_rep']
    for key in ('character', 'goal', 'challenge', 'why_it_matters', 'yes_but', 'next_rep'):
        story[key] = _strip_ids(story.get(key) or '')
    _attach_style(story, units, pick)
    return story


def _stamp(story: dict, progress_dir) -> dict:
    if progress_dir:
        story['_progress_dir'] = str(progress_dir)
    return story


def log_scorecard(progress_dir, label: str, metrics: list[dict]) -> None:
    """Which metric led, and its surprise, for this person."""
    if not progress_dir or not metrics:
        return
    lead = metrics[0]
    _record_attempt(progress_dir, label, '', [], {
        'scorecard': True,
        'lead_metric': lead['key'],
        'lead_label': lead['label'],
        'surprise': lead['surprise'],
        'metrics': [{'key': item['key'], 'surprise': item['surprise']} for item in metrics],
    })


def _attach_style(story: dict, units: list[dict], picked: str) -> None:
    from lib.style_units import chosen_style

    story['opening'] = ''
    story['style_unit'] = (chosen_style(units, picked) or {}).get('text') or ''


def _scrub_own_name(payload: dict, name: str) -> dict:
    """The report says you. The model does not rename the person."""
    if not name or len(name) < 2:
        return payload
    data = dict(payload)
    pattern = re.compile(rf'\b{re.escape(name)}\b', re.I)
    for key in ('character', 'goal', 'challenge', 'one_big_thing', 'why_it_matters', 'yes_but'):
        data[key] = pattern.sub('you', str(data.get(key) or ''))
    return data


def _raw_text(raw) -> str:
    content = raw.content if hasattr(raw, 'content') else raw
    if isinstance(content, list):
        content = ' '.join(str(part) for part in content)
    return str(content or '')


def _repeated_phrase(rows: list[dict]) -> str:
    from lib.report_stats import display_bigram, top_bigram

    found = top_bigram(' '.join(str(row.get('text') or '') for row in rows))
    if not found:
        return ''
    phrase, count = found
    return f'{display_bigram(phrase)} ({count})'
