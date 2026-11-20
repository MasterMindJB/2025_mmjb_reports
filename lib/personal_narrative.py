"""Personal narrative: code picks the evidence, one call writes the sections."""

from __future__ import annotations

import json
import logging
import os
import re
from contextlib import contextmanager

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
        'bullets: 3 items when there are 3 quotes, otherwise 2. Each item has behavior, quote, and why.\n'
        'behavior is one observed sentence. No advice, no question, no counts, no digits.\n'
        'The first behavior merges care and goal. It starts with "You seem to care" and includes "it seemed like you were trying to".\n'
        'No later behavior contains "You seem" or starts with "It seemed like you were trying to".\n'
        'Never write a behavior that is only "You said."\n'
        'quote is only an id such as "Q1". Each bullet uses a different id. Do not retype the words.\n'
        'why is one short sentence about why this quote matters for this person or the listener. '
        'Do not restate the quote. Do not say below, above, next page, or step below. '
        'No advice, no next time, and no should.\n'
        'Good why: That story shows people you know how long the climb takes.\n'
        'Bad why: You brought up making movies.\n'
        'Good behavior: You seem to care about the work, and it seemed like you were trying to name it.\n'
        'Bad: You said "life" 5 times.\n'
        'yes_but: one observed caveat. No advice and no next step.\n'
        'said_quote: a quote id such as "Q2", or "" if there is no Q2.\n'
        f'{retry}'
        f'QUOTES\n{quote_block}\n'
        f'STYLE UNITS\n{_style_block(units)}\n'
        'Return JSON only with keys bullets, yes_but, said_quote.\n'
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
    """Drop a pasted sentence quote. Keep a short quoted unit such as "life"."""
    def keep_short(match: re.Match) -> str:
        words = re.sub(r'[“”"]', '', match.group(0)).split()
        return match.group(0) if len(words) <= 3 else ' '

    cleaned = re.sub(r'(?:"[^"]*"|“[^”]*”)', keep_short, text or '')
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


def _lemma(word: str) -> str:
    token = re.sub(r"'s$", '', str(word or '').lower().strip("'\""))
    for suffix in ('ingly', 'ing', 'ied', 'ies', 'es', 'ed', 's'):
        if len(token) - len(suffix) >= 4 and token.endswith(suffix):
            return token[:-len(suffix)]
    return token


def _lemmas(text: str) -> list[str]:
    skip = {
        'about', 'their', 'there', 'which', 'would', 'could', 'should', 'while', 'still',
        'being', 'people', 'shows', 'showed', 'matters', 'means', 'because', 'room',
        'helps', 'help', 'this', 'that', 'your', 'with', 'from', 'have', 'been',
    }
    found = []
    for word in re.findall(r"[A-Za-z']+", text or ''):
        if "'" in word or word.lower() in _FUNCTION or word.lower() in skip:
            continue
        lemma = _lemma(word)
        if len(lemma) < 3 or lemma in skip:
            continue
        found.append(lemma)
    return found


def _lemma_hit(left: str, pool: list[str]) -> bool:
    return any(left == item or left.startswith(item) or item.startswith(left) for item in pool if min(len(left), len(item)) >= 4)


_RESTATE_OPEN = re.compile(
    r"^you (brought up|spoke about|talked about|stated|said|mentioned|made|named)\b",
    re.I,
)
_CONSEQUENCE = re.compile(
    r"\b(people|room|listener|listeners|someone|shows|showed|matters|means|because|lets|leaves|gives|understand|hear|heard)\b",
    re.I,
)


def why_on_topic(why: str, quote: str, behavior: str = '', quotes: list[str] | None = None) -> bool:
    """Reject a why only when every content word is absent from this person's speech."""
    lemmas = _lemmas(why)
    if not lemmas:
        return False
    speech = ' '.join([quote or '', behavior or '', *(quotes or [])])
    pool = _lemmas(speech)
    if any(_lemma_hit(item, pool) for item in lemmas):
        return True
    return False


def why_restates(why: str, quote: str) -> bool:
    """A why that mostly repeats the quote and names no consequence."""
    if _CONSEQUENCE.search(why or ''):
        return False
    if _RESTATE_OPEN.match(str(why or '').strip()):
        return True
    lemmas = _lemmas(why)
    pool = _lemmas(quote)
    if not lemmas or not pool:
        return False
    overlap = sum(1 for item in lemmas if _lemma_hit(item, pool))
    return overlap / len(lemmas) >= 0.67


def _why_problem(why: str, quote: str, behavior: str, quotes: list[str] | None = None) -> str:
    """A why-line failure, or '' when the line can stand."""
    if not str(why or '').strip():
        return 'why is missing'
    if not why_on_topic(why, quote, behavior, quotes):
        return 'why drifts from this quote'
    if why_restates(why, quote):
        return 'why restates the quote'
    if reading_grade(why) > 6:
        return 'a why line is above a sixth-grade reading level'
    return ''


def rejected_why_lines(items, quotes: list[str]) -> list[dict]:
    """Each rejected why, verbatim, with the reason. This is the report debug list."""
    found = []
    rows = [item for item in (items or []) if isinstance(item, dict)]
    for item in rows:
        why = str(item.get('why') or '')
        behavior = str(item.get('behavior') or '')
        quote = _quote_by_id(str(item.get('quote') or ''), quotes, -1)
        problem = _why_problem(why, quote, behavior, quotes)
        if problem:
            found.append({'why': why.strip(), 'reason': problem, 'quote': quote})
    return found


def _bullet_problems(items, quotes: list[str]) -> list[str]:
    """Each bullet is one observation, one new quote id, and one specific why."""
    reasons = []
    rows = [item for item in (items or []) if isinstance(item, dict)]
    usable = [quote for quote in quotes if quote]
    expected = 3 if len(usable) >= 3 else min(2, len(usable))
    if expected and len(rows) < 2:
        reasons.append('bullets should be 2')
    seen = []
    for index, item in enumerate(rows):
        behavior = str(item.get('behavior') or '')
        why = str(item.get('why') or '')
        token = str(item.get('quote') or '').strip()
        if re.fullmatch(r'you said\.?', behavior.strip().lower()):
            reasons.append('You said is not a behavior')
        if index == 0 and not behavior.lower().startswith('you seem to care'):
            reasons.append('the first bullet must start with You seem to care')
        if index == 0 and 'it seemed like you were trying to' not in behavior.lower():
            reasons.append('the first bullet must merge care and goal')
        if index and 'you seem' in behavior.lower():
            reasons.append('a later bullet repeats You seem')
        if index and behavior.lower().startswith('it seemed like you were trying to'):
            reasons.append('a later bullet repeats the goal opener')
        if not re.fullmatch(r'Q[1-3]', token, re.I):
            reasons.append('each bullet needs its own quote id')
        elif token.upper() in seen:
            reasons.append('each bullet needs a different quote')
        else:
            seen.append(token.upper())
            quote = _quote_by_id(token, quotes, -1)
            problem = _why_problem(why, quote, behavior, usable)
            if problem:
                reasons.append(f'{problem}: {why.strip()}')
        for reason in advice_problems(behavior) + advice_problems(why):
            reasons.append(reason)
        if _is_generic(behavior) or _is_generic(why):
            reasons.append('a bullet is a generic line')
        if re.search(r'\b(below|above|next page|step below)\b', f'{behavior} {why}', re.I):
            reasons.append('a bullet points at the page')
        if re.search(r'\b[QAU]\d+\b', f'{behavior} {why}'):
            reasons.append('a bullet contains a control id')
    return reasons


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


def _render_bullet(behavior: str, quote: str, why: str) -> str:
    """Behavior, the code-owned because line, then why. The quote is verbatim."""
    statement = _strip_typed_quotes(behavior).strip().rstrip('.')
    spoken = str(quote or '').strip().strip('"').strip('“”')
    if spoken and spoken[-1] not in '.!?':
        spoken += '.'
    reason = _strip_typed_quotes(why).strip()
    if reason and reason[-1] not in '.!?':
        reason += '.'
    line = f'{statement}. We came to that conclusion because of "{spoken}".'
    if reason:
        line = f'{line} {reason}'
    return line


def _bullet_sources(payload: dict) -> list[dict]:
    raw = payload.get('bullets')
    if isinstance(raw, list) and raw:
        return [item for item in raw if isinstance(item, dict)]
    sources = []
    if payload.get('character'):
        sources.append({
            'behavior': payload.get('character'),
            'quote': payload.get('character_quote') or 'Q1',
            'why': payload.get('why_it_matters') or '',
        })
    if payload.get('goal'):
        sources.append({'behavior': payload.get('goal'), 'quote': 'Q2', 'why': ''})
    if payload.get('challenge'):
        sources.append({
            'behavior': payload.get('challenge'),
            'quote': payload.get('challenge_quote') or 'Q3',
            'why': '',
        })
    return sources


def render_personal_bullets(payload: dict, quotes: list[str]) -> list[str]:
    """Two or three bullets. Each quote is inserted by code and used once."""
    limit = 3 if len([quote for quote in quotes if quote]) >= 3 else 2
    rendered = []
    used: set[str] = set()
    for source in _bullet_sources(payload):
        if len(rendered) >= limit:
            break
        quote = _quote_by_id(str(source.get('quote') or ''), quotes, -1)
        if not quote or quote in used:
            continue
        behavior = str(source.get('behavior') or '')
        why = str(source.get('why') or '')
        if not behavior.strip() or not why.strip() or re.fullmatch(r'you said\.?', behavior.strip().lower()):
            continue
        used.add(quote)
        rendered.append(_render_bullet(behavior, quote, why))
    return rendered if len(rendered) >= 2 else []


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
    data['bullets_text'] = render_personal_bullets(data, quotes)
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


_GENERIC_LINES = (
    'you seem to care about the work in front of you',
    'it seemed like you were trying to make one point',
    'the room can still hear',
    'you held one turn and then stopped',
)


def _is_generic(text: str) -> bool:
    lowered = _outside_quotes(text).lower()
    return any(phrase in lowered for phrase in _GENERIC_LINES)


def _content_words(text: str) -> list[str]:
    skip = {'about', 'their', 'there', 'which', 'would', 'could', 'should', 'while', 'still', 'being'}
    return [word.lower() for word in re.findall(r"[A-Za-z']{5,}", text or '') if word.lower() not in skip]


def _for_grade(text: str) -> str:
    """The mandated header is not part of the reading-grade check."""
    body = re.sub(r"^[^.\n]{0,80}, here's your report\.?\s*", '', str(text or ''), count=1)
    return re.sub(r"^Here's your report\.?\s*", '', body, count=1)


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

    use_bullets = isinstance(payload.get('bullets'), list) and payload.get('bullets')
    fields = {
        'yes_but': str(payload.get('yes_but') or ''),
    }
    if not use_bullets:
        fields.update({
            'character': str(payload.get('character') or ''),
            'goal': str(payload.get('goal') or ''),
            'challenge': str(payload.get('challenge') or ''),
            'why_it_matters': str(payload.get('why_it_matters') or ''),
        })
    if personal_next_rep_enabled():
        fields['next_rep'] = str(payload.get('next_rep') or '')
    if use_bullets:
        for reason in _bullet_problems(payload.get('bullets'), quotes):
            add('bullets', reason)
    if not use_bullets:
        character = fields.get('character', '')
        if not character.strip():
            add('character', 'character is missing')
        elif re.search(r'\d|%|\b(times|words|turns|percent)\b', _outside_quotes(character), re.I):
            add('character', 'character has a count')
        elif _is_generic(character):
            add('character', 'character is a generic line')
        token = str(payload.get('character_quote') or '').strip()
        if quotes and not re.fullmatch(r'Q[1-3]', token, re.I):
            add('character', 'character must point at a quote id')
        elif quotes and re.fullmatch(r'Q[1-3]', token, re.I) and int(token[1]) > len(quotes):
            add('character', 'character quote id is out of range')
        if not str(fields.get('goal') or '').startswith('It seemed like you were trying to '):
            add('goal', 'goal must start with the set phrase')
        elif _is_generic(fields['goal']):
            add('goal', 'goal is a generic line')
        for reason in advice_problems(fields.get('challenge', '')):
            add('challenge', reason)
    for reason in advice_problems(fields['yes_but']):
        add('yes_but', reason)
    why = str(fields.get('why_it_matters') or '').lower()
    if any(phrase in why for phrase in ('your voice brings', 'good ideas', 'brings good ideas', 'to the team', 'your aim')) or _is_generic(why):
        add('why_it_matters', 'why_it_matters is generic praise')
    if re.search(r'\b(below|above|next page|step below)\b', why):
        add('why_it_matters', 'why_it_matters points at the page')
    heard = _content_words(str(payload.get('said_quote') or ''))
    if not use_bullets and heard and not any(word in why for word in heard):
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
        'character': '',
        'why_it_matters': '',
        'yes_but': 'The turn was short.',
        'next_rep': step if personal_next_rep_enabled() else '',
        'numbers': lines,
        'said_quote': quote,
        'big_quote': quote,
        'drama': quote,
        'goal': '',
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
        'bullets': list(payload.get('bullets_text') or []),
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
        return ''
    if section == 'goal':
        return ''
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


def _note_rejected_whys(label, payload, quotes, progress_dir, raw, reasons, extra: dict | None = None) -> list[dict]:
    """Write each rejected why, verbatim, into the checkpoint debug log."""
    items = (payload or {}).get('bullets') if isinstance(payload, dict) else []
    rejected = rejected_why_lines(items, quotes)
    for item in rejected:
        logging.warning('rejected why (%s): %s', item['reason'], item['why'])
    note = dict(extra or {})
    note['rejected_whys'] = rejected
    _record_attempt(progress_dir, label, raw, reasons, note)
    return rejected


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
# Flash-lite accepts thinkingBudget only. Gemini 3.6 accepts thinkingLevel only.
THINKING_LEVEL = 'minimal'
THOUGHT_TOKENS = 64
LITE_MAX_OUTPUT = 512
UPGRADE_MAX_OUTPUT = 2048
# A 3.6 call spent about 490 tokens thinking inside a 512 cap, so the JSON was cut off.
UPGRADE_THOUGHT_TOKENS = 512
EXPECTED_JSON_TOKENS = 400
INPUT_PER_MILLION = 0.30
OUTPUT_PER_MILLION = 3.50


def personal_model_name() -> str:
    """The first personal call. Gemini 3.6 does not honor a thinking-token cap."""
    return os.getenv('GEMINI_PERSONAL_MODEL', 'gemini-3.1-flash-lite')


def upgrade_model_name() -> str:
    """Used once, only after the flash-lite draft fails a validator."""
    return os.getenv('GEMINI_UPGRADE_MODEL', 'gemini-3.6-flash')


def upgrade_thinking_fallback(mode: str) -> str:
    """3.6 tries minimal, then low if that field is rejected. Never a second field at once."""
    return 'low' if mode == 'minimal' else ''


def thinking_config(model: str, mode: str = 'budget') -> dict:
    """Exactly one thinking field. Lite honors a budget. 3.6 honors a level."""
    del model
    if mode == 'minimal':
        return {'thinkingLevel': 'minimal'}
    if mode == 'low':
        return {'thinkingLevel': 'low'}
    return {'thinkingBudget': 0}


def upgrade_output_fits(
    max_output: int = UPGRADE_MAX_OUTPUT,
    thoughts: int = UPGRADE_THOUGHT_TOKENS,
    expected_json: int = EXPECTED_JSON_TOKENS,
) -> bool:
    """Thinking tokens count against maxOutputTokens, so the JSON needs room after them."""
    return int(max_output) >= int(thoughts) + int(expected_json)


def personal_request(prompt: str, model: str = 'gemini-3.1-flash-lite', max_output: int = LITE_MAX_OUTPUT, mode: str = 'budget') -> dict:
    """The JSON body generateContent accepts. One thinking field, never both."""
    return {
        'contents': [{'role': 'user', 'parts': [{'text': prompt}]}],
        'generationConfig': {
            'temperature': 0,
            'maxOutputTokens': int(max_output),
            'thinkingConfig': thinking_config(model, mode),
        },
    }


# Kept for callers that inspect the lite request shape.
THINKING_REQUEST = personal_request('').get('generationConfig') and {
    'generationConfig': personal_request('')['generationConfig'],
}


def personal_client_settings(llm) -> dict:
    """Flash-lite first. The lite call sends thinkingBudget and no thinkingLevel."""
    return {
        'model': personal_model_name(),
        'temperature': 0,
        'timeout': getattr(llm, 'timeout', 120),
        'max_retries': 2,
        'thinking_level': 'budget',
        'max_output_tokens': LITE_MAX_OUTPUT,
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


def _bare_model(model: str) -> str:
    return str(model or '').strip().removeprefix('models/')


def _format_like(previous, wanted: str) -> str:
    """Keep the client's own model-name shape. models/gemini-… stays prefixed."""
    bare = _bare_model(wanted)
    if isinstance(previous, str) and previous.startswith('models/'):
        return f'models/{bare}'
    return bare


@contextmanager
def _scoped_model(llm, wanted: str):
    """Set the model for one call, then put the shared client back."""
    previous = getattr(llm, 'model', None)
    formatted = _format_like(previous, wanted)
    try:
        try:
            llm.model = formatted
        except Exception:
            logging.warning('personal model stayed on the client; wanted %s', formatted)
        yield _bare_model(formatted)
    finally:
        if previous is not None:
            try:
                llm.model = previous
            except Exception:
                logging.warning('could not restore model %s', previous)


class _Reply:
    def __init__(self, content: str, thoughts: int | None = None):
        self.content = content
        self.usage_metadata = {'thoughts_token_count': thoughts}


def _rest_generate(model: str, key: str, prompt: str, mode: str = 'budget', max_output: int = LITE_MAX_OUTPUT):
    """POST generateContent. One thinking field. A 400 on that field retries with the other."""
    import urllib.error
    import urllib.request

    bare = _bare_model(model)

    def _post(thinking_mode: str):
        body = personal_request(prompt, bare, max_output, thinking_mode)
        url = f'https://generativelanguage.googleapis.com/v1beta/models/{bare}:generateContent?key={key}'
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
        data = _post(mode)
    except RuntimeError as exc:
        detail = str(getattr(exc, 'body', '') or exc).lower()
        alternate = upgrade_thinking_fallback(mode) if '3.6' in bare else ''
        if alternate and 'thinking' in detail:
            logging.warning('thinking field %s was refused; retrying with %s', mode, alternate)
            data = _post(alternate)
        else:
            raise
    parts = (((data.get('candidates') or [{}])[0].get('content') or {}).get('parts') or [])
    text = ''.join(str(part.get('text') or '') for part in parts if isinstance(part, dict))
    usage = data.get('usageMetadata') or {}
    thoughts = usage.get('thoughtsTokenCount')
    logging.warning('personal call model=%s thinking=%s thought tokens=%s', model, mode, thoughts)
    return _Reply(text, thoughts)


def _call_plan(model: str, *, simplify: bool = False) -> tuple[str, int]:
    """Lite uses a budget of 0. The 3.6 upgrade tries thinkingLevel minimal."""
    if simplify:
        return 'budget', 128
    if '3.6' in _bare_model(model):
        return 'minimal', UPGRADE_MAX_OUTPUT
    return 'budget', LITE_MAX_OUTPUT


def _invoke_personal(llm, prompt, model: str | None = None, *, simplify: bool = False):
    """One personal call. The shared client's model is restored before this returns."""
    wanted = model or personal_model_name()
    mode, max_output = _call_plan(wanted, simplify=simplify)
    with _scoped_model(llm, wanted) as bare:
        key = _api_key(llm)
        if key:
            raw = _rest_generate(bare, key, prompt, mode=mode, max_output=max_output)
        else:
            config = personal_request(prompt, bare, max_output, mode)['generationConfig']
            try:
                raw = llm.invoke(prompt, generationConfig=config)
            except TypeError:
                raw = llm.invoke(prompt)
        meta = getattr(raw, 'usage_metadata', None) or {}
        thoughts = meta.get('thoughts_token_count') if isinstance(meta, dict) else getattr(meta, 'thoughts_token_count', None)
        logging.warning('personal call model=%s thinking=%s maxOutputTokens=%s thought tokens=%s', bare, mode, max_output, thoughts)
        return raw


def _simplify_prompt(section: str, text: str, units: list[dict] | None, quotes: list[str]) -> str:
    return (
        'Rewrite this one section in short words, about a fifth-grade reading level. '
        'One sentence. No advice, no counts, no digits, no ids. '
        'Use the repeated units and a quote. '
        'character must say what the person cares about. '
        'goal must start exactly "It seemed like you were trying to ". '
        'why_it_matters must use a word from a quote and must not say below, above, or next page. '
        'Return only the new sentence.\n'
        f'SECTION {section}\n'
        f'UNITS\n{_style_block(units)}\n'
        f'QUOTES\n' + ('\n'.join(quotes) or 'None.') + f'\nTEXT\n{text}'
    )


def _simplify(llm, text: str, section: str = 'why_it_matters', units: list[dict] | None = None, quotes: list[str] | None = None) -> str:
    """A no-thinking rewrite on flash-lite. It does not leave the shared client on another model."""
    prompt = _simplify_prompt(section, text, units, quotes or [])
    try:
        return _raw_text(_invoke_personal(llm, prompt, personal_model_name(), simplify=True)).strip()
    except Exception as exc:
        logging.warning('simplify call failed: %s', exc)
        return text


def _top_content_unit(units: list[dict] | None) -> str:
    """The strongest repeated unit that is not a hedge or a style marker."""
    from lib.style_units import _STYLE_MARKERS, display_unit

    for item in units or []:
        text = str(item.get('text') or '').strip()
        if text and text not in _STYLE_MARKERS:
            return display_unit(text)
    return ''


_VERBISH = {
    'wait', 'waits', 'sign', 'signs', 'land', 'lands', 'change', 'changes', 'changed',
    'begin', 'began', 'love', 'loved', 'keep', 'keeps', 'need', 'needs', 'make', 'makes',
    'made', 'start', 'starts', 'leave', 'leaves', 'take', 'takes', 'come', 'comes',
}


_GOAL_VERBS = {
    'name', 'make', 'find', 'keep', 'hold', 'show', 'tell', 'build', 'start',
    'finish', 'decide', 'share', 'explain', 'describe', 'choose', 'write',
    'draft', 'own', 'sign',
}


def _real_content_word(word: str) -> bool:
    """A content word. Function words and contractions are not quoted units."""
    token = str(word or '').strip().strip('"').strip("'")
    if not token or "'" in token:
        return False
    low = token.lower()
    if low in _FUNCTION:
        return False
    return len(low) > 3


def _phrases_in(quote: str) -> list[str]:
    """Noun phrases copied out of one quote. Nothing is invented."""
    tokens = re.findall(r"[A-Za-z']+", quote or '')
    phrases = []
    index = 0
    while index < len(tokens):
        low = tokens[index].lower()
        if low in {'a', 'an', 'the'} and index + 1 < len(tokens) and _real_content_word(tokens[index + 1]):
            words = [low, tokens[index + 1].lower()]
            cursor = index + 2
            if cursor < len(tokens) and _real_content_word(tokens[cursor]) and tokens[cursor].lower() not in _VERBISH:
                words.append(tokens[cursor].lower())
                cursor += 1
            phrases.append(' '.join(words))
            index = cursor
            continue
        index += 1
    return phrases


def _phrase_head(phrase: str) -> str:
    parts = [part for part in str(phrase or '').split() if part.lower() not in {'a', 'an', 'the'}]
    return parts[-1].lower() if parts else ''


def _topic_phrase(quote: str) -> str:
    """One noun phrase that actually occurs in this quote."""
    phrases = []
    seen = set()
    for phrase in _phrases_in(quote):
        head = _phrase_head(phrase)
        if not head or head in seen:
            continue
        seen.add(head)
        phrases.append(phrase)
    if phrases:
        return phrases[0]
    for word in re.findall(r"[A-Za-z']+", quote or ''):
        if _real_content_word(word):
            return word.lower()
    return ''


def _distinct_phrases(quote: str) -> list[str]:
    phrases = []
    seen = set()
    for phrase in _phrases_in(quote):
        head = _phrase_head(phrase)
        if not head or head in seen:
            continue
        seen.add(head)
        phrases.append(phrase)
    return phrases


def merged_care_goal(quote: str) -> str:
    """Care and goal from two different nouns in this one quote. '' when that is not possible."""
    text = quote if isinstance(quote, str) else ''
    if isinstance(quote, (list, tuple)):
        text = str(quote[0] if quote else '')
    phrases = _distinct_phrases(text)
    if len(phrases) < 2:
        return ''
    tokens = re.findall(r"[A-Za-z']+", text)
    if tokens and tokens[0].lower() in _GOAL_VERBS:
        goal_phrase = phrases[0]
        care = next((phrase for phrase in phrases if _phrase_head(phrase) != _phrase_head(goal_phrase)), '')
        if not care:
            return ''
        return (
            f'You seem to care about {care}, and it seemed like you were trying to '
            f'{tokens[0].lower()} {goal_phrase}'
        )
    care, other = phrases[0], phrases[1]
    if _phrase_head(care) == _phrase_head(other):
        return ''
    return f'You seem to care about {care}, and it seemed like you were trying to talk about {other}'


def _spoke_about(quote: str) -> str:
    topic = _topic_phrase(quote)
    if not topic:
        return ''
    return f'You spoke about {topic}'


def _unit_in_quote(unit: str, quote: str) -> bool:
    parts = [part for part in str(unit or '').split() if _real_content_word(part)]
    if not parts:
        return False
    hay = str(quote or '').lower()
    return all(part.lower() in hay for part in parts)


def later_return_line(unit: str) -> str:
    """A later bullet. Only a real content word is quoted."""
    shown = str(unit or '').strip().strip('"')
    parts = [part for part in shown.split() if part]
    if parts and parts[0].lower() in {'a', 'an', 'the'}:
        parts = parts[1:]
    if not parts or not all(_real_content_word(part) for part in parts):
        return ''
    return f'You kept coming back to "{" ".join(parts)}"'


def _care_behavior(unit: str) -> str:
    """A repeated unit, quoted only when it is a real content word."""
    return later_return_line(unit)


def beat_from_unit(units: list[dict] | None, kind: str) -> str:
    """Bullet 1 only. A non-noun unit is quoted instead of 'the hard'."""
    del kind
    unit = _top_content_unit(units)
    if not unit:
        return ''
    return _care_behavior(unit)


def character_template(units: list[dict] | None = None) -> str:
    return beat_from_unit(units, 'character')


def goal_template(units: list[dict] | None = None) -> str:
    return beat_from_unit(units, 'goal')


def challenge_template(quotes: list[str]) -> str:
    spoken = str(quotes[0] if quotes else '').strip().strip('"').strip('“”')
    if spoken:
        return f'You said "{spoken}"'
    return 'You held one turn and then stopped.'


def _first_bullet_behavior(quote: str) -> str:
    return merged_care_goal(quote) or _spoke_about(quote)


def _later_bullet_behavior(quote: str, units: list[dict] | None) -> str:
    for item in units or []:
        unit = str(item.get('text') or '')
        if _unit_in_quote(unit, quote):
            line = later_return_line(unit)
            if line:
                return line
    return _spoke_about(quote)


def _fallback_bullets(units: list[dict] | None, quotes: list[str]) -> list[str]:
    """Two bullets. Each noun comes from the quote on that bullet. No invented why."""
    usable = []
    for quote in quotes:
        if quote and quote not in usable:
            usable.append(quote)
    if len(usable) < 2:
        return []
    first = _first_bullet_behavior(usable[0])
    second = _later_bullet_behavior(usable[1], units)
    if not first or not second:
        return []
    if second.lower().startswith('it seemed') or second.lower().startswith('you seem'):
        second = _spoke_about(usable[1])
    if not second:
        return []
    return [
        _render_bullet(first, usable[0], ''),
        _render_bullet(second, usable[1], ''),
    ]


def why_template(rows: list[dict] | None = None, stats: dict | None = None, quote: str = '') -> str:
    """Why it matters is dropped when a rewrite cannot pass. There is no canned line."""
    del rows, stats, quote
    return ''


def _templated_story(name, rows, lines, quotes, stats, step, units, best_payload, best_found) -> dict:
    """Full report. Character and goal come from a repeated unit. Why is kept only if it still passes."""
    payload = dict(best_payload or {})
    character = str(payload.get('character') or '')
    goal = str(payload.get('goal') or '')
    if (best_found or {}).get('character') or _is_generic(character) or not character.strip():
        payload['character'] = beat_from_unit(units, 'character')
    if (best_found or {}).get('goal') or _is_generic(goal) or not goal.startswith('It seemed like you were trying to '):
        payload['goal'] = beat_from_unit(units, 'goal')
    payload['character_quote'] = 'Q1' if quotes else ''
    if (best_found or {}).get('challenge') or not str(payload.get('challenge') or '').strip():
        payload['challenge'] = challenge_template(quotes)
    payload['one_big_thing'] = ''
    why = _strip_ids(str(payload.get('why_it_matters') or ''))
    why_failed = (best_found or {}).get('why_it_matters') or not why or _is_generic(why) or advice_problems(why)
    if why_failed or reading_grade(why) > 6:
        why = ''
    payload['why_it_matters'] = why
    payload['bullets_text'] = _fallback_bullets(units, quotes)
    if len(payload['bullets_text']) < 2:
        if len([quote for quote in quotes if quote]) >= 2:
            payload['bullets_text'] = _fallback_bullets(None, quotes)
        if len(payload['bullets_text']) >= 2:
            pass
        else:
            story = short_story(name, rows, lines, quotes, stats)
            story['bullets'] = []
            story['character'] = ''
            story['goal'] = ''
            story['challenge'] = ''
            story['why_it_matters'] = ''
            story['short_mode'] = True
            story['fallback_sections'] = ['bullets']
            story['rejection_reasons'] = {'bullets': '; '.join((best_found or {}).get('bullets') or [])}
            return story
    yes_but = '' if (best_found or {}).get('yes_but') else str(payload.get('yes_but') or '')
    payload['yes_but'] = '' if advice_problems(yes_but) else yes_but
    payload['said_quote'] = quotes[1] if len(quotes) > 1 else (quotes[0] if quotes else '')
    story = _finish(full_story(payload, lines, name), step, units, str(payload.get('style_id') or ''))
    story['short_mode'] = False
    story['fallback_sections'] = ['character']
    story['rejection_reasons'] = {'character': '; '.join((best_found or {}).get('character') or [])}
    return story


def _grounded_simplify(llm, payload: dict, units: list[dict], quotes: list[str], found: dict) -> dict:
    """One no-thinking rewrite of the beats that failed, with this person's units and quotes."""
    data = dict(payload)
    for section in ('character', 'goal', 'challenge', 'why_it_matters', 'yes_but'):
        if not found.get(section):
            continue
        data[section] = _simplify(llm, str(data.get(section) or ''), section, units, quotes)
    return data


def build_personal_story(llm, transcript, label: str, info: dict, stats: dict, progress_dir=None) -> dict:
    """Flash-lite first. One 3.6 upgrade if validators fail. SHORT is only under 60 words."""
    original = llm
    saved_model = getattr(original, 'model', None)
    try:
        return _build_personal_story(llm, transcript, label, info, stats, progress_dir)
    finally:
        if saved_model is not None:
            try:
                original.model = saved_model
            except Exception:
                logging.warning('could not restore model %s', saved_model)


def _build_personal_story(llm, transcript, label: str, info: dict, stats: dict, progress_dir=None) -> dict:
    original = llm
    llm = for_personal_narrative(original)
    active = personal_model_name() if llm is original else _bare_model(getattr(llm, 'model', None) or personal_model_name())
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
        story['character'] = beat_from_unit(units, 'character')
        story['goal'] = beat_from_unit(units, 'goal')
        story['why_it_matters'] = ''
        story['bullets'] = _fallback_bullets(units, quotes)
        _attach_style(story, units, units[0]['id'] if units else '')
        return _stamp(story, progress_dir)
    failures: list[str] = []
    best_payload = None
    best_found: dict[str, list[str]] = {'one_big_thing': ['response was not JSON']}
    attempts = 0
    simplified = False
    repaired = False
    while attempts < 2:
        raw_text = ''
        try:
            raw_text = _raw_text(_invoke_personal(
                llm, _prompt(name, lines, quotes, failures, '', step, units), active,
            ))
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
        _note_rejected_whys(label, payload, quotes, progress_dir, raw_text, reasons, {**style_note, 'style_pick': pick, 'model': active})
        if payload is not None and not simplified and any(found.get(section) for section in ('character', 'goal', 'why_it_matters', 'challenge', 'yes_but')):
            payload = _grounded_simplify(llm, payload, units, quotes, found)
            payload = apply_quote_ids(payload, quotes)
            simplified = True
            found = section_problems(payload, quotes, allowed, moment, int(stats.get('share') or 0))
            reasons = [reason for items in found.values() for reason in items]
        if payload is not None and reasons and not repaired and not upgraded and _why_only(found):
            logging.warning('personal why repair for %s: %s', label, '; '.join(reasons))
            payload = _repair_whys(llm, payload, quotes)
            payload = apply_quote_ids(payload, quotes)
            repaired = True
            found = section_problems(payload, quotes, allowed, moment, int(stats.get('share') or 0))
            reasons = [reason for items in found.values() for reason in items]
            _note_rejected_whys(label, payload, quotes, progress_dir, json.dumps(payload), reasons, {**style_note, 'model': personal_model_name(), 'repair': True})
        if payload is not None and reasons and repaired and not upgraded and _why_only(found):
            logging.warning('personal why repair uses %s for %s', upgrade_model_name(), label)
            payload = _repair_whys(llm, payload, quotes, upgrade_model_name())
            payload = apply_quote_ids(payload, quotes)
            upgraded = True
            found = section_problems(payload, quotes, allowed, moment, int(stats.get('share') or 0))
            reasons = [reason for items in found.values() for reason in items]
            _note_rejected_whys(label, payload, quotes, progress_dir, json.dumps(payload), reasons, {**style_note, 'model': upgrade_model_name(), 'repair': True})
        if reasons and not upgraded:
            logging.warning('personal flash-lite rejected %s: %s', label, '; '.join(reasons))
        if payload is not None and not reasons:
            story = _finish(full_story(payload, lines, name), step, units, pick)
            story['fallback_sections'] = []
            return _stamp(story, progress_dir)
        if payload is not None and reasons and not upgraded:
            active = upgrade_model_name()
            logging.warning('personal narrative retry uses %s', active)
            upgraded = True
            failures = reasons
            continue
        if payload is not None and (best_payload is None or len(found) < len(best_found)):
            best_payload = payload
            best_found = found
        failures = reasons
    salvaged = _usable_model_bullets(best_payload, quotes)
    if len(salvaged) >= 2 and best_payload is not None:
        logging.warning('personal narrative kept the model bullets and dropped bad why lines: %s', failures)
        payload = dict(best_payload)
        payload['bullets_text'] = salvaged
        story = _finish(full_story(payload, lines, name), step, units, str(payload.get('style_id') or ''))
        story['short_mode'] = False
        story['fallback_sections'] = ['why']
        story['rejected_whys'] = rejected_why_lines(payload.get('bullets'), quotes)
        story['rejection_reasons'] = {
            'why': '; '.join(f"{item['reason']}: {item['why']}" for item in story['rejected_whys']),
        }
        return _stamp(story, progress_dir)
    if best_payload is None or best_found.get('character') or best_found.get('goal') or best_found.get('bullets'):
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


def _why_only(found: dict) -> bool:
    """True when every failure is a why line (drift or reading grade)."""
    reasons = [reason for items in found.values() for reason in items]
    if not reasons:
        return False
    return all(
        reason.startswith('why ') or 'reading level' in reason or reason.startswith('a why line')
        for reason in reasons
    )


def _fk_only(found: dict) -> bool:
    reasons = [reason for items in found.values() for reason in items]
    return bool(reasons) and all('reading level' in reason or 'grade' in reason for reason in reasons)


def _repair_sentence(llm, text: str, quote: str, behavior: str = '', model: str | None = None) -> str:
    """Rewrite one why. Flash-lite uses thinkingBudget 0 and a 128 cap. 3.6 uses minimal."""
    prompt = (
        'Rewrite this one sentence in short words, about a fifth-grade reading level. '
        'Say why it matters for this person or the listener. '
        'Keep the topic of the behavior and the quote. '
        'Do not restate the quote. No advice, no next time, no should. '
        'Good why: This helps people see that your love for art started early.\n'
        'Return only the new sentence.\n'
        f'BEHAVIOR\n{behavior or "None."}\n'
        f'QUOTE\n{quote or "None."}\nTEXT\n{text}'
    )
    wanted = model or personal_model_name()
    try:
        return _raw_text(_invoke_personal(
            llm, prompt, wanted, simplify='3.6' not in _bare_model(wanted),
        )).strip()
    except Exception as exc:
        logging.warning('why repair failed: %s', exc)
        return text


def _repair_whys(llm, payload: dict, quotes: list[str], model: str | None = None) -> dict:
    """Keep each behavior and quote. Rewrite only a why that drifts or fails the grade."""
    data = dict(payload)
    bullets = []
    for item in data.get('bullets') or []:
        if not isinstance(item, dict):
            continue
        bullet = dict(item)
        behavior = str(bullet.get('behavior') or '')
        why = str(bullet.get('why') or '')
        quote = _quote_by_id(str(bullet.get('quote') or ''), quotes, -1)
        problem = _why_problem(why, quote, behavior, quotes)
        if problem.startswith('why drifts') or 'reading level' in problem or problem.startswith('a why line'):
            bullet['why'] = _repair_sentence(llm, why, quote, behavior, model)
        bullets.append(bullet)
    if bullets:
        data['bullets'] = bullets
    return data


def _usable_model_bullets(payload: dict | None, quotes: list[str]) -> list[str]:
    """Keep behaviors and quotes that pass. Drop only a why that still fails."""
    if not isinstance(payload, dict):
        return []
    rendered = []
    used: set[str] = set()
    for source in payload.get('bullets') or []:
        if not isinstance(source, dict):
            continue
        behavior = str(source.get('behavior') or '').strip()
        if not behavior or re.fullmatch(r'you said\.?', behavior.lower()):
            continue
        if advice_problems(behavior) or _is_generic(behavior):
            continue
        quote = _quote_by_id(str(source.get('quote') or ''), quotes, -1)
        if not quote or quote in used:
            continue
        why = str(source.get('why') or '')
        if _why_problem(why, quote, behavior, quotes):
            why = ''
        used.add(quote)
        rendered.append(_render_bullet(behavior, quote, why))
    return rendered if len(rendered) >= 2 else []


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
