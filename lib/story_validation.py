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


def _inflected(token: str, phrase: str) -> bool:
    """True when the token is the phrase or a regular inflection of it.

    'leveraging' and 'leveraged' drop the final e, so a plain prefix test misses them.
    """
    if token == phrase or token.startswith(phrase):
        return True
    for suffix in ('ingly', 'ing', 'ed', 'es', 's'):
        if token.endswith(suffix) and len(token) > len(suffix) + 3:
            stem = token[: -len(suffix)]
            if phrase == stem or (phrase.startswith(stem) and len(phrase) - len(stem) <= 1):
                return True
            return False
    return False


def ban_hit(text: str, *, skip: tuple[str, ...] = ()) -> str | None:
    """Match a banned phrase or an inflection of a single banned word."""
    lowered = (text or '').lower()
    tokens = re.findall(r"[a-z']+", lowered)
    for phrase in BAN_PHRASES:
        if phrase in skip:
            continue
        if ' ' in phrase:
            if phrase in lowered:
                return phrase
            continue
        for token in tokens:
            if _inflected(token, phrase):
                return phrase
    return None


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


def clock_to_ms(timestamp: str) -> int:
    parts = str(timestamp or '').strip().split(':')
    if len(parts) not in (2, 3):
        _fail(f'timestamp is not a clock time: {timestamp}')
    try:
        numbers = [int(part) for part in parts]
    except ValueError:
        _fail(f'timestamp is not a clock time: {timestamp}')
    if len(numbers) == 2:
        minutes, seconds = numbers
        return (minutes * 60 + seconds) * 1000
    hours, minutes, seconds = numbers
    return (hours * 3600 + minutes * 60 + seconds) * 1000


def quote_word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9']+", text or ''))


def _quote_in_turn(quote: str, raw: str) -> str | None:
    """Return the transcript's own casing, with a leading ellipsis when the quote starts mid-turn.

    A trimmed leading word still matches. The rendered words keep the turn's case.
    """
    if not quote or not raw:
        return None

    def _shown(source: str, needle: str) -> str | None:
        index = source.casefold().find(needle.casefold())
        if index < 0:
            return None
        prefix = '…' if source[:index].strip() else ''
        return prefix + source[index:index + len(needle)]

    exact = _shown(raw, quote)
    if exact is not None:
        return exact
    squashed = re.sub(r'\s+', ' ', raw).strip()
    squashed_quote = re.sub(r'\s+', ' ', quote).strip()
    return _shown(squashed, squashed_quote)


def own_quote(text: str, timestamp: str, transcript: dict | None, label: str) -> str:
    """Accept a quote only when it is this speaker's line, in that turn's own words and clock."""
    from lib.story_evidence import clock

    cleaned = (text or '').strip().strip('"').strip()
    if quote_word_count(cleaned) < 3:
        _fail('quote is under 3 words and does not count as evidence')
    given = str(timestamp or '').strip()
    for row in (transcript or {}).get('utterances') or []:
        if row.get('speaker') != label:
            continue
        shown = _quote_in_turn(cleaned, str(row.get('text') or ''))
        if shown is None:
            continue
        if given == clock(row.get('start')):
            return shown
    _fail(f'quote is not inside a turn by {label}: {cleaned}')
    return cleaned


def _named_people(text: str) -> list[str]:
    return re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', text or '')


def quotes_belong_to_speaker(text: str, transcript: dict | None, label: str | None) -> None:
    rows = (transcript or {}).get('utterances') or []
    if label is None:
        blobs = [_normalize(str(row.get('text') or '')) for row in rows]
    else:
        blobs = [
            _normalize(str(row.get('text') or ''))
            for row in rows
            if row.get('speaker') == label
        ]
    for span in quoted_spans(text):
        needle = _normalize(span)
        if len(needle) < 4:
            continue
        if not any(needle in blob for blob in blobs):
            _fail(f'quoted span is not in this speaker\'s turn: {span}')


GLUE_WORDS = {
    'about', 'above', 'after', 'again', 'against', 'also', 'answers', 'asked', 'average',
    'back', 'before', 'began', 'below', 'came', 'center', 'conversation', 'decision',
    'doing', 'early', 'entered', 'every', 'exchange', 'finished', 'first', 'floor',
    'forward', 'from', 'held', 'here', 'into', 'interaction', 'last', 'later', 'left',
    'made', 'most', 'named', 'near', 'next', 'once', 'opened', 'panel', 'participation',
    'percent', 'point', 'question', 'questions', 'quiet', 'room', 'said', 'same',
    'second', 'serial', 'session', 'share', 'speaker', 'started', 'talk', 'that',
    'then', 'third', 'this', 'time', 'times', 'took', 'turn', 'turns', 'voice',
    'voices', 'wait', 'what', 'when', 'where', 'which', 'with', 'word', 'words',
    'talks', 'nobody', 'answered', 'whom', 'quietest', 'decision', 'stayed',
    'your', 'have', 'been', 'were', 'they', 'them', 'their', 'just', 'only', 'than',
    'more', 'less', 'much', 'very', 'into', 'over', 'under', 'after', 'before',
    'would', 'could', 'should', 'there', 'these', 'those', 'other', 'side', 'hard',
    'interesting', 'specific', 'moment', 'start', 'line', 'buys', 'costs', 'doing',
    'phrase', 'often', 'reads', 'right', 'after', 'open', 'using', 'used', 'reach',
    'familiar', 'second', 'offer', 'target', 'halfway', 'step', 'asked', 'question',
    'matters', 'means', 'shows', 'helps', 'seems', 'looks', 'feels', 'makes', 'takes',
    'gives', 'keeps', 'needs', 'wants', 'tries', 'works', 'comes', 'goes', 'says',
    'tells', 'asks', 'holds', 'stays', 'moves', 'opens', 'closes', 'starts', 'ends',
    'follows', 'leads', 'brings', 'leaves', 'happens', 'changes', 'remains', 'appears',
    'becomes', 'includes', 'suggests', 'points', 'shares', 'speaks', 'listens',
    'responds', 'agrees', 'thinks', 'knows', 'sees', 'hears', 'still', 'really',
    'already', 'always', 'often', 'sometimes', 'together', 'around', 'through',
    'during', 'within', 'without', 'between', 'because', 'while', 'clear', 'important',
    'group', 'people', 'person', 'meeting', 'round', 'table', 'success',
}


_SUFFIXES = ('ingly', 'ation', 'ness', 'ment', 'ting', 'ing', 'ied', 'ed', 'es', 'ly', 'er', 'est', 's')


def _stem(token: str) -> str:
    for suffix in _SUFFIXES:
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def unsupported_words(
    text: str,
    corpus: str,
    *,
    names: list[str] | None = None,
    allowed_numbers: set[str] | None = None,
) -> list[str]:
    """Flag content words that are absent from the transcript.

    Function words, inflections, the speaker's name, and fact-list numbers are allowed.
    A shared stem of four letters counts as a paraphrase, not a new claim.
    """
    corpus_tokens = set(re.findall(r"[a-z']{3,}", (corpus or '').lower()))
    allowed_names = set()
    for name in names or []:
        allowed_names.update(re.findall(r"[a-z']{2,}", name.lower()))
    missing = []
    for word in re.findall(r"[A-Za-z']{4,}", text or ''):
        token = word.lower().strip("'")
        if token in GLUE_WORDS or token in allowed_names or len(token) < 5:
            continue
        if token.isdigit() or (allowed_numbers and token in allowed_numbers):
            continue
        stem = _stem(token)
        if token in corpus_tokens or stem in corpus_tokens:
            continue
        key = stem[:5] if len(stem) >= 5 else stem
        if len(key) >= 4 and any(
            item.startswith(key) or key.startswith(item[:5])
            for item in corpus_tokens
            if len(item) >= 4
        ):
            continue
        missing.append(token)
    return missing


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
    speaker_label: str | None = None,
    participant_names: list[str] | None = None,
    skip_bans: tuple[str, ...] = (),
    personal_voice: bool = False,
) -> str:
    """Return the cleaned text, or raise OutputParserException."""
    cleaned = (text or '').strip()
    if not cleaned:
        _fail('story text is empty')
    lowered = cleaned.lower()
    hit = ban_hit(cleaned, skip=skip_bans)
    if hit:
        _fail(f'story text contains {hit!r}')
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
        quotes_belong_to_speaker(cleaned, transcript, speaker_label)
    if participant_names is not None:
        allowed = {name.casefold() for name in participant_names}
        allowed.update(
            part.casefold()
            for name in participant_names
            for part in name.split()
            if len(part) > 2
        )
        for person in _named_people(cleaned):
            if person.casefold() not in allowed and not person.casefold().startswith('speaker '):
                _fail(f'story text names someone who is not in the session: {person}')
    if personal_voice or speaker_label is not None:
        if re.search(r'\b(the speaker|the student|the participant)\b', lowered):
            _fail('personal story must say you, not a role label')
        if re.search(r'\b(he|she|his|her|they)\b', lowered):
            _fail('personal story must stay in second person')
        if re.search(r'(?<=\s)(He|She|They|His|Her)\b', cleaned):
            _fail('personal story capitalizes a pronoun mid-sentence')
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
