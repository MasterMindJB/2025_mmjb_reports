"""Turn structured story fields into sentences. Digits come from the facts, not the model."""

from __future__ import annotations

import re

from langchain_core.exceptions import OutputParserException

from lib.story_evidence import before_after, clock, halfway_target
from lib.story_validation import (
    BAN_PHRASES,
    own_quote,
    reject_craft_violations,
    unsupported_words,
    word_count,
)

SCHEMAS = {
    'personal': {
        'character': (
            '{"headline": "12 words or fewer, no digits", '
            '"number_keys": ["talk_share", "questions", "turns"]}'
        ),
        'goal': '{"claim": "no digits", "quote": {"text": "their words", "ts": "m:ss"}}',
        'challenge': '{"claim": "no digits", "fact_key": "questions"}',
        'drama': (
            '{"patterns": [{"quotes": [{"text": "", "ts": ""}], "pattern": "", '
            '"twist": "", "cost": "", "gift": ""}], '
            '"exception": {"quote": "", "ts": "", "what_happened": ""}}'
        ),
        'resolution': '{"claim": "no digits", "quote": {"text": "", "ts": ""}}',
        'one_thing': (
            '{"verb": "ask", "verb_phrase": "one question", "trigger": "before your second turn", '
            '"metric_key": "questions", "target_value": 0}'
        ),
    },
    'group': {
        'character': (
            '{"headline": "12 words or fewer, no digits", '
            '"number_keys": ["participation", "interaction", "word_interaction"]}'
        ),
        'goal': '{"claim": "no digits, only topics from the transcript"}',
        'challenge': '{"claim": "no digits", "fact_key": "interaction"}',
        'drama': '{"pattern_note": "one line, no digits, only topics from the transcript"}',
        'resolution': '{"claim": "no digits, only topics from the transcript"}',
        'one_thing': (
            '{"verb": "ask", "verb_phrase": "one question back", "trigger": "before the next decision", '
            '"metric_key": "interaction", "target_value": 0}'
        ),
    },
}

TRANSITIONS = {
    'after_character': (
        'The numbers are the picture. The aim is next.',
        'That is the shape of the turns.',
    ),
    'after_goal': (
        'Here is what got in the way.',
        'The obstacles were in the turns themselves.',
    ),
    'after_challenge': (
        'The pattern in the wording is next.',
        'Here is the repeated move.',
    ),
    'after_drama': (
        'Here is where the session landed.',
        'The outcome follows from those turns.',
    ),
    'after_resolution': (
        'The highest-leverage move is next.',
        'One change follows from the largest gap.',
    ),
}

GROUP_TRANSITIONS = {
    'after_character': (
        'The numbers describe the room. The aim comes next.',
        'That is the shape of the session.',
    ),
    'after_goal': (
        'Here is what made the exchange hard.',
        'The obstacles show up in the turn pattern.',
    ),
    'after_challenge': (
        'The chord shows who followed whom.',
        'The pattern of turns is next.',
    ),
    'after_drama': (
        'Here is where the meeting landed.',
        'The outcome follows from those turns.',
    ),
    'after_resolution': (
        'The move for the whole room is next.',
        'One change follows from the largest gap.',
    ),
}

PERSONAL_NUMBER_KEYS = ('talk_share', 'questions', 'turns', 'early_share', 'late_share')
GROUP_NUMBER_KEYS = ('participation', 'interaction', 'word_interaction')
VERBS = ('pause', 'ask', 'name', 'repeat', 'count', 'stop', 'wait', 'say')


def pick_transitions(seed: str, *, group: bool = False) -> dict[str, str]:
    """Each seam gets one stock line, rotated so a report does not repeat the same sentence."""
    options_for = GROUP_TRANSITIONS if group else TRANSITIONS
    base = sum(ord(char) for char in (seed or 'session'))
    chosen = {}
    for index, (seam, options) in enumerate(options_for.items()):
        chosen[seam] = options[(base + index) % len(options)]
    return chosen


def computed_signature(bigram: str, count: int, quotes: list[str]) -> dict[str, str]:
    """A real reading when the model does not return What / So, what / Now, what."""
    shown = ''
    if quotes:
        raw = re.sub(r'^\d+:\d+\s*', '', str(quotes[0])).strip().strip('"')
        shown = clip_clause(raw, 20)
    times = count_label(int(count or 0), 'time')
    if bigram:
        what = f'You reach for "{bigram}" while finding the next point.'
        so_what = f'It showed up {times} in this session.'
        if shown:
            so_what = f'{so_what} One line was "{shown}".'
    else:
        what = 'No two-word phrase repeated often enough to name.'
        so_what = 'The turns do not lean on one repeated pair.'
    return {
        'what': what,
        'so_what': so_what,
        'now_what': 'Pause once before using the phrase again.',
    }


def clip_clause(text: str, limit: int = 25) -> str:
    """One sentence or clause, capped so a whole turn is not pasted."""
    sentence = re.split(r'(?<=[.!?])\s+', (text or '').strip())[0]
    words = sentence.split()
    if len(words) <= limit:
        return sentence.strip()
    return ' '.join(words[:limit]).rstrip(',;:')


def one_thing_from_gap(
    *,
    share: int,
    speaker_count: int,
    questions: int,
    question_average: int,
) -> tuple[str, str]:
    """The habit follows this person's largest gap, not one question line for everyone."""
    average = room_average(speaker_count)
    share_gap = int(share) - average
    question_gap = max(int(question_average), 1) - int(questions)
    if share_gap > 5 and share_gap >= question_gap:
        target = halfway_target(int(share), max(average, 25) if int(share) > 25 else int(share))
        if target == int(share):
            target = max(0, int(share) - 1)
        sentence = f'Pause after the point. Target: {target}% talk time.'
        why = f'Talk time is {int(share)}%, {share_relation(share, speaker_count)}.'
        return sentence, why
    if share_gap < -5 and -share_gap >= max(question_gap, 0):
        target = halfway_target(int(share), average)
        sentence = f'Say one added point before handing the turn on. Target: {target}% talk time.'
        why = f'Talk time is {int(share)}%, {share_relation(share, speaker_count)}.'
        return sentence, why
    target = halfway_target(int(questions), 3 if int(questions) < 3 else int(questions))
    sentence = f'Ask one question before the second long turn. Target: {count_label(target, "question")}.'
    why = (
        f'{questions_sentence(questions, question_average)} '
        f'{count_label(target, "question")} is the halfway step.'
    )
    return sentence, why


def count_label(value: int, singular: str, plural: str | None = None) -> str:
    word = singular if int(value) == 1 else (plural or f'{singular}s')
    return f'{int(value)} {word}'


def room_average(speaker_count: int) -> int:
    if speaker_count <= 0:
        return 0
    return int(round(100 / speaker_count))


def share_relation(share: int, speaker_count: int) -> str:
    average = room_average(speaker_count)
    gap = int(share) - average
    if abs(gap) <= 5:
        return 'near the room average'
    if gap > 0 and int(share) >= 40:
        return 'above the room average, and you held the floor most of the way'
    if gap > 0:
        return 'above the room average'
    return 'below the room average'


def share_sentence(share: int, speaker_count: int) -> str:
    return f'{int(share)}% talk time: {share_relation(share, speaker_count)}.'


def questions_sentence(count: int, average: int) -> str:
    label = count_label(count, 'question')
    average_label = count_label(average, 'question')
    if int(count) == int(average):
        relation = 'the same as the session average'
    elif int(count) < int(average):
        relation = f'below the session average of {average_label}'
    else:
        relation = f'above the session average of {average_label}'
    if int(count) == 0:
        return f'{label}: you did not ask, {relation}.'
    return f'{label}: {relation}.'


def turns_sentence(count: int) -> str:
    label = count_label(count, 'turn')
    if int(count) == 1:
        return f'{label}: you came in once.'
    return f'{label}: you came in {int(count)} times.'


def rate_sentence(value: int, name: str) -> str:
    """A percentage is a rate. It is never phrased as a count of events."""
    return f'{int(value)}% {name}'


def _no_digits(text: str, label: str, allowed: set[str] | None = None) -> str:
    cleaned = (text or '').strip()
    extra = [number for number in re.findall(r'\d+', cleaned) if not allowed or number not in allowed]
    if extra and allowed is None:
        raise OutputParserException(f'{label} must not contain a digit; facts supply the numbers')
    if extra and allowed is not None:
        raise OutputParserException(f'{label} uses numbers that are not in the fact list: {extra}')
    return cleaned


def _prose(
    text: str,
    corpus: str,
    label: str,
    *,
    forbidden_names: list[str] | None = None,
    allowed_numbers: set[str] | None = None,
    speaker_names: list[str] | None = None,
) -> str:
    cleaned = _no_digits(text, label, allowed_numbers)
    if not cleaned:
        raise OutputParserException(f'{label} is empty')
    reject_craft_violations(cleaned, forbidden_names=forbidden_names, allowed_numbers=allowed_numbers)
    missing = unsupported_words(
        cleaned, corpus, names=speaker_names, allowed_numbers=allowed_numbers
    )
    if missing:
        raise OutputParserException(f'{label} uses words that are not in the transcript: {missing}')
    return cleaned


def personal_facts(
    *,
    transcript,
    label: str,
    share: int,
    turns: int,
    questions: int,
    question_average: int,
    speaker_count: int,
) -> dict:
    early = late = None
    contrast = before_after(transcript, label)
    if contrast:
        early_text, late_text = contrast.split(' then ')
        early = int(re.search(r'\d+', early_text).group(0))
        late = int(re.search(r'\d+', late_text).group(0))
    target = halfway_target(int(questions), 3 if int(questions) < 3 else int(questions))
    facts = {
        'talk_share': {
            'value': int(share),
            'unit': 'percent of talk time',
            'line': share_sentence(share, speaker_count),
        },
        'questions': {
            'value': int(questions),
            'unit': 'questions',
            'line': questions_sentence(questions, question_average),
        },
        'turns': {
            'value': int(turns),
            'unit': 'turns',
            'line': turns_sentence(turns),
        },
        'target_questions': {
            'value': target,
            'unit': 'questions',
            'line': f'Target {count_label(target, "question")}',
        },
    }
    if early is not None:
        facts['early_share'] = {
            'value': early,
            'unit': 'percent of talk time in the first third',
            'line': contrast,
        }
        facts['late_share'] = {
            'value': late,
            'unit': 'percent of talk time in the last third',
            'line': contrast,
        }
    return facts


def group_facts(*, participation: float, interaction: float, word_interaction: float) -> dict:
    participation_pct = int(round(100 * participation))
    interaction_pct = int(round(100 * interaction))
    word_pct = int(round(100 * word_interaction))
    target = halfway_target(interaction_pct, 70 if interaction_pct < 70 else interaction_pct)
    return {
        'participation': {
            'value': participation_pct,
            'unit': 'percent participation, a rate not a count of events',
            'line': rate_sentence(participation_pct, 'participation'),
        },
        'interaction': {
            'value': interaction_pct,
            'unit': 'percent interaction, a rate not a count of events',
            'line': rate_sentence(interaction_pct, 'interaction'),
        },
        'word_interaction': {
            'value': word_pct,
            'unit': 'percent of words exchanged, a rate not a count of events',
            'line': rate_sentence(word_pct, 'of the words exchanged'),
        },
        'target_interaction': {
            'value': target,
            'unit': 'percent interaction',
            'line': rate_sentence(target, 'interaction'),
        },
    }


def facts_prompt(facts: dict, quotes: list[dict], following: list[dict], excerpt: str = '') -> str:
    lines = ['FACTS']
    for index, key in enumerate(facts, start=1):
        fact = facts[key]
        lines.append(
            f'{index}. key={key} value={fact["value"]} unit={fact["unit"]}'
        )
    if quotes:
        lines.append('OWN QUOTES')
        for quote in quotes:
            lines.append(f'ts={quote["time"]} text={quote["text"]}')
    if following:
        lines.append('NEXT SPEAKER, not this person. Do not put these words in a quote field.')
        for quote in following:
            lines.append(f'ts={quote["time"]} text={quote["text"]}')
    if excerpt:
        lines.append('TRANSCRIPT EXCERPT')
        lines.append(excerpt)
    return '\n'.join(lines)


def transcript_excerpt(transcript: dict | None, names: dict[str, str], limit: int = 3500) -> str:
    rows = []
    for utterance in (transcript or {}).get('utterances') or []:
        text = str(utterance.get('text') or '').strip()
        if not text:
            continue
        who = names.get(utterance.get('speaker'), utterance.get('speaker') or 'Speaker')
        rows.append(f'{clock(utterance.get("start"))} {who}: {text}')
    excerpt = '\n'.join(rows)
    if len(excerpt) <= limit:
        return excerpt
    return excerpt[:limit].rsplit('\n', 1)[0]


def corpus_text(transcript, label: str | None = None) -> str:
    parts = []
    for utterance in (transcript or {}).get('utterances') or []:
        if label is not None and utterance.get('speaker') != label:
            continue
        parts.append(str(utterance.get('text') or ''))
    return ' '.join(parts)


def render_numbers(keys: list, facts: dict) -> list[str]:
    if not isinstance(keys, list) or len(keys) != 3:
        raise OutputParserException('number_keys must be three fact keys')
    lines = []
    seen = set()
    for key in keys:
        if key not in facts or key in seen or str(key).startswith('target_'):
            raise OutputParserException(f'number key is not an allowed fact: {key}')
        seen.add(key)
        lines.append(facts[key]['line'])
    return lines


def render_one_thing(data: dict, facts: dict, corpus: str, forbidden: list[str] | None) -> tuple[str, str]:
    verb = str(data.get('verb') or '').strip().lower()
    if verb not in VERBS:
        raise OutputParserException(f'one thing verb must be one of {VERBS}')
    phrase = _prose(str(data.get('verb_phrase') or ''), corpus, 'verb_phrase', forbidden_names=forbidden)
    trigger = _prose(str(data.get('trigger') or ''), corpus, 'trigger', forbidden_names=forbidden)
    metric = str(data.get('metric_key') or '')
    target_key = {'questions': 'target_questions', 'interaction': 'target_interaction'}.get(metric)
    if target_key is None or target_key not in facts:
        raise OutputParserException(f'one thing metric is not a fact: {metric}')
    fact = facts[target_key]
    try:
        given = int(data.get('target_value'))
    except (TypeError, ValueError):
        raise OutputParserException('target_value must be the halfway number')
    if given != int(fact['value']):
        raise OutputParserException(
            f'target_value {given} is not the halfway value {fact["value"]}'
        )
    unit = 'question' if metric == 'questions' and given == 1 else (
        'questions' if metric == 'questions' else 'percent'
    )
    if metric == 'interaction' or metric == 'talk_share':
        sentence = f'{verb.capitalize()} {phrase} {trigger}. Target: {given}%.'
    else:
        sentence = f'{verb.capitalize()} {phrase} {trigger}. Target: {count_label(given, "question")}.'
    sentence = re.sub(r'\s+', ' ', sentence).strip()
    if word_count(sentence) > 25:
        raise OutputParserException('rendered one thing is over 25 words')
    if re.search(r'\balso\b', sentence.lower()):
        raise OutputParserException("one thing must not say 'also'")
    why_unit = unit if metric == 'questions' else 'percent'
    current = facts.get(metric, fact)['value']
    if metric == 'questions':
        why = f'You asked {count_label(current, "question")}, so {count_label(given, "question")} is the halfway step.'
    else:
        why = f'The rate is {current}%, so {given}% is the halfway step.'
    return sentence, why


def render_drama(data: dict, transcript, label: str, corpus: str, forbidden: list[str] | None) -> str:
    patterns = data.get('patterns')
    if not isinstance(patterns, list) or not patterns or len(patterns) > 2:
        raise OutputParserException('drama needs one or two patterns')
    sentences = []
    mights = 0
    for pattern in patterns:
        quotes = pattern.get('quotes') if isinstance(pattern, dict) else None
        if not isinstance(quotes, list) or not quotes or len(quotes) > 3:
            raise OutputParserException('each pattern needs 1 to 3 quotes')
        rendered_quotes = []
        for item in quotes:
            text = own_quote(str(item.get('text') or ''), str(item.get('ts') or ''), transcript, label)
            rendered_quotes.append((str(item.get('ts')).strip(), text))
        pattern_text = _prose(str(pattern.get('pattern') or ''), corpus, 'pattern', forbidden_names=forbidden)
        twist = _prose(str(pattern.get('twist') or ''), corpus, 'twist', forbidden_names=forbidden)
        cost = _prose(str(pattern.get('cost') or ''), corpus, 'cost', forbidden_names=forbidden)
        gift = _prose(str(pattern.get('gift') or ''), corpus, 'gift', forbidden_names=forbidden)
        when, said = rendered_quotes[0]
        sentences.append(f'At {when} you said "{said}".')
        for extra_when, extra_said in rendered_quotes[1:]:
            sentences.append(f'At {extra_when} you said "{extra_said}".')
        sentences.append(pattern_text if pattern_text.endswith('.') else pattern_text + '.')
        sentences.append(f"Here's what's interesting: {twist.rstrip('.')}.")
        sentences.append(f"It's doing a job for you. It buys {gift.rstrip('.')}. It also costs {cost.rstrip('.')}.")
        mights += len(re.findall(r'\bmight\b', ' '.join(sentences).lower()))
    exception = data.get('exception') if isinstance(data.get('exception'), dict) else None
    if not exception:
        raise OutputParserException('drama needs an exception')
    exception_quote = own_quote(
        str(exception.get('quote') or ''), str(exception.get('ts') or ''), transcript, label
    )
    happened = _prose(
        str(exception.get('what_happened') or ''), corpus, 'what_happened', forbidden_names=forbidden
    )
    sentences.append(
        f"The one moment it didn't happen was {str(exception.get('ts')).strip()}, "
        f'when you said "{exception_quote}", and that\'s when {happened.rstrip(".")}.'
    )
    rendered = ' '.join(sentences)
    if len(re.findall(r'\bmight\b', rendered.lower())) > 1:
        raise OutputParserException('drama uses might more than once')
    for phrase in BAN_PHRASES:
        if phrase in rendered.lower():
            raise OutputParserException(f'drama contains {phrase!r}')
    return rendered


def render_personal_section(section: str, data: dict, *, transcript, label: str, facts: dict, forbidden: list[str]) -> dict:
    corpus = corpus_text(transcript, label)
    if section == 'character':
        headline = _prose(str(data.get('headline') or ''), corpus, 'headline', forbidden_names=forbidden)
        if word_count(headline) > 12:
            raise OutputParserException('headline is over 12 words')
        return {'headline': headline, 'numbers': render_numbers(data.get('number_keys'), facts)}
    if section == 'goal':
        claim = _prose(str(data.get('claim') or ''), corpus, 'claim', forbidden_names=forbidden)
        quote = data.get('quote') if isinstance(data.get('quote'), dict) else {}
        text = own_quote(str(quote.get('text') or ''), str(quote.get('ts') or ''), transcript, label)
        return {'goal': f'Inferred from what you said: {claim.rstrip(".")}. At {quote.get("ts")} you said "{text}".'}
    if section == 'challenge':
        claim = _prose(str(data.get('claim') or ''), corpus, 'claim', forbidden_names=forbidden)
        key = str(data.get('fact_key') or '')
        if key not in facts:
            raise OutputParserException(f'challenge fact_key is not allowed: {key}')
        return {'challenge': f'{facts[key]["line"]} {claim.rstrip(".")}.'}
    if section == 'drama':
        return {'drama': render_drama(data, transcript, label, corpus, forbidden)}
    if section == 'resolution':
        claim = _prose(str(data.get('claim') or ''), corpus, 'claim', forbidden_names=forbidden)
        quote = data.get('quote') if isinstance(data.get('quote'), dict) else {}
        text = own_quote(str(quote.get('text') or ''), str(quote.get('ts') or ''), transcript, label)
        return {'resolution': f'At {quote.get("ts")} you said "{text}". {claim.rstrip(".")}.'}
    if section == 'one_thing':
        sentence, why = render_one_thing(data, facts, corpus, forbidden)
        return {'one_thing': sentence, 'why': why}
    raise OutputParserException(f'unknown section {section}')


def render_group_section(section: str, data: dict, *, transcript, facts: dict, cast: str) -> dict:
    corpus = corpus_text(transcript, None)
    if section == 'character':
        headline = _prose(str(data.get('headline') or ''), corpus, 'headline')
        if word_count(headline) > 12:
            raise OutputParserException('headline is over 12 words')
        keys = data.get('number_keys')
        if not isinstance(keys, list) or sorted(keys) != sorted(GROUP_NUMBER_KEYS):
            raise OutputParserException('group character uses the three rate facts')
        lines = [facts[key]['line'].rstrip('.') + '.' for key in GROUP_NUMBER_KEYS]
        return {'headline': headline, 'character': ' '.join(lines)}
    if section in ('goal', 'resolution'):
        claim = _prose(str(data.get('claim') or ''), corpus, section)
        if section == 'goal':
            return {'goal': f'Inferred from the turns in this session: {claim.rstrip(".")}.'}
        return {'resolution': claim if claim.endswith('.') else claim + '.'}
    if section == 'challenge':
        claim = _prose(str(data.get('claim') or ''), corpus, 'claim')
        key = str(data.get('fact_key') or '')
        if key not in ('participation', 'interaction', 'word_interaction'):
            raise OutputParserException('group challenge must use a rate fact')
        return {'challenge': f'{facts[key]["line"]}. {claim.rstrip(".")}.'}
    if section == 'drama':
        note = _prose(str(data.get('pattern_note') or ''), corpus, 'pattern_note')
        if note.lower().strip('.') == 'who talks to whom, and who nobody answered':
            return {'drama': cast or note}
        return {'drama': f'{cast} {note}'.strip() if cast else note}
    if section == 'one_thing':
        sentence, why = render_one_thing(data, facts, corpus, None)
        return {'one_thing': sentence, 'why': why}
    raise OutputParserException(f'unknown section {section}')
