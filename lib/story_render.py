"""Turn structured story fields into sentences. Digits come from the facts, not the model."""

from __future__ import annotations

import re

from langchain_core.exceptions import OutputParserException

from lib.story_evidence import before_after, clock
from lib.story_validation import BAN_PHRASES, own_quote, reject_craft_violations, word_count

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
            '{"verb": "ask", "verb_phrase": "one question", "trigger": "before your second turn"}'
        ),
    },
    'group': {
        'character': (
            '{"headline": "12 words or fewer, no digits", '
            '"number_keys": ["interaction"]}'
        ),
        'goal': '{"claim": "no digits, only topics from the transcript"}',
        'challenge': '{"claim": "no digits", "fact_key": "interaction"}',
        'drama': '{"pattern_note": "one line, no digits, only topics from the transcript"}',
        'resolution': '{"claim": "no digits, only topics from the transcript"}',
        'one_thing': (
            '{"verb": "ask", "verb_phrase": "one question back", "trigger": "before the next decision"}'
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
        'The single move with the most effect is next.',
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
    from lib.report_stats import display_bigram

    shown = display_bigram(bigram)
    times = count_label(int(count or 0), 'time')
    if bigram:
        what = f'You reach for "{shown}" when you want the next point to land with the room.'
        so_what = f'It showed up {times}, so the room hears that phrase as your way in.'
    else:
        what = 'No two-word phrase repeated often enough to name one habit.'
        so_what = 'The turns do not lean on one repeated pair, so no phrase stands in for you.'
    return {
        'what': what,
        'so_what': so_what,
        'now_what': 'Pause once before you use it, and let the other person answer.',
    }


def clip_clause(text: str, limit: int = 25) -> str:
    """One sentence or clause, capped so a whole turn is not pasted."""
    sentence = re.split(r'(?<=[.!?])\s+', (text or '').strip())[0]
    words = sentence.split()
    if len(words) <= limit:
        return sentence.strip()
    return ' '.join(words[:limit]).rstrip(',;:')


def ranked_kinds(
    *,
    share: int,
    speaker_count: int,
    questions: int,
    question_average: int,
    turn_words: float = 0,
    session_turn_words: float = 0,
    early: int | None = None,
    late: int | None = None,
) -> list[str]:
    """Gaps this person does not already meet, largest first.

    Talk share is only a reduction when it is well above an even share.
    Questions are skipped when the person is already at or above the session average.
    """
    average = room_average(speaker_count)
    options: list[tuple[str, float]] = []
    if int(questions) < 1 or int(questions) < int(question_average):
        gap = max(1, int(question_average) - int(questions))
        options.append(('questions', gap))
    if int(share) > average + 10:
        options.append(('talk_high', int(share) - average))
    elif int(share) < average - 5:
        options.append(('talk_low', average - int(share)))
    if turn_words and session_turn_words and turn_words > session_turn_words * 1.2:
        options.append(('turns_long', turn_words - session_turn_words))
    elif turn_words and session_turn_words and turn_words < session_turn_words * 0.8:
        options.append(('turns_short', session_turn_words - turn_words))
    if early is not None and late is not None and abs(int(early) - int(late)) >= 15:
        options.append(('thirds', abs(int(early) - int(late))))
    if not options:
        options.append(('beat', 1))
    options.sort(key=lambda item: item[1], reverse=True)
    return options


def improve_toward(current: int, norm: int, *, increase: bool) -> int:
    """Move to the group norm when that is strictly better, otherwise one step past current.

    The direction comes from the metric. A talk share already under the room's
    even share is not pulled downward, and a count of zero does not stay at zero.
    """
    current = int(current)
    norm = int(norm)
    if increase:
        return norm if norm > current else current + 1
    return norm if norm < current else max(0, current - 1)


def _habit(kind: str, *, share, speaker_count, questions, question_average, turn_words, session_turn_words, early, late):
    average = room_average(speaker_count)
    if kind == 'questions':
        target = improve_toward(int(questions), max(int(question_average), int(questions) + 1), increase=True)
        sentence = (
            f'Ask one question before you add another point. '
            f'Target: {count_label(target, "question")}.'
        )
        why = (
            f'You asked {count_label(int(questions), "question")}, '
            f'against {count_label(int(question_average), "question")} for the session.'
        )
        return sentence, why
    if kind == 'talk_low':
        target = improve_toward(int(share), average, increase=True)
        sentence = f'Say one more detail before you stop. Target: {target}% talk time.'
        why = f'You are at {int(share)}% talk time, below an even share of {average}%.'
        return sentence, why
    if kind == 'turns_short':
        target = improve_toward(int(turn_words), int(session_turn_words), increase=True)
        sentence = f'Name one more detail before the turn ends. Target: {target} words in a turn.'
        why = f'A turn of yours averages {int(turn_words)} words, against {int(session_turn_words)}.'
        return sentence, why
    if kind == 'thirds':
        start = int(early or 0)
        end = int(late or 0)
        increase = end < start
        target = improve_toward(end, start, increase=increase)
        sentence = f'Wait through the opening before the long story. Target: {target}% in the last third.'
        why = f'Talk time moves from {start}% in the first third to {end}% in the last.'
        return sentence, why
    if kind == 'talk_high':
        target = improve_toward(int(share), average, increase=False)
        sentence = f'Pause after the point and hand the turn on. Target: {target}% talk time.'
        why = f'You are at {int(share)}% talk time, above an even share of {average}%.'
        return sentence, why
    if kind == 'beat':
        sentence = 'Pause once before you take the next turn. Target: 1 pause before the second point.'
        why = f'You are at {int(share)}% talk time, near an even share of {room_average(speaker_count)}%.'
        return sentence, why
    target = improve_toward(int(turn_words or 1), int(session_turn_words or 1), increase=False)
    sentence = f'Stop once the point is made. Target: {target} words in a turn.'
    why = f'A turn of yours averages {int(turn_words or 0)} words, against {int(session_turn_words or 0)}.'
    return sentence, why


def one_thing_from_gap(
    *,
    share: int,
    speaker_count: int,
    questions: int,
    question_average: int,
    turn_words: float = 0,
    session_turn_words: float = 0,
    early: int | None = None,
    late: int | None = None,
) -> tuple[str, str]:
    kind = ranked_kinds(
        share=share,
        speaker_count=speaker_count,
        questions=questions,
        question_average=question_average,
        turn_words=turn_words,
        session_turn_words=session_turn_words,
        early=early,
        late=late,
    )[0][0]
    return _habit(
        kind,
        share=share,
        speaker_count=speaker_count,
        questions=questions,
        question_average=question_average,
        turn_words=turn_words,
        session_turn_words=session_turn_words,
        early=early,
        late=late,
    )


def unique_habits(people: list[dict]) -> dict[str, tuple[str, str]]:
    """One habit sentence per person. The largest real gap is claimed first."""
    def ranked_for(person: dict) -> list[tuple[str, float]]:
        average = person.get('question_average', person.get('session_question_average', 0))
        return ranked_kinds(
            share=person['share'],
            speaker_count=person['speaker_count'],
            questions=person['questions'],
            question_average=average,
            turn_words=person.get('turn_words') or 0,
            session_turn_words=person.get('session_turn_words') or 0,
            early=person.get('early'),
            late=person.get('late'),
        )

    used: set[str] = set()
    assigned: dict[str, tuple[str, str]] = {}
    ordered = sorted(people, key=lambda person: ranked_for(person)[0][1], reverse=True)
    for person in ordered:
        label = person['label']
        kinds = ranked_for(person)
        placed = False
        for kind, _size in kinds:
            sentence, why = _habit(kind, **{
                'share': person['share'],
                'speaker_count': person['speaker_count'],
                'questions': person['questions'],
                'question_average': person.get('question_average', person.get('session_question_average', 0)),
                'turn_words': person.get('turn_words') or 0,
                'session_turn_words': person.get('session_turn_words') or 0,
                'early': person.get('early'),
                'late': person.get('late'),
            })
            if sentence not in used:
                used.add(sentence)
                assigned[label] = (sentence, why)
                placed = True
                break
        if not placed:
            sentence, why = _habit(
                kinds[0][0],
                share=person['share'],
                speaker_count=person['speaker_count'],
                questions=person['questions'],
                question_average=person.get('question_average', person.get('session_question_average', 0)),
                turn_words=person.get('turn_words') or 0,
                session_turn_words=person.get('session_turn_words') or 0,
                early=person.get('early'),
                late=person.get('late'),
            )
            sentence = f'{sentence} You are at {int(person["share"])}% now.'
            used.add(sentence)
            assigned[label] = (sentence, why)
    return assigned


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


def group_halfway_target(interaction: int, participation: int, word_interaction: int) -> tuple[int, int, str]:
    """Halfway from interaction toward a norm this session actually measured.

    The norm is the higher of participation and the share of words exchanged.
    It is not a fixed 70.
    """
    options = []
    if int(participation) > int(interaction):
        options.append((int(participation), 'participation'))
    if int(word_interaction) > int(interaction):
        options.append((int(word_interaction), 'the share of words exchanged'))
    if options:
        norm, name = max(options)
    else:
        norm, name = min(100, int(interaction) + 20), 'a wider exchange'
    target = int(round((int(interaction) + norm) / 2))
    if target <= int(interaction):
        target = min(100, int(interaction) + 1)
    return target, norm, name


_PERSON_HEADLINE = re.compile(
    r'\b(you|professional|individual|creative|participant|speaker)\b',
    re.I,
)


def group_headline_is_personal(text: str) -> bool:
    return bool(_PERSON_HEADLINE.search(text or ''))


def group_character_copy(facts: dict) -> str:
    """Two or three sentences from the room's rates, not one bare percentage."""
    interaction = int((facts.get('interaction') or {}).get('value') or 0)
    participation = int((facts.get('participation') or {}).get('value') or 0)
    portrait = facts.get('_portrait') or {}
    sentences = [
        (
            f'{interaction}% interaction is how often a turn moved to someone new, '
            f'beside {participation}% participation.'
        )
    ]
    pair = portrait.get('pair') or ()
    if len(pair) >= 2 and pair[0] and pair[1]:
        sentences.append(f'{pair[0]} and {pair[1]} followed each other more than any other pair.')
    least = portrait.get('least')
    low = portrait.get('low_share')
    high = portrait.get('high_share')
    if least and low is not None and high is not None:
        sentences.append(
            f'{least} was the least connected, and talk time ran from {int(low)}% to {int(high)}%.'
        )
    elif least:
        sentences.append(f'{least} was the least connected voice in the room.')
    return ' '.join(sentences[:3])


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
    transcript: dict | None = None,
    speaker_label: str | None = None,
) -> str:
    """Prose is free except quotes, digits, named people, and the ban list."""
    del corpus  # vocabulary overlap is not a grounding rule
    cleaned = _no_digits(text, label, allowed_numbers)
    if not cleaned:
        raise OutputParserException(f'{label} is empty')
    reject_craft_violations(
        cleaned,
        transcript=transcript,
        allowed_numbers=allowed_numbers,
        forbidden_names=forbidden_names,
        speaker_label=speaker_label,
        participant_names=speaker_names,
    )
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
    target = improve_toward(int(questions), max(int(question_average), int(questions) + 1), increase=True)
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
    facts['_meta'] = {
        'share': int(share),
        'speaker_count': int(speaker_count),
        'questions': int(questions),
        'question_average': int(question_average),
        'early': early,
        'late': late,
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
    target, _norm, _name = group_halfway_target(interaction_pct, participation_pct, word_pct)
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
    index = 1
    for key, fact in facts.items():
        if not isinstance(fact, dict) or 'value' not in fact:
            continue
        lines.append(f'{index}. key={key} value={fact["value"]} unit={fact["unit"]}')
        index += 1
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


def render_one_thing(
    data: dict,
    facts: dict,
    corpus: str,
    forbidden: list[str] | None,
    *,
    transcript: dict | None = None,
    label: str | None = None,
    names: list[str] | None = None,
) -> tuple[str, str]:
    """The model supplies the behavior. The metric and target come from the facts."""
    verb = str(data.get('verb') or '').strip().lower()
    if verb not in VERBS:
        raise OutputParserException(f'one thing verb must be one of {VERBS}')
    numbers = {str(item['value']) for key, item in facts.items() if isinstance(item, dict) and 'value' in item}
    phrase = _prose(
        str(data.get('verb_phrase') or ''),
        corpus,
        'verb_phrase',
        forbidden_names=forbidden,
        allowed_numbers=numbers,
        speaker_names=names,
        transcript=transcript,
        speaker_label=label,
    )
    trigger = _prose(
        str(data.get('trigger') or ''),
        corpus,
        'trigger',
        forbidden_names=forbidden,
        allowed_numbers=numbers,
        speaker_names=names,
        transcript=transcript,
        speaker_label=label,
    )
    if word_count(f'{phrase} {trigger}') > 18:
        raise OutputParserException('one thing behavior is over 18 words')
    meta = facts.get('_meta') or {}
    if 'interaction' in facts and 'questions' not in facts:
        current = int(facts['interaction']['value'])
        participation = int((facts.get('participation') or {}).get('value') or current)
        words = int((facts.get('word_interaction') or {}).get('value') or current)
        target, norm, name = group_halfway_target(current, participation, words)
        sentence = f'Name one reply before the next decision. Target: {target}% interaction.'
        why = f'Interaction is {current}%, halfway toward {norm}% {name}.'
        return sentence, why
    sentence, why = one_thing_from_gap(
        share=int(meta.get('share', facts['talk_share']['value'])),
        speaker_count=int(meta.get('speaker_count', 1)),
        questions=int(meta.get('questions', facts['questions']['value'])),
        question_average=int(meta.get('question_average', 0)),
        early=meta.get('early'),
        late=meta.get('late'),
    )
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
        pattern_text = _prose(
            str(pattern.get('pattern') or ''), corpus, 'pattern',
            forbidden_names=forbidden, transcript=transcript, speaker_label=label,
        )
        twist = _prose(
            str(pattern.get('twist') or ''), corpus, 'twist',
            forbidden_names=forbidden, transcript=transcript, speaker_label=label,
        )
        cost = _prose(
            str(pattern.get('cost') or ''), corpus, 'cost',
            forbidden_names=forbidden, transcript=transcript, speaker_label=label,
        )
        gift = _prose(
            str(pattern.get('gift') or ''), corpus, 'gift',
            forbidden_names=forbidden, transcript=transcript, speaker_label=label,
        )
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
        str(exception.get('what_happened') or ''), corpus, 'what_happened',
        forbidden_names=forbidden, transcript=transcript, speaker_label=label,
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
        headline = _prose(
            str(data.get('headline') or ''), corpus, 'headline',
            forbidden_names=forbidden, transcript=transcript, speaker_label=label,
        )
        if word_count(headline) > 12:
            raise OutputParserException('headline is over 12 words')
        return {'headline': headline, 'numbers': render_numbers(data.get('number_keys'), facts)}
    if section == 'goal':
        claim = _prose(
            str(data.get('claim') or ''), corpus, 'claim',
            forbidden_names=forbidden, transcript=transcript, speaker_label=label,
        )
        quote = data.get('quote') if isinstance(data.get('quote'), dict) else {}
        text = own_quote(str(quote.get('text') or ''), str(quote.get('ts') or ''), transcript, label)
        return {'goal': f'Inferred from what you said: {claim.rstrip(".")}. At {quote.get("ts")} you said "{text}".'}
    if section == 'challenge':
        claim = _prose(
            str(data.get('claim') or ''), corpus, 'claim',
            forbidden_names=forbidden, transcript=transcript, speaker_label=label,
        )
        key = str(data.get('fact_key') or '')
        if key not in facts:
            raise OutputParserException(f'challenge fact_key is not allowed: {key}')
        line = str(facts[key]['line']).rstrip('.')
        return {'challenge': f'{line}. {claim.rstrip(".")}.'}
    if section == 'drama':
        return {'drama': render_drama(data, transcript, label, corpus, forbidden)}
    if section == 'resolution':
        claim = _prose(
            str(data.get('claim') or ''), corpus, 'claim',
            forbidden_names=forbidden, transcript=transcript, speaker_label=label,
        )
        quote = data.get('quote') if isinstance(data.get('quote'), dict) else {}
        text = own_quote(str(quote.get('text') or ''), str(quote.get('ts') or ''), transcript, label)
        return {'resolution': f'At {quote.get("ts")} you said "{text}". {claim.rstrip(".")}.'}
    if section == 'one_thing':
        sentence, why = render_one_thing(
            data, facts, corpus, forbidden, transcript=transcript, label=label
        )
        return {'one_thing': sentence, 'why': why}
    raise OutputParserException(f'unknown section {section}')


def render_group_section(section: str, data: dict, *, transcript, facts: dict, cast: str) -> dict:
    corpus = corpus_text(transcript, None)
    if section == 'character':
        headline = _prose(str(data.get('headline') or ''), corpus, 'headline')
        if word_count(headline) > 12:
            raise OutputParserException('headline is over 12 words')
        if group_headline_is_personal(headline):
            raise OutputParserException(
                'group headline must describe the room, not one person: ' + headline
            )
        if 'interaction' not in facts and 'participation' not in facts:
            raise OutputParserException('group character needs at least one rate fact')
        return {'headline': headline, 'character': group_character_copy(facts)}
    if section in ('goal', 'resolution'):
        claim = _prose(str(data.get('claim') or ''), corpus, section)
        closing = ''
        rows = (transcript or {}).get('utterances') or []
        for row in reversed(rows):
            text = str(row.get('text') or '').strip()
            if text:
                closing = clip_clause(text, 12)
                break
        if section == 'goal':
            return {'goal': f'The turns keep testing one aim: {claim.rstrip(".")}.'}
        closing_fact = facts.get('_closing') or {}
        speaker = closing_fact.get('name') or 'the last voice'
        spoken = closing_fact.get('text') or closing
        if spoken:
            landed = f'The meeting closed on {speaker}, who said "{spoken}". '
        else:
            landed = ''
        return {'resolution': f'{landed}{claim.rstrip(".")}.'}
    if section == 'challenge':
        claim = _prose(str(data.get('claim') or ''), corpus, 'claim')
        key = str(data.get('fact_key') or 'interaction')
        if key not in facts:
            key = 'interaction' if 'interaction' in facts else next(iter(facts), '')
        if key not in facts:
            raise OutputParserException('group challenge needs one rate fact')
        if re.search(r'\d', claim):
            raise OutputParserException(
                'group challenge must use words; the rate is already in the facts'
            )
        line = str(facts[key]['line']).rstrip('.')
        return {
            'challenge': (
                f'{line}. {claim.rstrip(".")}. '
                'A turn could finish without an answer from across the room.'
            )
        }
    if section == 'drama':
        note = _prose(str(data.get('pattern_note') or ''), corpus, 'pattern_note')
        if note.lower().strip('.') == 'who talks to whom, and who nobody answered':
            return {'drama': cast or note}
        return {'drama': f'{cast} {note}'.strip() if cast else note}
    if section == 'one_thing':
        sentence, why = render_one_thing(data, facts, corpus, None)
        return {'one_thing': sentence, 'why': why}
    raise OutputParserException(f'unknown section {section}')
