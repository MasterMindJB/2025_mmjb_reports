"""Turn structured story fields into sentences. Digits come from the facts, not the model."""

from __future__ import annotations

import re

from langchain_core.exceptions import OutputParserException

from lib.story_evidence import before_after, clock
from lib.story_validation import BAN_PHRASES, own_quote, own_quote_turn, reject_craft_violations, word_count

SCHEMAS = {
    'personal': {
        'character': (
            '{"headline": "12 words or fewer, no digits", '
            '"number_keys": ["talk_share", "questions", "turns"]}'
        ),
        'goal': (
            '{"claim": "verb phrase starting with a base verb, not with the", '
            '"quote": {"text": "their words", "ts": "m:ss"}}'
        ),
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
        'goal': '{"claim": "verb phrase starting with a base verb, not with the"}',
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
    ),
        'after_drama': (
        'The last turns show where the exchange settled.',
        'The outcome follows from those turns.',
    ),
    'after_resolution': (
        'The single move with the most effect is next.',
        'The largest gap points to the next move.',
    ),
}

GROUP_TRANSITIONS = {
    'after_character': (
        'The numbers describe the room. The aim comes next.',
        'The aim comes next.',
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
        'The last turns show where the meeting settled.',
        'The outcome follows from those turns.',
    ),
    'after_resolution': (
        'The move for the whole room is next.',
        'The largest gap points to the next move.',
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


_ARTICLES = {'a', 'an', 'the', 'your', 'one', 'some', 'this', 'that'}


_FINITE_VERBS = {
    'establish', 'endure', 'secure', 'secured', 'is', 'are', 'was', 'were',
    'have', 'has', 'do', 'does', 'get', 'gets', 'make', 'makes', 'keep', 'keeps',
    'create', 'creates', 'build', 'builds', 'hold', 'holds', 'give', 'gives',
    'spend', 'spends', 'lose', 'loses', 'pay', 'pays', 'face', 'faces',
    'carry', 'carries', 'cost', 'costs', 'buy', 'buys', 'take', 'takes',
    'need', 'needs', 'want', 'wants', 'leave', 'leaves', 'bring', 'brings',
    'show', 'shows', 'find', 'finds', 'see', 'sees', 'set', 'sets', 'use', 'uses',
    'start', 'starts', 'stop', 'stops', 'wait', 'waits', 'hand', 'hands',
    'aim', 'aims', 'seek', 'seeks', 'try', 'tries', 'maintain', 'maintains',
}


def _finite_verb(word: str) -> bool:
    token = (word or '').lower().strip('.,')
    return token in _FINITE_VERBS


def _benefit_clause(phrase: str) -> str:
    """'you' plus a verb is a clause. A noun phrase after 'you' stays a noun phrase.

    'you you' collapses to one 'you'. 'you decades of professional struggle' has no
    finite verb, so it renders as the noun phrase without a second 'you'.
    """
    words = (phrase or '').strip().rstrip('.').split()
    while len(words) >= 2 and words[0].lower() == 'you' and words[1].lower() == 'you':
        words = words[1:]
    if not words or words == ['you']:
        raise OutputParserException('needs a clause')
    if words[0].lower() == 'you':
        if len(words) > 1 and _finite_verb(words[1]):
            return ' '.join(words)
        return ' '.join(words[1:])
    if _finite_verb(words[0]):
        return 'you ' + ' '.join(words)
    return ' '.join(words)


def _talk_sides(gift: str, cost: str) -> tuple[str, str]:
    """Cost and gift describe how this person talks, not someone inside an anecdote."""
    blob = f'{gift} {cost}'
    anecdote = re.search(r'\b(employee|her story|his story|their story)\b', blob, re.I)
    if anecdote:
        return (
            'What it gets you: you put the point in your own words.',
            'What it costs you: the next voice has to answer that move.',
        )
    return _gets_you(gift), _costs_you(cost)


def _gets_you(phrase: str) -> str:
    return f'What it gets you: {_benefit_clause(phrase)}.'


def _costs_you(phrase: str) -> str:
    return f'What it costs you: {_benefit_clause(phrase)}.'


def _strip_stated_by(text: str) -> str:
    return re.sub(r'\s*,?\s*as stated by\s+"[^"]*"', '', text or '', flags=re.I).strip()


def _topic_phrase(claim: str) -> str:
    """A topic noun phrase. Drop a leading wrapper or infinitive so it is not glued on twice."""
    text = (claim or '').strip().rstrip('.')
    text = re.sub(
        r'^(the turns keep returning to|the group seeks to|the primary objective involves)\s+',
        '',
        text,
        flags=re.I,
    )
    previous = None
    while previous != text:
        previous = text
        text = re.sub(r'^(returning to|to)\s+', '', text, flags=re.I)
        text = re.sub(r'^identifying\s+', '', text, flags=re.I)
        text = re.sub(
            r'^(identify|identifying)\s+(?=(where|when|what|how|why)\b)',
            '',
            text,
            flags=re.I,
        )
    if text[:1].isupper():
        text = text[0].lower() + text[1:]
    return text


_BASE_VERBS = {
    'ask', 'build', 'choose', 'clarify', 'close', 'compare', 'connect', 'create',
    'decide', 'define', 'describe', 'endure', 'establish', 'explain', 'find', 'give',
    'hold', 'identify', 'keep', 'leave', 'maintain', 'make', 'mark', 'measure',
    'move', 'name', 'notice', 'open', 'pause', 'place', 'protect', 'reduce', 'say',
    'secure', 'separate', 'set', 'share', 'show', 'start', 'state', 'stop', 'test',
    'use', 'wait', 'help', 'get', 'let', 'put', 'take', 'bring', 'tell', 'talk',
    'call', 'work', 'learn', 'lead', 'join', 'offer', 'focus', 'change', 'grow',
    'see', 'know', 'think', 'feel', 'read', 'write', 'hear', 'meet', 'reach',
    'support', 'explore', 'discuss', 'turn', 'add', 'bring', 'recognize',
    'prioritize', 'demonstrate', 'integrate', 'strengthen', 'advocate', 'live',
    'feel', 'defend', 'invite', 'practice', 'listen', 'notice',
}
_FRAME_VERBS = {
    'aim', 'aims', 'are', 'involve', 'involves', 'is', 'need', 'needs', 'seek',
    'seeks', 'try', 'tries', 'want', 'wants',
}


def _tagged_verb(word: str) -> bool:
    try:
        import nltk

        tag = nltk.pos_tag([word])[0][1]
    except Exception:
        return False
    return str(tag) in {'VB', 'VBP'}


def _base_verb(token: str) -> str | None:
    word = (token or '').lower().strip('.,')
    if word in _FRAME_VERBS:
        return None
    if word in _BASE_VERBS:
        return word
    if word.endswith('ing') and len(word) > 5:
        stem = word[:-3]
        if stem in _BASE_VERBS:
            return stem
        if f'{stem}e' in _BASE_VERBS:
            return f'{stem}e'
    return None


_DISCOURSE = re.compile(
    r'^(?:i just wanted to add|i just want to add|just because|i think|i mean|you know|'
    r'oh|um+|uh+|so|well|yeah|yes)\b[,:]?\s*',
    re.I,
)


def _trim_discourse(text: str) -> str:
    """Drop leading oh / I think / I just wanted to add, and the same kind of filler."""
    cleaned = (text or '').strip()
    previous = None
    while cleaned and previous != cleaned:
        previous = cleaned
        cleaned = _DISCOURSE.sub('', cleaned).strip(' ,')
    return cleaned


def _clause_has_verb(text: str) -> bool:
    for word in re.findall(r"[A-Za-z']+", text or ''):
        if _base_verb(word) or _tagged_verb(word):
            return True
    return False


def verb_phrase(claim: str) -> str:
    """A goal is a verb phrase. A grounded phrase that is not one gets a verb prefix."""
    text = _strip_stated_by(claim).strip().rstrip('.')
    text = re.sub(r'^(?:the room|you)\s+set out to\s+', '', text, flags=re.I)
    text = _trim_discourse(text)
    previous = None
    while previous != text:
        previous = text
        text = re.sub(
            r'^(?:i want to|i wanted to|we need to|we wanted to|trying to|wanted to|the goal is to|to|you)\s+',
            '',
            text,
            flags=re.I,
        )
        text = _trim_discourse(text)
    words = text.split()
    for index, word in enumerate(words):
        if word.lower() == 'the':
            continue
        base = _base_verb(word)
        if not base and index == 0 and _tagged_verb(word):
            base = word.lower().strip('.,')
        if not base:
            continue
        rest = ' '.join(words[index + 1:]).strip()
        phrase = f'{base} {rest}'.strip()
        if phrase.lower().startswith('the '):
            continue
        words_out = phrase.split()
        if len(words_out) < 3:
            continue
        if len(words_out) > 20:
            phrase = ' '.join(words_out[:20])
        if re.match(r'^(?:feel|live) rather than\b', phrase, re.I):
            rest = phrase.split('rather than', 1)[1].strip()
            return 'AIM:flexibility rather than ' + rest
        return phrase
    phrase = text.strip().rstrip('.')
    if phrase[:1].isupper():
        phrase = phrase[0].lower() + phrase[1:]
    words_out = phrase.split()
    if len(words_out) > 20:
        phrase = ' '.join(words_out[:20])
        words_out = phrase.split()
    if len(words_out) >= 3:
        first = re.sub(r'[^\w]', '', words_out[0])
        if _base_verb(first) or _tagged_verb(first):
            return phrase
        return 'AIM:' + phrase
    raise OutputParserException('goal must be a verb phrase starting with a base verb')


def _what_happened(claim: str) -> str:
    """A bare imperative becomes a what-happened clause."""
    body = (claim or '').strip().rstrip('.')
    if not body or re.match(r'^(what happened|you)\b', body, re.I):
        return body
    first = re.sub(r'[^\w]', '', body.split()[0])
    if _base_verb(first):
        lowered = body[0].lower() + body[1:]
        return f'What happened is that you {lowered}'
    return body


def goal_line(claim: str, *, personal: bool = False) -> str:
    """'The room set out to <verb phrase>.' A non-verb phrase is 'Your aim: ...', never 'share <verb>'."""
    phrase = verb_phrase(claim)
    if phrase.startswith('AIM:'):
        bare = phrase[4:].strip()
        if personal:
            return f'Your aim: {bare}.'
        return f"The room's aim: {bare}."
    subject = 'You' if personal else 'The room'
    return f'{subject} set out to {phrase}.'


def _cap_sentence(text: str) -> str:
    text = (text or '').strip()
    if text[:1].islower():
        return text[0].upper() + text[1:]
    return text


_QUESTION_OPENERS = {'what', 'why', 'how', 'when', 'where', 'who', 'which', 'so'}
_LOGISTICS_PHRASES = (
    'photo', 'photos', 'picture', 'pictures', 'headshot', 'smile',
    'eventbrite', 'signup', 'sign up',
    'thank you', 'thanks', 'goodbye', 'good bye',
    'packing up', 'pack up', 'packed up',
    'they closed', 'closed about', 'closed at',
)


def _word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z']+", text or ''))


_LOGISTICS_START = re.compile(
    r'^(thanks|thank you|let me stand up|smile|say cheese|take a photo|photo|picture)\b',
    re.I,
)
def is_logistics(text: str) -> bool:
    """Photos, smiles, venue hours, thanks, and packing up are not the closer.

    A turn that starts with thanks, 'let me stand up', or a photo cue is logistics
    at any length. Other logistics cues still have to be a short turn.
    """
    lowered = (text or '').lower().strip()
    if not lowered:
        return False
    if _LOGISTICS_START.match(lowered):
        return True
    count = _word_count(text)
    if count == 0 or count > 16:
        return False
    if any(phrase in lowered for phrase in _LOGISTICS_PHRASES):
        return True
    words = re.findall(r"[a-z']+", lowered)
    return 'present' in words or 'stay there' in lowered


def substantive_line(text: str) -> bool:
    """A closing line needs six words and must not be a question opener."""
    words = re.findall(r"[A-Za-z']+", text or '')
    if len(words) < 6:
        return False
    if (text or '').strip().endswith('?'):
        return False
    if words[0].lower() in _QUESTION_OPENERS:
        return False
    return True


def strongest_sentence(text: str) -> str | None:
    """The longest complete sentence, not the last fragment and not 'et cetera'."""
    best = None
    for part in re.split(r'(?<=[.!?])\s+', (text or '').strip()):
        if re.search(r'\bet cetera\b|\betc\.?\b', part, re.I):
            continue
        if not part.endswith(('.', '!', '?')):
            continue
        words = re.findall(r"[A-Za-z']+", part)
        if len(words) < 6:
            continue
        if best is None or len(words) > best[0]:
            best = (len(words), part.strip())
    return None if best is None else best[1]


def closing_clip(text: str, limit: int = 18) -> str | None:
    """The closer is the first complete sentence, at least six words, never cut mid-clause.

    'good sign. If Stan if stands, that's' is rejected: the first sentence is too short
    to borrow the next fragment, and that fragment does not finish a clause.
    """
    del limit  # a word cap would cut the clause
    raw = (text or '').strip()
    if not raw:
        return None
    first = re.split(r'(?<=[.!?])\s+', raw)[0].strip()
    if _word_count(first) < 6 or not first.endswith(('.', '!', '?')):
        return None
    if not substantive_line(first):
        return None
    return first


def discussion_rows(rows: list) -> list:
    """The analysis window: last on-topic turn, then the closing cues, then cut."""
    from lib.session_window import window_through_closing

    return window_through_closing(rows)


def _complete_clause(text: str) -> str:
    """Group drama has to finish a sentence. A dangling tail is cut, not printed."""
    raw = (text or '').strip()
    if not raw:
        raise OutputParserException('group drama ends mid-clause')
    parts = [part.strip() for part in re.split(r'(?<=[.!?])\s+', raw) if part.strip()]
    complete = [part for part in parts if part.endswith(('.', '!', '?')) and _word_count(part) >= 3]
    if raw.endswith(('.', '!', '?')) and complete and len(complete) == len(parts):
        return capitalize_sentences(raw)
    if complete:
        return capitalize_sentences(' '.join(complete))
    raise OutputParserException('group drama ends mid-clause')


def last_substantive_utterance(rows: list) -> dict | None:
    """The closer is the last on-topic turn before the closing cues."""
    from lib.session_window import closer_turn

    return closer_turn(rows)


def capitalize_sentences(text: str) -> str:
    """Capitalize the first letter after a sentence end, including a closing quote.

    'questions. you struggle' becomes 'questions. You struggle'.
    'form.". integrate' becomes 'form." Integrate'. A stray '.".' collapses to '."'.
    """
    cleaned = re.sub(r'([.!?])"\.', r'\1"', text or '')
    cleaned = re.sub(r'([.!?])”\.', r'\1”', cleaned)

    def _upper(match: re.Match) -> str:
        return match.group(1) + match.group(2).upper()

    return re.sub(r'((?:[.!?]["”]?)\s+)([a-z])', _upper, cleaned)


def lead_sentence(text: str) -> str:
    """A section opens with a capital. 'you advocate' becomes 'You advocate'."""
    text = capitalize_sentences(text)
    stripped = text.lstrip()
    if stripped[:1].islower():
        text = text[:len(text) - len(stripped)] + stripped[0].upper() + stripped[1:]
    return text


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
    facilitator: bool = False,
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
    if not facilitator and int(share) > average + 10:
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


EVEN_THIRD = 33


def _habit(kind: str, *, share, speaker_count, questions, question_average, turn_words, session_turn_words, early, late, facilitator=False):
    average = room_average(speaker_count)
    if kind == 'questions':
        target = improve_toward(int(questions), max(int(question_average), int(questions) + 1), increase=True)
        sentence = (
            f'Ask one question before you add another point. '
            f'Target: {count_label(target, "question")}.'
        )
        if int(questions) > int(question_average):
            why = 'Next time, leave one question for someone else to answer.'
        elif int(questions) < int(question_average):
            why = 'Next time, ask one question before you add a point.'
        else:
            why = 'Next time, keep one question open for someone else.'
        return sentence, why
    if kind == 'talk_low':
        target = improve_toward(int(share), average, increase=True)
        sentence = f'Say one more detail before you stop. Target: {target}% of this talk.'
        why = 'Next time, make that handoff one short point.'
        return sentence, why
    if kind == 'turns_short':
        target = improve_toward(int(turn_words), int(session_turn_words), increase=True)
        sentence = f'Name one more detail before the turn ends. Target: {target} words in a turn.'
        why = 'Next time, add one more detail before you stop.'
        return sentence, why
    if kind == 'thirds':
        start = int(early or 0)
        end = int(late or 0)
        # An even slice of the session. Below it, the last third rises. Above it, it falls.
        norm = EVEN_THIRD
        increase = end < norm
        target = improve_toward(end, norm, increase=increase)
        if target < 1:
            target = 1
        if increase and target <= end:
            target = min(norm, end + 1)
        if not increase and target >= end:
            target = max(1, end - 1)
        sentence = f'Wait through the opening before the long story. Target: {target}% in the last third.'
        why = 'Next time, save one point for the last third.'
        return sentence, why
    if kind == 'talk_high':
        target = improve_toward(int(share), average, increase=False)
        sentence = f'Pause after the point and hand the turn on. Target: {target}% of this talk.'
        why = 'Next time, pause after one point so someone else can speak.'
        return sentence, why
    if kind == 'beat':
        if facilitator:
            sentence = 'Pause after a point and hand the turn on. Target: 1 handoff before the next decision.'
            why = 'You are running the room, so the useful change is one handoff before the next decision.'
            return sentence, why
        sentence = 'Pause once before you take the next turn. Target: 1 pause before the second point.'
        why = 'The next change is one pause before the second point.'
        return sentence, why
    target = improve_toward(int(turn_words or 1), int(session_turn_words or 1), increase=False)
    sentence = f'Stop once the point is made. Target: {target} words in a turn.'
    why = 'Next time, stop once the point is made.'
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
    facilitator: bool = False,
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
        facilitator=facilitator,
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
        facilitator=facilitator,
    )


def _runs_agenda(person: dict, people: list[dict]) -> bool:
    """A facilitator, moderator, or host is not scored against an even share.

    In a large room the unique top talker is the person running the agenda.
    """
    role = str(person.get('role') or '').lower()
    if role in {'facilitator', 'moderator', 'host'}:
        return True
    count = int(person.get('speaker_count') or 0)
    if count < 8 or not people:
        return False
    mine = int(person.get('share') or 0)
    top = max(int(other.get('share') or 0) for other in people)
    if mine != top or mine <= room_average(count) + 10:
        return False
    return sum(int(other.get('share') or 0) == top for other in people) == 1


def unique_habits(people: list[dict]) -> dict[str, tuple[str, str]]:
    """One habit sentence per person. The largest real gap is claimed first."""
    def ranked_for(person: dict) -> list[tuple[str, float]]:
        average = person.get('question_average', person.get('session_question_average', 0))
        kinds = ranked_kinds(
            share=person['share'],
            speaker_count=person['speaker_count'],
            questions=person['questions'],
            question_average=average,
            turn_words=person.get('turn_words') or 0,
            session_turn_words=person.get('session_turn_words') or 0,
            early=person.get('early'),
            late=person.get('late'),
            facilitator=_runs_agenda(person, people),
        )
        if not any(kind == 'beat' for kind, _size in kinds):
            kinds.append(('beat', 0))
        return kinds

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
                'facilitator': _runs_agenda(person, people),
            })
            if sentence not in used:
                used.add(sentence)
                assigned[label] = (sentence, why)
                placed = True
                break
        if not placed:
            sentence, why = _habit('beat', **{
                'share': person['share'],
                'speaker_count': person['speaker_count'],
                'questions': person['questions'],
                'question_average': person.get('question_average', person.get('session_question_average', 0)),
                'turn_words': person.get('turn_words') or 0,
                'session_turn_words': person.get('session_turn_words') or 0,
                'early': person.get('early'),
                'late': person.get('late'),
                'facilitator': _runs_agenda(person, people),
            })
            for alternate in (
                sentence,
                'Wait for one answer before you add the next point. Target: 1 pause before the second point.',
                'Hand the turn on after one point. Target: 1 handoff before the next stretch.',
            ):
                if alternate not in used:
                    sentence = alternate
                    break
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
    """Talk share against an even share of 1/N. 'Near' is only ±2 points."""
    average = room_average(speaker_count)
    gap = int(round(int(share) - average))
    if abs(gap) <= 2:
        return 'near the room average'
    times = (int(share) / average) if average else 0
    if times >= 1.5:
        shown = f'{times:.1f}'.rstrip('0').rstrip('.')
        return f'about {shown}× an even share of {average}%'
    if gap > 0:
        return f'{gap} points above an even share of {average}%'
    return f'{abs(gap)} points below an even share of {average}%'


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


def room_median(values) -> int | None:
    """The lower middle value, so an even room never invents a count nobody had."""
    nums = sorted(int(value) for value in values)
    if not nums:
        return None
    return nums[(len(nums) - 1) // 2]


def session_talk_phrase(transcript) -> str:
    """The real length of the window, as in 'this 8-minute talk'."""
    rows = [
        row for row in (transcript or {}).get('utterances') or []
        if str(row.get('text') or '').strip()
    ]
    if not rows:
        return 'this talk'
    starts = [float(row.get('start') or 0) for row in rows]
    ends = [float(row.get('end') or 0) for row in rows]
    span = max(ends) - min(starts)
    if span >= 1000:
        span /= 1000.0
    if span < 30:
        return 'this talk'
    minutes = max(1, int(round(span / 60)))
    return f'this {minutes}-minute talk'


def turns_sentence(count: int, median: int | None = None, words: int | None = None) -> str:
    """The room median is the comparator. One long turn is described by its words."""
    counted = int(count)
    if counted == 1:
        amount = f'about {int(words)} words' if words else 'one stretch'
        return (
            f'You spoke in one long turn of {amount}. '
            'Next time, stop after two sentences and ask who has tried that.'
        )
    middle = counted if median is None else int(median)
    turn_word = 'turn' if counted == 1 else 'turns'
    median_word = 'turn' if middle == 1 else 'turns'
    if counted > middle:
        judge = 'That is more than most people in the room.'
        take = 'Next time, leave one of those turns for someone else.'
    elif counted < middle:
        judge = 'That is less than most people in the room.'
        take = 'Next time, take one more turn before the close.'
    else:
        judge = 'That matches the middle of the room.'
        take = 'Next time, use one of those turns to ask a question.'
    return (
        f'You took {counted} {turn_word}. The room median was {middle} {median_word}. '
        f'{judge} {take}'
    )


def talk_number_line(share: int, speaker_count: int, span: str = 'this talk') -> str:
    """Talk share against an even split of the people in the room."""
    even = room_average(speaker_count)
    gap = int(share) - even
    if abs(gap) <= 2:
        judge = f'That is close to an even split of {span}.'
        take = 'Next time, keep your turn about this long.'
    elif gap > 0:
        judge = f'That is more than an even split of {span}, so this was a lot of the talking.'
        take = 'Next time, stop after one point and let someone else speak.'
    else:
        judge = f'That is less than an even split of {span}.'
        take = 'Next time, add one short point before you stop.'
    return f'You spoke for {int(share)}% of {span}. That is your share of the talking. {judge} {take}'


def question_number_line(count: int, average: int) -> str:
    """Questions against the session's own typical count."""
    asked = int(count)
    typical = int(average)
    if asked > typical:
        judge = 'That is more than the typical person here.'
        take = 'Next time, leave one question for someone else to answer.'
    elif asked < typical:
        judge = 'That is less than the typical person here.'
        take = 'Next time, ask one question before you add a point.'
    else:
        judge = 'That matches the typical person here.'
        take = 'Next time, keep one of those questions open for someone else.'
    asked_word = 'question' if asked == 1 else 'questions'
    typical_word = 'question' if typical == 1 else 'questions'
    if asked == 0:
        meaning = 'That means you did not ask the room anything.'
    elif asked == 1:
        meaning = 'That means you asked the room one thing.'
    else:
        meaning = 'That means you asked the room more than once.'
    return f'You asked {asked} {asked_word}. {meaning} {judge} {take}'


def group_plain_numbers(
    interaction_pct: int,
    speaker_count: int,
    *,
    spoke: int | None = None,
    roster: int | None = None,
) -> list[str]:
    """Each rate in plain words, judged against a number this session can compute."""
    people = max(int(speaker_count or 0), 0)
    moved = int(interaction_pct)
    if people > 1:
        even = int(round(100 * (people - 1) / people))
        if moved + 10 < even:
            judge = (
                'That is low. If the next turn were just as likely to be any other person, '
                'more turns would move. Most of the time the same few people kept trading turns, '
                'so most of the room listened.'
            )
        elif moved >= even:
            judge = 'That is high. Turns moved to a new person as often as an even mix.'
        else:
            judge = 'That is close to an even mix, where any other person is just as likely to speak next.'
    else:
        judge = 'This shows how often the next voice was a new person.'
    handoff = (
        f'{moved}% of turns moved to someone new. {judge} '
        'Next time, after you answer, pass the question to someone who has not spoken.'
    )
    if spoke is None or not roster:
        people_line = 'Next time, invite one quiet person in before you close.'
    else:
        people_line = people_spoke_line(spoke, roster)
    return [handoff, people_line]


_SPOKEN_WORDS = {1: 'One', 2: 'Two', 3: 'Three', 4: 'Four', 5: 'Five', 6: 'Six', 7: 'Seven', 8: 'Eight'}


def people_spoke_line(spoke: int, total: int) -> str:
    """A headcount against the roster. The gini rate is not a count of people."""
    spoke = max(0, int(spoke))
    total = max(spoke, int(total or 0))
    if total <= 0:
        return f'{spoke} people spoke. Next time, invite one quiet person in before you close.'
    silent = total - spoke
    if silent <= 0:
        return (
            f'{spoke} of {total} people spoke. Everyone took a turn. '
            'Next time, keep passing the turn.'
        )
    quiet = _SPOKEN_WORDS.get(silent, str(silent))
    return (
        f'{spoke} of {total} people spoke. {quiet} never did. '
        'Next time, invite one quiet person in before you close.'
    )


def group_pattern_why(portrait: dict, shares=None) -> str:
    """Why the pattern matters. Talk share and back and forth stay distinct."""
    most, least_talk = _talk_ends(shares or [])
    pair = (portrait or {}).get('pair') or ()
    quiet = least_talk or (portrait or {}).get('least') or 'the quiet person'
    if most and least_talk and most != least_talk:
        opening = f'{most} talked the most, so {quiet} had less chance to change the choice.'
    elif len(pair) >= 2 and pair[0] and pair[1]:
        opening = f'Most of the back and forth ran between {pair[0]} and {pair[1]}.'
    else:
        opening = f'{quiet} had less chance to change the choice.'
    return f'{opening} Next time, ask {quiet} before you decide.'


def group_caveat(portrait: dict, shares=None) -> str:
    """One caveat. The person we ask is the one who talked least, not who was least connected."""
    pair = (portrait or {}).get('pair') or ()
    _most, least_talk = _talk_ends(shares or [])
    least = least_talk or (portrait or {}).get('least') or 'the quiet person'
    if len(pair) >= 2 and pair[0] and pair[1]:
        return (
            f'A point can end with {pair[0]} and {pair[1]} before {least} speaks. '
            f'Next time, stop and ask {least}.'
        )
    return f'A point can end before {least} speaks. Next time, stop and ask {least}.'


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


def fit_headline(text: str, fallback: str = 'You made one clear move.') -> str:
    """Six plain words or fewer. A 'focused dialogue' line is not a headline."""
    from lib.readability import reading_grade

    cleaned = re.sub(r'\s+', ' ', str(text or '')).strip().rstrip('.')
    if (
        not cleaned
        or re.search(r'focused dialogue', cleaned, re.I)
        or len(cleaned.split()) > 6
        or reading_grade(cleaned) > 6
    ):
        return fallback
    return cleaned + '.'


def personal_why_line(share: int, speaker_count: int, person: str = '') -> str:
    """Second person, one behavior, and a next step. Not a generic filler."""
    who = person or 'the next person'
    even = room_average(max(int(speaker_count or 1), 1))
    if int(share) > even + 2:
        return f'You held more of this talk than an even split. Next time, hand the next point to {who}.'
    if int(share) + 2 < even:
        return 'You spoke less than an even split of this talk. Next time, add one short point.'
    return f'You made one move the room could hear. Next time, let {who} answer it.'


SHORTHAND = (
    'the room handed the turn',
    'interaction rate',
    'participation',
)
_NARRATIVE_JARGON = re.compile(r'\b(interaction|participation|talk time)\b', re.I)
_NARRATIVE_FILLER = re.compile(r'\bthose rates describe this session\b', re.I)


def _outside_quotes(text: str) -> str:
    """Exact quotes stay in the story. The rules apply to the words around them."""
    return re.sub(r'(?:"[^"]*"|“[^”]*”)', '', text or '')


def narrative_problems(text: str, *, big: bool = False) -> list[str]:
    """A story section may not stack percentages or name a metric, and 1 big thing asks something."""
    problems = []
    prose = _outside_quotes(text or '')
    if _NARRATIVE_FILLER.search(prose):
        problems.append('filler')
    if _NARRATIVE_JARGON.search(prose):
        problems.append('metric jargon')
    lowered = prose.lower()
    if any(phrase in lowered for phrase in SHORTHAND):
        problems.append('shorthand')
    if prose.count('"') % 2:
        problems.append('open quote')
    for span in re.findall(r'"([^"]*)"', text or ''):
        body = span.lstrip('…').strip()
        if body and not body.endswith(('.', '!', '?')):
            problems.append('truncated quote')
        pieces = [part for part in re.split(r'(?<=[.!?])\s+', body) if part.strip()]
        if len(pieces) > 1:
            problems.append('spliced quote')
    used = 0
    for sentence in re.split(r'(?<=[.!?])\s+', (text or '').strip()):
        count = len(re.findall(r'\d+\s*%', _outside_quotes(sentence)))
        used += count
        if count >= 2:
            problems.append('two percentages in one sentence')
            break
    if used > 2 and 'two percentages in one sentence' not in problems:
        problems.append('more than two percentages in the section')
    if big and '?' not in prose:
        problems.append('no question')
    if big:
        problems.extend(big_thing_order_problems(text or ''))
    if re.search(r"\b(I|I'll|I'm|I've|I'd)\b", prose):
        problems.append('unquoted first person')
    return problems


def narrative_text(text: str, *, big: bool = False) -> str:
    """Drop metric dumps. Keep a grounded sentence, and keep one question on 1 big thing."""
    kept = []
    used = 0
    for sentence in re.split(r'(?<=[.!?])\s+', (text or '').strip()):
        if not sentence:
            continue
        outside = _outside_quotes(sentence)
        if _NARRATIVE_JARGON.search(outside) or _NARRATIVE_FILLER.search(outside):
            continue
        if re.search(r"\b(I|I'll|I'm|I've|I'd)\b", outside):
            continue
        count = len(re.findall(r'\d+\s*%', outside))
        if count >= 2 or used + count > 2:
            continue
        used += count
        kept.append(sentence)
    result = ' '.join(kept).strip()
    if big and '?' not in _outside_quotes(result):
        result = (result.rstrip('.') + '. ' if result else '') + closing_question()
    from lib.readability import reading_grade

    if result and reading_grade(result) > 6:
        result = (
            'You made one clear move. ' + closing_question()
            if big
            else 'You can change this next time. Try one small step.'
        )
    return result


def _talk_ends(shares) -> tuple[str, str]:
    rows = [(str(name), int(value)) for name, value in (shares or []) if str(name) != 'Average']
    if not rows:
        return '', ''
    ordered = sorted(rows, key=lambda item: (-item[1], item[0]))
    return ordered[0][0], ordered[-1][0]


def group_character_copy(facts: dict) -> str:
    """One pattern, one moment, one question. Talk most and least come from talk share."""
    portrait = facts.get('_portrait') or {}
    pair = portrait.get('pair') or ()
    most, least_talk = _talk_ends(facts.get('_shares') or [])
    if most and least_talk and most != least_talk:
        pattern = f'{most} talked the most, and {least_talk} the least.'
    elif len(pair) >= 2 and pair[0] and pair[1]:
        pattern = f'The most back and forth ran between {pair[0]} and {pair[1]}.'
    elif portrait.get('least'):
        pattern = f'{portrait.get("least")} had the least back and forth.'
    else:
        pattern = 'Most people gave one long answer and then went quiet.'
    closing = facts.get('_closing') or {}
    name = str(closing.get('name') or '').strip()
    quote = str(closing.get('text') or '').strip().strip('"')
    sentences = [pattern]
    if name and quote:
        sentences.append(f'{name} said "{quote}"')
    elif name:
        sentences.append(f'{name} said the last line.')
    ask = least_talk or portrait.get('least') or 'the person who talked the least'
    sentences.append(f'Who will ask {ask} to speak sooner?')
    return close_quotes(' '.join(sentences))


def recruits_someone_else(text: str, quiet: str, names: list[str]) -> bool:
    """True when a line asks a person to speak who is not the lowest talk share."""
    for name in names:
        if not name or name == quiet:
            continue
        if re.search(rf'\b(?:ask|invite)\s+{re.escape(name)}\b', text or '', re.I):
            return True
        if re.search(rf'\bleave (?:one reply|that reply) for {re.escape(name)}\b', text or '', re.I):
            return True
    return False


def close_quotes(text: str) -> str:
    """An opened quote always gets its closing mark."""
    if str(text).count('"') % 2:
        return f'{text}"'
    return text


_MOMENT_SKIP = {
    'could', 'would', 'should', 'about', 'there', 'their', 'which', 'where', 'while',
    'being', 'other', 'every', 'still', 'those', 'first', 'check', 'after', 'before',
}


def _split_sentences(text: str) -> list[str]:
    """Split on sentence ends that are not inside a quote."""
    quotes: list[str] = []

    def _hold(match: re.Match) -> str:
        quotes.append(match.group(0))
        return f'\x00{len(quotes) - 1}\x00'

    masked = re.sub(r'"[^"]*"', _hold, text or '')
    parts = [part.strip() for part in re.split(r'(?<=[.!?])\s+', masked) if part.strip()]

    def _back(match: re.Match) -> str:
        return quotes[int(match.group(1))]

    restored = [re.sub(r'\x00(\d+)\x00', _back, part) for part in parts]
    refined = []
    for part in restored:
        match = re.match(r'^(.*"[^"]*"[.!?]?)\s+(\S.*)$', part)
        if match:
            refined.append(match.group(1).strip())
            refined.append(match.group(2).strip())
        else:
            refined.append(part)
    return refined


def bounded_quote(text: str, *, max_words: int = 40, max_sentences: int = 2) -> str | None:
    """One or two whole sentences, at most 40 words. A fragment is not a quote."""
    sentences = [part.strip() for part in re.findall(r'[^.!?]*[.!?]', text or '') if part.strip()]
    if not sentences:
        return None
    chosen = []
    words = 0
    for sentence in sentences:
        count = len(re.findall(r"[A-Za-z0-9']+", sentence))
        if count > max_words:
            if chosen:
                break
            continue
        if chosen and (len(chosen) >= max_sentences or words + count > max_words):
            break
        chosen.append(sentence)
        words += count
        if len(chosen) >= max_sentences:
            break
    if not chosen:
        return None
    return ' '.join(chosen)


_VAGUE_ABOUT = {
    'something', 'anything', 'everything', 'determine', 'exploit', 'decide', 'thing',
    'it', 'this', 'that', 'there', 'stuff',
}


def closing_question(word: str = '', person: str = '') -> str:
    """Name a person or a real noun from the moment. Not 'about' plus a bare verb."""
    if person:
        return f'Next time, will you let {person} answer that point?'
    token = (word or '').lower()
    if token and token not in _VAGUE_ABOUT and not _base_verb(token):
        return f'Next time, who should take the {token} point?'
    return 'Next time, who should answer after this point?'


def question_problems(text: str) -> list[str]:
    problems = []
    if re.search(r'who else should speak about', text or '', re.I):
        problems.append('generic question')
    match = re.search(r'\babout\s+([A-Za-z]+)', text or '')
    if match and (match.group(1).lower() in _VAGUE_ABOUT or _base_verb(match.group(1))):
        problems.append('vague about')
    return problems


def _moment_word(sentence: str) -> str:
    spans = re.findall(r'"([^"]+)"', sentence or '')
    words = re.findall(r"[A-Za-z']+", spans[0] if spans else '')
    best = ''
    for word in words:
        token = word.lower()
        if len(token) > 4 and token not in _MOMENT_SKIP and len(token) > len(best):
            best = token
    return best


def personal_big_thing(drama: str) -> str:
    """Pattern, then the quote, then a question about that moment."""
    pattern = ''
    moment = ''
    for sentence in _split_sentences(drama or ''):
        outside = _outside_quotes(sentence)
        if not sentence or 'What it costs you' in sentence:
            continue
        if _NARRATIVE_JARGON.search(outside) or _NARRATIVE_FILLER.search(outside):
            continue
        if '"' in sentence and not moment:
            moment = sentence
            continue
        if not pattern and '?' not in outside:
            pattern = sentence
    if not pattern:
        pattern = 'You made one clear move.'
    if moment:
        spans = re.findall(r'"([^"]*)"', moment)
        if spans:
            short = bounded_quote(spans[0])
            if short:
                moment = re.sub(r'"[^"]*"', f'"{short}"', moment, count=1)
            else:
                moment = ''
    word = _moment_word(moment)
    question = closing_question(word)
    parts = [pattern]
    if moment:
        parts.append(moment)
    parts.append(question)
    return ' '.join(parts)


def big_thing_order_problems(text: str) -> list[str]:
    """1 big thing is the pattern, then the quote, then one question about that moment."""
    problems = []
    sentences = _split_sentences(text or '')
    if not sentences or '?' not in _outside_quotes(sentences[-1]):
        problems.append('question is not last')
    quote_at = next((index for index, sentence in enumerate(sentences) if '"' in sentence), None)
    pattern_at = next(
        (
            index for index, sentence in enumerate(sentences)
            if '"' not in sentence and '?' not in _outside_quotes(sentence)
        ),
        None,
    )
    if quote_at is not None and pattern_at is not None and quote_at < pattern_at:
        problems.append('quote before the pattern')
    if re.search(r'hand off one turn sooner', text or '', re.I):
        problems.append('generic question')
    problems.extend(question_problems(text or ''))
    return problems


def _no_digits(text: str, label: str, allowed: set[str] | None = None) -> str:
    cleaned = (text or '').strip()
    extra = [number for number in re.findall(r'\d+', cleaned) if not allowed or number not in allowed]
    if label == 'cost' and extra:
        def _keep(match: re.Match) -> str:
            number = re.search(r'\d+', match.group(0))
            if number and allowed and number.group(0) in allowed:
                return match.group(0)
            return ''

        cleaned = re.sub(r'\b\d+\s*[A-Za-z%]*', _keep, cleaned)
        return re.sub(r'\s{2,}', ' ', cleaned).strip(' ,;')
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
    confirmed_names: list[str] | None = None,
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
        confirmed_names=confirmed_names,
    )
    return capitalize_sentences(cleaned)


def question_directions(transcript, label: str) -> tuple[int, int]:
    """Questions this speaker asked, and questions the previous turn aimed at them."""
    rows = (transcript or {}).get('utterances') or []
    asked = 0
    received = 0
    for index, row in enumerate(rows):
        count = str(row.get('text') or '').count('?')
        if not count:
            continue
        if row.get('speaker') == label:
            asked += count
        elif index + 1 < len(rows) and rows[index + 1].get('speaker') == label:
            received += count
    return asked, received


_RECEIVED_CLAIM = re.compile(
    r'\b(inquiries|inquiry|questions? you faced|faced a high volume|volume of inquiries|'
    r'directed at you|asked of you|aimed at you)\b',
    re.I,
)


def inverted_question_claim(text: str, asked: int, received: int) -> str | None:
    """A line that says the questions ran the other way."""
    if asked > received and _RECEIVED_CLAIM.search(text or '') and not re.search(r'\byou asked\b', text or '', re.I):
        return 'challenge says questions were aimed at you, but you asked more than you received'
    if received > asked and re.search(r'\byou asked\b', text or '', re.I) and not _RECEIVED_CLAIM.search(text or ''):
        return 'challenge says you asked the questions, but more were aimed at you'
    return None


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
    asked, received = question_directions(transcript, label)
    if not asked:
        asked = int(questions)
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
        'questions_asked': {
            'value': int(asked),
            'unit': 'questions you asked',
            'line': f'You asked {int(asked)} questions.',
        },
        'questions_received': {
            'value': int(received),
            'unit': 'questions directed at you',
            'line': f'Other people directed {int(received)} questions at you.',
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
        'questions_asked': int(asked),
        'questions_received': int(received),
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


_FACT_ALIASES = {
    'talk': 'talk_share',
    'talk_time': 'talk_share',
    'speaking': 'talk_share',
    'speaking_time': 'talk_share',
    'share': 'talk_share',
    'question': 'questions',
    'question_count': 'questions',
    'questions_asked': 'questions',
    'turn': 'turns',
    'turn_count': 'turns',
    'turns_taken': 'turns',
    'interaction_rate': 'interaction',
    'participation_rate': 'participation',
}


def _fact_keys(facts: dict) -> list[str]:
    return [
        key
        for key, item in facts.items()
        if isinstance(item, dict) and 'line' in item and not str(key).startswith(('target_', '_'))
    ]


def _map_fact_key(key, facts: dict) -> str | None:
    if key in facts and isinstance(facts[key], dict) and 'line' in facts[key]:
        return str(key)
    token = re.sub(r'[^a-z0-9]+', '_', str(key or '').lower()).strip('_')
    if token in facts and isinstance(facts[token], dict) and 'line' in facts[token]:
        return token
    alias = _FACT_ALIASES.get(token)
    if alias and alias in facts:
        return alias
    for fact_key in _fact_keys(facts):
        if token and (token in fact_key or fact_key in token):
            return fact_key
    return None


def render_numbers(keys: list, facts: dict) -> list[str]:
    import logging

    logging.warning('number_keys raw %r', keys)
    available = _fact_keys(facts)
    if not isinstance(keys, list) or not keys:
        chosen = available[:3]
    else:
        chosen = []
        for key in keys[:3]:
            mapped = _map_fact_key(key, facts)
            if mapped and mapped not in chosen and not mapped.startswith('target_'):
                chosen.append(mapped)
        if not chosen:
            chosen = available[:3]
    if not chosen:
        raise OutputParserException('number_keys must be one to three fact keys')
    return [facts[key]['line'] for key in chosen]


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
        _most, least_talk = _talk_ends(facts.get('_shares') or [])
        least = least_talk or (facts.get('_portrait') or {}).get('least') or 'the person who talked the least'
        sentence = f'Ask {least} to speak before the next choice.'
        why = f'Next time, leave one reply for {least}.'
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


def render_drama(
    data: dict,
    transcript,
    label: str,
    corpus: str,
    forbidden: list[str] | None,
    facts: dict | None = None,
) -> str:
    patterns = data.get('patterns')
    if not isinstance(patterns, list) or not patterns or len(patterns) > 2:
        raise OutputParserException('drama needs one or two patterns')
    sentences = []
    mights = 0
    seen_quotes: set[str] = set()
    # Distinct inside Drama. A turn already used in Goal must not make the
    # counter-moment fail when this speaker still has another line.
    seen_turns: set[str] = set()
    allowed_numbers = {
        str(item['value'])
        for item in (facts or {}).values()
        if isinstance(item, dict) and 'value' in item
    }
    for row in (transcript or {}).get('utterances') or []:
        if row.get('speaker') != label:
            continue
        allowed_numbers.update(re.findall(r'\d+', str(row.get('text') or '')))
    for pattern in patterns:
        quotes = pattern.get('quotes') if isinstance(pattern, dict) else None
        if not isinstance(quotes, list) or not quotes or len(quotes) > 3:
            raise OutputParserException('each pattern needs 1 to 3 quotes')
        rendered_quotes = []
        for item in quotes:
            when, text, turn_id = _drama_quote(
                str(item.get('text') or ''), str(item.get('ts') or ''), transcript, label, seen_turns
            )
            rendered_quotes.append((when, text))
            seen_quotes.add(_normalize_quote(text))
            seen_turns.add(turn_id)
        pattern_text = _prose(
            str(pattern.get('pattern') or ''), corpus, 'pattern',
            forbidden_names=forbidden, transcript=transcript, speaker_label=label,
        )
        twist = _prose(
            str(pattern.get('twist') or ''), corpus, 'twist',
            forbidden_names=forbidden, allowed_numbers=allowed_numbers,
            transcript=transcript, speaker_label=label,
        )
        cost = _prose(
            str(pattern.get('cost') or ''), corpus, 'cost',
            forbidden_names=forbidden, allowed_numbers=allowed_numbers,
            transcript=transcript, speaker_label=label,
        )
        gift = _prose(
            str(pattern.get('gift') or ''), corpus, 'gift',
            forbidden_names=forbidden, allowed_numbers=allowed_numbers,
            transcript=transcript, speaker_label=label,
        )
        when, said = rendered_quotes[0]
        sentences.append(f'At {when} you said "{said}".')
        for extra_when, extra_said in rendered_quotes[1:]:
            sentences.append(f'At {extra_when} you said "{extra_said}".')
        pattern_done = pattern_text if pattern_text.endswith('.') else pattern_text + '.'
        sentences.extend(re.split(r'(?<=[.!?])\s+', pattern_done))
        sentences.append(f"Here's what's interesting: {twist.rstrip('.')}.")
        gets, costs = _talk_sides(gift, cost)
        sentences.append(f"It's doing a job for you. {gets} {costs}")
        mights += len(re.findall(r'\bmight\b', ' '.join(sentences).lower()))
    exception = data.get('exception') if isinstance(data.get('exception'), dict) else None
    one_turn = len(_speaker_turns(transcript, label)) < 2
    if not one_turn:
        if not exception:
            raise OutputParserException('drama needs an exception')
        exception_ts, exception_quote, exception_turn = _drama_counter(
            str(exception.get('quote') or ''),
            str(exception.get('ts') or ''),
            transcript,
            label,
            seen_turns,
        )
        happened = _prose(
            str(exception.get('what_happened') or ''), corpus, 'what_happened',
            forbidden_names=forbidden, transcript=transcript, speaker_label=label,
        )
        sentences.append(
            f"The one moment it didn't happen was {exception_ts}, "
            f'when you said "{exception_quote}", and that\'s when {happened.rstrip(".")}.'
        )
    kept = []
    for sentence in sentences:
        if '—' in sentence or re.search(r'\s--\s', sentence):
            continue
        if re.search(r'\bhere is the repeated move\b', sentence, re.I):
            continue
        kept.append(sentence)
    rendered = ' '.join(kept)
    if len(re.findall(r'\bmight\b', rendered.lower())) > 1:
        raise OutputParserException('drama uses might more than once')
    for phrase in BAN_PHRASES:
        if phrase in rendered.lower():
            raise OutputParserException(f'drama contains {phrase!r}')
    log = _quote_log(facts)
    if log is not None:
        for turn_id in seen_turns:
            if turn_id not in log:
                log.append(turn_id)
    return capitalize_sentences(rendered)


def _speaker_turns(transcript, label: str) -> list[tuple[int, str, str, str]]:
    """(word count, clock, text, turn id) for this speaker's turns of at least 3 words."""
    from lib.story_evidence import clock

    found = []
    for index, row in enumerate((transcript or {}).get('utterances') or []):
        if row.get('speaker') != label:
            continue
        text = str(row.get('text') or '').strip()
        words = _word_count(text)
        if words < 3:
            continue
        turn_id = str(row.get('start') if row.get('start') is not None else index)
        found.append((words, clock(row.get('start')), text, turn_id))
    return found


def _quote_floor(transcript, label: str) -> int:
    """A short speaker may be quoted from 3 words. Everyone else still needs 6."""
    turns = _speaker_turns(transcript, label)
    if turns and max(item[0] for item in turns) < 6:
        return 3
    return 6


def _next_distinct_turn(transcript, label: str, used: set[str]) -> tuple[str, str, str] | None:
    """The next unused turn of at least 3 words. A full sentence wins, then any short line."""
    sentences = []
    shorts = []
    for _words, when, text, turn_id in _speaker_turns(transcript, label):
        if turn_id in used:
            continue
        clip = closing_clip(text)
        if clip is not None:
            sentences.append((when, clip, turn_id))
        elif _word_count(text) >= 3:
            shorts.append((when, text, turn_id))
    pool = sentences or shorts
    if not pool:
        return None
    return pool[0]


def _quote_log(facts: dict | None) -> list | None:
    log = (facts or {}).get('_quote_log')
    return log if isinstance(log, list) else None


def _fresh_quote(text: str, timestamp: str, transcript, label: str, facts: dict) -> tuple[str, str]:
    """A quote whose turn has not already been cited in this report."""
    log = _quote_log(facts)
    used = set(log or [])
    shown, turn_id = own_quote_turn(text, timestamp, transcript, label)
    if turn_id in used:
        replacement = _next_distinct_turn(transcript, label, used)
        if replacement is not None:
            when, shown, turn_id = replacement
        else:
            when = str(timestamp or '').strip()
    else:
        when = str(timestamp or '').strip()
    if log is not None and turn_id not in log:
        log.append(turn_id)
    return when, shown


def _drama_quote(text: str, timestamp: str, transcript, label: str, used: set[str]) -> tuple[str, str, str]:
    """A quote of at least the speaker's floor. A too-short line uses their longest turn."""
    from lib.story_evidence import clock

    floor = _quote_floor(transcript, label)
    try:
        shown, turn_id = own_quote_turn(text, timestamp, transcript, label)
    except OutputParserException:
        shown, turn_id = '', ''
    if shown and turn_id not in used and _word_count(shown) >= floor:
        return str(timestamp or '').strip() or clock(None), shown, turn_id
    replacement = _next_distinct_turn(transcript, label, used)
    if replacement is not None:
        return replacement
    if shown and _word_count(shown) >= 3:
        return str(timestamp or '').strip() or clock(None), shown, turn_id
    raise OutputParserException('drama quote is under 3 words')


def _drama_counter(
    text: str, timestamp: str, transcript, label: str, used: set[str]
) -> tuple[str, str, str]:
    """Same-turn counter-moments are replaced before the quote is validated."""
    from lib.story_evidence import clock

    shown, turn_id = '', ''
    try:
        shown, turn_id = own_quote_turn(text, timestamp, transcript, label)
    except OutputParserException:
        shown, turn_id = '', ''
    if shown and turn_id not in used and _word_count(shown) >= _quote_floor(transcript, label):
        return str(timestamp or '').strip() or clock(None), shown, turn_id
    replacement = _next_distinct_turn(transcript, label, used)
    if replacement is None and turn_id:
        replacement = _next_distinct_turn(transcript, label, {turn_id})
    if replacement is not None:
        return replacement
    if _quote_floor(transcript, label) == 3 and shown and _word_count(shown) >= 3:
        return str(timestamp or '').strip() or clock(None), shown, turn_id
    if turn_id and turn_id in used:
        raise OutputParserException('drama counter-moment must come from a different turn')
    raise OutputParserException('drama quote is under 3 words')


def _normalize_quote(text: str) -> str:
    return re.sub(r'[^a-z0-9]+', '', (text or '').lower())


_ROLE_WORDS = {
    'adviser', 'advisor', 'analyst', 'architect', 'attorney', 'author', 'coach',
    'consultant', 'counselor', 'counsellor', 'designer', 'developer', 'director',
    'doctor', 'engineer', 'entrepreneur', 'executive', 'facilitator', 'founder',
    'lawyer', 'manager', 'mentor', 'moderator', 'officer', 'president',
    'professor', 'psychologist', 'researcher', 'scientist', 'specialist',
    'teacher', 'therapist', 'trainer',
}


def _role_identity(phrase: str) -> bool:
    """Role phrases such as 'research engineer'. 'knowledge management' is not one."""
    return any(word in _ROLE_WORDS for word in phrase.lower().split())


def foreign_phrase(text: str, transcript, label: str) -> str | None:
    """A role or identity phrase this speaker never said.

    A common noun phrase such as 'knowledge management' is fine when this speaker
    says it, even if someone else says it too. Only a multi-word role the speaker
    never used is a leak.
    """
    own = ' '.join(
        str(row.get('text') or '')
        for row in (transcript or {}).get('utterances') or []
        if row.get('speaker') == label
    ).lower()
    others = [
        str(row.get('text') or '').lower()
        for row in (transcript or {}).get('utterances') or []
        if row.get('speaker') != label
    ]

    def leaked(phrase: str) -> bool:
        if not _role_identity(phrase):
            return False
        if phrase in own:
            return False
        return any(phrase in other for other in others)

    content = [word for word in re.findall(r"[A-Za-z']+", text or '') if len(word) >= 4]
    for index in range(len(content) - 1):
        pair = f'{content[index]} {content[index + 1]}'.lower()
        if leaked(pair):
            return pair
    words = re.findall(r"[A-Za-z']+", text or '')
    for size in (6, 5, 4):
        if len(words) < size:
            continue
        for index in range(len(words) - size + 1):
            window = words[index:index + size]
            if not any(len(word) >= 4 for word in window):
                continue
            phrase = ' '.join(window).lower()
            if leaked(phrase):
                return phrase
    return None


def render_personal_section(section: str, data: dict, *, transcript, label: str, facts: dict, forbidden: list[str]) -> dict:
    corpus = corpus_text(transcript, label)
    if section == 'character':
        headline = _prose(
            str(data.get('headline') or ''), corpus, 'headline',
            forbidden_names=forbidden, transcript=transcript, speaker_label=label,
        )
        if headline[:1].islower():
            headline = headline[0].upper() + headline[1:]
        if word_count(headline) > 14:
            raise OutputParserException('headline is over 14 words')
        leaked = foreign_phrase(headline, transcript, label)
        if leaked:
            raise OutputParserException(f'headline uses another speaker\'s line: {leaked}')
        return {'headline': headline, 'numbers': render_numbers(data.get('number_keys'), facts)}
    if section == 'goal':
        claim = _prose(
            str(data.get('claim') or ''), corpus, 'claim',
            forbidden_names=forbidden, transcript=transcript, speaker_label=label,
        )
        quote = data.get('quote') if isinstance(data.get('quote'), dict) else {}
        when, text = _fresh_quote(
            str(quote.get('text') or ''), str(quote.get('ts') or ''), transcript, label, facts
        )
        sentence = goal_line(claim, personal=True).rstrip('.')
        return {'goal': capitalize_sentences(f'{sentence}. At {when} you said "{text}".')}
    if section == 'challenge':
        claim = _prose(
            str(data.get('claim') or ''), corpus, 'claim',
            forbidden_names=forbidden, transcript=transcript, speaker_label=label,
        )
        key = str(data.get('fact_key') or '')
        if key not in facts:
            raise OutputParserException(f'challenge fact_key is not allowed: {key}')
        asked = int((facts.get('_meta') or {}).get('questions_asked') or 0)
        received = int((facts.get('_meta') or {}).get('questions_received') or 0)
        inverted = inverted_question_claim(claim, asked, received)
        if inverted:
            raise OutputParserException(inverted)
        line = str(facts[key]['line']).rstrip('.')
        claim_text = claim.rstrip('.')
        if narrative_problems(line + '.'):
            text = capitalize_sentences(f'{claim_text}.')
        else:
            text = capitalize_sentences(f'{line}. {claim_text}.')
        problems = [item for item in narrative_problems(text) if item != 'no question']
        if problems:
            raise OutputParserException(f'challenge {problems[0]}')
        return {'challenge': text}
    if section == 'drama':
        return {'drama': render_drama(data, transcript, label, corpus, forbidden, facts)}
    if section == 'resolution':
        claim = _prose(
            str(data.get('claim') or ''), corpus, 'claim',
            forbidden_names=forbidden, transcript=transcript, speaker_label=label,
        )
        quote = data.get('quote') if isinstance(data.get('quote'), dict) else {}
        when, text = _fresh_quote(
            str(quote.get('text') or ''), str(quote.get('ts') or ''), transcript, label, facts
        )
        happened = _what_happened(claim)
        return {'resolution': capitalize_sentences(
            f'{happened.rstrip(".")}. At {when} you said "{text}".'
        )}
    if section == 'one_thing':
        sentence, why = render_one_thing(
            data, facts, corpus, forbidden, transcript=transcript, label=label
        )
        return {'one_thing': sentence, 'why': why}
    raise OutputParserException(f'unknown section {section}')


def _room_headline(candidate: str, facts: dict) -> str:
    """Keep a room headline. A personal, numbered, or over-long one is replaced, not fatal."""
    interaction = int((facts.get('interaction') or {}).get('value') or 0)
    fallback = (
        'The turns stayed with a few voices'
        if interaction < 50
        else 'The room kept handing the turn across'
    )
    try:
        headline = _prose(candidate, '', 'headline')
    except OutputParserException:
        return fallback
    if word_count(headline) > 12 or group_headline_is_personal(headline):
        return fallback
    return headline


def render_group_section(section: str, data: dict, *, transcript, facts: dict, cast: str) -> dict:
    corpus = corpus_text(transcript, None)
    if section == 'character':
        if 'interaction' not in facts and 'participation' not in facts:
            raise OutputParserException('group character needs at least one rate fact')
        # The paragraph is computed from the facts. A personal or over-long headline
        # used to reject the whole section on every run, so the report fell back.
        headline = _room_headline(str(data.get('headline') or ''), facts)
        return {
            'headline': headline,
            'character': capitalize_sentences(group_character_copy(facts)),
        }
    if section in ('goal', 'resolution'):
        claim = _prose(
            _strip_stated_by(str(data.get('claim') or '')),
            corpus,
            section,
            speaker_names=list(facts.get('_names') or []) or None,
            confirmed_names=list(facts.get('_names') or []) or None,
        )
        closing_row = last_substantive_utterance((transcript or {}).get('utterances') or [])
        closing = clip_clause(str(closing_row.get('text') or ''), 18) if closing_row else ''
        if section == 'goal':
            if re.search(r"\b[A-Z][a-z]+(?:'s|’s)\b", claim) or re.search(r'\bas stated by\b', claim, re.I):
                raise OutputParserException('group goal must be the room\'s topic, not one person\'s theme')
            return {'goal': goal_line(claim, personal=False)}
        from lib.session_window import closer_turn

        closing_fact = facts.get('_closing') or {}
        if not str(closing_fact.get('text') or '').strip():
            closer = closer_turn((transcript or {}).get('utterances') or [])
            if closer:
                closing_fact = {
                    'name': closer.get('name') or 'the last voice',
                    'text': closer.get('text') or '',
                }
                closing = ''
        speaker = closing_fact.get('name') or 'the last voice'
        spoken = strongest_sentence(_strip_stated_by(closing_fact.get('text') or closing)) or ''
        if spoken and _word_count(spoken) < 6:
            spoken = ''
        claim = _strip_stated_by(claim)
        if re.search(r'\bthe session begins with\b', claim, re.I):
            raise OutputParserException("resolution must not say 'The session begins with'")
        if re.search(r'\bthe opening segment\b', claim, re.I):
            raise OutputParserException("resolution must not call the ending 'the opening segment'")
        from lib.story_validation import unconfirmed_personal_name

        stray = unconfirmed_personal_name(claim, list(facts.get('_names') or []))
        if stray:
            raise OutputParserException(f'resolution names {stray}, who is not a confirmed speaker')
        follow = _cap_sentence(claim.rstrip('.'))
        if spoken:
            landed = f'The meeting closed on {speaker}, who said "{spoken}". '
        else:
            landed = ''
        resolution = capitalize_sentences(re.sub(r'\s*concluding with\s+"[^"]*"\.?', '', f'{landed}{follow}.', flags=re.I))
        return {'resolution': resolution}
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
        claim_text = claim.rstrip('.')
        text = capitalize_sentences(
            f'{claim_text}. A turn could finish without an answer from across the room.'
        )
        problems = [item for item in narrative_problems(text) if item != 'no question']
        if problems:
            raise OutputParserException(f'group challenge {problems[0]}')
        return {'challenge': text}
    if section == 'drama':
        note = _prose(str(data.get('pattern_note') or ''), corpus, 'pattern_note')
        if note.lower().strip('.') == 'who talks to whom, and who nobody answered':
            return {'drama': cast or note}
        rendered = f'{cast} {note}'.strip() if cast else note
        return {'drama': _complete_clause(rendered)}
    if section == 'one_thing':
        sentence, why = render_one_thing(data, facts, corpus, None)
        return {'one_thing': sentence, 'why': why}
    raise OutputParserException(f'unknown section {section}')
