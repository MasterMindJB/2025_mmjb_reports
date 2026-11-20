"""Four personal metrics, ordered by how far they sit from this meeting.

Surprise is the distance from an in-meeting comparator, divided by the spread
of that same number across the room:

    surprise = |value - comparator| / spread

spread is max(room) - min(room). When the spread is 0, surprise is 0.
Talk share uses an even share, 100 / people, as the comparator.
Words, questions, longest turn, and ideas use the room median. With an even
count of people the median is the upper middle (sorted[n // 2]).

The box always leads with Talk time, then Longest turn. The other two slots
are the most surprising of questions, words, and ideas. Words is left out when
it is the same stretch as the longest turn, which is what a single turn is.
A tie in those two slots keeps the earlier of questions, words, ideas.
"""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP


TIE_ORDER = ('share', 'words', 'questions', 'ideas', 'longest')
LABELS = {
    'share': 'Talk time',
    'words': 'Words',
    'questions': 'Questions',
    'longest': 'Longest turn',
    'ideas': 'Ideas',
}


def format_duration(milliseconds) -> str:
    """Speaking time as m:ss. Half seconds round up."""
    seconds = round_half_up((float(milliseconds) or 0) / 1000)
    seconds = max(0, seconds)
    minutes, secs = divmod(seconds, 60)
    return f'{minutes}:{secs:02d}'


def round_half_up(value) -> int:
    """One rounding rule for the box and the charts. 16.5 becomes 17."""
    try:
        number = Decimal(str(value))
    except Exception:
        return 0
    return int(number.quantize(Decimal('1'), rounding=ROUND_HALF_UP))


def _median(values: list[int]) -> int:
    if not values:
        return 0
    ordered = sorted(int(value) for value in values)
    return ordered[len(ordered) // 2]


def _surprise(value: int, comparator: int, room: list[int]) -> float:
    if not room:
        return 0.0
    spread = max(room) - min(room)
    if spread <= 0:
        return 0.0
    return abs(int(value) - int(comparator)) / spread


def _column(rows, name: str) -> tuple[int, list[int]]:
    """This person's number, and every person's number, matched on the first name too."""
    wanted = str(name or '').strip().casefold()
    first = wanted.split()[0] if wanted else ''
    you = 0
    found = False
    room = []
    for person, value in rows or []:
        if str(person) == 'Average':
            continue
        amount = round_half_up(value or 0)
        room.append(amount)
        label = str(person).strip().casefold()
        if label == wanted or (first and label.split()[0] == first):
            you = amount
            found = True
    if not found and room:
        you = 0
    return you, room


def _grade(value: int, comparator: int, kind: str) -> str:
    if kind == 'share':
        if abs(value - comparator) <= 2:
            standing = 'Near'
        elif value > comparator:
            standing = 'Above'
        else:
            standing = 'Below'
        return f'{standing} even share ({comparator}%)'
    if value == comparator:
        standing = 'Same as'
    elif value > comparator:
        standing = 'Above'
    else:
        standing = 'Below'
    return f'{standing} the room middle ({comparator})'


def personal_metrics(story: dict) -> list[dict]:
    """Talk time, longest turn, then the two most surprising of the rest."""
    name = str(story.get('name') or '')
    shares = list(story.get('roster_shares') or [])
    if not shares and story.get('share') is not None:
        shares = [(name or 'You', story.get('share'))]
    share, share_room = _column(shares, name)
    people = max(1, len(share_room) or int(story.get('speaker_count') or 1))
    even = round_half_up(Decimal(100) / Decimal(people))
    words, word_room = _column(story.get('word_counts') or [], name)
    if not word_room and story.get('words') is not None:
        words = int(story.get('words') or 0)
        word_room = [words]
    questions, question_room = _column(story.get('question_counts') or [], name)
    longest_words, _turn_words = _column(story.get('longest_turns') or [], name)
    ideas, idea_room = _column(story.get('proposal_counts') or [], name)
    talk_ms, talk_room = _column(story.get('talk_ms') or [], name)
    longest_ms, longest_room = _column(story.get('longest_ms') or [], name)
    if abs(share - even) <= 2:
        standing = 'near'
    elif share > even:
        standing = 'above'
    else:
        standing = 'below'
    talk_grade = f'{int(share)}% of the talk, {standing} an even share ({even}%)'
    middle = _median(longest_room or [longest_ms])
    if longest_ms == middle:
        turn_standing = 'Same as'
    elif longest_ms > middle:
        turn_standing = 'Above'
    else:
        turn_standing = 'Below'
    turn_grade = f'{turn_standing} the room middle ({format_duration(middle)})'

    def item(key: str, value: int, comparator: int, room: list[int], number: str, grade: str | None = None) -> dict:
        return {
            'key': key,
            'label': LABELS[key],
            'value': int(value),
            'number': number,
            'grade': grade or _grade(int(value), int(comparator), 'count'),
            'surprise': round(_surprise(value, comparator, room), 4),
        }

    metrics = {
        'share': item('share', talk_ms, 0, share_room or [share], format_duration(talk_ms), talk_grade),
        'words': item('words', words, _median(word_room), word_room or [words], str(int(words))),
        'questions': item('questions', questions, _median(question_room), question_room or [questions], str(int(questions))),
        'longest': item('longest', longest_ms, middle, longest_room or [longest_ms], format_duration(longest_ms), turn_grade),
        'ideas': item('ideas', ideas, _median(idea_room), idea_room or [ideas], str(int(ideas))),
    }
    # Surprise for talk time still uses the share gap, so a lopsided share can
    # win a later slot if we ever log it. The first two slots are fixed.
    metrics['share']['surprise'] = round(_surprise(share, even, share_room or [share]), 4)
    optional = ['questions', 'words', 'ideas']
    single_turn = int(story.get('turns') or 0) == 1 or (
        longest_words and words == longest_words
    )
    if single_turn:
        optional = [key for key in optional if key != 'words']
    optional.sort(key=lambda key: (-metrics[key]['surprise'], TIE_ORDER.index(key)))
    chosen = ['share', 'longest'] + optional[:2]
    return [metrics[key] for key in chosen]
