"""Four personal metrics, ordered by how far they sit from this meeting.

Surprise is the distance from an in-meeting comparator, divided by the spread
of that same number across the room:

    surprise = |value - comparator| / spread

spread is max(room) - min(room). When the spread is 0, surprise is 0.
Talk share uses an even share, 100 / people, as the comparator.
Words, questions, longest turn, and ideas use the room median. With an even
count of people the median is the upper middle (sorted[n // 2]).

Longest turn is always one of the four shown. The other three are the most
surprising of words, talk share, questions, and ideas. A tie keeps the earlier
of: talk share, words, questions, ideas, longest turn. The four are then
shown most surprising first, with that same tie break.
"""

from __future__ import annotations


TIE_ORDER = ('share', 'words', 'questions', 'ideas', 'longest')
LABELS = {
    'share': 'Talk share',
    'words': 'Words',
    'questions': 'Questions',
    'longest': 'Longest turn',
    'ideas': 'Ideas',
}


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
        amount = int(value or 0)
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
    """The four metrics for the box. Longest turn is always included."""
    name = str(story.get('name') or '')
    shares = list(story.get('roster_shares') or [])
    if not shares and story.get('share') is not None:
        shares = [(name or 'You', story.get('share'))]
    share, share_room = _column(shares, name)
    people = max(1, len(share_room) or int(story.get('speaker_count') or 1))
    even = int(round(100 / people))
    words, word_room = _column(story.get('word_counts') or [], name)
    if not word_room and story.get('words') is not None:
        words = int(story.get('words') or 0)
        word_room = [words]
    questions, question_room = _column(story.get('question_counts') or [], name)
    longest, turn_room = _column(story.get('longest_turns') or [], name)
    ideas, idea_room = _column(story.get('proposal_counts') or [], name)

    def item(key: str, value: int, comparator: int, room: list[int], number: str) -> dict:
        return {
            'key': key,
            'label': LABELS[key],
            'value': int(value),
            'number': number,
            'grade': _grade(int(value), int(comparator), 'share' if key == 'share' else 'count'),
            'surprise': round(_surprise(value, comparator, room), 4),
        }

    metrics = {
        'share': item('share', share, even, share_room or [share], f'{int(share)}%'),
        'words': item('words', words, _median(word_room), word_room or [words], str(int(words))),
        'questions': item('questions', questions, _median(question_room), question_room or [questions], str(int(questions))),
        'longest': item('longest', longest, _median(turn_room), turn_room or [longest], str(int(longest))),
        'ideas': item('ideas', ideas, _median(idea_room), idea_room or [ideas], str(int(ideas))),
    }
    optional = ['share', 'words', 'questions', 'ideas']
    optional.sort(key=lambda key: (-metrics[key]['surprise'], TIE_ORDER.index(key)))
    chosen = ['longest'] + optional[:3]
    chosen.sort(key=lambda key: (-metrics[key]['surprise'], TIE_ORDER.index(key)))
    return [metrics[key] for key in chosen]
