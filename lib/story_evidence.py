"""Counts, quotes, and next-speaker lines the story is allowed to use.

The model receives this packet. A number that is not in it is an invented number.
"""

from __future__ import annotations

import re

from lib.names import display_name


def format_clock_seconds(seconds: int) -> str:
    """m:ss, or h:mm:ss once the clock reaches 60 minutes."""
    seconds = max(0, int(seconds))
    minutes, secs = divmod(seconds, 60)
    if minutes >= 60:
        hours, minutes = divmod(minutes, 60)
        return f'{hours}:{minutes:02d}:{secs:02d}'
    return f'{minutes}:{secs:02d}'


def clock(milliseconds) -> str:
    """Nearest whole second, so a turn at 37.6s is shown as 0:38, not 0:37."""
    try:
        seconds = max(0, int(round(float(milliseconds) / 1000.0)))
    except (TypeError, ValueError):
        return '0:00'
    return format_clock_seconds(seconds)


def _rows(transcript: dict | None) -> list[dict]:
    rows = []
    for utterance in (transcript or {}).get('utterances') or []:
        text = str(utterance.get('text') or '').strip()
        if text:
            rows.append(utterance)
    return rows


def _duration(utterance: dict) -> float:
    return max(0.0, float(utterance.get('end') or 0) - float(utterance.get('start') or 0))


def session_thirds(transcript: dict | None) -> list[list[dict]]:
    """Split the session into three time buckets by the midpoint of each turn."""
    rows = _rows(transcript)
    buckets: list[list[dict]] = [[], [], []]
    if not rows:
        return buckets
    start = min(float(row.get('start') or 0) for row in rows)
    end = max(float(row.get('end') or 0) for row in rows)
    span = max(end - start, 1.0)
    for row in rows:
        mid = (float(row.get('start') or 0) + float(row.get('end') or 0)) / 2
        index = int((mid - start) / span * 3)
        buckets[min(2, max(0, index))].append(row)
    return buckets


def talk_share(rows: list[dict], label: str) -> int:
    total = sum(_duration(row) for row in rows)
    if total <= 0:
        return 0
    mine = sum(_duration(row) for row in rows if row.get('speaker') == label)
    return int(round(100 * mine / total))


def quote_events(transcript: dict | None, label: str) -> list[dict]:
    """Each of this person's lines, plus what the next speaker said. No names."""
    rows = _rows(transcript)
    events = []
    for index, row in enumerate(rows):
        if row.get('speaker') != label:
            continue
        nxt = rows[index + 1] if index + 1 < len(rows) else None
        events.append(
            {
                'time': clock(row.get('start')),
                'text': str(row.get('text') or '').strip(),
                'next_time': clock(nxt.get('start')) if nxt else '',
                'next_text': str(nxt.get('text') or '').strip() if nxt else '',
            }
        )
    return events


def before_after(transcript: dict | None, label: str) -> str:
    """A before/after sentence only when the first and last thirds differ."""
    buckets = session_thirds(transcript)
    if not buckets[0] or not buckets[2]:
        return ''
    early = talk_share(buckets[0], label)
    late = talk_share(buckets[2], label)
    if early == late:
        return ''
    return f'{early}% in the first third, then {late}% in the last third.'


def halfway_target(current: int, ideal: int) -> int:
    """About halfway from the current rate to the ideal. Rounded to a whole number."""
    return int(round((int(current) + int(ideal)) / 2))


def numbers_in(text: str) -> set[str]:
    return set(re.findall(r'\d+', text or ''))


def personal_evidence(
    *,
    transcript: dict | None,
    label: str,
    name: str,
    share: int,
    turns: int,
    questions: int,
    session_turns: int,
    session_question_average: int,
) -> tuple[str, set[str], int]:
    events = quote_events(transcript, label)
    early_late = before_after(transcript, label)
    rest = max(0, 100 - int(share))
    target = halfway_target(int(questions), 3 if int(questions) < 3 else int(questions))
    lines = [
        f'NAME: {display_name(name)}',
        f'TALK_SHARE: {int(share)}',
        f'REST_OF_ROOM_SHARE: {rest}',
        f'TURN_COUNT: {int(turns)}',
        f'SESSION_TURNS: {int(session_turns)}',
        f'QUESTION_COUNT: {int(questions)}',
        f'SESSION_QUESTION_AVERAGE: {int(session_question_average)}',
        f'TARGET: {target}',
    ]
    if early_late:
        lines.append(f'BEFORE_AFTER: {early_late}')
    for event in events:
        lines.append(f'QUOTE: {event["time"]} "{event["text"]}"')
        if event['next_text']:
            lines.append(f'NEXT: {event["next_time"]} "{event["next_text"]}"')
    text = '\n'.join(lines)
    return text, numbers_in(text), target


def cast_line(names_in_order: list[str], directed: list[list[int]]) -> str:
    """One line: who talks to whom, and who nobody answered."""
    from lib.chord_diagram import pair_directions

    if not names_in_order:
        return 'Who talks to whom, and who nobody answered.'
    incoming = [0 for _ in names_in_order]
    for source, row in enumerate(directed):
        for target, count in enumerate(row):
            if source != target:
                incoming[target] += count
    unanswered = [names_in_order[index] for index, count in enumerate(incoming) if count == 0]
    pairs = pair_directions(names_in_order, directed)
    if not pairs:
        return 'Who talks to whom, and who nobody answered.'
    left, right, _forward, _back, total = pairs[0]
    if unanswered:
        quiet = ', '.join(unanswered)
        return (
            f'Who talks to whom: {left} and {right} lead with {total}. '
            f'Nobody answered {quiet}.'
        )
    return f'Who talks to whom: {left} and {right} lead with {total}. Everyone was answered at least once.'


def group_evidence(
    *,
    participation: float,
    interaction: float,
    word_interaction: float,
    cast: str,
    target: int,
) -> tuple[str, set[str]]:
    lines = [
        f'PARTICIPATION: {round(100 * participation)}',
        f'INTERACTION: {round(100 * interaction)}',
        f'WORD_INTERACTION: {round(100 * word_interaction)}',
        f'TARGET: {target}',
        f'CAST: {cast}',
    ]
    text = '\n'.join(lines)
    return text, numbers_in(text)
