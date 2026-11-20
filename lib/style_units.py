"""The repeated unit that opens a personal report.

Candidates are counted in code. The model may only return an id.
"""

from __future__ import annotations

import math
import re

from lib.report_stats import BIGRAM_STOPWORDS

_FILLERS = {'um', 'uh', 'hmm', 'er', 'ah', 'erm', 'mm', 'hm'}
_STYLE_MARKERS = {
    'i think', 'i mean', 'you know', 'kind of', 'sort of', 'a little',
    'really', 'just', 'i guess', 'a bit', 'you see',
}
_STOP = set(BIGRAM_STOPWORDS) | {
    'like', 'yeah', 'yes', 'well', 'right', 'gonna', 'wanna', 'stuff', 'thing', 'things',
}


def _tokens(text: str) -> list[str]:
    return [word.strip("'") for word in re.findall(r"[A-Za-z']+", (text or '').lower()) if word.strip("'")]


def _name_tokens(names: list[str] | None) -> set[str]:
    found = set()
    for name in names or []:
        for part in _tokens(name):
            if len(part) > 2:
                found.add(part)
    return found


def _keep(unit: str, names: set[str]) -> bool:
    parts = unit.split()
    if any(part in _FILLERS or part in names for part in parts):
        return False
    if unit in _STYLE_MARKERS or (len(parts) == 1 and parts[0] in _STYLE_MARKERS):
        return True
    if all(part in _STOP for part in parts):
        return False
    return True


def _counts(turns: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for turn in turns:
        tokens = _tokens(turn)
        for size in (1, 2, 3, 4):
            if len(tokens) < size:
                continue
            for index in range(len(tokens) - size + 1):
                unit = ' '.join(tokens[index:index + size])
                counts[unit] = counts.get(unit, 0) + 1
    return counts


def _subsumed(unit: str, count: int, counts: dict[str, int]) -> bool:
    """A shorter unit that only occurs inside a longer one with the same count."""
    for longer, longer_count in counts.items():
        if longer_count != count or longer == unit:
            continue
        if f' {unit} ' in f' {longer} ' or longer.startswith(unit + ' ') or longer.endswith(' ' + unit):
            return True
    return False


def _speaker_turns(transcript: dict | None) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for row in (transcript or {}).get('utterances') or []:
        text = str(row.get('text') or '').strip()
        label = str(row.get('speaker') or '')
        if not text or not label:
            continue
        grouped.setdefault(label, []).append(text)
    return grouped


def style_candidates(transcript: dict | None, label: str, names: list[str] | None = None) -> list[dict]:
    """Top units for this speaker, scored by how distinctive they are in the room."""
    grouped = _speaker_turns(transcript)
    mine_turns = grouped.get(label) or []
    if not mine_turns:
        return []
    blocked = _name_tokens(names)
    mine = {unit: count for unit, count in _counts(mine_turns).items() if count >= 2 and _keep(unit, blocked)}
    mine = {unit: count for unit, count in mine.items() if not _subsumed(unit, count, mine)}
    if not mine:
        return []
    others: dict[str, dict[str, int]] = {}
    other_words = 0
    my_words = sum(len(_tokens(turn)) for turn in mine_turns) or 1
    for other, turns in grouped.items():
        if other == label:
            continue
        other_words += sum(len(_tokens(turn)) for turn in turns)
        for unit, count in _counts(turns).items():
            others.setdefault(unit, {})[other] = count
    other_words = other_words or 1
    ranked = []
    for unit, count in mine.items():
        per_person = others.get(unit) or {}
        other_total = sum(per_person.values())
        mine_rate = (count + 0.5) / (my_words + 1)
        room_rate = (other_total + 0.5) / (other_words + 1)
        score = math.log(mine_rate / room_rate) * count
        ranked.append({
            'text': unit,
            'count': count,
            'others_max': max(per_person.values(), default=0),
            'score': round(score, 4),
        })
    ranked.sort(key=lambda item: (-item['score'], -item['count'], item['text']))
    chosen = []
    for index, item in enumerate(ranked[:5]):
        chosen.append({'id': f'U{index + 1}', **item})
    return chosen


def opening_line(candidate: dict | None) -> str:
    """The code-owned opening. No article before the quoted unit."""
    if not candidate:
        return ''
    unit = str(candidate.get('text') or '').strip()
    count = int(candidate.get('count') or 0)
    if not unit or count < 2:
        return ''
    line = f'You said "{unit}" {count} times'
    if count > int(candidate.get('others_max') or 0):
        line += ', more than anyone else in the room'
    return line + '.'


def chosen_style(candidates: list[dict], picked: str) -> dict | None:
    """The model's id, or the top candidate when the id is missing."""
    token = str(picked or '').strip().upper()
    for item in candidates:
        if item['id'].upper() == token:
            return item
    return candidates[0] if candidates else None
