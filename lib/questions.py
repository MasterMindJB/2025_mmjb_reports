"""Count questions from transcript text. Placeholders stay in the count."""

from __future__ import annotations

import re

# A helping verb or a wh- word at the start of a sentence with no end mark.
_INTERROGATIVES = {
    'who', 'what', 'when', 'where', 'why', 'how',
    'is', 'are', 'am', 'was', 'were',
    'do', 'does', 'did',
    'can', 'could', 'would', 'should', 'will', 'shall',
}

QUESTION_FOOTNOTE = (
    'A question is a sentence that ends with a question mark. '
    'If that mark is missing, a sentence that starts with who, what, when, where, why, how, '
    'or with is, are, do, does, did, can, could, would, or should still counts.'
)


def questions_in_text(text: str) -> int:
    """How many questions are in one turn."""
    count = 0
    for part in re.split(r'(?<=[.!?])\s+|\n+', str(text or '').strip()):
        sentence = part.strip()
        if not sentence:
            continue
        if sentence.endswith('?'):
            count += 1
            continue
        if sentence.endswith(('.', '!')):
            continue
        words = re.findall(r"[A-Za-z']+", sentence)
        if words and words[0].lower() in _INTERROGATIVES:
            count += 1
    return count


def question_counts(transcript: dict | None, names: dict | None = None) -> list[tuple[str, int]]:
    """One count per person, including people who asked nothing."""
    totals: dict[str, int] = {}
    for name in (names or {}).values():
        if name:
            totals.setdefault(str(name), 0)
    for row in (transcript or {}).get('utterances') or []:
        label = row.get('speaker')
        name = row.get('name') or (names or {}).get(label) or (str(label) if label else 'Speaker')
        totals[str(name)] = totals.get(str(name), 0) + questions_in_text(str(row.get('text') or ''))
    return sorted(totals.items(), key=lambda item: (-item[1], item[0]))
