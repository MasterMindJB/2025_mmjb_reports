"""Optional names. A missing name stays a neutral label. Nothing here invents a person."""

from __future__ import annotations

import re

INTRO = re.compile(
    r"\b(?:i'm|i am|my name is|this is)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
    re.I,
)
ADDRESS = re.compile(r"\b(?:thanks|thank you|hey|hi),?\s+([A-Z][a-z]+)\b", re.I)

ROLE_BASE = {
    'host': 'Host',
    'audience': 'Audience member',
    'facilitator': 'Facilitator',
    'actor': 'Actor',
}


def suggest_names(transcript: dict | None) -> dict[str, str]:
    """Names heard as a self-introduction, or as a direct address to the next speaker."""
    suggestions: dict[str, str] = {}
    rows = (transcript or {}).get('utterances') or []
    for index, row in enumerate(rows):
        label = row.get('speaker')
        text = str(row.get('text') or '')
        intro = INTRO.search(text)
        if intro and label and label not in suggestions:
            suggestions[label] = intro.group(1).strip()
        address = ADDRESS.search(text)
        if address and index + 1 < len(rows):
            nxt = rows[index + 1].get('speaker')
            if nxt and nxt not in suggestions:
                suggestions[nxt] = address.group(1).strip()
    return suggestions


def placeholder_name(label: str, role: str, role_index: int) -> str:
    if role == 'participant':
        return f'Speaker {label}'
    base = ROLE_BASE.get(role, 'Speaker')
    if role == 'host' and role_index == 1:
        return 'Host'
    if role == 'participant':
        return f'Speaker {label}'
    return f'{base} {role_index}'


def assign_display_names(speakers: list[dict]) -> list[dict]:
    """Everyone except noise stays in the session. Only a real name plus an email gets a personal report."""
    counts: dict[str, int] = {}
    assigned = []
    for speaker in speakers:
        role = speaker.get('role') or 'participant'
        status = speaker.get('name_status') or 'confirmed'
        item = dict(speaker)
        if role == 'noise' or status == 'ignored':
            item['display_name'] = None
            item['include'] = False
            item['receives_report'] = False
            assigned.append(item)
            continue
        given = ' '.join(str(speaker.get('name') or '').split())
        skipped = status in {'skipped', 'ignored'} or not given
        if skipped:
            counts[role] = counts.get(role, 0) + 1
            item['display_name'] = placeholder_name(str(speaker.get('label') or ''), role, counts[role])
        else:
            item['display_name'] = given
        item['include'] = True
        item['receives_report'] = bool(
            role == 'participant' and given and not skipped and str(speaker.get('email') or '').strip()
        )
        assigned.append(item)
    return assigned
