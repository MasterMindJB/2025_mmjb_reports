"""Display names and short names.

A role title is not a first name. "Full Sail Moderator" stays whole.
"Alex Rivera" can be "Alex" because the first word is the given name.
"""

from __future__ import annotations

import re

ROLE_WORDS = {
    'actor',
    'coach',
    'facilitator',
    'host',
    'moderator',
    'participant',
    'person',
    'speaker',
}


def display_name(name: str) -> str:
    cleaned = re.sub(r'\s+', ' ', (name or '').strip())
    return cleaned or 'Participant'


def short_name(name: str) -> str:
    """A label for a chart. Role titles are never cut down to their first word."""
    full = display_name(name)
    parts = full.split(' ')
    if len(parts) <= 1:
        return full
    lowered = [re.sub(r'[^a-z]', '', part.lower()) for part in parts]
    if any(part in ROLE_WORDS for part in lowered):
        return full
    if len(parts) > 2:
        return full
    return parts[0]
