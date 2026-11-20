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


_EMAIL_ROLE = {
    'admin', 'contact', 'hello', 'info', 'noreply', 'office', 'sales', 'support', 'team',
}


def is_placeholder_name(name: str) -> bool:
    """A label such as Speaker G or F is not a person's name."""
    cleaned = re.sub(r'\s+', ' ', (name or '').strip())
    if not cleaned:
        return True
    if re.fullmatch(r'[A-Za-z]', cleaned):
        return True
    if re.fullmatch(r'(?i)speaker\s+\w+', cleaned):
        return True
    if cleaned.lower() in {'host', 'facilitator', 'moderator', 'participant', 'audience member'}:
        return True
    return False


def name_from_email(email: str) -> str | None:
    """first.last, first_last, and first-last become First Last. Handles and roles do not."""
    local = str(email or '').split('@', 1)[0].strip().lower()
    if not local or any(character.isdigit() for character in local):
        return None
    parts = [part for part in re.split(r'[._-]+', local) if part]
    if not parts or not all(part.isalpha() for part in parts):
        return None
    if any(part in _EMAIL_ROLE or part in ROLE_WORDS for part in parts):
        return None
    if len(parts) == 1:
        token = parts[0]
        if len(token) < 3:
            return None
        if len(token) <= 4 and token[0] not in 'aeiou' and token[1] not in 'aeiou':
            return None
        return token[:1].upper() + token[1:]
    if any(len(part) < 2 for part in parts):
        return None
    return ' '.join(part[:1].upper() + part[1:] for part in parts)


def header_name(*, entered: str = '', email: str = '', fallback: str = '') -> str | None:
    """The app-entered name wins. Otherwise a clear email name. Never a placeholder."""
    for candidate in (entered, fallback):
        cleaned = re.sub(r'\s+', ' ', (candidate or '').strip())
        if cleaned and not is_placeholder_name(cleaned):
            return cleaned
    return name_from_email(email)


def report_header(name: str | None) -> str:
    """The only top line. First name only. The model does not write it."""
    cleaned = re.sub(r'\s+', ' ', (name or '').strip())
    if cleaned:
        return f"{cleaned.split()[0]}, here's your report."
    return "Here's your report."


def table_name(name: str) -> str:
    """A pair-table label. 'Speaker F' is the letter. 'Audience member 1' is 'Aud. member 1'."""
    full = display_name(name)
    full = re.sub(r'(?i)^audience member\b', 'Aud. member', full)
    match = re.fullmatch(r'Speaker\s+([A-Za-z0-9]+)', full)
    if match:
        return match.group(1)
    return short_name(full)


def pair_cell(name_a: str, name_b: str) -> str:
    """Two lines, so 'Audience member 1 and Dr. Annie Green' does not wrap mid-name."""
    return f'{table_name(name_a)}\nand {table_name(name_b)}'


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
