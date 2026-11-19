"""Flesch-Kincaid grade for narrative text, with quoted speech left out."""

from __future__ import annotations

import re


def _syllables(word: str) -> int:
    token = re.sub(r'[^a-z]', '', word.lower())
    if len(token) > 2 and token.endswith('e') and not token.endswith('le'):
        token = token[:-1]
    groups = re.findall(r'[aeiouy]+', token)
    return max(1, len(groups) or 1)


def reading_grade(text: str) -> float:
    """Grade 6 or higher is too hard for the narrative. Quotes are not scored."""
    prose = re.sub(r'(?:"[^"]*"|“[^”]*”)', ' ', text or '')
    sentences = [part for part in re.split(r'[.!?]+', prose) if re.search(r'[A-Za-z]', part)]
    words = re.findall(r"[A-Za-z']+", prose)
    if len(words) < 4 or not sentences:
        return 0.0
    syllables = sum(_syllables(word) for word in words)
    return 0.39 * (len(words) / len(sentences)) + 11.8 * (syllables / len(words)) - 15.59
