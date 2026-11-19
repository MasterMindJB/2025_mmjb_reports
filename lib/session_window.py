"""Optional start and end for the analysis window.

The closer, the timeline, the stats, and the chord all read the trimmed transcript.
When the end is omitted, topic words are learned from the first 70 percent of the
window. The end is the last on-topic turn plus the closing cues that follow it.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path


def parse_clock_ms(value: str) -> int:
    """m:ss or h:mm:ss, in milliseconds. 108:43 is 108 minutes and 43 seconds."""
    text = str(value or '').strip()
    parts = text.split(':')
    if len(parts) not in (2, 3) or not all(part.isdigit() for part in parts):
        raise ValueError(f'not a clock time: {value}')
    numbers = [int(part) for part in parts]
    if len(numbers) == 2:
        minutes, seconds = numbers
        hours = 0
    else:
        hours, minutes, seconds = numbers
    if seconds > 59 or (len(numbers) == 3 and minutes > 59):
        raise ValueError(f'not a clock time: {value}')
    return ((hours * 3600) + (minutes * 60) + seconds) * 1000


# Function words of five letters or more. Shorter words are dropped before this list.
STOPWORDS = {
    'about', 'above', 'after', 'again', 'being', 'below', 'could', 'every', 'first',
    'going', 'other', 'please', 'really', 'still', 'their', 'there', 'these', 'thing',
    'think', 'those', 'thank', 'thanks', 'where', 'which', 'while', 'would', 'people',
}
# Generic small talk. tf-idf treats these as already common, so they do not become topic keys.
SMALLTALK = {
    'actually', 'applause', 'audience', 'books', 'camera', 'everyone', 'goodbye',
    'great', 'hello', 'maybe', 'middle', 'panel', 'panelists', 'photo', 'photos',
    'picture', 'pictures', 'posing', 'present', 'repeat', 'repetition', 'smile',
    'sorry', 'speakers', 'stand', 'stuff', 'welcome',
}
_CLOSING_CUE = re.compile(
    r'('
    r'\bthank(?:s| you)?(?:\s+(?:the|our|all|you))?\s+(?:panel(?:ists)?|speakers|everyone)\b'
    r'|\bthank you all\b'
    r'|\bthank the panelists\b'
    r'|\bround of applause\b'
    r'|\bbooks\b'
    r'|\bpictures\b'
    r')',
    re.I,
)
_PANEL_SKIP = {'audience', 'noise'}


def content_words(text: str) -> list[str]:
    """Content words only: no contractions, no stopwords, nothing under five letters."""
    words = []
    for word in re.findall(r"[a-z']+", (text or '').lower()):
        if "'" in word or len(word) < 5 or word in STOPWORDS:
            continue
        words.append(word)
    return words


def topic_keys(rows: list) -> set[str]:
    """tf-idf keys from the first 70 percent of the window, against the small-talk list."""
    usable = [row for row in rows or [] if str(row.get('text') or '').strip()]
    if len(usable) < 2:
        return set()
    starts = [float(row.get('start') or 0) for row in usable]
    ends = [float(row.get('end') or row.get('start') or 0) for row in usable]
    span = max(ends) - min(starts)
    if span <= 0:
        early = usable[: max(1, int(len(usable) * 0.7))]
    else:
        cutoff = min(starts) + 0.7 * span
        early = [row for row in usable if float(row.get('start') or 0) <= cutoff]
        if len(early) < 2:
            early = usable[: max(1, int(len(usable) * 0.7))]
    documents = max(len(early), 1)
    tf: Counter[str] = Counter()
    df: Counter[str] = Counter()
    for row in early:
        words = content_words(str(row.get('text') or ''))
        tf.update(words)
        df.update(set(words))
    keys = set()
    for word, count in tf.items():
        if df[word] < 2:
            continue
        # Small talk is scored as if it already fills many other conversations.
        boost = documents * 10 if word in SMALLTALK else 0
        idf = math.log((1 + documents + boost) / (1 + df[word] + boost))
        score = (count / documents) * idf
        if score >= 0.15:
            keys.add(word)
    return keys


def on_topic(text: str, keys: set[str]) -> bool:
    """On topic means two of the learned keys, not one shared small-talk word."""
    if len(keys) < 2:
        return False
    return len(set(content_words(text)) & keys) >= 2


def is_closing_cue(text: str) -> bool:
    return bool(_CLOSING_CUE.search(text or ''))


def _panel_voice(row: dict) -> bool:
    role = str(row.get('role') or '').lower()
    return role not in _PANEL_SKIP


def _legacy_tail(usable: list) -> list:
    from lib.story_render import closing_clip, is_logistics

    rows = list(usable)
    while len(rows) > 1:
        text = str(rows[-1].get('text') or '')
        if is_logistics(text) or closing_clip(text) is None:
            rows.pop()
            continue
        break
    return rows


def substantive_body(text: str) -> str:
    """Drop a trailing thanks so 'Last word. Thank you all for coming.' keeps the substance."""
    parts = [part.strip() for part in re.split(r'(?<=[.!?])\s+', (text or '').strip()) if part.strip()]
    while len(parts) > 1 and is_closing_cue(parts[-1]):
        parts.pop()
    return ' '.join(parts).strip()


def _key_count(text: str, keys: set[str]) -> int:
    if not keys:
        return 0
    return len(set(content_words(text)) & keys)


def _later_turn_counts(text: str, keys: set[str]) -> bool:
    """After a closing cue, only a long panel turn with three topic keys can move the end."""
    from lib.story_render import _word_count

    return _word_count(text) >= 15 and _key_count(text, keys) >= 3


def _last_substantive_index(usable: list, keys: set[str]) -> int | None:
    from lib.story_render import _word_count, closing_clip, is_logistics

    last = None
    after_close = False
    for index, row in enumerate(usable):
        text = str(row.get('text') or '')
        if is_closing_cue(text):
            after_close = True
        if not _panel_voice(row):
            continue
        body = substantive_body(text)
        mixed = bool(body) and body != text.strip() and not is_closing_cue(body)
        content = body if mixed else text
        if is_logistics(text) and not mixed:
            continue
        if is_closing_cue(content) or not content:
            continue
        if _word_count(content) < 6 or closing_clip(content) is None:
            continue
        counts = _later_turn_counts(content, keys) if after_close and not mixed else on_topic(content, keys)
        if not counts:
            continue
        last = index
        if mixed:
            after_close = True
    return last


def window_through_closing(rows: list) -> list:
    """Last substantive turn, then the first closing-cue block, then cut."""
    usable = [row for row in rows or [] if str(row.get('text') or '').strip()]
    if not usable:
        return []
    keys = topic_keys(usable)
    last_on = _last_substantive_index(usable, keys) if len(keys) >= 2 else None
    if last_on is None:
        return _legacy_tail(usable)
    end = last_on
    index = last_on + 1
    while index < len(usable) and is_closing_cue(str(usable[index].get('text') or '')):
        end = index
        index += 1
    return usable[: end + 1]


def closer_turn(rows: list) -> dict | None:
    """The last substantive on-topic turn. Trailing thanks in that turn are removed."""
    usable = [row for row in rows or [] if str(row.get('text') or '').strip()]
    keys = topic_keys(usable)
    last_on = _last_substantive_index(usable, keys) if len(keys) >= 2 else None
    if last_on is None:
        from lib.story_render import closing_clip, is_logistics

        for row in reversed(_legacy_tail(usable)):
            text = substantive_body(str(row.get('text') or ''))
            if is_closing_cue(text) or is_logistics(text):
                continue
            if closing_clip(text):
                chosen = dict(row)
                chosen['text'] = text
                return chosen
        return None
    chosen = dict(usable[last_on])
    chosen['text'] = substantive_body(str(chosen.get('text') or ''))
    return chosen


def apply_session_window(transcript: dict, start_ms: int | None = None, end_ms: int | None = None) -> dict:
    """Keep utterances that start inside the window. An omitted end is detected."""
    from lib.story_render import discussion_rows

    rows = list((transcript or {}).get('utterances') or [])
    auto_end = end_ms is None
    if auto_end:
        rows = discussion_rows(rows)
        if rows:
            end_ms = int(float(rows[-1].get('end') or rows[-1].get('start') or 0))
    start = int(start_ms or 0)
    stop = None if end_ms is None else int(end_ms)

    def _inside(item: dict) -> bool:
        begin = float(item.get('start') or 0)
        if begin < start:
            return False
        if stop is not None and begin > stop:
            return False
        return True

    trimmed = dict(transcript or {})
    trimmed['utterances'] = [row for row in rows if _inside(row)]
    if 'sentiment_analysis_results' in (transcript or {}):
        trimmed['sentiment_analysis_results'] = [
            row for row in (transcript.get('sentiment_analysis_results') or []) if _inside(row)
        ]
    trimmed['session_window'] = {'start_ms': start, 'end_ms': stop, 'auto_end': auto_end}
    return trimmed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Trim a transcript to the session window.')
    parser.add_argument('transcript', help='Path to a transcript JSON file')
    parser.add_argument('--output', help='Where to write the trimmed transcript')
    parser.add_argument('--session-start', default='', help='Optional window start, m:ss or h:mm:ss')
    parser.add_argument('--session-end', default='', help='Optional window end, m:ss or h:mm:ss')
    args = parser.parse_args(argv)
    start = parse_clock_ms(args.session_start) if args.session_start else None
    end = parse_clock_ms(args.session_end) if args.session_end else None
    payload = json.loads(Path(args.transcript).read_text(encoding='utf-8'))
    trimmed = apply_session_window(payload, start, end)
    text = json.dumps(trimmed, indent=2)
    if args.output:
        Path(args.output).write_text(text, encoding='utf-8')
    else:
        sys.stdout.write(text)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
