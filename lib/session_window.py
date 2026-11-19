"""Optional start and end for the analysis window.

The closer, the timeline, the stats, and the chord all read the trimmed transcript.
When the end is omitted, logistics at the tail are walked off until a substantive turn.
"""

from __future__ import annotations

import argparse
import json
import sys
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
