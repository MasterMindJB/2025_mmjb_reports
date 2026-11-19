"""Turn a session's own counts and lines into a short data story.

Nothing here invents an example the person did not say. Quotes and counts
come from the transcript and the statistics table.
"""

from __future__ import annotations

import re

from lib.names import display_name


def _clock(milliseconds) -> str:
    try:
        seconds = max(0, int(float(milliseconds) / 1000))
    except (TypeError, ValueError):
        return '0:00'
    return f'{seconds // 60}:{seconds % 60:02d}'


def _lines(transcript: dict | None, label: str) -> list[dict]:
    rows = []
    for utterance in (transcript or {}).get('utterances') or []:
        if utterance.get('speaker') != label:
            continue
        text = str(utterance.get('text') or '').strip()
        if text:
            rows.append(utterance)
    return rows


def _stat(frame, name: str, column: str, default=0):
    if frame is None or name not in getattr(frame, 'index', []):
        return default
    if column not in frame.columns:
        return default
    value = frame.loc[name, column]
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _top_framework(values, explanations, name: str) -> str:
    if values is None or name not in getattr(values, 'index', []):
        return 'Not scored for this session.'
    row = values.loc[name]
    best = max(row.index, key=lambda column: float(row[column] or 0))
    score = int(float(row[best]))
    note = ''
    if explanations is not None and name in explanations.index and best in explanations.columns:
        note = str(explanations.loc[name, best] or '').strip()
    line = f'{best} ({score}/5).'
    if note:
        from lib.story_validation import BAN_PHRASES

        if not any(phrase in note.lower() for phrase in BAN_PHRASES):
            line = f'{line} {note}'
    return line


def quotes_for_phrase(lines: list[dict], phrase: str, limit: int = 2) -> list[str]:
    needle = (phrase or '').lower()
    found = []
    for utterance in lines:
        text = str(utterance.get('text') or '')
        if needle and needle not in text.lower():
            continue
        found.append(f'{_clock(utterance.get("start"))} "{text}"')
        if len(found) >= limit:
            break
    if found or not lines:
        return found
    for utterance in lines[:limit]:
        found.append(f'{_clock(utterance.get("start"))} "{utterance.get("text")}"')
    return found


def signature_examples(speaker_text: str, bigram: str) -> str:
    """Up to four of this person's own sentences that contain the bigram."""
    sentences = re.split(r'(?<=[.!?])\s+', speaker_text or '')
    hits = [sentence.strip() for sentence in sentences if bigram and bigram in sentence.lower()]
    return '\n'.join(f'- {sentence}' for sentence in hits[:4])


def _strip_preamble(text: str) -> str:
    cleaned = text or ''
    cleaned = re.sub(r'(?im)^as an ai\b[^\n]*\n?', '', cleaned)
    cleaned = re.sub(r'(?im)^as a sociolinguist\b[^\n]*\n?', '', cleaned)
    cleaned = re.sub(r'(?im)^i felt\b[^\n]*\n?', '', cleaned)
    cleaned = re.sub(r'^\s*\|.*\|\s*$', '', cleaned, flags=re.M)
    return cleaned.strip()


def signature_card(analysis: str) -> dict[str, str]:
    """Pull What / So, what / Now, what into one or two sentences each."""
    text = _strip_preamble(analysis or '')
    text = re.sub(r'[#*>`]+', '', text)
    parts = {'what': '', 'so_what': '', 'now_what': ''}
    markers = (
        ('now_what', r'now,\s*what\??'),
        ('so_what', r'so,\s*what\??'),
        ('what', r'(?<!so, )(?<!now, )what\??'),
    )
    positions = []
    lowered = text.lower()
    for key, pattern in markers:
        match = re.search(pattern, lowered)
        if match:
            positions.append((match.start(), match.end(), key))
    positions.sort()
    for index, (start, end, key) in enumerate(positions):
        stop = positions[index + 1][0] if index + 1 < len(positions) else len(text)
        sentence = text[end:stop].strip(' \n:-')
        sentence = re.sub(r'\s+', ' ', sentence)
        pieces = re.split(r'(?<=[.!?])\s+', sentence)
        parts[key] = ' '.join(piece for piece in pieces[:2] if piece).strip()
    if not any(parts.values()) and text:
        pieces = re.split(r'(?<=[.!?])\s+', text)
        parts['what'] = ' '.join(pieces[:2]).strip()
    return parts


def _append_note(text: str, note: str) -> str:
    note = (note or '').strip()
    if not note or note in (text or ''):
        return text
    return f'{text} {note}'.strip()


def personal_story(
    *,
    label: str,
    info: dict,
    analysis: dict,
    transcript: dict | None,
    stats_frame,
    framework: dict,
    draft: dict | None = None,
) -> dict:
    name = display_name(info.get('name') or label)
    lines = _lines(transcript, label)
    words = int(_stat(stats_frame, name, 'Word Count', 0))
    if not words:
        words = sum(len(str(row.get('text') or '').split()) for row in lines)
    share = float(_stat(stats_frame, name, 'Speaking Time (%)', 0))
    if not share and lines:
        mine = sum(max(0, float(row.get('end') or 0) - float(row.get('start') or 0)) for row in lines)
        total = 0.0
        for row in (transcript or {}).get('utterances') or []:
            total += max(0, float(row.get('end') or 0) - float(row.get('start') or 0))
        share = (100 * mine / total) if total else 0
    questions = int(_stat(stats_frame, name, 'Questions Asked', 0))
    if questions == 0:
        questions = sum(str(row.get('text') or '').count('?') for row in lines)
    turns = int(_stat(stats_frame, name, 'Total Interactions', len(lines)))
    wpm = float(_stat(stats_frame, name, 'Words Per Minute (WPM)', 0))
    from lib.story_evidence import before_after, halfway_target, quote_events

    events = quote_events(transcript, label)
    session_turns = len((transcript or {}).get('utterances') or []) or int(turns)
    question_values = []
    if stats_frame is not None and 'Questions Asked' in getattr(stats_frame, 'columns', []):
        for row_name in stats_frame.index:
            if row_name == 'Average':
                continue
            question_values.append(int(float(stats_frame.loc[row_name, 'Questions Asked'] or 0)))
    session_question_average = (
        int(round(sum(question_values) / len(question_values))) if question_values else int(questions)
    )
    rest = max(0, 100 - int(round(share)))
    target = halfway_target(int(questions), 3 if int(questions) < 3 else int(questions))
    early_late = before_after(transcript, label)
    first_event = events[0] if events else None
    last_event = events[-1] if events else None
    opening = first_event['text'] if first_event else ''
    opening_time = first_event['time'] if first_event else '0:00'
    next_text = first_event['next_text'] if first_event else ''
    signature = analysis.get('personal_signature_phrase') or {}
    bigram = str(signature.get('bigram') or '')
    phrase_quotes = quotes_for_phrase(lines, bigram)
    cards = signature_card(str(signature.get('analysis') or ''))
    from lib.story_render import (
        clip_clause,
        one_thing_from_gap,
        pick_transitions,
        questions_sentence,
        share_sentence,
        turns_sentence,
    )

    share_pct = int(round(share))
    question_count = int(questions)
    turn_count = int(turns)
    speakers = {
        row.get('speaker')
        for row in (transcript or {}).get('utterances') or []
        if row.get('speaker')
    }
    speaker_count = max(1, len(speakers))
    ends = [float(row.get('end') or 0) for row in (transcript or {}).get('utterances') or []]
    numbers = [
        share_sentence(share_pct, speaker_count),
        questions_sentence(question_count, session_question_average),
        turns_sentence(turn_count),
    ]
    used_quotes: set[str] = set()

    def _once(event):
        if not event:
            return None
        clause = clip_clause(event['text'])
        key = clause.lower()
        if not clause or key in used_quotes:
            return None
        used_quotes.add(key)
        return clause, event['time']

    opening_bit = _once(first_event)
    later_bit = _once(last_event)
    if opening_bit:
        clause, _when = opening_bit
        headline = clause if len(clause.split()) <= 12 else ' '.join(clause.split()[:12])
    else:
        headline = turns_sentence(turn_count)
    if opening_bit:
        goal = f'Inferred from the opening line at {opening_bit[1]}.'
    else:
        goal = 'Inferred from the turns in this session.'
    question_line = questions_sentence(question_count, session_question_average)
    if early_late:
        challenge = f'{early_late.rstrip(".")}. {question_line}'
    else:
        challenge = question_line
    if opening_bit and later_bit:
        drama = (
            f'At {opening_bit[1]} you said "{opening_bit[0]}". '
            f'A later line, at {later_bit[1]}, was "{later_bit[0]}".'
        )
    elif opening_bit:
        drama = f'At {opening_bit[1]} you said "{opening_bit[0]}".'
    else:
        drama = 'No turn in this session gives a line to quote.'
    resolution = (
        f'{turns_sentence(turn_count)} {question_line} '
        'The gap in those counts is the part to change.'
    )
    early = late = None
    if early_late and ' then ' in early_late:
        early = int(re.search(r'\d+', early_late).group(0))
        late = int(re.search(r'then (\d+)', early_late).group(1))
    one_thing, why = one_thing_from_gap(
        share=share_pct,
        speaker_count=speaker_count,
        questions=question_count,
        question_average=session_question_average,
        early=early,
        late=late,
    )
    transitions = pick_transitions(label)
    draft = draft or {}
    if draft.get('headline'):
        headline = draft['headline']
    if draft.get('numbers'):
        numbers = draft['numbers']
    character_body = draft.get('character_body') or ''
    goal = draft.get('goal') or goal
    challenge = draft.get('challenge') or challenge
    drama = draft.get('drama') or drama
    resolution = draft.get('resolution') or resolution
    one_thing = draft.get('one_thing') or one_thing
    why = draft.get('why') or why
    if draft.get('transitions'):
        transitions = draft['transitions']
    if signature.get('what') or signature.get('so_what') or signature.get('now_what'):
        cards = {
            'what': signature.get('what') or '',
            'so_what': signature.get('so_what') or '',
            'now_what': signature.get('now_what') or '',
        }
    if not cards.get('what'):
        from lib.story_render import computed_signature

        cards = computed_signature(bigram, int(signature.get('count') or 0), phrase_quotes)
    return {
        'name': name,
        'headline': headline,
        'character_body': character_body,
        'numbers': numbers,
        'share': share,
        'turns': turn_count,
        'questions': question_count,
        'wpm': wpm,
        'words': int(words),
        'goal': goal,
        'challenge': challenge,
        'drama': drama,
        'resolution': resolution,
        'one_thing': one_thing,
        'why': why,
        'transitions': transitions,
        'session_end_ms': max(ends) if ends else 0,
        'actions': [],
        'signature': {
            'phrase': bigram,
            'count': int(signature.get('count') or 0),
            'quotes': phrase_quotes[:2],
            **cards,
        },
        'contributions': [],
        'lines': lines,
        'appendix': {
            'AMASE': _top_framework(framework.get('amase_values'), framework.get('amase_notes'), name),
            'Enneagram': _top_framework(
                framework.get('enneagram_values'), framework.get('enneagram_notes'), name
            ),
            'Doshas': _top_framework(framework.get('dosha_values'), framework.get('dosha_notes'), name),
        },
    }


def _meeting_topics(transcript: dict | None, limit: int = 3) -> list[str]:
    from lib.story_validation import GLUE_WORDS

    filler = GLUE_WORDS | {
        'think', 'thing', 'thats', "that's", 'just', 'like', 'know', 'yeah', 'really',
        'gonna', 'wanna', 'something', 'anything', 'everything', 'nothing', 'please',
        'thank', 'thanks', 'hello', 'right', 'well', 'okay', 'okay',
    }
    counts: dict[str, int] = {}
    for utterance in (transcript or {}).get('utterances') or []:
        for word in re.findall(r"[A-Za-z']{5,}", str(utterance.get('text') or '')):
            token = word.lower().replace("'", '')
            if token in filler:
                continue
            counts[token] = counts.get(token, 0) + 1
    ranked = sorted(counts, key=lambda item: (-counts[item], item))
    return ranked[:limit]


def group_story(
    *,
    summaries: dict,
    participation: float,
    interaction: float,
    word_interaction: float,
    cast: str = '',
    draft: dict | None = None,
    transcript: dict | None = None,
) -> dict:
    from lib.story_render import improve_toward

    interaction_pct = int(round(100 * interaction))
    participation_pct = int(round(100 * participation))
    word_pct = int(round(100 * word_interaction))
    target = improve_toward(
        interaction_pct,
        70 if interaction_pct < 70 else interaction_pct + 1,
        increase=True,
    )
    headline = f'A session at {interaction_pct}% interaction.'
    character = (
        f'{participation_pct}% participation. {word_pct}% of the words were exchanged across turns. '
        + ('A few voices carried the airtime.' if participation < 0.75 else 'Airtime was fairly shared.')
    )
    topics = _meeting_topics(transcript)
    if len(topics) >= 2:
        listed = ', '.join(topics[:-1]) + f', and {topics[-1]}'
        goal = f'The turns keep returning to {listed}.'
    elif topics:
        goal = f'The turns keep returning to {topics[0]}.'
    else:
        goal = f'The turns are the record of the meeting. Interaction was {interaction_pct}%.'
    challenge = f'Interaction was {interaction_pct}%.'
    drama = cast or 'Who talks to whom, and who nobody answered.'
    last_line = ''
    for utterance in reversed((transcript or {}).get('utterances') or []):
        text = str(utterance.get('text') or '').strip()
        if text:
            from lib.story_render import clip_clause

            last_line = clip_clause(text, 12)
            break
    if last_line:
        resolution = f'The meeting closed on "{last_line}". Interaction finished at {interaction_pct}%.'
    else:
        resolution = f'Interaction finished at {interaction_pct}%.'
    one_thing = f'Name one reply before the next decision. Target: {target}% interaction.'
    why = f'Interaction was {interaction_pct}%, so {target}% is the next step.'
    from lib.story_render import pick_transitions

    transitions = pick_transitions('group', group=True)
    draft = draft or {}
    if draft.get('headline'):
        headline = draft['headline']
    character = draft.get('character') or character
    goal = draft.get('goal') or goal
    challenge = draft.get('challenge') or challenge
    drama = draft.get('drama') or drama
    resolution = draft.get('resolution') or resolution
    one_thing = draft.get('one_thing') or one_thing
    why = draft.get('why') or why
    if draft.get('transitions'):
        transitions = draft['transitions']
    return {
        'headline': headline,
        'character': character,
        'goal': goal,
        'challenge': challenge,
        'drama': drama,
        'resolution': resolution,
        'one_thing': one_thing,
        'why': why,
        'transitions': transitions,
    }
