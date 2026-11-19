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
    share_pct = int(round(share))
    question_count = int(questions)
    turn_count = int(turns)
    numbers = [
        (
            f'{share_pct}% talk time: you held the floor most of the way, '
            'more than the rest of the room.'
        ),
        (
            f'{question_count} questions: you stated the point without asking, '
            'against the session average.'
        ),
        f'{turn_count} turns: you entered the conversation {turn_count} times.',
    ]
    headline = 'Opened on the plan, then got specific.' if opening else 'Held the floor for most of the session.'
    goal = (
        f'Inferred from what you said: you came to move the plan forward. '
        f'At {opening_time} you said "{opening}".'
        if opening
        else 'Inferred from what you said: the aim is the job in your first turn.'
    )
    if early_late:
        challenge = (
            f'{early_late}. You asked {question_count} questions, '
            'so the point landed before a question did.'
        )
    else:
        challenge = f'You asked {question_count} questions across {turn_count} turns.'
    if opening and next_text and last_event and last_event['time'] != opening_time:
        drama = (
            f'At {opening_time} you said "{opening}". '
            'You might already know this about yourself: this phrase often reads as a bid to get moving. '
            f'Here\'s what\'s interesting: the next speaker said "{next_text}". '
            'It\'s doing a job for you. It buys a start. It also costs the question. '
            f'The one moment it didn\'t happen was {last_event["time"]}, '
            'and that\'s when you got specific.'
        )
    elif opening:
        drama = (
            f'At {opening_time} you said "{opening}". '
            'This phrase often reads as a bid to get moving, right after your own words.'
        )
    else:
        drama = 'No turn in this session gives a line to mirror.'
    resolution = (
        f'At {last_event["time"]} you said "{last_event["text"]}".'
        if last_event
        else 'The session ended without another turn from you.'
    )
    one_thing = f'Ask one question before you offer the plan. Target: {target} questions.'
    why = f'You asked {question_count} questions, so {target} is the halfway step.'
    transitions = {
        'after_character': "That's who showed up. Here's what they came to do.",
        'after_goal': "It didn't go to plan, and not only because of you.",
        'after_challenge': 'Now the interesting part.',
        'after_drama': 'So how did it end?',
        'after_resolution': 'If you change one thing next time, make it this.',
    }
    draft = draft or {}
    if draft.get('headline'):
        headline = draft['headline']
    character_body = draft.get('character_body') or ''
    goal = draft.get('goal') or goal
    challenge = draft.get('challenge') or challenge
    drama = draft.get('drama') or drama
    resolution = draft.get('resolution') or resolution
    one_thing = draft.get('one_thing') or one_thing
    why = draft.get('why') or why
    transitions.update(draft.get('transitions') or {})
    resolution = _append_note(resolution, str(info.get('personal_summary') or ''))
    resolution = _append_note(resolution, str(analysis.get('personal_super_power') or ''))
    actions = list(info.get('personal_action_items') or [])[:3]
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
        'actions': actions,
        'signature': {
            'phrase': bigram,
            'count': int(signature.get('count') or 0),
            'quotes': phrase_quotes[:2],
            **cards,
        },
        'contributions': list(info.get('personal_key_contributions') or [])[:2],
        'lines': lines,
        'appendix': {
            'AMASE': _top_framework(framework.get('amase_values'), framework.get('amase_notes'), name),
            'Enneagram': _top_framework(
                framework.get('enneagram_values'), framework.get('enneagram_notes'), name
            ),
            'Doshas': _top_framework(framework.get('dosha_values'), framework.get('dosha_notes'), name),
        },
    }


def group_story(
    *,
    summaries: dict,
    participation: float,
    interaction: float,
    word_interaction: float,
    cast: str = '',
    draft: dict | None = None,
) -> dict:
    from lib.story_evidence import halfway_target

    interaction_pct = int(round(100 * interaction))
    participation_pct = int(round(100 * participation))
    word_pct = int(round(100 * word_interaction))
    target = halfway_target(interaction_pct, 70 if interaction_pct < 70 else interaction_pct)
    headline = f'A serial panel at {interaction_pct}% interaction.'
    character = (
        f'{participation_pct}% participation, against {word_pct}% of the words exchanged. '
        + ('A few voices carried the airtime.' if participation < 0.75 else 'Airtime was fairly shared.')
    )
    goal = 'Inferred from the session, not from a written brief.'
    challenge = f'Interaction was {interaction_pct}%, so answers went to the center of the room.'
    drama = cast or 'Who talks to whom, and who nobody answered.'
    resolution = ''
    one_thing = f'Ask one question back before the next decision. Target: {target}% interaction.'
    why = f'Interaction was {interaction_pct}%, so {target}% is the halfway step.'
    transitions = {
        'after_character': "That's who showed up. Here's what they came to do.",
        'after_goal': "It didn't go to plan, and not only because of you.",
        'after_challenge': 'Now the interesting part.',
        'after_drama': 'So how did it end?',
        'after_resolution': 'If you change one thing next time, make it this.',
    }
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
    transitions.update(draft.get('transitions') or {})
    goal = _append_note(goal, str(summaries.get('conversation_summary_short') or ''))
    resolution = _append_note(resolution, str(summaries.get('conversation_summary_long') or ''))
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
