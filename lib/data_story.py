"""Turn a session's own counts and lines into a short data story.

Nothing here invents an example the person did not say. Quotes and counts
come from the transcript and the statistics table.
"""

from __future__ import annotations

import re

from lib.names import display_name, short_name

HEDGES = ('maybe', 'might', 'i think', 'kind of', 'sort of', 'perhaps', 'just')


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


def _headline(name: str, share: float, questions: int, role: str) -> str:
    who = short_name(name)
    question_word = 'question' if int(questions) == 1 else 'questions'
    return (
        f'{who} was the room\'s {role}: {share:.0f}% of the words, '
        f'{int(questions)} {question_word}.'
    )


def _role_label(share: float, questions: int, turns: int) -> str:
    if questions >= 2 and share < 40:
        return 'questioner'
    if share >= 35:
        return 'storyteller'
    if turns <= 1:
        return 'listener'
    return 'contributor'


def personal_story(
    *,
    label: str,
    info: dict,
    analysis: dict,
    transcript: dict | None,
    stats_frame,
    framework: dict,
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
    role = _role_label(share, questions, turns)
    joined = ' '.join(str(row.get('text') or '') for row in lines)
    lowered = joined.lower()
    i_count = len(re.findall(r'\bi\b', lowered))
    we_count = len(re.findall(r'\bwe\b', lowered))
    hedge_count = sum(lowered.count(hedge) for hedge in HEDGES)
    signature = analysis.get('personal_signature_phrase') or {}
    bigram = str(signature.get('bigram') or '')
    phrase_quotes = quotes_for_phrase(lines, bigram)
    evidence = phrase_quotes or quotes_for_phrase(lines, '', limit=2)
    cards = signature_card(str(signature.get('analysis') or ''))
    first = lines[0]['text'] if lines else 'you did not take a turn'
    last = lines[-1]['text'] if lines else ''
    goal = (
        f'Inferred from what you said, not from a stated agenda: you opened with '
        f'"{first}". That is the aim this read uses.'
    )
    if questions < 2:
        challenge = (
            f'You asked {int(questions)} questions across {int(turns)} turns. '
            f'The room heard your point before it heard your curiosity. '
            + (f'At {evidence[0]}.' if evidence else '')
        )
        one_thing = (
            'Ask one open question before your second story; target 3+ questions next session.'
        )
        why = 'A question hands the next turn to someone else and tests the story you are about to tell.'
    else:
        challenge = (
            f'You carried {share:.0f}% of the speaking time. Holding the floor that long '
            f'makes it harder for a quieter person to enter. '
            + (f'At {evidence[0]}.' if evidence else '')
        )
        one_thing = 'Stop after your point and leave one full silence; target under 25% of the words next session.'
        why = 'The extra pause is where someone else can add the fact you do not have.'
    we_line = (
        f'You used "I" {i_count} times and "we" {we_count} times. '
        f'Hedges such as "might" or "I think" showed up {hedge_count} times.'
    )
    drama = (
        'You might already know this about yourself. '
        + (
            f'People who say "{bigram}" a lot tend to lean on that phrase while they find the next idea. '
            if bigram
            else 'No two-word phrase stood out often enough to call a signature. '
        )
        + we_line
    )
    resolution = (
        f'By the end you had said: "{last}". '
        f'{info.get("personal_summary") or ""} '
        f'{analysis.get("personal_super_power") or ""}'
    ).strip()
    actions = list(info.get('personal_action_items') or [])[:3]
    return {
        'name': name,
        'headline': _headline(name, share, questions, role),
        'share': share,
        'turns': int(turns),
        'questions': int(questions),
        'wpm': wpm,
        'words': int(words),
        'goal': goal,
        'challenge': challenge,
        'drama': drama,
        'resolution': resolution,
        'one_thing': one_thing,
        'why': why,
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


def group_story(*, summaries: dict, participation: float, interaction: float, word_interaction: float) -> dict:
    kind = 'a serial panel, not a discussion' if interaction < 0.6 else 'a real exchange'
    headline = (
        f'This conversation was {kind}: the interaction score was {round(100 * interaction)}%.'
    )
    return {
        'headline': headline,
        'character': (
            f'Participation was {round(100 * participation)}% and word interaction was '
            f'{round(100 * word_interaction)}%. '
            + ('A few voices carried the airtime.' if participation < 0.75 else 'Airtime was fairly shared.')
        ),
        'goal': (
            'Inferred from the session summary, not from a written brief: '
            + str(summaries.get('conversation_summary_short') or summaries.get('conversation_title') or '')
        ),
        'challenge': (
            'The hard part is structural. When the interaction score is low, people answer the '
            'facilitator instead of each other, so a point can land without being tested.'
        ),
        'resolution': str(summaries.get('conversation_summary_long') or ''),
        'one_thing': (
            'Before the next decision, ask the quietest person one open question and wait for the answer.'
        ),
        'why': 'A discussion is not a series of reports to the center of the room.',
    }
