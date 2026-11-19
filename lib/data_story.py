"""Turn a session's own counts and lines into a short data story.

Nothing here invents an example the person did not say. Quotes and counts
come from the transcript and the statistics table.
"""

from __future__ import annotations

import re

from lib.names import display_name


def _clock(milliseconds) -> str:
    from lib.story_evidence import clock

    return clock(milliseconds)


_INTRO = re.compile(
    r"^(hello|hi|hey|good morning|good afternoon|good evening)\b|\bmy name is\b"
    r"|^[A-Z][\w.']+(?:\s+[A-Z][\w.']+){0,3}\s+and\s+[A-Z]"
    r"|\bnoco international\b",
    re.I,
)


def _fallback_event(events: list[dict], rows: list) -> dict | None:
    """Longest on-topic turn of at least six words. Intros and 'Repetition.' never qualify."""
    from lib.session_window import on_topic, topic_keys
    from lib.story_render import closing_clip, is_logistics

    keys = topic_keys(rows)

    def usable(event: dict, require_topic: bool) -> bool:
        text = str(event.get('text') or '').strip()
        if _INTRO.search(text) or is_logistics(text):
            return False
        words = len(re.findall(r"[A-Za-z']+", text))
        if words < 3:
            return False
        if require_topic and keys and not on_topic(text, keys):
            return False
        if words >= 6:
            return closing_clip(text) is not None
        return True

    long = [event for event in events if usable(event, True) and len(re.findall(r"[A-Za-z']+", str(event.get('text') or ''))) >= 6]
    short = [event for event in events if usable(event, False)]
    pool = long or short
    if not pool:
        return None
    return max(pool, key=lambda event: len(re.findall(r"[A-Za-z']+", str(event.get('text') or ''))))


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
        clip = sentence_around(centered_bigram_clip(text, phrase) or phrase, text)
        if not clip:
            continue
        found.append(f'{_clock(utterance.get("start"))} "{clip}"')
        if len(found) >= limit:
            break
    return found


def sentence_around(fragment: str, source: str) -> str:
    """The full sentence that holds the fragment. An ellipsis only if it starts mid-sentence."""
    fragment = (fragment or '').strip().strip('"').lstrip('…').strip()
    if not fragment:
        return ''
    source = (source or fragment).strip()
    sentences = [part.strip() for part in re.split(r'(?<=[.!?])\s+', source) if part.strip()]
    key = re.sub(r'[^a-z0-9]+', '', fragment.lower())[:40]
    for sentence in sentences:
        bare = re.sub(r'[^a-z0-9]+', '', sentence.lower())
        if key and key in bare:
            return sentence
    if fragment[0].islower():
        return '…' + fragment
    return fragment


def centered_bigram_clip(text: str, bigram: str) -> str:
    """A 6 to 25 word window centered on one occurrence of the bigram."""
    tokens = re.findall(r'\S+', text or '')
    parts = (bigram or '').lower().split()
    if len(tokens) < 2 or len(parts) != 2:
        return ''

    def norm(token: str) -> str:
        return re.sub(r"[^a-z0-9']+", '', token.lower())

    norms = [norm(token) for token in tokens]
    for index in range(len(tokens) - 1):
        if norms[index:index + 2] != parts:
            continue
        start, end = index, index + 2
        while end - start < 6 and (start > 0 or end < len(tokens)):
            if start > 0:
                start -= 1
            else:
                end += 1
        while end - start < 12 and end < len(tokens) and end - start < 25:
            end += 1
        while end - start > 25:
            if index - start >= end - (index + 2):
                start += 1
            else:
                end -= 1
        if end - start < 6:
            continue
        clip = ' '.join(tokens[start:end]).strip(' ,;')
        if start > 0:
            clip = '…' + clip.lstrip()
        words = re.findall(r"[A-Za-z0-9']+", clip)
        if not 6 <= len(words) <= 25:
            continue
        if bigram.lower() not in re.sub(r'\s+', ' ', clip).lower():
            continue
        bare = re.sub(r'[^a-z]+', '', clip.lower())
        if bare in {'repetition', 'yeah', 'yes', 'right'}:
            continue
        return clip
    return ''


def signature_examples(speaker_text: str, bigram: str) -> str:
    """Up to four clips that contain the bigram, each 6 to 25 words."""
    hits = []
    for sentence in re.split(r'(?<=[.!?])\s+', speaker_text or ''):
        clip = centered_bigram_clip(sentence, bigram)
        if not clip:
            clip = centered_bigram_clip(speaker_text, bigram)
        if clip and clip not in hits:
            hits.append(clip)
        if len(hits) >= 4:
            break
    return '\n'.join(f'- {clip}' for clip in hits)


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
        closing_clip,
        goal_line,
        one_thing_from_gap,
        pick_transitions,
        questions_sentence,
        question_number_line,
        share_sentence,
        talk_number_line,
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
    starts = [float(row.get('start') or 0) for row in (transcript or {}).get('utterances') or []]
    window = (transcript or {}).get('session_window') or {}
    session_start_ms = window.get('start_ms')
    if not session_start_ms and starts:
        session_start_ms = min(starts)
    session_start_ms = int(session_start_ms or 0)
    turn_values = []
    if stats_frame is not None and 'Total Interactions' in getattr(stats_frame, 'columns', []):
        for row_name in getattr(stats_frame, 'index', []):
            if str(row_name) == 'Average':
                continue
            turn_values.append(int(float(stats_frame.loc[row_name, 'Total Interactions'] or 0)))
    from lib.story_render import room_median

    turn_median = room_median(turn_values)
    numbers = [
        talk_number_line(share_pct, speaker_count),
        question_number_line(question_count, session_question_average),
        turns_sentence(turn_count, turn_median),
    ]
    def _substantive(event) -> bool:
        words = re.findall(r"[A-Za-z']+", str((event or {}).get('text') or ''))
        if len(words) >= 8:
            return True
        return len(words) >= 4 and words[0].lower() not in {
            'yeah', 'yes', 'yep', 'right', 'ok', 'okay', 'sure', 'mm', 'hmm', 'uh', 'um',
        }

    thin = sum(1 for event in events if _substantive(event)) < 4
    question_line = questions_sentence(question_count, session_question_average).rstrip('.')
    if thin:
        headline = 'You entered in short turns'
    elif share_pct >= 40:
        headline = 'You held the floor for a long stretch'
    else:
        headline = 'You came in and set a point down'
    evidence_event = _fallback_event(events, (transcript or {}).get('utterances') or [])
    if evidence_event:
        said = closing_clip(evidence_event['text']) or clip_clause(evidence_event['text'], 18)
        when = evidence_event['time']
        from lib.story_render import _trim_discourse

        shared = _trim_discourse(said)
        if len(re.findall(r"[A-Za-z']+", shared)) < 3:
            shared = _trim_discourse(str(evidence_event.get('text') or ''))
        shared = shared.rstrip('.').strip(' ,')
        if shared[:1].isupper():
            shared = shared[0].lower() + shared[1:]
        if shared:
            first = re.sub(r'[^\w]', '', shared.split()[0])
            from lib.story_render import _base_verb, _tagged_verb

            if _base_verb(first) or _tagged_verb(first):
                goal = goal_line(shared, personal=True)
            else:
                goal = f'Your aim: {shared}.'
        else:
            goal = 'Your aim: what you added in the room.'
        if thin:
            drama = (
                f'At {when} you said "{said}". '
                f'You held {share_pct}% of the talk time across {turn_count} {"turn" if turn_count == 1 else "turns"}, '
                f'so the pattern is when you chose to enter, not a long story.'
            )
        else:
            drama = (
                f'At {when} you said "{said}". '
                f'That line does a job for you: it starts the point, and it also costs the room a reply.'
            )
    else:
        goal = 'Your aim: what you added in the room.'
        drama = (
            f'You held {share_pct}% of the talk time. '
            'The pattern is how little of the exchange waited for a longer line from you.'
        )
    challenge = (
        f'{question_line}. '
        'That is the gap a listener can hear, because a turn can end before anyone has to answer you.'
    )
    if thin:
        resolution = (
            f'You used {turn_count} {"turn" if turn_count == 1 else "turns"} and left the longer story to the room. '
            'One extra clause the next time you enter would give that short line a point.'
        )
    else:
        resolution = (
            f'{question_line}. '
            'The part to change is the next turn, where one added question would make the aim land.'
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
        'turns': int(draft.get('turns') or turn_count),
        'questions': question_count,
        'wpm': wpm,
        'words': int(words),
        'goal': goal,
        'challenge': challenge,
        'drama': drama,
        'resolution': resolution,
        'one_thing': draft.get('next_rep') or one_thing,
        'why': why,
        'short_mode': bool(draft.get('short_mode')),
        'one_big_thing': draft.get('one_big_thing') or '',
        'why_it_matters': draft.get('why_it_matters') or '',
        'yes_but': draft.get('yes_but') or '',
        'next_rep': draft.get('next_rep') or '',
        'said_quote': draft.get('said_quote') or '',
        'transitions': transitions,
        'session_start_ms': session_start_ms,
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
        'role': info.get('role') or 'participant',
        'handed_to': _handed_to(transcript, label),
        'roster_shares': _roster_shares(stats_frame),
        'appendix': _appendix(framework, name),
    }


def _roster_shares(frame) -> list[tuple[str, float]]:
    if frame is None or 'Speaking Time (%)' not in getattr(frame, 'columns', []):
        return []
    shares = []
    for person, value in frame['Speaking Time (%)'].items():
        if str(person) == 'Average':
            continue
        shares.append((str(person), float(value or 0)))
    return shares


def _handed_to(transcript: dict | None, label: str) -> str:
    """Top three people this speaker handed the turn to."""
    rows = (transcript or {}).get('utterances') or []
    counts: dict[str, int] = {}
    for index, row in enumerate(rows[:-1]):
        if row.get('speaker') != label:
            continue
        nxt = rows[index + 1]
        if nxt.get('speaker') == label:
            continue
        name = str(nxt.get('name') or nxt.get('speaker') or '').strip()
        if not name:
            continue
        counts[name] = counts.get(name, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:3]
    if not ranked:
        return ''
    listed = ', '.join(f'{name} {count}' for name, count in ranked)
    return f'Handed to: {listed}.'


def _appendix(framework: dict, name: str) -> dict:
    """AMASE stays. Enneagram and Doshas are opt-in and off unless asked for."""
    appendix = {
        'AMASE': _top_framework(framework.get('amase_values'), framework.get('amase_notes'), name),
    }
    if framework.get('include_personality'):
        appendix['Enneagram'] = _top_framework(
            framework.get('enneagram_values'), framework.get('enneagram_notes'), name
        )
        appendix['Doshas'] = _top_framework(framework.get('dosha_values'), framework.get('dosha_notes'), name)
    return appendix


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


def _duration_shares(transcript, names) -> list[tuple[str, int]]:
    totals: dict[str, float] = {}
    for name in (names or {}).values():
        if name:
            totals.setdefault(str(name), 0.0)
    for row in (transcript or {}).get('utterances') or []:
        who = row.get('name') or (names or {}).get(row.get('speaker')) or row.get('speaker')
        if not who:
            continue
        span = max(0.0, float(row.get('end') or 0) - float(row.get('start') or 0))
        totals[str(who)] = totals.get(str(who), 0.0) + span
    total = sum(totals.values()) or 1
    return [(name, int(round(100 * value / total))) for name, value in totals.items()]


def _spoke_count(transcript, names) -> int:
    roster = {str(name) for name in (names or {}).values() if name}
    spoken = set()
    for row in (transcript or {}).get('utterances') or []:
        if not str(row.get('text') or '').strip():
            continue
        who = row.get('name') or (names or {}).get(row.get('speaker')) or row.get('speaker')
        if who:
            spoken.add(str(who))
    if roster:
        return len(spoken & roster)
    return len(spoken)


def group_story(
    *,
    summaries: dict,
    participation: float,
    interaction: float,
    word_interaction: float,
    cast: str = '',
    draft: dict | None = None,
    transcript: dict | None = None,
    names: dict | None = None,
    pairs: list | None = None,
    shares: list | None = None,
) -> dict:
    from lib.session_window import closer_turn
    from lib.story_render import (
        closing_clip,
        group_character_copy,
        group_halfway_target,
        narrative_problems,
    )

    interaction_pct = int(round(100 * interaction))
    participation_pct = int(round(100 * participation))
    word_pct = int(round(100 * word_interaction))
    target, norm, norm_name = group_halfway_target(interaction_pct, participation_pct, word_pct)
    headline = (
        'The turns stayed with a few voices'
        if interaction_pct < 50
        else 'The room kept handing the turn across'
    )
    portrait = {}
    if pairs:
        portrait['pair'] = (pairs[0][0], pairs[0][1])
    if shares:
        low = min(shares, key=lambda item: item[1])
        portrait['least'] = low[0]
        portrait['low_share'] = min(item[1] for item in shares)
        portrait['high_share'] = max(item[1] for item in shares)
    facts = {
        'interaction': {'value': interaction_pct},
        'participation': {'value': participation_pct},
        '_portrait': portrait,
    }
    last_name = 'the last voice'
    last_line = ''
    closing_row = closer_turn((transcript or {}).get('utterances') or [])
    if closing_row:
        last_line = closing_clip(str(closing_row.get('text') or '')) or ''
        last_name = (names or {}).get(closing_row.get('speaker')) or last_name
    if last_line:
        facts['_closing'] = {'name': last_name, 'text': last_line}
    character = group_character_copy(facts)
    topics = _meeting_topics(transcript)
    if len(topics) >= 2:
        listed = ', '.join(topics[:-1]) + f', and {topics[-1]}'
        goal = f'The turns keep returning to {listed}.'
    elif topics:
        goal = f'The turns keep returning to {topics[0]}.'
    else:
        goal = 'The turns are the record of the meeting, and a point could finish inside one voice.'
    challenge = (
        'A point could finish inside one voice before anyone new answered. '
        'What would change if the next reply came from across the room?'
    )
    drama = cast or 'Who talks to whom, and who nobody answered.'
    if last_line:
        resolution = (
            f'The meeting closed on {last_name}, who said "{last_line}". '
            'That last line had to land with someone new.'
        )
    else:
        resolution = (
            'The meeting closed with the exchange still inside the same voices, '
            'so the last turn did not have to travel to a new voice.'
        )
    from lib.questions import question_counts
    from lib.story_render import group_caveat, group_pattern_why, group_plain_numbers

    if not shares:
        shares = _duration_shares(transcript, names)
    roster_names = {str(name) for name in (names or {}).values() if name}
    speaker_count = len(roster_names) or len(shares or []) or len({
        row.get('speaker') for row in (transcript or {}).get('utterances') or [] if row.get('speaker')
    })
    one_thing = 'Ask one quiet person to speak before the next choice.'
    why = 'Next time, leave that reply for them.'
    from lib.story_render import pick_transitions

    transitions = pick_transitions('group', group=True)
    draft = draft or {}
    if draft.get('headline'):
        headline = draft['headline']
    draft_character = str(draft.get('character') or '')
    if draft_character and not narrative_problems(draft_character, big=True):
        character = draft_character
    goal = draft.get('goal') or goal
    challenge = draft.get('challenge') or challenge
    drama = draft.get('drama') or drama
    resolution = draft.get('resolution') or resolution
    draft_one = str(draft.get('one_thing') or '')
    draft_why = str(draft.get('why') or '')
    if draft_one and not narrative_problems(draft_one):
        one_thing = draft_one
    if draft_why and not narrative_problems(draft_why):
        why = draft_why
    if draft.get('transitions'):
        transitions = draft['transitions']
    portrait = facts.get('_portrait') or {}
    return {
        'headline': headline,
        'character': character,
        'goal': goal,
        'challenge': challenge,
        'drama': drama,
        'resolution': resolution,
        'one_thing': one_thing,
        'why': why,
        'pattern_why': group_pattern_why(portrait),
        'caveat': group_caveat(portrait),
        'transitions': transitions,
        'numbers': group_plain_numbers(
            interaction_pct,
            speaker_count,
            spoke=_spoke_count(transcript, names),
            roster=speaker_count,
        ),
        'shares': list(shares or []),
        'question_counts': question_counts(transcript, names),
        'proposal_counts': draft.get('proposal_counts'),
    }
