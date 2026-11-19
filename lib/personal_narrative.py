"""Personal narrative: code picks the evidence, one call writes the sections."""

from __future__ import annotations

import json
import logging
import os
import re

from lib.readability import reading_grade
from lib.story_render import (
    question_number_line,
    room_average,
    room_median,
    session_talk_phrase,
    talk_number_line,
    turns_sentence,
)

PROMPT_MARK = 'PERSONAL_NARRATIVE'
_DANGLING = {
    'to', 'and', 'the', 'a', 'an', 'of', 'for', 'with', 'in', 'on', 'at',
    'from', 'by', 'or', 'as', 'if', 'that', 'so',
}
_FUNCTION = {
    'the', 'a', 'an', 'of', 'for', 'with', 'in', 'on', 'at', 'from', 'by', 'or',
    'as', 'if', 'that', 'so', 'to', 'and', 'we', 'you', 'i', 'it', 'is', 'are',
    'was', 'were', 'be', 'been', 'will', 'would', 'could', 'should', 'all', 'us',
    'our', 'probably', 'just', 'really', 'very', 'well', 'yeah', 'yes', 'right',
    'think', 'know', 'mean', 'like', 'this', 'these', 'those',
}
_FILLER_PREFIX = re.compile(
    r'^(?:(?:well|so|okay|ok|yeah|yes|right|um+|uh+)\s*,?\s*|(?:i think|you know|i mean)\s*,?\s*)+',
    re.I,
)
_BANNED = (
    'interaction', 'participation', 'talk time', 'focused dialogue',
    'this is the part to change', 'who else should speak about',
    'you name ',
)


def _words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z']+", text or '')


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.findall(r'[^.!?]*[.!?]', text or '') if part.strip()]


def good_sentence(sentence: str) -> bool:
    """A usable sentence is complete, at least six words, and does not end on 'to.'"""
    words = _words(sentence)
    if len(words) < 6 or not sentence.endswith(('.', '!', '?')):
        return False
    return words[-1].lower() not in _DANGLING


def _content_words(sentence: str) -> list[str]:
    return [word for word in _words(sentence) if word.lower() not in _FUNCTION and len(word) > 3]


def _strip_filler(sentence: str) -> str:
    cleaned = _FILLER_PREFIX.sub('', sentence or '').strip()
    if cleaned[:1].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned


def candidate_quotes(turns: list[str]) -> list[str]:
    """Up to three quotes. Each one is a single sentence copied from the turn."""
    scored: list[tuple[int, str]] = []
    for turn in turns:
        for sentence in _sentences(turn):
            if sentence not in turn or not good_sentence(sentence):
                continue
            if re.search(r'\ball of us will probably agree\b', sentence, re.I):
                continue
            content = _content_words(sentence)
            # "All of us will probably agree" has no concrete claim. A real noun keeps it.
            if len(content) < 2:
                continue
            # Filler can stay in the words we print. It only loses the ranking.
            score = len(content)
            if _FILLER_PREFIX.match(sentence):
                score -= 2
            scored.append((score, sentence))
    scored.sort(key=lambda item: (-item[0], len(item[1])))
    found: list[str] = []
    for _score, option in scored:
        if option not in found:
            found.append(option)
        if len(found) == 3:
            break
    return found


_MOMENT_SKIP = _FUNCTION | {
    'about', 'after', 'before', 'there', 'their', 'which', 'where', 'while', 'being',
    'other', 'every', 'still', 'those', 'first', 'going', 'gonna', 'people', 'really',
    'thing', 'would', 'could', 'should', 'something', 'anything', 'please', 'thank',
}


def moment_word(quotes: list[str], rows: list[dict]) -> str:
    """One concrete word from this person's own words, so the next step is not shared."""
    source = ' '.join(quotes) if quotes else ' '.join(str(row.get('text') or '') for row in rows)
    best = ''
    for word in re.findall(r"[A-Za-z']{5,}", source):
        token = word.lower()
        if token in _MOMENT_SKIP:
            continue
        if len(token) > len(best):
            best = token
    return best


def suggested_next_rep(rows: list[dict], stats: dict, moment: str = '') -> str:
    """A full script the person can say. Not 'ask about' plus one bare word."""
    turns = len(rows)
    words = word_total(rows)
    share = int(stats.get('share') or 0)
    speakers = max(int(stats.get('speaker_count') or 1), 1)
    questions = int(stats.get('questions') or 0)
    topic = moment if moment and not moment.endswith('ing') else 'that point'
    if turns <= 2 and words >= 60:
        return f'Say: "I will stop after two sentences about {topic}, then ask who has tried that."'
    if share > room_average(speakers) + 2:
        return f'Say: "I will make one point about {topic}, then hand it to someone else."'
    if questions == 0:
        return f'Say: "Before I add to {topic}, I want to ask who has tried that."'
    if turns < 3:
        return f'Say: "I will add one short point about {topic}, then ask what you would change."'
    return f'Say: "I will pause after {topic} and ask who wants to answer."'


def speech_headline(rows: list[dict]) -> str:
    """What they actually did. A long monologue is not 'a short turn'."""
    count = len(rows)
    if count <= 1:
        opening = 'You spoke once'
    elif count == 2:
        opening = 'You spoke twice'
    else:
        opening = f'You spoke {count} times'
    seconds = 0.0
    for row in rows:
        seconds += max(0.0, float(row.get('end') or 0) - float(row.get('start') or 0))
    if seconds >= 1000:
        seconds /= 1000.0
    elif seconds <= 0:
        seconds = word_total(rows) / 130 * 60
    if seconds < 50:
        span = 'under a minute'
    elif seconds < 90:
        span = 'about a minute'
    else:
        minutes = max(2, int(round(seconds / 60)))
        span = 'about two minutes' if minutes == 2 else f'about {minutes} minutes'
    return f'{opening}, for {span}.'


def speaker_rows(transcript: dict | None, label: str) -> list[dict]:
    return [
        row for row in (transcript or {}).get('utterances') or []
        if row.get('speaker') == label and str(row.get('text') or '').strip()
    ]


def word_total(rows: list[dict]) -> int:
    return sum(len(_words(str(row.get('text') or ''))) for row in rows)


def substantive_count(rows: list[dict]) -> int:
    count = 0
    for row in rows:
        if any(good_sentence(sentence) for sentence in _sentences(str(row.get('text') or ''))):
            count += 1
    return count


def needs_short_report(rows: list[dict]) -> bool:
    """Only a small amount of speech. A long monologue is still a full report."""
    return word_total(rows) < 60


def turn_counts(transcript: dict | None) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in (transcript or {}).get('utterances') or []:
        if not str(row.get('text') or '').strip():
            continue
        label = str(row.get('speaker') or '')
        if not label:
            continue
        counts[label] = counts.get(label, 0) + 1
    return counts


def computed_numbers(transcript, label: str, stats: dict) -> tuple[list[str], set[str]]:
    """Number lines built only from counts. The median is the room's own middle value."""
    counts = turn_counts(transcript)
    mine = counts.get(label, int(stats.get('turns') or 0))
    median = room_median(counts.values()) if counts else int(stats.get('turns') or 0)
    share = int(stats.get('share') or 0)
    speakers = int(stats.get('speaker_count') or len(counts) or 1)
    questions = int(stats.get('questions') or 0)
    average = int(stats.get('session_question_average') or 0)
    span = session_talk_phrase(transcript)
    own_words = word_total(speaker_rows(transcript, label))
    lines = [
        talk_number_line(share, speakers, span),
        question_number_line(questions, average),
        turns_sentence(mine, median if median is not None else mine, own_words if mine == 1 else None),
    ]
    allowed = {
        str(share), str(room_average(speakers)), str(mine), str(median if median is not None else mine),
        str(questions), str(average),
    }
    return lines, allowed


def _prompt(name: str, lines: list[str], quotes: list[str], failures: list[str], phrase: str, next_rep: str) -> str:
    quote_block = '\n'.join(f'Q{index + 1}: {quote}' for index, quote in enumerate(quotes)) or 'None.'
    number_block = '\n'.join(f'- {line}' for line in lines)
    retry = ''
    if failures:
        retry = 'The last draft failed:\n' + '\n'.join(f'- {item}' for item in failures) + '\n'
    return (
        f'{PROMPT_MARK}\n'
        f'Write the personal report for {name}. Use only the evidence below.\n'
        'Style: fifth-grade words, short sentences, second person.\n'
        'Headline: that person\'s finding, 6 words or fewer. Never "Here is your personal report."\n'
        'one_big_thing: one behavior in plain words, then a question that invites a change. '
        'Do not type a quote. Put the id in one_big_thing_quote, such as "Q2". Code inserts the words.\n'
        'Good: You told one long story with lots of detail. What if you stopped after the NBC example and asked who had tried that?\n'
        'Good: You answered and then kept going. What if you stopped after one point and asked Leslie what she would add?\n'
        'Bad: You repeated the phrase a niche. Who can speak Farsi?\n'
        'Bad: Here is your personal report. You said a lot. Who else should speak?\n'
        'why_it_matters: name this person\'s moment. Not generic praise such as "Your voice brings good ideas to the team."\n'
        'yes_but: second person, one specific limit.\n'
        'next_rep: a full script line, as in Say: "I will stop after two sentences, then ask who has tried that." '
        'Never "ask about" plus one bare word. A script may say I inside the quote marks.\n'
        'said_quote: a quote id such as "Q2", or "" if there is no Q2. Do not retype the words.\n'
        'Use at most one number in a sentence. Never name a metric.\n'
        f'{retry}'
        f'NUMBERS\n{number_block}\n'
        f'QUOTES\n{quote_block}\n'
        f'REPEATED PHRASE\n{phrase or "None."}\n'
        f'SUGGESTED NEXT STEP\n{next_rep}\n'
        'Return JSON only with keys headline, one_big_thing, one_big_thing_quote, why_it_matters, yes_but, next_rep, said_quote.\n'
    )


def _quote_by_id(token: str, quotes: list[str], default_index: int) -> str:
    """Q2 means the second candidate. A retyped sentence is not used."""
    raw = str(token or '').strip().strip('"').strip('“”')
    match = re.fullmatch(r'Q([1-3])', raw, re.I)
    if match:
        index = int(match.group(1)) - 1
        return quotes[index] if 0 <= index < len(quotes) else ''
    if raw in quotes:
        return raw
    if 0 <= default_index < len(quotes):
        return quotes[default_index]
    return ''


def _strip_typed_quotes(text: str) -> str:
    cleaned = re.sub(r'(?:"[^"]*"|“[^”]*”)', ' ', text or '')
    return re.sub(r'\s+', ' ', cleaned).strip()


def _place_quote(text: str, quote: str) -> str:
    """Pattern, then the verbatim quote, then the question. The model does not type it."""
    sentences = [part.strip() for part in re.split(r'(?<=[.!?])\s+', _strip_typed_quotes(text)) if part.strip()]
    sentences = [re.sub(r'\s+you said$', '', part, flags=re.I).strip() for part in sentences]
    sentences = [part for part in sentences if part]
    spoken = f'You said "{quote}"'
    if not sentences:
        return spoken
    if len(sentences) == 1:
        return f'{sentences[0]} {spoken}'
    return f'{sentences[0]} {spoken} ' + ' '.join(sentences[1:])


def apply_quote_ids(payload: dict, quotes: list[str]) -> dict:
    """Replace quote ids with the candidate text. Drop anything the model retyped."""
    data = dict(payload or {})
    big = _quote_by_id(str(data.get('one_big_thing_quote') or ''), quotes, 0)
    said_default = 1 if len(quotes) > 1 else 0
    said = _quote_by_id(str(data.get('said_quote') or ''), quotes, said_default if quotes else -1)
    data['one_big_thing'] = _place_quote(str(data.get('one_big_thing') or ''), big) if big else _strip_typed_quotes(str(data.get('one_big_thing') or ''))
    data['said_quote'] = said
    for key in ('why_it_matters', 'yes_but'):
        data[key] = _strip_typed_quotes(str(data.get(key) or ''))
    return data


def _outside_quotes(text: str) -> str:
    """First person and number checks ignore a quoted script, straight or curly."""
    return re.sub(r'(?:"[^"]*"|“[^”]*”)', ' ', text or '')


def _numbers_outside(sentence: str) -> list[str]:
    outside = _outside_quotes(sentence)
    outside = re.sub(r'\b\d+:\d{2}(?::\d{2})?\b', ' ', outside)
    return re.findall(r'\d+', outside)


_FIRST_PERSON = re.compile(r"\b(I|I'll|I'm|I've|I'd|I’ll|I’m|I’ve|I’d)\b")
_IMPERATIVE = re.compile(r'\b(stop|ask|name|say|pause|invite|hand|add|let|leave|pass)\b', re.I)


def _quote_is_verbatim(span: str, quotes: list[str]) -> bool:
    """One contiguous sentence that appears inside a candidate. A splice is not."""
    cleaned = (span or '').strip()
    if not cleaned or len(_sentences(cleaned)) != 1:
        return False
    return any(cleaned == quote or cleaned in quote for quote in quotes)


def section_problems(payload: dict, quotes: list[str], allowed: set[str], moment: str = '') -> dict[str, list[str]]:
    """Problems by section. Number lines are computed in code and are not judged here."""
    found: dict[str, list[str]] = {}
    if not isinstance(payload, dict):
        return {'one_big_thing': ['response was not an object']}

    def add(name: str, reason: str) -> None:
        found.setdefault(name, []).append(reason)

    headline = str(payload.get('headline') or '').strip()
    if (
        not headline
        or len(headline.split()) > 6
        or reading_grade(headline) > 6
        or 'focused dialogue' in headline.lower()
        or 'here is your personal report' in headline.lower()
    ):
        add('headline', 'headline must state a finding in 6 words or fewer')
    fields = {
        'one_big_thing': str(payload.get('one_big_thing') or ''),
        'why_it_matters': str(payload.get('why_it_matters') or ''),
        'yes_but': str(payload.get('yes_but') or ''),
        'next_rep': str(payload.get('next_rep') or ''),
    }
    if '?' not in _outside_quotes(fields['one_big_thing']):
        add('one_big_thing', 'one_big_thing needs a question')
    elif not re.search(r'\b(what if|will you|who will|stop|ask|tried|change)\b', fields['one_big_thing'], re.I):
        add('one_big_thing', 'the question must invite a change in what you do')
    if re.search(r'repeated the phrase|here is your personal report', fields['one_big_thing'], re.I):
        add('one_big_thing', 'one_big_thing must describe a behavior, not a label')
    why = fields['why_it_matters'].lower()
    if any(phrase in why for phrase in ('your voice brings', 'good ideas', 'brings good ideas', 'to the team')):
        add('why_it_matters', 'why_it_matters is generic praise')
    if moment and moment not in why:
        add('why_it_matters', "why_it_matters does not use this person's moment")
    for name, text in fields.items():
        plain = _outside_quotes(text)
        if reading_grade(text) > 6:
            add(name, f'{name} is above a sixth-grade reading level')
        lowered = plain.lower()
        for phrase in _BANNED:
            if phrase in lowered:
                add(name, f'{name} uses {phrase}')
        if _FIRST_PERSON.search(plain):
            add(name, f'{name} has an unquoted first-person line')
        # One number per sentence. The number lines are not part of this text.
        for sentence in re.split(r'(?<=[.!?])\s+', text.strip()):
            if len(_numbers_outside(sentence)) > 1:
                add(name, f'{name} has two numbers in one sentence')
        if name == 'next_rep':
            continue
        for span in re.findall(r'(?:"([^"]*)"|“([^”]*)”)', text):
            quoted = span[0] or span[1]
            if not _quote_is_verbatim(quoted, quotes):
                add(name, f'{name} quotes text that is not one verbatim sentence')
    next_rep = fields['next_rep']
    if next_rep and (
        re.search(r'\bask about \w+\b', next_rep, re.I)
        or (not _IMPERATIVE.search(next_rep) and 'next time' not in next_rep.lower())
        or len(_words(next_rep)) < 8
    ):
        add('next_rep', 'next_rep must be a full script line')
    next_numbers = _numbers_outside(next_rep)
    if any(number not in allowed for number in next_numbers):
        add('next_rep', 'next_rep uses a number that was not computed')
    said = str(payload.get('said_quote') or '').strip().strip('"').strip('“”')
    expected = quotes[1] if len(quotes) > 1 else ''
    if said != expected:
        add('said_quote', 'said_quote must be the second candidate')
    return found


def validate_narrative(payload: dict, quotes: list[str], allowed: set[str], moment: str = '') -> list[str]:
    found = section_problems(payload, quotes, allowed, moment)
    return [reason for reasons in found.values() for reason in reasons]


def _parse(raw) -> dict | None:
    content = raw.content if hasattr(raw, 'content') else raw
    if isinstance(content, list):
        content = ' '.join(str(part) for part in content)
    text = str(content or '').strip()
    if text.startswith('```'):
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def short_story(name: str, rows: list[dict], lines: list[str], quotes: list[str], stats: dict) -> dict:
    count = len(rows)
    quote = quotes[0] if quotes else ''
    step = suggested_next_rep(rows, stats, moment_word(quotes, rows))
    return {
        'short_mode': True,
        'headline': speech_headline(rows),
        'one_big_thing': '',
        'why_it_matters': '',
        'yes_but': '',
        'next_rep': step,
        'numbers': lines,
        'said_quote': quote,
        'big_quote': quote,
        'drama': quote,
        'goal': '',
        'challenge': '',
        'one_thing': step,
        'why': '',
        'turns': count,
        'fallback_sections': [],
        'name_seed': name,
    }


def full_story(payload: dict, lines: list[str], name: str) -> dict:
    one_big = str(payload.get('one_big_thing') or '').strip()
    why = str(payload.get('why_it_matters') or '').strip()
    yes_but = str(payload.get('yes_but') or '').strip()
    next_rep = str(payload.get('next_rep') or '').strip()
    return {
        'short_mode': False,
        'headline': str(payload.get('headline') or '').strip(),
        'one_big_thing': one_big,
        'why_it_matters': why,
        'yes_but': yes_but,
        'next_rep': next_rep,
        'numbers': lines,
        'said_quote': str(payload.get('said_quote') or '').strip().strip('"'),
        'big_quote': '',
        'drama': one_big,
        'goal': why,
        'challenge': yes_but,
        'one_thing': next_rep,
        'why': '',
        'fallback_sections': [],
        'name_seed': name,
    }


def _fallback_line(section: str, rows, lines, quotes, stats, moment: str) -> str:
    if section == 'headline':
        return speech_headline(rows)
    if section == 'why_it_matters':
        topic = moment or 'that point'
        if len(rows) <= 2 and word_total(rows) >= 60:
            return f'A long story about {topic} left no room for a reply. Next time, stop and ask what they tried.'
        return f'The room heard {topic} and then moved on. Next time, ask one person what they would change.'
    if section == 'yes_but':
        return 'You made the point in one stretch. Next time, leave room for a reply.'
    if section == 'next_rep':
        return suggested_next_rep(rows, stats, moment)
    if section == 'said_quote':
        return quotes[1] if len(quotes) > 1 else (quotes[0] if quotes else '')
    return ''


def _record_attempt(progress_dir, label: str, raw: str, reasons: list[str]) -> None:
    """Raw personal responses stay in the local checkpoint file, mode 600."""
    if not progress_dir:
        return
    from pathlib import Path

    directory = Path(progress_dir)
    directory.mkdir(parents=True, exist_ok=True)
    os.chmod(directory, 0o700)
    path = directory / 'checkpoints.json'
    existing = []
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding='utf-8'))
        except json.JSONDecodeError:
            existing = []
    if not isinstance(existing, list):
        existing = []
    existing.append({'section': f'personal:{label}', 'raw': raw, 'reasons': reasons})
    path.write_text(json.dumps(existing, indent=2), encoding='utf-8')
    os.chmod(path, 0o600)


THINKING_BUDGET = 512


def personal_model_name() -> str:
    """Non-lite flash for the personal call. gemini-3.5-flash is deprecated."""
    return os.getenv('GEMINI_PERSONAL_MODEL', 'gemini-3.6-flash')


def personal_client_settings(llm) -> dict:
    """A low thinking budget. The 3.5 run spent almost all of its cost on thinking."""
    return {
        'model': personal_model_name(),
        'temperature': 0,
        'timeout': getattr(llm, 'timeout', 120),
        'max_retries': 2,
        'thinking_budget': THINKING_BUDGET,
    }


def _model_not_found(exc: BaseException) -> bool:
    """A 404 from the personal model. Other failures stay failures."""
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if getattr(current, 'status_code', None) == 404 or getattr(current, 'code', None) == 404:
            return True
        if type(current).__name__.lower() == 'notfound':
            return True
        text = str(current).lower()
        if '404' in text and 'not found' in text:
            return True
        current = current.__cause__ or current.__context__
    return False


def for_personal_narrative(llm):
    """Personal narrative uses non-lite flash. Every other call keeps flash-lite.

    The default is gemini-3.6-flash. A 404 still falls back to flash-lite.
    GEMINI_PERSONAL_MODEL can name a different id. A test double is returned unchanged.
    """
    model = getattr(llm, 'model', None)
    if not isinstance(model, str) or 'lite' not in model:
        return llm
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError:
        return llm
    if not isinstance(llm, ChatGoogleGenerativeAI):
        return llm
    target = personal_model_name()
    if target == model:
        return llm
    kwargs = personal_client_settings(llm)
    key = getattr(llm, 'google_api_key', None)
    if key:
        kwargs['google_api_key'] = key
    return ChatGoogleGenerativeAI(**kwargs)


def build_personal_story(llm, transcript, label: str, info: dict, stats: dict, progress_dir=None) -> dict:
    """One call when there is enough speech. A failed 1 big thing becomes the short report."""
    fallback = llm
    llm = for_personal_narrative(llm)
    used_fallback = llm is fallback
    rows = speaker_rows(transcript, label)
    quotes = candidate_quotes([str(row.get('text') or '') for row in rows])
    lines, allowed = computed_numbers(transcript, label, stats)
    name = info.get('name') or label
    moment = moment_word(quotes, rows)
    step = suggested_next_rep(rows, stats, moment)
    phrase = _repeated_phrase(rows)
    if needs_short_report(rows):
        return short_story(name, rows, lines, quotes, stats)
    failures: list[str] = []
    best_payload = None
    best_found: dict[str, list[str]] = {'one_big_thing': ['response was not JSON']}
    attempts = 0
    while attempts < 2:
        raw_text = ''
        try:
            raw_text = _raw_text(llm.invoke(_prompt(name, lines, quotes, failures, phrase, step)))
            payload = _parse(raw_text)
            if payload is not None:
                payload = apply_quote_ids(payload, quotes)
        except Exception as exc:
            if not used_fallback and _model_not_found(exc):
                logging.warning(
                    'personal model %s returned 404; falling back to %s',
                    getattr(llm, 'model', None) or personal_model_name(),
                    getattr(fallback, 'model', None) or 'flash-lite',
                )
                llm = fallback
                used_fallback = True
                continue
            logging.warning('personal narrative call failed: %s', exc)
            payload = None
            raw_text = raw_text or str(exc)
        attempts += 1
        if payload is None:
            found = {'one_big_thing': ['response was not JSON']}
        else:
            found = section_problems(payload, quotes, allowed, moment)
        reasons = [reason for items in found.values() for reason in items]
        _record_attempt(progress_dir, label, raw_text, reasons)
        if payload is not None and not reasons:
            story = full_story(payload, lines, name)
            story['fallback_sections'] = []
            return story
        if payload is not None and (best_payload is None or len(found) < len(best_found)):
            best_payload = payload
            best_found = found
        failures = reasons
    if best_payload is None or best_found.get('one_big_thing'):
        logging.warning('personal narrative fell back to the short report: %s', failures)
        story = short_story(name, rows, lines, quotes, stats)
        story['fallback_sections'] = ['one_big_thing']
        story['rejection_reasons'] = {'one_big_thing': '; '.join(best_found.get('one_big_thing') or failures)}
        return story
    patched = dict(best_payload)
    replaced = []
    for section, reasons in best_found.items():
        if section == 'one_big_thing' or not reasons:
            continue
        patched[section] = _fallback_line(section, rows, lines, quotes, stats, moment)
        replaced.append(section)
    story = full_story(patched, lines, name)
    story['fallback_sections'] = replaced
    story['rejection_reasons'] = {section: '; '.join(best_found[section]) for section in replaced}
    story['short_mode'] = False
    return story


def _raw_text(raw) -> str:
    content = raw.content if hasattr(raw, 'content') else raw
    if isinstance(content, list):
        content = ' '.join(str(part) for part in content)
    return str(content or '')


def _repeated_phrase(rows: list[dict]) -> str:
    from lib.report_stats import display_bigram, top_bigram

    found = top_bigram(' '.join(str(row.get('text') or '') for row in rows))
    if not found:
        return ''
    phrase, count = found
    return f'{display_bigram(phrase)} ({count})'
