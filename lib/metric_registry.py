"""Swappable report metrics.

A metric is one function registered under a key. A profile (report_profiles.json)
names which keys fill the tiles for a kind of session, which roles get a report,
and what each number is compared with. Adding a metric means adding one function
here; changing a report means editing JSON, not code.

Comparison order for every tile:
  1. the person's own baseline (Rep 1, or their last session), when one is passed in
  2. the profile's target band for their role, when one is set
  3. the middle of the room, only among people in the same role and only with
     four or more of them
Otherwise the tile shows the number alone. No comparison is better than a fake one.

Three kinds of metric:
  counted  from transcript text
  timed    from utterance start and end times
  coded    needs a model or an observer to read the turn (Four Beats, reflections).
           Pass those values in through `coded`; without them the tile is left out.
"""

from __future__ import annotations

import json
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

PROFILE_PATH = Path(__file__).resolve().parents[1] / 'report_profiles.json'

FILLER_WORDS = {'um', 'uh', 'er', 'ah', 'erm', 'hmm', 'uhm', 'umm'}
FILLER_PHRASES = ('you know', 'i mean', 'sort of', 'kind of')
HEDGES = (
    'i think', 'i guess', 'i feel like', 'maybe', 'probably', 'perhaps', 'might',
    'a little bit', 'if that makes sense', 'does that make sense', 'i just',
)
OPEN_STARTS = ('what', 'how', 'why')
OPEN_PHRASES = ('tell me', 'walk me', 'describe', 'say more', 'help me understand', 'talk me through')
CLOSED_STARTS = {
    'is', 'are', 'am', 'was', 'were', 'do', 'does', 'did', 'can', 'could', 'would',
    'should', 'will', 'shall', 'have', 'has', 'had', 'who', 'when', 'where', 'which',
}
TAG_QUESTIONS = re.compile(
    r"^(right|okay|ok|yeah|you know|no|correct|does that make sense|make sense|you know what i mean)\?$",
    re.I,
)
# 'like' as a filler, not a verb or a comparison: after a comma or a linking
# word, or before a comma. An estimate, and labelled as one in the report.
_LIKE = re.compile(r"(?:(?<=,\s)|(?<=\band\s)|(?<=\bso\s)|(?<=\bbut\s)|(?<=\bwas\s)|(?<=\bjust\s))like\b|\blike(?=,)", re.I)


# ---------------------------------------------------------------- session

@dataclass
class Turn:
    label: str
    text: str
    start: float
    end: float

    @property
    def ms(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def words(self) -> list[str]:
        return re.findall(r"[A-Za-z']+", self.text.lower())


class Session:
    """One transcript, with the per-person slices every metric needs."""

    def __init__(self, transcript: dict, names: dict[str, str] | None = None, roles: dict[str, str] | None = None):
        self.turns = [
            Turn(str(u.get('speaker')), str(u.get('text') or ''), float(u.get('start') or 0), float(u.get('end') or 0))
            for u in (transcript or {}).get('utterances') or []
            if u.get('speaker') not in (None, 'UNK')
        ]
        self.labels = sorted({t.label for t in self.turns})
        self.names = {label: (names or {}).get(label) or f'Speaker {label}' for label in self.labels}
        self.roles = {label: (roles or {}).get(label) or 'participant' for label in self.labels}
        self.talk_ms = sum(t.ms for t in self.turns)
        self.length_ms = (max(t.end for t in self.turns) - min(t.start for t in self.turns)) if self.turns else 0.0

    def of(self, label: str) -> list[Turn]:
        return [t for t in self.turns if t.label == label]

    def text(self, label: str) -> str:
        return ' '.join(t.text for t in self.of(label))

    def word_count(self, label: str) -> int:
        return sum(len(t.words) for t in self.of(label))


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n+', text or '') if s.strip()]


def classify_questions(text: str) -> dict[str, int]:
    """Open, closed and check ('right?') questions in one stretch of speech."""
    out = {'open': 0, 'closed': 0, 'check': 0}
    for sentence in _sentences(text):
        if not sentence.endswith('?'):
            continue
        if TAG_QUESTIONS.match(sentence):
            out['check'] += 1
            continue
        lowered = sentence.lower()
        # Skip a lead-in ('So,', 'And then,') to find the word that makes it a question.
        words = re.findall(r"[a-z']+", lowered)
        while words and words[0] in {'so', 'and', 'but', 'okay', 'ok', 'well', 'now', 'then', 'um', 'uh'}:
            words.pop(0)
        first = words[0] if words else ''
        if first in OPEN_STARTS or any(p in lowered for p in OPEN_PHRASES):
            out['open'] += 1
        elif first in CLOSED_STARTS:
            out['closed'] += 1
        else:
            out['closed'] += 1  # statement with a rising end: it invites yes or no
    return out


def _per_100_words(count: int, words: int) -> float | None:
    return round(100 * count / words, 1) if words else None


def _gini(values: list[float]) -> float:
    values = sorted(v for v in values if v >= 0)
    total = sum(values)
    if not values or total == 0:
        return 0.0
    n = len(values)
    cum = sum((i + 1) * v for i, v in enumerate(values))
    return (2 * cum) / (n * total) - (n + 1) / n


# ---------------------------------------------------------------- registry

@dataclass(frozen=True)
class Metric:
    key: str
    label: str
    pillar: str          # clarity | composure | connection | participation
    unit: str            # pct | clock | count | rate | wpm | seconds | score
    kind: str            # counted | timed | coded
    scope: str           # person | group
    fn: Callable | None
    meaning: str         # one plain line for the report footnote


REGISTRY: dict[str, Metric] = {}


def metric(key, label, pillar, unit, kind='counted', scope='person', meaning=''):
    def wrap(fn):
        REGISTRY[key] = Metric(key, label, pillar, unit, kind, scope, fn, meaning)
        return fn
    return wrap


def coded(key, label, pillar, unit, meaning):
    REGISTRY[key] = Metric(key, label, pillar, unit, 'coded', 'person', None, meaning)


# ---- participation

@metric('talk_share', 'Talk share', 'participation', 'pct', 'timed', meaning='Your speaking time as a share of all speaking time.')
def _talk_share(s: Session, label: str):
    return round(100 * sum(t.ms for t in s.of(label)) / s.talk_ms, 1) if s.talk_ms else None


@metric('talk_time', 'Talk time', 'participation', 'clock', 'timed', meaning='Total time you held the floor.')
def _talk_time(s, label):
    return sum(t.ms for t in s.of(label))


@metric('turns', 'Turns', 'participation', 'count', meaning='How many times you took the floor.')
def _turns(s, label):
    return len(s.of(label))


@metric('words', 'Words', 'participation', 'count', meaning='Words you said.')
def _words(s, label):
    return s.word_count(label)


# ---- clarity

@metric('longest_turn', 'Longest turn', 'clarity', 'clock', 'timed', meaning='Your longest stretch without anyone else speaking.')
def _longest_turn(s, label):
    return max((t.ms for t in s.of(label)), default=None)


@metric('avg_turn', 'Typical turn', 'clarity', 'seconds', 'timed', meaning='Median length of your turns.')
def _avg_turn(s, label):
    spans = [t.ms / 1000 for t in s.of(label)]
    return round(statistics.median(spans), 1) if spans else None


@metric('filler_rate', 'Filler words', 'clarity', 'rate', meaning="um, uh, you know, I mean, sort of, kind of, and filler 'like', per 100 words. 'Like' is an estimate.")
def _filler_rate(s, label):
    text = s.text(label).lower()
    words = s.word_count(label)
    count = sum(1 for w in re.findall(r"[a-z']+", text) if w in FILLER_WORDS)
    count += sum(len(re.findall(rf'\b{re.escape(p)}\b', text)) for p in FILLER_PHRASES)
    count += len(_LIKE.findall(text))
    return _per_100_words(count, words)


@metric('hedge_rate', 'Hedges', 'clarity', 'rate', meaning='Softeners such as I think, maybe, I guess, per 100 words.')
def _hedge_rate(s, label):
    text = s.text(label).lower()
    count = sum(len(re.findall(rf'\b{re.escape(p)}\b', text)) for p in HEDGES)
    return _per_100_words(count, s.word_count(label))


# ---- composure

@metric('pace_wpm', 'Pace', 'composure', 'wpm', 'timed', meaning='Words per minute while you were speaking.')
def _pace(s, label):
    ms = sum(t.ms for t in s.of(label))
    return round(s.word_count(label) / (ms / 60000)) if ms else None


@metric('response_gap', 'Pause before you answer', 'composure', 'seconds', 'timed', meaning='Median gap between someone else finishing and you starting.')
def _response_gap(s, label):
    gaps = [
        max(0.0, (cur.start - prev.end) / 1000)
        for prev, cur in zip(s.turns, s.turns[1:])
        if cur.label == label and prev.label != label
    ]
    return round(statistics.median(gaps), 2) if gaps else None


@metric('cut_ins', 'Cut-ins', 'composure', 'count', 'timed', meaning='Times you started before the other person had finished.')
def _cut_ins(s, label):
    return sum(
        1 for prev, cur in zip(s.turns, s.turns[1:])
        if cur.label == label and prev.label != label and cur.start < prev.end
    )


# ---- connection

@metric('questions', 'Questions', 'connection', 'count', meaning="Real questions you asked. 'Right?' and other check-ins are left out.")
def _questions(s, label):
    q = classify_questions(s.text(label))
    return q['open'] + q['closed']


@metric('question_rate', 'Questions per 10 min', 'connection', 'rate', 'timed', meaning='Questions for every ten minutes of the session.')
def _question_rate(s, label):
    minutes = s.length_ms / 60000
    return round(10 * _questions(s, label) / minutes, 1) if minutes else None


@metric('open_question_share', 'Open questions', 'connection', 'pct', meaning='Share of your questions that start with what, how, why, or tell me. The rest can be answered yes or no.')
def _open_share(s, label):
    q = classify_questions(s.text(label))
    total = q['open'] + q['closed']
    return round(100 * q['open'] / total) if total else None


@metric('longest_listen', 'Longest listen', 'connection', 'clock', 'timed', meaning='The longest you let someone else speak before you came back in.')
def _longest_listen(s, label):
    spans = [prev.ms for prev, cur in zip(s.turns, s.turns[1:]) if cur.label == label and prev.label != label]
    return max(spans, default=None)


@metric('we_share', 'We vs. I', 'connection', 'pct', meaning='Of your we/us/our and I/me/my words, the share that were we.')
def _we_share(s, label):
    words = [w for t in s.of(label) for w in t.words]
    we = sum(1 for w in words if w in {'we', 'us', 'our', 'ours', "we're", "we'll", "we've"})
    me = sum(1 for w in words if w in {'i', 'me', 'my', 'mine', "i'm", "i'll", "i've", "i'd"})
    return round(100 * we / (we + me)) if (we + me) else None


# ---- coded: a model or an observer reads the turn and returns the value plus the quote that proves it

coded('four_beats', 'Four Beats landed', 'clarity', 'score', 'How many of Character, Goal, Challenge and Resolution a listener could repeat back (0 to 4).')
coded('time_to_point', 'Time to the point', 'clarity', 'seconds', 'Seconds until you said the headline or the ask.')
coded('ask_stated', 'Clear ask', 'clarity', 'score', 'Whether you made a specific ask the other person could say yes or no to (0 or 1).')
coded('acknowledged_first', 'Acknowledged first', 'connection', 'pct', 'Share of pushback moments where you named their concern before you answered it.')
coded('follow_ups', 'Follow-up questions', 'connection', 'count', 'Questions that built on what the other person had just said.')
coded('reflections', 'Reflections', 'connection', 'count', 'Times you said back what you heard before moving on.')
coded('held_under_pushback', 'Held under pushback', 'composure', 'pct', 'Share of pushback moments where your pace and your point both held.')
coded('next_step', 'Next step agreed', 'clarity', 'score', 'Whether the conversation ended with a named next step, owner and date (0 or 1).')
coded('rubric_total', 'Observer score', 'clarity', 'score', "The observer's 0 to 8 rubric for this week.")


# ---- group

@metric('balance', 'Balance', 'participation', 'score', 'timed', scope='group', meaning='100 means everyone spoke for the same time; 0 means one person did all the talking.')
def _balance(s: Session, labels: list[str]):
    return round(100 * (1 - _gini([sum(t.ms for t in s.of(l)) for l in labels]))) if labels else None


@metric('top_voice_share', 'Largest share', 'participation', 'pct', 'timed', scope='group', meaning='Share of talk time held by whoever spoke most.')
def _top_share(s, labels):
    shares = [_talk_share(s, l) or 0 for l in labels]
    return max(shares, default=None)


@metric('quiet_voices', 'Under half a fair share', 'participation', 'count', 'timed', scope='group', meaning='People who spoke for less than half of an even share.')
def _quiet(s, labels):
    if not labels:
        return None
    fair = 100 / len(labels)
    return sum(1 for l in labels if (_talk_share(s, l) or 0) < fair / 2)


@metric('group_open_share', 'Open questions', 'connection', 'pct', scope='group', meaning='Share of all questions in the room that were open.')
def _group_open(s, labels):
    totals = {'open': 0, 'closed': 0}
    for l in labels:
        q = classify_questions(s.text(l))
        totals['open'] += q['open']
        totals['closed'] += q['closed']
    asked = totals['open'] + totals['closed']
    return round(100 * totals['open'] / asked) if asked else None


@metric('group_questions', 'Questions asked', 'connection', 'count', scope='group', meaning='Real questions asked by everyone.')
def _group_questions(s, labels):
    return sum(_questions(s, l) for l in labels)


# ---------------------------------------------------------------- scorecard

def load_profiles(path: Path = PROFILE_PATH) -> dict:
    profiles = json.loads(Path(path).read_text(encoding='utf-8'))
    for name, profile in profiles.items():
        for key in list(profile.get('personal', [])) + list(profile.get('group', [])):
            if key not in REGISTRY:
                raise KeyError(f'Profile {name!r} names an unknown metric: {key}')
    return profiles


def show(unit: str, value) -> str:
    if value is None:
        return ''
    if unit == 'clock':
        seconds = int(round(float(value) / 1000))
        return f'{seconds // 60}:{seconds % 60:02d}'
    if unit == 'pct':
        return f'{round(float(value))}%'
    if unit == 'seconds':
        return f'{float(value):g}s'
    if unit == 'wpm':
        return f'{int(value)} wpm'
    if unit == 'rate':
        return f'{float(value):g}'
    return f'{value:g}' if isinstance(value, float) else str(value)


def _compare(m: Metric, value, *, baseline, band, room) -> tuple[str, str]:
    """(status, sentence). Status is one of changed, in_band, above, below, room, none."""
    if baseline is not None:
        if value == baseline:
            return 'changed', f'Same as last time ({show(m.unit, baseline)})'
        return 'changed', f'Was {show(m.unit, baseline)} last time'
    if band:
        low, high = band.get('min'), band.get('max')
        span = (
            f'{show(m.unit, low)} to {show(m.unit, high)}' if low is not None and high is not None
            else f'{show(m.unit, low)} or more' if low is not None
            else f'{show(m.unit, high)} or less'
        )
        if low is not None and value < low:
            return 'below', f'Below the target of {span}'
        if high is not None and value > high:
            return 'above', f'Above the target of {span}'
        return 'in_band', f'Inside the target of {span}'
    if room is not None and len(room) >= 4:
        middle = statistics.median(room)
        where = 'At' if value == middle else 'Above' if value > middle else 'Below'
        return 'room', f'{where} the middle of the room ({show(m.unit, middle)})'
    return 'none', ''


def build_scorecard(
    transcript: dict,
    profile: dict,
    *,
    names: dict[str, str] | None = None,
    roles: dict[str, str] | None = None,
    baseline: dict[str, dict] | None = None,
    coded: dict[str, dict] | None = None,
) -> dict:
    """Tiles for every person who gets a report, plus the group tiles.

    baseline and coded are keyed by speaker label, then by metric key.
    """
    s = Session(transcript, names, roles)
    report_roles = set(profile.get('report_roles') or ['participant'])
    floor = int(profile.get('min_words', 150))
    targets = profile.get('targets') or {}
    people = {}
    for label in s.labels:
        role = s.roles[label]
        if role not in report_roles:
            continue
        peers = [l for l in s.labels if s.roles[l] == role]
        brief = s.word_count(label) < floor
        keys = profile.get('brief', ['talk_time', 'words', 'questions']) if brief else profile['personal']
        tiles = []
        for key in keys:
            m = REGISTRY[key]
            value = (coded or {}).get(label, {}).get(key) if m.fn is None else m.fn(s, label)
            if value is None:
                continue
            room = None
            if m.fn is not None and not brief:
                room = [v for v in (m.fn(s, l) for l in peers) if v is not None]
            status, sentence = _compare(
                m, value,
                baseline=(baseline or {}).get(label, {}).get(key),
                band=None if brief else (targets.get(role) or {}).get(key),
                room=room,
            )
            tiles.append({
                'key': key, 'label': m.label, 'pillar': m.pillar, 'kind': m.kind,
                'value': value, 'number': show(m.unit, value),
                'status': status, 'compare': sentence, 'meaning': m.meaning,
            })
        people[label] = {'name': s.names[label], 'role': role, 'brief': brief, 'tiles': tiles}
    counted = [l for l in s.labels if s.roles[l] in set(profile.get('group_roles') or report_roles)]
    group = []
    for key in profile.get('group', []):
        m = REGISTRY[key]
        value = m.fn(s, counted)
        if value is not None:
            group.append({'key': key, 'label': m.label, 'value': value, 'number': show(m.unit, value), 'meaning': m.meaning})
    return {'profile': profile.get('title', ''), 'people': people, 'group': group}
