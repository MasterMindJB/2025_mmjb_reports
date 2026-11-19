"""Personal reports: evidence is chosen in code, and a failed call becomes the short report."""

import json

from lib.personal_narrative import (
    build_personal_story,
    candidate_quotes,
    good_sentence,
    needs_short_report,
    speaker_rows,
    substantive_count,
    word_total,
)
from lib.story_render import share_relation


def _long_rows():
    lines = [
        'We should try the new onboarding plan before Friday.',
        'The budget can wait until the client signs.',
        'I can own the numbers from this week.',
        'Name the owner before the next decision lands.',
        'That line still needs one reply from the room.',
        'Please draft the timeline after this meeting.',
    ]
    return [
        {'speaker': 'A', 'text': text, 'start': index * 5000, 'end': index * 5000 + 4000}
        for index, text in enumerate(lines)
    ]


def test_a_fragment_is_not_a_candidate_quote():
    assert not good_sentence('Right.')
    assert not good_sentence('And the key thing is to.')
    quotes = candidate_quotes(['Right. And the key thing is to.', 'We should try the new onboarding plan today.'])
    assert quotes == ['We should try the new onboarding plan today.']
    assert all(len(quote.split()) <= 40 for quote in quotes)


def test_two_long_monologues_are_a_full_report():
    sentence = 'The budget can wait until Monday if the room agrees on one clear plan for the work.'
    turn = ' '.join([sentence] * 2)
    rows = [
        {'speaker': 'L', 'text': turn, 'start': 0, 'end': 8000},
        {'speaker': 'L', 'text': turn, 'start': 9000, 'end': 16000},
    ]
    assert substantive_count(rows) == 2
    assert word_total(rows) >= 60
    assert needs_short_report(rows) is False


def test_short_mode_skips_the_model_when_there_are_not_enough_words():
    transcript = {'utterances': [
        {'speaker': 'L', 'text': 'The budget can wait until Monday if the room agrees.', 'start': 0, 'end': 4000},
        {'speaker': 'L', 'text': 'I can own the numbers from here.', 'start': 5000, 'end': 9000},
    ]}
    rows = speaker_rows(transcript, 'L')
    assert needs_short_report(rows)

    class Boom:
        def invoke(self, prompt):
            raise AssertionError('short mode must not call the model')

    story = build_personal_story(
        Boom(), transcript, 'L', {'name': 'Leslie'},
        {'share': 8, 'turns': 2, 'questions': 0, 'session_question_average': 1, 'speaker_count': 7},
    )
    assert story['short_mode'] is True
    assert story['headline'] == 'You spoke twice, for under a minute.'
    assert '1 big thing' not in story['one_big_thing']
    assert story['one_big_thing'] == ''
    assert story['why_it_matters'] == ''
    assert story['yes_but'] == ''
    assert 'budget can wait' in story['said_quote']
    assert 'to.' not in story['said_quote']


def test_a_failed_narrative_falls_back_to_the_short_report():
    transcript = {'utterances': _long_rows()}

    class Bad:
        def invoke(self, prompt):
            assert 'PERSONAL_NARRATIVE' in prompt
            return type('Reply', (), {'content': json.dumps({
                'headline': 'A focused dialogue between the moderator and Ric about careers.',
                'one_big_thing': 'You name probably. Next time, who else should speak about determine?',
                'why_it_matters': 'This is the part to change. Next time, try one small step.',
                'yes_but': 'I got into the business.',
                'next_rep': 'Target: 135 words in a turn.',
                'number_lines': ['The room median was 2 turns.'],
                'said_quote': 'not a candidate',
            })})()

    story = build_personal_story(
        Bad(), transcript, 'A', {'name': 'Ric'},
        {'share': 21, 'turns': 6, 'questions': 1, 'session_question_average': 1, 'speaker_count': 7},
    )
    assert story['short_mode'] is True
    assert '135' not in story['next_rep']
    assert 'focused dialogue' not in story['headline'].lower()
    assert 'room median was 1' in ' '.join(story['numbers']) or 'room median was 6' in ' '.join(story['numbers'])


def test_a_quoted_script_may_say_i_and_one_bad_section_stays_full(tmp_path):
    rows = []
    for row in _long_rows():
        rows.append({**row, 'text': f"{row['text']} {row['text']}", 'end': row['start'] + 70000})
    transcript = {'utterances': rows}
    quote = 'Name the owner before the next decision lands.'
    second = 'The budget can wait until the client signs.'

    class Scripted:
        def invoke(self, prompt):
            return type('Reply', (), {'content': json.dumps({
                'headline': 'You held the budget point.',
                'one_big_thing': f'You kept the budget in one stretch. You said "{quote}" Will you let the room answer?',
                'why_it_matters': 'The room could not answer while you were still talking.',
                'yes_but': 'The point was clear. It left no gap for a reply.',
                'next_rep': 'Say: "I would ask Leslie, what would you add?"',
                'number_lines': [
                    'You spoke for 21% of the hour. The even split is 14%.',
                    'You asked 1 question. The typical person asked 1.',
                    'You took 6 turns. The room median was 6 turns.',
                ],
                'said_quote': second,
            })})()

    story = build_personal_story(
        Scripted(), transcript, 'A', {'name': 'Ric'},
        {'share': 21, 'turns': 6, 'questions': 1, 'session_question_average': 1, 'speaker_count': 7},
        progress_dir=tmp_path,
    )
    assert story['short_mode'] is False
    assert 'decision' in story['one_big_thing']
    assert 'I would ask Leslie' not in story['next_rep']
    saved = json.loads((tmp_path / 'checkpoints.json').read_text(encoding='utf-8'))
    assert saved[0]['raw']
    assert 'unquoted first-person' not in ' '.join(saved[0]['reasons'])
    assert (tmp_path / 'checkpoints.json').stat().st_mode & 0o777 == 0o600


def test_a_splice_is_not_a_candidate_quote():
    turn = 'A lot of people say the work is hard. Then someone else talks for a while. It is very intimidating to begin now.'
    quotes = candidate_quotes([turn])
    assert quotes
    assert all(quote in turn for quote in quotes)
    assert not any('people say' in quote and 'intimidating' in quote for quote in quotes)


def test_long_fallback_headline_uses_the_clock_and_next_reps_differ():
    from lib.personal_narrative import speech_headline, suggested_next_rep

    long = [{'text': 'The budget can wait until Monday if the room agrees on one clear plan. ' * 8, 'start': 0, 'end': 60000}]
    assert speech_headline(long) == 'You spoke once, for about a minute.'
    budget = suggested_next_rep(long, {'share': 21, 'speaker_count': 6, 'questions': 0}, 'budget')
    niche = suggested_next_rep(
        [{'text': 'We can find a niche to exploit in this market before Friday arrives today.', 'start': 0, 'end': 70000}],
        {'share': 18, 'speaker_count': 6, 'questions': 1},
        'niche',
    )
    assert budget != niche
    assert 'budget' in budget and 'niche' in niche


def test_talk_share_not_the_quiet_handoff_is_who_we_ask():
    from lib.story_render import group_character_copy, render_one_thing

    facts = {
        'interaction': {'value': 20},
        '_shares': [('Steve', 21), ('G', 11), ('Leslie', 6)],
        '_portrait': {'pair': ('Moderator', 'Ric'), 'least': 'G'},
    }
    text = group_character_copy(facts)
    assert 'ask Leslie' in text
    assert 'ask G' not in text
    sentence, _why = render_one_thing(
        {'verb': 'ask', 'verb_phrase': 'one question', 'trigger': 'before the close'},
        facts, '', None,
    )
    assert sentence == 'Ask Leslie to speak before the next choice.'


def test_twenty_one_percent_of_seven_is_one_and_a_half_times():
    line = share_relation(21, 7)
    assert '1.5×' in line
    assert '2×' not in line


def test_personal_model_404_falls_back_to_flash_lite(monkeypatch, caplog):
    import logging

    from lib.personal_narrative import moment_word, personal_model_name

    monkeypatch.delenv('GEMINI_PERSONAL_MODEL', raising=False)
    assert personal_model_name() == 'gemini-3.5-flash'
    rows = []
    for row in _long_rows():
        rows.append({**row, 'text': f"{row['text']} {row['text']}", 'end': row['start'] + 70000})
    transcript = {'utterances': rows}
    quotes = candidate_quotes([row['text'] for row in rows])
    moment = moment_word(quotes, rows)
    payload = {
        'headline': 'You held the budget point.',
        'one_big_thing': (
            f'You kept the point in one stretch. You said "{quotes[0]}" Will you let the room answer?'
        ),
        'why_it_matters': 'The room could not answer while you were still talking.',
        'yes_but': 'The point was clear. It left no gap for a reply.',
        'next_rep': f'Next time, ask one question about {moment}.',
        'number_lines': ['You spoke for 21% of the hour.'],
        'said_quote': quotes[1],
    }

    class Missing:
        model = 'gemini-3.5-flash'

        def invoke(self, prompt):
            error = RuntimeError('404 Not Found: models/gemini-3.5-flash is not found')
            error.status_code = 404
            raise error

    class Lite:
        model = 'gemini-3.1-flash-lite'

        def __init__(self):
            self.calls = 0

        def invoke(self, prompt):
            self.calls += 1
            return type('Reply', (), {'content': json.dumps(payload)})()

    monkeypatch.setattr('lib.personal_narrative.for_personal_narrative', lambda llm: Missing())
    lite = Lite()
    with caplog.at_level(logging.WARNING):
        story = build_personal_story(
            lite, transcript, 'A', {'name': 'Ric'},
            {'share': 21, 'turns': 6, 'questions': 1, 'session_question_average': 1, 'speaker_count': 7},
        )
    assert lite.calls == 1
    assert story['short_mode'] is False
    assert '404' in caplog.text
    assert 'gemini-3.1-flash-lite' in caplog.text
