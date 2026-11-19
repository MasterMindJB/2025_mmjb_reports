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
    assert story['headline'] == 'A few short turns.'
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


def test_twenty_one_percent_of_seven_is_one_and_a_half_times():
    line = share_relation(21, 7)
    assert '1.5×' in line
    assert '2×' not in line
