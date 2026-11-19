import pytest
from langchain_core.exceptions import OutputParserException

from lib.story_render import GROUP_TRANSITIONS, TRANSITIONS, render_personal_section
from lib.story_validation import BAN_PHRASES, reject_craft_violations
from tests.synthetic import synthetic_transcript

FACTS = {
    'talk_share': {'value': 44, 'unit': 'percent', 'line': '44% talk time: above the room average.'},
    'questions': {'value': 0, 'unit': 'questions', 'line': '0 questions: you did not ask, the same as the session average.'},
    'turns': {'value': 2, 'unit': 'turns', 'line': '2 turns: you came in 2 times.'},
    'early_share': {'value': 100, 'unit': 'percent', 'line': '100% in the first third, then 57% in the last third.'},
    '_meta': {
        'share': 44,
        'speaker_count': 3,
        'questions': 0,
        'question_average': 1,
        'early': 100,
        'late': 57,
    },
}


def _render(section, data):
    return render_personal_section(
        section,
        data,
        transcript=synthetic_transcript(),
        label='A',
        facts=FACTS,
        forbidden=['Blake Chen'],
    )


def test_pages_are_not_left_nearly_empty(tmp_path):
    import pandas as pd

    from lib.pdf_functions import MM_PDF, create_report_personal
    from tests.synthetic import synthetic_transcript

    stats = pd.DataFrame(
        {
            'Word Count': [8],
            'Speaking Time (%)': [40.0],
            'Questions Asked': [0],
            'Total Interactions': [2],
            'Words Per Minute (WPM)': [120.0],
        },
        index=['Alex Rivera'],
    )
    values = pd.DataFrame({'Assertive': [3]}, index=['Alex Rivera'])
    notes = pd.DataFrame({'Assertive': ['A short note.']}, index=['Alex Rivera'])
    create_report_personal(
        path_report_assets=__import__('pathlib').Path('report_assets'),
        path_reports_dir=tmp_path,
        df_nice_stats_data=stats,
        speaker_map={
            'A': {
                'name': 'Alex Rivera',
                'personal_summary': '',
                'personal_key_contributions': [],
                'personal_action_items': [],
            }
        },
        link_feedback='',
        speaker_analysis={'A': {'personal_super_power': '', 'personal_signature_phrase': {'bigram': 'should try', 'count': 1, 'what': 'You reach for it.', 'so_what': 'It buys a second.', 'now_what': 'Pause once.'}}},
        df_framework_amase_values=values,
        df_framework_amase_explanation=notes,
        df_framework_enneagram_values=values,
        df_framework_enneagram_explanation=notes,
        df_framework_doshas_values=values,
        df_framework_doshas_explanation=notes,
        transcript=synthetic_transcript(),
    )
    ratios = MM_PDF.latest_ratios
    assert ratios
    for ratio in ratios[:-1]:
        assert ratio >= 0.40
    assert ratios[-1] >= 0.15


def test_builtin_templates_do_not_use_banned_words():
    blobs = []
    for options in list(TRANSITIONS.values()) + list(GROUP_TRANSITIONS.values()):
        blobs.extend(options)
    for blob in blobs:
        lowered = blob.lower()
        for phrase in BAN_PHRASES:
            assert phrase not in lowered, blob


def test_realistic_good_sections_pass():
    opening = 'We should try the new onboarding plan.'
    later = 'That might work if we start Monday.'
    goods = [
        ('character', {'headline': 'You opened on the onboarding plan', 'number_keys': ['talk_share', 'questions', 'turns']}),
        ('goal', {'claim': 'You wanted the onboarding plan in motion', 'quote': {'text': opening, 'ts': '0:00'}}),
        ('challenge', {'claim': 'The question never arrived', 'fact_key': 'questions'}),
        (
            'drama',
            {
                'patterns': [
                    {
                        'quotes': [{'text': opening, 'ts': '0:00'}],
                        'pattern': 'You set the plan in the first line',
                        'twist': 'the next voice took the turn',
                        'cost': 'a question',
                        'gift': 'a clear start',
                    }
                ],
                'exception': {'quote': later, 'ts': '0:11', 'what_happened': 'you named Monday'},
            },
        ),
        ('resolution', {'claim': 'Monday became the stake', 'quote': {'text': later, 'ts': '0:11'}}),
        ('one_thing', {'verb': 'pause', 'verb_phrase': 'after the point', 'trigger': 'before the next story'}),
        ('character', {'headline': 'Two turns carried the plan', 'number_keys': ['turns', 'talk_share', 'questions']}),
        ('goal', {'claim': 'The budget was the test of the plan', 'quote': {'text': opening, 'ts': '0:00'}}),
        ('challenge', {'claim': 'The first third held the whole share', 'fact_key': 'early_share'}),
        ('one_thing', {'verb': 'ask', 'verb_phrase': 'one open question', 'trigger': 'before the second turn'}),
    ]
    for section, data in goods:
        rendered = _render(section, data)
        assert rendered


def test_bad_sections_are_rejected():
    opening = 'We should try the new onboarding plan.'
    with pytest.raises(OutputParserException):
        _render('goal', {'claim': 'You borrowed a line', 'quote': {'text': 'Could we check the budget first?', 'ts': '0:04'}})
    with pytest.raises(OutputParserException):
        _render('character', {'headline': 'You spoke for 99 percent of the hour', 'number_keys': ['talk_share', 'questions', 'turns']})
    with pytest.raises(OutputParserException):
        reject_craft_violations('We can leverage the plan.', allowed_numbers={'44'})
    with pytest.raises(OutputParserException):
        _render('resolution', {'claim': 'We can leverage Monday', 'quote': {'text': opening, 'ts': '0:00'}})
