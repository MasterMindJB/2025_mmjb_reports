from pathlib import Path

from langchain_core.exceptions import OutputParserException
import pytest

from lib.data_story import personal_story, signature_card
from lib.llm_validation import validate_llm_output
from lib.names import display_name, short_name
from lib.pdf_functions import create_report_personal
from tests.synthetic import synthetic_transcript


def test_short_name_keeps_a_role_title_whole():
    assert short_name('Full Sail Moderator') == 'Full Sail Moderator'
    assert short_name('Alex Rivera') == 'Alex'
    assert short_name('Person 3') == 'Person 3'
    assert display_name('  Alex   Rivera ') == 'Alex Rivera'


def test_validator_rejects_model_voice_and_tables():
    with pytest.raises(OutputParserException):
        validate_llm_output('personal_summary', 'As a sociolinguist, I analyze this.')
    with pytest.raises(OutputParserException):
        validate_llm_output('personal_summary', 'I felt the room tighten.')
    with pytest.raises(OutputParserException):
        validate_llm_output('conversation_title', '| A | B |\n| --- | --- |')
    assert validate_llm_output('personal_summary', 'You opened with a plan.') == 'You opened with a plan.'


def test_signature_card_keeps_three_short_beats():
    card = signature_card('**What?** You reach for a familiar phrase.\nSo, what? It buys you a second.\nNow, what?\nTry a question first.')
    assert card['what'].startswith('You reach')
    assert 'second' in card['so_what']
    assert card['now_what'].startswith('Try a question')
    assert 'coffee' not in card['what']


def test_personal_report_section_order(tmp_path):
    import pandas as pd

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
    notes = pd.DataFrame({'Assertive': ['AMASE_TOKEN_Alex_Rivera']}, index=['Alex Rivera'])
    speaker_map = {
        'A': {
            'name': 'Alex Rivera',
            'personal_summary': 'PERSONAL_SUMMARY_TOKEN_ALEX',
            'personal_key_contributions': [{'tagline': 'QUOTE_TAG_ALEX', 'quote': 'We should try the new onboarding plan.'}],
            'personal_action_items': [{'title': 'ACTION_TITLE_ALEX', 'action': 'ACTION_BODY_ALEX'}],
        }
    }
    analysis = {
        'A': {
            'personal_super_power': 'SUPERPOWER_TOKEN_Alex_Rivera',
            'personal_signature_phrase': {
                'bigram': 'should try',
                'count': 1,
                'analysis': 'What? SIGNATURE_PHRASE_TOKEN So, what? It starts the plan. Now, what? Ask first.',
            },
        }
    }
    create_report_personal(
        path_report_assets=Path('report_assets'),
        path_reports_dir=tmp_path,
        df_nice_stats_data=stats,
        speaker_map=speaker_map,
        link_feedback='',
        speaker_analysis=analysis,
        df_framework_amase_values=values,
        df_framework_amase_explanation=notes,
        df_framework_enneagram_values=values,
        df_framework_enneagram_explanation=notes,
        df_framework_doshas_values=values,
        df_framework_doshas_explanation=notes,
        transcript=synthetic_transcript(),
    )
    from pypdf import PdfReader

    text = '\n'.join(page.extract_text() or '' for page in PdfReader(tmp_path / 'Alex Rivera.pdf').pages)
    order = ('By the numbers', '1 big thing', 'Why it matters', 'What you said', 'Yes, but', 'Your next rep')
    positions = [text.find(heading) for heading in order]
    assert all(position != -1 for position in positions), text
    assert positions == sorted(positions)
    assert text.find('Your next rep') < text.find('Appendix: style lenses')
    assert 'Enneagram' not in text
    assert 'Doshas' not in text
    assert 'AMASE' in text
    assert 'Harkness Training Solutions 2026' in text
    assert 'Your signature phrase:' not in text
    assert 'So, what?' not in text
    assert 'coffee machine' not in text.lower()
