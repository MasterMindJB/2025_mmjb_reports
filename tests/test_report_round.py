"""Charts, the header line, and the repeated-unit opening for this Full Sail round."""

import json

from lib.names import header_name, report_header
from lib.personal_narrative import THINKING_BUDGET, estimate_personal_cost, section_problems, suggested_next_rep
from lib.readability import reading_grade
from lib.style_units import opening_line, style_candidates


def _turns(label, lines):
    return [{'speaker': label, 'name': label, 'text': line} for line in lines]


def test_header_uses_the_entered_name_then_the_email_then_nothing():
    assert header_name(entered='Steve Switaj', email='jdoe@example.com') == 'Steve Switaj'
    assert header_name(entered='', email='first.last@example.com') == 'First Last'
    assert header_name(entered='', email='first_last@example.com') == 'First Last'
    assert header_name(entered='', email='first-last@example.com') == 'First Last'
    assert header_name(entered='Speaker G', email='jdoe@example.com') is None
    assert header_name(entered='F', email='jdoe@x.com') is None
    assert header_name(entered='', email='room2@example.com') is None
    assert header_name(entered='', email='info@example.com') is None
    assert header_name(entered='', email='') is None
    assert report_header('Steve Switaj') == "Steve Switaj, here's your report"
    assert report_header(None) == "Here's your report"
    assert not report_header('Leslie').endswith('.')


def test_steve_opening_is_not_the_garbled_line():
    transcript = {'utterances': _turns('Steve', [
        'We can find a niche.',
        'The niche is the plan.',
        'I want a niche in this market.',
        'A niche can wait until Monday.',
    ]) + _turns('Ric', ['The budget can wait.'])}
    units = style_candidates(transcript, 'Steve', ['Steve Switaj', 'Ric'])
    assert units
    line = opening_line(units[0])
    assert line.startswith('You said "')
    assert 'niche' in line
    assert 'most repeated' not in line
    assert ' a "' not in line
    assert line.endswith('.')


def test_a_hedge_can_beat_a_common_word_and_a_short_unit_is_subsumed():
    transcript = {'utterances': _turns('Leslie', [
        'I think we should try.',
        'I think the plan works.',
        'I think it can wait.',
        'I think this is ready.',
    ]) + _turns('Ric', [
        'The budget is the budget and the budget can wait today.',
    ] * 3)}
    units = style_candidates(transcript, 'Leslie', ['Leslie', 'Ric'])
    assert units[0]['text'] == 'i think'
    nested = {'utterances': _turns('Steve', ['We need a new onboarding plan.'] * 3)}
    kept = {item['text'] for item in style_candidates(nested, 'Steve', ['Steve'])}
    assert 'a new onboarding plan' in kept
    assert 'onboarding plan' not in kept
    assert 'plan' not in kept
    quiet = {'utterances': _turns('G', ['Hello there friend.'])}
    assert style_candidates(quiet, 'G', ['G']) == []
    assert opening_line(None) == ''


def test_low_share_does_not_get_the_long_turn_script_and_most_of_needs_a_majority():
    low = suggested_next_rep(
        [{'text': 'The budget can wait until Monday if the room agrees on one clear plan. ' * 8, 'start': 0, 'end': 60000}],
        {'share': 6, 'speaker_count': 6, 'questions': 0},
    )
    assert low == 'Come in within the first 2 minutes.'
    assert 'stop after' not in low.lower()
    problems = section_problems(
        {
            'one_big_thing': 'You told one long story. What if you stopped after one point?',
            'why_it_matters': 'The room could not answer.',
            'yes_but': 'You spoke for most of this 8-minute meeting.',
            'next_rep': 'Come in within the first 2 minutes.',
        },
        [],
        set(),
        share=21,
    )
    assert any('majority' in item for item in problems.get('yes_but', []))
    assert 'why_it_matters' not in problems or not any('moment' in item for item in problems.get('why_it_matters', []))


def test_chart_types_do_not_repeat_inside_one_report(tmp_path):
    from PIL import Image

    from lib.pdf_functions import MM_PDF
    from lib.story_pdf import chart_types, write_group_story, write_personal_report

    image = tmp_path / 'chord.png'
    Image.new('RGB', (80, 80), 'white').save(image)
    pdf = MM_PDF(path_report_assets=__import__('pathlib').Path('report_assets'))
    personal_dir = tmp_path / 'personal'
    write_personal_report(pdf, {
        'header_name': 'Leslie',
        'name': 'Leslie',
        'short_mode': False,
        'one_big_thing': 'You came in late. What if you came in within the first 2 minutes?',
        'why_it_matters': 'The choice was made before you spoke.',
        'yes_but': 'The point was clear.',
        'numbers': ['You spoke for 6% of this 8-minute talk. That is less than an even split.'],
        'action': 'Come in within the first 2 minutes.',
        'share': 6,
        'roster_shares': [('Leslie', 6), ('Steve', 21), ('Ric', 17)],
        'question_counts': [('Leslie', 1), ('Steve', 2), ('Ric', 1)],
        'word_counts': [('Leslie', 80), ('Steve', 300), ('Ric', 140)],
        'proposal_counts': [('Leslie', 0), ('Steve', 2), ('Ric', 1)],
        'said_quote': '',
        'opening': 'You said "plan" 2 times.',
    }, personal_dir)
    personal = chart_types(personal_dir)
    kinds = [item['type'] for item in personal.values()]
    assert 'horizontal_bar' in kinds
    assert 'vertical_bar' in kinds
    assert 'donut' in kinds
    assert 'lollipop' in kinds
    assert 'big_number' in kinds
    assert len(kinds) == len(set(kinds))
    share = personal['question-share.png']['values']
    assert sum(share) == 4

    group_pdf = MM_PDF(path_report_assets=__import__('pathlib').Path('report_assets'))
    story = {
        'headline': 'A few people talked most.',
        'character': 'Steve talked the most, and Leslie the least. Who will ask Leslie to speak sooner?',
        'goal': 'The turns keep returning to the plan.',
        'challenge': 'A point can end before Leslie speaks.',
        'pattern_why': 'Steve talked the most, so Leslie had less chance to change the choice. Next time, ask Leslie before you decide.',
        'caveat': 'A point can end before Leslie speaks. Next time, stop and ask Leslie.',
        'drama': 'Who talks to whom.',
        'resolution': 'The meeting closed on Leslie.',
        'one_thing': 'Team: invite Leslie by name in the first 3 minutes.',
        'why': '',
        'numbers': ['4 of 6 people spoke. Two never did. Next time, invite one quiet person in before you close.'],
        'shares': [('Steve', 21), ('Ric', 17), ('Leslie', 6)],
        'question_counts': [('Steve', 2), ('Ric', 1), ('Leslie', 0)],
        'proposal_counts': [('Steve', 4), ('Ric', 2), ('Leslie', 0), ('A', 1), ('B', 1), ('C', 1), ('D', 1)],
        'word_counts': [('Steve', 300), ('Ric', 140), ('Leslie', 80)],
    }
    write_group_story(group_pdf, story, image, [('Steve', 'Ric', 2, 1)])
    group = chart_types(tmp_path)
    group_kinds = [item['type'] for item in group.values()]
    assert 'horizontal_bar' in group_kinds
    assert 'vertical_bar' in group_kinds
    assert 'donut' in group_kinds
    assert 'lollipop' in group_kinds
    assert 'chord' in group_kinds
    assert len(group_kinds) == len(set(group_kinds))
    assert sum(group['ideas.png']['values']) == 4 + 2 + 1 + 1 + 1 + 1
    text = ''
    group_pdf.output(tmp_path / 'group.pdf')
    from pypdf import PdfReader
    text = '\n'.join(page.extract_text() or '' for page in PdfReader(tmp_path / 'group.pdf').pages)
    assert 'How to read this chart' not in text
    assert 'Appendix' in text


def test_passed_to_stays_at_grade_six_and_the_run_stays_under_two_cents():
    from lib.data_story import _handed_to

    line = _handed_to({
        'utterances': [
            {'speaker': 'R', 'text': 'The plan needs a budget before we leave today.', 'start': 0, 'end': 4000},
            {'speaker': 'M', 'name': 'Full Sail Moderator', 'text': 'I can take the next point from here today.', 'start': 4000, 'end': 8000},
            {'speaker': 'R', 'text': 'Then we should ask Stephen before the close today.', 'start': 8000, 'end': 12000},
            {'speaker': 'S', 'name': 'Stephen', 'text': 'I will own the timeline after this meeting.', 'start': 12000, 'end': 16000},
        ]
    }, 'R')
    assert 'Full Sail Moderator' in line
    assert reading_grade(line) <= 6
    assert estimate_personal_cost(6, attempts=2) < 0.02
    client_budget = THINKING_BUDGET
    from langchain_google_genai import ChatGoogleGenerativeAI
    client = ChatGoogleGenerativeAI(model='gemini-3.6-flash', thinking_budget=client_budget, google_api_key='test')
    config = client._prepare_params(None)
    thinking = config['thinking_config'] if isinstance(config, dict) else config.thinking_config
    if not isinstance(thinking, dict):
        thinking = {'thinking_budget': getattr(thinking, 'thinking_budget', None)}
    assert thinking['thinking_budget'] == 256
    assert json.dumps(thinking)
