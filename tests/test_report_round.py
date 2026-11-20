"""Charts, the header line, and the repeated-unit opening for this Full Sail round."""

import json

from lib.names import header_name, report_header
from lib.personal_narrative import estimate_personal_cost, section_problems, suggested_next_rep
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
    assert report_header('Steve Switaj') == "Steve, here's your report."
    assert report_header('Ric') == "Ric, here's your report."
    assert report_header(header_name(entered='', email='ric.viers@example.com')) == "Ric, here's your report."
    assert report_header(None) == "Here's your report."
    assert report_header('Leslie').endswith('.')


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
    assert opening_line(units[0]).startswith('You said "I think"')
    nested = {'utterances': _turns('Steve', ['We need a new onboarding plan.'] * 3)}
    kept = {item['text'] for item in style_candidates(nested, 'Steve', ['Steve'])}
    assert 'new onboarding plan' in kept
    assert 'a new onboarding plan' not in kept
    assert 'plan' not in kept
    quiet = {'utterances': _turns('G', ['Hello there friend.'])}
    assert style_candidates(quiet, 'G', ['G']) == []
    assert opening_line(None) == ''


def test_leslie_vocables_and_steve_contraction_are_not_units():
    leslie = {'utterances': _turns('Leslie', ['da, da, da. da da da.'] * 3)}
    assert style_candidates(leslie, 'Leslie', ['Leslie']) == []
    steve = {'utterances': _turns('Steve', ["he's ready. he's here. he's the one. he's back."])}
    assert style_candidates(steve, 'Steve', ['Steve']) == []
    assert opening_line(None) == ''


def test_chart_labels_do_not_cover_each_other(tmp_path):
    from lib import story_pdf
    from lib.story_pdf import _lollipop, _talk_share, _vertical_bars

    story_pdf._OVERLAPS.clear()
    talk = _talk_share({
        'name': 'Leslie', 'share': 21,
        'roster_shares': [('Leslie', 21), ('Steve', 14), ('Ric', 8)],
    }, tmp_path)
    columns = _vertical_bars(
        [('Leslie', 1), ('Steve Switaj', 4), ('Ric Viers', 2)],
        tmp_path, 'who-asked.png', 'Steve asked the most questions.', 'Leslie',
    )
    dots = _lollipop(
        [('Leslie', 40), ('Steve', 120)], tmp_path, 'turns.png', 'Steve had the longest turn.', 'Leslie',
    )
    assert talk.is_file() and columns.is_file() and dots.is_file()
    assert story_pdf._OVERLAPS == []


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


def test_control_ids_are_rejected_and_next_rep_templates_stay_easy():
    from lib.personal_narrative import section_problems
    from lib.readability import reading_grade
    from lib.story_render import session_action

    leaked = {
        'one_big_thing': 'You named the risk. Will you ask someone else?',
        'why_it_matters': 'The point in A1 is the one the room can use.',
        'yes_but': 'The turn was long.',
        'next_rep': 'Ask 1 question before you add a point.',
    }
    reasons = section_problems(leaked, ['The point is the one the room can use.'], {'1'}, '', 21)
    assert any('control id' in item for item in reasons['why_it_matters'])
    # A stale runner passed the share as the moment argument.
    stale = section_problems(leaked, ['The point is the one the room can use.'], {'1'}, 21)
    assert 'yes_but' not in stale or 'majority' not in ' '.join(stale.get('yes_but', []))
    for stats in (
        {'share': 4, 'speaker_count': 6, 'questions': 2, 'ideas': 1, 'words': 80, 'turns': 4},
        {'share': 30, 'speaker_count': 6, 'questions': 0, 'ideas': 1, 'words': 80, 'turns': 4},
        {'share': 30, 'speaker_count': 6, 'questions': 2, 'ideas': 1, 'words': 90, 'turns': 1},
        {'share': 30, 'speaker_count': 6, 'questions': 2, 'ideas': 0, 'words': 90, 'turns': 4},
        {'share': 30, 'speaker_count': 6, 'questions': 2, 'ideas': 1, 'words': 90, 'turns': 4},
    ):
        assert reading_grade(session_action(stats)) <= 5
    from lib.personal_narrative import _for_grade

    assert _for_grade("Ric Viers, here's your report The turn was long.").startswith('The turn')


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
    from lib.personal_narrative import personal_request

    body = personal_request('Count the turns.')
    assert list(body['generationConfig']['thinkingConfig']) == ['thinkingBudget']
    assert body['generationConfig']['maxOutputTokens'] == 512
    from lib.personal_narrative import upgrade_output_fits
    assert upgrade_output_fits()


def test_flash_lite_then_flash_upgrade_goes_through_the_entry_point(monkeypatch):
    from lib.personal_narrative import build_personal_story

    seen = []

    def fake_rest(model, key, prompt, mode='budget', max_output=512):
        seen.append((model, mode, max_output))
        if '3.6' not in model:
            return type('Reply', (), {'content': '{"character": ""}'})()
        return type('Reply', (), {'content': json.dumps({
            'bullets': [
                {
                    'behavior': 'You seem to care about the budget, and it seemed like you were trying to name an owner.',
                    'quote': 'Q1',
                    'why': 'The budget and the owner stay in that line.',
                },
                {
                    'behavior': 'The turn stayed on that point.',
                    'quote': 'Q2',
                    'why': 'The budget and the owner stay in that line.',
                },
            ],
            'yes_but': 'The point was clear.',
            'said_quote': 'Q2',
        })})()

    monkeypatch.setattr('lib.personal_narrative._rest_generate', fake_rest)
    transcript = {'utterances': [
        {'speaker': 'S', 'name': 'Stephen', 'text': 'The budget can wait until the client signs. ' * 4, 'start': 0, 'end': 40000},
        {'speaker': 'S', 'name': 'Stephen', 'text': 'Name the owner before the next decision lands. ' * 4, 'start': 40000, 'end': 80000},
    ]}

    class Client:
        google_api_key = 'test-key'
        model = 'gemini-3.1-flash-lite'

        def invoke(self, prompt, **kwargs):
            raise AssertionError('the entry point should call REST')

    story = build_personal_story(
        Client(), transcript, 'S', {'name': 'Stephen'},
        {'share': 18, 'turns': 1, 'questions': 1, 'session_question_average': 1, 'speaker_count': 6, 'words': 80},
    )
    assert seen[0][0] == 'gemini-3.1-flash-lite'
    assert seen[0][1] == 'budget'
    assert ('gemini-3.6-flash', 'minimal', 2048) in seen
    assert sum(1 for model, _mode, _cap in seen if model == 'gemini-3.6-flash') == 1
    assert story['short_mode'] is False


def test_a_personal_upgrade_restores_the_shared_client():
    from lib.personal_narrative import build_personal_story
    from lib.proposals import classify_proposals

    sentence = 'The budget can wait until the client signs. '

    class Shared:
        def __init__(self):
            self.model = 'models/gemini-3.1-flash-lite'
            self.seen = []

        def invoke(self, prompt, **kwargs):
            self.seen.append(self.model)
            if 'PERSONAL_NARRATIVE' in str(prompt) and '3.6' not in str(self.model):
                return type('Reply', (), {'content': '{"character": ""}'})()
            if 'PERSONAL_NARRATIVE' in str(prompt):
                return type('Reply', (), {'content': json.dumps({
                    'character': 'You seem to care a lot about the budget.',
                    'character_quote': 'Q1',
                    'goal': 'It seemed like you were trying to hold the budget.',
                    'challenge': 'The room waited.',
                    'why_it_matters': 'The budget stayed open.',
                    'yes_but': 'The point was clear.',
                    'said_quote': 'Q1',
                })})()
            return type('Reply', (), {'content': '{"turns": []}'})()

    client = Shared()
    transcript = {'utterances': [
        {'speaker': 'S', 'name': 'Stephen', 'text': sentence * 8, 'start': 0, 'end': 80000},
    ]}
    build_personal_story(
        client, transcript, 'S', {'name': 'Stephen'},
        {'share': 18, 'turns': 1, 'questions': 1, 'speaker_count': 6},
    )
    assert client.model == 'models/gemini-3.1-flash-lite'
    assert any(str(model).startswith('models/gemini-3.6-flash') for model in client.seen)
    classify_proposals(client, {'utterances': [{'speaker': 'S', 'name': 'Stephen', 'text': sentence, 'start': 0, 'end': 1000}]})
    assert client.seen[-1] == 'models/gemini-3.1-flash-lite'


def test_a_long_turn_stays_full_when_both_models_fail_the_grade():
    from lib.personal_narrative import build_personal_story

    first = 'The budget can wait until the client signs. '
    second = 'Name the owner before the next decision lands. '
    transcript = {'utterances': [
        {'speaker': 'S', 'name': 'Stephen', 'text': (first + second) * 6, 'start': 0, 'end': 90000},
    ]}

    class Bad:
        def invoke(self, prompt, **kwargs):
            return type('Reply', (), {'content': json.dumps({
                'one_big_thing': 'Your aim is a sophisticated articulation of the strategic implication.',
                'why_it_matters': 'Your aim: try the step below on the next page. See A1.',
                'yes_but': 'I felt the turn was long.',
                'next_rep': 'Target: 135 words.',
                'said_quote': 'not a quote',
            })})()

    story = build_personal_story(
        Bad(), transcript, 'S', {'name': 'Stephen'},
        {'share': 18, 'turns': 1, 'questions': 1, 'session_question_average': 1, 'speaker_count': 6},
    )
    assert story['short_mode'] is False
    assert len(story['bullets']) >= 2
    assert story['bullets'][0].startswith('You seem to care')
    assert 'it seemed like you were trying to' in story['bullets'][0].lower()
    assert 'That line is about' not in ' '.join(story['bullets'])
    assert 'The turn stayed on that point' not in ' '.join(story['bullets'])
    assert 'it seemed like you were trying to' not in story['bullets'][1].lower()
    assert 'You said.' not in ' '.join(story['bullets'])
    assert all('We came to that conclusion because of "' in bullet for bullet in story['bullets'])
    assert 'below' not in ' '.join(story['bullets']).lower()
    assert 'A1' not in ' '.join(story['bullets'])


def test_question_columns_include_every_speaker_with_unique_labels(tmp_path):
    from lib.story_pdf import _vertical_bars, chart_types

    rows = [
        ('Speaker F', 0),
        ('Speaker G', 0),
        ('Full Sail Moderator', 3),
        ('Steve Switaj', 4),
        ('Stephen', 2),
        ('Ric Viers', 1),
        ('Leslie Brathwaite', 1),
    ]
    path = _vertical_bars(rows, tmp_path, 'who-asked.png', 'Steve asked the most questions.')
    info = chart_types(tmp_path)['who-asked.png']
    assert path.is_file()
    assert info['values'] == [4, 3, 2, 1, 1, 0, 0]
    assert len(info['labels']) == 7
    assert len(set(info['labels'])) == 7
    assert 'Moderator' in info['labels']
    assert 'Full' not in info['labels']
    assert 'Speaker' not in info['labels']
    assert any(label.replace('\n', ' ') == 'Speaker F' for label in info['labels'])
    assert any(label.replace('\n', ' ') == 'Speaker G' for label in info['labels'])
    assert info['value_labels'].count('0') == 2


def test_group_page_uses_the_pattern_and_one_easy_action(tmp_path):
    from PIL import Image
    from pypdf import PdfReader

    from lib.pdf_functions import MM_PDF
    from lib.readability import reading_grade
    from lib.story_pdf import write_group_story
    from lib.story_render import group_big_thing_problems, group_caveat, team_action

    text = 'Steve Switaj held the floor, and Leslie Brathwaite stayed quiet. Steve Switaj said "We can find a niche in this market today." Who will ask Leslie to talk?'
    assert group_big_thing_problems(text) == []
    assert group_big_thing_problems('Steve Switaj held the floor, and Leslie Brathwaite stayed quiet. Who will ask Leslie to talk?')
    action = team_action('Full Sail Moderator', 'Leslie Brathwaite')
    caveat = group_caveat({'least': 'Leslie Brathwaite'}, [('Steve Switaj', 21), ('Leslie Brathwaite', 6)])
    assert action == 'Moderator, ask Leslie to talk. Do this in the first 3 minutes.'
    assert reading_grade(action) <= 5
    assert reading_grade(caveat) <= 5
    image = tmp_path / 'chord.png'
    Image.new('RGB', (80, 80), 'white').save(image)
    image.with_suffix('.title').write_text('Full Sail Moderator had the most back and forth.', encoding='utf-8')
    pdf = MM_PDF(path_report_assets=__import__('pathlib').Path('report_assets'))
    story = {
        'headline': 'Full Sail Moderator had the most back and forth.',
        'character': text,
        'goal': 'The room had one thread.',
        'pattern_why': 'Steve held the floor, so Leslie had less chance to change the choice.',
        'caveat': caveat,
        'one_thing': action,
        'shares': [('Steve Switaj', 21), ('Leslie Brathwaite', 6), ('Speaker F', 4)],
        'question_counts': [('Steve Switaj', 2), ('Leslie Brathwaite', 0)],
        'proposal_counts': [('Steve Switaj', 1)],
        'numbers': ['20% of turns moved to someone new, below an even mix of 86%.', '2 of 3 people spoke. One never did.'],
        'word_counts': [('Steve Switaj', 200), ('Leslie Brathwaite', 40)],
    }
    write_group_story(pdf, story, image, [('Steve Switaj', 'Leslie Brathwaite', 2, 1)])
    pdf.output(tmp_path / 'group.pdf')
    pages = [page.extract_text() or '' for page in PdfReader(tmp_path / 'group.pdf').pages]
    assert 'held the floor' in pages[0]
    assert 'back and forth' not in pages[0]
    assert 'from Steve Switaj' in pages[0]
    assert '1 idea, from Steve Switaj' not in '\n'.join(pages)
    assert '20% of turns moved' not in pages[0]
    joined = '\n'.join(pages)
    assert 'A question is a sentence that ends in ? or starts with a question word.' in joined
    assert 'back and forth' in joined


def test_personal_bullets_use_a_different_quote_and_drop_why_it_matters(tmp_path):
    from pypdf import PdfReader

    from lib.pdf_functions import MM_PDF
    from lib.personal_narrative import render_personal_bullets
    from lib.story_pdf import write_personal_report

    quotes = [
        'The budget can wait until the client signs.',
        'Name the owner before the next decision lands.',
        'Please draft the timeline after this meeting.',
    ]
    text = render_personal_bullets({
        'bullets': [
            {'behavior': 'You seem to care about the budget.', 'quote': 'Q1', 'why': 'The budget is the point that line keeps open.'},
            {'behavior': 'It seemed like you were trying to name an owner.', 'quote': 'Q2', 'why': 'The owner is still unnamed in that line.'},
            {'behavior': 'The turn stayed on the timeline.', 'quote': 'Q3', 'why': 'The timeline is what that sentence keeps.'},
        ],
    }, quotes)
    assert len(text) == 3
    assert text[0].startswith('You seem to care about the budget.')
    assert 'We came to that conclusion because of "The budget can wait until the client signs."' in text[0]
    assert 'You seem' not in text[1]
    assert 'You seem' not in text[2]
    assert len({line.split('because of ')[1] for line in text}) == 3
    short = render_personal_bullets({
        'bullets': [
            {'behavior': 'You seem to care about the budget.', 'quote': 'Q1', 'why': 'The budget stays open.'},
            {'behavior': 'The turn stayed on the owner.', 'quote': 'Q2', 'why': 'The owner is still unnamed.'},
            {'behavior': 'The timeline came last.', 'quote': 'Q3', 'why': 'The timeline can wait.'},
        ],
    }, quotes[:2])
    assert len(short) == 2
    pdf = MM_PDF(path_report_assets=__import__('pathlib').Path('report_assets'))
    write_personal_report(pdf, {
        'name': 'Leslie', 'header_name': 'Leslie Brathwaite', 'bullets': text,
        'yes_but': '', 'share': 6, 'roster_shares': [('Leslie', 6)],
        'appendix': {},
    }, tmp_path)
    pdf.output(tmp_path / 'leslie.pdf')
    page = '\n'.join(item.extract_text() or '' for item in PdfReader(tmp_path / 'leslie.pdf').pages)
    assert "Leslie, here's your report." in page
    assert 'Why it matters' not in page
    assert 'We came to that conclusion because of' in page


def test_personal_box_keeps_longest_turn_and_leads_with_surprise():
    from lib.scorecard import personal_metrics

    story = {
        'name': 'Ric',
        'roster_shares': [('Ric', 40), ('Steve', 20), ('Leslie', 10), ('G', 30)],
        'word_counts': [('Ric', 100), ('Steve', 90), ('Leslie', 80), ('G', 95)],
        'question_counts': [('Ric', 0), ('Steve', 4), ('Leslie', 3), ('G', 4)],
        'longest_turns': [('Ric', 200), ('Steve', 40), ('Leslie', 30), ('G', 50)],
        'proposal_counts': [('Ric', 1), ('Steve', 1), ('Leslie', 1), ('G', 1)],
    }
    metrics = personal_metrics(story)
    assert [item['key'] for item in metrics[:2]] == ['share', 'longest']
    assert metrics[0]['label'] == 'Talk time'
    assert len(metrics) == 4
    tail = metrics[2:]
    assert [item['surprise'] for item in tail] == sorted((item['surprise'] for item in tail), reverse=True)
    same = dict(story)
    same['word_counts'] = [('Ric', 200), ('Steve', 40), ('Leslie', 30), ('G', 50)]
    same['longest_turns'] = [('Ric', 200), ('Steve', 40), ('Leslie', 30), ('G', 50)]
    collapsed = personal_metrics(same)
    assert 'words' not in {item['key'] for item in collapsed}
    assert any(item['key'] == 'longest' for item in collapsed)
    from lib.scorecard import format_duration, round_half_up
    assert round_half_up(16.5) == 17
    assert round_half_up(5.5) == 6
    assert format_duration(95_000) == '1:35'
    assert format_duration(90_000) == '1:30'
    assert format_duration(0) == '0:00'
    timed = dict(story)
    timed['talk_ms'] = [('Ric', 95_000), ('Steve', 40_000), ('Leslie', 20_000), ('G', 30_000)]
    timed['longest_ms'] = [('Ric', 80_000), ('Steve', 20_000), ('Leslie', 15_000), ('G', 25_000)]
    shown = personal_metrics(timed)
    assert shown[0]['number'] == '1:35'
    assert shown[0]['grade'].startswith('40% of the talk,')
    assert shown[1]['label'] == 'Longest turn'
    assert shown[1]['number'] == '1:20'


def test_personal_report_has_no_next_rep_when_the_flag_is_off(tmp_path, monkeypatch):
    from PIL import Image
    from pypdf import PdfReader

    from lib.pdf_functions import MM_PDF
    from lib.personal_narrative import advice_problems, log_scorecard
    from lib.story_pdf import write_personal_report

    monkeypatch.delenv('PERSONAL_NEXT_REP', raising=False)
    assert advice_problems('Try this next time. What if you asked?') 
    assert not advice_problems('The room waited. You said "The budget can wait until Monday."')
    story = {
        'name': 'Ric Viers',
        'header_name': 'Ric Viers',
        'character': 'You seem to care a lot about doing work you love.',
        'goal': 'It seemed like you were trying to name the work.',
        'challenge': 'You said "The budget can wait until Monday."',
        'why_it_matters': 'The room can still hear you on budget.',
        'yes_but': 'The point was clear.',
        'next_rep': 'Ask 1 question before you add a point.',
        'said_quote': 'The budget can wait until Monday.',
        'share': 40,
        'roster_shares': [('Ric Viers', 40), ('Steve', 20)],
        'word_counts': [('Ric Viers', 100), ('Steve', 40)],
        'question_counts': [('Ric Viers', 0), ('Steve', 3)],
        'longest_turns': [('Ric Viers', 80), ('Steve', 20)],
        'proposal_counts': [('Ric Viers', 1), ('Steve', 1)],
        'appendix': {},
    }
    pdf = MM_PDF(path_report_assets=__import__('pathlib').Path('report_assets'))
    write_personal_report(pdf, story, tmp_path)
    pdf.output(tmp_path / 'ric.pdf')
    text = '\n'.join(page.extract_text() or '' for page in PdfReader(tmp_path / 'ric.pdf').pages)
    assert text.startswith("Ric, here's your report") or "Ric, here's your report" in text
    assert 'Your next rep' not in text
    assert 'Ask 1 question' not in text
    assert 'By the numbers' in text
    assert 'Longest turn' in text
    assert 'Your charts' in text
    log_scorecard(tmp_path, 'Ric', story.get('scorecard') or __import__('lib.scorecard', fromlist=['personal_metrics']).personal_metrics(story))
    saved = json.loads((tmp_path / 'checkpoints.json').read_text(encoding='utf-8'))
    assert saved[-1]['lead_metric']
    assert 'surprise' in saved[-1]


def test_bare_units_are_quoted_and_incomplete_bullets_are_dropped():
    from lib.personal_narrative import _fallback_bullets, later_return_line, merged_care_goal, render_personal_bullets

    quotes = [
        'The budget can wait until the client signs.',
        'Name the owner before the next decision lands.',
    ]
    assert merged_care_goal(quotes[0]).startswith('You seem to care about the budget, and it seemed like you were trying to talk about the client')
    assert 'loader' not in merged_care_goal(quotes[0])
    assert merged_care_goal('The world can wait until the world changes today.') == ''
    for unit in ('life', 'passion', 'hard', 'speak'):
        assert later_return_line(unit) == f'You kept coming back to "{unit}"'
    assert later_return_line("that's") == ''
    assert later_return_line("he's") == ''
    bullets = _fallback_bullets([{'text': 'speak', 'count': 3}], quotes)
    assert len(bullets) == 2
    assert bullets[0].startswith('You seem to care about the budget, and it seemed like you were trying to talk about the client')
    assert 'speak' not in bullets[1].lower()
    assert 'owner' in bullets[1].lower()
    assert 'That line is about' not in ' '.join(bullets)
    assert 'The turn stayed on that point' not in ' '.join(bullets)
    assert all('We came to that conclusion because of "' in bullet for bullet in bullets)
    assert not bullets[1].lower().startswith('it seemed')
    life = 'I have loved this life since the work began today.'
    paired = _fallback_bullets([{'text': 'life', 'count': 3}], [quotes[0], life])
    assert 'You kept coming back to "life"' in paired[1]
    assert _fallback_bullets([{'text': 'life', 'count': 3}], quotes[:1]) == []
    dropped = render_personal_bullets({
        'bullets': [
            {'behavior': 'You said.', 'quote': 'Q1', 'why': 'The budget stays in that line.'},
            {'behavior': 'You seem to care about the budget.', 'quote': 'Q2', 'why': ''},
        ],
    }, quotes)
    assert dropped == []


def test_a_why_can_share_the_behavior_and_a_restatement_is_rejected():
    from lib.personal_narrative import _bullet_problems, why_on_topic, why_restates

    quote = 'I have been making movies for most of my career.'
    behavior = 'You seem to care about the movies, and it seemed like you were trying to name the work.'
    assert why_restates('You brought up making movies.', quote)
    assert why_restates('You spoke about your career.', 'Your career has been the passion of your life.')
    assert why_restates('You stated that your passion is wide.', 'My passion is wide and it covers film.')
    assert why_restates('You made coffee before you moved up.', 'I made coffee before I moved up the hill.')
    assert not why_restates('That story shows people you know how long the climb takes.', 'The climb took years before the film opened.')
    assert why_on_topic('The room can hear how long that climb takes.', 'The climb took years.', behavior)
    assert why_on_topic('The movies still matter to people.', quote, 'You kept the point.')
    assert why_on_topic(
        'This helps people see that your love for art started early.',
        'The budget can wait until the client signs.',
        'You seem to care about the budget.',
        ['Your love for art started when you were young.'],
    )
    assert not why_on_topic(
        'The weather stayed cold all morning.',
        'The budget can wait until the client signs.',
        'You seem to care about the budget.',
        ['Name the owner before the next decision lands.'],
    )
    problems = _bullet_problems([
        {'behavior': behavior, 'quote': 'Q1', 'why': 'You brought up making movies.'},
    ], [quote])
    assert any('restates' in item for item in problems)
    shared = _bullet_problems([
        {
            'behavior': behavior,
            'quote': 'Q1',
            'why': 'That shows people the movies still matter.',
        },
        {
            'behavior': 'The turn named the work.',
            'quote': 'Q2',
            'why': 'That shows people the work still matters.',
        },
    ], [quote, 'Name the work before the next film opens.'])
    assert not any('drifts' in item for item in shared)


def test_a_grade_repair_stays_on_flash_lite_and_one_upgrade_is_the_cap(monkeypatch):
    from lib.personal_narrative import build_personal_story, reading_grade
    from lib.readability import reading_grade as grade

    hard = 'This extraordinary organizational demonstration of the budget shows people an extraordinarily complicated circumstance.'
    assert grade(hard) > 6
    seen = []

    def fake_rest(model, key, prompt, mode='budget', max_output=512):
        seen.append((model, mode, max_output, prompt[:24]))
        if prompt.startswith('Rewrite this one sentence'):
            assert model == 'gemini-3.1-flash-lite'
            assert mode == 'budget'
            assert max_output <= 256
            return type('Reply', (), {'content': 'That shows people the budget still matters.'})()
        if '3.6' in model:
            return type('Reply', (), {'content': '{"character": ""}'})()
        return type('Reply', (), {'content': json.dumps({
            'bullets': [
                {
                    'behavior': 'You seem to care about the budget, and it seemed like you were trying to name an owner.',
                    'quote': 'Q1',
                    'why': hard,
                },
                {
                    'behavior': 'The line named the owner.',
                    'quote': 'Q2',
                    'why': 'That shows people the owner still matters.',
                },
            ],
        })})()

    monkeypatch.setattr('lib.personal_narrative._rest_generate', fake_rest)

    class Client:
        google_api_key = 'test-key'
        model = 'gemini-3.1-flash-lite'

        def invoke(self, prompt, **kwargs):
            raise AssertionError('the entry point should call REST')

    transcript = {'utterances': [
        {'speaker': 'S', 'text': 'The budget can wait until the client signs. ' * 4, 'start': 0, 'end': 40000},
        {'speaker': 'S', 'text': 'Name the owner before the next decision lands. ' * 4, 'start': 40000, 'end': 80000},
    ]}
    story = build_personal_story(
        Client(), transcript, 'S', {'name': 'Stephen'},
        {'share': 18, 'turns': 2, 'questions': 1, 'speaker_count': 6, 'words': 80},
    )
    assert story['short_mode'] is False
    assert any(item[3].startswith('Rewrite') for item in seen)
    assert not any(item[0] == 'gemini-3.6-flash' for item in seen)
    assert reading_grade(story['bullets'][0].split('".')[-1]) <= 6

    seen.clear()

    def always_bad(model, key, prompt, mode='budget', max_output=512):
        seen.append(model)
        return type('Reply', (), {'content': json.dumps({
            'bullets': [{
                'behavior': 'You brought up the budget.',
                'quote': 'Q1',
                'why': 'You brought up the budget.',
            }],
        })})()

    monkeypatch.setattr('lib.personal_narrative._rest_generate', always_bad)
    build_personal_story(
        Client(), transcript, 'S', {'name': 'Stephen'},
        {'share': 18, 'turns': 2, 'questions': 1, 'speaker_count': 6, 'words': 80},
    )
    assert seen.count('gemini-3.6-flash') == 1


def test_talk_and_question_charts_stay_with_the_box(tmp_path):
    from pypdf import PdfReader

    from lib.pdf_functions import MM_PDF
    from lib.story_pdf import write_personal_report

    story = {
        'name': 'Stephen',
        'header_name': 'Stephen',
        'bullets': [
            'You seem to care about the budget, and it seemed like you were trying to name the owner. We came to that conclusion because of "The budget can wait until the client signs."',
            'You kept coming back to "owner". We came to that conclusion because of "Name the owner before the next decision lands."',
            'You kept coming back to "client". We came to that conclusion because of "The client signs when the budget is ready."',
        ],
        'share': 40,
        'roster_shares': [('Stephen', 40), ('Ric', 12), ('Leslie', 10), ('Steve', 9), ('Host', 8), ('Casey', 7)],
        'question_counts': [('Stephen', 1), ('Ric', 4), ('Leslie', 2), ('Steve', 0), ('Host', 1), ('Casey', 1)],
        'longest_turns': [('Stephen', 95), ('Ric', 20), ('Leslie', 11), ('Steve', 8), ('Host', 6), ('Casey', 4)],
        'proposal_counts': [('Stephen', 2), ('Ric', 1), ('Leslie', 0), ('Steve', 1)],
        'word_counts': [('Stephen', 400), ('Ric', 80)],
        'said_quote': 'The budget can wait until the client signs.',
        'yes_but': '',
    }
    pdf = MM_PDF(path_report_assets=__import__('pathlib').Path('report_assets'))
    write_personal_report(pdf, story, tmp_path)
    pdf.output(tmp_path / 'stephen.pdf')
    pages = PdfReader(tmp_path / 'stephen.pdf').pages
    first = pages[0].extract_text() or ''
    assert "Stephen, here's your report" in first
    assert 'By the numbers' in first
    assert 'Your charts' in first
    counts = [len(list(page.images)) for page in pages]
    assert counts[0] >= 5
    for count in counts[1:]:
        assert count == 1 or count >= 4


def test_a_drifting_why_is_rewritten_without_dropping_the_behavior(monkeypatch, tmp_path):
    from lib.personal_narrative import build_personal_story

    seen = []

    def fake_rest(model, key, prompt, mode='budget', max_output=512):
        seen.append((model, mode, max_output, prompt[:40]))
        if prompt.startswith('Rewrite this one sentence'):
            assert 'BEHAVIOR' in prompt
            assert 'This helps people see that your love for art started early.' in prompt
            assert mode == 'budget' and max_output <= 128
            return type('Reply', (), {'content': 'This helps people see that the budget still matters.'})()
        return type('Reply', (), {'content': json.dumps({
            'bullets': [
                {
                    'behavior': 'You seem to care about the budget, and it seemed like you were trying to name an owner.',
                    'quote': 'Q1',
                    'why': 'The weather stayed cold all morning.',
                },
                {
                    'behavior': 'The line named the owner.',
                    'quote': 'Q2',
                    'why': 'This helps people see that the owner still matters.',
                },
            ],
        })})()

    monkeypatch.setattr('lib.personal_narrative._rest_generate', fake_rest)

    class Client:
        google_api_key = 'test-key'
        model = 'gemini-3.1-flash-lite'

        def invoke(self, prompt, **kwargs):
            raise AssertionError('the entry point should call REST')

    transcript = {'utterances': [
        {'speaker': 'L', 'name': 'Leslie', 'text': 'The budget can wait until the client signs. ' * 4, 'start': 0, 'end': 40000},
        {'speaker': 'L', 'name': 'Leslie', 'text': 'Name the owner before the next decision lands. ' * 4, 'start': 40000, 'end': 80000},
    ]}
    story = build_personal_story(
        Client(), transcript, 'L', {'name': 'Leslie'},
        {'share': 18, 'turns': 4, 'questions': 1, 'speaker_count': 6, 'words': 90},
        progress_dir=tmp_path,
    )
    assert story['short_mode'] is False
    assert len(story['bullets']) >= 2
    assert 'name an owner' in story['bullets'][0]
    assert 'weather' not in story['bullets'][0].lower()
    assert not any(item[0] == 'gemini-3.6-flash' for item in seen)
    saved = json.loads((tmp_path / 'checkpoints.json').read_text(encoding='utf-8'))
    rejected = [item for entry in saved for item in (entry.get('rejected_whys') or [])]
    assert any(item['why'] == 'The weather stayed cold all morning.' and 'drifts' in item['reason'] for item in rejected)


def test_two_quotes_never_become_a_short_report_and_a_bad_why_is_dropped():
    from lib.personal_narrative import _fallback_bullets, build_personal_story

    quotes = [
        'The budget can wait until the client signs.',
        'Name the owner before the next decision lands.',
    ]
    bullets = _fallback_bullets([{'text': 'speak', 'count': 4}], quotes)
    assert len(bullets) == 2
    assert all('We came to that conclusion because of "' in bullet for bullet in bullets)
    assert 'speak' not in bullets[1].lower()

    class Drift:
        def invoke(self, prompt, **kwargs):
            if str(prompt).startswith('Rewrite'):
                return type('Reply', (), {'content': 'The weather stayed cold all morning.'})()
            return type('Reply', (), {'content': json.dumps({
                'bullets': [
                    {
                        'behavior': 'You seem to care about the budget, and it seemed like you were trying to name an owner.',
                        'quote': 'Q1',
                        'why': 'The weather stayed cold all morning.',
                    },
                    {
                        'behavior': 'The line named the owner.',
                        'quote': 'Q2',
                        'why': 'The weather stayed cold all morning.',
                    },
                ],
            })})()

    transcript = {'utterances': [
        {'speaker': 'L', 'name': 'Leslie', 'text': quotes[0] + ' ' + quotes[1], 'start': 0, 'end': 20000},
        {'speaker': 'L', 'name': 'Leslie', 'text': (quotes[0] + ' ') * 6, 'start': 20000, 'end': 80000},
    ]}
    story = build_personal_story(
        Drift(), transcript, 'L', {'name': 'Leslie'},
        {'share': 22, 'turns': 4, 'questions': 1, 'speaker_count': 6, 'words': 80},
    )
    assert story['short_mode'] is False
    assert len(story['bullets']) >= 2
    assert 'name an owner' in story['bullets'][0]
    assert 'a loader' not in ' '.join(story['bullets']).lower()
    assert 'talk about the world' not in ' '.join(story['bullets']).lower()
    assert 'We came to that conclusion because of' in story['bullets'][0]


def test_listener_whys_pass_and_a_missing_care_noun_is_repaired(caplog, monkeypatch):
    import logging

    from lib.personal_narrative import (
        _lemmas,
        _usable_model_bullets,
        _why_problem,
        build_personal_story,
        care_grounded,
        why_line_grade_problem,
        why_on_topic,
    )

    lines = [
        'That story shows people that your dream has been real for years.',
        'That moment lets others know this choice is not open for debate.',
        'This shows listeners that you value daily effort over simple luck.',
        'That history helps others see where your creative drive started.',
        'That growth shows others how quickly your hard work paid off.',
    ]
    behavior = 'You seem to care about the work, and it seemed like you were trying to name it.'
    quote = 'The budget can wait until the client signs.'
    for line in lines:
        assert why_on_topic(line, '', '', [], speech='')
        assert _why_problem(line, quote, behavior, [quote], '') == ''
    for word in ('dream', 'choice', 'work', 'effort', 'hard'):
        assert word in _lemmas(f'The {word} stayed in the talk today.')
    history = lines[3]
    other = 'This highlights the specific area where you refuse to yield.'
    with caplog.at_level(logging.INFO):
        assert why_line_grade_problem(history, quote) == ''
        other_problem = why_line_grade_problem(other, quote)
    assert history in caplog.text
    assert other in caplog.text
    assert other_problem == '' or 'grade' in other_problem or 'long word' in other_problem
    hinge = 'And you make it hinge upon yourself.'
    assert not care_grounded(
        'You seem to care about your team, and it seemed like you were trying to name the work.',
        hinge,
        hinge,
    )
    assert care_grounded(
        'You seem to care about owning your results, and it seemed like you were trying to name the work.',
        hinge,
        hinge,
    )
    quotes = [quote, 'Name the owner before the next decision lands.']
    kept = _usable_model_bullets({
        'bullets': [
            {'behavior': behavior, 'quote': 'Q1', 'why': history},
            {
                'behavior': 'The line named the owner.',
                'quote': 'Q2',
                'why': 'The weather stayed cold all morning.',
                'why_rejected': 'The weather stayed cold all morning.',
            },
        ],
    }, quotes, quote)
    assert 'weather' in kept[1].lower()
    assert 'history helps others' in kept[0].lower()
    dropped = _usable_model_bullets({
        'bullets': [
            {'behavior': behavior, 'quote': 'Q1', 'why': history},
            {'behavior': 'The line named the owner.', 'quote': 'Q2', 'why': lines[1]},
            {
                'behavior': 'The turn named the client.',
                'quote': 'Q3',
                'why': 'The weather stayed cold all morning.',
                'why_rejected': 'The weather stayed cold all morning.',
            },
        ],
    }, quotes + ['The client signs when the budget is ready.'], quote)
    assert len(dropped) == 2
    assert all('weather' not in line.lower() for line in dropped)

    seen = []

    def fake_rest(model, key, prompt, mode='budget', max_output=512):
        seen.append(prompt[:80])
        if prompt.startswith('Rewrite this one sentence. Care'):
            return type('Reply', (), {'content': 'You seem to care about owning your results, and it seemed like you were trying to name the work.'})()
        if prompt.startswith('Rewrite this one sentence'):
            return type('Reply', (), {'content': history})()
        if 'Simplify' in prompt or prompt.startswith('Rewrite this one section'):
            return type('Reply', (), {'content': 'The point was clear.'})()
        return type('Reply', (), {'content': json.dumps({
            'bullets': [
                {
                    'behavior': 'You seem to care about your team, and it seemed like you were trying to name the work.',
                    'quote': 'Q1',
                    'why': 'The weather stayed cold all morning.',
                },
                {
                    'behavior': 'The line named the owner.',
                    'quote': 'Q2',
                    'why': history,
                },
            ],
            'yes_but': 'Next time, try this step.',
        })})()

    monkeypatch.setattr('lib.personal_narrative._rest_generate', fake_rest)

    class Client:
        google_api_key = 'test-key'
        model = 'gemini-3.1-flash-lite'

        def invoke(self, prompt, **kwargs):
            raise AssertionError('the entry point should call REST')

    transcript = {'utterances': [
        {'speaker': 'L', 'text': hinge + ' ' + quote, 'start': 0, 'end': 20000},
        {'speaker': 'L', 'text': (quote + ' ') * 6, 'start': 20000, 'end': 80000},
    ]}
    story = build_personal_story(
        Client(), transcript, 'L', {'name': 'Leslie'},
        {'share': 22, 'turns': 4, 'questions': 1, 'speaker_count': 6, 'words': 80},
    )
    assert any(prompt.startswith('Rewrite this one sentence. Care') for prompt in seen)
    assert any(prompt.startswith('Rewrite this one sentence') for prompt in seen)
    assert story['short_mode'] is False
    assert 'your team' not in story['bullets'][0].lower()


def test_a_prompt_example_does_not_become_stephens_goal_and_bare_bullets_drop(monkeypatch):
    from lib.personal_narrative import _prompt, _usable_model_bullets, build_personal_story, prompt_example_problem

    quote = "I've always been into making movies, making music, whatever it is."
    leaked = 'You seem to care about making movies, and it seemed like you were trying to own your results.'
    assert prompt_example_problem(leaked, quote)
    assert not prompt_example_problem(leaked, quote + ' The results took years.')
    instructions = _prompt('Stephen', [], [quote], [], '', '', None)
    assert 'owning your results' not in instructions
    assert 'your team' not in instructions
    assert '<what the quote shows they value>' in instructions

    why = 'That story shows people that making movies started early.'
    other = 'That moment lets others know the music still matters.'
    merged_bare = 'You seem to care about making movies, and it seemed like you were trying to make music.'
    quotes = [quote, 'Making music is the part he kept coming back to today.', 'The movies took years before anyone heard them.']
    kept = _usable_model_bullets({
        'bullets': [
            {'behavior': merged_bare, 'quote': 'Q1', 'why': ''},
            {'behavior': 'The line stayed on the music.', 'quote': 'Q2', 'why': other},
            {'behavior': 'The turn named the movies.', 'quote': 'Q3', 'why': ''},
        ],
    }, quotes, quote)
    assert kept[0].lower().startswith('you seem to care')
    assert len(kept) == 2
    assert 'we came to that conclusion because of' in kept[1].lower()

    dropped = _usable_model_bullets({
        'bullets': [
            {'behavior': merged_bare, 'quote': 'Q1', 'why': ''},
            {'behavior': 'The line stayed on the music.', 'quote': 'Q2', 'why': other},
            {'behavior': 'The turn named the movies.', 'quote': 'Q3', 'why': why},
        ],
    }, quotes, quote)
    assert len(dropped) == 2
    assert not dropped[0].lower().startswith('you seem to care')

    seen = []

    def fake_rest(model, key, prompt, mode='budget', max_output=512):
        seen.append(prompt)
        if prompt.startswith('Rewrite'):
            return type('Reply', (), {'content': 'You seem to care about making movies, and it seemed like you were trying to make music.'})()
        return type('Reply', (), {'content': json.dumps({
            'bullets': [
                {'behavior': leaked, 'quote': 'Q1', 'why': why},
                {'behavior': 'The line stayed on the music.', 'quote': 'Q2', 'why': other},
            ],
        })})()

    monkeypatch.setattr('lib.personal_narrative._rest_generate', fake_rest)

    class Client:
        google_api_key = 'test-key'
        model = 'gemini-3.1-flash-lite'

        def invoke(self, prompt, **kwargs):
            raise AssertionError('the entry point should call REST')

    story = build_personal_story(
        Client(),
        {'utterances': [{'speaker': 'S', 'text': (quote + ' Making music stayed with him for years. ') * 6, 'start': 0, 'end': 80000}]},
        'S',
        {'name': 'Stephen'},
        {'share': 20, 'turns': 2, 'questions': 1, 'speaker_count': 4, 'words': 90},
    )
    assert any('own your results' in prompt or 'prompt example' in prompt or prompt.startswith('Rewrite') for prompt in seen)
    assert all('owning your results' not in prompt for prompt in seen)
    assert 'own your results' not in ' '.join(story['bullets']).lower()
