import json

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


def test_signature_examples_may_be_invented_and_must_be_long_enough():
    from lib.llm_functions import _signature_fields

    bigram = 'we should'
    good = {
        'what': 'You reach for "we should" when you want the room to move with you.',
        'so_what': 'It gives you a moment to decide what you want to say next.',
        'now_what': 'You can pause once before you lean on that phrase again today.',
    }
    import json
    parsed = _signature_fields(json.dumps(good), 'we should try the plan', bigram)
    assert 'we should' in parsed['what']
    with pytest.raises(ValueError):
        _signature_fields(json.dumps({**good, 'so_what': 'Repetition.'}), 'we should', bigram)


def test_ban_list_catches_inflections_and_voice():
    with pytest.raises(OutputParserException):
        reject_craft_violations('You are leveraging the moment now.')
    with pytest.raises(OutputParserException):
        reject_craft_violations('You are empowering the room from here.')
    with pytest.raises(OutputParserException):
        reject_craft_violations('The speaker opened the plan for everyone.', speaker_label='A')
    with pytest.raises(OutputParserException):
        reject_craft_violations("That's when He stopped the story cold.", speaker_label='A')
    with pytest.raises(OutputParserException):
        reject_craft_violations('The student kept the turn after that.', personal_voice=True)


def test_mid_turn_quote_keeps_case_and_adds_an_ellipsis():
    from lib.story_validation import own_quote

    transcript = {
        'utterances': [
            {
                'speaker': 'A',
                'text': 'And you have to know this now please.',
                'start': 0,
                'end': 4000,
            }
        ]
    }
    shown = own_quote('You have to know this now please.', '0:00', transcript, 'A')
    assert shown.startswith('…')
    assert shown[1:].startswith('you have to')


def test_question_target_moves_up_and_talk_target_respects_the_norm():
    from lib.story_render import one_thing_from_gap, unique_habits

    quiet = one_thing_from_gap(share=20, speaker_count=7, questions=0, question_average=1)
    assert quiet[0].startswith('Ask one question')
    assert 'Target: 1 question.' in quiet[0]
    assert 'talk time' not in quiet[0]
    zero = one_thing_from_gap(share=25, speaker_count=4, questions=0, question_average=0)
    assert 'Target: 1 question.' in zero[0]
    even = one_thing_from_gap(share=21, speaker_count=4, questions=2, question_average=1)
    assert '18%' not in even[0]
    assert '% talk time' not in even[0]
    louder = one_thing_from_gap(share=10, speaker_count=4, questions=3, question_average=1)
    assert 'Target: 25% talk time.' in louder[0]
    assert louder[0].startswith('Say')
    loud = one_thing_from_gap(share=50, speaker_count=4, questions=3, question_average=1)
    assert loud[0].startswith('Pause after the point')
    assert 'Target: 25% talk time.' in loud[0]
    person = {
        'share': 50, 'speaker_count': 7, 'questions': 0, 'question_average': 2,
        'turn_words': 0, 'session_turn_words': 0, 'early': None, 'late': None,
    }
    habits = unique_habits([
        {'label': 'A', **person},
        {'label': 'B', 'share': 10, 'speaker_count': 7, 'questions': 0, 'question_average': 2, 'turn_words': 0, 'session_turn_words': 0, 'early': None, 'late': None},
        {'label': 'C', **person},
    ])
    assert len({habits['A'][0], habits['B'][0], habits['C'][0]}) == 3
    assert habits['A'][0].startswith('Pause')
    assert habits['C'][0].startswith('Ask')


def test_group_character_accepts_one_fact_and_the_page_holds_the_chord(tmp_path):
    from PIL import Image

    from lib.pdf_functions import MM_PDF
    from lib.story_pdf import write_group_story
    from lib.story_render import render_group_section

    facts = {
        'interaction': {'value': 42, 'unit': 'percent', 'line': '42% interaction'},
        'participation': {'value': 80, 'unit': 'percent', 'line': '80% participation'},
        'word_interaction': {'value': 39, 'unit': 'percent', 'line': '39% of the words exchanged'},
        'target_interaction': {'value': 56, 'unit': 'percent', 'line': '56% interaction'},
    }
    rendered = render_group_section(
        'character',
        {'headline': 'A working session', 'number_keys': ['interaction']},
        transcript=synthetic_transcript(),
        facts=facts,
        cast='Who talks to whom.',
    )
    assert '42% interaction' in rendered['character']
    resolved = render_group_section(
        'resolution',
        {'claim': 'the budget held'},
        transcript=synthetic_transcript(),
        facts=facts,
        cast='Who talks to whom.',
    )
    assert resolved['resolution'].startswith('The meeting closed on')
    assert 'numbers from here' in resolved['resolution']
    assert 'as stated by' not in resolved['resolution'].lower()
    image = tmp_path / 'chord.png'
    Image.new('RGB', (400, 400), 'white').save(image)
    pdf = MM_PDF(path_report_assets=__import__('pathlib').Path('report_assets'))
    story = {
        'headline': 'A working session',
        'character': rendered['character'],
        'goal': 'The turns keep returning to the onboarding plan.',
        'challenge': 'Interaction was 42%.',
        'drama': 'Who talks to whom.',
        'resolution': 'The meeting closed on the budget.',
        'one_thing': 'Ask one question back before the next decision. Target: 56% interaction.',
        'why': 'Interaction is 42%, so 56% is a step up.',
        'transitions': {},
    }
    write_group_story(pdf, story, image, [('Alex Rivera', 'Blake Chen', 2, 1)])
    assert pdf._chord_page == 1
    pdf.output(tmp_path / 'group.pdf')
    pdf.set_y(-15)
    ratios = pdf.page_fill_ratios()
    for ratio in ratios[:-1]:
        assert ratio >= 0.40
    assert ratios[-1] >= 0.15


def test_a_heading_is_not_stranded_at_the_bottom_of_a_page(tmp_path):
    from lib.pdf_functions import MM_PDF
    from lib.story_pdf import _body, _heading

    pdf = MM_PDF(path_report_assets=__import__('pathlib').Path('report_assets'))
    pdf.add_page()
    pdf.set_y(pdf.h - pdf.b_margin - 12)
    previous = pdf.page
    _heading(pdf, 'Drama')
    _body(pdf, 'You opened on the plan and the next voice took the turn after you.')
    assert pdf._heading_page == pdf.page
    assert pdf._heading_page == previous + 1


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
        ('goal', {'claim': 'move the onboarding plan into motion', 'quote': {'text': opening, 'ts': '0:00'}}),
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
        ('goal', {'claim': 'test the budget against the plan', 'quote': {'text': opening, 'ts': '0:00'}}),
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


def test_rejection_reason_is_sent_on_the_retry():
    from tests.fakes import FakeLLM
    from lib.story_generation import generate_personal_draft, speaker_stats_for_story

    class JourneyOnce(FakeLLM):
        def _reply(self, text):
            if 'BATCH:' in text and 'was rejected' not in text:
                import json
                payload = json.loads(super()._reply(text))
                payload['goal']['claim'] = 'This journey stays with the plan'
                return json.dumps(payload)
            return super()._reply(text)

    llm = JourneyOnce()
    transcript = synthetic_transcript()
    speaker_map = {'A': {'name': 'Alex Rivera'}, 'B': {'name': 'Blake Chen'}}
    draft = generate_personal_draft(
        llm,
        transcript=transcript,
        label='A',
        info=speaker_map['A'],
        stats=speaker_stats_for_story(transcript, speaker_map, 'A'),
        forbidden_names=['Blake Chen'],
    )
    assert 'goal' not in draft['fallback_sections']
    assert any('was rejected' in call and 'journey' in call for call in llm.calls)
    assert all('NEXT SPEAKER' not in call for call in llm.calls if 'BATCH: personal' in call)


def test_gemini_timeout_is_retried_without_adding_a_story_call():
    from lib.gemini_retry import invoke_llm

    class Boom:
        def __init__(self):
            self.calls = 0

        def invoke(self, prompt):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError('504 Gateway Timeout')
            return type('Reply', (), {'content': 'ok'})()

    boom = Boom()
    reply = invoke_llm(boom, 'hello', pause=lambda _delay: None)
    assert reply.content == 'ok'
    assert boom.calls == 2


def test_relayed_story_is_not_written_as_the_readers_life():
    transcript = {
        'utterances': [
            {
                'speaker': 'S',
                'text': (
                    'A student of mine moved to New York. '
                    'Her family is still in Iraq and they speak Farsi at home.'
                ),
                'start': 0,
                'end': 8000,
            }
        ]
    }
    with pytest.raises(OutputParserException):
        reject_craft_violations(
            'You grew up in New York and your family speaks Farsi.',
            transcript=transcript,
            speaker_label='S',
        )
    kept = reject_craft_violations(
        "You told a student's story about New York and a family in Iraq.",
        transcript=transcript,
        speaker_label='S',
    )
    assert "student's story" in kept


def test_voice_post_fix_and_unquoted_first_person():
    from lib.story_validation import normalize_mid_sentence_capitals

    assert normalize_mid_sentence_capitals("that's when You stopped.") == "that's when you stopped."
    assert normalize_mid_sentence_capitals('It buys Living in the room.') == 'It buys living in the room.'
    fixed = reject_craft_violations("That's when You stopped the turn.", speaker_label='A')
    assert 'you stopped' in fixed
    with pytest.raises(OutputParserException):
        reject_craft_violations('The individual opened the plan today.', speaker_label='A')
    with pytest.raises(OutputParserException):
        reject_craft_violations('I will draft the timeline tomorrow.', speaker_label='A')
    quoted = reject_craft_violations('You said "I will draft the timeline."', speaker_label='A')
    assert 'I will' in quoted


def test_group_headline_and_halfway_target_are_about_the_room():
    from lib.story_render import group_halfway_target, render_group_section

    replaced = render_group_section(
        'character',
        {'headline': 'A determined creative professional'},
        transcript=synthetic_transcript(),
        facts={
            'interaction': {'value': 20, 'unit': 'percent', 'line': '20% interaction'},
            'participation': {'value': 80, 'unit': 'percent', 'line': '80% participation'},
            '_portrait': {
                'pair': ('Alex Rivera', 'Blake Chen'),
                'least': 'Leslie',
                'low_share': 6,
                'high_share': 40,
            },
        },
        cast='Who talks to whom.',
    )
    assert 'creative' not in replaced['headline'].lower()
    assert '20% interaction' in replaced['character']
    assert 'Alex Rivera and Blake Chen' in replaced['character']
    target, norm, name = group_halfway_target(20, 80, 40)
    assert target == 50
    assert norm == 80
    assert name == 'participation'
    rendered = render_group_section(
        'character',
        {'headline': 'A working session', 'number_keys': ['interaction']},
        transcript=synthetic_transcript(),
        facts={
            'interaction': {'value': 20, 'unit': 'percent', 'line': '20% interaction'},
            'participation': {'value': 80, 'unit': 'percent', 'line': '80% participation'},
            '_portrait': {
                'pair': ('Alex Rivera', 'Blake Chen'),
                'least': 'Leslie',
                'low_share': 6,
                'high_share': 40,
            },
        },
        cast='Who talks to whom.',
    )
    sentences = [part for part in rendered['character'].split('.') if part.strip()]
    assert 2 <= len(sentences) <= 3
    assert '20% interaction' in rendered['character']
    assert 'Leslie' in rendered['character']


def test_clock_uses_hours_after_sixty_minutes():
    from lib.story_evidence import clock
    from lib.story_validation import own_quote

    assert clock((108 * 60 + 43) * 1000) == '1:48:43'
    assert clock(160 * 60 * 1000) == '2:40:00'
    assert clock(38 * 1000) == '0:38'
    transcript = {
        'utterances': [
            {
                'speaker': 'A',
                'text': 'We should try the new onboarding plan today.',
                'start': (108 * 60 + 43) * 1000,
                'end': (108 * 60 + 50) * 1000,
            }
        ]
    }
    shown = own_quote('We should try the new onboarding plan today.', '108:43', transcript, 'A')
    assert 'onboarding' in shown


def test_facilitator_is_not_cut_to_an_even_share_and_later_talk_is_not_zero():
    from lib.story_render import one_thing_from_gap, unique_habits

    annie = one_thing_from_gap(
        share=24, speaker_count=14, questions=2, question_average=1, facilitator=True,
    )
    assert '7% talk time' not in annie[0]
    assert 'even share' not in annie[1]
    habits = unique_habits([
        {
            'label': 'A', 'role': 'facilitator', 'share': 24, 'speaker_count': 14,
            'questions': 2, 'question_average': 1, 'turn_words': 0, 'session_turn_words': 0,
            'early': None, 'late': None,
        },
    ])
    assert '7%' not in habits['A'][0]
    below = one_thing_from_gap(
        share=20, speaker_count=7, questions=2, question_average=1, early=0, late=17,
    )
    above = one_thing_from_gap(
        share=20, speaker_count=7, questions=2, question_average=1, early=10, late=50,
    )
    assert 'Target: 33%' in below[0]
    assert 'Target: 33%' in above[0]
    assert 'Target: 8%' not in below[0]
    assert 'even third is 33%' in below[1]


def test_signature_keeps_a_reading_that_only_has_to_name_the_bigram():
    from lib.llm_functions import _signature_fields
    import json

    long_line = 'You reach for "we should" when they want the room to move and then you keep going past the point today indeed.'
    parsed = _signature_fields(
        json.dumps({
            'what': long_line,
            'so_what': 'They hear the phrase as the way you open a new point in the room.',
            'now_what': 'You can pause once before you lean on that phrase again today.',
        }),
        'we should try the plan',
        'we should',
    )
    assert 'we should' in parsed['what'].lower()
    assert len(parsed['what'].split()) <= 25
    with pytest.raises(ValueError):
        _signature_fields(
            json.dumps({
                'what': 'You reach for another line when the room is waiting on you today.',
                'so_what': 'It gives you a moment to decide what you want to say next.',
                'now_what': 'You can pause once before you lean on that phrase again today.',
            }),
            '',
            'we should',
        )


def test_closing_line_skips_fragments_and_the_next_sentence_is_capitalized():
    from lib.story_render import render_group_section, substantive_line

    assert not substantive_line('It.')
    assert not substantive_line("So you're")
    assert not substantive_line('Could we check the budget first?')
    assert substantive_line('That might work if we start Monday.')
    facts = {
        'interaction': {'value': 42, 'unit': 'percent', 'line': '42% interaction'},
        'participation': {'value': 80, 'unit': 'percent', 'line': '80% participation'},
    }
    rendered = render_group_section(
        'resolution',
        {'claim': 'you told the room where the budget stood, as stated by "Ric Viers"'},
        transcript=synthetic_transcript(),
        facts=facts,
        cast='',
    )
    assert '." You told' in rendered['resolution']
    assert 'as stated by' not in rendered['resolution'].lower()
    with pytest.raises(OutputParserException):
        render_group_section(
            'goal',
            {'claim': "Annie's theme for the whole hour"},
            transcript=synthetic_transcript(),
            facts=facts,
            cast='',
        )


def test_drama_cost_is_about_the_speaker_not_the_employee_in_the_story():
    opening = 'We should try the new onboarding plan.'
    rendered = render_personal_section(
        'drama',
        {
            'patterns': [{
                'quotes': [{'text': opening, 'ts': '0:00'}],
                'pattern': 'You set the plan in the first line. The room shifted — and then nothing.',
                'twist': 'the next voice took the turn',
                'cost': 'the employee endured the pressure',
                'gift': 'the employee kept the job',
            }],
            'exception': {
                'quote': 'That might work if we start Monday.',
                'ts': '0:11',
                'what_happened': 'you named Monday',
            },
        },
        transcript=synthetic_transcript(),
        label='A',
        facts=FACTS,
        forbidden=[],
    )
    assert 'employee' not in rendered['drama'].lower()
    assert 'What it gets you: you put the point in your own words.' in rendered['drama']
    assert '—' not in rendered['drama']
    assert 'Here is the repeated move' not in rendered['drama']


def test_gift_and_cost_take_a_clause():
    from lib.story_render import _costs_you, _gets_you, turns_sentence

    assert _gets_you('you establish your core identity') == (
        'What it gets you: you establish your core identity.'
    )
    assert _gets_you('you you establish your core identity') == (
        'What it gets you: you establish your core identity.'
    )
    assert _costs_you('endure constant pressure') == (
        'What it costs you: you endure constant pressure.'
    )
    assert _costs_you('you endure constant pressure') == (
        'What it costs you: you endure constant pressure.'
    )
    with pytest.raises(OutputParserException):
        _gets_you('you')
    assert '1 turn' in turns_sentence(1)
    assert '1 turns' not in turns_sentence(1)
    for options in list(TRANSITIONS.values()) + list(GROUP_TRANSITIONS.values()):
        for line in options:
            assert line not in {
                'Here is where the session landed.',
                'One change follows from the largest gap.',
                'That is the shape of the session.',
                'Here is where the meeting landed.',
            }


def test_fill_metric_ignores_the_footer(tmp_path):
    from lib.pdf_functions import MM_PDF

    pdf = MM_PDF(path_report_assets=__import__('pathlib').Path('report_assets'))
    pdf.add_page()
    pdf.set_font('Poppins', '', 11)
    pdf.multi_cell(w=0, text='One short line so the last page is not full.', align='L')
    pdf.output(tmp_path / 'short.pdf')
    pdf.set_y(-15)
    ratios = pdf.page_fill_ratios()
    assert ratios
    assert ratios[-1] < 0.5
    assert all(ratio <= 1 for ratio in ratios)


def test_goal_wrapper_does_not_glue_a_second_sentence():
    from lib.story_render import _topic_phrase, capitalize_sentences, render_group_section

    assert _topic_phrase('The group seeks to identify the core need') == 'identify the core need'
    rendered = render_group_section(
        'goal',
        {'claim': 'The group seeks to identify the core need'},
        transcript=synthetic_transcript(),
        facts={'interaction': {'value': 20, 'line': '20% interaction', 'unit': 'percent'}},
        cast='',
    )
    assert 'to The group' not in rendered['goal']
    assert rendered['goal'] == 'The room set out to identify the core need.'
    assert capitalize_sentences('average. you struggle') == 'average. You struggle'


def test_closer_ignores_the_logistics_tail_and_short_clips():
    from lib.story_render import closing_clip, last_substantive_utterance

    assert closing_clip('good sign.') is None
    assert closing_clip("So you're") is None
    rows = [
        {'speaker': 'A', 'text': 'We should settle the budget and the timeline before we leave.'},
        {'speaker': 'B', 'text': 'The budget and the timeline are the two open items.'},
        {'speaker': 'A', 'text': 'good sign.'},
        {'speaker': 'B', 'text': "So you're"},
        {'speaker': 'A', 'text': 'Take a photo and I will send the Eventbrite link after.'},
    ]
    chosen = last_substantive_utterance(rows)
    assert 'budget' in chosen['text']
    assert closing_clip(chosen['text'])
    assert len(closing_clip(chosen['text']).split()) >= 6


def test_identical_question_habits_are_not_shared():
    from lib.story_render import unique_habits

    person = {
        'share': 12, 'speaker_count': 14, 'questions': 0, 'question_average': 14,
        'turn_words': 0, 'session_turn_words': 0, 'early': None, 'late': None, 'role': 'participant',
    }
    habits = unique_habits([
        {'label': 'Lynae', **person},
        {'label': 'Eden', **person},
    ])
    assert habits['Lynae'][0] != habits['Eden'][0]
    assert sum('14 question' in sentence for sentence, _why in habits.values()) == 1


def test_a_string_section_is_rejected_instead_of_crashing():
    from lib.story_generation import _fill_sections

    class OneString:
        def invoke(self, prompt):
            return type('Reply', (), {'content': '{"goal": "just a string"}'})()

    draft = {'accepted': []}
    _fill_sections(
        OneString(),
        audience='personal',
        sections=('goal',),
        evidence='FACTS',
        render=lambda section, data: data['claim'],
        progress_path=None,
        draft=draft,
    )
    assert 'goal' in draft['fallback_sections']


def test_signature_ban_list_rejects_leverage_and_cache_notices_a_gap():
    from lib.llm_functions import _signature_fields, _signatures_present
    import json

    with pytest.raises(ValueError):
        _signature_fields(
            json.dumps({
                'what': 'You reach for "we should" when you want the room to move with you.',
                'so_what': 'It gives you a moment to decide what you want to say next.',
                'now_what': 'You can leverage that phrase before you speak again today.',
            }),
            '',
            'we should',
        )
    assert not _signatures_present({'A': {'personal_signature_phrase': None}, 'B': {'personal_signature_phrase': {'bigram': 'we should'}}})
    assert _signatures_present({'A': {'personal_signature_phrase': {'bigram': 'we should'}}})


def test_panelists_are_not_inferred_to_be_facilitators():
    from app.analysis import apply_speaker_roles

    speaker_map = {
        'A': {'name': 'Annie'},
        'B': {'name': 'Eden'},
        'C': {'name': 'Stephanie'},
        'D': {'name': 'Jrick'},
        'H': {'name': 'Host'},
    }
    roles = {
        'A': 'facilitator', 'B': 'participant', 'C': 'participant',
        'D': 'participant', 'H': 'host',
    }
    apply_speaker_roles(speaker_map, roles, {'B', 'C', 'D'})
    assert speaker_map['A']['role'] == 'facilitator'
    assert speaker_map['H']['role'] == 'host'
    assert [speaker_map[label]['role'] for label in ('B', 'C', 'D')] == ['participant'] * 3


def test_stock_pipeline_puts_the_facilitator_role_on_the_speaker_map():
    from app.analysis import apply_speaker_roles

    speaker_map = {'A': {'name': 'Annie'}, 'B': {'name': 'Corey'}}
    apply_speaker_roles(speaker_map, {'A': 'facilitator', 'B': 'participant'}, {'B'})
    assert speaker_map['A']['role'] == 'facilitator'
    assert speaker_map['B']['role'] == 'participant'


def test_closer_rejects_a_mid_clause_cut_and_the_window_drops_logistics():
    from lib.session_window import apply_session_window, parse_clock_ms
    from lib.story_render import closing_clip, last_substantive_utterance

    broken = "good sign. If Stan if stands, that's"
    assert closing_clip(broken) is None
    rows = [
        {'speaker': 'A', 'text': 'We should settle the budget and the timeline before we leave.', 'start': 0, 'end': 4000},
        {'speaker': 'B', 'text': broken, 'start': 4000, 'end': 8000},
        {'speaker': 'A', 'text': 'Smile for the photo.', 'start': 8000, 'end': 9000},
        {'speaker': 'B', 'text': 'They closed about 9.', 'start': 9000, 'end': 10000},
        {'speaker': 'A', 'text': 'Thanks, goodbye.', 'start': 10000, 'end': 11000},
        {'speaker': 'C', 'text': 'Present for the picture.', 'start': 11000, 'end': 12000},
        {'speaker': 'B', 'text': 'We are packing up now.', 'start': 12000, 'end': 13000},
    ]
    chosen = last_substantive_utterance(rows)
    assert chosen['text'].startswith('We should settle')
    trimmed = apply_session_window({'utterances': rows, 'sentiment_analysis_results': rows})
    assert [row['text'] for row in trimmed['utterances']] == [rows[0]['text']]
    assert trimmed['session_window']['auto_end'] is True
    explicit = apply_session_window({'utterances': rows}, start_ms=0, end_ms=parse_clock_ms('0:04'))
    assert explicit['utterances'][0]['text'].startswith('We should settle')
    assert all(row['start'] <= 4000 for row in explicit['utterances'])
    import json
    from pathlib import Path

    from lib.session_window import main

    folder = Path(__import__('tempfile').mkdtemp())
    source = folder / 'transcript.json'
    source.write_text(json.dumps({'utterances': rows}), encoding='utf-8')
    output = folder / 'trimmed.json'
    assert main(['--session-start', '0:00', '--session-end', '0:04', str(source), '--output', str(output)]) == 0
    saved = json.loads(output.read_text(encoding='utf-8'))
    assert saved['utterances'][0]['text'].startswith('We should settle')


def test_render_capitalizes_after_a_quote_and_collapses_a_stray_period():
    from lib.story_render import capitalize_sentences

    assert capitalize_sentences('questions. you struggle') == 'questions. You struggle'
    assert capitalize_sentences('form.". integrate') == 'form." Integrate'
    assert capitalize_sentences('form.".') == 'form."'


def test_habits_do_not_leak_diarization_letters_and_the_why_matches():
    from lib.story_render import unique_habits

    shared = {
        'share': 8, 'speaker_count': 14, 'questions': 4, 'question_average': 2,
        'turn_words': 20, 'session_turn_words': 20, 'early': 10, 'late': 70, 'role': 'participant',
    }
    habits = unique_habits([
        {'label': 'N', **shared},
        {'label': 'H', **shared},
        {'label': 'Corey', **shared},
    ])
    for label, (sentence, why) in habits.items():
        assert 'This one is for N.' not in sentence
        assert 'This one is for H.' not in sentence
        assert 'This one is for' not in sentence
        if 'last third' in sentence:
            assert 'third' in why.lower()
        elif 'question' in sentence.lower():
            assert 'question' in why.lower()
        elif 'talk time' in sentence:
            assert 'talk time' in why.lower()
        else:
            assert 'pause' in why.lower() or 'handoff' in why.lower()
            assert 'third' not in why.lower()
    assert len({item[0] for item in habits.values()}) == 3


def test_challenge_rejects_inquiries_when_the_speaker_asked_them():
    from lib.story_render import personal_facts, question_directions

    transcript = {
        'utterances': [
            {'speaker': 'A', 'text': 'Where does this break? Who owns the next step? Why now?', 'start': 0, 'end': 4000},
            {'speaker': 'B', 'text': 'I can take the first one.', 'start': 4000, 'end': 6000},
            {'speaker': 'A', 'text': 'And what happens if we wait?', 'start': 6000, 'end': 8000},
        ]
    }
    asked, received = question_directions(transcript, 'A')
    assert asked == 4
    assert received == 0
    facts = personal_facts(
        transcript=transcript, label='A', share=40, turns=2, questions=4,
        question_average=1, speaker_count=2,
    )
    assert facts['questions_asked']['line'] == 'You asked 4 questions.'
    assert facts['questions_received']['value'] == 0
    with pytest.raises(OutputParserException, match='aimed at you'):
        render_personal_section(
            'challenge',
            {'claim': 'You faced a high volume of inquiries', 'fact_key': 'questions'},
            transcript=transcript,
            label='A',
            facts=facts,
            forbidden=[],
        )


def test_utilize_and_communication_style_are_banned_in_story_prose():
    with pytest.raises(OutputParserException):
        reject_craft_violations('You utilize the opening to set the plan for the room.')
    with pytest.raises(OutputParserException):
        reject_craft_violations('That communication style showed up in the first turn of the hour.')


def test_goal_strips_a_leading_infinitive_down_to_the_noun_phrase():
    from lib.story_render import _topic_phrase, render_group_section

    assert _topic_phrase('returning to identify where value breaks') == 'where value breaks'
    assert _topic_phrase('the primary objective involves identifying the break') == 'the break'
    rendered = render_group_section(
        'goal',
        {'claim': 'returning to identify where value breaks when the client leaves'},
        transcript=synthetic_transcript(),
        facts={'interaction': {'value': 20, 'line': '20% interaction', 'unit': 'percent'}},
        cast='',
    )
    assert rendered['goal'] == 'The room set out to identify where value breaks when the client leaves.'


def test_group_character_keeps_the_facts_when_the_headline_is_personal(tmp_path):
    from lib.story_generation import _fill_sections, write_fallback_reasons
    from lib.story_render import render_group_section

    facts = {'interaction': {'value': 42, 'line': '42% interaction', 'unit': 'percent'}, 'participation': {'value': 80}}
    rendered = render_group_section(
        'character',
        {'headline': 'A determined creative professional'},
        transcript=synthetic_transcript(),
        facts=facts,
        cast='',
    )
    assert 'creative' not in rendered['headline'].lower()
    assert '42% interaction' in rendered['character']

    class PersonalHeadline:
        def invoke(self, prompt):
            return type('Reply', (), {'content': '{"character": {"headline": "A determined creative professional"}}'})()

    draft = {'accepted': []}
    _fill_sections(
        PersonalHeadline(),
        audience='group',
        sections=('character',),
        evidence='FACTS',
        render=lambda section, data: render_group_section(
            section, data, transcript=synthetic_transcript(), facts=facts, cast=''
        ),
        progress_path=None,
        draft=draft,
    )
    assert 'character' not in draft.get('fallback_sections') or []
    assert '42% interaction' in draft['character']
    write_fallback_reasons(tmp_path, {'group:character': 'group headline must describe the room'})
    saved = json.loads((tmp_path / 'fallbacks.json').read_text(encoding='utf-8'))
    assert saved['group:character'] == 'group headline must describe the room'


def test_annie_like_seven_person_report_compacts_a_thin_last_page(tmp_path):
    from lib.pdf_functions import MM_PDF
    from lib.story_pdf import fit_personal_report, write_personal_report

    # Eden, Stephanie, and Jrick each finished on a page under 15% full.
    names = ['Annie', 'Corey', 'Eden', 'Stephanie', 'Jrick', 'Leslie', 'Noah']
    short = 'You set one point down and the next voice answered it before the hour moved on.'
    story = {
        'headline': 'You came in and set a point down',
        'numbers': [
            '8% talk time: below the room average.',
            '1 question: below the session average of 4 questions.',
            '6 turns: you came in 6 times.',
        ],
        'character_body': short,
        'share': 8,
        'goal': short,
        'challenge': short,
        'drama': short,
        'resolution': short,
        'one_thing': 'Ask one question before you add another point. Target: 4 questions.',
        'why': ' '.join(['The question count is the gap this session can measure next time.'] * 12),
        'transitions': {},
        'signature': {'phrase': '', 'count': 0, 'quotes': [], 'what': '', 'so_what': '', 'now_what': ''},
        'lines': [],
        'session_end_ms': 0,
        'appendix': {name: 'One line for the lens.' for name in ('AMASE', 'Enneagram', 'Doshas')},
    }
    assert len(names) == 7

    def make_pdf():
        return MM_PDF(path_report_assets=__import__('pathlib').Path('report_assets'))

    plain = make_pdf()
    write_personal_report(plain, story, tmp_path / 'charts')
    plain_ratios = plain.page_fill_ratios()
    assert plain_ratios[-1] < 0.15
    fitted = fit_personal_report(story, tmp_path / 'fitted', make_pdf)
    assert fitted.page_fill_ratios()[-1] >= 0.15


def test_personal_layout_fits_two_pages_and_keeps_the_doshas_line(tmp_path):
    from pypdf import PdfReader

    from lib.pdf_functions import MM_PDF
    from lib.story_pdf import fit_personal_report, write_personal_report

    dosha = (
        'Kapha (4/5). The steady pace keeps the room with you, and the last clause of this '
        'Doshas note has to stay visible even when the page is nearly full of the section above it.'
    )
    sentence = 'The room kept returning to the same point before anyone answered the question.'
    long = ' '.join([sentence] * 12)
    story = {
        'headline': 'Leslie',
        'numbers': ['8% talk time: below the room average.'] * 3,
        'character_body': long,
        'share': 8,
        'goal': long,
        'challenge': long,
        'drama': long,
        'resolution': long,
        'one_thing': 'Ask one question before you add another point. Target: 4 questions.',
        'why': 'The question count is the gap this session can measure next time.',
        'transitions': {},
        'signature': {
            'phrase': 'the room',
            'count': 2,
            'quotes': ['You said the room needed a longer line before the close today.'],
            'what': 'You reach for the room.',
            'so_what': 'It gives the next voice a place to start.',
            'now_what': 'You can leave one beat before the next line.',
        },
        'lines': [{'start': 0, 'text': 'hello'}],
        'session_end_ms': 60000,
        'appendix': {
            'AMASE': 'One line for the lens.',
            'Enneagram': 'One line for the lens.',
            'Doshas': dosha,
        },
    }

    def make_pdf():
        return MM_PDF(path_report_assets=__import__('pathlib').Path('report_assets'))

    plain = make_pdf()
    write_personal_report(plain, story, tmp_path / 'plain')
    assert len(plain.page_fill_ratios()) == 3
    fitted = fit_personal_report(story, tmp_path / 'fitted', make_pdf)
    assert len(fitted.page_fill_ratios()) == 2
    fitted.output(tmp_path / 'fitted.pdf')
    text = '\n'.join(page.extract_text() or '' for page in PdfReader(tmp_path / 'fitted.pdf').pages)
    collapsed = ' '.join(text.split())
    assert 'Leslie' in collapsed
    assert 'L e s l i e' not in collapsed
    assert 'Ask one question before you add another point' in collapsed
    assert 'last clause of this Doshas note has to stay visible' in collapsed


def test_auto_end_stops_on_the_last_topic_turn_not_the_posing_line():
    from lib.session_window import apply_session_window, parse_clock_ms
    from lib.story_render import is_logistics

    posing = "middle or you're going to stay there."
    assert is_logistics(
        'Let me stand up and gather everyone into the middle so the picture includes the whole panel after the hour.'
    )
    rows = [
        {
            'speaker': 'A',
            'role': 'participant',
            'text': 'Knowledge management hides the value when the source is buried in the system.',
            'start': parse_clock_ms('2:40:00'),
            'end': parse_clock_ms('2:41:10'),
        },
        {
            'speaker': 'B',
            'role': 'participant',
            'text': 'The value breaks again when knowledge management keeps the source from the next person.',
            'start': parse_clock_ms('2:42:25'),
            'end': parse_clock_ms('2:43:40'),
        },
        {
            'speaker': 'H',
            'role': 'host',
            'text': 'Thank you all for staying with the panel.',
            'start': parse_clock_ms('2:43:50'),
            'end': parse_clock_ms('2:44:04'),
        },
        {
            'speaker': 'Jay',
            'role': 'participant',
            'text': posing,
            'start': parse_clock_ms('2:49:56'),
            'end': parse_clock_ms('2:50:20'),
        },
    ]
    trimmed = apply_session_window({'utterances': rows})
    assert posing not in [row['text'] for row in trimmed['utterances']]
    assert trimmed['session_window']['end_ms'] == parse_clock_ms('2:44:04')
    assert trimmed['utterances'][-1]['text'].startswith('Thank you')


def test_timeline_axis_starts_at_the_window_not_zero():
    from lib.story_pdf import _clock_label, timeline_bounds

    start_ms = (46 * 60 + 57) * 1000
    start, _end = timeline_bounds({
        'session_start_ms': start_ms,
        'session_end_ms': (2 * 3600 + 44 * 60) * 1000,
        'lines': [{'start': start_ms, 'end': start_ms + 4000}],
    })
    assert _clock_label(start) == '46:57'
    assert _clock_label(start) != '0:00'


def test_goal_sentences_are_not_wrapped_as_noun_phrases():
    from lib.story_render import render_group_section

    facts = {'interaction': {'value': 20, 'line': '20% interaction', 'unit': 'percent'}}
    aims = render_group_section(
        'goal',
        {'claim': 'the group aims to identify where value breaks within knowledge management systems'},
        transcript=synthetic_transcript(),
        facts=facts,
        cast='',
    )['goal']
    autonomy = render_group_section(
        'goal',
        {'claim': 'the objective is to maintain personal autonomy'},
        transcript=synthetic_transcript(),
        facts=facts,
        cast='',
    )['goal']
    assert aims == (
        'The room set out to identify where value breaks within knowledge management systems.'
    )
    assert autonomy == 'The room set out to maintain personal autonomy.'
    assert 'returning to the group aims' not in aims
    assert 'returning to the objective is' not in autonomy
    normalized = render_group_section(
        'goal',
        {'claim': 'the objective involves maintaining personal autonomy'},
        transcript=synthetic_transcript(),
        facts=facts,
        cast='',
    )['goal']
    assert normalized == 'The room set out to maintain personal autonomy.'


def test_a_json_string_or_list_is_coerced_into_a_section_object(caplog):
    import json
    import logging

    from lib.story_generation import _fill_sections

    class Coercible:
        def invoke(self, prompt):
            return type('Reply', (), {'content': json.dumps({
                'goal': json.dumps({'claim': 'the budget held through the hour'}),
                'resolution': ['```json\n{"claim": "the budget held through the hour"}\n```'],
            })})()

    draft = {'accepted': []}
    with caplog.at_level(logging.WARNING):
        _fill_sections(
            Coercible(),
            audience='group',
            sections=('goal', 'resolution'),
            evidence='FACTS',
            render=lambda section, data: {section: data['claim']},
            progress_path=None,
            draft=draft,
        )
    assert 'goal' not in draft.get('fallback_sections', [])
    assert 'resolution' not in draft.get('fallback_sections', [])
    assert 'str' in caplog.text
    assert 'list' in caplog.text


def test_character_accepts_one_fact_key_and_rejects_another_speakers_line():
    transcript = {
        'utterances': [
            {'speaker': 'A', 'text': 'The onboarding plan needs a budget before Monday.', 'start': 0, 'end': 4000},
            {'speaker': 'B', 'text': 'I work as a dating coach on the weekends now.', 'start': 4000, 'end': 8000},
        ]
    }

    def render(data):
        return render_personal_section(
            'character', data, transcript=transcript, label='A', facts=FACTS, forbidden=[],
        )

    kept = render({'headline': 'You opened on the onboarding plan', 'number_keys': ['talk_share']})
    assert kept['numbers']
    with pytest.raises(OutputParserException):
        render({'headline': 'You kept the dating coach on the weekends', 'number_keys': ['talk_share']})


def test_cost_without_a_verb_stays_a_noun_phrase_and_drama_skips_short_quotes():
    from lib.data_story import personal_story
    from lib.story_render import _costs_you, render_group_section

    assert _costs_you('you decades of professional struggle') == (
        'What it costs you: decades of professional struggle.'
    )
    transcript = {
        'utterances': [
            {'speaker': 'A', 'text': 'us?', 'start': 0, 'end': 500},
            {'speaker': 'A', 'text': "we're doing.", 'start': 500, 'end': 1000},
            {'speaker': 'A', 'text': 'particular question?', 'start': 1000, 'end': 1500},
            {
                'speaker': 'A',
                'text': 'The knowledge management system hides the value until someone names the source.',
                'start': 1500,
                'end': 5000,
            },
        ]
    }
    story = personal_story(
        label='A',
        info={'name': 'Eden'},
        analysis={},
        transcript=transcript,
        stats_frame=None,
        framework={},
        draft={},
    )
    assert 'us?' not in story['drama']
    assert "we're doing." not in story['drama']
    assert 'particular question?' not in story['drama']
    assert 'knowledge management' in story['drama']
    with pytest.raises(OutputParserException):
        render_group_section(
            'drama',
            {'pattern_note': 'the value breaks when'},
            transcript=synthetic_transcript(),
            facts={'interaction': {'value': 20, 'line': '20% interaction', 'unit': 'percent'}},
            cast='Who talks to whom',
        )


def test_prompts_name_the_whole_ban_list_and_quotes_stay_distinct():
    from lib.names import table_name
    from lib.story_generation import _batch_prompt
    from lib.story_validation import ban_instruction

    prompt = _batch_prompt('personal', ('drama',), 'FACTS')
    assert ban_instruction() in prompt
    assert 'Never use:' in prompt
    assert 'journey' in prompt and 'leverage' in prompt
    assert table_name('Speaker F') == 'F'
    assert table_name('Full Sail Moderator') == 'Full Sail Moderator'
    from lib.names import pair_cell

    assert pair_cell('Audience member 1', 'Dr. Annie Green') == 'Aud. member 1\nand Dr. Annie Green'


def test_forty_turn_window_ends_on_the_host_thanks_block():
    from lib.session_window import apply_session_window, closer_turn

    def turn(minute, speaker, text, role='participant'):
        start = minute * 60 * 1000
        return {
            'speaker': speaker,
            'role': role,
            'text': text,
            'start': start,
            'end': start + 20_000,
        }

    rows = []
    for index in range(31):
        rows.append(turn(
            index,
            'A' if index % 2 == 0 else 'B',
            'The knowledge systems still hide the design until the workflow is named by the room.',
            'moderator' if index % 2 == 0 else 'participant',
        ))
    closer = turn(
        32,
        'A',
        'The knowledge systems still hide the design until the workflow is named for the record.',
        'moderator',
    )
    host = turn(33, 'H', 'Thank you all, and thank the panelists.', 'host')
    audience = turn(34, 'Aud', 'Thank the panelists for the hour.', 'audience')
    applause = turn(35, 'H', 'Give them a round of applause.', 'host')
    books = turn(36, 'H', 'Her books are on the table if you want one.', 'host')
    posing = turn(50, 'Jay', "middle or you're going to stay there.", 'participant')
    fragment = turn(51, 'Jay', "good sign. If Stan if stands, that's", 'participant')
    chatter = [
        turn(52, 'Aud', 'Smile and hold still.', 'audience'),
        turn(53, 'Aud', 'That was fun.', 'audience'),
        turn(54, 'Jay', 'One more picture after this.', 'participant'),
    ]
    # 31 panel + closer + 4 cues + posing + fragment + 3 chatter = 41. Drop one panel turn.
    rows = rows[:30] + [closer, host, audience, applause, books, posing, fragment, *chatter]
    assert len(rows) == 40
    trimmed = apply_session_window({'utterances': rows})
    texts = [row['text'] for row in trimmed['utterances']]
    assert texts[-1] == books['text']
    assert host['text'] in texts
    assert posing['text'] not in texts
    assert fragment['text'] not in texts
    assert closer_turn(rows)['text'] == closer['text']
    assert 'Thank' not in closer_turn(rows)['text']


def test_a_goal_that_already_starts_with_a_verb_is_not_prefixed_with_share():
    from lib.story_render import capitalize_sentences, goal_line

    assert goal_line('prioritize the handoff before the next decision', personal=True) == (
        'You set out to prioritize the handoff before the next decision.'
    )
    assert goal_line('demonstrate the gap for the whole room', personal=True) == (
        'You set out to demonstrate the gap for the whole room.'
    )
    assert 'share prioritize' not in goal_line('prioritize the work for the team', personal=True)
    fixed = goal_line('feel rather than a rigid plan for the room', personal=True)
    assert fixed.startswith('You set out to share a flexible aim rather than')
    assert 'to feel rather than' not in fixed
    assert capitalize_sentences('you advocate for the next voice.').startswith('You advocate')


def test_goal_keeps_a_grounded_phrase_and_trims_the_fallback_filler():
    from lib.data_story import personal_story
    from lib.story_render import goal_line

    assert goal_line('trying to keep the workflow open for the group', personal=True) == (
        'You set out to keep the workflow open for the group.'
    )
    assert goal_line('wanted to show the room one clear next step', personal=True) == (
        'You set out to show the room one clear next step.'
    )
    kept = goal_line(
        'Oh, I just wanted to add, I think the knowledge systems should stay visible to the team.',
        personal=True,
    )
    assert kept.startswith('You set out to ')
    assert 'wanted to add' not in kept.lower()
    assert 'knowledge systems' in kept
    transcript = {
        'utterances': [
            {
                'speaker': 'A',
                'start': 0,
                'end': 8000,
                'text': (
                    'Oh, I just wanted to add, I think the knowledge systems still hide the design '
                    'until the workflow is named.'
                ),
            },
        ],
    }
    story = personal_story(
        label='A',
        info={'name': 'Eden'},
        analysis={},
        transcript=transcript,
        stats_frame=None,
        framework={},
        draft={},
    )
    assert 'the point you brought into the room' not in story['goal']
    assert 'wanted to add' not in story['goal'].lower()
    assert 'knowledge systems' in story['goal']


def test_goal_fallback_accepts_a_short_verb_and_skips_a_self_introduction():
    from lib.data_story import personal_story
    from lib.story_render import goal_line

    assert goal_line('help the group see the gap', personal=True) == (
        'You set out to help the group see the gap.'
    )
    assert goal_line('show the room one clear next step', personal=True) == (
        'You set out to show the room one clear next step.'
    )
    transcript = {
        'utterances': [
            {
                'speaker': 'A', 'start': 0, 'end': 4000,
                'text': (
                    'Annie and Corey Cannon, Noco International brought the opening remarks '
                    'for the whole session today.'
                ),
            },
            {
                'speaker': 'A', 'start': 8000, 'end': 14000,
                'text': (
                    'The knowledge systems still hide the design until the workflow is named '
                    'by the people who use it.'
                ),
            },
        ],
    }
    story = personal_story(
        label='A',
        info={'name': 'Lynae'},
        analysis={},
        transcript=transcript,
        stats_frame=None,
        framework={},
        draft={},
    )
    assert 'No turn of yours is long enough' not in story['goal']
    assert 'Noco International' not in story['goal']
    assert story['goal'].startswith('You set out to share ')
    assert 'knowledge systems' in story['goal']


def test_fallback_skips_repetition_and_an_introduction():
    from lib.data_story import personal_story

    transcript = {
        'utterances': [
            {'speaker': 'A', 'text': 'Hello, my name is Lynae.', 'start': 0, 'end': 2000},
            {'speaker': 'A', 'text': 'Repetition.', 'start': 2000, 'end': 3000},
            {
                'speaker': 'A',
                'text': 'The knowledge systems still hide the design until the workflow is named by the room.',
                'start': 3000,
                'end': 8000,
            },
        ]
    }
    story = personal_story(
        label='A',
        info={'name': 'Lynae'},
        analysis={},
        transcript=transcript,
        stats_frame=None,
        framework={},
        draft={},
    )
    assert 'Repetition.' not in story['drama']
    assert 'my name is' not in story['drama'].lower()
    assert 'knowledge systems' in story['drama']
    assert 'You came in to move the work' not in story['goal']
    assert 'knowledge systems' in story['goal']


def test_plain_goal_prose_is_wrapped_and_empty_number_keys_use_facts(caplog):
    import json
    import logging

    from lib.story_generation import _fill_sections

    class ProseGoal:
        def invoke(self, prompt):
            return type('Reply', (), {'content': json.dumps({
                'goal': 'identify where value breaks between data and knowledge',
            })})()

    draft = {'accepted': []}
    _fill_sections(
        ProseGoal(),
        audience='group',
        sections=('goal',),
        evidence='FACTS',
        render=lambda section, data: {'goal': data['claim']},
        progress_path=None,
        draft=draft,
    )
    assert draft['goal'] == 'identify where value breaks between data and knowledge'
    assert 'goal' not in draft.get('fallback_sections', [])
    with caplog.at_level(logging.WARNING):
        rendered = render_personal_section(
            'character',
            {'headline': 'You opened on the onboarding plan', 'number_keys': []},
            transcript=synthetic_transcript(),
            label='A',
            facts=FACTS,
            forbidden=[],
        )
    assert rendered['numbers']
    assert 'number_keys raw' in caplog.text
    mapped = render_personal_section(
        'character',
        {'headline': 'You opened on the onboarding plan', 'number_keys': ['talk time']},
        transcript=synthetic_transcript(),
        label='A',
        facts=FACTS,
        forbidden=[],
    )
    assert 'talk time' in mapped['numbers'][0]


def test_a_turn_that_ends_with_thanks_is_still_the_closer():
    from lib.session_window import apply_session_window, closer_turn

    def turn(minute, speaker, text, role='participant'):
        start = minute * 60 * 1000
        return {
            'speaker': speaker, 'role': role, 'text': text,
            'start': start, 'end': start + 20_000,
        }

    rows = [
        turn(index, 'A' if index % 2 == 0 else 'B',
             'The knowledge systems still hide the design until the workflow is named by the room.',
             'moderator' if index % 2 == 0 else 'participant')
        for index in range(20)
    ]
    closing = turn(
        21, 'A',
        'The knowledge systems still hide the design until the workflow is named. Thank you all for coming.',
        'moderator',
    )
    venue = turn(
        28, 'B',
        'The knowledge systems closed about 9… they were gone after the hour.',
        'participant',
    )
    rows.extend([closing, venue])
    chosen = closer_turn(rows)
    assert chosen['text'].endswith('named.')
    assert 'Thank you all for coming.' not in chosen['text']
    trimmed = apply_session_window({'utterances': rows})
    assert 'closed about 9' not in ' '.join(row['text'] for row in trimmed['utterances'])
    assert trimmed['utterances'][-1]['text'].endswith('Thank you all for coming.')


def test_group_closer_uses_the_strongest_sentence_not_the_last_fragment():
    from lib.story_render import render_group_section, strongest_sentence

    turn = (
        'The knowledge systems still hide the design until the workflow is named by the people who use it. '
        'Thanks for the fragment.'
    )
    assert strongest_sentence(turn).startswith('The knowledge systems still hide the design')
    assert 'fragment' not in strongest_sentence(turn)
    rendered = render_group_section(
        'resolution',
        {'claim': 'the room named the workflow'},
        transcript={'utterances': [{
            'speaker': 'A', 'role': 'host', 'name': 'Annie Green',
            'text': turn + ' Thank you all for coming.',
            'start': 0, 'end': 8000,
        }]},
        facts={'interaction': {'value': 23, 'line': '23% interaction'}},
        cast='',
    )
    assert 'knowledge systems still hide the design' in rendered['resolution']
    assert 'fragment' not in rendered['resolution'].lower()
    with pytest.raises(OutputParserException, match='opening segment'):
        render_group_section(
            'resolution',
            {'claim': 'the opening segment closed the hour'},
            transcript=synthetic_transcript(),
            facts={'interaction': {'value': 23, 'line': '23% interaction'}, '_names': ['Annie Green']},
            cast='',
        )


def test_speaker_e_fragment_cannot_replace_the_hosts_thanks_closer():
    from lib.session_window import closer_turn

    def turn(minute, speaker, text, role='participant', name=None):
        start = minute * 60 * 1000
        row = {
            'speaker': speaker, 'role': role, 'text': text,
            'start': start, 'end': start + 20_000,
        }
        if name:
            row['name'] = name
        return row

    rows = [
        turn(
            index, 'A',
            'The knowledge systems still hide the design until the workflow is named by the room.',
            'moderator', 'Annie Green',
        )
        for index in range(8)
    ]
    host = turn(
        12, 'A',
        'The knowledge systems still hide the design until the workflow is named. Thank you all for coming.',
        'facilitator', 'Annie Green',
    )
    fragment = (
        'this era, like as users, as participants, et cetera the knowledge systems '
        'still hide the design until the workflow ends today.'
    )
    assert len(fragment.split()) == 21
    speaker_e = turn(16, 'E', fragment, 'participant', 'Speaker E')
    rows.extend([host, speaker_e])
    chosen = closer_turn(rows)
    assert 'Thank you all for coming' not in chosen['text']
    assert chosen['text'].startswith('The knowledge systems still hide the design')
    assert 'et cetera' not in chosen['text']


def test_goal_and_resolution_cannot_cite_the_same_turn():
    opening = 'We should try the new onboarding plan.'
    facts = {**FACTS, '_quote_log': []}
    goal = render_personal_section(
        'goal',
        {'claim': 'identify the onboarding plan', 'quote': {'text': opening, 'ts': '0:00'}},
        transcript=synthetic_transcript(),
        label='A',
        facts=facts,
        forbidden=[],
    )
    resolution = render_personal_section(
        'resolution',
        {'claim': 'Monday became the stake', 'quote': {'text': opening, 'ts': '0:00'}},
        transcript=synthetic_transcript(),
        label='A',
        facts=facts,
        forbidden=[],
    )
    assert '0:00' in goal['goal']
    assert '0:11' in resolution['resolution']
    assert 'Monday' in resolution['resolution']
    assert goal['goal'].count('0:00') == 1
    assert '0:00' not in resolution['resolution']


def test_resolution_uses_the_host_closer_and_a_what_happened_clause():
    from lib.story_generation import _coerce_payload
    from lib.story_render import render_group_section

    wrapped = _coerce_payload('The room named the workflow before it closed.', 'resolution')
    assert wrapped == {'claim': 'The room named the workflow before it closed.'}

    def turn(minute, speaker, text, role='participant', name=None):
        start = minute * 60 * 1000
        row = {
            'speaker': speaker, 'role': role, 'text': text,
            'start': start, 'end': start + 20_000,
        }
        if name:
            row['name'] = name
        return row

    rows = [
        turn(
            index, 'A',
            'The knowledge systems still hide the design until the workflow is named by the room.',
            'moderator', 'Annie Green',
        )
        for index in range(8)
    ]
    rows.append(turn(
        12, 'A',
        'The knowledge systems still hide the design until the workflow is named. Thank you all for coming.',
        'facilitator', 'Annie Green',
    ))
    rows.append(turn(
        16, 'E',
        'this era, like as users, as participants, et cetera the knowledge systems still hide the design until the workflow ends today.',
        'participant', 'Speaker E',
    ))
    rendered = render_group_section(
        'resolution',
        {'claim': 'the room named the workflow'},
        transcript={'utterances': rows},
        facts={'interaction': {'value': 23, 'line': '23% interaction'}},
        cast='',
    )
    assert 'Speaker E' not in rendered['resolution']
    assert 'et cetera' not in rendered['resolution']
    assert 'Thank you all for coming' not in rendered['resolution']
    assert 'The meeting closed on Annie Green' in rendered['resolution']
    assert 'knowledge systems still hide the design' in rendered['resolution']

    personal = render_personal_section(
        'resolution',
        {
            'claim': 'Recognize your specific talents',
            'quote': {'text': 'We should try the new onboarding plan.', 'ts': '0:00'},
        },
        transcript=synthetic_transcript(),
        label='A',
        facts=FACTS,
        forbidden=[],
    )
    assert personal['resolution'].startswith('What happened is that you recognize your specific talents.')


def test_dr_annie_green_matches_annie_green_and_seth_does_not():
    from lib.story_render import render_group_section
    from lib.story_validation import reject_craft_violations

    assert 'Annie' in reject_craft_violations(
        'Annie carried the point for the room today.',
        confirmed_names=['Dr. Annie Green'],
    )
    assert 'Green' in reject_craft_violations(
        'Green carried the point for the room today.',
        confirmed_names=['Dr. Annie Green'],
    )

    facts = {
        'interaction': {'value': 20, 'line': '20% interaction', 'unit': 'percent'},
        '_names': ['Dr. Annie Green'],
    }
    rendered = render_group_section(
        'resolution',
        {'claim': 'Annie Green closed the hour on the workflow'},
        transcript=synthetic_transcript(),
        facts=facts,
        cast='',
    )
    assert 'Annie Green' in rendered['resolution']
    with pytest.raises(OutputParserException, match='Seth'):
        render_group_section(
            'resolution',
            {'claim': 'Seth closed the hour on the budget'},
            transcript=synthetic_transcript(),
            facts=facts,
            cast='',
        )


def test_resolution_rejects_an_unconfirmed_name_and_a_session_begins_line():
    from lib.story_render import render_group_section

    facts = {
        'interaction': {'value': 20, 'line': '20% interaction', 'unit': 'percent'},
        '_names': ['Annie Green'],
    }
    with pytest.raises(OutputParserException, match='Seth'):
        render_group_section(
            'resolution',
            {'claim': 'Seth closed the hour on the budget'},
            transcript=synthetic_transcript(),
            facts=facts,
            cast='',
        )
    with pytest.raises(OutputParserException, match='The session begins with'):
        render_group_section(
            'resolution',
            {'claim': 'The session begins with the budget talk'},
            transcript=synthetic_transcript(),
            facts=facts,
            cast='',
        )


def test_jrick_headline_cannot_use_someone_elses_dating_coach_line():
    transcript = {
        'utterances': [
            {
                'speaker': 'J',
                'text': 'The knowledge systems still hide the design until the workflow is named.',
                'start': 0,
                'end': 4000,
            },
            {
                'speaker': 'B',
                'text': 'I work as a research engineer and dating coach on the side now.',
                'start': 4000,
                'end': 8000,
            },
        ]
    }
    kept = render_personal_section(
        'character',
        {'headline': 'the knowledge systems hide the design', 'number_keys': ['talk_share']},
        transcript=transcript,
        label='J',
        facts=FACTS,
        forbidden=[],
    )
    assert kept['headline'][0].isupper()
    assert len(kept['headline'].split()) <= 14
    with pytest.raises(OutputParserException, match='research engineer'):
        render_personal_section(
            'character',
            {'headline': 'A research engineer and dating coach', 'number_keys': ['talk_share']},
            transcript=transcript,
            label='J',
            facts=FACTS,
            forbidden=[],
        )


def test_common_noun_headline_stays_when_the_speaker_says_it():
    transcript = {
        'utterances': [
            {
                'speaker': 'E',
                'text': 'Knowledge management is how I keep the work findable for the next person.',
                'start': 0,
                'end': 4000,
            },
            {
                'speaker': 'S',
                'text': 'Human connection is the part I keep coming back to in this room.',
                'start': 4000,
                'end': 8000,
            },
            {
                'speaker': 'B',
                'text': 'I also care about knowledge management and human connection in the product.',
                'start': 8000,
                'end': 12000,
            },
        ]
    }
    eden = render_personal_section(
        'character',
        {'headline': 'Knowledge management is the point you set down', 'number_keys': ['talk_share']},
        transcript=transcript,
        label='E',
        facts=FACTS,
        forbidden=[],
    )
    stephanie = render_personal_section(
        'character',
        {'headline': 'Human connection is the point you set down', 'number_keys': ['talk_share']},
        transcript=transcript,
        label='S',
        facts=FACTS,
        forbidden=[],
    )
    assert eden['headline'].startswith('Knowledge management')
    assert stephanie['headline'].startswith('Human connection')
    assert 'You came in and set a point down' not in eden['headline']
    assert 'You came in and set a point down' not in stephanie['headline']


def test_goal_prefixes_and_gerunds_become_a_base_verb():
    from lib.story_render import goal_line

    assert goal_line('to identify where value breaks') == (
        'The room set out to identify where value breaks.'
    )
    assert goal_line('I want to identify where value breaks', personal=True) == (
        'You set out to identify where value breaks.'
    )
    assert goal_line('we need to maintain personal autonomy') == (
        'The room set out to maintain personal autonomy.'
    )
    assert goal_line('the goal is to identify where value breaks') == (
        'The room set out to identify where value breaks.'
    )
    assert goal_line('identifying where value breaks') == (
        'The room set out to identify where value breaks.'
    )
    assert goal_line('you maintaining personal autonomy', personal=True) == (
        'You set out to maintain personal autonomy.'
    )


def test_drama_counter_moment_must_be_a_different_turn_and_cost_keeps_fact_digits():
    opening = 'We should try the new onboarding plan.'
    same_turn = {
        'utterances': [
            {
                'speaker': 'A',
                'text': 'We should try the new onboarding plan. That might work if we start Monday.',
                'start': 0,
                'end': 8000,
            },
        ]
    }
    with pytest.raises(OutputParserException, match='different turn'):
        render_personal_section(
            'drama',
            {
                'patterns': [{
                    'quotes': [{'text': 'We should try the new onboarding plan', 'ts': '0:00'}],
                    'pattern': 'You set the plan in the first line',
                    'twist': 'the next voice took the turn',
                    'cost': 'a delay of 3 years',
                    'gift': 'a clear start',
                }],
                'exception': {
                    'quote': 'That might work if we start Monday',
                    'ts': '0:00',
                    'what_happened': 'you named Monday',
                },
            },
            transcript=same_turn,
            label='A',
            facts=FACTS,
            forbidden=[],
        )
    kept = render_personal_section(
        'drama',
        {
            'patterns': [{
                'quotes': [{'text': opening, 'ts': '0:00'}],
                'pattern': 'You set the plan in the first line',
                'twist': 'the next voice took the turn',
                'cost': 'a delay of 3 years',
                'gift': 'a clear start',
            }],
            'exception': {
                'quote': 'That might work if we start Monday.',
                'ts': '0:11',
                'what_happened': 'you named Monday',
            },
        },
        transcript=synthetic_transcript(),
        label='A',
        facts={**FACTS, 'questions': {'value': 3, 'line': '3 questions.', 'unit': 'questions'}},
        forbidden=[],
    )
    assert '3 years' in kept['drama']
    stripped = render_personal_section(
        'drama',
        {
            'patterns': [{
                'quotes': [{'text': opening, 'ts': '0:00'}],
                'pattern': 'You set the plan in the first line',
                'twist': 'the next voice took the turn',
                'cost': 'a delay of 9 years',
                'gift': 'a clear start',
            }],
            'exception': {
                'quote': 'That might work if we start Monday.',
                'ts': '0:11',
                'what_happened': 'you named Monday',
            },
        },
        transcript=synthetic_transcript(),
        label='A',
        facts=FACTS,
        forbidden=[],
    )
    assert '9' not in stripped['drama']


def test_drama_keeps_own_turn_digits_and_a_distinct_counter_moment():
    transcript = {
        'utterances': [
            {
                'speaker': 'A',
                'text': 'We should try the new onboarding plan for her team.',
                'start': 0,
                'end': 4000,
            },
            {
                'speaker': 'A',
                'text': 'That might work if we start Monday with 12 people.',
                'start': 11000,
                'end': 15000,
            },
            {
                'speaker': 'A',
                'text': 'The gift of the hour was 8 quiet minutes after the plan.',
                'start': 20000,
                'end': 24000,
            },
        ]
    }
    rendered = render_personal_section(
        'drama',
        {
            'patterns': [{
                'quotes': [{'text': 'We should try the new onboarding plan for her team.', 'ts': '0:00'}],
                'pattern': 'You set the plan in the first line',
                'twist': 'the next voice took her turn after 12 minutes',
                'cost': 'a delay of 3 years',
                'gift': 'a clear start in 8 minutes',
            }],
            'exception': {
                'quote': 'We should try the new onboarding plan for her team.',
                'ts': '0:00',
                'what_happened': 'you named the plan',
            },
        },
        transcript=transcript,
        label='A',
        facts=FACTS,
        forbidden=[],
    )
    drama = rendered['drama']
    assert '12' in drama
    assert '8' in drama
    assert 'her' in drama
    assert 'Monday' in drama
    assert drama.count('onboarding plan for her team') == 1


def test_short_drama_uses_the_next_turn_instead_of_stubbing():
    stephen = {
        'utterances': [
            {
                'speaker': 'A',
                'text': 'We should try the new onboarding plan before the hour ends.',
                'start': 0,
                'end': 4000,
            },
            {
                'speaker': 'A',
                'text': 'the budget can wait until Monday',
                'start': 8000,
                'end': 12000,
            },
        ]
    }
    rendered = render_personal_section(
        'drama',
        {
            'patterns': [{
                'quotes': [{'text': 'We should try the new onboarding plan before the hour ends.', 'ts': '0:00'}],
                'pattern': 'You set the plan in the first line',
                'twist': 'the next voice took the turn',
                'cost': 'a question',
                'gift': 'a clear start',
            }],
            'exception': {
                'quote': 'We should try the new onboarding plan before the hour ends.',
                'ts': '0:00',
                'what_happened': 'you named the plan',
            },
        },
        transcript=stephen,
        label='A',
        facts=FACTS,
        forbidden=[],
    )
    assert 'budget can wait' in rendered['drama']
    assert rendered['drama'].count('onboarding plan') == 1

    lynae = {
        'utterances': [
            {'speaker': 'L', 'text': 'I keep the thread', 'start': 0, 'end': 2000},
            {'speaker': 'L', 'text': 'Name the owner', 'start': 4000, 'end': 6000},
        ]
    }
    short = render_personal_section(
        'drama',
        {
            'patterns': [{
                'quotes': [{'text': 'I keep the thread', 'ts': '0:00'}],
                'pattern': 'You set the plan in the first line',
                'twist': 'the next voice took the turn',
                'cost': 'a question',
                'gift': 'a clear start',
            }],
            'exception': {
                'quote': 'I keep the thread',
                'ts': '0:00',
                'what_happened': 'you named the thread',
            },
        },
        transcript=lynae,
        label='L',
        facts=FACTS,
        forbidden=[],
    )
    assert 'I keep the thread' in short['drama']
    assert 'Name the owner' in short['drama']


def test_second_person_check_ignores_words_inside_quotes():
    from lib.story_validation import reject_craft_violations

    kept = reject_craft_violations(
        'You said "they left the room after the hour." and you stayed with the plan.',
        speaker_label='A',
    )
    assert 'they left' in kept
    with pytest.raises(OutputParserException, match="'they'"):
        reject_craft_violations(
            'You said they left the room after the hour and you stayed.',
            speaker_label='A',
        )


def test_group_pair_table_keeps_four_and_signature_stays_with_its_line(tmp_path):
    from PIL import Image
    from pypdf import PdfReader

    from lib.pdf_functions import MM_PDF
    from lib.story_pdf import write_group_story, write_personal_report

    image = tmp_path / 'chord.png'
    Image.new('RGB', (400, 400), 'white').save(image)
    pdf = MM_PDF(path_report_assets=__import__('pathlib').Path('report_assets'))
    pairs = [(f'Person {index}', f'Other {index}', index, 1) for index in range(6)]
    story = {
        'headline': 'The room kept handing the turn across',
        'character': 'Interaction moved between the voices in the room.',
        'goal': 'The room set out to identify the plan.',
        'challenge': 'A point could finish inside one voice.',
        'drama': 'Who talks to whom, and who nobody answered.',
        'resolution': 'The meeting closed on the budget.',
        'one_thing': 'Name one reply before the next decision. Target: 56% interaction.',
        'why': 'Interaction is 42%, so 56% is a step up.',
        'transitions': {},
    }
    write_group_story(pdf, story, image, pairs)
    pdf.output(tmp_path / 'group.pdf')
    text = '\n'.join(page.extract_text() or '' for page in PdfReader(tmp_path / 'group.pdf').pages)
    assert 'Person 0' in text
    assert 'Person 3' in text
    assert 'Person 4' not in text
    assert 'Person 5' not in text

    personal = MM_PDF(path_report_assets=__import__('pathlib').Path('report_assets'))
    long_drama = ' '.join(['You set the point and the next voice took it up before the hour moved on.'] * 40)
    write_personal_report(
        personal,
        {
            'headline': 'You came in and set a point down',
            'numbers': ['8% talk time: below the room average.'],
            'character_body': 'You set one point down.',
            'share': 8,
            'goal': 'You set one point down.',
            'challenge': 'You set one point down.',
            'drama': long_drama,
            'resolution': 'You set one point down.',
            'one_thing': 'Ask one question before you add another point. Target: 4 questions.',
            'why': 'You asked 1 question, against 4 questions for the session.',
            'transitions': {},
            'signature': {
                'phrase': 'we should', 'count': 2, 'quotes': [],
                'what': 'You reach for it when you want the room to move with you today.',
                'so_what': '', 'now_what': '',
            },
            'lines': [],
            'appendix': {},
        },
        tmp_path / 'charts',
    )
    personal.output(tmp_path / 'personal.pdf')
    for page in PdfReader(tmp_path / 'personal.pdf').pages:
        page_text = page.extract_text() or ''
        if 'Your signature phrase' in page_text:
            assert 'You reach for it' in page_text
            break
    else:
        raise AssertionError('signature heading missing')
    opening = 'We should try the new onboarding plan.'
    reused = _render(
        'drama',
        {
            'patterns': [{
                'quotes': [{'text': opening, 'ts': '0:00'}],
                'pattern': 'You set the plan in the first line',
                'twist': 'the next voice took the turn',
                'cost': 'a question',
                'gift': 'a clear start',
            }],
            'exception': {'quote': opening, 'ts': '0:00', 'what_happened': 'you named the plan again'},
        },
    )
    assert 'Monday' in reused['drama']
    assert reused['drama'].count(opening.rstrip('.')) == 1
