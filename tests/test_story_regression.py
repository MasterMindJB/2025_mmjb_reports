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
    assert 'Monday' in resolved['resolution']
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

    with pytest.raises(OutputParserException):
        render_group_section(
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
    assert '. You told' in rendered['resolution']
    assert 'as stated by' not in rendered['resolution'].lower()
    with pytest.raises(OutputParserException):
        render_group_section(
            'goal',
            {'claim': "Annie's theme for the whole hour"},
            transcript=synthetic_transcript(),
            facts=facts,
            cast='',
        )


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
    assert rendered['goal'].startswith('The turns keep returning to identify')
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


def test_stock_pipeline_puts_the_facilitator_role_on_the_speaker_map():
    from app.analysis import apply_speaker_roles

    speaker_map = {'A': {'name': 'Annie'}, 'B': {'name': 'Corey'}}
    apply_speaker_roles(speaker_map, {'A': 'facilitator', 'B': 'participant'}, {'B'})
    assert speaker_map['A']['role'] == 'facilitator'
    assert speaker_map['B']['role'] == 'participant'
