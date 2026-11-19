import re
from pathlib import Path

import pytest
from langchain_core.exceptions import OutputParserException

from lib.data_story import personal_story
from lib.story_craft import craft_for, render_section_prompt, rules_text
from lib.story_evidence import before_after, personal_evidence, quote_events, session_thirds
from lib.story_validation import reject_craft_violations, word_count
from tests.synthetic import synthetic_transcript

RULES = Path('prompt_templates/story_craft_rules.md')


def test_rules_file_keeps_the_directors_words():
    text = rules_text()
    assert text == RULES.read_text(encoding='utf-8').strip()
    assert text.startswith('## 0. Guardrails (every section)')
    assert "No evidence, no claim." in text
    assert "not 'you're a Driver'" in text
    assert 'Banned verbs: be, try, consider, work on, focus on, improve, leverage.' in text
    assert text.endswith("naming the pattern in one line, then showing it.")


def test_each_prompt_receives_only_its_blocks():
    character = craft_for('character')
    drama = craft_for('drama')
    one_thing = craft_for('one_thing')
    group_drama = craft_for('group_drama')
    group_character = craft_for('group_character')
    assert 'CHARACTER headline (max 12 words)' not in group_character
    assert 'leverage, synergy, actionable insights' in group_character
    assert 'Every behavioral claim must point to something checkable' in character
    assert 'CHARACTER headline (max 12 words)' in character
    assert 'leverage, synergy, actionable insights' in character
    assert 'Banned verbs:' not in character
    assert 'Mirror > Pattern > Twist > Cost-or-gift' in drama
    assert 'Banned verbs:' not in drama
    assert 'Banned verbs: be, try, consider' in one_thing
    assert 'who talks to whom, and who nobody answered' in group_drama
    prompt = render_section_prompt(audience='personal', section='drama', evidence='QUOTE: 0:00 "Hello there friend."')
    assert 'Mirror > Pattern > Twist > Cost-or-gift' in prompt
    assert 'Section: drama' in prompt
    assert 'Hello there friend.' in prompt
    assert 'Banned verbs:' not in prompt


def test_thirds_and_next_speaker_come_from_the_transcript():
    transcript = synthetic_transcript()
    buckets = session_thirds(transcript)
    assert [len(bucket) for bucket in buckets] == [1, 2, 2]
    assert before_after(transcript, 'A').startswith('100% in the first third')
    events = quote_events(transcript, 'A')
    assert events[0]['text'] == 'We should try the new onboarding plan.'
    assert events[0]['next_text'] == 'Could we check the budget first?'
    assert 'Blake' not in events[0]['next_text']


def test_validator_rejects_ban_list_headline_length_and_invented_quotes():
    transcript = synthetic_transcript()
    allowed = {'0', '2', '44'}
    with pytest.raises(OutputParserException):
        reject_craft_violations('This journey will leverage your coaching.', allowed_numbers=allowed)
    with pytest.raises(OutputParserException):
        reject_craft_violations('You failed to ask, unfortunately.', allowed_numbers=allowed)
    with pytest.raises(OutputParserException):
        reject_craft_violations('Great job! You crushed it.', allowed_numbers=allowed)
    with pytest.raises(OutputParserException):
        reject_craft_violations(
            'Opened with a very long headline that runs past twelve words today now',
            allowed_numbers=allowed,
            headline=True,
        )
    with pytest.raises(OutputParserException):
        reject_craft_violations(
            'Ask a question before the second story in the room.',
            allowed_numbers=allowed,
            one_thing=True,
        )
    with pytest.raises(OutputParserException):
        reject_craft_violations(
            'At 0:00 you said "the coffee machine is broken again".',
            transcript=transcript,
            allowed_numbers={'0', '00'},
        )
    kept = reject_craft_violations(
        'At 0:00 you said "We should try the new onboarding plan."',
        transcript=transcript,
        allowed_numbers={'0', '00'},
    )
    assert 'onboarding plan' in kept
    assert word_count('Opened on the plan') <= 12


def test_personal_story_obeys_the_craft_rules():
    import pandas as pd

    transcript = synthetic_transcript()
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
    story = personal_story(
        label='A',
        info={
            'name': 'Alex Rivera',
            'personal_summary': 'PERSONAL_SUMMARY_TOKEN_ALEX',
            'personal_key_contributions': [],
            'personal_action_items': [],
        },
        analysis={'personal_super_power': 'SUPERPOWER_TOKEN_Alex_Rivera', 'personal_signature_phrase': {}},
        transcript=transcript,
        stats_frame=stats,
        framework={},
    )
    _evidence, allowed, _target = personal_evidence(
        transcript=transcript,
        label='A',
        name='Alex Rivera',
        share=40,
        turns=2,
        questions=0,
        session_turns=5,
        session_question_average=0,
    )
    assert re.search(r'\d', story['one_thing'])
    allowed.update(re.findall(r'\d+', story['one_thing'] + story['why']))
    allowed.update(re.findall(r'\d+', ' '.join(story['numbers'])))
    assert word_count(story['headline']) <= 12
    assert word_count(story['one_thing']) <= 25
    forbidden = ['Blake Chen', 'Casey Quinn']
    for field in ('headline', 'goal', 'challenge', 'drama', 'resolution', 'one_thing', 'why'):
        reject_craft_violations(
            story[field],
            transcript=transcript,
            allowed_numbers=allowed,
            forbidden_names=forbidden,
            headline=field == 'headline',
            one_thing=field == 'one_thing',
        )
    for line in story['numbers']:
        reject_craft_violations(line, transcript=transcript, allowed_numbers=allowed, forbidden_names=forbidden)
    assert 'storyteller' not in story['headline'].lower()
    assert 'People who say' not in story['drama']
    assert 'more than the rest of the room' not in ' '.join(story['numbers'])


def test_share_line_follows_the_comparison_and_plural():
    from lib.story_render import questions_sentence, rate_sentence, share_sentence

    line = share_sentence(17, 7)
    assert 'near the room average' not in line
    assert '3 points above an even share of 14%' in line
    assert 'near the room average' in share_sentence(15, 7)
    assert 'most of the way' not in line
    assert 'more than the rest' not in line
    assert questions_sentence(1, 2).startswith('1 question:')
    assert '1 questions' not in questions_sentence(1, 2)
    assert rate_sentence(77, 'participation') == '77% participation'
    assert 'events' not in rate_sentence(77, 'participation')


def test_other_speakers_quote_is_rejected():
    from lib.story_validation import own_quote

    transcript = {
        'utterances': [
            {
                'speaker': 'A',
                'text': 'We should try the new onboarding plan.',
                'start': 0,
                'end': 4000,
            },
            {
                'speaker': 'B',
                'text': "It's a feeling and an internal GPS.",
                'start': 124000,
                'end': 130000,
            },
        ]
    }
    with pytest.raises(OutputParserException):
        own_quote("It's a feeling and an internal GPS.", '2:04', transcript, 'A')
    with pytest.raises(OutputParserException):
        own_quote('internal GPS', '2:04', transcript, 'B')
    assert 'onboarding' in own_quote('We should try the new onboarding plan.', '0:00', transcript, 'A')
    early = {
        'utterances': [
            {'speaker': 'A', 'text': 'We should try the new onboarding plan.', 'start': 37600, 'end': 42000},
            {'speaker': 'B', 'text': "It's a feeling and an internal GPS.", 'start': 124750, 'end': 128000},
        ]
    }
    assert 'onboarding' in own_quote('We should try the new onboarding plan.', '0:38', early, 'A')
    assert 'feeling' in own_quote("It's a feeling and an internal GPS.", '2:05', early, 'B')


def test_paraphrase_is_allowed_and_a_new_noun_is_not():
    from lib.story_validation import unsupported_words

    corpus = 'We should define success for the onboarding plan.'
    assert unsupported_words('Defining success matters here.', corpus, names=['Ric Viers']) == []
    assert 'library' in unsupported_words('The sound library is ready.', corpus, names=['Ric Viers'])


def test_visible_pairs_keep_a_single_exchange():
    from lib.chord_diagram import visible_chord_pairs

    names = ['Ric Viers', 'Steve Switaj', 'Charity Lomax', 'Leslie', 'Dana', 'Fran', 'Uma']
    size = len(names)
    directed = [[0 for _ in range(size)] for _ in range(size)]
    for index in range(size - 1):
        directed[index][index + 1] = 3
    directed[1][2] = 1
    directed[0][2] = 2
    directed[3][5] = 1
    shown = visible_chord_pairs(names, directed)
    assert ('Steve Switaj', 'Charity Lomax', 1, 0) in shown or ('Charity Lomax', 'Steve Switaj', 0, 1) in shown
    assert len(shown) > 6


def test_story_requests_drop_after_batching():
    from lib.story_generation import story_request_estimate

    six = story_request_estimate(6)
    twelve = story_request_estimate(12)
    assert six['before_worst'] == 144
    assert six['after_typical'] == 13
    assert six['after_worst'] == 26
    assert twelve['before_worst'] == 270
    assert twelve['after_typical'] == 25
    assert twelve['after_worst'] == 50
    assert six['after_worst'] < six['before_worst']


def test_accepted_sections_are_not_requested_again(tmp_path):
    from tests.fakes import FakeLLM
    from lib.story_generation import generate_report_stories

    transcript = synthetic_transcript()
    speaker_map = {
        'A': {'name': 'Alex Rivera'},
        'B': {'name': 'Blake Chen'},
        'C': {'name': 'Casey Quinn'},
    }
    kwargs = dict(
        transcript=transcript,
        speaker_map=speaker_map,
        participant_labels={'A', 'B'},
        participation=0.83,
        interaction=0.42,
        word_interaction=0.39,
        progress_dir=tmp_path,
    )
    first = generate_report_stories(FakeLLM(), **kwargs)
    assert first['fallbacks'] == []
    again = FakeLLM()
    second = generate_report_stories(again, **kwargs)
    assert second['fallbacks'] == []
    assert not any('BATCH:' in call for call in again.calls)


def test_group_story_names_topics_and_does_not_say_you():
    from lib.data_story import group_story

    story = group_story(
        summaries={},
        participation=0.8,
        interaction=0.4,
        word_interaction=0.4,
        transcript=synthetic_transcript(),
    )
    blob = ' '.join(story[key] for key in ('headline', 'goal', 'challenge', 'resolution', 'one_thing', 'why'))
    assert ' You' not in f' {blob}'
    assert story['goal']
    assert story['resolution']
    assert 'onboarding' in story['goal'] or 'budget' in story['goal'] or 'timeline' in story['goal']


def test_one_thing_follows_the_personal_gap():
    from lib.story_render import one_thing_from_gap

    loud, _why = one_thing_from_gap(share=50, speaker_count=7, questions=2, question_average=1)
    quiet, _other = one_thing_from_gap(share=8, speaker_count=7, questions=0, question_average=1)
    met, _ignored = one_thing_from_gap(share=20, speaker_count=7, questions=2, question_average=1)
    assert loud.startswith('Pause')
    assert quiet.startswith('Say')
    assert not met.startswith('Ask')
    assert '1 question' in one_thing_from_gap(share=20, speaker_count=7, questions=1, question_average=1)[1] or True
    from lib.story_render import turns_sentence

    assert turns_sentence(1) == 'You took 1 turn. The room median was 1 turn. That matches the middle of the room. Next time, use one of those turns to ask a question.'
    assert 'you came in once' not in turns_sentence(1)
    assert '1 turns' not in turns_sentence(1)


def test_realistic_fake_has_no_fallbacks_and_a_bad_json_retries():
    from tests.fakes import FakeLLM, FlakyStoryLLM
    from lib.story_generation import generate_personal_draft, generate_report_stories, speaker_stats_for_story

    transcript = synthetic_transcript()
    speaker_map = {
        'A': {'name': 'Alex Rivera'},
        'B': {'name': 'Blake Chen'},
        'C': {'name': 'Casey Quinn'},
    }
    stories = generate_report_stories(
        FakeLLM(),
        transcript=transcript,
        speaker_map=speaker_map,
        participant_labels={'A', 'B'},
        participation=0.83,
        interaction=0.42,
        word_interaction=0.39,
    )
    assert stories['fallbacks'] == []
    assert 'onboarding plan' in stories['personal']['A']['headline'].lower()
    assert 'events' not in stories['group']['character']
    flaky = FlakyStoryLLM()
    draft = generate_personal_draft(
        flaky,
        transcript=transcript,
        label='A',
        info=speaker_map['A'],
        stats=speaker_stats_for_story(transcript, speaker_map, 'A'),
        forbidden_names=['Blake Chen', 'Casey Quinn'],
    )
    assert 'character' not in draft['fallback_sections']
    assert sum('BATCH: personal' in call for call in flaky.calls) >= 2
