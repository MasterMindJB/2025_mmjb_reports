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
    _evidence, allowed, target = personal_evidence(
        transcript=transcript,
        label='A',
        name='Alex Rivera',
        share=40,
        turns=2,
        questions=0,
        session_turns=5,
        session_question_average=0,
    )
    assert str(target) in story['one_thing']
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
