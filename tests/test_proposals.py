"""Proposal counts, question counts, and the group bars."""

import json

from lib.proposals import (
    classify_proposals,
    counts_from_indices,
    in_window_turns,
    parse_proposal_indices,
)
from lib.questions import question_counts, questions_in_text
from lib.report_stats import NICE_STATS_COL_NAME
from tests.synthetic import synthetic_transcript


def test_question_detector_counts_marks_and_a_missing_mark():
    assert questions_in_text('Could we check the budget first?') == 1
    assert questions_in_text('Could we check the budget first') == 1
    assert questions_in_text('We should try the new onboarding plan.') == 0
    assert questions_in_text('Where does this break? Who owns the next step?') == 2
    transcript = synthetic_transcript()
    transcript['utterances'].append({
        'speaker': 'E',
        'name': 'Speaker E',
        'text': 'What if we wait until Monday',
        'start': 20000,
        'end': 24000,
    })
    counts = dict(question_counts(transcript, {'A': 'Alex', 'B': 'Blake', 'C': 'Casey', 'E': 'Speaker E'}))
    assert counts['Blake'] == 1
    assert counts['Alex'] == 0
    assert counts['Speaker E'] == 1
    assert counts['Casey'] == 0


def test_proposal_counts_use_the_model_indices_and_skip_a_bad_reply(tmp_path):
    turns = in_window_turns(synthetic_transcript())
    assert [turn['speaker'] for turn in turns] == ['A', 'B', 'C', 'A', 'B']
    assert parse_proposal_indices({'turns': [0, 2, 2]}, len(turns)) == [0, 2]
    assert parse_proposal_indices({'turns': [9]}, len(turns)) is None
    assert parse_proposal_indices('nope', len(turns)) is None
    counts = dict(counts_from_indices(turns, [0, 2]))
    assert counts == {'A': 1, 'B': 0, 'C': 1}

    class Stub:
        def invoke(self, prompt):
            assert 'PROPOSAL_TURNS' in prompt
            assert '0. A:' in prompt
            return type('Reply', (), {'content': json.dumps({'turns': [0, 2]})})()

    counted = classify_proposals(Stub(), synthetic_transcript(), tmp_path)
    assert dict(counted)['A'] == 1
    saved = json.loads((tmp_path / 'checkpoints.json').read_text(encoding='utf-8'))
    assert saved[-1]['turns'] == [0, 2]

    class Broken:
        def invoke(self, prompt):
            return type('Reply', (), {'content': 'not json'})()

    assert classify_proposals(Broken(), synthetic_transcript(), tmp_path) is None


def test_idea_generation_label_is_not_a_chart_column():
    assert 'Idea Generation' not in NICE_STATS_COL_NAME.values()


def test_ideas_chart_is_bars_only(tmp_path):
    from PIL import Image
    from pypdf import PdfReader

    from lib.pdf_functions import MM_PDF
    from lib.story_pdf import write_group_story

    image = tmp_path / 'chord.png'
    Image.new('RGB', (80, 80), 'white').save(image)
    pdf = MM_PDF(path_report_assets=__import__('pathlib').Path('report_assets'))
    story = {
        'headline': 'A few people did most of the talking.',
        'character': 'Alex and Blake did most of the talking, and Speaker E spoke the least. The last line still had to land. Next time, who will ask Speaker E to speak sooner?',
        'goal': 'The room set out to name the plan.',
        'pattern_why': 'When Alex and Blake do most of the talking, Speaker E cannot change the choice. Next time, ask Speaker E before you decide.',
        'challenge': 'A point can end before Speaker E speaks.',
        'caveat': 'A point can end before Speaker E speaks. Next time, stop and ask Speaker E.',
        'drama': 'Who talks to whom: Alex and Blake lead with 3.',
        'resolution': 'The meeting closed on Blake, who said "I can own the numbers from here."',
        'one_thing': 'Ask Speaker E to speak before the next choice.',
        'why': 'Next time, leave one reply for Speaker E.',
        'transitions': {},
        'shares': [('Ric', 21), ('Leslie', 6), ('Speaker E', 4)],
        'question_counts': [('Ric', 4), ('Leslie', 1), ('Speaker E', 0)],
        'proposal_counts': [('Ric', 7), ('Leslie', 3), ('Speaker E', 0)],
    }
    write_group_story(pdf, story, image, [('Ric', 'Leslie', 2, 1)])
    pdf.output(tmp_path / 'group.pdf')
    text = '\n'.join(page.extract_text() or '' for page in PdfReader(tmp_path / 'group.pdf').pages)
    assert 'Who talks to whom: Alex and Blake lead with 3' not in text
    assert 'I can own the numbers' not in text
    assert 'suggested 5 ideas' not in text
    assert 'more than anyone' not in text
    assert (tmp_path / 'ideas.png').is_file()
    assert (tmp_path / 'who-asked.png').is_file()
    assert (tmp_path / 'who-talked.png').is_file()


def test_a_cut_off_quote_grows_to_the_sentence_and_not_the_next_one():
    from lib.story_validation import complete_quote, quote_word_count

    source = 'I got into the business last year. Then I left the firm.'
    shown = complete_quote('I got into the', source)
    assert shown == 'I got into the business last year.'
    assert 'Then I left' not in shown
    assert complete_quote('I got into the', 'I got into the') is None
    fused = 'At 0:38 you said "I got into the business. Next time, who else should speak?"'
    from lib.story_render import SHORTHAND, narrative_problems

    assert 'spliced quote' in narrative_problems(fused)
    assert 'open quote' in narrative_problems('At 0:38 you said "I got into the')
    assert 'the room handed the turn' in SHORTHAND
    assert 'shorthand' in narrative_problems('The interaction rate was the whole story.')
    from lib.readability import reading_grade

    hard = 'The room set out to identify where value breaks within knowledge management systems.'
    assert reading_grade(hard) >= 6
    assert reading_grade('You made one clear move. Next time, try one small step.') < 6
    assert quote_word_count(shown) >= 3
