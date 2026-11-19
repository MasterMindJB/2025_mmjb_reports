from lib.helper_functions import gini
from lib.report_stats import compute_session_tables
from tests.synthetic import synthetic_transcript


def test_participation_score_matches_the_known_sample():
    score = 1 - gini([54, 78, 73, 15, 39])
    assert score == pytest_approx(0.753)


def pytest_approx(value):
    class _Approx:
        def __eq__(self, other):
            return abs(other - value) <= 0.001

        def __repr__(self):
            return f'approx {value}'

    return _Approx()


def test_synthetic_speaker_totals():
    speaker_map = {
        'A': {'name': 'Alex Rivera', 'personal_action_items': [{}, {}]},
        'B': {'name': 'Blake Chen', 'personal_action_items': [{}]},
        'C': {'name': 'Casey Quinn', 'personal_action_items': []},
    }
    tables = compute_session_tables(synthetic_transcript(), speaker_map)
    stats = tables.df_speaker_stats
    assert int(stats.loc['Alex Rivera', 'word_count']) == 14
    assert int(stats.loc['Blake Chen', 'word_count']) == 13
    assert int(stats.loc['Casey Quinn', 'word_count']) == 5
    assert int(stats.loc['Blake Chen', 'question_count']) == 1
    assert int(stats.loc['Alex Rivera', 'interaction_count']) == 2
    assert int(stats.loc['Alex Rivera', 'action_items_count']) == 2
    assert 'Alex Rivera' in tables.conversation_script
    assert tables.gini_reverse_interventions > 0
