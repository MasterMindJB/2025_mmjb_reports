from lib.naming import assign_display_names, suggest_names
from lib.story_validation import reject_craft_violations


def test_suggestions_come_from_introductions_and_address():
    transcript = {
        'utterances': [
            {'speaker': 'A', 'text': "Hi, I'm Alex Rivera.", 'start': 0, 'end': 1000},
            {'speaker': 'B', 'text': 'Thanks, Blake.', 'start': 1000, 'end': 2000},
            {'speaker': 'C', 'text': 'Hello.', 'start': 2000, 'end': 3000},
        ]
    }
    found = suggest_names(transcript)
    assert found['A'] == 'Alex Rivera'
    assert found['C'] == 'Blake'


def test_unnamed_speakers_stay_neutral_and_only_named_emails_get_a_report():
    assigned = assign_display_names(
        [
            {'label': 'A', 'name': 'Alex Rivera', 'email': 'alex@example.com', 'role': 'participant', 'name_status': 'confirmed'},
            {'label': 'B', 'name': '', 'email': '', 'role': 'participant', 'name_status': 'skipped'},
            {'label': 'C', 'name': '', 'email': '', 'role': 'host', 'name_status': 'skipped'},
            {'label': 'D', 'name': '', 'email': '', 'role': 'audience', 'name_status': 'skipped'},
            {'label': 'E', 'name': 'Hum', 'email': '', 'role': 'noise', 'name_status': 'ignored'},
        ]
    )
    by_label = {item['label']: item for item in assigned}
    assert by_label['B']['display_name'] == 'Speaker B'
    assert by_label['C']['display_name'] == 'Host'
    assert by_label['D']['display_name'] == 'Audience member 1'
    assert by_label['E']['include'] is False
    assert by_label['A']['receives_report'] is True
    assert by_label['B']['receives_report'] is False
    assert 'Alex' not in by_label['B']['display_name']


def test_placeholder_labels_are_valid_participant_names():
    text = 'Speaker C handed the turn to Host.'
    assert (
        reject_craft_violations(
            text,
            participant_names=['Speaker C', 'Host', 'Audience member 1'],
        )
        == text
    )
