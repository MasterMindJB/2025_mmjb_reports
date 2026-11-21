from lib.metric_registry import REGISTRY, build_scorecard, classify_questions, load_profiles


def call(seller_longest_ms=200_000):
    """A two-person sales call. S sells, B buys."""
    rows = [
        ('S', 'Thanks for making time. What made you take this call?', 0, 5_000),
        ('B', 'We promoted six engineers this year and two of them are struggling.', 5_400, 60_000),
        ('S', 'Um, so, like, we train leaders, you know, under pressure. ' + 'We do reps. ' * 60, 60_300, 60_300 + seller_longest_ms),
        ('B', 'How much does it cost?', 262_000, 265_000),
        ('S', 'Is next Tuesday open for a demo? Does that make sense?', 264_500, 270_000),
        ('B', 'Yes.', 271_000, 272_000),
    ]
    return {'utterances': [{'speaker': a, 'text': b, 'start': c, 'end': d} for a, b, c, d in rows]}


ROLES = {'S': 'seller', 'B': 'buyer'}


def test_every_profile_names_only_registered_metrics():
    profiles = load_profiles()
    assert {'hot_seat', 'sales_call', 'team_meeting', 'panel', 'family_interview'} <= set(profiles)


def test_questions_split_open_closed_and_check():
    counted = classify_questions('What changed? Is it late? Right? So how did they take it?')
    assert counted == {'open': 2, 'closed': 1, 'check': 1}


def test_sales_call_reports_the_seller_only_and_uses_the_target_band():
    card = build_scorecard(call(), load_profiles()['sales_call'], roles=ROLES, names={'S': 'Sam', 'B': 'Bea'})
    assert list(card['people']) == ['S']
    tiles = {t['key']: t for t in card['people']['S']['tiles']}
    assert tiles['longest_turn']['status'] == 'above'
    assert '2:30' in tiles['longest_turn']['compare']
    assert tiles['open_question_share']['number'] == '50%'
    assert tiles['longest_listen']['number'] == '0:55'
    # A two-person call never gets a 'middle of the room' line.
    assert all(t['status'] != 'room' for t in tiles.values())
    # Coded metrics stay out until someone supplies them.
    assert 'next_step' not in tiles


def test_baseline_beats_target_and_coded_values_show_when_passed():
    card = build_scorecard(
        call(), load_profiles()['sales_call'], roles=ROLES,
        baseline={'S': {'longest_turn': 250_000}}, coded={'S': {'next_step': 1}},
    )
    tiles = {t['key']: t for t in card['people']['S']['tiles']}
    assert tiles['longest_turn']['compare'] == 'Was 4:10 last time'
    assert tiles['next_step']['kind'] == 'coded'


def test_a_short_contribution_gets_a_brief_card():
    transcript = call()
    transcript['utterances'].append({'speaker': 'C', 'text': 'The liability goes to the doctor.', 'start': 273_000, 'end': 276_000})
    card = build_scorecard(transcript, load_profiles()['panel'], roles={'S': 'panelist', 'B': 'panelist', 'C': 'panelist'})
    brief = card['people']['C']
    assert brief['brief'] is True
    assert [t['key'] for t in brief['tiles']] == ['talk_time', 'words', 'turns']
    assert all(t['compare'] == '' for t in brief['tiles'])


def test_swapping_a_metric_is_a_profile_edit():
    profile = dict(load_profiles()['team_meeting'], personal=['talk_time', 'questions', 'hedge_rate', 'pace_wpm'])
    card = build_scorecard(call(), profile, roles={'S': 'participant', 'B': 'participant'})
    assert [t['key'] for t in card['people']['S']['tiles']] == ['talk_time', 'questions', 'hedge_rate', 'pace_wpm']
    assert all(key in REGISTRY for key in profile['personal'])
