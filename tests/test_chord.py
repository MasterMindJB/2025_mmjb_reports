from pathlib import Path

from lib.chord_diagram import CHORD_DEFINITION, save_chord_diagram, transition_ribbons


def test_ribbon_width_counts_both_directions_and_ignores_self_follows():
    names, ribbon, arcs = transition_ribbons(
        ['Alex', 'Alex', 'Blake', 'Alex', 'Casey', 'Blake'],
        names=['Alex', 'Blake', 'Casey', 'Dana'],
    )
    # Alex->Blake, Blake->Alex, Alex->Casey, Casey->Blake. Dana never speaks.
    alex, blake, casey, dana = range(4)
    assert ribbon[alex][blake] == 2
    assert ribbon[blake][alex] == 2
    assert ribbon[alex][casey] == 1
    assert ribbon[casey][blake] == 1
    assert ribbon[alex][alex] == 0
    assert arcs == [3, 3, 2, 0]
    assert names[dana] == 'Dana'
    assert 'immediately followed' in CHORD_DEFINITION
    assert 'either direction' in CHORD_DEFINITION


def test_chord_diagram_handles_two_speakers_and_a_speaker_with_no_transitions(tmp_path: Path):
    path = tmp_path / 'two.png'
    names, _ribbon, arcs = save_chord_diagram(
        ['Ada', 'Bea'],
        path,
        names=['Ada', 'Bea', 'No Turns'],
        roles={'Bea': 'facilitator', 'No Turns': 'actor'},
    )
    assert names == ['Ada', 'Bea', 'No Turns']
    assert arcs[2] == 0
    assert path.is_file()
    assert path.stat().st_size > 5_000


def test_chord_diagram_handles_twelve_speakers(tmp_path: Path):
    names = [f'Person {index}' for index in range(12)]
    sequence = []
    for index, name in enumerate(names):
        sequence.extend([name, names[(index + 1) % 12]])
    path = tmp_path / 'twelve.png'
    save_chord_diagram(sequence, path, names=names, roles={names[0]: 'facilitator', names[1]: 'actor'})
    assert path.is_file()
    assert path.stat().st_size > 5_000


def six_person_session() -> tuple[list[str], list[str], dict[str, str]]:
    names = ['Alex Rivera', 'Blake Chen', 'Casey Quinn', 'Dana Okonkwo', 'Eden Brooks', 'Fran Ibarra']
    sequence = [
        'Casey Quinn',
        'Alex Rivera',
        'Blake Chen',
        'Alex Rivera',
        'Dana Okonkwo',
        'Eden Brooks',
        'Fran Ibarra',
        'Eden Brooks',
        'Alex Rivera',
        'Fran Ibarra',
        'Blake Chen',
        'Dana Okonkwo',
        'Casey Quinn',
        'Eden Brooks',
        'Blake Chen',
        'Alex Rivera',
    ]
    roles = {'Casey Quinn': 'facilitator', 'Fran Ibarra': 'actor'}
    return sequence, names, roles
