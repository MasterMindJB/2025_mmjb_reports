from pathlib import Path

from lib.chord_diagram import (
    CHORD_DEFINITION,
    chord_caption,
    directed_follows,
    pair_directions,
    save_chord_diagram,
    strongest_pair_indexes,
    transition_ribbons,
)


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
    assert 'who spoke first' in CHORD_DEFINITION
    assert 'turn order' in CHORD_DEFINITION


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


def test_caption_names_the_strongest_pairs_and_the_least_connected_person():
    names = ['Alex Rivera', 'Blake Chen', 'Dana Okonkwo', 'Casey Quinn', 'Fran Ibarra']
    directed = [[0 for _ in names] for _ in names]
    alex, blake, dana, casey, fran = range(5)
    directed[alex][blake] = 6
    directed[blake][alex] = 3
    directed[dana][casey] = 4
    directed[casey][dana] = 2
    directed[alex][dana] = 1
    assert (
        chord_caption(names, directed)
        == 'Most back-and-forth: Alex ↔ Blake (9), Dana ↔ Casey (6). Least connected: Fran.'
    )
    pairs = pair_directions(names, directed)
    assert pairs[0][:4] == ('Alex Rivera', 'Blake Chen', 6, 3)
    assert pairs[1][:4] == ('Dana Okonkwo', 'Casey Quinn', 4, 2)
    assert strongest_pair_indexes(directed) == {(alex, blake), (dana, casey), (alex, dana)}
    assert fran == 4


def test_caption_when_nobody_follows_a_different_person():
    names, directed = directed_follows(['Ada', 'Ada'], names=['Ada', 'Bea'])
    assert chord_caption(names, directed) == 'No one spoke immediately after anyone else.'
    assert pair_directions(names, directed) == []


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
