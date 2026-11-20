import math
from pathlib import Path

import pytest
from matplotlib import font_manager
from PIL import Image

from lib.chord_diagram import (
    CHORD_DEFINITION,
    PARTICIPANT_COLORS,
    _directed_ribbon,
    chord_caption,
    directed_follows,
    font_dir,
    load_poppins,
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
    with Image.open(path) as image:
        dpi = image.info.get('dpi')
        assert dpi is not None and dpi[0] >= 299


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
        == 'Most back-and-forth: Alex and Blake (9), Dana and Casey (6). Least connected: Fran.'
    )
    pairs = pair_directions(names, directed)
    assert pairs[0][:4] == ('Alex Rivera', 'Blake Chen', 6, 3)
    assert pairs[1][:4] == ('Dana Okonkwo', 'Casey Quinn', 4, 2)
    assert strongest_pair_indexes(directed) == {(alex, blake), (dana, casey), (alex, dana)}
    assert fran == 4


def test_prose_and_chord_name_the_same_least_connected_people():
    from lib.chord_diagram import least_connected_names
    from lib.story_render import group_character_copy

    names = ['Annie', 'Lynae', 'Audience member 1', 'Audience member 3']
    directed = [[0 for _ in names] for _ in names]
    # Annie and Audience member 1 exchange often. Lynae and Audience member 3
    # each have one handoff, so they are tied for least connected.
    directed[0][2] = 5
    directed[2][0] = 4
    directed[0][1] = 1
    directed[0][3] = 1
    least = least_connected_names(names, directed)
    assert least == ['Lynae', 'Audience member 3']
    caption = chord_caption(names, directed)
    assert 'Lynae' in caption.split('Least connected:')[1]
    assert 'Audience member 3' in caption.split('Least connected:')[1]
    assert 'Audience member 1' not in caption.split('Least connected:')[1]
    # Talk time would have named Audience member 1. The prose uses the chord's people.
    copy = group_character_copy({
        'interaction': {'value': 40},
        'participation': {'value': 80},
        '_portrait': {'least': ' and '.join(least), 'least_names': least, 'low_share': 4, 'high_share': 40},
    })
    assert 'Lynae and Audience member 3 was the least connected' in copy or 'Lynae and Audience member 3' in copy
    assert 'Audience member 1 was the least connected' not in copy


def test_poppins_loads_from_the_vendored_files():
    expected = {
        'regular': 'Poppins-Regular.ttf',
        'medium': 'Poppins-Medium.ttf',
        'semibold': 'Poppins-SemiBold.ttf',
        'bold': 'Poppins-Bold.ttf',
        'extrabold': 'Poppins-ExtraBold.ttf',
    }
    assert (font_dir() / 'OFL.txt').is_file()
    for weight, filename in expected.items():
        prop = load_poppins(weight)
        assert prop.get_name().startswith('Poppins')
        vendored = (font_dir() / filename).resolve()
        assert Path(prop.get_file()).resolve() == vendored
        found = Path(font_manager.findfont(prop, fallback_to_default=False)).resolve()
        assert found == vendored
    with pytest.raises(ValueError):
        load_poppins('condensed')


def test_palette_stays_in_the_navy_sky_family():
    banned = {'#FFC659', '#FF0000', '#E69F00', '#D55E00'}
    assert banned.isdisjoint({color.upper() for color in PARTICIPANT_COLORS})
    for color in PARTICIPANT_COLORS:
        red = int(color[1:3], 16)
        blue = int(color[5:7], 16)
        assert blue + 8 >= red


def test_ribbons_stay_inside_the_circle_even_for_neighbors():
    radius = 1.0
    cases = [((0.9, 1.25), 1.45), ((0.1, 0.7), math.pi + 0.2), ((-0.4, 1.2), 1.35)]
    for source, target in cases:
        path = _directed_ribbon(source, target, radius)
        assert len(path.vertices) == len(path.codes)
        for x, y in path.vertices:
            assert math.hypot(x, y) <= radius + 1e-6


def _boxes_overlap(left, right, gap: float = 1.0) -> bool:
    return (
        left[0] < right[2] - gap
        and right[0] < left[2] - gap
        and left[1] < right[3] - gap
        and right[1] < left[3] - gap
    )


def test_neighboring_labels_and_the_wordmark_do_not_overlap(tmp_path: Path):
    names = [f'Speaker {letter}' for letter in 'ABCDEFG']
    sequence = []
    for index, name in enumerate(names):
        sequence += [name, names[(index + 1) % len(names)]]
    cases = [(sequence, names, None, 7)]
    ten_sequence, ten_names, ten_roles = ten_person_session()
    cases.append((ten_sequence, ten_names, ten_roles, 10))
    for sequence, names, roles, count in cases:
        boxes: dict = {}
        save_chord_diagram(
            sequence,
            tmp_path / f'{count}.png',
            names=names,
            roles=roles,
            layout_boxes=boxes,
        )
        labels = boxes['labels']
        assert len(labels) == count
        for index, label in enumerate(labels):
            for other in labels[index + 1:]:
                assert not _boxes_overlap(label, other)
        for mark in boxes['wordmark']:
            for legend in boxes['legend'] + boxes['captions']:
                assert not _boxes_overlap(mark, legend)


def test_fourteen_speaker_labels_stay_on_the_canvas(tmp_path: Path):
    names = [
        'Stephanie Barnes',
        'Audience member 2',
        'Audience member 3',
        'Corey',
        'Annie Green',
        'Ric Viers',
        'Leslie',
        'Speaker H',
        'Speaker I',
        'Jordy ipecker',
        'Dana Okonkwo',
        'Fran Ibarra',
        'Host',
        'Speaker N',
    ]
    sequence = []
    for index, name in enumerate(names):
        sequence += [name, names[(index + 1) % len(names)]]
    boxes: dict = {}
    save_chord_diagram(sequence, tmp_path / 'fourteen.png', names=names, layout_boxes=boxes)
    width, height = boxes['canvas']
    assert len(boxes['labels']) == 14
    for index, label in enumerate(boxes['labels']):
        assert label[0] >= -1, names[index]
        assert label[2] <= width + 1, names[index]
        assert label[1] >= -1
        assert label[3] <= height + 1
        for other in boxes['labels'][index + 1:]:
            assert not _boxes_overlap(label, other)


def test_caption_when_nobody_follows_a_different_person():
    names, directed = directed_follows(['Ada', 'Ada'], names=['Ada', 'Bea'])
    assert chord_caption(names, directed) == 'No one spoke immediately after anyone else.'
    assert pair_directions(names, directed) == []


def six_person_session() -> tuple[list[str], list[str], dict[str, str]]:
    """About 120 turns. Casey holds the middle, Alex and Blake lock in, Dana is quiet."""
    names = ['Alex Rivera', 'Blake Chen', 'Casey Quinn', 'Dana Okonkwo', 'Eden Brooks', 'Fran Ibarra']
    alex, blake, casey, dana, eden, fran = names
    sequence: list[str] = []
    for other in (alex, blake, eden, fran, alex, blake, eden, fran, alex, eden):
        sequence += [casey, other, casey]
    sequence += [alex, blake] * 16
    sequence += [blake, alex] * 8
    sequence += [eden, fran] * 7
    sequence += [fran, eden] * 3
    sequence += [casey, blake, alex, casey, fran, eden, casey, alex]
    sequence += [casey, eden, casey, fran, casey, blake]
    sequence += [casey, dana, casey, dana, casey]
    roles = {casey: 'facilitator', fran: 'actor'}
    return sequence, names, roles


def ten_person_session() -> tuple[list[str], list[str], dict[str, str]]:
    """About 200 turns. Maya holds the middle, Liam and Nora lock in, Uma is quiet."""
    names = [
        'Maya Ellis',
        'Liam Ortiz',
        'Nora Patel',
        'Owen Brooks',
        'Priya Shah',
        'Quinn Adler',
        'Rosa Nguyen',
        'Samir Cole',
        'Tess Marino',
        'Uma Walsh',
    ]
    facilitator, left, right, quiet = names[0], names[1], names[2], names[9]
    sequence: list[str] = []
    for other in [name for name in names if name != facilitator] * 2:
        sequence += [facilitator, other]
    sequence += [left, right] * 18
    sequence += [right, left] * 10
    for index in range(3, 8):
        sequence += [names[index], names[index + 1]] * 4
    sequence += [names[8], names[6], names[8], names[4]] * 3
    sequence += [names[4], names[5]] * 7
    sequence += [names[5], names[4]] * 4
    for other in (left, right, names[3], names[4], names[5], names[6], names[7], names[8]):
        sequence += [facilitator, other, facilitator]
    sequence += [facilitator, quiet, facilitator]
    roles = {facilitator: 'facilitator', names[8]: 'actor'}
    return sequence, names, roles
