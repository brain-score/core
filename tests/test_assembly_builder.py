"""Tests for shared Brain-Score assembly construction helpers."""

import numpy as np

from brainscore_core.assembly_builder import (
    concat_neuroid_assemblies,
    make_assembly,
    make_layer_neuroid_coords,
    make_neuroid_ids,
    replace_presentation_coord,
)
from brainscore_core.supported_data_standards.brainio.assemblies import (
    NeuroidAssembly,
)


def test_make_layer_neuroid_coords_preserves_requested_order():
    coords = make_layer_neuroid_coords(
        'model-a', 'layer.1', 2,
        include=('neuroid_id', 'neuroid_num', 'model', 'layer'),
    )

    assert list(coords) == ['neuroid_id', 'neuroid_num', 'model', 'layer']
    assert coords['neuroid_id'] == (
        'neuroid', ['model-a.layer.1.0', 'model-a.layer.1.1'])
    assert coords['layer'] == ('neuroid', ['layer.1', 'layer.1'])


def test_concat_neuroid_assemblies_matches_manual_wrapper_shape():
    left = make_assembly(
        np.ones((2, 2)),
        coords={
            'stimulus_id': ('presentation', ['s0', 's1']),
            **make_layer_neuroid_coords('m', 'l1', 2),
        },
        dims=['presentation', 'neuroid'],
    )
    right = make_assembly(
        np.zeros((2, 1)),
        coords={
            'stimulus_id': ('presentation', ['s0', 's1']),
            **make_layer_neuroid_coords('m', 'l2', 1),
        },
        dims=['presentation', 'neuroid'],
    )

    merged = concat_neuroid_assemblies([left, right], strategy='manual')

    assert isinstance(merged, NeuroidAssembly)
    assert merged.dims == ('presentation', 'neuroid')
    assert merged.shape == (2, 3)
    assert list(merged['presentation'].values) == ['s0', 's1']
    assert list(merged['neuroid_id'].values) == [
        'm.l1.0', 'm.l1.1', 'm.l2.0',
    ]


def test_replace_presentation_coord_rebuilds_assembly():
    assembly = make_assembly(
        np.ones((2, 1)),
        coords={
            'stimulus_id': ('presentation', ['old0', 'old1']),
            'video_path': ('presentation', ['a.mp4', 'b.mp4']),
            **make_layer_neuroid_coords('m', 'l', 1),
        },
        dims=['presentation', 'neuroid'],
    )

    replaced = replace_presentation_coord(
        assembly, 'stimulus_id', ['new0', 'new1'])

    assert list(replaced['stimulus_id'].values) == ['new0', 'new1']
    assert list(replaced['video_path'].values) == ['a.mp4', 'b.mp4']
    assert list(replaced['neuroid_id'].values) == make_neuroid_ids('m', 'l', 1)
