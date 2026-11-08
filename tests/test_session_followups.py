"""Follow-ups to the per-channel session change.

1. ``concat_neuroid_assemblies(strategy='manual')`` joins rows by presentation
   identity, never by position.
2. A task (readout or generation) refuses a region whose selected layers did
   not all contribute neuroids, with the rule the neural session path uses.
3. A model takes part in one session at a time.
"""
from contextlib import contextmanager
from dataclasses import replace
import inspect
import sys
import threading

import numpy as np
import pytest

from brainscore_core import BrainScoreModel, CompositeSelector, LayerSelector, TaskContext
from brainscore_core.assembly_builder import concat_neuroid_assemblies
from brainscore_core.streaming import BufferedSession
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly, walk_coords
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
from brainscore_core.responses import Choice
from tests.test_joint_behavior_neural import GENERATE, STIMULI, ToyExtractor, ToyNet, _assembly, _run
from tests.test_session_channels_keep_native_output import assert_same_output

pytestmark = pytest.mark.unit


# --- 1. manual concat joins rows by presentation identity ---------------------

def clock_block(ids, values, layer, *, duplicate=False):
    """One neuroid, two time bins; per-presentation bounds follow the stimulus."""
    coords = {'stimulus_id': ('presentation', ids),
              'trial': ('presentation', list(range(len(ids)))) if duplicate else ('presentation', ids),
              'neuroid_id': ('neuroid', [layer]), 'layer': ('neuroid', [layer])}
    a = NeuroidAssembly(np.repeat(np.asarray(values, dtype='float32')[:, None, None], 2, axis=1),
                        dims=['presentation', 'time_bin', 'neuroid'], coords=coords)
    starts = [[0., 10.], [20., 30.]] if duplicate else [[0., 10.] if i == 'a' else [20., 30.] for i in ids]
    return a.assign_coords(time_bin_start_ms=(('presentation', 'time_bin'), starts),
                           time_bin_end_ms=(('presentation', 'time_bin'), np.asarray(starts) + 10))


def still_block(ids, values, layer, **extra):
    """A 2-D wrapper output: presentations by neuroids, several presentation coords."""
    values = np.asarray(values, dtype='float32').reshape(len(ids), -1)
    n = values.shape[1]
    return NeuroidAssembly(values, dims=['presentation', 'neuroid'], coords={
        'stimulus_id': ('presentation', list(ids)),
        'image_path': ('presentation', [f'{i}.png' for i in ids]),
        **{name: ('presentation', list(v)) for name, v in extra.items()},
        'neuroid_id': ('neuroid', [f'm.{layer}.{u}' for u in range(n)]),
        'neuroid_num': ('neuroid', list(range(n))),
        'layer': ('neuroid', [layer] * n)})


def positional_concat(assemblies):
    """The manual join before this change, kept as the reference for equal-order inputs."""
    first = assemblies[0]
    merged = np.concatenate([a.values for a in assemblies], axis=first.dims.index('neuroid'))
    nonneuroid = {c: (d, v) for c, d, v in walk_coords(first) if set(d) != {'neuroid'}}
    neuroid = {c: [d, v] for c, d, v in walk_coords(first) if set(d) == {'neuroid'}}
    for assembly in assemblies[1:]:
        for coord in neuroid:
            neuroid[coord][1] = np.concatenate((neuroid[coord][1], assembly[coord].values))
    return type(first)(merged, coords={**nonneuroid, **{c: (d, v) for c, (d, v) in neuroid.items()}},
                       dims=first.dims)


@pytest.mark.parametrize('strategy', ['manual', 'xarray'])
def test_same_physical_clock_requires_actual_value_alignment(strategy):
    """Codex v6 probe: rows [a, b] and [b, a] with matching clocks pair by stimulus."""
    a = clock_block(['a', 'b'], [1, 2], 'early')
    b = clock_block(['b', 'a'], [20, 10], 'late')
    try:
        merged = concat_neuroid_assemblies([a, b], strategy=strategy)
    except ValueError as e:
        assert 'presentation' in str(e).lower() or 'order' in str(e).lower()
        return
    np.testing.assert_array_equal(merged.values[:, 0, :], [[1, 10], [2, 20]])


def test_manual_concat_puts_reordered_2d_rows_in_the_first_blocks_order():
    first = still_block(['a', 'b', 'c'], [[1, 2], [3, 4], [5, 6]], 'l1')
    second = still_block(['c', 'a', 'b'], [[50], [10], [30]], 'l2')
    merged = concat_neuroid_assemblies([first, second], strategy='manual', validate_non_neuroid=True)
    np.testing.assert_array_equal(merged.values, [[1, 2, 10], [3, 4, 30], [5, 6, 50]])
    assert list(merged['stimulus_id'].values) == ['a', 'b', 'c']
    assert list(merged['layer'].values) == ['l1', 'l1', 'l2']
    expected = positional_concat([first, second.isel(presentation=[1, 2, 0])])
    assert_same_output(merged, expected)


def test_manual_concat_aligns_repeated_ids_by_the_complete_index():
    first = still_block(['s', 's'], [1, 2], 'l1', trial=[0, 1])
    second = still_block(['s', 's'], [20, 10], 'l2', trial=[1, 0])
    merged = concat_neuroid_assemblies([first, second], strategy='manual')
    np.testing.assert_array_equal(merged.values, [[1, 10], [2, 20]])


@pytest.mark.parametrize('case', ['other_stimuli', 'other_index_schema', 'other_length'])
def test_manual_concat_refuses_rows_it_cannot_match(case):
    first = still_block(['s', 's'], [1, 2], 'l1', trial=[0, 1])
    second = {'other_stimuli': lambda: still_block(['t', 'u'], [10, 20], 'l2'),
              'other_index_schema': lambda: still_block(['s', 's'], [10, 20], 'l2', repeat=[0, 1]),
              'other_length': lambda: still_block(['s'], [10], 'l2', trial=[0])}[case]()
    with pytest.raises(ValueError, match='same presentations'):
        concat_neuroid_assemblies([first, second], strategy='manual')


EQUAL_ORDER = {
    'still': lambda: [still_block(['a', 'b'], [[1, 2], [3, 4]], 'l1'),
                      still_block(['a', 'b'], [[5], [6]], 'l2'),
                      still_block(['a', 'b'], [[7, 8, 9], [0, 1, 2]], 'l3')],
    'still-repeated-ids': lambda: [still_block(['s', 's'], [1, 2], 'l1', trial=[0, 1]),
                                   still_block(['s', 's'], [3, 4], 'l2', trial=[0, 1])],
    'clock': lambda: [clock_block(['a', 'b'], [1, 2], 'early'), clock_block(['a', 'b'], [3, 4], 'late')],
    'clock-repeated-ids': lambda: [clock_block(['s', 's'], [1, 2], 'early', duplicate=True),
                                   clock_block(['s', 's'], [3, 4], 'late', duplicate=True)],
}


@pytest.mark.parametrize('validate', [False, True])
@pytest.mark.parametrize('case', sorted(EQUAL_ORDER))
def test_manual_concat_of_equal_order_inputs_is_unchanged(case, validate):
    """Registered wrappers pass blocks in one order; their output stays byte-identical."""
    blocks = EQUAL_ORDER[case]()
    merged = concat_neuroid_assemblies(blocks, strategy='manual', validate_non_neuroid=validate)
    expected = positional_concat(blocks)
    assert_same_output(merged, expected)
    assert merged.values.tobytes() == expected.values.tobytes()
    assert list(merged.coords) == list(expected.coords)


