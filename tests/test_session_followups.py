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


# --- 2. task paths refuse a missing selected layer ----------------------------

class Incomplete(ToyExtractor):
    """Drops 'early' from every output, as a malformed extractor would."""
    def __call__(self, stimuli, layers, **kwargs):
        return super().__call__(stimuli, [l for l in layers if l != 'early'], **kwargs)

    @contextmanager
    def record(self, layers):
        with super().record([l for l in layers if l != 'early']) as rec:
            yield rec


def task_model(net, extractor_type, regions):
    def answer(stimulus_row, instruction, label_set):
        net.forward(stimulus_row['stimulus_id'], prompt=instruction)
        return label_set[0]
    return BrainScoreModel('task-layers', extractor={'text': extractor_type(net)}, region_layer_map=regions,
                           behavioral_readout_layer='late', answer=answer)


READOUT = TaskContext(task_type='probabilities', label_set=['cat', 'dog'], fitting_stimuli=STIMULI,
                      prefer_path='readout')
TASKS = {'readout': READOUT, 'generation': GENERATE}
MIX = {'mix': CompositeSelector((('early', None), ('late', None)), modality='text')}


@pytest.mark.parametrize('mode', ['readout', 'generation'])
def test_task_composite_refuses_missing_selected_layer(mode):
    """Codex v6 probe: readout and generation refuse it, like the neural session path."""
    m = task_model(ToyNet(), Incomplete, MIX)
    with pytest.raises(ValueError, match='layer|neuroid|activity'):
        _run(m, TASKS[mode], ['neural:mix'])


def test_neural_session_refuses_the_same_missing_layer():
    m = task_model(ToyNet(), Incomplete, MIX)
    with pytest.raises(ValueError, match='every selected layer|no neuroids'):
        _run(m, None, ['neural:mix'])


def test_readout_refuses_a_plain_region_whose_layer_is_missing():
    m = task_model(ToyNet(), Incomplete, {'E': LayerSelector('early', modality='text')})
    with pytest.raises(ValueError, match='every selected layer'):
        _run(m, READOUT, ['behavior', 'neural:E'])


@pytest.mark.parametrize('mode', ['readout', 'generation'])
def test_task_accepts_an_absent_layer_whose_unit_subset_is_empty(mode):
    """A composite layer with no selected units is not required (required_layers)."""
    regions = {'mix': CompositeSelector((('early', ()), ('late', None)), modality='text')}
    session = _run(task_model(ToyNet(), Incomplete, regions), TASKS[mode], ['behavior', 'neural:mix'])
    mix = session.collect('neural:mix')
    assert set(mix['layer'].values) == {'late'}
    complete = _run(task_model(ToyNet(), ToyExtractor, regions), TASKS[mode], ['behavior', 'neural:mix'])
    np.testing.assert_array_equal(mix.values, complete.collect('neural:mix').values)


@pytest.mark.parametrize('mode', ['readout', 'generation'])
def test_task_refuses_a_composite_that_selects_no_units(mode):
    regions = {'mix': CompositeSelector((('early', ()), ('late', ())), modality='text')}
    with pytest.raises(ValueError, match='no neuroids'):
        _run(task_model(ToyNet(), ToyExtractor, regions), TASKS[mode], ['behavior', 'neural:mix'])


@pytest.mark.parametrize('mode', ['readout', 'generation'])
def test_complete_composite_keeps_its_pass_count_and_values(mode):
    behavior_net, joint_net = ToyNet(), ToyNet()
    _run(task_model(behavior_net, ToyExtractor, MIX), TASKS[mode], ['behavior'])
    joint = _run(task_model(joint_net, ToyExtractor, MIX), TASKS[mode], ['behavior', 'neural:mix'])
    assert joint_net.passes == behavior_net.passes          # recording adds no forward pass
    if mode == 'generation':
        assert joint_net.passes == len(STIMULI)
    mix = joint.collect('neural:mix')
    assert list(mix['layer'].values) == ['early'] * 3 + ['late'] * 2


def conditional_extractor(pattern):
    """Records 'early' on no trial, every trial, or trials a and c only."""
    class Conditional(ToyExtractor):
        @contextmanager
        def record(self, layers):
            captured, count = [], [0]
            recording = type('Recording', (), {})()

            def take():
                row = captured[0]
                captured.clear()
                count[0] += 1
                keep_early = pattern == 'always' or (pattern == 'sometimes' and count[0] % 2)
                return _assembly([row], ['?'], [l for l in layers if l != 'early' or keep_early])
            recording.take = take
            self.net.hooks.append(captured.append)
            try:
                yield recording
            finally:
                self.net.hooks.remove(captured.append)
    return Conditional


GENERATION_FORMS = {'answer': GENERATE,
                    'choice': replace(GENERATE, task_type=None, label_set=None, response=Choice(('cat', 'dog')))}


@pytest.mark.parametrize('form', sorted(GENERATION_FORMS))
@pytest.mark.parametrize('pattern', ['never', 'always', 'sometimes'])
def test_generation_ignores_an_optional_layer_on_every_trial(pattern, form):
    """An empty unit subset stays optional, whatever the other trials record."""
    regions = {'mix': CompositeSelector((('early', ()), ('late', None)), modality='text')}
    net = ToyNet()
    joint = _run(task_model(net, conditional_extractor(pattern), regions), GENERATION_FORMS[form],
                 ['behavior', 'neural:mix'])
    expected = _run(task_model(ToyNet(), ToyExtractor, regions), GENERATION_FORMS[form],
                    ['behavior', 'neural:mix'])
    assert_same_output(joint.collect('neural:mix'), expected.collect('neural:mix'))
    assert net.passes == len(STIMULI)


@pytest.mark.parametrize('form', sorted(GENERATION_FORMS))
def test_generation_refuses_a_required_layer_missing_on_some_trials(form):
    m = task_model(ToyNet(), conditional_extractor('sometimes'), MIX)
    with pytest.raises(ValueError, match="No activity reached layer 'early' while answering stimulus 'b'"):
        _run(m, GENERATION_FORMS[form], ['behavior', 'neural:mix'])


# --- 3. one session at a time per model ----------------------------------------

def output(stimuli, layers):
    arrays = [np.tile(np.array([sum(map(ord, l)), sum(map(ord, l)) + 1], dtype='float32'),
                      (len(stimuli), 1)) for l in layers]
    return NeuroidAssembly(np.concatenate(arrays, axis=1), dims=['presentation', 'neuroid'], coords={
        'stimulus_id': ('presentation', list(stimuli.stimulus_id)),
        'origin': ('presentation', ['source'] * len(stimuli)),
        'neuroid_id': ('neuroid', [f'{l}.{i}' for l in layers for i in range(2)]),
        'layer': ('neuroid', [l for l in layers for _ in range(2)])})


REGIONS = {r: LayerSelector(l, modality='text') for r, l in
           [('A', 'early'), ('B', 'late'), ('C', 'other'), ('D', 'fourth')]}


def model(extract=output, identifier='one-at-a-time'):
    return BrainScoreModel(identifier, extractor={'text': extract}, region_layer_map=REGIONS)


def run(m, regions):
    s = BufferedSession(stimulus_set=StimulusSet({'stimulus_id': ['a', 'b'], 'sentence': ['x', 'x']}),
                        requested_output_channels=['neural:' + r for r in regions])
    m.participate(s)
    return s


def alone(region):
    return run(model(), [region]).collect('neural:' + region)


def paused_first_session(extract_hook=None):
    """Start session (A, B) on a model in a thread and pause it inside the extractor."""
    entered, release = threading.Event(), threading.Event()
    results, errors = {}, {}

    def extract(stimuli, layers):
        if threading.current_thread().name == 'first-session':
            entered.set()
            assert release.wait(3), 'test synchronization timed out'
        return output(stimuli, layers)
    m = model(extract)

    def worker():
        try:
            results['first'] = run(m, ['A', 'B'])
        except BaseException as e:
            errors['first'] = f'{type(e).__name__}: {e}'
    first = threading.Thread(target=worker, name='first-session')
    first.start()
    assert entered.wait(3), 'test synchronization timed out'
    return m, first, release, results, errors


def test_concurrent_sessions_on_one_model_keep_native_outputs():
    """Codex v6 probe on the per-channel default: the second session is refused
    clearly, and the first keeps exactly its standalone outputs."""
    m, first, release, results, errors = paused_first_session()
    try:
        with pytest.raises(RuntimeError, match='already taking part in a session'):
            run(m, ['C', 'D'])
    finally:
        release.set()
        first.join(3)
    assert not first.is_alive()
    assert not errors, errors
    for r in ('A', 'B'):
        assert_same_output(results['first'].collect('neural:' + r), alone(r))
    # The guard is released: the refused request now runs and is native too.
    second = run(m, ['C', 'D'])
    for r in ('C', 'D'):
        assert_same_output(second.collect('neural:' + r), alone(r))


def test_sessions_on_different_models_may_run_at_the_same_time():
    m, first, release, results, errors = paused_first_session()
    try:
        other = run(model(identifier='other'), ['C'])
    finally:
        release.set()
        first.join(3)
    assert not errors, errors
    assert_same_output(other.collect('neural:C'), alone('C'))


def test_sequential_sessions_on_one_model_still_work():
    m = model()
    outputs = [run(m, ['A', 'B']) for _ in range(3)]
    for session in outputs:
        for r in ('A', 'B'):
            assert_same_output(session.collect('neural:' + r), alone(r))


def test_a_nested_session_on_the_same_model_is_refused():
    controls = {'nest': True}

    def extract(stimuli, layers):
        if controls['nest']:
            run(m, ['C'])
        return output(stimuli, layers)
    m = model(extract)
    with pytest.raises(RuntimeError, match='already taking part in a session'):
        run(m, ['A'])
    controls['nest'] = False                # the guard was released by the failed session
    assert_same_output(run(m, ['A']).collect('neural:A'), alone('A'))


def test_a_refused_nested_call_keeps_the_outer_reservation():
    from brainscore_core import brainscore_model as module
    checks = []

    def extract(stimuli, layers):
        if not checks:
            for _ in range(2):          # a refusal must not release the running session
                with pytest.raises(RuntimeError, match='already taking part in a session'):
                    run(m, ['C'])
                checks.append(id(m) in module._active_sessions)
        return output(stimuli, layers)
    m = model(extract)
    assert_same_output(run(m, ['A']).collect('neural:A'), alone('A'))
    assert checks == [True, True]
    assert id(m) not in module._active_sessions


def test_an_interrupt_just_after_the_reservation_releases_the_model():
    """KeyboardInterrupt between the reservation and the session must not leave it reserved."""
    from brainscore_core import brainscore_model as module
    guard = module._one_session_at_a_time.__wrapped__
    lines, start = inspect.getsourcelines(guard)
    reserve = start + next(i for i, line in enumerate(lines) if '_active_sessions[key] = token' in line)
    m = model()
    reached = []

    def interrupt(frame, event, arg):
        if frame.f_code is guard.__code__ and event == 'line' and frame.f_lineno > reserve:
            reached.append(id(m) in module._active_sessions)
            sys.settrace(None)
            raise KeyboardInterrupt('just after the reservation')
        return interrupt
    try:
        sys.settrace(interrupt)
        with pytest.raises(KeyboardInterrupt, match='just after the reservation'):
            run(m, ['A'])
    finally:
        sys.settrace(None)
    assert reached == [True]                # the interrupt arrived while the model was reserved
    assert id(m) not in module._active_sessions
    assert_same_output(run(m, ['A']).collect('neural:A'), alone('A'))


@pytest.mark.parametrize('error', [RuntimeError, KeyboardInterrupt])
def test_the_model_is_released_after_a_failed_session(error):
    controls = {'fail': True}

    def extract(stimuli, layers):
        if controls['fail']:
            raise error('inside the extractor')
        return output(stimuli, layers)
    m = model(extract)
    with pytest.raises(error, match='inside the extractor'):
        run(m, ['A', 'B'])
    controls['fail'] = False
    session = run(m, ['A', 'B'])
    for r in ('A', 'B'):
        assert_same_output(session.collect('neural:' + r), alone(r))


def test_an_unsupported_session_releases_the_model():
    m = model()
    session = BufferedSession(stimulus_set=StimulusSet({'stimulus_id': ['a'], 'sentence': ['x']}),
                              requested_output_channels=['behavior'])
    with pytest.raises(Exception):
        m.participate(session)                  # no answer part: refused before any compute
    assert_same_output(run(m, ['A']).collect('neural:A'), alone('A'))
