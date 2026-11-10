"""The optional clock descriptor, its validators and the clock compatibility check."""
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

import brainscore_core
from brainscore_core import clock as public_clock
from brainscore_core._data import clock as clk
from brainscore_core._data.assembly_builder import make_assembly


def media_clock(**overrides):
    base = dict(units='ms', origin='media_start', reference_input='vision:clip', anchor='center',
                interval='half_open', accuracy='approximate',
                description='decoded-frame PTS, two frames per tubelet', tolerance_ms=20.0)
    base.update(overrides)
    return {key: value for key, value in base.items() if value is not None}


def scanner_clock(**overrides):
    base = dict(units='ms', origin='run_start', anchor='start', accuracy='nominal',
                description='arange(n) * TR from BIDS RepetitionTime')
    base.update(overrides)
    return base


def assembly(starts, ends, centers=None, attrs=None):
    coords = {'stimulus_id': ('presentation', ['clip-a']),
              'stimulus_path': ('presentation', ['a.mp4']),
              'time_bin_start_ms': ('time_bin', list(starts)),
              'time_bin_end_ms': ('time_bin', list(ends)),
              'neuroid_id': ('neuroid', ['n0', 'n1']), 'layer': ('neuroid', ['l', 'l'])}
    if centers is not None:
        coords['time_bin_center_ms'] = ('time_bin', list(centers))
    result = make_assembly(np.zeros((1, len(starts), 2)), coords,
                           ['presentation', 'time_bin', 'neuroid'])
    if attrs is not None:
        result.attrs[clk.CLOCK_ATTR] = attrs
    return result


# --- public location -------------------------------------------------------

def test_public_module_is_the_implementation():
    assert public_clock is clk
    assert brainscore_core.clock is clk


def test_import_is_framework_free():
    code = ("import sys\nimport brainscore_core.clock\n"
            "assert not {'torch', 'transformers', 'tensorflow', 'jax'} & sys.modules.keys()\n")
    subprocess.run([sys.executable, '-I', '-c', code], check=True)


# --- schema and validator --------------------------------------------------

def test_valid_media_and_scanner_clocks():
    assert clk.validate_clock(media_clock()) == []
    assert clk.validate_clock(scanner_clock()) == []


def test_interval_and_optional_keys_may_be_omitted():
    assert clk.validate_clock(media_clock(interval=None, tolerance_ms=None)) == []


@pytest.mark.parametrize('key', ['units', 'origin', 'anchor', 'accuracy', 'description'])
def test_missing_required_key(key):
    clock = media_clock()
    del clock[key]
    assert f'missing {key}' in clk.validate_clock(clock)


@pytest.mark.parametrize('key,value', [
    ('units', 's'), ('origin', 'wall_clock'), ('anchor', 'end'), ('accuracy', 'great'),
    ('interval', 'closed'),
])
def test_value_outside_vocabulary(key, value):
    problems = clk.validate_clock(media_clock(**{key: value}))
    assert any(problem.startswith(f'{key} {value!r}') for problem in problems)


def test_unknown_key_is_reported():
    assert "unknown key 'orgin'" in clk.validate_clock({**media_clock(), 'orgin': 'x'})


def test_blank_description_is_reported():
    assert any('evidence' in p for p in clk.validate_clock(media_clock(description='  ')))


def test_media_start_needs_reference_input():
    problems = clk.validate_clock(media_clock(reference_input=None))
    assert 'a media_start clock must name its reference_input' in problems
    assert clk.validate_clock(media_clock(origin='clip_start', reference_input=None)) == []


@pytest.mark.parametrize('value', [-1.0, float('nan'), float('inf'), '5', True])
def test_bad_tolerance(value):
    assert 'tolerance_ms must be a finite number >= 0' in clk.validate_clock(
        media_clock(tolerance_ms=value))


def test_coverage_and_missing_intervals():
    good = media_clock(coverage_ms=[[0.0, 879287.0]], missing_ms=[[879287.0, 882080.0]])
    assert clk.validate_clock(good) == []
    assert clk.validate_clock(media_clock(missing_ms=[[10.0, 5.0]]))
    assert clk.validate_clock(media_clock(missing_ms=[[0.0, 10.0], [5.0, 20.0]]))
    assert clk.validate_clock(media_clock(missing_ms=[[0.0]]))
    assert clk.validate_clock(media_clock(coverage_ms='0-10'))


def test_non_mapping_is_reported_not_raised():
    assert clk.validate_clock(None) == ['clock must be a mapping']


def test_builder_returns_valid_dict_and_raises_on_invalid():
    clock = clk.clock_descriptor(origin='clip_start', anchor='start', accuracy='exact',
                                 description='synthetic', missing_ms=[(2900.0, 3000.0)])
    assert clk.validate_clock(clock) == []
    assert clock['interval'] == 'half_open' and clock['missing_ms'] == [[2900.0, 3000.0]]
    with pytest.raises(ValueError, match='reference_input'):
        clk.clock_descriptor(origin='media_start', anchor='start', accuracy='exact',
                             description='synthetic')


# --- compatibility ---------------------------------------------------------

def test_identical_clocks_are_compatible():
    match = clk.check_clock_compatibility(media_clock(), media_clock())
    assert match and match.compatible and match.reasons == ()


def test_evidence_text_does_not_affect_compatibility():
    assert clk.check_clock_compatibility(media_clock(), media_clock(description='other'))


@pytest.mark.parametrize('key,value', [
    ('origin', 'clip_start'), ('anchor', 'start'), ('reference_input', 'audio'),
])
def test_mismatch_is_returned_with_reason(key, value):
    match = clk.check_clock_compatibility(media_clock(), media_clock(**{key: value}))
    assert not match
    assert any(reason.startswith(f'{key} differs') for reason in match.reasons)


def test_mismatched_units_report_the_invalid_clock():
    match = clk.check_clock_compatibility(media_clock(), media_clock(units='s'))
    assert not match
    assert any(r.startswith("second clock: units 's'") for r in match.reasons)


def test_accuracy_beyond_tolerance():
    a, b = media_clock(tolerance_ms=20.0), media_clock(tolerance_ms=40.0)
    assert clk.check_clock_compatibility(a, b, tolerance_ms=60.0)
    match = clk.check_clock_compatibility(a, b, tolerance_ms=50.0)
    assert not match
    assert any(reason.startswith('accuracy beyond tolerance') for reason in match.reasons)


def test_tolerance_needs_a_stated_uncertainty():
    match = clk.check_clock_compatibility(media_clock(), media_clock(tolerance_ms=None),
                                          tolerance_ms=100.0)
    assert not match and 'states no tolerance_ms' in match.reasons[0]
    assert clk.check_clock_compatibility(media_clock(), media_clock(tolerance_ms=None))


def test_strict_raises_only_on_mismatch():
    clk.check_clock_compatibility(media_clock(), media_clock(), strict=True)
    with pytest.raises(ValueError, match='origin differs'):
        clk.check_clock_compatibility(media_clock(), media_clock(origin='clip_start'),
                                      strict=True)


def test_compatibility_is_separate_from_channels():
    assert clk.check_clock_compatibility(scanner_clock(), media_clock()).reasons


# --- time axis -------------------------------------------------------------

def test_valid_center_axis_from_attrs():
    a = assembly([0, 750, 1500], [750, 1500, 2250], [375, 1125, 1875], attrs=media_clock())
    assert clk.validate_time_axis(a, duration_ms=3000.0) == []


def test_time_axis_without_clock():
    a = assembly([0, 750], [750, 1500])
    assert clk.validate_time_axis(a) == ["no clock given and no attrs['temporal_clock']"]


def test_time_axis_reports_invalid_clock():
    a = assembly([0, 750], [750, 1500])
    clock = scanner_clock()
    del clock['anchor']
    assert 'missing anchor' in clk.validate_time_axis(a, clock)


@pytest.mark.parametrize('starts,ends,centers,message', [
    ([0, 750], [750, 750], None, 'ends at or before'),
    ([750, 0], [1500, 750], None, 'not strictly increasing'),
    ([0, 750], [750, 1500], [375, 1500], 'outside its [start, end)'),
    ([0, np.nan], [750, 1500], None, 'non-finite'),
])
def test_bad_interval_axis(starts, ends, centers, message):
    problems = clk.validate_time_axis(assembly(starts, ends, centers), media_clock())
    assert any(message in problem for problem in problems)


def test_axis_past_duration_or_before_origin():
    over = assembly([0, 1500], [1500, 3100])
    assert any('past the duration' in p
               for p in clk.validate_time_axis(over, media_clock(), duration_ms=3000.0))
    early = assembly([-10, 1500], [1500, 3000])
    assert any('before the origin' in p
               for p in clk.validate_time_axis(early, media_clock(), duration_ms=3000.0))
    assert clk.validate_time_axis(over, media_clock()) == []


def test_instant_axis():
    a = assembly([0, 1490, 2980], [0, 1490, 2980])
    assert clk.validate_time_axis(a, scanner_clock(anchor='instant')) == []
    b = assembly([0, 0, 2980], [0, 0, 2980])
    assert clk.validate_time_axis(b, scanner_clock(anchor='instant')) == [
        'timestamps are not strictly increasing']


def test_axis_needs_time_bin_dimension():
    a = make_assembly(np.zeros((1, 2)), {'stimulus_id': ('presentation', ['a']),
                                         'stimulus_path': ('presentation', ['a.png']),
                                         'neuroid_id': ('neuroid', ['n0', 'n1']),
                                         'layer': ('neuroid', ['l', 'l'])},
                      ['presentation', 'neuroid'])
    assert clk.validate_time_axis(a, media_clock()) == ['assembly has no time_bin dimension']


def test_validation_does_not_change_the_assembly():
    a = assembly([0, 750], [750, 1500], attrs=media_clock())
    before = dict(a.attrs)
    clk.validate_time_axis(a, duration_ms=3000.0)
    assert a.attrs == before and list(a['time_bin_start_ms'].values) == [0, 750]


# --- review round: Codex 2026-11-09 ---------------------------------------

def test_empty_time_axis_is_rejected():
    a = assembly([], [])
    assert clk.validate_time_axis(a, media_clock()) == ['empty time axis']


def test_non_increasing_explicit_centers_are_rejected():
    # Bins increase and each center lies in its bin, but the centers do not increase.
    a = assembly([0, 100], [200, 300], centers=[150, 120])
    assert any('not strictly increasing' in p for p in clk.validate_time_axis(a, media_clock()))


def test_derived_centers_follow_validate_stream():
    # Without time_bin_center_ms, centers are (start + end) / 2 and must increase.
    a = assembly([0, 10], [100, 20])
    problems = clk.validate_time_axis(a, media_clock())
    assert any('not strictly increasing' in p for p in problems)


@pytest.mark.parametrize('bad', [np.inf, -np.inf, np.nan])
def test_non_finite_times_are_rejected(bad):
    a = assembly([0, 750], [750, 1500], centers=[375, bad])
    assert clk.validate_time_axis(a, media_clock()) == ['non-finite timestamps']


def test_duration_tolerance_matches_validate_stream():
    assert clk.DURATION_TOLERANCE_MS == 1e-6
    inside = assembly([0, 1500], [1500, 3000 + 5e-7])
    assert clk.validate_time_axis(inside, media_clock(), duration_ms=3000.0) == []
    outside = assembly([0, 1500], [1500, 3000 + 1e-3])
    assert any('past the duration' in p
               for p in clk.validate_time_axis(outside, media_clock(), duration_ms=3000.0))


def test_negative_times_without_duration_are_allowed():
    a = assembly([-200, 0], [0, 200])
    assert clk.validate_time_axis(a, media_clock(origin='trial_onset',
                                                 reference_input=None)) == []


def test_non_numeric_coordinate_is_a_problem_not_an_error():
    a = assembly(['a', 'b'], [750, 1500])
    assert clk.validate_time_axis(a, media_clock()) == ['time_bin_start_ms is not numeric']


def test_multiindex_level_coordinates_are_read():
    a = assembly([0, 750, 1500], [750, 1500, 2250], [375, 1125, 1875], attrs=media_clock())
    index = a.indexes['time_bin']
    assert isinstance(index, pd.MultiIndex)
    assert {'time_bin_start_ms', 'time_bin_end_ms', 'time_bin_center_ms'} <= set(index.names)
    assert clk.validate_time_axis(a, duration_ms=2250.0) == []
    bad = assembly([0, 750, 1500], [750, 1500, 2250], [375, 1125, 2250], attrs=media_clock())
    assert isinstance(bad.indexes['time_bin'], pd.MultiIndex)
    assert clk.validate_time_axis(bad) == ['a center lies outside its [start, end) bin']


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -1.0, '10', None])
def test_invalid_requested_tolerance_is_a_reason_not_an_error(value):
    if value is None:
        assert clk.check_clock_compatibility(media_clock(), media_clock(), tolerance_ms=value)
        return
    match = clk.check_clock_compatibility(media_clock(), media_clock(), tolerance_ms=value)
    assert not match
    assert any('requested tolerance_ms' in reason for reason in match.reasons)


def test_coverage_overlapping_missing_is_reported():
    clock = media_clock(coverage_ms=[[0.0, 1000.0]], missing_ms=[[900.0, 1200.0]])
    assert any('overlaps coverage_ms' in p for p in clk.validate_clock(clock))
    touching = media_clock(coverage_ms=[[0.0, 1000.0]], missing_ms=[[1000.0, 1200.0]])
    assert clk.validate_clock(touching) == []


def test_unknown_accuracy_fails_a_tolerance_check():
    a, b = media_clock(), media_clock(accuracy='unknown')
    assert clk.check_clock_compatibility(a, b)
    match = clk.check_clock_compatibility(a, b, tolerance_ms=100.0)
    assert not match and "accuracy 'unknown'" in match.reasons[0]


def test_reference_sha256_and_media_offset_keys():
    a = media_clock(reference_sha256='aa' * 32, media_offset_ms=1250.0)
    assert clk.validate_clock(a) == []
    assert clk.check_clock_compatibility(a, media_clock(reference_sha256='aa' * 32))
    assert clk.check_clock_compatibility(a, media_clock())  # one side unstated
    assert clk.check_clock_compatibility(a, {**a, 'media_offset_ms': 0.0})  # not compared
    match = clk.check_clock_compatibility(a, media_clock(reference_sha256='bb' * 32))
    assert not match and match.reasons[0].startswith('reference_sha256 differs')
    assert clk.validate_clock(media_clock(reference_sha256=''))
    assert clk.validate_clock(media_clock(media_offset_ms=float('nan')))
    built = clk.clock_descriptor(origin='media_start', reference_input='audio', anchor='center',
                                 accuracy='approximate', description='x',
                                 reference_sha256='cc' * 32, media_offset_ms=0.0)
    assert built['reference_sha256'] == 'cc' * 32 and built['media_offset_ms'] == 0.0


def test_docstrings_state_the_compatibility_meaning_and_migration():
    assert 'not proof that the timing' in clk.check_clock_compatibility.__doc__
    assert 'Migration note' in clk.__doc__ and 'arange(n) * TR' in clk.__doc__
