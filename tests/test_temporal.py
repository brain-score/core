"""Tests for temporal_bin() — the post-extraction aggregation that turns
per-frame activations into (clip, time_bin, neuroid) assemblies.

These tests use synthetic data with known patterns so we can assert
that the binning math is exact. No model dependencies.
"""

import numpy as np
import pytest

from brainscore_core.supported_data_standards.brainio.assemblies import (
    NeuroidAssembly,
)
from brainscore_core.temporal import temporal_bin


def _make_frame_assembly(clip_specs):
    """Build a frame-level NeuroidAssembly from a list of
    (clip_id, [(frame_time_ms, feature_vector), ...]) specs.

    This simulates what a vision preprocessor → PytorchWrapper pipeline
    would produce before temporal binning. Multiple coords per dim match
    real-world extractor outputs (TextWrapper, VLMVisionWrapper) which
    always produce MultiIndex-compatible assemblies.
    """
    stim_ids = []
    clip_ids = []
    frame_times = []
    rows = []
    for clip_id, frames in clip_specs:
        for (t_ms, vec) in frames:
            stim_ids.append(f'{clip_id}_t{int(t_ms)}')
            clip_ids.append(clip_id)
            frame_times.append(float(t_ms))
            rows.append(np.asarray(vec, dtype=np.float32))
    data = np.stack(rows, axis=0)
    n_features = data.shape[1]
    return NeuroidAssembly(
        data,
        coords={
            'stimulus_id': ('presentation', stim_ids),
            'clip_id': ('presentation', clip_ids),
            'frame_time_ms': ('presentation', frame_times),
            # Two neuroid coords force a MultiIndex (matches real wrappers)
            'neuroid_id': ('neuroid', [f'n{j}' for j in range(n_features)]),
            'neuroid_num': ('neuroid', list(range(n_features))),
        },
        dims=['presentation', 'neuroid'],
    )


class TestTemporalBinShape:
    def test_basic_shape(self):
        # 2 clips × 4 frames, feature dim 3
        assy = _make_frame_assembly([
            ('c0', [(0, [1, 0, 0]), (500, [2, 0, 0]),
                    (1000, [3, 0, 0]), (1500, [4, 0, 0])]),
            ('c1', [(0, [0, 1, 0]), (500, [0, 2, 0]),
                    (1000, [0, 3, 0]), (1500, [0, 4, 0])]),
        ])
        time_bins = [(0, 1000), (1000, 2000)]
        result = temporal_bin(assy, time_bins)
        assert result.dims == ('presentation', 'time_bin', 'neuroid')
        assert result.shape == (2, 2, 3)

    def test_coord_structure(self):
        assy = _make_frame_assembly([
            ('c0', [(0, [1, 1]), (500, [2, 2])]),
            ('c1', [(0, [3, 3]), (500, [4, 4])]),
        ])
        result = temporal_bin(assy, [(0, 500), (500, 1000)])
        # clip_id is a MultiIndex level on the presentation dim
        assert list(result.indexes['presentation'].get_level_values('clip_id')) == ['c0', 'c1']
        assert list(result.indexes['time_bin'].get_level_values('time_bin_start_ms')) == [0, 500]
        assert list(result.indexes['time_bin'].get_level_values('time_bin_end_ms')) == [500, 1000]

    def test_neuroid_coords_preserved(self):
        assy = _make_frame_assembly([
            ('c0', [(0, [1, 1, 1])]),
        ])
        result = temporal_bin(assy, [(0, 1000)])
        assert list(result.indexes['neuroid'].get_level_values('neuroid_id')) == ['n0', 'n1', 'n2']


class TestTemporalBinAggregation:
    def test_mean_aggregation_correctness(self):
        # In bin (0, 1000): frames at 0 and 500 with features [1, 0, 0] and [3, 0, 0]
        # Mean should be [2, 0, 0]
        # In bin (1000, 2000): frames at 1000 and 1500 with features [5, 0, 0] and [7, 0, 0]
        # Mean should be [6, 0, 0]
        assy = _make_frame_assembly([
            ('c0', [(0, [1, 0, 0]), (500, [3, 0, 0]),
                    (1000, [5, 0, 0]), (1500, [7, 0, 0])]),
        ])
        result = temporal_bin(assy, [(0, 1000), (1000, 2000)],
                              aggregation='mean')
        np.testing.assert_allclose(result.values[0, 0], [2.0, 0.0, 0.0])
        np.testing.assert_allclose(result.values[0, 1], [6.0, 0.0, 0.0])

    def test_bins_are_half_open(self):
        """Frame at t=500 should fall in bin (500, 1000), not (0, 500)."""
        assy = _make_frame_assembly([
            ('c0', [(0, [1.0]), (500, [9.0])]),
        ])
        result = temporal_bin(assy, [(0, 500), (500, 1000)])
        # Bin (0, 500) gets only frame at t=0
        np.testing.assert_allclose(result.values[0, 0], [1.0])
        # Bin (500, 1000) gets only frame at t=500
        np.testing.assert_allclose(result.values[0, 1], [9.0])

    def test_first_aggregation(self):
        assy = _make_frame_assembly([
            ('c0', [(100, [1.0]), (300, [2.0]),
                    (700, [3.0]), (900, [4.0])]),
        ])
        result = temporal_bin(assy, [(0, 500), (500, 1000)],
                              aggregation='first')
        # Earliest frame in (0, 500) is at 100 → 1.0
        # Earliest frame in (500, 1000) is at 700 → 3.0
        np.testing.assert_allclose(result.values[0, 0], [1.0])
        np.testing.assert_allclose(result.values[0, 1], [3.0])

    def test_none_aggregation_requires_one_frame_per_bin(self):
        # Aggregation 'none' must error when a bin has >1 frame.
        assy = _make_frame_assembly([
            ('c0', [(0, [1.0]), (500, [2.0])]),
        ])
        with pytest.raises(ValueError, match="aggregation='none'"):
            temporal_bin(assy, [(0, 1000)], aggregation='none')

    def test_none_aggregation_passes_with_one_frame(self):
        assy = _make_frame_assembly([
            ('c0', [(100, [5.0])]),
        ])
        result = temporal_bin(assy, [(0, 1000)], aggregation='none')
        np.testing.assert_allclose(result.values[0, 0], [5.0])


class TestTemporalBinEmptyCells:
    def test_empty_cell_is_nan(self):
        # Clip c0 has no frames in bin (500, 1000). That cell should be NaN.
        assy = _make_frame_assembly([
            ('c0', [(100, [1.0])]),
            ('c1', [(100, [2.0]), (700, [3.0])]),
        ])
        result = temporal_bin(assy, [(0, 500), (500, 1000)])
        assert not np.isnan(result.values[0, 0]).any()
        assert np.isnan(result.values[0, 1]).all()
        np.testing.assert_allclose(result.values[1, 1], [3.0])


class TestTemporalBinErrors:
    def test_missing_time_coord_raises(self):
        import pandas as pd
        assy = NeuroidAssembly(
            np.ones((2, 3)),
            coords={
                'stimulus_id': ('presentation', ['s0', 's1']),
                'clip_id': ('presentation', ['c0', 'c0']),
                'neuroid_id': ('neuroid', ['n0', 'n1', 'n2']),
            },
            dims=['presentation', 'neuroid'],
        )
        with pytest.raises(ValueError, match="frame_time_ms"):
            temporal_bin(assy, [(0, 1000)])

    def test_missing_group_coord_raises(self):
        assy = NeuroidAssembly(
            np.ones((2, 3)),
            coords={
                'stimulus_id': ('presentation', ['s0', 's1']),
                'frame_time_ms': ('presentation', [0.0, 500.0]),
                'neuroid_id': ('neuroid', ['n0', 'n1', 'n2']),
            },
            dims=['presentation', 'neuroid'],
        )
        with pytest.raises(ValueError, match="clip_id"):
            temporal_bin(assy, [(0, 1000)])

    def test_unsupported_aggregation_raises(self):
        assy = _make_frame_assembly([('c0', [(0, [1.0])])])
        with pytest.raises(ValueError, match="aggregation="):
            temporal_bin(assy, [(0, 500)], aggregation='median')


class TestClipOrderPreservation:
    def test_clip_order_is_first_appearance_not_sorted(self):
        """Clips should appear in the output in the order they first
        appear in the input, not alphabetically. This matters when
        clips are named in a non-alphabetical natural order."""
        assy = _make_frame_assembly([
            ('zebra', [(0, [1.0])]),
            ('apple', [(0, [2.0])]),
            ('zebra', [(500, [1.5])]),  # zebra appears again after apple
        ])
        result = temporal_bin(assy, [(0, 1000)])
        # zebra first, then apple — not alphabetical
        assert list(result['clip_id'].values) == ['zebra', 'apple']


class TestCustomCoordNames:
    def test_custom_group_and_time_coords(self):
        """temporal_bin should accept non-default coord names."""
        import xarray as xr
        assy = NeuroidAssembly(
            np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32),
            coords={
                'stimulus_id': ('presentation', ['a', 'b', 'c', 'd']),
                'trial': ('presentation', ['t1', 't1', 't2', 't2']),
                'time_sec': ('presentation', [0.0, 0.5, 0.0, 0.5]),
                'neuroid_id': ('neuroid', ['n0']),
            },
            dims=['presentation', 'neuroid'],
        )
        result = temporal_bin(
            assy, [(0.0, 1.0)],
            time_coord='time_sec', group_coord='trial',
        )
        assert result.shape == (2, 1, 1)
        np.testing.assert_allclose(result.values[0, 0], [1.5])  # (1+2)/2
        np.testing.assert_allclose(result.values[1, 0], [3.5])  # (3+4)/2
