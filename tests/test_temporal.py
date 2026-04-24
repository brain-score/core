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
from brainscore_core.temporal import (
    contiguous_block_cv,
    double_gamma_hrf,
    hrf_convolve,
    temporal_bin,
)


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


# ═════════════════════════════════════════════════════════════════
# Time-series CV + HRF correction
# ═════════════════════════════════════════════════════════════════


class TestContiguousBlockCV:

    def test_every_sample_held_out_exactly_once(self):
        n, block = 100, 10
        test_samples = []
        for train_idx, test_idx in contiguous_block_cv(n, block):
            assert len(set(train_idx) & set(test_idx)) == 0
            assert len(train_idx) + len(test_idx) == n
            test_samples.extend(test_idx.tolist())
        # Every sample appears in exactly one test set
        assert sorted(test_samples) == list(range(n))
        assert len(test_samples) == n

    def test_test_sets_are_contiguous(self):
        """Each test fold must be a contiguous slice (not striped)."""
        for train_idx, test_idx in contiguous_block_cv(50, 5):
            # Contiguous ⇔ max-min+1 == length and values sorted
            assert list(test_idx) == list(range(test_idx.min(), test_idx.max() + 1))

    def test_uneven_division_covers_all_samples(self):
        """13 samples / block=5 → folds of sizes 5, 5, 3."""
        folds = list(contiguous_block_cv(13, 5))
        assert len(folds) == 3
        sizes = [len(t) for _, t in folds]
        assert sizes == [5, 5, 3]
        # And all 13 appear exactly once
        all_test = sorted(np.concatenate([t for _, t in folds]).tolist())
        assert all_test == list(range(13))

    def test_explicit_n_splits(self):
        """n_splits=2 on 20 samples with block=5 → just the first two blocks."""
        folds = list(contiguous_block_cv(20, 5, n_splits=2))
        assert len(folds) == 2
        assert list(folds[0][1]) == [0, 1, 2, 3, 4]
        assert list(folds[1][1]) == [5, 6, 7, 8, 9]

    def test_block_larger_than_n_raises(self):
        with pytest.raises(ValueError, match="must not exceed"):
            list(contiguous_block_cv(10, 20))

    def test_zero_or_negative_raises(self):
        with pytest.raises(ValueError):
            list(contiguous_block_cv(0, 5))
        with pytest.raises(ValueError):
            list(contiguous_block_cv(100, 0))
        with pytest.raises(ValueError):
            list(contiguous_block_cv(100, -5))

    def test_beats_random_on_autocorrelated_series(self):
        """Demonstration: on a perfectly-autocorrelated series, contiguous
        CV correctly scores 0 (no predictive signal) while random KFold
        would leak adjacency and score > 0. Sanity-check test."""
        from sklearn.model_selection import KFold
        from sklearn.linear_model import Ridge

        rng = np.random.default_rng(0)
        n = 200
        # X is just a counter — 1-D feature
        X = np.arange(n).reshape(-1, 1).astype(float)
        # y is smoothed noise — adjacent samples share variance
        noise = rng.standard_normal(n)
        y = np.convolve(noise, np.ones(10) / 10, mode='same')

        # Random KFold — adjacency leaks; predictions look plausible
        random_preds = np.zeros(n)
        for train_idx, test_idx in KFold(5, shuffle=True,
                                         random_state=0).split(X):
            random_preds[test_idx] = Ridge().fit(
                X[train_idx], y[train_idx]).predict(X[test_idx])
        random_r = np.corrcoef(y, random_preds)[0, 1]

        # Contiguous blocks — no leakage; predictions should be chance
        contig_preds = np.zeros(n)
        for train_idx, test_idx in contiguous_block_cv(n, 40):
            contig_preds[test_idx] = Ridge().fit(
                X[train_idx], y[train_idx]).predict(X[test_idx])
        contig_r = np.corrcoef(y, contig_preds)[0, 1]

        # Contiguous score should be ≤ random score on a smooth series
        # (we're not asserting magnitude — only that the ordering is
        # correct on a canonical leakage example).
        assert contig_r <= random_r + 1e-6


class TestDoubleGammaHRF:

    def test_unit_sum_normalization(self):
        h = double_gamma_hrf(duration_sec=32, sampling_rate_hz=10)
        np.testing.assert_allclose(h.sum(), 1.0, rtol=1e-6)

    def test_peak_around_5_seconds(self):
        """Canonical SPM HRF peaks near t≈5s with default parameters."""
        sr = 100
        h = double_gamma_hrf(duration_sec=32, sampling_rate_hz=sr)
        peak_idx = int(np.argmax(h))
        peak_time_s = peak_idx / sr
        assert 4.0 <= peak_time_s <= 6.0, (
            f"Expected peak in [4s, 6s], got {peak_time_s}s")

    def test_undershoot_negative_after_peak(self):
        sr = 100
        h = double_gamma_hrf(duration_sec=32, sampling_rate_hz=sr)
        # Samples around 14-18 s should be negative (undershoot)
        late_idx = slice(int(14 * sr), int(18 * sr))
        assert h[late_idx].min() < 0

    def test_shape_matches_requested_length(self):
        h = double_gamma_hrf(duration_sec=10.5, sampling_rate_hz=4)
        assert h.shape == (42,)  # round(10.5 * 4) = 42

    def test_custom_parameters_shift_peak(self):
        """A later peak_delay should push the peak further in time."""
        sr = 100
        h_early = double_gamma_hrf(32, sr, peak_delay=3)
        h_late = double_gamma_hrf(32, sr, peak_delay=10)
        assert np.argmax(h_early) < np.argmax(h_late)

    def test_zero_duration_raises(self):
        with pytest.raises(ValueError, match="must be > 0"):
            double_gamma_hrf(duration_sec=0, sampling_rate_hz=1)


class TestHRFConvolve:

    def test_shape_preserved(self):
        rng = np.random.default_rng(0)
        features = rng.standard_normal((100, 3))
        out = hrf_convolve(features, sampling_rate_hz=1.0)
        assert out.shape == features.shape
        assert out.dtype == features.dtype

    def test_delta_input_reproduces_hrf(self):
        """A unit impulse at t=0 convolved with the HRF should match the
        HRF itself (truncated to the input length)."""
        sr = 10.0
        hrf = double_gamma_hrf(32, sr)
        n = len(hrf)
        impulse = np.zeros((n, 1))
        impulse[0, 0] = 1.0
        out = hrf_convolve(impulse, sampling_rate_hz=sr)
        np.testing.assert_allclose(out[:, 0], hrf, atol=1e-10)

    def test_causal_no_future_leakage(self):
        """Output at time t must not depend on features at times > t."""
        sr = 5.0
        # Build two features that agree up to t=50 and diverge after.
        base = np.random.default_rng(0).standard_normal(100)
        f1 = np.stack([base], axis=1)
        f2 = f1.copy()
        f2[50:, 0] += 1000.0
        out1 = hrf_convolve(f1, sampling_rate_hz=sr)
        out2 = hrf_convolve(f2, sampling_rate_hz=sr)
        # Outputs should match exactly for the first 50 time points.
        np.testing.assert_allclose(out1[:50], out2[:50], atol=1e-10)

    def test_custom_hrf_passed_through(self):
        """Convolving with a box-car HRF should produce a running mean."""
        box = np.array([1/3, 1/3, 1/3])
        features = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        out = hrf_convolve(features, sampling_rate_hz=1.0, hrf=box)
        # out[0] = 1/3, out[1] = (1+2)/3, out[2] = (1+2+3)/3, ...
        expected = np.array([[1/3], [1.0], [2.0], [3.0], [4.0]])
        np.testing.assert_allclose(out, expected, atol=1e-10)

    def test_empty_features(self):
        out = hrf_convolve(np.zeros((0, 5)), sampling_rate_hz=1.0)
        assert out.shape == (0, 5)

    def test_rank_validation(self):
        with pytest.raises(ValueError, match="2-D"):
            hrf_convolve(np.zeros(10), sampling_rate_hz=1.0)
        with pytest.raises(ValueError, match="2-D"):
            hrf_convolve(np.zeros((2, 3, 4)), sampling_rate_hz=1.0)

    def test_hrf_shape_validation(self):
        features = np.zeros((10, 1))
        bad_hrf = np.zeros((3, 3))
        with pytest.raises(ValueError, match="hrf must be 1-D"):
            hrf_convolve(features, sampling_rate_hz=1.0, hrf=bad_hrf)
