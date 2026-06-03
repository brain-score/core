"""Tests for temporal.synchronize_modalities — the cross-modal alignment step
that resamples several per-modality feature streams onto one shared time grid
(e.g. the brain's TR grid) before joint regression.
"""
import numpy as np
import pytest

from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from brainscore_core.temporal import synchronize_modalities, add_time_bin_axis


def _stream(n_pres, src_bins, n_neuroid, fill=1.0):
    """Build a (presentation, time_bin, neuroid) assembly.

    src_bins: list of (start_ms, end_ms). Each source bin's value is its
    bin index * fill, broadcast across presentations & neuroids, so we can
    check which source bins land in which target bin after resampling.
    """
    n_tb = len(src_bins)
    data = np.zeros((n_pres, n_tb, n_neuroid))
    for bi in range(n_tb):
        data[:, bi, :] = bi * fill
    coords = {
        'clip_id': ('presentation', [f'clip_{i}' for i in range(n_pres)]),
        'stimulus_id': ('presentation', [f'clip_{i}' for i in range(n_pres)]),
        'time_bin_start_ms': ('time_bin', [s for (s, _) in src_bins]),
        'time_bin_end_ms': ('time_bin', [e for (_, e) in src_bins]),
        'neuroid_id': ('neuroid', np.arange(n_neuroid)),
        'layer': ('neuroid', ['L'] * n_neuroid),
    }
    return NeuroidAssembly(data, coords=coords,
                           dims=['presentation', 'time_bin', 'neuroid'])


class TestSynchronizeModalities:
    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one stream"):
            synchronize_modalities({}, [(0, 1000)])

    def test_resample_to_coarser_grid_averages_source_bins(self):
        # 4 source bins at 250ms each (centers 125,375,625,875); target = two
        # 500ms bins. First target bin pools source bins 0,1 (values 0,1 ->0.5);
        # second pools 2,3 (values 2,3 -> 2.5).
        src = [(0, 250), (250, 500), (500, 750), (750, 1000)]
        stream = _stream(n_pres=3, src_bins=src, n_neuroid=2)
        out = synchronize_modalities(
            {'video': stream}, [(0, 500), (500, 1000)], concat_modality=False)
        vid = out['video']
        assert vid.dims == ('presentation', 'time_bin', 'neuroid')
        assert vid.sizes['time_bin'] == 2
        assert np.allclose(vid.values[:, 0, :], 0.5)
        assert np.allclose(vid.values[:, 1, :], 2.5)

    def test_empty_target_bin_is_nan(self):
        src = [(0, 250), (250, 500)]
        stream = _stream(n_pres=1, src_bins=src, n_neuroid=1)
        out = synchronize_modalities(
            {'video': stream}, [(0, 500), (5000, 5500)], concat_modality=False)
        vid = out['video']
        assert not np.isnan(vid.values[:, 0, :]).any()
        assert np.isnan(vid.values[:, 1, :]).all()

    def test_concat_tags_modality(self):
        src = [(0, 500), (500, 1000)]
        v = _stream(n_pres=2, src_bins=src, n_neuroid=3, fill=1.0)
        a = _stream(n_pres=2, src_bins=src, n_neuroid=5, fill=10.0)
        merged = synchronize_modalities(
            {'video': v, 'audio': a}, [(0, 500), (500, 1000)])
        assert merged.sizes['neuroid'] == 8                     # 3 + 5
        assert set(merged['modality'].values.tolist()) == {'video', 'audio'}
        assert (merged['modality'].values == 'video').sum() == 3
        assert (merged['modality'].values == 'audio').sum() == 5

    def test_mismatched_presentations_raise(self):
        src = [(0, 500)]
        v = _stream(n_pres=2, src_bins=src, n_neuroid=1)
        a = _stream(n_pres=3, src_bins=src, n_neuroid=1)
        with pytest.raises(ValueError, match="presentations"):
            synchronize_modalities({'video': v, 'audio': a}, [(0, 500)])

    def test_missing_time_bin_dim_raises(self):
        # A 2D (presentation, neuroid) assembly is rejected with a hint.
        data = np.zeros((2, 1))
        bad = NeuroidAssembly(
            data,
            coords={'clip_id': ('presentation', ['a', 'b']),
                    'stimulus_id': ('presentation', ['a', 'b']),
                    'neuroid_id': ('neuroid', [0]),
                    'layer': ('neuroid', ['L'])},
            dims=['presentation', 'neuroid'])
        with pytest.raises(ValueError, match="no 'time_bin' dim"):
            synchronize_modalities({'video': bad}, [(0, 500)])

    def test_still_image_via_add_time_bin_axis(self):
        # add_time_bin_axis makes a still-image stream synchronizable.
        data = np.ones((2, 4))
        img = NeuroidAssembly(
            data,
            coords={'clip_id': ('presentation', ['a', 'b']),
                    'stimulus_id': ('presentation', ['a', 'b']),
                    'neuroid_id': ('neuroid', np.arange(4)),
                    'layer': ('neuroid', ['L'] * 4)},
            dims=['presentation', 'neuroid'])
        img3d = add_time_bin_axis(img, time_bin_start_ms=0, time_bin_end_ms=1000)
        out = synchronize_modalities(
            {'image': img3d}, [(0, 1000)], concat_modality=False)
        assert out['image'].sizes['time_bin'] == 1
        assert np.allclose(out['image'].values, 1.0)
