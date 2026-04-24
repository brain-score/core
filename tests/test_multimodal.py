"""Tests for MultimodalStimulusSet — the multi-stream stimulus schema
used by naturalistic movie-watching / speech-comprehension benchmarks.
"""

import pandas as pd
import pytest

from brainscore_core.multimodal import (
    MODALITY_COLUMN_CANDIDATES,
    MultimodalStimulusSet,
)
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet


# ── Construction + detection ────────────────────────────────────────

class TestConstruction:

    def test_empty_df_has_no_modalities(self):
        mm = MultimodalStimulusSet(pd.DataFrame({'stimulus_id': ['a']}))
        assert mm.modalities == set()

    def test_video_only(self):
        mm = MultimodalStimulusSet(pd.DataFrame({
            'stimulus_id': ['c0', 'c1'],
            'video_path': ['c0.mp4', 'c1.mp4'],
        }))
        assert mm.modalities == {'video'}
        assert mm.column_for('video') == 'video_path'

    def test_multimodal_stimulus(self):
        mm = MultimodalStimulusSet(pd.DataFrame({
            'stimulus_id': ['c0'],
            'video_path': ['c0.mp4'],
            'audio_path': ['c0.wav'],
            'sentence': ['hello world'],
        }))
        assert mm.modalities == {'video', 'audio', 'text'}

    def test_image_path_resolves_to_vision(self):
        mm = MultimodalStimulusSet(pd.DataFrame({
            'stimulus_id': ['i0'],
            'image_path': ['i0.jpg'],
        }))
        assert mm.modalities == {'vision'}
        assert mm.column_for('vision') == 'image_path'

    def test_candidate_order_preference(self):
        """When multiple candidates are present, the first-listed wins."""
        # image_file_name comes first in MODALITY_COLUMN_CANDIDATES['vision']
        mm = MultimodalStimulusSet(pd.DataFrame({
            'stimulus_id': ['i0'],
            'image_file_name': ['i0.jpg'],
            'image_path': ['also/i0.jpg'],
        }))
        assert mm.column_for('vision') == 'image_file_name'


# ── Sampling-rate metadata ──────────────────────────────────────────

class TestSampleRates:

    def test_default_no_rates(self):
        mm = MultimodalStimulusSet(pd.DataFrame({
            'stimulus_id': ['c0'], 'video_path': ['c0.mp4'],
        }))
        assert mm.sample_rate_hz('video') is None

    def test_set_and_read_rates(self):
        mm = MultimodalStimulusSet(pd.DataFrame({
            'stimulus_id': ['c0'],
            'video_path': ['c0.mp4'],
            'audio_path': ['c0.wav'],
        }))
        mm.sample_rates_hz = {'video': 30.0, 'audio': 16000.0}
        assert mm.sample_rate_hz('video') == 30.0
        assert mm.sample_rate_hz('audio') == 16000.0
        assert mm.sample_rate_hz('text') is None


# ── Promotion from plain StimulusSet ────────────────────────────────

class TestFromStimulusSet:

    def test_wraps_plain_stimulus_set(self):
        base = StimulusSet(pd.DataFrame({
            'stimulus_id': ['c0'],
            'video_path': ['c0.mp4'],
            'audio_path': ['c0.wav'],
        }))
        base.identifier = 'test-set'
        mm = MultimodalStimulusSet.from_stimulus_set(
            base, sample_rates_hz={'video': 25},
        )
        assert isinstance(mm, MultimodalStimulusSet)
        assert mm.identifier == 'test-set'
        assert mm.sample_rate_hz('video') == 25
        assert mm.modalities == {'video', 'audio'}

    def test_preserves_stimulus_paths(self):
        base = StimulusSet(pd.DataFrame({
            'stimulus_id': ['c0'], 'video_path': ['c0.mp4'],
        }))
        base.stimulus_paths = {'c0': '/tmp/c0.mp4'}
        mm = MultimodalStimulusSet.from_stimulus_set(base)
        assert mm.stimulus_paths == {'c0': '/tmp/c0.mp4'}

    def test_override_identifier(self):
        base = StimulusSet(pd.DataFrame({'stimulus_id': ['c0']}))
        base.identifier = 'original'
        mm = MultimodalStimulusSet.from_stimulus_set(
            base, identifier='renamed')
        assert mm.identifier == 'renamed'


# ── Pandas subclassing behavior ─────────────────────────────────────

class TestPandasCompatibility:
    """MultimodalStimulusSet subclasses DataFrame; slicing/indexing must
    return the subclass, not a plain DataFrame, so attributes survive."""

    def test_slice_returns_same_class(self):
        mm = MultimodalStimulusSet(pd.DataFrame({
            'stimulus_id': ['a', 'b', 'c'],
            'video_path': ['a.mp4', 'b.mp4', 'c.mp4'],
        }))
        sliced = mm.iloc[:2]
        assert isinstance(sliced, MultimodalStimulusSet)

    def test_column_selection_returns_same_class(self):
        mm = MultimodalStimulusSet(pd.DataFrame({
            'stimulus_id': ['a'], 'video_path': ['a.mp4'],
            'audio_path': ['a.wav'],
        }))
        subset = mm[['stimulus_id', 'video_path']]
        assert isinstance(subset, MultimodalStimulusSet)


# ── Error cases ─────────────────────────────────────────────────────

class TestErrors:

    def test_missing_modality_raises(self):
        mm = MultimodalStimulusSet(pd.DataFrame({
            'stimulus_id': ['c0'], 'video_path': ['c0.mp4'],
        }))
        with pytest.raises(KeyError, match="audio"):
            mm.column_for('audio')


# ── Canonical mapping exposure ──────────────────────────────────────

class TestCanonicalMapping:

    def test_known_modalities(self):
        assert 'vision' in MODALITY_COLUMN_CANDIDATES
        assert 'audio' in MODALITY_COLUMN_CANDIDATES
        assert 'video' in MODALITY_COLUMN_CANDIDATES
        assert 'text' in MODALITY_COLUMN_CANDIDATES

    def test_candidates_are_tuples(self):
        for k, v in MODALITY_COLUMN_CANDIDATES.items():
            assert isinstance(v, tuple)
            assert all(isinstance(c, str) for c in v)
