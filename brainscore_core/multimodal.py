"""
MultimodalStimulusSet — a StimulusSet with multiple media columns and
per-modality sampling-rate metadata.

Motivation (design doc §7 + §8): a naturalistic movie-watching benchmark
presents subjects with a multi-stream stimulus (video frames + audio
waveform + optional text transcript). Models that can consume every
stream (TRIBEv2, Qwen2.5-VL) should receive the full stimulus; models
that only consume a subset (CLIP, VideoMAE, Wav2Vec-Bert) should still
be scorable on the same benchmark using only the streams they support.

The schema is intentionally thin — a pandas DataFrame with:
- per-row media columns (one column per modality stream, following the
  conventions already registered in BrainScoreModel.COLUMN_TO_MODALITY:
  ``image_path`` / ``video_path`` / ``audio_path`` / ``sentence``),
- per-modality sampling-rate metadata (a dict stored as an attribute
  rather than a per-row column, because it typically applies uniformly
  across stimuli in a benchmark),
- optional per-row temporal window metadata (``onset_ms`` / ``duration_ms``)
  for benchmarks that sub-sample continuous recordings into per-clip TRs.

What we DO NOT encode here:
- Subject/session identity — that belongs on the NeuroidAssembly (the
  neural side), not the stimulus. A single stimulus is shown to multiple
  subjects; one stimulus set is reused across sessions.
- Fusion strategies — models declare their own fusion; stimuli are
  just data.
- Alignment to neural time — the benchmark owns the alignment
  function (TR-to-stimulus-window), not the stimulus container.
"""

from typing import Dict, Optional

import pandas as pd

from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet


# Canonical mapping: modality name -> list of column names that carry
# that modality's content. Order matters: the first column that is
# present in the DataFrame wins as the primary column for that modality.
# This mirrors (and consolidates with) BrainScoreModel.COLUMN_TO_MODALITY
# while keeping the *stimulus* side authoritative about what columns
# signal what modality.
MODALITY_COLUMN_CANDIDATES: Dict[str, tuple] = {
    'vision': ('image_file_name', 'image_path', 'filename'),
    'video': ('video_path', 'video_file_name'),
    'audio': ('audio_path', 'audio_file_name', 'audio_file'),
    'text': ('sentence', 'text'),
}


class MultimodalStimulusSet(StimulusSet):
    """A StimulusSet with multi-modality metadata.

    Inherits from :class:`StimulusSet` (which itself subclasses
    :class:`pandas.DataFrame`). Adds:

    - ``sample_rates_hz``: ``dict[modality_name, float]`` — nominal
      sampling rate per modality. Video: frames/sec. Audio: Hz.
      Text: tokens/sec (optional; frequently None for sentence-level).
      Vision (still images): None.
    - ``modality_columns``: ``dict[modality_name, str]`` — resolved
      mapping from modality to the DataFrame column that carries its
      payload. Filled at construction from ``MODALITY_COLUMN_CANDIDATES``.

    Usage::

        mm = MultimodalStimulusSet(
            pd.DataFrame({
                'stimulus_id': ['clip_0', 'clip_1'],
                'video_path': ['clip_0.mp4', 'clip_1.mp4'],
                'audio_path': ['clip_0.wav', 'clip_1.wav'],
                'sentence': ['hello', 'world'],
            })
        )
        mm.sample_rates_hz = {'video': 30, 'audio': 16000}
        assert mm.modalities == {'video', 'audio', 'text'}
        assert mm.column_for('audio') == 'audio_path'
    """

    # Allow the extra attributes to survive pandas DataFrame subclassing.
    _metadata = StimulusSet._metadata + ['sample_rates_hz', 'modality_columns']

    @property
    def _constructor(self):
        return MultimodalStimulusSet

    # ── Public accessors ────────────────────────────────────────────

    @property
    def modalities(self) -> set:
        """Modalities present in the stimulus set, detected from columns.

        Recomputes on every access — do not rely on this for hot loops;
        read :py:attr:`modality_columns` instead (which is resolved at
        construction or after ``refresh_modality_columns()``)."""
        return set(self._detect_modality_columns().keys())

    def column_for(self, modality: str) -> str:
        """Return the column name this stimulus set uses for ``modality``.

        Raises :class:`KeyError` if the modality is not present.
        """
        mapping = getattr(self, 'modality_columns', None)
        if mapping is None:
            mapping = self._detect_modality_columns()
        if modality not in mapping:
            raise KeyError(
                f"Modality {modality!r} not present in MultimodalStimulusSet. "
                f"Available: {sorted(mapping.keys())}.")
        return mapping[modality]

    def sample_rate_hz(self, modality: str) -> Optional[float]:
        """Nominal sampling rate for a modality. None if not declared."""
        rates = getattr(self, 'sample_rates_hz', None) or {}
        return rates.get(modality)

    def refresh_modality_columns(self) -> None:
        """Re-scan columns and repopulate ``modality_columns``.

        Call this after mutating the DataFrame if you add or remove
        modality columns and want the resolved mapping to reflect that.
        """
        self.modality_columns = self._detect_modality_columns()

    # ── Internals ───────────────────────────────────────────────────

    def _detect_modality_columns(self) -> Dict[str, str]:
        resolved: Dict[str, str] = {}
        cols = set(self.columns)
        for modality, candidates in MODALITY_COLUMN_CANDIDATES.items():
            for candidate in candidates:
                if candidate in cols:
                    resolved[modality] = candidate
                    break
        return resolved

    @classmethod
    def from_stimulus_set(
        cls,
        base: 'StimulusSet',
        sample_rates_hz: Optional[Dict[str, float]] = None,
        identifier: Optional[str] = None,
    ) -> 'MultimodalStimulusSet':
        """Promote a plain :class:`StimulusSet` to a MultimodalStimulusSet.

        Useful when a benchmark receives a vanilla StimulusSet but needs
        to attach per-modality sampling metadata. The underlying DataFrame
        is not copied — this is a cheap wrapping.
        """
        mm = cls(base)
        if identifier is not None:
            mm.identifier = identifier
        elif hasattr(base, 'identifier'):
            mm.identifier = base.identifier
        if hasattr(base, 'stimulus_paths'):
            mm.stimulus_paths = base.stimulus_paths
        mm.sample_rates_hz = dict(sample_rates_hz) if sample_rates_hz else {}
        mm.refresh_modality_columns()
        return mm
