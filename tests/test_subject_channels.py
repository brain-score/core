import pytest

from brainscore_core.contract import UnifiedModel


class _ChannelSubject(UnifiedModel):
    def __init__(self):
        self._behavioral_readout_layer = "readout"

    @property
    def identifier(self):
        return "channel-subject"

    @property
    def region_layer_map(self):
        return {"V1": "layer1", "IT": "layer4"}

    @property
    def in_channels(self):
        return {"vision:image", "text"}

    @property
    def required_channels(self):
        return {"text"}

    def process(self, input_event):
        return None


def test_subject_declares_its_channels():
    subject = _ChannelSubject()

    assert subject.in_channels == {"vision:image", "text"}
    # Contract v3 rule 2.2: outputs come from declarations, never from private
    # attribute names. The fixture's ``_behavioral_readout_layer`` declares nothing.
    assert subject.out_channels == {"neural:V1", "neural:IT"}
    assert subject.required_channels == {"text"}


def test_subject_channel_properties_are_overridable():
    class ExplicitChannelSubject(_ChannelSubject):
        @property
        def in_channels(self):
            return {"vision:image", "instruction"}

        @property
        def out_channels(self):
            return {"behavior"}

        @property
        def required_channels(self):
            return {"instruction"}

    subject = ExplicitChannelSubject()

    assert subject.in_channels == {"vision:image", "instruction"}
    assert subject.out_channels == {"behavior"}
    assert subject.required_channels == {"instruction"}


def test_typed_participate_default_runs_sessions_and_rejects_unknown_outputs():
    from brainscore_core.streaming import Session

    subject = _ChannelSubject()
    with pytest.raises(NotImplementedError, match="runs sessions requesting"):
        subject.participate(Session((), outputs=["pupil"]))


def test_legacy_subclass_still_instantiates_without_participate():
    class ExistingSubject(UnifiedModel):
        @property
        def identifier(self):
            return "existing"

        @property
        def region_layer_map(self):
            return {}

        @property
        def in_channels(self):
            return {"vision:image"}

        def process(self, input_event):
            return "ok"

    subject = ExistingSubject()

    assert subject.process(None) == "ok"
    assert subject.in_channels == {"vision:image"}


def test_typed_model_without_multi_modality_runs_multimodal_session():
    """The driver asks for multi-modality only when the subject says it combines
    inputs; no private method is probed (Contract v3 rule 2.2)."""
    import pandas as pd
    from brainscore_core.streaming_helpers import neural_response
    from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

    class TwoInputModel(_ChannelSubject):
        def __init__(self):
            super().__init__()
            self.calls = []

        def start_recording(self, recording_target, time_bins=None, recording_type=None):
            pass

        def process(self, input_event):  # no multi_modality keyword
            self.calls.append(len(input_event))
            import numpy as np
            from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
            return NeuroidAssembly(np.zeros((len(input_event), 2)), dims=['presentation', 'neuroid'],
                                   coords={'stimulus_id': ('presentation', list(input_event['stimulus_id'])),
                                           'neuroid_id': ('neuroid', [0, 1])})

    stimuli = StimulusSet(pd.DataFrame({'stimulus_id': ['a'], 'image_path': ['/tmp/a.png'],
                                        'sentence': ['a sentence']}))
    stimuli.identifier = 'two-inputs'
    model = TwoInputModel()
    neural_response(model, stimuli, ['neural:IT'])
    assert model.calls == [1]
