import pytest

from brainscore_core.model_interface import BrainScoreModel, Subject, UnifiedModel


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
    def supported_modalities(self):
        return {"vision", "text"}

    @property
    def required_modalities(self):
        return {"text"}

    def process(self, input_event):
        return None


def test_subject_derives_channel_identity_from_v1_contract():
    subject = _ChannelSubject()

    assert subject.in_channels == {"vision", "text"}
    # Contract v3 rule 2.2: outputs come from declarations, never from private
    # attribute names. The fixture's ``_behavioral_readout_layer`` declares nothing.
    assert subject.out_channels == {"neural:V1", "neural:IT"}
    assert subject.required_channels == {"text"}


def test_subject_channel_properties_are_overridable():
    class ExplicitChannelSubject(_ChannelSubject):
        @property
        def in_channels(self):
            return {"vision", "instruction"}

        @property
        def out_channels(self):
            return {"behavior"}

        @property
        def required_channels(self):
            return {"instruction"}

    subject = ExplicitChannelSubject()

    assert subject.in_channels == {"vision", "instruction"}
    assert subject.out_channels == {"behavior"}
    assert subject.required_channels == {"instruction"}


def test_typed_interact_default_runs_sessions_and_rejects_unknown_outputs():
    from brainscore_core.streaming import BufferedSession

    subject = _ChannelSubject()
    with pytest.raises(NotImplementedError, match="runs sessions requesting"):
        subject.interact(BufferedSession(requested_output_channels=["pupil"]))


def test_legacy_subclass_still_instantiates_without_interact():
    class ExistingSubject(UnifiedModel):
        @property
        def identifier(self):
            return "existing"

        @property
        def region_layer_map(self):
            return {}

        @property
        def supported_modalities(self):
            return {"vision"}

        def process(self, input_event):
            return "ok"

    subject = ExistingSubject()

    assert subject.process(None) == "ok"
    assert subject.in_channels == {"vision"}


def test_brainscore_model_gets_channel_identity_from_preprocessors():
    model = BrainScoreModel(
        identifier="brain-score-model",
        model=None,
        region_layer_map={"IT": "layer4"},
        preprocessors={"vision": lambda *args, **kwargs: None},
        behavioral_readout_layer="layer4",
    )

    assert model.in_channels == {"vision"}
    assert model.out_channels == {"neural:IT", "behavior"}
    assert model.required_channels == set()


def test_unifiedmodel_remains_a_subject():
    assert issubclass(UnifiedModel, Subject)
    assert UnifiedModel is not Subject
