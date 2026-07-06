import pytest

from brainscore_core.model_interface import BrainScoreModel, Subject, UnifiedModel


class _ChannelSubject(Subject):
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
    assert subject.out_channels == {"neural:V1", "neural:IT", "behavior"}
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


def test_interact_default_is_non_abstract_and_clear():
    subject = _ChannelSubject()

    with pytest.raises(NotImplementedError, match="no v2 interact\\(\\) path yet"):
        subject.interact(session=None)


def test_existing_subject_subclass_still_instantiates_without_interact():
    class ExistingSubject(Subject):
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


def test_unifiedmodel_alias_still_resolves():
    assert UnifiedModel is Subject
