"""process() is a typed convenience of UnifiedModel, not a requirement on it."""
import pandas as pd
import pytest

from brainscore_core.contract import UnifiedModel
from brainscore_core.streaming_helpers import stimulus_session
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet


class NoProcess(UnifiedModel):
    @property
    def identifier(self):
        return 'no-process'

    @property
    def region_layer_map(self):
        return {'IT': 'layer'}

    @property
    def supported_modalities(self):
        return {'vision'}


def test_typed_subclass_instantiates_without_process_and_fails_clearly():
    subject = NoProcess()  # no abstract method stops instantiation
    stimuli = StimulusSet(pd.DataFrame({'stimulus_id': ['a'], 'image_path': ['/tmp/a.png']}))
    stimuli.identifier = 'one'
    session = stimulus_session(stimuli, ['neural:IT'])
    with pytest.raises(NotImplementedError, match='defines no process'):
        subject.participate(session)
