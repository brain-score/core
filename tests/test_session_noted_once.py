import numpy as np

from brainscore_core.streaming import StreamEvent


def _answer_session(session):
    from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
    while (event := session.next_input()) is not None:
        session.respond(StreamEvent(session.outputs[0], NeuroidAssembly(
            np.ones((1, 1)), dims=['presentation', 'neuroid'],
            coords={'stimulus_id': ('presentation', ['s0']), 'neuroid_id': ('neuroid', ['n0'])}),
            event.t_ms))

def test_a_session_is_noted_once_not_once_per_response(monkeypatch):
    from brainscore_core import streaming
    from brainscore_core.streaming import Session
    calls = []
    monkeypatch.setattr(streaming, 'note_session', lambda session, *a: calls.append(session))
    session = Session([StreamEvent('text', 'a', 0.0), StreamEvent('text', 'b', 1.0)], ['neural:IT'])
    _answer_session(session)
    assert len(session.emitted) == 2 and len(calls) == 1
