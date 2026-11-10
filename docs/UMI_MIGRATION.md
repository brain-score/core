# Migrating core concepts to UMI

`Subject` is the session contract. A native subject declares `identifier`,
`in_channels`, and `out_channels`, and implements `participate(session)`.
`required_channels` defaults to an empty set. Override `reset()` when the subject
holds state that must be cleared between independent evaluations. A `behavior`
response is a `BehavioralAssembly` labeled by `stimulus_id`, so a benchmark can
match each answer to the stimulus that produced it.

```python
from brainscore_core import Subject
from brainscore_core.streaming import BufferedSession, StreamEvent
from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly

class EchoSubject(Subject):
    identifier = "echo"
    in_channels = {"text"}
    out_channels = {"behavior"}
    required_channels = {"text"}

    def participate(self, session):
        while (event := session.next_input()) is not None:
            answer = BehavioralAssembly(
                [event.payload],
                coords={"stimulus_id": ("presentation", [event.meta["stimulus_id"]]),
                        "sentence": ("presentation", [event.payload])},
                dims=["presentation"])
            session.respond(StreamEvent(
                channel="behavior", payload=answer,
                t_ms=event.t_ms, meta=dict(event.meta)))

subject = EchoSubject()
session = BufferedSession(
    inputs=[StreamEvent(channel="text", payload="hello", t_ms=0.0,
                        meta={"stimulus_id": "s1"})],
    requested_output_channels=["behavior"])
subject.participate(session=session)
response = session.emitted[0].payload
assert response["stimulus_id"].values.tolist() == ["s1"]
assert response["sentence"].values.tolist() == ["hello"]
assert response.values.tolist() == ["hello"]
subject.reset()
```

A native subject does not need `process()`, `start_recording()`, `start_task()`,
a layer map, modality properties, or `check_session_support()`. Scoring preflight
uses channel declarations. A custom session only needs `next_input()` and
`respond()`; collection belongs to the session or the benchmark helper.
(`InMemorySession` and `emit()` are accepted compatibility spellings.)

For table-based evaluations, `neural_response` and `behavioral_response` in
`brainscore_core.streaming_helpers` build sessions and collect assemblies. The
benchmark applies its metric to those responses. `apply_state_change` and
`run_environment` do the same for interventions and environment episodes.

## Choose helpers for your subject

`UnifiedModel` is a `Subject` base with task, recording, and `process()` methods.
It requires `identifier`, `region_layer_map`, and `supported_modalities`, and
derives channel declarations from the modalities. Use `BrainScoreModel` from the
unified `brainscore` package (`from brainscore import BrainScoreModel`) for a
configurable implementation with extraction helpers. Core no longer contains
it: `brainscore_core.BrainScoreModel`, the unit selectors and
`brainscore_core.capabilities` moved to `brainscore`, `brainscore.selection`
and `brainscore.capabilities`.

`UnifiedModel` is a subclass of `Subject`, not an alias. `BrainScoreModel` and
the vision and language adapters inherit this compatibility base, so they
remain subjects and keep their existing computation paths.

Implement `Subject` directly when you want to own session handling. Benchmarks
reach any subject the same way:

- A native benchmark uses the typed helpers above or drives a session itself,
  so it works with any subject; it does not need `process()` or recording
  methods on the subject.
- An existing legacy benchmark (one that calls `look_at`, `digest_text`,
  `start_recording` and similar) is scored through the permanent legacy
  benchmark facade. `score()` wraps the subject in it automatically, and the
  facade runs each legacy call as a session through `participate`. Legacy
  benchmarks are never removed, so this path does not go away.

For a `BrainScoreModel`, configure the parts by name: `extractor=` for
recorded activity, `answer=` for instruction-following behavior, `policy=`
for environment actions, and `intervention=` for lesions or stimulation.

Writing an adapter for a model that already has those typed methods? Its
`participate(session)` can use the public functions in
`brainscore_core.streaming_helpers`. `drain_session_inputs(session)` reads
every input event in order. `session_stimuli(session, events)` returns the
stimulus table the session presented. `drive_neural_session(subject, session)`
and `drive_behavior_session(subject, session)` answer a `neural:<region>` or
`behavior` request with the subject's own `start_recording` or `start_task`
followed by one `process` call. These functions exist only for subjects that
have those methods. A subject written directly against `Subject` never needs
them.

Existing vision `BrainModel` and language `ArtificialSubject` plugins keep
working through their permanent adapters. No plugin removal or re-registration
is required. Ordinary compositional registrations can continue to instantiate
`BrainScoreModel` and use its supported task, recording, and process methods.
