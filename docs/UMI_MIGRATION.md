# Migrating core concepts to UMI

`Subject` is the session contract. A native subject declares `identifier`,
`in_channels`, and `out_channels`, and implements `participate(session)`.
`required_channels` defaults to an empty set. Override `reset()` when the subject
holds state that must be cleared between independent evaluations. A `behavior`
response is a `BehavioralAssembly` labeled by `stimulus_id`, so a benchmark can
match each answer to the stimulus that produced it.

```python
from brainscore_core import Subject
from brainscore_core.streaming import Session, StreamEvent
from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly

# Subject: answers each text with the text itself
class EchoSubject(Subject):
    identifier = "echo"
    in_channels = {"text"}
    out_channels = {"behavior"}
    required_channels = {"text"}

    def participate(self, session):
        while (event := session.next_input()) is not None:
            # One answer row, labelled with the input's stimulus_id
            answer = BehavioralAssembly(
                [event.payload],
                coords={"stimulus_id": ("presentation", [event.meta["stimulus_id"]]),
                        "sentence": ("presentation", [event.payload])},
                dims=["presentation"])
            session.respond(StreamEvent(
                channel="behavior", payload=answer,
                t_ms=event.t_ms, meta=dict(event.meta)))

# One input on 'text'; check the stored answer
subject = EchoSubject()
session = Session(
    [StreamEvent(channel="text", payload="hello", t_ms=0.0,
                 meta={"stimulus_id": "s1"})],
    outputs=["behavior"])
subject.participate(session=session)
response = session.emitted[0].payload
assert response["stimulus_id"].values.tolist() == ["s1"]
assert response["sentence"].values.tolist() == ["hello"]
assert response.values.tolist() == ["hello"]
subject.reset()
```

A native subject does not need `process()`, `start_recording()`, `start_task()`,
a layer map, modality properties, or `check_session_support()`. Scoring preflight
reads the channel declarations. There is one session class, `Session`. It takes
a source of input events (a list, iterator or generator), the outputs you
request, and the task. The `Subject` base class calls
`session.declare_inputs(in_channels)` before the `participate` body runs, so the
subject receives only the channels it declares. Channels match by exact name:
`vision:image` and `vision:video` are two channels. There is no bare `vision`
channel: a subject declares the visual form it takes, or both.

A session built from a stimulus table also holds control events on the
`control` channel: `begin`, `respond` and `end`. A subject gets them only if it
declares `control` as an input. Without that declaration it never sees them, so
a loop that answers every event stays correct.

For table-based evaluations, `neural_response` and `behavioral_response` in
`brainscore_core.streaming_helpers` build sessions and collect assemblies. The
benchmark applies its metric to those responses. `apply_state_change` and
`run_environment` do the same for interventions and environment episodes.

To build a session yourself, use one of two builders with the same shape:

```python
# Arguments: stimuli or environment, requested outputs, task
session = stimulus_session(stimuli, ['neural:IT'], task, time_bins=[(70, 170)])   # IT, 70-170 ms
session = stimulus_session(stimuli, ['behavior', 'neural:IT'], task, streaming=True)   # one row at a time
session = environment_session(environment, ['motor'], task, seed=0)
```

`stimulus_session` presents the rows of a stimulus table. With
`streaming=True` it makes the events one row at a time, and the subject
answers each row before it gets the next. `environment_session` runs a closed
loop; its outputs default to the action channel. For a stream, give
`Session` a generator of events. `TaskContext.trial_memory` is `'none'` (each
trial is its own episode) or `'history'` (all trials are one episode; the
subject needs the trait `memory:trial_history`).

## Choose helpers for your subject

Every subject inherits `Subject`; core has no other base class (`UnifiedModel` was
deleted). Use `BrainScoreModel` from the
unified `brainscore` package (`from brainscore import BrainScoreModel`) for a
configurable implementation with extraction helpers. Core no longer contains
it: `brainscore_core.BrainScoreModel`, the unit selectors and
`brainscore_core.capabilities` moved to `brainscore`, `brainscore.selection`
and `brainscore.capabilities`. `BrainScoreModel` and the vision and language
adapters inherit `Subject`; the adapters keep their existing computation paths. A
subject with `process` and `start_task` / `start_recording` methods can run a
session through them with `brainscore_core.streaming_helpers.drive_typed_session`.

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
A lesion, a stimulation or a pharmacological change arrives on the plain
channel `lesion`, `stimulation` or `pharmacological`. The target is in the
`StateChange` payload, not in the channel name. A subject that takes changes
declares the plain family; a subject that does not declare it never receives
the change.

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
is required. A compositional registration instantiates `BrainScoreModel` and
runs it through `participate(session)` and the session helpers
(`neural_response`, `behavioral_response`, `stimulus_session`).
`BrainScoreModel.process(row)` answers one step of a closed loop only.
