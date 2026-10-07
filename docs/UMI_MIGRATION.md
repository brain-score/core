# Migrating core concepts to UMI

`Subject` is the session contract. A native subject declares `identifier`,
`in_channels`, and `out_channels`, and implements `interact(session)`.
`required_channels` defaults to an empty set. Override `reset()` when the subject
holds state that must be cleared between independent evaluations.

```python
from brainscore_core import Subject
from brainscore_core.streaming import InMemorySession, StreamEvent

class EchoSubject(Subject):
    identifier = "echo"
    in_channels = {"text"}
    out_channels = {"behavior"}
    required_channels = {"text"}

    def interact(self, session):
        while (event := session.next_input()) is not None:
            session.emit(StreamEvent(
                "behavior", event.payload, event.t_ms, dict(event.meta)))

subject = EchoSubject()
session = InMemorySession([StreamEvent("text", "hello", 0.0)])
subject.interact(session)
assert session.emitted[0].payload == "hello"
subject.reset()
```

A native subject does not need `process()`, `start_recording()`, `start_task()`,
a layer map, modality properties, or `check_session_support()`. Scoring preflight
uses channel declarations. A custom session only needs `next_input()` and
`emit()`; collection belongs to the session or the benchmark helper.

For table-based evaluations, `neural_response` and `behavioral_response` in
`brainscore_core.streaming_helpers` build sessions and collect assemblies. The
benchmark applies its metric to those responses.

## Choose helpers for your subject

`UnifiedModel` is a `Subject` base with task, recording, and `process()` methods.
It requires `identifier`, `region_layer_map`, and `supported_modalities`, and
derives channel declarations from the modalities. Use `BrainScoreModel` for a
configurable implementation with extraction helpers.

`UnifiedModel` is a subclass of `Subject`, not an alias. `BrainScoreModel` and
the vision and language adapters inherit this compatibility base, so they
remain subjects and keep their existing computation paths.

Implement `Subject` directly when you want to own session handling. A benchmark
that calls `process()` or recording methods needs a subject providing those
methods; inheriting `Subject` alone does not provide them.

Existing vision `BrainModel` and language `ArtificialSubject` plugins keep
working through their permanent adapters. No plugin removal or re-registration
is required. Ordinary compositional registrations can continue to instantiate
`BrainScoreModel` and use its supported task, recording, and process methods.
