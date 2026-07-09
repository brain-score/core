# Migrating core concepts to UMI

The older Brain-Score documentation uses BrainModel and ArtificialSubject.
Those interfaces remain available for existing domain plugins, but new
cross-domain code should target Subject or BrainScoreModel.

| Pre-UMI concept | UMI concept |
| --- | --- |
| BrainModel or ArtificialSubject | Subject |
| look_at(stimuli) | process(stimuli) |
| digest_text(text) | process(stimuli) |
| start_behavioral_task(...) | start_task(TaskContext(...)) |
| start_recording(...) | start_recording(...) |

The unified package adapts compatible legacy models when they are loaded
through its registry. Use the distribution's unified/docs/getting_started.md
for a first run and unified/docs/umi_api_reference.md for the public lifecycle.
