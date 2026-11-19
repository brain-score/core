"""A test subject that runs sessions through its typed methods, for tests of core's typed drivers.

Test code only: the typed drivers of core (``drive_typed_session``) run a subject through
``process`` and ``start_task`` / ``start_recording``. Core has no base class for such a
subject; a subject in a repository inherits ``Subject`` and keeps the methods it uses.
"""
from abc import abstractmethod

from brainscore_core.contract import Subject


class TypedSubject(Subject):
    @property
    @abstractmethod
    def region_layer_map(self):
        ...

    @property
    def out_channels(self):
        return {f'neural:{region}' for region in self.region_layer_map}

    def process(self, input_event):
        raise NotImplementedError(f'{type(self).__name__} defines no process()')

    def participate(self, session) -> None:
        from brainscore_core.streaming_helpers import drive_typed_session
        drive_typed_session(self, session.declare_inputs(self.in_channels))

    def start_task(self, task) -> None:
        self._task = task

    def start_recording(self, recording_target, time_bins=None, recording_type=None) -> None:
        pass
