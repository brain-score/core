from brainscore_dummy import model_registry

from .model import DummyModel  # doesn't exist, but we only parse and never run this file

model_registry['dummy-model'] = DummyModel