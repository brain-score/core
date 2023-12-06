from brainscore_dummy import data_registry

from .data import DummyData  # doesn't exist, but we only parse and never run this file

data_registry['dummy-data'] = DummyData