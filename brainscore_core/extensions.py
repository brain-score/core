"""Public registration for external stimulus domains and channel payloads.

``register_channel`` is the one way to add a channel family and the stimulus
columns that feed it. It lives in the channel catalog; this module is its
public import path.
"""

from brainscore_core.io_catalog import CatalogEntry, register_channel
from brainscore_core.streaming import StreamEvent


__all__ = ['CatalogEntry', 'StreamEvent', 'register_channel']
