"""BrainScoreModel capability registry."""

from .base import Capability
from .registry import (
    capability_registry,
    enabled_capabilities,
    register_capability,
)

# Import built-ins so their modules self-register.
from . import behavioral_generation as _behavioral_generation  # noqa: F401
from . import behavioral_readout as _behavioral_readout  # noqa: F401
from . import embodied_action as _embodied_action  # noqa: F401
from . import neural as _neural  # noqa: F401
from . import state_change as _state_change  # noqa: F401
