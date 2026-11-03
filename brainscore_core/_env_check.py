"""Former private name of :mod:`brainscore_core.environment_check`.

The old name is the same module object, not a copy, so code that patches
``_env_check.version`` (as older tests do) still changes what the checks see.
"""
import sys

from . import environment_check

sys.modules[__name__] = environment_check
