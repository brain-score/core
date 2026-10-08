"""Small, serializable specifications for environment observations and actions.

Custom specifications need validate(value) and describe(). They do not need to
inherit a Brain-Score class. Validation never clips or silently casts values.
"""
from collections.abc import Mapping
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class ArraySpec:
    """Array shape and dtype, with optional inclusive bounds.

    A None dimension accepts any length, for example (None, None, 3) for RGB.
    dtype=None accepts numeric types without casting. Use ActionSpec when
    commands also need named dimensions, units, and a coordinate frame.
    """
    shape: tuple
    dtype: object = None
    lower: object = None
    upper: object = None

    def __post_init__(self):
        if any(d is not None and (type(d) is not int or d < 0) for d in self.shape):
            raise ValueError("Shape dimensions must be nonnegative integers or None")
        if self.dtype is not None:
            np.dtype(self.dtype)
        for bound in (self.lower, self.upper):
            if bound is not None and not np.all(np.isfinite(bound)):
                raise ValueError("Bounds must be finite")
        if self.lower is not None and self.upper is not None:
            if np.any(np.asarray(self.lower) > np.asarray(self.upper)):
                raise ValueError("Bounds must be ordered")

    def validate(self, value):
        array = np.asarray(value)
        if (array.ndim != len(self.shape) or any(
                d is not None and actual != d
                for actual, d in zip(array.shape, self.shape))):
            raise ValueError(f"Expected shape {self.shape}, got {array.shape}")
        if self.dtype is not None and array.dtype != np.dtype(self.dtype):
            raise ValueError(f"Expected dtype {np.dtype(self.dtype)}, got {array.dtype}")
        if array.dtype.kind not in 'biuf' or not np.all(np.isfinite(array)):
            raise ValueError("Expected finite numeric values")
        if ((self.lower is not None and np.any(array < self.lower)) or
                (self.upper is not None and np.any(array > self.upper))):
            raise ValueError("Value exceeds declared bounds; no implicit clipping")
        return array.copy()

    def describe(self):
        return {"kind": "array", "shape": list(self.shape),
                "dtype": str(np.dtype(self.dtype)) if self.dtype is not None else "numeric",
                "lower": None if self.lower is None else np.asarray(self.lower).tolist(),
                "upper": None if self.upper is None else np.asarray(self.upper).tolist()}


@dataclass(frozen=True)
class DiscreteSpec:
    """Integer choices [start, start + count). Accept a scalar or one-item vector."""
    count: int
    start: int = 0

    def __post_init__(self):
        if type(self.count) is not int or self.count < 1 or type(self.start) is not int:
            raise ValueError("count must be a positive integer and start an integer")

    def validate(self, value):
        array = np.asarray(value)
        if array.shape not in ((), (1,)) or array.dtype.kind not in 'iu':
            raise ValueError("Expected one integer action")
        value = int(array.item())
        if not self.start <= value < self.start + self.count:
            raise ValueError("Action exceeds declared bounds; no implicit wrapping")
        return value

    def describe(self):
        return {"kind": "discrete", "count": self.count, "start": self.start}


@dataclass(frozen=True)
class TextSpec:
    """A text field such as a task instruction."""

    def validate(self, value):
        if not isinstance(value, str):
            raise ValueError("Expected text")
        return value

    def describe(self):
        return {"kind": "text"}


@dataclass(frozen=True)
class MappingSpec:
    """Named fields, each with its own specification. Extra fields are explicit."""
    fields: Mapping
    allow_extra: bool = False

    def __post_init__(self):
        for spec in self.fields.values():
            if not callable(getattr(spec, 'validate', None)) or not callable(getattr(spec, 'describe', None)):
                raise TypeError("Specifications need validate(value) and describe()")

    def validate(self, value):
        if not isinstance(value, Mapping):
            raise ValueError("Expected a mapping")
        missing = self.fields.keys() - value.keys()
        extra = value.keys() - self.fields.keys()
        if missing or (extra and not self.allow_extra):
            raise ValueError(f"Invalid fields: missing={missing}, extra={extra}")
        result = dict(value)
        for name, spec in self.fields.items():
            try:
                result[name] = spec.validate(value[name])
            except (TypeError, ValueError) as error:
                raise ValueError(f"{name}: {error}") from error
        return result

    def describe(self):
        return {"kind": "mapping", "fields": {str(k): v.describe() for k, v in self.fields.items()},
                "allow_extra": self.allow_extra}


@dataclass(frozen=True)
class ActionSpec:
    """Named continuous commands with explicit units, bounds, frame and period."""
    names: tuple[str, ...]
    units: tuple[str, ...]
    frame: str
    period_ms: float
    lower: tuple[float, ...]
    upper: tuple[float, ...]

    def __post_init__(self):
        n = len(self.names)
        if not n or len(set(self.names)) != n or any(not x for x in self.names):
            raise ValueError('Action names must be nonempty and unique')
        if any(len(x) != n for x in (self.units, self.lower, self.upper)):
            raise ValueError('Action names, units and bounds must have equal length')
        if not self.frame or any(not x for x in self.units):
            raise ValueError('Explicit units and coordinate frame are required')
        if not np.isfinite(self.period_ms) or self.period_ms <= 0:
            raise ValueError('period_ms must be positive and finite')
        if not np.all(np.isfinite([self.lower, self.upper])) or np.any(
                np.asarray(self.lower) > np.asarray(self.upper)):
            raise ValueError('Action bounds must be finite and ordered')

    def validate(self, action):
        action = np.asarray(action)
        if action.shape != (len(self.names),) or action.dtype.kind not in 'fiu':
            raise ValueError(f'Expected numeric action shape {(len(self.names),)}')
        if not np.all(np.isfinite(action)):
            raise ValueError('Action contains nonfinite values')
        if np.any(action < self.lower) or np.any(action > self.upper):
            raise ValueError('Action exceeds declared bounds; no implicit clipping')
        return action.copy()


    def describe(self):
        return {"kind": "action", **asdict(self)}

