"""Streaming primitives for the Unified Model Interface v2.0 contract."""

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
import re
from typing import Any, Deque, Iterable, Optional


_FAMILY_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_ADDRESS_RE = re.compile(r"^[A-Za-z0-9_./\-\[\]:]+$")


@dataclass
class StreamEvent:
    channel: str
    payload: Any
    t_ms: float
    meta: dict = field(default_factory=dict)


def parse_channel(name: str) -> tuple[str, str | None]:
    """Parse a channel name into ``(family, address)``.

    Channel names follow the v2.0 grammar ``family`` or ``family:address``.
    Registry-specific meaning and payload validation are handled elsewhere.
    """
    if not isinstance(name, str):
        raise TypeError("channel name must be a string")

    if ":" in name:
        family, address = name.split(":", 1)
    else:
        family, address = name, None

    if not _FAMILY_RE.fullmatch(family):
        raise ValueError(
            "channel family must start with a lowercase letter and contain "
            "only lowercase letters, digits, and underscores"
        )

    if address is None:
        return family, None

    if (
        not address
        or address.startswith(":")
        or address.endswith(":")
        or not _ADDRESS_RE.fullmatch(address)
    ):
        raise ValueError(
            "channel address must be non-empty and contain only non-space "
            "address characters"
        )

    return family, address


class Session(ABC):
    @abstractmethod
    def next_input(self) -> Optional[StreamEvent]:
        """Return the next input event, or ``None`` when no input is ready."""

    @abstractmethod
    def emit(self, event: StreamEvent) -> None:
        """Record an output event emitted by the subject."""


class InMemorySession(Session):
    """Buffered session for local tests and open-loop synthetic interactions."""

    def __init__(self, inputs: Iterable[StreamEvent] = ()):
        self._inputs: Deque[StreamEvent] = deque(inputs)
        self.emitted: list[StreamEvent] = []

    def next_input(self) -> Optional[StreamEvent]:
        if not self._inputs:
            return None
        return self._inputs.popleft()

    def emit(self, event: StreamEvent) -> None:
        if not isinstance(event, StreamEvent):
            raise TypeError("event must be a StreamEvent")
        self.emitted.append(event)
