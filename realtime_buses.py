from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any, Optional


@dataclass
class FramePacket:
    frame: Any
    ts: float
    frame_idx: int


@dataclass
class KeypointPacket:
    points: dict[str, dict[str, float | None]]
    ts: float
    frame_idx: int


class LatestBus:
    def __init__(self) -> None:
        self._lock = Lock()
        self._value: Optional[Any] = None

    def put(self, value: Any) -> None:
        with self._lock:
            self._value = value

    def get(self) -> Optional[Any]:
        with self._lock:
            return self._value