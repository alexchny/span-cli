from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any


@dataclass
class Event:
    timestamp: str
    event_type: str
    data: dict[str, Any]

    @staticmethod
    def create(event_type: str, **data: Any) -> "Event":
        return Event(
            timestamp=datetime.now(UTC).isoformat(),
            event_type=event_type,
            data=data,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "data": self.data,
        }
