import json
from pathlib import Path
from typing import Any

from span.models.events import Event


class EventStream:
    def __init__(self, log_path: Path | None = None):
        if log_path is None:
            log_path = Path.cwd() / ".span" / "events.jsonl"
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event_type: str, **data: Any) -> None:
        event = Event.create(event_type, **data)
        self._write_event(event)

    def _write_event(self, event: Event) -> None:
        with open(self.log_path, "a") as f:
            json.dump(event.to_dict(), f)
            f.write("\n")

    def read_all(self) -> list[Event]:
        if not self.log_path.exists():
            return []

        events = []
        with open(self.log_path) as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    events.append(
                        Event(
                            timestamp=data["timestamp"],
                            event_type=data["event_type"],
                            data=data["data"],
                        )
                    )
        return events

    def clear(self) -> None:
        if self.log_path.exists():
            self.log_path.unlink()
