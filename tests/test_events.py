from pathlib import Path

from span.events.stream import EventStream
from span.models.events import Event


def test_event_creation() -> None:
    event = Event.create("test_event", message="hello", count=42)

    assert event.event_type == "test_event"
    assert event.data["message"] == "hello"
    assert event.data["count"] == 42
    assert "+00:00" in event.timestamp or event.timestamp.endswith("Z")


def test_event_to_dict() -> None:
    event = Event.create("test_event", key="value")
    data = event.to_dict()

    assert data["event_type"] == "test_event"
    assert data["data"]["key"] == "value"
    assert "timestamp" in data


def test_event_stream_append(tmp_path: Path) -> None:
    log_path = tmp_path / "events.jsonl"
    stream = EventStream(log_path)

    stream.append("session_start", task="test task")
    stream.append("step_started", step_number=1)

    assert log_path.exists()
    content = log_path.read_text()
    assert "session_start" in content
    assert "step_started" in content


def test_event_stream_read_all(tmp_path: Path) -> None:
    log_path = tmp_path / "events.jsonl"
    stream = EventStream(log_path)

    stream.append("event_1", data="first")
    stream.append("event_2", data="second")

    events = stream.read_all()

    assert len(events) == 2
    assert events[0].event_type == "event_1"
    assert events[0].data["data"] == "first"
    assert events[1].event_type == "event_2"
    assert events[1].data["data"] == "second"


def test_event_stream_read_empty(tmp_path: Path) -> None:
    log_path = tmp_path / "events.jsonl"
    stream = EventStream(log_path)

    events = stream.read_all()

    assert events == []


def test_event_stream_clear(tmp_path: Path) -> None:
    log_path = tmp_path / "events.jsonl"
    stream = EventStream(log_path)

    stream.append("test_event", message="test")
    assert log_path.exists()

    stream.clear()
    assert not log_path.exists()


def test_event_stream_default_path(tmp_path: Path) -> None:
    import os

    original_cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        stream = EventStream()

        stream.append("test_event", message="test")

        expected_path = tmp_path / ".span" / "events.jsonl"
        assert expected_path.exists()
    finally:
        os.chdir(original_cwd)


def test_event_stream_creates_directory(tmp_path: Path) -> None:
    log_path = tmp_path / "nested" / "dir" / "events.jsonl"
    stream = EventStream(log_path)

    stream.append("test_event", message="test")

    assert log_path.exists()
    assert log_path.parent.is_dir()
