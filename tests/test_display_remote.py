"""RemoteDisplayClient: pushes DisplaySink events over HTTP instead of
broadcasting in-process (see demo/display/remote.py). No real network — the
`urllib.request.urlopen` boundary is monkeypatched throughout, matching how
demo/detect_source.py's/demo/platform/robot_camera.py's own tests fake the
HTTP call rather than opening a socket.
"""
from __future__ import annotations

import json
import logging
import threading
import time

from demo.display.remote import (QUEUE_MAXSIZE, RemoteDisplayClient,
                                 ack_stream, dashboard_push_url,
                                 stream_request)


class _FakeResponse:
    def close(self) -> None:
        pass


def _wait_until(predicate, timeout: float = 2.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_dashboard_push_url_assembles_host_port_path():
    assert dashboard_push_url("mac.local", 8080) == "http://mac.local:8080/push"


def test_on_heard_pushes_event_and_data(monkeypatch):
    calls = []

    def fake_urlopen(req, timeout=None):
        calls.append((req.full_url, json.loads(req.data)))
        return _FakeResponse()

    monkeypatch.setattr("demo.display.remote.urllib.request.urlopen", fake_urlopen)
    client = RemoteDisplayClient("mac.local", 8080)
    try:
        client.on_heard("hello")
        assert _wait_until(lambda: calls), "push never reached urlopen"
    finally:
        client.close()
    url, body = calls[0]
    assert url == "http://mac.local:8080/push"
    assert body == {"event": "heard", "data": {"text": "hello"}}


def test_on_recall_and_on_speech_recall_shape_hits_like_web_dashboard(monkeypatch):
    calls = []

    def fake_urlopen(req, timeout=None):
        calls.append(json.loads(req.data))
        return _FakeResponse()

    monkeypatch.setattr("demo.display.remote.urllib.request.urlopen", fake_urlopen)
    client = RemoteDisplayClient("mac.local", 8080)
    try:
        client.on_recall([{"jpeg_b64": "AAAA", "detections": [{"label": "cup"}],
                          "score": 0.13, "found_by": "picture"}])
        client.on_speech_recall([{"text": "my stream starts at 16:00",
                                 "score": 0.812}])
        assert _wait_until(lambda: len(calls) == 2)
    finally:
        client.close()
    recall, speech = calls
    assert recall == {"event": "recall",
                      "data": {"frames": [{"jpeg_b64": "AAAA",
                                          "boxes": [{"label": "cup"}],
                                          "score": 0.13, "found_by": "picture",
                                          "weak": False}]}}
    assert speech == {"event": "speech_recall",
                      "data": {"items": [{"text": "my stream starts at 16:00",
                                         "score": 0.812, "source": ""}]}}


def test_queue_drops_oldest_event_when_full(monkeypatch):
    """Mirrors test_subscriber_queue_drops_oldest_when_full (WebDashboard's
    own SSE queues) — a sustained backlog drops the oldest push, not the
    newest, and never blocks the caller."""
    block_started = threading.Event()
    release = threading.Event()
    calls = []

    def fake_urlopen(req, timeout=None):
        if not calls:
            # Stall the very first call so the worker is busy while the
            # queue fills up behind it.
            block_started.set()
            release.wait(timeout=2.0)
        calls.append(json.loads(req.data))
        return _FakeResponse()

    monkeypatch.setattr("demo.display.remote.urllib.request.urlopen", fake_urlopen)
    client = RemoteDisplayClient("mac.local", 8080)
    try:
        client.on_heard("first")
        assert block_started.wait(timeout=1.0)
        for i in range(QUEUE_MAXSIZE + 5):
            client.on_detections([{"label": str(i)}])
        release.set()
        assert _wait_until(lambda: len(calls) == QUEUE_MAXSIZE + 1)
    finally:
        client.close()
    assert calls[0] == {"event": "heard", "data": {"text": "first"}}
    survivors = [c["data"]["boxes"][0]["label"] for c in calls[1:]]
    assert survivors[-1] == str(QUEUE_MAXSIZE + 4), "the newest event must survive"
    assert survivors[0] != "0", "the oldest backlogged events must be dropped"


def test_push_failure_is_logged_once_and_does_not_raise(monkeypatch, caplog):
    attempts = {"n": 0}

    def fake_urlopen(req, timeout=None):
        attempts["n"] += 1
        raise OSError("connection refused")

    monkeypatch.setattr("demo.display.remote.urllib.request.urlopen", fake_urlopen)
    client = RemoteDisplayClient("mac.local", 8080)
    try:
        with caplog.at_level(logging.WARNING):
            client.on_heard("a")
            client.on_heard("b")
            client.on_heard("c")
            assert _wait_until(lambda: attempts["n"] == 3)
    finally:
        client.close()
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1, "repeated failures must log the transition once, not every call"


def test_push_recovers_after_a_failure_and_logs_recovery(monkeypatch, caplog):
    attempts = {"n": 0}

    def fake_urlopen(req, timeout=None):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise OSError("connection refused")
        return _FakeResponse()

    monkeypatch.setattr("demo.display.remote.urllib.request.urlopen", fake_urlopen)
    client = RemoteDisplayClient("mac.local", 8080)
    try:
        with caplog.at_level(logging.INFO):
            client.on_heard("a")
            client.on_heard("b")
            assert _wait_until(lambda: attempts["n"] == 2)
    finally:
        client.close()
    assert any("recovered" in r.message for r in caplog.records)


def test_close_stops_the_worker_thread():
    client = RemoteDisplayClient("mac.local", 8080)
    client.close()
    assert not client._thread.is_alive()


class _FakeControlResponse:
    def __init__(self, body):
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_is_paused_asks_the_dashboard(monkeypatch):
    seen = []

    def fake_urlopen(url, timeout=None):
        seen.append(url)
        return _FakeControlResponse(b'{"paused": true}')

    monkeypatch.setattr("demo.display.remote.urllib.request.urlopen", fake_urlopen)
    client = RemoteDisplayClient("mac.local", 8091)
    try:
        assert client.is_paused() is True
        assert seen == ["http://mac.local:8091/control"]
    finally:
        client.close()


def test_the_restart_request_is_read_from_the_same_endpoint(monkeypatch):
    seen = []

    def fake_urlopen(url, timeout=None):
        seen.append(url)
        return _FakeControlResponse(b'{"paused": false, "restart": true}')

    monkeypatch.setattr("demo.display.remote.urllib.request.urlopen", fake_urlopen)
    client = RemoteDisplayClient("mac.local", 8091)
    try:
        assert client.restart_requested() is True
        assert seen == ["http://mac.local:8091/control"]
    finally:
        client.close()


def test_the_robot_acknowledges_the_restart_before_it_goes(monkeypatch):
    """Sent synchronously, not through the push queue: this process is about
    to end, and a queued event would die with it."""
    posted = []

    def fake_urlopen(request, timeout=None):
        posted.append((request.full_url, request.method, json.loads(request.data)))
        return _FakeControlResponse(b"{}")

    monkeypatch.setattr("demo.display.remote.urllib.request.urlopen", fake_urlopen)
    client = RemoteDisplayClient("mac.local", 8091)
    try:
        client.ack_restart()
    finally:
        client.close()
    assert posted == [("http://mac.local:8091/control", "POST",
                       {"restart_ack": True})]


def test_a_dashboard_that_cannot_be_reached_never_restarts_the_robot(monkeypatch):
    """A screen that is gone must not be read as "wipe everything you know"."""
    def refused(url, timeout=None):
        raise OSError("connection refused")

    monkeypatch.setattr("demo.display.remote.urllib.request.urlopen", refused)
    client = RemoteDisplayClient("mac.local", 8091)
    try:
        assert client.restart_requested() is False
        client.ack_restart()   # logs, does not raise
    finally:
        client.close()


def test_a_dashboard_that_cannot_be_reached_never_pauses_the_robot(monkeypatch):
    def refused(url, timeout=None):
        raise OSError("connection refused")

    monkeypatch.setattr("demo.display.remote.urllib.request.urlopen", refused)
    client = RemoteDisplayClient("mac.local", 8091)
    try:
        assert client.is_paused() is False
    finally:
        client.close()


# — the Stream button, read by demo/stage.py rather than by the robot —

def test_the_stream_request_is_read_from_the_same_control_endpoint(monkeypatch):
    seen = []

    def fake_urlopen(url, timeout=None):
        seen.append(url)
        return _FakeControlResponse(b'{"paused": false, "restart": false, '
                                    b'"stream": true, "streaming": false}')

    monkeypatch.setattr("demo.display.remote.urllib.request.urlopen", fake_urlopen)
    assert stream_request("mac.local", 8091) is True
    assert seen == ["http://mac.local:8091/control"]


def test_stream_off_is_read_as_a_request_not_as_nothing_pending(monkeypatch):
    """False is a request of its own here — it is how the robot is put to
    sleep — and None is the only "nothing to do"."""
    def fake_urlopen(url, timeout=None):
        return _FakeControlResponse(b'{"stream": false, "streaming": true}')

    monkeypatch.setattr("demo.display.remote.urllib.request.urlopen", fake_urlopen)
    assert stream_request("mac.local", 8091) is False

    def nothing(url, timeout=None):
        return _FakeControlResponse(b'{"stream": null, "streaming": true}')

    monkeypatch.setattr("demo.display.remote.urllib.request.urlopen", nothing)
    assert stream_request("mac.local", 8091) is None


def test_a_dashboard_that_cannot_be_reached_asks_for_nothing(monkeypatch):
    """A screen that has gone away must never read as "put the robot to sleep"
    halfway through a demo — the same rule the restart follows."""
    def refused(url, timeout=None):
        raise OSError("connection refused")

    monkeypatch.setattr("demo.display.remote.urllib.request.urlopen", refused)
    assert stream_request("mac.local", 8091) is None


def test_the_stage_acknowledges_with_the_state_the_robot_reached(monkeypatch):
    posted = []

    def fake_urlopen(request, timeout=None):
        posted.append((request.full_url, request.method, json.loads(request.data)))
        return _FakeControlResponse(b"{}")

    monkeypatch.setattr("demo.display.remote.urllib.request.urlopen", fake_urlopen)
    ack_stream("mac.local", 8091, True)
    ack_stream("mac.local", 8091, False)
    assert posted == [("http://mac.local:8091/control", "POST",
                       {"stream_ack": True}),
                      ("http://mac.local:8091/control", "POST",
                       {"stream_ack": False})]


def test_an_acknowledgement_that_cannot_be_delivered_does_not_raise(monkeypatch, caplog):
    """The robot is up either way; the caller only acts on a state it is not
    already in, so an unacknowledged request costs another ack, not a second
    wake-up in front of the room."""
    def refused(request, timeout=None):
        raise OSError("connection refused")

    monkeypatch.setattr("demo.display.remote.urllib.request.urlopen", refused)
    with caplog.at_level(logging.WARNING, logger="demo.display.remote"):
        ack_stream("mac.local", 8091, True)
    assert "stream ack failed" in caplog.text
