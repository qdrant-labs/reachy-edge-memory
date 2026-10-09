"""RemoteSiglipEmbedder/RemoteBgeEmbedder: vectors from demo/embed_service.py
over HTTP, for a robot that carries no models of its own.

No real network in these tests: urllib.request.urlopen is monkeypatched,
mirroring tests/test_robot_reachy.py's HttpReachyRobot tests.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pytest

from demo.embed_service import IMAGE_PATH, SPEECH_PATH
from demo.embed_client import (
    RemoteBgeEmbedder,
    RemoteSiglipEmbedder,
    embed_service_url,
)


class _FakeResponse:
    def __init__(self, body: bytes = b"{}"):
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _vector_urlopen(vector: list[float], captured: dict):
    """A fake urlopen that always answers {"vector": ...} and records the
    request it was called with, for assertions on path/body."""

    def fake_urlopen(request, timeout=None):
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data)
        captured["timeout"] = timeout
        return _FakeResponse(json.dumps({"vector": vector}).encode())

    return fake_urlopen


# --- embed_service_url: the one place a base URL is assembled ---

def test_embed_service_url_defaults_to_embed_services_port():
    from demo.embed_service import DEFAULT_PORT

    assert embed_service_url("10.0.0.5") == f"http://10.0.0.5:{DEFAULT_PORT}"


# --- RemoteSiglipEmbedder: embed_image ---

_FRAME = np.zeros((4, 4, 3), dtype=np.uint8)

def test_embed_image_posts_a_base64_jpeg_to_the_image_path(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr("urllib.request.urlopen",
                        _vector_urlopen([1.0, 0.0, 0.0], captured))
    embedder = RemoteSiglipEmbedder("10.0.0.5")
    frame = np.zeros((4, 4, 3), dtype=np.uint8)

    vector = embedder.embed_image(frame)

    assert captured["url"] == embed_service_url("10.0.0.5") + IMAGE_PATH
    assert "jpeg_b64" in captured["body"]
    assert isinstance(vector, np.ndarray)
    assert vector.dtype == np.float32
    assert vector.tolist() == [1.0, 0.0, 0.0]


def test_siglip_embedder_failure_raises_rather_than_returning_a_vector(monkeypatch):
    # A camera drop-out costs a frame; a memory that swallows a failed write
    # costs the demo its premise (see the module docstring) — the client must
    # never turn a connection failure into a fake vector.
    def failing_urlopen(request, timeout=None):
        raise OSError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", failing_urlopen)
    embedder = RemoteSiglipEmbedder("10.0.0.5")
    with pytest.raises(OSError):
        embedder.embed_image(_FRAME)


# --- RemoteBgeEmbedder: embed() (document side) vs query_embed() (query side) ---

def test_embed_does_not_set_the_query_flag(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr("urllib.request.urlopen",
                        _vector_urlopen([1.0, 0.0], captured))
    embedder = RemoteBgeEmbedder("10.0.0.5")

    vectors = list(embedder.embed(["a mug"]))

    assert captured["url"] == embed_service_url("10.0.0.5") + SPEECH_PATH
    assert captured["body"] == {"text": "a mug"}
    assert len(vectors) == 1
    assert vectors[0].tolist() == [1.0, 0.0]


def test_query_embed_sets_the_query_flag(monkeypatch):
    # Not cosmetic: bge trains the query and document sides differently, and
    # embed_service picks the side by this flag (see demo/embed_service.py's
    # _embed_speech) — mixing them up quietly degrades recall.
    captured: dict = {}
    monkeypatch.setattr("urllib.request.urlopen",
                        _vector_urlopen([0.0, 1.0], captured))
    embedder = RemoteBgeEmbedder("10.0.0.5")

    list(embedder.query_embed(["what could I drink from?"]))

    assert captured["body"] == {"text": "what could I drink from?", "query": True}


def test_embed_and_query_embed_handle_multiple_texts(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr("urllib.request.urlopen",
                        _vector_urlopen([1.0], captured))
    embedder = RemoteBgeEmbedder("10.0.0.5")

    vectors = list(embedder.embed(["a", "b", "c"]))
    assert len(vectors) == 3


def test_bge_embedder_failure_raises(monkeypatch):
    def failing_urlopen(request, timeout=None):
        raise OSError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", failing_urlopen)
    embedder = RemoteBgeEmbedder("10.0.0.5")
    with pytest.raises(OSError):
        list(embedder.embed(["hello"]))


# --- timeout + failure cooldown: DEFAULT_TIMEOUT_S used to be
# 15s with no breaker, and demo/run_demo.py's _handle_stream calls into this
# up to FOUR times per turn — a stalled (not refused) embed_service could
# silence the robot for up to a minute, repeating on every turn. Mirrors
# tests/test_robot_reachy.py's HttpReachyRobot tests, which cover the same
# shape of fix there.

def test_default_timeout_is_short(monkeypatch):
    # Matches RemoteDetectSource's own budget for the same shape of call (JPEG
    # upload + inference + small JSON reply) — see the module's own comment.
    from demo.embed_client import DEFAULT_TIMEOUT_S

    assert DEFAULT_TIMEOUT_S <= 5.0

    captured: dict = {}
    monkeypatch.setattr("urllib.request.urlopen",
                        _vector_urlopen([1.0, 0.0, 0.0], captured))
    RemoteSiglipEmbedder("10.0.0.5").embed_image(_FRAME)

    assert captured["timeout"] == DEFAULT_TIMEOUT_S


def test_siglip_embedder_trips_cooldown_after_consecutive_failures(monkeypatch):
    # After N failures in a row, further calls must not touch the network at
    # all — they should raise immediately instead of paying another timeout.
    from demo.embed_client import CONSECUTIVE_FAILURES_BEFORE_COOLDOWN

    calls = {"n": 0}

    def failing_urlopen(request, timeout=None):
        calls["n"] += 1
        raise OSError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", failing_urlopen)
    embedder = RemoteSiglipEmbedder("10.0.0.5")
    for _ in range(CONSECUTIVE_FAILURES_BEFORE_COOLDOWN):
        with pytest.raises(OSError):
            embedder.embed_image(_FRAME)
    assert calls["n"] == CONSECUTIVE_FAILURES_BEFORE_COOLDOWN

    with pytest.raises(TimeoutError):
        embedder.embed_image(_FRAME)
    assert calls["n"] == CONSECUTIVE_FAILURES_BEFORE_COOLDOWN, (
        "a call made during the cooldown window must not hit the network")


def test_siglip_embedder_cooldown_recovers_automatically(monkeypatch):
    from demo.embed_client import (
        CONSECUTIVE_FAILURES_BEFORE_COOLDOWN, FAILURE_COOLDOWN_S)

    clock = {"t": 0.0}
    monkeypatch.setattr("time.monotonic", lambda: clock["t"])

    def failing_urlopen(request, timeout=None):
        raise OSError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", failing_urlopen)
    embedder = RemoteSiglipEmbedder("10.0.0.5")
    for _ in range(CONSECUTIVE_FAILURES_BEFORE_COOLDOWN):
        with pytest.raises(OSError):
            embedder.embed_image(_FRAME)

    with pytest.raises(TimeoutError):
        embedder.embed_image(_FRAME)  # still inside the cooldown window

    clock["t"] += FAILURE_COOLDOWN_S + 0.01
    monkeypatch.setattr("urllib.request.urlopen",
                        _vector_urlopen([1.0, 0.0, 0.0], {}))
    vector = embedder.embed_image(_FRAME)  # cooldown elapsed — hits the network again
    assert vector.tolist() == [1.0, 0.0, 0.0]


def test_bge_embedder_trips_cooldown_after_consecutive_failures(monkeypatch):
    from demo.embed_client import CONSECUTIVE_FAILURES_BEFORE_COOLDOWN

    calls = {"n": 0}

    def failing_urlopen(request, timeout=None):
        calls["n"] += 1
        raise OSError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", failing_urlopen)
    embedder = RemoteBgeEmbedder("10.0.0.5")
    for _ in range(CONSECUTIVE_FAILURES_BEFORE_COOLDOWN):
        with pytest.raises(OSError):
            list(embedder.embed(["hello"]))
    assert calls["n"] == CONSECUTIVE_FAILURES_BEFORE_COOLDOWN

    with pytest.raises(TimeoutError):
        list(embedder.embed(["hello"]))
    assert calls["n"] == CONSECUTIVE_FAILURES_BEFORE_COOLDOWN, (
        "a call made during the cooldown window must not hit the network")


def test_siglip_and_bge_embedders_have_independent_cooldowns(monkeypatch):
    # Each embedder is a separate peer connection in demo/run_demo.py's
    # build_memories (one per memory) — one tripping its breaker must not
    # affect the other's.
    from demo.embed_client import CONSECUTIVE_FAILURES_BEFORE_COOLDOWN

    def failing_urlopen(request, timeout=None):
        raise OSError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", failing_urlopen)
    siglip = RemoteSiglipEmbedder("10.0.0.5")
    bge = RemoteBgeEmbedder("10.0.0.5")
    for _ in range(CONSECUTIVE_FAILURES_BEFORE_COOLDOWN):
        with pytest.raises(OSError):
            siglip.embed_image(_FRAME)
    with pytest.raises(TimeoutError):
        siglip.embed_image(_FRAME)

    # bge's own breaker hasn't tripped — it still hits the (failing) network
    # and raises the original OSError, not a cooldown TimeoutError.
    with pytest.raises(OSError):
        list(bge.embed(["hello"]))


# --- the point: a memory built with a remote embedder loads no
# model locally (emulator/frame_memory.py's SiglipEmbedder and
# emulator/memory.py's fastembed TextEmbedding are both imported lazily
# today; an injected embedder must never trigger either import) ---

def test_frame_memory_with_a_remote_embedder_never_imports_onnxruntime(monkeypatch):
    from emulator.frame_memory import FrameMemory

    captured: dict = {}
    monkeypatch.setattr("urllib.request.urlopen",
                        _vector_urlopen([1.0, 0.0, 0.0], captured))
    # Clear any prior import (from another test, another process init) so a
    # buggy FrameMemory that DOES import onnxruntime is caught here rather
    # than hidden behind an import some earlier test already paid for.
    monkeypatch.delitem(sys.modules, "onnxruntime", raising=False)

    FrameMemory(embedder=RemoteSiglipEmbedder("10.0.0.5"))

    assert "onnxruntime" not in sys.modules


def test_text_memory_with_a_remote_embedder_never_imports_fastembed(monkeypatch):
    from emulator.memory import TextMemory

    captured: dict = {}
    monkeypatch.setattr("urllib.request.urlopen",
                        _vector_urlopen([1.0, 0.0], captured))
    monkeypatch.delitem(sys.modules, "fastembed", raising=False)

    TextMemory(embedder=RemoteBgeEmbedder("10.0.0.5"))

    assert "fastembed" not in sys.modules


def test_faces_off_reads_why_from_the_services_health(monkeypatch):
    from demo import embed_client

    seen = {}

    def health(url, timeout=None):
        seen["url"] = url
        return _FakeResponse(json.dumps({"healthy": True, "faces_off": "no model"}).encode())

    monkeypatch.setattr(embed_client.urllib.request, "urlopen", health)
    assert embed_client.faces_off("mac", 9700) == "no model"
    assert seen["url"] == "http://mac:9700/health"
    monkeypatch.setattr(embed_client.urllib.request, "urlopen",
                        lambda url, timeout=None: _FakeResponse(b'{"healthy": true}'))
    assert embed_client.faces_off("mac") is None


def test_a_service_that_cannot_say_leaves_faces_on(monkeypatch):
    # The first face request then fails loudly, through the breaker — not here.
    import urllib.error

    from demo import embed_client

    def unreachable(url, timeout=None):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(embed_client.urllib.request, "urlopen", unreachable)
    assert embed_client.faces_off("mac") is None
