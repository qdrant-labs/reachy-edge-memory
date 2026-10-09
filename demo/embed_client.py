"""Vectors from the Mac's embed_service (demo/embed_service.py), for a robot
that carries no models of its own.

Two shapes, because the two memories consume two different interfaces (see
the module docstrings of emulator/frame_memory.py and emulator/memory.py):
`RemoteSiglipEmbedder` satisfies FrameMemory's `Embedder` protocol
(`embed_image`); `RemoteBgeEmbedder` satisfies the fastembed
`embed()`/`query_embed()` shape TextMemory uses. Both send bytes over HTTP and
get a vector back — constructing either loads nothing, which is the whole
point: a memory built on the robot must be constructible without the models
it used to compute its own vectors with (see tests/test_embed_client.py's
import-isolation test).

Deliberately NOT resilient the way demo/detect_source.py's RemoteDetectSource or
demo/platform/robot_camera.py's RobotCameraSource are: those degrade
gracefully because a dropped video frame just costs one detection or one
displayed frame. A memory write that silently no-ops costs the demo its
premise — an "I remember" that quietly isn't true — so every failure here (a
refused connection, a timeout, a malformed response) is left to propagate.
The callers already guard the turn: demo/conversation.py turns a failed
search into MemoryUnavailable, which the model is told as such, and a failed
store keeps the exchanges in the window to be tried again.

Host/port/paths come from demo.embed_service rather than being restated: a
client and its service in this repo once disagreed over a port and a path
with both test suites green over a pipeline that 404'd on every request.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import time
import urllib.request
from typing import TYPE_CHECKING

import numpy as np

from demo.embed_service import (DEFAULT_PORT, FACE_PATH, HEALTH_PATH,
                                IMAGE_PATH, SPEECH_PATH)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# A frame vector call is dominated by SigLIP inference on the Mac (~36ms,
# measured — see emulator/frame_memory.py) plus a JPEG upload; a stalled peer
# (Mac mid-model-load, a TCP blackhole on conference WiFi) should look like a
# failure quickly, not 15s later. Matches RemoteDetectSource's own budget for
# the same shape of call (demo/detect_source.py's `_post_detect`,
# urlopen(timeout=5)) — a JPEG upload followed by inference and a small JSON
# reply. A turn's memory search and the scene writer's stores call into this
# with no timeout budget of their own — this constant alone bounds how long a
# stalled embed_service can hold up a single call.
DEFAULT_TIMEOUT_S = 5.0

# Mirrors HttpReachyRobot's breaker (demo/robot_reachy.py): after a short run
# of failures, stop spending a full timeout on every subsequent call and fail
# immediately instead, until the cooldown elapses. Without this, a dropped
# Mac costs DEFAULT_TIMEOUT_S on EVERY memory call for the rest of the
# outage, rather than once per cooldown window.
CONSECUTIVE_FAILURES_BEFORE_COOLDOWN = 3
FAILURE_COOLDOWN_S = 5.0

# JPEG quality for the copy UPLOADED to embed_service — distinct from
# FrameMemory.JPEG_QUALITY (the quality of what's actually STORED and shown
# on the dashboard): this copy is discarded the instant a vector comes back,
# so it can be lower without the audience ever seeing the loss.
UPLOAD_JPEG_QUALITY = 80


def embed_service_url(host: str, port: int = DEFAULT_PORT) -> str:
    """The one place a client assembles embed_service's base URL."""
    return f"http://{host}:{port}"


class _FailureBreaker:
    """Per-embedder cooldown bookkeeping, one instance per
    RemoteSiglipEmbedder/RemoteBgeEmbedder (each built once in
    demo/run_demo.py's build_memories) — mirrors HttpReachyRobot's own
    breaker (demo/robot_reachy.py) rather than sharing one across unrelated
    embedders/hosts.
    """

    def __init__(self) -> None:
        self._consecutive_failures = 0
        self._cooldown_until = 0.0  # time.monotonic() deadline; 0 == not tripped

    def check(self, base_url: str) -> None:
        now = time.monotonic()
        if now < self._cooldown_until:
            raise TimeoutError(
                f"{base_url}: skipping call, {self._cooldown_until - now:.1f}s "
                "left in failure cooldown")

    def record_success(self) -> None:
        self._consecutive_failures = 0
        self._cooldown_until = 0.0

    def record_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= CONSECUTIVE_FAILURES_BEFORE_COOLDOWN:
            self._cooldown_until = time.monotonic() + FAILURE_COOLDOWN_S


def _post(base_url: str, path: str, body: dict, timeout: float,
          breaker: "_FailureBreaker") -> dict:
    breaker.check(base_url)
    request = urllib.request.Request(
        base_url + path, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            result = json.loads(resp.read() or b"{}")
    except OSError:
        # OSError covers both urllib.error.URLError (connection refused, DNS
        # failure) and a socket timeout — both mean "the peer isn't answering
        # right now", exactly what should trip the breaker. A malformed
        # response (bad JSON/missing key) is a different kind of bug and must
        # NOT count toward it — same reasoning as HttpReachyRobot._call.
        breaker.record_failure()
        raise
    breaker.record_success()
    return result


class RemoteSiglipEmbedder:
    """SigLIP over HTTP — satisfies frame_memory.Embedder without importing
    onnxruntime.

    FrameMemory sizes its shard by calling `embed_image` once at
    construction (see its docstring), so even that probe round-trips to the
    Mac: a robot with no path to embed_service fails to CONSTRUCT FrameMemory,
    rather than starting up and silently being unable to store or recall
    anything.
    """

    def __init__(self, host: str, port: int = DEFAULT_PORT, *,
                 timeout: float = DEFAULT_TIMEOUT_S) -> None:
        self._base = embed_service_url(host, port)
        self._timeout = timeout
        self._breaker = _FailureBreaker()

    def embed_image(self, frame_rgb: "NDArray[np.uint8]") -> "NDArray[np.float32]":
        from PIL import Image

        buf = io.BytesIO()
        Image.fromarray(np.asarray(frame_rgb, dtype=np.uint8)).convert("RGB").save(
            buf, format="JPEG", quality=UPLOAD_JPEG_QUALITY)
        jpeg_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        body = _post(self._base, IMAGE_PATH, {"jpeg_b64": jpeg_b64}, self._timeout,
                     self._breaker)
        return np.asarray(body["vector"], dtype=np.float32)


def faces_off(host: str, port: int = DEFAULT_PORT, *,
              timeout: float = DEFAULT_TIMEOUT_S) -> str | None:
    """Why the embed service has no face models, from its /health — None
    when it has them, or cannot be asked (then the first face says why)."""
    try:
        with urllib.request.urlopen(embed_service_url(host, port) + HEALTH_PATH,
                                    timeout=timeout) as resp:
            return json.loads(resp.read()).get("faces_off")
    except (OSError, ValueError) as exc:
        logger.warning("embed_service health: %s", exc)
        return None


class RemoteFaceReader:
    """Faces over HTTP — the robot sends a frame, the Mac sends back a box and
    an identity vector per face (demo/embed_service.py's FACE_PATH).

    Only the reading is remote. Which person that vector IS, and their name,
    is decided and stored on the robot (emulator/face_memory.py) — the Mac
    never sees a name and keeps nothing.
    """

    def __init__(self, host: str, port: int = DEFAULT_PORT, *,
                 timeout: float = DEFAULT_TIMEOUT_S) -> None:
        self._base = embed_service_url(host, port)
        self._timeout = timeout
        self._breaker = _FailureBreaker()

    def read(self, frame_rgb: "NDArray[np.uint8]", *, embed: bool = True) -> list[dict]:
        """Faces in this frame, largest first: `{box, score, embedding}` with
        the box in frame fractions."""
        from PIL import Image

        buf = io.BytesIO()
        Image.fromarray(np.asarray(frame_rgb, dtype=np.uint8)).convert("RGB").save(
            buf, format="JPEG", quality=UPLOAD_JPEG_QUALITY)
        body = _post(self._base, FACE_PATH,
                     {"jpeg_b64": base64.b64encode(buf.getvalue()).decode("ascii"),
                      "embed": embed},
                     self._timeout, self._breaker)
        return list(body.get("faces", []))


class RemoteBgeEmbedder:
    """bge over HTTP — satisfies the fastembed `embed()`/`query_embed()`
    shape TextMemory uses, without importing fastembed.

    The query/document split is not cosmetic (see demo/embed_service.py's
    `_embed_speech`): bge trains the two sides differently, and mixing them
    degrades recall. embed_service picks the side by a `query` flag in the
    request body — `embed()` must never set it, `query_embed()` always must.
    """

    def __init__(self, host: str, port: int = DEFAULT_PORT, *,
                 timeout: float = DEFAULT_TIMEOUT_S) -> None:
        self._base = embed_service_url(host, port)
        self._timeout = timeout
        self._breaker = _FailureBreaker()

    def embed(self, texts: list[str]) -> "Iterable[NDArray[np.float32]]":
        """Document-side vectors — what TextMemory stores."""
        return [self._embed_one(text, query=False) for text in texts]

    def query_embed(self, texts: list[str]) -> "Iterable[NDArray[np.float32]]":
        """Query-side vectors — what TextMemory searches with."""
        return [self._embed_one(text, query=True) for text in texts]

    def _embed_one(self, text: str, *, query: bool) -> "NDArray[np.float32]":
        body: dict = {"text": text}
        if query:
            body["query"] = True
        response = _post(self._base, SPEECH_PATH, body, self._timeout, self._breaker)
        return np.asarray(response["vector"], dtype=np.float32)
