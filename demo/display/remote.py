"""RemoteDisplayClient: push voice-loop events to a WebDashboard running on a
different machine, over HTTP — the "remote" `--display` kind (see
demo/display/__init__.py's build_display).

Why this exists: the voice loop, its memories, and its camera/mic/motors now
run ON the robot, but the architecture keeps
the audience's screen on the Mac (already in front of the room, costs the
CM4 nothing) — so a DisplaySink's events (heard text, replies, recalled
frames) have to LEAVE this process instead of being broadcast in-process the
way demo/display/web.py's WebDashboard does for the older, local/debug
`--display web` mode. The receiving side is that same WebDashboard, run
standalone on the Mac (`python -m demo.display.web`) with its own `/push`
endpoint (see that module) accepting exactly the (event, data) pairs sent
here, and re-broadcasting them over SSE precisely as if they had been
generated in-process.

The dashboard is DECORATIVE, not load-bearing — killing the Mac's dashboard
must not stop the robot remembering — so every push here is fire-and-forget off a background worker thread reading
a bounded, drop-oldest queue: a slow or dead Mac must never stall the voice
loop's own turn (VAD, recall, speech), the same reasoning demo/run_demo.py's
RobotDispatcher already applies to robot motor calls, and WebDashboard's own
`_broadcast` applies to its SSE subscriber queues. A sustained outage
degrades to "the dashboard falls behind, then catches up" — never "the queue
grows without bound" or "a turn waits on a network call to a screen no one
needs for the demo to work".

The same dashboard also holds the three buttons, on its `/control` endpoint:
the pause and the restart are read from here by the voice loop (the methods
below), and the Stream button is read by demo/stage.py through the two module
functions — one place assembles this dashboard's URLs, whichever process is
asking.
"""
from __future__ import annotations

import base64
import json
import logging
import queue
import threading
import urllib.request


from demo.display.web import CONTROL_PATH, PUSH_PATH

LOG = logging.getLogger(__name__)

# One HTTP POST at a time off the worker thread; short enough that a dead Mac
# fails fast and the worker moves on to the next queued event rather than
# holding up the whole backlog on one hung call.
DEFAULT_TIMEOUT_S = 2.0

# Bounded so a sustained outage can't grow memory without bound — mirrors
# WebDashboard.SUB_QUEUE_MAXSIZE (demo/display/web.py), same "drop the
# oldest, keep pushing the newest" shape.
QUEUE_MAXSIZE = 64


def dashboard_push_url(host: str, port: int) -> str:
    """The one place a client assembles the dashboard's push URL."""
    return f"http://{host}:{port}{PUSH_PATH}"


def dashboard_control_url(host: str, port: int) -> str:
    return f"http://{host}:{port}{CONTROL_PATH}"


def stream_request(host: str, port: int, timeout: float = 1.0) -> bool | None:
    """What the dashboard's Stream button is asking for: True to bring the
    robot up, False to put it to sleep, None for nothing pending.

    A module function rather than a RemoteDisplayClient method because the
    reader is demo/stage.py on the MAC — it is not a DisplaySink and pushes
    nothing, it only consumes the one flag the page cannot act on itself (the
    dashboard cannot reach the robot; the stage is the process that runs
    scripts/robot_service.sh). The URL is still assembled here, beside the
    client that shares this endpoint, so the two cannot drift apart.

    A dashboard that cannot be reached asks for NOTHING, the same rule
    restart_requested already follows: a screen that has gone away must never
    read as "put the robot to sleep" halfway through a demo.
    """
    try:
        with urllib.request.urlopen(dashboard_control_url(host, port),
                                    timeout=timeout) as response:
            asked = json.loads(response.read()).get("stream")
    except Exception as exc:  # noqa: BLE001 — the screen must not drive the robot
        LOG.debug("stream request check failed: %s: %s", type(exc).__name__, exc)
        return None
    return None if asked is None else bool(asked)


def ack_stream(host: str, port: int, streaming: bool,
               timeout: float = 2.0) -> None:
    """Tell the dashboard the request is spent and what state the robot ended
    up in — which is not always what was asked for, so the button can go back
    to grey instead of green when the camera service refused to start.

    Failing to land it is not fatal: the caller (demo/stage.py's
    StreamWatcher) only ACTS on a state it is not already in, so an
    unacknowledged request costs another ack on the next poll, never a second
    wake-up in front of the room."""
    body = json.dumps({"stream_ack": bool(streaming)}).encode()
    request = urllib.request.Request(
        dashboard_control_url(host, port), data=body, method="POST",
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout):
            pass
    except Exception as exc:  # noqa: BLE001 — the robot is up either way
        LOG.warning("stream ack failed (%s: %s) — the dashboard still shows "
                    "the previous state", type(exc).__name__, exc)


class RemoteDisplayClient:
    """DisplaySink that POSTs each event to a WebDashboard's `/push`
    endpoint instead of broadcasting it in-process. Satisfies
    demo/display/__init__.py's DisplaySink protocol structurally — no
    inheritance needed.
    """

    def __init__(self, host: str, port: int, *,
                timeout: float = DEFAULT_TIMEOUT_S) -> None:
        self._url = dashboard_push_url(host, port)
        self._control_url = dashboard_control_url(host, port)
        self._timeout = timeout
        self._queue: "queue.Queue[tuple[str, dict]]" = queue.Queue(maxsize=QUEUE_MAXSIZE)
        self._stop = threading.Event()
        self._consecutive_failures = 0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _enqueue(self, event: str, data: dict) -> None:
        item = (event, data)
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            # Drop the oldest queued push, not the newest — a stale
            # "detections" event is worthless once a fresher one exists, same
            # reasoning as WebDashboard._broadcast's own drop-oldest queues.
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(item)
            except queue.Full:
                pass  # lost a race with another producer — one event, harmless

    def _run(self) -> None:
        while True:
            try:
                event, data = self._queue.get(timeout=0.2)
            except queue.Empty:
                # Bounded wait, not an indefinite block: close() must still be
                # able to join this thread promptly once nothing is queued.
                if self._stop.is_set():
                    return
                continue
            self._push(event, data)

    def _push(self, event: str, data: dict) -> None:
        body = json.dumps({"event": event, "data": data}).encode()
        req = urllib.request.Request(
            self._url, data=body,
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            urllib.request.urlopen(req, timeout=self._timeout).close()
        except Exception as exc:  # noqa: BLE001 — see class docstring: decorative, never fatal
            # Log the transition only, not every failure: a refused
            # connection fails in milliseconds, so an unguarded warning here
            # would bury real output for as long as the Mac dashboard is down
            # (same shape as RemoteDetectSource/RobotCameraSource's own
            # transition-only logging).
            if self._consecutive_failures == 0:
                LOG.warning("dashboard push failed: %s: %s (further "
                           "failures silent until it recovers)",
                           type(exc).__name__, exc)
            self._consecutive_failures += 1
            return
        if self._consecutive_failures:
            LOG.info("dashboard push recovered after %d failed attempt(s)",
                     self._consecutive_failures)
        self._consecutive_failures = 0

    # — DisplaySink —
    def on_detections(self, detections: list[dict]) -> None:
        self._enqueue("detections", {"boxes": detections})

    def on_heard(self, text: str) -> None:
        self._enqueue("heard", {"text": text})

    def on_reply(self, text: str, done: bool = False) -> None:
        self._enqueue("reply", {"text": text, "done": done})

    def on_look(self, jpeg: bytes) -> None:
        self._enqueue("look", {"jpeg_b64": base64.b64encode(jpeg).decode()})

    def on_recall(self, hits: list[dict]) -> None:
        # Same shaping as WebDashboard.on_recall: only what the dashboard
        # actually renders (jpeg + boxes), not the whole FrameMemory hit.
        frames = [{"jpeg_b64": h["jpeg_b64"], "boxes": h.get("detections", []),
                   "score": round(float(h.get("score", 0.0)), 3),
                   "found_by": h.get("found_by"),
                   "weak": bool(h.get("weak"))}
                 for h in hits if h.get("jpeg_b64")]
        self._enqueue("recall", {"frames": frames})

    def on_speech_recall(self, hits: list[dict]) -> None:
        items = [{"text": h["text"], "score": round(float(h.get("score", 0.0)), 3),
                  "source": h.get("source", "")}
                for h in hits if h.get("text")]
        self._enqueue("speech_recall", {"items": items})

    def is_paused(self) -> bool:
        """Ask the dashboard whether the pause is on. Read synchronously, not
        through the push queue — the answer is needed now, before listening.
        A dashboard that cannot be reached is not a reason to stop the robot:
        the turn goes ahead, as it does when the dashboard is missing."""
        try:
            with urllib.request.urlopen(self._control_url, timeout=1.0) as response:
                return bool(json.loads(response.read()).get("paused"))
        except Exception as exc:  # noqa: BLE001 — presentation must not stop the robot
            LOG.debug("pause check failed: %s: %s", type(exc).__name__, exc)
            return False

    def restart_requested(self) -> bool:
        """Whether the dashboard's Restart button is waiting to be acted on.
        Read synchronously like the pause, for the same reason: the answer
        decides what happens next, and an unreachable dashboard is never a
        reason to restart — a failed read says no."""
        try:
            with urllib.request.urlopen(self._control_url, timeout=1.0) as response:
                return bool(json.loads(response.read()).get("restart"))
        except Exception as exc:  # noqa: BLE001 — presentation must not stop the robot
            LOG.debug("restart check failed: %s: %s", type(exc).__name__, exc)
            return False

    def ack_restart(self) -> None:
        """Say the restart is happening, so the loop that comes back does not
        read the same request and restart again. Sent synchronously, not
        through the push queue: this process is about to end, and a queued
        event would die with it."""
        body = json.dumps({"restart_ack": True}).encode()
        request = urllib.request.Request(
            self._control_url, data=body, method="POST",
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=1.0):
                pass
        except Exception as exc:  # noqa: BLE001 — the restart happens regardless
            LOG.warning("restart ack failed (%s: %s) — the loop that comes back "
                        "clears the request itself", type(exc).__name__, exc)

    def on_tool_call(self, name: str, arguments: dict) -> None:
        self._enqueue("tool_call", {"name": name, "arguments": arguments})

    def on_memory_write(self, texts: list[str]) -> None:
        self._enqueue("memory_write", {"items": list(texts)})

    def on_context(self, tokens: int | None, budget: int, exchanges: int) -> None:
        self._enqueue("context", {"tokens": tokens, "budget": budget,
                                  "exchanges": exchanges})

    def on_memory_count(self, frames: int, exchanges: int,
                        knowledge: int = 0) -> None:
        self._enqueue("memory_count", {"frames": frames, "exchanges": exchanges,
                                       "knowledge": knowledge})

    def on_face(self, name: str | None, box: list[float] | None,
                score: float) -> None:
        self._enqueue("face", {"name": name, "box": box,
                               "score": round(float(score), 3)})

    def close(self, timeout: float = 2.0) -> None:
        self._stop.set()
        self._thread.join(timeout=timeout)
