"""Web dashboard: live video + detections + query frame + reply stream.

The presentation half of the former WebUI: serves the MJPEG camera feed, SSE
events, and dashboard.html. Knows nothing about where detections come from —
`on_detections` is just a listener subscribed to RemoteDetectSource, or a
`/push` POST from a `RemoteDisplayClient` running elsewhere (see below).
Boxes are drawn by the BROWSER on a canvas over the video — no CPU here spent
on rendering. A single viewer-presenter.

Two ways this class is used, matching the two `--display` kinds in
demo/run_demo.py:
- `web` (local/debug): built in-process by the voice loop, fed directly by
  method calls — `camera` is whatever VideoSource the loop already has.
- `remote`: the voice loop runs on the robot and pushes its DisplaySink
  events here, over HTTP (demo/display/remote.py's RemoteDisplayClient) —
  this module's own `main()` below runs the dashboard STANDALONE on the
  laptop, reading the camera over HTTP from the robot's camera service
  (RobotCameraSource) and re-broadcasting whatever lands on `/push` as if it
  had been produced in-process.
"""
from __future__ import annotations

import argparse
import base64
import ipaddress
import json
import logging
import queue
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


from demo.detect_source import frame_to_jpeg
from demo.display import DEFAULT_DASHBOARD_PORT
from demo.display.events import sse
from demo.http_util import ascii_reason

LOG = logging.getLogger(__name__)

STATIC = Path(__file__).parent / "static"

# SSE subscriber queues are bounded: if the browser dropped off but events
# keep arriving, we don't buffer forever — we drop the oldest instead
# (drop-oldest), otherwise a slow/dead client's queue would grow unbounded.
SUB_QUEUE_MAXSIZE = 64
# Where the page fetches a recalled frame: /recall/<version>/<index>.jpg
RECALL_PATH = "/recall/"
# Where the page fetches the picture the camera tool gave the model: /look/<version>.jpg
LOOK_PATH = "/look/"

# Where a RemoteDisplayClient (demo/display/remote.py) POSTs pushed events —
# defined here, the service side, and imported by that client, mirroring
# demo/camera_service.py defining DEFAULT_PORT/FRAME_PATH for its own
# clients: one definition, no drift between a client and its service (see
# that module's docstring for what it costs when they disagree).
PUSH_PATH = "/push"

# Where the pause, the restart and the stream live. The buttons are on the
# Mac's screen, the listening happens on the robot, and the robot has no
# inbox — so the dashboard holds the flags and whoever can act asks for them:
# the pause before each turn (demo/run_demo.py's _wait_while_paused), the
# restart from the voice loop's own watcher thread (_RestartWatcher), which
# can act while the robot is standing quiet with nobody talking to it, and
# the stream from demo/stage.py, which is the only process that can start or
# stop the robot at all (it runs scripts/robot_service.sh).
CONTROL_PATH = "/control"

# A pushed event is small JSON except a picture (a look, a recalled frame: a
# JPEG of tens of KB) — generous next to that rather than tuned to it, same reasoning as
# embed_service.py's MAX_BODY: refuse a body that could only be a mistake
# without getting in the way of a real frame.
MAX_PUSH_BODY = 2 * 1024 * 1024


class WebDashboard:
    def __init__(self, camera) -> None:
        self.camera = camera
        self._paused = False
        # A REQUEST the robot consumes, not a counter it compares against.
        # A counter would reset to zero whenever this dashboard restarted,
        # and the robot — holding a baseline of 3 from before — would read
        # the difference as a restart and wipe its memory because someone
        # restarted the Mac's screen. A fresh dashboard simply has nothing
        # pending.
        self._restart_pending = False
        # The Stream button, the same shape for the same reason: a REQUEST
        # ("start it" / "stop it") that demo/stage.py consumes, plus the
        # state it last reported back. None means nothing is pending — a
        # dashboard that has just restarted asks for nothing, rather than
        # putting a robot to sleep in front of the room because the screen
        # was reloaded.
        self._stream_request: bool | None = None
        # None until something that can actually act says otherwise: a
        # dashboard nobody is polling (the in-process `--display web` debug
        # mode, or this module run on its own) would otherwise print "the
        # robot is asleep" over a live camera.
        self._streaming: bool | None = None
        self._subs: list[queue.Queue] = []
        self._subs_lock = threading.Lock()
        self._server: ThreadingHTTPServer | None = None
        # The frames the robot last recalled, kept here and served as plain
        # images (RECALL_PATH) — see _recall.
        self._recall_jpegs: list[bytes] = []
        self._recall_version = 0
        self._recall_lock = threading.Lock()
        self._look_jpeg: bytes | None = None
        self._look_version = 0

    def serve(self, host: str = "127.0.0.1", port: int = 8080) -> ThreadingHTTPServer:
        self._server = ThreadingHTTPServer((host, port), make_handler(self))
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return self._server

    # — DisplaySink —
    def on_detections(self, detections: list[dict]) -> None:
        self._broadcast(sse("detections", {"boxes": detections}))

    def on_recall(self, hits: list[dict]) -> None:
        """Show the frames a query recalled. `hits` are stored frames
        (emulator/frame_memory.py's recall_text, day_frames or a look): each
        carries the stored `jpeg_b64` and the frame's YOLO `detections`
        (drawn as boxes).

        ALWAYS broadcasts, even with zero frames: this used to skip the event
        on an empty result so the panel "kept its last state rather than
        flashing empty" — but on stage that meant the projector kept showing
        the PREVIOUS question's frames while the robot answered a new,
        unrelated one, a confident mismatch worse than a blank panel. An
        explicit empty broadcast lets the dashboard render its own clear
        "nothing recalled" resting state instead of silently going stale."""
        frames = [{"jpeg_b64": h["jpeg_b64"], "boxes": h.get("detections", []),
                   "score": round(float(h.get("score", 0.0)), 3),
                   "weak": bool(h.get("weak"))}
                  for h in hits if h.get("jpeg_b64")]
        self._recall(frames)

    def _recall(self, frames: list[dict]) -> None:
        """Keep the recalled pictures here and send the browser only where to
        fetch them. Live: a recall event carrying three frames as
        base64 (~210 KB) never reached the dashboard page in Chrome — the
        server wrote it to the page's socket in full, a curl reader got it,
        the page's EventSource never fired, and the room saw "nothing
        recalled". Small events arrived throughout. A picture is fetched as a
        picture now, like the live video; the event stays a few hundred bytes."""
        with self._recall_lock:
            self._recall_version += 1
            version = self._recall_version
            self._recall_jpegs = [base64.b64decode(f["jpeg_b64"]) for f in frames]
        light = [{"url": f"{RECALL_PATH}{version}/{i}.jpg",
                  **{k: v for k, v in f.items() if k != "jpeg_b64"}}
                 for i, f in enumerate(frames)]
        self._broadcast(sse("recall", {"frames": light}))

    def on_look(self, jpeg: bytes) -> None:
        """The picture the camera tool gave the model, shown under the tool
        line. Served by URL like a recalled frame (see _recall) — the same
        ~60 KB event that never reached the page in Chrome."""
        with self._recall_lock:
            self._look_version += 1
            self._look_jpeg = jpeg
            version = self._look_version
        self._broadcast(sse("look", {"url": f"{LOOK_PATH}{version}.jpg"}))

    def look_jpeg(self, path: str) -> bytes | None:
        """The JPEG a look event pointed at, if it is still the latest look."""
        try:
            version = int(path[len(LOOK_PATH):].removesuffix(".jpg"))
        except ValueError:
            return None
        with self._recall_lock:
            return self._look_jpeg if version == self._look_version else None

    def recall_jpeg(self, path: str) -> bytes | None:
        """The JPEG a recall event pointed at, if it is still the latest recall."""
        try:
            version, name = path[len(RECALL_PATH):].split("/", 1)
            index = int(name.removesuffix(".jpg"))
            with self._recall_lock:
                if int(version) != self._recall_version:
                    return None
                return self._recall_jpegs[index]
        except (ValueError, IndexError):
            return None

    def push(self, event: str, data: dict) -> None:
        """Re-broadcast an event exactly as if the matching `on_*` method had
        produced it in-process — what the `/push` HTTP endpoint below calls
        for a RemoteDisplayClient (demo/display/remote.py) event. One public
        entry point, rather than the Handler reaching into the private
        `_broadcast` every `on_*` method already uses. A recall keeps its
        pictures here (see _recall)."""
        if event == "recall":
            self._recall(list(data.get("frames") or []))
            return
        if event == "look":
            self.on_look(base64.b64decode(data.get("jpeg_b64") or ""))
            return
        self._broadcast(sse(event, data))

    def on_speech_recall(self, hits: list[dict]) -> None:
        """Show past utterances a query recalled (TextMemory.recall hits) —
        the "search by what was said" surface. Same reasoning as on_recall:
        always broadcasts, including empty, so the panel can't show a stale
        answer to a question that's no longer being asked."""
        items = [{"text": h["text"], "score": round(float(h.get("score", 0.0)), 3),
                  "source": h.get("source", "")}
                 for h in hits if h.get("text")]
        self._broadcast(sse("speech_recall", {"items": items}))

    def on_tool_call(self, name: str, arguments: dict) -> None:
        """Shown so the room sees the model DECIDE to remember or to look —
        not only what came back."""
        self._broadcast(sse("tool_call", {"name": name, "arguments": arguments}))

    def on_memory_write(self, texts: list[str]) -> None:
        self._broadcast(sse("memory_write", {"items": list(texts)}))

    def is_paused(self) -> bool:
        """Whether the pause is on — read by the voice loop before it listens."""
        return self._paused

    def control_state(self) -> dict:
        """What every button on the page stands at, in one shape: answered on
        GET /control, returned from every POST, and broadcast on any change.
        One payload rather than one per button — a page that reloads mid-demo,
        and a second screen at the back of the room, then show the same three
        buttons as the screen that pressed them."""
        return {"paused": self._paused, "restart": self._restart_pending,
                "stream": self._stream_request, "streaming": self._streaming}

    def set_paused(self, paused: bool) -> None:
        """Turn the pause on or off and tell every open page, so a second
        screen does not show the opposite of what the robot is doing."""
        self._paused = bool(paused)
        self._broadcast(sse("control", self.control_state()))

    def restart_requested(self) -> bool:
        """Whether the Restart button has been pressed and not yet acted on —
        polled by the robot's voice loop."""
        return self._restart_pending

    def request_restart(self) -> None:
        """The Restart button: ask the robot to wipe its memory and come back.

        The erasing happens on the robot (scripts/voice_loop.sh, around
        demo/run_demo.py's RESTART_EXIT_CODE) — this side only asks, and
        clears the screen so the panels do not describe a conversation that
        no longer exists. The pause goes with it: a robot that comes back from
        a restart should be listening, not waiting on a button pressed before
        the wipe."""
        self._restart_pending = True
        self._paused = False
        self._broadcast(sse("control", self.control_state()))
        self._broadcast(sse("reset", {}))

    def clear_restart(self) -> None:
        """The robot acknowledging the request — it is restarting now. Without
        this the freshly started loop would read the same flag and restart
        again, forever."""
        self._restart_pending = False

    def stream_requested(self) -> bool | None:
        """What the Stream button is asking for: True to bring the robot up,
        False to put it to sleep, None for nothing pending. Polled by
        demo/stage.py, which is the process that can actually do it — this
        dashboard cannot reach the robot at all."""
        return self._stream_request

    def is_streaming(self) -> bool | None:
        """Whether the robot is up, as demo/stage.py last reported it, or None
        if nothing has reported at all. The page shows "the robot is asleep"
        over the live view on FALSE only — a black rectangle on a projector
        reads as a broken demo, and so does that sentence over a camera that
        is in fact running."""
        return self._streaming

    def request_stream(self, on: bool) -> None:
        """The Stream button: ask the stage to bring the robot up or put it
        to sleep. Broadcast, because the answer takes 15-20 s (camera+mic
        service, motors, wake_up, voice loop) and every open page should say
        "Starting…" for that whole time, not only the one that was clicked."""
        self._stream_request = bool(on)
        self._broadcast(sse("control", self.control_state()))

    def clear_stream(self, streaming: bool) -> None:
        """The stage acknowledging: the request is spent, and THIS is the
        state the robot ended up in — not always the one asked for (a camera
        service that refuses to start leaves the stream off, and the button
        must say so instead of going green on a robot that is not there).

        Same reason clear_restart exists: without the acknowledgement the
        stage reads the same request on its next poll and starts the robot
        again, once a second, forever."""
        self._stream_request = None
        self._streaming = bool(streaming)
        self._broadcast(sse("control", self.control_state()))

    def on_context(self, tokens: int | None, budget: int, exchanges: int) -> None:
        """The bar the audience watches fill before the oldest turns move into
        Qdrant — the eviction is the point of the demo, so it should be seen
        coming, not only announced after the fact."""
        self._broadcast(sse("context", {"tokens": tokens, "budget": budget,
                                        "exchanges": exchanges}))

    def on_memory_count(self, frames: int, exchanges: int,
                        knowledge: int = 0) -> None:
        self._broadcast(sse("memory_count", {"frames": frames,
                                             "exchanges": exchanges,
                                             "knowledge": knowledge}))

    def on_face(self, name: str | None, box: list[float] | None,
                score: float) -> None:
        """The face the robot is looking at, named if it knows who it is —
        drawn on the live view, where the room can see it is the same person
        the robot is talking to."""
        self._broadcast(sse("face", {"name": name, "box": box,
                                     "score": round(float(score), 3)}))

    def on_heard(self, text: str) -> None:
        self._broadcast(sse("heard", {"text": text}))

    def on_reply(self, text: str, done: bool = False) -> None:
        self._broadcast(sse("reply", {"text": text, "done": done}))

    def close(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()

    # — SSE subscriptions —
    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=SUB_QUEUE_MAXSIZE)
        with self._subs_lock:
            self._subs.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._subs_lock:
            if q in self._subs:
                self._subs.remove(q)

    def _broadcast(self, msg: str) -> None:
        with self._subs_lock:
            for q in list(self._subs):
                try:
                    q.put_nowait(msg)
                except queue.Full:
                    # slow subscriber — drop the oldest event and push the
                    # fresh one, don't block the detect loop on its queue
                    try:
                        q.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        q.put_nowait(msg)
                    except queue.Full:
                        pass


# The dashboard has no login: it is for a trusted network (README.md,
# Security). Two checks keep a web page the presenter happens to open from
# driving it through their browser.
#
# The Host a request names must be one this machine is really reached by — an
# IP address, localhost, a single-label or mDNS `.local` name, this machine's
# own hostname, or one passed with --allow-host. A page on some other domain
# that re-points its DNS at this machine (DNS rebinding) arrives with its own
# domain here and is refused, for the camera stream as much as the buttons.
#
# A POST carrying an Origin must come from the dashboard's own page. Browsers
# send Origin on a cross-site POST; the robot and demo/stage.py send none.
_EXTRA_HOSTS: set[str] = set()


def allow_hosts(names) -> None:
    _EXTRA_HOSTS.update(name.lower().rstrip(".") for name in names)


def host_allowed(host_header: str | None) -> bool:
    if not host_header:
        return False
    host = urlsplit(f"//{host_header}").hostname or ""
    host = host.lower().rstrip(".")
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        pass
    own = socket.gethostname().lower().rstrip(".")
    return (host == "localhost" or "." not in host or host.endswith(".local")
            or host in (own, own.split(".")[0]) or host in _EXTRA_HOSTS)


def origin_allowed(origin: str | None, host_header: str | None) -> bool:
    if origin is None:
        return True
    parts = urlsplit(origin)
    return bool(host_header) and parts.scheme in ("http", "https") \
        and parts.netloc.lower() == host_header.lower()


def make_handler(dashboard: WebDashboard):
    class Handler(BaseHTTPRequestHandler):
        def _refused(self, write: bool) -> bool:
            host = self.headers.get("Host")
            if not host_allowed(host):
                LOG.warning("dashboard: refused a request for host %r", host)
                self.send_error(403, "unknown host")
                return True
            if write and not origin_allowed(self.headers.get("Origin"), host):
                LOG.warning("dashboard: refused a %s from origin %r",
                            self.path, self.headers.get("Origin"))
                self.send_error(403, "cross-origin request")
                return True
            return False

        def do_GET(self):
            # One bad request or missing asset shouldn't crash the
            # presentation server — a clean 404/500 instead.
            if self._refused(write=False):
                return
            try:
                if self.path == "/" or self.path == "/index.html":
                    self._page()
                elif self.path == "/live.mjpeg":
                    self._mjpeg()
                elif self.path == "/events":
                    self._events()
                elif self.path == CONTROL_PATH:
                    self._send_json(dashboard.control_state())
                elif self.path.startswith(RECALL_PATH) or self.path.startswith(LOOK_PATH):
                    jpeg = (dashboard.recall_jpeg(self.path) if self.path.startswith(RECALL_PATH)
                            else dashboard.look_jpeg(self.path))
                    if jpeg is None:
                        self.send_error(404)
                    else:
                        self.send_response(200)
                        self.send_header("Content-Type", "image/jpeg")
                        self.send_header("Content-Length", str(len(jpeg)))
                        self.send_header("Cache-Control", "no-store")
                        self.end_headers()
                        self.wfile.write(jpeg)
                else:
                    self.send_error(404)
            except (BrokenPipeError, ConnectionResetError):
                pass  # client dropped off — not a server error (backstop:
                      # _mjpeg/_events already handle this inside their own loops)
            except FileNotFoundError as exc:
                # E.g. dashboard.html missing from disk — that's an
                # environment/deploy fault, not the client's, but still a
                # specific, legible 404 rather than a bare traceback.
                LOG.warning("static asset missing: %s: %s",
                           type(exc).__name__, exc)
                try:
                    self.send_error(404, "asset not found")
                except Exception:
                    pass
            except Exception as exc:  # noqa: BLE001 — presentation must not crash
                # Broad boundary → log WITH the stack so an unexpected bug is
                # localizable, not just a one-line type+message.
                LOG.exception("unhandled exception: %s", type(exc).__name__)
                try:
                    # Raw exception text in the 500 body — fine for a demo only;
                    # ascii_reason so a non-ASCII message can't crash send_error.
                    self.send_error(500, ascii_reason(str(exc)))
                except Exception:
                    pass  # headers may already be gone (inside _mjpeg/_events)

        def do_POST(self):
            # Writers: the robot's RemoteDisplayClient (demo/display/
            # remote.py), demo/stage.py, and the dashboard page's own buttons.
            if self._refused(write=True):
                return
            try:
                if self.path == PUSH_PATH:
                    self._push()
                elif self.path == CONTROL_PATH:
                    self._control()
                else:
                    self.send_error(404)
            except (BrokenPipeError, ConnectionResetError):
                pass  # client dropped off — not a server error
            except (ValueError, KeyError, TypeError,
                    json.JSONDecodeError) as exc:
                LOG.warning("dashboard push: bad request: %s: %s",
                           type(exc).__name__, exc)
                self.send_error(400, ascii_reason(str(exc)))
            except Exception as exc:  # noqa: BLE001 — presentation must not crash
                LOG.exception("dashboard push: unhandled exception: %s",
                              type(exc).__name__)
                try:
                    self.send_error(500, ascii_reason(str(exc)))
                except Exception:
                    pass

        def _send_json(self, payload: dict, status: int = 200) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _control(self):
            """The three buttons. Whoever can act polls GET /control; this is
            the POST that moves them.

            Five bodies, all from the same trust boundary as /push:
              {"paused": true|false}  the pause button, from the page
              {"restart": true}       the Restart button, from the page
              {"restart_ack": true}   the robot, saying it is restarting now
              {"stream": true|false}  the Stream button, from the page
              {"stream_ack": true|false}  demo/stage.py, saying the robot is
                                          now up (true) or asleep (false)
            """
            length = int(self.headers.get("Content-Length", 0))
            if length <= 0 or length > MAX_PUSH_BODY:
                raise ValueError(f"control body out of bounds: {length} bytes")
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise ValueError("control needs a JSON object")
            known = {"paused", "restart", "restart_ack",
                     "stream", "stream_ack"} & set(body)
            if not known or any(not isinstance(body[key], bool) for key in known):
                raise ValueError('control needs {"paused": true|false}, '
                                 '{"restart": true}, {"restart_ack": true}, '
                                 '{"stream": true|false} or '
                                 '{"stream_ack": true|false}')
            if "paused" in body:
                dashboard.set_paused(body["paused"])
                LOG.info("dashboard: %s", "paused" if body["paused"] else "resumed")
            if body.get("restart"):
                dashboard.request_restart()
                LOG.info("dashboard: restart requested — the robot wipes its "
                         "memory and comes back")
            if body.get("restart_ack"):
                dashboard.clear_restart()
                LOG.info("dashboard: the robot is restarting")
            # `in body`, not .get(): unlike the restart, false is a request
            # of its own here — it is how the robot is put back to sleep.
            if "stream" in body:
                dashboard.request_stream(body["stream"])
                LOG.info("dashboard: stream %s requested",
                         "on" if body["stream"] else "off")
            if "stream_ack" in body:
                dashboard.clear_stream(body["stream_ack"])
                LOG.info("dashboard: the robot is %s",
                         "streaming" if body["stream_ack"] else "asleep")
            self._send_json(dashboard.control_state())

        def _push(self):
            """Re-broadcast a RemoteDisplayClient event exactly as if the
            matching `on_*` method had been called in-process (see that
            client's docstring: it applies the SAME shaping WebDashboard's
            own `on_*` methods do, so this side stays a dumb relay rather
            than a second copy of that shaping logic)."""
            length = int(self.headers.get("Content-Length", 0))
            if length <= 0 or length > MAX_PUSH_BODY:
                raise ValueError(f"push body out of bounds: {length} bytes")
            body = json.loads(self.rfile.read(length))
            dashboard.push(body["event"], body.get("data", {}))
            self.send_response(204)
            self.end_headers()

        def _page(self):
            html = (STATIC / "dashboard.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)

        def _mjpeg(self):
            self.send_response(200)
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    frame = dashboard.camera.latest()
                    if frame is None:
                        time.sleep(0.05)
                        continue
                    jpeg = frame_to_jpeg(frame)
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n"
                                     .encode())
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
                    time.sleep(1 / 20)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _events(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            q = dashboard.subscribe()
            try:
                while True:
                    self.wfile.write(q.get().encode())
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                dashboard.unsubscribe(q)

        def log_request(self, code="-", size="-"):
            # Stay quiet on ordinary successful responses (2xx: page, MJPEG,
            # SSE connection); 4xx/5xx go to the standard log, a backstop on
            # top of the targeted LOG.warning/error calls above.
            if isinstance(code, int) and 200 <= code < 300:
                return
            super().log_request(code, size)

    return Handler


def parse_args(argv=None):
    """Kept out of main() so a test actually executes the defaults: a name
    used only in an argparse default inside main() is not evaluated until
    someone runs the module — and this one reached the stage as a NameError
    on startup, with every test green."""
    from demo.camera_service import DEFAULT_PORT as CAMERA_SERVICE_PORT

    p = argparse.ArgumentParser(
        description="Standalone dashboard: the audience's screen, on the "
                    "Mac, while the voice loop runs on the robot")
    p.add_argument("--host", default="0.0.0.0",
                   help="bound wide by default: the robot pushes its events "
                        "here from across the network")
    p.add_argument("--allow-host", action="append", default=[], metavar="NAME",
                   help="another name this dashboard is reached by (beyond "
                        "IPs, localhost, .local and this machine's hostname)")
    p.add_argument("--port", type=int, default=DEFAULT_DASHBOARD_PORT)
    p.add_argument("--robot-host", default="reachy-mini.local")
    p.add_argument("--robot-camera-port", type=int, default=CAMERA_SERVICE_PORT)
    return p.parse_args(argv)


def main(argv=None) -> int:
    """The standalone dashboard on the laptop: the voice loop runs on the
    robot and pushes its events here (`--display remote`, demo/run_demo.py).
    The camera comes over HTTP from the robot's own camera service."""
    from demo.logs import setup_logging
    from demo.platform.robot_camera import RobotCameraSource, frame_url

    setup_logging()
    args = parse_args(argv)
    allow_hosts(args.allow_host)
    camera = RobotCameraSource(frame_url(args.robot_host, args.robot_camera_port))
    dashboard = WebDashboard(camera)
    dashboard.serve(host=args.host, port=args.port)
    print(f"dashboard on http://{args.host}:{args.port} "
         f"(camera: {args.robot_host}:{args.robot_camera_port}, "
         f"push: {PUSH_PATH})", flush=True)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        print("\n  bye", flush=True)
    finally:
        dashboard.close()
        camera.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
