import http.client
import base64
import json
import time

import numpy as np

from demo.display import NullSink, build_display
from demo.display.events import sse
from demo.display.web import WebDashboard


class FakeCamera:
    def __init__(self, frame):
        self._frame = frame

    def latest(self):
        return self._frame


def drain(q):
    out = []
    while not q.empty():
        out.append(q.get())
    return out


# --- sse() (moved from tests/test_web_events.py) ---

def test_sse_formats_event_and_json():
    out = sse("detections", {"boxes": [{"label": "cup"}]})
    assert out.startswith("event: detections\n")
    assert out.endswith("\n\n")
    line = [l for l in out.splitlines() if l.startswith("data: ")][0]
    assert json.loads(line[len("data: "):]) == {"boxes": [{"label": "cup"}]}


# --- NullSink: all methods are no-ops ---

def test_null_sink_methods_are_all_no_ops():
    sink = NullSink()
    assert sink.on_detections([{"label": "cup"}]) is None
    assert sink.on_heard("hello") is None
    assert sink.on_reply("hi", False) is None
    assert sink.on_reply("hi there", True) is None
    assert sink.on_recall([{"jpeg_b64": "x", "detections": []}]) is None
    assert sink.on_speech_recall([{"text": "hi", "score": 0.9}]) is None
    assert sink.close() is None


# --- WebDashboard: SSE events ---

def test_on_recall_broadcasts_frames_with_boxes():
    # on_recall takes FrameMemory.recall hits (jpeg + YOLO detections) and
    # broadcasts them as a recall event; the browser draws the boxes.
    dash = WebDashboard(FakeCamera(None))
    q = dash.subscribe()
    hits = [{"jpeg_b64": "AAAA",
             "detections": [{"label": "cup", "box": [0.4, 0.4, 0.6, 0.6],
                             "score": 0.8}],
             "labels": ["cup"], "score": 0.13, "found_by": "picture"}]
    dash.on_recall(hits)
    msgs = drain(q)
    recall = [m for m in msgs if m.startswith("event: recall\n")]
    assert len(recall) == 1
    payload = json.loads(recall[0].splitlines()[1][len("data: "):])
    # The picture is not in the event: the page fetches it from the server.
    assert payload["frames"] == [{"url": "/recall/1/0.jpg",
                                  "boxes": hits[0]["detections"],
                                  "score": 0.13, "found_by": "picture",
                                  "weak": False}]
    assert dash.recall_jpeg("/recall/1/0.jpg") == base64.b64decode("AAAA")
    assert dash.recall_jpeg("/recall/0/0.jpg") is None      # an older recall
    assert dash.recall_jpeg("/recall/1/5.jpg") is None      # no such frame


def test_a_pushed_recall_is_served_as_pictures_not_sent_inside_the_event():
    """Live: a ~210 KB recall event never reached the dashboard
    page in Chrome; the room saw "nothing recalled"."""
    dash = WebDashboard(FakeCamera(None))
    q = dash.subscribe()
    dash.push("recall", {"frames": [{"jpeg_b64": base64.b64encode(b"JPEG1").decode(),
                                     "boxes": [], "score": 0.5, "weak": False}]})
    msg = [m for m in drain(q) if m.startswith("event: recall\n")][0]
    assert "jpeg_b64" not in msg and len(msg) < 200
    assert dash.recall_jpeg("/recall/1/0.jpg") == b"JPEG1"


def test_on_recall_with_no_hits_broadcasts_an_explicit_empty_result():
    # Nothing recalled (empty, or all gated below the score floor) must still
    # broadcast — a suppressed event used to leave the PREVIOUS question's
    # frames on screen while the robot answered a new, unrelated one (a
    # confident mismatch on stage, worse than a blank panel). The explicit
    # empty payload lets the dashboard render its own "nothing recalled"
    # resting state instead of going stale.
    dash = WebDashboard(FakeCamera(None))
    q = dash.subscribe()
    dash.on_recall([])
    msgs = drain(q)
    recall = [m for m in msgs if m.startswith("event: recall\n")]
    assert len(recall) == 1
    payload = json.loads(recall[0].splitlines()[1][len("data: "):])
    assert payload["frames"] == []


def _payload(messages, event):
    lines = [m for m in messages if m.startswith(f"event: {event}\n")]
    assert len(lines) == 1
    return json.loads(lines[0].splitlines()[1][len("data: "):])


def test_on_context_broadcasts_the_fill_of_the_model_context():
    dash = WebDashboard(FakeCamera(None))
    q = dash.subscribe()
    dash.on_context(310, 500, 4)
    assert _payload(drain(q), "context") == {"tokens": 310, "budget": 500,
                                             "exchanges": 4}


def test_on_context_passes_a_missing_count_through():
    # An image turn runs in a side chat and reports none — the page decides
    # what to do with that, this side does not invent a number.
    dash = WebDashboard(FakeCamera(None))
    q = dash.subscribe()
    dash.on_context(None, 500, 2)
    assert _payload(drain(q), "context")["tokens"] is None


def test_on_memory_count_broadcasts_every_shard():
    dash = WebDashboard(FakeCamera(None))
    q = dash.subscribe()
    dash.on_memory_count(147, 12, 41)
    assert _payload(drain(q), "memory_count") == {
        "frames": 147, "exchanges": 12, "knowledge": 41}


def test_null_sink_takes_the_new_events():
    from demo.display import NullSink

    sink = NullSink()
    assert sink.on_context(310, 500, 4) is None
    assert sink.on_memory_count(147, 12) is None


def test_on_speech_recall_broadcasts_said_utterances():
    # "search by what was said": recalled past utterances (TextMemory hits)
    # go out as a speech_recall event with text + score.
    dash = WebDashboard(FakeCamera(None))
    q = dash.subscribe()
    hits = [{"text": "my stream starts at 16:00", "kind": "speech",
             "score": 0.82}]
    dash.on_speech_recall(hits)
    msgs = drain(q)
    sr = [m for m in msgs if m.startswith("event: speech_recall\n")]
    assert len(sr) == 1
    payload = json.loads(sr[0].splitlines()[1][len("data: "):])
    assert payload["items"] == [{"text": "my stream starts at 16:00",
                                 "score": 0.82, "source": ""}]


def test_on_speech_recall_with_no_hits_broadcasts_an_explicit_empty_result():
    # Same staleness fix as on_recall: a suppressed event on an empty result
    # would leave a PREVIOUS turn's recalled utterance on screen during an
    # unrelated answer. Broadcast the empty list so the panel hides/clears
    # instead of showing something nobody just said.
    dash = WebDashboard(FakeCamera(None))
    q = dash.subscribe()
    dash.on_speech_recall([])
    msgs = drain(q)
    sr = [m for m in msgs if m.startswith("event: speech_recall\n")]
    assert len(sr) == 1
    payload = json.loads(sr[0].splitlines()[1][len("data: "):])
    assert payload["items"] == []


def test_on_detections_is_the_source_listener():
    """on_detections is what source.add_listener(display.on_detections)
    calls on every detect cycle; it should broadcast a detections event."""
    dash = WebDashboard(FakeCamera(None))
    q = dash.subscribe()
    dets = [{"label": "person", "score": 0.9, "box": [0, 0, 1, 1]}]
    dash.on_detections(dets)
    msgs = drain(q)
    assert any(m.startswith("event: detections\n") for m in msgs)
    payload = json.loads(msgs[0].splitlines()[1][len("data: "):])
    assert payload["boxes"] == dets


def test_on_heard_and_reply_events():
    dash = WebDashboard(FakeCamera(None))
    q = dash.subscribe()
    dash.on_heard("hello")
    dash.on_reply("Hi.", done=False)
    dash.on_reply("Hi there.", done=True)
    msgs = drain(q)
    assert any(m.startswith("event: heard\n") for m in msgs)
    replies = [m for m in msgs if m.startswith("event: reply\n")]
    assert len(replies) == 2


def test_unsubscribe_stops_delivery():
    dash = WebDashboard(FakeCamera(None))
    q = dash.subscribe()
    dash.unsubscribe(q)
    dash.on_heard("hello")
    assert drain(q) == []


def test_subscriber_queue_drops_oldest_when_full():
    """The subscriber queue is bounded — overflow drops the oldest
    message rather than growing without bound or blocking the broadcast."""
    dash = WebDashboard(FakeCamera(None))
    q = dash.subscribe()
    from demo.display.web import SUB_QUEUE_MAXSIZE

    for i in range(SUB_QUEUE_MAXSIZE + 5):
        dash.on_heard(str(i))
    msgs = drain(q)
    assert len(msgs) == SUB_QUEUE_MAXSIZE
    # the oldest events (0..4) were dropped, the freshest ones are still here
    assert '"text": "0"' not in msgs[0]
    assert msgs[-1].strip().endswith(
        f'"text": "{SUB_QUEUE_MAXSIZE + 4}"}}')


# --- WebDashboard: real HTTP — MJPEG multipart + SSE framing ---

def _wait_for_subscriber(dash, timeout=2.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        with dash._subs_lock:
            if dash._subs:
                return
        time.sleep(0.01)
    raise AssertionError("subscriber never registered")


def test_mjpeg_endpoint_serves_multipart_boundary():
    dash = WebDashboard(FakeCamera(np.zeros((4, 4, 3), np.uint8)))
    server = dash.serve(host="127.0.0.1", port=0)
    try:
        host, port = server.server_address
        conn = http.client.HTTPConnection(host, port, timeout=5)
        conn.request("GET", "/live.mjpeg")
        resp = conn.getresponse()
        assert resp.status == 200
        assert resp.getheader("Content-Type").startswith(
            "multipart/x-mixed-replace")
        chunk = resp.read(64)
        assert chunk.startswith(b"--frame\r\nContent-Type: image/jpeg\r\n")
        conn.close()
    finally:
        dash.close()


def test_events_endpoint_streams_sse_framed_messages():
    dash = WebDashboard(FakeCamera(None))
    server = dash.serve(host="127.0.0.1", port=0)
    try:
        host, port = server.server_address
        conn = http.client.HTTPConnection(host, port, timeout=5)
        conn.request("GET", "/events")
        resp = conn.getresponse()
        assert resp.status == 200
        assert resp.getheader("Content-Type") == "text/event-stream"
        _wait_for_subscriber(dash)
        dash.on_heard("hello")
        expected = sse("heard", {"text": "hello"}).encode()
        assert resp.read(len(expected)) == expected
        conn.close()
    finally:
        dash.close()


def test_index_page_serves_dashboard_html():
    dash = WebDashboard(FakeCamera(None))
    server = dash.serve(host="127.0.0.1", port=0)
    try:
        host, port = server.server_address
        conn = http.client.HTTPConnection(host, port, timeout=5)
        conn.request("GET", "/")
        resp = conn.getresponse()
        assert resp.status == 200
        assert resp.getheader("Content-Type").startswith("text/html")
        body = resp.read()
        assert b"<html" in body
        conn.close()
    finally:
        dash.close()


def test_unknown_path_returns_404():
    dash = WebDashboard(FakeCamera(None))
    server = dash.serve(host="127.0.0.1", port=0)
    try:
        host, port = server.server_address
        conn = http.client.HTTPConnection(host, port, timeout=5)
        conn.request("GET", "/nope")
        resp = conn.getresponse()
        assert resp.status == 404
        resp.read()
        conn.close()
    finally:
        dash.close()


# --- build_display: factory ---

def test_build_display_none_returns_null_sink():
    display = build_display("none", camera=FakeCamera(None),
                            host="127.0.0.1", port=0)
    assert isinstance(display, NullSink)


def test_build_display_web_returns_started_web_dashboard():
    display = build_display("web", camera=FakeCamera(None),
                            host="127.0.0.1", port=0)
    try:
        assert isinstance(display, WebDashboard)
        assert display._server is not None
    finally:
        display.close()


def test_build_display_rejects_unknown_kind():
    import pytest

    with pytest.raises(ValueError):
        build_display("tui", camera=FakeCamera(None), host="127.0.0.1", port=0)


def test_build_display_remote_returns_a_remote_display_client():
    from demo.display.remote import RemoteDisplayClient

    display = build_display("remote", camera=FakeCamera(None), host="127.0.0.1",
                            port=0, dashboard_host="mac.local", dashboard_port=9999)
    try:
        assert isinstance(display, RemoteDisplayClient)
        assert display._url == "http://mac.local:9999/push"
    finally:
        display.close()


def test_build_display_remote_requires_a_dashboard_host():
    import pytest

    with pytest.raises(ValueError):
        build_display("remote", camera=FakeCamera(None), host="127.0.0.1", port=0)


# --- WebDashboard: /push (RemoteDisplayClient's receiving end) ---

def test_push_endpoint_rebroadcasts_event_to_subscribers():
    dash = WebDashboard(FakeCamera(None))
    server = dash.serve(host="127.0.0.1", port=0)
    try:
        host, port = server.server_address
        sub = http.client.HTTPConnection(host, port, timeout=5)
        sub.request("GET", "/events")
        resp = sub.getresponse()
        _wait_for_subscriber(dash)

        push = http.client.HTTPConnection(host, port, timeout=5)
        body = json.dumps({"event": "heard", "data": {"text": "hello"}}).encode()
        push.request("POST", "/push", body=body,
                     headers={"Content-Type": "application/json"})
        push_resp = push.getresponse()
        assert push_resp.status == 204
        push_resp.read()
        push.close()

        expected = sse("heard", {"text": "hello"}).encode()
        assert resp.read(len(expected)) == expected
        sub.close()
    finally:
        dash.close()


def test_push_endpoint_rejects_an_oversized_body():
    """The bound is checked against the announced Content-Length BEFORE
    reading the body (cap a request body's size before reading it fully into
    memory) — this only announces the big length, never actually
    sends that many bytes, so the assertion isn't racing a real multi-MB
    upload (and can't trip a BrokenPipeError once the server 400s early)."""
    from demo.display.web import MAX_PUSH_BODY

    dash = WebDashboard(FakeCamera(None))
    server = dash.serve(host="127.0.0.1", port=0)
    try:
        host, port = server.server_address
        conn = http.client.HTTPConnection(host, port, timeout=5)
        conn.putrequest("POST", "/push")
        conn.putheader("Content-Type", "application/json")
        conn.putheader("Content-Length", str(MAX_PUSH_BODY + 1))
        conn.endheaders()
        resp = conn.getresponse()
        assert resp.status == 400
        resp.read()
        conn.close()
    finally:
        dash.close()


def test_push_endpoint_unknown_path_404s():
    dash = WebDashboard(FakeCamera(None))
    server = dash.serve(host="127.0.0.1", port=0)
    try:
        host, port = server.server_address
        conn = http.client.HTTPConnection(host, port, timeout=5)
        conn.request("POST", "/nope", body=b"{}")
        resp = conn.getresponse()
        assert resp.status == 404
        resp.read()
        conn.close()
    finally:
        dash.close()


def test_the_dashboard_parses_its_arguments_with_real_defaults():
    # Regression: --port's default lived only inside main(), so a missing
    # import surfaced as a NameError on startup, in front of the user.
    from demo.display import DEFAULT_DASHBOARD_PORT
    from demo.display.web import parse_args

    args = parse_args([])
    assert args.port == DEFAULT_DASHBOARD_PORT
    assert parse_args(["--port", "9999"]).port == 9999


def test_the_pause_starts_off_and_is_broadcast_when_it_changes():
    dash = WebDashboard(FakeCamera(None))
    q = dash.subscribe()
    assert dash.is_paused() is False
    dash.set_paused(True)
    assert dash.is_paused() is True
    # One payload carries every button: a page opened halfway through the demo
    # gets the whole state, not just the flag that happened to change.
    assert _payload(drain(q), "control") == {"paused": True, "restart": False,
                                             "stream": None, "streaming": None}


def test_null_sink_is_never_paused():
    assert NullSink().is_paused() is False


def test_null_sink_never_asks_for_a_restart():
    assert NullSink().restart_requested() is False
    NullSink().ack_restart()   # and does not raise


def test_the_restart_button_asks_once_and_clears_the_screen():
    """It wipes the robot's memory (scripts/voice_loop.sh), so the panels must
    not keep describing a conversation that no longer exists — and the pause
    goes with it: a robot that just restarted should be listening."""
    dash = WebDashboard(FakeCamera(None))
    q = dash.subscribe()
    dash.set_paused(True)
    drain(q)
    assert dash.restart_requested() is False
    dash.request_restart()
    assert dash.restart_requested() is True
    events = drain(q)
    assert _payload(events, "control")["paused"] is False
    assert _payload(events, "reset") == {}
    assert dash.is_paused() is False
    # The robot says it is restarting; the request is spent, or the loop that
    # comes back up reads it again and restarts forever.
    dash.clear_restart()
    assert dash.restart_requested() is False


def test_the_pause_button_flips_it_over_http():
    dash = WebDashboard(FakeCamera(None))
    server = dash.serve(host="127.0.0.1", port=0)
    host, port = server.server_address

    def call(method, body=None):
        conn = http.client.HTTPConnection(host, port, timeout=5)
        conn.request(method, "/control", body=body,
                     headers={"Content-Type": "application/json"} if body else {})
        response = conn.getresponse()
        payload = response.read()
        conn.close()
        return response.status, payload

    try:
        assert json.loads(call("GET")[1]) == {"paused": False, "restart": False,
                                              "stream": None, "streaming": None}
        status, payload = call("POST", json.dumps({"paused": True}))
        assert status == 200 and json.loads(payload)["paused"] is True
        assert dash.is_paused() is True
        # Only a real boolean: a typo must not quietly leave the robot deaf.
        assert call("POST", json.dumps({"paused": "yes"}))[0] == 400
        assert dash.is_paused() is True
    finally:
        dash.close()


def test_the_restart_button_travels_over_http_and_is_acknowledged():
    """The page asks; the robot reads the request, then says it is restarting.
    A counter would have done neither: restarting this dashboard would reset
    it, and the robot — comparing against a baseline from before — would wipe
    its memory because someone restarted the Mac's screen."""
    dash = WebDashboard(FakeCamera(None))
    server = dash.serve(host="127.0.0.1", port=0)
    host, port = server.server_address

    def call(method, body=None):
        conn = http.client.HTTPConnection(host, port, timeout=5)
        conn.request(method, "/control", body=body,
                     headers={"Content-Type": "application/json"} if body else {})
        response = conn.getresponse()
        payload = response.read()
        conn.close()
        return response.status, payload

    try:
        status, payload = call("POST", json.dumps({"restart": True}))
        assert status == 200 and json.loads(payload)["restart"] is True
        # What the robot's watcher reads, poll after poll, until it acts.
        assert json.loads(call("GET")[1])["restart"] is True
        status, payload = call("POST", json.dumps({"restart_ack": True}))
        assert status == 200 and json.loads(payload)["restart"] is False
        assert dash.restart_requested() is False
        # Neither key, or a non-boolean: rejected rather than read as a wipe.
        assert call("POST", json.dumps({"nonsense": True}))[0] == 400
        assert call("POST", json.dumps({"restart": "yes"}))[0] == 400
        assert dash.restart_requested() is False
    finally:
        dash.close()


def _control_call(server):
    """POST/GET /control on a running dashboard, the way the page does."""
    host, port = server.server_address

    def call(method, body=None):
        conn = http.client.HTTPConnection(host, port, timeout=5)
        conn.request(method, "/control", body=body,
                     headers={"Content-Type": "application/json"} if body else {})
        response = conn.getresponse()
        payload = response.read()
        conn.close()
        return response.status, payload

    return call


def test_nothing_is_pending_and_nobody_is_managing_the_stream_by_default():
    """`streaming: None` is not "asleep": it means nothing that can act has
    said anything yet. The page keeps showing the live view — printing "the
    robot is asleep" over a camera that is running (the in-process
    `--display web` debug mode) is its own broken demo."""
    dash = WebDashboard(FakeCamera(None))
    assert dash.stream_requested() is None
    assert dash.is_streaming() is None
    assert dash.control_state() == {"paused": False, "restart": False,
                                    "stream": None, "streaming": None}


def test_the_stream_button_asks_and_the_stage_answers_with_the_state_it_reached():
    """The request is spent on acknowledgement, and what comes back is what
    the robot IS — not what was asked for. A camera+mic service that refused
    to start leaves the button grey rather than green over a robot that is
    not there."""
    dash = WebDashboard(FakeCamera(None))
    q = dash.subscribe()
    dash.request_stream(True)
    assert dash.stream_requested() is True
    assert _payload(drain(q), "control")["stream"] is True
    dash.clear_stream(False)
    assert dash.stream_requested() is None
    assert dash.is_streaming() is False
    state = _payload(drain(q), "control")
    assert state["stream"] is None and state["streaming"] is False


def test_the_stream_button_travels_over_http_and_is_acknowledged():
    """The page asks; demo/stage.py reads the request, does the 15-20 s of
    work, then says which state the robot ended in. A counter would have done
    neither: restarting this dashboard resets it, and the stage — comparing
    against a baseline from before — would put a talking robot to sleep
    because someone reloaded the Mac's screen."""
    dash = WebDashboard(FakeCamera(None))
    call = _control_call(dash.serve(host="127.0.0.1", port=0))
    try:
        status, payload = call("POST", json.dumps({"stream": True}))
        assert status == 200 and json.loads(payload)["stream"] is True
        # What the stage reads, poll after poll, until it acts.
        assert json.loads(call("GET")[1])["stream"] is True
        status, payload = call("POST", json.dumps({"stream_ack": True}))
        assert status == 200
        assert json.loads(payload) == {"paused": False, "restart": False,
                                       "stream": None, "streaming": True}
        assert dash.stream_requested() is None and dash.is_streaming() is True
        # A typo must not read as "put the robot to sleep".
        assert call("POST", json.dumps({"stream": "on"}))[0] == 400
        assert dash.stream_requested() is None
    finally:
        dash.close()


def test_stream_off_is_a_request_of_its_own_not_an_absent_one():
    """Unlike the restart, `false` here is the whole point of the button: it
    is how the robot is put to sleep. Read with truthiness it would be
    indistinguishable from nothing pending, and Stream off would do nothing."""
    dash = WebDashboard(FakeCamera(None))
    call = _control_call(dash.serve(host="127.0.0.1", port=0))
    try:
        call("POST", json.dumps({"stream_ack": True}))
        assert call("POST", json.dumps({"stream": False}))[0] == 200
        assert dash.stream_requested() is False
        assert json.loads(call("GET")[1])["stream"] is False
        call("POST", json.dumps({"stream_ack": False}))
        assert dash.stream_requested() is None and dash.is_streaming() is False
    finally:
        dash.close()


def test_the_look_is_served_by_url_and_only_the_latest_one():
    dash = WebDashboard(FakeCamera(None))
    q = dash.subscribe()
    dash.on_look(b"JPEG-A")
    dash.push("look", {"jpeg_b64": base64.b64encode(b"JPEG-B").decode()})
    msgs = [m for m in drain(q) if m.startswith("event: look\n")]
    assert [json.loads(m.splitlines()[1][len("data: "):])["url"] for m in msgs] == [
        "/look/1.jpg", "/look/2.jpg"]
    assert all(len(m) < 100 for m in msgs)
    assert dash.look_jpeg("/look/2.jpg") == b"JPEG-B"
    assert dash.look_jpeg("/look/1.jpg") is None
    assert dash.look_jpeg("/look/x.jpg") is None


# --- who the dashboard answers (README "Security") ---

def test_the_dashboard_answers_the_names_it_is_reached_by():
    from demo.display.web import allow_hosts, host_allowed

    for host in ("127.0.0.1:8091", "[::1]:8091", "192.168.1.20:8091",
                 "localhost:8091", "reachy-mini.local", "laptop"):
        assert host_allowed(host), host
    allow_hosts(["stage.example.org"])
    assert host_allowed("stage.example.org:8091")


def test_a_page_reaching_it_through_dns_rebinding_is_refused():
    # A rebinding page's requests carry its own domain as the Host.
    from demo.display.web import host_allowed

    assert not host_allowed("attacker.example.com:8091")
    assert not host_allowed(None)


def test_a_button_press_from_another_page_is_refused():
    from demo.display.web import origin_allowed

    host = "192.168.1.20:8091"
    assert origin_allowed(None, host)                       # curl, the stage
    assert origin_allowed(f"http://{host}", host)           # the page itself
    assert not origin_allowed("http://attacker.example.com", host)
    assert not origin_allowed("null", host)


def test_a_restart_from_another_page_never_reaches_the_robot():
    dash = WebDashboard(FakeCamera(None))
    server = dash.serve(host="127.0.0.1", port=0)
    try:
        host, port = server.server_address
        for headers in ({"Origin": "http://attacker.example.com"},
                        {"Host": "attacker.example.com"}):
            conn = http.client.HTTPConnection(host, port, timeout=5)
            conn.request("POST", "/control", body=json.dumps({"restart": True}),
                         headers={"Content-Type": "application/json", **headers})
            assert conn.getresponse().status == 403
            conn.close()
        assert dash.restart_requested() is False
    finally:
        dash.close()
