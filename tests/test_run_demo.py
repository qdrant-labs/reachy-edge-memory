import time
from pathlib import Path

import numpy as np
import pytest



@pytest.fixture(autouse=True)
def _the_laptops_embedders(monkeypatch):
    """build_memories asks the laptop's embed service which models it runs
    (emulator/embed_identity.py); here it answers without a network."""
    from emulator import embed_identity

    monkeypatch.setattr(embed_identity, "of_service",
                        lambda url, timeout=5.0: embed_identity.current())


class FakeRobot:
    def __init__(self):
        self.calls = []
    def look_at(self, x, y):
        self.calls.append(("look_at", (x, y)))
    def gesture(self, name):
        self.calls.append(("gesture", name))
    def emotion(self, name):
        self.calls.append(("emotion", name))
    def dance(self, name="yeah_nod"):
        self.calls.append(("dance", name))


class FakePlayer:
    """Stub player: tracks lengths of fed phrases, remembers close."""
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.feeds = []
        self.closed = False
    def feed(self, samples):
        self.feeds.append(len(samples))
    def close(self):
        self.closed = True


# --- streaming mode ---

def test_drive_robot_stream_turns_head_then_speaks_each_phrase():
    import base64
    import numpy as np
    from demo.run_demo import drive_robot_stream

    def audio_event(text):
        a = np.zeros(8000, dtype=np.float32)
        return {"type": "audio", "text": text,
                "audio_b64": base64.b64encode(a.tobytes()).decode(),
                "sample_rate": 16000}

    events = [
        {"type": "meta", "objects": ["person"], "heard": "hi", "gaze": [0.3, -0.1]},
        audio_event("Hello."),
        audio_event("I see you."),
        {"type": "done", "reply": "Hello. I see you.", "first_sound_ms": 1300,
         "total_ms": 3000},
    ]
    robot = FakeRobot()
    player = FakePlayer()
    done = drive_robot_stream(robot, events, make_player=lambda sr: player)
    # The robot only moves (gaze); audio goes to a single player, back to back.
    assert [c[0] for c in robot.calls] == ["look_at"]
    assert robot.calls[0][1] == (0.3, -0.1)
    assert player.feeds == [8000, 8000], "two phrases fed into one player"
    assert player.closed, "player closed at the end of the response"
    assert done["reply"] == "Hello. I see you."


def test_drive_robot_stream_skips_empty_audio():
    import base64
    import numpy as np
    from demo.run_demo import drive_robot_stream

    events = [
        {"type": "meta", "objects": [], "heard": "", "gaze": [0.0, 0.0]},
        {"type": "audio", "text": "x",
         "audio_b64": base64.b64encode(np.zeros(0, np.float32).tobytes()).decode(),
         "sample_rate": 16000},
        {"type": "done", "reply": "x"},
    ]
    robot = FakeRobot()
    player = FakePlayer()
    drive_robot_stream(robot, events, make_player=lambda sr: player)
    # Empty audio isn't fed to the player and doesn't trigger feed calls.
    assert player.feeds == []


def test_drive_robot_stream_forwards_reply_chunks():
    from demo.run_demo import drive_robot_stream
    import base64, numpy as np

    class DummyRobot:
        def look_at(self, x, y): pass
        def say(self, *a): pass
        def gesture(self, g): pass

    class NullPlayer:
        def __init__(self, sr): pass
        def feed(self, a): pass
        def close(self): pass

    audio_b64 = base64.b64encode(
        np.zeros(400, np.float32).tobytes()).decode()
    events = [
        {"type": "meta", "objects": [], "heard": "hi", "gaze": [0, 0]},
        {"type": "audio", "text": "Hi.", "audio_b64": audio_b64,
         "sample_rate": 16000},
        {"type": "done", "reply": "Hi there.", "first_sound_ms": 100},
    ]
    seen = []
    drive_robot_stream(DummyRobot(), iter(events),
                       on_reply=lambda t, d: seen.append((t, d)),
                       make_player=NullPlayer)
    assert ("Hi.", False) in seen
    assert ("Hi there.", True) in seen


# --- _handle_stream: the voice loop always goes through a DisplaySink ---
# (the source from RemoteDetectSource always returns a detections list — the
# "webui vs none" branch was removed from run_voice; _handle_stream accepts
# a DisplaySink of any implementation — here a spy stub, like
# NullSink/WebDashboard.)

class DisplaySpy:
    def __init__(self):
        self.detections = []
        self.queries = []
        self.heard = []
        self.replies = []
        self.recalls = []
        self.speech_recalls = []
        self.tool_calls = []
        self.memory_writes = []
        self.contexts = []
        self.memory_counts = []
        self.faces = []
        self.closed = False

    def on_detections(self, detections):
        self.detections.append(detections)

    def on_look(self, jpeg):
        self.queries.append(("look", jpeg))

    def on_heard(self, text):
        self.heard.append(text)

    def on_reply(self, text, done=False):
        self.replies.append((text, done))

    def on_recall(self, objects):
        self.recalls.append(objects)

    def on_speech_recall(self, hits):
        self.speech_recalls.append(hits)

    def on_tool_call(self, name, arguments):
        self.tool_calls.append((name, arguments))

    def on_memory_write(self, texts):
        self.memory_writes.append(texts)

    def on_context(self, tokens, budget, exchanges):
        self.contexts.append((tokens, budget, exchanges))

    def on_memory_count(self, frames, exchanges, knowledge=0):
        self.memory_counts.append((frames, exchanges))

    def on_face(self, name, box, score):
        self.faces.append((name, box, score))

    def is_paused(self):
        return False

    def close(self):
        self.closed = True


# --- transcribe(): step 1 of the turn sequence (transcribe -> retrieve ->
# generate, see the module docstring above demo.run_demo.transcribe) ---

def test_transcribe_posts_audio_and_returns_heard_text():
    from demo.run_demo import transcribe

    captured = {}

    def fake_post(endpoint, payload):
        captured["endpoint"] = endpoint
        captured["payload"] = payload
        return {"heard": "what did you see?"}

    heard = transcribe(np.zeros(8000, np.float32), 16000,
                       "http://pi:9500/transcribe", http_post=fake_post)

    assert heard == "what did you see?"
    assert captured["endpoint"] == "http://pi:9500/transcribe"
    assert "audio_b64" in captured["payload"]
    assert "frame_b64" not in captured["payload"], (
        "transcription carries only audio, never a picture")


def test_transcribe_returns_empty_string_for_a_false_vad_trigger():
    from demo.run_demo import transcribe

    heard = transcribe(np.zeros(400, np.float32), 16000, "http://pi:9500/transcribe",
                       http_post=lambda endpoint, payload: {"heard": ""})
    assert heard == ""


def test_transcribe_picks_up_monkeypatched_module_level_http_post(monkeypatch):
    # Regression: a plain `http_post=_http_post` DEFAULT PARAMETER would bind
    # the original function at import time — monkeypatching
    # demo.run_demo._http_post afterwards would then silently be ignored by
    # any caller (like _handle_stream) that doesn't pass http_post= itself.
    # transcribe() must look it up at CALL time instead.
    import demo.run_demo as run_demo_mod

    monkeypatch.setattr(run_demo_mod, "_http_post",
                        lambda endpoint, payload: {"heard": "patched"})
    assert run_demo_mod.transcribe(np.zeros(400, np.float32), 16000,
                                   "http://pi:9500/transcribe") == "patched"


def _say_event(text):
    return {"type": "audio", "text": text, "sample_rate": 16000,
            "audio_b64": _b64(np.zeros(400, np.float32).tobytes())}


def _fake_transcribe(heard: str, endpoints_seen: list[str] | None = None):
    """Monkeypatch target for demo.run_demo._http_post standing in for the
    laptop's /transcribe — every _handle_stream test needs one, since a turn
    always transcribes first. Records the endpoint it was called against
    when a list is given."""
    def fake_post(endpoint, payload):
        if endpoints_seen is not None:
            endpoints_seen.append(endpoint)
        if endpoint.endswith("/say"):
            return _say_event(payload["text"])
        return {"heard": heard}
    return fake_post








class _FakeFrameMemory:
    """Stand-in for emulator.frame_memory.FrameMemory — records the query and
    returns canned frames, without SigLIP/onnx. `hits` answer the frames'
    words search and `day` is the day's frames, each filtered by `before` as
    the real ones are, so a test sees whether the turn's start reached them.
    There is no picture search: nothing in the turn may ask for one. `calls`
    records recall/remember in the order they actually happened, for the
    recall-before-store tests."""

    def __init__(self, hits, day=()):
        self._hits = hits
        self._day = list(day)
        self.queries: list[str] = []
        self.remembered: list[tuple] = []
        self.calls: list[str] = []

    def remember(self, frame, detections=None, meta=None):
        self.calls.append("remember")
        self.remembered.append((frame, detections))

    def count(self):
        return len(self.remembered)

    def latest_looks(self, directions=None, limit=4, before=None):
        return []

    def latest_scenes(self, limit=3, before=None):
        return []

    def recall_text(self, query, k=3, before=None):
        self.calls.append("recall")
        self.queries.append(query)
        return [hit for hit in self._hits
                if before is None or hit.get("ts", 0.0) < before]

    def day_frames(self, before=None):
        return [frame for frame in self._day
                if before is None or frame.get("ts", 0.0) < before]






# --- RECALL BEFORE STORE (frames): this turn's frame must not already be in
# the index when its own recall runs, or the robot could "recall" the view
# it's looking at right now (matches the speech memory's existing ordering).

def _no_generate_http_stream(endpoint, payload):
    """Monkeypatch target for demo.run_demo._http_stream, for tests where
    the transcript is empty: generation must be skipped ENTIRELY (see
    _handle_stream's "cost nothing" short circuit) — calling this at all is
    the regression it guards against."""
    raise AssertionError(
        "the generate endpoint must not be called on an empty transcript")






# --- turn_started_at: RECALL BEFORE STORE above only stops THIS turn's own
# frame; SceneChangeWriter (emulator/frame_memory.py) stores independently on
# the detect thread and can beat it — hits it wrote during this very turn
# must be excluded too. ---





# --- _turn_started_at: anchored on the utterance itself, not on when
# listening started — run_voice used to take turn_started_at BEFORE
# collect_utterance blocked for the whole idle wait (which can be minutes on
# a demo floor), so every frame SceneChangeWriter stored during that wait
# was wrongly excluded as "part of this turn" by the filter tested above. ---

def test_turn_started_at_anchors_on_utterance_duration_not_now():
    from demo.run_demo import TURN_START_GUARD_S, _turn_started_at

    audio = np.zeros(32000, np.float32)  # 2.0 s of utterance at 16 kHz
    assert _turn_started_at(audio, now=102.0) == 102.0 - 2.0 - TURN_START_GUARD_S




# --- _turn_started_at duration cap: a longer question must not discard more
# of the past than a shorter one. Before TURN_DURATION_CAP_S, the exclusion
# window was `now - len(audio)/16000 - GUARD` with no ceiling, so:
#   object seen 4.0s ago, 2.0s question -> recalled   (fine)
#   object seen 6.0s ago, 5.0s question -> DROPPED    (bug: a natural longer
#     stage question, "do you remember that green bottle I showed you a
#     moment ago, where was it?", silently ate its own answer)
# Both cases must now recall. ---

def test_turn_started_at_caps_duration_contribution_for_long_questions():
    from demo.run_demo import (TURN_DURATION_CAP_S, TURN_START_GUARD_S,
                                _turn_started_at)

    short_audio = np.zeros(int(2.0 * 16000), np.float32)   # 2.0 s — under the cap
    long_audio = np.zeros(int(5.0 * 16000), np.float32)    # 5.0 s — over the cap

    assert _turn_started_at(short_audio, now=100.0) == (
        100.0 - 2.0 - TURN_START_GUARD_S), "under the cap: unaffected"
    assert _turn_started_at(long_audio, now=100.0) == (
        100.0 - TURN_DURATION_CAP_S - TURN_START_GUARD_S), (
        "over the cap: duration contribution ceilinged, not the full 5.0s")








# --- empty-heard turns must clear the recall panels too, not just "Heard:" —
# else the dashboard keeps showing the PREVIOUS question's frames/quote next
# to a blank Heard line on a false VAD trigger (cough, applause). ---



class _FakeSpeechMemory:
    """Stand-in for the Mac's bge TextMemory of what was said."""

    def __init__(self, hits):
        self._hits = hits
        self.queries: list[str] = []
        self.remembered: list[tuple[str, str]] = []

    def recall(self, query, k=3):
        self.queries.append(query)
        return self._hits

    def remember(self, text, kind, meta=None):
        self.remembered.append((kind, text))

    def count(self, kind=None):
        return sum(1 for stored, _ in self.remembered
                   if kind is None or stored == kind)








# --- _FailureGuard: the healthy flag logs the transition, not every call ---

def test_failure_guard_logs_failure_once_not_on_every_repeat(capsys):
    from demo.run_demo import _FailureGuard

    guard = _FailureGuard("robot")

    def boom():
        raise RuntimeError("disconnected")

    guard.call(boom)
    guard.call(boom)
    guard.call(boom)
    out = capsys.readouterr().out
    assert out.count("[robot] skip") == 1, "repeated failures shouldn't spam"
    assert guard.healthy is False


def test_failure_guard_logs_recovery_once(capsys):
    from demo.run_demo import _FailureGuard

    guard = _FailureGuard("robot")
    state = {"n": 0}

    def flaky():
        state["n"] += 1
        if state["n"] == 1:
            raise RuntimeError("disconnected")

    guard.call(flaky)  # first call fails
    assert guard.healthy is False
    guard.call(flaky)  # second call succeeds -> recovered
    assert guard.healthy is True
    out = capsys.readouterr().out
    assert out.count("[robot] skip") == 1
    assert out.count("recovered") == 1


def test_drive_robot_stream_logs_robot_failure_once_across_events(capsys):
    # Within a SINGLE response, several events hit a hopelessly failed robot
    # — the "skip" print should happen once, not on every event.
    from demo.run_demo import drive_robot_stream

    class FlakyRobot:
        def look_at(self, x, y):
            raise RuntimeError("disconnected")
        def gesture(self, name):
            raise RuntimeError("disconnected")
        def say(self, *a):
            pass

    events = [
        {"type": "meta", "objects": [], "heard": "hi", "gaze": [0.0, 0.0]},
        {"type": "meta", "objects": [], "heard": "and", "gaze": [0.1, 0.0]},
        {"type": "meta", "objects": [], "heard": "again", "gaze": [0.0, 0.0]},
        {"type": "done", "reply": ""},
    ]
    player = FakePlayer()
    drive_robot_stream(FlakyRobot(), events, make_player=lambda sr: player)
    out = capsys.readouterr().out
    assert out.count("[robot] skip") == 1


def test_robot_guard_healthy_state_persists_across_drive_calls():
    # "per-robot", not per-response: the same guard passed to two separate
    # drive_robot_stream calls (two consecutive responses) shouldn't print
    # again on a repeated failure of the same robot.
    from demo.run_demo import _FailureGuard, drive_robot_stream

    class AlwaysFails:
        def look_at(self, x, y):
            raise RuntimeError("dead")
        def gesture(self, name):
            pass
        def say(self, *a):
            pass

    guard = _FailureGuard("robot")
    robot = AlwaysFails()
    events = [{"type": "meta", "objects": [], "heard": "a", "gaze": [0.0, 0.0]},
             {"type": "done", "reply": ""}]

    drive_robot_stream(robot, events, make_player=lambda sr: FakePlayer(),
                       robot_guard=guard)
    assert guard.healthy is False

    drive_robot_stream(robot, list(events), make_player=lambda sr: FakePlayer(),
                       robot_guard=guard)
    assert guard.healthy is False  # same guard — state wasn't reset


def test_drive_robot_stream_audio_failure_does_not_crash_loop():
    # The audio path is under the same protection as the robot: a broken
    # player shouldn't kill the rest of the response.
    import base64
    from demo.run_demo import drive_robot_stream

    class BrokenPlayer:
        def __init__(self, sr):
            pass
        def feed(self, samples):
            raise RuntimeError("afplay missing")
        def close(self):
            raise RuntimeError("afplay missing")

    a = np.zeros(400, dtype=np.float32)
    audio_b64 = base64.b64encode(a.tobytes()).decode()
    events = [
        {"type": "meta", "objects": [], "heard": "hi", "gaze": [0.0, 0.0]},
        {"type": "audio", "text": "hi", "audio_b64": audio_b64,
         "sample_rate": 16000},
        {"type": "done", "reply": "hi"},
    ]
    done = drive_robot_stream(FakeRobot(), events,
                              make_player=lambda sr: BrokenPlayer(sr))
    assert done["reply"] == "hi"


# --- _handle_stream: `detections is not None`, not truthiness ---







# --- RobotDispatcher: robot actuation must never block the event loop that
# drives speech. ---

def test_robot_dispatcher_returns_immediately_and_still_runs_the_call():
    import threading
    import time

    from demo.run_demo import RobotDispatcher, _FailureGuard

    started = threading.Event()
    release = threading.Event()
    calls = []

    def slow_call(label):
        started.set()
        release.wait(timeout=2.0)
        calls.append(label)

    dispatcher = RobotDispatcher(_FailureGuard("robot"))
    try:
        t0 = time.monotonic()
        dispatcher.dispatch(slow_call, "first")
        assert started.wait(timeout=1.0), "worker never picked up the call"
        elapsed = time.monotonic() - t0
        assert elapsed < 0.5, "dispatch() must not wait for the robot call"
        release.set()
    finally:
        dispatcher.close()
    assert calls == ["first"]


def test_robot_dispatcher_drops_stale_motion_for_the_latest_call():
    # "dropping stale motion rather than queueing it up": a call that hasn't
    # started yet gets replaced by a newer one instead of both running.
    import threading

    from demo.run_demo import RobotDispatcher, _FailureGuard

    block_started = threading.Event()
    release = threading.Event()
    stale2_done = threading.Event()
    calls = []

    def action(label):
        if label == "block":
            block_started.set()
            release.wait(timeout=2.0)
        calls.append(label)
        if label == "stale-2":
            stale2_done.set()

    dispatcher = RobotDispatcher(_FailureGuard("robot"))
    try:
        dispatcher.dispatch(action, "block")
        assert block_started.wait(timeout=1.0)  # worker is now busy on "block"
        dispatcher.dispatch(action, "stale-1")
        dispatcher.dispatch(action, "stale-2")  # replaces stale-1 before it runs
        release.set()
        assert stale2_done.wait(timeout=1.0), "the latest pending call never ran"
    finally:
        dispatcher.close()
    assert calls == ["block", "stale-2"], "only the latest pending call should survive"


def test_robot_dispatcher_does_not_let_one_kind_evict_a_different_kind():
    # Regression: a queue.Queue(maxsize=1) shared across ALL kinds meant a
    # `dance` landing right after an `express` silently evicted the express
    # before it ever ran — the model announced an emotion the robot never
    # performed. Different kinds (methods) must each get their own slot.
    import threading

    from demo.run_demo import RobotDispatcher, _FailureGuard

    block_started = threading.Event()
    release = threading.Event()
    dance_done = threading.Event()

    class Robot:
        def __init__(self):
            self.calls = []

        def look_at(self, x, y):
            block_started.set()
            release.wait(timeout=2.0)
            self.calls.append(("look_at", x, y))

        def express(self, emotion):
            self.calls.append(("express", emotion))

        def dance(self):
            self.calls.append(("dance",))
            dance_done.set()

    robot = Robot()
    dispatcher = RobotDispatcher(_FailureGuard("robot"))
    try:
        # Occupy the worker with a slow look_at so express/dance both queue
        # up behind it as PENDING, not yet started.
        dispatcher.dispatch(robot.look_at, 0.0, 0.0)
        assert block_started.wait(timeout=1.0)
        dispatcher.dispatch(robot.express, "happy")
        dispatcher.dispatch(robot.dance)
        release.set()
        assert dance_done.wait(timeout=1.0), "dance never ran"
    finally:
        dispatcher.close()
    assert ("express", "happy") in robot.calls, (
        "express must not be evicted by a different-kind dance call")
    assert ("dance",) in robot.calls


def test_robot_dispatcher_logs_failure_through_the_shared_guard():
    # The worker thread must still route calls through the SAME _FailureGuard
    # the session already holds, so the unhealthy transition is logged once,
    # just from a different thread.
    from demo.run_demo import RobotDispatcher, _FailureGuard

    def boom():
        raise RuntimeError("disconnected")

    guard = _FailureGuard("robot")
    dispatcher = RobotDispatcher(guard)
    dispatcher.dispatch(boom)
    dispatcher.close()  # join()s the worker thread -> the call has finished
    assert guard.healthy is False


# --- AsyncSceneWriter: SceneChangeWriter.observe() used to run
# INLINE on RemoteDetectSource's detect thread, so a slow FrameMemory.remember()
# (RemoteSiglipEmbedder's network embed) froze detection itself — no new
# frame/detections, and display.on_detections (the other listener) stopped
# firing too. Mirrors RobotDispatcher's own tests above.

def test_async_scene_writer_returns_immediately_and_still_calls_observe():
    import threading
    import time

    from demo.run_demo import AsyncSceneWriter

    started = threading.Event()
    release = threading.Event()
    calls = []

    class SlowWriter:
        def observe(self, frame_rgb, detections):
            started.set()
            release.wait(timeout=2.0)
            calls.append((frame_rgb, detections))

    writer = AsyncSceneWriter(SlowWriter())
    try:
        t0 = time.monotonic()
        writer.observe("frame-1", ["det-1"])
        assert started.wait(timeout=1.0), "worker never picked up the call"
        elapsed = time.monotonic() - t0
        assert elapsed < 0.5, "observe() must not wait for the memory write"
        release.set()
    finally:
        writer.close()
    assert calls == [("frame-1", ["det-1"])]


def test_async_scene_writer_drops_stale_pending_for_the_latest_call():
    # Same "drop stale work rather than queue it" reasoning as
    # RobotDispatcher's per-kind slots: only the latest scene matters.
    import threading

    from demo.run_demo import AsyncSceneWriter

    block_started = threading.Event()
    release = threading.Event()
    last_done = threading.Event()
    calls = []

    class SlowWriter:
        def observe(self, frame_rgb, detections):
            if frame_rgb == "block":
                block_started.set()
                release.wait(timeout=2.0)
            calls.append(frame_rgb)
            if frame_rgb == "stale-2":
                last_done.set()

    writer = AsyncSceneWriter(SlowWriter())
    try:
        writer.observe("block", [])
        assert block_started.wait(timeout=1.0)  # worker is now busy
        writer.observe("stale-1", [])
        writer.observe("stale-2", [])  # replaces stale-1 before it runs
        release.set()
        assert last_done.wait(timeout=1.0), "the latest pending call never ran"
    finally:
        writer.close()
    assert calls == ["block", "stale-2"], "only the latest pending call should survive"


def test_async_scene_writer_survives_a_failing_observe(capsys):
    # A bad write must not kill the worker thread — the next dispatch still
    # has to run (mirrors RemoteDetectSource._cycle's own listener guard).
    from demo.run_demo import AsyncSceneWriter

    class FlakyWriter:
        def __init__(self):
            self.calls = []

        def observe(self, frame_rgb, detections):
            if frame_rgb == "boom":
                raise RuntimeError("embed_service unreachable")
            self.calls.append(frame_rgb)

    flaky = FlakyWriter()
    writer = AsyncSceneWriter(flaky)
    try:
        writer.observe("boom", [])
        writer.close()  # join()s the worker -> the failing call has finished
    finally:
        pass
    assert "embed_service unreachable" in capsys.readouterr().out

    writer2 = AsyncSceneWriter(flaky)
    try:
        writer2.observe("frame-2", [])
    finally:
        writer2.close()
    assert flaky.calls == ["frame-2"]


def test_drive_robot_stream_dispatches_robot_calls_instead_of_calling_directly():
    # With a dispatcher present, robot calls must go through it — never run
    # synchronously on drive_robot_stream's own thread.
    from demo.run_demo import drive_robot_stream

    class _RecordingDispatcher:
        def __init__(self):
            self.dispatched = []

        def dispatch(self, action, *args):
            self.dispatched.append((action, args))

    robot = FakeRobot()
    dispatcher = _RecordingDispatcher()
    events = [
        {"type": "meta", "objects": [], "heard": "hello",
         "gaze": [0.3, -0.1]},
        {"type": "done", "reply": ""},
    ]
    drive_robot_stream(robot, events, make_player=lambda sr: FakePlayer(),
                       robot_dispatcher=dispatcher)

    assert robot.calls == [], "robot calls must not run on this thread"
    assert len(dispatcher.dispatched) == 1
    for action, args in dispatcher.dispatched:
        action(*args)  # run what the worker thread would have run
    assert [c[0] for c in robot.calls] == ["look_at"]


# --- build_memories: on-disk persistence, separate paths per memory ---
# No real Qdrant/model loads — FrameMemory/
# TextMemory are monkeypatched to tiny capturing fakes.

class _CapturingMemory:
    def __init__(self, path=None, **kwargs):
        self.path = path

    def remember(self, *a, **kw):
        pass

    def recall(self, *a, **kw):
        return []


def test_build_memories_share_one_memory_shard_on_disk(monkeypatch, tmp_path):
    # The conversation and the frames live in one shard, `memory`, with named
    # vectors — an Edge shard locks its directory, so it is opened once and
    # handed to both.
    from types import SimpleNamespace

    import demo.run_demo as run_demo_mod

    seen = {}

    class CapturingFrameMemory(_CapturingMemory):
        def __init__(self, path=None, **kw):
            super().__init__(path, **kw)
            seen["frames"] = kw["store"]

    class CapturingTextMemory(_CapturingMemory):
        def __init__(self, path=None, **kw):
            super().__init__(path, **kw)
            seen["speech"] = kw["store"]

    monkeypatch.setattr("emulator.frame_memory.FrameMemory", CapturingFrameMemory)
    monkeypatch.setattr("emulator.memory.TextMemory", CapturingTextMemory)

    args = SimpleNamespace(no_memory=False, memory_dir=str(tmp_path / "mem"))
    run_demo_mod.build_memories(args)

    assert seen["frames"] is seen["speech"]
    assert seen["frames"].path == str(tmp_path / "mem" / "memory")
    assert (seen["frames"].size("text"), seen["frames"].size("image")) == (384, 768)


def test_build_memories_memory_colon_sentinel_stays_ephemeral(monkeypatch):
    import demo.run_demo as run_demo_mod
    from types import SimpleNamespace

    seen = {}

    class CapturingFrameMemory(_CapturingMemory):
        def __init__(self, path=None, **kw):
            super().__init__(path, **kw)
            seen["frame_path"] = path

    class CapturingTextMemory(_CapturingMemory):
        def __init__(self, path=None, **kw):
            super().__init__(path, **kw)
            seen["speech_path"] = path

    monkeypatch.setattr("emulator.frame_memory.FrameMemory", CapturingFrameMemory)
    monkeypatch.setattr("emulator.memory.TextMemory", CapturingTextMemory)

    args = SimpleNamespace(no_memory=False, memory_dir=":memory:")
    run_demo_mod.build_memories(args)

    assert seen["frame_path"] is None
    assert seen["speech_path"] is None


def test_build_memories_no_memory_flag_skips_both_regardless_of_dir():
    from types import SimpleNamespace

    from demo.run_demo import build_memories

    args = SimpleNamespace(no_memory=True, memory_dir="/should/not/be/used")
    assert build_memories(args) == (None, None)


def test_parse_args_memory_dir_defaults_to_a_persistent_path():
    from demo.run_demo import parse_args

    args = parse_args([])
    assert args.memory_dir != ":memory:"


def test_parse_args_memory_dir_accepts_ephemeral_sentinel():
    from demo.run_demo import parse_args

    args = parse_args(["--memory-dir", ":memory:"])
    assert args.memory_dir == ":memory:"


# --- --robot-host: expose the daemon host as a CLI flag ---

def test_parse_args_robot_host_defaults_to_mdns_name():
    from demo.run_demo import parse_args

    args = parse_args([])
    assert args.robot_host == "reachy-mini.local"


def test_parse_args_robot_host_accepts_a_plain_ip():
    from demo.run_demo import parse_args

    args = parse_args(["--robot-host", "192.168.1.50"])
    assert args.robot_host == "192.168.1.50"


# --- --camera: video source choice (Mac webcam vs the robot's own camera) ---
# The hard requirement is that what gets remembered is what the ROBOT saw,
# so this must be a configuration choice reachable from the CLI, not a
# rewrite — see demo/platform/mac.py::MacPlatform.video_source().

def test_parse_args_camera_defaults_to_mac():
    from demo.run_demo import parse_args

    args = parse_args([])
    assert args.camera == "mac"


def test_parse_args_camera_accepts_robot():
    from demo.run_demo import parse_args

    args = parse_args(["--camera", "robot"])
    assert args.camera == "robot"


def test_parse_args_camera_rejects_unknown_choice():
    from demo.run_demo import parse_args

    with pytest.raises(SystemExit):
        parse_args(["--camera", "webcam"])


def test_parse_args_robot_camera_port_has_a_default():
    from demo.run_demo import parse_args

    args = parse_args([])
    # The service's own default, not a literal — a second copy is how the
    # client and service came to disagree on the port in the first place.
    from demo.camera_service import DEFAULT_PORT
    assert args.robot_camera_port == DEFAULT_PORT


def test_parse_args_robot_camera_port_accepts_override():
    from demo.run_demo import parse_args

    args = parse_args(["--robot-camera-port", "9999"])
    assert args.robot_camera_port == 9999


# --- --mic / --speaker: audio I/O choice (Mac vs the robot's own hardware) ---
# The hard requirement is that the ROBOT is the thing talked TO and the
# thing that talks BACK, so both must be configuration reachable from the
# CLI, not a rewrite — see demo/platform/mac.py's mic_source()/make_player().

def test_parse_args_mic_defaults_to_mac():
    from demo.run_demo import parse_args

    assert parse_args([]).mic == "mac"


def test_parse_args_mic_accepts_robot():
    from demo.run_demo import parse_args

    assert parse_args(["--mic", "robot"]).mic == "robot"


def test_parse_args_mic_rejects_unknown_choice():
    from demo.run_demo import parse_args

    with pytest.raises(SystemExit):
        parse_args(["--mic", "webcam"])


def test_parse_args_speaker_defaults_to_mac():
    from demo.run_demo import parse_args

    assert parse_args([]).speaker == "mac"


def test_parse_args_speaker_accepts_robot():
    from demo.run_demo import parse_args

    assert parse_args(["--speaker", "robot"]).speaker == "robot"


def test_parse_args_speaker_rejects_unknown_choice():
    from demo.run_demo import parse_args

    with pytest.raises(SystemExit):
        parse_args(["--speaker", "webcam"])


def test_main_rejects_speaker_robot_with_no_robot(capsys):
    import demo.run_demo as run_demo_mod

    rc = run_demo_mod.main(["--speaker", "robot", "--no-robot"])

    assert rc == 1
    assert "speaker robot" in capsys.readouterr().out.lower()


# --- --brain: the laptop running the model services ---

def test_parse_args_brain_defaults_to_this_machine():
    from demo.run_demo import parse_args

    assert parse_args([]).brain == "127.0.0.1"


def test_main_builds_the_endpoint_from_the_brain_flag(monkeypatch):
    import demo.run_demo as run_demo_mod

    captured = {}
    monkeypatch.setattr(run_demo_mod, "run_voice",
                        lambda args, endpoint: captured.update(endpoint=endpoint) or 0)
    run_demo_mod.main(["--no-robot", "--brain", "10.0.0.9", "--port", "9500"])
    assert captured["endpoint"] == "http://10.0.0.9:9500"


def test_parse_args_embed_port_has_a_default():
    from demo.embed_service import DEFAULT_PORT
    from demo.run_demo import parse_args

    assert parse_args([]).embed_port == DEFAULT_PORT


def test_parse_args_embed_port_accepts_override():
    from demo.run_demo import parse_args

    assert parse_args(["--embed-port", "9999"]).embed_port == 9999


# --- --display remote / --dashboard-host: the
# audience's screen stays on the Mac while the voice loop runs on the robot.

def test_parse_args_display_accepts_remote():
    from demo.run_demo import parse_args

    assert parse_args(["--display", "remote"]).display == "remote"


def test_parse_args_display_rejects_unknown_choice():
    from demo.run_demo import parse_args

    with pytest.raises(SystemExit):
        parse_args(["--display", "tui"])


def test_parse_args_dashboard_host_defaults_to_none():
    """None, not --brain's value: run_voice resolves the fallback itself
    (`dashboard_host or args.brain`) so parse_args stays a pure reflection of
    what was actually passed on the command line."""
    from demo.run_demo import parse_args

    assert parse_args([]).dashboard_host is None


def test_parse_args_dashboard_host_accepts_override():
    from demo.run_demo import parse_args

    assert parse_args(["--dashboard-host", "mac.local"]).dashboard_host == "mac.local"


# --- _startup_display_message: run_voice used to print the
# LOOPBACK web-UI url for "remote" too, even though nothing is bound locally
# in that mode (build_display returns a RemoteDisplayClient that only
# pushes) — the operator's confirmation that voice-start finished ended with
# a URL that served nothing. Pulled into its own function so the branch on
# args.display is testable without running the rest of run_voice.

def test_startup_display_message_web_prints_the_local_loopback_url():
    from types import SimpleNamespace

    from demo.run_demo import _startup_display_message

    args = SimpleNamespace(display="web", web_port=8080)
    assert _startup_display_message(args) == "  web UI: http://127.0.0.1:8080"


def test_startup_display_message_none_prints_nothing():
    from types import SimpleNamespace

    from demo.run_demo import _startup_display_message

    args = SimpleNamespace(display="none", web_port=8080)
    assert _startup_display_message(args) is None


def test_startup_display_message_remote_prints_the_pushed_dashboard_url():
    from types import SimpleNamespace

    from demo.run_demo import _startup_display_message

    args = SimpleNamespace(display="remote", web_port=8080,
                           dashboard_host="mac.local", brain="laptop.local")
    message = _startup_display_message(args)

    assert "127.0.0.1" not in message, "nothing is bound locally under --display remote"
    assert "http://mac.local:8080" in message


def test_startup_display_message_remote_falls_back_to_brain_host():
    """No --dashboard-host: run_voice's own fallback (dashboard_host or
    args.brain), matching build_display's actual resolution in run_voice."""
    from types import SimpleNamespace

    from demo.run_demo import _startup_display_message

    args = SimpleNamespace(display="remote", web_port=8080,
                           dashboard_host=None, brain="laptop.local")
    message = _startup_display_message(args)

    assert "http://laptop.local:8080" in message


# --- build_memories: REMOTE embedders — the
# robot's memories must never load SigLIP/fastembed locally; they call
# demo/embed_service.py on --brain:--embed-port instead
# (demo/embed_client.py). Constructing a RemoteSiglipEmbedder/RemoteBgeEmbedder
# does no I/O (see that module's docstring), so this needs no network fake —
# only what got PASSED to FrameMemory/TextMemory is under test here.

def test_build_memories_wires_remote_embedders_addressed_at_brain_and_embed_port(monkeypatch):
    from types import SimpleNamespace

    import demo.run_demo as run_demo_mod
    from demo.embed_client import RemoteBgeEmbedder, RemoteSiglipEmbedder

    seen = {}

    class CapturingFrameMemory(_CapturingMemory):
        def __init__(self, path=None, **kw):
            super().__init__(path, **kw)
            seen["frame_embedder"] = kw["embedder"]

    class CapturingTextMemory(_CapturingMemory):
        def __init__(self, path=None, **kw):
            super().__init__(path, **kw)
            seen["speech_embedder"] = kw["embedder"]

    monkeypatch.setattr("emulator.frame_memory.FrameMemory", CapturingFrameMemory)
    monkeypatch.setattr("emulator.memory.TextMemory", CapturingTextMemory)

    args = SimpleNamespace(no_memory=False, memory_dir=":memory:",
                           brain="mac.local", embed_port=9901)
    run_demo_mod.build_memories(args)

    assert isinstance(seen["frame_embedder"], RemoteSiglipEmbedder)
    assert seen["frame_embedder"]._base == "http://mac.local:9901"
    assert isinstance(seen["speech_embedder"], RemoteBgeEmbedder)
    assert seen["speech_embedder"]._base == "http://mac.local:9901"


def test_build_memories_defaults_the_embedder_host_for_bare_namespaces(monkeypatch):
    """A bare SimpleNamespace with neither `brain` nor `embed_port` (see
    test_build_memories_uses_separate_persistent_subdirs above) must not
    raise AttributeError — same getattr-with-fallback shape as --camera/
    --robot-host elsewhere in this file."""
    from types import SimpleNamespace

    import demo.run_demo as run_demo_mod
    from demo.embed_service import DEFAULT_PORT as EMBED_SERVICE_PORT

    seen = {}

    class CapturingFrameMemory(_CapturingMemory):
        def __init__(self, path=None, **kw):
            super().__init__(path, **kw)
            seen["frame_embedder"] = kw["embedder"]

    monkeypatch.setattr("emulator.frame_memory.FrameMemory", CapturingFrameMemory)
    monkeypatch.setattr("emulator.memory.TextMemory", _CapturingMemory)

    run_demo_mod.build_memories(SimpleNamespace(no_memory=False, memory_dir=":memory:"))

    assert (seen["frame_embedder"]._base
           == f"http://127.0.0.1:{EMBED_SERVICE_PORT}")


# --- _handle_stream through /chat (demo/conversation.py): the model keeps
# the conversation and reaches for memory only by calling a tool
# — see the "A turn is" note above demo.run_demo.transcribe ---

import base64  # noqa: E402

from demo.contract import decode_chat_request  # noqa: E402

ENDPOINT = "http://brain:9500"


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode()


def _chat_script(*responses, seen=None):
    """Monkeypatch target for demo.run_demo._http_stream standing in for
    serve.py's /chat: each request gets the next scripted list of events;
    `seen` collects (endpoint, decoded request)."""
    queue = list(responses)

    def fake(endpoint, payload):
        if seen is not None:
            seen.append((endpoint, decode_chat_request(payload)))
        return iter(queue.pop(0))
    return fake


def _tool(name, **arguments):
    call = {"name": name, "arguments": arguments}
    return [{"type": "tool_call", **call},
            {"type": "done", "reply": "", "tool_call": call, "token_count": 280}]


def _said(reply, token_count=300):
    audio = _b64(np.zeros(400, np.float32).tobytes())
    return [{"type": "audio", "text": reply, "audio_b64": audio, "sample_rate": 16000},
            {"type": "done", "reply": reply, "first_sound_ms": 90,
             "token_count": token_count}]


def _turn(monkeypatch, heard, *responses, seen=None, detections=(), audio=None,
          display=None, robot=None, **kwargs):
    from demo.run_demo import _handle_stream

    monkeypatch.setattr("demo.run_demo._http_post", _fake_transcribe(heard))
    monkeypatch.setattr("demo.run_demo._http_stream",
                        _chat_script(*responses, seen=seen) if responses
                        else _no_generate_http_stream)
    monkeypatch.setattr("demo.audio_out.StreamPlayer", FakePlayer)
    display = display if display is not None else DisplaySpy()
    _handle_stream(list(detections), audio if audio is not None
                   else np.zeros(8000, np.float32), ENDPOINT,
                   robot if robot is not None else FakeRobot(), display, **kwargs)
    return display


def test_handle_stream_drives_display_spy(monkeypatch):
    seen, robot = [], FakeRobot()
    display = _turn(monkeypatch, "hi", _said("Hi there."), seen=seen, robot=robot,
                    detections=[{"label": "cup", "score": 0.9, "box": [0.1, 0.1, 0.2, 0.2]}])
    assert display.heard == ["hi"]
    assert display.replies == [("Hi there.", False), ("Hi there.", True)]
    assert seen[0][0] == "http://brain:9500/chat"
    assert (seen[0][1].history, seen[0][1].text) == ([], "hi")
    # the head still turns towards what it sees, before it speaks
    assert robot.calls[0][0] == "look_at"


def test_handle_stream_stays_quiet_on_transcription_failure(monkeypatch):
    # Failure of transcription must not kill the voice loop: the robot stays
    # quiet and listens again — no chat call, robot untouched. The dashboard
    # IS cleared, though: the previous question's heard line, reply and recall
    # next to a robot that just went silent read as an answer to what was said.
    from demo.run_demo import _handle_stream

    def broken_post(endpoint, payload):
        raise OSError("connection refused")

    monkeypatch.setattr("demo.run_demo._http_post", broken_post)
    monkeypatch.setattr("demo.run_demo._http_stream", _no_generate_http_stream)
    display, robot = DisplaySpy(), FakeRobot()
    _handle_stream([], np.zeros(8000, np.float32), ENDPOINT, robot, display)
    assert display.heard == [""]
    assert display.recalls == [[]] and display.speech_recalls == [[]]
    assert robot.calls == []


def test_handle_stream_transcription_failure_survives_a_bad_json_response(monkeypatch):
    # A malformed response (not a dict) is a network failure, not a crash.
    from demo.run_demo import _handle_stream

    monkeypatch.setattr("demo.run_demo._http_post",
                        lambda endpoint, payload: ["not", "a", "dict"])
    monkeypatch.setattr("demo.run_demo._http_stream", _no_generate_http_stream)
    display = DisplaySpy()
    _handle_stream([], np.zeros(8000, np.float32), ENDPOINT, FakeRobot(), display)
    assert display.heard == [""]


def test_handle_stream_carries_the_conversation_and_adds_this_turn(monkeypatch):
    from demo.conversation import ConversationWindow

    seen, window = [], ConversationWindow()
    window.add("Hi, I'm Sasha.", "Hello Sasha!")
    _turn(monkeypatch, "What's my name?", _said("Your name is Sasha."), seen=seen,
          conversation=window)
    assert seen[0][1].history == [("Hi, I'm Sasha.", "Hello Sasha!")]
    assert window.history[-1] == ("What's my name?", "Your name is Sasha.")


def test_handle_stream_does_not_search_memory_unless_the_model_asks(monkeypatch):
    fm = _FakeFrameMemory([{"jpeg_b64": _b64(b"x"), "detections": [], "score": 0.2}])
    display = _turn(monkeypatch, "How are you?", _said("Great!"), frame_memory=fm)
    assert fm.queries == []
    assert display.recalls == [[]]  # cleared for the new question, nothing more


def test_handle_stream_shows_visual_recall_when_the_model_calls_recall_seen(monkeypatch):
    # The tool's older name is still answered, as a question about what was
    # seen: the frame whose words hold it goes to the model and the screen.
    hits = [{"jpeg_b64": _b64(b"MUG"),
             "detections": [{"label": "cup", "box": [0, 0, 1, 1], "score": 0.8}],
             "score": 0.78}]
    fm, seen = _FakeFrameMemory(hits), []
    display = _turn(monkeypatch, "did you see a red mug?",
                    _tool("recall_seen", query="a red mug"), _said("A red mug."),
                    seen=seen, frame_memory=fm)
    assert fm.queries == ["a red mug"]
    assert display.recalls == [[], [{**hits[0], "weak": False}]]
    # The call is shown and run exactly as the model made it: nothing is
    # added to it from the words of the question.
    assert display.tool_calls == [("recall_seen", {"query": "a red mug"})]
    # the recalled frame goes to the model with the follow-up request
    assert seen[1][1].image_jpeg == b"MUG"
    assert seen[1][1].text == "did you see a red mug?"


def test_handle_stream_look_shows_the_model_this_turns_frame(monkeypatch):
    seen = []
    frame = np.full((8, 8, 3), 200, dtype=np.uint8)
    _turn(monkeypatch, "What do you see?", _tool("camera"), _said("A bright wall."),
          seen=seen, frame=frame)
    assert seen[1][1].image_jpeg.startswith(b"\xff\xd8")  # a JPEG of `frame`
    assert seen[1][1].image_note == "(Your camera, right now.)"


def test_a_turn_searches_what_was_seen_but_does_not_store_its_own_frame(monkeypatch):
    # Frames are kept when the objects in view change and when the robot is
    # asked to look — not once per turn, which filled memory with the same
    # picture of the presenter.
    this_turn_frame = np.zeros((2, 2, 3), dtype=np.uint8)
    dets = [{"label": "cup", "score": 0.9, "box": [0, 0, 1, 1]}]
    fm = _FakeFrameMemory([{"jpeg_b64": _b64(b"old"), "detections": [], "score": 0.2}])
    _turn(monkeypatch, "what did you see?", _tool("remember", query="seen", about="seen"),
          _said("A cup."), frame_memory=fm, frame=this_turn_frame, detections=dets)
    assert fm.calls == ["recall"]
    assert fm.remembered == []


def test_an_empty_turn_stores_no_frame(monkeypatch):
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    fm = _FakeFrameMemory([])
    _turn(monkeypatch, "", frame_memory=fm, frame=frame)
    assert fm.calls == []


def test_a_turn_without_its_memories_tells_the_model_they_are_off(monkeypatch):
    from demo.conversation import MEMORY_OFF_NOTE

    seen = []
    _turn(monkeypatch, "Do you remember my dog?",
          _tool("remember", query="my dog", about="said"), _said("It is off."),
          seen=seen, frame_memory=None, speech_memory=None)
    assert seen[1][1].tool_result["result"] == {"note": MEMORY_OFF_NOTE}


def test_a_dropped_brain_does_not_break_the_turn(monkeypatch):
    from demo.run_demo import _handle_stream

    def dropped(endpoint, payload):
        raise OSError("brain unreachable")

    monkeypatch.setattr("demo.run_demo._http_post", _fake_transcribe("hello"))
    monkeypatch.setattr("demo.run_demo._http_stream", dropped)
    frame, fm = np.zeros((2, 2, 3), dtype=np.uint8), _FakeFrameMemory([])
    _handle_stream([], np.zeros(8000, np.float32), ENDPOINT, FakeRobot(),
                   DisplaySpy(), frame_memory=fm, frame=frame)  # does not raise
    assert fm.calls == []


# --- turn_started_at: RECALL BEFORE STORE only stops THIS turn's own frame;
# SceneChangeWriter (emulator/frame_memory.py) stores independently on the
# detect thread and can beat it — frames it wrote during this very turn are
# the present, and the recall tool must drop them (demo/conversation.py). ---

def _recall_frames(monkeypatch, hits, day=(), **kwargs):
    display = _turn(monkeypatch, "did you see the bottle?",
                    _tool("remember", query="the bottle", about="seen"),
                    _said("ok"), frame_memory=_FakeFrameMemory(hits, day), **kwargs)
    # `weak` is the projector's flag for a dimmed match (never set now that
    # nothing guesses); these tests are about WHICH frames come back.
    return [{key: value for key, value in frame.items() if key != "weak"}
            for frame in display.recalls[-1]]


def test_handle_stream_excludes_frame_memory_hits_stored_during_this_turn(monkeypatch):
    past_hit = {"jpeg_b64": _b64(b"OLD"), "detections": [], "score": 0.5, "ts": 50.0}
    concurrent_hit = {"jpeg_b64": _b64(b"NEW"), "detections": [], "score": 0.9,
                      "ts": 100.5}
    assert _recall_frames(monkeypatch, [concurrent_hit, past_hit],
                          turn_started_at=100.0) == [past_hit]



def test_the_days_frames_leave_out_the_frame_stored_during_this_turn(monkeypatch):
    # The day starts from its newest frame: without the turn's start, the
    # frame the scene writer stored while the question was being asked would
    # be on the screen as a memory every time.
    past = {"jpeg_b64": _b64(b"OLD"), "detections": [], "score": 0.0, "ts": 50.0}
    concurrent = {"jpeg_b64": _b64(b"NEW"), "detections": [], "score": 0.0, "ts": 100.5}
    assert _recall_frames(monkeypatch, [], day=[concurrent, past],
                          turn_started_at=100.0) == [past]


def test_handle_stream_keeps_frame_memory_hits_when_turn_started_at_is_none(monkeypatch):
    hits = [{"jpeg_b64": _b64(b"NEW"), "detections": [], "score": 0.9, "ts": 1e12}]
    assert _recall_frames(monkeypatch, hits) == hits


def test_turn_started_at_does_not_exclude_a_frame_from_the_idle_wait(monkeypatch):
    # A frame stored while nobody was speaking yet (deep in the idle wait
    # collect_utterance blocked on) is a genuine PAST memory; only frames
    # stored WHILE the utterance was spoken are excluded.
    from demo.run_demo import _turn_started_at

    audio = np.zeros(32000, np.float32)  # 2.0 s utterance, ending at t=102
    idle_wait_hit = {"jpeg_b64": _b64(b"OLD"), "detections": [], "score": 0.5,
                     "ts": 50.0}
    during_utterance_hit = {"jpeg_b64": _b64(b"NEW"), "detections": [],
                            "score": 0.9, "ts": 99.0}
    assert _recall_frames(monkeypatch, [during_utterance_hit, idle_wait_hit],
                          audio=audio,
                          turn_started_at=_turn_started_at(audio, now=102.0)
                          ) == [idle_wait_hit]


def test_handle_stream_recalls_object_seen_before_a_short_question(monkeypatch):
    # Stage scenario 1: object seen 4.0s ago, 2.0s question -> recalled.
    from demo.run_demo import _turn_started_at

    audio = np.zeros(int(2.0 * 16000), np.float32)
    bottle = {"jpeg_b64": _b64(b"BOTTLE"), "detections": [], "score": 0.5, "ts": 96.0}
    assert _recall_frames(monkeypatch, [bottle], audio=audio,
                          turn_started_at=_turn_started_at(audio, now=100.0)
                          ) == [bottle]


def test_handle_stream_recalls_object_seen_before_a_long_natural_question(monkeypatch):
    # Stage scenario 2: object seen 6.0s ago, 5.0s question -> still recalled
    # (TURN_DURATION_CAP_S), not dropped because the question took 5s to say.
    from demo.run_demo import _turn_started_at

    audio = np.zeros(int(5.0 * 16000), np.float32)
    bottle = {"jpeg_b64": _b64(b"BOTTLE"), "detections": [], "score": 0.5, "ts": 94.0}
    assert _recall_frames(monkeypatch, [bottle], audio=audio,
                          turn_started_at=_turn_started_at(audio, now=100.0)
                          ) == [bottle]


def test_handle_stream_still_excludes_current_view_on_a_long_question(monkeypatch):
    # The cap must not weaken the invariant it sits on: stored WHILE this long
    # question was asked (inside the capped+guarded window) is the present.
    from demo.run_demo import _turn_started_at

    audio = np.zeros(int(5.0 * 16000), np.float32)
    current_view = {"jpeg_b64": _b64(b"NOW"), "detections": [], "score": 0.9,
                    "ts": 96.0}
    assert _recall_frames(monkeypatch, [current_view], audio=audio,
                          turn_started_at=_turn_started_at(audio, now=100.0)) == []


def test_handle_stream_clears_recall_panels_on_empty_heard(monkeypatch):
    # A false VAD trigger (cough, applause) must not leave the PREVIOUS
    # question's recall on screen next to a blank Heard line.
    display = _turn(monkeypatch, "", frame_memory=_FakeFrameMemory([]),
                    frame=np.zeros((2, 2, 3), dtype=np.uint8))
    assert display.recalls == [[]] and display.speech_recalls == [[]]


def test_handle_stream_no_recall_on_empty_heard(monkeypatch):
    fm = _FakeFrameMemory([{"jpeg_b64": _b64(b"x"), "detections": [], "score": 0.2}])
    _turn(monkeypatch, "", frame_memory=fm)
    assert fm.queries == []


def test_handle_stream_moves_old_exchanges_into_speech_memory_over_budget(monkeypatch):
    from demo.conversation import ConversationWindow

    sm = _FakeSpeechMemory([])
    window = ConversationWindow(sm, budget_tokens=250)
    window.add("Hi, I'm Sasha.", "Hello Sasha!")
    window.add("I'm giving a talk.", "Exciting!")
    display = _turn(monkeypatch, "Can you nod?", _said("Sure!", token_count=400),
                    conversation=window)
    assert sm.remembered == [("exchange", "Person: Hi, I'm Sasha. — Reachy: Hello Sasha!")]
    assert display.memory_writes == [["Person: Hi, I'm Sasha. — Reachy: Hello Sasha!"]]
    assert window.history == [("I'm giving a talk.", "Exciting!"), ("Can you nod?", "Sure!")]


@pytest.mark.parametrize("heard", ["♪♪", " ♪ ", "..."])
def test_handle_stream_ignores_a_transcript_with_no_words(monkeypatch, heard):
    display = _turn(monkeypatch, heard)  # _no_generate_http_stream: no chat call
    assert display.heard == [""]


@pytest.mark.parametrize("heard", ["You", "Thanks for watching!", "Thank you.", "42.",
                                   "You look happy!", "", "What do you see?"])
def test_words_are_never_judged_to_be_noise_by_what_they_say(heard):
    # Noise is turned away before it is a transcript, by what the recognisers
    # can tell about the audio (emulator/speech_detector.py, whisper_asr.py's
    # NO_SPEECH_MAX); no list of words decides it here.
    from demo.run_demo import _holds_no_words

    assert not _holds_no_words(heard)


def test_parse_args_context_budget_default_and_override():
    from demo.conversation import DEFAULT_CONTEXT_BUDGET
    from demo.run_demo import parse_args

    assert parse_args([]).context_budget == DEFAULT_CONTEXT_BUDGET
    assert parse_args(["--context-budget", "2000"]).context_budget == 2000


def test_drive_robot_stream_reports_a_brain_error_line(capsys):
    from demo.run_demo import drive_robot_stream

    done = drive_robot_stream(FakeRobot(), iter([{"type": "error", "message": "boom"}]))
    assert done == {}
    assert "[brain] error: boom" in capsys.readouterr().out


def test_the_voice_loop_imports_without_the_model_runtimes():
    # The robot's Python has numpy, PIL and qdrant-edge-py, not the LiteRT
    # stack the Mac-side models need. An import that reaches it breaks
    # voice-start on the robot while every test here stays green — caught
    # once already (demo.detections pulled in emulator.detector).
    import subprocess
    import sys

    code = ("import sys, demo.run_demo; print(sorted(m for m in ("
            "'emulator.litert_runtime', 'emulator.detector', 'emulator.engines', "
            "'ai_edge_litert', 'litert_lm') if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, check=True,
                         cwd=Path(__file__).resolve().parent.parent).stdout
    assert out.strip() == "[]"


def test_handle_stream_reports_the_size_of_the_robots_memory(monkeypatch):
    fm, sm = _FakeFrameMemory([]), _FakeSpeechMemory([])
    display = _turn(monkeypatch, "hello", _said("Hi!"), frame_memory=fm,
                    speech_memory=sm, frame=np.zeros((2, 2, 3), dtype=np.uint8))
    # Nothing stored for the turn itself, nothing evicted yet.
    assert display.memory_counts[-1] == (0, 0)
    from demo.conversation import DEFAULT_CONTEXT_BUDGET

    assert display.contexts[-1] == (300, DEFAULT_CONTEXT_BUDGET, 1)


def test_memory_size_is_reported_even_on_a_quiet_turn(monkeypatch):
    fm = _FakeFrameMemory([])
    display = _turn(monkeypatch, "", frame_memory=fm,
                    frame=np.zeros((2, 2, 3), dtype=np.uint8))
    assert display.memory_counts == [(0, 0)]


def test_a_memory_that_cannot_be_counted_does_not_break_the_turn(monkeypatch):
    class _Uncountable(_FakeFrameMemory):
        def count(self):
            raise OSError("shard busy")

    display = _turn(monkeypatch, "hello", _said("Hi!"),
                    frame_memory=_Uncountable([]))
    assert display.memory_counts == []
    assert display.replies[-1] == ("Hi!", True)


def test_handle_stream_shows_memory_and_body_tools_of_one_turn(monkeypatch):
    display = _turn(monkeypatch, "What's my name? And dance!",
                    _tool("remember", query="name"), _tool("move", how="dance"),
                    _said("Sure, and your name is Sasha!"))
    names = [name for name, _args in display.tool_calls]
    assert names[0] == "remember" and "move" in names


def test_the_voice_loop_waits_while_the_dashboard_pause_is_on(capsys):
    from demo.run_demo import _wait_while_paused

    class _Display:
        def __init__(self, answers):
            self._answers = list(answers)

        def is_paused(self):
            return self._answers.pop(0)

    slept = []
    assert _wait_while_paused(_Display([True, True, False]),
                              sleep=slept.append) is True
    assert len(slept) == 2
    out = capsys.readouterr().out
    assert out.count("paused") == 1, "say it once, not on every poll"
    assert "resumed" in out


def test_nothing_waits_when_the_pause_is_off():
    from demo.run_demo import _wait_while_paused

    class _Display:
        def is_paused(self):
            return False

    def _no_sleep(seconds):
        raise AssertionError("must not wait when the pause is off")

    assert _wait_while_paused(_Display(), sleep=_no_sleep) is False


# --- the dashboard's Restart button (demo/display/web.py's /control ->
# _RestartWatcher -> scripts/voice_loop.sh): the robot ends the run with
# RESTART_EXIT_CODE, and the supervisor wipes its memory and starts it again.


class _RestartDashboard:
    """A dashboard with a Restart button, as the robot sees it over HTTP."""

    def __init__(self, pending=False):
        self.pending = pending
        self.acks = 0

    def restart_requested(self):
        return self.pending

    def ack_restart(self):
        self.acks += 1
        self.pending = False


def _watch(display, monkeypatch, poll=0.001):
    """Run a watcher against `display` until it interrupts, or time runs out."""
    from demo import run_demo

    interrupts = []
    monkeypatch.setattr(run_demo._thread, "interrupt_main",
                        lambda: interrupts.append(True))
    watcher = run_demo._RestartWatcher(display, poll=poll).start()
    for _ in range(500):
        if interrupts:
            break
        time.sleep(0.002)
    watcher.close()
    return watcher, interrupts


def test_the_restart_button_ends_the_run_through_the_same_path_as_ctrl_c(monkeypatch):
    dash = _RestartDashboard()
    from demo import run_demo

    interrupts = []
    monkeypatch.setattr(run_demo._thread, "interrupt_main",
                        lambda: interrupts.append(True))
    watcher = run_demo._RestartWatcher(dash, poll=0.001).start()
    try:
        assert watcher.requested is False, "nothing pressed yet"
        dash.pending = True
        for _ in range(500):
            if interrupts:
                break
            time.sleep(0.002)
    finally:
        watcher.close()
    assert watcher.requested is True
    # KeyboardInterrupt in the main thread, not sys.exit from this one: the
    # conversation still flushes and the shards still close (run_voice's
    # finally) before the memory is wiped.
    assert interrupts == [True]
    # Acknowledged, or the loop that comes back reads the same request and
    # restarts again, forever.
    assert dash.acks == 1


def test_a_restart_left_over_from_the_last_run_is_cleared_not_acted_on(monkeypatch, capsys):
    """The old loop died before it could acknowledge the request. The fresh
    one must clear it — not wipe the memory it was just started with."""
    dash = _RestartDashboard(pending=True)
    watcher, interrupts = _watch(dash, monkeypatch)
    assert dash.acks == 1 and dash.pending is False
    assert watcher.requested is False and interrupts == []
    assert "left over" in capsys.readouterr().out


def test_a_display_with_no_restart_button_never_restarts_the_robot(monkeypatch):
    """--display none, or a dashboard from before the button existed."""
    from demo.display import NullSink

    watcher, interrupts = _watch(NullSink(), monkeypatch)
    assert watcher.requested is False and interrupts == []


def test_an_unreachable_dashboard_is_not_a_reason_to_wipe_the_memory(monkeypatch):
    class _Unreachable:
        def restart_requested(self):
            raise OSError("connection refused")

        def ack_restart(self):
            raise AssertionError("nothing was requested")

    watcher, interrupts = _watch(_Unreachable(), monkeypatch)
    assert watcher.requested is False and interrupts == []


# --- who the robot is talking to (demo/people.py): the face lookup happens
# before the conversation, so a greeting comes first and the memory this turn
# writes carries the right name ---

class _People:
    """People stand-in, scripted per turn."""

    def __init__(self, *, name=None, box=None, ask=False, awaiting=False,
                 answer=("Sasha", "Nice to meet you, Sasha. I'll remember you.")):
        from demo.people import Seen

        self.enabled = True
        self.current = Seen(name, 0.9 if name else 0.05, box)
        self._ask = ask
        self.awaiting_name = awaiting
        self._answer = answer
        self.answered = []
        self.asked = False

    def observe(self, frame):
        return self.current

    def greeting(self):
        return f"Hello again, {self.current.name}!" if self.current.name else None

    def should_ask_name(self):
        return self._ask

    def ask_name(self):
        self.asked = True
        self._ask = False
        self.awaiting_name = True
        return "I don't think we've met. What's your name?"

    def answer_name(self, heard):
        self.answered.append(heard)
        return self._answer


def test_a_stranger_in_front_does_not_inherit_the_last_persons_name(monkeypatch):
    from demo.conversation import ConversationWindow
    from demo.people import Seen

    window = ConversationWindow()
    window.set_speaker("Alice")
    people = _People()
    people.current = Seen(None, 0.05, [0.3, 0.2, 0.6, 0.7], stranger_arrived=True)
    _turn(monkeypatch, "Hello there.", _said("Hi!"), conversation=window,
          people=people, frame=np.zeros((4, 4, 3), np.uint8))
    assert window.speaker is None
    assert [e.text for e in window._exchanges] == ["Person: Hello there. — Reachy: Hi!"]


def test_alice_then_a_stranger_who_answers_reaches_memory_under_each_name(monkeypatch):
    # The whole way through the voice loop: Alice is recognised, steps away,
    # a stranger steps in, talks, is asked their name and answers "Bob".
    from demo.conversation import ConversationWindow
    from demo.people import Seen

    class _Remembered:
        def __init__(self):
            self.texts = []

        def remember(self, text, kind=None, meta=None):
            self.texts.append(text)

    memory = _Remembered()
    window = ConversationWindow(memory)
    frame = np.zeros((4, 4, 3), np.uint8)
    alice = _People(name="Alice", box=[0.3, 0.2, 0.6, 0.7])
    _turn(monkeypatch, "My dog is called Rex.", _said("Lovely name!"),
          conversation=window, people=alice, frame=frame)
    stranger = _People(ask=True, answer=("Bob", "Nice to meet you, Bob."))
    stranger.current = Seen(None, 0.05, [0.3, 0.2, 0.6, 0.7], stranger_arrived=True)
    _turn(monkeypatch, "I like trains.", _said("Me too!"),
          conversation=window, people=stranger, frame=frame)
    assert stranger.asked
    stranger.current = Seen(None, 0.05, [0.3, 0.2, 0.6, 0.7])   # still there
    _turn(monkeypatch, "I'm Bob.", conversation=window, people=stranger, frame=frame)
    window.flush()
    assert [text.split(":")[0] for text in memory.texts] == ["Alice", "Bob"]


def test_a_recognised_person_is_greeted_and_named_in_memory(monkeypatch):
    from demo.conversation import ConversationWindow

    window = ConversationWindow()
    people = _People(name="Sasha", box=[0.3, 0.2, 0.6, 0.7])
    display = _turn(monkeypatch, "Hello there.", _said("Hi!"),
                    conversation=window, people=people)
    assert display.faces[-1] == ("Sasha", [0.3, 0.2, 0.6, 0.7], 0.9)
    assert ("Hello again, Sasha!", True) in display.replies
    assert window.speaker == "Sasha"


def test_a_stranger_is_asked_for_a_name_after_the_answer(monkeypatch):
    people = _People(ask=True)
    display = _turn(monkeypatch, "What can you do?", _said("Quite a lot!"),
                    people=people)
    assert people.asked
    # The question comes after the reply — the person came to talk, not to be
    # interviewed.
    assert [text for text, _done in display.replies][-1] == (
        "I don't think we've met. What's your name?")


def test_a_name_question_nobody_heard_is_not_waiting_for_an_answer(monkeypatch):
    # The voice failed: the next thing said is a question, not a name — seen
    # live, "What's" was enrolled as someone's name.
    people = _People(ask=True)

    def no_voice(endpoint, payload):
        if endpoint.endswith("/say"):
            raise OSError("laptop gone")
        return {"heard": "What can you do?"}

    monkeypatch.setattr("demo.run_demo._http_post", no_voice)
    monkeypatch.setattr("demo.run_demo._http_stream",
                        _chat_script(_said("Quite a lot!")))
    monkeypatch.setattr("demo.audio_out.StreamPlayer", FakePlayer)
    from demo.run_demo import _handle_stream

    _handle_stream([], np.zeros(8000, np.float32), ENDPOINT, FakeRobot(),
                   DisplaySpy(), people=people)
    assert people.asked and not people.awaiting_name


def test_the_answer_to_the_name_question_never_reaches_the_model(monkeypatch):
    from demo.conversation import ConversationWindow
    from demo.run_demo import _handle_stream

    window = ConversationWindow()
    people = _People(awaiting=True)
    monkeypatch.setattr("demo.run_demo._http_post", _fake_transcribe("Sasha"))
    monkeypatch.setattr("demo.run_demo._http_stream", _no_generate_http_stream)
    monkeypatch.setattr("demo.audio_out.StreamPlayer", FakePlayer)
    display, fm = DisplaySpy(), _FakeFrameMemory([])
    _handle_stream([], np.zeros(8000, np.float32), ENDPOINT, FakeRobot(), display,
                   conversation=window, people=people, frame_memory=fm,
                   frame=np.zeros((2, 2, 3), dtype=np.uint8))
    assert people.answered == ["Sasha"]
    assert window.speaker == "Sasha"
    assert window.history == [], "the name answer is not an exchange"
    assert ("Nice to meet you, Sasha. I'll remember you.", True) in display.replies
    assert fm.calls == [], "a name answer stores no frame"


def test_the_robot_looks_at_the_face_not_the_chest(monkeypatch):
    seen, robot = [], FakeRobot()
    people = _People(name="Sasha", box=[0.4, 0.1, 0.6, 0.4])
    _turn(monkeypatch, "hello", _said("Hi!"), seen=seen, robot=robot,
          people=people,
          detections=[{"label": "person", "score": 0.9, "box": [0.1, 0.1, 0.9, 1.0]}])
    _name, (x, y) = [call for call in robot.calls if call[0] == "look_at"][0]
    # Centre of the FACE box in [-1, 1]; the person box would give (0.0, 0.1).
    assert (x, y) == (pytest.approx(0.0), pytest.approx(-0.5))


# --- FaceTracker: the head follows the face between turns ---

def _tracker(faces, moves, display=None, clock=None, **kwargs):
    from demo.run_demo import FaceTracker

    return FaceTracker(lambda: faces[0], lambda x, y: moves.append((x, y)),
                       display=display, clock=clock or (lambda: 0.0), **kwargs)


def test_the_head_turns_to_a_face_and_follows_it_when_it_moves():
    faces = [[{"box": [0.4, 0.2, 0.6, 0.5], "score": 0.9}]]
    moves, now = [], [0.0]
    tracker = _tracker(faces, moves, clock=lambda: now[0])
    tracker()
    assert moves == [(pytest.approx(0.0), pytest.approx(-0.3))]
    now[0] = 1.0
    tracker()
    assert len(moves) == 1, "the face has not moved — do not whirr"
    faces[0] = [{"box": [0.7, 0.2, 0.9, 0.5], "score": 0.9}]
    now[0] = 2.0
    tracker()
    # 70% of the way from 0.0 to 0.6: the head eases towards a moved face
    # instead of snapping to every jitter of the box — and no longer only
    # halfway: at 0.9 s and 50% it took three steps to settle (live).
    assert moves[-1] == (pytest.approx(0.42), pytest.approx(-0.3))


def test_the_head_does_not_chase_a_jittering_box():
    faces = [[{"box": [0.40, 0.20, 0.60, 0.50], "score": 0.9}]]
    moves, now = [], [0.0]
    tracker = _tracker(faces, moves, clock=lambda: now[0])
    tracker()
    faces[0] = [{"box": [0.41, 0.21, 0.61, 0.51], "score": 0.9}]  # a wobble
    now[0] = 5.0
    tracker()
    assert len(moves) == 1


def test_the_head_is_not_re_aimed_faster_than_the_interval():
    faces = [[{"box": [0.1, 0.1, 0.3, 0.4], "score": 0.9}]]
    moves, now = [], [0.0]
    tracker = _tracker(faces, moves, clock=lambda: now[0], min_interval=1.0)
    tracker()
    faces[0] = [{"box": [0.7, 0.1, 0.9, 0.4], "score": 0.9}]
    now[0] = 0.5
    tracker()
    assert len(moves) == 1, "half the interval: too soon"
    now[0] = 1.2
    tracker()
    assert len(moves) == 2


def test_the_dashboard_gets_the_box_every_cycle_even_when_the_head_stays():
    faces = [[{"box": [0.4, 0.2, 0.6, 0.5], "score": 0.88}]]
    moves, display = [], DisplaySpy()
    tracker = _tracker(faces, moves, display=display)
    tracker()
    tracker()
    assert len(display.faces) == 2 and len(moves) == 1
    assert display.faces[-1] == (None, [0.4, 0.2, 0.6, 0.5], 0.88)


def test_a_frame_with_nobody_in_it_clears_the_box_and_moves_nothing():
    moves, display = [], DisplaySpy()
    tracker = _tracker([[]], moves, display=display)
    tracker()
    assert display.faces == [(None, None, 0.0)]
    assert moves == []


# — the knowledge base (demo/knowledge.py) —

def test_build_knowledge_restores_the_shipped_snapshot_into_the_memory_dir(tmp_path):
    from types import SimpleNamespace

    from demo.knowledge import read_facts
    from demo.run_demo import build_knowledge

    base = build_knowledge(SimpleNamespace(memory_dir=str(tmp_path), brain="mac",
                                           embed_port=1))
    try:
        assert base.count() == len(read_facts())
        assert (tmp_path / "knowledge" / "wal").exists()
    finally:
        base.close()


def test_build_knowledge_is_off_with_its_flag_or_without_memory(tmp_path):
    from types import SimpleNamespace

    from demo.run_demo import build_knowledge

    assert build_knowledge(SimpleNamespace(memory_dir=str(tmp_path), no_knowledge=True)) is None
    assert build_knowledge(SimpleNamespace(memory_dir=str(tmp_path), no_memory=True)) is None
    assert not (tmp_path / "knowledge").exists()


def test_a_missing_snapshot_disables_knowledge_not_the_robot(tmp_path, capsys):
    from types import SimpleNamespace

    from demo.run_demo import build_knowledge

    args = SimpleNamespace(memory_dir=str(tmp_path), brain="mac", embed_port=1,
                           knowledge_snapshot=str(tmp_path / "missing.snapshot"))
    assert build_knowledge(args) is None
    assert "knowledge disabled" in capsys.readouterr().out


def test_the_memory_count_includes_the_knowledge_facts():
    from demo.run_demo import _report_memory_size

    class _Facts:
        def count(self):
            return 41

    counts = []

    class _Dash:
        def on_memory_count(self, frames, exchanges, knowledge=0):
            counts.append((frames, exchanges, knowledge))

    _report_memory_size(_Dash(), None, None, _Facts())
    assert counts == [(0, 0, 41)]


# --- Looker: the `camera` tool's direction turns the head ---

class _LookRobot:
    def __init__(self, fail=False):
        self.calls = []
        self._fail = fail

    def look(self, direction):
        if self._fail:
            raise OSError("robot unreachable")
        self.calls.append(("look", direction))

    def look_at(self, x, y):
        self.calls.append(("look_at", (x, y)))


class _LookSource:
    def __init__(self, frame):
        self._frame = frame

    def latest(self):
        return self._frame, [{"label": "plant"}]


class _LookMemory:
    def __init__(self):
        self.stored = []
        self.captions = []

    def remember(self, frame, detections, meta=None):
        self.stored.append((detections, meta))
        return len(self.stored) - 1

    def describe(self, point_id, caption):
        self.captions.append((point_id, caption))


def _looker(robot=None, frame=None, tracker=None, memory=None, sleeps=None):
    import numpy as np

    from demo.run_demo import Looker

    frame = np.zeros((4, 4, 3), np.uint8) if frame is None else frame
    return Looker(robot or _LookRobot(), _LookSource(frame), tracker, memory,
                  settle_s=1.6, sleep=(sleeps.append if sleeps is not None else lambda s: None))


def test_a_look_turns_the_head_waits_for_the_picture_and_remembers_where():
    robot, memory, sleeps, moves = _LookRobot(), _LookMemory(), [], []
    faces = [[{"box": [0.7, 0.2, 0.9, 0.5], "score": 0.9}]]
    tracker = _tracker(faces, moves)
    looker = _looker(robot, tracker=tracker, memory=memory, sleeps=sleeps)
    jpeg = looker.look("left")
    assert jpeg and jpeg[:2] == b"\xff\xd8"
    assert robot.calls == [("look", "left")]
    assert sleeps == [1.6]
    assert memory.stored == [([{"label": "plant"}], {"looked": "left"})]
    tracker()
    assert moves == [], "the tracker must not pull the head back mid-look"


def test_a_look_reads_who_is_in_THAT_picture_not_the_one_the_turn_started_with():
    """The note that goes with the picture is built from these names
    (demo/conversation.py). With the head turned to the wall, the turn's own
    frame still shows the person the robot was facing — and the note told the
    model "Sasha is in front of you" about a picture Sasha is not in."""
    from demo.run_demo import Looker

    class _People:
        enabled = True

        def __init__(self):
            self.frames = 0

        def in_frame(self, frame):
            self.frames += 1
            return [{"name": "Masha", "box": [0.1, 0.1, 0.2, 0.2], "score": 0.8},
                    {"name": None, "box": [0.5, 0.1, 0.6, 0.2], "score": 0.1}]

        def look_away(self):
            pass

    memory, people = _LookMemory(), _People()
    looker = Looker(_LookRobot(), _LookSource(np.zeros((4, 4, 3), np.uint8)), None,
                    memory, settle_s=0.0, sleep=lambda s: None, people=people)
    looker.look("left")
    assert looker.names == ["Masha"], "named faces only, off the turned frame"
    assert people.frames == 1, "one face read for the picture, not two"
    assert memory.stored[0][1]["people"][0]["name"] == "Masha"


def test_coming_back_faces_the_person_and_lets_the_tracker_aim_again():
    robot, moves = _LookRobot(), []
    faces = [[{"box": [0.4, 0.2, 0.6, 0.5], "score": 0.9}]]
    tracker = _tracker(faces, moves)
    looker = _looker(robot, tracker=tracker)
    looker.look("right")
    looker.come_back([0.4, 0.2, 0.6, 0.5])
    assert robot.calls[-1] == ("look_at", (pytest.approx(0.0), pytest.approx(-0.3)))
    tracker()
    assert len(moves) == 1


def test_coming_back_without_a_face_looks_ahead_and_only_after_a_look():
    robot = _LookRobot()
    looker = _looker(robot)
    looker.come_back(None)
    assert robot.calls == []
    looker.look("left")
    looker.come_back(None)
    assert robot.calls == [("look", "left"), ("look", "ahead")]


def test_a_head_that_did_not_turn_stores_nothing_and_says_so():
    # The picture would be what is in front, stored as what is on the left:
    # "what was on your left?" answered from it later, wrongly.
    from demo.conversation import LookFailed

    memory = _LookMemory()
    looker = _looker(_LookRobot(fail=True), memory=memory)
    with pytest.raises(LookFailed, match="could not turn to your left"):
        looker.look("left")
    looker.caption("On my left I see a window.")
    assert memory.stored == [] and memory.captions == []
    assert looker.turned, "come_back still puts the head where it belongs"


def test_a_head_that_cannot_turn_is_said_through_the_voice_loop(monkeypatch):
    # The whole turn: the model asks to look left, the robot cannot turn, the
    # model hears so, nothing is stored, the head comes back, and the face
    # tracker is let go even though coming back failed too.
    import numpy as np

    from demo.run_demo import Looker

    faces, moves = [[{"box": [0.4, 0.2, 0.6, 0.5], "score": 0.9}]], []
    tracker = _tracker(faces, moves)
    memory, seen = _LookMemory(), []
    looker = Looker(_LookRobot(fail=True), _LookSource(np.zeros((4, 4, 3), np.uint8)),
                    tracker, memory, settle_s=0.0, sleep=lambda s: None)
    _turn(monkeypatch, "Look to your left.", _tool("camera", direction="left"),
          _said("I can't turn my head right now."), seen=seen, looker=looker)
    assert seen[1][1].image_jpeg is None
    assert seen[1][1].tool_result["result"] == {"error": "your head could not turn to your left"}
    assert memory.stored == [] and memory.captions == []
    assert not looker.turned
    tracker()
    assert moves, "the tracker aims again"


def test_a_failed_look_does_not_caption_an_earlier_frame():
    # A look stored the left; its turn ended before a caption. The next turn's
    # look fails: the reply about that must not become the left's caption.
    from demo.conversation import LookFailed

    memory = _LookMemory()
    robot = _LookRobot()
    looker = _looker(robot, memory=memory)
    looker.look("left")
    robot._fail = True
    with pytest.raises(LookFailed):
        looker.look("right")
    looker.caption("I can't turn my head right now.")
    assert memory.captions == []


def test_a_turn_that_failed_after_a_look_leaves_the_picture_without_a_caption(monkeypatch):
    # Turn N looks left and stores the frame, then the reply fails. Turn N+1
    # looks nowhere — its reply used to become the caption of turn N's frame.
    from demo.run_demo import _handle_stream

    memory = _LookMemory()
    looker = _looker(memory=memory)
    replies = iter([_tool("camera", direction="left")])

    def fails_after_the_look(endpoint, payload):
        try:
            return iter(next(replies))
        except StopIteration:
            raise OSError("the laptop dropped") from None

    monkeypatch.setattr("demo.run_demo._http_post", _fake_transcribe("Look to your left."))
    monkeypatch.setattr("demo.run_demo._http_stream", fails_after_the_look)
    monkeypatch.setattr("demo.audio_out.StreamPlayer", FakePlayer)
    _handle_stream([], np.zeros(8000, np.float32), ENDPOINT, FakeRobot(),
                   DisplaySpy(), looker=looker)
    assert len(memory.stored) == 1

    _turn(monkeypatch, "My favourite colour is blue.",
          _said("Blue is my favourite colour."), looker=looker)
    assert memory.captions == []


def test_a_look_is_captioned_with_its_own_turns_reply(monkeypatch):
    # The path that must keep working: through the voice loop, the reply to
    # a look becomes that frame's caption.
    memory = _LookMemory()
    looker = _looker(memory=memory)
    _turn(monkeypatch, "Look to your left.", _tool("camera", direction="left"),
          _said("I see a lamp."), looker=looker)
    assert memory.captions == [(0, "I see a lamp.")]


def test_a_turn_that_raised_after_a_look_leaves_the_picture_without_a_caption(monkeypatch):
    # Any exception, not only the ones _talk catches: the voice loop catches
    # the rest per turn and keeps the same looker for the next one.
    from demo.run_demo import _handle_stream

    memory = _LookMemory()
    looker = _looker(memory=memory)
    replies = iter([_tool("camera", direction="left")])

    def breaks_after_the_look(endpoint, payload):
        try:
            return iter(next(replies))
        except StopIteration:
            raise RuntimeError("a malformed event") from None

    monkeypatch.setattr("demo.run_demo._http_post", _fake_transcribe("Look to your left."))
    monkeypatch.setattr("demo.run_demo._http_stream", breaks_after_the_look)
    monkeypatch.setattr("demo.audio_out.StreamPlayer", FakePlayer)
    with pytest.raises(RuntimeError):
        _handle_stream([], np.zeros(8000, np.float32), ENDPOINT, FakeRobot(),
                       DisplaySpy(), looker=looker)
    assert len(memory.stored) == 1
    _turn(monkeypatch, "My favourite colour is blue.",
          _said("Blue is my favourite colour."), looker=looker)
    assert memory.captions == []


def test_the_tracker_tells_people_a_face_is_still_there_and_a_look_does_not_break_it():
    class _People:
        def __init__(self):
            self.seen = 0
            self.current = type("C", (), {"name": None})()

        def face_seen(self):
            self.seen += 1

    from demo.run_demo import FaceTracker

    people, moves = _People(), []
    faces = [[{"box": [0.4, 0.2, 0.6, 0.5], "score": 0.9}]]
    tracker = FaceTracker(lambda: faces[0], lambda x, y: moves.append((x, y)),
                          people=people, clock=lambda: 0.0)
    tracker()
    assert people.seen == 1
    tracker.hold()
    tracker()
    assert people.seen == 1, "while the head looks away, nothing is concluded"
    tracker.release()
    assert people.seen == 1, "and coming back marks no face nobody saw"
    tracker()
    assert people.seen == 2

    # Paused from the dashboard, facing the room: the head keeps still, but
    # the faces in view are still the people in front of it.
    faces[0] = [{"box": [0.4, 0.2, 0.6, 0.5], "score": 0.9}]
    tracker.hold(watching=True)
    tracker()
    assert people.seen == 3
    faces[0] = []
    tracker()
    tracker.release()
    assert people.seen == 3, "an empty room is not stamped as a face on resume"


def test_coming_back_from_a_look_tells_people_the_person_did_not_leave():
    class _Kept:
        enabled = False
        calls = []

        def look_away(self):
            self.calls.append("away")

        def look_back(self):
            self.calls.append("back")

    people = _Kept()
    looker = Looker_with(people)
    looker.look("left")
    looker.come_back(None)
    assert people.calls == ["away", "back"]


def Looker_with(people):
    import numpy as np

    from demo.run_demo import Looker

    return Looker(_LookRobot(), _LookSource(np.zeros((4, 4, 3), np.uint8)), None,
                  None, settle_s=0.0, sleep=lambda s: None, people=people)


def test_a_look_ahead_takes_the_picture_without_turning_or_waiting():
    robot, memory, sleeps = _LookRobot(), _LookMemory(), []
    looker = _looker(robot, memory=memory, sleeps=sleeps)
    assert looker.look("ahead")
    assert robot.calls == [] and sleeps == []
    assert memory.stored[-1][1] == {"looked": "ahead"}
    looker.come_back(None)
    assert robot.calls == [], "the head never left"


def test_what_the_robot_said_about_a_look_is_kept_with_that_frame():
    memory = _LookMemory()
    looker = _looker(memory=memory)
    looker.caption("I see nothing.")            # nothing looked at yet
    looker.look("left")
    looker.caption("I see a lamp.")
    looker.caption("I see it again.")           # only once per look
    assert memory.captions == [(0, "I see a lamp.")]


def test_a_look_keeps_who_was_in_the_picture():
    class _Here:
        enabled = True

        def in_frame(self, frame):
            return [{"name": "Sasha", "box": [0, 0, 1, 1], "score": 0.7}]

    import numpy as np

    from demo.run_demo import Looker

    memory = _LookMemory()
    looker = Looker(_LookRobot(), _LookSource(np.zeros((4, 4, 3), np.uint8)), None, memory,
                    settle_s=0.0, sleep=lambda s: None, people=_Here())
    looker.look("ahead")
    assert memory.stored[-1][1] == {"looked": "ahead",
                                    "people": [{"name": "Sasha", "box": [0, 0, 1, 1], "score": 0.7}]}


def test_the_shards_are_closed_when_the_loop_stops():
    # A run killed without this came back up not knowing the person it had
    # just met: the last writes were still only in the shard's WAL buffer.
    from demo.people import People
    from emulator.edge_store import EdgeStore, TEXT
    from emulator.memory import TextMemory

    class _Words:
        def embed(self, texts):
            return [[1.0, 0.0, 0.0] for _ in texts]

        query_embed = embed

    store = EdgeStore(None, vectors={TEXT: 3})
    memory = TextMemory(store=store, embedder=_Words())
    memory.close()
    memory.close()  # two memories share one shard; both close at shutdown
    assert People().close() is None


def test_the_head_goes_part_of_the_way_to_a_new_target():
    faces = [[{"box": [0.4, 0.2, 0.6, 0.5], "score": 0.9}]]
    moves, now = [], [0.0]
    tracker = _tracker(faces, moves, clock=lambda: now[0], smoothing=0.5)
    tracker()
    assert moves[-1] == (pytest.approx(0.0), pytest.approx(-0.3))
    faces[0] = [{"box": [0.8, 0.2, 1.0, 0.5], "score": 0.9}]
    now[0] = 5.0
    tracker()
    assert moves[-1] == (pytest.approx(0.4), pytest.approx(-0.3)), "half of the way to 0.8"


def test_a_paused_robot_stops_following_faces():
    from demo.run_demo import _wait_while_paused

    class _Dash:
        def __init__(self):
            self.checks = 0

        def is_paused(self):
            self.checks += 1
            return self.checks <= 2

    faces = [[{"box": [0.4, 0.2, 0.6, 0.5], "score": 0.9}]]
    moves = []
    tracker = _tracker(faces, moves)
    dash = _Dash()
    held = []
    _wait_while_paused(dash, sleep=lambda s: held.append(tracker._held), tracker=tracker)
    assert held == [True, True], "the head stands still while paused"
    assert tracker._held is False, "and follows again when the pause is lifted"


def test_the_picture_carries_everyone_the_robot_recognises():
    from demo.run_demo import _names_in

    class _People:
        enabled = True
        current = type("C", (), {"known": False, "name": None})()

        def in_frame(self, frame):
            return [{"name": "Sasha"}, {"name": None}, {"name": "Robin"}]

    assert _names_in(_People(), object()) == ["Sasha", "Robin"]
    assert _names_in(None, object()) == []


def test_with_no_face_matched_the_picture_still_carries_the_person_in_view():
    from demo.run_demo import _names_in

    class _People:
        enabled = True
        current = type("C", (), {"known": True, "name": "Sasha"})()

        def in_frame(self, frame):
            return []

    assert _names_in(_People(), object()) == ["Sasha"]


def test_the_speech_threshold_is_fixed_by_default():
    """Calibrated from one second of sound at start, the threshold came out
    anywhere from 0.010 to 0.114 across a day of runs: deaf with anyone
    talking during calibration, flooded with noise when it was very quiet.
    It is fixed now; 0 brings the calibration back."""
    from demo.run_demo import SPEECH_THRESHOLD, parse_args

    assert SPEECH_THRESHOLD == 0.010
    assert parse_args([]).speech_threshold == SPEECH_THRESHOLD
    assert parse_args(["--speech-threshold", "0"]).speech_threshold == 0


def test_a_reply_cut_off_mid_stream_is_not_taken_for_an_empty_answer(monkeypatch, capsys):
    # The laptop died mid-reply: no `done` line. That used to end the turn as
    # an empty answer — and write it into the conversation as one.
    from demo.conversation import ConversationWindow

    window = ConversationWindow(None)
    cut_off = _said("Hello, I was about to")[:1]           # audio, no done
    _turn(monkeypatch, "Hi there", cut_off, conversation=window)
    assert window.history == []
    assert "[chat] failed" in capsys.readouterr().out


def test_a_second_sigterm_does_not_cut_the_shutdown_short(monkeypatch):
    import signal

    import demo.run_demo as run_demo_mod

    handlers = {}
    monkeypatch.setattr(signal, "signal",
                        lambda signum, handler: handlers.update({signum: handler}))
    run_demo_mod._interrupt_on_sigterm()
    handler = handlers[signal.SIGTERM]
    with pytest.raises(KeyboardInterrupt):
        handler(signal.SIGTERM, None)
    handler(signal.SIGTERM, None)   # during the `finally`: ignored


def test_the_laptops_own_camera_and_microphone_are_the_defaults():
    # A fixed index was one Mac's setup: a stock MacBook has no microphone 1,
    # and the run command in the README failed there.
    from demo.run_demo import parse_args

    args = parse_args([])
    assert (args.video, args.audio) == ("default", "default")
