"""The robot's voice loop: listen, look, remember, answer.

It runs on the robot itself (scripts/robot_service.sh voice-start) or, with
the emulator, on the laptop (README.md). The loop owns the conversation and
all of the robot's memory — the Qdrant Edge shards live next to it, on the
robot's own disk. The models it cannot carry run on the laptop and are called
over HTTP: the language model always, the others unless `--on-robot` moves
them here (demo/placement.py):

- demo/serve.py — speech recognition, the language model, the voice;
- demo/detect_service.py — the object and face detector;
- demo/embed_service.py — the embeddings (SigLIP 2, bge, face identities).

The camera, mic and speaker are the robot's, reached over loopback HTTP
(demo/camera_service.py and the Pollen daemon), or the laptop's own for a run
with no robot. The audience's screen is the dashboard on the laptop
(demo/display/web.py); this process pushes its events there (`--display
remote`), fire-and-forget: a dead dashboard must not stall a turn.

While the robot is talking the mic is ignored, and flushed afterwards, so it
does not answer its own voice.
"""

from __future__ import annotations

import _thread   # interrupt_main(): the Restart button, from its watcher thread
import argparse
import base64
import itertools
import json
import re
import threading
import urllib.request

import time
from typing import TYPE_CHECKING

import numpy as np

from demo.contract import (CHAT_PATH, NAME_PATH, SAY_PATH, TRANSCRIBE_PATH,
                           decode_transcribe_response, encode_transcribe_request)
from demo.conversation import (DEFAULT_CONTEXT_BUDGET, TURNED,
                               ConversationWindow, LookFailed, chat_turn,
                               day_frames, lookup, recall, recall_seen, who)
from demo.detect_source import (RemoteDetectSource, LocalDetectSource,
                                frame_to_jpeg)
from demo.detections import gaze_from_dicts
from demo.display import DEFAULT_DASHBOARD_PORT, build_display
from demo import placement
from demo.platform.mac import MacPlatform
from demo.vad import VoiceGate, calibrate_threshold, collect_utterance

if TYPE_CHECKING:
    from emulator.frame_memory import SceneChangeWriter

# The speech threshold the voice loop uses, fixed (see run_voice). 0.010 is
# where the good runs in the robot's log sat — the calibration's own floor.
SPEECH_THRESHOLD = 0.010


def _http_post(endpoint: str, payload: dict) -> dict:
    req = urllib.request.Request(
        endpoint, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=120).read())


def _http_stream(endpoint: str, payload: dict):
    """Open a streaming response, yield parsed NDJSON events."""
    req = urllib.request.Request(
        endpoint, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req, timeout=120)
    for raw in resp:
        line = raw.decode().strip()
        if line:
            yield json.loads(line)


# A turn is:
#   1. transcribe() below: POST the audio to serve.py's /transcribe, get the
#      text back (or transcribe here, with recognition placed on the robot).
#   2. demo/conversation.py's chat_turn: POST the text, with the conversation
#      still in the model's context, to serve.py's /chat — one live LLM chat
#      on the Mac that holds those turns. The model answers, or calls `remember`
#      (this robot searches its own Qdrant shards) or `camera` (this robot's
#      camera) and gets a second request with what came back.
#   3. Once the context passes its budget, the oldest exchanges move into
#      the robot's Qdrant shard as one batch (ConversationWindow).
# Memory is read only when the model asks for it — the earlier design recalled
# on every turn and attached the top frame to every prompt, and the robot
# narrated its camera instead of talking.
def transcribe(audio, sample_rate, endpoint, http_post=None) -> str:
    """POST the utterance audio to the laptop's /transcribe; return the
    heard text (possibly empty — a false VAD trigger is a successful call that
    heard nothing, not a failure).

    `http_post` is resolved at call time (`x = x or _default`), not bound as
    a default argument, so a test that patches `demo.run_demo._http_post`
    reaches this path too.
    """
    http_post = http_post or _http_post
    payload = encode_transcribe_request(audio, sample_rate)
    return decode_transcribe_response(http_post(endpoint, payload))


class _FailureGuard:
    """Wraps risky calls (robot/audio), logging the transition into
    "unhealthy" and back exactly ONCE, rather than on every event of every
    response. Without this, a failed robot or audio player would print the
    same message on EVERY look_at/gesture/feed of the next response — spam
    instead of signal. The instance lives for the whole voice session
    (created once in run_voice and threaded through calls), which is why the
    flag is "per-robot" rather than per-response.

    We deliberately catch broad Exception rather than narrowing the type
    list: both IPC to the MuJoCo sim/real Reachy SDK and afplay/WAV
    recording can fail for reasons no SDK documents exhaustively (dropped
    socket, busy audio device, corrupt data, etc.) — narrowing the types is
    risky: a real failure could slip past the catch and take down the whole
    voice loop, where before it just skipped a frame.
    """

    def __init__(self, label: str) -> None:
        self.label = label
        self.healthy = True

    def call(self, action, *args):
        """Run action(*args) under the guard. Returns its result, or None if
        it raised (so a guarded factory like make_player yields None on
        failure instead of taking down the caller)."""
        try:
            result = action(*args)
        except Exception as exc:  # noqa: BLE001 — see class docstring
            if self.healthy:
                print(f"  [{self.label}] skip ({type(exc).__name__}: {exc})")
            self.healthy = False
            return None
        else:
            if not self.healthy:
                print(f"  [{self.label}] recovered")
            self.healthy = True
            return result


class RobotDispatcher:
    """Runs robot actuation on a single background worker thread so a slow or
    dropped robot never blocks the NDJSON event loop that drives speech:
    drive_robot_stream hands calls to `dispatch(...)`
    instead of running the guarded call directly, gets back immediately, and
    moves on to the next event/phrase while the actual HTTP call runs
    elsewhere. Speech must never wait on motion.

    At most ONE pending call is held PER KIND (kind = the bound method's
    `__name__` — "look_at", "gesture", "emotion", "dance"): a newer command
    of the SAME kind (another look_at) REPLACES whatever of that kind hasn't
    started yet, rather than queueing up behind it — once the robot falls
    behind, playing out a backlog of stale head turns is worse than skipping
    straight to the latest one ("dropping stale motion rather than queueing
    it up"). A DIFFERENT kind gets its own slot and is never evicted by one:
    a head turn arriving must not cancel the emotion the model asked for a
    moment earlier. Per-kind slots keep the "never build a backlog" property
    without losing a different, model-requested action.

    `guard` is the SAME _FailureGuard the caller already holds for the whole
    session (see run_voice): the worker thread runs every call through
    `guard.call(...)`, so the unhealthy/recovered transition is still logged
    exactly once — just from the worker thread, not the event loop's.
    """

    def __init__(self, guard: _FailureGuard) -> None:
        self._guard = guard
        self._lock = threading.Lock()
        self._pending: dict[str, tuple] = {}   # kind -> (action, args)
        self._order: list[str] = []            # FIFO of kinds awaiting a run
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def dispatch(self, action, *args) -> None:
        """Enqueue action(*args), replacing any not-yet-started call of the
        SAME kind. Never blocks the caller."""
        kind = getattr(action, "__name__", repr(action))
        with self._lock:
            if kind not in self._pending:
                self._order.append(kind)
            self._pending[kind] = (action, args)
        self._wake.set()

    def _run(self) -> None:
        while True:
            with self._lock:
                if self._order:
                    kind = self._order.pop(0)
                    action, args = self._pending.pop(kind)
                else:
                    action = None
            if action is not None:
                # Drain whatever is already pending BEFORE honoring a stop
                # request — otherwise a dispatch() immediately followed by
                # close() (see test_robot_dispatcher_logs_failure_through_the
                # _shared_guard) could set `_stop` in the gap between the
                # item landing in `_pending` and this thread's next look, and
                # the already-enqueued call would never run.
                self._guard.call(action, *args)
                continue
            if self._stop.is_set():
                return
            # Bounded wait, not an indefinite block: close() must still be
            # able to join this thread promptly once nothing is pending.
            self._wake.wait(timeout=0.2)
            self._wake.clear()

    def close(self, timeout: float = 2.0) -> None:
        self._stop.set()
        self._thread.join(timeout=timeout)


class AsyncSceneWriter:
    """Runs SceneChangeWriter.observe() on its own worker thread so a slow
    memory write never stalls RemoteDetectSource's detect thread.

    RemoteDetectSource._cycle (demo/detect_source.py) calls every listener
    INLINE, on the same thread that POSTs frames to detect_service and updates
    `source.latest()`: `SceneChangeWriter.observe` used to be registered
    directly as a listener, so the moment it called `FrameMemory.remember`
    (emulator/frame_memory.py) — which embeds the frame via
    RemoteSiglipEmbedder, a blocking urlopen — a stalled Mac (mid-model-load,
    or a TCP blackhole on conference WiFi) froze detection itself: no new
    frame or detections stored, `source.latest()` stuck returning a stale
    pair, and `display.on_detections` (the other listener) stopped firing
    too, freezing the audience's live boxes with nothing logged (healthy()
    only counts failed detect_service POSTs, not a slow listener). Before this
    move remember() was a ~36 ms in-process SigLIP call, cheap enough to run
    inline; RemoteSiglipEmbedder made it a network round trip, which isn't.

    Mirrors RobotDispatcher above: dispatch() never blocks the caller, and at
    most one (frame, detections) pair is held — a newer one REPLACES
    whatever hasn't started yet, rather than queueing up behind it, since
    SceneChangeWriter only cares about the latest scene, not a backlog of
    stale ones (same "drop stale work rather than queue it" reasoning as
    RobotDispatcher's per-kind slots).
    """

    def __init__(self, writer: "SceneChangeWriter") -> None:
        self._writer = writer
        self._lock = threading.Lock()
        self._pending: tuple | None = None
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def observe(self, frame_rgb, detections) -> None:
        """Enqueue one detect cycle's (frame, detections), replacing
        whatever hasn't started yet. Never blocks the caller."""
        with self._lock:
            self._pending = (frame_rgb, detections)
        self._wake.set()

    def _run(self) -> None:
        while True:
            with self._lock:
                item = self._pending
                self._pending = None
            if item is not None:
                frame_rgb, detections = item
                try:
                    self._writer.observe(frame_rgb, detections)
                except Exception as exc:  # noqa: BLE001 — a bad write must not kill the worker
                    print(f"  [visual-memory] skip ({type(exc).__name__}: {exc})")
                continue
            if self._stop.is_set():
                return
            # Bounded wait, not an indefinite block: close() must still be
            # able to join this thread promptly once nothing is pending.
            self._wake.wait(timeout=0.2)
            self._wake.clear()

    def close(self, timeout: float = 6.0) -> None:
        # Longer than demo/embed_client.py's DEFAULT_TIMEOUT_S (5.0)
        # for the same reason as RemoteDetectSource.stop(): an in-flight
        # remember() should get the chance to actually finish (or time out on
        # its own) rather than being cut off mid-call by a shorter join.
        self._stop.set()
        self._thread.join(timeout=timeout)


def drive_robot_stream(robot, events, on_meta=None, make_player=None,
                       on_reply=None, robot_guard=None,
                       audio_guard=None, robot_dispatcher=None,
                       synthesizer=None) -> dict:
    """Play the response as it arrives. Returns the done event.

    `meta` turns the head towards the speaker; each phrase goes to ONE player,
    which plays the whole reply as a single clip, with no gaps, once the
    stream is done. Movement the model asks for is not here: it is the `move`
    tool, run by demo/conversation.py. Robot and audio calls run under their
    guards — a robot failure doesn't kill the dialogue, voice keeps working.

    on_meta(meta) — receives the meta event with detections.
    make_player(sample_rate) — player factory (for tests); defaults to
    demo/audio_out.py's StreamPlayer (afplay).
    on_reply(text, done) — for the dashboard: called on every played phrase
    (done=False) and on completion (done=True, full reply).
    robot_dispatcher — when given, robot calls are handed to its background
    worker instead of running on this thread, so a slow robot never stalls
    the audio below. None (the tests) keeps them synchronous.
    synthesizer — the voice, when synthesis is placed on the robot: `sentence`
    events are spoken here instead of arriving as audio.
    """
    import base64

    import numpy as np

    from demo.audio_out import StreamPlayer

    make_player = make_player or StreamPlayer
    robot_guard = robot_guard or _FailureGuard("robot")
    audio_guard = audio_guard or _FailureGuard("audio")

    def safe(action, *args) -> None:
        if robot_dispatcher is not None:
            robot_dispatcher.dispatch(action, *args)
        else:
            robot_guard.call(action, *args)

    def spoken_here(event):
        """A `sentence` event — the words, with synthesis placed on the robot
        — turned into the audio event the Mac would otherwise have sent, so
        everything below (one player, the guards, on_reply) stays one path."""
        text = event.get("text", "")
        audio = audio_guard.call(synthesizer.speak, text)
        if audio is None:
            return {"type": "audio", "text": text, "audio_b64": "",
                    "sample_rate": 0}
        return {"type": "audio", "text": text,
                "audio_b64": base64.b64encode(
                    np.asarray(audio, dtype=np.float32).tobytes()).decode("ascii"),
                "sample_rate": int(getattr(synthesizer, "sample_rate", 24000))}

    player = None
    done = {}
    for event in events:
        if event.get("type") == "sentence" and synthesizer is not None:
            event = spoken_here(event)
        kind = event.get("type")
        if kind == "meta":
            g = event.get("gaze", [0.0, 0.0])
            safe(robot.look_at, float(g[0]), float(g[1]))
            print(f"  I see:   {event.get('objects') or 'nothing'}")
            print(f"  I heard: {event.get('heard', '')!r}")
            for d in event.get("detections", []):
                print(f"    · {d['label']} {d['score']:.2f}")
            if on_meta is not None:
                on_meta(event)
        elif kind == "audio":
            audio = np.frombuffer(base64.b64decode(event["audio_b64"]),
                                  dtype=np.float32).copy()
            print(f"  saying:  {event.get('text', '')!r}")
            if len(audio) > 0:
                # The audio path is under the same protection as the robot:
                # a broken player — including its CONSTRUCTION (afplay
                # unavailable, WAV not writable) — shouldn't kill the rest of
                # the dialogue; this phrase just silently drops (logged once on
                # the transition to "unhealthy", see _FailureGuard).
                if player is None:
                    player = audio_guard.call(make_player,
                                              event.get("sample_rate", 16000))
                if player is not None:
                    audio_guard.call(player.feed, audio)
            if on_reply is not None:
                on_reply(event.get("text", ""), False)
        elif kind == "done":
            done = event
            if on_reply is not None:
                on_reply(event.get("reply", ""), True)
        elif kind == "error":
            # The brain failed after it had started answering (serve.py's
            # mid-stream error line): no done follows, the turn ends here.
            print(f"  [brain] error: {event.get('message', '')}")
    if player is not None:
        audio_guard.call(player.close)
    return done


def _say(text, endpoint, robot, display, *, make_player=None, robot_guard=None,
         audio_guard=None, robot_dispatcher=None, synthesizer=None) -> bool:
    """Say one fixed line (demo/people.py's greetings and the name question);
    returns whether there was a voice to say it with.

    Played through drive_robot_stream like any other speech, rather than a
    second playback path: one place knows how sound reaches the robot, and
    these lines get the same failure handling as the rest of a turn."""
    if not text:
        return False
    if synthesizer is not None:
        # Synthesis is on the robot: these lines are spoken here too, not
        # fetched from the laptop's /say.
        try:
            audio = synthesizer.speak(text)
        except Exception as exc:  # noqa: BLE001 — a line must not end the turn
            print(f"  [say] failed ({type(exc).__name__}: {exc}); staying quiet")
            return False
        event = {"type": "audio", "text": text,
                 "audio_b64": base64.b64encode(np.asarray(
                     audio, dtype=np.float32).tobytes()).decode("ascii"),
                 "sample_rate": int(getattr(synthesizer, "sample_rate", 24000))}
    else:
        try:
            event = _http_post(endpoint + SAY_PATH, {"text": text})
        except (OSError, ValueError) as exc:
            print(f"  [say] failed ({type(exc).__name__}: {exc}); staying quiet")
            return False
    if not event.get("audio_b64"):
        return False
    # No "saying:" line here: drive_robot_stream prints one for the audio event.
    drive_robot_stream(robot, iter([event, {"type": "done", "reply": text}]),
                       make_player=make_player, on_reply=display.on_reply,
                       robot_guard=robot_guard, audio_guard=audio_guard,
                       robot_dispatcher=robot_dispatcher)
    return True


# A transcript with no letter or digit in it ("♪♪", what a recogniser writes
# for music) is nothing anyone said. A turn built on one answers a question nobody asked
# — and, with the conversation kept, would sit in the model's context as if
# it had been said. Noise itself is turned away before it is a transcript, by
# what the recognisers can tell about the audio (a speech detector in front
# of either, Whisper's own no-speech estimate), not by a list of what it
# tends to come out as: "You", "Thanks for watching!".
def _holds_no_words(heard: str) -> bool:
    return bool(heard.strip()) and not re.search(r"[A-Za-z0-9]", heard)


def _handle_stream(detections, audio, endpoint, robot, display,
                   make_player=None, robot_guard=None, audio_guard=None,
                   frame_memory=None, speech_memory=None, frame=None,
                   robot_dispatcher=None, turn_started_at=None,
                   conversation=None, people=None, knowledge=None,
                   looker=None, transcriber=None, synthesizer=None) -> None:
    """One voice turn: transcribe, then talk through /chat (see the TURN
    SEQUENCE note above transcribe()).

    conversation — the session's ConversationWindow (demo/conversation.py):
    the exchanges still in the model's context, sent with every request, and
    the eviction of the oldest into speech_memory. None gives this turn a
    window of its own (tests; a one-off turn) — the model then sees no
    earlier conversation.
    frame_memory / speech_memory — the robot's SigLIP frames and bge exchanges
    (emulator/frame_memory.py, emulator/memory.py), searched only when the
    model calls `remember`.
    knowledge — the facts restored from the knowledge snapshot
    (demo/knowledge.py), searched by the same `remember` call.
    looker — turns the head for the `camera` tool's direction (Looker).
    frame — this turn's camera frame: who is recognised, and the `camera`
    tool's picture when there is no looker. It is not stored.
    turn_started_at — when this utterance began (run_voice's
    _turn_started_at); recall drops frames stored after it, which the scene
    writer can do mid-question (see demo/conversation.py's recall_seen).
    make_player / robot_guard / audio_guard / robot_dispatcher — passed
    through to drive_robot_stream, which plays each response.
    """
    try:
        # `transcriber` is the robot's own recognizer when this run placed
        # recognition there (build_transcriber); otherwise the Mac's endpoint.
        heard = (transcriber(audio, 16000) if transcriber is not None
                 else transcribe(audio, 16000, endpoint + TRANSCRIBE_PATH))
    except (OSError, ValueError, AttributeError) as exc:
        # Transcription itself failed (network drop, bad/partial response) —
        # stay quiet and listen again rather than take down the voice loop.
        print(f"  [transcribe] failed ({type(exc).__name__}: {exc}); staying quiet")
        # Clear the projected panels too: the PREVIOUS question's heard line,
        # reply and recall next to a robot that has just gone silent read as
        # the answer to what was said last.
        display.on_heard("")
        display.on_recall([])
        display.on_speech_recall([])
        return
    if _holds_no_words(heard):
        print(f"  [asr] ignoring {heard!r} — no words in it")
        heard = ""
    display.on_heard(heard)
    # Who is in front of the robot: one face lookup for this turn, before the
    # conversation, so a greeting comes first and the memory this turn writes
    # carries the right name (demo/people.py).
    if people is not None and people.enabled:
        seen = people.observe(frame)
        display.on_face(seen.name, seen.box, seen.score)
        if seen.known and conversation is not None:
            conversation.set_speaker(seen.name)
        elif seen.stranger_arrived and conversation is not None:
            # Someone the robot has never met has just stepped in: what they
            # say is not the previous person's. Named when they say who they are.
            conversation.someone_new()
        spoken = dict(make_player=make_player, robot_guard=robot_guard,
                      audio_guard=audio_guard, robot_dispatcher=robot_dispatcher,
                      synthesizer=synthesizer)
        if people.awaiting_name and heard.strip():
            # The robot asked for a name last turn; this is the answer, and it
            # is not a question for the language model.
            print(f"  I heard: {heard!r} (as the answer to the name question)")
            name, line = people.answer_name(heard)
            if name and conversation is not None:
                conversation.introduce(name)
            _say(line, endpoint, robot, display, **spoken)
            _report_memory_size(display, frame_memory, speech_memory, knowledge)
            return
        _say(people.greeting(), endpoint, robot, display, **spoken)
    # The last turn's recall goes blank: this turn shows recall only if the
    # model asks its memory.
    display.on_recall([])
    display.on_speech_recall([])
    try:
        if heard.strip():
            _talk(heard, detections, endpoint, robot, display,
                  conversation if conversation is not None
                  else ConversationWindow(speech_memory),
                  face_box=people.current.box if people is not None else None,
                  make_player=make_player, robot_guard=robot_guard,
                  audio_guard=audio_guard, robot_dispatcher=robot_dispatcher,
                  frame_memory=frame_memory, speech_memory=speech_memory,
                  frame=frame, turn_started_at=turn_started_at,
                  knowledge=knowledge, looker=looker, people=people,
                  synthesizer=synthesizer)
        if people is not None and people.should_ask_name():
            # Asked after the answer, not before it: the person came to talk,
            # not to be interviewed.
            if not _say(people.ask_name(), endpoint, robot, display,
                        make_player=make_player, robot_guard=robot_guard,
                        audio_guard=audio_guard,
                        robot_dispatcher=robot_dispatcher,
                        synthesizer=synthesizer):
                # A question nobody heard has no answer: the next thing said
                # is not a name.
                people.awaiting_name = False
    finally:
        # No frame is stored for the turn itself: frames are kept when the
        # objects in view change (SceneChangeWriter) and when the robot is
        # asked to look (Looker). One per turn filled the memory with the same
        # picture of the presenter, which then out-scored everything else.
        _report_memory_size(display, frame_memory, speech_memory, knowledge)


class FaceTracker:
    """Keeps the robot's head on the face between turns.

    A turn aims once, when it starts (_talk), which leaves the head frozen at
    the last target while the person keeps moving. This runs on the detect
    loop instead: the frames already go to the Mac, and now face boxes come
    back beside the object boxes (demo/detect_service.py's Faces), so the head
    follows at the detect rate with no extra round trip.

    Throttled three ways — no more often than `min_interval`, not for a move
    smaller than `min_move`, and only `smoothing` of the way to the target. A
    robot that re-aims on every jitter of a box whirs constantly and reads as
    nervous; one that re-aims on a real move reads as attentive. The box itself
    goes to the dashboard every cycle regardless: that costs nothing and keeps
    the drawn frame on the person.
    """

    # 0.9 s and halfway cured the twitching and made the head slow: three
    # steps and nearly three seconds to settle on a face that moved (live —
    # "it follows me very slowly"). The detect round trip is
    # 50-60 ms, so the throttle IS the latency. 0.4 s and 70% of the way puts
    # the head at 91% of the target in 0.8 s; min_move still eats the jitter.
    def __init__(self, faces, look_at, display=None, people=None, *,
                 min_interval: float = 0.4, min_move: float = 0.12,
                 smoothing: float = 0.7, clock=time.monotonic) -> None:
        self._faces = faces
        self._look_at = look_at
        self._display = display
        self._people = people
        self._min_interval = min_interval
        self._min_move = min_move
        self._smoothing = smoothing
        self._clock = clock
        self._aimed_at: list[float] | None = None
        self._aimed_when = float("-inf")
        self._held = False
        self._watching = False

    def hold(self, *, watching: bool = False) -> None:
        """Stop aiming at faces. The head was turned on purpose (Looker): what
        the camera sees now is not who the robot talks to, so nothing is
        concluded from it — People.look_away/look_back keep that person. Or the
        robot is paused facing the room (`watching`): the faces in view are
        still the people in front of it, only the head keeps still."""
        self._held = True
        self._watching = watching

    def release(self) -> None:
        # No face is marked seen here: a face nobody saw, stamped on resume,
        # once kept a person who had left "in view" for the next voice.
        self._held = False
        self._aimed_at = None  # aim again at once, wherever the face now is

    def __call__(self, _detections=None) -> None:
        faces = self._faces()
        box = faces[0].get("box") if faces else None
        if self._display is not None:
            name = self._people.current.name if self._people is not None else None
            score = float(faces[0].get("score", 0.0)) if faces else 0.0
            self._display.on_face(name, box, score)
        if box and self._people is not None and (not self._held or self._watching):
            self._people.face_seen()
        if not box or self._held:
            return
        target = _gaze_at(box)
        now = self._clock()
        if now - self._aimed_when < self._min_interval:
            return
        if self._aimed_at is not None and max(
                abs(target[0] - self._aimed_at[0]),
                abs(target[1] - self._aimed_at[1])) < self._min_move:
            return
        # Halfway there, not all the way: a face box jitters a little in every
        # frame, and a head that goes to each raw target twitches. Seen live
        # the robot looked nervous while nobody was even talking.
        if self._aimed_at is not None:
            target = [self._aimed_at[i] + self._smoothing * (target[i] - self._aimed_at[i])
                      for i in (0, 1)]
        self._aimed_at, self._aimed_when = target, now
        self._look_at(target[0], target[1])


# How long a head turn takes to reach the picture the voice loop reads.
# Measured on the robot: after /move/goto, the camera's own frame
# settled 0.73-1.64 s later; the detect loop then needs one more cycle (4 fps)
# before RemoteDetectSource.latest() holds it.
LOOK_SETTLE_S = 1.6


class Looker:
    """The `camera` tool with a direction: turn the head, wait for the picture
    to catch up, take a fresh frame.

    The frame is also stored in visual memory, marked with where the robot
    looked, so "what was on your left?" can find it later
    (demo/conversation.py's recall_seen). The head stays turned while the robot
    describes what is there; come_back() returns it to the person after the
    turn. The face tracker is held meanwhile — otherwise it pulls the head
    straight back to the face it can still see at the edge of the picture.
    """

    def __init__(self, robot, source, tracker=None, frame_memory=None, *,
                 settle_s: float = LOOK_SETTLE_S, sleep=time.sleep,
                 people=None) -> None:
        self._robot = robot
        self._people = people
        self._source = source
        self._tracker = tracker
        self._frame_memory = frame_memory
        self._settle_s = settle_s
        self._sleep = sleep
        self.turned = False
        self._stored: str | None = None
        # Who was in the last picture this took — for the note that goes with
        # it (demo/conversation.py). Read off THAT frame, never the turn's:
        # with the head turned to the wall on the left, the turn's frame still
        # shows the person the robot was facing, and the note used to tell the
        # model "Sasha is in front of you" about a picture Sasha is not in.
        # Live, asked what was on its left, the robot answered "I see a window
        # with dark brown curtains" and then "Sasha is in front of me now."
        self.names: list[str] = []

    def look(self, direction: str) -> bytes | None:
        """"left" or "right" turns the head first; "ahead" just takes the
        picture as it is now. Raises LookFailed when the head did not turn."""
        # Only this look's frame waits for a caption: one left over from an
        # earlier look got the reply about this one — "I can't turn my head"
        # became the caption of the left a turn before.
        self._stored = None
        if direction != "ahead":
            if self._tracker is not None:
                self._tracker.hold()
            if self._people is not None:
                self._people.look_away()
            self.turned = True
            try:
                self._robot.look(direction)
            except Exception as exc:  # noqa: BLE001 — said to the model instead
                # No picture and nothing stored: the camera still sees what is
                # in front, and stored as what is on the left it answered "what
                # was on your left?" wrongly for the rest of the run.
                print(f"  [look] head did not turn ({type(exc).__name__}: {exc})")
                raise LookFailed("your head could not turn "
                                 f"{TURNED.get(direction, direction)}") from exc
            self._sleep(self._settle_s)
        frame, detections = self._source.latest()
        if frame is None:
            return None
        seen = []
        if self._people is not None and self._people.enabled:
            try:
                seen = self._people.in_frame(frame)
            except Exception as exc:  # noqa: BLE001 — a picture without names
                print(f"  [look] faces skipped ({type(exc).__name__}: {exc})")
        self.names = [person["name"] for person in seen if person.get("name")]
        if self._frame_memory is not None:
            try:
                meta = {"looked": direction}
                if seen:
                    meta["people"] = seen
                self._stored = self._frame_memory.remember(frame, detections, meta=meta)
            except Exception as exc:  # noqa: BLE001 — memory is optional
                print(f"  [visual-memory] skip ({type(exc).__name__}: {exc})")
        return frame_to_jpeg(frame)

    def caption(self, reply: str | None) -> None:
        """What the robot said about the picture it just took, kept with it —
        part of the frame's words (emulator/frame_memory.py's frame_text), so
        a question about what was there finds the look and it is read out
        (demo/conversation.py's LOOKS_NOTE). "What did you see?" is answered
        from the day's pictures instead (day_frames)."""
        stored, self._stored = self._stored, None
        if stored is None or not reply or self._frame_memory is None:
            return
        try:
            self._frame_memory.describe(stored, reply)
        except Exception as exc:  # noqa: BLE001 — memory is optional
            print(f"  [visual-memory] no caption ({type(exc).__name__}: {exc})")

    def come_back(self, face_box=None) -> None:
        """After the turn: back to the face the turn started with, or ahead."""
        if not self.turned:
            return
        self.turned = False
        try:
            if face_box:
                self._robot.look_at(*_gaze_at(face_box))
            else:
                self._robot.look("ahead")
        except Exception as exc:  # noqa: BLE001
            print(f"  [look] head did not come back ({type(exc).__name__}: {exc})")
        if self._people is not None:
            # The robot looked away; the person did not leave (a long reply
            # with the head turned is no empty view).
            self._people.look_back()
        if self._tracker is not None:
            self._tracker.release()


def _gaze_at(box) -> list[float]:
    """Where to look for a box in frame fractions, in the [-1, 1] the robot's
    head takes (same convention as demo/detections.py's gaze_from_dicts)."""
    x1, y1, x2, y2 = box
    return [float((x1 + x2) - 1.0), float((y1 + y2) - 1.0)]


def _report_memory_size(display, frame_memory, speech_memory,
                        knowledge=None) -> None:
    """How much this robot is holding, for the dashboard — the number the talk
    is about. Counted after the turn, so what the turn stored is included.
    Guarded like every other memory call here: a counter must not cost a turn."""
    try:
        frames = frame_memory.count() if frame_memory is not None else 0
        exchanges = speech_memory.count() if speech_memory is not None else 0
        facts = knowledge.count() if knowledge is not None else 0
    except Exception as exc:  # noqa: BLE001 — presentation only
        print(f"  [memory] could not size the store ({type(exc).__name__}: {exc})")
        return
    display.on_memory_count(frames, exchanges, facts)


def _talk(heard, detections, endpoint, robot, display, conversation, *,
          make_player, robot_guard, audio_guard, robot_dispatcher,
          frame_memory, speech_memory, frame, turn_started_at,
          face_box=None, knowledge=None, looker=None, people=None,
          synthesizer=None) -> None:
    """The /chat part of a turn, wired to this robot's speaker, motors,
    memories, knowledge base and camera."""
    chat_endpoint = endpoint + CHAT_PATH
    # The head turn towards whoever is talking, and the console's "I see / I
    # heard" lines, come from a meta event as they always have — built here
    # now, from the detections this robot already has.
    # Look at the FACE when there is one: the head turning towards a person's
    # eyes reads as attention, while the centre of a YOLO person box is their
    # chest.
    gaze = (_gaze_at(face_box) if face_box
            else list(gaze_from_dicts(detections or [])))
    meta = [{"type": "meta", "heard": heard,
             "objects": [d.get("label") for d in detections or []],
             "detections": detections or [], "gaze": gaze}]

    def send(payload):
        if synthesizer is not None:
            # "Send me the words, I will make the sound" (demo/contract.py):
            # the Mac still cuts the reply into sentences at the same points,
            # so the robot starts talking just as early.
            payload = {**payload, "sentences": True}
        events = _http_stream(chat_endpoint, payload)
        if meta:
            events = itertools.chain([meta.pop()], events)
        done = drive_robot_stream(
            robot, events, make_player=make_player, on_reply=display.on_reply,
            robot_guard=robot_guard, audio_guard=audio_guard,
            robot_dispatcher=robot_dispatcher, synthesizer=synthesizer)
        if not done:
            # The stream ended without its `done` line: the laptop failed or
            # the connection dropped mid-reply. Said, rather than taken for
            # an empty answer and written into the conversation as one.
            raise OSError("the reply ended before it finished")
        return done

    done = None
    try:
        done = chat_turn(
            heard, window=conversation, send=send, display=display,
            # No search for a memory that is not there: the answer then says
            # the memory is off, not that it holds nothing (conversation.py).
            recall_fn=(lambda query: recall(
                query, window=conversation, speech_memory=speech_memory))
            if speech_memory is not None else None,
            recall_seen_fn=(lambda query, direction=None: recall_seen(
                query, frame_memory=frame_memory, turn_started_at=turn_started_at,
                direction=direction))
            if frame_memory is not None else None,
            look_fn=looker.look if looker is not None else None,
            look_names_fn=(lambda: list(looker.names)) if looker is not None else None,
            who_fn=lambda: who(people=people, frame=frame, frame_memory=frame_memory,
                               turn_started_at=turn_started_at),
            names_fn=lambda: _names_in(people, frame),
            move_fn=lambda how: _move(robot, how, robot_dispatcher),
            knowledge_fn=lambda query: lookup(query, knowledge=knowledge),
            day_frames_fn=lambda: day_frames(frame_memory=frame_memory,
                                             turn_started_at=turn_started_at),
            camera_jpeg=lambda: frame_to_jpeg(frame) if frame is not None else None)
    except (OSError, ValueError) as exc:
        # The brain dropped mid-turn (network, a malformed line): say nothing
        # more and listen again — the conversation so far is intact.
        print(f"  [chat] failed ({type(exc).__name__}: {exc}); listening again")
        return
    finally:
        if looker is not None:
            looker.come_back(face_box)
            # Whatever this turn looked at gets this turn's reply, or nothing:
            # a turn that failed after a look used to leave the frame waiting,
            # and the NEXT turn's reply became its caption.
            looker.caption(done.get("reply") if done else None)
    if done.get("first_sound_ms"):
        print(f"  first sound: {done['first_sound_ms']:.0f} ms")


# The `memory` shard's named vectors: BAAI/bge-small-en-v1.5 and SigLIP2 base
# patch16-224. A memory whose embedder gives another size refuses the shard
# rather than writing vectors that cannot be searched.
MEMORY_VECTORS = {"text": 384, "image": 768}

# Threads for the detector when it runs on the robot. Measured there on a
# 640x480 photo, yolo26n takes 969 ms on one thread, 634 ms on two, 591 ms on
# three and 616 ms on four: past two, a core buys little, and the other two go
# to the voice loop and the Pollen daemon, which share the same four.
LOCAL_DETECTOR_THREADS = 2


def _brain(args) -> str:
    """The laptop running the model services."""
    return getattr(args, "brain", None) or "127.0.0.1"


def _embed_port(args) -> int:
    from demo.embed_service import DEFAULT_PORT

    return getattr(args, "embed_port", None) or DEFAULT_PORT


def _shard_path(args, name: str) -> str | None:
    """`<memory-dir>/<name>`, or None for an ephemeral shard (`--memory-dir
    :memory:`, the tests)."""
    memory_dir = getattr(args, "memory_dir", None)
    if memory_dir and memory_dir != ":memory:":
        from pathlib import Path

        return str(Path(memory_dir) / name)
    return None


def build_synthesizer(args):
    """The voice this run speaks with, or None for the laptop's.

    Placed on the robot this is the same Inflect model the laptop's service
    uses, loaded here instead of there.
    """
    if not placement.on_robot(args, "tts"):
        return None
    from emulator.speech import build_synthesizer as _build

    try:
        return _build()
    except Exception as exc:  # noqa: BLE001 — named and re-raised below
        raise placement.missing("tts", "the Inflect voice and its frontend",
                                f"{type(exc).__name__}: {exc}") from exc


def build_transcriber(args):
    """The recognizer this run uses, or None for the laptop's /transcribe.

    Placed on the robot, this is moonshine-tiny — a DIFFERENT model from the
    laptop's default (Whisper), not the same one in another place, which is
    why the start report names it.
    """
    if not placement.on_robot(args, "asr"):
        return None
    from emulator.speech import build_recognizer

    try:
        recognizer = build_recognizer("moonshine")
    except Exception as exc:  # noqa: BLE001 — named and re-raised below
        raise placement.missing("asr", "moonshine-tiny, its tokenizer and the "
                                "Silero speech detector",
                                f"{type(exc).__name__}: {exc}") from exc
    return lambda audio, sample_rate: recognizer.transcribe(audio)


def build_detect_source(args, camera, detect_url: str):
    """The detector this run uses: the laptop's service, or the model here.

    Both are the same loop at the same cadence with the same listeners
    (demo/detect_source.py) — what changes is where a frame becomes boxes.
    A detector PLACED on the robot that cannot be loaded stops the start:
    falling back to the laptop would make the log say "robot" about a model
    that never ran there.
    """
    if not placement.on_robot(args, "detector"):
        return RemoteDetectSource(camera, detect_url)

    from emulator import models
    from emulator.detector import Detector

    try:
        detector = Detector(models.fetch(models.DETECTOR),
                            threads=LOCAL_DETECTOR_THREADS)
    except Exception as exc:  # noqa: BLE001 — named and re-raised below
        raise placement.missing("detector", models.DETECTOR,
                                f"{type(exc).__name__}: {exc}") from exc

    # The face boxes found on the same cycle, as the laptop's detect service
    # sends them along with its detections (demo/detect_service.py's Faces): the
    # head tracker is fed four times a second, not once a turn. From the face
    # models wherever they are placed — here, or on the laptop.
    faces = None
    if not getattr(args, "no_faces", False):
        if placement.on_robot(args, "faces"):
            from emulator.face import FaceReader

            try:
                reader = FaceReader(identities=False)
            except Exception as exc:  # noqa: BLE001 — named and re-raised below
                raise placement.missing("faces", "the face detector",
                                        f"{type(exc).__name__}: {exc}") from exc

            def faces(frame):
                return [{"box": face.box, "score": face.score}
                        for face in reader.read(frame, embed=False)]
        else:
            from demo.embed_client import RemoteFaceReader

            remote = RemoteFaceReader(_brain(args), _embed_port(args))

            def faces(frame):
                return [{"box": face["box"], "score": face["score"]}
                        for face in remote.read(frame, embed=False)]

    return LocalDetectSource(camera, detector, faces=faces)


def build_memories(args):
    """The robot's two memories, or Nones: (frame_memory, speech_memory).

    Visual = SigLIP whole-frame embeddings (emulator/frame_memory.py); speech =
    bge text of what was said (emulator/memory.py). Both live in ONE Qdrant
    Edge shard on the robot's own disk, `<memory-dir>/memory`: named vectors
    `text` and `image`, points told apart by `kind` (emulator/edge_store.py).
    An Edge shard locks its directory, so it is opened once here and handed to
    both.

    The vectors come from the laptop's embed service (demo/embed_service.py)
    unless `--on-robot embedder` loads the models here. With the laptop
    embedding, the memory is optional — unreachable, it is disabled and the
    robot still talks; placed on the robot, a failure stops the start
    (demo/placement.py). Building either memory embeds a probe to size the
    shard, which is why voice-start waits for the embed service first.

    `--memory-dir :memory:` gives an ephemeral shard in a tempdir (Edge has no
    in-memory mode), gone when the process exits — for tests.
    """
    if getattr(args, "no_memory", False):
        return None, None
    memory_path = _shard_path(args, "memory")
    local = placement.on_robot(args, "embedder")

    def unavailable(what, exc):
        # Optional only while the models are on the laptop: a run that ASKED
        # for the embeddings on the robot and cannot have them stops here,
        # rather than quietly becoming a run that remembers nothing.
        if local:
            raise placement.missing("embedder", what,
                                    f"{type(exc).__name__}: {exc}") from exc
        print(f"  {what} disabled ({type(exc).__name__}: {exc})")

    if local:
        # No `embedder=`: each memory loads its own model in this process.
        text_embedder = image_embedder = None
        where = "this robot"
    else:
        from demo.embed_client import RemoteBgeEmbedder, RemoteSiglipEmbedder

        text_embedder = RemoteBgeEmbedder(_brain(args), _embed_port(args))
        image_embedder = RemoteSiglipEmbedder(_brain(args), _embed_port(args))
        where = f"{_brain(args)}:{_embed_port(args)}"

    try:
        from demo.embed_service import HEALTH_PATH
        from emulator import embed_identity
        from emulator.edge_store import KIND, EdgeStore

        # Before the first write, not after: a shard filled with vectors from
        # one model and searched with another looks like a robot that forgot
        # everything, with no error anywhere (emulator/embed_identity.py).
        # The identity is the embedding machine's, not this one's.
        identity = (embed_identity.current() if local else
                    embed_identity.of_service(f"http://{where}{HEALTH_PATH}"))
        embed_identity.check(memory_path, identity)
        store = EdgeStore(memory_path, vectors=MEMORY_VECTORS, tenant_field=KIND)
    except Exception as exc:  # noqa: BLE001 — see unavailable()
        unavailable("memory", exc)
        return None, None
    print(f"  memory (text: bge, image: SigLIP @ {where}) -> {memory_path or ':memory:'}")

    speech_memory = None
    try:
        from emulator.memory import TextMemory

        speech_memory = TextMemory(store=store, **(
            {"embedder": text_embedder} if text_embedder is not None else {}))
    except Exception as exc:  # noqa: BLE001 — see unavailable()
        unavailable("speech memory", exc)
    frame_memory = None
    try:
        from emulator.frame_memory import FrameMemory

        # A described frame gets a `text` vector from the same bge the
        # conversation uses — locally, the one TextMemory already loaded.
        captions = (text_embedder if text_embedder is not None
                    else getattr(speech_memory, "_embedder", None))
        frame_memory = FrameMemory(store=store, text_embedder=captions, **(
            {"embedder": image_embedder} if image_embedder is not None else {}))
    except Exception as exc:  # noqa: BLE001 — see unavailable()
        unavailable("visual memory", exc)
    return frame_memory, speech_memory


# What the model's `move` tool means for the body (demo/chat_session.py's
# MOVES): the two head gestures, the dance, and everything else an emotion.
_HEAD_MOVES = ("nod", "shake")


def _names_in(people, frame) -> list[str]:
    """Everyone the robot recognises in the picture it is about to send the
    model — all of them, not just the closest face: two people in front of it
    is a demo, not an error. Falls back to whoever it is talking to when no
    face was matched this turn (demo/people.py's continuity)."""
    if people is None or not people.enabled:
        return []
    names = [person["name"] for person in people.in_frame(frame) if person.get("name")]
    if names:
        return names
    return [people.current.name] if people.current.known else []


def _move(robot, how: str, robot_dispatcher=None) -> None:
    """Run the movement the model asked for, off the turn's thread. The
    dashboard already shows the call (demo/conversation.py's chat_turn)."""
    how = (how or "nod").strip().lower()
    if how in _HEAD_MOVES:
        call, args = robot.gesture, (how,)
    elif how == "dance":
        call, args = robot.dance, ()
    else:
        call, args = robot.emotion, (how,)
    if robot_dispatcher is not None:
        robot_dispatcher.dispatch(call, *args)
    else:
        call(*args)


def build_knowledge(args):
    """The facts the robot was shipped with (demo/knowledge.py), or None.

    Restored from the snapshot into `<memory-dir>/knowledge` on every start: a
    copy, replaced each time, so the robot searches exactly the snapshot it was
    deployed with and nothing it heard can creep in. Queries are embedded
    wherever the memory's embeddings are (demo/placement.py). Optional like
    the memories while those are on the laptop: without it the robot still
    talks, and answers Qdrant questions from the model alone."""
    if getattr(args, "no_knowledge", False) or getattr(args, "no_memory", False):
        return None
    from pathlib import Path as _Path

    from demo import knowledge as kb

    snapshot = _Path(getattr(args, "knowledge_snapshot", None) or kb.SNAPSHOT_PATH)
    target = _shard_path(args, "knowledge")
    if target is None:
        import atexit
        import shutil
        import tempfile

        scratch = tempfile.mkdtemp(prefix="knowledge-")
        atexit.register(shutil.rmtree, scratch, ignore_errors=True)
        target = str(_Path(scratch) / "shard")
    local = placement.on_robot(args, "embedder")
    try:
        if local:
            from emulator.memory import DEFAULT_MODEL, _embedder

            embedder = _embedder(DEFAULT_MODEL)
        else:
            from demo.embed_client import RemoteBgeEmbedder

            embedder = RemoteBgeEmbedder(_brain(args), _embed_port(args))
        base = kb.open_knowledge(snapshot, _Path(target), embedder)
    except Exception as exc:  # noqa: BLE001 — knowledge is optional
        if local:
            raise placement.missing("embedder", "the knowledge base",
                                    f"{type(exc).__name__}: {exc}") from exc
        print(f"  knowledge disabled ({type(exc).__name__}: {exc})")
        return None
    print(f"  knowledge: {base.count()} facts restored from {snapshot.name} -> {target}")
    return base


class _FaceReaderAsDicts:
    """emulator/face.py's FaceReader answering the way RemoteFaceReader does:
    `{box, score, embedding}` dicts, which is what demo/people.py reads."""

    def __init__(self, reader) -> None:
        self._reader = reader

    def read(self, frame, *, embed: bool = True) -> list[dict]:
        return [{"box": face.box, "score": face.score, "embedding": face.embedding}
                for face in self._reader.read(frame, embed=embed)]


def build_people(args):
    """Who the robot recognises: its own face shard, and the face models.

    Same split as every other memory here — the vectors and the names live in
    a Qdrant Edge shard on the robot (`<memory-dir>/people`), the models that
    turn a frame into a vector run on the laptop (demo/embed_service.py) unless
    `--on-robot faces` loads them here. Faces are optional while the models
    are on the laptop: a robot that cannot reach them still talks, remembers
    and recalls, it just calls everyone "Person"."""
    from demo.people import People

    if getattr(args, "no_faces", False) or getattr(args, "no_memory", False):
        return People()
    path = _shard_path(args, "people")
    local = placement.on_robot(args, "faces")
    try:
        from emulator.face_memory import FaceMemory

        memory = FaceMemory(path)
        if local:
            from emulator.face import FaceReader

            reader = _FaceReaderAsDicts(FaceReader())
        else:
            from demo.embed_client import RemoteFaceReader, faces_off

            off = faces_off(_brain(args), _embed_port(args))
            if off:
                raise RuntimeError(f"the laptop has no face models: {off}")
            reader = RemoteFaceReader(_brain(args), _embed_port(args))
    except Exception as exc:  # noqa: BLE001 — faces are optional on the laptop
        if local:
            raise placement.missing("faces", "the face models or the people shard",
                                    f"{type(exc).__name__}: {exc}") from exc
        print(f"  faces disabled ({type(exc).__name__}: {exc})")
        return People()
    endpoint = f"http://{_brain(args)}:{getattr(args, 'port', None) or 9500}{NAME_PATH}"
    known = memory.people()
    print(f"  faces ({'this robot' if local else _brain(args)}) -> "
          f"{path or ':memory:'}; knows {len(known)}: {', '.join(known) or '-'}")
    return People(memory, reader, read_name=lambda heard: _read_name(endpoint, heard))


def _read_name(endpoint: str, heard: str) -> str | None:
    """Ask the model on the laptop what name was just said (demo/serve.py's
    /name): the name, or None when the answer holds none. A failure raises,
    and demo/people.py takes it as a name it did not catch."""
    return _http_post(endpoint, {"text": heard}).get("name")


# Safety margin subtracted from the utterance-anchored turn start below: the
# demoed object typically lands in view a moment BEFORE the person starts
# talking about it (place it, then ask), and the mic capture pipeline itself
# adds a little latency between a chunk being captured and reaching
# collect_utterance — both push the true start of "this turn" slightly
# earlier than the utterance's own duration alone would suggest. Kept small
# on purpose: a large guard reintroduces the bug this function fixes
# (treating a genuine past memory as part of the current turn).
TURN_START_GUARD_S = 1.5

# Cap on how much of the utterance's OWN duration counts toward backdating
# the turn start. SceneChangeWriter's writes cluster near the START of a
# turn (an object lands in view, then the question about it follows within a
# couple of seconds — see TURN_START_GUARD_S above), so protecting against
# them only needs to look a few seconds into the utterance, not its whole
# length. Without this cap the exclusion window grows with every extra
# second the presenter takes to ask — a perfectly natural longer stage
# question ("do you remember that red mug I showed you a moment ago,
# where was it?", ~5s) silently discarded a hidden object seen well before
# the question even started. Measured against the two stage cases this must
# get right: object seen 4.0s ago / 2.0s question -> recalled either way
# (well under the cap); object seen 6.0s ago / 5.0s question -> recalled only
# with the cap (uncapped, the 5.0s duration alone pushed the window back to
# 6.5s and silently dropped it).
TURN_DURATION_CAP_S = 3.0


def _turn_started_at(audio, now=None):
    """Wall-clock time this turn's utterance began speaking, anchored on the
    utterance's OWN duration (capped — see TURN_DURATION_CAP_S) rather than
    on when listening started.

    Anchoring on listening start (the old `time.time()` taken just before
    `collect_utterance` blocks) is wrong: `collect_utterance` blocks for the
    ENTIRE idle wait until someone starts speaking, which can be minutes on a
    demo floor. Every frame SceneChangeWriter (emulator/frame_memory.py)
    stored during that idle wait — including the ONE frame of an object
    shown and hidden again before the question — would then read as "part of
    this turn" to recall_seen's filter (demo/conversation.py) and get wrongly
    excluded.
    `now - min(len(audio)/16000, TURN_DURATION_CAP_S)` instead backdates only
    past the first few seconds of the utterance (plus TURN_START_GUARD_S —
    see above), not its full length — so a long question doesn't eat further
    into the past than a short one.
    """
    if now is None:
        now = time.time()
    duration = min(len(audio) / 16000.0, TURN_DURATION_CAP_S)
    return now - duration - TURN_START_GUARD_S


def _startup_display_message(args) -> str | None:
    """The URL to print once the display is up, or None for `--display none`.

    Pulled out of run_voice as its own pure function so the branch on
    `args.display` is testable without the rest of run_voice's hardware I/O
    (camera warmup, mic calibration, the `while True` listen loop). Printing
    the WRONG url here is exactly the failure this exists to prevent: for
    "remote" nothing is bound locally at all (build_display returns a
    RemoteDisplayClient that only pushes events to a dashboard on the Mac —
    see demo/display/__init__.py's build_display), so the URL that matters is
    that dashboard, not this process's own loopback, which serves no
    dashboard for "remote" to answer on.
    """
    if args.display == "web":
        return f"  web UI: http://127.0.0.1:{args.web_port}"
    if args.display == "remote":
        dashboard_host = getattr(args, "dashboard_host", None) or args.brain
        return (f"  dashboard: http://{dashboard_host}:{args.web_port} "
               f"(pushed from here, see demo/display/remote.py)")
    return None


def run_voice(args, endpoint) -> int:
    """Listen and watch continuously, respond to voice.

    Camera and microphone are both open the whole time: the camera is always
    warmed up (no black first frame) and doesn't fight over the device between
    grabs. Where each one is — the robot's, over loopback HTTP, or the
    laptop's — is MacPlatform's business (demo/platform/mac.py); the loop
    reads `source.latest()` and sends events to `display` without knowing.
    Everything is built inside try/finally, so an early failure (a server
    bind, the camera warmup) doesn't leave subprocesses or devices dangling.
    """
    platform = None
    source = None
    display = None
    restart = None
    robot_dispatcher = None
    async_scene_writer = None
    conversation = None
    people = knowledge = frame_memory = speech_memory = None
    try:
        _interrupt_on_sigterm()
        platform = MacPlatform(args)
        camera = platform.video_source()
        mic = platform.mic_source()
        robot = platform.robot()
        detect_url = f"http://{_brain(args)}:{args.detect_port}/detect"
        source = build_detect_source(args, camera, detect_url)
        display = build_display(
            args.display, camera=camera, host="127.0.0.1", port=args.web_port,
            dashboard_host=getattr(args, "dashboard_host", None) or args.brain,
            dashboard_port=args.web_port)
        source.add_listener(display.on_detections)
        # Watching for the Restart button from here on — before the camera
        # warms up and the noise is calibrated, so a bring-up that is going
        # badly can still be restarted from the screen.
        restart = _RestartWatcher(display).start()
        # Built before the loop, not per turn: a model placed on the robot
        # loads once, and a failure to load it stops the start (placement).
        transcriber = build_transcriber(args)
        synthesizer = build_synthesizer(args)
        frame_memory, speech_memory = build_memories(args)
        people = build_people(args)
        knowledge = build_knowledge(args)
        # The conversation for the whole session: what the model holds in
        # context, and the move of its oldest part into speech_memory.
        conversation = ConversationWindow(
            speech_memory,
            budget_tokens=getattr(args, "context_budget", DEFAULT_CONTEXT_BUDGET))

        if frame_memory is not None:
            # "on objects" trigger: store a frame whenever the YOLO scene
            # composition changes, off the detect loop — continuous visual
            # memory independent of speech. Registered BEFORE start() so the
            # listener list isn't mutated while the detect thread iterates it.
            # Wrapped in AsyncSceneWriter (see its docstring): the
            # actual FrameMemory.remember() — a RemoteSiglipEmbedder network
            # call — runs on its own worker thread, so a stalled Mac never
            # blocks the detect thread this listener runs on.
            from emulator.frame_memory import SceneChangeWriter

            scene_writer = SceneChangeWriter(
                frame_memory, people=people.in_frame if people.enabled else None)
            async_scene_writer = AsyncSceneWriter(scene_writer)
            source.add_listener(
                lambda dets: async_scene_writer.observe(source.latest()[0], dets))
        # Created before the detect loop starts: the face tracker below is a
        # listener on that loop, and listeners are registered before start()
        # so the list is never mutated while the detect thread walks it.
        robot_guard = _FailureGuard("robot")
        audio_guard = _FailureGuard("audio")
        # A gesture/look_at over HTTP to a slow or dropped robot must never
        # stall the event loop that feeds the audio below —
        # one background worker drains robot calls off this thread, so speech
        # keeps playing regardless of how long the robot call takes.
        robot_dispatcher = RobotDispatcher(robot_guard)
        tracker = FaceTracker(
            source.faces,
            lambda x, y: robot_dispatcher.dispatch(robot.look_at, x, y),
            display=display, people=people)
        source.add_listener(tracker)
        looker = Looker(robot, source, tracker, frame_memory, people=people)
        source.start()
        # See _startup_display_message: this used to unconditionally print
        # the LOOPBACK web-UI url for both "web" and "remote", which cost
        # real time once — the operator's confirmation that voice-start
        # finished ended with a URL that served nothing, at exactly the
        # moment (bring-up, robot possibly in another room) a wrong URL is
        # most expensive to chase.
        message = _startup_display_message(args)
        if message is not None:
            print(message)
        # Wait for the first camera frame before entering the loop. Never
        # open a second capture meanwhile: two ffmpegs on the same camera
        # hang fighting over the device.
        print("  warming up camera...")
        for _ in range(30):
            if camera.latest() is not None:
                break
            time.sleep(0.2)
        if camera.latest() is None:
            print("  WARNING: no camera frame in 6s — is another process "
                  "holding the camera?")
        # Where every model of this run actually is, before the first turn:
        # a run is verified from the log alone, and "the robot did it" is a
        # claim the log has to carry. The model's name is on the line because
        # recognition is not the same model on both sides (demo/placement.py).
        from emulator import models as _models  # lazy: only for the name

        embed_at = f"{args.brain}:{getattr(args, 'embed_port', '?')}"
        for line in placement.report(
                args,
                models={"asr": "moonshine-tiny" if placement.on_robot(args, "asr") else None,
                        "tts": "inflect-nano-v2" if placement.on_robot(args, "tts") else None,
                        "detector": _models.DETECTOR,
                        "faces": "YuNet + HSFace",
                        "embedder": "SigLIP + bge"},
                addresses={"asr": f"{args.brain}:{args.port}",
                           "tts": f"{args.brain}:{args.port}",
                           "detector": f"{args.brain}:{args.detect_port}",
                           "faces": embed_at, "embedder": embed_at},
                off=() if people.enabled else ("faces",)):
            print(line)
        # A fixed speech threshold, not one calibrated at start. Calibrated
        # from one second of sound, it came out anywhere from 0.010 to 0.114
        # across the runs in the robot's log: a start with anyone
        # talking or the motors whirring left the robot deaf for the whole run
        # (voice at a normal distance is ~0.04), and a very quiet one let every
        # noise through — 769 triggers of 858 with nothing in them, each a trip
        # to the laptop while real speech waited. --speech-threshold 0 brings
        # the calibration back.
        if args.speech_threshold > 0:
            threshold = args.speech_threshold
            print(f"  ready. Speech threshold: {threshold:.3f} (fixed). "
                  f"Just talk — Ctrl-C to quit.\n")
        else:
            print("  calibrating background noise (stay quiet ~1s)...")
            threshold = calibrate_threshold(mic.chunks(), seconds=1.0,
                                            multiplier=args.sensitivity)
            print(f"  ready. Speech threshold: {threshold:.3f}. "
                  f"Just talk — Ctrl-C to quit.\n")
        # Health of the vision path, surfaced once per transition below. The
        # signals (RemoteDetectSource.healthy(), CameraStream.alive) exist for
        # exactly this — without reading them a sustained detector/camera outage is
        # invisible beyond the detect thread's per-cycle warnings.
        detect_healthy = True
        camera_alive = True
        while True:
            if _wait_while_paused(display, tracker=tracker):
                # Whatever was said to the room while the robot was paused is
                # not for the robot — drop it rather than answer it late.
                mic.flush()
            gate = VoiceGate(threshold=threshold)
            print("  listening...")
            audio = collect_utterance(mic.chunks(), gate,
                                      max_chunks=int(args.max_utterance * 10),
                                      stop=display.is_paused)
            if audio is None:
                # The microphone's stream ended — its capture process died
                # (see the warning above it). Nothing more can be heard.
                print("  the microphone stopped — ending the run")
                break
            if audio.size == 0:
                # The pause button, pressed while the robot was waiting for
                # someone to speak: back to the top, where it waits it out.
                continue
            # Anchored on the utterance itself (see _turn_started_at), NOT on
            # when listening started: collect_utterance just blocked for the
            # entire idle wait, which can be minutes, and every frame
            # SceneChangeWriter (emulator/frame_memory.py, on the detect
            # thread) stored during that wait must still read as a genuine
            # PAST memory to recall_seen's filter (demo/conversation.py), not
            # "part of this turn".
            turn_started_at = _turn_started_at(audio)
            print(f"  heard speech ({len(audio) / 16000:.1f} s), looking...")
            frame, detections = source.latest()
            if source.healthy() != detect_healthy:
                detect_healthy = source.healthy()
                print("  vision: object detection "
                      + ("recovered" if detect_healthy
                         else "failing — continuing without detections"))
            camera_now = getattr(camera, "alive", True)
            if camera_now != camera_alive:
                camera_alive = camera_now
                print("  camera: " + ("recovered" if camera_alive
                                      else "stopped delivering frames"))
            # The turn's frame is not pushed to the dashboard: ~60 KB into the
            # event stream every turn is the size of event that stopped
            # reaching the page in Chrome. What the model sees is shown by
            # display.on_look, from the camera tool itself.
            try:
                _handle_stream(detections, audio, endpoint, robot, display,
                               make_player=platform.make_player,
                               robot_guard=robot_guard, audio_guard=audio_guard,
                               frame_memory=frame_memory,
                               speech_memory=speech_memory,
                               frame=frame, robot_dispatcher=robot_dispatcher,
                               turn_started_at=turn_started_at,
                               conversation=conversation, people=people,
                               knowledge=knowledge, looker=looker,
                               transcriber=transcriber, synthesizer=synthesizer)
            except Exception:  # noqa: BLE001 — one turn, not the demo
                # A bug in one turn must not end a demo that is in front of
                # a room: said in full, and the robot listens again. The
                # memory is untouched — every write guards itself.
                import traceback

                print("  [turn] failed — listening again")
                traceback.print_exc()
            mic.flush()   # discard the robot's own voice from the buffer
            print()
    except KeyboardInterrupt:
        # Ctrl-C, voice-stop's SIGTERM, or the Restart button's watcher
        # thread — all three land here, and the teardown below is the same.
        print("\n  bye")
    finally:
        if restart is not None:
            restart.close()
        # Stop the robot dispatcher's worker thread first: it holds a
        # reference into this function's locals (robot_guard) and nothing
        # else here depends on it, so there's no ordering reason to keep it
        # alive through the rest of teardown (Resource Lifecycle: threads
        # get an explicit stop signal and are joined on shutdown).
        if robot_dispatcher is not None:
            robot_dispatcher.close()
        # Stop the detect source BEFORE closing the display: the source fans
        # out every cycle to display.on_detections, so tearing the display
        # down first could deliver an in-flight cycle to an already-closed
        # sink. Harmless today (the listener guard swallows it), but stopping
        # the producer before its consumer is correct ordering, not just safe.
        if source is not None:
            source.stop()
        # AsyncSceneWriter's worker is a CONSUMER of the detect thread's
        # dispatch() calls (see its docstring) — stopped after source.stop()
        # so it drains whatever the last detect cycle handed it before this
        # process tears down FrameMemory/the Qdrant shard underneath it.
        if async_scene_writer is not None:
            async_scene_writer.close()
        # What is still in the model's context goes into memory too, so the
        # next session can recall this one. Before the display closes; after
        # the scene writer, so both memories are done being written by then.
        if conversation is not None:
            stored = conversation.flush()
            if stored:
                print(f"  memory: {len(stored)} exchange(s) still in context "
                      "-> Qdrant Edge")
        # The shards last: everything that writes to them (the scene writer,
        # the window's flush above) is done by now. Closing them is what puts
        # the last writes on disk — a run that was killed without this came
        # back up not knowing the person it had just met.
        for memory in (people, knowledge, frame_memory, speech_memory):
            close = getattr(memory, "close", None)
            if close is None:
                continue
            try:
                close()
            except Exception as exc:  # noqa: BLE001 — shutdown must not raise
                print(f"  [memory] could not close ({type(exc).__name__}: {exc})")
        if display is not None:
            display.close()
        if platform is not None:
            platform.close()
    # The supervisor reads this: RESTART_EXIT_CODE means "wipe the memory and
    # start me again", anything else means the run is over (see
    # scripts/voice_loop.sh).
    return RESTART_EXIT_CODE if restart is not None and restart.requested else 0


def _wait_while_paused(display, poll: float = 0.3, sleep=time.sleep,
                       tracker=None) -> bool:
    """Block while the dashboard's pause button is on; returns whether it was.

    Checked between turns, not during one: a turn already under way finishes
    speaking rather than being cut off mid-sentence. A paused robot also stops
    following faces — it should stand still, not keep turning its head at
    whoever walks past."""
    paused = False
    while display.is_paused():
        if not paused:
            print("  paused from the dashboard — not listening")
            paused = True
            if tracker is not None:
                tracker.hold(watching=True)
        sleep(poll)
    if paused:
        print("  resumed")
        if tracker is not None:
            tracker.release()
    return paused


# What this process exits with when the dashboard's Restart button was
# pressed. scripts/voice_loop.sh — the supervisor `robot_service.sh
# voice-start` runs this under — wipes the robot's memory directory and
# starts the loop again on exactly this code and NO other: a crash,
# voice-stop's SIGTERM and Ctrl-C all end the run with the memory intact.
# Erasing there rather than here is deliberate: by then this process is gone,
# so the Qdrant Edge shards are closed and their last writes are on disk (see
# emulator/edge_store.py — a shard locks its directory), and no half-written
# file survives into the fresh run.
RESTART_EXIT_CODE = 42


class _RestartWatcher:
    """Ends the run when the dashboard's Restart button is pressed.

    NOT checked between turns the way the pause is. Between turns this loop
    is usually blocked inside collect_utterance waiting for someone to speak
    — and the button gets pressed on a quiet stage, exactly when nobody is
    going to. So it watches on its own thread and raises KeyboardInterrupt in
    the main one, the same path Ctrl-C and voice-stop's SIGTERM already take
    (_interrupt_on_sigterm): run_voice's `finally` still flushes the
    conversation and closes the shards, rather than the memory being cut off
    mid-write and wiped in whatever state it was left.

    A display with no such button (NullSink, an older dashboard) simply never
    asks for anything, and an unreachable dashboard answers "no" — neither is
    a reason to restart a robot that is talking to a room.
    """

    def __init__(self, display, poll: float = 1.0) -> None:
        self._display = display
        self._poll = poll
        self._stop = threading.Event()
        self.requested = False
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="restart-watch")

    def start(self) -> "_RestartWatcher":
        # Consume a request left over from the run this one replaces: the
        # dashboard was told to restart and the old loop died before it could
        # acknowledge it. Without this the fresh loop reads the same flag and
        # restarts again — forever, wiping the memory each time round.
        if self._ask():
            print("  restart request left over from the last run — cleared")
            self._ack()
        self._thread.start()
        return self

    def _ask(self) -> bool:
        asked = getattr(self._display, "restart_requested", None)
        if asked is None:
            return False
        try:
            return bool(asked())
        except Exception as exc:  # noqa: BLE001 — the screen must not stop the robot
            print(f"  [restart] check failed ({type(exc).__name__}: {exc})")
            return False

    def _ack(self) -> None:
        ack = getattr(self._display, "ack_restart", None)
        if ack is None:
            return
        try:
            ack()
        except Exception as exc:  # noqa: BLE001
            print(f"  [restart] ack failed ({type(exc).__name__}: {exc})")

    def _run(self) -> None:
        while not self._stop.wait(self._poll):
            if not self._ask():
                continue
            self.requested = True
            print("\n  restart from the dashboard — wiping the memory and "
                  "starting over")
            self._ack()
            _thread.interrupt_main()
            return

    def close(self, timeout: float = 2.0) -> None:
        self._stop.set()
        self._thread.join(timeout=timeout)


def _interrupt_on_sigterm() -> None:
    """scripts/robot_service.sh voice-stop ends this process with SIGTERM,
    whose default kills it before any `finally` runs — and with it the flush
    that moves the conversation still in context into memory. Raise the same
    KeyboardInterrupt Ctrl-C does instead."""
    import signal

    raised = []

    def interrupt(signum, frame):
        # Once. A second SIGTERM while the `finally` below is closing the
        # shards would cut that short, and the last writes would be lost.
        if raised:
            return
        raised.append(signum)
        raise KeyboardInterrupt

    try:
        signal.signal(signal.SIGTERM, interrupt)
    except ValueError:
        pass  # not the main thread (a test driving run_voice) — nothing to do


def parse_args(argv=None):
    from demo.camera_service import DEFAULT_PORT as CAMERA_SERVICE_PORT
    from demo.embed_service import DEFAULT_PORT as EMBED_SERVICE_PORT

    p = argparse.ArgumentParser(description="The robot's voice loop")
    p.add_argument("--brain", default="127.0.0.1",
                   help="the laptop running serve.py, detect_service.py and "
                        "embed_service.py")
    p.add_argument("--port", type=int, default=9500,
                   help="demo/serve.py's port on --brain")
    # "default" is the system's own camera and microphone, which ffmpeg's
    # avfoundation input takes by that name. A fixed index was one Mac's
    # setup: index 1 is a second microphone a stock MacBook does not have.
    p.add_argument("--video", default="default",
                   help="the laptop's camera for --camera mac: \"default\" "
                        "(the system's) or an ffmpeg avfoundation index")
    p.add_argument("--audio", default="default",
                   help="the laptop's microphone for --mic mac: \"default\" "
                        "(the system's) or an ffmpeg avfoundation index")
    p.add_argument("--max-utterance", type=float, default=8.0,
                   help="max utterance length in voice mode, seconds")
    p.add_argument("--speech-threshold", type=float, default=SPEECH_THRESHOLD,
                   help="fixed loudness (RMS) a chunk must reach to count as "
                        "speech; 0 calibrates it from background noise at start")
    p.add_argument("--sensitivity", type=float, default=3.0,
                   help="with --speech-threshold 0: how many times the "
                        "threshold is above background noise")
    p.add_argument("--no-robot", action="store_true",
                   help="no Reachy: movements printed to the console, sound "
                        "played on the laptop")
    p.add_argument("--web-port", type=int, default=DEFAULT_DASHBOARD_PORT,
                   help="dashboard port: bound locally for --display web, or "
                        "the port pushed to on --dashboard-host for "
                        "--display remote")
    p.add_argument("--detect-port", type=int, default=9600,
                   help="demo/detect_service.py's port on --brain")
    p.add_argument("--embed-port", type=int, default=EMBED_SERVICE_PORT,
                   help="demo/embed_service.py's port on --brain")
    p.add_argument("--display", choices=["web", "none", "remote"], default="web",
                   help="the dashboard: served by this process (web), none, "
                        "or pushed to a standalone one on the laptop "
                        "(remote, python -m demo.display.web)")
    p.add_argument("--dashboard-host", default=None,
                   help="the laptop for --display remote; defaults to --brain")
    p.add_argument("--no-faces", action="store_true",
                   help="do not recognise people (demo/people.py); the robot "
                        "still talks, remembers and recalls")
    p.add_argument("--no-knowledge", action="store_true",
                   help="do not restore the Qdrant knowledge snapshot "
                        "(demo/knowledge.py); the robot still talks and remembers")
    p.add_argument("--knowledge-snapshot", default=None, metavar="FILE",
                   help="the knowledge snapshot restored at start "
                        "(default: demo/qdrant_knowledge.snapshot)")
    p.add_argument("--no-memory", action="store_true",
                   help="disable memory: frames (SigLIP; searched by their words) "
                        "and the conversation (bge)")
    p.add_argument("--on-robot", default=None, metavar="LIST",
                   help="which models run ON THE ROBOT, comma-separated "
                        "(asr, tts, detector, faces, embedder); anything not "
                        "named runs on --brain over HTTP. The language model "
                        "is never a choice — it does not fit in the robot's "
                        "memory. See demo/placement.py")
    p.add_argument("--context-budget", type=int, default=DEFAULT_CONTEXT_BUDGET,
                   metavar="TOKENS",
                   help="how full the model's context may get before the "
                        "oldest exchanges move into the robot's memory "
                        "(demo/conversation.py; the model's own limit is 4096)")
    p.add_argument("--memory-dir", default="results/memory", metavar="DIR",
                   help="on-disk Qdrant Edge shards for the memory, the "
                        "people and the knowledge base, so they survive a "
                        "restart; ':memory:' makes them ephemeral (tests)")
    p.add_argument("--robot-host", default="reachy-mini.local",
                   help="the Reachy daemon (:8000/api) and the robot's camera "
                        "service. A plain IP where mDNS does not resolve; "
                        "127.0.0.1 on the robot itself, or for the simulator")
    p.add_argument("--camera", choices=["mac", "robot"], default="mac",
                   help="the laptop's camera (ffmpeg), or the robot's "
                        "(demo/camera_service.py) — what the robot remembers "
                        "is what this camera saw")
    p.add_argument("--robot-camera-port", type=int, default=CAMERA_SERVICE_PORT,
                   help="port of the robot's camera+mic service on --robot-host")
    p.add_argument("--mic", choices=["mac", "robot"], default="mac",
                   help="the laptop's microphone (ffmpeg), or the robot's "
                        "(demo/camera_service.py)")
    p.add_argument("--speaker", choices=["mac", "robot"], default="mac",
                   help="afplay on the laptop, or the robot's own speaker "
                        "through its daemon (needs a robot: not --no-robot)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    from demo.logs import setup_logging

    setup_logging()
    args = parse_args(argv)
    if args.speaker == "robot" and args.no_robot:
        # Checked here, before the loop starts, so a contradictory pair fails
        # with one clear line instead of surfacing mid-turn as "[audio] skip".
        print("  ERROR: --speaker robot needs a real robot connection; "
              "drop --no-robot or remove --speaker robot")
        return 1
    endpoint = f"http://{args.brain}:{args.port}"
    print(f"voice loop. models: {endpoint}")
    return run_voice(args, endpoint)


if __name__ == "__main__":
    raise SystemExit(main())
