"""Who the robot is talking to.

One face lookup per turn, not per frame: the robot takes the frame it already
captured for the turn, asks the Mac for a face vector (emulator/face.py), and
matches it against the people it has met in its own shard
(emulator/face_memory.py). Conversation memory stays exactly as it was — read
only when the model calls `remember`. This is a different question, asked at a
different time: not "what do I remember about this" but "who is standing
there", and the answer is needed before the turn is answered, not during it.

Meeting someone is a two-turn exchange the robot drives itself: it asks for a
name, the next thing said is taken as the answer — the model on the laptop
reads the name out of it (demo/serve.py's /name) — and the shots it has
collected are enrolled under that name. The robot's two lines are fixed, not
generated, and no tool call is involved: these lines must be the same every
time, and a tool call that misfires here would greet the room with silence.

Calibrated on this robot's own camera (eight stored frames of one person):
shots of the same person taken minutes apart scored 0.60-0.77 against each
other, but two frames at a different angle and light fell to 0.25-0.40. That
is why enrollment takes several shots rather than one — the match is against
the closest of them.
"""
from __future__ import annotations

import dataclasses
import time

# Shots kept for one person at enrollment. More poses, more chances one of
# them is close to however they are standing later.
ENROLL_SHOTS = 5

# The person the robot last named stays "the person in front of it" for as long
# as a face never leaves the camera for longer than this — and only for a face
# that is too close to call (see Match.is_new), never for a clear stranger. Seen live
#: met with 2 shots, the same person two turns later scored under
# the new-person line and was asked their name again. A face that stayed in
# view is the same person, whatever the embedding says.
FACE_GONE_S = 10.0

# Shots added to that person while they stay in view and their face does not
# match — the poses enrollment missed. Capped per person: this runs every turn,
# and one visitor in bad light must not use up the next visitor's share.
MAX_LEARNED_SHOTS = 10

# What the robot says. Fixed lines, spoken through demo/serve.py's /say — the
# model is not asked to improvise the one moment the demo is named after.
ASK_NAME = "I don't think we've met. What's your name?"
GREET_KNOWN = "Hello again, {name}!"
GREET_NEW = "Nice to meet you, {name}. I'll remember you."
NOT_CAUGHT = "Sorry, I didn't catch your name."

@dataclasses.dataclass
class Seen:
    """What the robot saw this turn."""
    name: str | None
    score: float
    box: list[float] | None
    # Someone the robot cannot name has just come into view — a face that is
    # clearly nobody it has met (FaceMemory's is_new), or any face it does not
    # recognise after the view was empty: whoever was talking before, this is
    # someone else. True on the turn they appear, not on every turn they stay.
    stranger_arrived: bool = False

    @property
    def known(self) -> bool:
        return self.name is not None


class People:
    """The robot's side of meeting and recognising people."""

    def __init__(self, memory=None, reader=None, *, shots: int = ENROLL_SHOTS,
                 clock=time.monotonic, read_name=None) -> None:
        self._memory = memory
        self._reader = reader
        self._shots_wanted = shots
        self._read_name = read_name
        self._shots: list[list[float]] = []
        self.awaiting_name = False
        self.current = Seen(None, 0.0, None)
        self._greeted: set[str] = set()
        self._clock = clock
        self._here: str | None = None      # named, and still in view
        self._stranger_in_view = False     # someone unnamed, not yet named
        self._view_emptied = False         # nobody in view for FACE_GONE_S
        self._in_view_as_it_turned = False
        self._face_last_seen = float("-inf")
        self._emptied_after: float | None = None   # the spell already noticed
        self._learned: dict[str, int] = {}   # poses learned, per person

    def face_seen(self) -> None:
        """A face is in the camera now — from the detect loop (demo/run_demo.py's
        FaceTracker), several times a second. A gap longer than FACE_GONE_S
        means whoever is there now may be someone else."""
        self._notice_an_empty_view()
        self._face_last_seen = self._clock()

    def _notice_an_empty_view(self) -> None:
        """Nobody in view for FACE_GONE_S, after someone was (the run's first
        face follows nobody): whoever comes next is someone new to this moment
        — not the person named before, not the stranger whose face was being
        collected (those shots are theirs, never the next person's), and a
        name question asked of someone who left has no answer. Noticed when a
        face comes back, and on a turn that sees no face at all; once a spell."""
        last = self._face_last_seen
        if (last == float("-inf") or last == self._emptied_after
                or self._clock() - last <= FACE_GONE_S):
            return
        self._emptied_after = last
        self._here = None
        self._stranger_in_view = False
        self._shots.clear()
        self.awaiting_name = False
        self._view_emptied = True

    def look_away(self) -> None:
        """The robot is turning its head on purpose (demo/run_demo.py's
        Looker): note whether the person was still in view as it turned."""
        self._in_view_as_it_turned = (self._clock() - self._face_last_seen
                                      <= FACE_GONE_S)

    def look_back(self) -> None:
        """The head is back: the time it was turned is no gap — for a person
        who was there as it turned. One who had already left stays gone."""
        if self._in_view_as_it_turned:
            self._face_last_seen = self._clock()
        self._in_view_as_it_turned = False

    @property
    def enabled(self) -> bool:
        return self._memory is not None and self._reader is not None

    def observe(self, frame) -> Seen:
        """Look once: who is in front of the robot right now."""
        if not self.enabled or frame is None:
            return self._unchanged()
        try:
            faces = self._reader.read(frame)
        except Exception as exc:  # noqa: BLE001 — a turn must not hang on this
            print(f"  [faces] skip ({type(exc).__name__}: {exc})")
            return self._unchanged()
        face = next((f for f in faces if f.get("embedding")), None)
        if face is None:
            # No face to match this turn (turned away, too far, a bad frame) —
            # but if one was in the camera moments ago, it is still the person
            # the robot is talking to, and the picture it sends the model
            # should carry their name (demo/conversation.py's names_fn).
            self._notice_an_empty_view()
            box = faces[0]["box"] if faces else None
            self.current = Seen(self._here if self._still_here() else None, 0.0, box,
                                stranger_arrived=self._unknown_after_an_empty_view())
            return self.current
        self.face_seen()
        match = self._memory.recognize(face["embedding"])
        # Read once: in_frame on the scene writer's thread may end the
        # tracking meanwhile, and a pose learned under a name read twice
        # could go to nobody's point.
        here = self._here
        if self._tracked_at_a_bad_angle(match, here):
            return self._still_the_same_person(face, match.score, here)
        arrived = False
        if match.known:
            self._here = match.name
            self._stranger_in_view = False
            self._view_emptied = False
            self._shots.clear()
        else:
            # Nobody known, and not the tracked person at a bad angle (above):
            # someone unnamed — a clear stranger, or a face too close to call.
            # The person named before is not the one in front any more,
            # whatever the next frame shows. It is an arrival unless the same
            # unnamed person was already in view (the detect loop's face_seen
            # ends that when the view has been empty long enough).
            arrived = not self._stranger_in_view
            self._stranger_in_view = True
            self._view_emptied = False
            self._here = None
            if match.is_new and len(self._shots) < self._shots_wanted:
                # Collect while the person is here; enrollment needs several poses.
                self._shots.append(face["embedding"])
        self.current = Seen(match.name if match.known else None,
                            match.score, face["box"], stranger_arrived=arrived)
        return self.current

    def _unchanged(self) -> Seen:
        """No face could be read this turn: whoever is still in view still
        is — and after an empty view, it is someone unknown, not the last
        person named. An arrival counts on the turn it happened, not again."""
        self._notice_an_empty_view()
        self.current = Seen(self._here if self._still_here() else None, 0.0,
                            self.current.box,
                            stranger_arrived=self._unknown_after_an_empty_view())
        return self.current

    def _unknown_after_an_empty_view(self) -> bool:
        """The first unrecognised sighting after the view was empty: someone
        new to this moment, so the last person's name does not carry over."""
        if not self._view_emptied:
            return False
        self._view_emptied = False
        # Someone the camera can see stays "the unnamed person in view"; a
        # voice from outside the picture does not — the next face to step in
        # is its own arrival, not taken for them.
        self._stranger_in_view = (self._clock() - self._face_last_seen
                                  <= FACE_GONE_S)
        return True

    def in_frame(self, frame) -> list[dict]:
        """Who is in a frame: [{"name", "box", "score"}], largest face first,
        name None for someone the robot has not met. Stored with a frame
        (emulator/frame_memory.py). Does not enrol or greet — that stays with
        observe(). A frame whose faces cannot be read is kept without names."""
        if not self.enabled or frame is None:
            return []
        try:
            faces = self._reader.read(frame)
        except Exception as exc:  # noqa: BLE001 — a frame is kept without names
            print(f"  [faces] skip ({type(exc).__name__}: {exc})")
            return []
        return self._named(faces)

    def faces_in(self, frame) -> list[dict]:
        """in_frame, raising when the faces cannot be read: the `who` answer
        (demo/conversation.py) has to tell "nobody is there" from "I cannot
        tell who is there"."""
        if not self.enabled or frame is None:
            return []
        return self._named(self._reader.read(frame))

    def _named(self, faces) -> list[dict]:
        found = []
        for face in faces:
            if not face.get("embedding"):
                continue
            match = self._memory.recognize(face["embedding"])
            name = match.name if match.known else None
            if name is None and not found:
                if self._tracked_at_a_bad_angle(match, self._here):
                    name = match.name   # the tracked person: the name it matched
                else:
                    # Between turns too (a stored frame, a look): the face in
                    # front is not the tracked person, so a later frame must
                    # not give them that name — the same rule as observe().
                    self._here = None
            found.append({"name": name, "box": face.get("box"),
                          "score": round(float(match.score), 3)})
        return found

    def met(self) -> list[str]:
        """Everyone this robot has met, by name."""
        return self._memory.people() if self.enabled else []

    def met_when(self) -> dict[str, float]:
        """Name -> when the robot first met them (emulator/face_memory.py's
        met_at, stamped once and never moved by a later pose). Empty without a
        face shard, or on a shard written before the field existed."""
        if not self.enabled:
            return {}
        people_met = getattr(self._memory, "people_met", None)
        if people_met is None:
            return {}
        return {name: ts for name, ts in people_met()}

    def close(self) -> None:
        """Release the faces shard — the last enrolment lands on disk here."""
        if self._memory is not None:
            self._memory.close()

    def _tracked_at_a_bad_angle(self, match, here: str | None) -> bool:
        """Under the match line, with the person last named still in view:
        them, at an angle enrollment missed. Never a face that is clearly
        someone else (match.is_new), nor one nearest to another person the
        robot has met — seen live: a second person in front of the robot was
        called Sasha, and their face was learned into Sasha's point."""
        return (here is not None and not match.known and not match.is_new
                and match.name == here
                and self._clock() - self._face_last_seen <= FACE_GONE_S)

    def _still_here(self) -> bool:
        return (self._here is not None
                and self._clock() - self._face_last_seen <= FACE_GONE_S)

    def _still_the_same_person(self, face, score: float, name: str) -> Seen:
        """The face does not match, but it never left the camera: it is the
        person already named. Learn this pose, so it matches next time."""
        learned = self._learned.get(name, 0)
        if learned < MAX_LEARNED_SHOTS:
            try:
                self._learned[name] = learned + self._memory.enroll(name, [face["embedding"]])
            except Exception as exc:  # noqa: BLE001 — a turn must not hang on this
                print(f"  [faces] could not learn a pose ({type(exc).__name__}: {exc})")
        self._shots.clear()
        self.current = Seen(name, score, face["box"])
        return self.current

    def greeting(self) -> str | None:
        """A line to say before answering, or None. Said once per person per
        run: a robot that greets you on every turn is a robot with a glitch,
        not a memory."""
        if self.current.known and self.current.name not in self._greeted:
            self._greeted.add(self.current.name)
            return GREET_KNOWN.format(name=self.current.name)
        return None

    def should_ask_name(self) -> bool:
        """Whether to ask who this is: somebody is there, nobody we know, and
        we have shots to remember them by."""
        return (self.enabled and not self.awaiting_name
                and not self.current.known and bool(self._shots))

    def ask_name(self) -> str:
        self.awaiting_name = True
        return ASK_NAME

    def answer_name(self, heard: str) -> tuple[str | None, str]:
        """Take the answer to the name question: returns (name, what to say).
        A name that could not be made out is not enrolled — a wrong name
        attached to a face outlives the mistake.

        `read_name` (demo/run_demo.py, the laptop's /name) asks the model
        what the name was. A model that cannot be asked made nothing out,
        and no pattern over the words stands in for it."""
        self.awaiting_name = False
        name = None
        if self._read_name is not None:
            try:
                name = self._read_name(heard)
            except (OSError, ValueError, AttributeError) as exc:
                # The laptop could not be asked, or answered nonsense: not
                # caught, said so. A bug in the reader is not this — it
                # reaches the turn's own handler with its traceback. Why no
                # pattern stands in: demo/serve.py's NAME_SYSTEM.
                print(f"  [people] the name reader failed "
                      f"({type(exc).__name__}: {exc})")
        if not name:
            return None, NOT_CAUGHT
        stored = self._memory.enroll(name, self._shots)
        self._shots.clear()
        self._greeted.add(name)
        self._here = name
        self._stranger_in_view = False
        print(f"  people:  met {name} ({stored} shot(s)) -> Qdrant Edge")
        self.current = Seen(name, 1.0, self.current.box)
        return name, GREET_NEW.format(name=name)
