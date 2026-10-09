"""The robot's side of the conversation: what the model still holds in its
context, what moves into Qdrant when that fills up, and the tools the model
calls to reach memory, its camera and its head.

The model runs on the Mac (demo/serve.py's /chat, demo/chat_session.py) and
keeps the recent turns in a live chat. The robot keeps the same turns here — the
authoritative copy, sent with every request — and after each turn reads back
how many tokens the model's context now holds. Past the budget, the oldest part
of the conversation is written into the robot's own Qdrant Edge shard as one
batch and dropped from the window; the next request carries the shorter
history and the Mac rebuilds its chat from it. The conversation the model sees
stays small, and everything older lives only on the robot.

Memory is read only when the model asks for it: it calls `remember`, and the
robot answers from Qdrant — words, a stored frame, or one of the facts it was
shipped with (demo/knowledge.py), which are not memory at all. `camera` is a
picture from its camera now, and `move` a gesture.
"""
from __future__ import annotations

import base64
import dataclasses
import time

from demo.contract import encode_chat_request

# The context budget, in tokens. The model's hard limit is 4096 (measured);
# this sits far below it ON PURPOSE for the stage, so the audience can watch
# memory take over from context. 500 was too tight: the system prompt and
# tool schemas already take ~250, an exchange ~30-45, and a turn that calls
# `recall` adds the call and its result on top — the window collapsed to two
# or three exchanges and the robot came across as forgetful mid-conversation.
# 600 keeps four or five exchanges — enough that the robot does not come
# across as forgetful mid-conversation — and still fills inside a few minutes
# on stage, which 900 did not.
# 750 once the `knowledge` tool and the camera's direction took
# the system prompt and tool schemas to 512 tokens with one short exchange
# (measured), and at 600 the robot moved an exchange into Qdrant on every turn
# of a live run — the window held two or three. 750 gives back the room 600
# had before.
# 1000 now: with five tools the system prompt and
# schemas take about 640 tokens on their own, and a replay of a live
# conversation at 750 kept two or three exchanges and evicted after every tool
# result; at 1000 it kept up to six.
DEFAULT_CONTEXT_BUDGET = 1000

# How much of the window one eviction moves, oldest first. Half, not one
# exchange at a time: every eviction makes the Mac rebuild its chat (one
# prefill of the remaining history), so evicting in batches pays that rarely.
EVICT_FRACTION = 0.5

# A cap on exchanges regardless of tokens. The token count only arrives after
# a turn the main chat answered — image turns report none — and a failing
# store keeps exchanges in the window rather than lose them; neither may grow
# it without bound.
MAX_EXCHANGES = 24

# Tool calls the model may chain in one turn before the robot gives up on it.
MAX_TOOL_ROUNDS = 2

# How many of the latest stored exchanges answer a recall that matched nothing.
RECENT_EXCHANGES = 4

LOOK_NOTE = "(Your camera, right now.)"

# `remember` is the one memory tool (demo/chat_session.py); the older names are
# still answered, so a model that reaches for one of them still gets memory.
MEMORY_TOOLS = ("remember", "recall", "recall_seen", "knowledge", "who")
# The `camera` tool's directions that turn the head (demo/chat_session.py's
# DIRECTIONS), and how the note that comes with the picture says it.
TURNED = {"left": "to your left", "right": "to your right"}
# In the robot's own words: the model repeats these as they are, and "to your
# left" came back out loud as "a lamp to your left" (measured).
WHERE = {"ahead": "in front of me", "left": "on my left", "right": "on my right"}

# How an exchange nobody was named for is written (Exchange.text). A name comes
# from the robot's faces: one it recognises (ConversationWindow.set_speaker),
# or the answer to its name question, which the model reads (introduce).
# Nothing picks a name out of what is said in passing: a pattern over the
# words needed a list of words that are not names ("I'm giving", "I'm not"),
# and a wrong name on someone's words outlives the mistake — unnamed is the
# safe outcome.
UNNAMED_LABEL = "Person"


@dataclasses.dataclass
class Exchange:
    person: str
    reply: str
    said_at: float
    speaker: str | None = None   # None: nobody named for it yet
    # Said while a stranger was in front (ConversationWindow.someone_new):
    # which one, so only that person's own name ever names it.
    stranger: int | None = None
    # A turn whose answer came out of memory (the model called `remember` —
    # chat_turn says exactly when). It stays in the window for the
    # conversation to flow, but it never goes BACK into memory: measured
    # live, storing them built a loop — the robot answered "I recall we were
    # discussing something emotional", that sentence was stored, and the next
    # "what did we discuss?" recalled it instead of the thing actually
    # discussed, three hits deep.
    derived: bool = False

    @property
    def text(self) -> str:
        """How an exchange is stored and searched: both sides, one line — see
        EXCHANGE_KIND in emulator/memory.py for why. Named, once the person
        has said who they are."""
        return f"{self.speaker or UNNAMED_LABEL}: {self.person} — Reachy: {self.reply}"


class ConversationWindow:
    """The exchanges still in the model's context, and their move into memory.

    `memory` is the robot's TextMemory (emulator/memory.py) or None, in which
    case evicted exchanges are simply dropped — the context stays bounded
    either way.
    """

    def __init__(self, memory=None, *,
                 budget_tokens: int = DEFAULT_CONTEXT_BUDGET,
                 evict_fraction: float = EVICT_FRACTION,
                 max_exchanges: int = MAX_EXCHANGES, clock=time.time) -> None:
        self._memory = memory
        self._budget = budget_tokens
        self._fraction = evict_fraction
        self._max = max_exchanges
        self._clock = clock
        self._exchanges: list[Exchange] = []
        self._speaker: str | None = None
        self._stranger: int | None = None   # the stranger in front, if any
        self._strangers = 0

    @property
    def budget(self) -> int:
        """The token budget the context is measured against (the dashboard
        draws the fill against it)."""
        return self._budget

    @property
    def history(self) -> list[tuple[str, str]]:
        return [(e.person, e.reply) for e in self._exchanges]

    @property
    def speaker(self) -> str | None:
        """Who the robot is talking to, or None while nobody is named."""
        return self._speaker

    def set_speaker(self, name: str) -> None:
        """The person the robot is talking to, from a face it recognised
        (demo/people.py).

        What was said before anyone was recognised is theirs too: the person
        was talking before the camera knew them. What a stranger said is not,
        nor what another person was named for — called every turn, this used
        to hand words still in the window to whoever stepped in next."""
        if not name:
            return
        self._stranger = None
        if name == self._speaker:
            return
        self._speaker = name
        for exchange in self._exchanges:
            if exchange.speaker is None and exchange.stranger is None:
                exchange.speaker = name

    def someone_new(self) -> None:
        """A face the robot has never met has just come into view
        (demo/people.py's Seen.stranger_arrived): what is said from now on is
        not the last person's, and stays unnamed until this person says who
        they are (introduce)."""
        self._strangers += 1
        self._stranger = self._strangers
        self._speaker = None

    def introduce(self, name: str) -> None:
        """The person said who they are, answering the robot's name question
        (demo/people.py). Their own unnamed words take the name: the current
        stranger's, and what was said before anyone could be told apart;
        another stranger's, or anyone named, stay as they are."""
        for exchange in self._exchanges:
            if exchange.speaker is None and exchange.stranger in (None, self._stranger):
                exchange.speaker = name
        self._speaker = name
        self._stranger = None

    def add(self, person: str, reply: str, *, derived: bool = False) -> None:
        self._exchanges.append(Exchange(
            person, reply, self._clock(), derived=derived, speaker=self._speaker,
            stranger=self._stranger if self._speaker is None else None))

    def after_turn(self, token_count: int | None) -> list[str]:
        """Evict when the context is over budget (or the window over its cap).
        Returns the exchanges written to memory, for the dashboard."""
        over_budget = token_count is not None and token_count > self._budget
        over_cap = len(self._exchanges) > self._max
        if not (over_budget or over_cap) or len(self._exchanges) < 2:
            return []
        count = max(1, int(len(self._exchanges) * self._fraction))
        return self._move_to_memory(count)

    def flush(self) -> list[str]:
        """Everything still in context into memory — at shutdown, so the next
        session can recall this one."""
        return self._move_to_memory(len(self._exchanges))

    def search(self, query: str, min_score: float) -> list[dict]:
        """Exchanges still in context that match `query`, as recall hits.
        Measured: asked "what's my name?" with the answer two turns up, the
        model still called `recall` — and told only that nothing was stored,
        it answered "I don't have your name… Sasha". Searching the window
        too makes recall right wherever the answer is."""
        if self._memory is None or not self._exchanges:
            return []
        texts = [e.text for e in self._exchanges]
        scores = self._memory.score_texts(query, texts)
        return [{"text": text, "score": score, "source": "context"}
                for text, score in zip(texts, scores) if score >= min_score]

    def _move_to_memory(self, count: int) -> list[str]:
        batch = self._exchanges[:count]
        if not batch:
            return []
        stored: list[str] = []
        if self._memory is not None:
            from emulator.memory import EXCHANGE_KIND

            try:
                for exchange in batch:
                    if exchange.derived:
                        continue  # an answer read back out of memory (see Exchange)
                    self._memory.remember(exchange.text, EXCHANGE_KIND,
                                          {"said_at": exchange.said_at})
                    stored.append(exchange.text)
            except Exception as exc:  # noqa: BLE001 — the conversation goes on
                print(f"  [memory] could not store {len(batch)} exchange(s) "
                      f"({type(exc).__name__}: {exc})")
                if len(self._exchanges) <= self._max:
                    # Keep them in context and retry after the next turn:
                    # losing what was said is worse than a context that stays
                    # over budget a little longer. The cap still bounds it.
                    return []
                print(f"  [memory] window over its cap — dropping {count} "
                      "exchange(s) unsaved")
        del self._exchanges[:count]
        return stored


class LookFailed(Exception):
    """The `camera` tool could not look where it was asked: the head did not
    turn (demo/run_demo.py's Looker). Said to the model as it is — the
    picture it would get instead is what is in front, not what is there."""


class MemoryUnavailable(Exception):
    """A memory search that failed — as opposed to one that found nothing.
    The model is told which: answering "nothing in your memory" when the
    store could not be read had the robot deny what it remembered."""


@dataclasses.dataclass
class Recalled:
    """What one `recall` call found — words only; frames are `recall_seen`.

    `memories` and `in_context` are kept apart on purpose. Memory is what the
    robot no longer holds — that is what the room is shown as a recall, and
    what makes the demo's point. A line still inside the model's context
    window was not remembered: it is in front of the model already, said
    moments ago. Feeding those back as memories had the robot "recalling"
    its own last sentence, with the projector calling it recall (seen
    live)."""
    memories: list[str]      # out of Qdrant: what left the context
    speech_hits: list[dict]  # the same, with scores, for the dashboard
    in_context: list[str] = dataclasses.field(default_factory=list)
    # When nothing matched: the last exchanges in memory, oldest first.
    recent: list[str] = dataclasses.field(default_factory=list)


def recall(query: str, *, window: ConversationWindow, speech_memory=None,
           k: int = 3) -> Recalled:
    """The `recall` tool: what was SAID — stored exchanges, and the ones
    still in context, by meaning (bge, one gate for both). Never a picture:
    one tool that answered with words or a frame left the robot guessing
    which, and a guess showed the model a frame of the room in reply to a
    sentence about a database (demo/chat_session.py)."""
    hits: list[dict] = []
    in_context: list[dict] = []
    if speech_memory is not None:
        from emulator.memory import EXCHANGE_RECALL_MIN_SCORE

        try:
            hits = [{**hit, "source": "qdrant"}
                    for hit in speech_memory.recall_exchanges(query, k)]
        except Exception as exc:  # noqa: BLE001 — said to the model, below
            raise MemoryUnavailable(f"exchanges: {type(exc).__name__}: {exc}") from exc
        try:
            in_context = window.search(query, EXCHANGE_RECALL_MIN_SCORE)
        except Exception as exc:  # noqa: BLE001
            print(f"  [recall] context search skipped ({type(exc).__name__}: {exc})")
    hits = sorted(hits, key=lambda hit: hit["score"], reverse=True)[:k]
    in_context = sorted(in_context, key=lambda hit: hit["score"], reverse=True)[:k]
    recent: list[str] = []
    if not hits and speech_memory is not None:
        # "What did we talk about?" names no topic, so nothing clears the
        # gate — and "nothing about this in your memory" had the robot deny a
        # conversation it had just had. The latest exchanges answer it.
        try:
            recent = speech_memory.latest_exchanges(RECENT_EXCHANGES)
        except Exception as exc:  # noqa: BLE001 — said to the model, below
            raise MemoryUnavailable(f"exchanges: {type(exc).__name__}: {exc}") from exc
    return Recalled(memories=[hit["text"] for hit in hits], speech_hits=hits,
                    in_context=[hit["text"] for hit in in_context], recent=recent)


def recall_seen(query: str, *, frame_memory=None,
                turn_started_at: float | None = None,
                direction: str | None = None) -> list[dict]:
    """The `recall_seen` tool: the stored frames whose WORDS hold what `query`
    asked about — labels, names, the side it looked, the caption
    (emulator/frame_memory.py's recall_text) — best first. A question about a
    thing is a question in words: measured on the robot's own 371 frames,
    that search finds 32 right frames of 36 where a SigLIP text-to-image
    search (since removed) found 8.

    direction ("left"/"right") gives the last frame taken with the head turned
    that way: "what was on your left?" is about the LAST look there, and a
    search by words would rank the best-matching look that way first.

    A question that named nothing in particular — "what did you see today?"
    — finds nothing here: recall_text drops a frame that scores no better than
    an empty one. That is what sends a `seen` question to the day's frames
    (_answer_tool). Nothing falls back to the nearest picture: no gate on the
    picture search told the day from a thing, and it is gone
    (emulator/frame_memory.py's docstring has the numbers).

    turn_started_at drops frames stored during this very turn — the scene
    writer stores one the moment a new object lands in view, often while the
    question about it is still being asked (demo/run_demo.py's
    _turn_started_at); that frame is the present, not a memory."""
    if frame_memory is None:
        return []
    try:
        if direction in TURNED:
            return frame_memory.latest_looks([direction], limit=1,
                                             before=turn_started_at)
        return frame_memory.recall_text(query, before=turn_started_at)
    except Exception as exc:  # noqa: BLE001 — said to the model
        raise MemoryUnavailable(f"frames: {type(exc).__name__}: {exc}") from exc


def day_frames(*, frame_memory=None, turn_started_at: float | None = None) -> list[dict]:
    """The frames that stand for the day, for a question about what was seen
    that no frame's words answer (recall_seen).

    Pictures, not a list of objects: a frame IS the memory, and the labels are
    metadata that only help pick which one. The
    picking is emulator/frame_memory.py's day_frames — the newest frame, then
    whatever looks least like it — because the newest four frames of a real
    evening were the same picture four times (closest pair 0.914) and picked
    this way they are four different scenes (0.609)."""
    if frame_memory is None:
        return []
    try:
        return frame_memory.day_frames(before=turn_started_at)
    except Exception as exc:  # noqa: BLE001 — said to the model
        raise MemoryUnavailable(f"frames: {type(exc).__name__}: {exc}") from exc


def lookup(query: str, *, knowledge=None, k: int = 3) -> list[dict]:
    """The `knowledge` tool: facts from the knowledge base restored from a
    snapshot at start (demo/knowledge.py) — what the robot was taught, not
    anything it lived through."""
    if knowledge is None:
        return []
    try:
        return knowledge.search(query, k)
    except Exception as exc:  # noqa: BLE001 — said to the model
        raise MemoryUnavailable(f"knowledge: {type(exc).__name__}: {exc}") from exc


# What `who` says when it cannot tell who is there: faces off (no face models,
# --no-faces), no picture this turn, or the faces in it could not be read. The answer also
# carries {FACES_OFF: True}, which is what the code reads — never the note.
FACES_UNAVAILABLE_NOTE = "You cannot recognise faces right now."
FACES_OFF = "faces_off"


def who(*, people=None, frame=None, frame_memory=None,
        turn_started_at: float | None = None, clock=time.time) -> dict:
    """Who is in front of the robot now, by face (the faces shard), who it saw
    EARLIER and is no longer looking at (the names kept with stored frames),
    and everyone it has met.

    Somebody standing in front of the camera is not a memory of them. Live:
    asked what it had seen today, the robot answered "I saw Sasha
    moments ago" — about the person it was talking to, from a frame stored
    seconds earlier. Whoever is here now is left out of `seen_earlier`; the
    model is told they are here by other means (the note on a camera
    picture, the greeting), and every frame carries its own names anyway."""
    if people is None or not people.enabled:
        return {FACES_OFF: True, "note": FACES_UNAVAILABLE_NOTE}
    now = clock()
    # No picture is not nobody there: told "nobody is in front of your
    # camera", the `me` answer called a person the robot had met a stranger.
    here, readable = [], frame is not None
    if readable:
        try:
            here = people.faces_in(frame)
        except Exception as exc:  # noqa: BLE001 — who must not break the turn
            print(f"  [who] faces could not be read ({type(exc).__name__}: {exc})")
            readable = False
    result: dict = {"in_front_of_you": [p["name"] for p in here if p.get("name")]}
    strangers = sum(1 for p in here if not p.get("name"))
    if strangers:
        result["people_you_have_not_met_in_front_of_you"] = strangers
    if not readable:
        result[FACES_OFF] = True
        result["note"] = FACES_UNAVAILABLE_NOTE
    elif not here:
        result["note"] = "Nobody is in front of your camera right now."
    if frame_memory is not None:
        try:
            seen = frame_memory.people_seen(before=turn_started_at)
        except Exception as exc:  # noqa: BLE001 — who must not break the turn
            print(f"  [who] frames skipped ({type(exc).__name__}: {exc})")
            seen = []
        now_here = {p.get("name") for p in here if p.get("name")}
        # Whoever is here now is not a memory of them — except when the
        # question IS about them (demo/conversation.py's ME case), so that
        # sighting is kept, apart, instead of being dropped.
        result["you_last_saw_them"] = [f"{name}, {_ago(now - ts)}"
                                       for name, ts in seen if name in now_here]
        seen = [(name, ts) for name, ts in seen if name not in now_here]
        if seen:
            result["seen_earlier"] = [f"{name}, {_ago(now - ts)}" for name, ts in seen]
    met = people.met()
    if met:
        result["people_you_have_met"] = met
    # When the robot met the person in front of it — a first-enrolment stamp
    # that learning a new pose does not move (emulator/face_memory.py's
    # met_at). "When did we meet?" has no other answer.
    when_met = getattr(people, "met_when", None)
    if when_met is not None and result["in_front_of_you"]:
        try:
            stamps = when_met()
        except Exception as exc:  # noqa: BLE001 — who must not break the turn
            print(f"  [who] met_at skipped ({type(exc).__name__}: {exc})")
            stamps = {}
        result["you_met"] = [f"{name}, {_ago(now - stamps[name])}"
                             for name in result["in_front_of_you"] if name in stamps]
    return result


def _shown(frames: list[dict]) -> list[dict]:
    """Frames for the projector: those whose words answered (recall_text) and
    the day's. None is a guess, so none is dimmed."""
    return [{**frame, "weak": False} for frame in frames]


def frame_names(frame: dict) -> list[str]:
    """Who the robot recognised in a stored frame, by name."""
    names = list(frame.get("names") or [])
    if names:
        return names
    return [person["name"] for person in frame.get("people") or []
            if person.get("name")]


def memory_note(frame: dict, now: float) -> str:
    """The words that come with a recalled frame. Measured on real frames:
    "(What you saw N days ago.)" still got "I see a room…" in 2 answers of 4;
    saying plainly that it is a memory, not something in front of the robot or
    sent to it, got 0 of 4.

    It also says WHO was in the picture. The frame has known that since faces
    were stored with it, and demo/chat_session.py's IMAGE_SYSTEM already
    promises "when the note names the people in it, call them by their name"
    — but this note never named anyone, so a recalled picture of the person
    the robot had met came back as "a man with blonde hair"."""
    where = f" {TURNED[frame['looked']]}" if frame.get("looked") in TURNED else ""
    names = frame_names(frame)
    who_was_there = ""
    if names:
        who_was_there = (f" {' and '.join(names)} "
                         f"{'was' if len(names) == 1 else 'were'} in it.")
    return (f"(This is your MEMORY of what you saw{where} {_ago(now - frame.get('ts', now))} "
            "— it is not in front of you now, and nobody sent it to you."
            f"{who_was_there} Answer in the past tense.)")


def _ago(seconds: float) -> str:
    if seconds < 60:
        return "moments ago"
    if seconds < 3600:
        minutes = round(seconds / 60)
        return "a minute ago" if minutes == 1 else f"{minutes} minutes ago"
    if seconds < 86400:
        hours = round(seconds / 3600)
        return "an hour ago" if hours == 1 else f"{hours} hours ago"
    days = round(seconds / 86400)
    return "a day ago" if days == 1 else f"{days} days ago"


def chat_turn(heard: str, *, window: ConversationWindow, send, recall_fn,
              camera_jpeg, display, recall_seen_fn=None, knowledge_fn=None,
              look_fn=None, who_fn=None, names_fn=None, move_fn=None,
              day_frames_fn=None, look_names_fn=None,
              max_tool_rounds: int = MAX_TOOL_ROUNDS, clock=time.time) -> dict:
    """One voice turn through /chat; returns the final done event.

    send(payload) posts one request and plays what comes back (speech,
    motion), returning its done event — demo/run_demo.py wires it to
    _http_stream + drive_robot_stream. recall_fn(query) -> Recalled searches
    what was said, recall_seen_fn(query) -> frames what was seen — None for
    a memory the robot does not have, which the model is told is off —
    knowledge_fn(query) -> facts what it was taught, camera_jpeg() -> bytes | None is the `camera` tool's picture,
    and look_fn(direction) -> bytes | None the picture after the head turned
    "left" or "right" for it (demo/run_demo.py's Looker), raising LookFailed
    when the head did not turn. who_fn() -> dict
    answers `who`, and names_fn() -> the names of the people in front of the
    robot right now, so a picture comes with them. move_fn(how) moves the body
    for the `move` tool.
    """
    history = window.history
    payload = encode_chat_request(history, heard)
    done: dict = {}
    from_memory = False
    looked = False
    moved = False
    for _round in range(max_tool_rounds + 1):
        done = send(payload) or {}
        call = done.get("tool_call")
        if not call:
            break
        # The call is run as the model made it. Nothing here reads the words
        # of the question to change the tool or its `about`: which tool, and
        # which half of the past, is the model's decision (demo/chat_session.py).
        name, arguments = call.get("name", ""), call.get("arguments") or {}
        print(f"  tool:    {name}({arguments})")
        display.on_tool_call(name, arguments)
        try:
            payload, used_memory = _answer_tool(
                name, arguments, heard, history, recall_fn, recall_seen_fn,
                camera_jpeg, display, clock, knowledge_fn, look_fn, who_fn,
                names_fn, move_fn, moved, day_frames_fn, look_names_fn)
        except MemoryUnavailable as exc:
            print(f"  [memory] could not be searched ({exc})")
            payload, used_memory = encode_chat_request(history, heard, tool_result={
                "name": name, "result": {"note": MEMORY_UNAVAILABLE_NOTE}}), False
        moved = moved or name == "move"
        # A reply built from what memory handed back is not written back
        # (Exchange.derived) — only that. The model searches before answering
        # a statement too: live, it called remember(about=said) on "My dog is
        # called Rex.", found nothing, and the exchange was never stored, so
        # "what is my dog called?" later had nothing to find.
        from_memory = from_memory or used_memory
        looked = looked or name in ("camera", "look")
    else:
        # Out of tool rounds with still nothing said. The answer to the last
        # call is built and unsent; send it with a line that ends the chain,
        # rather than returning here — that returned the tool call itself,
        # whose reply is empty, and the robot stood there saying NOTHING for
        # the whole turn. A bounded chain is right; silence on stage is not.
        print("  [chat] the model kept calling tools; asking it for an answer")
        _ask_for_words(payload)
        done = send(payload) or {}
    reply = done.get("reply", "")
    # A look is stored ONCE, as the frame's caption (demo/run_demo.py's
    # Looker.caption) — findable by its words through the frame, shown as the
    # picture it is. Written into the conversation as an exchange too, it had
    # "what did we talk about?" answered "we talked about what I saw in the
    # room" (live): a look is not a conversation, and `seen` does not read
    # the conversation anyway.
    if reply and not looked:
        # A look stays out of the conversation the model sees. In it, "Look to
        # your left" answered "I see a lamp" with no camera in sight, and the
        # model copied that: the next "look to your right" got an invented
        # "a wooden desk with a lamp" — the camera was called 6 times in 16
        # with a look in the history, 12 in 12 without (measured). What the
        # robot said about the picture is kept with the frame instead
        # (demo/run_demo.py's Looker.caption).
        # An answer read out of memory is never written back (Exchange.
        # derived). `from_memory` is the turn's own record of that — memory
        # answered, or the model called `remember` about what was seen, said
        # or the person asking — not a reading of the question.
        # An answer the model gave with no tool at all is an ordinary exchange
        # whatever it was asked: nothing but the words could tell "Did you see
        # the pollution? — I do not see any pollution in front of me" (live)
        # from any other reply. What kept that line from coming
        # back as a memory of seeing is on the reading side: a `seen` question
        # never reads the conversation (_answer_tool).
        window.add(heard, reply, derived=from_memory)
    stored = window.after_turn(done.get("token_count"))
    if stored:
        print(f"  memory:  {len(stored)} exchange(s) -> Qdrant Edge")
        display.on_memory_write(stored)
    # Reported AFTER the eviction, and with the token count from before it: on
    # the turn that overflows, the bar reaches the budget and the "-> Qdrant
    # Edge" line appears beside it; the next turn shows it dropped.
    display.on_context(done.get("token_count"), window.budget, len(window.history))
    return done


def _answer_tool(name, arguments, heard, history, recall_fn,
                 recall_seen_fn, camera_jpeg, display, clock,
                 knowledge_fn=None, look_fn=None, who_fn=None,
                 names_fn=None, move_fn=None, moved=False,
                 day_frames_fn=None, look_names_fn=None) -> tuple[dict, bool]:
    """The follow-up request that answers one tool call, and whether it carried
    memories — a reply built from those is not stored back (Exchange.derived).
    A tool result goes back under the name the model called, so it matches
    the call it answers."""
    if name == "move":
        how = str(arguments.get("how") or "nod")
        if move_fn is not None:
            move_fn(how)
        return encode_chat_request(history, heard, tool_result={
            "name": name, "result": {"moved": how}}), False
    # "look" was the camera tool's first name (demo/chat_session.py); taken
    # too, so a model that reaches for the old word still gets a picture.
    if name in ("camera", "look"):
        direction = arguments.get("direction")
        turned = direction in TURNED and look_fn is not None
        if look_fn is not None:
            # Straight ahead goes through the looker too: that picture is kept
            # and described like the others, for "what did you see?" later.
            try:
                jpeg = look_fn(direction if turned else "ahead")
            except LookFailed as exc:
                return encode_chat_request(history, heard, tool_result={
                    "name": name, "result": {"error": str(exc)}}), False
        else:
            jpeg = camera_jpeg()
        if jpeg is None:
            return encode_chat_request(history, heard, tool_result={
                "name": name, "result": {"error": "the camera has no picture right now"}}), False
        # The room sees exactly the picture the model gets, under the tool
        # line.
        display.on_look(jpeg)
        # The same note whichever way the head is turned: "with your head
        # turned to your right" came back out loud as "Your head is turned to
        # your right, and I can see a doorway" (live). The side is
        # kept with the frame (`looked`) and said by _look_line later.
        note = LOOK_NOTE
        # The robot knows who it is looking at (demo/people.py); without this
        # the model describes "a man in a black t-shirt" to the person's face.
        # The names come off the picture being SENT, not off the frame this
        # turn started with: those are two different pictures the moment the
        # head turns, and the note used to name the person the robot had
        # turned away from (demo/run_demo.py's Looker.names).
        if look_fn is not None and look_names_fn is not None:
            names = look_names_fn()
        else:
            names = names_fn() if names_fn is not None else []
        if names:
            who_is_here = " and ".join(names)
            note = (f"{note[:-1]} {who_is_here} "
                    f"{'is' if len(names) == 1 else 'are'} in front of you.)")
        return encode_chat_request(history, heard, image_jpeg=jpeg,
                                   image_note=note), False
    if name not in MEMORY_TOOLS:
        return _no_tool_needed(history, heard, name), False
    # One memory tool. The model says which half of the past it means
    # (`about`); what in that half answers is decided by what the search
    # finds, never by the words of the question: words beat pictures, and a
    # picture comes back only when nothing was said about it — told about a
    # database, the robot used to answer with a photo of the room.
    query = str(arguments.get("query") or heard)
    direction = arguments.get("direction")
    # Which half of its past the question is about, as the MODEL sees it
    # (demo/chat_session.py's `about`): `seen` is answered from frames — by
    # their words, else the day's pictures — `said` and `taught` from words,
    # `anything` from the words of both, never the day. "What did you see
    # today?" and "what did we talk about today?" name nothing for either
    # store to match, and used to be answered by both at once. Measured on
    # 29 questions about what was seen that name nothing, the model said
    # `seen` for 26, `anything` for 2 ("what did you notice this morning?",
    # "what have you been watching?") and `taught` for 1; marked `anything`,
    # such a question finds no frame by its words and is answered from the
    # conversation — or, with nothing in it, by a bare sighting ("I saw Sasha
    # moments ago") or the nothing note. Told so and asked which half it
    # meant, the model never
    # called again (0 of 9) — so whatever it put there is what is answered:
    # the words of the question never change it.
    # The tool's older name `recall_seen` says it as plainly as `about` does.
    about = str(arguments.get("about") or (SEEN if name == "recall_seen" else ANYTHING))

    if about == ME:
        # "Do you remember me?", "have we met?", "what's my name?" — the
        # subject is the person in front of the camera, and nothing in a
        # vector store matches "me". Measured: left to the other cases, "did
        # you see me today?" came back "I do not have any memory of seeing you
        # today" while the person stood there and their name was in 34 frames.
        # This is also the one place a sighting of the person in front is the
        # answer rather than noise.
        me = (who_fn() or {} if who_fn is not None
              else {FACES_OFF: True, "note": FACES_UNAVAILABLE_NOTE})
        answer = {key: me[key] for key in
                  ("in_front_of_you", "you_met", "you_last_saw_them",
                   "people_you_have_met", "people_you_have_not_met_in_front_of_you")
                  if me.get(key)}
        if me.get(FACES_OFF):
            # Not a stranger: someone the robot cannot see the face of. Told
            # "you have not met them", it said so to a person it had met.
            answer["note"] = ("You cannot recognise faces right now, so you "
                              "cannot tell whether you have met them. Say so.")
        elif not answer.get("in_front_of_you"):
            answer["note"] = ("You have not met the person in front of you. "
                              "Say so, and ask their name.")
        display.on_speech_recall([{"text": person, "score": 1.0, "source": "faces"}
                                  for person in answer.get("in_front_of_you", [])])
        return encode_chat_request(history, heard,
                                   tool_result={"name": name, "result": answer}), True

    now = clock()
    # A side asked about needs no search at all.
    if direction in TURNED and recall_seen_fn is not None:
        turned = recall_seen_fn(query, direction=direction)
        if turned:
            # A side was asked about: that one picture is the answer.
            display.on_recall([{**turned[0], "weak": False}])
            return encode_chat_request(
                history, heard, image_jpeg=base64.b64decode(turned[0]["jpeg_b64"]),
                image_note=memory_note(turned[0], now)), True
    frames: list[dict] = []
    if about == SEEN and direction not in TURNED:
        # The frames whose words hold what was asked about (recall_seen). A
        # search that failed raises, and the model is told its memory could
        # not be searched: the day in its place would answer "did you see
        # Sasha?" with "I did not see Sasha" — the denial MemoryUnavailable
        # exists to prevent.
        if recall_seen_fn is not None:
            frames = recall_seen_fn(query)
        if not frames:
            # Seen, and no frame's words hold anything the question asked
            # about: it named nothing — "what did you see today?" — or a thing
            # no frame holds. Both are answered from the day itself, as
            # pictures: the newest frame and whatever looks least like it, so
            # the answer is several moments and not one picture three times,
            # and the model, looking at them, says whether the thing was
            # there. Nothing here tells the two kinds of question apart — that
            # took a list of words.
            day = ([frame for frame in day_frames_fn() if frame.get("jpeg_b64")]
                   if day_frames_fn is not None else [])
            if day:
                display.on_recall(_shown(day))
                return encode_chat_request(
                    history, heard,
                    image_jpeg=[base64.b64decode(frame["jpeg_b64"]) for frame in day]), True

    facts = knowledge_fn(query) if knowledge_fn is not None else []
    if about == TAUGHT and not facts:
        # The model says it was taught this, and nothing it was taught
        # answers: it is a question for the model itself. Live, "tell me about
        # black holes" came as `taught` and, given the latest exchanges
        # instead, the robot said it had "no information about black holes in
        # my memory".
        return encode_chat_request(history, heard, tool_result={
            "name": name, "result": {"found": [], "note": GENERAL_QUESTION_NOTE}}), False
    if about == SEEN or recall_fn is None:
        # A question about what was SEEN is answered from frames (A, B, C) —
        # never from the conversation, which is case D. Live:
        # "did you see people?" and "did you see any TV?" came back with
        # earlier exchanges — "Did you see the pollution? — I do not see any
        # pollution in front of me" — and the model copied their tense: "I do
        # not see any people in front of me right now."
        found = Recalled([], [])
    else:
        found = recall_fn(query)
    if (about not in (SEEN, SAID, TAUGHT) and recall_seen_fn is not None
            and direction not in TURNED):
        # `anything`: the frames by their words too — never the day, which is
        # pictures and cannot travel with the words of the other half. Live,
        # "tell me how does your memory work?" came as `anything`, the fact
        # answered at 0.74, and the latest looks came along under it (the old
        # fallback for a question that named nothing); a frame now comes only
        # when its words hold what was asked. `said` and `taught` (D) never
        # touch the frames.
        frames = recall_seen_fn(query)
    seen_people = (who_fn() or {}).get("seen_earlier", []) if who_fn is not None else []
    # Newest first, as LOOKS_NOTE tells the model they are.
    described = sorted((frame for frame in frames if frame.get("caption")),
                       key=lambda frame: frame.get("ts", 0.0), reverse=True)

    # ONE answer, assembled from everything that matched — not the first
    # branch that happened to be non-empty. Live: asked "what did
    # we talk about today?", the robot answered "We talked about Sasha and
    # what you were curious about", and asked "what did you see today?", "I
    # saw Sasha moments ago" — both times the only thing in the answer was a
    # face sighting from who(), which used to make the whole words branch fire
    # and hide the conversation and the looks behind it. A sighting is
    # context, never the answer on its own; the same turn's looks, facts and
    # exchanges belong in the same result.
    answer: dict = {}
    shown = list(facts) + list(found.speech_hits)
    if facts:
        answer["facts_you_were_taught"] = [fact["text"] for fact in facts]
    if found.memories:
        answer["the_person_told_you_before"] = found.memories
    if found.in_context:
        # Named for what it is, so the model says "you just said" instead of
        # "I recall": these lines are still in front of it.
        answer["said_moments_ago_in_this_conversation"] = found.in_context
    if described:
        answer["you_looked_at"] = [_look_line(frame, now)
                                   for frame in described[:LOOK_LINES]]
        answer["note"] = LOOKS_NOTE.format(n=len(answer["you_looked_at"]))
        display.on_recall([{**frame, "weak": False}
                           for frame in described[:LOOK_LINES]])
    # Nothing in words matched: read the latest of the conversation back.
    # "What did we talk about?" names no topic, so no exchange clears the gate
    # (recall's `recent`), and for a question that did name something the
    # latest exchanges beat "I have nothing about that". Not when anything
    # else answered — not for `taught` with a fact in hand above all: the
    # exchanges are the noise the model answered from instead of the fact,
    # live.
    if not answer:
        recent = [] if about == SEEN else found.recent[-RECENT_IN_ANSWER:]
        if recent:
            answer["you_talked_about"] = recent
            shown += [{"text": text, "score": 0.0, "source": "qdrant"}
                      for text in recent]
    if seen_people and not answer:
        # Last, and only when there is nothing else. A sighting is the thinnest
        # thing this memory holds — and the names are already IN the answers
        # that do exist: "On my left, a minute ago: a window (Sasha was
        # there)". As its own line beside them it added nothing and the model
        # led with it: "I saw Sasha moments ago" to "what did you see today?".
        # Alone, though, it is a real answer — "who did you see today?" on a
        # robot that has looked at nobody and been told nothing.
        answer["people_you_saw"] = seen_people
        shown += [{"text": person, "score": 1.0, "source": "faces"}
                  for person in seen_people]

    def in_words(result: dict, *, from_memory: bool = True) -> tuple[dict, bool]:
        display.on_speech_recall(shown)
        return encode_chat_request(history, heard,
                                   tool_result={"name": name, "result": result}), from_memory

    told = set(answer) - {"note"}
    if told - {"people_you_saw"}:
        # Words, and the looks the robot can put in words: these beat a
        # picture — told about a database, the robot used to answer with a
        # photo of the room. The picture still goes on the SCREEN when there
        # is a confident one and no look was read out: live, "did you see the
        # table?" was answered from the exchange of a look, rightly, and the
        # room saw no frame for it.
        if frames and not described:
            display.on_recall(_shown(frames))
        return in_words(answer)
    if frames:
        # Something was asked about that nothing was ever said about: the
        # frame whose words held it is the answer, as the picture. Ahead of a
        # bare sighting — "Sasha, moments ago" says less about what was asked
        # than the picture does, and memory_note names who was in it anyway.
        display.on_recall(_shown(frames))
        return encode_chat_request(
            history, heard, image_jpeg=base64.b64decode(frames[0]["jpeg_b64"]),
            image_note=memory_note(frames[0], now)), True
    if told:
        # Only sightings, and no picture to show: "who did you see today?"
        return in_words(answer)
    if about in (SEEN, SAID) or direction in TURNED:
        # A question about the past that nothing answers: "you never told
        # me" is the answer, and it is a recall — not stored back. Unless the
        # memory the question needed is not there at all (no search for it
        # was given): then nothing was found because nothing could be, and
        # "nothing in your memory" had the robot deny everything all run.
        # One that is there and simply holds nothing still says so.
        if about == SAID:
            needed_is_off = recall_fn is None
        else:
            needed_is_off = recall_seen_fn is None
        if needed_is_off:
            return in_words({"note": MEMORY_OFF_NOTE}, from_memory=False)
        return in_words({"note": NOTHING_IN_MEMORY_NOTE}, from_memory=False)
    # `anything`, and nothing anywhere: whether it was about the past at all
    # is for the model to say — it is the one that read the question, and a
    # list of words for "the past" read "tell me about black holes" as one.
    # So the note says both (GENERAL_TOO). And an answer the model gave on its
    # own is a real exchange: it goes into the window as one, and into Qdrant
    # when it leaves — live, "tell me about our planet" and "Earth, Mars,
    # Venus, Mercury" were lost to the `derived` flag, and "what did we talk
    # about?" found only the greeting.
    off = recall_fn is None or recall_seen_fn is None
    return in_words({"found": [], "note": (MEMORY_OFF_NOTE if off
                                           else NOTHING_IN_MEMORY_NOTE) + GENERAL_TOO},
                    from_memory=False)


# How much of the day one answer carries. Three of each: the model's context
# is 1000 tokens before the oldest turns start moving into Qdrant
# (DEFAULT_CONTEXT_BUDGET), and a tool result that reads back the whole
# afternoon evicts the conversation it was asked about.
LOOK_LINES = 3
RECENT_IN_ANSWER = 3
# What the model may say one `remember` call is about (demo/chat_session.py).
SEEN, SAID, ME, TAUGHT, ANYTHING = "seen", "said", "me", "taught", "anything"
# For a memory search that failed, not one that found nothing.
MEMORY_UNAVAILABLE_NOTE = ("Your memory could not be searched just now. Say "
                           "so plainly — never that you do not remember.")
# For a memory that is not there at all: it did not open at start, or the run
# has none (--no-memory). Nothing in it is not the same as nothing found.
MEMORY_OFF_NOTE = ("Your memory is off right now. Say so plainly — never that "
                   "you do not remember.")
# For a `taught` question no fact answers. Measured over 6 general questions:
# told only "Nothing in your memory about this." the model refused half of
# them — "I don't have specific information about black holes right now" —
# and told it is a general question, 6/6 answered. An `anything` question
# gets both: NOTHING_IN_MEMORY_NOTE + GENERAL_TOO.
GENERAL_QUESTION_NOTE = ("A general question, not a memory: answer it from your "
                         "own knowledge, in one or two sentences.")
# For a `seen` or `said` question memory has nothing on, for a side asked
# about with no look that way, and the first half of the note for an
# `anything` one.
NOTHING_IN_MEMORY_NOTE = "Nothing in your memory about this."
# What follows the note for an `anything` question nothing answers. Measured
# in fresh chats on questions the model marked `anything`: with
# NOTHING_IN_MEMORY_NOTE and this, it answered the 3 general questions of 3
# and said it did not recall 3 memory questions of 4 ("do you remember
# anything?" got "I remember a lot of things" under every note); with
# GENERAL_QUESTION_NOTE alone it made up a day for 2 of the 4 — "I've been
# busy helping people". After MEMORY_OFF_NOTE the model never said its memory
# was off, with this or without it (0 of 4 each); with this, it still
# answered the 3 general questions of 3.
GENERAL_TOO = (" If it is a general question, not one about the past, answer it "
               "from your own knowledge, in one or two sentences.")


# What comes with the looks whose words held what was asked (recall_seen),
# one short line each, newest first. Measured on captions the robot really
# said (two sentences, the second often about the picture's own left and
# right): as {where, when, what} objects, or as whole captions, the model
# named every look in 2 of 6
# answers and mixed the second sentence into the first look; one line with the
# first sentence only, "On my left, moments ago: a white light fixture…",
# named every look 6 of 6, each with its side.
# A day of frames carries NO note: the instruction lives in the side chat's
# own system message instead (demo/chat_session.py's DAY_SYSTEM). Measured —
# as a note, the robot answered "I saw a man looking intently in the first
# image"; as a system message, never once in three phrasings. Times, sides and
# names stay out of it either way: given each frame's time the model recited
# the metadata instead of looking, and given a name it bound it to the wrong
# picture.
LOOKS_NOTE = ("You looked {n} times. Here is each, newest first, with what you "
              "said about it then. Tell every one of the {n}, each with where "
              "it was, in the past tense: say \"I saw\". Answer everything "
              "else in this result too.")


def _look_line(frame: dict, now: float) -> str:
    """"On my left, 2 minutes ago: a lamp with a white shade." """
    caption = frame.get("caption", "").strip()
    end = caption.find(". ")
    first = caption if end < 0 else caption[:end + 1]
    for opening in ("I see ", "I can see "):
        if first.startswith(opening):
            first = first[len(opening):]
            break
    where = WHERE.get(frame.get("looked"), WHERE["ahead"])
    line = f"{where[0].upper()}{where[1:]}, {_ago(now - frame.get('ts', now))}: {first}"
    names = [p["name"] for p in frame.get("people") or [] if p.get("name")]
    if names:
        line += f" ({' and '.join(names)} {'was' if len(names) == 1 else 'were'} there.)"
    return line


# The line that ends a chain of tool calls. Added to whatever the last call
# came back with, so the model answers from it instead of reaching again.
NO_MORE_TOOLS = ("Answer the person now, in words, from what is here. Do not "
                 "call anything else.")


def _ask_for_words(payload: dict) -> None:
    """Add NO_MORE_TOOLS to a built request, whichever shape it has."""
    result = (payload.get("tool_result") or {}).get("result")
    if isinstance(result, dict):
        result["note"] = f"{result.get('note', '')} {NO_MORE_TOOLS}".strip()
    elif payload.get("image_note") is not None:
        payload["image_note"] = f"{payload['image_note']} {NO_MORE_TOOLS}".strip()


def _no_tool_needed(history, heard: str, called: str) -> dict:
    """Answer a tool call that nothing can serve — an invented tool, or the
    camera on a request to move.

    Measured live: answering "there is no tool called yes_dance" came back out
    loud as "I'm sorry, I don't have a specific dance move programmed" (2 of 3
    tries), while the motion pass was dancing. Answering that it is already
    handled — which is true, the body moves either way — gave a cheerful reply
    9 times out of 9."""
    return encode_chat_request(history, heard, tool_result={
        "name": called, "result": {
            "done": True,
            "note": "Your body performs movement on its own, no tool needed. "
                    "Just answer the person."}})
