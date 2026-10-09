"""demo/conversation.py — the robot's side of the chat: the window the model
holds in context, the move of its oldest part into memory, the tools, and
the request sequence of one turn."""
import base64

import pytest

from demo.contract import decode_chat_request
from demo.conversation import (DEFAULT_CONTEXT_BUDGET, LOOK_NOTE,
                               ConversationWindow, MemoryUnavailable, Recalled,
                               chat_turn, day_frames, lookup, memory_note,
                               recall, recall_seen)
from emulator.memory import EXCHANGE_KIND


class _Memory:
    """TextMemory stand-in: records what is stored, scores in-context texts
    from a table, returns scripted stored hits."""

    def __init__(self, scores=None, stored_hits=(), fail=False):
        self.remembered = []
        self._scores = scores or {}
        self._stored_hits = list(stored_hits)
        self._fail = fail

    def remember(self, text, kind, meta=None):
        if self._fail:
            raise OSError("embed service down")
        self.remembered.append((text, kind, meta))

    def recall_exchanges(self, query, k=3):
        return list(self._stored_hits)

    def score_texts(self, query, texts):
        return [self._scores.get(text, 0.0) for text in texts]


def _window(memory=None, **kwargs):
    ticks = iter(range(100, 10_000))
    return ConversationWindow(memory, clock=lambda: next(ticks), **kwargs)


def _filled(window, n):
    for i in range(n):
        window.add(f"q{i}", f"a{i}")
    return window


# — the window —

def test_history_is_the_exchanges_in_order():
    window = _filled(_window(), 2)
    assert window.history == [("q0", "a0"), ("q1", "a1")]


def test_under_budget_nothing_moves():
    memory = _Memory()
    window = _filled(_window(memory, budget_tokens=500), 4)
    assert window.after_turn(499) == []
    assert memory.remembered == []
    assert len(window.history) == 4


def test_over_budget_the_oldest_half_moves_into_memory_as_exchanges():
    memory = _Memory()
    window = _filled(_window(memory, budget_tokens=500), 6)
    stored = window.after_turn(501)
    assert stored == ["Person: q0 — Reachy: a0", "Person: q1 — Reachy: a1",
                      "Person: q2 — Reachy: a2"]
    assert [kind for _, kind, _ in memory.remembered] == [EXCHANGE_KIND] * 3
    assert memory.remembered[0][2] == {"said_at": 100}
    assert window.history == [("q3", "a3"), ("q4", "a4"), ("q5", "a5")]


def test_an_unknown_token_count_moves_nothing():
    # An image turn runs in a side chat and reports no count.
    window = _filled(_window(_Memory(), budget_tokens=10), 4)
    assert window.after_turn(None) == []
    assert len(window.history) == 4


def test_a_single_exchange_is_never_evicted():
    window = _filled(_window(_Memory(), budget_tokens=10), 1)
    assert window.after_turn(10_000) == []
    assert len(window.history) == 1


def test_the_cap_evicts_even_without_a_token_count():
    memory = _Memory()
    window = _filled(_window(memory, max_exchanges=3), 4)
    assert window.after_turn(None) == ["Person: q0 — Reachy: a0",
                                       "Person: q1 — Reachy: a1"]
    assert len(window.history) == 2


def test_a_failed_store_keeps_the_exchanges_in_context():
    window = _filled(_window(_Memory(fail=True), budget_tokens=500), 4)
    assert window.after_turn(900) == []
    assert len(window.history) == 4


def test_a_failed_store_over_the_cap_drops_rather_than_grows():
    window = _filled(_window(_Memory(fail=True), max_exchanges=3), 4)
    assert window.after_turn(None) == []  # nothing was stored
    assert len(window.history) == 2


def test_without_memory_eviction_still_bounds_the_context():
    window = _filled(_window(None, budget_tokens=500), 4)
    assert window.after_turn(900) == []
    assert len(window.history) == 2


def test_flush_moves_everything_still_in_context():
    memory = _Memory()
    window = _filled(_window(memory), 2)
    assert len(window.flush()) == 2
    assert window.history == []
    assert len(memory.remembered) == 2


def test_search_finds_in_context_exchanges_over_the_gate():
    said = "Person: Hi, I'm Sasha. — Reachy: Nice to meet you, Sasha!"
    window = _window(_Memory(scores={said: 0.7}))
    window.add("Hi, I'm Sasha.", "Nice to meet you, Sasha!")
    window.add("Can you nod?", "Sure!")
    assert window.search("what's my name?", 0.62) == [
        {"text": said, "score": 0.7, "source": "context"}]


# — the recall tool —

class _Frames:
    """FrameMemory stand-in: `words` answer the frame-words search (filtered
    by `before`, as the real recall_text is), `looks` are the frames taken on
    request (newest first, as latest_looks returns them), `day` the day's
    frames."""

    def __init__(self, looks=(), words=(), day=()):
        self.looks = list(looks)
        self.words = list(words)
        self.day = list(day)
        self.looks_asked = []
        self.word_queries = []
        self.word_befores = []
        self.day_befores = []

    def recall_text(self, query, k=3, before=None):
        self.word_queries.append(query)
        self.word_befores.append(before)
        return [frame for frame in self.words
                if before is None or frame.get("ts", 0.0) < before]

    def day_frames(self, before=None):
        self.day_befores.append(before)
        return [frame for frame in self.day
                if before is None or frame.get("ts", 0.0) < before]

    def latest_looks(self, directions=None, limit=4, before=None):
        self.looks_asked.append((directions, limit, before))
        return [look for look in self.looks
                if (directions is None or look.get("looked") in directions)
                and (before is None or look["ts"] < before)][:limit]


# — one turn —

class _Display:
    def __init__(self):
        self.events = []

    def on_tool_call(self, name, arguments):
        self.events.append(("tool", name, arguments))

    def on_memory_write(self, texts):
        self.events.append(("memory", texts))

    def on_recall(self, hits):
        self.events.append(("frames", hits))

    def on_look(self, jpeg):
        self.events.append(("look", jpeg))

    def on_speech_recall(self, hits):
        self.events.append(("speech", hits))

    def on_context(self, tokens, budget, exchanges):
        self.events.append(("context", tokens, budget, exchanges))

    def on_memory_count(self, frames, exchanges):
        self.events.append(("count", frames, exchanges))


class _Brain:
    """Answers each /chat request with the next scripted done event, and
    keeps every request it was sent, decoded."""

    def __init__(self, *dones):
        self._dones = list(dones)
        self.requests = []

    def __call__(self, payload):
        self.requests.append(decode_chat_request(payload))
        return self._dones.pop(0)


def _turn(heard, brain, *, window=None, recall_fn=None, recall_seen_fn=None,
          camera_jpeg=None,
          display=None, **kwargs):
    return chat_turn(heard, window=window if window is not None else _window(),
                     send=brain, recall_fn=recall_fn,
                     recall_seen_fn=recall_seen_fn, camera_jpeg=camera_jpeg,
                     display=display or _Display(), **kwargs)


def test_a_plain_reply_is_one_request_and_joins_the_window():
    window = _window()
    brain = _Brain({"reply": "Hello Sasha!", "token_count": 300})
    done = _turn("Hi, I'm Sasha.", brain, window=window)
    assert done["reply"] == "Hello Sasha!"
    assert [(r.history, r.text) for r in brain.requests] == [([], "Hi, I'm Sasha.")]
    assert window.history == [("Hi, I'm Sasha.", "Hello Sasha!")]


def test_the_window_rides_along_with_every_request():
    window = _filled(_window(), 1)
    brain = _Brain({"reply": "c", "token_count": 1})
    _turn("q", brain, window=window)
    assert brain.requests[0].history == [("q0", "a0")]


def test_look_answers_with_the_camera_picture():
    window, display = _window(), _Display()
    brain = _Brain({"tool_call": {"name": "camera", "arguments": {}}, "reply": ""},
                   {"reply": "I see a mug.", "token_count": None})
    _turn("What do you see?", brain, window=window, display=display,
          camera_jpeg=lambda: b"JPEG")
    second = brain.requests[1]
    assert (second.text, second.image_jpeg, second.image_note) == (
        "What do you see?", b"JPEG", LOOK_NOTE)
    assert ("tool", "camera", {}) in display.events
    assert window.history == [], "a look stays out of the model's conversation"


def test_look_without_a_picture_tells_the_model_so():
    brain = _Brain({"tool_call": {"name": "camera", "arguments": {}}},
                   {"reply": "My camera is dark.", "token_count": 1})
    _turn("What do you see?", brain, camera_jpeg=lambda: None)
    result = brain.requests[1].tool_result
    assert result["name"] == "camera" and "error" in result["result"]






def test_recall_with_no_query_searches_what_was_said():
    queries = []
    brain = _Brain({"tool_call": {"name": "recall", "arguments": {}}},
                   {"reply": "ok", "token_count": 1})
    _turn("What did I show you?", brain,
          recall_fn=lambda query: (queries.append(query), Recalled([], [], []))[1])
    assert queries == ["What did I show you?"]






def test_a_model_that_keeps_calling_tools_is_cut_off():
    window = _window()
    brain = _Brain(*[{"tool_call": {"name": "camera", "arguments": {}}}] * 4)
    _turn("look", brain, window=window, camera_jpeg=lambda: b"J",
          max_tool_rounds=2)
    # Three rounds of tools, then ONE request that asks for words — and no
    # more, whatever the model does with it.
    assert len(brain.requests) == 4
    assert window.history == []


def test_a_turn_out_of_tool_rounds_still_says_something():
    """It used to return the tool call itself, whose reply is empty: the robot
    stood there saying nothing for the whole turn. A bounded chain is right;
    silence in front of a room is not."""
    from demo.conversation import NO_MORE_TOOLS

    window = _window()
    brain = _Brain(*([{"tool_call": {"name": "remember", "arguments": {"query": "x"}}}] * 3
                     + [{"reply": "We talked about Qdrant.", "token_count": 1}]))
    _turn("What did we talk about?", brain, window=window, max_tool_rounds=2,
          recall_fn=lambda query: Recalled([], [], recent=["Person: hi — Reachy: hello"]),
          recall_seen_fn=lambda query, direction=None: [])
    assert NO_MORE_TOOLS in brain.requests[-1].tool_result["result"]["note"]
    assert window.history == [("What did we talk about?", "We talked about Qdrant.")]


def test_an_empty_reply_is_not_remembered():
    window = _window()
    _turn("hello?", _Brain({"reply": "", "token_count": 1}), window=window)
    assert window.history == []


def test_eviction_is_reported_to_the_dashboard():
    memory, display = _Memory(), _Display()
    window = _filled(_window(memory, budget_tokens=100), 2)
    _turn("q", _Brain({"reply": "r", "token_count": 200}), window=window,
          display=display)
    assert ("memory", ["Person: q0 — Reachy: a0"]) in display.events


def test_the_camera_tool_also_answers_to_its_old_name():
    brain = _Brain({"tool_call": {"name": "look", "arguments": {}}},
                   {"reply": "A wall.", "token_count": None})
    _turn("What do you see?", brain, camera_jpeg=lambda: b"JPEG")
    assert brain.requests[1].image_jpeg == b"JPEG"




















@pytest.mark.parametrize("heard", ["What do you see right now?", "Look at this.",
                                   "What am I holding?", "What's in front of you?"])
def test_a_camera_call_about_the_present_stays_a_camera_call(heard):
    brain = _Brain({"tool_call": {"name": "camera", "arguments": {}}},
                   {"reply": "ok", "token_count": 1})
    _turn(heard, brain, camera_jpeg=lambda: b"NOW",
          recall_fn=lambda query: pytest.fail("recall must not run"))
    assert brain.requests[1].image_jpeg == b"NOW"


def test_the_window_exposes_the_budget_the_dashboard_draws_against():
    assert _window(budget_tokens=321).budget == 321


def test_a_turn_reports_how_full_the_context_is():
    display = _Display()
    window = _window(_Memory(), budget_tokens=500)
    _turn("hi", _Brain({"reply": "hello", "token_count": 310}), window=window,
          display=display)
    assert ("context", 310, 500, 1) in display.events


def test_an_image_turn_reports_no_token_count():
    # It ran in a side chat, which has its own context — the bar must stay
    # where it was rather than draw a drop that did not happen.
    display = _Display()
    brain = _Brain({"tool_call": {"name": "camera", "arguments": {}}},
                   {"reply": "A wall.", "token_count": None})
    _turn("What do you see?", brain, display=display, camera_jpeg=lambda: b"J")
    assert ("context", None, DEFAULT_CONTEXT_BUDGET, 0) in display.events


def test_the_context_is_reported_after_the_eviction_it_caused():
    display, memory = _Display(), _Memory()
    window = _filled(_window(memory, budget_tokens=100), 2)
    _turn("q", _Brain({"reply": "r", "token_count": 200}), window=window,
          display=display)
    # The count is the context BEFORE the eviction, the window is after it.
    assert ("context", 200, 100, 2) in display.events


@pytest.mark.parametrize("heard", ["What do you see right now?", "Look at this.",
                                   "What am I holding?"])
def test_a_real_camera_question_still_gets_the_picture(heard):
    brain = _Brain({"tool_call": {"name": "camera", "arguments": {}}},
                   {"reply": "A mug.", "token_count": 1})
    _turn(heard, brain, camera_jpeg=lambda: b"NOW")
    assert brain.requests[1].image_jpeg == b"NOW"


def test_an_answer_read_out_of_memory_does_not_go_back_into_memory():
    # Measured live: storing them built a loop — "I recall we were discussing
    # something emotional" was stored, and the next "what did we discuss?"
    # recalled that sentence instead of the thing actually discussed.
    memory = _Memory()
    window = _window(memory, budget_tokens=100)
    window.add("Hi, I'm Sasha.", "Hello Sasha!")
    brain = _Brain({"tool_call": {"name": "recall", "arguments": {"query": "x"}}},
                   {"reply": "I recall we discussed something emotional.",
                    "token_count": 200})
    _turn("What did we discuss?", brain, window=window,
          recall_fn=lambda query: Recalled([], [], []))
    stored = [text for text, _kind, _meta in memory.remembered]
    assert stored == ["Person: Hi, I'm Sasha. — Reachy: Hello Sasha!"]
    assert "emotional" not in " ".join(stored)


def test_a_derived_exchange_leaves_the_window_without_being_stored():
    memory = _Memory()
    window = _window(memory, budget_tokens=10)
    window.add("What did we discuss?", "Something emotional.", derived=True)
    window.add("And then?", "More of it.", derived=True)
    assert window.after_turn(500) == []
    assert memory.remembered == []
    assert len(window.history) == 1


def test_a_plain_answer_still_goes_into_memory():
    memory = _Memory()
    window = _window(memory, budget_tokens=100)
    window.add("I'm giving a talk.", "Exciting!")
    window.add("Tell me a joke.", "Why did the robot cross the road?")
    assert window.after_turn(200) == ["Person: I'm giving a talk. — Reachy: Exciting!"]


# --- who the person is: a face, or the answer to the name question ---

def test_a_name_said_in_passing_names_nobody():
    # Nothing reads names out of the words: a pattern over them needed a list
    # of words that are not names ("I'm giving", "I'm not"), and a wrong name
    # outlives the mistake. The words keep the name; the label stays Person.
    memory = _Memory()
    window = _window(memory, budget_tokens=10)
    window.add("Hello there.", "Hi!")
    window.add("I'm Sasha, by the way.", "Nice to meet you, Sasha!")
    assert window.speaker is None
    window.flush()
    assert [text for text, _kind, _meta in memory.remembered] == [
        "Person: Hello there. — Reachy: Hi!",
        "Person: I'm Sasha, by the way. — Reachy: Nice to meet you, Sasha!"]


def test_a_face_names_what_nobody_was_named_for_yet():
    # The person was talking before the camera knew them: the same person.
    window = _window(_Memory())
    window.add("Hello there.", "Hi!")
    window.set_speaker("Sasha")
    window.add("Tell me a joke.", "Why did the robot cross the road?")
    assert [e.speaker for e in window._exchanges] == ["Sasha", "Sasha"]


def test_a_new_face_does_not_take_the_last_persons_words():
    # Alice talks, steps away, Bob steps in and is recognised: what Alice said
    # is still in the window, and it must reach memory as hers.
    memory = _Memory()
    window = _window(memory, budget_tokens=10)
    window.set_speaker("Alice")
    window.add("My dog is called Rex.", "Lovely name!")
    window.set_speaker("Bob")
    window.add("What's the weather?", "I can't see outside.")
    assert [e.speaker for e in window._exchanges] == ["Alice", "Bob"]
    window.flush()
    assert [text for text, _kind, _meta in memory.remembered] == [
        "Alice: My dog is called Rex. — Reachy: Lovely name!",
        "Bob: What's the weather? — Reachy: I can't see outside."]


def test_someone_new_names_only_their_own_words_when_they_say_who_they_are():
    # Alice was recognised; a stranger steps in, talks, and answers the name
    # question with "Bob". His words become Bob's; Alice's stay hers.
    window = _window(_Memory())
    window.set_speaker("Alice")
    window.add("My dog is called Rex.", "Lovely name!")
    window.someone_new()
    window.add("Hello there.", "Hi!")
    window.add("What can you do?", "I remember things.")
    window.introduce("Bob")
    assert [e.speaker for e in window._exchanges] == ["Alice", "Bob", "Bob"]


def test_a_known_face_does_not_take_a_strangers_words():
    # The stranger leaves unnamed; Alice comes back and is recognised. What
    # the stranger said is not hers.
    window = _window(_Memory())
    window.set_speaker("Alice")
    window.add("My dog is called Rex.", "Lovely name!")
    window.someone_new()
    window.add("I like trains.", "Me too!")
    window.set_speaker("Alice")
    window.add("Where were we?", "Your dog, Rex.")
    assert [e.text.split(":")[0] for e in window._exchanges] == ["Alice", "Person", "Alice"]


def test_a_stranger_who_said_their_name_keeps_it_while_they_stay():
    window = _window(_Memory())
    window.someone_new()
    window.add("Hi.", "Hello!")
    window.introduce("Bob")
    window.add("What can you do?", "I remember things.")
    assert [e.speaker for e in window._exchanges] == ["Bob", "Bob"]


def test_an_answered_name_question_names_only_that_strangers_words():
    window = _window(_Memory())
    window.someone_new()
    window.add("I like trains.", "Me too!")            # the first stranger
    window.someone_new()
    window.add("Hello there.", "Hi!")                 # another one
    window.introduce("Bob")
    assert [e.speaker for e in window._exchanges] == [None, "Bob"]

# --- two memories: `recall` answers with words, `recall_seen` with a frame ---

def _frame(score, ts=0.0):
    return {"jpeg_b64": base64.b64encode(b"FRAME").decode(), "ts": ts, "score": score}


def _call(name, query="q", **arguments):
    return {"tool_call": {"name": name,
                          "arguments": {"query": query, **arguments}}}


def test_recall_keeps_memory_and_the_live_context_apart():
    # What is still in the window was not remembered — the model is looking
    # at it. Only Qdrant hits count as memory, for the model and for the room.
    stored = {"text": "Person: I have a dog called Rex. — Reachy: Lovely!", "score": 0.65}
    said = "Person: Hi, I'm Sasha. — Reachy: Nice to meet you, Sasha!"
    memory = _Memory(scores={said: 0.7}, stored_hits=[stored])
    window = _window(memory)
    window.add("Hi, I'm Sasha.", "Nice to meet you, Sasha!")
    found = recall("my name", window=window, speech_memory=memory)
    assert found.memories == [stored["text"]]
    assert [hit["source"] for hit in found.speech_hits] == ["qdrant"]
    assert found.in_context == [said]


def test_a_memory_that_never_opened_is_not_an_empty_one():
    # The memory shard could not open at start: told "Nothing in your memory
    # about this" all run, the robot denied everything it was asked about.
    from demo.conversation import MEMORY_OFF_NOTE

    brain = _Brain(_call("remember", "my dog", about="said"),
                   {"reply": "My memory is off right now.", "token_count": 1})
    _turn("Do you remember my dog?", brain, recall_fn=None, recall_seen_fn=None)
    assert brain.requests[1].tool_result["result"] == {"note": MEMORY_OFF_NOTE}

    # A memory that is there and has nothing still says so.
    brain = _Brain(_call("remember", "my dog", about="said"),
                   {"reply": "You never told me.", "token_count": 1})
    _turn("Do you remember my dog?", brain,
          recall_fn=lambda query: Recalled([], []),
          recall_seen_fn=lambda query, direction=None: [])
    assert brain.requests[1].tool_result["result"] == {
        "note": "Nothing in your memory about this."}


def test_with_one_memory_off_the_question_is_answered_by_the_one_it_needs():
    # Frame memory can fail to open while the words open (or the other way
    # round): a question about the half that is there, and empty, still
    # gets "nothing"; a question about the half that is off gets "off".
    from demo.conversation import GENERAL_TOO, MEMORY_OFF_NOTE

    nothing = {"note": "Nothing in your memory about this."}
    no_words = lambda query: Recalled([], [])
    no_frames = lambda query, direction=None: []
    for about, question, words, frames, expected in (
            ("said", "Do you remember my dog?", no_words, None, nothing),
            ("seen", "Did you see my dog earlier?", None, no_frames, nothing),
            ("said", "Do you remember my dog?", None, no_frames, {"note": MEMORY_OFF_NOTE}),
            ("seen", "Did you see my dog earlier?", no_words, None, {"note": MEMORY_OFF_NOTE}),
            ("anything", "Do you remember my dog?", no_words, None,
             {"found": [], "note": MEMORY_OFF_NOTE + GENERAL_TOO}),
            ("anything", "Do you remember my dog?", None, no_frames,
             {"found": [], "note": MEMORY_OFF_NOTE + GENERAL_TOO})):
        brain = _Brain(_call("remember", "my dog", about=about),
                       {"reply": "...", "token_count": 1})
        _turn(question, brain, recall_fn=words, recall_seen_fn=frames,
              day_frames_fn=lambda: [])
        assert brain.requests[1].tool_result["result"] == expected, (about, words, frames)


def test_with_memory_off_a_general_question_is_still_a_general_one():
    # Off is said, and so is what to do with a question that is not about the
    # past: whether it was is the model's to tell, not a list of words.
    from demo.conversation import GENERAL_TOO, MEMORY_OFF_NOTE

    brain = _Brain(_call("remember", "airplanes", about="anything"),
                   {"reply": "Wings make lift.", "token_count": 1})
    _turn("How do airplanes fly?", brain, recall_fn=None, recall_seen_fn=None)
    assert brain.requests[1].tool_result["result"] == {
        "found": [], "note": MEMORY_OFF_NOTE + GENERAL_TOO}


def test_with_memory_off_the_facts_it_was_taught_still_answer():
    # The knowledge base is its own shard: it opens when the memory does not.
    fact = {"text": "Qdrant Edge runs inside the robot's own process.", "score": 0.8}
    brain = _Brain(_call("remember", "qdrant edge", about="taught"),
                   {"reply": "It runs in my own process.", "token_count": 1})
    _turn("Do you remember what Qdrant Edge is?", brain, recall_fn=None,
          recall_seen_fn=None, knowledge_fn=lambda query: [fact])
    result = brain.requests[1].tool_result["result"]
    assert result["facts_you_were_taught"] == [fact["text"]]
    assert "note" not in result


def test_a_memory_that_cannot_be_read_says_so_rather_than_finding_nothing():
    # "Nothing in your memory" for a store that could not be read had the
    # robot deny what it remembered.
    from demo.conversation import MemoryUnavailable

    class Down:
        def recall_exchanges(self, query, k=3):
            raise OSError("down")

        def score_texts(self, query, texts):
            raise OSError("down")

    down = Down()
    window = ConversationWindow(down)
    window.add("a", "b")
    with pytest.raises(MemoryUnavailable):
        recall("x", window=window, speech_memory=down)


def test_the_model_is_told_the_memory_could_not_be_searched():
    from demo.conversation import MEMORY_UNAVAILABLE_NOTE, MemoryUnavailable

    def down(query):
        raise MemoryUnavailable("exchanges: OSError: down")

    brain = _Brain(_call("remember", query="my name", about="said"),
                   {"reply": "My memory is not answering right now.", "token_count": 1})
    memory = _Memory()
    window = _window(memory)
    _turn("What's my name?", brain, window=window, recall_fn=down)
    assert brain.requests[1].tool_result["result"] == {"note": MEMORY_UNAVAILABLE_NOTE}
    # Nothing came out of memory, so nothing would be copied back into it:
    # what the person said is kept.
    assert not window._exchanges[-1].derived


def test_recall_seen_is_the_words_search_and_never_a_picture_guess():
    """A question that names nothing finds nothing: no nearest picture
    stands in for an answer (the picture search that did — "What did you
    see?" scored 0.107 on a frame of the presenter, over its gate — is gone)."""
    frames = _Frames()
    assert recall_seen("what did you see?", frame_memory=frames) == []
    lamp = _look("left", 30.0, "I see a lamp.")
    frames = _Frames(words=[lamp])
    assert recall_seen("was there a lamp?", frame_memory=frames) == [lamp]


def test_recall_seen_drops_frames_stored_during_this_turn():
    frames = _Frames(words=[_frame(0.8, ts=50.0), _frame(0.7, ts=10.0)])
    found = recall_seen("the mug", frame_memory=frames, turn_started_at=40.0)
    assert [frame["ts"] for frame in found] == [10.0]
    assert frames.word_befores == [40.0], "the turn's start goes to the search"


def test_the_day_leaves_out_the_frames_stored_during_this_turn():
    # The newest frame is always the first of the day: without the turn's
    # start it would be the present, on the screen, as a memory.
    frames = _Frames(day=[_frame(0.0, ts=50.0), _frame(0.0, ts=10.0)])
    assert [frame["ts"] for frame in day_frames(frame_memory=frames,
                                                turn_started_at=40.0)] == [10.0]
    assert frames.day_befores == [40.0]


def test_recall_seen_reports_a_failing_store():
    from demo.conversation import MemoryUnavailable

    class Down:
        def recall_text(self, query, before=None):
            raise OSError("down")

    with pytest.raises(MemoryUnavailable):
        recall_seen("x", frame_memory=Down())
    assert recall_seen("x", frame_memory=None) == []


def test_memory_note_says_it_is_a_memory_and_how_old():
    # "(What you saw N days ago.)" still got "I see a room…" in 2 answers of 4;
    # saying plainly that it is a memory got 0 of 4.
    note = memory_note({"ts": 1000.0}, 1300.0)
    assert "MEMORY" in note and "5 minutes ago" in note and "past tense" in note
    assert "2 days ago" in memory_note({"ts": 0.0}, 2 * 86400.0)


def test_the_projector_shows_the_frames_whose_words_answer_plainly():
    # Every frame that comes back cleared its gate by its words: none is a
    # guess, and the model gets the best of them.
    display = _Display()
    best, other = {**_frame(0.8), "ts": 900.0}, {**_frame(0.7), "ts": 800.0}
    brain = _Brain(_call("remember", "the mug", about="seen"),
                   {"reply": "ok", "token_count": 1})
    _turn("Did you see the mug?", brain, display=display, clock=lambda: 1000.0,
          recall_seen_fn=lambda query, direction=None: [best, other])
    assert ("frames", [{**best, "weak": False}, {**other, "weak": False}]) in display.events
    assert brain.requests[1].image_jpegs == (b"FRAME",)


def test_a_turn_that_used_either_memory_is_not_stored_back():
    for tool, kwargs in (("recall", {"recall_fn": lambda query: Recalled([], [])}),
                         ("recall_seen", {"recall_seen_fn": lambda query, direction=None: []})):
        memory = _Memory()
        window = _window(memory, budget_tokens=10)
        window.add("Hi.", "Hello!")
        _turn("What happened?", _Brain(_call(tool), {"reply": "Something.", "token_count": 500}),
              window=window, **kwargs)
        assert [text for text, _k, _m in memory.remembered] == ["Person: Hi. — Reachy: Hello!"]


# — the knowledge base —

def test_lookup_reports_a_failing_knowledge_base():
    from demo.conversation import MemoryUnavailable

    class _Broken:
        def search(self, query, k=3, min_score=None):
            raise OSError("embed service down")

    with pytest.raises(MemoryUnavailable):
        lookup("what is Qdrant Edge?", knowledge=_Broken())
    assert lookup("anything", knowledge=None) == []


# — looking around —

@pytest.mark.parametrize("direction", ["left", "right"])
def test_a_camera_call_with_a_direction_turns_the_head_and_says_so(direction):
    looks, display = [], _Display()
    brain = _Brain({"tool_call": {"name": "camera", "arguments": {"direction": direction}}},
                   {"reply": "A plant.", "token_count": None})
    _turn(f"Look to your {direction}. What do you see?", brain, display=display,
          camera_jpeg=lambda: pytest.fail("the picture from before the turn"),
          look_fn=lambda d: (looks.append(d), b"TURNED")[1])
    second = brain.requests[1]
    assert looks == [direction]
    assert (second.image_jpeg, second.image_note) == (
        b"TURNED", "(Your camera, right now.)")  # the side is kept with the frame, not said in the note
    assert ("tool", "camera", {"direction": direction}) in display.events


@pytest.mark.parametrize("arguments", [{}, {"direction": "ahead"}, {"direction": "up"}])
def test_a_camera_call_straight_ahead_looks_ahead(arguments):
    looks = []
    brain = _Brain({"tool_call": {"name": "camera", "arguments": arguments}},
                   {"reply": "A mug.", "token_count": None})
    _turn("What do you see?", brain, camera_jpeg=lambda: pytest.fail("the looker has it"),
          look_fn=lambda d: (looks.append(d), b"NOW")[1])
    assert looks == ["ahead"]
    assert (brain.requests[1].image_jpeg, brain.requests[1].image_note) == (b"NOW", LOOK_NOTE)


def test_a_head_that_could_not_turn_is_told_to_the_model():
    from demo.conversation import LookFailed

    def look(direction):
        raise LookFailed("your head could not turn to your left")

    display = _Display()
    brain = _Brain({"tool_call": {"name": "camera", "arguments": {"direction": "left"}}},
                   {"reply": "I can't turn my head right now.", "token_count": None})
    _turn("Look to your left.", brain, display=display, look_fn=look)
    second = brain.requests[1]
    assert second.image_jpeg is None
    assert second.tool_result["result"] == {"error": "your head could not turn to your left"}
    assert not [event for event in display.events if event[0] == "look"], \
        "no picture on the screen either"


def test_without_a_head_to_turn_the_camera_still_answers():
    brain = _Brain({"tool_call": {"name": "camera", "arguments": {"direction": "left"}}},
                   {"reply": "A mug.", "token_count": None})
    _turn("Look to your left.", brain, camera_jpeg=lambda: b"NOW")
    assert brain.requests[1].image_jpeg == b"NOW"


def _look(where, ts, caption=None):
    return {**_frame(0.0, ts=ts), "looked": where, **({"caption": caption} if caption else {})}


def test_recall_seen_by_direction_is_the_last_look_that_way_before_this_turn():
    frames = _Frames(looks=[_look("left", 50.0), _look("right", 40.0),
                            _look("left", 30.0)])
    found = recall_seen("what was on your left?", frame_memory=frames,
                        turn_started_at=45.0, direction="left")
    assert [frame["ts"] for frame in found] == [30.0]
    assert frames.looks_asked == [(["left"], 1, 45.0)]
    assert frames.word_queries == [], "no search: the side is the answer"


def test_recall_seen_in_general_is_the_frames_whose_words_answer():
    # "What did you see?" names nothing a frame's words hold: nothing comes
    # back — not the latest looks, not the nearest picture — and that is what
    # sends a `seen` question to the day's frames (_answer_tool).
    looks = [_look("right", 40.0, "I see a door."), _look("left", 30.0, "I see a lamp.")]
    frames = _Frames(looks=looks)
    assert recall_seen("what did you see?", frame_memory=frames) == []
    assert frames.looks_asked == []
    frames = _Frames(looks=looks, words=[looks[1]])
    assert recall_seen("was there a lamp?", frame_memory=frames) == [looks[1]]


def test_a_question_about_a_thing_searches_the_frames_own_words_first():
    """Measured on the robot's 371 stored frames: "did you see a bottle?",
    "what was on the table?", "a plant in a pot" — the picture search found 8
    right frames of 36, searching the words the frame already carries found
    32. The words go first; the day's frames answer when they find
    nothing."""
    bottle = _frame(0.8, ts=900.0)
    frames = _Frames(looks=[_look("left", 30.0, "I see a lamp.")],
                     words=[bottle])
    assert recall_seen("did you see a bottle?", frame_memory=frames) == [bottle]
    assert frames.word_queries == ["did you see a bottle?"]
    # …and nothing else was asked: not the looks.
    assert frames.looks_asked == []


def test_a_look_line_keeps_the_first_sentence_without_i_see():
    from demo.conversation import _look_line

    frame = _look("ahead", 940.0, "I see a man with a mug. He is on the right side of the image.")
    assert _look_line(frame, 1000.0) == "In front of me, a minute ago: a man with a mug."
    assert _look_line({**frame, "caption": "A door"}, 1000.0) == "In front of me, a minute ago: A door"


def test_a_look_stays_out_of_the_conversation_the_model_sees():
    window = _window()
    window.add("Hi.", "Hello!")
    brain = _Brain({"tool_call": {"name": "camera", "arguments": {"direction": "left"}}},
                   {"reply": "I see a lamp.", "token_count": None})
    _turn("Look to your left.", brain, window=window, look_fn=lambda d: b"J")
    assert window.history == [("Hi.", "Hello!")]


def test_the_last_look_one_way_is_shown_plainly():
    display = _Display()
    left = _look("left", 50.0)
    brain = _Brain({"tool_call": {"name": "recall_seen",
                                  "arguments": {"query": "left", "direction": "left"}}},
                   {"reply": "I saw a lamp.", "token_count": None})
    _turn("What was on your left?", brain, display=display,
          recall_seen_fn=lambda query, direction=None: [left])
    shown = [event for event in display.events if event[0] == "frames"][-1][1]
    assert [frame["weak"] for frame in shown] == [False]


# — who —

class _People:
    enabled = True

    def __init__(self, here=(), met=()):
        self._here = list(here)
        self._met = list(met)

    def faces_in(self, frame):
        return list(self._here)

    def met(self):
        return list(self._met)


class _SeenPeople:
    def __init__(self, seen):
        self.seen = seen
        self.asked = []

    def people_seen(self, before=None):
        self.asked.append(before)
        return list(self.seen)


def test_who_names_who_is_here_who_was_seen_and_who_was_met():
    from demo.conversation import who

    people = _People(here=[{"name": "Sasha"}, {"name": None}], met=["Sasha", "Robin"])
    frames = _SeenPeople([("Robin", 880.0)])
    result = who(people=people, frame=object(), frame_memory=frames,
                 turn_started_at=990.0, clock=lambda: 1000.0)
    assert result == {"in_front_of_you": ["Sasha"],
                      "people_you_have_not_met_in_front_of_you": 1,
                      "seen_earlier": ["Robin, 2 minutes ago"],
                      "you_last_saw_them": [],
                      "people_you_have_met": ["Sasha", "Robin"]}
    assert frames.asked == [990.0]


def test_who_with_nobody_there_or_no_face_models_says_so():
    from demo.conversation import who

    assert who(people=_People(), frame=object())["note"] == \
        "Nobody is in front of your camera right now."
    assert who() == {"faces_off": True, "note": "You cannot recognise faces right now."}


def test_who_without_a_picture_cannot_tell_who_is_there():
    # The camera gave no frame this turn: that is not an empty room.
    from demo.conversation import who

    result = who(people=_People(here=[{"name": "Sasha"}], met=["Sasha"]), frame=None)
    assert result["faces_off"] is True
    assert result["note"] == "You cannot recognise faces right now."


def test_a_look_line_says_who_was_there():
    from demo.conversation import _look_line

    frame = {**_look("ahead", 940.0, "I see a man with a mug."),
             "people": [{"name": "Sasha"}, {"name": None}]}
    assert _look_line(frame, 1000.0) == "In front of me, a minute ago: a man with a mug. (Sasha was there.)"


# — what a live run broke —

def test_a_look_is_neither_in_the_chat_nor_in_the_conversation_memory():
    """A look is stored once, as the frame's caption (demo/run_demo.py's
    Looker.caption). Written into the conversation as an exchange too, it
    had "what did we talk about?" answered "we talked about what I saw in the
    room" (live)."""
    memory = _Memory()
    window = _window(memory)
    display = _Display()
    brain = _Brain({"tool_call": {"name": "camera", "arguments": {"direction": "left"}}},
                   {"reply": "I see a lamp.", "token_count": None})
    _turn("Look to your left.", brain, window=window, display=display, look_fn=lambda d: b"J")
    assert window.history == []
    assert memory.remembered == []
    assert not [e for e in display.events if e[0] == "memory"]


# — the body moves because the model asked for it —

def test_the_move_tool_moves_the_body():
    moves = []
    brain = _Brain({"tool_call": {"name": "move", "arguments": {"how": "happy"}}},
                   {"reply": "Yes! I'm happy.", "token_count": 1})
    _turn("Show me your emotions.", brain, move_fn=moves.append,
          camera_jpeg=lambda: pytest.fail("a request to move is not a picture"))
    second = brain.requests[1]
    assert moves == ["happy"]
    assert second.tool_result == {"name": "move", "result": {"moved": "happy"}}


def test_a_move_without_a_body_still_answers():
    brain = _Brain({"tool_call": {"name": "move", "arguments": {"how": "dance"}}},
                   {"reply": "Dancing!", "token_count": 1})
    _turn("Dance for me!", brain)
    assert brain.requests[1].tool_result["result"] == {"moved": "dance"}


def test_the_move_tool_is_offered_with_the_moves_the_body_has():
    from demo.chat_session import CHAT_TOOLS, MOVES
    from demo.robot_reachy import EMOTION_MOVES, GESTURE_POSES

    move = next(t for t in CHAT_TOOLS if t["function"]["name"] == "move")
    assert move["function"]["parameters"]["properties"]["how"]["enum"] == MOVES
    for how in MOVES:
        assert how in GESTURE_POSES or how in EMOTION_MOVES or how == "dance"


# — one memory tool: the search decides which store answers —

def _remember(brain, **kwargs):
    return _turn("What do you remember?", brain, **kwargs)


def test_words_beat_pictures():
    # Told about a database, the robot used to answer with a photo of the room.
    told = "Sasha: Qdrant is a database. — Reachy: Got it!"
    found = Recalled(memories=[told],
                     speech_hits=[{"text": told, "score": 0.7, "source": "qdrant"}])
    display = _Display()
    brain = _Brain(_call("remember", "Qdrant"),
                   {"reply": "You said Qdrant is a database.", "token_count": 1})
    _turn("What did I tell you about Qdrant?", brain, display=display, clock=lambda: 1000.0,
          recall_fn=lambda query: found,
          knowledge_fn=lambda query: [{"text": "Qdrant is a vector database.",
                                       "score": 0.8, "source": "knowledge"}],
          recall_seen_fn=lambda query, direction=None: [_look("ahead", 10.0, "I see a room.")])
    result = brain.requests[1].tool_result["result"]
    assert brain.requests[1].image_jpeg is None
    assert result["facts_you_were_taught"] == ["Qdrant is a vector database."]
    assert result["the_person_told_you_before"] == [told]
    # The look comes along IN WORDS, in the same answer — one search, every
    # source that had something. What it must not do is arrive as a picture
    # instead of the sentence the person actually said. A fixed clock, not the
    # real one: the frame's ts=10.0 read against wall time made this "N days
    # ago" tick over and fail a day after it was written.
    assert result["you_looked_at"] == ["In front of me, 16 minutes ago: a room."]


def test_a_turned_look_is_labelled_with_who_is_in_THAT_picture():
    """Live: asked what was on its left, the robot described the window and
    then said "Sasha is in front of me now" — the note came from the frame the
    turn started with, taken while it was still facing Sasha."""
    brain = _Brain({"tool_call": {"name": "camera", "arguments": {"direction": "left"}}},
                   {"reply": "I saw a window.", "token_count": 1})
    _turn("What is on your left?", brain,
          look_fn=lambda direction: b"LEFT",
          look_names_fn=lambda: [],           # nobody is in the turned picture
          names_fn=lambda: ["Sasha"])         # …but Sasha is in front of the robot
    note = brain.requests[1].image_note
    assert "Sasha" not in note
    assert note == "(Your camera, right now.)"  # the side stays with the frame


def test_a_look_that_does_have_someone_in_it_still_names_them():
    brain = _Brain({"tool_call": {"name": "camera", "arguments": {"direction": "right"}}},
                   {"reply": "I saw Masha.", "token_count": 1})
    _turn("Look to your right.", brain,
          look_fn=lambda direction: b"RIGHT",
          look_names_fn=lambda: ["Masha"], names_fn=lambda: ["Sasha"])
    assert "Masha is in front of you" in brain.requests[1].image_note


def test_a_question_about_a_side_is_answered_with_that_picture():
    left = _look("left", 50.0)
    display = _Display()
    brain = _Brain({"tool_call": {"name": "remember",
                                  "arguments": {"query": "left", "direction": "left"}}},
                   {"reply": "I saw a lamp.", "token_count": None})
    asked = []
    _turn("What was on your left?", brain, display=display,
          recall_seen_fn=lambda query, direction=None: (asked.append(direction), [left])[1])
    assert asked == ["left"]
    assert brain.requests[1].image_jpeg == b"FRAME"
    assert brain.requests[1].image_note.startswith("(This is your MEMORY of what you saw to your left")


def test_with_nothing_said_about_it_the_looks_answer():
    # Two looks whose words hold what was asked, found best first: read out
    # newest first, as LOOKS_NOTE tells the model they are.
    door, lamp = _look("right", 940.0, "I see a door."), _look("left", 900.0, "I see a lamp.")
    display = _Display()
    brain = _Brain(_call("remember", "the door and the lamp"),
                   {"reply": "I saw a door and a lamp.", "token_count": 1})
    _turn("Did you see the door and the lamp?", brain, display=display, clock=lambda: 1000.0,
          recall_fn=lambda query: Recalled([], []),
          recall_seen_fn=lambda query, direction=None: [lamp, door])
    result = brain.requests[1].tool_result["result"]
    assert result["you_looked_at"] == ["On my right, a minute ago: a door.",
                                       "On my left, 2 minutes ago: a lamp."]


def test_a_recalled_picture_says_who_was_in_it():
    """demo/chat_session.py's IMAGE_SYSTEM promises "when the note names the
    people in it, call them by their name instead of describing them" — and
    this note named nobody, so a picture of someone the robot had met came
    back as "a man with blonde hair"."""
    from demo.conversation import memory_note

    frame = {"ts": 900.0, "looked": "left", "names": ["Sasha"]}
    note = memory_note(frame, 1000.0)
    assert "Sasha was in it." in note
    assert "to your left" in note and "2 minutes ago" in note
    # Two people, and the ones the robot could not name are left out.
    frame = {"ts": 900.0, "people": [{"name": "Sasha"}, {"name": "Masha"},
                                     {"name": None}]}
    assert "Sasha and Masha were in it." in memory_note(frame, 1000.0)
    assert "in it" not in memory_note({"ts": 900.0}, 1000.0)


def test_a_look_that_answers_keeps_the_latest_exchanges_out():
    """The latest exchanges are read back only when nothing else answered
    (_answer_tool). A look that answers is the answer; the exchanges beside
    it were noise the model answered from instead."""
    caption = "I see a window with dark brown curtains on the left side."
    looks = [_look("left", 940.0, caption)]
    found = Recalled([], [], recent=[
        "Sasha: Tell me a joke. — Reachy: Why did the robot go on vacation?"])
    # `about` left out, so `anything`: a `seen` answer leaves the exchanges
    # out anyway.
    brain = _Brain(_call("remember", "the window"),
                   {"reply": "I saw a window.", "token_count": 1})
    _turn("Do you remember the window?", brain, clock=lambda: 1000.0,
          recall_fn=lambda query: found,
          recall_seen_fn=lambda query, direction=None: looks)
    result = brain.requests[1].tool_result["result"]
    assert result["you_looked_at"] == [
        "On my left, a minute ago: a window with dark brown curtains on the left side."]
    assert "you_talked_about" not in result



def test_a_question_naming_nothing_that_memory_answers_keeps_the_latest_exchanges_out():
    # "What did we talk about?" names nothing; when an exchange still answers
    # it, that is the answer — the latest exchanges no longer come along just
    # because the question named nothing.
    told = "Sasha: my sister is called Anna — Reachy: lovely"
    found = Recalled(memories=[told],
                     speech_hits=[{"text": told, "score": 0.7, "source": "qdrant"}],
                     recent=["Sasha: hi — Reachy: hello"])
    brain = _Brain(_call("remember", "what did we talk about?", about="said"),
                   {"reply": "Your sister Anna.", "token_count": 1})
    _turn("What did we talk about?", brain, recall_fn=lambda query: found,
          recall_seen_fn=lambda query, direction=None: pytest.fail("no frames"))
    assert brain.requests[1].tool_result["result"] == {"the_person_told_you_before": [told]}


def test_an_undescribed_frame_comes_back_as_a_picture():
    frame = _frame(0.4, ts=900.0)
    brain = _Brain(_call("remember", "the mug"), {"reply": "I saw a mug.", "token_count": None})
    _turn("Do you remember the mug?", brain, clock=lambda: 1000.0,
          recall_fn=lambda query: Recalled([], []),
          recall_seen_fn=lambda query, direction=None: [frame])
    assert brain.requests[1].image_jpeg == b"FRAME"


def test_the_person_in_front_of_the_robot_is_not_a_memory_of_them():
    """Live: "I saw Sasha moments ago" — about the person it was talking to,
    from a frame stored seconds earlier."""
    from demo.conversation import who

    class _People:
        enabled = True

        def faces_in(self, frame):
            return [{"name": "Sasha", "box": [0.1, 0.1, 0.3, 0.3]}]

        def met(self):
            return ["Sasha", "Masha"]

    class _Frames:
        def people_seen(self, before=None):
            return [("Sasha", 990.0), ("Masha", 400.0)]

    result = who(people=_People(), frame=object(), frame_memory=_Frames(),
                 clock=lambda: 1000.0)
    assert result["in_front_of_you"] == ["Sasha"]
    assert result["seen_earlier"] == ["Masha, 10 minutes ago"]


def test_the_people_it_saw_answer_when_nothing_else_does():
    brain = _Brain(_call("remember", "who"), {"reply": "I saw Sasha.", "token_count": 1})
    _turn("Who did you see today?", brain,
          recall_fn=lambda query: Recalled([], []),
          who_fn=lambda: {"seen_earlier": ["Sasha, 2 minutes ago"]})
    assert brain.requests[1].tool_result["result"] == {
        "people_you_saw": ["Sasha, 2 minutes ago"]}


def test_nothing_at_all_brings_back_the_latest_conversation():
    found = Recalled([], [], recent=["Person: hi — Reachy: hello"])
    brain = _Brain(_call("remember", "anything"), {"reply": "We said hello.", "token_count": 1})
    _turn("What did we talk about?", brain, recall_fn=lambda query: found,
          recall_seen_fn=lambda query, direction=None: [])
    result = brain.requests[1].tool_result["result"]
    assert result["you_talked_about"] == found.recent


# --- what the robot answered live, and what it must answer
# instead. A face sighting from who() used to be enough to fire the words
# branch on its own, and everything behind it — the conversation, the looks —
# never reached the model: "what did we talk about today?" came back "We
# talked about Sasha and what you were curious about", "what did you see
# today?" came back "I saw Sasha moments ago". A sighting is the answer only
# when nothing else is — which, for a question about what was seen that the
# model marks `anything`, it can still be (see the test that records it).


def test_a_sighting_does_not_hide_the_conversation_it_was_asked_about():
    found = Recalled([], [], recent=["Sasha: what is Qdrant Edge? — Reachy: a vector "
                                     "database that runs on the robot"])
    brain = _Brain(_call("remember", "what did we talk about today?"),
                   {"reply": "We talked about Qdrant Edge.", "token_count": 1})
    _turn("What did we talk about today?", brain,
          recall_fn=lambda query: found,
          recall_seen_fn=lambda query, direction=None: [],
          who_fn=lambda: {"seen_earlier": ["Sasha, moments ago"]})
    result = brain.requests[1].tool_result["result"]
    assert result["you_talked_about"] == found.recent
    # And the sighting does not ride along beside it: the question was about
    # the conversation, and "Sasha, moments ago" is not part of the answer.
    assert "people_you_saw" not in result


def test_a_sighting_does_not_hide_what_the_robot_looked_at(monkeypatch):
    looks = [_look("left", 940.0, "I see a window with dark brown curtains.")]
    brain = _Brain(_call("remember", "the curtains"),
                   {"reply": "I saw curtains on my left.", "token_count": 1})
    _turn("Did you see the curtains?", brain, clock=lambda: 1000.0,
          recall_fn=lambda query: Recalled([], []),
          recall_seen_fn=lambda query, direction=None: looks,
          who_fn=lambda: {"seen_earlier": ["Sasha, moments ago"]})
    result = brain.requests[1].tool_result["result"]
    assert result["you_looked_at"] == [
        "On my left, a minute ago: a window with dark brown curtains."]
    # The look line carries whoever was in the frame (see _look_line); a flat
    # list of names beside it added nothing, and the model answered WITH it.
    assert "people_you_saw" not in result


def test_a_question_with_no_subject_comes_back_as_the_days_pictures():
    """"What did you see today?" names nothing any frame's words hold, and
    the answer is the frames themselves — a frame IS the memory, the labels
    only pick which one."""
    day = [_frame(0.0, ts=940.0), _frame(0.0, ts=700.0)]
    display = _Display()
    brain = _Brain(_call("remember", "what did you see today?", about="seen"),
                   {"reply": "A window, and a desk.", "token_count": 1})
    _turn("What did you see today?", brain, display=display, clock=lambda: 1000.0,
          recall_fn=lambda query: Recalled([], [], recent=["Sasha: hi — Reachy: hello"]),
          recall_seen_fn=lambda query, direction=None: [],
          day_frames_fn=lambda: day)
    request = brain.requests[1]
    assert request.image_jpegs == (b"FRAME", b"FRAME"), "both frames, one turn"
    assert request.tool_result is None, "pictures and a tool result cannot travel together"
    # No note: what to say about a day of frames is demo/chat_session.py's
    # DAY_SYSTEM, measured to work there and not in a note.
    assert not request.image_note
    # …and the room sees the same frames the model was given.
    assert [len(event[1]) for event in display.events if event[0] == "frames"] == [2]


def test_the_days_own_system_message_carries_no_times_sides_or_names():
    """Measured: given each frame's time and side the model stopped looking
    and recited the metadata — "something was in front of me 40 minutes ago"
    — and given a name it bound it to the wrong picture every way it was
    worded (names_note.py)."""
    from demo.chat_session import DAY_SYSTEM

    for word in ("minute", "ago", "left", "right", "Sasha"):
        assert word not in DAY_SYSTEM, word
    assert "I saw" in DAY_SYSTEM and "past tense" in DAY_SYSTEM


def test_a_day_with_no_frames_falls_through_to_the_words():
    day_asked = []
    brain = _Brain(_call("remember", "what did you see today?", about="seen"),
                   {"reply": "Nothing yet.", "token_count": 1})
    _turn("What did you see today?", brain, clock=lambda: 1000.0,
          recall_fn=lambda query: Recalled([], [], recent=["Sasha: hi — Reachy: hello"]),
          recall_seen_fn=lambda query, direction=None: [],
          day_frames_fn=lambda: (day_asked.append(1), [])[1])
    assert day_asked == [1]
    assert brain.requests[1].image_jpegs == ()
    assert brain.requests[1].tool_result is not None


def test_a_seen_question_a_frame_s_words_answer_gets_that_frame_not_the_day():
    """The frame whose words hold what was asked is the answer — with its
    note, which names who was in it; a day of frames carries no names, and
    measured, "did you see Sasha?" against the day came back "I did not see
    Sasha"."""
    sasha = {**_frame(0.8, ts=900.0), "names": ["Sasha"]}
    brain = _Brain(_call("remember", "Did you see Sasha?", about="seen"),
                   {"reply": "Yes, I saw Sasha.", "token_count": 1})
    _turn("Did you see Sasha?", brain, clock=lambda: 1000.0,
          recall_seen_fn=lambda query, direction=None: [sasha],
          day_frames_fn=lambda: pytest.fail("a frame answered: not the day"))
    request = brain.requests[1]
    assert request.image_jpegs == (b"FRAME",)
    assert "Sasha was in it" in request.image_note


def test_a_side_with_no_look_that_way_is_not_answered_with_the_day():
    # "What was on your left?" with no look to the left: there is nothing to
    # show for that side, and the day is not what was asked.
    brain = _Brain(_call("remember", "what was on your left?", about="seen",
                         direction="left"),
                   {"reply": "I did not look there.", "token_count": 1})
    _turn("What was on your left?", brain, clock=lambda: 1000.0,
          recall_fn=lambda query: Recalled([], []),
          recall_seen_fn=lambda query, direction=None: [],
          day_frames_fn=lambda: pytest.fail("not the day"))
    assert brain.requests[1].tool_result["result"] == {
        "note": "Nothing in your memory about this."}


def test_a_question_about_the_person_asking_is_answered_from_the_faces():
    """"Do you remember me?" and "did you see me today?" carry no subject a
    search can use. Left to the other cases the robot described its afternoon,
    and "did you see me today?" came back "I do not have any memory of seeing
    you today" — with the person in front of it."""
    brain = _Brain(_call("remember", "do you remember me?", about="me"),
                   {"reply": "Of course, Sasha.", "token_count": 1})
    _turn("Do you remember me?", brain, clock=lambda: 1000.0,
          recall_fn=lambda query: Recalled([], []),
          recall_seen_fn=lambda query, direction=None: pytest.fail("nor the frames' words"),
          day_frames_fn=lambda: pytest.fail("nor the day"),
          who_fn=lambda: {"in_front_of_you": ["Sasha"],
                          "you_met": ["Sasha, 2 hours ago"],
                          "you_last_saw_them": ["Sasha, 5 minutes ago"],
                          "seen_earlier": ["Masha, an hour ago"],
                          "people_you_have_met": ["Sasha", "Masha"]})
    result = brain.requests[1].tool_result["result"]
    assert result["in_front_of_you"] == ["Sasha"]
    assert result["you_met"] == ["Sasha, 2 hours ago"]
    # The one place a sighting of the person in front is the answer, not noise.
    assert result["you_last_saw_them"] == ["Sasha, 5 minutes ago"]
    assert "seen_earlier" not in result


def test_a_stranger_asking_if_they_are_remembered_is_told_the_truth():
    brain = _Brain(_call("remember", "have we met?", about="me"),
                   {"reply": "I don't think we have.", "token_count": 1})
    _turn("Have we met?", brain, who_fn=lambda: {"in_front_of_you": [],
                                                 "people_you_have_met": ["Masha"]})
    result = brain.requests[1].tool_result["result"]
    assert "not met" in result["note"]
    assert result["people_you_have_met"] == ["Masha"]


def test_a_robot_that_cannot_recognise_faces_does_not_call_anyone_a_stranger():
    # Faces off, or the face read failed: "Do you remember me?" came back "I
    # don't think we've met" — to someone the robot had met.
    for who_fn in (lambda: {"faces_off": True,
                            "note": "You cannot recognise faces right now."}, None):
        brain = _Brain(_call("remember", "do you remember me?", about="me"),
                       {"reply": "I can't tell right now.", "token_count": 1})
        _turn("Do you remember me?", brain, who_fn=who_fn)
        result = brain.requests[1].tool_result["result"]
        assert "not met" not in result["note"]
        assert "cannot recognise faces" in result["note"]
        assert "faces_off" not in result, "the flag is for the code, not the model"


def test_who_says_so_when_the_faces_cannot_be_read():
    from demo.conversation import who

    class _Unreadable(_People):
        def faces_in(self, frame):
            raise OSError("embed service down")

    result = who(people=_Unreadable(met=["Sasha"]), frame=object())
    assert result["faces_off"] is True
    assert result["note"] == "You cannot recognise faces right now."
    assert result["people_you_have_met"] == ["Sasha"]

    # Through the `me` answer, as the voice loop wires it: the model hears it
    # cannot tell, and still who it has met.
    brain = _Brain(_call("remember", "do you remember me?", about="me"),
                   {"reply": "I can't tell right now.", "token_count": 1})
    _turn("Do you remember me?", brain,
          who_fn=lambda: who(people=_Unreadable(met=["Sasha"]), frame=object()))
    answer = brain.requests[1].tool_result["result"]
    assert "cannot recognise faces" in answer["note"]
    assert answer["people_you_have_met"] == ["Sasha"]


def test_a_question_that_names_something_is_still_a_search():
    """The day is read back only for a question about what was seen. "What
    did I tell you about Qdrant?" must not come back with this afternoon's
    frames."""
    told = "Sasha: Qdrant Edge runs on the robot — Reachy: got it"
    found = Recalled(memories=[told],
                     speech_hits=[{"text": told, "score": 0.7, "source": "qdrant"}])
    brain = _Brain(_call("remember", "Qdrant"), {"reply": "You told me.", "token_count": 1})
    _turn("What did I tell you about Qdrant?", brain,
          recall_fn=lambda query: found,
          recall_seen_fn=lambda query, direction=None: [],
          day_frames_fn=lambda: pytest.fail("not `seen`: not the day"))
    assert brain.requests[1].tool_result["result"] == {
        "the_person_told_you_before": [told]}


def test_a_non_visual_question_with_no_subject_never_grabs_a_random_frame():
    """Live: "how do you work?" names nothing, the same as "what did you
    see?" — and recall_seen's old fallback for that answered it anyway, with
    whatever frame happened to be nearest, and the model summarised an
    unrelated caption. Now a frame comes back only when its words hold what
    was asked, which no frame's words do here (pinned on real bge in
    test_frame_memory.py), and there is no picture guess."""
    from demo.conversation import GENERAL_TOO, NOTHING_IN_MEMORY_NOTE

    frames = _Frames(looks=[_look("ahead", 900.0, "I see a room with a red curtain.")])
    brain = _Brain(_call("remember", "how do you work", about="anything"),
                   {"reply": "I am not sure.", "token_count": 1})
    _turn("How do you work?", brain, clock=lambda: 1000.0,
          recall_fn=lambda query: Recalled([], []),
          recall_seen_fn=lambda query, direction=None: recall_seen(query, frame_memory=frames))
    assert frames.word_queries == ["how do you work"]
    assert frames.looks_asked == []
    assert brain.requests[1].tool_result["result"] == {
        "found": [], "note": NOTHING_IN_MEMORY_NOTE + GENERAL_TOO}


def test_an_anything_question_that_names_nothing_is_answered_from_the_conversation():
    """Marked `anything`, "what have you seen?" finds no frame by its words —
    nothing it named is in them — and `anything` never gets the day: it is
    answered from the conversation. Measured, 2 of 29 questions about what
    was seen came as `anything`; the word lists that once sent such a
    question to the latest looks are gone, and asked which half it meant,
    the model never said (demo/conversation.py's `about`)."""
    frames = _Frames(looks=[_look("right", 940.0, "I see a door.")])
    found = Recalled([], [], recent=["Sasha: hi — Reachy: hello"])
    brain = _Brain(_call("remember", "anything"),
                   {"reply": "We said hello.", "token_count": 1})
    _turn("What have you seen?", brain, clock=lambda: 1000.0,
          recall_fn=lambda query: found,
          recall_seen_fn=lambda query, direction=None: recall_seen(query, frame_memory=frames),
          day_frames_fn=lambda: pytest.fail("`anything` never gets the day"))
    assert brain.requests[1].tool_result["result"] == {
        "you_talked_about": ["Sasha: hi — Reachy: hello"]}



def test_an_anything_question_about_what_was_seen_with_nothing_else_is_the_sighting():
    """Recorded, not wanted: marked `anything`, with an empty conversation,
    "what did you see today?" is answered with the bare sighting — "I saw
    Sasha moments ago" — because no frame's words hold it, `anything` never
    gets the day, and a sighting is the answer when nothing else is (the
    same rule "who did you see today?" needs). Measured, 2 of 29 questions
    about what was seen come as `anything`."""
    frames = _Frames(looks=[_look("right", 940.0, "I see a door.")])
    brain = _Brain(_call("remember", "what did you see today?"),
                   {"reply": "I saw Sasha moments ago.", "token_count": 1})
    _turn("What did you see today?", brain, clock=lambda: 1000.0,
          recall_fn=lambda query: Recalled([], []),
          recall_seen_fn=lambda query, direction=None: recall_seen(query, frame_memory=frames),
          day_frames_fn=lambda: pytest.fail("`anything` never gets the day"),
          who_fn=lambda: {"seen_earlier": ["Sasha, moments ago"]})
    assert brain.requests[1].tool_result["result"] == {"people_you_saw": ["Sasha, moments ago"]}


def test_a_fact_that_cleared_the_gate_on_a_bare_question_brings_no_pictures():
    """Live: "tell me how does your memory work?" came as
    `anything`, the fact answered at 0.74 — and the latest looks came along
    as "you_looked_at", so the screen showed the day's pictures under an
    answer about Qdrant Edge. No frame's words hold "how does your memory
    work" (real bge: test_frame_memory.py), and nothing falls back to the
    looks any more."""
    display = _Display()
    frames = _Frames(looks=[_look("left", 900.0, "I see a lamp.")])
    brain = _Brain(_call("remember", "how does your memory work", about="anything"),
                   {"reply": "In Qdrant Edge shards.", "token_count": 1})
    _turn("Tell me how does your memory work?", brain, display=display,
          clock=lambda: 1000.0,
          knowledge_fn=lambda query: [
              {"text": "My memory lives in Qdrant Edge shards.", "score": 0.74,
               "source": "knowledge"}],
          recall_fn=lambda query: Recalled([], []),
          recall_seen_fn=lambda query, direction=None: recall_seen(query, frame_memory=frames))
    result = brain.requests[1].tool_result["result"]
    assert result == {"facts_you_were_taught": ["My memory lives in Qdrant Edge shards."]}
    assert not [e for e in display.events if e[0] == "frames" and e[1]]
    assert frames.looks_asked == []


def test_about_anything_is_answered_as_anything_whatever_the_words():
    """The words of a question never change the model's `about`. "How do you
    work?" marked `anything` is answered as `anything`: the knowledge base is
    searched, the frames by their words — which hold nothing of it — and,
    with nothing over any gate, the latest exchanges."""
    asked = []

    def knowledge_fn(query):
        asked.append(query)
        return []

    frames = _Frames()
    found = Recalled([], [], recent=["Sasha: Look left. — Reachy: I see a curtain."])
    brain = _Brain(_call("remember", "how do you work", about="anything"),
                   {"reply": "I listen and remember.", "token_count": 1})
    _turn("How do you work?", brain, knowledge_fn=knowledge_fn,
          recall_fn=lambda query: found,
          recall_seen_fn=lambda query, direction=None: recall_seen(query, frame_memory=frames))
    assert asked == ["how do you work"]
    assert frames.word_queries == ["how do you work"]
    assert brain.requests[1].tool_result["result"] == {
        "you_talked_about": ["Sasha: Look left. — Reachy: I see a curtain."]}


def test_taught_is_answered_from_the_facts_and_keeps_the_exchanges_out():
    """Live: "how do you work?" missed its own fact, the tool came
    back with the last three exchanges instead, and the robot answered "I
    work by processing information" — no Qdrant. The phrasings stored with
    each fact find it now (demo/knowledge.py); and for `taught` the
    exchanges stay out — they were the noise it answered from."""
    found = Recalled([], [], recent=["Sasha: Look left. — Reachy: I see a curtain."])
    brain = _Brain(_call("remember", "how do you work", about="taught"),
                   {"reply": "I work with Qdrant Edge.", "token_count": 1})
    _turn("How do you work?", brain,
          knowledge_fn=lambda query: [{"text": "I work with Qdrant Edge.",
                                       "score": 0.93, "source": "knowledge"}],
          recall_fn=lambda query: found,
          recall_seen_fn=lambda query, direction=None: pytest.fail("no frames for taught"))
    assert brain.requests[1].tool_result["result"] == {
        "facts_you_were_taught": ["I work with Qdrant Edge."]}


def test_the_facts_are_searched_for_every_question_but_one_about_the_person():
    # The one memory tool searches every store, and the scores decide what
    # answers; only a question about the person asking goes to the faces.
    asked = []

    def knowledge_fn(query):
        asked.append(query)
        return []

    for about, question in (("seen", "Do you remember the mug?"),
                            ("said", "What did we talk about?"),
                            ("anything", "What do you remember?"),
                            ("taught", "How do you work?"),
                            ("me", "Do you remember me?")):
        brain = _Brain(_call("remember", question, about=about),
                       {"reply": "Nothing about that.", "token_count": 1})
        _turn(question, brain, knowledge_fn=knowledge_fn,
              recall_fn=lambda query: Recalled([], []),
              recall_seen_fn=lambda query, direction=None: [],
              day_frames_fn=lambda: [], who_fn=lambda: {})
    assert asked == ["Do you remember the mug?", "What did we talk about?",
                     "What do you remember?", "How do you work?"]


def test_a_camera_call_is_the_camera_whatever_tense_the_question_is_in():
    """The tool the model called is the tool that runs. "Did you see the
    pollution?" sent to `camera` used to be rewritten into `remember(seen)` on
    the tense of its words (a crutch, removed before publication): now the
    camera is looked through, and nothing in memory is read."""
    brain = _Brain(_call("camera", "did you see the pollution?", direction="ahead"),
                   {"reply": "I see no pollution.", "token_count": None})
    _turn("Did you see the pollution?", brain, clock=lambda: 1000.0,
          camera_jpeg=lambda: b"NOW",
          recall_fn=lambda query: pytest.fail("not memory"),
          recall_seen_fn=lambda query, direction=None: pytest.fail("not memory"),
          day_frames_fn=lambda: pytest.fail("not the day"))
    request = brain.requests[1]
    assert (request.image_jpeg, request.image_note) == (b"NOW", LOOK_NOTE)


def test_the_models_about_is_answered_as_it_was_given():
    """`about` is the model's word, run as given — the question's words never
    move it. "Did you see me today?" marked `seen` reads the day back (it
    used to be forced to `me`); "did you see anything today?" marked
    `anything` does not (it used to be forced to `seen`)."""
    day = [_frame(0.0, ts=900.0), _frame(0.0, ts=800.0)]
    brain = _Brain(_call("remember", "did you see me today?", about="seen"),
                   {"reply": "I saw a chair and a lamp.", "token_count": 1})
    _turn("Did you see me today?", brain, clock=lambda: 1000.0,
          recall_seen_fn=lambda query, direction=None: [],
          day_frames_fn=lambda: day,
          who_fn=lambda: pytest.fail("not the faces: the model said `seen`"))
    assert len(brain.requests[1].image_jpegs) == 2
    brain = _Brain(_call("remember", "what did you see", about="anything"),
                   {"reply": "Nothing much.", "token_count": 1})
    _turn("Did you see anything today?", brain, clock=lambda: 1000.0,
          recall_fn=lambda query: Recalled([], []),
          recall_seen_fn=lambda query, direction=None: [],
          day_frames_fn=lambda: pytest.fail("not the day: the model said `anything`"))
    assert brain.requests[1].image_jpegs == ()
    assert brain.requests[1].tool_result is not None


def test_no_question_gets_a_picture_guess():
    """Live: "tell me about the universe" — about=anything — matched nothing
    by words, so the nearest-picture search ran and put a frame of the room
    on the screen as a guess under an answer about galaxies. Now no question
    gets that guess: `anything` gets the frames' words, `seen` the frames'
    words and then the day — and with no day either, nothing."""
    from demo.conversation import NOTHING_IN_MEMORY_NOTE

    frames = _Frames()
    for about, question in (("anything", "Tell me about the universe."),
                            ("anything", "What was the thing I showed you?"),
                            ("seen", "Do you remember the bottle?")):
        display = _Display()
        brain = _Brain(_call("remember", question, about=about),
                       {"reply": "...", "token_count": 1})
        _turn(question, brain, display=display, recall_fn=lambda query: Recalled([], []),
              recall_seen_fn=lambda query, direction=None: recall_seen(query, frame_memory=frames),
              day_frames_fn=lambda: day_frames(frame_memory=frames))
        assert brain.requests[1].image_jpegs == (), question
        assert not [e for e in display.events if e[0] == "frames" and e[1]], question
    assert brain.requests[1].tool_result["result"] == {"note": NOTHING_IN_MEMORY_NOTE}


def test_a_seen_question_is_answered_from_frames_never_from_the_conversation():
    """Live: "did you see people?" and "did you see any TV?" came
    back with earlier exchanges — "Did you see the pollution? — I do not see
    any pollution in front of me" — and the model copied their tense: "I do
    not see any people in front of me right now". A question about what was
    SEEN is answered from frames (A, B, C); the conversation is case D. Here
    "did you see the table?" gets the frame of the table as a picture, and
    the exchange that would have matched is never asked for."""
    display = _Display()
    strong = _frame(0.4, ts=900.0)
    brain = _Brain(_call("remember", "table", about="seen"),
                   {"reply": "I saw a small table.", "token_count": 1})
    _turn("Did you see the table?", brain, display=display, clock=lambda: 1000.0,
          recall_fn=lambda query: pytest.fail("seen never reads the conversation"),
          recall_seen_fn=lambda query, direction=None: [strong])
    assert brain.requests[1].image_jpeg == b"FRAME"
    frames = [hits for kind, *rest in display.events for hits in rest if kind == "frames"]
    assert frames[-1] == [{**strong, "weak": False}]


def test_a_seen_question_no_frame_s_words_answer_gets_the_day():
    """Live: "Nice, what did you see today?" was judged by a list of words to
    be about "nice", missed the day-frames case, and came back as four frames
    of the same person under a sentence copied from an old exchange. Now
    nothing judges the words: no frame's words hold anything it asked
    about, so it gets the day."""
    asked = []

    def recall_seen_fn(query, direction=None):
        asked.append(query)
        return []

    day = [_frame(0.0, ts=900.0), _frame(0.0, ts=800.0)]
    brain = _Brain(_call("remember", "Nice, what did you see today?", about="seen"),
                   {"reply": "I saw a chair and a lamp.", "token_count": 1})
    _turn("Nice, what did you see today?", brain, clock=lambda: 1000.0,
          recall_seen_fn=recall_seen_fn, day_frames_fn=lambda: day)
    assert len(brain.requests[1].image_jpegs) == 2
    assert asked == ["Nice, what did you see today?"]


def test_a_seen_question_about_a_thing_no_frame_holds_gets_the_day():
    """"Did you see a dog?" with no dog in any frame used to be told "you did
    not see it"; now the model looks at the day and says so itself — the
    question named a thing, and that no longer decides anything. Measured:
    "... I did not see a dog in those views"."""
    day = [_frame(0.0, ts=900.0), _frame(0.0, ts=800.0)]
    for question, query in (("Did you see a dog?", "dog"),
                            ("What did you notice this morning?",
                             "what did you notice this morning")):
        brain = _Brain(_call("remember", query, about="seen"),
                       {"reply": "...", "token_count": 1})
        _turn(question, brain, clock=lambda: 1000.0,
              recall_fn=lambda query: pytest.fail("not the conversation"),
              recall_seen_fn=lambda query, direction=None: [],
              day_frames_fn=lambda: day)
        assert len(brain.requests[1].image_jpegs) == 2, question
        assert brain.requests[1].tool_result is None, question


def test_when_the_words_cannot_be_searched_the_memory_is_said_to_be_unavailable():
    """With bge unreachable the frames' words cannot be searched, and the
    model is told so — not handed the day: against the day, "did you see
    Sasha?" came back "I did not see Sasha" (measured), the denial
    MemoryUnavailable exists to prevent. The day is not even read."""
    from demo.conversation import MEMORY_UNAVAILABLE_NOTE

    def down(query, direction=None):
        raise MemoryUnavailable("frames: OSError: embed service down")

    brain = _Brain(_call("remember", "Did you see Sasha?", about="seen"),
                   {"reply": "I can't search my memory.", "token_count": 1})
    _turn("Did you see Sasha?", brain, clock=lambda: 1000.0,
          recall_seen_fn=down, day_frames_fn=lambda: pytest.fail("not the day"))
    assert brain.requests[1].image_jpegs == ()
    assert MEMORY_UNAVAILABLE_NOTE in str(brain.requests[1].tool_result["result"])


def test_the_old_recall_seen_name_is_a_question_about_what_was_seen():
    # The tool's older name still answers (MEMORY_TOOLS), and says which half
    # as plainly as `about` does: with no `about`, it gets the day too.
    day = [_frame(0.0, ts=900.0), _frame(0.0, ts=800.0)]
    brain = _Brain(_call("recall_seen", "what did you see"),
                   {"reply": "I saw a chair.", "token_count": 1})
    _turn("What did you see?", brain, clock=lambda: 1000.0,
          recall_fn=lambda query: pytest.fail("not the conversation"),
          recall_seen_fn=lambda query, direction=None: [], day_frames_fn=lambda: day)
    assert len(brain.requests[1].image_jpegs) == 2


def test_a_said_question_never_touches_the_frames():
    told = "Sasha: my sister is called Anna — Reachy: lovely"
    found = Recalled(memories=[told],
                     speech_hits=[{"text": told, "score": 0.7, "source": "qdrant"}])
    brain = _Brain(_call("remember", "my sister", about="said"),
                   {"reply": "Anna.", "token_count": 1})
    _turn("What did I say about my sister?", brain, recall_fn=lambda query: found,
          recall_seen_fn=lambda query, direction=None: pytest.fail("no frames"),
          day_frames_fn=lambda: pytest.fail("nor the day"))
    assert brain.requests[1].tool_result["result"] == {"the_person_told_you_before": [told]}


def test_a_side_asked_about_as_anything_with_no_look_that_way_is_nothing():
    # Only that side is searched — never the frames' words without it.
    from demo.conversation import NOTHING_IN_MEMORY_NOTE

    asked = []

    def recall_seen_fn(query, direction=None):
        asked.append(direction)
        return []

    brain = _Brain(_call("remember", "what was there", direction="left"),
                   {"reply": "I did not look there.", "token_count": 1})
    _turn("What was on your left?", brain, clock=lambda: 1000.0,
          recall_fn=lambda query: Recalled([], []), recall_seen_fn=recall_seen_fn,
          day_frames_fn=lambda: pytest.fail("not the day"))
    assert asked == ["left"]
    assert brain.requests[1].tool_result["result"] == {"note": NOTHING_IN_MEMORY_NOTE}


def test_what_is_stored_back_is_decided_by_the_turn_never_by_the_words():
    """A reply built from what memory handed back is never written back
    (Exchange.derived) — and only that, decided by the turn, never by the
    question's words. A search that found nothing leaves an ordinary
    exchange, whatever the model marked it: "Did you notice any pollution?",
    marked `seen`, found nothing. What keeps "I did not notice any" from
    coming back as a memory of seeing is the reading side: a `seen` question
    never reads the conversation (test_a_seen_question_is_answered_from_
    frames_never_from_the_conversation)."""
    nothing = {"recall_fn": lambda query: Recalled([], []),
               "recall_seen_fn": lambda query, direction=None: []}
    window = _window(_Memory())
    brain = _Brain(_call("remember", "pollution", about="seen"),
                   {"reply": "I did not notice any.", "token_count": 1})
    _turn("Did you notice any pollution?", brain, window=window, **nothing)
    brain = _Brain(_call("remember", "me", about="me"),
                   {"reply": "Yes, Sasha!", "token_count": 1})
    _turn("Have you seen me before?", brain, window=window,
          who_fn=lambda: {"in_front_of_you": ["Sasha"]})
    # A general question memory had nothing on: the model's own answer.
    brain = _Brain(_call("remember", "airplanes", about="anything"),
                   {"reply": "Wings make lift.", "token_count": 1})
    _turn("How do airplanes fly?", brain, window=window, **nothing)
    brain = _Brain({"reply": "I do not see any pollution.", "token_count": 1})
    _turn("Did you see the pollution?", brain, window=window)
    assert [e.derived for e in window._exchanges] == [False, True, False, False]


def test_a_statement_the_model_searched_on_is_still_remembered():
    # Live: the model called remember(about=said) before answering "My dog
    # is called Rex.", found nothing, and the exchange was never stored —
    # what the person told the robot was lost.
    memory = _Memory()
    window = _window(memory, budget_tokens=10)
    brain = _Brain(_call("remember", "the name of the dog", about="said"),
                   {"reply": "Rex is a lovely name!", "token_count": 500})
    _turn("My dog is called Rex.", brain, window=window,
          recall_fn=lambda query: Recalled([], []),
          recall_seen_fn=lambda query, direction=None: [])
    window.flush()
    assert any("Rex" in text for text, _kind, _meta in memory.remembered)


def test_a_taught_question_nothing_answers_is_a_general_one():
    # Live: "tell me about black holes" came as `taught`, was given the latest
    # exchanges instead, and the robot said it had no information "in my
    # memory".
    from demo.conversation import GENERAL_QUESTION_NOTE

    brain = _Brain(_call("remember", "black holes", about="taught"),
                   {"reply": "Black holes are regions of space.", "token_count": 1})
    _turn("Tell me about black holes.", brain, knowledge_fn=lambda query: [],
          recall_fn=lambda query: pytest.fail("not a memory question"))
    assert brain.requests[1].tool_result["result"] == {
        "found": [], "note": GENERAL_QUESTION_NOTE}


def test_an_invented_tool_is_answered_as_handled_not_refused():
    brain = _Brain(_call("yes_dance"), {"reply": "Sure, dancing!", "token_count": 1})
    _turn("Dance for me!", brain, recall_fn=lambda query: pytest.fail("not a memory call"))
    assert brain.requests[1].tool_result["result"]["done"] is True


def test_the_model_is_offered_one_memory_tool_the_camera_and_the_body():
    from demo.chat_session import CHAT_TOOLS

    assert [tool["function"]["name"] for tool in CHAT_TOOLS] == ["remember", "camera", "move"]


def test_the_picture_the_camera_gives_the_model_goes_to_the_screen():
    """The camera tool's own picture goes to the display, under the tool line
    — the same bytes the model gets."""
    display = _Display()
    brain = _Brain({"tool_call": {"name": "camera", "arguments": {"direction": "ahead"}}},
                   {"reply": "I see a mug.", "token_count": None})
    _turn("What do you see?", brain, display=display, camera_jpeg=lambda: b"JPEG-NOW")
    assert ("look", b"JPEG-NOW") in display.events
    assert brain.requests[1].image_jpeg == b"JPEG-NOW"


def test_tell_me_about_is_a_request_not_a_question_about_the_past():
    # "Tell me about black holes" read "tell" as the past and got "Nothing in
    # your memory about this." — the model then refused a general question.
    # Marked `anything`, it now gets both notes — nothing in memory, and if
    # it is a general question, answer it — and the model tells which.
    from demo.conversation import GENERAL_TOO, NOTHING_IN_MEMORY_NOTE

    brain = _Brain(_call("remember", "black holes", about="anything"),
                   {"reply": "Black holes are regions of space.", "token_count": 1})
    _turn("Tell me about black holes.", brain,
          recall_fn=lambda query: Recalled([], []),
          recall_seen_fn=lambda query, direction=None: [])
    assert brain.requests[1].tool_result["result"] == {
        "found": [], "note": NOTHING_IN_MEMORY_NOTE + GENERAL_TOO}


