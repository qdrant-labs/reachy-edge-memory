"""FrameMemory: SigLIP whole-frame visual memory, searched by its words.

The SigLIP ONNX embedder is NOT loaded here — a deterministic one-hot fake
stands in. The fake lets us assert the store and read logic exactly.
"""
from __future__ import annotations

import base64
import io

import numpy as np
import pytest

from emulator.frame_memory import FrameMemory


class FakeEmbedder:
    """One-hot embedder: an image's basis is selected by frame[0,0,0], so two
    frames stamped alike are the same picture (cosine 1.0) and two stamped
    differently share nothing (0.0)."""

    _BASIS = {0: [1.0, 0.0, 0.0], 1: [0.0, 1.0, 0.0], 2: [0.0, 0.0, 1.0]}
    # What each stamp is a picture of, for a question in the same space.
    _LOOKS_LIKE = {"cup": 0, "lamp": 1, "door": 2}

    def embed_image(self, frame_rgb):
        idx = int(np.asarray(frame_rgb)[0, 0, 0])
        return np.asarray(self._BASIS.get(idx, [0.0, 0.0, 0.0]), np.float32)

    def embed_text(self, text):
        # A question lands on the pictures of what it asks about, nearest the
        # first thing it names; one about nothing in particular is as near
        # every picture as any other.
        named = sorted((text.find(thing), i) for thing, i in self._LOOKS_LIKE.items()
                       if thing in text)
        if not named:
            return np.ones(3, np.float32)
        return sum(np.asarray(self._BASIS[i], np.float32) / (rank + 1)
                   for rank, (_, i) in enumerate(named))


def _frame(basis: int) -> np.ndarray:
    return np.full((4, 4, 3), basis, np.uint8)


def _mem() -> FrameMemory:
    return FrameMemory(embedder=FakeEmbedder())


def test_remember_stores_labels_and_a_decodable_jpeg():
    mem = _mem()
    mem.remember(_frame(0), [{"label": "apple", "box": [0, 0, 1, 1], "score": 0.9},
                             {"label": "apple", "box": [0, 0, 1, 1], "score": 0.7}])
    hit = mem.day_frames()[0]
    # labels are de-duplicated metadata, not the vector
    assert hit["labels"] == ["apple"]
    img = base64.b64decode(hit["jpeg_b64"])
    from PIL import Image
    assert Image.open(io.BytesIO(img)).format == "JPEG"


def test_remember_without_detections_still_stores_a_frame():
    mem = _mem()
    mem.remember(_frame(0))  # a frame with no YOLO boxes
    hits = mem.day_frames()
    assert len(hits) == 1
    assert hits[0]["detections"] == []
    assert hits[0]["labels"] == []


# --- SceneChangeWriter: the "on objects" continuous capture policy ---

class _RecordingMemory:
    def __init__(self):
        self.stored: list = []

    def remember(self, frame, detections=None, meta=None):
        self.stored.append((frame, detections))


def _writer(mem, clk, interval=2.0):
    from emulator.frame_memory import SceneChangeWriter
    return SceneChangeWriter(mem, min_interval=interval, clock=lambda: clk["t"])


def test_a_new_object_whose_write_failed_is_stored_on_the_next_try():
    # One embed timeout used to mark the object as stored: it never was, and
    # nothing retried it while it stayed in view.
    class FailsOnce(_RecordingMemory):
        failed = False

        def remember(self, frame, detections=None, meta=None):
            if not self.failed:
                self.failed = True
                raise OSError("embed service timed out")
            super().remember(frame, detections, meta)

    mem, clk = FailsOnce(), {"t": 0.0}
    w = _writer(mem, clk)
    with pytest.raises(OSError):
        w.observe(_frame(0), [{"label": "cup"}])
    clk["t"] = 0.5
    assert w.observe(_frame(1), [{"label": "cup"}]) is False, \
        "retried after the interval, not on every detect cycle"
    clk["t"] = 2.5
    assert w.observe(_frame(2), [{"label": "cup"}]) is True
    assert len(mem.stored) == 1


def test_scene_writer_stores_when_a_new_label_appears():
    mem, clk = _RecordingMemory(), {"t": 0.0}
    w = _writer(mem, clk)
    assert w.observe(_frame(0), [{"label": "cup"}]) is True
    assert len(mem.stored) == 1


def test_scene_writer_skips_a_static_scene():
    mem, clk = _RecordingMemory(), {"t": 0.0}
    w = _writer(mem, clk)
    w.observe(_frame(0), [{"label": "cup"}])          # store
    clk["t"] = 10.0
    assert w.observe(_frame(0), [{"label": "cup"}]) is False  # same labels
    assert len(mem.stored) == 1


def test_scene_writer_stores_again_on_a_genuinely_new_object():
    mem, clk = _RecordingMemory(), {"t": 0.0}
    w = _writer(mem, clk)
    w.observe(_frame(0), [{"label": "cup"}])
    clk["t"] = 10.0
    assert w.observe(_frame(0), [{"label": "cup"}, {"label": "book"}]) is True
    assert len(mem.stored) == 2


def test_scene_writer_throttles_rapid_changes():
    mem, clk = _RecordingMemory(), {"t": 0.0}
    w = _writer(mem, clk, interval=2.0)
    w.observe(_frame(0), [{"label": "cup"}])          # store at t=0
    clk["t"] = 1.0
    assert w.observe(_frame(0), [{"label": "book"}]) is False  # new but throttled
    clk["t"] = 2.5
    assert w.observe(_frame(0), [{"label": "phone"}]) is True   # past interval
    assert len(mem.stored) == 2


def test_an_object_that_arrives_inside_the_interval_is_stored_once_it_passes():
    # It stays in view, so it is still new when the throttle lets go. Marking
    # it seen while throttled lost it for good.
    mem, clk = _RecordingMemory(), {"t": 0.0}
    w = _writer(mem, clk, interval=10.0)
    w.observe(_frame(0), [{"label": "person"}])                       # t=0
    clk["t"] = 3.0
    assert w.observe(_frame(0), [{"label": "person"}, {"label": "cup"}]) is False
    clk["t"] = 11.0
    assert w.observe(_frame(0), [{"label": "person"}, {"label": "cup"}]) is True
    assert len(mem.stored) == 2


def test_scene_writer_skips_when_no_frame():
    mem, clk = _RecordingMemory(), {"t": 0.0}
    w = _writer(mem, clk)
    assert w.observe(None, [{"label": "cup"}]) is False
    assert mem.stored == []
    # the missed frame didn't consume the label change — a real frame still stores
    assert w.observe(_frame(0), [{"label": "cup"}]) is True


# --- concurrency: the detect thread (SceneChangeWriter.observe -> remember)
# and the voice thread (a `remember` search, a look stored by run_demo's Looker) hit
# the SAME FrameMemory with no coordination of their own. The on-disk shard
# ("local mode") does not synchronise its own payload/vector mutation, and
# `_next_id` was a plain read-modify-write — reproduced live with exactly this
# thread layout: 800 expected points landed as 555, with sqlite3.InterfaceError
# / IndexError from the writer and a numpy broadcast ValueError from the
# reader. This test exercises the same layout (on-disk store, one thread only
# writing, one alternating reads with remember) and fails without `_lock`. ---

def test_concurrent_remember_and_reads_do_not_corrupt_the_store(tmp_path):
    import threading


    mem = FrameMemory(str(tmp_path / "frames"), embedder=FakeEmbedder())
    iterations = 150
    errors: list[BaseException] = []

    def detect_thread():
        # Mirrors SceneChangeWriter.observe(): remember() only, off the
        # detect loop.
        for i in range(iterations):
            try:
                mem.remember(_frame(i % 3), [{"label": f"detect{i}"}])
            except Exception as exc:  # noqa: BLE001 — this IS the race being tested
                errors.append(exc)

    def voice_thread():
        # Mirrors the voice thread: a read, then a stored look, per turn.
        for i in range(iterations):
            try:
                mem.day_frames()
                mem.remember(_frame(i % 3), [{"label": f"voice{i}"}])
            except Exception as exc:  # noqa: BLE001 — this IS the race being tested
                errors.append(exc)

    t1 = threading.Thread(target=detect_thread)
    t2 = threading.Thread(target=voice_thread)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert errors == [], f"concurrent reads/remember raised: {errors!r}"
    # Every remember() must have landed its own point — no id collisions
    # silently overwriting one frame's storage with another's.
    assert mem._store.count() == iterations * 2


def test_count_reports_how_many_frames_are_stored():
    mem = _mem()
    assert mem.count() == 0
    mem.remember(_frame(0), [{"label": "apple", "box": [0, 0, 1, 1], "score": 0.9}])
    mem.remember(_frame(1), [])
    assert mem.count() == 2


def test_the_latest_looks_come_newest_first_with_their_captions(monkeypatch):
    import emulator.frame_memory as fm

    clock = iter([10.0, 20.0, 30.0, 40.0])
    monkeypatch.setattr(fm.time, "time", lambda: next(clock))
    mem = _mem()
    left = mem.remember(_frame(0), [{"label": "lamp"}], meta={"looked": "left"})
    mem.remember(_frame(1), [{"label": "mug"}])                 # not a look
    mem.remember(_frame(2), [{"label": "door"}], meta={"looked": "right"})
    mem.remember(_frame(0), [{"label": "book"}], meta={"looked": "ahead"})
    mem.describe(left, "I see a lamp.")
    looks = mem.latest_looks()
    assert [look["labels"] for look in looks] == [["book"], ["door"], ["lamp"]]
    assert looks[2]["caption"] == "I see a lamp." and looks[2]["jpeg_b64"]
    assert [look["labels"] for look in mem.latest_looks(["left", "right"], limit=1)] == [["door"]]
    assert [look["labels"] for look in mem.latest_looks(before=35.0)] == [["door"], ["lamp"]]


def test_scene_writer_names_the_people_only_when_a_person_is_in_view():
    class _Meta(_RecordingMemory):
        def remember(self, frame, detections=None, meta=None):
            self.stored.append(meta)

    from emulator.frame_memory import SceneChangeWriter

    asked, mem, clk = [], _Meta(), {"t": 0.0}
    w = SceneChangeWriter(mem, min_interval=0.0, clock=lambda: clk["t"],
                          people=lambda frame: (asked.append(1), [{"name": "Sasha"}])[1])
    w.observe(_frame(0), [{"label": "cup"}])
    w.observe(_frame(0), [{"label": "cup"}, {"label": "person"}])
    assert mem.stored == [None, {"people": [{"name": "Sasha"}]}]
    assert asked == [1], "no face lookup for a frame without a person"


def test_people_seen_is_who_was_in_a_frame_and_when_last(monkeypatch):
    import emulator.frame_memory as fm

    clock = iter([10.0, 20.0, 30.0])
    monkeypatch.setattr(fm.time, "time", lambda: next(clock))
    mem = _mem()
    mem.remember(_frame(0), [], meta={"people": [{"name": "Sasha"}, {"name": None}]})
    mem.remember(_frame(1), [], meta={"people": [{"name": "Robin"}]})
    mem.remember(_frame(2), [], meta={"people": [{"name": "Sasha"}]})
    assert mem.people_seen() == [("Sasha", 30.0), ("Robin", 20.0)]
    assert mem.people_seen(before=25.0) == [("Robin", 20.0), ("Sasha", 10.0)]


# --- the day, picked BY PICTURE ---
# FakeEmbedder's frames are one-hot, so two frames of the same basis are
# cosine 1.0 — "the same picture", well over DAY_MIN_DISTANCE — and two of
# different bases are 0.0, as far apart as this store can say. That is the
# 0.914-against-0.609 the real 371 frames measured, at its extremes.

def _clock(monkeypatch, stamps):
    import emulator.frame_memory as fm

    ticks = iter(stamps)
    monkeypatch.setattr(fm.time, "time", lambda: next(ticks))


def test_the_day_is_three_different_pictures_not_the_newest_three(monkeypatch):
    """The newest frames are not the day: on the robot's own 371 the newest
    four are one picture four times. Here the three newest are literally the
    same frame, and the answer has to reach past them."""
    _clock(monkeypatch, [10.0, 20.0, 30.0, 40.0, 50.0])
    mem = _mem()
    mem.remember(_frame(1), [{"label": "lamp"}])
    mem.remember(_frame(2), [{"label": "door"}])
    for label in ("cup", "mug", "jug"):        # one picture, three times
        mem.remember(_frame(0), [{"label": label}])
    day = mem.day_frames()
    assert [frame["labels"] for frame in day] == [["jug"], ["door"], ["lamp"]]
    assert all(frame["jpeg_b64"] for frame in day), "the picture IS the memory"


def test_a_day_of_identical_frames_is_one_frame(monkeypatch):
    """Three IDENTICAL frames are one memory — the near-duplicate cutoff.
    Merely similar ones (the same room, minutes apart) all come back now:
    DAY_MIN_DISTANCE moved from 0.85 to 0.97 after a live first minute of
    three chair frames answered "what did you see today?" with one picture."""
    _clock(monkeypatch, [10.0, 20.0, 30.0])
    mem = _mem()
    for label in ("cup", "mug", "jug"):
        mem.remember(_frame(0), [{"label": label}])
    assert [frame["labels"] for frame in mem.day_frames()] == [["jug"]]


def test_the_day_stops_at_before(monkeypatch):
    """`before` is the turn's own start: the frame the camera took while the
    question was being asked is not something the robot saw "today"."""
    _clock(monkeypatch, [10.0, 20.0, 30.0])
    mem = _mem()
    mem.remember(_frame(1), [{"label": "lamp"}])
    mem.remember(_frame(2), [{"label": "door"}])
    mem.remember(_frame(0), [{"label": "cup"}])
    day = mem.day_frames(before=25.0)
    assert [frame["labels"] for frame in day] == [["door"], ["lamp"]]


def test_a_look_is_a_candidate_however_many_scenes_buried_it(monkeypatch):
    """A look is in the pool by being a look, not by being recent: the robot
    stored 7 of them among 371 frames, and the scene writer adds one every
    10 s on top."""
    import itertools

    from emulator.frame_memory import DAY_POOL_SCENES

    _clock(monkeypatch, itertools.count(10.0, 10.0))
    mem = _mem()
    mem.remember(_frame(1), [{"label": "lamp"}], meta={"looked": "left"})
    for _ in range(DAY_POOL_SCENES + 5):
        mem.remember(_frame(0), [{"label": "cup"}])
    day = mem.day_frames()
    assert [frame["labels"] for frame in day] == [["cup"], ["lamp"]]
    assert day[1]["looked"] == "left"


def test_min_distance_is_what_counts_as_a_different_picture(monkeypatch):
    # Orthogonal frames are cosine 0.0, so a gate below zero is what makes
    # "different enough" unreachable — the knob is the caller's, and the
    # 0.85 default is only where this robot's evening happened to separate.
    _clock(monkeypatch, [10.0, 20.0])
    mem = _mem()
    mem.remember(_frame(1), [{"label": "lamp"}])
    mem.remember(_frame(2), [{"label": "door"}])
    assert len(mem.day_frames()) == 2
    assert [frame["labels"] for frame in mem.day_frames(min_distance=-0.5)] == [["door"]]


def test_the_day_of_a_robot_that_has_seen_nothing_is_empty():
    assert _mem().day_frames() == []
    assert _mem().day_frames(like="a door") == []


def test_the_day_is_led_by_the_picture_nearest_the_question(monkeypatch):
    """A door no label names is found by what it looks like: the picture
    search picks the day's first frame, and the rest are picked around it."""
    _clock(monkeypatch, [10.0, 20.0, 30.0])
    mem = _mem()
    mem.remember(_frame(2), [{"label": "person"}])   # a door; no word says so
    mem.remember(_frame(1), [{"label": "person"}])
    mem.remember(_frame(0), [{"label": "person"}])
    assert [frame["ts"] for frame in mem.day_frames(limit=1, like="did you see a door?")] == [10.0]
    assert [frame["ts"] for frame in mem.day_frames(limit=1)] == [30.0], "no question: the newest"
    assert [frame["ts"] for frame in mem.day_frames(limit=1, like="  ")] == [30.0]
    # Led by the door, the day is still the day, newest first.
    assert [frame["ts"] for frame in mem.day_frames(like="a door")] == [30.0, 20.0, 10.0]


def test_a_picture_older_than_the_pool_still_leads(monkeypatch):
    # The pool is the newest scenes; the search runs over every frame, so an
    # old picture of the thing asked about is not aged out of the answer.
    import itertools

    import emulator.frame_memory as frame_memory

    monkeypatch.setattr(frame_memory, "DAY_POOL_SCENES", 2)
    _clock(monkeypatch, itertools.count(10.0, 10.0))
    mem = _mem()
    mem.remember(_frame(2), [{"label": "person"}])
    for _ in range(4):
        mem.remember(_frame(0), [{"label": "person"}])
    assert [frame["ts"] for frame in mem.day_frames(limit=1, like="a door")] == [10.0]


def test_a_picture_taken_during_this_turn_never_leads(monkeypatch):
    # The camera's frame of the door, taken while the question was asked, is
    # the present: the nearest picture from before the turn leads instead,
    # not merely the newest.
    _clock(monkeypatch, [10.0, 20.0, 30.0])
    mem = _mem()
    mem.remember(_frame(1), [{"label": "person"}])   # a lamp
    mem.remember(_frame(0), [{"label": "person"}])   # a cup
    mem.remember(_frame(2), [{"label": "person"}])   # the door, during the turn
    day = mem.day_frames(limit=1, before=25.0, like="a door, or else a lamp")
    assert [frame["ts"] for frame in day] == [10.0]


def test_the_lead_is_the_frame_itself_not_its_time(monkeypatch):
    # Two frames stored in the same instant: the one whose picture matched
    # leads, not whichever of the two the walk met first.
    _clock(monkeypatch, [10.0, 10.0])
    mem = _mem()
    mem.remember(_frame(0), [{"label": "table"}])    # a cup
    mem.remember(_frame(2), [{"label": "chair"}])    # a door
    assert [frame["labels"] for frame in mem.day_frames(limit=1, like="a door")] == [["chair"]]
    assert [frame["labels"] for frame in mem.day_frames(limit=1, like="a cup")] == [["table"]]


def _down(text):
    raise OSError("connection refused")


def _tower_failed(text):
    raise RuntimeError("[ONNXRuntimeError] : 6 : RUNTIME_EXCEPTION : inference failed")


@pytest.mark.parametrize("embed_text", [
    _down,
    lambda text: np.asarray(None, dtype=np.float32),   # {"vector": null}
    lambda text: np.ones((1, 3), np.float32),           # nested
    lambda text: np.ones(4, np.float32),                # another size
    lambda text: np.full(3, np.nan, np.float32),
    lambda text: [10 ** 1000, 0, 0],                    # a number no float holds
    lambda text: np.zeros(3, np.float32),               # no direction at all
    lambda text: np.full(3, 1e-30, np.float32),         # a length that underflows
    lambda text: np.full(3, 3e38, np.float32),          # a length that overflows
    _tower_failed,                                       # the text tower on the robot
], ids=["down", "null", "nested", "size", "nan", "overflow", "zero", "tiny", "huge",
        "onnx"])
def test_no_picture_search_still_leaves_the_day(monkeypatch, caplog, embed_text):
    # The words were searched already and found nothing: with the embed
    # service down for the question, answering no vector, or the text tower
    # failing, the day is still the answer — led by the newest frame, and
    # said in the log.
    import itertools

    _clock(monkeypatch, itertools.count(10.0, 10.0))  # the log reads the clock too
    embedder = FakeEmbedder()
    mem = FrameMemory(embedder=embedder)
    mem.remember(_frame(2), [{"label": "person"}])
    mem.remember(_frame(0), [{"label": "person"}])
    embedder.embed_text = embed_text
    with caplog.at_level("WARNING", logger="emulator.frame_memory"):
        day = mem.day_frames(limit=1, like="a door")
    assert [frame["ts"] for frame in day] == [20.0]
    assert any("no picture search" in record.getMessage() for record in caplog.records)


@pytest.mark.parametrize("embed_text, traceback", [(_down, False), (_tower_failed, True)],
                         ids=["outage", "bug"])
def test_only_a_failure_that_is_no_outage_is_logged_with_its_traceback(
        monkeypatch, caplog, embed_text, traceback):
    # An embed service that is down says so in one line; anything else may
    # be a bug, and its traceback is what finds it.
    import itertools

    _clock(monkeypatch, itertools.count(10.0, 10.0))
    embedder = FakeEmbedder()
    mem = FrameMemory(embedder=embedder)
    mem.remember(_frame(0), [{"label": "person"}])
    embedder.embed_text = embed_text
    with caplog.at_level("WARNING", logger="emulator.frame_memory"):
        mem.day_frames(like="a door")
    [record] = [r for r in caplog.records if "no picture search" in r.getMessage()]
    assert bool(record.exc_info) is traceback


def test_a_question_and_a_picture_in_different_spaces_fail_the_start():
    # A text tower that is missing or of another size would fail the first
    # question no frame's words answer; it fails the start instead.
    class Mismatched(FakeEmbedder):
        def embed_text(self, text):
            return np.ones(4, np.float32)

    with pytest.raises(ValueError, match="not one space"):
        FrameMemory(embedder=Mismatched())


class FakeWords:
    """A bge stand-in for the frame's own words: a text lands on the basis of
    whichever keyword it contains, so a question about a bottle is cosine 1.0
    with a frame that has one, and 0.0 with one that does not."""

    _BASIS = {"bottle": [1.0, 0.0, 0.0], "sasha": [0.0, 1.0, 0.0],
              "curtain": [0.0, 0.0, 1.0]}

    def _one(self, text):
        low = text.lower()
        vector = np.zeros(3, np.float32)
        for word, basis in self._BASIS.items():
            if word in low:
                vector += np.asarray(basis, np.float32)
        norm = float(np.linalg.norm(vector))
        # A frame's words carry several keywords at once ("I saw bottle,
        # person. Sasha was there"), so they share the basis between them —
        # cosine 0.707 against a one-word question, still over the gate.
        return vector / norm if norm else vector

    def embed(self, texts):
        return [self._one(text) for text in texts]

    def query_embed(self, texts):
        return [self._one(text) for text in texts]


def test_a_stored_frame_carries_its_objects_and_its_people_and_is_found_by_them():
    """The objects and the names were always written into the point — and
    for a long time nothing ever searched them: the old picture search's
    label filter was never used by the demo, and the only text vector written was a look's caption (7 frames
    of the 371 on the robot). Stored is not indexed."""
    from emulator.frame_memory import SceneChangeWriter

    mem = FrameMemory(embedder=FakeEmbedder(), text_embedder=FakeWords())
    writer = SceneChangeWriter(mem, min_interval=0.0, clock=lambda: 0.0,
                               people=lambda frame: [{"name": "Sasha", "box": [0, 0, 1, 1],
                                                      "score": 0.8},
                                                     {"name": None, "box": [0, 0, 1, 1],
                                                      "score": 0.1}])
    assert writer.observe(_frame(0), [{"label": "bottle", "box": [0, 0, 1, 1], "score": 0.9},
                                      {"label": "person", "box": [0, 0, 1, 1], "score": 0.8}])
    point_id, payload = next(iter(mem._store.iter_points()))

    # the objects YOLO saw, the people the face shard named, and the raw boxes
    assert payload["labels"] == ["bottle", "person"]
    assert payload["names"] == ["Sasha"]
    assert [person["name"] for person in payload["people"]] == ["Sasha", None]
    assert payload["detections"][0]["label"] == "bottle"

    # …as one sentence, and as a vector on the point, which is what makes them
    # findable at all
    assert mem.frame_text(payload) == "I saw bottle, person. Sasha was there"
    assert "text" in mem._store.vectors(point_id)

    # …and both halves answer a question asked in words
    assert mem.recall_text("did you see a bottle?")[0]["labels"] == ["bottle", "person"]
    assert mem.recall_text("did you see Sasha?")[0]["names"] == ["Sasha"]
    assert mem.recall_text("was there a curtain?") == [], "the gate still holds"


class FakeWordsThatSee(FakeWords):
    """FakeWords with one more direction: seeing. Every frame's words say "I
    saw", and so does every question about seeing — the shared verb the real
    bge scores too (EMPTY_FRAME)."""

    _BASIS = {"bottle": [1.0, 0.0, 0.0, 0.0], "sasha": [0.0, 1.0, 0.0, 0.0],
              "curtain": [0.0, 0.0, 1.0, 0.0], "saw": [0.0, 0.0, 0.0, 1.0],
              "see": [0.0, 0.0, 0.0, 1.0]}

    def _one(self, text):
        low = text.lower()
        vector = np.zeros(4, np.float32)
        for word, basis in self._BASIS.items():
            if word in low:
                vector += np.asarray(basis, np.float32)
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm else vector


def test_a_frame_whose_words_only_share_i_saw_with_the_question_is_no_match():
    """"What did you see?" against "I saw bottle": cosine 0.707, over the
    gate — on the verb alone. The empty frame, "I saw something", scores
    1.0: no frame beats it, so nothing comes back. "Did you see a bottle?"
    scores 1.0 on the bottle and 0.707 on the empty frame, so it does."""
    from emulator.frame_memory import FRAME_TEXT_MIN_SCORE

    mem = FrameMemory(embedder=FakeEmbedder(), text_embedder=FakeWordsThatSee())
    mem.remember(_frame(0), [{"label": "bottle"}])
    assert np.isclose(mem._empty_frame_score(FakeWordsThatSee()._one("what did you see?")), 1.0)
    hit = mem._store.search([float(x) for x in FakeWordsThatSee()._one("what did you see?")],
                            1, mem._frames(), using="text")[0]
    assert hit["score"] >= FRAME_TEXT_MIN_SCORE, "over the gate on the verb alone"
    assert mem.recall_text("what did you see?") == []
    assert mem.recall_text("did you see a bottle?")[0]["labels"] == ["bottle"]


def _bge_cached() -> bool:
    try:
        from fastembed import TextEmbedding

        TextEmbedding("BAAI/bge-small-en-v1.5", local_files_only=True, lazy_load=True)
        return True
    except Exception:  # noqa: BLE001 — not cached, or fastembed missing
        return False


@pytest.mark.skipif(not _bge_cached(), reason="bge-small is not cached")
def test_with_the_real_bge_a_question_that_names_nothing_finds_no_frame():
    """The measurement EMPTY_FRAME rests on, on real bge: frames written the
    way the robot writes them. Without the empty frame, "What did you see?"
    scored 0.704 on "I saw person" and "Did you see a cup?" 0.676 on "I saw
    plant" — both over the 0.66 gate."""
    from emulator.memory import _embedder

    mem = FrameMemory(embedder=FakeEmbedder(),
                      text_embedder=_embedder("BAAI/bge-small-en-v1.5"))
    for i, (labels, meta) in enumerate((
            (["person"], {}), (["plant"], {}), (["person", "plant"], {}),
            (["person"], {"people": [{"name": "Sasha"}]}),
            (["chair", "laptop"], {}), (["window"], {"looked": "left"}))):
        mem.remember(_frame(i % 3), [{"label": label} for label in labels], meta=meta)
    for question in ("What did you see today?", "Okay, so what did you see?",
                     "What did you see?", "Nice, what did you see today?",
                     "What did you see before?", "Did you see a cup?",
                     "Did you see a dog?", "What did you notice this morning?",
                     "Anything interesting you saw?", "What did you see before lunch?",
                     "Have you seen my keys?",
                     # and the questions that are not about seeing at all,
                     # which `anything` sends here too
                     "How do you work?", "how does your memory work",
                     "Tell me about the universe.", "What did we talk about?"):
        assert mem.recall_text(question) == [], question
    assert "plant" in mem.recall_text("Did you see a plant?")[0]["labels"]
    assert mem.recall_text("Did you see Sasha?")[0]["names"] == ["Sasha"]
    assert "laptop" in mem.recall_text("Was there a laptop on the desk?")[0]["labels"]


def test_a_look_is_stored_with_its_side_and_re_embedded_when_described():
    mem = FrameMemory(embedder=FakeEmbedder(), text_embedder=FakeWords())
    point_id = mem.remember(_frame(0), [{"label": "bottle"}], meta={"looked": "left"})
    payload = dict(next(iter(mem._store.iter_points()))[1])
    assert mem.frame_text(payload) == "On my left. I saw bottle"
    mem.describe(point_id, "I see a window with dark brown curtains.")
    payload = dict(next(iter(mem._store.iter_points()))[1])
    # the caption joins the words, it does not replace them
    assert mem.frame_text(payload) == ("On my left. I saw bottle. "
                                       "I see a window with dark brown curtains.")
    assert mem.recall_text("was there a curtain?")[0]["looked"] == "left"


def test_the_names_in_a_frame_are_a_flat_list_like_the_labels():
    mem = _mem()
    mem.remember(_frame(0), [{"label": "person"}],
                 meta={"people": [{"name": "Sasha"}, {"name": None}, {"name": "Robin"}]})
    mem.remember(_frame(1), [{"label": "cup"}])
    stored = [payload for _id, payload in mem._store.iter_points()]
    assert sorted(p.get("names", []) for p in stored) == [[], ["Robin", "Sasha"]]


# — recall: by the words or by the picture, whichever singles a frame out —

def _words_and_pictures(monkeypatch, stamps_and_labels, *, names=None):
    import itertools

    _clock(monkeypatch, itertools.count(10.0, 10.0))
    mem = FrameMemory(embedder=FakeEmbedder(), text_embedder=FakeWords())
    for i, (stamp, labels) in enumerate(stamps_and_labels):
        meta = {"names": names[i]} if names and names[i] else None
        mem.remember(_frame(stamp), [{"label": label} for label in labels], meta=meta)
    return mem


def test_a_frame_its_picture_singles_out_beats_frames_the_words_only_match(monkeypatch):
    # The words hold "bottle" on five frames of the same cup, and on none of
    # the door; asked about a bottle by the door, the one frame of a door
    # stands out among all of them by its picture far more than any of the
    # five does by its words — the robot's "a door" on "I saw chair,
    # keyboard, person".
    mem = _words_and_pictures(monkeypatch, [(0, ["bottle"])] * 5 + [(2, ["person"])])
    found = mem.recall("a bottle by the door")
    assert found[0]["labels"] == ["person"] and found[0]["found_by"] == "picture"


def test_a_name_is_found_by_the_words_when_no_picture_stands_out(monkeypatch):
    # SigLIP knows no one's name: "Did you see Sasha?" lands as near every
    # picture as any other, and the frame her name is written on comes first.
    mem = _words_and_pictures(monkeypatch, [(0, ["person"]), (1, ["person"]), (2, ["person"])],
                              names=[None, ["Sasha"], None])
    found = mem.recall("Did you see Sasha?")
    assert found[0]["names"] == ["Sasha"] and found[0]["found_by"] == "words"


def test_nothing_the_words_hold_is_nothing_whatever_the_pictures_say(monkeypatch):
    # Whether the question names anything is the words' to say: the door's
    # picture is there, but no frame's words hold a curtain.
    mem = _words_and_pictures(monkeypatch, [(0, ["bottle"]), (2, ["person"])])
    assert mem.recall("a curtain by the door") == []


def test_a_frame_from_this_turn_is_never_found(monkeypatch):
    mem = _words_and_pictures(monkeypatch, [(0, ["bottle"])] * 3 + [(2, ["bottle"])])
    found = mem.recall("a bottle by the door", before=35.0)
    assert found and all(frame["ts"] < 35.0 for frame in found)


def test_without_the_picture_search_the_words_still_find(monkeypatch, caplog):
    mem = _words_and_pictures(monkeypatch, [(0, ["person"]), (2, ["bottle"])])
    mem._embedder.embed_text = _down
    with caplog.at_level("WARNING", logger="emulator.frame_memory"):
        found = mem.recall("a bottle by the door")
    assert found[0]["labels"] == ["bottle"] and found[0]["found_by"] == "words"
    assert any("no picture search" in record.getMessage() for record in caplog.records)


def test_frames_without_words_score_as_the_empty_frame_does(monkeypatch):
    # Frames stored while bge was down carry no words. Left out of the words'
    # spread, the one frame whose words held the bottle stood out by nothing,
    # and two frames of a lamp the picture search liked came first.
    import itertools

    _clock(monkeypatch, itertools.count(10.0, 10.0))
    words = FakeWords()
    mem = FrameMemory(embedder=FakeEmbedder(), text_embedder=words)
    mem.remember(_frame(0), [{"label": "bottle"}])
    working = words.embed
    words.embed = lambda texts: (_ for _ in ()).throw(OSError("bge down"))
    for stamp in (1, 1, 2, 2):
        mem.remember(_frame(stamp), [{"label": "person"}])
    words.embed = working
    mem._embedder.embed_text = lambda text: np.asarray([0.0, 1.0, 0.0], np.float32)  # the lamp
    found = mem.recall("a bottle")
    assert found[0]["labels"] == ["bottle"] and found[0]["found_by"] == "words"


def test_one_bge_call_finds_and_ranks(monkeypatch):
    # The gate and the ranking share one embedding: a second call that
    # failed would lose a question the first had already found.
    mem = _words_and_pictures(monkeypatch, [(0, ["bottle"]), (2, ["person"])])
    calls = []
    embed = mem._text_embedder.query_embed
    mem._text_embedder.query_embed = lambda texts: calls.append(texts) or embed(texts)
    found = mem.recall("a bottle by the door")
    assert found and len(calls) == 1
    assert all(0.0 < frame["score"] <= 1.0 for frame in found), "the dashboard shows it"


def test_a_frame_added_during_the_scoring_is_scored_by_both_searches(monkeypatch):
    # The searches run without this memory's lock: a frame the scene writer
    # stores between the count and them would be scored by one and not the
    # other, so the frames are scored again as they are now.
    mem = _words_and_pictures(monkeypatch, [(0, ["bottle"]), (0, ["bottle"])])
    counts = []
    count = mem._store.count
    mem._store.count = lambda *a, **k: counts.append(1) or count(*a, **k)
    search = mem._store.nearest_ids
    added = []

    def writer_runs_meanwhile(*args, **kwargs):
        if not added:
            added.append(mem.remember(_frame(2), [{"label": "bottle"}]))
        return search(*args, **kwargs)

    mem._store.nearest_ids = writer_runs_meanwhile
    found = mem.recall("a bottle by the door")
    assert len(counts) == 4, "counted, changed, counted again and held"
    assert found[0]["labels"] == ["bottle"] and found[0]["found_by"] == "picture"
