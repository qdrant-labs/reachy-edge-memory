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

    def embed_image(self, frame_rgb):
        idx = int(np.asarray(frame_rgb)[0, 0, 0])
        return np.asarray(self._BASIS.get(idx, [0.0, 0.0, 0.0]), np.float32)


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
