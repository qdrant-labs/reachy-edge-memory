"""demo/people.py — recognising the person in front of the robot, and meeting
one it has not seen before."""
import pytest

from demo.people import (ASK_NAME, GREET_KNOWN, GREET_NEW, NOT_CAUGHT, People,
                         Seen)
from emulator.face_memory import Match as _Match   # the real one


class _Memory:
    """FaceMemory stand-in: answers with a scripted match, records enrolments."""

    def __init__(self, match=None):
        self._match = match or _Match(None, 0.0)
        self.enrolled = []

    def recognize(self, embedding):
        return self._match

    def enroll(self, name, embeddings):
        shots = list(embeddings)
        self.enrolled.append((name, len(shots)))
        return len(shots)


class _Reader:
    def __init__(self, faces=None, fails=False):
        self._faces = faces if faces is not None else [
            {"box": [0.3, 0.2, 0.6, 0.7], "score": 0.9, "embedding": [0.1] * 512}]
        self._fails = fails
        self.reads = 0

    def read(self, frame, embed=True):
        self.reads += 1
        if self._fails:
            raise OSError("the Mac is not answering")
        return list(self._faces)


FRAME = object()


def _model(heard):
    """The model on the laptop reading the name (demo/serve.py's /name), for
    the answers these tests give: the name said, or None."""
    return {"Sorry, what?": None, "I'm Sasha.": "Sasha"}.get(heard, heard)


# — recognising —

def test_a_known_face_is_named_and_greeted_once():
    people = People(_Memory(_Match("Sasha", 0.72)), _Reader(), read_name=_model)
    seen = people.observe(FRAME)
    assert seen.known and seen.name == "Sasha" and seen.box == [0.3, 0.2, 0.6, 0.7]
    assert people.greeting() == GREET_KNOWN.format(name="Sasha")
    # A robot that greets you on every turn is a robot with a glitch.
    people.observe(FRAME)
    assert people.greeting() is None


def test_a_stranger_is_not_named_and_is_asked_who_they_are():
    memory = _Memory(_Match(None, 0.05))
    people = People(memory, _Reader(), read_name=_model)
    seen = people.observe(FRAME)
    assert not seen.known and seen.box is not None
    assert people.greeting() is None
    assert people.should_ask_name()
    assert people.ask_name() == ASK_NAME
    assert people.awaiting_name
    assert not people.should_ask_name(), "asked already; do not ask twice"


def test_meeting_someone_stores_several_shots_under_their_name():
    memory = _Memory(_Match(None, 0.05))
    people = People(memory, _Reader(), shots=3, read_name=_model)
    for _ in range(5):
        people.observe(FRAME)          # more turns than shots wanted
    people.ask_name()
    name, line = people.answer_name("Sasha")
    assert name == "Sasha"
    assert line == GREET_NEW.format(name="Sasha")
    assert memory.enrolled == [("Sasha", 3)], "several poses, capped"
    assert not people.awaiting_name
    assert people.current.name == "Sasha"
    # Already met: no second greeting on the next turn.
    assert people.greeting() is None


def test_a_name_that_was_not_understood_is_not_enrolled():
    # A wrong name attached to a face outlives the mistake.
    memory = _Memory(_Match(None, 0.05))
    people = People(memory, _Reader(), read_name=_model)
    people.observe(FRAME)
    people.ask_name()
    name, line = people.answer_name("Sorry, what?")
    assert name is None and line == NOT_CAUGHT
    assert memory.enrolled == []
    assert not people.awaiting_name


def test_a_borderline_match_is_neither_greeted_nor_enrolled():
    # Between the thresholds: too close to call.
    memory = _Memory(_Match("Sasha", 0.30))
    people = People(memory, _Reader(), read_name=_model)
    people.observe(FRAME)
    assert people.greeting() is None
    assert not people.should_ask_name(), "no shots collected for an unsure face"


def test_a_frame_with_no_face_keeps_the_box_but_names_nobody():
    people = People(_Memory(), _Reader(faces=[]), read_name=_model)
    seen = people.observe(FRAME)
    assert seen == Seen(None, 0.0, None)
    assert not people.should_ask_name()


def test_a_face_service_that_fails_does_not_break_the_turn(capsys):
    people = People(_Memory(), _Reader(fails=True), read_name=_model)
    assert people.observe(FRAME) == Seen(None, 0.0, None)
    assert "[faces] skip" in capsys.readouterr().out


def test_without_a_memory_or_a_reader_nothing_happens():
    people = People()
    assert not people.enabled
    assert people.observe(FRAME) == Seen(None, 0.0, None)
    assert people.greeting() is None and not people.should_ask_name()


# — the same person, still in front of the robot —

def _clocked(memory, shots=5):
    now = [0.0]
    return People(memory, _Reader(), shots=shots, clock=lambda: now[0],
                  read_name=_model), now


def test_someone_just_met_is_not_asked_again_while_they_stay_in_view():
    # Seen live: met with two shots, two turns later the face scored as new and
    # the robot asked "What's your name?" again.
    memory = _Memory(_Match(None, 0.05))
    people, now = _clocked(memory)
    people.observe(FRAME)
    people.ask_name()
    people.answer_name("I'm Sasha.")
    memory._match = _Match("Sasha", 0.30)    # the same face, under the line
    for t in (2.0, 4.0, 6.0):             # the detect loop keeps seeing a face
        now[0] = t
        people.face_seen()
    now[0] = 8.0
    seen = people.observe(FRAME)
    assert seen.name == "Sasha"
    assert not people.should_ask_name()
    assert memory.enrolled[-1] == ("Sasha", 1), "the missed pose is learned"


def test_a_face_that_was_gone_a_while_is_looked_at_afresh():
    memory = _Memory(_Match(None, 0.05))
    people, now = _clocked(memory)
    people.observe(FRAME)
    people.ask_name()
    people.answer_name("Sasha")
    now[0] = 30.0                          # nobody in view for 30 s
    seen = people.observe(FRAME)
    assert seen.name is None
    assert people.should_ask_name()


def test_a_recognised_person_is_kept_through_a_bad_angle():
    memory = _Memory(_Match("Sasha", 0.7))
    people, now = _clocked(memory)
    assert people.observe(FRAME).name == "Sasha"
    memory._match = _Match("Sasha", 0.30)     # turned their head: close, under the line
    now[0] = 3.0
    assert people.observe(FRAME).name == "Sasha"


def test_a_stranger_is_said_to_arrive_once_not_on_every_turn():
    memory = _Memory(_Match(None, 0.05))
    people, now = _clocked(memory)
    assert people.observe(FRAME).stranger_arrived
    now[0] = 1.0
    assert not people.observe(FRAME).stranger_arrived, "still the same stranger"
    memory._match = _Match("Sasha", 0.7)
    now[0] = 2.0
    assert not people.observe(FRAME).stranger_arrived
    memory._match = _Match(None, 0.05)
    now[0] = 3.0
    assert people.observe(FRAME).stranger_arrived, "after Sasha, someone new again"
    memory._match = _Match(None, 0.30)   # too close to call: someone unnamed too
    people, now = _clocked(memory)
    assert people.observe(FRAME).stranger_arrived
    now[0] = 1.0
    assert not people.observe(FRAME).stranger_arrived


def test_after_an_empty_view_the_next_stranger_is_someone_new():
    # Live, the detect loop calls face_seen several times a second: the gap
    # a stranger leaves is seen there, not by observe. A second stranger after
    # an empty room is a new arrival, and the first one's face is not theirs.
    memory = _Memory(_Match(None, 0.05))
    people, now = _clocked(memory)
    assert people.observe(FRAME).stranger_arrived
    now[0] = 20.0                       # the room was empty
    people.face_seen()                  # the detect loop sees a face again
    assert people.observe(FRAME).stranger_arrived
    people.ask_name()
    people.answer_name("Carol")
    assert memory.enrolled == [("Carol", 1)], "only Carol's own shot"


def test_after_an_empty_view_an_unrecognised_face_is_not_the_last_person():
    # Sasha leaves; the room is empty; someone steps in whom the camera cannot
    # place — too close to call, no usable face, or no picture at all. None of
    # them is Sasha, and each is someone new to the conversation.
    for unreadable in ("too close to call", "no embedding", "no picture"):
        people, memory = _met_sasha_clocked()
        people._clock = lambda: 30.0
        people.face_seen()                       # the detect loop: a face again
        if unreadable == "too close to call":
            memory._match = _Match("Sasha", 0.30)
            seen = people.observe(FRAME)
        elif unreadable == "no embedding":
            people._reader = _Reader(faces=[{"box": [0, 0, 1, 1], "embedding": None}])
            seen = people.observe(FRAME)
        else:
            seen = people.observe(None)
        assert seen.name is None, unreadable
        assert seen.stranger_arrived, unreadable


def test_a_name_question_asked_of_someone_who_left_is_dropped():
    memory = _Memory(_Match(None, 0.05))
    people, now = _clocked(memory)
    people.observe(FRAME)
    people.ask_name()
    now[0] = 30.0
    people.face_seen()
    assert not people.awaiting_name


def test_an_empty_view_is_noticed_on_a_turn_that_sees_no_face():
    # Bob was asked his name and walked off; 15 s later someone says "I'm
    # Carol" from outside the picture. No face came back to notice the empty
    # view: Bob's question is gone all the same, and his face shots with it.
    memory = _Memory(_Match(None, 0.05))
    people, now = _clocked(memory)
    people.observe(FRAME)
    people.ask_name()
    now[0] = 15.0
    people._reader = _Reader(faces=[])
    seen = people.observe(FRAME)
    assert not people.awaiting_name
    assert seen.name is None and seen.stranger_arrived
    assert not people.observe(None).stranger_arrived, "noticed once"

    # Then Dave steps into the picture: he is not the voice from outside it.
    now[0] = 25.0
    people._reader = _Reader()
    people.face_seen()
    assert people.observe(FRAME).stranger_arrived


def test_a_look_away_is_not_an_empty_view():
    # The robot turned its head for a long reply: the stranger it was asking
    # did not leave, and their face shots and the question still stand.
    memory = _Memory(_Match(None, 0.05))
    people, now = _clocked(memory)
    people.observe(FRAME)
    people.ask_name()
    now[0] = 1.0
    people.look_away()
    now[0] = 30.0
    people.look_back()
    people.face_seen()
    assert people.awaiting_name
    assert not people.observe(FRAME).stranger_arrived


def test_a_person_who_left_before_the_head_turned_stays_gone():
    # Sasha walked off; later the robot looked away and back. Coming back
    # must not bring her into view again for the next face to inherit.
    people, memory = _met_sasha_clocked()
    people._clock = lambda: 20.0
    people.look_away()
    people._clock = lambda: 30.0
    people.look_back()
    people.face_seen()                         # someone steps in
    memory._match = _Match(None, 0.30)
    seen = people.observe(FRAME)
    assert seen.name is None and seen.stranger_arrived


def test_a_turn_with_no_picture_is_not_another_arrival():
    people = People(_Memory(_Match(None, 0.05)), _Reader(), read_name=_model)
    assert people.observe(FRAME).stranger_arrived
    assert not people.observe(None).stranger_arrived
    people._reader = _Reader(fails=True)
    assert not people.observe(FRAME).stranger_arrived


def test_a_stranger_in_front_is_not_given_the_last_name_on_a_later_frame():
    # Sasha, then a clear stranger: a frame where the face cannot be matched,
    # or one too close to call, must not bring Sasha's name back.
    people, memory = _met_sasha_clocked()
    memory._match = _Match(None, 0.05)
    assert people.observe(FRAME).name is None
    memory._match = _Match(None, 0.30)
    assert people.observe(FRAME).name is None
    people._reader = _Reader(faces=[{"box": [0, 0, 1, 1], "embedding": None}])
    assert people.observe(FRAME).name is None


def _met_sasha_clocked():
    memory = _Memory(_Match(None, 0.05))
    people, now = _clocked(memory)
    people.observe(FRAME)
    people.ask_name()
    people.answer_name("Sasha")
    now[0] = 3.0
    people.face_seen()
    return people, memory


def test_learning_poses_is_capped():
    from demo.people import MAX_LEARNED_SHOTS

    memory = _Memory(_Match(None, 0.05))
    people, now = _clocked(memory)
    people.observe(FRAME)
    people.ask_name()
    people.answer_name("Sasha")
    memory._match = _Match("Sasha", 0.30)    # the same face, at a worse angle
    for turn in range(MAX_LEARNED_SHOTS + 5):
        now[0] = float(turn)
        people.observe(FRAME)
    learned = sum(count for name, count in memory.enrolled[1:])
    assert learned == MAX_LEARNED_SHOTS


def test_the_cap_on_learning_poses_is_per_person():
    # A demo has visitors: the first one standing in bad light used up the
    # whole run's cap, and nobody after them had a pose learned. The cap is
    # each person's for the run — coming back does not start it again.
    from demo.people import MAX_LEARNED_SHOTS

    memory = _Memory(_Match(None, 0.05))
    people, now = _clocked(memory)

    def at_bad_angles(name, start, turns):
        memory._match = _Match(name, 0.30)   # the same face, at a worse angle
        for turn in range(turns):
            now[0] = start + turn
            people.observe(FRAME)

    # Sasha's first visit learns only 3: under the cap, so her return must
    # carry on from there — neither start over nor stop.
    for name, start, turns in (("Sasha", 0.0, 3),
                               ("Robin", 1000.0, MAX_LEARNED_SHOTS + 5)):
        now[0] = start                       # far apart: a new person
        memory._match = _Match(None, 0.05)   # someone new
        people.observe(FRAME)
        people.ask_name()
        people.answer_name(name)
        at_bad_angles(name, start, turns)
    now[0] = 2000.0
    memory._match = _Match("Sasha", 0.7)     # Sasha comes back, recognised
    people.observe(FRAME)
    at_bad_angles("Sasha", 2000.0, MAX_LEARNED_SHOTS + 5)

    learned = {}
    for name, count in memory.enrolled:
        learned.setdefault(name, []).append(count)
    assert sum(learned["Sasha"][1:]) == MAX_LEARNED_SHOTS
    assert sum(learned["Robin"][1:]) == MAX_LEARNED_SHOTS


# — who is in a frame —

def test_in_frame_names_known_faces_and_leaves_strangers_unnamed():
    class _Two(_Reader):
        def read(self, frame, embed=True):
            return [{"box": [0, 0, 1, 1], "embedding": [0.1]},
                    {"box": [0.5, 0, 1, 1], "embedding": [0.2]}]

    class _ByVector(_Memory):
        def recognize(self, embedding):
            return _Match("Sasha", 0.7) if embedding == [0.1] else _Match("Sasha", 0.1)

        def people(self):
            return ["Sasha"]

    people = People(_ByVector(), _Two(), read_name=_model)
    found = people.in_frame(FRAME)
    assert [p["name"] for p in found] == ["Sasha", None]
    assert people.met() == ["Sasha"]
    assert memory_untouched(people)


def memory_untouched(people):
    return people._memory.enrolled == [] and not people.awaiting_name


def _met_sasha():
    memory = _Memory(_Match(None, 0.05))
    people, now = _clocked(memory)
    people.observe(FRAME)
    people.ask_name()
    people.answer_name("Sasha")
    now[0] = 3.0
    people.face_seen()
    return people, memory


def test_in_frame_keeps_the_tracked_person_named_at_a_bad_angle():
    people, memory = _met_sasha()
    memory._match = _Match("Sasha", 0.30)   # under the line, not someone else
    assert [p["name"] for p in people.in_frame(FRAME)] == ["Sasha"]


def test_in_frame_does_not_give_a_clear_stranger_the_last_persons_name():
    # Sasha steps away, someone else steps in within the 10 s: observe()
    # already refuses them Sasha's name; the frame stored with them must too.
    # FaceMemory names the nearest person whatever the score — as here.
    people, memory = _met_sasha()
    memory._match = _Match("Sasha", 0.05)
    assert [p["name"] for p in people.in_frame(FRAME)] == [None]


def test_someone_else_the_robot_has_met_is_not_the_tracked_person_at_an_angle():
    # Bob, met before, steps in at a bad angle while Sasha was being tracked:
    # nearest to Bob, under the line. Not Sasha — and his face is not learned
    # into Sasha's point.
    people, memory = _met_sasha()
    memory._match = _Match("Bob", 0.30)
    assert [p["name"] for p in people.in_frame(FRAME)] == [None]
    assert people.observe(FRAME).name is None
    assert [name for name, _count in memory.enrolled] == ["Sasha"], "nothing learned"
    # Nor does Sasha's name come back to that face a moment later.
    memory._match = _Match("Sasha", 0.30)
    assert people.observe(FRAME).name is None


def test_a_frame_between_turns_ends_the_tracking_too():
    # Bob is seen first in a stored frame (the scene writer, between turns):
    # his next frame, nearest to Sasha, must not be named or learned as her.
    people, memory = _met_sasha()
    memory._match = _Match("Sasha", 0.05)            # Bob, clearly not Sasha
    assert [p["name"] for p in people.in_frame(FRAME)] == [None]
    memory._match = _Match("Sasha", 0.30)
    assert [p["name"] for p in people.in_frame(FRAME)] == [None]
    assert people.observe(FRAME).name is None
    assert [name for name, _count in memory.enrolled] == ["Sasha"], "nothing learned"


def test_in_frame_without_faces_or_models_is_empty():
    assert People().in_frame(FRAME) == []
    assert People(_Memory(), _Reader(fails=True)).in_frame(FRAME) == []


def test_faces_in_says_when_the_faces_could_not_be_read():
    # in_frame keeps a frame without names; the `who` answer has to tell
    # "nobody is there" from "I cannot tell who is there".
    with pytest.raises(OSError):
        People(_Memory(), _Reader(fails=True)).faces_in(FRAME)
    assert People().faces_in(FRAME) == []


def test_a_clear_stranger_is_never_given_the_name_of_the_person_in_view():
    # Live: a second person in front of the robot was called Sasha,
    # and their face was learned into Sasha's point.
    memory = _Memory(_Match(None, 0.02))          # nothing like anyone known
    people, now = _clocked(memory)
    people.observe(FRAME)
    people.ask_name()
    people.answer_name("Sasha")
    memory._match = _Match("Sasha", 0.02)          # nearest is Sasha, far off
    enrolled_before = len(memory.enrolled)
    now[0] = 2.0
    people.face_seen()
    seen = people.observe(FRAME)
    assert seen.name is None, "a stranger keeps no name"
    assert len(memory.enrolled) == enrolled_before, "and teaches the robot nothing"


def test_the_same_person_at_a_bad_angle_keeps_their_name():
    memory = _Memory(_Match(None, 0.05))
    people, now = _clocked(memory)
    people.observe(FRAME)
    people.ask_name()
    people.answer_name("Sasha")
    memory._match = _Match("Sasha", 0.30)          # between the two thresholds
    now[0] = 2.0
    people.face_seen()
    assert people.observe(FRAME).name == "Sasha"


def test_the_name_is_read_by_the_model():
    memory = _Memory(_Match(None, 0.05))
    people = People(memory, _Reader(), read_name=lambda heard: "Robin")
    people.observe(FRAME)
    people.ask_name()
    name, line = people.answer_name("they call me Robin")
    assert name == "Robin" and memory.enrolled[-1][0] == "Robin"


def test_the_models_no_name_is_an_answer_not_a_reason_to_guess():
    # The model read "What's yours?" as no name; a pattern over the words
    # took "What's" for one.
    memory = _Memory(_Match(None, 0.05))
    people = People(memory, _Reader(), read_name=lambda heard: None)
    people.observe(FRAME)
    people.ask_name()
    assert people.answer_name("What's yours?") == (None, NOT_CAUGHT)
    assert memory.enrolled == []


def test_a_name_the_model_cannot_be_asked_about_is_not_caught():
    # No pattern over the words stands in for the model: nobody is enrolled
    # under a guess, and the robot asks again after its next answer.
    def unreachable(heard):
        raise OSError("laptop gone")

    memory = _Memory(_Match(None, 0.05))
    people = People(memory, _Reader(), read_name=unreachable)
    people.observe(FRAME)
    people.ask_name()
    assert people.answer_name("I'm Sasha.") == (None, NOT_CAUGHT)
    assert memory.enrolled == []
    assert people.should_ask_name()


def test_a_bug_in_the_name_reader_is_not_taken_for_a_name_not_caught():
    # Only the laptop not answering is "not caught"; a bug reaches the turn's
    # handler with its traceback instead of hiding behind the same line.
    def broken(heard):
        raise TypeError("a bug")

    people = People(_Memory(_Match(None, 0.05)), _Reader(), read_name=broken)
    people.observe(FRAME)
    people.ask_name()
    with pytest.raises(TypeError):
        people.answer_name("Sasha")


def test_a_pose_is_learned_under_the_name_it_was_judged_by():
    # The scene writer's thread can end the tracking right after observe()
    # judged the face to be the tracked person: the pose goes to the person
    # it was judged to be, never to a nameless point.
    people, memory = _met_sasha()
    memory._match = _Match("Sasha", 0.30)
    judged = people._tracked_at_a_bad_angle

    def judged_then_ended(match, here):
        result = judged(match, here)
        people._here = None                  # in_frame, on the other thread
        return result

    people._tracked_at_a_bad_angle = judged_then_ended
    assert people.observe(FRAME).name == "Sasha"
    assert [name for name, _count in memory.enrolled] == ["Sasha", "Sasha"]
