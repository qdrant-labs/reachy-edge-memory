"""Where each model runs: parsing the option, what each placement builds,
and the lines the start prints.

Pure functions over an args-like object, so no model is loaded here.
"""

from types import SimpleNamespace

import pytest

from demo import placement
from tests.test_run_demo import DisplaySpy


def args(**kwargs):
    return SimpleNamespace(**kwargs)


# --- parsing the list ---

def test_no_option_given_names_nothing():
    assert placement.parse(None) == set()
    assert placement.parse("") == set()


def test_a_list_is_split_and_normalized():
    assert placement.parse(" ASR , tts,detector ") == {"asr", "tts", "detector"}


def test_an_unknown_family_stops_the_start_with_the_choices():
    with pytest.raises(SystemExit) as exc:
        placement.parse("asr,vision")
    assert "vision" in str(exc.value)
    assert "detector" in str(exc.value)


def test_asking_for_the_language_model_says_why_it_cannot_move():
    with pytest.raises(SystemExit) as exc:
        placement.parse("asr,llm")
    message = str(exc.value)
    assert "language model" in message
    assert "2.5 GB" in message


# --- the default: everything on the Mac ---

def test_nothing_given_puts_every_model_on_the_mac():
    places = placement.resolve(args())
    assert set(places.values()) == {placement.MAC}
    assert set(places) == set(placement.FAMILIES)


def test_on_robot_answers_per_family():
    a = args(on_robot="detector,faces")
    assert placement.on_robot(a, "detector") is True
    assert placement.on_robot(a, "asr") is False


def test_families_on_robot_lists_them_in_report_order():
    assert placement.families_on_robot(args(on_robot="faces,asr")) == ("asr", "faces")


# --- the start report ---

def test_the_report_names_every_family_and_where_it_runs():
    lines = placement.report(args(on_robot="asr"))
    assert len(lines) == len(placement.FAMILIES)
    assert any("speech recognition: robot" in line for line in lines)
    assert any("object detector: Mac" in line for line in lines)


def test_the_report_carries_the_model_name_and_the_address():
    # Recognition is a different model on each side, so "robot" alone would
    # hide what actually transcribed.
    lines = placement.report(
        args(on_robot="asr"),
        models={"asr": "moonshine-tiny", "tts": "inflect-nano-v2"},
        addresses={"tts": "mac.local:9500"})
    assert any("speech recognition: robot (moonshine-tiny)" in line
               for line in lines)
    assert any("speech synthesis: Mac (inflect-nano-v2) at mac.local:9500"
               in line for line in lines)


def test_a_missing_local_model_names_the_family_placement_and_cause():
    exc = placement.missing("detector", "yolo26n_conv2d_f16_weights.tflite",
                            "FileNotFoundError")
    message = str(exc)
    assert "object detector" in message
    assert "on the robot" in message
    assert "yolo26n_conv2d_f16_weights.tflite" in message
    assert "FileNotFoundError" in message


# --- a local family fails the start; a remote one only turns itself off ---

def test_memory_that_cannot_open_stops_a_start_that_placed_it_on_the_robot(tmp_path):
    from demo import run_demo

    # A path that cannot become a shard: a plain file where a directory
    # belongs, so EdgeStore raises inside build_memories.
    blocked = tmp_path / "memory"
    blocked.write_text("not a directory")
    with pytest.raises(SystemExit) as exc:
        run_demo.build_memories(args(no_memory=False, memory_dir=str(tmp_path),
                                     on_robot="embedder"))
    message = str(exc.value)
    assert "memory embeddings" in message and "on the robot" in message


def test_the_same_failure_only_disables_memory_when_it_is_on_the_mac(tmp_path, capsys):
    from demo import run_demo

    blocked = tmp_path / "memory"
    blocked.write_text("not a directory")
    frame_memory, speech_memory = run_demo.build_memories(
        args(no_memory=False, memory_dir=str(tmp_path)))
    assert frame_memory is None and speech_memory is None
    assert "memory disabled" in capsys.readouterr().out


# --- what a placement actually builds ---

def test_faces_on_the_robot_are_read_the_way_the_laptops_are(monkeypatch, tmp_path):
    # demo/people.py reads `{box, score, embedding}` dicts, the laptop's
    # answer; the local FaceReader returns Face dataclasses. Handed over as
    # they were, every turn with a face in view raised AttributeError.
    import numpy as np

    from demo import run_demo
    from emulator import face as face_module

    class OneFace:
        def __init__(self, *args, **kwargs):
            pass

        def read(self, frame, *, embed=True):
            return [face_module.Face(box=[0.4, 0.3, 0.6, 0.7], score=0.9,
                                     embedding=[1.0] + [0.0] * 511 if embed else None)]

    monkeypatch.setattr(face_module, "FaceReader", OneFace)
    people = run_demo.build_people(args(on_robot="faces", memory_dir=str(tmp_path),
                                        brain="127.0.0.1", port=9500))
    frame = np.zeros((8, 8, 3), np.uint8)
    assert people.observe(frame).box == [0.4, 0.3, 0.6, 0.7]
    assert people.in_frame(frame) is not None
    people.close()


def test_the_name_is_read_by_the_laptops_model(monkeypatch, tmp_path):
    # The only way a name is read now: nothing falls back to a pattern, so a
    # People built without the reader would enroll nobody, ever.
    import numpy as np

    from demo import run_demo
    from demo.contract import NAME_PATH
    from emulator import face as face_module

    class OneFace:
        def __init__(self, *args, **kwargs):
            pass

        def read(self, frame, *, embed=True):
            return [face_module.Face(box=[0.4, 0.3, 0.6, 0.7], score=0.9,
                                     embedding=[1.0] + [0.0] * 511 if embed else None)]

    asked = []
    monkeypatch.setattr(face_module, "FaceReader", OneFace)
    monkeypatch.setattr(run_demo, "_http_post",
                        lambda endpoint, payload: asked.append((endpoint, payload))
                        or {"name": "Robin"})
    people = run_demo.build_people(args(on_robot="faces", memory_dir=str(tmp_path),
                                        brain="127.0.0.1", port=9500))
    people.observe(np.zeros((8, 8, 3), np.uint8))
    people.ask_name()
    assert people.answer_name("they call me Robin")[0] == "Robin"
    assert asked == [("http://127.0.0.1:9500" + NAME_PATH,
                      {"text": "they call me Robin"})]
    people.close()


def test_faces_the_laptop_does_not_have_are_off_on_the_robot_too(monkeypatch, tmp_path, capsys):
    # The embed service says so in its health; asking it every frame instead
    # would only collect 503s.
    from demo import embed_client, run_demo

    monkeypatch.setattr(embed_client, "faces_off",
                        lambda host, port: "FileNotFoundError: hsface10k.tflite is missing")
    monkeypatch.setattr(embed_client, "RemoteFaceReader",
                        lambda *a, **k: pytest.fail("faces asked of a laptop without them"))
    people = run_demo.build_people(args(memory_dir=str(tmp_path), brain="127.0.0.1"))
    assert "faces disabled" in capsys.readouterr().out
    assert not people.enabled
    people.close()


def test_a_family_this_run_turned_off_is_reported_off():
    lines = placement.report(args(), models={"faces": "YuNet + HSFace"},
                             addresses={"faces": "mac:9900"}, off=("faces",))
    assert "  face models: off" in lines
    assert not any("YuNet" in line for line in lines)


def test_the_detector_source_is_the_mac_service_by_default():
    from demo.detect_source import RemoteDetectSource
    from demo.run_demo import build_detect_source

    source = build_detect_source(args(), camera=None,
                                 detect_url="http://mac:9600/detect")
    assert isinstance(source, RemoteDetectSource)
    assert source.detect_url == "http://mac:9600/detect"


def test_a_detector_placed_on_the_robot_that_cannot_load_stops_the_start(monkeypatch):
    from demo import run_demo

    class Broken:
        def __init__(self, *a, **kw):
            raise FileNotFoundError("yolo26n_conv2d_f16_weights.tflite")

    monkeypatch.setattr("emulator.detector.Detector", Broken)
    monkeypatch.setattr("emulator.models.fetch", lambda name: name)
    with pytest.raises(SystemExit) as exc:
        run_demo.build_detect_source(args(on_robot="detector"), camera=None,
                                     detect_url="http://mac:9600/detect")
    message = str(exc.value)
    assert "object detector" in message and "on the robot" in message
    assert "yolo26n_conv2d_f16_weights.tflite" in message


def test_a_local_detector_with_the_faces_on_the_laptop_still_follows_faces(monkeypatch):
    # With the detector here and the face models on the laptop, the boxes
    # the head follows come from the laptop — otherwise nothing finds them
    # and the head stops following anyone.
    from demo import run_demo

    monkeypatch.setattr("emulator.detector.Detector", lambda path, threads=4: object())
    monkeypatch.setattr("emulator.models.fetch", lambda name: name)

    class Remote:
        def __init__(self, host, port):
            self.host = host

        def read(self, frame, embed=True):
            assert embed is False, "the head needs boxes, not identities"
            return [{"box": [0.1, 0.1, 0.3, 0.3], "score": 0.9, "embedding": None}]

    monkeypatch.setattr("demo.embed_client.RemoteFaceReader", Remote)
    source = run_demo.build_detect_source(args(on_robot="asr,detector", brain="mac"),
                                          camera=None, detect_url="http://mac/detect")
    assert source._faces("frame") == [{"box": [0.1, 0.1, 0.3, 0.3], "score": 0.9}]


def test_the_local_detector_runs_on_two_threads(monkeypatch):
    # Two of the robot's four cores: the other two go to the voice loop and
    # the Pollen daemon.
    from demo import run_demo

    captured = {}

    class FakeDetector:
        def __init__(self, path, threads=4):
            captured["threads"] = threads

    monkeypatch.setattr("emulator.detector.Detector", FakeDetector)
    monkeypatch.setattr("emulator.models.fetch", lambda name: name)
    run_demo.build_detect_source(args(on_robot="detector", no_faces=True),
                                 camera=None, detect_url="http://mac/detect")
    assert captured["threads"] == run_demo.LOCAL_DETECTOR_THREADS == 2


def test_no_transcriber_is_built_when_recognition_stays_on_the_mac():
    from demo.run_demo import build_transcriber

    assert build_transcriber(args()) is None


def test_a_recognizer_placed_on_the_robot_that_cannot_load_stops_the_start(monkeypatch):
    from demo import run_demo

    def broken(kind, model=None):
        raise FileNotFoundError("moonshine_tiny_5s_f32.tflite")

    monkeypatch.setattr("emulator.speech.build_recognizer", broken)
    with pytest.raises(SystemExit) as exc:
        run_demo.build_transcriber(args(on_robot="asr"))
    message = str(exc.value)
    assert "speech recognition" in message and "on the robot" in message
    assert "moonshine" in message


def test_the_turn_transcribes_locally_when_one_is_given(monkeypatch):
    from demo import run_demo

    def must_not_be_called(*a, **kw):
        raise AssertionError("the Mac's /transcribe must not be called")

    monkeypatch.setattr(run_demo, "transcribe", must_not_be_called)
    # Nothing follows an empty transcript: the turn short-circuits, which is
    # enough to prove which recognizer was asked.
    run_demo._handle_stream(
        [], audio=None, endpoint="http://mac:9500", robot=None,
        display=DisplaySpy(), transcriber=lambda audio, rate: "")


# --- synthesis on the robot ---

class FakeSynthesizer:
    sample_rate = 24000

    def __init__(self):
        self.said = []

    def speak(self, text):
        import numpy as np

        self.said.append(text)
        return np.linspace(-0.1, 0.1, 480, dtype=np.float32)


class _StillRobot:
    """Only what a turn's meta event asks of a robot."""

    def look_at(self, yaw, pitch):
        pass


class FakePlayer:
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
        self.fed = []

    def feed(self, audio):
        self.fed.append(len(audio))

    def close(self):
        pass


def test_sentences_are_spoken_here_in_order_when_synthesis_is_local():
    from demo.run_demo import drive_robot_stream

    synth = FakeSynthesizer()
    players = []

    def make_player(sample_rate):
        players.append(FakePlayer(sample_rate))
        return players[-1]

    said = []
    events = [{"type": "sentence", "text": "Hello Sasha."},
              {"type": "sentence", "text": "Nice to meet you."},
              {"type": "done", "reply": "Hello Sasha. Nice to meet you."}]
    done = drive_robot_stream(None, iter(events), make_player=make_player,
                              on_reply=lambda text, done: said.append(text),
                              synthesizer=synth)
    assert synth.said == ["Hello Sasha.", "Nice to meet you."]
    assert players and players[0].sample_rate == 24000
    assert len(players[0].fed) == 2, "both sentences reach the one live player"
    assert done["reply"] == "Hello Sasha. Nice to meet you."
    assert said[0] == "Hello Sasha.", "the screen updates per sentence, as with audio"


def test_the_fixed_lines_are_spoken_here_too(monkeypatch):
    # The greeting and the name question go to the Mac's /say today; a run
    # that claims only the reply leaves the robot has to cover them as well.
    from demo import run_demo

    def must_not_be_called(*a, **kw):
        raise AssertionError("/say must not be called with synthesis local")

    monkeypatch.setattr(run_demo, "_http_post", must_not_be_called)
    synth = FakeSynthesizer()
    run_demo._say("Hi, I'm Reachy.", "http://mac:9500", None,
                  DisplaySpy(), make_player=lambda rate: FakePlayer(rate),
                  synthesizer=synth)
    assert synth.said == ["Hi, I'm Reachy."]


def test_no_synthesizer_is_built_when_the_voice_stays_on_the_mac():
    from demo.run_demo import build_synthesizer

    assert build_synthesizer(args()) is None


def test_a_voice_placed_on_the_robot_that_cannot_load_stops_the_start(monkeypatch):
    from demo import run_demo

    def broken():
        raise FileNotFoundError("inflect_decoder.tflite")

    monkeypatch.setattr("emulator.speech.build_synthesizer", broken)
    with pytest.raises(SystemExit) as exc:
        run_demo.build_synthesizer(args(on_robot="tts"))
    message = str(exc.value)
    assert "speech synthesis" in message and "on the robot" in message
    assert "inflect_decoder.tflite" in message


def test_the_chat_request_asks_for_words_only_when_synthesis_is_local(monkeypatch):
    from demo import run_demo

    sent = []

    def fake_stream(endpoint, payload):
        sent.append(payload)
        yield {"type": "done", "reply": "Hi.", "token_count": 10}

    monkeypatch.setattr(run_demo, "_http_stream", fake_stream)
    monkeypatch.setattr(run_demo, "transcribe", lambda *a, **kw: "hello")

    run_demo._handle_stream([], audio=None, endpoint="http://mac:9500",
                            robot=_StillRobot(), display=DisplaySpy(),
                            make_player=lambda rate: FakePlayer(rate),
                            synthesizer=FakeSynthesizer())
    assert sent and sent[0].get("sentences") is True

    sent.clear()
    run_demo._handle_stream([], audio=None, endpoint="http://mac:9500",
                            robot=_StillRobot(), display=DisplaySpy(),
                            make_player=lambda rate: FakePlayer(rate))
    assert sent and "sentences" not in sent[0], \
        "a run with the voice on the Mac sends exactly what it sends today"
