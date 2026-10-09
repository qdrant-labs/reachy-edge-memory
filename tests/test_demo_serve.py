import http.client
import json
import logging
import threading
from http.server import HTTPServer
from types import SimpleNamespace

import numpy as np
import pytest

from demo.contract import (
    ChatRequest,
    decode_transcribe_response,
    encode_chat_request,
    encode_transcribe_request,
)
from demo.serve import (
    MAX_BODY,
    BadRequest,
    Models,
    chat_respond,
    handle_name,
    handle_say,
    handle_transcribe,
    make_handler,
)


class _Recognizer:
    def __init__(self, heard="hello"):
        self._heard = heard

    def transcribe(self, pcm):
        return self._heard


class _Synth:
    sample_rate = 16000

    def speak(self, text):
        return np.linspace(-0.1, 0.1, 800, dtype=np.float32)


class _Llm:
    def __init__(self, said="Hi there."):
        self.said = said

    def reply_stream(self, prompt, system, tools=None, image=None):
        yield {"type": "content", "text": self.said}


def _stack(heard="hello", said="Hi there."):
    return Models(recognizer=_Recognizer(heard), synthesizer=_Synth(),
                  llm=_Llm(said))


class _SessionStub:
    """ChatSession's surface as chat_respond uses it."""

    def __init__(self, events, token_count=321):
        self._events = list(events)
        self.finished = []
        self.requests = []
        self.token_count = token_count

    def turn(self, request):
        self.requests.append(request)
        yield from self._events

    def finish(self, request, reply, tool_call):
        self.finished.append((request.text, reply, tool_call))


class _RunningServer:
    """A single-threaded HTTPServer, as in serve.main(), on a background
    thread — tested through real HTTP requests."""

    def __init__(self, stack=None, session=None):
        self.server = HTTPServer(("127.0.0.1", 0), make_handler(
            stack or _stack(), session or _SessionStub([])))
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def post(self, path, body=None, raw=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.server.server_port,
                                          timeout=10)
        data = raw if raw is not None else json.dumps(body).encode()
        conn.request("POST", path, body=data, headers=headers or {})
        response = conn.getresponse()
        content = response.read()
        conn.close()
        return response.status, content

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def server():
    servers = []

    def make(stack=None, session=None):
        servers.append(_RunningServer(stack, session))
        return servers[-1]

    yield make
    for srv in servers:
        srv.close()


# --- the handlers ---

def test_handle_transcribe_returns_heard_text():
    payload = encode_transcribe_request(np.zeros(8000, np.float32), 16000)
    body = handle_transcribe(payload, _stack(heard="what time is it"))
    assert decode_transcribe_response(body) == "what time is it"


def test_handle_say_speaks_the_line_in_the_voice():
    event = handle_say({"text": "Hello."}, _stack())
    assert event["type"] == "audio" and event["text"] == "Hello."
    assert event["sample_rate"] == 16000 and event["audio_b64"]


def test_handle_say_needs_a_line():
    with pytest.raises(BadRequest):
        handle_say({"text": "  "}, _stack())


def test_handle_name_reads_a_name_and_refuses_a_sentence():
    assert handle_name({"text": "they call me sasha"},
                       _stack(said="sasha")) == {"name": "Sasha"}
    assert handle_name({"text": "what's yours?"},
                       _stack(said="NONE")) == {"name": None}
    assert handle_name({"text": "tell me a joke"}, _stack(
        said="Why did the robot cross the road?")) == {"name": None}
    with pytest.raises(BadRequest):
        handle_name({"text": "  "}, _stack())


def test_strip_tool_call_leak_removes_leading_garbage():
    from demo.serve import _strip_tool_call_leak
    text = ('<|tool_call>call:move{how:<|"|>shake<|"|>} '
            "Yes, I am shaking my head!")
    assert _strip_tool_call_leak(text) == "Yes, I am shaking my head!"
    assert _strip_tool_call_leak("Hello there!") == "Hello there!"


# --- HTTP: what is the caller's fault, and what is ours ---

def test_an_empty_or_oversized_body_is_refused(server):
    srv = server()
    assert srv.post("/transcribe", raw=b"")[0] == 413
    assert srv.post("/transcribe", raw=b"",
                    headers={"Content-Length": str(MAX_BODY + 1)})[0] == 413


def test_an_unknown_path_is_404(server):
    assert server().post("/respond", {"text": "hi"})[0] == 404


def test_a_body_that_is_not_an_object_is_400(server):
    assert server().post("/say", ["hi"])[0] == 400


def test_transcribe_rejects_bad_or_missing_audio(server):
    srv = server()
    assert srv.post("/transcribe", {"audio_b64": "not-valid-base64!!!",
                                    "sample_rate": 16000})[0] == 400
    assert srv.post("/transcribe", {"sample_rate": 16000})[0] == 400


def test_transcribe_answers_over_http(server):
    payload = encode_transcribe_request(np.zeros(8000, np.float32), 16000)
    status, body = server(_stack(heard="what time is it")).post("/transcribe", payload)
    assert status == 200
    assert decode_transcribe_response(json.loads(body)) == "what time is it"


def test_a_model_that_fails_is_a_500_not_the_callers_fault(server):
    # A ValueError from inside a model is the service's failure: answering
    # 400 would tell the robot its request was wrong, and hide the traceback.
    class Broken:
        def transcribe(self, pcm):
            raise ValueError("the model tripped over itself")

    stack = _stack()
    stack.recognizer = Broken()
    payload = encode_transcribe_request(np.zeros(8000, np.float32), 16000)
    assert server(stack).post("/transcribe", payload)[0] == 500


# --- /chat ---

def test_chat_respond_speaks_and_reports_the_context():
    session = _SessionStub([{"type": "content", "text": "Hello Sasha, nice to meet you."}])
    events = list(chat_respond(ChatRequest([], "Hi, I'm Sasha."), session, _stack()))
    kinds = [e["type"] for e in events]
    assert kinds[-1] == "done" and set(kinds[:-1]) == {"audio"}
    assert events[-1]["reply"] == "Hello Sasha, nice to meet you."
    assert events[-1]["token_count"] == 321
    assert events[-1]["first_sound_ms"] is not None
    assert session.finished == [("Hi, I'm Sasha.", "Hello Sasha, nice to meet you.", None)]


def test_chat_respond_hands_a_tool_call_back_without_speaking():
    call = {"name": "remember", "arguments": {"query": "name"}}
    session = _SessionStub([{"type": "tool_call", **call}])
    events = list(chat_respond(ChatRequest([], "What's my name?"), session, _stack()))
    assert events == [
        {"type": "tool_call", **call},
        {"type": "done", "reply": "", "tool_call": call, "first_sound_ms": None,
         "token_count": 321, "total_ms": events[1]["total_ms"]}]
    assert session.finished == [("What's my name?", "", call)]


def test_a_second_tool_call_in_one_reply_is_reported_not_swallowed(caplog):
    """Only the first is answered — the robot runs one tool per round and the
    model can ask again in the next one. But a `move` lost this way is a robot
    that was asked to dance and did not, so the log says it happened."""
    calls = [{"name": "remember", "arguments": {"query": "name"}},
             {"name": "move", "arguments": {"how": "dance"}}]
    session = _SessionStub([{"type": "tool_call", **call} for call in calls])
    with caplog.at_level(logging.WARNING, logger="demo.serve"):
        events = list(chat_respond(ChatRequest([], "Dance and tell me my name."),
                                   session, _stack()))
    assert events[-1]["tool_call"] == calls[0]
    assert "move" in caplog.text and "2 tool calls" in caplog.text


def test_chat_endpoint_rejects_a_malformed_request_before_streaming(server):
    assert server().post("/chat", {"history": []})[0] == 400


def test_chat_endpoint_streams_the_turn_as_ndjson(server):
    session = _SessionStub([{"type": "content", "text": "Hi there."}])
    status, body = server(session=session).post(
        "/chat", {"history": [["a", "b"]], "text": "hello"})
    assert status == 200
    events = [json.loads(line) for line in body.splitlines() if line]
    assert events[0]["type"] == "audio"
    assert events[-1]["type"] == "done" and events[-1]["reply"] == "Hi there."


def test_chat_endpoint_ends_a_failed_stream_with_an_error_line(server):
    class Failing(_SessionStub):
        def turn(self, request):
            yield {"type": "content", "text": "Hel"}
            raise RuntimeError("engine died")

    status, body = server(session=Failing([])).post("/chat", {"text": "hello"})
    assert status == 200
    events = [json.loads(line) for line in body.splitlines() if line]
    assert events[-1]["type"] == "error" and "engine died" in events[-1]["message"]


def test_chat_endpoint_carries_a_days_pictures_through_to_the_session(server):
    # A day's answer is several frames in one turn: they have to survive the
    # wire, not just the codec.
    session = _SessionStub([{"type": "content", "text": "A window and a desk."}])
    status, _ = server(session=session).post("/chat", encode_chat_request(
        [], "What did you see today?", image_jpeg=[b"ONE", b"TWO", b"THREE"],
        image_note="(These are your memories of today.)"))
    assert status == 200
    assert session.requests[0].image_jpegs == (b"ONE", b"TWO", b"THREE")
    assert session.requests[0].image_note == "(These are your memories of today.)"


# --- the reply as words, for a robot that synthesizes it itself ---

def test_a_reply_cuts_into_the_same_sentences_with_or_without_audio():
    reply = [{"type": "content", "text": "Hello Sasha. "},
             {"type": "content", "text": "Nice to meet you."}]
    spoken = list(chat_respond(ChatRequest([], "Hi."), _SessionStub(reply), _stack()))
    written = list(chat_respond(ChatRequest([], "Hi.", sentences=True),
                                _SessionStub(reply), _stack()))
    assert [e["type"] for e in spoken[:-1]] == ["audio"] * (len(spoken) - 1)
    assert [e["type"] for e in written[:-1]] == ["sentence"] * (len(written) - 1)
    # The cut points must not change: the robot has to start talking exactly
    # as early as the laptop would have.
    assert [e["text"] for e in spoken[:-1]] == [e["text"] for e in written[:-1]]
    assert spoken[-1]["reply"] == written[-1]["reply"]


def test_the_words_path_never_calls_the_synthesizer():
    class Loud:
        sample_rate = 24000

        def speak(self, text):
            raise AssertionError("with synthesis on the robot the laptop "
                                 "must not synthesize")

    stack = _stack()
    stack.synthesizer = Loud()
    events = list(chat_respond(ChatRequest([], "Hi.", sentences=True),
                               _SessionStub([{"type": "content", "text": "Hello."}]),
                               stack))
    assert [e["type"] for e in events] == ["sentence", "done"]


# --- startup ---

def test_warm_up_exercises_the_slow_first_calls():
    """The opening reply took 7.6 s live while later ones took 1.5-3.2 s —
    all one-time cost landing on the first thing the presenter says. Warmup
    pays it while nobody is listening, and must touch the vision path."""
    from demo.serve import warm_up

    seen = {"images": [], "spoke": False, "heard": False}

    class Llm:
        def reply_stream(self, prompt, system, tools=None, image=None):
            seen["images"].append(image)
            yield {"type": "content", "text": "hi"}

    class Synth(_Synth):
        def speak(self, text):
            seen["spoke"] = True
            return super().speak(text)

    class Rec:
        def transcribe(self, pcm):
            seen["heard"] = pcm
            return ""

    warm_up(Models(recognizer=Rec(), synthesizer=Synth(), llm=Llm()))
    assert seen["images"] and seen["images"][0], "warmup must include an image"
    # The voice's own words, not silence: a speech detector in front of the
    # recognizer would keep silence from the model and leave it cold.
    assert seen["spoke"] and np.abs(seen["heard"]).max() > 0


def test_warm_up_survives_a_broken_component():
    from demo.serve import warm_up

    class Boom:
        sample_rate = 16000

        def reply_stream(self, *a, **k):
            raise RuntimeError("model exploded")

        def speak(self, text):
            raise RuntimeError("tts exploded")

        def transcribe(self, pcm):
            raise RuntimeError("asr exploded")

    warm_up(Models(recognizer=Boom(), synthesizer=Boom(), llm=Boom()))


def test_parse_args_defaults():
    from demo.serve import parse_args

    args = parse_args([])
    assert args.llm is None and args.asr == "whisper"
    # The robot reaches this across the network (README.md, Security).
    assert args.host == "0.0.0.0"


def test_main_builds_what_the_flags_name(monkeypatch, tmp_path):
    import emulator.engines
    import emulator.speech

    built = {}
    monkeypatch.setattr(emulator.speech, "build_recognizer",
                        lambda kind, model=None: built.update(asr=(kind, model)) or _Recognizer())
    monkeypatch.setattr(emulator.speech, "build_synthesizer", lambda: _Synth())
    monkeypatch.setattr(emulator.engines, "build_llm",
                        lambda spec: built.update(llm=spec) or SimpleNamespace(
                            open_chat=None))

    class FakeServer:
        def __init__(self, *args, **kwargs):
            pass

        def serve_forever(self):
            pass

    monkeypatch.setattr("demo.serve.HTTPServer", FakeServer)
    from demo.serve import main

    model = tmp_path / "custom.litertlm"
    model.write_bytes(b"")
    main(["--asr", "moonshine", "--asr-model", "base.en",
          "--llm", str(model), "--no-warmup"])
    assert built["asr"] == ("moonshine", "base.en")
    assert built["llm"].path == model
