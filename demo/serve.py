"""The laptop's model service: speech recognition, the language model and the
voice, over HTTP, for the robot's voice loop (demo/run_demo.py).

Four endpoints (demo/contract.py):

- /transcribe — audio in, the words heard out;
- /chat — one turn of the conversation, streamed back sentence by sentence;
- /say — a fixed line in the robot's voice (a greeting, the name question);
- /name — the name in an answer to "what's your name?".

The robot owns the conversation and its memory; this side only computes.
"""

from __future__ import annotations

import argparse
import base64
import dataclasses
import json
import logging
import re
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np

from demo.chat_session import ChatSession
from demo.contract import (
    CHAT_PATH,
    NAME_PATH,
    SAY_PATH,
    TRANSCRIBE_PATH,
    ChatRequest,
    decode_chat_request,
    decode_transcribe_request,
    encode_transcribe_response,
)
from demo.http_util import ascii_reason as _ascii_reason
from demo.stream import extract_sentences, flush_tail
from emulator import models

LOG = logging.getLogger(__name__)

# Cap on a request body. The largest is /chat carrying pictures: 4 frames (the
# engine's MAX_IMAGES_PER_TURN) at ~66 KB each as base64 is ~350 KB; /transcribe
# carries up to ~8 s of 16 kHz float32 audio, ~700 KB as base64.
MAX_BODY = 8 * 1024 * 1024


class BadRequest(Exception):
    """A request this service cannot read: answered 400. Anything else that
    goes wrong is the service's own failure, answered 500 with a traceback in
    the log."""


def _decoded(decode, payload):
    try:
        return decode(payload)
    except (KeyError, ValueError, TypeError) as exc:
        raise BadRequest(f"{type(exc).__name__}: {exc}") from exc


@dataclasses.dataclass
class Models:
    """What this service runs: the recognizer, the voice and the LLM."""
    recognizer: object
    synthesizer: object
    llm: object


def _audio_event(text, audio, sample_rate):
    return {
        "type": "audio",
        "text": text,
        "audio_b64": base64.b64encode(
            np.asarray(audio, dtype=np.float32).tobytes()).decode("ascii"),
        "sample_rate": int(sample_rate),
    }


def handle_say(payload: dict, stack: Models) -> dict:
    """Synthesize one fixed line. The robot speaks a handful of these itself —
    greeting someone it recognises, asking a stranger for their name
    (demo/people.py) — and those must come out the same every time, so they do
    not go through the language model at all."""
    text = payload.get("text", "")
    if not isinstance(text, str) or not text.strip():
        raise BadRequest("say needs a non-empty 'text'")
    audio = stack.synthesizer.speak(text)
    return _audio_event(text, audio, stack.synthesizer.sample_rate)


# Reading a name out of an answer is a language question, so the model answers
# it rather than a regular expression. Measured over 15 answers: the model read
# 14 right, a pattern 12 — it took "What's yours?" and "Um, I think so." for
# names. The guard below is on the shape of the reply, not on the words: a name
# is one short word.
NAME_SYSTEM = ('You extract a name. The robot asked "What is your name?" and '
               'this is the answer. Reply with the name alone, nothing else — '
               'no punctuation, no sentence. If the answer holds no name, '
               'reply NONE.')
NAME_MAX_CHARS = 20


def handle_name(payload: dict, stack: Models) -> dict:
    """An answer to the name question in, `{"name": "Sasha"}` or
    `{"name": null}` out."""
    heard = payload.get("text")
    if not isinstance(heard, str) or not heard.strip():
        raise BadRequest("name needs a non-empty 'text'")
    said = "".join(event.get("text", "") for event
                   in stack.llm.reply_stream(heard.strip(), system=NAME_SYSTEM)
                   if event.get("type") == "content")
    name = said.strip().strip(".!?,\"'").strip()
    if not name or name.upper() == "NONE":
        return {"name": None}
    if len(name) > NAME_MAX_CHARS or " " in name:
        print(f"  [name] not a name: {said!r}")
        return {"name": None}
    return {"name": name[:1].upper() + name[1:]}


def handle_transcribe(payload: dict, stack: Models) -> dict:
    """Audio in, text out — no model, no voice."""
    audio, _sample_rate = _decoded(decode_transcribe_request, payload)
    return encode_transcribe_response(stack.recognizer.transcribe(audio))


# gemma sometimes starts a reply with its raw tool-call tokens instead of a
# clean tool_calls entry; spoken, that would be read out character by
# character. Stripped from the start of the text before it reaches the voice.
_TOOL_CALL_LEAK = re.compile(r"^\s*<\|tool_call\|?>.*?\}\s*", re.DOTALL)


def _strip_tool_call_leak(text: str) -> str:
    """Strip a leaked `<|tool_call>...}` fragment from the start of the text."""
    return _TOOL_CALL_LEAK.sub("", text, count=1)


@dataclasses.dataclass
class _Spoken:
    """What a reply said, for its caller to act on once it has been streamed:
    the text, when the first sound was ready, and any tool calls that came
    instead of speech."""
    text: str = ""
    first_sound_ms: float | None = None
    tool_calls: list[dict] = dataclasses.field(default_factory=list)


def _sentence_event(text):
    """The reply's words, for a caller that synthesizes them itself."""
    return {"type": "sentence", "text": text}


def _speak(messages, stack: Models, t0: float, spoken: _Spoken,
           as_sentences: bool = False):
    """Audio events for a stream of reply events, synthesized sentence by
    sentence so the first one plays while the rest is still being generated.
    What was said lands in `spoken`.

    `as_sentences`: the robot synthesizes the voice itself and asked for the
    words instead. The cut points are the same either way — extract_sentences
    decides them — so both start talking at the same moment in the reply."""
    sample_rate = stack.synthesizer.sample_rate

    def synth(text):
        if as_sentences:
            if not text:
                return None
            if spoken.first_sound_ms is None:
                spoken.first_sound_ms = (time.perf_counter() - t0) * 1000
            return _sentence_event(text)
        audio = stack.synthesizer.speak(text)
        if len(audio) > 0:
            if spoken.first_sound_ms is None:
                spoken.first_sound_ms = (time.perf_counter() - t0) * 1000
            return _audio_event(text, audio, sample_rate)
        return None

    buffer = ""
    for msg in messages:
        if msg["type"] == "tool_call":
            spoken.tool_calls.append(msg)
            continue
        buffer = _strip_tool_call_leak(buffer + msg["text"])
        spoken.text = _strip_tool_call_leak(spoken.text + msg["text"])
        sentences, buffer = extract_sentences(buffer)
        for sentence in sentences:
            event = synth(sentence)
            if event is not None:
                yield event
    tail = flush_tail(buffer)
    if tail:
        event = synth(tail)
        if event is not None:
            yield event


def chat_respond(request: ChatRequest, session: ChatSession, stack: Models):
    """One /chat request as events — see demo/contract.py's CHAT_PATH note
    and demo/chat_session.py.

    A reply streams sentence by sentence, then `done`. A tool call streams
    nothing to hear — a `tool_call` event and a `done` that carries it; the
    robot runs the tool and answers with a second request. done's
    `token_count` is how full the model's context is now: the robot decides
    from it when older turns move into its memory.
    """
    t0 = time.perf_counter()
    spoken = _Spoken()
    yield from _speak(session.turn(request), stack, t0, spoken,
                      as_sentences=request.sentences)
    if spoken.tool_calls:
        if len(spoken.tool_calls) > 1:
            # Only the first is answered: the robot runs one tool per round
            # and the model may ask again in the next one (demo/conversation.
            # py's MAX_TOOL_ROUNDS). Logged rather than dropped in silence — a
            # `move` lost this way is a robot that was asked to dance and did
            # not, with nothing in the log to say why.
            LOG.warning("chat_respond: %d tool calls in one reply; answering "
                        "%r and leaving %s for the next round",
                        len(spoken.tool_calls), spoken.tool_calls[0]["name"],
                        [call["name"] for call in spoken.tool_calls[1:]])
        call = spoken.tool_calls[0]
        tool_call = {"name": call["name"],
                     "arguments": call.get("arguments") or {}}
        session.finish(request, "", tool_call)
        yield {"type": "tool_call", **tool_call}
        yield {"type": "done", "reply": "", "tool_call": tool_call,
               "first_sound_ms": spoken.first_sound_ms,
               "token_count": session.token_count,
               "total_ms": (time.perf_counter() - t0) * 1000}
        return
    reply_text = spoken.text.strip()
    session.finish(request, reply_text, None)
    yield {"type": "done", "reply": reply_text,
           "first_sound_ms": spoken.first_sound_ms,
           "token_count": session.token_count,
           "total_ms": (time.perf_counter() - t0) * 1000}


def make_handler(stack: Models, chat_session: ChatSession):
    class Handler(BaseHTTPRequestHandler):
        # One worker for the whole process (HTTPServer, not the threading
        # one): there is one robot and one chat. The timeout keeps a hung
        # client from holding the only socket.
        timeout = 60

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", 0))
            except ValueError as exc:
                LOG.warning("bad Content-Length: %s: %s",
                            type(exc).__name__, exc)
                self.send_error(400, "bad Content-Length")
                return
            if length <= 0 or length > MAX_BODY:
                LOG.warning("body size out of bounds: %d bytes", length)
                self.send_error(413, "request body out of bounds")
                return
            try:
                payload = json.loads(self.rfile.read(length))
            except json.JSONDecodeError as exc:
                LOG.warning("bad JSON: %s: %s", type(exc).__name__, exc)
                self.send_error(400, "bad JSON")
                return
            if not isinstance(payload, dict):
                self.send_error(400, "the body must be a JSON object")
                return
            routes = {TRANSCRIBE_PATH: lambda: self._json(handle_transcribe(payload, stack)),
                      SAY_PATH: lambda: self._json(handle_say(payload, stack)),
                      NAME_PATH: lambda: self._json(handle_name(payload, stack)),
                      CHAT_PATH: lambda: self._chat(payload)}
            route = routes.get(self.path)
            if route is None:
                self.send_error(404)
                return
            try:
                route()
            except BadRequest as exc:
                LOG.warning("bad request to %s: %s", self.path, exc)
                self.send_error(400, _ascii_reason(str(exc)))
            except Exception as exc:  # noqa: BLE001 — the server must survive
                LOG.exception("%s failed: %s", self.path, type(exc).__name__)
                try:
                    self.send_error(500, _ascii_reason(str(exc)))
                except Exception:
                    pass  # the response may already be partly sent

        def _json(self, body):
            data = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _chat(self, payload):
            # Decoded — and so validated — BEFORE the 200: a malformed
            # request gets a clean 400, not an error line mid-stream.
            request = _decoded(decode_chat_request, payload)
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()
            events = chat_respond(request, chat_session, stack)
            try:
                for event in events:
                    self.wfile.write((json.dumps(event) + "\n").encode())
                    self.wfile.flush()
            except Exception as exc:  # noqa: BLE001 — headers are already sent
                LOG.exception("chat: mid-stream failure: %s", type(exc).__name__)
                error_event = {"type": "error", "message": str(exc)[:200]}
                try:
                    self.wfile.write((json.dumps(error_event) + "\n").encode())
                    self.wfile.flush()
                except Exception:
                    pass  # the connection may have already dropped
            finally:
                # Reaches ChatSession.turn's cleanup: a reply the robot stopped
                # reading leaves the chat in an unknown state — dropped, and
                # rebuilt from the robot's copy on the next request.
                events.close()

        def log_request(self, code="-", size="-"):
            # Quiet on success; 4xx/5xx still reach the log.
            if isinstance(code, int) and 200 <= code < 300:
                return
            super().log_request(code, size)

    return Handler


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="The laptop's model service: ASR, "
                                            "the language model and the voice")
    # Bound wide by default: the robot reaches it across the network. See the
    # README's security section — this service has no authentication.
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=9500)
    p.add_argument("--asr", choices=("whisper", "moonshine"), default="whisper",
                   help="whisper (faster-whisper small.en) or moonshine (the "
                        "small LiteRT one the robot runs); see "
                        "emulator/whisper_asr.py")
    p.add_argument("--asr-model", default=None, metavar="NAME",
                   help="which whisper (small.en by default; base.en and "
                        "tiny.en trade words for time)")
    p.add_argument("--llm", default=None,
                   help="catalog name (emulator/models.py) or a path to a "
                        ".litertlm file; default: models.LLM")
    p.add_argument("--no-warmup", action="store_true",
                   help="skip the startup warmup (the first reply will be slow)")
    return p.parse_args(argv)


# The warmup's one question. Short, and not the chat's system prompt: the
# point is to pay for the engine's first generation and its vision executor,
# not to rehearse a conversation.
WARMUP_SYSTEM = "You are a friendly robot. Reply in one short sentence."


def warm_up(stack: Models, chat_session: ChatSession | None = None) -> None:
    """Pay the first-call costs before anyone is listening.

    Measured on the first live run: the opening reply took 7.6 s while every
    later one took 1.5-3.2 s. The difference is all one-time — the LLM's first
    generation, the vision executor's ~3 s initialisation, the voice loading
    its phonemiser — and it lands on the first thing the presenter says.

    Each step is guarded: a warmup is an optimisation, and a failure here must
    not stop the server from starting. If something is genuinely broken the
    real request will say so.
    """
    print("  warming up (first call is the slow one)...", flush=True)
    t0 = time.perf_counter()
    try:
        # With a picture, so the vision executor is built too.
        for _ in stack.llm.reply_stream("Say hi.", system=WARMUP_SYSTEM,
                                        image=_blank_jpeg()):
            pass
    except Exception as exc:  # noqa: BLE001 — warmup must never block startup
        print(f"    llm warmup skipped ({type(exc).__name__}: {exc})")
    if chat_session is not None:
        # The chat's first turn, with its tool schemas, so the robot's first
        # question is not it.
        try:
            for _ in chat_session.turn(ChatRequest(history=[], text="Hi.")):
                pass
        except Exception as exc:  # noqa: BLE001
            print(f"    chat warmup skipped ({type(exc).__name__}: {exc})")
        finally:
            chat_session.reset()
    spoken = None
    try:
        spoken = stack.synthesizer.speak("Ready.")
    except Exception as exc:  # noqa: BLE001
        print(f"    tts warmup skipped ({type(exc).__name__}: {exc})")
    try:
        # The voice's own "Ready.", not silence: moonshine sits behind a
        # speech detector (emulator/speech_detector.py), which would keep
        # silence from the model and leave it cold.
        stack.recognizer.transcribe(_at_16k(
            spoken, getattr(stack.synthesizer, "sample_rate", 16000)))
    except Exception as exc:  # noqa: BLE001
        print(f"    asr warmup skipped ({type(exc).__name__}: {exc})")
    print(f"  warm in {time.perf_counter() - t0:.1f}s", flush=True)


def _at_16k(audio, rate: int) -> np.ndarray:
    """`audio` resampled to the recognizers' 16 kHz — a second of silence
    when there is none."""
    if audio is None or not len(audio):
        return np.zeros(16000, dtype=np.float32)
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if rate == 16000:
        return audio
    count = int(len(audio) * 16000 / rate)
    return np.interp(np.linspace(0, len(audio) - 1, count),
                     np.arange(len(audio)), audio).astype(np.float32)


def _blank_jpeg(size: int = 64) -> bytes:
    """A minimal JPEG, just to force the vision path to initialise."""
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.zeros((size, size, 3), np.uint8)).save(buf, "JPEG")
    return buf.getvalue()


def main(argv=None) -> int:
    from demo.logs import setup_logging
    from emulator.engines import build_llm
    from emulator.speech import build_recognizer, build_synthesizer

    setup_logging()
    args = parse_args(argv)
    stack = Models(recognizer=build_recognizer(args.asr, args.asr_model),
                   synthesizer=build_synthesizer(),
                   llm=build_llm(models.resolve_llm(args.llm or models.LLM)))
    chat_session = ChatSession(stack.llm)
    server = HTTPServer((args.host, args.port), make_handler(stack, chat_session))
    print(f"model service on {args.host}:{args.port}: asr {args.asr}, "
          f"tts {models.TTS}, llm {args.llm or models.LLM}", flush=True)
    if not args.no_warmup:
        warm_up(stack, chat_session)
    print("  ready", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
