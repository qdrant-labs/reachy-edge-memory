"""Codec for the HTTP contract between the robot and the laptop.

Pure functions, kept separate from the network: both the server
(demo/serve.py) and its caller on the robot depend on one shared format, and
it can be tested without either. Audio travels as base64-encoded float32 —
JSON can't carry binary, and shipping a WAV container over HTTP would be
overkill.

Four endpoints: /transcribe (audio in, text out), /chat (the conversation),
/say (a fixed line to speak) and /name (the name in an answer).
"""

from __future__ import annotations

import base64
import dataclasses
from collections.abc import Sequence

import numpy as np


# Explicit little-endian dtype (`<f4`), not bare `np.float32`: numpy's native
# float32 follows host byte order, which happens to be LE on most of our hosts
# (the robot, the laptop — arm64), but the HTTP contract outlives any given
# host — an explicit byte order makes the codec portable, rather than "works
# because both ends currently happen to be LE".
_AUDIO_DTYPE = "<f4"


def _encode_audio(audio: np.ndarray) -> str:
    return base64.b64encode(
        np.asarray(audio, dtype=_AUDIO_DTYPE).tobytes()).decode("ascii")


def _decode_audio(text: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(text), dtype=_AUDIO_DTYPE).copy()


def encode_transcribe_request(audio: np.ndarray, sample_rate: int) -> dict:
    """Request body for /transcribe: the utterance's audio, nothing else."""
    return {"audio_b64": _encode_audio(audio), "sample_rate": int(sample_rate)}


def decode_transcribe_request(payload: dict) -> tuple[np.ndarray, int]:
    return _decode_audio(payload["audio_b64"]), int(payload["sample_rate"])


def encode_transcribe_response(heard: str) -> dict:
    return {"heard": heard}


def decode_transcribe_response(payload: dict) -> str:
    return payload.get("heard", "")


# — /chat: the conversation (demo/serve.py's chat handler; demo/conversation.py
# on the robot is the caller) —
#
# The robot owns the conversation; the Mac keeps a live LLM chat as a CACHE of
# it (demo/chat_session.py). Every request carries the robot's whole current
# window — `history`, the exchanges still in context, oldest first — so the
# Mac can tell whether its live chat still matches (reuse it, KV cache and
# all) or has to be rebuilt from the robot's copy: after an eviction into
# Qdrant, after an image turn, after a restart of either side.
#
# A turn is one request, or two when the model calls a tool. The second one
# repeats the history and text and adds EITHER `tool_result` (a recall answered
# in words) OR `images_b64` + `image_note` (pictures for the model to look at:
# the camera now, or the frames a recall found). Never both — see
# decode_chat_request.
#
# `images_b64` is a LIST because a day's answer is up to three frames in one
# turn, under one note — a list of one
# is the ordinary single-frame case, not a special shape.
CHAT_PATH = "/chat"

# Audio in, the words heard out. A separate endpoint from /chat because the
# robot needs the text first: it is what the robot answers a name question
# with, and what it drops when it holds no words before the model sees it.
TRANSCRIBE_PATH = "/transcribe"

# Fixed lines the robot speaks itself, without the model: greeting someone it
# recognised, asking a stranger for their name (demo/people.py).
SAY_PATH = "/say"

# The name in an answer to "what's your name?", read by the model rather than
# by a pattern: "they call me Sasha" and "Sasha here" are names too.
NAME_PATH = "/name"


def _as_jpegs(images: bytes | Sequence[bytes] | None) -> tuple[bytes, ...]:
    """One picture, several, or none — as a tuple either way. A bare `bytes`
    is ONE picture, never a sequence of ints: that is what keeps a caller
    attaching a single frame (demo/conversation.py) writing exactly the body
    it wrote before."""
    if images is None:
        return ()
    if isinstance(images, (bytes, bytearray, memoryview)):
        return (bytes(images),)
    return tuple(images)


@dataclasses.dataclass(frozen=True)
class ChatRequest:
    history: list[tuple[str, str]]
    text: str
    tool_result: dict | None = None
    image_jpegs: tuple[bytes, ...] = ()
    image_note: str | None = None
    # "Send me the words, I will make the sound": set when synthesis runs on
    # the robot (demo/placement.py). The reply then streams as `sentence`
    # events at the same points audio would have been, so the robot still
    # starts speaking before the reply is finished.
    sentences: bool = False

    @property
    def image_jpeg(self) -> bytes | None:
        """The first picture, or None when the turn carries none. Every
        picture on a turn shares one note and one question, so a reader that
        only ever shows one asks for it by this name instead of indexing a
        tuple."""
        return self.image_jpegs[0] if self.image_jpegs else None


def encode_chat_request(history, text: str, *, tool_result: dict | None = None,
                        image_jpeg: bytes | Sequence[bytes] | None = None,
                        image_note: str | None = None,
                        sentences: bool = False) -> dict:
    """`image_jpeg` is one picture's JPEG bytes or several of them."""
    payload = {"history": [[person, reply] for person, reply in history],
               "text": text}
    if sentences:
        payload["sentences"] = True
    if tool_result is not None:
        payload["tool_result"] = tool_result
    jpegs = _as_jpegs(image_jpeg)
    if jpegs:
        payload["images_b64"] = [base64.b64encode(jpeg).decode("ascii")
                                 for jpeg in jpegs]
        payload["image_note"] = image_note or ""
    return payload


def decode_chat_request(payload: dict) -> ChatRequest:
    """Validates while it decodes: a ValueError here becomes a 400 before the
    streamed response has started (demo/serve.py), rather than an error line
    in the middle of one."""
    if not isinstance(payload, dict):
        raise ValueError("chat request must be a JSON object")
    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("chat request needs a non-empty 'text'")
    history = []
    for pair in payload.get("history") or []:
        if (not isinstance(pair, (list, tuple)) or len(pair) != 2
                or not all(isinstance(part, str) for part in pair)):
            raise ValueError("history entries must be [person, reply] strings")
        history.append((pair[0], pair[1]))
    tool_result = payload.get("tool_result")
    if tool_result is not None and (not isinstance(tool_result, dict)
                                    or not isinstance(tool_result.get("name"), str)):
        raise ValueError("tool_result must be an object with a 'name'")
    images_b64 = payload.get("images_b64") or []
    if (not isinstance(images_b64, list)
            or not all(isinstance(item, str) for item in images_b64)):
        raise ValueError("images_b64 must be a list of base64 strings")
    # validate=True: without it b64decode silently skips junk characters and
    # hands the vision encoder a corrupt picture instead of refusing it here.
    images = tuple(base64.b64decode(item, validate=True)
                   for item in images_b64 if item)
    if images and tool_result is not None:
        raise ValueError("a chat request carries a tool_result or images, not both")
    return ChatRequest(history=history, text=text, tool_result=tool_result,
                       image_jpegs=images,
                       image_note=payload.get("image_note") or None,
                       sentences=bool(payload.get("sentences")))
