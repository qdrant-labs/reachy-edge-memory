"""Speech recognition via moonshine-tiny on LiteRT.

The contract was captured on-device, not guessed.

    encode signature
      input  args_0  [1, 80000] float32   raw PCM 16 kHz, exactly 5 seconds
      output output_0 [1, 207, 288]

    decode signature
      input  args_0  [1, 207, 288]  float32  encoder output
      input  args_1  [1, 64]        int32    full token buffer
      input  args_2  [1, 1, 64, 64] float32  attention mask
      output output_0 [1, 64, 32768]         logits over the full vocab

No mel frontend needed: the model consumes raw audio directly. 52 subgraphs total.

On cost. The decoder gets the entire 64-token buffer on every step, meaning
there's no KV cache in this export and each step costs the same regardless of
position — 37 ms. Full window cost = 51 + 37*N; for a typical 16 tokens that's
643 ms for 5 seconds of audio, i.e. a 7.8x margin.

The decode inputs are distinguished by shape and dtype, not by name order: the
order in the signature's dict is not guaranteed, and args_N-style names carry
no information.

Two properties were found by brute force, since guessing didn't work, and they
only work together:

  1. **The mask is additive**: 0 allows a position, a large negative blocks it.
     A multiplicative 1/0 mask doesn't work at all.
  2. **The buffer starts with BOS = 1** (`<s>`), not zero. Zero is `<unk>`,
     and with it the model outputs `<unk>` forever regardless of the mask.

Neither fix helps alone: with BOS=0, all four mask variants produced nothing;
with BOS=1 and an additive mask, coherent text came out. The bug would have
been invisible without a reference recording, because the cost matched the
budget perfectly either way.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from emulator.litert_runtime import build_runner

if TYPE_CHECKING:
    from emulator.speech_detector import SpeechDetector

logger = logging.getLogger(__name__)

WINDOW_SAMPLES = 80000        # 5 seconds at 16 kHz
MIN_TAIL_SAMPLES = 8000       # half a second
MAX_DECODE_TOKENS = 64
BOS_TOKEN = 1                 # `<s>` in the vocab; zero is `<unk>`
MASK_BLOCKED = -1e9           # mask is additive, not multiplicative

# SentencePiece's byte fallback: a character outside the vocab comes as its
# UTF-8 bytes, one token each. "♪" is three of them, and before they were
# put back together the robot heard "<0xE2><0x99><0xAA>".
_BYTE_TOKEN = re.compile(r"<0x([0-9A-Fa-f]{2})>")


def prepare_window(pcm: np.ndarray) -> np.ndarray:
    """Reshape audio to [1, 80000] — the model requires an exact length."""
    if pcm.ndim != 1:
        raise ValueError(f"expected mono, got {pcm.shape}")
    if len(pcm) < WINDOW_SAMPLES:
        pcm = np.pad(pcm, (0, WINDOW_SAMPLES - len(pcm)))
    return pcm[:WINDOW_SAMPLES].astype(np.float32).reshape(1, WINDOW_SAMPLES)


def classify_decode_inputs(details: dict) -> dict[str, str]:
    """Which decode input is which.

    Distinguished by shape and dtype: 3D float is the encoder output,
    integer is the token buffer, the remaining one is the attention mask.
    Order or names can't be relied on, and feeding garbage into the wrong
    slot means getting plausible-looking but wrong text.
    """
    hidden = tokens = mask = None
    for key, detail in details.items():
        shape = [int(x) for x in detail["shape"]]
        dtype = np.dtype(detail["dtype"])
        if np.issubdtype(dtype, np.integer):
            tokens = key
        elif len(shape) == 3:
            hidden = key
        else:
            mask = key
    if hidden is None or tokens is None or mask is None:
        raise ValueError(
            f"decode contract changed: didn't find all three inputs among "
            f"{ {k: [int(x) for x in v['shape']] for k, v in details.items()} }"
        )
    return {"hidden": hidden, "tokens": tokens, "mask": mask}


def causal_mask(step: int, size: int = MAX_DECODE_TOKENS,
                additive: bool = True) -> np.ndarray:
    """Attention mask: at step N, positions zero through N are visible.

    Additive by default — 0 allows, a large negative blocks. This is exactly
    what this export expects: a multiplicative 1/0 mask gives an empty output.
    The additive parameter is kept to test the opposite variant in tests.
    """
    if step >= size:
        raise ValueError(f"step {step} is not less than buffer size {size}")
    rows = np.arange(size).reshape(-1, 1)
    cols = np.arange(size).reshape(1, -1)
    allowed = (cols <= rows) & (rows <= step)
    if additive:
        mask = np.where(allowed, 0.0, MASK_BLOCKED)
    else:
        mask = allowed.astype(np.float32)
    return mask.astype(np.float32).reshape(1, 1, size, size)


class MoonshineTokenizer:
    """Reverse mapping from token ids to text."""

    SPECIAL = frozenset({"<s>", "</s>", "<pad>", "<unk>"})
    END = "</s>"

    def __init__(self, vocab_path: Path) -> None:
        data = json.loads(Path(vocab_path).read_text(encoding="utf-8"))
        vocab = data.get("model", {}).get("vocab", data.get("vocab", {}))
        if not vocab:
            # A wrong-format tokenizer JSON yields {} here; init would then
            # succeed with an empty table and EVERY transcribe() would return
            # "" — the robot would only ever wake on scene change, never on
            # speech, with no error. Fail loud at construction instead.
            raise ValueError(
                f"tokenizer at {vocab_path} has an empty vocab: expected a "
                "'model.vocab' or top-level 'vocab' mapping")
        self._by_id = {int(v): k for k, v in vocab.items()}

    def is_end(self, token_id: int) -> bool:
        """End of sequence. Public method: the decode loop must be able to
        stop without reaching into the tokenizer's internals."""
        return self._by_id.get(int(token_id)) == self.END

    def decode(self, ids: list[int]) -> str:
        pieces: list[str] = []
        unknown: list[int] = []
        raw = bytearray()      # byte-fallback tokens not yet put together
        for token_id in ids:
            token = self._by_id.get(int(token_id))
            if token is None:
                unknown.append(int(token_id))
                continue
            if token == self.END:
                break          # what follows is garbage from the unrun buffer
            if token in self.SPECIAL:
                continue
            byte = _BYTE_TOKEN.fullmatch(token)
            if byte:
                raw.append(int(byte.group(1), 16))
                continue
            if raw:
                pieces.append(raw.decode("utf-8", errors="replace"))
                raw.clear()
            pieces.append(token)
        if raw:
            pieces.append(raw.decode("utf-8", errors="replace"))
        if unknown:
            # An unknown id being silently dropped would mask a vocab/model
            # mismatch. Log both the count and the ids themselves so this
            # is visible instead of getting lost in the decoded text.
            logger.warning("decode: %d unknown token id(s) dropped: %s",
                           len(unknown), unknown)
        # The ▁ marker denotes a word start: join pieces and turn it into a
        # space, otherwise words would fall apart into subwords.
        joined = "".join(pieces)
        if "▁" in joined:
            return joined.replace("▁", " ").strip()
        return " ".join(pieces).strip()


class Recognizer:
    def __init__(self, model_path: Path, tokenizer: MoonshineTokenizer,
                 threads: int = 4,
                 speech: "SpeechDetector | None" = None) -> None:
        """`speech` (emulator/speech_detector.py's SpeechDetector) says
        whether the utterance holds speech at all; without one, everything
        is transcribed."""
        # CompiledModel is the current LiteRT API (Interpreter is deprecated),
        # but a CPU without the ARMv8 crypto extensions (the robot's CM4)
        # cannot build one at all, so the runner is chosen for the board
        # (emulator/litert_runtime.py's build_runner). The two signatures —
        # encode, decode — are called the same way through either, so
        # transcribe() below is unchanged.
        runner = build_runner(model_path, threads=threads)
        self._encode = runner.signature("encode")
        self._decode = runner.signature("decode")
        self._tokenizer = tokenizer
        self._keys = classify_decode_inputs(self._decode.get_input_details())
        self._runner = runner
        self._speech = speech

    def transcribe(self, pcm: np.ndarray) -> str:
        """Text for any length of audio. The model hears exactly 5 s; a
        longer utterance (the voice loop takes up to 8 s) is heard window by
        window rather than cut off at 5 s. A last piece under half a second
        is the silence the voice gate waits for, not words.

        Nothing for an utterance with no speech in it: this model writes
        something for any noise ("You", for most), and its confidence in it
        is no lower than in real words (see emulator/speech_detector.py)."""
        pcm = np.asarray(pcm, dtype=np.float32).reshape(-1)
        if self._speech is not None and not self._speech.holds_speech(pcm):
            # Said, so a robot that stops hearing someone is not silent about
            # why: a far microphone or a clipped one-word answer lands here too.
            print(f"  [asr] no speech in {len(pcm) / 16000:.1f} s of sound — "
                  "not transcribed")
            return ""
        pieces = [pcm[start:start + WINDOW_SAMPLES]
                  for start in range(0, max(len(pcm), 1), WINDOW_SAMPLES)]
        if len(pieces) > 1 and len(pieces[-1]) < MIN_TAIL_SAMPLES:
            pieces.pop()
        return " ".join(text for text in map(self._transcribe_window, pieces)
                        if text)

    def _transcribe_window(self, pcm: np.ndarray) -> str:
        window = prepare_window(pcm)
        encode_key = next(iter(self._encode.get_input_details()))
        hidden = next(iter(self._encode(**{encode_key: window}).values()))

        details = self._decode.get_input_details()
        tokens_detail = details[self._keys["tokens"]]
        size = int(tokens_detail["shape"][1])
        tokens = np.zeros(
            (1, size), dtype=np.dtype(tokens_detail["dtype"]).type)
        # The first token must be BOS: with zero (`<unk>`) the model outputs
        # `<unk>` forever regardless of the mask.
        tokens[0, 0] = BOS_TOKEN

        produced: list[int] = []
        for step in range(size - 1):
            out = self._decode(**{
                self._keys["hidden"]: hidden,
                self._keys["tokens"]: tokens,
                self._keys["mask"]: causal_mask(step, size),
            })
            logits = next(iter(out.values()))
            token_id = int(np.argmax(logits[0, step]))
            produced.append(token_id)
            if self._tokenizer.is_end(token_id):
                break
            tokens[0, step + 1] = token_id

        return self._tokenizer.decode(produced)
