"""Speech recognition with Whisper, for the machine that has room for it.

moonshine-tiny is fast enough to run on the robot itself (emulator/asr.py),
and that was all it was chosen for; its accuracy was never measured. On the
laptop, where recognition runs by default (demo/serve.py), the first live
conversation showed what that choice costs: "I mean what
did you see not what do you see now" came back as "Me to remember what we did
just cut", and the robot answered the garbage. Half the wrong turns that
evening started here.

Whisper small.en through faster-whisper (CTranslate2, int8) is a different
class of transcript and still well inside a turn on Apple silicon. Nothing
about this reaches the robot: the import below is lazy, so a machine without
faster-whisper installed can still run everything else.

`vad_filter` matters as much as the model: Whisper invents speech in silence
("Thank you.", "Thanks for watching!") and the filter drops those windows
before they reach the decoder. Noise that gets past it (one clip of 63,
measured) the decoder itself says holds no speech (NO_SPEECH_MAX).
"""
from __future__ import annotations

import re

import numpy as np

# small.en over base.en: the demo is English-only, and an English-only model
# of this size is the accuracy sweet spot before medium, which does not fit
# inside a conversational turn.
DEFAULT_MODEL = "small.en"

# int8 on CPU: CTranslate2's quantized path is what keeps this inside a turn
# on this machine. float16 on the GPU is not an option — CTranslate2 has no
# Metal backend.
DEFAULT_COMPUTE_TYPE = "int8"

# CTranslate2 picks a conservative thread count on its own; measured on this
# Mac over three synthesized utterances, saying so explicitly cuts a turn's
# transcription from 0.87 s to 0.55 s. base.en is 0.21 s and starts losing
# words ("I'm Sasha" -> "and Sasha"), tiny.en 0.08 s and is moonshine-grade —
# both are there for `--asr-model` if the stage needs the time back.
DEFAULT_THREADS = 8


# Words the demo says that a general recognizer does not expect. Measured on
# synthesized speech, clean and with noise, 18 utterances: without them
# "Qdrant" comes back as "footprint", "current", "Grunt", "the Droned edge";
# with them, right every time — 16/18 against 8/18, and the two misses
# ("Reachy" as "Reachi") are the same either way.
#
# THREE entries, not the six this used to carry. "Sasha", "vector database"
# and "embeddings" were in the list and bought nothing: Whisper writes
# "Sasha", "I'm Sasha" and "My name is Sasha" correctly without any help
# (16/18 with them, 16/18 without). What they cost was real — see
# is_hotword_echo below, and the name they put in the robot's mouth.
HOTWORDS = "Qdrant, Qdrant Edge, Reachy"

# The decoder's own estimate that a segment holds no speech (no_speech_prob),
# from which the segment is dropped. faster-whisper drops one only when that
# estimate is over 0.6 AND its words came out unsure (log-probability under
# -1.0), and Whisper is sure of what it invents. Measured with the VAD off, so
# that nothing filtered the noise first: on 63 noise clips (white, pink, hum,
# motor whir, clicks, breath, tones) every segment Whisper wrote — each one an
# echo of the hotwords: "Reachy", "Qdrant Edge, Reachy", "Edge, Reachy" — said
# 0.385 or more; on 288 utterances (24 phrases, six voices, clean and under
# noise), 0.139 at most. 0.25 sits between the two. With the VAD on, as here,
# one noise clip of the 63 got through it, as "Qdrant Edge, Reachy" at 0.65.
NO_SPEECH_MAX = 0.25

_WORDS = re.compile(r"[a-z]+")


def is_hotword_echo(text: str, hotwords: str = HOTWORDS) -> bool:
    """Whether this transcript is the hotword list coming back, not speech.

    faster-whisper feeds `hotwords` to the decoder as preceding context, and
    when a short utterance carries nothing it can decode, the decoder simply
    continues that context — it writes the list out. Seen live on stage:
    asked for a name, the robot heard "Qdrant Edge, Reachy, Sasha, vector
    database, embeddings" and answered "Sorry, I didn't catch
    your name"; another turn came back "Qdrant Edge, Reachy, Reachy," and the
    robot searched its memory for Qdrant. The person had said "Sasha", twice.
    The words are the SAME words either way, so nothing downstream can tell
    the echo from speech — only this side knows what was put in the prompt.

    Not a list of words of its own: the transcript is compared with the
    prompt this recognizer was given. Over noise the decoder knows it heard
    no speech (NO_SPEECH_MAX drops it first); over speech it does not —
    measured, "How do you work?" in the robot's own voice came back as
    "Qdrant Edge, Reachy" with a no-speech estimate of 0.007, and its
    words' probabilities without the prompt (word alignment: 0.001, 0.000,
    0.001) are as low as a real "Qdrant Edge" gets in the macOS Flo voice
    (0.000, 0.008). The prompt is the only thing that tells them apart.

    An echo is a transcript made ENTIRELY of hotwords, carrying two or more
    of the listed entries. One entry is left alone: "Qdrant Edge" on its own
    is a thing a person says to this robot, and the demo is about answering
    it. Two or more, and nothing else, is the list — the prompt back."""
    words = _WORDS.findall(text.lower())
    if not words:
        return False
    entries = [tuple(_WORDS.findall(entry))
               for entry in hotwords.lower().split(",")]
    entries = [entry for entry in entries if entry]
    if not set(words) <= {word for entry in entries for word in entry}:
        return False   # a word nobody put in the prompt: this is speech
    said = {entry for entry in entries if set(entry) <= set(words)}
    # "Qdrant" is inside "Qdrant Edge": count the longest form only, or a
    # genuine "Qdrant Edge?" would count as two entries and be thrown away.
    longest = [entry for entry in said
               if not any(other is not entry and set(entry) < set(other)
                          for other in said)]
    return len(longest) >= 2


class WhisperRecognizer:
    """Same contract as emulator/asr.py's Recognizer: float32 PCM at 16 kHz
    in, text out."""

    def __init__(self, model: str = DEFAULT_MODEL, *,
                 compute_type: str = DEFAULT_COMPUTE_TYPE,
                 device: str = "cpu", language: str = "en",
                 beam_size: int = 1, threads: int = DEFAULT_THREADS) -> None:
        from faster_whisper import WhisperModel

        self.model_name = model
        self._language = language
        self._beam_size = beam_size
        self._model = WhisperModel(model, device=device,
                                   compute_type=compute_type,
                                   cpu_threads=threads)

    def transcribe(self, pcm: np.ndarray) -> str:
        # HOTWORDS: the names this demo lives on. Measured on synthesized
        # speech: without them small.en wrote "QDrand Edge" and "QDRant"; with
        # them, "Qdrant Edge" and "Qdrant" — and a misheard name never reaches
        # the knowledge tool, which is how "tell me about quadrant" was
        # answered out of the model's own head instead of the shard.
        audio = np.asarray(pcm, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return ""
        segments, _info = self._model.transcribe(
            audio, language=self._language, beam_size=self._beam_size,
            vad_filter=True, without_timestamps=True,
            # Each utterance stands alone: conditioning on the previous one
            # makes Whisper continue a sentence nobody said when a turn is
            # short, which is most of them here.
            condition_on_previous_text=False, hotwords=HOTWORDS)
        kept = []
        for segment in segments:
            if segment.no_speech_prob < NO_SPEECH_MAX:
                kept.append(segment.text.strip())
            else:
                print(f"  [asr] dropping {segment.text.strip()!r} — Whisper "
                      f"says it holds no speech ({segment.no_speech_prob:.2f})")
        text = " ".join(kept).strip()
        if is_hotword_echo(text):
            # Not "the person said Qdrant": the prompt came back. Dropped
            # here rather than downstream — nothing after this can tell the
            # difference, and the robot answering its own hotword list is
            # worse than the robot hearing nothing.
            print(f"  [asr] ignoring {text!r} — the hotword list came back, "
                  "not speech")
            return ""
        return text
