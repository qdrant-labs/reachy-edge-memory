"""The hotword guard: Whisper handing back its own prompt instead of speech.

faster-whisper feeds `hotwords` to the decoder as preceding context, and on a
short utterance it can decode nothing from, the decoder continues that context
— it writes the list out. Live on stage, asked for a name, the
robot heard "Qdrant Edge, Reachy, Sasha, vector database, embeddings" and
answered "Sorry, I didn't catch your name"; the person had said "Sasha".

No model is loaded here: WhisperRecognizer imports faster_whisper lazily, and
these test the pure rule that decides what the transcript IS.
"""
from __future__ import annotations

from emulator.whisper_asr import HOTWORDS, is_hotword_echo


def test_the_list_coming_back_is_not_speech():
    assert is_hotword_echo(HOTWORDS)
    assert is_hotword_echo("Qdrant, Qdrant Edge, Reachy")
    # As it actually came back, with the decoder's own punctuation and a word
    # repeated (the shape that made the robot search its memory for Qdrant).
    assert is_hotword_echo("Qdrant Edge, Reachy, Reachy,")
    assert is_hotword_echo("Reachy. Qdrant.")


def test_one_hotword_on_its_own_is_something_a_person_says():
    """The demo is ABOUT Qdrant: "Qdrant Edge" as a whole utterance is a
    question to answer, not a prompt echo."""
    for said in ("Qdrant", "Qdrant Edge", "Reachy", "Qdrant Edge?"):
        assert not is_hotword_echo(said), said


def test_anything_with_a_word_of_its_own_is_speech():
    for said in ("Tell me about Qdrant", "Reachy, what is Qdrant Edge?",
                 "Sasha", "I'm Sasha", "Qdrant Edge is fast, Reachy"):
        assert not is_hotword_echo(said), said


def test_nothing_is_not_an_echo():
    assert not is_hotword_echo("")
    assert not is_hotword_echo("   ")


def test_the_name_is_not_in_the_prompt_any_more():
    """It bought nothing — measured, Whisper writes "Sasha", "I'm Sasha" and
    "My name is Sasha" correctly without it (16/18 either way) — and it cost the one thing that must never be fabricated:
    with "Sasha" in the list, an echo READS LIKE AN ANSWER to "what's your
    name?", and a wrong name attached to a face outlives the mistake."""
    assert "Sasha" not in HOTWORDS
    assert "Qdrant" in HOTWORDS and "Reachy" in HOTWORDS


class _Segment:
    def __init__(self, text, no_speech_prob):
        self.text = text
        self.no_speech_prob = no_speech_prob


class _Model:
    def __init__(self, segments):
        self.segments = segments
        self.options = {}

    def transcribe(self, audio, **options):
        self.options = options
        return iter(self.segments), None


def _recognizer(segments):
    import numpy as np

    from emulator.whisper_asr import WhisperRecognizer

    recognizer = object.__new__(WhisperRecognizer)
    recognizer._model = _Model(segments)
    recognizer._language = "en"
    recognizer._beam_size = 1
    return recognizer, np.zeros(16000, np.float32)


def test_a_segment_the_decoder_says_holds_no_speech_is_dropped():
    """With the VAD off, every segment Whisper wrote for 63 noise clips said
    0.385 or more, and none of 288 utterances over 0.139 — faster-whisper's
    own rule keeps them unless the words also came out unsure, and Whisper
    is sure of what it invents."""
    from emulator.whisper_asr import NO_SPEECH_MAX

    assert 0.139 < NO_SPEECH_MAX < 0.385
    recognizer, audio = _recognizer([_Segment(" Thanks for watching!", 0.62)])
    assert recognizer.transcribe(audio) == ""
    # The VAD stays in front: the 1-in-63 noise figure is for both together.
    assert recognizer._model.options["vad_filter"] is True
    recognizer, audio = _recognizer([_Segment(" Yes.", 0.139),
                                     _Segment(" Reachy", 0.45)])
    assert recognizer.transcribe(audio) == "Yes."


def test_the_hotword_list_over_speech_is_still_dropped():
    # Over speech the decoder is sure it heard some (0.007): only the prompt
    # tells the echo apart.
    recognizer, audio = _recognizer([_Segment(" Qdrant Edge, Reachy", 0.007)])
    assert recognizer.transcribe(audio) == ""
