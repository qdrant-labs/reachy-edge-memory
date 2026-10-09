"""The recognizer and the voice, built by name from the catalog
(emulator/models.py). Shared by the laptop's service (demo/serve.py) and the
robot's own loop when a model is placed there (demo/run_demo.py)."""

from __future__ import annotations

from emulator import models


def build_recognizer(kind: str, model: str | None = None):
    """"whisper" (faster-whisper, on the laptop) or "moonshine" (the small
    LiteRT one, for the robot). Both take float32 PCM at 16 kHz and return
    text."""
    if kind == "moonshine":
        from emulator.asr import MoonshineTokenizer, Recognizer
        from emulator.speech_detector import SpeechDetector

        return Recognizer(models.fetch(models.ASR),
                          MoonshineTokenizer(models.fetch("moonshine-tokenizer")),
                          speech=SpeechDetector(models.fetch("silero-vad")))
    if kind == "whisper":
        from emulator.whisper_asr import DEFAULT_MODEL, WhisperRecognizer

        return WhisperRecognizer(model or DEFAULT_MODEL)
    raise ValueError(f"unknown ASR {kind!r} (whisper, moonshine)")


def build_synthesizer():
    """The Inflect voice (emulator/inflect_tts.py)."""
    from emulator.inflect_tts import InflectSynthesizer

    return InflectSynthesizer(models.fetch(models.TTS))
