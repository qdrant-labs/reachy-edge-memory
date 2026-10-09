"""emulator/speech_detector.py — whether an utterance holds speech at all.

The rule is Silero's own default (speech from 0.5, ended by 100 ms under
0.35, counted from 250 ms); a chunk is 32 ms, so 250 ms is just under eight
chunks. The model itself runs only when it is in the Hugging Face cache.
"""
from __future__ import annotations

import numpy as np
import pytest

from emulator.speech_detector import holds_speech


def test_a_quarter_second_of_speech_is_speech():
    assert holds_speech([0.0] * 5 + [0.9] * 8 + [0.0] * 5)


def test_less_than_a_quarter_second_is_not():
    # The longest run any noise clip reached was 7 chunks of motor whir.
    assert not holds_speech([0.0] * 5 + [0.9] * 7 + [0.0] * 5)


def test_a_dip_shorter_than_the_silence_rule_does_not_split_the_speech():
    # Two quiet chunks (64 ms) inside a word are not the end of it.
    assert holds_speech([0.9] * 5 + [0.1] * 2 + [0.9] * 3)


def test_a_real_pause_ends_the_speech_and_what_is_left_is_too_short():
    assert not holds_speech([0.9] * 5 + [0.1] * 5 + [0.9] * 5)


def test_a_started_stretch_goes_on_while_it_stays_over_the_lower_line():
    assert holds_speech([0.6] + [0.4] * 8)


def test_nothing_over_the_line_is_never_speech():
    assert not holds_speech([0.45] * 50)
    assert not holds_speech([])


def test_the_first_stretch_of_speech_is_enough():
    # The detector is lazy: past the answer it reads nothing more, so a turn
    # does not wait on the rest of the utterance.
    read = []

    def probabilities():
        for prob in [0.9] * 8 + [0.0] * 100:
            read.append(prob)
            yield prob

    assert holds_speech(probabilities())
    assert len(read) == 8


def _silero():
    try:
        from huggingface_hub import hf_hub_download

        from emulator.models import get

        spec = get("silero-vad")
        return hf_hub_download(spec.repo, spec.file, local_files_only=True)
    except Exception:  # noqa: BLE001 — not cached, or offline
        return None


@pytest.mark.skipif(_silero() is None, reason="Silero VAD is not cached")
def test_with_the_real_model_noise_and_silence_hold_no_speech():
    from emulator.speech_detector import SpeechDetector

    detector = SpeechDetector(_silero())
    rng = np.random.default_rng(0)
    assert not detector.holds_speech(rng.normal(0, 0.02, 32000).astype(np.float32))
    assert not detector.holds_speech(np.zeros(32000, np.float32))


def _spoken(text):
    """`text` in a macOS voice, float32 at 16 kHz, or None off a Mac."""
    import shutil
    import subprocess
    import tempfile
    import wave
    from pathlib import Path

    if shutil.which("say") is None:
        return None
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "said.wav"
        subprocess.run(["say", "-o", str(path), "--data-format=LEI16@16000", text],
                       check=True)
        with wave.open(str(path)) as wav:
            raw = wav.readframes(wav.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


@pytest.mark.skipif(_silero() is None or _spoken("Hi") is None,
                    reason="Silero VAD is not cached, or no `say` here")
def test_with_the_real_model_a_one_word_answer_holds_speech():
    # The shortest utterances measured ("Thanks.", "Bye.") held 9 chunks of
    # speech against the rule's 8: a bare name is that short.
    from emulator.speech_detector import SpeechDetector

    detector = SpeechDetector(_silero())
    for word in ("Sasha.", "Yes.", "Thanks."):
        audio = np.concatenate([np.zeros(4800, np.float32), _spoken(word),
                                np.zeros(9600, np.float32)])
        assert detector.holds_speech(audio), word


def test_a_padded_last_chunk_counts_only_the_audio_it_holds():
    # Eight chunks of speech reach 256 ms only if all of the last one is
    # audio; padded, it is the audio's own length that counts.
    assert holds_speech([0.9] * 8, length=8 * 512)
    assert not holds_speech([0.9] * 8, length=7 * 512 + 300)


def test_speech_that_ends_in_a_short_last_chunk_is_heard():
    from emulator.speech_detector import SpeechDetector

    seen = []

    class _Session:
        def run(self, _outputs, feed):
            seen.append(feed["input"].shape)
            return np.array([[0.9]], np.float32), feed["state"]

    detector = object.__new__(SpeechDetector)
    detector._session = _Session()
    # 7 chunks and 450 samples: 252 ms, over the rule only with the last bit.
    assert detector.holds_speech(np.ones(7 * 512 + 450, np.float32) * 0.1)
    assert seen == [(1, 576)] * 8, "the short last chunk is padded, not dropped"


def test_a_quiet_stretch_shorter_than_100_ms_is_inside_the_speech():
    # Four quiet chunks (96 ms after the first) do not end it; five would.
    assert holds_speech([0.9] * 5 + [0.1] * 4 + [0.9] * 5)


def test_speech_that_the_audio_ends_inside_a_pause_still_counts():
    # Silero's own end-of-audio rule: a stretch still open counts to the end.
    assert holds_speech([0.9] * 7 + [0.1] * 2)


def _robot_voice():
    try:
        from emulator.speech import build_synthesizer

        return build_synthesizer()
    except Exception:  # noqa: BLE001 — not cached, or no espeak-ng
        return None


@pytest.mark.skipif(_silero() is None, reason="Silero VAD is not cached")
def test_with_the_real_model_the_robots_own_voice_holds_speech():
    # "Bob." in the robot's voice has only 7 chunks over 0.5: the lower line
    # (0.35) is what carries it over 250 ms.
    voice = _robot_voice()
    if voice is None:
        pytest.skip("the Inflect voice is not available")
    from emulator.speech_detector import SpeechDetector

    detector = SpeechDetector(_silero())
    for word in ("Bob.", "Yes.", "No.", "Sasha."):
        audio = np.asarray(voice.speak(word), dtype=np.float32)
        count = int(len(audio) * 16000 / voice.sample_rate)
        audio = np.interp(np.linspace(0, len(audio) - 1, count),
                          np.arange(len(audio)), audio).astype(np.float32)
        audio = np.concatenate([np.zeros(4800, np.float32), audio,
                                np.zeros(9600, np.float32)])
        assert detector.holds_speech(audio), word
