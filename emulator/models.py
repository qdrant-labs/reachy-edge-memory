"""Model catalog: every model the demo loads, defined once, by name.

Each entry is a Hugging Face repo and the file (or files) the demo needs from
it; `fetch` downloads them into the Hugging Face cache on first use and
returns the local path. The one exception is the face embedder, which has no
LiteRT build on the Hub: it is built from its PyTorch weights with
`scripts/convert_hsface.py` (about two minutes, once), when `fetch` is asked
to — by demo/stage.py at start and by this module's main, never by a service
in the middle of a request.

    uv run python -m emulator.models            # everything up front
    uv run python -m emulator.models hsface     # just these

Swap a model for a stage by changing its entry or the stage's name below.
The embedding models and Whisper are loaded by name by their own libraries
(emulator/frame_memory.py, memory.py, whisper_asr.py); `main` fetches them too.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ASSETS = REPO / "assets"


@dataclass(frozen=True)
class Model:
    """One model: a single file in a Hugging Face repo (`repo` + `file`),
    several files from one (`repo` + `patterns`, fetched as a directory), or
    a file on this machine (`path`) — made by `build`, run from the repository
    root, when it is not there yet."""

    repo: str | None = None
    file: str | None = None
    patterns: tuple[str, ...] = ()
    path: Path | None = None
    build: tuple[str, ...] = ()
    how_to_get: str = ""


MODELS: dict[str, Model] = {
    # YOLO26n, COCO, AGPL-3.0, in Arm's LiteRT export. See
    # emulator/detector.py for its contract.
    "yolo26n": Model(repo="Arm/yolo26n-fp16-litert",
                     file="yolo26n_conv2d_f16_weights.tflite"),
    # Speech recognition when it runs on the robot (the laptop uses Whisper,
    # emulator/whisper_asr.py). The tokenizer is the original model's.
    "moonshine-tiny": Model(repo="litert-community/moonshine-tiny",
                            file="moonshine_tiny_5s_f32.tflite"),
    "moonshine-tokenizer": Model(repo="UsefulSensors/moonshine-tiny",
                                 file="tokenizer.json"),
    # Whether an utterance holds speech at all, before moonshine hears it
    # (emulator/speech_detector.py). Silero VAD v5, MIT. Whisper on the laptop
    # runs Silero too: v6, bundled with faster-whisper, under its own looser rule.
    "silero-vad": Model(repo="onnx-community/silero-vad", file="onnx/model.onnx"),
    # Speech synthesis: the LiteRT graphs, their runtime (say.py) and the
    # espeak-ng text frontend, all from one repo (emulator/inflect_tts.py).
    "inflect-nano-v2": Model(repo="litert-community/Inflect-Nano-v2",
                             patterns=("say.py", "frontend/*", "frontend/**/*",
                                       "inflect_text_encoder.tflite",
                                       "inflect_decoder.tflite")),
    # Faces: OpenCV's YuNet finds them, HSFace turns one into an identity.
    "yunet": Model(repo="opencv/face_detection_yunet",
                   file="face_detection_yunet_2023mar.onnx"),
    "hsface": Model(path=ASSETS / "hsface10k.tflite",
                    build=("uv", "run", "--with", "torch", "--with", "litert-torch",
                           "python", "scripts/convert_hsface.py"),
                    how_to_get="convert it with `uv run --with torch --with "
                               "litert-torch python scripts/convert_hsface.py`"),
    # The language model, in-process through litert-lm (emulator/engines.py).
    "gemma-4-e2b": Model(repo="litert-community/gemma-4-E2B-it-litert-lm",
                         file="gemma-4-E2B-it.litertlm"),
}

# Which model each stage uses. Swap a stage = change one name here.
DETECTOR = "yolo26n"
ASR = "moonshine-tiny"
TTS = "inflect-nano-v2"
LLM = "gemma-4-e2b"


def get(name: str) -> Model:
    """Look a model up by name; raise loudly on an unknown name."""
    try:
        return MODELS[name]
    except KeyError:
        raise KeyError(
            f"unknown model {name!r}; known: {sorted(MODELS)}") from None


def _shown(path: Path) -> str:
    """A model's path as a message says it: from the repository root when it
    is in it — shorter, and the same on every machine."""
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def fetch(model: str | Model, *, build: bool = False) -> Path:
    """The local path to a model — a file, or a directory for a `patterns`
    model — downloading it on first use.

    The cache is tried first, offline: on the robot, which may have no route
    to the Hub, a model carried over by `scripts/robot_service.sh prepare`
    loads without a network round trip. A local model that is missing is
    built by its `build` command only with build=True; otherwise it fails
    fast, saying how to get it."""
    spec = get(model) if isinstance(model, str) else model
    if spec.path is not None:
        # Built only when asked (demo/stage.py at start, `python -m
        # emulator.models`): never inside a request — a service that finds
        # the model missing fails fast, as it always did.
        if not spec.path.exists() and spec.build and build:
            print(f"  building {spec.path.name}, once (about two minutes)...",
                  flush=True)
            try:
                code = _build(spec.build)
            except OSError as exc:  # no `uv` on this machine
                raise FileNotFoundError(
                    f"{_shown(spec.path)} could not be built ({exc}) — "
                    f"{spec.how_to_get or 'see README.md'}") from exc
            if code:
                raise FileNotFoundError(
                    f"{_shown(spec.path)} could not be built (exit {code}, its output "
                    f"is above) — {spec.how_to_get or 'see README.md'}")
        if not spec.path.exists():
            raise FileNotFoundError(
                f"{_shown(spec.path)} is missing — {spec.how_to_get or 'see README.md'}")
        return spec.path
    from huggingface_hub import hf_hub_download, snapshot_download

    def download(**offline):
        if spec.patterns:
            return snapshot_download(repo_id=spec.repo,
                                     allow_patterns=list(spec.patterns),
                                     **offline)
        return hf_hub_download(repo_id=spec.repo, filename=spec.file, **offline)

    try:
        local = Path(download(local_files_only=True))
        # An offline snapshot answers with the folder even when only some of
        # its files were ever downloaded; the named ones must all be there.
        if all((local / name).exists() for name in spec.patterns
               if "*" not in name):
            return local
    except Exception:  # noqa: BLE001 — not cached yet: fetch it
        pass
    return Path(download())


def _build(command: tuple[str, ...]) -> int:
    """Run a model's build command from the repository root; its output goes
    straight to this terminal — a build takes minutes and says why. Its exit
    code."""
    return subprocess.run(command, cwd=REPO, check=False).returncode


def resolve_llm(name_or_path: str) -> Model:
    """The LLM to load: a catalog name, or a path to any .litertlm file — so
    comparing models downloaded outside the catalog needs a flag, not an edit
    (demo/serve.py's --llm)."""
    path = Path(name_or_path).expanduser()
    if path.suffix == ".litertlm" or path.is_file():
        if not path.is_file():
            raise SystemExit(f"model file not found: {path}")
        return Model(path=path)
    return get(name_or_path)


def main(argv: list[str] | None = None) -> int:
    """Download every model in the catalog (building the face embedder), and
    the embedding models the laptop's services load by name (SigLIP 2,
    bge-small, Whisper) — or only the catalog models named."""
    names = list(sys.argv[1:] if argv is None else argv)
    failed = []
    for name in names or MODELS:
        try:
            print(f"  {name:<20} {fetch(name, build=True)}", flush=True)
        except Exception as exc:  # noqa: BLE001 — report every missing model
            print(f"  {name:<20} MISSING: {exc}", flush=True)
            failed.append(name)
    if names:
        return 1 if failed else 0
    from emulator.frame_memory import SiglipEmbedder
    from emulator.memory import DEFAULT_MODEL, _embedder
    from emulator.whisper_asr import DEFAULT_MODEL as WHISPER, WhisperRecognizer

    for label, load in (("siglip2", SiglipEmbedder),
                        ("bge-small", lambda: _embedder(DEFAULT_MODEL)),
                        (f"whisper {WHISPER}", lambda: WhisperRecognizer(WHISPER))):
        try:
            load()
            print(f"  {label:<20} ready", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  {label:<20} MISSING: {exc}", flush=True)
            failed.append(label)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
