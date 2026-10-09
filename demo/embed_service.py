"""The embedding models, on the laptop, for a robot that owns everything else.

The robot runs the demo: it captures, remembers, searches, decides and moves.
What it cannot carry cheaply is the models — SigLIP's vision tower costs 2.2 s
per frame on its compute module against 36 ms here (measured) — so by default
the embedders live on this side and the robot asks for a vector when it needs
one (`--on-robot embedder` moves them onto the robot instead).

Nothing about the robot's memory lives here. This service is stateless: bytes
in, a vector out. The frames, the vectors, the index and everything the robot
has seen stay in the Qdrant Edge shard on the robot itself — a frame passes
through here to be measured, and is stored there.

Two embedders, one for pictures and one for words:

- SigLIP 2 (768-d) for frames: the picture's own vector, which the robot
  picks the day's most different frames by (emulator/frame_memory.py's
  day_frames). Only the vision tower: no question is searched against a
  picture (see that module's docstring).
- bge-small (384-d) for text: what was said, the words about a frame and the
  facts, searched by meaning.

And the face models (emulator/face.py): a frame in, a box and an identity
vector per face out.
"""
from __future__ import annotations

import argparse
import base64
import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from PIL import Image

from demo.http_util import ascii_reason, public

# Decompression-bomb guard: a robot frame is under 1 Mpx.
Image.MAX_IMAGE_PIXELS = 10_000_000

LOG = logging.getLogger(__name__)

DEFAULT_PORT = 9900
IMAGE_PATH = "/embed/image"
# Faces: a frame in, a box and a 512-d identity vector per face out
# (emulator/face.py). Nothing about a face is kept here — the robot stores and
# matches it in its own shard (emulator/face_memory.py).
FACE_PATH = "/embed/face"
SPEECH_PATH = "/embed/speech"
HEALTH_PATH = "/health"

# A 640x480 JPEG is tens of kilobytes; the cap refuses a body that could only
# be a mistake without getting in the way of a real frame.
MAX_BODY = 16 * 1024 * 1024


class BadRequest(Exception):
    """A request this service cannot read: answered 400. A failure inside a
    model is the service's own, answered 500 with a traceback in the log."""


def _field(body: dict, name: str):
    try:
        return body[name]
    except (KeyError, TypeError) as exc:
        raise BadRequest(f"missing field {name!r}") from exc


def _frame(body: dict) -> np.ndarray:
    """The request's JPEG as an RGB array."""
    import io

    try:
        jpeg = base64.b64decode(_field(body, "jpeg_b64"), validate=True)
        return np.asarray(Image.open(io.BytesIO(jpeg)).convert("RGB"),
                          dtype=np.uint8)
    except BadRequest:
        raise
    except Exception as exc:  # noqa: BLE001 — bad base64, not an image, too big
        raise BadRequest(f"not a readable JPEG: {type(exc).__name__}: {exc}") from exc


class FacesOff(RuntimeError):
    """The face models could not be loaded; answered 503, with why."""


class Embedders:
    """Both embedders, loaded once and shared across requests.

    Loading is LAZY per embedder: a run that only ever asks for speech vectors
    should not pay SigLIP's load, and the robot's first frame arrives long
    after its first utterance.
    """

    def __init__(self) -> None:
        self._siglip = None
        self._bge = None
        self._faces = None
        self._face_boxes = None
        self.faces_off: str | None = None

    def siglip(self):
        if self._siglip is None:
            from emulator.frame_memory import SiglipEmbedder

            LOG.info("embed_service: loading SigLIP...")
            self._siglip = SiglipEmbedder()
        return self._siglip

    def bge(self):
        if self._bge is None:
            from emulator.memory import DEFAULT_MODEL, _embedder

            LOG.info("embed_service: loading bge...")
            self._bge = _embedder(DEFAULT_MODEL)
        return self._bge

    def faces(self):
        """The face models — or FacesOff, why, from the first failure on: they
        are optional, and a frame a second asking again would only repeat it."""
        if self.faces_off:
            raise FacesOff(self.faces_off)
        if self._faces is None:
            from emulator.face import FaceReader

            LOG.info("embed_service: loading the face models...")
            try:
                self._faces = FaceReader()
            except Exception as exc:  # noqa: BLE001 — optional, see above
                self.faces_off = public(f"{type(exc).__name__}: {exc}")
                LOG.warning("embed_service: faces off — %s", self.faces_off,
                            exc_info=True)
                raise FacesOff(self.faces_off) from exc
        return self._faces

    def face_boxes(self):
        """Where the faces are, without who: the robot's own detect loop
        (`--on-robot detector`) keeps its head on a face with these. YuNet
        alone, as demo/detect_service.py reads them — faces off must not take
        the head tracking, and the objects found with it, along."""
        if self._faces is not None:
            return self._faces
        if self._face_boxes is None:
            from emulator.face import FaceReader

            self._face_boxes = FaceReader(identities=False)
        return self._face_boxes

    def warm(self) -> None:
        """Pay both load costs before the robot is waiting on a turn."""
        self.siglip().embed_image(np.zeros((64, 64, 3), np.uint8))
        next(iter(self.bge().embed(["warm"])))
        try:
            self.faces()
        except FacesOff:
            # The face models are optional: everything else works without
            # them, and saying so once here (faces() did) beats failing on
            # the first face.
            pass


def make_handler(embedders: Embedders):
    class Handler(BaseHTTPRequestHandler):
        timeout = 60

        def _read_json(self) -> dict:
            try:
                length = int(self.headers.get("Content-Length", 0))
            except ValueError as exc:
                raise BadRequest(f"bad Content-Length: {exc}") from exc
            if length <= 0 or length > MAX_BODY:
                raise BadRequest("request body out of bounds")
            try:
                body = json.loads(self.rfile.read(length))
            except ValueError as exc:
                raise BadRequest(f"bad JSON: {exc}") from exc
            if not isinstance(body, dict):
                raise BadRequest("the body must be a JSON object")
            return body

        def _send_json(self, payload: dict, status: int = 200) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            try:
                if self.path == IMAGE_PATH:
                    self._embed_image()
                elif self.path == SPEECH_PATH:
                    self._embed_speech()
                elif self.path == FACE_PATH:
                    self._embed_face()
                else:
                    self.send_error(404)
            except BadRequest as exc:
                LOG.warning("embed_service: bad request to %s: %s",
                            self.path, exc)
                self.send_error(400, ascii_reason(str(exc)))
            except FacesOff as exc:  # said once, when they failed to load
                self.send_error(503, ascii_reason(f"faces off: {exc}"))
            except Exception as exc:  # noqa: BLE001 — the service must survive
                LOG.exception("embed_service: %s", type(exc).__name__)
                try:
                    self.send_error(500, ascii_reason(str(exc)))
                except Exception:
                    pass

        def do_GET(self):
            if self.path == HEALTH_PATH:
                # The models this service embeds with, so the robot can
                # check them against the ones that wrote its memory
                # (emulator/embed_identity.py) without guessing from a
                # version number or a path. And whether it has faces, so the
                # robot turns them off instead of asking every frame.
                from emulator.embed_identity import current as _embedders

                self._send_json({"healthy": True, "embedders": _embedders(),
                                 "faces_off": embedders.faces_off})
            else:
                self.send_error(404)

        def _embed_image(self):
            """A JPEG in, a SigLIP frame vector out."""
            frame = _frame(self._read_json())
            vector = embedders.siglip().embed_image(frame)
            self._send_json({"vector": [float(x) for x in vector]})

        def _embed_face(self):
            """A JPEG in, every face out: box in frame fractions, detector
            score, and the identity vector the robot matches against the
            people it has met."""
            body = self._read_json()
            frame = _frame(body)
            embed = body.get("embed", True)
            reader = embedders.faces() if embed else embedders.face_boxes()
            faces = reader.read(frame, embed=embed)
            self._send_json({"faces": [
                {"box": face.box, "score": face.score,
                 "embedding": face.embedding} for face in faces]})

        def _embed_speech(self):
            """An utterance in, a bge vector out. `query=true` uses the
            query-side embedding, which bge trains differently from the
            document side — mixing them up quietly degrades recall."""
            body = self._read_json()
            text = str(_field(body, "text"))
            bge = embedders.bge()
            embed = bge.query_embed if body.get("query") else bge.embed
            vector = next(iter(embed([text])))
            self._send_json({"vector": [float(x) for x in vector]})

        def log_request(self, code="-", size="-"):
            if isinstance(code, int) and 200 <= code < 300:
                return
            super().log_request(code, size)

    return Handler


def main(argv=None) -> int:
    from demo.logs import setup_logging

    setup_logging()
    p = argparse.ArgumentParser(
        description="Embedding models for the robot (SigLIP + bge + faces)")
    # Bound wide by default: the robot reaches it across the network. See the
    # README's security section — this service has no authentication.
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--no-warmup", action="store_true",
                   help="skip loading the models at startup")
    args = p.parse_args(argv)

    embedders = Embedders()
    # Warm BEFORE binding: ThreadingHTTPServer's constructor itself calls
    # socket.bind()+listen(), so a server built first would sit there
    # ACCEPTING connections — the OS completes the TCP handshake — while
    # loading models, for however long that takes on a cold cache. The
    # robot's memories embed a probe in their own constructors
    # (demo/run_demo.py's build_memories), so they would connect and then
    # hang until their timeout, disabling the memory for the whole run,
    # instead of failing fast with a clean connection-refused.
    if not args.no_warmup:
        print("  loading models (SigLIP + bge)...", flush=True)
        embedders.warm()
        print("  ready", flush=True)
    server = ThreadingHTTPServer((args.host, args.port),
                                 make_handler(embedders))
    print(f"embed_service on {args.host}:{args.port}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
