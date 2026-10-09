"""Tests for demo/embed_service.py's main(): the warm-before-bind ordering
fix. No real model load, no real socket
— ThreadingHTTPServer and Embedders are both replaced with recording fakes.
"""
from __future__ import annotations


class _FakeServer:
    """Stands in for ThreadingHTTPServer: main() must not actually bind a
    socket or block on serve_forever() in a test."""

    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def serve_forever(self) -> None:
        pass


def test_main_warms_models_before_binding_the_socket(monkeypatch):
    """Regression test: ThreadingHTTPServer's constructor itself calls
    socket.bind()+listen(), so building it BEFORE embedders.warm() leaves the
    port accepting connections while the models are still loading — a robot
    connecting to embed_service in that window gets a request queued behind
    a server that never accepts it, instead of a clean, fast connection
    refusal. warm() must complete before the socket exists at all.
    """
    import demo.embed_service as embed_service_mod

    order = []

    class RecordingEmbedders:
        def warm(self):
            order.append("warm")

    def fake_server(*_args, **_kwargs):
        order.append("bind")
        return _FakeServer()

    monkeypatch.setattr(embed_service_mod, "Embedders", RecordingEmbedders)
    monkeypatch.setattr(embed_service_mod, "ThreadingHTTPServer", fake_server)

    embed_service_mod.main([])

    assert order == ["warm", "bind"]


def test_main_no_warmup_skips_warm_but_still_binds(monkeypatch):
    import demo.embed_service as embed_service_mod

    order = []

    class RecordingEmbedders:
        def warm(self):
            order.append("warm")

    def fake_server(*_args, **_kwargs):
        order.append("bind")
        return _FakeServer()

    monkeypatch.setattr(embed_service_mod, "Embedders", RecordingEmbedders)
    monkeypatch.setattr(embed_service_mod, "ThreadingHTTPServer", fake_server)

    embed_service_mod.main(["--no-warmup"])

    assert order == ["bind"]


# — faces off —

def _no_face_models(monkeypatch):
    """HSFace missing; YuNet, from the Hub, there — boxes, no identities."""
    from pathlib import Path

    from emulator import face

    loads = []

    class Boxes:
        def read(self, frame, *, embed=True):
            assert not embed
            return [face.Face(box=[0.4, 0.3, 0.6, 0.7], score=0.9, embedding=None)]

    def missing(*args, identities=True, **kwargs):
        if not identities:
            return Boxes()
        loads.append(True)
        raise FileNotFoundError(f"{Path.home()}/repo/assets/hsface10k.tflite is missing")

    monkeypatch.setattr(face, "FaceReader", missing)
    return loads


def test_face_models_that_fail_to_load_are_off_and_not_tried_every_frame(monkeypatch):
    import pytest

    from demo.embed_service import Embedders, FacesOff

    loads = _no_face_models(monkeypatch)
    embedders = Embedders()
    for _ in range(3):
        with pytest.raises(FacesOff, match="hsface10k.tflite is missing"):
            embedders.faces()
    assert loads == [True]
    assert "hsface10k.tflite is missing" in embedders.faces_off


def test_faces_off_is_answered_503_with_why_and_said_in_health(monkeypatch, caplog):
    import base64
    import io
    import json
    import threading
    import urllib.error
    import urllib.request
    from http.server import ThreadingHTTPServer
    from pathlib import Path

    from PIL import Image

    from demo.embed_service import (FACE_PATH, HEALTH_PATH, Embedders,
                                    make_handler)

    _no_face_models(monkeypatch)
    embedders = Embedders()
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(embedders))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        jpeg = io.BytesIO()
        Image.new("RGB", (8, 8)).save(jpeg, format="JPEG")
        request = urllib.request.Request(
            base + FACE_PATH, method="POST",
            data=json.dumps({"jpeg_b64": base64.b64encode(jpeg.getvalue()).decode()}).encode(),
            headers={"Content-Type": "application/json"})
        with caplog.at_level("WARNING", logger="demo.embed_service"):
            for _ in range(2):
                try:
                    urllib.request.urlopen(request, timeout=5)
                    raise AssertionError("a face request without face models answered")
                except urllib.error.HTTPError as exc:
                    assert exc.code == 503
                    assert "faces off" in exc.reason
                    assert str(Path.home()) not in exc.reason, "no user name on the wire"
        said = [r for r in caplog.records if "faces off" in r.getMessage()]
        assert len(said) == 1, "said once, when they failed to load"
        assert [r for r in caplog.records if r.exc_info] == said, "no traceback per frame"
        with urllib.request.urlopen(base + HEALTH_PATH, timeout=5) as resp:
            off = json.loads(resp.read())["faces_off"]
            assert "~/repo/assets/hsface10k.tflite is missing" in off
            assert str(Path.home()) not in off
        # The robot's own detect loop asks for boxes only: those it still gets.
        boxes = urllib.request.Request(
            base + FACE_PATH, method="POST",
            data=json.dumps({"jpeg_b64": base64.b64encode(jpeg.getvalue()).decode(),
                             "embed": False}).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(boxes, timeout=5) as resp:
            assert json.loads(resp.read())["faces"] == [
                {"box": [0.4, 0.3, 0.6, 0.7], "score": 0.9, "embedding": None}]
    finally:
        server.shutdown()
        server.server_close()


def test_warm_pays_both_load_costs_and_survives_faces_off():
    # SigLIP is warmed through its picture side, the only one there is now.
    import numpy as np

    from demo.embed_service import Embedders

    called = []

    class Pictures:
        def embed_image(self, frame):
            called.append(("image", np.asarray(frame).shape))
            return np.zeros(3, np.float32)

    class Words:
        def embed(self, texts):
            called.append(("words", list(texts)))
            return iter([np.zeros(3, np.float32)])

    embedders = Embedders()
    embedders._siglip, embedders._bge = Pictures(), Words()
    embedders.faces_off = "no face models here"
    embedders.warm()
    assert called == [("image", (64, 64, 3)), ("words", ["warm"])]
