#!/usr/bin/env python3
"""Check the demo can run, before an audience is watching.

Every piece of this demo has been verified on its own; this checks them
together, in the order a turn actually uses them, and says which one is
broken rather than leaving you to infer it from a robot that just sits there.

    uv run python scripts/preflight.py                    # everything
    uv run python scripts/preflight.py --skip-models      # fast, no model loads

Exit code is 0 only if nothing is broken. Warnings (a robot that is not
plugged in, say) do not fail the run — they are reported and you decide.
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

OK, WARN, FAIL = "ok", "warn", "fail"
_MARK = {OK: "  ok  ", WARN: " warn ", FAIL: " FAIL "}

results: list[tuple[str, str, str]] = []


def report(status: str, name: str, detail: str = "") -> None:
    results.append((status, name, detail))
    print(f"[{_MARK[status]}] {name}" + (f" — {detail}" if detail else ""), flush=True)


def _get(url: str, timeout: float = 4.0) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, b""


def check_robot(host: str) -> bool:
    """The robot's daemon: reachable, and what it thinks it is holding."""
    try:
        status, body = _get(f"http://{host}:8000/api/media/status")
    except OSError as exc:
        report(WARN, "robot daemon", f"{host}:8000 unreachable ({exc}) — "
               "is it powered on and on this network?")
        return False
    if status != 200:
        report(FAIL, "robot daemon", f"HTTP {status} from {host}:8000")
        return False
    media = json.loads(body or b"{}")
    report(OK, "robot daemon", f"{host}:8000 answering")
    if media.get("available"):
        # This is the one that silently ruins the camera: while the daemon
        # holds media, rpicam cannot open the camera at all, so the camera
        # service will start and then never produce a frame.
        report(WARN, "robot media", "the daemon HOLDS the camera "
               "(available=true) — the camera service cannot start while it "
               "does; POST /api/media/release first")
    else:
        report(OK, "robot media", "released — the camera service can take it")
    return True


def check_camera_service(host: str, port: int) -> None:
    """The service that runs ON the robot and serves its camera and mic."""
    from demo.camera_service import AUDIO_PATH, FRAME_PATH, HEALTH_PATH

    base = f"http://{host}:{port}"
    try:
        status, body = _get(base + HEALTH_PATH)
    except OSError:
        report(WARN, "camera service", f"not running at {base} — start it on "
               f"the robot (scripts/robot_service.sh start)")
        return
    health = json.loads(body or b"{}")
    if status == 200 and health.get("healthy"):
        report(OK, "camera service", f"healthy, frame age "
               f"{health.get('frame_age_s', 0):.2f}s")
    else:
        report(FAIL, "camera service", f"unhealthy: {health.get('last_error')}")

    try:
        status, jpeg = _get(base + FRAME_PATH, timeout=5.0)
        if status == 200 and jpeg[:2] == b"\xff\xd8":
            report(OK, "robot camera", f"{len(jpeg)} byte frame")
        else:
            report(FAIL, "robot camera", f"HTTP {status}, {len(jpeg)} bytes")
    except OSError as exc:
        report(FAIL, "robot camera", str(exc))

    # The mic is a stream, so read just enough to prove audio is flowing.
    try:
        with urllib.request.urlopen(base + AUDIO_PATH, timeout=5.0) as resp:
            rate = resp.headers.get("X-Sample-Rate")
            chunk = resp.read(8000)
        if len(chunk) >= 8000:
            report(OK, "robot mic", f"streaming at {rate} Hz")
        else:
            report(FAIL, "robot mic", f"only {len(chunk)} bytes arrived")
    except OSError as exc:
        report(FAIL, "robot mic", str(exc))


def check_service(name: str, host: str, port: int, hint: str) -> None:
    """A local service we expect to have started ourselves."""
    with socket.socket() as sock:
        sock.settimeout(2.0)
        if sock.connect_ex((host, port)) == 0:
            report(OK, name, f"{host}:{port} listening")
        else:
            report(WARN, name, f"{host}:{port} not listening — {hint}")


def check_assets() -> None:
    """Every model in the catalog (emulator/models.py), fetched now if it is
    not in the cache yet — a download belongs before the show, not in it."""
    from emulator import models

    for name, spec in models.MODELS.items():
        try:
            path = models.fetch(spec)
        except Exception as exc:  # noqa: BLE001 — reported, the run goes on
            # The face embedder is optional: without it the robot talks and
            # remembers, it just calls everyone "Person".
            status = WARN if name == "hsface" else FAIL
            report(status, name, f"{type(exc).__name__}: {exc}")
            continue
        size = (sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
                if path.is_dir() else path.stat().st_size)
        report(OK, name, f"{size / 1e6:.1f} MB")


def check_models() -> None:
    """Load what a turn actually loads. Slow on purpose — the first vision
    call costs ~3s of executor init, and finding that out mid-demo is worse."""
    t0 = time.time()
    try:
        from emulator.frame_memory import SiglipEmbedder
        import numpy as np

        emb = SiglipEmbedder()
        emb.embed_image(np.zeros((64, 64, 3), np.uint8))
        report(OK, "SigLIP (visual memory)", f"loaded in {time.time() - t0:.1f}s")
    except Exception as exc:  # noqa: BLE001
        report(FAIL, "SigLIP (visual memory)", f"{type(exc).__name__}: {exc}")

    t0 = time.time()
    try:
        from emulator.memory import TextMemory

        TextMemory(path=None).remember("Person: hi — Reachy: hello")
        report(OK, "bge (speech memory)", f"loaded in {time.time() - t0:.1f}s")
    except Exception as exc:  # noqa: BLE001
        report(FAIL, "bge (speech memory)", f"{type(exc).__name__}: {exc}")


def check_robot_motion(host: str) -> None:
    """Ask the robot to do something small and visible. If this works, the
    whole actuation path works — it is the same HTTP call a tool call makes."""
    try:
        req = urllib.request.Request(
            f"http://{host}:8000/api/move/goto",
            data=json.dumps({"antennas": [0.4, -0.4], "duration": 0.3}).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        urllib.request.urlopen(req, timeout=5.0).read()
        time.sleep(0.5)
        req = urllib.request.Request(
            f"http://{host}:8000/api/move/goto",
            data=json.dumps({"antennas": [0.0, 0.0], "duration": 0.3}).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        urllib.request.urlopen(req, timeout=5.0).read()
        report(OK, "robot motion", "antennas moved — watch the robot")
    except OSError as exc:
        report(FAIL, "robot motion", str(exc))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Pre-demo check")
    p.add_argument("--robot-host", default="reachy-mini.local")
    p.add_argument("--robot-camera-port", type=int, default=9700)
    p.add_argument("--serve-port", type=int, default=9500)
    p.add_argument("--detect-port", type=int, default=9600)
    p.add_argument("--skip-models", action="store_true",
                   help="skip the slow model loads")
    p.add_argument("--no-motion", action="store_true",
                   help="do not move the robot")
    args = p.parse_args(argv)

    print("\n--- files and weights ---")
    check_assets()

    print("\n--- services on this laptop ---")
    check_service("models (serve.py)", "127.0.0.1", args.serve_port,
                  "uv run python -m demo.serve")
    check_service("detector (detect_service)", "127.0.0.1", args.detect_port,
                  "uv run python -m demo.detect_service")

    print("\n--- the robot ---")
    if check_robot(args.robot_host):
        check_camera_service(args.robot_host, args.robot_camera_port)
        if not args.no_motion:
            check_robot_motion(args.robot_host)

    if not args.skip_models:
        print("\n--- models (slow) ---")
        check_models()

    failed = [r for r in results if r[0] == FAIL]
    warned = [r for r in results if r[0] == WARN]
    print(f"\n{len(results)} checks: {len(results) - len(failed) - len(warned)} ok, "
          f"{len(warned)} warnings, {len(failed)} failed")
    for _, name, detail in failed:
        print(f"  FAILED: {name} — {detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
