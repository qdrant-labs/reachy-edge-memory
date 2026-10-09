"""Display package: the DisplaySink protocol, a no-op stub, and a factory.

The voice loop sends events here without knowing where they end up — a
browser (`WebDashboard`) or nowhere (`NullSink`). This removes the
`if webui is not None` branching from the loop: with no display, the methods
simply do nothing.
"""
from __future__ import annotations

from typing import Protocol

# Where the dashboard listens, and where the robot pushes its events. NOT
# 8080: that one is crowded (a Firebase emulator held it on this machine for
# an afternoon), and these services bind 0.0.0.0 — another process on
# 127.0.0.1 keeps the loopback, so the page silently serves someone else
# while the dashboard's own log looks healthy. One constant, imported by the
# dashboard, the voice loop and demo/stage.py, so the three cannot drift.
DEFAULT_DASHBOARD_PORT = 8091


class DisplaySink(Protocol):
    """Presentation interface: the voice loop sends events here."""

    def on_detections(self, detections: list[dict]) -> None:
        """Listener for RemoteDetectSource: fresh boxes every detect cycle."""
        ...

    def on_look(self, jpeg: bytes) -> None:
        """The picture the camera tool just handed the model, as JPEG bytes —
        shown under the tool line, so the room sees what the model saw."""
        ...

    def on_heard(self, text: str) -> None:
        ...

    def on_reply(self, text: str, done: bool = False) -> None:
        ...

    def on_recall(self, hits: list[dict]) -> None:
        """Frames recalled this turn (FrameMemory.recall_text hits, the
        day_frames or a look, each with a jpeg + YOLO detections); the sink
        shows them boxed (see WebDashboard)."""
        ...

    def on_speech_recall(self, hits: list[dict]) -> None:
        """Past utterances recalled this turn (TextMemory.recall_exchanges hits, each
        with `text` + `score`) — search by what was said."""
        ...

    def on_tool_call(self, name: str, arguments: dict) -> None:
        """A tool the model called this turn, as it called it: `remember`,
        `camera` or `move` (demo/conversation.py). Several can arrive in one
        turn."""
        ...

    def on_context(self, tokens: int | None, budget: int, exchanges: int) -> None:
        """How full the model's context is after this turn (None after an
        image turn, which runs in a side chat and reports no count)."""
        ...

    def on_memory_count(self, frames: int, exchanges: int,
                        knowledge: int = 0) -> None:
        """How much the robot's own shards hold, all told: frames and
        exchanges it remembers, and the facts restored from its knowledge
        snapshot."""
        ...

    def on_face(self, name: str | None, box: list[float] | None,
                score: float) -> None:
        """Who the robot is looking at this turn (demo/people.py): the name it
        recognised, or None for someone it has not met, with the face box in
        frame fractions."""
        ...

    def is_paused(self) -> bool:
        """Whether the audience-side pause is on. The voice loop asks before
        each turn — the button is on the Mac, the microphone is on the robot."""
        ...

    def restart_requested(self) -> bool:
        """Whether the Restart button has been pressed: the robot is to wipe
        its memory and start over (demo/run_demo.py's _RestartWatcher). Asked
        on its own thread, not between turns — a quiet robot must answer the
        button too."""
        ...

    def ack_restart(self) -> None:
        """Tell the dashboard the restart is happening, so the loop that comes
        back up does not read the same request again."""
        ...

    def on_memory_write(self, texts: list[str]) -> None:
        """Exchanges that just left the model's context for the robot's
        Qdrant shard (demo/conversation.py's ConversationWindow)."""
        ...

    def close(self) -> None:
        ...


class NullSink:
    """No-op display: without a screen, the loop runs without a single `if display`."""

    def on_detections(self, detections: list[dict]) -> None:
        pass

    def on_look(self, jpeg: bytes) -> None:
        pass

    def on_heard(self, text: str) -> None:
        pass

    def on_reply(self, text: str, done: bool = False) -> None:
        pass

    def on_recall(self, hits: list[dict]) -> None:
        pass

    def on_speech_recall(self, hits: list[dict]) -> None:
        pass

    def on_tool_call(self, name: str, arguments: dict) -> None:
        pass

    def on_context(self, tokens: int | None, budget: int, exchanges: int) -> None:
        pass

    def on_memory_count(self, frames: int, exchanges: int,
                        knowledge: int = 0) -> None:
        pass

    def on_face(self, name, box, score) -> None:
        pass

    def is_paused(self) -> bool:
        return False

    def restart_requested(self) -> bool:
        return False

    def ack_restart(self) -> None:
        pass

    def on_memory_write(self, texts: list[str]) -> None:
        pass

    def close(self) -> None:
        pass


def build_display(kind: str, *, camera, host: str, port: int,
                  dashboard_host: str | None = None,
                  dashboard_port: int = DEFAULT_DASHBOARD_PORT) -> DisplaySink:
    """Display factory keyed by the CLI name (`--display {web,none,remote}`).

    `camera`/`host`/`port` only matter for "web" (a local dashboard server,
    reading `camera` directly for MJPEG); `dashboard_host`/`dashboard_port`
    only matter for "remote" (push events to a WebDashboard running
    elsewhere — see demo/display/remote.py). The voice loop running on the
    robot uses "remote": the audience's screen
    stays on the Mac, at no cost to the robot's own compute, rather than this
    process serving MJPEG/SSE itself over the conference network.
    """
    if kind == "none":
        return NullSink()
    if kind == "web":
        from demo.display.web import WebDashboard

        dash = WebDashboard(camera)   # MJPEG reads the camera directly
        dash.serve(host=host, port=port)
        return dash
    if kind == "remote":
        from demo.display.remote import RemoteDisplayClient

        if not dashboard_host:
            raise ValueError("--display remote needs a dashboard host "
                             "(--dashboard-host or --brain)")
        return RemoteDisplayClient(dashboard_host, dashboard_port)
    raise ValueError(f"unknown display {kind!r}")
