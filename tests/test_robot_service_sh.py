"""Regression tests for scripts/robot_service.sh's `voice-start` health gate.

`voice-start` used to check ONLY the camera+mic service's top-level /health
(backed by CameraCapture.status()) before starting the voice loop, never
demo/camera_service.py's separate MIC_HEALTH_PATH (backed by
MicCapture.status() — camera and mic health are tracked independently
there, see its own module docstring). A dead `arecord` with a healthy
`rpicam-vid` passed the gate, the voice loop started, and
RobotMicSource (demo/platform/robot_audio.py) fed it silent chunks forever
BY DESIGN — a running, apparently-healthy voice loop that could never hear
anything, with nothing printed anywhere.

Parses the script's TEXT rather than executing it: these tests must not
touch the network or require the robot, and a shell script has no
importable surface to call into directly. The one exception, at the end,
runs a check that happens before the script reaches for ssh.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from demo.camera_service import MIC_HEALTH_PATH

SCRIPT = (Path(__file__).resolve().parents[1] / "scripts" / "robot_service.sh").read_text()


def _prepare_block() -> str:
    """The function that installs and carries what a local placement needs —
    shared by `voice-start` and the `prepare` action."""
    start = SCRIPT.index("prepare_placement() {")
    return SCRIPT[start:SCRIPT.index("\n}\n", start)]


def _voice_start_block() -> str:
    start = SCRIPT.index("\n  voice-start)")
    end = SCRIPT.index("\n  voice-stop)", start)
    return SCRIPT[start:end]


def test_voice_start_sets_full_volume_and_reads_it_back():
    """ALSA forgets the level on a reboot: after the battery ran out once the
    robot came back at 62%, and the volume had only ever been set by hand.
    Every voice-start sets both PCM controls to 100% and reports what it read
    back, before the loop starts."""
    block = _voice_start_block()
    assert "amixer -c 0 sset PCM,0 100%" in block
    assert "amixer -c 0 sset PCM,1 100%" in block
    assert "amixer -c 0 sget PCM,0" in block and "amixer -c 0 sget PCM,1" in block
    assert 'echo "volume 100%"' in block
    assert "WARNING: volume is" in block
    assert block.index("sset PCM,0 100%") < block.index("voice_loop.sh' '$MEMORY_DIR'")


def test_voice_start_runs_the_loop_under_the_restart_supervisor():
    """The dashboard's Restart button ends the loop with an exit code; only
    scripts/voice_loop.sh turns that into a wipe and a fresh start. Started
    bare, the button would stop the demo and never bring it back."""
    block = _voice_start_block()

    assert "voice_loop.sh" in block, "the loop must run under the supervisor"
    started = re.search(r"voice_loop\.sh' '\$MEMORY_DIR'", block)
    assert started, "the supervisor is given the memory directory to wipe"
    assert started.start() < block.index("-m demo.run_demo"), (
        "the supervisor wraps the loop: voice_loop.sh <memory-dir> <command…>")
    assert "scp" in block and "voice_loop.sh" in block, (
        "the supervisor lives in scripts/, which the rsync of demo/ and "
        "emulator/ does not carry — it has to be copied over on its own")


def test_voice_start_gates_on_mic_health_not_just_camera_health():
    block = _voice_start_block()

    assert MIC_HEALTH_PATH in block, (
        "voice-start must curl the mic's own health endpoint, not rely on "
        "the camera's /health alone"
    )


def test_voice_start_checks_mic_health_before_deploying_the_voice_loop():
    block = _voice_start_block()

    mic_check_idx = block.index(MIC_HEALTH_PATH)
    deploy_idx = block.index("rsync")

    assert mic_check_idx < deploy_idx, (
        "the mic health gate must run before the voice loop is deployed/started"
    )


def test_voice_start_checks_mic_health_after_camera_health():
    # Ordering isn't load-bearing for correctness, but mirrors the existing
    # camera health check immediately above it (see the script's own
    # comments) rather than being bolted on somewhere unrelated.
    block = _voice_start_block()

    camera_check_idx = block.index('/health"')
    mic_check_idx = block.index(MIC_HEALTH_PATH)

    assert camera_check_idx < mic_check_idx


def test_voice_start_mic_health_check_fails_loudly_on_bad_grep():
    # The gate must actually grep for a healthy response and exit non-zero
    # on failure, matching the camera health gate's own shape, not merely
    # curl the endpoint and ignore the result.
    block = _voice_start_block()
    mic_line_start = block.index(MIC_HEALTH_PATH)
    # The `if ! ... ; then ... exit 1; fi` guard immediately preceding/
    # following the curl — look at a window around the health check.
    window = block[max(0, mic_line_start - 200):mic_line_start + 300]

    assert '"healthy": true' in window
    assert "exit 1" in window


# --- `stop` must actually stop things, and say so truthfully ----------------
# Both regressions below were reproduced live on the robot, and
# neither is visible to a test that only checks the happy path: `stop` exited
# 255 having killed its own ssh session before freeing the camera, and
# `status` printed "nothing running" with two matching processes up.


def _block(name: str) -> str:
    start = SCRIPT.index(f"\n  {name})")
    return SCRIPT[start:SCRIPT.index("\n    ;;", start)]


def _code(block: str) -> str:
    """Only the lines that become a REMOTE command line. Comments never leave
    the Mac, and `start`'s comment quotes an unbracketed pattern deliberately,
    to explain the very trap these tests guard.
    """
    return "\n".join(
        line for line in block.splitlines() if not line.strip().startswith("#"))


# A pkill pattern is bracketed ('[d]emo') so it cannot match the command line
# it is travelling in. That protects the pkill's own copy of the literal —
# and nothing else: a SECOND, unbracketed copy anywhere on the same line
# (`stop` had one, in its trailing pgrep) is still a match, and the pkill
# kills the shell running it.
KILL_TRAPS = ("demo.run_demo", "camera_service.py", "rpicam-vid", "arecord -D")


@pytest.mark.parametrize("block_name", ["stop", "voice-stop"])
def test_kill_blocks_carry_no_unbracketed_process_literal(block_name):
    code = _code(_block(block_name))

    for trap in KILL_TRAPS:
        assert trap not in code, (
            f"{block_name!r} puts the unbracketed literal {trap!r} on the same "
            f"remote command line as a pkill — the pkill matches the shell "
            f"running it and kills the ssh session before the later clauses run"
        )


def test_shared_lookup_patterns_are_bracketed():
    # These are interpolated INTO the pkill command lines above, so they carry
    # the same constraint as anything else written there.
    for line in SCRIPT.splitlines():
        if line.startswith(("ROBOT_PROCS=", "VOICE_PROC=")):
            for trap in KILL_TRAPS:
                assert trap not in line, (
                    f"{line.split('=')[0]} must bracket {trap!r}: it is "
                    f"expanded into command lines that also run pkill"
                )


def test_every_process_lookup_matches_the_full_command_line():
    # The voice loop and the camera+mic service both run as `python3 ...`, so
    # the process NAME is "python3" and a name-only `pgrep -a` can never match
    # them — pgrep itself warns that a >15-character name pattern "will result
    # in zero matches". Verified live: with two decoys up, `pgrep -a` on the
    # old pattern said "nothing running" while `pgrep -af` found both.
    code = _code(SCRIPT)
    flags = re.findall(r"pgrep\s+(-\S+)", code)

    assert flags, "expected the script to look up processes with pgrep"
    for flag in flags:
        assert "f" in flag, (
            f"pgrep {flag} matches the process NAME only; these processes are "
            f"all 'python3', so it can never match. Use -f."
        )


# --- ON_ROBOT: the placement has to reach the voice loop, and the gate has
# to follow it ---

def test_voice_start_forwards_the_placement_to_the_voice_loop():
    """A placement that cannot travel through this script cannot be used in
    the live demo at all: `voice-start` takes its action and builds a fixed
    command line, so --on-robot has to be part of that line."""
    block = _voice_start_block()
    assert "--on-robot" in block, "the voice loop must be told where models run"
    assert "${ON_ROBOT:+--on-robot" in block, (
        "with nothing set, no flag is passed and the run behaves as it does today")


def test_the_placement_defaults_to_empty():
    assert 'ON_ROBOT="${ON_ROBOT:-}"' in SCRIPT, (
        "unset means every model on the Mac — today's behaviour")


def test_the_embed_service_gate_is_skipped_when_the_embeddings_are_local():
    """The gate exists because FrameMemory/TextMemory round-trip to
    embed_service inside their constructors and silently disable themselves
    when it is missing. With the embeddings on the robot there is no round
    trip to protect, and a stopped embed_service is the expected state."""
    block = _voice_start_block()
    guard = block.index('grep -q \',embedder,\'')
    gate = block.index("embed_service is not healthy")
    assert guard < gate, "the placement is checked before the gate refuses"
    assert "not checking embed_service" in block


def test_the_embed_service_gate_still_refuses_when_the_embeddings_are_remote():
    block = _voice_start_block()
    assert "embed_service is not healthy at $BRAIN:$EMBED_PORT" in block
    assert "exit 1" in block[block.index("embed_service is not healthy"):]


# --- what a local placement needs, delivered by the deploy ---

def test_the_demo_venv_is_built_over_the_daemons_packages_not_inside_it():
    """LiteRT pulls its own numpy, and /venvs/mini_daemon is where the robot's
    own daemon lives — installing there would change the versions it runs on."""
    block = _prepare_block()
    assert "--system-site-packages" in block
    assert "$DAEMON_PY' -m venv" in block
    assert "/venvs/mini_daemon" not in block.split("-m pip install")[0][-400:], (
        "nothing is installed into the daemon's own venv")


def test_the_runtime_is_installed_only_when_it_is_missing():
    # voice-start runs before every take; reinstalling each time would spend
    # the talk doing it.
    block = _prepare_block()
    assert "import ai_edge_litert' 2>/dev/null" in block
    assert "pip install -q ai-edge-litert" in block


def test_each_family_brings_what_it_calls():
    block = _prepare_block()
    assert "opencv-python-headless" in block, "the face detector is OpenCV's YuNet"
    assert "espeak-ng" in block and "phonemizer" in block, "synthesis needs its frontend"


def test_only_the_named_families_models_are_carried():
    block = _prepare_block()
    for family, model in (("detector", "yolo26n"), ("faces", "yunet"),
                          ("asr", "moonshine-tokenizer"), ("asr", "silero-vad"),
                          ("tts", "inflect-nano-v2")):
        case = block.index(f"*,{family},*")
        assert model in block[case:case + 700], f"{family} must bring {model}"
    faces = block.rindex("*,faces,*")
    assert "hsface10k.tflite" in block[faces:faces + 700], (
        "the face embedder is the one model not on the Hub")
    assert "emulator.models hsface" in block[faces:faces + 700], (
        "and it is built on the laptop first, not asked of the person")
    built = block.index("emulator.models hsface", faces)
    carried = block.index("assets/hsface10k.tflite", faces)
    assert built < block.index("exit 1", built) < carried, (
        "a model that could not be built stops the deploy before the copy")
    assert "carried to the robot:" in block, "the deploy says what it sent"


def test_the_loop_runs_under_the_venv_when_models_are_local():
    assert "'$RUN_PY' -u -m demo.run_demo" in _voice_start_block()
    assert 'RUN_PY="$DAEMON_PY"' in _prepare_block(), (
        "with nothing placed locally the loop runs exactly as it does today")
    assert "prepare_placement" in _voice_start_block(), (
        "voice-start prepares before it starts")


def test_the_preparation_can_be_rehearsed_on_its_own():
    """A deploy that first downloads a runtime during the talk is a deploy
    nobody rehearsed: `prepare` does the installing and carrying, and starts
    nothing."""
    assert "\n  prepare)" in SCRIPT
    action = SCRIPT[SCRIPT.index("\n  prepare)"):SCRIPT.index("\n  voice-stop)")]
    assert "prepare_placement" in action
    assert "voice_loop.sh" not in action, "it starts no loop"
    assert "amixer" not in action and "curl" not in action, (
        "it does not wake the robot or touch its volume either")


def test_a_restart_waits_for_the_old_loop_to_close_its_memory():
    # A new loop started while the old one still flushes its context finds
    # the shard locked and runs the whole session without memory.
    text = SCRIPT
    helper = text[text.index("stop_voice_loop() {"):]
    helper = helper[:helper.index("\n}\n")]
    assert "pgrep -f" in helper and "sleep 0.5" in helper
    start = text[text.index("  voice-start)"):]
    assert "stop_voice_loop" in start[:start.index("setsid --fork")]


def test_the_voice_loop_accepts_the_command_line_voice_start_gives_it():
    # Two programs, deployed together: a flag the loop stops accepting breaks
    # voice-start only on the robot, where the log is the only witness.
    import re
    import shlex

    from demo.run_demo import parse_args

    marker = "'$RUN_PY' -u -m demo.run_demo"
    start = SCRIPT.index(marker) + len(marker)
    command = SCRIPT[start:SCRIPT.index(">>", start)]
    command = command.replace("\\\n", " ")
    command = re.sub(r"\$\{ON_ROBOT:\+([^}]*)\}", r"\1", command)
    command = re.sub(r"\$[A-Z_]+", "9000", command)
    args = parse_args(shlex.split(command))
    assert (args.camera, args.mic, args.speaker) == ("robot", "robot", "robot")
    assert args.display == "remote" and args.on_robot == "9000"


# --- values pasted into the robot's shell ---
# The one test here that runs the script: the check is the first thing it
# does, before any ssh, and `ssh` on PATH is a stand-in that fails the test
# if it is ever reached.

@pytest.mark.parametrize("name, value", [
    ("MEMORY_DIR", "x'; touch /tmp/pwned; '"),
    ("BRAIN_PORT", "9500; reboot"),
    ("ON_ROBOT", "asr tts"),
])
def test_a_value_that_would_break_out_of_the_remote_command_is_refused(tmp_path, name, value):
    import os
    import subprocess

    (tmp_path / "ssh").write_text("#!/bin/sh\necho reached ssh; exit 99\n")
    (tmp_path / "ssh").chmod(0o755)
    env = {**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}", name: value}
    script = Path(__file__).resolve().parents[1] / "scripts" / "robot_service.sh"
    result = subprocess.run(["bash", str(script), "status"], env=env,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 2
    assert name in result.stderr
    assert "reached ssh" not in result.stdout + result.stderr


def test_the_robot_is_carried_the_detector_the_code_loads():
    # Change the detector in emulator/models.py, and the deploy follows: a
    # robot without it would download it mid-demo, or fail offline.
    from emulator import models

    case = SCRIPT.index("*,detector,*)")
    carried = SCRIPT[case:SCRIPT.index(";;", case)].split()[-1].strip('"')
    assert carried == models.DETECTOR
