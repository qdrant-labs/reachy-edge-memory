#!/usr/bin/env bash
# Deploy and control what runs ON the robot: the camera+mic service, and the
# voice loop itself.
#
#   scripts/robot_service.sh start        # camera+mic service: copy it over and run it
#   scripts/robot_service.sh stop         # stop EVERYTHING below and free the camera/mic
#   scripts/robot_service.sh status       # what is running, and is it healthy
#   scripts/robot_service.sh logs         # tail the camera+mic service log
#
#   scripts/robot_service.sh voice-start  # voice loop: rsync demo/+emulator/, then run it
#   scripts/robot_service.sh voice-stop   # stop just the voice loop
#   scripts/robot_service.sh voice-status # is it running, tail its log
#   scripts/robot_service.sh voice-logs   # tail -f the voice loop's log
#
# The camera is EXCLUSIVE: while the camera+mic service holds it, the Pollen
# daemon's own face tracking and the Reachy app cannot use it. `stop` gives it
# back — and ALSO stops the voice loop (not just the camera+mic service): an
# orphaned voice loop still holds the robot's Qdrant Edge shard (see
# emulator/edge_store.py — a shard locks its storage directory exclusively)
# and keeps calling the daemon's motor API, and the robot may be in another
# room by the time someone notices — "never leave one running" applies to
# both processes, not just the one actually gripping /dev/video0. Use
# `voice-stop` instead when you want to restart just the loop (e.g. after a
# code change) without disturbing a camera+mic service that's fine as-is.
#
#   ROBOT=reachy-mini.local     # or a plain IP, for networks where mDNS fails
#   PORT=9700                   # camera+mic service port
#
# voice-start ALSO needs to know where the models are (demo/run_demo.py's
# --brain flag), so BRAIN has no default and voice-start fails loudly if it's
# unset rather than silently pointing at the wrong host:
#   BRAIN=<laptop-host-or-ip>   # REQUIRED for voice-start: serve.py/detect_service/embed_service
#   BRAIN_PORT=9500             # demo/serve.py
#   DETECT_PORT=9600            # demo/detect_service.py
#   EMBED_PORT=9900             # demo/embed_service.py
#   WEB_PORT=8091               # the dashboard on the Mac (demo/display/web.py);
#                               # the voice loop pushes its events there, so it
#                               # must match the port the dashboard actually
#                               # listens on (demo/stage.py passes this through)
#   MEMORY_DIR=<remote path>    # default: $REMOTE_DIR/memory (ON THE ROBOT'S OWN DISK —
#                               # this is what makes "memory never leaves the robot" literal)
#   DAEMON_PY=/venvs/mini_daemon/bin/python3   # needs qdrant-edge-py and pillow (README); NOT system python3
#   ON_ROBOT=asr,tts,detector,faces,embedder  # which models run on the robot (default: none)
set -euo pipefail

ROBOT="${ROBOT:-reachy-mini.local}"
USER_AT="${ROBOT_USER:-pollen}@${ROBOT}"
PORT="${PORT:-9700}"
REMOTE_DIR="/home/${ROBOT_USER:-pollen}/reachy-demo"
LOG="${REMOTE_DIR}/service.log"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# — voice loop —
BRAIN_PORT="${BRAIN_PORT:-9500}"
DETECT_PORT="${DETECT_PORT:-9600}"
EMBED_PORT="${EMBED_PORT:-9900}"
WEB_PORT="${WEB_PORT:-8091}"
MEMORY_DIR="${MEMORY_DIR:-${REMOTE_DIR}/memory}"
DAEMON_PY="${DAEMON_PY:-/venvs/mini_daemon/bin/python3}"
# Which models run ON THE ROBOT (demo/placement.py): a comma list of
# asr,tts,detector,faces,embedder; empty (the default) keeps every model on
# the laptop. Passed straight through to demo.run_demo's --on-robot, and read
# below by the embed_service gate, which only makes sense while the
# embeddings are still on the Mac.
ON_ROBOT="${ON_ROBOT:-}"
# The venv the voice loop runs in when it carries models of its own. Built
# OVER the daemon's packages (--system-site-packages), never inside
# /venvs/mini_daemon: LiteRT pulls its own numpy, and the robot's own daemon
# lives in there.
DEMO_VENV="${DEMO_VENV:-${REMOTE_DIR}/venv}"
VOICE_LOG="${REMOTE_DIR}/voice.log"

# Every value above ends up inside a command the robot's shell runs. None of
# them has a reason to hold a quote, a space or a semicolon, so one that does
# is refused here rather than quoted around everywhere it is used.
for name in ROBOT ROBOT_USER PORT BRAIN BRAIN_PORT DETECT_PORT EMBED_PORT \
            WEB_PORT MEMORY_DIR DAEMON_PY ON_ROBOT DEMO_VENV; do
  value="${!name:-}"
  case "$value" in
    *[!A-Za-z0-9._/:,@%+-]*)
      echo "$name has a character this script will not pass to the robot's" \
           "shell: $value" >&2
      exit 2;;
  esac
done

# Patterns for FINDING these processes, kept apart from the `pkill` patterns
# that stop them — and bracketed the same way, for a reason the pkill
# brackets alone did NOT cover. `stop` used to verify itself with an
# unbracketed `pgrep -a 'demo.run_demo|...'` in the SAME remote command line
# as its first `pkill -f '[d]emo\.run_demo'`. The bracket stopped that pkill
# from matching its own pattern, but the plain copy of the literal further
# down the line still matched it, so the pkill killed the remote shell
# running it: ssh exited 255 and NOTHING after the first clause ever ran —
# `stop` never stopped the camera+mic service, rpicam-vid or arecord at all,
# and never printed its "released" line (reproduced live, over
# both mDNS and the bare IP). Any literal added to these command lines has
# to stay bracketed.
#
# `-f`, never a plain `-a`: every process here is some flavour of `python3
# -m demo.run_demo` / `python3 camera_service.py`, so the process NAME is
# just "python3" and a name-only pgrep cannot match — pgrep itself warns
# that a >15-character name pattern "will result in zero matches". Measured
# live with two decoy processes up: `pgrep -a` on the old pattern printed
# "nothing running" while `pgrep -af` on this one found both, i.e. `status`
# could never once have reported a running voice loop or camera service.
ROBOT_PROCS='[d]emo\.run_demo|[c]amera_service\.py|[r]picam-vid|[a]record -D'
VOICE_PROC='[d]emo\.run_demo'

# -n: never read this terminal's stdin. Without it ssh competes with the
# calling script for stdin and a backgrounded remote command can wedge the
# session (seen live: exit 255 and the service never starting).
ssh_robot() { ssh -n -o ConnectTimeout=10 "$USER_AT" "$@"; }


# What a local placement needs on the robot — the runtime for the models it
# will load, the libraries they call, and the weight files themselves. Used by
# `voice-start` before it starts the loop, and by `prepare` on its own, so a
# deploy can be rehearsed (and its downloads paid for) long before a talk.
# Sets RUN_PY: the demo's venv when anything runs on the robot, the daemon's
# python when nothing does.
prepare_placement() {
RUN_PY="$DAEMON_PY"
if [ -n "$ON_ROBOT" ]; then
  RUN_PY="$DEMO_VENV/bin/python3"
  echo "placement: $ON_ROBOT runs on the robot — preparing its venv"
  # --system-site-packages is not enough here: a venv built from another
  # VENV inherits the base interpreter's packages, not its parent's, so
  # the demo would lose onnxruntime/fastembed/qdrant-edge. A .pth adds the
  # daemon's site-packages to the import path instead — shared read-only,
  # nothing installed into it.
  ssh_robot "test -x '$RUN_PY' || '$DAEMON_PY' -m venv --system-site-packages '$DEMO_VENV'"
  ssh_robot "'$DAEMON_PY' -c 'import site,sys;print(site.getsitepackages()[0])' \
             > /tmp/daemon_site && \
             cp /tmp/daemon_site \$('$RUN_PY' -c 'import site;print(site.getsitepackages()[0])')/daemon-packages.pth"
  # Idempotent by design: voice-start runs before every take, and a demo
  # that reinstalled its runtime each time would spend the talk doing it.
  ssh_robot "'$RUN_PY' -c 'import ai_edge_litert' 2>/dev/null \
             || '$RUN_PY' -m pip install -q ai-edge-litert ml_dtypes"
  # The models come from this laptop's Hugging Face cache, fetched here
  # first (emulator/models.py): the robot gets a copy rather than
  # downloading mid-demo, and loads it offline (models.fetch tries the
  # cache first).
  models=""
  case ",$ON_ROBOT," in *,detector,*) models="$models yolo26n";; esac
  case ",$ON_ROBOT," in
    *,faces,*)
      models="$models yunet"
      # The face detector is OpenCV's YuNet (emulator/face.py).
      ssh_robot "'$RUN_PY' -c 'import cv2' 2>/dev/null \
                 || '$RUN_PY' -m pip install -q opencv-python-headless";;
  esac
  case ",$ON_ROBOT," in *,asr,*) models="$models moonshine-tiny moonshine-tokenizer silero-vad";; esac
  case ",$ON_ROBOT," in
    *,embedder,*)
      # SigLIP 2 and bge: fastembed loads bge, and brings the ONNX runtime
      # and huggingface_hub SigLIP's vision tower needs. Both models are
      # downloaded from the Hub by the robot itself, on first use.
      ssh_robot "'$RUN_PY' -c 'import fastembed' 2>/dev/null \
                 || '$RUN_PY' -m pip install -q fastembed";;
  esac
  case ",$ON_ROBOT," in
    *,tts,*)
      models="$models inflect-nano-v2"
      ssh_robot "'$RUN_PY' -c 'import phonemizer, num2words, unidecode' 2>/dev/null \
                 || '$RUN_PY' -m pip install -q phonemizer num2words Unidecode"
      ssh_robot "dpkg -s espeak-ng >/dev/null 2>&1 \
                 || sudo apt-get install -y -q espeak-ng";;
  esac
  if [ -n "$models" ]; then
    repos=$(cd "$REPO" && uv run --quiet python -c "
import sys
from emulator import models
for name in sys.argv[1:]:
    models.fetch(name)
    print('models--' + models.get(name).repo.replace('/', '--'))
" $models)
    ssh_robot "mkdir -p ~/.cache/huggingface/hub"
    for repo in $repos; do
      rsync -a -e "ssh -o ConnectTimeout=10" \
        "$HOME/.cache/huggingface/hub/$repo" "$USER_AT:.cache/huggingface/hub/"
    done
    echo "carried to the robot:$models"
  fi
  case ",$ON_ROBOT," in
    *,faces,*)
      # The face embedder is the one model not on the Hub: built here, on
      # the laptop, the first time (emulator/models.py). 175 MB.
      (cd "$REPO" && uv run --quiet python -m emulator.models hsface) || {
        echo "faces on the robot need the face embedder, and it could not be built" >&2
        exit 1
      }
      ssh_robot "mkdir -p '$REMOTE_DIR/assets'"
      rsync -a -e "ssh -o ConnectTimeout=10" "$REPO/assets/hsface10k.tflite" \
        "$USER_AT:$REMOTE_DIR/assets/"
      echo "carried to the robot: hsface10k.tflite";;
  esac
fi
}


# Stop the voice loop and wait until it has exited — it closes its Qdrant
# Edge shards on the way out, which is what puts the last writes on disk.
stop_voice_loop() {
  ssh_robot "(pkill -f '[d]emo\.run_demo' || true); \
             for i in \$(seq 1 40); do \
               pgrep -f '[d]emo\.run_demo' >/dev/null || { echo 'voice loop stopped'; exit 0; }; \
               sleep 0.5; \
             done; \
             echo 'voice loop did not exit in 20 s' >&2; exit 1"
}


case "${1:-status}" in
  start)
    echo "deploying to ${USER_AT}:${REMOTE_DIR}"
    ssh_robot "mkdir -p '$REMOTE_DIR'"
    scp -q -o ConnectTimeout=10 "$REPO/demo/camera_service.py" \
        "$USER_AT:$REMOTE_DIR/camera_service.py"

    # The daemon and this service cannot both hold the camera. Releasing is
    # idempotent, and doing it here beats a service that starts fine and then
    # never produces a frame.
    ssh_robot "curl -s --max-time 5 -X POST http://127.0.0.1:8000/api/media/release >/dev/null || true"

    # The pattern is bracketed on purpose: `pkill -f camera_service.py` also
    # matches the remote shell running this very command, so it kills its own
    # ssh session (reproduced live — the script died silently right here).
    ssh_robot "(pkill -f '[c]amera_service.py' || true); sleep 1"

    # `setsid --fork` and NO trailing '&': the service must outlive this ssh
    # session, but backgrounding it with '&' inside the remote command made
    # ssh exit 255 with the service never starting (reproduced live).
    # --fork returns immediately and leaves the process in its own session,
    # so it survives the connection closing.
    ssh_robot "cd '$REMOTE_DIR' && PYTHONPATH='$REMOTE_DIR' setsid --fork \
      python3 -u camera_service.py --host 0.0.0.0 --port $PORT \
      > '$LOG' 2>&1 < /dev/null"
    sleep 4

    if ssh_robot "curl -s --max-time 5 http://127.0.0.1:$PORT/health" | grep -q '"healthy": true'; then
      echo "started — http://${ROBOT}:${PORT}/frame"
    else
      echo "started but NOT healthy; last log lines:"
      ssh_robot "tail -15 '$LOG'"
      exit 1
    fi
    ;;

  stop)
    # Voice loop first, camera+mic service second: the loop is the CONSUMER
    # of the service (frames/audio over localhost HTTP) and of the daemon's
    # motor API, so tearing it down first means nothing is still calling out
    # while the thing underneath it disappears. Then the service itself: it
    # supervises the capture processes and would restart one killed out from
    # under it, so it goes before them too.
    ssh_robot "(pkill -f '[d]emo\.run_demo' || true); sleep 1; \
               (pkill -f '[c]amera_service.py' || true); sleep 2; \
               (pkill -f '[r]picam-vid' || true); (pkill -f '[a]record -D' || true); sleep 1; \
               pgrep -af '$ROBOT_PROCS' || echo 'voice loop, camera and mic released'"
    ;;

  status)
    echo "--- processes ---"
    ssh_robot "pgrep -af '$ROBOT_PROCS' || echo '  nothing running'"
    echo "--- camera+mic health ---"
    ssh_robot "curl -s --max-time 5 http://127.0.0.1:$PORT/health || echo '  not answering'"
    echo
    ssh_robot "curl -s --max-time 5 http://127.0.0.1:$PORT/mic/health || echo '  mic: not answering'"
    echo
    echo "--- voice loop: last lines ---"
    ssh_robot "tail -n 10 '$VOICE_LOG' 2>/dev/null || echo '  no voice log yet'"
    ;;

  logs)
    ssh_robot "tail -n ${2:-40} -f '$LOG'"
    ;;

  voice-start)
    # BRAIN has no default (unlike ROBOT/PORT above) on purpose: this points
    # at wherever serve.py/detect_service.py/embed_service.py run (the Mac, in
    # the robot-native architecture — see demo/run_demo.py's --brain), and
    # there's no host that's a safe guess for that. Fail loudly here rather
    # than silently start a voice loop that can never reach its own brain.
    : "${BRAIN:?set BRAIN=<mac-host-or-ip> before voice-start}"

    # The camera+mic service is a SEPARATE process this script also manages
    # (`start`/`stop` above) — the voice loop is one of its CLIENTS (over
    # localhost HTTP, see demo/run_demo.py's CAPTURE decision), not something
    # it starts on its own. Failing loudly here beats a voice loop that comes
    # up, then silently sits with no camera or mic for the whole demo.
    if ! ssh_robot "curl -s --max-time 5 http://127.0.0.1:$PORT/health" | grep -q '"healthy": true'; then
      echo "camera+mic service is not healthy on :$PORT — run '$0 start' first" >&2
      exit 1
    fi

    # The camera and the mic are tracked as SEPARATE health signals inside
    # camera_service.py (CameraCapture.status() / MicCapture.status(), each
    # with its own STALE_AFTER_S) precisely because one can die without the
    # other: the check above only ever proves the CAMERA side is up. Without
    # this gate, a dead `arecord` (the mic's capture process) still leaves
    # /health reporting "healthy": true, this script says "voice loop
    # started", and RobotMicSource (demo/platform/robot_audio.py) then feeds
    # the loop silent chunks forever BY DESIGN (its docstring: a stall/EOF/
    # connect failure yields a SILENT chunk rather than ending the
    # generator) — calibrate_threshold's `floor: float = 0.01`
    # (demo/vad.py) keeps that silence from even mis-triggering, so nothing
    # ever prints. The result is a running, apparently-healthy voice loop
    # that can never hear anything, discovered only when the robot doesn't
    # answer.
    if ! ssh_robot "curl -s --max-time 5 http://127.0.0.1:$PORT/mic/health" | grep -q '"healthy": true'; then
      echo "camera+mic service's MIC is not healthy on :$PORT/mic/health —" \
           "run '$0 start' first, or check arecord on the robot" >&2
      exit 1
    fi

    # embed_service is likewise a SEPARATE process on BRAIN, not something
    # this script starts — but the failure mode here is worse than a missing
    # camera: FrameMemory/TextMemory round-trip to it INSIDE their own
    # constructor to size the shard (see demo/run_demo.py's build_memories
    # docstring), so a voice loop started before embed_service finishes
    # loading SigLIP+bge doesn't just fail loudly — it silently disables BOTH
    # memories for the whole run. Checked FROM THE ROBOT (ssh_robot): that is
    # the network path that actually matters, not whether the machine running
    # this script can reach BRAIN.
    # ...and only while the embeddings ARE on the Mac: with ON_ROBOT naming
    # the embedder there is no round trip to protect, and a stopped
    # embed_service is the expected state of an all-on-robot run.
    if echo ",$ON_ROBOT," | grep -q ',embedder,'; then
      echo "embeddings run on the robot (ON_ROBOT=$ON_ROBOT) — not checking embed_service"
    elif ! ssh_robot "curl -s --max-time 5 http://$BRAIN:$EMBED_PORT/health" | grep -q '"healthy": true'; then
      echo "embed_service is not healthy at $BRAIN:$EMBED_PORT — start it on" \
           "the Mac ('uv run python -m demo.embed_service') and wait for its" \
           "'  ready' line before voice-start" >&2
      exit 1
    fi

    echo "deploying demo/ + emulator/ to ${USER_AT}:${REMOTE_DIR}"
    ssh_robot "mkdir -p '$REMOTE_DIR'"
    # The supervisor the loop runs under (scripts/voice_loop.sh): it restarts
    # the loop and wipes MEMORY_DIR when the dashboard's Restart button ends a
    # run with exit code 42. Copied separately — the rsync below carries demo/
    # and emulator/, not scripts/.
    scp -q -o ConnectTimeout=10 "$REPO/scripts/voice_loop.sh" \
        "$USER_AT:$REMOTE_DIR/voice_loop.sh"
    # --delete: this repo's own tree is the single source of truth for what
    # runs on the robot — a stale module left over from an earlier deploy
    # must not linger and get imported by accident.
    rsync -a --delete -e "ssh -o ConnectTimeout=10" \
      --exclude='__pycache__' --exclude='*.pyc' \
      "$REPO/demo" "$REPO/emulator" "$USER_AT:$REMOTE_DIR/"

    prepare_placement

    # Same bracket trick as camera_service.py above — `pkill -f demo.run_demo`
    # would otherwise also match the remote shell running this pkill. And
    # WAIT for it to be gone: on SIGTERM the old loop flushes its context
    # into memory and closes its shards, and a new one started meanwhile
    # finds the shard locked and runs the whole session without memory.
    stop_voice_loop

    # Wake the robot before it listens. Put to sleep, its motors are disabled,
    # and the voice loop's head turns and gestures would do nothing; the wake
    # move needs the motors enabled first (goto_sleep leaves them disabled).
    ssh_robot "curl -s --max-time 5 -X POST http://127.0.0.1:8000/api/motors/set_mode/enabled >/dev/null && \
               curl -s --max-time 15 -X POST http://127.0.0.1:8000/api/move/play/wake_up >/dev/null" \
      && echo "robot awake" || echo "WARNING: could not wake the robot — it will not move"

    # Full volume, every start, and read back. ALSA forgets it on a reboot —
    # the robot came back at 62% after its battery ran out once, and on stage
    # a robot the room cannot hear is a robot that did not answer.
    # Both PCM controls of card 0 (the Reachy Mini Audio device): the daemon
    # plays through it, and either one below 100% turns the voice down.
    VOLUME=$(ssh_robot "amixer -c 0 sset PCM,0 100% >/dev/null 2>&1; \
                        amixer -c 0 sset PCM,1 100% >/dev/null 2>&1; \
                        amixer -c 0 sget PCM,0 | grep -o '\[[0-9]*%\]' | head -1; \
                        amixer -c 0 sget PCM,1 | grep -o '\[[0-9]*%\]' | head -1" | tr -d '[]' | tr '\n' ' ')
    case "$VOLUME" in
      "100% 100% ") echo "volume 100%" ;;
      *) echo "WARNING: volume is '$VOLUME', not 100% — the room may not hear the robot" ;;
    esac

    # PYTHONPATH=$REMOTE_DIR, not the daemon's own site-packages layout: that
    # venv HAS the heavy deps this needs (numpy, PIL, qdrant-edge-py — see
    # this script's header) but not this repo's own demo/emulator packages,
    # which live only in the rsynced tree above. `--camera robot --mic robot
    # --speaker robot --robot-host 127.0.0.1` talks to the camera+mic service
    # and the Pollen daemon over loopback. `--display remote --dashboard-host
    # "$BRAIN"` pushes the audience's screen to the dashboard on the laptop
    # instead of serving it here. `setsid --fork`/no trailing `&`/no
    # stdin: same reasoning as the camera+mic service's own start above.
    # Under voice_loop.sh, not bare: the loop exits 42 when the dashboard's
    # Restart button is pressed, and the supervisor then wipes MEMORY_DIR and
    # starts it again (see that script). Any other exit — a crash, the SIGTERM
    # from `voice-stop`, Ctrl-C — ends the run with the memory untouched, so
    # `voice-stop`'s pkill still stops the demo for good.
    ssh_robot "cd '$REMOTE_DIR' && PYTHONPATH='$REMOTE_DIR' setsid --fork \
      bash '$REMOTE_DIR/voice_loop.sh' '$MEMORY_DIR' \
      '$RUN_PY' -u -m demo.run_demo \
      --camera robot --mic robot --speaker robot \
      --robot-host 127.0.0.1 --robot-camera-port $PORT \
      --brain '$BRAIN' --port $BRAIN_PORT --detect-port $DETECT_PORT \
      --embed-port $EMBED_PORT --display remote --dashboard-host '$BRAIN' \
      --web-port $WEB_PORT \
      --memory-dir '$MEMORY_DIR' \
      ${ON_ROBOT:+--on-robot '$ON_ROBOT'} \
      >> '$VOICE_LOG' 2>&1 < /dev/null"

    # No HTTP surface of its own to poll (the dashboard is remote, the
    # camera/mic are pulled FROM the service, not served) — wait out the
    # camera warmup (demo/run_demo.py's run_voice: up to ~6s) and then just
    # confirm the process is still alive, reading why from the log if not.
    sleep 8
    if ssh_robot "pgrep -f '[d]emo\.run_demo' >/dev/null"; then
      echo "voice loop started — last log lines:"
      ssh_robot "tail -15 '$VOICE_LOG'"
    else
      echo "voice loop exited before startup finished; last log lines:"
      ssh_robot "tail -30 '$VOICE_LOG'"
      exit 1
    fi
    ;;

  prepare)
    # Everything voice-start would install and carry, and nothing else: no
    # wake, no loop, no camera. Safe to run any number of times.
    prepare_placement
    ssh_robot "ls -la '$REMOTE_DIR/assets' 2>/dev/null | tail -n +2 | awk '{print \$5, \$9}'"
    echo "the loop would run under: $RUN_PY"
    ;;

  voice-stop)
    stop_voice_loop
    ;;

  voice-status)
    echo "--- process ---"
    ssh_robot "pgrep -af '$VOICE_PROC' || echo '  not running'"
    echo "--- last lines ---"
    ssh_robot "tail -n ${2:-20} '$VOICE_LOG' 2>/dev/null || echo '  no voice log yet'"
    ;;

  voice-logs)
    ssh_robot "tail -n ${2:-40} -f '$VOICE_LOG'"
    ;;

  *)
    echo "usage: $0 {start|stop|status|logs|prepare|voice-start|voice-stop|voice-status|voice-logs}" >&2
    exit 2
    ;;
esac
