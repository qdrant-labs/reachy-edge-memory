"""What the robot saw: frames, as SigLIP 2 embeddings in its Qdrant Edge shard.

The robot stores the frames it sees and can be asked, by voice, what it saw.
Every frame carries an `image` vector, and a `text` one when it has words.
`image` is SigLIP's embedding of the picture: what the day's frames are picked
by (day_frames — the most different ones). `text` is bge over the frame's own
words — labels, names, the side it looked, the caption (frame_text) — and it
is what a question about a thing finds (recall_text): measured, 32 right
frames of 36 against a SigLIP text-to-image search's 8, so the conversation
answers by the words and, when they hold nothing, by the day
(demo/conversation.py).

Only SigLIP's vision tower is loaded. Its text tower — 1.1 GB of ONNX against
the vision tower's 0.37 GB — served that text-to-image search, and no gate
on it held: an early eval set put absent things under 0.07 and present ones
over 0.10, but on a laptop run's frames "a plant" scored 0.061 with a plant
in view and "what did you see today?", which names nothing, 0.106.

Torch-free: the onnx-community SigLIP2-base vision ONNX on the onnxruntime
CPU EP. Measured on the laptop: ~36 ms per frame.

The frames live in the `memory` shard beside the conversation
(emulator/memory.py), told apart by `kind`; the JPEG travels in the payload,
so a recall is self-contained and the dashboard gets the picture back.
"""

from __future__ import annotations

import base64
import io
import logging
import threading
import time
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# onnx-community's SigLIP2-base as plain ONNX; only its vision tower is used.
MODEL_REPO = "onnx-community/siglip2-base-patch16-224-ONNX"
IMAGE_SIZE = 224            # SigLIP2-base patch16-224
COLLECTION = "frames"

# The gate on the frame search — the one over the frame's own words
# (frame_text: labels, names, the side it looked, the caption). bge cosine, so
# the same scale as the conversation's 0.62, and calibrated the same way:
# measured against this robot's 371 stored frames, a
# question a frame really answers scores 0.64-0.88 ("did you see a sink?"
# 0.884, "did you see a bottle?" 0.859, "the window with the curtains" 0.749),
# and a question no picture answers scores 0.55-0.63 ("what did we talk
# about?" 0.590, "do you remember my name?" 0.630). 0.66 sits in the gap.
FRAME_TEXT_MIN_SCORE = 0.66

# A frame with nothing in it, in frame_text's own words: "I saw something".
# Every labelled frame's words say "I saw ...", so a question about seeing
# matches every such frame on that alone, whatever it names. Measured with bge on the 19 frames of
# a run on the laptop: "What did you see?" scored 0.704 on "I saw person", "Did
# you see a cup?" 0.676 on "I saw plant" — both over the gate, neither a match.
# A frame counts only when it also beats this empty one, which is when its
# CONTENT matched: there and on 150 frames written like the robot's, every
# question that named nothing (14 of them) scored under the empty frame
# (-0.023 at most), and every question about a thing that was there over it
# (+0.075 at least) — "did you see Sasha?" by +0.173 even with Sasha in every
# frame, which comparing the frames with each other would have lost. The
# margins are not wide: "Did you see a dog?" with no dog anywhere scores 0.732
# on "I saw person", over the gate, and only 0.025 under the empty frame. And
# a question about people is about the "person" label: "who did you see
# today?" scores 0.697 on "I saw person", over the empty frame's 0.675, and
# gets that frame, not the day.
EMPTY_FRAME = {"labels": ["something"]}

# Where the robot can be asked to look (demo/chat_session.py's DIRECTIONS).
LOOKED = ("ahead", "left", "right")

# The YOLO label that makes a stored frame worth a face lookup.
PERSON_LABEL = "person"

# The least time between two frames kept by the scene writer. 2 s filled the
# memory with 177 near-identical frames in half an hour of talking — YOLO's
# labels flicker (a tie, a bottle, a second person) and every flicker counted
# as a new scene.
SCENE_MIN_INTERVAL_S = 10.0

# How the day is read back (day_frames): this many frames whenever the store
# has them, picked farthest-point on the SigLIP cosine of the picture itself,
# so they are the most different frames of the day; only a near-duplicate of
# one already picked is left out. Measured on this robot's own evenings: the
# newest frames of an unchanged room sit at 0.90-0.97 of each other, frames of
# genuinely different scenes at 0.61-0.79. The cutoff used to be 0.85 —
# "nothing different enough means one frame is the answer" — and live, the
# first minute of a demo (three frames of the same chair) came
# back as one picture; the room wants several every time, and the day
# system message (demo/chat_session.py's DAY_SYSTEM) folds alike frames into
# one sentence rather than reciting them.
DAY_FRAMES = 3
DAY_MIN_DISTANCE = 0.97

# How far back the picture-based pick is allowed to look. iter_points does not
# hand back vectors, so every candidate costs a retrieve of its own: measured
# on the robot's own 371 frames, 47 candidates cost 0.5 ms against 4.1 ms
# for all 371, and the store gains a frame every 10 s for as long as the robot
# is awake. Every look is a candidate however
# old it is — a whole evening produced 7 of them, and they are the frames the
# robot was asked for — plus this many of the newest scenes.
DAY_POOL_SCENES = 40

# How a frame is told apart from an exchange in the shared `memory` shard, and
# the named vectors it carries (emulator/edge_store.py).
FRAME_KIND = "frame"
IMAGE_VECTOR = "image"
TEXT_VECTOR = "text"

# JPEG quality for the frame kept in the payload (shown on the dashboard).
JPEG_QUALITY = 80

# Loading the ONNX session costs real wall time; share one loaded
# embedder across FrameMemory instances in a process, as TextMemory does with
# its fastembed model.
_embedder_cache: dict[str, "SiglipEmbedder"] = {}


class Embedder(Protocol):
    """What FrameMemory needs from an embedder: a picture's vector.
    Structural, so tests inject a tiny deterministic fake with no ONNX."""

    def embed_image(self, frame_rgb: "NDArray[np.uint8]") -> "NDArray[np.float32]":
        ...


def _l2(vec: "NDArray[np.float32]") -> "NDArray[np.float32]":
    return vec / (float(np.linalg.norm(vec)) + 1e-9)


class SiglipEmbedder:
    """SigLIP 2's vision tower on the onnxruntime CPU EP.

    Heavy deps (onnxruntime, huggingface_hub) are imported lazily
    in __init__ — only a process that actually builds visual memory pays for
    them, the same reasoning as TextMemory's lazy fastembed import. The model
    files download once via huggingface_hub and are cached, exactly like
    fastembed's own model provisioning.
    """

    def __init__(self, repo: str = MODEL_REPO) -> None:
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download

        self._vision = ort.InferenceSession(
            hf_hub_download(repo, "onnx/vision_model.onnx"),
            providers=["CPUExecutionProvider"])
        self._vision_in = self._vision.get_inputs()[0].name

    def _pooler(self, session, outputs) -> "NDArray[np.float32]":
        # The tower emits last_hidden_state + pooler_output; we want the
        # pooled [*,768] vector. Pick it by name (order isn't contractual).
        for out, spec in zip(outputs, session.get_outputs()):
            if "pool" in spec.name.lower():
                return np.asarray(out, dtype=np.float32).reshape(-1)
        for out in outputs:  # fallback: the 2-D [*,768] output
            arr = np.asarray(out, dtype=np.float32)
            if arr.ndim == 2:
                return arr.reshape(-1)
        return np.asarray(outputs[0], dtype=np.float32).reshape(-1)

    def embed_image(self, frame_rgb: "NDArray[np.uint8]") -> "NDArray[np.float32]":
        from PIL import Image

        # SigLIP preprocessing (NOT CLIP's): bicubic resize to 224, /255,
        # (x-0.5)/0.5 per channel (mean=std=0.5), NCHW float32. This exact
        # recipe validated to cosine 1.0 against an fp32 reference.
        img = Image.fromarray(np.asarray(frame_rgb, dtype=np.uint8)).convert("RGB")
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = ((arr - 0.5) / 0.5).transpose(2, 0, 1)[None].astype(np.float32)
        out = self._vision.run(None, {self._vision_in: arr})
        return _l2(self._pooler(self._vision, out))


def _embedder(repo: str) -> "SiglipEmbedder":
    cached = _embedder_cache.get(repo)
    if cached is None:
        cached = SiglipEmbedder(repo)
        _embedder_cache[repo] = cached
    return cached


class FrameMemory:
    """Store of frames the robot has seen, searchable by their words.

    `remember(frame, detections)` embeds the whole frame (SigLIP's vision
    tower) and upserts it with the YOLO detections + the jpeg as payload, and
    the frame's words (frame_text) as a bge vector beside it. `recall_text`
    searches those words; `day_frames` picks the day by picture.

    Called from two threads with no coordination of their own: SceneChangeWriter
    (below) drives `remember()` off the detect thread while the voice
    thread's turn handler alternates reads (`recall_text()`, `day_frames()`)
    with `remember()` (see
    demo/run_demo.py's `_handle_stream`). The Qdrant Edge shard does not
    synchronise itself — the on-disk collection's payload/vector mutation is
    not atomic — so `_lock` below serialises every Qdrant client call. The embedding
    (`embed_image`, bge) and jpeg encoding stay OUTSIDE the lock: they
    touch no shared state, and holding the lock through a SigLIP inference
    would stall the OTHER thread's Qdrant call on model compute for nothing.
    """

    def __init__(self, path: str | None = None, *, embedder: "Embedder | None" = None,
                 repo: str = MODEL_REPO, store=None, text_embedder=None) -> None:
        """`store` is a shared EdgeStore with an `image` vector (and a `text`
        one for the frame's words) — the robot's `memory` shard, shared with
        the conversation. `text_embedder` (bge, the fastembed shape) gives
        every frame with words its `text` vector."""
        from emulator.edge_store import IMAGE, KIND, TEXT, EdgeStore

        self._embedder = embedder if embedder is not None else _embedder(repo)
        self._text_embedder = text_embedder
        self._empty_vector = None  # EMPTY_FRAME's words, embedded on first use
        self._lock = threading.Lock()
        # Size the shard from a real embedding, so a different model can't
        # silently mismatch the shard, mirroring TextMemory.
        dim = len(self._embedder.embed_image(
            np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), np.uint8)))
        if store is None:
            vectors = {IMAGE: dim}
            if text_embedder is not None:
                vectors[TEXT] = len(next(iter(text_embedder.embed(["_"]))))
            store = EdgeStore(path, vectors=vectors, tenant_field=KIND)
        elif store.size(IMAGE) not in (None, dim):
            raise ValueError(f"the shard's image vector is {store.size(IMAGE)}-d, "
                             f"the embedder gives {dim}")
        self._store = store

    def frame_text(self, payload: dict) -> str:
        """Everything a frame already knows about itself, as one sentence —
        what it is searched BY.

        Measured on this robot's own 371 stored frames: asked "did you see a
        bottle?", "was there a cup?", "what was on the table?", "a plant in a
        pot", a SigLIP
        text-to-image search found 8 right frames out of 36; searching these
        words with bge found 32 — and the four it "missed" are questions with
        fewer than three right answers in the whole store, so it found every
        one there was. The labels and the names were in the payload the whole
        time; nothing looked at them. Only 7 of those 371 frames had a
        caption, which is why the caption alone was never enough."""
        where = {"left": "on my left", "right": "on my right",
                 "ahead": "in front of me"}.get(payload.get("looked"))
        parts = []
        if where:
            parts.append(where[0].upper() + where[1:])
        labels = list(payload.get("labels") or [])
        if labels:
            parts.append("I saw " + ", ".join(labels))
        names = [name for name in (payload.get("names") or []) if name]
        if names:
            parts.append(" and ".join(names)
                         + (" was there" if len(names) == 1 else " were there"))
        caption = (payload.get("caption") or "").strip()
        if caption:
            parts.append(caption)
        return ". ".join(parts)

    def _text_vector(self, payload: dict) -> list[float] | None:
        """The frame's words as a bge vector, or None without an embedder."""
        if self._text_embedder is None:
            return None
        text = self.frame_text(payload)
        if not text:
            return None
        try:
            return [float(x) for x in next(iter(self._text_embedder.embed([text])))]
        except Exception as exc:  # noqa: BLE001 — a frame without words is still a frame
            print(f"  [visual-memory] no text vector ({type(exc).__name__}: {exc})")
            return None

    def remember(self, frame_rgb: "NDArray[np.uint8]",
                 detections: list[dict] | None = None,
                 meta: dict | None = None) -> str:
        """Store a frame; returns its point id (describe() takes it)."""
        detections = detections or []
        # Embedding + jpeg encoding are pure CPU work over `frame_rgb` alone —
        # no shared state touched — so they run OUTSIDE `_lock` (see class
        # docstring): the detect thread's writes never wait on the voice
        # thread's Qdrant call, or vice versa, just on model/codec compute.
        vector = self._embedder.embed_image(frame_rgb)
        payload = {
            "detections": detections,
            # a flat label list (YOLO as metadata): frame_text's words
            "labels": sorted({d["label"] for d in detections if d.get("label")}),
            "jpeg_b64": _encode_jpeg(frame_rgb),
            "ts": time.time(),
            **(meta or {}),
            "kind": FRAME_KIND,
        }
        # A flat list of who is in the frame, like `labels` for objects.
        names = sorted({person["name"] for person in payload.get("people") or []
                        if person.get("name")})
        if names:
            payload["names"] = names
        # The frame's own words (labels, names, the side it looked), embedded
        # like any other text: this is what a question about a THING finds.
        # Outside the lock, like the image embedding above.
        text_vector = self._text_vector(payload)
        vectors: dict = {IMAGE_VECTOR: vector.tolist()}
        if text_vector is not None:
            vectors[TEXT_VECTOR] = text_vector
        with self._lock:
            # id allocation + upsert as one step under the lock, like every
            # other call into the shard (see the class docstring).
            point_id = self._store.new_id()
            self._store.add(point_id, vectors, payload)
        return point_id

    def describe(self, point_id: str, caption: str) -> None:
        """Keep what the robot said about a frame it looked at, beside it —
        and re-embed the frame's words with the caption in them, so what it
        said about the picture is searchable too.

        The whole sentence is rebuilt, not just the caption: the labels and
        the names are what answer a question about a thing, and a caption
        that says "I see a window" must not replace them."""
        with self._lock:
            rows = self._store.get([point_id])
        payload = dict(rows[0][1]) if rows else {}
        payload["caption"] = caption
        vector = self._text_vector(payload)
        with self._lock:
            self._store.set_payload(point_id, {"caption": caption})
            if vector is not None:
                self._store.set_vector(point_id, TEXT_VECTOR, vector)

    def latest_looks(self, directions: list[str] | None = None, limit: int = 4,
                     before: float | None = None) -> list[dict]:
        """The frames the robot took when asked to look (stored with
        meta={"looked": ...} by demo/run_demo.py's Looker), newest first —
        optionally only one direction, only those taken before `before`."""
        from emulator.edge_store import all_of, match_any

        looked = all_of(self._frames(), match_any("looked", list(directions or LOOKED)))
        with self._lock:
            stamps = [(payload.get("ts", 0.0), point_id) for point_id, payload
                      in self._store.iter_points(looked, fields=["ts"])]
            if before is not None:
                stamps = [(ts, point_id) for ts, point_id in stamps if ts < before]
            newest = [point_id for _ts, point_id in sorted(stamps, reverse=True)[:limit]]
            return [payload for _point_id, payload in self._store.get(newest)]

    def day_frames(self, limit: int = DAY_FRAMES, before: float | None = None,
                   min_distance: float = DAY_MIN_DISTANCE) -> list[dict]:
        """The day as FRAMES, picked BY PICTURE — what a `seen` question no
        frame's words answer gets: "what did you see today?", and "did you see
        a dog?" with no dog in any frame — newest first, each with its jpeg.

        recall_text finds nothing for such a question, and the newest frames
        are not the day either.
        Measured on this robot's own 371 frames: the newest 4 are the
        same picture four times — even the least similar PAIR of them is
        0.914. So the day is picked farthest-point on the `image` vector, the
        SigLIP embedding already stored with every frame: start from the
        newest, then keep taking the frame least like everything taken so far,
        up to `limit` — only a near-duplicate (closer than `min_distance` to
        something already picked) is left out. Picked that way, four frames of
        the same evening are four different scenes: least similar pair 0.609
        over every frame, 0.695 over the capped pool below. A stretch where
        nothing changed still comes back as several frames of the same room
        (DAY_MIN_DISTANCE says why); only frames that ARE the same picture
        collapse into one.

        The candidates are capped, because `iter_points` does not hand back
        vectors and each candidate therefore costs a retrieve of its own:
        every look — the cap can never age one out, a whole evening produced
        7 of them and they are the frames the robot was asked for — plus the
        newest DAY_POOL_SCENES scenes. On the 371 that is 47 candidates:
        0.5 ms of vectors against 4.1 ms for all of them, and 5 ms for the
        whole call warm, of which the walk that finds the candidates (not the
        vectors) is 3.8 ms; the first call after a start pays ~160 ms more,
        paging the shard in. Measured on the Mac — the CM4 is the slower
        machine, and the per-point fetch is what the cap bounds there.
        """
        if limit < 1:
            return []
        with self._lock:
            rows = [(point_id, payload) for point_id, payload
                    in self._store.iter_points(self._frames(),
                                               fields=["ts", "looked"])]
        rows = [row for row in rows
                if before is None or row[1].get("ts", 0.0) < before]
        rows.sort(key=lambda row: row[1].get("ts", 0.0), reverse=True)
        pool = [row for row in rows if row[1].get("looked")]
        pool += [row for row in rows if not row[1].get("looked")][:DAY_POOL_SCENES]
        pool.sort(key=lambda row: row[1].get("ts", 0.0), reverse=True)
        point_ids, vectors = [], []
        with self._lock:
            for point_id, _payload in pool:
                vector = self._store.vectors(point_id).get(IMAGE_VECTOR)
                if vector is not None:
                    point_ids.append(point_id)
                    vectors.append(vector)
        if not point_ids:
            return []
        matrix = np.asarray(vectors, dtype=np.float32)
        matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-9)
        picked = [0]  # the pool is newest first, so this is the newest frame
        while len(picked) < limit and len(picked) < len(point_ids):
            closest = np.max(matrix @ matrix[picked].T, axis=1)
            closest[picked] = np.inf
            nearest = int(np.argmin(closest))
            if closest[nearest] > min_distance:
                break  # only near-duplicates of the answer are left
            picked.append(nearest)
        with self._lock:
            return [payload for _point_id, payload
                    in self._store.get([point_ids[i] for i in sorted(picked)])]

    def people_seen(self, before: float | None = None) -> list[tuple[str, float]]:
        """Everyone named in a stored frame, with when they were last in one,
        most recent first."""
        with self._lock:
            rows = list(self._store.iter_points(self._frames(), fields=["ts", "names"]))
        latest: dict[str, float] = {}
        for _point_id, payload in rows:
            ts = payload.get("ts", 0.0)
            if before is not None and ts >= before:
                continue
            for name in payload.get("names") or []:
                if ts > latest.get(name, float("-inf")):
                    latest[name] = ts
        return sorted(latest.items(), key=lambda item: item[1], reverse=True)

    def recall_text(self, query: str, k: int = 3,
                    before: float | None = None) -> list[dict]:
        """Frames nearest to `query` BY THEIR WORDS — bge over frame_text,
        gated at FRAME_TEXT_MIN_SCORE and at the empty frame (EMPTY_FRAME),
        best first. Nothing back means no frame holds what the question asked
        about — or that it asked about nothing in particular.

        This is the search for a question about a THING: "did you see a
        bottle?", "what was on the table?". Measured on this robot's own
        frames it finds 32 right frames out of 36 against the picture
        search's 8 (see frame_text) — the labels and the names were always
        in the payload, and nothing searched them. The conversation answers
        by these words and, when they hold nothing, by the day's pictures
        (demo/conversation.py's _answer_tool)."""
        query = query.strip()
        if self._text_embedder is None or not query:
            return []
        vector = next(iter(self._text_embedder.query_embed([query])))
        with self._lock:
            hits = self._store.search([float(x) for x in vector], k,
                                      self._frames(), using=TEXT_VECTOR)
        empty = self._empty_frame_score(vector)
        hits = [hit for hit in hits
                if hit["score"] >= FRAME_TEXT_MIN_SCORE and hit["score"] > empty]
        if before is not None:
            hits = [hit for hit in hits if hit.get("ts", 0.0) < before]
        return hits

    def _empty_frame_score(self, query_vector) -> float:
        """What `query_vector` scores against a frame with nothing in it
        (EMPTY_FRAME), on the same cosine the store searches with."""
        if self._empty_vector is None:
            empty = np.asarray(next(iter(self._text_embedder.embed(
                [self.frame_text(EMPTY_FRAME)]))), dtype=np.float32)
            self._empty_vector = empty / max(float(np.linalg.norm(empty)), 1e-9)
        query = np.asarray(query_vector, dtype=np.float32)
        return float(query @ self._empty_vector) / max(float(np.linalg.norm(query)), 1e-9)

    def count(self) -> int:
        """How many frames are stored — the robot's side of "what it has
        seen", for the dashboard's counter."""
        with self._lock:
            return self._store.count(self._frames())

    def close(self) -> None:
        """Put everything on disk and release the shard."""
        with self._lock:
            self._store.close()

    @staticmethod
    def _frames():
        from emulator.edge_store import match_value

        return match_value("kind", FRAME_KIND)


class SceneChangeWriter:
    """Continuous visual capture policy: store a frame in FrameMemory when the
    YOLO scene composition changes — a label appears that wasn't in the last
    stored frame — throttled so a static or flickering scene doesn't spam the
    store.

    This is the "on objects" trigger, independent of speech: it runs off the
    detect loop (~4 fps) so the robot keeps a visual memory of what it sees
    whether or not anyone is talking. The other way a frame is kept is a look
    the robot was asked for (demo/run_demo.py's Looker).

    `people` (optional) names who is in a frame that has a person in it:
    frame -> [{"name", "box", "score"}], stored as the frame's `people`, so the
    fact "Sasha was here, then" is kept with the picture.

    Trigger is on ADDED labels only (something new entered), not removals — so
    the memory grows with new sightings rather than logging the scene emptying
    out. `min_interval` debounces YOLO's per-frame flicker.
    """

    def __init__(self, memory: "FrameMemory", *, min_interval: float = SCENE_MIN_INTERVAL_S,
                 clock=time.monotonic, people=None) -> None:
        self._memory = memory
        self._people = people
        self._min_interval = min_interval
        self._clock = clock
        self._last_labels: frozenset[str] = frozenset()
        self._last_store = float("-inf")

    def observe(self, frame_rgb: "NDArray[np.uint8] | None",
                detections: list[dict] | None) -> bool:
        """Feed one detect cycle. Returns True iff a frame was stored."""
        # No frame (camera glitch) → don't advance state: a later cycle with
        # the same labels but a real frame should still get its chance.
        if frame_rgb is None:
            return False
        detections = detections or []
        labels = frozenset(d["label"] for d in detections if d.get("label"))
        new = labels - self._last_labels
        if not new:
            self._last_labels = labels
            return False
        now = self._clock()
        if now - self._last_store < self._min_interval:
            # Throttled: the new label stays new, and is stored on the first
            # cycle past the interval. Marking it seen here lost every object
            # that arrived within ten seconds of the last stored frame.
            return False
        # The attempt starts the interval, so a store that is down is tried
        # once per interval rather than on every detect cycle...
        self._last_store = now
        meta = None
        if self._people is not None and PERSON_LABEL in labels:
            meta = {"people": self._people(frame_rgb)}
        self._memory.remember(frame_rgb, detections, meta)
        # ...but the labels count as stored only once they are: one embed
        # timeout used to lose the object for as long as it stayed in view.
        self._last_labels = labels
        return True


def _encode_jpeg(frame_rgb: "NDArray[np.uint8]") -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.asarray(frame_rgb, dtype=np.uint8)).convert("RGB").save(
        buf, format="JPEG", quality=JPEG_QUALITY)
    return base64.b64encode(buf.getvalue()).decode("ascii")
