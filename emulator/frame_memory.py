"""What the robot saw: frames, as SigLIP 2 embeddings in its Qdrant Edge shard.

The robot stores the frames it sees and can be asked, by voice, what it saw.
Every frame carries an `image` vector, and a `text` one when it has words.
`text` is bge over the frame's own words — labels, names, the side it looked,
the caption (frame_text) — and it says whether a question names anything the
frames hold (recall_text: a gate, and an empty frame to beat). `image` is
SigLIP's embedding of the picture, and SigLIP's text tower puts a question in
the same space. A question that names something is answered by both: every
frame is scored by its words and by its picture, and the frame that stands
out most among all of them, in either, comes first (recall) — a door no
label names by its picture, Sasha by her name. A question that names nothing
gets the day: its frames start from the picture nearest the question, the
rest picked as the most different (day_frames; demo/conversation.py decides
when).

The picture search ranks; it has no gate. On the robot's own 376 frames it put
a right frame first for 9 of 11 things no frame's words name — a door, a red
blanket, someone with arms crossed, a microphone stand — but what those right
frames scored, 0.081-0.136, is what "what did you see today?", which names
nothing, scores on its best frame: 0.109-0.123 for four such questions. So
nothing is cut at a score: the model looks at the frames the search led with
and says whether the thing is there.

Torch-free: the onnx-community SigLIP2-base vision and text ONNX on the
onnxruntime CPU EP — 0.37 GB and 1.1 GB. Measured on the laptop: ~36 ms per
frame.

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

# onnx-community's SigLIP2-base as plain ONNX, both towers.
MODEL_REPO = "onnx-community/siglip2-base-patch16-224-ONNX"
IMAGE_SIZE = 224            # SigLIP2-base patch16-224
MAX_TEXT_TOKENS = 64        # SigLIP pads text to a fixed 64
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

# How far back the picture-based pick is allowed to look — all but its start:
# the picture nearest the question is searched among every frame, however
# old (day_frames' `like`). iter_points does not
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
    """What FrameMemory needs from an embedder: a picture's vector, and a
    question's in the same space. Structural, so tests inject a tiny
    deterministic fake with no ONNX."""

    def embed_image(self, frame_rgb: "NDArray[np.uint8]") -> "NDArray[np.float32]":
        ...

    def embed_text(self, text: str) -> "NDArray[np.float32]":
        ...


def _l2(vec: "NDArray[np.float32]") -> "NDArray[np.float32]":
    return vec / (float(np.linalg.norm(vec)) + 1e-9)


class SiglipEmbedder:
    """SigLIP 2 vision + text towers on the onnxruntime CPU EP.

    Heavy deps (onnxruntime, tokenizers, huggingface_hub) are imported lazily
    in __init__ — only a process that actually builds visual memory pays for
    them, the same reasoning as TextMemory's lazy fastembed import. The model
    files download once via huggingface_hub and are cached, exactly like
    fastembed's own model provisioning.
    """

    def __init__(self, repo: str = MODEL_REPO) -> None:
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        cpu = ["CPUExecutionProvider"]
        self._vision = ort.InferenceSession(
            hf_hub_download(repo, "onnx/vision_model.onnx"), providers=cpu)
        self._text = ort.InferenceSession(
            hf_hub_download(repo, "onnx/text_model.onnx"), providers=cpu)
        self._vision_in = self._vision.get_inputs()[0].name
        self._text_inputs = {i.name for i in self._text.get_inputs()}
        self._tokenizer = Tokenizer.from_file(hf_hub_download(repo, "tokenizer.json"))
        # SigLIP: fixed-length pad to 64 (pad id 0 = <pad>), truncate to 64.
        self._tokenizer.enable_padding(length=MAX_TEXT_TOKENS, pad_id=0,
                                       pad_token="<pad>")
        self._tokenizer.enable_truncation(MAX_TEXT_TOKENS)

    def _pooler(self, session, outputs) -> "NDArray[np.float32]":
        # Both towers emit last_hidden_state + pooler_output; we want the
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

    def embed_text(self, text: str) -> "NDArray[np.float32]":
        ids = self._tokenizer.encode(text.lower()).ids
        feed = {"input_ids": np.asarray([ids], dtype=np.int64)}
        if "attention_mask" in self._text_inputs:
            feed["attention_mask"] = (feed["input_ids"] != 0).astype(np.int64)
        out = self._text.run(None, feed)
        return _l2(self._pooler(self._text, out))


def _embedder(repo: str) -> "SiglipEmbedder":
    cached = _embedder_cache.get(repo)
    if cached is None:
        cached = SiglipEmbedder(repo)
        _embedder_cache[repo] = cached
    return cached


class FrameMemory:
    """Store of frames the robot has seen, searchable by their words and by
    their pictures.

    `remember(frame, detections)` embeds the whole frame (SigLIP's vision
    tower) and upserts it with the YOLO detections + the jpeg as payload, and
    the frame's words (frame_text) as a bge vector beside it. `recall_text`
    searches those words; `day_frames` picks the day by picture, led by the
    picture nearest a question when it is given one.

    Called from two threads with no coordination of their own: SceneChangeWriter
    (below) drives `remember()` off the detect thread while the voice
    thread's turn handler alternates reads (`recall_text()`, `day_frames()`)
    with `remember()` (see
    demo/run_demo.py's `_handle_stream`). The Qdrant Edge shard does not
    synchronise itself; EdgeStore does, with its own lock around every call
    into it (emulator/edge_store.py), and that is what keeps the shard whole
    — TextMemory writes the same store without the lock here. `_lock` below
    only keeps this memory's own steps apart: a scan of the day, page by
    page, never runs while this memory adds a frame. The embedding
    (`embed_image`, bge) and jpeg encoding stay OUTSIDE it: they touch no
    shared state, and holding it through a SigLIP inference would stall the
    OTHER thread's frame call on model compute for nothing.
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
        # The question's side too, in the same space: a text tower that is
        # missing or of another size fails the start, not the first question
        # no frame's words answer (day_frames).
        question_dim = len(self._embedder.embed_text("a picture"))
        if question_dim != dim:
            raise ValueError(f"the embedder puts a question in {question_dim}-d "
                             f"and a picture in {dim}-d: not one space")
        self._dim = dim
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
            # Not while this memory scans its day (see the class docstring).
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
                   min_distance: float = DAY_MIN_DISTANCE,
                   like: str | None = None) -> list[dict]:
        """The day as FRAMES, picked BY PICTURE — what a `seen` question no
        frame's words answer gets: "what did you see today?", "did you see a
        door?" with no label for one, and "did you see a dog?" with no dog in
        any frame — newest first, each with its jpeg.

        Given `like` — the question, as the model put it — the pick starts
        from the frame whose picture is nearest it instead of the newest
        (_nearest_picture), so that frame is always among those returned: the
        door no label names, found by what it looks like. Nothing decides
        whether the question named anything; a question that named nothing
        starts from some frame of the day, which is still the day.

        recall_text finds nothing for such a question, and the newest frames
        are not the day either.
        Measured on this robot's own 371 frames: the newest 4 are the
        same picture four times — even the least similar PAIR of them is
        0.914. So the day is picked farthest-point on the `image` vector, the
        SigLIP embedding already stored with every frame: start from the
        picture nearest the question (the newest, without one), then keep
        taking the frame least like everything taken so far,
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
        machine, and the per-point fetch is what the cap bounds there. A
        question to start from adds its SigLIP embedding and one search over
        every frame: 24-38 ms for the whole call on the Mac, over the robot's
        376 frames.
        """
        if limit < 1:
            return []
        lead_id = self._nearest_picture(like, before) if like else None
        with self._lock:
            rows = [(point_id, payload) for point_id, payload
                    in self._store.iter_points(self._frames(),
                                               fields=["ts", "looked"])]
        rows = [row for row in rows
                if before is None or row[1].get("ts", 0.0) < before]
        rows.sort(key=lambda row: row[1].get("ts", 0.0), reverse=True)
        lead = next((row for row in rows if row[0] == lead_id), None)
        pool = [row for row in rows if row[1].get("looked")]
        pool += [row for row in rows if not row[1].get("looked")][:DAY_POOL_SCENES]
        if lead is not None and lead not in pool:
            pool.append(lead)  # older than the pool: it leads all the same
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
        # The pool is newest first: without a picture to lead, the newest.
        picked = [point_ids.index(lead[0]) if lead is not None and lead[0] in point_ids
                  else 0]
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

    def _nearest_picture(self, question: str, before: float | None) -> str | None:
        """The id of the stored frame, taken before `before`, whose PICTURE
        is nearest `question`: SigLIP's text tower puts the question in the
        pictures' space, and the shard searches every such frame's `image`
        vector for it — no gate (see the module docstring). None for an
        empty question.

        None too when the question cannot be put in that space, whatever the
        reason — the embed service down, a reply that is no vector, the text
        tower itself failing on the robot. The words were searched already
        and found nothing, so the day is still the answer; only its lead is
        lost. That is logged rather than taking the memory down with it, and
        anything but a service or a reply at fault is logged with its
        traceback, so a bug is not mistaken for an outage. The shard's own
        search below is not covered: a store that fails is the memory down."""
        vector = self._in_pictures_space(question)
        if vector is None:
            return None
        with self._lock:
            nearest = self._store.nearest_ids(vector, 1, self._frames_before(before),
                                              using=IMAGE_VECTOR)
        return nearest[0][0] if nearest else None

    def _in_pictures_space(self, question: str) -> list[float] | None:
        """`question` as SigLIP's text tower puts it, beside the pictures —
        None for an empty question, or when it cannot be put there for any
        reason (the embed service down, a reply that is no vector of the
        pictures' size and some length, the text tower failing): logged —
        with its traceback unless a service or a reply was at fault — and
        the caller does without the picture search."""
        question = question.strip()
        if not question:
            return None
        try:
            vector = np.asarray(self._embedder.embed_text(question), dtype=np.float32)
            # Of the pictures' size, every number finite, and a length the
            # search can divide by: a zero vector, or one whose length
            # underflows or overflows, has no direction and would lead the
            # day with an arbitrary frame.
            if (vector.shape != (self._dim,) or not np.isfinite(vector).all()
                    or not 0.0 < float(np.linalg.norm(vector)) < np.inf):
                raise ValueError(f"a question vector of shape {vector.shape}, "
                                 f"not {self._dim} numbers of some length")
        except Exception as exc:  # noqa: BLE001 — the caller does without it
            logger.warning("no picture search for %r (%s: %s)",
                           question, type(exc).__name__, exc,
                           exc_info=not isinstance(exc, (OSError, ValueError, KeyError)))
            return None
        return vector.tolist()

    def _frames_before(self, before: float | None):
        from emulator.edge_store import all_of, less_than

        return all_of(self._frames(),
                      less_than("ts", before) if before is not None else None)

    def recall(self, query: str, k: int = 3, before: float | None = None) -> list[dict]:
        """The frames a question about a THING finds, best first — by their
        words or by their pictures, whichever singles a frame out.

        Whether the question names anything the frames hold is the words' to
        say (recall_text: FRAME_TEXT_MIN_SCORE and the empty frame); nothing
        found there is nothing here, and the day answers instead
        (demo/conversation.py). Which frames answer is both searches': every
        frame stored before `before` is scored by its words and by its
        picture, each search's scores are measured against all of them — how
        far a frame stands out among everything seen — and a frame ranks by
        the higher of the two (_standing).

        The words alone put a frame of "person" first for things no label
        names: on the robot's own 376 frames "a door" scored 0.667 on "I saw
        chair, keyboard, person", over the gate and the empty frame, and the
        door itself never came up. Measured there over 34 questions — 18
        about things no frame's words name, 16 about labels and names — the
        right frame came first for 29 by the higher standing, 17 by the words
        alone and 24 by the picture alone, and was among the first three for
        all 34. A frame of "person" stands out by its words by 1.7-2.1 of
        their spread; the door, the red blanket, the microphone stand stand
        out by their pictures by 4.4-6.5. Measured against only the 100 best
        of each search instead of all frames, 21 of 34: among a hundred
        frames of "person", one stands out more than it does among all."""
        query = query.strip()
        if self._text_embedder is None or not query:
            return []
        # One bge call for the gate and the ranking both: a second one that
        # failed would lose a question the first had already found.
        words = [float(x) for x in next(iter(self._text_embedder.query_embed([query])))]
        if not self._words_hits(words, k, before):
            return []
        best = self._standing(query, words, before)[:k]
        with self._lock:
            rows = dict(self._store.get([point_id for point_id, *_ in best]))
        # `score` is the found_by search's own cosine, as a frame the words
        # alone found always carried it (the dashboard shows it).
        return [{**rows[point_id], "found_by": found_by, "score": score}
                for point_id, found_by, score in best if point_id in rows]

    def _standing(self, query: str, words: list[float],
                  before: float | None) -> list[tuple[str, str, float]]:
        """Every frame stored before `before`, best first, by how far it
        stands out — in its words' scores or its picture's, each measured in
        that search's own spread over all the frames — with the search it
        stood out in ("words" or "picture") and its score there.

        A frame stored without words of its own (bge was down when it was
        written) scores what the empty frame does: it names nothing. Left
        out, the words' spread would be measured over the few frames that
        have words — over one, it is none at all — and any picture would
        beat a frame whose words held exactly what was asked.

        The shard's searches are not under this memory's lock: they only
        read, the store serialises each call itself, and the scene writer
        is not held up for the length of two scorings of every frame."""
        query_filter = self._frames_before(before)
        in_pictures = self._in_pictures_space(query)
        for _attempt in range(3):
            every = self._store.count(query_filter)
            if not every:
                return []
            scored = {"words": dict(self._store.nearest_ids(words, every, query_filter,
                                                            using=TEXT_VECTOR))}
            if in_pictures is not None:
                scored["picture"] = dict(self._store.nearest_ids(
                    in_pictures, every, query_filter, using=IMAGE_VECTOR))
            # Unlocked, a frame the scene writer adds between the count and
            # the searches would be scored by one search and not the other:
            # scored again over the frames as they are now.
            if self._store.count(query_filter) == every:
                break
        if in_pictures is not None:
            empty = self._empty_frame_score(words)
            for point_id in scored["picture"]:
                scored["words"].setdefault(point_id, empty)
        standing: dict[str, tuple[float, str, float]] = {}
        for name, scores in scored.items():
            values = np.fromiter(scores.values(), dtype=np.float64, count=len(scores))
            # A search that scores every frame alike singles none out: each
            # stands out by nothing in it, rather than dropping out of the
            # answer altogether.
            mean, spread = values.mean(), values.std() or 1.0
            for point_id, score in scores.items():
                stands_out = (score - mean) / spread
                if stands_out > standing.get(point_id, (-np.inf,))[0]:
                    standing[point_id] = (stands_out, name, score)
        return [(point_id, found_by, score) for point_id, (_z, found_by, score)
                in sorted(standing.items(), key=lambda item: item[1][0], reverse=True)]

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
        return self._words_hits(next(iter(self._text_embedder.query_embed([query]))),
                                k, before)

    def _words_hits(self, vector, k: int, before: float | None) -> list[dict]:
        """recall_text for a question bge has already embedded."""
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
