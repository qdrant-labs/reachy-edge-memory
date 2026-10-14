"""The vector store every memory sits on: Qdrant Edge, in-process.

Qdrant Edge is a separate product from the `qdrant-client` package — an
embedded engine (Rust, via PyO3) that runs *inside* this process with no
server, no port and no network. That is the whole premise of the demo: the
robot's memory lives on the robot, and nothing it has seen leaves the device.

The robot keeps three shards, one per lifecycle:

    memory/     what it lived through — named vectors `text` (bge, 384) and
                `image` (SigLIP2, 768); exchanges carry `text`, frames carry
                `image` and, once described, `text`; told apart by `kind`
    people/     who it has met — `face` (HSFace, 512) as a multivector: one
                point per person, every shot of their face in it
    knowledge/  what it was taught — `text` (bge, 384), restored from a
                snapshot on every start

This module adapts the Edge API once for all of them. Points are named by
UUID, so two memories writing into one shard never hand out the same id, and
every shard call goes through one lock: the voice loop and the scene writer
use the same shard from different threads.
"""

from __future__ import annotations

import atexit
import dataclasses
import json
import os
import shutil
import tempfile
import threading
import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

# How many points one scroll page fetches when walking the whole shard.
SCROLL_PAGE = 256

# The named vectors, and the payload field that tells kinds of points apart.
TEXT = "text"
IMAGE = "image"
FACE = "face"
KIND = "kind"


@dataclasses.dataclass(frozen=True)
class Vectors:
    """One named vector: its size, and whether a point holds several of them
    compared by the closest (MaxSim) — a person's face shots."""
    size: int
    multivector: bool = False


class EdgeStore:
    """One Qdrant Edge shard, with the operations the memories actually use.

    `vectors` names the vectors a new shard is created with ({"text": 384} or
    {"face": Vectors(512, multivector=True)}); an existing shard is loaded with
    whatever it was created with. `tenant_field` gets a keyword index marked
    `is_tenant`, for a shard whose points are told apart by that field.

    `path=None` gives an EPHEMERAL store in a temporary directory, removed when
    the process exits: Edge is WAL-backed and has no in-memory mode.
    """

    def __init__(self, path: str | None, *, vectors: dict[str, int | Vectors],
                 tenant_field: str | None = None, distance: str = "Cosine") -> None:
        from qdrant_edge import (Distance, EdgeConfig, EdgeShard, EdgeVectorParams,
                                 KeywordIndexParams, MultiVectorComparator,
                                 MultiVectorConfig, UpdateOperation)

        self._lock = threading.RLock()
        self._closed = False
        self._temp_dir: str | None = None
        if path is None:
            self._temp_dir = tempfile.mkdtemp(prefix="edge-store-")
            path = os.path.join(self._temp_dir, "shard")
            # Removed at exit rather than in __del__: a shard holds file
            # handles, and interpreter shutdown ordering makes __del__ an
            # unreliable place to touch the filesystem.
            atexit.register(self._cleanup_temp)
        self._path = path

        if os.path.exists(os.path.join(path, "wal")):
            # An existing shard: load it, keeping everything stored before.
            self._shard = EdgeShard.load(path)
            return
        specs = {name: spec if isinstance(spec, Vectors) else Vectors(int(spec))
                 for name, spec in vectors.items()}
        config = EdgeConfig(vectors={
            name: EdgeVectorParams(
                size=spec.size, distance=getattr(Distance, distance),
                multivector_config=(MultiVectorConfig(MultiVectorComparator.MaxSim)
                                    if spec.multivector else None))
            for name, spec in specs.items()})
        # create() needs the directory to exist already, and fails on a
        # half-made one, so build the path before handing it over.
        os.makedirs(path, exist_ok=True)
        self._shard = EdgeShard.create(path, config)
        if tenant_field:
            self._shard.update(UpdateOperation.create_field_index(
                tenant_field, KeywordIndexParams(is_tenant=True)))

    @property
    def path(self) -> str:
        return self._path

    def size(self, name: str) -> int | None:
        """The size of a named vector, as the shard on disk was created."""
        try:
            with open(os.path.join(self._path, "edge_config.json")) as f:
                return int(json.load(f)["vectors"][name]["size"])
        except (OSError, KeyError, ValueError, TypeError):
            return None

    @staticmethod
    def new_id() -> str:
        return str(uuid.uuid4())

    # — writing —

    def add(self, point_id: str, vectors: dict[str, Any], payload: dict) -> None:
        """Insert or replace a point with its named vectors."""
        from qdrant_edge import Point, UpdateOperation

        with self._lock:
            self._shard.update(UpdateOperation.upsert_points(
                [Point(point_id, dict(vectors), payload)]))

    def set_vector(self, point_id: str, name: str, vector: Any) -> None:
        """Add or replace one named vector of an existing point."""
        from qdrant_edge import PointVectors, UpdateOperation

        with self._lock:
            self._shard.update(UpdateOperation.update_vectors(
                [PointVectors(point_id, {name: vector})]))

    def set_payload(self, point_id: str, payload: dict) -> None:
        """Merge fields into an existing point without re-embedding it."""
        from qdrant_edge import UpdateOperation

        with self._lock:
            self._shard.update(UpdateOperation.set_payload([point_id], payload))

    def delete(self, query_filter: Any) -> None:
        from qdrant_edge import UpdateOperation

        with self._lock:
            self._shard.update(UpdateOperation.delete_points_by_filter(query_filter))

    # — reading —

    def search(self, vector: Any, limit: int, query_filter: Any | None = None, *,
               using: str) -> list[dict]:
        """Nearest points by one named vector, best first, as
        `{**payload, "score": float}`. Only points that carry that vector
        can come back."""
        from qdrant_edge import Query, QueryRequest

        with self._lock:
            hits = self._shard.query(QueryRequest(
                query=Query.Nearest(vector, using=using), limit=limit,
                filter=query_filter, with_payload=True))
        return [{**(hit.payload or {}), "score": float(hit.score)} for hit in hits]

    def nearest_ids(self, vector: Any, limit: int, query_filter: Any | None = None, *,
                    using: str) -> list[tuple[str, float]]:
        """(id, score) of the nearest points by one named vector, best first —
        search() for a caller that needs the point itself and its score, not
        what it carries."""
        from qdrant_edge import Query, QueryRequest

        with self._lock:
            hits = self._shard.query(QueryRequest(
                query=Query.Nearest(vector, using=using), limit=limit,
                filter=query_filter, with_payload=False))
        return [(str(hit.id), float(hit.score)) for hit in hits]

    def iter_points(self, query_filter: Any | None = None,
                    fields: list[str] | None = None) -> "Iterator[tuple[str, dict]]":
        """Every stored point as (id, payload), a page at a time — only those
        matching `query_filter`, with only `fields` of the payload if given
        (a frame's payload carries its JPEG; a walk that needs a timestamp
        should not read every picture)."""
        from qdrant_edge import ScrollRequest

        offset = None
        while True:
            with self._lock:
                rows, offset = self._shard.scroll(ScrollRequest(
                    limit=SCROLL_PAGE, offset=offset, filter=query_filter,
                    with_payload=fields if fields is not None else True))
            for row in rows:
                yield str(row.id), (row.payload or {})
            if offset is None:
                return

    def get(self, point_ids: list[str]) -> list[tuple[str, dict]]:
        """Stored points by id, with their whole payload, in the order asked."""
        with self._lock:
            records = self._shard.retrieve(list(point_ids), True, False)
        rows = {str(row.id): (row.payload or {}) for row in records}
        return [(point_id, rows[point_id]) for point_id in point_ids if point_id in rows]

    def vectors(self, point_id: str) -> dict[str, Any]:
        """The named vectors of one point, or {} when there is no such point."""
        with self._lock:
            records = self._shard.retrieve([point_id], False, True)
        return dict(records[0].vector or {}) if records else {}

    def count(self, query_filter: Any | None = None) -> int:
        from qdrant_edge import CountRequest

        with self._lock:
            return self._shard.count(CountRequest(exact=True, filter=query_filter))

    # — lifecycle —

    def close(self) -> None:
        """Flush and let the directory go. Safe to call twice: two memories
        share one shard, and both are closed at shutdown."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            # Edge runs no background optimizer: segments left by a session's
            # updates and deletes are merged only when asked. On close, when
            # nothing is searching; a shard with nothing to merge returns at
            # once (measured: 600 points, one segment, no time).
            try:
                self._shard.optimize()
            except Exception:  # noqa: BLE001 — closing must still happen
                pass
            self._shard.close()

    def _cleanup_temp(self) -> None:
        if self._temp_dir and os.path.isdir(self._temp_dir):
            shutil.rmtree(self._temp_dir, ignore_errors=True)
            self._temp_dir = None


def match_value(key: str, value: Any):
    """A payload equality condition — `kind == "frame"` and the like."""
    from qdrant_edge import FieldCondition, Filter, MatchValue

    return Filter(must=[FieldCondition(key=key, match=MatchValue(value=value))])


def match_any(key: str, values: list):
    """A payload membership condition — a label in this set."""
    from qdrant_edge import FieldCondition, Filter, MatchAny

    return Filter(must=[FieldCondition(key=key, match=MatchAny(any=list(values)))])


def less_than(key: str, value: float):
    """A payload range condition — `ts < value` and the like."""
    from qdrant_edge import FieldCondition, Filter, RangeFloat

    return Filter(must=[FieldCondition(key=key, range=RangeFloat(lt=value))])


def all_of(*filters):
    """Every condition of every filter given (Nones skipped), or None."""
    from qdrant_edge import Filter

    conditions = [condition for f in filters if f is not None for condition in f.must]
    return Filter(must=conditions) if conditions else None
