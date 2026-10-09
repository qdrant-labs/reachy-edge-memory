"""TextMemory: what was said, after it left the model's context — stored in
the robot's Qdrant Edge shard and searched by meaning.

One test uses the real BAAI/bge-small-en-v1.5 model, when it is already
cached: a search by meaning only means something against the real model. The
rest use a deterministic fake, the same reasoning as test_frame_memory.py's
FakeEmbedder.
"""

from __future__ import annotations

import numpy as np
import pytest

from emulator.memory import EXCHANGE_KIND, TextMemory

DOG = "Person: my dog is called Rex. — Reachy: What a lovely name!"


def _bge_cached() -> bool:
    """Whether fastembed has bge on disk — checked without downloading, so
    a fresh clone runs the suite offline."""
    try:
        from fastembed import TextEmbedding

        TextEmbedding("BAAI/bge-small-en-v1.5", local_files_only=True, lazy_load=True)
        return True
    except Exception:  # noqa: BLE001 — not cached, or fastembed missing
        return False


class _FakeBgeEmbedder:
    """Two dimensions: one for anything about the dog, one for the rest."""

    def _vector(self, text):
        low = text.lower()
        return np.asarray([1.0, 0.0] if ("dog" in low or "rex" in low) else [0.0, 1.0],
                          np.float32)

    def embed(self, texts):
        return [self._vector(t) for t in texts]

    query_embed = embed


@pytest.mark.skipif(not _bge_cached(), reason="bge-small is not cached")
def test_a_stored_exchange_is_found_by_meaning():
    memory = TextMemory(path=None)
    memory.remember(DOG, EXCHANGE_KIND, {"said_at": 1.0})
    memory.remember("Person: I work on databases. — Reachy: Interesting!",
                    EXCHANGE_KIND, {"said_at": 2.0})
    hits = memory.recall_exchanges("what's my pet called?")
    assert hits and hits[0]["text"] == DOG
    assert hits[0]["score"] >= 0.62


def test_an_injected_embedder_is_used_instead_of_the_default_model():
    from emulator.memory import _embedder_cache

    _embedder_cache.clear()
    memory = TextMemory(path=None, embedder=_FakeBgeEmbedder())
    memory.remember(DOG)
    assert memory.recall_exchanges("the dog")[0]["text"] == DOG
    assert _embedder_cache == {}, "an injected embedder must not load fastembed"


def test_a_question_nothing_answers_finds_nothing():
    memory = TextMemory(path=None, embedder=_FakeBgeEmbedder())
    memory.remember(DOG)
    assert memory.recall_exchanges("what time is it?") == []


def test_this_memory_stores_exchanges_only():
    memory = TextMemory(path=None, embedder=_FakeBgeEmbedder())
    with pytest.raises(ValueError):
        memory.remember("a mug", "object")


def test_the_memory_survives_a_restart_and_is_not_stored_twice(tmp_path):
    path = str(tmp_path / "memory")
    first = TextMemory(path=path, embedder=_FakeBgeEmbedder())
    first.remember(DOG)
    first.close()
    second = TextMemory(path=path, embedder=_FakeBgeEmbedder())
    assert second.recall_exchanges("the dog")[0]["text"] == DOG
    second.remember(DOG)   # flushed again at shutdown after an eviction
    assert second.count() == 1
    second.close()


def test_latest_exchanges_are_the_last_ones_said_oldest_first():
    memory = TextMemory(embedder=_FakeBgeEmbedder())
    for said_at, text in ((3, "third"), (1, "first"), (2, "second")):
        memory.remember(text, EXCHANGE_KIND, {"said_at": said_at})
    assert memory.latest_exchanges(2) == ["second", "third"]


def test_score_texts_uses_the_same_scale_as_the_shard():
    memory = TextMemory(embedder=_FakeBgeEmbedder())
    assert memory.score_texts("the dog", [DOG, "the weather"]) == pytest.approx([1.0, 0.0])
    assert memory.score_texts("anything", []) == []


def test_the_conversation_and_the_frames_share_one_shard_without_mixing():
    from emulator.edge_store import IMAGE, KIND, TEXT, EdgeStore
    from emulator.frame_memory import FrameMemory

    class Words:
        def embed(self, texts):
            return [np.asarray([1.0, 0.0, 0.0] if "lamp" in t else [0.0, 1.0, 0.0], np.float32)
                    for t in texts]

        query_embed = embed

    class Pictures:
        def embed_image(self, frame):
            return np.asarray([1.0, 0.0], np.float32)

    store = EdgeStore(None, vectors={TEXT: 3, IMAGE: 2}, tenant_field=KIND)
    speech = TextMemory(store=store, embedder=Words())
    frames = FrameMemory(store=store, embedder=Pictures(), text_embedder=Words())
    speech.remember("Sasha: where is the lamp? — Reachy: on the left")
    frame = frames.remember(np.zeros((4, 4, 3), np.uint8), [], {"looked": "left"})
    frames.describe(frame, "I see a lamp.")

    assert (speech.count(), frames.count()) == (1, 1)
    assert [hit["kind"] for hit in speech.recall_exchanges("lamp")] == ["exchange"]
    assert store.search([1.0, 0.0, 0.0], 5, using=TEXT)[0]["score"] > 0.99  # the caption is searchable
    assert frames.latest_looks()[0]["caption"] == "I see a lamp."
    reopened = TextMemory(store=store, embedder=Words())
    assert reopened.count() == 1, "the frames are not read back as exchanges"
