# SPDX-License-Identifier: MPL-2.0
from ragql.storage import ChunkStore, VectorStore
from ragql.config import Settings
import numpy as np


def test_chunk_and_vector_roundtrip(tmp_path):
    db = tmp_path / "vec.db"
    cfg = Settings(db)
    chunks = ChunkStore(db)
    vecs = VectorStore(db)

    h = chunks.make_hash("file.txt", 0)
    chunks.add(h, "file.txt", 0, "hello world", cfg.embed_model)
    assert chunks.build_context([(h, 1.0)])  # non-empty

    x = np.ones((1, 3), dtype="float32")
    vecs.add_vectors([h], x)
    vecs.load_faiss()
    hits = vecs.search(x, 1)
    assert hits and hits[0][0] == h
