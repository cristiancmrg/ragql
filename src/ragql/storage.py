# SPDX-License-Identifier: MPL-2.0
"""
ragql.storage
~~~~~~~~~~~~~
SQLite + Faiss helpers for RagQL.
"""

from __future__ import annotations

import sqlite3
import logging
from hashlib import md5
from pathlib import Path
from typing import Callable, Iterable, List, Optional

import faiss
import numpy as np

from ragql.config import Settings
from ragql.embeddings import get_embeddings

logger = logging.getLogger(__name__)


# SQLite helpers
def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")  # better concurrency
    return conn


# ChunkStore – text & metadata
class ChunkStore:
    def __init__(self, db_path: Path):
        """Initialize the ChunkStore.

        Opens a connection to the SQLite database at `db_path` and ensures
        the required schema is in place.

        Args:
            db_path (Path): Path to the SQLite database file.
        """
        self.db_path = db_path
        self.conn = _connect(db_path)
        self._ensure_schema()
        logger = logging.getLogger(__name__)
        logger.debug("ChunkStore init")

    # ---------- private -------------------------------------------------

    def _ensure_schema(self) -> None:
        """Ensure the 'chunks' table and necessary columns exist in the database.

        If the 'chunks' table does not exist, it will be created with columns:
          - hash (TEXT PRIMARY KEY)
          - file (TEXT)
          - start (INT)
          - text (TEXT)
          - model (TEXT)

        If the 'model' column is missing, it will be added with a default empty string.

        Returns:
            None
        """
        logger.debug("Ensuring schema")
        cur = self.conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chunks(
                hash TEXT PRIMARY KEY, 
                file TEXT,
                start INT, 
                text TEXT, 
                model TEXT 
            )
        """)
        cols = [row[1] for row in cur.execute("PRAGMA table_info(chunks)").fetchall()]
        if "model" not in cols:
            cur.execute("ALTER TABLE chunks ADD COLUMN model TEXT DEFAULT ''")
        self.conn.commit()

    # ---------- public --------------------------------------------------

    @staticmethod
    def make_hash(doc_id: str, idx: int) -> str:
        """Generate a deterministic MD5 hash for a document chunk.

        Combines the document identifier and chunk index to produce
        a unique hash string.

        Args:
            doc_id (str): Identifier of the source document.
            idx (int): Index of the chunk within the document.

        Returns:
            str: MD5 hash of the form md5(f"{doc_id}:{idx}").
        """
        logger.debug("Creating hash for the document")
        return md5(f"{doc_id}:{idx}".encode()).hexdigest()

    def add(self, h: str, file: str, start: int, text: str, model: str) -> None:
        """Insert a new chunk into the store if it doesn't already exist.

        Uses INSERT OR IGNORE to avoid duplicates based on the primary-key hash.

        Args:
            h (str): Hash identifier for the chunk.
            file (str): Originating file path or name.
            start (int): Character offset at which this chunk begins.
            text (str): The textual content of the chunk.
            model (str): Name of the embedding/model source.

        Returns:
            None
        """
        logger.debug("Adding data to the ChunkStoree")
        self.conn.execute(
            "INSERT OR IGNORE INTO chunks(hash, file, start, text, model) VALUES (?,?,?,?,?)",
            (h, file, start, text, model),
        )
        self.conn.commit()

    def build_context(
        self,
        hits: list[tuple[str, float]],
        max_len: int = 140,
        model: str | None = None,
    ) -> str:
        """Construct a formatted context string from matching chunks.

        Fetches stored chunks whose hashes match the provided `hits`,
        optionally filtering by `model`, and returns a multi-line string
        with each line showing the file name, a text excerpt (up to
        `max_len` characters), and its similarity score.

        Args:
            hits (list[tuple[str, float]]): List of (hash, score) tuples,
                in descending order of relevance.
            max_len (int, optional): Maximum characters of text excerpt to include.
                Defaults to 140.
            model (str, optional): If given, only include chunks created with this model.

        Returns:
            str: Joined lines of the form "[file] excerpt … (sim 0.XX)".
        """  # Build WHERE clauses
        hashes = [h for h, _ in hits]
        params = hashes[:]
        where = f"hash IN ({','.join('?' * len(hashes))})"
        if model:
            where += " AND model = ?"
            params.append(model)

        cur = self.conn.execute(
            f"SELECT hash, file, text FROM chunks WHERE {where}", params
        )
        row_map = {h: (f, t) for h, f, t in cur.fetchall()}

        lines = []
        for h, score in hits:
            if h not in row_map:
                continue  # either a mismatch in model or a missing chunk
            f, t = row_map[h]
            lines.append(f"[{f}] {t[:max_len]} … (sim {score:.2f})")

        return "\n".join(lines)


# VectorStore – blobs & Faiss index
class VectorStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn = _connect(db_path)
        self._ensure_schema()
        self.index: faiss.Index | None = None
        self._hashes: list[str] = []

    # ---------- public --------------------------------------------------

    def has_vector(self, h: str) -> bool:
        condition = (
            self.conn.execute("SELECT 1 FROM vectors WHERE hash=?", (h,)).fetchone()
            is not None
        )

        if condition:
            logger.debug("The vector exists!")
        else:
            logger.debug("The vector doesn't exists!")

        return condition

    def add_vectors(self, ids: list[str], vecs: np.ndarray) -> None:
        """
        ids : list of md5 strings
        vecs: (N, dim) float32
        """
        logger.debug("Trying to add an vector")
        cur = self.conn.cursor()
        for h, v in zip(ids, vecs):
            cur.execute("INSERT OR REPLACE INTO vectors VALUES (?,?)", (h, v.tobytes()))
        self.conn.commit()

    # - - - Faiss --------------------------------------------------------

    def load_faiss(self) -> None:
        logger.debug("Loading faiss")
        cur = self.conn.execute("SELECT hash, vec FROM vectors")
        rows = cur.fetchall()
        if not rows:
            logger.error("Faiss loaded successfully")
            raise RuntimeError("VectorStore is empty")

        mat = np.vstack([np.frombuffer(v, dtype="float32") for _, v in rows])
        faiss.normalize_L2(mat)
        index = faiss.IndexFlatIP(mat.shape[1])
        index.add(x=mat)

        self.index = index
        self._hashes = [h for h, _ in rows]
        logger.debug("Faiss loaded successfully")

    def search(
        self, qvec: np.ndarray, top_k: int = 6
    ) -> list[tuple[str, float]]:  # [(hash, score)]
        logger.debug(f"Searching top_k: {top_k} embeddings")
        logger.debug(f"{qvec}")

        if self.index is None:
            logger.error("Faiss index not loaded; call load_faiss() first")
            raise RuntimeError("Faiss index not loaded; call load_faiss() first")

        qvec = qvec.astype("float32").reshape(1, -1)
        faiss.normalize_L2(qvec)

        distancies, indicies = self.index.search(x=qvec, k=top_k)

        logger.debug(
            f"Results of the embeddings search, distancies: {distancies}, indicies: {indicies}"
        )

        return [
            (self._hashes[i], float(distancies[0][rank]))
            for rank, i in enumerate(indicies[0])
            if i != -1
        ]

    # ---------- private -------------------------------------------------

    def _ensure_schema(self) -> None:
        logger.debug("Ensuring VectorStore schema")
        cur = self.conn.cursor()
        cur.execute(
            "CREATE TABLE IF NOT EXISTS vectors(hash TEXT PRIMARY KEY, vec  BLOB)"
        )
        self.conn.commit()


# Convenience: build everything in one call (optional helper)
def ingest_vectors(
    chunk_store: ChunkStore,
    vec_store: VectorStore,
    docs: Iterable[tuple[str, str]],
    chunker: Callable[[str], Iterable[str]],
    cfg: Settings,
    embed_fn: Optional[Callable[[list[str]], np.ndarray]] = None,
) -> None:
    """
    Full ingestion pipeline: chunk texts, embed new chunks, and store results.

    This helper performs two passes over the input documents:

    1. **Chunk & store**
       Splits each document’s text into chunks using `chunker`, computes a
       unique hash for each chunk, and inserts any previously‐unseen chunks
       into `chunk_store` along with the `cfg.embed_model` tag.
    2. **Embed & index**
       Batches all new chunks, obtains their embedding vectors via `embed_fn`
       (or the default `get_embeddings(texts, cfg)`), and adds those vectors
       to `vec_store`.

    Args:
        chunk_store (ChunkStore): Storage backend for text chunks.
        vec_store (VectorStore): Storage backend for embedding vectors.
        docs (Iterable[tuple[str, str]]): Sequence of `(doc_id, text)` pairs.
        chunker (Callable[[str], Iterable[str]]): Function that splits a text
            string into an iterable of chunk strings.
        cfg (Settings): Configuration object providing `embed_model` and
            provider information.
        embed_fn (Optional[Callable[[list[str]], np.ndarray]]): A function that
            takes a list of chunk strings and returns a NumPy array of
            embeddings. If `None`, defaults to `get_embeddings(texts, cfg)`.

    Raises:
        ValueError: If `cfg.embed_provider` is not supported and no valid
            `embed_fn` is supplied.

    Side Effects:
        - Inserts new chunks into `chunk_store`.
        - Writes embeddings for those chunks into `vec_store`.
    """

    logger.debug("Ingesting vectors with model %r", cfg.embed_model)

    # default embed function to your centralized helper
    if embed_fn is None:

        def default_embed_fn(texts: List[str]) -> np.ndarray:
            return get_embeddings(texts, cfg)

        embed_fn = default_embed_fn

    new_ids: list[str] = []
    new_chunks: list[str] = []

    # pass 1 – collect new chunks
    for doc_id, text in docs:
        for idx, chunk in enumerate(chunker(text)):
            h = chunk_store.make_hash(doc_id, idx)
            if not vec_store.has_vector(h):
                new_ids.append(h)
                new_chunks.append(chunk)
                # store chunk along with its embed_model identifier
                chunk_store.add(h, doc_id, idx, chunk, cfg.embed_model)

    # pass 2 – embed & store
    if new_chunks:
        vecs = embed_fn(new_chunks)
        vec_store.add_vectors(new_ids, vecs)
