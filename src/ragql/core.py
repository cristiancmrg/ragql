# SPDX-License-Identifier: MPL-2.0
"""
ragql.core
~~~~~~~~~~
High-level façade that ties together loaders, embeddings,
vector storage, and the answering engine.
"""

from __future__ import annotations
from pathlib import Path
from typing import Iterable, Iterator

from .config import Settings
from .loaders import REGISTRY
from .loaders import Doc
from .storage import VectorStore, ChunkStore
import logging

logger = logging.getLogger(__name__)


class RagQL:
    """
    Orchestrates end-to-end retrieval-augmented Q&A.

    Typical use:
        >>> rq = RagQL(Path("my/logs"), Settings())
        >>> rq.build()                 # embed + index
        >>> answer = rq.query("Why ...?")
    """

    def __init__(self, root: Path, cfg: Settings):
        self.root = root
        self.cfg = cfg
        self.chunks = ChunkStore(cfg.db_path)
        self.vstore = VectorStore(cfg.db_path)
        self.logger = logger

        self.logger.debug(
            "RagQL init: root=%s, db=%s, verbose=%s", root, cfg.db_path, cfg.verbose
        )

    # Indexing:

    def _iter_documents(self) -> Iterator[Doc]:
        """
        Walk every file under self.root, hand it off to each loader in REGISTRY,
        and yield back any (doc_id, text) tuples they emit.
        """
        self.logger.debug("Scanning for documents under %s", self.root)
        for path in self.root.rglob("*"):
            if not path.is_file():
                continue

            for load in REGISTRY:  # now `load` is Callable[[Path], Iterable[Doc]]
                try:
                    docs = load(path)  # call the loader function directly
                except Exception as e:
                    # loader didn’t like this file, skip it
                    self.logger.debug("Loader %r skipped %s (%s)", load, path, e)
                    continue

                if not docs:
                    # loader returned empty or None, skip
                    continue

                # finally, yield each (doc_id, text)
                for doc_id, text in docs:
                    self.logger.debug("Discovered doc %r from %s", doc_id, path)
                    yield doc_id, text

    def build(self) -> None:
        """
        1) Harvest new chunks
        2) Embed + index them
        3) Load FAISS index for querying
        """
        from .embeddings import get_embeddings

        self.logger.info("Starting build(): scanning and indexing new chunks")

        new_texts = []
        new_ids = []

        for doc_id, text in self._iter_documents():
            for idx, chunk in enumerate(self._chunk(text)):
                h = self.chunks.make_hash(doc_id, idx)
                if not self.vstore.has_vector(h):
                    new_ids.append(h)
                    new_texts.append(chunk)
                    self.chunks.add(h, doc_id, idx, chunk, self.cfg.embed_model_name)

        self.logger.info("Found %d new chunks to embed", len(new_texts))

        if new_texts:
            self.logger.debug("Calling get_embeddings on %d texts", len(new_texts))
            vecs = get_embeddings(new_texts, self.cfg)
            self.logger.debug(
                "Received embeddings shape %r", getattr(vecs, "shape", None)
            )
            self.vstore.add_vectors(new_ids, vecs)
            self.logger.info("Added %d vectors to the store", len(new_ids))
        else:
            self.logger.info("No new chunks—skipping embedding step")

        self.logger.info("Loading FAISS index into memory")
        self.vstore.load_faiss()  # ready for search
        self.logger.info("Build complete: ready to answer queries")

    # Querying:

    def query(self, prompt: str, top_k: int = 6) -> str:
        """
        1) Embed the user prompt
        2) Search the vector store
        3) Build context
        4) Delegate to LLM
        """

        from .embeddings import get_embeddings  # <- lazy import

        self.logger.info("Received query: %r (top_k=%d)", prompt, top_k)
        vec = get_embeddings([prompt], self.cfg)
        self.logger.debug("Prompt embedding shape: %r", getattr(vec, "shape", None))

        hits = self.vstore.search(vec, top_k)
        self.logger.info("Search returned %d hits", len(hits))
        self.logger.debug("Hits detail: %r", hits)

        context = self.chunks.build_context(hits)
        self.logger.debug("Built context (len=%d chars)", len(context))

        answer = self._call_llm(prompt, context)
        self.logger.info("LLM returned answer of length %d", len(answer))

        return answer

    # Helpers:

    def _chunk(self, text: str) -> Iterable[str]:
        words = text.split()
        step = self.cfg.chunk_size - self.cfg.chunk_overlap
        for i in range(0, len(words), step):
            yield " ".join(words[i : i + self.cfg.chunk_size])

    def _call_llm(self, prompt: str, context: str) -> str:
        """
        Route through Ollama or OpenAI depending on cfg.
        """
        # defer heavy import
        if self.cfg.use_ollama:
            self.logger.info("Using Ollama for chat completion")
            from .embeddings import call_ollama_chat

            return call_ollama_chat(prompt, context, self.cfg)
        else:
            self.logger.info("Using OpenAI for chat completion")
            from .embeddings import call_openai_chat

            return call_openai_chat(prompt, context, self.cfg)
