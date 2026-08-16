"""Make `src/` importable from tests without installing the project."""
import os
import sys
from pathlib import Path

# Keep the local embedding stack from reaching for the network on import.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


class FakeCollection:
    """A stand-in for a Chroma collection over an in-memory list of chunks.

    The retriever queries the store rather than holding the corpus in memory, so
    tests need something that answers `get(ids=...)` and `get(where=...)`. This
    supports the equality and `$and` clauses the retriever actually builds.
    """

    def __init__(self, hits):
        self._hits = list(hits)

    @staticmethod
    def _matches(meta, where):
        if not where:
            return True
        clauses = where["$and"] if "$and" in where else [where]
        return all(meta.get(k) == v for c in clauses for k, v in c.items())

    def get(self, ids=None, where=None, include=None):
        sel = [h for h in self._hits
               if (ids is None or h["id"] in ids) and self._matches(h["metadata"], where)]
        return {"ids": [h["id"] for h in sel],
                "documents": [h["document"] for h in sel],
                "metadatas": [h["metadata"] for h in sel]}

    def count(self):
        return len(self._hits)


def make_retriever(hits):
    """A Retriever backed by `hits`, with no embedding model or real store."""
    from retrieval import Retriever
    r = Retriever.__new__(Retriever)
    r.collection = FakeCollection(hits)
    r._corpus = None
    r._bm25 = None
    r._reranker = None
    return r
