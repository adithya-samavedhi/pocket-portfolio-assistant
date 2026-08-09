"""Unified retrieval: dense, BM25, and hybrid (RRF fusion) — Milestone 4.

Dense retrieval (local embeddings over Chroma) handles semantics and company
routing. BM25 over the raw chunk text handles exact financial terms and figures
that dense embeddings smear. Reciprocal Rank Fusion combines the two rankings.

Both the CLI (query.py) and the eval (run_eval.py) go through this module so a
change here is measured the same way it ships.

Milestone 4 finding: on the eval set, neither hybrid (BM25+dense) nor
cross-encoder reranking beats plain dense retrieval. The Milestone 2 context
prefix (ticker/type/period/section baked into each embedding) already makes
dense strong at company/section routing, and the section-level gold gives BM25's
exact-figure strength nothing to grab. Hybrid hurt (company-blind BM25 pollutes
top ranks); reranking hurt overall but lifted cross-quarter rank-1 — a weakness
Milestone 5 addresses structurally via per-period fan-out. So `dense` is the
default; `hybrid`/`rerank` remain available as measured, one-flag experiments.
"""
import re
from collections import defaultdict
from typing import List, Dict, Optional

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

import config

_POOL = 50       # candidates pulled from each retriever before fusion
_RRF_K = 60      # standard RRF damping constant
_RERANK_POOL = 20  # dense candidates fed to the cross-encoder reranker
_TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")

# Cross-encoder reranker from the same family as the bge embedder; loaded lazily.
_RERANK_MODEL = "BAAI/bge-reranker-base"


def tokenize(text: str) -> List[str]:
    """Lowercase word/number tokens; keeps decimals like '119.8' intact."""
    return _TOKEN.findall(text.lower())


def _matches(meta: dict, where: Optional[dict]) -> bool:
    """Minimal equality matcher mirroring the CLI's Chroma `where` clauses."""
    if not where:
        return True
    clauses = where["$and"] if "$and" in where else [where]
    return all(meta.get(k) == v for c in clauses for k, v in c.items())


class Retriever:
    def __init__(self):
        self.model = SentenceTransformer(config.EMBED_MODEL)
        client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        self.collection = client.get_collection(config.COLLECTION)

        # Pull the whole corpus once to build the BM25 index (aligned arrays).
        got = self.collection.get(include=["documents", "metadatas"])
        self.ids = got["ids"]
        self.docs = got["documents"]
        self.metas = got["metadatas"]
        self._by_id = {i: (d, m) for i, d, m in zip(self.ids, self.docs, self.metas)}
        self._bm25 = None       # lazy: only BM25/hybrid need it
        self._reranker = None   # lazy: only rerank needs it

    @property
    def bm25(self):
        if self._bm25 is None:
            self._bm25 = BM25Okapi([tokenize(d) for d in self.docs])
        return self._bm25

    @property
    def reranker(self):
        if self._reranker is None:
            self._reranker = CrossEncoder(_RERANK_MODEL)
        return self._reranker

    def _rerank(self, query: str, ids: List[str], k: int) -> List[str]:
        # Prepend the section label so the cross-encoder sees which part of the
        # filing a passage is from, then score each (query, passage) pair.
        pairs, cand = [], []
        for _id in ids:
            doc, meta = self._by_id[_id]
            pairs.append([query, f"{meta.get('section', '')}\n{doc}"])
            cand.append(_id)
        scores = self.reranker.predict(pairs)
        order = sorted(range(len(cand)), key=lambda i: scores[i], reverse=True)
        return [cand[i] for i in order[:k]]

    # --- individual rankers return an ordered list of chunk ids ---
    def _dense_ids(self, query: str, pool: int, where) -> List[str]:
        emb = self.model.encode([query], normalize_embeddings=True).tolist()
        res = self.collection.query(query_embeddings=emb, n_results=pool, where=where)
        return res["ids"][0]

    def _bm25_ids(self, query: str, pool: int, where) -> List[str]:
        scores = self.bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        out = []
        for i in order:
            if _matches(self.metas[i], where):
                out.append(self.ids[i])
            if len(out) >= pool:
                break
        return out

    @staticmethod
    def _rrf(rank_lists: List[List[str]], weights: Optional[List[float]] = None) -> List[str]:
        weights = weights or [1.0] * len(rank_lists)
        scores = defaultdict(float)
        for ids, w in zip(rank_lists, weights):
            for rank, _id in enumerate(ids):
                scores[_id] += w / (_RRF_K + rank + 1)
        return sorted(scores, key=scores.get, reverse=True)

    def search(self, query: str, k: int = 5, method: str = "dense",
               where: Optional[dict] = None,
               weights: Optional[List[float]] = None) -> List[Dict]:
        if method == "dense":
            ids = self._dense_ids(query, k, where)
        elif method == "bm25":
            ids = self._bm25_ids(query, k, where)
        elif method == "hybrid":
            fused = self._rrf([self._dense_ids(query, _POOL, where),
                               self._bm25_ids(query, _POOL, where)],
                              weights=weights or [2.0, 1.0])
            ids = fused[:k]
        elif method == "rerank":
            # Cross-encoder over dense candidates (already company-correct).
            ids = self._rerank(query, self._dense_ids(query, _RERANK_POOL, where), k)
        else:
            raise ValueError(f"unknown method: {method}")

        hits = []
        for _id in ids[:k]:
            doc, meta = self._by_id[_id]
            hits.append({"id": _id, "document": doc, "metadata": meta})
        return hits

    # --- Milestone 5: temporal fan-out ---
    def periods_for(self, ticker: str, filing_type: Optional[str] = None,
                    period_type: Optional[str] = "quarter"):
        """Distinct (fiscal_period, period_end) for a ticker, oldest first.

        `period_type` defaults to "quarter": a 10-K's fiscal year *contains* the
        quarters around it, so including it in a fan-out invites the model to
        read a year-vs-quarter jump as a trend. Pass None to include everything.
        """
        seen = {}
        for m in self.metas:
            if m.get("ticker") != ticker:
                continue
            if filing_type and m.get("filing_type") != filing_type:
                continue
            if period_type and m.get("period_type", "quarter") != period_type:
                continue
            # period_end is the calendar anchor, so chronological ordering works
            # across fiscal calendars (fiscal labels alone do not sort).
            seen[m["fiscal_period"]] = m.get("period_end") or m.get("report_date", "")
        return sorted(seen.items(), key=lambda kv: kv[1])

    def temporal_search(self, query: str, ticker: str, k_per_period: int = 2,
                        filing_type: Optional[str] = None,
                        section: Optional[str] = None,
                        period_type: Optional[str] = "quarter") -> List[Dict]:
        """Fan out: retrieve top-k *within each period* so no quarter is crowded
        out. Returns period groups ordered chronologically. This is what makes
        "how has X changed over N quarters" answerable — every period is present.

        Annual filings are excluded by default; see `periods_for`.
        """
        # Over-fetch when section-filtering so we can still fill k_per_period.
        pool = k_per_period * 8 if section else k_per_period
        needle = section.lower() if section else None
        groups = []
        for period, report_date in self.periods_for(ticker, filing_type, period_type):
            where = {"$and": [{"ticker": ticker}, {"fiscal_period": period}]}
            hits = []
            for _id in self._dense_ids(query, pool, where):
                doc, meta = self._by_id[_id]
                if needle and needle not in meta.get("section", "").lower():
                    continue
                hits.append({"id": _id, "document": doc, "metadata": meta})
                if len(hits) >= k_per_period:
                    break
            groups.append({"period": period, "report_date": report_date, "hits": hits})
        return groups
