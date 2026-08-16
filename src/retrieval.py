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

        # The corpus is NOT loaded here. It used to be, to build a BM25 index —
        # at ~13KB of Python objects per chunk that is 3.9GB for a 50-company,
        # 5-year corpus, which fits on no free host. Everything the default
        # (dense) path needs is answered by targeted Chroma queries instead; the
        # full load happens only if BM25/hybrid is explicitly asked for, and the
        # eval says those lose to dense anyway (MRR 0.545/0.731 vs 0.853).
        self._corpus = None     # lazy: (ids, docs, metas) — BM25 only
        self._bm25 = None       # lazy: only BM25/hybrid need it
        self._reranker = None   # lazy: only rerank needs it

    # --- corpus access ---
    @property
    def corpus(self):
        """The whole corpus in memory. Only BM25 needs this; it is expensive."""
        if self._corpus is None:
            got = self.collection.get(include=["documents", "metadatas"])
            self._corpus = (got["ids"], got["documents"], got["metadatas"])
        return self._corpus

    @property
    def ids(self):
        return self.corpus[0]

    @property
    def docs(self):
        return self.corpus[1]

    @property
    def metas(self):
        return self.corpus[2]

    def _fetch(self, ids: List[str]) -> Dict[str, tuple]:
        """Look up documents by id straight from the store, without holding the
        corpus in memory."""
        if not ids:
            return {}
        got = self.collection.get(ids=ids, include=["documents", "metadatas"])
        return {i: (d, m) for i, d, m in
                zip(got["ids"], got["documents"], got["metadatas"])}

    def _where_get(self, where: dict) -> List[Dict]:
        """All chunks matching a where clause, as hit dicts."""
        got = self.collection.get(where=where, include=["documents", "metadatas"])
        return [{"id": i, "document": d, "metadata": m} for i, d, m in
                zip(got["ids"], got["documents"], got["metadatas"])]

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
        got = self._fetch(ids)
        pairs, cand = [], []
        for _id in ids:
            doc, meta = got[_id]
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

    @staticmethod
    def _with(where: Optional[dict], extra: dict) -> dict:
        """AND an extra condition onto an existing where clause."""
        if not where:
            return extra
        return {"$and": [where, extra]}

    def search(self, query: str, k: int = 5, method: str = "dense",
               where: Optional[dict] = None,
               weights: Optional[List[float]] = None,
               table_quota: int = 0) -> List[Dict]:
        """Retrieve k passages.

        `table_quota` ADDS table chunks on top of the k passages — it does not
        take slots from them. Without it, financial tables never surface: a
        question says "segments" while the table says "| Compute & Networking |
        193,479 |", so there is neither a semantic nor a lexical bridge, and
        narrative prose wins every slot. Measured on "top revenue growing
        segments": 0 tables in the top 8 under dense, bm25 AND hybrid — but the
        right tables rank 4th-5th once they only compete with each other.

        Additive rather than displacing, because taking prose slots measurably
        cost recall (MRR 0.886 -> 0.875) for no reason: a sub-agent's own context
        is allowed to be large, and only its *summary* is budgeted.
        """
        if table_quota > 0:
            # Leave the normal ranking completely untouched, then top up with
            # any tables it missed. Filtering the main k to text-only instead
            # would demote a table that legitimately ranked first.
            base = self.search(query, k=k, method=method, where=where, weights=weights)
            seen = {h["id"] for h in base}
            extra = [h for h in self.search(query, k=table_quota + len(base), method=method,
                                            where=self._with(where, {"chunk_kind": "table"}),
                                            weights=weights)
                     if h["id"] not in seen][:table_quota]
            return base + extra            # tables last: closest to the question
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

        chosen = ids[:k]
        got = self._fetch(chosen)
        return [{"id": i, "document": got[i][0], "metadata": got[i][1]}
                for i in chosen if i in got]

    def section_chunks(self, ticker: str, fiscal_period: str, section: str,
                       filing_type: Optional[str] = None) -> List[Dict]:
        """Every chunk of one section of one filing, in document order.

        Retrieval returns fragments; an analytical question ("what is the moat",
        "what are the headwinds") is answered by a whole section, because the
        argument is distributed across it. k=6 shows ~1.2% of a 10-K.
        """
        clauses = [{"ticker": ticker}, {"fiscal_period": fiscal_period},
                   {"section": section}]
        if filing_type:
            clauses.append({"filing_type": filing_type})
        out = self._where_get({"$and": clauses})
        # chunk_index restores the order the filing was written in, which reads
        # far better than similarity order for a continuous argument.
        return sorted(out, key=lambda h: h["metadata"].get("chunk_index", 0))

    def expand_sections(self, hits: List[Dict], budget_chars: int,
                        max_sections: int = 2) -> List[Dict]:
        """Grow the retrieved hits into the full sections they came from.

        Sections are taken in the order retrieval ranked them, so the best
        section is filled in first, and the whole thing is capped: Item 1A alone
        can run to 130k characters, which is more than a cheap model should be
        asked to read.
        """
        seen, groups = set(), []
        for h in hits:
            m = h["metadata"]
            key = (m.get("ticker"), m.get("fiscal_period"), m.get("section"),
                   m.get("filing_type"))
            if key not in seen:
                seen.add(key)
                groups.append(key)
            if len(groups) >= max_sections:
                break

        # Start from what retrieval already found, so expansion can only ever
        # ADD context. Filling only the top sections silently dropped passages
        # the chunk search had got right (measured: two questions went to zero).
        out = list(hits)
        have = {h["id"] for h in out}
        used = sum(len(h["document"]) for h in out)

        for ticker, period, section, ftype in groups:
            for h in self.section_chunks(ticker, period, section, ftype):
                if h["id"] in have:
                    continue
                n = len(h["document"])
                if used + n > budget_chars:
                    return out
                out.append(h)
                have.add(h["id"])
                used += n
        return out

    # --- Milestone 5: temporal fan-out ---
    def periods_for(self, ticker: str, filing_type: Optional[str] = None,
                    period_type: Optional[str] = "quarter"):
        """Distinct (fiscal_period, period_end) for a ticker, oldest first.

        `period_type` defaults to "quarter": a 10-K's fiscal year *contains* the
        quarters around it, so including it in a fan-out invites the model to
        read a year-vs-quarter jump as a trend. Pass None to include everything.
        """
        seen = {}
        got = self.collection.get(where={"ticker": ticker}, include=["metadatas"])
        for m in got["metadatas"]:
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
                        period_type: Optional[str] = "quarter",
                        table_quota: int = 0) -> List[Dict]:
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
            # Reserve table slots per period, so a quarter contributes its
            # figures and not only its commentary.
            n_tab, hits = table_quota, []
            if n_tab:
                hits = self.search(query, k=n_tab, where=self._with(where, {"chunk_kind": "table"}))
            seen = {h["id"] for h in hits}
            cand = [i for i in self._dense_ids(query, pool + n_tab, where) if i not in seen]
            fetched = self._fetch(cand)
            for _id in cand:
                if _id not in fetched:
                    continue
                doc, meta = fetched[_id]
                if needle and needle not in meta.get("section", "").lower():
                    continue
                hits.insert(len(hits) - n_tab if n_tab else len(hits),
                            {"id": _id, "document": doc, "metadata": meta})
                if len(hits) >= k_per_period + n_tab:
                    break
            groups.append({"period": period, "report_date": report_date, "hits": hits})
        return groups
