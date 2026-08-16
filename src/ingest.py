"""Ingest filings: parse -> section-aware chunk -> embed -> store in Chroma.

Usage:
    python src/ingest.py                 # ingest only the latest 10-K
    python src/ingest.py --all           # ingest every filing in the manifest
    python src/ingest.py --reset         # drop the collection first

Each chunk is tagged with its section (e.g. "Item 1A: Risk Factors") and kind
("text" or "table"), plus the filing's ticker/type/period/date/source_url.
The embedding model is local (sentence-transformers); no API key needed.
"""
import argparse
import json

import chromadb
from sentence_transformers import SentenceTransformer

import config
from parse import load_and_parse
from sections import chunk_filing

MANIFEST = config.RAW / "manifest.json"


def load_manifest():
    with open(MANIFEST) as f:
        return json.load(f)


def get_collection(client, reset: bool):
    if reset:
        try:
            client.delete_collection(config.COLLECTION)
            print(f"Dropped existing collection '{config.COLLECTION}'.")
        except Exception:
            pass
    # cosine space matches the normalized bge embeddings we produce below.
    # The embedding model is recorded so a later model swap can't silently
    # mix incompatible vectors into one collection — the failure mode there is
    # not an error but quietly worse retrieval, which is far harder to notice.
    coll = client.get_or_create_collection(
        config.COLLECTION,
        metadata={"hnsw:space": "cosine", "embed_model": config.EMBED_MODEL},
    )
    stored = (coll.metadata or {}).get("embed_model")
    if stored and stored != config.EMBED_MODEL:
        raise SystemExit(
            f"Collection '{config.COLLECTION}' was built with '{stored}' but "
            f"config.EMBED_MODEL is now '{config.EMBED_MODEL}'. Mixing vectors "
            f"from different models silently degrades retrieval.\n"
            f"Re-ingest from scratch:  python src/ingest.py --all --reset"
        )
    return coll


def ingest_file(entry, model, collection):
    path = config.RAW / entry["ticker"] / entry["file"]
    text, tables = load_and_parse(path)
    chunks = chunk_filing(text, tables, entry["filing_type"])

    base_meta = {
        "ticker": entry["ticker"],
        "filing_type": entry["filing_type"],
        # The company's own label ("Q2 FY2026") — agrees with the filing text.
        "fiscal_period": entry["fiscal_period"],
        "fiscal_year": entry["fiscal_year"],
        "fiscal_quarter": entry["fiscal_quarter"],
        # "quarter" | "annual" — keeps a 10-K's full year from being compared
        # against the quarters it contains.
        "period_type": entry["period_type"],
        # Calendar anchor: sorts periods, and joins filings to price data.
        "period_end": entry["period_end"],
        "calendar_quarter": entry["calendar_quarter"],
        "report_date": entry["report_date"],
        "filing_date": entry["filing_date"],
        "source_url": entry["source_url"],
    }

    # Prepend a compact context line to the embedded text. This meaningfully
    # helps the "right company, wrong quarter" and section-targeting cases.
    # The period here must be the company's own label: the passage text says
    # "quarter ended July 27, 2025", so a calendar label would contradict it.
    def embed_text(c):
        return f"[{entry['ticker']} {entry['filing_type']} {entry['fiscal_period']}] {c['section']}\n{c['text']}"

    embeddings = model.encode(
        [embed_text(c) for c in chunks],
        normalize_embeddings=True, show_progress_bar=False,
    ).tolist()

    ids = [f"{entry['file']}::{i}" for i in range(len(chunks))]
    documents = [c["text"] for c in chunks]
    metadatas = [
        {**base_meta, "section": c["section"], "chunk_kind": c["kind"], "chunk_index": i}
        for i, c in enumerate(chunks)
    ]

    collection.upsert(
        ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas
    )
    n_tbl = sum(1 for c in chunks if c["kind"] == "table")
    print(
        f"  ingested {entry['file']:<32} "
        f"{len(text):>9,} chars -> {len(chunks):>4} chunks ({n_tbl} table)"
    )
    return len(chunks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="ingest all filings, not just the 10-K")
    ap.add_argument("--reset", action="store_true", help="drop the collection first")
    ap.add_argument("--force", action="store_true",
                    help="re-embed filings that are already indexed (needed after "
                         "re-labelling periods or changing chunking)")
    args = ap.parse_args()

    manifest = load_manifest()
    targets = manifest if args.all else [e for e in manifest if e["filing_type"] == "10-K"]

    print(f"Loading embedding model '{config.EMBED_MODEL}' (first run downloads it)...")
    model = SentenceTransformer(config.EMBED_MODEL)

    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    collection = get_collection(client, args.reset)

    # Only embed what is not already indexed. Re-embedding the whole corpus to
    # add one new quarter would take hours at 50 companies; chunk ids are stable
    # (`<file>::<n>`), so the first chunk's presence is a reliable marker.
    if not (args.reset or args.force):
        before = len(targets)
        targets = [e for e in targets
                   if not collection.get(ids=[f"{e['file']}::0"])["ids"]]
        skipped = before - len(targets)
        if skipped:
            print(f"Skipping {skipped} filing(s) already indexed "
                  f"(--force to re-embed).")

    if not targets:
        print(f"Nothing to do. Collection '{config.COLLECTION}' has "
              f"{collection.count():,} chunks.")
        return

    print(f"Ingesting {len(targets)} filing(s):")
    total = sum(ingest_file(e, model, collection) for e in targets)
    print(f"Done. {total} chunks in collection '{config.COLLECTION}' "
          f"(total now {collection.count()}).")


if __name__ == "__main__":
    main()
