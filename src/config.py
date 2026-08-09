"""Shared configuration for the filings RAG pipeline.

Milestone 1 keeps everything local: filings on disk, a local embedding model,
and an on-disk Chroma store. No API keys are needed until Milestone 6.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# --- paths ---
ROOT = Path(__file__).resolve().parent.parent

# Load API keys / settings from .env at the repo root (does nothing if absent).
load_dotenv(ROOT / ".env")

DATA = ROOT / "data"
RAW = DATA / "raw"          # raw filing HTML, saved so we never re-hit EDGAR
CACHE = DATA / "cache"      # cached API responses
CHROMA_DIR = DATA / "chroma"  # on-disk vector store

# --- embedding model (local, CPU) ---
# bge-small is a strong, tiny default: ~130MB, ~2GB RAM, good enough for a skeleton.
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

# --- chunking (fixed-size with overlap; snapped to word boundaries) ---
CHUNK_SIZE = 1000          # characters
CHUNK_OVERLAP = 150        # characters
MIN_CHUNK_CHARS = 40       # drop fragments shorter than this

# --- Chroma collection ---
COLLECTION = "filings"

# --- the covered universe, and how people refer to it ---
# Shared so the filings and market sides resolve a company the same way.
TICKERS = {"GOOGL", "NVDA", "MSFT", "AAPL", "AMZN"}
TICKER_ALIASES = {
    "GOOGLE": "GOOGL", "ALPHABET": "GOOGL", "GOOG": "GOOGL",
    "NVIDIA": "NVDA", "MICROSOFT": "MSFT",
    "APPLE": "AAPL", "AMAZON": "AMZN",
}


def mentioned_tickers(text: str):
    """Every covered company named in the text, in order of first mention.

    Comparative questions ("compare Microsoft and Amazon…") need this: global
    top-k covered both named companies only 75% of the time, so the model was
    asked to compare two companies while seeing passages from one.
    """
    import re
    out = []
    for word in re.findall(r"[A-Za-z]{2,}", text):
        t = TICKER_ALIASES.get(word.upper(), word.upper() if word.upper() in TICKERS else None)
        if t and t not in out:
            out.append(t)
    return out


def resolve_ticker(text: str):
    """The single covered company a question is about, or None."""
    found = mentioned_tickers(text)
    return found[0] if len(found) >= 1 else None


# --- EDGAR requires a User-Agent identifying the requester (name + email). ---
# Set SEC_USER_AGENT in your .env — SEC may throttle or block the placeholder.
# Deliberately not a real address in source: this repo is public.
USER_AGENT = os.environ.get(
    "SEC_USER_AGENT", "pocket-portfolio-assistant (set SEC_USER_AGENT)"
)
