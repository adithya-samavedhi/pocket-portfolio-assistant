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

# --- EDGAR requires a User-Agent identifying the requester (name + email). ---
# Set SEC_USER_AGENT in your .env — SEC may throttle or block the placeholder.
# Deliberately not a real address in source: this repo is public.
USER_AGENT = os.environ.get(
    "SEC_USER_AGENT", "pocket-portfolio-assistant (set SEC_USER_AGENT)"
)
