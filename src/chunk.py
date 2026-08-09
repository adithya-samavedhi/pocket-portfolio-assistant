"""Fixed-size chunking with overlap, snapped to word boundaries.

Naive character slicing cuts words in half ("enou|gh"), which hurts both
embedding quality and the readability of cited passages. Here the loop still
advances by a fixed step (guaranteeing progress and roughly constant overlap),
but each chunk's start and end are nudged to the nearest whitespace so chunks
begin and end on whole words.
"""
from typing import List

from config import CHUNK_SIZE, CHUNK_OVERLAP

_SNAP = 60  # how far to look for a whitespace boundary when snapping


def fixed_size_chunks(
    text: str,
    size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """Split text into ~`size`-char windows with ~`overlap`, on word boundaries."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")

    text = text.strip()
    n = len(text)
    if n <= size:
        return [text] if text else []

    chunks: List[str] = []
    step = size - overlap
    for base in range(0, n, step):
        start, end = base, min(base + size, n)

        # Snap start forward to a word boundary if we'd begin mid-word.
        if start > 0 and not text[start - 1].isspace() and not text[start].isspace():
            sp = text.find(" ", start, start + _SNAP)
            if sp != -1:
                start = sp + 1
        # Snap end back to the last whitespace in the window (skip if it's the end).
        if end < n:
            sp = text.rfind(" ", base + step, end)
            if sp > start:
                end = sp

        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
    return chunks
