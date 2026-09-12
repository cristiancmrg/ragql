# SPDX-License-Identifier: MPL-2.0
# loaders/text_files.py
from pathlib import Path
from typing import Iterable, Tuple

Doc = Tuple[str, str]  # (doc_id, full_text)

# only these extensions (add more if you like)
EXTS = {".txt", ".log", ".md", ".rst", ".out", ".json"}


def _looks_binary(p: Path, sample: int = 256) -> bool:
    try:
        chunk = p.read_bytes()[:sample]
    except Exception:
        return True
    non_printable = sum(b < 9 or (13 < b < 32) or b > 126 for b in chunk)
    return non_printable / max(len(chunk), 1) > 0.2


def load(path: Path) -> Iterable[Doc]:
    """
    • If you give me a file, I index it.
    • If you give me a directory, I recurse *all* sub‐dirs for *every* file.
    """
    # one‐liner: file → singleton; dir → deep walk
    candidates = [path] if path.is_file() else path.rglob("*")

    for p in candidates:
        if not p.is_file():
            continue
        if p.suffix.lower() not in EXTS:
            continue
        if _looks_binary(p):
            continue
        try:
            text = p.read_text(errors="ignore")
        except Exception:
            text = p.read_bytes().decode("latin-1", errors="ignore")
        yield str(p), text
