# SPDX-License-Identifier: MPL-2.0
# loaders/sqlite_events.py
import sqlite3
from pathlib import Path
from typing import Iterable, Tuple

Doc = Tuple[str, str]          # (doc_id, full_text)

def _to_text(rows, cols) -> str:
    header = " | ".join(cols)
    body   = "\n".join(" | ".join(map(str, row)) for row in rows)
    return f"{header}\n{body}"

def load(db_path: Path) -> Iterable[Doc]:
    # ⬇️ skip if the path is not an existing *.db file
    if not (db_path.is_file() and db_path.suffix.lower() == ".db"):
        return []

    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()
    try:
        # ---- Events -------------------------------------------------
        cur.execute("SELECT name FROM sqlite_master "
                    "WHERE type='table' AND name='Events'")
        if cur.fetchone():
            cur.execute("SELECT * FROM Events")
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description]
            yield f"{db_path}::Events", _to_text(rows, cols)

        # ---- SystemStatus -------------------------------------------
        cur.execute("SELECT name FROM sqlite_master "
                    "WHERE type='table' AND name='SystemStatus'")
        if cur.fetchone():
            cur.execute("SELECT * FROM SystemStatus")
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description]
            yield f"{db_path}::SystemStatus", _to_text(rows, cols)
    finally:
        conn.close()
