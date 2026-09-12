# SPDX-License-Identifier: MPL-2.0
# test/test_ragql_flow.py
from pathlib import Path
import numpy as np
import ragql.core as core

def fake_embed(texts, _cfg=None):
    return np.stack([np.ones(3, "float32") for _ in texts])

def fake_chat(prompt, context, _cfg):
    # very dumb offline answer that satisfies the assertion
    return "The wizard casts the spell."

def test_end_to_end(tmp_path, cfg, monkeypatch):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "a.log").write_text("dragon breathes fire")
    (logs / "b.log").write_text("wizard casts spell")

    monkeypatch.setattr("ragql.embeddings.get_embeddings", fake_embed)
    monkeypatch.setattr("ragql.core.RagQL._call_llm", lambda self, prompt, ctx: "The wizard casts the spell.")

    rq = core.RagQL(logs, cfg)
    rq.build()
    answer = rq.query("Who casts a spell?")
    assert "wizard" in answer.lower()
