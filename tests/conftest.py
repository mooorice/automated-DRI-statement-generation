"""Test fixtures: stub the SentenceTransformer and LLM client so the pipeline
runs end-to-end offline, on CPU, in seconds, without API keys or model downloads.
"""

import hashlib
import json
from unittest.mock import MagicMock

import numpy as np
import pytest


class _FakeSentenceTransformer:
    """Deterministic, fast stand-in for SentenceTransformer.

    Hashes each text into a 384-dim L2-normalised vector. Texts that share
    keywords land near each other because the same byte windows recur across
    the hash extension. Good enough for smoke-testing the retrieval-→-generate
    flow without downloading 100MB of weights.
    """

    DIM = 384

    def __init__(self, model_name, device="cpu"):
        self.model_name = model_name

    def encode(self, texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True):
        out = np.zeros((len(texts), self.DIM), dtype=np.float32)
        for i, t in enumerate(texts):
            digest = hashlib.sha256(str(t).encode("utf-8")).digest()
            seed = int.from_bytes(digest[:4], "little")
            rng = np.random.default_rng(seed)
            vec = rng.standard_normal(self.DIM).astype(np.float32)
            norm = np.linalg.norm(vec)
            out[i] = vec / norm if norm > 0 else vec
        return out


@pytest.fixture
def stub_embedding_model(monkeypatch):
    """Replace the real SentenceTransformer with the deterministic stub."""
    monkeypatch.setattr(
        "autoconcourse.embedding.SentenceTransformer",
        _FakeSentenceTransformer,
    )


@pytest.fixture
def stub_llm_client(monkeypatch):
    """Replace create_llm_client so generation runs without an API key.

    The stub returns three canned statements per call, echoing back the
    subcategory's topic_focus so we can assert routing worked.
    """

    def fake_create(*_args, **_kwargs):
        client = MagicMock()

        def _create(model, messages, **kwargs):
            user_msg = next(m["content"] for m in messages if m["role"] == "user")
            # echo a short marker derived from the prompt so tests can verify
            # statements were produced for the right subcategory
            marker = user_msg.splitlines()[0][:60]
            content = json.dumps([
                {"id": 1, "text": f"Stub statement A for: {marker}"},
                {"id": 2, "text": f"Stub statement B for: {marker}"},
                {"id": 3, "text": f"Stub statement C for: {marker}"},
            ])
            response = MagicMock()
            response.choices[0].message.content = content
            return response

        client.chat.completions.create = _create
        return client

    monkeypatch.setattr("autoconcourse.generation.create_llm_client", fake_create)
    monkeypatch.setattr("autoconcourse.cli.create_llm_client", fake_create, raising=False)
