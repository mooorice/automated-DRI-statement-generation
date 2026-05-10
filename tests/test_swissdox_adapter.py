"""Swissdox adapter regression: tiny synthetic Swissdox-shaped TSV → canonical shape."""

from pathlib import Path

import pandas as pd

from autoconcourse.adapters import load_with_adapter


def _write_synthetic_swissdox(path: Path) -> None:
    """Two documents, each with HTML <p>-paragraphed content."""
    rows = [
        {
            "content_id": "doc_001",
            "language": "de",
            "medium_code": "NZZ",
            "content": (
                "<p>Erster Absatz über Klimapolitik in der Schweiz.</p>"
                "<p>Zweiter Absatz mit weiteren Details zu Massnahmen.</p>"
                "<p>x</p>"  # too short, should be filtered
            ),
        },
        {
            "content_id": "doc_002",
            "language": "fr",
            "medium_code": "LE_TEMPS",
            "content": (
                "<p>Premier paragraphe sur la politique climatique en Suisse.</p>"
                "<p>Deuxième paragraphe avec plus de détails sur les mesures.</p>"
            ),
        },
        # duplicate content_id; should be deduped
        {
            "content_id": "doc_001",
            "language": "de",
            "medium_code": "NZZ",
            "content": "<p>duplicate document body that should be dropped</p>",
        },
    ]
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


def test_swissdox_adapter_canonical_shape(tmp_path):
    src = tmp_path / "swissdox.tsv"
    _write_synthetic_swissdox(src)

    df = load_with_adapter("swissdox", src)

    # Canonical columns are present; Swissdox-specific names are renamed
    assert "text" in df.columns
    assert "id" in df.columns
    assert "language" in df.columns
    assert "group" in df.columns       # was medium_code
    assert "source_id" in df.columns   # was content_id

    # 2 docs (after dedup) × 2 valid paragraphs each = 4 rows
    assert len(df) == 4

    # Languages preserved per paragraph
    de_rows = df[df["language"] == "de"]
    fr_rows = df[df["language"] == "fr"]
    assert len(de_rows) == 2
    assert len(fr_rows) == 2

    # Group (medium_code) preserved
    assert set(df["group"].unique()) == {"NZZ", "LE_TEMPS"}

    # Source ids preserved (one per document, repeated across that doc's paragraphs)
    assert set(df["source_id"].unique()) == {"doc_001", "doc_002"}

    # Short paragraph (single 'x') was filtered by min_length default
    assert all(len(t) >= 20 for t in df["text"])


def test_swissdox_adapter_xz(tmp_path):
    """End-to-end: write a .tsv.xz and load it through the adapter."""
    import lzma

    src = tmp_path / "swissdox.tsv.xz"
    rows = [{
        "content_id": "doc_a",
        "language": "en",
        "medium_code": "BBC",
        "content": "<p>A long enough paragraph about climate policy in some country.</p>",
    }]
    raw = pd.DataFrame(rows).to_csv(sep="\t", index=False).encode("utf-8")
    with lzma.open(src, "wb") as f:
        f.write(raw)

    df = load_with_adapter("swissdox", src)
    assert len(df) == 1
    assert df["language"].iloc[0] == "en"
    assert "climate policy" in df["text"].iloc[0]
