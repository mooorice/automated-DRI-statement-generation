"""Swissdox adapter: TSV(.xz) with HTML article content, deduped by content_id, chunked into paragraphs."""

import logging
import lzma
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

logger = logging.getLogger("autoconcourse.adapters.swissdox")


def load(
    path: Path,
    min_length: int = 20,
    text_column: str = "content",
    language_column: str | None = "language",
    group_column: str | None = "medium_code",
    source_id_column: str | None = "content_id",
    **kwargs,
) -> pd.DataFrame:
    """Load a Swissdox dump and return canonical-shape paragraphs.

    Steps:
      1. Read TSV (auto-detects .xz compression by suffix).
      2. Drop duplicate documents by content_id.
      3. Split each document's HTML into <p>-tag paragraphs (or fall back to
         double-newline split).
      4. Map metadata columns (language, medium_code → group, content_id →
         source_id) onto each paragraph.
    """
    logger.info(f"Loading Swissdox dump from {path}")

    if path.suffix == ".xz":
        with lzma.open(path) as f:
            df = pd.read_csv(f, delimiter="\t")
    else:
        df = pd.read_csv(path, delimiter="\t")

    if source_id_column and source_id_column in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=[source_id_column], keep="first")
        logger.info(f"Deduped by {source_id_column}: {before} → {len(df)} documents")

    if text_column not in df.columns:
        raise ValueError(
            f"Swissdox adapter: text column '{text_column}' not found. "
            f"Columns available: {list(df.columns)}"
        )

    metadata_cols = [c for c in (language_column, group_column, source_id_column) if c and c in df.columns]
    rows = []
    for _, doc in df.iterrows():
        text = doc[text_column]
        if pd.isna(text):
            continue

        soup = BeautifulSoup(str(text), "html.parser")
        paragraphs = soup.find_all("p")
        if paragraphs:
            chunks = [p.get_text(strip=True) for p in paragraphs]
        else:
            chunks = [t.strip() for t in str(text).split("\n\n")]

        for chunk in chunks:
            if len(chunk) < min_length:
                continue
            row = {"text": chunk}
            for col in metadata_cols:
                row[col] = doc[col]
            rows.append(row)

    canonical = pd.DataFrame(rows)
    canonical["id"] = range(1, len(canonical) + 1)

    rename = {}
    if language_column and language_column != "language" and language_column in canonical.columns:
        rename[language_column] = "language"
    if group_column and group_column != "group" and group_column in canonical.columns:
        rename[group_column] = "group"
    if source_id_column and source_id_column != "source_id" and source_id_column in canonical.columns:
        rename[source_id_column] = "source_id"
    if rename:
        canonical = canonical.rename(columns=rename)

    logger.info(f"Swissdox adapter: {len(canonical)} paragraphs, columns={list(canonical.columns)}")
    return canonical
