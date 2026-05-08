"""Data loading, chunking, and keyword filtering."""

import json
import logging
import re
import unicodedata
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

from .config import PipelineConfig

logger = logging.getLogger("autoconcourse.corpus")


def load_corpus(config: PipelineConfig) -> pd.DataFrame:
    """Load corpus from the configured source into a DataFrame.

    Returns a DataFrame with at minimum a 'text' column and an 'id' column.
    """
    source = config.corpus.source
    path = config.resolve_path(config.corpus.path)

    logger.info(f"Loading corpus from {path} (source={source})")

    if source == "csv":
        df = pd.read_csv(path)
    elif source == "tsv":
        df = pd.read_csv(path, delimiter="\t")
    elif source == "tsv.xz":
        import lzma
        with lzma.open(path) as f:
            df = pd.read_csv(f, delimiter="\t")
    elif source == "jsonl":
        df = pd.read_json(path, lines=True)
    else:
        raise ValueError(f"Unsupported corpus source: {source}")

    # Ensure text column exists
    text_col = config.corpus.text_column
    if text_col not in df.columns:
        raise ValueError(f"Text column '{text_col}' not found in corpus. Available: {list(df.columns)}")

    # Rename to standard 'text' if needed
    if text_col != "text":
        df = df.rename(columns={text_col: "text"})

    # Ensure id column
    if "id" not in df.columns:
        df["id"] = range(1, len(df) + 1)

    # Drop duplicates if content_id exists (swissdox format)
    if "content_id" in df.columns:
        df = df.drop_duplicates(subset=["content_id"], keep="first")

    logger.info(f"Loaded corpus: {len(df)} rows")
    return df


def chunk_paragraphs(df: pd.DataFrame, method: str = "paragraph", min_length: int = 20) -> pd.DataFrame:
    """Split text into chunks (paragraphs, sentences, or fixed-length).

    For HTML content, extracts <p> tags. For plain text, splits on double newlines.
    Returns a new DataFrame with one row per chunk, preserving metadata columns.
    """
    logger.info(f"Chunking with method={method}, min_length={min_length}")

    if method == "paragraph":
        return _chunk_html_paragraphs(df, min_length)
    elif method == "sentence":
        return _chunk_sentences(df, min_length)
    elif method == "none":
        return df
    else:
        raise ValueError(f"Unsupported chunking method: {method}")


def _chunk_html_paragraphs(df: pd.DataFrame, min_length: int) -> pd.DataFrame:
    """Extract paragraphs from HTML content using BeautifulSoup."""
    text_col = "content" if "content" in df.columns else "text"
    metadata_cols = [c for c in df.columns if c not in (text_col, "text")]

    rows = []
    for _, row in df.iterrows():
        text = row[text_col]
        if pd.isna(text):
            continue

        # Try HTML parsing first
        soup = BeautifulSoup(str(text), "html.parser")
        paragraphs = soup.find_all("p")

        if paragraphs:
            texts = [p.get_text(strip=True) for p in paragraphs]
        else:
            # Fall back to double-newline splitting
            texts = [t.strip() for t in str(text).split("\n\n")]

        for para_text in texts:
            if len(para_text) < min_length:
                continue
            new_row = {col: row[col] for col in metadata_cols if col in row.index}
            new_row["text"] = para_text
            rows.append(new_row)

    result = pd.DataFrame(rows)
    result["id"] = range(1, len(result) + 1)
    logger.info(f"Chunked into {len(result)} paragraphs")
    return result


def _chunk_sentences(df: pd.DataFrame, min_length: int) -> pd.DataFrame:
    """Split text into sentences."""
    metadata_cols = [c for c in df.columns if c != "text"]

    rows = []
    for _, row in df.iterrows():
        text = row["text"]
        if pd.isna(text):
            continue
        # Simple sentence splitting
        sentences = re.split(r"(?<=[.!?])\s+", str(text))
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < min_length:
                continue
            new_row = {col: row[col] for col in metadata_cols if col in row.index}
            new_row["text"] = sent
            rows.append(new_row)

    result = pd.DataFrame(rows)
    result["id"] = range(1, len(result) + 1)
    logger.info(f"Chunked into {len(result)} sentences")
    return result


def _normalize_text(text: str) -> str:
    """Normalize text by removing accents and lowercasing."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ASCII", "ignore").decode("utf-8")
    return text.lower()


def load_keywords(keywords_file: str | Path) -> list[str]:
    """Load keywords from a JSON file. Flattens language-grouped keywords into a single list."""
    with open(keywords_file, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        # Language-grouped format: {"de": [...], "fr": [...], ...}
        keywords = []
        for lang_keywords in data.values():
            keywords.extend(lang_keywords)
        return keywords
    elif isinstance(data, list):
        return data
    else:
        raise ValueError(f"Unexpected keywords format in {keywords_file}")


def filter_by_keywords(df: pd.DataFrame, keywords_file: str | Path) -> pd.DataFrame:
    """Filter DataFrame to rows where 'text' contains at least one keyword."""
    keywords = load_keywords(keywords_file)
    normalized_keywords = [_normalize_text(kw) for kw in keywords]
    pattern = "|".join(re.escape(kw) for kw in normalized_keywords)

    logger.info(f"Filtering with {len(keywords)} keywords")

    df = df.copy()
    text_normalized = df["text"].apply(_normalize_text)
    mask = text_normalized.str.contains(pattern, case=False, na=False)
    result = df[mask].copy()

    # Re-index
    result["id"] = range(1, len(result) + 1)

    logger.info(f"Filtered: {len(df)} -> {len(result)} rows")
    return result


def prepare_corpus(config: PipelineConfig) -> pd.DataFrame:
    """Full corpus preparation pipeline: load, chunk, filter."""
    df = load_corpus(config)

    # Chunk if configured
    chunking = config.corpus.chunking
    if chunking.method != "none":
        # Only chunk if we have raw content (not pre-chunked)
        if "content" in df.columns:
            df = chunk_paragraphs(df, method=chunking.method, min_length=chunking.min_length)

    # Filter by keywords if configured
    if config.corpus.filtering.require_keyword and config.corpus.filtering.keywords_file:
        keywords_path = config.resolve_path(config.corpus.filtering.keywords_file)
        df = filter_by_keywords(df, keywords_path)

    return df
