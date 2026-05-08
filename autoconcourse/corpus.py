"""Corpus preparation: adapter dispatch + canonical-shape filters.

The pipeline operates on a single canonical DataFrame shape (defined in
autoconcourse/adapters/__init__.py). Format-specific work (Swissdox HTML,
content_id dedup, TSV.xz reading, etc.) lives in adapter modules.
"""

import json
import logging
import re
import unicodedata
from pathlib import Path

import pandas as pd

from .adapters import load_with_adapter
from .config import PipelineConfig

logger = logging.getLogger("autoconcourse.corpus")


def load_corpus(config: PipelineConfig) -> pd.DataFrame:
    """Dispatch to the configured adapter and return canonical-shape DataFrame."""
    adapter = config.corpus.adapter
    path = config.resolve_path(config.corpus.path)

    opts = {
        "text_column": config.corpus.text_column,
        "id_column": config.corpus.id_column,
        "language_column": config.corpus.language_column,
        "group_column": config.corpus.group_column,
        "source_id_column": config.corpus.source_id_column,
    }
    opts.update(config.corpus.options or {})

    return load_with_adapter(adapter, path, **opts)


def _normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ASCII", "ignore").decode("utf-8")
    return text.lower()


def load_keywords(keywords_file: str | Path) -> list[str]:
    """Load keywords from JSON. Accepts a flat list or a language-grouped dict."""
    with open(keywords_file, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        keywords = []
        for lang_keywords in data.values():
            keywords.extend(lang_keywords)
        return keywords
    if isinstance(data, list):
        return data
    raise ValueError(f"Unexpected keywords format in {keywords_file}")


def filter_by_keywords(df: pd.DataFrame, keywords_file: str | Path) -> pd.DataFrame:
    keywords = load_keywords(keywords_file)
    normalized = [_normalize_text(kw) for kw in keywords]
    pattern = "|".join(re.escape(kw) for kw in normalized)

    logger.info(f"Filtering by {len(keywords)} keywords")
    text_normalized = df["text"].apply(_normalize_text)
    mask = text_normalized.str.contains(pattern, case=False, na=False)
    result = df[mask].copy()
    result["id"] = range(1, len(result) + 1)
    logger.info(f"Keyword filter: {len(df)} → {len(result)} rows")
    return result


def filter_by_min_length(df: pd.DataFrame, min_length: int) -> pd.DataFrame:
    if min_length <= 0:
        return df
    mask = df["text"].astype(str).str.len() >= min_length
    result = df[mask].copy()
    result["id"] = range(1, len(result) + 1)
    logger.info(f"Min-length filter ({min_length}): {len(df)} → {len(result)} rows")
    return result


def filter_by_language(df: pd.DataFrame, languages: list[str]) -> pd.DataFrame:
    if "language" not in df.columns:
        logger.warning("language filter requested but no 'language' column; skipping")
        return df
    mask = df["language"].isin(languages)
    result = df[mask].copy()
    result["id"] = range(1, len(result) + 1)
    logger.info(f"Language filter ({languages}): {len(df)} → {len(result)} rows")
    return result


def prepare_corpus(config: PipelineConfig) -> pd.DataFrame:
    """Full corpus prep: adapter → canonical-shape filters."""
    df = load_corpus(config)

    filt = config.corpus.filtering
    if filt.min_length and filt.min_length > 0:
        df = filter_by_min_length(df, filt.min_length)

    if filt.languages:
        df = filter_by_language(df, filt.languages)

    if filt.require_keyword and filt.keywords_file:
        keywords_path = config.resolve_path(filt.keywords_file)
        df = filter_by_keywords(df, keywords_path)

    return df
