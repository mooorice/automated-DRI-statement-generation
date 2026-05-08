"""CSV adapter: maps columns into the canonical shape."""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger("autoconcourse.adapters.csv")


def load(
    path: Path,
    text_column: str = "text",
    id_column: str | None = None,
    language_column: str | None = None,
    group_column: str | None = None,
    source_id_column: str | None = None,
    delimiter: str = ",",
    **kwargs,
) -> pd.DataFrame:
    """Load a CSV (or TSV via delimiter='\\t') and rename columns to canonical names.

    Any column not mapped is preserved as-is for downstream use.
    """
    logger.info(f"Loading CSV from {path}")
    df = pd.read_csv(path, delimiter=delimiter)

    rename = {}
    if text_column != "text":
        if text_column not in df.columns:
            raise ValueError(f"text_column '{text_column}' not in CSV. Columns: {list(df.columns)}")
        rename[text_column] = "text"
    if id_column and id_column != "id":
        rename[id_column] = "id"
    if language_column and language_column != "language":
        rename[language_column] = "language"
    if group_column and group_column != "group":
        rename[group_column] = "group"
    if source_id_column and source_id_column != "source_id":
        rename[source_id_column] = "source_id"

    if rename:
        df = df.rename(columns=rename)

    if "text" not in df.columns:
        raise ValueError(f"After mapping, no 'text' column. Columns: {list(df.columns)}")

    if "id" not in df.columns:
        df["id"] = range(1, len(df) + 1)

    logger.info(f"CSV adapter: {len(df)} rows, columns={list(df.columns)}")
    return df
