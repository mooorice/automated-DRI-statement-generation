"""Text directory adapter: each .txt file is one document, double-newline split into paragraphs."""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger("autoconcourse.adapters.text_dir")


def load(
    path: Path,
    min_length: int = 20,
    glob: str = "*.txt",
    language: str | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Load every file matching `glob` under `path`, split into paragraphs."""
    logger.info(f"Loading text directory: {path} (glob={glob})")
    if not path.is_dir():
        raise NotADirectoryError(f"text_dir adapter expects a directory, got: {path}")

    rows = []
    for file_path in sorted(path.glob(glob)):
        text = file_path.read_text(encoding="utf-8", errors="replace")
        chunks = [t.strip() for t in text.split("\n\n")]
        for chunk in chunks:
            if len(chunk) < min_length:
                continue
            row = {"text": chunk, "source_id": file_path.stem}
            if language:
                row["language"] = language
            rows.append(row)

    df = pd.DataFrame(rows)
    df["id"] = range(1, len(df) + 1)
    logger.info(f"text_dir adapter: {len(df)} paragraphs from {len(list(path.glob(glob)))} files")
    return df
