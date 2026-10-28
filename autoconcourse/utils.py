"""Shared utilities for embedding I/O, JSON schemas, and common operations."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("autoconcourse")


# Column order for the CSV deliverable. Anything not in this list is appended
# to the right, in insertion order. Keeps Excel views consistent across runs.
_CSV_PRIMARY_COLS = [
    "id",
    "statement_type",
    "subcategory_id",
    "subcategory",
    "group",
    "text",
    "source_paragraphs",
]


def setup_logging(level: str = "INFO") -> None:
    """Configure logging for the autoconcourse package."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def save_embeddings(embeddings: np.ndarray, path: str | Path) -> None:
    """Save embeddings as a .npy file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, embeddings)
    logger.info(f"Saved embeddings {embeddings.shape} to {path}")


def load_embeddings(path: str | Path) -> np.ndarray:
    """Load embeddings from a .npy file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {path}")
    embeddings = np.load(path)
    logger.info(f"Loaded embeddings {embeddings.shape} from {path}")
    return embeddings


def save_concourse(
    statements: list[dict],
    output_path: str | Path,
    system: str,
    topic: str,
    model: str,
    config_hash: str,
    write_csv: bool = True,
) -> None:
    """Save a concourse as JSON plus an Excel-friendly CSV next to it.

    The CSV uses the same stem as `output_path` (e.g. `concourse.csv` next to
    `concourse.json`), is written as UTF-8 with BOM so Excel on Windows opens
    it correctly, and flattens `source_paragraphs` to a `;`-separated string
    so a single cell doesn't get split into columns.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    concourse = {
        "system": system,
        "topic": topic,
        "model": model,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config_hash": config_hash,
        "statements": statements,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(concourse, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(statements)} statements to {output_path}")

    if write_csv:
        csv_path = output_path.with_suffix(".csv")
        save_concourse_csv(statements, csv_path)


def save_concourse_csv(statements: list[dict], output_path: str | Path) -> None:
    """Write the statements list as an Excel-friendly CSV.

    - UTF-8 with BOM (so Excel auto-detects encoding on Windows)
    - List-valued fields (e.g. `source_paragraphs`) are joined with ';' so
      they fit into a single cell
    - Columns ordered per `_CSV_PRIMARY_COLS`, with any extras appended
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not statements:
        # Still write an empty file with the canonical header so downstream
        # tooling (Excel templates, scripts) doesn't choke on missing files.
        pd.DataFrame(columns=_CSV_PRIMARY_COLS).to_csv(
            output_path, index=False, encoding="utf-8-sig"
        )
        logger.info(f"Saved 0 statements (empty CSV) to {output_path}")
        return

    df = pd.DataFrame(statements)

    for col in df.columns:
        if df[col].apply(lambda v: isinstance(v, (list, tuple))).any():
            df[col] = df[col].apply(
                lambda v: ";".join(str(x) for x in v) if isinstance(v, (list, tuple)) else v
            )

    ordered = [c for c in _CSV_PRIMARY_COLS if c in df.columns]
    extras = [c for c in df.columns if c not in ordered]
    df = df[ordered + extras]

    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info(f"Saved {len(statements)} statements to {output_path}")


def load_concourse(path: str | Path) -> dict:
    """Load a concourse JSON file."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_json(path: str | Path) -> dict | list:
    """Load a JSON file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict | list, path: str | Path) -> None:
    """Save data as a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
