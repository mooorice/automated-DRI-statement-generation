"""Shared utilities for embedding I/O, JSON schemas, and common operations."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

logger = logging.getLogger("autoconcourse")


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
) -> None:
    """Save a concourse in the standard JSON format."""
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
