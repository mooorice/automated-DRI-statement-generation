"""Sentence embedding using SentenceTransformers with numpy-based storage."""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from .config import PipelineConfig
from .utils import load_embeddings, save_embeddings

logger = logging.getLogger("autoconcourse.embedding")


class EmbeddingModel:
    """Wrapper around SentenceTransformer for consistent embedding generation."""

    def __init__(self, model_name: str, device: str = "cpu"):
        logger.info(f"Loading embedding model: {model_name} (device={device})")
        self.model = SentenceTransformer(model_name, device=device)
        self.model_name = model_name
        self.device = device

    def embed(self, texts: list[str], batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
        """Embed a list of texts, returning a numpy array of shape (n_texts, dim)."""
        logger.info(f"Embedding {len(texts)} texts (batch_size={batch_size})")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        logger.info(f"Embeddings shape: {embeddings.shape}")
        return embeddings

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed one batch of texts."""
        return self.model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
        )


def embed_texts(
    texts: list[str],
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
    batch_size: int = 32,
    device: str = "cpu",
) -> np.ndarray:
    """Embed a list of texts using the specified model."""
    model = EmbeddingModel(model_name, device)
    return model.embed(texts, batch_size=batch_size)


def embed_dataframe(
    df: pd.DataFrame,
    config: PipelineConfig,
    output_path: str | Path | None = None,
) -> np.ndarray:
    """Embed the 'text' column of a DataFrame.

    If output_path is given and exists, loads cached embeddings.
    Returns the embedding array (same row order as df).
    """
    if output_path:
        output_path = Path(output_path)
        if output_path.exists():
            logger.info(f"Loading cached embeddings from {output_path}")
            return load_embeddings(output_path)

    # Clean text column
    texts = df["text"].astype(str).fillna("").tolist()
    total = len(texts)
    if total == 0:
        return np.empty((0, 0), dtype=np.float32)

    model = EmbeddingModel(config.embedding.model, config.embedding.device)
    batch_size = config.embedding.batch_size

    # Optional incremental checkpointing: write a .npy progressively, one batch at a time.
    if output_path:
        output_path = Path(output_path)
        tmp_path = output_path.parent / f"{output_path.stem}.partial.npy"
        progress_path = output_path.parent / f"{output_path.stem}.progress.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        completed = 0
        mmap = None

        if tmp_path.exists() and progress_path.exists():
            try:
                progress = json.loads(progress_path.read_text(encoding="utf-8"))
                if progress.get("total") == total:
                    completed = int(progress.get("completed", 0))
                    mmap = np.lib.format.open_memmap(tmp_path, mode="r+")
                    logger.info(f"Resuming embeddings from row {completed}/{total}")
                else:
                    logger.warning("Partial embedding checkpoint shape mismatch; restarting embedding run")
            except Exception as exc:
                logger.warning(f"Failed to load embedding checkpoint metadata ({exc}); restarting run")

        if mmap is None:
            first_end = min(batch_size, total)
            first_batch = model.embed_batch(texts[:first_end])
            mmap = np.lib.format.open_memmap(
                tmp_path,
                mode="w+",
                dtype=first_batch.dtype,
                shape=(total, first_batch.shape[1]),
            )
            mmap[:first_end] = first_batch
            mmap.flush()
            completed = first_end
            progress_path.write_text(
                json.dumps({"total": total, "completed": completed}, ensure_ascii=True),
                encoding="utf-8",
            )

        while completed < total:
            end = min(completed + batch_size, total)
            batch = model.embed_batch(texts[completed:end])
            mmap[completed:end] = batch
            completed = end
            mmap.flush()
            progress_path.write_text(
                json.dumps({"total": total, "completed": completed}, ensure_ascii=True),
                encoding="utf-8",
            )

            if completed % max(batch_size * 10, 1) == 0 or completed == total:
                logger.info(f"Embedded {completed}/{total} rows")

        # Finalize checkpoint files into canonical output artifact.
        del mmap
        tmp_path.replace(output_path)
        progress_path.unlink(missing_ok=True)
        return load_embeddings(output_path)

    embeddings = model.embed(texts, batch_size=batch_size)

    if output_path:
        save_embeddings(embeddings, output_path)

    return embeddings
