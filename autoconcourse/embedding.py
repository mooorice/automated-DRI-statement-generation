"""Sentence embedding using SentenceTransformers with numpy-based storage."""

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

    model = EmbeddingModel(config.embedding.model, config.embedding.device)
    embeddings = model.embed(texts, batch_size=config.embedding.batch_size)

    if output_path:
        save_embeddings(embeddings, output_path)

    return embeddings
