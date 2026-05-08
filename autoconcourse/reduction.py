"""Dimensionality reduction (UMAP, PCA, or none)."""

import logging

import numpy as np

from .config import PipelineConfig

logger = logging.getLogger("autoconcourse.reduction")


def reduce_embeddings(
    embeddings_main: np.ndarray,
    embeddings_anchor: np.ndarray,
    config: PipelineConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce dimensionality of embeddings.

    Fits the reducer on the combined set (main + anchor), then splits back.
    Returns (reduced_main, reduced_anchor).
    """
    method = config.reduction.method

    if method == "none":
        logger.info("Skipping dimensionality reduction (method=none)")
        return embeddings_main, embeddings_anchor

    # Combine for joint reduction
    n_main = len(embeddings_main)
    combined = np.vstack([embeddings_main, embeddings_anchor])
    n_components = min(config.reduction.n_components, combined.shape[0] - 1)

    logger.info(
        f"Reducing {combined.shape} -> ({combined.shape[0]}, {n_components}) using {method}"
    )

    if method == "umap":
        reduced = _umap_reduce(combined, n_components, config)
    elif method == "pca":
        reduced = _pca_reduce(combined, n_components)
    else:
        raise ValueError(f"Unsupported reduction method: {method}")

    reduced_main = reduced[:n_main]
    reduced_anchor = reduced[n_main:]

    logger.info(
        f"Reduction complete: main={reduced_main.shape}, anchor={reduced_anchor.shape}"
    )
    return reduced_main, reduced_anchor


def _umap_reduce(embeddings: np.ndarray, n_components: int, config: PipelineConfig) -> np.ndarray:
    """Apply UMAP dimensionality reduction."""
    import umap

    params = config.reduction.umap_params
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=params.n_neighbors,
        min_dist=params.min_dist,
        metric=params.metric,
        random_state=42,
    )
    return reducer.fit_transform(embeddings)


def _pca_reduce(embeddings: np.ndarray, n_components: int) -> np.ndarray:
    """Apply PCA dimensionality reduction."""
    from sklearn.decomposition import PCA

    reducer = PCA(n_components=n_components, random_state=42)
    return reducer.fit_transform(embeddings)
