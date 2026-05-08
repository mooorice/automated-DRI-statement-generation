"""Anchor-guided cosine retrieval and alternative retrieval methods."""

import json
import logging

import numpy as np
import pandas as pd

from .config import PipelineConfig

logger = logging.getLogger("autoconcourse.retrieval")


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between each row of a and each row of b.

    Args:
        a: shape (n, d)
        b: shape (m, d)

    Returns:
        shape (n, m) similarity matrix
    """
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return a_norm @ b_norm.T


def cosine_similarity_vec(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between each row of a and a single vector b.

    Args:
        a: shape (n, d)
        b: shape (d,)

    Returns:
        shape (n,) similarity scores
    """
    a_norm = np.linalg.norm(a, axis=1)
    b_norm = np.linalg.norm(b)
    return (a @ b) / (a_norm * b_norm + 1e-10)


def aggregate_anchor_embeddings(
    anchor_embeddings: np.ndarray,
    anchor_categories: list[int],
    num_categories: int,
) -> np.ndarray:
    """Mean-pool anchor embeddings per category.

    Args:
        anchor_embeddings: shape (n_anchors, d)
        anchor_categories: list of category ids, one per anchor
        num_categories: total number of categories

    Returns:
        shape (num_categories, d) aggregated embeddings
    """
    d = anchor_embeddings.shape[1]
    aggregated = np.zeros((num_categories, d))

    for cat_id in range(num_categories):
        mask = [i for i, c in enumerate(anchor_categories) if c == cat_id]
        if mask:
            aggregated[cat_id] = anchor_embeddings[mask].mean(axis=0)

    return aggregated


def anchor_cosine_retrieval(
    paragraphs_df: pd.DataFrame,
    paragraph_embeddings: np.ndarray,
    anchor_embeddings: np.ndarray,
    anchor_categories: list[int],
    num_categories: int,
    top_k: int = 500,
) -> dict[int, pd.DataFrame]:
    """Retrieve top-k paragraphs per subcategory using anchor cosine similarity.

    Returns:
        dict mapping subcategory_id to a DataFrame of top-k paragraphs
        with a 'similarity_score' column added.
    """
    logger.info(
        f"Running anchor cosine retrieval: {len(paragraphs_df)} paragraphs, "
        f"{num_categories} categories, top_k={top_k}"
    )

    # Aggregate anchor embeddings by category
    agg_anchors = aggregate_anchor_embeddings(
        anchor_embeddings, anchor_categories, num_categories
    )

    results = {}
    for cat_id in range(num_categories):
        # Compute similarity between all paragraphs and this category's anchor
        scores = cosine_similarity_vec(paragraph_embeddings, agg_anchors[cat_id])

        # Get top-k indices
        k = min(top_k, len(scores))
        top_indices = np.argsort(scores)[-k:][::-1]

        # Build result DataFrame
        top_df = paragraphs_df.iloc[top_indices].copy()
        top_df["similarity_score"] = scores[top_indices]
        top_df["subcategory_id"] = cat_id

        results[cat_id] = top_df
        logger.info(
            f"  Category {cat_id}: top-{k} scores "
            f"[{scores[top_indices[-1]]:.4f} - {scores[top_indices[0]]:.4f}]"
        )

    return results


def compute_all_similarity_scores(
    paragraph_embeddings: np.ndarray,
    anchor_embeddings: np.ndarray,
    anchor_categories: list[int],
    num_categories: int,
) -> np.ndarray:
    """Compute similarity scores between all paragraphs and all categories.

    Returns:
        shape (n_paragraphs, num_categories) array of similarity scores
    """
    agg_anchors = aggregate_anchor_embeddings(
        anchor_embeddings, anchor_categories, num_categories
    )
    return cosine_similarity_matrix(paragraph_embeddings, agg_anchors)


def retrieve_for_pipeline(
    paragraphs_df: pd.DataFrame,
    paragraph_embeddings: np.ndarray,
    anchor_embeddings: np.ndarray,
    anchor_categories: list[int],
    config: PipelineConfig,
    subcategories: list[dict],
) -> dict[int, pd.DataFrame] | dict[tuple[str, int], pd.DataFrame]:
    """Full retrieval step using config settings.

    Optionally splits by perspective groups before retrieval.

    Returns:
        When perspective_groups disabled: dict mapping subcategory_id -> DataFrame
        When perspective_groups enabled: dict mapping (group_name, subcategory_id) -> DataFrame
    """
    num_categories = len(subcategories)
    top_k = config.retrieval.top_k

    if not config.perspective_groups.enabled:
        return anchor_cosine_retrieval(
            paragraphs_df, paragraph_embeddings,
            anchor_embeddings, anchor_categories,
            num_categories, top_k,
        )

    # Perspective-stratified retrieval
    groups_path = config.resolve_path(config.perspective_groups.groups_file)
    with open(groups_path, encoding="utf-8") as f:
        groups_data = json.load(f)

    meta_col = groups_data.get("metadata_column", config.perspective_groups.metadata_column)
    results = {}

    for group_name, group_info in groups_data["groups"].items():
        codes = group_info["medium_codes"]
        mask = paragraphs_df[meta_col].isin(codes)
        group_df = paragraphs_df[mask].copy()
        group_embeddings = paragraph_embeddings[mask.values]

        logger.info(f"Perspective group '{group_name}': {len(group_df)} paragraphs")

        group_results = anchor_cosine_retrieval(
            group_df, group_embeddings,
            anchor_embeddings, anchor_categories,
            num_categories, top_k,
        )

        # Store with group info
        for cat_id, cat_df in group_results.items():
            cat_df["perspective_group"] = group_name
            key = (group_name, cat_id)
            results[key] = cat_df

    return results
