"""Anchor-guided cosine retrieval.

Multilingual: a subcategory may have several anchors (typically one per language).
For each passage, the score for a subcategory is `max` (or `mean`, configurable)
over the cosine similarities to that subcategory's anchors. Monolingual is just
the length-1 case.
"""

import json
import logging

import numpy as np
import pandas as pd

from .config import PipelineConfig

logger = logging.getLogger("autoconcourse.retrieval")


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return a_norm @ b_norm.T


def cosine_similarity_vec(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = np.linalg.norm(a, axis=1)
    b_norm = np.linalg.norm(b)
    return (a @ b) / (a_norm * b_norm + 1e-10)


def aggregate_anchor_embeddings(
    anchor_embeddings: np.ndarray,
    anchor_categories: list[int],
    num_categories: int,
) -> np.ndarray:
    """Mean-pool anchor embeddings per category. Kept for paper baselines."""
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
    aggregation: str = "max",
) -> dict[int, pd.DataFrame]:
    """Retrieve top-k paragraphs per subcategory.

    For each subcategory, score = aggregation(cosine(passage, anchor)) over all
    anchors belonging to that subcategory. `aggregation`:
      - "max":  picks the best-matching anchor per passage (multilingual-friendly:
                a German passage matches its German anchor, French its French anchor)
      - "mean": averages — equivalent to scoring against the mean anchor embedding

    Returns dict mapping subcategory_id → DataFrame of top-k paragraphs (with
    columns added: similarity_score, subcategory_id).
    """
    if aggregation not in ("max", "mean"):
        raise ValueError(f"aggregation must be 'max' or 'mean', got {aggregation!r}")

    logger.info(
        f"Anchor cosine retrieval: {len(paragraphs_df)} paragraphs × "
        f"{len(anchor_embeddings)} anchors / {num_categories} categories, "
        f"top_k={top_k}, aggregation={aggregation}"
    )

    # All-pairs similarity: (n_paragraphs, n_anchors)
    sims = cosine_similarity_matrix(paragraph_embeddings, anchor_embeddings)

    results = {}
    for cat_id in range(num_categories):
        anchor_idx = [i for i, c in enumerate(anchor_categories) if c == cat_id]
        if not anchor_idx:
            logger.warning(f"No anchors for subcategory {cat_id}; skipping")
            continue

        cat_sims = sims[:, anchor_idx]  # (n_paragraphs, n_anchors_in_cat)
        scores = cat_sims.max(axis=1) if aggregation == "max" else cat_sims.mean(axis=1)

        k = min(top_k, len(scores))
        top_indices = np.argsort(scores)[-k:][::-1]

        top_df = paragraphs_df.iloc[top_indices].copy()
        top_df["similarity_score"] = scores[top_indices]
        top_df["subcategory_id"] = cat_id

        results[cat_id] = top_df
        logger.info(
            f"  Subcategory {cat_id}: top-{k} score range "
            f"[{scores[top_indices[-1]]:.4f} – {scores[top_indices[0]]:.4f}]"
        )

    return results


def retrieve_for_pipeline(
    paragraphs_df: pd.DataFrame,
    paragraph_embeddings: np.ndarray,
    anchor_embeddings: np.ndarray,
    anchor_categories: list[int],
    config: PipelineConfig,
    subcategories: list[dict],
) -> dict[int, pd.DataFrame] | dict[tuple[str, int], pd.DataFrame]:
    """Full retrieval step. Optionally stratifies by groups before retrieval.

    Returns:
        groups disabled → dict[subcategory_id → DataFrame]
        groups enabled  → dict[(group_name, subcategory_id) → DataFrame]
    """
    num_categories = len(subcategories)
    top_k = config.retrieval.top_k
    aggregation = config.retrieval.aggregation

    if not config.groups.enabled:
        return anchor_cosine_retrieval(
            paragraphs_df, paragraph_embeddings,
            anchor_embeddings, anchor_categories,
            num_categories, top_k, aggregation,
        )

    groups_path = config.resolve_path(config.groups.groups_file)
    with open(groups_path, encoding="utf-8") as f:
        groups_data = json.load(f)

    column = groups_data.get("column", config.groups.column)
    if column not in paragraphs_df.columns:
        raise ValueError(
            f"Group stratification requested but column '{column}' not in corpus. "
            f"Columns available: {list(paragraphs_df.columns)}"
        )

    results = {}
    for group_name, group_info in groups_data["groups"].items():
        values = group_info.get("values", group_info.get("medium_codes", []))
        mask = paragraphs_df[column].isin(values)
        group_df = paragraphs_df[mask].copy()
        group_embeddings = paragraph_embeddings[mask.values]

        logger.info(f"Group '{group_name}': {len(group_df)} paragraphs")
        if len(group_df) == 0:
            continue

        group_results = anchor_cosine_retrieval(
            group_df, group_embeddings,
            anchor_embeddings, anchor_categories,
            num_categories, top_k, aggregation,
        )
        for cat_id, cat_df in group_results.items():
            cat_df["group"] = group_name
            results[(group_name, cat_id)] = cat_df

    return results
