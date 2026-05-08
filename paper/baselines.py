"""Baseline systems for comparison (B1-B6)."""

import json
import logging
import random
from collections import Counter

import numpy as np
import pandas as pd

from autoconcourse.config import PipelineConfig
from autoconcourse.generation import create_llm_client, generate_statements, build_generation_prompt, _call_llm_with_retry
from autoconcourse.prompts import load_prompt_sections
from autoconcourse.retrieval import cosine_similarity_vec, aggregate_anchor_embeddings

logger = logging.getLogger("autoconcourse.baselines")


def run_baseline(
    baseline_name: str,
    paragraphs_df: pd.DataFrame,
    paragraph_embeddings: np.ndarray,
    anchor_embeddings: np.ndarray | None,
    anchor_categories: list[int] | None,
    subcategories: list[dict],
    config: PipelineConfig,
    client=None,
) -> list[dict]:
    """Run a specific baseline and return generated statements."""
    baselines = {
        "B1_zero_context": run_zero_context,
        "B2_random_retrieval": run_random_retrieval,
        "B3_bm25_retrieval": run_bm25_retrieval,
        "B4_bertopic": run_bertopic,
        "B5_extractive": run_extractive,
        "B6_dense_no_anchor": run_dense_no_anchor,
    }

    if baseline_name not in baselines:
        raise ValueError(f"Unknown baseline: {baseline_name}. Available: {list(baselines.keys())}")

    logger.info(f"Running baseline: {baseline_name}")
    return baselines[baseline_name](
        paragraphs_df=paragraphs_df,
        paragraph_embeddings=paragraph_embeddings,
        anchor_embeddings=anchor_embeddings,
        anchor_categories=anchor_categories,
        subcategories=subcategories,
        config=config,
        client=client,
    )


def run_zero_context(
    subcategories: list[dict],
    config: PipelineConfig,
    client=None,
    **kwargs,
) -> list[dict]:
    """B1: LLM generates statements with topic + subcategories only, no corpus."""
    if client is None:
        client = create_llm_client(config)

    sections = load_prompt_sections("statement_generation")
    system_prompt = sections.get("system", "")

    all_statements = []
    counter = 1

    for subcat in subcategories:
        topic_focus = subcat.get("topic_focus", subcat["name"])
        prompt = (
            f"Generate {config.generation.statements_per_group_per_subcategory} consideration "
            f"statements about: {topic_focus}\n\n"
            f"This is for a survey about: {config.topic.name}\n\n"
            f"Write all statements in {config.generation.output_language}.\n\n"
            "Format: Return a JSON array of objects, each with 'id' (int) and 'text' (string) fields."
        )

        raw = _call_llm_with_retry(
            client=client,
            model=config.generation.model,
            system_prompt=system_prompt,
            user_prompt=prompt,
            temperature=config.generation.temperature,
        )

        for stmt in raw:
            text = stmt.get("text", stmt) if isinstance(stmt, dict) else str(stmt)
            all_statements.append({
                "id": f"{config.topic.id}_B1_{counter:03d}",
                "text": text,
                "subcategory": subcat["name"],
                "subcategory_id": subcat["id"],
                "perspective_group": None,
                "source_paragraphs": [],
                "type": "consideration",
            })
            counter += 1

    logger.info(f"B1 zero-context: generated {len(all_statements)} statements")
    return all_statements


def run_random_retrieval(
    paragraphs_df: pd.DataFrame,
    subcategories: list[dict],
    config: PipelineConfig,
    client=None,
    **kwargs,
) -> list[dict]:
    """B2: Random paragraph sampling → LLM generation."""
    if client is None:
        client = create_llm_client(config)

    random.seed(42)
    top_k = config.retrieval.top_k

    retrieved = {}
    for subcat in subcategories:
        cat_id = subcat["id"]
        sample_size = min(top_k, len(paragraphs_df))
        indices = random.sample(range(len(paragraphs_df)), sample_size)
        retrieved[cat_id] = paragraphs_df.iloc[indices].copy()
        retrieved[cat_id]["similarity_score"] = 0.0
        retrieved[cat_id]["subcategory_id"] = cat_id

    return generate_statements(retrieved, subcategories, config, client)


def run_bm25_retrieval(
    paragraphs_df: pd.DataFrame,
    subcategories: list[dict],
    config: PipelineConfig,
    client=None,
    **kwargs,
) -> list[dict]:
    """B3: BM25 sparse keyword retrieval using anchor text as query → LLM."""
    from rank_bm25 import BM25Okapi

    if client is None:
        client = create_llm_client(config)

    top_k = config.retrieval.top_k

    # Tokenize corpus
    corpus_texts = paragraphs_df["text"].tolist()
    tokenized_corpus = [doc.lower().split() for doc in corpus_texts]
    bm25 = BM25Okapi(tokenized_corpus)

    retrieved = {}
    for subcat in subcategories:
        cat_id = subcat["id"]
        query = subcat.get("description", subcat["name"])
        tokenized_query = query.lower().split()

        scores = bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[-top_k:][::-1]

        top_df = paragraphs_df.iloc[top_indices].copy()
        top_df["similarity_score"] = scores[top_indices]
        top_df["subcategory_id"] = cat_id
        retrieved[cat_id] = top_df

    return generate_statements(retrieved, subcategories, config, client)


def run_bertopic(
    paragraphs_df: pd.DataFrame,
    paragraph_embeddings: np.ndarray,
    subcategories: list[dict],
    config: PipelineConfig,
    client=None,
    **kwargs,
) -> list[dict]:
    """B4: BERTopic clustering → map clusters to subcategories → LLM."""
    from hdbscan import HDBSCAN
    from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

    if client is None:
        client = create_llm_client(config)

    num_categories = len(subcategories)
    top_k = config.retrieval.top_k

    # Cluster with HDBSCAN
    clusterer = HDBSCAN(min_cluster_size=50, metric="euclidean")
    cluster_labels = clusterer.fit_predict(paragraph_embeddings)

    # Compute cluster centroids
    unique_clusters = [c for c in set(cluster_labels) if c >= 0]
    centroids = {}
    for c in unique_clusters:
        mask = cluster_labels == c
        centroids[c] = paragraph_embeddings[mask].mean(axis=0)

    # Map clusters to subcategories using anchor similarity
    anchor_emb = kwargs.get("anchor_embeddings")
    anchor_cats = kwargs.get("anchor_categories")
    if anchor_emb is not None and anchor_cats is not None:
        agg_anchors = aggregate_anchor_embeddings(anchor_emb, anchor_cats, num_categories)
    else:
        from autoconcourse.embedding import embed_texts
        subcat_texts = [s.get("description", s["name"]) for s in subcategories]
        agg_anchors = embed_texts(subcat_texts, config.embedding.model)

    # Assign each cluster to nearest subcategory
    cluster_to_cat = {}
    for c, centroid in centroids.items():
        sims = sklearn_cosine([centroid], agg_anchors)[0]
        cluster_to_cat[c] = int(np.argmax(sims))

    # Retrieve paragraphs per subcategory
    retrieved = {}
    for cat_id in range(num_categories):
        matching_clusters = [c for c, mapped in cluster_to_cat.items() if mapped == cat_id]
        mask = np.isin(cluster_labels, matching_clusters)
        cat_df = paragraphs_df[mask].copy()

        if len(cat_df) > top_k:
            cat_df = cat_df.sample(top_k, random_state=42)

        cat_df["subcategory_id"] = cat_id
        retrieved[cat_id] = cat_df

    return generate_statements(retrieved, subcategories, config, client)


def run_extractive(
    paragraphs_df: pd.DataFrame,
    paragraph_embeddings: np.ndarray,
    subcategories: list[dict],
    config: PipelineConfig,
    anchor_embeddings: np.ndarray | None = None,
    anchor_categories: list[int] | None = None,
    **kwargs,
) -> list[dict]:
    """B5: Centrality-based sentence extraction, no LLM generation.

    Selects the most central paragraphs (closest to centroid) per subcategory.
    Returns these directly as statements.
    """
    num_categories = len(subcategories)
    n_per_cat = config.generation.statements_per_group_per_subcategory

    # Use anchor-guided retrieval to assign paragraphs to categories
    if anchor_embeddings is not None and anchor_categories is not None:
        agg_anchors = aggregate_anchor_embeddings(
            anchor_embeddings, anchor_categories, num_categories
        )
    else:
        from autoconcourse.embedding import embed_texts
        subcat_texts = [s.get("description", s["name"]) for s in subcategories]
        agg_anchors = embed_texts(subcat_texts, config.embedding.model)

    all_statements = []
    counter = 1

    for cat_id in range(num_categories):
        # Get similarity scores
        scores = cosine_similarity_vec(paragraph_embeddings, agg_anchors[cat_id])
        top_indices = np.argsort(scores)[-n_per_cat:][::-1]

        for idx in top_indices:
            text = paragraphs_df.iloc[idx]["text"]
            # Truncate long paragraphs for statement use
            if len(text) > 300:
                text = text[:297] + "..."

            all_statements.append({
                "id": f"{config.topic.id}_B5_{counter:03d}",
                "text": text,
                "subcategory": subcategories[cat_id]["name"],
                "subcategory_id": cat_id,
                "perspective_group": None,
                "source_paragraphs": [int(paragraphs_df.iloc[idx].get("id", idx))],
                "type": "consideration",
            })
            counter += 1

    logger.info(f"B5 extractive: selected {len(all_statements)} statements")
    return all_statements


def run_dense_no_anchor(
    paragraphs_df: pd.DataFrame,
    paragraph_embeddings: np.ndarray,
    subcategories: list[dict],
    config: PipelineConfig,
    client=None,
    **kwargs,
) -> list[dict]:
    """B6: Single topic embedding → global top-k → LLM (no anchor subcategories)."""
    from autoconcourse.embedding import embed_texts

    if client is None:
        client = create_llm_client(config)

    top_k = config.retrieval.top_k

    # Create a single embedding from the topic name
    topic_embedding = embed_texts(
        [config.topic.name],
        config.embedding.model,
        device=config.embedding.device,
    )[0]

    # Global top-k by similarity to topic embedding
    scores = cosine_similarity_vec(paragraph_embeddings, topic_embedding)
    top_indices = np.argsort(scores)[-top_k:][::-1]

    # Create a single retrieval group (no subcategories)
    retrieved = {}
    for subcat in subcategories:
        cat_id = subcat["id"]
        top_df = paragraphs_df.iloc[top_indices].copy()
        top_df["similarity_score"] = scores[top_indices]
        top_df["subcategory_id"] = cat_id
        retrieved[cat_id] = top_df

    return generate_statements(retrieved, subcategories, config, client)


ALL_BASELINES = [
    "B1_zero_context",
    "B2_random_retrieval",
    "B3_bm25_retrieval",
    "B4_bertopic",
    "B5_extractive",
    "B6_dense_no_anchor",
]
