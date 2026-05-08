"""Automatic evaluation metrics for concourse quality."""

import logging
from collections import Counter

import numpy as np
from sentence_transformers import SentenceTransformer

from autoconcourse.config import PipelineConfig

logger = logging.getLogger("autoconcourse.evaluation")


def _get_embeddings(texts: list[str], model_name: str = "paraphrase-multilingual-MiniLM-L12-v2") -> np.ndarray:
    """Embed texts for metric computation."""
    model = SentenceTransformer(model_name)
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=False)


def coverage(
    generated: list[str],
    reference: list[str],
    thresholds: list[float] | None = None,
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
) -> dict[float, float]:
    """Compute Coverage@threshold: fraction of reference statements matched by generated.

    For each reference statement, find the most similar generated statement.
    A reference is "covered" if max similarity >= threshold.

    Returns dict mapping threshold -> coverage score.
    """
    if thresholds is None:
        thresholds = [0.7, 0.8, 0.9]

    gen_emb = _get_embeddings(generated, model_name)
    ref_emb = _get_embeddings(reference, model_name)

    # Cosine similarity matrix: (n_ref, n_gen)
    gen_norm = gen_emb / np.linalg.norm(gen_emb, axis=1, keepdims=True)
    ref_norm = ref_emb / np.linalg.norm(ref_emb, axis=1, keepdims=True)
    sim_matrix = ref_norm @ gen_norm.T

    # Max similarity for each reference statement
    max_sims = sim_matrix.max(axis=1)

    results = {}
    for t in thresholds:
        covered = (max_sims >= t).sum()
        results[t] = covered / len(reference)

    logger.info(f"Coverage: {results}")
    return results


def groundedness(
    generated: list[str],
    corpus_paragraphs: list[str],
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
    top_k: int = 5,
) -> dict:
    """Compute groundedness: how well generated statements are supported by corpus.

    For each generated statement, find the max similarity to any corpus paragraph.
    Returns mean and per-statement grounding scores.
    """
    gen_emb = _get_embeddings(generated, model_name)

    # Process corpus in batches to avoid memory issues
    batch_size = 10000
    max_sims = np.full(len(generated), -1.0)

    for i in range(0, len(corpus_paragraphs), batch_size):
        batch = corpus_paragraphs[i : i + batch_size]
        batch_emb = _get_embeddings(batch, model_name)

        gen_norm = gen_emb / np.linalg.norm(gen_emb, axis=1, keepdims=True)
        batch_norm = batch_emb / np.linalg.norm(batch_emb, axis=1, keepdims=True)
        sim_matrix = gen_norm @ batch_norm.T

        batch_max = sim_matrix.max(axis=1)
        max_sims = np.maximum(max_sims, batch_max)

    results = {
        "mean_groundedness": float(max_sims.mean()),
        "min_groundedness": float(max_sims.min()),
        "max_groundedness": float(max_sims.max()),
        "per_statement": max_sims.tolist(),
    }

    logger.info(f"Groundedness: mean={results['mean_groundedness']:.4f}")
    return results


def distinctiveness(
    generated: list[str],
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
    redundancy_threshold: float = 0.9,
) -> dict:
    """Compute distinctiveness: how diverse/non-redundant the generated statements are.

    Measures pairwise similarity and counts near-duplicate pairs.
    """
    gen_emb = _get_embeddings(generated, model_name)
    gen_norm = gen_emb / np.linalg.norm(gen_emb, axis=1, keepdims=True)
    sim_matrix = gen_norm @ gen_norm.T

    n = len(generated)
    # Get upper triangle (excluding diagonal)
    upper_idx = np.triu_indices(n, k=1)
    pairwise_sims = sim_matrix[upper_idx]

    redundant_pairs = int((pairwise_sims >= redundancy_threshold).sum())
    total_pairs = len(pairwise_sims)

    results = {
        "mean_pairwise_similarity": float(pairwise_sims.mean()),
        "max_pairwise_similarity": float(pairwise_sims.max()),
        "redundant_pairs": redundant_pairs,
        "total_pairs": total_pairs,
        "redundancy_rate": redundant_pairs / total_pairs if total_pairs > 0 else 0.0,
        "distinctiveness_score": 1.0 - (redundant_pairs / total_pairs if total_pairs > 0 else 0.0),
    }

    logger.info(
        f"Distinctiveness: mean_sim={results['mean_pairwise_similarity']:.4f}, "
        f"redundant={redundant_pairs}/{total_pairs}"
    )
    return results


def balance(
    generated: list[dict],
    subcategories: list[dict],
) -> dict:
    """Compute balance: Jensen-Shannon Divergence between generated and uniform subcategory distribution.

    Lower JSD = better balance across subcategories.
    """
    from scipy.spatial.distance import jensenshannon

    n_cats = len(subcategories)
    uniform = np.ones(n_cats) / n_cats

    # Count statements per subcategory
    cat_counts = Counter()
    for stmt in generated:
        cat_id = stmt.get("subcategory_id", 0)
        cat_counts[cat_id] += 1

    # Build distribution
    total = sum(cat_counts.values())
    generated_dist = np.array([cat_counts.get(i, 0) / total for i in range(n_cats)])

    # Ensure valid probability distribution
    generated_dist = np.maximum(generated_dist, 1e-10)
    generated_dist = generated_dist / generated_dist.sum()

    jsd = float(jensenshannon(generated_dist, uniform))

    results = {
        "jsd_vs_uniform": jsd,
        "balance_score": 1.0 - jsd,
        "generated_distribution": generated_dist.tolist(),
        "uniform_distribution": uniform.tolist(),
        "statements_per_subcategory": dict(cat_counts),
    }

    logger.info(f"Balance: JSD={jsd:.4f}, score={results['balance_score']:.4f}")
    return results


def novelty(
    generated: list[str],
    reference: list[str],
    threshold: float = 0.7,
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
) -> dict:
    """Compute novelty: fraction of generated statements not found in reference.

    A generated statement is "novel" if its max similarity to any reference < threshold.
    """
    gen_emb = _get_embeddings(generated, model_name)
    ref_emb = _get_embeddings(reference, model_name)

    gen_norm = gen_emb / np.linalg.norm(gen_emb, axis=1, keepdims=True)
    ref_norm = ref_emb / np.linalg.norm(ref_emb, axis=1, keepdims=True)
    sim_matrix = gen_norm @ ref_norm.T

    max_sims = sim_matrix.max(axis=1)
    novel_mask = max_sims < threshold

    results = {
        "novelty_rate": float(novel_mask.mean()),
        "novel_count": int(novel_mask.sum()),
        "total_generated": len(generated),
        "threshold": threshold,
        "per_statement_max_ref_similarity": max_sims.tolist(),
    }

    logger.info(f"Novelty: {results['novel_count']}/{results['total_generated']} novel statements")
    return results


def evaluate_all(
    system_output: dict,
    reference: dict,
    config: PipelineConfig,
    corpus_paragraphs: list[str] | None = None,
    subcategories: list[dict] | None = None,
) -> dict:
    """Run all evaluation metrics.

    Args:
        system_output: Concourse JSON dict with 'statements' list
        reference: Reference concourse JSON dict with 'statements' list
        config: Pipeline configuration
        corpus_paragraphs: Optional list of corpus texts for groundedness
        subcategories: Optional subcategory definitions for balance

    Returns:
        Comprehensive results dict with all metrics
    """
    gen_texts = [s["text"] for s in system_output["statements"]]
    ref_texts = [s["text"] for s in reference["statements"]]
    model_name = config.embedding.model

    results = {
        "system": system_output.get("system", "unknown"),
        "topic": system_output.get("topic", "unknown"),
        "n_generated": len(gen_texts),
        "n_reference": len(ref_texts),
    }

    # Coverage
    results["coverage"] = coverage(gen_texts, ref_texts, config.evaluation.coverage_thresholds, model_name)

    # Distinctiveness
    results["distinctiveness"] = distinctiveness(gen_texts, model_name)

    # Novelty
    results["novelty"] = novelty(gen_texts, ref_texts, model_name=model_name)

    # Groundedness (if corpus provided)
    if corpus_paragraphs:
        results["groundedness"] = groundedness(gen_texts, corpus_paragraphs, model_name)

    # Balance (if subcategories provided)
    if subcategories:
        results["balance"] = balance(system_output["statements"], subcategories)

    return results


def results_to_latex(results: dict | list[dict], metric_keys: list[str] | None = None) -> str:
    """Generate a LaTeX table from evaluation results.

    Args:
        results: Single result dict or list of result dicts (one per system)
        metric_keys: Which metrics to include in columns

    Returns:
        LaTeX table string
    """
    if isinstance(results, dict):
        results = [results]

    if metric_keys is None:
        metric_keys = ["coverage_0.7", "coverage_0.8", "distinctiveness_score", "balance_score"]

    # Header
    cols = ["System"] + [k.replace("_", " ").title() for k in metric_keys]
    header = " & ".join(cols) + " \\\\"

    lines = [
        "\\begin{table}[h]",
        "\\centering",
        f"\\begin{{tabular}}{{{'l' + 'c' * len(metric_keys)}}}",
        "\\toprule",
        header,
        "\\midrule",
    ]

    for r in results:
        system = r.get("system", "?")
        values = [system]
        for key in metric_keys:
            if "coverage_" in key:
                threshold = float(key.split("_")[1])
                val = r.get("coverage", {}).get(threshold, "—")
                values.append(f"{val:.3f}" if isinstance(val, float) else str(val))
            elif key == "distinctiveness_score":
                val = r.get("distinctiveness", {}).get("distinctiveness_score", "—")
                values.append(f"{val:.3f}" if isinstance(val, float) else str(val))
            elif key == "balance_score":
                val = r.get("balance", {}).get("balance_score", "—")
                values.append(f"{val:.3f}" if isinstance(val, float) else str(val))
            else:
                values.append("—")

        lines.append(" & ".join(values) + " \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Evaluation results}",
        "\\label{tab:eval_results}",
        "\\end{table}",
    ])

    return "\n".join(lines)
