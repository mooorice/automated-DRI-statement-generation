"""CLI entry point for the AutoConcourse pipeline."""

import argparse
import logging

from .config import load_config
from .utils import setup_logging, load_json, save_concourse

logger = logging.getLogger("autoconcourse.cli")

PIPELINE_STEPS = ["corpus", "anchors", "embed", "retrieve", "generate"]


def pipeline_main(args=None):
    parser = argparse.ArgumentParser(description="Run the AutoConcourse pipeline")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--steps",
        default="all",
        help=f"Comma-separated steps to run: {','.join(PIPELINE_STEPS)} (default: all)",
    )
    parser.add_argument("--force", action="store_true", help="Force re-run even if cached outputs exist")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parsed = parser.parse_args(args)

    setup_logging(parsed.log_level)
    config = load_config(parsed.config)

    steps = PIPELINE_STEPS if parsed.steps == "all" else parsed.steps.split(",")
    run_pipeline(config, steps, force=parsed.force)


def run_pipeline(config, steps, force=False):
    from .corpus import prepare_corpus
    from .anchors import load_or_generate_anchors, load_anchors
    from .embedding import embed_dataframe
    from .retrieval import retrieve_for_pipeline
    from .generation import generate_statements, create_llm_client
    from .utils import save_embeddings, load_embeddings

    cache = config.cache_dir
    cache.mkdir(parents=True, exist_ok=True)

    subcat_path = config.resolve_path(config.anchors.subcategories_file)
    subcategories = load_json(subcat_path)

    paragraphs_df = None
    anchors_df = None
    para_embeddings = None
    anchor_embeddings = None
    retrieved = None

    if "corpus" in steps:
        logger.info("=== Step: Corpus preparation ===")
        corpus_cache = cache / "paragraphs.csv"
        if corpus_cache.exists() and not force:
            import pandas as pd
            paragraphs_df = pd.read_csv(corpus_cache)
            logger.info(f"Loaded cached corpus: {len(paragraphs_df)} rows")
        else:
            paragraphs_df = prepare_corpus(config)
            paragraphs_df.to_csv(corpus_cache, index=False)

    if "anchors" in steps:
        logger.info("=== Step: Anchor preparation ===")
        if not config.anchors.enabled:
            logger.info("Anchors disabled in config, skipping")
        else:
            anchors_df = load_or_generate_anchors(config)

    if "embed" in steps:
        logger.info("=== Step: Embedding ===")
        if paragraphs_df is None:
            import pandas as pd
            paragraphs_df = pd.read_csv(cache / "paragraphs.csv")
        if anchors_df is None:
            anchors_df = load_anchors(cache / "anchors.csv")

        para_emb_path = cache / "paragraph_embeddings.npy"
        anchor_emb_path = cache / "anchor_embeddings.npy"

        para_embeddings = embed_dataframe(paragraphs_df, config, para_emb_path if not force else None)
        anchor_embeddings = embed_dataframe(anchors_df, config, anchor_emb_path if not force else None)

        if force or not para_emb_path.exists():
            save_embeddings(para_embeddings, para_emb_path)
        if force or not anchor_emb_path.exists():
            save_embeddings(anchor_embeddings, anchor_emb_path)

    if "retrieve" in steps:
        logger.info("=== Step: Retrieval ===")
        if paragraphs_df is None:
            import pandas as pd
            paragraphs_df = pd.read_csv(cache / "paragraphs.csv")
        if anchors_df is None:
            anchors_df = load_anchors(cache / "anchors.csv")
        if para_embeddings is None:
            para_embeddings = load_embeddings(cache / "paragraph_embeddings.npy")
        if anchor_embeddings is None:
            anchor_embeddings = load_embeddings(cache / "anchor_embeddings.npy")

        anchor_categories = anchors_df["category"].tolist()
        retrieved = retrieve_for_pipeline(
            paragraphs_df, para_embeddings,
            anchor_embeddings, anchor_categories,
            config, subcategories,
        )

    if "generate" in steps:
        logger.info("=== Step: Generation ===")
        if retrieved is None:
            raise RuntimeError("Retrieval must run before generation. Include 'retrieve' in steps.")

        client = create_llm_client(config)
        statements = generate_statements(retrieved, subcategories, config, client)

        output_dir = config.output_dir / "anchor_guided"
        save_concourse(
            statements=statements,
            output_path=output_dir / "concourse.json",
            system="anchor_guided",
            topic=config.topic.id,
            model=config.generation.model,
            config_hash=config.config_hash,
        )

        logger.info(f"Pipeline complete. Output: {output_dir / 'concourse.json'}")
        return statements

    logger.info("Pipeline steps completed.")


if __name__ == "__main__":
    pipeline_main()
