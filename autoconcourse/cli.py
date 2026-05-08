"""CLI entry points for AutoConcourse pipeline."""

import argparse
import copy
import json
import logging
import sys
from pathlib import Path

from .config import load_config
from .utils import setup_logging, load_json, save_json, save_concourse

logger = logging.getLogger("autoconcourse.cli")


def pipeline_main(args=None):
    """Main entry point for running the pipeline."""
    parser = argparse.ArgumentParser(description="Run the AutoConcourse pipeline")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--steps",
        default="all",
        help="Comma-separated steps to run: corpus,anchors,embed,reduce,retrieve,generate (default: all)",
    )
    parser.add_argument("--force", action="store_true", help="Force re-run even if outputs exist")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parsed = parser.parse_args(args)

    setup_logging(parsed.log_level)
    config = load_config(parsed.config)

    steps = parsed.steps.split(",") if parsed.steps != "all" else [
        "corpus", "anchors", "embed", "reduce", "retrieve", "generate"
    ]

    run_pipeline(config, steps, force=parsed.force)


def run_pipeline(config, steps, force=False):
    """Execute pipeline steps."""
    from .corpus import prepare_corpus
    from .anchors import load_or_generate_anchors
    from .embedding import embed_dataframe
    from .reduction import reduce_embeddings
    from .retrieval import retrieve_for_pipeline
    from .generation import generate_statements, create_llm_client
    from .utils import save_embeddings, load_embeddings

    cache = config.cache_dir
    cache.mkdir(parents=True, exist_ok=True)

    # Subcategories (needed throughout)
    subcat_path = config.resolve_path(config.anchors.subcategories_file)
    subcategories = load_json(subcat_path)

    paragraphs_df = None
    anchors_df = None
    para_embeddings = None
    anchor_embeddings = None
    para_reduced = None
    anchor_reduced = None
    retrieved = None

    # Step 1: Corpus
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

    # Step 2: Anchors
    if "anchors" in steps:
        logger.info("=== Step: Anchor preparation ===")
        if not config.anchors.enabled:
            logger.info("Anchors disabled in config, skipping")
        else:
            anchors_df = load_or_generate_anchors(config)

    # Step 3: Embed
    if "embed" in steps:
        logger.info("=== Step: Embedding ===")
        if paragraphs_df is None:
            import pandas as pd
            paragraphs_df = pd.read_csv(cache / "paragraphs.csv")
        if anchors_df is None:
            from .anchors import load_anchors
            anchors_df = load_anchors(cache / "anchors.csv")

        para_emb_path = cache / "paragraph_embeddings.npy"
        anchor_emb_path = cache / "anchor_embeddings.npy"

        para_embeddings = embed_dataframe(paragraphs_df, config, para_emb_path if not force else None)
        anchor_embeddings = embed_dataframe(anchors_df, config, anchor_emb_path if not force else None)

        if force or not para_emb_path.exists():
            save_embeddings(para_embeddings, para_emb_path)
        if force or not anchor_emb_path.exists():
            save_embeddings(anchor_embeddings, anchor_emb_path)

    # Step 4: Reduce
    if "reduce" in steps:
        logger.info("=== Step: Dimensionality reduction ===")
        if para_embeddings is None:
            para_embeddings = load_embeddings(cache / "paragraph_embeddings.npy")
        if anchor_embeddings is None:
            anchor_embeddings = load_embeddings(cache / "anchor_embeddings.npy")

        para_reduced_path = cache / "paragraph_embeddings_reduced.npy"
        anchor_reduced_path = cache / "anchor_embeddings_reduced.npy"

        if para_reduced_path.exists() and anchor_reduced_path.exists() and not force:
            para_reduced = load_embeddings(para_reduced_path)
            anchor_reduced = load_embeddings(anchor_reduced_path)
        else:
            para_reduced, anchor_reduced = reduce_embeddings(
                para_embeddings, anchor_embeddings, config
            )
            save_embeddings(para_reduced, para_reduced_path)
            save_embeddings(anchor_reduced, anchor_reduced_path)

    # Step 5: Retrieve
    if "retrieve" in steps:
        logger.info("=== Step: Retrieval ===")
        if paragraphs_df is None:
            import pandas as pd
            paragraphs_df = pd.read_csv(cache / "paragraphs.csv")
        if anchors_df is None:
            from .anchors import load_anchors
            anchors_df = load_anchors(cache / "anchors.csv")

        # Use reduced embeddings if available, else raw
        if para_reduced is None:
            try:
                para_reduced = load_embeddings(cache / "paragraph_embeddings_reduced.npy")
                anchor_reduced = load_embeddings(cache / "anchor_embeddings_reduced.npy")
            except FileNotFoundError:
                para_reduced = load_embeddings(cache / "paragraph_embeddings.npy")
                anchor_reduced = load_embeddings(cache / "anchor_embeddings.npy")

        anchor_categories = anchors_df["category"].tolist()

        retrieved = retrieve_for_pipeline(
            paragraphs_df, para_reduced, anchor_reduced,
            anchor_categories, config, subcategories,
        )

    # Step 6: Generate
    if "generate" in steps:
        logger.info("=== Step: Generation ===")
        if retrieved is None:
            raise RuntimeError("Retrieval must run before generation. Include 'retrieve' in steps.")

        client = create_llm_client(config)
        statements = generate_statements(retrieved, subcategories, config, client)

        # Save output
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


def baselines_main(args=None):
    """Main entry point for running baselines."""
    parser = argparse.ArgumentParser(description="Run baseline systems")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--baselines",
        default="all",
        help="Comma-separated baselines: B1_zero_context,B2_random_retrieval,... (default: all)",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parsed = parser.parse_args(args)

    setup_logging(parsed.log_level)
    config = load_config(parsed.config)

    from .baselines import ALL_BASELINES, run_baseline
    from .utils import load_embeddings

    baseline_list = ALL_BASELINES if parsed.baselines == "all" else parsed.baselines.split(",")

    # Load shared data
    cache = config.cache_dir
    subcat_path = config.resolve_path(config.anchors.subcategories_file)
    subcategories = load_json(subcat_path)

    import pandas as pd
    paragraphs_df = pd.read_csv(cache / "paragraphs.csv")

    # Prefer reduced embeddings; fall back to raw if reduction was skipped
    reduced_path = cache / "paragraph_embeddings_reduced.npy"
    raw_path = cache / "paragraph_embeddings.npy"
    if reduced_path.exists():
        para_emb = load_embeddings(reduced_path)
    else:
        para_emb = load_embeddings(raw_path)

    anchor_emb = None
    anchor_cats = None
    try:
        from .anchors import load_anchors
        anchors_df = load_anchors(cache / "anchors.csv")
        anchor_reduced_path = cache / "anchor_embeddings_reduced.npy"
        anchor_raw_path = cache / "anchor_embeddings.npy"
        if anchor_reduced_path.exists():
            anchor_emb = load_embeddings(anchor_reduced_path)
        else:
            anchor_emb = load_embeddings(anchor_raw_path)
        anchor_cats = anchors_df["category"].tolist()
    except FileNotFoundError:
        logger.warning("Anchor data not found; some baselines may fail")

    for baseline_name in baseline_list:
        logger.info(f"=== Running baseline: {baseline_name} ===")
        statements = run_baseline(
            baseline_name,
            paragraphs_df=paragraphs_df,
            paragraph_embeddings=para_emb,
            anchor_embeddings=anchor_emb,
            anchor_categories=anchor_cats,
            subcategories=subcategories,
            config=config,
        )

        output_dir = config.output_dir / baseline_name
        save_concourse(
            statements=statements,
            output_path=output_dir / "concourse.json",
            system=baseline_name,
            topic=config.topic.id,
            model=config.generation.model,
            config_hash=config.config_hash,
        )


def evaluation_main(args=None):
    """Main entry point for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate system outputs")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--systems", default="all", help="Comma-separated system names or 'all'")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parsed = parser.parse_args(args)

    setup_logging(parsed.log_level)
    config = load_config(parsed.config)

    from .evaluation import evaluate_all, results_to_latex
    from .utils import load_concourse

    # Load reference
    if not config.evaluation.reference_file:
        logger.error("No reference_file specified in config")
        sys.exit(1)

    ref_path = config.resolve_path(config.evaluation.reference_file)
    reference = load_concourse(ref_path)

    # Find system outputs
    output_base = config.output_dir
    if parsed.systems == "all":
        system_dirs = [d for d in output_base.iterdir() if d.is_dir()]
    else:
        system_dirs = [output_base / s for s in parsed.systems.split(",")]

    # Load subcategories
    subcat_path = config.resolve_path(config.anchors.subcategories_file)
    subcategories = load_json(subcat_path)

    # Load corpus paragraphs for groundedness metric
    corpus_paragraphs = None
    corpus_cache = config.cache_dir / "paragraphs.csv"
    if corpus_cache.exists():
        import pandas as pd
        corpus_df = pd.read_csv(corpus_cache)
        text_col = config.corpus.text_column
        if text_col in corpus_df.columns:
            corpus_paragraphs = corpus_df[text_col].dropna().tolist()
            logger.info(f"Loaded {len(corpus_paragraphs)} corpus paragraphs for groundedness")
    else:
        logger.warning(f"Corpus cache not found at {corpus_cache}; groundedness metric will be skipped")

    all_results = []
    for sys_dir in sorted(system_dirs):
        concourse_path = sys_dir / "concourse.json"
        if not concourse_path.exists():
            logger.warning(f"No concourse.json in {sys_dir}, skipping")
            continue

        system_output = load_concourse(concourse_path)
        logger.info(f"Evaluating: {sys_dir.name}")

        results = evaluate_all(
            system_output, reference, config,
            corpus_paragraphs=corpus_paragraphs,
            subcategories=subcategories,
        )
        all_results.append(results)

        # Save individual results
        results_path = Path("evaluation/results") / config.topic.id / f"{sys_dir.name}.json"
        save_json(results, results_path)

    # Generate comparison table
    if all_results:
        latex = results_to_latex(all_results)
        table_path = Path("evaluation/results") / config.topic.id / "comparison_table.tex"
        table_path.parent.mkdir(parents=True, exist_ok=True)
        table_path.write_text(latex)
        logger.info(f"Comparison table saved to {table_path}")


ABLATIONS = {
    "A1": {
        "name": "no_reduction",
        "description": "No dimensionality reduction",
        "overrides": {"reduction": {"method": "none"}},
    },
    "A2": {
        "name": "pca_reduction",
        "description": "PCA reduction",
        "overrides": {"reduction": {"method": "pca", "n_components": 50}},
    },
    "A3": {
        "name": "umap_reduction",
        "description": "UMAP reduction",
        "overrides": {"reduction": {"method": "umap", "n_components": 50}},
    },
    "A4": {
        "name": "no_perspective_groups",
        "description": "No perspective group stratification",
        "overrides": {"perspective_groups": {"enabled": False}},
    },
    "A5": {
        "name": "topk_variation",
        "description": "Top-k retrieval variation",
        "variants": [
            {"retrieval": {"top_k": 50}},
            {"retrieval": {"top_k": 100}},
            {"retrieval": {"top_k": 500}},
            {"retrieval": {"top_k": 1000}},
        ],
    },
    "A6": {
        "name": "embedding_model",
        "description": "Embedding model variation",
        "variants": [
            {"embedding": {"model": "paraphrase-multilingual-MiniLM-L12-v2"}},
            {"embedding": {"model": "intfloat/multilingual-e5-large"}},
        ],
    },
    "A7": {
        "name": "generator_model",
        "description": "Generator LLM variation",
        "variants": [
            {"generation": {"model": "openai/gpt-4o"}},
            {"generation": {"model": "openai/gpt-4o-mini"}},
            {"generation": {"model": "meta-llama/llama-3.1-70b-instruct"}},
        ],
    },
    "A8": {
        "name": "no_anchors",
        "description": "No anchor guidance",
        "overrides": {"anchors": {"enabled": False}},
    },
}


def _apply_overrides(raw_config: dict, overrides: dict) -> dict:
    """Deep-merge overrides into a copy of the raw config."""
    result = copy.deepcopy(raw_config)
    for section, values in overrides.items():
        if section not in result:
            result[section] = {}
        if isinstance(values, dict):
            result[section].update(values)
        else:
            result[section] = values
    return result


def experiment_main(args=None):
    """Main entry point for running ablation experiments."""
    parser = argparse.ArgumentParser(description="Run ablation experiments")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--ablations",
        default="all",
        help=f"Comma-separated ablation IDs or 'all'. Available: {','.join(ABLATIONS.keys())}",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print experiment plan without running")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parsed = parser.parse_args(args)

    setup_logging(parsed.log_level)

    import yaml
    config_path = Path(parsed.config)
    with open(config_path) as f:
        base_raw = yaml.safe_load(f)

    ablation_ids = list(ABLATIONS.keys()) if parsed.ablations == "all" else parsed.ablations.split(",")

    experiments = []
    for aid in ablation_ids:
        if aid not in ABLATIONS:
            logger.error(f"Unknown ablation: {aid}. Available: {list(ABLATIONS.keys())}")
            sys.exit(1)

        ablation = ABLATIONS[aid]
        if "variants" in ablation:
            for i, variant in enumerate(ablation["variants"]):
                variant_key = list(variant.values())[0]
                variant_val = list(variant_key.values())[0] if isinstance(variant_key, dict) else variant_key
                name = f"{aid}_{ablation['name']}_{variant_val}"
                experiments.append({
                    "name": name,
                    "ablation_id": aid,
                    "description": f"{ablation['description']}: {variant_val}",
                    "overrides": variant,
                })
        else:
            experiments.append({
                "name": f"{aid}_{ablation['name']}",
                "ablation_id": aid,
                "description": ablation["description"],
                "overrides": ablation["overrides"],
            })

    if parsed.dry_run:
        print(f"\nExperiment plan ({len(experiments)} runs):")
        print("-" * 60)
        for exp in experiments:
            print(f"  {exp['name']}: {exp['description']}")
            print(f"    Overrides: {json.dumps(exp['overrides'], indent=None)}")
        return

    logger.info(f"Running {len(experiments)} ablation experiments")

    for exp in experiments:
        logger.info(f"\n{'='*60}")
        logger.info(f"Experiment: {exp['name']} — {exp['description']}")
        logger.info(f"{'='*60}")

        modified_raw = _apply_overrides(base_raw, exp["overrides"])

        # Write temp config and load it
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            yaml.dump(modified_raw, tmp)
            tmp_path = tmp.name

        try:
            config = load_config(tmp_path)
            # Override base_dir to point to original project root
            config._base_dir = config_path.parent.parent

            run_pipeline(config, steps=["corpus", "anchors", "embed", "reduce", "retrieve", "generate"])

        except Exception as e:
            logger.error(f"Experiment {exp['name']} failed: {e}")
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    logger.info(f"\nAll {len(experiments)} experiments complete.")
