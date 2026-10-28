"""CLI entry points for the parked paper validation work (baselines, evaluation, ablations).

These were lifted out of autoconcourse/cli.py during the bare-method strip.
They will need rewiring when the paper validation phase resumes — in particular,
the ablation overrides reference config sections (reduction, perspective_groups)
that no longer exist in the bare method's config schema.
"""

import argparse
import copy
import json
import logging
import sys
from pathlib import Path

from autoconcourse.config import load_config
from autoconcourse.cli import run_pipeline
from autoconcourse.utils import setup_logging, load_json, save_json, save_concourse

logger = logging.getLogger("paper.cli")


def baselines_main(args=None):
    """Run baseline systems (B1–B6) against a topic config."""
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
    from autoconcourse.utils import load_embeddings

    baseline_list = ALL_BASELINES if parsed.baselines == "all" else parsed.baselines.split(",")

    subcat_path = config.resolve_path(config.anchors.subcategories_file)
    subcategories = load_json(subcat_path)

    import pandas as pd
    paragraphs_df = pd.read_csv(config.paragraphs_cache_path)

    # Bare method has no reduction step; baselines use raw embeddings.
    para_emb = load_embeddings(config.paragraph_embeddings_path)

    anchor_emb = None
    anchor_cats = None
    try:
        from autoconcourse.anchors import load_anchors, resolve_anchors_path
        anchors_df = load_anchors(resolve_anchors_path(config))
        anchor_emb = load_embeddings(config.anchor_embeddings_path)
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
    """Evaluate system outputs against a reference concourse."""
    parser = argparse.ArgumentParser(description="Evaluate system outputs")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--systems", default="all", help="Comma-separated system names or 'all'")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parsed = parser.parse_args(args)

    setup_logging(parsed.log_level)
    config = load_config(parsed.config)

    from .evaluation import evaluate_all, results_to_latex
    from autoconcourse.utils import load_concourse

    if not config.evaluation.reference_file:
        logger.error("No reference_file specified in config")
        sys.exit(1)

    ref_path = config.resolve_path(config.evaluation.reference_file)
    reference = load_concourse(ref_path)

    output_base = config.output_dir
    if parsed.systems == "all":
        system_dirs = [d for d in output_base.iterdir() if d.is_dir()]
    else:
        system_dirs = [output_base / s for s in parsed.systems.split(",")]

    subcat_path = config.resolve_path(config.anchors.subcategories_file)
    subcategories = load_json(subcat_path)

    corpus_paragraphs = None
    corpus_cache = config.paragraphs_cache_path
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

        results_path = Path("evaluation/results") / config.topic.id / f"{sys_dir.name}.json"
        save_json(results, results_path)

    if all_results:
        latex = results_to_latex(all_results)
        table_path = Path("evaluation/results") / config.topic.id / "comparison_table.tex"
        table_path.parent.mkdir(parents=True, exist_ok=True)
        table_path.write_text(latex)
        logger.info(f"Comparison table saved to {table_path}")


# NOTE: ablation overrides below reference config sections (reduction, perspective_groups)
# that may no longer exist in the bare-method config schema. Update these when reattaching
# the paper work — the schema needs to be restored or these ablations rewritten.
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
    """Run ablation experiments."""
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
            for variant in ablation["variants"]:
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

        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            yaml.dump(modified_raw, tmp)
            tmp_path = tmp.name

        try:
            config = load_config(tmp_path)
            config._base_dir = config_path.parent.parent
            run_pipeline(config, steps=["corpus", "anchors", "embed", "retrieve", "generate"])
        except Exception as e:
            logger.error(f"Experiment {exp['name']} failed: {e}")
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    logger.info(f"\nAll {len(experiments)} experiments complete.")
