# AutoConcourse

Automated generation of Q-concourse statements from discourse corpora. Given a corpus of media texts and a set of user-defined subtopics, AutoConcourse retrieves relevant passages, optionally stratifies by source perspective, and uses LLMs to produce structured consideration statements suitable for Q-methodology surveys.

## Quick Start

```bash
# 1. Install
pip install -e ".[dev]"

# 2. Set API key (for LLM-based steps)
echo "OPENROUTER_API_KEY=your-key-here" > .env

# 3. Run the pipeline
python run_pipeline.py --config configs/healthcare_ch.yaml
```

## Pipeline Overview

AutoConcourse is a config-driven pipeline with six stages:

1. **Corpus** — Load, chunk, and keyword-filter a text corpus (CSV, TSV, JSONL)
2. **Anchors** — Load or LLM-generate multilingual "content anchor" paragraphs per subtopic
3. **Embed** — Encode corpus paragraphs and anchors with a SentenceTransformer model
4. **Reduce** — Optional dimensionality reduction (PCA, UMAP, or none)
5. **Retrieve** — Cosine similarity retrieval of top-k paragraphs per subtopic anchor
6. **Generate** — LLM generation of consideration statements from retrieved paragraphs

Each stage caches its outputs under `cache/{topic_id}/`. Re-running skips completed stages unless `--force` is passed.

```
python run_pipeline.py --config configs/healthcare_ch.yaml --steps corpus,embed,generate --force
```

## Project Structure

```
autoconcourse/           # Core Python package
├── config.py            # YAML config loading and validation
├── corpus.py            # Data loading, chunking, keyword filtering
├── anchors.py           # Anchor loading and LLM generation
├── embedding.py         # SentenceTransformer embedding
├── reduction.py         # PCA / UMAP dimensionality reduction
├── retrieval.py         # Anchor-guided cosine retrieval
├── generation.py        # LLM statement generation
├── baselines.py         # Six baseline systems (B1–B6)
├── evaluation.py        # Coverage, groundedness, distinctiveness metrics
├── prompts/             # Prompt templates (editable text files)
├── utils.py             # Logging, embedding I/O, JSON helpers
└── cli.py               # CLI orchestration

configs/                 # Topic configurations (YAML)
├── schema.yaml          # Reference config with all options documented
├── healthcare_ch.yaml   # Swiss healthcare costs
└── food_policy_ch.yaml  # Swiss food policy

data/                    # Topic data files
├── example/             # Example formats and templates for new topics
├── healthcare_ch/       # Keywords, subcategories, perspective groups
└── food_policy_ch/      # Keywords, subcategories, perspective groups

run_pipeline.py          # Main entry point
run_baselines.py         # Run baseline systems
run_evaluation.py        # Evaluate outputs against reference concourse
run_experiment.py        # Ablation experiments
legacy/                  # Original thesis notebooks (archived)
```

## Configuration

Each topic is defined by a single YAML file. See `configs/schema.yaml` for all options or `data/example/example_config.yaml` for a minimal template.

Key configuration sections:

| Section | Purpose |
|---|---|
| `topic` | Name, ID, languages |
| `corpus` | Source file, chunking method, keyword filtering |
| `anchors` | Method (`llm_generated` or `from_file`), subcategory definitions |
| `embedding` | SentenceTransformer model, batch size, device |
| `reduction` | Method (`none`, `pca`, `umap`), target dimensions |
| `retrieval` | Method, top-k per subcategory |
| `perspective_groups` | Optional source stratification by political leaning |
| `generation` | LLM model, statements per group, output language |
| `evaluation` | Reference concourse file, coverage thresholds |

## Baselines

Six baseline systems for comparison:

| ID | Name | Description |
|---|---|---|
| B1 | Zero-context | LLM generates from topic description only (no corpus) |
| B2 | Random retrieval | Random paragraphs fed to LLM |
| B3 | BM25 retrieval | BM25 keyword retrieval instead of dense |
| B4 | BERTopic | HDBSCAN clustering with anchor mapping |
| B5 | Extractive | Top paragraphs as statements (no LLM) |
| B6 | Dense no-anchor | Single topic embedding, global retrieval |

```bash
python run_baselines.py --config configs/healthcare_ch.yaml --baselines B1,B3,B6
```

## Evaluation

Compares generated concourses against a human-authored reference:

```bash
python run_evaluation.py --config configs/healthcare_ch.yaml --systems all
```

Metrics: coverage (at thresholds), groundedness, distinctiveness, balance, novelty. Outputs per-system JSON and a LaTeX comparison table under `evaluation/results/`.

## Ablation Experiments

Run systematic ablations (dim. reduction method, top-k, embedding model, generator model, etc.):

```bash
python run_experiment.py --config configs/healthcare_ch.yaml --ablations all
python run_experiment.py --config configs/healthcare_ch.yaml --ablations A1,A5 --dry-run
```

Available ablations: A1 (no reduction), A2 (PCA), A3 (UMAP), A4 (no perspective groups), A5 (top-k variation), A6 (embedding model), A7 (generator model), A8 (no anchors).

## Setting Up a New Topic

1. Create data files using `data/example/` as templates:
   - `keywords.json` — filtering keywords per language
   - `subcategories.json` — subtopic definitions with anchor prompts
   - (Optional) `perspective_groups.json` — source groupings
   - (Optional) `reference_concourse.json` — human reference for evaluation
2. Place your corpus CSV at the path specified in your config
3. Copy `configs/schema.yaml` to `configs/your_topic.yaml` and fill in paths
4. Run: `python run_pipeline.py --config configs/your_topic.yaml`

## Environment

- Python >= 3.11
- API key for LLM access (OpenRouter recommended; any OpenAI-compatible endpoint works)
- Set `OPENROUTER_API_KEY` or `OPENAI_API_KEY` in `.env`

## License

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC_BY--NC_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
