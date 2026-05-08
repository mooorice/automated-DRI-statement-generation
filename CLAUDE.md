# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Master's thesis project that automates the generation of Deliberative Reason Index (DRI) statements for evaluating group deliberation quality. The domain is Swiss health costs policy. Content is multilingual (German, French, Italian).

## Environment Setup

- Python 3.12 virtual environment in `.dri_env/`, auto-activated via `.envrc` (direnv)
- Install dependencies: `pip install -r requirements.txt`
- API keys are loaded from `.env` via `python-dotenv` (OPENAI_API_KEY, XAI_API_KEY)
- The Validation directory has its own `.env` for separate API keys (including swissdox liri credentials)

## Pipeline Architecture

The project is a sequential NLP pipeline. Each step produces CSV outputs consumed by the next. Steps must be run in order:

1. **`01a_get_articles.ipynb`** - Fetches news articles from the swissdox API (Liri credentials required). Outputs compressed TSV to `data/00_downloads/`.

2. **`01b_content_anchor_generation.ipynb`** - Manually constructs "content anchors" — trilingual (DE/FR/IT) summary paragraphs for 8 policy subtopics generated via GPT-4o prompts against a policy analysis PDF (`data/99_anchor_data/`). Outputs anchors to `data/01_text_data/anchors.csv`. The 8 categories (0-7) are: general health costs, health system reforms, healthcare utilization, hospital planning/financing, compulsory insurance design, health prevention incentives, health service coordination, household financial burden.

3. **`02_chunk_into_paragraphs.ipynb`** - Extracts paragraphs from HTML content in articles, filters by health-related keywords, outputs to `data/01_text_data/paragraphs.csv`.

4. **`03_generate_embeddings.py`** - Embeds text using `paraphrase-multilingual-MiniLM-L12-v2` (sentence-transformers). Must be run separately for paragraphs and anchors by changing `input_data_path`/`output_data_path`. Outputs to `data/02_embedded_data/`.

5. **`04_dimensionality_reduction.py`** - UMAP reduction (50 dimensions, cosine metric) on combined paragraph + anchor embeddings. Outputs to `data/03_dimensionality_reduced_data/`.

6. **`05_categorization.py`** - Cosine similarity scoring between each paragraph and each of the 8 aggregated anchor category embeddings. Outputs to `data/04_categorized_data/`.

7. **`06_similarity_score_analysis.ipynb`** - Analyzes distribution of similarity scores (exploratory).

8. **`07_generate_dri_statements.ipynb`** - Groups media outlets by political leaning (rechts/rechtsliberal/mitte/linksliberal/links), selects top-500 paragraphs per similarity score per group, then uses OpenAI Assistants API (GPT-4o with file_search) to generate consideration statements and policy options. Outputs to `data/06_LLM_data/outputs/`.

## Data Directory Layout

```
data/
  00_downloads/          # Raw swissdox downloads
  01_text_data/          # Chunked paragraphs + anchor texts (CSVs)
  02_embedded_data/      # Sentence-transformer embeddings
  03_dimensionality_reduced_data/  # UMAP-reduced embeddings
  04_categorized_data/   # Similarity-scored paragraphs
  05_LLM_data/inputs/    # Example DRI statements, top similarity texts per group
  98_side_results/       # Alternative approaches (BERTopic, clustering) — ran on Azure
  99_anchor_data/        # Policy analysis PDF used for anchor generation
```

## Running Scripts

```bash
# Standalone Python scripts (steps 3-5)
python 03_generate_embeddings.py
python 04_dimensionality_reduction.py
python 05_categorization.py
```

Notebooks (steps 1, 2, 6-8) should be run interactively in Jupyter. Step 7 (`07_generate_dri_statements.ipynb`) requires manual iteration — you run it once per political leaning group, changing `political_leaning_list` index each time.

## Validation

The `Validation/` directory mirrors the main pipeline but for a food policy topic, used to validate the methodology against a traditionally-generated DRI survey. Final comparison is manual, done in `Validation/data/06_comparison/` Excel files.

## Key Technical Details

- Embedding model: `paraphrase-multilingual-MiniLM-L12-v2` (handles DE/FR/IT)
- UMAP params: `n_components=50, n_neighbors=30, min_dist=0.0, metric='cosine'`
- The pipeline stores embeddings as string-serialized lists in CSVs, parsed back with `ast.literal_eval`
- Step 7 uses the OpenAI Assistants API (beta) with vector stores and file_search tool
- Media outlet codes map to Swiss newspapers grouped by political alignment (defined in `07_generate_dri_statements.ipynb`)

# Main Task

Your task will be to improve the codebase to make this a proper scientific contribution. We will generalize the method for concourse generation and build a robust pipeline for reproducibility. We validate test that pipeline across at least 3 sufficiently different cases. We will present the work at CompText conference and submit to EMNLP conference for peer reviewed publication.
z