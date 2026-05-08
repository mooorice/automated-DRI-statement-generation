# AutoConcourse: Codebase Refactoring & Pipeline Implementation Plan

## Context

This project transforms a Master's thesis on automated DRI statement generation into **AutoConcourse** — a generalizable, config-driven, end-to-end pipeline for automatic generation of Q-concourse statements from large discourse corpora. The pipeline will be validated across 3 topics, benchmarked against baselines, and submitted as a paper to CompText (mid-April) and EMNLP (mid-May).

**Current state:** A collection of numbered Jupyter notebooks and scripts with hardcoded Swiss health cost parameters, using the OpenAI Assistants API (beta). A parallel copy in `Validation/` handles food policy.

**Target state:** A modular, config-driven Python package (`autoconcourse/`) where any researcher can run the full pipeline on a new topic by providing a YAML config file and optionally defining anchors interactively. LLM access is configurable via the `openai` Python package (default: OpenRouter, but any OpenAI-compatible endpoint). Includes baselines, evaluation metrics, and reproducibility infrastructure.

---

## Phase 1: Project Restructuring (Foundation)

### 1.1 Create new package structure

```
autoconcourse/
├── __init__.py
├── config.py              # Config loading, validation, schema
├── corpus.py              # Data loading, chunking, keyword filtering
├── anchors.py             # Anchor loading, validation, LLM-assisted generation
├── embedding.py           # Sentence embedding (any SBERT model)
├── reduction.py           # Dimensionality reduction (PCA, UMAP, none) — experimental
├── retrieval.py           # Anchor-guided cosine retrieval + alternatives
├── generation.py          # LLM statement generation (configurable endpoint)
├── prompt_optimization.py # Iterative LLM-driven prompt self-improvement
├── evaluation.py          # Automatic metrics (TBD — discuss before implementing)
├── baselines.py           # Baseline systems (TBD — discuss before implementing)
├── utils.py               # Shared utilities (embedding I/O, JSON schemas)
├── cli.py                 # CLI entry points
├── prompts/               # Prompt templates (Jinja2 or plain text)
│   ├── anchor_generation.txt
│   ├── statement_generation.txt
│   └── prompt_optimization.txt
notebooks/
├── anchor_workshop.ipynb  # Interactive anchor definition & generation
├── exploration.ipynb      # Corpus exploration & similarity analysis
configs/
├── schema.yaml            # Config schema / reference with all options documented
├── healthcare_ch.yaml     # Swiss healthcare topic
├── food_policy_ch.yaml    # Swiss food policy topic
├── topic3.yaml            # 3rd topic (TBD)
data/
├── example/               # Example data formats for new users
│   ├── README.md          # Documentation of all data formats
│   ├── example_config.yaml
│   ├── example_keywords.json
│   ├── example_subcategories.json
│   ├── example_anchors.json
│   ├── example_perspective_groups.json
│   ├── example_reference_concourse.json
│   └── example_corpus.csv
├── healthcare_ch/
│   ├── keywords.json
│   ├── subcategories.json
│   ├── anchors.json       # Pre-generated or manually defined anchors
│   ├── perspective_groups.json   # Optional political leaning groups
│   ├── reference_concourse.json  # Human reference for evaluation
│   └── corpus/                   # Raw/processed corpus files
├── food_policy_ch/
│   └── ...
outputs/                   # Pipeline outputs per topic × system
├── healthcare_ch/
│   ├── anchor_guided/
│   │   └── concourse.json
│   ├── zero_context/
│   │   └── concourse.json
│   └── ...
evaluation/
├── results/               # Computed metrics per experiment
├── human/                 # Human evaluation data
│   ├── annotation_guidelines.md
│   └── annotations.csv
└── figures/               # Generated plots
run_pipeline.py            # Main entry: config → concourse
run_evaluation.py          # Evaluate: outputs + reference → metrics
run_baselines.py           # Run all baselines for a topic
requirements.txt           # Pinned dependencies
pyproject.toml             # Package metadata
README.md                  # Setup, usage, reproduction instructions
```

### 1.2 Config schema design

Each topic is defined by a single YAML config. The LLM client is configured via `openai` package with a configurable `base_url` (OpenRouter by default, but any OpenAI-compatible endpoint works).

```yaml
topic:
  name: "Rising healthcare costs in Switzerland"
  id: "healthcare_ch"
  languages: ["de", "fr", "it"]

llm:
  base_url: "https://openrouter.ai/api/v1"  # Any OpenAI-compatible endpoint
  api_key_env: "OPENROUTER_API_KEY"          # Name of env var holding the key
  default_model: "openai/gpt-4o"             # Default model for all LLM steps
  extra_headers:                              # Optional headers (e.g. OpenRouter)
    HTTP-Referer: "https://autoconcourse.org"
    X-Title: "AutoConcourse"

corpus:
  source: "csv"  # or "tsv_compressed", "jsonl", "huggingface"
  path: "data/healthcare_ch/corpus/paragraphs.csv"
  text_column: "text"
  metadata_columns: ["medium_code", "language", "id"]
  chunking:
    method: "paragraph"  # "paragraph", "sentence", "fixed_length"
    min_length: 20
  filtering:
    keywords_file: "data/healthcare_ch/keywords.json"
    require_keyword: true

anchors:
  enabled: true              # Set to false to skip anchors entirely (unguided retrieval)
  method: "from_file"        # "from_file" (pre-defined), "interactive" (notebook), "llm_generated"
  anchors_file: "data/healthcare_ch/anchors.json"        # Required for "from_file"
  subcategories_file: "data/healthcare_ch/subcategories.json"  # Topic definitions (user provides these)
  # For "llm_generated" method only:
  context_document: "data/healthcare_ch/policy_analysis.pdf"
  languages: ["de", "fr", "it"]
  model: "openai/gpt-4o"    # Override default_model for anchor generation

embedding:
  model: "paraphrase-multilingual-MiniLM-L12-v2"
  batch_size: 32
  device: "cpu"  # or "cuda"

reduction:
  method: "none"  # "none" (default), "pca", "umap"
  # Dimensionality reduction is experimental. UMAP is designed for visualization,
  # not necessarily for cosine similarity preservation. PCA may be more appropriate.
  # The effect of reduction on retrieval quality is an open research question —
  # we include it as an ablation option.
  n_components: 50
  umap_params:   # Only used when method: "umap"
    n_neighbors: 30
    min_dist: 0.0
    metric: "cosine"

retrieval:
  method: "anchor_cosine"  # "anchor_cosine", "bm25", "random", "dense_no_anchor"
  top_k: 500

perspective_groups:  # Optional — omit or set enabled: false to skip stratification
  enabled: true
  groups_file: "data/healthcare_ch/perspective_groups.json"
  metadata_column: "medium_code"

generation:
  model: "openai/gpt-4o"    # Override default_model for statement generation
  statements_per_group_per_subcategory: 5
  output_language: "en"
  prompt_template: "default"  # or path to custom prompt template
  prompt_optimization:
    enabled: false           # When true, iteratively improve prompts via LLM
    iterations: 3
    optimization_model: "openai/gpt-4o"  # Model used to critique/improve prompts

evaluation:
  reference_file: "data/healthcare_ch/reference_concourse.json"
  coverage_thresholds: [0.7, 0.8, 0.9]
  # NOTE: Specific metrics TBD — will be discussed before implementation
```

**Key files to create:**
- `autoconcourse/config.py` — Load YAML, validate against schema, resolve paths
- `configs/schema.yaml` — Reference config with all options documented

### 1.3 Standard data formats

**Concourse output format** (JSON):
```json
{
  "system": "anchor_guided",
  "topic": "healthcare_ch",
  "model": "openai/gpt-4o",
  "timestamp": "2026-02-20T12:00:00Z",
  "config_hash": "abc123",
  "statements": [
    {
      "id": "hc_001",
      "text": "The mandatory health insurance system...",
      "subcategory": "insurance_organization",
      "perspective_group": "left",
      "source_paragraphs": ["p_12345", "p_67890"]
    }
  ]
}
```

**Anchors format** (`anchors.json`):
```json
{
  "topic": "healthcare_ch",
  "anchors": [
    {
      "subcategory_id": 0,
      "subcategory_name": "General health costs",
      "texts": [
        {"language": "de", "text": "Die steigenden Gesundheitskosten..."},
        {"language": "fr", "text": "L'augmentation des coûts..."},
        {"language": "it", "text": "L'aumento dei costi sanitari..."}
      ]
    }
  ]
}
```

**Subcategories format** (`subcategories.json`) — user defines these:
```json
[
  {
    "id": 0,
    "name": "General health costs",
    "description": "Overall rising healthcare costs in Switzerland and the factors driving them",
    "anchor_prompt": "Write a paragraph about rising healthcare costs in Switzerland covering insurance, social aspects, hospitals, healthcare provision, and prevention"
  }
]
```

**Reference concourse format** (JSON):
```json
{
  "topic": "healthcare_ch",
  "source": "Bevölkerungsrat 2025",
  "statements": [
    {"id": "ref_001", "text": "...", "subcategory": "insurance_organization", "type": "consideration"},
    {"id": "ref_002", "text": "...", "subcategory": null, "type": "policy_option"}
  ]
}
```

All example formats documented in `data/example/` with a `README.md`.

---

## Phase 2: Core Module Implementation

### 2.1 `autoconcourse/config.py`
- Load YAML config with defaults
- Validate required fields, resolve relative paths
- Provide typed access to config sections
- LLM client factory: `create_llm_client(config)` → `openai.OpenAI` client with configurable `base_url`, `api_key`, and `extra_headers`

### 2.2 `autoconcourse/corpus.py`
Refactor from: `01a_get_articles.ipynb`, `02_chunk_into_paragraphs.ipynb`

- `load_corpus(config)` → DataFrame with `text`, `id`, metadata columns
- `chunk_paragraphs(df, method)` → Split HTML/text into paragraphs
- `filter_by_keywords(df, keywords_file)` → Keyword-based filtering
- Support multiple input formats: CSV, TSV (compressed), JSONL, HuggingFace datasets
- **Key change:** Extract all hardcoded keywords into `keywords.json` per topic

### 2.3 `autoconcourse/anchors.py`
Refactor from: `01b_content_anchor_generation.ipynb`

Anchor generation is **optional** (`anchors.enabled: false` skips it entirely). Three modes:

1. **`from_file`** (default) — Load pre-defined anchors from `anchors.json`. User writes anchors manually or via the interactive notebook.
2. **`interactive`** — Use `notebooks/anchor_workshop.ipynb` (Jupyter notebook) where the user:
   - Defines subcategory topics
   - Optionally provides context documents
   - Generates candidate anchors via LLM
   - Reviews, edits, and approves each anchor
   - Saves final anchors to `anchors.json`
3. **`llm_generated`** — Fully automatic: generates anchors from `subcategories.json` + optional context document via LLM. User still defines the subcategory topics.

Functions:
- `load_anchors(path)` → Load anchors from JSON
- `generate_anchors_auto(subcategories, config, client)` → Automatic LLM generation
- `save_anchors(anchors, path)` → Persist anchors
- `validate_anchors(anchors, subcategories)` → Check completeness

**Key principle:** The user always defines the subcategory topics. Anchor text generation can be assisted by LLM but the topics are user-defined.

### 2.4 `autoconcourse/embedding.py`
Refactor from: `03_generate_embeddings.py`

- `EmbeddingModel` class wrapping SentenceTransformer
- `embed_texts(texts, model_name, batch_size, device)` → numpy array
- `embed_dataframe(df, text_column, model)` → adds embedding column
- Save/load embeddings as `.npy` files (not string-serialized in CSVs)
- **Key change:** Store embeddings as numpy arrays, not CSV-serialized strings parsed with `ast.literal_eval`

### 2.5 `autoconcourse/reduction.py`
Refactor from: `04_dimensionality_reduction.py`

- `reduce_embeddings(embeddings_main, embeddings_anchor, config)` → reduced arrays
- Support `none` (default — skip reduction), `pca`, `umap`
- Fit on combined main+anchor, then split back

**Open research question:** UMAP is designed for visualization (preserves local structure), not for cosine similarity in high-dimensional retrieval. PCA is a more principled choice for dimensionality reduction before cosine similarity. Default is `none` (operate on raw embeddings). The effect of reduction is treated as an **ablation experiment** — we compare `none` vs. `pca` vs. `umap` at various dimensions to determine whether reduction helps, hurts, or is neutral for retrieval quality.

### 2.6 `autoconcourse/retrieval.py`
Refactor from: `05_categorization.py`

- `anchor_cosine_retrieval(paragraphs, anchors, top_k, config)` → top-k paragraphs per subcategory
- `aggregate_anchor_embeddings(anchor_embeddings)` → mean pooling
- `cosine_similarity(a, b)` → vectorized similarity
- `unguided_retrieval(paragraphs, topic_embedding, top_k)` → retrieval without anchors
- Returns: dict of `{subcategory_id: DataFrame of top-k paragraphs}`
- When `anchors.enabled: false`, falls back to unguided dense retrieval or BM25

### 2.7 `autoconcourse/generation.py`
Refactor from: `07_generate_dri_statements.ipynb`

- `generate_statements(retrieved_texts, config, client)` → list of statements
- `build_prompt(subcategory, perspective_group, retrieved_texts, config)` → prompt string
- Prompt templates stored in `autoconcourse/prompts/`
- **Key changes:**
  - Replace OpenAI Assistants API with standard `chat.completions.create`
  - Configurable `base_url` (OpenRouter, OpenAI direct, local, etc.)
  - Context injection: concatenate retrieved paragraphs directly into the user message
  - Support configurable models via config
  - JSON mode for structured output
  - Rate limiting / retry logic

### 2.8 `autoconcourse/prompt_optimization.py`
New module — iterative LLM-driven prompt self-improvement.

Once the pipeline produces initial results, this module:
1. Takes the current prompt template + generated statements
2. Asks an LLM to critique the outputs against quality criteria (distinctiveness, clarity, groundedness, etc.)
3. Generates an improved prompt based on the critique
4. Re-runs generation with the improved prompt
5. Repeats for N iterations, tracking improvement

This runs **after** the basic pipeline is working, not during initial implementation.

### 2.9 `autoconcourse/evaluation.py`
**NOTE:** Specific metrics to be discussed before implementation. Placeholder structure:

- `evaluate_all(system_output, reference, config)` → comprehensive results dict
- `results_to_latex(results)` → LaTeX table generation
- Candidate metrics from EMNLP plan: coverage, groundedness, distinctiveness, balance, novelty
- Final metric selection depends on what's meaningful for the paper's argument

### 2.10 `autoconcourse/baselines.py`
**NOTE:** Specific baselines to be discussed before implementation. Placeholder structure:

- Each baseline produces the same output format as the main system
- Candidate baselines from EMNLP plan: zero-context, random retrieval, BM25, BERTopic, extractive, dense no-anchor
- Final baseline selection depends on what comparisons are most informative

---

## Phase 3: CLI & Pipeline Orchestration

### 3.1 `run_pipeline.py`
```
python run_pipeline.py --config configs/healthcare_ch.yaml [--steps all|embed,reduce,retrieve,generate] [--force]
```

Steps:
1. Load config
2. Load/prepare corpus (chunk, filter)
3. Load anchors (from file) — or skip if anchors disabled
4. Embed paragraphs + anchors (if enabled)
5. Dimensionality reduction (if enabled, default: skip)
6. Retrieval (anchor-guided or unguided)
7. Optionally split by perspective groups
8. LLM statement generation
9. Save concourse to `outputs/{topic}/{system}/concourse.json`

Support resuming from intermediate steps (skip completed steps if outputs exist). `--force` re-runs everything.

### 3.2 `run_baselines.py`
```
python run_baselines.py --config configs/healthcare_ch.yaml --baselines all|B1,B3,B5
```

### 3.3 `run_evaluation.py`
```
python run_evaluation.py --config configs/healthcare_ch.yaml --systems all
```

---

## Phase 4: Data Migration

The pipeline will be built fresh — existing intermediate files from the thesis are not reused. We extract only the *configuration-level* information (keywords, subcategories, prompts, media groups) from the old notebooks into structured JSON files.

### 4.1 Example data (`data/example/`)
- Create `README.md` documenting all data formats with field descriptions
- Create minimal example files for each format (keywords, subcategories, anchors, perspective groups, reference concourse, corpus CSV)
- This serves as documentation and template for new users setting up their own topic

### 4.2 Healthcare topic (`healthcare_ch`)
- Extract keywords from `02_chunk_into_paragraphs.ipynb` → `data/healthcare_ch/keywords.json`
- Extract 8 subcategory definitions + anchor prompts from `01b_content_anchor_generation.ipynb` → `data/healthcare_ch/subcategories.json`
- Extract anchor texts from `01b_content_anchor_generation.ipynb` → `data/healthcare_ch/anchors.json`
- Extract media groups from `07_generate_dri_statements.ipynb` → `data/healthcare_ch/perspective_groups.json`
- Reference concourse → `data/healthcare_ch/reference_concourse.json` (to be added when available)

### 4.3 Food policy topic (`food_policy_ch`)
- Same extraction from `Validation/` notebooks
- Reference concourse → `data/food_policy_ch/reference_concourse.json` (to be added when available)

### 4.4 Third topic (TBD)
- Config template ready; topic-specific data to be added once decided
- For EMNLP: ideally an English-language topic with open corpus for full reproducibility

---

## Phase 5: Ablation & Experiment Infrastructure

### 5.1 Ablation configs
Create config variants for systematic ablations:

| Ablation | Config change | Research question |
|---|---|---|
| A1: No reduction | `reduction.method: none` | Does dim. reduction help or hurt retrieval? |
| A2: PCA reduction | `reduction.method: pca` | Is PCA better than UMAP for this task? |
| A3: UMAP reduction | `reduction.method: umap` | How does UMAP compare? |
| A4: No perspective groups | `perspective_groups.enabled: false` | Does stratification matter? |
| A5: Top-k variation | `retrieval.top_k: 50/100/500/1000` | How sensitive is quality to k? |
| A6: Embedding model | `embedding.model: multilingual-e5-large` | Does embedding model matter? |
| A7: Generator model | Various models via OpenRouter | Does generator model matter? |
| A8: No anchors | `anchors.enabled: false` | How much do anchors contribute? |

### 5.2 Experiment runner
```
python run_experiment.py --config configs/healthcare_ch.yaml --ablations all
```

---

## Phase 6: Prompt Optimization & Refinement

After the basic pipeline produces initial results:

1. Implement `prompt_optimization.py`
2. Run optimization loop on one topic
3. Evaluate whether optimized prompts generalize across topics
4. If they do, update default prompt templates
5. Report prompt evolution in paper appendix

---

## Phase 7: Validation & Paper Support

### 7.1 Run all experiments
For each of the 3 topics:
- Main system (anchor-guided)
- Baselines (TBD)
- Key ablations

### 7.2 Generate paper assets
- `scripts/generate_tables.py` → LaTeX tables from results JSON
- `scripts/generate_figures.py` → Matplotlib figures
- `scripts/corpus_statistics.py` → Corpus stats table

---

## Implementation Order & Dependencies

### Sprint 1 (Week 1-2): Core Infrastructure
1. Create package structure + `config.py` + config schema
2. Create `data/example/` with example formats and README
3. Implement `corpus.py` (data loading, chunking, filtering)
4. Implement `embedding.py` (numpy-based, no CSV serialization)
5. Implement `reduction.py` (none/PCA/UMAP, default: none)
6. Implement `retrieval.py` (anchor-guided cosine)
7. Implement `anchors.py` (file loading + validation)
8. Create `notebooks/anchor_workshop.ipynb` (interactive anchor definition)
9. Migrate healthcare data files (keywords, subcategories, anchors, perspective groups)

### Sprint 2 (Week 3-4): Generation + Orchestration
10. Implement `generation.py` (configurable LLM endpoint, prompt templates, JSON output)
11. Implement `run_pipeline.py` (end-to-end orchestration)
12. Migrate food policy data files
13. End-to-end test on healthcare topic
14. **Discussion checkpoint: evaluation metrics** — discuss and implement `evaluation.py`
15. **Discussion checkpoint: baselines** — discuss and implement `baselines.py`

### Sprint 3 (Week 5-6): Experiments + Optimization
16. Implement `prompt_optimization.py`
17. Run main system + baselines on healthcare + food policy
18. Set up 3rd topic (data, config, reference concourse)
19. Run ablations (especially reduction method comparison)
20. Run prompt optimization
21. Generate paper figures and tables

### Sprint 4 (Week 7): Polish + Submit
22. README with full reproduction instructions
23. Clean up, final testing
24. Package for submission

---

## Key Technical Decisions

1. **Embeddings stored as `.npy`** files, not string-serialized in CSVs.

2. **LLM access via `openai` Python package** with configurable `base_url`. Default: OpenRouter. Users can point to OpenAI directly, Azure, local vLLM, etc. Model specified in config per pipeline step.

3. **Anchor generation is optional.** Users can: (a) skip anchors entirely, (b) define anchors manually via JSON, (c) use the interactive notebook, or (d) auto-generate with LLM. The user always defines the subcategory topics.

4. **Dimensionality reduction defaults to `none`.** UMAP is designed for visualization, not cosine retrieval. Whether reduction helps is an empirical question treated as an ablation. PCA is the more principled alternative if reduction is needed.

5. **Prompt templates** stored in `autoconcourse/prompts/`. Domain-specific content comes from config/data files, not from code. Prompts can be iteratively improved by LLM after initial pipeline run.

6. **No Jupyter notebooks in the pipeline itself.** Notebooks for interactive work only (anchor workshop, exploration). Pipeline is fully scriptable. Original notebooks preserved in `legacy/`.

7. **Perspective stratification is optional.** Config-driven. Some topics may not have perspective groups.

8. **Intermediate outputs cached.** Each pipeline step checks for existing outputs before re-running.

9. **Evaluation metrics and baselines deferred.** Will be discussed and agreed upon before implementation to avoid wasted work.

---

## Verification Plan

1. **Unit tests:** `tests/` directory with tests for each module
2. **Integration test:** Run full pipeline on a small synthetic corpus (10 paragraphs, 2 subcategories)
3. **Regression test:** Run on healthcare topic with thesis parameters, verify comparable outputs
4. **Example data test:** Verify pipeline runs end-to-end with `data/example/` files

---

## Files to Preserve (Legacy)

Move original notebooks to `legacy/` for reference:
- `01a_get_articles.ipynb` → `legacy/`
- `01b_content_anchor_generation.ipynb` → `legacy/`
- `02_chunk_into_paragraphs.ipynb` → `legacy/`
- `03_generate_embeddings.py` → `legacy/`
- `04_dimensionality_reduction.py` → `legacy/`
- `05_categorization.py` → `legacy/`
- `06_similarity_score_analysis.ipynb` → `legacy/`
- `07_generate_dri_statements.ipynb` → `legacy/`
- `07a_generate_dri_statements_22.7.25.ipynb` → `legacy/`
- `Validation/` → `legacy/validation/`
- `data/98_side_results/` → `legacy/side_results/`
